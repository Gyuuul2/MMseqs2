#include "Align2clust.h"
#include "kmermatcher.h"
#include "Util.h"
#include "Debug.h"
#include "DBWriter.h"
#include "FastSort.h"
#include "Alignment.h"
#include "FileUtil.h"
#include <cerrno>
#include <climits>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <thread>
#include <ctime>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>
#include <list>
#include <queue>
#include <deque>

#ifdef OPENMP
#include <omp.h>
#endif

static const size_t PREFETCH_DISTANCE = 16;

static size_t getSetCoverReadSize(size_t memoryLimit) {
    return std::min<size_t>(memoryLimit / 64, 4ULL * 1024 * 1024 * 1024);
}

template <typename T>
class SetCoverMergeReader {
public:
    SetCoverMergeReader() : writtenRecordCount(0) {}

    void add(size_t records) {
        Run run = { writtenRecordCount, writtenRecordCount + records, 0, std::vector<T>() };
        runs.push_back(run);
        writtenRecordCount += records;
    }
    bool empty() const { return runs.empty(); }

    template <typename Less, typename Prefetch, typename Take>
    void merge(const std::string &fileName, size_t recordElementCount, size_t readSize, Less less, Prefetch prefetch, Take take) {
        FILE *reader = FileUtil::openSharedFile(fileName, "rb");
        size_t recordsPerRead = std::max<size_t>(1, readSize / (recordElementCount * sizeof(T)) / runs.size());
        const auto headLess = [&](const Head &first, const Head &second) { return less(first.record, second.record); };
        std::priority_queue<Head, std::vector<Head>, decltype(headLess)> heads(headLess);
        for (size_t run = 0; run < runs.size(); run++) {
            if (fillRunBuffer(runs[run], fileno(reader), fileName, recordElementCount, recordsPerRead)) {
                heads.push(Head{ runs[run].buffer.data(), run });
            }
        }
        while (heads.empty() == false) {
            const Head head = heads.top();
            heads.pop();
            Run &run = runs[head.run];
            if ((run.bufferPosition + PREFETCH_DISTANCE) * recordElementCount < run.buffer.size()) {
                prefetch(run.buffer.data() + (run.bufferPosition + PREFETCH_DISTANCE) * recordElementCount);
            }
            take(head.record);
            run.bufferPosition++;
            if (fillRunBuffer(run, fileno(reader), fileName, recordElementCount, recordsPerRead)) {
                heads.push(Head{ run.buffer.data() + run.bufferPosition * recordElementCount, head.run });
            }
        }
        if (fclose(reader) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        FileUtil::removeSharedFile(fileName);
        std::vector<Run>().swap(runs);
    }

private:
    struct Run {
        size_t next;
        size_t end;
        size_t bufferPosition;
        std::vector<T> buffer;
    };
    struct Head {
        const T *record;
        size_t run;
    };

    static bool fillRunBuffer(Run &run, int fd, const std::string &fileName, size_t recordElementCount, size_t recordsPerRead) {
        if (run.bufferPosition * recordElementCount == run.buffer.size()) {
            if (run.next == run.end) {
                return false;
            }
            size_t records = std::min(recordsPerRead, run.end - run.next);
            run.buffer.resize(records * recordElementCount);
            FileUtil::preadAll(fd, fileName.c_str(), reinterpret_cast<char *>(run.buffer.data()), run.buffer.size() * sizeof(T), run.next * recordElementCount * sizeof(T));
            run.next += records;
            run.bufferPosition = 0;
        }
        return true;
    }

    size_t writtenRecordCount;
    std::vector<Run> runs;
};

class SetCoverPairFile {
public:
    SetCoverPairFile(std::vector<ClusterPair> &pairs, const std::string &fileName, size_t maxPairs, size_t memoryLimit, const uint64_t *assigned)
        : pairs(pairs), fileName(fileName), maxPairs(maxPairs), memoryLimit(memoryLimit), assigned(assigned), file(NULL) {}

    bool prepareBatch(size_t count) {
        if (count > maxPairs) {
            flush();
            return false;
        }
        if (count > maxPairs - pairs.size()) {
            flush();
        }
        return true;
    }

    void push(const ClusterPair &pair) {
        if (pairs.size() == maxPairs) {
            flush();
        }
        pairs.push_back(pair);
    }

    template <typename AddClusterPair>
    void finish(AddClusterPair addClusterPair) {
        if (runs.empty()) {
            SORT_PARALLEL(pairs.begin(), pairs.end(), compareDescending);
            for (size_t i = 0; i < pairs.size(); i++) {
                if (i + PREFETCH_DISTANCE < pairs.size()) {
                    prefetchPair(pairs[i + PREFETCH_DISTANCE]);
                }
                addClusterPair(pairs[i]);
            }
            return;
        }
        flush();
        if (fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        file = NULL;
        runs.merge(fileName, 1, getSetCoverReadSize(memoryLimit),
                   [](const ClusterPair *first, const ClusterPair *second) { return ClusterPair::compareByRep(*first, *second); },
                   [&](const ClusterPair *pair) { prefetchPair(*pair); },
                   [&](const ClusterPair *pair) { addClusterPair(*pair); });
    }

private:
    static bool compareDescending(const ClusterPair &first, const ClusterPair &second) {
        return ClusterPair::compareByRep(second, first);
    }
    bool isAssigned(DBKeyType key) const {
        return ((assigned[key / 64] >> (key % 64)) & 1) != 0;
    }
    void prefetchPair(const ClusterPair &pair) const {
        __builtin_prefetch(&assigned[pair.repKey / 64]);
        __builtin_prefetch(&assigned[pair.memberKey / 64]);
    }

    void flush() {
        if (pairs.empty()) {
            return;
        }
        pairs.erase(std::remove_if(pairs.begin(), pairs.end(), [&](const ClusterPair &pair) {
            return isAssigned(pair.repKey) || isAssigned(pair.memberKey);
        }), pairs.end());
        if (pairs.empty()) {
            return;
        }
        SORT_PARALLEL(pairs.begin(), pairs.end(), compareDescending);
        if (file == NULL) {
            file = FileUtil::openAndDelete(fileName.c_str(), "wb");
        }
        if (fwrite(pairs.data(), sizeof(ClusterPair), pairs.size(), file) != pairs.size()) {
            Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        runs.add(pairs.size());
        pairs.clear();
    }

    std::vector<ClusterPair> &pairs;
    const std::string fileName;
    const size_t maxPairs;
    const size_t memoryLimit;
    const uint64_t *assigned;
    FILE *file;
    SetCoverMergeReader<ClusterPair> runs;
};

class SetCoverSmallSetFiles {
public:
    static const size_t MAX_SIZE = 64;

    SetCoverSmallSetFiles(const std::string &prefix, size_t memoryLimit, const uint64_t *assigned)
        : prefix(prefix), memoryLimit(memoryLimit), assigned(assigned), setSizeFiles(MAX_SIZE + 1) {}

    void append(size_t count, const DBKeyType *members) {
        SetSizeFile &setSizeFile = setSizeFiles[count];
        size_t setSizeBufferSize = std::max(count * sizeof(DBKeyType), memoryLimit / 64 / (MAX_SIZE - 2));
        if (setSizeFile.memberKeys.size() / count >= std::max<size_t>(1, setSizeBufferSize / sizeof(DBKeyType) / count)) {
            flush(count);
        }
        setSizeFile.memberKeys.insert(setSizeFile.memberKeys.end(), members, members + count);
    }

    template <typename AssignSet>
    void assignSetsLargerThan(size_t maxSetSize, AssignSet assignSet) {
        for (size_t count = MAX_SIZE; count > 2 && count > maxSetSize; count--) {
            assignSetsOfSize(count, assignSet);
        }
    }

private:
    struct SetSizeFile {
        std::vector<DBKeyType> memberKeys;
        SetCoverMergeReader<DBKeyType> runs;
        FILE *writer;
        bool isMerged;
        SetSizeFile() : writer(NULL), isMerged(false) {}
    };

    std::string fileName(size_t count) const {
        return prefix + "." + SSTR(count);
    }
    void flush(size_t count) {
        SetSizeFile &setSizeFile = setSizeFiles[count];
        if (setSizeFile.memberKeys.empty()) {
            return;
        }
        size_t recordCount = setSizeFile.memberKeys.size() / count;
        std::vector<std::pair<DBKeyType, size_t> > order(recordCount);
        for (size_t i = 0; i < recordCount; i++) {
            order[i] = std::make_pair(setSizeFile.memberKeys[i * count], i);
        }
        SORT_PARALLEL(order.begin(), order.end(), std::greater<std::pair<DBKeyType, size_t> >());
        const std::string path = fileName(count);
        if (setSizeFile.writer == NULL) {
            setSizeFile.writer = FileUtil::openAndDelete(path.c_str(), "wb");
        }
        size_t writeWords = std::max<size_t>(count, (4ULL * 1024 * 1024 / sizeof(DBKeyType) / count) * count);
        std::vector<DBKeyType> output;
        output.reserve(writeWords);
        for (size_t i = 0; i < recordCount; i++) {
            const DBKeyType *record = setSizeFile.memberKeys.data() + order[i].second * count;
            output.insert(output.end(), record, record + count);
            if (output.size() >= writeWords || i + 1 == recordCount) {
                if (fwrite(output.data(), sizeof(DBKeyType), output.size(), setSizeFile.writer) != output.size()) {
                    Debug(Debug::ERROR) << "Cannot write " << path << "\n";
                    EXIT(EXIT_FAILURE);
                }
                output.clear();
            }
        }
        setSizeFile.runs.add(recordCount);
        setSizeFile.memberKeys.clear();
    }
    template <typename AssignSet>
    void assignSetsOfSize(size_t count, AssignSet assignSet) {
        SetSizeFile &setSizeFile = setSizeFiles[count];
        if (setSizeFile.isMerged) {
            return;
        }
        flush(count);
        setSizeFile.isMerged = true;
        const std::string path = fileName(count);
        if (setSizeFile.writer == NULL) {
            unlink(path.c_str());
            return;
        }
        if (fclose(setSizeFile.writer) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        setSizeFile.writer = NULL;
        setSizeFile.runs.merge(path, count, getSetCoverReadSize(memoryLimit),
                               [](const DBKeyType *first, const DBKeyType *second) { return first[0] < second[0]; },
                               [&](const DBKeyType *ahead) {
                                   for (size_t member = 0; member < count; member++) {
                                       __builtin_prefetch(&assigned[ahead[member] / 64]);
                                   }
                               },
                               [&](const DBKeyType *members) { assignSet(count, members); });
        std::vector<DBKeyType>().swap(setSizeFile.memberKeys);
    }

    const std::string prefix;
    const size_t memoryLimit;
    const uint64_t *assigned;
    std::vector<SetSizeFile> setSizeFiles;
};

class SetCoverLargeSetFiles {
public:
    static const size_t MAX_OPEN_RUNS = 256;

    SetCoverLargeSetFiles(const std::string &prefix, size_t memoryLimit)
        : prefix(prefix), memoryBudget(std::max(65 * sizeof(DBKeyType), getSetCoverReadSize(memoryLimit))),
          nextFile(0), writtenSize(0) {}

    template <typename Reps, typename IsFree, typename Word>
    void appendStep(const Reps &reps, const std::vector<size_t> &setReps, const std::vector<size_t> &setSizes, size_t paddedDbSize, IsFree isFree, Word word) {
        std::vector<size_t> order;
        for (size_t set = 0; set < setSizes.size(); set++) {
            if (setSizes[set] > SetCoverSmallSetFiles::MAX_SIZE) {
                order.push_back(set);
            }
        }
        if (order.empty()) {
            return;
        }
        std::sort(order.begin(), order.end(), [&](size_t first, size_t second) {
            return setSizes[first] == setSizes[second] ? reps[setReps[first]].repKey % paddedDbSize > reps[setReps[second]].repKey % paddedDbSize : setSizes[first] > setSizes[second];
        });
        FILE *file = NULL;
        size_t run = createSortedRun(file);
        for (size_t set : order) {
            writeWord(file, setSizes[set]);
            writeWord(file, reps[setReps[set]].repKey % paddedDbSize);
            for (size_t rep = setReps[set]; rep < setReps[set + 1]; rep++) {
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (isFree(i)) {
                        writeWord(file, word(i));
                    }
                }
            }
        }
        finishRun(run, file);
    }

    template <typename IsAssigned, typename AssignMember>
    void assignSetsLargerThan(size_t maxSetSize, IsAssigned isAssigned, AssignMember assignMember, SetCoverPairFile &pairFile,
                              SetCoverSmallSetFiles &smallSetFiles) {
        while (true) {
            bool fromPending = pending.empty() == false && (heads.empty() || isLargerSet(pending.front().memberCount, pending.front().representativeId,
                                                                                              heads.top().count, heads.top().repKey));
            size_t count = fromPending ? pending.front().memberCount : (heads.empty() ? 0 : heads.top().count);
            if (count <= maxSetSize) {
                break;
            }
            DBKeyType repKey;
            size_t start = 0;
            size_t run = 0;
            if (fromPending) {
                std::pop_heap(pending.begin(), pending.end(), SetCoverComparator());
                const SetCoverCandidate candidate = pending.back();
                pending.pop_back();
                repKey = candidate.representativeId;
                start = candidate.memberOffset;
            } else {
                const Head head = heads.top();
                heads.pop();
                run = head.run;
                repKey = head.repKey;
                start = runs[run].position + 1;
            }
            // unassigned members, while they fit the budget
            unassignedMembers.clear();
            size_t validCount = 0;
            if (isAssigned(repKey) == false) {
                for (size_t member = 0; member < count; member++) {
                    DBKeyType key = fromPending ? pendingMembers[start + member] : readWord(run, start + member);
                    if (isAssigned(key) == false) {
                        if (validCount < memoryBudget / sizeof(DBKeyType)) {
                            unassignedMembers.push_back(key);
                        }
                        validCount++;
                    }
                }
            }
            if (validCount > 1) {
                if (validCount == 2) {
                    pairFile.push(ClusterPair{repKey, unassignedMembers[1]});
                } else if (validCount <= SetCoverSmallSetFiles::MAX_SIZE) {
                    smallSetFiles.append(validCount, unassignedMembers.data());
                } else if (validCount < count) {
                    if (validCount > memoryBudget / sizeof(DBKeyType)) {
                        FILE *file = NULL;
                        size_t nextRun = createSortedRun(file);
                        writeWord(file, validCount);
                        for (size_t member = 0; member < count; member++) {
                            DBKeyType key = fromPending ? pendingMembers[start + member] : readWord(run, start + member);
                            if (isAssigned(key) == false) {
                                writeWord(file, key);
                            }
                        }
                        finishRun(nextRun, file);
                    } else {
                        if (pendingMembers.size() + validCount > memoryBudget / sizeof(DBKeyType)) {
                            flushPending();
                        }
                        size_t offset = pendingMembers.size();
                        pendingMembers.insert(pendingMembers.end(), unassignedMembers.begin(), unassignedMembers.end());
                        pending.push_back(SetCoverCandidate{repKey, validCount, offset});
                        std::push_heap(pending.begin(), pending.end(), SetCoverComparator());
                    }
                } else if (unassignedMembers.size() == count) {
                    for (size_t member = 0; member < count; member++) {
                        assignMember(repKey, unassignedMembers[member]);
                    }
                } else {
                    for (size_t member = 0; member < count; member++) {
                        assignMember(repKey, fromPending ? pendingMembers[start + member] : readWord(run, start + member));
                    }
                }
            }
            if (fromPending) {
                if (pending.empty()) {
                    std::vector<DBKeyType>().swap(pendingMembers);
                }
            } else {
                Run &source = runs[run];
                source.position += count + 1;
                if (source.position == source.words) {
                    if (source.reader != NULL) {
                        if (fclose(source.reader) != 0) {
                            Debug(Debug::ERROR) << "Cannot close " << source.path << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        openRuns.erase(source.openPosition);
                        source.reader = NULL;
                    }
                    FileUtil::remove(source.path.c_str());
                    source = Run();
                    freeRuns.push_back(run);
                } else {
                    pushHead(run);
                }
            }
        }
        std::vector<DBKeyType>().swap(unassignedMembers);
    }

private:
    struct Run {
        std::string path;
        size_t words = 0;
        size_t position = 0;
        size_t bufferStart = 0;
        std::vector<DBKeyType> buffer;
        FILE *reader = NULL;
        std::list<size_t>::iterator openPosition;
    };
    struct Head {
        size_t count;
        DBKeyType repKey;
        size_t run;
    };
    struct HeadLess {
        bool operator()(const Head &first, const Head &second) const {
            return first.count == second.count ? first.repKey < second.repKey : first.count < second.count;
        }
    };
    static bool isLargerSet(size_t firstCount, DBKeyType firstRep, size_t secondCount, DBKeyType secondRep) {
        return firstCount == secondCount ? firstRep > secondRep : firstCount > secondCount;
    }
    size_t createSortedRun(FILE *&file) {
        size_t index = freeRuns.empty() ? runs.size() : freeRuns.back();
        if (freeRuns.empty()) {
            runs.emplace_back();
        } else {
            freeRuns.pop_back();
        }
        runs[index].path = prefix + "." + SSTR(nextFile++);
        file = FileUtil::openAndDelete(runs[index].path.c_str(), "wb");
        // share the read buffers with the new run
        size_t bufferWords = getBufferWords();
        for (size_t run = 0; run < runs.size(); run++) {
            if (runs[run].buffer.size() > bufferWords) {
                std::vector<DBKeyType>(runs[run].buffer.begin(), runs[run].buffer.begin() + bufferWords).swap(runs[run].buffer);
            }
        }
        return index;
    }
    // at least 64 KiB per open run
    size_t getBufferWords() const {
        return std::max((size_t) 8192, memoryBudget / sizeof(DBKeyType) / (runs.size() - freeRuns.size()));
    }
    void writeWord(FILE *file, DBKeyType word) {
        writeBuffer.push_back(word);
        if (writeBuffer.size() == 4 * 1024 * 1024 / sizeof(DBKeyType)) {
            flushWrites(file);
        }
        writtenSize += sizeof(word);
    }
    void flushWrites(FILE *file) {
        if (fwrite(writeBuffer.data(), sizeof(DBKeyType), writeBuffer.size(), file) != writeBuffer.size()) {
            Debug(Debug::ERROR) << "Cannot write large set file " << prefix << "\n";
            EXIT(EXIT_FAILURE);
        }
        writeBuffer.clear();
    }
    void finishRun(size_t run, FILE *file) {
        flushWrites(file);
        if (fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << runs[run].path << "\n";
            EXIT(EXIT_FAILURE);
        }
        runs[run].words = writtenSize / sizeof(DBKeyType) - totalRunWords;
        totalRunWords = writtenSize / sizeof(DBKeyType);
        pushHead(run);
    }
    DBKeyType readWord(size_t run, size_t offset) {
        Run &source = runs[run];
        if (source.buffer.empty() || offset < source.bufferStart || offset >= source.bufferStart + source.buffer.size()) {
            source.bufferStart = offset;
            std::vector<DBKeyType>(std::min(getBufferWords(), source.words - offset)).swap(source.buffer);
            if (source.reader == NULL) {
                if (openRuns.size() == MAX_OPEN_RUNS) {
                    Run &oldest = runs[openRuns.front()];
                    if (fclose(oldest.reader) != 0) {
                        Debug(Debug::ERROR) << "Cannot close " << oldest.path << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    oldest.reader = NULL;
                    openRuns.pop_front();
                }
                source.reader = FileUtil::openSharedFile(source.path, "rb");
                openRuns.push_back(run);
                source.openPosition = --openRuns.end();
            } else {
                openRuns.splice(openRuns.end(), openRuns, source.openPosition);
            }
            FileUtil::preadAll(fileno(source.reader), source.path.c_str(), reinterpret_cast<char *>(source.buffer.data()),
                               source.buffer.size() * sizeof(DBKeyType), source.bufferStart * sizeof(DBKeyType));
        }
        return source.buffer[offset - source.bufferStart];
    }
    void pushHead(size_t run) {
        size_t count = readWord(run, runs[run].position);
        heads.push(Head{count, readWord(run, runs[run].position + 1), run});
    }
    void flushPending() {
        if (pending.empty()) {
            std::vector<DBKeyType>().swap(pendingMembers);
            return;
        }
        std::sort(pending.begin(), pending.end(), [](const SetCoverCandidate &first, const SetCoverCandidate &second) {
            return isLargerSet(first.memberCount, first.representativeId, second.memberCount, second.representativeId);
        });
        FILE *file = NULL;
        size_t run = createSortedRun(file);
        for (const SetCoverCandidate &candidate : pending) {
            writeWord(file, candidate.memberCount);
            for (size_t member = 0; member < candidate.memberCount; member++) {
                writeWord(file, pendingMembers[candidate.memberOffset + member]);
            }
        }
        finishRun(run, file);
        std::vector<SetCoverCandidate>().swap(pending);
        std::vector<DBKeyType>().swap(pendingMembers);
    }

    const std::string prefix;
    const size_t memoryBudget;
    size_t nextFile;
    size_t writtenSize;
    size_t totalRunWords = 0;
    std::vector<Run> runs;
    std::vector<size_t> freeRuns;
    std::list<size_t> openRuns;
    std::priority_queue<Head, std::vector<Head>, HeadLess> heads;
    std::vector<SetCoverCandidate> pending;
    std::vector<DBKeyType> pendingMembers;
    std::vector<DBKeyType> unassignedMembers;
    std::vector<DBKeyType> writeBuffer;
};

__extension__ typedef unsigned __int128 uint128;

// entries [firstEntry, endEntry) of node's target split files
struct PrefWindow {
    size_t firstKey;
    size_t node;
    size_t firstEntry;
    size_t endEntry;
    uint128 alignmentCells;
};

struct RepTargetsRecord {
    ClusterPair repOrderKeyFirstTarget;
    unsigned int targetCount;
    size_t targetOffset;
    size_t readSize;

    static bool compareByRep(const RepTargetsRecord &first, const RepTargetsRecord &second) {
        return ClusterPair::compareByRep(first.repOrderKeyFirstTarget, second.repOrderKeyFirstTarget);
    }
};

static std::vector<size_t> targetSplitBlockOffsets;
static std::vector<unsigned int> targetSplitAlignNodes;
// kmermatcher key split files
static std::vector<size_t> keySplitFirstKeys;
static std::vector<size_t> classFileFirstEntries;

// by target split
static size_t getBlockAlignNode(size_t block) {
    return targetSplitAlignNodes[std::upper_bound(targetSplitBlockOffsets.begin(), targetSplitBlockOffsets.end(), block) - targetSplitBlockOffsets.begin() - 1];
}

// search again only once key leaves its block
static size_t getKeyAlignNode(const DBReader<DBKeyType> &seqDbr, size_t key, size_t *block = NULL) {
    if (block == NULL) {
        return getBlockAlignNode(seqDbr.getKeyBlock(key));
    }
    const std::vector<size_t> &blockKeyOffsets = seqDbr.getBlockKeyOffsets();
    if (*block == SIZE_MAX || key < blockKeyOffsets[*block] || key >= blockKeyOffsets[*block + 1]) {
        *block = seqDbr.getKeyBlock(key);
    }
    return getBlockAlignNode(*block);
}

// retry a missing or short file, then fail
static void failIfMissingTooLong(time_t &missingSince, unsigned int &attempts, const std::string &error) {
    if (FileUtil::hasFailedTooLong(missingSince, attempts)) {
        Debug(Debug::ERROR) << error;
        EXIT(EXIT_FAILURE);
    }
}

static std::string describeExchangeFile(size_t size, size_t observedSize, uint64_t fileNumber, uint64_t observedFileNumber, int error) {
    return " (expected " + SSTR(size) + " bytes, observed " + (observedSize == SIZE_MAX ? std::string("unavailable") : SSTR(observedSize))
           + " bytes; expected file number " + SSTR(fileNumber) + ", observed " + (observedFileNumber == UINT64_MAX ? std::string("unavailable") : SSTR(observedFileNumber))
           + "; error " + SSTR(error) + ")\n";
}

static std::string formatSeconds(double seconds) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.1f s", seconds);
    return buffer;
}

static std::string formatRatio(size_t size, size_t total) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.3f", static_cast<double>(size) / std::max<size_t>(1, total));
    return buffer;
}

// prefilter buffer records [begin, end) and their entries
struct FetchedSequences {
    FetchedSequences() : prefilterBuffer(0), prefilterBufferNumber(0), begin(0), end(0), targetCount(0), pendingPairCount(0), firstPendingRepKey(0), plannedSize(0),
                         isLast(false), entries(NULL), allocatedSize(0), acceptedSize(0), ownKeyCopyEnd(0), isReady(false) {}
    size_t prefilterBuffer;
    size_t prefilterBufferNumber;
    size_t begin;
    size_t end;
    size_t targetCount;
    // pending pairs whose members it brings
    size_t pendingPairCount;
    // first rep waiting for member checks at selection
    size_t firstPendingRepKey;
    // bytes of the fetch size rule
    size_t plannedSize;
    bool isLast;
    std::vector<DBKeyType> keys;
    std::vector<size_t> fetchSourceKeyOffsets;
    char *entries;
    size_t allocatedSize;
    std::vector<size_t> entryOffsets;
    // entry of each record's rep and targets, SIZE_MAX if not fetched
    std::vector<size_t> recordEntryOffsets;
    std::vector<size_t> recordEntries;
    // accepted bytes at hand-off, for the master gate
    size_t acceptedSize;
    // own keys before it are copied
    size_t ownKeyCopyEnd;
    bool isReady;

    size_t memorySize() const {
        return allocatedSize + keys.capacity() * sizeof(DBKeyType) + (entryOffsets.capacity() + recordEntryOffsets.capacity() + recordEntries.capacity()) * sizeof(size_t);
    }
};

// per thread cursor of each fetch source and last block
struct FetchedKeyCursor {
    FetchedKeyCursor(size_t fetchSources) : positions(fetchSources, 0), block(SIZE_MAX) {}
    std::vector<size_t> positions;
    size_t block;
};

// gallop from position for ascending keys
static const char *findFetchedKey(const FetchedSequences &sequences, size_t fetchSource, DBKeyType key, size_t *seqLen = NULL, size_t *position = NULL) {
    const DBKeyType *keys = sequences.keys.data();
    size_t sourceEnd = sequences.fetchSourceKeyOffsets[fetchSource + 1];
    size_t first = sequences.fetchSourceKeyOffsets[fetchSource];
    size_t last = sourceEnd;
    if (position != NULL && *position >= first && *position <= last && (*position == first || keys[*position - 1] < key)) {
        size_t step = 1;
        while (*position + step < last && keys[*position + step] < key) {
            step *= 2;
        }
        first = *position + step / 2;
        last = std::min(*position + step + 1, last);
    }
    const DBKeyType *found = std::lower_bound(keys + first, keys + last, key);
    if (position != NULL) {
        *position = found - keys;
    }
    if (found == keys + sourceEnd || *found != key) {
        return NULL;
    }
    size_t entry = found - keys;
    if (seqLen != NULL) {
        *seqLen = sequences.entryOffsets[entry + 1] - sequences.entryOffsets[entry] - 2;
    }
    return sequences.entries + sequences.entryOffsets[entry];
}

// assigned keys are served as zeros
static const char *getRecordSequence(const FetchedSequences &sequences, size_t entry, size_t *seqLen) {
    if (entry == SIZE_MAX || sequences.entries[sequences.entryOffsets[entry]] == '\0') {
        return NULL;
    }
    *seqLen = sequences.entryOffsets[entry + 1] - sequences.entryOffsets[entry] - 2;
    return sequences.entries + sequences.entryOffsets[entry];
}

static const double PUBLISH_SECONDS = 1;
// period of the assigned key reads
static const double READ_SECONDS = 1;
// min aligning seconds of the fetches in flight
static const double MIN_FETCH_AHEAD_SECONDS = 1;
// read the assigned keys a few times a second
static const double BATCH_READ_SECONDS = PUBLISH_SECONDS / 4.0;
// at most 1 / LOG_COST_FACTOR of a thread's time on the logs
static const double LOG_COST_FACTOR = 16;
static const double MIN_PUBLISH_SECONDS = 0.1;
static const double MIN_READ_SECONDS = 0.05;
// align node prints its progress once a minute
static const double STATUS_SECONDS = 60;
static const uint64_t ACCEPTED_REP_BIT = 1ULL << 63;

static size_t getRepOrderKey(uint64_t word) {
    return word & ~ACCEPTED_REP_BIT;
}
static size_t getAssignedKeysHeaderWords() {
    return MMseqsNode::count + 5;
}

static const size_t PREFILTER_BUFFER_MEMORY_DIVISOR = 32;
// 10 bytes per pref byte, 48 bytes and a record per piece
static size_t getMaxPrefReadSize(size_t prefilterBufferSize) {
    return std::max((size_t) 1, prefilterBufferSize / (10 + (48 + sizeof(RepTargetsRecord)) / 4));
}

static const size_t MEMBER_SPLIT_MEMORY_DIVISOR = 16;
static const size_t CLUSTERED_LOG_BUFFER_SIZE = 16 * 1024 * 1024;
static const size_t MEMBER_READ_SIZE = 1024 * 1024;
static const size_t FETCHED_KEY_SIZE = 4 * sizeof(DBKeyType) + sizeof(unsigned int) + sizeof(size_t);
// same on every node whatever its page size
static const size_t ASSIGNED_PAGE_KEYS = 32768;
static const char PACKED_ENTRY_ALPHABET[] = "\0\nABCDEFGHIJKLMNOPQRSTUVWXYZ*-";
static const unsigned char PACKED_RAW_CODE = 31;

struct PackedEntryCodes {
    PackedEntryCodes() {
        memset(codes, PACKED_RAW_CODE, sizeof(codes));
        for (size_t i = 0; i < sizeof(PACKED_ENTRY_ALPHABET) - 1; i++) {
            codes[static_cast<unsigned char>(PACKED_ENTRY_ALPHABET[i])] = static_cast<unsigned char>(i);
        }
    }
    unsigned char codes[256];
};
static const PackedEntryCodes packedEntryCodes;

static size_t getPackedEntrySize(size_t entryLen) {
    return (5 * entryLen + 7) / 8;
}

// 8 entry bytes pack into 5
static bool packEntry(const char *entry, size_t entryLen, char *packedEntry) {
    for (size_t i = 0; i < entryLen; i += 8) {
        size_t groupLen = std::min<size_t>(8, entryLen - i);
        uint64_t bits = 0;
        for (size_t j = 0; j < groupLen; j++) {
            uint64_t code = packedEntryCodes.codes[static_cast<unsigned char>(entry[i + j])];
            if (code == PACKED_RAW_CODE) {
                return false;
            }
            bits |= code << (5 * j);
        }
        for (size_t byte = 0; byte < getPackedEntrySize(groupLen); byte++) {
            packedEntry[i / 8 * 5 + byte] = static_cast<char>(bits >> (8 * byte));
        }
    }
    return true;
}

static void unpackEntry(const char *packedEntry, size_t entryLen, char *entry) {
    for (size_t i = 0; i < entryLen; i += 8) {
        size_t groupLen = std::min<size_t>(8, entryLen - i);
        uint64_t bits = 0;
        for (size_t byte = 0; byte < getPackedEntrySize(groupLen); byte++) {
            bits |= static_cast<uint64_t>(static_cast<unsigned char>(packedEntry[i / 8 * 5 + byte])) << (8 * byte);
        }
        for (size_t j = 0; j < groupLen; j++) {
            entry[i + j] = PACKED_ENTRY_ALPHABET[(bits >> (5 * j)) & 31];
        }
    }
}

// chunks of ascending keys, run by runChunks in any order
static void unpackEntries(const char *servedEntries, size_t keyCount, const size_t *entryOffsets, char *data, size_t chunks, const std::function<void(size_t, const std::function<void(size_t)> &)> &runChunks) {
    const char *packedEntries = servedEntries + (keyCount + 7) / 8;
    std::vector<size_t> packedEntryOffsets(keyCount + 1, 0);
    for (size_t i = 0; i < keyCount; i++) {
        bool isAssigned = (static_cast<unsigned char>(servedEntries[i / 8]) >> (i % 8)) & 1;
        packedEntryOffsets[i + 1] = packedEntryOffsets[i] + (isAssigned ? 0 : getPackedEntrySize(entryOffsets[i + 1] - entryOffsets[i]));
    }
    std::vector<std::vector<size_t> > chunkRawEntries(chunks);
    runChunks(chunks, [&](size_t chunk) {
        for (size_t i = keyCount * chunk / chunks; i < keyCount * (chunk + 1) / chunks; i++) {
            if (packedEntryOffsets[i + 1] == packedEntryOffsets[i]) {
                memset(data + entryOffsets[i], 0, entryOffsets[i + 1] - entryOffsets[i]);
            } else if ((static_cast<unsigned char>(packedEntries[packedEntryOffsets[i]]) & 31) == PACKED_RAW_CODE) {
                chunkRawEntries[chunk].push_back(i);
            } else {
                unpackEntry(packedEntries + packedEntryOffsets[i], entryOffsets[i + 1] - entryOffsets[i], data + entryOffsets[i]);
            }
        }
    });
    const char *rawEntry = packedEntries + packedEntryOffsets[keyCount];
    for (size_t chunk = 0; chunk < chunks; chunk++) {
        for (size_t j = 0; j < chunkRawEntries[chunk].size(); j++) {
            size_t i = chunkRawEntries[chunk][j];
            memcpy(data + entryOffsets[i], rawEntry, entryOffsets[i + 1] - entryOffsets[i]);
            rawEntry += entryOffsets[i + 1] - entryOffsets[i];
        }
    }
}

static const size_t SERVICE_POLL_NODES = 8;
// progress or index reads per node and second
static const size_t PROGRESS_READS_PER_SECOND = 100;

// prefix sums in parallel
static void sumOffsets(size_t *offsets, size_t count, int threads) {
    std::vector<size_t> threadOffsets(threads + 1, 0);
#pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threads; thread++) {
        size_t end = count * (thread + 1) / threads + 1;
        for (size_t i = count * thread / threads + 2; i < end; i++) {
            offsets[i] += offsets[i - 1];
        }
        threadOffsets[thread + 1] = (count * thread / threads + 1 < end) ? offsets[end - 1] : 0;
    }
    for (int thread = 0; thread < threads; thread++) {
        threadOffsets[thread + 1] += threadOffsets[thread];
    }
#pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threads; thread++) {
        for (size_t i = count * thread / threads + 1; i < count * (thread + 1) / threads + 1; i++) {
            offsets[i] += threadOffsets[thread];
        }
    }
}

// one chunk with isOneStep, else all with the caller's team
template <typename Body>
static size_t forEachEntryChunk(size_t begin, size_t end, size_t chunkSize, bool isOneStep, Body body) {
    if (isOneStep) {
        body(begin, std::min(end, begin + chunkSize));
        return std::min(end, begin + chunkSize);
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t first = begin; first < end; first += chunkSize) {
        body(first, std::min(end, first + chunkSize));
    }
    return end;
}

// parts of a pair follow each other
static size_t getPairPartEnd(const std::vector<unsigned char> &isLastPart, size_t pairCount, size_t part) {
    size_t partEnd = part + 1;
    while (partEnd < pairCount && isLastPart[partEnd - 1] == 0) {
        partEnd++;
    }
    return partEnd;
}

static size_t getStepWords(size_t masterMemoryLimit) {
    return std::max<size_t>(1, masterMemoryLimit / 16 / (sizeof(uint64_t) + 4 * sizeof(std::pair<DBKeyType, size_t>)));
}

struct ClusterMemberLookup {
    struct Entry {
        size_t offset;
        size_t end;
        DBKeyType key;
    };
    struct EntryReader {
        Entry entry;
        size_t bufferBegin;
        size_t bufferEnd;
        // buffer or part data
        const char *data;
        std::vector<char> buffer;
        // index pairs from indexFirst
        size_t indexFirst;
        std::vector<uint64_t> indexPairs;

        EntryReader() : indexFirst(0) {}

        void start(const Entry &first) {
            entry = first;
            bufferBegin = 0;
            bufferEnd = 0;
        }
    };
    // entries of [keyBegin, keyEnd) in node's cluster part
    struct PreloadedRange {
        size_t keyBegin;
        size_t keyEnd;
        size_t node;
        size_t firstEntry;
        size_t endEntry;
        // index pairs of the entries and the pair behind
        std::vector<uint64_t> indexPairs;
        // empty if the entries are read from the part
        std::vector<char> data;

        static bool compareKeyEnd(size_t key, const PreloadedRange &range) {
            return key < range.keyEnd;
        }
    };

    // preload other parts' ranges that fit, the own part gets the rest
    ClusterMemberLookup(const std::string &clu, const std::string &localClu, const std::vector<size_t> &cluKeyOffsets, size_t entriesPerSampledKey, size_t partMemorySize,
                        const std::vector<std::pair<size_t, size_t> > &preloadKeyRanges)
        : preloadSize(0), targetSplitSize(0), preloadSeconds(0), clu(clu), entriesPerSampledKey(entriesPerSampledKey), cluKeyOffsets(cluKeyOffsets), rangeSize(0) {
        cluDbr = new DBReader<DBKeyType>((localClu + "." + SSTR(MMseqsNode::rank)).c_str(), (clu + ".index." + SSTR(MMseqsNode::rank)).c_str(), 1,
                                         DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDbr->open(DBReader<DBKeyType>::NOSORT);
        for (size_t i = 0; i < preloadKeyRanges.size(); i++) {
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                PreloadedRange range;
                range.keyBegin = std::max(preloadKeyRanges[i].first, cluKeyOffsets[node]);
                range.keyEnd = std::min(preloadKeyRanges[i].second, cluKeyOffsets[node + 1]);
                if (node == MMseqsNode::rank || range.keyBegin >= range.keyEnd) {
                    continue;
                }
                DBReader<DBKeyType> cluDb((clu + "." + SSTR(node)).c_str(), (clu + ".index." + SSTR(node)).c_str(), 1,
                                          DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
                cluDb.open(DBReader<DBKeyType>::NOSORT);
                range.node = node;
                range.firstEntry = cluDb.findEntryByKey(range.keyBegin);
                range.endEntry = cluDb.findEntryByKey(range.keyEnd);
                uint64_t firstPair[2];
                uint64_t endPair[2];
                cluDb.readIndexPairs(range.firstEntry, range.firstEntry + 1, firstPair);
                cluDb.readIndexPairs(range.endEntry, range.endEntry + 1, endPair);
                cluDb.close();
                size_t bytes = (range.endEntry - range.firstEntry + 1) * 2 * sizeof(uint64_t) + endPair[1] - firstPair[1];
                targetSplitSize += bytes;
                if (range.endEntry > range.firstEntry && rangeSize + bytes <= partMemorySize) {
                    rangeSize += bytes;
                    preloadedRanges.push_back(range);
                }
            }
        }
        preloadSize = rangeSize;
        size_t partIndexSize = (cluDbr->getSize() + 1) * 2 * sizeof(uint64_t);
        if (rangeSize + partIndexSize <= partMemorySize) {
            PreloadedRange &range = *preloadedRanges.insert(std::upper_bound(preloadedRanges.begin(), preloadedRanges.end(), cluKeyOffsets[MMseqsNode::rank], PreloadedRange::compareKeyEnd),
                                                            PreloadedRange());
            range.keyBegin = cluKeyOffsets[MMseqsNode::rank];
            range.keyEnd = cluKeyOffsets[MMseqsNode::rank + 1];
            range.node = MMseqsNode::rank;
            range.firstEntry = 0;
            range.endEntry = cluDbr->getSize();
            range.indexPairs.resize(partIndexSize / sizeof(uint64_t));
            cluDbr->readIndexPairs(0, cluDbr->getSize() + 1, range.indexPairs.data());
            if (localClu == clu && rangeSize + partIndexSize + cluDbr->getDataSize() <= partMemorySize) {
                range.data.resize(range.indexPairs.back() - range.indexPairs[1]);
                FileUtil::preadAll(cluDbr->getBufferedFd(), cluDbr->getDataFileName(), range.data.data(), range.data.size(), range.indexPairs[1]);
            }
            rangeSize += partIndexSize + range.data.size();
        } else {
            sampledKeys.resize((cluDbr->getSize() + entriesPerSampledKey - 1) / entriesPerSampledKey);
            cluDbr->readSampledKeys(entriesPerSampledKey, std::max(BatchReader::MAX_READ_REQUEST_SIZE, partMemorySize), sampledKeys.data());
        }
    }

    // read range i from the shared folder unless it is own
    void preloadRange(size_t i, int threads) {
        PreloadedRange &range = preloadedRanges[i];
        if (range.node == MMseqsNode::rank) {
            return;
        }
        Timer preloadTimer;
        DBReader<DBKeyType> cluDb((clu + "." + SSTR(range.node)).c_str(), (clu + ".index." + SSTR(range.node)).c_str(), threads,
                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDb.open(DBReader<DBKeyType>::NOSORT);
        range.indexPairs.resize(2 * (range.endEntry - range.firstEntry + 1));
        cluDb.readIndexPairs(range.firstEntry, range.endEntry + 1, range.indexPairs.data());
        range.data.resize(range.indexPairs.back() - range.indexPairs[1]);
        cluDb.readData(range.indexPairs[1], range.data.size(), range.data.data());
        cluDb.close();
        preloadSeconds += preloadTimer.getTimediff();
    }

    size_t getPreloadedRangeCount() const {
        return preloadedRanges.size();
    }

    const PreloadedRange *findPreloaded(DBKeyType key) const {
        std::vector<PreloadedRange>::const_iterator range = std::upper_bound(preloadedRanges.begin(), preloadedRanges.end(), key, PreloadedRange::compareKeyEnd);
        return (range != preloadedRanges.end() && range->keyBegin <= key) ? &*range : NULL;
    }

    ~ClusterMemberLookup() {
        cluDbr->close();
        delete cluDbr;
    }

    // node of key's cluster part, grows with key
    size_t getNode(DBKeyType key) const {
        return std::upper_bound(cluKeyOffsets.begin(), cluKeyOffsets.end(), key) - cluKeyOffsets.begin() - 1;
    }

    // search the range in memory or between two sampled keys
    Entry find(DBKeyType key, EntryReader &entryReader) const {
        Entry entry = { 0, 0, key };
        const PreloadedRange *range = findPreloaded(key);
        const uint64_t *indexPairs;
        size_t count;
        if (range != NULL) {
            indexPairs = range->indexPairs.data();
            count = range->endEntry - range->firstEntry;
        } else {
            size_t sampledKey = std::lower_bound(sampledKeys.begin(), sampledKeys.end(), key) - sampledKeys.begin();
            // entry is between the sampled keys, ends at the next start
            size_t entryCount = cluDbr->getSize();
            size_t first = (sampledKey > 0) ? (sampledKey - 1) * entriesPerSampledKey : 0;
            size_t last = std::min(sampledKey * entriesPerSampledKey + 1, entryCount);
            size_t pairEnd = std::min(last + 2, entryCount + 1);
            if (first < entryReader.indexFirst || pairEnd > entryReader.indexFirst + entryReader.indexPairs.size() / 2) {
                entryReader.indexFirst = first;
                entryReader.indexPairs.resize(2 * (pairEnd - first));
                cluDbr->readIndexPairs(first, pairEnd, entryReader.indexPairs.data());
            }
            indexPairs = entryReader.indexPairs.data() + 2 * (first - entryReader.indexFirst);
            count = last - first;
        }
        size_t found = 0;
        while (count > 0) {
            size_t half = count / 2;
            if (indexPairs[2 * (found + half)] < key) {
                found += half + 1;
                count -= half + 1;
            } else {
                count = half;
            }
        }
        // pair behind the searched ones has another key
        if (indexPairs[2 * found] == key) {
            entry.offset = indexPairs[2 * found + 1];
            entry.end = indexPairs[2 * found + 3] - 1;
        }
        return entry;
    }

    bool next(EntryReader &entryReader, DBKeyType &memberKey) const {
        Entry &entry = entryReader.entry;
        while (entry.offset < entry.end) {
            const char *lineEnd = (entry.offset < entryReader.bufferEnd)
                                  ? static_cast<const char *>(memchr(entryReader.data + (entry.offset - entryReader.bufferBegin), '\n', entryReader.bufferEnd - entry.offset)) : NULL;
            const PreloadedRange *range = (lineEnd == NULL) ? findPreloaded(entry.key) : NULL;
            if (range != NULL && range->data.empty() == false) {
                entryReader.data = range->data.data();
                entryReader.bufferBegin = range->indexPairs[1];
                entryReader.bufferEnd = range->indexPairs[1] + range->data.size();
                continue;
            }
            if (lineEnd == NULL) {
                size_t bytes = std::min(entry.end - entry.offset, MEMBER_READ_SIZE);
                entryReader.buffer.resize(bytes);
                FileUtil::preadAll(cluDbr->getBufferedFd(), cluDbr->getDataFileName(), entryReader.buffer.data(), bytes, entry.offset);
                entryReader.data = entryReader.buffer.data();
                entryReader.bufferBegin = entry.offset;
                entryReader.bufferEnd = entry.offset + bytes;
                continue;
            }
            const char *line = entryReader.data + (entry.offset - entryReader.bufferBegin);
            DBKeyType key = Util::fast_atoi<DBKeyType>(line);
            entry.offset += lineEnd - line + 1;
            if (key != entry.key) {
                memberKey = key;
                return true;
            }
        }
        return false;
    }

    size_t memorySize() const {
        return sizeof(*this) + cluKeyOffsets.capacity() * sizeof(size_t) + sizeof(DBReader<DBKeyType>) + sampledKeys.capacity() * sizeof(DBKeyType) + rangeSize;
    }

    // find reads at most entriesPerSampledKey + 3 index pairs
    size_t getEntryReaderSize() const {
        return MEMBER_READ_SIZE + (entriesPerSampledKey + 3) * 2 * sizeof(uint64_t);
    }

    // preloaded and total target split bytes, preload seconds
    size_t preloadSize;
    size_t targetSplitSize;
    double preloadSeconds;

private:
    const std::string clu;
    const size_t entriesPerSampledKey;
    DBReader<DBKeyType> *cluDbr;
    std::vector<size_t> cluKeyOffsets;
    // empty if the own part is in memory
    std::vector<DBKeyType> sampledKeys;
    // in key order, own part too if it fits
    std::vector<PreloadedRange> preloadedRanges;
    size_t rangeSize;
};

// --filter-cludb-file on several nodes: serve own cluster part
struct ClusterMemberExchange {
    ClusterMemberExchange(const std::string &requests, const std::string &served, ClusterMemberLookup &lookup, int threads, size_t firstAlignNode)
        : servedSize(0), sizeWaitSeconds(0), memberWaitSeconds(0), preloadWaitSeconds(0),
          requests(requests), served(served), lookup(lookup), threads(threads), requestCount(0), memberFileCount(0), isSizeRequest(false),
          nextMemberLines(MMseqsNode::count, 0), memberIndexes(MMseqsNode::count), isAnswered(MMseqsNode::count, 1), nodeCodes(MMseqsNode::count),
          nodeConsumedBytes(MMseqsNode::count, 0), memberLines(std::max<size_t>(16, 2 * MMseqsNode::count) * MEMBER_LINE_WORDS), isReceiving(false),
          firstAlignNode(firstAlignNode), requestLines(16 * (MMseqsNode::count + 6)), nextRequestLines(MMseqsNode::count, 0), requestIndexes(MMseqsNode::count), isRequesting(MMseqsNode::count, 0), requesterCount(0),
          nextRead(0), isPreloadDone(lookup.preloadSize == 0), nextPreloadedRange(0) {
        for (size_t node = firstAlignNode; node < MMseqsNode::count; node++) {
            isRequesting[node] = node != MMseqsNode::rank;
            requesterCount += isRequesting[node];
        }
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            memberIndexes[node].file = getMemberIndex(node);
            requestIndexes[node].file = getRequestIndex(node);
        }
        // poll idle requesters every idleSeconds, asking ones from 10 ms on
        idleSeconds = std::max(0.04, (double) requesterCount / PROGRESS_READS_PER_SECOND);
        readSeconds.assign(MMseqsNode::count, idleSeconds);
        nextReads.assign(MMseqsNode::count, 0);
        fastReadEnds.assign(MMseqsNode::count, 0);
    }

    // offsets and bytes (maxMembers 0), members (SIZE_MAX) or maxMembers of one entry
    void request(const std::vector<DBKeyType> &keys, const size_t *entryOffsets, const size_t *entryBytes, size_t consumedBytes, size_t maxMembers) {
        unsigned int nodes = MMseqsNode::count;
        unsigned int rank = MMseqsNode::rank;
        isSizeRequest = maxMembers == 0;
        // keys of a node are contiguous
        std::vector<size_t> nodeKeyOffsets(nodes + 1);
        for (unsigned int node = 0; node <= nodes; node++) {
            nodeKeyOffsets[node] = std::partition_point(keys.begin(), keys.end(), [&](DBKeyType key) { return lookup.getNode(key) < node; }) - keys.begin();
        }
        keyNodes.resize(keys.size());
        size_t localBegin = 0;
        size_t localEnd = 0;
        bool hasRequest = false;
        for (unsigned int node = 0; node < nodes; node++) {
            isAnswered[node] = 1;
            for (size_t i = nodeKeyOffsets[node]; i < nodeKeyOffsets[node + 1]; i++) {
                keyNodes[i] = (node == rank || lookup.findPreloaded(keys[i]) != NULL) ? rank : node;
                if (keyNodes[i] != rank) {
                    isAnswered[node] = 0;
                    continue;
                }
                localBegin = (localEnd == 0) ? i : localBegin;
                localEnd = i + 1;
            }
            hasRequest |= isAnswered[node] == 0;
            nodeCodes[node].clear();
        }
        if (hasRequest) {
            // request line: request, isLast, consumed bytes, max members, key offsets per node, end, magic
            std::vector<uint64_t> line(nodes + 6);
            requestCodes.clear();
            char code[10];
            for (unsigned int node = 0; node < nodes; node++) {
                line[4 + node] = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + requestCodes.size();
                DBKeyType previousKey = 0;
                size_t previousOffset = 0;
                for (size_t i = nodeKeyOffsets[node]; i < nodeKeyOffsets[node + 1]; i++) {
                    if (keyNodes[i] == rank) {
                        continue;
                    }
                    requestCodes.append(code, Util::putPrefKey(code, keys[i] - previousKey));
                    previousKey = keys[i];
                    // read the members at the answered offsets
                    if (maxMembers != 0) {
                        requestCodes.append(code, Util::putPrefKey(code, entryOffsets[i] - previousOffset));
                        requestCodes.append(code, Util::putPrefKey(code, entryBytes[i]));
                        previousOffset = entryOffsets[i];
                    }
                }
            }
            line[0] = requestCount;
            line[1] = 0;
            line[2] = consumedBytes;
            line[3] = maxMembers;
            line[4 + nodes] = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + requestCodes.size();
            MMseqsNode::writeExchangeFile(getRequestFile(rank), requestCount, requestCodes.data(), requestCodes.size(), getRequestIndex(rank), line, 1);
            requestCount++;
            std::string().swap(requestCodes);
        }
        waitForPreload();
        answer(keys.data() + localBegin, localEnd - localBegin, (maxMembers != 0) ? entryOffsets + localBegin : NULL, (maxMembers != 0) ? entryBytes + localBegin : NULL,
               consumedBytes, maxMembers, nodeCodes[rank], nodeConsumedBytes[rank], keyNodes.data() + localBegin);
    }

    // read answers to the last request, true once all answered
    bool tryReceive() {
        unsigned int nodes = MMseqsNode::count;
        unsigned int rank = MMseqsNode::rank;
        if (isReceiving == false) {
            isReceiving = true;
            receiveTimer.reset();
            receiveWarning.reset();
        }
        // owing nodes answer together, poll them in turn
        for (unsigned int node = 0; node < nodes;) {
            if (isAnswered[node]) {
                node++;
                continue;
            }
            size_t lineCount = MMseqsNode::readExchangeIndexLines(memberIndexes[node], nextMemberLines[node], MEMBER_LINE_WORDS, memberLines);
            nextMemberLines[node] += lineCount;
            for (size_t l = 0; l < lineCount; l++) {
                // member line: member file, requester, request, size, consumed bytes, magic
                const uint64_t *line = memberLines.data() + l * MEMBER_LINE_WORDS;
                if (line[1] != rank || line[2] != requestCount - 1) {
                    continue;
                }
                const std::string memberFile = getMemberFile(node, line[0]);
                int fd = MMseqsNode::waitForExchangeFile(memberFile, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + line[3], line[0]);
                nodeCodes[node].resize(line[3]);
                FileUtil::preadAll(fd, memberFile.c_str(), &nodeCodes[node][0], nodeCodes[node].size(), MMseqsNode::EXCHANGE_FILE_HEADER_BYTES);
                // ignore the close error of a foreign file
                close(fd);
                // only this node reads it
                FileUtil::removeSharedFile(memberFile);
                nodeConsumedBytes[node] = line[4];
                isAnswered[node] = 1;
            }
            // more lines may follow a full read
            if (isAnswered[node] == 0 && lineCount * MEMBER_LINE_WORDS < memberLines.size()) {
                receiveWarning.check(served);
                return false;
            }
        }
        isReceiving = false;
        (isSizeRequest ? sizeWaitSeconds : memberWaitSeconds) += receiveTimer.getTimediff();
        return true;
    }

    // entry offsets and bytes of the last request, released once read
    void getEntries(std::vector<size_t> &entryOffsets, std::vector<size_t> &entryBytes) {
        entryOffsets.resize(keyNodes.size());
        entryBytes.resize(keyNodes.size());
        std::vector<const char *> data(MMseqsNode::count);
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            data[node] = nodeCodes[node].data();
        }
        for (size_t i = 0; i < keyNodes.size(); i++) {
            entryOffsets[i] = Util::getPrefKey(data[keyNodes[i]]);
            entryBytes[i] = Util::getPrefKey(data[keyNodes[i]]);
        }
        nodeCodes.assign(MMseqsNode::count, std::string());
    }

    // members of key i of the last request, released once read
    void getMembers(std::vector<size_t> &memberOffsets, std::vector<DBKeyType> &members) {
        memberOffsets.assign(keyNodes.size() + 1, 0);
        members.clear();
        std::vector<const char *> data(MMseqsNode::count);
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            data[node] = nodeCodes[node].data();
        }
        for (size_t i = 0; i < keyNodes.size(); i++) {
            for (uint64_t code = Util::getPrefKey(data[keyNodes[i]]); code != 0; code = Util::getPrefKey(data[keyNodes[i]])) {
                members.push_back(static_cast<DBKeyType>(code - 1));
            }
            memberOffsets[i + 1] = members.size();
        }
        nodeCodes.assign(MMseqsNode::count, std::string());
    }

    // consumed bytes of the single key of the last request
    size_t getConsumedBytes() const {
        return nodeConsumedBytes[keyNodes[0]];
    }

    // no more requests from this node
    void finish() {
        unsigned int nodes = MMseqsNode::count;
        waitForPreload();
        std::vector<uint64_t> line(nodes + 6, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES);
        line[0] = requestCount;
        line[1] = 1;
        MMseqsNode::writeExchangeFile(getRequestFile(MMseqsNode::rank), requestCount, NULL, 0, getRequestIndex(MMseqsNode::rank), line, 1);
        requestCount++;
    }

    // read due requests, one caller at a time, false if none
    bool serveRequests() {
        std::unique_lock<std::mutex> taskLock(serveMutex, std::try_to_lock);
        if (taskLock.owns_lock() == false) {
            return false;
        }
        unsigned int nodes = MMseqsNode::count;
        unsigned int rank = MMseqsNode::rank;
        size_t requestLineWords = nodes + 6;
        bool hasRead = false;
        nextRead = serveTimer.getTimediff() + idleSeconds;
        for (unsigned int requester = firstAlignNode; requester < nodes; requester++) {
            if (isRequesting[requester] == 0) {
                continue;
            }
            nextReads[requester] = std::min(nextReads[requester], serveTimer.getTimediff() + idleSeconds);
            if (serveTimer.getTimediff() < nextReads[requester]) {
                nextRead = std::min(nextRead.load(), nextReads[requester]);
                continue;
            }
            size_t lineCount = MMseqsNode::readExchangeIndexLines(requestIndexes[requester], nextRequestLines[requester], requestLineWords, requestLines);
            nextRequestLines[requester] += lineCount;
            hasRead |= lineCount > 0;
            if (lineCount > 0) {
                fastReadEnds[requester] = (requestLines[(lineCount - 1) * requestLineWords + 3] == 0) ? serveTimer.getTimediff() + 1.0 : 0;
            }
            readSeconds[requester] = (lineCount > 0 || serveTimer.getTimediff() < fastReadEnds[requester]) ? 0.01 : std::min(2.0 * readSeconds[requester], idleSeconds);
            nextReads[requester] = serveTimer.getTimediff() + readSeconds[requester];
            nextRead = std::min(nextRead.load(), nextReads[requester]);
            for (size_t l = 0; l < lineCount; l++) {
                const uint64_t *line = requestLines.data() + l * requestLineWords;
                size_t sectionBegin = line[4 + rank];
                size_t sectionEnd = line[5 + rank];
                if (sectionEnd > sectionBegin) {
                    const std::string requestFile = getRequestFile(requester);
                    int fd = MMseqsNode::waitForExchangeFile(requestFile, line[4 + nodes], line[0]);
                    section.resize(sectionEnd - sectionBegin);
                    FileUtil::preadAll(fd, requestFile.c_str(), section.data(), section.size(), sectionBegin);
                    // ignore the close error of a foreign file
                    close(fd);
                    requestedKeys.clear();
                    requestedOffsets.clear();
                    requestedBytes.clear();
                    DBKeyType key = 0;
                    size_t offset = 0;
                    for (const char *data = section.data(); data < section.data() + section.size();) {
                        key += static_cast<DBKeyType>(Util::getPrefKey(data));
                        requestedKeys.push_back(key);
                        if (line[3] != 0) {
                            offset += Util::getPrefKey(data);
                            requestedOffsets.push_back(offset);
                            requestedBytes.push_back(Util::getPrefKey(data));
                        }
                    }
                    size_t consumedBytes = 0;
                    answer(requestedKeys.data(), requestedKeys.size(), requestedOffsets.data(), requestedBytes.data(), line[2], line[3], reply, consumedBytes);
                    const std::vector<uint64_t> memberLine = { memberFileCount, requester, line[0], reply.size(), consumedBytes, 0 };
                    MMseqsNode::writeExchangeFile(getMemberFile(rank, memberFileCount), memberFileCount, reply.data(), reply.size(), getMemberIndex(rank), memberLine, 1);
                    memberFileCount++;
                    servedSize += reply.size();
                    std::string().swap(reply);
                }
                if (line[1] != 0) {
                    isRequesting[requester] = 0;
                    requesterCount--;
                }
            }
        }
        return hasRead;
    }

    // answer requests until every requester finished
    void serve() {
        while (isServed() == false) {
            if (serveRequests() == false) {
                double waitSeconds = nextRead - serveTimer.getTimediff();
                if (waitSeconds > 0) {
                    usleep(static_cast<useconds_t>(waitSeconds * 1000000));
                }
            }
        }
    }

    bool isServed() const {
        return requesterCount == 0;
    }

    // read the next preloaded range, false once all are read
    bool preloadNextRange() {
        std::unique_lock<std::mutex> taskLock(preloadMutex, std::try_to_lock);
        if (taskLock.owns_lock() == false || isPreloadDone) {
            return false;
        }
        lookup.preloadRange(nextPreloadedRange++, threads);
        if (nextPreloadedRange == lookup.getPreloadedRangeCount()) {
            {
                std::lock_guard<std::mutex> lock(preloadWaitMutex);
                isPreloadDone = true;
            }
            preloadCondition.notify_all();
        }
        return true;
    }

    bool isPreloaded() const {
        return isPreloadDone;
    }

    // bytes this node served, read once serve is done
    size_t servedSize;
    // seconds in receive, and before the first request
    double sizeWaitSeconds;
    double memberWaitSeconds;
    double preloadWaitSeconds;

private:
    static const size_t MEMBER_LINE_WORDS = 6;

    std::string getRequestFile(unsigned int requester) const {
        return requests + "." + SSTR(requester);
    }
    std::string getRequestIndex(unsigned int requester) const {
        return requests + ".index." + SSTR(requester);
    }
    std::string getMemberFile(unsigned int node, uint64_t memberFile) const {
        return served + "." + SSTR(node) + "." + SSTR(memberFile);
    }
    std::string getMemberIndex(unsigned int node) const {
        return served + ".index." + SSTR(node);
    }

    // offsets and bytes or members ended by 0, thread safe
    void answer(const DBKeyType *keys, size_t answeredKeyCount, const size_t *entryOffsets, const size_t *entryBytes, size_t consumedBytes, size_t maxMembers,
                std::string &codes, size_t &endConsumedBytes, const unsigned int *answeringNodes = NULL) const {
        endConsumedBytes = 0;
        int maxThreads = 1;
#ifdef OPENMP
        maxThreads = omp_get_max_threads();
#endif
        std::vector<std::string> threadCodes(maxThreads);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            ClusterMemberLookup::EntryReader entryReader;
            char code[10];
            // static schedule gives each thread ascending keys
#pragma omp for schedule(static)
            for (size_t i = 0; i < answeredKeyCount; i++) {
                if (answeringNodes != NULL && answeringNodes[i] != MMseqsNode::rank) {
                    continue;
                }
                if (maxMembers == 0) {
                    // whole offsets, threads answer in parallel
                    const ClusterMemberLookup::Entry entry = lookup.find(keys[i], entryReader);
                    threadCodes[thread_idx].append(code, Util::putPrefKey(code, entry.offset));
                    threadCodes[thread_idx].append(code, Util::putPrefKey(code, entry.end - entry.offset));
                    continue;
                }
                size_t entryBegin = entryOffsets[i];
                ClusterMemberLookup::Entry entry = { entryBegin + consumedBytes, entryBegin + entryBytes[i], keys[i] };
                entryReader.start(entry);
                DBKeyType memberKey;
                for (size_t member = 0; member < maxMembers && lookup.next(entryReader, memberKey); member++) {
                    threadCodes[thread_idx].append(code, Util::putPrefKey(code, static_cast<uint64_t>(memberKey) + 1));
                }
                threadCodes[thread_idx].append(code, Util::putPrefKey(code, 0));
                // one key: consumed bytes end behind its last member
                if (answeredKeyCount == 1) {
                    endConsumedBytes = entryReader.entry.offset - entryBegin;
                }
            }
        }
        codes.clear();
        for (int thread = 0; thread < maxThreads; thread++) {
            codes += threadCodes[thread];
        }
    }

    const std::string requests;
    const std::string served;
    ClusterMemberLookup &lookup;
    const int threads;
    uint64_t requestCount;
    uint64_t memberFileCount;
    bool isSizeRequest;
    std::vector<size_t> nextMemberLines;
    std::vector<MMseqsNode::PolledFile> memberIndexes;
    std::vector<unsigned char> isAnswered;
    std::vector<std::string> nodeCodes;
    std::vector<size_t> nodeConsumedBytes;
    std::string requestCodes;
    // node that answers each key of the last request
    std::vector<unsigned int> keyNodes;
    // fits lines of every requester between two answers
    std::vector<uint64_t> memberLines;
    bool isReceiving;
    Timer receiveTimer;
    MMseqsNode::WaitWarning receiveWarning;
    const size_t firstAlignNode;
    std::mutex serveMutex;
    std::vector<uint64_t> requestLines;
    std::vector<size_t> nextRequestLines;
    std::vector<MMseqsNode::PolledFile> requestIndexes;
    std::vector<unsigned char> isRequesting;
    std::atomic<size_t> requesterCount;
    double idleSeconds;
    std::vector<double> readSeconds;
    std::vector<double> nextReads;
    std::vector<double> fastReadEnds;
    Timer serveTimer;
    std::atomic<double> nextRead;
    std::vector<char> section;
    std::vector<DBKeyType> requestedKeys;
    std::vector<size_t> requestedOffsets;
    std::vector<size_t> requestedBytes;
    std::string reply;
    // read while the first fetches align
    std::mutex preloadMutex;
    std::mutex preloadWaitMutex;
    std::condition_variable preloadCondition;
    std::atomic<bool> isPreloadDone;
    size_t nextPreloadedRange;

    void waitForPreload() {
        if (isPreloadDone) {
            return;
        }
        Timer waitTimer;
        while (isPreloadDone == false) {
            if (preloadNextRange() == false) {
                std::unique_lock<std::mutex> lock(preloadWaitMutex);
                preloadCondition.wait_for(lock, std::chrono::milliseconds(10), [&]() { return isPreloadDone.load(); });
            }
        }
        preloadWaitSeconds = waitTimer.getTimediff();
    }
};

struct PrefRead {
    size_t prefNode;
    size_t firstEntry;
    size_t endEntry;
    uint64_t sizeClasses;
    // window of the entries, or their alignment split
    size_t window;
};

// started pref read (SIZE_MAX: none), counted then decoded by part
struct PrefReadParse {
    PrefReadParse() : prefRead(SIZE_MAX), firstEntry(0), entryCount(0), nextEntry(0), isCounted(false), firstRecord(0), firstTarget(0), size(0), fetchSize(0), alignCells(0) {}
    size_t prefRead;
    size_t firstEntry;
    size_t entryCount;
    size_t nextEntry;
    bool isCounted;
    size_t firstRecord;
    size_t firstTarget;
    // pref bytes of the pref read
    size_t size;
    // added by the chunks
    size_t fetchSize;
    size_t alignCells;
};

// add record bytes per size class, return the classes
static uint64_t addClassSizes(const std::vector<RepTargetsRecord> &records, size_t recordCount, size_t paddedDbSize, std::vector<size_t> &classSize) {
    size_t sizeClassCount = classSize.size();
    uint64_t sizeClasses = 0;
#pragma omp parallel
    {
        std::vector<size_t> threadClassSize(sizeClassCount, 0);
#pragma omp for schedule(dynamic, 100) reduction(|:sizeClasses)
        for (size_t i = 0; i < recordCount; i++) {
            const RepTargetsRecord &record = records[i];
            size_t sizeClass = sizeClassCount - 1 - record.repOrderKeyFirstTarget.repKey / paddedDbSize;
            sizeClasses |= 1ULL << sizeClass;
            threadClassSize[sizeClass] += sizeof(RepTargetsRecord) + record.targetCount * sizeof(DBKeyType);
        }
#pragma omp critical
        for (size_t sizeClass = 0; sizeClass < sizeClassCount; sizeClass++) {
            classSize[sizeClass] += threadClassSize[sizeClass];
        }
    }
    return sizeClasses;
}

// --remove-tmp-files best effort, .partitions first so a rerun repeats kmermatcher
static void removePrefFile(const std::string &pref, const std::string &file) {
    const std::string files[] = { pref + ".partitions", pref + "." + file, pref + ".index." + file };
    for (size_t i = 0; i < 3; i++) {
        if (unlink(files[i].c_str()) != 0 && errno != ENOENT) {
            Debug(Debug::WARNING) << "Cannot delete " << files[i] << ": " << strerror(errno) << "\n";
            return;
        }
    }
}

// pref reads of an align node in passes over size classes
struct PrefPassReader {
    PrefPassReader(size_t memoryLimit, size_t sizeClassCount)
        : maxPrefilterBufferSize(std::max<size_t>(1, memoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR)), prefReadPos(0), passClasses((sizeClassCount > 1) ? ~0ULL : 1), endClass(0),
          classSize(sizeClassCount, 0), isLoaded(false), bufferFetchSize(0), bufferAlignmentCells(0), isFilling(false), fillMaxAlignmentCells(0) {}

    virtual ~PrefPassReader() {}

    // next fill step opens, reads or prefetches pref DBs
    bool isReadStepNext() const {
        return parse.prefRead == SIZE_MAX && isLoaded == false;
    }

    // one step with isOneStep, else up to the next read step
    bool fillPrefilterBuffer(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, size_t maxReadSize, size_t maxBufferAlignmentCells, bool isOneStep) {
        if (isFilling == false) {
            startBuffer(records, targets);
            bufferFetchSize = 0;
            bufferAlignmentCells = 0;
            // resumed fill keeps the cell limit it started with
            fillMaxAlignmentCells = maxBufferAlignmentCells;
            isFilling = true;
        }
        do {
            if (parse.prefRead != SIZE_MAX) {
                parseStep(records, targets, isOneStep);
            } else if (isLoaded == false) {
                readStep(records, targets, maxReadSize);
            } else {
                // order is total, an ordered buffer needs no sort
                if (std::is_sorted(records.begin(), records.end(), RepTargetsRecord::compareByRep) == false) {
                    SORT_PARALLEL(records.begin(), records.end(), RepTargetsRecord::compareByRep);
                }
                isFilling = false;
                isLoaded = false;
            }
        } while (isFilling && isOneStep == false && isReadStepNext() == false);
        return isFilling == false;
    }

protected:
    const size_t maxPrefilterBufferSize;
    std::vector<PrefRead> prefReads;
    size_t prefReadPos;
    uint64_t passClasses;
    size_t endClass;
    std::vector<size_t> classSize;
    // no pref read left, sort next
    bool isLoaded;
    PrefReadParse parse;
    // last window per pref node, windows.size() if none, SIZE_MAX once removed
    std::vector<size_t> lastNodeWindows;
    size_t bufferFetchSize;
    size_t bufferAlignmentCells;
    // fill resumed in later calls
    bool isFilling;
    size_t fillMaxAlignmentCells;

    virtual void startBuffer(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets) = 0;
    // true once the started pref read is in the buffer
    virtual bool parseStep(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, bool isOneStep) = 0;
    // start the next pref read or set isLoaded
    virtual void readStep(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, size_t maxReadSize) = 0;

    // true once no later pref read of the last pass reads node
    bool takeLastNodeRead(size_t node, size_t window) {
        if (endClass != 0 || lastNodeWindows.empty() || lastNodeWindows[node] != window) {
            return false;
        }
        for (size_t i = prefReadPos; i < prefReads.size() && prefReads[i].window == window; i++) {
            if (prefReads[i].prefNode == node && (prefReads[i].sizeClasses & passClasses) != 0) {
                return false;
            }
        }
        lastNodeWindows[node] = SIZE_MAX;
        return true;
    }

    size_t findPassPrefRead(size_t prefRead) {
        while (prefRead < prefReads.size() && (prefReads[prefRead].sizeClasses & passClasses) == 0) {
            prefRead++;
        }
        return prefRead;
    }

    bool startPass() {
        if (endClass == 0) {
            return false;
        }
        endClass--;
        passClasses = 1ULL << endClass;
        for (size_t passSize = classSize[endClass]; endClass > 0 && passSize + classSize[endClass - 1] < maxPrefilterBufferSize; passSize += classSize[endClass]) {
            endClass--;
            passClasses |= 1ULL << endClass;
        }
        prefReadPos = findPassPrefRead(0);
        return true;
    }

    // start parsing the next pref read of the pass
    PrefRead takePrefRead() {
        const PrefRead prefRead = prefReads[prefReadPos];
        parse = PrefReadParse();
        parse.prefRead = prefReadPos;
        parse.firstEntry = prefRead.firstEntry;
        parse.entryCount = prefRead.endEntry - prefRead.firstEntry;
        prefReadPos = findPassPrefRead(prefReadPos + 1);
        return prefRead;
    }
};

// entry 0 is window entry firstEntry
struct PrefFile {
    size_t number;
    DBKeyType firstKey;
    size_t firstEntry;
};

// targets of a rep in one target split
struct TargetSplitPiece {
    const char *data;
    const char *end;
    DBKeyType repKey;
    size_t sizeClass;
    const char *targets;
    size_t targetCount;

    // DB_KEY_INVALID after the last piece
    void next() {
        if (data == end) {
            repKey = DB_KEY_INVALID;
            return;
        }
        repKey += Util::getPrefKey(data);
        sizeClass = Util::getPrefKey(data);
        targets = data;
        // target gaps are >= 1, a 0 byte ends the piece
        for (targetCount = 0; *data != '\0'; data++) {
            targetCount += static_cast<unsigned char>(*data) < 0x80;
        }
        data++;
    }
};

// join the pieces of a rep entry by entry
struct TargetSplitReader : PrefPassReader {
    TargetSplitReader(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<PrefWindow> &windows, size_t maxRecordSize, size_t memoryLimit,
                      size_t paddedDbSize, size_t sizeClassCount, bool isRemovingPrefFiles)
        : PrefPassReader(memoryLimit, sizeClassCount), maxRecordSize(maxRecordSize), par(par), seqDbr(seqDbr), windows(windows),
          targetSplitCount(targetSplitBlockOffsets.size() - 1), paddedDbSize(paddedDbSize), sizeClassCount(sizeClassCount),
          maxRecordTargetCount(std::min((size_t) UINT_MAX, std::max((size_t) 1, maxRecordSize / std::max(seqDbr.getMaxSeqLen() + 2, Util::getPageSize())))),
          windowPos(0), prefNode(SIZE_MAX), partSize(0) {
        for (size_t split = 0; split < targetSplitCount; split++) {
            if (targetSplitAlignNodes[split] == MMseqsNode::rank) {
                targetSplits.push_back(split);
            }
        }
        prefOffsets.resize(keySplitFirstKeys.size() - 1 + MMseqsNode::count * targetSplitCount);
        if (isRemovingPrefFiles) {
            lastNodeWindows.assign(MMseqsNode::count, windows.size());
            for (size_t window = 0; window < windows.size(); window++) {
                if (windows[window].alignmentCells > 0) {
                    lastNodeWindows[windows[window].node] = window;
                }
            }
        }
    }

    // load every window once and mark its size classes
    void countSizeClasses() {
        std::vector<RepTargetsRecord> records;
        std::vector<DBKeyType> targets;
        passClasses = ~0ULL;
        endClass = sizeClassCount;
        for (prefReadPos = 0; prefReadPos < prefReads.size() || takeWindow();) {
            // loading takes the next window ahead
            size_t prefRead = prefReadPos;
            records.clear();
            targets.clear();
            loadPrefRead(records, targets);
            prefReads[prefRead].sizeClasses = addClassSizes(records, records.size(), paddedDbSize, classSize);
        }
    }

    void close() {
        closePrefDbs();
        for (size_t node = 0; node < lastNodeWindows.size(); node++) {
            if (lastNodeWindows[node] != SIZE_MAX) {
                removePrefFiles(node);
            }
        }
        // key split files are read by every node's windows
        if (lastNodeWindows.empty() == false) {
            std::vector<PrefFile> files = getPrefFiles(0);
            for (size_t i = targetSplits.size(); i < files.size(); i++) {
                removePrefFile(par.db2, SSTR(files[i].number));
            }
        }
    }

    const size_t maxRecordSize;

private:
    Parameters &par;
    DBReader<DBKeyType> &seqDbr;
    const std::vector<PrefWindow> &windows;
    const size_t targetSplitCount;
    const size_t paddedDbSize;
    const size_t sizeClassCount;
    const size_t maxRecordTargetCount;
    std::vector<size_t> targetSplits;
    size_t windowPos;
    std::vector<DBReader<DBKeyType> *> prefDbs;
    size_t prefNode;
    // pref files of prefNode in target order
    std::vector<PrefFile> prefFiles;
    std::vector<std::vector<uint64_t> > prefOffsets;
    std::vector<size_t> entrySizes;
    // a part starts from the pieces at its first rep
    size_t partSize;
    std::vector<size_t> entryPartOffsets;
    std::vector<size_t> partTargetOffsets;
    std::vector<size_t> partRecordOffsets;
    std::vector<TargetSplitPiece> partPieces;

    void startBuffer(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets) {
        records.clear();
        targets.clear();
    }

    bool parseStep(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, bool isOneStep) {
        return parsePrefRead(records, targets, isOneStep);
    }

    void readStep(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, size_t maxReadSize) {
        for (isLoaded = true; records.size() * sizeof(RepTargetsRecord) + targets.size() * sizeof(DBKeyType) < maxPrefilterBufferSize;) {
            if (prefReadPos < prefReads.size() || takeWindow()) {
                // keep a pass of several size classes in one buffer
                if (records.empty() == false && (passClasses & (passClasses - 1)) == 0 && (bufferFetchSize >= maxReadSize || bufferAlignmentCells >= fillMaxAlignmentCells)) {
                    break;
                }
                startPrefRead();
                isLoaded = false;
                break;
            } else if (records.empty() == false || startPass() == false) {
                // after the last pass, closing frees the removed pref files
                if (records.empty()) {
                    closePrefDbs();
                }
                break;
            }
        }
    }

    // class file of node per own target split, then key split files
    std::vector<PrefFile> getPrefFiles(size_t node) const {
        std::vector<PrefFile> files;
        for (size_t i = 0; i < targetSplits.size(); i++) {
            size_t splitFirstKey = seqDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[targetSplits[i]]];
            PrefFile file = { keySplitFirstKeys.size() - 1 + node * targetSplitCount + targetSplits[i], static_cast<DBKeyType>(splitFirstKey), classFileFirstEntries[node] };
            files.push_back(file);
        }
        for (size_t i = 0; i < targetSplits.size(); i++) {
            size_t splitFirstKey = seqDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[targetSplits[i]]];
            size_t splitEndKey = seqDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[targetSplits[i] + 1]];
            for (size_t keySplit = std::lower_bound(keySplitFirstKeys.begin(), keySplitFirstKeys.end() - 1, splitFirstKey) - keySplitFirstKeys.begin();
                 keySplit + 1 < keySplitFirstKeys.size() && keySplitFirstKeys[keySplit] < splitEndKey; keySplit++) {
                PrefFile file = { keySplit, static_cast<DBKeyType>(keySplitFirstKeys[keySplit]), 0 };
                files.push_back(file);
            }
        }
        return files;
    }

    // key split files stay open
    void openPrefDbs(size_t node) {
        if (node == prefNode) {
            return;
        }
        prefNode = node;
        prefFiles = getPrefFiles(node);
        prefDbs.resize(prefFiles.size(), NULL);
        for (size_t i = 0; i < prefFiles.size(); i++) {
            if (prefDbs[i] != NULL) {
                if (i >= targetSplits.size()) {
                    continue;
                }
                prefDbs[i]->close();
                delete prefDbs[i];
            }
            std::string file = SSTR(prefFiles[i].number);
            prefDbs[i] = new DBReader<DBKeyType>((par.db2 + "." + file).c_str(), (par.db2 + ".index." + file).c_str(), par.threads,
                                                 DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
            prefDbs[i]->open(DBReader<DBKeyType>::NOSORT);
        }
    }

    void closePrefDbs() {
        for (size_t i = 0; i < prefDbs.size(); i++) {
            prefDbs[i]->close();
            delete prefDbs[i];
        }
        prefDbs.clear();
        prefNode = SIZE_MAX;
    }

    void removePrefFiles(size_t node) {
        std::vector<PrefFile> files = getPrefFiles(node);
        for (size_t i = 0; i < targetSplits.size(); i++) {
            removePrefFile(par.db2, SSTR(files[i].number));
        }
    }

    // window has the same entries in every target split of its node
    bool takeWindow() {
        size_t maxPrefReadSize = getMaxPrefReadSize(maxPrefilterBufferSize);
        size_t prefReadsBefore = prefReads.size();
        for (; prefReads.size() == prefReadsBefore && windowPos < windows.size(); windowPos++) {
            const PrefWindow &window = windows[windowPos];
            if (window.alignmentCells == 0) {
                continue;
            }
            // read each index once
            std::vector<PrefFile> files = getPrefFiles(window.node);
            for (size_t i = 0; i < files.size(); i++) {
                std::vector<uint64_t> &fileOffsets = prefOffsets[files[i].number];
                if (fileOffsets.empty()) {
                    std::string index = FileUtil::readFile(par.db2 + ".index." + SSTR(files[i].number));
                    fileOffsets.resize(index.size() / (2 * sizeof(uint64_t)));
                    for (size_t entry = 0; entry < fileOffsets.size(); entry++) {
                        memcpy(&fileOffsets[entry], index.data() + (2 * entry + 1) * sizeof(uint64_t), sizeof(uint64_t));
                    }
                }
            }
            size_t entryCount = window.endEntry - window.firstEntry;
            entrySizes.assign(entryCount, 0);
            for (size_t i = 0; i < files.size(); i++) {
                const std::vector<uint64_t> &fileOffsets = prefOffsets[files[i].number];
                size_t firstEntry = window.firstEntry - files[i].firstEntry;
                for (size_t entry = 0; entry < entryCount; entry++) {
                    entrySizes[entry] += fileOffsets[firstEntry + entry + 1] - fileOffsets[firstEntry + entry];
                }
            }
            // cut the window into pref reads
            size_t size = 0;
            for (size_t entry = 0; entry < entryCount; entry++) {
                if (entry == 0 || size + entrySizes[entry] > maxPrefReadSize) {
                    PrefRead prefRead = { window.node, window.firstEntry + entry, window.firstEntry + entry + 1, ~0ULL, windowPos };
                    prefReads.push_back(prefRead);
                    size = 0;
                } else {
                    prefReads.back().endEntry++;
                }
                size += entrySizes[entry];
            }
        }
        return prefReads.size() > prefReadsBefore;
    }

    void startEntry(size_t entry, std::vector<TargetSplitPiece> &pieces) {
        for (size_t i = 0; i < prefDbs.size(); i++) {
            size_t fileEntry = entry - prefFiles[i].firstEntry;
            pieces[i].data = prefDbs[i]->getData(fileEntry, 0);
            pieces[i].end = pieces[i].data + prefDbs[i]->getEntryLen(fileEntry) - 1;
            pieces[i].repKey = prefDbs[i]->getDbKey(fileEntry);
            pieces[i].next();
        }
    }

    static DBKeyType getNextRepKey(const std::vector<TargetSplitPiece> &pieces) {
        DBKeyType repKey = DB_KEY_INVALID;
        for (size_t i = 0; i < pieces.size(); i++) {
            repKey = std::min(repKey, pieces[i].repKey);
        }
        return repKey;
    }

    void loadPrefRead(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets) {
        startPrefRead();
        parsePrefRead(records, targets, false);
    }

    // read the next pref read, prefetch the one behind, size the entries
    void startPrefRead() {
        const PrefRead prefRead = takePrefRead();
        openPrefDbs(prefRead.prefNode);
        for (size_t i = 0; i < prefDbs.size(); i++) {
            prefDbs[i]->loadData(prefRead.firstEntry - prefFiles[i].firstEntry, prefRead.endEntry - prefFiles[i].firstEntry);
        }
        // last pref read of a window takes the next window to prefetch it
        if (prefReadPos == prefReads.size()) {
            takeWindow();
        }
        // prefetch here, io_uring reads can fail once their thread ends
        if (prefReadPos < prefReads.size() && prefReads[prefReadPos].prefNode == prefRead.prefNode) {
            for (size_t i = 0; i < prefDbs.size(); i++) {
                prefDbs[i]->prefetchData(prefReads[prefReadPos].firstEntry - prefFiles[i].firstEntry,
                                         prefReads[prefReadPos].endEntry - prefFiles[i].firstEntry);
            }
        }
        // open DBs keep the removed files until they are closed
        if (takeLastNodeRead(prefRead.prefNode, prefRead.window)) {
            removePrefFiles(prefRead.prefNode);
        }
        entryPartOffsets.resize(parse.entryCount + 1);
        entryPartOffsets[0] = 0;
        for (size_t entry = parse.firstEntry; entry < parse.firstEntry + parse.entryCount; entry++) {
            size_t entrySize = 0;
            for (size_t i = 0; i < prefDbs.size(); i++) {
                entrySize += prefDbs[i]->getEntryLen(entry - prefFiles[i].firstEntry);
            }
            parse.size += entrySize;
            entryPartOffsets[entry - parse.firstEntry + 1] = entrySize;
        }
        // parts of about 1 / threads of the read, fewer than entries + 2 * threads
        partSize = std::max((size_t) 1, parse.size / par.threads);
        for (size_t entry = 0; entry < parse.entryCount; entry++) {
            entryPartOffsets[entry + 1] = entryPartOffsets[entry] + entryPartOffsets[entry + 1] / partSize + 1;
        }
        partTargetOffsets.assign(entryPartOffsets[parse.entryCount] + 1, 0);
        partRecordOffsets.assign(entryPartOffsets[parse.entryCount] + 1, 0);
        partPieces.resize(entryPartOffsets[parse.entryCount] * prefDbs.size());
    }

    // count targets and records per part, then write them at their offsets
    bool parsePrefRead(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, bool isOneStep) {
        size_t firstEntry = parse.firstEntry;
        size_t entryCount = parse.entryCount;
        if (parse.isCounted == false) {
            parse.nextEntry = forEachEntryChunk(parse.nextEntry, entryCount, 1, isOneStep, [&](size_t begin, size_t end) {
                std::vector<TargetSplitPiece> pieces(prefDbs.size());
                for (size_t entry = begin; entry < end; entry++) {
                    startEntry(firstEntry + entry, pieces);
                    size_t part = entryPartOffsets[entry];
                    std::copy(pieces.begin(), pieces.end(), partPieces.begin() + part * pieces.size());
                    size_t targetCount = 0;
                    size_t recordCount = 0;
                    size_t partEntrySize = 0;
                    for (DBKeyType repKey = getNextRepKey(pieces); repKey != DB_KEY_INVALID; repKey = getNextRepKey(pieces)) {
                        if (partEntrySize >= partSize) {
                            partTargetOffsets[part + 1] = targetCount;
                            partRecordOffsets[part + 1] = recordCount;
                            part++;
                            std::copy(pieces.begin(), pieces.end(), partPieces.begin() + part * pieces.size());
                            targetCount = 0;
                            recordCount = 0;
                            partEntrySize = 0;
                        }
                        size_t repTargetCount = 0;
                        size_t sizeClass = 0;
                        for (size_t i = 0; i < pieces.size(); i++) {
                            if (pieces[i].repKey == repKey) {
                                repTargetCount += pieces[i].targetCount;
                                // key split pieces hold class 0
                                sizeClass = std::max(sizeClass, std::min(sizeClassCount - 1, pieces[i].sizeClass));
                                partEntrySize += pieces[i].data - pieces[i].targets;
                                pieces[i].next();
                            }
                        }
                        if (((passClasses >> sizeClass) & 1) != 0) {
                            targetCount += repTargetCount;
                            recordCount += (repTargetCount + maxRecordTargetCount - 1) / maxRecordTargetCount;
                        }
                    }
                    partTargetOffsets[part + 1] = targetCount;
                    partRecordOffsets[part + 1] = recordCount;
                }
            });
            if (parse.nextEntry < entryCount) {
                return false;
            }
            parse.isCounted = true;
            parse.nextEntry = 0;
            if (isOneStep) {
                return false;
            }
        }
        size_t partCount = entryPartOffsets[entryCount];
        if (parse.nextEntry == 0) {
            sumOffsets(partTargetOffsets.data(), partCount, par.threads);
            sumOffsets(partRecordOffsets.data(), partCount, par.threads);
            parse.firstRecord = records.size();
            parse.firstTarget = targets.size();
            records.resize(parse.firstRecord + partRecordOffsets[partCount]);
            targets.resize(parse.firstTarget + partTargetOffsets[partCount]);
        }
        size_t firstRecord = parse.firstRecord;
        size_t firstTarget = parse.firstTarget;
        parse.nextEntry = forEachEntryChunk(parse.nextEntry, partCount, 1, isOneStep, [&](size_t begin, size_t end) {
            std::vector<TargetSplitPiece> pieces(prefDbs.size());
            size_t fetchSize = 0;
            size_t alignCells = 0;
            for (size_t part = begin; part < end; part++) {
                if (partTargetOffsets[part + 1] == partTargetOffsets[part]) {
                    continue;
                }
                std::copy(partPieces.begin() + part * pieces.size(), partPieces.begin() + (part + 1) * pieces.size(), pieces.begin());
                DBKeyType *repTargets = targets.data() + firstTarget + partTargetOffsets[part];
                const DBKeyType *partTargetsEnd = targets.data() + firstTarget + partTargetOffsets[part + 1];
                size_t j = firstRecord + partRecordOffsets[part];
                for (DBKeyType repKey = getNextRepKey(pieces); repTargets != partTargetsEnd; repKey = getNextRepKey(pieces)) {
                    // pieces in target split order, so targets ascend
                    size_t repTargetCount = 0;
                    size_t sizeClass = 0;
                    for (size_t i = 0; i < pieces.size(); i++) {
                        if (pieces[i].repKey == repKey) {
                            sizeClass = std::max(sizeClass, std::min(sizeClassCount - 1, pieces[i].sizeClass));
                            const char *data = pieces[i].targets;
                            size_t targetCount = (((passClasses >> sizeClass) & 1) != 0) ? pieces[i].targetCount : 0;
                            DBKeyType target = prefFiles[i].firstKey - 1;
                            for (size_t k = 0; k < targetCount; k++) {
                                target += Util::getPrefKey(data);
                                repTargets[repTargetCount++] = target;
                            }
                            pieces[i].next();
                        }
                    }
                    if (repTargetCount == 0) {
                        continue;
                    }
                    size_t repSize = seqDbr.getEntryLen(repKey) + FETCHED_KEY_SIZE;
                    size_t targetRange = SIZE_MAX;
                    for (size_t first = 0; first < repTargetCount; first += maxRecordTargetCount, j++) {
                        RepTargetsRecord &record = records[j];
                        record.repOrderKeyFirstTarget.repKey = (sizeClassCount - 1 - sizeClass) * paddedDbSize + repKey;
                        record.repOrderKeyFirstTarget.memberKey = repTargets[first];
                        record.targetCount = static_cast<unsigned int>(std::min(repTargetCount - first, maxRecordTargetCount));
                        record.targetOffset = repTargets + first - targets.data();
                        record.readSize = repSize;
                        for (unsigned int i = 0; i < record.targetCount; i++) {
                            size_t targetLength = seqDbr.getSeqLenFromRange(targetRange, repTargets[first + i]);
                            record.readSize += targetLength + 2 + FETCHED_KEY_SIZE;
                            alignCells += (repSize - FETCHED_KEY_SIZE - 2) * targetLength;
                        }
                        fetchSize += record.readSize;
                    }
                    repTargets += repTargetCount;
                }
            }
            __sync_fetch_and_add(&parse.fetchSize, fetchSize);
            __sync_fetch_and_add(&parse.alignCells, alignCells);
        });
        if (parse.nextEntry < partCount) {
            return false;
        }
        bufferFetchSize += parse.fetchSize;
        bufferAlignmentCells += parse.alignCells;
        parse.prefRead = SIZE_MAX;
        return true;
    }
};

// examined (rep, target) pairs of the aligning loop
static std::atomic<size_t> examinedTargetPairCount(0);

struct TargetAligners {
    Sequence query;
    Sequence target;
    BlockAligner blockAligner;
    // --align2clust-aligner-mode 1: reps of a batch share the SWIPE lanes
    SwipeTargets swipeTargets;

    TargetAligners(const Parameters &par, AlignmentScoring &scoring, size_t maxSeqLen)
        : query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection),
          target(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection),
          blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                       &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                       -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid()),
          swipeTargets(par, scoring.subMat, scoring.evaluer) {}
};

struct MemberAligners {
    Sequence query;
    Sequence element;
    BlockAligner blockAligner;
    Matcher matcher;
    SwipeBatch swipeMembers;
    SwipeTargets memberTargets;
    MemberChecks memberChecks;

    MemberAligners(const Parameters &par, AlignmentScoring &scoring, size_t maxSeqLen, bool isSwipeEnabled)
        : query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection),
          element(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection),
          blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                       &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                       -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid()),
          matcher(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                  par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid(), 0.0, par.zdrop),
          swipeMembers(par, scoring.subMat, scoring.evaluer, maxSeqLen),
          memberTargets(par, scoring.subMat, scoring.evaluer),
          memberChecks(isSwipeEnabled ? &swipeMembers : NULL, true, isSwipeEnabled ? &memberTargets : NULL) {}
};

// lane outsiders are among the first and last 3 by (-length difference, position)
struct ScalarMembers {
    size_t memberCount;
    std::vector<std::pair<int, size_t> > first;
    std::vector<std::pair<int, size_t> > last;

    ScalarMembers() : memberCount(0) {}

    void clear() {
        memberCount = 0;
        first.clear();
        last.clear();
    }

    void add(int lengthDifference) {
        const std::pair<int, size_t> member(-lengthDifference, memberCount++);
        first.insert(std::upper_bound(first.begin(), first.end(), member), member);
        if (first.size() > 3) {
            first.pop_back();
        }
        last.insert(std::upper_bound(last.begin(), last.end(), member), member);
        if (last.size() > 3) {
            last.erase(last.begin());
        }
    }

    // list positions of the members outside the lanes
    void getPositions(std::vector<size_t> &positions) const {
        positions.clear();
        for (size_t i = 0; i < first.size(); i++) {
            if (isSwipeLaneMember(i, memberCount) == false) {
                positions.push_back(first[i].second);
            }
        }
        for (size_t i = 0; i < last.size(); i++) {
            if (isSwipeLaneMember(memberCount - last.size() + i, memberCount) == false) {
                positions.push_back(last[i].second);
            }
        }
    }
};

struct AcceptedRep {
    size_t repKey;
    size_t begin;
    size_t end;
    unsigned int log;
};

struct MergedReps {
    size_t firstLogRep;
    size_t firstMergedRep;
    unsigned int log;
};

// accepted pairs of a node, in rep order
struct AcceptedLog {
    AcceptedLog() : readSize(0), nextRepKey(0), clusteredWordCount(0), wordOffset(0), bitOffset(0), mergedReps(0) {}
    // clustered bytes
    size_t readSize;
    // read and not clustered
    std::vector<uint64_t> unclusteredWords;
    std::vector<size_t> repWordOffsets;
    size_t nextRepKey;
    size_t clusteredWordCount;
    size_t wordOffset;
    size_t bitOffset;
    size_t mergedReps;
};

struct ClusteredLog {
    FILE *file;
    size_t bytes;
    // bits of the log words from 64 * firstBitsWord on
    std::vector<uint64_t> bits;
    size_t firstBitsWord;
    char *buffer;
};

// master clusters the accepted pairs in key order, step by step
struct AcceptedPairClustering {
    static const int STEP_DONE = 0;
    static const int STEP_WORKED = 1;
    static const int STEP_IDLE = 2;

    AcceptedPairClustering(int threads, size_t dbSize, size_t memoryLimit, const DBReader<DBKeyType> &seqDbr, const std::vector<unsigned int> &pageAlignNodes, int mode,
                           size_t paddedDbSize, size_t sizeClassCount, const std::string &accepted, const std::string &assignedKeysLog, const std::string &clusteredLogPrefix,
                           const std::string &setCoverPrefix, const std::string &assignedBits, bool hasWriterThread)
        : idleMicroseconds(10000), isDone(false), threads(threads), dbSize(dbSize), seqDbr(seqDbr), pageAlignNodes(pageAlignNodes), mode(mode), paddedDbSize(paddedDbSize),
          sizeClassCount(sizeClassCount), accepted(accepted), assignedKeysLog(assignedKeysLog), clusteredLogPrefix(clusteredLogPrefix), assignedBits(assignedBits),
          hasWriterThread(hasWriterThread), assignedWords((dbSize + 63) / 64), orderKeyEnd((sizeClassCount - 1) * paddedDbSize + dbSize),
          assigned(static_cast<uint64_t *>(mem_align(2 * 1024 * 1024, assignedWords * sizeof(uint64_t)))), threadAssignedKeys(threads),
          nodeHasProgress(MMseqsNode::count, 0), nodeRepOrderKeys(MMseqsNode::count, 0), nodeLogSize(MMseqsNode::count, 0), nodeClusteredLogs(MMseqsNode::count),
          nodeAcceptedLogs(MMseqsNode::count), nodeAcceptedProgress(MMseqsNode::count), assignedKeysProgress(assignedKeysLog + ".progress"), stepWords(getStepWords(memoryLimit)),
          nodeStepWords(std::max<size_t>(1, stepWords / MMseqsNode::count)),
          logReadBytesPerNode(std::max<size_t>(1, std::min(stepWords, memoryLimit / 64 / sizeof(uint64_t)) / MMseqsNode::count) * sizeof(uint64_t)),
          isWritingAssignedKeys(false), memberCount(0), pairFile(pairSets, setCoverPrefix + "_clustered_pairsets", stepWords, memoryLimit, assigned),
          smallSetFiles(setCoverPrefix + "_clustered_sets", memoryLimit, assigned), largeSetFiles(setCoverPrefix + "_clustered_large", memoryLimit), waitSeconds(0),
          clusteredOrderKey(0), isIdle(false), progress(100) {
#ifdef MADV_HUGEPAGE
        // random probes of a bitmap larger than the TLB reach
        madvise(assigned, assignedWords * sizeof(uint64_t), MADV_HUGEPAGE);
#endif
        memset(assigned, 0, assignedWords * sizeof(uint64_t));
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const ClusteredLog clusteredLog = { FileUtil::openAndDelete((clusteredLogPrefix + "." + SSTR(node)).c_str(), "w"), 0, std::vector<uint64_t>(), 0, new(std::nothrow) char[CLUSTERED_LOG_BUFFER_SIZE] };
            Util::checkAllocation(clusteredLog.buffer, "Can not allocate clustered log buffer memory in align2clust");
            setvbuf(clusteredLog.file, clusteredLog.buffer, _IOFBF, CLUSTERED_LOG_BUFFER_SIZE);
            nodeClusteredLogs[node] = clusteredLog;
            nodeAcceptedProgress[node].file = accepted + "." + SSTR(node) + ".progress";
        }
        assignedKeysHandle = FileUtil::openAndDelete(assignedKeysLog.c_str(), "w");
        assignedKeysSize = 0;
    }

    ~AcceptedPairClustering() {
        if (assignedKeysWriter.joinable()) {
            assignedKeysWriter.join();
        }
    }

    int step() {
        // idle step reads again after its sleep
        if (isIdle && gapTimer.getTimediff() < idleMicroseconds / 1000000.0) {
            return STEP_IDLE;
        }
        // time between two steps counts as waiting
        waitSeconds += gapTimer.getTimediff();
        gapTimer.reset();
        isIdle = false;
        if (clusteredOrderKey >= orderKeyEnd) {
            isDone = true;
            return STEP_DONE;
        }
        // write and publish the decided steps while the writer is busy
        if (assignedKeyWords.empty() == false && (isWritingAssignedKeys == false || assignedKeyWords.size() >= stepWords)) {
            if (assignedKeysWriter.joinable()) {
                assignedKeysWriter.join();
            }
            assignedKeyWords.swap(writtenAssignedKeys);
            assignedKeyWords.clear();
            isWritingAssignedKeys = true;
            if (hasWriterThread) {
                assignedKeysWriter = std::thread(&AcceptedPairClustering::writeAssignedKeys, this, clusteredOrderKey);
            } else {
                writeAssignedKeys(clusteredOrderKey);
            }
        }

        Timer pollTimer;
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            size_t repKey;
            size_t logSize;
            if ((nodeHasProgress[node] && nodeRepOrderKeys[node] == orderKeyEnd) || MMseqsNode::readLogProgress(nodeAcceptedProgress[node], repKey, logSize) == false) {
                continue;
            }
            if (nodeHasProgress[node] && repKey <= nodeRepOrderKeys[node] && logSize <= nodeLogSize[node]) {
                continue;
            }
            nodeHasProgress[node] = 1;
            nodeRepOrderKeys[node] = repKey;
            nodeLogSize[node] = logSize;
        }
        double pollCostSeconds = pollTimer.getTimediff();
        size_t doneRepOrderKey = orderKeyEnd;
        size_t slowestNode = 0;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            size_t nodeRepKey = nodeHasProgress[node] ? nodeRepOrderKeys[node] : 0;
            if (nodeRepKey < doneRepOrderKey) {
                doneRepOrderKey = nodeRepKey;
                slowestNode = node;
            }
        }
        if (doneRepOrderKey <= clusteredOrderKey) {
            // only the slowest node gates the step
            idleMicroseconds = static_cast<unsigned int>(1000000 * std::min(0.16, std::max(0.01, LOG_COST_FACTOR * pollCostSeconds)));
            waitWarning.check(accepted + "." + SSTR(slowestNode) + ".progress");
            gapTimer = pollTimer;
            isIdle = true;
            return STEP_IDLE;
        }

        // read every log up to doneRepOrderKey
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            AcceptedLog &acceptedLog = nodeAcceptedLogs[node];
            std::vector<uint64_t> &nodeWords = acceptedLog.unclusteredWords;
            std::vector<size_t> &repWordOffsets = acceptedLog.repWordOffsets;
            repWordOffsets.clear();
            size_t word = 0;
            if (acceptedLog.nextRepKey < doneRepOrderKey) {
                acceptedLog.nextRepKey = nodeRepOrderKeys[node];
                const std::string log = accepted + "." + SSTR(node);
                size_t from = acceptedLog.readSize + nodeWords.size() * sizeof(uint64_t);
                bool isCut = false;
                while (true) {
                    for (; word < nodeWords.size(); word++) {
                        if ((nodeWords[word] & ACCEPTED_REP_BIT) == 0) {
                            continue;
                        }
                        size_t repKey = getRepOrderKey(nodeWords[word]);
                        if (repKey >= doneRepOrderKey || (word >= nodeStepWords && repKey > clusteredOrderKey)) {
                            acceptedLog.nextRepKey = repKey;
                            isCut = true;
                            break;
                        }
                        repWordOffsets.push_back(word);
                    }
                    if (isCut || from >= nodeLogSize[node]) {
                        break;
                    }
                    size_t to = std::min(nodeLogSize[node], from + logReadBytesPerNode);
                    nodeWords.resize(word + (to - from) / sizeof(uint64_t));
                    MMseqsNode::readLog(log, from, to, reinterpret_cast<char *>(nodeWords.data() + word));
                    from = to;
                }
            }
            repWordOffsets.push_back(word);
        }

        // end the step at the first cut rep
        size_t stepRepOrderKey = doneRepOrderKey;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            stepRepOrderKey = std::min(stepRepOrderKey, nodeAcceptedLogs[node].nextRepKey);
        }
        size_t stepWordCount = 0;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            AcceptedLog &acceptedLog = nodeAcceptedLogs[node];
            std::vector<size_t> &repWordOffsets = acceptedLog.repWordOffsets;
            const std::vector<uint64_t> &nodeWords = acceptedLog.unclusteredWords;
            size_t clusteredReps = std::lower_bound(repWordOffsets.begin(), repWordOffsets.end() - 1, stepRepOrderKey,
                [&](size_t word, size_t repKey) { return getRepOrderKey(nodeWords[word]) < repKey; }) - repWordOffsets.begin();
            if (clusteredReps + 1 < repWordOffsets.size()) {
                acceptedLog.nextRepKey = getRepOrderKey(nodeWords[repWordOffsets[clusteredReps]]);
            }
            acceptedLog.clusteredWordCount = repWordOffsets[clusteredReps];
            repWordOffsets.resize(clusteredReps);
            repWordOffsets.push_back(acceptedLog.clusteredWordCount);
            acceptedLog.wordOffset = stepWordCount;
            stepWordCount += acceptedLog.clusteredWordCount;
            // clustered bit of a word is at its node log position
            acceptedLog.bitOffset = acceptedLog.readSize / sizeof(uint64_t) - 64 * nodeClusteredLogs[node].firstBitsWord;
            acceptedLog.readSize += acceptedLog.clusteredWordCount * sizeof(uint64_t);
        }
        logWords.resize(stepWordCount);
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            AcceptedLog &acceptedLog = nodeAcceptedLogs[node];
            std::vector<uint64_t> &nodeWords = acceptedLog.unclusteredWords;
            std::copy(nodeWords.begin(), nodeWords.begin() + acceptedLog.clusteredWordCount, logWords.begin() + acceptedLog.wordOffset);
            nodeWords.erase(nodeWords.begin(), nodeWords.begin() + acceptedLog.clusteredWordCount);
            std::vector<size_t> &repWordOffsets = acceptedLog.repWordOffsets;
            for (size_t rep = 0; rep < repWordOffsets.size(); rep++) {
                repWordOffsets[rep] += acceptedLog.wordOffset;
            }
        }

        // merge the reps of the logs in key order
        std::priority_queue<std::pair<size_t, unsigned int>, std::vector<std::pair<size_t, unsigned int> >, std::greater<std::pair<size_t, unsigned int> > > nextReps;
        for (unsigned int log = 0; log < MMseqsNode::count; log++) {
            nodeAcceptedLogs[log].mergedReps = 0;
            if (nodeAcceptedLogs[log].repWordOffsets.size() > 1) {
                nextReps.push(std::make_pair(getRepOrderKey(logWords[nodeAcceptedLogs[log].repWordOffsets[0]]), log));
            }
        }
        mergedReps.clear();
        size_t repCount = 0;
        while (nextReps.empty() == false) {
            unsigned int log = nextReps.top().second;
            nextReps.pop();
            AcceptedLog &acceptedLog = nodeAcceptedLogs[log];
            const std::vector<size_t> &repWordOffsets = acceptedLog.repWordOffsets;
            size_t endKey = nextReps.empty() ? stepRepOrderKey : nextReps.top().first;
            size_t endRep = std::lower_bound(repWordOffsets.begin() + acceptedLog.mergedReps + 1, repWordOffsets.end() - 1, endKey,
                [&](size_t word, size_t repKey) { return getRepOrderKey(logWords[word]) < repKey; }) - repWordOffsets.begin();
            const MergedReps logReps = { acceptedLog.mergedReps, repCount, log };
            mergedReps.push_back(logReps);
            repCount += endRep - acceptedLog.mergedReps;
            acceptedLog.mergedReps = endRep;
            if (endRep + 1 < repWordOffsets.size()) {
                nextReps.push(std::make_pair(getRepOrderKey(logWords[repWordOffsets[endRep]]), log));
            }
        }

        reps.resize(repCount);
        const size_t repsPerTask = 1024;
#pragma omp parallel for schedule(static)
        for (size_t first = 0; first < repCount; first += repsPerTask) {
            const size_t end = first + std::min(repsPerTask, repCount - first);
            size_t merged = std::upper_bound(mergedReps.begin(), mergedReps.end(), first,
                [](size_t rep, const MergedReps &next) { return rep < next.firstMergedRep; }) - mergedReps.begin() - 1;
            for (size_t rep = first; rep < end;) {
                const MergedReps &logReps = mergedReps[merged];
                size_t segmentEnd = std::min(end, (merged + 1 < mergedReps.size()) ? mergedReps[merged + 1].firstMergedRep : repCount);
                const std::vector<size_t> &repWordOffsets = nodeAcceptedLogs[logReps.log].repWordOffsets;
                for (; rep < segmentEnd; rep++) {
                    size_t logRep = logReps.firstLogRep + rep - logReps.firstMergedRep;
                    const AcceptedRep acceptedRep = { getRepOrderKey(logWords[repWordOffsets[logRep]]), repWordOffsets[logRep] + 1, repWordOffsets[logRep + 1], logReps.log };
                    reps[rep] = acceptedRep;
                }
                merged++;
            }
        }

        // pairs with assigned keys stay out
        pairIsFree.resize(logWords.size());
        repKeyOffsets.assign(repCount + 1, 0);
#pragma omp parallel for schedule(dynamic, 1024)
        for (size_t rep = 0; rep < repCount; rep++) {
            bool repIsFree = isAssigned(reps[rep].repKey % paddedDbSize) == false;
            size_t freeCount = 0;
            for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                pairIsFree[i] = repIsFree && isAssigned(logWords[i]) == false;
                freeCount += pairIsFree[i];
            }
            repKeyOffsets[rep + 1] = (freeCount > 0) ? freeCount + 1 : 0;
        }
        sumOffsets(repKeyOffsets.data(), repCount, threads);
        if (mode == Parameters::SET_COVER) {
            // pieces of a rep from several align nodes form one set
            setReps.clear();
            setSizes.clear();
            for (size_t rep = 0; rep < repCount; rep++) {
                if (rep == 0 || reps[rep].repKey != reps[rep - 1].repKey) {
                    setReps.push_back(rep);
                    setSizes.push_back(1);
                }
                setSizes.back() += std::max((size_t) 1, repKeyOffsets[rep + 1] - repKeyOffsets[rep]) - 1;
            }
            setReps.push_back(repCount);
            // set waits until no larger set can come
            size_t setPairs = 0;
            for (size_t set = 0; set < setSizes.size(); set++) {
                setPairs += setSizes[set] == 2;
            }
            bool isBufferingPairs = pairFile.prepareBatch(setPairs);
            size_t pairOffset = pairSets.size();
            if (isBufferingPairs) {
                pairSets.resize(pairOffset + setPairs);
            }
            size_t pairIndex = pairOffset;
            for (size_t set = 0; set < setSizes.size(); set++) {
                for (size_t rep = setReps[set]; setSizes[set] == 2 && rep < setReps[set + 1]; rep++) {
                    for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                        if (pairIsFree[i]) {
                            const ClusterPair pairSet = { reps[rep].repKey % paddedDbSize, static_cast<DBKeyType>(logWords[i]) };
                            if (isBufferingPairs) {
                                pairSets[pairIndex++] = pairSet;
                            } else {
                                pairFile.push(pairSet);
                            }
                        }
                    }
                }
            }
            DBKeyType smallMembers[SetCoverSmallSetFiles::MAX_SIZE];
            for (size_t set = 0; set < setSizes.size(); set++) {
                size_t setSize = setSizes[set];
                if (setSize < 3 || setSize > SetCoverSmallSetFiles::MAX_SIZE) {
                    continue;
                }
                smallMembers[0] = reps[setReps[set]].repKey % paddedDbSize;
                size_t member = 1;
                for (size_t rep = setReps[set]; rep < setReps[set + 1]; rep++) {
                    for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                        if (pairIsFree[i]) {
                            smallMembers[member++] = logWords[i];
                        }
                    }
                }
                smallSetFiles.append(setSize, smallMembers);
            }
            largeSetFiles.appendStep(reps, setReps, setSizes, paddedDbSize, [&](size_t i) { return pairIsFree[i]; },
                                     [&](size_t i) { return logWords[i]; });
            // no larger set can come
            size_t maxSetSize = (stepRepOrderKey < orderKeyEnd) ? static_cast<size_t>(2) << (sizeClassCount - 1 - stepRepOrderKey / paddedDbSize) : 0;
            largeSetFiles.assignSetsLargerThan(maxSetSize, [&](DBKeyType key) { return isAssigned(key); }, [&](DBKeyType repKey, DBKeyType memberKey) {
                setAssignedSerial(memberKey, threadAssignedKeys[0]);
                if (memberKey != repKey) {
                    addClusterPair(repKey, memberKey);
                }
            }, pairFile, smallSetFiles);
            smallSetFiles.assignSetsLargerThan(maxSetSize, [&](size_t setSize, const DBKeyType *members) {
                if (isAssigned(members[0])) {
                    return;
                }
                DBKeyType validMembers[SetCoverSmallSetFiles::MAX_SIZE];
                size_t validCount = 0;
                for (size_t member = 0; member < setSize; member++) {
                    if (isAssigned(members[member]) == false) {
                        validMembers[validCount++] = members[member];
                    }
                }
                if (validCount <= 1) {
                    return;
                }
                if (validCount < setSize) {
                    if (validCount == 2) {
                        const ClusterPair pairSet = { validMembers[0], validMembers[1] };
                        pairFile.push(pairSet);
                    } else {
                        smallSetFiles.append(validCount, validMembers);
                    }
                    return;
                }
                for (size_t member = 0; member < setSize; member++) {
                    setAssignedSerial(members[member], threadAssignedKeys[0]);
                    if (members[member] != members[0]) {
                        addClusterPair(members[0], members[member]);
                    }
                }
            });
            // sets of two go last, the larger rep first
            if (stepRepOrderKey == orderKeyEnd) {
                pairFile.finish([&](const ClusterPair &pair) {
                    if (isAssigned(pair.repKey) == false && isAssigned(pair.memberKey) == false) {
                        setAssignedSerial(pair.repKey, threadAssignedKeys[0]);
                        setAssignedSerial(pair.memberKey, threadAssignedKeys[0]);
                        addClusterPair(pair.repKey, pair.memberKey);
                    }
                });
                std::vector<ClusterPair>().swap(pairSets);
            }
        } else {
            // first rep of every key of the step
            size_t tableBits = 1;
            while ((1ULL << tableBits) < 2 * repKeyOffsets[repCount]) {
                tableBits++;
            }
            size_t tableSize = 1ULL << tableBits;
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                std::vector<uint64_t> &bits = nodeClusteredLogs[node].bits;
                bits.resize(std::max(bits.size(), (nodeAcceptedLogs[node].bitOffset + nodeAcceptedLogs[node].clusteredWordCount + 63) / 64), 0);
            }
            keyReps.resize(std::max(keyReps.size(), tableSize));
            repSharesKey.assign(repCount, 0);
#pragma omp parallel for schedule(static)
            for (size_t slot = 0; slot < tableSize; slot++) {
                keyReps[slot] = std::make_pair(DB_KEY_INVALID, std::numeric_limits<size_t>::max());
            }
            const auto addStepKey = [&](DBKeyType key, size_t rep) {
                size_t slot = (key * 0x9E3779B97F4A7C15ULL) >> (64 - tableBits);
                DBKeyType slotKey = DB_KEY_INVALID;
                while (__atomic_compare_exchange_n(&keyReps[slot].first, &slotKey, key, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED) == false && slotKey != key) {
                    slot = (slot + 1) & (tableSize - 1);
                    slotKey = DB_KEY_INVALID;
                }
                size_t firstRep = __atomic_load_n(&keyReps[slot].second, __ATOMIC_RELAXED);
                while (rep < firstRep && __atomic_compare_exchange_n(&keyReps[slot].second, &firstRep, rep, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED) == false) {
                }
                // pieces of one rep share only the rep key
                if (firstRep != std::numeric_limits<size_t>::max() && (firstRep == rep || reps[firstRep].repKey != reps[rep].repKey)) {
                    __atomic_store_n(&repSharesKey[std::max(rep, firstRep)], 1, __ATOMIC_RELAXED);
                }
            };
#pragma omp parallel for schedule(dynamic, 1024)
            for (size_t rep = 0; rep < repCount; rep++) {
                if (repKeyOffsets[rep + 1] == repKeyOffsets[rep]) {
                    continue;
                }
                addStepKey(reps[rep].repKey, rep);
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (pairIsFree[i]) {
                        addStepKey(logWords[i], rep);
                    }
                }
            }
            // pieces of a rep go together
            for (size_t rep = 1; rep < repCount; rep++) {
                if (reps[rep].repKey == reps[rep - 1].repKey) {
                    repSharesKey[rep] |= repSharesKey[rep - 1];
                }
            }
            for (size_t rep = repCount; rep > 1; rep--) {
                if (reps[rep - 1].repKey == reps[rep - 2].repKey) {
                    repSharesKey[rep - 2] |= repSharesKey[rep - 1];
                }
            }
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
#pragma omp for schedule(dynamic, 1024) reduction(+:memberCount)
                for (size_t rep = 0; rep < repCount; rep++) {
                    if (repKeyOffsets[rep + 1] == repKeyOffsets[rep] || repSharesKey[rep]) {
                        continue;
                    }
                    setAssigned(reps[rep].repKey, threadAssignedKeys[thread_idx]);
                    for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                        if (pairIsFree[i]) {
                            setAssigned(logWords[i], threadAssignedKeys[thread_idx]);
                            const AcceptedLog &acceptedLog = nodeAcceptedLogs[reps[rep].log];
                            size_t bit = i - acceptedLog.wordOffset + acceptedLog.bitOffset;
                            __sync_fetch_and_or(&nodeClusteredLogs[reps[rep].log].bits[bit / 64], 1ULL << (bit % 64));
                            memberCount++;
                        }
                    }
                }
            }
            sharingReps.clear();
            for (size_t rep = 0; rep < repCount; rep++) {
                if (repKeyOffsets[rep + 1] != repKeyOffsets[rep] && repSharesKey[rep] != 0) {
                    sharingReps.push_back(rep);
                }
            }
            bool isSharingRepFree = false;
            for (size_t sharingRep = 0; sharingRep < sharingReps.size(); sharingRep++) {
                if (sharingRep + PREFETCH_DISTANCE < sharingReps.size()) {
                    const AcceptedRep &nextRep = reps[sharingReps[sharingRep + PREFETCH_DISTANCE]];
                    __builtin_prefetch(&assigned[nextRep.repKey / 64]);
                    for (size_t i = nextRep.begin; i < std::min(nextRep.end, nextRep.begin + PREFETCH_DISTANCE); i++) {
                        __builtin_prefetch(&assigned[logWords[i] / 64]);
                    }
                }
                size_t rep = sharingReps[sharingRep];
                DBKeyType repKey = reps[rep].repKey;
                // rep assigned by its own earlier piece is still free
                if (sharingRep == 0 || reps[sharingReps[sharingRep - 1]].repKey != repKey) {
                    isSharingRepFree = isAssigned(repKey) == false;
                }
                if (isSharingRepFree == false) {
                    continue;
                }
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (i + PREFETCH_DISTANCE < reps[rep].end) {
                        __builtin_prefetch(&assigned[logWords[i + PREFETCH_DISTANCE] / 64]);
                    }
                    DBKeyType memberKey = logWords[i];
                    if (pairIsFree[i] && isAssigned(memberKey) == false) {
                        setAssignedSerial(repKey, threadAssignedKeys[0]);
                        setAssignedSerial(memberKey, threadAssignedKeys[0]);
                        const AcceptedLog &acceptedLog = nodeAcceptedLogs[reps[rep].log];
                        size_t bit = i - acceptedLog.wordOffset + acceptedLog.bitOffset;
                        nodeClusteredLogs[reps[rep].log].bits[bit / 64] |= 1ULL << (bit % 64);
                        memberCount++;
                    }
                }
            }
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                // write the bits before the first word that is not clustered
                ClusteredLog &nodeFile = nodeClusteredLogs[node];
                size_t wordCount = nodeAcceptedLogs[node].readSize / sizeof(uint64_t) / 64 - nodeFile.firstBitsWord;
                if (wordCount > 0 && fwrite(nodeFile.bits.data(), sizeof(uint64_t), wordCount, nodeFile.file) != wordCount) {
                    Debug(Debug::ERROR) << "Cannot write " << clusteredLogPrefix << "." << node << "\n";
                    EXIT(EXIT_FAILURE);
                }
                nodeFile.bits.erase(nodeFile.bits.begin(), nodeFile.bits.begin() + wordCount);
                nodeFile.firstBitsWord += wordCount;
                nodeFile.bytes += wordCount * sizeof(uint64_t);
            }
        }
        MMseqsNode::printProgress(progress, clusteredOrderKey, stepRepOrderKey, orderKeyEnd);
        clusteredOrderKey = stepRepOrderKey;

        // sort and Rice code the keys of the step
        stepKeys.clear();
        for (int thread = 0; thread < threads; thread++) {
            stepKeys.insert(stepKeys.end(), threadAssignedKeys[thread].begin(), threadAssignedKeys[thread].end());
            threadAssignedKeys[thread].clear();
        }
        SORT_PARALLEL(stepKeys.begin(), stepKeys.end());
        unsigned int keyDiffLowBits = stepKeys.empty() ? 0 : getBitWidth(std::max((size_t) 1, (stepKeys.back() - stepKeys.front() + 1) / stepKeys.size())) - 1;
        size_t codeBits = 0;
        for (size_t i = 1; i < stepKeys.size(); i++) {
            codeBits += ((stepKeys[i] - stepKeys[i - 1] - 1) >> keyDiffLowBits) + 1 + keyDiffLowBits;
        }
        size_t headerWords = getAssignedKeysHeaderWords();
        size_t stepStart = assignedKeyWords.size();
        size_t codeWords = (codeBits + 63) / 64;
        assignedKeyWords.resize(stepStart + headerWords + codeWords, 0);
        assignedKeyWords[stepStart] = clusteredOrderKey;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            assignedKeyWords[stepStart + 1 + node] = nodeAcceptedLogs[node].readSize;
        }
        assignedKeyWords[stepStart + 1 + MMseqsNode::count] = stepKeys.size();
        assignedKeyWords[stepStart + 2 + MMseqsNode::count] = stepKeys.empty() ? 0 : stepKeys.front();
        assignedKeyWords[stepStart + 3 + MMseqsNode::count] = keyDiffLowBits;
        assignedKeyWords[stepStart + 4 + MMseqsNode::count] = codeWords;
        KmerCodeWriter codeWrite((char *) (assignedKeyWords.data() + stepStart + headerWords), codeWords * sizeof(uint64_t), 0);
        for (size_t i = 1; i < stepKeys.size(); i++) {
            putRice(codeWrite, stepKeys[i] - stepKeys[i - 1] - 1, keyDiffLowBits);
        }
        putBits(codeWrite, 0, (CHAR_BIT - codeWrite.bitCount) % CHAR_BIT);
        waitWarning.reset();
        gapTimer.reset();
        return STEP_WORKED;
    }

    // finish the last steps and logs, return the clustered members
    size_t finish(std::string &phaseValues) {
        if (assignedKeysWriter.joinable()) {
            assignedKeysWriter.join();
        }
        if (assignedKeyWords.empty() == false) {
            assignedKeyWords.swap(writtenAssignedKeys);
            writeAssignedKeys(orderKeyEnd);
        }
        MMseqsNode::addPhaseSeconds(phaseValues, "clustering the accepted pairs", clusterTimer.getTimediff() - waitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the align nodes", waitSeconds);
        if (fclose(assignedKeysHandle) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << assignedKeysLog << "\n";
            EXIT(EXIT_FAILURE);
        }
        // assigned key is no singleton
        if (assignedBits.empty() == false) {
            FILE *assignedHandle = FileUtil::openAndDelete(assignedBits.c_str(), "w");
            if (fwrite(assigned, sizeof(uint64_t), assignedWords, assignedHandle) != assignedWords) {
                Debug(Debug::ERROR) << "Cannot write " << assignedBits << "\n";
                EXIT(EXIT_FAILURE);
            }
            MMseqsNode::PolledFile assignedBitsProgress(assignedBits + ".progress");
            MMseqsNode::publishLog(assignedHandle, assignedBits, assignedBitsProgress, dbSize, assignedWords * sizeof(uint64_t));
            if (fclose(assignedHandle) != 0) {
                Debug(Debug::ERROR) << "Cannot close " << assignedBits << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
        free(assigned);
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const std::string clusteredLogFile = clusteredLogPrefix + "." + SSTR(node);
            const std::vector<uint64_t> &bits = nodeClusteredLogs[node].bits;
            if (bits.empty() == false && fwrite(bits.data(), sizeof(uint64_t), bits.size(), nodeClusteredLogs[node].file) != bits.size()) {
                Debug(Debug::ERROR) << "Cannot write " << clusteredLogFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            MMseqsNode::PolledFile clusteredProgress(clusteredLogFile + ".progress");
            MMseqsNode::publishLog(nodeClusteredLogs[node].file, clusteredLogFile, clusteredProgress, dbSize, nodeClusteredLogs[node].bytes + bits.size() * sizeof(uint64_t));
            if (fclose(nodeClusteredLogs[node].file) != 0) {
                Debug(Debug::ERROR) << "Cannot close " << clusteredLogFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            delete[] nodeClusteredLogs[node].buffer;
        }
        return memberCount;
    }

    // sleep after an idle step, end of the steps
    std::atomic<unsigned int> idleMicroseconds;
    std::atomic<bool> isDone;

private:
    bool isAssigned(DBKeyType key) const {
        return ((assigned[key / 64] >> (key % 64)) & 1) != 0;
    }

    void setAssigned(DBKeyType key, std::vector<DBKeyType> &keys) {
        if ((__sync_fetch_and_or(&assigned[key / 64], 1ULL << (key % 64)) & (1ULL << (key % 64))) == 0) {
            keys.push_back(key);
        }
    }

    void setAssignedSerial(DBKeyType key, std::vector<DBKeyType> &keys) {
        uint64_t &word = assigned[key / 64];
        uint64_t bit = 1ULL << (key % 64);
        if ((word & bit) == 0) {
            word |= bit;
            keys.push_back(key);
        }
    }

    void writeAssignedKeys(size_t clusteredKey) {
        if (fwrite(writtenAssignedKeys.data(), sizeof(uint64_t), writtenAssignedKeys.size(), assignedKeysHandle) != writtenAssignedKeys.size()) {
            Debug(Debug::ERROR) << "Cannot write " << assignedKeysLog << "\n";
            EXIT(EXIT_FAILURE);
        }
        assignedKeysSize += writtenAssignedKeys.size() * sizeof(uint64_t);
        MMseqsNode::publishLog(assignedKeysHandle, assignedKeysLog, assignedKeysProgress, clusteredKey, assignedKeysSize);
        isWritingAssignedKeys = false;
    }

    void addClusterPair(DBKeyType repKey, DBKeyType memberKey) {
        const ClusterPair pair = { repKey, memberKey };
        unsigned int pageNode = pageAlignNodes[repKey / ASSIGNED_PAGE_KEYS];
        ClusteredLog &nodeFile = nodeClusteredLogs[(pageNode != UINT_MAX) ? pageNode : getKeyAlignNode(seqDbr, repKey)];
        if (fwrite(&pair, sizeof(ClusterPair), 1, nodeFile.file) != 1) {
            Debug(Debug::ERROR) << "Cannot write " << clusteredLogPrefix << "\n";
            EXIT(EXIT_FAILURE);
        }
        nodeFile.bytes += sizeof(ClusterPair);
        memberCount++;
    }

    const int threads;
    const size_t dbSize;
    const DBReader<DBKeyType> &seqDbr;
    const std::vector<unsigned int> &pageAlignNodes;
    const int mode;
    const size_t paddedDbSize;
    const size_t sizeClassCount;
    const std::string accepted;
    const std::string assignedKeysLog;
    const std::string clusteredLogPrefix;
    const std::string assignedBits;
    const bool hasWriterThread;
    const size_t assignedWords;
    const size_t orderKeyEnd;
    uint64_t *assigned;
    std::vector<std::vector<DBKeyType> > threadAssignedKeys;
    std::vector<unsigned char> nodeHasProgress;
    std::vector<size_t> nodeRepOrderKeys;
    std::vector<size_t> nodeLogSize;
    std::vector<ClusteredLog> nodeClusteredLogs;
    std::vector<AcceptedLog> nodeAcceptedLogs;
    std::vector<MMseqsNode::PolledFile> nodeAcceptedProgress;
    MMseqsNode::PolledFile assignedKeysProgress;
    FILE *assignedKeysHandle;
    size_t assignedKeysSize;
    const size_t stepWords;
    const size_t nodeStepWords;
    const size_t logReadBytesPerNode;
    std::vector<uint64_t> logWords;
    std::vector<MergedReps> mergedReps;
    std::vector<AcceptedRep> reps;
    std::vector<unsigned char> pairIsFree;
    std::vector<size_t> repKeyOffsets;
    std::vector<size_t> setReps;
    std::vector<size_t> setSizes;
    std::vector<std::pair<DBKeyType, size_t> > keyReps;
    std::vector<unsigned char> repSharesKey;
    std::vector<size_t> sharingReps;
    std::vector<uint64_t> assignedKeyWords;
    std::vector<uint64_t> writtenAssignedKeys;
    std::thread assignedKeysWriter;
    std::atomic<bool> isWritingAssignedKeys;
    std::vector<DBKeyType> stepKeys;
    size_t memberCount;
    // --cluster-mode 0: open sets in a heap, sets of two as pairs
    std::vector<ClusterPair> pairSets;
    SetCoverPairFile pairFile;
    SetCoverSmallSetFiles smallSetFiles;
    SetCoverLargeSetFiles largeSetFiles;
    double waitSeconds;
    Timer clusterTimer;
    MMseqsNode::WaitWarning waitWarning;
    size_t clusteredOrderKey;
    // since the last step end or idle poll
    bool isIdle;
    Timer gapTimer;
    Debug::Progress progress;
};

// --local-tmp-fallback 0 stops the run, else read blocks from the shared folder
static void reportServedCopyProblem(const Parameters &par, const std::string &problem, std::string &phaseValues) {
    if (par.localTmpFallback == false) {
        Debug(Debug::ERROR) << "Cannot copy the blocks of this node to --local-tmp " << par.localTmp << ": " << problem << ". Please choose another --local-tmp or set --local-tmp-fallback 1 to read them from the database instead.\n";
        EXIT(EXIT_FAILURE);
    }
    MMseqsNode::reportUnusableLocalTmp("Cannot copy the blocks of this node to --local-tmp " + par.localTmp + ": " + problem);
    MMseqsNode::addPhaseValue(phaseValues, "Local tmp", "not used, " + problem);
}

// served blocks of an align node in key order, copies NULL if none
struct ServedBlockCopy {
    ServedBlockCopy() : maxBytes(0), openFiles(0), copies(NULL) {}
    std::string localDir;
    std::vector<std::string> dataFiles;
    std::vector<size_t> blocks;
    // bytes before each block, last one is the served size
    std::vector<size_t> blockOffsets;
    std::vector<size_t> blockFiles;
    std::vector<size_t> blockBegins;
    size_t maxBytes;
    // open files the node needs besides the copy
    size_t openFiles;
    // problem to report if no copy fits, empty once the cluster part is copied
    std::string spaceProblem;
    std::vector<std::string> *copies;
};

static void alignRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, TargetSplitReader &prefilterReader,
                            size_t memoryLimit, size_t masterMemoryLimit, size_t servedFileEntrySize, size_t representativeResidues,
                            size_t paddedDbSize, size_t sizeClassCount, const ClusterMemberLookup *memberLookup, ClusterMemberExchange *memberExchange,
                            size_t memberKeyBudget, DBReader<DBKeyType> *memberDbr, int ioThreadCount, AcceptedPairClustering *clustering, ServedBlockCopy &servedBlockCopy,
                            std::string &phaseValues) {
    Timer startTimer;
    unsigned int nodes = MMseqsNode::count;
    unsigned int rank = MMseqsNode::rank;
    size_t firstAlignNode = MMseqsNode::getFirstAlignNode();
    size_t alignNodeCount = nodes - firstAlignNode;
    bool isAligningNode = rank >= firstAlignNode;
    const std::string assignedKeysLog = par.db3 + "_assigned_keys";
    const std::string requests = par.db3 + "_sequence_requests";
    const std::string served = par.db3 + "_served_sequences";
    size_t dbSize = seqDbr.getSize();
    const size_t orderKeyEnd = (sizeClassCount - 1) * paddedDbSize + dbSize;
    const size_t maxSeqLen = seqDbr.getMaxSeqLen();
    size_t maxMemberSeqLen = (memberDbr != NULL) ? std::max(maxSeqLen, memberDbr->getMaxSeqLen()) : maxSeqLen;
    size_t memberSplitSize = memoryLimit / MEMBER_SPLIT_MEMORY_DIVISOR;
    size_t pageSize = Util::getPageSize();
    AlignmentScoring scoring(par, representativeResidues);
    unsigned int swMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);
    const std::string acceptedFile = par.db3 + "_accepted." + SSTR(rank);
    FILE *acceptedHandle = FileUtil::openAndDelete(acceptedFile.c_str(), "w");
    MMseqsNode::PolledFile acceptedProgress(acceptedFile + ".progress");
    size_t acceptedSize = 0;
    std::vector<uint64_t> acceptedWords;
    DBKeyType loggedRepOrderKey = DB_KEY_INVALID;
    size_t publishCount = 0;
    double publishCostSeconds = 0;
    std::atomic<double> publishPeriod(MIN_PUBLISH_SECONDS);
    std::atomic<size_t> publishedDoneRepKey(0);
    const auto publishAccepted = [&](size_t repKey, size_t bytes) {
        Timer publishTimer;
        publishCount++;
        MMseqsNode::publishLog(acceptedHandle, acceptedFile, acceptedProgress, repKey, bytes);
        publishedDoneRepKey.store(repKey);
        publishCostSeconds += publishTimer.getTimediff();
        publishPeriod.store(std::min<double>(PUBLISH_SECONDS, std::max(MIN_PUBLISH_SECONDS, LOG_COST_FACTOR * publishCostSeconds / publishCount)));
    };
    // publishes member checks while the next fetch aligns
    std::thread publishThread;

    // assigned bits of the served and aligned keys, by page
    std::vector<uint64_t *> assignedPages(isAligningNode ? (dbSize + ASSIGNED_PAGE_KEYS - 1) / ASSIGNED_PAGE_KEYS : 0, NULL);
    const auto isAssigned = [&](size_t orderKey) {
        size_t key = orderKey % paddedDbSize;
        const uint64_t *page = __atomic_load_n(&assignedPages[key / ASSIGNED_PAGE_KEYS], __ATOMIC_ACQUIRE);
        return page != NULL && ((__atomic_load_n(&page[key % ASSIGNED_PAGE_KEYS / 64], __ATOMIC_RELAXED) >> (key % 64)) & 1) != 0;
    };
    // master drops pairs with assigned keys anyway
    const auto logAcceptedPairs = [&](const ClusterPair *pairs, size_t pairCount) {
        acceptedWords.clear();
        for (size_t i = 0; i < pairCount; i++) {
            if (isAssigned(pairs[i].repKey) || isAssigned(pairs[i].memberKey)) {
                continue;
            }
            if (pairs[i].repKey != loggedRepOrderKey) {
                loggedRepOrderKey = pairs[i].repKey;
                acceptedWords.push_back(loggedRepOrderKey | ACCEPTED_REP_BIT);
            }
            acceptedWords.push_back(pairs[i].memberKey);
        }
        if (acceptedWords.empty() == false && fwrite(acceptedWords.data(), sizeof(uint64_t), acceptedWords.size(), acceptedHandle) != acceptedWords.size()) {
            Debug(Debug::ERROR) << "Cannot write " << acceptedFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        acceptedSize += acceptedWords.size() * sizeof(uint64_t);
    };

    size_t clusteredOrderKey = 0;
    size_t clusteredAcceptedSize = 0;
    size_t headerWords = getAssignedKeysHeaderWords();
    size_t assignedKeysSize = 0;
    std::vector<uint64_t> assignedKeyWords;
    MMseqsNode::PolledFile assignedKeysProgress(assignedKeysLog + ".progress");
    // without wait an unreadable step waits for the next call
    const auto readAssignedKeys = [&](bool wait) {
        size_t publishedKey;
        size_t publishedSize;
        if (MMseqsNode::readLogProgress(assignedKeysProgress, publishedKey, publishedSize) == false) {
            return;
        }
        if (publishedSize <= assignedKeysSize) {
            return;
        }
        // published steps in one read
        assignedKeyWords.resize((publishedSize - assignedKeysSize) / sizeof(uint64_t));
        if (MMseqsNode::readLog(assignedKeysLog, assignedKeysSize, publishedSize, (char *) assignedKeyWords.data(), wait) == false) {
            return;
        }
        for (const uint64_t *step = assignedKeyWords.data(); assignedKeysSize < publishedSize;) {
            size_t stepClusteredKey = step[0];
            size_t stepClusteredSize = step[1 + MMseqsNode::rank];
            size_t keyCount = step[1 + MMseqsNode::count];
            size_t key = step[2 + MMseqsNode::count];
            unsigned int keyDiffLowBits = step[3 + MMseqsNode::count];
            size_t codeWords = step[4 + MMseqsNode::count];
            step += headerWords;
            KmerCodeReader codeRead = { (char *) step, 0, 0, 0, -1, assignedKeysLog.c_str(), codeWords * sizeof(uint64_t), 0, codeWords * sizeof(uint64_t) };
            // only the thread that holds assignedKeysMutex adds pages
            for (size_t i = 0; i < keyCount; i++) {
                key += (i > 0) ? getRice(codeRead, keyDiffLowBits) + 1 : 0;
                uint64_t *page = __atomic_load_n(&assignedPages[key / ASSIGNED_PAGE_KEYS], __ATOMIC_RELAXED);
                if (page == NULL) {
                    page = new(std::nothrow) uint64_t[ASSIGNED_PAGE_KEYS / 64]();
                    Util::checkAllocation(page, "Can not allocate assigned page memory in align2clust");
                    __atomic_store_n(&assignedPages[key / ASSIGNED_PAGE_KEYS], page, __ATOMIC_RELEASE);
                }
                __atomic_fetch_or(&page[key % ASSIGNED_PAGE_KEYS / 64], 1ULL << (key % 64), __ATOMIC_RELAXED);
            }
            step += codeWords;
            assignedKeysSize += (headerWords + codeWords) * sizeof(uint64_t);
            // receiving pass reads it for the gate slack
            __atomic_store_n(&clusteredOrderKey, stepClusteredKey, __ATOMIC_RELAXED);
            clusteredAcceptedSize = stepClusteredSize;
        }
    };

    // buffer n is filled once the fetches use buffer n - 1
    std::vector<RepTargetsRecord> prefilterRecords[3];
    std::vector<DBKeyType> prefilterTargets[3];
    size_t prefilterRecordPos[3] = { 0, 0, 0 };
    size_t prefilterBuffer = 0;
    std::mutex prefilterBufferMutex;
    std::condition_variable prefilterBufferCondition;
    size_t filledPrefilterBufferCount = 0;
    size_t usedPrefilterBufferCount = 0;
    double prefilterWaitSeconds = 0;
    std::vector<std::vector<DBKeyType> > threadKeys(par.threads);
    // key and its record entry, SIZE_MAX for members
    std::vector<std::vector<std::pair<DBKeyType, size_t> > > threadRecordKeys(par.threads);
    std::vector<std::pair<DBKeyType, size_t> > recordKeys;
    std::vector<unsigned int> keyFetchSources;
    std::vector<size_t> threadSourceOffsets;
    // accepted pairs waiting for their member checks, in log order
    std::vector<ClusterPair> pendingPairs;
    std::vector<size_t> pendingPairSize;
    std::vector<size_t> pendingMemberOffsets(1, 0);
    std::vector<DBKeyType> pendingMembers;
    // one lane flag set per pair, its parts align members alike
    std::vector<unsigned char> pendingMemberIsLane;
    // pair too large for one fetch goes in parts that pass only together
    std::vector<unsigned char> pendingPairIsLastPart;
    bool isPendingPairFailing = false;
    bool isPendingPairOpen = false;
    ClusterPair openPendingPair = { 0, 0 };
    // accepted pairs waiting for room in pendingMembers, in log order
    std::vector<ClusterPair> acceptedPairs;
    std::vector<size_t> acceptedEntryOffsets;
    std::vector<size_t> acceptedEntryBytes;
    std::vector<DBKeyType> lookupKeys;
    std::vector<size_t> lookupEntryOffsets;
    std::vector<size_t> lookupEntryBytes;
    std::vector<std::pair<DBKeyType, size_t> > lookupKeyPairs;
    std::vector<size_t> lookupMemberOffsets;
    std::vector<DBKeyType> lookupMembers;
    // first accepted pair, read in slices if too large for pendingMembers
    size_t slicedConsumedBytes = 0;
    bool isSlicingPair = false;
    size_t slicedRepSeqLen = 0;
    std::vector<size_t> slicedScalarPositions;
    size_t slicedMemberPosition = 0;

    // coverable bytes of a rep in this node's blocks
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    size_t rangeCount = seqDbr.getIndexSize();
    const std::vector<size_t> &blockKeyOffsets = seqDbr.getBlockKeyOffsets();
    size_t sequenceSize = 0;
    std::vector<size_t> nodeRangeByteOffsets(rangeCount + 1, 0);
    size_t nodeBlockCount = 0;
    for (size_t range = 0, block = 0; range < rangeCount; range++) {
        size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
        sequenceSize += (keyEnd - index[range].id) * index[range].length;
        nodeRangeByteOffsets[range + 1] = nodeRangeByteOffsets[range];
        for (size_t key = index[range].id; key < keyEnd; key = std::min(keyEnd, blockKeyOffsets[block + 1])) {
            for (; blockKeyOffsets[block + 1] <= key; block++) {}
            if (getBlockAlignNode(block) == rank) {
                nodeRangeByteOffsets[range + 1] += (std::min(keyEnd, blockKeyOffsets[block + 1]) - key) * index[range].length;
                nodeBlockCount++;
            }
        }
    }

    double coverableBytesSum = 0;
    size_t maxCoverableSize = 0;
    for (size_t range = 0; range < rangeCount; range++) {
        float repLength = static_cast<float>(index[range].length - 2);
        const auto isCovered = [&](const DBReader<DBKeyType>::Index &coveredRange) {
            return Util::canBeCovered(par.covThr, par.covMode, repLength, static_cast<float>(coveredRange.length - 2));
        };
        // ranges descend in length, a rep covers an interval of lengths
        size_t coveredFirst = std::partition_point(index, index + range, [&](const DBReader<DBKeyType>::Index &coveredRange) { return isCovered(coveredRange) == false; }) - index;
        size_t coveredEnd = std::partition_point(index + range, index + rangeCount, isCovered) - index;
        size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
        coverableBytesSum += static_cast<double>(keyEnd - index[range].id) * (nodeRangeByteOffsets[coveredEnd] - nodeRangeByteOffsets[coveredFirst]);
        maxCoverableSize = std::max(maxCoverableSize, nodeRangeByteOffsets[coveredEnd] - nodeRangeByteOffsets[coveredFirst]);
    }
    double coverableSize = coverableBytesSum / std::max((size_t) 1, dbSize);
    size_t averageEntrySize = std::max<size_t>(1, sequenceSize / std::max<size_t>(1, dbSize));
    size_t maxServedKeys = std::max<size_t>(1, servedFileEntrySize / FETCHED_KEY_SIZE);
    size_t maxServedFileEntrySize = servedFileEntrySize + SERVICE_POLL_NODES * (maxSeqLen + 2);
    size_t maxServedFileSize = SERVICE_POLL_NODES * ((maxServedKeys + 7) / 8) + getPackedEntrySize(maxServedFileEntrySize) + maxServedFileEntrySize;
    // read directly once a miss costs less than two packed copies
    const auto isCacheTooSmall = [&](size_t cacheSize) {
        return alignNodeCount > 1 && static_cast<double>(cacheSize) * pageSize <= 2.0 * getPackedEntrySize(averageEntrySize) * coverableSize;
    };
    // largest served block while the I/O threads copy
    size_t blockCopyBufferSize = 0;
    // memory besides the block cache
    const auto getReservedMemory = [&](size_t fetchSize) {
        return ((memberLookup != NULL) ? memberLookup->memorySize() + (par.threads + 1) * memberLookup->getEntryReaderSize() : 0) + 3 * 2 * (memoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR) + blockCopyBufferSize
               + ((memberDbr != NULL) ? 2 * memberSplitSize : 0)
               + 2 * std::max(fetchSize, prefilterReader.maxRecordSize) + fetchSize
               + ((isAligningNode && alignNodeCount > 1) ? 3 * maxServedFileSize + 6 * maxServedKeys * sizeof(size_t) : 0);
    };
    // largest fetch that leaves a block cache for the coverable sequences
    bool canCache = isCacheTooSmall(memoryLimit - std::min(memoryLimit, getReservedMemory(0))) == false;
    size_t servedSequenceSize = nodeRangeByteOffsets[rangeCount];
    // fetch holds a rep and a member of the largest length
    size_t minFetchSize = 2 * (maxMemberSeqLen + 2 + FETCHED_KEY_SIZE);
    size_t maxFetchSize = std::max(memoryLimit / 8, minFetchSize);
    while (maxFetchSize / 2 >= minFetchSize && (getReservedMemory(maxFetchSize) > memoryLimit || (canCache && isCacheTooSmall(memoryLimit - getReservedMemory(maxFetchSize)))
                                 || (alignNodeCount > 1 && maxFetchSize > prefilterReader.maxRecordSize && memoryLimit - std::min(memoryLimit, getReservedMemory(maxFetchSize)) < servedSequenceSize))) {
        maxFetchSize /= 2;
    }

    // block read is rounded to pages at both ends
    size_t nodeCacheSize = servedSequenceSize + 2 * pageSize * nodeBlockCount;
    // copy to --local-tmp what the cache cannot hold, if it holds the copy buffer
    size_t maxServedBlockSize = 0;
    for (size_t i = 0; i < servedBlockCopy.blocks.size(); i++) {
        maxServedBlockSize = std::max(maxServedBlockSize, servedBlockCopy.blockOffsets[i + 1] - servedBlockCopy.blockOffsets[i]);
    }
    size_t uncopiedCacheSize = memoryLimit - std::min(memoryLimit, getReservedMemory(maxFetchSize));
    bool isCopyWanted = servedBlockCopy.copies != NULL && uncopiedCacheSize < nodeCacheSize && uncopiedCacheSize >= maxServedBlockSize;
    bool isCopying = isCopyWanted && servedBlockCopy.maxBytes >= servedBlockCopy.blockOffsets[1];
    blockCopyBufferSize = (isCopying && ioThreadCount > 0) ? maxServedBlockSize : 0;
    if (getReservedMemory(maxFetchSize) > memoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to align. Memory limit: " << memoryLimit << " bytes, the buffers besides the block cache need: " << getReservedMemory(maxFetchSize)
                            << " bytes. Increase --split-memory-limit\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t blockCacheSize = memoryLimit - getReservedMemory(maxFetchSize);
    seqDbr.setBlockCacheSize(blockCacheSize);

    // reserve the blocks that fit, I/O threads copy them while aligning, else now
    size_t localBlockSize = 0;
    size_t localBlockCount = 0;
    size_t reservedBlockCount = 0;
    size_t copiedBlockCount = 0;
    size_t firstBlockCopy = 0;
    std::vector<int> blockSourceFds;
    std::vector<int> blockCopyFds;
    std::vector<char> blockCopyBuffer;
    if (servedBlockCopy.copies != NULL) {
        const std::vector<size_t> &blockOffsets = servedBlockCopy.blockOffsets;
        std::vector<std::string> &copies = *servedBlockCopy.copies;
        const std::vector<std::string> &dataFiles = servedBlockCopy.dataFiles;
        if (isCopying) {
            std::vector<std::vector<std::pair<size_t, size_t> > > fileRanges(dataFiles.size());
            for (size_t i = 0; i < servedBlockCopy.blocks.size(); i++) {
                std::vector<std::pair<size_t, size_t> > &ranges = fileRanges[servedBlockCopy.blockFiles[i]];
                const size_t begin = servedBlockCopy.blockBegins[i];
                const size_t end = begin + blockOffsets[i + 1] - blockOffsets[i];
                if (ranges.empty() == false && ranges.back().second == begin) {
                    ranges.back().second = end;
                } else {
                    ranges.emplace_back(begin, end);
                }
            }
            // two fds per data file for the copy, two more for I/O threads
            FileUtil::checkRlimitNoFile(servedBlockCopy.openFiles + ((ioThreadCount > 0) ? 4 : 2) * dataFiles.size());
            Timer copyTimer;
            firstBlockCopy = copies.size();
            size_t reservedSize = 0;
            const std::string localSeqDb = FileUtil::copyToLocalDir(par.db1, servedBlockCopy.localDir, par.threads - ioThreadCount, copies, reservedSize, fileRanges, servedBlockCopy.maxBytes,
                                                                    ioThreadCount > 0);
            reservedBlockCount = std::upper_bound(blockOffsets.begin(), blockOffsets.end(), reservedSize) - blockOffsets.begin() - 1;
            if (reservedBlockCount == 0) {
                for (size_t i = firstBlockCopy; i < copies.size(); i++) {
                    FileUtil::remove(copies[i].c_str());
                }
                copies.resize(firstBlockCopy);
                if (copies.empty()) {
                    rmdir(servedBlockCopy.localDir.c_str());
                }
            } else {
                copiedBlockCount = (ioThreadCount > 0) ? 0 : reservedBlockCount;
                std::vector<char> localBlocks(blockKeyOffsets.size() - 1, 0);
                blockSourceFds.assign(dataFiles.size(), -1);
                blockCopyFds.assign(dataFiles.size(), -1);
                for (size_t i = 0; i < reservedBlockCount; i++) {
                    size_t block = servedBlockCopy.blocks[i];
                    localBlocks[block] = i < copiedBlockCount;
                    // range of a block counts like in nodeBlockCount
                    localBlockCount += seqDbr.findIdRange(blockKeyOffsets[block + 1] - 1) - seqDbr.findIdRange(blockKeyOffsets[block]) + 1;
                    size_t file = servedBlockCopy.blockFiles[i];
                    if (ioThreadCount > 0 && blockCopyFds[file] < 0) {
                        blockSourceFds[file] = FileUtil::openSharedFd(dataFiles[file], O_RDONLY | O_CLOEXEC);
                        blockCopyFds[file] = FileUtil::openSharedFd(copies[firstBlockCopy + file], O_WRONLY | O_CLOEXEC);
                    }
                }
                seqDbr.setLocalBlocks(localSeqDb, localBlocks);
                blockCopyBuffer.resize(blockCopyBufferSize);
            }
            MMseqsNode::addPhaseTime(phaseValues, (ioThreadCount > 0) ? "reserving room for the blocks of this node in --local-tmp" : "copying the blocks of this node to --local-tmp", copyTimer);
        }
        localBlockSize = blockOffsets[reservedBlockCount];
        if (isCopyWanted && reservedBlockCount == 0 && servedBlockCopy.spaceProblem.empty() == false) {
            reportServedCopyProblem(par, servedBlockCopy.spaceProblem, phaseValues);
        } else {
            MMseqsNode::addPhaseValue(phaseValues, "Local tmp", ((copiedBlockCount < reservedBlockCount) ? "reserves " : "holds ") + MMseqsNode::formatBytes(localBlockSize) + " of " + MMseqsNode::formatBytes(blockOffsets.back())
                                      + " of sequences in " + SSTR(reservedBlockCount) + " of " + SSTR(servedBlockCopy.blocks.size()) + " blocks"
                                      + (par.filterCluDBFile.empty() ? "" : (servedBlockCopy.spaceProblem.empty() ? " and the cluster part" : ", the cluster part is read from the shared temporary folder"))
                                      + ((uncopiedCacheSize >= nodeCacheSize) ? ", the block cache holds the sequences" : ""));
        }
    }

    size_t maxPendingPairSize = std::min(memberSplitSize, maxFetchSize);
    // blocks in --local-tmp count like the block cache
    bool isReadingDirectly = isCacheTooSmall(blockCacheSize + localBlockSize);
    // local blocks leave cache room for shared ones, reads round to pages
    size_t sharedBlockSize = servedSequenceSize - localBlockSize + 2 * pageSize * (nodeBlockCount - localBlockCount);
    seqDbr.setLocalBlockCacheSize(blockCacheSize - std::min(blockCacheSize, sharedBlockSize));
    if (isAligningNode) {
        MMseqsNode::addPhaseValue(phaseValues, "Block cache", MMseqsNode::formatBytes(blockCacheSize) + " for " + MMseqsNode::formatBytes(servedSequenceSize) + " of sequences on this node ("
                                 + MMseqsNode::formatBytes(localBlockSize) + " in --local-tmp), about " + MMseqsNode::formatBytes(static_cast<size_t>(coverableSize))
                                 + " in one representative's length range" + (isReadingDirectly ? ", the sequences are read directly" : ""));
    }
    if (isAligningNode && blockCacheSize + localBlockSize < servedSequenceSize) {
        size_t cacheCapacity = std::max<size_t>(1, blockCacheSize + servedBlockCopy.maxBytes);
        Debug(Debug::WARNING) << "Node " << rank << " holds " << MMseqsNode::formatBytes(servedSequenceSize) << " (" << servedSequenceSize << " bytes) of sequences, its block cache in memory has "
                              << MMseqsNode::formatBytes(blockCacheSize) << " (" << blockCacheSize << " bytes) and --local-tmp holds " << MMseqsNode::formatBytes(localBlockSize) << " ("
                              << localBlockSize << " bytes) of them, so the other sequences are read one at a time, which is slower. About "
                              << (alignNodeCount * servedSequenceSize + cacheCapacity - 1) / cacheCapacity << " align nodes would hold them all; use more nodes or increase --split-memory-limit\n";
    }

    // small first fetch (1/64 of the largest, at most 64 MiB), then doubling
    size_t fetchBytesLimit = std::min<size_t>(maxFetchSize / 64, 64 * 1024 * 1024);
    size_t maxFetchBytesLimit = maxFetchSize;
    // fetches in flight share the two fetches of getReservedMemory
    size_t fetchBudget = 2 * std::max(maxFetchSize, prefilterReader.maxRecordSize);
    std::vector<FetchedSequences> fetchSlots(1 + fetchBudget / std::max(fetchBytesLimit, minFetchSize));
    // take the last released slot first, few slots get buffers
    std::vector<FetchedSequences *> idleFetches;
    for (size_t i = fetchSlots.size(); i > 0; i--) {
        fetchSlots[i - 1].fetchSourceKeyOffsets.assign(nodes + 2, 0);
        idleFetches.push_back(&fetchSlots[i - 1]);
    }
    std::deque<FetchedSequences *> fetchesInFlight;
    const auto getFetchSource = [&](size_t key, size_t &block) -> size_t {
        if (isReadingDirectly) {
            return rank;
        }
        return getKeyAlignNode(seqDbr, key, &block);
    };
    const auto getFetchedEntry = [&](const FetchedSequences &sequences, size_t key, size_t *seqLen = NULL, FetchedKeyCursor *cursor = NULL) -> const char * {
        const char *data = findFetchedKey(sequences, nodes, key, seqLen, (cursor != NULL) ? &cursor->positions[nodes] : NULL);
        if (data != NULL) {
            return data;
        }
        size_t block = SIZE_MAX;
        size_t fetchSource = getFetchSource(key, (cursor != NULL) ? cursor->block : block);
        return findFetchedKey(sequences, fetchSource, key, seqLen, (cursor != NULL) ? &cursor->positions[fetchSource] : NULL);
    };
    const auto getFetchedSequence = [&](const FetchedSequences &sequences, size_t key, size_t *seqLen = NULL, FetchedKeyCursor *cursor = NULL) -> const char * {
        const char *data = getFetchedEntry(sequences, key, seqLen, cursor);
        // assigned keys are served as zeros
        return (data != NULL && data[0] != '\0') ? data : NULL;
    };

    std::mutex serviceMutex;
    std::condition_variable serviceCondition;
    // I/O threads sleep until it changes
    uint64_t ioWork = 0;
    std::atomic<double> alignedCellsPerSecond(0);
    // aligning seconds in flight: shortest of the last three round trips
    std::atomic<double> fetchAheadSeconds(MIN_FETCH_AHEAD_SECONDS);
    double roundTrips[3] = { 0, 0, 0 };
    std::mutex prefilterFillMutex;
    std::atomic<bool> isPrefilterDone(false);
    // waiting main thread gets the fill mutex but for read steps
    std::atomic<bool> isPrefilterWaited(false);
    std::condition_variable prefilterReadCondition;
    // fill buffer n once n - 1 is in use, a waiting main thread finishes it
    // prefilter reader issues and waits for every pref read
    const auto fillNextPrefilterBuffer = [&](bool isWaiting, bool isPrefilterReader) {
        std::unique_lock<std::mutex> taskLock(prefilterFillMutex, std::defer_lock);
        if (isWaiting) {
            isPrefilterWaited = true;
            taskLock.lock();
            isPrefilterWaited = false;
        } else if ((isPrefilterWaited && isPrefilterReader == false) || taskLock.try_lock() == false) {
            return false;
        } else if (prefilterReader.isReadStepNext() ? isPrefilterReader == false : isPrefilterWaited.load()) {
            // read steps for the prefilter reader, others for a waiting main thread
            return false;
        }
        if (isPrefilterDone) {
            return false;
        }
        size_t next;
        {
            std::lock_guard<std::mutex> lock(prefilterBufferMutex);
            if (usedPrefilterBufferCount < filledPrefilterBufferCount) {
                return false;
            }
            next = filledPrefilterBufferCount;
        }
        bool isFilled = false;
        do {
            // waiting main thread hands a read step to the prefilter reader
            if (isPrefilterReader == false && prefilterReader.isReadStepNext()) {
                isPrefilterWaited = true;
                {
                    std::lock_guard<std::mutex> lock(serviceMutex);
                    ioWork++;
                }
                serviceCondition.notify_all();
                prefilterReadCondition.wait(taskLock, [&]() { return prefilterReader.isReadStepNext() == false; });
                isPrefilterWaited = false;
            }
            // covers the fetches in flight
            isFilled = prefilterReader.fillPrefilterBuffer(prefilterRecords[next % 3], prefilterTargets[next % 3], maxFetchSize, static_cast<size_t>(2.0 * fetchAheadSeconds.load() * alignedCellsPerSecond.load()), isWaiting == false);
        } while (isWaiting && isFilled == false);
        prefilterReadCondition.notify_all();
        bool isHanded = isPrefilterReader == false && prefilterReader.isReadStepNext();
        if (isFilled) {
            bool isLast = prefilterRecords[next % 3].empty();
            {
                std::lock_guard<std::mutex> lock(prefilterBufferMutex);
                filledPrefilterBufferCount = next + 1;
            }
            isPrefilterDone = isLast;
            prefilterBufferCondition.notify_all();
        }
        // prefilter reader takes the read step once the fill mutex is free
        taskLock.unlock();
        if (isHanded) {
            {
                std::lock_guard<std::mutex> lock(serviceMutex);
                ioWork++;
            }
            serviceCondition.notify_all();
        }
        return true;
    };

    // take records while they fit, one at least, assigned reps take no bytes
    const auto selectRecordKeys = [&](FetchedSequences &sequences, const FetchedSequences &previous) {
        // empty prefilter buffer is the last one
        if (prefilterRecordPos[prefilterBuffer] == prefilterRecords[prefilterBuffer].size() && prefilterRecords[prefilterBuffer].empty() == false) {
            Timer waitTimer;
            // waiting thread finishes the next buffer with its idle team
            fillNextPrefilterBuffer(true, ioThreadCount == 0);
            std::unique_lock<std::mutex> lock(prefilterBufferMutex);
            prefilterBufferCondition.wait(lock, [&]() { return filledPrefilterBufferCount > usedPrefilterBufferCount; });
            prefilterBuffer = usedPrefilterBufferCount++ % 3;
            lock.unlock();
            prefilterBufferCondition.notify_all();
            {
                std::lock_guard<std::mutex> serviceLock(serviceMutex);
                ioWork++;
            }
            serviceCondition.notify_all();
            prefilterWaitSeconds += waitTimer.getTimediff();
            prefilterRecordPos[prefilterBuffer] = 0;
        }
        sequences.prefilterBuffer = prefilterBuffer;
        sequences.prefilterBufferNumber = usedPrefilterBufferCount - 1;
        sequences.begin = prefilterRecordPos[prefilterBuffer];
        sequences.end = sequences.begin;
        size_t aheadPairs = previous.pendingPairCount;
        size_t aheadTargets = previous.targetCount;
        bool hasRecordsAhead = previous.end > previous.begin;
        for (size_t i = 0; i < fetchesInFlight.size(); i++) {
            aheadPairs += fetchesInFlight[i]->pendingPairCount;
            aheadTargets += fetchesInFlight[i]->targetCount;
            hasRecordsAhead |= fetchesInFlight[i]->end > fetchesInFlight[i]->begin;
        }
        // pending pairs first, at least one
        size_t fetchSize = 0;
        sequences.pendingPairCount = 0;
        for (size_t i = aheadPairs; i < pendingPairs.size() && (sequences.pendingPairCount == 0 || fetchSize + pendingPairSize[i] <= fetchBytesLimit); i++) {
            fetchSize += pendingPairSize[i];
            sequences.pendingPairCount++;
        }
        // accepted pairs follow the pending pairs in log order
        sequences.firstPendingRepKey = (sequences.pendingPairCount > 0) ? static_cast<size_t>(pendingPairs[aheadPairs].repKey)
                                       : (acceptedPairs.empty() ? orderKeyEnd : static_cast<size_t>(acceptedPairs[0].repKey));
        // accepted pairs and targets ahead fit memberKeyBudget, or one record alone
        size_t acceptablePairCount = acceptedPairs.size() + aheadTargets;
        sequences.targetCount = 0;
        size_t pendingFetchSize = fetchSize;
        for (; sequences.end < prefilterRecords[prefilterBuffer].size(); sequences.end++) {
            const RepTargetsRecord &record = prefilterRecords[prefilterBuffer][sequences.end];
            size_t recordSize = isAssigned(record.repOrderKeyFirstTarget.repKey) ? 0 : record.readSize;
            if ((fetchSize > pendingFetchSize && fetchSize + recordSize > fetchBytesLimit)
                || (memberDbr != NULL && acceptablePairCount > 0 && acceptablePairCount + record.targetCount > memberKeyBudget)) {
                break;
            }
            fetchSize += recordSize;
            acceptablePairCount += record.targetCount;
            sequences.targetCount += record.targetCount;
        }
        sequences.plannedSize = fetchSize;
        prefilterRecordPos[prefilterBuffer] = sequences.end;
        // fetches ahead may still add pending pairs
        sequences.isLast = sequences.end == sequences.begin && sequences.pendingPairCount == 0 && (memberDbr == NULL || hasRecordsAhead == false) && acceptedPairs.empty();
        fetchBytesLimit = std::min(2 * fetchBytesLimit, maxFetchBytesLimit);
        // assigned reps get no entries
        sequences.recordEntryOffsets.resize(sequences.end - sequences.begin + 1);
        sequences.recordEntryOffsets[0] = 0;
        for (size_t i = sequences.begin; i < sequences.end; i++) {
            const RepTargetsRecord &record = prefilterRecords[sequences.prefilterBuffer][i];
            sequences.recordEntryOffsets[i - sequences.begin + 1] = sequences.recordEntryOffsets[i - sequences.begin] + (isAssigned(record.repOrderKeyFirstTarget.repKey) ? 0 : record.targetCount + 1);
        }
        sequences.recordEntries.assign(sequences.recordEntryOffsets.back(), SIZE_MAX);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            threadRecordKeys[thread_idx].clear();
#pragma omp for schedule(dynamic, 100)
            for (size_t i = sequences.begin; i < sequences.end; i++) {
                const RepTargetsRecord &record = prefilterRecords[sequences.prefilterBuffer][i];
                size_t repKey = record.repOrderKeyFirstTarget.repKey % paddedDbSize;
                const char *previousRep = getFetchedEntry(previous, repKey);
                if (isAssigned(record.repOrderKeyFirstTarget.repKey) || (previousRep != NULL && previousRep[0] == '\0')) {
                    continue;
                }
                size_t targetsBegin = threadRecordKeys[thread_idx].size();
                size_t entry = sequences.recordEntryOffsets[i - sequences.begin];
                const DBKeyType *targets = prefilterTargets[sequences.prefilterBuffer].data() + record.targetOffset;
                for (unsigned int j = 0; j < record.targetCount; j++) {
                    if (isAssigned(targets[j]) == false) {
                        threadRecordKeys[thread_idx].emplace_back(targets[j], entry + 1 + j);
                    }
                }
                // rep without targets left is not fetched
                if (threadRecordKeys[thread_idx].size() > targetsBegin) {
                    threadRecordKeys[thread_idx].emplace_back(repKey, entry);
                }
            }
            // members share the keys and data files of the sequence database
#pragma omp for schedule(dynamic, 100)
            for (size_t i = aheadPairs; i < aheadPairs + sequences.pendingPairCount; i++) {
                if (isAssigned(pendingPairs[i].repKey) == false && isAssigned(pendingPairs[i].memberKey) == false) {
                    threadRecordKeys[thread_idx].emplace_back(pendingPairs[i].repKey % paddedDbSize, SIZE_MAX);
                    for (size_t member = pendingMemberOffsets[i]; member < pendingMemberOffsets[i + 1]; member++) {
                        threadRecordKeys[thread_idx].emplace_back(pendingMembers[member], SIZE_MAX);
                    }
                }
            }
        }
        recordKeys.clear();
        for (size_t thread = 0; thread < threadRecordKeys.size(); thread++) {
            recordKeys.insert(recordKeys.end(), threadRecordKeys[thread].begin(), threadRecordKeys[thread].end());
        }
        SORT_PARALLEL(recordKeys.begin(), recordKeys.end());
        // record entries hold the first pair of their key until the keys are placed
        keyFetchSources.resize(recordKeys.size());
        // sort keys by fetch source, assigned keys go last and are dropped
        threadSourceOffsets.assign((nodes + 2) * par.threads + 1, 0);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<size_t> fetchSourceOffsets(nodes + 2, 0);
            FetchedKeyCursor keyCursor(nodes + 1);
#pragma omp for schedule(static)
            for (size_t i = 0; i < recordKeys.size(); i++) {
                DBKeyType key = recordKeys[i].first;
                if (i > 0 && recordKeys[i - 1].first == key) {
                    continue;
                }
                for (size_t j = i; j < recordKeys.size() && recordKeys[j].first == key; j++) {
                    if (recordKeys[j].second != SIZE_MAX) {
                        sequences.recordEntries[recordKeys[j].second] = i;
                    }
                }
                keyFetchSources[i] = isAssigned(key) ? nodes + 1 : ((getFetchedEntry(previous, key, NULL, &keyCursor) != NULL) ? nodes : getFetchSource(key, keyCursor.block));
                fetchSourceOffsets[keyFetchSources[i]]++;
            }
            for (unsigned int node = 0; node <= nodes + 1; node++) {
                threadSourceOffsets[node * par.threads + thread_idx + 1] = fetchSourceOffsets[node];
            }
#pragma omp barrier
#pragma omp single
            {
                for (size_t i = 0; i + 1 < threadSourceOffsets.size(); i++) {
                    threadSourceOffsets[i + 1] += threadSourceOffsets[i];
                }
                sequences.keys.resize(threadSourceOffsets.back());
            }
            for (unsigned int node = 0; node <= nodes + 1; node++) {
                fetchSourceOffsets[node] = threadSourceOffsets[node * par.threads + thread_idx];
            }
#pragma omp for schedule(static)
            for (size_t i = 0; i < recordKeys.size(); i++) {
                if (i == 0 || recordKeys[i - 1].first != recordKeys[i].first) {
                    recordKeys[i].second = fetchSourceOffsets[keyFetchSources[i]]++;
                    sequences.keys[recordKeys[i].second] = recordKeys[i].first;
                }
            }
        }
        sequences.fetchSourceKeyOffsets.resize(nodes + 2);
        for (unsigned int node = 0; node <= nodes + 1; node++) {
            sequences.fetchSourceKeyOffsets[node] = threadSourceOffsets[node * par.threads];
        }
        sequences.ownKeyCopyEnd = sequences.fetchSourceKeyOffsets[rank];
        size_t keyCount = sequences.fetchSourceKeyOffsets[nodes + 1];
        sequences.keys.resize(keyCount);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < sequences.recordEntries.size(); i++) {
            size_t &entry = sequences.recordEntries[i];
            entry = (entry != SIZE_MAX && recordKeys[entry].second < keyCount) ? recordKeys[entry].second : SIZE_MAX;
        }
        sequences.entryOffsets.resize(keyCount + 1);
        sequences.entryOffsets[0] = 0;
        // keys of a fetch source ascend
#pragma omp parallel
        {
            size_t keyRange = SIZE_MAX;
#pragma omp for schedule(static)
            for (size_t i = 0; i < keyCount; i++) {
                sequences.entryOffsets[i + 1] = seqDbr.getSeqLenFromRange(keyRange, sequences.keys[i]) + 2;
            }
        }
        sumOffsets(sequences.entryOffsets.data(), keyCount, par.threads);
        // not zeroed, the entries overwrite it
        if (sequences.allocatedSize < sequences.entryOffsets[keyCount]) {
            delete[] sequences.entries;
            sequences.allocatedSize = sequences.entryOffsets[keyCount];
            sequences.entries = new(std::nothrow) char[sequences.allocatedSize];
            Util::checkAllocation(sequences.entries, "Can not allocate fetched sequence memory in align2clust");
        }
#pragma omp parallel
        {
            FetchedKeyCursor keyCursor(nodes + 1);
#pragma omp for schedule(dynamic, 1024)
            for (size_t i = sequences.fetchSourceKeyOffsets[nodes]; i < keyCount; i++) {
                memcpy(sequences.entries + sequences.entryOffsets[i], getFetchedEntry(previous, sequences.keys[i], NULL, &keyCursor), sequences.entryOffsets[i + 1] - sequences.entryOffsets[i]);
            }
        }
    };

    const auto getFirstRepKey = [&](const FetchedSequences &sequences) {
        if (sequences.end > sequences.begin) {
            return static_cast<size_t>(prefilterRecords[sequences.prefilterBuffer][sequences.begin].repOrderKeyFirstTarget.repKey);
        }
        // member budget may leave a fetch without records before others
        const std::vector<RepTargetsRecord> &bufferRecords = prefilterRecords[sequences.prefilterBuffer];
        return (sequences.end < bufferRecords.size()) ? static_cast<size_t>(bufferRecords[sequences.end].repOrderKeyFirstTarget.repKey) : orderKeyEnd;
    };
    // reps of pending and accepted pairs are not done
    const auto getFirstUndoneRepKey = [&](size_t nextRepKey) {
        if (pendingPairs.empty() == false) {
            return static_cast<size_t>(pendingPairs[0].repKey);
        }
        return acceptedPairs.empty() ? nextRepKey : static_cast<size_t>(acceptedPairs[0].repKey);
    };

    FetchedSequences currentSequences;
    while (prefilterReader.fillPrefilterBuffer(prefilterRecords[0], prefilterTargets[0], maxFetchSize, 0, false) == false) {
    }
    filledPrefilterBufferCount = 1;
    usedPrefilterBufferCount = 1;
    selectRecordKeys(currentSequences, fetchSlots[0]);
    publishAccepted(getFirstRepKey(currentSequences), 0);
    size_t publishedSize = 0;
    double publishTime = startTimer.getTimediff();
    double assignedKeysTime = publishTime;
    Timer assignedKeysTimer;
    double lastBatchReadTime = 0;
    double lastServeTime = 0;
    size_t batchReadCount = 0;
    double batchReadSeconds = 0;
    double batchReadPeriod = MIN_READ_SECONDS;

    std::mutex acceptedMutex;
    std::mutex publishMutex;
    std::mutex assignedKeysMutex;
    std::vector<size_t> batchOffsets;
    std::vector<unsigned int> batchFirstTargets;
    std::vector<size_t> recordAlignmentCells;
    std::vector<unsigned char> batchDone;
    std::vector<std::vector<ClusterPair> > batchPairs;
    std::vector<size_t> pairPartCounts;
    std::vector<size_t> partSizes;
    std::vector<size_t> partMemberCounts;
    std::vector<std::vector<size_t> > threadPartSizes(par.threads);
    std::vector<std::vector<size_t> > threadPartMemberCounts(par.threads);
    std::vector<std::vector<unsigned char> > threadMemberIsLane(par.threads);
    std::vector<unsigned char> membersPassed;
    std::vector<unsigned char> needsMemberCheck;
    std::vector<size_t> repPairOffsets;
    std::vector<size_t> batchRepOffsets;
    // made on first use and kept, deleting adds the SWIPE counters
    std::vector<TargetAligners *> threadTargetAligners(par.threads, NULL);
    std::vector<MemberAligners *> threadMemberAligners(par.threads, NULL);
    double waitSeconds = 0;
    double sequenceWaitSeconds = 0;
    // held by an I/O thread, owed by other nodes, other
    double sequenceWaitParts[3] = { 0, 0, 0 };
    int teamReceiveState = 2;
    // [0] I/O threads, [1] aligning team
    double receiveSeconds[2] = { 0, 0 };
    size_t receiveCalls[2] = { 0, 0 };
    double serveSeconds[2] = { 0, 0 };
    size_t serveCalls[2] = { 0, 0 };
    double servePackSeconds = 0;
    double serveWriteSeconds = 0;
    const std::thread::id alignThreadId = std::this_thread::get_id();
    double alignedCells = 0;
    double alignSeconds = 0;
    // wall time of the lookups, not thread time
    double memberLookupSeconds = 0;
    const auto mapRep = [&](Sequence &query, BlockAligner &blockAligner, size_t repKey) {
        size_t repLength = 0;
        const char *repData = getFetchedSequence(currentSequences, repKey, &repLength);
        query.mapSequence(repKey, repKey, repData, repLength);
        blockAligner.setQuery(&query);
    };
    bool isSwipeEnabled = par.align2clustAlignerMode != Parameters::ALIGNER_MODE_FARRAR && par.gapOpen.values.aminoacid() >= par.gapExtend.values.aminoacid();

    // handed to I/O threads, not requested yet, in selection order
    std::deque<FetchedSequences *> handedOffFetches;
    // handed-off fetches not taken yet, I/O threads copy own keys ahead
    std::deque<FetchedSequences *> ownKeyFetches;
    // node aligns ahead of the master by at most its share of a step
    size_t maxAheadSize = std::max<size_t>(1, getStepWords(masterMemoryLimit) / nodes) * sizeof(uint64_t);
    bool isWaitingForMaster = false;
    // one load of seqDbr at a time
    std::mutex sourceMutex;
    std::vector<DBKeyType> loadKeys;
    std::vector<std::vector<size_t> > threadRawEntries(par.threads);
    size_t servedFileCount = 0;
    size_t servedSize = 0;
    size_t servedEntrySize = 0;
    // served own keys longer than the representative this node aligns
    size_t servedBehindKeyCount = 0;
    size_t servedBehindSize = 0;
    // own blocks that a load touched, under sourceMutex
    std::vector<bool> isOwnBlockTouched(blockKeyOffsets.size() - 1, false);
    size_t receivedSize = 0;
    Timer serviceTimer;
    size_t requestCount = 0;

    // copy or pack the entries of keys, assigned keys get zeros
    const auto copyEntries = [&](const DBKeyType *keys, size_t keyCount, const size_t *offsets, char *data, std::vector<DBKeyType> *rawKeys, size_t maxLoadSize) {
        size_t loadRange = SIZE_MAX;
        for (size_t first = 0; first < keyCount;) {
            const size_t loadFirst = first;
            // with the block cache a small load reads one block
            const size_t loadKeyEnd = (blockCacheSize > 0 && maxLoadSize <= BatchReader::MAX_READ_REQUEST_SIZE) ? blockKeyOffsets[seqDbr.getKeyBlock(keys[first]) + 1] : SIZE_MAX;
            for (size_t loadSize = 0; first < keyCount; first++) {
                size_t entrySize = seqDbr.getSeqLenFromRange(loadRange, keys[first]) + 2 + 2 * pageSize;
                if (first > loadFirst && (loadSize + entrySize > maxLoadSize || keys[first] >= loadKeyEnd)) {
                    break;
                }
                loadSize += entrySize;
            }
            std::lock_guard<std::mutex> sourceLock(sourceMutex);
            loadKeys.clear();
            for (size_t i = loadFirst; i < first; i++) {
                if (isAssigned(keys[i]) == false) {
                    loadKeys.push_back(keys[i]);
                }
            }
            size_t touchedBlock = SIZE_MAX;
            for (size_t i = 0; i < loadKeys.size(); i++) {
                if (getKeyAlignNode(seqDbr, loadKeys[i], &touchedBlock) == rank) {
                    isOwnBlockTouched[touchedBlock] = true;
                }
            }
            seqDbr.loadData(loadKeys);
            for (int thread = 0; thread < par.threads; thread++) {
                threadRawEntries[thread].clear();
            }
            int copyThreads = 1;
#ifdef OPENMP
            copyThreads = omp_get_max_threads();
#endif
#pragma omp parallel num_threads(copyThreads)
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                // static chunks give a thread ascending keys
                size_t packRange = SIZE_MAX;
                DBReader<DBKeyType>::LoadedBufferCursor cursor;
#pragma omp for schedule(static)
                for (size_t i = loadFirst; i < first; i++) {
                    // bits are never cleared
                    if (isAssigned(keys[i])) {
                        memset(data + offsets[i], 0, offsets[i + 1] - offsets[i]);
                        continue;
                    }
                    const char *entry = seqDbr.getDataFromLoadedBuffer(keys[i], cursor);
                    if (rawKeys == NULL) {
                        memcpy(data + offsets[i], entry, offsets[i + 1] - offsets[i]);
                    } else if (packEntry(entry, seqDbr.getSeqLenFromRange(packRange, keys[i]) + 2, data + offsets[i]) == false) {
                        data[offsets[i]] = PACKED_RAW_CODE;
                        threadRawEntries[thread_idx].push_back(i);
                    }
                }
            }
            // static parts of the threads follow in key order
            for (int thread = 0; rawKeys != NULL && thread < par.threads; thread++) {
                for (size_t j = 0; j < threadRawEntries[thread].size(); j++) {
                    rawKeys->push_back(keys[threadRawEntries[thread][j]]);
                }
            }
        }
    };

    // own request: code offsets, received pieces and keys per node
    struct SequenceRequest {
        FetchedSequences *sequences;
        size_t number;
        size_t size;
        std::vector<size_t> codeOffsets;
        std::vector<size_t> receivedPieces;
        std::vector<size_t> receivedKeys;
        bool isComplete;
        double sentTime;
    };
    std::deque<SequenceRequest> ownRequests;
    size_t receivedRequestCount = 0;
    bool isLastRequested = false;
    // requested fetches become ready in selection order
    std::deque<FetchedSequences *> requestedFetches;
    std::vector<char> requestCodes;
    std::vector<char> requesterCodes;
    // piece of a served file
    struct ServedPiece {
        size_t request;
        size_t piece;
        size_t file;
        size_t fileSize;
        size_t offset;
        size_t size;
        size_t keyCount;
    };
    std::vector<double> nextServedProgressReads(nodes, 0);
    // seconds to the next read of an owing node, doubling to progressSeconds
    std::vector<double> servedProgressSeconds(nodes, BATCH_READ_SECONDS);
    // owing nodes are read fast until fastReadEnd
    double fastReadSeconds = BATCH_READ_SECONDS;
    double fastReadEnd = 0;
    double nextRequestListing = 0;
    double nextServedListing = 0;
    // master progress while the gate is closed
    double nextGateRead = 0;
    std::vector<std::string> requestNames;
    std::vector<std::string> servedNames;
    // one past the newest served file per node, listed and in progress
    std::vector<size_t> listedServedFileCounts(nodes, 0);
    std::vector<size_t> progressServedFileCounts(nodes, 0);
    std::vector<std::vector<ServedPiece> > servedPieces(nodes);
    // requesting node and the pieces this node served it
    struct Requester {
        Requester() : requestCount(0), receivedCount(0), isLast(false), isDone(false), nextProgressRead(0), servedCount(0), codePos(0), lastKey(0), piece(0) {}
        size_t requestCount;
        size_t receivedCount;
        bool isLast;
        bool isDone;
        double nextProgressRead;
        std::vector<size_t> openRequests;
        size_t servedCount;
        size_t codePos;
        DBKeyType lastKey;
        size_t piece;
        std::vector<ServedPiece> pieces;
    };
    std::vector<Requester> requesters(nodes);
    // own progress files are rewritten, others stay open
    std::vector<MMseqsNode::PolledFile> requestProgress(nodes);
    std::vector<MMseqsNode::PolledFile> servedProgress(nodes);
    for (unsigned int node = 0; node < nodes; node++) {
        requestProgress[node].file = requests + "." + SSTR(node) + ".progress";
        servedProgress[node].file = served + "." + SSTR(node) + ".progress";
    }
    size_t requesterCount = 0;
    for (unsigned int node = 0; node < nodes; node++) {
        requesters[node].isDone = node == rank || node < firstAlignNode;
        requesterCount += requesters[node].isDone == false;
    }
    // seconds between two reads of the progress file of a node
    double progressSeconds = std::max(BATCH_READ_SECONDS, 2.0 * requesterCount / PROGRESS_READS_PER_SECOND);
    double listingSeconds = std::max(BATCH_READ_SECONDS, (nodes - 1.0) / PROGRESS_READS_PER_SECOND) / 4.0;
    std::vector<const char *> requestColumns(nodes + 3);
    bool isServedProgressChanged = false;
    std::vector<char> receivedEntries;
    std::vector<DBKeyType> servedKeys;
    std::vector<DBKeyType> rawKeys;
    std::vector<size_t> requestOffsets;
    // pieces of a pass in one served file
    std::vector<char> servedFileData;
    std::vector<std::pair<unsigned int, ServedPiece> > passPieces;
    std::vector<size_t> filePieceCounts;
    // next pass starts at the first requester left out
    unsigned int firstRequester = rank + 1;
    // failure tracking of served files and requests per node
    std::vector<time_t> missingServedSince(nodes, 0);
    std::vector<unsigned int> missingServedAttempts(nodes, 0);
    std::vector<time_t> missingRequestSince(nodes, 0);
    std::vector<unsigned int> missingRequestAttempts(nodes, 0);
    const auto getServedFile = [&](unsigned int alignNode, size_t file) {
        return served + "/" + SSTR(alignNode) + "." + SSTR(file);
    };
    MMseqsNode::WaitWarning waitWarning;
    // serving passes that worked, they reset the wait warning too
    std::atomic<size_t> exportPassCount(0);
    size_t seenExportPassCount = 0;
    time_t assignedKeysReadTime = 0;
    bool isImportUrgent = false;
    // under serviceMutex
    bool isReceiveDone = false;
    bool isServeDone = false;
    bool isAligningDone = false;
    double nextStatusTime = STATUS_SECONDS;
    const auto printStatus = [&]() {
        double now = serviceTimer.getTimediff();
        size_t doneRepKey = publishedDoneRepKey.load();
        std::string value = SSTR(std::min(doneRepKey, orderKeyEnd) * 100 / orderKeyEnd) + "% aligned (representative order key " + SSTR(doneRepKey) + " of " + SSTR(orderKeyEnd) + ")";
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            value += ", waited " + formatSeconds(sequenceWaitSeconds) + " for sequences and " + formatSeconds(waitSeconds) + " for the master";
        }
        std::string status;
        MMseqsNode::addPhaseValue(status, "Align node status after " + SSTR(static_cast<size_t>(now)) + " s", value);
        MMseqsNode::printPhaseValues(status);
    };

    // chunks of a task loop, idle threads join with their team
    struct SharedLoop {
        const std::function<void(size_t)> *body;
        size_t count;
        std::atomic<size_t> next;
        size_t users;
    };
    std::mutex sharedLoopMutex;
    std::condition_variable sharedLoopCondition;
    std::vector<SharedLoop *> sharedLoops;
    const auto runLoopChunks = [&](SharedLoop &loop) {
#pragma omp parallel
        {
            for (size_t chunk = loop.next++; chunk < loop.count; chunk = loop.next++) {
                (*loop.body)(chunk);
            }
        }
    };
    // returns once all chunks are done, the caller completes the task
    const std::function<void(size_t, const std::function<void(size_t)> &)> runSharedLoop = [&](size_t count, const std::function<void(size_t)> &body) {
        SharedLoop loop;
        loop.body = &body;
        loop.count = count;
        loop.next = 0;
        loop.users = 0;
        {
            std::lock_guard<std::mutex> lock(sharedLoopMutex);
            sharedLoops.push_back(&loop);
        }
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            ioWork++;
        }
        serviceCondition.notify_all();
        runLoopChunks(loop);
        std::unique_lock<std::mutex> lock(sharedLoopMutex);
        sharedLoops.erase(std::find(sharedLoops.begin(), sharedLoops.end(), &loop));
        sharedLoopCondition.wait(lock, [&]() { return loop.users == 0; });
    };
    const auto joinSharedLoop = [&]() {
        SharedLoop *loop = NULL;
        {
            std::lock_guard<std::mutex> lock(sharedLoopMutex);
            for (size_t i = 0; i < sharedLoops.size() && loop == NULL; i++) {
                if (sharedLoops[i]->next < sharedLoops[i]->count) {
                    loop = sharedLoops[i];
                    loop->users++;
                }
            }
        }
        if (loop == NULL) {
            return false;
        }
        runLoopChunks(*loop);
        {
            std::lock_guard<std::mutex> lock(sharedLoopMutex);
            loop->users--;
        }
        sharedLoopCondition.notify_all();
        return true;
    };

    std::mutex receiveMutex;
    std::mutex serveMutex;
    // pass over own requests and their served pieces
    const auto receiveSequences = [&]() {
        bool isTeam = std::this_thread::get_id() == alignThreadId;
        std::unique_lock<std::mutex> taskLock(receiveMutex, std::try_to_lock);
        if (taskLock.owns_lock() == false) {
            teamReceiveState = isTeam ? 0 : teamReceiveState;
            return false;
        }
        Timer receiveTimer;
        bool hasWorked = false;
        // keep the bits of the served keys current
        time_t passTime = time(NULL);
        if (passTime - assignedKeysReadTime >= READ_SECONDS) {
            std::unique_lock<std::mutex> assignedKeysLock(assignedKeysMutex, std::try_to_lock);
            if (assignedKeysLock.owns_lock()) {
                readAssignedKeys(false);
                assignedKeysReadTime = passTime;
            }
        }

        bool isRequestProgressChanged = false;
        // request handed-off fetches while the master is close behind
        bool isGated = false;
        while (isGated == false) {
            FetchedSequences *nextRequest = NULL;
            size_t nextAcceptedSize = 0;
            {
                std::lock_guard<std::mutex> lock(serviceMutex);
                if (handedOffFetches.empty() == false) {
                    nextRequest = handedOffFetches.front();
                    nextAcceptedSize = nextRequest->acceptedSize;
                }
            }
            if (nextRequest == NULL) {
                break;
            }
            bool isLast = nextRequest->isLast;
            // fetch without records and pending pairs logs nothing, never waits
            bool canWait = isLast == false && (nextRequest->end > nextRequest->begin || nextRequest->pendingPairCount > 0);
            {
                std::lock_guard<std::mutex> assignedKeysLock(assignedKeysMutex);
                size_t repKey = std::min(nextRequest->firstPendingRepKey, getFirstRepKey(*nextRequest));
                isGated = canWait && repKey > clusteredOrderKey && nextAcceptedSize >= clusteredAcceptedSize + maxAheadSize;
                if (isGated && serviceTimer.getTimediff() >= nextGateRead) {
                    nextGateRead = serviceTimer.getTimediff() + BATCH_READ_SECONDS / 2;
                    readAssignedKeys(true);
                    isGated = repKey > clusteredOrderKey && nextAcceptedSize >= clusteredAcceptedSize + maxAheadSize;
                }
            }
            {
                std::lock_guard<std::mutex> lock(serviceMutex);
                isWaitingForMaster = isGated;
                if (isGated == false) {
                    handedOffFetches.pop_front();
                }
            }
            if (isGated == false) {
                // keys of a node ascend, a request holds their gaps
                SequenceRequest request;
                request.sequences = nextRequest;
                request.number = requestCount;
                request.codeOffsets.assign(nodes + 1, 0);
                requestCodes.clear();
                for (unsigned int node = 0; node < nodes; node++) {
                    request.codeOffsets[node] = requestCodes.size();
                    size_t keyBegin = nextRequest->fetchSourceKeyOffsets[node];
                    size_t keyEnd = (node != rank) ? nextRequest->fetchSourceKeyOffsets[node + 1] : keyBegin;
                    for (size_t i = keyBegin; i < keyEnd; i++) {
                        char code[10];
                        size_t codeSize = Util::putPrefKey(code, nextRequest->keys[i] - ((i > keyBegin) ? nextRequest->keys[i - 1] : 0));
                        requestCodes.insert(requestCodes.end(), code, code + codeSize);
                    }
                }
                request.codeOffsets[nodes] = requestCodes.size();
                // fetch without keys of other nodes has no request
                if (requestCodes.empty() == false) {
                    MMseqsNode::writeExchangeFile(requests + "/" + SSTR(rank) + "." + SSTR(request.number), request.number, requestCodes.data(), requestCodes.size(), "", std::vector<uint64_t>(), 1);
                    request.size = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + requestCodes.size();
                    request.receivedPieces.assign(nodes, 0);
                    request.receivedKeys.assign(nodes, 0);
                    request.isComplete = false;
                    request.sentTime = serviceTimer.getTimediff();
                    ownRequests.push_back(request);
                    size_t owedCount = 0;
                    for (unsigned int node = 0; node < nodes; node++) {
                        owedCount += request.codeOffsets[node + 1] > request.codeOffsets[node];
                    }
                    fastReadSeconds = std::max(0.01, 2.0 * owedCount / PROGRESS_READS_PER_SECOND);
                    fastReadEnd = serviceTimer.getTimediff() + BATCH_READ_SECONDS;
                    std::fill(servedProgressSeconds.begin(), servedProgressSeconds.end(), fastReadSeconds);
                    std::fill(nextServedProgressReads.begin(), nextServedProgressReads.end(), serviceTimer.getTimediff() + fastReadSeconds);
                    requestCount++;
                    isRequestProgressChanged = true;
                }
                isRequestProgressChanged |= isLast;
                isLastRequested = isLast;
                requestedFetches.push_back(nextRequest);
                hasWorked = true;
            }
        }

        // read an owing node's progress once a listing shows a new file
        bool isServedListed = ownRequests.empty() == false && serviceTimer.getTimediff() >= nextServedListing;
        if (isServedListed) {
            nextServedListing = serviceTimer.getTimediff() + listingSeconds;
            MMseqsNode::listDirectory(served, servedNames);
            std::fill(listedServedFileCounts.begin(), listedServedFileCounts.end(), 0);
            for (size_t i = 0; i < servedNames.size(); i++) {
                const char *name = servedNames[i].c_str();
                const char *number = strchr(name, '.');
                size_t node = Util::fast_atoi<size_t>(name);
                if (number != NULL && number != name && node < nodes) {
                    listedServedFileCounts[node] = std::max(listedServedFileCounts[node], Util::fast_atoi<size_t>(number + 1) + 1);
                }
            }
        }

        // receive from the nodes that owe a request
        for (unsigned int node = 0; node < nodes; node++) {
            bool isOwed = false;
            for (size_t i = 0; i < ownRequests.size() && isOwed == false; i++) {
                isOwed = node != rank && ownRequests[i].receivedKeys[node] < ownRequests[i].sequences->fetchSourceKeyOffsets[node + 1] - ownRequests[i].sequences->fetchSourceKeyOffsets[node];
            }
            if (isOwed && isServedListed && listedServedFileCounts[node] > progressServedFileCounts[node] && serviceTimer.getTimediff() < nextServedProgressReads[node]) {
                nextServedProgressReads[node] = 0;
            }
            if (isOwed == false || serviceTimer.getTimediff() < nextServedProgressReads[node]) {
                continue;
            }
            std::string content;
            nextServedProgressReads[node] = serviceTimer.getTimediff() + servedProgressSeconds[node];
            if (serviceTimer.getTimediff() >= fastReadEnd) {
                servedProgressSeconds[node] = std::min(2.0 * servedProgressSeconds[node], progressSeconds);
            }
            bool hasProgress = servedProgress[node].read(content);
            if (hasProgress) {
                servedPieces[node].clear();
                for (char *line = &content[0]; *line != '\0'; line = Util::skipLine(line)) {
                    const char *columns[8];
                    if (Util::getWordsOfLine(line, columns, 8) != 8) {
                        continue;
                    }
                    // count files of all requesters, a foreign file costs one read
                    progressServedFileCounts[node] = std::max(progressServedFileCounts[node], Util::fast_atoi<size_t>(columns[3]) + 1);
                    if (Util::fast_atoi<size_t>(columns[0]) == rank) {
                        ServedPiece piece = { Util::fast_atoi<size_t>(columns[1]), Util::fast_atoi<size_t>(columns[2]), Util::fast_atoi<size_t>(columns[3]), Util::fast_atoi<size_t>(columns[4]),
                                              Util::fast_atoi<size_t>(columns[5]), Util::fast_atoi<size_t>(columns[6]), Util::fast_atoi<size_t>(columns[7]) };
                        servedPieces[node].push_back(piece);
                    }
                }
            }
            for (size_t i = 0; i < ownRequests.size(); i++) {
                SequenceRequest &request = ownRequests[i];
                for (size_t j = 0; j < servedPieces[node].size(); j++) {
                    const ServedPiece &piece = servedPieces[node][j];
                    if (piece.request != request.number || piece.piece != request.receivedPieces[node]) {
                        continue;
                    }
                    // next piece from its served file, retry after the next progress read
                    std::string fileName = getServedFile(node, piece.file);
                    size_t observedSize;
                    uint64_t observedNumber;
                    int lastError;
                    int fd = MMseqsNode::tryOpenExchangeFile(fileName, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + piece.fileSize, piece.file, observedSize, observedNumber, lastError);
                    if (fd < 0) {
                        if (lastError != 0 && lastError != ENOENT && FileUtil::isTransientError(lastError) == false) {
                            Debug(Debug::ERROR) << "Cannot open sequence file " << fileName << ": " << strerror(lastError) << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        failIfMissingTooLong(missingServedSince[node], missingServedAttempts[node], "Cannot open complete served sequence file " + fileName
                                             + describeExchangeFile(MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + piece.fileSize, observedSize, piece.file, observedNumber, lastError));
                        break;
                    }
                    if (receivedEntries.size() < piece.size) {
                        receivedEntries.resize(piece.size);
                    }
                    // parts keep their errors
                    size_t partCount = (piece.size + BatchReader::MAX_READ_REQUEST_SIZE - 1) / BatchReader::MAX_READ_REQUEST_SIZE;
                    std::vector<int> readErrors(partCount, 0);
                    runSharedLoop(partCount, [&](size_t part) {
                        size_t offset = part * BatchReader::MAX_READ_REQUEST_SIZE;
                        size_t partSize = std::min(BatchReader::MAX_READ_REQUEST_SIZE, piece.size - offset);
                        int partError;
                        if (FileUtil::tryPreadAll(fd, receivedEntries.data() + offset, partSize, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + piece.offset + offset, partError) < partSize) {
                            readErrors[part] = (partError != 0) ? partError : -1;
                        }
                    });
                    int readError = 0;
                    for (size_t part = 0; part < partCount; part++) {
                        readError = (readErrors[part] != 0) ? readErrors[part] : readError;
                    }
                    int closeError = (close(fd) != 0) ? errno : 0;
                    int error = (readError > 0) ? readError : closeError;
                    if (error != 0 && error != EINTR && FileUtil::isTransientError(error) == false) {
                        Debug(Debug::ERROR) << "Cannot read sequence file " << fileName << ": " << strerror(error) << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    if (readError != 0 || closeError != 0) {
                        failIfMissingTooLong(missingServedSince[node], missingServedAttempts[node], "Cannot read complete served sequence file " + fileName + " (error " + SSTR(error) + ")\n");
                        break;
                    }
                    missingServedAttempts[node] = 0;
                    FetchedSequences *sequences = request.sequences;
                    size_t firstKey = sequences->fetchSourceKeyOffsets[node] + request.receivedKeys[node];
                    unpackEntries(receivedEntries.data(), piece.keyCount, sequences->entryOffsets.data() + firstKey, sequences->entries, par.threads, runSharedLoop);
                    request.receivedKeys[node] += piece.keyCount;
                    request.receivedPieces[node]++;
                    receivedSize += piece.size;
                    servedProgressSeconds[node] = fastReadSeconds;
                    nextServedProgressReads[node] = serviceTimer.getTimediff() + fastReadSeconds;
                    hasWorked = true;
                }
            }
        }

        for (size_t i = 0; i < ownRequests.size(); i++) {
            SequenceRequest &request = ownRequests[i];
            bool isComplete = request.isComplete == false;
            for (unsigned int node = 0; node < nodes && isComplete; node++) {
                isComplete = node == rank || request.receivedKeys[node] == request.sequences->fetchSourceKeyOffsets[node + 1] - request.sequences->fetchSourceKeyOffsets[node];
            }
            if (isComplete) {
                request.isComplete = true;
                roundTrips[request.number % 3] = serviceTimer.getTimediff() - request.sentTime;
                fetchAheadSeconds.store(std::max(MIN_FETCH_AHEAD_SECONDS, *std::min_element(roundTrips, roundTrips + 3)));
                FileUtil::removeSharedFile(requests + "/" + SSTR(rank) + "." + SSTR(request.number));
                hasWorked = true;
            }
        }
        for (; ownRequests.empty() == false && ownRequests.front().isComplete; receivedRequestCount++) {
            ownRequests.pop_front();
            isRequestProgressChanged = true;
        }

        // requests not received yet with the code offsets of all nodes
        if (isRequestProgressChanged && alignNodeCount > 1) {
            std::string content = SSTR(requestCount) + "\t" + SSTR(receivedRequestCount) + "\t" + SSTR(isLastRequested ? 1 : 0) + "\n";
            for (size_t i = 0; i < ownRequests.size(); i++) {
                content += SSTR(ownRequests[i].number) + "\t" + SSTR(ownRequests[i].size);
                for (unsigned int node = 0; node <= nodes; node++) {
                    content += "\t" + SSTR(ownRequests[i].codeOffsets[node]);
                }
                content += "\n";
            }
            requestProgress[rank].publish(content);
        }

        // fetch goes to the aligners once its request is received
        while (requestedFetches.empty() == false && (ownRequests.empty() || ownRequests.front().sequences != requestedFetches.front())) {
            {
                std::lock_guard<std::mutex> lock(serviceMutex);
                requestedFetches.front()->isReady = true;
            }
            serviceCondition.notify_all();
            requestedFetches.pop_front();
            hasWorked = true;
        }

        bool isExchangeDone;
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            isReceiveDone = isLastRequested && ownRequests.empty() && requestedFetches.empty();
            isImportUrgent = requestedFetches.empty() == false || isGated;
            isExchangeDone = isReceiveDone && isServeDone;
        }
        if (serviceTimer.getTimediff() >= nextStatusTime) {
            nextStatusTime = serviceTimer.getTimediff() + STATUS_SECONDS;
            printStatus();
        }
        // no wait if aligning without a request or the exchange ended
        size_t exportPasses = exportPassCount;
        if (hasWorked || isExchangeDone || exportPasses != seenExportPassCount || (requestedFetches.empty() && isGated == false && isLastRequested == false)) {
            waitWarning.reset();
        } else {
            waitWarning.check(requestedFetches.empty() == false ? served + " of other nodes" : (isGated ? assignedKeysLog : requests + " of other nodes"));
        }
        seenExportPassCount = exportPasses;
        receiveSeconds[isTeam] += receiveTimer.getTimediff();
        receiveCalls[isTeam]++;
        if (isTeam) {
            teamReceiveState = (hasWorked == false && ownRequests.empty() == false) ? 1 : 2;
        }
        return hasWorked;
    };

    // pass over the requests of the other nodes
    const auto serveSequences = [&]() {
        std::unique_lock<std::mutex> taskLock(serveMutex, std::try_to_lock);
        if (taskLock.owns_lock() == false) {
            return false;
        }
        bool isTeam = std::this_thread::get_id() == alignThreadId;
        Timer serveTimer;
        bool hasWorked = false;
        // read a requester's progress once a listing shows a new request
        if (serviceTimer.getTimediff() >= nextRequestListing) {
            nextRequestListing = serviceTimer.getTimediff() + listingSeconds;
            MMseqsNode::listDirectory(requests, requestNames);
            for (size_t i = 0; i < requestNames.size(); i++) {
                const char *name = requestNames[i].c_str();
                const char *number = strchr(name, '.');
                size_t node = Util::fast_atoi<size_t>(name);
                if (number != NULL && number != name && node < nodes && Util::fast_atoi<size_t>(number + 1) >= requesters[node].requestCount) {
                    requesters[node].nextProgressRead = 0;
                }
            }
        }

        // piece a pass for each requesting node
        Timer packTimer;
        bool isServing = false;
        size_t passEntrySize = 0;
        size_t passKeyCount = 0;
        unsigned int deferredRequester = UINT_MAX;
        for (unsigned int i = 0; i < nodes; i++) {
            unsigned int node = (firstRequester + i) % nodes;
            Requester &requester = requesters[node];
            if (requester.isDone) {
                continue;
            }
            if (serviceTimer.getTimediff() >= requester.nextProgressRead) {
                std::string content;
                requester.nextProgressRead = serviceTimer.getTimediff() + progressSeconds;
                bool hasProgress = requestProgress[node].read(content);
                if (hasProgress) {
                    char *line = &content[0];
                    const char *columns[3];
                    if (Util::getWordsOfLine(line, columns, 3) < 3) {
                        Debug(Debug::ERROR) << "Cannot read " << requests << "." << node << ".progress\n";
                        EXIT(EXIT_FAILURE);
                    }
                    size_t count = Util::fast_atoi<size_t>(columns[0]);
                    size_t received = Util::fast_atoi<size_t>(columns[1]);
                    bool isLast = Util::fast_atoi<size_t>(columns[2]) != 0;
                    requester.requestCount = count;
                    requester.receivedCount = received;
                    requester.isLast = isLast;
                    // received requests count as served
                    requester.servedCount = std::max(requester.servedCount, received);
                    // per open request: number, size and this node's code range
                    requester.openRequests.clear();
                    for (line = Util::skipLine(line); *line != '\0'; line = Util::skipLine(line)) {
                        Util::getWordsOfLine(line, requestColumns.data(), nodes + 3);
                        requester.openRequests.push_back(Util::fast_atoi<size_t>(requestColumns[0]));
                        requester.openRequests.push_back(Util::fast_atoi<size_t>(requestColumns[1]));
                        requester.openRequests.push_back(Util::fast_atoi<size_t>(requestColumns[2 + rank]));
                        requester.openRequests.push_back(Util::fast_atoi<size_t>(requestColumns[3 + rank]));
                    }
                    // drop pieces of received requests, a served file with its last piece
                    size_t keptPieces = 0;
                    for (size_t j = 0; j < requester.pieces.size(); j++) {
                        const ServedPiece &piece = requester.pieces[j];
                        if (piece.request < received) {
                            if (--filePieceCounts[piece.file] == 0) {
                                FileUtil::removeSharedFile(getServedFile(rank, piece.file));
                            }
                            isServedProgressChanged = true;
                        } else {
                            requester.pieces[keptPieces++] = piece;
                        }
                    }
                    requester.pieces.resize(keptPieces);
                    requester.isDone = isLast && received == count;
                }
            }
            // one piece of the next open request
            while (requester.servedCount < requester.requestCount) {
                size_t request = 0;
                while (request < requester.openRequests.size() && requester.openRequests[request] != requester.servedCount) {
                    request += 4;
                }
                if (request == requester.openRequests.size()) {
                    break;
                }
                size_t requestSize = requester.openRequests[request + 1];
                size_t codeBegin = requester.openRequests[request + 2];
                size_t codeEnd = requester.openRequests[request + 3];
                if (codeBegin == codeEnd) {
                    requester.servedCount++;
                    continue;
                }
                // full pass leaves the rest to the next pass
                if (passEntrySize >= servedFileEntrySize || passKeyCount >= maxServedKeys) {
                    deferredRequester = (deferredRequester == UINT_MAX) ? node : deferredRequester;
                    break;
                }
                // read the next codes, a request is read in pieces
                std::string requestFile = requests + "/" + SSTR(node) + "." + SSTR(requester.servedCount);
                size_t observedSize;
                uint64_t observedNumber;
                int lastError;
                int requestFd = MMseqsNode::tryOpenExchangeFile(requestFile, requestSize, requester.servedCount, observedSize, observedNumber, lastError);
                if (requestFd < 0 && lastError != 0 && lastError != ENOENT && FileUtil::isTransientError(lastError) == false) {
                    Debug(Debug::ERROR) << "Cannot open sequence request " << requestFile << ": " << strerror(lastError) << "\n";
                    EXIT(EXIT_FAILURE);
                }
                requesterCodes.resize(std::min(codeEnd - codeBegin - requester.codePos, maxServedKeys * sizeof(DBKeyType)));
                bool isRequestRead = requestFd >= 0 && FileUtil::tryPreadAll(requestFd, requesterCodes.data(), requesterCodes.size(), MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + codeBegin + requester.codePos, lastError) == requesterCodes.size();
                if (requestFd >= 0 && close(requestFd) != 0 && isRequestRead) {
                    lastError = errno;
                    isRequestRead = false;
                }
                // missing or short request is tried again in the next pass
                if (isRequestRead == false) {
                    if (lastError != 0 && lastError != ENOENT && lastError != EINTR && FileUtil::isTransientError(lastError) == false) {
                        Debug(Debug::ERROR) << "Cannot read sequence request " << requestFile << ": " << strerror(lastError) << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    failIfMissingTooLong(missingRequestSince[node], missingRequestAttempts[node], "Cannot read complete sequence request " + requestFile
                                         + describeExchangeFile(requestSize, observedSize, requester.servedCount, observedNumber, lastError));
                    break;
                }
                missingRequestAttempts[node] = 0;
                // only the last byte of a code is below 0x80
                const char *codesEnd = requesterCodes.data() + requesterCodes.size();
                while (codesEnd > requesterCodes.data() && static_cast<unsigned char>(codesEnd[-1]) >= 0x80) {
                    codesEnd--;
                }
                // unassigned entries of one key at least, within the pass limits
                servedKeys.clear();
                const char *code = requesterCodes.data();
                size_t keyRange = SIZE_MAX;
                while (passKeyCount + servedKeys.size() < maxServedKeys && code < codesEnd) {
                    const char *nextCode = code;
                    DBKeyType key = requester.lastKey + Util::getPrefKey(nextCode);
                    size_t keySize = isAssigned(key) ? 0 : seqDbr.getSeqLenFromRange(keyRange, key) + 2;
                    if (servedKeys.empty() == false && passEntrySize + keySize > servedFileEntrySize) {
                        break;
                    }
                    servedKeys.push_back(key);
                    requester.lastKey = key;
                    code = nextCode;
                    passEntrySize += keySize;
                }
                requester.codePos += code - requesterCodes.data();
                const DBKeyType *keys = servedKeys.data();
                size_t keyCount = servedKeys.size();
                // bitmap of a piece takes whole bytes
                passKeyCount += (keyCount + 7) / 8 * 8;
                requestOffsets.resize(keyCount + 1);
                requestOffsets[0] = (keyCount + 7) / 8;
#pragma omp parallel
                {
                    size_t servedKeyRange = SIZE_MAX;
#pragma omp for schedule(static)
                    for (size_t j = 0; j < keyCount; j++) {
                        size_t packedSize = getPackedEntrySize(seqDbr.getSeqLenFromRange(servedKeyRange, keys[j]) + 2);
                        requestOffsets[j + 1] = isAssigned(keys[j]) ? 0 : packedSize;
                    }
                }
                for (size_t j = 0; j < keyCount; j++) {
                    requestOffsets[j + 1] += requestOffsets[j];
                }
                size_t pieceOffset = servedFileData.size();
                // served file of a pass fits maxServedFileSize
                servedFileData.reserve(maxServedFileSize);
                servedFileData.resize(pieceOffset + requestOffsets[keyCount]);
                // empty slot marks an assigned key
#pragma omp parallel for schedule(static)
                for (size_t byte = 0; byte < requestOffsets[0]; byte++) {
                    unsigned char bits = 0;
                    for (size_t j = 8 * byte; j < std::min(keyCount, 8 * byte + 8); j++) {
                        bits |= static_cast<unsigned char>(requestOffsets[j + 1] == requestOffsets[j]) << (j % 8);
                    }
                    servedFileData[pieceOffset + byte] = static_cast<char>(bits);
                }
                rawKeys.clear();
                copyEntries(keys, keyCount, requestOffsets.data(), servedFileData.data() + pieceOffset, &rawKeys, BatchReader::MAX_READ_REQUEST_SIZE);
                size_t pieceSize = requestOffsets[keyCount];
                requestOffsets.resize(rawKeys.size() + 1);
                requestOffsets[0] = 0;
                size_t rawKeyRange = SIZE_MAX;
                for (size_t j = 0; j < rawKeys.size(); j++) {
                    requestOffsets[j + 1] = requestOffsets[j] + seqDbr.getSeqLenFromRange(rawKeyRange, rawKeys[j]) + 2;
                }
                servedFileData.resize(pieceOffset + pieceSize + requestOffsets[rawKeys.size()]);
                copyEntries(rawKeys.data(), rawKeys.size(), requestOffsets.data(), servedFileData.data() + pieceOffset + pieceSize, NULL, BatchReader::MAX_READ_REQUEST_SIZE);
                pieceSize += requestOffsets[rawKeys.size()];
                size_t entryRange = SIZE_MAX;
                size_t frontierKey = publishedDoneRepKey.load() % paddedDbSize;
                size_t frontierRange = SIZE_MAX;
                size_t frontierLength = (frontierKey < dbSize) ? seqDbr.getSeqLenFromRange(frontierRange, frontierKey) : 0;
                for (size_t j = 0; j < keyCount; j++) {
                    size_t entrySize = isAssigned(keys[j]) ? 0 : seqDbr.getSeqLenFromRange(entryRange, keys[j]) + 2;
                    servedEntrySize += entrySize;
                    if (entrySize > frontierLength + 2) {
                        servedBehindKeyCount++;
                        servedBehindSize += entrySize;
                    }
                }
                ServedPiece piece = { requester.servedCount, requester.piece, servedFileCount, 0, pieceOffset, pieceSize, keyCount };
                passPieces.push_back(std::make_pair(node, piece));
                servedSize += pieceSize;
                requester.piece++;
                if (requester.codePos == codeEnd - codeBegin) {
                    requester.codePos = 0;
                    requester.lastKey = 0;
                    requester.piece = 0;
                    requester.servedCount++;
                }
                hasWorked = true;
                break;
            }
            isServing |= requester.isDone == false;
        }
        firstRequester = (deferredRequester == UINT_MAX) ? firstRequester : deferredRequester;

        servePackSeconds += packTimer.getTimediff();
        if (passPieces.empty() == false) {
            Timer writeTimer;
            // idle aligning team joins the chunk writes, header last once all are done
            std::string servedFile = getServedFile(rank, servedFileCount);
            FILE *file = FileUtil::openAndDelete(servedFile.c_str(), "w");
            const size_t chunkSize = BatchReader::MAX_READ_REQUEST_SIZE;
            runSharedLoop((servedFileData.size() + chunkSize - 1) / chunkSize, [&](size_t chunk) {
                size_t offset = chunk * chunkSize;
                FileUtil::pwriteAll(fileno(file), servedFile.c_str(), servedFileData.data() + offset, std::min(chunkSize, servedFileData.size() - offset), MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + offset);
            });
            MMseqsNode::finishExchangeFile(file, servedFile, servedFileCount, servedFileData.size(), "", std::vector<uint64_t>());
            for (size_t j = 0; j < passPieces.size(); j++) {
                passPieces[j].second.fileSize = servedFileData.size();
                requesters[passPieces[j].first].pieces.push_back(passPieces[j].second);
            }
            filePieceCounts.push_back(passPieces.size());
            passPieces.clear();
            servedFileData.clear();
            servedFileCount++;
            isServedProgressChanged = true;
            serveWriteSeconds += writeTimer.getTimediff();
        }

        // pieces not yet deleted, per requester
        if (isServedProgressChanged) {
            std::string content;
            for (unsigned int node = 0; node < nodes; node++) {
                for (size_t i = 0; i < requesters[node].pieces.size(); i++) {
                    const ServedPiece &piece = requesters[node].pieces[i];
                    content += SSTR(node) + "\t" + SSTR(piece.request) + "\t" + SSTR(piece.piece) + "\t" + SSTR(piece.file) + "\t" + SSTR(piece.fileSize)
                               + "\t" + SSTR(piece.offset) + "\t" + SSTR(piece.size) + "\t" + SSTR(piece.keyCount) + "\n";
                }
            }
            servedProgress[rank].publish(content);
            isServedProgressChanged = false;
        }
        if (hasWorked) {
            exportPassCount++;
        }
        serveSeconds[isTeam] += serveTimer.getTimediff();
        serveCalls[isTeam]++;
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            if (isServing || isServeDone) {
                return hasWorked;
            }
            isServeDone = true;
            ioWork++;
        }
        serviceCondition.notify_all();
        return hasWorked;
    };

    std::mutex ownKeyCopyMutex;
    // own keys not copied ahead, copied by the idle aligning team
    const auto copyOwnKeys = [&](FetchedSequences &sequences) {
        Timer copyTimer;
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            ownKeyFetches.erase(std::remove(ownKeyFetches.begin(), ownKeyFetches.end(), &sequences), ownKeyFetches.end());
        }
        size_t ownKey;
        {
            // chunk copied ahead ends before the fetch aligns
            std::lock_guard<std::mutex> copyLock(ownKeyCopyMutex);
            ownKey = sequences.ownKeyCopyEnd;
        }
        size_t ownKeyEnd = sequences.fetchSourceKeyOffsets[rank + 1];
        copyEntries(sequences.keys.data() + ownKey, ownKeyEnd - ownKey, sequences.entryOffsets.data() + ownKey, sequences.entries, NULL, maxFetchSize);
        // aligning team waits for these sequences too
        sequenceWaitSeconds += copyTimer.getTimediff();
        sequenceWaitParts[2] += copyTimer.getTimediff();
    };
    // copy own keys of the next block of the queued fetches
    const auto copyOwnKeysAhead = [&]() {
        std::unique_lock<std::mutex> taskLock(ownKeyCopyMutex, std::try_to_lock);
        if (taskLock.owns_lock() == false) {
            return false;
        }
        FetchedSequences *sequences = NULL;
        size_t first = 0;
        size_t end = 0;
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            while (ownKeyFetches.empty() == false && sequences == NULL) {
                FetchedSequences &front = *ownKeyFetches.front();
                size_t ownKeyEnd = front.fetchSourceKeyOffsets[rank + 1];
                if (front.ownKeyCopyEnd < ownKeyEnd) {
                    sequences = &front;
                    first = front.ownKeyCopyEnd;
                    const DBKeyType *keys = front.keys.data();
                    end = std::lower_bound(keys + first, keys + ownKeyEnd, blockKeyOffsets[seqDbr.getKeyBlock(keys[first]) + 1]) - keys;
                    front.ownKeyCopyEnd = end;
                }
                if (front.ownKeyCopyEnd == ownKeyEnd) {
                    ownKeyFetches.pop_front();
                }
            }
        }
        if (sequences == NULL) {
            return false;
        }
        copyEntries(sequences->keys.data() + first, end - first, sequences->entryOffsets.data() + first, sequences->entries, NULL, maxFetchSize);
        return true;
    };

    std::mutex blockCopyMutex;
    int blockCopyError = 0;
    double blockCopySeconds = 0;
    size_t cachedCopySize = 0;
    std::mutex clusterMutex;
    // clustering step of a single node
    const auto clusterStep = [&]() {
        std::unique_lock<std::mutex> taskLock(clusterMutex, std::try_to_lock);
        return taskLock.owns_lock() && clustering != NULL && clustering->step() == AcceptedPairClustering::STEP_WORKED;
    };
    // work the waiting main thread needs, else work other nodes need
    const auto runWaitTasks = [&]() {
        return receiveSequences() || joinSharedLoop() || copyOwnKeysAhead() || (memberExchange != NULL && memberExchange->serveRequests()) || serveSequences() || clusterStep();
    };
    const auto runIoLoop = [&](bool isPrefilterReader) {
        for (unsigned int polls = 0;;) {
            uint64_t seenWork;
            {
                std::lock_guard<std::mutex> lock(serviceMutex);
                if (isAligningDone && isReceiveDone && isServeDone && isPrefilterDone && (memberExchange == NULL || (memberExchange->isServed() && memberExchange->isPreloaded()))
                    && (clustering == NULL || clustering->isDone)) {
                    return;
                }
                seenWork = ioWork;
            }
            // I/O tasks in window order, a block copy only if none worked
            bool hasWorked = memberExchange != NULL && memberExchange->serveRequests();
            hasWorked |= clusterStep();
            hasWorked |= serveSequences();
            hasWorked |= receiveSequences();
            hasWorked |= joinSharedLoop();
            hasWorked |= copyOwnKeysAhead();
            hasWorked |= memberExchange != NULL && memberExchange->preloadNextRange();
            hasWorked |= fillNextPrefilterBuffer(false, isPrefilterReader);
            // copy the next reserved block, used once complete, a write error ends the copies
            if (hasWorked == false) {
                std::unique_lock<std::mutex> copyLock(blockCopyMutex, std::try_to_lock);
                if (copyLock.owns_lock() && copiedBlockCount != reservedBlockCount && blockCopyError == 0) {
                    Timer copyTimer;
                    size_t file = servedBlockCopy.blockFiles[copiedBlockCount];
                    size_t begin = servedBlockCopy.blockBegins[copiedBlockCount];
                    size_t bytes = servedBlockCopy.blockOffsets[copiedBlockCount + 1] - servedBlockCopy.blockOffsets[copiedBlockCount];
                    const std::string &copy = (*servedBlockCopy.copies)[firstBlockCopy + file];
                    // copy a cached block from memory, a waiting load waits for this copy at most
                    bool isCached = false;
                    {
                        std::unique_lock<std::mutex> sourceLock(sourceMutex, std::try_to_lock);
                        isCached = sourceLock.owns_lock() && seqDbr.copyCachedBlockData(servedBlockCopy.blocks[copiedBlockCount], begin, bytes, blockCopyBuffer.data());
                    }
                    int writeError = 0;
                    if (isCached == false) {
                        blockCopyError = FileUtil::copyRange(blockSourceFds[file], servedBlockCopy.dataFiles[file].c_str(), blockCopyFds[file], blockCopyBuffer.data(), bytes, begin);
                    } else if (FileUtil::tryPwriteAll(blockCopyFds[file], blockCopyBuffer.data(), bytes, begin, writeError) < bytes) {
                        blockCopyError = (writeError != 0) ? writeError : EIO;
                    }
                    if (blockCopyError == 0) {
                        cachedCopySize += isCached ? bytes : 0;
                        // load plans and reads its blocks under sourceMutex
                        std::lock_guard<std::mutex> sourceLock(sourceMutex);
                        seqDbr.setLocalBlock(servedBlockCopy.blocks[copiedBlockCount]);
                        copiedBlockCount++;
                    } else {
                        Debug(Debug::WARNING) << "Cannot write " << copy << " at offset " << begin << ": " << strerror(blockCopyError) << ", the blocks of this node not copied yet are read from the database\n";
                    }
                    blockCopySeconds += copyTimer.getTimediff();
                    hasWorked = true;
                }
            }
            if (hasWorked) {
                polls = 0;
                continue;
            }
            // sleep 10 ms doubling to 20 ms if others may wait, 160 ms if a fetch waits, else 1.28 s
            std::unique_lock<std::mutex> lock(serviceMutex);
            unsigned int maxShift = (alignNodeCount > 1 || (memberExchange != NULL && memberExchange->isServed() == false)) ? 1 : (isImportUrgent ? 4 : 7);
            // idle clustering step is due again after its sleep
            unsigned int clusterSleep = (clustering != NULL && clustering->isDone == false) ? clustering->idleMicroseconds.load() : UINT_MAX;
            serviceCondition.wait_for(lock, std::chrono::microseconds(std::min(10000u << std::min(polls++, maxShift), clusterSleep)), [&]() { return ioWork != seenWork || (isPrefilterReader && isPrefilterWaited.load()); });
        }
    };

    const auto waitForSequences = [&](FetchedSequences &fetched, size_t doneRepOrderKey) {
        std::unique_lock<std::mutex> lock(serviceMutex);
        if (fetched.isReady) {
            return;
        }
        lock.unlock();
        if (publishThread.joinable()) {
            publishThread.join();
        }
        publishAccepted(doneRepOrderKey, acceptedSize);
        publishedSize = acceptedSize;
        publishTime = startTimer.getTimediff();
        Timer waitTimer;
        lock.lock();
        // later fetch at the gate is no wait for the master
        bool waitsForMaster = isWaitingForMaster && handedOffFetches.empty() == false && handedOffFetches.front() == &fetched;
        for (unsigned int polls = 0; fetched.isReady == false;) {
            Timer pollTimer;
            uint64_t seenWork = ioWork;
            lock.unlock();
            teamReceiveState = 2;
            // run the work it waits for, else work others wait for
            bool hasWorked = runWaitTasks();
            lock.lock();
            if (hasWorked == false) {
                serviceCondition.wait_for(lock, std::chrono::microseconds(10000u << std::min(polls++, 1u)), [&]() { return fetched.isReady || ioWork != seenWork; });
            }
            sequenceWaitParts[teamReceiveState] += waitsForMaster ? 0 : pollTimer.getTimediff();
        }
        (waitsForMaster ? waitSeconds : sequenceWaitSeconds) += waitTimer.getTimediff();
    };
    // poll member answers, run the wait tasks in between
    const auto receiveMembers = [&]() {
        while (memberExchange->tryReceive() == false) {
            Timer pollTimer;
            for (bool hasWorked = true; hasWorked && pollTimer.getTimediff() < 1.0 / PROGRESS_READS_PER_SECOND;) {
                hasWorked = runWaitTasks();
            }
            double waitSeconds = 1.0 / PROGRESS_READS_PER_SECOND - pollTimer.getTimediff();
            if (waitSeconds > 0) {
                usleep(static_cast<useconds_t>(waitSeconds * 1000000));
            }
        }
    };

    std::vector<std::thread> ioThreads;
    for (int i = 0; isAligningNode && i < ioThreadCount; i++) {
        // first I/O thread is the prefilter reader
        ioThreads.emplace_back([&, i]() {
#ifdef OPENMP
            omp_set_num_threads(1);
#endif
            runIoLoop(i == 0);
        });
    }
    if (isAligningNode) {
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            currentSequences.acceptedSize = acceptedSize;
            handedOffFetches.push_back(&currentSequences);
            ioWork++;
        }
        serviceCondition.notify_all();
        copyOwnKeys(currentSequences);
        waitForSequences(currentSequences, getFirstRepKey(currentSequences));
    }

    double firstFetchSeconds = startTimer.getTimediff();
    size_t alignedFetchCount = 0;
    // planned bytes per aligning second of the last planned fetch
    double plannedBytesPerSecond = 0;
    double maxFetchAheadSeconds = 0;
    double fetchAheadSecondsSum = 0;
    size_t alignThreadCount = par.threads - ioThreadCount;
    while (currentSequences.isLast == false) {
        // fetch ahead within fetchAheadSeconds, memory and the current buffer
        for (;;) {
            bool isFirstAhead = fetchesInFlight.empty();
            if (isFirstAhead == false) {
                const FetchedSequences &newest = *fetchesInFlight.back();
                size_t inFlightSize = 0;
                for (size_t i = 0; i < fetchesInFlight.size(); i++) {
                    inFlightSize += fetchesInFlight[i]->plannedSize;
                }
                // more pending pairs come only after aligning
                if (newest.isLast || (newest.end == newest.begin && newest.pendingPairCount == 0) || idleFetches.empty()
                    || inFlightSize >= fetchAheadSeconds.load() * plannedBytesPerSecond) {
                    break;
                }
            }
            // next fetch holds pending pairs up to the limit and a record
            size_t limitedFetchSize = (memberDbr != NULL) ? std::max(fetchBytesLimit, maxPendingPairSize) + prefilterReader.maxRecordSize
                                                                : std::max(fetchBytesLimit, prefilterReader.maxRecordSize);
            // free idle slot buffers until the next fetch fits, but not the next slot's
            size_t fetchMemory = currentSequences.memorySize();
            for (size_t i = 0; i < fetchSlots.size(); i++) {
                fetchMemory += fetchSlots[i].memorySize();
            }
            for (size_t i = 0; i + 1 < idleFetches.size() && fetchMemory + limitedFetchSize > fetchBudget; i++) {
                FetchedSequences &idle = *idleFetches[i];
                fetchMemory -= idle.memorySize();
                delete[] idle.entries;
                idle.entries = NULL;
                idle.allocatedSize = 0;
                std::vector<DBKeyType>().swap(idle.keys);
                std::vector<size_t>().swap(idle.entryOffsets);
                std::vector<size_t>().swap(idle.recordEntryOffsets);
                std::vector<size_t>().swap(idle.recordEntries);
            }
            if (isFirstAhead == false && fetchMemory + limitedFetchSize > fetchBudget) {
                break;
            }
            // next buffer must be filled, its next fill refills the one before current's
            if (isFirstAhead == false && prefilterRecordPos[prefilterBuffer] == prefilterRecords[prefilterBuffer].size() && prefilterRecords[prefilterBuffer].empty() == false) {
                std::lock_guard<std::mutex> lock(prefilterBufferMutex);
                if (filledPrefilterBufferCount <= usedPrefilterBufferCount || currentSequences.prefilterBufferNumber + 1 < usedPrefilterBufferCount) {
                    break;
                }
            }
            FetchedSequences &selected = *idleFetches.back();
            idleFetches.pop_back();
            selectRecordKeys(selected, currentSequences);
            {
                std::lock_guard<std::mutex> lock(serviceMutex);
                // reused slot holds the state of its last fetch
                selected.isReady = false;
                selected.acceptedSize = acceptedSize;
                handedOffFetches.push_back(&selected);
                ownKeyFetches.push_back(&selected);
                ioWork++;
            }
            serviceCondition.notify_all();
            fetchesInFlight.push_back(&selected);
        }
        maxFetchAheadSeconds = std::max(maxFetchAheadSeconds, fetchAheadSeconds.load());
        fetchAheadSecondsSum += fetchAheadSeconds.load();
        FetchedSequences &nextSequences = *fetchesInFlight.front();
        Timer alignTimer;
        size_t nextRepKey = getFirstRepKey(nextSequences);
        const std::vector<RepTargetsRecord> &records = prefilterRecords[currentSequences.prefilterBuffer];
        size_t recordCount = currentSequences.end - currentSequences.begin;
        // batches of about equal alignment cells, a large rep spans several
        size_t maxBatches = alignThreadCount * (isSwipeEnabled ? 64 * 16 / SwipeExtender::LANES : 64);
        if (isSwipeEnabled) {
            // estimate the SWIPE jobs of this fetch
            size_t swipeJobCount = 0;
            for (int thread = 0; thread < par.threads; thread++) {
                swipeJobCount += (threadTargetAligners[thread] != NULL) ? threadTargetAligners[thread]->swipeTargets.targetPairs : 0;
            }
            size_t fetchSwipeJobs = static_cast<size_t>(static_cast<uint128>(currentSequences.targetCount) * swipeJobCount / std::max((size_t) 1, examinedTargetPairCount.load()));
            size_t swipeBatches = std::max(8 * alignThreadCount, fetchSwipeJobs / (64 * SwipeExtender::LANES));
            maxBatches = std::min(maxBatches, swipeBatches);
        }
        recordAlignmentCells.resize(recordCount);
        // reps of a size class ascend
#pragma omp parallel
        {
            size_t repRange = SIZE_MAX;
#pragma omp for schedule(static)
            for (size_t i = 0; i < recordCount; i++) {
                const RepTargetsRecord &record = records[currentSequences.begin + i];
                size_t repLength = seqDbr.getSeqLenFromRange(repRange, record.repOrderKeyFirstTarget.repKey % paddedDbSize);
                // readSize holds the rep and target entries
                recordAlignmentCells[i] = repLength * (record.readSize - (repLength + 2 + FETCHED_KEY_SIZE) - record.targetCount * (2 + FETCHED_KEY_SIZE));
            }
        }
        uint128 fetchAlignmentCells = 0;
        for (size_t i = 0; i < recordCount; i++) {
            fetchAlignmentCells += recordAlignmentCells[i];
        }

        batchOffsets.assign(1, currentSequences.begin);
        batchFirstTargets.assign(1, 0);
        uint128 cells = 0;
        size_t repRange = SIZE_MAX;
        for (size_t i = currentSequences.begin; i < currentSequences.end; i++) {
            uint128 recordEndCells = cells + recordAlignmentCells[i - currentSequences.begin];
            if (recordEndCells * maxBatches > fetchAlignmentCells * batchOffsets.size()) {
                const RepTargetsRecord &record = records[i];
                const DBKeyType *targets = prefilterTargets[currentSequences.prefilterBuffer].data() + record.targetOffset;
                size_t repLength = seqDbr.getSeqLenFromRange(repRange, record.repOrderKeyFirstTarget.repKey % paddedDbSize);
                size_t targetRange = SIZE_MAX;
                for (unsigned int j = 0; j + 1 < record.targetCount && recordEndCells * maxBatches > fetchAlignmentCells * batchOffsets.size(); j++) {
                    cells += repLength * seqDbr.getSeqLenFromRange(targetRange, targets[j]);
                    if (cells * maxBatches >= fetchAlignmentCells * batchOffsets.size()) {
                        batchOffsets.push_back(i);
                        batchFirstTargets.push_back(j + 1);
                    }
                }
            }
            cells = recordEndCells;
            if (i + 1 < currentSequences.end && cells * maxBatches >= fetchAlignmentCells * batchOffsets.size()) {
                batchOffsets.push_back(i + 1);
                batchFirstTargets.push_back(0);
            }
        }
        batchOffsets.push_back(currentSequences.end);
        batchFirstTargets.push_back(0);
        size_t batchCount = batchOffsets.size() - 1;
        batchDone.assign(batchCount, 0);
        batchPairs.assign(batchCount, std::vector<ClusterPair>());
        size_t firstUnfinishedBatch = 0;
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            alignmentThread = thread_idx;
            std::vector<ClusterPair> pairs;
#pragma omp for schedule(dynamic, 1)
            for (size_t batch = 0; batch < batchCount; batch++) {
                {
                    std::unique_lock<std::mutex> assignedKeysLock(assignedKeysMutex, std::try_to_lock);
                    if (assignedKeysLock.owns_lock() && assignedKeysTimer.getTimediff() - lastBatchReadTime >= batchReadPeriod) {
                        Timer readTimer;
                        readAssignedKeys(false);
                        lastBatchReadTime = assignedKeysTimer.getTimediff();
                        batchReadCount++;
                        batchReadSeconds += readTimer.getTimediff();
                        batchReadPeriod = std::min(BATCH_READ_SECONDS, std::max(MIN_READ_SECONDS, LOG_COST_FACTOR * batchReadSeconds / batchReadCount));
                    }
                }
                // without I/O threads run the I/O tasks between batches
                if (ioThreadCount == 0 && assignedKeysTimer.getTimediff() - lastServeTime >= BATCH_READ_SECONDS) {
                    lastServeTime = assignedKeysTimer.getTimediff();
                    if (memberExchange != NULL) {
                        memberExchange->serveRequests();
                    }
                    serveSequences();
                    receiveSequences();
                    clusterStep();
                }
                pairs.clear();
                TargetAligners *aligners = threadTargetAligners[thread_idx];
                size_t examinedCount = 0;
                for (size_t i = batchOffsets[batch]; i < batchOffsets[batch + 1] + (batchFirstTargets[batch + 1] > 0); i++) {
                    const RepTargetsRecord &record = records[i];
                    unsigned int firstTarget = (i == batchOffsets[batch]) ? batchFirstTargets[batch] : 0;
                    unsigned int endTarget = (i == batchOffsets[batch + 1]) ? batchFirstTargets[batch + 1] : record.targetCount;
                    examinedCount += endTarget - firstTarget;
                    size_t repKey = record.repOrderKeyFirstTarget.repKey % paddedDbSize;
                    if (isAssigned(record.repOrderKeyFirstTarget.repKey)) {
                        continue;
                    }
                    const size_t *recordEntries = currentSequences.recordEntries.data() + currentSequences.recordEntryOffsets[i - currentSequences.begin];
                    size_t repLength = 0;
                    const char *repData = getRecordSequence(currentSequences, recordEntries[0], &repLength);
                    if (repData == NULL) {
                        continue;
                    }
                    bool isRepMapped = false;
                    bool blockAlignerInitialized = false;
                    const DBKeyType *targets = prefilterTargets[currentSequences.prefilterBuffer].data() + record.targetOffset;
                    for (unsigned int j = firstTarget; j < endTarget && isAssigned(record.repOrderKeyFirstTarget.repKey) == false; j++) {
                        DBKeyType targetKey = targets[j];
                        if (isAssigned(targetKey)) {
                            continue;
                        }
                        size_t targetLength = 0;
                        const char *targetData = getRecordSequence(currentSequences, recordEntries[1 + j], &targetLength);
                        if (targetData == NULL) {
                            continue;
                        }
                        // alignTarget checks the coverage first
                        if (Util::canBeCovered(par.covThr, par.covMode, static_cast<float>(repLength), static_cast<float>(targetLength)) == false) {
                            continue;
                        }
                        if (aligners == NULL) {
                            aligners = new TargetAligners(par, scoring, maxSeqLen);
                            threadTargetAligners[thread_idx] = aligners;
                        }
                        if (isRepMapped == false) {
                            aligners->query.mapSequence(repKey, repKey, repData, repLength);
                            aligners->blockAligner.setQuery(&aligners->query);
                            aligners->swipeTargets.startRep(record.repOrderKeyFirstTarget.repKey);
                            isRepMapped = true;
                        }
                        aligners->target.mapSequence(targetKey, targetKey, targetData, targetLength);
                        unsigned short diagonal = 0;
                        if (alignTarget(par, aligners->blockAligner, blockAlignerInitialized, aligners->query, aligners->target, scoring.scorePerColThreshold, diagonal, NULL, 0, NULL,
                                        isSwipeEnabled ? &aligners->swipeTargets : NULL)) {
                            const ClusterPair pair = { record.repOrderKeyFirstTarget.repKey, targetKey };
                            pairs.push_back(pair);
                        }
                    }
                }
                examinedTargetPairCount += examinedCount;
                if (aligners != NULL) {
                    alignTargetsSwipe(par, aligners->swipeTargets, isAssigned, pairs);
                    aligners->swipeTargets.clearTargets();
                }
                // lanes finish the pairs out of rep order
                SORT_SERIAL(pairs.begin(), pairs.end(), ClusterPair::compareByRep);
                // log the pairs in batch order
                bool needsKeyRead = false;
                size_t doneRepOrderKey = 0;
                size_t doneSize = 0;
                std::unique_lock<std::mutex> publishLock(publishMutex, std::defer_lock);
                {
                    std::lock_guard<std::mutex> lock(acceptedMutex);
                    batchPairs[batch].swap(pairs);
                    batchDone[batch] = 1;
                    if (memberDbr == NULL) {
                        for (; firstUnfinishedBatch < batchCount && batchDone[firstUnfinishedBatch]; firstUnfinishedBatch++) {
                            logAcceptedPairs(batchPairs[firstUnfinishedBatch].data(), batchPairs[firstUnfinishedBatch].size());
                            std::vector<ClusterPair>().swap(batchPairs[firstUnfinishedBatch]);
                        }
                        double now = startTimer.getTimediff();
                        // one thread publishes at a time
                        if ((acceptedSize - publishedSize >= 4 * 1024 * 1024 || now - publishTime >= publishPeriod.load()) && publishLock.try_lock()) {
                            doneRepOrderKey = (firstUnfinishedBatch < batchCount) ? static_cast<size_t>(records[batchOffsets[firstUnfinishedBatch]].repOrderKeyFirstTarget.repKey) : nextRepKey;
                            doneSize = acceptedSize;
                            publishedSize = acceptedSize;
                            publishTime = now;
                            needsKeyRead = doneRepOrderKey < orderKeyEnd;
                        }
                        if (needsKeyRead || now - assignedKeysTime >= READ_SECONDS) {
                            needsKeyRead = true;
                            assignedKeysTime = now;
                        }
                    }
                }
                if (publishLock.owns_lock()) {
                    publishAccepted(doneRepOrderKey, doneSize);
                    publishLock.unlock();
                }
                if (needsKeyRead) {
                    std::unique_lock<std::mutex> assignedKeysLock(assignedKeysMutex, std::try_to_lock);
                    if (assignedKeysLock.owns_lock()) {
                        readAssignedKeys(true);
                        lastBatchReadTime = assignedKeysTimer.getTimediff();
                    }
                }
            }
            // aligners stay, their batch buffers go
            if (threadTargetAligners[thread_idx] != NULL) {
                threadTargetAligners[thread_idx]->swipeTargets.release();
            }
        }

        if (memberDbr != NULL) {
            // pairs assigned by now skip their member checks
            {
                std::unique_lock<std::mutex> assignedKeysLock(assignedKeysMutex, std::try_to_lock);
                if (assignedKeysLock.owns_lock()) {
                    readAssignedKeys(false);
                }
            }
            // size the entries of the new pairs during the member checks
            size_t sizedCount = acceptedEntryBytes.size();
            for (size_t batch = 0; batch < batchCount; batch++) {
                acceptedPairs.insert(acceptedPairs.end(), batchPairs[batch].begin(), batchPairs[batch].end());
            }
            Timer memberRequestTimer;
            lookupKeys.clear();
            for (size_t i = sizedCount; i < acceptedPairs.size(); i++) {
                if (isAssigned(acceptedPairs[i].repKey) == false && isAssigned(acceptedPairs[i].memberKey) == false) {
                    lookupKeys.push_back(acceptedPairs[i].memberKey);
                }
            }
            SORT_PARALLEL(lookupKeys.begin(), lookupKeys.end());
            lookupKeys.erase(std::unique(lookupKeys.begin(), lookupKeys.end()), lookupKeys.end());
            memberExchange->request(lookupKeys, NULL, NULL, 0, 0);
            memberLookupSeconds += memberRequestTimer.getTimediff();
            // check this fetch's pending pairs, an unserved rep is assigned
            size_t pairCount = currentSequences.pendingPairCount;
            membersPassed.assign(pairCount, 0);
            // pair that already failed does not align its remaining parts
            size_t failedPartEnd = isPendingPairFailing ? getPairPartEnd(pendingPairIsLastPart, pairCount, 0) : 0;
            // pairs of a rep share the query
            repPairOffsets.clear();
            for (size_t i = 0; i < pairCount; i++) {
                if (i == 0 || pendingPairs[i].repKey != pendingPairs[i - 1].repKey) {
                    repPairOffsets.push_back(i);
                }
            }
            repPairOffsets.push_back(pairCount);
            size_t repCount = repPairOffsets.size() - 1;
            // rep batches that fill the lanes, 4 per thread
            size_t batchMemberLimit = std::min<size_t>(4096, std::max<size_t>(pendingMemberOffsets[pairCount] / (par.threads * 4), 256));
            batchRepOffsets.assign(1, 0);
            for (size_t rep = 0, batchMembers = 0; rep < repCount; rep++) {
                batchMembers += pendingMemberOffsets[repPairOffsets[rep + 1]] - pendingMemberOffsets[repPairOffsets[rep]];
                if (isSwipeEnabled == false || batchMembers >= batchMemberLimit || rep + 1 == repCount) {
                    batchRepOffsets.push_back(rep + 1);
                    batchMembers = 0;
                }
            }
            size_t memberBatchCount = batchRepOffsets.size() - 1;
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                alignmentThread = thread_idx;
                std::vector<size_t> repMemberCheckOffsets;
                std::vector<DBKeyType> repKeys;
#pragma omp for schedule(dynamic, 1)
                for (size_t batch = 0; batch < memberBatchCount; batch++) {
                    if (threadMemberAligners[thread_idx] == NULL) {
                        threadMemberAligners[thread_idx] = new MemberAligners(par, scoring, maxMemberSeqLen, isSwipeEnabled);
                    }
                    MemberAligners &aligners = *threadMemberAligners[thread_idx];
                    aligners.memberChecks.checkCount = 0;
                    repMemberCheckOffsets.assign(1, 0);
                    repKeys.clear();
                    for (size_t rep = batchRepOffsets[batch]; rep < batchRepOffsets[batch + 1]; rep++) {
                        // one check for all parts of a pair
                        for (size_t i = repPairOffsets[rep]; i < repPairOffsets[rep + 1]; i = getPairPartEnd(pendingPairIsLastPart, pairCount, i)) {
                            const ClusterPair &pair = pendingPairs[i];
                            if (i < failedPartEnd || isAssigned(pair.repKey) || isAssigned(pair.memberKey) || getFetchedSequence(currentSequences, pair.repKey % paddedDbSize) == NULL) {
                                continue;
                            }
                            MemberCheck &check = aligners.memberChecks.addCheck(i, 0);
                            size_t memberEnd = pendingMemberOffsets[getPairPartEnd(pendingPairIsLastPart, pairCount, i)];
                            for (size_t member = pendingMemberOffsets[i]; member < memberEnd; member++) {
                                check.memberKeys.push_back(pendingMembers[member]);
                                check.memberData.push_back(getFetchedEntry(currentSequences, pendingMembers[member]));
                                check.isLaneMember.push_back(pendingMemberIsLane[member]);
                            }
                        }
                        if (aligners.memberChecks.checkCount > repMemberCheckOffsets.back()) {
                            repMemberCheckOffsets.push_back(aligners.memberChecks.checkCount);
                            repKeys.push_back(pendingPairs[repPairOffsets[rep]].repKey % paddedDbSize);
                        }
                    }
                    // without SWIPE a batch holds one rep's checks
                    if (isSwipeEnabled) {
                        alignMemberChecksSwipe(par, aligners.blockAligner, aligners.matcher, aligners.query, aligners.element, memberDbr, swMode, thread_idx, aligners.memberChecks, repMemberCheckOffsets, repKeys, mapRep);
                    } else if (aligners.memberChecks.checkCount > 0) {
                        mapRep(aligners.query, aligners.blockAligner, repKeys[0]);
                        bool matcherInitialized = false;
                        alignClusterMembers(par, aligners.blockAligner, aligners.matcher, matcherInitialized, aligners.query, aligners.element, memberDbr, swMode, thread_idx, false, NULL, aligners.memberChecks);
                    }
                    for (size_t g = 0; g < aligners.memberChecks.checkCount; g++) {
                        const MemberCheck &check = aligners.memberChecks.checks[g];
                        size_t partEnd = getPairPartEnd(pendingPairIsLastPart, pairCount, check.targetId);
                        for (size_t i = check.targetId; i < partEnd; i++) {
                            membersPassed[i] = check.isPassing;
                        }
                    }
                }
                if (threadMemberAligners[thread_idx] != NULL) {
                    threadMemberAligners[thread_idx]->memberChecks.release();
                    threadMemberAligners[thread_idx]->swipeMembers.release();
                    threadMemberAligners[thread_idx]->memberTargets.release();
                }
            }
            // pair passes with its last part if no part failed, parts span fetches
            size_t acceptedCount = 0;
            for (size_t i = 0; i < pairCount; i++) {
                isPendingPairFailing |= membersPassed[i] == 0;
                if (pendingPairIsLastPart[i]) {
                    if (isPendingPairFailing == false) {
                        pendingPairs[acceptedCount++] = pendingPairs[i];
                    }
                    isPendingPairFailing = false;
                    isPendingPairOpen = false;
                }
            }
            if (pairCount > 0) {
                isPendingPairOpen = pendingPairIsLastPart[pairCount - 1] == 0;
                // pairs above are written before an open last part only
                openPendingPair = pendingPairs[pairCount - 1];
            }
            logAcceptedPairs(pendingPairs.data(), acceptedCount);
            size_t checkedMembers = pendingMemberOffsets[pairCount];
            pendingPairs.erase(pendingPairs.begin(), pendingPairs.begin() + pairCount);
            pendingPairSize.erase(pendingPairSize.begin(), pendingPairSize.begin() + pairCount);
            pendingPairIsLastPart.erase(pendingPairIsLastPart.begin(), pendingPairIsLastPart.begin() + pairCount);
            pendingMemberOffsets.erase(pendingMemberOffsets.begin(), pendingMemberOffsets.begin() + pairCount);
            pendingMembers.erase(pendingMembers.begin(), pendingMembers.begin() + checkedMembers);
            pendingMemberIsLane.erase(pendingMemberIsLane.begin(), pendingMemberIsLane.begin() + checkedMembers);
            for (size_t i = 0; i < pendingMemberOffsets.size(); i++) {
                pendingMemberOffsets[i] -= checkedMembers;
            }

            // accepted pairs wait for a later fetch behind earlier ones
            Timer memberLookupTimer;
            receiveMembers();
            memberExchange->getEntries(lookupEntryOffsets, lookupEntryBytes);
            acceptedEntryOffsets.resize(acceptedPairs.size(), 0);
            acceptedEntryBytes.resize(acceptedPairs.size(), 0);
            for (size_t i = sizedCount; i < acceptedPairs.size(); i++) {
                size_t lookupKey = std::lower_bound(lookupKeys.begin(), lookupKeys.end(), acceptedPairs[i].memberKey) - lookupKeys.begin();
                if (lookupKey < lookupKeys.size() && lookupKeys[lookupKey] == acceptedPairs[i].memberKey) {
                    acceptedEntryOffsets[i] = lookupEntryOffsets[lookupKey];
                    acceptedEntryBytes[i] = lookupEntryBytes[lookupKey];
                }
            }
            // e entry bytes hold < e / 2 members and at most as many parts
            size_t freeKeys = memberKeyBudget - std::min(memberKeyBudget, pendingMembers.size() + pendingPairs.size());
            size_t plannedCount = 0;
            for (size_t keyBound = 0; isSlicingPair == false && plannedCount < acceptedPairs.size(); plannedCount++) {
                size_t pairBound = std::max<size_t>(1, acceptedEntryBytes[plannedCount]);
                if (keyBound + pairBound > freeKeys) {
                    break;
                }
                keyBound += pairBound;
            }
            pairPartCounts.resize(plannedCount);
            needsMemberCheck.resize(plannedCount);
            lookupKeyPairs.clear();
            for (size_t i = 0; i < plannedCount; i++) {
                // assigned key or an uncoverable member fails without a check
                needsMemberCheck[i] = isAssigned(acceptedPairs[i].repKey) == false && isAssigned(acceptedPairs[i].memberKey) == false;
                if (needsMemberCheck[i]) {
                    lookupKeyPairs.emplace_back(acceptedPairs[i].memberKey, i);
                }
            }
            SORT_PARALLEL(lookupKeyPairs.begin(), lookupKeyPairs.end());
            lookupKeys.clear();
            lookupEntryOffsets.clear();
            lookupEntryBytes.clear();
            for (size_t j = 0; j < lookupKeyPairs.size(); j++) {
                if (lookupKeys.empty() || lookupKeys.back() != lookupKeyPairs[j].first) {
                    lookupKeys.push_back(lookupKeyPairs[j].first);
                    lookupEntryOffsets.push_back(acceptedEntryOffsets[lookupKeyPairs[j].second]);
                    lookupEntryBytes.push_back(acceptedEntryBytes[lookupKeyPairs[j].second]);
                }
            }
            memberExchange->request(lookupKeys, lookupEntryOffsets.data(), lookupEntryBytes.data(), 0, SIZE_MAX);
            receiveMembers();
            memberExchange->getMembers(lookupMemberOffsets, lookupMembers);
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                threadKeys[thread_idx].clear();
                threadMemberIsLane[thread_idx].clear();
                threadPartSizes[thread_idx].clear();
                threadPartMemberCounts[thread_idx].clear();
                ScalarMembers scalarMembers;
                std::vector<size_t> scalarPositions;
#pragma omp for schedule(static)
                for (size_t i = 0; i < plannedCount; i++) {
                    size_t memberBegin = threadKeys[thread_idx].size();
                    size_t partBegin = threadPartSizes[thread_idx].size();
                    size_t repKey = acceptedPairs[i].repKey % paddedDbSize;
                    size_t repSeqLen = seqDbr.getSeqLen(repKey);
                    size_t repSize = repSeqLen + 2 + FETCHED_KEY_SIZE;
                    size_t member = 0;
                    size_t memberEnd = 0;
                    if (needsMemberCheck[i]) {
                        size_t lookupKey = std::lower_bound(lookupKeys.begin(), lookupKeys.end(), acceptedPairs[i].memberKey) - lookupKeys.begin();
                        member = lookupMemberOffsets[lookupKey];
                        memberEnd = lookupMemberOffsets[lookupKey + 1];
                    }
                    // entry members ascend, the range cursor moves forward
                    size_t memberRange = SIZE_MAX;
                    size_t partSize = repSize;
                    size_t partMembers = 0;
                    scalarMembers.clear();
                    while (needsMemberCheck[i] && member < memberEnd) {
                        DBKeyType memberKey = lookupMembers[member++];
                        size_t memberSeqLen = seqDbr.getSeqLenFromRange(memberRange, memberKey);
                        needsMemberCheck[i] = canCoverMember(par, static_cast<int>(repSeqLen), static_cast<int>(memberSeqLen));
                        if (needsMemberCheck[i] == false) {
                            break;
                        }
                        size_t memberSize = memberSeqLen + 2 + FETCHED_KEY_SIZE;
                        // one member per part and one pending pair per fetch at least
                        if (partMembers > 0 && partSize + memberSize > maxPendingPairSize) {
                            threadPartSizes[thread_idx].push_back(partSize);
                            threadPartMemberCounts[thread_idx].push_back(partMembers);
                            partSize = repSize;
                            partMembers = 0;
                        }
                        threadKeys[thread_idx].push_back(memberKey);
                        threadMemberIsLane[thread_idx].push_back(1);
                        scalarMembers.add(std::abs(static_cast<int>(memberSeqLen) - static_cast<int>(repSeqLen)));
                        partSize += memberSize;
                        partMembers++;
                    }
                    if (needsMemberCheck[i]) {
                        threadPartSizes[thread_idx].push_back(partSize);
                        threadPartMemberCounts[thread_idx].push_back(partMembers);
                        scalarMembers.getPositions(scalarPositions);
                        for (size_t p = 0; p < scalarPositions.size(); p++) {
                            threadMemberIsLane[thread_idx][memberBegin + scalarPositions[p]] = 0;
                        }
                    } else {
                        threadKeys[thread_idx].resize(memberBegin);
                        threadMemberIsLane[thread_idx].resize(memberBegin);
                        threadPartSizes[thread_idx].resize(partBegin);
                        threadPartMemberCounts[thread_idx].resize(partBegin);
                    }
                    pairPartCounts[i] = threadPartSizes[thread_idx].size() - partBegin;
                }
            }
            partSizes.clear();
            partMemberCounts.clear();
            for (size_t thread = 0; thread < threadKeys.size(); thread++) {
                pendingMembers.insert(pendingMembers.end(), threadKeys[thread].begin(), threadKeys[thread].end());
                pendingMemberIsLane.insert(pendingMemberIsLane.end(), threadMemberIsLane[thread].begin(), threadMemberIsLane[thread].end());
                partSizes.insert(partSizes.end(), threadPartSizes[thread].begin(), threadPartSizes[thread].end());
                partMemberCounts.insert(partMemberCounts.end(), threadPartMemberCounts[thread].begin(), threadPartMemberCounts[thread].end());
            }
            for (size_t i = 0, part = 0; i < plannedCount; i++) {
                for (size_t p = 0; p < pairPartCounts[i]; p++, part++) {
                    pendingPairs.push_back(acceptedPairs[i]);
                    pendingPairSize.push_back(partSizes[part]);
                    pendingMemberOffsets.push_back(pendingMemberOffsets.back() + partMemberCounts[part]);
                    pendingPairIsLastPart.push_back(p + 1 == pairPartCounts[i]);
                }
            }
            acceptedPairs.erase(acceptedPairs.begin(), acceptedPairs.begin() + plannedCount);
            acceptedEntryOffsets.erase(acceptedEntryOffsets.begin(), acceptedEntryOffsets.begin() + plannedCount);
            acceptedEntryBytes.erase(acceptedEntryBytes.begin(), acceptedEntryBytes.begin() + plannedCount);
            const auto addPendingPart = [&](const ClusterPair &pair, size_t partSize, bool isLastPart) {
                pendingPairs.push_back(pair);
                pendingPairSize.push_back(partSize);
                pendingMemberOffsets.push_back(pendingMembers.size());
                pendingPairIsLastPart.push_back(isLastPart);
            };
            // pair too large for the budget goes alone, sliced by free keys
            if (isSlicingPair == false && plannedCount == 0 && acceptedPairs.empty() == false && acceptedEntryBytes[0] > memberKeyBudget) {
                isSlicingPair = true;
                slicedConsumedBytes = 0;
                slicedMemberPosition = 0;
                slicedRepSeqLen = seqDbr.getSeqLen(acceptedPairs[0].repKey % paddedDbSize);
                // first read finds the lane outsiders, failed pairs stop early
                ScalarMembers scalarMembers;
                bool needsCheck = isAssigned(acceptedPairs[0].repKey) == false && isAssigned(acceptedPairs[0].memberKey) == false;
                lookupKeys.assign(1, acceptedPairs[0].memberKey);
                for (size_t consumedBytes = 0; needsCheck && consumedBytes < acceptedEntryBytes[0];) {
                    memberExchange->request(lookupKeys, &acceptedEntryOffsets[0], &acceptedEntryBytes[0], consumedBytes, memberKeyBudget);
                    receiveMembers();
                    memberExchange->getMembers(lookupMemberOffsets, lookupMembers);
                    consumedBytes = memberExchange->getConsumedBytes();
                    size_t memberRange = SIZE_MAX;
                    for (size_t member = 0; needsCheck && member < lookupMembers.size(); member++) {
                        size_t memberSeqLen = seqDbr.getSeqLenFromRange(memberRange, lookupMembers[member]);
                        needsCheck = canCoverMember(par, static_cast<int>(slicedRepSeqLen), static_cast<int>(memberSeqLen));
                        scalarMembers.add(std::abs(static_cast<int>(memberSeqLen) - static_cast<int>(slicedRepSeqLen)));
                    }
                }
                scalarMembers.getPositions(slicedScalarPositions);
            }
            // slice of m members adds m + 1 parts at most
            if (isSlicingPair && freeKeys > 2) {
                const ClusterPair pair = acceptedPairs[0];
                size_t repSize = slicedRepSeqLen + 2 + FETCHED_KEY_SIZE;
                size_t partSize = repSize;
                size_t partMembers = 0;
                // assigned or failed pair ends with an empty part
                bool hasFailed = isPendingPairOpen && isPendingPairFailing && openPendingPair.repKey == pair.repKey && openPendingPair.memberKey == pair.memberKey;
                bool isEntryEnd = hasFailed || isAssigned(pair.repKey) || isAssigned(pair.memberKey);
                lookupMembers.clear();
                if (isEntryEnd == false) {
                    lookupKeys.assign(1, pair.memberKey);
                    memberExchange->request(lookupKeys, &acceptedEntryOffsets[0], &acceptedEntryBytes[0], slicedConsumedBytes, (freeKeys - 1) / 2);
                    receiveMembers();
                    memberExchange->getMembers(lookupMemberOffsets, lookupMembers);
                    slicedConsumedBytes = memberExchange->getConsumedBytes();
                    isEntryEnd = slicedConsumedBytes >= acceptedEntryBytes[0];
                }
                size_t memberRange = SIZE_MAX;
                for (size_t member = 0; member < lookupMembers.size(); member++) {
                    DBKeyType memberKey = lookupMembers[member];
                    size_t memberSeqLen = seqDbr.getSeqLenFromRange(memberRange, memberKey);
                    size_t memberSize = memberSeqLen + 2 + FETCHED_KEY_SIZE;
                    if (partMembers > 0 && partSize + memberSize > maxPendingPairSize) {
                        addPendingPart(pair, partSize, false);
                        partSize = repSize;
                        partMembers = 0;
                    }
                    pendingMembers.push_back(memberKey);
                    pendingMemberIsLane.push_back(std::find(slicedScalarPositions.begin(), slicedScalarPositions.end(), slicedMemberPosition++) == slicedScalarPositions.end());
                    partSize += memberSize;
                    partMembers++;
                    // uncoverable member fails its part and so the pair
                    if (canCoverMember(par, static_cast<int>(slicedRepSeqLen), static_cast<int>(memberSeqLen)) == false) {
                        isEntryEnd = true;
                        break;
                    }
                }
                if (partMembers > 0 || isEntryEnd) {
                    addPendingPart(pair, partSize, isEntryEnd);
                }
                if (isEntryEnd) {
                    isSlicingPair = false;
                    acceptedPairs.erase(acceptedPairs.begin());
                    acceptedEntryOffsets.erase(acceptedEntryOffsets.begin());
                    acceptedEntryBytes.erase(acceptedEntryBytes.begin());
                }
            }
            memberLookupSeconds += memberLookupTimer.getTimediff();
            if (publishThread.joinable()) {
                publishThread.join();
            }
            publishThread = std::thread(publishAccepted, getFirstUndoneRepKey(nextRepKey), acceptedSize);
            publishedSize = acceptedSize;
            publishTime = startTimer.getTimediff();
            if (nextRepKey < orderKeyEnd) {
                std::lock_guard<std::mutex> assignedKeysLock(assignedKeysMutex);
                readAssignedKeys(true);
            }
        }
        double fetchAlignSeconds = alignTimer.getTimediff();
        alignedCells += static_cast<double>(fetchAlignmentCells);
        alignSeconds += fetchAlignSeconds;
        alignedCellsPerSecond.store(alignedCells / std::max(alignSeconds, 1e-3));
        // fetch aligns in half of fetchAheadSeconds, two fetches in flight
        if (currentSequences.plannedSize > 0) {
            plannedBytesPerSecond = currentSequences.plannedSize / std::max(fetchAlignSeconds, 1e-3);
            maxFetchBytesLimit = std::max(minFetchSize, std::min(maxFetchSize, static_cast<size_t>(fetchAheadSeconds.load() / 2.0 * plannedBytesPerSecond)));
            fetchBytesLimit = std::min(fetchBytesLimit, maxFetchBytesLimit);
        }

        alignedFetchCount++;

        // reps of pending pairs are not done
        copyOwnKeys(nextSequences);
        waitForSequences(nextSequences, getFirstUndoneRepKey(nextRepKey));
        std::swap(currentSequences, nextSequences);
        fetchesInFlight.pop_front();
        idleFetches.push_back(&nextSequences);
    }
    for (int thread = 0; thread < par.threads; thread++) {
        delete threadTargetAligners[thread];
        delete threadMemberAligners[thread];
    }
    if (memberExchange != NULL && isAligningNode && nodes > 1) {
        memberExchange->finish();
    }
    if (publishThread.joinable()) {
        publishThread.join();
    }
    publishAccepted(orderKeyEnd, acceptedSize);
    if (fclose(acceptedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << acceptedFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    // serve until every aligning node is done
    size_t alignedCopiedBlockCount = 0;
    if (isAligningNode) {
        {
            std::lock_guard<std::mutex> sourceLock(sourceMutex);
            alignedCopiedBlockCount = copiedBlockCount;
        }
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            isAligningDone = true;
            ioWork++;
        }
        serviceCondition.notify_all();
        runIoLoop(ioThreadCount == 0);
        // I/O threads may sleep before they see the end
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            ioWork++;
        }
        serviceCondition.notify_all();
        for (size_t i = 0; i < ioThreads.size(); i++) {
            ioThreads[i].join();
        }
        printStatus();
        MMseqsNode::addPhaseValue(phaseValues, "Sequence data exchanged", "sent " + MMseqsNode::formatBytes(servedSize) + " for " + MMseqsNode::formatBytes(servedEntrySize) + " of sequences, received "
                                  + MMseqsNode::formatBytes(receivedSize));
    }
    delete[] currentSequences.entries;
    for (size_t i = 0; i < fetchSlots.size(); i++) {
        delete[] fetchSlots[i].entries;
    }
    for (size_t page = 0; page < assignedPages.size(); page++) {
        delete[] assignedPages[page];
    }
    MMseqsNode::addPhaseValue(phaseValues, "Sequence data read in blocks", MMseqsNode::formatBytes(seqDbr.getBlockReadSize()));
    MMseqsNode::addPhaseValue(phaseValues, "Sequence data read one at a time", MMseqsNode::formatBytes(seqDbr.getKeyReadSize()));
    MMseqsNode::addPhaseValue(phaseValues, "Sequence data read from --local-tmp in blocks", MMseqsNode::formatBytes(seqDbr.getLocalBlockReadSize()));
    MMseqsNode::addPhaseValue(phaseValues, "Sequence data read from --local-tmp one at a time", MMseqsNode::formatBytes(seqDbr.getLocalKeyReadSize()));
    if (isAligningNode) {
        size_t sharedKeyReadSize = seqDbr.getKeyReadSize() - seqDbr.getLocalKeyReadSize();
        MMseqsNode::addPhaseValue(phaseValues, "Sequence data read one at a time from the shared folder", MMseqsNode::formatBytes(sharedKeyReadSize) + ", "
                                  + formatRatio(sharedKeyReadSize, servedSequenceSize) + " times the " + MMseqsNode::formatBytes(servedSequenceSize) + " of sequences of this node");
        size_t ownBlockCount = 0;
        size_t touchedBlockCount = 0;
        size_t touchedSize = 0;
        for (size_t block = 0; block + 1 < blockKeyOffsets.size(); block++) {
            ownBlockCount += getBlockAlignNode(block) == rank;
            if (isOwnBlockTouched[block] == false) {
                continue;
            }
            touchedBlockCount++;
            for (size_t range = seqDbr.findIdRange(blockKeyOffsets[block]); range < rangeCount && static_cast<size_t>(index[range].id) < blockKeyOffsets[block + 1]; range++) {
                size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
                touchedSize += (std::min(keyEnd, blockKeyOffsets[block + 1]) - std::max(static_cast<size_t>(index[range].id), blockKeyOffsets[block])) * index[range].length;
            }
        }
        MMseqsNode::addPhaseValue(phaseValues, "Blocks of this node loaded", SSTR(touchedBlockCount) + " of " + SSTR(ownBlockCount) + " with " + MMseqsNode::formatBytes(touchedSize) + ", "
                                  + formatRatio(touchedSize, servedSequenceSize) + " of the sequences of this node");
        MMseqsNode::addPhaseValue(phaseValues, "Largest length range of one representative", MMseqsNode::formatBytes(maxCoverableSize) + " of the sequences of this node");
        MMseqsNode::addPhaseValue(phaseValues, "Sequences served that are longer than the representative this node aligns",
                                  SSTR(servedBehindKeyCount) + " with " + MMseqsNode::formatBytes(servedBehindSize));
    }
    for (size_t i = 0; i < blockCopyFds.size(); i++) {
        if (blockCopyFds[i] < 0) {
            continue;
        }
        const int sourceClosed = close(blockSourceFds[i]);
        if (close(blockCopyFds[i]) != 0 || sourceClosed != 0) {
            Debug(Debug::ERROR) << "Cannot copy " << servedBlockCopy.dataFiles[i] << " to " << (*servedBlockCopy.copies)[firstBlockCopy + i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    if (ioThreadCount > 0 && reservedBlockCount > 0) {
        MMseqsNode::addPhaseValue(phaseValues, "Copy to --local-tmp", SSTR(alignedCopiedBlockCount) + " of " + SSTR(reservedBlockCount) + " reserved blocks copied while aligning, "
                                  + SSTR(copiedBlockCount - alignedCopiedBlockCount) + " after aligning, in " + formatSeconds(blockCopySeconds)
                                  + ", " + MMseqsNode::formatBytes(cachedCopySize) + " from the block cache"
                                  + ((blockCopyError != 0) ? std::string(", the copies ended at a write error: ") + strerror(blockCopyError) : ""));
    }
    // master still finds key nodes in seqDbr blocks
    seqDbr.releaseDataPages();
    if (isAligningNode) {
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the master", waitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the sequences", sequenceWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the sequences while an I/O thread receives", sequenceWaitParts[0]);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the sequences served by other nodes", sequenceWaitParts[1]);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the sequences otherwise", sequenceWaitParts[2]);
        MMseqsNode::addPhaseValue(phaseValues, "Receiving sequences", formatSeconds(receiveSeconds[0]) + " in " + SSTR(receiveCalls[0]) + " calls on I/O threads, "
                                  + formatSeconds(receiveSeconds[1]) + " in " + SSTR(receiveCalls[1]) + " calls on the aligning team");
        MMseqsNode::addPhaseValue(phaseValues, "Serving sequences", formatSeconds(serveSeconds[0]) + " in " + SSTR(serveCalls[0]) + " calls on I/O threads, "
                                  + formatSeconds(serveSeconds[1]) + " in " + SSTR(serveCalls[1]) + " calls on the aligning team, "
                                  + formatSeconds(servePackSeconds) + " packing, " + formatSeconds(serveWriteSeconds) + " writing");
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the candidate pairs", prefilterWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "computing alignments", alignSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Sequences fetched ahead", "at most " + formatSeconds(maxFetchAheadSeconds) + ", mean " + formatSeconds(fetchAheadSecondsSum / std::max<size_t>(1, alignedFetchCount)) + " of aligning");
        MMseqsNode::addPhaseSeconds(phaseValues, "fetching the first sequences", firstFetchSeconds);
    }
    MMseqsNode::addPhaseValue(phaseValues, "Accepted pairs published", SSTR(publishCount) + " times, " + SSTR(static_cast<size_t>(1e6 * publishCostSeconds / std::max<size_t>(1, publishCount))) + " us each on average with the log flush");
    if (memberLookup != NULL) {
        MMseqsNode::addPhaseSeconds(phaseValues, "looking up the cluster members", memberLookupSeconds);
    }
}

static uint64_t *readSingletonBits(const std::string &assignedBits, size_t firstKey, size_t keyCount) {
    size_t words = (keyCount + 63) / 64;
    size_t firstWord = firstKey / 64;
    size_t logWords = (keyCount > 0) ? (firstKey + keyCount - 1) / 64 + 1 - firstWord : 0;
    std::vector<uint64_t> assignedWords(logWords + 1, 0);
    MMseqsNode::readLog(assignedBits, firstWord * sizeof(uint64_t), (firstWord + logWords) * sizeof(uint64_t), (char *) assignedWords.data());
    uint64_t *bits = new(std::nothrow) uint64_t[words];
    Util::checkAllocation(bits, "Can not allocate singletonBits memory in align2clust");
    size_t shift = firstKey % 64;
    for (size_t word = 0; word < words; word++) {
        uint64_t assigned = (assignedWords[word] >> shift) | ((shift > 0) ? assignedWords[word + 1] << (64 - shift) : 0);
        if (word + 1 == words && keyCount % 64 != 0) {
            assigned &= (1ULL << (keyCount % 64)) - 1;
        }
        bits[word] = ~assigned;
    }
    return bits;
}

int align2clustLinfinity(Parameters &par) {
    std::string phaseValues;
    Timer phaseValueTimer;
    size_t firstAlignNode = MMseqsNode::getFirstAlignNode();
    bool isAligningNode = MMseqsNode::rank >= firstAlignNode;
    size_t alignNodeCount = MMseqsNode::count - firstAlignNode;
    // .partitions: key_splits line, target split blocks, key split first keys, first class file entries
    std::string partitionsData = FileUtil::readFile(par.db2 + ".partitions");
    char *partitionLine = Util::skipLine((char *) partitionsData.c_str());
    for (; *partitionLine != '\n'; partitionLine += Util::skipWhitespace(partitionLine)) {
        targetSplitBlockOffsets.push_back(Util::fast_atoi<size_t>(partitionLine));
        partitionLine += Util::skipNoneWhitespace(partitionLine);
    }
    for (partitionLine++; *partitionLine != '\n'; partitionLine += Util::skipWhitespace(partitionLine)) {
        keySplitFirstKeys.push_back(Util::fast_atoi<size_t>(partitionLine));
        partitionLine += Util::skipNoneWhitespace(partitionLine);
    }
    for (partitionLine++; *partitionLine != '\n'; partitionLine += Util::skipWhitespace(partitionLine)) {
        classFileFirstEntries.push_back(Util::fast_atoi<size_t>(partitionLine));
        partitionLine += Util::skipNoneWhitespace(partitionLine);
    }

    std::vector<PrefWindow> windows;
    uint128 totalAlignmentCells = 0;
    size_t totalPrefSize = 0;
    for (char *line = partitionLine + 1; *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[5];
        Util::getWordsOfLine(line, columns, 5);
        size_t prefSize = Util::fast_atoi<size_t>(columns[2]);
        size_t alignmentCells = Util::fast_atoi<size_t>(columns[3]);
        // window of one entry until the windows are cut
        size_t entry = Util::fast_atoi<size_t>(columns[1]);
        const PrefWindow window = { Util::fast_atoi<size_t>(columns[0]), Util::fast_atoi<size_t>(columns[4]), entry, entry + 1, alignmentCells };
        windows.push_back(window);
        totalAlignmentCells += alignmentCells;
        totalPrefSize += prefSize;
    }

    // block table for the target splits and the local copy
    DBReader<DBKeyType> blockDbr(par.db1.c_str(), par.db1Index.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
    blockDbr.open(DBReader<DBKeyType>::NOSORT);
    blockDbr.setBlockCacheSize(0);
    size_t targetSplitCount = targetSplitBlockOffsets.size() - 1;
    const DBReader<DBKeyType>::Index *blockIndex = blockDbr.getIndex();
    std::vector<size_t> splitSizes(targetSplitCount, 0);
    size_t totalSize = 0;
    for (size_t split = 0; split < targetSplitCount; split++) {
        size_t keyBegin = blockDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[split]];
        size_t keyEnd = blockDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[split + 1]];
        for (size_t range = blockDbr.findIdRange(keyBegin); range < blockDbr.getIndexSize() && static_cast<size_t>(blockIndex[range].id) < keyEnd; range++) {
            size_t rangeEnd = (range + 1 < blockDbr.getIndexSize()) ? static_cast<size_t>(blockIndex[range + 1].id) : blockDbr.getSize();
            splitSizes[split] += (std::min(keyEnd, rangeEnd) - std::max(keyBegin, static_cast<size_t>(blockIndex[range].id))) * blockIndex[range].length;
        }
        totalSize += splitSizes[split];
    }

    // splits go over the align nodes back and forth (0 1 2 2 1 0 0 1 ...)
    targetSplitAlignNodes.resize(targetSplitCount);
    std::vector<size_t> nodeSizes(alignNodeCount, 0);
    for (size_t split = 0; split < targetSplitCount; split++) {
        size_t alignNode = ((split / alignNodeCount) % 2 == 0) ? split % alignNodeCount : alignNodeCount - 1 - split % alignNodeCount;
        nodeSizes[alignNode] += splitSizes[split];
        targetSplitAlignNodes[split] = firstAlignNode + alignNode;
    }
    if (isAligningNode) {
        MMseqsNode::addPhaseValue(phaseValues, "Target splits", SSTR(std::count(targetSplitAlignNodes.begin(), targetSplitAlignNodes.end(), MMseqsNode::rank)) + " of " + SSTR(targetSplitCount)
                                  + " with " + MMseqsNode::formatBytes(nodeSizes[MMseqsNode::rank - firstAlignNode]) + " of " + MMseqsNode::formatBytes(totalSize) + " of sequences");
    }

    // I/O threads within --threads, two if data crosses nodes
    bool isExchangingSequences = isAligningNode && alignNodeCount > 1;
    // every align node but itself asks for members
    bool isServingMembers = par.filterCluDBFile.empty() == false && MMseqsNode::count > 1 && (MMseqsNode::isMaster() || alignNodeCount > 1);
    int ioThreadCount = std::min((isExchangingSequences || isServingMembers) ? 2 : 1, par.threads - 1);
    int alignThreadCount = par.threads - ioThreadCount;
    // leave room for the I/O threads from the start
#ifdef OPENMP
    omp_set_num_threads(alignThreadCount);
#endif
    // master that only clusters reads no sequences

    bool isCopyingServedBlocks = par.localTmp.empty() == false && isAligningNode;
    const std::vector<std::string> dataFiles = FileUtil::findDatafiles(par.db1.c_str());
    // fds: 4 or 6 per data file, 5 per own pref file, 7 or 2 (NFS) per node
    size_t ownPrefFiles = std::count(targetSplitAlignNodes.begin(), targetSplitAlignNodes.end(), MMseqsNode::rank);
    for (size_t keySplit = 0; keySplit + 1 < keySplitFirstKeys.size(); keySplit++) {
        ownPrefFiles += getKeyAlignNode(blockDbr, keySplitFirstKeys[keySplit]) == MMseqsNode::rank;
    }
    size_t minOpenFiles = (par.filterCluDBFile.empty() ? 4 : 6) * dataFiles.size() + FileUtil::UNCOUNTED_OPEN_FILES + (FileUtil::needsDirectoryRefresh(par.db3) ? 2 : 7) * MMseqsNode::count + 5 * ownPrefFiles;
    // set cover: open runs, small set writers and the pair file
    if (MMseqsNode::isMaster() && par.clusteringMode == Parameters::SET_COVER) {
        minOpenFiles += SetCoverLargeSetFiles::MAX_OPEN_RUNS + SetCoverSmallSetFiles::MAX_SIZE;
    }
    FileUtil::checkRlimitNoFile(minOpenFiles);

    std::string cluPartDb = par.filterCluDBFile;
    std::vector<std::string> localCopies;
    std::string localDir;
    ServedBlockCopy servedBlockCopy;
    if (isCopyingServedBlocks) {
        std::string localTmpProblem = MMseqsNode::getLocalTmpProblem(par);
        if (localTmpProblem.empty()) {
            // blocks of this rank in key order
            const std::vector<size_t> &blockKeyOffsets = blockDbr.getBlockKeyOffsets();
            const DBReader<DBKeyType>::Index *blockIndex = blockDbr.getIndex();
            std::vector<size_t> &servedBlockOffsets = servedBlockCopy.blockOffsets;
            servedBlockOffsets.assign(1, 0);
            for (size_t block = 0; block + 1 < blockKeyOffsets.size(); block++) {
                if (getBlockAlignNode(block) != MMseqsNode::rank) {
                    continue;
                }
                size_t firstRange = blockDbr.findIdRange(blockKeyOffsets[block]);
                size_t lastRange = blockDbr.findIdRange(blockKeyOffsets[block + 1] - 1);
                size_t begin = blockIndex[firstRange].offset + (blockKeyOffsets[block] - blockIndex[firstRange].id) * blockIndex[firstRange].length;
                size_t end = blockIndex[lastRange].offset + (blockKeyOffsets[block + 1] - blockIndex[lastRange].id) * blockIndex[lastRange].length;
                servedBlockCopy.blocks.push_back(block);
                servedBlockCopy.blockFiles.push_back(blockDbr.getRangeDataFile(firstRange));
                servedBlockCopy.blockBegins.push_back(begin);
                servedBlockOffsets.push_back(servedBlockOffsets.back() + end - begin);
            }
            size_t servedSize = servedBlockOffsets.back();
            // a killed run's copy must not take free space
            localDir = par.localTmp + "/" + FileUtil::baseName(par.db3) + ".served." + SSTR(MMseqsNode::rank);
            std::vector<std::string> oldCopies = FileUtil::findDatafiles((localDir + "/" + FileUtil::baseName(par.db1)).c_str());
            // and the own cluster part
            std::vector<std::vector<std::pair<size_t, size_t> > > cluFileRanges(par.filterCluDBFile.empty() ? 0 : MMseqsNode::count);
            if (cluFileRanges.empty() == false) {
                std::vector<std::string> oldCluCopies = FileUtil::findDatafiles((localDir + "/" + FileUtil::baseName(par.filterCluDBFile)).c_str());
                oldCopies.insert(oldCopies.end(), oldCluCopies.begin(), oldCluCopies.end());
                cluFileRanges[MMseqsNode::rank].emplace_back(0, FileUtil::getFileSize(par.filterCluDBFile + "." + SSTR(MMseqsNode::rank)));
            }
            for (size_t i = 0; i < oldCopies.size(); i++) {
                FileUtil::remove(oldCopies[i].c_str());
            }
            size_t cluSize = cluFileRanges.empty() ? 0 : cluFileRanges[MMseqsNode::rank][0].second;
            // SIZE_MAX if statvfs failed
            size_t freeSize = FileUtil::getFreeSpace(par.localTmp.c_str());
            size_t localTmpSize = (freeSize == SIZE_MAX) ? 0 : freeSize;
            // master open sets and run files need what the copies leave
            struct stat localTmpStat, tmpStat;
            bool isTmpDevice = stat(par.localTmp.c_str(), &localTmpStat) == 0 && stat(FileUtil::dirName(par.db3).c_str(), &tmpStat) == 0 && localTmpStat.st_dev == tmpStat.st_dev;
            if (((MMseqsNode::isMaster() && par.clusteringMode == Parameters::SET_COVER) || isTmpDevice) && localTmpSize < cluSize + servedSize) {
                localTmpSize = 0;
            }
            // cluster part whole or not at all, then blocks that fit
            size_t cluCopySize = 0;
            if (cluSize > 0 && cluSize <= localTmpSize) {
                size_t firstCopy = localCopies.size();
                cluPartDb = FileUtil::copyToLocalDir(par.filterCluDBFile, localDir, alignThreadCount, localCopies, cluCopySize, cluFileRanges, localTmpSize);
                if (cluCopySize < cluSize) {
                    for (size_t i = firstCopy; i < localCopies.size(); i++) {
                        FileUtil::remove(localCopies[i].c_str());
                    }
                    localCopies.resize(firstCopy);
                    cluPartDb = par.filterCluDBFile;
                    cluCopySize = 0;
                }
            }
            if (cluCopySize > 0) {
                MMseqsNode::addPhaseTime(phaseValues, "copying the cluster part to --local-tmp", phaseValueTimer);
            }
            // alignRepTargets copies the blocks and reports if none fits
            const std::string spaceProblem = (freeSize == SIZE_MAX) ? std::string("its free space is unknown")
                                             : "it has " + MMseqsNode::formatBytes(freeSize) + " (" + SSTR(freeSize) + " bytes) free"
                                               + ", the copies need " + MMseqsNode::formatBytes(cluSize + servedSize) + " (" + SSTR(cluSize + servedSize) + " bytes)";
            if (servedBlockCopy.blocks.empty() == false) {
                servedBlockCopy.localDir = localDir;
                servedBlockCopy.dataFiles = dataFiles;
                servedBlockCopy.maxBytes = localTmpSize - cluCopySize;
                servedBlockCopy.openFiles = minOpenFiles;
                servedBlockCopy.spaceProblem = (cluCopySize > 0) ? "" : spaceProblem;
                servedBlockCopy.copies = &localCopies;
            } else if (cluCopySize > 0) {
                MMseqsNode::addPhaseValue(phaseValues, "Local tmp", "holds the cluster part");
            } else if (cluSize > 0) {
                localTmpProblem = spaceProblem;
            }
        }
        if (localTmpProblem.empty() == false) {
            reportServedCopyProblem(par, localTmpProblem, phaseValues);
        }
    }

    // only the master writes and reads the open sets of set cover
    std::string setCoverPrefix = par.db3;
    if (MMseqsNode::isMaster() && par.clusteringMode == Parameters::SET_COVER && par.localTmp.empty() == false) {
        std::string localTmpProblem = MMseqsNode::getLocalTmpProblem(par);
        if (localTmpProblem.empty()) {
            setCoverPrefix = par.localTmp + "/" + FileUtil::baseName(par.db3);
        } else if (par.localTmpFallback == false) {
            Debug(Debug::ERROR) << "Cannot keep the open sets in --local-tmp " << par.localTmp << ": " << localTmpProblem << ". Please choose another --local-tmp or set --local-tmp-fallback 1 to keep them in the shared temporary folder.\n";
            EXIT(EXIT_FAILURE);
        } else {
            MMseqsNode::reportUnusableLocalTmp("Cannot keep the open sets in --local-tmp " + par.localTmp + ": " + localTmpProblem);
        }
        MMseqsNode::addPhaseValue(phaseValues, "Open sets", localTmpProblem.empty() ? "in --local-tmp" : "in the shared temporary folder, " + localTmpProblem);
    }

    blockDbr.close();
    DBReader<DBKeyType> seqDbr(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    seqDbr.open(DBReader<DBKeyType>::NOSORT);
    const std::vector<size_t> nodeKeyOffsets = seqDbr.getEvenNodeKeyOffsets(MMseqsNode::count);
    size_t dbSize = seqDbr.getSize();
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    size_t nodeMemoryLimit = Util::computeMemory(par.splitMemoryLimit);
    bool isMasterAligning = firstAlignNode == 0;
    size_t masterMemoryLimit = nodeMemoryLimit / (isMasterAligning ? 2 : 1);
    // master holds the assigned bits and a log buffer per node
    size_t masterFixedSize = (dbSize + 63) / 64 * sizeof(uint64_t) + MMseqsNode::count * CLUSTERED_LOG_BUFFER_SIZE;
    size_t masterMemorySize = masterFixedSize + ((par.clusteringMode == Parameters::SET_COVER) ? masterMemoryLimit : masterMemoryLimit / 4);
    if (MMseqsNode::isMaster() && masterFixedSize >= nodeMemoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to cluster. Memory limit: " << nodeMemoryLimit << " bytes, the assigned bits of " << dbSize << " sequences need: "
                            << (dbSize + 63) / 64 * sizeof(uint64_t) << " bytes, the clustered logs of " << MMseqsNode::count << " nodes need: "
                            << MMseqsNode::count * CLUSTERED_LOG_BUFFER_SIZE << " bytes.\n"
                            << "Raise RAM size or --split-memory-limit. The assigned bits do not shrink with more nodes.\n";
        EXIT(EXIT_FAILURE);
    }

    // builds the block table of getKeyBlock
    seqDbr.setBlockCacheSize(0);
    const std::vector<size_t> &blockKeyOffsets = seqDbr.getBlockKeyOffsets();
    // master keeps the align node of every assigned page
    size_t pageCount = (dbSize + ASSIGNED_PAGE_KEYS - 1) / ASSIGNED_PAGE_KEYS;
    std::vector<unsigned int> pageAlignNodes(MMseqsNode::isMaster() ? pageCount : 0);
    for (size_t block = 0, pageEnd = 0; MMseqsNode::isMaster() && block + 1 < blockKeyOffsets.size(); block++) {
        unsigned int node = getBlockAlignNode(block);
        size_t firstPage = blockKeyOffsets[block] / ASSIGNED_PAGE_KEYS;
        size_t lastPage = (blockKeyOffsets[block + 1] - 1) / ASSIGNED_PAGE_KEYS;
        for (size_t page = firstPage; page <= lastPage; page++) {
            // page with keys of two nodes uses getKeyAlignNode
            pageAlignNodes[page] = (page < pageEnd && pageAlignNodes[page] != node) ? UINT_MAX : node;
        }
        pageEnd = lastPage + 1;
    }
    // align node gets every assigned key
    size_t assignedPageCount = isAligningNode ? pageCount : 0;
    size_t assignedMemorySize = assignedPageCount * (ASSIGNED_PAGE_KEYS / 8) + ((assignedPageCount > 0) ? pageCount * sizeof(uint64_t *) : 0) + pageAlignNodes.size() * sizeof(unsigned int);
    size_t memoryLimit = nodeMemoryLimit - std::min(nodeMemoryLimit, (isMasterAligning ? std::min(masterMemoryLimit, masterMemorySize) : 0) + assignedMemorySize);

    // clusthashfast subsets have bits and count reps in index column 6
    bool hasRepresentativeBits = FileUtil::statSharedFile(par.db1 + ".bits." + SSTR(MMseqsNode::rank));
    size_t representativeCount = 0;
    size_t representativeResidues = 0;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        representativeCount += seqDbr.getRangeRepresentativeCount(range);
        representativeResidues += seqDbr.getRangeRepresentativeCount(range) * (index[range].length - 2);
    }

    const std::string accepted = par.db3 + "_accepted";
    const std::string assignedKeysLog = par.db3 + "_assigned_keys";
    const std::string clusteredLogPrefix = par.db3 + "_clustered";
    // assigned bits only for singletons of a DB without bits
    const std::string assignedBits = hasRepresentativeBits ? "" : par.db3 + "_assigned_keys.bits";
    // create the exchange indexes before the memory barrier
    const std::string requests = par.db3 + "_sequence_requests";
    const std::string served = par.db3 + "_served_sequences";
    const std::string memberRequests = par.db3 + "_member_requests";
    const std::string servedMembers = par.db3 + "_members";
    std::vector<std::string> sequenceIndexes;
    std::vector<std::string> sequenceFiles;
    // empty progress file has no requests and no served files
    if (isAligningNode && alignNodeCount > 1) {
        sequenceIndexes.push_back(requests + "." + SSTR(MMseqsNode::rank) + ".progress");
        sequenceIndexes.push_back(served + "." + SSTR(MMseqsNode::rank) + ".progress");
        // request and served files share a directory each, one listing shows new files
        const std::string exchangeDirs[] = { requests, served };
        for (size_t i = 0; i < 2; i++) {
            if (FileUtil::makeDir(exchangeDirs[i].c_str()) == false && FileUtil::directoryExists(exchangeDirs[i].c_str()) == false) {
                Debug(Debug::ERROR) << "Cannot create " << exchangeDirs[i] << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }
    // every node serves its cluster part, aligning nodes ask
    if (par.filterCluDBFile.empty() == false && MMseqsNode::count > 1) {
        sequenceIndexes.push_back(servedMembers + ".index." + SSTR(MMseqsNode::rank));
        if (isAligningNode) {
            sequenceIndexes.push_back(memberRequests + ".index." + SSTR(MMseqsNode::rank));
            sequenceFiles.push_back(memberRequests + "." + SSTR(MMseqsNode::rank));
        }
    }
    // readers keep log progress files open, removeLog removes them
    std::vector<std::string> createdFiles(sequenceIndexes);
    createdFiles.push_back(accepted + "." + SSTR(MMseqsNode::rank) + ".progress");
    createdFiles.push_back(clusteredLogPrefix + "." + SSTR(MMseqsNode::rank) + ".progress");
    if (MMseqsNode::isMaster()) {
        createdFiles.push_back(assignedKeysLog + ".progress");
        if (assignedBits.empty() == false) {
            createdFiles.push_back(assignedBits + ".progress");
        }
    }
    for (size_t i = 0; i < createdFiles.size(); i++) {
        if (fclose(FileUtil::openAndDelete(createdFiles[i].c_str(), "w")) != 0) {
            Debug(Debug::ERROR) << "Cannot create " << createdFiles[i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }

    // all nodes use the smallest memory limit
    size_t smallestMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db3 + ".memory", nodeMemoryLimit - std::min(nodeMemoryLimit, assignedMemorySize));
    size_t paddedDbSize = (dbSize + 63) / 64 * 64;
    size_t sizeClassCount = (par.clusteringMode == Parameters::SET_COVER) ? 64 - __builtin_clzll(dbSize | 1) : 1;
    // record's reads fit the smallest memory limit
    size_t maxRecordSize = std::max<size_t>(1, smallestMemoryLimit / 32);
    // all nodes cut served files at the same size
    size_t servedFileEntrySize = std::max(BatchReader::MAX_READ_REQUEST_SIZE, smallestMemoryLimit / 512);
    // windows of about equal cost, each in one node's consecutive entries
    size_t prefReadSize = alignNodeCount * getMaxPrefReadSize(smallestMemoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR);
    size_t windowCount = std::max((size_t) 1, (totalPrefSize + prefReadSize - 1) / prefReadSize);
    uint128 windowCost = std::max((uint128) 1, (totalAlignmentCells + windowCount - 1) / windowCount);
    uint128 costBefore = 0;
    uint128 nextCut = windowCost;
    size_t window = 0;
    for (size_t i = 0; i < windows.size(); i++) {
        const PrefWindow row = windows[i];
        bool isCostCut = costBefore >= nextCut;
        if (isCostCut) {
            nextCut = (costBefore / windowCost + 1) * windowCost;
        }
        if (i > 0 && (isCostCut || row.node != windows[window].node || row.firstEntry != windows[window].endEntry)) {
            windows[++window] = row;
        } else if (i > 0) {
            windows[window].endEntry = row.endEntry;
            windows[window].alignmentCells += row.alignmentCells;
        }
        costBefore += row.alignmentCells;
    }
    size_t prefEntryCount = windows.size();
    windows.resize(std::min(windows.size(), window + 1));
    MMseqsNode::addPhaseValue(phaseValues, "Candidate pair entries", SSTR(prefEntryCount) + ", read in " + SSTR(windows.size()) + " windows");

    // align node of a pref file is its only reader
    bool isRemovingPrefFiles = par.removeTmpFiles && isAligningNode;
    TargetSplitReader targetSplitReader(par, seqDbr, windows, maxRecordSize, memoryLimit, paddedDbSize, sizeClassCount, isRemovingPrefFiles);
    if (sizeClassCount > 1) {
        targetSplitReader.countSizeClasses();
    }
    MMseqsNode::addPhaseTime(phaseValues, "counting the candidate pairs of each size class", phaseValueTimer);

    ClusterMemberLookup *memberLookup = NULL;
    ClusterMemberExchange *memberExchange = NULL;
    DBReader<DBKeyType> *memberDbr = NULL;
    if (par.filterCluDBFile.empty() == false) {
        size_t entriesPerSampledKey = std::max<size_t>(256, dbSize / std::max<size_t>(1, smallestMemoryLimit / 64 / sizeof(DBKeyType)));
        std::vector<size_t> cluKeyOffsets = MMseqsNode::getClusterKeyOffsets(par.db3 + ".cluster_key_offsets", par.filterCluDBFile, dbSize);
        cluKeyOffsets[0] = 0;
        // aligning node preloads entries of its target splits
        std::vector<std::pair<size_t, size_t> > targetSplitKeyRanges;
        for (size_t block = 0; block + 1 < blockKeyOffsets.size(); block++) {
            if (getBlockAlignNode(block) != MMseqsNode::rank) {
                continue;
            }
            if (targetSplitKeyRanges.empty() == false && targetSplitKeyRanges.back().second == blockKeyOffsets[block]) {
                targetSplitKeyRanges.back().second = blockKeyOffsets[block + 1];
            } else {
                targetSplitKeyRanges.emplace_back(blockKeyOffsets[block], blockKeyOffsets[block + 1]);
            }
        }
        // every node answers for its cluster part, the master while clustering
        size_t lookupMemoryLimit = nodeMemoryLimit - std::min(nodeMemoryLimit, (MMseqsNode::isMaster() ? std::min(masterMemoryLimit, masterMemorySize) : 0) + assignedMemorySize);
        memberLookup = new ClusterMemberLookup(par.filterCluDBFile, cluPartDb, cluKeyOffsets, entriesPerSampledKey, lookupMemoryLimit / 16, targetSplitKeyRanges);
        memberExchange = new ClusterMemberExchange(memberRequests, servedMembers, *memberLookup, std::max(1, par.threads / 4), firstAlignNode);
        memberDbr = new DBReader<DBKeyType>(par.filterSeqDBFile.c_str(), (par.filterSeqDBFile + ".index").c_str(), par.threads,
                                            DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        memberDbr->open(DBReader<DBKeyType>::NOSORT);
        MMseqsNode::addPhaseTime(phaseValues, "sampling the cluster index keys", phaseValueTimer);
    }

    // single node clusters in I/O tasks while it aligns
    size_t memberCount = 0;
    std::string clusterPhaseValues;
    AcceptedPairClustering *clustering = NULL;
    if (MMseqsNode::isMaster() && isMasterAligning) {
        clustering = new AcceptedPairClustering(par.threads, dbSize, masterMemoryLimit, seqDbr, pageAlignNodes, par.clusteringMode, paddedDbSize, sizeClassCount,
                                                accepted, assignedKeysLog, clusteredLogPrefix, setCoverPrefix, assignedBits, false);
    }
    phaseValueTimer.reset();
    // align nodes serve members in I/O tasks, the master on a thread or while idle
    bool hasMemberServer = memberExchange != NULL && MMseqsNode::count > 1 && MMseqsNode::isMaster() && ioThreadCount > 0;
    std::thread memberServer;
    if (hasMemberServer) {
        memberServer = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(1);
#endif
            memberExchange->serve();
        });
    }
    // member key 7 words + 2 flags, pending part 9 + 2, accepted pair 25 + 1
    size_t memberKeyBudget = std::max<size_t>(1, 2 * std::min(memoryLimit, smallestMemoryLimit) / MEMBER_SPLIT_MEMORY_DIVISOR / ((7 + 9 + 25) * sizeof(uint64_t) + 5));
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Aligning the candidate pairs on " << alignNodeCount << " of " << MMseqsNode::count << " nodes and clustering the accepted pairs on node 0\n";
        std::string layout;
        MMseqsNode::addPhaseValue(layout, "Layout", "the align nodes share " + SSTR(targetSplitCount) + " target splits with " + MMseqsNode::formatBytes(totalSize) + " of sequences and read "
                                  + SSTR(prefEntryCount) + " candidate pair entries in " + SSTR(windows.size()) + " windows; node 0 clusters ("
                                  + ((par.clusteringMode == Parameters::SET_COVER) ? "set cover" : "greedy") + ") the pairs they accept as they publish them");
        MMseqsNode::printPhaseValues(layout);
    }
    alignRepTargets(par, seqDbr, targetSplitReader, memoryLimit, masterMemoryLimit, servedFileEntrySize, representativeResidues, paddedDbSize, sizeClassCount,
                    memberLookup, memberExchange, memberKeyBudget, memberDbr, ioThreadCount, clustering, servedBlockCopy, phaseValues);
    // master of several nodes clusters, writer on a spare thread
    if (MMseqsNode::isMaster() && isMasterAligning == false) {
        clustering = new AcceptedPairClustering(par.threads, dbSize, masterMemoryLimit, seqDbr, pageAlignNodes, par.clusteringMode, paddedDbSize, sizeClassCount,
                                                accepted, assignedKeysLog, clusteredLogPrefix, setCoverPrefix, assignedBits, ioThreadCount - (hasMemberServer ? 1 : 0) > 0);
        // serve the members while idle unless the member server thread does
        for (int step = clustering->step(); step != AcceptedPairClustering::STEP_DONE; step = clustering->step()) {
            if (step == AcceptedPairClustering::STEP_IDLE && (hasMemberServer || memberExchange == NULL || memberExchange->serveRequests() == false)) {
                usleep(clustering->idleMicroseconds);
            }
        }
    }
    if (clustering != NULL) {
        memberCount = clustering->finish(clusterPhaseValues);
        delete clustering;
    }
    // clusters are written with all threads
#ifdef OPENMP
    omp_set_num_threads(par.threads);
#endif
    if (memberExchange != NULL) {
        // serve until every aligning node has its members
        if (memberServer.joinable()) {
            memberServer.join();
        }
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the cluster member sizes", memberExchange->sizeWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the cluster members", memberExchange->memberWaitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member data sent to other nodes", MMseqsNode::formatBytes(memberExchange->servedSize));
        MMseqsNode::addPhaseSeconds(phaseValues, "preloading the cluster members", memberLookup->preloadSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the preloaded cluster members", memberExchange->preloadWaitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Cluster members preloaded", MMseqsNode::formatBytes(memberLookup->preloadSize) + " of " + MMseqsNode::formatBytes(memberLookup->targetSplitSize) + " of the target splits");
    }
    delete memberExchange;
    delete memberLookup;
    targetSplitReader.close();
    if (isAligningNode) {
        MMseqsNode::addPhaseTime(phaseValues, "aligning the candidate pairs", phaseValueTimer);
    }
    MMseqsNode::addPhaseValue(phaseValues, "Memory limit", MMseqsNode::formatBytes(memoryLimit));

    const std::string clusteredLogFile = clusteredLogPrefix + "." + SSTR(MMseqsNode::rank);
    size_t clusteredKey;
    size_t clusteredSize;
    MMseqsNode::waitForLog(clusteredLogFile, clusteredKey, clusteredSize);
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Number of clusters: " << representativeCount - memberCount << "\n";
        Debug(Debug::INFO) << "Writing the clusters on all nodes\n";
        phaseValues += clusterPhaseValues;
        std::vector<unsigned int>().swap(pageAlignNodes);
    }
    // clustering was the last user of seqDbr
    seqDbr.close();
    for (size_t i = 0; i < localCopies.size(); i++) {
        FileUtil::remove(localCopies[i].c_str());
    }
    if (localCopies.empty() == false) {
        rmdir(localDir.c_str());
    }
    MMseqsNode::addPhaseTime(phaseValues, "waiting for the clustering", phaseValueTimer);

    size_t nodeFirstKey = nodeKeyOffsets[MMseqsNode::rank];
    size_t nodeKeys = nodeKeyOffsets[MMseqsNode::rank + 1] - nodeFirstKey;
    const std::string clusterPairs = par.db3 + "_cluster_pairs";
    std::vector<size_t> balancedKeyOffsets;
    MMseqsNode::ClusterPairWriter clusterPairWriter(clusterPairs, std::max<size_t>(64, memoryLimit / 16 / sizeof(ClusterPair)), ClusterPair::compareByRep, nodeKeyOffsets,
                                                    &balancedKeyOffsets);
    const std::string acceptedFile = accepted + "." + SSTR(MMseqsNode::rank);
    if (par.clusteringMode == Parameters::SET_COVER) {
        for (size_t from = 0; from < clusteredSize;) {
            size_t to = std::min(clusteredSize, from + clusterPairWriter.pairCapacity * sizeof(ClusterPair));
            MMseqsNode::readLog(clusteredLogFile, from, to, reinterpret_cast<char *>(clusterPairWriter.reservePairs((to - from) / sizeof(ClusterPair))));
            from = to;
        }
    } else {
        std::vector<uint64_t> clusteredBits(clusteredSize / sizeof(uint64_t));
        MMseqsNode::readLog(clusteredLogFile, 0, clusteredBits.size() * sizeof(uint64_t), reinterpret_cast<char *>(clusteredBits.data()));
        // last progress of the accepted log has its size
        size_t acceptedKey;
        size_t acceptedBytes;
        MMseqsNode::waitForLog(acceptedFile, acceptedKey, acceptedBytes);
        size_t acceptedWordCount = acceptedBytes / sizeof(uint64_t);
        // read takes whole words of clustered bits
        size_t readWordCount = std::min(clusterPairWriter.pairCapacity, std::max<size_t>(1, clusteredBits.size()) * 64) / 64 * 64;
        std::vector<uint64_t> acceptedWords(readWordCount);
        std::vector<std::vector<size_t> > threadRepWordPos(par.threads);
        std::vector<size_t> repWordPos;
        uint64_t previousRepWord = 0;
        for (size_t firstWord = 0; firstWord * 64 < acceptedWordCount; firstWord += readWordCount / 64) {
            size_t wordCount = std::min(readWordCount, acceptedWordCount - firstWord * 64);
            MMseqsNode::readLog(acceptedFile, firstWord * 64 * sizeof(uint64_t), (firstWord * 64 + wordCount) * sizeof(uint64_t), reinterpret_cast<char *>(acceptedWords.data()));
            size_t clusteredWordCount = (wordCount + 63) / 64;
            const uint64_t *clusteredWords = clusteredBits.data() + firstWord;
            size_t clusteredCount = 0;
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                threadRepWordPos[thread_idx].clear();
#pragma omp for schedule(static) reduction(+:clusteredCount)
                for (size_t word = 0; word < clusteredWordCount; word++) {
                    clusteredCount += __builtin_popcountll(clusteredWords[word]);
                    for (size_t i = word * 64; i < std::min(wordCount, word * 64 + 64); i++) {
                        if (acceptedWords[i] & ACCEPTED_REP_BIT) {
                            threadRepWordPos[thread_idx].push_back(i);
                        }
                    }
                }
            }
            repWordPos.clear();
            for (int thread = 0; thread < par.threads; thread++) {
                repWordPos.insert(repWordPos.end(), threadRepWordPos[thread].begin(), threadRepWordPos[thread].end());
            }
            ClusterPair *clusteredPairs = clusterPairWriter.reservePairs(clusteredCount);
            size_t clusteredPairCount = 0;
#pragma omp parallel for schedule(static)
            for (size_t word = 0; word < clusteredWordCount; word++) {
                uint64_t bits = clusteredWords[word];
                if (bits == 0) {
                    continue;
                }
                size_t pairOffset = __sync_fetch_and_add(&clusteredPairCount, __builtin_popcountll(bits));
                size_t rep = std::upper_bound(repWordPos.begin(), repWordPos.end(), word * 64) - repWordPos.begin();
                for (; bits != 0; bits &= bits - 1) {
                    size_t i = word * 64 + __builtin_ctzll(bits);
                    while (rep < repWordPos.size() && repWordPos[rep] < i) {
                        rep++;
                    }
                    uint64_t repWord = (rep > 0) ? acceptedWords[repWordPos[rep - 1]] : previousRepWord;
                    const ClusterPair clusteredPair = { static_cast<DBKeyType>((repWord & ~ACCEPTED_REP_BIT) % paddedDbSize), static_cast<DBKeyType>(acceptedWords[i]) };
                    clusteredPairs[pairOffset++] = clusteredPair;
                }
            }
            if (repWordPos.empty() == false) {
                previousRepWord = acceptedWords[repWordPos.back()];
            }
        }
    }
    MMseqsNode::removeLog(clusteredLogFile);
    MMseqsNode::removeLog(acceptedFile);

    // write the clusters and, without bits, the singletons
    uint64_t *singletonBits = NULL;
    if (assignedBits.empty() == false) {
        size_t assignedKey;
        size_t assignedSize;
        MMseqsNode::waitForLog(assignedBits, assignedKey, assignedSize);
        singletonBits = readSingletonBits(assignedBits, nodeFirstKey, nodeKeys);
    }
    const std::string clusterPairsIndex = clusterPairWriter.close(singletonBits, nodeFirstKey, nodeKeys);
    // every node has read what this node requested and served
    for (size_t i = 0; i < sequenceIndexes.size(); i++) {
        FileUtil::removeSharedFile(sequenceIndexes[i]);
    }
    for (size_t i = 0; i < sequenceFiles.size(); i++) {
        FileUtil::removeSharedFile(sequenceFiles[i]);
    }
    rmdir(requests.c_str());
    rmdir(served.c_str());
    MMseqsNode::addPhaseTime(phaseValues, "sending the cluster pairs", phaseValueTimer);
    if (memberDbr != NULL) {
        MMseqsNode::addPhaseValue(phaseValues, "Member sequence data read in blocks", MMseqsNode::formatBytes(memberDbr->getBlockReadSize()));
        MMseqsNode::addPhaseValue(phaseValues, "Member sequence data read one at a time", MMseqsNode::formatBytes(memberDbr->getKeyReadSize()));
        memberDbr->close();
        delete memberDbr;
    }

    // write the representatives of the balanced keys and their singletons
    size_t repFirstKey = balancedKeyOffsets[MMseqsNode::rank];
    size_t repKeyCount = balancedKeyOffsets[MMseqsNode::rank + 1] - repFirstKey;
    if (assignedBits.empty() == false) {
        delete[] singletonBits;
        singletonBits = readSingletonBits(assignedBits, repFirstKey, repKeyCount);
    }
    MMseqsNode::writeClusterPairs(clusterPairsIndex, clusterPairs, par.db3, singletonBits, repFirstKey, repKeyCount);
    delete[] singletonBits;
    MMseqsNode::addPhaseTime(phaseValues, "writing the clusters", phaseValueTimer);
    MMseqsNode::removeExchanges(par.db3, clusterPairs, std::vector<std::string>(1, clusterPairs), phaseValues);
    MMseqsNode::writeNodeLog(par.db3);
    if (MMseqsNode::isMaster()) {
        // every node is done with both logs
        MMseqsNode::removeLog(assignedKeysLog);
        FileUtil::removeSharedFile(par.db3 + ".memory");
        if (assignedBits.empty() == false) {
            MMseqsNode::removeLog(assignedBits);
        }
        DBWriter::writeDbtypeFile(par.db3.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_CLUSTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
    }
    return EXIT_SUCCESS;
}
