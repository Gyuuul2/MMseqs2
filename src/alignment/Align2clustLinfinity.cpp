#include "Align2clust.h"
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
#include <sys/statvfs.h>
#include <cstring>
#include <thread>
#include <ctime>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>
#include <list>
#include <queue>

#ifdef OPENMP
#include <omp.h>
#endif

static const size_t PREFETCH_DISTANCE = 16;

static size_t getSetCoverFileSize(size_t memoryLimit) {
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
        FILE *reader = FileUtil::openFileOrDie(fileName.c_str(), "rb", true);
        const size_t recordsPerRead = std::max<size_t>(1, readSize / (recordElementCount * sizeof(T)) / runs.size());
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
        FileUtil::remove(fileName.c_str());
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
            const size_t records = std::min(recordsPerRead, run.end - run.next);
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
        : pairs(pairs), fileName(fileName), maxPairs(std::max<size_t>(1, maxPairs)), memoryLimit(memoryLimit), assigned(assigned), file(NULL) {}

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
        runs.merge(fileName, 1, getSetCoverFileSize(memoryLimit),
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
        const size_t setSizeBufferSize = std::max<size_t>(count * sizeof(DBKeyType), memoryLimit / 64 / (MAX_SIZE - 2));
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
        const size_t recordCount = setSizeFile.memberKeys.size() / count;
        std::vector<std::pair<DBKeyType, size_t> > order(recordCount);
        for (size_t i = 0; i < recordCount; i++) {
            order[i] = std::make_pair(setSizeFile.memberKeys[i * count], i);
        }
        SORT_PARALLEL(order.begin(), order.end(), std::greater<std::pair<DBKeyType, size_t> >());
        const std::string path = fileName(count);
        if (setSizeFile.writer == NULL) {
            setSizeFile.writer = FileUtil::openAndDelete(path.c_str(), "wb");
        }
        const size_t writeWords = std::max<size_t>(count, (4ULL * 1024 * 1024 / sizeof(DBKeyType) / count) * count);
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
        setSizeFile.runs.merge(path, count, getSetCoverFileSize(memoryLimit),
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
    SetCoverLargeSetFiles(const std::string &prefix, size_t memoryLimit)
        : prefix(prefix), memoryBudget(std::max<size_t>(65 * sizeof(DBKeyType), getSetCoverFileSize(memoryLimit))),
          nextFile(0), writtenSize(0) {}

    template <typename Reps, typename Counts, typename IsFree, typename Word>
    void appendStep(const Reps &reps, const Counts &offsets, size_t paddedDbSize, IsFree isFree, Word word) {
        std::vector<size_t> order;
        for (size_t rep = 0; rep < reps.size(); rep++) {
            if (offsets[rep + 1] - offsets[rep] > SetCoverSmallSetFiles::MAX_SIZE) {
                order.push_back(rep);
            }
        }
        if (order.empty()) {
            return;
        }
        std::sort(order.begin(), order.end(), [&](size_t first, size_t second) {
            const size_t firstCount = offsets[first + 1] - offsets[first];
            const size_t secondCount = offsets[second + 1] - offsets[second];
            return firstCount == secondCount ? reps[first].repKey % paddedDbSize > reps[second].repKey % paddedDbSize : firstCount > secondCount;
        });
        FILE *file = NULL;
        const size_t run = createSortedRun(file);
        for (size_t rep : order) {
            const size_t count = offsets[rep + 1] - offsets[rep];
            writeWord(file, count);
            writeWord(file, reps[rep].repKey % paddedDbSize);
            for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                if (isFree(i)) {
                    writeWord(file, word(i));
                }
            }
        }
        finishRun(run, file);
    }

    template <typename IsAssigned, typename AssignMember>
    void assignSetsLargerThan(size_t maxSetSize, IsAssigned isAssigned, AssignMember assignMember, SetCoverPairFile &pairFile,
                              SetCoverSmallSetFiles &smallSetFiles) {
        while (true) {
            const bool fromPending = pending.empty() == false && (heads.empty() || isLargerSet(pending.front().memberCount, pending.front().representativeId,
                                                                                              heads.top().count, heads.top().repKey));
            const size_t count = fromPending ? pending.front().memberCount : (heads.empty() ? 0 : heads.top().count);
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
                    const DBKeyType key = fromPending ? pendingMembers[start + member] : readWord(run, start + member);
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
                        const size_t nextRun = createSortedRun(file);
                        writeWord(file, validCount);
                        for (size_t member = 0; member < count; member++) {
                            const DBKeyType key = fromPending ? pendingMembers[start + member] : readWord(run, start + member);
                            if (isAssigned(key) == false) {
                                writeWord(file, key);
                            }
                        }
                        finishRun(nextRun, file);
                    } else {
                        if (pendingMembers.size() + validCount > memoryBudget / sizeof(DBKeyType)) {
                            flushPending();
                        }
                        const size_t offset = pendingMembers.size();
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
        const size_t index = freeRuns.empty() ? runs.size() : freeRuns.back();
        if (freeRuns.empty()) {
            runs.emplace_back();
        } else {
            freeRuns.pop_back();
        }
        runs[index].path = prefix + "." + SSTR(nextFile++);
        file = FileUtil::openAndDelete(runs[index].path.c_str(), "wb");
        return index;
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
            const size_t bufferWords = 8192;
            source.bufferStart = (offset / bufferWords) * bufferWords;
            source.buffer.resize(std::min(bufferWords, source.words - source.bufferStart));
            if (source.reader == NULL) {
                if (openRuns.size() == 256) {
                    Run &oldest = runs[openRuns.front()];
                    if (fclose(oldest.reader) != 0) {
                        Debug(Debug::ERROR) << "Cannot close " << oldest.path << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    oldest.reader = NULL;
                    openRuns.pop_front();
                }
                source.reader = FileUtil::openFileOrDie(source.path.c_str(), "rb", true);
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
        const size_t count = readWord(run, runs[run].position);
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
        const size_t run = createSortedRun(file);
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

struct PrefPartition {
    size_t firstKey;
    size_t prefOffset;
    size_t prefSize;
    size_t alignmentCells;
    size_t node;
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

// at 1 T a split is about a minute of work of a node
static const size_t ALIGNMENT_SPLITS_PER_NODE = 4096;
// at least 8 MiB of prefilter per split
static const size_t MIN_ALIGNMENT_SPLIT_PREF_SIZE = 8 * 1024 * 1024;

static size_t getFirstAlignNode() {
    return (MMseqsNode::count > 1) ? 1 : 0;
}

// aligning nodes serve the blocks in turn
static size_t getBlockServingNode(size_t block) {
    return getFirstAlignNode() + block % (MMseqsNode::count - getFirstAlignNode());
}

static size_t getKeyServingNode(const DBReader<DBKeyType> &seqDbr, size_t key) {
    return getBlockServingNode(seqDbr.getKeyBlock(key));
}

static size_t getSizeClass(size_t targetCount, size_t sizeClassCount) {
    return (targetCount > 0) ? std::min(sizeClassCount - 1, (size_t) (63 - __builtin_clzll(targetCount))) : 0;
}

// retry a missing or short file in later passes, then fail
static void failIfMissingTooLong(time_t &missingSince, unsigned int &attempts, const std::string &error) {
    time_t now = time(NULL);
    if (attempts++ == 0) {
        missingSince = now;
    }
    if (attempts >= 100 && now - missingSince >= 60) {
        Debug(Debug::ERROR) << error;
        EXIT(EXIT_FAILURE);
    }
}

static std::string describeServedFile(size_t size, size_t observedSize, uint64_t fileNumber, uint64_t observedFileNumber, int error) {
    return " (expected " + SSTR(size) + " bytes, observed " + (observedSize == SIZE_MAX ? std::string("unavailable") : SSTR(observedSize))
           + " bytes; expected file number " + SSTR(fileNumber) + ", observed " + (observedFileNumber == UINT64_MAX ? std::string("unavailable") : SSTR(observedFileNumber))
           + "; error " + SSTR(error) + ")\n";
}

// linfinity: records [begin, end) of a prefilter buffer and their fetched entries
struct FetchedSequences {
    FetchedSequences() : prefilterBuffer(0), begin(0), end(0), pendingPairCount(0), firstPendingRepKey(0), isLast(false), entries(NULL), allocatedSize(0) {}
    size_t prefilterBuffer;
    size_t begin;
    size_t end;
    // pending pairs whose members it brings
    size_t pendingPairCount;
    size_t firstPendingRepKey;
    bool isLast;
    std::vector<DBKeyType> keys;
    std::vector<size_t> fetchSourceKeyOffsets;
    char *entries;
    size_t allocatedSize;
    std::vector<size_t> entryOffsets;
};

static const char *findFetchedKey(const FetchedSequences &sequences, size_t fetchSource, DBKeyType key) {
    const DBKeyType *keys = sequences.keys.data();
    const DBKeyType *found = std::lower_bound(keys + sequences.fetchSourceKeyOffsets[fetchSource], keys + sequences.fetchSourceKeyOffsets[fetchSource + 1], key);
    return (found != keys + sequences.fetchSourceKeyOffsets[fetchSource + 1] && *found == key) ? sequences.entries + sequences.entryOffsets[found - keys] : NULL;
}

static const size_t PUBLISH_SIZE = 4 * 1024 * 1024;
static const time_t PUBLISH_SECONDS = 1;
static const time_t READ_SECONDS = 1;
// read the assigned keys a few times a second
static const double BATCH_READ_SECONDS = PUBLISH_SECONDS / 4.0;
static const double PREFILTER_BUFFER_WORK_SECONDS = 2.0 * PUBLISH_SECONDS;
static const size_t ALIGN_BATCHES_PER_THREAD = 64;
static const size_t ALIGN_BATCH_LANES = 16;
static const int CLUSTER_THREADS = 4;
static const uint64_t ACCEPTED_REP_BIT = 1ULL << 63;

static size_t getRepOrderKey(uint64_t word) {
    return word & ~ACCEPTED_REP_BIT;
}
static size_t getAssignedKeysHeaderWords() {
    return 2 * MMseqsNode::count + 2;
}

// fractions of the memory limit
static const size_t MASTER_MEMORY_DIVISOR = 2;
static const size_t PREFILTER_BUFFER_MEMORY_DIVISOR = 32;
static const size_t FETCH_MEMORY_DIVISOR = 8;
// a record's reads fit the smallest memory limit divided by this
static const size_t RECORD_MEMORY_DIVISOR = 32;
static const size_t MEMBER_SPLIT_MEMORY_DIVISOR = 16;
static const size_t MEMBER_BATCH_SIZE = 4096;
static const size_t STEP_MEMORY_DIVISOR = 4;
static const size_t STEP_PAIRS_MEMORY_DIVISOR = 16;
static const size_t LOG_READ_MEMORY_DIVISOR = 64;
static const size_t CLUSTERED_LOG_BUFFER_SIZE = 16 * 1024 * 1024;
static const size_t CLUSTER_PAIRS_MEMORY_DIVISOR = 16;
static const size_t SAMPLED_KEY_MEMORY_DIVISOR = 64;
static const size_t MIN_ENTRIES_PER_SAMPLED_KEY = 256;
static const size_t MEMBER_READ_SIZE = 1024 * 1024;
static const size_t FETCHED_KEY_SIZE = 4 * sizeof(DBKeyType) + sizeof(unsigned int) + sizeof(size_t);
// same on every node whatever its page size
static const size_t ASSIGNED_PAGE_KEYS = 32768;
// the first fetch is small, then they double
static const size_t FIRST_FETCH_SIZE = 64 * 1024 * 1024;
static const size_t SERVED_FILE_MEMORY_DIVISOR = 512;
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
        const size_t groupLen = std::min<size_t>(8, entryLen - i);
        uint64_t bits = 0;
        for (size_t j = 0; j < groupLen; j++) {
            const uint64_t code = packedEntryCodes.codes[static_cast<unsigned char>(entry[i + j])];
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
        const size_t groupLen = std::min<size_t>(8, entryLen - i);
        uint64_t bits = 0;
        for (size_t byte = 0; byte < getPackedEntrySize(groupLen); byte++) {
            bits |= static_cast<uint64_t>(static_cast<unsigned char>(packedEntry[i / 8 * 5 + byte])) << (8 * byte);
        }
        for (size_t j = 0; j < groupLen; j++) {
            entry[i + j] = PACKED_ENTRY_ALPHABET[(bits >> (5 * j)) & 31];
        }
    }
}

static void unpackEntries(const char *servedEntries, size_t keyCount, const size_t *entryOffsets, char *data, int threads) {
    const char *packedEntries = servedEntries + (keyCount + 7) / 8;
    std::vector<size_t> packedEntryOffsets(keyCount + 1, 0);
    for (size_t i = 0; i < keyCount; i++) {
        const bool isAssigned = (static_cast<unsigned char>(servedEntries[i / 8]) >> (i % 8)) & 1;
        packedEntryOffsets[i + 1] = packedEntryOffsets[i] + (isAssigned ? 0 : getPackedEntrySize(entryOffsets[i + 1] - entryOffsets[i]));
    }
    std::vector<std::vector<size_t> > threadRawEntries(threads);
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
#pragma omp for schedule(static)
        for (size_t i = 0; i < keyCount; i++) {
            if (packedEntryOffsets[i + 1] == packedEntryOffsets[i]) {
                memset(data + entryOffsets[i], 0, entryOffsets[i + 1] - entryOffsets[i]);
            } else if ((static_cast<unsigned char>(packedEntries[packedEntryOffsets[i]]) & 31) == PACKED_RAW_CODE) {
                threadRawEntries[thread_idx].push_back(i);
            } else {
                unpackEntry(packedEntries + packedEntryOffsets[i], entryOffsets[i + 1] - entryOffsets[i], data + entryOffsets[i]);
            }
        }
    }
    const char *rawEntry = packedEntries + packedEntryOffsets[keyCount];
    for (int thread = 0; thread < threads; thread++) {
        for (size_t j = 0; j < threadRawEntries[thread].size(); j++) {
            const size_t i = threadRawEntries[thread][j];
            memcpy(data + entryOffsets[i], rawEntry, entryOffsets[i + 1] - entryOffsets[i]);
            rawEntry += entryOffsets[i + 1] - entryOffsets[i];
        }
    }
}

static const size_t SERVICE_POLL_NODES = 8;
static const size_t SERVICE_OPENS_PER_SECOND = 10000;
static const size_t MAX_SERVICE_PASS_MICROSECONDS = 1280000;

// prefix sums in parallel
static void sumOffsets(size_t *offsets, size_t count, int threads) {
    std::vector<size_t> threadOffsets(threads + 1, 0);
#pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threads; thread++) {
        const size_t end = count * (thread + 1) / threads + 1;
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

static size_t getStepWords(size_t masterMemoryLimit) {
    return std::max<size_t>(1, masterMemoryLimit / STEP_PAIRS_MEMORY_DIVISOR / (sizeof(uint64_t) + 4 * sizeof(std::pair<DBKeyType, size_t>)));
}

struct ClusterMemberLookup {
    struct Entry {
        size_t node;
        size_t offset;
        size_t end;
        DBKeyType key;
    };
    struct EntryReader {
        Entry entry;
        size_t bufferBegin;
        size_t bufferEnd;
        std::vector<char> buffer;

        void start(const Entry &first) {
            entry = first;
            bufferBegin = 0;
            bufferEnd = 0;
        }
    };

    ClusterMemberLookup(const std::string &clu, const std::string &sampledKeyData, size_t entriesPerSampledKey)
        : entriesPerSampledKey(entriesPerSampledKey), nodeCluDbrs(MMseqsNode::count), cluKeyOffsets(MMseqsNode::count + 1, SIZE_MAX), nodeSampledKeyOffsets(MMseqsNode::count + 1, 0),
          sampledKeys(sampledKeyData.size() / sizeof(DBKeyType)) {
        memcpy(sampledKeys.data(), sampledKeyData.data(), sampledKeyData.size());
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            nodeCluDbrs[node] = new DBReader<DBKeyType>((clu + "." + SSTR(node)).c_str(), (clu + ".index." + SSTR(node)).c_str(), 1,
                                                        DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
            nodeCluDbrs[node]->open(DBReader<DBKeyType>::NOSORT);
            nodeSampledKeyOffsets[node + 1] = nodeSampledKeyOffsets[node] + (nodeCluDbrs[node]->getSize() + entriesPerSampledKey - 1) / entriesPerSampledKey;
        }
        // a node file without entries starts where the next one starts
        for (size_t node = MMseqsNode::count; node > 0; node--) {
            cluKeyOffsets[node - 1] = (nodeSampledKeyOffsets[node - 1] < nodeSampledKeyOffsets[node]) ? static_cast<size_t>(sampledKeys[nodeSampledKeyOffsets[node - 1]]) : cluKeyOffsets[node];
        }
        cluKeyOffsets[0] = 0;
    }

    ~ClusterMemberLookup() {
        for (size_t node = 0; node < nodeCluDbrs.size(); node++) {
            nodeCluDbrs[node]->close();
            delete nodeCluDbrs[node];
        }
    }

    Entry find(DBKeyType key) const {
        const size_t node = std::upper_bound(cluKeyOffsets.begin(), cluKeyOffsets.end(), key) - cluKeyOffsets.begin() - 1;
        Entry entry = { node, 0, 0, key };
        const DBKeyType *begin = sampledKeys.data() + nodeSampledKeyOffsets[node];
        const size_t sampledKey = std::lower_bound(begin, sampledKeys.data() + nodeSampledKeyOffsets[node + 1], key) - begin;
        // the entry is between the sampled keys around key
        const size_t found = nodeCluDbrs[node]->findEntryByKey(key, (sampledKey > 0) ? (sampledKey - 1) * entriesPerSampledKey : 0, std::min(sampledKey * entriesPerSampledKey + 1, nodeCluDbrs[node]->getSize()));
        if (found < nodeCluDbrs[node]->getSize()) {
            const DBReader<DBKeyType>::Index index = nodeCluDbrs[node]->getIndexEntry(found);
            if (index.id == key) {
                entry.offset = index.offset;
                entry.end = index.offset + index.length - 1;
            }
        }
        return entry;
    }

    bool next(EntryReader &entryReader, DBKeyType &memberKey) const {
        Entry &entry = entryReader.entry;
        while (entry.offset < entry.end) {
            const char *lineEnd = (entry.offset < entryReader.bufferEnd)
                                  ? static_cast<const char *>(memchr(entryReader.buffer.data() + (entry.offset - entryReader.bufferBegin), '\n', entryReader.bufferEnd - entry.offset)) : NULL;
            if (lineEnd == NULL) {
                const size_t bytes = std::min(entry.end - entry.offset, MEMBER_READ_SIZE);
                entryReader.buffer.resize(bytes);
                FileUtil::preadAll(nodeCluDbrs[entry.node]->getBufferedFd(), nodeCluDbrs[entry.node]->getDataFileName(), entryReader.buffer.data(), bytes, entry.offset);
                entryReader.bufferBegin = entry.offset;
                entryReader.bufferEnd = entry.offset + bytes;
                continue;
            }
            const char *line = entryReader.buffer.data() + (entry.offset - entryReader.bufferBegin);
            const DBKeyType key = Util::fast_atoi<DBKeyType>(line);
            entry.offset += lineEnd - line + 1;
            if (key != entry.key) {
                memberKey = key;
                return true;
            }
        }
        return false;
    }

    size_t memorySize() const {
        return sizeof(*this) + cluKeyOffsets.capacity() * sizeof(size_t) + nodeCluDbrs.size() * sizeof(DBReader<DBKeyType>) + nodeSampledKeyOffsets.capacity() * sizeof(size_t)
               + sampledKeys.capacity() * sizeof(DBKeyType);
    }

private:
    const size_t entriesPerSampledKey;
    std::vector<DBReader<DBKeyType> *> nodeCluDbrs;
    std::vector<size_t> cluKeyOffsets;
    std::vector<size_t> nodeSampledKeyOffsets;
    std::vector<DBKeyType> sampledKeys;
};

struct PrefRead {
    size_t prefNode;
    size_t firstEntry;
    size_t endEntry;
    uint64_t sizeClasses;
};

struct RepTargetsReader {
    RepTargetsReader(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<PrefPartition> &partitions, const std::vector<size_t> &alignmentSplitPartitionOffsets,
                     MMseqsNode::WorkQueue &alignmentSplitQueue, size_t maxRecordSize, size_t memoryLimit, size_t paddedDbSize, size_t sizeClassCount)
        : maxRecordSize(maxRecordSize), par(par), seqDbr(seqDbr), partitions(partitions), alignmentSplitPartitionOffsets(alignmentSplitPartitionOffsets), alignmentSplitQueue(alignmentSplitQueue),
          paddedDbSize(paddedDbSize), sizeClassCount(sizeClassCount), maxPrefilterBufferSize(std::max<size_t>(1, memoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR)),
          maxRecordTargetCount(std::min<size_t>(UINT_MAX, std::max<size_t>(1, maxRecordSize / std::max<size_t>(seqDbr.getMaxSeqLen() + 2, Util::getPageSize())))),
          prefReadPos(0), passClasses(~0ULL), endClass(0), classSize(sizeClassCount, 0), prefDb(NULL), prefNode(0), bufferRecordCount(0), bufferTargetCount(0), bufferFetchSize(0), bufferAlignmentCells(0),
          entryTargetOffsets(NULL), entryRecordOffsets(NULL), allocatedEntries(0) {}

    bool takeAlignmentSplit() {
        // per pref byte two buffers and 8 bytes of targets, per entry 48 bytes and a record
        const double bytesPerPrefByte = 10.0;
        const double bytesPerEntry = 48.0 + sizeof(RepTargetsRecord);
        const size_t alignmentSplitCount = alignmentSplitPartitionOffsets.size() - 1;
        const size_t prefReadsBefore = prefReads.size();
        while (prefReads.size() == prefReadsBefore) {
            const size_t alignmentSplit = (MMseqsNode::rank >= getFirstAlignNode()) ? alignmentSplitQueue.takeNextItemInOrder() : alignmentSplitCount;
            if (alignmentSplit == alignmentSplitCount) {
                break;
            }
            for (size_t i = alignmentSplitPartitionOffsets[alignmentSplit]; i < alignmentSplitPartitionOffsets[alignmentSplit + 1];) {
                if (partitions[i].alignmentCells == 0) {
                    i++;
                    continue;
                }
                const size_t node = partitions[i].node;
                const size_t prefBegin = partitions[i].prefOffset;
                openPrefDb(node);
                // read neighbouring partitions together
                size_t prefEnd = prefBegin;
                for (; i < alignmentSplitPartitionOffsets[alignmentSplit + 1] && partitions[i].node == node && partitions[i].prefOffset == prefEnd; i++) {
                    prefEnd += partitions[i].prefSize;
                    bufferAlignmentCells += partitions[i].alignmentCells;
                }
                const size_t endEntry = prefDb->findEntryByOffset(prefEnd);
                const double entriesPerByte = static_cast<double>(endEntry - prefDb->findEntryByOffset(prefBegin)) / (prefEnd - prefBegin);
                const size_t maxPrefReadSize = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(maxPrefilterBufferSize) / (bytesPerPrefByte + bytesPerEntry * entriesPerByte)));
                const size_t firstPrefRead = prefReads.size();
                for (size_t offset = prefBegin; offset < prefEnd; offset += maxPrefReadSize) {
                    const size_t entry = prefDb->findEntryByOffset(offset);
                    if (prefReads.size() == firstPrefRead || prefReads.back().firstEntry != entry) {
                        const PrefRead prefRead = { node, entry, entry, ~0ULL };
                        prefReads.push_back(prefRead);
                    }
                }
                for (size_t prefRead = firstPrefRead; prefRead < prefReads.size(); prefRead++) {
                    prefReads[prefRead].endEntry = (prefRead + 1 < prefReads.size()) ? prefReads[prefRead + 1].firstEntry : endEntry;
                }
            }
        }
        return prefReads.size() > prefReadsBefore;
    }

    void countSizeClasses() {
        std::vector<RepTargetsRecord> records;
        std::vector<DBKeyType> targets;
        for (prefReadPos = 0; prefReadPos < prefReads.size() || takeAlignmentSplit();) {
            PrefRead &prefRead = prefReads[prefReadPos];
            bufferRecordCount = 0;
            bufferTargetCount = 0;
            loadPrefRead(records, targets);
            uint64_t sizeClasses = 0;
#pragma omp parallel
            {
                std::vector<size_t> threadClassSize(sizeClassCount, 0);
#pragma omp for schedule(dynamic, 100) reduction(|:sizeClasses)
                for (size_t i = 0; i < bufferRecordCount; i++) {
                    const RepTargetsRecord &record = records[i];
                    const size_t sizeClass = sizeClassCount - 1 - record.repOrderKeyFirstTarget.repKey / paddedDbSize;
                    sizeClasses |= 1ULL << sizeClass;
                    threadClassSize[sizeClass] += sizeof(RepTargetsRecord) + record.targetCount * sizeof(DBKeyType);
                }
#pragma omp critical
                for (size_t sizeClass = 0; sizeClass < sizeClassCount; sizeClass++) {
                    classSize[sizeClass] += threadClassSize[sizeClass];
                }
            }
            prefRead.sizeClasses = sizeClasses;
        }
        endClass = sizeClassCount;
    }

    void fillPrefilterBuffer(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, size_t maxReadSize, size_t maxBufferAlignmentCells) {
        records.reserve(maxPrefilterBufferSize / sizeof(RepTargetsRecord));
        targets.reserve(maxPrefilterBufferSize / sizeof(DBKeyType));
        bufferRecordCount = 0;
        bufferTargetCount = 0;
        bufferFetchSize = 0;
        bufferAlignmentCells = 0;
        while (bufferRecordCount * sizeof(RepTargetsRecord) + bufferTargetCount * sizeof(DBKeyType) < maxPrefilterBufferSize) {
            if (prefReadPos < prefReads.size()) {
                loadPrefRead(records, targets);
            } else if (bufferRecordCount > 0 && (bufferFetchSize >= maxReadSize || bufferAlignmentCells >= maxBufferAlignmentCells)) {
                break;
            } else if (takeAlignmentSplit() == false && (bufferRecordCount > 0 || startPass() == false)) {
                break;
            }
        }
        records.resize(bufferRecordCount);
        targets.resize(bufferTargetCount);
        SORT_PARALLEL(records.begin(), records.end(), RepTargetsRecord::compareByRep);
    }

    void close() {
        closePrefDb();
        delete[] entryTargetOffsets;
        delete[] entryRecordOffsets;
    }

    size_t getTakenItemCount() const {
        return alignmentSplitQueue.getTakenItemCount();
    }

    const size_t maxRecordSize;

private:
    Parameters &par;
    DBReader<DBKeyType> &seqDbr;
    const std::vector<PrefPartition> &partitions;
    const std::vector<size_t> &alignmentSplitPartitionOffsets;
    MMseqsNode::WorkQueue &alignmentSplitQueue;
    const size_t paddedDbSize;
    const size_t sizeClassCount;
    const size_t maxPrefilterBufferSize;
    const size_t maxRecordTargetCount;
    std::vector<PrefRead> prefReads;
    size_t prefReadPos;
    uint64_t passClasses;
    size_t endClass;
    std::vector<size_t> classSize;
    DBReader<DBKeyType> *prefDb;
    size_t prefNode;
    size_t bufferRecordCount;
    size_t bufferTargetCount;
    size_t bufferFetchSize;
    size_t bufferAlignmentCells;
    // not zeroed, each pref read overwrites its part
    size_t *entryTargetOffsets;
    size_t *entryRecordOffsets;
    size_t allocatedEntries;

    void openPrefDb(size_t node) {
        if (prefDb != NULL && prefNode == node) {
            return;
        }
        closePrefDb();
        prefNode = node;
        prefDb = new DBReader<DBKeyType>((par.db2 + "." + SSTR(node)).c_str(), (par.db2 + ".index." + SSTR(node)).c_str(), par.threads,
                                         DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        prefDb->open(DBReader<DBKeyType>::NOSORT);
    }

    void closePrefDb() {
        if (prefDb != NULL) {
            prefDb->close();
            delete prefDb;
            prefDb = NULL;
        }
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

    void loadPrefRead(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets) {
        const PrefRead &prefRead = prefReads[prefReadPos];
        openPrefDb(prefRead.prefNode);
        prefDb->loadData(prefRead.firstEntry, prefRead.endEntry);
        prefReadPos = findPassPrefRead(prefReadPos + 1);
        // prefetch here, io_uring reads can fail once their thread ends
        if (prefReadPos < prefReads.size() && prefReads[prefReadPos].prefNode == prefRead.prefNode) {
            prefDb->prefetchData(prefReads[prefReadPos].firstEntry, prefReads[prefReadPos].endEntry);
        }
        const size_t firstEntry = prefRead.firstEntry;
        const size_t entryCount = prefRead.endEntry - firstEntry;
        if (allocatedEntries < entryCount + 1) {
            delete[] entryTargetOffsets;
            delete[] entryRecordOffsets;
            allocatedEntries = entryCount + 1;
            entryTargetOffsets = new(std::nothrow) size_t[allocatedEntries];
            entryRecordOffsets = new(std::nothrow) size_t[allocatedEntries];
            Util::checkAllocation(entryTargetOffsets, "Can not allocate entryTargetOffsets memory in align2clust");
            Util::checkAllocation(entryRecordOffsets, "Can not allocate entryRecordOffsets memory in align2clust");
        }
        entryTargetOffsets[0] = 0;
        entryRecordOffsets[0] = 0;
#pragma omp parallel for schedule(dynamic, 100)
        for (size_t entry = 0; entry < entryCount; entry++) {
            const char *data = prefDb->getData(firstEntry + entry, 0);
            const size_t targetCount = std::count_if(data, data + prefDb->getEntryLen(firstEntry + entry) - 1, [](char byte) { return static_cast<unsigned char>(byte) < 0x80; }) - 1;
            const bool isInPass = ((passClasses >> getSizeClass(targetCount, sizeClassCount)) & 1) != 0;
            entryTargetOffsets[entry + 1] = isInPass ? targetCount : 0;
            entryRecordOffsets[entry + 1] = isInPass ? (targetCount + maxRecordTargetCount - 1) / maxRecordTargetCount : 0;
        }
        sumOffsets(entryTargetOffsets, entryCount, par.threads);
        sumOffsets(entryRecordOffsets, entryCount, par.threads);
        const size_t firstRecord = bufferRecordCount;
        const size_t firstTarget = bufferTargetCount;
        bufferRecordCount += entryRecordOffsets[entryCount];
        bufferTargetCount += entryTargetOffsets[entryCount];
        records.resize(std::max(records.size(), bufferRecordCount));
        targets.resize(std::max(targets.size(), bufferTargetCount));
        size_t prefReadFetchSize = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+:prefReadFetchSize)
        for (size_t entry = 0; entry < entryCount; entry++) {
            const size_t targetCount = entryTargetOffsets[entry + 1] - entryTargetOffsets[entry];
            if (targetCount == 0) {
                continue;
            }
            const char *data = prefDb->getData(firstEntry + entry, 0);
            const DBKeyType repKey = Util::getPrefKey(data);
            const size_t repSize = seqDbr.getEntryLen(repKey) + FETCHED_KEY_SIZE;
            DBKeyType *entryTargets = targets.data() + firstTarget + entryTargetOffsets[entry];
            for (size_t j = 0; j < targetCount; j++) {
                entryTargets[j] = ((j > 0) ? entryTargets[j - 1] : 0) + Util::getPrefKey(data);
            }
            size_t j = firstRecord + entryRecordOffsets[entry];
            for (size_t first = 0; first < targetCount; first += maxRecordTargetCount, j++) {
                RepTargetsRecord &record = records[j];
                record.repOrderKeyFirstTarget.repKey = (sizeClassCount - 1 - getSizeClass(targetCount, sizeClassCount)) * paddedDbSize + repKey;
                record.repOrderKeyFirstTarget.memberKey = entryTargets[first];
                record.targetCount = static_cast<unsigned int>(std::min(targetCount - first, maxRecordTargetCount));
                record.targetOffset = entryTargets + first - targets.data();
                record.readSize = repSize;
                for (unsigned int i = 0; i < record.targetCount; i++) {
                    record.readSize += seqDbr.getEntryLen(entryTargets[first + i]) + FETCHED_KEY_SIZE;
                }
                prefReadFetchSize += record.readSize;
            }
        }
        bufferFetchSize += prefReadFetchSize;
    }
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
          memberTargets(par, scoring.subMat, scoring.evaluer, maxSeqLen, true),
          memberChecks(isSwipeEnabled ? &swipeMembers : NULL, true, isSwipeEnabled ? &memberTargets : NULL) {}
};

static void alignRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, RepTargetsReader &repTargetsReader,
                            size_t memoryLimit, size_t masterMemoryLimit, size_t servedFileEntrySize, size_t representativeResidues,
                            size_t paddedDbSize, size_t sizeClassCount, ClusterMemberLookup *memberLookup, DBReader<DBKeyType> *memberDbr, bool hasLocalServedBlocks, std::string &phaseValues) {
    const unsigned int nodes = MMseqsNode::count;
    const unsigned int rank = MMseqsNode::rank;
    const size_t firstAlignNode = getFirstAlignNode();
    const size_t alignNodeCount = nodes - firstAlignNode;
    // a service thread besides --threads fetches and serves the sequences
    const bool isAligningNode = rank >= firstAlignNode;
    const std::string assignedKeysLog = par.db3 + "_assigned_keys";
    const std::string requests = par.db3 + "_sequence_requests";
    const std::string served = par.db3 + "_served_sequences";
    const size_t dbSize = seqDbr.getSize();
    const size_t orderKeyEnd = (sizeClassCount - 1) * paddedDbSize + dbSize;
    const size_t maxSeqLen = seqDbr.getMaxSeqLen();
    const size_t maxMemberSeqLen = (memberDbr != NULL) ? std::max(maxSeqLen, memberDbr->getMaxSeqLen()) : maxSeqLen;
    const size_t memberSplitSize = memoryLimit / MEMBER_SPLIT_MEMORY_DIVISOR;
    const size_t pageSize = Util::getPageSize();
    AlignmentScoring scoring(par, representativeResidues);
    const unsigned int swMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);
    const std::string acceptedFile = par.db3 + "_accepted." + SSTR(rank);
    FILE *acceptedHandle = FileUtil::openAndDelete(acceptedFile.c_str(), "w");
    size_t acceptedSize = 0;
    std::vector<uint64_t> acceptedWords;
    DBKeyType loggedRepOrderKey = DB_KEY_INVALID;
    const auto publishAccepted = [&](size_t repKey, size_t bytes) {
        MMseqsNode::publishLog(acceptedHandle, acceptedFile, repKey, bytes, repTargetsReader.getTakenItemCount());
    };

    // assigned bits of the served and aligned keys, by page
    std::vector<uint64_t *> assignedPages(isAligningNode ? (dbSize + ASSIGNED_PAGE_KEYS - 1) / ASSIGNED_PAGE_KEYS : 0, NULL);
    const auto isAssigned = [&](size_t orderKey) {
        const size_t key = orderKey % paddedDbSize;
        const uint64_t *page = __atomic_load_n(&assignedPages[key / ASSIGNED_PAGE_KEYS], __ATOMIC_ACQUIRE);
        return page != NULL && ((__atomic_load_n(&page[key % ASSIGNED_PAGE_KEYS / 64], __ATOMIC_RELAXED) >> (key % 64)) & 1) != 0;
    };
    // the master drops pairs with assigned keys anyway
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
    const size_t headerWords = getAssignedKeysHeaderWords();
    const size_t assignedKeysReadWords = 1024 * 1024;
    size_t assignedKeysSize = 0;
    std::vector<uint64_t> assignedKeyWords;
    // without wait, an unreadable step stays for the next call
    const auto readAssignedKeys = [&](bool wait) {
        size_t publishedKey;
        size_t publishedSize;
        if (MMseqsNode::readLogProgress(assignedKeysLog, publishedKey, publishedSize) == false) {
            return;
        }
        while (assignedKeysSize < publishedSize) {
            assignedKeyWords.resize(headerWords);
            if (MMseqsNode::readLog(assignedKeysLog, assignedKeysSize, assignedKeysSize + headerWords * sizeof(uint64_t), (char *) assignedKeyWords.data(), wait) == false) {
                return;
            }
            const size_t stepClusteredKey = assignedKeyWords[0];
            const size_t stepClusteredSize = assignedKeyWords[1 + MMseqsNode::rank];
            const size_t nodeKeyOffset = assignedKeyWords[1 + MMseqsNode::count + MMseqsNode::rank];
            const size_t nodeKeyEnd = assignedKeyWords[2 + MMseqsNode::count + MMseqsNode::rank];
            const size_t keysStart = assignedKeysSize + headerWords * sizeof(uint64_t);
            const size_t stepEnd = keysStart + assignedKeyWords[1 + 2 * MMseqsNode::count] * sizeof(uint64_t);
            for (size_t first = nodeKeyOffset; first < nodeKeyEnd; first += assignedKeysReadWords) {
                const size_t count = std::min(assignedKeysReadWords, nodeKeyEnd - first);
                assignedKeyWords.resize(count);
                if (MMseqsNode::readLog(assignedKeysLog, keysStart + first * sizeof(uint64_t), keysStart + (first + count) * sizeof(uint64_t), (char *) assignedKeyWords.data(), wait) == false) {
                    return;
                }
                // only the thread that holds assignedKeysMutex adds pages
                for (size_t i = 0; i < count; i++) {
                    const size_t key = assignedKeyWords[i];
                    uint64_t *page = assignedPages[key / ASSIGNED_PAGE_KEYS];
                    if (page == NULL) {
                        page = new(std::nothrow) uint64_t[ASSIGNED_PAGE_KEYS / 64]();
                        Util::checkAllocation(page, "Can not allocate assigned page memory in align2clust");
                        __atomic_store_n(&assignedPages[key / ASSIGNED_PAGE_KEYS], page, __ATOMIC_RELEASE);
                    }
                    __atomic_fetch_or(&page[key % ASSIGNED_PAGE_KEYS / 64], 1ULL << (key % 64), __ATOMIC_RELAXED);
                }
            }
            assignedKeysSize = stepEnd;
            clusteredOrderKey = stepClusteredKey;
            clusteredAcceptedSize = stepClusteredSize;
        }
    };

    // the loader fills buffer n once the fetches use buffer n - 1
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
    std::vector<DBKeyType> uniqueKeys;
    std::vector<unsigned int> keyFetchSources;
    std::vector<size_t> threadFetchSourceKeyOffsets;
    // accepted pairs waiting for their member checks, in log order
    std::vector<ClusterPair> pendingPairs;
    std::vector<size_t> pendingPairSize;
    std::vector<size_t> pendingMemberOffsets(1, 0);
    std::vector<DBKeyType> pendingMembers;
    // coverable bytes of a rep per serving node
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const size_t rangeCount = seqDbr.getIndexSize();
    std::vector<size_t> rangeByteOffsets(rangeCount + 1, 0);
    for (size_t range = 0; range < rangeCount; range++) {
        const size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
        rangeByteOffsets[range + 1] = rangeByteOffsets[range] + (keyEnd - index[range].id) * index[range].length;
    }
    double coverableBytesSum = 0;
    for (size_t range = 0; range < rangeCount; range++) {
        const float repLength = static_cast<float>(index[range].length - 2);
        const auto isCovered = [&](const DBReader<DBKeyType>::Index &coveredRange) {
            return Util::canBeCovered(par.covThr, par.covMode, repLength, static_cast<float>(coveredRange.length - 2));
        };
        // ranges descend in length, a rep covers an interval of lengths
        const size_t coveredFirst = std::partition_point(index, index + range, [&](const DBReader<DBKeyType>::Index &coveredRange) { return isCovered(coveredRange) == false; }) - index;
        const size_t coveredEnd = std::partition_point(index + range, index + rangeCount, isCovered) - index;
        const size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
        coverableBytesSum += static_cast<double>(keyEnd - index[range].id) * (rangeByteOffsets[coveredEnd] - rangeByteOffsets[coveredFirst]);
    }
    const double servingNodeCoverableSize = coverableBytesSum / std::max<size_t>(1, dbSize) / alignNodeCount;
    const size_t averageEntrySize = std::max<size_t>(1, rangeByteOffsets[rangeCount] / std::max<size_t>(1, dbSize));
    const size_t maxServedKeys = std::max<size_t>(1, servedFileEntrySize / FETCHED_KEY_SIZE);
    const size_t maxServedFileEntrySize = servedFileEntrySize + SERVICE_POLL_NODES * (maxSeqLen + 2);
    const size_t maxServedFileSize = SERVICE_POLL_NODES * ((maxServedKeys + 7) / 8) + getPackedEntrySize(maxServedFileEntrySize) + maxServedFileEntrySize;
    // read directly once a cache miss costs less than two packed copies
    const auto isCacheTooSmall = [&](size_t cacheSize) {
        return alignNodeCount > 1 && static_cast<double>(cacheSize) * pageSize <= 2.0 * getPackedEntrySize(averageEntrySize) * servingNodeCoverableSize;
    };
    // memory besides the block cache
    const auto getReservedMemory = [&](size_t fetchSize) {
        return ((memberLookup != NULL) ? memberLookup->memorySize() + (par.threads + 1) * MEMBER_READ_SIZE : 0) + 3 * 2 * (memoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR)
               + ((memberDbr != NULL) ? 2 * memberSplitSize : 0)
               + 2 * std::max(fetchSize, repTargetsReader.maxRecordSize) + fetchSize
               + ((isAligningNode && alignNodeCount > 1) ? 3 * maxServedFileSize + 6 * maxServedKeys * sizeof(size_t) : 0);
    };
    // largest fetch that leaves a block cache for the coverable sequences
    const bool canCache = isCacheTooSmall(memoryLimit - std::min(memoryLimit, getReservedMemory(0))) == false;
    const size_t servedSequenceSize = rangeByteOffsets[rangeCount] / alignNodeCount;
    size_t maxFetchSize = memoryLimit / FETCH_MEMORY_DIVISOR;
    while (maxFetchSize > 1 && (getReservedMemory(maxFetchSize) > memoryLimit || (canCache && isCacheTooSmall(memoryLimit - getReservedMemory(maxFetchSize)))
                                 || (alignNodeCount > 1 && maxFetchSize > repTargetsReader.maxRecordSize && memoryLimit - std::min(memoryLimit, getReservedMemory(maxFetchSize)) < servedSequenceSize))) {
        maxFetchSize /= 2;
    }
    const size_t blockCacheSize = memoryLimit - std::min(memoryLimit, getReservedMemory(maxFetchSize));
    seqDbr.setBlockCacheSize(blockCacheSize);
    const size_t maxPendingPairSize = std::min(memberSplitSize, maxFetchSize);
    // the copy on --local-tmp holds only the served blocks
    const bool isReadingDirectly = hasLocalServedBlocks == false && isCacheTooSmall(blockCacheSize);
    if (isAligningNode) {
        MMseqsNode::addPhaseValue(phaseValues, "Block cache", SSTR(blockCacheSize) + " bytes for " + SSTR(static_cast<size_t>(servingNodeCoverableSize)) + " coverable bytes and " + SSTR(servedSequenceSize) + " sequence bytes of the serving node"
                                 + (isReadingDirectly ? ", the sequences are read directly" : ""));
    }
    size_t fetchBytesLimit = std::min(maxFetchSize / 64, FIRST_FETCH_SIZE);
    const auto getFetchedEntry = [&](const FetchedSequences &sequences, size_t key) -> const char * {
        const char *data = findFetchedKey(sequences, nodes, key);
        return (data != NULL) ? data : findFetchedKey(sequences, isReadingDirectly ? rank : getKeyServingNode(seqDbr, key), key);
    };
    const auto getFetchedSequence = [&](const FetchedSequences &sequences, size_t key) -> const char * {
        const char *data = getFetchedEntry(sequences, key);
        // assigned keys are served as zeros
        return (data != NULL && data[0] != '\0') ? data : NULL;
    };
    // take records while they fit, at least one
    const auto selectRecordKeys = [&](FetchedSequences &sequences, const FetchedSequences &previous) {
        // an empty prefilter buffer is the last one
        if (prefilterRecordPos[prefilterBuffer] == prefilterRecords[prefilterBuffer].size() && prefilterRecords[prefilterBuffer].empty() == false) {
            Timer waitTimer;
            std::unique_lock<std::mutex> lock(prefilterBufferMutex);
            prefilterBufferCondition.wait(lock, [&]() { return filledPrefilterBufferCount > usedPrefilterBufferCount; });
            prefilterBuffer = usedPrefilterBufferCount++ % 3;
            lock.unlock();
            prefilterBufferCondition.notify_all();
            prefilterWaitSeconds += waitTimer.getTimediff();
            prefilterRecordPos[prefilterBuffer] = 0;
        }
        sequences.prefilterBuffer = prefilterBuffer;
        sequences.begin = prefilterRecordPos[prefilterBuffer];
        sequences.end = sequences.begin;
        // pending pairs first, at least one
        size_t fetchSize = 0;
        sequences.pendingPairCount = 0;
        for (size_t i = previous.pendingPairCount; i < pendingPairs.size() && (sequences.pendingPairCount == 0 || fetchSize + pendingPairSize[i] <= fetchBytesLimit); i++) {
            fetchSize += pendingPairSize[i];
            sequences.pendingPairCount++;
        }
        sequences.firstPendingRepKey = (sequences.pendingPairCount > 0) ? static_cast<size_t>(pendingPairs[previous.pendingPairCount].repKey) : orderKeyEnd;
        for (; sequences.end < prefilterRecords[prefilterBuffer].size(); sequences.end++) {
            fetchSize += prefilterRecords[prefilterBuffer][sequences.end].readSize;
            if (sequences.end > sequences.begin && fetchSize > fetchBytesLimit) {
                break;
            }
        }
        prefilterRecordPos[prefilterBuffer] = sequences.end;
        // previous may still add pending pairs
        sequences.isLast = sequences.end == sequences.begin && sequences.pendingPairCount == 0 && (memberDbr == NULL || previous.end == previous.begin);
        fetchBytesLimit = std::min(2 * fetchBytesLimit, maxFetchSize);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            threadKeys[thread_idx].clear();
#pragma omp for schedule(dynamic, 100)
            for (size_t i = sequences.begin; i < sequences.end; i++) {
                const RepTargetsRecord &record = prefilterRecords[sequences.prefilterBuffer][i];
                const size_t repKey = record.repOrderKeyFirstTarget.repKey % paddedDbSize;
                const char *previousRep = getFetchedEntry(previous, repKey);
                if (isAssigned(record.repOrderKeyFirstTarget.repKey) || (previousRep != NULL && previousRep[0] == '\0')) {
                    continue;
                }
                const size_t targetsBegin = threadKeys[thread_idx].size();
                const DBKeyType *targets = prefilterTargets[sequences.prefilterBuffer].data() + record.targetOffset;
                for (unsigned int j = 0; j < record.targetCount; j++) {
                    if (isAssigned(targets[j]) == false) {
                        threadKeys[thread_idx].push_back(targets[j]);
                    }
                }
                // a rep without targets left is not fetched
                if (threadKeys[thread_idx].size() > targetsBegin) {
                    threadKeys[thread_idx].push_back(repKey);
                }
            }
            // members share the keys and data files of the sequence database
#pragma omp for schedule(dynamic, 100)
            for (size_t i = previous.pendingPairCount; i < previous.pendingPairCount + sequences.pendingPairCount; i++) {
                if (isAssigned(pendingPairs[i].repKey) == false && isAssigned(pendingPairs[i].memberKey) == false) {
                    threadKeys[thread_idx].push_back(pendingPairs[i].repKey % paddedDbSize);
                    threadKeys[thread_idx].insert(threadKeys[thread_idx].end(), pendingMembers.begin() + pendingMemberOffsets[i], pendingMembers.begin() + pendingMemberOffsets[i + 1]);
                }
            }
        }
        uniqueKeys.clear();
        for (size_t thread = 0; thread < threadKeys.size(); thread++) {
            uniqueKeys.insert(uniqueKeys.end(), threadKeys[thread].begin(), threadKeys[thread].end());
        }
        SORT_PARALLEL(uniqueKeys.begin(), uniqueKeys.end());
        uniqueKeys.erase(std::unique(uniqueKeys.begin(), uniqueKeys.end()), uniqueKeys.end());
        keyFetchSources.resize(uniqueKeys.size());
        sequences.keys.resize(uniqueKeys.size());
        // sort the keys by fetch source; assigned keys go last and are dropped
        threadFetchSourceKeyOffsets.assign((nodes + 2) * par.threads + 1, 0);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<size_t> fetchSourceOffsets(nodes + 2, 0);
#pragma omp for schedule(static)
            for (size_t i = 0; i < uniqueKeys.size(); i++) {
                keyFetchSources[i] = isAssigned(uniqueKeys[i]) ? nodes + 1 : ((getFetchedEntry(previous, uniqueKeys[i]) != NULL) ? nodes : (isReadingDirectly ? rank : getKeyServingNode(seqDbr, uniqueKeys[i])));
                fetchSourceOffsets[keyFetchSources[i]]++;
            }
            for (unsigned int node = 0; node <= nodes + 1; node++) {
                threadFetchSourceKeyOffsets[node * par.threads + thread_idx + 1] = fetchSourceOffsets[node];
            }
#pragma omp barrier
#pragma omp single
            for (size_t i = 0; i + 1 < threadFetchSourceKeyOffsets.size(); i++) {
                threadFetchSourceKeyOffsets[i + 1] += threadFetchSourceKeyOffsets[i];
            }
            for (unsigned int node = 0; node <= nodes + 1; node++) {
                fetchSourceOffsets[node] = threadFetchSourceKeyOffsets[node * par.threads + thread_idx];
            }
#pragma omp for schedule(static)
            for (size_t i = 0; i < uniqueKeys.size(); i++) {
                sequences.keys[fetchSourceOffsets[keyFetchSources[i]]++] = uniqueKeys[i];
            }
        }
        sequences.fetchSourceKeyOffsets.resize(nodes + 2);
        for (unsigned int node = 0; node <= nodes + 1; node++) {
            sequences.fetchSourceKeyOffsets[node] = threadFetchSourceKeyOffsets[node * par.threads];
        }
        const size_t keyCount = sequences.fetchSourceKeyOffsets[nodes + 1];
        sequences.keys.resize(keyCount);
        sequences.entryOffsets.resize(keyCount + 1);
        sequences.entryOffsets[0] = 0;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < keyCount; i++) {
            sequences.entryOffsets[i + 1] = seqDbr.getEntryLen(sequences.keys[i]);
        }
        sumOffsets(sequences.entryOffsets.data(), keyCount, par.threads);
        // not zeroed, the entries overwrite it
        if (sequences.allocatedSize < sequences.entryOffsets[keyCount]) {
            delete[] sequences.entries;
            sequences.allocatedSize = sequences.entryOffsets[keyCount];
            sequences.entries = new(std::nothrow) char[sequences.allocatedSize];
            Util::checkAllocation(sequences.entries, "Can not allocate fetched sequence memory in align2clust");
        }
#pragma omp parallel for schedule(dynamic, 1024)
        for (size_t i = sequences.fetchSourceKeyOffsets[nodes]; i < keyCount; i++) {
            memcpy(sequences.entries + sequences.entryOffsets[i], getFetchedEntry(previous, sequences.keys[i]), sequences.entryOffsets[i + 1] - sequences.entryOffsets[i]);
        }
    };
    const auto getFirstRepKey = [&](const FetchedSequences &sequences) {
        return (sequences.end > sequences.begin) ? static_cast<size_t>(prefilterRecords[sequences.prefilterBuffer][sequences.begin].repOrderKeyFirstTarget.repKey) : orderKeyEnd;
    };

    FetchedSequences currentSequences;
    FetchedSequences nextSequences;
    nextSequences.fetchSourceKeyOffsets.assign(nodes + 2, 0);
    // the service thread and the loader use a quarter of the threads
    const int serviceThreads = std::max(1, par.threads / 4);
    std::atomic<double> alignedCellsPerSecond(0);
    repTargetsReader.fillPrefilterBuffer(prefilterRecords[0], prefilterTargets[0], maxFetchSize, 0);
    filledPrefilterBufferCount = 1;
    usedPrefilterBufferCount = 1;
    // the loader ends with the first empty prefilter buffer
    std::thread prefilterLoader([&]() {
#ifdef OPENMP
        omp_set_num_threads(serviceThreads);
#endif
        for (size_t next = 1; ; next++) {
            {
                std::unique_lock<std::mutex> lock(prefilterBufferMutex);
                prefilterBufferCondition.wait(lock, [&]() { return usedPrefilterBufferCount >= next; });
            }
            repTargetsReader.fillPrefilterBuffer(prefilterRecords[next % 3], prefilterTargets[next % 3], maxFetchSize, static_cast<size_t>(alignedCellsPerSecond.load() * PREFILTER_BUFFER_WORK_SECONDS));
            const bool isLast = prefilterRecords[next % 3].empty();
            {
                std::lock_guard<std::mutex> lock(prefilterBufferMutex);
                filledPrefilterBufferCount = next + 1;
            }
            prefilterBufferCondition.notify_all();
            if (isLast) {
                break;
            }
        }
    });
    selectRecordKeys(currentSequences, nextSequences);
    publishAccepted(getFirstRepKey(currentSequences), 0);
    size_t publishedSize = 0;
    time_t publishTime = time(NULL);
    time_t assignedKeysTime = publishTime;
    Timer assignedKeysTimer;
    double lastBatchReadTime = 0;
    std::mutex acceptedMutex;
    std::mutex publishMutex;
    std::mutex assignedKeysMutex;
    std::vector<size_t> batchOffsets;
    std::vector<unsigned int> batchFirstTargets;
    std::vector<size_t> recordAlignmentCells;
    std::vector<unsigned char> batchDone;
    std::vector<std::vector<ClusterPair> > batchPairs;
    std::vector<ClusterPair> acceptedPairs;
    std::vector<size_t> memberCounts;
    std::vector<size_t> memberSize;
    std::vector<unsigned char> membersPassed;
    std::vector<unsigned char> needsMemberCheck;
    std::vector<DBKeyType> memberSplitKeys;
    std::vector<DBKeyType> memberSplitMembers;
    std::vector<size_t> repPairOffsets;
    std::vector<size_t> batchRepOffsets;
    double waitSeconds = 0;
    double sequenceWaitSeconds = 0;
    double alignedCells = 0;
    double alignSeconds = 0;
    const auto mapRep = [&](Sequence &query, BlockAligner &blockAligner, size_t repKey) {
        query.mapSequence(repKey, repKey, getFetchedSequence(currentSequences, repKey), seqDbr.getSeqLen(repKey));
        blockAligner.setQuery(&query);
    };
    const bool isSwipeEnabled = par.align2clustAlignerMode != Parameters::ALIGNER_MODE_FARRAR && par.gapOpen.values.aminoacid() >= par.gapExtend.values.aminoacid();

    std::mutex serviceMutex;
    std::condition_variable serviceCondition;
    FetchedSequences *pendingSequences = NULL;
    // a node aligns ahead of the master by at most its share of a master step
    size_t requestAcceptedSize = 0;
    const size_t maxUnclusteredAcceptedSize = std::max<size_t>(1, getStepWords(masterMemoryLimit) / nodes) * sizeof(uint64_t);
    FetchedSequences *completedSequences = NULL;
    bool isWaitingForMaster = false;
    std::vector<DBKeyType> loadKeys;
    std::vector<std::vector<size_t> > threadRawEntries(serviceThreads);
    size_t servedFileCount = 0;
    size_t servedSize = 0;
    size_t receivedServedEntriesSize = 0;
    size_t requestCount = 0;
    double servingSeconds = 0;
    double servedFileWriterWaitSeconds = 0;
    double readingSeconds = 0;
    double pollingSeconds = 0;
    // copy, or pack with rawKeys, the entries of keys; assigned keys get zeros
    const auto copyEntries = [&](const DBKeyType *keys, size_t keyCount, const size_t *offsets, char *data, std::vector<DBKeyType> *rawKeys) {
        for (size_t first = 0; first < keyCount;) {
            const size_t loadFirst = first;
            // a load fits maxFetchSize
            for (size_t loadSize = 0; first < keyCount && (first == loadFirst || loadSize + seqDbr.getEntryLen(keys[first]) + 2 * pageSize <= maxFetchSize); first++) {
                loadSize += seqDbr.getEntryLen(keys[first]) + 2 * pageSize;
            }
            loadKeys.clear();
            for (size_t i = loadFirst; i < first; i++) {
                if (isAssigned(keys[i]) == false) {
                    loadKeys.push_back(keys[i]);
                }
            }
            seqDbr.loadData(loadKeys);
            for (int thread = 0; thread < serviceThreads; thread++) {
                threadRawEntries[thread].clear();
            }
#pragma omp parallel num_threads(serviceThreads)
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
#pragma omp for schedule(static)
                for (size_t i = loadFirst; i < first; i++) {
                    // bits are never cleared
                    if (isAssigned(keys[i])) {
                        memset(data + offsets[i], 0, offsets[i + 1] - offsets[i]);
                    } else if (rawKeys == NULL) {
                        memcpy(data + offsets[i], seqDbr.getData(keys[i], 0), offsets[i + 1] - offsets[i]);
                    } else if (packEntry(seqDbr.getData(keys[i], 0), seqDbr.getEntryLen(keys[i]), data + offsets[i]) == false) {
                        data[offsets[i]] = PACKED_RAW_CODE;
                        threadRawEntries[thread_idx].push_back(i);
                    }
                }
            }
            // the static parts of the threads follow in key order
            for (int thread = 0; rawKeys != NULL && thread < serviceThreads; thread++) {
                for (size_t j = 0; j < threadRawEntries[thread].size(); j++) {
                    rawKeys->push_back(keys[threadRawEntries[thread][j]]);
                }
            }
        }
    };
    std::thread serviceThread;
    if (isAligningNode) {
        serviceThread = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(1);
#endif
            std::vector<std::string> requestIndexes(nodes);
            std::vector<std::string> requestFiles(nodes);
            std::vector<std::string> servedIndexes(nodes);
            for (unsigned int node = firstAlignNode; node < nodes; node++) {
                requestIndexes[node] = requests + ".index." + SSTR(node);
                requestFiles[node] = requests + "." + SSTR(node);
                servedIndexes[node] = served + ".index." + SSTR(node);
            }
            // request index line: request, isLast, key offsets per node, end, magic
            std::vector<uint64_t> requestLine(nodes + 4);
            // served index line: file, entry offsets per node, end, key ends per node, magic
            const size_t servedLineWords = 2 * nodes + 3;
            std::vector<uint64_t> servedLine(servedLineWords);
            std::vector<uint64_t> servedLines(nodes * servedLineWords);
            std::vector<size_t> nodeRequests(nodes, 0);
            std::vector<size_t> nextServedIndexLines(nodes, 0);
            std::vector<size_t> servedFileNumbers(nodes, SIZE_MAX);
            std::vector<size_t> servedEntriesBegins(nodes, 0);
            std::vector<size_t> servedEntriesEnds(nodes, 0);
            std::vector<size_t> polledServedFileSize(nodes, 0);
            std::vector<size_t> servedKeyEnds(nodes, 0);
            std::vector<size_t> servingNodeReceivedKeyCount(nodes, 0);
            std::vector<unsigned char> isRequesting(nodes, 0);
            for (size_t node = firstAlignNode; node < nodes; node++) {
                isRequesting[node] = (node != rank);
            }
            const size_t polledNodes = std::min<size_t>(SERVICE_POLL_NODES, nodes - 1);
            // idle and busy passes open the same number of indexes
            const size_t pollMicroseconds = std::min(MAX_SERVICE_PASS_MICROSECONDS, 2 * polledNodes * nodes * 1000000 / SERVICE_OPENS_PER_SECOND);
            Timer serviceClock;
            std::vector<double> nextServedIndexPolls(nodes, 0);
            std::vector<double> nextRequestIndexPolls(nodes, 0);
            std::vector<time_t> missingServedFileSince(nodes, 0);
            std::vector<unsigned int> missingServedFileAttempts(nodes, 0);
            std::vector<time_t> missingRequestSince(nodes, 0);
            std::vector<unsigned int> missingRequestAttempts(nodes, 0);
            struct ServedRequest {
                unsigned int node;
                size_t requestFileNumber;
                size_t codeStart;
                size_t codeOffset;
                size_t codeEnd;
                size_t requestSize;
                size_t servedKeys;
                DBKeyType lastKey;
            };
            std::vector<ServedRequest> servedRequests;
            std::vector<unsigned char> isServing(nodes, 0);
            std::vector<DBKeyType> requestKeys;
            std::vector<char> requestCodes((alignNodeCount > 1) ? maxServedKeys * sizeof(DBKeyType) + 10 : 0);
            std::vector<std::vector<size_t> > nodeUnreadServedFiles(nodes);
            std::map<size_t, size_t> servedFileUnreadNodes;
            std::vector<size_t> requestOffsets;
            std::vector<char> servedFileData;
            // a thread writes the served file while the next one is built
            std::thread servedFileWriter;
            std::vector<char> writtenServedFileData;
            std::vector<DBKeyType> rawKeys;
            std::vector<char> receivedEntries;
            struct ServedEntriesRead {
                int fd;
                unsigned int servingNode;
                size_t receivedOffset;
                size_t bytes;
            };
            std::vector<ServedEntriesRead> servedEntriesReads;
            MMseqsNode::WaitWarning waitWarning;
            FetchedSequences *requestedSequences = NULL;
            bool isLastRequested = false;
            time_t assignedKeysReadTime = 0;
            size_t pollTurn = 0;
            for (unsigned int polls = 0; isLastRequested == false || requestedSequences != NULL || std::find(isRequesting.begin(), isRequesting.end(), 1) != isRequesting.end()
                                         || servedRequests.empty() == false;) {
                bool hasWorked = false;
                // keep the bits of the served keys current
                const time_t passTime = time(NULL);
                if (passTime - assignedKeysReadTime >= READ_SECONDS) {
                    std::unique_lock<std::mutex> assignedKeysLock(assignedKeysMutex, std::try_to_lock);
                    if (assignedKeysLock.owns_lock()) {
                        readAssignedKeys(false);
                        assignedKeysReadTime = passTime;
                    }
                }
                // request once the master is close enough behind
                FetchedSequences *nextRequest = NULL;
                size_t nextAcceptedSize = 0;
                if (requestedSequences == NULL) {
                    std::lock_guard<std::mutex> lock(serviceMutex);
                    nextRequest = pendingSequences;
                    nextAcceptedSize = requestAcceptedSize;
                }
                if (nextRequest != NULL) {
                    const bool isLast = nextRequest->isLast;
                    bool isAhead;
                    {
                        std::lock_guard<std::mutex> assignedKeysLock(assignedKeysMutex);
                        const size_t repKey = std::min(nextRequest->firstPendingRepKey, getFirstRepKey(*nextRequest));
                        if (isLast == false && repKey > clusteredOrderKey && nextAcceptedSize >= clusteredAcceptedSize + maxUnclusteredAcceptedSize) {
                            readAssignedKeys(true);
                        }
                        isAhead = isLast == false && repKey > clusteredOrderKey && nextAcceptedSize >= clusteredAcceptedSize + maxUnclusteredAcceptedSize;
                    }
                    {
                        std::lock_guard<std::mutex> lock(serviceMutex);
                        isWaitingForMaster = isAhead;
                        if (isAhead == false) {
                            pendingSequences = NULL;
                        }
                    }
                    if (isAhead == false) {
                        if (alignNodeCount > 1) {
                            FILE *requestHandle = MMseqsNode::createExchangeFile(requestFiles[rank], requestCount);
                            size_t requestSize = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES;
                            for (unsigned int node = 0; node < nodes; node++) {
                                requestLine[2 + node] = requestSize;
                                const size_t keyBegin = nextRequest->fetchSourceKeyOffsets[node];
                                const size_t keyEnd = (node != rank) ? nextRequest->fetchSourceKeyOffsets[node + 1] : keyBegin;
                                // keys ascend, so send their gaps
                                for (size_t i = keyBegin; i < keyEnd;) {
                                    size_t codeSize = 0;
                                    for (; i < keyEnd && codeSize + 10 <= requestCodes.size(); i++) {
                                        codeSize += Util::putPrefKey(requestCodes.data() + codeSize, nextRequest->keys[i] - ((i > keyBegin) ? nextRequest->keys[i - 1] : 0));
                                    }
                                    if (fwrite(requestCodes.data(), sizeof(char), codeSize, requestHandle) != codeSize) {
                                        Debug(Debug::ERROR) << "Cannot write " << requestFiles[rank] << "\n";
                                        EXIT(EXIT_FAILURE);
                                    }
                                    requestSize += codeSize;
                                }
                            }
                            requestLine[0] = requestCount++;
                            requestLine[1] = isLast;
                            requestLine[2 + nodes] = requestSize;
                            MMseqsNode::publishExchangeFile(requestHandle, requestFiles[rank], requestSize, requestIndexes[rank], requestLine);
                        }
                        const size_t ownKey = nextRequest->fetchSourceKeyOffsets[rank];
                        copyEntries(nextRequest->keys.data() + ownKey, nextRequest->fetchSourceKeyOffsets[rank + 1] - ownKey, nextRequest->entryOffsets.data() + ownKey, nextRequest->entries, NULL);
                        isLastRequested = isLast;
                        requestedSequences = nextRequest;
                        hasWorked = true;
                    }
                }
                // receive served entries from up to SERVICE_POLL_NODES nodes
                if (requestedSequences != NULL) {
                    bool isComplete = true;
                    size_t polledServingNodeCount = 0;
                    size_t receivedSize = 0;
                    for (size_t i = 0; i + 1 < nodes; i++) {
                        const unsigned int servingNode = (rank + 1 + (i + pollTurn) % (nodes - 1)) % nodes;
                        const size_t servingNodeKeyCount = requestedSequences->fetchSourceKeyOffsets[servingNode + 1] - requestedSequences->fetchSourceKeyOffsets[servingNode];
                        if (servingNodeReceivedKeyCount[servingNode] == servingNodeKeyCount) {
                            continue;
                        }
                        if (++polledServingNodeCount > polledNodes) {
                            isComplete = false;
                            continue;
                        }
                        // stop at the line with entries for this node
                        if (servedFileNumbers[servingNode] == SIZE_MAX) {
                            if (serviceClock.getTimediff() < nextServedIndexPolls[servingNode]) {
                                isComplete = false;
                                continue;
                            }
                            Timer pollTimer;
                            const size_t lineCount = MMseqsNode::readExchangeIndexLines(servedIndexes[servingNode], nextServedIndexLines[servingNode], servedLineWords, servedLines);
                            pollingSeconds += pollTimer.getTimediff();
                            if (lineCount == 0) {
                                nextServedIndexPolls[servingNode] = serviceClock.getTimediff() + pollMicroseconds / 1e6;
                            }
                            for (size_t line = 0; line < lineCount && servedFileNumbers[servingNode] == SIZE_MAX; line++) {
                                const uint64_t *servedEntriesOffsets = servedLines.data() + line * servedLineWords + 1;
                                if (servedEntriesOffsets[rank] < servedEntriesOffsets[rank + 1]) {
                                    servedFileNumbers[servingNode] = nextServedIndexLines[servingNode];
                                    servedEntriesBegins[servingNode] = servedEntriesOffsets[rank];
                                    servedEntriesEnds[servingNode] = servedEntriesOffsets[rank + 1];
                                    polledServedFileSize[servingNode] = servedEntriesOffsets[nodes];
                                    servedKeyEnds[servingNode] = servedEntriesOffsets[nodes + 1 + rank];
                                }
                                nextServedIndexLines[servingNode]++;
                            }
                            if (servedFileNumbers[servingNode] == SIZE_MAX) {
                                isComplete = false;
                                continue;
                            }
                        }
                        if (servedEntriesReads.empty() == false && receivedSize + servedEntriesEnds[servingNode] - servedEntriesBegins[servingNode] > maxServedFileSize) {
                            isComplete = false;
                            continue;
                        }
                        const std::string servedFile = served + "." + SSTR(servingNode) + "." + SSTR(servedFileNumbers[servingNode]);
                        size_t observedSize;
                        uint64_t observedExchangeFileNumber;
                        int lastError;
                        Timer openTimer;
                        const int fd = MMseqsNode::tryOpenExchangeFile(servedFile, polledServedFileSize[servingNode], servedFileNumbers[servingNode], observedSize, observedExchangeFileNumber, lastError);
                        readingSeconds += openTimer.getTimediff();
                        if (fd < 0) {
                            if (lastError != 0 && lastError != ENOENT && lastError != ESTALE) {
                                Debug(Debug::ERROR) << "Cannot open served sequence file " << servedFile << ": " << strerror(lastError) << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                            failIfMissingTooLong(missingServedFileSince[servingNode], missingServedFileAttempts[servingNode], "Cannot open complete served sequence file " + servedFile
                                                 + describeServedFile(polledServedFileSize[servingNode], observedSize, servedFileNumbers[servingNode], observedExchangeFileNumber, lastError));
                            isComplete = false;
                            continue;
                        }
                        const ServedEntriesRead request = { fd, servingNode, receivedSize, servedEntriesEnds[servingNode] - servedEntriesBegins[servingNode] };
                        servedEntriesReads.push_back(request);
                        receivedSize += request.bytes;
                    }
                    if (isComplete || servedEntriesReads.empty() == false) {
                        if (servedEntriesReads.empty() == false) {
                            Timer readTimer;
                            if (receivedEntries.size() < receivedSize) {
                                receivedEntries.resize(receivedSize);
                            }
                            // read in parts; a failed part keeps its error
                            std::vector<std::pair<size_t, size_t> > readRequests;
                            for (size_t i = 0; i < servedEntriesReads.size(); i++) {
                                for (size_t offset = 0; offset < servedEntriesReads[i].bytes; offset += BatchReader::MAX_READ_REQUEST_SIZE) {
                                    readRequests.push_back(std::make_pair(i, offset));
                                }
                            }
                            std::vector<int> readErrors(readRequests.size(), 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(serviceThreads)
                            for (size_t i = 0; i < readRequests.size(); i++) {
                                const ServedEntriesRead &request = servedEntriesReads[readRequests[i].first];
                                const size_t offset = readRequests[i].second;
                                int readError;
                                if (MMseqsNode::tryPreadAll(request.fd, receivedEntries.data() + request.receivedOffset + offset, std::min(BatchReader::MAX_READ_REQUEST_SIZE, request.bytes - offset),
                                                            servedEntriesBegins[request.servingNode] + offset, readError) == false) {
                                    readErrors[i] = (readError != 0) ? readError : -1;
                                }
                            }
                            readingSeconds += readTimer.getTimediff();
                            for (size_t i = 0, readRequest = 0; i < servedEntriesReads.size(); i++) {
                                int readError = 0;
                                for (; readRequest < readRequests.size() && readRequests[readRequest].first == i; readRequest++) {
                                    readError = (readErrors[readRequest] != 0) ? readErrors[readRequest] : readError;
                                }
                                const unsigned int servingNode = servedEntriesReads[i].servingNode;
                                const std::string servedFile = served + "." + SSTR(servingNode) + "." + SSTR(servedFileNumbers[servingNode]);
                                const int closeError = (close(servedEntriesReads[i].fd) != 0) ? errno : 0;
                                const int error = (readError > 0) ? readError : closeError;
                                if (error != 0 && error != ESTALE && error != EINTR) {
                                    Debug(Debug::ERROR) << "Cannot read served sequence file " << servedFile << ": " << strerror(error) << "\n";
                                    EXIT(EXIT_FAILURE);
                                }
                                if (readError != 0 || closeError != 0) {
                                    failIfMissingTooLong(missingServedFileSince[servingNode], missingServedFileAttempts[servingNode],
                                                         "Cannot read complete served sequence file " + servedFile + " (file number " + SSTR(servedFileNumbers[servingNode]) + "; error " + SSTR(error) + ")\n");
                                    isComplete = false;
                                } else {
                                    const size_t firstKey = requestedSequences->fetchSourceKeyOffsets[servingNode] + servingNodeReceivedKeyCount[servingNode];
                                    unpackEntries(receivedEntries.data() + servedEntriesReads[i].receivedOffset, servedKeyEnds[servingNode] - servingNodeReceivedKeyCount[servingNode],
                                                  requestedSequences->entryOffsets.data() + firstKey, requestedSequences->entries, serviceThreads);
                                    servingNodeReceivedKeyCount[servingNode] = servedKeyEnds[servingNode];
                                    servedFileNumbers[servingNode] = SIZE_MAX;
                                    missingServedFileAttempts[servingNode] = 0;
                                    receivedServedEntriesSize += servedEntriesReads[i].bytes;
                                    hasWorked = true;
                                    if (servingNodeReceivedKeyCount[servingNode] < requestedSequences->fetchSourceKeyOffsets[servingNode + 1] - requestedSequences->fetchSourceKeyOffsets[servingNode]) {
                                        isComplete = false;
                                    }
                                }
                            }
                        }
                        servedEntriesReads.clear();
                        if (isComplete) {
                            servingNodeReceivedKeyCount.assign(nodes, 0);
                            {
                                std::lock_guard<std::mutex> lock(serviceMutex);
                                completedSequences = requestedSequences;
                            }
                            serviceCondition.notify_all();
                            requestedSequences = NULL;
                            hasWorked = true;
                        }
                    }
                }
                // take the next requests of the polled nodes
                for (size_t i = 0; i < polledNodes; i++) {
                    const unsigned int node = (rank + 1 + (i + pollTurn) % (nodes - 1)) % nodes;
                    while (servedRequests.size() < SERVICE_POLL_NODES && isRequesting[node] && isServing[node] == 0 && serviceClock.getTimediff() >= nextRequestIndexPolls[node]) {
                        Timer pollTimer;
                        const bool hasLine = MMseqsNode::readExchangeIndexLines(requestIndexes[node], nodeRequests[node], requestLine.size(), requestLine) == 1;
                        pollingSeconds += pollTimer.getTimediff();
                        if (hasLine == false) {
                            nextRequestIndexPolls[node] = serviceClock.getTimediff() + pollMicroseconds / 1e6;
                            break;
                        }
                        // a served file goes once all nodes have read it
                        for (size_t j = 0; j < nodeUnreadServedFiles[node].size(); j++) {
                            const size_t unreadServedFile = nodeUnreadServedFiles[node][j];
                            if (--servedFileUnreadNodes[unreadServedFile] == 0) {
                                FileUtil::remove((served + "." + SSTR(rank) + "." + SSTR(unreadServedFile)).c_str());
                                servedFileUnreadNodes.erase(unreadServedFile);
                            }
                        }
                        nodeUnreadServedFiles[node].clear();
                        if (requestLine[1] != 0) {
                            isRequesting[node] = 0;
                        } else if (requestLine[3 + rank] > requestLine[2 + rank]) {
                            const ServedRequest request = { node, nodeRequests[node], requestLine[2 + rank], requestLine[2 + rank], requestLine[3 + rank], requestLine[2 + nodes], 0, 0 };
                            servedRequests.push_back(request);
                            isServing[node] = 1;
                        }
                        nodeRequests[node]++;
                        hasWorked = true;
                    }
                }
                // one served file a pass, in node order
                size_t servedNodeCount = 0;
                size_t servedFileSize = 0;
                std::sort(servedRequests.begin(), servedRequests.end(), [](const ServedRequest &first, const ServedRequest &second) { return first.node < second.node; });
                const size_t requestEntrySize = servedFileEntrySize / std::max<size_t>(1, servedRequests.size());
                for (size_t request = 0, node = 0; node < nodes && servedRequests.empty() == false; node++) {
                    servedLine[1 + node] = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + servedFileSize;
                    servedLine[nodes + 2 + node] = 0;
                    if (request == servedRequests.size() || servedRequests[request].node != node) {
                        continue;
                    }
                    ServedRequest &servedRequest = servedRequests[request++];
                    const size_t readKeys = std::min(maxServedKeys, 2 * requestEntrySize / averageEntrySize + 1);
                    // guess the code bytes of the next keys
                    const double keyCodeSize = (servedRequest.servedKeys > 0) ? static_cast<double>(servedRequest.codeOffset - servedRequest.codeStart) / servedRequest.servedKeys : sizeof(DBKeyType);
                    const size_t codeSize = std::min(std::min(servedRequest.codeEnd - servedRequest.codeOffset, requestCodes.size()), static_cast<size_t>(readKeys * keyCodeSize) + 10);
                    requestKeys.resize(readKeys);
                    size_t observedSize;
                    uint64_t observedExchangeFileNumber;
                    int lastError;
                    Timer requestTimer;
                    const int requestFd = MMseqsNode::tryOpenExchangeFile(requestFiles[node], servedRequest.requestSize, servedRequest.requestFileNumber,
                                                                            observedSize, observedExchangeFileNumber, lastError);
                    if (requestFd < 0 && lastError != 0 && lastError != ENOENT && lastError != ESTALE) {
                        Debug(Debug::ERROR) << "Cannot open sequence request " << requestFiles[node] << ": " << strerror(lastError) << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    bool isRequestRead = requestFd >= 0;
                    if (isRequestRead) {
                        isRequestRead = MMseqsNode::tryPreadAll(requestFd, requestCodes.data(), codeSize, servedRequest.codeOffset, lastError);
                        if (close(requestFd) != 0 && isRequestRead) {
                            lastError = errno;
                            isRequestRead = false;
                        }
                    }
                    servingSeconds += requestTimer.getTimediff();
                    if (isRequestRead == false) {
                        if (lastError != 0 && lastError != ENOENT && lastError != ESTALE && lastError != EINTR) {
                            Debug(Debug::ERROR) << "Cannot read sequence request " << requestFiles[node] << ": " << strerror(lastError) << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        failIfMissingTooLong(missingRequestSince[node], missingRequestAttempts[node], "Cannot read complete sequence request " + requestFiles[node]
                                             + describeServedFile(servedRequest.requestSize, observedSize, servedRequest.requestFileNumber, observedExchangeFileNumber, lastError));
                        continue;
                    }
                    missingRequestAttempts[node] = 0;
                    Timer servingTimer;
                    const DBKeyType *keys = requestKeys.data();
                    // only the last byte of a code is below 0x80
                    const char *codeEnd = requestCodes.data() + codeSize;
                    while (codeEnd > requestCodes.data() && static_cast<unsigned char>(codeEnd[-1]) >= 0x80) {
                        codeEnd--;
                    }
                    const char *code = requestCodes.data();
                    size_t keyCount = 0;
                    for (size_t entrySize = 0; keyCount < readKeys && code < codeEnd; keyCount++) {
                        const char *nextCode = code;
                        const DBKeyType key = ((keyCount > 0) ? keys[keyCount - 1] : servedRequest.lastKey) + Util::getPrefKey(nextCode);
                        const size_t keySize = isAssigned(key) ? 0 : seqDbr.getEntryLen(key);
                        if (keyCount > 0 && entrySize + keySize > requestEntrySize) {
                            break;
                        }
                        requestKeys[keyCount] = key;
                        code = nextCode;
                        entrySize += keySize;
                    }
                    servedRequest.codeOffset += code - requestCodes.data();
                    servedRequest.lastKey = keys[keyCount - 1];
                    requestOffsets.resize(keyCount + 1);
                    requestOffsets[0] = (keyCount + 7) / 8;
#pragma omp parallel for schedule(static) num_threads(serviceThreads)
                    for (size_t j = 0; j < keyCount; j++) {
                        const size_t packedSize = getPackedEntrySize(seqDbr.getEntryLen(keys[j]));
                        const bool isServed = isAssigned(keys[j]) == false;
                        requestOffsets[j + 1] = isServed ? packedSize : 0;
                    }
                    for (size_t j = 0; j < keyCount; j++) {
                        requestOffsets[j + 1] += requestOffsets[j];
                    }
                    if (servedFileData.size() < servedFileSize + requestOffsets[keyCount]) {
                        servedFileData.resize(servedFileSize + requestOffsets[keyCount]);
                    }
                    // an empty slot marks an assigned key
#pragma omp parallel for schedule(static) num_threads(serviceThreads)
                    for (size_t byte = 0; byte < requestOffsets[0]; byte++) {
                        unsigned char bits = 0;
                        for (size_t j = 8 * byte; j < std::min(keyCount, 8 * byte + 8); j++) {
                            bits |= static_cast<unsigned char>(requestOffsets[j + 1] == requestOffsets[j]) << (j % 8);
                        }
                        servedFileData[servedFileSize + byte] = static_cast<char>(bits);
                    }
                    rawKeys.clear();
                    copyEntries(keys, keyCount, requestOffsets.data(), servedFileData.data() + servedFileSize, &rawKeys);
                    servedFileSize += requestOffsets[keyCount];
                    requestOffsets.resize(rawKeys.size() + 1);
                    requestOffsets[0] = 0;
                    for (size_t j = 0; j < rawKeys.size(); j++) {
                        requestOffsets[j + 1] = requestOffsets[j] + seqDbr.getEntryLen(rawKeys[j]);
                    }
                    if (servedFileData.size() < servedFileSize + requestOffsets[rawKeys.size()]) {
                        servedFileData.resize(servedFileSize + requestOffsets[rawKeys.size()]);
                    }
                    copyEntries(rawKeys.data(), rawKeys.size(), requestOffsets.data(), servedFileData.data() + servedFileSize, NULL);
                    servedFileSize += requestOffsets[rawKeys.size()];
                    servingSeconds += servingTimer.getTimediff();
                    servedRequest.servedKeys += keyCount;
                    servedLine[nodes + 2 + node] = servedRequest.servedKeys;
                    nodeUnreadServedFiles[node].push_back(servedFileCount);
                    servedNodeCount++;
                }
                for (size_t request = 0; request < servedRequests.size();) {
                    if (servedRequests[request].codeOffset == servedRequests[request].codeEnd) {
                        isServing[servedRequests[request].node] = 0;
                        servedRequests.erase(servedRequests.begin() + request);
                    } else {
                        request++;
                    }
                }
                if (servedNodeCount > 0) {
                    servedLine[0] = servedFileCount;
                    servedLine[1 + nodes] = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + servedFileSize;
                    servedFileUnreadNodes[servedFileCount] = servedNodeCount;
                    // one served file at a time keeps the index in file order
                    Timer waitTimer;
                    if (servedFileWriter.joinable()) {
                        servedFileWriter.join();
                    }
                    servedFileWriterWaitSeconds += waitTimer.getTimediff();
                    servedFileData.swap(writtenServedFileData);
                    const size_t writtenFileNumber = servedFileCount;
                    servedFileWriter = std::thread([&, servedFileSize, servedLine, writtenFileNumber]() {
                        MMseqsNode::writeExchangeFile(served + "." + SSTR(rank) + "." + SSTR(writtenFileNumber), writtenFileNumber, writtenServedFileData.data(), servedFileSize,
                                                      servedIndexes[rank], servedLine, serviceThreads);
                    });
                    servedFileCount++;
                    servedSize += servedFileSize;
                    hasWorked = true;
                }
                pollTurn += polledNodes;
                // sleep longer while no node waits
                if (hasWorked) {
                    polls = 0;
                } else {
                    usleep(pollMicroseconds);
                    MMseqsNode::sleepBeforePoll(polls++, (alignNodeCount > 1 || requestedSequences != NULL || nextRequest != NULL) ? 4 : 7);
                }
                // aligning without a request is no wait
                if (hasWorked || (requestedSequences == NULL && nextRequest == NULL && isLastRequested == false)) {
                    waitWarning.reset();
                } else {
                    waitWarning.check(requestedSequences != NULL ? served + " of other nodes" : (nextRequest != NULL ? assignedKeysLog : requests + " of other nodes"));
                }
            }
            if (servedFileWriter.joinable()) {
                servedFileWriter.join();
            }
        });
    }
    const auto waitForSequences = [&](FetchedSequences &fetched, size_t doneRepOrderKey) {
        std::unique_lock<std::mutex> lock(serviceMutex);
        if (completedSequences == &fetched) {
            return;
        }
        lock.unlock();
        publishAccepted(doneRepOrderKey, acceptedSize);
        publishedSize = acceptedSize;
        publishTime = time(NULL);
        Timer waitTimer;
        lock.lock();
        const bool waitsForMaster = isWaitingForMaster;
        serviceCondition.wait(lock, [&]() { return completedSequences == &fetched; });
        (waitsForMaster ? waitSeconds : sequenceWaitSeconds) += waitTimer.getTimediff();
    };
    if (isAligningNode) {
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            completedSequences = NULL;
            pendingSequences = &currentSequences;
            requestAcceptedSize = acceptedSize;
        }
        waitForSequences(currentSequences, getFirstRepKey(currentSequences));
    }
    while (currentSequences.isLast == false) {
        selectRecordKeys(nextSequences, currentSequences);
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            completedSequences = NULL;
            pendingSequences = &nextSequences;
            requestAcceptedSize = acceptedSize;
        }
        Timer alignTimer;
        const size_t nextRepKey = getFirstRepKey(nextSequences);
        const std::vector<RepTargetsRecord> &records = prefilterRecords[currentSequences.prefilterBuffer];
        const size_t recordCount = currentSequences.end - currentSequences.begin;
        // batches of about equal alignment cells; a large rep spans several
        const size_t maxBatches = par.threads * (isSwipeEnabled ? ALIGN_BATCHES_PER_THREAD * ALIGN_BATCH_LANES / SwipeExtender::LANES : ALIGN_BATCHES_PER_THREAD);
        recordAlignmentCells.resize(recordCount);
#pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < recordCount; i++) {
            const RepTargetsRecord &record = records[currentSequences.begin + i];
            const DBKeyType *targets = prefilterTargets[currentSequences.prefilterBuffer].data() + record.targetOffset;
            size_t targetRange = SIZE_MAX;
            size_t targetResidues = 0;
            for (unsigned int j = 0; j < record.targetCount; j++) {
                targetResidues += seqDbr.getSeqLenFromRange(targetRange, targets[j]);
            }
            recordAlignmentCells[i] = seqDbr.getSeqLen(record.repOrderKeyFirstTarget.repKey % paddedDbSize) * targetResidues;
        }
        uint128 fetchAlignmentCells = 0;
        for (size_t i = 0; i < recordCount; i++) {
            fetchAlignmentCells += recordAlignmentCells[i];
        }
        batchOffsets.assign(1, currentSequences.begin);
        batchFirstTargets.assign(1, 0);
        uint128 cells = 0;
        for (size_t i = currentSequences.begin; i < currentSequences.end; i++) {
            const uint128 recordEndCells = cells + recordAlignmentCells[i - currentSequences.begin];
            if (recordEndCells * maxBatches > fetchAlignmentCells * batchOffsets.size()) {
                const RepTargetsRecord &record = records[i];
                const DBKeyType *targets = prefilterTargets[currentSequences.prefilterBuffer].data() + record.targetOffset;
                const size_t repLength = seqDbr.getSeqLen(record.repOrderKeyFirstTarget.repKey % paddedDbSize);
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
        const size_t batchCount = batchOffsets.size() - 1;
        batchDone.assign(batchCount, 0);
        batchPairs.assign(batchCount, std::vector<ClusterPair>());
        size_t firstUnfinishedBatch = 0;
#pragma omp parallel
        {
            Sequence query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
            Sequence target(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
            BlockAligner blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                                     &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                                     -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid());
            // --align2clust-aligner-mode 1: the reps of a batch share the SWIPE lanes
            SwipeTargets swipeTargets(par, scoring.subMat, scoring.evaluer, maxSeqLen);
            std::vector<ClusterPair> pairs;
#pragma omp for schedule(dynamic, 1)
            for (size_t batch = 0; batch < batchCount; batch++) {
                {
                    std::unique_lock<std::mutex> assignedKeysLock(assignedKeysMutex, std::try_to_lock);
                    if (assignedKeysLock.owns_lock() && assignedKeysTimer.getTimediff() - lastBatchReadTime >= BATCH_READ_SECONDS) {
                        readAssignedKeys(false);
                        lastBatchReadTime = assignedKeysTimer.getTimediff();
                    }
                }
                pairs.clear();
                for (size_t i = batchOffsets[batch]; i < batchOffsets[batch + 1] + (batchFirstTargets[batch + 1] > 0); i++) {
                    const RepTargetsRecord &record = records[i];
                    const unsigned int endTarget = (i == batchOffsets[batch + 1]) ? batchFirstTargets[batch + 1] : record.targetCount;
                    const size_t repKey = record.repOrderKeyFirstTarget.repKey % paddedDbSize;
                    const char *repData = getFetchedSequence(currentSequences, repKey);
                    if (isAssigned(record.repOrderKeyFirstTarget.repKey) || repData == NULL) {
                        continue;
                    }
                    query.mapSequence(repKey, repKey, repData, seqDbr.getSeqLen(repKey));
                    blockAligner.setQuery(&query);
                    bool blockAlignerInitialized = false;
                    swipeTargets.startRep(record.repOrderKeyFirstTarget.repKey);
                    const DBKeyType *targets = prefilterTargets[currentSequences.prefilterBuffer].data() + record.targetOffset;
                    for (unsigned int j = (i == batchOffsets[batch]) ? batchFirstTargets[batch] : 0; j < endTarget && isAssigned(record.repOrderKeyFirstTarget.repKey) == false; j++) {
                        const DBKeyType targetKey = targets[j];
                        const char *targetData = getFetchedSequence(currentSequences, targetKey);
                        if (isAssigned(targetKey) || targetData == NULL) {
                            continue;
                        }
                        target.mapSequence(targetKey, targetKey, targetData, seqDbr.getSeqLen(targetKey));
                        unsigned short diagonal = 0;
                        if (alignTarget(par, blockAligner, blockAlignerInitialized, query, target, scoring.scorePerColThreshold, diagonal, NULL, 0, NULL,
                                        isSwipeEnabled ? &swipeTargets : NULL)) {
                            const ClusterPair pair = { record.repOrderKeyFirstTarget.repKey, targetKey };
                            pairs.push_back(pair);
                        }
                    }
                }
                alignTargetsSwipe(par, swipeTargets, isAssigned, pairs);
                swipeTargets.clearTargets();
                // the lanes finish the pairs out of rep order
                SORT_SERIAL(pairs.begin(), pairs.end(), ClusterPair::compareByRep);
                // log the pairs in batch order
                bool hasAssignedKeysToRead = false;
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
                        const time_t now = time(NULL);
                        // one thread publishes at a time
                        if ((acceptedSize - publishedSize >= PUBLISH_SIZE || now - publishTime >= PUBLISH_SECONDS) && publishLock.try_lock()) {
                            doneRepOrderKey = (firstUnfinishedBatch < batchCount) ? static_cast<size_t>(records[batchOffsets[firstUnfinishedBatch]].repOrderKeyFirstTarget.repKey) : nextRepKey;
                            doneSize = acceptedSize;
                            publishedSize = acceptedSize;
                            publishTime = now;
                            hasAssignedKeysToRead = doneRepOrderKey < orderKeyEnd;
                        }
                        if (hasAssignedKeysToRead || now - assignedKeysTime >= READ_SECONDS) {
                            hasAssignedKeysToRead = true;
                            assignedKeysTime = now;
                        }
                    }
                }
                if (publishLock.owns_lock()) {
                    publishAccepted(doneRepOrderKey, doneSize);
                    publishLock.unlock();
                }
                if (hasAssignedKeysToRead) {
                    std::unique_lock<std::mutex> assignedKeysLock(assignedKeysMutex, std::try_to_lock);
                    if (assignedKeysLock.owns_lock()) {
                        readAssignedKeys(true);
                    }
                }
            }
        }
        if (memberDbr != NULL) {
            // the pairs assigned by now skip their member checks
            {
                std::unique_lock<std::mutex> assignedKeysLock(assignedKeysMutex, std::try_to_lock);
                if (assignedKeysLock.owns_lock()) {
                    readAssignedKeys(false);
                }
            }
            // without the extensions a batch holds the checks of one rep
            const auto alignMemberChecks = [&](MemberAligners &aligners, unsigned int threadIdx, const std::vector<size_t> &repMemberCheckOffsets, const std::vector<DBKeyType> &repKeys) {
                if (isSwipeEnabled) {
                    alignMemberChecksSwipe(par, aligners.blockAligner, aligners.matcher, aligners.query, aligners.element, memberDbr, swMode, threadIdx, aligners.memberChecks, repMemberCheckOffsets, repKeys, mapRep);
                } else if (aligners.memberChecks.checkCount > 0) {
                    mapRep(aligners.query, aligners.blockAligner, repKeys[0]);
                    bool matcherInitialized = false;
                    alignClusterMembers(par, aligners.blockAligner, aligners.matcher, matcherInitialized, aligners.query, aligners.element, memberDbr, swMode, threadIdx, false, NULL, aligners.memberChecks);
                }
            };
            // check the pending pairs of this fetch; an unserved rep is assigned
            const size_t pairCount = currentSequences.pendingPairCount;
            membersPassed.assign(pairCount, 0);
            ClusterMemberLookup::EntryReader entryReader;
            for (size_t i = 0; i < pairCount; i++) {
                const ClusterPair &pair = pendingPairs[i];
                if (pendingPairSize[i] <= maxPendingPairSize || isAssigned(pair.repKey) || isAssigned(pair.memberKey) || getFetchedSequence(currentSequences, pair.repKey % paddedDbSize) == NULL) {
                    continue;
                }
                // read the members in member splits
                const size_t repKey = pair.repKey % paddedDbSize;
                MemberAligners aligners(par, scoring, maxMemberSeqLen, isSwipeEnabled);
                const std::vector<size_t> repMemberCheckOffsets = {0, 1};
                const std::vector<DBKeyType> repKeys(1, repKey);
                entryReader.start(memberLookup->find(pair.memberKey));
                DBKeyType memberKey;
                bool hasMember = memberLookup->next(entryReader, memberKey);
                bool passed = true;
                while (passed && hasMember) {
                    memberSplitMembers.clear();
                    for (size_t splitSize = 0; hasMember; hasMember = memberLookup->next(entryReader, memberKey)) {
                        const size_t entrySize = std::max(memberDbr->getEntryLen(memberKey), pageSize);
                        if (entrySize > memberSplitSize) {
                            Debug(Debug::ERROR) << "Cluster member sequence " << memberKey << " exceeds the " << memberSplitSize << " byte member split\n";
                            EXIT(EXIT_FAILURE);
                        }
                        if (memberSplitMembers.empty() == false && entrySize > memberSplitSize - splitSize) {
                            break;
                        }
                        memberSplitMembers.push_back(memberKey);
                        splitSize += entrySize;
                    }
                    memberSplitKeys = memberSplitMembers;
                    SORT_PARALLEL(memberSplitKeys.begin(), memberSplitKeys.end());
                    memberSplitKeys.erase(std::unique(memberSplitKeys.begin(), memberSplitKeys.end()), memberSplitKeys.end());
                    memberDbr->loadData(memberSplitKeys);
                    aligners.memberChecks.checkCount = 0;
                    MemberCheck &check = aligners.memberChecks.addCheck(i, 0);
                    check.memberKeys = memberSplitMembers;
                    alignMemberChecks(aligners, 0, repMemberCheckOffsets, repKeys);
                    passed = check.isPassing;
                }
                membersPassed[i] = passed;
            }
            // the pairs of a rep share the query
            repPairOffsets.clear();
            for (size_t i = 0; i < pairCount; i++) {
                if (i == 0 || pendingPairs[i].repKey != pendingPairs[i - 1].repKey) {
                    repPairOffsets.push_back(i);
                }
            }
            repPairOffsets.push_back(pairCount);
            const size_t repCount = repPairOffsets.size() - 1;
            // batches of reps that fill the lanes, 4 per thread
            const size_t batchMemberLimit = std::min(MEMBER_BATCH_SIZE, std::max(pendingMemberOffsets[pairCount] / (par.threads * 4), MEMBER_BATCH_SIZE / 16));
            batchRepOffsets.assign(1, 0);
            for (size_t rep = 0, batchMembers = 0; rep < repCount; rep++) {
                batchMembers += pendingMemberOffsets[repPairOffsets[rep + 1]] - pendingMemberOffsets[repPairOffsets[rep]];
                if (isSwipeEnabled == false || batchMembers >= batchMemberLimit || rep + 1 == repCount) {
                    batchRepOffsets.push_back(rep + 1);
                    batchMembers = 0;
                }
            }
            const size_t memberBatchCount = batchRepOffsets.size() - 1;
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                MemberAligners aligners(par, scoring, maxMemberSeqLen, isSwipeEnabled);
                std::vector<size_t> repMemberCheckOffsets;
                std::vector<DBKeyType> repKeys;
#pragma omp for schedule(dynamic, 1)
                for (size_t batch = 0; batch < memberBatchCount; batch++) {
                    aligners.memberChecks.checkCount = 0;
                    repMemberCheckOffsets.assign(1, 0);
                    repKeys.clear();
                    for (size_t rep = batchRepOffsets[batch]; rep < batchRepOffsets[batch + 1]; rep++) {
                        for (size_t i = repPairOffsets[rep]; i < repPairOffsets[rep + 1]; i++) {
                            const ClusterPair &pair = pendingPairs[i];
                            if (pendingPairSize[i] > maxPendingPairSize || isAssigned(pair.repKey) || isAssigned(pair.memberKey) || getFetchedSequence(currentSequences, pair.repKey % paddedDbSize) == NULL) {
                                continue;
                            }
                            MemberCheck &check = aligners.memberChecks.addCheck(i, 0);
                            for (size_t member = pendingMemberOffsets[i]; member < pendingMemberOffsets[i + 1]; member++) {
                                check.memberKeys.push_back(pendingMembers[member]);
                                check.memberData.push_back(getFetchedEntry(currentSequences, pendingMembers[member]));
                            }
                        }
                        if (aligners.memberChecks.checkCount > repMemberCheckOffsets.back()) {
                            repMemberCheckOffsets.push_back(aligners.memberChecks.checkCount);
                            repKeys.push_back(pendingPairs[repPairOffsets[rep]].repKey % paddedDbSize);
                        }
                    }
                    alignMemberChecks(aligners, thread_idx, repMemberCheckOffsets, repKeys);
                    for (size_t g = 0; g < aligners.memberChecks.checkCount; g++) {
                        membersPassed[aligners.memberChecks.checks[g].targetId] = aligners.memberChecks.checks[g].isPassing;
                    }
                }
            }
            size_t acceptedCount = 0;
            for (size_t i = 0; i < pairCount; i++) {
                if (membersPassed[i]) {
                    pendingPairs[acceptedCount++] = pendingPairs[i];
                }
            }
            logAcceptedPairs(pendingPairs.data(), acceptedCount);
            size_t checkedMembers = pendingMemberOffsets[pairCount];
            pendingPairs.erase(pendingPairs.begin(), pendingPairs.begin() + pairCount);
            pendingPairSize.erase(pendingPairSize.begin(), pendingPairSize.begin() + pairCount);
            pendingMemberOffsets.erase(pendingMemberOffsets.begin(), pendingMemberOffsets.begin() + pairCount);
            pendingMembers.erase(pendingMembers.begin(), pendingMembers.begin() + checkedMembers);
            for (size_t i = 0; i < pendingMemberOffsets.size(); i++) {
                pendingMemberOffsets[i] -= checkedMembers;
            }

            // the accepted pairs wait for a later fetch
            acceptedPairs.clear();
            for (size_t batch = 0; batch < batchCount; batch++) {
                acceptedPairs.insert(acceptedPairs.end(), batchPairs[batch].begin(), batchPairs[batch].end());
            }
            memberCounts.resize(acceptedPairs.size());
            memberSize.resize(acceptedPairs.size());
            needsMemberCheck.resize(acceptedPairs.size());
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                ClusterMemberLookup::EntryReader entryReader;
                threadKeys[thread_idx].clear();
#pragma omp for schedule(static)
                for (size_t i = 0; i < acceptedPairs.size(); i++) {
                    size_t memberBegin = threadKeys[thread_idx].size();
                    size_t repKey = acceptedPairs[i].repKey % paddedDbSize;
                    // an assigned key or an uncoverable member fails without a check
                    needsMemberCheck[i] = isAssigned(acceptedPairs[i].repKey) == false && isAssigned(acceptedPairs[i].memberKey) == false;
                    memberSize[i] = seqDbr.getEntryLen(repKey) + FETCHED_KEY_SIZE;
                    if (needsMemberCheck[i]) {
                        entryReader.start(memberLookup->find(acceptedPairs[i].memberKey));
                    }
                    DBKeyType memberKey;
                    while (needsMemberCheck[i] && memberSize[i] <= maxPendingPairSize && memberLookup->next(entryReader, memberKey)) {
                        needsMemberCheck[i] = canCoverMember(par, static_cast<int>(seqDbr.getSeqLen(repKey)), static_cast<int>(seqDbr.getSeqLen(memberKey)));
                        memberSize[i] += seqDbr.getEntryLen(memberKey) + FETCHED_KEY_SIZE;
                        threadKeys[thread_idx].push_back(memberKey);
                    }
                    if (needsMemberCheck[i] == false || memberSize[i] > maxPendingPairSize) {
                        threadKeys[thread_idx].resize(memberBegin);
                    }
                    memberCounts[i] = threadKeys[thread_idx].size() - memberBegin;
                }
            }
            for (size_t thread = 0; thread < threadKeys.size(); thread++) {
                pendingMembers.insert(pendingMembers.end(), threadKeys[thread].begin(), threadKeys[thread].end());
            }
            for (size_t i = 0; i < acceptedPairs.size(); i++) {
                if (needsMemberCheck[i]) {
                    pendingPairs.push_back(acceptedPairs[i]);
                    pendingPairSize.push_back(memberSize[i]);
                    pendingMemberOffsets.push_back(pendingMemberOffsets.back() + memberCounts[i]);
                }
            }
            publishAccepted(pendingPairs.empty() ? nextRepKey : static_cast<size_t>(pendingPairs[0].repKey), acceptedSize);
            publishedSize = acceptedSize;
            publishTime = time(NULL);
            if (nextRepKey < orderKeyEnd) {
                std::lock_guard<std::mutex> assignedKeysLock(assignedKeysMutex);
                readAssignedKeys(true);
            }
        }
        alignedCells += static_cast<double>(fetchAlignmentCells);
        alignSeconds += alignTimer.getTimediff();
        alignedCellsPerSecond.store(alignedCells / std::max(alignSeconds, 1e-3));
        // the reps of pending pairs are not done
        waitForSequences(nextSequences, pendingPairs.empty() ? nextRepKey : static_cast<size_t>(pendingPairs[0].repKey));
        std::swap(currentSequences, nextSequences);
    }
    prefilterLoader.join();
    publishAccepted(orderKeyEnd, acceptedSize);
    if (fclose(acceptedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << acceptedFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    // serve until every aligning node is done
    if (isAligningNode) {
        serviceThread.join();
        MMseqsNode::addPhaseValue(phaseValues, "Sequence exchange bytes", SSTR(servedSize) + " served, " + SSTR(receivedServedEntriesSize) + " received, " + SSTR(requestCount) + " requests");
        MMseqsNode::addPhaseSeconds(phaseValues, "service: serving the requests", servingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: waiting for the served file writer", servedFileWriterWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: reading the served entries", readingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: polling", pollingSeconds);
    }
    delete[] currentSequences.entries;
    delete[] nextSequences.entries;
    for (size_t page = 0; page < assignedPages.size(); page++) {
        delete[] assignedPages[page];
    }
    MMseqsNode::addPhaseValue(phaseValues, "Block read bytes", SSTR(seqDbr.getBlockReadSize()));
    MMseqsNode::addPhaseValue(phaseValues, "Key read bytes", SSTR(seqDbr.getKeyReadSize()));
    // the master still finds the nodes of keys in the blocks of seqDbr
    seqDbr.releaseDataPages();
    if (isAligningNode) {
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the master", waitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the sequences", sequenceWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the prefilter", prefilterWaitSeconds);
    }
}

struct AcceptedRep {
    size_t repKey;
    size_t begin;
    size_t end;
    unsigned int node;
};

struct MergedReps {
    size_t firstNodeRep;
    size_t firstMergedRep;
    unsigned int node;
};

struct ClusteredLog {
    FILE *file;
    size_t bytes;
    uint64_t partialWord;
    unsigned int partialWordBits;
    char *buffer;
};

static size_t getKeyServingNodeFromPage(const std::vector<unsigned int> &pageServingNodes, const DBReader<DBKeyType> &seqDbr, DBKeyType key) {
    return (pageServingNodes[key / ASSIGNED_PAGE_KEYS] != UINT_MAX) ? pageServingNodes[key / ASSIGNED_PAGE_KEYS] : getKeyServingNode(seqDbr, key);
}

// linfinity: the master clusters the accepted pairs in key order
static size_t clusterAcceptedPairs(int threads, size_t dbSize, size_t memoryLimit, const std::vector<size_t> &alignmentSplitKeyOffsets, const MMseqsNode::WorkQueue &alignmentSplitQueue, const DBReader<DBKeyType> &seqDbr,
                                   const std::vector<unsigned int> &pageServingNodes, int mode, size_t paddedDbSize, size_t sizeClassCount,
                                   const std::string &accepted, const std::string &assignedKeysLog, const std::string &clusteredLogPrefix, const std::string &assignedBits, std::string &phaseValues) {
    const size_t assignedWords = (dbSize + 63) / 64;
    const size_t orderKeyEnd = (sizeClassCount - 1) * paddedDbSize + dbSize;
    uint64_t *assigned = static_cast<uint64_t *>(mem_align(2 * 1024 * 1024, assignedWords * sizeof(uint64_t)));
#ifdef MADV_HUGEPAGE
    // random probes of a bitmap larger than the TLB reach
    madvise(assigned, assignedWords * sizeof(uint64_t), MADV_HUGEPAGE);
#endif
    memset(assigned, 0, assignedWords * sizeof(uint64_t));
    std::vector<std::vector<DBKeyType> > threadAssignedKeys(threads);
    const auto isAssigned = [&](DBKeyType key) {
        return ((assigned[key / 64] >> (key % 64)) & 1) != 0;
    };
    const auto setAssigned = [&](DBKeyType key, std::vector<DBKeyType> &keys) {
        if ((__sync_fetch_and_or(&assigned[key / 64], 1ULL << (key % 64)) & (1ULL << (key % 64))) == 0) {
            keys.push_back(key);
        }
    };
    const auto setAssignedSerial = [&](DBKeyType key, std::vector<DBKeyType> &keys) {
        uint64_t &word = assigned[key / 64];
        const uint64_t bit = 1ULL << (key % 64);
        if ((word & bit) == 0) {
            word |= bit;
            keys.push_back(key);
        }
    };

    std::vector<unsigned char> nodeHasProgress(MMseqsNode::count, 0);
    std::vector<size_t> nodeRepOrderKeys(MMseqsNode::count, 0);
    std::vector<size_t> nodeLogSize(MMseqsNode::count, 0);
    std::vector<size_t> nodeReadSize(MMseqsNode::count, 0);
    std::vector<ClusteredLog> nodeClusteredLogs(MMseqsNode::count);
    std::vector<std::vector<uint64_t> > nodeClusteredBits(MMseqsNode::count);
    std::vector<size_t> nodeClusteredWordCounts(MMseqsNode::count, 0);
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        const ClusteredLog clusteredLog = { FileUtil::openAndDelete((clusteredLogPrefix + "." + SSTR(node)).c_str(), "w"), 0, 0, 0, new(std::nothrow) char[CLUSTERED_LOG_BUFFER_SIZE] };
        Util::checkAllocation(clusteredLog.buffer, "Can not allocate clustered log buffer memory in align2clust");
        setvbuf(clusteredLog.file, clusteredLog.buffer, _IOFBF, CLUSTERED_LOG_BUFFER_SIZE);
        nodeClusteredLogs[node] = clusteredLog;
    }
    FILE *assignedKeysHandle = FileUtil::openAndDelete(assignedKeysLog.c_str(), "w");
    size_t assignedKeysSize = 0;
    const size_t stepWords = getStepWords(memoryLimit);
    const size_t nodeStepWords = std::max<size_t>(1, stepWords / MMseqsNode::count);
    const size_t logReadBytesPerNode = std::max<size_t>(1, std::min(stepWords, memoryLimit / LOG_READ_MEMORY_DIVISOR / sizeof(uint64_t)) / MMseqsNode::count) * sizeof(uint64_t);
    std::vector<uint64_t> logWords;
    std::vector<std::vector<uint64_t> > nodeUnclusteredWords(MMseqsNode::count);
    std::vector<size_t> nodeWordOffsets(MMseqsNode::count, 0);
    std::vector<std::vector<size_t> > nodeRepWordOffsets(MMseqsNode::count);
    std::vector<size_t> nodeNextRepKeys(MMseqsNode::count, 0);
    std::vector<size_t> nodeMergedReps(MMseqsNode::count, 0);
    std::vector<MergedReps> mergedReps;
    std::vector<AcceptedRep> reps;
    std::vector<unsigned char> pairIsFree;
    std::vector<size_t> repKeyOffsets;
    std::vector<std::pair<DBKeyType, size_t> > keyReps;
    std::vector<unsigned char> repSharesKey;
    std::vector<size_t> sharingReps;
    std::vector<uint64_t> words;
    std::vector<uint64_t> assignedKeyWords;
    std::vector<uint64_t> writtenAssignedKeys;
    std::thread assignedKeysWriter;
    std::atomic<bool> isWritingAssignedKeys(false);
    std::vector<size_t> threadKeyOffsets(threads + 1, 0);
    const auto writeAssignedKeys = [&](size_t clusteredOrderKey) {
        if (fwrite(writtenAssignedKeys.data(), sizeof(uint64_t), writtenAssignedKeys.size(), assignedKeysHandle) != writtenAssignedKeys.size()) {
            Debug(Debug::ERROR) << "Cannot write " << assignedKeysLog << "\n";
            EXIT(EXIT_FAILURE);
        }
        assignedKeysSize += writtenAssignedKeys.size() * sizeof(uint64_t);
        MMseqsNode::publishLog(assignedKeysHandle, assignedKeysLog, clusteredOrderKey, assignedKeysSize);
        isWritingAssignedKeys = false;
    };
    std::vector<std::vector<size_t> > threadNodeOffsets(threads, std::vector<size_t>(MMseqsNode::count));
    size_t memberCount = 0;
    // --cluster-mode 0: open sets in a heap, sets of two as pairs
    std::vector<ClusterPair> pairSets;
    SetCoverPairFile pairFile(pairSets, clusteredLogPrefix + "_pairsets", stepWords, memoryLimit, assigned);
    SetCoverSmallSetFiles smallSetFiles(clusteredLogPrefix + "_sets", memoryLimit, assigned);
    SetCoverLargeSetFiles largeSetFiles(clusteredLogPrefix + "_large", memoryLimit);
    std::vector<unsigned int> alignmentSplitNodes;
    std::vector<std::vector<DBKeyType> > nodeAlignmentSplitKeys(MMseqsNode::count);
    const auto addClusterPair = [&](DBKeyType repKey, DBKeyType memberKey) {
        const ClusterPair pair = { repKey, memberKey };
        ClusteredLog &nodeFile = nodeClusteredLogs[getKeyServingNodeFromPage(pageServingNodes, seqDbr, repKey)];
        if (fwrite(&pair, sizeof(ClusterPair), 1, nodeFile.file) != 1) {
            Debug(Debug::ERROR) << "Cannot write " << clusteredLogPrefix << "\n";
            EXIT(EXIT_FAILURE);
        }
        nodeFile.bytes += sizeof(ClusterPair);
        memberCount++;
    };
    double waitSeconds = 0;
    Timer clusterTimer;
    unsigned int polls = 0;
    MMseqsNode::WaitWarning waitWarning;
    size_t takenItemCount = 0;
    for (size_t clusteredOrderKey = 0; clusteredOrderKey < orderKeyEnd;) {
        // write and publish the steps decided while the writer is busy
        if (assignedKeyWords.empty() == false && (isWritingAssignedKeys == false || assignedKeyWords.size() >= stepWords)) {
            if (assignedKeysWriter.joinable()) {
                assignedKeysWriter.join();
            }
            assignedKeyWords.swap(writtenAssignedKeys);
            assignedKeyWords.clear();
            isWritingAssignedKeys = true;
            assignedKeysWriter = std::thread(writeAssignedKeys, clusteredOrderKey);
        }
        Timer pollTimer;
        bool hasNewProgress = false;
#pragma omp parallel for schedule(dynamic, 1) reduction(||:hasNewProgress) reduction(max:takenItemCount)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const std::string log = accepted + "." + SSTR(node);
            size_t repKey;
            size_t logSize;
            size_t nodeTakenItemCount;
            if ((nodeHasProgress[node] && nodeRepOrderKeys[node] == orderKeyEnd) || MMseqsNode::readLogProgress(log, repKey, logSize, &nodeTakenItemCount) == false) {
                continue;
            }
            takenItemCount = std::max(takenItemCount, nodeTakenItemCount);
            if (nodeHasProgress[node] && repKey <= nodeRepOrderKeys[node] && logSize <= nodeLogSize[node]) {
                continue;
            }
            hasNewProgress = true;
            nodeHasProgress[node] = 1;
            nodeRepOrderKeys[node] = repKey;
            nodeLogSize[node] = logSize;
        }
        size_t doneRepOrderKey = orderKeyEnd;
        size_t slowestNode = 0;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const size_t nodeRepKey = nodeHasProgress[node] ? nodeRepOrderKeys[node] : 0;
            if (nodeRepKey < doneRepOrderKey) {
                doneRepOrderKey = nodeRepKey;
                slowestNode = node;
            }
        }
        if (doneRepOrderKey <= clusteredOrderKey) {
            if (hasNewProgress == false) {
                MMseqsNode::sleepBeforePoll(polls++);
            }
            waitWarning.check(accepted + "." + SSTR(slowestNode) + ".progress");
            waitSeconds += pollTimer.getTimediff();
            continue;
        }
        polls = 0;
        // a node gets the assigned keys of its alignment splits
        const size_t firstUnseenAlignmentSplit = alignmentSplitNodes.size();
        std::vector<unsigned int> unseenAlignmentSplitNodes(std::max(firstUnseenAlignmentSplit, std::min(takenItemCount, alignmentSplitKeyOffsets.size() - 1)) - firstUnseenAlignmentSplit);
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < unseenAlignmentSplitNodes.size(); i++) {
            unseenAlignmentSplitNodes[i] = alignmentSplitQueue.getNode(firstUnseenAlignmentSplit + i);
        }
        for (size_t i = 0; i < unseenAlignmentSplitNodes.size(); i++) {
            const unsigned int node = unseenAlignmentSplitNodes[i];
            if (node >= MMseqsNode::count) {
                break;
            }
            const size_t keyEnd = alignmentSplitKeyOffsets[alignmentSplitNodes.size() + 1];
            for (size_t key = alignmentSplitKeyOffsets[alignmentSplitNodes.size()]; key < keyEnd; key = (key | 63) + 1) {
                for (uint64_t bits = assigned[key / 64] >> (key % 64); bits != 0 && key + __builtin_ctzll(bits) < keyEnd; bits &= bits - 1) {
                    if (getKeyServingNodeFromPage(pageServingNodes, seqDbr, key + __builtin_ctzll(bits)) != node) {
                        nodeAlignmentSplitKeys[node].push_back(key + __builtin_ctzll(bits));
                    }
                }
            }
            alignmentSplitNodes.push_back(node);
        }

        // read every log up to doneRepOrderKey
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            std::vector<uint64_t> &nodeWords = nodeUnclusteredWords[node];
            std::vector<size_t> &repWordOffsets = nodeRepWordOffsets[node];
            repWordOffsets.clear();
            size_t word = 0;
            if (nodeNextRepKeys[node] < doneRepOrderKey) {
                nodeNextRepKeys[node] = nodeRepOrderKeys[node];
                const std::string log = accepted + "." + SSTR(node);
                size_t from = nodeReadSize[node] + nodeWords.size() * sizeof(uint64_t);
                bool isCut = false;
                while (true) {
                    for (; word < nodeWords.size(); word++) {
                        if ((nodeWords[word] & ACCEPTED_REP_BIT) == 0) {
                            continue;
                        }
                        const size_t repKey = getRepOrderKey(nodeWords[word]);
                        if (repKey >= doneRepOrderKey || (word >= nodeStepWords && repKey > clusteredOrderKey)) {
                            nodeNextRepKeys[node] = repKey;
                            isCut = true;
                            break;
                        }
                        repWordOffsets.push_back(word);
                    }
                    if (isCut || from >= nodeLogSize[node]) {
                        break;
                    }
                    const size_t to = std::min(nodeLogSize[node], from + logReadBytesPerNode);
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
            stepRepOrderKey = std::min(stepRepOrderKey, nodeNextRepKeys[node]);
        }
        size_t stepWordCount = 0;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            std::vector<size_t> &repWordOffsets = nodeRepWordOffsets[node];
            const std::vector<uint64_t> &nodeWords = nodeUnclusteredWords[node];
            const size_t clusteredReps = std::lower_bound(repWordOffsets.begin(), repWordOffsets.end() - 1, stepRepOrderKey,
                [&](size_t word, size_t repKey) { return getRepOrderKey(nodeWords[word]) < repKey; }) - repWordOffsets.begin();
            if (clusteredReps + 1 < repWordOffsets.size()) {
                nodeNextRepKeys[node] = getRepOrderKey(nodeWords[repWordOffsets[clusteredReps]]);
            }
            nodeClusteredWordCounts[node] = repWordOffsets[clusteredReps];
            repWordOffsets.resize(clusteredReps);
            repWordOffsets.push_back(nodeClusteredWordCounts[node]);
            nodeWordOffsets[node] = stepWordCount;
            stepWordCount += nodeClusteredWordCounts[node];
        }
        logWords.resize(stepWordCount);
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            std::vector<uint64_t> &nodeWords = nodeUnclusteredWords[node];
            std::copy(nodeWords.begin(), nodeWords.begin() + nodeClusteredWordCounts[node], logWords.begin() + nodeWordOffsets[node]);
            nodeWords.erase(nodeWords.begin(), nodeWords.begin() + nodeClusteredWordCounts[node]);
            std::vector<size_t> &repWordOffsets = nodeRepWordOffsets[node];
            for (size_t rep = 0; rep < repWordOffsets.size(); rep++) {
                repWordOffsets[rep] += nodeWordOffsets[node];
            }
            nodeReadSize[node] += nodeClusteredWordCounts[node] * sizeof(uint64_t);
            nodeClusteredBits[node].assign((nodeClusteredWordCounts[node] + 63) / 64, 0);
        }
        // merge the reps of the logs in key order
        std::priority_queue<std::pair<size_t, unsigned int>, std::vector<std::pair<size_t, unsigned int> >, std::greater<std::pair<size_t, unsigned int> > > nextReps;
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            nodeMergedReps[node] = 0;
            if (nodeRepWordOffsets[node].size() > 1) {
                nextReps.push(std::make_pair(getRepOrderKey(logWords[nodeRepWordOffsets[node][0]]), node));
            }
        }
        mergedReps.clear();
        size_t repCount = 0;
        while (nextReps.empty() == false) {
            const unsigned int node = nextReps.top().second;
            nextReps.pop();
            const std::vector<size_t> &repWordOffsets = nodeRepWordOffsets[node];
            const size_t endKey = nextReps.empty() ? stepRepOrderKey : nextReps.top().first;
            const size_t endRep = std::lower_bound(repWordOffsets.begin() + nodeMergedReps[node] + 1, repWordOffsets.end() - 1, endKey,
                [&](size_t word, size_t repKey) { return getRepOrderKey(logWords[word]) < repKey; }) - repWordOffsets.begin();
            const MergedReps nodeReps = { nodeMergedReps[node], repCount, node };
            mergedReps.push_back(nodeReps);
            repCount += endRep - nodeMergedReps[node];
            nodeMergedReps[node] = endRep;
            if (endRep + 1 < repWordOffsets.size()) {
                nextReps.push(std::make_pair(getRepOrderKey(logWords[repWordOffsets[endRep]]), node));
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
                const MergedReps &nodeReps = mergedReps[merged];
                const size_t segmentEnd = std::min(end, (merged + 1 < mergedReps.size()) ? mergedReps[merged + 1].firstMergedRep : repCount);
                const std::vector<size_t> &repWordOffsets = nodeRepWordOffsets[nodeReps.node];
                for (; rep < segmentEnd; rep++) {
                    const size_t nodeRep = nodeReps.firstNodeRep + rep - nodeReps.firstMergedRep;
                    const AcceptedRep acceptedRep = { getRepOrderKey(logWords[repWordOffsets[nodeRep]]), repWordOffsets[nodeRep] + 1, repWordOffsets[nodeRep + 1], nodeReps.node };
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
            const bool repIsFree = isAssigned(reps[rep].repKey % paddedDbSize) == false;
            size_t freeCount = 0;
            for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                pairIsFree[i] = repIsFree && isAssigned(logWords[i]) == false;
                freeCount += pairIsFree[i];
            }
            repKeyOffsets[rep + 1] = (freeCount > 0) ? freeCount + 1 : 0;
        }
        sumOffsets(repKeyOffsets.data(), repCount, threads);
        if (mode == Parameters::SET_COVER) {
            // a set waits until no larger set can come
            size_t setPairs = 0;
            for (size_t rep = 0; rep < repCount; rep++) {
                const size_t setSize = repKeyOffsets[rep + 1] - repKeyOffsets[rep];
                setPairs += setSize == 2;
            }
            const bool isBufferingPairs = pairFile.prepareBatch(setPairs);
            const size_t pairOffset = pairSets.size();
            if (isBufferingPairs) {
                pairSets.resize(pairOffset + setPairs);
            }
            size_t pairIndex = pairOffset;
            for (size_t rep = 0; rep < repCount; rep++) {
                if (repKeyOffsets[rep + 1] - repKeyOffsets[rep] == 2) {
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
            for (size_t rep = 0; rep < repCount; rep++) {
                const size_t setSize = repKeyOffsets[rep + 1] - repKeyOffsets[rep];
                if (setSize < 3 || setSize > SetCoverSmallSetFiles::MAX_SIZE) {
                    continue;
                }
                smallMembers[0] = reps[rep].repKey % paddedDbSize;
                size_t member = 1;
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (pairIsFree[i]) {
                        smallMembers[member++] = logWords[i];
                    }
                }
                smallSetFiles.append(setSize, smallMembers);
            }
            largeSetFiles.appendStep(reps, repKeyOffsets, paddedDbSize, [&](size_t i) { return pairIsFree[i]; },
                                     [&](size_t i) { return logWords[i]; });
            // no larger set can come
            const size_t maxSetSize = (stepRepOrderKey < orderKeyEnd) ? static_cast<size_t>(2) << (sizeClassCount - 1 - stepRepOrderKey / paddedDbSize) : 0;
            largeSetFiles.assignSetsLargerThan(maxSetSize, isAssigned, [&](DBKeyType repKey, DBKeyType memberKey) {
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
            // keyReps: the first rep of every key of the step
            size_t tableBits = 1;
            while ((1ULL << tableBits) < 2 * repKeyOffsets[repCount]) {
                tableBits++;
            }
            const size_t tableSize = 1ULL << tableBits;
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
                if (firstRep != std::numeric_limits<size_t>::max()) {
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
                            const size_t bit = i - nodeWordOffsets[reps[rep].node];
                            __sync_fetch_and_or(&nodeClusteredBits[reps[rep].node][bit / 64], 1ULL << (bit % 64));
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
            for (size_t sharingRep = 0; sharingRep < sharingReps.size(); sharingRep++) {
                if (sharingRep + PREFETCH_DISTANCE < sharingReps.size()) {
                    const AcceptedRep &nextRep = reps[sharingReps[sharingRep + PREFETCH_DISTANCE]];
                    __builtin_prefetch(&assigned[nextRep.repKey / 64]);
                    for (size_t i = nextRep.begin; i < std::min(nextRep.end, nextRep.begin + PREFETCH_DISTANCE); i++) {
                        __builtin_prefetch(&assigned[logWords[i] / 64]);
                    }
                }
                const size_t rep = sharingReps[sharingRep];
                const DBKeyType repKey = reps[rep].repKey;
                if (isAssigned(repKey)) {
                    continue;
                }
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (i + PREFETCH_DISTANCE < reps[rep].end) {
                        __builtin_prefetch(&assigned[logWords[i + PREFETCH_DISTANCE] / 64]);
                    }
                    const DBKeyType memberKey = logWords[i];
                    if (pairIsFree[i] && isAssigned(memberKey) == false) {
                        setAssignedSerial(repKey, threadAssignedKeys[0]);
                        setAssignedSerial(memberKey, threadAssignedKeys[0]);
                        const size_t bit = i - nodeWordOffsets[reps[rep].node];
                        nodeClusteredBits[reps[rep].node][bit / 64] |= 1ULL << (bit % 64);
                        memberCount++;
                    }
                }
            }
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                ClusteredLog &nodeFile = nodeClusteredLogs[node];
                words.clear();
                for (size_t i = 0; i < nodeClusteredWordCounts[node]; i += 64) {
                    const uint64_t bits = nodeClusteredBits[node][i / 64];
                    const unsigned int wordBits = std::min<size_t>(64, nodeClusteredWordCounts[node] - i);
                    nodeFile.partialWord |= bits << nodeFile.partialWordBits;
                    if (nodeFile.partialWordBits + wordBits >= 64) {
                        words.push_back(nodeFile.partialWord);
                        nodeFile.partialWord = (nodeFile.partialWordBits > 0) ? bits >> (64 - nodeFile.partialWordBits) : 0;
                        nodeFile.partialWordBits = nodeFile.partialWordBits + wordBits - 64;
                    } else {
                        nodeFile.partialWordBits += wordBits;
                    }
                }
                if (words.empty() == false && fwrite(words.data(), sizeof(uint64_t), words.size(), nodeFile.file) != words.size()) {
                    Debug(Debug::ERROR) << "Cannot write " << clusteredLogPrefix << "." << node << "\n";
                    EXIT(EXIT_FAILURE);
                }
                nodeFile.bytes += words.size() * sizeof(uint64_t);
            }
        }
        clusteredOrderKey = stepRepOrderKey;

        // count the step keys per node in parallel
        for (int thread = 0; thread < threads; thread++) {
            threadKeyOffsets[thread + 1] = threadKeyOffsets[thread] + threadAssignedKeys[thread].size();
        }
        const size_t stepKeyCount = threadKeyOffsets[threads];
        const size_t headerWords = getAssignedKeysHeaderWords();
        const size_t stepStart = assignedKeyWords.size();
        for (int pass = 0; pass < 2; pass++) {
#pragma omp parallel for schedule(static)
            for (int task = 0; task < threads; task++) {
                std::vector<size_t> &nodeOffsets = threadNodeOffsets[task];
                if (pass == 0) {
                    nodeOffsets.assign(MMseqsNode::count, 0);
                }
                const size_t taskEnd = stepKeyCount * (task + 1) / threads;
                size_t thread = std::upper_bound(threadKeyOffsets.begin(), threadKeyOffsets.end(), stepKeyCount * task / threads) - threadKeyOffsets.begin() - 1;
                for (size_t i = stepKeyCount * task / threads; i < taskEnd; i++) {
                    while (i == threadKeyOffsets[thread + 1]) {
                        thread++;
                    }
                    const DBKeyType key = threadAssignedKeys[thread][i - threadKeyOffsets[thread]];
                    const size_t keyNode = getKeyServingNodeFromPage(pageServingNodes, seqDbr, key);
                    const size_t alignmentSplit = std::upper_bound(alignmentSplitKeyOffsets.begin(), alignmentSplitKeyOffsets.end(), key) - alignmentSplitKeyOffsets.begin() - 1;
                    const size_t groupNode = (alignmentSplit < alignmentSplitNodes.size()) ? alignmentSplitNodes[alignmentSplit] : keyNode;
                    const size_t keyPosition = nodeOffsets[keyNode]++;
                    const size_t groupPosition = (groupNode != keyNode) ? nodeOffsets[groupNode]++ : keyPosition;
                    if (pass == 1) {
                        assignedKeyWords[keyPosition] = key;
                        assignedKeyWords[groupPosition] = key;
                    }
                }
            }
            if (pass == 0) {
                assignedKeyWords.resize(stepStart + headerWords);
                assignedKeyWords[stepStart] = clusteredOrderKey;
                size_t nodeKeyOffset = 0;
                for (size_t node = 0; node < MMseqsNode::count; node++) {
                    assignedKeyWords[stepStart + 1 + node] = nodeReadSize[node];
                    assignedKeyWords[stepStart + 1 + MMseqsNode::count + node] = nodeKeyOffset;
                    for (int task = 0; task < threads; task++) {
                        const size_t count = threadNodeOffsets[task][node];
                        threadNodeOffsets[task][node] = stepStart + headerWords + nodeKeyOffset;
                        nodeKeyOffset += count;
                    }
                    nodeKeyOffset += nodeAlignmentSplitKeys[node].size();
                }
                assignedKeyWords[stepStart + 1 + 2 * MMseqsNode::count] = nodeKeyOffset;
                assignedKeyWords.resize(stepStart + headerWords + nodeKeyOffset);
                for (size_t node = 0; node < MMseqsNode::count; node++) {
                    std::copy(nodeAlignmentSplitKeys[node].begin(), nodeAlignmentSplitKeys[node].end(), assignedKeyWords.begin() + stepStart + headerWords + assignedKeyWords[stepStart + 2 + MMseqsNode::count + node] - nodeAlignmentSplitKeys[node].size());
                    nodeAlignmentSplitKeys[node].clear();
                }
            }
        }
        for (int thread = 0; thread < threads; thread++) {
            threadAssignedKeys[thread].clear();
        }
        waitWarning.reset();
    }
    if (assignedKeysWriter.joinable()) {
        assignedKeysWriter.join();
    }
    if (assignedKeyWords.empty() == false) {
        assignedKeyWords.swap(writtenAssignedKeys);
        writeAssignedKeys(orderKeyEnd);
    }
    MMseqsNode::addPhaseSeconds(phaseValues, "clustering the accepted pairs", clusterTimer.getTimediff() - waitSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the aligning nodes", waitSeconds);
    if (fclose(assignedKeysHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << assignedKeysLog << "\n";
        EXIT(EXIT_FAILURE);
    }
    // an assigned key is no singleton
    if (assignedBits.empty() == false) {
        FILE *assignedHandle = FileUtil::openAndDelete(assignedBits.c_str(), "w");
        if (fwrite(assigned, sizeof(uint64_t), assignedWords, assignedHandle) != assignedWords) {
            Debug(Debug::ERROR) << "Cannot write " << assignedBits << "\n";
            EXIT(EXIT_FAILURE);
        }
        MMseqsNode::publishLog(assignedHandle, assignedBits, dbSize, assignedWords * sizeof(uint64_t));
        if (fclose(assignedHandle) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << assignedBits << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    free(assigned);
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        const std::string clusteredLogFile = clusteredLogPrefix + "." + SSTR(node);
        if (nodeClusteredLogs[node].partialWordBits > 0 && fwrite(&nodeClusteredLogs[node].partialWord, sizeof(uint64_t), 1, nodeClusteredLogs[node].file) != 1) {
            Debug(Debug::ERROR) << "Cannot write " << clusteredLogFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        MMseqsNode::publishLog(nodeClusteredLogs[node].file, clusteredLogFile, dbSize, nodeClusteredLogs[node].bytes + ((nodeClusteredLogs[node].partialWordBits > 0) ? sizeof(uint64_t) : 0));
        if (fclose(nodeClusteredLogs[node].file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << clusteredLogFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        delete[] nodeClusteredLogs[node].buffer;
    }
    return memberCount;
}

static uint64_t *readSingletonBits(FILE *assignedHandle, const std::string &assignedBits, size_t firstKey, size_t keyCount) {
    size_t words = (keyCount + 63) / 64;
    std::vector<uint64_t> assignedWords(words + 1, 0);
    MMseqsNode::readKeyBits(fileno(assignedHandle), assignedBits, 0, 0, firstKey, firstKey + keyCount, firstKey, assignedWords.data());
    uint64_t *bits = new(std::nothrow) uint64_t[words];
    Util::checkAllocation(bits, "Can not allocate singletonBits memory in align2clust");
    for (size_t word = 0; word < words; word++) {
        bits[word] = ~assignedWords[word];
    }
    return bits;
}

int align2clustLinfinity(Parameters &par) {
    Timer timer;
    std::string phaseValues;
    Timer phaseValueTimer;
    std::string seqDb = par.db1;
    std::vector<std::string> localCopies;
    size_t localCopySize = 0;
    bool hasLocalServedBlocks = false;
    // a master that only clusters reads no sequences
    if (par.localDbDir.empty() == false && MMseqsNode::rank >= getFirstAlignNode()) {
        seqDb = FileUtil::copyToLocalDir(par.db1, par.localDbDir + "/" + FileUtil::baseName(par.db3) + "." + SSTR(MMseqsNode::rank), par.threads, localCopies, localCopySize);
        MMseqsNode::addPhaseTime(phaseValues, "copying the sequences to --local-db-dir", phaseValueTimer);
    } else if (par.localTmp.empty() == false && MMseqsNode::rank >= getFirstAlignNode()) {
        // ranks on one machine may create it at once
        if (FileUtil::makeDir(par.localTmp.c_str()) == false && FileUtil::directoryExists(par.localTmp.c_str()) == false) {
            Debug(Debug::ERROR) << "Cannot create local tmp dir " << par.localTmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        const int localTmpFd = open(par.localTmp.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        const StorageKind localTmpStorage = getStorageKind(localTmpFd);
        close(localTmpFd);
        if (localTmpStorage != STORAGE_LOCAL_FLASH) {
            Debug(Debug::INFO) << "Not copying the served blocks to " << par.localTmp << ": it is on " << getStorageName(localTmpStorage) << ", not local flash\n";
        } else {
            // copy only the served blocks
            DBReader<DBKeyType> blockDbr(par.db1.c_str(), par.db1Index.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
            blockDbr.open(DBReader<DBKeyType>::NOSORT);
            blockDbr.setBlockCacheSize(0);
            const std::vector<size_t> &blockKeyOffsets = blockDbr.getBlockKeyOffsets();
            const DBReader<DBKeyType>::Index *blockIndex = blockDbr.getIndex();
            const std::vector<std::string> dataFiles = FileUtil::findDatafiles(par.db1.c_str());
            std::vector<std::vector<std::pair<size_t, size_t> > > servedFileRanges(dataFiles.size());
            size_t servedSize = 0;
            for (size_t block = 0; block + 1 < blockKeyOffsets.size(); block++) {
                if (getBlockServingNode(block) != MMseqsNode::rank) {
                    continue;
                }
                const size_t firstRange = blockDbr.findIdRange(blockKeyOffsets[block]);
                const size_t lastRange = blockDbr.findIdRange(blockKeyOffsets[block + 1] - 1);
                std::vector<std::pair<size_t, size_t> > &ranges = servedFileRanges[blockDbr.getRangeDataFile(firstRange)];
                const size_t begin = blockIndex[firstRange].offset + (blockKeyOffsets[block] - blockIndex[firstRange].id) * blockIndex[firstRange].length;
                const size_t end = blockIndex[lastRange].offset + (blockKeyOffsets[block + 1] - blockIndex[lastRange].id) * blockIndex[lastRange].length;
                if (ranges.empty() == false && ranges.back().second == begin) {
                    ranges.back().second = end;
                } else {
                    ranges.emplace_back(begin, end);
                }
                servedSize += end - begin;
            }
            blockDbr.close();
            struct statvfs fs;
            const size_t freeSize = (statvfs(par.localTmp.c_str(), &fs) == 0) ? static_cast<size_t>(fs.f_bavail) * fs.f_frsize : 0;
            if (freeSize < servedSize) {
                Debug(Debug::INFO) << "Not copying the served blocks to " << par.localTmp << ": it has " << freeSize << " free bytes, the served blocks need " << servedSize << "\n";
            } else {
                seqDb = FileUtil::copyToLocalDir(par.db1, par.localTmp + "/" + FileUtil::baseName(par.db3) + "." + SSTR(MMseqsNode::rank), par.threads, localCopies, localCopySize, &servedFileRanges);
                hasLocalServedBlocks = true;
                MMseqsNode::addPhaseTime(phaseValues, "copying the served blocks to --local-tmp", phaseValueTimer);
            }
        }
    }
    DBReader<DBKeyType> seqDbr(seqDb.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    seqDbr.open(DBReader<DBKeyType>::NOSORT);
    const std::vector<size_t> nodeKeyOffsets = seqDbr.getEvenNodeKeyOffsets(MMseqsNode::count);
    const size_t dbSize = seqDbr.getSize();
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    // a copy on tmpfs holds the sequences in memory
    const size_t nodeMemoryLimit = std::max(Util::computeMemory(par.splitMemoryLimit), localCopySize) - localCopySize;
    const bool isMasterAligning = getFirstAlignNode() == 0;
    const size_t masterMemoryLimit = nodeMemoryLimit / (isMasterAligning ? MASTER_MEMORY_DIVISOR : 1);
    const size_t masterMemorySize = ((par.clusteringMode == Parameters::SET_COVER) ? masterMemoryLimit : (dbSize + 63) / 64 * sizeof(uint64_t) + masterMemoryLimit / STEP_MEMORY_DIVISOR)
                                + MMseqsNode::count * CLUSTERED_LOG_BUFFER_SIZE;

    // cut the groups into alignment splits of about equal cells
    const std::string partitionsData = MMseqsNode::readFile(par.db2 + ".partitions");
    std::vector<PrefPartition> partitions;
    uint128 totalAlignmentCells = 0;
    size_t totalPrefSize = 0;
    for (char *line = (char *) partitionsData.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[5];
        Util::getWordsOfLine(line, columns, 5);
        PrefPartition partition;
        partition.firstKey = Util::fast_atoi<size_t>(columns[0]);
        partition.prefOffset = Util::fast_atoi<size_t>(columns[1]);
        partition.prefSize = Util::fast_atoi<size_t>(columns[2]);
        partition.alignmentCells = Util::fast_atoi<size_t>(columns[3]);
        partition.node = Util::fast_atoi<size_t>(columns[4]);
        partitions.push_back(partition);
        totalAlignmentCells += partition.alignmentCells;
        totalPrefSize += partition.prefSize;
    }
    const size_t alignNodeCount = MMseqsNode::count - getFirstAlignNode();
    const size_t maxAlignmentSplitCount = std::max<size_t>(1, std::min(ALIGNMENT_SPLITS_PER_NODE * alignNodeCount, totalPrefSize / MIN_ALIGNMENT_SPLIT_PREF_SIZE));
    std::vector<size_t> alignmentSplitPartitionOffsets(1, 0);
    std::vector<size_t> alignmentSplitKeyOffsets(1, 0);
    uint128 cells = 0;
    for (size_t i = 0; i + 1 < partitions.size(); i++) {
        cells += partitions[i].alignmentCells;
        if (alignmentSplitPartitionOffsets.size() < maxAlignmentSplitCount && cells * maxAlignmentSplitCount >= totalAlignmentCells * alignmentSplitPartitionOffsets.size()) {
            alignmentSplitPartitionOffsets.push_back(i + 1);
            alignmentSplitKeyOffsets.push_back(partitions[i + 1].firstKey);
        }
    }
    alignmentSplitPartitionOffsets.push_back(partitions.size());
    alignmentSplitKeyOffsets.push_back(dbSize);
    MMseqsNode::WorkQueue alignmentSplitQueue(par.db3 + "_alignment_splits", alignmentSplitPartitionOffsets.size() - 1);

    // builds the block table of getKeyBlock
    seqDbr.setBlockCacheSize(0);
    const std::vector<size_t> &blockKeyOffsets = seqDbr.getBlockKeyOffsets();
    // the master keeps the serving node of every page of assigned bits
    const size_t pageCount = (dbSize + ASSIGNED_PAGE_KEYS - 1) / ASSIGNED_PAGE_KEYS;
    std::vector<unsigned int> pageServingNodes(MMseqsNode::isMaster() ? pageCount : 0);
    for (size_t block = 0, pageEnd = 0; block + 1 < blockKeyOffsets.size(); block++) {
        const unsigned int node = getBlockServingNode(block);
        const size_t lastPage = (blockKeyOffsets[block + 1] - 1) / ASSIGNED_PAGE_KEYS;
        for (size_t page = blockKeyOffsets[block] / ASSIGNED_PAGE_KEYS; MMseqsNode::isMaster() && page <= lastPage; page++) {
            // a page with keys of two nodes uses getKeyServingNode
            pageServingNodes[page] = (page < pageEnd && pageServingNodes[page] != node) ? UINT_MAX : node;
        }
        pageEnd = lastPage + 1;
    }
    // about twice the share of pages of a node
    const size_t assignedPageCount = (MMseqsNode::rank >= getFirstAlignNode()) ? std::min(pageCount, 2 * ((pageCount + alignNodeCount - 1) / alignNodeCount)) : 0;
    const size_t assignedMemorySize = assignedPageCount * (ASSIGNED_PAGE_KEYS / 8) + ((assignedPageCount > 0) ? pageCount * sizeof(uint64_t *) : 0) + pageServingNodes.size() * sizeof(unsigned int);
    const size_t memoryLimit = nodeMemoryLimit - std::min(nodeMemoryLimit, (isMasterAligning ? std::min(masterMemoryLimit, masterMemorySize) : 0) + assignedMemorySize);
    const int clusterThreads = isMasterAligning ? std::min(par.threads, CLUSTER_THREADS) : par.threads;

    // a clusthashfast subset has bits; its sixth index column counts the representatives for the e-values
    const bool hasRepresentativeBits = FileUtil::fileExists((par.db1 + ".bits." + SSTR(MMseqsNode::rank)).c_str());
    size_t representativeCount = 0;
    size_t representativeResidues = 0;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        representativeCount += seqDbr.getRangeRepresentativeCount(range);
        representativeResidues += seqDbr.getRangeRepresentativeCount(range) * (index[range].length - 2);
    }

    const std::string accepted = par.db3 + "_accepted";
    const std::string assignedKeysLog = par.db3 + "_assigned_keys";
    const std::string clusteredLogPrefix = par.db3 + "_clustered";
    // only the singletons of a database without bits need the assigned bits
    const std::string assignedBits = hasRepresentativeBits ? "" : par.db3 + "_assigned_keys.bits";
    // create the exchange indexes before the memory barrier
    const std::string requests = par.db3 + "_sequence_requests";
    const std::string served = par.db3 + "_served_sequences";
    std::vector<std::string> sequenceIndexes;
    std::vector<std::string> sequenceFiles;
    if (MMseqsNode::rank >= getFirstAlignNode() && alignNodeCount > 1) {
        sequenceIndexes.push_back(requests + ".index." + SSTR(MMseqsNode::rank));
        sequenceIndexes.push_back(served + ".index." + SSTR(MMseqsNode::rank));
        sequenceFiles.push_back(requests + "." + SSTR(MMseqsNode::rank));
    }
    for (size_t i = 0; i < sequenceIndexes.size(); i++) {
        if (fclose(FileUtil::openAndDelete(sequenceIndexes[i].c_str(), "w")) != 0) {
            Debug(Debug::ERROR) << "Cannot create " << sequenceIndexes[i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }

    // remove the item files a killed run left
    alignmentSplitQueue.removeTakenItemFiles();
    // all nodes use the smallest memory limit
    const size_t smallestMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db3 + ".memory", nodeMemoryLimit - std::min(nodeMemoryLimit, assignedMemorySize));
    const size_t paddedDbSize = (dbSize + 63) / 64 * 64;
    const size_t sizeClassCount = (par.clusteringMode == Parameters::SET_COVER) ? 64 - __builtin_clzll(dbSize | 1) : 1;
    const size_t maxRecordSize = std::max<size_t>(1, smallestMemoryLimit / RECORD_MEMORY_DIVISOR);
    // all nodes cut served files at the same size
    const size_t servedFileEntrySize = std::max(BatchReader::MAX_READ_REQUEST_SIZE, smallestMemoryLimit / SERVED_FILE_MEMORY_DIVISOR);

    RepTargetsReader repTargetsReader(par, seqDbr, partitions, alignmentSplitPartitionOffsets, alignmentSplitQueue, maxRecordSize, memoryLimit, paddedDbSize, sizeClassCount);
    if (sizeClassCount > 1) {
        repTargetsReader.countSizeClasses();
    }
    MMseqsNode::addPhaseTime(phaseValues, "counting the target size classes", phaseValueTimer);

    ClusterMemberLookup *memberLookup = NULL;
    DBReader<DBKeyType> *memberDbr = NULL;
    if (par.filterCluDBFile.empty() == false) {
        // share the sampled keys of the cluster node files
        const size_t entriesPerSampledKey = std::max(MIN_ENTRIES_PER_SAMPLED_KEY, dbSize / std::max<size_t>(1, smallestMemoryLimit / SAMPLED_KEY_MEMORY_DIVISOR / sizeof(DBKeyType)));
        DBReader<DBKeyType> cluDb((par.filterCluDBFile + "." + SSTR(MMseqsNode::rank)).c_str(), (par.filterCluDBFile + ".index." + SSTR(MMseqsNode::rank)).c_str(), par.threads,
                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDb.open(DBReader<DBKeyType>::NOSORT);
        std::vector<DBKeyType> sampledKeys((cluDb.getSize() + entriesPerSampledKey - 1) / entriesPerSampledKey);
#pragma omp parallel for schedule(static)
        for (size_t sampledKey = 0; sampledKey < sampledKeys.size(); sampledKey++) {
            sampledKeys[sampledKey] = cluDb.getIndexEntry(sampledKey * entriesPerSampledKey).id;
        }
        cluDb.close();
        const std::string sampledKeyData = MMseqsNode::barrier(par.db3 + ".sampled_keys", std::string(reinterpret_cast<const char *>(sampledKeys.data()), sampledKeys.size() * sizeof(DBKeyType)));
        MMseqsNode::removeOnceRead(par.db3 + ".sampled_keys");
        if (MMseqsNode::rank >= getFirstAlignNode()) {
            memberLookup = new ClusterMemberLookup(par.filterCluDBFile, sampledKeyData, entriesPerSampledKey);
        }
        memberDbr = new DBReader<DBKeyType>(par.filterSeqDBFile.c_str(), (par.filterSeqDBFile + ".index").c_str(), par.threads,
                                            DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        memberDbr->open(DBReader<DBKeyType>::NOSORT);
        MMseqsNode::addPhaseTime(phaseValues, "sampling the cluster index keys", phaseValueTimer);
    }

    // the master clusters while the nodes align
    size_t memberCount = 0;
    std::string clusterPhaseValues;
    std::thread clusterThread;
    if (MMseqsNode::isMaster()) {
        clusterThread = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(clusterThreads);
#endif
            memberCount = clusterAcceptedPairs(clusterThreads, dbSize, masterMemoryLimit, alignmentSplitKeyOffsets, alignmentSplitQueue, seqDbr, pageServingNodes, par.clusteringMode, paddedDbSize, sizeClassCount,
                                               accepted, assignedKeysLog, clusteredLogPrefix, assignedBits, clusterPhaseValues);
        });
    }
    phaseValueTimer.reset();
    alignRepTargets(par, seqDbr, repTargetsReader, memoryLimit, masterMemoryLimit, servedFileEntrySize, representativeResidues, paddedDbSize, sizeClassCount,
                    memberLookup, memberDbr, hasLocalServedBlocks, phaseValues);
    delete memberLookup;
    repTargetsReader.close();
    if (MMseqsNode::rank >= getFirstAlignNode()) {
        MMseqsNode::addPhaseTime(phaseValues, "aligning the rep targets", phaseValueTimer);
        if (par.align2clustAlignerMode != Parameters::ALIGNER_MODE_FARRAR) {
            MMseqsNode::addPhaseValue(phaseValues, "SWIPE member alignments", SSTR(swipeLanePairCount.load()) + " in lanes ("
                                     + SSTR(swipeLanePairCount.load() * 100 / std::max<size_t>(1, swipeLaneCapacity.load()))
                                     + "% active lane rows), " + SSTR(swipeScalarPairCount.load()) + " scalar");
            if (swipeMemberExtensionCount.load() > 0) {
                MMseqsNode::addPhaseValue(phaseValues, "SWIPE member extensions", SSTR(swipeMemberExtensionCount.load()) + " in lanes ("
                                         + SSTR(swipeMemberActiveLaneRowCount.load() * 100 / std::max<size_t>(1, swipeMemberTotalLaneRowCount.load())) + "% active lane rows)");
            }
            MMseqsNode::addPhaseValue(phaseValues, "SWIPE target alignments", SSTR(swipeTargetPairCount.load()) + " in lanes ("
                                     + SSTR(swipeTargetActiveLaneRowCount.load() * 100 / std::max<size_t>(1, swipeTargetTotalLaneRowCount.load()))
                                     + "% active lane rows), " + SSTR(swipeLongTargetPairCount.load()) + " long extensions");
            MMseqsNode::addPhaseValue(phaseValues, "SWIPE extensions in 8 bit lanes", SSTR(swipeByteTotalLaneRowCount.load() * 100 / std::max<size_t>(1, swipeMemberTotalLaneRowCount.load() + swipeTargetTotalLaneRowCount.load()))
                                     + "% of the lane rows (" + SSTR(swipeByteActiveLaneRowCount.load() * 100 / std::max<size_t>(1, swipeByteTotalLaneRowCount.load())) + "% active), "
                                     + SSTR(swipeRerunPairCount.load()) + " again in 16 bit lanes");
        }
    }
    MMseqsNode::addPhaseValue(phaseValues, "Memory limit", SSTR(memoryLimit));

    const std::string clusteredLogFile = clusteredLogPrefix + "." + SSTR(MMseqsNode::rank);
    size_t clusteredKey;
    size_t clusteredSize;
    MMseqsNode::waitForLog(clusteredLogFile, clusteredKey, clusteredSize);
    // every node took its last alignment split
    alignmentSplitQueue.removeTakenItemFiles();
    if (MMseqsNode::isMaster()) {
        clusterThread.join();
        Debug(Debug::INFO) << "Number of clusters: " << representativeCount - memberCount << "\n";
        phaseValues += clusterPhaseValues;
        std::vector<unsigned int>().swap(pageServingNodes);
    }
    // the cluster thread was the last user of seqDbr
    seqDbr.close();
    for (size_t i = 0; i < localCopies.size(); i++) {
        FileUtil::remove(localCopies[i].c_str());
    }
    if (localCopies.empty() == false) {
        rmdir(FileUtil::dirName(seqDb).c_str());
    }
    MMseqsNode::addPhaseTime(phaseValues, "waiting for the clustering", phaseValueTimer);
    const size_t nodeFirstKey = nodeKeyOffsets[MMseqsNode::rank];
    const size_t nodeKeys = nodeKeyOffsets[MMseqsNode::rank + 1] - nodeFirstKey;
    const std::string clusterPairs = par.db3 + "_cluster_pairs";
    std::vector<size_t> balancedKeyOffsets;
    MMseqsNode::ClusterPairWriter clusterPairWriter(clusterPairs, std::max<size_t>(64, memoryLimit / CLUSTER_PAIRS_MEMORY_DIVISOR / sizeof(ClusterPair)), ClusterPair::compareByRep, nodeKeyOffsets,
                                                    &balancedKeyOffsets);
    const std::string acceptedFile = accepted + "." + SSTR(MMseqsNode::rank);
    if (par.clusteringMode == Parameters::SET_COVER) {
        for (size_t from = 0; from < clusteredSize;) {
            const size_t to = std::min(clusteredSize, from + clusterPairWriter.pairCapacity * sizeof(ClusterPair));
            MMseqsNode::readLog(clusteredLogFile, from, to, reinterpret_cast<char *>(clusterPairWriter.reservePairs((to - from) / sizeof(ClusterPair))));
            from = to;
        }
    } else {
        std::vector<uint64_t> clusteredBits(clusteredSize / sizeof(uint64_t));
        MMseqsNode::readLog(clusteredLogFile, 0, clusteredBits.size() * sizeof(uint64_t), reinterpret_cast<char *>(clusteredBits.data()));
        FILE *acceptedHandle = FileUtil::openFileOrDie(acceptedFile.c_str(), "r", true);
        // a read takes whole words of clustered bits
        const size_t readWordCount = std::min(clusterPairWriter.pairCapacity, std::max<size_t>(1, clusteredBits.size()) * 64) / 64 * 64;
        std::vector<uint64_t> acceptedWords(readWordCount);
        std::vector<std::vector<size_t> > threadRepWordPositions(par.threads);
        std::vector<size_t> repWordPositions;
        uint64_t previousRepWord = 0;
        for (size_t wordCount, firstWord = 0; (wordCount = fread(acceptedWords.data(), sizeof(uint64_t), readWordCount, acceptedHandle)) > 0; firstWord += readWordCount / 64) {
            const size_t clusteredWordCount = (wordCount + 63) / 64;
            const uint64_t *clusteredWords = clusteredBits.data() + firstWord;
            size_t clusteredCount = 0;
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                threadRepWordPositions[thread_idx].clear();
#pragma omp for schedule(static) reduction(+:clusteredCount)
                for (size_t word = 0; word < clusteredWordCount; word++) {
                    clusteredCount += __builtin_popcountll(clusteredWords[word]);
                    for (size_t i = word * 64; i < std::min(wordCount, word * 64 + 64); i++) {
                        if (acceptedWords[i] & ACCEPTED_REP_BIT) {
                            threadRepWordPositions[thread_idx].push_back(i);
                        }
                    }
                }
            }
            repWordPositions.clear();
            for (int thread = 0; thread < par.threads; thread++) {
                repWordPositions.insert(repWordPositions.end(), threadRepWordPositions[thread].begin(), threadRepWordPositions[thread].end());
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
                size_t rep = std::upper_bound(repWordPositions.begin(), repWordPositions.end(), word * 64) - repWordPositions.begin();
                for (; bits != 0; bits &= bits - 1) {
                    const size_t i = word * 64 + __builtin_ctzll(bits);
                    while (rep < repWordPositions.size() && repWordPositions[rep] < i) {
                        rep++;
                    }
                    const uint64_t repWord = (rep > 0) ? acceptedWords[repWordPositions[rep - 1]] : previousRepWord;
                    const ClusterPair clusteredPair = { static_cast<DBKeyType>((repWord & ~ACCEPTED_REP_BIT) % paddedDbSize), static_cast<DBKeyType>(acceptedWords[i]) };
                    clusteredPairs[pairOffset++] = clusteredPair;
                }
            }
            if (repWordPositions.empty() == false) {
                previousRepWord = acceptedWords[repWordPositions.back()];
            }
        }
        if (fclose(acceptedHandle) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << acceptedFile << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    MMseqsNode::removeLog(clusteredLogFile);
    MMseqsNode::removeLog(acceptedFile);
    // write the clusters and, without bits, the singletons
    uint64_t *singletonBits = NULL;
    FILE *assignedHandle = NULL;
    if (assignedBits.empty() == false) {
        size_t assignedKey;
        size_t assignedSize;
        MMseqsNode::waitForLog(assignedBits, assignedKey, assignedSize);
        assignedHandle = FileUtil::openFileOrDie(assignedBits.c_str(), "r", true);
        singletonBits = readSingletonBits(assignedHandle, assignedBits, nodeFirstKey, nodeKeys);
    }
    const std::string clusterPairsIndex = clusterPairWriter.close(singletonBits, nodeFirstKey, nodeKeys);
    // every node has read the sequences that this node requested and served
    for (size_t i = 0; i < sequenceIndexes.size(); i++) {
        FileUtil::remove(sequenceIndexes[i].c_str());
    }
    for (size_t i = 0; i < sequenceFiles.size(); i++) {
        if (FileUtil::fileExists(sequenceFiles[i].c_str())) {
            FileUtil::remove(sequenceFiles[i].c_str());
        }
    }
    MMseqsNode::addPhaseTime(phaseValues, "sending the cluster pairs", phaseValueTimer);
    if (memberDbr != NULL) {
        MMseqsNode::addPhaseValue(phaseValues, "Member block read bytes", SSTR(memberDbr->getBlockReadSize()));
        MMseqsNode::addPhaseValue(phaseValues, "Member key read bytes", SSTR(memberDbr->getKeyReadSize()));
        memberDbr->close();
        delete memberDbr;
    }

    // write the representatives of the balanced keys and their singletons
    const size_t repFirstKey = balancedKeyOffsets[MMseqsNode::rank];
    const size_t repKeyCount = balancedKeyOffsets[MMseqsNode::rank + 1] - repFirstKey;
    if (assignedHandle != NULL) {
        delete[] singletonBits;
        singletonBits = readSingletonBits(assignedHandle, assignedBits, repFirstKey, repKeyCount);
        if (fclose(assignedHandle) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << assignedBits << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    MMseqsNode::writeClusterPairs(clusterPairsIndex, clusterPairs, par.db3, singletonBits, repFirstKey, repKeyCount);
    delete[] singletonBits;
    MMseqsNode::addPhaseTime(phaseValues, "writing the clusters", phaseValueTimer);
    MMseqsNode::removeExchanges(par.db3, clusterPairs, std::vector<std::string>(1, clusterPairs), phaseValues);
    if (MMseqsNode::isMaster()) {
        // every node is done with both logs
        MMseqsNode::removeLog(assignedKeysLog);
        FileUtil::remove((par.db3 + ".memory").c_str());
        if (assignedBits.empty() == false) {
            MMseqsNode::removeLog(assignedBits);
        }
        DBWriter::writeDbtypeFile(par.db3.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_CLUSTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
    }
    Debug(Debug::INFO) << "Time for run Align2Clust: " << timer.lap() << " sec\n";
    return EXIT_SUCCESS;
}
