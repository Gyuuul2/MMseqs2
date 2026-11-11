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
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <sys/resource.h>
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
    SetCoverLargeSetFiles(const std::string &prefix, size_t memoryLimit)
        : prefix(prefix), memoryBudget(std::max<size_t>(65 * sizeof(DBKeyType), getSetCoverReadSize(memoryLimit))),
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
        const size_t run = createSortedRun(file);
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
    size_t prefOffset;
    size_t prefSize;
    size_t alignmentCells;
    size_t node;
    size_t targetSplit;
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

// stolen share per window behind
static const double STEAL_SHARE_STEP = 1.0 / 8;

// target split mode
static std::vector<size_t> targetSplitBlockOffsets;
static std::vector<unsigned int> targetSplitAlignNodes;

// blocks in turn or by target split
static size_t getBlockAlignNode(size_t block) {
    if (targetSplitAlignNodes.empty()) {
        return MMseqsNode::getFirstAlignNode() + block % (MMseqsNode::count - MMseqsNode::getFirstAlignNode());
    }
    return targetSplitAlignNodes[std::upper_bound(targetSplitBlockOffsets.begin(), targetSplitBlockOffsets.end(), block) - targetSplitBlockOffsets.begin() - 1];
}

// a block cursor searches again only once key leaves its block
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

// getSeqLenFromRange only moves its range forward, a smaller key searches again
static size_t getSeqLenFromCursor(DBReader<DBKeyType> &seqDbr, size_t &range, size_t key) {
    if (range != SIZE_MAX && key < seqDbr.getIndex()[range].id) {
        range = SIZE_MAX;
    }
    return seqDbr.getSeqLenFromRange(range, key);
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
    FetchedSequences() : prefilterBuffer(0), begin(0), end(0), targetCount(0), pendingPairCount(0), firstPendingRepKey(0), isLast(false), entries(NULL), allocatedSize(0) {}
    size_t prefilterBuffer;
    size_t begin;
    size_t end;
    size_t targetCount;
    // pending pairs whose members it brings
    size_t pendingPairCount;
    // the first rep of the pairs that wait for their member checks when it was selected
    size_t firstPendingRepKey;
    bool isLast;
    std::vector<DBKeyType> keys;
    std::vector<size_t> fetchSourceKeyOffsets;
    char *entries;
    size_t allocatedSize;
    std::vector<size_t> entryOffsets;
};

// per thread, where the last keys were found per fetch source and the block of the last key
struct FetchedKeyCursor {
    FetchedKeyCursor(size_t fetchSources) : positions(fetchSources, 0), block(SIZE_MAX) {}
    std::vector<size_t> positions;
    size_t block;
};

// from a position behind key the search goes on in doubling steps, so ascending keys take few steps
static const char *findFetchedKey(const FetchedSequences &sequences, size_t fetchSource, DBKeyType key, size_t *seqLen = NULL, size_t *position = NULL) {
    const DBKeyType *keys = sequences.keys.data();
    const size_t sourceEnd = sequences.fetchSourceKeyOffsets[fetchSource + 1];
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
    const size_t entry = found - keys;
    if (seqLen != NULL) {
        *seqLen = sequences.entryOffsets[entry + 1] - sequences.entryOffsets[entry] - 2;
    }
    return sequences.entries + sequences.entryOffsets[entry];
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
    return 2 * MMseqsNode::count + 5;
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
// progress file reads of the sequence service per second
static const size_t PROGRESS_READS_PER_SECOND = 100;

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

// the parts of a pair follow each other
static size_t getPairPartEnd(const std::vector<unsigned char> &isLastPart, size_t pairCount, size_t part) {
    size_t partEnd = part + 1;
    while (partEnd < pairCount && isLastPart[partEnd - 1] == 0) {
        partEnd++;
    }
    return partEnd;
}

static size_t getStepWords(size_t masterMemoryLimit) {
    return std::max<size_t>(1, masterMemoryLimit / STEP_PAIRS_MEMORY_DIVISOR / (sizeof(uint64_t) + 4 * sizeof(std::pair<DBKeyType, size_t>)));
}

// counted per reader and added when it is destroyed, like the SWIPE counters
static std::atomic<size_t> memberLookupCount(0);
static std::atomic<size_t> scannedMemberCount(0);
static std::atomic<size_t> memberLookupReadSize(0);
static std::atomic<size_t> memberIndexReadSize(0);
static std::atomic<size_t> memberLookupReadCount(0);
static std::atomic<size_t> memberIndexReadCount(0);

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
        size_t lookupCount;
        size_t memberCount;
        size_t readBytes;
        size_t indexReadBytes;
        size_t readCount;
        size_t indexReadCount;
        // buffer or part data
        const char *data;
        std::vector<char> buffer;
        // index pairs from indexFirst
        size_t indexFirst;
        std::vector<uint64_t> indexPairs;

        EntryReader() : lookupCount(0), memberCount(0), readBytes(0), indexReadBytes(0), readCount(0), indexReadCount(0), indexFirst(0) {}

        ~EntryReader() {
            memberLookupCount += lookupCount;
            scannedMemberCount += memberCount;
            memberLookupReadSize += readBytes;
            memberIndexReadSize += indexReadBytes;
            memberLookupReadCount += readCount;
            memberIndexReadCount += indexReadCount;
        }

        void start(const Entry &first) {
            entry = first;
            bufferBegin = 0;
            bufferEnd = 0;
            lookupCount++;
        }
    };

    // a node answers only for its own cluster part
    ClusterMemberLookup(const std::string &clu, const std::string &localClu, const std::vector<size_t> &cluKeyOffsets, size_t entriesPerSampledKey, size_t partMemorySize)
        : entriesPerSampledKey(entriesPerSampledKey), cluKeyOffsets(cluKeyOffsets) {
        cluDbr = new DBReader<DBKeyType>((localClu + "." + SSTR(MMseqsNode::rank)).c_str(), (clu + ".index." + SSTR(MMseqsNode::rank)).c_str(), 1,
                                         DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDbr->open(DBReader<DBKeyType>::NOSORT);
        sampledKeys.resize((cluDbr->getSize() + entriesPerSampledKey - 1) / entriesPerSampledKey);
        // read the whole part if it fits
        size_t partIndexSize = (cluDbr->getSize() + 1) * 2 * sizeof(uint64_t);
        if (partIndexSize <= partMemorySize) {
            partIndexPairs.resize(partIndexSize / sizeof(uint64_t));
            cluDbr->readIndexPairs(0, cluDbr->getSize() + 1, partIndexPairs.data());
            memberIndexReadSize += partIndexSize;
            memberIndexReadCount++;
            for (size_t sampledKey = 0; sampledKey < sampledKeys.size(); sampledKey++) {
                sampledKeys[sampledKey] = partIndexPairs[2 * sampledKey * entriesPerSampledKey];
            }
        } else {
            size_t readSize = 0;
            size_t readCount = 0;
            cluDbr->readSampledKeys(entriesPerSampledKey, std::max(BatchReader::MAX_READ_REQUEST_SIZE, partMemorySize), sampledKeys.data(), readSize, readCount);
            memberIndexReadSize += readSize;
            memberIndexReadCount += readCount;
        }
        if (localClu == clu && partIndexSize + cluDbr->getDataSize() <= partMemorySize) {
            partData.resize(cluDbr->getDataSize());
            FileUtil::preadAll(cluDbr->getBufferedFd(), cluDbr->getDataFileName(), partData.data(), partData.size(), 0);
            memberLookupReadSize += partData.size();
            memberLookupReadCount++;
        }
    }

    ~ClusterMemberLookup() {
        cluDbr->close();
        delete cluDbr;
    }

    // the node whose cluster part holds key; it grows with key
    size_t getNode(DBKeyType key) const {
        return std::upper_bound(cluKeyOffsets.begin(), cluKeyOffsets.end(), key) - cluKeyOffsets.begin() - 1;
    }

    // search the index pairs between two sampled keys
    Entry find(DBKeyType key, EntryReader &entryReader) const {
        Entry entry = { 0, 0, key };
        size_t sampledKey = std::lower_bound(sampledKeys.begin(), sampledKeys.end(), key) - sampledKeys.begin();
        // the entry is between the sampled keys around key, it ends where the next one starts
        size_t entryCount = cluDbr->getSize();
        const size_t first = (sampledKey > 0) ? (sampledKey - 1) * entriesPerSampledKey : 0;
        const size_t last = std::min(sampledKey * entriesPerSampledKey + 1, entryCount);
        const size_t pairEnd = std::min(last + 2, entryCount + 1);
        const bool isIndexInMemory = partIndexPairs.empty() == false;
        if (isIndexInMemory == false && (first < entryReader.indexFirst || pairEnd > entryReader.indexFirst + entryReader.indexPairs.size() / 2)) {
            entryReader.indexFirst = first;
            entryReader.indexPairs.resize(2 * (pairEnd - first));
            cluDbr->readIndexPairs(first, pairEnd, entryReader.indexPairs.data());
            entryReader.indexReadBytes += entryReader.indexPairs.size() * sizeof(uint64_t);
            entryReader.indexReadCount++;
        }
        const uint64_t *indexPairs = isIndexInMemory ? partIndexPairs.data() + 2 * first : entryReader.indexPairs.data() + 2 * (first - entryReader.indexFirst);
        size_t found = 0;
        for (size_t count = last - first; count > 0;) {
            const size_t half = count / 2;
            if (indexPairs[2 * (found + half)] < key) {
                found += half + 1;
                count -= half + 1;
            } else {
                count = half;
            }
        }
        if (first + found < entryCount && indexPairs[2 * found] == key) {
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
            if (lineEnd == NULL && partData.empty() == false) {
                entryReader.data = partData.data();
                entryReader.bufferBegin = 0;
                entryReader.bufferEnd = partData.size();
                continue;
            }
            if (lineEnd == NULL) {
                const size_t bytes = std::min(entry.end - entry.offset, MEMBER_READ_SIZE);
                entryReader.buffer.resize(bytes);
                FileUtil::preadAll(cluDbr->getBufferedFd(), cluDbr->getDataFileName(), entryReader.buffer.data(), bytes, entry.offset);
                entryReader.readBytes += bytes;
                entryReader.readCount++;
                entryReader.data = entryReader.buffer.data();
                entryReader.bufferBegin = entry.offset;
                entryReader.bufferEnd = entry.offset + bytes;
                continue;
            }
            const char *line = entryReader.data + (entry.offset - entryReader.bufferBegin);
            const DBKeyType key = Util::fast_atoi<DBKeyType>(line);
            entry.offset += lineEnd - line + 1;
            if (key != entry.key) {
                memberKey = key;
                entryReader.memberCount++;
                return true;
            }
        }
        return false;
    }

    size_t memorySize() const {
        return sizeof(*this) + cluKeyOffsets.capacity() * sizeof(size_t) + sizeof(DBReader<DBKeyType>) + sampledKeys.capacity() * sizeof(DBKeyType)
               + partIndexPairs.capacity() * sizeof(uint64_t) + partData.capacity();
    }

    // the member buffer and index pairs of an EntryReader; find reads at most entriesPerSampledKey + 3 index pairs
    size_t getEntryReaderSize() const {
        return MEMBER_READ_SIZE + (entriesPerSampledKey + 3) * 2 * sizeof(uint64_t);
    }

private:
    const size_t entriesPerSampledKey;
    DBReader<DBKeyType> *cluDbr;
    std::vector<size_t> cluKeyOffsets;
    std::vector<DBKeyType> sampledKeys;
    // empty if the part does not fit
    std::vector<uint64_t> partIndexPairs;
    std::vector<char> partData;
};

// --filter-cludb-file on several nodes: every node answers for the keys of its own cluster part, through exchange files like the
// sequence requests, so each part is read by one node only
struct ClusterMemberExchange {
    ClusterMemberExchange(const std::string &requests, const std::string &served, const ClusterMemberLookup &lookup, int threads)
        : requests(requests), served(served), lookup(lookup), threads(std::max(1, threads)), requestCount(0), memberFileCount(0), keyCount(0), isSizeRequest(false),
          nodeKeyOffsets(MMseqsNode::count + 1, 0), nextMemberLines(MMseqsNode::count, 0), isAnswered(MMseqsNode::count, 1), nodeCodes(MMseqsNode::count),
          nodeConsumedBytes(MMseqsNode::count, 0), requestedKeyCount(0), servedKeyCount(0), servedSize(0), sizeWaitSeconds(0), memberWaitSeconds(0) {}

    // asks the nodes whose cluster parts hold the keys, which ascend, for the offsets and bytes of their entries (maxMembers 0), or for the members of
    // those entries (SIZE_MAX), or for the members of one entry after consumedBytes, at most maxMembers; the own cluster part answers at once
    void request(const std::vector<DBKeyType> &keys, const size_t *entryOffsets, const size_t *entryBytes, size_t consumedBytes, size_t maxMembers) {
        const unsigned int nodes = MMseqsNode::count;
        const unsigned int rank = MMseqsNode::rank;
        keyCount = keys.size();
        isSizeRequest = maxMembers == 0;
        // the cluster parts hold ascending key ranges, so the keys of a node follow each other
        for (unsigned int node = 0; node <= nodes; node++) {
            nodeKeyOffsets[node] = std::partition_point(keys.begin(), keys.end(), [&](DBKeyType key) { return lookup.getNode(key) < node; }) - keys.begin();
        }
        bool hasRequest = false;
        for (unsigned int node = 0; node < nodes; node++) {
            isAnswered[node] = node == rank || nodeKeyOffsets[node + 1] == nodeKeyOffsets[node];
            hasRequest |= isAnswered[node] == 0;
            nodeCodes[node].clear();
        }
        if (hasRequest) {
            // request line: request, isLast, consumed bytes, max members, key offsets per node, end, magic; the keys of a node are sent as gaps
            std::vector<uint64_t> line(nodes + 6);
            requestCodes.clear();
            char code[10];
            for (unsigned int node = 0; node < nodes; node++) {
                line[4 + node] = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + requestCodes.size();
                for (size_t i = nodeKeyOffsets[node]; node != rank && i < nodeKeyOffsets[node + 1]; i++) {
                    requestCodes.append(code, Util::putPrefKey(code, keys[i] - ((i > nodeKeyOffsets[node]) ? keys[i - 1] : 0)));
                    // read the members at the answered offsets
                    if (maxMembers != 0) {
                        requestCodes.append(code, Util::putPrefKey(code, entryOffsets[i] - ((i > nodeKeyOffsets[node]) ? entryOffsets[i - 1] : 0)));
                        requestCodes.append(code, Util::putPrefKey(code, entryBytes[i]));
                    }
                    requestedKeyCount++;
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
        const size_t ownKey = nodeKeyOffsets[rank];
        answer(keys.data() + ownKey, nodeKeyOffsets[rank + 1] - ownKey, (maxMembers != 0) ? entryOffsets + ownKey : NULL, (maxMembers != 0) ? entryBytes + ownKey : NULL,
               consumedBytes, maxMembers, nodeCodes[rank], nodeConsumedBytes[rank]);
    }

    // wait for the answers to the last request
    void receive() {
        const unsigned int nodes = MMseqsNode::count;
        const unsigned int rank = MMseqsNode::rank;
        size_t owingCount = std::count(isAnswered.begin(), isAnswered.end(), 0);
        const size_t polledNodes = std::min<size_t>(SERVICE_POLL_NODES, std::max<size_t>(1, owingCount));
        const size_t pollMicroseconds = std::min(MAX_SERVICE_PASS_MICROSECONDS, polledNodes * nodes * 1000000 / SERVICE_OPENS_PER_SECOND);
        std::vector<uint64_t> memberLines(16 * MEMBER_LINE_WORDS);
        MMseqsNode::WaitWarning waitWarning;
        Timer waitTimer;
        for (size_t pollTurn = 0; owingCount > 0;) {
            bool hasWorked = false;
            size_t polledCount = 0;
            size_t i = 0;
            for (; i < nodes && polledCount < polledNodes; i++) {
                unsigned int node = static_cast<unsigned int>((pollTurn + i) % nodes);
                if (isAnswered[node]) {
                    continue;
                }
                polledCount++;
                const size_t lineCount = MMseqsNode::readExchangeIndexLines(getMemberIndex(node), nextMemberLines[node], MEMBER_LINE_WORDS, memberLines);
                nextMemberLines[node] += lineCount;
                // more lines may follow a full read
                hasWorked |= lineCount * MEMBER_LINE_WORDS == memberLines.size();
                for (size_t l = 0; l < lineCount; l++) {
                    // member line: member file, requester, request, size, consumed bytes, magic; a line of another requester is skipped
                    const uint64_t *line = memberLines.data() + l * MEMBER_LINE_WORDS;
                    if (line[1] != rank || line[2] != requestCount - 1) {
                        continue;
                    }
                    const std::string memberFile = getMemberFile(node, line[0]);
                    const int fd = MMseqsNode::waitForExchangeFile(memberFile, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + line[3], line[0]);
                    nodeCodes[node].resize(line[3]);
                    FileUtil::preadAll(fd, memberFile.c_str(), &nodeCodes[node][0], nodeCodes[node].size(), MMseqsNode::EXCHANGE_FILE_HEADER_BYTES);
                    if (close(fd) != 0) {
                        Debug(Debug::ERROR) << "Cannot close " << memberFile << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    // only this node reads it
                    FileUtil::remove(memberFile.c_str());
                    nodeConsumedBytes[node] = line[4];
                    isAnswered[node] = 1;
                    owingCount--;
                    hasWorked = true;
                }
            }
            // start the next pass behind the last node
            pollTurn += i;
            if (hasWorked == false && owingCount > 0) {
                usleep(pollMicroseconds);
                waitWarning.check(served);
            }
        }
        (isSizeRequest ? sizeWaitSeconds : memberWaitSeconds) += waitTimer.getTimediff();
    }

    // the entry offsets and bytes of the keys of the last request; an answer is released once read
    void getEntries(std::vector<size_t> &entryOffsets, std::vector<size_t> &entryBytes) {
        entryOffsets.resize(keyCount);
        entryBytes.resize(keyCount);
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            const char *data = nodeCodes[node].data();
            for (size_t i = nodeKeyOffsets[node]; i < nodeKeyOffsets[node + 1]; i++) {
                entryOffsets[i] = Util::getPrefKey(data);
                entryBytes[i] = Util::getPrefKey(data);
            }
            std::string().swap(nodeCodes[node]);
        }
    }

    // members[memberOffsets[i], memberOffsets[i + 1]) belong to the key i of the last request; an answer is released once read
    void getMembers(std::vector<size_t> &memberOffsets, std::vector<DBKeyType> &members) {
        memberOffsets.assign(keyCount + 1, 0);
        members.clear();
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            const char *data = nodeCodes[node].data();
            for (size_t i = nodeKeyOffsets[node]; i < nodeKeyOffsets[node + 1]; i++) {
                for (uint64_t code = Util::getPrefKey(data); code != 0; code = Util::getPrefKey(data)) {
                    members.push_back(static_cast<DBKeyType>(code - 1));
                }
                memberOffsets[i + 1] = members.size();
            }
            std::string().swap(nodeCodes[node]);
        }
    }

    // the consumed bytes of the entry of the one key of the last request
    size_t getConsumedBytes() const {
        size_t node = 0;
        while (nodeKeyOffsets[node + 1] == nodeKeyOffsets[node]) {
            node++;
        }
        return nodeConsumedBytes[node];
    }

    // no more requests from this node
    void finish() {
        const unsigned int nodes = MMseqsNode::count;
        std::vector<uint64_t> line(nodes + 6, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES);
        line[0] = requestCount;
        line[1] = 1;
        MMseqsNode::writeExchangeFile(getRequestFile(MMseqsNode::rank), requestCount, NULL, 0, getRequestIndex(MMseqsNode::rank), line, 1);
        requestCount++;
    }

    // answers the requests of the aligning nodes one at a time until every one of them called finish
    void serve(size_t firstAlignNode) {
        const unsigned int nodes = MMseqsNode::count;
        const unsigned int rank = MMseqsNode::rank;
        const size_t requestLineWords = nodes + 6;
        std::vector<uint64_t> requestLines(16 * requestLineWords);
        std::vector<size_t> nextRequestLines(nodes, 0);
        std::vector<unsigned char> isRequesting(nodes, 0);
        size_t requesterCount = 0;
        for (size_t node = firstAlignNode; node < nodes; node++) {
            isRequesting[node] = node != rank;
            requesterCount += isRequesting[node];
        }
        // like the sequence service, every node opens at most SERVICE_OPENS_PER_SECOND / nodes indexes a second while idle
        const size_t polledNodes = std::min<size_t>(SERVICE_POLL_NODES, std::max<size_t>(1, requesterCount));
        const size_t pollMicroseconds = std::min(MAX_SERVICE_PASS_MICROSECONDS, polledNodes * nodes * 1000000 / SERVICE_OPENS_PER_SECOND);
        std::vector<char> section;
        std::vector<DBKeyType> requestedKeys;
        std::vector<size_t> requestedOffsets;
        std::vector<size_t> requestedBytes;
        std::string reply;
        unsigned int polls = 0;
        for (size_t pollTurn = 0; requesterCount > 0;) {
            bool hasWorked = false;
            size_t polledCount = 0;
            size_t i = 0;
            for (; i < nodes - firstAlignNode && polledCount < polledNodes; i++) {
                unsigned int requester = static_cast<unsigned int>(firstAlignNode + (pollTurn + i) % (nodes - firstAlignNode));
                if (isRequesting[requester] == 0) {
                    continue;
                }
                polledCount++;
                const size_t lineCount = MMseqsNode::readExchangeIndexLines(getRequestIndex(requester), nextRequestLines[requester], requestLineWords, requestLines);
                nextRequestLines[requester] += lineCount;
                for (size_t l = 0; l < lineCount; l++) {
                    const uint64_t *line = requestLines.data() + l * requestLineWords;
                    const size_t sectionBegin = line[4 + rank];
                    const size_t sectionEnd = line[5 + rank];
                    if (sectionEnd > sectionBegin) {
                        const std::string requestFile = getRequestFile(requester);
                        const int fd = MMseqsNode::waitForExchangeFile(requestFile, line[4 + nodes], line[0]);
                        section.resize(sectionEnd - sectionBegin);
                        FileUtil::preadAll(fd, requestFile.c_str(), section.data(), section.size(), sectionBegin);
                        if (close(fd) != 0) {
                            Debug(Debug::ERROR) << "Cannot close " << requestFile << "\n";
                            EXIT(EXIT_FAILURE);
                        }
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
                        servedKeyCount += requestedKeys.size();
                        servedSize += reply.size();
                        std::string().swap(reply);
                    }
                    if (line[1] != 0) {
                        isRequesting[requester] = 0;
                        requesterCount--;
                    }
                    hasWorked = true;
                }
            }
            // start the next pass behind the last requester
            pollTurn += i;
            if (hasWorked) {
                polls = 0;
            } else if (requesterCount > 0) {
                // idle passes last 10, 20, then 40 ms
                usleep(std::max(pollMicroseconds, (size_t) 10000 << std::min(polls++, 2u)));
            }
        }
    }

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

    // the answer for keys of the own cluster part in key order: the entry offset and bytes per key, or per key its members plus one,
    // ended by 0; the serving thread and the aligning thread call it at once, so it keeps no state
    void answer(const DBKeyType *keys, size_t answeredKeyCount, const size_t *entryOffsets, const size_t *entryBytes, size_t consumedBytes, size_t maxMembers,
                std::string &codes, size_t &endConsumedBytes) const {
        endConsumedBytes = 0;
        std::vector<std::string> threadCodes(threads);
#pragma omp parallel num_threads(threads)
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            ClusterMemberLookup::EntryReader entryReader;
            char code[10];
            // a static schedule gives every thread one run of ascending keys
#pragma omp for schedule(static)
            for (size_t i = 0; i < answeredKeyCount; i++) {
                if (maxMembers == 0) {
                    // the threads answer in parallel, so the offsets go whole
                    const ClusterMemberLookup::Entry entry = lookup.find(keys[i], entryReader);
                    threadCodes[thread_idx].append(code, Util::putPrefKey(code, entry.offset));
                    threadCodes[thread_idx].append(code, Util::putPrefKey(code, entry.end - entry.offset));
                    continue;
                }
                const size_t entryBegin = entryOffsets[i];
                ClusterMemberLookup::Entry entry = { entryBegin + consumedBytes, entryBegin + entryBytes[i], keys[i] };
                entryReader.start(entry);
                DBKeyType memberKey;
                for (size_t member = 0; member < maxMembers && lookup.next(entryReader, memberKey); member++) {
                    threadCodes[thread_idx].append(code, Util::putPrefKey(code, static_cast<uint64_t>(memberKey) + 1));
                }
                threadCodes[thread_idx].append(code, Util::putPrefKey(code, 0));
                // a slice has one key, its entry offset ends behind its last member
                if (answeredKeyCount == 1) {
                    endConsumedBytes = entryReader.entry.offset - entryBegin;
                }
            }
        }
        codes.clear();
        for (int thread = 0; thread < threads; thread++) {
            codes += threadCodes[thread];
        }
    }

    const std::string requests;
    const std::string served;
    const ClusterMemberLookup &lookup;
    const int threads;
    uint64_t requestCount;
    uint64_t memberFileCount;
    size_t keyCount;
    bool isSizeRequest;
    std::vector<size_t> nodeKeyOffsets;
    std::vector<size_t> nextMemberLines;
    std::vector<unsigned char> isAnswered;
    std::vector<std::string> nodeCodes;
    std::vector<size_t> nodeConsumedBytes;
    std::string requestCodes;

public:
    // the keys this node asked the other nodes for, and the keys and bytes it served; read once serve is done
    size_t requestedKeyCount;
    size_t servedKeyCount;
    size_t servedSize;
    // seconds in receive
    double sizeWaitSeconds;
    double memberWaitSeconds;

    uint64_t getRequestCount() const {
        return requestCount;
    }
    uint64_t getAnswerCount() const {
        return memberFileCount;
    }
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
            size_t alignmentSplit = (MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) ? alignmentSplitQueue.takeNextItemInOrder() : alignmentSplitCount;
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
            size_t targetRange = SIZE_MAX;
            for (size_t first = 0; first < targetCount; first += maxRecordTargetCount, j++) {
                RepTargetsRecord &record = records[j];
                record.repOrderKeyFirstTarget.repKey = (sizeClassCount - 1 - getSizeClass(targetCount, sizeClassCount)) * paddedDbSize + repKey;
                record.repOrderKeyFirstTarget.memberKey = entryTargets[first];
                record.targetCount = static_cast<unsigned int>(std::min(targetCount - first, maxRecordTargetCount));
                record.targetOffset = entryTargets + first - targets.data();
                record.readSize = repSize;
                for (unsigned int i = 0; i < record.targetCount; i++) {
                    record.readSize += seqDbr.getSeqLenFromRange(targetRange, entryTargets[first + i]) + 2 + FETCHED_KEY_SIZE;
                }
                prefReadFetchSize += record.readSize;
            }
        }
        bufferFetchSize += prefReadFetchSize;
    }
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

// target split mode: join the pieces of a rep entry by entry
struct TargetSplitReader {
    TargetSplitReader(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<PrefPartition> &cells, MMseqsNode::WorkQueue &cellQueue, size_t maxRecordSize, size_t memoryLimit,
                      size_t paddedDbSize, size_t sizeClassCount)
        : readSize(0), prefSize(0), homeCellCount(0), stolenCellCount(0), lostCellCount(0), stolenCost(0), takenCost(0), stolenTargetBytes(0), targetBytes(0),
          stealShares(MMseqsNode::count, 0), doneRepKeys(MMseqsNode::count, 0), stolenCostSignChanges(0), stealWindowCount(0),
          isClaiming(sizeClassCount == 1 && MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()), par(par), seqDbr(seqDbr), cells(cells), cellQueue(cellQueue),
          targetSplitCount(targetSplitBlockOffsets.size() - 1), paddedDbSize(paddedDbSize), sizeClassCount(sizeClassCount),
          maxPrefilterBufferSize(std::max((size_t) 1, memoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR)),
          maxRecordTargetCount(std::min((size_t) UINT_MAX, std::max((size_t) 1, maxRecordSize / std::max(seqDbr.getMaxSeqLen() + 2, Util::getPageSize())))),
          idleShares(MMseqsNode::count, 0), stealCarries(MMseqsNode::count, 0), wasStolen(targetSplitCount, 0), cellsPerSecond(0), waitSeconds(0), usedWaitSeconds(0),
          lastStolenCost(0), lastStolenCostChange(0), cellPos(0), prefReadPos(0), passClasses(1), endClass(0), classSize(sizeClassCount, 0), prefNode(SIZE_MAX), bufferFetchSize(0), bufferAlignmentCells(0) {
        // greedy claims its splits per window
        for (size_t split = 0; isClaiming == false && split < targetSplitCount; split++) {
            if (targetSplitAlignNodes[split] == MMseqsNode::rank) {
                targetSplits.push_back(split);
                targetSplitFirstKeys.push_back(seqDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[split]]);
                targetSplitIsStolen.push_back(0);
            }
        }
        for (size_t i = 0; isClaiming == false && i < cells.size(); i++) {
            prefSize += (targetSplitAlignNodes[cells[i].targetSplit] == MMseqsNode::rank) ? cells[i].prefSize : 0;
        }
        prefOffsets.resize(MMseqsNode::count * targetSplitCount);
    }

    // cells per second and master wait so far
    void setWindowWork(double rate, double idle) {
        cellsPerSecond = rate;
        waitSeconds = idle;
    }

    // load every window once and mark its size classes
    void countSizeClasses() {
        std::vector<RepTargetsRecord> records;
        std::vector<DBKeyType> targets;
        passClasses = ~0ULL;
        for (prefReadPos = 0; prefReadPos < prefReads.size() || takeWindow();) {
            PrefRead &prefRead = prefReads[prefReadPos];
            records.clear();
            targets.clear();
            loadPrefRead(records, targets);
            uint64_t sizeClasses = 0;
#pragma omp parallel
            {
                std::vector<size_t> threadClassSize(sizeClassCount, 0);
#pragma omp for schedule(dynamic, 100) reduction(|:sizeClasses)
                for (size_t i = 0; i < records.size(); i++) {
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
            prefRead.sizeClasses = sizeClasses;
        }
        endClass = sizeClassCount;
    }

    void fillPrefilterBuffer(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, size_t maxReadSize, size_t maxBufferAlignmentCells) {
        records.clear();
        targets.clear();
        bufferFetchSize = 0;
        bufferAlignmentCells = 0;
        while (records.size() * sizeof(RepTargetsRecord) + targets.size() * sizeof(DBKeyType) < maxPrefilterBufferSize) {
            // a buffer ends at a window end
            if (prefReadPos < prefReads.size() || (records.empty() && takeWindow())) {
                // keep a pass of several size classes in one buffer
                if (records.empty() == false && (passClasses & (passClasses - 1)) == 0 && (bufferFetchSize >= maxReadSize || bufferAlignmentCells >= maxBufferAlignmentCells)) {
                    break;
                }
                loadPrefRead(records, targets);
            } else if (records.empty() == false || startPass() == false) {
                break;
            }
        }
        SORT_PARALLEL(records.begin(), records.end(), RepTargetsRecord::compareByRep);
    }

    void close() {
        closePrefDbs();
    }

    size_t readSize;
    size_t prefSize;
    size_t homeCellCount;
    size_t stolenCellCount;
    size_t lostCellCount;
    double stolenCost;
    double takenCost;
    size_t stolenTargetBytes;
    size_t targetBytes;
    std::vector<double> stealShares;
    std::vector<size_t> doneRepKeys;
    size_t stolenCostSignChanges;
    size_t stealWindowCount;
    const bool isClaiming;

private:
    Parameters &par;
    DBReader<DBKeyType> &seqDbr;
    const std::vector<PrefPartition> &cells;
    MMseqsNode::WorkQueue &cellQueue;
    const size_t targetSplitCount;
    const size_t paddedDbSize;
    const size_t sizeClassCount;
    const size_t maxPrefilterBufferSize;
    const size_t maxRecordTargetCount;
    std::vector<double> idleShares;
    std::vector<double> stealCarries;
    std::vector<size_t> windowDoneRepKeys;
    std::vector<unsigned char> wasStolen;
    double cellsPerSecond;
    double waitSeconds;
    double usedWaitSeconds;
    double lastStolenCost;
    int lastStolenCostChange;
    std::vector<size_t> targetSplits;
    std::vector<DBKeyType> targetSplitFirstKeys;
    std::vector<unsigned char> targetSplitIsStolen;
    size_t cellPos;
    std::vector<PrefRead> prefReads;
    size_t prefReadPos;
    uint64_t passClasses;
    size_t endClass;
    std::vector<size_t> classSize;
    std::vector<DBReader<DBKeyType> *> prefDbs;
    size_t prefNode;
    // per node and target split
    std::vector<std::vector<uint64_t> > prefOffsets;
    std::vector<size_t> entrySizes;
    size_t bufferFetchSize;
    size_t bufferAlignmentCells;
    std::vector<size_t> entryTargetOffsets;
    std::vector<size_t> entryRecordOffsets;

    void openPrefDbs(size_t node) {
        if (node == prefNode) {
            return;
        }
        closePrefDbs();
        prefNode = node;
        for (size_t i = 0; i < targetSplits.size(); i++) {
            std::string file = SSTR(node * targetSplitCount + targetSplits[i]);
            prefDbs.push_back(new DBReader<DBKeyType>((par.db2 + "." + file).c_str(), (par.db2 + ".index." + file).c_str(), par.threads,
                                                      DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY));
            prefDbs.back()->open(DBReader<DBKeyType>::NOSORT);
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

    // a window has a cell per target split
    bool takeWindow() {
        // 10 bytes per pref byte, 48 bytes and a record per piece
        size_t maxPrefReadSize = std::max((size_t) 1, maxPrefilterBufferSize / (10 + (48 + sizeof(RepTargetsRecord)) / 4));
        size_t prefReadsBefore = prefReads.size();
        for (; prefReads.size() == prefReadsBefore && cellPos < cells.size(); cellPos += targetSplitCount) {
            if (isClaiming) {
                claimWindowCells(cellPos / targetSplitCount);
            }
            size_t windowCost = 0;
            for (size_t i = 0; i < targetSplits.size(); i++) {
                windowCost += cells[cellPos + targetSplits[i]].alignmentCells;
            }
            if (windowCost == 0) {
                continue;
            }
            const PrefPartition &cell = cells[cellPos + targetSplits[0]];
            // read each index once
            std::vector<uint64_t> *offsets = prefOffsets.data() + cell.node * targetSplitCount;
            for (size_t i = 0; i < targetSplits.size(); i++) {
                std::vector<uint64_t> &splitOffsets = offsets[targetSplits[i]];
                if (splitOffsets.empty()) {
                    std::string index = MMseqsNode::readFile(par.db2 + ".index." + SSTR(cell.node * targetSplitCount + targetSplits[i]));
                    splitOffsets.resize(index.size() / (2 * sizeof(uint64_t)));
                    for (size_t entry = 0; entry < splitOffsets.size(); entry++) {
                        memcpy(&splitOffsets[entry], index.data() + (2 * entry + 1) * sizeof(uint64_t), sizeof(uint64_t));
                    }
                }
            }

            const std::vector<uint64_t> &firstOffsets = offsets[targetSplits[0]];
            size_t firstEntry = std::lower_bound(firstOffsets.begin(), firstOffsets.end(), cell.prefOffset) - firstOffsets.begin();
            size_t entryCount = std::lower_bound(firstOffsets.begin(), firstOffsets.end(), cell.prefOffset + cell.prefSize) - firstOffsets.begin() - firstEntry;
            entrySizes.assign(entryCount, 0);
            for (size_t i = 0; i < targetSplits.size(); i++) {
                const std::vector<uint64_t> &splitOffsets = offsets[targetSplits[i]];
                for (size_t entry = 0; entry < entryCount; entry++) {
                    entrySizes[entry] += splitOffsets[firstEntry + entry + 1] - splitOffsets[firstEntry + entry];
                }
            }

            // cut the window into pref reads
            size_t size = 0;
            for (size_t entry = 0; entry < entryCount; entry++) {
                if (entry == 0 || size + entrySizes[entry] > maxPrefReadSize) {
                    PrefRead prefRead = { cell.node, firstEntry + entry, firstEntry + entry + 1, ~0ULL };
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

    // take the free home cells, then cells of nodes behind
    void claimWindowCells(size_t window) {
        size_t firstCell = window * targetSplitCount;
        size_t firstAlignNode = MMseqsNode::getFirstAlignNode();
        size_t alignNodeCount = MMseqsNode::count - firstAlignNode;
        std::vector<size_t> nodeCosts(MMseqsNode::count, 0);
        std::vector<unsigned char> isTaken(targetSplitCount, 0);
        double windowTakenCost = 0;
        for (size_t split = 0; split < targetSplitCount; split++) {
            size_t cost = cells[firstCell + split].alignmentCells;
            nodeCosts[targetSplitAlignNodes[split]] += cost;
            if (cost > 0 && targetSplitAlignNodes[split] == MMseqsNode::rank) {
                isTaken[split] = cellQueue.takeItem(firstCell + split);
                homeCellCount += isTaken[split];
                lostCellCount += 1 - isTaken[split];
                windowTakenCost += isTaken[split] ? cost : 0;
            }
        }

        // a node is behind by the windows since this node had its done rep key
        std::vector<unsigned char> isBehind(MMseqsNode::count, 0);
        std::vector<size_t> behindWindows(MMseqsNode::count, 0);
        size_t behindNodes = 0;
        size_t aheadNodes = 1;
        for (size_t node = firstAlignNode; node < MMseqsNode::count; node++) {
            if (node == MMseqsNode::rank || nodeCosts[node] == 0) {
                continue;
            }
            size_t doneRepKey = __atomic_load_n(&doneRepKeys[node], __ATOMIC_RELAXED);
            size_t behind = windowDoneRepKeys.end() - std::upper_bound(windowDoneRepKeys.begin(), windowDoneRepKeys.end(), doneRepKey);
            if (behind == 0) {
                idleShares[node] /= 2;
                aheadNodes++;
                continue;
            }
            isBehind[node] = 1;
            behindNodes++;
            behindWindows[node] = std::min(behind - 1, (size_t) (0.5 / STEAL_SHARE_STEP));
        }
        // a done key once, so behind counts the master steps and not the claims between them
        size_t ownDoneRepKey = __atomic_load_n(&doneRepKeys[MMseqsNode::rank], __ATOMIC_RELAXED);
        if (windowDoneRepKeys.empty() || windowDoneRepKeys.back() != ownDoneRepKey) {
            windowDoneRepKeys.push_back(ownDoneRepKey);
        }

        double idleCells = cellsPerSecond * std::max(0.0, waitSeconds - usedWaitSeconds);
        usedWaitSeconds = waitSeconds;
        double windowStolenCost = 0;
        std::vector<size_t> splits;
        for (size_t i = 1; i < alignNodeCount; i++) {
            size_t node = firstAlignNode + (MMseqsNode::rank - firstAlignNode + i) % alignNodeCount;
            if (isBehind[node] == 0) {
                continue;
            }
            // idle seconds add up, windows behind add the cost difference
            double cost = static_cast<double>(nodeCosts[node]);
            idleShares[node] = std::min(0.5, idleShares[node] + idleCells / (2.0 * cost * behindNodes));
            double balanceShare = (behindWindows[node] > 0) ? std::max(0.0, cost - windowTakenCost) / (2.0 * cost) : 0.0;
            stealShares[node] = std::min(0.5, idleShares[node] + (balanceShare + behindWindows[node] * STEAL_SHARE_STEP) / aheadNodes);
            stealCarries[node] = std::min(stealCarries[node] + stealShares[node] * cost, cost);
            // splits stolen before first, then the largest cells
            splits.clear();
            for (size_t split = 0; split < targetSplitCount; split++) {
                if (targetSplitAlignNodes[split] == node && cells[firstCell + split].alignmentCells > 0) {
                    splits.push_back(split);
                }
            }
            std::sort(splits.begin(), splits.end(), [&](size_t first, size_t second) {
                size_t firstCost = cells[firstCell + first].alignmentCells;
                size_t secondCost = cells[firstCell + second].alignmentCells;
                return wasStolen[first] > wasStolen[second] || (wasStolen[first] == wasStolen[second] && (firstCost > secondCost || (firstCost == secondCost && first < second)));
            });
            for (size_t j = 0; j < splits.size() && cells[firstCell + splits[j]].alignmentCells <= stealCarries[node]; j++) {
                if (cellQueue.takeItem(firstCell + splits[j])) {
                    isTaken[splits[j]] = 1;
                    stealCarries[node] -= cells[firstCell + splits[j]].alignmentCells;
                    windowTakenCost += cells[firstCell + splits[j]].alignmentCells;
                    windowStolenCost += cells[firstCell + splits[j]].alignmentCells;
                    stolenCellCount++;
                }
            }
        }

        stolenCost += windowStolenCost;
        stealWindowCount += windowStolenCost > 0;
        if (windowStolenCost != lastStolenCost) {
            int change = (windowStolenCost > lastStolenCost) ? 1 : -1;
            stolenCostSignChanges += lastStolenCostChange != 0 && change != lastStolenCostChange;
            lastStolenCostChange = change;
            lastStolenCost = windowStolenCost;
        }

        splits.clear();
        targetSplitIsStolen.clear();
        for (size_t split = 0; split < targetSplitCount; split++) {
            wasStolen[split] = isTaken[split] && targetSplitAlignNodes[split] != MMseqsNode::rank;
            if (isTaken[split]) {
                splits.push_back(split);
                targetSplitIsStolen.push_back(wasStolen[split]);
                prefSize += cells[firstCell + split].prefSize;
                takenCost += cells[firstCell + split].alignmentCells;
            }
        }
        // reopen the pref readers for new splits
        if (splits != targetSplits) {
            closePrefDbs();
            targetSplits.swap(splits);
            targetSplitFirstKeys.clear();
            for (size_t i = 0; i < targetSplits.size(); i++) {
                targetSplitFirstKeys.push_back(seqDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[targetSplits[i]]]);
            }
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

    void startEntry(size_t entry, std::vector<TargetSplitPiece> &pieces) {
        for (size_t i = 0; i < prefDbs.size(); i++) {
            pieces[i].data = prefDbs[i]->getData(entry, 0);
            pieces[i].end = pieces[i].data + prefDbs[i]->getEntryLen(entry) - 1;
            pieces[i].repKey = prefDbs[i]->getDbKey(entry);
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
        const PrefRead &prefRead = prefReads[prefReadPos];
        openPrefDbs(prefRead.prefNode);
        for (size_t i = 0; i < prefDbs.size(); i++) {
            prefDbs[i]->loadData(prefRead.firstEntry, prefRead.endEntry);
        }
        prefReadPos = findPassPrefRead(prefReadPos + 1);
        // prefetch here, io_uring reads can fail once their thread ends
        if (prefReadPos < prefReads.size() && prefReads[prefReadPos].prefNode == prefRead.prefNode) {
            for (size_t i = 0; i < prefDbs.size(); i++) {
                prefDbs[i]->prefetchData(prefReads[prefReadPos].firstEntry, prefReads[prefReadPos].endEntry);
            }
        }

        // count the targets and records of every entry
        size_t firstEntry = prefRead.firstEntry;
        size_t entryCount = prefRead.endEntry - firstEntry;
        entryTargetOffsets.resize(entryCount + 1);
        entryRecordOffsets.resize(entryCount + 1);
        entryTargetOffsets[0] = 0;
        entryRecordOffsets[0] = 0;
        size_t size = 0;
#pragma omp parallel reduction(+:size)
        {
            std::vector<TargetSplitPiece> pieces(prefDbs.size());
#pragma omp for schedule(dynamic, 1)
            for (size_t entry = 0; entry < entryCount; entry++) {
                startEntry(firstEntry + entry, pieces);
                size_t targetCount = 0;
                size_t recordCount = 0;
                for (DBKeyType repKey = getNextRepKey(pieces); repKey != DB_KEY_INVALID; repKey = getNextRepKey(pieces)) {
                    size_t repTargetCount = 0;
                    size_t sizeClass = 0;
                    for (size_t i = 0; i < pieces.size(); i++) {
                        if (pieces[i].repKey == repKey) {
                            repTargetCount += pieces[i].targetCount;
                            sizeClass = std::min(sizeClassCount - 1, pieces[i].sizeClass);
                            pieces[i].next();
                        }
                    }
                    if (((passClasses >> sizeClass) & 1) != 0) {
                        targetCount += repTargetCount;
                        recordCount += (repTargetCount + maxRecordTargetCount - 1) / maxRecordTargetCount;
                    }
                }
                entryTargetOffsets[entry + 1] = targetCount;
                entryRecordOffsets[entry + 1] = recordCount;
                for (size_t i = 0; i < prefDbs.size(); i++) {
                    size += prefDbs[i]->getEntryLen(firstEntry + entry);
                }
            }
        }
        readSize += size;
        sumOffsets(entryTargetOffsets.data(), entryCount, par.threads);
        sumOffsets(entryRecordOffsets.data(), entryCount, par.threads);

        // write the records of an entry at its offsets
        size_t firstRecord = records.size();
        size_t firstTarget = targets.size();
        records.resize(firstRecord + entryRecordOffsets[entryCount]);
        targets.resize(firstTarget + entryTargetOffsets[entryCount]);
        size_t fetchSize = 0;
        size_t alignCells = 0;
        size_t targetSize = 0;
        size_t stolenSize = 0;
#pragma omp parallel reduction(+:fetchSize, alignCells, targetSize, stolenSize)
        {
            std::vector<TargetSplitPiece> pieces(prefDbs.size());
            size_t stolenRange = SIZE_MAX;
#pragma omp for schedule(dynamic, 1)
            for (size_t entry = 0; entry < entryCount; entry++) {
                if (entryTargetOffsets[entry + 1] == entryTargetOffsets[entry]) {
                    continue;
                }
                startEntry(firstEntry + entry, pieces);
                DBKeyType *repTargets = targets.data() + firstTarget + entryTargetOffsets[entry];
                size_t j = firstRecord + entryRecordOffsets[entry];
                for (DBKeyType repKey = getNextRepKey(pieces); repKey != DB_KEY_INVALID; repKey = getNextRepKey(pieces)) {
                    // pieces in target split order, so targets ascend
                    size_t repTargetCount = 0;
                    size_t sizeClass = 0;
                    for (size_t i = 0; i < pieces.size(); i++) {
                        if (pieces[i].repKey == repKey) {
                            sizeClass = std::min(sizeClassCount - 1, pieces[i].sizeClass);
                            const char *data = pieces[i].targets;
                            size_t targetCount = (((passClasses >> sizeClass) & 1) != 0) ? pieces[i].targetCount : 0;
                            DBKeyType target = targetSplitFirstKeys[i] - 1;
                            for (size_t k = 0; k < targetCount; k++) {
                                target += Util::getPrefKey(data);
                                repTargets[repTargetCount++] = target;
                                stolenSize += targetSplitIsStolen[i] ? getSeqLenFromCursor(seqDbr, stolenRange, target) + 2 : 0;
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
                            targetSize += targetLength + 2;
                        }
                        fetchSize += record.readSize;
                    }
                    repTargets += repTargetCount;
                }
            }
        }
        bufferFetchSize += fetchSize;
        bufferAlignmentCells += alignCells;
        targetBytes += targetSize;
        stolenTargetBytes += stolenSize;
    }
};

// examined (rep, target) pairs of the aligning loop
static std::atomic<size_t> examinedTargetPairCount(0);
static std::atomic<size_t> repSkippedTargetPairCount(0);
static std::atomic<size_t> targetSkippedTargetPairCount(0);
static std::atomic<size_t> uncoverableTargetPairCount(0);
static std::atomic<size_t> alignedTargetPairCount(0);
// skips by empty slot and pairs the master drops
static std::atomic<size_t> repSlotSkippedTargetPairCount(0);
static std::atomic<size_t> repDroppedAcceptedPairCount(0);
static std::atomic<size_t> targetDroppedAcceptedPairCount(0);

struct TargetAligners {
    Sequence query;
    Sequence target;
    BlockAligner blockAligner;
    // --align2clust-aligner-mode 1: the reps of a batch share the SWIPE lanes
    SwipeTargets swipeTargets;

    TargetAligners(const Parameters &par, AlignmentScoring &scoring, size_t maxSeqLen)
        : query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection),
          target(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection),
          blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                       &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                       -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid()),
          swipeTargets(par, scoring.subMat, scoring.evaluer, maxSeqLen) {}
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

// one check of all members of a pair aligns its rounds of less than 4 members outside the SWIPE lanes; those are among the first 3 and the
// last 3 members in the order of prepareMemberChecks, which is the order of (-length difference, list position)
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

    // the list positions of the members outside the lanes
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

static void alignRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, RepTargetsReader &repTargetsReader, TargetSplitReader *targetSplitReader,
                            size_t memoryLimit, size_t masterMemoryLimit, size_t servedFileEntrySize, size_t representativeResidues,
                            size_t paddedDbSize, size_t sizeClassCount, const ClusterMemberLookup *memberLookup, ClusterMemberExchange *memberExchange,
                            size_t memberKeyBudget, DBReader<DBKeyType> *memberDbr, bool hasLocalServedBlocks, std::string &phaseValues) {
    Timer startTimer;
    const unsigned int nodes = MMseqsNode::count;
    const unsigned int rank = MMseqsNode::rank;
    const size_t firstAlignNode = MMseqsNode::getFirstAlignNode();
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
        MMseqsNode::publishLog(acceptedHandle, acceptedFile, repKey, bytes);
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
    size_t assignedKeysSize = 0;
    size_t assignedKeyCount = 0;
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
            size_t keyCount = assignedKeyWords[1 + MMseqsNode::count];
            size_t key = assignedKeyWords[2 + MMseqsNode::count];
            unsigned int keyDiffLowBits = assignedKeyWords[3 + MMseqsNode::count];
            size_t codeWords = assignedKeyWords[4 + MMseqsNode::count];
            for (size_t node = 0; targetSplitReader != NULL && node < MMseqsNode::count; node++) {
                __atomic_store_n(&targetSplitReader->doneRepKeys[node], assignedKeyWords[5 + MMseqsNode::count + node], __ATOMIC_RELAXED);
            }
            size_t codeStart = assignedKeysSize + headerWords * sizeof(uint64_t);
            assignedKeyWords.resize(codeWords);
            if (MMseqsNode::readLog(assignedKeysLog, codeStart, codeStart + codeWords * sizeof(uint64_t), (char *) assignedKeyWords.data(), wait) == false) {
                return;
            }
            KmerCodeReader codeRead = { (char *) assignedKeyWords.data(), 0, 0, 0, -1, assignedKeysLog.c_str(), codeWords * sizeof(uint64_t), 0, codeWords * sizeof(uint64_t) };
            // only the thread that holds assignedKeysMutex adds pages
            for (size_t i = 0; i < keyCount; i++) {
                key += (i > 0) ? getRice(codeRead, keyDiffLowBits) + 1 : 0;
                uint64_t *page = assignedPages[key / ASSIGNED_PAGE_KEYS];
                if (page == NULL) {
                    page = new(std::nothrow) uint64_t[ASSIGNED_PAGE_KEYS / 64]();
                    Util::checkAllocation(page, "Can not allocate assigned page memory in align2clust");
                    __atomic_store_n(&assignedPages[key / ASSIGNED_PAGE_KEYS], page, __ATOMIC_RELEASE);
                }
                __atomic_fetch_or(&page[key % ASSIGNED_PAGE_KEYS / 64], 1ULL << (key % 64), __ATOMIC_RELAXED);
            }
            assignedKeysSize = codeStart + codeWords * sizeof(uint64_t);
            assignedKeyCount += keyCount;
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
    // the lane flags of one check of the whole pair, so that its parts align every member alike
    std::vector<unsigned char> pendingMemberIsLane;
    // a pair whose members do not fit one fetch goes in parts, consecutive pending pairs that pass only together
    std::vector<unsigned char> pendingPairIsLastPart;
    bool isPendingPairFailing = false;
    bool isPendingPairOpen = false;
    ClusterPair openPendingPair = { 0, 0 };
    // accepted pairs whose members wait for room in pendingMembers, behind the pending pairs in log order
    std::vector<ClusterPair> acceptedPairs;
    std::vector<size_t> acceptedEntryOffsets;
    std::vector<size_t> acceptedEntryBytes;
    std::vector<DBKeyType> lookupKeys;
    std::vector<size_t> lookupEntryOffsets;
    std::vector<size_t> lookupEntryBytes;
    std::vector<std::pair<DBKeyType, size_t> > lookupKeyPairs;
    std::vector<size_t> lookupMemberOffsets;
    std::vector<DBKeyType> lookupMembers;
    // the first accepted pair, read in slices when its members alone do not fit pendingMembers
    size_t slicedConsumedBytes = 0;
    bool isSlicingPair = false;
    size_t slicedRepSeqLen = 0;
    std::vector<size_t> slicedScalarPositions;
    size_t slicedMemberPosition = 0;
    size_t multiFetchPairCount = 0;
    size_t slicedPairCount = 0;
    size_t multiPartPairCount = 0;
    size_t unstartedMemberCount = 0;
    size_t fetchCount = 0;
    size_t fewerBatchFetchCount = 0;
    // coverable bytes of a rep in the blocks of this node
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const size_t rangeCount = seqDbr.getIndexSize();
    const std::vector<size_t> &blockKeyOffsets = seqDbr.getBlockKeyOffsets();
    std::vector<size_t> rangeByteOffsets(rangeCount + 1, 0);
    std::vector<size_t> nodeRangeByteOffsets(rangeCount + 1, 0);
    for (size_t range = 0, block = 0; range < rangeCount; range++) {
        const size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
        rangeByteOffsets[range + 1] = rangeByteOffsets[range] + (keyEnd - index[range].id) * index[range].length;
        nodeRangeByteOffsets[range + 1] = nodeRangeByteOffsets[range];
        for (size_t key = index[range].id; key < keyEnd; key = std::min(keyEnd, blockKeyOffsets[block + 1])) {
            for (; blockKeyOffsets[block + 1] <= key; block++) {}
            nodeRangeByteOffsets[range + 1] += (getBlockAlignNode(block) == rank) ? (std::min(keyEnd, blockKeyOffsets[block + 1]) - key) * index[range].length : 0;
        }
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
        coverableBytesSum += static_cast<double>(keyEnd - index[range].id) * (nodeRangeByteOffsets[coveredEnd] - nodeRangeByteOffsets[coveredFirst]);
    }
    double alignNodeCoverableSize = coverableBytesSum / std::max((size_t) 1, dbSize);
    const size_t averageEntrySize = std::max<size_t>(1, rangeByteOffsets[rangeCount] / std::max<size_t>(1, dbSize));
    const size_t maxServedKeys = std::max<size_t>(1, servedFileEntrySize / FETCHED_KEY_SIZE);
    const size_t maxServedFileEntrySize = servedFileEntrySize + SERVICE_POLL_NODES * (maxSeqLen + 2);
    const size_t maxServedFileSize = SERVICE_POLL_NODES * ((maxServedKeys + 7) / 8) + getPackedEntrySize(maxServedFileEntrySize) + maxServedFileEntrySize;
    // read directly once a cache miss costs less than two packed copies
    const auto isCacheTooSmall = [&](size_t cacheSize) {
        return alignNodeCount > 1 && static_cast<double>(cacheSize) * pageSize <= 2.0 * getPackedEntrySize(averageEntrySize) * alignNodeCoverableSize;
    };
    // memory besides the block cache
    const auto getReservedMemory = [&](size_t fetchSize) {
        return ((memberLookup != NULL) ? memberLookup->memorySize() + (par.threads + 1) * memberLookup->getEntryReaderSize() : 0) + 3 * 2 * (memoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR)
               + ((memberDbr != NULL) ? 2 * memberSplitSize : 0)
               + 2 * std::max(fetchSize, repTargetsReader.maxRecordSize) + fetchSize
               + ((isAligningNode && alignNodeCount > 1) ? 3 * maxServedFileSize + 6 * maxServedKeys * sizeof(size_t) : 0);
    };
    // largest fetch that leaves a block cache for the coverable sequences
    const bool canCache = isCacheTooSmall(memoryLimit - std::min(memoryLimit, getReservedMemory(0))) == false;
    size_t servedSequenceSize = nodeRangeByteOffsets[rangeCount];
    // a fetch holds a rep and a member of the largest length at least
    const size_t minFetchSize = 2 * (maxMemberSeqLen + 2 + FETCHED_KEY_SIZE);
    size_t maxFetchSize = std::max(memoryLimit / FETCH_MEMORY_DIVISOR, minFetchSize);
    while (maxFetchSize / 2 >= minFetchSize && (getReservedMemory(maxFetchSize) > memoryLimit || (canCache && isCacheTooSmall(memoryLimit - getReservedMemory(maxFetchSize)))
                                 || (alignNodeCount > 1 && maxFetchSize > repTargetsReader.maxRecordSize && memoryLimit - std::min(memoryLimit, getReservedMemory(maxFetchSize)) < servedSequenceSize))) {
        maxFetchSize /= 2;
    }
    if (getReservedMemory(maxFetchSize) > memoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to align. Memory limit: " << memoryLimit << " bytes, the buffers besides the block cache need: " << getReservedMemory(maxFetchSize)
                            << " bytes. Increase --split-memory-limit\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t blockCacheSize = memoryLimit - getReservedMemory(maxFetchSize);
    seqDbr.setBlockCacheSize(blockCacheSize);
    const size_t maxPendingPairSize = std::min(memberSplitSize, maxFetchSize);
    // a --local-tmp copy holds only the blocks this rank serves, the keys of other nodes come from them
    const bool isReadingDirectly = hasLocalServedBlocks == false && isCacheTooSmall(blockCacheSize);
    if (isAligningNode) {
        MMseqsNode::addPhaseValue(phaseValues, "Block cache", SSTR(blockCacheSize) + " bytes for " + SSTR(static_cast<size_t>(alignNodeCoverableSize)) + " coverable bytes and " + SSTR(servedSequenceSize) + " sequence bytes of the align node"
                                 + (isReadingDirectly ? ", the sequences are read directly" : ""));
    }
    size_t fetchBytesLimit = std::min(maxFetchSize / 64, FIRST_FETCH_SIZE);
    const auto getFetchedEntry = [&](const FetchedSequences &sequences, size_t key, size_t *seqLen = NULL, FetchedKeyCursor *cursor = NULL) -> const char * {
        const char *data = findFetchedKey(sequences, nodes, key, seqLen, (cursor != NULL) ? &cursor->positions[nodes] : NULL);
        if (data != NULL) {
            return data;
        }
        size_t fetchSource = isReadingDirectly ? rank : getKeyAlignNode(seqDbr, key, (cursor != NULL) ? &cursor->block : NULL);
        return findFetchedKey(sequences, fetchSource, key, seqLen, (cursor != NULL) ? &cursor->positions[fetchSource] : NULL);
    };
    const auto getFetchedSequence = [&](const FetchedSequences &sequences, size_t key, size_t *seqLen = NULL, FetchedKeyCursor *cursor = NULL) -> const char * {
        const char *data = getFetchedEntry(sequences, key, seqLen, cursor);
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
        // the accepted pairs follow the pending pairs in log order
        sequences.firstPendingRepKey = (sequences.pendingPairCount > 0) ? static_cast<size_t>(pendingPairs[previous.pendingPairCount].repKey)
                                       : (acceptedPairs.empty() ? orderKeyEnd : static_cast<size_t>(acceptedPairs[0].repKey));
        // the targets of previous and of this fetch can join the accepted pairs before any of those leave, so all of them fit memberKeyBudget;
        // a record goes alone once no pair waits
        size_t acceptablePairCount = acceptedPairs.size() + previous.targetCount;
        sequences.targetCount = 0;
        for (; sequences.end < prefilterRecords[prefilterBuffer].size(); sequences.end++) {
            const RepTargetsRecord &record = prefilterRecords[prefilterBuffer][sequences.end];
            fetchSize += record.readSize;
            if ((sequences.end > sequences.begin && fetchSize > fetchBytesLimit)
                || (memberDbr != NULL && acceptablePairCount > 0 && acceptablePairCount + record.targetCount > memberKeyBudget)) {
                break;
            }
            acceptablePairCount += record.targetCount;
            sequences.targetCount += record.targetCount;
        }
        prefilterRecordPos[prefilterBuffer] = sequences.end;
        // previous may still add pending pairs
        sequences.isLast = sequences.end == sequences.begin && sequences.pendingPairCount == 0 && (memberDbr == NULL || previous.end == previous.begin) && acceptedPairs.empty();
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
            FetchedKeyCursor keyCursor(nodes + 1);
#pragma omp for schedule(static)
            for (size_t i = 0; i < uniqueKeys.size(); i++) {
                keyFetchSources[i] = isAssigned(uniqueKeys[i]) ? nodes + 1 : ((getFetchedEntry(previous, uniqueKeys[i], NULL, &keyCursor) != NULL) ? nodes : (isReadingDirectly ? rank : getKeyAlignNode(seqDbr, uniqueKeys[i], &keyCursor.block)));
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
        // the keys of a fetch source ascend
#pragma omp parallel
        {
            size_t keyRange = SIZE_MAX;
#pragma omp for schedule(static)
            for (size_t i = 0; i < keyCount; i++) {
                sequences.entryOffsets[i + 1] = getSeqLenFromCursor(seqDbr, keyRange, sequences.keys[i]) + 2;
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
        // a fetch that the member budget left without records has records behind it
        const std::vector<RepTargetsRecord> &bufferRecords = prefilterRecords[sequences.prefilterBuffer];
        return (sequences.end < bufferRecords.size()) ? static_cast<size_t>(bufferRecords[sequences.end].repOrderKeyFirstTarget.repKey) : orderKeyEnd;
    };
    // the reps of the pending pairs and of the accepted pairs behind them are not done
    const auto getFirstUndoneRepKey = [&](size_t nextRepKey) {
        if (pendingPairs.empty() == false) {
            return static_cast<size_t>(pendingPairs[0].repKey);
        }
        return acceptedPairs.empty() ? nextRepKey : static_cast<size_t>(acceptedPairs[0].repKey);
    };

    FetchedSequences currentSequences;
    FetchedSequences nextSequences;
    nextSequences.fetchSourceKeyOffsets.assign(nodes + 2, 0);
    // the service thread and the loader use a quarter of the threads
    const int serviceThreads = std::max(1, par.threads / 4);
    std::atomic<double> alignedCellsPerSecond(0);
    std::atomic<double> masterWaitSeconds(0);
    const auto fillPrefilterBuffer = [&](size_t buffer, size_t maxBufferAlignmentCells) {
        if (targetSplitReader != NULL) {
            targetSplitReader->setWindowWork(alignedCellsPerSecond.load(), masterWaitSeconds.load());
            targetSplitReader->fillPrefilterBuffer(prefilterRecords[buffer], prefilterTargets[buffer], maxFetchSize, maxBufferAlignmentCells);
        } else {
            repTargetsReader.fillPrefilterBuffer(prefilterRecords[buffer], prefilterTargets[buffer], maxFetchSize, maxBufferAlignmentCells);
        }
    };
    fillPrefilterBuffer(0, 0);
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
            fillPrefilterBuffer(next % 3, static_cast<size_t>(alignedCellsPerSecond.load() * PREFILTER_BUFFER_WORK_SECONDS));
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
    // made by a thread when it first needs them and kept over the fetches; deleting them adds the SWIPE counters
    std::vector<TargetAligners *> threadTargetAligners(par.threads, NULL);
    std::vector<MemberAligners *> threadMemberAligners(par.threads, NULL);
    double waitSeconds = 0;
    double sequenceWaitSeconds = 0;
    double alignedCells = 0;
    double alignSeconds = 0;
    // wall time of the lookup blocks, not the sum of the thread times
    double memberLookupSeconds = 0;
    const auto mapRep = [&](Sequence &query, BlockAligner &blockAligner, size_t repKey) {
        size_t repLength = 0;
        const char *repData = getFetchedSequence(currentSequences, repKey, &repLength);
        query.mapSequence(repKey, repKey, repData, repLength);
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
    // wakes the service thread: a pending fetch or the loaded keys of this node
    bool hasServiceWork = false;
    // one load of seqDbr at a time
    std::mutex sourceMutex;
    std::vector<DBKeyType> loadKeys;
    std::vector<std::vector<size_t> > threadRawEntries(serviceThreads);
    size_t servedFileCount = 0;
    size_t servedSize = 0;
    size_t servedEntrySize = 0;
    size_t receivedServedEntriesSize = 0;
    size_t receivedEntrySize = 0;
    Timer serviceTimer;
    size_t requestCount = 0;
    size_t requestedEntrySize = 0;
    size_t progressReadCount = 0;
    size_t requestProgressReadCount = 0;
    size_t requestProgressChangeCount = 0;
    size_t requestListingCount = 0;
    size_t objectReadCount = 0;
    size_t objectWriteCount = 0;
    size_t deleteCount = 0;
    size_t liveServedSize = 0;
    size_t maxLiveServedSize = 0;
    double servingSeconds = 0;
    double servedFileWriterWaitSeconds = 0;
    double readingSeconds = 0;
    double pollingSeconds = 0;
    double sourceLoadSeconds = 0;
    double packSeconds = 0;
    double servedFileWriteSeconds = 0;
    double unpackSeconds = 0;
    // copy, or pack with rawKeys, the entries of keys; assigned keys get zeros
    const auto copyEntries = [&](const DBKeyType *keys, size_t keyCount, const size_t *offsets, char *data, std::vector<DBKeyType> *rawKeys) {
        size_t loadRange = SIZE_MAX;
        for (size_t first = 0; first < keyCount;) {
            const size_t loadFirst = first;
            // a load fits maxFetchSize
            for (size_t loadSize = 0; first < keyCount; first++) {
                const size_t entrySize = seqDbr.getSeqLenFromRange(loadRange, keys[first]) + 2 + 2 * pageSize;
                if (first > loadFirst && loadSize + entrySize > maxFetchSize) {
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
            Timer copyTimer;
            seqDbr.loadData(loadKeys);
            sourceLoadSeconds += copyTimer.getTimediff();
            copyTimer.reset();
            for (int thread = 0; thread < serviceThreads; thread++) {
                threadRawEntries[thread].clear();
            }
#pragma omp parallel num_threads(serviceThreads)
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                // the static chunks give a thread ascending keys
                size_t packRange = SIZE_MAX;
                DBReader<DBKeyType>::LoadedBufferCursor cursor;
#pragma omp for schedule(static)
                for (size_t i = loadFirst; i < first; i++) {
                    // bits are never cleared
                    if (isAssigned(keys[i])) {
                        memset(data + offsets[i], 0, offsets[i + 1] - offsets[i]);
                    } else if (rawKeys == NULL) {
                        memcpy(data + offsets[i], seqDbr.getDataFromLoadedBuffer(keys[i], cursor), offsets[i + 1] - offsets[i]);
                    } else if (packEntry(seqDbr.getDataFromLoadedBuffer(keys[i], cursor), seqDbr.getSeqLenFromRange(packRange, keys[i]) + 2, data + offsets[i]) == false) {
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
            packSeconds += copyTimer.getTimediff();
        }
    };
    std::thread serviceThread;
    if (isAligningNode) {
        serviceThread = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(1);
#endif
            // seconds between two reads of the progress file of a node
            double progressSeconds = std::max(BATCH_READ_SECONDS, 2.0 * (nodes - 1) / PROGRESS_READS_PER_SECOND);
            // own request: code offsets per node, files and keys received per node
            struct SequenceRequest {
                FetchedSequences *sequences;
                size_t number;
                size_t size;
                std::vector<size_t> codeOffsets;
                std::vector<size_t> receivedFiles;
                std::vector<size_t> receivedKeys;
                bool isComplete;
            };
            std::deque<SequenceRequest> ownRequests;
            size_t receivedRequestCount = 0;
            bool isLastRequested = false;
            FetchedSequences *requestedSequences = NULL;
            std::vector<char> requestCodes;
            // served file: request, file of the request, entry bytes, keys
            struct ServedFile {
                size_t request;
                size_t file;
                size_t size;
                size_t keyCount;
            };
            std::vector<double> nextServedProgressReads(nodes, 0);
            // seconds to the next read of a node that owes a request, doubled up to progressSeconds while it has not served
            std::vector<double> servedProgressSeconds(nodes, BATCH_READ_SECONDS);
            // the nodes that owe are read at half of PROGRESS_READS_PER_SECOND until fastReadEnd
            double fastReadSeconds = BATCH_READ_SECONDS;
            double fastReadEnd = 0;
            double nextRequestListing = 0;
            // the master progress while the gate is closed, about as often as before the faster polls
            double nextGateRead = 0;
            std::vector<std::string> requestNames;
            std::vector<std::vector<ServedFile> > servedFiles(nodes);
            // requesting node: progress, code range of this node per open request, codes served so far, served files not yet deleted
            struct Requester {
                Requester() : requestCount(0), receivedCount(0), isLast(false), isDone(false), nextProgressRead(0), servedCount(0), codePos(0), lastKey(0), file(0) {}
                size_t requestCount;
                size_t receivedCount;
                bool isLast;
                bool isDone;
                double nextProgressRead;
                std::vector<size_t> openRequests;
                size_t servedCount;
                size_t codePos;
                DBKeyType lastKey;
                size_t file;
                std::vector<ServedFile> files;
            };
            std::vector<Requester> requesters(nodes);
            for (unsigned int node = 0; node < nodes; node++) {
                requesters[node].isDone = node == rank || node < firstAlignNode;
            }
            std::vector<const char *> requestColumns(nodes + 3);
            bool isServedProgressChanged = false;
            std::vector<char> receivedEntries;
            std::vector<DBKeyType> servedKeys;
            std::vector<DBKeyType> rawKeys;
            std::vector<size_t> requestOffsets;
            std::vector<char> servedFileData;
            std::vector<char> writtenServedFileData;
            // a thread writes the served file while the next one is built
            std::thread servedFileWriter;
            unsigned int writtenRequester = UINT_MAX;
            ServedFile writtenFile = { 0, 0, 0, 0 };
            // a thread loads the keys of this node while the service thread serves and receives
            std::thread ownKeyLoader;
            bool isOwnKeysLoaded = false;
            std::vector<time_t> missingFileSince(nodes, 0);
            std::vector<unsigned int> missingFileAttempts(nodes, 0);
            std::string requestProgress = requests + "." + SSTR(rank) + ".progress";
            std::string servedProgress = served + "." + SSTR(rank) + ".progress";
            const auto getServedFile = [&](unsigned int alignNode, unsigned int requester, const ServedFile &file) {
                return served + "." + SSTR(alignNode) + "." + SSTR(requester) + "." + SSTR(file.request) + "." + SSTR(file.file);
            };
            const auto finishServedFileWrite = [&]() {
                if (servedFileWriter.joinable()) {
                    Timer waitTimer;
                    servedFileWriter.join();
                    servedFileWriterWaitSeconds += waitTimer.getTimediff();
                }
                if (writtenRequester != UINT_MAX) {
                    requesters[writtenRequester].files.push_back(writtenFile);
                    liveServedSize += writtenFile.size;
                    maxLiveServedSize = std::max(maxLiveServedSize, liveServedSize);
                    writtenRequester = UINT_MAX;
                    isServedProgressChanged = true;
                }
            };
            MMseqsNode::WaitWarning waitWarning;
            time_t assignedKeysReadTime = 0;
            for (unsigned int polls = 0;;) {
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
                // request once the master is close enough behind
                FetchedSequences *nextRequest = NULL;
                size_t nextAcceptedSize = 0;
                bool isLoaded;
                {
                    std::lock_guard<std::mutex> lock(serviceMutex);
                    hasServiceWork = false;
                    isLoaded = isOwnKeysLoaded;
                    if (requestedSequences == NULL) {
                        nextRequest = pendingSequences;
                        nextAcceptedSize = requestAcceptedSize;
                    }
                }
                if (nextRequest != NULL) {
                    bool isLast = nextRequest->isLast;
                    // a fetch without records and pending pairs logs nothing, so it never waits; the pairs the aligning fetch adds come from it
                    bool canWait = isLast == false && (nextRequest->end > nextRequest->begin || nextRequest->pendingPairCount > 0);
                    bool isAhead;
                    {
                        std::lock_guard<std::mutex> assignedKeysLock(assignedKeysMutex);
                        size_t repKey = std::min(nextRequest->firstPendingRepKey, getFirstRepKey(*nextRequest));
                        if (canWait && repKey > clusteredOrderKey && nextAcceptedSize >= clusteredAcceptedSize + maxUnclusteredAcceptedSize && serviceTimer.getTimediff() >= nextGateRead) {
                            nextGateRead = serviceTimer.getTimediff() + BATCH_READ_SECONDS / 2;
                            readAssignedKeys(true);
                        }
                        isAhead = canWait && repKey > clusteredOrderKey && nextAcceptedSize >= clusteredAcceptedSize + maxUnclusteredAcceptedSize;
                    }
                    {
                        std::lock_guard<std::mutex> lock(serviceMutex);
                        isWaitingForMaster = isAhead;
                        if (isAhead == false) {
                            pendingSequences = NULL;
                        }
                    }
                    if (isAhead == false) {
                        // the keys of a node ascend, so a request holds their gaps
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
                                requestedEntrySize += nextRequest->entryOffsets[i + 1] - nextRequest->entryOffsets[i];
                            }
                        }
                        request.codeOffsets[nodes] = requestCodes.size();
                        // a fetch without keys of other nodes has no request
                        if (requestCodes.empty() == false) {
                            MMseqsNode::writeExchangeFile(requests + "/" + SSTR(rank) + "." + SSTR(request.number), request.number, requestCodes.data(), requestCodes.size(), "", std::vector<uint64_t>(), 1);
                            request.size = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + requestCodes.size();
                            request.receivedFiles.assign(nodes, 0);
                            request.receivedKeys.assign(nodes, 0);
                            request.isComplete = false;
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
                            objectWriteCount++;
                            isRequestProgressChanged = true;
                        }
                        isRequestProgressChanged |= isLast;
                        isLastRequested = isLast;
                        requestedSequences = nextRequest;
                        hasWorked = true;
                    }
                }
                // receive from the nodes that owe a request
                for (unsigned int node = 0; node < nodes; node++) {
                    bool isOwed = false;
                    for (size_t i = 0; i < ownRequests.size() && isOwed == false; i++) {
                        isOwed = node != rank && ownRequests[i].receivedKeys[node] < ownRequests[i].sequences->fetchSourceKeyOffsets[node + 1] - ownRequests[i].sequences->fetchSourceKeyOffsets[node];
                    }
                    if (isOwed == false || serviceTimer.getTimediff() < nextServedProgressReads[node]) {
                        continue;
                    }
                    Timer pollTimer;
                    std::string content;
                    nextServedProgressReads[node] = serviceTimer.getTimediff() + servedProgressSeconds[node];
                    if (serviceTimer.getTimediff() >= fastReadEnd) {
                        servedProgressSeconds[node] = std::min(2.0 * servedProgressSeconds[node], progressSeconds);
                    }
                    progressReadCount++;
                    bool hasProgress = MMseqsNode::readRunFile(served + "." + SSTR(node) + ".progress", content);
                    pollingSeconds += pollTimer.getTimediff();
                    if (hasProgress) {
                        servedFiles[node].clear();
                        for (char *line = &content[0]; *line != '\0'; line = Util::skipLine(line)) {
                            const char *columns[5];
                            if (Util::getWordsOfLine(line, columns, 5) == 5 && Util::fast_atoi<size_t>(columns[0]) == rank) {
                                ServedFile file = { Util::fast_atoi<size_t>(columns[1]), Util::fast_atoi<size_t>(columns[2]), Util::fast_atoi<size_t>(columns[3]), Util::fast_atoi<size_t>(columns[4]) };
                                servedFiles[node].push_back(file);
                            }
                        }
                    }
                    for (size_t i = 0; i < ownRequests.size(); i++) {
                        SequenceRequest &request = ownRequests[i];
                        for (size_t j = 0; j < servedFiles[node].size(); j++) {
                            const ServedFile &file = servedFiles[node][j];
                            if (file.request != request.number || file.file != request.receivedFiles[node]) {
                                continue;
                            }
                            // the next served file whole, a failed read is tried again after the next progress read
                            std::string fileName = getServedFile(node, rank, file);
                            size_t observedSize;
                            uint64_t observedNumber;
                            int lastError;
                            Timer readTimer;
                            objectReadCount++;
                            int fd = MMseqsNode::tryOpenExchangeFile(fileName, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + file.size, file.request, observedSize, observedNumber, lastError);
                            if (fd < 0) {
                                readingSeconds += readTimer.getTimediff();
                                if (lastError != 0 && lastError != ENOENT && lastError != ESTALE) {
                                    Debug(Debug::ERROR) << "Cannot open served sequence file " << fileName << ": " << strerror(lastError) << "\n";
                                    EXIT(EXIT_FAILURE);
                                }
                                failIfMissingTooLong(missingFileSince[node], missingFileAttempts[node], "Cannot open complete served sequence file " + fileName
                                                     + describeServedFile(MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + file.size, observedSize, file.request, observedNumber, lastError));
                                break;
                            }
                            if (receivedEntries.size() < file.size) {
                                receivedEntries.resize(file.size);
                            }
                            // the parts keep their errors
                            size_t partCount = (file.size + BatchReader::MAX_READ_REQUEST_SIZE - 1) / BatchReader::MAX_READ_REQUEST_SIZE;
                            std::vector<int> readErrors(partCount, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(serviceThreads)
                            for (size_t part = 0; part < partCount; part++) {
                                size_t offset = part * BatchReader::MAX_READ_REQUEST_SIZE;
                                int partError;
                                if (MMseqsNode::tryPreadAll(fd, receivedEntries.data() + offset, std::min(BatchReader::MAX_READ_REQUEST_SIZE, file.size - offset), MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + offset, partError) == false) {
                                    readErrors[part] = (partError != 0) ? partError : -1;
                                }
                            }
                            int readError = 0;
                            for (size_t part = 0; part < partCount; part++) {
                                readError = (readErrors[part] != 0) ? readErrors[part] : readError;
                            }
                            int closeError = (close(fd) != 0) ? errno : 0;
                            readingSeconds += readTimer.getTimediff();
                            int error = (readError > 0) ? readError : closeError;
                            if (error != 0 && error != ESTALE && error != EINTR) {
                                Debug(Debug::ERROR) << "Cannot read served sequence file " << fileName << ": " << strerror(error) << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                            if (readError != 0 || closeError != 0) {
                                failIfMissingTooLong(missingFileSince[node], missingFileAttempts[node], "Cannot read complete served sequence file " + fileName + " (error " + SSTR(error) + ")\n");
                                break;
                            }
                            missingFileAttempts[node] = 0;
                            FetchedSequences *sequences = request.sequences;
                            size_t firstKey = sequences->fetchSourceKeyOffsets[node] + request.receivedKeys[node];
                            Timer unpackTimer;
                            unpackEntries(receivedEntries.data(), file.keyCount, sequences->entryOffsets.data() + firstKey, sequences->entries, serviceThreads);
                            unpackSeconds += unpackTimer.getTimediff();
                            for (size_t key = firstKey; key < firstKey + file.keyCount; key++) {
                                receivedEntrySize += (sequences->entries[sequences->entryOffsets[key]] != '\0') ? sequences->entryOffsets[key + 1] - sequences->entryOffsets[key] : 0;
                            }
                            request.receivedKeys[node] += file.keyCount;
                            request.receivedFiles[node]++;
                            receivedServedEntriesSize += file.size;
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
                        FileUtil::remove((requests + "/" + SSTR(rank) + "." + SSTR(request.number)).c_str());
                        deleteCount++;
                        hasWorked = true;
                    }
                }
                for (; ownRequests.empty() == false && ownRequests.front().isComplete; receivedRequestCount++) {
                    ownRequests.pop_front();
                    isRequestProgressChanged = true;
                }
                // the requests not received yet with the code offsets of all nodes
                if (isRequestProgressChanged && alignNodeCount > 1) {
                    std::string content = SSTR(requestCount) + "\t" + SSTR(receivedRequestCount) + "\t" + SSTR(isLastRequested ? 1 : 0) + "\n";
                    for (size_t i = 0; i < ownRequests.size(); i++) {
                        content += SSTR(ownRequests[i].number) + "\t" + SSTR(ownRequests[i].size);
                        for (unsigned int node = 0; node <= nodes; node++) {
                            content += "\t" + SSTR(ownRequests[i].codeOffsets[node]);
                        }
                        content += "\n";
                    }
                    MMseqsNode::publishRunFile(requestProgress, content);
                    objectWriteCount++;
                }
                // the keys of this node while the other nodes serve the new request
                if (nextRequest != NULL && requestedSequences == nextRequest) {
                    isLoaded = false;
                    isOwnKeysLoaded = false;
                    ownKeyLoader = std::thread([&, nextRequest]() {
#ifdef OPENMP
                        omp_set_num_threads(1);
#endif
                        size_t ownKey = nextRequest->fetchSourceKeyOffsets[rank];
                        copyEntries(nextRequest->keys.data() + ownKey, nextRequest->fetchSourceKeyOffsets[rank + 1] - ownKey, nextRequest->entryOffsets.data() + ownKey, nextRequest->entries, NULL);
                        {
                            std::lock_guard<std::mutex> lock(serviceMutex);
                            isOwnKeysLoaded = true;
                            hasServiceWork = true;
                        }
                        serviceCondition.notify_all();
                    });
                }
                // a fetch goes to the aligning threads once its request is received
                if (requestedSequences != NULL && ownRequests.empty() && isLoaded) {
                    ownKeyLoader.join();
                    {
                        std::lock_guard<std::mutex> lock(serviceMutex);
                        completedSequences = requestedSequences;
                    }
                    serviceCondition.notify_all();
                    requestedSequences = NULL;
                    hasWorked = true;
                }
                // the progress of a node is read at once when the names hold a request it did not show yet
                if (serviceTimer.getTimediff() >= nextRequestListing) {
                    nextRequestListing = serviceTimer.getTimediff() + std::max(BATCH_READ_SECONDS, (nodes - 1.0) / PROGRESS_READS_PER_SECOND) / 4.0;
                    MMseqsNode::listDirectory(requests, requestNames);
                    requestListingCount++;
                    for (size_t i = 0; i < requestNames.size(); i++) {
                        const char *name = requestNames[i].c_str();
                        const char *number = strchr(name, '.');
                        size_t node = Util::fast_atoi<size_t>(name);
                        if (number != NULL && number != name && node < nodes && Util::fast_atoi<size_t>(number + 1) >= requesters[node].requestCount) {
                            requesters[node].nextProgressRead = 0;
                        }
                    }
                }
                // a served file a pass for each requesting node
                bool isServing = false;
                for (unsigned int i = 1; i < nodes; i++) {
                    unsigned int node = (rank + i) % nodes;
                    Requester &requester = requesters[node];
                    if (requester.isDone) {
                        continue;
                    }
                    if (serviceTimer.getTimediff() >= requester.nextProgressRead) {
                        Timer pollTimer;
                        std::string content;
                        requester.nextProgressRead = serviceTimer.getTimediff() + progressSeconds;
                        progressReadCount++;
                        requestProgressReadCount++;
                        bool hasProgress = MMseqsNode::readRunFile(requests + "." + SSTR(node) + ".progress", content);
                        pollingSeconds += pollTimer.getTimediff();
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
                            requestProgressChangeCount += count != requester.requestCount || received != requester.receivedCount || isLast != requester.isLast;
                            requester.requestCount = count;
                            requester.receivedCount = received;
                            requester.isLast = isLast;
                            // a received request was served, also one without keys of this node
                            requester.servedCount = std::max(requester.servedCount, received);
                            // per open request: number, size and the codes of this node
                            requester.openRequests.clear();
                            for (line = Util::skipLine(line); *line != '\0'; line = Util::skipLine(line)) {
                                Util::getWordsOfLine(line, requestColumns.data(), nodes + 3);
                                requester.openRequests.push_back(Util::fast_atoi<size_t>(requestColumns[0]));
                                requester.openRequests.push_back(Util::fast_atoi<size_t>(requestColumns[1]));
                                requester.openRequests.push_back(Util::fast_atoi<size_t>(requestColumns[2 + rank]));
                                requester.openRequests.push_back(Util::fast_atoi<size_t>(requestColumns[3 + rank]));
                            }
                            // the files of the received requests go
                            size_t keptFiles = 0;
                            for (size_t j = 0; j < requester.files.size(); j++) {
                                if (requester.files[j].request < received) {
                                    FileUtil::remove(getServedFile(rank, node, requester.files[j]).c_str());
                                    deleteCount++;
                                    liveServedSize -= requester.files[j].size;
                                    isServedProgressChanged = true;
                                } else {
                                    requester.files[keptFiles++] = requester.files[j];
                                }
                            }
                            requester.files.resize(keptFiles);
                            requester.isDone = isLast && received == count;
                        }
                    }
                    // one served file of the next open request
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
                        Timer servingTimer;
                        // the codes of the next keys, a request is read in pieces
                        std::string requestFile = requests + "/" + SSTR(node) + "." + SSTR(requester.servedCount);
                        size_t observedSize;
                        uint64_t observedNumber;
                        int lastError;
                        objectReadCount++;
                        int requestFd = MMseqsNode::tryOpenExchangeFile(requestFile, requestSize, requester.servedCount, observedSize, observedNumber, lastError);
                        if (requestFd < 0 && lastError != 0 && lastError != ENOENT && lastError != ESTALE) {
                            Debug(Debug::ERROR) << "Cannot open sequence request " << requestFile << ": " << strerror(lastError) << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        requestCodes.resize(std::min(codeEnd - codeBegin - requester.codePos, maxServedKeys * sizeof(DBKeyType)));
                        bool isRequestRead = requestFd >= 0 && MMseqsNode::tryPreadAll(requestFd, requestCodes.data(), requestCodes.size(), MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + codeBegin + requester.codePos, lastError);
                        if (requestFd >= 0 && close(requestFd) != 0 && isRequestRead) {
                            lastError = errno;
                            isRequestRead = false;
                        }
                        // a missing or short request is tried again in the next pass
                        if (isRequestRead == false) {
                            servingSeconds += servingTimer.getTimediff();
                            if (lastError != 0 && lastError != ENOENT && lastError != ESTALE && lastError != EINTR) {
                                Debug(Debug::ERROR) << "Cannot read sequence request " << requestFile << ": " << strerror(lastError) << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                            failIfMissingTooLong(missingFileSince[node], missingFileAttempts[node], "Cannot read complete sequence request " + requestFile
                                                 + describeServedFile(requestSize, observedSize, requester.servedCount, observedNumber, lastError));
                            break;
                        }
                        missingFileAttempts[node] = 0;
                        // only the last byte of a code is below 0x80
                        const char *codesEnd = requestCodes.data() + requestCodes.size();
                        while (codesEnd > requestCodes.data() && static_cast<unsigned char>(codesEnd[-1]) >= 0x80) {
                            codesEnd--;
                        }
                        // the unassigned entries of at least one key, up to servedFileEntrySize bytes and maxServedKeys keys
                        servedKeys.clear();
                        const char *code = requestCodes.data();
                        size_t keyRange = SIZE_MAX;
                        for (size_t entrySize = 0; servedKeys.size() < maxServedKeys && code < codesEnd;) {
                            const char *nextCode = code;
                            DBKeyType key = requester.lastKey + Util::getPrefKey(nextCode);
                            size_t keySize = isAssigned(key) ? 0 : seqDbr.getSeqLenFromRange(keyRange, key) + 2;
                            if (servedKeys.empty() == false && entrySize + keySize > servedFileEntrySize) {
                                break;
                            }
                            servedKeys.push_back(key);
                            requester.lastKey = key;
                            code = nextCode;
                            entrySize += keySize;
                        }
                        requester.codePos += code - requestCodes.data();
                        const DBKeyType *keys = servedKeys.data();
                        size_t keyCount = servedKeys.size();
                        requestOffsets.resize(keyCount + 1);
                        requestOffsets[0] = (keyCount + 7) / 8;
#pragma omp parallel num_threads(serviceThreads)
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
                        servedFileData.resize(requestOffsets[keyCount]);
                        // an empty slot marks an assigned key
#pragma omp parallel for schedule(static) num_threads(serviceThreads)
                        for (size_t byte = 0; byte < requestOffsets[0]; byte++) {
                            unsigned char bits = 0;
                            for (size_t j = 8 * byte; j < std::min(keyCount, 8 * byte + 8); j++) {
                                bits |= static_cast<unsigned char>(requestOffsets[j + 1] == requestOffsets[j]) << (j % 8);
                            }
                            servedFileData[byte] = static_cast<char>(bits);
                        }
                        rawKeys.clear();
                        copyEntries(keys, keyCount, requestOffsets.data(), servedFileData.data(), &rawKeys);
                        size_t servedFileSize = requestOffsets[keyCount];
                        requestOffsets.resize(rawKeys.size() + 1);
                        requestOffsets[0] = 0;
                        size_t rawKeyRange = SIZE_MAX;
                        for (size_t j = 0; j < rawKeys.size(); j++) {
                            requestOffsets[j + 1] = requestOffsets[j] + seqDbr.getSeqLenFromRange(rawKeyRange, rawKeys[j]) + 2;
                        }
                        servedFileData.resize(servedFileSize + requestOffsets[rawKeys.size()]);
                        copyEntries(rawKeys.data(), rawKeys.size(), requestOffsets.data(), servedFileData.data() + servedFileSize, NULL);
                        servedFileSize += requestOffsets[rawKeys.size()];
                        size_t entryRange = SIZE_MAX;
                        for (size_t j = 0; j < keyCount; j++) {
                            servedEntrySize += isAssigned(keys[j]) ? 0 : seqDbr.getSeqLenFromRange(entryRange, keys[j]) + 2;
                        }
                        servingSeconds += servingTimer.getTimediff();
                        finishServedFileWrite();
                        servedFileData.swap(writtenServedFileData);
                        writtenFile.request = requester.servedCount;
                        writtenFile.file = requester.file;
                        writtenFile.size = servedFileSize;
                        writtenFile.keyCount = keyCount;
                        writtenRequester = node;
                        std::string servedFile = getServedFile(rank, node, writtenFile);
                        uint64_t servedRequest = writtenFile.request;
                        servedFileWriter = std::thread([&, servedFile, servedRequest, servedFileSize]() {
                            Timer writeTimer;
                            MMseqsNode::writeExchangeFile(servedFile, servedRequest, writtenServedFileData.data(), servedFileSize, "", std::vector<uint64_t>(), serviceThreads);
                            servedFileWriteSeconds += writeTimer.getTimediff();
                        });
                        servedSize += servedFileSize;
                        servedFileCount++;
                        objectWriteCount++;
                        requester.file++;
                        if (requester.codePos == codeEnd - codeBegin) {
                            requester.codePos = 0;
                            requester.lastKey = 0;
                            requester.file = 0;
                            requester.servedCount++;
                        }
                        hasWorked = true;
                        break;
                    }
                    isServing |= requester.isDone == false;
                }
                finishServedFileWrite();
                // the served files not yet deleted, per requester
                if (isServedProgressChanged) {
                    std::string content;
                    for (unsigned int node = 0; node < nodes; node++) {
                        for (size_t i = 0; i < requesters[node].files.size(); i++) {
                            const ServedFile &file = requesters[node].files[i];
                            content += SSTR(node) + "\t" + SSTR(file.request) + "\t" + SSTR(file.file) + "\t" + SSTR(file.size) + "\t" + SSTR(file.keyCount) + "\n";
                        }
                    }
                    MMseqsNode::publishRunFile(servedProgress, content);
                    objectWriteCount++;
                    isServedProgressChanged = false;
                }
                if (isLastRequested && ownRequests.empty() && requestedSequences == NULL && isServing == false) {
                    break;
                }
                // sleep longer while no node waits
                if (hasWorked) {
                    polls = 0;
                } else {
                    unsigned int maxShift = (requestedSequences != NULL || nextRequest != NULL) ? 4 : 7;
                    maxShift = (alignNodeCount > 1) ? 1 : maxShift;
                    std::unique_lock<std::mutex> lock(serviceMutex);
                    serviceCondition.wait_for(lock, std::chrono::microseconds(10000u << std::min(polls++, maxShift)), [&]() { return hasServiceWork; });
                }
                // aligning without a request is no wait
                if (hasWorked || (requestedSequences == NULL && nextRequest == NULL && isLastRequested == false)) {
                    waitWarning.reset();
                } else {
                    waitWarning.check(requestedSequences != NULL ? served + " of other nodes" : (nextRequest != NULL ? assignedKeysLog : requests + " of other nodes"));
                }
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
            hasServiceWork = true;
            requestAcceptedSize = acceptedSize;
        }
        serviceCondition.notify_all();
        waitForSequences(currentSequences, getFirstRepKey(currentSequences));
    }
    const double firstFetchSeconds = startTimer.getTimediff();
    while (currentSequences.isLast == false) {
        selectRecordKeys(nextSequences, currentSequences);
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            completedSequences = NULL;
            pendingSequences = &nextSequences;
            hasServiceWork = true;
            requestAcceptedSize = acceptedSize;
        }
        serviceCondition.notify_all();
        Timer alignTimer;
        const size_t nextRepKey = getFirstRepKey(nextSequences);
        const std::vector<RepTargetsRecord> &records = prefilterRecords[currentSequences.prefilterBuffer];
        const size_t recordCount = currentSequences.end - currentSequences.begin;
        // batches of about equal alignment cells; a large rep spans several
        size_t maxBatches = par.threads * (isSwipeEnabled ? ALIGN_BATCHES_PER_THREAD * ALIGN_BATCH_LANES / SwipeExtender::LANES : ALIGN_BATCHES_PER_THREAD);
        if (isSwipeEnabled) {
            // estimate the SWIPE jobs of this fetch
            size_t swipeJobCount = 0;
            for (int thread = 0; thread < par.threads; thread++) {
                swipeJobCount += (threadTargetAligners[thread] != NULL) ? threadTargetAligners[thread]->swipeTargets.targetPairs : 0;
            }
            size_t fetchSwipeJobs = static_cast<size_t>(static_cast<uint128>(currentSequences.targetCount) * swipeJobCount / std::max((size_t) 1, examinedTargetPairCount.load()));
            size_t swipeBatches = std::max((size_t) 8 * par.threads, fetchSwipeJobs / (64 * SwipeExtender::LANES));
            fetchCount++;
            fewerBatchFetchCount += swipeBatches < maxBatches;
            maxBatches = std::min(maxBatches, swipeBatches);
        }
        recordAlignmentCells.resize(recordCount);
        // the reps of a size class ascend
#pragma omp parallel
        {
            size_t repRange = SIZE_MAX;
#pragma omp for schedule(static)
            for (size_t i = 0; i < recordCount; i++) {
                const RepTargetsRecord &record = records[currentSequences.begin + i];
                size_t repLength = getSeqLenFromCursor(seqDbr, repRange, record.repOrderKeyFirstTarget.repKey % paddedDbSize);
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
            const uint128 recordEndCells = cells + recordAlignmentCells[i - currentSequences.begin];
            if (recordEndCells * maxBatches > fetchAlignmentCells * batchOffsets.size()) {
                const RepTargetsRecord &record = records[i];
                const DBKeyType *targets = prefilterTargets[currentSequences.prefilterBuffer].data() + record.targetOffset;
                const size_t repLength = getSeqLenFromCursor(seqDbr, repRange, record.repOrderKeyFirstTarget.repKey % paddedDbSize);
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
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            FetchedKeyCursor targetCursor(nodes + 1);
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
                TargetAligners *aligners = threadTargetAligners[thread_idx];
                size_t examinedCount = 0;
                size_t repSkippedCount = 0;
                size_t repSlotSkippedCount = 0;
                size_t targetSkippedCount = 0;
                size_t uncoverableCount = 0;
                for (size_t i = batchOffsets[batch]; i < batchOffsets[batch + 1] + (batchFirstTargets[batch + 1] > 0); i++) {
                    const RepTargetsRecord &record = records[i];
                    unsigned int firstTarget = (i == batchOffsets[batch]) ? batchFirstTargets[batch] : 0;
                    const unsigned int endTarget = (i == batchOffsets[batch + 1]) ? batchFirstTargets[batch + 1] : record.targetCount;
                    examinedCount += endTarget - firstTarget;
                    const size_t repKey = record.repOrderKeyFirstTarget.repKey % paddedDbSize;
                    if (isAssigned(record.repOrderKeyFirstTarget.repKey)) {
                        repSkippedCount += endTarget - firstTarget;
                        continue;
                    }
                    size_t repLength = 0;
                    const char *repData = getFetchedSequence(currentSequences, repKey, &repLength);
                    if (repData == NULL) {
                        repSkippedCount += endTarget - firstTarget;
                        repSlotSkippedCount += endTarget - firstTarget;
                        continue;
                    }
                    bool isRepMapped = false;
                    bool blockAlignerInitialized = false;
                    const DBKeyType *targets = prefilterTargets[currentSequences.prefilterBuffer].data() + record.targetOffset;
                    unsigned int j = firstTarget;
                    for (; j < endTarget && isAssigned(record.repOrderKeyFirstTarget.repKey) == false; j++) {
                        const DBKeyType targetKey = targets[j];
                        if (isAssigned(targetKey)) {
                            targetSkippedCount++;
                            continue;
                        }
                        size_t targetLength = 0;
                        const char *targetData = getFetchedSequence(currentSequences, targetKey, &targetLength, &targetCursor);
                        if (targetData == NULL) {
                            targetSkippedCount++;
                            continue;
                        }
                        // alignTarget checks the coverage first
                        if (Util::canBeCovered(par.covThr, par.covMode, static_cast<float>(repLength), static_cast<float>(targetLength)) == false) {
                            uncoverableCount++;
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
                    repSkippedCount += endTarget - j;
                }
                examinedTargetPairCount += examinedCount;
                repSkippedTargetPairCount += repSkippedCount;
                repSlotSkippedTargetPairCount += repSlotSkippedCount;
                targetSkippedTargetPairCount += targetSkippedCount;
                uncoverableTargetPairCount += uncoverableCount;
                alignedTargetPairCount += examinedCount - repSkippedCount - targetSkippedCount - uncoverableCount;
                if (aligners != NULL) {
                    alignTargetsSwipe(par, aligners->swipeTargets, isAssigned, pairs);
                    aligners->swipeTargets.clearTargets();
                }
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
            // the aligners stay over the fetches, the buffers of their batches do not
            if (threadTargetAligners[thread_idx] != NULL) {
                threadTargetAligners[thread_idx]->swipeTargets.release();
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
            // size the entries of the new pairs during the member checks
            const size_t sizedCount = acceptedEntryBytes.size();
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
            // a pair that already failed does not align its remaining parts
            size_t failedPartEnd = 0;
            if (isPendingPairFailing) {
                while (failedPartEnd < pairCount && pendingPairIsLastPart[failedPartEnd] == 0) {
                    failedPartEnd++;
                }
                failedPartEnd = std::min(pairCount, failedPartEnd + 1);
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
#pragma omp parallel reduction(+:multiPartPairCount, unstartedMemberCount)
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
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
                    alignMemberChecks(aligners, thread_idx, repMemberCheckOffsets, repKeys);
                    for (size_t g = 0; g < aligners.memberChecks.checkCount; g++) {
                        const MemberCheck &check = aligners.memberChecks.checks[g];
                        size_t partEnd = getPairPartEnd(pendingPairIsLastPart, pairCount, check.targetId);
                        for (size_t i = check.targetId; i < partEnd; i++) {
                            membersPassed[i] = check.isPassing;
                        }
                        if (partEnd - check.targetId > 1) {
                            multiPartPairCount++;
                            unstartedMemberCount += check.memberKeys.size() - check.nextMember;
                        }
                    }
                }
                if (threadMemberAligners[thread_idx] != NULL) {
                    threadMemberAligners[thread_idx]->memberChecks.release();
                    threadMemberAligners[thread_idx]->swipeMembers.release();
                    threadMemberAligners[thread_idx]->memberTargets.release();
                }
            }
            // a pair passes with its last part if none of its parts failed; the parts of a pair can span fetches
            size_t acceptedCount = 0;
            for (size_t i = 0; i < pairCount; i++) {
                isPendingPairFailing |= membersPassed[i] == 0;
                if (pendingPairIsLastPart[i]) {
                    if (isPendingPairFailing == false) {
                        pendingPairs[acceptedCount++] = pendingPairs[i];
                    }
                    isPendingPairFailing = false;
                    multiFetchPairCount += isPendingPairOpen;
                    isPendingPairOpen = false;
                }
            }
            if (pairCount > 0) {
                isPendingPairOpen = pendingPairIsLastPart[pairCount - 1] == 0;
                // the accepted pairs above are written before an open last part only
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

            // the accepted pairs wait for a later fetch, behind those that did not fit yet
            Timer memberLookupTimer;
            memberExchange->receive();
            memberExchange->getEntries(lookupEntryOffsets, lookupEntryBytes);
            acceptedEntryOffsets.resize(acceptedPairs.size(), 0);
            acceptedEntryBytes.resize(acceptedPairs.size(), 0);
            for (size_t i = sizedCount; i < acceptedPairs.size(); i++) {
                const size_t lookupKey = std::lower_bound(lookupKeys.begin(), lookupKeys.end(), acceptedPairs[i].memberKey) - lookupKeys.begin();
                if (lookupKey < lookupKeys.size() && lookupKeys[lookupKey] == acceptedPairs[i].memberKey) {
                    acceptedEntryOffsets[i] = lookupEntryOffsets[lookupKey];
                    acceptedEntryBytes[i] = lookupEntryBytes[lookupKey];
                }
            }
            // the pending members and parts take memberKeyBudget; a pair of e entry bytes has less than e / 2 members, as a line has two bytes at least,
            // and no more parts than members, but one part at least
            const size_t freeKeys = memberKeyBudget - std::min(memberKeyBudget, pendingMembers.size() + pendingPairs.size());
            size_t plannedCount = 0;
            for (size_t keyBound = 0; isSlicingPair == false && plannedCount < acceptedPairs.size(); plannedCount++) {
                const size_t pairBound = std::max<size_t>(1, acceptedEntryBytes[plannedCount]);
                if (keyBound + pairBound > freeKeys) {
                    break;
                }
                keyBound += pairBound;
            }
            pairPartCounts.resize(plannedCount);
            needsMemberCheck.resize(plannedCount);
            lookupKeyPairs.clear();
            for (size_t i = 0; i < plannedCount; i++) {
                // an assigned key or an uncoverable member fails without a check
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
            memberExchange->receive();
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
                    const size_t memberBegin = threadKeys[thread_idx].size();
                    const size_t partBegin = threadPartSizes[thread_idx].size();
                    const size_t repKey = acceptedPairs[i].repKey % paddedDbSize;
                    const size_t repSeqLen = seqDbr.getSeqLen(repKey);
                    const size_t repSize = repSeqLen + 2 + FETCHED_KEY_SIZE;
                    size_t member = 0;
                    size_t memberEnd = 0;
                    if (needsMemberCheck[i]) {
                        const size_t lookupKey = std::lower_bound(lookupKeys.begin(), lookupKeys.end(), acceptedPairs[i].memberKey) - lookupKeys.begin();
                        member = lookupMemberOffsets[lookupKey];
                        memberEnd = lookupMemberOffsets[lookupKey + 1];
                    }
                    // the members of a cluster entry ascend, so the range cursor only moves forward inside it
                    size_t memberRange = SIZE_MAX;
                    size_t partSize = repSize;
                    size_t partMembers = 0;
                    scalarMembers.clear();
                    while (needsMemberCheck[i] && member < memberEnd) {
                        const DBKeyType memberKey = lookupMembers[member++];
                        const size_t memberSeqLen = seqDbr.getSeqLenFromRange(memberRange, memberKey);
                        needsMemberCheck[i] = canCoverMember(par, static_cast<int>(repSeqLen), static_cast<int>(memberSeqLen));
                        if (needsMemberCheck[i] == false) {
                            break;
                        }
                        const size_t memberSize = memberSeqLen + 2 + FETCHED_KEY_SIZE;
                        // a part takes one member at least, and a fetch takes one pending pair at least
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
            // a pair whose members alone exceed the budget goes alone, as many members per fetch as are free
            if (isSlicingPair == false && plannedCount == 0 && acceptedPairs.empty() == false && acceptedEntryBytes[0] > memberKeyBudget) {
                isSlicingPair = true;
                slicedPairCount++;
                slicedConsumedBytes = 0;
                slicedMemberPosition = 0;
                slicedRepSeqLen = seqDbr.getSeqLen(acceptedPairs[0].repKey % paddedDbSize);
                // a first read of the members finds those outside the lanes; the slices fail an assigned pair or one with an uncoverable member anyway
                ScalarMembers scalarMembers;
                bool needsCheck = isAssigned(acceptedPairs[0].repKey) == false && isAssigned(acceptedPairs[0].memberKey) == false;
                lookupKeys.assign(1, acceptedPairs[0].memberKey);
                for (size_t consumedBytes = 0; needsCheck && consumedBytes < acceptedEntryBytes[0];) {
                    memberExchange->request(lookupKeys, &acceptedEntryOffsets[0], &acceptedEntryBytes[0], consumedBytes, memberKeyBudget);
                    memberExchange->receive();
                    memberExchange->getMembers(lookupMemberOffsets, lookupMembers);
                    consumedBytes = memberExchange->getConsumedBytes();
                    size_t memberRange = SIZE_MAX;
                    for (size_t member = 0; needsCheck && member < lookupMembers.size(); member++) {
                        const size_t memberSeqLen = seqDbr.getSeqLenFromRange(memberRange, lookupMembers[member]);
                        needsCheck = canCoverMember(par, static_cast<int>(slicedRepSeqLen), static_cast<int>(memberSeqLen));
                        scalarMembers.add(std::abs(static_cast<int>(memberSeqLen) - static_cast<int>(slicedRepSeqLen)));
                    }
                }
                scalarMembers.getPositions(slicedScalarPositions);
            }
            // a slice of m members adds m + 1 parts at most
            if (isSlicingPair && freeKeys > 2) {
                const ClusterPair pair = acceptedPairs[0];
                const size_t repSize = slicedRepSeqLen + 2 + FETCHED_KEY_SIZE;
                size_t partSize = repSize;
                size_t partMembers = 0;
                // an assigned pair, or one whose checked parts failed, fails anyway, so it ends with an empty part
                const bool hasFailed = isPendingPairOpen && isPendingPairFailing && openPendingPair.repKey == pair.repKey && openPendingPair.memberKey == pair.memberKey;
                bool isEntryEnd = hasFailed || isAssigned(pair.repKey) || isAssigned(pair.memberKey);
                lookupMembers.clear();
                if (isEntryEnd == false) {
                    lookupKeys.assign(1, pair.memberKey);
                    memberExchange->request(lookupKeys, &acceptedEntryOffsets[0], &acceptedEntryBytes[0], slicedConsumedBytes, (freeKeys - 1) / 2);
                    memberExchange->receive();
                    memberExchange->getMembers(lookupMemberOffsets, lookupMembers);
                    slicedConsumedBytes = memberExchange->getConsumedBytes();
                    isEntryEnd = slicedConsumedBytes >= acceptedEntryBytes[0];
                }
                size_t memberRange = SIZE_MAX;
                for (size_t member = 0; member < lookupMembers.size(); member++) {
                    const DBKeyType memberKey = lookupMembers[member];
                    const size_t memberSeqLen = seqDbr.getSeqLenFromRange(memberRange, memberKey);
                    const size_t memberSize = memberSeqLen + 2 + FETCHED_KEY_SIZE;
                    if (partMembers > 0 && partSize + memberSize > maxPendingPairSize) {
                        addPendingPart(pair, partSize, false);
                        partSize = repSize;
                        partMembers = 0;
                    }
                    pendingMembers.push_back(memberKey);
                    pendingMemberIsLane.push_back(std::find(slicedScalarPositions.begin(), slicedScalarPositions.end(), slicedMemberPosition++) == slicedScalarPositions.end());
                    partSize += memberSize;
                    partMembers++;
                    // an uncoverable member fails its part and so the pair
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
            publishAccepted(getFirstUndoneRepKey(nextRepKey), acceptedSize);
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
        masterWaitSeconds.store(waitSeconds);
        // the reps of pending pairs are not done
        waitForSequences(nextSequences, getFirstUndoneRepKey(nextRepKey));
        std::swap(currentSequences, nextSequences);
    }
    for (int thread = 0; thread < par.threads; thread++) {
        delete threadTargetAligners[thread];
        delete threadMemberAligners[thread];
    }
    if (memberExchange != NULL && isAligningNode && nodes > 1) {
        memberExchange->finish();
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
        MMseqsNode::addPhaseValue(phaseValues, "Sequence exchange bytes", SSTR(servedSize) + " served for " + SSTR(servedEntrySize) + " entry bytes, " + SSTR(receivedServedEntriesSize) + " received, "
                                  + SSTR(requestCount) + " requests");
        MMseqsNode::addPhaseSeconds(phaseValues, "service: serving the requests", servingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: loading the source sequences", sourceLoadSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: packing the served entries", packSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: writing the served files", servedFileWriteSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: waiting for the served file writer", servedFileWriterWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: reading the served entries", readingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: unpacking the served entries", unpackSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: polling", pollingSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Requested entry bytes", SSTR(requestedEntrySize) + " requested, " + SSTR(receivedEntrySize) + " received not assigned, of " + SSTR(rangeByteOffsets[rangeCount]) + " sequence bytes");
        MMseqsNode::addPhaseValue(phaseValues, "Sequence service requests", SSTR(progressReadCount) + " polls, " + SSTR(objectReadCount) + " object reads, " + SSTR(objectWriteCount) + " object writes, "
                                  + SSTR(deleteCount) + " deletes in " + SSTR(static_cast<size_t>(serviceTimer.getTimediff())) + " s");
        MMseqsNode::addPhaseValue(phaseValues, "Request progress reads", SSTR(requestProgressReadCount) + ", unchanged: " + SSTR(requestProgressReadCount - requestProgressChangeCount)
                                  + ", changes: " + SSTR(requestProgressChangeCount) + ", request listings: " + SSTR(requestListingCount));
        MMseqsNode::addPhaseValue(phaseValues, "Served files", SSTR(servedFileCount) + ", at most " + SSTR(maxLiveServedSize) + " bytes not yet deleted");
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
        MMseqsNode::addPhaseSeconds(phaseValues, "aligning the fetches", alignSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Fetches with fewer SWIPE batches", SSTR(fewerBatchFetchCount) + " of " + SSTR(fetchCount));
        MMseqsNode::addPhaseValue(phaseValues, "Assigned keys read", SSTR(assignedKeyCount) + " in " + SSTR(assignedKeysSize) + " bytes");
        MMseqsNode::addPhaseSeconds(phaseValues, "the first fetch", firstFetchSeconds);
    }
    if (memberLookup != NULL) {
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member pairs", SSTR(multiFetchPairCount) + " over several fetches, " + SSTR(slicedPairCount) + " in slices");
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member pairs with several parts in a fetch", SSTR(multiPartPairCount) + ", " + SSTR(unstartedMemberCount) + " members not aligned after their pair failed");
        MMseqsNode::addPhaseSeconds(phaseValues, "looking up the cluster members", memberLookupSeconds);
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

static size_t getKeyAlignNodeFromPage(const std::vector<unsigned int> &pageAlignNodes, const DBReader<DBKeyType> &seqDbr, DBKeyType key) {
    return (pageAlignNodes[key / ASSIGNED_PAGE_KEYS] != UINT_MAX) ? pageAlignNodes[key / ASSIGNED_PAGE_KEYS] : getKeyAlignNode(seqDbr, key);
}

// linfinity: the master clusters the accepted pairs in key order
static size_t clusterAcceptedPairs(int threads, size_t dbSize, size_t memoryLimit, const DBReader<DBKeyType> &seqDbr,
                                   const std::vector<unsigned int> &pageAlignNodes, int mode, size_t paddedDbSize, size_t sizeClassCount,
                                   const std::string &accepted, const std::string &assignedKeysLog, const std::string &clusteredLogPrefix, const std::string &setCoverPrefix,
                                   const std::string &assignedBits, std::string &phaseValues) {
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
    std::vector<size_t> setReps;
    std::vector<size_t> setSizes;
    std::vector<std::pair<DBKeyType, size_t> > keyReps;
    std::vector<unsigned char> repSharesKey;
    std::vector<size_t> sharingReps;
    std::vector<uint64_t> words;
    std::vector<uint64_t> assignedKeyWords;
    std::vector<uint64_t> writtenAssignedKeys;
    std::thread assignedKeysWriter;
    std::atomic<bool> isWritingAssignedKeys(false);
    const auto writeAssignedKeys = [&](size_t clusteredOrderKey) {
        if (fwrite(writtenAssignedKeys.data(), sizeof(uint64_t), writtenAssignedKeys.size(), assignedKeysHandle) != writtenAssignedKeys.size()) {
            Debug(Debug::ERROR) << "Cannot write " << assignedKeysLog << "\n";
            EXIT(EXIT_FAILURE);
        }
        assignedKeysSize += writtenAssignedKeys.size() * sizeof(uint64_t);
        MMseqsNode::publishLog(assignedKeysHandle, assignedKeysLog, clusteredOrderKey, assignedKeysSize);
        isWritingAssignedKeys = false;
    };
    std::vector<DBKeyType> stepKeys;
    size_t memberCount = 0;
    // --cluster-mode 0: open sets in a heap, sets of two as pairs
    std::vector<ClusterPair> pairSets;
    SetCoverPairFile pairFile(pairSets, setCoverPrefix + "_clustered_pairsets", stepWords, memoryLimit, assigned);
    SetCoverSmallSetFiles smallSetFiles(setCoverPrefix + "_clustered_sets", memoryLimit, assigned);
    SetCoverLargeSetFiles largeSetFiles(setCoverPrefix + "_clustered_large", memoryLimit);
    const auto addClusterPair = [&](DBKeyType repKey, DBKeyType memberKey) {
        const ClusterPair pair = { repKey, memberKey };
        ClusteredLog &nodeFile = nodeClusteredLogs[getKeyAlignNodeFromPage(pageAlignNodes, seqDbr, repKey)];
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
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const std::string log = accepted + "." + SSTR(node);
            size_t repKey;
            size_t logSize;
            if ((nodeHasProgress[node] && nodeRepOrderKeys[node] == orderKeyEnd) || MMseqsNode::readLogProgress(log, repKey, logSize) == false) {
                continue;
            }
            if (nodeHasProgress[node] && repKey <= nodeRepOrderKeys[node] && logSize <= nodeLogSize[node]) {
                continue;
            }
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
            // only the slowest node gates the step
            MMseqsNode::sleepBeforePoll(polls++);
            waitWarning.check(accepted + "." + SSTR(slowestNode) + ".progress");
            waitSeconds += pollTimer.getTimediff();
            continue;
        }
        polls = 0;
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
        size_t repDroppedPairs = 0;
        size_t targetDroppedPairs = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+:repDroppedPairs, targetDroppedPairs)
        for (size_t rep = 0; rep < repCount; rep++) {
            const bool repIsFree = isAssigned(reps[rep].repKey % paddedDbSize) == false;
            size_t freeCount = 0;
            for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                pairIsFree[i] = repIsFree && isAssigned(logWords[i]) == false;
                freeCount += pairIsFree[i];
                repDroppedPairs += (repIsFree == false);
                targetDroppedPairs += (repIsFree && pairIsFree[i] == false);
            }
            repKeyOffsets[rep + 1] = (freeCount > 0) ? freeCount + 1 : 0;
        }
        repDroppedAcceptedPairCount += repDroppedPairs;
        targetDroppedAcceptedPairCount += targetDroppedPairs;
        sumOffsets(repKeyOffsets.data(), repCount, threads);
        if (mode == Parameters::SET_COVER) {
            // pieces of a rep from several align nodes are one set
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
            // a set waits until no larger set can come
            size_t setPairs = 0;
            for (size_t set = 0; set < setSizes.size(); set++) {
                setPairs += setSizes[set] == 2;
            }
            const bool isBufferingPairs = pairFile.prepareBatch(setPairs);
            const size_t pairOffset = pairSets.size();
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
            bool isSharingRepFree = false;
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
                // a rep assigned by its own earlier piece is still free
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
        const size_t headerWords = getAssignedKeysHeaderWords();
        const size_t stepStart = assignedKeyWords.size();
        size_t codeWords = (codeBits + 63) / 64;
        assignedKeyWords.resize(stepStart + headerWords + codeWords, 0);
        assignedKeyWords[stepStart] = clusteredOrderKey;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            assignedKeyWords[stepStart + 1 + node] = nodeReadSize[node];
        }
        assignedKeyWords[stepStart + 1 + MMseqsNode::count] = stepKeys.size();
        assignedKeyWords[stepStart + 2 + MMseqsNode::count] = stepKeys.empty() ? 0 : stepKeys.front();
        assignedKeyWords[stepStart + 3 + MMseqsNode::count] = keyDiffLowBits;
        assignedKeyWords[stepStart + 4 + MMseqsNode::count] = codeWords;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            assignedKeyWords[stepStart + 5 + MMseqsNode::count + node] = nodeHasProgress[node] ? nodeRepOrderKeys[node] : 0;
        }
        KmerCodeWriter codeWrite((char *) (assignedKeyWords.data() + stepStart + headerWords), codeWords * sizeof(uint64_t), 0);
        for (size_t i = 1; i < stepKeys.size(); i++) {
            putRice(codeWrite, stepKeys[i] - stepKeys[i - 1] - 1, keyDiffLowBits);
        }
        putBits(codeWrite, 0, (CHAR_BIT - codeWrite.bitCount) % CHAR_BIT);
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
    MMseqsNode::addPhaseValue(phaseValues, "Logged pairs dropped", SSTR(repDroppedAcceptedPairCount.load()) + " with an absorbed rep, " + SSTR(targetDroppedAcceptedPairCount.load()) + " with a taken target");
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

// empty if --local-tmp can hold files of this rank
static std::string getLocalTmpProblem(const Parameters &par) {
    // ranks on one machine may create it at once
    if (FileUtil::makeDir(par.localTmp.c_str()) == false && FileUtil::directoryExists(par.localTmp.c_str()) == false) {
        Debug(Debug::ERROR) << "Cannot create local tmp dir " << par.localTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    int localTmpFd = open(par.localTmp.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    const StorageKind localTmpStorage = getStorageKind(localTmpFd);
    struct stat localTmpStat, dbStat;
    const bool isDbDevice = fstat(localTmpFd, &localTmpStat) == 0 && stat(par.db1Index.c_str(), &dbStat) == 0 && localTmpStat.st_dev == dbStat.st_dev;
    close(localTmpFd);
    if (localTmpStorage != STORAGE_LOCAL_FLASH) {
        return std::string("it is on ") + getStorageName(localTmpStorage) + ", not local flash";
    }
    return isDbDevice ? "it is on the device of the database" : "";
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
    size_t alignNodeCount = MMseqsNode::count - MMseqsNode::getFirstAlignNode();
    // the first .partitions line holds the target split blocks
    const bool hasTargetSplits = par.align2clustMode == Parameters::ALIGN2CLUST_MODE_TARGET_SPLIT && alignNodeCount > 1;
    std::string partitionsData = MMseqsNode::readFile(par.db2 + ".partitions");
    char *partitionLine = (char *) partitionsData.c_str();
    for (; hasTargetSplits && *partitionLine != '\n'; partitionLine += Util::skipWhitespace(partitionLine)) {
        targetSplitBlockOffsets.push_back(Util::fast_atoi<size_t>(partitionLine));
        partitionLine += Util::skipNoneWhitespace(partitionLine);
    }
    std::vector<PrefPartition> partitions;
    uint128 totalAlignmentCells = 0;
    size_t totalPrefSize = 0;
    for (char *line = partitionLine + (hasTargetSplits ? 1 : 0); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[6];
        Util::getWordsOfLine(line, columns, 6);
        PrefPartition partition;
        partition.prefOffset = Util::fast_atoi<size_t>(columns[1]);
        partition.prefSize = Util::fast_atoi<size_t>(columns[2]);
        partition.alignmentCells = Util::fast_atoi<size_t>(columns[3]);
        partition.node = Util::fast_atoi<size_t>(columns[4]);
        partition.targetSplit = hasTargetSplits ? Util::fast_atoi<size_t>(columns[5]) : 0;
        partitions.push_back(partition);
        totalAlignmentCells += partition.alignmentCells;
        totalPrefSize += partition.prefSize;
    }
    // block table for the target splits and the local copy
    DBReader<DBKeyType> blockDbr(par.db1.c_str(), par.db1Index.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
    blockDbr.open(DBReader<DBKeyType>::NOSORT);
    blockDbr.setBlockCacheSize(0);
    if (hasTargetSplits) {
        size_t targetSplitCount = targetSplitBlockOffsets.size() - 1;
        const DBReader<DBKeyType>::Index *blockIndex = blockDbr.getIndex();
        std::vector<uint128> splitCosts(targetSplitCount, 0);
        for (size_t i = 0; i < partitions.size(); i++) {
            splitCosts[partitions[i].targetSplit] += partitions[i].alignmentCells;
        }
        std::vector<size_t> splitSizes(targetSplitCount, 0);
        std::vector<size_t> order(targetSplitCount);
        size_t totalSize = 0;
        size_t maxSplitSize = 0;
        for (size_t split = 0; split < targetSplitCount; split++) {
            size_t keyBegin = blockDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[split]];
            size_t keyEnd = blockDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[split + 1]];
            for (size_t range = blockDbr.findIdRange(keyBegin); range < blockDbr.getIndexSize() && static_cast<size_t>(blockIndex[range].id) < keyEnd; range++) {
                size_t rangeEnd = (range + 1 < blockDbr.getIndexSize()) ? static_cast<size_t>(blockIndex[range + 1].id) : blockDbr.getSize();
                splitSizes[split] += (std::min(keyEnd, rangeEnd) - std::max(keyBegin, static_cast<size_t>(blockIndex[range].id))) * blockIndex[range].length;
            }
            totalSize += splitSizes[split];
            maxSplitSize = std::max(maxSplitSize, splitSizes[split]);
            order[split] = split;
        }

        // largest cost first to the align node with the least cost
        SORT_SERIAL(order.begin(), order.end(), [&](size_t first, size_t second) {
            return splitCosts[first] > splitCosts[second] || (splitCosts[first] == splitCosts[second] && first < second);
        });
        size_t maxNodeSize = (totalSize + alignNodeCount - 1) / alignNodeCount + maxSplitSize;
        std::vector<uint128> nodeCosts(alignNodeCount, 0);
        std::vector<size_t> nodeSizes(alignNodeCount, 0);
        targetSplitAlignNodes.resize(targetSplitCount);
        for (size_t i = 0; i < targetSplitCount; i++) {
            size_t split = order[i];
            size_t node = SIZE_MAX;
            for (size_t alignNode = 0; alignNode < alignNodeCount; alignNode++) {
                if (nodeSizes[alignNode] + splitSizes[split] <= maxNodeSize && (node == SIZE_MAX || nodeCosts[alignNode] < nodeCosts[node])) {
                    node = alignNode;
                }
            }
            nodeCosts[node] += splitCosts[split];
            nodeSizes[node] += splitSizes[split];
            targetSplitAlignNodes[split] = MMseqsNode::getFirstAlignNode() + node;
        }
        if (MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) {
            size_t alignNode = MMseqsNode::rank - MMseqsNode::getFirstAlignNode();
            MMseqsNode::addPhaseValue(phaseValues, "Target splits", SSTR(std::count(targetSplitAlignNodes.begin(), targetSplitAlignNodes.end(), MMseqsNode::rank)) + " of " + SSTR(targetSplitCount)
                                      + " with " + SSTR(nodeSizes[alignNode]) + " of " + SSTR(totalSize) + " sequence bytes and "
                                      + SSTR(static_cast<size_t>(nodeCosts[alignNode] * 1000 / std::max((uint128) 1, totalAlignmentCells))) + " per mille of the cost");
        }
    }
    std::string seqDb = par.db1;
    std::string cluPartDb = par.filterCluDBFile;
    std::vector<std::string> localCopies;
    size_t localCopySize = 0;
    bool hasLocalServedBlocks = false;
    // a master that only clusters reads no sequences
    if (par.localTmp.empty() == false && MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) {
        std::string localTmpProblem = getLocalTmpProblem(par);
        if (localTmpProblem.empty()) {
            // copy only the served blocks
            const std::vector<size_t> &blockKeyOffsets = blockDbr.getBlockKeyOffsets();
            const DBReader<DBKeyType>::Index *blockIndex = blockDbr.getIndex();
            const std::vector<std::string> dataFiles = FileUtil::findDatafiles(par.db1.c_str());
            std::vector<std::vector<std::pair<size_t, size_t> > > servedFileRanges(dataFiles.size());
            size_t servedSize = 0;
            for (size_t block = 0; block + 1 < blockKeyOffsets.size(); block++) {
                if (getBlockAlignNode(block) != MMseqsNode::rank) {
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
            // a copy left by a killed run does not take the free space
            const std::string localDir = par.localTmp + "/" + FileUtil::baseName(par.db3) + ".served." + SSTR(MMseqsNode::rank);
            std::vector<std::string> oldCopies = FileUtil::findDatafiles((localDir + "/" + FileUtil::baseName(par.db1)).c_str());
            // and the own cluster part
            std::vector<std::vector<std::pair<size_t, size_t> > > cluFileRanges(par.filterCluDBFile.empty() ? 0 : MMseqsNode::count);
            if (cluFileRanges.empty() == false) {
                std::vector<std::string> oldCluCopies = FileUtil::findDatafiles((localDir + "/" + FileUtil::baseName(par.filterCluDBFile)).c_str());
                oldCopies.insert(oldCopies.end(), oldCluCopies.begin(), oldCluCopies.end());
                cluFileRanges[MMseqsNode::rank].emplace_back(0, FileUtil::getFileSize(par.filterCluDBFile + "." + SSTR(MMseqsNode::rank)));
                servedSize += cluFileRanges[MMseqsNode::rank][0].second;
            }
            for (size_t i = 0; i < oldCopies.size(); i++) {
                FileUtil::remove(oldCopies[i].c_str());
            }
            struct statvfs fs;
            const size_t freeSize = (statvfs(par.localTmp.c_str(), &fs) == 0) ? static_cast<size_t>(fs.f_bavail) * fs.f_frsize : 0;
            if (freeSize < servedSize) {
                localTmpProblem = "it has " + SSTR(freeSize) + " free bytes, the copies need " + SSTR(servedSize);
            } else {
                seqDb = FileUtil::copyToLocalDir(par.db1, localDir, par.threads, localCopies, localCopySize, servedFileRanges);
                if (cluFileRanges.empty() == false) {
                    size_t cluCopySize = 0;
                    cluPartDb = FileUtil::copyToLocalDir(par.filterCluDBFile, localDir, par.threads, localCopies, cluCopySize, cluFileRanges);
                    localCopySize += cluCopySize;
                }
                hasLocalServedBlocks = true;
                MMseqsNode::addPhaseTime(phaseValues, "copying the served blocks to --local-tmp", phaseValueTimer);
            }
        }
        if (hasLocalServedBlocks == false) {
            if (par.localTmpFallback == false) {
                Debug(Debug::ERROR) << "Cannot copy the served blocks to --local-tmp " << par.localTmp << ": " << localTmpProblem << ". Please choose another --local-tmp or set --local-tmp-fallback 1 to read them from the shared folder.\n";
                EXIT(EXIT_FAILURE);
            }
            MMseqsNode::reportUnusableLocalTmp("Cannot copy the served blocks to --local-tmp " + par.localTmp + ": " + localTmpProblem);
        }
        MMseqsNode::addPhaseValue(phaseValues, "Local tmp", hasLocalServedBlocks ? (par.filterCluDBFile.empty() ? "holds the served blocks" : "holds the served blocks and the cluster part") : "not used, " + localTmpProblem);
    }

    // only the master writes and reads the open sets of set cover
    std::string setCoverPrefix = par.db3;
    if (MMseqsNode::isMaster() && par.clusteringMode == Parameters::SET_COVER && par.localTmp.empty() == false) {
        std::string localTmpProblem = getLocalTmpProblem(par);
        if (localTmpProblem.empty()) {
            setCoverPrefix = par.localTmp + "/" + FileUtil::baseName(par.db3);
        } else if (par.localTmpFallback == false) {
            Debug(Debug::ERROR) << "Cannot keep the open sets in --local-tmp " << par.localTmp << ": " << localTmpProblem << ". Please choose another --local-tmp or set --local-tmp-fallback 1 to keep them in the shared folder.\n";
            EXIT(EXIT_FAILURE);
        } else {
            MMseqsNode::reportUnusableLocalTmp("Cannot keep the open sets in --local-tmp " + par.localTmp + ": " + localTmpProblem);
        }
        MMseqsNode::addPhaseValue(phaseValues, "Open sets", localTmpProblem.empty() ? "in --local-tmp" : "in the shared folder, " + localTmpProblem);
    }

    blockDbr.close();
    DBReader<DBKeyType> seqDbr(seqDb.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    seqDbr.open(DBReader<DBKeyType>::NOSORT);
    // a claiming node may read every target split, three files each
    struct rlimit openFileLimit = {0, 0};
    getrlimit(RLIMIT_NOFILE, &openFileLimit);
    size_t minOpenFiles = seqDbr.getDataFileCnt() + 64 + 2 * MMseqsNode::count + 3 * (hasTargetSplits ? targetSplitBlockOffsets.size() - 1 : 0);
    if (minOpenFiles > openFileLimit.rlim_cur) {
        Debug(Debug::ERROR) << "Raise the open file limit (ulimit -n) to at least " << minOpenFiles << "\n";
        EXIT(EXIT_FAILURE);
    }
    const std::vector<size_t> nodeKeyOffsets = seqDbr.getEvenNodeKeyOffsets(MMseqsNode::count);
    const size_t dbSize = seqDbr.getSize();
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    // a copy on tmpfs holds the sequences in memory
    const size_t nodeMemoryLimit = MMseqsNode::getMemoryLimit(par, localCopySize);
    const bool isMasterAligning = MMseqsNode::getFirstAlignNode() == 0;
    const size_t masterMemoryLimit = nodeMemoryLimit / (isMasterAligning ? MASTER_MEMORY_DIVISOR : 1);
    // the master holds the assigned bits and a log buffer per node in both clustering modes
    const size_t masterFixedSize = (dbSize + 63) / 64 * sizeof(uint64_t) + MMseqsNode::count * CLUSTERED_LOG_BUFFER_SIZE;
    const size_t masterMemorySize = masterFixedSize + ((par.clusteringMode == Parameters::SET_COVER) ? masterMemoryLimit : masterMemoryLimit / STEP_MEMORY_DIVISOR);
    if (MMseqsNode::isMaster() && masterFixedSize >= nodeMemoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to cluster. Memory limit: " << nodeMemoryLimit << " bytes, the assigned bits of " << dbSize << " sequences need: "
                            << (dbSize + 63) / 64 * sizeof(uint64_t) << " bytes, the clustered logs of " << MMseqsNode::count << " nodes need: "
                            << MMseqsNode::count * CLUSTERED_LOG_BUFFER_SIZE << " bytes.\n"
                            << "Raise RAM size or --split-memory-limit. The assigned bits do not shrink with more nodes.\n";
        EXIT(EXIT_FAILURE);
    }

    // cut the groups into alignment splits of about equal cells
    const size_t maxAlignmentSplitCount = std::max<size_t>(1, std::min(ALIGNMENT_SPLITS_PER_NODE * alignNodeCount, totalPrefSize / MIN_ALIGNMENT_SPLIT_PREF_SIZE));
    std::vector<size_t> alignmentSplitPartitionOffsets(1, 0);
    uint128 cells = 0;
    for (size_t i = 0; i + 1 < partitions.size(); i++) {
        cells += partitions[i].alignmentCells;
        if (alignmentSplitPartitionOffsets.size() < maxAlignmentSplitCount && cells * maxAlignmentSplitCount >= totalAlignmentCells * alignmentSplitPartitionOffsets.size()) {
            alignmentSplitPartitionOffsets.push_back(i + 1);
        }
    }
    alignmentSplitPartitionOffsets.push_back(partitions.size());
    MMseqsNode::WorkQueue alignmentSplitQueue(par.db3 + "_alignment_splits", alignmentSplitPartitionOffsets.size() - 1);
    // a cell is a window of a target split
    MMseqsNode::WorkQueue cellQueue(par.db3 + "_cells", hasTargetSplits ? partitions.size() : 0);

    // builds the block table of getKeyBlock
    seqDbr.setBlockCacheSize(0);
    const std::vector<size_t> &blockKeyOffsets = seqDbr.getBlockKeyOffsets();
    // the master keeps the align node of every page of assigned bits
    const size_t pageCount = (dbSize + ASSIGNED_PAGE_KEYS - 1) / ASSIGNED_PAGE_KEYS;
    std::vector<unsigned int> pageAlignNodes(MMseqsNode::isMaster() ? pageCount : 0);
    for (size_t block = 0, pageEnd = 0; MMseqsNode::isMaster() && block + 1 < blockKeyOffsets.size(); block++) {
        unsigned int node = getBlockAlignNode(block);
        size_t firstPage = blockKeyOffsets[block] / ASSIGNED_PAGE_KEYS;
        const size_t lastPage = (blockKeyOffsets[block + 1] - 1) / ASSIGNED_PAGE_KEYS;
        for (size_t page = firstPage; page <= lastPage; page++) {
            // a page with keys of two nodes uses getKeyAlignNode
            pageAlignNodes[page] = (page < pageEnd && pageAlignNodes[page] != node) ? UINT_MAX : node;
        }
        pageEnd = lastPage + 1;
    }
    // an align node gets every assigned key
    size_t assignedPageCount = (MMseqsNode::rank < MMseqsNode::getFirstAlignNode()) ? 0 : pageCount;
    size_t assignedMemorySize = assignedPageCount * (ASSIGNED_PAGE_KEYS / 8) + ((assignedPageCount > 0) ? pageCount * sizeof(uint64_t *) : 0) + pageAlignNodes.size() * sizeof(unsigned int);
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
    const std::string memberRequests = par.db3 + "_member_requests";
    const std::string servedMembers = par.db3 + "_members";
    std::vector<std::string> sequenceIndexes;
    std::vector<std::string> sequenceFiles;
    // an empty progress file has no requests and no served files
    if (MMseqsNode::rank >= MMseqsNode::getFirstAlignNode() && alignNodeCount > 1) {
        sequenceIndexes.push_back(requests + "." + SSTR(MMseqsNode::rank) + ".progress");
        sequenceIndexes.push_back(served + "." + SSTR(MMseqsNode::rank) + ".progress");
        // the request files of all nodes share a directory, one listing shows the new requests
        if (FileUtil::makeDir(requests.c_str()) == false && FileUtil::directoryExists(requests.c_str()) == false) {
            Debug(Debug::ERROR) << "Cannot create " << requests << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    // every node serves the members of its own cluster part, the aligning nodes ask for them
    if (par.filterCluDBFile.empty() == false && MMseqsNode::count > 1) {
        sequenceIndexes.push_back(servedMembers + ".index." + SSTR(MMseqsNode::rank));
        if (MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) {
            sequenceIndexes.push_back(memberRequests + ".index." + SSTR(MMseqsNode::rank));
            sequenceFiles.push_back(memberRequests + "." + SSTR(MMseqsNode::rank));
        }
    }
    for (size_t i = 0; i < sequenceIndexes.size(); i++) {
        if (fclose(FileUtil::openAndDelete(sequenceIndexes[i].c_str(), "w")) != 0) {
            Debug(Debug::ERROR) << "Cannot create " << sequenceIndexes[i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }

    // remove the item files a killed run left
    if (hasTargetSplits == false) {
        alignmentSplitQueue.removeTakenItemFiles();
    }
    cellQueue.removeTakenItemFiles();
    // all nodes use the smallest memory limit
    const size_t smallestMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db3 + ".memory", nodeMemoryLimit - std::min(nodeMemoryLimit, assignedMemorySize));
    const size_t paddedDbSize = (dbSize + 63) / 64 * 64;
    const size_t sizeClassCount = (par.clusteringMode == Parameters::SET_COVER) ? 64 - __builtin_clzll(dbSize | 1) : 1;
    const size_t maxRecordSize = std::max<size_t>(1, smallestMemoryLimit / RECORD_MEMORY_DIVISOR);
    // all nodes cut served files at the same size
    const size_t servedFileEntrySize = std::max(BatchReader::MAX_READ_REQUEST_SIZE, smallestMemoryLimit / SERVED_FILE_MEMORY_DIVISOR);

    RepTargetsReader repTargetsReader(par, seqDbr, partitions, alignmentSplitPartitionOffsets, alignmentSplitQueue, maxRecordSize, memoryLimit, paddedDbSize, sizeClassCount);
    TargetSplitReader *targetSplitReader = hasTargetSplits ? new TargetSplitReader(par, seqDbr, partitions, cellQueue, maxRecordSize, memoryLimit, paddedDbSize, sizeClassCount) : NULL;
    if (sizeClassCount > 1 && targetSplitReader != NULL) {
        targetSplitReader->countSizeClasses();
    } else if (sizeClassCount > 1) {
        repTargetsReader.countSizeClasses();
    }
    MMseqsNode::addPhaseTime(phaseValues, "counting the target size classes", phaseValueTimer);

    ClusterMemberLookup *memberLookup = NULL;
    ClusterMemberExchange *memberExchange = NULL;
    DBReader<DBKeyType> *memberDbr = NULL;
    if (par.filterCluDBFile.empty() == false) {
        const size_t entriesPerSampledKey = std::max(MIN_ENTRIES_PER_SAMPLED_KEY, dbSize / std::max<size_t>(1, smallestMemoryLimit / SAMPLED_KEY_MEMORY_DIVISOR / sizeof(DBKeyType)));
        std::vector<size_t> cluKeyOffsets = MMseqsNode::getClusterKeyOffsets(par.db3 + ".cluster_key_offsets", par.filterCluDBFile, dbSize);
        cluKeyOffsets[0] = 0;
        // every node, the master too, answers for its own cluster part
        // the master clusters while it answers, aligning or not
        size_t lookupMemoryLimit = nodeMemoryLimit - std::min(nodeMemoryLimit, (MMseqsNode::isMaster() ? std::min(masterMemoryLimit, masterMemorySize) : 0) + assignedMemorySize);
        memberLookup = new ClusterMemberLookup(par.filterCluDBFile, cluPartDb, cluKeyOffsets, entriesPerSampledKey, lookupMemoryLimit / 16);
        memberExchange = new ClusterMemberExchange(memberRequests, servedMembers, *memberLookup, std::max(1, par.threads / 4));
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
            memberCount = clusterAcceptedPairs(clusterThreads, dbSize, masterMemoryLimit, seqDbr, pageAlignNodes, par.clusteringMode, paddedDbSize, sizeClassCount,
                                               accepted, assignedKeysLog, clusteredLogPrefix, setCoverPrefix, assignedBits, clusterPhaseValues);
        });
    }
    phaseValueTimer.reset();
    std::thread memberServer;
    if (memberExchange != NULL && MMseqsNode::count > 1) {
        memberServer = std::thread([&]() { memberExchange->serve(MMseqsNode::getFirstAlignNode()); });
    }
    // all nodes ask and answer within the 2 * memberSplitSize of getReservedMemory, for memberKeyBudget member keys of 7 words and 2 flags (pending, thread copy,
    // received, codes), as many pending parts of 9 words and 2 flags (pending pair, size, member offset, part arrays, rep offset) and as many accepted pairs of
    // 25 words and a flag (pair, entry, lookup, codes, a served request); a code takes a word at most
    const size_t memberKeyBudget = std::max<size_t>(1, 2 * std::min(memoryLimit, smallestMemoryLimit) / MEMBER_SPLIT_MEMORY_DIVISOR / ((7 + 9 + 25) * sizeof(uint64_t) + 5));
    alignRepTargets(par, seqDbr, repTargetsReader, targetSplitReader, memoryLimit, masterMemoryLimit, servedFileEntrySize, representativeResidues, paddedDbSize, sizeClassCount,
                    memberLookup, memberExchange, memberKeyBudget, memberDbr, hasLocalServedBlocks, phaseValues);
    // serve until every aligning node has its members
    if (memberServer.joinable()) {
        memberServer.join();
    }
    if (memberExchange != NULL) {
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the cluster member sizes", memberExchange->sizeWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the cluster members", memberExchange->memberWaitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member exchange", SSTR(memberExchange->servedSize) + " bytes served for "
                                  + SSTR(memberExchange->servedKeyCount) + " keys in " + SSTR(memberExchange->getAnswerCount()) + " answers, "
                                  + SSTR(memberExchange->requestedKeyCount) + " keys requested in " + SSTR(memberExchange->getRequestCount()) + " requests");
    }
    delete memberExchange;
    delete memberLookup;
    repTargetsReader.close();
    if (targetSplitReader != NULL) {
        targetSplitReader->close();
        if (MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) {
            MMseqsNode::addPhaseValue(phaseValues, "Target split prefilter bytes", SSTR(targetSplitReader->readSize) + " read of " + SSTR(targetSplitReader->prefSize));
        }
        if (targetSplitReader->isClaiming) {
            std::string stealShares;
            for (size_t node = MMseqsNode::getFirstAlignNode(); node < MMseqsNode::count; node++) {
                stealShares += ((stealShares.empty()) ? "" : " ") + ((node == MMseqsNode::rank) ? std::string("-") : SSTR(static_cast<int>(targetSplitReader->stealShares[node] * 1000)));
            }
            MMseqsNode::addPhaseValue(phaseValues, "Target split cells", SSTR(targetSplitReader->homeCellCount) + " home, " + SSTR(targetSplitReader->stolenCellCount) + " stolen, "
                                      + SSTR(targetSplitReader->lostCellCount) + " lost; stolen cost " + SSTR(static_cast<size_t>(1000 * targetSplitReader->stolenCost / std::max(1.0, targetSplitReader->takenCost)))
                                      + " per mille of the cost in " + SSTR(targetSplitReader->stealWindowCount) + " windows, "
                                      + SSTR(targetSplitReader->stolenCostSignChanges) + " sign changes of its difference; steal shares per mille " + stealShares);
            MMseqsNode::addPhaseValue(phaseValues, "Stolen target bytes", SSTR(targetSplitReader->stolenTargetBytes) + " of " + SSTR(targetSplitReader->targetBytes) + " target bytes");
        }
        delete targetSplitReader;
    }
    if (MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) {
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
            MMseqsNode::addPhaseValue(phaseValues, "Target pairs", SSTR(examinedTargetPairCount.load()) + " examined, " + SSTR(repSkippedTargetPairCount.load()) + " skipped by rep (" + SSTR(repSlotSkippedTargetPairCount.load()) + " by empty slot), "
                                     + SSTR(targetSkippedTargetPairCount.load()) + " skipped by target, " + SSTR(uncoverableTargetPairCount.load()) + " uncoverable, "
                                     + SSTR(alignedTargetPairCount.load()) + " aligned");
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
    if (hasTargetSplits == false) {
        alignmentSplitQueue.removeTakenItemFiles();
    }
    cellQueue.removeTakenItemFiles();
    if (MMseqsNode::isMaster()) {
        clusterThread.join();
        Debug(Debug::INFO) << "Number of clusters: " << representativeCount - memberCount << "\n";
        phaseValues += clusterPhaseValues;
        std::vector<unsigned int>().swap(pageAlignNodes);
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
    rmdir(requests.c_str());
    MMseqsNode::addPhaseTime(phaseValues, "sending the cluster pairs", phaseValueTimer);
    if (memberDbr != NULL) {
        // every reader, the serving ones too, has added its lookups by now
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member lookup", SSTR(memberLookupCount.load()) + " entries read, " + SSTR(scannedMemberCount.load())
                                  + " members, " + SSTR(memberLookupReadSize.load()) + " bytes in " + SSTR(memberLookupReadCount.load()) + " reads, "
                                  + SSTR(memberIndexReadSize.load()) + " index bytes in " + SSTR(memberIndexReadCount.load()) + " reads");
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
