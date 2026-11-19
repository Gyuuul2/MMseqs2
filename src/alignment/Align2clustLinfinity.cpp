#include "Align2clust.h"
#include "kmermatcher.h"
#include "Util.h"
#include "Debug.h"
#include "DBWriter.h"
#include "FastSort.h"
#include "Alignment.h"
#include "FileUtil.h"
#include "S3Client.h"
#include <cerrno>
#include <climits>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <dirent.h>
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
        FILE *reader = MMseqsNode::openSharedFile(fileName, "rb");
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
                source.reader = MMseqsNode::openSharedFile(source.path, "rb");
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

// a .partitions row of one align node: pref bytes [prefOffset, prefOffset + prefSize) of node
struct PrefPartition {
    size_t firstKey;
    size_t prefOffset;
    size_t prefSize;
    size_t alignmentCells;
    size_t node;
};

// target split mode: the entries [firstEntry, endEntry) of the target split files of node
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

// target split mode
static std::vector<size_t> targetSplitBlockOffsets;
static std::vector<unsigned int> targetSplitAlignNodes;
// a window belongs to one align group, a node holds the target split table of its own group
static std::vector<unsigned int> windowAlignGroups;
static unsigned int alignGroup;
// the nodes of the align group share its target split table, so only they request sequences from each other
static size_t alignGroupNodeBegin;
static size_t alignGroupNodeEnd;

// by target split, without target splits there is one align node
static size_t getBlockAlignNode(size_t block) {
    if (targetSplitAlignNodes.empty()) {
        return MMseqsNode::getFirstAlignNode();
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

// the blocks of a window's reps go round the align group
static size_t getBlockRepServingNode(size_t block) {
    return alignGroupNodeBegin + block % (alignGroupNodeEnd - alignGroupNodeBegin);
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
    if (MMseqsNode::hasFailedTooLong(missingSince, attempts)) {
        Debug(Debug::ERROR) << error;
        EXIT(EXIT_FAILURE);
    }
}

static std::string describeServedFile(size_t size, size_t observedSize, uint64_t fileNumber, uint64_t observedFileNumber, int error) {
    return " (expected " + SSTR(size) + " bytes, observed " + (observedSize == SIZE_MAX ? std::string("unavailable") : SSTR(observedSize))
           + " bytes; expected file number " + SSTR(fileNumber) + ", observed " + (observedFileNumber == UINT64_MAX ? std::string("unavailable") : SSTR(observedFileNumber))
           + "; error " + SSTR(error) + ")\n";
}

static std::string formatSeconds(double seconds) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.1f s", seconds);
    return buffer;
}

// windowFirstKeys starts with 0; a cursor searches again only once key leaves its window
static size_t findWindow(const std::vector<size_t> &windowFirstKeys, size_t key, size_t &window) {
    if (window == SIZE_MAX || key < windowFirstKeys[window] || (window + 1 < windowFirstKeys.size() && key >= windowFirstKeys[window + 1])) {
        window = std::upper_bound(windowFirstKeys.begin() + 1, windowFirstKeys.end(), key) - windowFirstKeys.begin() - 1;
    }
    return window;
}

// the windows of the set cover passes follow each other
static size_t getWindowPosition(const std::vector<size_t> &windowFirstKeys, size_t paddedDbSize, size_t orderKey) {
    size_t window = SIZE_MAX;
    return orderKey / paddedDbSize * windowFirstKeys.size() + findWindow(windowFirstKeys, orderKey % paddedDbSize, window);
}

// the threads of this process, io_uring workers included, and the io_uring workers
static void countProcessThreads(size_t &threads, size_t &ringWorkers) {
    threads = 0;
    ringWorkers = 0;
    DIR *tasks = opendir("/proc/self/task");
    if (tasks == NULL) {
        return;
    }
    for (struct dirent *entry; (entry = readdir(tasks)) != NULL;) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        threads++;
        FILE *comm = fopen(("/proc/self/task/" + std::string(entry->d_name) + "/comm").c_str(), "r");
        char name[32] = "";
        if (comm != NULL) {
            ringWorkers += fgets(name, sizeof(name), comm) != NULL && strncmp(name, "iou-", 4) == 0;
            fclose(comm);
        }
    }
    closedir(tasks);
}

// per window of an align node, for the decile table at the end
struct WindowCounters {
    WindowCounters() : examinedPairs(0), alignedPairs(0), targetSkippedPairs(0), repSkippedRecords(0), alignmentCells(0), fetchCount(0), prefilterBufferCount(0),
                       alignSeconds(0), otherSeconds(0), lastFetch(SIZE_MAX), lastPrefilterBuffer(SIZE_MAX) {}
    // added by the aligning threads
    size_t examinedPairs;
    size_t alignedPairs;
    size_t targetSkippedPairs;
    size_t repSkippedRecords;
    // the cells the node loaded for the window
    size_t alignmentCells;
    size_t fetchCount;
    size_t prefilterBufferCount;
    double alignSeconds;
    // fetch work and waits
    double otherSeconds;
    size_t lastFetch;
    size_t lastPrefilterBuffer;
};

// per log2 bucket of the planned fetch bytes
struct FetchSizeCounters {
    FetchSizeCounters() : fetchCount(0), plannedSize(0), unassignedRecordSize(0), alignedPairs(0), alignSeconds(0) {}
    size_t fetchCount;
    size_t plannedSize;
    size_t unassignedRecordSize;
    size_t alignedPairs;
    double alignSeconds;
};

// linfinity: records [begin, end) of a prefilter buffer and their fetched entries
struct FetchedSequences {
    FetchedSequences() : prefilterBuffer(0), prefilterBufferNumber(0), begin(0), end(0), repKeyBegin(0), repKeyEnd(0), targetCount(0), pendingPairCount(0), firstPendingRepKey(0), plannedSize(0),
                         unassignedRecordSize(0), isLast(false), entries(NULL), allocatedSize(0), acceptedSize(0), ownKeyCopyEnd(0), isReady(false) {}
    size_t prefilterBuffer;
    size_t prefilterBufferNumber;
    size_t begin;
    size_t end;
    // first and end of its rep keys, other nodes' blocks between them come from their rep serving node
    size_t repKeyBegin;
    size_t repKeyEnd;
    size_t targetCount;
    // pending pairs whose members it brings
    size_t pendingPairCount;
    // the first rep of the pairs that wait for their member checks when it was selected
    size_t firstPendingRepKey;
    // the bytes of the fetch size rule and those of its records whose rep was not assigned
    size_t plannedSize;
    size_t unassignedRecordSize;
    bool isLast;
    std::vector<DBKeyType> keys;
    std::vector<size_t> fetchSourceKeyOffsets;
    char *entries;
    size_t allocatedSize;
    std::vector<size_t> entryOffsets;
    // the accepted bytes at its hand-off, for the master gate
    size_t acceptedSize;
    // the own keys before it are copied
    size_t ownKeyCopyEnd;
    bool isReady;

    size_t memorySize() const {
        return allocatedSize + keys.capacity() * sizeof(DBKeyType) + entryOffsets.capacity() * sizeof(size_t);
    }
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

static const time_t PUBLISH_SECONDS = 1;
static const time_t READ_SECONDS = 1;
// read the assigned keys a few times a second
static const double BATCH_READ_SECONDS = PUBLISH_SECONDS / 4.0;
// a thread spends at most 1 / LOG_COST_FACTOR of its time on publishing and reading, a fast file system gets shorter periods
static const double LOG_COST_FACTOR = 16;
static const double MIN_PUBLISH_SECONDS = 0.1;
static const double MIN_READ_SECONDS = 0.05;
// an align node prints its counters so far once a minute
static const double STATUS_SECONDS = 60;
static const uint64_t ACCEPTED_REP_BIT = 1ULL << 63;

static size_t getRepOrderKey(uint64_t word) {
    return word & ~ACCEPTED_REP_BIT;
}
static size_t getAssignedKeysHeaderWords() {
    return MMseqsNode::count + 5;
}

static const size_t PREFILTER_BUFFER_MEMORY_DIVISOR = 32;
// pref bytes that fill a prefilter buffer: 10 bytes per pref byte, 48 bytes and a record per piece
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
#pragma omp parallel
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
// progress or index reads per node and second
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

// entries [begin, end) in chunks: one chunk on the calling thread with isOneStep, else all of them by the caller's team; returns the first entry left
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

// the parts of a pair follow each other
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
    // the entries of [keyBegin, keyEnd) in the cluster part of node
    struct PreloadedRange {
        size_t keyBegin;
        size_t keyEnd;
        size_t node;
        size_t firstEntry;
        size_t endEntry;
        // the index pairs of the entries and the pair behind
        std::vector<uint64_t> indexPairs;
        // empty if the entries are read from the part
        std::vector<char> data;

        static bool compareKeyEnd(size_t key, const PreloadedRange &range) {
            return key < range.keyEnd;
        }
    };

    // a node answers for its own cluster part, and for the entries of preloadKeyRanges in the other parts that fit in partMemorySize; the own part gets what is left
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
                const size_t bytes = (range.endEntry - range.firstEntry + 1) * 2 * sizeof(uint64_t) + endPair[1] - firstPair[1];
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
            memberIndexReadSize += partIndexSize;
            memberIndexReadCount++;
            if (localClu == clu && rangeSize + partIndexSize + cluDbr->getDataSize() <= partMemorySize) {
                range.data.resize(range.indexPairs.back() - range.indexPairs[1]);
                FileUtil::preadAll(cluDbr->getBufferedFd(), cluDbr->getDataFileName(), range.data.data(), range.data.size(), range.indexPairs[1]);
                memberLookupReadSize += range.data.size();
                memberLookupReadCount++;
            }
            rangeSize += partIndexSize + range.data.size();
        } else {
            sampledKeys.resize((cluDbr->getSize() + entriesPerSampledKey - 1) / entriesPerSampledKey);
            size_t readSize = 0;
            size_t readCount = 0;
            cluDbr->readSampledKeys(entriesPerSampledKey, std::max(BatchReader::MAX_READ_REQUEST_SIZE, partMemorySize), sampledKeys.data(), readSize, readCount);
            memberIndexReadSize += readSize;
            memberIndexReadCount += readCount;
        }
    }

    // reads range i if it is in another part from the shared folder
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

    // the node whose cluster part holds key; it grows with key
    size_t getNode(DBKeyType key) const {
        return std::upper_bound(cluKeyOffsets.begin(), cluKeyOffsets.end(), key) - cluKeyOffsets.begin() - 1;
    }

    // search the index pairs of the range of key in memory, or those between two sampled keys
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
            // the entry is between the sampled keys around key, it ends where the next one starts
            size_t entryCount = cluDbr->getSize();
            const size_t first = (sampledKey > 0) ? (sampledKey - 1) * entriesPerSampledKey : 0;
            const size_t last = std::min(sampledKey * entriesPerSampledKey + 1, entryCount);
            const size_t pairEnd = std::min(last + 2, entryCount + 1);
            if (first < entryReader.indexFirst || pairEnd > entryReader.indexFirst + entryReader.indexPairs.size() / 2) {
                entryReader.indexFirst = first;
                entryReader.indexPairs.resize(2 * (pairEnd - first));
                cluDbr->readIndexPairs(first, pairEnd, entryReader.indexPairs.data());
                entryReader.indexReadBytes += entryReader.indexPairs.size() * sizeof(uint64_t);
                entryReader.indexReadCount++;
            }
            indexPairs = entryReader.indexPairs.data() + 2 * (first - entryReader.indexFirst);
            count = last - first;
        }
        size_t found = 0;
        while (count > 0) {
            const size_t half = count / 2;
            if (indexPairs[2 * (found + half)] < key) {
                found += half + 1;
                count -= half + 1;
            } else {
                count = half;
            }
        }
        // the pair behind the searched ones has another key
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
        return sizeof(*this) + cluKeyOffsets.capacity() * sizeof(size_t) + sizeof(DBReader<DBKeyType>) + sampledKeys.capacity() * sizeof(DBKeyType) + rangeSize;
    }

    // the member buffer and index pairs of an EntryReader; find reads at most entriesPerSampledKey + 3 index pairs
    size_t getEntryReaderSize() const {
        return MEMBER_READ_SIZE + (entriesPerSampledKey + 3) * 2 * sizeof(uint64_t);
    }

    // the preloaded bytes of the target splits in the other parts, all their bytes, and seconds to read them
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
    // in key order, the own part too if it fits; their bytes
    std::vector<PreloadedRange> preloadedRanges;
    size_t rangeSize;
};

// --filter-cludb-file on several nodes: each node answers for its own cluster part through exchange files
struct ClusterMemberExchange {
    ClusterMemberExchange(const std::string &requests, const std::string &served, ClusterMemberLookup &lookup, int threads, size_t firstAlignNode)
        : requests(requests), served(served), lookup(lookup), threads(threads), requestCount(0), memberFileCount(0), isSizeRequest(false),
          nextMemberLines(MMseqsNode::count, 0), isAnswered(MMseqsNode::count, 1), nodeCodes(MMseqsNode::count),
          nodeConsumedBytes(MMseqsNode::count, 0), memberLines(std::max<size_t>(16, 2 * MMseqsNode::count) * MEMBER_LINE_WORDS), isReceiving(false),
          firstAlignNode(firstAlignNode), requestLines(16 * (MMseqsNode::count + 6)), nextRequestLines(MMseqsNode::count, 0), isRequesting(MMseqsNode::count, 0), requesterCount(0),
          nextRead(0), isPreloadDone(lookup.preloadSize == 0), nextPreloadedRange(0),
          requestedKeyCount(0), servedKeyCount(0), servedSize(0), sizeWaitSeconds(0), memberWaitSeconds(0), preloadWaitSeconds(0) {
        for (size_t node = firstAlignNode; node < MMseqsNode::count; node++) {
            isRequesting[node] = node != MMseqsNode::rank;
            requesterCount += isRequesting[node];
        }
        // an idle requester is read every idleSeconds, one that asked after 10 ms and then at doubling intervals;
        // a size request is followed by the member request of the same fetch, so its requester is read every 10 ms for up to a second
        idleSeconds = std::max(0.04, (double) requesterCount / PROGRESS_READS_PER_SECOND);
        readSeconds.assign(MMseqsNode::count, idleSeconds);
        nextReads.assign(MMseqsNode::count, 0);
        fastReadEnds.assign(MMseqsNode::count, 0);
    }

    // for ascending keys: entry offsets and bytes (maxMembers 0), their members (SIZE_MAX), or up to maxMembers members of one entry after consumedBytes
    void request(const std::vector<DBKeyType> &keys, const size_t *entryOffsets, const size_t *entryBytes, size_t consumedBytes, size_t maxMembers) {
        const unsigned int nodes = MMseqsNode::count;
        const unsigned int rank = MMseqsNode::rank;
        isSizeRequest = maxMembers == 0;
        // the cluster parts hold ascending key ranges, so the keys of a node follow each other
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
            // request line: request, isLast, consumed bytes, max members, key offsets per node, end, magic; the keys of a node are sent as gaps
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
        waitForPreload();
        answer(keys.data() + localBegin, localEnd - localBegin, (maxMembers != 0) ? entryOffsets + localBegin : NULL, (maxMembers != 0) ? entryBytes + localBegin : NULL,
               consumedBytes, maxMembers, nodeCodes[rank], nodeConsumedBytes[rank], keyNodes.data() + localBegin);
    }

    // a read of the answers to the last request from the first node that owes one; true once all answered
    bool tryReceive() {
        const unsigned int nodes = MMseqsNode::count;
        const unsigned int rank = MMseqsNode::rank;
        if (isReceiving == false) {
            isReceiving = true;
            receiveTimer.reset();
            receiveWarning.reset();
        }
        // the owing nodes answer about together, so they are polled one after the other
        for (unsigned int node = 0; node < nodes;) {
            if (isAnswered[node]) {
                node++;
                continue;
            }
            const size_t lineCount = MMseqsNode::readExchangeIndexLines(getMemberIndex(node), nextMemberLines[node], MEMBER_LINE_WORDS, memberLines);
            nextMemberLines[node] += lineCount;
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
                MMseqsNode::releaseExchangeCopy(fd);
                // another node wrote it, its close has no error of this node
                close(fd);
                // only this node reads it
                MMseqsNode::removeExchangeFile(memberFile, line[0]);
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

    // the entry offsets and bytes of the keys of the last request; the answers are released once read
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

    // members[memberOffsets[i], memberOffsets[i + 1]) belong to the key i of the last request; the answers are released once read
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

    // the consumed bytes of the entry of the one key of the last request
    size_t getConsumedBytes() const {
        return nodeConsumedBytes[keyNodes[0]];
    }

    // no more requests from this node
    void finish() {
        const unsigned int nodes = MMseqsNode::count;
        waitForPreload();
        std::vector<uint64_t> line(nodes + 6, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES);
        line[0] = requestCount;
        line[1] = 1;
        MMseqsNode::writeExchangeFile(getRequestFile(MMseqsNode::rank), requestCount, NULL, 0, getRequestIndex(MMseqsNode::rank), line, 1);
        requestCount++;
    }

    // the requests of the aligning nodes that are due to be read, one at a time; false if no line was read
    bool serveRequests() {
        std::unique_lock<std::mutex> taskLock(serveMutex, std::try_to_lock);
        if (taskLock.owns_lock() == false) {
            return false;
        }
        const unsigned int nodes = MMseqsNode::count;
        const unsigned int rank = MMseqsNode::rank;
        const size_t requestLineWords = nodes + 6;
        bool hasRead = false;
        nextRead = serveTimer.getTimediff() + idleSeconds;
        for (unsigned int requester = firstAlignNode; requester < nodes; requester++) {
            if (isRequesting[requester] == 0) {
                continue;
            }
            nextReads[requester] = std::min(nextReads[requester], serveTimer.getTimediff() + idleSeconds);
            if (serveTimer.getTimediff() < nextReads[requester]) {
                nextRead = std::min(nextRead, nextReads[requester]);
                continue;
            }
            const size_t lineCount = MMseqsNode::readExchangeIndexLines(getRequestIndex(requester), nextRequestLines[requester], requestLineWords, requestLines);
            nextRequestLines[requester] += lineCount;
            hasRead |= lineCount > 0;
            if (lineCount > 0) {
                fastReadEnds[requester] = (requestLines[(lineCount - 1) * requestLineWords + 3] == 0) ? serveTimer.getTimediff() + 1.0 : 0;
            }
            readSeconds[requester] = (lineCount > 0 || serveTimer.getTimediff() < fastReadEnds[requester]) ? 0.01 : std::min(2.0 * readSeconds[requester], idleSeconds);
            nextReads[requester] = serveTimer.getTimediff() + readSeconds[requester];
            nextRead = std::min(nextRead, nextReads[requester]);
            for (size_t l = 0; l < lineCount; l++) {
                const uint64_t *line = requestLines.data() + l * requestLineWords;
                const size_t sectionBegin = line[4 + rank];
                const size_t sectionEnd = line[5 + rank];
                if (sectionEnd > sectionBegin) {
                    const std::string requestFile = getRequestFile(requester);
                    const int fd = MMseqsNode::waitForExchangeFile(requestFile, line[4 + nodes], line[0]);
                    section.resize(sectionEnd - sectionBegin);
                    FileUtil::preadAll(fd, requestFile.c_str(), section.data(), section.size(), sectionBegin);
                    MMseqsNode::releaseExchangeCopy(fd);
                    // another node wrote it, its close has no error of this node
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
                    servedKeyCount += requestedKeys.size();
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

    // answers the requests of the aligning nodes until every one of them called finish
    void serve() {
        while (isServed() == false) {
            if (serveRequests() == false) {
                const double waitSeconds = nextRead - serveTimer.getTimediff();
                if (waitSeconds > 0) {
                    usleep(static_cast<useconds_t>(waitSeconds * 1000000));
                }
            }
        }
    }

    bool isServed() const {
        return requesterCount == 0;
    }

    // reads the next preloaded range; false once all are read
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

    // per key its entry offset and bytes, or its members plus one ended by 0; no state, the serving and the aligning thread call it at once
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
            // a static schedule gives every thread one run of ascending keys
#pragma omp for schedule(static)
            for (size_t i = 0; i < answeredKeyCount; i++) {
                if (answeringNodes != NULL && answeringNodes[i] != MMseqsNode::rank) {
                    continue;
                }
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
    std::vector<unsigned char> isAnswered;
    std::vector<std::string> nodeCodes;
    std::vector<size_t> nodeConsumedBytes;
    std::string requestCodes;
    // the node that answers each key of the last request
    std::vector<unsigned int> keyNodes;
    // a node answers every requester between two answers to this node
    std::vector<uint64_t> memberLines;
    bool isReceiving;
    Timer receiveTimer;
    MMseqsNode::WaitWarning receiveWarning;
    const size_t firstAlignNode;
    std::mutex serveMutex;
    std::vector<uint64_t> requestLines;
    std::vector<size_t> nextRequestLines;
    std::vector<unsigned char> isRequesting;
    std::atomic<size_t> requesterCount;
    double idleSeconds;
    std::vector<double> readSeconds;
    std::vector<double> nextReads;
    std::vector<double> fastReadEnds;
    Timer serveTimer;
    double nextRead;
    std::vector<char> section;
    std::vector<DBKeyType> requestedKeys;
    std::vector<size_t> requestedOffsets;
    std::vector<size_t> requestedBytes;
    std::string reply;
    // read while the first fetches align, the first request waits for them
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

public:
    // the keys this node asked the other nodes for, and the keys and bytes it served; read once serve is done
    size_t requestedKeyCount;
    size_t servedKeyCount;
    size_t servedSize;
    // seconds in receive, and before the first request
    double sizeWaitSeconds;
    double memberWaitSeconds;
    double preloadWaitSeconds;

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
    // target split mode: the window of the entries
    size_t window;
};

// a started pref read whose entries are counted, then decoded, in chunks; prefRead SIZE_MAX: none
struct PrefReadParse {
    PrefReadParse() : prefRead(SIZE_MAX), firstEntry(0), entryCount(0), nextEntry(0), isCounted(false), firstRecord(0), firstTarget(0), size(0), maxEntrySize(0), fetchSize(0), alignCells(0) {}
    size_t prefRead;
    size_t firstEntry;
    size_t entryCount;
    size_t nextEntry;
    bool isCounted;
    size_t firstRecord;
    size_t firstTarget;
    // pref bytes of the pref read and of its largest entry
    size_t size;
    size_t maxEntrySize;
    // added by the chunks
    size_t fetchSize;
    size_t alignCells;
};

// the prefilter fills of an align node, for the log; written by the thread that owns the fill
struct PrefilterFillCounters {
    static const int READ_STEP = 0;
    static const int PARSE_STEP = 1;
    static const int SORT_STEP = 2;

    PrefilterFillCounters() : bufferCount(0), mainBufferCount(0), orderedBufferCount(0), entryCount(0), maxBufferEntryCount(0), maxEntryShare(0), lastStepKind(READ_STEP),
                              bufferEntryCount(0), bufferPrefSize(0), bufferMaxEntrySize(0) {
        for (int kind = 0; kind < 3; kind++) {
            stepSeconds[kind] = 0;
            blockedSeconds[kind] = 0;
        }
        startDelaySeconds[0] = startDelaySeconds[1] = 0;
        startedBufferCount[0] = startedBufferCount[1] = 0;
    }

    void startBuffer() {
        bufferEntryCount = 0;
        bufferPrefSize = 0;
        bufferMaxEntrySize = 0;
    }

    void addPrefRead(const PrefReadParse &parse) {
        bufferEntryCount += parse.entryCount;
        bufferPrefSize += parse.size;
        bufferMaxEntrySize = std::max(bufferMaxEntrySize, parse.maxEntrySize);
    }

    void addBuffer(bool isOrdered) {
        bufferCount++;
        orderedBufferCount += isOrdered;
        entryCount += bufferEntryCount;
        maxBufferEntryCount = std::max(maxBufferEntryCount, bufferEntryCount);
        maxEntryShare = std::max(maxEntryShare, bufferMaxEntrySize * 1000 / std::max<size_t>(1, bufferPrefSize));
    }

    size_t bufferCount;
    size_t mainBufferCount;
    size_t orderedBufferCount;
    size_t entryCount;
    size_t maxBufferEntryCount;
    // per mille of the pref bytes of its buffer
    size_t maxEntryShare;
    double stepSeconds[3];
    // the main thread waited for a step of an I/O thread
    double blockedSeconds[3];
    // from a buffer switch to the first step of the next fill, started by an I/O thread or by the waiting main thread
    double startDelaySeconds[2];
    size_t startedBufferCount[2];
    int lastStepKind;
    // the buffer being filled
    size_t bufferEntryCount;
    size_t bufferPrefSize;
    size_t bufferMaxEntrySize;
};

// one align node: the prefilter of the alignment splits in key order
struct RepTargetsReader {
    RepTargetsReader(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<PrefPartition> &partitions, const std::vector<size_t> &alignmentSplitPartitionOffsets,
                     size_t maxRecordSize, size_t memoryLimit, size_t paddedDbSize, size_t sizeClassCount)
        : maxRecordSize(maxRecordSize), par(par), seqDbr(seqDbr), partitions(partitions), alignmentSplitPartitionOffsets(alignmentSplitPartitionOffsets),
          alignmentSplit((MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) ? 0 : alignmentSplitPartitionOffsets.size() - 1),
          paddedDbSize(paddedDbSize), sizeClassCount(sizeClassCount), maxPrefilterBufferSize(std::max<size_t>(1, memoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR)),
          maxRecordTargetCount(std::min<size_t>(UINT_MAX, std::max<size_t>(1, maxRecordSize / std::max<size_t>(seqDbr.getMaxSeqLen() + 2, Util::getPageSize())))),
          prefReadPos(0), passClasses(~0ULL), endClass(0), classSize(sizeClassCount, 0), prefDb(NULL), prefNode(0), bufferRecordCount(0), bufferTargetCount(0), bufferFetchSize(0), bufferAlignmentCells(0),
          entryTargetOffsets(NULL), entryRecordOffsets(NULL), allocatedEntries(0), isFilling(false), isLoaded(false), fillMaxReadSize(0), fillMaxAlignmentCells(0) {}

    bool takeAlignmentSplit() {
        const size_t prefReadsBefore = prefReads.size();
        for (; prefReads.size() == prefReadsBefore && alignmentSplit + 1 < alignmentSplitPartitionOffsets.size(); alignmentSplit++) {
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
                // per pref byte two buffers and 8 bytes of targets, per entry 48 bytes and a record
                const size_t maxPrefReadSize = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(maxPrefilterBufferSize) / (10.0 + (48.0 + sizeof(RepTargetsRecord)) * entriesPerByte)));
                const size_t firstPrefRead = prefReads.size();
                for (size_t offset = prefBegin; offset < prefEnd; offset += maxPrefReadSize) {
                    const size_t entry = prefDb->findEntryByOffset(offset);
                    if (prefReads.size() == firstPrefRead || prefReads.back().firstEntry != entry) {
                        const PrefRead prefRead = { node, entry, entry, ~0ULL, 0 };
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

    // one step with isOneStep (a pref read start with its read, an entry chunk or the sort), else all steps by the caller's team; true once complete
    bool fillPrefilterBuffer(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, size_t maxReadSize, size_t maxBufferAlignmentCells, bool isOneStep) {
        if (isFilling == false) {
            records.reserve(maxPrefilterBufferSize / sizeof(RepTargetsRecord));
            targets.reserve(maxPrefilterBufferSize / sizeof(DBKeyType));
            bufferRecordCount = 0;
            bufferTargetCount = 0;
            bufferFetchSize = 0;
            bufferAlignmentCells = 0;
            // a resumed fill keeps the limits it started with
            fillMaxReadSize = maxReadSize;
            fillMaxAlignmentCells = maxBufferAlignmentCells;
            fillCounters.startBuffer();
            isFilling = true;
        }
        do {
            Timer stepTimer;
            if (parse.prefRead != SIZE_MAX) {
                fillCounters.lastStepKind = PrefilterFillCounters::PARSE_STEP;
                if (parsePrefRead(records, targets, isOneStep)) {
                    fillCounters.addPrefRead(parse);
                }
            } else if (isLoaded == false) {
                fillCounters.lastStepKind = PrefilterFillCounters::READ_STEP;
                for (isLoaded = true; bufferRecordCount * sizeof(RepTargetsRecord) + bufferTargetCount * sizeof(DBKeyType) < maxPrefilterBufferSize;) {
                    if (prefReadPos < prefReads.size()) {
                        startPrefRead();
                        isLoaded = false;
                        break;
                    } else if (bufferRecordCount > 0 && (bufferFetchSize >= fillMaxReadSize || bufferAlignmentCells >= fillMaxAlignmentCells)) {
                        break;
                    } else if (takeAlignmentSplit() == false && (bufferRecordCount > 0 || startPass() == false)) {
                        break;
                    }
                }
            } else {
                fillCounters.lastStepKind = PrefilterFillCounters::SORT_STEP;
                records.resize(bufferRecordCount);
                targets.resize(bufferTargetCount);
                // the order is total, an ordered buffer needs no sort
                const bool isOrdered = std::is_sorted(records.begin(), records.end(), RepTargetsRecord::compareByRep);
                if (isOrdered == false) {
                    SORT_PARALLEL(records.begin(), records.end(), RepTargetsRecord::compareByRep);
                }
                fillCounters.addBuffer(isOrdered);
                isFilling = false;
                isLoaded = false;
            }
            fillCounters.stepSeconds[fillCounters.lastStepKind] += stepTimer.getTimediff();
        } while (isFilling && isOneStep == false);
        return isFilling == false;
    }

    void close() {
        closePrefDb();
        delete[] entryTargetOffsets;
        delete[] entryRecordOffsets;
    }

    const size_t maxRecordSize;
    PrefilterFillCounters fillCounters;

private:
    Parameters &par;
    DBReader<DBKeyType> &seqDbr;
    const std::vector<PrefPartition> &partitions;
    const std::vector<size_t> &alignmentSplitPartitionOffsets;
    size_t alignmentSplit;
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
    // a buffer whose fill goes on in later calls, the limits it started with and its pref read in progress
    bool isFilling;
    bool isLoaded;
    size_t fillMaxReadSize;
    size_t fillMaxAlignmentCells;
    PrefReadParse parse;

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
        startPrefRead();
        parsePrefRead(records, targets, false);
    }

    // reads the next pref read of the pass, prefetches the one behind it and sizes the entry offsets
    void startPrefRead() {
        const PrefRead &prefRead = prefReads[prefReadPos];
        parse = PrefReadParse();
        parse.prefRead = prefReadPos;
        parse.firstEntry = prefRead.firstEntry;
        parse.entryCount = prefRead.endEntry - prefRead.firstEntry;
        openPrefDb(prefRead.prefNode);
        prefDb->loadData(prefRead.firstEntry, prefRead.endEntry);
        prefReadPos = findPassPrefRead(prefReadPos + 1);
        // prefetch here, io_uring reads can fail once their thread ends
        if (prefReadPos < prefReads.size() && prefReads[prefReadPos].prefNode == prefRead.prefNode) {
            prefDb->prefetchData(prefReads[prefReadPos].firstEntry, prefReads[prefReadPos].endEntry);
        }
        for (size_t entry = parse.firstEntry; entry < parse.firstEntry + parse.entryCount; entry++) {
            const size_t entrySize = prefDb->getEntryLen(entry);
            parse.size += entrySize;
            parse.maxEntrySize = std::max(parse.maxEntrySize, entrySize);
        }
        if (allocatedEntries < parse.entryCount + 1) {
            delete[] entryTargetOffsets;
            delete[] entryRecordOffsets;
            allocatedEntries = parse.entryCount + 1;
            entryTargetOffsets = new(std::nothrow) size_t[allocatedEntries];
            entryRecordOffsets = new(std::nothrow) size_t[allocatedEntries];
            Util::checkAllocation(entryTargetOffsets, "Can not allocate entryTargetOffsets memory in align2clust");
            Util::checkAllocation(entryRecordOffsets, "Can not allocate entryRecordOffsets memory in align2clust");
        }
        entryTargetOffsets[0] = 0;
        entryRecordOffsets[0] = 0;
    }

    // counts, then decodes the entries of the started pref read, a chunk a call with isOneStep; true once its records are in the buffer
    bool parsePrefRead(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, bool isOneStep) {
        const size_t firstEntry = parse.firstEntry;
        const size_t entryCount = parse.entryCount;
        if (parse.isCounted == false) {
            parse.nextEntry = forEachEntryChunk(parse.nextEntry, entryCount, 100, isOneStep, [&](size_t begin, size_t end) {
                for (size_t entry = begin; entry < end; entry++) {
                    const char *data = prefDb->getData(firstEntry + entry, 0);
                    const size_t targetCount = std::count_if(data, data + prefDb->getEntryLen(firstEntry + entry) - 1, [](char byte) { return static_cast<unsigned char>(byte) < 0x80; }) - 1;
                    const bool isInPass = ((passClasses >> getSizeClass(targetCount, sizeClassCount)) & 1) != 0;
                    entryTargetOffsets[entry + 1] = isInPass ? targetCount : 0;
                    entryRecordOffsets[entry + 1] = isInPass ? (targetCount + maxRecordTargetCount - 1) / maxRecordTargetCount : 0;
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
        if (parse.nextEntry == 0) {
            sumOffsets(entryTargetOffsets, entryCount, par.threads);
            sumOffsets(entryRecordOffsets, entryCount, par.threads);
            parse.firstRecord = bufferRecordCount;
            parse.firstTarget = bufferTargetCount;
            bufferRecordCount += entryRecordOffsets[entryCount];
            bufferTargetCount += entryTargetOffsets[entryCount];
            records.resize(std::max(records.size(), bufferRecordCount));
            targets.resize(std::max(targets.size(), bufferTargetCount));
        }
        const size_t firstRecord = parse.firstRecord;
        const size_t firstTarget = parse.firstTarget;
        parse.nextEntry = forEachEntryChunk(parse.nextEntry, entryCount, 100, isOneStep, [&](size_t begin, size_t end) {
            size_t prefReadFetchSize = 0;
            for (size_t entry = begin; entry < end; entry++) {
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
            __sync_fetch_and_add(&parse.fetchSize, prefReadFetchSize);
        });
        if (parse.nextEntry < entryCount) {
            return false;
        }
        bufferFetchSize += parse.fetchSize;
        parse.prefRead = SIZE_MAX;
        return true;
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
    TargetSplitReader(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<PrefWindow> &windows, size_t maxRecordSize, size_t memoryLimit,
                      size_t paddedDbSize, size_t sizeClassCount)
        : readSize(0), prefSize(0), maxEntrySize(0), windowCells(windows.size(), 0), par(par), seqDbr(seqDbr), windows(windows),
          targetSplitCount(targetSplitBlockOffsets.size() - 1), paddedDbSize(paddedDbSize), sizeClassCount(sizeClassCount),
          maxPrefilterBufferSize(std::max((size_t) 1, memoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR)),
          maxRecordTargetCount(std::min((size_t) UINT_MAX, std::max((size_t) 1, maxRecordSize / std::max(seqDbr.getMaxSeqLen() + 2, Util::getPageSize())))),
          windowPos(0), prefReadPos(0), passClasses(1), endClass(0), classSize(sizeClassCount, 0), prefNode(SIZE_MAX), bufferFetchSize(0), bufferAlignmentCells(0),
          isFilling(false), isLoaded(false), fillMaxReadSize(0), fillMaxAlignmentCells(0) {
        for (size_t split = 0; split < targetSplitCount; split++) {
            if (targetSplitAlignNodes[split] == MMseqsNode::rank) {
                targetSplits.push_back(split);
                targetSplitFirstKeys.push_back(seqDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[split]]);
            }
        }
        prefOffsets.resize(MMseqsNode::count * targetSplitCount);
    }

    // load every window once and mark its size classes
    void countSizeClasses() {
        std::vector<RepTargetsRecord> records;
        std::vector<DBKeyType> targets;
        passClasses = ~0ULL;
        for (prefReadPos = 0; prefReadPos < prefReads.size() || takeWindow();) {
            // loading takes the next window ahead
            const size_t prefRead = prefReadPos;
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
            prefReads[prefRead].sizeClasses = sizeClasses;
        }
        endClass = sizeClassCount;
    }

    // one step with isOneStep (a pref read start with its reads, an entry chunk or the sort), else all steps by the caller's team; true once complete
    bool fillPrefilterBuffer(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, size_t maxReadSize, size_t maxBufferAlignmentCells, bool isOneStep) {
        if (isFilling == false) {
            records.clear();
            targets.clear();
            bufferFetchSize = 0;
            bufferAlignmentCells = 0;
            // a resumed fill keeps the limits it started with
            fillMaxReadSize = maxReadSize;
            fillMaxAlignmentCells = maxBufferAlignmentCells;
            fillCounters.startBuffer();
            isFilling = true;
        }
        do {
            Timer stepTimer;
            if (parse.prefRead != SIZE_MAX) {
                fillCounters.lastStepKind = PrefilterFillCounters::PARSE_STEP;
                const size_t window = prefReads[parse.prefRead].window;
                if (parsePrefRead(records, targets, isOneStep)) {
                    windowCells[window] += parse.alignCells;
                    fillCounters.addPrefRead(parse);
                }
            } else if (isLoaded == false) {
                fillCounters.lastStepKind = PrefilterFillCounters::READ_STEP;
                for (isLoaded = true; records.size() * sizeof(RepTargetsRecord) + targets.size() * sizeof(DBKeyType) < maxPrefilterBufferSize;) {
                    if (prefReadPos < prefReads.size() || takeWindow()) {
                        // keep a pass of several size classes in one buffer
                        if (records.empty() == false && (passClasses & (passClasses - 1)) == 0 && (bufferFetchSize >= fillMaxReadSize || bufferAlignmentCells >= fillMaxAlignmentCells)) {
                            break;
                        }
                        startPrefRead();
                        isLoaded = false;
                        break;
                    } else if (records.empty() == false || startPass() == false) {
                        break;
                    }
                }
            } else {
                fillCounters.lastStepKind = PrefilterFillCounters::SORT_STEP;
                // the order is total, an ordered buffer needs no sort
                const bool isOrdered = std::is_sorted(records.begin(), records.end(), RepTargetsRecord::compareByRep);
                if (isOrdered == false) {
                    SORT_PARALLEL(records.begin(), records.end(), RepTargetsRecord::compareByRep);
                }
                fillCounters.addBuffer(isOrdered);
                isFilling = false;
                isLoaded = false;
            }
            fillCounters.stepSeconds[fillCounters.lastStepKind] += stepTimer.getTimediff();
        } while (isFilling && isOneStep == false);
        return isFilling == false;
    }

    void close() {
        closePrefDbs();
    }

    size_t readSize;
    size_t prefSize;
    // this node's bytes of its largest entry
    size_t maxEntrySize;
    // the cells this node loaded per window
    std::vector<size_t> windowCells;
    PrefilterFillCounters fillCounters;

private:
    Parameters &par;
    DBReader<DBKeyType> &seqDbr;
    const std::vector<PrefWindow> &windows;
    const size_t targetSplitCount;
    const size_t paddedDbSize;
    const size_t sizeClassCount;
    const size_t maxPrefilterBufferSize;
    const size_t maxRecordTargetCount;
    std::vector<size_t> targetSplits;
    std::vector<DBKeyType> targetSplitFirstKeys;
    size_t windowPos;
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
    // a buffer whose fill goes on in later calls, the limits it started with and its pref read in progress
    bool isFilling;
    bool isLoaded;
    size_t fillMaxReadSize;
    size_t fillMaxAlignmentCells;
    PrefReadParse parse;

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

    // a window has the same entries in every target split of its node
    bool takeWindow() {
        const size_t maxPrefReadSize = getMaxPrefReadSize(maxPrefilterBufferSize);
        const size_t prefReadsBefore = prefReads.size();
        for (; prefReads.size() == prefReadsBefore && windowPos < windows.size(); windowPos++) {
            const PrefWindow &window = windows[windowPos];
            // only the windows of the own align group
            if (windowAlignGroups[windowPos] != alignGroup || window.alignmentCells == 0) {
                continue;
            }
            // read each index once
            std::vector<uint64_t> *offsets = prefOffsets.data() + window.node * targetSplitCount;
            for (size_t i = 0; i < targetSplits.size(); i++) {
                std::vector<uint64_t> &splitOffsets = offsets[targetSplits[i]];
                if (splitOffsets.empty()) {
                    std::string index = MMseqsNode::readFile(par.db2 + ".index." + SSTR(window.node * targetSplitCount + targetSplits[i]));
                    splitOffsets.resize(index.size() / (2 * sizeof(uint64_t)));
                    for (size_t entry = 0; entry < splitOffsets.size(); entry++) {
                        memcpy(&splitOffsets[entry], index.data() + (2 * entry + 1) * sizeof(uint64_t), sizeof(uint64_t));
                    }
                }
            }
            const size_t entryCount = window.endEntry - window.firstEntry;
            entrySizes.assign(entryCount, 0);
            for (size_t i = 0; i < targetSplits.size(); i++) {
                const std::vector<uint64_t> &splitOffsets = offsets[targetSplits[i]];
                for (size_t entry = 0; entry < entryCount; entry++) {
                    entrySizes[entry] += splitOffsets[window.firstEntry + entry + 1] - splitOffsets[window.firstEntry + entry];
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
                prefSize += entrySizes[entry];
                maxEntrySize = std::max(maxEntrySize, entrySizes[entry]);
            }
        }
        return prefReads.size() > prefReadsBefore;
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
        startPrefRead();
        parsePrefRead(records, targets, false);
    }

    // reads the next pref read of the pass in every target split, prefetches the one behind it and sizes the entry offsets
    void startPrefRead() {
        const PrefRead prefRead = prefReads[prefReadPos];
        parse = PrefReadParse();
        parse.prefRead = prefReadPos;
        parse.firstEntry = prefRead.firstEntry;
        parse.entryCount = prefRead.endEntry - prefRead.firstEntry;
        openPrefDbs(prefRead.prefNode);
        for (size_t i = 0; i < prefDbs.size(); i++) {
            prefDbs[i]->loadData(prefRead.firstEntry, prefRead.endEntry);
        }
        prefReadPos = findPassPrefRead(prefReadPos + 1);
        // the last pref read of a window takes the next window, so that its first read is prefetched
        if (prefReadPos == prefReads.size()) {
            takeWindow();
        }
        // prefetch here, io_uring reads can fail once their thread ends
        if (prefReadPos < prefReads.size() && prefReads[prefReadPos].prefNode == prefRead.prefNode) {
            for (size_t i = 0; i < prefDbs.size(); i++) {
                prefDbs[i]->prefetchData(prefReads[prefReadPos].firstEntry, prefReads[prefReadPos].endEntry);
            }
        }
        for (size_t entry = parse.firstEntry; entry < parse.firstEntry + parse.entryCount; entry++) {
            size_t entrySize = 0;
            for (size_t i = 0; i < prefDbs.size(); i++) {
                entrySize += prefDbs[i]->getEntryLen(entry);
            }
            parse.size += entrySize;
            parse.maxEntrySize = std::max(parse.maxEntrySize, entrySize);
        }
        readSize += parse.size;
        entryTargetOffsets.resize(parse.entryCount + 1);
        entryRecordOffsets.resize(parse.entryCount + 1);
        entryTargetOffsets[0] = 0;
        entryRecordOffsets[0] = 0;
    }

    // counts the targets and records of every entry, then writes the records of an entry at its offsets, a chunk a call with isOneStep; true once they are in the buffer
    bool parsePrefRead(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, bool isOneStep) {
        const size_t firstEntry = parse.firstEntry;
        const size_t entryCount = parse.entryCount;
        if (parse.isCounted == false) {
            parse.nextEntry = forEachEntryChunk(parse.nextEntry, entryCount, 1, isOneStep, [&](size_t begin, size_t end) {
                std::vector<TargetSplitPiece> pieces(prefDbs.size());
                for (size_t entry = begin; entry < end; entry++) {
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
        if (parse.nextEntry == 0) {
            sumOffsets(entryTargetOffsets.data(), entryCount, par.threads);
            sumOffsets(entryRecordOffsets.data(), entryCount, par.threads);
            parse.firstRecord = records.size();
            parse.firstTarget = targets.size();
            records.resize(parse.firstRecord + entryRecordOffsets[entryCount]);
            targets.resize(parse.firstTarget + entryTargetOffsets[entryCount]);
        }
        const size_t firstRecord = parse.firstRecord;
        const size_t firstTarget = parse.firstTarget;
        parse.nextEntry = forEachEntryChunk(parse.nextEntry, entryCount, 1, isOneStep, [&](size_t begin, size_t end) {
            std::vector<TargetSplitPiece> pieces(prefDbs.size());
            size_t fetchSize = 0;
            size_t alignCells = 0;
            for (size_t entry = begin; entry < end; entry++) {
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
        if (parse.nextEntry < entryCount) {
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

// the members outside the SWIPE lanes are among the first and last 3 in the order of prepareMemberChecks, (-length difference, list position)
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

// the accepted pairs of a node, in rep order
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
    // bits of the log words from 64 * firstBitsWord on, written once the words before them are clustered
    std::vector<uint64_t> bits;
    size_t firstBitsWord;
    char *buffer;
};

// linfinity: the master clusters the accepted pairs in key order, a step at a time
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
          nodeAcceptedLogs(MMseqsNode::count), stepWords(getStepWords(memoryLimit)), nodeStepWords(std::max<size_t>(1, stepWords / MMseqsNode::count)),
          logReadBytesPerNode(std::max<size_t>(1, std::min(stepWords, memoryLimit / 64 / sizeof(uint64_t)) / MMseqsNode::count) * sizeof(uint64_t)),
          isWritingAssignedKeys(false), memberCount(0), pairFile(pairSets, setCoverPrefix + "_clustered_pairsets", stepWords, memoryLimit, assigned),
          smallSetFiles(setCoverPrefix + "_clustered_sets", memoryLimit, assigned), largeSetFiles(setCoverPrefix + "_clustered_large", memoryLimit), waitSeconds(0),
          clusteredOrderKey(0), isIdle(false) {
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
        }
        assignedKeysHandle = FileUtil::openAndDelete(assignedKeysLog.c_str(), "w");
        assignedKeysSize = 0;
    }

    int step() {
        // an idle step reads the logs again after the sleep of its rule
        if (isIdle && gapTimer.getTimediff() < idleMicroseconds / 1000000.0) {
            return STEP_IDLE;
        }
        // the time between two steps counts as waiting
        waitSeconds += gapTimer.getTimediff();
        gapTimer.reset();
        isIdle = false;
        if (clusteredOrderKey >= orderKeyEnd) {
            isDone = true;
            return STEP_DONE;
        }
        // write and publish the steps decided while the writer is busy
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
        const double pollCostSeconds = pollTimer.getTimediff();
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
                        const size_t repKey = getRepOrderKey(nodeWords[word]);
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
            stepRepOrderKey = std::min(stepRepOrderKey, nodeAcceptedLogs[node].nextRepKey);
        }
        size_t stepWordCount = 0;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            AcceptedLog &acceptedLog = nodeAcceptedLogs[node];
            std::vector<size_t> &repWordOffsets = acceptedLog.repWordOffsets;
            const std::vector<uint64_t> &nodeWords = acceptedLog.unclusteredWords;
            const size_t clusteredReps = std::lower_bound(repWordOffsets.begin(), repWordOffsets.end() - 1, stepRepOrderKey,
                [&](size_t word, size_t repKey) { return getRepOrderKey(nodeWords[word]) < repKey; }) - repWordOffsets.begin();
            if (clusteredReps + 1 < repWordOffsets.size()) {
                acceptedLog.nextRepKey = getRepOrderKey(nodeWords[repWordOffsets[clusteredReps]]);
            }
            acceptedLog.clusteredWordCount = repWordOffsets[clusteredReps];
            repWordOffsets.resize(clusteredReps);
            repWordOffsets.push_back(acceptedLog.clusteredWordCount);
            acceptedLog.wordOffset = stepWordCount;
            stepWordCount += acceptedLog.clusteredWordCount;
            // the clustered bit of a word is at its position in the log of the node
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
            const unsigned int log = nextReps.top().second;
            nextReps.pop();
            AcceptedLog &acceptedLog = nodeAcceptedLogs[log];
            const std::vector<size_t> &repWordOffsets = acceptedLog.repWordOffsets;
            const size_t endKey = nextReps.empty() ? stepRepOrderKey : nextReps.top().first;
            const size_t endRep = std::lower_bound(repWordOffsets.begin() + acceptedLog.mergedReps + 1, repWordOffsets.end() - 1, endKey,
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
                const size_t segmentEnd = std::min(end, (merged + 1 < mergedReps.size()) ? mergedReps[merged + 1].firstMergedRep : repCount);
                const std::vector<size_t> &repWordOffsets = nodeAcceptedLogs[logReps.log].repWordOffsets;
                for (; rep < segmentEnd; rep++) {
                    const size_t logRep = logReps.firstLogRep + rep - logReps.firstMergedRep;
                    const AcceptedRep acceptedRep = { getRepOrderKey(logWords[repWordOffsets[logRep]]), repWordOffsets[logRep] + 1, repWordOffsets[logRep + 1], logReps.log };
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
            // keyReps: the first rep of every key of the step
            size_t tableBits = 1;
            while ((1ULL << tableBits) < 2 * repKeyOffsets[repCount]) {
                tableBits++;
            }
            const size_t tableSize = 1ULL << tableBits;
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
                            const size_t bit = i - acceptedLog.wordOffset + acceptedLog.bitOffset;
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
                        const AcceptedLog &acceptedLog = nodeAcceptedLogs[reps[rep].log];
                        const size_t bit = i - acceptedLog.wordOffset + acceptedLog.bitOffset;
                        nodeClusteredLogs[reps[rep].log].bits[bit / 64] |= 1ULL << (bit % 64);
                        memberCount++;
                    }
                }
            }
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                // write the bits before the first word that is not clustered
                ClusteredLog &nodeFile = nodeClusteredLogs[node];
                const size_t wordCount = nodeAcceptedLogs[node].readSize / sizeof(uint64_t) / 64 - nodeFile.firstBitsWord;
                if (wordCount > 0 && fwrite(nodeFile.bits.data(), sizeof(uint64_t), wordCount, nodeFile.file) != wordCount) {
                    Debug(Debug::ERROR) << "Cannot write " << clusteredLogPrefix << "." << node << "\n";
                    EXIT(EXIT_FAILURE);
                }
                nodeFile.bits.erase(nodeFile.bits.begin(), nodeFile.bits.begin() + wordCount);
                nodeFile.firstBitsWord += wordCount;
                nodeFile.bytes += wordCount * sizeof(uint64_t);
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

    // the last steps and the logs; the clustered members
    size_t finish(std::string &phaseValues) {
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
            const std::vector<uint64_t> &bits = nodeClusteredLogs[node].bits;
            if (bits.empty() == false && fwrite(bits.data(), sizeof(uint64_t), bits.size(), nodeClusteredLogs[node].file) != bits.size()) {
                Debug(Debug::ERROR) << "Cannot write " << clusteredLogFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            MMseqsNode::publishLog(nodeClusteredLogs[node].file, clusteredLogFile, dbSize, nodeClusteredLogs[node].bytes + bits.size() * sizeof(uint64_t));
            if (fclose(nodeClusteredLogs[node].file) != 0) {
                Debug(Debug::ERROR) << "Cannot close " << clusteredLogFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            delete[] nodeClusteredLogs[node].buffer;
        }
        return memberCount;
    }

    // the sleep after an idle step and the end of the steps, read by the I/O threads of a single node
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
        const uint64_t bit = 1ULL << (key % 64);
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
        MMseqsNode::publishLog(assignedKeysHandle, assignedKeysLog, clusteredKey, assignedKeysSize);
        isWritingAssignedKeys = false;
    }

    void addClusterPair(DBKeyType repKey, DBKeyType memberKey) {
        const ClusterPair pair = { repKey, memberKey };
        const unsigned int pageNode = pageAlignNodes[repKey / ASSIGNED_PAGE_KEYS];
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
    // since the end of the last step, or the poll of the last idle step
    bool isIdle;
    Timer gapTimer;
};

// the master of several nodes, the members are served while it waits for the align nodes if no member server thread does it
static size_t clusterAcceptedPairs(int threads, size_t dbSize, size_t memoryLimit, const DBReader<DBKeyType> &seqDbr,
                                   const std::vector<unsigned int> &pageAlignNodes, int mode, size_t paddedDbSize, size_t sizeClassCount,
                                   const std::string &accepted, const std::string &assignedKeysLog, const std::string &clusteredLogPrefix, const std::string &setCoverPrefix,
                                   const std::string &assignedBits, ClusterMemberExchange *idleMemberExchange, bool hasWriterThread, std::string &phaseValues) {
    AcceptedPairClustering clustering(threads, dbSize, memoryLimit, seqDbr, pageAlignNodes, mode, paddedDbSize, sizeClassCount, accepted, assignedKeysLog, clusteredLogPrefix,
                                      setCoverPrefix, assignedBits, hasWriterThread);
    for (int step = clustering.step(); step != AcceptedPairClustering::STEP_DONE; step = clustering.step()) {
        if (step == AcceptedPairClustering::STEP_IDLE && (idleMemberExchange == NULL || idleMemberExchange->serveRequests() == false)) {
            usleep(clustering.idleMicroseconds);
        }
    }
    return clustering.finish(phaseValues);
}

static void alignRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, RepTargetsReader &repTargetsReader, TargetSplitReader *targetSplitReader,
                            const std::vector<PrefPartition> &partitions, const std::vector<PrefWindow> &windows, size_t memoryLimit, size_t masterMemoryLimit, size_t servedFileEntrySize, size_t representativeResidues,
                            size_t paddedDbSize, size_t sizeClassCount, const ClusterMemberLookup *memberLookup, ClusterMemberExchange *memberExchange,
                            size_t memberKeyBudget, DBReader<DBKeyType> *memberDbr, int ioThreadCount, AcceptedPairClustering *clustering, std::string &phaseValues) {
    Timer startTimer;
    const unsigned int nodes = MMseqsNode::count;
    const unsigned int rank = MMseqsNode::rank;
    const size_t firstAlignNode = MMseqsNode::getFirstAlignNode();
    const size_t alignNodeCount = nodes - firstAlignNode;
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
    size_t publishCount = 0;
    double publishCostSeconds = 0;
    std::atomic<double> publishPeriod(MIN_PUBLISH_SECONDS);
    std::atomic<size_t> publishedDoneRepKey(0);
    const auto publishAccepted = [&](size_t repKey, size_t bytes) {
        Timer publishTimer;
        publishCount++;
        MMseqsNode::publishLog(acceptedHandle, acceptedFile, repKey, bytes);
        publishedDoneRepKey.store(repKey);
        publishCostSeconds += publishTimer.getTimediff();
        publishPeriod.store(std::min<double>(PUBLISH_SECONDS, std::max(MIN_PUBLISH_SECONDS, LOG_COST_FACTOR * publishCostSeconds / publishCount)));
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
        if (publishedSize <= assignedKeysSize) {
            return;
        }
        // the published steps in one read
        assignedKeyWords.resize((publishedSize - assignedKeysSize) / sizeof(uint64_t));
        if (MMseqsNode::readLog(assignedKeysLog, assignedKeysSize, publishedSize, (char *) assignedKeyWords.data(), wait) == false) {
            return;
        }
        for (const uint64_t *step = assignedKeyWords.data(); assignedKeysSize < publishedSize;) {
            const size_t stepClusteredKey = step[0];
            const size_t stepClusteredSize = step[1 + MMseqsNode::rank];
            size_t keyCount = step[1 + MMseqsNode::count];
            size_t key = step[2 + MMseqsNode::count];
            unsigned int keyDiffLowBits = step[3 + MMseqsNode::count];
            size_t codeWords = step[4 + MMseqsNode::count];
            step += headerWords;
            KmerCodeReader codeRead = { (char *) step, 0, 0, 0, -1, assignedKeysLog.c_str(), codeWords * sizeof(uint64_t), 0, codeWords * sizeof(uint64_t) };
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
            step += codeWords;
            assignedKeysSize += (headerWords + codeWords) * sizeof(uint64_t);
            assignedKeyCount += keyCount;
            // the receiving pass reads it for the gate slack
            __atomic_store_n(&clusteredOrderKey, stepClusteredKey, __ATOMIC_RELAXED);
            clusteredAcceptedSize = stepClusteredSize;
        }
    };

    // the windows in key order for the counters
    std::vector<size_t> windowFirstKeys(1, 0);
    for (size_t i = 0; i < partitions.size(); i++) {
        windowFirstKeys.push_back(partitions[i].firstKey);
    }
    for (size_t i = 0; i < windows.size(); i++) {
        windowFirstKeys.push_back(windows[i].firstKey);
    }
    std::sort(windowFirstKeys.begin(), windowFirstKeys.end());
    windowFirstKeys.erase(std::unique(windowFirstKeys.begin(), windowFirstKeys.end()), windowFirstKeys.end());
    std::vector<WindowCounters> windowCounters(windowFirstKeys.size());
    FetchSizeCounters fetchSizeCounters[64];
    // MMSEQS_ALIGNED_PAIR_LOG=<dir>: the (rep key, target key) pairs the node aligns, for offline counts
    const char *alignedPairDir = getenv("MMSEQS_ALIGNED_PAIR_LOG");
    const std::string alignedPairFile = (alignedPairDir != NULL) ? std::string(alignedPairDir) + "/" + FileUtil::baseName(par.db3) + "." + SSTR(rank) : "";
    const bool isLoggingAlignedPairs = alignedPairDir != NULL && isAligningNode;
    FILE *alignedPairLog = isLoggingAlignedPairs ? FileUtil::openFileOrDie(alignedPairFile.c_str(), "w", false) : NULL;
    std::mutex alignedPairLogMutex;

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
    size_t nodeBlockCount = 0;
    size_t localBlockSize = 0;
    size_t localBlockCount = 0;
    for (size_t range = 0, block = 0; range < rangeCount; range++) {
        const size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
        rangeByteOffsets[range + 1] = rangeByteOffsets[range] + (keyEnd - index[range].id) * index[range].length;
        nodeRangeByteOffsets[range + 1] = nodeRangeByteOffsets[range];
        for (size_t key = index[range].id; key < keyEnd; key = std::min(keyEnd, blockKeyOffsets[block + 1])) {
            for (; blockKeyOffsets[block + 1] <= key; block++) {}
            const size_t blockBytes = (getBlockAlignNode(block) == rank) ? (std::min(keyEnd, blockKeyOffsets[block + 1]) - key) * index[range].length : 0;
            nodeRangeByteOffsets[range + 1] += blockBytes;
            nodeBlockCount += getBlockAlignNode(block) == rank;
            localBlockSize += seqDbr.isLocalBlock(block) ? blockBytes : 0;
            localBlockCount += seqDbr.isLocalBlock(block);
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
    size_t maxFetchSize = std::max(memoryLimit / 8, minFetchSize);
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
    // the blocks in --local-tmp count like the block cache
    const bool isReadingDirectly = isCacheTooSmall(blockCacheSize + localBlockSize);
    // the blocks of other align nodes are read from the shared database, in a block cache of their own
    DBReader<DBKeyType> *repDbr = NULL;
    size_t repBlockCacheSize = 0;
    if (isAligningNode && alignGroupNodeEnd - alignGroupNodeBegin > 1 && sizeClassCount == 1 && isReadingDirectly == false) {
        repDbr = new DBReader<DBKeyType>(par.db1.c_str(), par.db1Index.c_str(), std::max(1, par.threads / 4),
                                         DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        repDbr->open(DBReader<DBKeyType>::NOSORT);
        // its cache is what the blocks of this node leave free of the block cache, a block read is rounded to pages at both ends
        const size_t nodeCacheSize = servedSequenceSize + 2 * pageSize * nodeBlockCount;
        repBlockCacheSize = std::min(maxFetchSize, blockCacheSize - std::min(blockCacheSize, nodeCacheSize));
        repDbr->setBlockCacheSize(repBlockCacheSize);
        seqDbr.setBlockCacheSize(blockCacheSize - repBlockCacheSize);
    }
    // local blocks leave the block cache room for the other blocks of this node, a block read is rounded to pages at both ends
    const size_t seqBlockCacheSize = blockCacheSize - repBlockCacheSize;
    const size_t sharedBlockSize = servedSequenceSize - localBlockSize + 2 * pageSize * (nodeBlockCount - localBlockCount);
    seqDbr.setLocalBlockCacheSize(seqBlockCacheSize - std::min(seqBlockCacheSize, sharedBlockSize));
    if (isAligningNode) {
        MMseqsNode::addPhaseValue(phaseValues, "Block cache", SSTR(blockCacheSize) + " bytes for " + SSTR(static_cast<size_t>(alignNodeCoverableSize)) + " coverable bytes and " + SSTR(servedSequenceSize) + " sequence bytes of the align node"
                                 + ", " + SSTR(localBlockSize) + " of them in --local-tmp" + (isReadingDirectly ? ", the sequences are read directly" : ""));
    }
    if (isAligningNode && blockCacheSize + localBlockSize < servedSequenceSize) {
        Debug(Debug::WARNING) << "Node " << rank << " holds " << servedSequenceSize << " sequence bytes, --local-tmp holds " << localBlockSize << " of them and its block cache has " << blockCacheSize
                              << " bytes, the other sequences are read one by one. Use more nodes or increase --split-memory-limit\n";
    }
    // the first fetch is small, then they double up to half of READ_SECONDS of aligning at the rate of the last fetch; MMSEQS_ALIGN_FETCH_BYTES=<n> fixes the limit for measurements
    size_t fetchBytesLimit = std::min<size_t>(maxFetchSize / 64, 64 * 1024 * 1024);
    const char *fixedFetchBytes = getenv("MMSEQS_ALIGN_FETCH_BYTES");
    char *fetchBytesEnd = NULL;
    const size_t fetchBytes = (fixedFetchBytes != NULL) ? strtoull(fixedFetchBytes, &fetchBytesEnd, 10) : 0;
    const bool isFetchBytesFixed = fixedFetchBytes != NULL && fetchBytesEnd != fixedFetchBytes && *fetchBytesEnd == '\0';
    if (fixedFetchBytes != NULL && isFetchBytesFixed == false) {
        Debug(Debug::WARNING) << "Ignoring MMSEQS_ALIGN_FETCH_BYTES=" << fixedFetchBytes << ", it must be a number of bytes\n";
    }
    if (isFetchBytesFixed) {
        fetchBytesLimit = std::min(fetchBytes, maxFetchSize);
        if (isAligningNode) {
            MMseqsNode::addPhaseValue(phaseValues, "Fetch bytes limit", SSTR(fetchBytesLimit) + " fixed by MMSEQS_ALIGN_FETCH_BYTES, the largest fetch is " + SSTR(maxFetchSize) + " bytes");
        }
    }
    size_t maxFetchBytesLimit = isFetchBytesFixed ? fetchBytesLimit : maxFetchSize;
    // the fetches in flight share the two fetches of getReservedMemory, the slots hold as many as fit at the first limit
    const size_t fetchBudget = 2 * std::max(maxFetchSize, repTargetsReader.maxRecordSize);
    std::vector<FetchedSequences> fetchSlots(1 + fetchBudget / std::max(fetchBytesLimit, minFetchSize));
    // the slot released last is taken first, so few slots get buffers
    std::vector<FetchedSequences *> idleFetches;
    for (size_t i = fetchSlots.size(); i > 0; i--) {
        fetchSlots[i - 1].fetchSourceKeyOffsets.assign(nodes + 2, 0);
        idleFetches.push_back(&fetchSlots[i - 1]);
    }
    std::deque<FetchedSequences *> fetchesInFlight;
    const auto getFetchSource = [&](const FetchedSequences &sequences, size_t key, size_t &block) -> size_t {
        if (isReadingDirectly) {
            return rank;
        }
        size_t alignNode = getKeyAlignNode(seqDbr, key, &block);
        if (alignNode == rank || key < sequences.repKeyBegin || key >= sequences.repKeyEnd) {
            return alignNode;
        }
        return getBlockRepServingNode(block);
    };
    const auto getFetchedEntry = [&](const FetchedSequences &sequences, size_t key, size_t *seqLen = NULL, FetchedKeyCursor *cursor = NULL) -> const char * {
        const char *data = findFetchedKey(sequences, nodes, key, seqLen, (cursor != NULL) ? &cursor->positions[nodes] : NULL);
        if (data != NULL) {
            return data;
        }
        size_t block = SIZE_MAX;
        size_t fetchSource = getFetchSource(sequences, key, (cursor != NULL) ? cursor->block : block);
        return findFetchedKey(sequences, fetchSource, key, seqLen, (cursor != NULL) ? &cursor->positions[fetchSource] : NULL);
    };
    const auto getFetchedSequence = [&](const FetchedSequences &sequences, size_t key, size_t *seqLen = NULL, FetchedKeyCursor *cursor = NULL) -> const char * {
        const char *data = getFetchedEntry(sequences, key, seqLen, cursor);
        // assigned keys are served as zeros
        return (data != NULL && data[0] != '\0') ? data : NULL;
    };
    // the entry bytes requested from other nodes by their use
    const bool isCountingRequests = isAligningNode && alignNodeCount > 1 && isReadingDirectly == false;
    std::vector<std::vector<DBKeyType> > threadRepKeys(par.threads);
    std::vector<unsigned char> keyUses;
    size_t requestedRepSize = 0;
    size_t requestedPendingSize = 0;
    std::mutex serviceMutex;
    std::condition_variable serviceCondition;
    // the I/O threads sleep until it changes: a hand-off, a used prefilter buffer, the end of the serving or of the aligning
    uint64_t ioWork = 0;
    std::atomic<double> alignedCellsPerSecond(0);
    // a step of the fill of a buffer, or all of its steps; true once the buffer is complete
    const auto fillPrefilterBuffer = [&](size_t buffer, size_t maxBufferAlignmentCells, bool isOneStep) {
        if (targetSplitReader != NULL) {
            return targetSplitReader->fillPrefilterBuffer(prefilterRecords[buffer], prefilterTargets[buffer], maxFetchSize, maxBufferAlignmentCells, isOneStep);
        }
        return repTargetsReader.fillPrefilterBuffer(prefilterRecords[buffer], prefilterTargets[buffer], maxFetchSize, maxBufferAlignmentCells, isOneStep);
    };
    PrefilterFillCounters &fillCounters = (targetSplitReader != NULL) ? targetSplitReader->fillCounters : repTargetsReader.fillCounters;
    std::mutex prefilterFillMutex;
    std::atomic<bool> isPrefilterDone(false);
    // the main thread waits for the fill, the I/O threads leave it the fill mutex
    std::atomic<bool> isPrefilterWaited(false);
    double prefilterFillSeconds = 0;
    // the time of the last buffer switch, under prefilterBufferMutex, and the buffers whose fill started
    double prefilterSwitchTime = 0;
    size_t startedPrefilterBufferCount = 1;
    // a step of the fill of buffer n once the fetches use buffer n - 1, until the first empty buffer; the waiting main thread finishes the buffer
    const auto fillNextPrefilterBuffer = [&](bool isWaiting) {
        std::unique_lock<std::mutex> taskLock(prefilterFillMutex, std::defer_lock);
        if (isWaiting) {
            isPrefilterWaited = true;
            Timer blockedTimer;
            taskLock.lock();
            isPrefilterWaited = false;
            fillCounters.blockedSeconds[fillCounters.lastStepKind] += blockedTimer.getTimediff();
        } else if (isPrefilterWaited || taskLock.try_lock() == false || isPrefilterWaited) {
            return false;
        }
        if (isPrefilterDone) {
            return false;
        }
        size_t next;
        double switchTime;
        {
            std::lock_guard<std::mutex> lock(prefilterBufferMutex);
            if (usedPrefilterBufferCount < filledPrefilterBufferCount) {
                return false;
            }
            next = filledPrefilterBufferCount;
            switchTime = prefilterSwitchTime;
        }
        if (next == startedPrefilterBufferCount) {
            fillCounters.startDelaySeconds[isWaiting] += startTimer.getTimediff() - switchTime;
            fillCounters.startedBufferCount[isWaiting]++;
            startedPrefilterBufferCount++;
        }
        Timer fillTimer;
        const bool isFilled = fillPrefilterBuffer(next % 3, static_cast<size_t>(2.0 * PUBLISH_SECONDS * alignedCellsPerSecond.load()), isWaiting == false);
        prefilterFillSeconds += fillTimer.getTimediff();
        if (isFilled == false) {
            return true;
        }
        fillCounters.mainBufferCount += isWaiting;
        const bool isLast = prefilterRecords[next % 3].empty();
        {
            std::lock_guard<std::mutex> lock(prefilterBufferMutex);
            filledPrefilterBufferCount = next + 1;
        }
        isPrefilterDone = isLast;
        prefilterBufferCondition.notify_all();
        return true;
    };
    // take records while they fit, at least one that is fetched; records whose rep is assigned take no bytes
    const auto selectRecordKeys = [&](FetchedSequences &sequences, const FetchedSequences &previous) {
        // an empty prefilter buffer is the last one
        if (prefilterRecordPos[prefilterBuffer] == prefilterRecords[prefilterBuffer].size() && prefilterRecords[prefilterBuffer].empty() == false) {
            Timer waitTimer;
            // the waiting thread finishes the next buffer with its idle team
            fillNextPrefilterBuffer(true);
            std::unique_lock<std::mutex> lock(prefilterBufferMutex);
            prefilterBufferCondition.wait(lock, [&]() { return filledPrefilterBufferCount > usedPrefilterBufferCount; });
            prefilterBuffer = usedPrefilterBufferCount++ % 3;
            prefilterSwitchTime = startTimer.getTimediff();
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
        size_t aheadPendingPairCount = previous.pendingPairCount;
        size_t aheadTargetCount = previous.targetCount;
        bool hasRecordsAhead = previous.end > previous.begin;
        for (size_t i = 0; i < fetchesInFlight.size(); i++) {
            aheadPendingPairCount += fetchesInFlight[i]->pendingPairCount;
            aheadTargetCount += fetchesInFlight[i]->targetCount;
            hasRecordsAhead |= fetchesInFlight[i]->end > fetchesInFlight[i]->begin;
        }
        // pending pairs first, at least one
        size_t fetchSize = 0;
        sequences.pendingPairCount = 0;
        for (size_t i = aheadPendingPairCount; i < pendingPairs.size() && (sequences.pendingPairCount == 0 || fetchSize + pendingPairSize[i] <= fetchBytesLimit); i++) {
            fetchSize += pendingPairSize[i];
            sequences.pendingPairCount++;
        }
        // the accepted pairs follow the pending pairs in log order
        sequences.firstPendingRepKey = (sequences.pendingPairCount > 0) ? static_cast<size_t>(pendingPairs[aheadPendingPairCount].repKey)
                                       : (acceptedPairs.empty() ? orderKeyEnd : static_cast<size_t>(acceptedPairs[0].repKey));
        // the accepted pairs and the targets of the fetches ahead and this fetch fit memberKeyBudget, a record goes alone once no pair waits
        size_t acceptablePairCount = acceptedPairs.size() + aheadTargetCount;
        sequences.targetCount = 0;
        const size_t pendingFetchSize = fetchSize;
        for (; sequences.end < prefilterRecords[prefilterBuffer].size(); sequences.end++) {
            const RepTargetsRecord &record = prefilterRecords[prefilterBuffer][sequences.end];
            const size_t recordSize = isAssigned(record.repOrderKeyFirstTarget.repKey) ? 0 : record.readSize;
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
        // the records of a fetch ascend by rep
        sequences.repKeyBegin = 0;
        sequences.repKeyEnd = 0;
        if (repDbr != NULL && sequences.end > sequences.begin) {
            sequences.repKeyBegin = prefilterRecords[prefilterBuffer][sequences.begin].repOrderKeyFirstTarget.repKey % paddedDbSize;
            sequences.repKeyEnd = prefilterRecords[prefilterBuffer][sequences.end - 1].repOrderKeyFirstTarget.repKey % paddedDbSize + 1;
        }
        // the fetches ahead may still add pending pairs
        sequences.isLast = sequences.end == sequences.begin && sequences.pendingPairCount == 0 && (memberDbr == NULL || hasRecordsAhead == false) && acceptedPairs.empty();
        fetchBytesLimit = std::min(2 * fetchBytesLimit, maxFetchBytesLimit);
        size_t unassignedRecordSize = 0;
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            threadKeys[thread_idx].clear();
            threadRepKeys[thread_idx].clear();
#pragma omp for schedule(dynamic, 100) reduction(+:unassignedRecordSize)
            for (size_t i = sequences.begin; i < sequences.end; i++) {
                const RepTargetsRecord &record = prefilterRecords[sequences.prefilterBuffer][i];
                const size_t repKey = record.repOrderKeyFirstTarget.repKey % paddedDbSize;
                const char *previousRep = getFetchedEntry(previous, repKey);
                if (isAssigned(record.repOrderKeyFirstTarget.repKey) || (previousRep != NULL && previousRep[0] == '\0')) {
                    continue;
                }
                unassignedRecordSize += record.readSize;
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
                    if (isCountingRequests) {
                        threadRepKeys[thread_idx].push_back(repKey);
                    }
                }
            }
            // members share the keys and data files of the sequence database
#pragma omp for schedule(dynamic, 100)
            for (size_t i = aheadPendingPairCount; i < aheadPendingPairCount + sequences.pendingPairCount; i++) {
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
        sequences.unassignedRecordSize = unassignedRecordSize;
        // a requested key counts for a rep first, then for a member or pending pair
        if (isCountingRequests) {
            keyUses.assign(uniqueKeys.size(), 0);
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                for (size_t i = 0; i < threadRepKeys[thread_idx].size(); i++) {
                    __atomic_fetch_or(&keyUses[std::lower_bound(uniqueKeys.begin(), uniqueKeys.end(), threadRepKeys[thread_idx][i]) - uniqueKeys.begin()], 1, __ATOMIC_RELAXED);
                }
#pragma omp for schedule(dynamic, 100)
                for (size_t i = aheadPendingPairCount; i < aheadPendingPairCount + sequences.pendingPairCount; i++) {
                    if (isAssigned(pendingPairs[i].repKey) == false && isAssigned(pendingPairs[i].memberKey) == false) {
                        __atomic_fetch_or(&keyUses[std::lower_bound(uniqueKeys.begin(), uniqueKeys.end(), pendingPairs[i].repKey % paddedDbSize) - uniqueKeys.begin()], 2, __ATOMIC_RELAXED);
                        for (size_t member = pendingMemberOffsets[i]; member < pendingMemberOffsets[i + 1]; member++) {
                            __atomic_fetch_or(&keyUses[std::lower_bound(uniqueKeys.begin(), uniqueKeys.end(), pendingMembers[member]) - uniqueKeys.begin()], 2, __ATOMIC_RELAXED);
                        }
                    }
                }
            }
        }
        keyFetchSources.resize(uniqueKeys.size());
        sequences.keys.resize(uniqueKeys.size());
        // sort the keys by fetch source; assigned keys go last and are dropped
        threadFetchSourceKeyOffsets.assign((nodes + 2) * par.threads + 1, 0);
        size_t repRequestSize = 0;
        size_t pendingRequestSize = 0;
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<size_t> fetchSourceOffsets(nodes + 2, 0);
            FetchedKeyCursor keyCursor(nodes + 1);
            size_t requestRange = SIZE_MAX;
#pragma omp for schedule(static) reduction(+:repRequestSize, pendingRequestSize)
            for (size_t i = 0; i < uniqueKeys.size(); i++) {
                keyFetchSources[i] = isAssigned(uniqueKeys[i]) ? nodes + 1 : ((getFetchedEntry(previous, uniqueKeys[i], NULL, &keyCursor) != NULL) ? nodes : getFetchSource(sequences, uniqueKeys[i], keyCursor.block));
                fetchSourceOffsets[keyFetchSources[i]]++;
                if (isCountingRequests && keyFetchSources[i] < nodes && keyFetchSources[i] != rank) {
                    const size_t entrySize = getSeqLenFromCursor(seqDbr, requestRange, uniqueKeys[i]) + 2;
                    repRequestSize += (keyUses[i] & 1) ? entrySize : 0;
                    pendingRequestSize += (keyUses[i] == 2) ? entrySize : 0;
                }
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
        requestedRepSize += repRequestSize;
        requestedPendingSize += pendingRequestSize;
        sequences.fetchSourceKeyOffsets.resize(nodes + 2);
        for (unsigned int node = 0; node <= nodes + 1; node++) {
            sequences.fetchSourceKeyOffsets[node] = threadFetchSourceKeyOffsets[node * par.threads];
        }
        sequences.ownKeyCopyEnd = sequences.fetchSourceKeyOffsets[rank];
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
    fillPrefilterBuffer(0, 0, false);
    filledPrefilterBufferCount = 1;
    usedPrefilterBufferCount = 1;
    prefilterSwitchTime = startTimer.getTimediff();
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

    // the fetches handed to the I/O threads and not requested yet, in selection order
    std::deque<FetchedSequences *> handedOffFetches;
    // the handed-off fetches the main thread has not taken yet, the I/O threads copy their own keys ahead in this order
    std::deque<FetchedSequences *> ownKeyFetches;
    // a node aligns ahead of the master by at most its share of a master step
    const size_t maxUnclusteredAcceptedSize = std::max<size_t>(1, getStepWords(masterMemoryLimit) / nodes) * sizeof(uint64_t);
    bool isWaitingForMaster = false;
    // one load of seqDbr and repDbr at a time
    std::mutex sourceMutex;
    std::vector<DBKeyType> loadKeys;
    std::vector<DBKeyType> repLoadKeys;
    std::vector<std::vector<size_t> > threadRawEntries(par.threads);
    size_t servedFileCount = 0;
    size_t servedPieceCount = 0;
    size_t maxPassPieceCount = 0;
    size_t receivedPieceCount = 0;
    size_t servedSize = 0;
    size_t servedEntrySize = 0;
    size_t receivedServedEntriesSize = 0;
    size_t receivedEntrySize = 0;
    Timer serviceTimer;
    size_t requestCount = 0;
    size_t requestedEntrySize = 0;
    std::atomic<size_t> progressReadCount(0);
    size_t requestProgressReadCount = 0;
    size_t requestProgressChangeCount = 0;
    size_t requestListingCount = 0;
    size_t servedListingCount = 0;
    size_t servedListingReadCount = 0;
    std::atomic<size_t> objectReadCount(0);
    std::atomic<size_t> objectWriteCount(0);
    std::atomic<size_t> deleteCount(0);
    size_t liveServedSize = 0;
    size_t maxLiveServedSize = 0;
    double servingSeconds = 0;
    double readingSeconds = 0;
    double importPollingSeconds = 0;
    double exportPollingSeconds = 0;
    double sourceLoadSeconds = 0;
    double packSeconds = 0;
    double servedFileWriteSeconds = 0;
    double unpackSeconds = 0;
    std::vector<size_t> requesterServedSizes(nodes, 0);
    // the own front rep key ahead of the clustered order key, over the seconds of the service passes
    double gateSlackSeconds = 0;
    double gateSlackKeySeconds = 0;
    double gateSlackWindowSeconds = 0;
    size_t maxGateSlackKeys = 0;
    size_t maxGateSlackWindows = 0;
    size_t threadSampleCount = 0;
    size_t maxProcessThreads = 0;
    size_t maxThreadsBesidesRingWorkers = 0;
    // copy, or pack with rawKeys, the entries of keys; assigned keys get zeros
    const auto copyEntries = [&](const DBKeyType *keys, size_t keyCount, const size_t *offsets, char *data, std::vector<DBKeyType> *rawKeys, size_t maxLoadSize) {
        size_t loadRange = SIZE_MAX;
        for (size_t first = 0; first < keyCount;) {
            const size_t loadFirst = first;
            // with the block cache a load of at most a block reads one block
            const size_t loadKeyEnd = (blockCacheSize > 0 && maxLoadSize <= BatchReader::MAX_READ_REQUEST_SIZE) ? blockKeyOffsets[seqDbr.getKeyBlock(keys[first]) + 1] : SIZE_MAX;
            // a load fits maxLoadSize, the buffers of both readers fit maxFetchSize
            for (size_t loadSize = 0; first < keyCount; first++) {
                const size_t entrySize = seqDbr.getSeqLenFromRange(loadRange, keys[first]) + 2 + 2 * pageSize;
                if (first > loadFirst && (loadSize + entrySize > maxLoadSize || keys[first] >= loadKeyEnd)) {
                    break;
                }
                loadSize += entrySize;
            }
            std::lock_guard<std::mutex> sourceLock(sourceMutex);
            loadKeys.clear();
            repLoadKeys.clear();
            size_t loadBlock = SIZE_MAX;
            for (size_t i = loadFirst; i < first; i++) {
                if (isAssigned(keys[i])) {
                    continue;
                }
                if (repDbr == NULL || getKeyAlignNode(seqDbr, keys[i], &loadBlock) == rank) {
                    loadKeys.push_back(keys[i]);
                } else {
                    repLoadKeys.push_back(keys[i]);
                }
            }
            Timer copyTimer;
            seqDbr.loadData(loadKeys);
            if (repDbr != NULL) {
                repDbr->loadData(repLoadKeys);
            }
            sourceLoadSeconds += copyTimer.getTimediff();
            copyTimer.reset();
            for (int thread = 0; thread < par.threads; thread++) {
                threadRawEntries[thread].clear();
            }
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                // the static chunks give a thread ascending keys
                size_t packRange = SIZE_MAX;
                DBReader<DBKeyType>::LoadedBufferCursor cursor;
                DBReader<DBKeyType>::LoadedBufferCursor repCursor;
                size_t block = SIZE_MAX;
#pragma omp for schedule(static)
                for (size_t i = loadFirst; i < first; i++) {
                    // bits are never cleared
                    if (isAssigned(keys[i])) {
                        memset(data + offsets[i], 0, offsets[i + 1] - offsets[i]);
                        continue;
                    }
                    const char *entry = (repDbr == NULL || getKeyAlignNode(seqDbr, keys[i], &block) == rank) ? seqDbr.getDataFromLoadedBuffer(keys[i], cursor)
                                                                                                             : repDbr->getDataFromLoadedBuffer(keys[i], repCursor);
                    if (rawKeys == NULL) {
                        memcpy(data + offsets[i], entry, offsets[i + 1] - offsets[i]);
                    } else if (packEntry(entry, seqDbr.getSeqLenFromRange(packRange, keys[i]) + 2, data + offsets[i]) == false) {
                        data[offsets[i]] = PACKED_RAW_CODE;
                        threadRawEntries[thread_idx].push_back(i);
                    }
                }
            }
            // the static parts of the threads follow in key order
            for (int thread = 0; rawKeys != NULL && thread < par.threads; thread++) {
                for (size_t j = 0; j < threadRawEntries[thread].size(); j++) {
                    rawKeys->push_back(keys[threadRawEntries[thread][j]]);
                }
            }
            packSeconds += copyTimer.getTimediff();
        }
    };
    // own request: code offsets per node, pieces and keys received per node
    struct SequenceRequest {
        FetchedSequences *sequences;
        size_t number;
        size_t size;
        std::vector<size_t> codeOffsets;
        std::vector<size_t> receivedPieces;
        std::vector<size_t> receivedKeys;
        bool isComplete;
    };
    std::deque<SequenceRequest> ownRequests;
    size_t receivedRequestCount = 0;
    bool isLastRequested = false;
    // the requested fetches become ready in selection order
    std::deque<FetchedSequences *> requestedFetches;
    std::vector<char> requestCodes;
    std::vector<char> requesterCodes;
    // piece of a served file: request, piece of the request, served file and its size, offset, bytes, keys
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
    // seconds to the next read of a node that owes a request, doubled up to progressSeconds while it has not served
    std::vector<double> servedProgressSeconds(nodes, BATCH_READ_SECONDS);
    // the nodes that owe are read at half of PROGRESS_READS_PER_SECOND until fastReadEnd
    double fastReadSeconds = BATCH_READ_SECONDS;
    double fastReadEnd = 0;
    double nextRequestListing = 0;
    double nextServedListing = 0;
    // the master progress while the gate is closed, about as often as before the faster polls
    double nextGateRead = 0;
    std::vector<std::string> requestNames;
    std::vector<std::string> servedNames;
    // one past the newest served file of a node in the last listing and in its progress
    std::vector<size_t> listedServedFileCounts(nodes, 0);
    std::vector<size_t> progressServedFileCounts(nodes, 0);
    std::vector<std::vector<ServedPiece> > servedPieces(nodes);
    // requesting node: progress, code range of this node per open request, codes served so far, pieces not yet deleted
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
    size_t requesterCount = 0;
    for (unsigned int node = 0; node < nodes; node++) {
        requesters[node].isDone = node == rank || node < alignGroupNodeBegin || node >= alignGroupNodeEnd;
        requesterCount += requesters[node].isDone == false;
    }
    // seconds between two reads of the progress file of a node
    const double progressSeconds = std::max(BATCH_READ_SECONDS, 2.0 * requesterCount / PROGRESS_READS_PER_SECOND);
    std::vector<const char *> requestColumns(nodes + 3);
    bool isServedProgressChanged = false;
    std::vector<char> receivedEntries;
    std::vector<DBKeyType> servedKeys;
    std::vector<DBKeyType> rawKeys;
    std::vector<size_t> requestOffsets;
    // the pieces of a pass in one served file
    std::vector<char> servedFileData;
    std::vector<std::pair<unsigned int, ServedPiece> > passPieces;
    std::vector<size_t> filePieceCounts;
    // the next pass starts with the first requester left out of a full served file
    unsigned int firstRequester = rank + 1;
    // the served files of an align node and the requests of a requester fail on their own
    std::vector<time_t> missingServedSince(nodes, 0);
    std::vector<unsigned int> missingServedAttempts(nodes, 0);
    std::vector<time_t> missingRequestSince(nodes, 0);
    std::vector<unsigned int> missingRequestAttempts(nodes, 0);
    std::string requestProgress = requests + "." + SSTR(rank) + ".progress";
    std::string servedProgress = served + "." + SSTR(rank) + ".progress";
    const auto getServedFile = [&](unsigned int alignNode, size_t file) {
        return served + "/" + SSTR(alignNode) + "." + SSTR(file);
    };
    MMseqsNode::WaitWarning waitWarning;
    // the serving passes that worked, a wait warning resets after them as after the receiving passes
    std::atomic<size_t> exportPassCount(0);
    size_t seenExportPassCount = 0;
    time_t assignedKeysReadTime = 0;
    // the sleeps by their one reason: an owed request, the master gate, no fetch of the aligning threads, else; set by the receiving pass
    const char *idleReasons[] = { "owed", "gate", "no request", "other" };
    double idleSeconds[4] = { 0, 0, 0, 0 };
    size_t idleCounts[4] = { 0, 0, 0, 0 };
    size_t importIdleReason = 2;
    bool isImportUrgent = false;
    // under serviceMutex
    bool isReceiveDone = false;
    bool isServeDone = false;
    bool isAligningDone = false;
    size_t completedFetchCount = 0;
    double statusPackSeconds = 0;
    // the timers of the serving pass at its end, for the status of the receiving pass
    double statusServingSeconds = 0;
    double statusServedFileWriteSeconds = 0;
    double statusExportPollingSeconds = 0;
    double nextStatusTime = STATUS_SECONDS;
    // the gate key of the newest fetch
    size_t frontRepKey = orderKeyEnd;
    size_t gateSlackKeys = 0;
    size_t gateSlackWindows = 0;
    double gateSlackTime = 0;
    const auto printStatus = [&]() {
        const double now = serviceTimer.getTimediff();
        std::string value = "done rep key " + SSTR(publishedDoneRepKey.load()) + ", " + SSTR(completedFetchCount) + " fetches, " + SSTR(requestCount) + " requests";
        double idleSum = 0;
        std::string idle;
        double servingTime;
        double servedFileWriteTime;
        double exportPollingTime;
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            value += "; waiting for the sequences " + formatSeconds(sequenceWaitSeconds) + ", the master " + formatSeconds(waitSeconds);
            for (size_t i = 0; i < 4; i++) {
                idleSum += idleSeconds[i];
                idle += std::string((i > 0) ? ", " : "") + idleReasons[i] + " " + formatSeconds(idleSeconds[i]) + " in " + SSTR(idleCounts[i]) + " sleeps";
            }
            servingTime = statusServingSeconds;
            servedFileWriteTime = statusServedFileWriteSeconds;
            exportPollingTime = statusExportPollingSeconds;
        }
        value += "; service idle " + formatSeconds(idleSum) + " of " + formatSeconds(now) + ": " + idle;
        // the main thread copies the own keys too, a status does not wait for it
        {
            std::unique_lock<std::mutex> sourceLock(sourceMutex, std::try_to_lock);
            statusPackSeconds = sourceLock.owns_lock() ? packSeconds : statusPackSeconds;
        }
        value += "; service: serving " + formatSeconds(servingTime) + ", packing " + formatSeconds(statusPackSeconds) + ", writing served files " + formatSeconds(servedFileWriteTime)
                 + ", reading served entries " + formatSeconds(readingSeconds) + ", polling " + formatSeconds(importPollingSeconds + exportPollingTime);
        value += "; gate slack " + SSTR(gateSlackKeys) + " keys and " + SSTR(gateSlackWindows) + " windows, at most " + SSTR(maxGateSlackKeys) + " keys and " + SSTR(maxGateSlackWindows) + " windows";
        size_t processThreads;
        size_t ringWorkers;
        countProcessThreads(processThreads, ringWorkers);
        threadSampleCount++;
        maxProcessThreads = std::max(maxProcessThreads, processThreads);
        maxThreadsBesidesRingWorkers = std::max(maxThreadsBesidesRingWorkers, processThreads - ringWorkers);
        value += "; process threads (incl. io_uring workers) " + SSTR(processThreads) + ", " + SSTR(ringWorkers) + " io_uring workers, --threads " + SSTR(par.threads);
        std::string status;
        MMseqsNode::addPhaseValue(status, "Align node status after " + SSTR(static_cast<size_t>(now)) + " s", value);
        MMseqsNode::printPhaseValues(status);
    };
    std::mutex receiveMutex;
    std::mutex serveMutex;
    // a pass over the requests of this node and the pieces served for them
    const auto receiveSequences = [&]() {
        std::unique_lock<std::mutex> taskLock(receiveMutex, std::try_to_lock);
        if (taskLock.owns_lock() == false) {
            return false;
        }
        bool hasWorked = false;
        // the slack since the last pass
        const double slackTime = serviceTimer.getTimediff();
        if (frontRepKey < orderKeyEnd) {
            const size_t clusteredKey = std::min<size_t>(frontRepKey, __atomic_load_n(&clusteredOrderKey, __ATOMIC_RELAXED));
            gateSlackKeys = frontRepKey - clusteredKey;
            gateSlackWindows = getWindowPosition(windowFirstKeys, paddedDbSize, frontRepKey) - getWindowPosition(windowFirstKeys, paddedDbSize, clusteredKey);
            maxGateSlackKeys = std::max(maxGateSlackKeys, gateSlackKeys);
            maxGateSlackWindows = std::max(maxGateSlackWindows, gateSlackWindows);
            gateSlackSeconds += slackTime - gateSlackTime;
            gateSlackKeySeconds += (slackTime - gateSlackTime) * gateSlackKeys;
            gateSlackWindowSeconds += (slackTime - gateSlackTime) * gateSlackWindows;
        }
        gateSlackTime = slackTime;
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
        // request the handed-off fetches in order while the master is close enough behind
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
            // a fetch without records and pending pairs logs nothing, so it never waits; the pairs the aligning fetch adds come from it
            bool canWait = isLast == false && (nextRequest->end > nextRequest->begin || nextRequest->pendingPairCount > 0);
            bool isAhead;
            {
                std::lock_guard<std::mutex> assignedKeysLock(assignedKeysMutex);
                size_t repKey = std::min(nextRequest->firstPendingRepKey, getFirstRepKey(*nextRequest));
                frontRepKey = repKey;
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
                    handedOffFetches.pop_front();
                }
            }
            isGated = isAhead;
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
                    request.receivedPieces.assign(nodes, 0);
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
                requestedFetches.push_back(nextRequest);
                hasWorked = true;
            }
        }
        // the progress of an owed node is read at once when the names hold a served file it did not show yet
        bool isServedListed = ownRequests.empty() == false && serviceTimer.getTimediff() >= nextServedListing;
        if (isServedListed) {
            nextServedListing = serviceTimer.getTimediff() + std::max(BATCH_READ_SECONDS, (nodes - 1.0) / PROGRESS_READS_PER_SECOND) / 4.0;
            MMseqsNode::listDirectory(served, servedNames);
            servedListingCount++;
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
                servedListingReadCount++;
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
            importPollingSeconds += pollTimer.getTimediff();
            if (hasProgress) {
                servedPieces[node].clear();
                for (char *line = &content[0]; *line != '\0'; line = Util::skipLine(line)) {
                    const char *columns[8];
                    if (Util::getWordsOfLine(line, columns, 8) != 8) {
                        continue;
                    }
                    // the files of all requesters, a listed file without a piece of this node moves up one read only
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
                    // the next piece from its served file, a failed read is tried again after the next progress read
                    std::string fileName = getServedFile(node, piece.file);
                    size_t observedSize;
                    uint64_t observedNumber;
                    int lastError;
                    Timer readTimer;
                    objectReadCount++;
                    int fd = MMseqsNode::tryOpenExchangeFile(fileName, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + piece.fileSize, piece.file, observedSize, observedNumber, lastError);
                    if (fd < 0) {
                        readingSeconds += readTimer.getTimediff();
                        if (lastError != 0 && lastError != ENOENT && lastError != ESTALE && lastError != EIO) {
                            Debug(Debug::ERROR) << "Cannot open served sequence file " << fileName << ": " << strerror(lastError) << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        failIfMissingTooLong(missingServedSince[node], missingServedAttempts[node], "Cannot open complete served sequence file " + fileName
                                             + describeServedFile(MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + piece.fileSize, observedSize, piece.file, observedNumber, lastError));
                        break;
                    }
                    if (receivedEntries.size() < piece.size) {
                        receivedEntries.resize(piece.size);
                    }
                    // the parts keep their errors
                    size_t partCount = (piece.size + BatchReader::MAX_READ_REQUEST_SIZE - 1) / BatchReader::MAX_READ_REQUEST_SIZE;
                    std::vector<int> readErrors(partCount, 0);
#pragma omp parallel for schedule(dynamic, 1)
                    for (size_t part = 0; part < partCount; part++) {
                        size_t offset = part * BatchReader::MAX_READ_REQUEST_SIZE;
                        int partError;
                        if (MMseqsNode::tryPreadAll(fd, receivedEntries.data() + offset, std::min(BatchReader::MAX_READ_REQUEST_SIZE, piece.size - offset), MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + piece.offset + offset, partError) == false) {
                            readErrors[part] = (partError != 0) ? partError : -1;
                        }
                    }
                    int readError = 0;
                    for (size_t part = 0; part < partCount; part++) {
                        readError = (readErrors[part] != 0) ? readErrors[part] : readError;
                    }
                    MMseqsNode::releaseExchangeCopy(fd);
                    int closeError = (close(fd) != 0) ? errno : 0;
                    readingSeconds += readTimer.getTimediff();
                    int error = (readError > 0) ? readError : closeError;
                    if (error != 0 && error != ESTALE && error != EINTR && error != EIO) {
                        Debug(Debug::ERROR) << "Cannot read served sequence file " << fileName << ": " << strerror(error) << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    if (readError != 0 || closeError != 0) {
                        failIfMissingTooLong(missingServedSince[node], missingServedAttempts[node], "Cannot read complete served sequence file " + fileName + " (error " + SSTR(error) + ")\n");
                        break;
                    }
                    missingServedAttempts[node] = 0;
                    FetchedSequences *sequences = request.sequences;
                    size_t firstKey = sequences->fetchSourceKeyOffsets[node] + request.receivedKeys[node];
                    Timer unpackTimer;
                    unpackEntries(receivedEntries.data(), piece.keyCount, sequences->entryOffsets.data() + firstKey, sequences->entries, par.threads);
                    unpackSeconds += unpackTimer.getTimediff();
                    for (size_t key = firstKey; key < firstKey + piece.keyCount; key++) {
                        receivedEntrySize += (sequences->entries[sequences->entryOffsets[key]] != '\0') ? sequences->entryOffsets[key + 1] - sequences->entryOffsets[key] : 0;
                    }
                    request.receivedKeys[node] += piece.keyCount;
                    request.receivedPieces[node]++;
                    receivedPieceCount++;
                    receivedServedEntriesSize += piece.size;
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
                MMseqsNode::removeExchangeFile(requests + "/" + SSTR(rank) + "." + SSTR(request.number), request.number);
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
        // a fetch goes to the aligning threads once its request is received, the main thread copies its own keys; the requests of the earlier fetches are gone
        while (requestedFetches.empty() == false && (ownRequests.empty() || ownRequests.front().sequences != requestedFetches.front())) {
            {
                std::lock_guard<std::mutex> lock(serviceMutex);
                requestedFetches.front()->isReady = true;
            }
            serviceCondition.notify_all();
            requestedFetches.pop_front();
            completedFetchCount++;
            hasWorked = true;
        }
        bool isExchangeDone;
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            isReceiveDone = isLastRequested && ownRequests.empty() && requestedFetches.empty();
            importIdleReason = requestedFetches.empty() ? (isGated ? 1 : 2) : ((ownRequests.empty() == false) ? 0 : 3);
            isImportUrgent = requestedFetches.empty() == false || isGated;
            isExchangeDone = isReceiveDone && isServeDone;
        }
        if (serviceTimer.getTimediff() >= nextStatusTime) {
            nextStatusTime = serviceTimer.getTimediff() + STATUS_SECONDS;
            printStatus();
        }
        // aligning without a request is no wait, nor is the end of the exchange
        const size_t exportPasses = exportPassCount;
        if (hasWorked || isExchangeDone || exportPasses != seenExportPassCount || (requestedFetches.empty() && isGated == false && isLastRequested == false)) {
            waitWarning.reset();
        } else {
            waitWarning.check(requestedFetches.empty() == false ? served + " of other nodes" : (isGated ? assignedKeysLog : requests + " of other nodes"));
        }
        seenExportPassCount = exportPasses;
        return hasWorked;
    };
    // a pass over the requests of the other nodes
    const auto serveSequences = [&]() {
        std::unique_lock<std::mutex> taskLock(serveMutex, std::try_to_lock);
        if (taskLock.owns_lock() == false) {
            return false;
        }
        bool hasWorked = false;
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
        // a piece a pass for each requesting node
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
                Timer pollTimer;
                std::string content;
                requester.nextProgressRead = serviceTimer.getTimediff() + progressSeconds;
                progressReadCount++;
                requestProgressReadCount++;
                bool hasProgress = MMseqsNode::readRunFile(requests + "." + SSTR(node) + ".progress", content);
                exportPollingSeconds += pollTimer.getTimediff();
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
                    // the pieces of the received requests go, a served file with its last piece
                    size_t keptPieces = 0;
                    for (size_t j = 0; j < requester.pieces.size(); j++) {
                        const ServedPiece &piece = requester.pieces[j];
                        if (piece.request < received) {
                            if (--filePieceCounts[piece.file] == 0) {
                                MMseqsNode::removeExchangeFile(getServedFile(rank, piece.file), piece.file);
                                deleteCount++;
                                liveServedSize -= piece.fileSize;
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
                // a full pass leaves the rest to the next pass
                if (passEntrySize >= servedFileEntrySize || passKeyCount >= maxServedKeys) {
                    deferredRequester = (deferredRequester == UINT_MAX) ? node : deferredRequester;
                    break;
                }
                Timer servingTimer;
                // the codes of the next keys, a request is read in pieces
                std::string requestFile = requests + "/" + SSTR(node) + "." + SSTR(requester.servedCount);
                size_t observedSize;
                uint64_t observedNumber;
                int lastError;
                objectReadCount++;
                int requestFd = MMseqsNode::tryOpenExchangeFile(requestFile, requestSize, requester.servedCount, observedSize, observedNumber, lastError);
                if (requestFd < 0 && lastError != 0 && lastError != ENOENT && lastError != ESTALE && lastError != EIO) {
                    Debug(Debug::ERROR) << "Cannot open sequence request " << requestFile << ": " << strerror(lastError) << "\n";
                    EXIT(EXIT_FAILURE);
                }
                requesterCodes.resize(std::min(codeEnd - codeBegin - requester.codePos, maxServedKeys * sizeof(DBKeyType)));
                bool isRequestRead = requestFd >= 0 && MMseqsNode::tryPreadAll(requestFd, requesterCodes.data(), requesterCodes.size(), MMseqsNode::EXCHANGE_FILE_HEADER_BYTES + codeBegin + requester.codePos, lastError);
                if (requestFd >= 0) {
                    MMseqsNode::releaseExchangeCopy(requestFd);
                }
                if (requestFd >= 0 && close(requestFd) != 0 && isRequestRead) {
                    lastError = errno;
                    isRequestRead = false;
                }
                // a missing or short request is tried again in the next pass
                if (isRequestRead == false) {
                    servingSeconds += servingTimer.getTimediff();
                    if (lastError != 0 && lastError != ENOENT && lastError != ESTALE && lastError != EINTR && lastError != EIO) {
                        Debug(Debug::ERROR) << "Cannot read sequence request " << requestFile << ": " << strerror(lastError) << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    failIfMissingTooLong(missingRequestSince[node], missingRequestAttempts[node], "Cannot read complete sequence request " + requestFile
                                         + describeServedFile(requestSize, observedSize, requester.servedCount, observedNumber, lastError));
                    break;
                }
                missingRequestAttempts[node] = 0;
                // only the last byte of a code is below 0x80
                const char *codesEnd = requesterCodes.data() + requesterCodes.size();
                while (codesEnd > requesterCodes.data() && static_cast<unsigned char>(codesEnd[-1]) >= 0x80) {
                    codesEnd--;
                }
                // the unassigned entries of at least one key, up to servedFileEntrySize bytes and maxServedKeys keys a pass
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
                // the bitmap of a piece takes whole bytes
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
                // the served file of a pass fits maxServedFileSize
                servedFileData.reserve(maxServedFileSize);
                servedFileData.resize(pieceOffset + requestOffsets[keyCount]);
                // an empty slot marks an assigned key
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
                for (size_t j = 0; j < keyCount; j++) {
                    servedEntrySize += isAssigned(keys[j]) ? 0 : seqDbr.getSeqLenFromRange(entryRange, keys[j]) + 2;
                }
                servingSeconds += servingTimer.getTimediff();
                ServedPiece piece = { requester.servedCount, requester.piece, servedFileCount, 0, pieceOffset, pieceSize, keyCount };
                passPieces.push_back(std::make_pair(node, piece));
                servedSize += pieceSize;
                requesterServedSizes[node] += pieceSize;
                servedPieceCount++;
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
        if (passPieces.empty() == false) {
            Timer writeTimer;
            // the team of the caller, one on an I/O thread
            int maxThreads = 1;
#ifdef OPENMP
            maxThreads = omp_get_max_threads();
#endif
            MMseqsNode::writeExchangeFile(getServedFile(rank, servedFileCount), servedFileCount, servedFileData.data(), servedFileData.size(), "", std::vector<uint64_t>(), maxThreads);
            servedFileWriteSeconds += writeTimer.getTimediff();
            for (size_t j = 0; j < passPieces.size(); j++) {
                passPieces[j].second.fileSize = servedFileData.size();
                requesters[passPieces[j].first].pieces.push_back(passPieces[j].second);
            }
            filePieceCounts.push_back(passPieces.size());
            maxPassPieceCount = std::max(maxPassPieceCount, passPieces.size());
            liveServedSize += servedFileData.size();
            maxLiveServedSize = std::max(maxLiveServedSize, liveServedSize);
            passPieces.clear();
            servedFileData.clear();
            servedFileCount++;
            objectWriteCount++;
            isServedProgressChanged = true;
        }
        // the pieces not yet deleted, per requester
        if (isServedProgressChanged) {
            std::string content;
            for (unsigned int node = 0; node < nodes; node++) {
                for (size_t i = 0; i < requesters[node].pieces.size(); i++) {
                    const ServedPiece &piece = requesters[node].pieces[i];
                    content += SSTR(node) + "\t" + SSTR(piece.request) + "\t" + SSTR(piece.piece) + "\t" + SSTR(piece.file) + "\t" + SSTR(piece.fileSize)
                               + "\t" + SSTR(piece.offset) + "\t" + SSTR(piece.size) + "\t" + SSTR(piece.keyCount) + "\n";
                }
            }
            MMseqsNode::publishRunFile(servedProgress, content);
            objectWriteCount++;
            isServedProgressChanged = false;
        }
        if (hasWorked) {
            exportPassCount++;
        }
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            statusServingSeconds = servingSeconds;
            statusServedFileWriteSeconds = servedFileWriteSeconds;
            statusExportPollingSeconds = exportPollingSeconds;
            if (isServing || isServeDone) {
                return hasWorked;
            }
            isServeDone = true;
            ioWork++;
        }
        serviceCondition.notify_all();
        return hasWorked;
    };
    double ownKeyCopySeconds = 0;
    size_t ownKeyCopySize = 0;
    std::mutex ownKeyCopyMutex;
    size_t ownKeyAheadSize = 0;
    // the own keys of a fetch the main thread takes that the I/O threads did not copy ahead, copied by its idle aligning team
    const auto copyOwnKeys = [&](FetchedSequences &sequences) {
        Timer copyTimer;
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            ownKeyFetches.erase(std::remove(ownKeyFetches.begin(), ownKeyFetches.end(), &sequences), ownKeyFetches.end());
        }
        size_t ownKey;
        {
            // a chunk copied ahead ends before the fetch aligns
            std::lock_guard<std::mutex> copyLock(ownKeyCopyMutex);
            ownKey = sequences.ownKeyCopyEnd;
        }
        const size_t ownKeyEnd = sequences.fetchSourceKeyOffsets[rank + 1];
        copyEntries(sequences.keys.data() + ownKey, ownKeyEnd - ownKey, sequences.entryOffsets.data() + ownKey, sequences.entries, NULL, (repDbr != NULL) ? maxFetchSize / 2 : maxFetchSize);
        ownKeyCopySeconds += copyTimer.getTimediff();
        ownKeyCopySize += sequences.entryOffsets[ownKeyEnd] - sequences.entryOffsets[ownKey];
    };
    // copies the own keys of the next block of the queued fetches, the main thread copies what is left when it takes a fetch
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
                const size_t ownKeyEnd = front.fetchSourceKeyOffsets[rank + 1];
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
        copyEntries(sequences->keys.data() + first, end - first, sequences->entryOffsets.data() + first, sequences->entries, NULL, (repDbr != NULL) ? maxFetchSize / 2 : maxFetchSize);
        ownKeyAheadSize += sequences->entryOffsets[end] - sequences->entryOffsets[first];
        return true;
    };
    std::mutex clusterMutex;
    // a clustering step of a single node
    const auto clusterStep = [&]() {
        std::unique_lock<std::mutex> taskLock(clusterMutex, std::try_to_lock);
        return taskLock.owns_lock() && clustering != NULL && clustering->step() == AcceptedPairClustering::STEP_WORKED;
    };
    // a pass of an I/O thread, in window order
    const auto runIoTasks = [&]() {
        bool hasWorked = memberExchange != NULL && memberExchange->serveRequests();
        hasWorked |= clusterStep();
        hasWorked |= serveSequences();
        hasWorked |= receiveSequences();
        hasWorked |= copyOwnKeysAhead();
        hasWorked |= memberExchange != NULL && memberExchange->preloadNextRange();
        hasWorked |= fillNextPrefilterBuffer(false);
        return hasWorked;
    };
    // a piece of the work the waiting main thread waits for, else of the work other nodes wait for
    const auto runWaitTasks = [&]() {
        return receiveSequences() || copyOwnKeysAhead() || (memberExchange != NULL && memberExchange->serveRequests()) || serveSequences() || clusterStep();
    };
    // under serviceMutex
    const auto isIoDone = [&]() {
        return isAligningDone && isReceiveDone && isServeDone && isPrefilterDone && (memberExchange == NULL || (memberExchange->isServed() && memberExchange->isPreloaded()))
               && (clustering == NULL || clustering->isDone);
    };
    const auto runIoLoop = [&](bool isCountingIdle) {
        for (unsigned int polls = 0;;) {
            uint64_t seenWork;
            {
                std::lock_guard<std::mutex> lock(serviceMutex);
                if (isIoDone()) {
                    return;
                }
                seenWork = ioWork;
            }
            if (runIoTasks()) {
                polls = 0;
                continue;
            }
            // sleep longer while no node waits
            Timer idleTimer;
            std::unique_lock<std::mutex> lock(serviceMutex);
            const unsigned int maxShift = (alignNodeCount > 1 || (memberExchange != NULL && memberExchange->isServed() == false)) ? 1 : (isImportUrgent ? 4 : 7);
            // an idle clustering step is due again after the sleep of its rule
            const unsigned int clusterSleep = (clustering != NULL && clustering->isDone == false) ? clustering->idleMicroseconds.load() : UINT_MAX;
            serviceCondition.wait_for(lock, std::chrono::microseconds(std::min(10000u << std::min(polls++, maxShift), clusterSleep)), [&]() { return ioWork != seenWork; });
            if (isCountingIdle) {
                idleSeconds[importIdleReason] += idleTimer.getTimediff();
                idleCounts[importIdleReason]++;
            }
        }
    };
    const auto waitForSequences = [&](FetchedSequences &fetched, size_t doneRepOrderKey) {
        std::unique_lock<std::mutex> lock(serviceMutex);
        if (fetched.isReady) {
            return;
        }
        lock.unlock();
        publishAccepted(doneRepOrderKey, acceptedSize);
        publishedSize = acceptedSize;
        publishTime = startTimer.getTimediff();
        Timer waitTimer;
        lock.lock();
        // a later fetch at the gate is no wait for the master
        const bool waitsForMaster = isWaitingForMaster && handedOffFetches.empty() == false && handedOffFetches.front() == &fetched;
        for (unsigned int polls = 0; fetched.isReady == false;) {
            const uint64_t seenWork = ioWork;
            lock.unlock();
            // the waiting thread runs, with its idle team, the work it waits for and the work other nodes wait for
            const bool hasWorked = runWaitTasks();
            lock.lock();
            if (hasWorked == false) {
                serviceCondition.wait_for(lock, std::chrono::microseconds(10000u << std::min(polls++, 1u)), [&]() { return fetched.isReady || ioWork != seenWork; });
            }
        }
        (waitsForMaster ? waitSeconds : sequenceWaitSeconds) += waitTimer.getTimediff();
    };
    // the answers to the last member request, read at most PROGRESS_READS_PER_SECOND times a second; in between the waiting thread runs the work it waits for and the work other nodes wait for
    const auto receiveMembers = [&]() {
        while (memberExchange->tryReceive() == false) {
            Timer pollTimer;
            for (bool hasWorked = true; hasWorked && pollTimer.getTimediff() < 1.0 / PROGRESS_READS_PER_SECOND;) {
                hasWorked = runWaitTasks();
            }
            const double waitSeconds = 1.0 / PROGRESS_READS_PER_SECOND - pollTimer.getTimediff();
            if (waitSeconds > 0) {
                usleep(static_cast<useconds_t>(waitSeconds * 1000000));
            }
        }
    };
    std::vector<std::thread> ioThreads;
    for (int i = 0; isAligningNode && i < ioThreadCount; i++) {
        ioThreads.emplace_back([&]() {
#ifdef OPENMP
            omp_set_num_threads(1);
#endif
            runIoLoop(true);
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
    const double firstFetchSeconds = startTimer.getTimediff();
    // the fetch work and waits before a fetch aligns
    double fetchOtherSeconds = firstFetchSeconds;
    size_t alignedFetchCount = 0;
    std::vector<std::pair<size_t, double> > fetchWindowCells;
    std::atomic<size_t> staleRepSize(0);
    // the planned bytes per aligning second of the last fetch with planned bytes
    double plannedBytesPerSecond = 0;
    size_t maxFetchesInFlight = 0;
    size_t fetchesInFlightSum = 0;
    const size_t alignThreadCount = par.threads - ioThreadCount;
    while (currentSequences.isLast == false) {
        Timer otherTimer;
        // one fetch ahead always; more while those in flight align in READ_SECONDS, fit the fetch memory and keep the prefilter buffer of current
        for (;;) {
            const bool isFirstAhead = fetchesInFlight.empty();
            if (isFirstAhead == false) {
                const FetchedSequences &newest = *fetchesInFlight.back();
                size_t inFlightPlannedSize = 0;
                for (size_t i = 0; i < fetchesInFlight.size(); i++) {
                    inFlightPlannedSize += fetchesInFlight[i]->plannedSize;
                }
                // more pending pairs come only after aligning
                if (newest.isLast || (newest.end == newest.begin && newest.pendingPairCount == 0) || idleFetches.empty()
                    || inFlightPlannedSize >= READ_SECONDS * plannedBytesPerSecond) {
                    break;
                }
            }
            // the next fetch holds pending pairs up to the limit and a record on top
            const size_t limitedFetchSize = (memberDbr != NULL) ? std::max(fetchBytesLimit, maxPendingPairSize) + repTargetsReader.maxRecordSize
                                                                : std::max(fetchBytesLimit, repTargetsReader.maxRecordSize);
            // idle slots free their buffers until the next fetch fits, but the slot taken next keeps them
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
            }
            if (isFirstAhead == false && fetchMemory + limitedFetchSize > fetchBudget) {
                break;
            }
            // the next prefilter buffer must be filled, and the fill then refills the one before the buffer of current
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
                // a reused slot holds the state of its last fetch
                selected.isReady = false;
                selected.acceptedSize = acceptedSize;
                handedOffFetches.push_back(&selected);
                ownKeyFetches.push_back(&selected);
                ioWork++;
            }
            serviceCondition.notify_all();
            fetchesInFlight.push_back(&selected);
        }
        maxFetchesInFlight = std::max(maxFetchesInFlight, fetchesInFlight.size());
        fetchesInFlightSum += fetchesInFlight.size();
        double nextOtherSeconds = otherTimer.getTimediff();
        FetchedSequences &nextSequences = *fetchesInFlight.front();
        Timer alignTimer;
        const size_t nextRepKey = getFirstRepKey(nextSequences);
        const std::vector<RepTargetsRecord> &records = prefilterRecords[currentSequences.prefilterBuffer];
        const size_t recordCount = currentSequences.end - currentSequences.begin;
        // batches of about equal alignment cells; a large rep spans several
        size_t maxBatches = alignThreadCount * (isSwipeEnabled ? 64 * 16 / SwipeExtender::LANES : 64);
        if (isSwipeEnabled) {
            // estimate the SWIPE jobs of this fetch
            size_t swipeJobCount = 0;
            for (int thread = 0; thread < par.threads; thread++) {
                swipeJobCount += (threadTargetAligners[thread] != NULL) ? threadTargetAligners[thread]->swipeTargets.targetPairs : 0;
            }
            size_t fetchSwipeJobs = static_cast<size_t>(static_cast<uint128>(currentSequences.targetCount) * swipeJobCount / std::max((size_t) 1, examinedTargetPairCount.load()));
            size_t swipeBatches = std::max(8 * alignThreadCount, fetchSwipeJobs / (64 * SwipeExtender::LANES));
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
        const size_t fetchFirstAlignedPairs = alignedTargetPairCount.load();
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            FetchedKeyCursor targetCursor(nodes + 1);
            std::vector<ClusterPair> pairs;
            std::vector<uint64_t> alignedPairWords;
            size_t windowCursor = SIZE_MAX;
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
                // without I/O threads the main thread serves the other nodes, requests and receives the next fetches and, on a single node, clusters between the batches
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
                size_t repSkippedCount = 0;
                size_t repSlotSkippedCount = 0;
                size_t targetSkippedCount = 0;
                size_t uncoverableCount = 0;
                size_t repSkippedRecordCount = 0;
                size_t staleSize = 0;
                for (size_t i = batchOffsets[batch]; i < batchOffsets[batch + 1] + (batchFirstTargets[batch + 1] > 0); i++) {
                    const RepTargetsRecord &record = records[i];
                    unsigned int firstTarget = (i == batchOffsets[batch]) ? batchFirstTargets[batch] : 0;
                    const unsigned int endTarget = (i == batchOffsets[batch + 1]) ? batchFirstTargets[batch + 1] : record.targetCount;
                    examinedCount += endTarget - firstTarget;
                    const size_t repKey = record.repOrderKeyFirstTarget.repKey % paddedDbSize;
                    if (isAssigned(record.repOrderKeyFirstTarget.repKey)) {
                        repSkippedCount += endTarget - firstTarget;
                        repSkippedRecordCount++;
                        // a rep another node served for this fetch, at its first record
                        if (isCountingRequests && firstTarget == 0 && (i == currentSequences.begin || records[i - 1].repOrderKeyFirstTarget.repKey != record.repOrderKeyFirstTarget.repKey)
                            && findFetchedKey(currentSequences, nodes, repKey) == NULL) {
                            size_t repBlock = SIZE_MAX;
                            const size_t repSource = getFetchSource(currentSequences, repKey, repBlock);
                            size_t repLength = 0;
                            const char *repData = (repSource != rank) ? findFetchedKey(currentSequences, repSource, repKey, &repLength) : NULL;
                            staleSize += (repData != NULL && repData[0] != '\0') ? repLength + 2 : 0;
                        }
                        continue;
                    }
                    size_t repLength = 0;
                    const char *repData = getFetchedSequence(currentSequences, repKey, &repLength);
                    if (repData == NULL) {
                        repSkippedCount += endTarget - firstTarget;
                        repSlotSkippedCount += endTarget - firstTarget;
                        repSkippedRecordCount++;
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
                        if (isLoggingAlignedPairs) {
                            alignedPairWords.push_back(repKey);
                            alignedPairWords.push_back(targetKey);
                        }
                        unsigned short diagonal = 0;
                        if (alignTarget(par, aligners->blockAligner, blockAlignerInitialized, aligners->query, aligners->target, scoring.scorePerColThreshold, diagonal, NULL, 0, NULL,
                                        isSwipeEnabled ? &aligners->swipeTargets : NULL)) {
                            const ClusterPair pair = { record.repOrderKeyFirstTarget.repKey, targetKey };
                            pairs.push_back(pair);
                        }
                    }
                    repSkippedCount += endTarget - j;
                }
                staleRepSize += staleSize;
                if (alignedPairWords.empty() == false) {
                    std::lock_guard<std::mutex> lock(alignedPairLogMutex);
                    if (alignedPairLog != NULL && fwrite(alignedPairWords.data(), sizeof(uint64_t), alignedPairWords.size(), alignedPairLog) != alignedPairWords.size()) {
                        Debug(Debug::WARNING) << "Cannot write " << alignedPairFile << ", the aligned pairs are not logged any more\n";
                        fclose(alignedPairLog);
                        alignedPairLog = NULL;
                    }
                    alignedPairWords.clear();
                }
                examinedTargetPairCount += examinedCount;
                repSkippedTargetPairCount += repSkippedCount;
                repSlotSkippedTargetPairCount += repSlotSkippedCount;
                targetSkippedTargetPairCount += targetSkippedCount;
                uncoverableTargetPairCount += uncoverableCount;
                const size_t alignedCount = examinedCount - repSkippedCount - targetSkippedCount - uncoverableCount;
                alignedTargetPairCount += alignedCount;
                // a batch counts for the window of its first rep
                if (batchOffsets[batch] < currentSequences.end) {
                    WindowCounters &counters = windowCounters[findWindow(windowFirstKeys, records[batchOffsets[batch]].repOrderKeyFirstTarget.repKey % paddedDbSize, windowCursor)];
                    __atomic_fetch_add(&counters.examinedPairs, examinedCount, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&counters.alignedPairs, alignedCount, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&counters.targetSkippedPairs, targetSkippedCount, __ATOMIC_RELAXED);
                    __atomic_fetch_add(&counters.repSkippedRecords, repSkippedRecordCount, __ATOMIC_RELAXED);
                }
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
                        const double now = startTimer.getTimediff();
                        // one thread publishes at a time
                        if ((acceptedSize - publishedSize >= 4 * 1024 * 1024 || now - publishTime >= publishPeriod.load()) && publishLock.try_lock()) {
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
                        lastBatchReadTime = assignedKeysTimer.getTimediff();
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
            const size_t batchMemberLimit = std::min<size_t>(4096, std::max<size_t>(pendingMemberOffsets[pairCount] / (par.threads * 4), 256));
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
                    // without the extensions a batch holds the checks of one rep
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
            receiveMembers();
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
            // pending members and parts fit memberKeyBudget: a pair of e entry bytes has fewer than e / 2 members and at most as many parts, one at least
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
                    receiveMembers();
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
                    receiveMembers();
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
            publishTime = startTimer.getTimediff();
            if (nextRepKey < orderKeyEnd) {
                std::lock_guard<std::mutex> assignedKeysLock(assignedKeysMutex);
                readAssignedKeys(true);
            }
        }
        const double fetchAlignSeconds = alignTimer.getTimediff();
        alignedCells += static_cast<double>(fetchAlignmentCells);
        alignSeconds += fetchAlignSeconds;
        alignedCellsPerSecond.store(alignedCells / std::max(alignSeconds, 1e-3));
        // a fetch aligns in about half of READ_SECONDS at the rate of the last fetch, so that the fetches in flight hold two
        if (currentSequences.plannedSize > 0) {
            plannedBytesPerSecond = currentSequences.plannedSize / std::max(fetchAlignSeconds, 1e-3);
            if (isFetchBytesFixed == false) {
                maxFetchBytesLimit = std::max(minFetchSize, std::min(maxFetchSize, static_cast<size_t>(READ_SECONDS / 2.0 * plannedBytesPerSecond)));
                fetchBytesLimit = std::min(fetchBytesLimit, maxFetchBytesLimit);
            }
        }

        // the seconds of a fetch go to its windows by their cells, a fetch without records to the window of its first rep
        fetchWindowCells.clear();
        size_t window = SIZE_MAX;
        for (size_t i = 0; i < recordCount; i++) {
            findWindow(windowFirstKeys, records[currentSequences.begin + i].repOrderKeyFirstTarget.repKey % paddedDbSize, window);
            if (fetchWindowCells.empty() || fetchWindowCells.back().first != window) {
                fetchWindowCells.emplace_back(window, 0);
            }
            fetchWindowCells.back().second += recordAlignmentCells[i];
        }
        if (fetchAlignmentCells == 0) {
            const size_t frontKey = std::min(currentSequences.firstPendingRepKey, getFirstRepKey(currentSequences));
            fetchWindowCells.assign(1, std::make_pair((frontKey < orderKeyEnd) ? findWindow(windowFirstKeys, frontKey % paddedDbSize, window) : windowFirstKeys.size() - 1, 1.0));
        }
        double fetchCells = 0;
        for (size_t j = 0; j < fetchWindowCells.size(); j++) {
            fetchCells += fetchWindowCells[j].second;
        }
        for (size_t j = 0; j < fetchWindowCells.size(); j++) {
            WindowCounters &counters = windowCounters[fetchWindowCells[j].first];
            counters.alignSeconds += fetchAlignSeconds * fetchWindowCells[j].second / fetchCells;
            counters.otherSeconds += fetchOtherSeconds * fetchWindowCells[j].second / fetchCells;
            counters.fetchCount += counters.lastFetch != alignedFetchCount;
            counters.lastFetch = alignedFetchCount;
            counters.prefilterBufferCount += recordCount > 0 && counters.lastPrefilterBuffer != currentSequences.prefilterBufferNumber;
            counters.lastPrefilterBuffer = (recordCount > 0) ? currentSequences.prefilterBufferNumber : counters.lastPrefilterBuffer;
        }
        alignedFetchCount++;
        FetchSizeCounters &fetchSize = fetchSizeCounters[(currentSequences.plannedSize > 0) ? 63 - __builtin_clzll(currentSequences.plannedSize) : 0];
        fetchSize.fetchCount++;
        fetchSize.plannedSize += currentSequences.plannedSize;
        fetchSize.unassignedRecordSize += currentSequences.unassignedRecordSize;
        fetchSize.alignedPairs += alignedTargetPairCount.load() - fetchFirstAlignedPairs;
        fetchSize.alignSeconds += fetchAlignSeconds;

        // the reps of pending pairs are not done
        otherTimer.reset();
        copyOwnKeys(nextSequences);
        waitForSequences(nextSequences, getFirstUndoneRepKey(nextRepKey));
        fetchOtherSeconds = nextOtherSeconds + otherTimer.getTimediff();
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
    publishAccepted(orderKeyEnd, acceptedSize);
    if (fclose(acceptedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << acceptedFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (alignedPairLog != NULL && fclose(alignedPairLog) != 0) {
        Debug(Debug::WARNING) << "Cannot close " << alignedPairFile << ", the aligned pair log may be incomplete\n";
    }
    // serve until every aligning node is done
    if (isAligningNode) {
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            isAligningDone = true;
            ioWork++;
        }
        serviceCondition.notify_all();
        runIoLoop(ioThreadCount == 0);
        // the I/O threads may sleep before they see the end
        {
            std::lock_guard<std::mutex> lock(serviceMutex);
            ioWork++;
        }
        serviceCondition.notify_all();
        for (size_t i = 0; i < ioThreads.size(); i++) {
            ioThreads[i].join();
        }
        printStatus();
        MMseqsNode::addPhaseValue(phaseValues, "Sequence exchange bytes", SSTR(servedSize) + " served for " + SSTR(servedEntrySize) + " entry bytes, " + SSTR(receivedServedEntriesSize) + " received, "
                                  + SSTR(requestCount) + " requests, " + SSTR(receivedPieceCount) + " received pieces");
        MMseqsNode::addPhaseSeconds(phaseValues, "service: serving the requests", servingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: loading the source sequences", sourceLoadSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: packing the served entries", packSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: writing the served files", servedFileWriteSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: reading the served entries", readingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: unpacking the served entries", unpackSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: polling", importPollingSeconds + exportPollingSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Requested entry bytes", SSTR(requestedEntrySize) + " requested, " + SSTR(receivedEntrySize) + " received not assigned, of " + SSTR(rangeByteOffsets[rangeCount]) + " sequence bytes");
        MMseqsNode::addPhaseValue(phaseValues, "Sequence service requests", SSTR(progressReadCount.load()) + " polls, " + SSTR(objectReadCount.load()) + " object reads, " + SSTR(objectWriteCount.load()) + " object writes, "
                                  + SSTR(deleteCount.load()) + " deletes in " + SSTR(static_cast<size_t>(serviceTimer.getTimediff())) + " s");
        MMseqsNode::addPhaseValue(phaseValues, "Request progress reads", SSTR(requestProgressReadCount) + ", unchanged: " + SSTR(requestProgressReadCount - requestProgressChangeCount)
                                  + ", changes: " + SSTR(requestProgressChangeCount) + ", request listings: " + SSTR(requestListingCount));
        MMseqsNode::addPhaseValue(phaseValues, "Served progress reads", SSTR(progressReadCount - requestProgressReadCount) + ", moved up by a listing: " + SSTR(servedListingReadCount)
                                  + ", served listings: " + SSTR(servedListingCount));
        MMseqsNode::addPhaseValue(phaseValues, "Served files", SSTR(servedFileCount) + ", at most " + SSTR(maxLiveServedSize) + " bytes not yet deleted, " + SSTR(servedPieceCount) + " pieces, at most "
                                  + SSTR(maxPassPieceCount) + " pieces a file");
    }
    delete[] currentSequences.entries;
    for (size_t i = 0; i < fetchSlots.size(); i++) {
        delete[] fetchSlots[i].entries;
    }
    for (size_t page = 0; page < assignedPages.size(); page++) {
        delete[] assignedPages[page];
    }
    MMseqsNode::addPhaseValue(phaseValues, "Block read bytes", SSTR(seqDbr.getBlockReadSize()));
    MMseqsNode::addPhaseValue(phaseValues, "Key read bytes", SSTR(seqDbr.getKeyReadSize()));
    MMseqsNode::addPhaseValue(phaseValues, "Local block read bytes", SSTR(seqDbr.getLocalBlockReadSize()));
    MMseqsNode::addPhaseValue(phaseValues, "Local key read bytes", SSTR(seqDbr.getLocalKeyReadSize()));
    if (repDbr != NULL) {
        MMseqsNode::addPhaseValue(phaseValues, "Rep block read bytes", SSTR(repDbr->getBlockReadSize()));
        MMseqsNode::addPhaseValue(phaseValues, "Rep key read bytes", SSTR(repDbr->getKeyReadSize()));
        MMseqsNode::addPhaseValue(phaseValues, "Rep block cache", SSTR(repBlockCacheSize) + " bytes");
        repDbr->close();
        delete repDbr;
    }
    // the master still finds the nodes of keys in the blocks of seqDbr
    seqDbr.releaseDataPages();
    if (isAligningNode) {
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the master", waitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the sequences", sequenceWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the prefilter", prefilterWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "aligning the fetches", alignSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "copying the own keys", ownKeyCopySeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Own key bytes", SSTR(ownKeyCopySize) + " copied by the main thread, " + SSTR(ownKeyAheadSize) + " ahead");
        MMseqsNode::addPhaseSeconds(phaseValues, "filling the prefilter buffers", prefilterFillSeconds);
        const auto formatFillSeconds = [](double seconds) { return SSTR(static_cast<float>(seconds), 3) + " s"; };
        MMseqsNode::addPhaseValue(phaseValues, "Prefilter fill", SSTR(fillCounters.bufferCount) + " buffers, " + SSTR(fillCounters.mainBufferCount) + " finished by the waiting main thread, "
                                  + SSTR(fillCounters.orderedBufferCount) + " in order without a sort; " + SSTR(fillCounters.entryCount) + " entries, at most " + SSTR(fillCounters.maxBufferEntryCount)
                                  + " in a buffer, the largest entry at most " + SSTR(fillCounters.maxEntryShare) + " per mille of its buffer's pref bytes; reading "
                                  + formatFillSeconds(fillCounters.stepSeconds[PrefilterFillCounters::READ_STEP]) + ", parsing " + formatFillSeconds(fillCounters.stepSeconds[PrefilterFillCounters::PARSE_STEP])
                                  + ", sorting " + formatFillSeconds(fillCounters.stepSeconds[PrefilterFillCounters::SORT_STEP]) + "; main blocked on I/O steps "
                                  + formatFillSeconds(fillCounters.blockedSeconds[PrefilterFillCounters::READ_STEP] + fillCounters.blockedSeconds[PrefilterFillCounters::PARSE_STEP] + fillCounters.blockedSeconds[PrefilterFillCounters::SORT_STEP])
                                  + " (reading " + formatFillSeconds(fillCounters.blockedSeconds[PrefilterFillCounters::READ_STEP]) + ", parsing " + formatFillSeconds(fillCounters.blockedSeconds[PrefilterFillCounters::PARSE_STEP])
                                  + ", sorting " + formatFillSeconds(fillCounters.blockedSeconds[PrefilterFillCounters::SORT_STEP]) + "); " + formatFillSeconds(fillCounters.startDelaySeconds[0])
                                  + " from the buffer switches to the first fill steps of " + SSTR(fillCounters.startedBufferCount[0]) + " fills started by the I/O threads, "
                                  + formatFillSeconds(fillCounters.startDelaySeconds[1]) + " of " + SSTR(fillCounters.startedBufferCount[1]) + " started by the waiting main thread");
        MMseqsNode::addPhaseValue(phaseValues, "Fetches with fewer SWIPE batches", SSTR(fewerBatchFetchCount) + " of " + SSTR(fetchCount));
        MMseqsNode::addPhaseValue(phaseValues, "Fetches in flight", "at most " + SSTR(maxFetchesInFlight) + ", mean " + SSTR(static_cast<float>(fetchesInFlightSum) / std::max<size_t>(1, alignedFetchCount), 2));
        MMseqsNode::addPhaseValue(phaseValues, "Assigned keys read", SSTR(assignedKeyCount) + " in " + SSTR(assignedKeysSize) + " bytes");
        MMseqsNode::addPhaseSeconds(phaseValues, "the first fetch", firstFetchSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Gate slack", "at most " + SSTR(maxGateSlackKeys) + " keys and " + SSTR(maxGateSlackWindows) + " windows, mean "
                                  + SSTR(static_cast<size_t>(gateSlackKeySeconds / std::max(gateSlackSeconds, 1e-9))) + " keys and " + SSTR(static_cast<size_t>(gateSlackWindowSeconds / std::max(gateSlackSeconds, 1e-9))) + " windows over " + formatSeconds(gateSlackSeconds));
        MMseqsNode::addPhaseValue(phaseValues, "Process threads", "at most " + SSTR(maxProcessThreads) + " incl. io_uring workers, " + SSTR(maxThreadsBesidesRingWorkers)
                                  + " besides them (BatchReader read threads included) in " + SSTR(threadSampleCount) + " status samples, --threads " + SSTR(par.threads));
        size_t requesterCount = 0;
        size_t servedSizeSum = 0;
        size_t maxRequesterServedSize = 0;
        for (size_t node = alignGroupNodeBegin; node < alignGroupNodeEnd; node++) {
            requesterCount += node != rank;
            servedSizeSum += (node != rank) ? requesterServedSizes[node] : 0;
            maxRequesterServedSize = std::max(maxRequesterServedSize, (node != rank) ? requesterServedSizes[node] : 0);
        }
        MMseqsNode::addPhaseValue(phaseValues, "Served bytes per requesting node", "at most " + SSTR(maxRequesterServedSize) + ", mean " + SSTR(servedSizeSum / std::max<size_t>(1, requesterCount))
                                  + " over " + SSTR(requesterCount) + " nodes");
        MMseqsNode::addPhaseValue(phaseValues, "Requested entry bytes by use", SSTR(requestedRepSize) + " for reps, " + SSTR(requestedPendingSize) + " for members and pending pairs");
        MMseqsNode::addPhaseValue(phaseValues, "Received rep bytes assigned before they aligned", SSTR(staleRepSize.load()));
        for (size_t bucket = 0; bucket < 64; bucket++) {
            const FetchSizeCounters &fetchSize = fetchSizeCounters[bucket];
            if (fetchSize.fetchCount > 0) {
                MMseqsNode::addPhaseValue(phaseValues, "Fetches from " + SSTR(1ULL << bucket) + " planned bytes", SSTR(fetchSize.fetchCount) + " fetches, " + SSTR(fetchSize.plannedSize) + " planned bytes, "
                                          + SSTR(fetchSize.unassignedRecordSize) + " bytes of unassigned records, " + SSTR(fetchSize.alignedPairs) + " aligned pairs, " + formatSeconds(fetchSize.alignSeconds) + " aligning");
            }
        }
        // the cells the node loaded, with one align node the cells of .partitions
        if (targetSplitReader != NULL) {
            for (size_t i = 0, window = SIZE_MAX; i < windows.size(); i++) {
                windowCounters[findWindow(windowFirstKeys, windows[i].firstKey, window)].alignmentCells += targetSplitReader->windowCells[i];
            }
        } else {
            for (size_t i = 0, window = SIZE_MAX; i < partitions.size(); i++) {
                windowCounters[findWindow(windowFirstKeys, partitions[i].firstKey, window)].alignmentCells += partitions[i].alignmentCells;
            }
        }
        // deciles of the windows of this node in key order
        std::vector<size_t> nodeWindows;
        for (size_t window = 0; window < windowCounters.size(); window++) {
            if (windowCounters[window].alignmentCells > 0 || windowCounters[window].examinedPairs > 0 || windowCounters[window].fetchCount > 0) {
                nodeWindows.push_back(window);
            }
        }
        for (size_t decile = 0; decile < 10; decile++) {
            const size_t first = decile * nodeWindows.size() / 10;
            const size_t end = (decile + 1) * nodeWindows.size() / 10;
            WindowCounters sum;
            for (size_t i = first; i < end; i++) {
                const WindowCounters &counters = windowCounters[nodeWindows[i]];
                sum.examinedPairs += counters.examinedPairs;
                sum.alignedPairs += counters.alignedPairs;
                sum.targetSkippedPairs += counters.targetSkippedPairs;
                sum.repSkippedRecords += counters.repSkippedRecords;
                sum.alignmentCells += counters.alignmentCells;
                sum.fetchCount += counters.fetchCount;
                sum.prefilterBufferCount += counters.prefilterBufferCount;
                sum.alignSeconds += counters.alignSeconds;
                sum.otherSeconds += counters.otherSeconds;
            }
            if (end > first) {
                MMseqsNode::addPhaseValue(phaseValues, "Window decile " + SSTR(decile + 1), SSTR(end - first) + " windows, " + SSTR(sum.alignmentCells) + " cells, "
                                          + SSTR(sum.examinedPairs) + " examined pairs, " + SSTR(sum.alignedPairs) + " aligned, " + SSTR(sum.targetSkippedPairs) + " skipped by target, "
                                          + SSTR(sum.repSkippedRecords) + " records skipped by rep, " + formatSeconds(sum.alignSeconds) + " aligning, " + formatSeconds(sum.otherSeconds)
                                          + " not aligning, " + SSTR(sum.fetchCount) + " fetches, " + SSTR(sum.prefilterBufferCount) + " prefilter buffers");
            }
        }
    }
    if (memberLookup != NULL) {
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member pairs", SSTR(multiFetchPairCount) + " over several fetches, " + SSTR(slicedPairCount) + " in slices");
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member pairs with several parts in a fetch", SSTR(multiPartPairCount) + ", " + SSTR(unstartedMemberCount) + " members not aligned after their pair failed");
        MMseqsNode::addPhaseSeconds(phaseValues, "looking up the cluster members", memberLookupSeconds);
    }
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

// the log of the master, read by the other nodes from S3 in aws mode
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
    Timer timer;
    std::string phaseValues;
    Timer phaseValueTimer;
    size_t alignNodeCount = MMseqsNode::count - MMseqsNode::getFirstAlignNode();
    const bool hasTargetSplits = alignNodeCount > 1;
    alignGroupNodeBegin = MMseqsNode::getFirstAlignNode();
    alignGroupNodeEnd = MMseqsNode::count;
    // the first .partitions line holds the align group count kmermatcher chose, the second the target split blocks
    MMseqsNode::fetchFile(par.db2 + ".partitions");
    std::string partitionsData = MMseqsNode::readFile(par.db2 + ".partitions");
    const size_t alignGroupCount = Util::fast_atoi<size_t>(partitionsData.c_str());
    char *partitionLine = Util::skipLine((char *) partitionsData.c_str());
    for (; hasTargetSplits && *partitionLine != '\n'; partitionLine += Util::skipWhitespace(partitionLine)) {
        targetSplitBlockOffsets.push_back(Util::fast_atoi<size_t>(partitionLine));
        partitionLine += Util::skipNoneWhitespace(partitionLine);
    }
    std::vector<PrefPartition> partitions;
    std::vector<PrefWindow> windows;
    uint128 totalAlignmentCells = 0;
    size_t totalPrefSize = 0;
    for (char *line = partitionLine + (hasTargetSplits ? 1 : 0); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[5];
        Util::getWordsOfLine(line, columns, 5);
        const size_t prefSize = Util::fast_atoi<size_t>(columns[2]);
        const size_t alignmentCells = Util::fast_atoi<size_t>(columns[3]);
        const size_t node = Util::fast_atoi<size_t>(columns[4]);
        if (hasTargetSplits) {
            // a window of one entry until the windows are cut
            const size_t entry = Util::fast_atoi<size_t>(columns[1]);
            const PrefWindow window = { Util::fast_atoi<size_t>(columns[0]), node, entry, entry + 1, alignmentCells };
            windows.push_back(window);
        } else {
            const PrefPartition partition = { Util::fast_atoi<size_t>(columns[0]), Util::fast_atoi<size_t>(columns[1]), prefSize, alignmentCells, node };
            partitions.push_back(partition);
        }
        totalAlignmentCells += alignmentCells;
        totalPrefSize += prefSize;
    }
    // block table for the target splits and the local copy
    DBReader<DBKeyType> blockDbr(par.db1.c_str(), par.db1Index.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
    blockDbr.open(DBReader<DBKeyType>::NOSORT);
    blockDbr.setBlockCacheSize(0);
    std::vector<size_t> groupNodeOffsets(alignGroupCount + 1);
    if (hasTargetSplits) {
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

        // the align nodes in groups, the master is in no group
        for (size_t group = 0; group <= alignGroupCount; group++) {
            groupNodeOffsets[group] = group * alignNodeCount / alignGroupCount;
        }
        const size_t alignNode = MMseqsNode::rank - std::min<size_t>(MMseqsNode::rank, MMseqsNode::getFirstAlignNode());
        alignGroup = (MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) ? std::upper_bound(groupNodeOffsets.begin(), groupNodeOffsets.end(), alignNode) - groupNodeOffsets.begin() - 1 : alignGroupCount;

        // the splits go over the group nodes back and forth (0 1 2 2 1 0 0 1 ...); the master spreads the splits over the groups
        targetSplitAlignNodes.resize(targetSplitCount);
        for (size_t group = 0; group < alignGroupCount; group++) {
            const size_t groupNodeCount = groupNodeOffsets[group + 1] - groupNodeOffsets[group];
            std::vector<size_t> nodeSizes(groupNodeCount, 0);
            for (size_t split = 0; split < targetSplitCount; split++) {
                const size_t groupNode = ((split / groupNodeCount) % 2 == 0) ? split % groupNodeCount : groupNodeCount - 1 - split % groupNodeCount;
                nodeSizes[groupNode] += splitSizes[split];
                if (group == ((alignGroup < alignGroupCount) ? alignGroup : split % alignGroupCount)) {
                    targetSplitAlignNodes[split] = MMseqsNode::getFirstAlignNode() + groupNodeOffsets[group] + groupNode;
                }
            }
            if (group == alignGroup) {
                const size_t groupNode = alignNode - groupNodeOffsets[group];
                MMseqsNode::addPhaseValue(phaseValues, "Target splits", SSTR(std::count(targetSplitAlignNodes.begin(), targetSplitAlignNodes.end(), MMseqsNode::rank)) + " of " + SSTR(targetSplitCount)
                                          + " with " + SSTR(nodeSizes[groupNode]) + " of " + SSTR(totalSize) + " sequence bytes");
            }
        }
        if (alignGroup < alignGroupCount) {
            alignGroupNodeBegin = MMseqsNode::getFirstAlignNode() + groupNodeOffsets[alignGroup];
            alignGroupNodeEnd = MMseqsNode::getFirstAlignNode() + groupNodeOffsets[alignGroup + 1];
        }
    }
    // a node runs at most --threads threads, the aligning or clustering team and its I/O threads: two where the sequences go to and come from
    // the other nodes of the align group or where the node serves members to the align nodes besides its other I/O, else one
    const bool isExchangingSequences = MMseqsNode::rank >= MMseqsNode::getFirstAlignNode() && alignGroupNodeEnd - alignGroupNodeBegin > 1;
    // member requests cross the align groups, every align node but the node itself asks
    const bool isServingMembers = par.filterCluDBFile.empty() == false && MMseqsNode::count > 1 && (MMseqsNode::isMaster() || alignNodeCount > 1);
    const int ioThreadCount = std::min((isExchangingSequences || isServingMembers) ? 2 : 1, par.threads - 1);
    // the teams leave room for the I/O threads from the start, so that no idle pool thread of a larger team stays
#ifdef OPENMP
    omp_set_num_threads(par.threads - ioThreadCount);
#endif
    // a master that only clusters reads no sequences; with S3 the local sparse copies replace the copies
    const bool isCopyingServedBlocks = par.localTmp.empty() == false && MMseqsNode::rank >= MMseqsNode::getFirstAlignNode() && S3Client::isEnabled() == false;
    const std::vector<std::string> dataFiles = FileUtil::findDatafiles(par.db1.c_str());
    // seqDbr, its local copy, repDbr and the member DB of iteration 2 open each data file, O_DIRECT doubles the fds, up to five fds per own target split
    struct rlimit openFileLimit = {0, 0};
    getrlimit(RLIMIT_NOFILE, &openFileLimit);
    size_t minOpenFiles = ((par.filterCluDBFile.empty() ? 4 : 6) + (isCopyingServedBlocks ? 2 : 0)) * dataFiles.size() + 64 + 2 * MMseqsNode::count + 5 * std::count(targetSplitAlignNodes.begin(), targetSplitAlignNodes.end(), MMseqsNode::rank);
    if (minOpenFiles > openFileLimit.rlim_cur) {
        Debug(Debug::ERROR) << "Raise the open file limit (ulimit -n) to at least " << minOpenFiles << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::string cluPartDb = par.filterCluDBFile;
    std::vector<std::string> localCopies;
    std::string localDir;
    std::string localSeqDb;
    std::vector<char> localBlocks;
    if (isCopyingServedBlocks) {
        std::string localTmpProblem = getLocalTmpProblem(par);
        if (localTmpProblem.empty()) {
            // the blocks of this rank in key order
            const std::vector<size_t> &blockKeyOffsets = blockDbr.getBlockKeyOffsets();
            const DBReader<DBKeyType>::Index *blockIndex = blockDbr.getIndex();
            std::vector<std::vector<std::pair<size_t, size_t> > > servedFileRanges(dataFiles.size());
            std::vector<size_t> servedBlocks;
            std::vector<size_t> servedBlockOffsets(1, 0);
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
                servedBlocks.push_back(block);
                servedBlockOffsets.push_back(servedBlockOffsets.back() + end - begin);
            }
            const size_t servedSize = servedBlockOffsets.back();
            // a copy left by a killed run does not take the free space
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
            const size_t cluSize = cluFileRanges.empty() ? 0 : cluFileRanges[MMseqsNode::rank][0].second;
            // SIZE_MAX if statvfs failed
            const size_t freeSize = FileUtil::getFreeSpace(par.localTmp.c_str());
            size_t localTmpSize = (freeSize == SIZE_MAX) ? 0 : freeSize;
            // MMSEQS_LOCAL_COPY_BYTES=<n> limits the copies for measurements
            const char *localCopyBytes = getenv("MMSEQS_LOCAL_COPY_BYTES");
            char *localCopyBytesEnd = NULL;
            const size_t maxLocalCopySize = (localCopyBytes != NULL) ? strtoull(localCopyBytes, &localCopyBytesEnd, 10) : 0;
            if (localCopyBytes != NULL && (localCopyBytesEnd == localCopyBytes || *localCopyBytesEnd != '\0')) {
                Debug(Debug::WARNING) << "Ignoring MMSEQS_LOCAL_COPY_BYTES=" << localCopyBytes << ", it must be a number of bytes\n";
            } else if (localCopyBytes != NULL) {
                localTmpSize = std::min(localTmpSize, maxLocalCopySize);
            }
            // the open sets of the master and the files of the run need what the copies leave
            struct stat localTmpStat, tmpStat;
            const bool isTmpDevice = stat(par.localTmp.c_str(), &localTmpStat) == 0 && stat(FileUtil::dirName(par.db3).c_str(), &tmpStat) == 0 && localTmpStat.st_dev == tmpStat.st_dev;
            const size_t maxCopySize = localTmpSize;
            if (((MMseqsNode::isMaster() && par.clusteringMode == Parameters::SET_COVER) || isTmpDevice) && localTmpSize < cluSize + servedSize) {
                localTmpSize = 0;
            }
            // the cluster part whole or not at all, then the blocks while they fit
            size_t cluCopySize = 0;
            if (cluSize > 0 && cluSize <= localTmpSize) {
                const size_t firstCopy = localCopies.size();
                cluPartDb = FileUtil::copyToLocalDir(par.filterCluDBFile, localDir, par.threads - ioThreadCount, localCopies, cluCopySize, cluFileRanges, localTmpSize);
                if (cluCopySize < cluSize) {
                    for (size_t i = firstCopy; i < localCopies.size(); i++) {
                        FileUtil::remove(localCopies[i].c_str());
                    }
                    localCopies.resize(firstCopy);
                    cluPartDb = par.filterCluDBFile;
                    cluCopySize = 0;
                }
            }
            size_t seqCopySize = 0;
            const size_t firstSeqCopy = localCopies.size();
            if (servedBlocks.empty() == false && localTmpSize - cluCopySize >= servedBlockOffsets[1]) {
                localSeqDb = FileUtil::copyToLocalDir(par.db1, localDir, par.threads - ioThreadCount, localCopies, seqCopySize, servedFileRanges, localTmpSize - cluCopySize);
            }
            // a block is local once all its bytes are
            const size_t localBlockCount = std::upper_bound(servedBlockOffsets.begin(), servedBlockOffsets.end(), seqCopySize) - servedBlockOffsets.begin() - 1;
            if (localBlockCount == 0) {
                // less than a block was copied
                for (size_t i = firstSeqCopy; i < localCopies.size(); i++) {
                    FileUtil::remove(localCopies[i].c_str());
                }
                localCopies.resize(firstSeqCopy);
                if (localCopies.empty()) {
                    rmdir(localDir.c_str());
                }
            } else {
                localBlocks.assign(blockKeyOffsets.size() - 1, 0);
                for (size_t i = 0; i < localBlockCount; i++) {
                    localBlocks[servedBlocks[i]] = 1;
                }
            }
            MMseqsNode::addPhaseTime(phaseValues, "copying the served blocks to --local-tmp", phaseValueTimer);
            if (localBlockCount == 0 && cluCopySize == 0 && cluSize + servedSize > 0) {
                localTmpProblem = (freeSize == SIZE_MAX) ? std::string("its free space is unknown")
                                  : "it has " + SSTR(freeSize) + " free bytes" + ((maxCopySize < freeSize) ? ", MMSEQS_LOCAL_COPY_BYTES leaves " + SSTR(maxCopySize) : "")
                                    + ", the copies need " + SSTR(cluSize + servedSize);
            } else {
                MMseqsNode::addPhaseValue(phaseValues, "Local tmp", "holds " + SSTR(servedBlockOffsets[localBlockCount]) + " of " + SSTR(servedSize) + " sequence bytes in "
                                          + SSTR(localBlockCount) + " of " + SSTR(servedBlocks.size()) + " blocks"
                                          + (par.filterCluDBFile.empty() ? "" : (cluCopySize > 0 ? " and the cluster part" : ", the cluster part is read from the shared folder")));
            }
        }
        if (localTmpProblem.empty() == false) {
            if (par.localTmpFallback == false) {
                Debug(Debug::ERROR) << "Cannot copy the served blocks to --local-tmp " << par.localTmp << ": " << localTmpProblem << ". Please choose another --local-tmp or set --local-tmp-fallback 1 to read them from the shared folder.\n";
                EXIT(EXIT_FAILURE);
            }
            MMseqsNode::reportUnusableLocalTmp("Cannot copy the served blocks to --local-tmp " + par.localTmp + ": " + localTmpProblem);
            MMseqsNode::addPhaseValue(phaseValues, "Local tmp", "not used, " + localTmpProblem);
        }
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
    DBReader<DBKeyType> seqDbr(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    seqDbr.open(DBReader<DBKeyType>::NOSORT);
    const std::vector<size_t> nodeKeyOffsets = seqDbr.getEvenNodeKeyOffsets(MMseqsNode::count);
    const size_t dbSize = seqDbr.getSize();
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const size_t nodeMemoryLimit = MMseqsNode::getMemoryLimit(par);
    const bool isMasterAligning = MMseqsNode::getFirstAlignNode() == 0;
    const size_t masterMemoryLimit = nodeMemoryLimit / (isMasterAligning ? 2 : 1);
    // the master holds the assigned bits and a log buffer per node in both clustering modes
    const size_t masterFixedSize = (dbSize + 63) / 64 * sizeof(uint64_t) + MMseqsNode::count * CLUSTERED_LOG_BUFFER_SIZE;
    const size_t masterMemorySize = masterFixedSize + ((par.clusteringMode == Parameters::SET_COVER) ? masterMemoryLimit : masterMemoryLimit / 4);
    if (MMseqsNode::isMaster() && masterFixedSize >= nodeMemoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to cluster. Memory limit: " << nodeMemoryLimit << " bytes, the assigned bits of " << dbSize << " sequences need: "
                            << (dbSize + 63) / 64 * sizeof(uint64_t) << " bytes, the clustered logs of " << MMseqsNode::count << " nodes need: "
                            << MMseqsNode::count * CLUSTERED_LOG_BUFFER_SIZE << " bytes.\n"
                            << "Raise RAM size or --split-memory-limit. The assigned bits do not shrink with more nodes.\n";
        EXIT(EXIT_FAILURE);
    }

    // alignment splits of about equal cells, at least 8 MiB of prefilter each
    const size_t maxAlignmentSplitCount = std::max<size_t>(1, std::min<size_t>(4096, totalPrefSize / (8 * 1024 * 1024)));
    std::vector<size_t> alignmentSplitPartitionOffsets(1, 0);
    uint128 cells = 0;
    for (size_t i = 0; i + 1 < partitions.size(); i++) {
        cells += partitions[i].alignmentCells;
        if (alignmentSplitPartitionOffsets.size() < maxAlignmentSplitCount && cells * maxAlignmentSplitCount >= totalAlignmentCells * alignmentSplitPartitionOffsets.size()) {
            alignmentSplitPartitionOffsets.push_back(i + 1);
        }
    }
    alignmentSplitPartitionOffsets.push_back(partitions.size());

    // builds the block table of getKeyBlock
    seqDbr.setBlockCacheSize(0);
    if (localBlocks.empty() == false) {
        seqDbr.setLocalBlocks(localSeqDb, localBlocks);
    }
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

    // a clusthashfast subset has bits; its sixth index column counts the representatives for the e-values
    struct stat sb;
    const bool hasRepresentativeBits = MMseqsNode::statSharedFile(par.db1 + ".bits." + SSTR(MMseqsNode::rank), sb);
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
        // the request files and the served files of all nodes share a directory each, one listing shows the new files
        const std::string exchangeDirs[] = { requests, served };
        for (size_t i = 0; i < 2; i++) {
            if (FileUtil::makeDir(exchangeDirs[i].c_str()) == false && FileUtil::directoryExists(exchangeDirs[i].c_str()) == false) {
                Debug(Debug::ERROR) << "Cannot create " << exchangeDirs[i] << "\n";
                EXIT(EXIT_FAILURE);
            }
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

    // all nodes use the smallest memory limit
    const size_t smallestMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db3 + ".memory", nodeMemoryLimit - std::min(nodeMemoryLimit, assignedMemorySize));
    const size_t paddedDbSize = (dbSize + 63) / 64 * 64;
    const size_t sizeClassCount = (par.clusteringMode == Parameters::SET_COVER) ? 64 - __builtin_clzll(dbSize | 1) : 1;
    // a record's reads fit the smallest memory limit
    const size_t maxRecordSize = std::max<size_t>(1, smallestMemoryLimit / 32);
    // all nodes cut served files at the same size
    const size_t servedFileEntrySize = std::max(BatchReader::MAX_READ_REQUEST_SIZE, smallestMemoryLimit / 512);
    if (hasTargetSplits) {
        // windows of about equal cost; a window holds consecutive entries of one node
        const size_t groupPrefReadSize = alignNodeCount / alignGroupCount * getMaxPrefReadSize(smallestMemoryLimit / PREFILTER_BUFFER_MEMORY_DIVISOR);
        const size_t windowCount = std::max((size_t) 1, (totalPrefSize + groupPrefReadSize - 1) / groupPrefReadSize);
        const uint128 windowCost = std::max((uint128) 1, (totalAlignmentCells + windowCount - 1) / windowCount);
        uint128 costBefore = 0;
        uint128 nextCut = windowCost;
        size_t window = 0;
        for (size_t i = 0; i < windows.size(); i++) {
            const PrefWindow row = windows[i];
            const bool isCostCut = costBefore >= nextCut;
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
        const size_t prefEntryCount = windows.size();
        windows.resize(std::min(windows.size(), window + 1));
        MMseqsNode::addPhaseValue(phaseValues, "Target split entries", SSTR(prefEntryCount) + " in " + SSTR(windows.size()) + " windows");

        // windows in key order to the align group with the least cost per node, so that the groups pass the keys side by side
        std::vector<uint128> groupCosts(alignGroupCount, 0);
        std::vector<size_t> groupWindows(alignGroupCount, 0);
        windowAlignGroups.resize(windows.size());
        for (size_t w = 0; w < windows.size(); w++) {
            size_t windowGroup = 0;
            for (size_t group = 1; group < alignGroupCount; group++) {
                if (groupCosts[group] * (groupNodeOffsets[windowGroup + 1] - groupNodeOffsets[windowGroup]) < groupCosts[windowGroup] * (groupNodeOffsets[group + 1] - groupNodeOffsets[group])) {
                    windowGroup = group;
                }
            }
            windowAlignGroups[w] = windowGroup;
            groupWindows[windowGroup]++;
            groupCosts[windowGroup] += windows[w].alignmentCells;
        }
        if (alignGroup < alignGroupCount) {
            MMseqsNode::addPhaseValue(phaseValues, "Align group", SSTR(alignGroup) + " of " + SSTR(alignGroupCount) + " with " + SSTR(groupNodeOffsets[alignGroup + 1] - groupNodeOffsets[alignGroup]) + " nodes, "
                                      + SSTR(groupWindows[alignGroup]) + " of " + SSTR(windowAlignGroups.size()) + " windows and "
                                      + SSTR(static_cast<size_t>(groupCosts[alignGroup] * 1000 / std::max((uint128) 1, totalAlignmentCells))) + " per mille of the cost");
        }
    }

    RepTargetsReader repTargetsReader(par, seqDbr, partitions, alignmentSplitPartitionOffsets, maxRecordSize, memoryLimit, paddedDbSize, sizeClassCount);
    TargetSplitReader *targetSplitReader = hasTargetSplits ? new TargetSplitReader(par, seqDbr, windows, maxRecordSize, memoryLimit, paddedDbSize, sizeClassCount) : NULL;
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
        const size_t entriesPerSampledKey = std::max<size_t>(256, dbSize / std::max<size_t>(1, smallestMemoryLimit / 64 / sizeof(DBKeyType)));
        std::vector<size_t> cluKeyOffsets = MMseqsNode::getClusterKeyOffsets(par.db3 + ".cluster_key_offsets", par.filterCluDBFile, dbSize);
        cluKeyOffsets[0] = 0;
        // most targets of an aligning node are in its target splits, it preloads their entries
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
        // every node answers for its own cluster part, the master while it clusters
        size_t lookupMemoryLimit = nodeMemoryLimit - std::min(nodeMemoryLimit, (MMseqsNode::isMaster() ? std::min(masterMemoryLimit, masterMemorySize) : 0) + assignedMemorySize);
        memberLookup = new ClusterMemberLookup(par.filterCluDBFile, cluPartDb, cluKeyOffsets, entriesPerSampledKey, lookupMemoryLimit / 16, targetSplitKeyRanges);
        memberExchange = new ClusterMemberExchange(memberRequests, servedMembers, *memberLookup, std::max(1, par.threads / 4), MMseqsNode::getFirstAlignNode());
        memberDbr = new DBReader<DBKeyType>(par.filterSeqDBFile.c_str(), (par.filterSeqDBFile + ".index").c_str(), par.threads,
                                            DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        memberDbr->open(DBReader<DBKeyType>::NOSORT);
        MMseqsNode::addPhaseTime(phaseValues, "sampling the cluster index keys", phaseValueTimer);
    }

    // a single node clusters in the I/O tasks while it aligns, it writes the assigned keys in its steps
    size_t memberCount = 0;
    std::string clusterPhaseValues;
    AcceptedPairClustering *clustering = NULL;
    if (MMseqsNode::isMaster() && isMasterAligning) {
        clustering = new AcceptedPairClustering(par.threads, dbSize, masterMemoryLimit, seqDbr, pageAlignNodes, par.clusteringMode, paddedDbSize, sizeClassCount,
                                                accepted, assignedKeysLog, clusteredLogPrefix, setCoverPrefix, assignedBits, false);
    }
    phaseValueTimer.reset();
    // the align nodes serve the members in their I/O tasks, the master on one thread of --threads or while it waits for the align nodes
    const bool hasMemberServer = memberExchange != NULL && MMseqsNode::count > 1 && MMseqsNode::isMaster() && ioThreadCount > 0;
    std::thread memberServer;
    if (hasMemberServer) {
        memberServer = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(1);
#endif
            memberExchange->serve();
        });
    }
    // a member key, a pending part and an accepted pair take 41 words and 5 flags of the 2 * memberSplitSize in getReservedMemory
    const size_t memberKeyBudget = std::max<size_t>(1, 2 * std::min(memoryLimit, smallestMemoryLimit) / MEMBER_SPLIT_MEMORY_DIVISOR / ((7 + 9 + 25) * sizeof(uint64_t) + 5));
    alignRepTargets(par, seqDbr, repTargetsReader, targetSplitReader, partitions, windows, memoryLimit, masterMemoryLimit, servedFileEntrySize, representativeResidues, paddedDbSize, sizeClassCount,
                    memberLookup, memberExchange, memberKeyBudget, memberDbr, ioThreadCount, clustering, phaseValues);
    if (clustering != NULL) {
        memberCount = clustering->finish(clusterPhaseValues);
        delete clustering;
    }
    // the master of several nodes clusters with the teams of its main thread, the writer runs on a thread of --threads if one is left
    if (MMseqsNode::isMaster() && isMasterAligning == false) {
        memberCount = clusterAcceptedPairs(par.threads, dbSize, masterMemoryLimit, seqDbr, pageAlignNodes, par.clusteringMode, paddedDbSize, sizeClassCount,
                                           accepted, assignedKeysLog, clusteredLogPrefix, setCoverPrefix, assignedBits, hasMemberServer ? NULL : memberExchange,
                                           ioThreadCount - (hasMemberServer ? 1 : 0) > 0, clusterPhaseValues);
    }
    // serve until every aligning node has its members
    if (memberServer.joinable()) {
        memberServer.join();
    }
    // the clusters are written with all threads
#ifdef OPENMP
    omp_set_num_threads(par.threads);
#endif
    if (memberExchange != NULL) {
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the cluster member sizes", memberExchange->sizeWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the cluster members", memberExchange->memberWaitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member exchange", SSTR(memberExchange->servedSize) + " bytes served for "
                                  + SSTR(memberExchange->servedKeyCount) + " keys in " + SSTR(memberExchange->getAnswerCount()) + " answers, "
                                  + SSTR(memberExchange->requestedKeyCount) + " keys requested in " + SSTR(memberExchange->getRequestCount()) + " requests");
        MMseqsNode::addPhaseSeconds(phaseValues, "preloading the cluster members", memberLookup->preloadSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the preloaded cluster members", memberExchange->preloadWaitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Cluster member preload", SSTR(memberLookup->preloadSize) + " of " + SSTR(memberLookup->targetSplitSize) + " bytes of the target splits");
    }
    delete memberExchange;
    delete memberLookup;
    repTargetsReader.close();
    if (targetSplitReader != NULL) {
        targetSplitReader->close();
        if (MMseqsNode::rank >= MMseqsNode::getFirstAlignNode()) {
            MMseqsNode::addPhaseValue(phaseValues, "Target split prefilter bytes", SSTR(targetSplitReader->readSize) + " read of " + SSTR(targetSplitReader->prefSize) + ", largest entry "
                                      + SSTR(targetSplitReader->maxEntrySize) + " bytes");
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
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Number of clusters: " << representativeCount - memberCount << "\n";
        phaseValues += clusterPhaseValues;
        std::vector<unsigned int>().swap(pageAlignNodes);
    }
    // the clustering was the last user of seqDbr
    seqDbr.close();
    for (size_t i = 0; i < localCopies.size(); i++) {
        FileUtil::remove(localCopies[i].c_str());
    }
    if (localCopies.empty() == false) {
        rmdir(localDir.c_str());
    }
    MMseqsNode::addPhaseTime(phaseValues, "waiting for the clustering", phaseValueTimer);
    const size_t nodeFirstKey = nodeKeyOffsets[MMseqsNode::rank];
    const size_t nodeKeys = nodeKeyOffsets[MMseqsNode::rank + 1] - nodeFirstKey;
    const std::string clusterPairs = par.db3 + "_cluster_pairs";
    std::vector<size_t> balancedKeyOffsets;
    MMseqsNode::ClusterPairWriter clusterPairWriter(clusterPairs, std::max<size_t>(64, memoryLimit / 16 / sizeof(ClusterPair)), ClusterPair::compareByRep, nodeKeyOffsets,
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
        // the last progress of the accepted log has its size
        size_t acceptedKey;
        size_t acceptedBytes;
        MMseqsNode::waitForLog(acceptedFile, acceptedKey, acceptedBytes);
        const size_t acceptedWordCount = acceptedBytes / sizeof(uint64_t);
        // a read takes whole words of clustered bits
        const size_t readWordCount = std::min(clusterPairWriter.pairCapacity, std::max<size_t>(1, clusteredBits.size()) * 64) / 64 * 64;
        std::vector<uint64_t> acceptedWords(readWordCount);
        std::vector<std::vector<size_t> > threadRepWordPositions(par.threads);
        std::vector<size_t> repWordPositions;
        uint64_t previousRepWord = 0;
        for (size_t firstWord = 0; firstWord * 64 < acceptedWordCount; firstWord += readWordCount / 64) {
            const size_t wordCount = std::min(readWordCount, acceptedWordCount - firstWord * 64);
            MMseqsNode::readLog(acceptedFile, firstWord * 64 * sizeof(uint64_t), (firstWord * 64 + wordCount) * sizeof(uint64_t), reinterpret_cast<char *>(acceptedWords.data()));
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
    // every node has read the sequences that this node requested and served
    for (size_t i = 0; i < sequenceIndexes.size(); i++) {
        MMseqsNode::removeFile(sequenceIndexes[i]);
    }
    for (size_t i = 0; i < sequenceFiles.size(); i++) {
        if (FileUtil::fileExists(sequenceFiles[i].c_str())) {
            FileUtil::remove(sequenceFiles[i].c_str());
        }
    }
    rmdir(requests.c_str());
    rmdir(served.c_str());
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
    if (assignedBits.empty() == false) {
        delete[] singletonBits;
        singletonBits = readSingletonBits(assignedBits, repFirstKey, repKeyCount);
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
    MMseqsNode::publishDatabase(par.db3);
    Debug(Debug::INFO) << "Time for run Align2Clust: " << timer.lap() << " sec\n";
    return EXIT_SUCCESS;
}
