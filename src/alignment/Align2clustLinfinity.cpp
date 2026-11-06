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
#include <sys/mman.h>
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

// the sorted runs of a spool file of records of recordSize elements; merge takes the records largest first by less, with a read buffer per run
template <typename T>
class SetCoverSpoolRuns {
public:
    SetCoverSpoolRuns() : writtenRecords(0) {}

    void add(size_t records) {
        Run run = { writtenRecords, writtenRecords + records, 0, std::vector<T>() };
        runs.push_back(run);
        writtenRecords += records;
    }
    bool empty() const { return runs.empty(); }
    size_t records() const { return writtenRecords; }

    // reads and removes the file; returns the bytes read
    template <typename Less, typename Prefetch, typename Take>
    size_t merge(const std::string &fileName, size_t recordSize, size_t readBytes, Less less, Prefetch prefetch, Take take) {
        FILE *reader = FileUtil::openFileOrDie(fileName.c_str(), "rb", true);
        const size_t chunkRecords = std::max<size_t>(1, readBytes / (recordSize * sizeof(T)) / runs.size());
        const auto headLess = [&](const Head &first, const Head &second) { return less(first.record, second.record); };
        std::priority_queue<Head, std::vector<Head>, decltype(headLess)> heads(headLess);
        size_t bytesRead = 0;
        for (size_t run = 0; run < runs.size(); run++) {
            if (current(runs[run], fileno(reader), fileName, recordSize, chunkRecords, bytesRead)) {
                heads.push(Head{ runs[run].buffer.data(), run });
            }
        }
        while (heads.empty() == false) {
            const Head head = heads.top();
            heads.pop();
            Run &run = runs[head.run];
            if ((run.bufferPosition + PREFETCH_DISTANCE) * recordSize < run.buffer.size()) {
                prefetch(run.buffer.data() + (run.bufferPosition + PREFETCH_DISTANCE) * recordSize);
            }
            take(head.record);
            run.bufferPosition++;
            if (current(run, fileno(reader), fileName, recordSize, chunkRecords, bytesRead)) {
                heads.push(Head{ run.buffer.data() + run.bufferPosition * recordSize, head.run });
            }
        }
        if (fclose(reader) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        FileUtil::remove(fileName.c_str());
        std::vector<Run>().swap(runs);
        return bytesRead;
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

    static bool current(Run &run, int fd, const std::string &fileName, size_t recordSize, size_t chunkRecords, size_t &bytesRead) {
        if (run.bufferPosition * recordSize == run.buffer.size()) {
            if (run.next == run.end) {
                return false;
            }
            const size_t records = std::min(chunkRecords, run.end - run.next);
            run.buffer.resize(records * recordSize);
            const size_t bytes = run.buffer.size() * sizeof(T);
            FileUtil::preadAll(fd, fileName.c_str(), reinterpret_cast<char *>(run.buffer.data()), bytes, run.next * recordSize * sizeof(T));
            bytesRead += bytes;
            run.next += records;
            run.bufferPosition = 0;
        }
        return true;
    }

    size_t writtenRecords;
    std::vector<Run> runs;
};

class SetCoverPairSpool {
public:
    SetCoverPairSpool(std::vector<ClusterPair> &pairs, const std::string &fileName, size_t maxPairs, size_t memoryLimit, const uint64_t *assigned)
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

    template <typename TakePair>
    void finish(TakePair takePair) {
        if (runs.empty()) {
            SORT_PARALLEL(pairs.begin(), pairs.end(), compareDescending);
            for (size_t i = 0; i < pairs.size(); i++) {
                if (i + PREFETCH_DISTANCE < pairs.size()) {
                    prefetchPair(pairs[i + PREFETCH_DISTANCE]);
                }
                takePair(pairs[i]);
            }
            return;
        }
        flush();
        if (fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        file = NULL;
        runs.merge(fileName, 1, std::min<size_t>(memoryLimit / 64, 4ULL * 1024 * 1024 * 1024),
                   [](const ClusterPair *first, const ClusterPair *second) { return ClusterPair::compareByRep(*first, *second); },
                   [&](const ClusterPair *pair) { prefetchPair(*pair); },
                   [&](const ClusterPair *pair) { takePair(*pair); });
    }

    size_t writtenBytes() const { return runs.records() * sizeof(ClusterPair); }

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
    SetCoverSpoolRuns<ClusterPair> runs;
};

struct SetCoverSizeClasses {
    size_t liveAllCandidates = 0;
    size_t peakAllCandidates = 0;
    size_t liveAllMembers = 0;
    size_t peakAllMembers = 0;

    void add(size_t count) {
        liveAllCandidates++;
        liveAllMembers += count;
        peakAllCandidates = std::max(peakAllCandidates, liveAllCandidates);
        peakAllMembers = std::max(peakAllMembers, liveAllMembers);
    }
    void remove(size_t count) {
        liveAllCandidates--;
        liveAllMembers -= count;
    }
};

class SetCoverSmallSetSpool {
public:
    static const size_t MAX_SIZE = 64;

    SetCoverSmallSetSpool(const std::string &prefix, size_t memoryLimit, const uint64_t *assigned)
        : prefix(prefix), memoryLimit(memoryLimit), assigned(assigned), buckets(MAX_SIZE + 1), writtenBytes(0), readBytes(0),
          liveDiskBytes(0), peakDiskBytes(0) {}

    void append(size_t count, const DBKeyType *members) {
        Bucket &bucket = buckets[count];
        const size_t bucketBytes = std::max<size_t>(count * sizeof(DBKeyType), memoryLimit / 64 / (MAX_SIZE - 2));
        if (bucket.words.size() / count >= std::max<size_t>(1, bucketBytes / sizeof(DBKeyType) / count)) {
            flush(count);
        }
        bucket.words.insert(bucket.words.end(), members, members + count);
    }

    template <typename TakeSet>
    void drain(size_t maxSetSize, TakeSet takeSet) {
        for (size_t count = MAX_SIZE; count > 2 && count > maxSetSize; count--) {
            drainSize(count, takeSet);
        }
    }

    size_t totalWrittenBytes() const { return writtenBytes; }
    size_t totalReadBytes() const { return readBytes; }
    size_t peakBytesOnDisk() const { return peakDiskBytes; }
    size_t mergeReadBudgetBytes() const { return std::min<size_t>(memoryLimit / 64, 4ULL * 1024 * 1024 * 1024); }

private:
    struct Bucket {
        std::vector<DBKeyType> words;
        SetCoverSpoolRuns<DBKeyType> runs;
        FILE *writer;
        bool drained;
        Bucket() : writer(NULL), drained(false) {}
    };

    std::string fileName(size_t count) const {
        return prefix + "." + SSTR(count);
    }
    void flush(size_t count) {
        Bucket &bucket = buckets[count];
        if (bucket.words.empty()) {
            return;
        }
        const size_t recordCount = bucket.words.size() / count;
        std::vector<std::pair<DBKeyType, size_t> > order(recordCount);
        for (size_t i = 0; i < recordCount; i++) {
            order[i] = std::make_pair(bucket.words[i * count], i);
        }
        SORT_PARALLEL(order.begin(), order.end(), std::greater<std::pair<DBKeyType, size_t> >());
        const std::string path = fileName(count);
        if (bucket.writer == NULL) {
            bucket.writer = FileUtil::openAndDelete(path.c_str(), "wb");
        }
        const size_t writeWords = std::max<size_t>(count, (4ULL * 1024 * 1024 / sizeof(DBKeyType) / count) * count);
        std::vector<DBKeyType> output;
        output.reserve(writeWords);
        for (size_t i = 0; i < recordCount; i++) {
            const DBKeyType *record = bucket.words.data() + order[i].second * count;
            output.insert(output.end(), record, record + count);
            if (output.size() >= writeWords || i + 1 == recordCount) {
                if (fwrite(output.data(), sizeof(DBKeyType), output.size(), bucket.writer) != output.size()) {
                    Debug(Debug::ERROR) << "Cannot write " << path << "\n";
                    EXIT(EXIT_FAILURE);
                }
                output.clear();
            }
        }
        bucket.runs.add(recordCount);
        const size_t bytes = recordCount * count * sizeof(DBKeyType);
        writtenBytes += bytes;
        liveDiskBytes += bytes;
        peakDiskBytes = std::max(peakDiskBytes, liveDiskBytes);
        bucket.words.clear();
    }
    template <typename TakeSet>
    void drainSize(size_t count, TakeSet takeSet) {
        Bucket &bucket = buckets[count];
        if (bucket.drained) {
            return;
        }
        flush(count);
        bucket.drained = true;
        const std::string path = fileName(count);
        if (bucket.writer == NULL) {
            unlink(path.c_str());
            return;
        }
        if (fclose(bucket.writer) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        bucket.writer = NULL;
        readBytes += bucket.runs.merge(path, count, mergeReadBudgetBytes(),
                                       [](const DBKeyType *first, const DBKeyType *second) { return first[0] < second[0]; },
                                       [&](const DBKeyType *ahead) {
                                           for (size_t member = 0; member < count; member++) {
                                               __builtin_prefetch(&assigned[ahead[member] / 64]);
                                           }
                                       },
                                       [&](const DBKeyType *members) { takeSet(count, members); });
        liveDiskBytes -= bucket.runs.records() * count * sizeof(DBKeyType);
        std::vector<DBKeyType>().swap(bucket.words);
    }

    const std::string prefix;
    const size_t memoryLimit;
    const uint64_t *assigned;
    std::vector<Bucket> buckets;
    size_t writtenBytes;
    size_t readBytes;
    size_t liveDiskBytes;
    size_t peakDiskBytes;
};

class SetCoverLargeSetSpool {
public:
    SetCoverLargeSetSpool(const std::string &prefix, size_t memoryLimit)
        : prefix(prefix), memoryBudget(std::max<size_t>(65 * sizeof(DBKeyType), std::min<size_t>(memoryLimit / 64, 4ULL * 1024 * 1024 * 1024))),
          nextFile(0), writtenBytes(0), readBytes(0), peakDiskBytes(0), liveDiskBytes(0), peakMemoryBytes(0), peakMemberBytes(0),
          runBufferBytes(0), runPathBytes(0) {}

    template <typename Reps, typename Counts, typename IsFree, typename Word>
    void appendStep(const Reps &reps, const Counts &offsets, size_t paddedDbSize, IsFree isFree, Word word, SetCoverSizeClasses &classes) {
        std::vector<size_t> order;
        for (size_t rep = 0; rep < reps.size(); rep++) {
            if (offsets[rep + 1] - offsets[rep] > SetCoverSmallSetSpool::MAX_SIZE) {
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
        const size_t run = newRun(file);
        for (size_t rep : order) {
            const size_t count = offsets[rep + 1] - offsets[rep];
            writeWord(file, count);
            writeWord(file, reps[rep].repKey % paddedDbSize);
            for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                if (isFree(i)) {
                    writeWord(file, word(i));
                }
            }
            classes.add(count);
        }
        finishRun(run, file);
    }

    template <typename IsAssigned, typename TakeMember>
    void drain(size_t maxSetSize, IsAssigned isAssigned, TakeMember takeMember, SetCoverPairSpool &pairSpool,
               SetCoverSmallSetSpool &smallSpool, SetCoverSizeClasses &classes) {
        while (true) {
            const bool fromPending = pending.empty() == false && (heads.empty() || greater(pending.front().memberCount, pending.front().representativeId,
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
            classes.remove(count);
            // the survivors in order, all of them unless more than the memory budget
            survivors.clear();
            size_t validCount = 0;
            if (isAssigned(repKey) == false) {
                for (size_t member = 0; member < count; member++) {
                    const DBKeyType key = fromPending ? pendingMembers[start + member] : readWord(run, start + member);
                    if (isAssigned(key) == false) {
                        if (validCount < memoryBudget / sizeof(DBKeyType)) {
                            survivors.push_back(key);
                        }
                        validCount++;
                    }
                }
            }
            if (validCount > 1) {
                if (validCount == 2) {
                    pairSpool.push(ClusterPair{repKey, survivors[1]});
                } else if (validCount <= SetCoverSmallSetSpool::MAX_SIZE) {
                    smallSpool.append(validCount, survivors.data());
                    classes.add(validCount);
                } else if (validCount < count) {
                    if (validCount > memoryBudget / sizeof(DBKeyType)) {
                        FILE *file = NULL;
                        const size_t nextRun = newRun(file);
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
                        pendingMembers.insert(pendingMembers.end(), survivors.begin(), survivors.end());
                        pending.push_back(SetCoverCandidate{repKey, validCount, offset});
                        std::push_heap(pending.begin(), pending.end(), SetCoverComparator());
                        peakMemoryBytes = std::max(peakMemoryBytes, memoryBytes());
                        peakMemberBytes = std::max(peakMemberBytes, pendingMembers.capacity() * sizeof(DBKeyType));
                    }
                    classes.add(validCount);
                } else if (survivors.size() == count) {
                    for (size_t member = 0; member < count; member++) {
                        takeMember(repKey, survivors[member]);
                    }
                } else {
                    for (size_t member = 0; member < count; member++) {
                        takeMember(repKey, fromPending ? pendingMembers[start + member] : readWord(run, start + member));
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
                    liveDiskBytes -= source.words * sizeof(DBKeyType);
                    if (source.reader != NULL) {
                        if (fclose(source.reader) != 0) {
                            Debug(Debug::ERROR) << "Cannot close " << source.path << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        openRuns.erase(source.openPosition);
                        source.reader = NULL;
                    }
                    FileUtil::remove(source.path.c_str());
                    runBufferBytes -= source.buffer.capacity() * sizeof(DBKeyType);
                    runPathBytes -= source.path.capacity();
                    source = Run();
                    freeRuns.push_back(run);
                } else {
                    pushHead(run);
                }
            }
        }
        std::vector<DBKeyType>().swap(survivors);
    }

    size_t memoryBytes() const {
        return pending.capacity() * sizeof(SetCoverCandidate) + pendingMembers.capacity() * sizeof(DBKeyType)
            + runs.capacity() * sizeof(Run) + freeRuns.capacity() * sizeof(size_t) + heads.size() * sizeof(Head)
            + survivors.capacity() * sizeof(DBKeyType) + writeBuffer.capacity() * sizeof(DBKeyType) + runBufferBytes + runPathBytes + openRuns.size() * sizeof(size_t);
    }
    size_t memberBytes() const { return peakMemberBytes; }
    size_t peakMemory() const { return peakMemoryBytes; }
    size_t peakDisk() const { return peakDiskBytes; }
    size_t bytesWritten() const { return writtenBytes; }
    size_t bytesRead() const { return readBytes; }

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
    static bool greater(size_t firstCount, DBKeyType firstRep, size_t secondCount, DBKeyType secondRep) {
        return firstCount == secondCount ? firstRep > secondRep : firstCount > secondCount;
    }
    size_t newRun(FILE *&file) {
        const size_t index = freeRuns.empty() ? runs.size() : freeRuns.back();
        if (freeRuns.empty()) {
            runs.emplace_back();
        } else {
            freeRuns.pop_back();
        }
        runs[index].path = prefix + "." + SSTR(nextFile++);
        runPathBytes += runs[index].path.capacity();
        file = FileUtil::openAndDelete(runs[index].path.c_str(), "wb");
        return index;
    }
    void writeWord(FILE *file, DBKeyType word) {
        writeBuffer.push_back(word);
        if (writeBuffer.size() == 4 * 1024 * 1024 / sizeof(DBKeyType)) {
            flushWrites(file);
        }
        writtenBytes += sizeof(word);
    }
    void flushWrites(FILE *file) {
        if (fwrite(writeBuffer.data(), sizeof(DBKeyType), writeBuffer.size(), file) != writeBuffer.size()) {
            Debug(Debug::ERROR) << "Cannot write large set spool " << prefix << "\n";
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
        runs[run].words = writtenBytes / sizeof(DBKeyType) - totalRunWords;
        totalRunWords = writtenBytes / sizeof(DBKeyType);
        liveDiskBytes += runs[run].words * sizeof(DBKeyType);
        peakDiskBytes = std::max(peakDiskBytes, liveDiskBytes);
        pushHead(run);
        peakMemoryBytes = std::max(peakMemoryBytes, memoryBytes());
    }
    DBKeyType readWord(size_t run, size_t offset) {
        Run &source = runs[run];
        if (source.buffer.empty() || offset < source.bufferStart || offset >= source.bufferStart + source.buffer.size()) {
            const size_t bufferWords = 8192;
            source.bufferStart = (offset / bufferWords) * bufferWords;
            const size_t oldCapacity = source.buffer.capacity();
            source.buffer.resize(std::min(bufferWords, source.words - source.bufferStart));
            runBufferBytes += (source.buffer.capacity() - oldCapacity) * sizeof(DBKeyType);
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
            readBytes += source.buffer.size() * sizeof(DBKeyType);
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
            return greater(first.memberCount, first.representativeId, second.memberCount, second.representativeId);
        });
        FILE *file = NULL;
        const size_t run = newRun(file);
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
    size_t writtenBytes;
    size_t readBytes;
    size_t peakDiskBytes;
    size_t liveDiskBytes;
    size_t peakMemoryBytes;
    size_t peakMemberBytes;
    size_t runBufferBytes;
    size_t runPathBytes;
    size_t totalRunWords = 0;
    std::vector<Run> runs;
    std::vector<size_t> freeRuns;
    std::list<size_t> openRuns;
    std::priority_queue<Head, std::vector<Head>, HeadLess> heads;
    std::vector<SetCoverCandidate> pending;
    std::vector<DBKeyType> pendingMembers;
    std::vector<DBKeyType> survivors;
    std::vector<DBKeyType> writeBuffer;
};

// whether the forward and backward extension of target i of alignTargetsSwipe, joined at their seed cell, meet the clustering criteria
static bool swipeExtensionsPass(const Parameters &par, SwipeTargets &swipe, size_t i) {
    const s_align &forward = swipe.aligns[i];
    const s_align &backward = swipe.backAligns[i];
    // the seed cell is in both extensions
    const int rawScore = static_cast<int>(forward.score1) + static_cast<int>(backward.score1)
                         - swipe.extender.cellScore(swipe.queryIds[i], swipe.seedQueries[i], swipe.targets[i][swipe.seedTargets[i]]);
    std::string &gappedBacktrace = swipe.backBacktraces[i];
    gappedBacktrace.pop_back();
    gappedBacktrace += swipe.backtraces[i];
    s_align joined = forward;
    joined.qStartPos1 = backward.qStartPos1;
    joined.dbStartPos1 = backward.dbStartPos1;
    const char *querySequence = swipe.repSequences[swipe.queryIds[i]];
    const char *targetSequence = swipe.targetSequences[i];
    joined.identicalAACnt = 0;
    for (size_t b = 0, q = joined.qStartPos1, t = joined.dbStartPos1; b < gappedBacktrace.size(); b++) {
        const char op = gappedBacktrace[b];
        joined.identicalAACnt += (op == 'M' && querySequence[q] == targetSequence[t]);
        q += (op != 'D');
        t += (op != 'I');
    }
    Matcher::result_t result = swipeResult(par, swipe.evaluer, swipe.targetKeys[i], rawScore, joined, swipe.repLengths[swipe.queryIds[i]], swipe.lengths[i],
                                           gappedBacktrace);
    return Alignment::checkCriteria(result, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr);
}

// the SWIPE extensions of the queued targets (--align2clust-aligner-mode 1), skipping the ones assigned since
template <typename Assigned>
static void alignTargetsSwipe(const Parameters &par, SwipeTargets &swipe, const Assigned &isAssigned, std::vector<ClusterPair> &pairs) {
    const int32_t xDrop = (MIN_SIZE * par.gapExtend.values.aminoacid() + par.gapOpen.values.aminoacid());
    // twice one past the longest gap within the X-drop, which is MIN_SIZE + 1 long
    const int bandSide = 2 * (MIN_SIZE + 2);
    size_t count = 0;
    for (size_t i = 0; i < swipe.targetKeys.size(); i++) {
        if (isAssigned(swipe.repKeys[swipe.queryIds[i]]) || isAssigned(swipe.targetKeys[i])) {
            continue;
        }
        swipe.queryIds[count] = swipe.queryIds[i];
        swipe.offsets[count] = swipe.offsets[i];
        swipe.lengths[count] = swipe.lengths[i];
        swipe.seedQueries[count] = swipe.seedQueries[i];
        swipe.seedTargets[count] = swipe.seedTargets[i];
        swipe.targetKeys[count] = swipe.targetKeys[i];
        swipe.targetSequences[count] = swipe.targetSequences[i];
        count++;
    }
    if (count == 0) {
        return;
    }
    swipe.targets.resize(count);
    swipe.aligns.resize(count);
    swipe.backtraces.resize(count);
    for (size_t i = 0; i < count; i++) {
        swipe.targets[i] = swipe.sequences.data() + swipe.offsets[i];
    }
    swipe.extender.extendBanded(count, swipe.queryIds.data(), swipe.targets.data(), swipe.lengths.data(), swipe.seedQueries.data(), swipe.seedTargets.data(), false, xDrop,
                               bandSide, swipe.aligns.data(), swipe.backtraces.data());
    // the targets whose forward extension covers enough move to the front
    size_t passed = 0;
    for (size_t i = 0; i < count; i++) {
        const s_align &forward = swipe.aligns[i];
        if (forward.score1 == 0 || Util::hasCoverage(par.covThr, par.covMode, SmithWaterman::computeCov(0, forward.qEndPos1, swipe.repLengths[swipe.queryIds[i]]),
                                                     SmithWaterman::computeCov(0, forward.dbEndPos1, swipe.lengths[i])) == false) {
            continue;
        }
        swipe.queryIds[passed] = swipe.queryIds[i];
        swipe.targets[passed] = swipe.targets[i];
        swipe.lengths[passed] = swipe.lengths[i];
        swipe.seedQueries[passed] = swipe.seedQueries[i];
        swipe.seedTargets[passed] = swipe.seedTargets[i];
        swipe.targetKeys[passed] = swipe.targetKeys[i];
        swipe.targetSequences[passed] = swipe.targetSequences[i];
        swipe.aligns[passed] = swipe.aligns[i];
        swipe.backtraces[passed].swap(swipe.backtraces[i]);
        passed++;
    }
    swipe.targetPairs += count;
    count = passed;
    swipe.backAligns.resize(count);
    swipe.backBacktraces.resize(count);
    swipe.extender.extendBanded(count, swipe.queryIds.data(), swipe.targets.data(), swipe.lengths.data(), swipe.seedQueries.data(), swipe.seedTargets.data(), true, xDrop,
                               bandSide, swipe.backAligns.data(), swipe.backBacktraces.data());
    for (size_t i = 0; i < count; i++) {
        if (swipe.backAligns[i].score1 > 0 && swipeExtensionsPass(par, swipe, i)) {
            const ClusterPair pair = { swipe.repKeys[swipe.queryIds[i]], swipe.targetKeys[i] };
            pairs.push_back(pair);
        }
    }
}

// the member gates of a batch of reps in shared rounds: the gapped members of all reps share the extension lanes, the ones no extension accepted go through alignMembersSwipe per rep
template <typename MapRep>
static void alignMemberGatesSwipe(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, Sequence &query, Sequence &element, DBReader<DBKeyType> *cluSeqDbr,
                                  unsigned int swMode, unsigned int threadIdx, ClusterMembers &members, const std::vector<size_t> &repGateOffsets,
                                  const std::vector<DBKeyType> &repKeys, const MapRep &mapRep) {
    SwipeTargets &targets = *members.targets;
    SwipeBatch &swipe = *members.swipe;
    bool matcherInitialized = false;
    for (size_t batchSize = 1; ; batchSize *= 2) {
        bool open = false;
        members.gappedGates.clear();
        members.gappedMembers.clear();
        for (size_t rep = 0; rep < repKeys.size(); rep++) {
            bool repOpen = (batchSize == 1);
            for (size_t g = repGateOffsets[rep]; g < repGateOffsets[rep + 1]; g++) {
                repOpen = repOpen || (members.gates[g].live && members.gates[g].next < members.gates[g].memberKeys.size());
            }
            if (repOpen == false) {
                continue;
            }
            mapRep(query, blockAligner, repKeys[rep]);
            if (batchSize == 1) {
                prepareMemberGates(par, query, cluSeqDbr, false, members, repGateOffsets[rep], repGateOffsets[rep + 1]);
            }
            targets.startRep(repKeys[rep]);
            open = alignMemberRound(par, blockAligner, matcher, matcherInitialized, query, element, cluSeqDbr, swMode, threadIdx, false, NULL, members,
                                    repGateOffsets[rep], repGateOffsets[rep + 1], batchSize) || open;
        }
        if (open == false) {
            break;
        }
        members.extendedPairs.clear();
        alignTargetsSwipe(par, targets, [](DBKeyType) { return false; }, members.extendedPairs);
        targets.clearTargets();
        for (size_t i = 0; i < members.extendedPairs.size(); i++) {
            members.gappedMembers[members.extendedPairs[i].memberKey] = SIZE_MAX;
        }
        size_t gapped = 0;
        for (size_t rep = 0; rep < repKeys.size(); rep++) {
            swipe.clear();
            members.poolGates.clear();
            for (; gapped < members.gappedGates.size() && members.gappedGates[gapped] < repGateOffsets[rep + 1]; gapped++) {
                const size_t i = members.gappedMembers[gapped];
                if (i == SIZE_MAX) {
                    continue;
                }
                const size_t g = members.gappedGates[gapped];
                const DBKeyType elementKey = members.gates[g].memberKeys[i];
                const size_t elementId = cluSeqDbr->getId(elementKey);
                element.mapSequence(elementId, elementKey, cluSeqDbr->getData(elementId, threadIdx), cluSeqDbr->getSeqLen(elementId));
                swipe.add(element);
                swipe.gapped.push_back(i);
                members.poolGates.push_back(g);
            }
            if (swipe.gapped.empty()) {
                continue;
            }
            mapRep(query, blockAligner, repKeys[rep]);
            swipe.startRep();
            matcherInitialized = false;
            alignMembersSwipe(par, matcher, matcherInitialized, members, query, element, cluSeqDbr, swMode, threadIdx, false, NULL);
        }
    }
}

__extension__ typedef unsigned __int128 uint128;

// --createdb-mode 3: a row of <pref>.partitions, the entries from firstKey on in <pref>.<node>
struct PrefPartition {
    size_t firstKey;
    size_t prefOffset;
    size_t prefBytes;
    size_t cost;
    size_t node;
};

// --createdb-mode 3: a rep with the first of its targets next to each other and their count
struct RepTargetsRecord {
    ClusterPair repFirstTarget;
    unsigned int targetCount;
    size_t targetOffset;
    size_t readBytes;

    static bool compareByRep(const RepTargetsRecord &first, const RepTargetsRecord &second) {
        return ClusterPair::compareByRep(first.repFirstTarget, second.repFirstTarget);
    }
};

// group ranges per aligning node; at 1 T a range is about a minute of one node's work, so the nodes align next to each other
static const size_t GROUP_RANGES_PER_NODE = 4096;
// the group ranges read 8 MiB of the prefilter on average at least, which sets their count below 32 GiB of prefilter per aligning node
static const size_t MIN_GROUP_RANGE_PREF_BYTES = 8 * 1024 * 1024;

// the master only clusters while other nodes align
static size_t getFirstAlignNode() {
    return (MMseqsNode::count > 1) ? 1 : 0;
}

// the blocks of the sequences go to the aligning nodes in turn, which serve their keys to the others
static size_t getBlockNode(size_t block) {
    return getFirstAlignNode() + block % (MMseqsNode::count - getFirstAlignNode());
}

static size_t getKeyNode(const DBReader<DBKeyType> &seqDbr, size_t key) {
    return getBlockNode(seqDbr.getKeyBlock(key));
}

// --createdb-mode 3: the records [begin, end) of a chunk aligned at once, with their keys and entries: the keys to fetch by home, then the keys kept from the window before as the last home
struct RepWindow {
    RepWindow() : chunk(0), begin(0), end(0), data(NULL), allocatedBytes(0) {}
    size_t chunk;
    size_t begin;
    size_t end;
    std::vector<DBKeyType> keys;
    std::vector<size_t> homeKeyOffsets;
    char *data;
    size_t allocatedBytes;
    std::vector<size_t> entryOffsets;
};

static const size_t PUBLISH_BYTES = 4 * 1024 * 1024;
static const time_t PUBLISH_SECONDS = 1;
static const time_t READ_SECONDS = 1;
// the nodes publish and read the changes right after the turn of a second, before the master step that answers them; a batch start reads them this often to see it soon
static const double BATCH_READ_SECONDS = PUBLISH_SECONDS / 4.0;
// a chunk claims group ranges of about this much work of its node, unless one range is more
static const double CLAIM_SECONDS = 8.0 * PUBLISH_SECONDS;
// batches per thread of the records aligned at once, taken in rep order
static const size_t ALIGN_BATCHES_PER_THREAD = 64;
// the SWIPE lanes that ALIGN_BATCHES_PER_THREAD is for, more lanes take larger batches with as many extensions per lane
static const size_t ALIGN_BATCH_LANES = 16;
static const int CLUSTER_THREADS = 4;
// an accepted log: the order key of every rep with this bit, then its accepted targets
static const uint64_t ACCEPTED_REP_BIT = 1ULL << 63;
// a step of the assigned changes log: the clustered key, the clustered accepted log bytes of every node, where the keys of every node start, then their end
static size_t getChangeHeaderWords() {
    return 2 * MMseqsNode::count + 2;
}

// each of these takes the memory limit divided by its divisor
static const size_t MASTER_MEMORY_DIVISOR = 2;
static const size_t CHUNK_MEMORY_DIVISOR = 32;
static const size_t WINDOW_MEMORY_DIVISOR = 8;
// the records are the same whatever window a node takes, as the reads of a record fit the smallest memory limit divided by this
static const size_t RECORD_MEMORY_DIVISOR = 32;
static const size_t MEMBER_PART_MEMORY_DIVISOR = 16;
// the members of a batch of reps of alignMemberGatesSwipe
static const size_t MEMBER_BATCH_SIZE = 4096;
static const size_t STEP_MEMORY_DIVISOR = 4;
static const size_t STEP_PAIRS_MEMORY_DIVISOR = 16;
static const size_t LOG_PART_MEMORY_DIVISOR = 64;
// the stdio buffer of every taken log of the master
static const size_t TAKEN_BUFFER_BYTES = 16 * 1024 * 1024;
static const size_t CLUSTER_PAIRS_MEMORY_DIVISOR = 16;
static const size_t ANCHOR_MEMORY_DIVISOR = 64;
// --filter-cludb-file: the (key, offset) cells of a 4 KiB page from one anchor to the next at least, and the bytes of an entry read at once
static const size_t ANCHOR_ENTRIES = 256;
static const size_t MEMBER_READ_BYTES = 1024 * 1024;
// the bytes of a window key besides its entry: its copies, its home and its entry offset
static const size_t WINDOW_KEY_BYTES = 4 * sizeof(DBKeyType) + sizeof(unsigned int) + sizeof(size_t);
// the keys of a page of assigned bits, the same on every node whatever its memory page size
static const size_t ASSIGNED_PAGE_KEYS = 32768;
// the first window is small so aligning starts soon; every next one doubles
static const size_t FIRST_WINDOW_BYTES = 64 * 1024 * 1024;
// a served flush is cut once the entries of its slices reach the memory limit divided by this, at least MMseqsNode::EXCHANGE_WRITE_BYTES
static const size_t FLUSH_MEMORY_DIVISOR = 512;
// a slice of a request: a bit per key, set when it is assigned and not served, then a slot per served entry of 5 bits per byte, or PACKED_RAW_CODE and its bytes after the slots
static const char PACKED_ENTRY_BYTES[] = "\0\nABCDEFGHIJKLMNOPQRSTUVWXYZ*-";
static const unsigned char PACKED_RAW_CODE = 31;

struct PackedEntryCodes {
    PackedEntryCodes() {
        memset(codes, PACKED_RAW_CODE, sizeof(codes));
        for (size_t i = 0; i < sizeof(PACKED_ENTRY_BYTES) - 1; i++) {
            codes[static_cast<unsigned char>(PACKED_ENTRY_BYTES[i])] = static_cast<unsigned char>(i);
        }
    }
    unsigned char codes[256];
};
static const PackedEntryCodes packedEntryCodes;

static size_t getPackedEntryBytes(size_t entryLen) {
    return (5 * entryLen + 7) / 8;
}

// every 8 bytes of entry become 5 bytes of slot
static bool packEntry(const char *entry, size_t entryLen, char *slot) {
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
        for (size_t byte = 0; byte < getPackedEntryBytes(groupLen); byte++) {
            slot[i / 8 * 5 + byte] = static_cast<char>(bits >> (8 * byte));
        }
    }
    return true;
}

static void unpackEntry(const char *slot, size_t entryLen, char *entry) {
    for (size_t i = 0; i < entryLen; i += 8) {
        const size_t groupLen = std::min<size_t>(8, entryLen - i);
        uint64_t bits = 0;
        for (size_t byte = 0; byte < getPackedEntryBytes(groupLen); byte++) {
            bits |= static_cast<uint64_t>(static_cast<unsigned char>(slot[i / 8 * 5 + byte])) << (8 * byte);
        }
        for (size_t j = 0; j < groupLen; j++) {
            entry[i + j] = PACKED_ENTRY_BYTES[(bits >> (5 * j)) & 31];
        }
    }
}

// writes the entries at entryOffsets of data from the slice of keyCount keys: zeros for the assigned keys, else their packed slots, then their raw entries
static void unpackEntries(const char *slice, size_t keyCount, const size_t *entryOffsets, char *data, int threads) {
    const char *slots = slice + (keyCount + 7) / 8;
    std::vector<size_t> slotOffsets(keyCount + 1, 0);
    for (size_t i = 0; i < keyCount; i++) {
        const bool isAssigned = (static_cast<unsigned char>(slice[i / 8]) >> (i % 8)) & 1;
        slotOffsets[i + 1] = slotOffsets[i] + (isAssigned ? 0 : getPackedEntryBytes(entryOffsets[i + 1] - entryOffsets[i]));
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
            if (slotOffsets[i + 1] == slotOffsets[i]) {
                memset(data + entryOffsets[i], 0, entryOffsets[i + 1] - entryOffsets[i]);
            } else if ((static_cast<unsigned char>(slots[slotOffsets[i]]) & 31) == PACKED_RAW_CODE) {
                threadRawEntries[thread_idx].push_back(i);
            } else {
                unpackEntry(slots + slotOffsets[i], entryOffsets[i + 1] - entryOffsets[i], data + entryOffsets[i]);
            }
        }
    }
    const char *rawEntry = slots + slotOffsets[keyCount];
    for (int thread = 0; thread < threads; thread++) {
        for (size_t j = 0; j < threadRawEntries[thread].size(); j++) {
            const size_t i = threadRawEntries[thread][j];
            memcpy(data + entryOffsets[i], rawEntry, entryOffsets[i + 1] - entryOffsets[i]);
            rawEntry += entryOffsets[i + 1] - entryOffsets[i];
        }
    }
}

// a pass of a service thread polls this many other nodes in turn, so that it stays short on many nodes
static const size_t SERVICE_POLL_NODES = 8;
// the service threads of all nodes open about this many indexes a second, as a pass opens two of every node that it polls; a pass waits for 1.28 s at most
static const size_t SERVICE_OPENS_PER_SECOND = 10000;
static const size_t MAX_SERVICE_PASS_MICROSECONDS = 1280000;

// offsets[0] is 0, offsets[1..count] become their sums from the front: every thread sums its part, then adds the parts before it
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

// a step of the master clusters at most this many words of the logs, each with up to four slots of keyReps, unless its first rep spans more
static size_t getStepWords(size_t masterMemoryLimit) {
    return std::max<size_t>(1, masterMemoryLimit / STEP_PAIRS_MEMORY_DIVISOR / (sizeof(uint64_t) + 4 * sizeof(std::pair<DBKeyType, size_t>)));
}

// --filter-cludb-file: the members of the entries of the parts <clu>.<node>, found from the key of every anchorEntries-th entry of every part
struct ClusterMemberLookup {
    struct Entry {
        size_t node;
        size_t offset;
        // where the lines of the entry end, before its '\0'
        size_t end;
        DBKeyType key;
    };
    struct Cursor {
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

    ClusterMemberLookup(const std::string &clu, const std::string &anchorKeys, size_t anchorEntries)
        : anchorEntries(anchorEntries), parts(MMseqsNode::count), partKeyOffsets(MMseqsNode::count + 1, SIZE_MAX), nodeAnchorOffsets(MMseqsNode::count + 1, 0),
          anchors(anchorKeys.size() / sizeof(DBKeyType)) {
        memcpy(anchors.data(), anchorKeys.data(), anchorKeys.size());
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            parts[node] = new DBReader<DBKeyType>((clu + "." + SSTR(node)).c_str(), (clu + ".index." + SSTR(node)).c_str(), 1,
                                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
            parts[node]->open(DBReader<DBKeyType>::NOSORT);
            nodeAnchorOffsets[node + 1] = nodeAnchorOffsets[node] + (parts[node]->getSize() + anchorEntries - 1) / anchorEntries;
        }
        // the first anchor of a part is its first entry key; a part without entries starts where the next part starts
        for (size_t node = MMseqsNode::count; node > 0; node--) {
            partKeyOffsets[node - 1] = (nodeAnchorOffsets[node - 1] < nodeAnchorOffsets[node]) ? static_cast<size_t>(anchors[nodeAnchorOffsets[node - 1]]) : partKeyOffsets[node];
        }
        partKeyOffsets[0] = 0;
    }

    ~ClusterMemberLookup() {
        for (size_t node = 0; node < parts.size(); node++) {
            parts[node]->close();
            delete parts[node];
        }
    }

    Entry find(DBKeyType key) const {
        const size_t node = std::upper_bound(partKeyOffsets.begin(), partKeyOffsets.end(), key) - partKeyOffsets.begin() - 1;
        Entry entry = { node, 0, 0, key };
        const DBKeyType *begin = anchors.data() + nodeAnchorOffsets[node];
        const size_t anchor = std::lower_bound(begin, anchors.data() + nodeAnchorOffsets[node + 1], key) - begin;
        // the entry of key, if any, is one from the anchor before key to the anchor at key or after it
        const size_t found = parts[node]->findEntryByKey(key, (anchor > 0) ? (anchor - 1) * anchorEntries : 0, std::min(anchor * anchorEntries + 1, parts[node]->getSize()));
        if (found < parts[node]->getSize()) {
            const DBReader<DBKeyType>::Index index = parts[node]->getIndexEntry(found);
            if (index.id == key) {
                entry.offset = index.offset;
                entry.end = index.offset + index.length - 1;
            }
        }
        return entry;
    }

    // the next member of the entry of cursor besides its key, false once none is left
    bool next(Cursor &cursor, DBKeyType &memberKey) const {
        Entry &entry = cursor.entry;
        while (entry.offset < entry.end) {
            const char *lineEnd = (entry.offset < cursor.bufferEnd)
                                  ? static_cast<const char *>(memchr(cursor.buffer.data() + (entry.offset - cursor.bufferBegin), '\n', cursor.bufferEnd - entry.offset)) : NULL;
            if (lineEnd == NULL) {
                const size_t bytes = std::min(entry.end - entry.offset, MEMBER_READ_BYTES);
                cursor.buffer.resize(bytes);
                FileUtil::preadAll(parts[entry.node]->getBufferedFd(), parts[entry.node]->getDataFileName(), cursor.buffer.data(), bytes, entry.offset);
                cursor.bufferBegin = entry.offset;
                cursor.bufferEnd = entry.offset + bytes;
                continue;
            }
            const char *line = cursor.buffer.data() + (entry.offset - cursor.bufferBegin);
            const DBKeyType key = Util::fast_atoi<DBKeyType>(line);
            entry.offset += lineEnd - line + 1;
            if (key != entry.key) {
                memberKey = key;
                return true;
            }
        }
        return false;
    }

    size_t memoryBytes() const {
        return sizeof(*this) + partKeyOffsets.capacity() * sizeof(size_t) + parts.size() * sizeof(DBReader<DBKeyType>) + nodeAnchorOffsets.capacity() * sizeof(size_t)
               + anchors.capacity() * sizeof(DBKeyType);
    }

private:
    const size_t anchorEntries;
    std::vector<DBReader<DBKeyType> *> parts;
    // where the keys of every part start, then SIZE_MAX
    std::vector<size_t> partKeyOffsets;
    std::vector<size_t> nodeAnchorOffsets;
    std::vector<DBKeyType> anchors;
};

// --createdb-mode 3: a part of a pref file, read at once
struct PrefSplit {
    size_t prefNode;
    size_t firstEntry;
    size_t endEntry;
    size_t prefBytes;
    // the size classes of its records, all until the scan has read them
    uint64_t sizeClasses;
};

// --createdb-mode 3: reads the records of the group ranges that this node claims, in order key order
struct RepTargetsReader {
    RepTargetsReader(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<PrefPartition> &partitions, const std::vector<size_t> &groupRangePartitionOffsets,
                     MMseqsNode::WorkQueue &groupRangeQueue, size_t maxRecordBytes, size_t memoryLimit, size_t paddedDbSize, size_t prefSizeBits)
        : maxRecordBytes(maxRecordBytes), par(par), seqDbr(seqDbr), partitions(partitions), groupRangePartitionOffsets(groupRangePartitionOffsets), groupRangeQueue(groupRangeQueue),
          paddedDbSize(paddedDbSize), prefSizeBits(prefSizeBits), maxChunkBytes(std::max<size_t>(1, memoryLimit / CHUNK_MEMORY_DIVISOR)),
          // a record holds at most maxRecordTargets targets, so they fit maxRecordBytes as pages
          maxRecordTargets(std::min<size_t>(UINT_MAX, std::max<size_t>(1, maxRecordBytes / std::max<size_t>(seqDbr.getMaxSeqLen() + 2, Util::getPageSize())))),
          splitPos(0), passClasses(~0ULL), endClass(0), classBytes(prefSizeBits, 0), prefDb(NULL), prefNode(0), chunkRecordCount(0), chunkTargetCount(0), chunkReadBytes(0), chunkClaimedCost(0),
          entryTargetOffsets(NULL), entryRecordOffsets(NULL), allocatedEntries(0), prefReadBytes(0), prefReads(0), chunkCount(0), claimedGroupRanges(0), readSeconds(0) {}

    // adds the splits of the next group ranges that this node claims, up to one with splits; false once none is left
    bool claimGroupRange() {
        Timer readTimer;
        // a pref byte: two buffers, 8 bytes of targets; an entry: 48 bytes of offsets and keys, a record
        const double bytesPerPrefByte = 10.0;
        const double bytesPerEntry = 48.0 + sizeof(RepTargetsRecord);
        const size_t groupRangeCount = groupRangePartitionOffsets.size() - 1;
        const size_t splitCount = prefSplits.size();
        while (prefSplits.size() == splitCount) {
            const size_t groupRange = (MMseqsNode::rank >= getFirstAlignNode()) ? groupRangeQueue.claimInOrder() : groupRangeCount;
            if (groupRange == groupRangeCount) {
                break;
            }
            claimedGroupRanges++;
            for (size_t i = groupRangePartitionOffsets[groupRange]; i < groupRangePartitionOffsets[groupRange + 1];) {
                if (partitions[i].cost == 0) {
                    i++;
                    continue;
                }
                const size_t node = partitions[i].node;
                const size_t prefBegin = partitions[i].prefOffset;
                openPrefDb(node);
                // partitions next to each other in their pref file are read together
                size_t prefEnd = prefBegin;
                for (; i < groupRangePartitionOffsets[groupRange + 1] && partitions[i].node == node && partitions[i].prefOffset == prefEnd; i++) {
                    prefEnd += partitions[i].prefBytes;
                    chunkClaimedCost += partitions[i].cost;
                }
                const size_t endEntry = prefDb->findEntryByOffset(prefEnd);
                const double entriesPerByte = static_cast<double>(endEntry - prefDb->findEntryByOffset(prefBegin)) / (prefEnd - prefBegin);
                const size_t prefSplitBytes = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(maxChunkBytes) / (bytesPerPrefByte + bytesPerEntry * entriesPerByte)));
                const size_t firstSplit = prefSplits.size();
                for (size_t offset = prefBegin; offset < prefEnd; offset += prefSplitBytes) {
                    const size_t entry = prefDb->findEntryByOffset(offset);
                    if (prefSplits.size() == firstSplit || prefSplits.back().firstEntry != entry) {
                        const PrefSplit split = { node, entry, entry, std::min(prefSplitBytes, prefEnd - offset), ~0ULL };
                        prefSplits.push_back(split);
                    }
                }
                for (size_t split = firstSplit; split < prefSplits.size(); split++) {
                    prefSplits[split].endEntry = (split + 1 < prefSplits.size()) ? prefSplits[split + 1].firstEntry : endEntry;
                }
            }
        }
        readSeconds += readTimer.getTimediff();
        return prefSplits.size() > splitCount;
    }

    // claims all group ranges that this node takes and reads their splits once for the size classes of their records, then the passes start
    void scan() {
        std::vector<RepTargetsRecord> records;
        std::vector<DBKeyType> targets;
        for (splitPos = 0; splitPos < prefSplits.size() || claimGroupRange();) {
            PrefSplit &split = prefSplits[splitPos];
            chunkRecordCount = 0;
            chunkTargetCount = 0;
            loadSplit(records, targets);
            uint64_t sizeClasses = 0;
#pragma omp parallel
            {
                std::vector<size_t> threadClassBytes(prefSizeBits, 0);
#pragma omp for schedule(dynamic, 100) reduction(|:sizeClasses)
                for (size_t i = 0; i < chunkRecordCount; i++) {
                    const RepTargetsRecord &record = records[i];
                    const size_t sizeClass = prefSizeBits - 1 - record.repFirstTarget.repKey / paddedDbSize;
                    sizeClasses |= 1ULL << sizeClass;
                    threadClassBytes[sizeClass] += sizeof(RepTargetsRecord) + record.targetCount * sizeof(DBKeyType);
                }
#pragma omp critical
                for (size_t sizeClass = 0; sizeClass < prefSizeBits; sizeClass++) {
                    classBytes[sizeClass] += threadClassBytes[sizeClass];
                }
            }
            split.sizeClasses = sizeClasses;
        }
        endClass = prefSizeBits;
    }

    // the next records in order key order, about maxChunkBytes of them unless the pass ends; none once no record is left
    // a chunk claims more group ranges while their reads stay below maxReadBytes and their cost below claimCost
    void loadChunk(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets, size_t maxReadBytes, size_t claimCost) {
        records.reserve(maxChunkBytes / sizeof(RepTargetsRecord));
        targets.reserve(maxChunkBytes / sizeof(DBKeyType));
        chunkRecordCount = 0;
        chunkTargetCount = 0;
        chunkReadBytes = 0;
        chunkClaimedCost = 0;
        chunkCount++;
        while (chunkRecordCount * sizeof(RepTargetsRecord) + chunkTargetCount * sizeof(DBKeyType) < maxChunkBytes) {
            if (splitPos < prefSplits.size()) {
                loadSplit(records, targets);
            } else if (chunkRecordCount > 0 && (chunkReadBytes >= maxReadBytes || chunkClaimedCost >= claimCost)) {
                break;
            } else if (claimGroupRange() == false && (chunkRecordCount > 0 || startPass() == false)) {
                break;
            }
        }
        records.resize(chunkRecordCount);
        targets.resize(chunkTargetCount);
        SORT_PARALLEL(records.begin(), records.end(), RepTargetsRecord::compareByRep);
    }

    void close(std::string &phaseValues) {
        closePrefDb();
        delete[] entryTargetOffsets;
        delete[] entryRecordOffsets;
        MMseqsNode::addPhaseSeconds(phaseValues, "reading the prefilter", readSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Prefilter read bytes", SSTR(prefReadBytes) + " in " + SSTR(prefReads) + " reads, " + SSTR(chunkCount) + " chunks");
        MMseqsNode::addPhaseValue(phaseValues, "Claimed group ranges", SSTR(claimedGroupRanges) + " of " + SSTR(groupRangePartitionOffsets.size() - 1));
    }

    size_t getOrderedClaims() const {
        return groupRangeQueue.getOrderedClaims();
    }

    const size_t maxRecordBytes;

private:
    Parameters &par;
    DBReader<DBKeyType> &seqDbr;
    const std::vector<PrefPartition> &partitions;
    const std::vector<size_t> &groupRangePartitionOffsets;
    MMseqsNode::WorkQueue &groupRangeQueue;
    const size_t paddedDbSize;
    const size_t prefSizeBits;
    const size_t maxChunkBytes;
    const size_t maxRecordTargets;
    std::vector<PrefSplit> prefSplits;
    size_t splitPos;
    uint64_t passClasses;
    // the size classes from endClass on have had their pass
    size_t endClass;
    std::vector<size_t> classBytes;
    DBReader<DBKeyType> *prefDb;
    size_t prefNode;
    size_t chunkRecordCount;
    size_t chunkTargetCount;
    size_t chunkReadBytes;
    size_t chunkClaimedCost;
    // not zeroed, as each split overwrites all of its part of them
    size_t *entryTargetOffsets;
    size_t *entryRecordOffsets;
    size_t allocatedEntries;
    size_t prefReadBytes;
    size_t prefReads;
    size_t chunkCount;
    size_t claimedGroupRanges;
    double readSeconds;

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

    // the first split from split on with records of this pass
    size_t findPassSplit(size_t split) {
        while (split < prefSplits.size() && (prefSplits[split].sizeClasses & passClasses) == 0) {
            split++;
        }
        return split;
    }

    bool startPass() {
        if (endClass == 0) {
            return false;
        }
        endClass--;
        passClasses = 1ULL << endClass;
        for (size_t passBytes = classBytes[endClass]; endClass > 0 && passBytes + classBytes[endClass - 1] < maxChunkBytes; passBytes += classBytes[endClass]) {
            endClass--;
            passClasses |= 1ULL << endClass;
        }
        splitPos = findPassSplit(0);
        return true;
    }

    // adds the records of the pass in the split at splitPos and their targets to the chunk and moves on; records and targets only grow here, as growing zeroes them
    void loadSplit(std::vector<RepTargetsRecord> &records, std::vector<DBKeyType> &targets) {
        const PrefSplit &split = prefSplits[splitPos];
        Timer readTimer;
        openPrefDb(split.prefNode);
        prefDb->loadData(split.firstEntry, split.endEntry);
        prefReadBytes += split.prefBytes;
        prefReads++;
        splitPos = findPassSplit(splitPos + 1);
        // a thread that ends before the reads of its io_uring finish can fail them, so this thread prefetches
        if (splitPos < prefSplits.size() && prefSplits[splitPos].prefNode == split.prefNode) {
            prefDb->prefetchData(prefSplits[splitPos].firstEntry, prefSplits[splitPos].endEntry);
        }
        const size_t firstEntry = split.firstEntry;
        const size_t entryCount = split.endEntry - firstEntry;
        if (allocatedEntries < entryCount + 1) {
            delete[] entryTargetOffsets;
            delete[] entryRecordOffsets;
            allocatedEntries = entryCount + 1;
            entryTargetOffsets = new(std::nothrow) size_t[allocatedEntries];
            entryRecordOffsets = new(std::nothrow) size_t[allocatedEntries];
            Util::checkAllocation(entryTargetOffsets, "Can not allocate entryTargetOffsets memory in align2clust");
            Util::checkAllocation(entryRecordOffsets, "Can not allocate entryRecordOffsets memory in align2clust");
        }
        const auto getSizeClass = [&](size_t targetCount) {
            return (targetCount > 0) ? std::min<size_t>(prefSizeBits - 1, 63 - __builtin_clzll(targetCount)) : 0;
        };
        entryTargetOffsets[0] = 0;
        entryRecordOffsets[0] = 0;
#pragma omp parallel for schedule(dynamic, 100)
        for (size_t entry = 0; entry < entryCount; entry++) {
            const char *data = prefDb->getData(firstEntry + entry, 0);
            const size_t targetCount = std::count_if(data, data + prefDb->getEntryLen(firstEntry + entry) - 1, [](char byte) { return static_cast<unsigned char>(byte) < 0x80; }) - 1;
            const bool isInPass = ((passClasses >> getSizeClass(targetCount)) & 1) != 0;
            entryTargetOffsets[entry + 1] = isInPass ? targetCount : 0;
            entryRecordOffsets[entry + 1] = isInPass ? (targetCount + maxRecordTargets - 1) / maxRecordTargets : 0;
        }
        sumOffsets(entryTargetOffsets, entryCount, par.threads);
        sumOffsets(entryRecordOffsets, entryCount, par.threads);
        const size_t firstRecord = chunkRecordCount;
        const size_t firstTarget = chunkTargetCount;
        chunkRecordCount += entryRecordOffsets[entryCount];
        chunkTargetCount += entryTargetOffsets[entryCount];
        records.resize(std::max(records.size(), chunkRecordCount));
        targets.resize(std::max(targets.size(), chunkTargetCount));
        size_t splitReadBytes = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+:splitReadBytes)
        for (size_t entry = 0; entry < entryCount; entry++) {
            const size_t targetCount = entryTargetOffsets[entry + 1] - entryTargetOffsets[entry];
            if (targetCount == 0) {
                continue;
            }
            const char *data = prefDb->getData(firstEntry + entry, 0);
            const DBKeyType repKey = Util::getPrefKey(data);
            const size_t repBytes = seqDbr.getEntryLen(repKey) + WINDOW_KEY_BYTES;
            DBKeyType *entryTargets = targets.data() + firstTarget + entryTargetOffsets[entry];
            for (size_t j = 0; j < targetCount; j++) {
                entryTargets[j] = ((j > 0) ? entryTargets[j - 1] : 0) + Util::getPrefKey(data);
            }
            size_t j = firstRecord + entryRecordOffsets[entry];
            for (size_t first = 0; first < targetCount; first += maxRecordTargets, j++) {
                RepTargetsRecord &record = records[j];
                record.repFirstTarget.repKey = (prefSizeBits - 1 - getSizeClass(targetCount)) * paddedDbSize + repKey;
                record.repFirstTarget.memberKey = entryTargets[first];
                record.targetCount = static_cast<unsigned int>(std::min(targetCount - first, maxRecordTargets));
                record.targetOffset = entryTargets + first - targets.data();
                record.readBytes = repBytes;
                for (unsigned int i = 0; i < record.targetCount; i++) {
                    record.readBytes += seqDbr.getEntryLen(entryTargets[first + i]) + WINDOW_KEY_BYTES;
                }
                splitReadBytes += record.readBytes;
            }
        }
        chunkReadBytes += splitReadBytes;
        readSeconds += readTimer.getTimediff();
    }
};

// the aligners of a thread for the member gates of --filter-cludb-file
struct MemberAligners {
    Sequence query;
    Sequence element;
    BlockAligner blockAligner;
    Matcher matcher;
    SwipeBatch swipe;
    SwipeTargets memberTargets;
    ClusterMembers memberBuffers;

    MemberAligners(const Parameters &par, AlignmentScoring &scoring, size_t maxSeqLen, bool swipeLanes)
        : query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection),
          element(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection),
          blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                       &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                       -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid()),
          matcher(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                  par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid(), 0.0, par.zdrop),
          swipe(par, scoring.subMat, scoring.evaluer, maxSeqLen),
          memberTargets(par, scoring.subMat, scoring.evaluer, maxSeqLen, true),
          memberBuffers(swipeLanes ? &swipe : NULL, true, swipeLanes ? &memberTargets : NULL) {}
};

// --createdb-mode 3: aligns the records window by window and logs the accepted pairs
static void alignRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, RepTargetsReader &repTargetsReader,
                            size_t memoryLimit, size_t masterMemoryLimit, size_t flushEntryBytes, size_t representativeResidues,
                            size_t paddedDbSize, size_t prefSizeBits, ClusterMemberLookup *members, DBReader<DBKeyType> *memberDbr, std::string &phaseValues) {
    const unsigned int nodes = MMseqsNode::count;
    const unsigned int rank = MMseqsNode::rank;
    const size_t firstAlignNode = getFirstAlignNode();
    const size_t alignNodes = nodes - firstAlignNode;
    // a service thread besides --threads gets the entries of a window from their homes and serves the keys of its blocks to the others
    const bool hasService = rank >= firstAlignNode;
    const std::string assignedChanges = par.db3 + "_assigned";
    const std::string requests = par.db3 + "_sequence_requests";
    const std::string served = par.db3 + "_served_sequences";
    const size_t dbSize = seqDbr.getSize();
    const size_t orderKeyEnd = (prefSizeBits - 1) * paddedDbSize + dbSize;
    const size_t maxSeqLen = seqDbr.getMaxSeqLen();
    const size_t maxMemberSeqLen = (memberDbr != NULL) ? std::max(maxSeqLen, memberDbr->getMaxSeqLen()) : maxSeqLen;
    const size_t pageSize = Util::getPageSize();
    AlignmentScoring scoring(par, representativeResidues);
    const unsigned int swMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);
    const std::string acceptedFile = par.db3 + "_accepted." + SSTR(rank);
    FILE *acceptedHandle = FileUtil::openAndDelete(acceptedFile.c_str(), "w");
    size_t acceptedBytes = 0;
    std::vector<uint64_t> acceptedWords;
    DBKeyType loggedRepKey = DB_KEY_INVALID;
    const auto publishAccepted = [&](size_t repKey, size_t bytes) {
        MMseqsNode::publishLog(acceptedHandle, acceptedFile, repKey, bytes, repTargetsReader.getOrderedClaims());
    };
    const auto logAcceptedPairs = [&](const ClusterPair *pairs, size_t pairCount) {
        acceptedWords.clear();
        for (size_t i = 0; i < pairCount; i++) {
            if (pairs[i].repKey != loggedRepKey) {
                loggedRepKey = pairs[i].repKey;
                acceptedWords.push_back(loggedRepKey | ACCEPTED_REP_BIT);
            }
            acceptedWords.push_back(pairs[i].memberKey);
        }
        if (acceptedWords.empty() == false && fwrite(acceptedWords.data(), sizeof(uint64_t), acceptedWords.size(), acceptedHandle) != acceptedWords.size()) {
            Debug(Debug::ERROR) << "Cannot write " << acceptedFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        acceptedBytes += acceptedWords.size() * sizeof(uint64_t);
    };

    // the assigned bits of the keys that this node serves or that are in its group ranges, a page of bits from the first change of a key in it on; other keys read as not assigned
    std::vector<uint64_t *> assignedPages(hasService ? (dbSize + ASSIGNED_PAGE_KEYS - 1) / ASSIGNED_PAGE_KEYS : 0, NULL);
    size_t clusteredKey = 0;
    size_t clusteredAcceptedBytes = 0;
    const auto isAssigned = [&](size_t orderKey) {
        const size_t key = orderKey % paddedDbSize;
        const uint64_t *page = __atomic_load_n(&assignedPages[key / ASSIGNED_PAGE_KEYS], __ATOMIC_ACQUIRE);
        return page != NULL && ((__atomic_load_n(&page[key % ASSIGNED_PAGE_KEYS / 64], __ATOMIC_RELAXED) >> (key % 64)) & 1) != 0;
    };
    const size_t headerWords = getChangeHeaderWords();
    const size_t changeReadWords = 1024 * 1024;
    size_t changesBytes = 0;
    size_t changeReadBytes = 0;
    std::vector<uint64_t> changes;
    const auto readChanges = [&](size_t from, size_t words, bool wait) {
        changes.resize(words);
        const bool isRead = MMseqsNode::readLog(assignedChanges, from, from + words * sizeof(uint64_t), reinterpret_cast<char *>(changes.data()), wait);
        changeReadBytes += isRead ? words * sizeof(uint64_t) : 0;
        return isRead;
    };
    // without wait, a step whose log is not readable yet stays for the next call
    const auto readAssignedChanges = [&](bool wait) {
        size_t publishedKey;
        size_t publishedBytes;
        if (MMseqsNode::readLogProgress(assignedChanges, publishedKey, publishedBytes) == false) {
            return;
        }
        while (changesBytes < publishedBytes) {
            if (readChanges(changesBytes, headerWords, wait) == false) {
                return;
            }
            const size_t stepClusteredKey = changes[0];
            const size_t stepClusteredBytes = changes[1 + MMseqsNode::rank];
            const size_t nodeKeyOffset = changes[1 + MMseqsNode::count + MMseqsNode::rank];
            const size_t nodeKeyEnd = changes[2 + MMseqsNode::count + MMseqsNode::rank];
            const size_t keysStart = changesBytes + headerWords * sizeof(uint64_t);
            const size_t stepEnd = keysStart + changes[1 + 2 * MMseqsNode::count] * sizeof(uint64_t);
            for (size_t first = nodeKeyOffset; first < nodeKeyEnd; first += changeReadWords) {
                const size_t count = std::min(changeReadWords, nodeKeyEnd - first);
                if (readChanges(keysStart + first * sizeof(uint64_t), count, wait) == false) {
                    return;
                }
                // only the thread that holds assignedChangesMutex adds pages
                for (size_t i = 0; i < count; i++) {
                    const size_t key = changes[i];
                    uint64_t *page = assignedPages[key / ASSIGNED_PAGE_KEYS];
                    if (page == NULL) {
                        page = new(std::nothrow) uint64_t[ASSIGNED_PAGE_KEYS / 64]();
                        Util::checkAllocation(page, "Can not allocate assigned page memory in align2clust");
                        __atomic_store_n(&assignedPages[key / ASSIGNED_PAGE_KEYS], page, __ATOMIC_RELEASE);
                    }
                    __atomic_fetch_or(&page[key % ASSIGNED_PAGE_KEYS / 64], 1ULL << (key % 64), __ATOMIC_RELAXED);
                }
            }
            changesBytes = stepEnd;
            clusteredKey = stepClusteredKey;
            clusteredAcceptedBytes = stepClusteredBytes;
        }
    };

    // a loader thread reads chunk n into chunkRecords[n % 3] once the windows have moved on to chunk n - 1, so no window reads chunk n - 3 any more
    std::vector<RepTargetsRecord> chunkRecords[3];
    std::vector<DBKeyType> chunkTargets[3];
    size_t chunkPos[3] = { 0, 0, 0 };
    size_t chunk = 0;
    std::mutex chunkMutex;
    std::condition_variable chunkCondition;
    size_t loadedChunks = 0;
    size_t usedChunks = 0;
    double prefilterWaitSeconds = 0;
    std::vector<std::vector<DBKeyType> > threadKeys(par.threads);
    std::vector<DBKeyType> uniqueKeys;
    std::vector<unsigned int> keyHomes;
    std::vector<size_t> threadHomeOffsets;
    // the bytes a rep can cover, averaged over the keys, of the blocks of a home
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const size_t rangeCount = seqDbr.getIndexSize();
    std::vector<size_t> rangeByteOffsets(rangeCount + 1, 0);
    for (size_t range = 0; range < rangeCount; range++) {
        const size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
        rangeByteOffsets[range + 1] = rangeByteOffsets[range] + (keyEnd - index[range].id) * index[range].length;
    }
    double bandBytes = 0;
    for (size_t range = 0; range < rangeCount; range++) {
        const float repLength = static_cast<float>(index[range].length - 2);
        const auto isCovered = [&](const DBReader<DBKeyType>::Index &coveredRange) {
            return Util::canBeCovered(par.covThr, par.covMode, repLength, static_cast<float>(coveredRange.length - 2));
        };
        // ranges are in length descending order and every --cov-mode covers an interval of lengths that holds the rep length
        const size_t coveredFirst = std::partition_point(index, index + range, [&](const DBReader<DBKeyType>::Index &coveredRange) { return isCovered(coveredRange) == false; }) - index;
        const size_t coveredEnd = std::partition_point(index + range, index + rangeCount, isCovered) - index;
        const size_t keyEnd = (range + 1 < rangeCount) ? static_cast<size_t>(index[range + 1].id) : dbSize;
        bandBytes += static_cast<double>(keyEnd - index[range].id) * (rangeByteOffsets[coveredEnd] - rangeByteOffsets[coveredFirst]);
    }
    const double homeBandBytes = bandBytes / std::max<size_t>(1, dbSize) / alignNodes;
    const size_t averageEntryBytes = std::max<size_t>(1, rangeByteOffsets[rangeCount] / std::max<size_t>(1, dbSize));
    // a flush holds at most sliceKeyLimit keys of every slice, flushEntryBytes of entries plus a key over the share of each request, so at most maxFlushBytes after its header
    const size_t sliceKeyLimit = std::max<size_t>(1, flushEntryBytes / WINDOW_KEY_BYTES);
    const size_t maxFlushEntryBytes = flushEntryBytes + SERVICE_POLL_NODES * (maxSeqLen + 2);
    const size_t maxFlushBytes = SERVICE_POLL_NODES * ((sliceKeyLimit + 7) / 8) + getPackedEntryBytes(maxFlushEntryBytes) + maxFlushEntryBytes;
    // a cache miss reads a page either way, so read directly once that costs less than two packed copies
    const auto isCacheTooSmall = [&](size_t cacheBytes) {
        return alignNodes > 1 && static_cast<double>(cacheBytes) * pageSize <= 2.0 * getPackedEntryBytes(averageEntryBytes) * homeBandBytes;
    };
    // besides the block cache: members, chunks, member parts, two windows, a copyEntries load and, with other aligning nodes, three flushes and the key lists of a slice
    const auto getReservedMemory = [&](size_t windowBytes) {
        return ((members != NULL) ? members->memoryBytes() + (par.threads + 1) * MEMBER_READ_BYTES : 0) + 3 * 2 * (memoryLimit / CHUNK_MEMORY_DIVISOR)
               + ((memberDbr != NULL) ? 2 * (memoryLimit / MEMBER_PART_MEMORY_DIVISOR) : 0)
               + 2 * std::max(windowBytes, repTargetsReader.maxRecordBytes) + windowBytes
               + ((hasService && alignNodes > 1) ? 3 * maxFlushBytes + 6 * sliceKeyLimit * sizeof(size_t) : 0);
    };
    // the largest window that keeps the memory limit and, if any window can, leaves a block cache for the band of the home, down to a record for the share of the home
    const bool canCacheBand = isCacheTooSmall(memoryLimit - std::min(memoryLimit, getReservedMemory(0))) == false;
    const size_t homeShareBytes = rangeByteOffsets[rangeCount] / alignNodes;
    size_t maxWindowBytes = memoryLimit / WINDOW_MEMORY_DIVISOR;
    while (maxWindowBytes > 1 && (getReservedMemory(maxWindowBytes) > memoryLimit || (canCacheBand && isCacheTooSmall(memoryLimit - getReservedMemory(maxWindowBytes)))
                                  || (alignNodes > 1 && maxWindowBytes > repTargetsReader.maxRecordBytes && memoryLimit - std::min(memoryLimit, getReservedMemory(maxWindowBytes)) < homeShareBytes))) {
        maxWindowBytes /= 2;
    }
    const size_t blockCacheBytes = memoryLimit - std::min(memoryLimit, getReservedMemory(maxWindowBytes));
    seqDbr.setBlockCacheBytes(blockCacheBytes);
    const bool readsDirectly = isCacheTooSmall(blockCacheBytes);
    if (hasService) {
        MMseqsNode::addPhaseValue(phaseValues, "Block cache", SSTR(blockCacheBytes) + " bytes for a band of " + SSTR(static_cast<size_t>(homeBandBytes)) + " bytes and a share of " + SSTR(homeShareBytes) + " bytes of the home"
                                 + (readsDirectly ? ", the sequences are read directly" : ""));
    }
    size_t windowLimit = std::min(maxWindowBytes / 64, FIRST_WINDOW_BYTES);
    size_t windowKeptKeys = 0;
    size_t windowKeptBytes = 0;
    const auto findWindowKey = [&](const RepWindow &window, size_t home, DBKeyType key) -> const char * {
        const DBKeyType *keys = window.keys.data();
        const DBKeyType *found = std::lower_bound(keys + window.homeKeyOffsets[home], keys + window.homeKeyOffsets[home + 1], key);
        return (found != keys + window.homeKeyOffsets[home + 1] && *found == key) ? window.data + window.entryOffsets[found - keys] : NULL;
    };
    const auto getWindowEntry = [&](const RepWindow &window, size_t key) -> const char * {
        const char *data = findWindowKey(window, nodes, key);
        return (data != NULL) ? data : findWindowKey(window, readsDirectly ? rank : getKeyNode(seqDbr, key), key);
    };
    const auto getWindowData = [&](const RepWindow &window, size_t key) -> const char * {
        const char *data = getWindowEntry(window, key);
        // the entry of an assigned key is served as zeros
        return (data != NULL && data[0] != '\0') ? data : NULL;
    };
    // a window takes records while their reps and targets fit, at least one; previous is complete and unchanged while it is read
    const auto readWindow = [&](RepWindow &window, const RepWindow &previous) {
        if (chunkPos[chunk] == chunkRecords[chunk].size()) {
            Timer waitTimer;
            std::unique_lock<std::mutex> lock(chunkMutex);
            chunkCondition.wait(lock, [&]() { return loadedChunks > usedChunks; });
            chunk = usedChunks++ % 3;
            lock.unlock();
            chunkCondition.notify_all();
            prefilterWaitSeconds += waitTimer.getTimediff();
            chunkPos[chunk] = 0;
        }
        window.chunk = chunk;
        window.begin = chunkPos[chunk];
        window.end = window.begin;
        for (size_t windowBytes = 0; window.end < chunkRecords[chunk].size(); window.end++) {
            windowBytes += chunkRecords[chunk][window.end].readBytes;
            if (window.end > window.begin && windowBytes > windowLimit) {
                break;
            }
        }
        chunkPos[chunk] = window.end;
        windowLimit = std::min(2 * windowLimit, maxWindowBytes);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            threadKeys[thread_idx].clear();
#pragma omp for schedule(dynamic, 100)
            for (size_t i = window.begin; i < window.end; i++) {
                const RepTargetsRecord &record = chunkRecords[window.chunk][i];
                const size_t repKey = record.repFirstTarget.repKey % paddedDbSize;
                const char *previousRep = getWindowEntry(previous, repKey);
                if (isAssigned(record.repFirstTarget.repKey) || (previousRep != NULL && previousRep[0] == '\0')) {
                    continue;
                }
                const size_t targetsBegin = threadKeys[thread_idx].size();
                const DBKeyType *targets = chunkTargets[window.chunk].data() + record.targetOffset;
                for (unsigned int j = 0; j < record.targetCount; j++) {
                    if (isAssigned(targets[j]) == false) {
                        threadKeys[thread_idx].push_back(targets[j]);
                    }
                }
                // a rep without a target left is not aligned, so its entry is not fetched
                if (threadKeys[thread_idx].size() > targetsBegin) {
                    threadKeys[thread_idx].push_back(repKey);
                }
            }
        }
        uniqueKeys.clear();
        for (size_t thread = 0; thread < threadKeys.size(); thread++) {
            uniqueKeys.insert(uniqueKeys.end(), threadKeys[thread].begin(), threadKeys[thread].end());
        }
        SORT_PARALLEL(uniqueKeys.begin(), uniqueKeys.end());
        uniqueKeys.erase(std::unique(uniqueKeys.begin(), uniqueKeys.end()), uniqueKeys.end());
        keyHomes.resize(uniqueKeys.size());
        window.keys.resize(uniqueKeys.size());
        // the keys go to their homes in key order, every thread placing its static part after those of the lower threads; the keys assigned by now go last and are dropped
        threadHomeOffsets.assign((nodes + 2) * par.threads + 1, 0);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<size_t> homeOffsets(nodes + 2, 0);
#pragma omp for schedule(static)
            for (size_t i = 0; i < uniqueKeys.size(); i++) {
                keyHomes[i] = isAssigned(uniqueKeys[i]) ? nodes + 1 : ((getWindowEntry(previous, uniqueKeys[i]) != NULL) ? nodes : (readsDirectly ? rank : getKeyNode(seqDbr, uniqueKeys[i])));
                homeOffsets[keyHomes[i]]++;
            }
            for (unsigned int node = 0; node <= nodes + 1; node++) {
                threadHomeOffsets[node * par.threads + thread_idx + 1] = homeOffsets[node];
            }
#pragma omp barrier
#pragma omp single
            for (size_t i = 0; i + 1 < threadHomeOffsets.size(); i++) {
                threadHomeOffsets[i + 1] += threadHomeOffsets[i];
            }
            for (unsigned int node = 0; node <= nodes + 1; node++) {
                homeOffsets[node] = threadHomeOffsets[node * par.threads + thread_idx];
            }
#pragma omp for schedule(static)
            for (size_t i = 0; i < uniqueKeys.size(); i++) {
                window.keys[homeOffsets[keyHomes[i]]++] = uniqueKeys[i];
            }
        }
        window.homeKeyOffsets.resize(nodes + 2);
        for (unsigned int node = 0; node <= nodes + 1; node++) {
            window.homeKeyOffsets[node] = threadHomeOffsets[node * par.threads];
        }
        const size_t keyCount = window.homeKeyOffsets[nodes + 1];
        window.keys.resize(keyCount);
        window.entryOffsets.resize(keyCount + 1);
        window.entryOffsets[0] = 0;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < keyCount; i++) {
            window.entryOffsets[i + 1] = seqDbr.getEntryLen(window.keys[i]);
        }
        sumOffsets(window.entryOffsets.data(), keyCount, par.threads);
        // not zeroed, as the entries overwrite all of it
        if (window.allocatedBytes < window.entryOffsets[keyCount]) {
            delete[] window.data;
            window.allocatedBytes = window.entryOffsets[keyCount];
            window.data = new(std::nothrow) char[window.allocatedBytes];
            Util::checkAllocation(window.data, "Can not allocate window data memory in align2clust");
        }
#pragma omp parallel for schedule(dynamic, 1024)
        for (size_t i = window.homeKeyOffsets[nodes]; i < keyCount; i++) {
            memcpy(window.data + window.entryOffsets[i], getWindowEntry(previous, window.keys[i]), window.entryOffsets[i + 1] - window.entryOffsets[i]);
        }
        windowKeptKeys += keyCount - window.homeKeyOffsets[nodes];
        windowKeptBytes += window.entryOffsets[keyCount] - window.entryOffsets[window.homeKeyOffsets[nodes]];
    };
    const auto getFirstRepKey = [&](const RepWindow &window) {
        return (window.end > window.begin) ? static_cast<size_t>(chunkRecords[window.chunk][window.begin].repFirstTarget.repKey) : orderKeyEnd;
    };
    size_t windowCount = 0;
    size_t windowKeys = 0;

    RepWindow window;
    RepWindow nextWindow;
    nextWindow.homeKeyOffsets.assign(nodes + 2, 0);
    // the service thread and the chunk loader take a quarter of the threads for their parallel parts
    const int serviceThreads = std::max(1, par.threads / 4);
    // the alignment cells, as the cost of a partition, that this node aligns per second
    std::atomic<double> alignedCostPerSecond(0);
    repTargetsReader.loadChunk(chunkRecords[0], chunkTargets[0], maxWindowBytes, 0);
    loadedChunks = 1;
    usedChunks = 1;
    // the loader ends with the first empty chunk
    std::thread chunkLoader([&]() {
#ifdef OPENMP
        omp_set_num_threads(serviceThreads);
#endif
        for (size_t next = 1; ; next++) {
            {
                std::unique_lock<std::mutex> lock(chunkMutex);
                chunkCondition.wait(lock, [&]() { return usedChunks >= next; });
            }
            repTargetsReader.loadChunk(chunkRecords[next % 3], chunkTargets[next % 3], maxWindowBytes, static_cast<size_t>(alignedCostPerSecond.load() * CLAIM_SECONDS));
            const bool isLast = chunkRecords[next % 3].empty();
            {
                std::lock_guard<std::mutex> lock(chunkMutex);
                loadedChunks = next + 1;
            }
            chunkCondition.notify_all();
            if (isLast) {
                break;
            }
        }
    });
    readWindow(window, nextWindow);
    publishAccepted(getFirstRepKey(window), 0);
    size_t publishedBytes = 0;
    time_t publishTime = time(NULL);
    time_t changesTime = publishTime;
    Timer changesTimer;
    double batchReadSeconds = 0;
    std::mutex acceptedMutex;
    std::mutex publishMutex;
    std::mutex assignedChangesMutex;
    std::vector<size_t> batchOffsets;
    std::vector<unsigned int> batchFirstTargets;
    std::vector<size_t> recordCosts;
    std::vector<unsigned char> batchDone;
    std::vector<std::vector<ClusterPair> > batchPairs;
    std::vector<ClusterPair> windowPairs;
    std::vector<size_t> memberOffsets;
    std::vector<size_t> memberBytes;
    std::vector<size_t> partOffsets;
    std::vector<unsigned char> membersPassed;
    std::vector<unsigned char> hasMemberGate;
    std::vector<ClusterMemberLookup::Entry> memberEntries;
    std::vector<DBKeyType> partKeys;
    std::vector<DBKeyType> partMembers;
    std::vector<DBKeyType> nextPartKeys;
    std::vector<DBKeyType> nextPartMembers;
    std::vector<size_t> repGroups;
    std::vector<size_t> batchGroupOffsets;
    double waitSeconds = 0;
    double sequenceWaitSeconds = 0;
    size_t alignedPairs = 0;
    size_t assignedRepSkips = 0;
    size_t unservedRepSkips = 0;
    size_t assignedTargetSkips = 0;
    size_t unservedTargetSkips = 0;
    size_t repBatchCount = 0;
    double repBatchWorkerSeconds = 0;
    double repBatchWallSeconds = 0;
    double memberLookupSeconds = 0;
    double memberReadSeconds = 0;
    double memberGateSeconds = 0;
    size_t skippedMemberGates = 0;
    double repBatchLockWaitSeconds = 0;
    double repBatchLongestSeconds = 0;
    double alignedCost = 0;
    double alignSeconds = 0;
    const auto mapRep = [&](Sequence &query, BlockAligner &blockAligner, size_t repKey) {
        query.mapSequence(repKey, repKey, getWindowData(window, repKey), seqDbr.getSeqLen(repKey));
        blockAligner.setQuery(&query);
    };
    const bool swipeLanes = par.align2clustAlignerMode != Parameters::ALIGNER_MODE_DEFAULT && par.gapOpen.values.aminoacid() >= par.gapExtend.values.aminoacid();

    std::mutex serviceMutex;
    std::condition_variable serviceCondition;
    RepWindow *pendingWindow = NULL;
    // the accepted log bytes when pendingWindow was read; a node aligns ahead of the master by at most its share of a master step
    size_t requestAcceptedBytes = 0;
    const size_t maxUnclusteredAcceptedBytes = std::max<size_t>(1, getStepWords(masterMemoryLimit) / nodes) * sizeof(uint64_t);
    RepWindow *fetchedWindow = NULL;
    bool isWaitingForMaster = false;
    std::vector<DBKeyType> loadKeys;
    std::vector<std::vector<size_t> > threadRawEntries(serviceThreads);
    size_t unservedSequenceBytes = 0;
    size_t servedFlushCount = 0;
    size_t largestServedFlushBytes = 0;
    size_t servedBytes = 0;
    size_t receivedSliceCount = 0;
    size_t receivedSliceBytes = 0;
    size_t requestFlushCount = 0;
    double servingSeconds = 0;
    double writingSeconds = 0;
    double flushWriterWaitSeconds = 0;
    double readingSeconds = 0;
    double unpackingSeconds = 0;
    double pollingSeconds = 0;
    // copies or, with rawKeys, packs the entries of keys to data at offsets on the service threads; an assigned key gets zeros, rawKeys gets the keys without codes
    const auto copyEntries = [&](const DBKeyType *keys, size_t keyCount, const size_t *offsets, char *data, std::vector<DBKeyType> *rawKeys) {
        for (size_t first = 0; first < keyCount;) {
            const size_t loadFirst = first;
            // a key reads at most its entry and two pages, so a load fits maxWindowBytes
            for (size_t loadBytes = 0; first < keyCount && (first == loadFirst || loadBytes + seqDbr.getEntryLen(keys[first]) + 2 * pageSize <= maxWindowBytes); first++) {
                loadBytes += seqDbr.getEntryLen(keys[first]) + 2 * pageSize;
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
                    // bits are never cleared, so a key left out of loadKeys is still assigned
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
    if (hasService) {
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
            // a request index line: the flush, whether it is the last, where the keys of every node start, then the end
            std::vector<uint64_t> requestLine(nodes + 4);
            // a served index line: the flush, where the slice of every node starts in its file, the end, then where the slice of every node ends in the keys of its request for this home
            const size_t servedLineWords = 2 * nodes + 3;
            std::vector<uint64_t> servedLine(servedLineWords);
            // the lines of a home read at once
            std::vector<uint64_t> servedLines(nodes * servedLineWords);
            std::vector<size_t> nodeRequests(nodes, 0);
            // per home: the next served index line, the flush, start, end, file bytes and key end of the next slice of the requested window, and the keys received so far
            std::vector<size_t> nodeServedFlushes(nodes, 0);
            std::vector<size_t> sliceFlushes(nodes, SIZE_MAX);
            std::vector<size_t> sliceBegins(nodes, 0);
            std::vector<size_t> sliceEnds(nodes, 0);
            std::vector<size_t> flushFileBytes(nodes, 0);
            std::vector<size_t> sliceKeyEnds(nodes, 0);
            std::vector<size_t> homeReceivedKeys(nodes, 0);
            std::vector<unsigned char> isRequesting(nodes, 0);
            for (size_t node = firstAlignNode; node < nodes; node++) {
                isRequesting[node] = (node != rank);
            }
            const size_t polledNodes = std::min<size_t>(SERVICE_POLL_NODES, nodes - 1);
            // an idle pass sleeps pollMicroseconds and an index that had no new line is read again after it, so busy passes open no more indexes than idle ones
            const size_t pollMicroseconds = std::min(MAX_SERVICE_PASS_MICROSECONDS, 2 * polledNodes * nodes * 1000000 / SERVICE_OPENS_PER_SECOND);
            Timer serviceClock;
            std::vector<double> nextServedIndexPolls(nodes, 0);
            std::vector<double> nextRequestIndexPolls(nodes, 0);
            std::vector<time_t> missingSliceSince(nodes, 0);
            std::vector<unsigned int> missingSliceAttempts(nodes, 0);
            std::vector<time_t> lastDirectoryRefresh(nodes, 0);
            std::vector<time_t> missingRequestSince(nodes, 0);
            std::vector<unsigned int> missingRequestAttempts(nodes, 0);
            std::vector<time_t> lastRequestDirectoryRefresh(nodes, 0);
            // a request this home serves slice by slice: its node, its request flush, where its key codes for this home start, where the next one is and where they end in the request file, the request file bytes, the keys served so far and the last of them
            struct ServedRequest {
                unsigned int node;
                size_t flush;
                size_t codeStart;
                size_t codeOffset;
                size_t codeEnd;
                size_t requestBytes;
                size_t servedKeys;
                DBKeyType lastKey;
            };
            std::vector<ServedRequest> servedRequests;
            std::vector<unsigned char> isServing(nodes, 0);
            std::vector<DBKeyType> sliceKeys;
            std::vector<char> requestCodes((alignNodes > 1) ? sliceKeyLimit * sizeof(DBKeyType) + 10 : 0);
            // the flushes with a slice of every node not read yet, and the unread slices of every flush
            std::vector<std::vector<size_t> > nodeUnreadFlushes(nodes);
            std::map<size_t, size_t> flushUnreadSlices;
            std::vector<size_t> requestOffsets;
            std::vector<char> slice;
            // a thread writes and publishes a flush while the next one is built
            std::thread flushWriter;
            std::vector<char> writtenSlice;
            std::vector<DBKeyType> rawKeys;
            std::vector<char> receivedSlices;
            struct SliceRead {
                int fd;
                unsigned int home;
                size_t receivedOffset;
                size_t bytes;
            };
            std::vector<SliceRead> sliceReads;
            // a flush that is missing or short is tried again in a later pass, until it has been so for too many attempts in a row, as the time alone includes the work between two attempts
            const auto failIfMissingTooLong = [](time_t &missingSince, unsigned int &attempts, const auto &getError) {
                const time_t now = time(NULL);
                if (attempts++ == 0) {
                    missingSince = now;
                }
                if (attempts >= 100 && now - missingSince >= 60) {
                    Debug(Debug::ERROR) << getError();
                    EXIT(EXIT_FAILURE);
                }
            };
            const auto describeFlush = [](size_t bytes, size_t observedBytes, uint64_t generation, uint64_t observedGeneration, int error) {
                return " (expected " + SSTR(bytes) + " bytes, observed " + (observedBytes == SIZE_MAX ? std::string("unavailable") : SSTR(observedBytes))
                       + " bytes; expected generation " + SSTR(generation) + ", observed " + (observedGeneration == UINT64_MAX ? std::string("unavailable") : SSTR(observedGeneration))
                       + "; error " + SSTR(error) + ")\n";
            };
            MMseqsNode::WaitWarning waitWarning;
            RepWindow *requestedWindow = NULL;
            bool isLastRequested = false;
            time_t changesReadTime = 0;
            size_t pollTurn = 0;
            for (unsigned int polls = 0; isLastRequested == false || requestedWindow != NULL || std::find(isRequesting.begin(), isRequesting.end(), 1) != isRequesting.end()
                                         || servedRequests.empty() == false;) {
                bool hasWorked = false;
                // the bits of the served keys stay current while the aligning threads wait
                const time_t passTime = time(NULL);
                if (passTime - changesReadTime >= READ_SECONDS) {
                    std::unique_lock<std::mutex> changesLock(assignedChangesMutex, std::try_to_lock);
                    if (changesLock.owns_lock()) {
                        readAssignedChanges(false);
                        changesReadTime = passTime;
                    }
                }
                // request the next window once the master has clustered all but maxUnclusteredAcceptedBytes of the log before it, or is at its first rep
                RepWindow *nextRequest = NULL;
                size_t nextAcceptedBytes = 0;
                if (requestedWindow == NULL) {
                    std::lock_guard<std::mutex> lock(serviceMutex);
                    nextRequest = pendingWindow;
                    nextAcceptedBytes = requestAcceptedBytes;
                }
                if (nextRequest != NULL) {
                    const bool isLast = nextRequest->end == nextRequest->begin;
                    bool isAhead;
                    {
                        std::lock_guard<std::mutex> changesLock(assignedChangesMutex);
                        const size_t repKey = getFirstRepKey(*nextRequest);
                        if (isLast == false && repKey > clusteredKey && nextAcceptedBytes >= clusteredAcceptedBytes + maxUnclusteredAcceptedBytes) {
                            readAssignedChanges(true);
                        }
                        isAhead = isLast == false && repKey > clusteredKey && nextAcceptedBytes >= clusteredAcceptedBytes + maxUnclusteredAcceptedBytes;
                    }
                    {
                        std::lock_guard<std::mutex> lock(serviceMutex);
                        isWaitingForMaster = isAhead;
                        if (isAhead == false) {
                            pendingWindow = NULL;
                        }
                    }
                    if (isAhead == false) {
                        if (alignNodes > 1) {
                            FILE *requestHandle = FileUtil::openAndDelete(requestFiles[rank].c_str(), "w");
                            MMseqsNode::writeExchangeSliceHeader(requestHandle, requestFiles[rank], requestFlushCount);
                            size_t requestBytes = MMseqsNode::EXCHANGE_SLICE_HEADER_BYTES;
                            for (unsigned int node = 0; node < nodes; node++) {
                                requestLine[2 + node] = requestBytes;
                                const size_t keyBegin = nextRequest->homeKeyOffsets[node];
                                const size_t keyEnd = (node != rank) ? nextRequest->homeKeyOffsets[node + 1] : keyBegin;
                                // the keys of a home ascend, so they go as gaps, in parts that fit requestCodes
                                for (size_t i = keyBegin; i < keyEnd;) {
                                    size_t codeBytes = 0;
                                    for (; i < keyEnd && codeBytes + 10 <= requestCodes.size(); i++) {
                                        codeBytes += Util::putPrefKey(requestCodes.data() + codeBytes, nextRequest->keys[i] - ((i > keyBegin) ? nextRequest->keys[i - 1] : 0));
                                    }
                                    if (fwrite(requestCodes.data(), sizeof(char), codeBytes, requestHandle) != codeBytes) {
                                        Debug(Debug::ERROR) << "Cannot write " << requestFiles[rank] << "\n";
                                        EXIT(EXIT_FAILURE);
                                    }
                                    requestBytes += codeBytes;
                                }
                            }
                            requestLine[0] = requestFlushCount++;
                            requestLine[1] = isLast;
                            requestLine[2 + nodes] = requestBytes;
                            MMseqsNode::publishExchangeFlush(requestHandle, requestFiles[rank], requestBytes, requestIndexes[rank], requestLine);
                        }
                        const size_t ownKey = nextRequest->homeKeyOffsets[rank];
                        copyEntries(nextRequest->keys.data() + ownKey, nextRequest->homeKeyOffsets[rank + 1] - ownKey, nextRequest->entryOffsets.data() + ownKey, nextRequest->data, NULL);
                        isLastRequested = isLast;
                        requestedWindow = nextRequest;
                        hasWorked = true;
                    }
                }
                // receive the next slice of up to SERVICE_POLL_NODES homes that have served, from pollTurn on so the nodes do not all read one home first, while they fit maxFlushBytes or one is read
                if (requestedWindow != NULL) {
                    bool isComplete = true;
                    size_t polledHomes = 0;
                    size_t receivedBytes = 0;
                    for (size_t i = 0; i + 1 < nodes; i++) {
                        const unsigned int home = (rank + 1 + (i + pollTurn) % (nodes - 1)) % nodes;
                        const size_t homeKeyCount = requestedWindow->homeKeyOffsets[home + 1] - requestedWindow->homeKeyOffsets[home];
                        if (homeReceivedKeys[home] == homeKeyCount) {
                            continue;
                        }
                        if (++polledHomes > polledNodes) {
                            isComplete = false;
                            continue;
                        }
                        // the read stops at the line with a slice for this node
                        if (sliceFlushes[home] == SIZE_MAX) {
                            if (serviceClock.getTimediff() < nextServedIndexPolls[home]) {
                                isComplete = false;
                                continue;
                            }
                            Timer pollTimer;
                            const size_t lineCount = MMseqsNode::readExchangeFlushes(servedIndexes[home], nodeServedFlushes[home], servedLineWords, servedLines);
                            pollingSeconds += pollTimer.getTimediff();
                            if (lineCount == 0) {
                                nextServedIndexPolls[home] = serviceClock.getTimediff() + pollMicroseconds / 1e6;
                            }
                            for (size_t line = 0; line < lineCount && sliceFlushes[home] == SIZE_MAX; line++) {
                                const uint64_t *sliceOffsets = servedLines.data() + line * servedLineWords + 1;
                                if (sliceOffsets[rank] < sliceOffsets[rank + 1]) {
                                    sliceFlushes[home] = nodeServedFlushes[home];
                                    sliceBegins[home] = sliceOffsets[rank];
                                    sliceEnds[home] = sliceOffsets[rank + 1];
                                    flushFileBytes[home] = sliceOffsets[nodes];
                                    sliceKeyEnds[home] = sliceOffsets[nodes + 1 + rank];
                                }
                                nodeServedFlushes[home]++;
                            }
                            if (sliceFlushes[home] == SIZE_MAX) {
                                isComplete = false;
                                continue;
                            }
                        }
                        if (sliceReads.empty() == false && receivedBytes + sliceEnds[home] - sliceBegins[home] > maxFlushBytes) {
                            isComplete = false;
                            continue;
                        }
                        const std::string flushFile = served + "." + SSTR(home) + "." + SSTR(sliceFlushes[home]);
                        size_t observedBytes;
                        uint64_t observedGeneration;
                        int lastError;
                        Timer openTimer;
                        const int fd = MMseqsNode::tryOpenExchangeSlice(flushFile, flushFileBytes[home], sliceFlushes[home], lastDirectoryRefresh[home], observedBytes, observedGeneration, lastError);
                        readingSeconds += openTimer.getTimediff();
                        if (fd < 0) {
                            if (lastError != 0 && lastError != ENOENT && lastError != ESTALE) {
                                Debug(Debug::ERROR) << "Cannot open sequence slice " << flushFile << ": " << strerror(lastError) << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                            failIfMissingTooLong(missingSliceSince[home], missingSliceAttempts[home], [&]() { return "Cannot open complete sequence slice " + flushFile
                                                 + describeFlush(flushFileBytes[home], observedBytes, sliceFlushes[home], observedGeneration, lastError); });
                            isComplete = false;
                            continue;
                        }
                        const SliceRead request = { fd, home, receivedBytes, sliceEnds[home] - sliceBegins[home] };
                        sliceReads.push_back(request);
                        receivedBytes += request.bytes;
                    }
                    if (isComplete || sliceReads.empty() == false) {
                        if (sliceReads.empty() == false) {
                            Timer readTimer;
                            if (receivedSlices.size() < receivedBytes) {
                                receivedSlices.resize(receivedBytes);
                            }
                            // the slices are read in parts of the bytes a flush writes at once, by slice and offset; a failed part keeps its error, -1 without errno, 0 when it did not fail
                            std::vector<std::pair<size_t, size_t> > chunks;
                            for (size_t i = 0; i < sliceReads.size(); i++) {
                                for (size_t offset = 0; offset < sliceReads[i].bytes; offset += MMseqsNode::EXCHANGE_WRITE_BYTES) {
                                    chunks.push_back(std::make_pair(i, offset));
                                }
                            }
                            std::vector<int> chunkErrors(chunks.size(), 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(serviceThreads)
                            for (size_t i = 0; i < chunks.size(); i++) {
                                const SliceRead &request = sliceReads[chunks[i].first];
                                const size_t offset = chunks[i].second;
                                int readError;
                                if (MMseqsNode::tryReadExchangeSlice(request.fd, receivedSlices.data() + request.receivedOffset + offset, std::min(MMseqsNode::EXCHANGE_WRITE_BYTES, request.bytes - offset),
                                                                     sliceBegins[request.home] + offset, readError) == false) {
                                    chunkErrors[i] = (readError != 0) ? readError : -1;
                                }
                            }
                            readingSeconds += readTimer.getTimediff();
                            for (size_t i = 0, chunk = 0; i < sliceReads.size(); i++) {
                                int readError = 0;
                                for (; chunk < chunks.size() && chunks[chunk].first == i; chunk++) {
                                    readError = (chunkErrors[chunk] != 0) ? chunkErrors[chunk] : readError;
                                }
                                const unsigned int home = sliceReads[i].home;
                                const std::string flushFile = served + "." + SSTR(home) + "." + SSTR(sliceFlushes[home]);
                                const int closeError = (close(sliceReads[i].fd) != 0) ? errno : 0;
                                const int error = (readError > 0) ? readError : closeError;
                                if (error != 0 && error != ESTALE && error != EINTR) {
                                    Debug(Debug::ERROR) << "Cannot read sequence slice " << flushFile << ": " << strerror(error) << "\n";
                                    EXIT(EXIT_FAILURE);
                                }
                                if (readError != 0 || closeError != 0) {
                                    failIfMissingTooLong(missingSliceSince[home], missingSliceAttempts[home], [&]() {
                                        return "Cannot read complete sequence slice " + flushFile + " (generation " + SSTR(sliceFlushes[home]) + "; error " + SSTR(error) + ")\n"; });
                                    isComplete = false;
                                } else {
                                    Timer unpackTimer;
                                    const size_t firstKey = requestedWindow->homeKeyOffsets[home] + homeReceivedKeys[home];
                                    unpackEntries(receivedSlices.data() + sliceReads[i].receivedOffset, sliceKeyEnds[home] - homeReceivedKeys[home],
                                                  requestedWindow->entryOffsets.data() + firstKey, requestedWindow->data, serviceThreads);
                                    unpackingSeconds += unpackTimer.getTimediff();
                                    homeReceivedKeys[home] = sliceKeyEnds[home];
                                    sliceFlushes[home] = SIZE_MAX;
                                    missingSliceAttempts[home] = 0;
                                    receivedSliceCount++;
                                    receivedSliceBytes += sliceReads[i].bytes;
                                    hasWorked = true;
                                    if (homeReceivedKeys[home] < requestedWindow->homeKeyOffsets[home + 1] - requestedWindow->homeKeyOffsets[home]) {
                                        isComplete = false;
                                    }
                                }
                            }
                        }
                        sliceReads.clear();
                        if (isComplete) {
                            homeReceivedKeys.assign(nodes, 0);
                            {
                                std::lock_guard<std::mutex> lock(serviceMutex);
                                fetchedWindow = requestedWindow;
                            }
                            serviceCondition.notify_all();
                            requestedWindow = NULL;
                            hasWorked = true;
                        }
                    }
                }
                // take the next request of the polled nodes, up to SERVICE_POLL_NODES served at once
                for (size_t i = 0; i < polledNodes; i++) {
                    const unsigned int node = (rank + 1 + (i + pollTurn) % (nodes - 1)) % nodes;
                    while (servedRequests.size() < SERVICE_POLL_NODES && isRequesting[node] && isServing[node] == 0 && serviceClock.getTimediff() >= nextRequestIndexPolls[node]) {
                        Timer pollTimer;
                        const bool hasLine = MMseqsNode::readExchangeFlushes(requestIndexes[node], nodeRequests[node], requestLine.size(), requestLine) == 1;
                        pollingSeconds += pollTimer.getTimediff();
                        if (hasLine == false) {
                            nextRequestIndexPolls[node] = serviceClock.getTimediff() + pollMicroseconds / 1e6;
                            break;
                        }
                        // a node requests once it has read its slices before, so a flush file goes once all its slices are read
                        for (size_t j = 0; j < nodeUnreadFlushes[node].size(); j++) {
                            const size_t unreadFlush = nodeUnreadFlushes[node][j];
                            if (--flushUnreadSlices[unreadFlush] == 0) {
                                FileUtil::remove((served + "." + SSTR(rank) + "." + SSTR(unreadFlush)).c_str());
                                flushUnreadSlices.erase(unreadFlush);
                            }
                        }
                        nodeUnreadFlushes[node].clear();
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
                // a flush a pass: in node order a slice of each request, its next keys whose entries fit its share of flushEntryBytes, at least one
                size_t flushSlices = 0;
                size_t sliceBytes = 0;
                std::sort(servedRequests.begin(), servedRequests.end(), [](const ServedRequest &first, const ServedRequest &second) { return first.node < second.node; });
                const size_t share = flushEntryBytes / std::max<size_t>(1, servedRequests.size());
                for (size_t request = 0, node = 0; node < nodes && servedRequests.empty() == false; node++) {
                    servedLine[1 + node] = MMseqsNode::EXCHANGE_SLICE_HEADER_BYTES + sliceBytes;
                    servedLine[nodes + 2 + node] = 0;
                    if (request == servedRequests.size() || servedRequests[request].node != node) {
                        continue;
                    }
                    ServedRequest &servedRequest = servedRequests[request++];
                    const size_t readKeys = std::min(sliceKeyLimit, 2 * share / averageEntryBytes + 1);
                    // up to 8 B a key at first, then the bytes a key of this request took so far, and at least a longest code
                    const double keyCodeBytes = (servedRequest.servedKeys > 0) ? static_cast<double>(servedRequest.codeOffset - servedRequest.codeStart) / servedRequest.servedKeys : sizeof(DBKeyType);
                    const size_t codeBytes = std::min(std::min(servedRequest.codeEnd - servedRequest.codeOffset, requestCodes.size()), static_cast<size_t>(readKeys * keyCodeBytes) + 10);
                    sliceKeys.resize(readKeys);
                    size_t observedBytes;
                    uint64_t observedGeneration;
                    int lastError;
                    Timer requestTimer;
                    const int requestFd = MMseqsNode::tryOpenExchangeSlice(requestFiles[node], servedRequest.requestBytes, servedRequest.flush,
                                                                            lastRequestDirectoryRefresh[node], observedBytes, observedGeneration, lastError);
                    if (requestFd < 0 && lastError != 0 && lastError != ENOENT && lastError != ESTALE) {
                        Debug(Debug::ERROR) << "Cannot open sequence request " << requestFiles[node] << ": " << strerror(lastError) << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    bool readRequest = requestFd >= 0;
                    if (readRequest) {
                        readRequest = MMseqsNode::tryReadExchangeSlice(requestFd, requestCodes.data(), codeBytes, servedRequest.codeOffset, lastError);
                        if (close(requestFd) != 0 && readRequest) {
                            lastError = errno;
                            readRequest = false;
                        }
                    }
                    servingSeconds += requestTimer.getTimediff();
                    if (readRequest == false) {
                        if (lastError != 0 && lastError != ENOENT && lastError != ESTALE && lastError != EINTR) {
                            Debug(Debug::ERROR) << "Cannot read sequence request " << requestFiles[node] << ": " << strerror(lastError) << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        failIfMissingTooLong(missingRequestSince[node], missingRequestAttempts[node], [&]() { return "Cannot read complete sequence request " + requestFiles[node]
                                             + describeFlush(servedRequest.requestBytes, observedBytes, servedRequest.flush, observedGeneration, lastError); });
                        continue;
                    }
                    missingRequestAttempts[node] = 0;
                    Timer servingTimer;
                    const DBKeyType *keys = sliceKeys.data();
                    // the read may end inside a code; only the last byte of a code is below 0x80
                    const char *codeEnd = requestCodes.data() + codeBytes;
                    while (codeEnd > requestCodes.data() && static_cast<unsigned char>(codeEnd[-1]) >= 0x80) {
                        codeEnd--;
                    }
                    const char *code = requestCodes.data();
                    size_t keyCount = 0;
                    for (size_t entryBytes = 0; keyCount < readKeys && code < codeEnd; keyCount++) {
                        const char *nextCode = code;
                        const DBKeyType key = ((keyCount > 0) ? keys[keyCount - 1] : servedRequest.lastKey) + Util::getPrefKey(nextCode);
                        const size_t keyBytes = isAssigned(key) ? 0 : seqDbr.getEntryLen(key);
                        if (keyCount > 0 && entryBytes + keyBytes > share) {
                            break;
                        }
                        sliceKeys[keyCount] = key;
                        code = nextCode;
                        entryBytes += keyBytes;
                    }
                    servedRequest.codeOffset += code - requestCodes.data();
                    servedRequest.lastKey = keys[keyCount - 1];
                    requestOffsets.resize(keyCount + 1);
                    requestOffsets[0] = (keyCount + 7) / 8;
                    size_t requestUnservedBytes = 0;
#pragma omp parallel for schedule(static) num_threads(serviceThreads) reduction(+:requestUnservedBytes)
                    for (size_t j = 0; j < keyCount; j++) {
                        const size_t packedBytes = getPackedEntryBytes(seqDbr.getEntryLen(keys[j]));
                        const bool isServed = isAssigned(keys[j]) == false;
                        requestOffsets[j + 1] = isServed ? packedBytes : 0;
                        requestUnservedBytes += isServed ? 0 : packedBytes;
                    }
                    unservedSequenceBytes += requestUnservedBytes;
                    for (size_t j = 0; j < keyCount; j++) {
                        requestOffsets[j + 1] += requestOffsets[j];
                    }
                    if (slice.size() < sliceBytes + requestOffsets[keyCount]) {
                        slice.resize(sliceBytes + requestOffsets[keyCount]);
                    }
                    // a served entry has at least one byte, so an empty slot marks an assigned key; bits are never cleared, so copyEntries writes nothing for it
#pragma omp parallel for schedule(static) num_threads(serviceThreads)
                    for (size_t byte = 0; byte < requestOffsets[0]; byte++) {
                        unsigned char bits = 0;
                        for (size_t j = 8 * byte; j < std::min(keyCount, 8 * byte + 8); j++) {
                            bits |= static_cast<unsigned char>(requestOffsets[j + 1] == requestOffsets[j]) << (j % 8);
                        }
                        slice[sliceBytes + byte] = static_cast<char>(bits);
                    }
                    rawKeys.clear();
                    copyEntries(keys, keyCount, requestOffsets.data(), slice.data() + sliceBytes, &rawKeys);
                    sliceBytes += requestOffsets[keyCount];
                    requestOffsets.resize(rawKeys.size() + 1);
                    requestOffsets[0] = 0;
                    for (size_t j = 0; j < rawKeys.size(); j++) {
                        requestOffsets[j + 1] = requestOffsets[j] + seqDbr.getEntryLen(rawKeys[j]);
                    }
                    if (slice.size() < sliceBytes + requestOffsets[rawKeys.size()]) {
                        slice.resize(sliceBytes + requestOffsets[rawKeys.size()]);
                    }
                    copyEntries(rawKeys.data(), rawKeys.size(), requestOffsets.data(), slice.data() + sliceBytes, NULL);
                    sliceBytes += requestOffsets[rawKeys.size()];
                    servingSeconds += servingTimer.getTimediff();
                    servedRequest.servedKeys += keyCount;
                    servedLine[nodes + 2 + node] = servedRequest.servedKeys;
                    nodeUnreadFlushes[node].push_back(servedFlushCount);
                    flushSlices++;
                }
                for (size_t request = 0; request < servedRequests.size();) {
                    if (servedRequests[request].codeOffset == servedRequests[request].codeEnd) {
                        isServing[servedRequests[request].node] = 0;
                        servedRequests.erase(servedRequests.begin() + request);
                    } else {
                        request++;
                    }
                }
                if (flushSlices > 0) {
                    servedLine[0] = servedFlushCount;
                    servedLine[1 + nodes] = MMseqsNode::EXCHANGE_SLICE_HEADER_BYTES + sliceBytes;
                    flushUnreadSlices[servedFlushCount] = flushSlices;
                    // one flush is written at a time, so the index lines stay in flush order
                    Timer waitTimer;
                    if (flushWriter.joinable()) {
                        flushWriter.join();
                    }
                    flushWriterWaitSeconds += waitTimer.getTimediff();
                    slice.swap(writtenSlice);
                    const size_t writtenFlush = servedFlushCount;
                    flushWriter = std::thread([&, sliceBytes, servedLine, writtenFlush]() {
                        Timer writeTimer;
                        MMseqsNode::writeExchangeFlush(served + "." + SSTR(rank) + "." + SSTR(writtenFlush), writtenFlush, writtenSlice.data(), sliceBytes,
                                                       servedIndexes[rank], servedLine, serviceThreads);
                        writingSeconds += writeTimer.getTimediff();
                    });
                    servedFlushCount++;
                    largestServedFlushBytes = std::max(largestServedFlushBytes, sliceBytes);
                    servedBytes += sliceBytes;
                    hasWorked = true;
                }
                pollTurn += polledNodes;
                // idle: poll again after up to 160 ms while this or another aligning node may wait, else up to 1.28 s
                if (hasWorked) {
                    polls = 0;
                } else {
                    usleep(pollMicroseconds);
                    MMseqsNode::sleepBeforePoll(polls++, (alignNodes > 1 || requestedWindow != NULL || nextRequest != NULL) ? 4 : 7);
                }
                // the main thread aligns while it requests nothing, so that is no wait
                if (hasWorked || (requestedWindow == NULL && nextRequest == NULL && isLastRequested == false)) {
                    waitWarning.reset();
                } else {
                    waitWarning.check(requestedWindow != NULL ? served + " of other nodes" : (nextRequest != NULL ? assignedChanges : requests + " of other nodes"));
                }
            }
            if (flushWriter.joinable()) {
                flushWriter.join();
            }
        });
    }
    const auto fetchWindow = [&](RepWindow &fetched) {
        std::lock_guard<std::mutex> lock(serviceMutex);
        fetchedWindow = NULL;
        pendingWindow = &fetched;
        requestAcceptedBytes = acceptedBytes;
    };
    const auto waitForWindow = [&](RepWindow &fetched, size_t doneRepKey) {
        std::unique_lock<std::mutex> lock(serviceMutex);
        if (fetchedWindow == &fetched) {
            return;
        }
        lock.unlock();
        publishAccepted(doneRepKey, acceptedBytes);
        publishedBytes = acceptedBytes;
        publishTime = time(NULL);
        Timer waitTimer;
        lock.lock();
        const bool waitsForMaster = isWaitingForMaster;
        serviceCondition.wait(lock, [&]() { return fetchedWindow == &fetched; });
        (waitsForMaster ? waitSeconds : sequenceWaitSeconds) += waitTimer.getTimediff();
    };
    if (hasService) {
        fetchWindow(window);
        waitForWindow(window, getFirstRepKey(window));
    }
    while (window.end > window.begin) {
        readWindow(nextWindow, window);
        fetchWindow(nextWindow);
        Timer alignTimer;
        const size_t nextRepKey = getFirstRepKey(nextWindow);
        const std::vector<RepTargetsRecord> &records = chunkRecords[window.chunk];
        const size_t recordCount = window.end - window.begin;
        // batches of about the same cost; batch b starts at target batchFirstTargets[b] of record batchOffsets[b], so a rep that costs more than a batch goes to several
        const size_t maxBatches = par.threads * (swipeLanes ? ALIGN_BATCHES_PER_THREAD * ALIGN_BATCH_LANES / SwipeExtender::LANES : ALIGN_BATCHES_PER_THREAD);
        // a record costs its rep length times its target lengths, like a partition
        recordCosts.resize(recordCount);
#pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < recordCount; i++) {
            const RepTargetsRecord &record = records[window.begin + i];
            const DBKeyType *targets = chunkTargets[window.chunk].data() + record.targetOffset;
            size_t targetRange = SIZE_MAX;
            size_t targetResidues = 0;
            for (unsigned int j = 0; j < record.targetCount; j++) {
                targetResidues += seqDbr.getSeqLenFromRange(targetRange, targets[j]);
            }
            recordCosts[i] = seqDbr.getSeqLen(record.repFirstTarget.repKey % paddedDbSize) * targetResidues;
        }
        uint128 windowCost = 0;
        for (size_t i = 0; i < recordCount; i++) {
            windowCost += recordCosts[i];
        }
        batchOffsets.assign(1, window.begin);
        batchFirstTargets.assign(1, 0);
        uint128 cost = 0;
        for (size_t i = window.begin; i < window.end; i++) {
            const uint128 recordEndCost = cost + recordCosts[i - window.begin];
            if (recordEndCost * maxBatches > windowCost * batchOffsets.size()) {
                const RepTargetsRecord &record = records[i];
                const DBKeyType *targets = chunkTargets[window.chunk].data() + record.targetOffset;
                const size_t repLength = seqDbr.getSeqLen(record.repFirstTarget.repKey % paddedDbSize);
                size_t targetRange = SIZE_MAX;
                for (unsigned int j = 0; j + 1 < record.targetCount && recordEndCost * maxBatches > windowCost * batchOffsets.size(); j++) {
                    cost += repLength * seqDbr.getSeqLenFromRange(targetRange, targets[j]);
                    if (cost * maxBatches >= windowCost * batchOffsets.size()) {
                        batchOffsets.push_back(i);
                        batchFirstTargets.push_back(j + 1);
                    }
                }
            }
            cost = recordEndCost;
            if (i + 1 < window.end && cost * maxBatches >= windowCost * batchOffsets.size()) {
                batchOffsets.push_back(i + 1);
                batchFirstTargets.push_back(0);
            }
        }
        batchOffsets.push_back(window.end);
        batchFirstTargets.push_back(0);
        const size_t batchCount = batchOffsets.size() - 1;
        repBatchCount += batchCount;
        batchDone.assign(batchCount, 0);
        batchPairs.assign(batchCount, std::vector<ClusterPair>());
        Timer repBatchTimer;
        size_t firstUnfinishedBatch = 0;
#pragma omp parallel reduction(+:alignedPairs, assignedRepSkips, unservedRepSkips, assignedTargetSkips, unservedTargetSkips, repBatchWorkerSeconds, repBatchLockWaitSeconds) reduction(max:repBatchLongestSeconds)
        {
            Sequence query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
            Sequence target(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
            BlockAligner blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                                     &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                                     -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid());
            // --align2clust-aligner-mode 1: the gapped alignments of the reps of a batch go through the SWIPE lanes together
            SwipeTargets swipe(par, scoring.subMat, scoring.evaluer, maxSeqLen);
            std::vector<ClusterPair> pairs;
#pragma omp for schedule(dynamic, 1)
            for (size_t batch = 0; batch < batchCount; batch++) {
                Timer batchTimer;
                {
                    std::unique_lock<std::mutex> changesLock(assignedChangesMutex, std::try_to_lock);
                    if (changesLock.owns_lock() && changesTimer.getTimediff() - batchReadSeconds >= BATCH_READ_SECONDS) {
                        readAssignedChanges(false);
                        batchReadSeconds = changesTimer.getTimediff();
                    }
                }
                pairs.clear();
                for (size_t i = batchOffsets[batch]; i < batchOffsets[batch + 1] + (batchFirstTargets[batch + 1] > 0); i++) {
                    const RepTargetsRecord &record = records[i];
                    const unsigned int endTarget = (i == batchOffsets[batch + 1]) ? batchFirstTargets[batch + 1] : record.targetCount;
                    const size_t repKey = record.repFirstTarget.repKey % paddedDbSize;
                    const char *repData = getWindowData(window, repKey);
                    const bool isAssignedRep = isAssigned(record.repFirstTarget.repKey);
                    if (isAssignedRep || repData == NULL) {
                        (isAssignedRep ? assignedRepSkips : unservedRepSkips)++;
                        continue;
                    }
                    query.mapSequence(repKey, repKey, repData, seqDbr.getSeqLen(repKey));
                    blockAligner.setQuery(&query);
                    bool blockAlignerInitialized = false;
                    swipe.startRep(record.repFirstTarget.repKey);
                    const DBKeyType *targets = chunkTargets[window.chunk].data() + record.targetOffset;
                    for (unsigned int j = (i == batchOffsets[batch]) ? batchFirstTargets[batch] : 0; j < endTarget && isAssigned(record.repFirstTarget.repKey) == false; j++) {
                        const DBKeyType targetKey = targets[j];
                        const char *targetData = getWindowData(window, targetKey);
                        const bool isAssignedTarget = isAssigned(targetKey);
                        if (isAssignedTarget || targetData == NULL) {
                            (isAssignedTarget ? assignedTargetSkips : unservedTargetSkips)++;
                            continue;
                        }
                        target.mapSequence(targetKey, targetKey, targetData, seqDbr.getSeqLen(targetKey));
                        alignedPairs++;
                        unsigned short diagonal = 0;
                        if (alignTarget(par, blockAligner, blockAlignerInitialized, query, target, scoring.scorePerColThreshold, diagonal, NULL, 0, NULL,
                                        swipeLanes ? &swipe : NULL)) {
                            const ClusterPair pair = { record.repFirstTarget.repKey, targetKey };
                            pairs.push_back(pair);
                        }
                    }
                }
                alignTargetsSwipe(par, swipe, isAssigned, pairs);
                swipe.clearTargets();
                // the lanes finish the pairs of the batch out of rep order
                SORT_SERIAL(pairs.begin(), pairs.end(), ClusterPair::compareByRep);
                // pairs go to the log in batch order, so the reps before the first unfinished batch are done
                bool readsChanges = false;
                size_t doneRepKey = 0;
                size_t doneBytes = 0;
                std::unique_lock<std::mutex> publishLock(publishMutex, std::defer_lock);
                Timer lockTimer;
                {
                    std::lock_guard<std::mutex> lock(acceptedMutex);
                    repBatchLockWaitSeconds += lockTimer.getTimediff();
                    batchPairs[batch].swap(pairs);
                    batchDone[batch] = 1;
                    if (memberDbr == NULL) {
                        for (; firstUnfinishedBatch < batchCount && batchDone[firstUnfinishedBatch]; firstUnfinishedBatch++) {
                            logAcceptedPairs(batchPairs[firstUnfinishedBatch].data(), batchPairs[firstUnfinishedBatch].size());
                            std::vector<ClusterPair>().swap(batchPairs[firstUnfinishedBatch]);
                        }
                        const time_t now = time(NULL);
                        // one thread publishes at a time, in the order of these values, while the others go on
                        if ((acceptedBytes - publishedBytes >= PUBLISH_BYTES || now - publishTime >= PUBLISH_SECONDS) && publishLock.try_lock()) {
                            doneRepKey = (firstUnfinishedBatch < batchCount) ? static_cast<size_t>(records[batchOffsets[firstUnfinishedBatch]].repFirstTarget.repKey) : nextRepKey;
                            doneBytes = acceptedBytes;
                            publishedBytes = acceptedBytes;
                            publishTime = now;
                            readsChanges = doneRepKey < orderKeyEnd;
                        }
                        if (readsChanges || now - changesTime >= READ_SECONDS) {
                            readsChanges = true;
                            changesTime = now;
                        }
                    }
                }
                if (publishLock.owns_lock()) {
                    publishAccepted(doneRepKey, doneBytes);
                    publishLock.unlock();
                }
                if (readsChanges) {
                    std::unique_lock<std::mutex> changesLock(assignedChangesMutex, std::try_to_lock);
                    if (changesLock.owns_lock()) {
                        readAssignedChanges(true);
                    }
                }
                const double batchSeconds = batchTimer.getTimediff();
                repBatchWorkerSeconds += batchSeconds;
                repBatchLongestSeconds = std::max(repBatchLongestSeconds, batchSeconds);
            }
        }
        repBatchWallSeconds += repBatchTimer.getTimediff();
        if (memberDbr != NULL) {
            Timer memberScanTimer;
            windowPairs.clear();
            for (size_t batch = 0; batch < batchCount; batch++) {
                windowPairs.insert(windowPairs.end(), batchPairs[batch].begin(), batchPairs[batch].end());
            }
            memberOffsets.resize(windowPairs.size() + 1);
            memberBytes.resize(windowPairs.size());
            const size_t memberPartBudget = memoryLimit / MEMBER_PART_MEMORY_DIVISOR;
            hasMemberGate.resize(windowPairs.size());
            memberEntries.resize(windowPairs.size());
#pragma omp parallel
            {
                ClusterMemberLookup::Cursor cursor;
#pragma omp for schedule(dynamic, 1024) reduction(+:skippedMemberGates)
                for (size_t i = 0; i < windowPairs.size(); i++) {
                    memberBytes[i] = 0;
                    memberOffsets[i + 1] = 0;
                    // the members of an assigned pair or of a gate that fails on a length are not read
                    hasMemberGate[i] = isAssigned(windowPairs[i].repKey) == false && isAssigned(windowPairs[i].memberKey) == false;
                    const int repLength = static_cast<int>(seqDbr.getSeqLen(windowPairs[i].repKey % paddedDbSize));
                    if (hasMemberGate[i]) {
                        memberEntries[i] = members->find(windowPairs[i].memberKey);
                        cursor.start(memberEntries[i]);
                    }
                    DBKeyType memberKey;
                    while (hasMemberGate[i] && members->next(cursor, memberKey)) {
                        hasMemberGate[i] = canCoverMember(par, repLength, static_cast<int>(memberDbr->getSeqLen(memberKey)));
                        const size_t entryBytes = std::max(memberDbr->getEntryLen(memberKey), pageSize);
                        if (entryBytes > memberPartBudget - memberBytes[i]) {
                            memberBytes[i] = memberPartBudget + 1;
                            break;
                        }
                        memberBytes[i] += entryBytes;
                        memberOffsets[i + 1]++;
                    }
                    if (hasMemberGate[i] == false) {
                        memberBytes[i] = 0;
                        memberOffsets[i + 1] = 0;
                        skippedMemberGates++;
                    }
                }
            }
            memberLookupSeconds += memberScanTimer.getTimediff();
            partOffsets.assign(1, 0);
            for (size_t i = 0, partBytes = 0; i < windowPairs.size(); i++) {
                if (i > partOffsets.back() && (partBytes > memberPartBudget || memberBytes[i] > memberPartBudget - partBytes)) {
                    partOffsets.push_back(i);
                    partBytes = 0;
                }
                partBytes += memberBytes[i];
            }
            partOffsets.push_back(windowPairs.size());
            memberOffsets[0] = 0;
            sumOffsets(memberOffsets.data(), windowPairs.size(), par.threads);
            membersPassed.assign(windowPairs.size(), 0);
            const auto isLargePart = [&](size_t part) {
                return partOffsets[part + 1] == partOffsets[part] + 1 && memberBytes[partOffsets[part]] > memberPartBudget;
            };
            // without the extensions a batch holds the gates of one rep
            const auto alignMemberGates = [&](MemberAligners &aligners, unsigned int threadIdx, const std::vector<size_t> &repGateOffsets, const std::vector<DBKeyType> &repKeys) {
                if (swipeLanes) {
                    alignMemberGatesSwipe(par, aligners.blockAligner, aligners.matcher, aligners.query, aligners.element, memberDbr, swMode, threadIdx, aligners.memberBuffers, repGateOffsets, repKeys, mapRep);
                } else if (aligners.memberBuffers.gateCount > 0) {
                    mapRep(aligners.query, aligners.blockAligner, repKeys[0]);
                    bool matcherInitialized = false;
                    alignClusterMembers(par, aligners.blockAligner, aligners.matcher, matcherInitialized, aligners.query, aligners.element, memberDbr, swMode, threadIdx, false, NULL, aligners.memberBuffers);
                }
            };
            // looks up the members of a part, the first at memberOffsets of its first pair, and starts reading them
            const auto prefetchPartMembers = [&](size_t part) {
                Timer memberTimer;
                const size_t first = partOffsets[part];
                nextPartMembers.resize(memberOffsets[partOffsets[part + 1]] - memberOffsets[first]);
#pragma omp parallel
                {
                    ClusterMemberLookup::Cursor cursor;
#pragma omp for schedule(dynamic, 1024)
                    for (size_t i = first; i < partOffsets[part + 1]; i++) {
                        if (hasMemberGate[i] == false) {
                            continue;
                        }
                        cursor.start(memberEntries[i]);
                        size_t member = memberOffsets[i] - memberOffsets[first];
                        DBKeyType memberKey;
                        while (members->next(cursor, memberKey)) {
                            nextPartMembers[member++] = memberKey;
                        }
                    }
                }
                nextPartKeys = nextPartMembers;
                SORT_PARALLEL(nextPartKeys.begin(), nextPartKeys.end());
                nextPartKeys.erase(std::unique(nextPartKeys.begin(), nextPartKeys.end()), nextPartKeys.end());
                memberDbr->prefetchData(nextPartKeys);
                memberLookupSeconds += memberTimer.getTimediff();
            };
            ClusterMemberLookup::Cursor partCursor;
            size_t prefetchedPart = SIZE_MAX;
            for (size_t part = 0; part + 1 < partOffsets.size(); part++) {
                const size_t first = partOffsets[part];
                const size_t last = partOffsets[part + 1];
                if (isLargePart(part)) {
                    const ClusterPair &pair = windowPairs[first];
                    if (isAssigned(pair.repKey) || isAssigned(pair.memberKey)) {
                        continue;
                    }
                    const size_t repKey = pair.repKey % paddedDbSize;
                    MemberAligners aligners(par, scoring, maxMemberSeqLen, swipeLanes);
                    const std::vector<size_t> repGateOffsets = {0, 1};
                    const std::vector<DBKeyType> repKeys(1, repKey);
                    partCursor.start(memberEntries[first]);
                    DBKeyType memberKey;
                    bool hasMember = members->next(partCursor, memberKey);
                    bool passed = true;
                    while (passed && hasMember) {
                        partMembers.clear();
                        for (size_t partBytes = 0; hasMember; hasMember = members->next(partCursor, memberKey)) {
                            const size_t entryBytes = std::max(memberDbr->getEntryLen(memberKey), pageSize);
                            if (entryBytes > memberPartBudget) {
                                Debug(Debug::ERROR) << "Cluster member sequence " << memberKey << " exceeds the " << memberPartBudget << " byte part budget\n";
                                EXIT(EXIT_FAILURE);
                            }
                            if (partMembers.empty() == false && entryBytes > memberPartBudget - partBytes) {
                                break;
                            }
                            partMembers.push_back(memberKey);
                            partBytes += entryBytes;
                        }
                        partKeys = partMembers;
                        SORT_PARALLEL(partKeys.begin(), partKeys.end());
                        partKeys.erase(std::unique(partKeys.begin(), partKeys.end()), partKeys.end());
                        memberDbr->loadData(partKeys);
                        aligners.memberBuffers.gateCount = 0;
                        MemberGate &gate = aligners.memberBuffers.addGate(first, 0);
                        gate.memberKeys = partMembers;
                        alignMemberGates(aligners, 0, repGateOffsets, repKeys);
                        passed = gate.live;
                    }
                    membersPassed[first] = passed;
                    continue;
                }
                if (prefetchedPart != part) {
                    prefetchPartMembers(part);
                }
                partMembers.swap(nextPartMembers);
                partKeys.swap(nextPartKeys);
                Timer memberTimer;
                memberDbr->loadData(partKeys);
                memberReadSeconds += memberTimer.getTimediff();
                // the next part is read while the gates of this one run
                if (part + 2 < partOffsets.size() && isLargePart(part + 1) == false) {
                    prefetchPartMembers(part + 1);
                    prefetchedPart = part + 1;
                }
                memberTimer.reset();
                // the pairs of a rep are next to each other, they share the query and the rounds of their member gates
                repGroups.clear();
                for (size_t i = first; i < last; i++) {
                    if (i == first || windowPairs[i].repKey != windowPairs[i - 1].repKey) {
                        repGroups.push_back(i);
                    }
                }
                repGroups.push_back(last);
                const size_t groupCount = repGroups.size() - 1;
                // with the extensions reps go in batches that fill the lanes and give a thread 4 of them
                const size_t batchMemberLimit = std::min(MEMBER_BATCH_SIZE, std::max(partMembers.size() / (par.threads * 4), MEMBER_BATCH_SIZE / 16));
                batchGroupOffsets.assign(1, 0);
                for (size_t group = 0, batchMembers = 0; group < groupCount; group++) {
                    batchMembers += memberOffsets[repGroups[group + 1]] - memberOffsets[repGroups[group]];
                    if (swipeLanes == false || batchMembers >= batchMemberLimit || group + 1 == groupCount) {
                        batchGroupOffsets.push_back(group + 1);
                        batchMembers = 0;
                    }
                }
                const size_t memberBatchCount = batchGroupOffsets.size() - 1;
#pragma omp parallel
                {
                    unsigned int thread_idx = 0;
#ifdef OPENMP
                    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                    MemberAligners aligners(par, scoring, maxMemberSeqLen, swipeLanes);
                    std::vector<size_t> repGateOffsets;
                    std::vector<DBKeyType> repKeys;
#pragma omp for schedule(dynamic, 1)
                    for (size_t batch = 0; batch < memberBatchCount; batch++) {
                        aligners.memberBuffers.gateCount = 0;
                        repGateOffsets.assign(1, 0);
                        repKeys.clear();
                        for (size_t group = batchGroupOffsets[batch]; group < batchGroupOffsets[batch + 1]; group++) {
                            for (size_t i = repGroups[group]; i < repGroups[group + 1]; i++) {
                                const ClusterPair &pair = windowPairs[i];
                                if (hasMemberGate[i] == false || isAssigned(pair.repKey) || isAssigned(pair.memberKey)) {
                                    continue;
                                }
                                MemberGate &gate = aligners.memberBuffers.addGate(i, 0);
                                for (size_t member = memberOffsets[i]; member < memberOffsets[i + 1]; member++) {
                                    gate.memberKeys.push_back(partMembers[member - memberOffsets[first]]);
                                }
                            }
                            if (aligners.memberBuffers.gateCount > repGateOffsets.back()) {
                                repGateOffsets.push_back(aligners.memberBuffers.gateCount);
                                repKeys.push_back(windowPairs[repGroups[group]].repKey % paddedDbSize);
                            }
                        }
                        alignMemberGates(aligners, thread_idx, repGateOffsets, repKeys);
                        for (size_t g = 0; g < aligners.memberBuffers.gateCount; g++) {
                            membersPassed[aligners.memberBuffers.gates[g].targetId] = aligners.memberBuffers.gates[g].live;
                        }
                    }
                }
                memberGateSeconds += memberTimer.getTimediff();
            }
            size_t acceptedCount = 0;
            for (size_t i = 0; i < windowPairs.size(); i++) {
                if (membersPassed[i]) {
                    windowPairs[acceptedCount++] = windowPairs[i];
                }
            }
            logAcceptedPairs(windowPairs.data(), acceptedCount);
            publishAccepted(nextRepKey, acceptedBytes);
            publishedBytes = acceptedBytes;
            publishTime = time(NULL);
            if (nextRepKey < orderKeyEnd) {
                std::lock_guard<std::mutex> changesLock(assignedChangesMutex);
                readAssignedChanges(true);
            }
        }
        alignedCost += static_cast<double>(windowCost);
        alignSeconds += alignTimer.getTimediff();
        alignedCostPerSecond.store(alignedCost / std::max(alignSeconds, 1e-3));
        windowCount++;
        windowKeys += window.keys.size();
        waitForWindow(nextWindow, nextRepKey);
        std::swap(window, nextWindow);
    }
    chunkLoader.join();
    MMseqsNode::addPhaseValue(phaseValues, "Sequence windows", SSTR(windowCount) + " (" + SSTR(windowKeys) + " keys, "
                             + SSTR(windowKeptKeys) + " keys and " + SSTR(windowKeptBytes) + " bytes reused)");
    publishAccepted(orderKeyEnd, acceptedBytes);
    if (fclose(acceptedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << acceptedFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    // the service thread serves until every aligning node has requested its last window
    if (hasService) {
        serviceThread.join();
        MMseqsNode::addPhaseValue(phaseValues, "Served sequence flushes", SSTR(servedFlushCount) + " (largest " + SSTR(largestServedFlushBytes) + " bytes)");
        MMseqsNode::addPhaseValue(phaseValues, "Received sequence slices", SSTR(receivedSliceCount));
        MMseqsNode::addPhaseValue(phaseValues, "Sequence exchange bytes", SSTR(servedBytes) + " served, " + SSTR(receivedSliceBytes) + " received, " + SSTR(requestFlushCount) + " requests");
        MMseqsNode::addPhaseSeconds(phaseValues, "service: serving the slices", servingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: writing the served flushes", writingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: waiting for the flush writer", flushWriterWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: reading the sequence slices", readingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: unpacking the sequence slices", unpackingSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "service: polling", pollingSeconds);
    }
    delete[] window.data;
    delete[] nextWindow.data;
    for (size_t page = 0; page < assignedPages.size(); page++) {
        delete[] assignedPages[page];
    }
    MMseqsNode::addPhaseValue(phaseValues, "Block read bytes", SSTR(seqDbr.getBlockReadBytes()));
    MMseqsNode::addPhaseValue(phaseValues, "Key read bytes", SSTR(seqDbr.getKeyReadBytes()));
    MMseqsNode::addPhaseValue(phaseValues, "Assigned change bytes read", SSTR(changeReadBytes));
    MMseqsNode::addPhaseValue(phaseValues, "Assigned packed sequence bytes not served", SSTR(unservedSequenceBytes));
    // the master still finds the nodes of keys in the blocks of seqDbr
    seqDbr.releaseDataPages();
    if (hasService) {
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the master", waitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the sequences", sequenceWaitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the prefilter", prefilterWaitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Rep alignment batches", SSTR(repBatchCount));
        MMseqsNode::addPhaseSeconds(phaseValues, "cumulative rep batch worker time", repBatchWorkerSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "longest rep batch", repBatchLongestSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "all rep batches", repBatchWallSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "looking up the members", memberLookupSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "reading the members", memberReadSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "member gates", memberGateSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Member gates skipped before reading", SSTR(skippedMemberGates));
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the accepted log mutex", repBatchLockWaitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Aligned pairs", SSTR(alignedPairs));
        MMseqsNode::addPhaseValue(phaseValues, "Skipped reps", SSTR(assignedRepSkips) + " assigned, " + SSTR(unservedRepSkips) + " not served");
        MMseqsNode::addPhaseValue(phaseValues, "Skipped targets", SSTR(assignedTargetSkips) + " assigned, " + SSTR(unservedTargetSkips) + " not served");
        MMseqsNode::addPhaseValue(phaseValues, "Claim seconds", SSTR(CLAIM_SECONDS) + " (" + SSTR(alignedCostPerSecond.load()) + " cells per second, " + SSTR(alignSeconds) + " s aligning)");
    }
}

// a rep of a clustering step, its accepted targets [begin, end) in logWords and the node of its log
struct AcceptedRep {
    size_t repKey;
    size_t begin;
    size_t end;
    unsigned int node;
};

// the merge gives the reps of node from firstNodeRep at once, at firstMergedRep of the merged reps
struct MergedReps {
    size_t firstNodeRep;
    size_t firstMergedRep;
    unsigned int node;
};

// the taken bits of a log in whole words; carry holds the bits of the last partial word
struct TakenLog {
    FILE *file;
    size_t bytes;
    uint64_t carry;
    unsigned int carryBits;
    char *buffer;
};

// --createdb-mode 3: the master clusters the accepted pairs in order key order; returns the member count
static size_t clusterAcceptedPairs(int threads, size_t dbSize, size_t memoryLimit, const std::vector<size_t> &groupRangeKeyOffsets, const MMseqsNode::WorkQueue &groupRangeQueue, const DBReader<DBKeyType> &seqDbr,
                                   const std::vector<unsigned int> &pageNodes, int mode, size_t paddedDbSize, size_t prefSizeBits,
                                   const std::string &accepted, const std::string &assignedChanges, const std::string &taken, const std::string &assignedBits, std::string &phaseValues) {
    const size_t assignedWords = (dbSize + 63) / 64;
    const size_t orderKeyEnd = (prefSizeBits - 1) * paddedDbSize + dbSize;
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
    std::vector<size_t> nodeRepKeys(MMseqsNode::count, 0);
    std::vector<size_t> nodeLogBytes(MMseqsNode::count, 0);
    std::vector<size_t> nodeReadBytes(MMseqsNode::count, 0);
    std::vector<TakenLog> nodeTaken(MMseqsNode::count);
    std::vector<std::vector<uint64_t> > nodeTakenBits(MMseqsNode::count);
    std::vector<size_t> nodeTakenCounts(MMseqsNode::count, 0);
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        const TakenLog takenLog = { FileUtil::openAndDelete((taken + "." + SSTR(node)).c_str(), "w"), 0, 0, 0, new(std::nothrow) char[TAKEN_BUFFER_BYTES] };
        Util::checkAllocation(takenLog.buffer, "Cannot allocate the buffer of " + taken);
        setvbuf(takenLog.file, takenLog.buffer, _IOFBF, TAKEN_BUFFER_BYTES);
        nodeTaken[node] = takenLog;
    }
    FILE *changesHandle = FileUtil::openAndDelete(assignedChanges.c_str(), "w");
    size_t changesBytes = 0;
    const size_t stepWords = getStepWords(memoryLimit);
    const size_t nodeStepWords = std::max<size_t>(1, stepWords / MMseqsNode::count);
    const size_t partBytes = std::max<size_t>(1, std::min(stepWords, memoryLimit / LOG_PART_MEMORY_DIVISOR / sizeof(uint64_t)) / MMseqsNode::count) * sizeof(uint64_t);
    std::vector<uint64_t> logWords;
    // per node: the words read past nodeReadBytes, starting at a rep; between steps only those not clustered yet
    std::vector<std::vector<uint64_t> > nodeTails(MMseqsNode::count);
    std::vector<size_t> nodeWordOffsets(MMseqsNode::count, 0);
    std::vector<std::vector<size_t> > nodeRepStarts(MMseqsNode::count);
    // per node: no unread word of its log has a smaller rep
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
    std::vector<uint64_t> changes;
    std::vector<uint64_t> writtenChanges;
    std::thread changesWriter;
    std::atomic<bool> isWritingChanges(false);
    std::vector<size_t> threadKeyOffsets(threads + 1, 0);
    const auto writeChanges = [&](size_t clusteredKey) {
        if (fwrite(writtenChanges.data(), sizeof(uint64_t), writtenChanges.size(), changesHandle) != writtenChanges.size()) {
            Debug(Debug::ERROR) << "Cannot write " << assignedChanges << "\n";
            EXIT(EXIT_FAILURE);
        }
        changesBytes += writtenChanges.size() * sizeof(uint64_t);
        MMseqsNode::publishLog(changesHandle, assignedChanges, clusteredKey, changesBytes);
        isWritingChanges = false;
    };
    std::vector<std::vector<size_t> > sliceNodeOffsets(threads, std::vector<size_t>(MMseqsNode::count));
    size_t memberCount = 0;
    // --cluster-mode 0: the open sets in a heap as in clusterThreadFuncSetcover, the sets of two as pairs
    std::vector<ClusterPair> pairSets;
    SetCoverPairSpool pairSpool(pairSets, taken + "_pairsets", stepWords, memoryLimit, assigned);
    SetCoverSmallSetSpool smallSpool(taken + "_sets", memoryLimit, assigned);
    SetCoverLargeSetSpool largeSpool(taken + "_large", memoryLimit);
    SetCoverSizeClasses sizeClasses;
    // the node that serves a key, from its page unless the page holds the keys of more than one node
    const auto getServingNode = [&](DBKeyType key) {
        return (pageNodes[key / ASSIGNED_PAGE_KEYS] != UINT_MAX) ? pageNodes[key / ASSIGNED_PAGE_KEYS] : getKeyNode(seqDbr, key);
    };
    // the nodes of the group ranges that the master knows to be claimed, the first ones, and the keys of the ranges known since the last step that their nodes get
    std::vector<unsigned int> groupRangeNodes;
    std::vector<std::vector<DBKeyType> > nodeGroupRangeKeys(MMseqsNode::count);
    const auto takePair = [&](DBKeyType repKey, DBKeyType memberKey) {
        const ClusterPair pair = { repKey, memberKey };
        TakenLog &nodeFile = nodeTaken[getServingNode(repKey)];
        if (fwrite(&pair, sizeof(ClusterPair), 1, nodeFile.file) != 1) {
            Debug(Debug::ERROR) << "Cannot write " << taken << "\n";
            EXIT(EXIT_FAILURE);
        }
        nodeFile.bytes += sizeof(ClusterPair);
        memberCount++;
    };
    const auto getRepKey = [](uint64_t word) {
        return static_cast<size_t>(word & ~ACCEPTED_REP_BIT);
    };
    size_t clusteredPairs = 0;
    size_t logReadBytes = 0;
    double waitSeconds = 0;
    double mergeSeconds = 0;
    double decisionSeconds = 0;
    double placingSeconds = 0;
    double logReadSeconds = 0;
    double tailEraseSeconds = 0;
    size_t tailMovedWords = 0;
    size_t freePairs = 0;
    size_t assignedRepPairs = 0;
    size_t assignedMemberPairs = 0;
    size_t sharingRepPairs = 0;
    double takenWriteSeconds = 0;
    size_t peakChangeBytes = 0;
    size_t stepCount = 0;
    size_t peakSetCoverStateBytes = 0;
    Timer clusterTimer;
    unsigned int polls = 0;
    MMseqsNode::WaitWarning waitWarning;
    size_t orderedClaims = 0;
    for (size_t clusteredKey = 0; clusteredKey < orderKeyEnd;) {
        // the steps decided while the writer is busy are written and published together, waiting for it once they reach stepWords
        if (changes.empty() == false && (isWritingChanges == false || changes.size() >= stepWords)) {
            if (changesWriter.joinable()) {
                changesWriter.join();
            }
            peakChangeBytes = std::max(peakChangeBytes, changes.size() * sizeof(uint64_t));
            changes.swap(writtenChanges);
            changes.clear();
            isWritingChanges = true;
            changesWriter = std::thread(writeChanges, clusteredKey);
        }
        Timer pollTimer;
        bool hasNewProgress = false;
#pragma omp parallel for schedule(dynamic, 1) reduction(||:hasNewProgress) reduction(max:orderedClaims)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const std::string log = accepted + "." + SSTR(node);
            size_t repKey;
            size_t logBytes;
            size_t nodeClaims;
            if ((nodeHasProgress[node] && nodeRepKeys[node] == orderKeyEnd) || MMseqsNode::readLogProgress(log, repKey, logBytes, &nodeClaims) == false) {
                continue;
            }
            orderedClaims = std::max(orderedClaims, nodeClaims);
            if (nodeHasProgress[node] && repKey <= nodeRepKeys[node] && logBytes <= nodeLogBytes[node]) {
                continue;
            }
            hasNewProgress = true;
            nodeHasProgress[node] = 1;
            nodeRepKeys[node] = repKey;
            nodeLogBytes[node] = logBytes;
        }
        size_t doneRepKey = orderKeyEnd;
        size_t slowestNode = 0;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const size_t nodeRepKey = nodeHasProgress[node] ? nodeRepKeys[node] : 0;
            if (nodeRepKey < doneRepKey) {
                doneRepKey = nodeRepKey;
                slowestNode = node;
            }
        }
        if (doneRepKey <= clusteredKey) {
            if (hasNewProgress == false) {
                MMseqsNode::sleepBeforePoll(polls++);
            }
            waitWarning.check(accepted + "." + SSTR(slowestNode) + ".progress");
            waitSeconds += pollTimer.getTimediff();
            continue;
        }
        polls = 0;
        Timer stepTimer;
        // a node that claimed a group range gets the keys of it assigned before this step, then those that its steps assign; the claim files below the logged cursor are read at once
        const size_t firstNewRange = groupRangeNodes.size();
        std::vector<unsigned int> claimNodes(std::max(firstNewRange, std::min(orderedClaims, groupRangeKeyOffsets.size() - 1)) - firstNewRange);
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < claimNodes.size(); i++) {
            claimNodes[i] = groupRangeQueue.getNode(firstNewRange + i);
        }
        for (size_t i = 0; i < claimNodes.size(); i++) {
            const unsigned int node = claimNodes[i];
            if (node >= MMseqsNode::count) {
                break;
            }
            const size_t keyEnd = groupRangeKeyOffsets[groupRangeNodes.size() + 1];
            for (size_t key = groupRangeKeyOffsets[groupRangeNodes.size()]; key < keyEnd; key = (key | 63) + 1) {
                for (uint64_t bits = assigned[key / 64] >> (key % 64); bits != 0 && key + __builtin_ctzll(bits) < keyEnd; bits &= bits - 1) {
                    if (getServingNode(key + __builtin_ctzll(bits)) != node) {
                        nodeGroupRangeKeys[node].push_back(key + __builtin_ctzll(bits));
                    }
                }
            }
            groupRangeNodes.push_back(node);
        }

        // read every log up to doneRepKey after its tail, the logs at once; past nodeStepWords a log ends before its next rep, but the rep of clusteredKey is read whole
#pragma omp parallel for schedule(dynamic, 1) reduction(+:logReadBytes, logReadSeconds)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            std::vector<uint64_t> &nodeWords = nodeTails[node];
            std::vector<size_t> &repStarts = nodeRepStarts[node];
            repStarts.clear();
            size_t word = 0;
            if (nodeNextRepKeys[node] < doneRepKey) {
                nodeNextRepKeys[node] = nodeRepKeys[node];
                const std::string log = accepted + "." + SSTR(node);
                size_t from = nodeReadBytes[node] + nodeWords.size() * sizeof(uint64_t);
                bool isCut = false;
                while (true) {
                    for (; word < nodeWords.size(); word++) {
                        if ((nodeWords[word] & ACCEPTED_REP_BIT) == 0) {
                            continue;
                        }
                        const size_t repKey = getRepKey(nodeWords[word]);
                        if (repKey >= doneRepKey || (word >= nodeStepWords && repKey > clusteredKey)) {
                            nodeNextRepKeys[node] = repKey;
                            isCut = true;
                            break;
                        }
                        repStarts.push_back(word);
                    }
                    if (isCut || from >= nodeLogBytes[node]) {
                        break;
                    }
                    const size_t to = std::min(nodeLogBytes[node], from + partBytes);
                    nodeWords.resize(word + (to - from) / sizeof(uint64_t));
                    Timer readTimer;
                    MMseqsNode::readLog(log, from, to, reinterpret_cast<char *>(nodeWords.data() + word));
                    logReadSeconds += readTimer.getTimediff();
                    logReadBytes += to - from;
                    from = to;
                }
            }
            repStarts.push_back(word);
        }
        // the step ends at the first cut rep, the same as reading the logs one after another with the smallest cut so far
        size_t stepRepKey = doneRepKey;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            stepRepKey = std::min(stepRepKey, nodeNextRepKeys[node]);
        }
        size_t stepWordCount = 0;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            std::vector<size_t> &repStarts = nodeRepStarts[node];
            const std::vector<uint64_t> &nodeWords = nodeTails[node];
            const size_t clusteredReps = std::lower_bound(repStarts.begin(), repStarts.end() - 1, stepRepKey,
                [&](size_t word, size_t repKey) { return getRepKey(nodeWords[word]) < repKey; }) - repStarts.begin();
            if (clusteredReps + 1 < repStarts.size()) {
                nodeNextRepKeys[node] = getRepKey(nodeWords[repStarts[clusteredReps]]);
            }
            nodeTakenCounts[node] = repStarts[clusteredReps];
            repStarts.resize(clusteredReps);
            repStarts.push_back(nodeTakenCounts[node]);
            nodeWordOffsets[node] = stepWordCount;
            stepWordCount += nodeTakenCounts[node];
        }
        logWords.resize(stepWordCount);
#pragma omp parallel for schedule(dynamic, 1) reduction(+:tailEraseSeconds, tailMovedWords)
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            std::vector<uint64_t> &nodeWords = nodeTails[node];
            std::copy(nodeWords.begin(), nodeWords.begin() + nodeTakenCounts[node], logWords.begin() + nodeWordOffsets[node]);
            Timer eraseTimer;
            tailMovedWords += nodeWords.size() - nodeTakenCounts[node];
            nodeWords.erase(nodeWords.begin(), nodeWords.begin() + nodeTakenCounts[node]);
            tailEraseSeconds += eraseTimer.getTimediff();
            std::vector<size_t> &repStarts = nodeRepStarts[node];
            for (size_t rep = 0; rep < repStarts.size(); rep++) {
                repStarts[rep] += nodeWordOffsets[node];
            }
            nodeReadBytes[node] += nodeTakenCounts[node] * sizeof(uint64_t);
            nodeTakenBits[node].assign((nodeTakenCounts[node] + 63) / 64, 0);
        }
        // merge the reps of the logs in order key order, a node giving its reps up to the next rep of another node at once
        std::priority_queue<std::pair<size_t, unsigned int>, std::vector<std::pair<size_t, unsigned int> >, std::greater<std::pair<size_t, unsigned int> > > nextReps;
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            nodeMergedReps[node] = 0;
            if (nodeRepStarts[node].size() > 1) {
                nextReps.push(std::make_pair(getRepKey(logWords[nodeRepStarts[node][0]]), node));
            }
        }
        mergedReps.clear();
        size_t repCount = 0;
        while (nextReps.empty() == false) {
            const unsigned int node = nextReps.top().second;
            nextReps.pop();
            const std::vector<size_t> &repStarts = nodeRepStarts[node];
            const size_t endKey = nextReps.empty() ? stepRepKey : nextReps.top().first;
            const size_t endRep = std::lower_bound(repStarts.begin() + nodeMergedReps[node] + 1, repStarts.end() - 1, endKey,
                [&](size_t word, size_t repKey) { return getRepKey(logWords[word]) < repKey; }) - repStarts.begin();
            const MergedReps nodeReps = { nodeMergedReps[node], repCount, node };
            mergedReps.push_back(nodeReps);
            repCount += endRep - nodeMergedReps[node];
            clusteredPairs += repStarts[endRep] - repStarts[nodeMergedReps[node]] - (endRep - nodeMergedReps[node]);
            nodeMergedReps[node] = endRep;
            if (endRep + 1 < repStarts.size()) {
                nextReps.push(std::make_pair(getRepKey(logWords[repStarts[endRep]]), node));
            }
        }
        reps.resize(repCount);
        const size_t repChunk = 1024;
#pragma omp parallel for schedule(static)
        for (size_t first = 0; first < repCount; first += repChunk) {
            const size_t end = first + std::min(repChunk, repCount - first);
            size_t merged = std::upper_bound(mergedReps.begin(), mergedReps.end(), first,
                [](size_t rep, const MergedReps &next) { return rep < next.firstMergedRep; }) - mergedReps.begin() - 1;
            for (size_t rep = first; rep < end;) {
                const MergedReps &nodeReps = mergedReps[merged];
                const size_t segmentEnd = std::min(end, (merged + 1 < mergedReps.size()) ? mergedReps[merged + 1].firstMergedRep : repCount);
                const std::vector<size_t> &repStarts = nodeRepStarts[nodeReps.node];
                for (; rep < segmentEnd; rep++) {
                    const size_t nodeRep = nodeReps.firstNodeRep + rep - nodeReps.firstMergedRep;
                    const AcceptedRep acceptedRep = { getRepKey(logWords[repStarts[nodeRep]]), repStarts[nodeRep] + 1, repStarts[nodeRep + 1], nodeReps.node };
                    reps[rep] = acceptedRep;
                }
                merged++;
            }
        }
        mergeSeconds += stepTimer.getTimediff();
        stepTimer.reset();

        // a pair with an assigned rep or member stays out, as bits are never cleared
        pairIsFree.resize(logWords.size());
        repKeyOffsets.assign(repCount + 1, 0);
#pragma omp parallel for schedule(dynamic, 1024) reduction(+:freePairs, assignedRepPairs, assignedMemberPairs)
        for (size_t rep = 0; rep < repCount; rep++) {
            const bool repIsFree = isAssigned(reps[rep].repKey % paddedDbSize) == false;
            size_t freeCount = 0;
            for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                pairIsFree[i] = repIsFree && isAssigned(logWords[i]) == false;
                freeCount += pairIsFree[i];
            }
            repKeyOffsets[rep + 1] = (freeCount > 0) ? freeCount + 1 : 0;
            freePairs += freeCount;
            assignedRepPairs += repIsFree ? 0 : reps[rep].end - reps[rep].begin;
            assignedMemberPairs += repIsFree ? reps[rep].end - reps[rep].begin - freeCount : 0;
        }
        if (repCount > 16384 && threads > 1) {
            sumOffsets(repKeyOffsets.data(), repCount, threads);
        } else {
            for (size_t rep = 0; rep < repCount; rep++) {
                repKeyOffsets[rep + 1] += repKeyOffsets[rep];
            }
        }
        if (mode == Parameters::SET_COVER) {
            // a set waits until no later rep can have a larger one, the sets of two as pairs
            size_t setPairs = 0;
            for (size_t rep = 0; rep < repCount; rep++) {
                const size_t setSize = repKeyOffsets[rep + 1] - repKeyOffsets[rep];
                setPairs += setSize == 2;
            }
            const bool bufferedPairs = pairSpool.prepareBatch(setPairs);
            const size_t pairOffset = pairSets.size();
            if (bufferedPairs) {
                pairSets.resize(pairOffset + setPairs);
            }
            size_t pairIndex = pairOffset;
            for (size_t rep = 0; rep < repCount; rep++) {
                if (repKeyOffsets[rep + 1] - repKeyOffsets[rep] == 2) {
                    for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                        if (pairIsFree[i]) {
                            const ClusterPair pairSet = { reps[rep].repKey % paddedDbSize, static_cast<DBKeyType>(logWords[i]) };
                            if (bufferedPairs) {
                                pairSets[pairIndex++] = pairSet;
                            } else {
                                pairSpool.push(pairSet);
                            }
                        }
                    }
                }
            }
            DBKeyType smallMembers[SetCoverSmallSetSpool::MAX_SIZE];
            for (size_t rep = 0; rep < repCount; rep++) {
                const size_t setSize = repKeyOffsets[rep + 1] - repKeyOffsets[rep];
                if (setSize < 3 || setSize > SetCoverSmallSetSpool::MAX_SIZE) {
                    continue;
                }
                smallMembers[0] = reps[rep].repKey % paddedDbSize;
                size_t member = 1;
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (pairIsFree[i]) {
                        smallMembers[member++] = logWords[i];
                    }
                }
                smallSpool.append(setSize, smallMembers);
                sizeClasses.add(setSize);
            }
            largeSpool.appendStep(reps, repKeyOffsets, paddedDbSize, [&](size_t i) { return pairIsFree[i]; },
                                  [&](size_t i) { return logWords[i]; }, sizeClasses);
            // the reps from stepRepKey on have fewer than 2^(prefSizeBit + 1) targets, so no larger set can come
            const size_t maxSetSize = (stepRepKey < orderKeyEnd) ? static_cast<size_t>(2) << (prefSizeBits - 1 - stepRepKey / paddedDbSize) : 0;
            largeSpool.drain(maxSetSize, isAssigned, [&](DBKeyType repKey, DBKeyType memberKey) {
                setAssignedSerial(memberKey, threadAssignedKeys[0]);
                if (memberKey != repKey) {
                    takePair(repKey, memberKey);
                }
            }, pairSpool, smallSpool, sizeClasses);
            smallSpool.drain(maxSetSize, [&](size_t setSize, const DBKeyType *members) {
                sizeClasses.remove(setSize);
                if (isAssigned(members[0])) {
                    return;
                }
                DBKeyType validMembers[SetCoverSmallSetSpool::MAX_SIZE];
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
                        pairSpool.push(pairSet);
                    } else {
                        smallSpool.append(validCount, validMembers);
                        sizeClasses.add(validCount);
                    }
                    return;
                }
                for (size_t member = 0; member < setSize; member++) {
                    setAssignedSerial(members[member], threadAssignedKeys[0]);
                    if (members[member] != members[0]) {
                        takePair(members[0], members[member]);
                    }
                }
            });
            peakSetCoverStateBytes = std::max(peakSetCoverStateBytes, largeSpool.peakMemory() + pairSets.capacity() * sizeof(ClusterPair));
            // the sets of two are the smallest, so the last step takes them after the heap, the larger rep first
            if (stepRepKey == orderKeyEnd) {
                pairSpool.finish([&](const ClusterPair &pair) {
                    if (isAssigned(pair.repKey) == false && isAssigned(pair.memberKey) == false) {
                        setAssignedSerial(pair.repKey, threadAssignedKeys[0]);
                        setAssignedSerial(pair.memberKey, threadAssignedKeys[0]);
                        takePair(pair.repKey, pair.memberKey);
                    }
                });
                std::vector<ClusterPair>().swap(pairSets);
            }
        } else {
            // keyReps: the first rep of every key of the step in a table of at least twice the keys; every later rep of a key shares it
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
                            __sync_fetch_and_or(&nodeTakenBits[reps[rep].node][bit / 64], 1ULL << (bit % 64));
                            memberCount++;
                        }
                    }
                }
            }
            // the sharing reps take in key order; no rep that does not share holds a key of theirs
            sharingReps.clear();
            for (size_t rep = 0; rep < repCount; rep++) {
                if (repKeyOffsets[rep + 1] != repKeyOffsets[rep] && repSharesKey[rep] != 0) {
                    sharingReps.push_back(rep);
                    sharingRepPairs += reps[rep].end - reps[rep].begin;
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
                        nodeTakenBits[reps[rep].node][bit / 64] |= 1ULL << (bit % 64);
                        memberCount++;
                    }
                }
            }
            Timer takenTimer;
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                TakenLog &nodeFile = nodeTaken[node];
                words.clear();
                for (size_t i = 0; i < nodeTakenCounts[node]; i += 64) {
                    const uint64_t bits = nodeTakenBits[node][i / 64];
                    const unsigned int wordBits = std::min<size_t>(64, nodeTakenCounts[node] - i);
                    nodeFile.carry |= bits << nodeFile.carryBits;
                    if (nodeFile.carryBits + wordBits >= 64) {
                        words.push_back(nodeFile.carry);
                        nodeFile.carry = (nodeFile.carryBits > 0) ? bits >> (64 - nodeFile.carryBits) : 0;
                        nodeFile.carryBits = nodeFile.carryBits + wordBits - 64;
                    } else {
                        nodeFile.carryBits += wordBits;
                    }
                }
                if (words.empty() == false && fwrite(words.data(), sizeof(uint64_t), words.size(), nodeFile.file) != words.size()) {
                    Debug(Debug::ERROR) << "Cannot write " << taken << "." << node << "\n";
                    EXIT(EXIT_FAILURE);
                }
                nodeFile.bytes += words.size() * sizeof(uint64_t);
            }
            takenWriteSeconds += takenTimer.getTimediff();
        }
        decisionSeconds += stepTimer.getTimediff();
        stepCount++;
        stepTimer.reset();
        clusteredKey = stepRepKey;

        // the step keys by the node that serves them and the node of their group range: every thread counts the nodes of a slice of the keys of all threads, then places it after the slices before it
        for (int thread = 0; thread < threads; thread++) {
            threadKeyOffsets[thread + 1] = threadKeyOffsets[thread] + threadAssignedKeys[thread].size();
        }
        const size_t stepKeys = threadKeyOffsets[threads];
        const size_t headerWords = getChangeHeaderWords();
        const size_t stepStart = changes.size();
        for (int pass = 0; pass < 2; pass++) {
#pragma omp parallel for schedule(static)
            for (int slice = 0; slice < threads; slice++) {
                std::vector<size_t> &nodeOffsets = sliceNodeOffsets[slice];
                if (pass == 0) {
                    nodeOffsets.assign(MMseqsNode::count, 0);
                }
                const size_t sliceEnd = stepKeys * (slice + 1) / threads;
                size_t thread = std::upper_bound(threadKeyOffsets.begin(), threadKeyOffsets.end(), stepKeys * slice / threads) - threadKeyOffsets.begin() - 1;
                for (size_t i = stepKeys * slice / threads; i < sliceEnd; i++) {
                    while (i == threadKeyOffsets[thread + 1]) {
                        thread++;
                    }
                    const DBKeyType key = threadAssignedKeys[thread][i - threadKeyOffsets[thread]];
                    // the node of its group range gets it too once that is known
                    const size_t keyNode = getServingNode(key);
                    const size_t groupRange = std::upper_bound(groupRangeKeyOffsets.begin(), groupRangeKeyOffsets.end(), key) - groupRangeKeyOffsets.begin() - 1;
                    const size_t groupNode = (groupRange < groupRangeNodes.size()) ? groupRangeNodes[groupRange] : keyNode;
                    const size_t keyPosition = nodeOffsets[keyNode]++;
                    const size_t groupPosition = (groupNode != keyNode) ? nodeOffsets[groupNode]++ : keyPosition;
                    if (pass == 1) {
                        changes[keyPosition] = key;
                        changes[groupPosition] = key;
                    }
                }
            }
            if (pass == 0) {
                changes.resize(stepStart + headerWords);
                changes[stepStart] = clusteredKey;
                size_t nodeKeyOffset = 0;
                for (size_t node = 0; node < MMseqsNode::count; node++) {
                    changes[stepStart + 1 + node] = nodeReadBytes[node];
                    changes[stepStart + 1 + MMseqsNode::count + node] = nodeKeyOffset;
                    for (int slice = 0; slice < threads; slice++) {
                        const size_t count = sliceNodeOffsets[slice][node];
                        sliceNodeOffsets[slice][node] = stepStart + headerWords + nodeKeyOffset;
                        nodeKeyOffset += count;
                    }
                    nodeKeyOffset += nodeGroupRangeKeys[node].size();
                }
                changes[stepStart + 1 + 2 * MMseqsNode::count] = nodeKeyOffset;
                changes.resize(stepStart + headerWords + nodeKeyOffset);
                for (size_t node = 0; node < MMseqsNode::count; node++) {
                    std::copy(nodeGroupRangeKeys[node].begin(), nodeGroupRangeKeys[node].end(), changes.begin() + stepStart + headerWords + changes[stepStart + 2 + MMseqsNode::count + node] - nodeGroupRangeKeys[node].size());
                    nodeGroupRangeKeys[node].clear();
                }
            }
        }
        for (int thread = 0; thread < threads; thread++) {
            threadAssignedKeys[thread].clear();
        }
        placingSeconds += stepTimer.getTimediff();
        waitWarning.reset();
    }
    if (changesWriter.joinable()) {
        changesWriter.join();
    }
    if (changes.empty() == false) {
        peakChangeBytes = std::max(peakChangeBytes, changes.size() * sizeof(uint64_t));
        changes.swap(writtenChanges);
        writeChanges(orderKeyEnd);
    }
    MMseqsNode::addPhaseSeconds(phaseValues, "clustering the accepted pairs", clusterTimer.getTimediff() - waitSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the aligning nodes", waitSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "merging accepted logs", mergeSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "deciding accepted pairs", decisionSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "placing the assigned changes", placingSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "cumulative accepted log read time", logReadSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "cumulative accepted log tail erase time", tailEraseSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "writing the taken bits", takenWriteSeconds);
    MMseqsNode::addPhaseValue(phaseValues, "Clustering steps", SSTR(stepCount));
    MMseqsNode::addPhaseValue(phaseValues, "Accepted log tail words moved", SSTR(tailMovedWords));
    MMseqsNode::addPhaseValue(phaseValues, "Free pairs", SSTR(freePairs));
    MMseqsNode::addPhaseValue(phaseValues, "Pairs of assigned reps", SSTR(assignedRepPairs) + ", of assigned members " + SSTR(assignedMemberPairs));
    MMseqsNode::addPhaseValue(phaseValues, "Pairs of sharing reps", SSTR(sharingRepPairs));
    MMseqsNode::addPhaseValue(phaseValues, "Peak assigned change bytes", SSTR(peakChangeBytes));
    MMseqsNode::addPhaseValue(phaseValues, "Clustered pairs", SSTR(clusteredPairs));
    MMseqsNode::addPhaseValue(phaseValues, "Accepted log bytes read", SSTR(logReadBytes));
    if (mode == Parameters::SET_COVER) {
        MMseqsNode::addPhaseValue(phaseValues, "Peak set-cover heap/pair bytes", SSTR(peakSetCoverStateBytes));
        MMseqsNode::addPhaseValue(phaseValues, "Peak set-cover live members", SSTR(sizeClasses.peakAllMembers));
        MMseqsNode::addPhaseValue(phaseValues, "Peak set-cover candidates", SSTR(sizeClasses.peakAllCandidates));
        MMseqsNode::addPhaseValue(phaseValues, "Peak set-cover member pool bytes", SSTR(largeSpool.memberBytes()));
        MMseqsNode::addPhaseValue(phaseValues, "Set-cover large-set spool written bytes", SSTR(largeSpool.bytesWritten()));
        MMseqsNode::addPhaseValue(phaseValues, "Set-cover large-set spool read bytes", SSTR(largeSpool.bytesRead()));
        MMseqsNode::addPhaseValue(phaseValues, "Set-cover large-set spool peak disk bytes", SSTR(largeSpool.peakDisk()));
        MMseqsNode::addPhaseValue(phaseValues, "Set-cover pair spool bytes", SSTR(pairSpool.writtenBytes()));
        MMseqsNode::addPhaseValue(phaseValues, "Set-cover small-set spool written bytes", SSTR(smallSpool.totalWrittenBytes()));
        MMseqsNode::addPhaseValue(phaseValues, "Set-cover small-set spool read bytes", SSTR(smallSpool.totalReadBytes()));
        MMseqsNode::addPhaseValue(phaseValues, "Set-cover small-set spool peak disk bytes", SSTR(smallSpool.peakBytesOnDisk()));
    }
    if (fclose(changesHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << assignedChanges << "\n";
        EXIT(EXIT_FAILURE);
    }
    // every node reads the assigned bits of its keys; an assigned key is no singleton
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
        const std::string takenFile = taken + "." + SSTR(node);
        if (nodeTaken[node].carryBits > 0 && fwrite(&nodeTaken[node].carry, sizeof(uint64_t), 1, nodeTaken[node].file) != 1) {
            Debug(Debug::ERROR) << "Cannot write " << takenFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        MMseqsNode::publishLog(nodeTaken[node].file, takenFile, dbSize, nodeTaken[node].bytes + ((nodeTaken[node].carryBits > 0) ? sizeof(uint64_t) : 0));
        if (fclose(nodeTaken[node].file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << takenFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        delete[] nodeTaken[node].buffer;
    }
    return memberCount;
}

// --createdb-mode 3: the nodes align the reps of their targets while the master clusters
int align2clustLinfinity(Parameters &par) {
    Timer timer;
    std::string phaseValues;
    Timer phaseValueTimer;
    std::string seqDb = par.db1;
    std::vector<std::string> localCopies;
    size_t localCopyMemory = 0;
    // a master that only clusters reads the index, not the sequences
    if (par.localDbDir.empty() == false && MMseqsNode::rank >= getFirstAlignNode()) {
        seqDb = FileUtil::copyToLocalDir(par.db1, par.localDbDir + "/" + FileUtil::baseName(par.db3) + "." + SSTR(MMseqsNode::rank), par.threads, localCopies, localCopyMemory);
        MMseqsNode::addPhaseTime(phaseValues, "copying the sequences to --local-db-dir", phaseValueTimer);
    }
    DBReader<DBKeyType> seqDbr(seqDb.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    seqDbr.open(DBReader<DBKeyType>::NOSORT);
    const std::vector<size_t> nodeKeyOffsets = seqDbr.getNodeKeyOffsets(MMseqsNode::count);
    const size_t dbSize = seqDbr.getSize();
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    // a copy on tmpfs holds the sequences in memory
    const size_t nodeMemoryLimit = std::max(Util::computeMemory(par.splitMemoryLimit), localCopyMemory) - localCopyMemory;
    const bool masterAligns = getFirstAlignNode() == 0;
    const size_t masterMemoryLimit = nodeMemoryLimit / (masterAligns ? MASTER_MEMORY_DIVISOR : 1);
    const size_t masterMemory = ((par.clusteringMode == Parameters::SET_COVER) ? masterMemoryLimit : (dbSize + 63) / 64 * sizeof(uint64_t) + masterMemoryLimit / STEP_MEMORY_DIVISOR)
                                + MMseqsNode::count * TAKEN_BUFFER_BYTES;

    // the groups in key order are cut into ranges of about equal cost that the aligning nodes claim in key order
    const std::string partitionsData = MMseqsNode::readFile(par.db2 + ".partitions");
    std::vector<PrefPartition> partitions;
    uint128 totalCost = 0;
    size_t totalPrefBytes = 0;
    for (char *line = (char *) partitionsData.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[5];
        Util::getWordsOfLine(line, columns, 5);
        PrefPartition partition;
        partition.firstKey = Util::fast_atoi<size_t>(columns[0]);
        partition.prefOffset = Util::fast_atoi<size_t>(columns[1]);
        partition.prefBytes = Util::fast_atoi<size_t>(columns[2]);
        partition.cost = Util::fast_atoi<size_t>(columns[3]);
        partition.node = Util::fast_atoi<size_t>(columns[4]);
        partitions.push_back(partition);
        totalCost += partition.cost;
        totalPrefBytes += partition.prefBytes;
    }
    const size_t alignNodes = MMseqsNode::count - getFirstAlignNode();
    const size_t maxGroupRangeCount = std::max<size_t>(1, std::min(GROUP_RANGES_PER_NODE * alignNodes, totalPrefBytes / MIN_GROUP_RANGE_PREF_BYTES));
    std::vector<size_t> groupRangePartitionOffsets(1, 0);
    std::vector<size_t> groupRangeKeyOffsets(1, 0);
    uint128 cost = 0;
    for (size_t i = 0; i + 1 < partitions.size(); i++) {
        cost += partitions[i].cost;
        if (groupRangePartitionOffsets.size() < maxGroupRangeCount && cost * maxGroupRangeCount >= totalCost * groupRangePartitionOffsets.size()) {
            groupRangePartitionOffsets.push_back(i + 1);
            groupRangeKeyOffsets.push_back(partitions[i + 1].firstKey);
        }
    }
    groupRangePartitionOffsets.push_back(partitions.size());
    groupRangeKeyOffsets.push_back(dbSize);
    MMseqsNode::WorkQueue groupRangeQueue(par.db3 + "_group_ranges", groupRangePartitionOffsets.size() - 1);

    // builds the block table of getKeyBlock
    seqDbr.setBlockCacheBytes(0);
    const std::vector<size_t> &blockKeyOffsets = seqDbr.getBlockKeyOffsets();
    // the master keeps the serving node of every page of assigned bits
    const size_t pageCount = (dbSize + ASSIGNED_PAGE_KEYS - 1) / ASSIGNED_PAGE_KEYS;
    std::vector<unsigned int> pageNodes(MMseqsNode::isMaster() ? pageCount : 0);
    for (size_t block = 0, pageEnd = 0; block + 1 < blockKeyOffsets.size(); block++) {
        const unsigned int node = getBlockNode(block);
        const size_t lastPage = (blockKeyOffsets[block + 1] - 1) / ASSIGNED_PAGE_KEYS;
        for (size_t page = blockKeyOffsets[block] / ASSIGNED_PAGE_KEYS; MMseqsNode::isMaster() && page <= lastPage; page++) {
            // a page that holds the keys of two nodes leaves them to getKeyNode
            pageNodes[page] = (page < pageEnd && pageNodes[page] != node) ? UINT_MAX : node;
        }
        pageEnd = lastPage + 1;
    }
    // an aligning node keeps the pages of the blocks that it serves and of its group ranges, about twice its share of the pages if the nodes are about as fast
    const size_t assignedPageCount = (MMseqsNode::rank >= getFirstAlignNode()) ? std::min(pageCount, 2 * ((pageCount + alignNodes - 1) / alignNodes)) : 0;
    const size_t assignedMemory = assignedPageCount * (ASSIGNED_PAGE_KEYS / 8) + ((assignedPageCount > 0) ? pageCount * sizeof(uint64_t *) : 0) + pageNodes.size() * sizeof(unsigned int);
    const size_t memoryLimit = nodeMemoryLimit - std::min(nodeMemoryLimit, (masterAligns ? std::min(masterMemoryLimit, masterMemory) : 0) + assignedMemory);
    const int clusterThreads = masterAligns ? std::min(par.threads, CLUSTER_THREADS) : par.threads;

    // a clusthashfast subset has bits; its sixth index column counts the representatives for the e-values
    const bool hasRepresentativeBits = FileUtil::fileExists((par.db1 + ".bits." + SSTR(MMseqsNode::rank)).c_str());
    size_t representatives = 0;
    size_t representativeResidues = 0;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        representatives += seqDbr.getRangeRepresentativeCount(range);
        representativeResidues += seqDbr.getRangeRepresentativeCount(range) * (index[range].length - 2);
    }

    const std::string accepted = par.db3 + "_accepted";
    const std::string assignedChanges = par.db3 + "_assigned";
    const std::string taken = par.db3 + "_taken";
    // only the singletons of a database without bits need the assigned bits
    const std::string assignedBits = hasRepresentativeBits ? "" : par.db3 + "_assigned.bits";
    // the sequence exchange indexes exist from before the memory barrier on, so no node looks one up before it is there
    const std::string requests = par.db3 + "_sequence_requests";
    const std::string served = par.db3 + "_served_sequences";
    std::vector<std::string> sequenceIndexes;
    std::vector<std::string> sequenceFiles;
    if (MMseqsNode::rank >= getFirstAlignNode() && alignNodes > 1) {
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

    // every node removes the claims of the group ranges rank, rank + count, ... that a killed run left before the barrier of the memory limits
    groupRangeQueue.removeClaims();
    // all nodes use the smallest memory limit besides the assigned bits, so they agree on the splits and fail together
    const size_t smallestMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db3 + ".memory", nodeMemoryLimit - std::min(nodeMemoryLimit, assignedMemory));
    const size_t paddedDbSize = (dbSize + 63) / 64 * 64;
    const size_t prefSizeBits = (par.clusteringMode == Parameters::SET_COVER) ? 64 - __builtin_clzll(dbSize | 1) : 1;
    const size_t maxRecordBytes = std::max<size_t>(1, smallestMemoryLimit / RECORD_MEMORY_DIVISOR);
    // every node cuts its served flushes at the same bytes, so a received slice fits the reservation of any node
    const size_t flushEntryBytes = std::max(MMseqsNode::EXCHANGE_WRITE_BYTES, smallestMemoryLimit / FLUSH_MEMORY_DIVISOR);

    RepTargetsReader repTargetsReader(par, seqDbr, partitions, groupRangePartitionOffsets, groupRangeQueue, maxRecordBytes, memoryLimit, paddedDbSize, prefSizeBits);
    if (prefSizeBits > 1) {
        repTargetsReader.scan();
    }
    MMseqsNode::addPhaseTime(phaseValues, "scanning the prefilter", phaseValueTimer);

    ClusterMemberLookup *members = NULL;
    DBReader<DBKeyType> *memberDbr = NULL;
    if (par.filterCluDBFile.empty() == false) {
        // every node reads the anchors of its part of the clustering, which every aligning node gets in node order
        const size_t anchorEntries = std::max(ANCHOR_ENTRIES, dbSize / std::max<size_t>(1, smallestMemoryLimit / ANCHOR_MEMORY_DIVISOR / sizeof(DBKeyType)));
        DBReader<DBKeyType> cluDb((par.filterCluDBFile + "." + SSTR(MMseqsNode::rank)).c_str(), (par.filterCluDBFile + ".index." + SSTR(MMseqsNode::rank)).c_str(), par.threads,
                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDb.open(DBReader<DBKeyType>::NOSORT);
        std::vector<DBKeyType> anchorKeys((cluDb.getSize() + anchorEntries - 1) / anchorEntries);
#pragma omp parallel for schedule(static)
        for (size_t anchor = 0; anchor < anchorKeys.size(); anchor++) {
            anchorKeys[anchor] = cluDb.getIndexEntry(anchor * anchorEntries).id;
        }
        cluDb.close();
        const std::string anchors = MMseqsNode::barrier(par.db3 + ".anchors", std::string(reinterpret_cast<const char *>(anchorKeys.data()), anchorKeys.size() * sizeof(DBKeyType)));
        MMseqsNode::removeOnceRead(par.db3 + ".anchors");
        if (MMseqsNode::rank >= getFirstAlignNode()) {
            members = new ClusterMemberLookup(par.filterCluDBFile, anchors, anchorEntries);
        }
        memberDbr = new DBReader<DBKeyType>(par.filterSeqDBFile.c_str(), (par.filterSeqDBFile + ".index").c_str(), par.threads,
                                            DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        memberDbr->open(DBReader<DBKeyType>::NOSORT);
        MMseqsNode::addPhaseTime(phaseValues, "indexing the cluster members", phaseValueTimer);
    }

    // the master clusters while the nodes align; a master that only clusters keeps an empty log
    size_t memberCount = 0;
    std::string clusterPhaseValues;
    std::thread clusterThread;
    if (MMseqsNode::isMaster()) {
        clusterThread = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(clusterThreads);
#endif
            memberCount = clusterAcceptedPairs(clusterThreads, dbSize, masterMemoryLimit, groupRangeKeyOffsets, groupRangeQueue, seqDbr, pageNodes, par.clusteringMode, paddedDbSize, prefSizeBits,
                                               accepted, assignedChanges, taken, assignedBits, clusterPhaseValues);
        });
    }
    phaseValueTimer.reset();
    alignRepTargets(par, seqDbr, repTargetsReader, memoryLimit, masterMemoryLimit, flushEntryBytes, representativeResidues, paddedDbSize, prefSizeBits,
                    members, memberDbr, phaseValues);
    delete members;
    repTargetsReader.close(phaseValues);
    if (MMseqsNode::rank >= getFirstAlignNode()) {
        MMseqsNode::addPhaseTime(phaseValues, "aligning the rep targets", phaseValueTimer);
        if (par.align2clustAlignerMode != Parameters::ALIGNER_MODE_DEFAULT) {
            MMseqsNode::addPhaseValue(phaseValues, "SWIPE member alignments", SSTR(swipeLanePairs.load()) + " in lanes ("
                                     + SSTR(swipeLanePairs.load() * 100 / std::max<size_t>(1, swipeLaneSlots.load()))
                                     + "% lane fill), " + SSTR(swipeSinglePairs.load()) + " scalar");
            if (swipeMemberExtensions.load() > 0) {
                MMseqsNode::addPhaseValue(phaseValues, "SWIPE member extensions", SSTR(swipeMemberExtensions.load()) + " in lanes ("
                                         + SSTR(swipeMemberLaneRows.load() * 100 / std::max<size_t>(1, swipeMemberSlotRows.load())) + "% active lane rows)");
            }
            MMseqsNode::addPhaseValue(phaseValues, "SWIPE target alignments", SSTR(swipeTargetPairs.load()) + " in lanes ("
                                     + SSTR(swipeTargetLaneRows.load() * 100 / std::max<size_t>(1, swipeTargetSlotRows.load()))
                                     + "% active lane rows), " + SSTR(swipeTargetLongPairs.load()) + " long extensions");
            MMseqsNode::addPhaseValue(phaseValues, "SWIPE extensions in 8 bit lanes", SSTR(swipeByteSlotRows.load() * 100 / std::max<size_t>(1, swipeMemberSlotRows.load() + swipeTargetSlotRows.load()))
                                     + "% of the lane rows (" + SSTR(swipeByteLaneRows.load() * 100 / std::max<size_t>(1, swipeByteSlotRows.load())) + "% active), "
                                     + SSTR(swipeRerunPairs.load()) + " again in 16 bit lanes");
        }
    }
    MMseqsNode::addPhaseValue(phaseValues, "Memory limit", SSTR(memoryLimit));

    const std::string takenFile = taken + "." + SSTR(MMseqsNode::rank);
    size_t takenKey;
    size_t takenBytes;
    MMseqsNode::waitForLog(takenFile, takenKey, takenBytes);
    // every node has claimed its last group range before the master published the taken logs
    groupRangeQueue.removeClaims();
    if (MMseqsNode::isMaster()) {
        clusterThread.join();
        Debug(Debug::INFO) << "Number of clusters: " << representatives - memberCount << "\n";
        phaseValues += clusterPhaseValues;
        std::vector<unsigned int>().swap(pageNodes);
    }
    // the cluster thread was the last user of seqDbr, the last reader of the copy
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
    const size_t nodeWords = (nodeKeys + 63) / 64;
    const std::string clusterPairs = par.db3 + "_cluster_pairs";
    MMseqsNode::ClusterPairWriter clusterPairWriter(clusterPairs, std::max<size_t>(64, memoryLimit / CLUSTER_PAIRS_MEMORY_DIVISOR / sizeof(ClusterPair)), ClusterPair::compareByRep, nodeKeyOffsets);
    const std::string acceptedFile = accepted + "." + SSTR(MMseqsNode::rank);
    if (par.clusteringMode == Parameters::SET_COVER) {
        for (size_t from = 0; from < takenBytes;) {
            const size_t to = std::min(takenBytes, from + clusterPairWriter.pairCapacity * sizeof(ClusterPair));
            MMseqsNode::readLog(takenFile, from, to, reinterpret_cast<char *>(clusterPairWriter.reservePairs((to - from) / sizeof(ClusterPair))));
            from = to;
        }
    } else {
        std::vector<uint64_t> takenBits(takenBytes / sizeof(uint64_t));
        MMseqsNode::readLog(takenFile, 0, takenBits.size() * sizeof(uint64_t), reinterpret_cast<char *>(takenBits.data()));
        FILE *acceptedHandle = FileUtil::openFileOrDie(acceptedFile.c_str(), "r", true);
        // a chunk takes whole words of taken bits and fits into the writer; bit i is word i of the log
        const size_t chunkWords = std::min(clusterPairWriter.pairCapacity, std::max<size_t>(1, takenBits.size()) * 64) / 64 * 64;
        std::vector<uint64_t> acceptedWords(chunkWords);
        std::vector<std::vector<size_t> > threadRepStarts(par.threads);
        std::vector<size_t> repStarts;
        uint64_t previousRepWord = 0;
        for (size_t wordCount, firstWord = 0; (wordCount = fread(acceptedWords.data(), sizeof(uint64_t), chunkWords, acceptedHandle)) > 0; firstWord += chunkWords / 64) {
            const size_t takenWordCount = (wordCount + 63) / 64;
            const uint64_t *takenWords = takenBits.data() + firstWord;
            size_t takenCount = 0;
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                threadRepStarts[thread_idx].clear();
#pragma omp for schedule(static) reduction(+:takenCount)
                for (size_t word = 0; word < takenWordCount; word++) {
                    takenCount += __builtin_popcountll(takenWords[word]);
                    for (size_t i = word * 64; i < std::min(wordCount, word * 64 + 64); i++) {
                        if (acceptedWords[i] & ACCEPTED_REP_BIT) {
                            threadRepStarts[thread_idx].push_back(i);
                        }
                    }
                }
            }
            repStarts.clear();
            for (int thread = 0; thread < par.threads; thread++) {
                repStarts.insert(repStarts.end(), threadRepStarts[thread].begin(), threadRepStarts[thread].end());
            }
            ClusterPair *takenPairs = clusterPairWriter.reservePairs(takenCount);
            size_t takenPairCount = 0;
#pragma omp parallel for schedule(static)
            for (size_t word = 0; word < takenWordCount; word++) {
                uint64_t bits = takenWords[word];
                if (bits == 0) {
                    continue;
                }
                size_t pairOffset = __sync_fetch_and_add(&takenPairCount, __builtin_popcountll(bits));
                size_t rep = std::upper_bound(repStarts.begin(), repStarts.end(), word * 64) - repStarts.begin();
                for (; bits != 0; bits &= bits - 1) {
                    const size_t i = word * 64 + __builtin_ctzll(bits);
                    while (rep < repStarts.size() && repStarts[rep] < i) {
                        rep++;
                    }
                    const uint64_t repWord = (rep > 0) ? acceptedWords[repStarts[rep - 1]] : previousRepWord;
                    const ClusterPair takenPair = { static_cast<DBKeyType>((repWord & ~ACCEPTED_REP_BIT) % paddedDbSize), static_cast<DBKeyType>(acceptedWords[i]) };
                    takenPairs[pairOffset++] = takenPair;
                }
            }
            if (repStarts.empty() == false) {
                previousRepWord = acceptedWords[repStarts.back()];
            }
        }
        if (fclose(acceptedHandle) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << acceptedFile << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    MMseqsNode::removeLog(takenFile);
    MMseqsNode::removeLog(acceptedFile);
    const std::string clusterPairsIndex = clusterPairWriter.close();
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
        memberDbr->close();
        delete memberDbr;
    }

    // every node writes its clusters, and its singletons unless the subset has bits, as mergeclusters takes those from the lower clustering; readKeyBits fills a word past the keys
    uint64_t *singletonBits = NULL;
    if (assignedBits.empty() == false) {
        size_t assignedKey;
        size_t assignedBytes;
        MMseqsNode::waitForLog(assignedBits, assignedKey, assignedBytes);
        FILE *assignedHandle = FileUtil::openFileOrDie(assignedBits.c_str(), "r", true);
        std::vector<uint64_t> nodeAssignedBits(nodeWords + 1, 0);
        MMseqsNode::readKeyBits(fileno(assignedHandle), assignedBits, 0, 0, nodeFirstKey, nodeFirstKey + nodeKeys, nodeFirstKey, nodeAssignedBits.data());
        if (fclose(assignedHandle) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << assignedBits << "\n";
            EXIT(EXIT_FAILURE);
        }
        singletonBits = new(std::nothrow) uint64_t[nodeWords];
        Util::checkAllocation(singletonBits, "Can not allocate singletonBits memory in align2clust");
        for (size_t word = 0; word < nodeWords; word++) {
            singletonBits[word] = ~nodeAssignedBits[word];
        }
    }
    MMseqsNode::writeClusterPairs(clusterPairsIndex, clusterPairs, par.db3, singletonBits, nodeFirstKey, nodeKeys);
    delete[] singletonBits;
    MMseqsNode::addPhaseTime(phaseValues, "writing the clusters", phaseValueTimer);
    MMseqsNode::removeExchanges(par.db3, clusterPairs, std::vector<std::string>(1, clusterPairs), phaseValues);
    if (MMseqsNode::isMaster()) {
        // every node is done with both logs
        MMseqsNode::removeLog(assignedChanges);
        FileUtil::remove((par.db3 + ".memory").c_str());
        if (assignedBits.empty() == false) {
            MMseqsNode::removeLog(assignedBits);
        }
        DBWriter::writeDbtypeFile(par.db3.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_CLUSTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
    }
    Debug(Debug::INFO) << "Time for run Align2Clust: " << timer.lap() << " sec\n";
    return EXIT_SUCCESS;
}
