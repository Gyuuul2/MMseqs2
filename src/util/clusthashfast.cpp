#include "DBWriter.h"
#include "FileUtil.h"
#include "Util.h"
#include "Parameters.h"
#include "Debug.h"
#include "DBReader.h"
#include "ReducedMatrix.h"
#include "SubstitutionMatrix.h"
#include "DistanceCalculator.h"
#include "Orf.h"
#include "FastSort.h"
#include "itoa.h"
#include "Timer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <fcntl.h>

#ifdef OPENMP
#include <omp.h>
#endif

// writes the clustering directly, like align2clust does, so no alignment DB and no clust step follow

// many chunks per thread so a few huge runs cannot unbalance the fan-out
static const size_t CLUSTHASHFAST_CHUNK_RECORDS = 65536;
static const size_t CLUSTHASHFAST_RESULT_RESERVE = 4096;
// one huge run must not leave every thread holding a giant buffer
static const size_t CLUSTHASHFAST_MAX_RETAINED_RESULT = 4 * 1024 * 1024;
static const size_t CLUSTHASHFAST_MAX_RETAINED_CLAIMS = 1024 * 1024;
static const unsigned int CLUSTHASHFAST_MAX_PARTITIONS = 4096;
// per thread write buffer budget for the hash pass, split across the partitions
static const size_t CLUSTHASHFAST_PENDING_BYTES = 8 * 1024 * 1024;
// progress is observability, not work: billions of atomic/progress calls can otherwise dominate scans
static const size_t CLUSTHASHFAST_PROGRESS_STEP = 1ull << 20;

// Debug::Progress paints the id-1..id delta, so a sparse id prints nothing: count updates, not ids
static size_t clusthashfastProgressSteps(size_t dbSize) {
    const size_t steps = (dbSize + CLUSTHASHFAST_PROGRESS_STEP - 1) / CLUSTHASHFAST_PROGRESS_STEP;
    return (steps == 0) ? 1 : steps;
}
// small enough that the threads' active id-order window stays inside a couple of length blocks
static const size_t CLUSTHASHFAST_RUN_ORDER_CHUNK = 4096;
static const size_t CLUSTHASHFAST_BATCH_IDS = 4096;
// far enough behind the sweep that a straggling thread is not evicted out from under it
static const size_t CLUSTHASHFAST_CACHE_DROP_BYTES = 256 * 1024 * 1024;

// a partition file is read once and deleted straight after, so its cache has no second reader
static void clusthashfastDropFileCache(FILE *file, const std::string &name) {
#if defined(HAVE_POSIX_FADVISE)
    const int fd = fileno(file);
    if (fd < 0) {
        return;
    }
    const int ret = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (ret != 0) {
        Debug(Debug::WARNING) << "POSIX_FADV_DONTNEED failed for " << name << ": " << ret << "\n";
    }
#else
    (void) file;
    (void) name;
#endif
}

struct ClusterCounts {
    size_t clusters;
    size_t merged;
    size_t runs;

    ClusterCounts() : clusters(0), merged(0), runs(0) {}

    void add(const ClusterCounts &other) {
        clusters += other.clusters;
        merged += other.merged;
        runs += other.runs;
    }
};

// packed: the padding was 4 of every 16 bytes, and this array is the largest allocation at scale
struct __attribute__((packed)) HashEntry {
    size_t hash;
    DBLocalId id;

    // the sort key and the ascending id scan are what make the result independent of the thread count
    static bool compareByHashAndId(const HashEntry &first, const HashEntry &second) {
        if (first.hash != second.hash) {
            return first.hash < second.hash;
        }
        return first.id < second.id;
    }
};

static void appendKey(std::string &buffer, DBKeyType key, char *lineBuffer) {
    char *outPos = Itoa::u64toa_sse2(static_cast<uint64_t>(key), lineBuffer);
    buffer.append(lineBuffer, (outPos - lineBuffer - 1));
    buffer.push_back('\n');
}

// per-thread output buffer, claim flags and counters of the greedy scan
struct ClusterWorker {
    std::string result;
    std::string querySeq;
    std::vector<unsigned char> claimed;
    std::vector<unsigned int> memberLength;
    std::vector<size_t> batchIds;
    std::vector<size_t> chunkIds;
    std::vector<size_t> chunkRunPos;
    std::vector<size_t> chunkRunSlot;
    char lineBuffer[32];
    ClusterCounts counts;

    ClusterWorker() {
        result.reserve(CLUSTHASHFAST_RESULT_RESERVE);
    }

    void beginCluster(DBKeyType representativeKey) {
        if (result.capacity() > CLUSTHASHFAST_MAX_RETAINED_RESULT) {
            std::string().swap(result);
            result.reserve(CLUSTHASHFAST_RESULT_RESERVE);
        } else {
            result.clear();
        }
        appendKey(result, representativeKey, lineBuffer);
    }

    void addMember(DBKeyType memberKey) {
        appendKey(result, memberKey, lineBuffer);
        counts.merged++;
    }

    void writeCluster(DBWriter &writer, DBWriter &representativeWriter, DBReader<DBKeyType> &reader,
                      DBLocalId representativeId, DBKeyType representativeKey, unsigned int thread_idx) {
        writer.writeData(result.c_str(), result.length(), representativeKey, thread_idx);
        representativeWriter.writeIndexEntry(representativeKey, reader.getOffset(representativeId),
                                             reader.getEntryLen(representativeId), thread_idx);
        counts.clusters++;
    }

    void trimClaimed() {
        if (claimed.capacity() > CLUSTHASHFAST_MAX_RETAINED_CLAIMS) {
            std::vector<unsigned char>().swap(claimed);
        }
        if (batchIds.capacity() > CLUSTHASHFAST_MAX_RETAINED_CLAIMS) {
            std::vector<size_t>().swap(batchIds);
        }
        if (memberLength.capacity() > CLUSTHASHFAST_MAX_RETAINED_CLAIMS) {
            std::vector<unsigned int>().swap(memberLength);
        }
    }
};

// the smaller of both strands, so a nucleotide sequence and its reverse complement hash the same
static size_t hashNucleotideSequence(const char *data, size_t length) {
    const size_t A = 31;
    size_t h1 = 0;
    size_t h2 = 0;
    for (size_t i = 0; i < length; ++i) {
        h1 = ((h1 * A) + data[i]);
        h2 = ((h2 * A) + Orf::complement(data[length - i - 1]));
    }
    return std::min(h1, h2);
}

// folding the length in keeps a run to one length, which shrinks the runs and drops the length scan
static size_t hashWithLength(size_t hash, size_t length) {
    return hash ^ (length + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2));
}

// every pass over the sequence data now walks it in id order, which is what the readahead wants
static bool hashScanIsSequential(DBReader<DBKeyType> &reader) {
    return reader.isSortedByOffset();
}

// a block wider than any readahead window keeps neighbouring threads from allocating the same folios
static size_t idScanBlock(DBReader<DBKeyType> &reader, size_t dbSize, int threads) {
    const size_t DEFAULT_BLOCK = 1000;
    const size_t MIN_BLOCKS_PER_THREAD = 16;
    const size_t TARGET_SPAN = 64 * 1024 * 1024;
    if (threads < 2 || dbSize == 0 || hashScanIsSequential(reader) == false) {
        return DEFAULT_BLOCK;
    }
    size_t fileSize = 0;
    for (size_t fileIdx = 0; fileIdx < reader.getDataFileCnt(); fileIdx++) {
        fileSize += reader.getDataSizeForFile(fileIdx);
    }
    size_t lastEnd = std::min(reader.getIndex(dbSize - 1)->offset + reader.getEntryLen(dbSize - 1), fileSize);
    if (fileSize == 0 || lastEnd == 0) {
        return DEFAULT_BLOCK;
    }
    size_t stride = std::max<size_t>(lastEnd / dbSize, 1);
    size_t wanted = (TARGET_SPAN + stride - 1) / stride;
    size_t balanced = dbSize / (static_cast<size_t>(threads) * MIN_BLOCKS_PER_THREAD);
    return std::max(std::min(wanted, balanced), DEFAULT_BLOCK);
}

// the amino acid hash runs over the reduced alphabet, so a run is a candidate set, not a duplicate set
static size_t hashOf(DBReader<DBKeyType> &reader, size_t id, Sequence *seq, unsigned int thread_idx) {
    const size_t length = reader.getSeqLen(id);
    const char *data = reader.getData(id, thread_idx);
    if (seq == NULL) {
        return hashWithLength(hashNucleotideSequence(data, length), length);
    }
    seq->mapSequence(id, 0, data, length);
    return hashWithLength(Util::hash(seq->numSequence, seq->L), length);
}

static size_t hashRunEnd(const HashEntry *entries, size_t entryCount, size_t runBegin) {
    const size_t hash = entries[runBegin].hash;
    size_t runEnd = runBegin + 1;
    while (runEnd < entryCount && entries[runEnd].hash == hash) {
        runEnd++;
    }
    return runEnd;
}

// a run is owned by the chunk that starts it, so skip the tail of the run the previous chunk owns
static size_t firstOwnedRun(const HashEntry *entries, size_t chunkBegin, size_t chunkEnd) {
    size_t runBegin = chunkBegin;
    while (runBegin < chunkEnd && runBegin > 0 && entries[runBegin - 1].hash == entries[runBegin].hash) {
        runBegin++;
    }
    return runBegin;
}

// This reader has no batch arena, so an entry is taken one at a time straight from the mapping. The
// batch was a queue depth, not an answer: what it returns is the same bytes in the same order.

// the first unclaimed entry of the run represents it and claims every unclaimed entry it covers
static void clusterHashRun(DBReader<DBKeyType> &reader, DBWriter &writer, DBWriter &representativeWriter,
                           const HashEntry *run, size_t runSize, float seqIdThr, ClusterWorker &worker,
                           unsigned int thread_idx) {
    worker.counts.runs++;
    if (runSize == 1) {
        const DBKeyType representativeKey = reader.getDbKey(run[0].id);
        worker.beginCluster(representativeKey);
        worker.writeCluster(writer, representativeWriter, reader, run[0].id, representativeKey, thread_idx);
        return;
    }

    // resolve lengths once per member, so a large run spends its O(n^2) work on the distance only
    worker.memberLength.resize(runSize);
    worker.batchIds.resize(runSize);
    for (size_t k = 0; k < runSize; ++k) {
        worker.memberLength[k] = static_cast<unsigned int>(reader.getSeqLen(run[k].id));
        worker.batchIds[k] = static_cast<size_t>(run[k].id);
    }

    worker.claimed.assign(runSize, 0);
    for (size_t i = 0; i < runSize; i++) {
        if (worker.claimed[i]) {
            continue;
        }
        worker.claimed[i] = 1;
        const DBLocalId queryId = run[i].id;
        const unsigned int queryLength = worker.memberLength[i];
        const DBKeyType representativeKey = reader.getDbKey(queryId);
        worker.beginCluster(representativeKey);

        const char *querySeq = NULL;
        for (size_t j = i + 1; j < runSize; j++) {
            if (worker.claimed[j] || worker.memberLength[j] != queryLength) {
                continue;
            }
            if (querySeq == NULL) {
                worker.querySeq.assign(reader.getData(worker.batchIds[i], thread_idx), queryLength);
                querySeq = worker.querySeq.data();
            }
            const char *targetSeq = reader.getData(worker.batchIds[j], thread_idx);
            const unsigned int distance =
                DistanceCalculator::computeInverseHammingDistance(querySeq, targetSeq, queryLength);
            const float seqId = static_cast<float>(distance) / static_cast<float>(queryLength);
            if (seqId >= seqIdThr) {
                worker.addMember(reader.getDbKey(run[j].id));
                worker.claimed[j] = 1;
            }
        }
        worker.writeCluster(writer, representativeWriter, reader, queryId, representativeKey, thread_idx);
    }
    worker.trimClaimed();
}

static ClusterCounts clusterHashRuns(DBReader<DBKeyType> &reader, DBWriter &writer,
                                     DBWriter &representativeWriter, const HashEntry *entries,
                                     size_t entryCount, float seqIdThr, bool showProgress, int threads) {
    const size_t chunkCount = (entryCount + CLUSTHASHFAST_CHUNK_RECORDS - 1) / CLUSTHASHFAST_CHUNK_RECORDS;
    // one shared atomic per run is six billion increments on one cache line, so count chunks instead
    Debug::Progress progress(chunkCount);
    size_t totalClusters = 0;
    size_t totalMerged = 0;
    size_t totalRuns = 0;
#pragma omp parallel num_threads(threads) reduction(+:totalClusters) reduction(+:totalMerged) reduction(+:totalRuns)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        ClusterWorker worker;

#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
            const size_t chunkBegin = chunk * CLUSTHASHFAST_CHUNK_RECORDS;
            const size_t chunkEnd = std::min(entryCount, chunkBegin + CLUSTHASHFAST_CHUNK_RECORDS);
            if (showProgress) {
                progress.updateProgress(chunk);
            }
            worker.chunkRunPos.clear();
            size_t pos = firstOwnedRun(entries, chunkBegin, chunkEnd);
            while (pos < chunkEnd) {
                const size_t runEnd = hashRunEnd(entries, entryCount, pos);
                // a singleton is its own cluster and needs no sequence read at all
                if (runEnd - pos == 1) {
                    clusterHashRun(reader, writer, representativeWriter, entries + pos, 1, seqIdThr,
                                   worker, thread_idx);
                } else {
                    worker.chunkRunPos.push_back(pos);
                    worker.chunkRunPos.push_back(runEnd - pos);
                }
                pos = runEnd;
            }

            const size_t runCount = worker.chunkRunPos.size() / 2;
            // Without a batch arena there is nothing to fill, so a run is clustered where it is found.
            for (size_t r = 0; r < runCount; r++) {
                clusterHashRun(reader, writer, representativeWriter,
                               entries + worker.chunkRunPos[2 * r], worker.chunkRunPos[2 * r + 1],
                               seqIdThr, worker, thread_idx);
            }
        }
        totalClusters += worker.counts.clusters;
        totalMerged += worker.counts.merged;
        totalRuns += worker.counts.runs;
    }

    ClusterCounts counts;
    counts.clusters = totalClusters;
    counts.merged = totalMerged;
    counts.runs = totalRuns;
    return counts;
}

// the sweep never returns to a block it has passed, so the pages behind it are page cache it will not use
static void dropSweptCache(DBReader<DBKeyType> &reader, std::atomic<size_t> &droppedTo, size_t end) {
    size_t seen = droppedTo.load(std::memory_order_relaxed);
    if (end < seen + CLUSTHASHFAST_CACHE_DROP_BYTES) {
        return;
    }
    // this reader cannot be told to forget a range it has passed, so the sweep only records where it is
    droppedTo.compare_exchange_strong(seen, end, std::memory_order_relaxed);
}

// on a length-descending db a run lives inside one length block, so an id-order sweep caches one block
static ClusterCounts clusterPartition(DBReader<DBKeyType> &reader, DBWriter &writer,
                                      DBWriter &representativeWriter, const HashEntry *entries,
                                      size_t entryCount, float seqIdThr, bool showProgress,
                                      bool idOrderSweep, bool dropSweptPages, int threads) {
    if (idOrderSweep == false) {
        return clusterHashRuns(reader, writer, representativeWriter, entries, entryCount, seqIdThr, showProgress, threads);
    }

    // singletons never read sequence data, so cluster them here and only order the multi-member runs
    Timer orderTimer;
    const size_t chunkCount = (entryCount + CLUSTHASHFAST_CHUNK_RECORDS - 1) / CLUSTHASHFAST_CHUNK_RECORDS;
    std::vector<std::vector<size_t> > perThread(static_cast<size_t>(threads));
    size_t totalClusters = 0;
    size_t totalMerged = 0;
    size_t totalRuns = 0;
#pragma omp parallel num_threads(threads) reduction(+:totalClusters) reduction(+:totalMerged) reduction(+:totalRuns)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        ClusterWorker worker;
        std::vector<size_t> &mine = perThread[thread_idx];
#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
            const size_t chunkBegin = chunk * CLUSTHASHFAST_CHUNK_RECORDS;
            const size_t chunkEnd = std::min(entryCount, chunkBegin + CLUSTHASHFAST_CHUNK_RECORDS);
            size_t pos = firstOwnedRun(entries, chunkBegin, chunkEnd);
            while (pos < chunkEnd) {
                const size_t runEnd = hashRunEnd(entries, entryCount, pos);
                if (runEnd - pos == 1) {
                    clusterHashRun(reader, writer, representativeWriter, entries + pos, 1, seqIdThr,
                                   worker, thread_idx);
                } else {
                    mine.push_back(pos);
                }
                pos = runEnd;
            }
        }
        totalClusters += worker.counts.clusters;
        totalMerged += worker.counts.merged;
        totalRuns += worker.counts.runs;
    }
    ClusterCounts counts;
    counts.clusters = totalClusters;
    counts.merged = totalMerged;
    counts.runs = totalRuns;

    size_t runCount = 0;
    for (size_t t = 0; t < perThread.size(); t++) {
        runCount += perThread[t].size();
    }
    std::vector<size_t> runOrder(runCount);
    size_t out = 0;
    for (size_t t = 0; t < perThread.size(); t++) {
        if (perThread[t].empty() == false) {
            memcpy(&runOrder[out], perThread[t].data(), perThread[t].size() * sizeof(size_t));
            out += perThread[t].size();
            std::vector<size_t>().swap(perThread[t]);
        }
    }
    // first-member ids are unique across runs, so this order is total and thread-count independent
    SORT_PARALLEL(runOrder.begin(), runOrder.end(),
                  [entries](size_t first, size_t second) { return entries[first].id < entries[second].id; });
    Debug(Debug::INFO) << "Run order index: " << runCount << " multi-member runs, "
                       << runCount * sizeof(size_t) << " byte (" << orderTimer.lap() << ")\n";
    if (runCount == 0) {
        return counts;
    }

    Timer sweepTimer;
    std::atomic<size_t> droppedTo(0);
    const size_t orderChunkCount = (runCount + CLUSTHASHFAST_RUN_ORDER_CHUNK - 1)
                                   / CLUSTHASHFAST_RUN_ORDER_CHUNK;
    Debug::Progress progress(orderChunkCount);
    totalClusters = 0;
    totalMerged = 0;
    totalRuns = 0;
#pragma omp parallel num_threads(threads) reduction(+:totalClusters) reduction(+:totalMerged) reduction(+:totalRuns)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        ClusterWorker worker;
#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < orderChunkCount; ++chunk) {
            if (showProgress) {
                progress.updateProgress(chunk);
            }
            const size_t begin = chunk * CLUSTHASHFAST_RUN_ORDER_CHUNK;
            const size_t end = std::min(runCount, begin + CLUSTHASHFAST_RUN_ORDER_CHUNK);
            for (size_t idx = begin; idx < end; ++idx) {
                const size_t pos = runOrder[idx];
                clusterHashRun(reader, writer, representativeWriter, entries + pos,
                               hashRunEnd(entries, entryCount, pos) - pos, seqIdThr, worker, thread_idx);
            }
            if (dropSweptPages) {
                dropSweptCache(reader, droppedTo, reader.getOffset(entries[runOrder[end - 1]].id));
            }
        }
        totalClusters += worker.counts.clusters;
        totalMerged += worker.counts.merged;
        totalRuns += worker.counts.runs;
    }
    counts.clusters += totalClusters;
    counts.merged += totalMerged;
    counts.runs += totalRuns;
    Debug(Debug::INFO) << "Time for the id-order sweep: " << sweepTimer.lap() << "\n";
    return counts;
}

// a run never straddles a hash, so ascending high-bit partitions sort in bounded memory in hash order
struct HashPartitions {
    std::vector<FILE *> files;
    std::vector<std::string> names;
    std::vector<size_t> counts;
    unsigned int bits;

    HashPartitions() : bits(0) {}

    ~HashPartitions() {
        close();
        for (size_t part = 0; part < names.size(); part++) {
            if (FileUtil::fileExists(names[part].c_str())) {
                FileUtil::remove(names[part].c_str());
            }
        }
    }

    bool open(const std::string &prefix, unsigned int partitions) {
        bits = 0;
        while ((1u << bits) < partitions) {
            bits++;
        }
        files.assign(partitions, NULL);
        counts.assign(partitions, 0);
        for (unsigned int part = 0; part < partitions; part++) {
            names.push_back(prefix + "." + SSTR(part));
            files[part] = fopen(names[part].c_str(), "wb");
            if (files[part] == NULL) {
                return false;
            }
        }
        return true;
    }

    void close() {
        for (size_t part = 0; part < files.size(); part++) {
            if (files[part] != NULL) {
                fclose(files[part]);
                files[part] = NULL;
            }
        }
    }

    // a shift of 64 is undefined, so the single partition case must not reach the shift at all
    unsigned int partitionOf(size_t hash) const {
        return (bits == 0) ? 0u : static_cast<unsigned int>(hash >> (64 - bits));
    }
};

static bool flushPartition(HashPartitions &parts, std::vector<std::mutex> &locks, unsigned int partition,
                           std::vector<HashEntry> &pending) {
    std::lock_guard<std::mutex> lock(locks[partition]);
    const bool ok = fwrite(pending.data(), sizeof(HashEntry), pending.size(), parts.files[partition])
                    == pending.size();
    parts.counts[partition] += pending.size();
    pending.clear();
    return ok;
}

// the one pass that reads every sequence, hashing straight into the partitions
static bool hashIntoPartitions(DBReader<DBKeyType> &reader, HashPartitions &parts, bool isNuclInput,
                               BaseMatrix *subMat, size_t maxSeqLen, bool showProgress, int threads) {
    const size_t dbSize = reader.getSize();
    const size_t scanChunk = idScanBlock(reader, dbSize, threads);
    const unsigned int partitionCount = static_cast<unsigned int>(parts.files.size());
    // a fixed byte budget, not a fixed depth: per partition depth would grow what it is dividing
    const size_t flush = std::max<size_t>(64, CLUSTHASHFAST_PENDING_BYTES
                                              / (partitionCount * sizeof(HashEntry)));
    std::vector<std::mutex> locks(partitionCount);
    Debug::Progress progress(clusthashfastProgressSteps(dbSize));
    bool ok = true;
#pragma omp parallel num_threads(threads) reduction(&&:ok)
    {
        bool threadOk = true;
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        std::vector<std::vector<HashEntry> > pending(partitionCount);
        Sequence *seq = isNuclInput ? NULL : new Sequence(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
#pragma omp for schedule(static, scanChunk)
        for (size_t id = 0; id < dbSize; ++id) {
            if (showProgress && (id & (CLUSTHASHFAST_PROGRESS_STEP - 1)) == 0) {
                progress.updateProgress();
            }
            HashEntry entry;
            entry.hash = hashOf(reader, id, seq, thread_idx);
            entry.id = static_cast<DBLocalId>(id);
            const unsigned int partition = parts.partitionOf(entry.hash);
            pending[partition].push_back(entry);
            if (pending[partition].size() >= flush) {
                threadOk = flushPartition(parts, locks, partition, pending[partition]) && threadOk;
            }
        }
        for (unsigned int partition = 0; partition < partitionCount; partition++) {
            if (pending[partition].empty() == false) {
                threadOk = flushPartition(parts, locks, partition, pending[partition]) && threadOk;
            }
        }
        delete seq;
        ok = ok && threadOk;
    }
    return ok;
}

// one partition's entries must fit the budget, so the sort footprint never depends on the db size
static unsigned int hashPartitionCount(size_t dbSize, size_t budget) {
    const size_t needed = dbSize * sizeof(HashEntry);
    const size_t usable = std::max<size_t>(budget, 1);
    unsigned int partitions = 1;
    while (partitions < CLUSTHASHFAST_MAX_PARTITIONS && needed / partitions > usable) {
        partitions *= 2;
    }
    return partitions;
}

// one entry per id written in place, so a single partition needs no buffers, no locks and no file
static void hashAllSequences(DBReader<DBKeyType> &reader, HashEntry *entries, size_t dbSize,
                             bool isNuclInput, BaseMatrix *subMat, size_t maxSeqLen, bool showProgress,
                             int threads) {
    const size_t scanChunk = idScanBlock(reader, dbSize, threads);
    Debug::Progress progress(clusthashfastProgressSteps(dbSize));
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        Sequence *seq = isNuclInput ? NULL : new Sequence(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
#pragma omp for schedule(static, scanChunk)
        for (size_t id = 0; id < dbSize; ++id) {
            if (showProgress && (id & (CLUSTHASHFAST_PROGRESS_STEP - 1)) == 0) {
                progress.updateProgress();
            }
            entries[id].hash = hashOf(reader, id, seq, thread_idx);
            entries[id].id = static_cast<DBLocalId>(id);
        }
        delete seq;
    }
}

// the hash pass wants mmap readahead, so the clustering IO policy is only decided after it
static void clusthashfastSetClusterIo(DBReader<DBKeyType> &reader, bool idOrderSweep, bool cacheStarved) {
    if (idOrderSweep) {
        Debug(Debug::INFO) << "Sequence IO policy: mmap, the id-order sweep works one length block at a time\n";
        return;
    }
    if (cacheStarved == false) {
        Debug(Debug::INFO) << "Sequence IO policy: mmap, the working set fits in memory\n";
        return;
    }
    // a sequence belongs to one hash run, and a run's rereads are served by the batch arena, so unlike
    // align2clust nothing here is ever read twice and the page cache would only be pollution
    Debug(Debug::WARNING) << "Sequence IO policy: mmap. The working set does not fit and this reader "
                          << "has no descriptor path to fall back to\n";
}

static ClusterCounts clusterSequences(const Parameters &par, DBReader<DBKeyType> &reader, DBWriter &writer,
                                      DBWriter &representativeWriter, BaseMatrix *subMat, bool isNuclInput) {
    ClusterCounts counts;
    const size_t dbSize = reader.getSize();
    if (dbSize == 0) {
        return counts;
    }

    const bool showProgress = (Debug::debugLevel >= Debug::INFO);
    const std::string tmpPrefix = std::string(writer.getDataFileName());
    const size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    // a partition holds this much of the entry array, the rest of memory is free for the page cache
    const size_t entryBudget = std::max<size_t>(memoryLimit / 2, 1);
    const bool cacheStarved = (reader.getDataSize() + dbSize * sizeof(HashEntry) > memoryLimit);
    const unsigned int partitionCount = hashPartitionCount(dbSize, entryBudget);
    Debug(Debug::INFO) << "Memory limit " << memoryLimit << " byte, entries "
                       << dbSize * sizeof(HashEntry) / partitionCount << " byte per partition\n";

    Timer lengthOrderTimer;
    bool lengthDescending = true;
    for (size_t i = 1; i < reader.getSize() && lengthDescending; i++) {
        lengthDescending = reader.getSeqLen(i - 1) >= reader.getSeqLen(i);
    }
    Debug(Debug::INFO) << "Sequence DB length order: " << (lengthDescending ? "descending" : "unsorted")
                       << " (" << lengthOrderTimer.lap() << ")\n";
    bool idOrderSweep = lengthDescending;
    if (lengthDescending && hashScanIsSequential(reader) == false) {
        Debug(Debug::WARNING) << "Length-descending db is not offset-sorted, falling back to batched reads\n";
        idOrderSweep = false;
    }
    // the sweep passes each length block once, so holding it after is cache the sweep cannot use
    const bool dropSweptPages = idOrderSweep && cacheStarved;
    if (hashScanIsSequential(reader)) {
        reader.setSequentialAdvice();
    }

    if (partitionCount == 1) {
        Debug(Debug::INFO) << "Hashing " << dbSize << " sequences...\n";
        // HashEntry is trivial, so this allocation is not written until the pass below touches it
        HashEntry *entries = new(std::nothrow) HashEntry[dbSize];
        Util::checkAllocation(entries, "Cannot allocate hash entry memory in clusthashfast");
        hashAllSequences(reader, entries, dbSize, isNuclInput, subMat, par.maxSeqLen, showProgress,
                         par.threads);
        Debug(Debug::INFO) << "Sort sequence hashes...\n";
        Timer sortTimer;
        SORT_PARALLEL(entries, entries + dbSize, HashEntry::compareByHashAndId);
        Debug(Debug::INFO) << "Time for sorting hashes: " << sortTimer.lap() << "\n";
        clusthashfastSetClusterIo(reader, idOrderSweep, cacheStarved);
        Debug(Debug::INFO) << "Cluster equal length sequences...\n";
        counts = clusterPartition(reader, writer, representativeWriter, entries, dbSize, par.seqIdThr,
                                  showProgress, idOrderSweep, dropSweptPages, par.threads);
        delete[] entries;
        Debug(Debug::INFO) << "Found " << counts.runs << " unique hash/length groups\n";
        return counts;
    }

    Debug(Debug::INFO) << "Hashing " << dbSize << " sequences into " << partitionCount << " partitions...\n";
    HashPartitions parts;
    if (parts.open(tmpPrefix + ".hashpart", partitionCount) == false) {
        Debug(Debug::ERROR) << "Cannot open hash partitions under " << tmpPrefix << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (hashIntoPartitions(reader, parts, isNuclInput, subMat, par.maxSeqLen, showProgress,
                           par.threads) == false) {
        Debug(Debug::ERROR) << "Cannot write hash partitions under " << tmpPrefix << "\n";
        EXIT(EXIT_FAILURE);
    }
    parts.close();
    clusthashfastSetClusterIo(reader, idOrderSweep, cacheStarved);

    std::vector<HashEntry> part;
    for (unsigned int partition = 0; partition < partitionCount; partition++) {
        const size_t partSize = parts.counts[partition];
        if (partSize == 0) {
            continue;
        }
        part.resize(partSize);
        Timer partReadTimer;
        FILE *in = fopen(parts.names[partition].c_str(), "rb");
        if (in == NULL || fread(part.data(), sizeof(HashEntry), partSize, in) != partSize) {
            Debug(Debug::ERROR) << "Cannot read hash partition " << parts.names[partition] << "\n";
            EXIT(EXIT_FAILURE);
        }
        clusthashfastDropFileCache(in, parts.names[partition]);
        fclose(in);
        Debug(Debug::INFO) << "Time for reading partition " << (partition + 1) << ": "
                           << partReadTimer.lap() << "\n";
        FileUtil::remove(parts.names[partition].c_str());

        Timer partSortTimer;
        SORT_PARALLEL(part.data(), part.data() + partSize, HashEntry::compareByHashAndId);
        Debug(Debug::INFO) << "Time for sorting partition " << (partition + 1) << ": "
                           << partSortTimer.lap() << "\n";
        const ClusterCounts partCounts = clusterPartition(reader, writer, representativeWriter, part.data(),
                                                          partSize, par.seqIdThr, showProgress, idOrderSweep, dropSweptPages,
                                                          par.threads);
        counts.add(partCounts);
        Debug(Debug::INFO) << "Partition " << (partition + 1) << "/" << partitionCount << ": "
                           << partCounts.runs << " groups, " << partCounts.clusters << " clusters\n";
    }
    Debug(Debug::INFO) << "Found " << counts.runs << " unique hash/length groups\n";
    return counts;
}

int clusthashfast(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.alphabetSize = MultiParam<NuclAA<int> >(NuclAA<int>(Parameters::CLUST_HASH_DEFAULT_ALPH_SIZE, 5));
    par.seqIdThr = static_cast<float>(Parameters::CLUST_HASH_DEFAULT_MIN_SEQ_ID) / 100.0f;
    par.parseParameters(argc, argv, command, true, 0, 0);

    // NOSORT: nothing here needs a sorted access order and it saves the two 8 byte per sequence id maps
    DBReader<DBKeyType> reader(par.db1.c_str(), par.db1Index.c_str(), par.threads,
                               DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX);
    reader.open(DBReader<DBKeyType>::NOSORT);
    // only preload what can stay resident once the reader index is accounted for
    const bool dataFitsInMemory = reader.getDataSize() < Util::getTotalSystemMemory() / 2;
    if (par.preloadMode == Parameters::PRELOAD_MODE_MMAP_TOUCH
        || (par.preloadMode != Parameters::PRELOAD_MODE_MMAP && dataFitsInMemory)) {
        reader.readMmapedDataInMemory();
    }

    const bool isNuclInput = Parameters::isEqualDbtype(reader.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES);
    BaseMatrix *subMat = NULL;
    if (isNuclInput == false) {
        SubstitutionMatrix sMat(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.2);
        subMat = new ReducedMatrix(sMat.probMatrix, sMat.subMatrixPseudoCounts, sMat.aa2num, sMat.num2aa,
                                   sMat.alphabetSize, par.alphabetSize.values.aminoacid(), 2.0);
    }

    DBWriter writer(par.db2.c_str(), par.db2Index.c_str(), par.threads, par.compressed,
                    Parameters::DBTYPE_CLUSTER_RES);
    writer.open();

    // the workflow clusters the representatives next, so publish them as an index-only sub-db
    const std::string representativeDb = par.db2 + "_redundancy";
    DBWriter representativeWriter(representativeDb.c_str(), (representativeDb + ".index").c_str(),
                                  par.threads, 0, Parameters::DBTYPE_OMIT_FILE);
    representativeWriter.open();

    const ClusterCounts counts = clusterSequences(par, reader, writer, representativeWriter, subMat,
                                                  isNuclInput);

    if (subMat != NULL) {
        delete subMat;
    }
    writer.close();

    const bool mergeRepresentativeIndex = Parameters::isEqualDbtype(reader.getDbtype(), Parameters::DBTYPE_HMM_PROFILE)
        || Parameters::isEqualDbtype(reader.getDbtype(), Parameters::DBTYPE_AMINO_ACIDS)
        || Parameters::isEqualDbtype(reader.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES);
    representativeWriter.close(mergeRepresentativeIndex, true);
    DBReader<DBKeyType>::softlinkDb(par.db1, representativeDb, DBFiles::DATA);
    DBWriter::writeDbtypeFile(representativeDb.c_str(), reader.getDbtype(), reader.isCompressed());
    DBReader<DBKeyType>::softlinkDb(par.db1, representativeDb, DBFiles::SEQUENCE_ANCILLARY);
    reader.close();

    Debug(Debug::INFO) << counts.clusters << " clusters, " << counts.merged << " sequences merged by hamming distance\n";
    return EXIT_SUCCESS;
}
