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

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>

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

    void writeCluster(DBWriter &writer, DBKeyType representativeKey, unsigned int thread_idx) {
        writer.writeData(result.c_str(), result.length(), representativeKey, thread_idx);
        counts.clusters++;
    }

    void trimClaimed() {
        if (claimed.capacity() > CLUSTHASHFAST_MAX_RETAINED_CLAIMS) {
            std::vector<unsigned char>().swap(claimed);
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

// only multi-member runs are ever compared, and gathering just those in id order is one sequential pass
struct MemberArena {
    static const size_t BLOCK = 64;

    std::vector<uint64_t> bits;         // one bit per sequence, set when its run has other members
    std::vector<size_t> blockOffset;    // arena offset of the first gathered member of each 64 id block
    std::vector<size_t> touched;        // ascending indices of the blocks this window actually marked
    char *data;

    MemberArena() : data(NULL) {}

    ~MemberArena() {
        release();
    }

    // the two id-space arrays outlive a window, so only the gathered bytes and the block list go
    void releaseData() {
        delete[] data;
        data = NULL;
        for (size_t i = 0; i < touched.size(); i++) {
            bits[touched[i]] = 0;
        }
        std::vector<size_t>().swap(touched);
    }

    void release() {
        releaseData();
        std::vector<uint64_t>().swap(bits);
        std::vector<size_t>().swap(blockOffset);
    }

    // the block base plus the lengths gathered before it, so per id offsets are never materialised
    size_t offsetOf(DBReader<DBKeyType> &reader, DBLocalId id) const {
        const size_t word = id / BLOCK;
        const size_t bit = id % BLOCK;
        size_t offset = blockOffset[word];
        uint64_t before = bits[word] & ((bit == 0) ? 0ULL : ((1ULL << bit) - 1ULL));
        while (before != 0) {
            const size_t k = static_cast<size_t>(__builtin_ctzll(before));
            offset += reader.getSeqLen(word * BLOCK + k);
            before &= before - 1;
        }
        return offset;
    }

    const char *at(DBReader<DBKeyType> &reader, DBLocalId id) const {
        return data + offsetOf(reader, id);
    }
};

// marks the members of multi-member runs and prefix sums the blocks, returning the bytes they need
static size_t planArena(DBReader<DBKeyType> &reader, MemberArena &arena, const HashEntry *entries,
                        size_t entryCount, size_t idSpace, int threads) {
    const size_t blocks = (idSpace + MemberArena::BLOCK - 1) / MemberArena::BLOCK;
    // sized once for the whole id space, then reused: a window marks a small, scattered subset of it
    if (arena.bits.size() != blocks) {
        arena.bits.assign(blocks, 0);
        arena.blockOffset.resize(blocks);
    }
    std::vector<std::vector<size_t> > perThread(static_cast<size_t>(threads));
#pragma omp parallel num_threads(threads)
    {
        int t = 0;
#ifdef OPENMP
        t = omp_get_thread_num();
#endif
        std::vector<size_t> &mine = perThread[t];
#pragma omp for schedule(dynamic, 4096)
        for (size_t pos = 0; pos < entryCount; ++pos) {
            const bool aloneBefore = (pos == 0) || (entries[pos - 1].hash != entries[pos].hash);
            const bool aloneAfter = (pos + 1 == entryCount) || (entries[pos + 1].hash != entries[pos].hash);
            if (aloneBefore && aloneAfter) {
                continue;
            }
            const DBLocalId id = entries[pos].id;
            const size_t word = id / MemberArena::BLOCK;
            // exactly one thread turns a block from empty to non-empty, so this lists it without a dedup pass
            const uint64_t was = __sync_fetch_and_or(&arena.bits[word], 1ULL << (id % MemberArena::BLOCK));
            if (was == 0) {
                mine.push_back(word);
            }
        }
    }
    size_t marked = 0;
    for (size_t t = 0; t < perThread.size(); t++) {
        marked += perThread[t].size();
    }
    if (marked == 0) {
        return 0;
    }
    arena.touched.resize(marked);
    size_t at = 0;
    for (size_t t = 0; t < perThread.size(); t++) {
        memcpy(&arena.touched[at], perThread[t].data(), perThread[t].size() * sizeof(size_t));
        at += perThread[t].size();
        std::vector<size_t>().swap(perThread[t]);
    }
    // ascending order makes the arena layout independent of how the threads split the entries
    SORT_PARALLEL(arena.touched.begin(), arena.touched.end());

    const size_t used = arena.touched.size();
    const size_t stripe = (used + threads - 1) / static_cast<size_t>(threads);
    std::vector<size_t> stripeBase(static_cast<size_t>(threads) + 1, 0);
#pragma omp parallel for schedule(static, 1) num_threads(threads)
    for (int t = 0; t < threads; ++t) {
        size_t sum = 0;
        for (size_t i = t * stripe; i < std::min(used, (t + 1) * stripe); ++i) {
            const size_t word = arena.touched[i];
            arena.blockOffset[word] = sum;   // stripe-local offset, rebased once the stripes are summed
            uint64_t set = arena.bits[word];
            while (set != 0) {
                const size_t k = static_cast<size_t>(__builtin_ctzll(set));
                sum += reader.getSeqLen(word * MemberArena::BLOCK + k);
                set &= set - 1;
            }
        }
        stripeBase[t + 1] = sum;
    }
    for (int t = 0; t < threads; ++t) {
        stripeBase[t + 1] += stripeBase[t];
    }
#pragma omp parallel for schedule(static, 1) num_threads(threads)
    for (int t = 0; t < threads; ++t) {
        for (size_t i = t * stripe; i < std::min(used, (t + 1) * stripe); ++i) {
            arena.blockOffset[arena.touched[i]] += stripeBase[t];
        }
    }
    return stripeBase[threads];
}

// one sequential pass over the db, each thread copying its own contiguous id range into the arena
static void fillArena(DBReader<DBKeyType> &reader, MemberArena &arena, size_t total, int threads) {
    arena.data = new(std::nothrow) char[total];
    Util::checkAllocation(arena.data, "Can not allocate member arena in clusthashfast");
    const size_t used = arena.touched.size();
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        // schedule(static) gives each thread one contiguous, disjoint word range and byte range
#pragma omp for schedule(static)
        for (size_t i = 0; i < used; ++i) {
            const size_t word = arena.touched[i];
            uint64_t set = arena.bits[word];
            size_t offset = arena.blockOffset[word];
            while (set != 0) {
                const size_t k = static_cast<size_t>(__builtin_ctzll(set));
                const DBLocalId id = static_cast<DBLocalId>(word * MemberArena::BLOCK + k);
                const size_t length = reader.getSeqLen(id);
                memcpy(arena.data + offset, reader.getData(id, thread_idx), length);
                offset += length;
                set &= set - 1;
            }
        }
    }
}

// a window is a maximal set of whole runs whose members fit the budget, so a run is never split
static void planWindows(DBReader<DBKeyType> &reader, const HashEntry *entries, size_t entryCount,
                        size_t budget, std::vector<size_t> &bounds) {
    bounds.clear();
    bounds.push_back(0);
    size_t used = 0;
    size_t pos = 0;
    while (pos < entryCount) {
        const size_t runEnd = hashRunEnd(entries, entryCount, pos);
        size_t runBytes = 0;
        if (runEnd - pos > 1) {
            for (size_t k = pos; k < runEnd; ++k) {
                runBytes += reader.getSeqLen(entries[k].id);
            }
        }
        if (used > 0 && used + runBytes > budget) {
            bounds.push_back(pos);
            used = 0;
        }
        used += runBytes;
        pos = runEnd;
    }
    bounds.push_back(entryCount);
}

// a run is owned by the chunk that starts it, so skip the tail of the run the previous chunk owns
static size_t firstOwnedRun(const HashEntry *entries, size_t chunkBegin, size_t chunkEnd) {
    size_t runBegin = chunkBegin;
    while (runBegin < chunkEnd && runBegin > 0 && entries[runBegin - 1].hash == entries[runBegin].hash) {
        runBegin++;
    }
    return runBegin;
}

static const char *memberSeq(DBReader<DBKeyType> &reader, const MemberArena *arena, DBLocalId id,
                             unsigned int thread_idx) {
    return (arena != NULL) ? arena->at(reader, id) : reader.getData(id, thread_idx);
}

// the first unclaimed entry of the run represents it and claims every unclaimed entry it covers
static void clusterHashRun(DBReader<DBKeyType> &reader, DBWriter &writer, const HashEntry *run, size_t runSize,
                           float seqIdThr, ClusterWorker &worker, unsigned int thread_idx,
                           const MemberArena *arena) {
    worker.counts.runs++;
    // the common case at scale: a run of one touches no sequence data
    if (runSize == 1) {
        const DBKeyType representativeKey = reader.getDbKey(run[0].id);
        worker.beginCluster(representativeKey);
        worker.writeCluster(writer, representativeKey, thread_idx);
        return;
    }

    worker.claimed.assign(runSize, 0);
    for (size_t i = 0; i < runSize; i++) {
        if (worker.claimed[i]) {
            continue;
        }
        worker.claimed[i] = 1;
        const DBLocalId queryId = run[i].id;
        const unsigned int queryLength = reader.getSeqLen(queryId);
        const DBKeyType representativeKey = reader.getDbKey(queryId);
        // the seed passes the threshold against every member, so every cluster mode elects this seed
        worker.beginCluster(representativeKey);

        const char *querySeq = NULL;
        for (size_t j = i + 1; j < runSize; j++) {
            if (worker.claimed[j]) {
                continue;
            }
            const DBLocalId targetId = run[j].id;
            // the length is folded into the hash, but a hash collision can still differ in length
            if (reader.getSeqLen(targetId) != queryLength) {
                continue;
            }
            if (querySeq == NULL) {
                // arena bytes stay put, but getData hands back the buffer the target read reuses
                if (arena != NULL) {
                    querySeq = arena->at(reader, queryId);
                } else {
                    worker.querySeq.assign(reader.getData(queryId, thread_idx), queryLength);
                    querySeq = worker.querySeq.data();
                }
            }
            const char *targetSeq = memberSeq(reader, arena, targetId, thread_idx);
            const unsigned int distance =
                DistanceCalculator::computeInverseHammingDistance(querySeq, targetSeq, queryLength);
            const float seqId = static_cast<float>(distance) / static_cast<float>(queryLength);
            if (seqId >= seqIdThr) {
                worker.addMember(reader.getDbKey(targetId));
                worker.claimed[j] = 1;
            }
        }
        worker.writeCluster(writer, representativeKey, thread_idx);
    }
    worker.trimClaimed();
}

static ClusterCounts clusterHashRuns(DBReader<DBKeyType> &reader, DBWriter &writer, const HashEntry *entries,
                                     size_t entryCount, float seqIdThr, bool showProgress,
                                     const MemberArena *arena, int threads) {
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
            size_t pos = firstOwnedRun(entries, chunkBegin, chunkEnd);
            while (pos < chunkEnd) {
                const size_t runEnd = hashRunEnd(entries, entryCount, pos);
                clusterHashRun(reader, writer, entries + pos, runEnd - pos, seqIdThr, worker, thread_idx, arena);
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
    return counts;
}

// gathers the run members this partition compares, then clusters with every read coming from memory
static ClusterCounts clusterPartition(DBReader<DBKeyType> &reader, DBWriter &writer, const HashEntry *entries,
                                      size_t entryCount, size_t idSpace, float seqIdThr, bool showProgress,
                                      bool gather, size_t arenaBudget, int threads, MemberArena &arena) {
    if (gather == false) {
        return clusterHashRuns(reader, writer, entries, entryCount, seqIdThr, showProgress, NULL, threads);
    }

    const size_t total = planArena(reader, arena, entries, entryCount, idSpace, threads);
    if (total <= arenaBudget) {
        if (total > 0) {
            fillArena(reader, arena, total, threads);
        }
        ClusterCounts once = clusterHashRuns(reader, writer, entries, entryCount, seqIdThr, showProgress,
                                             (total > 0) ? &arena : NULL, threads);
        arena.releaseData();
        return once;
    }
    arena.releaseData();

    // the members do not fit at once, so split the hash range and gather one window per pass
    std::vector<size_t> bounds;
    planWindows(reader, entries, entryCount, arenaBudget, bounds);
    Debug(Debug::INFO) << "Gathering " << total << " byte of run members in " << (bounds.size() - 1)
                       << " passes\n";
    ClusterCounts counts;
    for (size_t window = 0; window + 1 < bounds.size(); ++window) {
        const size_t begin = bounds[window];
        const size_t size = bounds[window + 1] - begin;
        const size_t windowTotal = planArena(reader, arena, entries + begin, size, idSpace, threads);
        // a single run wider than the whole budget cannot be gathered, so read that one from the db
        const bool gathered = (windowTotal > 0 && windowTotal <= arenaBudget);
        if (gathered) {
            fillArena(reader, arena, windowTotal, threads);
        }
        counts.add(clusterHashRuns(reader, writer, entries + begin, size, seqIdThr, showProgress,
                                   gathered ? &arena : NULL, threads));
        arena.releaseData();
    }
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
    Debug::Progress progress(dbSize);
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
            if (showProgress) {
                progress.updateProgress(id);
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
    // an explicit count is how the partitioning is exercised on inputs small enough to need one
    const char *env = getenv("MMSEQS_CLUSTHASHFAST_PARTITIONS");
    if (env != NULL && *env != '\0') {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && parsed >= 1 && parsed <= CLUSTHASHFAST_MAX_PARTITIONS) {
            unsigned int partitions = 1;
            while (partitions < parsed) {
                partitions *= 2;
            }
            return partitions;
        }
        Debug(Debug::WARNING) << "Ignoring invalid MMSEQS_CLUSTHASHFAST_PARTITIONS=" << env << "\n";
    }
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
    Debug::Progress progress(dbSize);
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        Sequence *seq = isNuclInput ? NULL : new Sequence(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
#pragma omp for schedule(static, scanChunk)
        for (size_t id = 0; id < dbSize; ++id) {
            if (showProgress) {
                progress.updateProgress(id);
            }
            entries[id].hash = hashOf(reader, id, seq, thread_idx);
            entries[id].id = static_cast<DBLocalId>(id);
        }
        delete seq;
    }
}

static ClusterCounts clusterSequences(const Parameters &par, DBReader<DBKeyType> &reader, DBWriter &writer,
                                      BaseMatrix *subMat, bool isNuclInput) {
    ClusterCounts counts;
    const size_t dbSize = reader.getSize();
    if (dbSize == 0) {
        return counts;
    }

    const bool showProgress = (Debug::debugLevel >= Debug::INFO);
    const std::string tmpPrefix = std::string(writer.getDataFileName());
    // the sorted entries and the gathered members are live at the same time, so each gets half
    const size_t halfBudget = std::max<size_t>(Util::computeMemory(par.splitMemoryLimit) / 2, 1);
    // gathering is a second sequential pass, so it only pays once the db stops fitting in memory
    const bool gather = (reader.getDataSize() + dbSize * sizeof(HashEntry)
                         > Util::computeMemory(par.splitMemoryLimit));
    const unsigned int partitionCount = hashPartitionCount(dbSize, halfBudget);
    if (hashScanIsSequential(reader)) {
        reader.setSequentialAdvice();
    }

    if (partitionCount == 1) {
        Debug(Debug::INFO) << "Hashing sequences...\n";
        // HashEntry is trivial, so this allocation is not written until the pass below touches it
        HashEntry *entries = new(std::nothrow) HashEntry[dbSize];
        Util::checkAllocation(entries, "Cannot allocate hash entry memory in clusthashfast");
        hashAllSequences(reader, entries, dbSize, isNuclInput, subMat, par.maxSeqLen, showProgress,
                         par.threads);
        Debug(Debug::INFO) << "Sort sequence hashes...\n";
        SORT_PARALLEL(entries, entries + dbSize, HashEntry::compareByHashAndId);
        Debug(Debug::INFO) << "Cluster equal length sequences...\n";
        MemberArena arena;
        counts = clusterPartition(reader, writer, entries, dbSize, dbSize, par.seqIdThr, showProgress,
                                  gather, halfBudget, par.threads, arena);
        delete[] entries;
        Debug(Debug::INFO) << "Found " << counts.runs << " unique hash/length groups\n";
        return counts;
    }

    Debug(Debug::INFO) << "Hashing sequences into " << partitionCount << " partitions...\n";
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

    std::vector<HashEntry> part;
    // the id-space bitmap and block offsets are sized by the db, so every partition reuses them
    MemberArena arena;
    for (unsigned int partition = 0; partition < partitionCount; partition++) {
        const size_t partSize = parts.counts[partition];
        if (partSize == 0) {
            continue;
        }
        part.resize(partSize);
        FILE *in = fopen(parts.names[partition].c_str(), "rb");
        if (in == NULL || fread(part.data(), sizeof(HashEntry), partSize, in) != partSize) {
            Debug(Debug::ERROR) << "Cannot read hash partition " << parts.names[partition] << "\n";
            EXIT(EXIT_FAILURE);
        }
        fclose(in);
        FileUtil::remove(parts.names[partition].c_str());

        SORT_PARALLEL(part.data(), part.data() + partSize, HashEntry::compareByHashAndId);
        const ClusterCounts partCounts = clusterPartition(reader, writer, part.data(), partSize, dbSize,
                                                          par.seqIdThr, showProgress, gather, halfBudget,
                                                          par.threads, arena);
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

    const ClusterCounts counts = clusterSequences(par, reader, writer, subMat, isNuclInput);

    reader.close();
    if (subMat != NULL) {
        delete subMat;
    }
    writer.close();

    Debug(Debug::INFO) << counts.clusters << " clusters, " << counts.merged << " sequences merged by hamming distance\n";
    return EXIT_SUCCESS;
}
