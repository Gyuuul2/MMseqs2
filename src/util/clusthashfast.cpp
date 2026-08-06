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
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
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

struct ClusterCounts {
    size_t clusters;
    size_t merged;

    ClusterCounts() : clusters(0), merged(0) {}
};

struct HashEntry {
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

// hashing reads in offset order and wants sequential advice; clustering reads in hash order and must not
static bool hashScanIsSequential(DBReader<DBKeyType> &reader) {
    return reader.getDataFileCnt() == 1 && reader.isSortedByOffset();
}

// Nothing reads these bytes again until the members are staged, and the cache they leave behind is
// what makes the kernel swap the sort array out instead. glibc turns posix_madvise(DONTNEED) into a
// no-op because POSIX forbids it from changing semantics, so drop the mappings with madvise directly;
// while the pages stay mapped fadvise cannot reclaim them either.
static void dropDataCache(DBReader<DBKeyType> &reader) {
#ifdef MADV_DONTNEED
    for (size_t fileIdx = 0; fileIdx < reader.getDataFileCnt(); fileIdx++) {
        madvise(reader.getDataForFile(fileIdx), reader.getDataSizeForFile(fileIdx), MADV_DONTNEED);
    }
#endif
#ifdef HAVE_POSIX_FADVISE
    std::vector<std::string> names = reader.getDataFileNames();
    for (size_t fileIdx = 0; fileIdx < names.size(); fileIdx++) {
        int fd = ::open(names[fileIdx].c_str(), O_RDONLY);
        if (fd < 0) {
            continue;
        }
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
#endif
}

static void dropSequentialAdvice(DBReader<DBKeyType> &reader) {
    // hash order is random to the end, and the default heuristic keeps issuing a full readahead
    // window because 128 threads racing on the shared mmap_miss counter never let it saturate
    for (size_t fileIdx = 0; fileIdx < reader.getDataFileCnt(); fileIdx++) {
        Util::madviseLogged(reader.getDataForFile(fileIdx), reader.getDataSizeForFile(fileIdx),
                            POSIX_MADV_RANDOM, "clusthashfast clustering");
    }
}

// a block wider than any readahead window keeps neighbouring threads from allocating the same folios
static size_t hashScanBlock(DBReader<DBKeyType> &reader, size_t dbSize, int threads) {
    const size_t DEFAULT_BLOCK = 1000;
    const size_t MIN_BLOCKS_PER_THREAD = 16;
    const size_t TARGET_SPAN = 64 * 1024 * 1024;
    if (threads < 2 || dbSize == 0 || hashScanIsSequential(reader) == false) {
        return DEFAULT_BLOCK;
    }
    size_t fileSize = reader.getDataSizeForFile(0);
    size_t lastEnd = std::min(reader.getIndex(dbSize - 1)->offset + reader.getEntryLen(dbSize - 1), fileSize);
    if (fileSize == 0 || lastEnd == 0) {
        return DEFAULT_BLOCK;
    }
    size_t stride = std::max<size_t>(lastEnd / dbSize, 1);
    size_t wanted = (TARGET_SPAN + stride - 1) / stride;
    size_t balanced = dbSize / (static_cast<size_t>(threads) * MIN_BLOCKS_PER_THREAD);
    return std::max(std::min(wanted, balanced), DEFAULT_BLOCK);
}

// the single definition of the hash, so the array pass and the partitioned pass cannot drift apart
static size_t hashOf(DBReader<DBKeyType> &reader, size_t id, Sequence *seq, unsigned int thread_idx) {
    const size_t length = reader.getSeqLen(id);
    const char *data = reader.getData(id, thread_idx);
    if (seq == NULL) {
        return hashWithLength(hashNucleotideSequence(data, length), length);
    }
    seq->mapSequence(id, 0, data, length);
    return hashWithLength(Util::hash(seq->numSequence, seq->L), length);
}

static void hashSequences(DBReader<DBKeyType> &reader, HashEntry *entries, bool isNuclInput,
                          BaseMatrix *subMat, size_t maxSeqLen, bool showProgress, int threads) {
    const size_t dbSize = reader.getSize();
    const size_t scanChunk = hashScanBlock(reader, dbSize, threads);
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

static size_t countHashRuns(const HashEntry *entries, size_t dbSize) {
    size_t runCount = 1;
#pragma omp parallel for schedule(static) reduction(+:runCount)
    for (size_t id = 1; id < dbSize; ++id) {
        if (entries[id - 1].hash != entries[id].hash) {
            runCount++;
        }
    }
    return runCount;
}

static size_t hashRunEnd(const HashEntry *entries, size_t dbSize, size_t runBegin) {
    const size_t hash = entries[runBegin].hash;
    size_t runEnd = runBegin + 1;
    while (runEnd < dbSize && entries[runEnd].hash == hash) {
        runEnd++;
    }
    return runEnd;
}

// Only members of runs longer than one are ever read for their bytes, and in hash order those reads
// are scattered over the whole sequence db. Copying just them out in id order gives a file small
// enough to stay in the page cache, which turns the clustering pass back into cached reads.
struct MemberStore {
    static const size_t BLOCK = 64;

    std::vector<uint64_t> bits;         // one bit per sequence, set when its run has other members
    std::vector<size_t> blockOffset;    // byte offset of the first stored member of each 64 id block
    std::string path;
    char *data;
    size_t dataSize;
    int fd;

    MemberStore() : data(NULL), dataSize(0), fd(-1) {}

    ~MemberStore() {
        if (data != NULL) {
            munmap(data, dataSize);
        }
        if (fd >= 0) {
            close(fd);
        }
        if (path.empty() == false) {
            FileUtil::remove(path.c_str());
        }
    }

    bool holds(DBLocalId id) const {
        return (bits[id / BLOCK] >> (id % BLOCK)) & 1ULL;
    }

    // the offset is the block base plus the lengths of the stored members before this one, so the
    // per sequence offsets never have to be materialised
    const char *get(DBReader<DBKeyType> &reader, DBLocalId id) const {
        const size_t word = id / BLOCK;
        const size_t bit = id % BLOCK;
        size_t offset = blockOffset[word];
        uint64_t before = bits[word] & ((bit == 0) ? 0ULL : ((1ULL << bit) - 1ULL));
        while (before != 0) {
            const size_t k = static_cast<size_t>(__builtin_ctzll(before));
            offset += reader.getSeqLen(word * BLOCK + k);
            before &= before - 1;
        }
        return data + offset;
    }
};

// runs of one never need bytes, so mark only the members of longer runs
static size_t markRunMembers(const HashEntry *entries, size_t entryCount, size_t idSpace,
                             std::vector<uint64_t> &bits, int threads) {
    bits.assign((idSpace + MemberStore::BLOCK - 1) / MemberStore::BLOCK, 0);
    size_t marked = 0;
#pragma omp parallel for schedule(dynamic, 4096) num_threads(threads) reduction(+:marked)
    for (size_t pos = 0; pos < entryCount; ++pos) {
        const bool aloneBefore = (pos == 0) || (entries[pos - 1].hash != entries[pos].hash);
        const bool aloneAfter = (pos + 1 == entryCount) || (entries[pos + 1].hash != entries[pos].hash);
        if (aloneBefore && aloneAfter) {
            continue;
        }
        const DBLocalId id = entries[pos].id;
        __sync_fetch_and_or(&bits[id / MemberStore::BLOCK], 1ULL << (id % MemberStore::BLOCK));
        marked++;
    }
    return marked;
}

// One pwrite per 64 ids is 156 million syscalls at ten billion sequences, and with 128 threads that is
// where the whole machine ends up: all system time, no user time. schedule(static) hands each thread a
// contiguous word range and blockOffset is a prefix sum, so a thread's blocks are contiguous on disk.
static const size_t CLUSTHASHFAST_STAGE_BUFFER = 16 * 1024 * 1024;

static void flushStageBuffer(int fd, const char *path, std::vector<char> &batch, size_t offset) {
    if (batch.empty()) {
        return;
    }
    if (pwrite(fd, batch.data(), batch.size(), static_cast<off_t>(offset))
            != static_cast<ssize_t>(batch.size())) {
        Debug(Debug::ERROR) << "Cannot write " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    batch.clear();
}

// one sequential pass over the sequence db, each thread copying its own contiguous id range
static bool buildMemberStore(DBReader<DBKeyType> &reader, MemberStore &store, const HashEntry *entries,
                             size_t entryCount, size_t idSpace, const std::string &tmpPath, int threads) {
    const size_t marked = markRunMembers(entries, entryCount, idSpace, store.bits, threads);
    if (marked == 0) {
        return false;
    }
    const size_t blocks = store.bits.size();
    store.blockOffset.assign(blocks, 0);

    // per block byte size first, so an exclusive prefix sum gives every block its offset
#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t word = 0; word < blocks; ++word) {
        size_t sum = 0;
        uint64_t set = store.bits[word];
        while (set != 0) {
            const size_t k = static_cast<size_t>(__builtin_ctzll(set));
            sum += reader.getSeqLen(word * MemberStore::BLOCK + k);
            set &= set - 1;
        }
        store.blockOffset[word] = sum;
    }
    size_t total = 0;
    for (size_t word = 0; word < blocks; ++word) {
        const size_t size = store.blockOffset[word];
        store.blockOffset[word] = total;
        total += size;
    }

    store.path = tmpPath;
    store.fd = ::open(store.path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (store.fd < 0 || ftruncate(store.fd, static_cast<off_t>(total)) != 0) {
        Debug(Debug::WARNING) << "Cannot stage " << store.path << ", clustering reads the db directly\n";
        return false;
    }
    Debug(Debug::INFO) << "Staging " << marked << " run members (" << total << " byte) for clustering\n";

#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        std::vector<char> batch;
        batch.reserve(CLUSTHASHFAST_STAGE_BUFFER + MemberStore::BLOCK * reader.getMaxSeqLen());
        size_t batchOffset = 0;
#pragma omp for schedule(static)
        for (size_t word = 0; word < blocks; ++word) {
            uint64_t set = store.bits[word];
            if (set == 0) {
                continue;
            }
            // a gap would mean this word does not follow the last one written, which only the loop
            // boundary can produce, so start a new batch there
            if (batch.empty() == false && store.blockOffset[word] != batchOffset + batch.size()) {
                flushStageBuffer(store.fd, store.path.c_str(), batch, batchOffset);
            }
            if (batch.empty()) {
                batchOffset = store.blockOffset[word];
            }
            while (set != 0) {
                const size_t k = static_cast<size_t>(__builtin_ctzll(set));
                const DBLocalId id = static_cast<DBLocalId>(word * MemberStore::BLOCK + k);
                const size_t length = reader.getSeqLen(id);
                const char *src = reader.getData(id, thread_idx);
                batch.insert(batch.end(), src, src + length);
                set &= set - 1;
            }
            if (batch.size() >= CLUSTHASHFAST_STAGE_BUFFER) {
                flushStageBuffer(store.fd, store.path.c_str(), batch, batchOffset);
            }
        }
        flushStageBuffer(store.fd, store.path.c_str(), batch, batchOffset);
    }

    store.data = static_cast<char *>(mmap(NULL, total, PROT_READ, MAP_SHARED, store.fd, 0));
    if (store.data == MAP_FAILED) {
        store.data = NULL;
        Debug(Debug::WARNING) << "Cannot map " << store.path << ", clustering reads the db directly\n";
        return false;
    }
    store.dataSize = total;
    Util::madviseLogged(store.data, total, POSIX_MADV_RANDOM, "clusthashfast member store");
    return true;
}

// a run is owned by the chunk that starts it, so skip the tail of the run the previous chunk owns
static size_t firstOwnedRun(const HashEntry *entries, size_t chunkBegin, size_t chunkEnd) {
    size_t runBegin = chunkBegin;
    while (runBegin < chunkEnd && runBegin > 0 && entries[runBegin - 1].hash == entries[runBegin].hash) {
        runBegin++;
    }
    return runBegin;
}

// the first unclaimed entry of the run represents it and claims every unclaimed entry it covers
static void clusterHashRun(DBReader<DBKeyType> &reader, DBWriter &writer, const HashEntry *run, size_t runSize,
                           float seqIdThr, ClusterWorker &worker, unsigned int thread_idx,
                           const MemberStore *store) {
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

        bool queryCopied = false;
        for (size_t j = i + 1; j < runSize; j++) {
            if (worker.claimed[j]) {
                continue;
            }
            const DBLocalId targetId = run[j].id;
            // the length is folded into the hash, but a hash collision can still differ in length
            if (reader.getSeqLen(targetId) != queryLength) {
                continue;
            }
            // getData hands back a shared per-thread buffer, so copy the query out before the target
            if (queryCopied == false) {
                const char *querySeq = (store != NULL) ? store->get(reader, queryId)
                                                       : reader.getData(queryId, thread_idx);
                worker.querySeq.assign(querySeq, queryLength);
                queryCopied = true;
            }
            const char *targetSeq = (store != NULL) ? store->get(reader, targetId)
                                                    : reader.getData(targetId, thread_idx);
            const unsigned int distance =
                DistanceCalculator::computeInverseHammingDistance(worker.querySeq.data(), targetSeq, queryLength);
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
                                     size_t dbSize, size_t runCount, float seqIdThr, bool showProgress,
                                     const MemberStore *store) {
    const size_t chunkCount = (dbSize + CLUSTHASHFAST_CHUNK_RECORDS - 1) / CLUSTHASHFAST_CHUNK_RECORDS;
    Debug::Progress progress(runCount);
    size_t totalClusters = 0;
    size_t totalMerged = 0;
#pragma omp parallel reduction(+:totalClusters) reduction(+:totalMerged)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        ClusterWorker worker;

#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
            const size_t chunkBegin = chunk * CLUSTHASHFAST_CHUNK_RECORDS;
            const size_t chunkEnd = std::min(dbSize, chunkBegin + CLUSTHASHFAST_CHUNK_RECORDS);
            size_t pos = firstOwnedRun(entries, chunkBegin, chunkEnd);
            while (pos < chunkEnd) {
                const size_t runEnd = hashRunEnd(entries, dbSize, pos);
                if (showProgress) {
                    progress.updateProgress();
                }
                clusterHashRun(reader, writer, entries + pos, runEnd - pos, seqIdThr, worker, thread_idx, store);
                pos = runEnd;
            }
        }
        totalClusters += worker.counts.clusters;
        totalMerged += worker.counts.merged;
    }

    ClusterCounts counts;
    counts.clusters = totalClusters;
    counts.merged = totalMerged;
    return counts;
}

// Sorting every hash at once needs 16 byte per sequence resident while the sort streams the whole
// array, which at ten billion sequences competes with the reader index for memory and swaps. Runs are
// independent and never straddle a hash, so partitioning on the high hash bits and sorting one
// partition at a time gives the same grouping with a bounded footprint. Ascending partitions also
// keep the runs in the same global hash order they had before, so the output order is unchanged.
static unsigned int hashBucketCount(const Parameters &par, size_t dbSize) {
    // an explicit count is how the partitioning is exercised on inputs small enough to need one
    const char *env = getenv("MMSEQS_CLUSTHASHFAST_PARTITIONS");
    if (env != NULL && *env != '\0') {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && parsed >= 1 && parsed <= 4096) {
            unsigned int buckets = 1;
            while (buckets < parsed) {
                buckets *= 2;
            }
            return buckets;
        }
        Debug(Debug::WARNING) << "Ignoring invalid MMSEQS_CLUSTHASHFAST_PARTITIONS=" << env << "\n";
    }
    const size_t needed = dbSize * sizeof(HashEntry);
    // a large reader index can leave computeMemory with almost nothing, which would ask for the
    // maximum partition count on every default run, so keep a floor it cannot fall through
    const size_t budget = std::max(Util::computeMemory(par.splitMemoryLimit),
                                   Util::getTotalSystemMemory() / 4);
    if (needed <= budget) {
        return 1;
    }
    unsigned int buckets = 1;
    while (buckets < 4096 && needed / buckets > budget) {
        buckets *= 2;
    }
    return buckets;
}

struct HashBucketWriter {
    std::vector<FILE *> files;
    std::vector<std::string> names;
    std::vector<size_t> counts;
    unsigned int shift;

    HashBucketWriter() : shift(0) {}

    ~HashBucketWriter() {
        close();
        for (size_t b = 0; b < names.size(); b++) {
            if (FileUtil::fileExists(names[b].c_str())) {
                FileUtil::remove(names[b].c_str());
            }
        }
    }

    bool open(const std::string &prefix, unsigned int buckets) {
        unsigned int bits = 0;
        while ((1u << bits) < buckets) {
            bits++;
        }
        shift = 64 - bits;
        files.assign(buckets, NULL);
        counts.assign(buckets, 0);
        for (unsigned int b = 0; b < buckets; b++) {
            names.push_back(prefix + "." + SSTR(b));
            files[b] = fopen(names[b].c_str(), "wb");
            if (files[b] == NULL) {
                return false;
            }
        }
        return true;
    }

    void close() {
        for (size_t b = 0; b < files.size(); b++) {
            if (files[b] != NULL) {
                fclose(files[b]);
                files[b] = NULL;
            }
        }
    }

    unsigned int bucketOf(size_t hash) const {
        return static_cast<unsigned int>(hash >> shift);
    }
};

// per thread write buffer budget, split across the partitions
static const size_t CLUSTHASHFAST_PENDING_BYTES = 8 * 1024 * 1024;

static bool flushBucket(HashBucketWriter &buckets, std::vector<std::mutex> &locks, unsigned int bucket,
                        std::vector<HashEntry> &pending) {
    std::lock_guard<std::mutex> lock(locks[bucket]);
    const bool ok = fwrite(pending.data(), sizeof(HashEntry), pending.size(), buckets.files[bucket])
                    == pending.size();
    buckets.counts[bucket] += pending.size();
    pending.clear();
    return ok;
}

// hashes straight into the partitions, so the full array is never resident
static bool hashIntoBuckets(DBReader<DBKeyType> &reader, HashBucketWriter &buckets, bool isNuclInput,
                            BaseMatrix *subMat, size_t maxSeqLen, bool showProgress, int threads) {
    const size_t dbSize = reader.getSize();
    const size_t scanChunk = hashScanBlock(reader, dbSize, threads);
    const unsigned int bucketCount = static_cast<unsigned int>(buckets.files.size());
    // partitioning exists to bound memory, so the write buffers get a fixed budget rather than a
    // fixed depth: a per bucket depth would grow the footprint by the very factor it is dividing by
    const size_t flush = std::max<size_t>(64, CLUSTHASHFAST_PENDING_BYTES
                                              / (bucketCount * sizeof(HashEntry)));
    std::vector<std::mutex> locks(bucketCount);
    Debug::Progress progress(dbSize);
    bool ok = true;
#pragma omp parallel num_threads(threads) reduction(&&:ok)
    {
        bool threadOk = true;
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        std::vector<std::vector<HashEntry> > pending(bucketCount);
        Sequence *seq = isNuclInput ? NULL : new Sequence(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
#pragma omp for schedule(static, scanChunk)
        for (size_t id = 0; id < dbSize; ++id) {
            if (showProgress) {
                progress.updateProgress(id);
            }
            HashEntry entry;
            entry.hash = hashOf(reader, id, seq, thread_idx);
            entry.id = static_cast<DBLocalId>(id);
            const unsigned int b = buckets.bucketOf(entry.hash);
            pending[b].push_back(entry);
            if (pending[b].size() >= flush) {
                threadOk = flushBucket(buckets, locks, b, pending[b]) && threadOk;
            }
        }
        for (unsigned int b = 0; b < bucketCount; b++) {
            if (pending[b].empty() == false) {
                threadOk = flushBucket(buckets, locks, b, pending[b]) && threadOk;
            }
        }
        delete seq;
        ok = ok && threadOk;
    }
    return ok;
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
    // dropping or copying a cache that would have been reused only forces re-reads, so weigh the
    // data against the sort array it would otherwise be evicting
    const bool cacheContended = reader.getDataSize() + dbSize * sizeof(HashEntry)
                                > Util::computeMemory(par.splitMemoryLimit);
    // staging only buys anything once the scattered reads actually miss; on a db the page cache
    // already holds it is a second copy of the data and a wasted sweep
    const bool canStage = (reader.getDataFileCnt() == 1 && reader.isCompressed() == 0 && cacheContended);
    const unsigned int bucketCount = hashBucketCount(par, dbSize);
    const bool sequentialHashScan = hashScanIsSequential(reader);
    if (sequentialHashScan) {
        reader.setSequentialAdvice();
    }

    if (bucketCount == 1) {
        HashEntry *entries = new(std::nothrow) HashEntry[dbSize];
        Util::checkAllocation(entries, "Can not allocate hash entry memory in clusthashfast");

        Debug(Debug::INFO) << "Hashing sequences...\n";
        hashSequences(reader, entries, isNuclInput, subMat, par.maxSeqLen, showProgress, par.threads);
        if (cacheContended) {
            dropDataCache(reader);
        }

        Debug(Debug::INFO) << "Sort sequence hashes...\n";
        SORT_PARALLEL(entries, entries + dbSize, HashEntry::compareByHashAndId);

        const size_t runCount = countHashRuns(entries, dbSize);
        Debug(Debug::INFO) << "Found " << runCount << " unique hash/length groups\n";

        // staging the run members keeps the clustering reads inside a file that fits the page cache
        MemberStore store;
        const bool staged = canStage
            && buildMemberStore(reader, store, entries, dbSize, dbSize, tmpPrefix + ".members", par.threads);
        if (staged == false && sequentialHashScan) {
            dropSequentialAdvice(reader);
        }

        Debug(Debug::INFO) << "Cluster equal length sequences...\n";
        counts = clusterHashRuns(reader, writer, entries, dbSize, runCount, par.seqIdThr, showProgress,
                                 staged ? &store : NULL);
        delete[] entries;
        return counts;
    }

    Debug(Debug::INFO) << "Hashing sequences into " << bucketCount << " partitions...\n";
    HashBucketWriter buckets;
    if (buckets.open(tmpPrefix + ".hashpart", bucketCount) == false) {
        Debug(Debug::ERROR) << "Cannot open hash partitions under " << tmpPrefix << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (hashIntoBuckets(reader, buckets, isNuclInput, subMat, par.maxSeqLen, showProgress, par.threads) == false) {
        Debug(Debug::ERROR) << "Cannot write hash partitions under " << tmpPrefix << "\n";
        EXIT(EXIT_FAILURE);
    }
    buckets.close();
    if (cacheContended) {
        dropDataCache(reader);
    }

    size_t totalRuns = 0;
    std::vector<HashEntry> part;
    for (unsigned int b = 0; b < bucketCount; b++) {
        const size_t partSize = buckets.counts[b];
        if (partSize == 0) {
            continue;
        }
        part.resize(partSize);
        FILE *in = fopen(buckets.names[b].c_str(), "rb");
        if (in == NULL || fread(part.data(), sizeof(HashEntry), partSize, in) != partSize) {
            Debug(Debug::ERROR) << "Cannot read hash partition " << buckets.names[b] << "\n";
            EXIT(EXIT_FAILURE);
        }
        fclose(in);
        FileUtil::remove(buckets.names[b].c_str());

        SORT_PARALLEL(part.data(), part.data() + partSize, HashEntry::compareByHashAndId);
        const size_t runCount = countHashRuns(part.data(), partSize);
        totalRuns += runCount;

        MemberStore store;
        const bool staged = canStage
            && buildMemberStore(reader, store, part.data(), partSize, dbSize, tmpPrefix + ".members", par.threads);
        const ClusterCounts partCounts = clusterHashRuns(reader, writer, part.data(), partSize, runCount,
                                                        par.seqIdThr, showProgress, staged ? &store : NULL);
        counts.clusters += partCounts.clusters;
        counts.merged += partCounts.merged;
        Debug(Debug::INFO) << "Partition " << (b + 1) << "/" << bucketCount << ": " << runCount
                           << " groups, " << partCounts.clusters << " clusters\n";
    }
    Debug(Debug::INFO) << "Found " << totalRuns << " unique hash/length groups\n";
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
