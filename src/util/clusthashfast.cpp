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
// feed enough offset-ordered ids to DBReader that adjacent short entries can collapse into large reads
static const size_t CLUSTHASHFAST_GATHER_BATCH_IDS = 65536;
// progress is observability, not work: billions of atomic/progress calls can otherwise dominate scans
static const size_t CLUSTHASHFAST_PROGRESS_STEP = 1ull << 20;
// floor under the proportional cache reserve, so a tiny machine still keeps room for the stream
static const size_t CLUSTHASHFAST_MIN_CACHE_RESERVE = 1ull * 1024 * 1024 * 1024;

// Test-only I/O knobs. The defaults reproduce the current behavior, so one binary can A/B.
static size_t clusthashfastEnvSize(const char *name, size_t fallback, size_t minimum) {
    const char *value = getenv(name);
    if (value == NULL || *value == '\0') {
        return fallback;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum) {
        Debug(Debug::WARNING) << "Ignoring invalid " << name << "=" << value << "\n";
        return fallback;
    }
    return static_cast<size_t>(parsed);
}

static bool clusthashfastEnvFlag(const char *name, bool fallback) {
    const char *value = getenv(name);
    if (value == NULL || *value == '\0') {
        return fallback;
    }
    if (strcmp(value, "1") == 0) {
        return true;
    }
    if (strcmp(value, "0") == 0) {
        return false;
    }
    Debug(Debug::WARNING) << "Ignoring invalid " << name << "=" << value << "\n";
    return fallback;
}

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
    std::vector<const char *> arenaSeq;
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

// a run is owned by the chunk that starts it, so skip the tail of the run the previous chunk owns
static size_t firstOwnedRun(const HashEntry *entries, size_t chunkBegin, size_t chunkEnd) {
    size_t runBegin = chunkBegin;
    while (runBegin < chunkEnd && runBegin > 0 && entries[runBegin - 1].hash == entries[runBegin].hash) {
        runBegin++;
    }
    return runBegin;
}

// only multi-member runs are gathered. The bitmap makes the gather order ascending by id without
// storing one extra id/offset pair per member, which is important once ids themselves are 64 bit.
struct MemberArena {
    static const size_t BLOCK = 64;

    std::vector<uint64_t> bits;         // one bit per sequence that belongs to a multi-member run
    std::vector<size_t> blockOffset;    // byte offset of the first gathered member in each 64-id word
    std::vector<size_t> touched;        // ascending words marked by the current window
    char *data;
    size_t capacity;

    MemberArena() : data(NULL), capacity(0) {}

    ~MemberArena() {
        release();
    }

    void clearMarks() {
        for (size_t i = 0; i < touched.size(); i++) {
            bits[touched[i]] = 0;
        }
        // keep the allocation: every gather window needs roughly the same block list again
        touched.clear();
    }

    void ensureData(size_t bytes) {
        if (bytes <= capacity) {
            return;
        }
        char *next = new(std::nothrow) char[bytes];
        Util::checkAllocation(next, "Can not allocate member arena in clusthashfast");
        delete[] data;
        data = next;
        capacity = bytes;
    }

    // release the byte arena between hash partitions, but keep the id-space tables for reuse
    void releaseData() {
        delete[] data;
        data = NULL;
        capacity = 0;
        clearMarks();
    }

    void release() {
        releaseData();
        std::vector<uint64_t>().swap(bits);
        std::vector<size_t>().swap(blockOffset);
        std::vector<size_t>().swap(touched);
    }

    // the block base plus the lengths gathered before it; clusterHashRun calls this once per member,
    // then keeps the resulting pointers in thread-local scratch for the O(n^2) comparison loop.
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

// Mark whole hash runs, not individual positions. Relaxed atomics are sufficient: the only shared
// fact is the final bitmap, and exactly one successful 0->nonzero transition records each word.
static size_t planArena(DBReader<DBKeyType> &reader, MemberArena &arena, const HashEntry *entries,
                        size_t entryCount, size_t idSpace, int threads) {
    const size_t blocks = (idSpace + MemberArena::BLOCK - 1) / MemberArena::BLOCK;
    if (arena.bits.size() != blocks) {
        arena.bits.assign(blocks, 0);
        arena.blockOffset.resize(blocks);
    }

    std::vector<std::vector<size_t> > perThread(static_cast<size_t>(threads));
    const size_t chunkCount = (entryCount + CLUSTHASHFAST_CHUNK_RECORDS - 1) / CLUSTHASHFAST_CHUNK_RECORDS;
#pragma omp parallel num_threads(threads)
    {
        int t = 0;
#ifdef OPENMP
        t = omp_get_thread_num();
#endif
        std::vector<size_t> &mine = perThread[t];
#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
            const size_t chunkBegin = chunk * CLUSTHASHFAST_CHUNK_RECORDS;
            const size_t chunkEnd = std::min(entryCount, chunkBegin + CLUSTHASHFAST_CHUNK_RECORDS);
            size_t pos = firstOwnedRun(entries, chunkBegin, chunkEnd);
            while (pos < chunkEnd) {
                const size_t runEnd = hashRunEnd(entries, entryCount, pos);
                if (runEnd - pos > 1) {
                    // ids are ascending inside a hash run, so collapse all bits in the same 64-id
                    // word before the atomic update. Dense duplicate runs can turn dozens of locked
                    // operations into one without any extra storage.
                    size_t word = static_cast<size_t>(entries[pos].id) / MemberArena::BLOCK;
                    uint64_t mask = 0;
                    for (size_t k = pos; k < runEnd; ++k) {
                        const DBLocalId id = entries[k].id;
                        const size_t nextWord = static_cast<size_t>(id) / MemberArena::BLOCK;
                        if (nextWord != word) {
                            const uint64_t was = __atomic_fetch_or(&arena.bits[word], mask, __ATOMIC_RELAXED);
                            if (was == 0) {
                                mine.push_back(word);
                            }
                            word = nextWord;
                            mask = 0;
                        }
                        mask |= 1ULL << (static_cast<size_t>(id) % MemberArena::BLOCK);
                    }
                    const uint64_t was = __atomic_fetch_or(&arena.bits[word], mask, __ATOMIC_RELAXED);
                    if (was == 0) {
                        mine.push_back(word);
                    }
                }
                pos = runEnd;
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

    // Sorting random touched-word ids is good when sparse. Once enough of the bitmap is populated,
    // two linear bitmap scans are cheaper than an O(n log n) sort and produce sorted words directly.
    if (marked > blocks / 32) {
        for (size_t t = 0; t < perThread.size(); ++t) {
            std::vector<size_t>().swap(perThread[t]);
        }
        std::vector<size_t> count(static_cast<size_t>(threads) + 1, 0);
#pragma omp parallel for schedule(static, 1) num_threads(threads)
        for (int t = 0; t < threads; ++t) {
            const size_t begin = (blocks / static_cast<size_t>(threads)) * static_cast<size_t>(t)
                               + std::min(static_cast<size_t>(t), blocks % static_cast<size_t>(threads));
            const size_t end = (blocks / static_cast<size_t>(threads)) * static_cast<size_t>(t + 1)
                             + std::min(static_cast<size_t>(t + 1), blocks % static_cast<size_t>(threads));
            size_t local = 0;
            for (size_t word = begin; word < end; ++word) {
                local += (arena.bits[word] != 0);
            }
            count[t + 1] = local;
        }
        for (int t = 0; t < threads; ++t) {
            count[t + 1] += count[t];
        }
        arena.touched.resize(count[threads]);
#pragma omp parallel for schedule(static, 1) num_threads(threads)
        for (int t = 0; t < threads; ++t) {
            const size_t begin = (blocks / static_cast<size_t>(threads)) * static_cast<size_t>(t)
                               + std::min(static_cast<size_t>(t), blocks % static_cast<size_t>(threads));
            const size_t end = (blocks / static_cast<size_t>(threads)) * static_cast<size_t>(t + 1)
                             + std::min(static_cast<size_t>(t + 1), blocks % static_cast<size_t>(threads));
            size_t out = count[t];
            for (size_t word = begin; word < end; ++word) {
                if (arena.bits[word] != 0) {
                    arena.touched[out++] = word;
                }
            }
        }
    } else {
        arena.touched.resize(marked);
        size_t out = 0;
        for (size_t t = 0; t < perThread.size(); t++) {
            if (perThread[t].empty() == false) {
                memcpy(&arena.touched[out], perThread[t].data(), perThread[t].size() * sizeof(size_t));
                out += perThread[t].size();
            }
        }
        SORT_PARALLEL(arena.touched.begin(), arena.touched.end());
    }

    const size_t used = arena.touched.size();
    const size_t stripe = (used + threads - 1) / static_cast<size_t>(threads);
    std::vector<size_t> stripeBase(static_cast<size_t>(threads) + 1, 0);
#pragma omp parallel for schedule(static, 1) num_threads(threads)
    for (int t = 0; t < threads; ++t) {
        size_t sum = 0;
        for (size_t i = static_cast<size_t>(t) * stripe;
             i < std::min(used, static_cast<size_t>(t + 1) * stripe); ++i) {
            const size_t word = arena.touched[i];
            arena.blockOffset[word] = sum;
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
        for (size_t i = static_cast<size_t>(t) * stripe;
             i < std::min(used, static_cast<size_t>(t + 1) * stripe); ++i) {
            arena.blockOffset[arena.touched[i]] += stripeBase[t];
        }
    }
    return stripeBase[threads];
}

// The hash pass wants mmap/readahead, but the sparse gather is exactly the opposite. Once gather is
// needed, evict the one-pass hash-scan cache and switch to O_DIRECT. DBReader coalesces neighbouring
// aligned spans, so several short entries in the same block cost one read instead of one read each.
static bool clusthashfastSetGatherIo(DBReader<DBKeyType> &reader) {
    static bool warned = false;
    if (reader.isDirectIo()) {
        return true;
    }
    reader.dropCacheAll();
    if (reader.setIoDirect(true) == false) {
        if (warned == false) {
            warned = true;
            Debug(Debug::WARNING) << "Gather stays on the mapping, this reader cannot use descriptors\n";
        }
        return false;
    }
    return true;
}

// Copy offset-ordered members into the arena. Threads own contiguous byte ranges rather than equal
// block counts, which balances variable sequence lengths while preserving monotonically increasing IO.
static void fillArena(DBReader<DBKeyType> &reader, MemberArena &arena, size_t total, int threads) {
    arena.ensureData(total);
    const size_t used = arena.touched.size();
    if (used == 0) {
        return;
    }

    std::vector<size_t> split(static_cast<size_t>(threads) + 1, 0);
    split[threads] = used;
    for (int t = 1; t < threads; ++t) {
        const size_t q = total / static_cast<size_t>(threads);
        const size_t r = total % static_cast<size_t>(threads);
        const size_t target = q * static_cast<size_t>(t)
                            + (r * static_cast<size_t>(t)) / static_cast<size_t>(threads);
        const std::vector<size_t>::const_iterator it = std::lower_bound(
            arena.touched.begin(), arena.touched.end(), target,
            [&](size_t word, size_t byteOffset) { return arena.blockOffset[word] < byteOffset; });
        split[t] = static_cast<size_t>(it - arena.touched.begin());
    }

#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        const size_t begin = split[thread_idx];
        const size_t end = split[thread_idx + 1];
        std::vector<size_t> ids;
        std::vector<size_t> dst;
        ids.reserve(CLUSTHASHFAST_GATHER_BATCH_IDS);
        dst.reserve(CLUSTHASHFAST_GATHER_BATCH_IDS);

        const auto flushBatch = [&]() {
            size_t pos = 0;
            while (pos < ids.size()) {
                const size_t got = reader.loadBatch(&ids[pos], ids.size() - pos, thread_idx);
                if (got == 0) {
                    Debug(Debug::ERROR) << "DBReader::loadBatch returned zero entries in clusthashfast\n";
                    EXIT(EXIT_FAILURE);
                }
                for (size_t slot = 0; slot < got; ++slot) {
                    memcpy(arena.data + dst[pos + slot], reader.batchAt(thread_idx, slot),
                           reader.getSeqLen(ids[pos + slot]));
                }
                pos += got;
            }
            ids.clear();
            dst.clear();
        };

        for (size_t i = begin; i < end; ++i) {
            const size_t word = arena.touched[i];
            uint64_t set = arena.bits[word];
            size_t offset = arena.blockOffset[word];
            while (set != 0) {
                const size_t k = static_cast<size_t>(__builtin_ctzll(set));
                const DBLocalId id = static_cast<DBLocalId>(word * MemberArena::BLOCK + k);
                ids.push_back(static_cast<size_t>(id));
                dst.push_back(offset);
                offset += reader.getSeqLen(id);
                set &= set - 1;
                if (ids.size() >= CLUSTHASHFAST_GATHER_BATCH_IDS) {
                    flushBatch();
                }
            }
        }
        if (ids.empty() == false) {
            flushBatch();
        }
    }
}

// Build approximate windows in O(number_of_windows), not by rescanning every run serially. The
// parallel arena planner measures the real bytes later and oversized windows are split on demand.
static void planWindows(const HashEntry *entries, size_t entryCount, size_t windowCount,
                        std::vector<size_t> &bounds) {
    bounds.clear();
    bounds.push_back(0);
    windowCount = std::max<size_t>(windowCount, 1);
    const size_t q = entryCount / windowCount;
    const size_t r = entryCount % windowCount;
    size_t last = 0;
    for (size_t window = 1; window < windowCount; ++window) {
        size_t pos = q * window + std::min(window, r);
        while (pos < entryCount && pos > 0 && entries[pos - 1].hash == entries[pos].hash) {
            ++pos;
        }
        if (pos > last && pos < entryCount) {
            bounds.push_back(pos);
            last = pos;
        }
    }
    bounds.push_back(entryCount);
}

// Split an oversized window at a hash boundary. Returning begin/end means one hash run itself is
// wider than the normal arena budget and the caller has to use the emergency budget or DB reads.
static size_t splitWindowAtRunBoundary(const HashEntry *entries, size_t begin, size_t end) {
    if (end - begin < 2) {
        return end;
    }
    size_t mid = begin + (end - begin) / 2;
    size_t forward = mid;
    while (forward < end && forward > begin && entries[forward - 1].hash == entries[forward].hash) {
        ++forward;
    }
    if (forward < end) {
        return forward;
    }
    size_t backward = mid;
    while (backward > begin && entries[backward - 1].hash == entries[backward].hash) {
        --backward;
    }
    return (backward > begin) ? backward : end;
}

// the first unclaimed entry of the run represents it and claims every unclaimed entry it covers
static void clusterHashRun(DBReader<DBKeyType> &reader, DBWriter &writer, const HashEntry *run, size_t runSize,
                           float seqIdThr, ClusterWorker &worker, unsigned int thread_idx,
                           const MemberArena *arena) {
    worker.counts.runs++;
    if (runSize == 1) {
        const DBKeyType representativeKey = reader.getDbKey(run[0].id);
        worker.beginCluster(representativeKey);
        worker.writeCluster(writer, representativeKey, thread_idx);
        return;
    }

    // Length and arena-pointer lookup used to sit in the inner comparison loop. Resolve both once per
    // member so a large hash collision spends its O(n^2) work only on the distance calculation.
    worker.memberLength.resize(runSize);
    if (arena != NULL) {
        worker.arenaSeq.resize(runSize);
    } else {
        worker.arenaSeq.clear();
    }
    for (size_t k = 0; k < runSize; ++k) {
        worker.memberLength[k] = static_cast<unsigned int>(reader.getSeqLen(run[k].id));
        if (arena != NULL) {
            worker.arenaSeq[k] = arena->at(reader, run[k].id);
        }
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

        const char *querySeq = (arena != NULL) ? worker.arenaSeq[i] : NULL;
        for (size_t j = i + 1; j < runSize; j++) {
            if (worker.claimed[j] || worker.memberLength[j] != queryLength) {
                continue;
            }
            if (querySeq == NULL) {
                // descriptor reads reuse a per-thread bounce buffer, so pin the query before target IO
                worker.querySeq.assign(reader.getData(queryId, thread_idx), queryLength);
                querySeq = worker.querySeq.data();
            }
            const char *targetSeq = (arena != NULL)
                ? worker.arenaSeq[j]
                : reader.getData(run[j].id, thread_idx);
            const unsigned int distance =
                DistanceCalculator::computeInverseHammingDistance(querySeq, targetSeq, queryLength);
            const float seqId = static_cast<float>(distance) / static_cast<float>(queryLength);
            if (seqId >= seqIdThr) {
                worker.addMember(reader.getDbKey(run[j].id));
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
                                      bool gather, size_t arenaBudget, size_t directArenaBudget, int threads,
                                      MemberArena &arena) {
    if (gather == false) {
        return clusterHashRuns(reader, writer, entries, entryCount, seqIdThr, showProgress, NULL, threads);
    }

    Timer planTimer;
    const size_t total = planArena(reader, arena, entries, entryCount, idSpace, threads);
    Debug(Debug::INFO) << "Member arena plan: " << total << " byte in " << planTimer.lap() << "\n";
    if (total == 0) {
        arena.clearMarks();
        return clusterHashRuns(reader, writer, entries, entryCount, seqIdThr, showProgress, NULL, threads);
    }

    const bool descriptorGather = clusthashfastSetGatherIo(reader);
    const size_t budget = descriptorGather ? directArenaBudget : arenaBudget;
    Timer gatherTimer;
    if (total <= budget) {
        fillArena(reader, arena, total, threads);
        Debug(Debug::INFO) << "Gathered " << total << " byte of run members in one pass: "
                           << gatherTimer.lap() << "\n";
        ClusterCounts once = clusterHashRuns(reader, writer, entries, entryCount, seqIdThr, showProgress,
                                             &arena, threads);
        arena.releaseData();
        return once;
    }
    arena.clearMarks();

    // Descriptor windows do not reread unrelated file pages, so an approximate split is enough.
    // Measure each range in parallel and bisect only the rare skewed range that exceeds the budget.
    const size_t wantedWindows = std::max<size_t>(total / budget + ((total % budget) != 0), 1);
    std::vector<size_t> bounds;
    planWindows(entries, entryCount, wantedWindows, bounds);
    struct Window { size_t begin; size_t end; };
    std::vector<Window> windows;
    windows.reserve(bounds.size());
    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        Window w = {bounds[i], bounds[i + 1]};
        windows.push_back(w);
    }
    Debug(Debug::INFO) << "Gathering " << total << " byte of run members in about " << windows.size()
                       << " passes\n";

    ClusterCounts counts;
    size_t window = 0;
    while (window < windows.size()) {
        const size_t begin = windows[window].begin;
        const size_t end = windows[window].end;
        const size_t size = end - begin;
        const size_t windowTotal = planArena(reader, arena, entries + begin, size, idSpace, threads);

        if (windowTotal > budget) {
            const size_t split = splitWindowAtRunBoundary(entries, begin, end);
            if (split > begin && split < end) {
                arena.clearMarks();
                windows[window].end = split;
                Window tail = {split, end};
                // result order is irrelevant here, so append instead of O(n) vector insertion
                windows.push_back(tail);
                continue;
            }
        }

        // A single run can be wider than the normal descriptor window. Gather it if the larger safe
        // mapping budget can hold it; otherwise fall back to direct per-entry reads for that run only.
        const bool gathered = windowTotal > 0
            && (windowTotal <= budget || (descriptorGather && windowTotal <= arenaBudget));
        if (gathered) {
            fillArena(reader, arena, windowTotal, threads);
        }
        Debug(Debug::INFO) << "Pass " << (window + 1) << "/" << windows.size() << " gathered "
                           << windowTotal << " byte: " << gatherTimer.lap() << "\n";
        counts.add(clusterHashRuns(reader, writer, entries + begin, size, seqIdThr, showProgress,
                                   gathered ? &arena : NULL, threads));
        arena.clearMarks();
        ++window;
    }
    arena.releaseData();
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
    const size_t pendingBytes = clusthashfastEnvSize("MMSEQS_CLUSTHASHFAST_PENDING_BYTES",
                                                     CLUSTHASHFAST_PENDING_BYTES, sizeof(HashEntry));
    const size_t flush = std::max<size_t>(64, pendingBytes / (partitionCount * sizeof(HashEntry)));
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
            if (showProgress && ((id & (CLUSTHASHFAST_PROGRESS_STEP - 1)) == 0 || id + 1 == dbSize)) {
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
            if (showProgress && ((id & (CLUSTHASHFAST_PROGRESS_STEP - 1)) == 0 || id + 1 == dbSize)) {
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
    const size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    // a partition holds this much of the entry array, the rest of memory is free for the members
    const size_t entryBudget = std::max<size_t>(memoryLimit / 2, 1);
    // gathering is a second sequential pass, so it only pays once the db stops fitting in memory
    const bool gather = (reader.getDataSize() + dbSize * sizeof(HashEntry) > memoryLimit);
    const unsigned int partitionCount = hashPartitionCount(dbSize, entryBudget);
    // the gather streams the db through page cache next to the arena, so keep a proportional slice
    const size_t liveEntryBytes = dbSize * sizeof(HashEntry) / partitionCount;
    const size_t cacheReserve = std::max<size_t>(CLUSTHASHFAST_MIN_CACHE_RESERVE, memoryLimit / 12);
    const size_t unreserved = (memoryLimit > liveEntryBytes + cacheReserve)
        ? (memoryLimit - liveEntryBytes - cacheReserve) : 0;
    // the floor keeps the arena from collapsing to nothing just above the reserve threshold
    const size_t arenaBudget = std::max(unreserved, std::max<size_t>(memoryLimit / 4, 1));
    // an arena this size is only reachable on the descriptor path, where windows are free
    const size_t directArenaBudget = std::max<size_t>(memoryLimit / 16, 1);
    Debug(Debug::INFO) << "Memory limit " << memoryLimit << " byte, entries " << liveEntryBytes
                       << " byte, member arena budget " << arenaBudget << " byte, "
                       << directArenaBudget << " byte on descriptors\n";
    if (hashScanIsSequential(reader)
        && clusthashfastEnvFlag("MMSEQS_CLUSTHASHFAST_SEQUENTIAL_ADVICE", true)) {
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
        Timer sortTimer;
        SORT_PARALLEL(entries, entries + dbSize, HashEntry::compareByHashAndId);
        Debug(Debug::INFO) << "Time for sorting hashes: " << sortTimer.lap() << "\n";
        Debug(Debug::INFO) << "Cluster equal length sequences...\n";
        MemberArena arena;
        counts = clusterPartition(reader, writer, entries, dbSize, dbSize, par.seqIdThr, showProgress,
                                  gather, arenaBudget, directArenaBudget, par.threads, arena);
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
        const ClusterCounts partCounts = clusterPartition(reader, writer, part.data(), partSize, dbSize,
                                                          par.seqIdThr, showProgress, gather, arenaBudget,
                                                          directArenaBudget, par.threads, arena);
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
    // The hash pass is mmap/sequential; if a huge-db gather follows, keep an fd so its cache can be
    // dropped before switching to O_DIRECT instead of competing with the member arena.
    reader.setIoCacheAdvice(true);
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
