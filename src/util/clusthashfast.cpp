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
#include <sys/mman.h>

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

static void clusterHashRun(DBReader<DBKeyType> &reader, DBWriter &writer, DBWriter &representativeWriter,
                           const HashEntry *run, size_t runSize,
                           float seqIdThr, ClusterWorker &worker, unsigned int thread_idx) {
    worker.counts.runs++;
    if (runSize == 1) {
        const DBKeyType representativeKey = reader.getDbKey(run[0].id);
        worker.beginCluster(representativeKey);
        worker.writeCluster(writer, representativeWriter, reader, run[0].id, representativeKey, thread_idx);
        return;
    }

    worker.memberLength.resize(runSize);
    for (size_t k = 0; k < runSize; ++k) {
        worker.memberLength[k] = static_cast<unsigned int>(reader.getSeqLen(run[k].id));
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
                worker.querySeq.assign(reader.getData(queryId, thread_idx), queryLength);
                querySeq = worker.querySeq.data();
            }
            const char *targetSeq = reader.getData(run[j].id, thread_idx);
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

    unsigned int partitionOf(size_t hash) const {
        return (bits == 0) ? 0u : static_cast<unsigned int>(hash >> (64 - bits));
    }
};

struct HashGroupFiles {
    std::vector<FILE *> files;
    std::vector<std::string> names;
    std::vector<size_t> counts;

    ~HashGroupFiles() {
        close();
        for (size_t i = 0; i < names.size(); ++i) {
            if (FileUtil::fileExists(names[i].c_str())) {
                FileUtil::remove(names[i].c_str());
            }
        }
    }

    bool open(const std::string &prefix, int lanes) {
        files.assign(static_cast<size_t>(lanes), NULL);
        counts.assign(static_cast<size_t>(lanes), 0);
        for (int lane = 0; lane < lanes; ++lane) {
            names.push_back(prefix + "." + SSTR(lane));
            files[lane] = fopen(names.back().c_str(), "wb");
            if (files[lane] == NULL) {
                return false;
            }
        }
        return true;
    }

    void close() {
        for (size_t i = 0; i < files.size(); ++i) {
            if (files[i] != NULL) {
                fclose(files[i]);
                files[i] = NULL;
            }
        }
    }
};

static std::vector<size_t> makeHashGroupQsplit(const HashEntry *entries, size_t entryCount, size_t idSpace,
                                               int lanes, int threads) {
    const size_t bins = std::max<size_t>(1024, std::min<size_t>(65536, static_cast<size_t>(lanes) * 64));
    std::vector<std::vector<long double> > perThread(static_cast<size_t>(threads),
                                                       std::vector<long double>(bins, 0));
    const size_t chunkCount = (entryCount + CLUSTHASHFAST_CHUNK_RECORDS - 1) / CLUSTHASHFAST_CHUNK_RECORDS;
#pragma omp parallel num_threads(threads)
    {
        unsigned int threadIdx = 0;
#ifdef OPENMP
        threadIdx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        std::vector<long double> &weight = perThread[threadIdx];
#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
            const size_t chunkBegin = chunk * CLUSTHASHFAST_CHUNK_RECORDS;
            const size_t chunkEnd = std::min(entryCount, chunkBegin + CLUSTHASHFAST_CHUNK_RECORDS);
            size_t pos = firstOwnedRun(entries, chunkBegin, chunkEnd);
            while (pos < chunkEnd) {
                const size_t runEnd = hashRunEnd(entries, entryCount, pos);
                const size_t runSize = runEnd - pos;
                if (runSize > 1) {
                    const size_t id = static_cast<size_t>(entries[pos].id);
                    const size_t bin = std::min(bins - 1, (id * bins) / std::max<size_t>(idSpace, 1));
                    weight[bin] += static_cast<long double>(runSize) * static_cast<long double>(runSize);
                }
                pos = runEnd;
            }
        }
    }

    std::vector<long double> weight(bins, 0);
    for (size_t thread = 0; thread < perThread.size(); ++thread) {
        for (size_t bin = 0; bin < bins; ++bin) {
            weight[bin] += perThread[thread][bin];
        }
    }
    long double total = 0;
    for (size_t bin = 0; bin < bins; ++bin) {
        total += weight[bin];
    }

    std::vector<size_t> bounds(static_cast<size_t>(lanes) + 1, idSpace);
    bounds[0] = 0;
    if (total == 0) {
        for (int lane = 1; lane < lanes; ++lane) {
            bounds[lane] = (idSpace * static_cast<size_t>(lane)) / static_cast<size_t>(lanes);
        }
        return bounds;
    }
    long double cumulative = 0;
    size_t bin = 0;
    for (int lane = 1; lane < lanes; ++lane) {
        const long double target = total * static_cast<long double>(lane) / static_cast<long double>(lanes);
        while (bin + 1 < bins && cumulative + weight[bin] < target) {
            cumulative += weight[bin++];
        }
        const size_t boundary = ((bin + 1) * idSpace) / bins;
        bounds[lane] = std::max(bounds[lane - 1], boundary);
    }
    return bounds;
}

static unsigned int hashGroupLane(const std::vector<size_t> &bounds, DBLocalId representativeId) {
    const size_t id = static_cast<size_t>(representativeId);
    const size_t lane = static_cast<size_t>(
        std::upper_bound(bounds.begin(), bounds.end(), id) - bounds.begin()) - 1;
    // an id at or past the planned id space would otherwise land one lane past the last
    return static_cast<unsigned int>(std::min(lane, bounds.size() - 2));
}

static bool spillMultiMemberHashRuns(const HashEntry *entries, size_t entryCount, HashGroupFiles &files,
                                     const std::vector<size_t> &qsplitBounds, int threads) {
    const size_t chunkCount = (entryCount + CLUSTHASHFAST_CHUNK_RECORDS - 1) / CLUSTHASHFAST_CHUNK_RECORDS;
    const size_t lanes = files.files.size();
    const size_t pendingBytes = clusthashfastEnvSize("MMSEQS_CLUSTHASHFAST_PENDING_BYTES",
                                                     CLUSTHASHFAST_PENDING_BYTES, sizeof(HashEntry));
    const size_t flush = std::max<size_t>(64, pendingBytes / (lanes * sizeof(HashEntry)));
    std::vector<std::mutex> locks(lanes);
    bool ok = true;
#pragma omp parallel num_threads(threads) reduction(&&:ok)
    {
        bool threadOk = true;
        std::vector<std::vector<HashEntry> > pending(lanes);
        std::vector<size_t> written(lanes, 0);
#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
            const size_t chunkBegin = chunk * CLUSTHASHFAST_CHUNK_RECORDS;
            const size_t chunkEnd = std::min(entryCount, chunkBegin + CLUSTHASHFAST_CHUNK_RECORDS);
            size_t pos = firstOwnedRun(entries, chunkBegin, chunkEnd);
            while (pos < chunkEnd) {
                const size_t runEnd = hashRunEnd(entries, entryCount, pos);
                if (runEnd - pos > 1) {
                    const unsigned int lane = hashGroupLane(qsplitBounds, entries[pos].id);
                    std::vector<HashEntry> &out = pending[lane];
                    out.insert(out.end(), entries + pos, entries + runEnd);
                    if (out.size() >= flush) {
                        std::lock_guard<std::mutex> lock(locks[lane]);
                        threadOk = fwrite(out.data(), sizeof(HashEntry), out.size(), files.files[lane])
                                   == out.size() && threadOk;
                        written[lane] += out.size();
                        out.clear();
                    }
                }
                pos = runEnd;
            }
        }
        for (size_t lane = 0; lane < lanes; ++lane) {
            if (pending[lane].empty()) {
                continue;
            }
            std::lock_guard<std::mutex> lock(locks[lane]);
            threadOk = fwrite(pending[lane].data(), sizeof(HashEntry), pending[lane].size(), files.files[lane])
                       == pending[lane].size() && threadOk;
            written[lane] += pending[lane].size();
        }
        for (size_t lane = 0; lane < lanes; ++lane) {
            if (written[lane] != 0) {
                std::lock_guard<std::mutex> lock(locks[lane]);
                files.counts[lane] += written[lane];
            }
        }
        ok = ok && threadOk;
    }
    return ok;
}

static ClusterCounts writeSingletonHashRuns(DBReader<DBKeyType> &reader, DBWriter &writer,
                                            DBWriter &representativeWriter, const HashEntry *entries,
                                            size_t entryCount, int threads) {
    const size_t chunkCount = (entryCount + CLUSTHASHFAST_CHUNK_RECORDS - 1) / CLUSTHASHFAST_CHUNK_RECORDS;
    size_t totalClusters = 0;
    size_t totalRuns = 0;
#pragma omp parallel num_threads(threads) reduction(+:totalClusters) reduction(+:totalRuns)
    {
        unsigned int threadIdx = 0;
#ifdef OPENMP
        threadIdx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        ClusterWorker worker;
#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
            const size_t chunkBegin = chunk * CLUSTHASHFAST_CHUNK_RECORDS;
            const size_t chunkEnd = std::min(entryCount, chunkBegin + CLUSTHASHFAST_CHUNK_RECORDS);
            size_t pos = firstOwnedRun(entries, chunkBegin, chunkEnd);
            while (pos < chunkEnd) {
                const size_t runEnd = hashRunEnd(entries, entryCount, pos);
                if (runEnd - pos == 1) {
                    clusterHashRun(reader, writer, representativeWriter, entries + pos, 1, 1.0f, worker,
                                   threadIdx);
                }
                pos = runEnd;
            }
        }
        totalClusters += worker.counts.clusters;
        totalRuns += worker.counts.runs;
    }
    ClusterCounts counts;
    counts.clusters = totalClusters;
    counts.runs = totalRuns;
    return counts;
}

// runs stay indivisible in a lane file, so hashRunEnd recovers them whatever order they were spilled in
static ClusterCounts clusterMappedHashGroupFiles(DBReader<DBKeyType> &reader, DBWriter &writer,
                                                 DBWriter &representativeWriter,
                                                 const HashGroupFiles &files, float seqIdThr,
                                                 int threads) {
    std::vector<ClusterCounts> perThread(static_cast<size_t>(threads));
#pragma omp parallel num_threads(threads)
    {
        unsigned int threadIdx = 0;
#ifdef OPENMP
        threadIdx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        ClusterCounts localCounts;
#pragma omp for schedule(dynamic, 1)
        for (size_t lane = 0; lane < files.names.size(); ++lane) {
            if (files.counts[lane] == 0) {
                FileUtil::remove(files.names[lane].c_str());
            } else {
                FILE *file = fopen(files.names[lane].c_str(), "rb");
                if (file == NULL) {
                    Debug(Debug::ERROR) << "Cannot open hash group " << files.names[lane] << "\n";
                    EXIT(EXIT_FAILURE);
                }
                size_t dataSize = 0;
                HashEntry *entries = static_cast<HashEntry *>(FileUtil::mmapFile(file, &dataSize));
                if (dataSize != files.counts[lane] * sizeof(HashEntry)) {
                    Debug(Debug::ERROR) << "Malformed hash group " << files.names[lane] << "\n";
                    EXIT(EXIT_FAILURE);
                }
                Util::madviseLogged(entries, dataSize, POSIX_MADV_SEQUENTIAL, files.names[lane].c_str());

                ClusterWorker worker;
                size_t pos = 0;
                const size_t entryCount = files.counts[lane];
                while (pos < entryCount) {
                    const size_t runEnd = hashRunEnd(entries, entryCount, pos);
                    clusterHashRun(reader, writer, representativeWriter, entries + pos, runEnd - pos, seqIdThr,
                                   worker, threadIdx);
                    pos = runEnd;
                }
                localCounts.add(worker.counts);

                Util::madviseLogged(entries, dataSize, POSIX_MADV_DONTNEED, files.names[lane].c_str());
                FileUtil::munmapData(entries, dataSize);
                clusthashfastDropFileCache(file, files.names[lane]);
                if (fclose(file) != 0) {
                    Debug(Debug::ERROR) << "Cannot close hash group " << files.names[lane] << "\n";
                    EXIT(EXIT_FAILURE);
                }
                FileUtil::remove(files.names[lane].c_str());
            }
        }
        perThread[threadIdx].add(localCounts);
    }
    ClusterCounts counts;
    for (size_t thread = 0; thread < perThread.size(); ++thread) {
        counts.add(perThread[thread]);
    }
    return counts;
}

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
                                      DBWriter &representativeWriter, BaseMatrix *subMat, bool isNuclInput) {
    ClusterCounts counts;
    const size_t dbSize = reader.getSize();
    if (dbSize == 0) {
        return counts;
    }

    const bool showProgress = (Debug::debugLevel >= Debug::INFO);
    const std::string tmpPrefix = std::string(writer.getDataFileName());
    const size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    // the reader index stays resident at 24 byte per sequence, more than the entry array itself
    const size_t indexBytes = dbSize * sizeof(DBReader<DBKeyType>::Index);
    const size_t usableMemory = (memoryLimit > indexBytes) ? memoryLimit - indexBytes : 0;
    const size_t entryBudget = std::max<size_t>(usableMemory - (usableMemory / 5), 1);
    const unsigned int partitionCount = hashPartitionCount(dbSize, entryBudget);
    const size_t liveEntryBytes = dbSize * sizeof(HashEntry) / partitionCount;
    Debug(Debug::INFO) << "Memory limit " << memoryLimit << " byte, hash entries "
                       << liveEntryBytes << " byte\n";
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
        Debug(Debug::INFO) << "Write non-singleton hash groups...\n";
        HashGroupFiles groups;
        const std::vector<size_t> qsplitBounds = makeHashGroupQsplit(entries, dbSize, dbSize, par.threads,
                                                                       par.threads);
        if (groups.open(tmpPrefix + ".hashgroups", par.threads) == false
            || spillMultiMemberHashRuns(entries, dbSize, groups, qsplitBounds, par.threads) == false) {
            Debug(Debug::ERROR) << "Cannot write hash groups under " << tmpPrefix << "\n";
            EXIT(EXIT_FAILURE);
        }
        groups.close();
        counts.add(writeSingletonHashRuns(reader, writer, representativeWriter, entries, dbSize, par.threads));
        delete[] entries;
        Debug(Debug::INFO) << "Map hash groups and cluster equal length sequences in parallel...\n";
        counts.add(clusterMappedHashGroupFiles(reader, writer, representativeWriter, groups, par.seqIdThr,
                                               par.threads));
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
        HashGroupFiles groups;
        const std::string groupPrefix = tmpPrefix + ".hashgroups." + SSTR(partition);
        const std::vector<size_t> qsplitBounds = makeHashGroupQsplit(part.data(), partSize, dbSize, par.threads,
                                                                       par.threads);
        if (groups.open(groupPrefix, par.threads) == false
            || spillMultiMemberHashRuns(part.data(), partSize, groups, qsplitBounds, par.threads) == false) {
            Debug(Debug::ERROR) << "Cannot write hash groups under " << groupPrefix << "\n";
            EXIT(EXIT_FAILURE);
        }
        groups.close();

        ClusterCounts partCounts = writeSingletonHashRuns(reader, writer, representativeWriter, part.data(),
                                                           partSize, par.threads);
        part.clear();
        part.shrink_to_fit();
        partCounts.add(clusterMappedHashGroupFiles(reader, writer, representativeWriter, groups, par.seqIdThr,
                                                   par.threads));
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
