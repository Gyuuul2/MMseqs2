#include "DBWriter.h"
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
#include <string>
#include <vector>

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

static void hashSequences(DBReader<DBKeyType> &reader, HashEntry *entries, bool isNuclInput,
                          BaseMatrix *subMat, size_t maxSeqLen, bool showProgress, int threads) {
    const size_t dbSize = reader.getSize();
    const size_t scanChunk = hashScanBlock(reader, dbSize, threads);
    Debug::Progress progress(dbSize);
#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        if (isNuclInput) {
#pragma omp for schedule(static, scanChunk)
            for (size_t id = 0; id < dbSize; ++id) {
                if (showProgress) {
                    progress.updateProgress();
                }
                const size_t length = reader.getSeqLen(id);
                const size_t hash = hashNucleotideSequence(reader.getData(id, thread_idx), length);
                entries[id].hash = hashWithLength(hash, length);
                entries[id].id = static_cast<DBLocalId>(id);
            }
        } else {
            Sequence seq(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
#pragma omp for schedule(static, scanChunk)
            for (size_t id = 0; id < dbSize; ++id) {
                if (showProgress) {
                    progress.updateProgress();
                }
                const size_t length = reader.getSeqLen(id);
                seq.mapSequence(id, 0, reader.getData(id, thread_idx), length);
                entries[id].hash = hashWithLength(Util::hash(seq.numSequence, seq.L), length);
                entries[id].id = static_cast<DBLocalId>(id);
            }
        }
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
                           float seqIdThr, ClusterWorker &worker, unsigned int thread_idx) {
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
                worker.querySeq.assign(reader.getData(queryId, thread_idx), queryLength);
                queryCopied = true;
            }
            const char *targetSeq = reader.getData(targetId, thread_idx);
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
                                     size_t dbSize, size_t runCount, float seqIdThr, bool showProgress) {
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
                clusterHashRun(reader, writer, entries + pos, runEnd - pos, seqIdThr, worker, thread_idx);
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

static ClusterCounts clusterSequences(const Parameters &par, DBReader<DBKeyType> &reader, DBWriter &writer,
                                      BaseMatrix *subMat, bool isNuclInput) {
    ClusterCounts counts;
    const size_t dbSize = reader.getSize();
    if (dbSize == 0) {
        return counts;
    }

    HashEntry *entries = new(std::nothrow) HashEntry[dbSize];
    Util::checkAllocation(entries, "Can not allocate hash entry memory in clusthashfast");
    const bool showProgress = (Debug::debugLevel >= Debug::INFO);

    Debug(Debug::INFO) << "Hashing sequences...\n";
    const bool sequentialHashScan = hashScanIsSequential(reader);
    if (sequentialHashScan) {
        reader.setSequentialAdvice();
    }
    hashSequences(reader, entries, isNuclInput, subMat, par.maxSeqLen, showProgress, par.threads);
    if (sequentialHashScan) {
        // hash order is random to the end, and the default heuristic keeps issuing a full readahead
        // window because 128 threads racing on the shared mmap_miss counter never let it saturate
        for (size_t fileIdx = 0; fileIdx < reader.getDataFileCnt(); fileIdx++) {
            Util::madviseLogged(reader.getDataForFile(fileIdx), reader.getDataSizeForFile(fileIdx),
                                POSIX_MADV_RANDOM, "clusthashfast clustering");
        }
    }

    Debug(Debug::INFO) << "Sort sequence hashes...\n";
    SORT_PARALLEL(entries, entries + dbSize, HashEntry::compareByHashAndId);

    const size_t runCount = countHashRuns(entries, dbSize);
    Debug(Debug::INFO) << "Found " << runCount << " unique hash/length groups\n";

    Debug(Debug::INFO) << "Cluster equal length sequences...\n";
    counts = clusterHashRuns(reader, writer, entries, dbSize, runCount, par.seqIdThr, showProgress);

    delete[] entries;
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
    // only preload what can stay resident; computeMemory returns --split-memory-limit, not a memory size
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
