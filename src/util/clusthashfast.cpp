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
#include <string>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

// clusthash writes an alignment DB with one entry per sequence and needs a following clust step.
// This writes the clustering directly, like align2clust does, which removes the N-entry alignment
// DB, its merge, the formatted alignment columns and the clust invocation.
//
// The grouping is always the same greedy scan, so the cluster count never depends on
// --cluster-mode; the mode only decides which member of a formed cluster is its representative.
// That mirrors the old pipeline, where clusthash fixed the grouping into disjoint stars and clust
// only relabelled them.
namespace {

// chunks are many per thread so a few huge groups cannot unbalance the fan-out
const size_t CLUSTHASH_CHUNK_RECORDS = 65536;
// a single huge group must not leave every thread holding a giant buffer
const size_t CLUSTHASH_MAX_RETAINED_RESULT = 4 * 1024 * 1024;
const size_t CLUSTHASH_MAX_RETAINED_MEMBERS = 1024 * 1024;
// set cover compares every pair of a formed cluster, so bound it and keep the greedy pick above
const size_t CLUSTHASH_MAX_SETCOVER_CLUSTER = 4096;

struct HashEntry {
    size_t hash;
    DBLocalId id;

    static bool compare(const HashEntry &lhs, const HashEntry &rhs) {
        if (lhs.hash < rhs.hash) return true;
        if (rhs.hash < lhs.hash) return false;
        return lhs.id < rhs.id;
    }
};

// folding the length in means a group holds only equal length sequences, so the length scan
// inside a group disappears and the groups get smaller
inline size_t mixHashAndLength(size_t hash, size_t length) {
    return hash ^ (length + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2));
}

inline void appendKey(std::string &out, DBKeyType key, char *buffer) {
    char *end = Itoa::u64toa_sse2(static_cast<uint64_t>(key), buffer);
    out.append(buffer, end - buffer - 1);
    out.push_back('\n');
}

inline void resetResult(std::string &result) {
    if (result.capacity() > CLUSTHASH_MAX_RETAINED_RESULT) {
        std::string().swap(result);
        result.reserve(4096);
    } else {
        result.clear();
    }
}

}

int clusthashfast(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.alphabetSize = MultiParam<NuclAA<int> >(NuclAA<int>(Parameters::CLUST_HASH_DEFAULT_ALPH_SIZE, 5));
    par.seqIdThr = static_cast<float>(Parameters::CLUST_HASH_DEFAULT_MIN_SEQ_ID) / 100.0f;
    par.parseParameters(argc, argv, command, true, 0, 0);

    if (par.clusteringMode == Parameters::CONNECTED_COMPONENT) {
        Debug(Debug::ERROR) << "clusthashfast does not implement --cluster-mode 1; use 0 or 2\n";
        EXIT(EXIT_FAILURE);
    }
    const bool wantSetCover = (par.clusteringMode == Parameters::SET_COVER);

    // NOSORT keeps DBReader from building the id2local/local2id maps, which cost 2 x 8 byte per
    // sequence; nothing here needs a sorted access order.
    DBReader<DBKeyType> reader(par.db1.c_str(), par.db1Index.c_str(), par.threads,
                               DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX);
    reader.open(DBReader<DBKeyType>::NOSORT);
    if (par.preloadMode != Parameters::PRELOAD_MODE_MMAP) {
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

    const size_t dbSize = reader.getSize();
    if (dbSize == 0) {
        writer.close();
        reader.close();
        if (subMat != NULL) {
            delete subMat;
        }
        return EXIT_SUCCESS;
    }

    HashEntry *entries = new(std::nothrow) HashEntry[dbSize];
    Util::checkAllocation(entries, "Can not allocate hash entry memory in clusthashfast");
    const bool showProgress = (Debug::debugLevel >= Debug::INFO);

    Debug(Debug::INFO) << "Hashing sequences...\n";
    Debug::Progress hashProgress(dbSize);
#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        if (isNuclInput) {
#pragma omp for schedule(dynamic, 1000)
            for (size_t id = 0; id < dbSize; ++id) {
                if (showProgress) {
                    hashProgress.updateProgress();
                }
                const char *data = reader.getData(id, thread_idx);
                const size_t length = reader.getSeqLen(id);
                const size_t A = 31;
                size_t h1 = 0;
                size_t h2 = 0;
                for (size_t i = 0; i < length; ++i) {
                    h1 = ((h1 * A) + data[i]);
                    h2 = ((h2 * A) + Orf::complement(data[length - i - 1]));
                }
                entries[id].hash = mixHashAndLength(std::min(h1, h2), length);
                entries[id].id = static_cast<DBLocalId>(id);
            }
        } else {
            Sequence seq(par.maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
#pragma omp for schedule(dynamic, 1000)
            for (size_t id = 0; id < dbSize; ++id) {
                if (showProgress) {
                    hashProgress.updateProgress();
                }
                const size_t length = reader.getSeqLen(id);
                seq.mapSequence(id, 0, reader.getData(id, thread_idx), length);
                entries[id].hash = mixHashAndLength(Util::hash(seq.numSequence, seq.L), length);
                entries[id].id = static_cast<DBLocalId>(id);
            }
        }
    }

    Debug(Debug::INFO) << "Sort sequence hashes...\n";
    SORT_PARALLEL(entries, entries + dbSize, HashEntry::compare);

    size_t uniqHashes = 1;
#pragma omp parallel for schedule(static) reduction(+:uniqHashes)
    for (size_t id = 1; id < dbSize; ++id) {
        if (entries[id - 1].hash != entries[id].hash) {
            uniqHashes++;
        }
    }
    Debug(Debug::INFO) << "Found " << uniqHashes << " unique hash/length groups\n";

    const size_t chunkCount = (dbSize + CLUSTHASH_CHUNK_RECORDS - 1) / CLUSTHASH_CHUNK_RECORDS;
    Debug(Debug::INFO) << "Cluster equal length sequences...\n";
    Debug::Progress progress(uniqHashes);
    size_t totalClusters = 0;
    size_t totalMerged = 0;
#pragma omp parallel reduction(+:totalClusters) reduction(+:totalMerged)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        std::string result;
        result.reserve(4096);
        std::vector<unsigned char> claimed;
        std::vector<size_t> cluster;
        std::vector<unsigned int> degree;
        char buffer[32];

#pragma omp for schedule(dynamic, 1)
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
            size_t pos = chunk * CLUSTHASH_CHUNK_RECORDS;
            const size_t chunkEnd = std::min(dbSize, pos + CLUSTHASH_CHUNK_RECORDS);
            // a group is owned by the chunk that starts it, so skip the tail of the previous one
            while (pos < chunkEnd && pos > 0 && entries[pos - 1].hash == entries[pos].hash) {
                pos++;
            }

            while (pos < chunkEnd) {
                const size_t groupBegin = pos;
                const size_t groupHash = entries[groupBegin].hash;
                size_t groupEnd = groupBegin + 1;
                while (groupEnd < dbSize && entries[groupEnd].hash == groupHash) {
                    groupEnd++;
                }
                const size_t groupSize = groupEnd - groupBegin;
                if (showProgress) {
                    progress.updateProgress();
                }

                // the common case at scale: a unique hash reads no sequence data at all
                if (groupSize == 1) {
                    const DBKeyType key = reader.getDbKey(entries[groupBegin].id);
                    resetResult(result);
                    appendKey(result, key, buffer);
                    writer.writeData(result.c_str(), result.length(), key, thread_idx);
                    totalClusters++;
                    pos = groupEnd;
                    continue;
                }

                claimed.assign(groupSize, 0);
                for (size_t i = 0; i < groupSize; i++) {
                    if (claimed[i]) {
                        continue;
                    }
                    // the grouping is mode independent, so the cluster count cannot follow the mode
                    cluster.clear();
                    cluster.push_back(i);
                    claimed[i] = 1;
                    const DBLocalId queryId = entries[groupBegin + i].id;
                    const unsigned int queryLength = reader.getSeqLen(queryId);
                    const char *querySeq = NULL;
                    for (size_t j = i + 1; j < groupSize; j++) {
                        if (claimed[j]) {
                            continue;
                        }
                        const DBLocalId targetId = entries[groupBegin + j].id;
                        // the length is folded into the hash, but a collision can still differ
                        if (reader.getSeqLen(targetId) != queryLength) {
                            continue;
                        }
                        if (querySeq == NULL) {
                            querySeq = reader.getData(queryId, thread_idx);
                        }
                        const char *targetSeq = reader.getData(targetId, thread_idx);
                        const unsigned int distance =
                            DistanceCalculator::computeInverseHammingDistance(querySeq, targetSeq, queryLength);
                        const float seqId = static_cast<float>(distance) / static_cast<float>(queryLength);
                        if (seqId >= par.seqIdThr) {
                            cluster.push_back(j);
                            claimed[j] = 1;
                            totalMerged++;
                        }
                    }

                    // only the representative may follow the mode, and a two member cluster is
                    // always a tie, so there is nothing to decide below three
                    size_t repSlot = 0;
                    if (wantSetCover && cluster.size() > 2 && cluster.size() <= CLUSTHASH_MAX_SETCOVER_CLUSTER) {
                        degree.assign(cluster.size(), 0);
                        for (size_t a = 0; a < cluster.size(); a++) {
                            const char *aSeq = reader.getData(entries[groupBegin + cluster[a]].id, thread_idx);
                            for (size_t b = a + 1; b < cluster.size(); b++) {
                                const char *bSeq = reader.getData(entries[groupBegin + cluster[b]].id, thread_idx);
                                const unsigned int distance =
                                    DistanceCalculator::computeInverseHammingDistance(aSeq, bSeq, queryLength);
                                const float pairSeqId = static_cast<float>(distance) / static_cast<float>(queryLength);
                                if (pairSeqId >= par.seqIdThr) {
                                    degree[a]++;
                                    degree[b]++;
                                }
                            }
                        }
                        // highest degree wins, ties go to the lowest slot, which is the lowest id
                        for (size_t a = 1; a < cluster.size(); a++) {
                            if (degree[a] > degree[repSlot]) {
                                repSlot = a;
                            }
                        }
                    }

                    resetResult(result);
                    const DBKeyType repKey = reader.getDbKey(entries[groupBegin + cluster[repSlot]].id);
                    appendKey(result, repKey, buffer);
                    for (size_t s = 0; s < cluster.size(); s++) {
                        if (s == repSlot) {
                            continue;
                        }
                        appendKey(result, reader.getDbKey(entries[groupBegin + cluster[s]].id), buffer);
                    }
                    writer.writeData(result.c_str(), result.length(), repKey, thread_idx);
                    totalClusters++;
                }
                if (claimed.capacity() > CLUSTHASH_MAX_RETAINED_MEMBERS) {
                    std::vector<unsigned char>().swap(claimed);
                }
                pos = groupEnd;
            }
        }
    }

    delete[] entries;
    reader.close();
    if (subMat != NULL) {
        delete subMat;
    }
    writer.close();

    Debug(Debug::INFO) << totalClusters << " clusters, " << totalMerged << " sequences merged by hamming distance\n";
    return EXIT_SUCCESS;
}
