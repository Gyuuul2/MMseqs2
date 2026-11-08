#include "DBWriter.h"
#include "Util.h"
#include "Parameters.h"
#include "Debug.h"
#include "DBReader.h"
#include "ReducedMatrix.h"
#include "SubstitutionMatrix.h"
#include "DistanceCalculator.h"
#include "FastSort.h"
#include "MMseqsNode.h"
#include "itoa.h"
#include "FileUtil.h"
#include "Timer.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

struct __attribute__((__packed__)) HashEntry {
    size_t hash;
    DBLocalId id;

    static bool compareByHashAndId(const HashEntry &first, const HashEntry &second) {
        if (first.hash != second.hash) {
            return first.hash < second.hash;
        }
        return first.id < second.id;
    }

    static bool compareById(const HashEntry &first, const HashEntry &second) {
        return first.id < second.id;
    }
};

struct LengthRange {
    size_t startKey;
    size_t offset;
    size_t entryLength;
};

struct MemberBits {
    uint64_t memberMask;
    size_t membersBefore;
};

static size_t getMemberIndex(const MemberBits *memberBits, size_t pos) {
    return memberBits[pos / 64].membersBefore + __builtin_popcountll(memberBits[pos / 64].memberMask & ((1ull << (pos % 64)) - 1));
}

static const size_t WRITE_SEQS_PER_THREAD = 1024 * 1024;

static size_t findHashStart(const HashEntry *hashIdPairs, size_t pos, size_t end) {
    if (pos == 0) {
        return 0;
    }
    HashEntry last = { hashIdPairs[pos - 1].hash, std::numeric_limits<DBLocalId>::max() };
    return std::upper_bound(hashIdPairs + pos, hashIdPairs + end, last, HashEntry::compareByHashAndId) - hashIdPairs;
}

static void cutThreadOffsets(const HashEntry *hashIdPairs, size_t begin, size_t end, std::vector<size_t> &threadOffsets) {
    size_t threads = threadOffsets.size() - 1;
    for (size_t thread = 0; thread < threads; thread++) {
        threadOffsets[thread] = findHashStart(hashIdPairs, begin + (end - begin) / threads * thread, end);
    }
    threadOffsets[threads] = end;
}

static size_t seqHash(Sequence &seq, size_t id, const char *data, size_t length) {
    seq.mapSequence(id, 0, data, length);
    size_t hash = Util::hash(seq.numSequence, seq.L);
    return hash ^ (length + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2));
}

static void fillHashIdPairs(DBReader<DBKeyType> &reader, const BaseMatrix *subMat, size_t maxSeqLen, const LengthRange *ranges, size_t rangeCount,
                            size_t keyBegin, size_t keyEnd, HashEntry *hashIdPairs) {
#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        Sequence seq(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
        for (size_t range = 0; range < rangeCount; range++) {
            size_t rangeBegin = std::max(ranges[range].startKey, keyBegin);
            size_t rangeEnd = std::min(ranges[range + 1].startKey, keyEnd);
            if (rangeBegin >= rangeEnd) {
                continue;
            }
            size_t entryLength = ranges[range].entryLength;
            const char *rangeData = (entryLength > 0) ? reader.getDataFromLoadedBuffer(rangeBegin) : NULL;
#pragma omp for schedule(dynamic, 100) nowait
            for (size_t key = rangeBegin; key < rangeEnd; key++) {
                const char *data = (entryLength > 0) ? rangeData + (key - rangeBegin) * entryLength : reader.getData(key, thread_idx);
                size_t length = (entryLength > 0) ? entryLength - 2 : reader.getSeqLen(key);
                hashIdPairs[key - keyBegin].hash = seqHash(seq, key, data, length);
                hashIdPairs[key - keyBegin].id = key;
            }
        }
    }
}

static size_t removeSingletons(HashEntry *hashIdPairs, size_t seqCount, uint64_t *representativeBits, size_t localFirstKey, std::vector<size_t> &taskOffsets) {
    size_t taskCount = taskOffsets.size() - 1;
    cutThreadOffsets(hashIdPairs, 0, seqCount, taskOffsets);
    std::vector<size_t> taskCandidateCounts(taskCount);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t task = 0; task < taskCount; task++) {
        size_t writePos = taskOffsets[task];
        for (size_t hashGroupBegin = taskOffsets[task]; hashGroupBegin < taskOffsets[task + 1];) {
            size_t hashGroupEnd = hashGroupBegin + 1;
            while (hashGroupEnd < taskOffsets[task + 1] && hashIdPairs[hashGroupEnd].hash == hashIdPairs[hashGroupBegin].hash) {
                hashGroupEnd++;
            }
            if (hashGroupEnd - hashGroupBegin == 1) {
                size_t bit = hashIdPairs[hashGroupBegin].id - localFirstKey;
                __sync_fetch_and_or(&representativeBits[bit / 64], 1ull << (bit % 64));
            } else {
                for (size_t i = hashGroupBegin; i < hashGroupEnd; i++) {
                    hashIdPairs[writePos++] = hashIdPairs[i];
                }
            }
            hashGroupBegin = hashGroupEnd;
        }
        taskCandidateCounts[task] = writePos - taskOffsets[task];
    }
    // move tasks in order so none is overwritten
    size_t candidateCount = 0;
    for (size_t task = 0; task < taskCount; task++) {
        memmove(hashIdPairs + candidateCount, hashIdPairs + taskOffsets[task], taskCandidateCounts[task] * sizeof(HashEntry));
        candidateCount += taskCandidateCounts[task];
    }
    return candidateCount;
}

static size_t loadReadChunk(DBReader<DBKeyType> &reader, size_t firstKey, size_t seqCount, size_t readChunkSeqCount, size_t readChunkBegin) {
    size_t readChunkEnd = std::min(readChunkBegin + readChunkSeqCount, seqCount);
    reader.loadData(firstKey + readChunkBegin, firstKey + readChunkEnd);
    if (readChunkEnd < seqCount) {
        reader.prefetchData(firstKey + readChunkEnd, firstKey + std::min(readChunkEnd + readChunkSeqCount, seqCount));
    }
    return readChunkEnd;
}

static std::vector<std::pair<size_t, size_t>> setupHashRanges(DBReader<DBKeyType> &reader, const BaseMatrix *subMat, size_t maxSeqLen,
                                                              size_t firstKey, size_t seqCount, size_t readChunkSeqCount, size_t maxRangeSeqCount) {
    size_t *hashDist = new(std::nothrow) size_t[USHRT_MAX + 1];
    Util::checkAllocation(hashDist, "Can not allocate hashDist memory");
    memset(hashDist, 0, sizeof(size_t) * (USHRT_MAX + 1));
    for (size_t readChunkBegin = 0; readChunkBegin < seqCount; readChunkBegin += readChunkSeqCount) {
        size_t readChunkEnd = loadReadChunk(reader, firstKey, seqCount, readChunkSeqCount, readChunkBegin);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            Sequence seq(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
#pragma omp for schedule(dynamic, 100)
            for (size_t i = readChunkBegin; i < readChunkEnd; i++) {
                size_t key = firstKey + i;
                unsigned short hashValue = static_cast<unsigned short>(seqHash(seq, key, reader.getData(key, thread_idx), reader.getSeqLen(key)));
                __sync_fetch_and_add(&hashDist[hashValue], 1);
            }
        }
    }

    std::vector<std::pair<size_t, size_t>> hashRanges;
    size_t rangeStart = 0;
    size_t rangeSeqCount = 0;
    for (size_t hashValue = 0; hashValue <= USHRT_MAX; hashValue++) {
        if (hashDist[hashValue] > maxRangeSeqCount) {
            Debug(Debug::ERROR) << "Not enough memory to hash " << hashDist[hashValue] << " sequences of one 16 bit hash value\n"
                                << "Raise RAM size or --split-memory-limit.\n";
            EXIT(EXIT_FAILURE);
        }
        if (rangeSeqCount + hashDist[hashValue] > maxRangeSeqCount) {
            hashRanges.emplace_back(rangeStart, hashValue);
            rangeStart = hashValue;
            rangeSeqCount = 0;
        }
        rangeSeqCount += hashDist[hashValue];
    }
    hashRanges.emplace_back(rangeStart, USHRT_MAX + 1);
    delete[] hashDist;
    return hashRanges;
}

static size_t fillHashRange(DBReader<DBKeyType> &reader, const BaseMatrix *subMat, size_t maxSeqLen, size_t firstKey, size_t seqCount, size_t readChunkSeqCount,
                            const std::pair<size_t, size_t> &hashRange, HashEntry *hashIdPairs, MemberBits *memberBits) {
    size_t rangeSeqCount = 0;
    for (size_t readChunkBegin = 0; readChunkBegin < seqCount; readChunkBegin += readChunkSeqCount) {
        size_t readChunkEnd = loadReadChunk(reader, firstKey, seqCount, readChunkSeqCount, readChunkBegin);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            Sequence seq(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
            // read chunks start at a multiple of 64
#pragma omp for schedule(dynamic, 1)
            for (size_t i = readChunkBegin / 64; i < (readChunkEnd + 63) / 64; i++) {
                memberBits[i].memberMask = 0;
                for (size_t pos = i * 64; pos < std::min(i * 64 + 64, readChunkEnd); pos++) {
                    size_t key = firstKey + pos;
                    unsigned short hashValue = static_cast<unsigned short>(seqHash(seq, key, reader.getData(key, thread_idx), reader.getSeqLen(key)));
                    if (hashValue >= hashRange.first && hashValue < hashRange.second) {
                        memberBits[i].memberMask |= 1ull << (pos % 64);
                    }
                }
            }
#pragma omp single
            for (size_t i = readChunkBegin / 64; i < (readChunkEnd + 63) / 64; i++) {
                memberBits[i].membersBefore = rangeSeqCount;
                rangeSeqCount += __builtin_popcountll(memberBits[i].memberMask);
            }
#pragma omp for schedule(dynamic, 1)
            for (size_t i = readChunkBegin / 64; i < (readChunkEnd + 63) / 64; i++) {
                size_t writePos = memberBits[i].membersBefore;
                for (uint64_t memberMask = memberBits[i].memberMask; memberMask != 0; memberMask &= memberMask - 1) {
                    size_t id = firstKey + i * 64 + __builtin_ctzll(memberMask);
                    hashIdPairs[writePos].hash = seqHash(seq, id, reader.getData(id, thread_idx), reader.getSeqLen(id));
                    hashIdPairs[writePos].id = id;
                    writePos++;
                }
            }
        }
    }
    return rangeSeqCount;
}

static void copyBatchSeqs(DBReader<DBKeyType> &reader, size_t firstKey, size_t seqCount, size_t readChunkSeqCount, const HashEntry *batchHashIdPairs, size_t batchSeqCount,
                          MemberBits *memberBits, char *batchSeqs) {
    size_t length = reader.getSeqLen(firstKey);
    memset(memberBits, 0, sizeof(MemberBits) * (seqCount / 64 + 1));
#pragma omp parallel for
    for (size_t i = 0; i < batchSeqCount; i++) {
        size_t pos = batchHashIdPairs[i].id - firstKey;
        __sync_fetch_and_or(&memberBits[pos / 64].memberMask, 1ull << (pos % 64));
    }

    size_t memberCount = 0;
    for (size_t i = 0; i < seqCount / 64 + 1; i++) {
        memberBits[i].membersBefore = memberCount;
        memberCount += __builtin_popcountll(memberBits[i].memberMask);
    }

    // free the prefetched read chunk for the key list
    reader.releaseDataPages();
    std::vector<DBKeyType> readChunkKeys;
    for (size_t readChunkBegin = 0; readChunkBegin < seqCount; readChunkBegin += readChunkSeqCount) {
        size_t readChunkEnd = std::min(readChunkBegin + readChunkSeqCount, seqCount);
        size_t firstMember = memberBits[readChunkBegin / 64].membersBefore;
        size_t endMember = (readChunkEnd < seqCount) ? memberBits[readChunkEnd / 64].membersBefore : memberCount;
        if (firstMember == endMember) {
            continue;
        }
        readChunkKeys.resize(endMember - firstMember);
#pragma omp parallel for schedule(static)
        for (size_t i = readChunkBegin / 64; i < (readChunkEnd + 63) / 64; i++) {
            size_t member = memberBits[i].membersBefore - firstMember;
            for (uint64_t memberMask = memberBits[i].memberMask; memberMask != 0; memberMask &= memberMask - 1) {
                readChunkKeys[member++] = firstKey + i * 64 + __builtin_ctzll(memberMask);
            }
        }
        reader.loadData(readChunkKeys);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
#pragma omp for schedule(dynamic, 1)
            for (size_t i = readChunkBegin / 64; i < (readChunkEnd + 63) / 64; i++) {
                size_t member = memberBits[i].membersBefore;
                for (uint64_t memberMask = memberBits[i].memberMask; memberMask != 0; memberMask &= memberMask - 1) {
                    size_t id = firstKey + i * 64 + __builtin_ctzll(memberMask);
                    memcpy(batchSeqs + member * length, reader.getData(id, thread_idx), length);
                    member++;
                }
            }
        }
    }
}

int clusthashfast(int argc, const char **argv, const Command &command) {
    MMseqsNode::init();
    Parameters &par = Parameters::getInstance();
    par.alphabetSize = MultiParam<NuclAA<int>>(NuclAA<int>(Parameters::CLUST_HASH_DEFAULT_ALPH_SIZE, 5));
    par.seqIdThr = (float)Parameters::CLUST_HASH_DEFAULT_MIN_SEQ_ID / 100.0f;
    par.parseParameters(argc, argv, command, true, 0, 0);

    DBReader<DBKeyType> reader(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
    reader.open(DBReader<DBKeyType>::NOSORT);
    if (Parameters::isEqualDbtype(reader.getDbtype(), Parameters::DBTYPE_AMINO_ACIDS) == false) {
        Debug(Debug::ERROR) << "clusthashfast only supports amino acid databases\n";
        EXIT(EXIT_FAILURE);
    }

    // compressed or padded entries share one buffer per thread
    const bool isCopyingRepSeq = reader.isCompressed() || (DBReader<DBKeyType>::getExtendedDbtype(reader.getDbtype()) & Parameters::DBTYPE_EXTENDED_GPU);
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(reader.getDbtype()) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity == false && par.preloadMode != Parameters::PRELOAD_MODE_MMAP) {
        reader.readMmapedDataInMemory();
    }

    std::vector<LengthRange> nodeLengthRanges;
    if (isLinfinity) {
        size_t endKey = reader.getFileKeyOffsets(MMseqsNode::count)[MMseqsNode::rank + 1];
        const DBReader<DBKeyType>::Index *index = reader.getIndex();
        size_t endOffset = 0;
        for (size_t range = 0; range < reader.getIndexSize(); range++) {
            if (reader.getRangeDataFile(range) == MMseqsNode::rank) {
                nodeLengthRanges.push_back({index[range].id, index[range].offset, index[range].length});
                endOffset = index[range].offset + (endKey - index[range].id) * index[range].length;
            }
        }
        nodeLengthRanges.push_back({endKey, endOffset, 0});
    } else {
        nodeLengthRanges.push_back({0, 0, 0});
        nodeLengthRanges.push_back({reader.getSize(), reader.getDataSize(), 0});
    }
    size_t localFirstKey = nodeLengthRanges.front().startKey;
    size_t localKeyCount = nodeLengthRanges.back().startKey - localFirstKey;

    if (isLinfinity && par.compressed) {
        Debug(Debug::WARNING) << "Compressed output cannot be combined with linfinity databases (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ")\n";
        Debug(Debug::WARNING) << "We recompute with --compressed 0\n";
        par.compressed = 0;
    }

    std::string resultDb = isLinfinity ? par.db2 + "." + SSTR(MMseqsNode::rank) : par.db2;
    DBWriter resultWriter(resultDb.c_str(), (resultDb + ".index").c_str(), 1, par.compressed, Parameters::DBTYPE_CLUSTER_RES);
    std::string resultIndex = par.db2 + ".index." + SSTR(MMseqsNode::rank);
    FILE *resultDataFile = NULL;
    FILE *resultIndexFile = NULL;
    size_t resultDataBytes = 0;
    size_t resultIndexBytes = 0;
    if (isLinfinity) {
        resultDataFile = FileUtil::openAndDelete(resultDb.c_str(), "w");
        resultIndexFile = FileUtil::openAndDelete(resultIndex.c_str(), "w");
    } else {
        resultWriter.open();
    }

    std::string representativeDb = par.db2 + "_redundancy";
    DBWriter representativeWriter(representativeDb.c_str(), (representativeDb + ".index").c_str(), 1, 0, Parameters::DBTYPE_OMIT_FILE);
    if (isLinfinity == false) {
        representativeWriter.open();
    }

    size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    // linfinity: one representative bit per key
    size_t bitWords = isLinfinity ? (localKeyCount + 63) / 64 : 0;
    uint64_t *representativeBits = new(std::nothrow) uint64_t[bitWords]();
    Util::checkAllocation(representativeBits, "Can not allocate representative bit memory");
    // key, newline, entry end and index entry, doubled for growth
    size_t writeBufferBytes = par.threads * WRITE_SEQS_PER_THREAD * 2 * (20 + 1 + 1 + 2 * sizeof(uint64_t));
    if (bitWords * sizeof(uint64_t) + writeBufferBytes >= memoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to run clusthashfast. Memory limit: " << memoryLimit
                            << " bytes, the representative bits and write buffers need: " << bitWords * sizeof(uint64_t) + writeBufferBytes << " bytes.\n"
                            << "Raise RAM size or --split-memory-limit.\n";
        EXIT(EXIT_FAILURE);
    }
    size_t splitMemoryLimit = static_cast<size_t>(static_cast<double>(memoryLimit - bitWords * sizeof(uint64_t) - writeBufferBytes) * 0.95);

    // group lengths that fit into splits
    size_t readChunkBytes = std::min(BatchReader::MAX_INFLIGHT_READ_SIZE, splitMemoryLimit / 4);
    size_t seqBufferLimit = isLinfinity ? splitMemoryLimit / 3 : 0;
    size_t hashIdPairsLimit = splitMemoryLimit - 2 * seqBufferLimit;
    std::vector<size_t> hashIdPairsCapacity(nodeLengthRanges.size() - 1, 0);
    std::vector<size_t> splitRangeOffsets;
    for (size_t i = 0; i + 1 < nodeLengthRanges.size(); i++) {
        size_t lengthSeqCount = nodeLengthRanges[i + 1].startKey - nodeLengthRanges[i].startKey;
        size_t lengthBytes = (isLinfinity ? nodeLengthRanges[i + 1].offset - nodeLengthRanges[i].offset : 0) + lengthSeqCount * sizeof(HashEntry);
        if (lengthBytes > splitMemoryLimit) {
            size_t readChunkSeqCount = isLinfinity ? std::min(readChunkBytes / nodeLengthRanges[i].entryLength / 64 * 64, lengthSeqCount) : 0;
            size_t chunkBufferBytes = 2 * reader.getLoadBufferSize(nodeLengthRanges[i].startKey, nodeLengthRanges[i].startKey + readChunkSeqCount);
            size_t keptBytes = isLinfinity ? (lengthSeqCount / 64 + 1) * sizeof(MemberBits) : lengthSeqCount / 4;
            size_t hashBytes = sizeof(HashEntry) + (isLinfinity ? nodeLengthRanges[i].entryLength - 2 : 0);
            // pairs to write a split length in key order
            size_t pairBytes = isLinfinity ? lengthSeqCount * sizeof(ClusterPair) : 0;
            if ((isLinfinity && readChunkSeqCount == 0) || splitMemoryLimit <= keptBytes + chunkBufferBytes + pairBytes + hashBytes) {
                Debug(Debug::ERROR) << "Not enough memory to hash " << lengthSeqCount << " sequences\n"
                                    << "Raise RAM size or --split-memory-limit.\n";
                EXIT(EXIT_FAILURE);
            }
            // copy batches go after the candidates in hashIdPairs
            hashIdPairsCapacity[i] = (splitMemoryLimit - keptBytes - chunkBufferBytes - pairBytes) / sizeof(HashEntry);
        }
        if (splitRangeOffsets.empty()
            || (isLinfinity ? nodeLengthRanges[i + 1].offset - nodeLengthRanges[splitRangeOffsets.back()].offset + 2 * Util::getPageSize() : 0) > seqBufferLimit
            || (nodeLengthRanges[i + 1].startKey - nodeLengthRanges[splitRangeOffsets.back()].startKey) * sizeof(HashEntry) > hashIdPairsLimit) {
            splitRangeOffsets.push_back(i);
        }
    }
    splitRangeOffsets.push_back(nodeLengthRanges.size() - 1);
    HashEntry *hashIdPairs = NULL;
    size_t allocatedHashEntryCount = 0;

    std::vector<size_t> threadOffsets(par.threads + 1);
    // more tasks than threads for large hash groups
    std::vector<size_t> taskOffsets(par.threads * 16 + 1);
    std::vector<std::string> threadClusterText(par.threads);
    // (representative id, offset) per entry
    std::vector<std::vector<uint64_t> > threadEntries(par.threads);
    ClusterPair *splitPairs = NULL;
    size_t splitPairCount = 0;
    bool needsSort = false;
    size_t clusters = 0;
    SubstitutionMatrix sMat(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.2);
    ReducedMatrix subMat(sMat.probMatrix, sMat.subMatrixPseudoCounts, sMat.aa2num, sMat.num2aa, sMat.alphabetSize, par.alphabetSize.values.aminoacid(), 2.0);

    Debug(Debug::INFO) << "Hashing " << localKeyCount << " sequences in " << splitRangeOffsets.size() - 1 << " splits\n";
    Timer timer;
    Debug::Progress progress(splitRangeOffsets.size() - 1);
    for (size_t split = 0; split + 1 < splitRangeOffsets.size(); split++) {
        progress.updateProgress();
        size_t splitFirstKey = nodeLengthRanges[splitRangeOffsets[split]].startKey;
        size_t splitSeqCount = nodeLengthRanges[splitRangeOffsets[split + 1]].startKey - splitFirstKey;
        bool isLargeLength = hashIdPairsCapacity[splitRangeOffsets[split]] > 0;
        size_t length = isLinfinity ? nodeLengthRanges[splitRangeOffsets[split]].entryLength - 2 : 0;
        bool isCopyingSeqs = isLargeLength && isLinfinity;
        if (isCopyingSeqs) {
            splitPairs = new(std::nothrow) ClusterPair[splitSeqCount];
            Util::checkAllocation(splitPairs, "Can not allocate splitPairs memory in clusthashfast");
            splitPairCount = 0;
        }

        size_t readChunkSeqCount = isCopyingSeqs ? std::min(readChunkBytes / (length + 2) / 64 * 64, splitSeqCount) : splitSeqCount;
        size_t hashedSeqCapacity = isLargeLength ? hashIdPairsCapacity[splitRangeOffsets[split]] : splitSeqCount;
        bool canStoreHashIdPairs = hashedSeqCapacity >= splitSeqCount;
        bool isNextPrefetched = false;

        if (allocatedHashEntryCount < hashedSeqCapacity) {
            Util::releasePages(hashIdPairs, allocatedHashEntryCount * sizeof(HashEntry));
            delete[] hashIdPairs;
            hashIdPairs = new(std::nothrow) HashEntry[hashedSeqCapacity];
            Util::checkAllocation(hashIdPairs, "Can not allocate hash entry memory");
            allocatedHashEntryCount = hashedSeqCapacity;
        }

        std::vector<std::pair<size_t, size_t>> hashRanges(1, std::make_pair(0, USHRT_MAX + 1));
        std::vector<size_t> copyBatchOffsets(1, 0);
        MemberBits *memberBits = NULL;
        char *batchSeqs = NULL;
        if (isLargeLength) {
            memberBits = new(std::nothrow) MemberBits[splitSeqCount / 64 + 1];
            Util::checkAllocation(memberBits, "Can not allocate member bit memory");
        }

        if (isCopyingSeqs && canStoreHashIdPairs) {
            for (size_t readChunkBegin = 0; readChunkBegin < splitSeqCount; readChunkBegin += readChunkSeqCount) {
                size_t readChunkEnd = loadReadChunk(reader, splitFirstKey, splitSeqCount, readChunkSeqCount, readChunkBegin);
                fillHashIdPairs(reader, &subMat, par.maxSeqLen, nodeLengthRanges.data() + splitRangeOffsets[split], splitRangeOffsets[split + 1] - splitRangeOffsets[split],
                                splitFirstKey + readChunkBegin, splitFirstKey + readChunkEnd, hashIdPairs + readChunkBegin);
            }
            SORT_PARALLEL(hashIdPairs, hashIdPairs + splitSeqCount, HashEntry::compareByHashAndId);
            size_t candidateCount = removeSingletons(hashIdPairs, splitSeqCount, representativeBits, localFirstKey, taskOffsets);
            // cut the candidates at hash groups into copy batches
            size_t copyCapacity = (hashedSeqCapacity - candidateCount) * sizeof(HashEntry) / length;
            for (size_t hashGroupBegin = 0; hashGroupBegin < candidateCount;) {
                size_t hashGroupEnd = hashGroupBegin + 1;
                while (hashGroupEnd < candidateCount && hashIdPairs[hashGroupEnd].hash == hashIdPairs[hashGroupBegin].hash) {
                    hashGroupEnd++;
                }
                // a hash group does not fit, use hash ranges
                if (hashGroupEnd - hashGroupBegin > copyCapacity) {
                    canStoreHashIdPairs = false;
                    break;
                }
                if (hashGroupEnd - copyBatchOffsets.back() > copyCapacity) {
                    copyBatchOffsets.push_back(hashGroupBegin);
                }
                hashGroupBegin = hashGroupEnd;
            }
            if (candidateCount > copyBatchOffsets.back()) {
                copyBatchOffsets.push_back(candidateCount);
            }
            batchSeqs = reinterpret_cast<char *>(hashIdPairs + candidateCount);
        }

        if (isLargeLength && canStoreHashIdPairs == false) {
            // copy the sequences of a hash range after its hashIdPairs
            size_t maxRangeSeqCount = hashedSeqCapacity * sizeof(HashEntry) / (sizeof(HashEntry) + length);
            hashRanges = setupHashRanges(reader, &subMat, par.maxSeqLen, splitFirstKey, splitSeqCount, readChunkSeqCount, maxRangeSeqCount);
            batchSeqs = isCopyingSeqs ? reinterpret_cast<char *>(hashIdPairs + maxRangeSeqCount) : NULL;
        } else if (isLargeLength == false) {
            // hash while the rest of the split is read
            reader.startLoadData(splitFirstKey, splitFirstKey + splitSeqCount, readChunkBytes);
            for (size_t readChunkBegin = splitFirstKey; readChunkBegin < splitFirstKey + splitSeqCount;) {
                size_t readChunkEnd = reader.waitForLoadedData();
                // prefetch the next split if both fit
                if (readChunkEnd == splitFirstKey + splitSeqCount && isLinfinity && split + 2 < splitRangeOffsets.size()
                    && hashIdPairsCapacity[splitRangeOffsets[split + 1]] == 0) {
                    size_t nextFirstKey = nodeLengthRanges[splitRangeOffsets[split + 1]].startKey;
                    size_t nextEndKey = nodeLengthRanges[splitRangeOffsets[split + 2]].startKey;
                    size_t nextHashBytes = std::max(allocatedHashEntryCount, nextEndKey - nextFirstKey) * sizeof(HashEntry);
                    isNextPrefetched = reader.getPrefetchBufferSize(nextFirstKey, nextEndKey) + nextHashBytes <= splitMemoryLimit;
                    if (isNextPrefetched) {
                        reader.prefetchData(nextFirstKey, nextEndKey);
                    }
                }
                fillHashIdPairs(reader, &subMat, par.maxSeqLen, nodeLengthRanges.data() + splitRangeOffsets[split], splitRangeOffsets[split + 1] - splitRangeOffsets[split],
                                readChunkBegin, readChunkEnd, hashIdPairs + (readChunkBegin - splitFirstKey));
                readChunkBegin = readChunkEnd;
            }
        }

        size_t rangeCount = (isCopyingSeqs && canStoreHashIdPairs) ? copyBatchOffsets.size() - 1 : hashRanges.size();
        needsSort |= rangeCount > 1;
        for (size_t range = 0; range < rangeCount; range++) {
            HashEntry *rangeHashIdPairs = hashIdPairs;
            size_t hashedSeqCount = 0;
            if (isCopyingSeqs && canStoreHashIdPairs) {
                // already sorted by hash and id
                rangeHashIdPairs = hashIdPairs + copyBatchOffsets[range];
                hashedSeqCount = copyBatchOffsets[range + 1] - copyBatchOffsets[range];
            } else {
                hashedSeqCount = splitSeqCount;
                if (isLargeLength) {
                    hashedSeqCount = fillHashRange(reader, &subMat, par.maxSeqLen, splitFirstKey, splitSeqCount, readChunkSeqCount, hashRanges[range], hashIdPairs, memberBits);
                }
                SORT_PARALLEL(hashIdPairs, hashIdPairs + hashedSeqCount, HashEntry::compareByHashAndId);
                // linfinity singletons only set their bit
                if (isLinfinity) {
                    hashedSeqCount = removeSingletons(hashIdPairs, hashedSeqCount, representativeBits, localFirstKey, taskOffsets);
                }
            }
            if (isCopyingSeqs) {
                copyBatchSeqs(reader, splitFirstKey, splitSeqCount, readChunkSeqCount, rangeHashIdPairs, hashedSeqCount, memberBits, batchSeqs);
            }

            // cut tasks before hash becomes the representative id
            cutThreadOffsets(rangeHashIdPairs, 0, hashedSeqCount, taskOffsets);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t task = 0; task < taskOffsets.size() - 1; task++) {
                unsigned int thread = 0;
#ifdef OPENMP
                thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
                size_t taskEnd = taskOffsets[task + 1];
                std::string repSeqCopy;
                for (size_t hashGroupBegin = taskOffsets[task]; hashGroupBegin < taskEnd;) {
                    size_t hashGroupEnd = hashGroupBegin + 1;
                    while (hashGroupEnd < taskEnd && rangeHashIdPairs[hashGroupEnd].hash == rangeHashIdPairs[hashGroupBegin].hash) {
                        hashGroupEnd++;
                    }
                    // the smallest unassigned id is the next representative
                    for (size_t i = hashGroupBegin; i < hashGroupEnd;) {
                        std::iter_swap(rangeHashIdPairs + i, std::min_element(rangeHashIdPairs + i, rangeHashIdPairs + hashGroupEnd, HashEntry::compareById));
                        DBLocalId repId = rangeHashIdPairs[i].id;
                        rangeHashIdPairs[i++].hash = repId;
                        if (i == hashGroupEnd) {
                            break;
                        }
                        unsigned int repLength = reader.getSeqLen(repId);
                        const char *repSeq = (batchSeqs == NULL) ? reader.getData(repId, thread) : batchSeqs + getMemberIndex(memberBits, repId - splitFirstKey) * length;
                        if (isCopyingRepSeq) {
                            repSeqCopy.assign(repSeq, repLength);
                            repSeq = repSeqCopy.data();
                        }
                        for (size_t j = i; j < hashGroupEnd; j++) {
                            DBLocalId memberId = rangeHashIdPairs[j].id;
                            if (reader.getSeqLen(memberId) != repLength) {
                                continue;
                            }
                            const char *memberSeq = (batchSeqs == NULL) ? reader.getData(memberId, thread) : batchSeqs + getMemberIndex(memberBits, memberId - splitFirstKey) * length;
                            unsigned int distance = DistanceCalculator::computeInverseHammingDistance(repSeq, memberSeq, repLength);
                            if (static_cast<float>(distance) / static_cast<float>(repLength) >= par.seqIdThr) {
                                std::swap(rangeHashIdPairs[i], rangeHashIdPairs[j]);
                                rangeHashIdPairs[i++].hash = repId;
                            }
                        }
                    }
                    hashGroupBegin = hashGroupEnd;
                }
            }

            // gather each cluster, representative first
            SORT_PARALLEL(rangeHashIdPairs, rangeHashIdPairs + hashedSeqCount, HashEntry::compareByHashAndId);
            for (size_t writeBegin = 0, writeEnd; writeBegin < hashedSeqCount; writeBegin = writeEnd) {
                writeEnd = findHashStart(rangeHashIdPairs, std::min(hashedSeqCount, writeBegin + WRITE_SEQS_PER_THREAD * par.threads), hashedSeqCount);
                cutThreadOffsets(rangeHashIdPairs, writeBegin, writeEnd, threadOffsets);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                for (int thread = 0; thread < par.threads; thread++) {
                    std::string &data = threadClusterText[thread];
                    std::vector<uint64_t> &entries = threadEntries[thread];
                    data.clear();
                    entries.clear();
                    char keyBuffer[32];
                    size_t threadEnd = threadOffsets[thread + 1];
                    for (size_t clusterBegin = threadOffsets[thread]; clusterBegin < threadEnd;) {
                        size_t clusterEnd = clusterBegin + 1;
                        while (clusterEnd < threadEnd && rangeHashIdPairs[clusterEnd].hash == rangeHashIdPairs[clusterBegin].hash) {
                            clusterEnd++;
                        }
                        DBLocalId repId = rangeHashIdPairs[clusterBegin].id;
                        if (isLinfinity) {
                            __sync_fetch_and_or(&representativeBits[(repId - localFirstKey) / 64], 1ull << ((repId - localFirstKey) % 64));
                        }
                        // split lengths are written in key order after the last hash range
                        if (isCopyingSeqs) {
                            for (size_t i = clusterBegin; i < clusterEnd; i++) {
                                ClusterPair pair = { repId, rangeHashIdPairs[i].id };
                                splitPairs[splitPairCount + i] = pair;
                            }
                        } else if (isLinfinity == false || clusterEnd - clusterBegin > 1) {
                            // singletons only have their bit
                            entries.push_back(repId);
                            entries.push_back(data.size());
                            for (size_t i = clusterBegin; i < clusterEnd; i++) {
                                char *end = Itoa::u64toa_sse2(static_cast<uint64_t>(reader.getDbKey(rangeHashIdPairs[i].id)), keyBuffer);
                                *(end - 1) = '\n';
                                data.append(keyBuffer, end - keyBuffer);
                            }
                            data.push_back('\0');
                        }
                        clusterBegin = clusterEnd;
                    }
                }
                if (isLinfinity) {
                    MMseqsNode::writeClusterThreadEntries(resultDataFile, resultDb, resultIndexFile, resultIndex, threadClusterText, threadEntries, resultDataBytes, resultIndexBytes);
                    continue;
                }
                for (int thread = 0; thread < par.threads; thread++) {
                    const std::string &data = threadClusterText[thread];
                    const std::vector<uint64_t> &entries = threadEntries[thread];
                    for (size_t i = 0; i < entries.size(); i += 2) {
                        DBLocalId repId = entries[i];
                        size_t entryEnd = (i + 2 < entries.size()) ? entries[i + 3] : data.size();
                        resultWriter.writeData(data.data() + entries[i + 1], entryEnd - entries[i + 1] - 1, reader.getDbKey(repId), 0);
                        representativeWriter.writeIndexEntry(reader.getDbKey(repId), reader.getOffset(repId), reader.getEntryLen(repId), 0);
                    }
                    clusters += entries.size() / 2;
                }
            }
            if (isCopyingSeqs) {
                splitPairCount += hashedSeqCount;
            }
        }

        if (isCopyingSeqs) {
            SORT_PARALLEL(splitPairs, splitPairs + splitPairCount, ClusterPair::compareByRep);
            size_t splitEndKey = splitFirstKey + splitSeqCount;
            for (size_t writeBegin = splitFirstKey; writeBegin < splitEndKey; writeBegin += WRITE_SEQS_PER_THREAD * par.threads) {
                size_t writeKeys = std::min(splitEndKey - writeBegin, WRITE_SEQS_PER_THREAD * par.threads);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                for (int thread = 0; thread < par.threads; thread++) {
                    std::string &data = threadClusterText[thread];
                    std::vector<uint64_t> &entries = threadEntries[thread];
                    data.clear();
                    entries.clear();
                    char keyBuffer[32];
                    DBKeyType beginKey = writeBegin + writeKeys * thread / par.threads;
                    DBKeyType endKey = writeBegin + writeKeys * (thread + 1) / par.threads;
                    ClusterPair beginRep = { beginKey, beginKey };
                    ClusterPair endRep = { endKey, endKey };
                    size_t pair = std::lower_bound(splitPairs, splitPairs + splitPairCount, beginRep, ClusterPair::compareByRep) - splitPairs;
                    size_t endPair = std::lower_bound(splitPairs, splitPairs + splitPairCount, endRep, ClusterPair::compareByRep) - splitPairs;
                    while (pair < endPair) {
                        DBKeyType repKey = splitPairs[pair].repKey;
                        // singletons only have their bit
                        if (pair + 1 == endPair || splitPairs[pair + 1].repKey != repKey) {
                            pair++;
                            continue;
                        }
                        entries.push_back(repKey);
                        entries.push_back(data.size());
                        for (; pair < endPair && splitPairs[pair].repKey == repKey; pair++) {
                            char *end = Itoa::u64toa_sse2(static_cast<uint64_t>(splitPairs[pair].memberKey), keyBuffer);
                            *(end - 1) = '\n';
                            data.append(keyBuffer, end - keyBuffer);
                        }
                        data.push_back('\0');
                    }
                }
                MMseqsNode::writeClusterThreadEntries(resultDataFile, resultDb, resultIndexFile, resultIndex, threadClusterText, threadEntries, resultDataBytes, resultIndexBytes);
            }
            Util::releasePages(splitPairs, splitSeqCount * sizeof(ClusterPair));
            delete[] splitPairs;
            splitPairs = NULL;
        }

        delete[] memberBits;
        // keep the buffers for a prefetched next split
        if (isLinfinity && isNextPrefetched == false) {
            reader.releaseDataPages();
            Util::releasePages(hashIdPairs, allocatedHashEntryCount * sizeof(HashEntry));
            delete[] hashIdPairs;
            hashIdPairs = NULL;
            allocatedHashEntryCount = 0;
        }
    }

    if (isLinfinity) {
        std::string phaseValues;
        MMseqsNode::addPhaseTime(phaseValues, "hashing sequences", timer);
        MMseqsNode::printPhaseValues(phaseValues);
        MMseqsNode::closeClusterFiles(resultDataFile, resultDb, resultIndexFile, resultIndex, resultDataBytes, resultIndexBytes);
        MMseqsNode::writeRepresentativeBits(representativeDb, reader, representativeBits, localFirstKey, localKeyCount);
    } else {
        // unsorted only if a length was split into hash ranges
        resultWriter.close(false, needsSort);
        representativeWriter.close(false, needsSort);
    }

    if (MMseqsNode::isMaster()) {
        DBReader<DBKeyType>::softlinkDb(par.db1, representativeDb, DBFiles::DATA);
        DBWriter::writeDbtypeFile(representativeDb.c_str(), reader.getDbtype(), reader.isCompressed());
        if (isLinfinity) {
            DBReader<DBKeyType> representativeReader(representativeDb.c_str(), (representativeDb + ".index").c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
            representativeReader.open(DBReader<DBKeyType>::NOSORT);
            for (size_t range = 0; range < representativeReader.getIndexSize(); range++) {
                clusters += representativeReader.getRangeRepresentativeCount(range);
            }
            representativeReader.close();
            // the next steps read no headers, lookup or source
            DBWriter::writeDbtypeFile(par.db2.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_CLUSTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
        } else {
            DBReader<DBKeyType>::softlinkDb(par.db1, representativeDb, DBFiles::SEQUENCE_ANCILLARY);
        }
        Debug(Debug::INFO) << "Size of the sequence database: " << reader.getSize() << "\n";
        Debug(Debug::INFO) << "Number of clusters: " << clusters << "\n";
    }
    delete[] representativeBits;
    Util::releasePages(hashIdPairs, allocatedHashEntryCount * sizeof(HashEntry));
    delete[] hashIdPairs;
    reader.close();
    return EXIT_SUCCESS;
}
