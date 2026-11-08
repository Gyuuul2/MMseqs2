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

// keys of one length, starting at startKey and at offset bytes into the data file of the node, entries of entryLength bytes
struct LengthRange {
    size_t startKey;
    size_t offset;
    size_t entryLength;
};

// positions 64 * i .. 64 * i + 63 of a split length: which are members of the current hash range or copy batch, and how many members come before them
struct MemberBits {
    uint64_t memberMask;
    size_t membersBefore;
};

// index of the sequence at position pos of a split length among the members of the current hash range or copy batch
static size_t getMemberIndex(const MemberBits *memberBits, size_t pos) {
    return memberBits[pos / 64].membersBefore + __builtin_popcountll(memberBits[pos / 64].memberMask & ((1ull << (pos % 64)) - 1));
}

// the sequences whose clusters a thread formats at once before they are written
static const size_t WRITE_SEQS_PER_THREAD = 1024 * 1024;
// a formatted sequence takes at most its key and a newline, a cluster end and an index entry, in buffers that grow to twice their content
static const size_t WRITE_SEQ_BYTES = 2 * (20 + 1 + 1 + 2 * sizeof(uint64_t));

// the first position from pos on in hashIdPairs [0, end), sorted by hash, that does not continue the hash before it
static size_t findHashStart(const HashEntry *hashIdPairs, size_t pos, size_t end) {
    return (pos == 0) ? 0 : std::upper_bound(hashIdPairs + pos, hashIdPairs + end, HashEntry{hashIdPairs[pos - 1].hash, std::numeric_limits<DBLocalId>::max()}, HashEntry::compareByHashAndId) - hashIdPairs;
}

// cuts hashIdPairs [begin, end), sorted by hash, into threadOffsets.size() - 1 tasks without cutting a hash
static void cutThreadOffsets(const HashEntry *hashIdPairs, size_t begin, size_t end, std::vector<size_t> &threadOffsets) {
    const size_t threads = threadOffsets.size() - 1;
    for (size_t thread = 0; thread < threads; thread++) {
        threadOffsets[thread] = findHashStart(hashIdPairs, begin + (end - begin) / threads * thread, end);
    }
    threadOffsets[threads] = end;
}

static size_t seqHash(Sequence &seq, size_t id, const char *data, size_t length) {
    seq.mapSequence(id, 0, data, length);
    const size_t hash = Util::hash(seq.numSequence, seq.L);
    return hash ^ (length + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2));
}

// hashes the loaded keys [keyBegin, keyEnd) of ranges [0, rangeCount), ended by ranges[rangeCount].startKey, into hashIdPairs
static void fillHashIdPairs(DBReader<DBKeyType> &reader, const BaseMatrix *subMat, size_t maxSeqLen, const LengthRange *ranges, size_t rangeCount, size_t keyBegin, size_t keyEnd, HashEntry *hashIdPairs) {
#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        Sequence seq(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
        for (size_t range = 0; range < rangeCount; range++) {
            const size_t rangeBegin = std::max(ranges[range].startKey, keyBegin);
            const size_t rangeEnd = std::min(ranges[range + 1].startKey, keyEnd);
            if (rangeBegin >= rangeEnd) {
                continue;
            }
            const size_t entryLength = ranges[range].entryLength;
            const char *rangeData = (entryLength > 0) ? reader.getDataFromLoadedBuffer(rangeBegin) : NULL;
#pragma omp for schedule(dynamic, 100) nowait
            for (size_t key = rangeBegin; key < rangeEnd; key++) {
                const char *data = (entryLength > 0) ? rangeData + (key - rangeBegin) * entryLength : reader.getData(key, thread_idx);
                const size_t length = (entryLength > 0) ? entryLength - 2 : reader.getSeqLen(key);
                hashIdPairs[key - keyBegin].hash = seqHash(seq, key, data, length);
                hashIdPairs[key - keyBegin].id = key;
            }
        }
    }
}

// hash groups of one sequence of hashIdPairs [0, seqCount), sorted by hash, become representatives, the other groups are moved to the front and their count is returned
static size_t removeSingletons(HashEntry *hashIdPairs, size_t seqCount, uint64_t *representativeBits, size_t localFirstKey, std::vector<size_t> &taskOffsets) {
    const size_t taskCount = taskOffsets.size() - 1;
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
                const size_t bit = hashIdPairs[hashGroupBegin].id - localFirstKey;
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
    // the tasks move to the front in order, so no task overwrites one that has not moved yet
    size_t candidateCount = 0;
    for (size_t task = 0; task < taskCount; task++) {
        memmove(hashIdPairs + candidateCount, hashIdPairs + taskOffsets[task], taskCandidateCounts[task] * sizeof(HashEntry));
        candidateCount += taskCandidateCounts[task];
    }
    return candidateCount;
}

// loads the read chunk from readChunkBegin of keys [firstKey, firstKey + seqCount), prefetches the next one and returns the end of the loaded one
static size_t loadReadChunk(DBReader<DBKeyType> &reader, size_t firstKey, size_t seqCount, size_t readChunkSeqCount, size_t readChunkBegin) {
    const size_t readChunkEnd = std::min(readChunkBegin + readChunkSeqCount, seqCount);
    reader.loadData(firstKey + readChunkBegin, firstKey + readChunkEnd);
    if (readChunkEnd < seqCount) {
        reader.prefetchData(firstKey + readChunkEnd, firstKey + std::min(readChunkEnd + readChunkSeqCount, seqCount));
    }
    return readChunkEnd;
}

// counts the sequences of each 16 bit hash value of keys [firstKey, firstKey + seqCount) and cuts them into hash ranges of at most maxRangeSeqCount, like setupKmerSplits
static std::vector<std::pair<size_t, size_t>> setupHashRanges(DBReader<DBKeyType> &reader, const BaseMatrix *subMat, size_t maxSeqLen,
                                                              size_t firstKey, size_t seqCount, size_t readChunkSeqCount, size_t maxRangeSeqCount) {
    size_t *hashDist = new(std::nothrow) size_t[USHRT_MAX + 1];
    Util::checkAllocation(hashDist, "Can not allocate hashDist memory");
    memset(hashDist, 0, sizeof(size_t) * (USHRT_MAX + 1));
    for (size_t readChunkBegin = 0; readChunkBegin < seqCount; readChunkBegin += readChunkSeqCount) {
        const size_t readChunkEnd = loadReadChunk(reader, firstKey, seqCount, readChunkSeqCount, readChunkBegin);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            Sequence seq(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
#pragma omp for schedule(dynamic, 100)
            for (size_t i = readChunkBegin; i < readChunkEnd; i++) {
                __sync_fetch_and_add(&hashDist[static_cast<unsigned short>(seqHash(seq, firstKey + i, reader.getData(firstKey + i, thread_idx), reader.getSeqLen(firstKey + i)))], 1);
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

// keeps the sequences of keys [firstKey, firstKey + seqCount) whose 16 bit hash value is in hashRange
static size_t fillHashRange(DBReader<DBKeyType> &reader, const BaseMatrix *subMat, size_t maxSeqLen, size_t firstKey, size_t seqCount, size_t readChunkSeqCount,
                            const std::pair<size_t, size_t> &hashRange, HashEntry *hashIdPairs, MemberBits *memberBits) {
    size_t rangeSeqCount = 0;
    for (size_t readChunkBegin = 0; readChunkBegin < seqCount; readChunkBegin += readChunkSeqCount) {
        const size_t readChunkEnd = loadReadChunk(reader, firstKey, seqCount, readChunkSeqCount, readChunkBegin);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            Sequence seq(maxSeqLen, reader.getDbtype(), subMat, 0, false, false);
            // a thread marks whole groups of 64 positions, so read chunks start at a multiple of 64
#pragma omp for schedule(dynamic, 1)
            for (size_t i = readChunkBegin / 64; i < (readChunkEnd + 63) / 64; i++) {
                memberBits[i].memberMask = 0;
                for (size_t pos = i * 64; pos < std::min(i * 64 + 64, readChunkEnd); pos++) {
                    const unsigned short hashValue = static_cast<unsigned short>(seqHash(seq, firstKey + pos, reader.getData(firstKey + pos, thread_idx), reader.getSeqLen(firstKey + pos)));
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
                    const size_t id = firstKey + i * 64 + __builtin_ctzll(memberMask);
                    hashIdPairs[writePos].hash = seqHash(seq, id, reader.getData(id, thread_idx), reader.getSeqLen(id));
                    hashIdPairs[writePos].id = id;
                    writePos++;
                }
            }
        }
    }
    return rangeSeqCount;
}

// copies the sequences of batchHashIdPairs, keys of [firstKey, firstKey + seqCount), to their places in batchSeqs, reading only their pages of each readChunkSeqCount keys
static void copyBatchSeqs(DBReader<DBKeyType> &reader, size_t firstKey, size_t seqCount, size_t readChunkSeqCount, const HashEntry *batchHashIdPairs, size_t batchSeqCount,
                          MemberBits *memberBits, char *batchSeqs) {
    const size_t length = reader.getSeqLen(firstKey);
    memset(memberBits, 0, sizeof(MemberBits) * (seqCount / 64 + 1));
#pragma omp parallel for
    for (size_t i = 0; i < batchSeqCount; i++) {
        const size_t pos = batchHashIdPairs[i].id - firstKey;
        __sync_fetch_and_or(&memberBits[pos / 64].memberMask, 1ull << (pos % 64));
    }
    size_t memberCount = 0;
    for (size_t i = 0; i < seqCount / 64 + 1; i++) {
        memberBits[i].membersBefore = memberCount;
        memberCount += __builtin_popcountll(memberBits[i].memberMask);
    }
    // the key list replaces the prefetched read chunk in the read chunk budget
    reader.releaseDataPages();
    std::vector<DBKeyType> readChunkKeys;
    for (size_t readChunkBegin = 0; readChunkBegin < seqCount; readChunkBegin += readChunkSeqCount) {
        const size_t readChunkEnd = std::min(readChunkBegin + readChunkSeqCount, seqCount);
        const size_t firstMember = memberBits[readChunkBegin / 64].membersBefore;
        const size_t endMember = (readChunkEnd < seqCount) ? memberBits[readChunkEnd / 64].membersBefore : memberCount;
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
                    const size_t id = firstKey + i * 64 + __builtin_ctzll(memberMask);
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
    // a compressed or padded entry is decoded into one buffer per thread, so the representative is copied before its members are read
    const bool isCopyingRepSeq = reader.isCompressed() || (DBReader<DBKeyType>::getExtendedDbtype(reader.getDbtype()) & Parameters::DBTYPE_EXTENDED_GPU);
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(reader.getDbtype()) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity == false && par.preloadMode != Parameters::PRELOAD_MODE_MMAP) {
        reader.readMmapedDataInMemory();
    }

    std::vector<LengthRange> nodeLengthRanges;
    if (isLinfinity) {
        const size_t endKey = reader.getFileKeyOffsets(MMseqsNode::count)[MMseqsNode::rank + 1];
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
    const size_t localFirstKey = nodeLengthRanges.front().startKey;
    const size_t localKeyCount = nodeLengthRanges.back().startKey - localFirstKey;

    if (isLinfinity && par.compressed) {
        Debug(Debug::WARNING) << "Compressed output cannot be combined with linfinity databases (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ")\n";
        Debug(Debug::WARNING) << "We recompute with --compressed 0\n";
        par.compressed = 0;
    }
    const std::string resultDb = isLinfinity ? par.db2 + "." + SSTR(MMseqsNode::rank) : par.db2;
    DBWriter resultWriter(resultDb.c_str(), (resultDb + ".index").c_str(), 1, par.compressed, Parameters::DBTYPE_CLUSTER_RES);
    const std::string resultIndex = par.db2 + ".index." + SSTR(MMseqsNode::rank);
    // a linfinity node writes its data and index file on all threads at prefix-summed offsets
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
    const std::string representativeDb = par.db2 + "_redundancy";
    DBWriter representativeWriter(representativeDb.c_str(), (representativeDb + ".index").c_str(), 1, 0, Parameters::DBTYPE_OMIT_FILE);
    if (isLinfinity == false) {
        representativeWriter.open();
    }
    const size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    // a linfinity node keeps its representatives as one bit per key
    const size_t representativeBitsWordCount = isLinfinity ? (localKeyCount + 63) / 64 : 0;
    uint64_t *representativeBits = new(std::nothrow) uint64_t[representativeBitsWordCount]();
    Util::checkAllocation(representativeBits, "Can not allocate representative bit memory");
    const size_t writeBufferBytes = par.threads * WRITE_SEQS_PER_THREAD * WRITE_SEQ_BYTES;
    if (representativeBitsWordCount * sizeof(uint64_t) + writeBufferBytes >= memoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to run clusthashfast. Memory limit: " << memoryLimit
                            << " bytes, the representative bits and write buffers need: " << representativeBitsWordCount * sizeof(uint64_t) + writeBufferBytes << " bytes.\n"
                            << "Raise RAM size or --split-memory-limit.\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t splitMemoryLimit = static_cast<size_t>(static_cast<double>(memoryLimit - representativeBitsWordCount * sizeof(uint64_t) - writeBufferBytes) * 0.95);

    // lengths that fit are a split together, a larger length a split alone (linfinity: per read chunk and copy batch, standard: per hash range)
    const size_t readChunkBytes = std::min(MAX_INFLIGHT_READ_BYTES, splitMemoryLimit / 4);
    const size_t seqBufferLimit = isLinfinity ? splitMemoryLimit / 3 : 0;
    const size_t hashIdPairsLimit = splitMemoryLimit - 2 * seqBufferLimit;
    std::vector<size_t> hashIdPairsCapacity(nodeLengthRanges.size() - 1, 0);
    std::vector<size_t> splitRangeOffsets;
    for (size_t i = 0; i + 1 < nodeLengthRanges.size(); i++) {
        const size_t lengthSeqCount = nodeLengthRanges[i + 1].startKey - nodeLengthRanges[i].startKey;
        const size_t lengthBytes = (isLinfinity ? nodeLengthRanges[i + 1].offset - nodeLengthRanges[i].offset : 0) + lengthSeqCount * sizeof(HashEntry);
        if (lengthBytes > splitMemoryLimit) {
            const size_t readChunkSeqCount = isLinfinity ? std::min(readChunkBytes / nodeLengthRanges[i].entryLength / 64 * 64, lengthSeqCount) : 0;
            const size_t readChunkBufferBytes = 2 * reader.getLoadBufferBytes(nodeLengthRanges[i].startKey, nodeLengthRanges[i].startKey + readChunkSeqCount);
            const size_t keptBytes = isLinfinity ? (lengthSeqCount / 64 + 1) * sizeof(MemberBits) : lengthSeqCount / 4;
            const size_t hashBytes = sizeof(HashEntry) + (isLinfinity ? nodeLengthRanges[i].entryLength - 2 : 0);
            // a split linfinity length keeps a pair for every member to write its clusters in key order
            const size_t pairBytes = isLinfinity ? lengthSeqCount * sizeof(ClusterPair) : 0;
            if ((isLinfinity && readChunkSeqCount == 0) || splitMemoryLimit <= keptBytes + readChunkBufferBytes + pairBytes + hashBytes) {
                Debug(Debug::ERROR) << "Not enough memory to hash " << lengthSeqCount << " sequences\n"
                                    << "Raise RAM size or --split-memory-limit.\n";
                EXIT(EXIT_FAILURE);
            }
            // a split linfinity length keeps its hashIdPairs and, after its candidates, the sequences of a copy batch in hashIdPairs
            hashIdPairsCapacity[i] = (splitMemoryLimit - keptBytes - readChunkBufferBytes - pairBytes) / sizeof(HashEntry);
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
    // a large hash group keeps one thread busy, so the cluster step cuts more tasks than threads, like clusthash does with its bins
    std::vector<size_t> taskOffsets(par.threads * 16 + 1);
    std::vector<std::string> threadClusterText(par.threads);
    // (representative id, offset) of the entries of the task of a thread; linfinity: (key, offset), the offset in the data file once written
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
        const size_t splitFirstKey = nodeLengthRanges[splitRangeOffsets[split]].startKey;
        const size_t splitSeqCount = nodeLengthRanges[splitRangeOffsets[split + 1]].startKey - splitFirstKey;
        const bool isLengthLargerThanMemory = hashIdPairsCapacity[splitRangeOffsets[split]] > 0;
        const size_t length = isLinfinity ? nodeLengthRanges[splitRangeOffsets[split]].entryLength - 2 : 0;
        const bool isCopyingSeqs = isLengthLargerThanMemory && isLinfinity;
        if (isCopyingSeqs) {
            splitPairs = new(std::nothrow) ClusterPair[splitSeqCount];
            Util::checkAllocation(splitPairs, "Can not allocate splitPairs memory in clusthashfast");
            splitPairCount = 0;
        }
        const size_t readChunkSeqCount = isCopyingSeqs ? std::min(readChunkBytes / (length + 2) / 64 * 64, splitSeqCount) : splitSeqCount;
        const size_t hashedSeqCapacity = isLengthLargerThanMemory ? hashIdPairsCapacity[splitRangeOffsets[split]] : splitSeqCount;
        bool canStoreHashIdPairs = hashedSeqCapacity >= splitSeqCount;
        bool isNextSplitPrefetched = false;
        if (allocatedHashEntryCount < hashedSeqCapacity) {
            Util::releasePages(hashIdPairs, allocatedHashEntryCount * sizeof(HashEntry));
            delete[] hashIdPairs;
            hashIdPairs = new(std::nothrow) HashEntry[hashedSeqCapacity];
            Util::checkAllocation(hashIdPairs, "Can not allocate hash entry memory");
            allocatedHashEntryCount = hashedSeqCapacity;
        }

        // [first, second) of 16 bit hash values of a split length
        std::vector<std::pair<size_t, size_t>> hashRanges(1, std::make_pair(0, USHRT_MAX + 1));
        // copy batch b holds hashIdPairs [seqCopyBatchOffsets[b], seqCopyBatchOffsets[b + 1]) of a split linfinity length whose hashIdPairs fit
        std::vector<size_t> seqCopyBatchOffsets(1, 0);
        MemberBits *memberBits = NULL;
        char *batchSeqs = NULL;
        if (isLengthLargerThanMemory) {
            memberBits = new(std::nothrow) MemberBits[splitSeqCount / 64 + 1];
            Util::checkAllocation(memberBits, "Can not allocate member bit memory");
        }
        if (isCopyingSeqs && canStoreHashIdPairs) {
            for (size_t readChunkBegin = 0; readChunkBegin < splitSeqCount; readChunkBegin += readChunkSeqCount) {
                const size_t readChunkEnd = loadReadChunk(reader, splitFirstKey, splitSeqCount, readChunkSeqCount, readChunkBegin);
                fillHashIdPairs(reader, &subMat, par.maxSeqLen, nodeLengthRanges.data() + splitRangeOffsets[split], splitRangeOffsets[split + 1] - splitRangeOffsets[split], splitFirstKey + readChunkBegin, splitFirstKey + readChunkEnd, hashIdPairs + readChunkBegin);
            }
            SORT_PARALLEL(hashIdPairs, hashIdPairs + splitSeqCount, HashEntry::compareByHashAndId);
            const size_t candidateCount = removeSingletons(hashIdPairs, splitSeqCount, representativeBits, localFirstKey, taskOffsets);
            // the candidates are cut at hash groups into copy batches whose copies fit
            const size_t copyCapacity = (hashedSeqCapacity - candidateCount) * sizeof(HashEntry) / length;
            for (size_t hashGroupBegin = 0; hashGroupBegin < candidateCount;) {
                size_t hashGroupEnd = hashGroupBegin + 1;
                while (hashGroupEnd < candidateCount && hashIdPairs[hashGroupEnd].hash == hashIdPairs[hashGroupBegin].hash) {
                    hashGroupEnd++;
                }
                // the copies of one hash group do not fit next to all hashIdPairs, so the length is hashed once per hash range
                if (hashGroupEnd - hashGroupBegin > copyCapacity) {
                    canStoreHashIdPairs = false;
                    break;
                }
                if (hashGroupEnd - seqCopyBatchOffsets.back() > copyCapacity) {
                    seqCopyBatchOffsets.push_back(hashGroupBegin);
                }
                hashGroupBegin = hashGroupEnd;
            }
            if (candidateCount > seqCopyBatchOffsets.back()) {
                seqCopyBatchOffsets.push_back(candidateCount);
            }
            batchSeqs = reinterpret_cast<char *>(hashIdPairs + candidateCount);
        }
        if (isLengthLargerThanMemory && canStoreHashIdPairs == false) {
            // the sequences of a hash range of a linfinity length are copied after its hashIdPairs
            const size_t maxRangeSeqCount = hashedSeqCapacity * sizeof(HashEntry) / (sizeof(HashEntry) + length);
            hashRanges = setupHashRanges(reader, &subMat, par.maxSeqLen, splitFirstKey, splitSeqCount, readChunkSeqCount, maxRangeSeqCount);
            batchSeqs = isCopyingSeqs ? reinterpret_cast<char *>(hashIdPairs + maxRangeSeqCount) : NULL;
        } else if (isLengthLargerThanMemory == false) {
            // the split is hashed while its later parts are read
            reader.startLoadData(splitFirstKey, splitFirstKey + splitSeqCount, readChunkBytes);
            for (size_t readChunkBegin = splitFirstKey; readChunkBegin < splitFirstKey + splitSeqCount;) {
                const size_t readChunkEnd = reader.waitForLoadedData();
                // once the split is read, the next one is read during its last hashing if both sequence buffers fit next to the hashIdPairs of both
                if (readChunkEnd == splitFirstKey + splitSeqCount && isLinfinity && split + 2 < splitRangeOffsets.size() && hashIdPairsCapacity[splitRangeOffsets[split + 1]] == 0) {
                    const size_t nextSplitFirstKey = nodeLengthRanges[splitRangeOffsets[split + 1]].startKey;
                    const size_t nextSplitEndKey = nodeLengthRanges[splitRangeOffsets[split + 2]].startKey;
                    isNextSplitPrefetched = reader.getPrefetchBufferBytes(nextSplitFirstKey, nextSplitEndKey) + std::max(allocatedHashEntryCount, nextSplitEndKey - nextSplitFirstKey) * sizeof(HashEntry) <= splitMemoryLimit;
                    if (isNextSplitPrefetched) {
                        reader.prefetchData(nextSplitFirstKey, nextSplitEndKey);
                    }
                }
                fillHashIdPairs(reader, &subMat, par.maxSeqLen, nodeLengthRanges.data() + splitRangeOffsets[split], splitRangeOffsets[split + 1] - splitRangeOffsets[split], readChunkBegin, readChunkEnd, hashIdPairs + (readChunkBegin - splitFirstKey));
                readChunkBegin = readChunkEnd;
            }
        }
        const size_t rangeCount = (isCopyingSeqs && canStoreHashIdPairs) ? seqCopyBatchOffsets.size() - 1 : hashRanges.size();
        needsSort |= rangeCount > 1;
        for (size_t range = 0; range < rangeCount; range++) {
            HashEntry *rangeHashIdPairs = hashIdPairs;
            size_t hashedSeqCount = 0;
            if (isCopyingSeqs && canStoreHashIdPairs) {
                // hashIdPairs of a copy batch are sorted by hash and id already
                rangeHashIdPairs = hashIdPairs + seqCopyBatchOffsets[range];
                hashedSeqCount = seqCopyBatchOffsets[range + 1] - seqCopyBatchOffsets[range];
            } else {
                hashedSeqCount = isLengthLargerThanMemory
                               ? fillHashRange(reader, &subMat, par.maxSeqLen, splitFirstKey, splitSeqCount, readChunkSeqCount, hashRanges[range], hashIdPairs, memberBits)
                               : splitSeqCount;
                SORT_PARALLEL(hashIdPairs, hashIdPairs + hashedSeqCount, HashEntry::compareByHashAndId);
                // a linfinity singleton only sets its bit, so it is neither clustered, sorted again nor written
                if (isLinfinity) {
                    hashedSeqCount = removeSingletons(hashIdPairs, hashedSeqCount, representativeBits, localFirstKey, taskOffsets);
                }
            }
            if (isCopyingSeqs) {
                copyBatchSeqs(reader, splitFirstKey, splitSeqCount, readChunkSeqCount, rangeHashIdPairs, hashedSeqCount, memberBits, batchSeqs);
            }

            // hash groups never cross tasks; taskOffsets are fixed before hash is overwritten with the representative id
            cutThreadOffsets(rangeHashIdPairs, 0, hashedSeqCount, taskOffsets);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t task = 0; task < taskOffsets.size() - 1; task++) {
                unsigned int thread = 0;
#ifdef OPENMP
                thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
                const size_t taskEnd = taskOffsets[task + 1];
                std::string repSeqCopy;
                for (size_t hashGroupBegin = taskOffsets[task]; hashGroupBegin < taskEnd;) {
                    size_t hashGroupEnd = hashGroupBegin + 1;
                    while (hashGroupEnd < taskEnd && rangeHashIdPairs[hashGroupEnd].hash == rangeHashIdPairs[hashGroupBegin].hash) {
                        hashGroupEnd++;
                    }
                    // [hashGroupBegin, i) is assigned, [i, hashGroupEnd) is not: the smallest id left is the next representative, members are swapped in front
                    for (size_t i = hashGroupBegin; i < hashGroupEnd;) {
                        std::iter_swap(rangeHashIdPairs + i, std::min_element(rangeHashIdPairs + i, rangeHashIdPairs + hashGroupEnd, HashEntry::compareById));
                        const DBLocalId repId = rangeHashIdPairs[i].id;
                        rangeHashIdPairs[i++].hash = repId;
                        if (i == hashGroupEnd) {
                            break;
                        }
                        const unsigned int repLength = reader.getSeqLen(repId);
                        const char *repSeq = (batchSeqs == NULL) ? reader.getData(repId, thread) : batchSeqs + getMemberIndex(memberBits, repId - splitFirstKey) * length;
                        if (isCopyingRepSeq) {
                            repSeqCopy.assign(repSeq, repLength);
                            repSeq = repSeqCopy.data();
                        }
                        for (size_t j = i; j < hashGroupEnd; j++) {
                            const DBLocalId memberId = rangeHashIdPairs[j].id;
                            if (reader.getSeqLen(memberId) != repLength) {
                                continue;
                            }
                            const unsigned int distance = DistanceCalculator::computeInverseHammingDistance(repSeq, (batchSeqs == NULL) ? reader.getData(memberId, thread) : batchSeqs + getMemberIndex(memberBits, memberId - splitFirstKey) * length, repLength);
                            if (static_cast<float>(distance) / static_cast<float>(repLength) >= par.seqIdThr) {
                                std::swap(rangeHashIdPairs[i], rangeHashIdPairs[j]);
                                rangeHashIdPairs[i++].hash = repId;
                            }
                        }
                    }
                    hashGroupBegin = hashGroupEnd;
                }
            }

            // hash now holds the representative id, so sorting again gathers each cluster with its representative first
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
                    const size_t threadEnd = threadOffsets[thread + 1];
                    for (size_t clusterBegin = threadOffsets[thread]; clusterBegin < threadEnd;) {
                        size_t clusterEnd = clusterBegin + 1;
                        while (clusterEnd < threadEnd && rangeHashIdPairs[clusterEnd].hash == rangeHashIdPairs[clusterBegin].hash) {
                            clusterEnd++;
                        }
                        const DBLocalId repId = rangeHashIdPairs[clusterBegin].id;
                        if (isLinfinity) {
                            __sync_fetch_and_or(&representativeBits[(repId - localFirstKey) / 64], 1ull << ((repId - localFirstKey) % 64));
                        }
                        // the clusters of a split linfinity length are written in key order after its last hash range, the representative first
                        if (isCopyingSeqs) {
                            for (size_t i = clusterBegin; i < clusterEnd; i++) {
                                const ClusterPair pair = { repId, rangeHashIdPairs[i].id };
                                splitPairs[splitPairCount + i] = pair;
                            }
                        } else if (isLinfinity == false || clusterEnd - clusterBegin > 1) {
                            // a linfinity representative without members has its bit and no entry
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
                        const DBLocalId repId = entries[i];
                        const size_t entryEnd = (i + 2 < entries.size()) ? entries[i + 3] : data.size();
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
            const size_t splitEndKey = splitFirstKey + splitSeqCount;
            for (size_t writeBegin = splitFirstKey; writeBegin < splitEndKey; writeBegin += WRITE_SEQS_PER_THREAD * par.threads) {
                const size_t writeKeys = std::min(splitEndKey - writeBegin, WRITE_SEQS_PER_THREAD * par.threads);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                for (int thread = 0; thread < par.threads; thread++) {
                    std::string &data = threadClusterText[thread];
                    std::vector<uint64_t> &entries = threadEntries[thread];
                    data.clear();
                    entries.clear();
                    char keyBuffer[32];
                    const auto findRepKey = [&](size_t key) {
                        return std::lower_bound(splitPairs, splitPairs + splitPairCount, key, [](const ClusterPair &first, size_t second) { return first.repKey < second; }) - splitPairs;
                    };
                    size_t pair = findRepKey(writeBegin + writeKeys * thread / par.threads);
                    const size_t endPair = findRepKey(writeBegin + writeKeys * (thread + 1) / par.threads);
                    while (pair < endPair) {
                        const DBKeyType repKey = splitPairs[pair].repKey;
                        // a linfinity representative without members has its bit and no entry
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
        // only a split and the next one it prefetched keep both sequence buffers and hashIdPairs
        if (isLinfinity && isNextSplitPrefetched == false) {
            reader.releaseDataPages();
            Util::releasePages(hashIdPairs, allocatedHashEntryCount * sizeof(HashEntry));
            delete[] hashIdPairs;
            hashIdPairs = NULL;
            allocatedHashEntryCount = 0;
        }
    }

    if (isLinfinity) {
        std::string phaseValues;
        MMseqsNode::addPhaseTime(phaseValues, "hashing the sequences", timer);
        MMseqsNode::printPhaseValues(phaseValues);
        MMseqsNode::closeClusterFiles(resultDataFile, resultDb, resultIndexFile, resultIndex, resultDataBytes, resultIndexBytes);
        MMseqsNode::writeRepresentativeBits(representativeDb, reader, representativeBits, localFirstKey, localKeyCount);
    } else {
        // clusters are written in key order unless a length was split into hash ranges
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
            // a linfinity subset only feeds the next pipeline steps, which read no headers, lookup or source
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
