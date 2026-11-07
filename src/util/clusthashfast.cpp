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

static size_t hashWithLength(size_t hash, size_t length) {
    return hash ^ (length + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2));
}

// positions 64 * i .. 64 * i + 63 of a split length: which are members of the current hash range or block, and how many members come before them
struct MemberBits {
    uint64_t isMember;
    size_t membersBefore;
};

// index of the sequence at position pos of a split length among the members of the current hash range or block
static size_t memberIndex(const MemberBits *memberBits, size_t pos) {
    return memberBits[pos / 64].membersBefore + __builtin_popcountll(memberBits[pos / 64].isMember & ((1ull << (pos % 64)) - 1));
}

// the sequences whose clusters a thread formats at once before they are written
static const size_t WRITE_THREAD_SEQS = 1024 * 1024;
// a formatted sequence takes at most its key and a newline, a cluster end and an index entry, in buffers that grow to twice their content
static const size_t WRITE_SEQ_BYTES = 2 * (20 + 1 + 1 + 2 * sizeof(uint64_t));

// the first position from pos on in hashIdPairs [0, end), sorted by hash, that does not continue the hash before it
static size_t findHashStart(const HashEntry *hashIdPairs, size_t pos, size_t end) {
    return (pos == 0) ? 0 : std::upper_bound(hashIdPairs + pos, hashIdPairs + end, HashEntry{hashIdPairs[pos - 1].hash, std::numeric_limits<DBLocalId>::max()}, HashEntry::compareByHashAndId) - hashIdPairs;
}

// cuts hashIdPairs [begin, end), sorted by hash, into threadOffsets.size() - 1 parts without cutting a hash
static void cutThreadOffsets(const HashEntry *hashIdPairs, size_t begin, size_t end, std::vector<size_t> &threadOffsets) {
    const size_t threads = threadOffsets.size() - 1;
    for (size_t thread = 0; thread < threads; thread++) {
        threadOffsets[thread] = findHashStart(hashIdPairs, begin + (end - begin) / threads * thread, end);
    }
    threadOffsets[threads] = end;
}

static size_t seqHash(Sequence &seq, size_t id, const char *data, size_t length) {
    seq.mapSequence(id, 0, data, length);
    return hashWithLength(Util::hash(seq.numSequence, seq.L), length);
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

// hash bins of one sequence of hashIdPairs [0, seqCount), sorted by hash, become representatives, the other bins are moved to the front and their count is returned
static size_t removeSingletons(HashEntry *hashIdPairs, size_t seqCount, uint64_t *representativeBits, size_t localFirstKey, std::vector<size_t> &partOffsets) {
    const size_t partCount = partOffsets.size() - 1;
    cutThreadOffsets(hashIdPairs, 0, seqCount, partOffsets);
    std::vector<size_t> partCandidates(partCount);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t part = 0; part < partCount; part++) {
        size_t writePos = partOffsets[part];
        for (size_t hashBinBegin = partOffsets[part]; hashBinBegin < partOffsets[part + 1];) {
            size_t hashBinEnd = hashBinBegin + 1;
            while (hashBinEnd < partOffsets[part + 1] && hashIdPairs[hashBinEnd].hash == hashIdPairs[hashBinBegin].hash) {
                hashBinEnd++;
            }
            if (hashBinEnd - hashBinBegin == 1) {
                const size_t bit = hashIdPairs[hashBinBegin].id - localFirstKey;
                __sync_fetch_and_or(&representativeBits[bit / 64], 1ull << (bit % 64));
            } else {
                for (size_t i = hashBinBegin; i < hashBinEnd; i++) {
                    hashIdPairs[writePos++] = hashIdPairs[i];
                }
            }
            hashBinBegin = hashBinEnd;
        }
        partCandidates[part] = writePos - partOffsets[part];
    }
    // the parts move to the front in order, so no part overwrites one that has not moved yet
    size_t candidateCount = 0;
    for (size_t part = 0; part < partCount; part++) {
        memmove(hashIdPairs + candidateCount, hashIdPairs + partOffsets[part], partCandidates[part] * sizeof(HashEntry));
        candidateCount += partCandidates[part];
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

// counts the sequences of each hash bucket of keys [firstKey, firstKey + seqCount), read readChunkSeqCount at a time,
// and cuts the buckets into hash ranges of at most maxRangeSeqCount sequences, like kmermatcher setupKmerSplits
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
    for (size_t bucket = 0; bucket <= USHRT_MAX; bucket++) {
        if (hashDist[bucket] > maxRangeSeqCount) {
            Debug(Debug::ERROR) << "Not enough memory to hash " << hashDist[bucket] << " sequences of one hash bucket\n"
                                << "Raise RAM size or --split-memory-limit.\n";
            EXIT(EXIT_FAILURE);
        }
        if (rangeSeqCount + hashDist[bucket] > maxRangeSeqCount) {
            hashRanges.emplace_back(rangeStart, bucket);
            rangeStart = bucket;
            rangeSeqCount = 0;
        }
        rangeSeqCount += hashDist[bucket];
    }
    hashRanges.emplace_back(rangeStart, USHRT_MAX + 1);
    delete[] hashDist;
    return hashRanges;
}

// keeps the sequences of keys [firstKey, firstKey + seqCount) whose hash bucket is in hashRange
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
                memberBits[i].isMember = 0;
                for (size_t pos = i * 64; pos < std::min(i * 64 + 64, readChunkEnd); pos++) {
                    const unsigned short bucket = static_cast<unsigned short>(seqHash(seq, firstKey + pos, reader.getData(firstKey + pos, thread_idx), reader.getSeqLen(firstKey + pos)));
                    if (bucket >= hashRange.first && bucket < hashRange.second) {
                        memberBits[i].isMember |= 1ull << (pos % 64);
                    }
                }
            }
#pragma omp single
            for (size_t i = readChunkBegin / 64; i < (readChunkEnd + 63) / 64; i++) {
                memberBits[i].membersBefore = rangeSeqCount;
                rangeSeqCount += __builtin_popcountll(memberBits[i].isMember);
            }
#pragma omp for schedule(dynamic, 1)
            for (size_t i = readChunkBegin / 64; i < (readChunkEnd + 63) / 64; i++) {
                size_t slot = memberBits[i].membersBefore;
                for (uint64_t isMember = memberBits[i].isMember; isMember != 0; isMember &= isMember - 1) {
                    const size_t id = firstKey + i * 64 + __builtin_ctzll(isMember);
                    hashIdPairs[slot].hash = seqHash(seq, id, reader.getData(id, thread_idx), reader.getSeqLen(id));
                    hashIdPairs[slot].id = id;
                    slot++;
                }
            }
        }
    }
    return rangeSeqCount;
}

// copies the sequences of blockHashIdPairs, keys of [firstKey, firstKey + seqCount), to their slots in blockSeqs, reading only their pages of each readChunkSeqCount keys
static void loadBlockSeqs(DBReader<DBKeyType> &reader, size_t firstKey, size_t seqCount, size_t readChunkSeqCount, const HashEntry *blockHashIdPairs, size_t blockSeqCount,
                          MemberBits *memberBits, char *blockSeqs) {
    const size_t length = reader.getSeqLen(firstKey);
    memset(memberBits, 0, sizeof(MemberBits) * (seqCount / 64 + 1));
#pragma omp parallel for
    for (size_t i = 0; i < blockSeqCount; i++) {
        const size_t pos = blockHashIdPairs[i].id - firstKey;
        __sync_fetch_and_or(&memberBits[pos / 64].isMember, 1ull << (pos % 64));
    }
    size_t memberCount = 0;
    for (size_t i = 0; i < seqCount / 64 + 1; i++) {
        memberBits[i].membersBefore = memberCount;
        memberCount += __builtin_popcountll(memberBits[i].isMember);
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
            for (uint64_t isMember = memberBits[i].isMember; isMember != 0; isMember &= isMember - 1) {
                readChunkKeys[member++] = firstKey + i * 64 + __builtin_ctzll(isMember);
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
                for (uint64_t isMember = memberBits[i].isMember; isMember != 0; isMember &= isMember - 1) {
                    const size_t id = firstKey + i * 64 + __builtin_ctzll(isMember);
                    memcpy(blockSeqs + member * length, reader.getData(id, thread_idx), length);
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
    const bool copyRepSeq = reader.isCompressed() || (DBReader<DBKeyType>::getExtendedDbtype(reader.getDbtype()) & Parameters::DBTYPE_EXTENDED_GPU);
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(reader.getDbtype()) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity == false && par.preloadMode != Parameters::PRELOAD_MODE_MMAP) {
        reader.readMmapedDataInMemory();
    }

    std::vector<LengthRange> nodeSeqRange;
    if (isLinfinity) {
        const size_t endKey = reader.getFileKeyOffsets(MMseqsNode::count)[MMseqsNode::rank + 1];
        const DBReader<DBKeyType>::Index *index = reader.getIndex();
        size_t endOffset = 0;
        for (size_t range = 0; range < reader.getIndexSize(); range++) {
            if (reader.getIndexFile(range) == MMseqsNode::rank) {
                nodeSeqRange.push_back({index[range].id, index[range].offset, index[range].length});
                endOffset = index[range].offset + (endKey - index[range].id) * index[range].length;
            }
        }
        nodeSeqRange.push_back({endKey, endOffset, 0});
    } else {
        nodeSeqRange.push_back({0, 0, 0});
        nodeSeqRange.push_back({reader.getSize(), reader.getDataSize(), 0});
    }
    const size_t localFirstKey = nodeSeqRange.front().startKey;
    const size_t localKeySize = nodeSeqRange.back().startKey - localFirstKey;

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
    const size_t representativeBitsSize = isLinfinity ? (localKeySize + 63) / 64 : 0;
    uint64_t *representativeBits = new(std::nothrow) uint64_t[representativeBitsSize]();
    Util::checkAllocation(representativeBits, "Can not allocate representative bit memory");
    const size_t writeBufferBytes = par.threads * WRITE_THREAD_SEQS * WRITE_SEQ_BYTES;
    if (representativeBitsSize * sizeof(uint64_t) + writeBufferBytes >= memoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to run clusthashfast. Memory limit: " << memoryLimit
                            << " bytes, the representative bits and write buffers need: " << representativeBitsSize * sizeof(uint64_t) + writeBufferBytes << " bytes.\n"
                            << "Raise RAM size or --split-memory-limit.\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t splitMemoryLimit = static_cast<size_t>(static_cast<double>(memoryLimit - representativeBitsSize * sizeof(uint64_t) - writeBufferBytes) * 0.95);

    // consecutive lengths that fit together are loaded at once;
    // a length that does not fit is loaded alone: a linfinity length is read one read chunk at a time to hash it and once per block of candidates, a standard DB length once per hash range that fits, after 2 bits per key
    const size_t readChunkBytes = std::min(MAX_INFLIGHT_READ_SIZE, splitMemoryLimit / 4);
    const size_t seqBufferLimit = isLinfinity ? splitMemoryLimit / 3 : 0;
    const size_t hashIdPairsLimit = splitMemoryLimit - 2 * seqBufferLimit;
    std::vector<size_t> hashIdPairsCapacity(nodeSeqRange.size() - 1, 0);
    std::vector<size_t> loadOffsets;
    for (size_t i = 0; i + 1 < nodeSeqRange.size(); i++) {
        const size_t lengthSeqCount = nodeSeqRange[i + 1].startKey - nodeSeqRange[i].startKey;
        const size_t lengthBytes = (isLinfinity ? nodeSeqRange[i + 1].offset - nodeSeqRange[i].offset : 0) + lengthSeqCount * sizeof(HashEntry);
        if (lengthBytes > splitMemoryLimit) {
            const size_t readChunkSeqCount = isLinfinity ? std::min(readChunkBytes / nodeSeqRange[i].entryLength / 64 * 64, lengthSeqCount) : 0;
            const size_t readChunkBufferBytes = 2 * reader.getBufferSizeToLoad(nodeSeqRange[i].startKey, nodeSeqRange[i].startKey + readChunkSeqCount);
            const size_t keptBytes = isLinfinity ? (lengthSeqCount / 64 + 1) * sizeof(MemberBits) : lengthSeqCount / 4;
            const size_t hashBytes = sizeof(HashEntry) + (isLinfinity ? nodeSeqRange[i].entryLength - 2 : 0);
            // a split linfinity length keeps a pair for every member to write its clusters in key order
            const size_t pairBytes = isLinfinity ? lengthSeqCount * sizeof(ClusterPair) : 0;
            if ((isLinfinity && readChunkSeqCount == 0) || splitMemoryLimit <= keptBytes + readChunkBufferBytes + pairBytes + hashBytes) {
                Debug(Debug::ERROR) << "Not enough memory to hash " << lengthSeqCount << " sequences\n"
                                    << "Raise RAM size or --split-memory-limit.\n";
                EXIT(EXIT_FAILURE);
            }
            // a split linfinity length keeps its hashIdPairs and, after its candidates, the sequences of a block in hashIdPairs
            hashIdPairsCapacity[i] = (splitMemoryLimit - keptBytes - readChunkBufferBytes - pairBytes) / sizeof(HashEntry);
        }
        if (loadOffsets.empty()
            || (isLinfinity ? nodeSeqRange[i + 1].offset - nodeSeqRange[loadOffsets.back()].offset + 2 * Util::getPageSize() : 0) > seqBufferLimit
            || (nodeSeqRange[i + 1].startKey - nodeSeqRange[loadOffsets.back()].startKey) * sizeof(HashEntry) > hashIdPairsLimit) {
            loadOffsets.push_back(i);
        }
    }
    loadOffsets.push_back(nodeSeqRange.size() - 1);
    HashEntry *hashIdPairs = NULL;
    size_t hashIdPairsSize = 0;

    std::vector<size_t> threadOffsets(par.threads + 1);
    // a large hash bin keeps one thread busy, so the cluster step cuts more parts than threads, like clusthash does with its bins
    std::vector<size_t> partOffsets(par.threads * 16 + 1);
    std::vector<std::string> threadData(par.threads);
    // (representative id, offset) of the entries of the part of a thread; linfinity: (key, offset), the offset in the data file once written
    std::vector<std::vector<uint64_t> > threadEntries(par.threads);
    ClusterPair *loadPairs = NULL;
    size_t loadPairCount = 0;
    bool needsSort = false;
    size_t clusters = 0;
    SubstitutionMatrix sMat(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.2);
    ReducedMatrix subMat(sMat.probMatrix, sMat.subMatrixPseudoCounts, sMat.aa2num, sMat.num2aa, sMat.alphabetSize, par.alphabetSize.values.aminoacid(), 2.0);

    Debug(Debug::INFO) << "Hashing " << localKeySize << " sequences in " << loadOffsets.size() - 1 << " loads\n";
    Timer timer;
    Debug::Progress progress(loadOffsets.size() - 1);
    for (size_t load = 0; load + 1 < loadOffsets.size(); load++) {
        progress.updateProgress();
        const size_t loadFirstKey = nodeSeqRange[loadOffsets[load]].startKey;
        const size_t loadSeqCount = nodeSeqRange[loadOffsets[load + 1]].startKey - loadFirstKey;
        const bool isSplitLength = hashIdPairsCapacity[loadOffsets[load]] > 0;
        const size_t length = isLinfinity ? nodeSeqRange[loadOffsets[load]].entryLength - 2 : 0;
        const bool copiesSeqs = isSplitLength && isLinfinity;
        if (copiesSeqs) {
            loadPairs = new(std::nothrow) ClusterPair[loadSeqCount];
            Util::checkAllocation(loadPairs, "Can not allocate loadPairs memory in clusthashfast");
            loadPairCount = 0;
        }
        const size_t readChunkSeqCount = copiesSeqs ? std::min(readChunkBytes / (length + 2) / 64 * 64, loadSeqCount) : loadSeqCount;
        const size_t hashedSeqCapacity = isSplitLength ? hashIdPairsCapacity[loadOffsets[load]] : loadSeqCount;
        bool hashIdPairsFit = hashedSeqCapacity >= loadSeqCount;
        bool prefetchesNextLoad = false;
        if (hashIdPairsSize < hashedSeqCapacity) {
            Util::releasePages(hashIdPairs, hashIdPairsSize * sizeof(HashEntry));
            delete[] hashIdPairs;
            hashIdPairs = new(std::nothrow) HashEntry[hashedSeqCapacity];
            Util::checkAllocation(hashIdPairs, "Can not allocate hash entry memory");
            hashIdPairsSize = hashedSeqCapacity;
        }

        // [first, second) of hash buckets of a split length
        std::vector<std::pair<size_t, size_t>> hashRanges(1, std::make_pair(0, USHRT_MAX + 1));
        // block b holds hashIdPairs [blockOffsets[b], blockOffsets[b + 1]) of a split linfinity length whose hashIdPairs fit
        std::vector<size_t> blockOffsets(1, 0);
        MemberBits *memberBits = NULL;
        char *blockSeqs = NULL;
        if (isSplitLength) {
            memberBits = new(std::nothrow) MemberBits[loadSeqCount / 64 + 1];
            Util::checkAllocation(memberBits, "Can not allocate member bit memory");
        }
        if (copiesSeqs && hashIdPairsFit) {
            for (size_t readChunkBegin = 0; readChunkBegin < loadSeqCount; readChunkBegin += readChunkSeqCount) {
                const size_t readChunkEnd = loadReadChunk(reader, loadFirstKey, loadSeqCount, readChunkSeqCount, readChunkBegin);
                fillHashIdPairs(reader, &subMat, par.maxSeqLen, nodeSeqRange.data() + loadOffsets[load], loadOffsets[load + 1] - loadOffsets[load], loadFirstKey + readChunkBegin, loadFirstKey + readChunkEnd, hashIdPairs + readChunkBegin);
            }
            SORT_PARALLEL(hashIdPairs, hashIdPairs + loadSeqCount, HashEntry::compareByHashAndId);
            const size_t candidateCount = removeSingletons(hashIdPairs, loadSeqCount, representativeBits, localFirstKey, partOffsets);
            // the candidates are cut at hash bins into blocks whose copies fit
            const size_t copyCapacity = (hashedSeqCapacity - candidateCount) * sizeof(HashEntry) / length;
            for (size_t hashBinBegin = 0; hashBinBegin < candidateCount;) {
                size_t hashBinEnd = hashBinBegin + 1;
                while (hashBinEnd < candidateCount && hashIdPairs[hashBinEnd].hash == hashIdPairs[hashBinBegin].hash) {
                    hashBinEnd++;
                }
                // the copies of one hash bin do not fit next to all hashIdPairs, so the length is hashed once per hash range
                if (hashBinEnd - hashBinBegin > copyCapacity) {
                    hashIdPairsFit = false;
                    break;
                }
                if (hashBinEnd - blockOffsets.back() > copyCapacity) {
                    blockOffsets.push_back(hashBinBegin);
                }
                hashBinBegin = hashBinEnd;
            }
            if (candidateCount > blockOffsets.back()) {
                blockOffsets.push_back(candidateCount);
            }
            blockSeqs = reinterpret_cast<char *>(hashIdPairs + candidateCount);
        }
        if (isSplitLength && hashIdPairsFit == false) {
            // the sequences of a hash range of a linfinity length are copied after its hashIdPairs
            const size_t maxRangeSeqCount = hashedSeqCapacity * sizeof(HashEntry) / (sizeof(HashEntry) + length);
            hashRanges = setupHashRanges(reader, &subMat, par.maxSeqLen, loadFirstKey, loadSeqCount, readChunkSeqCount, maxRangeSeqCount);
            blockSeqs = copiesSeqs ? reinterpret_cast<char *>(hashIdPairs + maxRangeSeqCount) : NULL;
        } else if (isSplitLength == false) {
            // the load is hashed while its later parts are read
            reader.startLoadData(loadFirstKey, loadFirstKey + loadSeqCount, readChunkBytes);
            for (size_t readChunkBegin = loadFirstKey; readChunkBegin < loadFirstKey + loadSeqCount;) {
                const size_t readChunkEnd = reader.waitForLoadedData();
                // once the load is read, the next one is read during its last hashing if both sequence buffers fit next to the hashIdPairs of both
                if (readChunkEnd == loadFirstKey + loadSeqCount && isLinfinity && load + 2 < loadOffsets.size() && hashIdPairsCapacity[loadOffsets[load + 1]] == 0) {
                    const size_t nextLoadFirstKey = nodeSeqRange[loadOffsets[load + 1]].startKey;
                    const size_t nextLoadEndKey = nodeSeqRange[loadOffsets[load + 2]].startKey;
                    prefetchesNextLoad = reader.getBufferSizeToPrefetch(nextLoadFirstKey, nextLoadEndKey) + std::max(hashIdPairsSize, nextLoadEndKey - nextLoadFirstKey) * sizeof(HashEntry) <= splitMemoryLimit;
                    if (prefetchesNextLoad) {
                        reader.prefetchData(nextLoadFirstKey, nextLoadEndKey);
                    }
                }
                fillHashIdPairs(reader, &subMat, par.maxSeqLen, nodeSeqRange.data() + loadOffsets[load], loadOffsets[load + 1] - loadOffsets[load], readChunkBegin, readChunkEnd, hashIdPairs + (readChunkBegin - loadFirstKey));
                readChunkBegin = readChunkEnd;
            }
        }
        const size_t rangeCount = (copiesSeqs && hashIdPairsFit) ? blockOffsets.size() - 1 : hashRanges.size();
        needsSort |= rangeCount > 1;
        for (size_t range = 0; range < rangeCount; range++) {
            HashEntry *rangeHashIdPairs = hashIdPairs;
            size_t hashedSeqCount = 0;
            if (copiesSeqs && hashIdPairsFit) {
                // hashIdPairs of a block are sorted by hash and id already
                rangeHashIdPairs = hashIdPairs + blockOffsets[range];
                hashedSeqCount = blockOffsets[range + 1] - blockOffsets[range];
            } else {
                hashedSeqCount = isSplitLength
                               ? fillHashRange(reader, &subMat, par.maxSeqLen, loadFirstKey, loadSeqCount, readChunkSeqCount, hashRanges[range], hashIdPairs, memberBits)
                               : loadSeqCount;
                SORT_PARALLEL(hashIdPairs, hashIdPairs + hashedSeqCount, HashEntry::compareByHashAndId);
                // a linfinity singleton only sets its bit, so it is neither clustered, sorted again nor written
                if (isLinfinity) {
                    hashedSeqCount = removeSingletons(hashIdPairs, hashedSeqCount, representativeBits, localFirstKey, partOffsets);
                }
            }
            if (copiesSeqs) {
                loadBlockSeqs(reader, loadFirstKey, loadSeqCount, readChunkSeqCount, rangeHashIdPairs, hashedSeqCount, memberBits, blockSeqs);
            }

            // bins never cross parts; partOffsets are fixed before hash is overwritten with the representative id
            cutThreadOffsets(rangeHashIdPairs, 0, hashedSeqCount, partOffsets);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t part = 0; part < partOffsets.size() - 1; part++) {
                unsigned int thread = 0;
#ifdef OPENMP
                thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
                const size_t partEnd = partOffsets[part + 1];
                std::string repSeqCopy;
                for (size_t hashBinBegin = partOffsets[part]; hashBinBegin < partEnd;) {
                    size_t hashBinEnd = hashBinBegin + 1;
                    while (hashBinEnd < partEnd && rangeHashIdPairs[hashBinEnd].hash == rangeHashIdPairs[hashBinBegin].hash) {
                        hashBinEnd++;
                    }
                    // [hashBinBegin, i) is assigned, [i, hashBinEnd) is not: the smallest id left is the next representative, members are swapped in front
                    for (size_t i = hashBinBegin; i < hashBinEnd;) {
                        std::iter_swap(rangeHashIdPairs + i, std::min_element(rangeHashIdPairs + i, rangeHashIdPairs + hashBinEnd, HashEntry::compareById));
                        const DBLocalId repId = rangeHashIdPairs[i].id;
                        rangeHashIdPairs[i++].hash = repId;
                        if (i == hashBinEnd) {
                            break;
                        }
                        const unsigned int repLength = reader.getSeqLen(repId);
                        const char *repSeq = (blockSeqs == NULL) ? reader.getData(repId, thread) : blockSeqs + memberIndex(memberBits, repId - loadFirstKey) * length;
                        if (copyRepSeq) {
                            repSeqCopy.assign(repSeq, repLength);
                            repSeq = repSeqCopy.data();
                        }
                        for (size_t j = i; j < hashBinEnd; j++) {
                            const DBLocalId memberId = rangeHashIdPairs[j].id;
                            if (reader.getSeqLen(memberId) != repLength) {
                                continue;
                            }
                            const unsigned int distance = DistanceCalculator::computeInverseHammingDistance(repSeq, (blockSeqs == NULL) ? reader.getData(memberId, thread) : blockSeqs + memberIndex(memberBits, memberId - loadFirstKey) * length, repLength);
                            if (static_cast<float>(distance) / static_cast<float>(repLength) >= par.seqIdThr) {
                                std::swap(rangeHashIdPairs[i], rangeHashIdPairs[j]);
                                rangeHashIdPairs[i++].hash = repId;
                            }
                        }
                    }
                    hashBinBegin = hashBinEnd;
                }
            }

            // hash now holds the representative id, so sorting again gathers each cluster with its representative first
            SORT_PARALLEL(rangeHashIdPairs, rangeHashIdPairs + hashedSeqCount, HashEntry::compareByHashAndId);
            for (size_t writeBegin = 0, writeEnd; writeBegin < hashedSeqCount; writeBegin = writeEnd) {
                writeEnd = findHashStart(rangeHashIdPairs, std::min(hashedSeqCount, writeBegin + WRITE_THREAD_SEQS * par.threads), hashedSeqCount);
                cutThreadOffsets(rangeHashIdPairs, writeBegin, writeEnd, threadOffsets);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                for (int thread = 0; thread < par.threads; thread++) {
                    std::string &data = threadData[thread];
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
                        if (copiesSeqs) {
                            for (size_t i = clusterBegin; i < clusterEnd; i++) {
                                const ClusterPair pair = { repId, rangeHashIdPairs[i].id };
                                loadPairs[loadPairCount + i] = pair;
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
                    MMseqsNode::writeClusterThreadParts(resultDataFile, resultDb, resultIndexFile, resultIndex, threadData, threadEntries, resultDataBytes, resultIndexBytes);
                    continue;
                }
                for (int thread = 0; thread < par.threads; thread++) {
                    const std::string &data = threadData[thread];
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
            if (copiesSeqs) {
                loadPairCount += hashedSeqCount;
            }
        }
        if (copiesSeqs) {
            SORT_PARALLEL(loadPairs, loadPairs + loadPairCount, ClusterPair::compareByRep);
            const size_t loadEndKey = loadFirstKey + loadSeqCount;
            for (size_t writeBegin = loadFirstKey; writeBegin < loadEndKey; writeBegin += WRITE_THREAD_SEQS * par.threads) {
                const size_t writeKeys = std::min(loadEndKey - writeBegin, WRITE_THREAD_SEQS * par.threads);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                for (int thread = 0; thread < par.threads; thread++) {
                    std::string &data = threadData[thread];
                    std::vector<uint64_t> &entries = threadEntries[thread];
                    data.clear();
                    entries.clear();
                    char keyBuffer[32];
                    const auto findRepKey = [&](size_t key) {
                        return std::lower_bound(loadPairs, loadPairs + loadPairCount, key, [](const ClusterPair &first, size_t second) { return first.repKey < second; }) - loadPairs;
                    };
                    size_t pair = findRepKey(writeBegin + writeKeys * thread / par.threads);
                    const size_t endPair = findRepKey(writeBegin + writeKeys * (thread + 1) / par.threads);
                    while (pair < endPair) {
                        const DBKeyType repKey = loadPairs[pair].repKey;
                        // a linfinity representative without members has its bit and no entry
                        if (pair + 1 == endPair || loadPairs[pair + 1].repKey != repKey) {
                            pair++;
                            continue;
                        }
                        entries.push_back(repKey);
                        entries.push_back(data.size());
                        for (; pair < endPair && loadPairs[pair].repKey == repKey; pair++) {
                            char *end = Itoa::u64toa_sse2(static_cast<uint64_t>(loadPairs[pair].memberKey), keyBuffer);
                            *(end - 1) = '\n';
                            data.append(keyBuffer, end - keyBuffer);
                        }
                        data.push_back('\0');
                    }
                }
                MMseqsNode::writeClusterThreadParts(resultDataFile, resultDb, resultIndexFile, resultIndex, threadData, threadEntries, resultDataBytes, resultIndexBytes);
            }
            Util::releasePages(loadPairs, loadSeqCount * sizeof(ClusterPair));
            delete[] loadPairs;
            loadPairs = NULL;
        }
        delete[] memberBits;
        // only a load and the next one it prefetched keep both sequence buffers and hashIdPairs
        if (isLinfinity && prefetchesNextLoad == false) {
            reader.releaseDataPages();
            Util::releasePages(hashIdPairs, hashIdPairsSize * sizeof(HashEntry));
            delete[] hashIdPairs;
            hashIdPairs = NULL;
            hashIdPairsSize = 0;
        }
    }

    if (isLinfinity) {
        std::string phaseValues;
        MMseqsNode::addPhaseTime(phaseValues, "hashing the sequences", timer);
        MMseqsNode::addPhaseValue(phaseValues, "Sequences hashed", SSTR(localKeySize));
        MMseqsNode::printPhaseValues(phaseValues);
        MMseqsNode::publishRunFile(par.db2 + ".phases." + SSTR(MMseqsNode::rank), "");
        MMseqsNode::closeClusterFiles(resultDataFile, resultDb, resultIndexFile, resultIndex, resultDataBytes, resultIndexBytes);
        MMseqsNode::writeRepresentativeBits(representativeDb, reader, representativeBits, localFirstKey, localKeySize);
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
            MMseqsNode::gatherRunFiles(par.db2 + ".phases.");
            // a linfinity subset only feeds the next pipeline steps, which read no headers, lookup or source
            DBWriter::writeDbtypeFile(par.db2.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_CLUSTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
        } else {
            DBReader<DBKeyType>::softlinkDb(par.db1, representativeDb, DBFiles::SEQUENCE_ANCILLARY);
        }
        Debug(Debug::INFO) << "Size of the sequence database: " << reader.getSize() << "\n";
        Debug(Debug::INFO) << "Number of clusters: " << clusters << "\n";
    }
    delete[] representativeBits;
    Util::releasePages(hashIdPairs, hashIdPairsSize * sizeof(HashEntry));
    delete[] hashIdPairs;
    reader.close();
    return EXIT_SUCCESS;
}
