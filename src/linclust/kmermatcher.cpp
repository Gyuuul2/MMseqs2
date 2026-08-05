// include xxhash early to avoid incompatibilites with SIMDe
#define XXH_INLINE_ALL
#include "xxhash.h"


#include "kmermatcher.h"
#include "Debug.h"
#include "Indexer.h"
#include "SubstitutionMatrix.h"
#include "ReducedMatrix.h"
#include "ExtendedSubstitutionMatrix.h"
#include "NucleotideMatrix.h"
#include "QueryMatcher.h"
#include "KmerGenerator.h"
#include "MarkovKmerScore.h"
#include "FileUtil.h"
#include "FastSort.h"
#include "SequenceWeights.h"
#include "Masker.h"

#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <limits>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <zstd.h>

#ifdef OPENMP
#include <omp.h>
#endif
#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

// one page table of sequences per chunk, capped so every thread still gets many chunks for balance
static size_t kmerScanChunkSize(DBReader<DBKeyType> &reader, size_t startId, size_t cnt, int threads) {
    const size_t DEFAULT_CHUNK = 100;
    const size_t MIN_CHUNKS_PER_THREAD = 16;
    if (threads < 2 || cnt == 0 || reader.getDataFileCnt() != 1 || reader.isSortedByOffset() == false) {
        return DEFAULT_CHUNK;
    }
    size_t fileSize = reader.getDataSizeForFile(0);
    if (fileSize == 0) {
        return DEFAULT_CHUNK;
    }
    size_t firstOffset = reader.getIndex(startId)->offset;
    size_t lastId = startId + cnt - 1;
    // a soft-linked createdb-mode 1 db has no room for the entry terminator, so clamp to the file
    size_t lastEnd = std::min(reader.getIndex(lastId)->offset + reader.getEntryLen(lastId), fileSize);
    if (lastEnd <= firstOffset) {
        return DEFAULT_CHUNK;
    }
    // one page table maps pageSize/sizeof(void*) pages, so derive its span instead of hardcoding 2 MB
    size_t pageSize = Util::getPageSize();
    size_t pmdSpan = pageSize * (pageSize / sizeof(void *));
    size_t stride = std::max<size_t>((lastEnd - firstOffset) / cnt, 1);
    size_t wanted = (pmdSpan + stride - 1) / stride;
    size_t balanced = cnt / (static_cast<size_t>(threads) * MIN_CHUNKS_PER_THREAD);
    return std::max(std::min(wanted, balanced), DEFAULT_CHUNK);
}

uint64_t hashUInt64(uint64_t in, uint64_t seed) {
#if SIMDE_ENDIAN_ORDER == SIMDE_ENDIAN_BIG
    in = __builtin_bswap64(in);
#endif
    return XXH64(&in, sizeof(uint64_t), seed);
}

template <typename T, bool includeAdjacency, bool IncludeSeqLen>
KmerPosition<T, includeAdjacency, IncludeSeqLen> *initKmerPositionMemory(size_t size) {
    KmerPosition<T, includeAdjacency, IncludeSeqLen> * hashSeqPair = new(std::nothrow) KmerPosition<T, includeAdjacency, IncludeSeqLen>[size + 1];
    Util::checkAllocation(hashSeqPair, "Can not allocate memory");
    size_t pageSize = Util::getPageSize()/sizeof(KmerPosition<T, includeAdjacency, IncludeSeqLen>);
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (size_t page = 0; page < size+1; page += pageSize) {
            size_t readUntil = std::min(size+1, page + pageSize) - page;
            memset(hashSeqPair+page, 0xFF, sizeof(KmerPosition<T, includeAdjacency, IncludeSeqLen>)* readUntil);
        }
    }
    return hashSeqPair;
}

// batches the atomic reservation into the shared k-mer array; a contention/memory trade-off only
static const size_t KMER_STAGING_BUFFER_SIZE = 65536;

static void removeKmerTmpFileIfExists(const std::string &fileName);
static FILE *openKmerTmpFileForOverwriteOrDie(const std::string &fileName, const char *mode);

// Bucket file IO wrappers (raw or --compress-kmer-tmp-files zstd). Non-template so they can be
// declared here and defined after ZstdKmerTmpFileWriter, and used by the templated sink/loader.
static void *bucketWriterOpen(const std::string &fileName, bool compress);
static void bucketWriterAppend(void *writer, bool compress, const void *data, size_t byteSize);
static void bucketWriterClose(void *writer, bool compress);
static size_t bucketReadFile(const std::string &fileName, bool compress, void *dst, size_t maxBytes, bool &overflow);

template <typename T, bool includeAdjacency, bool IncludeSeqLen>
static void flushKmerBuffer(KmerPosition<T, includeAdjacency, IncludeSeqLen> *kmerArray,
                            size_t kmerArraySize,
                            KmerPosition<T, includeAdjacency, IncludeSeqLen> *threadKmerBuffer,
                            size_t bufferPos,
                            size_t *offset) {
    size_t writeOffset = __sync_fetch_and_add(offset, bufferPos);
    if (writeOffset > kmerArraySize || bufferPos > kmerArraySize - writeOffset) {
        Debug(Debug::ERROR) << "Kmer array overflow. currKmerArrayOffset=" << writeOffset
                            << ", kmerBufferPos=" << bufferPos
                            << ", kmerArraySize=" << kmerArraySize << ".\n";
        EXIT(EXIT_FAILURE);
    }
    if (kmerArray != NULL) {
        memcpy(kmerArray + writeOffset, threadKmerBuffer,
               sizeof(KmerPosition<T, includeAdjacency, IncludeSeqLen>) * bufferPos);
    }
}

// ---------------- write-to-disk bucket partitioning ----------------
// One extraction pass routes every emitted k-mer to a per-thread, per-bucket file (bucket =
// the hash range of the split it belongs to), so each split later reads only its bucket
// instead of re-scanning the whole sequence DB. The result is identical to the re-scan path:
// bucket b holds exactly the k-mers fillKmerPositionArray(range_b) would emit, and the
// per-split sort removes any order dependence (the same invariant that makes the split count
// not affect the result). Per-thread files keep the write lock-free, like writeKmersToDisk.
static std::string kmerBucketCountFileName(const std::string &base, size_t bucket) {
    return base + "_" + SSTR(bucket) + ".cnt";
}

static std::string kmerBucketFileName(const std::string &base, size_t bucket, int tid, bool compressed) {
    std::string name = base + "_" + SSTR(bucket) + "_" + SSTR(tid);
    if (compressed) {
        name += ".zst";
    }
    return name;
}

// hash (unsigned short) -> bucket index, from the contiguous split hash ranges.
static unsigned int *buildHashToBucketLookup(const std::vector<std::pair<size_t, size_t>> &ranges) {
    unsigned int *lut = new(std::nothrow) unsigned int[USHRT_MAX + 1];
    Util::checkAllocation(lut, "Can not allocate hash-to-bucket lookup");
    for (size_t b = 0; b < ranges.size(); b++) {
        size_t hi = std::min(ranges[b].second, static_cast<size_t>(USHRT_MAX));
        for (size_t h = ranges[b].first; h <= hi; h++) {
            lut[h] = static_cast<unsigned int>(b);
        }
    }
    return lut;
}

template <typename T, bool includeAdjacency, bool IncludeSeqLen>
struct KmerPartitionSink {
    typedef KmerPosition<T, includeAdjacency, IncludeSeqLen> KP;
    static const size_t BUCKET_BUFFER = 512;
    int numThreads;
    size_t numBuckets;
    bool compress;
    const unsigned int *hashToBucket;   // [USHRT_MAX+1]
    std::string base;
    std::vector<void *> writers;        // indexed [tid * numBuckets + bucket]: FILE* or zstd writer
    std::vector<KP *> buffers;
    std::vector<size_t> bufPos;
    std::vector<size_t> recordsWritten;

    KmerPartitionSink(const std::string &base, int numThreads, size_t numBuckets,
                      const unsigned int *hashToBucket, bool compress)
        : numThreads(numThreads), numBuckets(numBuckets), compress(compress), hashToBucket(hashToBucket), base(base) {
        size_t slots = static_cast<size_t>(numThreads) * numBuckets;
        writers.assign(slots, NULL);
        buffers.assign(slots, NULL);
        bufPos.assign(slots, 0);
        recordsWritten.assign(slots, 0);
        for (int tid = 0; tid < numThreads; tid++) {
            for (size_t b = 0; b < numBuckets; b++) {
                size_t k = static_cast<size_t>(tid) * numBuckets + b;
                writers[k] = bucketWriterOpen(kmerBucketFileName(base, b, tid, compress), compress);
                buffers[k] = new(std::nothrow) KP[BUCKET_BUFFER];
                Util::checkAllocation(buffers[k], "Can not allocate k-mer bucket buffer");
            }
        }
    }

    inline void flushSlot(size_t k) {
        if (bufPos[k] == 0) {
            return;
        }
        bucketWriterAppend(writers[k], compress, buffers[k], sizeof(KP) * bufPos[k]);
        recordsWritten[k] += bufPos[k];
        bufPos[k] = 0;
    }

    // Called concurrently by the fill threads, but each thread touches only its own [tid]
    // slots, so no locking is needed.
    inline void emit(int tid, const KP &rec, unsigned int hash) {
        size_t k = static_cast<size_t>(tid) * numBuckets + hashToBucket[hash];
        buffers[k][bufPos[k]++] = rec;
        if (bufPos[k] >= BUCKET_BUFFER) {
            flushSlot(k);
        }
    }

    // slots share no state, so closing them in parallel keeps this off the threads x buckets path
    void finish() {
        size_t slots = static_cast<size_t>(numThreads) * numBuckets;
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < slots; k++) {
            flushSlot(k);
            bucketWriterClose(writers[k], compress);
            delete[] buffers[k];
        }
        // per-thread record counts let loadKmerBucket place each file without decompressing first
        for (size_t b = 0; b < numBuckets; b++) {
            std::string countFile = kmerBucketCountFileName(base, b);
            FILE *cf = fopen(countFile.c_str(), "w");
            if (cf == NULL) {
                Debug(Debug::ERROR) << "Can not open " << countFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            for (int tid = 0; tid < numThreads; tid++) {
                fprintf(cf, "%zu\n", recordsWritten[static_cast<size_t>(tid) * numBuckets + b]);
            }
            if (fclose(cf) != 0) {
                Debug(Debug::ERROR) << "Can not write " << countFile << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }
};

// Read all per-thread files of one bucket into arr[0..count) and delete them; returns count.
// arr must have been sentinel-initialised by initKmerPositionMemory; the tail stays sentinel.
template <typename T, bool includeAdjacency, bool IncludeSeqLen>
size_t loadKmerBucket(const std::string &base, size_t bucket, int numThreads,
                      KmerPosition<T, includeAdjacency, IncludeSeqLen> *arr, size_t cap, bool compress) {
    typedef KmerPosition<T, includeAdjacency, IncludeSeqLen> KP;
    // the partition pass recorded per-thread counts, so every file has a fixed slot and reads run in parallel
    std::vector<size_t> counts;
    std::string countFile = kmerBucketCountFileName(base, bucket);
    FILE *cf = fopen(countFile.c_str(), "r");
    if (cf != NULL) {
        size_t v;
        while (fscanf(cf, "%zu", &v) == 1) {
            counts.push_back(v);
        }
        fclose(cf);
    }
    if (counts.size() == static_cast<size_t>(numThreads)) {
        std::vector<size_t> offset(static_cast<size_t>(numThreads) + 1, 0);
        for (int tid = 0; tid < numThreads; tid++) {
            offset[tid + 1] = offset[tid] + counts[tid];
        }
        if (offset[numThreads] > cap) {
            Debug(Debug::ERROR) << "k-mer bucket " << bucket << " exceeds the allocated split array\n";
            EXIT(EXIT_FAILURE);
        }
        bool failed = false;
#pragma omp parallel for schedule(dynamic, 1)
        for (int tid = 0; tid < numThreads; tid++) {
            std::string fileName = kmerBucketFileName(base, bucket, tid, compress);
            if (counts[tid] == 0 || FileUtil::fileExists(fileName.c_str()) == false) {
                removeKmerTmpFileIfExists(fileName);
                continue;
            }
            bool overflow = false;
            size_t want = counts[tid] * sizeof(KP);
            size_t bytes = bucketReadFile(fileName, compress, arr + offset[tid], want, overflow);
            if (overflow || bytes != want) {
                failed = true;
            }
            removeKmerTmpFileIfExists(fileName);
        }
        if (failed) {
            Debug(Debug::ERROR) << "k-mer bucket " << bucket << " does not match its recorded record count\n";
            EXIT(EXIT_FAILURE);
        }
        removeKmerTmpFileIfExists(countFile);
        return offset[numThreads];
    }
    size_t count = 0;
    for (int tid = 0; tid < numThreads; tid++) {
        std::string fileName = kmerBucketFileName(base, bucket, tid, compress);
        if (FileUtil::fileExists(fileName.c_str()) == false) {
            continue;
        }
        bool overflow = false;
        size_t bytes = bucketReadFile(fileName, compress, arr + count, (cap - count) * sizeof(KP), overflow);
        // The bucket is written as whole KmerPosition records; anything else means the file, or
        // the cap (totalKmersPerSplit), is inconsistent.
        if (overflow || (bytes % sizeof(KP)) != 0) {
            Debug(Debug::ERROR) << "k-mer bucket " << bucket << " exceeds the allocated split array\n";
            EXIT(EXIT_FAILURE);
        }
        count += bytes / sizeof(KP);
        removeKmerTmpFileIfExists(fileName);
    }
    return count;
}

template <int TYPE, typename T, bool includeAdjacency, bool IncludeSeqLen>
std::pair<size_t, size_t> fillKmerPositionArray(KmerPosition<T, includeAdjacency, IncludeSeqLen> * kmerArray, size_t kmerArraySize, DBReader<DBKeyType> &seqDbr,
                                                Parameters & par, BaseMatrix * subMat, bool hashWholeSequence,
                                                size_t hashStartRange, size_t hashEndRange, size_t * hashDistribution,
                                                KmerPartitionSink<T, includeAdjacency, IncludeSeqLen> *partitionSink){
    size_t offset = 0;
    int querySeqType  =  seqDbr.getDbtype();
    size_t longestKmer = par.kmerSize;
    const unsigned char xIndex = subMat->aa2num[static_cast<int>('X')];


    ScoreMatrix two;
    ScoreMatrix three;
    if (TYPE == Parameters::DBTYPE_HMM_PROFILE) {
        two = ExtendedSubstitutionMatrix::calcScoreMatrix(*subMat, 2);
        three = ExtendedSubstitutionMatrix::calcScoreMatrix(*subMat, 3);
    }

    Debug::Progress progress(seqDbr.getSize());
#pragma omp parallel num_threads(par.threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        unsigned short * scoreDist= new(std::nothrow) unsigned short[65536];
        Util::checkAllocation(scoreDist, "Can not allocate scoreDist memory in fillKmerPositionArray");
        unsigned int * hierarchicalScoreDist= new(std::nothrow) unsigned int[128];
        Util::checkAllocation(hierarchicalScoreDist, "Can not allocate hierarchicalScoreDist memory in fillKmerPositionArray");
        // Zero once; kept clean by clearing only touched bins after each sequence (below).
        memset(scoreDist, 0, sizeof(unsigned short) * 65536);
        memset(hierarchicalScoreDist, 0, sizeof(unsigned int) * 128);

        Masker *masker = NULL;
        if (par.maskMode == 1) {
            masker = new Masker(*subMat);
        }
        const int adjustedKmerSize = (par.adjustKmerLength) ? std::min( par.kmerSize+5, 23) :   par.kmerSize;
        Sequence seq(par.maxSeqLen, querySeqType, subMat, adjustedKmerSize, par.spacedKmer, false, true, par.spacedKmerPattern);
        KmerGenerator* generator;
        if (TYPE == Parameters::DBTYPE_HMM_PROFILE) {
            generator = new KmerGenerator( par.kmerSize, subMat->alphabetSize, 150);
            generator->setDivideStrategy(&three, &two);
        }
        Indexer idxer(subMat->alphabetSize - 1,  par.kmerSize);
        const unsigned int BUFFER_SIZE = static_cast<unsigned int>(KMER_STAGING_BUFFER_SIZE);
        size_t bufferPos = 0;
        KmerPosition<T, includeAdjacency, IncludeSeqLen> * threadKmerBuffer = NULL;
        if (hashDistribution == NULL) {
            threadKmerBuffer = new(std::nothrow) KmerPosition<T, includeAdjacency, IncludeSeqLen>[BUFFER_SIZE];
            Util::checkAllocation(threadKmerBuffer, "Can not allocate threadKmerBuffer memory in fillKmerPositionArray");
        }
        SequencePosition * kmers = (SequencePosition *) malloc((par.pickNbest * (par.maxSeqLen + 1) + 1) * sizeof(SequencePosition) * 2);
        Util::checkAllocation(kmers, "Can not allocate kmers memory in fillKmerPositionArray");
        size_t kmersArraySize = par.maxSeqLen;
        const size_t flushSize = 100000000;
        size_t iterations = static_cast<size_t>(ceil(static_cast<double>(seqDbr.getSize()) / static_cast<double>(flushSize)));
        for (size_t i = 0; i < iterations; i++) {
            size_t start = (i * flushSize);
            size_t bucketSize = std::min(seqDbr.getSize() - (i * flushSize), flushSize);

            size_t scanChunk = kmerScanChunkSize(seqDbr, start, bucketSize, par.threads);
// every thread in the team computes the same chunk from shared read-only state, as the schedule needs
#pragma omp for schedule(dynamic, scanChunk)
            for (size_t id = start; id < (start + bucketSize); id++) {
                progress.updateProgress();

                seq.mapSequence(id, seqDbr.getDbKey(id), seqDbr.getData(id, thread_idx), seqDbr.getSeqLen(id));

                size_t seqHash =  SIZE_T_MAX;
                //TODO, how to handle this in reverse?
                if(hashWholeSequence){
                    seqHash = Util::hash(seq.numSequence, seq.L);
                    seqHash = hashUInt64(seqHash, par.hashShift);
                }
                if(masker != NULL){
                    masker->maskSequence(seq, par.maskMode,  par.maskProb, par.maskLowerCaseMode, par.maskNrepeats);
                }
                size_t seqKmerCount = 0;
                DBKeyType seqId = seq.getDbKey();
                while (seq.hasNextKmer()) {
                    unsigned char *kmer = (unsigned char*) seq.nextKmer();
                    if(seq.kmerContainsX()){
                        continue;
                    }
                    if(TYPE == Parameters::DBTYPE_NUCLEOTIDES){
                        NucleotideMatrix * nuclMatrix = (NucleotideMatrix*)subMat;
                        size_t kmerLen =  par.kmerSize;
                        size_t kmerIdx = Indexer::computeKmerIdx(kmer, kmerLen);
                        size_t revkmerIdx = Util::revComplement(kmerIdx, kmerLen);
                        // skip forward and rev. identical k-mers.
                        // We can not know how to align these afterwards
                        if(revkmerIdx == kmerIdx){
                            continue;
                        }
                        bool pickReverseKmer = (revkmerIdx<kmerIdx);
                        kmerIdx = (pickReverseKmer) ? revkmerIdx : kmerIdx;
                        const unsigned short hash = hashUInt64(kmerIdx, par.hashShift);

                        if(par.adjustKmerLength) {
                            unsigned char revKmer[32];
                            unsigned char * kmerToHash = kmer;
                            if(pickReverseKmer){
                                for(int pos = static_cast<int>(adjustedKmerSize)-1; pos > -1; pos--){
                                    revKmer[(adjustedKmerSize - 1) - pos]=nuclMatrix->reverseResidue(kmer[pos]);
                                }
                                kmerToHash = revKmer;
                            }
                            kmerLen = MarkovKmerScore::adjustedLength(kmerToHash, adjustedKmerSize,
                                                                      (par.kmerSize - MarkovScores::MARKOV_ORDER) * MarkovScores::MEDIAN_SCORE);
                            longestKmer = std::max(kmerLen, longestKmer);
                            kmerIdx = Indexer::computeKmerIdx(kmerToHash, kmerLen);
                        }

                        // set signed bit for normal kmers to make the  SIZE_T_MAX logic easier
                        // reversed kmers do not have a signed bit
                        size_t kmerRev = (pickReverseKmer) ? BIT_CLEAR(kmerIdx, 63) : BIT_SET(kmerIdx, 63);
                        (kmers + seqKmerCount)->kmer = kmerRev;
                        int pos = seq.getCurrentPosition();
                        (kmers + seqKmerCount)->pos = (pickReverseKmer) ? (seq.L) - pos - kmerLen : pos;
                        (kmers + seqKmerCount)->score = hash;
                        scoreDist[hash]++;
                        hierarchicalScoreDist[hash >> 9]++;
                        seqKmerCount++;
                    } else if(TYPE == Parameters::DBTYPE_HMM_PROFILE) {
                        std::pair<size_t*, size_t>  scoreMat = generator->generateKmerList(kmer, true);
                        for(size_t kmerPos = 0; kmerPos < scoreMat.second && kmerPos < static_cast<size_t >(par.pickNbest); kmerPos++){
                            size_t kmerIdx = scoreMat.first[kmerPos];
                            (kmers + seqKmerCount)->kmer = kmerIdx;
                            (kmers + seqKmerCount)->pos = seq.getCurrentPosition();
                            const unsigned short hash = hashUInt64(kmerIdx, par.hashShift);
                            (kmers + seqKmerCount)->score = hash;
                            scoreDist[hash]++;
                            hierarchicalScoreDist[hash >> 9]++;
                            seqKmerCount++;
                        }
                    } else {
                        size_t kmerIdx = idxer.int2index(kmer, 0, par.kmerSize);
                        (kmers + seqKmerCount)->kmer = kmerIdx;
                        (kmers + seqKmerCount)->pos = seq.getCurrentPosition();
                        const unsigned short hash = hashUInt64(kmerIdx, par.hashShift);
                        (kmers + seqKmerCount)->score = hash;
                        scoreDist[hash]++;
                        hierarchicalScoreDist[hash >> 9]++;
                        seqKmerCount++;
                    }
                    if(seqKmerCount >= kmersArraySize){
                        kmersArraySize = seq.getMaxLen();
                        SequencePosition *newKmers = (SequencePosition *) realloc(kmers, (par.pickNbest * (kmersArraySize + 1) + 1) * sizeof(SequencePosition) * 2);
                        Util::checkAllocation(newKmers, "Can not reallocate kmers memory in fillKmerPositionArray");
                        kmers = newKmers;
                    }

                }
                float kmersPerSequenceScale = (TYPE == Parameters::DBTYPE_NUCLEOTIDES) ? par.kmersPerSequenceScale.values.nucleotide()
                                                                                       : par.kmersPerSequenceScale.values.aminoacid();
                size_t kmerConsidered = std::min(static_cast<size_t >(par.kmersPerSequence  - 1 + (kmersPerSequenceScale * seq.L)), seqKmerCount);

                unsigned int threshold = 0;
                size_t kmerInBins = 0;
                if (seqKmerCount > 0) {
                    size_t hierarchicaThreshold = 0;
                    for(hierarchicaThreshold = 0; hierarchicaThreshold < 128 && kmerInBins < kmerConsidered; hierarchicaThreshold++){
                        kmerInBins += hierarchicalScoreDist[hierarchicaThreshold];
                    }
                    hierarchicaThreshold -= (hierarchicaThreshold > 0) ? 1: 0;
                    kmerInBins -= hierarchicalScoreDist[hierarchicaThreshold];
                    for(threshold = hierarchicaThreshold*512; threshold <= USHRT_MAX && kmerInBins < kmerConsidered; threshold++){
                        kmerInBins += scoreDist[threshold];
                    }
                }
                int tooMuchElemInLastBin = (kmerInBins - kmerConsidered);

                // add k-mer to represent the identity
                if (static_cast<unsigned short>(seqHash) >= hashStartRange && static_cast<unsigned short>(seqHash) <= hashEndRange) {
                    if(hashDistribution != NULL){
                        __sync_fetch_and_add(&hashDistribution[static_cast<unsigned short>(seqHash)], 1);
                    }
                    else{
                        threadKmerBuffer[bufferPos].kmer = seqHash;
                        threadKmerBuffer[bufferPos].id = seqId;
                        threadKmerBuffer[bufferPos].pos = 0;
                        threadKmerBuffer[bufferPos].sl.setSeqLen(static_cast<T>(seq.L));
                        if (includeAdjacency) {
                            for (size_t i = 0; i < 6; i++) {
                                threadKmerBuffer[bufferPos].setAdjacentSeq(i, xIndex);
                            }
                        }
                        if (partitionSink != NULL) {
                            partitionSink->emit(thread_idx, threadKmerBuffer[bufferPos], static_cast<unsigned short>(seqHash));
                        } else {
                            bufferPos++;
                            if (bufferPos >= BUFFER_SIZE) {
                                flushKmerBuffer(kmerArray, kmerArraySize, threadKmerBuffer, bufferPos, &offset);
                                bufferPos = 0;
                            }
                        }
                    }
                }

                if(par.ignoreMultiKmer){
                    if(TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                        SORT_SERIAL(kmers, kmers + seqKmerCount, SequencePosition::compareByScoreReverse);
                    }else{
                        SORT_SERIAL(kmers, kmers + seqKmerCount, SequencePosition::compareByScore);
                    }
                }
                size_t selectedKmer = 0;
                for (size_t kmerIdx = 0; kmerIdx < seqKmerCount && selectedKmer < kmerConsidered; kmerIdx++) {

                    /* skip repeated kmer */
                    if (par.ignoreMultiKmer) {
                        size_t kmer = (kmers + kmerIdx)->kmer;
                        if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                            kmer = BIT_SET(kmer, 63);
                        }
                        if (kmerIdx + 1 < seqKmerCount) {
                            size_t nextKmer = (kmers + kmerIdx + 1)->kmer;
                            if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                                nextKmer = BIT_SET(nextKmer, 63);
                            }
                            if (kmer == nextKmer) {
                                while (kmer == nextKmer && kmerIdx < seqKmerCount) {
                                    kmerIdx++;
                                    if(kmerIdx >= seqKmerCount)
                                        break;
                                    nextKmer = (kmers + kmerIdx)->kmer;
                                    if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                                        nextKmer = BIT_SET(nextKmer, 63);
                                    }
                                }
                            }
                        }
                        if(kmerIdx >= seqKmerCount)
                            break;
                    }

                    if ((kmers + kmerIdx)->score < threshold ){
                        if((kmers + kmerIdx)->score == (threshold - 1) && tooMuchElemInLastBin){
                            tooMuchElemInLastBin--;
                            threshold -= (tooMuchElemInLastBin == 0) ? 1 : 0;
                        }

                        selectedKmer++;
                        if ((kmers + kmerIdx)->score >= hashStartRange && (kmers + kmerIdx)->score <= hashEndRange)
                        {
                            if(hashDistribution != NULL){
                                __sync_fetch_and_add(&hashDistribution[(kmers + kmerIdx)->score], 1);
                                continue;
                            }
                            threadKmerBuffer[bufferPos].kmer = (kmers + kmerIdx)->kmer;
                            threadKmerBuffer[bufferPos].id = seqId;
                            threadKmerBuffer[bufferPos].pos = (kmers + kmerIdx)->pos;
                            threadKmerBuffer[bufferPos].sl.setSeqLen(static_cast<T>(seq.L));
                            if (includeAdjacency) {
                                unsigned int startPos = (kmers + kmerIdx)->pos;
                                unsigned int endPos = (kmers + kmerIdx)->pos + seq.getEffectiveKmerSize() - 1;
                                for (size_t i = 0; i < 6; i++) {
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(i, xIndex);
                                }

                                if (startPos >= 3) {
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(0, seq.numSequence[startPos - 3]);
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(1, seq.numSequence[startPos - 2]);
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(2, seq.numSequence[startPos - 1]);
                                } else if (startPos == 2) {
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(1, seq.numSequence[startPos - 2]);
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(2, seq.numSequence[startPos - 1]);
                                } else if (startPos == 1) {
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(2, seq.numSequence[startPos - 1]);
                                }

                                if (endPos + 3 <= static_cast<unsigned int>(seq.L) - 1) {
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(3, seq.numSequence[endPos + 1]);
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(4, seq.numSequence[endPos + 2]);
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(5, seq.numSequence[endPos + 3]);
                                } else if (endPos + 2 == static_cast<unsigned int>(seq.L) - 1) {
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(3, seq.numSequence[endPos + 1]);
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(4, seq.numSequence[endPos + 2]);
                                } else if (endPos + 1 == static_cast<unsigned int>(seq.L) - 1) {
                                    threadKmerBuffer[bufferPos].setAdjacentSeq(3, seq.numSequence[endPos + 1]);
                                }
                            }
                            if (partitionSink != NULL) {
                                partitionSink->emit(thread_idx, threadKmerBuffer[bufferPos], (kmers + kmerIdx)->score);
                            } else {
                                bufferPos++;
                                if (bufferPos >= BUFFER_SIZE) {
                                    flushKmerBuffer(kmerArray, kmerArraySize, threadKmerBuffer, bufferPos, &offset);
                                    bufferPos = 0;
                                }
                            }
                        }
                    }
                }
                // Restore scoreDist/hierarchicalScoreDist to all-zero for the next sequence by
                // clearing only the bins this sequence touched (each recorded in kmers[].score),
                // instead of a 128 KB memset per sequence.
                for (size_t k = 0; k < seqKmerCount; k++) {
                    scoreDist[(kmers + k)->score] = 0;
                    hierarchicalScoreDist[(kmers + k)->score >> 9] = 0;
                }
            }
#pragma omp barrier
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            if (thread_idx == 0) {
                seqDbr.remapData();
            }
#pragma omp barrier
        }
        if (masker != NULL) {
            delete masker;
        }

        if(threadKmerBuffer != NULL && bufferPos > 0){
            flushKmerBuffer(kmerArray, kmerArraySize, threadKmerBuffer, bufferPos, &offset);
        }
        free(kmers);
        delete[] threadKmerBuffer;
        delete[] hierarchicalScoreDist;
        delete[] scoreDist;
        if (TYPE == Parameters::DBTYPE_HMM_PROFILE) {
            delete generator;
        }
    }

    if (TYPE == Parameters::DBTYPE_HMM_PROFILE) {
        ExtendedSubstitutionMatrix::freeScoreMatrix(three);
        ExtendedSubstitutionMatrix::freeScoreMatrix(two);
    }

    return std::make_pair(offset, longestKmer);
}


template <int TYPE, typename T, bool includeAdjacency, bool IncludeSeqLen>
void swapCenterSequence(KmerPosition<T, includeAdjacency, IncludeSeqLen> *hashSeqPair, size_t splitKmerCount, SequenceWeights &seqWeights) {

    size_t prevHash = hashSeqPair[0].kmer;
    if(TYPE == Parameters::DBTYPE_NUCLEOTIDES){
        prevHash = BIT_SET(prevHash, 63);
    }

    size_t repSeqPos = 0;
    size_t prevHashStart = 0;
    float repSeqWeight = seqWeights.getWeightById(hashSeqPair[repSeqPos].id);
    for (size_t elementIdx = 0; elementIdx < splitKmerCount; elementIdx++) {

        size_t currKmer = hashSeqPair[elementIdx].kmer;
        if(TYPE == Parameters::DBTYPE_NUCLEOTIDES){
            currKmer = BIT_SET(currKmer, 63);
        }
        if (prevHash != currKmer) {

            if (repSeqPos != prevHashStart)
                std::swap(hashSeqPair[repSeqPos],hashSeqPair[prevHashStart]);

            prevHashStart = elementIdx;
            prevHash = hashSeqPair[elementIdx].kmer;
            if(TYPE == Parameters::DBTYPE_NUCLEOTIDES){
                prevHash = BIT_SET(prevHash, 63);
            }
            repSeqPos = elementIdx;
            repSeqWeight = seqWeights.getWeightById(hashSeqPair[repSeqPos].id);
        }
        else {
            float currWeight = seqWeights.getWeightById(hashSeqPair[elementIdx].id);
            if (currWeight > repSeqWeight) {
                repSeqWeight = currWeight;
                repSeqPos = elementIdx;
            }
        }

        if (hashSeqPair[elementIdx].kmer == SIZE_T_MAX) {
            break;
        }

    }
}

template <int TYPE, typename T, bool includeAdjacency, bool IncludeSeqLen>
size_t assignGroup(KmerPosition<T, includeAdjacency, IncludeSeqLen> *hashSeqPair, KmerPosition<T, false, IncludeSeqLen> *writeSeqPair,
                    bool includeOnlyExtendable, int covMode, float covThr,
                    SequenceWeights *sequenceWeights, float weightThr, int threads,
                    std::vector<size_t> &threadOffsets, BaseMatrix *subMat,
                    AssignGroupMask assignGroupMask, ComputationPhase phase, short *countTable) {

    // Current assign group mode based on assignGroupMask
    const bool useAdjacentSeq = (includeAdjacency && hasFeature(assignGroupMask, AssignGroupFeature::AdjacentSeq));
    const bool useCountTable = hasFeature(assignGroupMask, AssignGroupFeature::CountTable);
    const bool isSetupCountTable = (phase == ComputationPhase::SetupCountTable);

    if (isSetupCountTable) {
        Debug(Debug::INFO) << "Assign group Mode: SetupCountTable: ";
    } else if (useAdjacentSeq) {
        Debug(Debug::INFO) << "Assign group Mode: Adjacent sequence: ";
    } else if (useCountTable) {
        Debug(Debug::INFO) << "Assign group Mode: CountTable: ";
    } else {
        Debug(Debug::INFO) << "Assign group Mode: Longest center: ";
    }

    std::vector<size_t> localWritePos;
    localWritePos.resize(threads);
    for (int thread = 0; thread < threads; thread++) {
        localWritePos[thread] = threadOffsets[thread];
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int thread = 0; thread < threads; thread++) {
        size_t startIdx = threadOffsets[thread];
        size_t endIdx = threadOffsets[thread + 1];
        if (startIdx >= endIdx) {
            continue;
        }

        size_t prevHash = hashSeqPair[startIdx].kmer;
        size_t repSeqKey = hashSeqPair[startIdx].id;
        size_t repSeqId = repSeqKey;
        bool repIsReverse = false;

        if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
            repIsReverse = (BIT_CHECK(hashSeqPair[startIdx].kmer, 63) == false);
            repSeqId = (repIsReverse) ? BIT_CLEAR(repSeqId, 63) : BIT_SET(repSeqId, 63);
            prevHash = BIT_SET(prevHash, 63);
        }

        size_t prevHashStart = startIdx;
        size_t prevSetSize = 0;
        size_t skipByWeightCount = 0;
        T queryLen = hashSeqPair[startIdx].sl.getSeqLen(hashSeqPair[startIdx].id);
        T repSeq_i_pos = hashSeqPair[startIdx].pos;

        short *subMatPos[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

        // prepare subMatPos for adj mode
        if (useAdjacentSeq && hashSeqPair[prevHashStart].getAdjacentSeq(0) != UCHAR_MAX) {
            for (size_t i = 0; i < 6; i++) {
                subMatPos[i] = subMat->subMatrix[hashSeqPair[prevHashStart].getAdjacentSeq(i)];
            }
        }

        for (size_t elementIdx = startIdx; elementIdx <= endIdx; elementIdx++) {
            size_t currKmer = hashSeqPair[elementIdx].kmer;
            if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                currKmer = BIT_SET(currKmer, 63);
            }

            if (prevHash != currKmer) {

                // Phase 1: find and swap in the best representative for this mode
                if (useAdjacentSeq && subMatPos[0] != NULL) {
                    // find member with lowest adj score → swap to prevHashStart
                    size_t bestPos = prevHashStart;
                    int minAdjScore = INT_MAX;
                    for (size_t i = prevHashStart; i < elementIdx; i++) {
                        if (i > prevHashStart && sequenceWeights != nullptr &&
                            sequenceWeights->getWeightById(hashSeqPair[i].id) > weightThr) {
                            continue;
                        }
                        size_t kmer = hashSeqPair[i].kmer;
                        if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                            kmer = BIT_SET(kmer, 63);
                        }
                        if (kmer == SIZE_T_MAX) continue;

                        if (hashSeqPair[i].id == repSeqKey) {
                            hashSeqPair[i].setAdjacentSeq(0, UCHAR_MAX);
                        }
                        if (hashSeqPair[i].getAdjacentSeq(0) != UCHAR_MAX) {
                            int currAdjScore = 0;
                            for (size_t j = 0; j < 6; j++) {
                                currAdjScore += subMatPos[j][hashSeqPair[i].getAdjacentSeq(j)];
                            }
                            if (currAdjScore <= minAdjScore) {
                                minAdjScore = currAdjScore;
                                bestPos = i;
                            }
                        }
                    }
                    if (bestPos != prevHashStart &&
                        hashSeqPair[bestPos].kmer != SIZE_T_MAX &&
                        hashSeqPair[bestPos].getAdjacentSeq(0) != UCHAR_MAX) {
                        std::swap(hashSeqPair[bestPos], hashSeqPair[prevHashStart]);
                    }
                } else if (useCountTable && countTable != NULL) {
                    // find member with highest count → swap to prevHashStart
                    size_t bestPos = prevHashStart;
                    int maxCount = -1;
                    for (size_t i = prevHashStart + 1; i < elementIdx; i++) {
                        if (sequenceWeights != nullptr &&
                            sequenceWeights->getWeightById(hashSeqPair[i].id) > weightThr) {
                            continue;
                        }
                        size_t kmer = hashSeqPair[i].kmer;
                        if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                            kmer = BIT_SET(kmer, 63);
                        }
                        if (kmer == SIZE_T_MAX) continue;

                        const size_t memid = hashSeqPair[i].id;
                        if (memid != repSeqKey) {
                            int cnt = countTable[memid];
                            if (cnt >= maxCount) {
                                if (includeAdjacency == false || hashSeqPair[i].getAdjacentSeq(0) != UCHAR_MAX) {
                                    maxCount = cnt;
                                    bestPos = i;
                                }
                            }
                        }
                    }
                    if (bestPos != prevHashStart &&
                        hashSeqPair[bestPos].kmer != SIZE_T_MAX &&
                        (includeAdjacency == false || hashSeqPair[bestPos].getAdjacentSeq(0) != UCHAR_MAX)) {
                        std::swap(hashSeqPair[bestPos], hashSeqPair[prevHashStart]);
                    }
                }

                // After swap, update rep info
                repSeqKey = hashSeqPair[prevHashStart].id;
                repSeqId = repSeqKey;
                repIsReverse = false;
                if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                    repIsReverse = (BIT_CHECK(hashSeqPair[prevHashStart].kmer, 63) == false);
                    repSeqId = (repIsReverse) ? BIT_CLEAR(repSeqId, 63) : BIT_SET(repSeqId, 63);
                }
                queryLen = hashSeqPair[prevHashStart].sl.getSeqLen(hashSeqPair[prevHashStart].id);
                repSeq_i_pos = hashSeqPair[prevHashStart].pos;

                // Phase 2: assign group members to representative
                bool skipProcessing = false;
                if (useAdjacentSeq) {
                    skipProcessing = (hashSeqPair[prevHashStart].getAdjacentSeq(0) == UCHAR_MAX);
                }

                if (skipProcessing == false) {
                    for (size_t i = prevHashStart; i < elementIdx; i++) {
                        if (i > prevHashStart && sequenceWeights != nullptr &&
                            sequenceWeights->getWeightById(hashSeqPair[i].id) > weightThr) {
                            continue;
                        }

                        size_t kmer = hashSeqPair[i].kmer;
                        if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                            kmer = BIT_SET(hashSeqPair[i].kmer, 63);
                        }

                        size_t rId = (kmer != SIZE_T_MAX) ? ((prevSetSize - skipByWeightCount == 1) ? SIZE_T_MAX : repSeqId) : SIZE_T_MAX;

                        if (rId != SIZE_T_MAX) {
                            int diagonal = repSeq_i_pos - hashSeqPair[i].pos;
                            if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                                bool targetIsReverse = (BIT_CHECK(hashSeqPair[i].kmer, 63) == false);
                                bool queryNeedsToBeRev = false;

                                T queryPos = 0;
                                T targetPos = 0;
                                T targetLen = hashSeqPair[i].sl.getSeqLen(hashSeqPair[i].id);

                                if (repIsReverse == true && targetIsReverse == false) {
                                    queryPos = repSeq_i_pos;
                                    targetPos = hashSeqPair[i].pos;
                                    queryNeedsToBeRev = true;
                                } else if (repIsReverse == true && targetIsReverse == true) {
                                    queryPos = (queryLen - 1) - repSeq_i_pos;
                                    targetPos = (targetLen - 1) - hashSeqPair[i].pos;
                                    queryNeedsToBeRev = false;
                                } else if (repIsReverse == false && targetIsReverse == true) {
                                    queryPos = (queryLen - 1) - repSeq_i_pos;
                                    targetPos = (targetLen - 1) - hashSeqPair[i].pos;
                                    queryNeedsToBeRev = true;
                                } else {
                                    queryPos = repSeq_i_pos;
                                    targetPos = hashSeqPair[i].pos;
                                    queryNeedsToBeRev = false;
                                }

                                diagonal = queryPos - targetPos;
                                rId = (queryNeedsToBeRev) ? BIT_CLEAR(rId, 63) : BIT_SET(rId, 63);
                            }

                            T targetLen = hashSeqPair[i].sl.getSeqLen(hashSeqPair[i].id);
                            bool canBeExtended = (diagonal < 0) || (diagonal > (queryLen - targetLen));
                            bool canBeCovered = Util::canBeCovered(covThr, covMode, static_cast<float>(queryLen), static_cast<float>(targetLen));

                            if ((includeOnlyExtendable == false && canBeCovered) ||
                                (canBeExtended && includeOnlyExtendable == true)) {
                                if (isSetupCountTable) {
                                    if (countTable != NULL) {
                                        __sync_fetch_and_add(&countTable[hashSeqPair[i].id], 1);
                                    }
                                } else {
                                    if (writeSeqPair != NULL) {
                                        if (queryLen < hashSeqPair[i].sl.getSeqLen(hashSeqPair[i].id) && covMode == Parameters::COV_MODE_TARGET) {
                                            writeSeqPair[localWritePos[thread]].kmer = hashSeqPair[i].id;
                                            writeSeqPair[localWritePos[thread]].pos = static_cast<short>(-diagonal);
                                            writeSeqPair[localWritePos[thread]].sl.setSeqLen(targetLen);
                                            writeSeqPair[localWritePos[thread]].id = rId;
                                        } else {
                                            writeSeqPair[localWritePos[thread]].kmer = rId;
                                            writeSeqPair[localWritePos[thread]].pos = static_cast<short>(diagonal);
                                            writeSeqPair[localWritePos[thread]].sl.setSeqLen(targetLen);
                                            writeSeqPair[localWritePos[thread]].id = hashSeqPair[i].id;
                                        }
                                    } else {
                                        if (queryLen < hashSeqPair[i].sl.getSeqLen(hashSeqPair[i].id) && covMode == Parameters::COV_MODE_TARGET) {
                                            hashSeqPair[localWritePos[thread]].kmer = hashSeqPair[i].id;
                                            hashSeqPair[localWritePos[thread]].pos = static_cast<short>(-diagonal);
                                            hashSeqPair[localWritePos[thread]].sl.setSeqLen(targetLen);
                                            hashSeqPair[localWritePos[thread]].id = rId;
                                        } else {
                                            hashSeqPair[localWritePos[thread]].kmer = rId;
                                            hashSeqPair[localWritePos[thread]].pos = static_cast<short>(diagonal);
                                            hashSeqPair[localWritePos[thread]].sl.setSeqLen(targetLen);
                                            hashSeqPair[localWritePos[thread]].id = hashSeqPair[i].id;
                                        }
                                    }
                                    localWritePos[thread]++;
                                }
                            }
                        }
                    }
                }

                if (elementIdx == endIdx || hashSeqPair[elementIdx].kmer == SIZE_T_MAX) {
                    break;
                }

                prevSetSize = 0;
                skipByWeightCount = 0;
                prevHash = currKmer;
                prevHashStart = elementIdx;

                // Reset rep info for next hash group (will be updated after Phase 1 swap)
                repSeqKey = hashSeqPair[elementIdx].id;
                repSeqId = repSeqKey;
                repIsReverse = false;
                if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                    prevHash = BIT_SET(prevHash, 63);
                    repIsReverse = (BIT_CHECK(hashSeqPair[elementIdx].kmer, 63) == 0);
                    repSeqId = (repIsReverse) ? BIT_CLEAR(repSeqId, 63) : BIT_SET(repSeqId, 63);
                }

                queryLen = hashSeqPair[elementIdx].sl.getSeqLen(hashSeqPair[elementIdx].id);
                repSeq_i_pos = hashSeqPair[elementIdx].pos;

                if (useAdjacentSeq && hashSeqPair[prevHashStart].getAdjacentSeq(0) != UCHAR_MAX) {
                    for (size_t i = 0; i < 6; i++) {
                        subMatPos[i] = subMat->subMatrix[hashSeqPair[prevHashStart].getAdjacentSeq(i)];
                    }
                } else {
                    for (size_t i = 0; i < 6; i++) {
                        subMatPos[i] = NULL;
                    }
                }
            }

            if (hashSeqPair[elementIdx].kmer == SIZE_T_MAX) {
                break;
            }

            prevSetSize++;
            if (prevSetSize > 1 && sequenceWeights != nullptr &&
                sequenceWeights->getWeightById(hashSeqPair[elementIdx].id) > weightThr) {
                skipByWeightCount++;
            }
        }
    }

    if (isSetupCountTable) {
        return 0;
    }

    size_t writePos = localWritePos[0];
    if (writeSeqPair != nullptr) {
        for (int thread = 1; thread < threads; thread++) {
            size_t startIdx = threadOffsets[thread];
            size_t endIdx = localWritePos[thread];

            for (size_t cpid = startIdx; cpid < endIdx; cpid++) {
                writeSeqPair[writePos++] = writeSeqPair[cpid];
            }
        }
        writeSeqPair[writePos].kmer = SIZE_T_MAX;
    } else {
        for (int thread = 1; thread < threads; thread++) {
            size_t startIdx = threadOffsets[thread];
            size_t endIdx = localWritePos[thread];
            for (size_t cpid = startIdx; cpid < endIdx; cpid++) {
                hashSeqPair[writePos++] = hashSeqPair[cpid];
            }
        }
        hashSeqPair[writePos].kmer = SIZE_T_MAX;
    }
    return writePos;
}


template <typename T, bool includeAdjacency, bool IncludeSeqLen>
static void runIteration(
    AssignGroupMask mask, int &iteration, size_t &writePos,
    size_t hashEndRange, const std::string &splitFile,
    KmerPosition<T, includeAdjacency, IncludeSeqLen> *hashSeqPair,
    KmerPosition<T, false, IncludeSeqLen> *writeSeqPair,
    DBReader<DBKeyType> &seqDbr, Parameters &par,
    BaseMatrix *subMat, short *countTable,
    SequenceWeights *sequenceWeights,
    std::vector<size_t> &threadOffsets, Timer &timer) {

    timer.reset();
    if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
        writePos = assignGroup<Parameters::DBTYPE_NUCLEOTIDES, T, includeAdjacency, IncludeSeqLen>(
            hashSeqPair, writeSeqPair,
            par.includeOnlyExtendable, par.covMode, par.covThr,
            sequenceWeights, par.weightThr,
            par.threads, threadOffsets, subMat,
            mask, ComputationPhase::Main, countTable);
    } else {
        writePos = assignGroup<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, IncludeSeqLen>(
            hashSeqPair, writeSeqPair,
            par.includeOnlyExtendable, par.covMode, par.covThr,
            sequenceWeights, par.weightThr,
            par.threads, threadOffsets, subMat,
            mask, ComputationPhase::Main, countTable);
    }
    Debug(Debug::INFO) << "Time for assign: " << timer.lap() << "\n";

    Debug(Debug::INFO) << "Sort by rep. sequence ";
    timer.reset();
    if (par.needWriteBuffer) {
        if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
            SORT_PARALLEL(
                writeSeqPair, writeSeqPair + writePos,
                KmerPosition<T, false, IncludeSeqLen>::compareRepSequenceAndIdAndDiagReverse);
        } else {
            SORT_PARALLEL(
                writeSeqPair, writeSeqPair + writePos,
                KmerPosition<T, false, IncludeSeqLen>::compareRepSequenceAndIdAndDiag);
        }
    } else {
        if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
            SORT_PARALLEL(
                hashSeqPair, hashSeqPair + writePos,
                KmerPosition<T, includeAdjacency, IncludeSeqLen>::compareRepSequenceAndIdAndDiagReverse);
        } else {
            SORT_PARALLEL(
                hashSeqPair, hashSeqPair + writePos,
                KmerPosition<T, includeAdjacency, IncludeSeqLen>::compareRepSequenceAndIdAndDiag);
        }
    }
    Debug(Debug::INFO) << timer.lap() << "\n";

    if (hashEndRange != SIZE_T_MAX || par.needWriteBuffer) {
        std::vector<size_t> threadQueryOffsets(par.threads + 1);
        size_t qSplitSize = seqDbr.getSize() / par.threads;
        threadQueryOffsets[0] = 0;

        if (par.needWriteBuffer) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (int thread = 1; thread < par.threads; thread++) {
                size_t startqid = qSplitSize * thread;
                KmerPosition<T, false, IncludeSeqLen> *it = std::lower_bound(
                    writeSeqPair, writeSeqPair + writePos, startqid,
                    [](const KmerPosition<T, false, IncludeSeqLen> &elem, size_t k) {
                        return elem.kmer < k;
                    });
                threadQueryOffsets[thread] = it - writeSeqPair;
            }
        } else {
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (int thread = 1; thread < par.threads; thread++) {
                size_t startqid = qSplitSize * thread;
                KmerPosition<T, includeAdjacency, IncludeSeqLen> *it = std::lower_bound(
                    hashSeqPair, hashSeqPair + writePos, startqid,
                    [](const KmerPosition<T, includeAdjacency, IncludeSeqLen> &elem, size_t k) {
                        return elem.kmer < k;
                    });
                threadQueryOffsets[thread] = it - hashSeqPair;
            }
        }
        threadQueryOffsets[par.threads] = writePos;

        timer.reset();
        if (par.needWriteBuffer) {
            if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
                writeKmersToDisk<Parameters::DBTYPE_NUCLEOTIDES, KmerEntryRev, T, false, IncludeSeqLen>(
                    splitFile, writeSeqPair, writePos + 1,
                    par.threads, &threadQueryOffsets, iteration);
            } else {
                writeKmersToDisk<Parameters::DBTYPE_AMINO_ACIDS, KmerEntry, T, false, IncludeSeqLen>(
                    splitFile, writeSeqPair, writePos + 1,
                    par.threads, &threadQueryOffsets, iteration);
            }
        } else {
            if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
                writeKmersToDisk<Parameters::DBTYPE_NUCLEOTIDES, KmerEntryRev, T, includeAdjacency, IncludeSeqLen>(
                    splitFile, hashSeqPair, writePos + 1,
                    par.threads, &threadQueryOffsets, iteration);
            } else {
                writeKmersToDisk<Parameters::DBTYPE_AMINO_ACIDS, KmerEntry, T, includeAdjacency, IncludeSeqLen>(
                    splitFile, hashSeqPair, writePos + 1,
                    par.threads, &threadQueryOffsets, iteration);
            }
        }
        Debug(Debug::INFO) << "Time for write: " << timer.lap() << "\n";
    }

    iteration++;
}


template <typename T, bool includeAdjacency, bool IncludeSeqLen>
KmerPosition<T, includeAdjacency, IncludeSeqLen> *doComputation(
    size_t totalKmers, size_t hashStartRange, size_t hashEndRange,
    const std::string &splitFile, AssignGroupMask assignGroupMask,
    ComputationPhase phase, DBReader<DBKeyType> &seqDbr,
    Parameters &par, BaseMatrix *subMat, short *countTable,
    const std::string *kmerWriteBase = NULL, size_t kmerWriteBucket = 0) {

    KmerPosition<T, includeAdjacency, IncludeSeqLen> *hashSeqPair =
        initKmerPositionMemory<T, includeAdjacency, IncludeSeqLen>(totalKmers);

    // The write buffer is only used by the assignGroup/runIteration step below, not by the
    // fill step. Allocate it after fill frees its per-thread scratch to reduce peak memory
    // by keeping the scratch and full-size write buffer from coexisting.
    KmerPosition<T, false, IncludeSeqLen> *writeSeqPair = NULL;

    size_t elementsToSort;
    if (kmerWriteBase != NULL) {
        // Write mode: this split's k-mers were written to a bucket by the one-shot partition
        // pass, so read them back instead of re-scanning the whole sequence DB.
        elementsToSort = loadKmerBucket<T, includeAdjacency, IncludeSeqLen>(
            *kmerWriteBase, kmerWriteBucket, par.threads, hashSeqPair, totalKmers, par.compressKmerTmpFiles);
    } else if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
        std::pair<size_t, size_t> ret =
            fillKmerPositionArray<Parameters::DBTYPE_NUCLEOTIDES, T, includeAdjacency, IncludeSeqLen>(
                hashSeqPair, totalKmers, seqDbr, par,
                subMat, true, hashStartRange, hashEndRange, NULL);
        elementsToSort = ret.first;
        par.kmerSize = ret.second;
        Debug(Debug::INFO) << "\nAdjusted k-mer length " << par.kmerSize << "\n";
    } else {
        std::pair<size_t, size_t> ret =
            fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, IncludeSeqLen>(
                hashSeqPair, totalKmers, seqDbr, par,
                subMat, true, hashStartRange, hashEndRange, NULL);
        elementsToSort = ret.first;
    }

    if (hashEndRange == SIZE_T_MAX) {
        seqDbr.unmapData();
    }

    Debug(Debug::INFO) << "Sort kmer ";
    Timer timer;
    if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
        SORT_PARALLEL(
            hashSeqPair, hashSeqPair + elementsToSort,
            KmerPosition<T, includeAdjacency, IncludeSeqLen>::compareRepSequenceAndIdAndPosReverse);
    } else {
        SORT_PARALLEL(
            hashSeqPair, hashSeqPair + elementsToSort,
            KmerPosition<T, includeAdjacency, IncludeSeqLen>::compareRepSequenceAndIdAndPos);
    }
    Debug(Debug::INFO) << timer.lap() << "\n";

    SequenceWeights *sequenceWeights = NULL;
    if (par.PARAM_WEIGHT_FILE.wasSet) {
        sequenceWeights = new SequenceWeights(par.weightFile.c_str());
        if (sequenceWeights != NULL) {
            if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
                swapCenterSequence<Parameters::DBTYPE_NUCLEOTIDES, T, includeAdjacency, IncludeSeqLen>(
                    hashSeqPair, totalKmers, *sequenceWeights);
            } else {
                swapCenterSequence<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, IncludeSeqLen>(
                    hashSeqPair, totalKmers, *sequenceWeights);
            }
        }
    }

    std::vector<size_t> threadOffsets;
    size_t splitSize = elementsToSort / par.threads;

    threadOffsets.push_back(0);
    for (int thread = 1; thread < par.threads; thread++) {
        if (!par.useParallelism) {
            threadOffsets.push_back(elementsToSort);
            continue;
        }

        size_t prevHash = hashSeqPair[thread * splitSize].kmer;
        if (prevHash == SIZE_T_MAX) {
            for (int i = thread; i < par.threads; i++) {
                threadOffsets.push_back(elementsToSort);
            }
            break;
        }

        if (seqDbr.getDbtype() == Parameters::DBTYPE_NUCLEOTIDES) {
            prevHash = BIT_SET(prevHash, 63);
        }

        bool wasSet = false;
        for (size_t pos = thread * splitSize; pos < elementsToSort; pos++) {
            size_t currKmer = hashSeqPair[pos].kmer;
            if (seqDbr.getDbtype() == Parameters::DBTYPE_NUCLEOTIDES) {
                currKmer = BIT_SET(currKmer, 63);
            }
            if (prevHash != currKmer) {
                wasSet = true;
                threadOffsets.push_back(pos);
                break;
            }
        }

        if (wasSet == false) {
            for (int i = thread; i < par.threads; i++) {
                threadOffsets.push_back(elementsToSort);
            }
            break;
        }
    }
    threadOffsets.push_back(elementsToSort);

    if (phase == ComputationPhase::SetupCountTable) {
        if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
            assignGroup<Parameters::DBTYPE_NUCLEOTIDES, T, includeAdjacency, IncludeSeqLen>(
                hashSeqPair, NULL,
                par.includeOnlyExtendable, par.covMode, par.covThr,
                sequenceWeights, par.weightThr,
                par.threads, threadOffsets, subMat,
                AssignGroupFeature::Default,
                ComputationPhase::SetupCountTable, countTable);
        } else {
            assignGroup<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, IncludeSeqLen>(
                hashSeqPair, NULL,
                par.includeOnlyExtendable, par.covMode, par.covThr,
                sequenceWeights, par.weightThr,
                par.threads, threadOffsets, subMat,
                AssignGroupFeature::Default,
                ComputationPhase::SetupCountTable, countTable);
        }

        delete sequenceWeights;
        delete[] hashSeqPair;
        delete[] writeSeqPair;
        return NULL;
    }

    // Now that the fill step (and its per-thread scratch) is done, allocate the write
    // buffer used by the assignGroup iterations.
    if (par.needWriteBuffer) {
        writeSeqPair = initKmerPositionMemory<T, false, IncludeSeqLen>(totalKmers);
    }

    int iteration = 0;
    size_t writePos = 0;

    runIteration<T, includeAdjacency, IncludeSeqLen>(
        AssignGroupFeature::Default, iteration, writePos,
        hashEndRange, splitFile, hashSeqPair, writeSeqPair,
        seqDbr, par, subMat, countTable,
        sequenceWeights, threadOffsets, timer);

    if (hasFeature(assignGroupMask, AssignGroupFeature::AdjacentSeq)) {
        for (int iter = 0; iter < par.adjIteration; iter++) {
            runIteration<T, includeAdjacency, IncludeSeqLen>(
                AssignGroupFeature::AdjacentSeq, iteration, writePos,
                hashEndRange, splitFile, hashSeqPair, writeSeqPair,
                seqDbr, par, subMat, countTable,
                sequenceWeights, threadOffsets, timer);
        }
    }

    if (hasFeature(assignGroupMask, AssignGroupFeature::CountTable)) {
        for (int iter = 0; iter < par.countTableIteration; iter++) {
            runIteration<T, includeAdjacency, IncludeSeqLen>(
                AssignGroupFeature::CountTable, iteration, writePos,
                hashEndRange, splitFile, hashSeqPair, writeSeqPair,
                seqDbr, par, subMat, countTable,
                sequenceWeights, threadOffsets, timer);
        }
    }

    std::string splitDone = splitFile + ".done";
    removeKmerTmpFileIfExists(splitDone);
    FILE *done = openKmerTmpFileForOverwriteOrDie(splitDone, "w");
    if (fclose(done) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << splitDone << "\n";
        EXIT(EXIT_FAILURE);
    }

    delete sequenceWeights;
    if (hashEndRange != SIZE_T_MAX || par.needWriteBuffer) {
        delete[] hashSeqPair;
        hashSeqPair = NULL;
    }
    delete[] writeSeqPair;
    writeSeqPair = NULL;

    return hashSeqPair;
}

void setLinearFilterDefault(Parameters *p) {
    p->covThr = 0.8;
    p->maskMode = 0;
    p->spacedKmer = 0;
    p->kmerSize = Parameters::CLUST_LINEAR_DEFAULT_K;
    p->alphabetSize = MultiParam<NuclAA<int>>(NuclAA<int>(Parameters::CLUST_LINEAR_DEFAULT_ALPH_SIZE, 5));
    p->kmersPerSequence = Parameters::CLUST_LINEAR_KMER_PER_SEQ;
}


size_t computeKmerCount(DBReader<DBKeyType> &reader, size_t KMER_SIZE, size_t chooseTopKmer, float chooseTopKmerScale) {
    size_t totalKmers = 0;
    for(size_t id = 0; id < reader.getSize(); id++ ){
        int seqLen = static_cast<int>(reader.getSeqLen(id));
        int kmerAdjustedSeqLen = std::max(1, seqLen  - static_cast<int>(KMER_SIZE ) + 2) ;
        totalKmers += std::min(kmerAdjustedSeqLen, static_cast<int>( chooseTopKmer + (chooseTopKmerScale * seqLen)));
    }
    return totalKmers;
}

template <typename T, bool includeAdjacency, bool IncludeSeqLen>
size_t computeMemoryNeededLinearfilter(size_t totalKmer) {
    return sizeof(KmerPosition<T, includeAdjacency, IncludeSeqLen>) * totalKmer;
}


template <typename T, bool includeAdjacency, bool IncludeSeqLen>
std::vector<std::pair<size_t, size_t>> setupCountTable(
    Parameters &par,
    BaseMatrix *subMat,
    DBReader<DBKeyType> &seqDbr,
    size_t splits,
    size_t totalKmersPerSplit,
    size_t totalKmersCountTable
) {
    std::vector<std::pair<size_t, size_t>> hashRanges;
    Debug(Debug::INFO) << "Initiating count table\n";
    // fill hashDist 
    size_t * hashDist = new(std::nothrow) size_t[USHRT_MAX+1];
    Util::checkAllocation(hashDist, "Can not allocate hashDist memory");
    memset(hashDist, 0 , sizeof(size_t) * (USHRT_MAX+1));
    if(Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)){
        fillKmerPositionArray<Parameters::DBTYPE_NUCLEOTIDES, T, includeAdjacency, IncludeSeqLen>(NULL, SIZE_T_MAX, seqDbr, par, subMat, true, 0, SIZE_T_MAX, hashDist);
    }else{
        fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, IncludeSeqLen>(NULL, SIZE_T_MAX, seqDbr, par, subMat, true, 0, SIZE_T_MAX, hashDist);
    }

    seqDbr.remapData();


    if (splits > 1) {
        Debug(Debug::INFO) << "Not enough memory to process at once need to split for initiating count table\n";
        size_t maxBucketSize = 0;
        for(size_t i = 0; i < (USHRT_MAX+1); i++) {
            if(maxBucketSize < hashDist[i]){
                maxBucketSize = hashDist[i];
            }
        }
        if(maxBucketSize > totalKmersPerSplit){
            Debug(Debug::INFO) << "Not enough memory to run the kmermatcher. Minimum is at least " << maxBucketSize* sizeof(KmerPosition<T, includeAdjacency, IncludeSeqLen>) << " bytes\n";
            EXIT(EXIT_FAILURE);
        }
    }

    // find hashRange [0, totalKmersCountTable]
    size_t currBucketSize = 0;
    size_t currBucketStart = 0;
    size_t totalBucketSize = 0;
    for(size_t i = 0; i < (USHRT_MAX+1); i++){
        // if bucketsize exceeds subsampled countable range then break
        if(totalBucketSize+hashDist[i] >= totalKmersCountTable){
            hashRanges.emplace_back(currBucketStart, i - 1);
            break;
        }
        // if bucketsize exceeds memory limits -> multiple hashranges and reset startpos
        if(currBucketSize+hashDist[i] >= totalKmersPerSplit){
            hashRanges.emplace_back(currBucketStart, i - 1);
            currBucketSize = 0;
            currBucketStart = i;
        }
        currBucketSize+=hashDist[i];
        totalBucketSize+=hashDist[i];
    }
    delete [] hashDist;
    return hashRanges;
}


template <typename T, bool includeAdjacency, bool IncludeSeqLen>
int kmermatcherInner(Parameters& par, DBReader<DBKeyType>& seqDbr) {
    int querySeqType = seqDbr.getDbtype();
    size_t dbKeySize = seqDbr.getLastKey() +1 ;
    BaseMatrix *subMat;
    if (Parameters::isEqualDbtype(querySeqType, Parameters::DBTYPE_NUCLEOTIDES)) {
        subMat = new NucleotideMatrix(par.scoringMatrixFile.values.nucleotide().c_str(), 1.0, 0.0);
    }else {
        if (par.alphabetSize.values.aminoacid() == 21) {
            subMat = new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, 0.0);
        } else {
            SubstitutionMatrix sMat(par.scoringMatrixFile.values.aminoacid().c_str(), 8.0, -0.2f);
            subMat = new ReducedMatrix(sMat.probMatrix, sMat.subMatrixPseudoCounts, sMat.aa2num, sMat.num2aa, sMat.alphabetSize, par.alphabetSize.values.aminoacid(), 2.0);
        }
    }

    size_t memoryLimit=Util::computeMemory(par.splitMemoryLimit);

    Debug(Debug::INFO) << "\n";
    float kmersPerSequenceScale = (Parameters::isEqualDbtype(querySeqType, Parameters::DBTYPE_NUCLEOTIDES)) ?
                                        par.kmersPerSequenceScale.values.nucleotide() : par.kmersPerSequenceScale.values.aminoacid();
    size_t totalKmers = computeKmerCount(seqDbr, par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
    // The write buffer is KmerPosition<..,false,..> (no adjacency), so it is smaller than the
    // main array when adjacency is on. Size each separately instead of doubling the larger one,
    // matching the per-element cost used for totalKmersPerSplit below (else splits is over-counted).
    size_t totalSizeNeeded = computeMemoryNeededLinearfilter<T, includeAdjacency, IncludeSeqLen>(totalKmers)
                             + (par.needWriteBuffer ? computeMemoryNeededLinearfilter<T, false, IncludeSeqLen>(totalKmers) : 0);

    // --split-memory-limit sizes the main k-mer/write buffers (hashSeqPair [+ writeSeqPair]).
    // Reserve the tables that stay resident next to them for the whole split loop first,
    // then keep 5% headroom for transient allocations:
    //   seqLenTable : per-sequence length lookup (skipped when lengths are stored inline)
    //   countTable  : per-sequence k-mer counts, used by the center-swapping iterations
    size_t seqLenTableMemory = (!IncludeSeqLen) ? (dbKeySize + 1) * sizeof(T) : 0;
    size_t countTableMemory = 0;
    if (par.includeCountTable) {
        if (dbKeySize > std::numeric_limits<size_t>::max() / sizeof(short)) {
            Debug(Debug::ERROR) << "Count table is too large to allocate.\n";
            EXIT(EXIT_FAILURE);
        }
        countTableMemory = dbKeySize * sizeof(short);
    }
    size_t fixedMemory = seqLenTableMemory + countTableMemory;
    if (fixedMemory >= memoryLimit) {
        Debug(Debug::ERROR) << "Not enough memory to run the kmermatcher. Memory limit: " << memoryLimit
                            << " bytes, resident tables (seqkey_to_len + count table) need: " << fixedMemory << " bytes.\n"
                            << "Raise --split-memory-limit.\n";
        EXIT(EXIT_FAILURE);
    }
    size_t splitMemoryLimit = static_cast<size_t>(static_cast<double>(memoryLimit - fixedMemory) * 0.95);
    if (splitMemoryLimit == 0) {
        Debug(Debug::ERROR) << "Not enough memory to run the kmermatcher after reserving fixed memory\n";
        EXIT(EXIT_FAILURE);
    }

    size_t splits = std::max(static_cast<size_t>(1), static_cast<size_t>(std::ceil(static_cast<float>(totalSizeNeeded) / splitMemoryLimit)));
    size_t totalKmersPerSplit = std::max(
                                    static_cast<size_t>(1024 + 1),
                                    static_cast<size_t>(
                                        std::min(totalSizeNeeded, splitMemoryLimit) /
                                        (sizeof(KmerPosition<T, includeAdjacency, IncludeSeqLen>) +
                                        (par.needWriteBuffer ? sizeof(KmerPosition<T, false, IncludeSeqLen>) : 0))
                                    ) + 1
                                );

    if (!IncludeSeqLen) {
        T * seqkey_to_len = new(std::nothrow) T[dbKeySize+1];
        Util::checkAllocation(seqkey_to_len, "Can not allocate seqkey_to_len memory");
        memset(seqkey_to_len, 0, sizeof(T)*(dbKeySize+1));
#pragma omp parallel
        {
#pragma omp for schedule(dynamic, 1000)
            for (size_t id = 0; id < seqDbr.getSize(); id++) {
                DBKeyType seqKey = seqDbr.getDbKey(id);
                seqkey_to_len[seqKey] = static_cast<T>(seqDbr.getSeqLen(id));
            }
        }
        SeqLenData<T, false>::seqkey_to_len = seqkey_to_len;
    }

    std::vector<short> countTable;
    if (par.includeCountTable) {
        // countTable is already reserved in fixedMemory above; its fill step allocates its own
        // k-mer buffers, which fit in the same splitMemoryLimit (seqkey_to_len + countTable stay
        // resident during the fill too).
        countTable.assign(dbKeySize, 0);
        size_t countTableTotalKmers = static_cast<size_t>(totalKmers * par.countTableScale);
        // hashSeqPair + writeSeqPair for the count-table fill
        size_t countTableTotalSizeNeeded = computeMemoryNeededLinearfilter<T, false, false>(countTableTotalKmers) * 2;

        size_t countTableSplits = std::max(static_cast<size_t>(1), static_cast<size_t>(
            std::ceil(static_cast<double>(countTableTotalSizeNeeded) / splitMemoryLimit)
        ));

        size_t countTableKmersPerSplit = std::max(
            static_cast<size_t>(1024 + 1),
            static_cast<size_t>(
                std::min(countTableTotalSizeNeeded, splitMemoryLimit) /
                (sizeof(KmerPosition<T, false, false>) * 2)
            ) + 1
        );

        std::vector<std::pair<size_t, size_t>> countTableHashRanges;
        countTableHashRanges = setupCountTable<T, false, false>( 
            par, subMat, seqDbr,
            countTableSplits, countTableKmersPerSplit,
            static_cast<size_t>(totalKmers * par.countTableScale)
        );
        for (size_t split = 0; split < countTableHashRanges.size(); split++) {
            Debug(Debug::INFO) << "Fill count table for " << (split + 1) << " split\n";

            doComputation<T, false, false>(
                countTableKmersPerSplit,
                countTableHashRanges[split].first,
                countTableHashRanges[split].second,
                "COUNT_TABLE",
                AssignGroupFeature::Default,
                ComputationPhase::SetupCountTable,
                seqDbr, par, subMat, countTable.data());
        }
    }
    // set assigngroup matching mode
    AssignGroupMask assignGroupMask = AssignGroupFeature::Default;
    if (par.includeAdjacency) {
        assignGroupMask |= AssignGroupFeature::AdjacentSeq;
    }
    if (par.includeCountTable) {
        assignGroupMask |= AssignGroupFeature::CountTable;
    }

    std::vector<std::pair<size_t, size_t>> hashRanges = setupKmerSplits<T, includeAdjacency, IncludeSeqLen>(par, subMat, seqDbr, totalKmersPerSplit, splits);
    if(splits > 1){
        Debug(Debug::INFO) << "Process file into " << hashRanges.size() << " parts\n";
    }
    std::vector<std::string> splitFiles;
    KmerPosition<T, includeAdjacency, IncludeSeqLen> *hashSeqPair = NULL;

    std::string kmerWriteBase;
    bool useKmerWrite = false;

    size_t mpiRank = 0;
#ifdef HAVE_MPI
    splits = hashRanges.size();
    size_t fromSplit = 0;
    size_t splitCount = 1;
    mpiRank = MMseqsMPI::rank;
    unsigned int * splitCntPerProc = new unsigned int[MMseqsMPI::numProc];
    memset(splitCntPerProc, 0, sizeof(unsigned int) * MMseqsMPI::numProc);
    for(size_t i = 0; i < splits; i++){
        splitCntPerProc[i % MMseqsMPI::numProc] += 1;
    }
    for(int i = 0; i < MMseqsMPI::rank; i++){
        fromSplit += splitCntPerProc[i];
    }
    splitCount = splitCntPerProc[MMseqsMPI::rank];
    delete[] splitCntPerProc;

    for(size_t split = fromSplit; split < fromSplit+splitCount; split++) {
        std::string splitFileName = par.db2 + "_split_" +SSTR(split);
        hashSeqPair = doComputation<T, includeAdjacency, IncludeSeqLen>(
                        totalKmers, hashRanges[split].first, hashRanges[split].second, splitFileName,
                        assignGroupMask, ComputationPhase::Main,
                        seqDbr, par, subMat, countTable.empty() ? NULL : countTable.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(mpiRank == 0){
        for(size_t split = 0; split < splits; split++) {
            std::string splitFileName = par.db2 + "_split_" +SSTR(split);
            splitFiles.push_back(splitFileName);
        }
    }
#else
    // Optional write-to-disk (--kmer-write-to-disk): extract once and partition all k-mers into
    // per-split hash-range buckets on disk, so each split below reads its bucket instead of
    // re-scanning the sequence DB. Only helps when splitting (splits > 1).
    if ((splits > 1) && par.kmerWriteToDisk) {
        useKmerWrite = true;
        kmerWriteBase = par.db2 + "_bucket";
        size_t openFiles = static_cast<size_t>(par.threads) * hashRanges.size();
        long openMax = sysconf(_SC_OPEN_MAX);
        size_t fdBudget = (openMax > 128) ? static_cast<size_t>(openMax - 64) : 64;
        if (openFiles >= fdBudget) {
            Debug(Debug::ERROR) << "The k-mer buckets need " << openFiles << " open bucket files (threads x splits) "
                                << "but the open-file limit is " << openMax << ". Lower --threads or raise ulimit -n.\n";
            EXIT(EXIT_FAILURE);
        }
        unsigned int *hashToBucket = buildHashToBucketLookup(hashRanges);
        Debug(Debug::INFO) << "Partition k-mers into " << hashRanges.size() << " buckets\n";
        KmerPartitionSink<T, includeAdjacency, IncludeSeqLen> sink(kmerWriteBase, par.threads, hashRanges.size(), hashToBucket, par.compressKmerTmpFiles);
        if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
            std::pair<size_t, size_t> ret = fillKmerPositionArray<Parameters::DBTYPE_NUCLEOTIDES, T, includeAdjacency, IncludeSeqLen>(
                NULL, SIZE_T_MAX, seqDbr, par, subMat, true, 0, SIZE_T_MAX, NULL, &sink);
            par.kmerSize = ret.second;
        } else {
            fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, IncludeSeqLen>(
                NULL, SIZE_T_MAX, seqDbr, par, subMat, true, 0, SIZE_T_MAX, NULL, &sink);
        }
        sink.finish();
        seqDbr.remapData();
        delete[] hashToBucket;
    }
    for(size_t split = 0; split < hashRanges.size(); split++) {
        std::string splitFileName = par.db2 + "_split_" +SSTR(split);
        Debug(Debug::INFO) << "Generate k-mers list for " << (split+1) <<" split\n";

        std::string splitFileNameDone = splitFileName + ".done";
        if(FileUtil::fileExists(splitFileNameDone.c_str()) == false){
            hashSeqPair = doComputation<T, includeAdjacency, IncludeSeqLen>(
                        totalKmersPerSplit, hashRanges[split].first, hashRanges[split].second, splitFileName,
                        assignGroupMask, ComputationPhase::Main,
                        seqDbr, par, subMat, countTable.empty() ? NULL : countTable.data(),
                        useKmerWrite ? &kmerWriteBase : NULL, split);
        }

        splitFiles.push_back(splitFileName);
    }
#endif
    if(mpiRank == 0){
        std::vector<char> repSequence(seqDbr.getLastKey()+1);
        std::fill(repSequence.begin(), repSequence.end(), false);
        DBWriter dbw(par.db2.c_str(), par.db2Index.c_str(), par.threads, par.compressed,
                     (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) ? Parameters::DBTYPE_PREFILTER_REV_RES : Parameters::DBTYPE_PREFILTER_RES );
        dbw.open();

        Timer timer;
        if (splits > 1 || par.needWriteBuffer) {
            int maxIter = 1;
            if (par.includeAdjacency) {
                maxIter += par.adjIteration;
            }
            if (par.includeCountTable) {
                maxIter += par.countTableIteration;
            }

            seqDbr.unmapData();
            if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
                mergeKmerFilesAndOutput<Parameters::DBTYPE_NUCLEOTIDES, KmerEntryRev>(dbw, splitFiles, repSequence, par.threads, maxIter);
            } else {
                mergeKmerFilesAndOutput<Parameters::DBTYPE_AMINO_ACIDS, KmerEntry>(dbw, splitFiles, repSequence, par.threads, maxIter);
            }

            for (int iter = 0; iter < maxIter; ++iter) {
                for (size_t i = 0; i < splitFiles.size(); ++i) {
                    std::string doneFilePath = splitFiles[i] + "_iter_" + std::to_string(iter) + ".done";
                    if (FileUtil::fileExists(doneFilePath.c_str())) {
                        FileUtil::remove(doneFilePath.c_str());
                    }
                }
            }

            for (size_t i = 0; i < splitFiles.size(); ++i) {
                std::string splitDone = splitFiles[i] + ".done";
                if (FileUtil::fileExists(splitDone.c_str())) {
                    FileUtil::remove(splitDone.c_str());
                }
            }
        } else {
            if (Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)) {
                writeKmerMatcherResult<Parameters::DBTYPE_NUCLEOTIDES, T, includeAdjacency, IncludeSeqLen>(dbw, hashSeqPair, totalKmersPerSplit, repSequence, 1);
            } else {
                writeKmerMatcherResult<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, IncludeSeqLen>(dbw, hashSeqPair, totalKmersPerSplit, repSequence, 1);
            }
        }
        Debug(Debug::INFO) << "Time for fill: " << timer.lap() << "\n";

#pragma omp parallel num_threads(1)
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
#pragma omp for
            for (size_t id = 0; id < seqDbr.getSize(); id++) {
                char buffer[100];
                DBKeyType dbKey = seqDbr.getDbKey(id);
                if (repSequence[dbKey] == false) {
                    hit_t h;
                    h.prefScore = 0;
                    h.diagonal = 0;
                    h.seqId = dbKey;
                    int len = QueryMatcher::prefilterHitToBuffer(buffer, h);
                    dbw.writeData(buffer, len, dbKey, thread_idx);
                }
            }
        }
        dbw.close(false, false);
    }
    delete subMat;
    if(hashSeqPair){
        delete [] hashSeqPair;
    }
    // free the per-sequence length table allocated above; reset the static so a
    // later call in the same process does not dangle.
    if (!IncludeSeqLen && SeqLenData<T, false>::seqkey_to_len != NULL) {
        delete[] SeqLenData<T, false>::seqkey_to_len;
        SeqLenData<T, false>::seqkey_to_len = NULL;
    }

    return EXIT_SUCCESS;
}

template <typename T, bool includeAdjacency, bool IncludeSeqLen>
std::vector<std::pair<size_t, size_t>> setupKmerSplits(Parameters &par, BaseMatrix * subMat, DBReader<DBKeyType> &seqDbr, size_t totalKmers, size_t splits){
    std::vector<std::pair<size_t, size_t>> hashRanges;
    if (splits > 1) {
        Debug(Debug::INFO) << "Not enough memory to process at once need to split\n";
        size_t * hashDist = new(std::nothrow) size_t[USHRT_MAX+1];
        Util::checkAllocation(hashDist, "Can not allocate hashDist memory");
        memset(hashDist, 0 , sizeof(size_t) * (USHRT_MAX+1));
        if(Parameters::isEqualDbtype(seqDbr.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES)){
            fillKmerPositionArray<Parameters::DBTYPE_NUCLEOTIDES, T, includeAdjacency, IncludeSeqLen>(NULL, SIZE_T_MAX, seqDbr, par, subMat, true, 0, SIZE_T_MAX, hashDist);
        }else{
            fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, IncludeSeqLen>(NULL, SIZE_T_MAX, seqDbr, par, subMat, true, 0, SIZE_T_MAX, hashDist);
        }
        seqDbr.remapData();
        size_t maxBucketSize = 0;
        for(size_t i = 0; i < (USHRT_MAX+1); i++) {
            if(maxBucketSize < hashDist[i]){
                maxBucketSize = hashDist[i];
            }
        }
        if(maxBucketSize > totalKmers){
            Debug(Debug::INFO) << "Not enough memory to run the kmermatcher. Minimum is at least " << maxBucketSize* sizeof(KmerPosition<T, includeAdjacency, IncludeSeqLen>) << " bytes\n";
            EXIT(EXIT_FAILURE);
        }
        size_t currBucketSize = 0;
        size_t currBucketStart = 0;
        for(size_t i = 0; i < (USHRT_MAX+1); i++){
            if(currBucketSize+hashDist[i] >= totalKmers){
                hashRanges.emplace_back(currBucketStart, i - 1);
                currBucketSize = 0;
                currBucketStart = i;
            }
            currBucketSize+=hashDist[i];
        }
        hashRanges.emplace_back(currBucketStart, (USHRT_MAX+1));
        delete [] hashDist;
    }else{
        hashRanges.emplace_back(0, SIZE_T_MAX);
    }
    return hashRanges;
}

int kmermatcher(int argc, const char **argv, const Command &command) {
    MMseqsMPI::init(argc, argv);

    Parameters &par = Parameters::getInstance();
    setLinearFilterDefault(&par);
    par.parseParameters(argc, argv, command, true, 0, MMseqsParameter::COMMAND_CLUSTLINEAR);

    DBReader<DBKeyType> seqDbr(par.db1.c_str(), par.db1Index.c_str(), par.threads,
                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA);
    seqDbr.open(DBReader<DBKeyType>::NOSORT);
    // NOSORT is key order and the index is offset monotone in it, so the whole-db scans this module
    // runs once per hash pass and once per split really are sequential
    int querySeqType = seqDbr.getDbtype();

    setKmerLengthAndAlphabet(par, seqDbr.getAminoAcidDBSize(), querySeqType);
    std::vector<MMseqsParameter *> *params = command.params;
    par.printParameters(command.cmd, argc, argv, *params);
    Debug(Debug::INFO) << "Database size: " << seqDbr.getSize() << " type: " << seqDbr.getDbTypeName() << "\n";

    
    if (par.includeCountTable || par.includeAdjacency) {
        par.needWriteBuffer = true;
        par.useParallelism = true;
    } else {
        // if user turns on parallelism, it lead to using writebuffer with extra memory usage
        // linclust 1 setting: useParallelism = false
        par.needWriteBuffer = par.useParallelism;
    }

    if (par.linclustVersion == 1) {
        par.needWriteBuffer = false;
        par.includeCountTable = false;
        par.includeAdjacency = false;
        par.adjIteration = 0;
        par.countTableIteration = 0;
    } else if (par.linclustVersion == 2){
        if (par.includeAdjacency && par.adjIteration == 0) {
            Debug(Debug::ERROR) << "Adjacent Iteration must be greater than 0 when include Adjacent is true\n";
            EXIT(EXIT_FAILURE);
        }
        if (par.includeCountTable && par.countTableIteration == 0) {
            Debug(Debug::ERROR) << "CountTable Iteration must be greater than 0 when include CountTable is true\n";
            EXIT(EXIT_FAILURE);
        }
    }
    
    if (seqDbr.getMaxSeqLen() < SHRT_MAX) {
        if (par.includeAdjacency) { 
            kmermatcherInner<short, true, false>(par, seqDbr);
        }
        else {
            kmermatcherInner<short, false, false>(par, seqDbr);
        }
    }
    else {
        if (par.includeAdjacency) {
            kmermatcherInner<int, true, false>(par, seqDbr);
        }
        else {
            kmermatcherInner<int, false, false>(par, seqDbr);
        }
    }

    seqDbr.close();

    return EXIT_SUCCESS;
}

template <int TYPE, typename T, bool includeAdjacency, bool IncludeSeqLen>
void writeKmerMatcherResult(DBWriter & dbw,
                            KmerPosition<T, includeAdjacency, IncludeSeqLen> *hashSeqPair, size_t totalKmers,
                            std::vector<char> &repSequence, size_t threads) {
    std::vector<size_t> threadOffsets;
    size_t splitSize = totalKmers/threads;
    threadOffsets.push_back(0);
    for(size_t thread = 1; thread < threads; thread++){
        size_t kmer = hashSeqPair[thread*splitSize].kmer;
        size_t repSeqId = static_cast<size_t>(kmer);
        repSeqId=BIT_SET(repSeqId, 63);
        bool wasSet = false;
        for(size_t pos = thread*splitSize; pos < totalKmers; pos++){
            size_t currSeqId = hashSeqPair[pos].kmer;
            currSeqId=BIT_SET(currSeqId, 63);
            if(repSeqId != currSeqId){
                wasSet = true;
                threadOffsets.push_back(pos);
                break;
            }
        }
        if(wasSet == false){
            threadOffsets.push_back(totalKmers - 1 );
        }
    }
    threadOffsets.push_back(totalKmers);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for(size_t thread = 0; thread < threads; thread++){
        std::string prefResultsOutString;
        prefResultsOutString.reserve(100000000);
        char buffer[100];
        size_t lastTargetId = SIZE_T_MAX;
        unsigned int writeSets = 0;
        size_t kmerPos=0;
        size_t repSeqId = SIZE_T_MAX;
        for(kmerPos = threadOffsets[thread]; kmerPos < threadOffsets[thread+1] && hashSeqPair[kmerPos].kmer != SIZE_T_MAX; kmerPos++){
            size_t currKmer = hashSeqPair[kmerPos].kmer;
            int reverMask = 0;
            if(TYPE == Parameters::DBTYPE_NUCLEOTIDES){
                reverMask  = BIT_CHECK(currKmer, 63)==false;
                currKmer = BIT_CLEAR(currKmer, 63);
            }
            if(repSeqId != currKmer) {
                if (writeSets > 0) {
                    repSequence[repSeqId] = true;
                    dbw.writeData(prefResultsOutString.c_str(), prefResultsOutString.length(), repSeqId, thread);
                }else{
                    if(repSeqId != SIZE_T_MAX) {
                        repSequence[repSeqId] = false;
                    }
                }
                lastTargetId = SIZE_T_MAX;
                prefResultsOutString.clear();
                repSeqId = currKmer;
                hit_t h;
                h.seqId = repSeqId;
                h.prefScore = 0;
                h.diagonal = 0;
                int len = QueryMatcher::prefilterHitToBuffer(buffer, h);
                prefResultsOutString.append(buffer, len);
            }
            DBKeyType targetId = hashSeqPair[kmerPos].id;
            T diagonal = hashSeqPair[kmerPos].pos;
            size_t kmerOffset = 0;
            T prevDiagonal = diagonal;
            size_t maxDiagonal = 0;
            size_t diagonalCnt = 0;
            size_t topScore =0;
            int bestReverMask = reverMask;
            while(lastTargetId != targetId
                  && kmerPos+kmerOffset < threadOffsets[thread+1]
                  && hashSeqPair[kmerPos+kmerOffset].kmer == repSeqId
                  && hashSeqPair[kmerPos+kmerOffset].id == targetId){
                if(prevDiagonal == hashSeqPair[kmerPos+kmerOffset].pos){
                    diagonalCnt++;
                }else{
                    diagonalCnt = 1;
                }
                if(diagonalCnt >= maxDiagonal){
                    diagonal = hashSeqPair[kmerPos+kmerOffset].pos;
                    maxDiagonal = diagonalCnt;
                    if(TYPE == Parameters::DBTYPE_NUCLEOTIDES){
                        bestReverMask = BIT_CHECK(hashSeqPair[kmerPos+kmerOffset].kmer, 63) == false;
                    }
                }
                prevDiagonal = hashSeqPair[kmerPos+kmerOffset].pos;
                kmerOffset++;
                topScore++;
            }
            if(targetId == repSeqId || lastTargetId == targetId){
                lastTargetId = targetId;
                continue;
            }
            hit_t h;
            h.seqId = targetId;
            h.prefScore = (bestReverMask) ? -topScore : topScore;
            h.diagonal = diagonal;
            int len = QueryMatcher::prefilterHitToBuffer(buffer, h);
            prefResultsOutString.append(buffer, len);
            lastTargetId = targetId;
            writeSets++;
        }
        if (writeSets > 0) {
            repSequence[repSeqId] = true;
            dbw.writeData(prefResultsOutString.c_str(), prefResultsOutString.length(), repSeqId, thread);
        }else{
            if(repSeqId != SIZE_T_MAX) {
                repSequence[repSeqId] = false;
            }
        }
    }
}

static const size_t KMER_TMP_ZSTD_INPUT_BUFFER_SIZE = 65536;
static const size_t KMER_TMP_ZSTD_OUTPUT_BUFFER_SIZE = 65536;
static const int KMER_TMP_ZSTD_COMPRESSION_LEVEL = 2;
static const size_t KMER_MERGE_RESULT_BUFFER_RESERVE = 1024 * 1024;
static const size_t KMER_MERGE_RESULT_BUFFER_MAX_KEEP = 16 * 1024 * 1024;
static const size_t KMER_MERGE_MIN_MEMORY_HEADROOM = 1024ull * 1024ull * 1024ull;

static std::string kmerTmpFileName(const std::string &tmpFile, int iteration, int threadIdx, bool compressed) {
    std::string fileName = tmpFile + "_iter_" + std::to_string(iteration) + "_thread_" + std::to_string(threadIdx);
    if (compressed) {
        fileName.append(".zst");
    }
    return fileName;
}

static std::string existingKmerTmpFileName(const std::string &baseName, bool preferCompressed) {
    const std::string compressedName = baseName + ".zst";
    if (preferCompressed && FileUtil::fileExists(compressedName.c_str())) {
        return compressedName;
    }
    if (FileUtil::fileExists(baseName.c_str())) {
        return baseName;
    }
    if (FileUtil::fileExists(compressedName.c_str())) {
        return compressedName;
    }
    return "";
}

static void removeKmerTmpFileIfExists(const std::string &fileName) {
    if (FileUtil::fileExists(fileName.c_str())) {
        FileUtil::remove(fileName.c_str());
    }
}

static FILE *openKmerTmpFileForOverwriteOrDie(const std::string &fileName, const char *mode) {
    FILE *file = fopen(fileName.c_str(), mode);
    if (file == NULL) {
        perror(fileName.c_str());
        EXIT(EXIT_FAILURE);
    }
    return file;
}

static void writeAllOrDie(FILE *file, const void *data, size_t dataSize, const std::string &fileName) {
    if (dataSize == 0) {
        return;
    }
    size_t written = fwrite(data, sizeof(char), dataSize, file);
    if (written != dataSize) {
        Debug(Debug::ERROR) << "Can not write to file " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
}

class ZstdKmerTmpFileWriter {
public:
    ZstdKmerTmpFileWriter(const std::string &fileName, int compressionLevel)
        : fileName(fileName), file(NULL), cstream(NULL), outBuffer(NULL), closed(true) {
        file = openKmerTmpFileForOverwriteOrDie(fileName, "wb");
        cstream = ZSTD_createCStream();
        if (cstream == NULL) {
            Debug(Debug::ERROR) << "ZSTD_createCStream() failed for " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        outBuffer = static_cast<char *>(malloc(KMER_TMP_ZSTD_OUTPUT_BUFFER_SIZE));
        if (outBuffer == NULL) {
            Debug(Debug::ERROR) << "Cannot allocate zstd output buffer for " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (compressionLevel < 1) {
            compressionLevel = 1;
        }
        size_t ret = ZSTD_initCStream(cstream, compressionLevel);
        if (ZSTD_isError(ret)) {
            Debug(Debug::ERROR) << "ZSTD_initCStream() error for " << fileName << ". Error "
                                << ZSTD_getErrorName(ret) << "\n";
            EXIT(EXIT_FAILURE);
        }
        closed = false;
    }

    ~ZstdKmerTmpFileWriter() {
        if (closed == false) {
            close();
        }
        if (outBuffer != NULL) {
            free(outBuffer);
        }
        if (cstream != NULL) {
            ZSTD_freeCStream(cstream);
        }
    }

    void write(const void *data, size_t dataSize) {
        ZSTD_inBuffer input = { data, dataSize, 0 };
        while (input.pos < input.size) {
            ZSTD_outBuffer output = { outBuffer, KMER_TMP_ZSTD_OUTPUT_BUFFER_SIZE, 0 };
            size_t ret = ZSTD_compressStream(cstream, &output, &input);
            if (ZSTD_isError(ret)) {
                Debug(Debug::ERROR) << "ZSTD_compressStream() error for " << fileName << ". Error "
                                    << ZSTD_getErrorName(ret) << "\n";
                EXIT(EXIT_FAILURE);
            }
            writeAllOrDie(file, outBuffer, output.pos, fileName);
        }
    }

    void close() {
        if (closed) {
            return;
        }
        size_t remainingToFlush = 0;
        do {
            ZSTD_outBuffer output = { outBuffer, KMER_TMP_ZSTD_OUTPUT_BUFFER_SIZE, 0 };
            remainingToFlush = ZSTD_endStream(cstream, &output);
            if (ZSTD_isError(remainingToFlush)) {
                Debug(Debug::ERROR) << "ZSTD_endStream() error for " << fileName << ". Error "
                                    << ZSTD_getErrorName(remainingToFlush) << "\n";
                EXIT(EXIT_FAILURE);
            }
            writeAllOrDie(file, outBuffer, output.pos, fileName);
        } while (remainingToFlush != 0);

        if (fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close file " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        file = NULL;
        closed = true;
    }

private:
    std::string fileName;
    FILE *file;
    ZSTD_CStream *cstream;
    char *outBuffer;
    bool closed;
};

// --- bucket file IO wrappers (declared near the top, defined here after ZstdKmerTmpFileWriter) ---
// Buckets are transient spill: favour compression speed over ratio (lower level than the
// split-result tmp files). MMSEQS_BUCKET_ZSTD_LEVEL overrides for tuning.
static const int KMER_BUCKET_ZSTD_LEVEL = 2;
static void *bucketWriterOpen(const std::string &fileName, bool compress) {
    if (compress) {
        int level = KMER_BUCKET_ZSTD_LEVEL;
        const char *env = getenv("MMSEQS_BUCKET_ZSTD_LEVEL");
        if (env != NULL) {
            level = atoi(env);
        }
        return new ZstdKmerTmpFileWriter(fileName, level);
    }
    return openKmerTmpFileForOverwriteOrDie(fileName, "wb");
}

static void bucketWriterAppend(void *writer, bool compress, const void *data, size_t byteSize) {
    if (byteSize == 0) {
        return;
    }
    if (compress) {
        static_cast<ZstdKmerTmpFileWriter *>(writer)->write(data, byteSize);
    } else if (fwrite(data, 1, byteSize, static_cast<FILE *>(writer)) != byteSize) {
        Debug(Debug::ERROR) << "Can not write k-mer bucket file\n";
        EXIT(EXIT_FAILURE);
    }
}

static void bucketWriterClose(void *writer, bool compress) {
    if (writer == NULL) {
        return;
    }
    if (compress) {
        ZstdKmerTmpFileWriter *z = static_cast<ZstdKmerTmpFileWriter *>(writer);
        z->close();
        delete z;
    } else {
        fclose(static_cast<FILE *>(writer));
    }
}

// Read a whole bucket file into dst (up to maxBytes); returns bytes written, sets overflow if the
// file held more (bucket bigger than the split array, which should not happen).
static size_t bucketReadFile(const std::string &fileName, bool compress, void *dst, size_t maxBytes, bool &overflow) {
    overflow = false;
    FILE *f = fopen(fileName.c_str(), "rb");
    if (f == NULL) {
        return 0;
    }
    size_t written = 0;
    if (compress == false) {
        written = fread(dst, 1, maxBytes, f);
        char probe;
        if (fread(&probe, 1, 1, f) == 1) {
            overflow = true;
        }
    } else {
        ZSTD_DStream *dstream = ZSTD_createDStream();
        if (dstream == NULL) {
            Debug(Debug::ERROR) << "ZSTD_createDStream() failed for " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        ZSTD_initDStream(dstream);
        char *inBuffer = static_cast<char *>(malloc(KMER_TMP_ZSTD_INPUT_BUFFER_SIZE));
        Util::checkAllocation(inBuffer, "Can not allocate zstd input buffer");
        size_t inSize = 0;
        size_t inPos = 0;
        while (overflow == false) {
            if (inPos == inSize) {
                inSize = fread(inBuffer, 1, KMER_TMP_ZSTD_INPUT_BUFFER_SIZE, f);
                inPos = 0;
                if (inSize == 0) {
                    break;  // EOF
                }
            }
            ZSTD_inBuffer input = { inBuffer, inSize, inPos };
            ZSTD_outBuffer output = { dst, maxBytes, written };
            size_t ret = ZSTD_decompressStream(dstream, &output, &input);
            if (ZSTD_isError(ret)) {
                Debug(Debug::ERROR) << "ZSTD_decompressStream() error for " << fileName << ": "
                                    << ZSTD_getErrorName(ret) << "\n";
                EXIT(EXIT_FAILURE);
            }
            written = output.pos;
            inPos = input.pos;
            if (written == maxBytes && inPos < inSize) {
                overflow = true;  // output full but input remains
            }
        }
        free(inBuffer);
        ZSTD_freeDStream(dstream);
    }
    fclose(f);
    return written;
}

template <typename T>
class KmerTmpFileReader {
public:
    explicit KmerTmpFileReader(const std::string &fileName)
        : fileName(fileName), file(NULL), compressed(Util::endsWith(".zst", fileName)),
          entries(NULL), entrySize(0), offsetPos(0), dataSize(0),
          dstream(NULL), inBuffer(NULL), outBuffer(NULL), inBufferSize(0),
          input(), eof(false), zstdFrameComplete(false), decodedOffset(0), closed(true) {
        open();
    }

    ~KmerTmpFileReader() {
        close();
    }

    bool next(T &entry) {
        if (compressed) {
            return nextCompressed(entry);
        }
        if (offsetPos >= entrySize) {
            return false;
        }
        entry = entries[offsetPos];
        offsetPos++;
        return true;
    }

    void close() {
        if (closed) {
            return;
        }
        if (dataSize > 0 && entries != NULL && munmap((void *) entries, dataSize) < 0) {
            Debug(Debug::ERROR) << "Failed to munmap memory dataSize=" << dataSize << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (dstream != NULL) {
            ZSTD_freeDStream(dstream);
            dstream = NULL;
        }
        if (inBuffer != NULL) {
            free(inBuffer);
            inBuffer = NULL;
        }
        if (outBuffer != NULL) {
            free(outBuffer);
            outBuffer = NULL;
        }
        if (file != NULL && fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close file " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        file = NULL;
        closed = true;
    }

private:
    void open() {
        file = FileUtil::openFileOrDie(fileName.c_str(), "rb", true);
        if (compressed) {
            dstream = ZSTD_createDStream();
            if (dstream == NULL) {
                Debug(Debug::ERROR) << "ZSTD_createDStream() failed for " << fileName << "\n";
                EXIT(EXIT_FAILURE);
            }
            size_t ret = ZSTD_initDStream(dstream);
            if (ZSTD_isError(ret)) {
                Debug(Debug::ERROR) << "ZSTD_initDStream() error for " << fileName << ". Error "
                                    << ZSTD_getErrorName(ret) << "\n";
                EXIT(EXIT_FAILURE);
            }
            inBufferSize = KMER_TMP_ZSTD_INPUT_BUFFER_SIZE;
            inBuffer = static_cast<char *>(malloc(inBufferSize));
            outBuffer = static_cast<char *>(malloc(KMER_TMP_ZSTD_OUTPUT_BUFFER_SIZE));
            if (inBuffer == NULL || outBuffer == NULL) {
                Debug(Debug::ERROR) << "Cannot allocate zstd buffer for " << fileName << "\n";
                EXIT(EXIT_FAILURE);
            }
            input.src = inBuffer;
            input.size = 0;
            input.pos = 0;
            eof = false;
            zstdFrameComplete = false;
        } else {
            struct stat sb;
            if (fstat(fileno(file), &sb) == 0 && sb.st_size > 0) {
                entries = static_cast<T *>(FileUtil::mmapFile(file, &dataSize));
                if (dataSize % sizeof(T) != 0) {
                    Debug(Debug::ERROR) << "Malformed kmer temporary file " << fileName
                                        << ": size is not a multiple of entry size\n";
                    EXIT(EXIT_FAILURE);
                }
                Util::madviseLogged(entries, dataSize, POSIX_MADV_SEQUENTIAL, fileName.c_str());
            } else {
                entries = NULL;
                dataSize = 0;
            }
            entrySize = dataSize / sizeof(T);
            offsetPos = 0;
        }
        closed = false;
    }

    bool nextCompressed(T &entry) {
        while (decodedBytes.size() - decodedOffset < sizeof(T)) {
            if (decompressMore() == false) {
                if (decodedBytes.size() != decodedOffset) {
                    Debug(Debug::ERROR) << "Malformed zstd kmer temporary file " << fileName
                                        << ": trailing partial entry\n";
                    EXIT(EXIT_FAILURE);
                }
                return false;
            }
        }

        memcpy(&entry, decodedBytes.data() + decodedOffset, sizeof(T));
        decodedOffset += sizeof(T);
        compactDecodedBuffer();
        return true;
    }

    bool decompressMore() {
        while (true) {
            if (input.pos == input.size && eof == false) {
                size_t read = fread(inBuffer, sizeof(char), inBufferSize, file);
                if (read == 0) {
                    if (ferror(file)) {
                        Debug(Debug::ERROR) << "Cannot read file " << fileName << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    eof = true;
                }
                input.src = inBuffer;
                input.size = read;
                input.pos = 0;
            }
            if (input.pos == input.size && eof) {
                if (zstdFrameComplete == false) {
                    Debug(Debug::ERROR) << "Malformed zstd kmer temporary file " << fileName
                                        << ": truncated zstd frame\n";
                    EXIT(EXIT_FAILURE);
                }
                return false;
            }

            ZSTD_outBuffer output = { outBuffer, KMER_TMP_ZSTD_OUTPUT_BUFFER_SIZE, 0 };
            size_t ret = ZSTD_decompressStream(dstream, &output, &input);
            if (ZSTD_isError(ret)) {
                Debug(Debug::ERROR) << "ZSTD_decompressStream() error for " << fileName << ". Error "
                                    << ZSTD_getErrorName(ret) << "\n";
                EXIT(EXIT_FAILURE);
            }
            zstdFrameComplete = (ret == 0);
            if (output.pos > 0) {
                const size_t oldSize = decodedBytes.size();
                decodedBytes.resize(oldSize + output.pos);
                memcpy(decodedBytes.data() + oldSize, outBuffer, output.pos);
                return true;
            }
        }
    }

    void compactDecodedBuffer() {
        if (decodedOffset > KMER_TMP_ZSTD_OUTPUT_BUFFER_SIZE && decodedOffset * 2 > decodedBytes.size()) {
            decodedBytes.erase(decodedBytes.begin(), decodedBytes.begin() + decodedOffset);
            decodedOffset = 0;
        }
    }

    std::string fileName;
    FILE *file;
    bool compressed;
    T *entries;
    size_t entrySize;
    size_t offsetPos;
    size_t dataSize;
    ZSTD_DStream *dstream;
    char *inBuffer;
    char *outBuffer;
    size_t inBufferSize;
    ZSTD_inBuffer input;
    bool eof;
    bool zstdFrameComplete;
    std::vector<char> decodedBytes;
    size_t decodedOffset;
    bool closed;
};

template <int TYPE, typename T>
static bool queueNextEntryStreaming(KmerPositionQueue &queue, int file,
                                    KmerTmpFileReader<T> &reader, DBKeyType &repSeqId) {
    T entry;
    while (reader.next(entry)) {
        if (entry.seqId == DB_KEY_INVALID) {
            repSeqId = DB_KEY_INVALID;
            continue;
        }
        if (repSeqId == DB_KEY_INVALID) {
            repSeqId = entry.seqId;
            continue;
        }
        if(TYPE == Parameters::DBTYPE_NUCLEOTIDES){
            queue.push(FileKmerPosition(repSeqId, entry.seqId, entry.diagonal,
                                        entry.score, entry.getRev(), file));
        }else{
            queue.push(FileKmerPosition(repSeqId, entry.seqId, entry.diagonal,
                                        entry.score, file));
        }
        return true;
    }
    return false;
}

template <int TYPE, typename T>
void mergeKmerFilesAndOutput(DBWriter &dbw,
                             std::vector<std::string> tmpFiles,
                             std::vector<char> &repSequence,
                             int numThreads, int maxIter) {
    Debug(Debug::INFO) << "Merge splits ... ";

    std::vector<std::vector<std::string>> threadedFiles;
    threadedFiles.resize(numThreads);

    const bool preferCompressedTmpFiles = Parameters::getInstance().compressKmerTmpFiles;
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
        for (int iter = 0; iter < maxIter; iter++) {
            for (size_t i = 0; i < tmpFiles.size(); ++i) {
                std::string splitFileName = existingKmerTmpFileName(
                    kmerTmpFileName(tmpFiles[i], iter, threadIdx, false), preferCompressedTmpFiles);
                if (splitFileName.empty() == false) {
                    threadedFiles[threadIdx].push_back(splitFileName);
                }
            }
        }
    }

    int mergeThreads = numThreads;
    if (preferCompressedTmpFiles) {
        size_t maxFilesPerThread = 0;
        for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
            maxFilesPerThread = std::max(maxFilesPerThread, threadedFiles[threadIdx].size());
        }

        if (maxFilesPerThread > 0) {
            long openMax = sysconf(_SC_OPEN_MAX);
            size_t fdReserve = 2 * static_cast<size_t>(numThreads) + 64;
            size_t fdBudget = (openMax > 0 && static_cast<size_t>(openMax) > fdReserve)
                                  ? (static_cast<size_t>(openMax) - fdReserve) : 1;
            if (maxFilesPerThread >= fdBudget) {
                Debug(Debug::ERROR) << "Too many compressed k-mer temporary files for one merge lane ("
                                    << maxFilesPerThread << ") relative to open-file limit ("
                                    << openMax << "). Reduce kmermatcher splits or disable "
                                    << "--compress-kmer-tmp-files.\n";
                EXIT(EXIT_FAILURE);
            }
            mergeThreads = std::min(mergeThreads,
                                    std::max(1, static_cast<int>(fdBudget / maxFilesPerThread)));

            const size_t perReaderBytes =
                KMER_TMP_ZSTD_INPUT_BUFFER_SIZE + (3 * KMER_TMP_ZSTD_OUTPUT_BUFFER_SIZE) + 4096;
            const size_t memoryLimit = Util::computeMemory(Parameters::getInstance().splitMemoryLimit);
            size_t mergeHeadroom = std::max(KMER_MERGE_MIN_MEMORY_HEADROOM, memoryLimit / 10);
            const char *mergeHeadroomEnv = getenv("MMSEQS_KMER_MERGE_MEMORY_HEADROOM");
            if (mergeHeadroomEnv != NULL) {
                long value = atol(mergeHeadroomEnv);
                if (value > 0) {
                    mergeHeadroom = static_cast<size_t>(value);
                }
            }
            const size_t memBudget = (memoryLimit > mergeHeadroom) ? (memoryLimit - mergeHeadroom) : memoryLimit / 2;
            const size_t bytesPerLane = maxFilesPerThread * perReaderBytes;
            if (bytesPerLane > 0) {
                mergeThreads = std::min(mergeThreads,
                                        std::max(1, static_cast<int>(memBudget / bytesPerLane)));
            }
        }

        if (mergeThreads < numThreads) {
            Debug(Debug::INFO) << "Compressed k-mer tmp merge uses " << mergeThreads
                               << " concurrent lanes for " << numThreads
                               << " writer lanes to stay within file descriptor/memory bounds.\n";
        }
    }

#pragma omp parallel for num_threads(mergeThreads)
    for (int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
        const int fileCnt = threadedFiles[threadIdx].size();
        if (fileCnt == 0) {
            continue;
        }

        KmerTmpFileReader<T> **readers = new KmerTmpFileReader<T> *[fileCnt];
        DBKeyType *repSeqIds = new DBKeyType[fileCnt];

        for (size_t file = 0; file < threadedFiles[threadIdx].size(); file++) {
            readers[file] = new KmerTmpFileReader<T>(threadedFiles[threadIdx][file]);
            repSeqIds[file]  = DB_KEY_INVALID;
        }

        KmerPositionQueue queue;
        for (int file = 0; file < fileCnt; file++) {
            queueNextEntryStreaming<TYPE, T>(queue, file, *readers[file], repSeqIds[file]);
        }

        std::string prefResultsOutString;
        prefResultsOutString.reserve(KMER_MERGE_RESULT_BUFFER_RESERVE);
        char buffer[1024];
        bool hasRepSeq = (repSequence.size() > 0);
        DBKeyType currRepSeq = DB_KEY_INVALID;
        DBKeyType prevHitId = DB_KEY_INVALID;
        short prevDiagonal = 0;
        int diagonalScore = 0;
        int bestDiagonalCnt = 0;
        int bestRevertMask = 0;
        short bestDiagonal = 0;
        int topScore = 0;
        bool hasHit = false;

        const auto flushHit = [&]() {
            if(hasHit == false){
                return;
            }
            if(diagonalScore >= bestDiagonalCnt){
                bestDiagonalCnt = diagonalScore;
                bestDiagonal = prevDiagonal;
            }
            hit_t h;
            h.seqId = prevHitId;
            h.prefScore = (bestRevertMask) ? -topScore : topScore;
            h.diagonal = bestDiagonal;
            int len = QueryMatcher::prefilterHitToBuffer(buffer, h);
            prefResultsOutString.append(buffer, len);

            prevHitId = DB_KEY_INVALID;
            prevDiagonal = 0;
            diagonalScore = 0;
            bestDiagonalCnt = 0;
            bestRevertMask = 0;
            bestDiagonal = 0;
            topScore = 0;
            hasHit = false;
        };

        const auto flushRepSeq = [&]() {
            if(currRepSeq == DB_KEY_INVALID){
                return;
            }
            flushHit();
            dbw.writeData(prefResultsOutString.c_str(), prefResultsOutString.length(), currRepSeq, threadIdx);
            if (hasRepSeq) {
                repSequence[currRepSeq] = true;
            }
            prefResultsOutString.clear();
            if (prefResultsOutString.capacity() > KMER_MERGE_RESULT_BUFFER_MAX_KEEP) {
                std::string tmp;
                tmp.reserve(KMER_MERGE_RESULT_BUFFER_RESERVE);
                prefResultsOutString.swap(tmp);
            }
        };

        const auto startRepSeq = [&](DBKeyType repSeq) {
            flushRepSeq();
            currRepSeq = repSeq;
            if (hasRepSeq) {
                hit_t h;
                h.seqId = currRepSeq;
                h.prefScore = 0;
                h.diagonal = 0;
                int len = QueryMatcher::prefilterHitToBuffer(buffer, h);
                prefResultsOutString.append(buffer, len);
            }
        };

        while (queue.empty() == false) {
            FileKmerPosition res = queue.top();
            queue.pop();

            if (currRepSeq != res.repSeq) {
                startRepSeq(res.repSeq);
            }

            bool hitIsRepSeq = (currRepSeq == res.id);
            if (hitIsRepSeq) {
                queueNextEntryStreaming<TYPE, T>(queue, res.file, *readers[res.file], repSeqIds[res.file]);
                continue;
            }

            if (hasHit == false || prevHitId != res.id) {
                flushHit();
                hasHit = true;
                prevHitId = res.id;
                prevDiagonal = res.pos;
                diagonalScore = 0;
            } else if (prevDiagonal != res.pos) {
                if(diagonalScore >= bestDiagonalCnt){
                    bestDiagonalCnt = diagonalScore;
                    bestDiagonal = prevDiagonal;
                }
                prevDiagonal = res.pos;
                diagonalScore = 0;
            }

            diagonalScore += res.score;
            if(diagonalScore >= bestDiagonalCnt){
                bestDiagonalCnt = diagonalScore;
                bestDiagonal = res.pos;
                bestRevertMask = res.reverse;
            }
            topScore += res.score;

            queueNextEntryStreaming<TYPE, T>(queue, res.file, *readers[res.file], repSeqIds[res.file]);
        }

        if(currRepSeq != DB_KEY_INVALID){
            flushRepSeq();
        }

        for (size_t file = 0; file < threadedFiles[threadIdx].size(); file++) {
            readers[file]->close();
            delete readers[file];
        }

        delete[] repSeqIds;
        delete[] readers;
    }

    for (int tid = 0; tid < numThreads; ++tid) {
        for (std::string file : threadedFiles[tid]) {
            FileUtil::remove(file.c_str());
        }
    }
}

template <int TYPE, typename T, typename seqLenType, bool includeAdjacency, bool IncludeSeqLen>
void writeKmersToDisk(std::string tmpFile, KmerPosition<seqLenType, includeAdjacency, IncludeSeqLen> *hashSeqPair, size_t totalKmers,
                      int numThreads, std::vector<size_t> *threadQueryOffsets, int iteration) {
    const size_t BUFFER_SIZE = 2048;
    const Parameters &par = Parameters::getInstance();
    const bool compressTmpFiles = par.compressKmerTmpFiles;
#ifndef OPENMP
    (void) numThreads;
#endif

#pragma omp parallel num_threads(numThreads)
    {
        int tid = 0;
#ifdef OPENMP
        tid = omp_get_thread_num();
#endif

        size_t startIdx, endIdx;

        if (threadQueryOffsets == nullptr) {
            startIdx = 0;
            endIdx = totalKmers;
        } else {
            startIdx = (*threadQueryOffsets)[tid];
            endIdx = (*threadQueryOffsets)[tid + 1];
        }
        
        std::string tmpFileThread = kmerTmpFileName(tmpFile, iteration, tid, compressTmpFiles);
        removeKmerTmpFileIfExists(tmpFileThread);
        removeKmerTmpFileIfExists(kmerTmpFileName(tmpFile, iteration, tid, !compressTmpFiles));

        if (startIdx < endIdx && hashSeqPair[startIdx].kmer != SIZE_T_MAX) {
            FILE *filePtr = NULL;
            ZstdKmerTmpFileWriter *zstdWriter = NULL;
            if (compressTmpFiles) {
                zstdWriter = new ZstdKmerTmpFileWriter(tmpFileThread, KMER_TMP_ZSTD_COMPRESSION_LEVEL);
            } else {
                filePtr = openKmerTmpFileForOverwriteOrDie(tmpFileThread, "wb");
            }

            const auto writeEntries = [&](const T *entries, size_t count) {
                if (count == 0) {
                    return;
                }
                if (compressTmpFiles) {
                    zstdWriter->write(entries, sizeof(T) * count);
                } else {
                    size_t written = fwrite(entries, sizeof(T), count, filePtr);
                    if (written != count) {
                        Debug(Debug::ERROR) << "Can not write to file " << tmpFileThread << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                }
            };

            size_t repSeqId = SIZE_T_MAX;
            size_t lastTargetId = SIZE_T_MAX;
            seqLenType lastDiagonal = 0;
            int diagonalScore = 0;
            unsigned int writeSets = 0;
            size_t bufferPos = 0;
            size_t elementCnt = 0;

            T writeBuffer[BUFFER_SIZE];
            T nullEntry;
            nullEntry.seqId = DB_KEY_INVALID;
            nullEntry.diagonal = 0;

            for (size_t kmerPos = startIdx; kmerPos < endIdx && hashSeqPair[kmerPos].kmer != SIZE_T_MAX; kmerPos++) {
                size_t currKmer = hashSeqPair[kmerPos].kmer;
                if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                    currKmer = BIT_CLEAR(currKmer, 63);
                }

                if (repSeqId != currKmer) {
                    if (writeSets > 0 && elementCnt > 0) {
                        if (bufferPos > 0) {
                            writeEntries(writeBuffer, bufferPos);
                        }
                        writeEntries(&nullEntry, 1);
                    }
                    lastTargetId = SIZE_T_MAX;
                    bufferPos = 0;
                    elementCnt = 0;
                    repSeqId = currKmer;
                    writeBuffer[bufferPos].seqId = repSeqId;
                    writeBuffer[bufferPos].score = 0;
                    writeBuffer[bufferPos].diagonal = 0;
                    if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                        bool isReverse = (BIT_CHECK(hashSeqPair[kmerPos].kmer, 63) == false);
                        writeBuffer[bufferPos].setReverse(isReverse);
                    }
                    bufferPos++;
                }

                DBKeyType targetId = hashSeqPair[kmerPos].id;
                seqLenType diagonal = hashSeqPair[kmerPos].pos;
                int forward = 0;
                int reverse = 0;

                do {
                    diagonalScore += (diagonalScore == 0 || (lastTargetId == targetId && lastDiagonal == diagonal));
                    lastTargetId = hashSeqPair[kmerPos].id;
                    lastDiagonal = hashSeqPair[kmerPos].pos;
                    if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                        bool isReverse = (BIT_CHECK(hashSeqPair[kmerPos].kmer, 63) == false);
                        forward += (isReverse == false);
                        reverse += (isReverse == true);
                    }
                    kmerPos++;
                } while (kmerPos < endIdx &&
                        hashSeqPair[kmerPos].kmer != SIZE_T_MAX &&
                        repSeqId == hashSeqPair[kmerPos].kmer &&
                        targetId == hashSeqPair[kmerPos].id &&
                        hashSeqPair[kmerPos].pos == diagonal);
                kmerPos--;

                elementCnt++;
                // score is one byte, so a longer run is emitted as several records that the merge adds up
                int remainingScore = diagonalScore;
                diagonalScore = 0;
                do {
                    int chunk = (remainingScore > UCHAR_MAX) ? UCHAR_MAX : remainingScore;
                    writeBuffer[bufferPos].seqId = targetId;
                    writeBuffer[bufferPos].score = static_cast<unsigned char>(chunk);
                    writeBuffer[bufferPos].diagonal = diagonal;
                    if (TYPE == Parameters::DBTYPE_NUCLEOTIDES) {
                        bool isReverse = (reverse > forward) ? true : false;
                        writeBuffer[bufferPos].setReverse(isReverse);
                    }
                    bufferPos++;

                    if (bufferPos >= BUFFER_SIZE) {
                        writeEntries(writeBuffer, bufferPos);
                        bufferPos = 0;
                    }
                    remainingScore -= chunk;
                } while (remainingScore > 0);
                lastTargetId = targetId;
                writeSets++;
            }

            if (writeSets > 0 && elementCnt > 0) {
                if (bufferPos > 0) {
                    writeEntries(writeBuffer, bufferPos);
                }
                writeEntries(&nullEntry, 1);
            }

            if (compressTmpFiles) {
                zstdWriter->close();
                delete zstdWriter;
            } else {
                if (fclose(filePtr) != 0) {
                    Debug(Debug::ERROR) << "Cannot close file " << tmpFileThread << "\n";
                    EXIT(EXIT_FAILURE);
                }
            }
        }
    }
    
    std::string fileName = tmpFile + "_iter_" + std::to_string(iteration) + ".done";
    removeKmerTmpFileIfExists(fileName);
    FILE *done = openKmerTmpFileForOverwriteOrDie(fileName, "w");
    if (fclose(done) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
}

void setKmerLengthAndAlphabet(Parameters &parameters, size_t aaDbSize, int seqTyp) {
    if(Parameters::isEqualDbtype(seqTyp, Parameters::DBTYPE_NUCLEOTIDES)){
        if(parameters.kmerSize == 0) {
            parameters.kmerSize = std::max(17, static_cast<int>(log(static_cast<float>(aaDbSize))/log(4)));
            parameters.spacedKmerPattern = "";
            parameters.alphabetSize = 5;
        }
        if(parameters.kmersPerSequence == 0){
            parameters.kmersPerSequence = 60;
        }
    }else{
        if(parameters.kmerSize == 0){
            if((parameters.seqIdThr+0.001)>=0.99){
                parameters.kmerSize = 14;
                parameters.alphabetSize = 21;
            }else if((parameters.seqIdThr+0.001)>=0.9){
                parameters.kmerSize = 14;
                parameters.alphabetSize = 13;
            }else{
                parameters.kmerSize = std::max(10, static_cast<int>(log(static_cast<float>(aaDbSize))/log(8.7)));
                parameters.alphabetSize = 13;
            }
            parameters.spacedKmerPattern = "";
        }
        if(parameters.kmersPerSequence == 0){
            parameters.kmersPerSequence = 20;
        }
    }
}

// Only the instantiations other translation units need are explicit; the rest are implicit from use here.
// kmersearch.cpp / kmerindexdb.cpp use the IncludeSeqLen=true (linsearch) variants.
template std::pair<size_t, size_t>  fillKmerPositionArray<0, short, false, true>(KmerPosition<short, false, true> *, size_t, DBReader<DBKeyType> &, Parameters &, BaseMatrix *, bool, size_t, size_t, size_t *, KmerPartitionSink<short, false, true> *);
template std::pair<size_t, size_t>  fillKmerPositionArray<1, short, false, true>(KmerPosition<short, false, true> *, size_t, DBReader<DBKeyType> &, Parameters &, BaseMatrix *, bool, size_t, size_t, size_t *, KmerPartitionSink<short, false, true> *);
template std::pair<size_t, size_t>  fillKmerPositionArray<2, short, false, true>(KmerPosition<short, false, true> *, size_t, DBReader<DBKeyType> &, Parameters &, BaseMatrix *, bool, size_t, size_t, size_t *, KmerPartitionSink<short, false, true> *);

template KmerPosition<short, false, true> *initKmerPositionMemory(size_t size);

template size_t computeMemoryNeededLinearfilter<short, false, true>(size_t totalKmer);

template std::vector<std::pair<size_t, size_t>>  setupKmerSplits<short, false, true>(Parameters &, BaseMatrix *, DBReader<DBKeyType> &, size_t, size_t);

template void writeKmersToDisk<Parameters::DBTYPE_NUCLEOTIDES, KmerEntryRev, short, false, true>(std::string, KmerPosition<short, false, true> *, size_t, int, std::vector<size_t> *, int);
template void writeKmersToDisk<Parameters::DBTYPE_AMINO_ACIDS, KmerEntry, short, false, true>(std::string, KmerPosition<short, false, true> *, size_t, int, std::vector<size_t> *, int);

template void mergeKmerFilesAndOutput<Parameters::DBTYPE_NUCLEOTIDES, KmerEntryRev>(DBWriter &, std::vector<std::string>, std::vector<char> &, int, int);
template void mergeKmerFilesAndOutput<Parameters::DBTYPE_AMINO_ACIDS, KmerEntry>(DBWriter &, std::vector<std::string>, std::vector<char> &, int, int);

#undef SIZE_T_MAX
