
#include "Lin8Db.h"
#include "Lin8DbReader.h"
#include "Parameters.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"
#include "Timer.h"
#include "NodePlacement.h"
#include "SubstitutionMatrix.h"
#include "ReducedMatrix.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
#include "FastSort.h"
#include <cstdio>

class KmerExtractor {
public:
    KmerExtractor(unsigned int kmerLength, unsigned int residueClassCount, unsigned int mostKmersOneSequenceCanKeep,
                  unsigned int syncmerSmerLength);

    // the longest s-mer the syncmer test can pack five bits a residue into
    static const unsigned int MAX_SMER = 12;

    size_t selectFromSequence(const char *sequence, size_t length, const unsigned char *letterToCode,
                   unsigned int keepPerSequence);

    static std::vector<unsigned char> buildResidueToClassTable(const unsigned char *aa2num,
                                                       unsigned int reducedSize,
                                                       bool maskLowerCase);
    unsigned char unknownResidueClass() const { return static_cast<unsigned char>(residueClassCount); }

    const uint64_t *selectedKmerCodes() const { return selectedCodes.data(); }
    const uint16_t *selectedKmerPositions() const { return selectedPositions.data(); }

    // the score is this many low bits; the rest names the bucket and must never be compared
    static const unsigned int SCORE_BITS = 48;
    static const uint64_t SCORE_MASK = (uint64_t(1) << SCORE_BITS) - 1;

    static uint64_t selectionScore(uint64_t mixed) { return mixed & SCORE_MASK; }

    static const unsigned int CODE_BITS = 51;
    static const uint64_t CODE_MASK = (uint64_t(1) << CODE_BITS) - 1;

    static uint64_t spreadKmerCode(uint64_t code) {
        uint64_t v = code & CODE_MASK;
        v = (v * 0x9e3779b97f4a7c15ull) & CODE_MASK;
        v ^= v >> 26;
        v = (v * 0xbf58476d1ce4e5b9ull) & CODE_MASK;
        v ^= v >> 31;
        return v & CODE_MASK;
    }

    static uint64_t bucketForKmerHash(uint64_t spreadCode, uint64_t buckets) {
        return (spreadCode >> (CODE_BITS - KmerRecord::BUCKET_BITS)) % buckets;
    }
    static uint64_t recordKeyForKmerHash(uint64_t spreadCode) { return spreadCode & KmerRecord::KEY_MAX; }

    static uint64_t mixHash64(uint64_t value) {
        // splitmix64 finaliser: two multiplies and three shifts, enough to make the halves independent
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ull;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebull;
        value ^= value >> 31;
        return value;
    }

    static uint64_t fallbackSequenceHash(const char *sequence, size_t length,
                                const unsigned char *letterToCode);

    uint64_t packedFlankingClassesAt(const char *sequence, size_t length, const unsigned char *letterToCode,
                        size_t pos) const;
    uint64_t packedUnknownFlankingClasses() const;

private:
    void considerCandidate(uint64_t score, uint64_t key, uint16_t pos);
    bool doesNotBeatSelected(uint64_t score, uint16_t pos, size_t at) const;
    void findWorstSelectedIndex();
    size_t worstSelectedIndex;

    unsigned int kmerLength;
    unsigned int residueClassCount;
    unsigned int maxSelectedKmers;
    unsigned int selectionLimit;
    uint64_t leadingResidueWeight;
    unsigned int syncmerSmerLength;
    std::vector<uint64_t> selectedScores;
    std::vector<uint64_t> selectedCodes;
    std::vector<uint16_t> selectedPositions;
};

KmerExtractor::KmerExtractor(unsigned int kmerLength, unsigned int residueClassCount,
                             unsigned int mostKmersOneSequenceCanKeep, unsigned int syncmerSmerLength)
    : kmerLength(kmerLength), residueClassCount(residueClassCount), syncmerSmerLength(syncmerSmerLength),
      maxSelectedKmers(mostKmersOneSequenceCanKeep), selectionLimit(mostKmersOneSequenceCanKeep), worstSelectedIndex(0) {
    uint64_t span = 1;
    for (unsigned int i = 0; i < kmerLength; i++) {
        if (span > CODE_MASK / residueClassCount) {
            Debug(Debug::ERROR) << "A k-mer of " << kmerLength << " over an alphabet of "
                                << residueClassCount << " needs more than " << CODE_BITS
                                << " bit. Lower -k to " << i << " or fewer\n";
            EXIT(EXIT_FAILURE);
        }
        span *= residueClassCount;
    }
    leadingResidueWeight = span / residueClassCount;
    selectedScores.reserve(maxSelectedKmers);
    selectedCodes.reserve(maxSelectedKmers);
    selectedPositions.reserve(maxSelectedKmers);

}

uint64_t KmerExtractor::fallbackSequenceHash(const char *sequence, size_t length,
                                    const unsigned char *letterToCode) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < length; i++) {
        hash = (hash ^ letterToCode[static_cast<unsigned char>(sequence[i])]) * 1099511628211ull;
    }
    return mixHash64(hash) & CODE_MASK;
}

uint64_t KmerExtractor::packedFlankingClassesAt(const char *sequence, size_t length,
                                   const unsigned char *letterToCode, size_t pos) const {
    const unsigned int half = KmerRecord::ADJACENT_COUNT / 2;
    uint64_t packed = 0;
    for (unsigned int slot = 0; slot < KmerRecord::ADJACENT_COUNT; slot++) {
        uint64_t code = unknownResidueClass();
        if (slot < half) {
            const size_t before = half - slot;
            if (pos >= before) {
                code = letterToCode[static_cast<unsigned char>(sequence[pos - before])];
            }
        } else {
            const size_t after = pos + kmerLength + (slot - half);
            if (after < length) {
                code = letterToCode[static_cast<unsigned char>(sequence[after])];
            }
        }
        packed |= (code & KmerRecord::ADJACENT_MAX) << (slot * KmerRecord::ADJACENT_BITS);
    }
    return packed;
}

uint64_t KmerExtractor::packedUnknownFlankingClasses() const {
    uint64_t packed = 0;
    for (unsigned int slot = 0; slot < KmerRecord::ADJACENT_COUNT; slot++) {
        packed |= uint64_t(unknownResidueClass()) << (slot * KmerRecord::ADJACENT_BITS);
    }
    return packed;
}

std::vector<unsigned char> KmerExtractor::buildResidueToClassTable(const unsigned char *aa2num,
                                                           unsigned int reducedSize,
                                                           bool maskLowerCase) {
    std::vector<unsigned char> table(256, static_cast<unsigned char>(reducedSize));
    for (int letter = 'A'; letter <= 'Z'; letter++) {
        const unsigned char code = aa2num[letter];
        if (code < reducedSize) {
            table[letter] = code;
            // reading a masked database without honouring the mask is what the flag turns off
            table[letter - 'A' + 'a'] = maskLowerCase ? static_cast<unsigned char>(reducedSize) : code;
        }
    }
    return table;
}

// ties break on position so the choice does not depend on the order they were offered in
bool KmerExtractor::doesNotBeatSelected(uint64_t score, uint16_t pos, size_t at) const {
    return score > selectedScores[at] || (score == selectedScores[at] && pos >= selectedPositions[at]);
}

void KmerExtractor::findWorstSelectedIndex() {
    worstSelectedIndex = 0;
    for (size_t i = 1; i < selectedScores.size(); i++) {
        if (doesNotBeatSelected(selectedScores[i], selectedPositions[i], worstSelectedIndex)) {
            worstSelectedIndex = i;
        }
    }
}

// Most of what is offered once the set is full is worse than everything in it, and the one it has
// to beat is the worst, so remember which that is and let those go in a comparison. Finding the
// next worst costs the scan, but only a candidate that actually got in pays for it.
void KmerExtractor::considerCandidate(uint64_t score, uint64_t key, uint16_t pos) {
    if (selectionLimit == 0) {
        return;
    }
    if (selectedScores.size() < selectionLimit) {
        selectedScores.push_back(score);
        selectedCodes.push_back(key);
        selectedPositions.push_back(pos);
        if (selectedScores.size() == selectionLimit) {
            findWorstSelectedIndex();
        }
        return;
    }
    if (doesNotBeatSelected(score, pos, worstSelectedIndex)) {
        return;
    }
    selectedScores[worstSelectedIndex] = score;
    selectedCodes[worstSelectedIndex] = key;
    selectedPositions[worstSelectedIndex] = pos;
    findWorstSelectedIndex();
}

// A closed syncmer keeps a k-mer when its least hashing s-mer sits at one of its two ends. The test
// reads the k-mer and nothing else, so two sequences that share a k-mer either both keep it or both
// drop it, which the lowest hashing k-mers of a sequence cannot promise (Edgar 2021).
static bool isClosedSyncmerAt(const char *sequence, const unsigned char *letterToCode, size_t kmerStart,
                            unsigned int kmerLength, unsigned int syncmerSmerLength) {
    const uint64_t mask = (uint64_t(1) << (5 * syncmerSmerLength)) - 1;
    uint64_t smer = 0;
    uint64_t minimumSmerHash = UINT64_MAX, firstSmerHash = UINT64_MAX, lastSmerHash = UINT64_MAX;
    for (unsigned int i = 0; i < kmerLength; i++) {
        smer = ((smer << 5) | letterToCode[static_cast<unsigned char>(sequence[kmerStart + i])]) & mask;
        if (i + 1 >= syncmerSmerLength) {
            lastSmerHash = KmerExtractor::mixHash64(smer);
            firstSmerHash = (i + 1 == syncmerSmerLength) ? lastSmerHash : firstSmerHash;
            minimumSmerHash = std::min(minimumSmerHash, lastSmerHash);
        }
    }
    return firstSmerHash == minimumSmerHash || lastSmerHash == minimumSmerHash;
}

size_t KmerExtractor::selectFromSequence(const char *sequence, size_t length,
                              const unsigned char *letterToCode,
                              unsigned int keepPerSequence) {
    selectionLimit = keepPerSequence < maxSelectedKmers ? keepPerSequence : maxSelectedKmers;
    const unsigned char unknownCode = unknownResidueClass();
    selectedScores.clear();
    selectedCodes.clear();
    selectedPositions.clear();
    if (length < kmerLength) {
        return 0;
    }
    uint64_t code = 0;
    unsigned int filled = 0;
    for (size_t at = 0; at < length; at++) {
        const unsigned char in = letterToCode[static_cast<unsigned char>(sequence[at])];
        if (in >= unknownCode) {
            // the window cannot span an unknown residue, so it starts again after one
            code = 0;
            filled = 0;
            continue;
        }
        if (filled == kmerLength) {
            code -= static_cast<uint64_t>(
                        letterToCode[static_cast<unsigned char>(sequence[at - kmerLength])])
                    * leadingResidueWeight;
            filled--;
        }
        code = code * residueClassCount + in;
        filled++;
        if (filled == kmerLength) {
            const size_t kmerStart = at + 1 - kmerLength;
            if (syncmerSmerLength == 0
                || isClosedSyncmerAt(sequence, letterToCode, kmerStart, kmerLength, syncmerSmerLength)) {
                considerCandidate(selectionScore(mixHash64(code)), code, static_cast<uint16_t>(kmerStart));
            }
        }
    }
    return selectedScores.size();
}

// ---- lin8extractkmers ----
#ifdef OPENMP
#endif

static const uint64_t CHECKPOINT_BYTE_LIMIT = 64ull * 1024 * 1024 * 1024;

struct ExtractionChunk {
    uint64_t firstRank;
    uint64_t endRankExclusive;
    size_t sourceFileSlot;
};

static std::string checkpointManifestPath(const std::string &out, unsigned int node, size_t chunk) {
    return out + "." + SSTR(node) + "." + SSTR(chunk);
}

static std::vector<ExtractionChunk> planExtractionChunks(const RunDbReader &reader, const std::vector<size_t> &assignedFileSlots,
                                         uint64_t targetChunkBytes) {
    std::vector<ExtractionChunk> extractionChunks;
    const SequenceLocator &runs = reader.getSequenceLocator();
    for (size_t at = 0; at < assignedFileSlots.size(); at++) {
        uint64_t slotFirstRank = 0;
        uint64_t slotEndRankExclusive = 0;
        for (size_t i = 0; i < runs.size(); i++) {
            if (runs[i].fileIdx() % runs.filesPerNode() != assignedFileSlots[at]) {
                continue;
            }
            slotFirstRank = (slotEndRankExclusive == 0) ? runs[i].rankBase() : std::min(slotFirstRank, runs[i].rankBase());
            slotEndRankExclusive = std::max(slotEndRankExclusive, runs.rankEnd(i));
        }
        if (slotEndRankExclusive <= slotFirstRank) {
            continue;
        }
        const uint64_t firstByteOffset = runs.byteAtRank(slotFirstRank);
        const uint64_t endByteOffset = runs.byteAtRank(slotEndRankExclusive);
        const uint64_t slotByteCount = endByteOffset - firstByteOffset;
        const uint64_t chunkCount = std::max<uint64_t>(1, (slotByteCount + targetChunkBytes - 1) / targetChunkBytes);
        const uint64_t targetChunkSpanBytes = std::max<uint64_t>(1, (slotByteCount + chunkCount - 1) / chunkCount);
        for (uint64_t byteAt = firstByteOffset; byteAt < endByteOffset; byteAt += targetChunkSpanBytes) {
            ExtractionChunk chunk;
            chunk.sourceFileSlot = assignedFileSlots[at];
            chunk.firstRank = runs.rankAtByte(byteAt);
            chunk.endRankExclusive = std::min<uint64_t>(runs.rankAtByte(std::min(byteAt + targetChunkSpanBytes, endByteOffset)), slotEndRankExclusive);
            if (chunk.endRankExclusive > chunk.firstRank) {
                extractionChunks.push_back(chunk);
            }
        }
    }
    return extractionChunks;
}

static unsigned int selectedKmerLimitForLength(unsigned int baseKmersPerSequence, float kmersPerResidue, uint32_t length) {
    const double asked = (double) baseKmersPerSequence + (double) kmersPerResidue * (double) length;
    return (unsigned int) (asked < (double) length ? asked : (double) length);
}

static unsigned int maxSelectedKmersForAnySequence(unsigned int baseKmersPerSequence, float kmersPerResidue) {
    return selectedKmerLimitForLength(baseKmersPerSequence, kmersPerResidue, SequenceLocator::MAX_SEQ_LEN);
}

// ranks to read at once: mmap faults one readahead window at a time and never fills the queue
static const uint64_t RANKS_PER_READ_BATCH = 2048;
// a read arena a thread, the same size the other passes use
static const size_t BATCH_READ_ARENA_BYTES = 64u << 20;

static uint64_t extractAndWriteChunkKmers(const RunDbReader &dbReader, const ExtractionChunk &extractionChunk, BucketWriter<KmerRecord> &kmerBucketWriter,
                         const unsigned char *residueToClass, unsigned int kmerLength,
                         unsigned int residueClassCount, unsigned int baseKmersPerSequence, float kmersPerResidue,
                         unsigned int syncmerSmerLength, unsigned int threadCount,
                         std::vector<std::vector<uint64_t> > &subBucketCountsByThread) {
    const size_t bucketCount = KmerRecord::BUCKET_COUNT;
    uint64_t emittedRecordCount = 0;
#pragma omp parallel num_threads(threadCount) reduction(+ : emittedRecordCount)
    {
        unsigned int threadIdx = 0;
#ifdef OPENMP
        threadIdx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        KmerExtractor kmerExtractor(kmerLength, residueClassCount,
                                maxSelectedKmersForAnySequence(baseKmersPerSequence, kmersPerResidue), syncmerSmerLength);
        RunDbReader::Cursor dbCursor;
        KmerRecord kmerRecord;
        std::vector<uint64_t> &threadSubBucketCounts = subBucketCountsByThread[threadIdx];
        std::vector<uint64_t> ranksToRead;
        ranksToRead.reserve(RANKS_PER_READ_BATCH);
#pragma omp for schedule(dynamic, 1)
        for (uint64_t batchFirstRank = extractionChunk.firstRank; batchFirstRank < extractionChunk.endRankExclusive; batchFirstRank += RANKS_PER_READ_BATCH) {
            const uint64_t batchEndRankExclusive = std::min(batchFirstRank + RANKS_PER_READ_BATCH, extractionChunk.endRankExclusive);
            uint64_t nextRank = batchFirstRank;
            while (nextRank < batchEndRankExclusive) {
                ranksToRead.clear();
                while (nextRank < batchEndRankExclusive && ranksToRead.size() < RANKS_PER_READ_BATCH) {
                    if (dbReader.isValid(nextRank) && dbReader.getSeqLen(nextRank, dbCursor) <= SequenceLocator::MAX_SEQ_LEN) {
                        ranksToRead.push_back(nextRank);
                    }
                    nextRank++;
                }
                if (ranksToRead.empty()) {
                    break;
                }
                const size_t loadedRankCount = 1 + dbReader.startBatch(ranksToRead[0], ranksToRead.data() + 1, ranksToRead.size() - 1, threadIdx, 0);
                // the lane can fill before the batch does, so the rest waits for the next one
                if (loadedRankCount < ranksToRead.size()) {
                    nextRank = ranksToRead[loadedRankCount];
                }
                dbReader.awaitBatch(threadIdx, 0);
                for (size_t sequenceIndexInBatch = 0; sequenceIndexInBatch < loadedRankCount; sequenceIndexInBatch++) {
                    const uint64_t rank = ranksToRead[sequenceIndexInBatch];
                    const uint32_t length = dbReader.getSeqLen(rank, dbCursor);
                    const char *sequence = (sequenceIndexInBatch == 0) ? dbReader.batchQueryAt(threadIdx, 0)
                                                       : dbReader.batchAt(threadIdx, 0, sequenceIndexInBatch - 1);
                    const size_t selectedKmerCount = kmerExtractor.selectFromSequence(sequence, length, residueToClass,
                                                          selectedKmerLimitForLength(baseKmersPerSequence, kmersPerResidue, length));
                    for (size_t i = 0; i < selectedKmerCount; i++) {
                        const uint64_t spreadKmerHash = KmerExtractor::spreadKmerCode(kmerExtractor.selectedKmerCodes()[i]);
                        const uint64_t kmerStart = kmerExtractor.selectedKmerPositions()[i];
                        kmerRecord.set(KmerExtractor::recordKeyForKmerHash(spreadKmerHash), rank, kmerStart,
                                   kmerExtractor.packedFlankingClassesAt(sequence, length, residueToClass, kmerStart));
                        const size_t bucketIndex = KmerExtractor::bucketForKmerHash(spreadKmerHash, bucketCount);
                        kmerBucketWriter.add(threadIdx, kmerRecord, bucketIndex);
                        threadSubBucketCounts[bucketIndex * KmerRecord::SUB_BUCKET_COUNT + kmerRecord.subBucket()]++;
                        emittedRecordCount++;
                    }
                    // a sequence with no k-mer of its own would otherwise never be seen again
                    if (selectedKmerCount == 0) {
                        const uint64_t spreadKmerHash =
                            KmerExtractor::spreadKmerCode(KmerExtractor::fallbackSequenceHash(sequence, length, residueToClass));
                        kmerRecord.set(KmerExtractor::recordKeyForKmerHash(spreadKmerHash), rank, 0,
                                   kmerExtractor.packedUnknownFlankingClasses());
                        const size_t bucketIndex = KmerExtractor::bucketForKmerHash(spreadKmerHash, bucketCount);
                        kmerBucketWriter.add(threadIdx, kmerRecord, bucketIndex);
                        threadSubBucketCounts[bucketIndex * KmerRecord::SUB_BUCKET_COUNT + kmerRecord.subBucket()]++;
                        emittedRecordCount++;
                    }
                }
            }
        }
    }
    return emittedRecordCount;
}

static unsigned int longestKmer(unsigned int validResidueClassCount) {
    uint64_t span = 1;
    unsigned int k = 0;
    while (span <= KmerExtractor::CODE_MASK / validResidueClassCount) {
        span *= validResidueClassCount;
        k++;
    }
    return k;
}

static void configureKmerParameters(Parameters &par, uint64_t residues) {
    const bool nearIdentical = par.seqIdThr + 0.001 >= 0.99;
    if (par.alphabetSize.values.aminoacid() == 0) {
        const int reduced = (par.kmerSize == 0 && nearIdentical)
                                ? 21
                                : Parameters::CLUST_LINEAR_DEFAULT_ALPH_SIZE;
        par.alphabetSize = MultiParam<NuclAA<int> >(NuclAA<int>(reduced, 5));
    }
    // one class short of the alphabet: the unknown class ends a window rather than joining it
    const unsigned int validResidueClassCount = (unsigned int) par.alphabetSize.values.aminoacid() - 1;
    const unsigned int longest = longestKmer(validResidueClassCount);
    if (par.kmerSize == 0) {
        par.kmerSize = par.seqIdThr + 0.001 >= 0.9
                           ? 14
                           : std::max(10, (int)(log((double)residues) / log(8.7)));
        if ((unsigned int) par.kmerSize > longest) {
            Debug(Debug::WARNING) << "A database this size asks for a k-mer of " << par.kmerSize
                                  << ", which does not fit " << KmerExtractor::CODE_BITS
                                  << " bit over " << validResidueClassCount << " classes. Using " << longest << "\n";
            par.kmerSize = (int) longest;
        }
    }
}

int lin8extractkmers(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.kmerSize = 0;
    par.alphabetSize = MultiParam<NuclAA<int> >(NuclAA<int>(0, 5));
    par.parseParameters(argc, argv, command, true, 0, 0);

    // a bucket file stays open for the whole run, so the default limit of a thousand is not enough
    FileUtil::fixRlimitNoFile();

    const NodePlacement node = NodePlacement::resolve(par);
    RunDbReader reader(par.db1);
    reader.open();
    configureKmerParameters(par, reader.getTotalBytes());

    SubstitutionMatrix fullMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.2);
    const int reducedAlphabetSize = par.alphabetSize.values.aminoacid();
    ReducedMatrix reducedMatrix(fullMatrix.probMatrix, fullMatrix.subMatrixPseudoCounts, fullMatrix.aa2num, fullMatrix.num2aa,
                         fullMatrix.alphabetSize, reducedAlphabetSize, 2.0);
    const unsigned int validResidueClassCount = static_cast<unsigned int>(reducedAlphabetSize) - 1;
    const std::vector<unsigned char> letterToCode =
        KmerExtractor::buildResidueToClassTable(reducedMatrix.aa2num, validResidueClassCount, par.maskLowerCaseMode == 1);

    const unsigned int baseKmersPerSequence =
        par.kmersPerSequence > 1 ? par.kmersPerSequence - 1 : 1;
    const unsigned int syncmerSmerLength = (unsigned int) par.syncmerS;
    if (syncmerSmerLength > 0 && (syncmerSmerLength > KmerExtractor::MAX_SMER || syncmerSmerLength > (unsigned int) par.kmerSize)) {
        Debug(Debug::ERROR) << "--syncmer-s " << syncmerSmerLength << " must be at most the k-mer length "
                            << par.kmerSize << " and at most " << KmerExtractor::MAX_SMER << "\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t workingMemoryBudgetBytes = static_cast<size_t>(Util::computeMemory(par.splitMemoryLimit) * 0.95);
    const std::vector<size_t> assignedFileSlots = nodeFileSlots(reader.getSequenceLocator(), node);
    const std::vector<ExtractionChunk> extractionChunks = planExtractionChunks(reader, assignedFileSlots, CHECKPOINT_BYTE_LIMIT);
    Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes " << assignedFileSlots.size()
                       << " length blocks in " << extractionChunks.size() << " chunks, k " << par.kmerSize
                       << ", alphabet " << reducedAlphabetSize << ", keeping " << baseKmersPerSequence
                       << (syncmerSmerLength > 0 ? " of the closed syncmers at s " + SSTR(syncmerSmerLength)
                                        : std::string(" lowest hashing"))
                       << "\n";

    const std::string nodeBucketFilePrefix = par.db2 + "." + SSTR(node.index);
    std::vector<uint64_t> existingRecordsPerBucket(KmerRecord::BUCKET_COUNT, 0);
    const size_t subBucketCountEntryCount = KmerRecord::BUCKET_COUNT * KmerRecord::SUB_BUCKET_COUNT;
    const std::string subBucketCountsPath = par.db2 + "." + SSTR(node.index) + ".counts";
    size_t completedCountCheckpoints = 0;
    std::vector<uint64_t> persistedSubBucketCounts =
        readSubBucketCounts(subBucketCountsPath, subBucketCountEntryCount, completedCountCheckpoints);
    // a manifest covers a run of planned chunks and names in its span how many it reaches
    uint64_t completedExtractionChunkCount = 0;
    const size_t completedManifestCount =
        readBucketManifests(par.db2 + "." + SSTR(node.index), extractionChunks.size(), existingRecordsPerBucket, &completedExtractionChunkCount);
    size_t checkpointIndex = std::min(completedManifestCount, completedCountCheckpoints);
    if (checkpointIndex < completedManifestCount) {
        existingRecordsPerBucket.assign(existingRecordsPerBucket.size(), 0);
        completedExtractionChunkCount = 0;
        readBucketManifests(par.db2 + "." + SSTR(node.index), checkpointIndex, existingRecordsPerBucket, &completedExtractionChunkCount);
    }
    const size_t firstUnprocessedChunk = completedExtractionChunkCount;
    if (firstUnprocessedChunk > 0) {
        Debug(Debug::INFO) << "Resuming after " << firstUnprocessedChunk << " chunks a previous run finished\n";
    }

    std::vector<std::vector<uint64_t> > pendingSubBucketCountsByThread(par.threads,
                                                        std::vector<uint64_t>(subBucketCountEntryCount, 0));

    Timer timer;
    // It sees a sequence once, which was the argument for reading past the cache: nothing comes
    // back for it. That holds where the cache is a place to keep things. Where it is also what
    // reads ahead, giving it up costs more than the space it saves, and over NFS that was the
    // difference between 208 and 86 MB/s on the same 1500 GB.
    reader.openBatch(par.threads, BATCH_READ_ARENA_BYTES, workingMemoryBudgetBytes, RunDbReader::READ_AGAIN);
    BucketWriter<KmerRecord> writer(nodeBucketFilePrefix, KmerRecord::BUCKET_COUNT, par.threads,
                                    workingMemoryBudgetBytes - (size_t) par.threads * BATCH_READ_ARENA_BYTES);
    writer.openAt(existingRecordsPerBucket);
    uint64_t writtenRecordCount = 0;
    Debug::Progress progress(extractionChunks.size() - firstUnprocessedChunk);
    size_t checkpointFirstChunk = firstUnprocessedChunk;
    uint64_t pendingCheckpointRecordCount = 0;
    // what the chunk boundary costs: it flushes every bucket and waits for the disk to take it,
    // which over NFS is a commit a bucket
    double extractionSeconds = 0, checkpointFlushSeconds = 0;
    writer.resetCounts();
    for (size_t at = firstUnprocessedChunk; at < extractionChunks.size(); at++) {
        double phaseStartTime = omp_get_wtime();
        pendingCheckpointRecordCount += extractAndWriteChunkKmers(reader, extractionChunks[at], writer, letterToCode.data(), par.kmerSize,
                                       validResidueClassCount, baseKmersPerSequence,
                                       par.kmersPerSequenceScale.values.aminoacid(), syncmerSmerLength,
                                       par.threads, pendingSubBucketCountsByThread);
        extractionSeconds += omp_get_wtime() - phaseStartTime;
        // a durable chunk is a run of planned chunks ended by bytes, so a small database is one flush
        if (pendingCheckpointRecordCount * KmerRecord::DISK_BYTES >= CHECKPOINT_BYTE_LIMIT || at + 1 == extractionChunks.size()) {
            phaseStartTime = omp_get_wtime();
            writer.endChunk(par.threads);
            // the manifest goes down after the data it describes, so its presence means it is whole
            writeBucketManifest(checkpointManifestPath(par.db2, node.index, checkpointIndex) + ".manifest",
                                writer.chunkCounts(), "chunk", checkpointFirstChunk, at + 1);
            writeSubBucketCounts(subBucketCountsPath, persistedSubBucketCounts, pendingSubBucketCountsByThread, checkpointIndex + 1);
            writtenRecordCount += pendingCheckpointRecordCount;
            writer.resetCounts();
            checkpointFirstChunk = at + 1;
            pendingCheckpointRecordCount = 0;
            checkpointIndex++;
            checkpointFlushSeconds += omp_get_wtime() - phaseStartTime;
        }
        if (at + 1 == extractionChunks.size() || extractionChunks[at + 1].sourceFileSlot != extractionChunks[at].sourceFileSlot) {
            reader.releaseFileSlot(extractionChunks[at].sourceFileSlot);
        }
        progress.updateProgress();
    }
    writer.close();
    writeSubBucketCounts(subBucketCountsPath, persistedSubBucketCounts, pendingSubBucketCountsByThread, checkpointIndex);

    const std::string metadataTmpPath = par.db2 + "." + SSTR(node.index) + ".tmp";
    FILE *metadataFile = FileUtil::openAndDelete(metadataTmpPath.c_str(), "w");
    fprintf(metadataFile, "nodes\t%u\nbuckets\t%zu\nalphabet\t%d\nkmer\t%d\nranks\t%zu\n",
            node.count, (size_t) KmerRecord::BUCKET_COUNT, reducedAlphabetSize, par.kmerSize,
            (size_t) reader.getSize());
    if (fclose(metadataFile) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << metadataTmpPath << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(metadataTmpPath, par.db2);
    markNodeDone(par.db2, node.index);

    Debug(Debug::INFO) << "Where the time went: extracting " << (uint64_t) extractionSeconds
                       << "s, ending " << checkpointIndex << " chunks " << (uint64_t) checkpointFlushSeconds
                       << "s\n";
    Debug(Debug::INFO) << "Wrote " << writtenRecordCount << " k-mer records in " << timer.lap() << "\n";
    reader.close();
    return EXIT_SUCCESS;
}

// ---- lin8assignedpairs ----
#ifdef OPENMP
#endif

#ifdef OPENMP
#endif

static void readKmerBucketShape(const std::string &path, unsigned int &nodes, size_t &buckets, int &alphabet,
                      uint64_t &ranks) {
    FILE *in = fopen(path.c_str(), "r");
    if (in == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << path << ". Run lin8extractkmers first\n";
        EXIT(EXIT_FAILURE);
    }
    nodes = 0;
    buckets = 0;
    int kmerSize = 0;
    size_t seen = 0;
    const bool read = fscanf(in, "nodes\t%u\nbuckets\t%zu\nalphabet\t%d\nkmer\t%d\nranks\t%zu",
                             &nodes, &buckets, &alphabet, &kmerSize, &seen) == 5;
    fclose(in);
    ranks = seen;
    if (read == false || nodes == 0 || alphabet < 2 || buckets != KmerRecord::BUCKET_COUNT) {
        Debug(Debug::ERROR) << path << " names " << nodes << " nodes and " << buckets
                            << " buckets, and this build makes " << KmerRecord::BUCKET_COUNT << "\n";
        EXIT(EXIT_FAILURE);
    }
}

struct BucketExtent {
    unsigned int node;
    size_t first;
    size_t count;
};

struct BySubBucket {
    size_t operator()(const KmerRecord &record) const { return record.subBucket(); }
};

struct ByRepRankSubBlock {
    uint64_t ranks;
    size_t repRankBlocks;
    ByRepRankSubBlock(uint64_t ranks, size_t repRankBlocks) : ranks(ranks), repRankBlocks(repRankBlocks) {}
    size_t operator()(const PairRecord &record) const {
        return PairRecord::repRankSubBlockOf(record.rep(), ranks, repRankBlocks);
    }
};

// the descriptor comes from the caller: the extents of one node read one file at different offsets,
// and opening it once an extent is a round trip an extent on a shared filesystem
template <typename Record, typename SubOf>
static void scanBucketExtent(int fd, const std::string &path, const BucketExtent &extent,
                      const SubOf &subOf,
                      std::vector<uint64_t> &perPrefix, std::vector<size_t> &place,
                      RawArray<Record> &into) {
    if (fd < 0) {
        Debug(Debug::ERROR) << "Cannot open " << path << ", which node " << extent.node
                            << " should have written\n";
        EXIT(EXIT_FAILURE);
    }
    const off_t at = static_cast<off_t>(extent.first * Record::DISK_BYTES);
    const off_t span = static_cast<off_t>(extent.count * Record::DISK_BYTES);
    posix_fadvise(fd, at, span, POSIX_FADV_SEQUENTIAL);
    const size_t batch = std::min<size_t>(extent.count, 1u << 16);
    std::vector<unsigned char> raw(batch * Record::DISK_BYTES);
    for (size_t done = 0; done < extent.count; ) {
        const size_t want = std::min<size_t>(extent.count - done, batch);
        size_t got = 0;
        while (got < want * Record::DISK_BYTES) {
            const off_t at = static_cast<off_t>((extent.first + done) * Record::DISK_BYTES + got);
            const ssize_t read = pread(fd, raw.data() + got, want * Record::DISK_BYTES - got, at);
            if (read <= 0) {
                Debug(Debug::ERROR) << "Cannot read " << path << "\n";
                EXIT(EXIT_FAILURE);
            }
            got += static_cast<size_t>(read);
        }
        for (size_t i = 0; i < want; i++) {
            Record record;
            record.unpack(&raw[i * Record::DISK_BYTES]);
            // one slot means the caller wants them in the order the file holds them
            const size_t prefix = place.size() == 1 ? 0 : subOf(record);
            if (place.empty()) {
                perPrefix[prefix]++;
            } else {
                into[place[prefix]++] = record;
            }
        }
        done += want;
    }
    if (place.empty() == false) {
        posix_fadvise(fd, at, span, POSIX_FADV_DONTNEED);
    }
}

template <typename Record, typename SubOf, typename Less>
static std::vector<size_t> loadBucket(const std::string &prefix, const BucketCounts &counts,
                                      unsigned int nodes, size_t bucket, size_t prefixes,
                                      const SubOf &subOf, size_t budget, unsigned int threads,
                                      const char *what, const char *narrower, const char *producer,
                                      const Less &less, RawArray<Record> &into) {
    const std::vector<uint64_t> subCounts = counts.of(bucket);
    std::vector<size_t> starts(prefixes + 1, 0);
    for (size_t i = 0; i < prefixes; i++) {
        starts[i + 1] = starts[i] + subCounts[i];
    }
    requireArena(std::string(what) + " " + SSTR(bucket), starts.back() * sizeof(Record), budget,
                 narrower);
    into.resize(starts.back());

    std::vector<BucketExtent> extents;
    std::vector<std::string> path(nodes);
    for (unsigned int node = 0; node < nodes; node++) {
        path[node] = prefix + "." + SSTR(node) + "." + SSTR(bucket);
        const std::vector<uint64_t> mine = counts.of(bucket, node);
        size_t records = 0;
        for (size_t i = 0; i < prefixes; i++) {
            records += mine[i];
        }
        const size_t cut = std::min<size_t>(std::max<size_t>(threads / nodes, 1), records);
        for (size_t i = 0; i < cut; i++) {
            BucketExtent extent;
            extent.node = node;
            extent.first = records * i / cut;
            extent.count = records * (i + 1) / cut - extent.first;
            extents.push_back(extent);
        }
    }

    // one descriptor a node, shared by that node's extents: pread carries its own offset. Opened
    // here rather than inside the reads so a missing file is one message and not one a thread.
    std::vector<int> bucketFd(nodes, -1);
    for (unsigned int node = 0; node < nodes; node++) {
        bucketFd[node] = open(path[node].c_str(), O_RDONLY);
        if (bucketFd[node] < 0) {
            Debug(Debug::ERROR) << "Cannot open " << path[node] << ", which " << producer
                                << " wrote and this pass has not read yet. With --remove-tmp-files"
                                << " it drops them as it consumes them, so a run that got part way"
                                << " and lost its own output cannot be resumed: rerun " << producer
                                << " to make them again\n";
            EXIT(EXIT_FAILURE);
        }
    }
    // Read once, in the order the files hold them, then sort. The pass that counted first existed
    // only to find where each record belonged, and the sub bucket is the top of the sort key, so
    // sorting puts them there anyway. Over NFS that halved what this reads.
    std::vector<size_t> nodeStart(nodes + 1, 0);
    for (size_t i = 0; i < extents.size(); i++) {
        nodeStart[extents[i].node + 1] += extents[i].count;
    }
    for (unsigned int node = 0; node < nodes; node++) {
        nodeStart[node + 1] += nodeStart[node];
    }
    std::vector<uint64_t> nothing;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t i = 0; i < extents.size(); i++) {
        std::vector<size_t> place(1, nodeStart[extents[i].node] + extents[i].first);
        scanBucketExtent(bucketFd[extents[i].node], path[extents[i].node], extents[i], subOf,
                         nothing, place, into);
    }
    SORT_PARALLEL(into.begin(), into.begin() + into.size(), less);
    // the counts said where each sub bucket ends; if the files disagree the sort shows it here
    for (size_t j = 0; j < prefixes; j++) {
        if (starts[j] == starts[j + 1]) {
            continue;
        }
        if (subOf(into[starts[j]]) != j || subOf(into[starts[j + 1] - 1]) != j) {
            Debug(Debug::ERROR) << what << " " << bucket << " prefix " << j << " is not where the "
                                << "counts put it. Was " << producer << " still running?\n";
            EXIT(EXIT_FAILURE);
        }
    }
    for (unsigned int node = 0; node < nodes; node++) {
        if (bucketFd[node] >= 0) {
            close(bucketFd[node]);
        }
    }
    return starts;
}


static int adjacencyScore(const KmerRecord &member, const short **centerRow) {
    int score = 0;
    for (unsigned int slot = 0; slot < KmerRecord::ADJACENT_COUNT; slot++) {
        score += centerRow[slot][member.adjacentAt(slot)];
    }
    return score;
}

// ranksRepeat says whether a rank can appear twice; when it cannot, a member past the centres
// already chosen cannot be one of them, and the scan that looks for that is pure cost
static void swapCenterSequence(KmerRecord *group, std::vector<uint32_t> &lengths, size_t size,
                               size_t round, BaseMatrix *subMat, bool ranksRepeat) {
    const short *centerRow[KmerRecord::ADJACENT_COUNT];
    for (unsigned int slot = 0; slot < KmerRecord::ADJACENT_COUNT; slot++) {
        centerRow[slot] = subMat->subMatrix[group[round - 1].adjacentAt(slot)];
    }
    size_t best = round;
    int lowest = INT_MAX;
    for (size_t i = round; i < size; i++) {
        bool spent = false;
        for (size_t used = 0; ranksRepeat && used < round && spent == false; used++) {
            spent = group[used].rank() == group[i].rank();
        }
        if (spent) {
            continue;
        }
        const int score = adjacencyScore(group[i], centerRow);
        if (score <= lowest) {
            lowest = score;
            best = i;
        }
    }
    std::swap(group[round], group[best]);
    std::swap(lengths[round], lengths[best]);
}

// The lengths are read once and carried through the rounds, the way the original keeps a length
// beside every k-mer. A trillion will not fit in an array, so they come from the sequence locator, and
// asking it once a member rather than once a member a round is what makes that affordable.
static void assignGroup(KmerRecord *group, size_t size, const RunDbReader &reader,
                         float covThr, int covMode, bool onlyExtendable, BaseMatrix *subMat,
                         int adjacentRounds, std::vector<PairRecord> &out,
                         std::vector<uint32_t> &lengths, RunDbReader::Cursor &at) {
    if (size < 2) {
        return;
    }
    const size_t before = out.size();
    // the group arrives sorted by rank, so the cursor only moves forward and the same pass answers
    // whether a rank repeats, which is the only way a pair can
    lengths.resize(size);
    bool ranksRepeat = false;
    for (size_t i = 0; i < size; i++) {
        lengths[i] = reader.getSeqLen(group[i].rank(), at);
        ranksRepeat = ranksRepeat || (i > 0 && group[i].rank() == group[i - 1].rank());
    }
    for (size_t round = 0; round <= (size_t) adjacentRounds && round < size; round++) {
        if (round > 0) {
            swapCenterSequence(group, lengths, size, round, subMat, ranksRepeat);
        }
        const uint64_t rep = group[round].rank();
        const uint64_t repPos = group[round].pos();
        const uint32_t queryLen = lengths[round];
        for (size_t i = 0; i < size; i++) {
            const uint64_t member = group[i].rank();
            // the centre pairs with itself under every coverage mode, and align2clust drops it again
            if (member == rep) {
                continue;
            }
            const uint32_t targetLen = lengths[i];
            const int diagonal = static_cast<int>(repPos) - static_cast<int>(group[i].pos());
            const bool extendable =
                diagonal < 0 || diagonal > static_cast<int>(queryLen) - static_cast<int>(targetLen);
            const bool covered = Util::canBeCovered(covThr, covMode, queryLen, targetLen);
            if (onlyExtendable ? extendable : covered) {
                PairRecord pair;
                pair.set(rep, member, diagonal);
                out.push_back(pair);
            }
        }
    }
    if (adjacentRounds > 0 && ranksRepeat) {
        std::sort(out.begin() + before, out.end(), PairRecord::byRepAndMember);
        out.erase(std::unique(out.begin() + before, out.end(), PairRecord::sameRepAndMember),
                  out.end());
    }
}

static std::vector<size_t> setupThreadOffsets(const RawArray<KmerRecord> &records,
                                             size_t workSplits) {
    std::vector<size_t> threadOffsets(1, 0);
    const size_t even = records.size() / std::max<size_t>(1, workSplits);
    for (size_t thread = 1; thread < workSplits; thread++) {
        size_t at = std::max<size_t>(std::max(threadOffsets.back(), thread * even), 1);
        while (at < records.size() && records[at].key() == records[at - 1].key()) {
            at++;
        }
        threadOffsets.push_back(at);
    }
    threadOffsets.push_back(records.size());
    return threadOffsets;
}

int lin8assignedpairs(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    FileUtil::fixRlimitNoFile();
    const NodePlacement node = NodePlacement::resolve(par);
    RunDbReader reader(par.db1);
    reader.open();
    unsigned int writerNodes = 0;
    size_t buckets = 0;
    int alphabet = 0;
    uint64_t ranks = 0;
    readKmerBucketShape(par.db2, writerNodes, buckets, alphabet, ranks);
    requireEveryNodeDone(par.db2, writerNodes);
    const BucketCounts bucketCounts(par.db2, writerNodes, KmerRecord::SUB_BUCKET_COUNT,
                                    KmerRecord::BUCKET_COUNT);

    // the same reduced alphabet the extraction pass stored the residues in
    SubstitutionMatrix full(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.2);
    ReducedMatrix subMat(full.probMatrix, full.subMatrixPseudoCounts, full.aa2num, full.num2aa,
                         full.alphabetSize, alphabet, 2.0);
    const int adjacentRounds = par.includeAdjacency ? par.adjIteration : 0;

    Timer timer;
    uint64_t pairs = 0;
    uint64_t groups = 0;
    size_t myBuckets = 0;
    for (size_t bucket = node.index; bucket < buckets; bucket += node.count) {
        myBuckets++;
    }
    Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes " << myBuckets
                       << " of " << buckets << " buckets\n";

    if (par.lin8RepRankBlocks < 1 || (size_t) par.lin8RepRankBlocks > PairRecord::MAX_REP_RANK_BLOCKS) {
        Debug(Debug::ERROR) << "--repRankBlocks must be between 1 and " << PairRecord::MAX_REP_RANK_BLOCKS
                            << ", past which the sub repRankBlock arithmetic leaves sixty four bits\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t repRankBlockCount = (size_t) par.lin8RepRankBlocks;

    const uint64_t CHECKPOINT_BYTE_LIMIT = 4ull * 1024 * 1024 * 1024;
    const std::string prefix = par.db3 + "." + SSTR(node.index);
    const size_t budget = static_cast<size_t>(Util::computeMemory(par.splitMemoryLimit) * 0.95);
    std::vector<uint64_t> keep(repRankBlockCount, 0);
    // as in the extraction pass: the lower of what the manifests claim and what the table covers
    uint64_t resumeAt = 0;
    const size_t countEntries = repRankBlockCount * PairRecord::REP_RANK_SUB_BLOCKS;
    const std::string countsPath = par.db3 + "." + SSTR(node.index) + ".counts";
    size_t tableChunks = 0;
    std::vector<uint64_t> repRankSubBlockBase =
        readSubBucketCounts(countsPath, countEntries, tableChunks);
    const size_t doneChunks = readBucketManifests(prefix, myBuckets, keep, &resumeAt);
    const size_t done = std::min(doneChunks, tableChunks);
    if (done < doneChunks) {
        keep.assign(keep.size(), 0);
        resumeAt = 0;
        readBucketManifests(prefix, done, keep, &resumeAt);
    }
    if (done > 0) {
        Debug(Debug::INFO) << "Resuming after " << done << " chunks a previous run finished\n";
    }
    std::vector<std::vector<uint64_t> > repRankSubBlockCounts(par.threads,
                                                       std::vector<uint64_t>(countEntries, 0));
    // loadBucket has to hold one whole bucket out of the same budget, so the writer takes the rest
    uint64_t largestBucket = 0;
    for (size_t bucket = 0; bucket < buckets; bucket++) {
        const std::vector<uint64_t> counts = bucketCounts.of(bucket);
        uint64_t here = 0;
        for (size_t i = 0; i < counts.size(); i++) {
            here += counts[i];
        }
        largestBucket = std::max(largestBucket, here);
    }
    const size_t forReading = (size_t) largestBucket * sizeof(KmerRecord);
    if (forReading >= budget) {
        Debug(Debug::ERROR) << "The largest k-mer bucket is " << (forReading >> 30)
                            << " GB and the limit is " << (budget >> 30)
                            << " GB, so raise --split-memory-limit\n";
        EXIT(EXIT_FAILURE);
    }
    BucketWriter<PairRecord> writer(prefix, repRankBlockCount, par.threads, budget - forReading);
    writer.openAt(keep);

    Debug::Progress progress(myBuckets);
    uint64_t pending = 0;
    size_t chunk = done;
    size_t first = node.index;
    while (first < resumeAt) {
        first += node.count;
    }
    size_t chunkFirst = first;
    double spentReading = 0, spentSorting = 0, spentGrouping = 0, spentWriting = 0;
    writer.resetCounts();
    for (size_t bucket = first; bucket < buckets; bucket += node.count) {
        RawArray<KmerRecord> records;
        double mark = omp_get_wtime();
        const std::vector<size_t> starts =
            loadBucket(par.db2, bucketCounts, writerNodes, bucket, KmerRecord::SUB_BUCKET_COUNT,
                       BySubBucket(), budget - writer.bytesHeld(), par.threads, "Bucket",
                       "build with more buckets", "lin8-extractkmers", KmerRecord::byKeyAndRank,
                       records);
        spentReading += omp_get_wtime() - mark;
        mark = omp_get_wtime();
        spentSorting += omp_get_wtime() - mark;
        mark = omp_get_wtime();
        // a thread owns a stretch of the sorted bucket, so moving a center only touches its own
        const size_t bucketWorkSplits = par.threads * 8;
        const std::vector<size_t> threadOffsets = setupThreadOffsets(records, bucketWorkSplits);
        std::vector<uint64_t> made(par.threads, 0);
        std::vector<uint64_t> seen(par.threads, 0);
#pragma omp parallel num_threads(par.threads)
        {
            unsigned int thread = 0;
#ifdef OPENMP
            thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<PairRecord> out;
            std::vector<uint32_t> lengths;
            RunDbReader::Cursor at;
            std::vector<uint64_t> &counts = repRankSubBlockCounts[thread];
#pragma omp for schedule(dynamic, 1)
            for (int part = 0; part < (int) bucketWorkSplits; part++) {
                size_t begin = threadOffsets[part];
                while (begin < threadOffsets[part + 1]) {
                    size_t end = begin + 1;
                    while (end < threadOffsets[part + 1] && records[end].key() == records[begin].key()) {
                        end++;
                    }
                    out.clear();
                    assignGroup(&records[begin], end - begin, reader, par.covThr, par.covMode,
                                par.includeOnlyExtendable, &subMat, adjacentRounds, out,
                                lengths, at);
                    for (size_t i = 0; i < out.size(); i++) {
                        // one division for the file and the place inside it, not one each
                        const size_t fine = PairRecord::fineOf(out[i].rep(), ranks, repRankBlockCount);
                        writer.add(thread, out[i], fine / PairRecord::REP_RANK_SUB_BLOCKS);
                        counts[fine]++;
                    }
                    made[thread] += out.size();
                    seen[thread] += (end - begin > 1);
                    begin = end;
                }
            }
        }
        uint64_t inBucket = 0;
        for (unsigned int thread = 0; thread < par.threads; thread++) {
            inBucket += made[thread];
            groups += seen[thread];
        }
        spentGrouping += omp_get_wtime() - mark;
        pairs += inBucket;
        pending += inBucket * PairRecord::DISK_BYTES;
        const bool last = bucket + node.count >= buckets;
        if (pending >= CHECKPOINT_BYTE_LIMIT || last) {
            const double put = omp_get_wtime();
            writer.endChunk(par.threads);
            writeBucketManifest(prefix + "." + SSTR(chunk) + ".manifest", writer.chunkCounts(),
                                "bucket", chunkFirst, bucket + 1);
            writeSubBucketCounts(countsPath, repRankSubBlockBase, repRankSubBlockCounts, chunk + 1);
            if (par.removeTmpFiles) {
                dropConsumed(par.db2, writerNodes, chunkFirst, bucket + 1, node.count);
            }
            writer.resetCounts();
            spentWriting += omp_get_wtime() - put;
            chunkFirst = bucket + node.count;
            pending = 0;
            chunk++;
        }
        progress.updateProgress();
    }
    writer.close();

    const std::string shapeTmp = par.db3 + "." + SSTR(node.index) + ".tmp";
    FILE *shape = FileUtil::openAndDelete(shapeTmp.c_str(), "w");
    fprintf(shape, "nodes\t%u\nrepRankBlocks\t%zu\nranks\t%zu\n", node.count, repRankBlockCount,
            (size_t) ranks);
    if (fclose(shape) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << shapeTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(shapeTmp, par.db3);
    markNodeDone(par.db3, node.index);
    Debug(Debug::INFO) << "Where the time went: reading " << (uint64_t) spentReading
                       << "s, sorting " << (uint64_t) spentSorting << "s, grouping "
                       << (uint64_t) spentGrouping << "s, publishing " << (uint64_t) spentWriting
                       << "s\n";
    Debug(Debug::INFO) << "Made " << pairs << " pairs from " << groups << " groups in " << timer.lap() << "\n";
    reader.close();
    return EXIT_SUCCESS;
}

// ---- lin8pref ----
#ifdef OPENMP
#endif

static void readRepRankBlockShape(const std::string &path, unsigned int &nodes, size_t &repRankBlocks,
                      uint64_t &ranks) {
    FILE *in = fopen(path.c_str(), "r");
    if (in == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << path << ". Run lin8assignedpairs first\n";
        EXIT(EXIT_FAILURE);
    }
    nodes = 0;
    repRankBlocks = 0;
    size_t seen = 0;
    const bool read = fscanf(in, "nodes\t%u\nrepRankBlocks\t%zu\nranks\t%zu", &nodes, &repRankBlocks, &seen) == 3;
    fclose(in);
    ranks = seen;
    if (read == false || nodes == 0 || repRankBlocks == 0 || repRankBlocks > PairRecord::MAX_REP_RANK_BLOCKS) {
        Debug(Debug::ERROR) << path << " names " << nodes << " nodes and " << repRankBlocks
                            << " repRankBlocks, which is not a set of candidate pairs\n";
        EXIT(EXIT_FAILURE);
    }
}

static void pickBestDiagonal(const PairRecord *run, size_t size, PairRecord &out) {
    size_t best = 0;
    size_t bestCount = 0;
    size_t at = 0;
    while (at < size) {
        size_t end = at + 1;
        while (end < size && run[end].diagonal() == run[at].diagonal()) {
            end++;
        }
        if (end - at >= bestCount) {
            bestCount = end - at;
            best = at;
        }
        at = end;
    }
    out = run[best];
}

static void addRanksWithNoRows(uint64_t from, uint64_t until,
                               const std::vector<std::vector<PairRecord> > &rows,
                               const RunDbReader &live, std::vector<PairRecord> &out) {
    size_t piece = 0;
    size_t at = 0;
    for (uint64_t rank = from; rank < until; rank++) {
        while (piece < rows.size()) {
            if (at >= rows[piece].size()) {
                piece++;
                at = 0;
            } else if (rows[piece][at].rep() < rank) {
                at++;
            } else {
                break;
            }
        }
        if (piece < rows.size() && rows[piece][at].rep() == rank) {
            continue;
        }
        if (live.isValid(rank) == false) {
            continue;
        }
        PairRecord alone;
        alone.set(rank, rank, 0);
        out.push_back(alone);
    }
}

static void keepBestPairPerMember(const PairRecord *pairs, size_t size, std::vector<PairRecord> &out) {
    size_t at = 0;
    while (at < size) {
        size_t end = at + 1;
        while (end < size && PairRecord::sameRepAndMember(pairs[at], pairs[end])) {
            end++;
        }
        PairRecord best;
        pickBestDiagonal(pairs + at, end - at, best);
        out.push_back(best);
        at = end;
    }
}

int lin8pref(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    FileUtil::fixRlimitNoFile();
    const NodePlacement node = NodePlacement::resolve(par);
    unsigned int writerNodes = 0;
    size_t repRankBlocks = 0;
    uint64_t ranks = 0;
    readRepRankBlockShape(par.db1, writerNodes, repRankBlocks, ranks);
    const size_t budget = static_cast<size_t>(Util::computeMemory(par.splitMemoryLimit) * 0.95);
    requireEveryNodeDone(par.db1, writerNodes);
    const BucketCounts repRankBlockCounts(par.db1, writerNodes, PairRecord::REP_RANK_SUB_BLOCKS, repRankBlocks);

    // an absent bitmap makes every rank valid, so there is nothing here to be null
    RunDbReader live(par.db2);
    live.open();

    const size_t outEntries = repRankBlocks * PairRecord::REP_RANK_SUB_BLOCKS;
    const std::string outCounts = par.db3 + "." + SSTR(node.index) + ".counts";
    std::vector<std::vector<uint64_t> > outRepRankSubBlock(1, std::vector<uint64_t>(outEntries, 0));
    std::vector<uint64_t> outBase(outEntries, 0);

    Timer timer;
    uint64_t read = 0;
    uint64_t kept = 0;
    size_t myRepRankBlocks = 0;
    for (size_t repRankBlock = node.index; repRankBlock < repRankBlocks; repRankBlock += node.count) {
        myRepRankBlocks++;
    }
    Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes " << myRepRankBlocks
                       << " of " << repRankBlocks << " repRankBlocks\n";

    Debug::Progress progress(myRepRankBlocks);
    std::vector<std::pair<std::string, std::string> > pending;
    size_t pendingFirst = node.index;
    uint64_t pendingBytes = 0;
    std::vector<std::vector<PairRecord> > bestPairs(PairRecord::REP_RANK_SUB_BLOCKS);
    std::vector<PairRecord> alone;
    std::vector<size_t> aloneAt(PairRecord::REP_RANK_SUB_BLOCKS + 1, 0);
    std::vector<size_t> outAt(PairRecord::REP_RANK_SUB_BLOCKS + 1, 0);
    std::vector<std::vector<unsigned char> > packed(par.threads);
    RawArray<PairRecord> pairs;
    for (size_t repRankBlock = node.index; repRankBlock < repRankBlocks; repRankBlock += node.count) {
        const std::vector<size_t> starts =
            loadBucket(par.db1, repRankBlockCounts, writerNodes, repRankBlock, PairRecord::REP_RANK_SUB_BLOCKS,
                       ByRepRankSubBlock(ranks, repRankBlocks), budget, par.threads, "Representative rank block", "raise --rep-rank-blocks",
                       "lin8-assignedpairs", PairRecord::byRepAndMember, pairs);
        read += pairs.size();

#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t i = 0; i < PairRecord::REP_RANK_SUB_BLOCKS; i++) {
            bestPairs[i].clear();
            keepBestPairPerMember(pairs.begin() + starts[i], starts[i + 1] - starts[i], bestPairs[i]);
        }

        alone.clear();
        addRanksWithNoRows(PairRecord::firstRankOf(repRankBlock, ranks, repRankBlocks),
                           PairRecord::firstRankOf(repRankBlock + 1, ranks, repRankBlocks), bestPairs, live, alone);

        {
            size_t at = 0;
            for (size_t i = 0; i < PairRecord::REP_RANK_SUB_BLOCKS; i++) {
                aloneAt[i] = at;
                while (at < alone.size()
                       && PairRecord::repRankSubBlockOf(alone[at].rep(), ranks, repRankBlocks) == i) {
                    at++;
                }
            }
            aloneAt[PairRecord::REP_RANK_SUB_BLOCKS] = alone.size();
        }

        std::vector<uint64_t> &outCount = outRepRankSubBlock[0];
        outAt[0] = 0;
        for (size_t i = 0; i < PairRecord::REP_RANK_SUB_BLOCKS; i++) {
            const size_t mine = bestPairs[i].size() + (aloneAt[i + 1] - aloneAt[i]);
            outAt[i + 1] = outAt[i] + mine;
            outCount[repRankBlock * PairRecord::REP_RANK_SUB_BLOCKS + i] = mine;
        }

        const std::string path = par.db3 + "." + SSTR(node.index) + "." + SSTR(repRankBlock);
        const std::string tmp = path + ".tmp";
        const int out = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (out < 0 || ftruncate(out, static_cast<off_t>(outAt.back() * PairRecord::DISK_BYTES)) != 0) {
            Debug(Debug::ERROR) << "Cannot write " << tmp << "\n";
            EXIT(EXIT_FAILURE);
        }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t i = 0; i < PairRecord::REP_RANK_SUB_BLOCKS; i++) {
            const size_t mine = outAt[i + 1] - outAt[i];
            if (mine == 0) {
                continue;
            }
            unsigned int thread = 0;
#ifdef OPENMP
            thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<unsigned char> &into = packed[thread];
            into.resize(mine * PairRecord::DISK_BYTES);
            size_t wrote = 0;
            size_t row = 0;
            size_t next = aloneAt[i];
            while (row < bestPairs[i].size()) {
                while (next < aloneAt[i + 1] && alone[next].rep() < bestPairs[i][row].rep()) {
                    alone[next++].pack(&into[wrote++ * PairRecord::DISK_BYTES]);
                }
                bestPairs[i][row++].pack(&into[wrote++ * PairRecord::DISK_BYTES]);
            }
            while (next < aloneAt[i + 1]) {
                alone[next++].pack(&into[wrote++ * PairRecord::DISK_BYTES]);
            }
            size_t got = 0;
            const size_t bytes = mine * PairRecord::DISK_BYTES;
            while (got < bytes) {
                const off_t at = static_cast<off_t>(outAt[i] * PairRecord::DISK_BYTES + got);
                const ssize_t put = pwrite(out, into.data() + got, bytes - got, at);
                if (put <= 0) {
                    Debug(Debug::ERROR) << "Cannot write " << tmp << "\n";
                    EXIT(EXIT_FAILURE);
                }
                got += static_cast<size_t>(put);
            }
        }
        kept += outAt.back();
        sync_file_range(out, 0, 0, SYNC_FILE_RANGE_WRITE);
        if (close(out) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        pending.push_back(std::make_pair(tmp, path));
        pendingBytes += outAt.back() * PairRecord::DISK_BYTES;
        // batched like a chunk: a restart redoes the whole pass, so a repRankBlock needs no flush of its own
        if (pendingBytes >= PUBLISH_BATCH_BYTES || pending.size() >= PUBLISH_BATCH_FILES
            || repRankBlock + node.count >= repRankBlocks) {
            publishAllAtomically(pending, par.threads);
            if (par.removeTmpFiles) {
                dropConsumed(par.db1, writerNodes, pendingFirst, repRankBlock + 1, node.count);
            }
            pendingFirst = repRankBlock + node.count;
            pendingBytes = 0;
        }
        progress.updateProgress();
    }
    writeSubBucketCounts(outCounts, outBase, outRepRankSubBlock, repRankBlocks);

    const std::string shapeTmp = par.db3 + "." + SSTR(node.index) + ".tmp";
    FILE *shape = FileUtil::openAndDelete(shapeTmp.c_str(), "w");
    fprintf(shape, "nodes\t%u\nrepRankBlocks\t%zu\nranks\t%zu\n", node.count, repRankBlocks, (size_t) ranks);
    if (fclose(shape) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << shapeTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(shapeTmp, par.db3);
    markNodeDone(par.db3, node.index);
    live.close();
    Debug(Debug::INFO) << "Read " << read << " pairs, kept " << kept << " in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}
