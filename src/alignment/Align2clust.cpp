#include "DistanceCalculator.h"
#include "Util.h"
#include "Parameters.h"
#include "Matcher.h"
#include "Debug.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "QueryMatcher.h"
#include "IndexReader.h"
#include "FastSort.h"
#include "BlockAligner.h"
#include "SwipeAligner.h"
#include "Alignment.h"
#include "AlignmentSymmetry.h"
#include "MMseqsNode.h"
#include "FileUtil.h"
#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <ctime>
#include <mutex>
#include <condition_variable>
#include <algorithm>

#ifdef OPENMP
#include <omp.h>
#endif

#define MIN_SIZE 32

struct ClusterResult {
    size_t sequenceIdx;
    DBLocalId representativeId;
    size_t prefSize;
    std::vector<DBLocalId> memberIds;
};

struct PrefInfo {
    DBLocalId id;
    DBLocalId len;
    size_t size;

    static bool compareBySizeAndId(const PrefInfo &first, const PrefInfo &second){
        if(first.size > second.size)
            return true;
        if(second.size > first.size)
            return false;
        if(first.len > second.len)
            return true;
        if(second.len > first.len)
            return false;
        if(first.id < second.id)
            return true;
        if(second.id < first.id)
            return false;
        return false;
    }
};

struct SetCoverCandidate {
    DBLocalId representativeId;
    size_t memberCount;
    size_t memberOffset;

    static bool compareByMemberOffset(const SetCoverCandidate &first, const SetCoverCandidate &second) {
        return first.memberOffset < second.memberOffset;
    }
};

struct SetCoverComparator {
    bool operator()(const SetCoverCandidate& a, const SetCoverCandidate& b) const {
        if (a.memberCount < b.memberCount) {
            return true;
        }
        if (b.memberCount < a.memberCount) {
            return false;
        }
        if (a.representativeId < b.representativeId) {
            return true;
        }
        if (b.representativeId < a.representativeId) {
            return false;
        }
        return false;
    }
};

static std::mutex clusterMutex;
static std::condition_variable clusterCondition;
static std::condition_variable reorderSpaceCondition;
static std::vector<ClusterResult> reorderSlots;    // slot storage, size == reorderCapacity
static std::vector<unsigned char> reorderFilled;   // 1 == slot holds an unconsumed result
static size_t reorderCapacity = 0;                 // max out-of-order window
static size_t reorderBufferedCount = 0;            // number of filled slots
// Max-heap of candidates by member count (plain vector so the pool stays compactable).
static std::vector<SetCoverCandidate> setCoverCandidates;
static SetCoverComparator setCoverComparator;
// Shared backing store for all member ids referenced by setCoverCandidates.
static std::vector<DBLocalId> setCoverMemberPool;
static size_t setCoverLiveMemberCount = 0;

static size_t currentProcessPosition = 0;
static size_t currentPrefSize = 0;
static bool allCalculationsDone = false;

typedef std::atomic<DBLocalId> ClusterAssignment;

static DBLocalId loadAssignedCluster(const ClusterAssignment *assignedCluster, size_t sequenceId) {
    return assignedCluster[sequenceId].load(std::memory_order_relaxed);
}

static void storeAssignedCluster(ClusterAssignment *assignedCluster, size_t sequenceId, DBLocalId representativeId) {
    assignedCluster[sequenceId].store(representativeId, std::memory_order_relaxed);
}

// Serialize a single alignment record and append it to the per-representative buffer.
static void appendAlignmentResult(std::string &alnResultBuffer, char *lineBuffer, const Matcher::result_t &result, bool addBacktrace) {
    size_t len = Matcher::resultToBuffer(lineBuffer, result, addBacktrace);
    alnResultBuffer.append(lineBuffer, len);
}

// Compact setCoverMemberPool in place when over a quarter is dead. (member count, rep) orders the candidates strictly, so the rebuilt heap pops as before.
static const size_t ALIGN2CLUST_MIN_COMPACTION_DEAD_MEMBERS = 16 * 1024 * 1024 / sizeof(DBLocalId);

template <typename T>
static void compactSetCoverMemberPool(std::vector<SetCoverCandidate> &candidates, std::vector<T> &memberPool, size_t liveMemberCount) {
    const size_t deadMemberCount = memberPool.size() - liveMemberCount;
    if (memberPool.empty() ||
        deadMemberCount < ALIGN2CLUST_MIN_COMPACTION_DEAD_MEMBERS ||
        deadMemberCount * 4 <= memberPool.size()) {
        return;
    }
    std::sort(candidates.begin(), candidates.end(), SetCoverCandidate::compareByMemberOffset);
    size_t newOffset = 0;
    for (SetCoverCandidate &candidate : candidates) {
        if (candidate.memberOffset != newOffset) {
            std::copy(memberPool.begin() + candidate.memberOffset, memberPool.begin() + candidate.memberOffset + candidate.memberCount, memberPool.begin() + newOffset);
            candidate.memberOffset = newOffset;
        }
        newOffset += candidate.memberCount;
    }
    memberPool.resize(newOffset);
    std::make_heap(candidates.begin(), candidates.end(), SetCoverComparator());
}

// Env override for the reorder-buffer size; 0 (unset/invalid) means size from memory.
static size_t getReorderBufferLimitFromEnv() {
    const char *envValue = getenv("MMSEQS_ALIGN2CLUST_REORDER_LIMIT");
    if (envValue == nullptr || *envValue == '\0') {
        return 0;
    }

    char *end = nullptr;
    unsigned long long parsedValue = strtoull(envValue, &end, 10);
    if (end == envValue || *end != '\0' || parsedValue == 0) {
        Debug(Debug::WARNING) << "Ignoring invalid MMSEQS_ALIGN2CLUST_REORDER_LIMIT=" << envValue
                              << "; sizing the reorder buffer automatically\n";
        return 0;
    }
    return static_cast<size_t>(parsedValue);
}

static size_t computeReorderCapacity(const Parameters &par, size_t dbSize, int mode, size_t resultCount) {
    const size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);

    size_t fixedMemory = dbSize * sizeof(ClusterAssignment);
    if (mode == Parameters::SET_COVER) {
        fixedMemory += dbSize * sizeof(PrefInfo);
    }

    const size_t budget = (memoryLimit > fixedMemory)
        ? static_cast<size_t>(static_cast<double>(memoryLimit - fixedMemory) * 0.9)
        : 0;
    const size_t bytesPerResult = sizeof(ClusterResult) + (par.maxResListLen + 1) * sizeof(DBLocalId);

    size_t capacity = std::min(resultCount, std::max<size_t>(1, budget / bytesPerResult));
    const size_t userCap = getReorderBufferLimitFromEnv();  // 0 == auto
    if (userCap != 0) {
        capacity = std::min(capacity, userCap);
    }
    capacity = std::max<size_t>(1, capacity);

    Debug(Debug::INFO) << "Reorder buffer sizing: memory limit " << memoryLimit << " byte, reserved (fixed) "
                       << fixedMemory << " byte, budget " << budget << " byte, " << bytesPerResult
                       << " byte/result\n";
    Debug(Debug::INFO) << "Reorder buffer capacity: " << capacity << " results ("
                       << capacity * sizeof(ClusterResult) << " byte pre-allocated slots)\n";
    return capacity;
}

static void pushClusterResult(ClusterResult &&clusterResult) {
    const size_t idx = clusterResult.sequenceIdx;
    bool shouldNotifyClusterThread = false;
    {
        std::unique_lock<std::mutex> lock(clusterMutex);
        // Wait for this result's slot to free up. The next-in-order slot is always
        // free, so the producer the consumer is waiting on never blocks (deadlock-free).
        reorderSpaceCondition.wait(lock, [&] {
            return allCalculationsDone || idx < currentProcessPosition + reorderCapacity;
        });
        const size_t slot = idx % reorderCapacity;
        reorderSlots[slot] = std::move(clusterResult);   // O(1) vector move, no heap sift
        reorderFilled[slot] = 1;
        reorderBufferedCount++;
        shouldNotifyClusterThread = (idx == currentProcessPosition);
    }
    if (shouldNotifyClusterThread) {
        clusterCondition.notify_one();
    }
}

static float parsePrecisionLib(const std::string &scoreFile, double targetSeqid, double targetCov, double targetPrecision) {
    std::stringstream in(scoreFile);
    std::string line;
    int intTargetSeqid = static_cast<int>((targetSeqid + 0.0001) * 100);
    int seqIdRest = (intTargetSeqid % 5);
    targetSeqid = static_cast<float>(intTargetSeqid - seqIdRest) / 100;
    targetCov = static_cast<float>(static_cast<int>((targetCov + 0.0001) * 10)) / 10;
    
    while (std::getline(in, line)) {
        std::vector<std::string> values = Util::split(line, " ");
        float cov = strtod(values[0].c_str(), NULL);
        float seqid = strtod(values[1].c_str(), NULL);
        float scorePerCol = strtod(values[2].c_str(), NULL);
        float precision = strtod(values[3].c_str(), NULL);
        if (MathUtil::AreSame(cov, targetCov) && MathUtil::AreSame(seqid, targetSeqid) && precision >= targetPrecision) {
            return scorePerCol;
        }
    }
    
    Debug(Debug::WARNING) << "Can not find any score per column for coverage "
                          << targetCov << " and sequence identity " << targetSeqid 
                          << ". No hit will be filtered.\n";
    return 0;
}

static void writeClustering(DBWriter *dbWriter, const std::pair<DBKeyType, DBKeyType> * results, size_t dbSize) {
    std::string resultString;
    resultString.reserve(1024 * 1024 * 1024);
    char buffer[32];
    DBKeyType previousRepresentativeKey = DB_KEY_INVALID;
    
    for (size_t i = 0; i < dbSize; i++) {
        DBKeyType currentRepresentativeKey = results[i].first;
        
        if (previousRepresentativeKey != currentRepresentativeKey) {
            if (previousRepresentativeKey != DB_KEY_INVALID) {
                dbWriter->writeData(resultString.c_str(), resultString.length(), previousRepresentativeKey);
            }
            resultString.clear();
            char *outPos = Itoa::u64toa_sse2(static_cast<uint64_t>(currentRepresentativeKey), buffer);
            resultString.append(buffer, (outPos - buffer - 1));
            resultString.push_back('\n');
        }
        
        DBKeyType memberKey = results[i].second;
        if (memberKey != currentRepresentativeKey) {
            char *outPos = Itoa::u64toa_sse2(static_cast<uint64_t>(memberKey), buffer);
            resultString.append(buffer, (outPos - buffer - 1));
            resultString.push_back('\n');
        }
        
        previousRepresentativeKey = currentRepresentativeKey;
    }
    
    if (previousRepresentativeKey != DB_KEY_INVALID) {
        dbWriter->writeData(resultString.c_str(), resultString.length(), previousRepresentativeKey);
    }
}

// the minimum ungapped score per column of a pair that is aligned with gaps
static float getScorePerColThreshold(const Parameters &par) {
    std::string libraryString = (par.covMode == Parameters::COV_MODE_BIDIRECTIONAL)
                                    ? getCovSeqidQscPercMinDiag()
                                    : getCovSeqidQscPercMinDiagTargetCov();
    return parsePrecisionLib(libraryString, par.seqIdThr, par.covThr, 0.98);
}

struct AlignmentScoring {
    BaseMatrix *subMat;
    SubstitutionMatrix::FastMatrix fastMatrix;
    float scorePerColThreshold;
    EvalueComputation evaluer;

    AlignmentScoring(const Parameters &par, size_t dbResidues)
        : subMat(new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, 0.0)), fastMatrix(SubstitutionMatrix::createAsciiSubMat(*subMat)),
          scorePerColThreshold(getScorePerColThreshold(par)),
          evaluer(dbResidues, subMat, par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid()) {}

    ~AlignmentScoring() {
        delete[] fastMatrix.matrix;
        delete[] fastMatrix.matrixData;
        delete subMat;
    }
};

// false if no diagonal can reach the required overlap
static bool canBeCoveredOnDiagonals(const Parameters &par, int queryLength, int targetLength, int &minDiag, int &maxDiag) {
    int diagBand = std::max(queryLength, targetLength) + targetLength;
    if (par.covMode == Parameters::COV_MODE_BIDIRECTIONAL) {
        diagBand = static_cast<int>((1.0f - par.covThr) * std::max(queryLength, targetLength) + (1.0f - par.seqIdThr) * (queryLength + targetLength)) + 1;
    }
    const int minOverlap = static_cast<int>(std::ceil(Util::computeMinOverlap(Parameters::CLUST_LINEAR_DIAG_BAND_COV_SCALE * par.covThr, par.covMode,
                                                                            static_cast<float>(queryLength), static_cast<float>(targetLength))));
    minDiag = std::max(-diagBand, minOverlap - targetLength);
    maxDiag = std::min(diagBand, queryLength - minOverlap);
    return minDiag <= maxDiag;
}

// the keys of the other members of the cluster of targetKey in --filter-cludb-file
static void getClusterMembers(DBReader<DBKeyType> *cluDbr, DBKeyType targetKey, unsigned int threadIdx, std::vector<DBKeyType> &memberKeys) {
    memberKeys.clear();
    char buffer[1024];
    for (char *cluData = cluDbr->getData(cluDbr->getId(targetKey), threadIdx); *cluData != '\0'; cluData = Util::skipLine(cluData)) {
        Util::parseKey(cluData, buffer);
        const DBKeyType key = Util::fast_atoi<DBKeyType>(buffer);
        if (key != targetKey) {
            memberKeys.push_back(key);
        }
    }
}

// member gapped alignments in SWIPE lanes, the lane slots they took and the ones aligned one pair at a time
static std::atomic<size_t> swipeLanePairs(0);
static std::atomic<size_t> swipeLaneSlots(0);
static std::atomic<size_t> swipeSinglePairs(0);

static void printSwipeLanes() {
    Debug(Debug::INFO) << "SWIPE member gapped alignments: " << swipeLanePairs.load() << " in lanes ("
                       << swipeLanePairs.load() * 100 / std::max<size_t>(1, swipeLaneSlots.load()) << "% lane fill), "
                       << swipeSinglePairs.load() << " one pair at a time\n";
}

// per thread buffers of --aligner-mode 1-3: the cluster members of a rep that go through the SWIPE lanes
struct SwipeBatch {
    SwipeAligner aligner;
    EvalueComputation &evaluer;
    bool initialized;
    // 8 bit (0) or 16 bit (1) lanes first for the members of the current rep
    int memberWord;
    std::vector<unsigned char> sequences;
    std::vector<size_t> offsets;
    std::vector<const unsigned char *> targets;
    std::vector<int> lengths;
    std::vector<int> laneLengths;
    std::vector<int> diagonals;
    std::vector<int> bandWidths;
    std::vector<int> columnEnds;
    std::vector<int> minScores;
    std::vector<s_align> aligns;
    std::vector<std::string> backtraces;
    std::vector<size_t> entries;
    // alignClusterMembers
    std::vector<size_t> gapped;
    size_t lanePairs;
    size_t laneSlots;
    size_t singlePairs;

    SwipeBatch(const Parameters &par, BaseMatrix *subMat, EvalueComputation &evaluer, size_t maxSeqLen)
        : aligner(subMat, maxSeqLen, par.compBiasCorrection, par.compBiasCorrectionScale, par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid()),
          evaluer(evaluer), initialized(false), memberWord(1), lanePairs(0), laneSlots(0), singlePairs(0) {}

    ~SwipeBatch() {
        swipeLanePairs += lanePairs;
        swipeLaneSlots += laneSlots;
        swipeSinglePairs += singlePairs;
    }

    // the SIMD lanes of the 8 bit (word 0) or 16 bit lanes the members start with
    size_t lanes() const {
        return (memberWord == 0) ? VECSIZE_INT * 4 : VECSIZE_INT * 2;
    }

    void countLanes(size_t count) {
        lanePairs += count;
        laneSlots += (count + lanes() - 1) / lanes() * lanes();
    }

    // most members of a cluster are similar enough to saturate 8 bit
    void startRep() {
        initialized = false;
        memberWord = 1;
    }

    void clear() {
        sequences.clear();
        offsets.clear();
        lengths.clear();
    }

    size_t add(const Sequence &sequence) {
        offsets.push_back(sequences.size());
        sequences.insert(sequences.end(), sequence.numSequence, sequence.numSequence + sequence.L);
        lengths.push_back(sequence.L);
        return offsets.size() - 1;
    }

    // the entries in order, with the lanes starting at word
    void select(size_t count, int word) {
        targets.resize(count);
        diagonals.resize(count);
        bandWidths.assign(count, 0);
        aligns.resize(count);
        for (size_t i = 0; i < count; i++) {
            targets[i] = sequences.data() + offsets[entries[i]];
            aligns[i].word = word;
        }
    }

    void setQuery(Sequence &query) {
        if (initialized == false) {
            aligner.initQuery(&query);
            initialized = true;
        }
    }

    // 16 bit lanes first once most 8 bit lanes of the rep saturate, 8 bit again once all 16 bit scores fit
    void updateWord(int &word, size_t count) {
        size_t wide = 0;
        size_t fitting = 0;
        for (size_t i = 0; i < count; i++) {
            wide += (aligns[i].word >= 1);
            fitting += (aligns[i].score1 < static_cast<uint32_t>(INT8_MAX));
        }
        if (word == 0 && 2 * wide > count) {
            word = 1;
        } else if (word == 1 && fitting == count) {
            word = 0;
        }
    }
};

// a member gate of a target: its members in the order they are checked, the next of them, its member alignments in member order
// and whether no member failed so far
struct MemberGate {
    size_t targetId;
    short startDiagonal;
    std::vector<DBKeyType> memberKeys;
    std::vector<size_t> order;
    std::vector<int> lengths;
    std::vector<std::string> records;
    size_t next;
    size_t batch;
    bool live;
};

// per thread buffers of the member gates; with SWIPE (--aligner-mode 1-3) the gates of all targets of a rep can wait and go in rounds
// that share the lanes (deferGates)
struct ClusterMembers {
    std::vector<MemberGate> gates;
    size_t gateCount;
    std::vector<size_t> poolGates;
    std::vector<size_t> lanePool;
    SwipeBatch *swipe;
    bool deferGates;

    ClusterMembers(SwipeBatch *swipe, bool deferGates) : gateCount(0), swipe(swipe), deferGates(deferGates) {}

    MemberGate &addGate(size_t targetId, short startDiagonal) {
        if (gateCount == gates.size()) {
            gates.emplace_back();
        }
        MemberGate &gate = gates[gateCount++];
        gate.targetId = targetId;
        gate.startDiagonal = startDiagonal;
        gate.memberKeys.clear();
        return gate;
    }
};

// the result of a SWIPE alignment with start, end, identities and backtrace
static Matcher::result_t swipeResult(const Parameters &par, EvalueComputation &evaluer, DBKeyType key, int rawScore, const s_align &alignment,
                                     int queryLength, int targetLength, const std::string &backtrace) {
    const float seqId = Util::computeSeqId(par.seqIdMode, alignment.identicalAACnt, queryLength, targetLength, backtrace.size());
    return Matcher::result_t(key, static_cast<int>(evaluer.computeBitScore(rawScore) + 0.5),
                             SmithWaterman::computeCov(alignment.qStartPos1, alignment.qEndPos1, queryLength),
                             SmithWaterman::computeCov(alignment.dbStartPos1, alignment.dbEndPos1, targetLength), seqId,
                             evaluer.computeEvalue(rawScore, queryLength), backtrace.size(), alignment.qStartPos1, alignment.qEndPos1, queryLength,
                             alignment.dbStartPos1, alignment.dbEndPos1, targetLength, backtrace);
}

// whether a forward score and end can still meet the e-value and coverage of the clustering criteria
static bool swipeForwardPasses(const Parameters &par, EvalueComputation &evaluer, const s_align &alignment, int queryLength, int targetLength) {
    if (alignment.dbEndPos1 < 0 || alignment.qEndPos1 < 0) {
        return false;
    }
    const float qCov = SmithWaterman::computeCov(0, alignment.qEndPos1, queryLength);
    const float tCov = SmithWaterman::computeCov(0, alignment.dbEndPos1, targetLength);
    return Util::hasCoverage(par.covThr, par.covMode, qCov, tCov) && evaluer.computeEvalue(alignment.score1, queryLength) <= par.evalThr;
}

// the gapped alignments of the members of a round of alignClusterMembers, of all its gates; the members of a gate whose batch fills
// the 16 bit lanes go through the lanes together, the others one pair at a time, which only depends on the member order of the gate;
// a failing member closes its gate
static void alignMembersSwipe(const Parameters &par, Matcher &matcher, bool &matcherInitialized, ClusterMembers &members, Sequence &query, Sequence &element,
                              DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer) {
    SwipeBatch &swipe = *members.swipe;
    if (matcherInitialized == false) {
        matcher.initQuery(&query);
        matcherInitialized = true;
    }
    std::vector<size_t> &lanePool = members.lanePool;
    lanePool.assign(swipe.gapped.size(), SIZE_MAX);
    swipe.entries.clear();
    for (size_t g = 0; g < swipe.gapped.size(); g++) {
        if (members.gates[members.poolGates[g]].batch >= VECSIZE_INT * 2) {
            lanePool[g] = swipe.entries.size();
            swipe.entries.push_back(g);
        }
    }
    const size_t laneCount = swipe.entries.size();
    swipe.singlePairs += swipe.gapped.size() - laneCount;
    if (laneCount > 0) {
        swipe.countLanes(laneCount);
        swipe.select(laneCount, swipe.memberWord);
        swipe.laneLengths.resize(laneCount);
        for (size_t l = 0; l < laneCount; l++) {
            swipe.laneLengths[l] = swipe.lengths[swipe.entries[l]];
        }
        swipe.setQuery(query);
        swipe.aligner.alignForward(laneCount, swipe.targets.data(), swipe.laneLengths.data(), swipe.aligns.data());
        swipe.updateWord(swipe.memberWord, laneCount);
        for (size_t l = 0; l < laneCount; l++) {
            const s_align &alignment = swipe.aligns[l];
            if (alignment.word < 2 && swipeForwardPasses(par, swipe.evaluer, alignment, query.L, swipe.laneLengths[l]) == false) {
                members.gates[members.poolGates[swipe.entries[l]]].live = false;
            }
        }
        if (par.alignerMode == Parameters::ALIGNER_MODE_SWIPE) {
            swipe.aligner.alignBackward(laneCount, swipe.targets.data(), swipe.aligns.data());
        } else if (par.alignerMode == Parameters::ALIGNER_MODE_SWIPE_TRACEBACK) {
            swipe.backtraces.resize(laneCount);
            swipe.columnEnds.resize(laneCount);
            swipe.minScores.resize(laneCount);
            for (size_t l = 0; l < laneCount; l++) {
                swipe.diagonals[l] = swipe.aligns[l].dbEndPos1 - swipe.aligns[l].qEndPos1;
                swipe.columnEnds[l] = swipe.aligns[l].dbEndPos1 + 1;
                swipe.minScores[l] = swipe.aligns[l].score1;
            }
            swipe.aligner.alignBanded(laneCount, swipe.targets.data(), swipe.laneLengths.data(), swipe.diagonals.data(), swipe.bandWidths.data(),
                                      swipe.aligns.data(), swipe.backtraces.data(), swipe.columnEnds.data(), swipe.minScores.data());
        }
    }
    for (size_t g = 0; g < swipe.gapped.size(); g++) {
        MemberGate &gate = members.gates[members.poolGates[g]];
        if (gate.live == false) {
            continue;
        }
        const size_t i = swipe.gapped[g];
        const DBKeyType elementKey = gate.memberKeys[i];
        const size_t elementId = cluSeqDbr->getId(elementKey);
        element.mapSequence(elementId, elementKey, cluSeqDbr->getData(elementId, threadIdx), cluSeqDbr->getSeqLen(elementId));
        const size_t l = lanePool[g];
        Matcher::result_t res_element;
        if (l == SIZE_MAX || swipe.aligns[l].word >= 2) {
            res_element = matcher.getSWResult(&element, 0, false, par.covMode, par.covThr, par.evalThr, swMode, par.seqIdMode, false);
        } else if (par.alignerMode == Parameters::ALIGNER_MODE_SWIPE_TRACEBACK) {
            res_element = swipeResult(par, swipe.evaluer, elementKey, swipe.minScores[l], swipe.aligns[l], query.L, element.L, swipe.backtraces[l]);
        } else {
            res_element = matcher.getSWResult(&element, 0, false, par.covMode, par.covThr, par.evalThr, swMode, par.seqIdMode, false, false, &swipe.aligns[l]);
        }
        if (Alignment::checkCriteria(res_element, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr) == false) {
            gate.live = false;
            continue;
        }
        if (withRecords) {
            size_t len = Matcher::resultToBuffer(alnLineBuffer, res_element, par.addBacktrace);
            gate.records[i].assign(alnLineBuffer, len);
        }
    }
}

// whether every --filter-cludb-file member of the cluster of the target of each gate meets the clustering criteria against the query
// (MemberGate::live); the members that cannot be covered fail first and the most different lengths go next, with the decision of the
// member order; with SWIPE (--aligner-mode 1-3) the gates go in rounds of 1, 2, 4, ... members each, and the gapped alignments of a
// round of all gates share the lanes once they fill them
static void alignClusterMembers(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                                DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer,
                                ClusterMembers &members) {
    const char *querySequence = query.getSeqData();
    SwipeBatch *swipe = members.swipe;
    const int queryLength = query.L;
    for (size_t g = 0; g < members.gateCount; g++) {
        MemberGate &gate = members.gates[g];
        gate.next = 0;
        gate.live = true;
        gate.order.resize(gate.memberKeys.size());
        gate.lengths.resize(gate.memberKeys.size());
        for (size_t i = 0; i < gate.memberKeys.size(); i++) {
            gate.order[i] = i;
            gate.lengths[i] = static_cast<int>(cluSeqDbr->getSeqLen(cluSeqDbr->getId(gate.memberKeys[i])));
            int minDiag, maxDiag;
            if (Util::canBeCovered(par.covThr, par.covMode, query.L, gate.lengths[i]) == false
                || (par.diagonalMode == Parameters::DIAGONAL_MODE_UNGAPPED && canBeCoveredOnDiagonals(par, query.L, gate.lengths[i], minDiag, maxDiag) == false)) {
                gate.live = false;
                break;
            }
        }
        if (gate.live == false) {
            continue;
        }
        const std::vector<int> &memberLengths = gate.lengths;
        std::stable_sort(gate.order.begin(), gate.order.end(), [&memberLengths, queryLength](size_t first, size_t second) {
            return std::abs(memberLengths[first] - queryLength) > std::abs(memberLengths[second] - queryLength);
        });
        if (withRecords) {
            gate.records.assign(gate.memberKeys.size(), std::string());
        }
    }
    for (size_t batchSize = 1; ; batchSize *= (swipe != NULL) ? 2 : 1) {
        bool open = false;
        if (swipe != NULL) {
            swipe->gapped.clear();
            swipe->clear();
            members.poolGates.clear();
        }
        for (size_t g = 0; g < members.gateCount; g++) {
            MemberGate &gate = members.gates[g];
            if (gate.live == false || gate.next == gate.memberKeys.size()) {
                continue;
            }
            open = true;
            const size_t batchEnd = std::min(gate.memberKeys.size(), gate.next + batchSize);
            gate.batch = batchEnd - gate.next;
            for (size_t o = gate.next; o < batchEnd && gate.live; o++) {
                const size_t i = gate.order[o];
                const DBKeyType elementKey = gate.memberKeys[i];
                const size_t elementId = cluSeqDbr->getId(elementKey);
                char *elementSequence = cluSeqDbr->getData(elementId, threadIdx);
                size_t elementLength = cluSeqDbr->getSeqLen(elementId);
                short elementDiagonal = gate.startDiagonal;

                // 1. ungapped alignment
                element.mapSequence(elementId, elementKey, elementSequence, elementLength);
                if (Util::canBeCovered(par.covThr, par.covMode, query.L, element.L) == false) {
                    gate.live = false;
                    break;
                }
                if (par.diagonalMode == Parameters::DIAGONAL_MODE_UNGAPPED) {
                    int minDiag, maxDiag;
                    if (canBeCoveredOnDiagonals(par, query.L, element.L, minDiag, maxDiag) == false) {
                        gate.live = false;
                        break;
                    }
                    elementDiagonal = blockAligner.bestDiagonal(&element, minDiag, maxDiag);
                }
                BlockAligner::UngappedAln_res elementUngappedAlignment = blockAligner.ungappedAlign(&element, elementDiagonal);

                // 2. check the criteria
                bool elementHasEvalue = (elementUngappedAlignment.eval <= par.evalThr);
                bool elementHasAlnLen = (elementUngappedAlignment.alnLen >= par.alnLenThr);
                bool elementHasCoverage = Util::hasCoverage(par.covThr, par.covMode, elementUngappedAlignment.qcov, elementUngappedAlignment.tcov);
                int elementIdenticalCount = 0;
                for (int q = elementUngappedAlignment.qStart; q <= elementUngappedAlignment.qEnd; q++) {
                    char queryLetter = querySequence[q] & static_cast<unsigned char>(~0x20);
                    char elementLetter = elementSequence[elementUngappedAlignment.tStart + (q - elementUngappedAlignment.qStart)] & static_cast<unsigned char>(~0x20);
                    elementIdenticalCount += (queryLetter == elementLetter) ? 1 : 0;
                }
                float elementSeqId = Util::computeSeqId(par.seqIdMode, elementIdenticalCount, query.L, elementLength, elementUngappedAlignment.alnLen);
                bool elementHasSeqId = elementSeqId >= (par.seqIdThr - std::numeric_limits<float>::epsilon());

                if (!(elementHasAlnLen && elementHasCoverage && elementHasSeqId && elementHasEvalue)) {
                    // 3. gapped alignment, after the round with swipe
                    if (swipe != NULL) {
                        swipe->add(element);
                        swipe->gapped.push_back(i);
                        members.poolGates.push_back(g);
                        continue;
                    }
                    if (matcherInitialized == false) {
                        matcher.initQuery(&query);
                        matcherInitialized = true;
                    }
                    Matcher::result_t res_element = matcher.getSWResult(&element, static_cast<int>(elementDiagonal), false, par.covMode, par.covThr, par.evalThr,
                                                                        swMode, par.seqIdMode, false);
                    if (Alignment::checkCriteria(res_element, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr) == false) {
                        gate.live = false;
                        break;
                    }
                    // stage member alignment (flushed only if all members pass)
                    if (withRecords) {
                        appendAlignmentResult(gate.records[i], alnLineBuffer, res_element, par.addBacktrace);
                    }
                } else if (withRecords) {
                    // member passed ungapped: gap-free (all 'M') record
                    std::string elementBacktrace = par.addBacktrace ? std::string(elementUngappedAlignment.alnLen, 'M') : std::string();
                    Matcher::result_t elementResult(elementKey, elementUngappedAlignment.score, elementUngappedAlignment.qcov,
                        elementUngappedAlignment.tcov, elementSeqId, elementUngappedAlignment.eval, elementUngappedAlignment.alnLen,
                        elementUngappedAlignment.qStart, elementUngappedAlignment.qEnd, query.L,
                        elementUngappedAlignment.tStart, elementUngappedAlignment.tEnd, elementLength, elementBacktrace);
                    appendAlignmentResult(gate.records[i], alnLineBuffer, elementResult, par.addBacktrace);
                }
            }
            gate.next = batchEnd;
        }
        if (open == false) {
            break;
        }
        if (swipe != NULL && swipe->gapped.empty() == false) {
            alignMembersSwipe(par, matcher, matcherInitialized, members, query, element, cluSeqDbr, swMode, threadIdx, withRecords, alnLineBuffer);
        }
    }
}

// the member gate of the target of alignTarget: checked now, or queued for the rounds of the rep and passed for now with deferGates
static bool alignTargetMembers(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                               DBReader<DBKeyType> *cluDbr, DBReader<DBKeyType> *cluSeqDbr, DBKeyType targetKey, size_t targetId, short startDiagonal,
                               unsigned int swMode, unsigned int threadIdx, std::string *pendingMemberAln, char *alnLineBuffer, ClusterMembers &members) {
    if (members.deferGates == false) {
        members.gateCount = 0;
    }
    MemberGate &gate = members.addGate(targetId, startDiagonal);
    getClusterMembers(cluDbr, targetKey, threadIdx, gate.memberKeys);
    if (pendingMemberAln != NULL) {
        pendingMemberAln->clear();
    }
    if (members.deferGates) {
        return true;
    }
    alignClusterMembers(par, blockAligner, matcher, matcherInitialized, query, element, cluSeqDbr, swMode, threadIdx, pendingMemberAln != NULL, alnLineBuffer, members);
    if (gate.live && pendingMemberAln != NULL) {
        for (size_t r = 0; r < gate.records.size(); r++) {
            *pendingMemberAln += gate.records[r];
        }
    }
    return gate.live;
}

// whether the target meets the clustering criteria against the query of blockAligner, with its --filter-cludb-file members if cluDbr is given
static bool alignTarget(const Parameters &par, BlockAligner &blockAligner, bool &blockAlignerInitialized, Matcher *matcher, bool &matcherInitialized,
                        Sequence &query, Sequence &target, Sequence *element, DBReader<DBKeyType> *cluDbr, DBReader<DBKeyType> *cluSeqDbr,
                        float scorePerColThreshold, unsigned int swMode, unsigned short diagonal, unsigned int threadIdx,
                        const ClusterAssignment *assignedCluster, size_t targetId,
                        std::string *alnResultBuffer, std::string *pendingMemberAln, char *alnLineBuffer, ClusterMembers *members) {
    const char *querySequence = query.getSeqData();
    const char *targetSequence = target.getSeqData();
    if (Util::canBeCovered(par.covThr, par.covMode, query.L, target.L) == false) {
        return false;
    }

    if (par.diagonalMode == Parameters::DIAGONAL_MODE_UNGAPPED) {
        int minDiag, maxDiag;
        if (canBeCoveredOnDiagonals(par, query.L, target.L, minDiag, maxDiag) == false) {
            return false;
        }
        diagonal = blockAligner.bestDiagonal(&target, minDiag, maxDiag);
    }
    BlockAligner::UngappedAln_res ungappedAlignment = blockAligner.ungappedAlign(&target, diagonal);

    bool hasEvalue = (ungappedAlignment.eval <= par.evalThr);
    bool hasAlnLen = (ungappedAlignment.alnLen >= par.alnLenThr);
    bool hasCoverage = Util::hasCoverage(par.covThr, par.covMode, ungappedAlignment.qcov, ungappedAlignment.tcov);
    float seqId = 0;

    if (hasEvalue) {
        int identicalCount = 0;
        for (int q = ungappedAlignment.qStart; q <= ungappedAlignment.qEnd; q++) {
            char queryLetter = querySequence[q] & static_cast<unsigned char>(~0x20);
            char targetLetter = targetSequence[ungappedAlignment.tStart + (q - ungappedAlignment.qStart)] & static_cast<unsigned char>(~0x20);
            identicalCount += (queryLetter == targetLetter) ? 1 : 0;
        }
        seqId = Util::computeSeqId(par.seqIdMode, identicalCount, query.L, target.L, ungappedAlignment.alnLen);
    }

    bool hasSeqId = seqId >= (par.seqIdThr - std::numeric_limits<float>::epsilon());
    const auto isAssigned = [&]() {
        return assignedCluster != NULL && loadAssignedCluster(assignedCluster, targetId) != DB_LOCAL_ID_INVALID;
    };
    if (isAssigned()) {
        return false;
    }

    if (hasAlnLen && hasCoverage && hasSeqId && hasEvalue) {
        if (isAssigned()) {
            return false;
        }
        if (cluDbr != NULL && alignTargetMembers(par, blockAligner, *matcher, matcherInitialized, query, *element, cluDbr, cluSeqDbr, target.getDbKey(), targetId,
                                                 diagonal, swMode, threadIdx, pendingMemberAln, alnLineBuffer, *members) == false) {
            return false;
        }
        if (alnResultBuffer != NULL) {
            std::string backtrace = par.addBacktrace ? std::string(ungappedAlignment.alnLen, 'M') : std::string();
            Matcher::result_t ungappedResult(target.getDbKey(), ungappedAlignment.score, ungappedAlignment.qcov,
                ungappedAlignment.tcov, seqId, ungappedAlignment.eval, ungappedAlignment.alnLen,
                ungappedAlignment.qStart, ungappedAlignment.qEnd, query.L,
                ungappedAlignment.tStart, ungappedAlignment.tEnd, target.L, backtrace);
            appendAlignmentResult(*alnResultBuffer, alnLineBuffer, ungappedResult, par.addBacktrace);
            // flush staged member alignments (empty unless filter gate ran)
            *alnResultBuffer += *pendingMemberAln;
        }
        return true;
    }

    float currentScorePerCol = static_cast<float>(ungappedAlignment.score) / static_cast<float>(ungappedAlignment.diagonalLen);
    if (currentScorePerCol < scorePerColThreshold) {
        return false;
    }

    int alignmentLength = ungappedAlignment.alnLen;
    int queryStartPos = ungappedAlignment.qStart;
    int targetStartPos = ungappedAlignment.tStart;
    int newQueryStartPos = queryStartPos;
    int newTargetStartPos = targetStartPos;

    if (queryStartPos == -1 || targetStartPos == -1 || alignmentLength < 3) {
        return false;
    }

    if (isAssigned()) {
        return false;
    }

    bool foundConsecutiveMatchSeed = false;
    for (int blockIdx = 0; blockIdx <= alignmentLength - 3; ++blockIdx) {
        int queryPos = queryStartPos + blockIdx;
        int targetPos = targetStartPos + blockIdx;

        if (querySequence[queryPos] == targetSequence[targetPos] &&
            querySequence[queryPos + 1] == targetSequence[targetPos + 1] &&
            querySequence[queryPos + 2] == targetSequence[targetPos + 2]) {
            newQueryStartPos = queryPos + 1;
            newTargetStartPos = targetPos + 1;
            foundConsecutiveMatchSeed = true;
            break;
        }
    }
    if (foundConsecutiveMatchSeed == false) {
        return false;
    }

    std::string gappedBacktrace;
    if (blockAlignerInitialized == false) {
        blockAligner.initQuery(&query);
        blockAlignerInitialized = true;
    }
    const int32_t xDrop = (MIN_SIZE * par.gapExtend.values.aminoacid() + par.gapOpen.values.aminoacid());
    s_align gappedAlignment = blockAligner.bandedalign(&target, newQueryStartPos, newTargetStartPos,
                                                       gappedBacktrace, xDrop, par.covThr, par.covMode);
    // bandedalign fails with evalue < 0 and an empty backtrace, whose length the default seq-id mode divides by
    if (gappedAlignment.evalue < 0 || gappedBacktrace.empty()) {
        return false;
    }
    unsigned int gappedAlnLength = gappedBacktrace.size();
    double gappedSeqId = Util::computeSeqId(par.seqIdMode, gappedAlignment.identicalAACnt,
                                           query.L, target.L, gappedAlnLength);
    Matcher::result_t result = Matcher::result_t(
        target.getDbKey(), gappedAlignment.score1, gappedAlignment.qCov, gappedAlignment.tCov,
        gappedSeqId, gappedAlignment.evalue, gappedAlnLength,
        gappedAlignment.qStartPos1, gappedAlignment.qEndPos1, query.L,
        gappedAlignment.dbStartPos1, gappedAlignment.dbEndPos1, target.L, gappedBacktrace
    );
    if (Alignment::checkCriteria(result, false, par.evalThr, par.seqIdThr,
                                 par.alnLenThr, par.covMode, par.covThr) == false) {
        return false;
    }
    if (isAssigned()) {
        return false;
    }
    if (cluDbr != NULL && alignTargetMembers(par, blockAligner, *matcher, matcherInitialized, query, *element, cluDbr, cluSeqDbr, target.getDbKey(), targetId,
                                             0, swMode, threadIdx, pendingMemberAln, alnLineBuffer, *members) == false) {
        return false;
    }
    if (alnResultBuffer != NULL) {
        appendAlignmentResult(*alnResultBuffer, alnLineBuffer, result, par.addBacktrace);
        // flush staged member alignments (empty unless filter gate ran)
        *alnResultBuffer += *pendingMemberAln;
    }
    return true;
}

static void (*clusterThreadFunc)(ClusterAssignment*) = nullptr;

void clusterThreadFuncSetcover(ClusterAssignment* assignedCluster) {
    while (true) {
        std::unique_lock<std::mutex> lock(clusterMutex);
        
        clusterCondition.wait(lock, [] {
            return reorderFilled[currentProcessPosition % reorderCapacity] != 0 ||
                   allCalculationsDone;
        });

        // 1) reorder buffer → setCoverCandidates
        while (reorderFilled[currentProcessPosition % reorderCapacity] != 0) {
            const size_t slot = currentProcessPosition % reorderCapacity;
            ClusterResult result = std::move(reorderSlots[slot]);
            reorderFilled[slot] = 0;
            reorderBufferedCount--;
            currentProcessPosition++;
            reorderSpaceCondition.notify_all();
            currentPrefSize = result.prefSize;

            if (result.memberIds.size() > 1) {
                SetCoverCandidate candidate;
                candidate.representativeId = result.representativeId;
                candidate.memberCount = result.memberIds.size();
                candidate.memberOffset = setCoverMemberPool.size();
                setCoverMemberPool.insert(setCoverMemberPool.end(),
                                          result.memberIds.begin(), result.memberIds.end());
                setCoverLiveMemberCount += candidate.memberCount;
                setCoverCandidates.push_back(candidate);
                std::push_heap(setCoverCandidates.begin(), setCoverCandidates.end(), setCoverComparator);
            }
        }

        // 2) assign candidates guaranteed to be the currently largest set
        while (setCoverCandidates.empty() == false &&
               (allCalculationsDone ||
                setCoverCandidates.front().memberCount > currentPrefSize)) {

            std::pop_heap(setCoverCandidates.begin(), setCoverCandidates.end(), setCoverComparator);
            SetCoverCandidate candidate = setCoverCandidates.back();
            setCoverCandidates.pop_back();
            setCoverLiveMemberCount -= candidate.memberCount;

            if (loadAssignedCluster(assignedCluster, candidate.representativeId) != DB_LOCAL_ID_INVALID) {
                continue;
            }

            // Drop already-assigned members, compacting the survivors in place
            // within the pool region [memberOffset, memberOffset + memberCount).
            DBLocalId *members = setCoverMemberPool.data() + candidate.memberOffset;
            size_t validCount = 0;
            for (size_t i = 0; i < candidate.memberCount; i++) {
                if (loadAssignedCluster(assignedCluster, members[i]) == DB_LOCAL_ID_INVALID) {
                    members[validCount++] = members[i];
                }
            }

            if (validCount <= 1) {
                continue;
            }

            if (validCount != candidate.memberCount) {
                candidate.memberCount = validCount;
                setCoverLiveMemberCount += validCount;
                setCoverCandidates.push_back(candidate);
                std::push_heap(setCoverCandidates.begin(), setCoverCandidates.end(), setCoverComparator);
                continue;
            }

            for (size_t i = 0; i < candidate.memberCount; i++) {
                storeAssignedCluster(assignedCluster, members[i], candidate.representativeId);
            }
        }

        lock.unlock();
        compactSetCoverMemberPool(setCoverCandidates, setCoverMemberPool, setCoverLiveMemberCount);
        lock.lock();

        if (allCalculationsDone &&
            reorderBufferedCount == 0 &&
            setCoverCandidates.empty()) {
            break;
        }
    }
}

void clusterThreadFuncGreedy(ClusterAssignment* assignedCluster) {
    while (true) {
        std::unique_lock<std::mutex> lock(clusterMutex);
        
        clusterCondition.wait(lock, [] {
            return reorderFilled[currentProcessPosition % reorderCapacity] != 0 ||
                   allCalculationsDone;
        });

        if (allCalculationsDone && reorderBufferedCount == 0) {
            break;
        }

        while (reorderFilled[currentProcessPosition % reorderCapacity] != 0) {
            const size_t slot = currentProcessPosition % reorderCapacity;
            ClusterResult result = std::move(reorderSlots[slot]);
            reorderFilled[slot] = 0;
            reorderBufferedCount--;
            currentProcessPosition++;
            reorderSpaceCondition.notify_all();

            if (loadAssignedCluster(assignedCluster, result.representativeId) != DB_LOCAL_ID_INVALID) {
                continue;
            }

            std::vector<DBLocalId> validMemberIds;
            validMemberIds.reserve(result.memberIds.size());
            for (DBLocalId memberId : result.memberIds) {
                if (loadAssignedCluster(assignedCluster, memberId) == DB_LOCAL_ID_INVALID) {
                    validMemberIds.push_back(memberId);
                }
            }

            if (validMemberIds.size() <= 1) {
                continue;
            }

            for (DBLocalId memberId : validMemberIds) {
                storeAssignedCluster(assignedCluster, memberId, result.representativeId);
            }
        }
    }
}

int doAlign2clust(Parameters &par, DBWriter &resultWriter, DBReader<DBKeyType> &alnDbr, DBWriter *alnWriter) {
    DBReader<DBKeyType> *seqDbr = new DBReader<DBKeyType>(
        par.db1.c_str(), par.db1Index.c_str(), par.threads, 
        DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX
    );
    seqDbr->open(DBReader<DBKeyType>::NOSORT);
 
    DBReader<DBKeyType> *cluDbr = nullptr;
    DBReader<DBKeyType> *cluSeqDbr = nullptr;
    if (par.filterCluDBFile.empty() == false) {
        std::string cluIndex = par.filterCluDBFile + ".index";
        cluDbr = new DBReader<DBKeyType>(
            par.filterCluDBFile.c_str(), cluIndex.c_str(), par.threads, 
            DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX
        );
        cluDbr->open(DBReader<DBKeyType>::LINEAR_ACCCESS);

        std::string cluSeqIndex = par.filterSeqDBFile + ".index";
        cluSeqDbr = new DBReader<DBKeyType>(
            par.filterSeqDBFile.c_str(), cluSeqIndex.c_str(), par.threads, 
            DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX
        );
            
        cluSeqDbr->open(DBReader<DBKeyType>::LINEAR_ACCCESS);
    }

    const size_t dbSize = seqDbr->getSize();

    AlignmentScoring scoring(par, seqDbr->getAminoAcidDBSize());
    Debug(Debug::INFO) << "Score per column threshold for filtering: " << scoring.scorePerColThreshold << "\n";

    ClusterAssignment *assignedCluster = new(std::nothrow) ClusterAssignment[dbSize];
    Util::checkAllocation(assignedCluster, "Can not allocate assignedCluster memory in Align2Clust");
    for (size_t i = 0; i < dbSize; ++i) {
        storeAssignedCluster(assignedCluster, i, DB_LOCAL_ID_INVALID);
    }

    int mode = par.clusteringMode;
    
    if (mode == Parameters::SET_COVER) {
        clusterThreadFunc = clusterThreadFuncSetcover;
        Debug(Debug::INFO) << "Using SET_COVER clustering mode\n";
    } else if (mode == Parameters::GREEDY || mode == Parameters::GREEDY_MEM) {
        clusterThreadFunc = clusterThreadFuncGreedy;
        Debug(Debug::INFO) << "Using GREEDY clustering mode\n";
    } else {
        Debug(Debug::ERROR) << "MMseqs2 align2clust doesn't support clustering mode: " << mode << "\n";
        delete[] assignedCluster;
        seqDbr->close();
        delete seqDbr;
        return EXIT_FAILURE;
    }

    const size_t align2clustResultCount = (mode == Parameters::SET_COVER) ? dbSize : alnDbr.getSize();
    const size_t reorderCapacityChosen = computeReorderCapacity(par, dbSize, mode, align2clustResultCount);
    {
        std::lock_guard<std::mutex> lock(clusterMutex);
        reorderCapacity = reorderCapacityChosen;
        reorderSlots.clear();
        reorderSlots.resize(reorderCapacity);
        reorderFilled.assign(reorderCapacity, 0);
        reorderBufferedCount = 0;
        setCoverCandidates.clear();
        setCoverCandidates.shrink_to_fit();
        setCoverMemberPool.clear();
        setCoverMemberPool.shrink_to_fit();
        setCoverLiveMemberCount = 0;
        currentProcessPosition = 0;
        currentPrefSize = 0;
        allCalculationsDone = false;
    }

    std::thread clusterThread(clusterThreadFunc, assignedCluster);
    
    Timer timer;
    timer.reset();
    PrefInfo *prefRepSizePair = nullptr;
    DBLocalId *lengthOrder = nullptr;
    if (mode != Parameters::SET_COVER) {
        bool alreadyDescending = true;
        for (size_t i = 1; i < dbSize; i++) {
            if (seqDbr->getSeqLen(i) > seqDbr->getSeqLen(i - 1)) {
                alreadyDescending = false;
                break;
            }
        }
        if (alreadyDescending == false) {
            lengthOrder = new(std::nothrow) DBLocalId[dbSize];
            Util::checkAllocation(lengthOrder, "Can not allocate lengthOrder memory in align2clust");
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < dbSize; i++) {
                lengthOrder[i] = static_cast<DBLocalId>(i);
            }
            DBReader<DBKeyType> *order = seqDbr;
            SORT_PARALLEL(lengthOrder, lengthOrder + dbSize,
                          [order](DBLocalId first, DBLocalId second) {
                              const size_t firstLen = order->getSeqLen(first);
                              const size_t secondLen = order->getSeqLen(second);
                              if (firstLen != secondLen) {
                                  return firstLen > secondLen;
                              }
                              return first < second;
                          });
        }
        Debug(Debug::INFO) << "Length order: " << (lengthOrder == nullptr ? "db is already descending" : "built")
                           << " (" << timer.lap() << ")\n";
    }
    
    if (mode == Parameters::SET_COVER) {
        prefRepSizePair = new(std::nothrow) PrefInfo[dbSize];
        Util::checkAllocation(prefRepSizePair, "Can not allocate prefRepSizePair memory in ClusteringAlgorithms::execute");
        
#pragma omp parallel
        {
            int thread_idx = 0;
#ifdef OPENMP
            thread_idx = omp_get_thread_num();
#endif
#pragma omp for schedule(dynamic, 1000)
            for (size_t i = 0; i < seqDbr->getSize(); i++) {
                const DBKeyType clusterId = seqDbr->getDbKey(i);
                const size_t alnId = alnDbr.getId(clusterId);
                const char *data = alnDbr.getData(alnId, thread_idx);
                const size_t dataSize = alnDbr.getEntryLen(alnId);
                prefRepSizePair[i].id = seqDbr->getId(clusterId);
                prefRepSizePair[i].len = static_cast<DBLocalId>(seqDbr->getSeqLen(prefRepSizePair[i].id));
                prefRepSizePair[i].size = (*data == '\0') ? 1 : Util::countLines(data, dataSize);
            }
        }
        SORT_PARALLEL(prefRepSizePair, prefRepSizePair + dbSize, PrefInfo::compareBySizeAndId);
    }

    timer.reset();

    size_t endRange = (mode == Parameters::SET_COVER) ? dbSize : alnDbr.getSize();
    unsigned int swMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);
    Debug::Progress progress(endRange);
    size_t db_maxseqlen = (cluSeqDbr != nullptr)
        ? std::max(seqDbr->getMaxSeqLen(), cluSeqDbr->getMaxSeqLen())
        : seqDbr->getMaxSeqLen();
#pragma omp parallel
    {
        unsigned int threadIdx = 0;
#ifdef OPENMP
        threadIdx = (unsigned int) omp_get_thread_num();
#endif
        Matcher matcher(Parameters::DBTYPE_AMINO_ACIDS, db_maxseqlen, scoring.subMat, &scoring.evaluer,
                       par.compBiasCorrection, par.compBiasCorrectionScale,
                       par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid(),
                       0.0, par.zdrop);
        Sequence query(db_maxseqlen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
        Sequence target(db_maxseqlen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
        Sequence element(db_maxseqlen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
        BlockAligner blockAligner(Parameters::DBTYPE_AMINO_ACIDS, db_maxseqlen, scoring.subMat, &scoring.fastMatrix,
                                 &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                                 -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid());
        std::vector<std::pair<size_t, unsigned short>> targetsWithDiagonal;
        targetsWithDiagonal.reserve(1000);
        const bool swipeLanes = (par.alignerMode != Parameters::ALIGNER_MODE_DEFAULT);
        SwipeBatch swipe(par, scoring.subMat, scoring.evaluer, db_maxseqlen);
        // the member gates of a rep wait for its last target and go in rounds that share the lanes, unless the alignments are written in target order
        ClusterMembers members(swipeLanes ? &swipe : NULL, swipeLanes && cluDbr != NULL && alnWriter == nullptr);
        std::string querySeqBuffer;
        querySeqBuffer.reserve(db_maxseqlen + 1);

        const bool includeAlignFiles = (alnWriter != nullptr);
        std::string alnResultBuffer;
        // Staged member alignments; flushed only if the allpass-gate fully passes.
        std::string pendingMemberAln;
        std::vector<char> alnLineBuffer;
        if (includeAlignFiles) {
            alnLineBuffer.resize(1024 + 32768 * 4);
        }

#pragma omp for schedule(dynamic, 1) nowait
        for (size_t i = 0; i < endRange; i++) {
            progress.updateProgress();
            ClusterResult clusterResult;
            clusterResult.sequenceIdx = i;
            targetsWithDiagonal.clear();
            if (includeAlignFiles) {
                alnResultBuffer.clear();
            }

            size_t representativeId;
            DBKeyType queryKey;

            if (mode == Parameters::SET_COVER) {
                representativeId = prefRepSizePair[i].id;
                queryKey = seqDbr->getDbKey(representativeId);
                clusterResult.prefSize = prefRepSizePair[i].size;   // precomputed in the prefix pass
            } else { // GREEDY || GREEDY_MEM
                representativeId = (lengthOrder != nullptr) ? lengthOrder[i] : static_cast<DBLocalId>(i);
                queryKey = seqDbr->getDbKey(representativeId);
                clusterResult.prefSize = 0;                         // greedy has no currentPrefSize gate
            }
            clusterResult.representativeId = representativeId;

            if (loadAssignedCluster(assignedCluster, representativeId) != DB_LOCAL_ID_INVALID) {
                pushClusterResult(std::move(clusterResult));
                continue;
            }

            const size_t alignmentId = alnDbr.getId(queryKey);
            char *alignmentData = alnDbr.getData(alignmentId, threadIdx);
            size_t prefSize = 0;
            while (*alignmentData != '\0') {
                hit_t hit = QueryMatcher::parsePrefilterHit(alignmentData);
                const size_t targetId = seqDbr->getId(hit.seqId);
                if (mode == Parameters::SET_COVER || loadAssignedCluster(assignedCluster, targetId) == DB_LOCAL_ID_INVALID) {
                    targetsWithDiagonal.push_back(std::make_pair(targetId, hit.diagonal));
                }
                alignmentData = Util::skipLine(alignmentData);
                prefSize++;
            }
            clusterResult.prefSize = prefSize;   // exact parsed count for the aligned path

            size_t queryId = representativeId;
            size_t queryLength = seqDbr->getSeqLen(queryId);
            // every entry holds the representative itself, so it is read only when another target is left
            if (targetsWithDiagonal.size() > 1) {
                querySeqBuffer.assign(seqDbr->getData(queryId, threadIdx), queryLength);
                query.mapSequence(queryId, queryKey, querySeqBuffer.c_str(), queryLength);
                blockAligner.setQuery(&query);
            }
            // the gapped aligners are only set up once an ungapped check fails
            bool blockAlignerInitialized = false;
            bool matcherInitialized = false;
            swipe.startRep();

            for (size_t targetIdx = 0; targetIdx < targetsWithDiagonal.size(); targetIdx++) {
                if (loadAssignedCluster(assignedCluster, representativeId) != DB_LOCAL_ID_INVALID) {
                    break;
                }

                const size_t targetId = targetsWithDiagonal[targetIdx].first;
                const DBKeyType targetKey = seqDbr->getDbKey(targetId);

                const bool isIdentity = (queryKey == targetKey);
                if (isIdentity) {
                    clusterResult.memberIds.push_back(queryId);
                    if (includeAlignFiles) {
                        std::string backtrace = par.addBacktrace ? std::string(queryLength, 'M') : std::string();
                        Matcher::result_t selfResult(queryKey, queryLength, 1.0f, 1.0f, 1.0f, 0.0,
                            queryLength, 0, queryLength - 1, queryLength, 0, queryLength - 1, queryLength, backtrace);
                        appendAlignmentResult(alnResultBuffer, alnLineBuffer.data(), selfResult, par.addBacktrace);
                    }
                    continue;
                }

                if (loadAssignedCluster(assignedCluster, targetId) != DB_LOCAL_ID_INVALID) {
                    continue;
                }

                char *targetSequence = seqDbr->getData(targetId, threadIdx);
                size_t targetLength = seqDbr->getSeqLen(targetId);
                target.mapSequence(targetId, targetKey, targetSequence, targetLength);
                if (alignTarget(par, blockAligner, blockAlignerInitialized, &matcher, matcherInitialized, query, target, &element, cluDbr, cluSeqDbr,
                                scoring.scorePerColThreshold, swMode, targetsWithDiagonal[targetIdx].second, threadIdx, assignedCluster, targetId,
                                includeAlignFiles ? &alnResultBuffer : NULL, includeAlignFiles ? &pendingMemberAln : NULL, alnLineBuffer.data(),
                                &members)) {
                    clusterResult.memberIds.push_back(targetId);
                }
            }
            if (members.deferGates && members.gateCount > 0) {
                alignClusterMembers(par, blockAligner, matcher, matcherInitialized, query, element, cluSeqDbr, swMode, threadIdx, false, NULL, members);
                for (size_t g = 0; g < members.gateCount; g++) {
                    if (members.gates[g].live == false) {
                        clusterResult.memberIds.erase(std::find(clusterResult.memberIds.begin(), clusterResult.memberIds.end(), members.gates[g].targetId));
                    }
                }
                members.gateCount = 0;
            }

            if (includeAlignFiles) {
                alnWriter->writeData(alnResultBuffer.c_str(), alnResultBuffer.length(), queryKey, threadIdx);
            }
            pushClusterResult(std::move(clusterResult));
        }
    }
    if (par.alignerMode != Parameters::ALIGNER_MODE_DEFAULT) {
        printSwipeLanes();
    }

    {
        std::lock_guard<std::mutex> lock(clusterMutex);
        allCalculationsDone = true;
    }
    clusterCondition.notify_one();
    reorderSpaceCondition.notify_all();
    
    if (clusterThread.joinable()) {
        clusterThread.join(); 
    }

    for (size_t i = 0; i < dbSize; ++i) {
        if (loadAssignedCluster(assignedCluster, i) == DB_LOCAL_ID_INVALID) {
            storeAssignedCluster(assignedCluster, i, i);
        }
    }

    std::pair<DBKeyType, DBKeyType> *assignment = new std::pair<DBKeyType, DBKeyType>[dbSize];
    
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (size_t i = 0; i < dbSize; i++) {
            const DBLocalId representativeId = loadAssignedCluster(assignedCluster, i);
            if (representativeId == DB_LOCAL_ID_INVALID) {
                Debug(Debug::ERROR) << "There must be an error: " << i 
                                    << " is not assigned to a cluster\n";
                continue;
            }

            assignment[i].first = seqDbr->getDbKey(representativeId);
            assignment[i].second = seqDbr->getDbKey(i);
        }
    }
    
    SORT_PARALLEL(assignment, assignment + dbSize);

    size_t clusterCount = (dbSize > 0) ? 1 : 0;
    for (size_t i = 1; i < dbSize; i++) {
        clusterCount += (assignment[i].first != assignment[i - 1].first);
    }

    Debug(Debug::INFO) << "Size of the alignment database: " << dbSize << "\n";
    Debug(Debug::INFO) << "Number of clusters: " << clusterCount << "\n";
    
    writeClustering(&resultWriter, assignment, dbSize);
    
    delete[] assignedCluster;
    delete[] assignment;
    if (prefRepSizePair != nullptr) {
        delete[] prefRepSizePair;
    }
    if (lengthOrder != nullptr) {
        delete[] lengthOrder;
    }
    seqDbr->close();
    delete seqDbr;

    if (cluDbr != nullptr) {
        cluDbr->close();
        delete cluDbr;
    }
    if (cluSeqDbr != nullptr) {
        cluSeqDbr->close();
        delete cluSeqDbr;
    }
    return 0;
}

// --createdb-mode 3: record header { rep, first target }, followed by the target keys
struct RepTargets {
    ClusterPair pair;
    unsigned int targetCount;

    // 0 if bytes do not hold the whole record
    static size_t recordSize(const char *record, size_t bytes) {
        if (bytes < sizeof(RepTargets)) {
            return 0;
        }
        RepTargets header;
        memcpy(&header, record, sizeof(RepTargets));
        const size_t size = sizeof(RepTargets) + header.targetCount * sizeof(DBKeyType);
        return (bytes >= size) ? size : 0;
    }
};

// --createdb-mode 3: a row of <pref>.partitions, the entries of keys [firstKey, endKey) in <pref>.<node>, endKey = firstKey of the next row
struct PrefPartition {
    size_t firstKey;
    size_t prefOffset;
    size_t prefBytes;
    size_t cost;
    size_t node;
    size_t endKey;
};

// --createdb-mode 3: a record of the rep targets, where its targets start in the split or chunk, and the bytes read for its rep and targets
struct RepTargetsRecord {
    RepTargets header;
    size_t targetOffset;
    size_t readBytes;

    static bool compareByRep(const RepTargetsRecord &first, const RepTargetsRecord &second) {
        return ClusterPair::compareByRep(first.header.pair, second.header.pair);
    }
};

// the master aligns only on one node
static size_t getFirstAlignNode() {
    return (MMseqsNode::count > 1) ? 1 : 0;
}

// the group of a key: groupKeyOffsets holds where the key range of every group starts, then the key count
static size_t getKeyGroup(const std::vector<size_t> &groupKeyOffsets, size_t key) {
    return std::upper_bound(groupKeyOffsets.begin(), groupKeyOffsets.end(), key) - groupKeyOffsets.begin() - 1;
}

// the groups go to the aligning nodes in turn, back and forth, which align the reps of their groups
static size_t getGroupNode(size_t group) {
    const size_t alignNodes = MMseqsNode::count - getFirstAlignNode();
    const size_t turn = group % (2 * alignNodes);
    return getFirstAlignNode() + std::min(turn, 2 * alignNodes - 1 - turn);
}

static size_t getAlignNode(const std::vector<size_t> &groupKeyOffsets, size_t key) {
    return getGroupNode(getKeyGroup(groupKeyOffsets, key));
}

// --createdb-mode 3: the records [begin, end) of a chunk aligned at once and the rep and target keys to load
struct RepWindow {
    size_t chunk;
    size_t begin;
    size_t end;
    std::vector<DBKeyType> keys;
};

// a pref key: 7 bits a byte, the high bit set on all bytes but the last
static inline DBKeyType getPrefKey(const char *&data) {
    DBKeyType key = 0;
    for (unsigned int shift = 0;; shift += 7) {
        const unsigned char byte = static_cast<unsigned char>(*data++);
        key |= static_cast<DBKeyType>(byte & 0x7F) << shift;
        if (byte < 0x80) {
            return key;
        }
    }
}

// a node publishes its log after this many bytes or seconds
static const size_t PUBLISH_BYTES = 4 * 1024 * 1024;
static const time_t PUBLISH_SECONDS = 10;
static const time_t READ_SECONDS = 1;
// batches per thread of the records aligned at once, taken in rep order
static const size_t ALIGN_BATCHES_PER_THREAD = 64;
// how many pairs ahead the master prefetches the assigned word of a pair
static const size_t PREFETCH_PAIRS = 16;
// cluster threads of the master on one node, at most a quarter of its threads
static const int CLUSTER_THREADS = 4;

// each of these takes the memory limit divided by its divisor
static const size_t MASTER_MEMORY_DIVISOR = 2;        // the master on one node, which also aligns
static const size_t ASSIGNED_BITS_MEMORY_DIVISOR = 2; // the assigned bit of every key on the master
static const size_t RING_MEMORY_DIVISOR = 256;        // the ring of assigned bits of every node
static const size_t OWN_BITS_MEMORY_DIVISOR = 8;      // the assigned bits of the own groups of an aligning node
static const size_t MEMBERS_MEMORY_DIVISOR = 4;       // --filter-cludb-file: the members of the targets of a node
static const size_t PREF_SPLIT_MEMORY_DIVISOR = 16;   // a pref split, the next one, their targets and member requests
static const size_t WRITE_BATCH_MEMORY_DIVISOR = 16;  // the records of a batch
static const size_t CHUNK_MEMORY_DIVISOR = 32;        // a chunk of received records
static const size_t WINDOW_MEMORY_DIVISOR = 8;        // the reps and targets of a window, which every record fits
static const size_t MEMBER_PART_MEMORY_DIVISOR = 16;  // --filter-cludb-file: the members aligned at once
static const size_t STEP_MEMORY_DIVISOR = 4;          // a clustering step, besides the sets of set cover
static const size_t STEP_PAIRS_MEMORY_DIVISOR = 16;   // the pairs of a clustering step
static const size_t LOG_PART_MEMORY_DIVISOR = 64;     // a part of a log read at once
static const size_t CLUSTERS_MEMORY_DIVISOR = 16;     // the cluster pairs or member requests that a node sends at once

// --createdb-mode 3: write every rep of the partitions of this node with its targets for this node to align; memberRequests: request the --filter-cludb-file members of the targets
static std::string writeRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<PrefPartition> &nodePartitions, size_t maxWindowBytes,
                                   size_t memoryLimit, size_t paddedDbSize, size_t prefSizeBits, const std::string &repTargets, MMseqsNode::ClusterPairWriter *memberRequests) {
    const size_t dbSize = seqDbr.getSize();
    // a record holds at most maxRecordTargets targets, so they fit maxWindowBytes as pages
    const size_t batchBytes = std::max<size_t>(1, memoryLimit / WRITE_BATCH_MEMORY_DIVISOR);
    const size_t maxRecordTargets = std::min<size_t>(UINT_MAX, std::max<size_t>(1, std::min(batchBytes / 2 / sizeof(DBKeyType), maxWindowBytes / std::max<size_t>(seqDbr.getMaxSeqLen() + 2, Util::getPageSize()))));

    const std::string repTargetsFile = repTargets + "." + SSTR(MMseqsNode::rank);
    FILE *repTargetsHandle = FileUtil::openAndDelete(repTargetsFile.c_str(), "w");
    std::string repTargetsIndexData;
    size_t repTargetsBytes = 0;
    std::vector<size_t> prefSplitOffsets;
    std::vector<DBKeyType> entryRepKeys;
    std::vector<size_t> entryTargetOffsets;
    std::vector<size_t> entryRecordOffsets;
    std::vector<DBKeyType> splitTargets;
    std::vector<size_t> batchEntryOffsets;
    std::vector<RepTargetsRecord> records;
    std::vector<size_t> recordOffsets;
    std::vector<char> flushBuffer;
    std::vector<size_t> nodeOffsets(MMseqsNode::count + 1);
    std::vector<std::vector<DBKeyType> > threadRequestKeys(par.threads);
    std::vector<DBKeyType> requestKeys;
    DBReader<DBKeyType> *prefDb = NULL;
    for (size_t i = 0; i < nodePartitions.size(); i++) {
        const PrefPartition &partition = nodePartitions[i];
        const std::string prefFile = par.db2 + "." + SSTR(partition.node);
        if (i == 0 || partition.node != nodePartitions[i - 1].node) {
            if (prefDb != NULL) {
                prefDb->close();
                delete prefDb;
            }
            prefDb = new DBReader<DBKeyType>(prefFile.c_str(), (par.db2 + ".index." + SSTR(partition.node)).c_str(), par.threads,
                                             DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
            prefDb->open(DBReader<DBKeyType>::NOSORT);
        }
        // splits of about prefSplitBytes that end where an entry starts
        const size_t prefEnd = partition.prefOffset + partition.prefBytes;
        const size_t partitionEndEntry = prefDb->findEntryByOffset(prefEnd);
        // a pref byte takes two buffers, up to 8 bytes of targets and 16 more of member requests, an entry 64 bytes of offsets and keys
        const double entriesPerByte = static_cast<double>(partitionEndEntry - prefDb->findEntryByOffset(partition.prefOffset)) / std::max<size_t>(1, partition.prefBytes);
        const double bytesPerPrefByte = (memberRequests != NULL) ? 26.0 : 10.0;
        const size_t prefSplitBytes = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(memoryLimit / PREF_SPLIT_MEMORY_DIVISOR) / (bytesPerPrefByte + 64.0 * entriesPerByte)));
        prefSplitOffsets.clear();
        for (size_t offset = partition.prefOffset; offset < prefEnd; offset += prefSplitBytes) {
            const size_t entry = prefDb->findEntryByOffset(offset);
            if (prefSplitOffsets.empty() || prefSplitOffsets.back() != entry) {
                prefSplitOffsets.push_back(entry);
            }
        }
        prefSplitOffsets.push_back(partitionEndEntry);
        size_t nextRepKey = partition.firstKey;
        size_t partitionBytes = 0;
        for (size_t split = 0; split + 1 < prefSplitOffsets.size(); split++) {
            const size_t splitBegin = prefSplitOffsets[split];
            const size_t entryCount = prefSplitOffsets[split + 1] - splitBegin;
            prefDb->loadData(splitBegin, prefSplitOffsets[split + 1]);
            if (split + 2 < prefSplitOffsets.size()) {
                prefDb->prefetchData(prefSplitOffsets[split + 1], prefSplitOffsets[split + 2]);
            }
            entryRepKeys.resize(entryCount);
            entryTargetOffsets.assign(entryCount + 1, 0);
            entryRecordOffsets.assign(entryCount + 1, 0);
#pragma omp parallel for schedule(dynamic, 100)
            for (size_t entry = 0; entry < entryCount; entry++) {
                const char *data = prefDb->getData(splitBegin + entry, 0);
                const size_t keys = std::count_if(data, data + prefDb->getEntryLen(splitBegin + entry) - 1, [](char byte) { return static_cast<unsigned char>(byte) < 0x80; });
                if (keys == 0) {
                    Debug(Debug::ERROR) << prefFile << " has an entry without its key\n";
                    EXIT(EXIT_FAILURE);
                }
                entryTargetOffsets[entry + 1] = keys - 1;
            }
            for (size_t entry = 0; entry < entryCount; entry++) {
                entryTargetOffsets[entry + 1] += entryTargetOffsets[entry];
                partitionBytes += prefDb->getEntryLen(splitBegin + entry);
            }
            splitTargets.resize(entryTargetOffsets[entryCount]);
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                threadRequestKeys[thread_idx].clear();
#pragma omp for schedule(dynamic, 100)
                for (size_t entry = 0; entry < entryCount; entry++) {
                    const char *data = prefDb->getData(splitBegin + entry, 0);
                    const DBKeyType repKey = getPrefKey(data);
                    entryRepKeys[entry] = repKey;
                    DBKeyType *targets = splitTargets.data() + entryTargetOffsets[entry];
                    const size_t targetCount = entryTargetOffsets[entry + 1] - entryTargetOffsets[entry];
                    const size_t repLength = seqDbr.getSeqLen(repKey);
                    for (size_t j = 0; j < targetCount; j++) {
                        targets[j] = ((j > 0) ? targets[j - 1] : 0) + getPrefKey(data);
                        if (targets[j] >= dbSize || targets[j] == repKey || (j > 0 && targets[j] <= targets[j - 1])) {
                            Debug(Debug::ERROR) << "Entry " << repKey << " of " << prefFile << " has the target " << targets[j] << " twice, out of order or not in " << par.db1 << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        // the targets that alignTarget may accept
                        if (memberRequests != NULL && Util::canBeCovered(par.covThr, par.covMode, repLength, seqDbr.getSeqLen(targets[j]))) {
                            threadRequestKeys[thread_idx].push_back(targets[j]);
                        }
                    }
                    entryRecordOffsets[entry + 1] = (targetCount + maxRecordTargets - 1) / maxRecordTargets;
                }
            }
            // a node requests the members of a target once per split
            if (memberRequests != NULL) {
                requestKeys.clear();
                for (int thread = 0; thread < par.threads; thread++) {
                    requestKeys.insert(requestKeys.end(), threadRequestKeys[thread].begin(), threadRequestKeys[thread].end());
                }
                SORT_PARALLEL(requestKeys.begin(), requestKeys.end());
                requestKeys.erase(std::unique(requestKeys.begin(), requestKeys.end()), requestKeys.end());
                for (size_t j = 0; j < requestKeys.size(); j++) {
                    memberRequests->add(requestKeys[j], MMseqsNode::rank);
                }
            }
            // a batch takes entries while their records fit, at least one
            batchEntryOffsets.assign(1, 0);
            for (size_t entry = 0, recordBytes = 0; entry < entryCount; entry++) {
                if (entryRepKeys[entry] < nextRepKey || entryRepKeys[entry] >= partition.endKey) {
                    Debug(Debug::ERROR) << prefFile << " has the entry " << entryRepKeys[entry] << " out of key order or outside its partition\n";
                    EXIT(EXIT_FAILURE);
                }
                nextRepKey = entryRepKeys[entry] + 1;
                const size_t entryRecordBytes = entryRecordOffsets[entry + 1] * sizeof(RepTargets) + (entryTargetOffsets[entry + 1] - entryTargetOffsets[entry]) * sizeof(DBKeyType);
                entryRecordOffsets[entry + 1] += entryRecordOffsets[entry];
                if (entry > batchEntryOffsets.back() && recordBytes + entryRecordBytes > batchBytes) {
                    batchEntryOffsets.push_back(entry);
                    recordBytes = 0;
                }
                recordBytes += entryRecordBytes;
            }
            batchEntryOffsets.push_back(entryCount);
            for (size_t batch = 0; batch + 1 < batchEntryOffsets.size(); batch++) {
                const size_t firstEntry = batchEntryOffsets[batch];
                const size_t endEntry = batchEntryOffsets[batch + 1];
                const size_t firstRecord = entryRecordOffsets[firstEntry];
                records.resize(entryRecordOffsets[endEntry] - firstRecord);
                if (records.empty()) {
                    continue;
                }
#pragma omp parallel for schedule(dynamic, 100)
                for (size_t entry = firstEntry; entry < endEntry; entry++) {
                    const size_t targetCount = entryTargetOffsets[entry + 1] - entryTargetOffsets[entry];
                    size_t j = entryRecordOffsets[entry] - firstRecord;
                    const size_t prefSizeBit = (targetCount > 0) ? std::min<size_t>(prefSizeBits - 1, 63 - __builtin_clzll(targetCount)) : 0;
                    for (size_t first = 0; first < targetCount; first += maxRecordTargets, j++) {
                        RepTargets &header = records[j].header;
                        header.pair.repKey = (prefSizeBits - 1 - prefSizeBit) * paddedDbSize + entryRepKeys[entry];
                        header.pair.memberKey = splitTargets[entryTargetOffsets[entry] + first];
                        header.targetCount = static_cast<unsigned int>(std::min(targetCount - first, maxRecordTargets));
                        records[j].targetOffset = entryTargetOffsets[entry] + first;
                    }
                }
                // the node aligns its records in rep order
                SORT_PARALLEL(records.begin(), records.end(), RepTargetsRecord::compareByRep);
                recordOffsets.resize(records.size() + 1);
                recordOffsets[0] = 0;
                for (size_t j = 0; j < records.size(); j++) {
                    recordOffsets[j + 1] = recordOffsets[j] + sizeof(RepTargets) + records[j].header.targetCount * sizeof(DBKeyType);
                }
                flushBuffer.resize(recordOffsets.back());
#pragma omp parallel for schedule(dynamic, 100)
                for (size_t j = 0; j < records.size(); j++) {
                    const RepTargets &header = records[j].header;
                    char *data = flushBuffer.data() + recordOffsets[j];
                    memcpy(data, &header, sizeof(RepTargets));
                    memcpy(data + sizeof(RepTargets), splitTargets.data() + records[j].targetOffset, header.targetCount * sizeof(DBKeyType));
                }
                // all bytes of the flush are for this node
                for (size_t node = 0; node <= MMseqsNode::count; node++) {
                    nodeOffsets[node] = (node <= MMseqsNode::rank) ? 0 : flushBuffer.size();
                }
                if (fwrite(flushBuffer.data(), sizeof(char), flushBuffer.size(), repTargetsHandle) != flushBuffer.size()) {
                    Debug(Debug::ERROR) << "Cannot write " << repTargetsFile << "\n";
                    EXIT(EXIT_FAILURE);
                }
                MMseqsNode::addExchangeFlush(repTargetsIndexData, repTargetsBytes, nodeOffsets);
            }
        }
        if (partitionBytes != partition.prefBytes) {
            Debug(Debug::ERROR) << prefFile << " has an entry that ends after its partition\n";
            EXIT(EXIT_FAILURE);
        }
    }
    if (prefDb != NULL) {
        prefDb->close();
        delete prefDb;
    }
    // only this node reads its file, so it needs no barrier
    if (fclose(repTargetsHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << repTargetsFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    return repTargetsIndexData;
}

// --createdb-mode 3: align the received records window by window and log the accepted pairs
static void alignRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<size_t> &groupKeyOffsets, size_t lagKeys, size_t memoryLimit, size_t representativeResidues,
                            size_t paddedDbSize, size_t prefSizeBits, const std::string &repTargetsIndex, const std::string &repTargets, const std::string &accepted,
                            const std::string &assignedChanges, const std::vector<ClusterPair> &members, DBReader<DBKeyType> *memberDbr,
                            double &waitSeconds, size_t &alignedPairs) {
    const size_t dbSize = seqDbr.getSize();
    const size_t orderKeyEnd = (prefSizeBits - 1) * paddedDbSize + dbSize;
    const size_t maxSeqLen = seqDbr.getMaxSeqLen();
    // a scattered entry is read as at least one page
    const size_t pageSize = Util::getPageSize();
    AlignmentScoring scoring(par, representativeResidues);
    const unsigned int swMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);
    std::vector<MMseqsNode::ExchangeBlock> repBlocks;
    std::vector<FILE *> repFiles;
    // the flushes of a node are in rep order, but not in order key order with set cover
    MMseqsNode::openExchange(repTargetsIndex, repTargets, repBlocks, repFiles, par.clusteringMode != Parameters::SET_COVER);
    if (repBlocks.size() * MMseqsNode::EXCHANGE_BUFFER_SIZE > std::max(MMseqsNode::EXCHANGE_BUFFER_SIZE, memoryLimit / CHUNK_MEMORY_DIVISOR)) {
        Debug(Debug::ERROR) << "An aligning node needs " << repBlocks.size() * MMseqsNode::EXCHANGE_BUFFER_SIZE << " bytes to read the " << repBlocks.size()
                            << " flushes of its rep targets, more than a 32nd of the memory limit of " << memoryLimit << " bytes. Run with more memory\n";
        EXIT(EXIT_FAILURE);
    }
    const std::string acceptedLog = accepted + "." + SSTR(MMseqsNode::rank);
    FILE *acceptedHandle = FileUtil::openAndDelete(acceptedLog.c_str(), "w");
    size_t acceptedBytes = 0;

    // assigned bits: the keys of the own groups one after the other in ownWords, other reps in a ring of order key words from the master
    std::vector<size_t> ownKeyOffsets(groupKeyOffsets.size() - 1, 0);
    size_t ownKeys = 0;
    for (size_t group = 0; group < ownKeyOffsets.size(); group++) {
        ownKeyOffsets[group] = ownKeys;
        if (getGroupNode(group) == MMseqsNode::rank) {
            ownKeys += groupKeyOffsets[group + 1] - groupKeyOffsets[group];
        }
    }
    std::vector<uint64_t> ownWords((ownKeys + 63) / 64, 0);
    std::vector<uint64_t> ring(lagKeys / 64 + 1, 0);
    size_t ringFirstWord = 0;
    size_t clusteredKey = 0;
    const auto getOwnKey = [&](size_t group, size_t key) {
        return ownKeyOffsets[group] + key - groupKeyOffsets[group];
    };
    const auto isAssigned = [&](size_t orderKey) {
        const size_t key = orderKey % paddedDbSize;
        const size_t group = getKeyGroup(groupKeyOffsets, key);
        if (getGroupNode(group) == MMseqsNode::rank) {
            const size_t ownKey = getOwnKey(group, key);
            return ((__atomic_load_n(&ownWords[ownKey / 64], __ATOMIC_RELAXED) >> (ownKey % 64)) & 1) != 0;
        }
        const size_t word = orderKey / 64;
        const size_t firstWord = __atomic_load_n(&ringFirstWord, __ATOMIC_ACQUIRE);
        if (word < firstWord || word - firstWord >= ring.size()) {
            return false;
        }
        const uint64_t bits = __atomic_load_n(&ring[word % ring.size()], __ATOMIC_ACQUIRE);
        // the ring may have moved on and reused the slot
        return __atomic_load_n(&ringFirstWord, __ATOMIC_ACQUIRE) == firstWord && ((bits >> (orderKey % 64)) & 1) != 0;
    };
    const auto setRingKey = [&](size_t key) {
        const size_t word = key / 64;
        if (word >= ringFirstWord && word - ringFirstWord < ring.size()) {
            __atomic_fetch_or(&ring[word % ring.size()], 1ULL << (key % 64), __ATOMIC_RELEASE);
        }
    };
    // step header: clustered key, entering first word and count, ring key count, node key offsets
    const size_t headerWords = 4 + MMseqsNode::count + 1;
    size_t changesBytes = 0;
    std::vector<uint64_t> changes;
    const auto readChanges = [&](size_t from, size_t words) {
        changes.resize(words);
        if (words > 0) {
            MMseqsNode::readLog(assignedChanges, from, from + words * sizeof(uint64_t), reinterpret_cast<char *>(changes.data()));
        }
    };
    const auto readAssignedChanges = [&]() {
        size_t publishedKey;
        size_t publishedBytes;
        if (MMseqsNode::readLogProgress(assignedChanges, publishedKey, publishedBytes) == false) {
            return;
        }
        while (changesBytes + headerWords * sizeof(uint64_t) <= publishedBytes) {
            readChanges(changesBytes, headerWords);
            const size_t stepClusteredKey = changes[0];
            const size_t enteringFirstWord = changes[1];
            const size_t enteringWords = changes[2];
            const size_t lagKeyCount = changes[3];
            const size_t nodeKeyOffset = changes[4 + MMseqsNode::rank];
            const size_t nodeKeyEnd = changes[5 + MMseqsNode::rank];
            const size_t commonStart = changesBytes + headerWords * sizeof(uint64_t);
            const size_t ownStart = commonStart + (enteringWords + lagKeyCount + nodeKeyOffset) * sizeof(uint64_t);
            changesBytes = commonStart + (enteringWords + lagKeyCount + changes[4 + MMseqsNode::count]) * sizeof(uint64_t);
            if (changesBytes > publishedBytes || stepClusteredKey < clusteredKey || nodeKeyEnd < nodeKeyOffset) {
                Debug(Debug::ERROR) << "Cannot read " << assignedChanges << "\n";
                EXIT(EXIT_FAILURE);
            }
            // clear the slots that leave the ring before releasing ringFirstWord
            const size_t firstWord = stepClusteredKey / 64;
            for (size_t word = ringFirstWord; word < std::min(firstWord, ringFirstWord + ring.size()); word++) {
                __atomic_store_n(&ring[word % ring.size()], 0, __ATOMIC_RELAXED);
            }
            __atomic_store_n(&ringFirstWord, firstWord, __ATOMIC_RELEASE);
            clusteredKey = stepClusteredKey;
            readChanges(commonStart, enteringWords + lagKeyCount);
            for (size_t i = 0; i < enteringWords; i++) {
                if (enteringFirstWord + i >= firstWord && enteringFirstWord + i - firstWord < ring.size()) {
                    __atomic_fetch_or(&ring[(enteringFirstWord + i) % ring.size()], changes[i], __ATOMIC_RELEASE);
                }
            }
            for (size_t i = enteringWords; i < changes.size(); i++) {
                for (size_t orderKey = changes[i]; orderKey < orderKeyEnd; orderKey += paddedDbSize) {
                    setRingKey(orderKey);
                }
            }
            readChanges(ownStart, nodeKeyEnd - nodeKeyOffset);
            for (size_t i = 0; i < changes.size(); i++) {
                const size_t group = getKeyGroup(groupKeyOffsets, changes[i]);
                if (getGroupNode(group) != MMseqsNode::rank) {
                    Debug(Debug::ERROR) << "Cannot read " << assignedChanges << ": a key of another node\n";
                    EXIT(EXIT_FAILURE);
                }
                const size_t ownKey = getOwnKey(group, changes[i]);
                __atomic_fetch_or(&ownWords[ownKey / 64], 1ULL << (ownKey % 64), __ATOMIC_RELAXED);
            }
        }
    };

    // two chunks, as the previous window may still use the other one
    std::vector<char> chunkData[2];
    std::vector<RepTargetsRecord> chunkRecords[2];
    std::vector<DBKeyType> chunkTargets[2];
    std::vector<size_t> recordOffsets;
    size_t chunkPos[2] = { 0, 0 };
    size_t chunk = 0;
    const ClusterPair lastRecord = { DB_KEY_INVALID, DB_KEY_INVALID };
    std::vector<std::vector<DBKeyType> > threadKeys(par.threads);
    // a window: records while their reps and targets fit, at least one, and the rep and unassigned target keys to load
    const auto readWindow = [&](RepWindow &window) {
        if (chunkPos[chunk] == chunkRecords[chunk].size()) {
            chunk = 1 - chunk;
            const size_t recordCount = MMseqsNode::readExchangeRange(repBlocks, memoryLimit / CHUNK_MEMORY_DIVISOR, RepTargets::recordSize, ClusterPair::compareByRep, lastRecord,
                                                                     chunkData[chunk], &recordOffsets, repTargets);
            std::vector<RepTargetsRecord> &records = chunkRecords[chunk];
            records.resize(recordCount);
            size_t targetCount = 0;
            for (size_t i = 0; i < recordCount; i++) {
                memcpy(&records[i].header, chunkData[chunk].data() + recordOffsets[i], sizeof(RepTargets));
                records[i].targetOffset = targetCount;
                targetCount += records[i].header.targetCount;
            }
            chunkTargets[chunk].resize(targetCount);
#pragma omp parallel for schedule(dynamic, 100)
            for (size_t i = 0; i < recordCount; i++) {
                RepTargetsRecord &record = records[i];
                DBKeyType *targets = chunkTargets[chunk].data() + record.targetOffset;
                memcpy(targets, chunkData[chunk].data() + recordOffsets[i] + sizeof(RepTargets), record.header.targetCount * sizeof(DBKeyType));
                const size_t repKey = record.header.pair.repKey % paddedDbSize;
                record.readBytes = seqDbr.getEntryLen(repKey);
                for (unsigned int j = 0; j < record.header.targetCount; j++) {
                    record.readBytes += std::max(seqDbr.getEntryLen(targets[j]), pageSize);
                }
            }
            SORT_PARALLEL(records.begin(), records.end(), RepTargetsRecord::compareByRep);
            chunkPos[chunk] = 0;
        }
        window.chunk = chunk;
        window.begin = chunkPos[chunk];
        window.end = window.begin;
        for (size_t windowBytes = 0; window.end < chunkRecords[chunk].size(); window.end++) {
            windowBytes += chunkRecords[chunk][window.end].readBytes;
            if (window.end > window.begin && windowBytes > memoryLimit / WINDOW_MEMORY_DIVISOR) {
                break;
            }
        }
        chunkPos[chunk] = window.end;
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            threadKeys[thread_idx].clear();
#pragma omp for schedule(dynamic, 100)
            for (size_t i = window.begin; i < window.end; i++) {
                const RepTargetsRecord &record = chunkRecords[window.chunk][i];
                threadKeys[thread_idx].push_back(record.header.pair.repKey % paddedDbSize);
                if (isAssigned(record.header.pair.repKey)) {
                    continue;
                }
                const DBKeyType *targets = chunkTargets[window.chunk].data() + record.targetOffset;
                for (unsigned int j = 0; j < record.header.targetCount; j++) {
                    if (isAssigned(targets[j]) == false) {
                        threadKeys[thread_idx].push_back(targets[j]);
                    }
                }
            }
        }
        window.keys.clear();
        for (int thread = 0; thread < par.threads; thread++) {
            window.keys.insert(window.keys.end(), threadKeys[thread].begin(), threadKeys[thread].end());
        }
        SORT_PARALLEL(window.keys.begin(), window.keys.end());
        window.keys.erase(std::unique(window.keys.begin(), window.keys.end()), window.keys.end());
    };
    const auto getFirstRepKey = [&](const RepWindow &window) {
        return (window.end > window.begin) ? static_cast<size_t>(chunkRecords[window.chunk][window.begin].header.pair.repKey) : orderKeyEnd;
    };
    // a band: the keys from the first window key on that fit the window memory, kept while the windows fit it; the window keys if they do not fit
    size_t bandBegin = 0;
    size_t bandEnd = 0;
    const auto readBand = [&](const std::vector<DBKeyType> &windowKeys, bool isPrefetch) {
        if (windowKeys.empty() || (windowKeys.front() >= bandBegin && windowKeys.back() < bandEnd)) {
            return;
        }
        const size_t begin = windowKeys.front();
        const size_t end = seqDbr.getSplitKeyOffsets(begin, dbSize, memoryLimit / WINDOW_MEMORY_DIVISOR, 0)[1];
        const bool fitsBand = windowKeys.back() < end;
        if (isPrefetch) {
            if (fitsBand) {
                seqDbr.prefetchData(begin, end);
            } else {
                seqDbr.prefetchData(windowKeys);
            }
            return;
        }
        bandBegin = begin;
        bandEnd = fitsBand ? end : begin;
        if (fitsBand) {
            seqDbr.loadData(begin, end);
        } else {
            seqDbr.loadData(windowKeys);
        }
    };

    RepWindow window;
    RepWindow nextWindow;
    readWindow(window);
    MMseqsNode::publishLog(acceptedHandle, acceptedLog, getFirstRepKey(window), 0);
    size_t publishedBytes = 0;
    time_t publishTime = time(NULL);
    time_t changesTime = publishTime;
    std::mutex acceptedMutex;
    std::mutex assignedChangesMutex;
    std::vector<unsigned char> batchDone;
    std::vector<std::vector<ClusterPair> > batchPairs;
    std::vector<ClusterPair> windowPairs;
    std::vector<size_t> memberStarts;
    std::vector<size_t> memberBytes;
    std::vector<size_t> partOffsets;
    std::vector<unsigned char> membersPassed;
    std::vector<DBKeyType> partKeys;
    std::vector<size_t> repGroups;
    while (window.end > window.begin) {
        // the master fills the ring up to lagKeys after the clustered key
        const size_t windowRepKey = getFirstRepKey(window);
        if (windowRepKey >= clusteredKey + lagKeys) {
            Timer waitTimer;
            MMseqsNode::publishLog(acceptedHandle, acceptedLog, windowRepKey, acceptedBytes);
            publishedBytes = acceptedBytes;
            for (unsigned int polls = 0; windowRepKey >= clusteredKey + lagKeys; polls++) {
                MMseqsNode::sleepBeforePoll(polls);
                readAssignedChanges();
            }
            publishTime = time(NULL);
            waitSeconds += waitTimer.getTimediff();
        }
        readBand(window.keys, false);
        readWindow(nextWindow);
        readBand(nextWindow.keys, true);
        const size_t nextRepKey = getFirstRepKey(nextWindow);
        const std::vector<RepTargetsRecord> &records = chunkRecords[window.chunk];
        const size_t recordCount = window.end - window.begin;
        const size_t batchCount = std::min(recordCount, par.threads * ALIGN_BATCHES_PER_THREAD);
        batchDone.assign(batchCount, 0);
        batchPairs.assign(batchCount, std::vector<ClusterPair>());
        size_t firstUnfinishedBatch = 0;
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            Sequence query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
            Sequence target(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
            BlockAligner blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                                     &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                                     -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid());
            std::vector<ClusterPair> pairs;
            size_t threadAlignedPairs = 0;
#pragma omp for schedule(dynamic, 1)
            for (size_t batch = 0; batch < batchCount; batch++) {
                pairs.clear();
                for (size_t i = window.begin + recordCount * batch / batchCount; i < window.begin + recordCount * (batch + 1) / batchCount; i++) {
                    const RepTargets &header = records[i].header;
                    if (isAssigned(header.pair.repKey)) {
                        continue;
                    }
                    const size_t repKey = header.pair.repKey % paddedDbSize;
                    query.mapSequence(repKey, repKey, seqDbr.getData(repKey, thread_idx), seqDbr.getSeqLen(repKey));
                    blockAligner.setQuery(&query);
                    bool blockAlignerInitialized = false;
                    bool matcherInitialized = false;
                    const DBKeyType *targets = chunkTargets[window.chunk].data() + records[i].targetOffset;
                    for (unsigned int j = 0; j < header.targetCount && isAssigned(header.pair.repKey) == false; j++) {
                        const DBKeyType targetKey = targets[j];
                        // readWindow did not load the targets of reps assigned then, which the ring may have dropped since
                        if (isAssigned(targetKey) || std::binary_search(window.keys.begin(), window.keys.end(), targetKey) == false) {
                            continue;
                        }
                        target.mapSequence(targetKey, targetKey, seqDbr.getData(targetKey, thread_idx), seqDbr.getSeqLen(targetKey));
                        threadAlignedPairs++;
                        if (alignTarget(par, blockAligner, blockAlignerInitialized, NULL, matcherInitialized, query, target, NULL, NULL, NULL,
                                        scoring.scorePerColThreshold, 0, 0, thread_idx, NULL, 0, NULL, NULL, NULL, NULL)) {
                            const ClusterPair pair = { header.pair.repKey, targetKey };
                            pairs.push_back(pair);
                        }
                    }
                }
                // pairs go to the log in batch order, so the reps before the first unfinished batch are done
                bool readsChanges = false;
                {
                    std::lock_guard<std::mutex> lock(acceptedMutex);
                    batchPairs[batch].swap(pairs);
                    batchDone[batch] = 1;
                    // --filter-cludb-file logs the pairs once their members are aligned
                    if (memberDbr != NULL) {
                        continue;
                    }
                    for (; firstUnfinishedBatch < batchCount && batchDone[firstUnfinishedBatch]; firstUnfinishedBatch++) {
                        std::vector<ClusterPair> &donePairs = batchPairs[firstUnfinishedBatch];
                        if (fwrite(donePairs.data(), sizeof(ClusterPair), donePairs.size(), acceptedHandle) != donePairs.size()) {
                            Debug(Debug::ERROR) << "Cannot write " << acceptedLog << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        acceptedBytes += donePairs.size() * sizeof(ClusterPair);
                        std::vector<ClusterPair>().swap(donePairs);
                    }
                    const time_t now = time(NULL);
                    if (acceptedBytes - publishedBytes >= PUBLISH_BYTES || now - publishTime >= PUBLISH_SECONDS) {
                        const size_t doneRepKey = (firstUnfinishedBatch < batchCount) ? static_cast<size_t>(records[window.begin + recordCount * firstUnfinishedBatch / batchCount].header.pair.repKey) : nextRepKey;
                        MMseqsNode::publishLog(acceptedHandle, acceptedLog, doneRepKey, acceptedBytes);
                        publishedBytes = acceptedBytes;
                        publishTime = now;
                        readsChanges = doneRepKey < orderKeyEnd;
                    }
                    if (readsChanges || now - changesTime >= READ_SECONDS) {
                        readsChanges = true;
                        changesTime = now;
                    }
                }
                // one thread reads the changes while the others go on
                if (readsChanges) {
                    std::unique_lock<std::mutex> changesLock(assignedChangesMutex, std::try_to_lock);
                    if (changesLock.owns_lock()) {
                        readAssignedChanges();
                    }
                }
            }
            __sync_fetch_and_add(&alignedPairs, threadAlignedPairs);
        }
        // --filter-cludb-file: align the members of the accepted targets in parts that fit the memory
        if (memberDbr != NULL) {
            windowPairs.clear();
            for (size_t batch = 0; batch < batchCount; batch++) {
                windowPairs.insert(windowPairs.end(), batchPairs[batch].begin(), batchPairs[batch].end());
            }
            memberStarts.resize(windowPairs.size());
            memberBytes.resize(windowPairs.size());
#pragma omp parallel for schedule(dynamic, 1024)
            for (size_t i = 0; i < windowPairs.size(); i++) {
                const ClusterPair target = { windowPairs[i].memberKey, windowPairs[i].memberKey };
                memberStarts[i] = std::lower_bound(members.begin(), members.end(), target, ClusterPair::compareByRep) - members.begin();
                memberBytes[i] = 0;
                for (size_t member = memberStarts[i]; member < members.size() && members[member].repKey == target.repKey; member++) {
                    memberBytes[i] += std::max(memberDbr->getEntryLen(members[member].memberKey), pageSize);
                }
            }
            partOffsets.assign(1, 0);
            for (size_t i = 0, partBytes = 0; i < windowPairs.size(); i++) {
                if (i > partOffsets.back() && partBytes + memberBytes[i] > memoryLimit / MEMBER_PART_MEMORY_DIVISOR) {
                    partOffsets.push_back(i);
                    partBytes = 0;
                }
                partBytes += memberBytes[i];
            }
            partOffsets.push_back(windowPairs.size());
            membersPassed.assign(windowPairs.size(), 0);
            for (size_t part = 0; part + 1 < partOffsets.size(); part++) {
                partKeys.clear();
                for (size_t i = partOffsets[part]; i < partOffsets[part + 1]; i++) {
                    for (size_t member = memberStarts[i]; member < members.size() && members[member].repKey == windowPairs[i].memberKey; member++) {
                        partKeys.push_back(members[member].memberKey);
                    }
                }
                SORT_PARALLEL(partKeys.begin(), partKeys.end());
                partKeys.erase(std::unique(partKeys.begin(), partKeys.end()), partKeys.end());
                memberDbr->loadData(partKeys);
                // the pairs of a rep are next to each other, they share the query and the rounds of their member gates
                repGroups.clear();
                for (size_t i = partOffsets[part]; i < partOffsets[part + 1]; i++) {
                    if (i == partOffsets[part] || windowPairs[i].repKey != windowPairs[i - 1].repKey) {
                        repGroups.push_back(i);
                    }
                }
                repGroups.push_back(partOffsets[part + 1]);
                const size_t groupCount = repGroups.size() - 1;
#pragma omp parallel
                {
                    unsigned int thread_idx = 0;
#ifdef OPENMP
                    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                    Sequence query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
                    Sequence element(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
                    BlockAligner blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                                             &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                                             -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid());
                    Matcher matcher(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                                    par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid(), 0.0, par.zdrop);
                    const bool swipeLanes = (par.alignerMode != Parameters::ALIGNER_MODE_DEFAULT);
                    SwipeBatch swipe(par, scoring.subMat, scoring.evaluer, maxSeqLen);
                    ClusterMembers memberBuffers(swipeLanes ? &swipe : NULL, true);
#pragma omp for schedule(dynamic, 1)
                    for (size_t group = 0; group < groupCount; group++) {
                        memberBuffers.gateCount = 0;
                        for (size_t i = repGroups[group]; i < repGroups[group + 1]; i++) {
                            const ClusterPair &pair = windowPairs[i];
                            if (isAssigned(pair.repKey) || isAssigned(pair.memberKey)) {
                                continue;
                            }
                            MemberGate &gate = memberBuffers.addGate(i, 0);
                            for (size_t member = memberStarts[i]; member < members.size() && members[member].repKey == pair.memberKey; member++) {
                                gate.memberKeys.push_back(members[member].memberKey);
                            }
                        }
                        if (memberBuffers.gateCount == 0) {
                            continue;
                        }
                        const size_t repKey = windowPairs[repGroups[group]].repKey % paddedDbSize;
                        query.mapSequence(repKey, repKey, seqDbr.getData(repKey, thread_idx), seqDbr.getSeqLen(repKey));
                        blockAligner.setQuery(&query);
                        bool matcherInitialized = false;
                        swipe.startRep();
                        alignClusterMembers(par, blockAligner, matcher, matcherInitialized, query, element, memberDbr, swMode, thread_idx, false, NULL, memberBuffers);
                        for (size_t g = 0; g < memberBuffers.gateCount; g++) {
                            membersPassed[memberBuffers.gates[g].targetId] = memberBuffers.gates[g].live;
                        }
                    }
                }
            }
            // log the pairs whose members passed, in window order
            size_t acceptedCount = 0;
            for (size_t i = 0; i < windowPairs.size(); i++) {
                if (membersPassed[i]) {
                    windowPairs[acceptedCount++] = windowPairs[i];
                }
            }
            if (fwrite(windowPairs.data(), sizeof(ClusterPair), acceptedCount, acceptedHandle) != acceptedCount) {
                Debug(Debug::ERROR) << "Cannot write " << acceptedLog << "\n";
                EXIT(EXIT_FAILURE);
            }
            acceptedBytes += acceptedCount * sizeof(ClusterPair);
            MMseqsNode::publishLog(acceptedHandle, acceptedLog, nextRepKey, acceptedBytes);
            publishedBytes = acceptedBytes;
            publishTime = time(NULL);
            if (nextRepKey < orderKeyEnd) {
                readAssignedChanges();
            }
        }
        std::swap(window, nextWindow);
    }
    // readExchangeRange stops at an out of order block, so every block must be read to its end
    for (size_t i = 0; i < repBlocks.size(); i++) {
        if (repBlocks[i].pos != repBlocks[i].end || repBlocks[i].partsBytes != 0 || repBlocks[i].bufferPos != repBlocks[i].bufferEnd) {
            Debug(Debug::ERROR) << "Cannot read " << repTargets << ": its records are not in rep order\n";
            EXIT(EXIT_FAILURE);
        }
    }
    MMseqsNode::publishLog(acceptedHandle, acceptedLog, orderKeyEnd, acceptedBytes);
    if (fclose(acceptedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << acceptedLog << "\n";
        EXIT(EXIT_FAILURE);
    }
    MMseqsNode::closeExchangeFiles(repFiles, repTargets);
    seqDbr.close();
}

// an accepted pair from the log of a node and its bit in the taken bits of the step
struct AcceptedPair {
    ClusterPair pair;
    size_t takenBit;
    unsigned int node;

    static bool compareByRep(const AcceptedPair &first, const AcceptedPair &second) {
        return ClusterPair::compareByRep(first.pair, second.pair);
    }

    static bool compareByRepKey(const AcceptedPair &first, const AcceptedPair &second) {
        return first.pair.repKey < second.pair.repKey;
    }
};

// the taken bits of a log in whole words; carry holds the bits of the last partial word
struct TakenBits {
    FILE *file;
    size_t bytes;
    uint64_t carry;
    unsigned int carryBits;
};

static void appendTakenBits(TakenBits &taken, const uint64_t *bits, size_t bitCount, std::vector<uint64_t> &words, const std::string &fileName) {
    words.clear();
    for (size_t i = 0; i < bitCount; i += 64) {
        const unsigned int wordBits = std::min<size_t>(64, bitCount - i);
        taken.carry |= bits[i / 64] << taken.carryBits;
        if (taken.carryBits + wordBits >= 64) {
            words.push_back(taken.carry);
            taken.carry = (taken.carryBits > 0) ? bits[i / 64] >> (64 - taken.carryBits) : 0;
            taken.carryBits = taken.carryBits + wordBits - 64;
        } else {
            taken.carryBits += wordBits;
        }
    }
    if (fwrite(words.data(), sizeof(uint64_t), words.size(), taken.file) != words.size()) {
        Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    taken.bytes += words.size() * sizeof(uint64_t);
}

// --createdb-mode 3: the master clusters the accepted pairs in order key order; returns the member count
static size_t clusterAcceptedPairs(int threads, size_t dbSize, size_t memoryLimit, const std::vector<size_t> &groupKeyOffsets, size_t lagKeys, int mode, size_t paddedDbSize, size_t prefSizeBits,
                                   const std::string &accepted, const std::string &assignedChanges, const std::string &taken, const std::string &assignedKeys, double &clusterSeconds, double &waitSeconds) {
    const size_t assignedWords = (dbSize + 63) / 64;
    const size_t orderKeyEnd = (prefSizeBits - 1) * paddedDbSize + dbSize;
    const size_t orderWords = (orderKeyEnd + 63) / 64;
    uint64_t *assigned = new(std::nothrow) uint64_t[assignedWords];
    Util::checkAllocation(assigned, "Can not allocate assigned memory in align2clust");
    memset(assigned, 0, assignedWords * sizeof(uint64_t));
    std::vector<std::vector<DBKeyType> > threadAssignedKeys(threads);
    const auto isAssigned = [&](DBKeyType key) {
        return ((assigned[key / 64] >> (key % 64)) & 1) != 0;
    };
    const auto setAssigned = [&](DBKeyType key, std::vector<DBKeyType> &keys) {
        if ((__sync_fetch_and_or(&assigned[key / 64], 1ULL << (key % 64)) & (1ULL << (key % 64))) == 0) {
            keys.push_back(key);
        }
    };

    // per node: the first rep not done, the published log bytes and the clustered log bytes
    std::vector<unsigned char> nodeHasProgress(MMseqsNode::count, 0);
    std::vector<size_t> nodeRepKeys(MMseqsNode::count, 0);
    std::vector<size_t> nodeLogBytes(MMseqsNode::count, 0);
    std::vector<size_t> nodeReadBytes(MMseqsNode::count, 0);
    std::vector<TakenBits> nodeTaken(MMseqsNode::count);
    std::vector<std::vector<uint64_t> > nodeTakenBits(MMseqsNode::count);
    std::vector<size_t> nodeTakenCounts(MMseqsNode::count, 0);
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        const TakenBits takenBits = { FileUtil::openAndDelete((taken + "." + SSTR(node)).c_str(), "w"), 0, 0, 0 };
        nodeTaken[node] = takenBits;
    }
    FILE *changesHandle = FileUtil::openAndDelete(assignedChanges.c_str(), "w");
    size_t changesBytes = 0;
    // a step clusters a bounded number of pairs, read from the logs in order of their next rep, unless its first rep spans more
    const size_t stepPairs = std::max<size_t>(1, memoryLimit / STEP_PAIRS_MEMORY_DIVISOR / sizeof(AcceptedPair));
    const size_t partBytes = std::max<size_t>(1, std::min(stepPairs * sizeof(ClusterPair), memoryLimit / LOG_PART_MEMORY_DIVISOR) / sizeof(ClusterPair)) * sizeof(ClusterPair);
    std::vector<AcceptedPair> readPairs;
    std::vector<size_t> nodeFirstPairs(MMseqsNode::count, 0);
    std::vector<size_t> nodeEndPairs(MMseqsNode::count, 0);
    // per node: no unread pair of its log has a smaller rep
    std::vector<size_t> nodeNextRepKeys(MMseqsNode::count, 0);
    std::vector<size_t> nodeOrder(MMseqsNode::count);
    std::vector<char> logData;
    std::vector<size_t> repOffsets;
    std::vector<std::vector<size_t> > threadRepOffsets(threads);
    std::vector<unsigned char> pairIsFree;
    std::vector<size_t> repKeyOffsets;
    std::vector<std::pair<DBKeyType, size_t> > keyReps;
    std::vector<unsigned char> repSharesKey;
    std::vector<uint64_t> words;
    std::vector<uint64_t> changes;
    std::vector<std::vector<size_t> > threadSectionCounts(threads, std::vector<size_t>(MMseqsNode::count + 1));
    std::vector<size_t> sectionOffsets(MMseqsNode::count + 2);
    size_t memberCount = 0;
    // --cluster-mode 0: the open sets in a heap as in clusterThreadFuncSetcover, the sets of two as pairs
    std::vector<SetCoverCandidate> candidates;
    std::vector<DBKeyType> memberPool;
    std::vector<ClusterPair> pairSets;
    std::vector<size_t> repSetOffsets;
    size_t liveMemberCount = 0;
    // the sets get the memory left by the assigned bits and the steps, reserved once
    const size_t setCoverBytes = memoryLimit - std::min(memoryLimit, assignedWords * sizeof(uint64_t) + memoryLimit / STEP_MEMORY_DIVISOR);
    if (mode == Parameters::SET_COVER) {
        memberPool.reserve(setCoverBytes / sizeof(DBKeyType));
        candidates.reserve(setCoverBytes / sizeof(SetCoverCandidate));
        pairSets.reserve(setCoverBytes / sizeof(ClusterPair));
    }
    // set cover sends a taken pair to the node that aligned its rep
    const auto takePair = [&](DBKeyType repKey, DBKeyType memberKey) {
        const ClusterPair pair = { repKey, memberKey };
        TakenBits &nodeFile = nodeTaken[getAlignNode(groupKeyOffsets, repKey)];
        if (fwrite(&pair, sizeof(ClusterPair), 1, nodeFile.file) != 1) {
            Debug(Debug::ERROR) << "Cannot write " << taken << "\n";
            EXIT(EXIT_FAILURE);
        }
        nodeFile.bytes += sizeof(ClusterPair);
        memberCount++;
    };
    Timer clusterTimer;
    unsigned int polls = 0;
    for (size_t clusteredKey = 0; clusteredKey < orderKeyEnd;) {
        Timer pollTimer;
        bool hasNewProgress = false;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const std::string log = accepted + "." + SSTR(node);
            size_t repKey;
            size_t logBytes;
            if ((nodeHasProgress[node] && nodeRepKeys[node] == orderKeyEnd) || MMseqsNode::readLogProgress(log, repKey, logBytes) == false
                || (nodeHasProgress[node] && repKey <= nodeRepKeys[node] && logBytes <= nodeLogBytes[node])) {
                continue;
            }
            if (logBytes % sizeof(ClusterPair) != 0 || logBytes < nodeLogBytes[node] || repKey < nodeRepKeys[node]) {
                Debug(Debug::ERROR) << "Cannot read " << log << ".progress\n";
                EXIT(EXIT_FAILURE);
            }
            hasNewProgress = true;
            nodeHasProgress[node] = 1;
            nodeRepKeys[node] = repKey;
            nodeLogBytes[node] = logBytes;
        }
        size_t doneRepKey = orderKeyEnd;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            doneRepKey = std::min(doneRepKey, nodeHasProgress[node] ? nodeRepKeys[node] : 0);
        }
        if (doneRepKey <= clusteredKey) {
            if (hasNewProgress == false) {
                MMseqsNode::sleepBeforePoll(polls++);
            }
            waitSeconds += pollTimer.getTimediff();
            continue;
        }
        polls = 0;

        // read every log up to doneRepKey; once the step holds stepPairs, a log ends the step before its next rep, but the rep of clusteredKey is read whole
        readPairs.clear();
        size_t stepRepKey = doneRepKey;
        const auto isCut = [&](size_t repKey, size_t readRepKey) {
            return readPairs.size() >= stepPairs && repKey > readRepKey;
        };
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            nodeOrder[node] = node;
        }
        std::stable_sort(nodeOrder.begin(), nodeOrder.end(), [&](size_t first, size_t second) { return nodeNextRepKeys[first] < nodeNextRepKeys[second]; });
        for (size_t n = 0; n < MMseqsNode::count; n++) {
            const size_t node = nodeOrder[n];
            const std::string log = accepted + "." + SSTR(node);
            nodeFirstPairs[node] = readPairs.size();
            // a log is read only if it may have an unread pair before doneRepKey and is not cut at its next rep
            const bool readsLog = nodeNextRepKeys[node] < doneRepKey && isCut(nodeNextRepKeys[node], clusteredKey) == false;
            size_t nodeRepKey = doneRepKey;
            if (readsLog) {
                nodeNextRepKeys[node] = nodeRepKeys[node];
            } else {
                nodeRepKey = std::min(doneRepKey, nodeNextRepKeys[node]);
            }
            size_t readRepKey = clusteredKey;
            for (size_t from = nodeReadBytes[node]; readsLog && from < nodeLogBytes[node];) {
                const size_t to = std::min(nodeLogBytes[node], from + partBytes);
                logData.resize(to - from);
                MMseqsNode::readLog(log, from, to, logData.data());
                for (size_t i = 0; from < to; i++, from += sizeof(ClusterPair)) {
                    AcceptedPair next;
                    memcpy(&next.pair, logData.data() + i * sizeof(ClusterPair), sizeof(ClusterPair));
                    next.takenBit = readPairs.size() - nodeFirstPairs[node];
                    next.node = static_cast<unsigned int>(node);
                    if (next.pair.repKey >= doneRepKey || isCut(next.pair.repKey, readRepKey)) {
                        nodeNextRepKeys[node] = next.pair.repKey;
                        nodeRepKey = std::min<size_t>(doneRepKey, next.pair.repKey);
                        break;
                    }
                    readPairs.push_back(next);
                    readRepKey = next.pair.repKey;
                }
                if (from < to) {
                    break;
                }
            }
            nodeEndPairs[node] = readPairs.size();
            stepRepKey = std::min(stepRepKey, nodeRepKey);
        }
        // cluster the pairs before stepRepKey; the others are read again in a later step
        size_t pairCount = 0;
        for (size_t n = 0; n < MMseqsNode::count; n++) {
            const size_t node = nodeOrder[n];
            const AcceptedPair firstNotDone = { { static_cast<DBKeyType>(stepRepKey), static_cast<DBKeyType>(stepRepKey) }, 0, 0 };
            const size_t nodeEnd = std::lower_bound(readPairs.begin() + nodeFirstPairs[node], readPairs.begin() + nodeEndPairs[node], firstNotDone, AcceptedPair::compareByRepKey) - readPairs.begin();
            if (nodeEnd < nodeEndPairs[node]) {
                nodeNextRepKeys[node] = readPairs[nodeEnd].pair.repKey;
            }
            nodeTakenCounts[node] = nodeEnd - nodeFirstPairs[node];
            nodeReadBytes[node] += nodeTakenCounts[node] * sizeof(ClusterPair);
            nodeTakenBits[node].assign((nodeTakenCounts[node] + 63) / 64, 0);
            if (pairCount != nodeFirstPairs[node]) {
                memmove(&readPairs[pairCount], &readPairs[nodeFirstPairs[node]], nodeTakenCounts[node] * sizeof(AcceptedPair));
            }
            pairCount += nodeTakenCounts[node];
        }
        SORT_PARALLEL(readPairs.begin(), readPairs.begin() + pairCount, AcceptedPair::compareByRep);
        const AcceptedPair *logged = readPairs.data();
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            threadRepOffsets[thread_idx].clear();
#pragma omp for schedule(static)
            for (size_t i = 0; i < pairCount; i++) {
                if (i == 0 || logged[i].pair.repKey != logged[i - 1].pair.repKey) {
                    threadRepOffsets[thread_idx].push_back(i);
                }
            }
        }
        repOffsets.clear();
        for (int thread = 0; thread < threads; thread++) {
            repOffsets.insert(repOffsets.end(), threadRepOffsets[thread].begin(), threadRepOffsets[thread].end());
        }
        repOffsets.push_back(pairCount);
        const size_t repCount = repOffsets.size() - 1;

        // a pair with an assigned rep or member stays out, as bits are never cleared
        pairIsFree.resize(pairCount);
        repKeyOffsets.assign(repCount + 1, 0);
#pragma omp parallel for schedule(dynamic, 1024)
        for (size_t rep = 0; rep < repCount; rep++) {
            const bool repIsFree = isAssigned(logged[repOffsets[rep]].pair.repKey % paddedDbSize) == false;
            size_t freeCount = 0;
            for (size_t i = repOffsets[rep]; i < repOffsets[rep + 1]; i++) {
                pairIsFree[i] = repIsFree && isAssigned(logged[i].pair.memberKey) == false;
                freeCount += pairIsFree[i];
            }
            repKeyOffsets[rep + 1] = (freeCount > 0) ? freeCount + 1 : 0;
        }
        for (size_t rep = 0; rep < repCount; rep++) {
            repKeyOffsets[rep + 1] += repKeyOffsets[rep];
        }
        if (mode == Parameters::SET_COVER) {
            // a set waits until no later rep can have a larger one: the sets of two as pairs, larger ones in the heap
            repSetOffsets.resize(repCount);
            size_t setMembers = 0;
            size_t setCount = 0;
            size_t setPairs = 0;
            for (size_t rep = 0; rep < repCount; rep++) {
                const size_t setSize = repKeyOffsets[rep + 1] - repKeyOffsets[rep];
                repSetOffsets[rep] = (setSize == 2) ? setPairs++ : setMembers;
                if (setSize > 2) {
                    setMembers += setSize;
                    setCount++;
                }
            }
            if ((memberPool.size() + setMembers) * sizeof(DBKeyType) + (candidates.size() + setCount) * sizeof(SetCoverCandidate)
                + (pairSets.size() + setPairs) * sizeof(ClusterPair) > setCoverBytes) {
                Debug(Debug::ERROR) << "The master needs more than " << setCoverBytes << " bytes for the sets of --cluster-mode " << Parameters::SET_COVER
                                    << ". Run with --cluster-mode " << Parameters::GREEDY << " or on nodes with more memory\n";
                EXIT(EXIT_FAILURE);
            }
            const size_t poolOffset = memberPool.size();
            const size_t pairOffset = pairSets.size();
            memberPool.resize(poolOffset + setMembers);
            pairSets.resize(pairOffset + setPairs);
#pragma omp parallel for schedule(dynamic, 1024)
            for (size_t rep = 0; rep < repCount; rep++) {
                const size_t setSize = repKeyOffsets[rep + 1] - repKeyOffsets[rep];
                if (setSize == 0) {
                    continue;
                }
                const DBKeyType repKey = logged[repOffsets[rep]].pair.repKey % paddedDbSize;
                size_t member = poolOffset + repSetOffsets[rep];
                if (setSize > 2) {
                    memberPool[member++] = repKey;
                }
                for (size_t i = repOffsets[rep]; i < repOffsets[rep + 1]; i++) {
                    if (pairIsFree[i] && setSize == 2) {
                        const ClusterPair pairSet = { repKey, logged[i].pair.memberKey };
                        pairSets[pairOffset + repSetOffsets[rep]] = pairSet;
                    } else if (pairIsFree[i]) {
                        memberPool[member++] = logged[i].pair.memberKey;
                    }
                }
            }
            for (size_t rep = 0; rep < repCount; rep++) {
                if (repKeyOffsets[rep + 1] - repKeyOffsets[rep] > 2) {
                    const SetCoverCandidate candidate = { logged[repOffsets[rep]].pair.repKey % paddedDbSize, repKeyOffsets[rep + 1] - repKeyOffsets[rep], poolOffset + repSetOffsets[rep] };
                    candidates.push_back(candidate);
                    std::push_heap(candidates.begin(), candidates.end(), setCoverComparator);
                }
            }
            liveMemberCount += setMembers;
            // the reps from stepRepKey on have fewer than 2^(prefSizeBit + 1) targets, so no larger set can come
            const size_t maxSetSize = (stepRepKey < orderKeyEnd) ? static_cast<size_t>(2) << (prefSizeBits - 1 - stepRepKey / paddedDbSize) : 0;
            while (candidates.empty() == false && candidates.front().memberCount > maxSetSize) {
                std::pop_heap(candidates.begin(), candidates.end(), setCoverComparator);
                SetCoverCandidate candidate = candidates.back();
                candidates.pop_back();
                liveMemberCount -= candidate.memberCount;
                if (isAssigned(candidate.representativeId)) {
                    continue;
                }
                DBKeyType *members = memberPool.data() + candidate.memberOffset;
                size_t validCount = 0;
                for (size_t i = 0; i < candidate.memberCount; i++) {
                    if (isAssigned(members[i]) == false) {
                        members[validCount++] = members[i];
                    }
                }
                if (validCount <= 1) {
                    continue;
                }
                // the rep stays first
                if (validCount == 2) {
                    const ClusterPair pairSet = { members[0], members[1] };
                    pairSets.push_back(pairSet);
                    continue;
                }
                if (validCount != candidate.memberCount) {
                    candidate.memberCount = validCount;
                    liveMemberCount += validCount;
                    candidates.push_back(candidate);
                    std::push_heap(candidates.begin(), candidates.end(), setCoverComparator);
                    continue;
                }
                for (size_t i = 0; i < candidate.memberCount; i++) {
                    setAssigned(members[i], threadAssignedKeys[0]);
                    if (members[i] != candidate.representativeId) {
                        takePair(candidate.representativeId, members[i]);
                    }
                }
            }
            compactSetCoverMemberPool(candidates, memberPool, liveMemberCount);
            // the sets of two are the smallest, so the last step takes them after the heap, the larger rep first
            if (stepRepKey == orderKeyEnd) {
                SORT_PARALLEL(pairSets.begin(), pairSets.end(), ClusterPair::compareByRep);
                for (size_t i = pairSets.size(); i-- > 0;) {
                    if (isAssigned(pairSets[i].repKey) == false && isAssigned(pairSets[i].memberKey) == false) {
                        setAssigned(pairSets[i].repKey, threadAssignedKeys[0]);
                        setAssigned(pairSets[i].memberKey, threadAssignedKeys[0]);
                        takePair(pairSets[i].repKey, pairSets[i].memberKey);
                    }
                }
                std::vector<ClusterPair>().swap(pairSets);
            }
        } else {
            keyReps.resize(repKeyOffsets[repCount]);
#pragma omp parallel for schedule(dynamic, 1024)
            for (size_t rep = 0; rep < repCount; rep++) {
                size_t key = repKeyOffsets[rep];
                if (key == repKeyOffsets[rep + 1]) {
                    continue;
                }
                keyReps[key++] = std::make_pair(logged[repOffsets[rep]].pair.repKey, rep);
                for (size_t i = repOffsets[rep]; i < repOffsets[rep + 1]; i++) {
                    if (pairIsFree[i]) {
                        keyReps[key++] = std::make_pair(logged[i].pair.memberKey, rep);
                    }
                }
            }
            // every rep after the first of a key shares it with an earlier rep of the step
            SORT_PARALLEL(keyReps.begin(), keyReps.end());
            repSharesKey.assign(repCount, 0);
#pragma omp parallel for schedule(static)
            for (size_t i = 1; i < keyReps.size(); i++) {
                if (keyReps[i].first == keyReps[i - 1].first) {
                    __atomic_store_n(&repSharesKey[keyReps[i].second], 1, __ATOMIC_RELAXED);
                }
            }
            // the reps that share no key take all their free members at once
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
#pragma omp for schedule(dynamic, 1024) reduction(+:memberCount)
                for (size_t rep = 0; rep < repCount; rep++) {
                    if (repKeyOffsets[rep + 1] == repKeyOffsets[rep] || repSharesKey[rep]) {
                        continue;
                    }
                    setAssigned(logged[repOffsets[rep]].pair.repKey, threadAssignedKeys[thread_idx]);
                    for (size_t i = repOffsets[rep]; i < repOffsets[rep + 1]; i++) {
                        if (pairIsFree[i]) {
                            setAssigned(logged[i].pair.memberKey, threadAssignedKeys[thread_idx]);
                            __sync_fetch_and_or(&nodeTakenBits[logged[i].node][logged[i].takenBit / 64], 1ULL << (logged[i].takenBit % 64));
                            memberCount++;
                        }
                    }
                }
            }
            // the sharing reps take in key order; no rep that does not share holds a key of theirs
            for (size_t rep = 0; rep < repCount; rep++) {
                const DBKeyType repKey = logged[repOffsets[rep]].pair.repKey;
                if (repKeyOffsets[rep + 1] == repKeyOffsets[rep] || repSharesKey[rep] == 0 || isAssigned(repKey)) {
                    continue;
                }
                for (size_t i = repOffsets[rep]; i < repOffsets[rep + 1]; i++) {
                    if (i + PREFETCH_PAIRS < pairCount) {
                        __builtin_prefetch(&assigned[logged[i + PREFETCH_PAIRS].pair.memberKey / 64]);
                    }
                    const DBKeyType memberKey = logged[i].pair.memberKey;
                    if (pairIsFree[i] && isAssigned(memberKey) == false) {
                        setAssigned(repKey, threadAssignedKeys[0]);
                        setAssigned(memberKey, threadAssignedKeys[0]);
                        nodeTakenBits[logged[i].node][logged[i].takenBit / 64] |= 1ULL << (logged[i].takenBit % 64);
                        memberCount++;
                    }
                }
            }
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                appendTakenBits(nodeTaken[node], nodeTakenBits[node].data(), nodeTakenCounts[node], words, taken + "." + SSTR(node));
            }
        }
        const size_t clusteredBefore = clusteredKey;
        clusteredKey = stepRepKey;

        // the step keys by aligning node after the ring keys of all nodes, and the entering ring words
        const auto isRingKey = [&](size_t key) {
            const size_t orderKey = key + ((clusteredKey > key) ? (clusteredKey - key + paddedDbSize - 1) / paddedDbSize * paddedDbSize : 0);
            return orderKey < orderKeyEnd && orderKey < clusteredBefore + lagKeys;
        };
        const size_t enteringFirstWord = std::min(orderWords, std::max(clusteredBefore + lagKeys, clusteredKey) / 64);
        const size_t enteringWords = std::min(orderWords, (clusteredKey + lagKeys + 63) / 64) - enteringFirstWord;
#pragma omp parallel for schedule(static)
        for (int thread = 0; thread < threads; thread++) {
            std::vector<size_t> &sectionCounts = threadSectionCounts[thread];
            sectionCounts.assign(MMseqsNode::count + 1, 0);
            for (size_t i = 0; i < threadAssignedKeys[thread].size(); i++) {
                const DBKeyType key = threadAssignedKeys[thread][i];
                sectionCounts[getAlignNode(groupKeyOffsets, key) + 1]++;
                sectionCounts[0] += isRingKey(key);
            }
        }
        // turn the counts into where the keys of every thread start in every section, the ring section first
        sectionOffsets[0] = 0;
        for (size_t section = 0; section <= MMseqsNode::count; section++) {
            sectionOffsets[section + 1] = sectionOffsets[section];
            for (int thread = 0; thread < threads; thread++) {
                const size_t count = threadSectionCounts[thread][section];
                threadSectionCounts[thread][section] = sectionOffsets[section + 1];
                sectionOffsets[section + 1] += count;
            }
        }
        const size_t headerWords = 4 + MMseqsNode::count + 1;
        changes.resize(headerWords + enteringWords + sectionOffsets[MMseqsNode::count + 1]);
        changes[0] = clusteredKey;
        changes[1] = enteringFirstWord;
        changes[2] = enteringWords;
        changes[3] = sectionOffsets[1];
        for (size_t node = 0; node <= MMseqsNode::count; node++) {
            changes[4 + node] = sectionOffsets[node + 1] - sectionOffsets[1];
        }
        for (size_t i = 0; i < enteringWords; i++) {
            changes[headerWords + i] = assigned[(enteringFirstWord + i) % assignedWords];
        }
#pragma omp parallel for schedule(static)
        for (int thread = 0; thread < threads; thread++) {
            std::vector<size_t> &sectionPositions = threadSectionCounts[thread];
            for (size_t i = 0; i < threadAssignedKeys[thread].size(); i++) {
                const DBKeyType key = threadAssignedKeys[thread][i];
                changes[headerWords + enteringWords + sectionPositions[getAlignNode(groupKeyOffsets, key) + 1]++] = key;
                if (isRingKey(key)) {
                    changes[headerWords + enteringWords + sectionPositions[0]++] = key;
                }
            }
            threadAssignedKeys[thread].clear();
        }
        if (fwrite(changes.data(), sizeof(uint64_t), changes.size(), changesHandle) != changes.size()) {
            Debug(Debug::ERROR) << "Cannot write " << assignedChanges << "\n";
            EXIT(EXIT_FAILURE);
        }
        changesBytes += changes.size() * sizeof(uint64_t);
        MMseqsNode::publishLog(changesHandle, assignedChanges, clusteredKey, changesBytes);
    }
    clusterSeconds = clusterTimer.getTimediff() - waitSeconds;
    // every node is done with both logs
    if (fclose(changesHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << assignedChanges << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::remove(assignedChanges.c_str());
    FileUtil::remove((assignedChanges + ".progress").c_str());
    // every node reads the assigned bits of its keys; an assigned key is no singleton
    FILE *assignedHandle = FileUtil::openAndDelete(assignedKeys.c_str(), "w");
    if (fwrite(assigned, sizeof(uint64_t), assignedWords, assignedHandle) != assignedWords) {
        Debug(Debug::ERROR) << "Cannot write " << assignedKeys << "\n";
        EXIT(EXIT_FAILURE);
    }
    MMseqsNode::publishLog(assignedHandle, assignedKeys, dbSize, assignedWords * sizeof(uint64_t));
    if (fclose(assignedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << assignedKeys << "\n";
        EXIT(EXIT_FAILURE);
    }
    delete[] assigned;
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        const std::string takenFile = taken + "." + SSTR(node);
        if (nodeTaken[node].carryBits > 0 && fwrite(&nodeTaken[node].carry, sizeof(uint64_t), 1, nodeTaken[node].file) != 1) {
            Debug(Debug::ERROR) << "Cannot write " << takenFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        MMseqsNode::publishLog(nodeTaken[node].file, takenFile, dbSize, nodeTaken[node].bytes + ((nodeTaken[node].carryBits > 0) ? sizeof(uint64_t) : 0));
        if (fclose(nodeTaken[node].file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << takenFile << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    return memberCount;
}

// --createdb-mode 3: the nodes align the reps of their targets while the master clusters
static int align2clustLinfinity(Parameters &par) {
    Timer timer;
    // an old dbtype would mark the result as done too early
    if (MMseqsNode::isMaster() && FileUtil::fileExists((par.db3 + ".dbtype").c_str())) {
        FileUtil::remove((par.db3 + ".dbtype").c_str());
    }
    DBReader<DBKeyType> seqDbr(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    seqDbr.open(DBReader<DBKeyType>::NOSORT);
    const std::vector<size_t> nodeKeyOffsets = seqDbr.getFileKeyOffsets(MMseqsNode::count);
    const size_t dbSize = seqDbr.getSize();
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const size_t nodeMemoryLimit = Util::computeMemory(par.splitMemoryLimit);
    // on one node the master also clusters, with part of the memory and a few threads
    const size_t memoryLimit = nodeMemoryLimit / ((MMseqsNode::isMaster() && getFirstAlignNode() == 0) ? MASTER_MEMORY_DIVISOR : 1);
    const int clusterThreads = (getFirstAlignNode() > 0) ? par.threads : std::min(CLUSTER_THREADS, std::max(1, par.threads / 4));

    // a clusthashfast subset has bits; its sixth index column counts the representatives for the e-values
    const bool hasRepresentativeBits = FileUtil::fileExists((par.db1 + ".bits." + SSTR(MMseqsNode::rank)).c_str());
    size_t representatives = 0;
    size_t representativeResidues = 0;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        representatives += seqDbr.getRangeRepresentativeCount(range);
        representativeResidues += seqDbr.getRangeRepresentativeCount(range) * (index[range].length - 2);
    }

    // remove the progress of an earlier run before the memory barrier
    const std::string accepted = par.db3 + "_swap_1";
    const std::string assignedChanges = par.db3 + "_assigned";
    const std::string taken = par.db3 + "_taken";
    const std::string assignedKeys = par.db3 + "_assigned.bits";
    std::vector<std::string> progressFiles;
    progressFiles.push_back(accepted + "." + SSTR(MMseqsNode::rank) + ".progress");
    progressFiles.push_back(taken + "." + SSTR(MMseqsNode::rank) + ".progress");
    if (MMseqsNode::isMaster()) {
        progressFiles.push_back(assignedChanges + ".progress");
        progressFiles.push_back(assignedKeys + ".progress");
    }
    for (size_t i = 0; i < progressFiles.size(); i++) {
        if (FileUtil::fileExists(progressFiles[i].c_str())) {
            FileUtil::remove(progressFiles[i].c_str());
        }
    }

    // all nodes use the smallest memory limit, so they agree on the splits and the ring and fail together
    const size_t smallestMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db3 + ".memory", nodeMemoryLimit);
    const size_t assignedWords = (dbSize + 63) / 64;
    if (assignedWords * sizeof(uint64_t) > smallestMemoryLimit / ASSIGNED_BITS_MEMORY_DIVISOR) {
        Debug(Debug::ERROR) << "The master needs " << assignedWords * sizeof(uint64_t) << " bytes for a bit of each of the " << dbSize << " keys, more than half of the smallest memory limit of "
                            << smallestMemoryLimit << " bytes. Run on nodes with more memory\n";
        EXIT(EXIT_FAILURE);
    }
    // --cluster-mode 0 orders the reps by order key, larger target counts first (see prefSizeBit)
    const size_t paddedDbSize = assignedWords * 64;
    const size_t prefSizeBits = (par.clusteringMode == Parameters::SET_COVER) ? 64 - __builtin_clzll(dbSize | 1) : 1;
    const size_t maxWindowBytes = std::max<size_t>(1, smallestMemoryLimit / WINDOW_MEMORY_DIVISOR);
    const size_t lagKeys = std::max<size_t>(64, smallestMemoryLimit / RING_MEMORY_DIVISOR * 8 / 64 * 64);

    // every partition of the prefilter is a group, whose key range starts at groupKeyOffsets; taken in turn, the aligning nodes align close keys at the same time
    const std::string partitionsFile = par.db2 + ".partitions";
    if (FileUtil::fileExists(partitionsFile.c_str()) == false) {
        Debug(Debug::ERROR) << "Cannot find " << partitionsFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    const std::string partitionsData = MMseqsNode::waitForFile(partitionsFile);
    std::vector<PrefPartition> partitions;
    for (char *line = (char *) partitionsData.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[5];
        if (Util::getWordsOfLine(line, columns, 5) < 5) {
            Debug(Debug::ERROR) << "Cannot read " << partitionsFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        PrefPartition partition;
        partition.firstKey = Util::fast_atoi<size_t>(columns[0]);
        partition.prefOffset = Util::fast_atoi<size_t>(columns[1]);
        partition.prefBytes = Util::fast_atoi<size_t>(columns[2]);
        partition.cost = Util::fast_atoi<size_t>(columns[3]);
        partition.node = Util::fast_atoi<size_t>(columns[4]);
        partition.endKey = dbSize;
        if (partitions.empty() == false) {
            partitions.back().endKey = std::min(partitions.back().endKey, partition.firstKey);
        }
        // a partition without keys may start at the end
        if (partition.firstKey > dbSize || (partition.firstKey == dbSize && partition.cost > 0)) {
            Debug(Debug::ERROR) << partitionsFile << " has a partition of key " << partition.firstKey << ", which is not in " << par.db1 << "\n";
            EXIT(EXIT_FAILURE);
        }
        // the groups are key ranges, and alignRepTargets reads the flushes of a node as one block in rep order
        if (partitions.empty() == false && partition.firstKey < partitions.back().firstKey) {
            Debug(Debug::ERROR) << partitionsFile << " has the partition of key " << partition.firstKey << " out of key order\n";
            EXIT(EXIT_FAILURE);
        }
        partitions.push_back(partition);
    }
    std::vector<size_t> groupKeyOffsets(1, 0);
    std::vector<PrefPartition> nodePartitions;
    for (size_t i = 0; i < partitions.size(); i++) {
        if (i > 0) {
            groupKeyOffsets.push_back(partitions[i].firstKey);
        }
        // a partition without targets has nothing to align
        if (getGroupNode(i) == MMseqsNode::rank && partitions[i].cost > 0) {
            nodePartitions.push_back(partitions[i]);
        }
    }
    groupKeyOffsets.push_back(dbSize);
    std::vector<size_t> nodeOwnKeys(MMseqsNode::count, 0);
    for (size_t group = 0; group + 1 < groupKeyOffsets.size(); group++) {
        nodeOwnKeys[getGroupNode(group)] += groupKeyOffsets[group + 1] - groupKeyOffsets[group];
    }
    const size_t ownKeys = *std::max_element(nodeOwnKeys.begin(), nodeOwnKeys.end());
    if ((ownKeys + 63) / 64 * sizeof(uint64_t) > smallestMemoryLimit / OWN_BITS_MEMORY_DIVISOR) {
        Debug(Debug::ERROR) << "An aligning node needs " << (ownKeys + 63) / 64 * sizeof(uint64_t) << " bytes for a bit of each of the " << ownKeys
                            << " keys of its groups, more than an eighth of the smallest memory limit of " << smallestMemoryLimit << " bytes. Run on more nodes\n";
        EXIT(EXIT_FAILURE);
    }

    // --filter-cludb-file: a node requests the members of the clusters of its targets from the nodes of their keys
    const std::string memberRequests = par.db3 + "_swap_3";
    const std::string membersExchange = par.db3 + "_swap_4";
    MMseqsNode::ClusterPairWriter *memberRequestWriter = NULL;
    if (par.filterCluDBFile.empty() == false) {
        memberRequestWriter = new MMseqsNode::ClusterPairWriter(memberRequests, std::max<size_t>(1, memoryLimit / CLUSTERS_MEMORY_DIVISOR / sizeof(ClusterPair)),
                                                                ClusterPair::compareByRep, nodeKeyOffsets);
    }
    const std::string repTargets = par.db3 + "_swap_0";
    const std::string repTargetsIndex = writeRepTargets(par, seqDbr, nodePartitions, maxWindowBytes, memoryLimit, paddedDbSize, prefSizeBits, repTargets, memberRequestWriter);
    Debug(Debug::INFO) << "Time for writing the rep targets: " << timer.lap() << "\n";

    // --filter-cludb-file: the aligning nodes get and keep the members of the clusters of their targets
    std::vector<ClusterPair> members;
    DBReader<DBKeyType> *memberDbr = NULL;
    if (memberRequestWriter != NULL) {
        const std::string memberRequestsIndex = memberRequestWriter->close();
        delete memberRequestWriter;
        const std::string membersIndex = MMseqsNode::sendClusterMembers(par.filterCluDBFile, memberRequestsIndex, memberRequests, membersExchange);
        // every node has read the requests
        FileUtil::remove((memberRequests + "." + SSTR(MMseqsNode::rank)).c_str());
        std::vector<MMseqsNode::ExchangeBlock> memberBlocks;
        std::vector<FILE *> memberFiles;
        MMseqsNode::openExchange(membersIndex, membersExchange, memberBlocks, memberFiles);
        std::vector<size_t> blockOffsets(memberBlocks.size() + 1, 0);
        for (size_t i = 0; i < memberBlocks.size(); i++) {
            blockOffsets[i + 1] = blockOffsets[i] + memberBlocks[i].end - memberBlocks[i].pos;
        }
        if (blockOffsets.back() % sizeof(ClusterPair) != 0) {
            Debug(Debug::ERROR) << "Cannot read " << membersExchange << ": a record ends after its block\n";
            EXIT(EXIT_FAILURE);
        }
        if (blockOffsets.back() > smallestMemoryLimit / MEMBERS_MEMORY_DIVISOR) {
            Debug(Debug::ERROR) << "An aligning node needs " << blockOffsets.back() << " bytes for the members of the clusters of its targets, more than a quarter of the smallest memory limit of "
                                << smallestMemoryLimit << " bytes. Run on more nodes\n";
            EXIT(EXIT_FAILURE);
        }
        members.resize(blockOffsets.back() / sizeof(ClusterPair));
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < memberBlocks.size(); i++) {
            FileUtil::preadAll(fileno(memberBlocks[i].file), membersExchange.c_str(), reinterpret_cast<char *>(members.data()) + blockOffsets[i], blockOffsets[i + 1] - blockOffsets[i], memberBlocks[i].pos);
        }
        MMseqsNode::closeExchangeFiles(memberFiles, membersExchange);
        SORT_PARALLEL(members.begin(), members.end(), ClusterPair::compareByRep);
        memberDbr = new DBReader<DBKeyType>(par.filterSeqDBFile.c_str(), (par.filterSeqDBFile + ".index").c_str(), par.threads,
                                            DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        memberDbr->open(DBReader<DBKeyType>::NOSORT);
        Debug(Debug::INFO) << "Time for sending the cluster members: " << timer.lap() << "\n";
    }
    Timer phaseTimer;

    // the master clusters while the nodes align; on several nodes its log stays empty
    size_t memberCount = 0;
    double clusterSeconds = 0;
    double masterWaitSeconds = 0;
    std::thread clusterThread;
    if (MMseqsNode::isMaster()) {
        clusterThread = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(clusterThreads);
#endif
            memberCount = clusterAcceptedPairs(clusterThreads, dbSize, memoryLimit, groupKeyOffsets, lagKeys, par.clusteringMode, paddedDbSize, prefSizeBits,
                                               accepted, assignedChanges, taken, assignedKeys, clusterSeconds, masterWaitSeconds);
        });
    }
    std::string phaseValues;
    Timer phaseValueTimer;
    double waitSeconds = 0;
    size_t alignedPairs = 0;
    alignRepTargets(par, seqDbr, groupKeyOffsets, lagKeys, memoryLimit, representativeResidues, paddedDbSize, prefSizeBits, repTargetsIndex, repTargets, accepted, assignedChanges, members, memberDbr,
                    waitSeconds, alignedPairs);
    FileUtil::remove((repTargets + "." + SSTR(MMseqsNode::rank)).c_str());
    if (MMseqsNode::rank >= getFirstAlignNode()) {
        MMseqsNode::addPhaseTime(phaseValues, "aligning the rep targets", phaseValueTimer);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the master", waitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Aligned pairs", SSTR(alignedPairs));
        if (par.alignerMode != Parameters::ALIGNER_MODE_DEFAULT) {
            printSwipeLanes();
        }
    }

    // once the master is done, every node sends the taken pairs of its log to their rep nodes
    const std::string takenFile = taken + "." + SSTR(MMseqsNode::rank);
    size_t takenKey;
    size_t takenBytes;
    MMseqsNode::waitForLog(takenFile, takenKey, takenBytes);
    if (MMseqsNode::isMaster()) {
        clusterThread.join();
        Debug(Debug::INFO) << "Number of clusters: " << representatives - memberCount << "\n";
        MMseqsNode::addPhaseSeconds(phaseValues, "clustering the accepted pairs", clusterSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the aligning nodes", masterWaitSeconds);
    }
    Debug(Debug::INFO) << "Time for aligning: " << phaseTimer.lap() << "\n";
    phaseValueTimer.reset();
    const size_t nodeFirstKey = nodeKeyOffsets[MMseqsNode::rank];
    const size_t nodeKeys = nodeKeyOffsets[MMseqsNode::rank + 1] - nodeFirstKey;
    const size_t nodeWords = (nodeKeys + 63) / 64;
    const std::string clusters = par.db3 + "_swap_2";
    MMseqsNode::ClusterPairWriter clustersWriter(clusters, std::max<size_t>(64, memoryLimit / CLUSTERS_MEMORY_DIVISOR / sizeof(ClusterPair)), ClusterPair::compareByRep, nodeKeyOffsets);
    const std::string acceptedFile = accepted + "." + SSTR(MMseqsNode::rank);
    // set cover: the taken pairs, greedy: the taken bits of the log
    if (par.clusteringMode == Parameters::SET_COVER) {
        if (takenBytes % sizeof(ClusterPair) != 0) {
            Debug(Debug::ERROR) << "Cannot read " << takenFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        for (size_t from = 0; from < takenBytes;) {
            if (clustersWriter.pairCount == clustersWriter.pairCapacity) {
                clustersWriter.flush();
            }
            const size_t to = std::min(takenBytes, from + (clustersWriter.pairCapacity - clustersWriter.pairCount) * sizeof(ClusterPair));
            MMseqsNode::readLog(takenFile, from, to, reinterpret_cast<char *>(clustersWriter.pairs + clustersWriter.pairCount));
            clustersWriter.pairCount += (to - from) / sizeof(ClusterPair);
            from = to;
        }
    } else {
        std::vector<uint64_t> takenBits(takenBytes / sizeof(uint64_t));
        MMseqsNode::readLog(takenFile, 0, takenBits.size() * sizeof(uint64_t), reinterpret_cast<char *>(takenBits.data()));
        FILE *acceptedHandle = FileUtil::openFileOrDie(acceptedFile.c_str(), "r", true);
        // a chunk of pairs takes whole words of taken bits, at most those of the log, and fits into the writer
        const size_t chunkPairs = std::min(clustersWriter.pairCapacity, std::max<size_t>(1, takenBits.size()) * 64) / 64 * 64;
        std::vector<ClusterPair> acceptedPairs(chunkPairs);
        for (size_t pairCount, firstWord = 0; (pairCount = fread(acceptedPairs.data(), sizeof(ClusterPair), chunkPairs, acceptedHandle)) > 0; firstWord += chunkPairs / 64) {
            const size_t wordCount = (pairCount + 63) / 64;
            if (firstWord + wordCount > takenBits.size()) {
                Debug(Debug::ERROR) << "Cannot read " << takenFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            const uint64_t *takenWords = takenBits.data() + firstWord;
            size_t takenCount = 0;
#pragma omp parallel for schedule(static) reduction(+:takenCount)
            for (size_t word = 0; word < wordCount; word++) {
                takenCount += __builtin_popcountll(takenWords[word]);
            }
            if (clustersWriter.pairCount + takenCount > clustersWriter.pairCapacity) {
                clustersWriter.flush();
            }
#pragma omp parallel for schedule(static)
            for (size_t word = 0; word < wordCount; word++) {
                uint64_t bits = takenWords[word];
                if (bits == 0) {
                    continue;
                }
                size_t pairOffset = __sync_fetch_and_add(&clustersWriter.pairCount, __builtin_popcountll(bits));
                for (; bits != 0; bits &= bits - 1) {
                    const ClusterPair &pair = acceptedPairs[word * 64 + __builtin_ctzll(bits)];
                    const ClusterPair takenPair = { pair.repKey % paddedDbSize, pair.memberKey };
                    clustersWriter.pairs[pairOffset++] = takenPair;
                }
            }
        }
        if (fclose(acceptedHandle) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << acceptedFile << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    FileUtil::remove(takenFile.c_str());
    FileUtil::remove((takenFile + ".progress").c_str());
    FileUtil::remove(acceptedFile.c_str());
    FileUtil::remove((acceptedFile + ".progress").c_str());
    const std::string clustersIndex = clustersWriter.close();
    MMseqsNode::addPhaseTime(phaseValues, "sending the cluster pairs", phaseValueTimer);
    // every node has read the members that this node sent
    std::vector<std::string> exchanges(1, clusters);
    if (memberDbr != NULL) {
        FileUtil::remove((membersExchange + "." + SSTR(MMseqsNode::rank)).c_str());
        exchanges.push_back(memberRequests);
        exchanges.push_back(membersExchange);
        memberDbr->close();
        delete memberDbr;
    }

    // every node writes its clusters and its singletons; readKeyBits fills a word past the keys
    size_t assignedKey;
    size_t assignedBytes;
    MMseqsNode::waitForLog(assignedKeys, assignedKey, assignedBytes);
    FILE *assignedHandle = FileUtil::openFileOrDie(assignedKeys.c_str(), "r", true);
    std::vector<uint64_t> assignedBits(nodeWords + 1, 0);
    MMseqsNode::readKeyBits(fileno(assignedHandle), assignedKeys, 0, 0, nodeFirstKey, nodeFirstKey + nodeKeys, nodeFirstKey, assignedBits.data());
    if (fclose(assignedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << assignedKeys << "\n";
        EXIT(EXIT_FAILURE);
    }
    uint64_t *singletonBits = NULL;
    if (hasRepresentativeBits) {
        singletonBits = MMseqsNode::readRepresentativeBits(par.db1, nodeKeyOffsets, nodeFirstKey, nodeKeys);
    } else {
        singletonBits = new(std::nothrow) uint64_t[nodeWords];
        Util::checkAllocation(singletonBits, "Can not allocate singletonBits memory in align2clust");
        memset(singletonBits, 0xFF, nodeWords * sizeof(uint64_t));
    }
    for (size_t word = 0; word < nodeWords; word++) {
        singletonBits[word] &= ~assignedBits[word];
    }
    std::vector<uint64_t>().swap(assignedBits);
    MMseqsNode::writeClusterPairs(clustersIndex, clusters, par.db3, singletonBits, nodeFirstKey, nodeKeys);
    delete[] singletonBits;
    MMseqsNode::addPhaseTime(phaseValues, "writing the clusters", phaseValueTimer);
    MMseqsNode::removeExchanges(par.db3, clusters, exchanges, phaseValues);
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db3 + ".memory").c_str());
        FileUtil::remove(assignedKeys.c_str());
        FileUtil::remove((assignedKeys + ".progress").c_str());
        DBWriter::writeDbtypeFile(par.db3.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_CLUSTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
    }
    Debug(Debug::INFO) << "Time for run Align2Clust: " << timer.lap() << " sec\n";
    return EXIT_SUCCESS;
}

int align2clust(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);
    if (par.filterCluDBFile.empty() != par.filterSeqDBFile.empty()) {
        Debug(Debug::ERROR)<< "Error: Both filterCluDBFile and filterSeqDBFile should be provided together.\n";
        EXIT(EXIT_FAILURE);
    }
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(FileUtil::parseDbType(par.db1.c_str())) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity) {
        MMseqsNode::init();
        if (par.clusteringMode != Parameters::SET_COVER && par.clusteringMode != Parameters::GREEDY && par.clusteringMode != Parameters::GREEDY_MEM) {
            Debug(Debug::ERROR) << "align2clust supports only --cluster-mode " << Parameters::SET_COVER << ", " << Parameters::GREEDY << " and " << Parameters::GREEDY_MEM
                                << " for --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << " databases\n";
            EXIT(EXIT_FAILURE);
        }
        const int prefDbtype = FileUtil::parseDbType(par.db2.c_str());
        if (Parameters::isEqualDbtype(prefDbtype, Parameters::DBTYPE_PREFILTER_RES) == false
            || (DBReader<DBKeyType>::getExtendedDbtype(prefDbtype) & Parameters::DBTYPE_EXTENDED_LINFINITY) == 0) {
            Debug(Debug::ERROR) << par.db2 << " is not a --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << " prefilter result\n";
            EXIT(EXIT_FAILURE);
        }
        if (par.compressed) {
            Debug(Debug::WARNING) << "--compressed cannot be combined with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << " databases\n";
            Debug(Debug::WARNING) << "We recompute with --compressed 0\n";
            par.compressed = 0;
        }
        if (par.includeAlignFiles) {
            Debug(Debug::WARNING) << "--include-align-files cannot be combined with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << " databases\n";
            Debug(Debug::WARNING) << "We recompute with --include-align-files 0\n";
            par.includeAlignFiles = false;
        }
        par.diagonalMode = Parameters::DIAGONAL_MODE_UNGAPPED;
        return align2clustLinfinity(par);
    }

    Timer timer;
    timer.reset();
    
    DBReader<DBKeyType> alnDbr(par.db2.c_str(), par.db2Index.c_str(), par.threads,
                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA);
    alnDbr.open(DBReader<DBKeyType>::LINEAR_ACCCESS);
    int dbtype =  Parameters::DBTYPE_CLUSTER_RES;

    DBWriter resultWriter(par.db3.c_str(), par.db3Index.c_str(), 1, par.compressed, dbtype);
    resultWriter.open();

    // Optional alignment-result output; path derived from the cluster DB (db3 + "_aln").
    // Alignment output needs a backtrace (-a) and score+cov+seqid so member records carry a CIGAR.
    if (par.includeAlignFiles) {
        const unsigned int effectiveSwMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);
        if (par.addBacktrace == false || effectiveSwMode != Matcher::SCORE_COV_SEQID) {
            Debug(Debug::ERROR) << "Writing alignment files requires backtrace and score+cov+seqid alignment.\n"
                                << "Please re-run with '-a 1' and '--alignment-mode "
                                << Parameters::ALIGNMENT_MODE_SCORE_COV_SEQID << "'.\n";
            EXIT(EXIT_FAILURE);
        }
    }
    DBWriter *alnWriter = nullptr;
    if (par.includeAlignFiles) {
        std::string alnDb = par.db3 + "_aln";
        std::string alnDbIndex = alnDb + ".index";
        alnWriter = new DBWriter(alnDb.c_str(), alnDbIndex.c_str(), par.threads, par.compressed, Parameters::DBTYPE_ALIGNMENT_RES);
        alnWriter->open();
    }

    int status = doAlign2clust(par, resultWriter, alnDbr, alnWriter);

    Debug(Debug::INFO) << "Time for run Align2Clust: " << timer.lap() << " sec\n";

    resultWriter.close();
    if (alnWriter != nullptr) {
        alnWriter->close();
        delete alnWriter;
    }
    alnDbr.close();

    return status;
}
