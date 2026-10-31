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
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <ctime>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <functional>
#include <queue>

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

// takes the largest sets with more than maxSetSize members whose members are all unassigned; a set that lost members goes back, to pairSets if given and two are left
template <typename T, typename IsAssigned, typename TakeMember>
static void takeSetCoverCandidates(std::vector<SetCoverCandidate> &candidates, std::vector<T> &memberPool, size_t &liveMemberCount, size_t maxSetSize,
                                   std::vector<ClusterPair> *pairSets, IsAssigned isAssigned, TakeMember takeMember) {
    while (candidates.empty() == false && candidates.front().memberCount > maxSetSize) {
        std::pop_heap(candidates.begin(), candidates.end(), SetCoverComparator());
        SetCoverCandidate candidate = candidates.back();
        candidates.pop_back();
        liveMemberCount -= candidate.memberCount;
        if (isAssigned(candidate.representativeId)) {
            continue;
        }
        T *members = memberPool.data() + candidate.memberOffset;
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
        if (pairSets != NULL && validCount == 2) {
            const ClusterPair pairSet = { members[0], members[1] };
            pairSets->push_back(pairSet);
            continue;
        }
        if (validCount != candidate.memberCount) {
            candidate.memberCount = validCount;
            liveMemberCount += validCount;
            candidates.push_back(candidate);
            std::push_heap(candidates.begin(), candidates.end(), SetCoverComparator());
            continue;
        }
        for (size_t i = 0; i < candidate.memberCount; i++) {
            takeMember(candidate.representativeId, members[i]);
        }
    }
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

struct AlignmentScoring {
    BaseMatrix *subMat;
    SubstitutionMatrix::FastMatrix fastMatrix;
    // the minimum ungapped score per column of a pair that is aligned with gaps
    float scorePerColThreshold;
    EvalueComputation evaluer;

    AlignmentScoring(const Parameters &par, size_t dbResidues)
        : subMat(new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, 0.0)), fastMatrix(SubstitutionMatrix::createAsciiSubMat(*subMat)),
          scorePerColThreshold(parsePrecisionLib((par.covMode == Parameters::COV_MODE_BIDIRECTIONAL) ? getCovSeqidQscPercMinDiag() : getCovSeqidQscPercMinDiagTargetCov(),
                                                 par.seqIdThr, par.covThr, 0.98)),
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

// mode 3 target gapped alignments in SWIPE lanes, their lane rows and all lane rows, and the extensions longer than the trace of the last lane rows
static std::atomic<size_t> swipeTargetPairs(0);
static std::atomic<size_t> swipeTargetLaneRows(0);
static std::atomic<size_t> swipeTargetSlotRows(0);
static std::atomic<size_t> swipeTargetLongPairs(0);

static void printSwipeLanes() {
    Debug(Debug::INFO) << "SWIPE member gapped alignments: " << swipeLanePairs.load() << " in lanes ("
                       << swipeLanePairs.load() * 100 / std::max<size_t>(1, swipeLaneSlots.load()) << "% lane fill), "
                       << swipeSinglePairs.load() << " one pair at a time\n";
    if (swipeTargetPairs.load() + swipeTargetLongPairs.load() > 0) {
        Debug(Debug::INFO) << "SWIPE target gapped alignments: " << swipeTargetPairs.load() << " in lanes ("
                           << swipeTargetLaneRows.load() * 100 / std::max<size_t>(1, swipeTargetSlotRows.load()) << "% lane fill), "
                           << swipeTargetLongPairs.load() << " extensions longer than the lane trace\n";
    }
}

// per thread buffers of --aligner-mode 1-3: the cluster members of a rep that go through the SWIPE lanes
struct SwipeBatch {
    SwipeAligner aligner;
    EvalueComputation &evaluer;
    bool initialized;
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
          evaluer(evaluer), initialized(false), lanePairs(0), laneSlots(0), singlePairs(0) {}

    ~SwipeBatch() {
        swipeLanePairs += lanePairs;
        swipeLaneSlots += laneSlots;
        swipeSinglePairs += singlePairs;
    }

    void countLanes(size_t count) {
        const size_t lanes = VECSIZE_INT * 2;
        lanePairs += count;
        laneSlots += (count + lanes - 1) / lanes * lanes;
    }

    void startRep() {
        initialized = false;
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

    // the entries in order in 16 bit lanes, as most members of a cluster are similar enough to saturate 8 bit
    void select(size_t count) {
        targets.resize(count);
        diagonals.resize(count);
        bandWidths.assign(count, 0);
        aligns.resize(count);
        for (size_t i = 0; i < count; i++) {
            targets[i] = sequences.data() + offsets[entries[i]];
            aligns[i].word = 1;
        }
    }

    void setQuery(Sequence &query) {
        if (initialized == false) {
            aligner.initQuery(&query);
            initialized = true;
        }
    }
};

// per thread buffers of --aligner-mode 3 on mode 3 DBs: the targets of the reps of a batch that alignTarget queued with their query and seed
// for the SWIPE extensions, their backward extensions and the pairs that do not fit the lanes; a rep and its query go to the queue with
// its first queued target
struct SwipeTargets {
    SwipeExtender extender;
    EvalueComputation &evaluer;
    DBKeyType repKey;
    bool initialized;
    size_t queryId;
    std::vector<DBKeyType> repKeys;
    std::vector<const char *> repSequences;
    std::vector<int> repLengths;
    std::vector<size_t> queryIds;
    std::vector<unsigned char> sequences;
    std::vector<size_t> offsets;
    std::vector<const unsigned char *> targets;
    std::vector<int> lengths;
    std::vector<s_align> aligns;
    std::vector<std::string> backtraces;
    std::vector<DBKeyType> targetKeys;
    std::vector<const char *> targetSequences;
    std::vector<int> seedQueries;
    std::vector<int> seedTargets;
    std::vector<s_align> backAligns;
    std::vector<std::string> backBacktraces;
    size_t targetPairs;

    SwipeTargets(const Parameters &par, BaseMatrix *subMat, EvalueComputation &evaluer, size_t maxSeqLen)
        : extender(subMat, maxSeqLen, par.compBiasCorrection, par.compBiasCorrectionScale, par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid()),
          evaluer(evaluer), repKey(0), initialized(false), queryId(0), targetPairs(0) {}

    ~SwipeTargets() {
        swipeTargetPairs += targetPairs;
        swipeTargetLaneRows += extender.extendLaneRows;
        swipeTargetSlotRows += extender.extendSlotRows;
        swipeTargetLongPairs += extender.extendLongJobs;
    }

    void addTarget(Sequence &query, Sequence &target, int seedQuery, int seedTarget) {
        if (initialized == false) {
            queryId = extender.addQuery(&query);
            repKeys.push_back(repKey);
            repSequences.push_back(query.getSeqData());
            repLengths.push_back(query.L);
            initialized = true;
        }
        queryIds.push_back(queryId);
        offsets.push_back(sequences.size());
        sequences.insert(sequences.end(), target.numSequence, target.numSequence + target.L);
        lengths.push_back(target.L);
        targetKeys.push_back(target.getDbKey());
        targetSequences.push_back(target.getSeqData());
        seedQueries.push_back(seedQuery);
        seedTargets.push_back(seedTarget);
    }

    void clearTargets() {
        extender.clearQueries();
        initialized = false;
        repKeys.clear();
        repSequences.clear();
        repLengths.clear();
        queryIds.clear();
        sequences.clear();
        offsets.clear();
        lengths.clear();
        targetKeys.clear();
        targetSequences.clear();
        seedQueries.clear();
        seedTargets.clear();
    }

    void startRep(DBKeyType key) {
        repKey = key;
        initialized = false;
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

// the members of a gate whose batch of a round has this many go through the SWIPE lanes, on every ISA
static const size_t SWIPE_GATE_MIN_BATCH = 4;

// the gapped alignments of the members of a round of alignClusterMembers, of all its gates; the members of a gate whose batch has
// SWIPE_GATE_MIN_BATCH members go through the lanes together, the others one pair at a time, which only depends on the member order of the gate;
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
        if (members.gates[members.poolGates[g]].batch >= SWIPE_GATE_MIN_BATCH) {
            lanePool[g] = swipe.entries.size();
            swipe.entries.push_back(g);
        }
    }
    const size_t laneCount = swipe.entries.size();
    swipe.singlePairs += swipe.gapped.size() - laneCount;
    if (laneCount > 0) {
        swipe.countLanes(laneCount);
        swipe.select(laneCount);
        swipe.laneLengths.resize(laneCount);
        for (size_t l = 0; l < laneCount; l++) {
            swipe.laneLengths[l] = swipe.lengths[swipe.entries[l]];
        }
        swipe.setQuery(query);
        swipe.aligner.alignForward(laneCount, swipe.targets.data(), swipe.laneLengths.data(), swipe.aligns.data());
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
                    // most members score beyond the byte pass, which their ungapped score shows
                    Matcher::result_t res_element = matcher.getSWResult(&element, static_cast<int>(elementDiagonal), false, par.covMode, par.covThr, par.evalThr,
                                                                        swMode, par.seqIdMode, false, false, NULL, elementUngappedAlignment.score >= UCHAR_MAX);
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

// whether the forward and backward extension of target i of alignTargetsSwipe, joined at their seed cell, meet the clustering criteria
static bool swipeExtensionsPass(const Parameters &par, SwipeTargets &swipe, size_t i) {
    const s_align &forward = swipe.aligns[i];
    const s_align &backward = swipe.backAligns[i];
    // the seed cell is in both extensions
    const int rawScore = static_cast<int>(forward.score1) + static_cast<int>(backward.score1)
                         - swipe.extender.cellScore(swipe.queryIds[i], swipe.seedQueries[i], swipe.targets[i][swipe.seedTargets[i]]);
    std::string &gappedBacktrace = swipe.backBacktraces[i];
    gappedBacktrace.pop_back();
    gappedBacktrace += swipe.backtraces[i];
    s_align joined = forward;
    joined.qStartPos1 = backward.qStartPos1;
    joined.dbStartPos1 = backward.dbStartPos1;
    const char *querySequence = swipe.repSequences[swipe.queryIds[i]];
    const char *targetSequence = swipe.targetSequences[i];
    joined.identicalAACnt = 0;
    for (size_t b = 0, q = joined.qStartPos1, t = joined.dbStartPos1; b < gappedBacktrace.size(); b++) {
        const char op = gappedBacktrace[b];
        joined.identicalAACnt += (op == 'M' && querySequence[q] == targetSequence[t]);
        q += (op != 'D');
        t += (op != 'I');
    }
    Matcher::result_t result = swipeResult(par, swipe.evaluer, swipe.targetKeys[i], rawScore, joined, swipe.repLengths[swipe.queryIds[i]], swipe.lengths[i],
                                           gappedBacktrace);
    return Alignment::checkCriteria(result, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr);
}

// the gapped alignments of the targets that alignTarget queued (--aligner-mode 3) in SWIPE lanes, the steps of BlockAligner::bandedalign;
// the ones whose rep or which were assigned since are left out
template <typename Assigned>
static void alignTargetsSwipe(const Parameters &par, SwipeTargets &swipe, const Assigned &isAssigned, std::vector<ClusterPair> &pairs) {
    const int32_t xDrop = (MIN_SIZE * par.gapExtend.values.aminoacid() + par.gapOpen.values.aminoacid());
    // twice one past the longest gap within the X-drop, which is MIN_SIZE + 1 long
    const int bandSide = 2 * (MIN_SIZE + 2);
    size_t count = 0;
    for (size_t i = 0; i < swipe.targetKeys.size(); i++) {
        if (isAssigned(swipe.repKeys[swipe.queryIds[i]]) || isAssigned(swipe.targetKeys[i])) {
            continue;
        }
        swipe.queryIds[count] = swipe.queryIds[i];
        swipe.offsets[count] = swipe.offsets[i];
        swipe.lengths[count] = swipe.lengths[i];
        swipe.seedQueries[count] = swipe.seedQueries[i];
        swipe.seedTargets[count] = swipe.seedTargets[i];
        swipe.targetKeys[count] = swipe.targetKeys[i];
        swipe.targetSequences[count] = swipe.targetSequences[i];
        count++;
    }
    if (count == 0) {
        return;
    }
    swipe.targets.resize(count);
    swipe.aligns.resize(count);
    swipe.backtraces.resize(count);
    for (size_t i = 0; i < count; i++) {
        swipe.targets[i] = swipe.sequences.data() + swipe.offsets[i];
    }
    swipe.extender.extendBanded(count, swipe.queryIds.data(), swipe.targets.data(), swipe.lengths.data(), swipe.seedQueries.data(), swipe.seedTargets.data(), false, xDrop,
                               bandSide, swipe.aligns.data(), swipe.backtraces.data());
    // the targets whose forward extension covers enough move to the front
    size_t passed = 0;
    for (size_t i = 0; i < count; i++) {
        const s_align &forward = swipe.aligns[i];
        if (forward.score1 == 0 || Util::hasCoverage(par.covThr, par.covMode, SmithWaterman::computeCov(0, forward.qEndPos1, swipe.repLengths[swipe.queryIds[i]]),
                                                     SmithWaterman::computeCov(0, forward.dbEndPos1, swipe.lengths[i])) == false) {
            continue;
        }
        swipe.queryIds[passed] = swipe.queryIds[i];
        swipe.targets[passed] = swipe.targets[i];
        swipe.lengths[passed] = swipe.lengths[i];
        swipe.seedQueries[passed] = swipe.seedQueries[i];
        swipe.seedTargets[passed] = swipe.seedTargets[i];
        swipe.targetKeys[passed] = swipe.targetKeys[i];
        swipe.targetSequences[passed] = swipe.targetSequences[i];
        swipe.aligns[passed] = swipe.aligns[i];
        swipe.backtraces[passed].swap(swipe.backtraces[i]);
        passed++;
    }
    swipe.targetPairs += count;
    count = passed;
    swipe.backAligns.resize(count);
    swipe.backBacktraces.resize(count);
    swipe.extender.extendBanded(count, swipe.queryIds.data(), swipe.targets.data(), swipe.lengths.data(), swipe.seedQueries.data(), swipe.seedTargets.data(), true, xDrop,
                               bandSide, swipe.backAligns.data(), swipe.backBacktraces.data());
    for (size_t i = 0; i < count; i++) {
        if (swipe.backAligns[i].score1 > 0 && swipeExtensionsPass(par, swipe, i)) {
            const ClusterPair pair = { swipe.repKeys[swipe.queryIds[i]], swipe.targetKeys[i] };
            pairs.push_back(pair);
        }
    }
}

// whether the target meets the clustering criteria against the query of blockAligner; diagonal becomes the one its --filter-cludb-file members start from
static bool alignTarget(const Parameters &par, BlockAligner &blockAligner, bool &blockAlignerInitialized, Sequence &query, Sequence &target,
                        float scorePerColThreshold, unsigned short &diagonal, const ClusterAssignment *assignedCluster, size_t targetId, Matcher::result_t *result,
                        SwipeTargets *swipe) {
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
        if (result != NULL) {
            std::string backtrace = par.addBacktrace ? std::string(ungappedAlignment.alnLen, 'M') : std::string();
            *result = Matcher::result_t(target.getDbKey(), ungappedAlignment.score, ungappedAlignment.qcov,
                ungappedAlignment.tcov, seqId, ungappedAlignment.eval, ungappedAlignment.alnLen,
                ungappedAlignment.qStart, ungappedAlignment.qEnd, query.L,
                ungappedAlignment.tStart, ungappedAlignment.tEnd, target.L, backtrace);
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
    // the gapped alignment in the SWIPE lanes, with the other queued targets of the rep
    if (swipe != NULL) {
        swipe->addTarget(query, target, newQueryStartPos, newTargetStartPos);
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
    Matcher::result_t gappedResult = Matcher::result_t(
        target.getDbKey(), gappedAlignment.score1, gappedAlignment.qCov, gappedAlignment.tCov,
        gappedSeqId, gappedAlignment.evalue, gappedAlnLength,
        gappedAlignment.qStartPos1, gappedAlignment.qEndPos1, query.L,
        gappedAlignment.dbStartPos1, gappedAlignment.dbEndPos1, target.L, gappedBacktrace
    );
    if (Alignment::checkCriteria(gappedResult, false, par.evalThr, par.seqIdThr,
                                 par.alnLenThr, par.covMode, par.covThr) == false) {
        return false;
    }
    if (isAssigned()) {
        return false;
    }
    diagonal = 0;
    if (result != NULL) {
        *result = gappedResult;
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
        takeSetCoverCandidates(setCoverCandidates, setCoverMemberPool, setCoverLiveMemberCount, allCalculationsDone ? 0 : currentPrefSize, NULL,
                               [&](DBLocalId id) { return loadAssignedCluster(assignedCluster, id) != DB_LOCAL_ID_INVALID; },
                               [&](DBLocalId representativeId, DBLocalId memberId) { storeAssignedCluster(assignedCluster, memberId, representativeId); });

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
        const bool swipeLanes = (par.alignerMode != Parameters::ALIGNER_MODE_DEFAULT && par.gapOpen.values.aminoacid() >= par.gapExtend.values.aminoacid());
        SwipeBatch swipe(par, scoring.subMat, scoring.evaluer, db_maxseqlen);
        // the member gates of a rep wait for its last target and go in rounds that share the lanes, unless the alignments are written in target order
        ClusterMembers members(swipeLanes ? &swipe : NULL, swipeLanes && cluDbr != NULL && alnWriter == nullptr);
        std::string querySeqBuffer;
        querySeqBuffer.reserve(db_maxseqlen + 1);

        const bool includeAlignFiles = (alnWriter != nullptr);
        std::string alnResultBuffer;
        Matcher::result_t targetResult;
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
                unsigned short diagonal = targetsWithDiagonal[targetIdx].second;
                bool isAccepted = alignTarget(par, blockAligner, blockAlignerInitialized, query, target, scoring.scorePerColThreshold, diagonal, assignedCluster, targetId,
                                              includeAlignFiles ? &targetResult : NULL, NULL);
                if (isAccepted && cluDbr != NULL) {
                    isAccepted = alignTargetMembers(par, blockAligner, matcher, matcherInitialized, query, element, cluDbr, cluSeqDbr, targetKey, targetId,
                                                    diagonal, swMode, threadIdx, includeAlignFiles ? &pendingMemberAln : NULL, alnLineBuffer.data(), members);
                }
                if (isAccepted) {
                    if (includeAlignFiles) {
                        appendAlignmentResult(alnResultBuffer, alnLineBuffer.data(), targetResult, par.addBacktrace);
                        alnResultBuffer += pendingMemberAln;
                    }
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
    ClusterPair repFirstTarget;
    unsigned int targetCount;

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

// --createdb-mode 3: a row of <pref>.partitions, the entries from firstKey on in <pref>.<node>
struct PrefPartition {
    size_t firstKey;
    size_t prefOffset;
    size_t prefBytes;
    size_t cost;
    size_t node;
};

struct RepTargetsRecord {
    RepTargets header;
    size_t targetOffset;
    size_t readBytes;

    static bool compareByRep(const RepTargetsRecord &first, const RepTargetsRecord &second) {
        return ClusterPair::compareByRep(first.header.repFirstTarget, second.header.repFirstTarget);
    }
};

// the master aligns only on one node
static size_t getFirstAlignNode() {
    return (MMseqsNode::count > 1) ? 1 : 0;
}

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

// --createdb-mode 3: the records [begin, end) of a chunk aligned at once, with the keys to load and their entries
struct RepWindow {
    RepWindow() : chunk(0), begin(0), end(0), data(NULL), allocatedBytes(0) {}
    size_t chunk;
    size_t begin;
    size_t end;
    std::vector<DBKeyType> keys;
    std::vector<size_t> homeKeyOffsets;
    char *data;
    size_t allocatedBytes;
    std::vector<size_t> entryOffsets;
};

static void collectUniqueKeys(const std::vector<std::vector<DBKeyType> > &threadKeys, std::vector<DBKeyType> &keys) {
    keys.clear();
    for (size_t thread = 0; thread < threadKeys.size(); thread++) {
        keys.insert(keys.end(), threadKeys[thread].begin(), threadKeys[thread].end());
    }
    SORT_PARALLEL(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
}

static const size_t PUBLISH_BYTES = 4 * 1024 * 1024;
static const time_t PUBLISH_SECONDS = 10;
static const time_t READ_SECONDS = 1;
// batches per thread of the records aligned at once, taken in rep order
static const size_t ALIGN_BATCHES_PER_THREAD = 64;
static const size_t PREFETCH_PAIRS = 16;
static const int CLUSTER_THREADS = 4;
// an accepted log: the order key of every rep with this bit, then its accepted targets
static const uint64_t ACCEPTED_REP_BIT = 1ULL << 63;

// each of these takes the memory limit divided by its divisor
static const size_t MASTER_MEMORY_DIVISOR = 2;
static const size_t RING_MEMORY_DIVISOR = 256;
static const size_t PREF_SPLIT_MEMORY_DIVISOR = 16;
static const size_t WRITE_BATCH_MEMORY_DIVISOR = 16;
static const size_t CHUNK_MEMORY_DIVISOR = 32;
static const size_t WINDOW_MEMORY_DIVISOR = 8;
static const size_t MEMBER_PART_MEMORY_DIVISOR = 16;
static const size_t STEP_MEMORY_DIVISOR = 4;
static const size_t STEP_PAIRS_MEMORY_DIVISOR = 16;
static const size_t LOG_PART_MEMORY_DIVISOR = 64;
static const size_t CLUSTER_PAIRS_MEMORY_DIVISOR = 16;
// the bytes of a window key besides its entry: its copies, its home and its entry offset
static const size_t WINDOW_KEY_BYTES = 4 * sizeof(DBKeyType) + sizeof(unsigned int) + sizeof(size_t);
// the first window is small so aligning starts soon; every next one doubles
static const size_t FIRST_WINDOW_BYTES = 64 * 1024 * 1024;
// the bytes of the rep targets that a thread writes at once
static const size_t REP_TARGETS_WRITE_BYTES = 16 * 1024 * 1024;

// offsets[0] is 0, offsets[1..count] become their sums from the front: every thread sums its part, then adds the parts before it
static void sumOffsets(size_t *offsets, size_t count, int threads) {
    std::vector<size_t> threadOffsets(threads + 1, 0);
#pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threads; thread++) {
        const size_t end = count * (thread + 1) / threads + 1;
        for (size_t i = count * thread / threads + 2; i < end; i++) {
            offsets[i] += offsets[i - 1];
        }
        threadOffsets[thread + 1] = (count * thread / threads + 1 < end) ? offsets[end - 1] : 0;
    }
    for (int thread = 0; thread < threads; thread++) {
        threadOffsets[thread + 1] += threadOffsets[thread];
    }
#pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threads; thread++) {
        for (size_t i = count * thread / threads + 1; i < count * (thread + 1) / threads + 1; i++) {
            offsets[i] += threadOffsets[thread];
        }
    }
}

static size_t getClusterStepMemory(size_t dbSize, size_t memoryLimit) {
    return (dbSize + 63) / 64 * sizeof(uint64_t) + memoryLimit / STEP_MEMORY_DIVISOR;
}

// --createdb-mode 3: writes the reps of the partitions of this node with their targets; memberRequests: requests their --filter-cludb-file members
static std::string writeRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<PrefPartition> &nodePartitions, size_t maxWindowBytes,
                                   size_t memoryLimit, size_t paddedDbSize, size_t prefSizeBits, const std::string &repTargets, MMseqsNode::ClusterPairWriter *memberRequests,
                                   std::string &phaseValues) {
    // a record holds at most maxRecordTargets targets, so they fit maxWindowBytes as pages
    const size_t batchBytes = std::max<size_t>(1, memoryLimit / WRITE_BATCH_MEMORY_DIVISOR);
    const size_t maxRecordTargets = std::min<size_t>(UINT_MAX, std::max<size_t>(1, std::min(batchBytes / 2 / sizeof(DBKeyType), maxWindowBytes / std::max<size_t>(seqDbr.getMaxSeqLen() + 2, Util::getPageSize()))));

    const std::string repTargetsFile = repTargets + "." + SSTR(MMseqsNode::rank);
    FILE *repTargetsHandle = FileUtil::openAndDelete(repTargetsFile.c_str(), "w");
    std::string repTargetsIndexData;
    size_t repTargetsBytes = 0;
    std::vector<std::pair<size_t, size_t> > prefSplits;
    // not zeroed, as each split or batch overwrites all of its part of them
    DBKeyType *entryRepKeys = NULL;
    size_t *entryTargetOffsets = NULL;
    size_t *entryRecordOffsets = NULL;
    size_t allocatedEntries = 0;
    DBKeyType *splitTargets = NULL;
    size_t allocatedTargets = 0;
    RepTargetsRecord *records = NULL;
    size_t allocatedRecords = 0;
    char *flushBuffer = NULL;
    size_t allocatedBytes = 0;
    std::vector<size_t> threadOffsets(par.threads + 1);
    std::vector<size_t> nodeOffsets(MMseqsNode::count + 1);
    std::vector<std::vector<DBKeyType> > threadRequestKeys(par.threads);
    std::vector<DBKeyType> requestKeys;
    double readSeconds = 0;
    double writeSeconds = 0;
    for (size_t i = 0; i < nodePartitions.size();) {
        Timer readTimer;
        const size_t prefNode = nodePartitions[i].node;
        DBReader<DBKeyType> *prefDb = new DBReader<DBKeyType>((par.db2 + "." + SSTR(prefNode)).c_str(), (par.db2 + ".index." + SSTR(prefNode)).c_str(), par.threads,
                                                              DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        prefDb->open(DBReader<DBKeyType>::NOSORT);
        // the splits of all partitions in this pref file, so the next split is prefetched across partitions
        prefSplits.clear();
        for (; i < nodePartitions.size() && nodePartitions[i].node == prefNode; i++) {
            const PrefPartition &partition = nodePartitions[i];
            const size_t prefEnd = partition.prefOffset + partition.prefBytes;
            const size_t partitionEndEntry = prefDb->findEntryByOffset(prefEnd);
            // a pref byte: two buffers, 8 bytes of targets, 16 of member requests; an entry: 64 bytes of offsets and keys
            const double entriesPerByte = static_cast<double>(partitionEndEntry - prefDb->findEntryByOffset(partition.prefOffset)) / std::max<size_t>(1, partition.prefBytes);
            const double bytesPerPrefByte = (memberRequests != NULL) ? 26.0 : 10.0;
            const size_t prefSplitBytes = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(memoryLimit / PREF_SPLIT_MEMORY_DIVISOR) / (bytesPerPrefByte + 64.0 * entriesPerByte)));
            const size_t firstSplit = prefSplits.size();
            for (size_t offset = partition.prefOffset; offset < prefEnd; offset += prefSplitBytes) {
                const size_t entry = prefDb->findEntryByOffset(offset);
                if (prefSplits.size() == firstSplit || prefSplits.back().first != entry) {
                    prefSplits.push_back(std::make_pair(entry, entry));
                }
            }
            for (size_t split = firstSplit; split < prefSplits.size(); split++) {
                prefSplits[split].second = (split + 1 < prefSplits.size()) ? prefSplits[split + 1].first : partitionEndEntry;
            }
        }
        readSeconds += readTimer.getTimediff();
        std::thread prefetchThread;
        for (size_t split = 0; split < prefSplits.size(); split++) {
            const size_t splitBegin = prefSplits[split].first;
            const size_t entryCount = prefSplits[split].second - splitBegin;
            readTimer.reset();
            if (prefetchThread.joinable()) {
                prefetchThread.join();
            }
            prefDb->loadData(splitBegin, prefSplits[split].second);
            readSeconds += readTimer.getTimediff();
            // prefetchData reads the entry offsets before it returns, so it runs beside the work on this split
            if (split + 1 < prefSplits.size()) {
                prefetchThread = std::thread(&DBReader<DBKeyType>::prefetchData, prefDb, prefSplits[split + 1].first, prefSplits[split + 1].second);
            }
            if (allocatedEntries < entryCount + 1) {
                delete[] entryRepKeys;
                delete[] entryTargetOffsets;
                delete[] entryRecordOffsets;
                allocatedEntries = entryCount + 1;
                entryRepKeys = new(std::nothrow) DBKeyType[allocatedEntries];
                entryTargetOffsets = new(std::nothrow) size_t[allocatedEntries];
                entryRecordOffsets = new(std::nothrow) size_t[allocatedEntries];
                Util::checkAllocation(entryRepKeys, "Can not allocate entryRepKeys memory in align2clust");
                Util::checkAllocation(entryTargetOffsets, "Can not allocate entryTargetOffsets memory in align2clust");
                Util::checkAllocation(entryRecordOffsets, "Can not allocate entryRecordOffsets memory in align2clust");
            }
            entryTargetOffsets[0] = 0;
            entryRecordOffsets[0] = 0;
#pragma omp parallel for schedule(dynamic, 100)
            for (size_t entry = 0; entry < entryCount; entry++) {
                const char *data = prefDb->getData(splitBegin + entry, 0);
                const size_t targetCount = std::count_if(data, data + prefDb->getEntryLen(splitBegin + entry) - 1, [](char byte) { return static_cast<unsigned char>(byte) < 0x80; }) - 1;
                entryTargetOffsets[entry + 1] = targetCount;
                entryRecordOffsets[entry + 1] = (targetCount + maxRecordTargets - 1) / maxRecordTargets;
            }
            sumOffsets(entryTargetOffsets, entryCount, par.threads);
            sumOffsets(entryRecordOffsets, entryCount, par.threads);
            if (allocatedTargets < entryTargetOffsets[entryCount]) {
                delete[] splitTargets;
                allocatedTargets = entryTargetOffsets[entryCount];
                splitTargets = new(std::nothrow) DBKeyType[allocatedTargets];
                Util::checkAllocation(splitTargets, "Can not allocate splitTargets memory in align2clust");
            }
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
                    const DBKeyType repKey = Util::getPrefKey(data);
                    entryRepKeys[entry] = repKey;
                    DBKeyType *targets = splitTargets + entryTargetOffsets[entry];
                    const size_t targetCount = entryTargetOffsets[entry + 1] - entryTargetOffsets[entry];
                    const size_t repLength = seqDbr.getSeqLen(repKey);
                    for (size_t j = 0; j < targetCount; j++) {
                        targets[j] = ((j > 0) ? targets[j - 1] : 0) + Util::getPrefKey(data);
                        // the targets that alignTarget may accept
                        if (memberRequests != NULL && Util::canBeCovered(par.covThr, par.covMode, repLength, seqDbr.getSeqLen(targets[j]))) {
                            threadRequestKeys[thread_idx].push_back(targets[j]);
                        }
                    }
                }
            }
            if (memberRequests != NULL) {
                collectUniqueKeys(threadRequestKeys, requestKeys);
                for (size_t j = 0; j < requestKeys.size(); j++) {
                    memberRequests->add(requestKeys[j], MMseqsNode::rank);
                }
            }
            const auto getRecordBytes = [&](size_t entry) {
                return entryRecordOffsets[entry] * sizeof(RepTargets) + entryTargetOffsets[entry] * sizeof(DBKeyType);
            };
            // a batch ends before the entry whose records would exceed batchBytes, but takes at least one entry
            for (size_t firstEntry = 0, endEntry = 0; firstEntry < entryCount; firstEntry = endEntry) {
                const size_t maxRecordBytes = getRecordBytes(firstEntry) + batchBytes;
                endEntry = firstEntry + 1;
                for (size_t entries = entryCount - endEntry; entries > 0;) {
                    const size_t half = entries / 2;
                    if (getRecordBytes(endEntry + half + 1) <= maxRecordBytes) {
                        endEntry += half + 1;
                        entries -= half + 1;
                    } else {
                        entries = half;
                    }
                }
                const size_t firstRecord = entryRecordOffsets[firstEntry];
                const size_t recordCount = entryRecordOffsets[endEntry] - firstRecord;
                if (recordCount == 0) {
                    continue;
                }
                if (allocatedRecords < recordCount) {
                    delete[] records;
                    allocatedRecords = recordCount;
                    records = new(std::nothrow) RepTargetsRecord[allocatedRecords];
                    Util::checkAllocation(records, "Can not allocate records memory in align2clust");
                }
#pragma omp parallel for schedule(dynamic, 100)
                for (size_t entry = firstEntry; entry < endEntry; entry++) {
                    const size_t targetCount = entryTargetOffsets[entry + 1] - entryTargetOffsets[entry];
                    size_t j = entryRecordOffsets[entry] - firstRecord;
                    const size_t prefSizeBit = (targetCount > 0) ? std::min<size_t>(prefSizeBits - 1, 63 - __builtin_clzll(targetCount)) : 0;
                    for (size_t first = 0; first < targetCount; first += maxRecordTargets, j++) {
                        RepTargets &header = records[j].header;
                        // the file gets the padding too
                        memset(&header, 0, sizeof(RepTargets));
                        header.repFirstTarget.repKey = (prefSizeBits - 1 - prefSizeBit) * paddedDbSize + entryRepKeys[entry];
                        header.repFirstTarget.memberKey = splitTargets[entryTargetOffsets[entry] + first];
                        header.targetCount = static_cast<unsigned int>(std::min(targetCount - first, maxRecordTargets));
                        records[j].targetOffset = entryTargetOffsets[entry] + first;
                    }
                }
                // the records of entries in rep order are already sorted
                bool isSorted = true;
#pragma omp parallel for schedule(static) reduction(&&: isSorted)
                for (size_t j = 1; j < recordCount; j++) {
                    isSorted = isSorted && RepTargetsRecord::compareByRep(records[j], records[j - 1]) == false;
                }
                if (isSorted == false) {
                    SORT_PARALLEL(records, records + recordCount, RepTargetsRecord::compareByRep);
                }
                std::fill(threadOffsets.begin(), threadOffsets.end(), 0);
#pragma omp parallel for schedule(static)
                for (int thread = 0; thread < par.threads; thread++) {
                    for (size_t j = recordCount * thread / par.threads; j < recordCount * (thread + 1) / par.threads; j++) {
                        threadOffsets[thread + 1] += sizeof(RepTargets) + records[j].header.targetCount * sizeof(DBKeyType);
                    }
                }
                for (int thread = 0; thread < par.threads; thread++) {
                    threadOffsets[thread + 1] += threadOffsets[thread];
                }
                const size_t flushBytes = threadOffsets[par.threads];
                if (allocatedBytes < flushBytes) {
                    delete[] flushBuffer;
                    allocatedBytes = flushBytes;
                    flushBuffer = new(std::nothrow) char[allocatedBytes];
                    Util::checkAllocation(flushBuffer, "Can not allocate flushBuffer memory in align2clust");
                }
#pragma omp parallel for schedule(static)
                for (int thread = 0; thread < par.threads; thread++) {
                    size_t bufferPos = threadOffsets[thread];
                    for (size_t j = recordCount * thread / par.threads; j < recordCount * (thread + 1) / par.threads; j++) {
                        const RepTargets &header = records[j].header;
                        memcpy(flushBuffer + bufferPos, &header, sizeof(RepTargets));
                        memcpy(flushBuffer + bufferPos + sizeof(RepTargets), splitTargets + records[j].targetOffset, header.targetCount * sizeof(DBKeyType));
                        bufferPos += sizeof(RepTargets) + header.targetCount * sizeof(DBKeyType);
                    }
                }
                Timer writeTimer;
#pragma omp parallel for schedule(dynamic, 1)
                for (size_t offset = 0; offset < flushBytes; offset += REP_TARGETS_WRITE_BYTES) {
                    FileUtil::pwriteAll(fileno(repTargetsHandle), repTargetsFile.c_str(), flushBuffer + offset, std::min(REP_TARGETS_WRITE_BYTES, flushBytes - offset), repTargetsBytes + offset);
                }
                writeSeconds += writeTimer.getTimediff();
                // all bytes of the flush are for this node
                for (size_t node = 0; node <= MMseqsNode::count; node++) {
                    nodeOffsets[node] = (node <= MMseqsNode::rank) ? 0 : flushBytes;
                }
                MMseqsNode::addExchangeFlush(repTargetsIndexData, repTargetsBytes, nodeOffsets);
            }
        }
        prefDb->close();
        delete prefDb;
    }
    delete[] entryRepKeys;
    delete[] entryTargetOffsets;
    delete[] entryRecordOffsets;
    delete[] splitTargets;
    delete[] records;
    delete[] flushBuffer;
    // only this node reads its file, so it needs no barrier
    if (fclose(repTargetsHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << repTargetsFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    MMseqsNode::addPhaseSeconds(phaseValues, "reading the prefilter", readSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "writing the rep targets file", writeSeconds);
    MMseqsNode::addPhaseValue(phaseValues, "Rep targets bytes", SSTR(repTargetsBytes));
    return repTargetsIndexData;
}

// --createdb-mode 3: aligns the received records window by window and logs the accepted pairs
static void alignRepTargets(Parameters &par, DBReader<DBKeyType> &seqDbr, const std::vector<size_t> &groupKeyOffsets, size_t lagKeys, size_t memoryLimit, size_t representativeResidues,
                            size_t paddedDbSize, size_t prefSizeBits, const std::string &repTargetsIndex, const std::string &repTargets, const std::string &accepted,
                            const std::string &assignedChanges, const std::string &requests, const std::string &served,
                            const std::vector<ClusterPair> &members, DBReader<DBKeyType> *memberDbr, std::string &phaseValues) {
    const size_t dbSize = seqDbr.getSize();
    const size_t orderKeyEnd = (prefSizeBits - 1) * paddedDbSize + dbSize;
    const size_t maxSeqLen = seqDbr.getMaxSeqLen();
    const size_t pageSize = Util::getPageSize();
    AlignmentScoring scoring(par, representativeResidues);
    const unsigned int swMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);
    std::vector<MMseqsNode::ExchangeBlock> repBlocks;
    std::vector<FILE *> repFiles;
    // the flushes of a node are in rep order, but not in order key order with set cover
    MMseqsNode::openExchange(repTargetsIndex, repTargets, repBlocks, repFiles, par.clusteringMode != Parameters::SET_COVER);
    const std::string acceptedFile = accepted + "." + SSTR(MMseqsNode::rank);
    FILE *acceptedHandle = FileUtil::openAndDelete(acceptedFile.c_str(), "w");
    size_t acceptedBytes = 0;
    std::vector<uint64_t> acceptedWords;
    DBKeyType loggedRepKey = DB_KEY_INVALID;
    const auto logAcceptedPairs = [&](const ClusterPair *pairs, size_t pairCount) {
        acceptedWords.clear();
        for (size_t i = 0; i < pairCount; i++) {
            if (pairs[i].repKey != loggedRepKey) {
                loggedRepKey = pairs[i].repKey;
                acceptedWords.push_back(loggedRepKey | ACCEPTED_REP_BIT);
            }
            acceptedWords.push_back(pairs[i].memberKey);
        }
        if (fwrite(acceptedWords.data(), sizeof(uint64_t), acceptedWords.size(), acceptedHandle) != acceptedWords.size()) {
            Debug(Debug::ERROR) << "Cannot write " << acceptedFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        acceptedBytes += acceptedWords.size() * sizeof(uint64_t);
    };

    // assigned bits: the keys of the own groups in ownWords, the other reps in a ring of order key words from the master
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
                const size_t ownKey = getOwnKey(group, changes[i]);
                __atomic_fetch_or(&ownWords[ownKey / 64], 1ULL << (ownKey % 64), __ATOMIC_RELAXED);
            }
        }
    };

    std::vector<char> chunkData[2];
    std::vector<RepTargetsRecord> chunkRecords[2];
    std::vector<DBKeyType> chunkTargets[2];
    std::vector<size_t> recordOffsets;
    size_t chunkPos[2] = { 0, 0 };
    size_t chunk = 0;
    const ClusterPair lastRecord = { DB_KEY_INVALID, DB_KEY_INVALID };
    std::vector<std::vector<DBKeyType> > threadKeys(par.threads);
    const unsigned int nodes = MMseqsNode::count;
    const unsigned int rank = MMseqsNode::rank;
    const size_t firstAlignNode = getFirstAlignNode();
    const size_t alignNodes = nodes - firstAlignNode;
    const auto getKeyHome = [&](size_t key) {
        return firstAlignNode + seqDbr.getKeyBlock(key) % alignNodes;
    };
    std::vector<DBKeyType> uniqueKeys;
    std::vector<unsigned int> keyHomes;
    std::vector<size_t> threadHomeOffsets;
    size_t windowLimit = std::min(memoryLimit / WINDOW_MEMORY_DIVISOR / 64, FIRST_WINDOW_BYTES);
    // a window takes records while their reps and targets fit, at least one
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
                const size_t repKey = record.header.repFirstTarget.repKey % paddedDbSize;
                record.readBytes = seqDbr.getEntryLen(repKey) + WINDOW_KEY_BYTES;
                for (unsigned int j = 0; j < record.header.targetCount; j++) {
                    record.readBytes += seqDbr.getEntryLen(targets[j]) + WINDOW_KEY_BYTES;
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
            if (window.end > window.begin && windowBytes > windowLimit) {
                break;
            }
        }
        chunkPos[chunk] = window.end;
        windowLimit = std::min(2 * windowLimit, memoryLimit / WINDOW_MEMORY_DIVISOR);
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
                threadKeys[thread_idx].push_back(record.header.repFirstTarget.repKey % paddedDbSize);
                if (isAssigned(record.header.repFirstTarget.repKey)) {
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
        collectUniqueKeys(threadKeys, uniqueKeys);
        const size_t keyCount = uniqueKeys.size();
        keyHomes.resize(keyCount);
        window.keys.resize(keyCount);
        // the keys go to their homes in key order: every thread counts the homes of its static part, then places it after the parts of the lower threads
        threadHomeOffsets.assign(nodes * par.threads + 1, 0);
#pragma omp parallel
        {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<size_t> homeOffsets(nodes, 0);
#pragma omp for schedule(static)
            for (size_t i = 0; i < keyCount; i++) {
                keyHomes[i] = getKeyHome(uniqueKeys[i]);
                homeOffsets[keyHomes[i]]++;
            }
            for (unsigned int node = 0; node < nodes; node++) {
                threadHomeOffsets[node * par.threads + thread_idx + 1] = homeOffsets[node];
            }
#pragma omp barrier
#pragma omp single
            for (size_t i = 0; i + 1 < threadHomeOffsets.size(); i++) {
                threadHomeOffsets[i + 1] += threadHomeOffsets[i];
            }
            for (unsigned int node = 0; node < nodes; node++) {
                homeOffsets[node] = threadHomeOffsets[node * par.threads + thread_idx];
            }
#pragma omp for schedule(static)
            for (size_t i = 0; i < keyCount; i++) {
                window.keys[homeOffsets[keyHomes[i]]++] = uniqueKeys[i];
            }
        }
        window.homeKeyOffsets.resize(nodes + 1);
        for (unsigned int node = 0; node <= nodes; node++) {
            window.homeKeyOffsets[node] = threadHomeOffsets[node * par.threads];
        }
        window.entryOffsets.resize(keyCount + 1);
        window.entryOffsets[0] = 0;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < keyCount; i++) {
            window.entryOffsets[i + 1] = seqDbr.getEntryLen(window.keys[i]);
        }
        for (size_t i = 0; i < keyCount; i++) {
            window.entryOffsets[i + 1] += window.entryOffsets[i];
        }
        // not zeroed, as the entries overwrite all of it
        if (window.allocatedBytes < window.entryOffsets[keyCount]) {
            delete[] window.data;
            window.allocatedBytes = window.entryOffsets[keyCount];
            window.data = new(std::nothrow) char[window.allocatedBytes];
            Util::checkAllocation(window.data, "Can not allocate window data memory in align2clust");
        }
    };
    const auto getFirstRepKey = [&](const RepWindow &window) {
        return (window.end > window.begin) ? static_cast<size_t>(chunkRecords[window.chunk][window.begin].header.repFirstTarget.repKey) : orderKeyEnd;
    };
    // the block cache takes the memory left by the bits, members, chunks, windows and a member part
    const size_t reservedMemory = (ownWords.size() + ring.size()) * sizeof(uint64_t) + members.size() * sizeof(ClusterPair) + repBlocks.size() * MMseqsNode::EXCHANGE_BUFFER_SIZE
                                  + 2 * 2 * (memoryLimit / CHUNK_MEMORY_DIVISOR) + 2 * 2 * (memoryLimit / WINDOW_MEMORY_DIVISOR) + ((memberDbr != NULL) ? 2 * (memoryLimit / MEMBER_PART_MEMORY_DIVISOR) : 0);
    seqDbr.setBlockCacheBytes(memoryLimit - std::min(memoryLimit, reservedMemory));
    size_t windowCount = 0;
    size_t windowKeys = 0;

    RepWindow window;
    RepWindow nextWindow;
    readWindow(window);
    MMseqsNode::publishLog(acceptedHandle, acceptedFile, getFirstRepKey(window), 0);
    size_t publishedBytes = 0;
    time_t publishTime = time(NULL);
    time_t changesTime = publishTime;
    std::mutex acceptedMutex;
    std::mutex assignedChangesMutex;
    std::vector<size_t> batchOffsets;
    std::vector<unsigned char> batchDone;
    std::vector<std::vector<ClusterPair> > batchPairs;
    std::vector<ClusterPair> windowPairs;
    std::vector<size_t> memberStarts;
    std::vector<size_t> memberBytes;
    std::vector<size_t> partOffsets;
    std::vector<unsigned char> membersPassed;
    std::vector<DBKeyType> partKeys;
    std::vector<size_t> repGroups;
    double waitSeconds = 0;
    double sequenceWaitSeconds = 0;
    size_t alignedPairs = 0;
    const auto getWindowData = [&](size_t key) -> const char * {
        const size_t home = getKeyHome(key);
        const DBKeyType *keys = window.keys.data();
        const DBKeyType *found = std::lower_bound(keys + window.homeKeyOffsets[home], keys + window.homeKeyOffsets[home + 1], static_cast<DBKeyType>(key));
        return (found != keys + window.homeKeyOffsets[home + 1] && *found == key) ? window.data + window.entryOffsets[found - keys] : NULL;
    };

    // a service thread besides --threads gets the entries of a window from their homes and serves the keys of its blocks to the others
    const bool hasService = rank >= firstAlignNode;
    std::mutex serviceMutex;
    std::condition_variable serviceCondition;
    RepWindow *pendingWindow = NULL;
    RepWindow *fetchedWindow = NULL;
    bool isWaitingForMaster = false;
    std::vector<DBKeyType> loadKeys;
    // copies the entries of keys to data at entryOffsets on --threads threads besides the aligning ones, as one thread could not keep up with them
    const auto copyEntries = [&](const DBKeyType *keys, size_t keyCount, const size_t *entryOffsets, char *data) {
        for (size_t first = 0; first < keyCount;) {
            const size_t loadFirst = first;
            for (size_t loadBytes = 0; first < keyCount && (first == loadFirst || loadBytes + std::max(entryOffsets[first + 1] - entryOffsets[first], pageSize) <= memoryLimit / WINDOW_MEMORY_DIVISOR); first++) {
                loadBytes += std::max(entryOffsets[first + 1] - entryOffsets[first], pageSize);
            }
            loadKeys.assign(keys + loadFirst, keys + first);
            seqDbr.loadData(loadKeys);
#pragma omp parallel for schedule(static) num_threads(par.threads)
            for (size_t i = loadFirst; i < first; i++) {
                memcpy(data + entryOffsets[i], seqDbr.getData(keys[i], 0), entryOffsets[i + 1] - entryOffsets[i]);
            }
        }
    };
    std::thread serviceThread;
    if (hasService) {
        serviceThread = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(1);
#endif
            std::vector<std::string> requestIndexes(nodes);
            std::vector<std::string> requestFiles(nodes);
            std::vector<std::string> receivedIndexes(nodes);
            std::vector<std::string> receivedFiles(nodes);
            std::vector<std::string> servedIndexes(nodes);
            std::vector<std::string> servedFiles(nodes);
            for (unsigned int node = firstAlignNode; node < nodes; node++) {
                requestIndexes[node] = requests + ".index." + SSTR(node);
                requestFiles[node] = requests + "." + SSTR(node);
                receivedIndexes[node] = served + ".index." + SSTR(node) + "." + SSTR(rank);
                receivedFiles[node] = served + "." + SSTR(node) + "." + SSTR(rank);
                servedIndexes[node] = served + ".index." + SSTR(rank) + "." + SSTR(node);
                servedFiles[node] = served + "." + SSTR(rank) + "." + SSTR(node);
            }
            // a request index line: the flush, whether it is the last, where the keys of every node start, then the end
            std::vector<uint64_t> requestLine(nodes + 3);
            std::vector<uint64_t> servedLine(1);
            std::vector<size_t> nodeRequests(nodes, 0);
            std::vector<size_t> nodeServed(nodes, 0);
            std::vector<size_t> nodeReceived(nodes, 0);
            std::vector<unsigned char> isRequesting(nodes, 0);
            for (size_t node = firstAlignNode; node < nodes; node++) {
                isRequesting[node] = (node != rank);
            }
            std::vector<unsigned char> isReceived(nodes, 0);
            std::vector<DBKeyType> requestKeys;
            std::vector<size_t> requestOffsets;
            std::vector<char> slice;
            std::vector<BatchReader::ReadRequest> sliceReads;
            BatchReader *sliceReader = NULL;
            RepWindow *requestedWindow = NULL;
            size_t flush = 0;
            bool isLastRequested = false;
            for (unsigned int polls = 0; isLastRequested == false || requestedWindow != NULL || std::find(isRequesting.begin(), isRequesting.end(), 1) != isRequesting.end();) {
                bool hasWorked = false;
                // request the next window once the master is less than lagKeys behind its first rep
                RepWindow *nextRequest = NULL;
                if (requestedWindow == NULL) {
                    std::lock_guard<std::mutex> lock(serviceMutex);
                    nextRequest = pendingWindow;
                }
                if (nextRequest != NULL) {
                    const bool isLast = nextRequest->end == nextRequest->begin;
                    bool isAhead;
                    {
                        std::lock_guard<std::mutex> changesLock(assignedChangesMutex);
                        const size_t repKey = getFirstRepKey(*nextRequest);
                        if (isLast == false && repKey >= clusteredKey + lagKeys) {
                            readAssignedChanges();
                        }
                        isAhead = isLast == false && repKey >= clusteredKey + lagKeys;
                    }
                    {
                        std::lock_guard<std::mutex> lock(serviceMutex);
                        isWaitingForMaster = isAhead;
                        if (isAhead == false) {
                            pendingWindow = NULL;
                        }
                    }
                    if (isAhead == false) {
                        if (alignNodes > 1) {
                            FILE *requestHandle = FileUtil::openAndDelete(requestFiles[rank].c_str(), "w");
                            size_t requestBytes = 0;
                            for (unsigned int node = 0; node < nodes; node++) {
                                requestLine[2 + node] = requestBytes;
                                const size_t keyCount = (node != rank) ? nextRequest->homeKeyOffsets[node + 1] - nextRequest->homeKeyOffsets[node] : 0;
                                if (fwrite(nextRequest->keys.data() + nextRequest->homeKeyOffsets[node], sizeof(DBKeyType), keyCount, requestHandle) != keyCount) {
                                    Debug(Debug::ERROR) << "Cannot write " << requestFiles[rank] << "\n";
                                    EXIT(EXIT_FAILURE);
                                }
                                requestBytes += keyCount * sizeof(DBKeyType);
                            }
                            requestLine[0] = flush++;
                            requestLine[1] = isLast;
                            requestLine[2 + nodes] = requestBytes;
                            MMseqsNode::publishExchangeFlush(requestHandle, requestFiles[rank], requestBytes, requestIndexes[rank], requestLine);
                        }
                        const size_t ownKey = nextRequest->homeKeyOffsets[rank];
                        copyEntries(nextRequest->keys.data() + ownKey, nextRequest->homeKeyOffsets[rank + 1] - ownKey, nextRequest->entryOffsets.data() + ownKey, nextRequest->data);
                        isLastRequested = isLast;
                        requestedWindow = nextRequest;
                        hasWorked = true;
                    }
                }
                // receive from every home once it has served, starting at the next node so the nodes do not all read one home first
                if (requestedWindow != NULL) {
                    bool isComplete = true;
                    for (unsigned int i = 1; i < nodes; i++) {
                        const unsigned int home = (rank + i) % nodes;
                        const size_t sliceBegin = requestedWindow->entryOffsets[requestedWindow->homeKeyOffsets[home]];
                        const size_t sliceEnd = requestedWindow->entryOffsets[requestedWindow->homeKeyOffsets[home + 1]];
                        if (isReceived[home] || sliceBegin == sliceEnd) {
                            continue;
                        }
                        if (MMseqsNode::readExchangeFlush(receivedIndexes[home], nodeReceived[home], servedLine) == false) {
                            isComplete = false;
                            continue;
                        }
                        const int fd = open(receivedFiles[home].c_str(), O_RDONLY | O_CLOEXEC);
                        if (fd < 0) {
                            Debug(Debug::ERROR) << "Cannot open " << receivedFiles[home] << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        const BatchReader::ReadRequest request = { fd, receivedFiles[home].c_str(), requestedWindow->data + sliceBegin, 0, sliceEnd - sliceBegin, sliceEnd - sliceBegin };
                        sliceReads.push_back(request);
                        isReceived[home] = 1;
                        nodeReceived[home]++;
                    }
                    if (isComplete) {
                        if (sliceReads.empty() == false) {
                            if (sliceReader == NULL) {
                                sliceReader = new BatchReader(par.threads, sliceReads[0].fileToRead);
                            }
                            sliceReader->readSync(sliceReads);
                        }
                        for (size_t i = 0; i < sliceReads.size(); i++) {
                            if (close(sliceReads[i].fileToRead) != 0) {
                                Debug(Debug::ERROR) << "Cannot close " << sliceReads[i].fileName << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                        }
                        sliceReads.clear();
                        isReceived.assign(nodes, 0);
                        {
                            std::lock_guard<std::mutex> lock(serviceMutex);
                            fetchedWindow = requestedWindow;
                        }
                        serviceCondition.notify_all();
                        requestedWindow = NULL;
                        hasWorked = true;
                    }
                }
                for (unsigned int i = 1; i < nodes; i++) {
                    const unsigned int node = (rank + i) % nodes;
                    while (isRequesting[node] && MMseqsNode::readExchangeFlush(requestIndexes[node], nodeRequests[node], requestLine)) {
                        nodeRequests[node]++;
                        hasWorked = true;
                        if (requestLine[1] != 0) {
                            isRequesting[node] = 0;
                            continue;
                        }
                        if (requestLine[3 + rank] == requestLine[2 + rank]) {
                            continue;
                        }
                        requestKeys.resize((requestLine[3 + rank] - requestLine[2 + rank]) / sizeof(DBKeyType));
                        MMseqsNode::readLog(requestFiles[node], requestLine[2 + rank], requestLine[3 + rank], reinterpret_cast<char *>(requestKeys.data()));
                        requestOffsets.resize(requestKeys.size() + 1);
                        requestOffsets[0] = 0;
#pragma omp parallel for schedule(static) num_threads(par.threads)
                        for (size_t j = 0; j < requestKeys.size(); j++) {
                            requestOffsets[j + 1] = seqDbr.getEntryLen(requestKeys[j]);
                        }
                        for (size_t j = 0; j < requestKeys.size(); j++) {
                            requestOffsets[j + 1] += requestOffsets[j];
                        }
                        const size_t sliceBytes = requestOffsets[requestKeys.size()];
                        if (slice.size() < sliceBytes) {
                            slice.resize(sliceBytes);
                        }
                        copyEntries(requestKeys.data(), requestKeys.size(), requestOffsets.data(), slice.data());
                        FILE *servedHandle = FileUtil::openAndDelete(servedFiles[node].c_str(), "w");
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                        for (size_t offset = 0; offset < sliceBytes; offset += REP_TARGETS_WRITE_BYTES) {
                            FileUtil::pwriteAll(fileno(servedHandle), servedFiles[node].c_str(), slice.data() + offset, std::min(REP_TARGETS_WRITE_BYTES, sliceBytes - offset), offset);
                        }
                        servedLine[0] = nodeServed[node]++;
                        MMseqsNode::publishExchangeFlush(servedHandle, servedFiles[node], sliceBytes, servedIndexes[node], servedLine);
                    }
                }
                // idle: poll again after up to 160 ms while the main thread may wait, else up to 1.28 s
                if (hasWorked) {
                    polls = 0;
                } else {
                    MMseqsNode::sleepBeforePoll(polls++, (requestedWindow != NULL || nextRequest != NULL) ? 4 : 7);
                }
            }
            delete sliceReader;
        });
    }
    const auto fetchWindow = [&](RepWindow &fetched) {
        std::lock_guard<std::mutex> lock(serviceMutex);
        fetchedWindow = NULL;
        pendingWindow = &fetched;
    };
    const auto waitForWindow = [&](RepWindow &fetched, size_t doneRepKey) {
        std::unique_lock<std::mutex> lock(serviceMutex);
        if (fetchedWindow == &fetched) {
            return;
        }
        lock.unlock();
        MMseqsNode::publishLog(acceptedHandle, acceptedFile, doneRepKey, acceptedBytes);
        publishedBytes = acceptedBytes;
        publishTime = time(NULL);
        Timer waitTimer;
        lock.lock();
        const bool waitsForMaster = isWaitingForMaster;
        serviceCondition.wait(lock, [&]() { return fetchedWindow == &fetched; });
        (waitsForMaster ? waitSeconds : sequenceWaitSeconds) += waitTimer.getTimediff();
    };
    if (hasService) {
        fetchWindow(window);
        waitForWindow(window, getFirstRepKey(window));
    }
    while (window.end > window.begin) {
        readWindow(nextWindow);
        fetchWindow(nextWindow);
        const size_t nextRepKey = getFirstRepKey(nextWindow);
        const std::vector<RepTargetsRecord> &records = chunkRecords[window.chunk];
        const size_t recordCount = window.end - window.begin;
        // batches of about the same rep length times targets, so the longest reps do not fall into one batch
        const size_t maxBatches = std::min(recordCount, par.threads * ALIGN_BATCHES_PER_THREAD);
        __extension__ typedef unsigned __int128 uint128;
        size_t windowCost = 0;
        for (size_t i = window.begin; i < window.end; i++) {
            windowCost += seqDbr.getSeqLen(records[i].header.repFirstTarget.repKey % paddedDbSize) * records[i].header.targetCount;
        }
        batchOffsets.assign(1, window.begin);
        for (size_t i = window.begin, cost = 0; i + 1 < window.end; i++) {
            cost += seqDbr.getSeqLen(records[i].header.repFirstTarget.repKey % paddedDbSize) * records[i].header.targetCount;
            if (static_cast<uint128>(cost) * maxBatches >= static_cast<uint128>(windowCost) * batchOffsets.size()) {
                batchOffsets.push_back(i + 1);
            }
        }
        batchOffsets.push_back(window.end);
        const size_t batchCount = batchOffsets.size() - 1;
        batchDone.assign(batchCount, 0);
        batchPairs.assign(batchCount, std::vector<ClusterPair>());
        size_t firstUnfinishedBatch = 0;
#pragma omp parallel
        {
            Sequence query(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
            Sequence target(maxSeqLen, Parameters::DBTYPE_AMINO_ACIDS, scoring.subMat, 0, false, par.compBiasCorrection);
            BlockAligner blockAligner(Parameters::DBTYPE_AMINO_ACIDS, maxSeqLen, scoring.subMat, &scoring.fastMatrix,
                                     &scoring.evaluer, par.compBiasCorrection, par.compBiasCorrectionScale,
                                     -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid());
            // --aligner-mode 3: the gapped alignments of the reps of a batch go through the SWIPE lanes together
            const bool swipeLanes = par.alignerMode == Parameters::ALIGNER_MODE_SWIPE_TRACEBACK && par.gapOpen.values.aminoacid() >= par.gapExtend.values.aminoacid();
            SwipeTargets swipe(par, scoring.subMat, scoring.evaluer, maxSeqLen);
            std::vector<ClusterPair> pairs;
            size_t threadAlignedPairs = 0;
#pragma omp for schedule(dynamic, 1)
            for (size_t batch = 0; batch < batchCount; batch++) {
                pairs.clear();
                for (size_t i = batchOffsets[batch]; i < batchOffsets[batch + 1]; i++) {
                    const RepTargets &header = records[i].header;
                    if (isAssigned(header.repFirstTarget.repKey)) {
                        continue;
                    }
                    const size_t repKey = header.repFirstTarget.repKey % paddedDbSize;
                    query.mapSequence(repKey, repKey, getWindowData(repKey), seqDbr.getSeqLen(repKey));
                    blockAligner.setQuery(&query);
                    bool blockAlignerInitialized = false;
                    swipe.startRep(header.repFirstTarget.repKey);
                    const DBKeyType *targets = chunkTargets[window.chunk].data() + records[i].targetOffset;
                    for (unsigned int j = 0; j < header.targetCount && isAssigned(header.repFirstTarget.repKey) == false; j++) {
                        const DBKeyType targetKey = targets[j];
                        // readWindow skipped the targets of reps assigned then, which the ring may have dropped since
                        const char *targetData = getWindowData(targetKey);
                        if (isAssigned(targetKey) || targetData == NULL) {
                            continue;
                        }
                        target.mapSequence(targetKey, targetKey, targetData, seqDbr.getSeqLen(targetKey));
                        threadAlignedPairs++;
                        unsigned short diagonal = 0;
                        if (alignTarget(par, blockAligner, blockAlignerInitialized, query, target, scoring.scorePerColThreshold, diagonal, NULL, 0, NULL,
                                        swipeLanes ? &swipe : NULL)) {
                            const ClusterPair pair = { header.repFirstTarget.repKey, targetKey };
                            pairs.push_back(pair);
                        }
                    }
                }
                alignTargetsSwipe(par, swipe, isAssigned, pairs);
                swipe.clearTargets();
                // the lanes finish the pairs of the batch out of rep order
                SORT_SERIAL(pairs.begin(), pairs.end(), ClusterPair::compareByRep);
                // pairs go to the log in batch order, so the reps before the first unfinished batch are done
                bool readsChanges = false;
                {
                    std::lock_guard<std::mutex> lock(acceptedMutex);
                    batchPairs[batch].swap(pairs);
                    batchDone[batch] = 1;
                    if (memberDbr != NULL) {
                        continue;
                    }
                    for (; firstUnfinishedBatch < batchCount && batchDone[firstUnfinishedBatch]; firstUnfinishedBatch++) {
                        logAcceptedPairs(batchPairs[firstUnfinishedBatch].data(), batchPairs[firstUnfinishedBatch].size());
                        std::vector<ClusterPair>().swap(batchPairs[firstUnfinishedBatch]);
                    }
                    const time_t now = time(NULL);
                    if (acceptedBytes - publishedBytes >= PUBLISH_BYTES || now - publishTime >= PUBLISH_SECONDS) {
                        const size_t doneRepKey = (firstUnfinishedBatch < batchCount) ? static_cast<size_t>(records[batchOffsets[firstUnfinishedBatch]].header.repFirstTarget.repKey) : nextRepKey;
                        MMseqsNode::publishLog(acceptedHandle, acceptedFile, doneRepKey, acceptedBytes);
                        publishedBytes = acceptedBytes;
                        publishTime = now;
                        readsChanges = doneRepKey < orderKeyEnd;
                    }
                    if (readsChanges || now - changesTime >= READ_SECONDS) {
                        readsChanges = true;
                        changesTime = now;
                    }
                }
                if (readsChanges) {
                    std::unique_lock<std::mutex> changesLock(assignedChangesMutex, std::try_to_lock);
                    if (changesLock.owns_lock()) {
                        readAssignedChanges();
                    }
                }
            }
            __sync_fetch_and_add(&alignedPairs, threadAlignedPairs);
        }
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
                    const bool swipeLanes = (par.alignerMode != Parameters::ALIGNER_MODE_DEFAULT && par.gapOpen.values.aminoacid() >= par.gapExtend.values.aminoacid());
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
                        query.mapSequence(repKey, repKey, getWindowData(repKey), seqDbr.getSeqLen(repKey));
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
            size_t acceptedCount = 0;
            for (size_t i = 0; i < windowPairs.size(); i++) {
                if (membersPassed[i]) {
                    windowPairs[acceptedCount++] = windowPairs[i];
                }
            }
            logAcceptedPairs(windowPairs.data(), acceptedCount);
            MMseqsNode::publishLog(acceptedHandle, acceptedFile, nextRepKey, acceptedBytes);
            publishedBytes = acceptedBytes;
            publishTime = time(NULL);
            if (nextRepKey < orderKeyEnd) {
                std::lock_guard<std::mutex> changesLock(assignedChangesMutex);
                readAssignedChanges();
            }
        }
        windowCount++;
        windowKeys += window.keys.size();
        waitForWindow(nextWindow, nextRepKey);
        std::swap(window, nextWindow);
    }
    MMseqsNode::addPhaseValue(phaseValues, "Windows", SSTR(windowCount));
    MMseqsNode::addPhaseValue(phaseValues, "Window keys", SSTR(windowKeys));
    MMseqsNode::publishLog(acceptedHandle, acceptedFile, orderKeyEnd, acceptedBytes);
    if (fclose(acceptedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << acceptedFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    // the service thread serves until every aligning node has requested its last window
    if (hasService) {
        serviceThread.join();
    }
    delete[] window.data;
    delete[] nextWindow.data;
    MMseqsNode::addPhaseValue(phaseValues, "Block read bytes", SSTR(seqDbr.getBlockReadBytes()));
    MMseqsNode::addPhaseValue(phaseValues, "Key read bytes", SSTR(seqDbr.getKeyReadBytes()));
    MMseqsNode::closeExchangeFiles(repFiles, repTargets);
    seqDbr.close();
    if (MMseqsNode::rank >= getFirstAlignNode()) {
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the master", waitSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the sequences", sequenceWaitSeconds);
        MMseqsNode::addPhaseValue(phaseValues, "Aligned pairs", SSTR(alignedPairs));
    }
}

// a rep of a clustering step, its accepted targets [begin, end) in logWords and the node of its log
struct AcceptedRep {
    size_t repKey;
    size_t begin;
    size_t end;
    unsigned int node;
};

// the merge gives the reps of node from firstNodeRep at once, at firstMergedRep of the merged reps
struct MergedReps {
    size_t firstNodeRep;
    size_t firstMergedRep;
    unsigned int node;
};

// the taken bits of a log in whole words; carry holds the bits of the last partial word
struct TakenLog {
    FILE *file;
    size_t bytes;
    uint64_t carry;
    unsigned int carryBits;
};

// --createdb-mode 3: the master clusters the accepted pairs in order key order; returns the member count
static size_t clusterAcceptedPairs(int threads, size_t dbSize, size_t memoryLimit, const std::vector<size_t> &groupKeyOffsets, size_t lagKeys, int mode, size_t paddedDbSize, size_t prefSizeBits,
                                   const std::string &accepted, const std::string &assignedChanges, const std::string &taken, const std::string &assignedBits, std::string &phaseValues) {
    const size_t assignedWords = (dbSize + 63) / 64;
    const size_t orderKeyEnd = (prefSizeBits - 1) * paddedDbSize + dbSize;
    const size_t orderWords = (orderKeyEnd + 63) / 64;
    uint64_t *assigned = new(std::nothrow) uint64_t[assignedWords];
    Util::checkAllocation(assigned, "Can not allocate assigned memory in align2clust");
    memset(assigned, 0, assignedWords * sizeof(uint64_t));
    std::vector<std::vector<DBKeyType> > threadAssignedKeys(threads);
    std::vector<DBKeyType> stepAssignedKeys;
    const auto isAssigned = [&](DBKeyType key) {
        return ((assigned[key / 64] >> (key % 64)) & 1) != 0;
    };
    const auto setAssigned = [&](DBKeyType key, std::vector<DBKeyType> &keys) {
        if ((__sync_fetch_and_or(&assigned[key / 64], 1ULL << (key % 64)) & (1ULL << (key % 64))) == 0) {
            keys.push_back(key);
        }
    };

    std::vector<unsigned char> nodeHasProgress(MMseqsNode::count, 0);
    std::vector<size_t> nodeRepKeys(MMseqsNode::count, 0);
    std::vector<size_t> nodeLogBytes(MMseqsNode::count, 0);
    std::vector<size_t> nodeReadBytes(MMseqsNode::count, 0);
    std::vector<TakenLog> nodeTaken(MMseqsNode::count);
    std::vector<std::vector<uint64_t> > nodeTakenBits(MMseqsNode::count);
    std::vector<size_t> nodeTakenCounts(MMseqsNode::count, 0);
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        const TakenLog takenLog = { FileUtil::openAndDelete((taken + "." + SSTR(node)).c_str(), "w"), 0, 0, 0 };
        nodeTaken[node] = takenLog;
    }
    FILE *changesHandle = FileUtil::openAndDelete(assignedChanges.c_str(), "w");
    size_t changesBytes = 0;
    // a step clusters at most stepWords words of the logs, each with up to four slots of keyReps, unless its first rep spans more
    const size_t stepWords = std::max<size_t>(1, memoryLimit / STEP_PAIRS_MEMORY_DIVISOR / (sizeof(uint64_t) + 4 * sizeof(std::pair<DBKeyType, size_t>)));
    const size_t partBytes = std::max<size_t>(1, std::min(stepWords, memoryLimit / LOG_PART_MEMORY_DIVISOR / sizeof(uint64_t))) * sizeof(uint64_t);
    std::vector<uint64_t> logWords;
    std::vector<size_t> nodeWordOffsets(MMseqsNode::count, 0);
    std::vector<std::vector<size_t> > nodeRepStarts(MMseqsNode::count);
    // per node: no unread word of its log has a smaller rep
    std::vector<size_t> nodeNextRepKeys(MMseqsNode::count, 0);
    std::vector<size_t> nodeOrder(MMseqsNode::count);
    std::vector<size_t> nodeMergedReps(MMseqsNode::count, 0);
    std::vector<MergedReps> mergedReps;
    std::vector<AcceptedRep> reps;
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
    const size_t setCoverBytes = memoryLimit - std::min(memoryLimit, getClusterStepMemory(dbSize, memoryLimit));
    if (mode == Parameters::SET_COVER) {
        memberPool.reserve(setCoverBytes / sizeof(DBKeyType));
        candidates.reserve(setCoverBytes / sizeof(SetCoverCandidate));
        pairSets.reserve(setCoverBytes / sizeof(ClusterPair));
    }
    const auto takePair = [&](DBKeyType repKey, DBKeyType memberKey) {
        const ClusterPair pair = { repKey, memberKey };
        TakenLog &nodeFile = nodeTaken[getAlignNode(groupKeyOffsets, repKey)];
        if (fwrite(&pair, sizeof(ClusterPair), 1, nodeFile.file) != 1) {
            Debug(Debug::ERROR) << "Cannot write " << taken << "\n";
            EXIT(EXIT_FAILURE);
        }
        nodeFile.bytes += sizeof(ClusterPair);
        memberCount++;
    };
    const auto getRepKey = [&](size_t word) {
        return static_cast<size_t>(logWords[word] & ~ACCEPTED_REP_BIT);
    };
    const auto isRepBefore = [&](size_t word, size_t repKey) {
        return getRepKey(word) < repKey;
    };
    const auto setTakenBit = [&](unsigned int node, size_t word) {
        const size_t bit = word - nodeWordOffsets[node];
        __sync_fetch_and_or(&nodeTakenBits[node][bit / 64], 1ULL << (bit % 64));
    };
    size_t clusteredPairs = 0;
    size_t logReadBytes = 0;
    double waitSeconds = 0;
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

        // read every log up to doneRepKey; past stepWords a log ends the step before its next rep, but the rep of clusteredKey is read whole
        logWords.clear();
        size_t stepRepKey = doneRepKey;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            nodeOrder[node] = node;
        }
        std::stable_sort(nodeOrder.begin(), nodeOrder.end(), [&](size_t first, size_t second) { return nodeNextRepKeys[first] < nodeNextRepKeys[second]; });
        for (size_t n = 0; n < MMseqsNode::count; n++) {
            const unsigned int node = nodeOrder[n];
            const std::string log = accepted + "." + SSTR(node);
            std::vector<size_t> &repStarts = nodeRepStarts[node];
            repStarts.clear();
            nodeWordOffsets[node] = logWords.size();
            if (nodeNextRepKeys[node] < stepRepKey) {
                nodeNextRepKeys[node] = nodeRepKeys[node];
                bool isCut = false;
                for (size_t from = nodeReadBytes[node]; isCut == false && from < nodeLogBytes[node];) {
                    const size_t to = std::min(nodeLogBytes[node], from + partBytes);
                    size_t word = logWords.size();
                    logWords.resize(word + (to - from) / sizeof(uint64_t));
                    MMseqsNode::readLog(log, from, to, reinterpret_cast<char *>(logWords.data() + word));
                    logReadBytes += to - from;
                    from = to;
                    for (; isCut == false && word < logWords.size(); word++) {
                        if ((logWords[word] & ACCEPTED_REP_BIT) == 0) {
                            continue;
                        }
                        const size_t repKey = getRepKey(word);
                        if (repKey >= stepRepKey || (word >= stepWords && repKey > clusteredKey)) {
                            nodeNextRepKeys[node] = repKey;
                            stepRepKey = std::min(stepRepKey, repKey);
                            logWords.resize(word);
                            isCut = true;
                        } else {
                            repStarts.push_back(word);
                        }
                    }
                }
            }
            repStarts.push_back(logWords.size());
        }
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            std::vector<size_t> &repStarts = nodeRepStarts[node];
            const size_t clusteredReps = std::lower_bound(repStarts.begin(), repStarts.end() - 1, stepRepKey, isRepBefore) - repStarts.begin();
            if (clusteredReps + 1 < repStarts.size()) {
                nodeNextRepKeys[node] = getRepKey(repStarts[clusteredReps]);
            }
            const size_t clusteredEnd = repStarts[clusteredReps];
            repStarts.resize(clusteredReps);
            repStarts.push_back(clusteredEnd);
            nodeTakenCounts[node] = clusteredEnd - nodeWordOffsets[node];
            nodeReadBytes[node] += nodeTakenCounts[node] * sizeof(uint64_t);
            nodeTakenBits[node].assign((nodeTakenCounts[node] + 63) / 64, 0);
        }
        // merge the reps of the logs in order key order, a node giving its reps up to the next rep of another node at once
        std::priority_queue<std::pair<size_t, unsigned int>, std::vector<std::pair<size_t, unsigned int> >, std::greater<std::pair<size_t, unsigned int> > > nextReps;
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            nodeMergedReps[node] = 0;
            if (nodeRepStarts[node].size() > 1) {
                nextReps.push(std::make_pair(getRepKey(nodeRepStarts[node][0]), node));
            }
        }
        mergedReps.clear();
        size_t repCount = 0;
        while (nextReps.empty() == false) {
            const unsigned int node = nextReps.top().second;
            nextReps.pop();
            const std::vector<size_t> &repStarts = nodeRepStarts[node];
            const size_t endKey = nextReps.empty() ? stepRepKey : nextReps.top().first;
            const size_t endRep = std::lower_bound(repStarts.begin() + nodeMergedReps[node] + 1, repStarts.end() - 1, endKey, isRepBefore) - repStarts.begin();
            const MergedReps nodeReps = { nodeMergedReps[node], repCount, node };
            mergedReps.push_back(nodeReps);
            repCount += endRep - nodeMergedReps[node];
            clusteredPairs += repStarts[endRep] - repStarts[nodeMergedReps[node]] - (endRep - nodeMergedReps[node]);
            nodeMergedReps[node] = endRep;
            if (endRep + 1 < repStarts.size()) {
                nextReps.push(std::make_pair(getRepKey(repStarts[endRep]), node));
            }
        }
        reps.resize(repCount);
#pragma omp parallel for schedule(static)
        for (size_t rep = 0; rep < repCount; rep++) {
            const MergedReps &nodeReps = *(std::upper_bound(mergedReps.begin(), mergedReps.end(), rep, [](size_t merged, const MergedReps &next) { return merged < next.firstMergedRep; }) - 1);
            const std::vector<size_t> &repStarts = nodeRepStarts[nodeReps.node];
            const size_t nodeRep = nodeReps.firstNodeRep + rep - nodeReps.firstMergedRep;
            const AcceptedRep acceptedRep = { getRepKey(repStarts[nodeRep]), repStarts[nodeRep] + 1, repStarts[nodeRep + 1], nodeReps.node };
            reps[rep] = acceptedRep;
        }

        // a pair with an assigned rep or member stays out, as bits are never cleared
        pairIsFree.resize(logWords.size());
        repKeyOffsets.assign(repCount + 1, 0);
#pragma omp parallel for schedule(dynamic, 1024)
        for (size_t rep = 0; rep < repCount; rep++) {
            const bool repIsFree = isAssigned(reps[rep].repKey % paddedDbSize) == false;
            size_t freeCount = 0;
            for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                pairIsFree[i] = repIsFree && isAssigned(logWords[i]) == false;
                freeCount += pairIsFree[i];
            }
            repKeyOffsets[rep + 1] = (freeCount > 0) ? freeCount + 1 : 0;
        }
        for (size_t rep = 0; rep < repCount; rep++) {
            repKeyOffsets[rep + 1] += repKeyOffsets[rep];
        }
        if (mode == Parameters::SET_COVER) {
            // a set waits until no later rep can have a larger one, the sets of two as pairs
            repSetOffsets.resize(repCount);
            size_t setMembers = 0;
            size_t setPairs = 0;
            for (size_t rep = 0; rep < repCount; rep++) {
                const size_t setSize = repKeyOffsets[rep + 1] - repKeyOffsets[rep];
                repSetOffsets[rep] = (setSize == 2) ? setPairs++ : setMembers;
                if (setSize > 2) {
                    setMembers += setSize;
                }
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
                const DBKeyType repKey = reps[rep].repKey % paddedDbSize;
                size_t member = poolOffset + repSetOffsets[rep];
                if (setSize > 2) {
                    memberPool[member++] = repKey;
                }
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (pairIsFree[i] && setSize == 2) {
                        const ClusterPair pairSet = { repKey, static_cast<DBKeyType>(logWords[i]) };
                        pairSets[pairOffset + repSetOffsets[rep]] = pairSet;
                    } else if (pairIsFree[i]) {
                        memberPool[member++] = logWords[i];
                    }
                }
            }
            for (size_t rep = 0; rep < repCount; rep++) {
                if (repKeyOffsets[rep + 1] - repKeyOffsets[rep] > 2) {
                    const SetCoverCandidate candidate = { static_cast<DBLocalId>(reps[rep].repKey % paddedDbSize), repKeyOffsets[rep + 1] - repKeyOffsets[rep], poolOffset + repSetOffsets[rep] };
                    candidates.push_back(candidate);
                    std::push_heap(candidates.begin(), candidates.end(), setCoverComparator);
                }
            }
            liveMemberCount += setMembers;
            // the reps from stepRepKey on have fewer than 2^(prefSizeBit + 1) targets, so no larger set can come
            const size_t maxSetSize = (stepRepKey < orderKeyEnd) ? static_cast<size_t>(2) << (prefSizeBits - 1 - stepRepKey / paddedDbSize) : 0;
            takeSetCoverCandidates(candidates, memberPool, liveMemberCount, maxSetSize, &pairSets, isAssigned, [&](DBKeyType repKey, DBKeyType memberKey) {
                setAssigned(memberKey, threadAssignedKeys[0]);
                if (memberKey != repKey) {
                    takePair(repKey, memberKey);
                }
            });
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
            // keyReps: the first rep of every key of the step in a table of at least twice the keys; every later rep of a key shares it
            size_t tableBits = 1;
            while ((1ULL << tableBits) < 2 * repKeyOffsets[repCount]) {
                tableBits++;
            }
            const size_t tableSize = 1ULL << tableBits;
            keyReps.resize(std::max(keyReps.size(), tableSize));
            repSharesKey.assign(repCount, 0);
#pragma omp parallel for schedule(static)
            for (size_t slot = 0; slot < tableSize; slot++) {
                keyReps[slot] = std::make_pair(DB_KEY_INVALID, std::numeric_limits<size_t>::max());
            }
            const auto addStepKey = [&](DBKeyType key, size_t rep) {
                size_t slot = (key * 0x9E3779B97F4A7C15ULL) >> (64 - tableBits);
                DBKeyType slotKey = DB_KEY_INVALID;
                while (__atomic_compare_exchange_n(&keyReps[slot].first, &slotKey, key, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED) == false && slotKey != key) {
                    slot = (slot + 1) & (tableSize - 1);
                    slotKey = DB_KEY_INVALID;
                }
                size_t firstRep = __atomic_load_n(&keyReps[slot].second, __ATOMIC_RELAXED);
                while (rep < firstRep && __atomic_compare_exchange_n(&keyReps[slot].second, &firstRep, rep, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED) == false) {
                }
                if (firstRep != std::numeric_limits<size_t>::max()) {
                    __atomic_store_n(&repSharesKey[std::max(rep, firstRep)], 1, __ATOMIC_RELAXED);
                }
            };
#pragma omp parallel for schedule(dynamic, 1024)
            for (size_t rep = 0; rep < repCount; rep++) {
                if (repKeyOffsets[rep + 1] == repKeyOffsets[rep]) {
                    continue;
                }
                addStepKey(reps[rep].repKey, rep);
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (pairIsFree[i]) {
                        addStepKey(logWords[i], rep);
                    }
                }
            }
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
                    setAssigned(reps[rep].repKey, threadAssignedKeys[thread_idx]);
                    for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                        if (pairIsFree[i]) {
                            setAssigned(logWords[i], threadAssignedKeys[thread_idx]);
                            setTakenBit(reps[rep].node, i);
                            memberCount++;
                        }
                    }
                }
            }
            // the sharing reps take in key order; no rep that does not share holds a key of theirs
            for (size_t rep = 0; rep < repCount; rep++) {
                const DBKeyType repKey = reps[rep].repKey;
                if (repKeyOffsets[rep + 1] == repKeyOffsets[rep] || repSharesKey[rep] == 0 || isAssigned(repKey)) {
                    continue;
                }
                for (size_t i = reps[rep].begin; i < reps[rep].end; i++) {
                    if (i + PREFETCH_PAIRS < reps[rep].end) {
                        __builtin_prefetch(&assigned[logWords[i + PREFETCH_PAIRS] / 64]);
                    }
                    const DBKeyType memberKey = logWords[i];
                    if (pairIsFree[i] && isAssigned(memberKey) == false) {
                        setAssigned(repKey, threadAssignedKeys[0]);
                        setAssigned(memberKey, threadAssignedKeys[0]);
                        setTakenBit(reps[rep].node, i);
                        memberCount++;
                    }
                }
            }
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                TakenLog &nodeFile = nodeTaken[node];
                words.clear();
                for (size_t i = 0; i < nodeTakenCounts[node]; i += 64) {
                    const uint64_t bits = nodeTakenBits[node][i / 64];
                    const unsigned int wordBits = std::min<size_t>(64, nodeTakenCounts[node] - i);
                    nodeFile.carry |= bits << nodeFile.carryBits;
                    if (nodeFile.carryBits + wordBits >= 64) {
                        words.push_back(nodeFile.carry);
                        nodeFile.carry = (nodeFile.carryBits > 0) ? bits >> (64 - nodeFile.carryBits) : 0;
                        nodeFile.carryBits = nodeFile.carryBits + wordBits - 64;
                    } else {
                        nodeFile.carryBits += wordBits;
                    }
                }
                if (fwrite(words.data(), sizeof(uint64_t), words.size(), nodeFile.file) != words.size()) {
                    Debug(Debug::ERROR) << "Cannot write " << taken << "." << node << "\n";
                    EXIT(EXIT_FAILURE);
                }
                nodeFile.bytes += words.size() * sizeof(uint64_t);
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
        stepAssignedKeys.clear();
        for (int thread = 0; thread < threads; thread++) {
            stepAssignedKeys.insert(stepAssignedKeys.end(), threadAssignedKeys[thread].begin(), threadAssignedKeys[thread].end());
            threadAssignedKeys[thread].clear();
        }
#pragma omp parallel for schedule(static)
        for (int thread = 0; thread < threads; thread++) {
            std::vector<size_t> &sectionCounts = threadSectionCounts[thread];
            sectionCounts.assign(MMseqsNode::count + 1, 0);
            for (size_t i = stepAssignedKeys.size() * thread / threads; i < stepAssignedKeys.size() * (thread + 1) / threads; i++) {
                const DBKeyType key = stepAssignedKeys[i];
                sectionCounts[getAlignNode(groupKeyOffsets, key) + 1]++;
                sectionCounts[0] += isRingKey(key);
            }
        }
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
            for (size_t i = stepAssignedKeys.size() * thread / threads; i < stepAssignedKeys.size() * (thread + 1) / threads; i++) {
                const DBKeyType key = stepAssignedKeys[i];
                changes[headerWords + enteringWords + sectionPositions[getAlignNode(groupKeyOffsets, key) + 1]++] = key;
                if (isRingKey(key)) {
                    changes[headerWords + enteringWords + sectionPositions[0]++] = key;
                }
            }
        }
        if (fwrite(changes.data(), sizeof(uint64_t), changes.size(), changesHandle) != changes.size()) {
            Debug(Debug::ERROR) << "Cannot write " << assignedChanges << "\n";
            EXIT(EXIT_FAILURE);
        }
        changesBytes += changes.size() * sizeof(uint64_t);
        MMseqsNode::publishLog(changesHandle, assignedChanges, clusteredKey, changesBytes);
    }
    MMseqsNode::addPhaseSeconds(phaseValues, "clustering the accepted pairs", clusterTimer.getTimediff() - waitSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "waiting for the aligning nodes", waitSeconds);
    MMseqsNode::addPhaseValue(phaseValues, "Clustered pairs", SSTR(clusteredPairs));
    MMseqsNode::addPhaseValue(phaseValues, "Accepted log bytes read", SSTR(logReadBytes));
    // every node is done with both logs
    if (fclose(changesHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << assignedChanges << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::remove(assignedChanges.c_str());
    FileUtil::remove((assignedChanges + ".progress").c_str());
    // every node reads the assigned bits of its keys; an assigned key is no singleton
    FILE *assignedHandle = FileUtil::openAndDelete(assignedBits.c_str(), "w");
    if (fwrite(assigned, sizeof(uint64_t), assignedWords, assignedHandle) != assignedWords) {
        Debug(Debug::ERROR) << "Cannot write " << assignedBits << "\n";
        EXIT(EXIT_FAILURE);
    }
    MMseqsNode::publishLog(assignedHandle, assignedBits, dbSize, assignedWords * sizeof(uint64_t));
    if (fclose(assignedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << assignedBits << "\n";
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
    std::string phaseValues;
    Timer phaseValueTimer;
    std::string seqDb = par.db1;
    std::vector<std::string> localCopies;
    size_t localCopyMemory = 0;
    if (par.localDbDir.empty() == false) {
        seqDb = FileUtil::copyToLocalDir(par.db1, par.localDbDir + "/" + FileUtil::baseName(par.db3) + "." + SSTR(MMseqsNode::rank), par.threads, localCopies, localCopyMemory);
        MMseqsNode::addPhaseTime(phaseValues, "copying the sequences to --local-db-dir", phaseValueTimer);
    }
    DBReader<DBKeyType> seqDbr(seqDb.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    seqDbr.open(DBReader<DBKeyType>::NOSORT);
    const std::vector<size_t> nodeKeyOffsets = seqDbr.getNodeKeyOffsets(MMseqsNode::count);
    const std::vector<size_t> fileKeyOffsets = seqDbr.getFileKeyOffsets(MMseqsNode::count);
    const size_t dbSize = seqDbr.getSize();
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    // a copy on tmpfs holds the sequences in memory
    const size_t nodeMemoryLimit = std::max(Util::computeMemory(par.splitMemoryLimit), localCopyMemory) - localCopyMemory;
    const size_t masterMemoryLimit = nodeMemoryLimit / ((getFirstAlignNode() == 0) ? MASTER_MEMORY_DIVISOR : 1);
    const size_t masterMemory = (par.clusteringMode == Parameters::SET_COVER) ? masterMemoryLimit : getClusterStepMemory(dbSize, masterMemoryLimit);
    const size_t memoryLimit = nodeMemoryLimit - ((getFirstAlignNode() == 0) ? std::min(masterMemoryLimit, masterMemory) : 0);
    const int clusterThreads = (getFirstAlignNode() > 0) ? par.threads : std::min(CLUSTER_THREADS, std::max(1, par.threads / 4));

    // a clusthashfast subset has bits; its sixth index column counts the representatives for the e-values
    const bool hasRepresentativeBits = FileUtil::fileExists((par.db1 + ".bits." + SSTR(MMseqsNode::rank)).c_str());
    size_t representatives = 0;
    size_t representativeResidues = 0;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        representatives += seqDbr.getRangeRepresentativeCount(range);
        representativeResidues += seqDbr.getRangeRepresentativeCount(range) * (index[range].length - 2);
    }

    const std::string accepted = par.db3 + "_accepted";
    const std::string assignedChanges = par.db3 + "_assigned";
    const std::string taken = par.db3 + "_taken";
    const std::string assignedBits = par.db3 + "_assigned.bits";
    // the sequence exchange indexes exist from before the memory barrier on, so no node looks one up before it is there
    const std::string requests = par.db3 + "_swap_2";
    const std::string served = par.db3 + "_swap_3";
    std::vector<std::string> sequenceIndexes;
    std::vector<std::string> sequenceFiles;
    if (MMseqsNode::rank >= getFirstAlignNode() && MMseqsNode::count - getFirstAlignNode() > 1) {
        sequenceIndexes.push_back(requests + ".index." + SSTR(MMseqsNode::rank));
        sequenceFiles.push_back(requests + "." + SSTR(MMseqsNode::rank));
        for (size_t node = getFirstAlignNode(); node < MMseqsNode::count; node++) {
            if (node != MMseqsNode::rank) {
                sequenceIndexes.push_back(served + ".index." + SSTR(MMseqsNode::rank) + "." + SSTR(node));
                sequenceFiles.push_back(served + "." + SSTR(MMseqsNode::rank) + "." + SSTR(node));
            }
        }
    }
    for (size_t i = 0; i < sequenceIndexes.size(); i++) {
        if (fclose(FileUtil::openAndDelete(sequenceIndexes[i].c_str(), "w")) != 0) {
            Debug(Debug::ERROR) << "Cannot create " << sequenceIndexes[i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }

    // all nodes use the smallest memory limit, so they agree on the splits and the ring and fail together
    const size_t smallestMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db3 + ".memory", nodeMemoryLimit);
    const size_t assignedWords = (dbSize + 63) / 64;
    const size_t paddedDbSize = assignedWords * 64;
    const size_t prefSizeBits = (par.clusteringMode == Parameters::SET_COVER) ? 64 - __builtin_clzll(dbSize | 1) : 1;
    const size_t maxWindowBytes = std::max<size_t>(1, smallestMemoryLimit / WINDOW_MEMORY_DIVISOR);
    const size_t lagKeys = std::max<size_t>(64, smallestMemoryLimit / RING_MEMORY_DIVISOR * 8 / 64 * 64);

    // every partition of the prefilter is a group of keys, taken by the aligning nodes in turn so that they align close keys at the same time
    const std::string partitionsFile = par.db2 + ".partitions";
    const std::string partitionsData = MMseqsNode::readFile(partitionsFile);
    std::vector<PrefPartition> partitions;
    for (char *line = (char *) partitionsData.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[5];
        Util::getWordsOfLine(line, columns, 5);
        PrefPartition partition;
        partition.firstKey = Util::fast_atoi<size_t>(columns[0]);
        partition.prefOffset = Util::fast_atoi<size_t>(columns[1]);
        partition.prefBytes = Util::fast_atoi<size_t>(columns[2]);
        partition.cost = Util::fast_atoi<size_t>(columns[3]);
        partition.node = Util::fast_atoi<size_t>(columns[4]);
        partitions.push_back(partition);
    }
    std::vector<size_t> groupKeyOffsets(1, 0);
    std::vector<PrefPartition> nodePartitions;
    for (size_t i = 0; i < partitions.size(); i++) {
        if (i > 0) {
            groupKeyOffsets.push_back(partitions[i].firstKey);
        }
        if (getGroupNode(i) == MMseqsNode::rank && partitions[i].cost > 0) {
            nodePartitions.push_back(partitions[i]);
        }
    }
    groupKeyOffsets.push_back(dbSize);

    const std::string memberRequests = par.db3 + "_swap_0";
    const std::string membersExchange = par.db3 + "_swap_1";
    MMseqsNode::ClusterPairWriter *memberRequestWriter = NULL;
    const std::vector<size_t> filterCluKeyOffsets = par.filterCluDBFile.empty() ? nodeKeyOffsets : MMseqsNode::getClusterKeyOffsets(par.db3 + ".keys", par.filterCluDBFile, dbSize);
    if (par.filterCluDBFile.empty() == false) {
        memberRequestWriter = new MMseqsNode::ClusterPairWriter(memberRequests, std::max<size_t>(1, memoryLimit / CLUSTER_PAIRS_MEMORY_DIVISOR / sizeof(ClusterPair)),
                                                                ClusterPair::compareByRep, filterCluKeyOffsets);
    }
    const std::string repTargets = par.db3 + "_rep_targets";
    const std::string repTargetsIndex = writeRepTargets(par, seqDbr, nodePartitions, maxWindowBytes, memoryLimit, paddedDbSize, prefSizeBits, repTargets, memberRequestWriter, phaseValues);
    MMseqsNode::addPhaseTime(phaseValues, "writing the rep targets", phaseValueTimer);

    std::vector<ClusterPair> members;
    DBReader<DBKeyType> *memberDbr = NULL;
    if (memberRequestWriter != NULL) {
        const std::string memberRequestsIndex = memberRequestWriter->close();
        delete memberRequestWriter;
        const std::string membersIndex = MMseqsNode::sendClusterMembers(par.filterCluDBFile, memberRequestsIndex, memberRequests, membersExchange);
        // every node has read the requests
        FileUtil::remove((memberRequests + "." + SSTR(MMseqsNode::rank)).c_str());
        MMseqsNode::readExchangePairs(membersIndex, membersExchange, members);
        SORT_PARALLEL(members.begin(), members.end(), ClusterPair::compareByRep);
        memberDbr = new DBReader<DBKeyType>(par.filterSeqDBFile.c_str(), (par.filterSeqDBFile + ".index").c_str(), par.threads,
                                            DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        memberDbr->open(DBReader<DBKeyType>::NOSORT);
        MMseqsNode::addPhaseTime(phaseValues, "sending the cluster members", phaseValueTimer);
    }

    // the master clusters while the nodes align; on several nodes its log stays empty
    size_t memberCount = 0;
    std::string clusterPhaseValues;
    std::thread clusterThread;
    if (MMseqsNode::isMaster()) {
        clusterThread = std::thread([&]() {
#ifdef OPENMP
            omp_set_num_threads(clusterThreads);
#endif
            memberCount = clusterAcceptedPairs(clusterThreads, dbSize, masterMemoryLimit, groupKeyOffsets, lagKeys, par.clusteringMode, paddedDbSize, prefSizeBits,
                                               accepted, assignedChanges, taken, assignedBits, clusterPhaseValues);
        });
    }
    phaseValueTimer.reset();
    alignRepTargets(par, seqDbr, groupKeyOffsets, lagKeys, memoryLimit, representativeResidues, paddedDbSize, prefSizeBits, repTargetsIndex, repTargets, accepted, assignedChanges,
                    requests, served, members, memberDbr, phaseValues);
    FileUtil::remove((repTargets + "." + SSTR(MMseqsNode::rank)).c_str());
    // alignRepTargets closed seqDbr, the last reader of the copy
    for (size_t i = 0; i < localCopies.size(); i++) {
        FileUtil::remove(localCopies[i].c_str());
    }
    if (localCopies.empty() == false) {
        rmdir(FileUtil::dirName(seqDb).c_str());
    }
    if (MMseqsNode::rank >= getFirstAlignNode()) {
        MMseqsNode::addPhaseTime(phaseValues, "aligning the rep targets", phaseValueTimer);
        if (par.alignerMode != Parameters::ALIGNER_MODE_DEFAULT) {
            printSwipeLanes();
        }
    }
    MMseqsNode::addPhaseValue(phaseValues, "Memory limit", SSTR(memoryLimit));

    const std::string takenFile = taken + "." + SSTR(MMseqsNode::rank);
    size_t takenKey;
    size_t takenBytes;
    MMseqsNode::waitForLog(takenFile, takenKey, takenBytes);
    if (MMseqsNode::isMaster()) {
        clusterThread.join();
        Debug(Debug::INFO) << "Number of clusters: " << representatives - memberCount << "\n";
        phaseValues += clusterPhaseValues;
    }
    MMseqsNode::addPhaseTime(phaseValues, "waiting for the clustering", phaseValueTimer);
    const size_t nodeFirstKey = nodeKeyOffsets[MMseqsNode::rank];
    const size_t nodeKeys = nodeKeyOffsets[MMseqsNode::rank + 1] - nodeFirstKey;
    const size_t nodeWords = (nodeKeys + 63) / 64;
    const std::string clusterPairs = par.db3 + "_swap_4";
    MMseqsNode::ClusterPairWriter clusterPairWriter(clusterPairs, std::max<size_t>(64, memoryLimit / CLUSTER_PAIRS_MEMORY_DIVISOR / sizeof(ClusterPair)), ClusterPair::compareByRep, nodeKeyOffsets);
    const std::string acceptedFile = accepted + "." + SSTR(MMseqsNode::rank);
    if (par.clusteringMode == Parameters::SET_COVER) {
        for (size_t from = 0; from < takenBytes;) {
            const size_t to = std::min(takenBytes, from + clusterPairWriter.pairCapacity * sizeof(ClusterPair));
            MMseqsNode::readLog(takenFile, from, to, reinterpret_cast<char *>(clusterPairWriter.reservePairs((to - from) / sizeof(ClusterPair))));
            from = to;
        }
    } else {
        std::vector<uint64_t> takenBits(takenBytes / sizeof(uint64_t));
        MMseqsNode::readLog(takenFile, 0, takenBits.size() * sizeof(uint64_t), reinterpret_cast<char *>(takenBits.data()));
        FILE *acceptedHandle = FileUtil::openFileOrDie(acceptedFile.c_str(), "r", true);
        // a chunk takes whole words of taken bits and fits into the writer; bit i is word i of the log
        const size_t chunkWords = std::min(clusterPairWriter.pairCapacity, std::max<size_t>(1, takenBits.size()) * 64) / 64 * 64;
        std::vector<uint64_t> acceptedWords(chunkWords);
        std::vector<std::vector<size_t> > threadRepStarts(par.threads);
        std::vector<size_t> repStarts;
        uint64_t previousRepWord = 0;
        for (size_t wordCount, firstWord = 0; (wordCount = fread(acceptedWords.data(), sizeof(uint64_t), chunkWords, acceptedHandle)) > 0; firstWord += chunkWords / 64) {
            const size_t takenWordCount = (wordCount + 63) / 64;
            const uint64_t *takenWords = takenBits.data() + firstWord;
            size_t takenCount = 0;
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                threadRepStarts[thread_idx].clear();
#pragma omp for schedule(static) reduction(+:takenCount)
                for (size_t word = 0; word < takenWordCount; word++) {
                    takenCount += __builtin_popcountll(takenWords[word]);
                    for (size_t i = word * 64; i < std::min(wordCount, word * 64 + 64); i++) {
                        if (acceptedWords[i] & ACCEPTED_REP_BIT) {
                            threadRepStarts[thread_idx].push_back(i);
                        }
                    }
                }
            }
            repStarts.clear();
            for (int thread = 0; thread < par.threads; thread++) {
                repStarts.insert(repStarts.end(), threadRepStarts[thread].begin(), threadRepStarts[thread].end());
            }
            ClusterPair *takenPairs = clusterPairWriter.reservePairs(takenCount);
            size_t takenPairCount = 0;
#pragma omp parallel for schedule(static)
            for (size_t word = 0; word < takenWordCount; word++) {
                uint64_t bits = takenWords[word];
                if (bits == 0) {
                    continue;
                }
                size_t pairOffset = __sync_fetch_and_add(&takenPairCount, __builtin_popcountll(bits));
                size_t rep = std::upper_bound(repStarts.begin(), repStarts.end(), word * 64) - repStarts.begin();
                for (; bits != 0; bits &= bits - 1) {
                    const size_t i = word * 64 + __builtin_ctzll(bits);
                    while (rep < repStarts.size() && repStarts[rep] < i) {
                        rep++;
                    }
                    const uint64_t repWord = (rep > 0) ? acceptedWords[repStarts[rep - 1]] : previousRepWord;
                    const ClusterPair takenPair = { static_cast<DBKeyType>((repWord & ~ACCEPTED_REP_BIT) % paddedDbSize), static_cast<DBKeyType>(acceptedWords[i]) };
                    takenPairs[pairOffset++] = takenPair;
                }
            }
            if (repStarts.empty() == false) {
                previousRepWord = acceptedWords[repStarts.back()];
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
    const std::string clusterPairsIndex = clusterPairWriter.close();
    // every node has read the sequences that this node requested and served
    for (size_t i = 0; i < sequenceIndexes.size(); i++) {
        FileUtil::remove(sequenceIndexes[i].c_str());
        if (FileUtil::fileExists(sequenceFiles[i].c_str())) {
            FileUtil::remove(sequenceFiles[i].c_str());
        }
    }
    MMseqsNode::addPhaseTime(phaseValues, "sending the cluster pairs", phaseValueTimer);
    // every node has read the members that this node sent
    std::vector<std::string> exchanges(1, clusterPairs);
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
    MMseqsNode::waitForLog(assignedBits, assignedKey, assignedBytes);
    FILE *assignedHandle = FileUtil::openFileOrDie(assignedBits.c_str(), "r", true);
    std::vector<uint64_t> nodeAssignedBits(nodeWords + 1, 0);
    MMseqsNode::readKeyBits(fileno(assignedHandle), assignedBits, 0, 0, nodeFirstKey, nodeFirstKey + nodeKeys, nodeFirstKey, nodeAssignedBits.data());
    if (fclose(assignedHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << assignedBits << "\n";
        EXIT(EXIT_FAILURE);
    }
    uint64_t *singletonBits = NULL;
    if (hasRepresentativeBits) {
        singletonBits = MMseqsNode::readRepresentativeBits(par.db1, fileKeyOffsets, nodeFirstKey, nodeKeys);
    } else {
        singletonBits = new(std::nothrow) uint64_t[nodeWords];
        Util::checkAllocation(singletonBits, "Can not allocate singletonBits memory in align2clust");
        memset(singletonBits, 0xFF, nodeWords * sizeof(uint64_t));
    }
    for (size_t word = 0; word < nodeWords; word++) {
        singletonBits[word] &= ~nodeAssignedBits[word];
    }
    std::vector<uint64_t>().swap(nodeAssignedBits);
    MMseqsNode::writeClusterPairs(clusterPairsIndex, clusterPairs, par.db3, singletonBits, nodeFirstKey, nodeKeys);
    delete[] singletonBits;
    MMseqsNode::addPhaseTime(phaseValues, "writing the clusters", phaseValueTimer);
    MMseqsNode::removeExchanges(par.db3, clusterPairs, exchanges, phaseValues);
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db3 + ".memory").c_str());
        FileUtil::remove(assignedBits.c_str());
        FileUtil::remove((assignedBits + ".progress").c_str());
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
