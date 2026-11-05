#ifndef ALIGN2CLUST_H
#define ALIGN2CLUST_H

#include "Parameters.h"
#include "Matcher.h"
#include "DBReader.h"
#include "SubstitutionMatrix.h"
#include "BlockAligner.h"
#include "SwipeAligner.h"
#include "MMseqsNode.h"
#include <atomic>
#include <string>
#include <vector>

#define MIN_SIZE 32

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

typedef std::atomic<DBLocalId> ClusterAssignment;

struct AlignmentScoring {
    BaseMatrix *subMat;
    SubstitutionMatrix::FastMatrix fastMatrix;
    // the minimum ungapped score per column of a pair that is aligned with gaps
    float scorePerColThreshold;
    EvalueComputation evaluer;

    AlignmentScoring(const Parameters &par, size_t dbResidues);

    ~AlignmentScoring() {
        delete[] fastMatrix.matrix;
        delete[] fastMatrix.matrixData;
        delete subMat;
    }
};

extern std::atomic<size_t> swipeLanePairs;
extern std::atomic<size_t> swipeLaneSlots;
extern std::atomic<size_t> swipeSinglePairs;
extern std::atomic<size_t> swipeMemberExtensions;
extern std::atomic<size_t> swipeMemberLaneRows;
extern std::atomic<size_t> swipeMemberSlotRows;
extern std::atomic<size_t> swipeTargetPairs;
extern std::atomic<size_t> swipeTargetLaneRows;
extern std::atomic<size_t> swipeTargetSlotRows;
extern std::atomic<size_t> swipeTargetLongPairs;
extern std::atomic<size_t> swipeByteLaneRows;
extern std::atomic<size_t> swipeByteSlotRows;
extern std::atomic<size_t> swipeRerunPairs;

// per thread buffers of --align2clust-aligner-mode 1: the cluster members of a rep that go through the SWIPE lanes
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
        gapped.clear();
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

// per thread queue of the rep targets for the SWIPE extensions (--align2clust-aligner-mode 1 on mode 3 DBs)
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
    // the extensions of the member gates, which count apart from the targets
    bool forMembers;

    SwipeTargets(const Parameters &par, BaseMatrix *subMat, EvalueComputation &evaluer, size_t maxSeqLen, bool forMembers = false)
        : extender(subMat, maxSeqLen, par.compBiasCorrection, par.compBiasCorrectionScale, par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid()),
          evaluer(evaluer), repKey(0), initialized(false), queryId(0), targetPairs(0), forMembers(forMembers) {}

    ~SwipeTargets() {
        if (forMembers) {
            swipeMemberExtensions += targetPairs;
            swipeMemberLaneRows += extender.extendLaneRows;
            swipeMemberSlotRows += extender.extendSlotRows;
            swipeByteLaneRows += extender.extendByteLaneRows;
            swipeByteSlotRows += extender.extendByteSlotRows;
            swipeRerunPairs += extender.extendRerunJobs;
            return;
        }
        swipeTargetPairs += targetPairs;
        swipeTargetLaneRows += extender.extendLaneRows;
        swipeTargetSlotRows += extender.extendSlotRows;
        swipeTargetLongPairs += extender.extendLongJobs;
        swipeByteLaneRows += extender.extendByteLaneRows;
        swipeByteSlotRows += extender.extendByteSlotRows;
        swipeRerunPairs += extender.extendRerunJobs;
    }

    void addTarget(Sequence &query, Sequence &target, DBKeyType targetKey, int seedQuery, int seedTarget) {
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
        targetKeys.push_back(targetKey);
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

// per thread buffers of the member gates; with SWIPE (--align2clust-aligner-mode 1) the gates of all targets of a rep can wait and go in rounds
// that share the lanes (deferGates)
struct ClusterMembers {
    std::vector<MemberGate> gates;
    size_t gateCount;
    std::vector<size_t> poolGates;
    std::vector<size_t> lanePool;
    SwipeBatch *swipe;
    bool deferGates;
    // the gapped members of a round of alignMemberGatesSwipe and the ones its extensions accepted
    SwipeTargets *targets;
    std::vector<size_t> gappedGates;
    std::vector<size_t> gappedMembers;
    std::vector<ClusterPair> extendedPairs;

    ClusterMembers(SwipeBatch *swipe, bool deferGates, SwipeTargets *targets = NULL) : gateCount(0), swipe(swipe), deferGates(deferGates), targets(targets) {}

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

Matcher::result_t swipeResult(const Parameters &par, EvalueComputation &evaluer, DBKeyType key, int rawScore, const s_align &alignment,
                              int queryLength, int targetLength, const std::string &backtrace);
void alignMembersSwipe(const Parameters &par, Matcher &matcher, bool &matcherInitialized, ClusterMembers &members, Sequence &query, Sequence &element,
                       DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer);
bool canCoverMember(const Parameters &par, int queryLength, int memberLength);
void prepareMemberGates(const Parameters &par, Sequence &query, DBReader<DBKeyType> *cluSeqDbr, bool withRecords, ClusterMembers &members, size_t gateBegin, size_t gateEnd);
bool alignMemberRound(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                      DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer, ClusterMembers &members,
                      size_t gateBegin, size_t gateEnd, size_t batchSize);
void alignClusterMembers(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                         DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer,
                         ClusterMembers &members);
bool alignTarget(const Parameters &par, BlockAligner &blockAligner, bool &blockAlignerInitialized, Sequence &query, Sequence &target,
                 float scorePerColThreshold, unsigned short &diagonal, const ClusterAssignment *assignedCluster, size_t targetId, Matcher::result_t *result,
                 SwipeTargets *swipe);
int align2clustLinfinity(Parameters &par);

#endif
