#ifndef ALIGN2CLUST_H
#define ALIGN2CLUST_H

#include "Parameters.h"
#include "Matcher.h"
#include "DBReader.h"
#include "SubstitutionMatrix.h"
#include "BlockAligner.h"
#include "SwipeAligner.h"
#include "MMseqsNode.h"
#include "Util.h"
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
    // min ungapped score per column for a gapped alignment
    float scorePerColThreshold;
    EvalueComputation evaluer;

    AlignmentScoring(const Parameters &par, size_t dbResidues);

    ~AlignmentScoring() {
        delete[] fastMatrix.matrix;
        delete[] fastMatrix.matrixData;
        delete subMat;
    }
};

extern std::atomic<size_t> swipeLanePairCount;
extern std::atomic<size_t> swipeLaneCapacity;
extern std::atomic<size_t> swipeScalarPairCount;
extern std::atomic<size_t> swipeMemberExtensionCount;
extern std::atomic<size_t> swipeMemberActiveLaneRowCount;
extern std::atomic<size_t> swipeMemberTotalLaneRowCount;
extern std::atomic<size_t> swipeTargetPairCount;
extern std::atomic<size_t> swipeTargetActiveLaneRowCount;
extern std::atomic<size_t> swipeTargetTotalLaneRowCount;
extern std::atomic<size_t> swipeLongTargetPairCount;
extern std::atomic<size_t> swipeByteActiveLaneRowCount;
extern std::atomic<size_t> swipeByteTotalLaneRowCount;
extern std::atomic<size_t> swipeRerunPairCount;

// member alignments in SWIPE lanes, per thread
struct SwipeBatch {
    SwipeAligner aligner;
    EvalueComputation &evaluer;
    bool isQueryInitialized;
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
    size_t lanePairCount;
    size_t laneCapacity;
    size_t scalarPairCount;

    SwipeBatch(const Parameters &par, BaseMatrix *subMat, EvalueComputation &evaluer, size_t maxSeqLen)
        : aligner(subMat, maxSeqLen, par.compBiasCorrection, par.compBiasCorrectionScale, par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid()),
          evaluer(evaluer), isQueryInitialized(false), lanePairCount(0), laneCapacity(0), scalarPairCount(0) {}

    ~SwipeBatch() {
        swipeLanePairCount += lanePairCount;
        swipeLaneCapacity += laneCapacity;
        swipeScalarPairCount += scalarPairCount;
    }

    void addLanePairs(size_t count) {
        size_t lanes = VECSIZE_INT * 2;
        lanePairCount += count;
        laneCapacity += (count + lanes - 1) / lanes * lanes;
    }

    void startRep() {
        isQueryInitialized = false;
    }

    void clear() {
        gapped.clear();
        sequences.clear();
        offsets.clear();
        lengths.clear();
    }

    // linfinity keeps the aligner over the fetches, but not the buffers of its largest batch
    void release() {
        std::vector<unsigned char>().swap(sequences);
        std::vector<size_t>().swap(offsets);
        std::vector<const unsigned char *>().swap(targets);
        std::vector<int>().swap(lengths);
        std::vector<int>().swap(laneLengths);
        std::vector<int>().swap(diagonals);
        std::vector<int>().swap(columnEnds);
        std::vector<int>().swap(minScores);
        std::vector<s_align>().swap(aligns);
        std::vector<std::string>().swap(backtraces);
        std::vector<size_t>().swap(entries);
        std::vector<size_t>().swap(gapped);
        aligner.release();
    }

    size_t add(const Sequence &sequence) {
        offsets.push_back(sequences.size());
        sequences.insert(sequences.end(), sequence.numSequence, sequence.numSequence + sequence.L);
        lengths.push_back(sequence.L);
        return offsets.size() - 1;
    }

    // start in 16 bit lanes, most members saturate 8 bit
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
        if (isQueryInitialized == false) {
            aligner.initQuery(&query);
            isQueryInitialized = true;
        }
    }
};

// targets for the SWIPE extensions, per thread
struct SwipeTargets {
    SwipeExtender extender;
    EvalueComputation &evaluer;
    DBKeyType repKey;
    bool isQueryInitialized;
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
    std::vector<char> seqData;
    std::vector<const char *> targetSequences;
    std::vector<int> seedQueries;
    std::vector<int> seedTargets;
    std::vector<s_align> backAligns;
    std::vector<std::string> backBacktraces;
    size_t targetPairs;
    bool isForMembers;

    SwipeTargets(const Parameters &par, BaseMatrix *subMat, EvalueComputation &evaluer, size_t maxSeqLen, bool isForMembers = false)
        : extender(subMat, maxSeqLen, par.compBiasCorrection, par.compBiasCorrectionScale, par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid()),
          evaluer(evaluer), repKey(0), isQueryInitialized(false), queryId(0), targetPairs(0), isForMembers(isForMembers) {}

    ~SwipeTargets() {
        if (isForMembers) {
            swipeMemberExtensionCount += targetPairs;
            swipeMemberActiveLaneRowCount += extender.extensionActiveLaneRowCount;
            swipeMemberTotalLaneRowCount += extender.extensionTotalLaneRowCount;
            swipeByteActiveLaneRowCount += extender.byteExtensionActiveLaneRowCount;
            swipeByteTotalLaneRowCount += extender.byteExtensionTotalLaneRowCount;
            swipeRerunPairCount += extender.rerunExtensionCount;
            return;
        }
        swipeTargetPairCount += targetPairs;
        swipeTargetActiveLaneRowCount += extender.extensionActiveLaneRowCount;
        swipeTargetTotalLaneRowCount += extender.extensionTotalLaneRowCount;
        swipeLongTargetPairCount += extender.longExtensionCount;
        swipeByteActiveLaneRowCount += extender.byteExtensionActiveLaneRowCount;
        swipeByteTotalLaneRowCount += extender.byteExtensionTotalLaneRowCount;
        swipeRerunPairCount += extender.rerunExtensionCount;
    }

    void addTarget(Sequence &query, Sequence &target, DBKeyType targetKey, int seedQuery, int seedTarget) {
        if (isQueryInitialized == false) {
            queryId = extender.addQuery(&query);
            repKeys.push_back(repKey);
            repSequences.push_back(query.getSeqData());
            repLengths.push_back(query.L);
            isQueryInitialized = true;
        }
        queryIds.push_back(queryId);
        offsets.push_back(sequences.size());
        sequences.insert(sequences.end(), target.numSequence, target.numSequence + target.L);
        lengths.push_back(target.L);
        targetKeys.push_back(targetKey);
        seqData.insert(seqData.end(), target.getSeqData(), target.getSeqData() + target.L);
        seedQueries.push_back(seedQuery);
        seedTargets.push_back(seedTarget);
    }

    void clearTargets() {
        extender.clearQueries();
        isQueryInitialized = false;
        repKeys.clear();
        repSequences.clear();
        repLengths.clear();
        queryIds.clear();
        sequences.clear();
        offsets.clear();
        lengths.clear();
        targetKeys.clear();
        seqData.clear();
        seedQueries.clear();
        seedTargets.clear();
    }

    // linfinity keeps the extender over the fetches, but not the buffers of its largest batch
    void release() {
        clearTargets();
        std::vector<DBKeyType>().swap(repKeys);
        std::vector<const char *>().swap(repSequences);
        std::vector<int>().swap(repLengths);
        std::vector<size_t>().swap(queryIds);
        std::vector<unsigned char>().swap(sequences);
        std::vector<size_t>().swap(offsets);
        std::vector<const unsigned char *>().swap(targets);
        std::vector<int>().swap(lengths);
        std::vector<s_align>().swap(aligns);
        std::vector<std::string>().swap(backtraces);
        std::vector<DBKeyType>().swap(targetKeys);
        std::vector<char>().swap(seqData);
        std::vector<const char *>().swap(targetSequences);
        std::vector<int>().swap(seedQueries);
        std::vector<int>().swap(seedTargets);
        std::vector<s_align>().swap(backAligns);
        std::vector<std::string>().swap(backBacktraces);
        extender.release();
    }

    void startRep(DBKeyType key) {
        repKey = key;
        isQueryInitialized = false;
    }
};

struct MemberCheck {
    size_t targetId;
    short startDiagonal;
    std::vector<DBKeyType> memberKeys;
    // linfinity: fetched entries of memberKeys
    std::vector<const char *> memberData;
    std::vector<size_t> order;
    std::vector<size_t> memberIds;
    std::vector<int> lengths;
    std::vector<std::string> records;
    // filled by prepareMemberChecks unless a part of a pair brings those of the whole pair
    std::vector<unsigned char> isLaneMember;
    size_t nextMember;
    bool isPassing;
};

struct MemberChecks {
    std::vector<MemberCheck> checks;
    size_t checkCount;
    std::vector<size_t> gappedMemberChecks;
    std::vector<size_t> gappedMemberBatchIndices;
    SwipeBatch *swipeMembers;
    bool isMemberCheckDeferred;
    SwipeTargets *swipeTargets;
    std::vector<size_t> extendedMemberChecks;
    std::vector<size_t> gappedMembers;
    std::vector<ClusterPair> extendedPairs;

    MemberChecks(SwipeBatch *swipeMembers, bool isMemberCheckDeferred, SwipeTargets *swipeTargets = NULL) : checkCount(0), swipeMembers(swipeMembers), isMemberCheckDeferred(isMemberCheckDeferred), swipeTargets(swipeTargets) {}

    MemberCheck &addCheck(size_t targetId, short startDiagonal) {
        if (checkCount == checks.size()) {
            checks.emplace_back();
        }
        MemberCheck &check = checks[checkCount++];
        check.targetId = targetId;
        check.startDiagonal = startDiagonal;
        check.memberKeys.clear();
        check.memberData.clear();
        check.isLaneMember.clear();
        return check;
    }

    // addCheck keeps the buffers of the largest checks, linfinity frees them after each fetch
    void release() {
        std::vector<MemberCheck>().swap(checks);
        checkCount = 0;
        std::vector<size_t>().swap(gappedMemberChecks);
        std::vector<size_t>().swap(gappedMemberBatchIndices);
        std::vector<size_t>().swap(extendedMemberChecks);
        std::vector<size_t>().swap(gappedMembers);
        std::vector<ClusterPair>().swap(extendedPairs);
    }
};

Matcher::result_t createSwipeResult(const Parameters &par, EvalueComputation &evaluer, DBKeyType key, int rawScore, const s_align &alignment,
                                    int queryLength, int targetLength, const std::string &backtrace);
void alignMembersSwipe(const Parameters &par, Matcher &matcher, bool &matcherInitialized, MemberChecks &memberChecks, Sequence &query, Sequence &element,
                       DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer);
bool canCoverMember(const Parameters &par, int queryLength, int memberLength);
bool isSwipeLaneMember(size_t position, size_t memberCount);
void prepareMemberChecks(const Parameters &par, Sequence &query, DBReader<DBKeyType> *cluSeqDbr, bool withRecords, MemberChecks &memberChecks, size_t checkBegin, size_t checkEnd);
bool alignNextMembers(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                      DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer, MemberChecks &memberChecks,
                      size_t checkBegin, size_t checkEnd, size_t batchSize);
void alignClusterMembers(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                         DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer,
                         MemberChecks &memberChecks);
bool alignTarget(const Parameters &par, BlockAligner &blockAligner, bool &blockAlignerInitialized, Sequence &query, Sequence &target,
                 float scorePerColThreshold, unsigned short &diagonal, const ClusterAssignment *assignedCluster, size_t targetId, Matcher::result_t *result,
                 SwipeTargets *swipeTargets);
bool doSwipeExtensionsPass(const Parameters &par, SwipeTargets &swipeTargets, size_t i);
int align2clustLinfinity(Parameters &par);

// skip the targets assigned since they were queued
template <typename Assigned>
void alignTargetsSwipe(const Parameters &par, SwipeTargets &swipeTargets, const Assigned &isAssigned, std::vector<ClusterPair> &pairs) {
    int32_t xDrop = (MIN_SIZE * par.gapExtend.values.aminoacid() + par.gapOpen.values.aminoacid());
    // twice one past the longest gap within the X-drop
    int bandSide = 2 * (MIN_SIZE + 2);
    size_t count = 0;
    for (size_t i = 0; i < swipeTargets.targetKeys.size(); i++) {
        if (isAssigned(swipeTargets.repKeys[swipeTargets.queryIds[i]]) || isAssigned(swipeTargets.targetKeys[i])) {
            continue;
        }
        swipeTargets.queryIds[count] = swipeTargets.queryIds[i];
        swipeTargets.offsets[count] = swipeTargets.offsets[i];
        swipeTargets.lengths[count] = swipeTargets.lengths[i];
        swipeTargets.seedQueries[count] = swipeTargets.seedQueries[i];
        swipeTargets.seedTargets[count] = swipeTargets.seedTargets[i];
        swipeTargets.targetKeys[count] = swipeTargets.targetKeys[i];
        count++;
    }
    if (count == 0) {
        return;
    }
    swipeTargets.targets.resize(count);
    swipeTargets.targetSequences.resize(count);
    swipeTargets.aligns.resize(count);
    swipeTargets.backtraces.resize(count);
    for (size_t i = 0; i < count; i++) {
        swipeTargets.targets[i] = swipeTargets.sequences.data() + swipeTargets.offsets[i];
        swipeTargets.targetSequences[i] = swipeTargets.seqData.data() + swipeTargets.offsets[i];
    }
    swipeTargets.extender.extendBanded(count, swipeTargets.queryIds.data(), swipeTargets.targets.data(), swipeTargets.lengths.data(), swipeTargets.seedQueries.data(), swipeTargets.seedTargets.data(), false, xDrop,
                               bandSide, swipeTargets.aligns.data(), swipeTargets.backtraces.data());
    // keep the targets whose forward extension covers enough
    size_t passed = 0;
    for (size_t i = 0; i < count; i++) {
        const s_align &forward = swipeTargets.aligns[i];
        if (forward.score1 == 0 || Util::hasCoverage(par.covThr, par.covMode, SmithWaterman::computeCov(0, forward.qEndPos1, swipeTargets.repLengths[swipeTargets.queryIds[i]]),
                                                     SmithWaterman::computeCov(0, forward.dbEndPos1, swipeTargets.lengths[i])) == false) {
            continue;
        }
        swipeTargets.queryIds[passed] = swipeTargets.queryIds[i];
        swipeTargets.targets[passed] = swipeTargets.targets[i];
        swipeTargets.lengths[passed] = swipeTargets.lengths[i];
        swipeTargets.seedQueries[passed] = swipeTargets.seedQueries[i];
        swipeTargets.seedTargets[passed] = swipeTargets.seedTargets[i];
        swipeTargets.targetKeys[passed] = swipeTargets.targetKeys[i];
        swipeTargets.targetSequences[passed] = swipeTargets.targetSequences[i];
        swipeTargets.aligns[passed] = swipeTargets.aligns[i];
        swipeTargets.backtraces[passed].swap(swipeTargets.backtraces[i]);
        passed++;
    }
    swipeTargets.targetPairs += count;
    count = passed;
    swipeTargets.backAligns.resize(count);
    swipeTargets.backBacktraces.resize(count);
    swipeTargets.extender.extendBanded(count, swipeTargets.queryIds.data(), swipeTargets.targets.data(), swipeTargets.lengths.data(), swipeTargets.seedQueries.data(), swipeTargets.seedTargets.data(), true, xDrop,
                               bandSide, swipeTargets.backAligns.data(), swipeTargets.backBacktraces.data());
    for (size_t i = 0; i < count; i++) {
        if (swipeTargets.backAligns[i].score1 > 0 && doSwipeExtensionsPass(par, swipeTargets, i)) {
            ClusterPair pair = { swipeTargets.repKeys[swipeTargets.queryIds[i]], swipeTargets.targetKeys[i] };
            pairs.push_back(pair);
        }
    }
}

// the gapped members of all reps share the extension lanes
template <typename MapRep>
void alignMemberChecksSwipe(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, Sequence &query, Sequence &element, DBReader<DBKeyType> *cluSeqDbr,
                            unsigned int swMode, unsigned int threadIdx, MemberChecks &memberChecks, const std::vector<size_t> &repMemberCheckOffsets,
                            const std::vector<DBKeyType> &repKeys, const MapRep &mapRep) {
    SwipeTargets &swipeTargets = *memberChecks.swipeTargets;
    SwipeBatch &swipeMembers = *memberChecks.swipeMembers;
    bool matcherInitialized = false;
    for (size_t batchSize = 1; ; batchSize *= 2) {
        bool hasOpenCheck = false;
        memberChecks.extendedMemberChecks.clear();
        memberChecks.gappedMembers.clear();
        for (size_t rep = 0; rep < repKeys.size(); rep++) {
            bool repHasOpenCheck = (batchSize == 1);
            for (size_t g = repMemberCheckOffsets[rep]; g < repMemberCheckOffsets[rep + 1]; g++) {
                repHasOpenCheck = repHasOpenCheck || (memberChecks.checks[g].isPassing && memberChecks.checks[g].nextMember < memberChecks.checks[g].memberKeys.size());
            }
            if (repHasOpenCheck == false) {
                continue;
            }
            mapRep(query, blockAligner, repKeys[rep]);
            if (batchSize == 1) {
                prepareMemberChecks(par, query, cluSeqDbr, false, memberChecks, repMemberCheckOffsets[rep], repMemberCheckOffsets[rep + 1]);
            }
            swipeTargets.startRep(repKeys[rep]);
            hasOpenCheck = alignNextMembers(par, blockAligner, matcher, matcherInitialized, query, element, cluSeqDbr, swMode, threadIdx, false, NULL, memberChecks,
                                            repMemberCheckOffsets[rep], repMemberCheckOffsets[rep + 1], batchSize) || hasOpenCheck;
        }
        if (hasOpenCheck == false) {
            break;
        }
        memberChecks.extendedPairs.clear();
        alignTargetsSwipe(par, swipeTargets, [](DBKeyType) { return false; }, memberChecks.extendedPairs);
        swipeTargets.clearTargets();
        for (size_t i = 0; i < memberChecks.extendedPairs.size(); i++) {
            memberChecks.gappedMembers[memberChecks.extendedPairs[i].memberKey] = SIZE_MAX;
        }
        size_t gapped = 0;
        for (size_t rep = 0; rep < repKeys.size(); rep++) {
            swipeMembers.clear();
            memberChecks.gappedMemberChecks.clear();
            for (; gapped < memberChecks.extendedMemberChecks.size() && memberChecks.extendedMemberChecks[gapped] < repMemberCheckOffsets[rep + 1]; gapped++) {
                size_t i = memberChecks.gappedMembers[gapped];
                if (i == SIZE_MAX) {
                    continue;
                }
                size_t g = memberChecks.extendedMemberChecks[gapped];
                const MemberCheck &check = memberChecks.checks[g];
                DBKeyType elementKey = check.memberKeys[i];
                size_t elementId = check.memberIds[i];
                element.mapSequence(elementId, elementKey, check.memberData.empty() ? cluSeqDbr->getData(elementId, threadIdx) : check.memberData[i], check.lengths[i]);
                swipeMembers.add(element);
                swipeMembers.gapped.push_back(i);
                memberChecks.gappedMemberChecks.push_back(g);
            }
            if (swipeMembers.gapped.empty()) {
                continue;
            }
            mapRep(query, blockAligner, repKeys[rep]);
            swipeMembers.startRep();
            matcherInitialized = false;
            alignMembersSwipe(par, matcher, matcherInitialized, memberChecks, query, element, cluSeqDbr, swMode, threadIdx, false, NULL);
        }
    }
}

#endif
