#include "Align2clust.h"
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
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>

#ifdef OPENMP
#include <omp.h>
#endif

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

static void compactSetCoverMemberPool(std::vector<SetCoverCandidate> &candidates, std::vector<DBLocalId> &memberPool, size_t liveMemberCount) {
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

// takes the largest sets with more than maxSetSize members whose members are all unassigned; a set that lost members goes back
template <typename T, typename IsAssigned, typename AssignMember>
static void assignSetCoverCandidates(std::vector<SetCoverCandidate> &candidates, std::vector<T> &memberPool, size_t &liveMemberCount, size_t maxSetSize,
                                     IsAssigned isAssigned, AssignMember assignMember) {
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
        if (validCount != candidate.memberCount) {
            candidate.memberCount = validCount;
            liveMemberCount += validCount;
            candidates.push_back(candidate);
            std::push_heap(candidates.begin(), candidates.end(), SetCoverComparator());
            continue;
        }
        for (size_t i = 0; i < candidate.memberCount; i++) {
            assignMember(candidate.representativeId, members[i]);
        }
    }
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

AlignmentScoring::AlignmentScoring(const Parameters &par, size_t dbResidues)
    : subMat(new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, 0.0)), fastMatrix(SubstitutionMatrix::createAsciiSubMat(*subMat)),
      scorePerColThreshold(parsePrecisionLib((par.covMode == Parameters::COV_MODE_BIDIRECTIONAL) ? getCovSeqidQscPercMinDiag() : getCovSeqidQscPercMinDiagTargetCov(),
                                             par.seqIdThr, par.covThr, 0.98)),
      evaluer(dbResidues, subMat, par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid()) {}

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

// member gapped alignments in SWIPE lanes, the lane capacity they took and the scalar ones
std::atomic<size_t> swipeLanePairCount(0);
std::atomic<size_t> swipeLaneCapacity(0);
std::atomic<size_t> swipeScalarPairCount(0);

// member extensions in SWIPE lanes, their lane rows and all lane rows
std::atomic<size_t> swipeMemberExtensionCount(0);
std::atomic<size_t> swipeMemberActiveLaneRowCount(0);
std::atomic<size_t> swipeMemberTotalLaneRowCount(0);

// target gapped alignments in SWIPE lanes, their lane rows and all lane rows, and the extensions longer than the trace of the last lane rows
std::atomic<size_t> swipeTargetPairCount(0);
std::atomic<size_t> swipeTargetActiveLaneRowCount(0);
std::atomic<size_t> swipeTargetTotalLaneRowCount(0);
std::atomic<size_t> swipeLongTargetPairCount(0);
// the lane rows of the member and target extensions in 8 bit lanes that extended a target, all of them, and the extensions whose 8 bit lanes saturated
std::atomic<size_t> swipeByteActiveLaneRowCount(0);
std::atomic<size_t> swipeByteTotalLaneRowCount(0);
std::atomic<size_t> swipeRerunPairCount(0);

static void printSwipeLanes() {
    Debug(Debug::INFO) << "SWIPE member gapped alignments: " << swipeLanePairCount.load() << " in lanes ("
                       << swipeLanePairCount.load() * 100 / std::max<size_t>(1, swipeLaneCapacity.load()) << "% active lane rows), "
                       << swipeScalarPairCount.load() << " scalar\n";
    if (swipeMemberExtensionCount.load() > 0) {
        Debug(Debug::INFO) << "SWIPE member extensions: " << swipeMemberExtensionCount.load() << " in lanes ("
                           << swipeMemberActiveLaneRowCount.load() * 100 / std::max<size_t>(1, swipeMemberTotalLaneRowCount.load()) << "% active lane rows)\n";
    }
    if (swipeTargetPairCount.load() + swipeLongTargetPairCount.load() > 0) {
        Debug(Debug::INFO) << "SWIPE target gapped alignments: " << swipeTargetPairCount.load() << " in lanes ("
                           << swipeTargetActiveLaneRowCount.load() * 100 / std::max<size_t>(1, swipeTargetTotalLaneRowCount.load()) << "% active lane rows), "
                           << swipeLongTargetPairCount.load() << " extensions longer than the lane trace\n";
    }
    if (swipeMemberTotalLaneRowCount.load() + swipeTargetTotalLaneRowCount.load() > 0) {
        Debug(Debug::INFO) << "SWIPE extensions in 8 bit lanes: " << swipeByteTotalLaneRowCount.load() * 100 / (swipeMemberTotalLaneRowCount.load() + swipeTargetTotalLaneRowCount.load())
                           << "% of the lane rows (" << swipeByteActiveLaneRowCount.load() * 100 / std::max<size_t>(1, swipeByteTotalLaneRowCount.load()) << "% active), "
                           << swipeRerunPairCount.load() << " again in 16 bit lanes\n";
    }
}

// the result of a SWIPE alignment with start, end, identities and backtrace
Matcher::result_t createSwipeResult(const Parameters &par, EvalueComputation &evaluer, DBKeyType key, int rawScore, const s_align &alignment,
                                    int queryLength, int targetLength, const std::string &backtrace) {
    const float seqId = Util::computeSeqId(par.seqIdMode, alignment.identicalAACnt, queryLength, targetLength, backtrace.size());
    return Matcher::result_t(key, static_cast<int>(evaluer.computeBitScore(rawScore) + 0.5),
                             SmithWaterman::computeCov(alignment.qStartPos1, alignment.qEndPos1, queryLength),
                             SmithWaterman::computeCov(alignment.dbStartPos1, alignment.dbEndPos1, targetLength), seqId,
                             evaluer.computeEvalue(rawScore, queryLength), backtrace.size(), alignment.qStartPos1, alignment.qEndPos1, queryLength,
                             alignment.dbStartPos1, alignment.dbEndPos1, targetLength, backtrace);
}

// whether a forward score and end can still meet the e-value and coverage of the clustering criteria
static bool canSwipeForwardPass(const Parameters &par, EvalueComputation &evaluer, const s_align &alignment, int queryLength, int targetLength) {
    if (alignment.dbEndPos1 < 0 || alignment.qEndPos1 < 0) {
        return false;
    }
    const float qCov = SmithWaterman::computeCov(0, alignment.qEndPos1, queryLength);
    const float tCov = SmithWaterman::computeCov(0, alignment.dbEndPos1, targetLength);
    return Util::hasCoverage(par.covThr, par.covMode, qCov, tCov) && evaluer.computeEvalue(alignment.score1, queryLength) <= par.evalThr;
}

// whether the forward and backward extension of target i of alignTargetsSwipe, joined at their seed cell, meet the clustering criteria
bool doSwipeExtensionsPass(const Parameters &par, SwipeTargets &swipeTargets, size_t i) {
    const s_align &forward = swipeTargets.aligns[i];
    const s_align &backward = swipeTargets.backAligns[i];
    // the seed cell is in both extensions
    const int rawScore = static_cast<int>(forward.score1) + static_cast<int>(backward.score1)
                         - swipeTargets.extender.getCellScore(swipeTargets.queryIds[i], swipeTargets.seedQueries[i], swipeTargets.targets[i][swipeTargets.seedTargets[i]]);
    std::string &gappedBacktrace = swipeTargets.backBacktraces[i];
    gappedBacktrace.pop_back();
    gappedBacktrace += swipeTargets.backtraces[i];
    s_align joined = forward;
    joined.qStartPos1 = backward.qStartPos1;
    joined.dbStartPos1 = backward.dbStartPos1;
    const char *querySequence = swipeTargets.repSequences[swipeTargets.queryIds[i]];
    const char *targetSequence = swipeTargets.targetSequences[i];
    joined.identicalAACnt = 0;
    for (size_t b = 0, q = joined.qStartPos1, t = joined.dbStartPos1; b < gappedBacktrace.size(); b++) {
        const char op = gappedBacktrace[b];
        joined.identicalAACnt += (op == 'M' && (querySequence[q] & static_cast<unsigned char>(~0x20)) == (targetSequence[t] & static_cast<unsigned char>(~0x20)));
        q += (op != 'D');
        t += (op != 'I');
    }
    Matcher::result_t result = createSwipeResult(par, swipeTargets.evaluer, swipeTargets.targetKeys[i], rawScore, joined, swipeTargets.repLengths[swipeTargets.queryIds[i]], swipeTargets.lengths[i],
                                                 gappedBacktrace);
    return Alignment::checkCriteria(result, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr);
}

// the cell after the first 3 matches in a row of an ungapped alignment, which seeds its gapped alignment
static bool findMatchSeed(const char *querySequence, const char *targetSequence, const BlockAligner::UngappedAln_res &alignment, int &seedQuery, int &seedTarget) {
    if (alignment.qStart == -1 || alignment.tStart == -1) {
        return false;
    }
    for (int blockIdx = 0; blockIdx <= alignment.alnLen - 3; ++blockIdx) {
        int queryPos = alignment.qStart + blockIdx;
        int targetPos = alignment.tStart + blockIdx;
        if (querySequence[queryPos] == targetSequence[targetPos] &&
            querySequence[queryPos + 1] == targetSequence[targetPos + 1] &&
            querySequence[queryPos + 2] == targetSequence[targetPos + 2]) {
            seedQuery = queryPos + 1;
            seedTarget = targetPos + 1;
            return true;
        }
    }
    return false;
}

// the members of a check whose current members number this many go through the SWIPE lanes of alignMembersSwipe, on every ISA
static const size_t SWIPE_MIN_MEMBERS_PER_CHECK = 4;

// the gapped member alignments of a round; checks with SWIPE_MIN_MEMBERS_PER_CHECK current members share the lanes, the others go scalar; a failing member fails its check
void alignMembersSwipe(const Parameters &par, Matcher &matcher, bool &matcherInitialized, MemberChecks &memberChecks, Sequence &query, Sequence &element,
                       DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer) {
    SwipeBatch &swipeMembers = *memberChecks.swipeMembers;
    if (matcherInitialized == false) {
        matcher.initQuery(&query);
        matcherInitialized = true;
    }
    std::vector<size_t> &gappedMemberBatchIndices = memberChecks.gappedMemberBatchIndices;
    gappedMemberBatchIndices.assign(swipeMembers.gapped.size(), SIZE_MAX);
    swipeMembers.entries.clear();
    for (size_t g = 0; g < swipeMembers.gapped.size(); g++) {
        if (memberChecks.checks[memberChecks.gappedMemberChecks[g]].currentMemberCount >= SWIPE_MIN_MEMBERS_PER_CHECK) {
            gappedMemberBatchIndices[g] = swipeMembers.entries.size();
            swipeMembers.entries.push_back(g);
        }
    }
    const size_t laneCount = swipeMembers.entries.size();
    swipeMembers.scalarPairCount += swipeMembers.gapped.size() - laneCount;
    if (laneCount > 0) {
        swipeMembers.addLanePairs(laneCount);
        swipeMembers.select(laneCount);
        swipeMembers.laneLengths.resize(laneCount);
        for (size_t l = 0; l < laneCount; l++) {
            swipeMembers.laneLengths[l] = swipeMembers.lengths[swipeMembers.entries[l]];
        }
        swipeMembers.setQuery(query);
        swipeMembers.aligner.alignForward(laneCount, swipeMembers.targets.data(), swipeMembers.laneLengths.data(), swipeMembers.aligns.data());
        for (size_t l = 0; l < laneCount; l++) {
            const s_align &alignment = swipeMembers.aligns[l];
            if (alignment.word < 2 && canSwipeForwardPass(par, swipeMembers.evaluer, alignment, query.L, swipeMembers.laneLengths[l]) == false) {
                memberChecks.checks[memberChecks.gappedMemberChecks[swipeMembers.entries[l]]].isPassing = false;
            }
        }
        swipeMembers.backtraces.resize(laneCount);
        swipeMembers.columnEnds.resize(laneCount);
        swipeMembers.minScores.resize(laneCount);
        for (size_t l = 0; l < laneCount; l++) {
            swipeMembers.diagonals[l] = swipeMembers.aligns[l].dbEndPos1 - swipeMembers.aligns[l].qEndPos1;
            swipeMembers.columnEnds[l] = swipeMembers.aligns[l].dbEndPos1 + 1;
            swipeMembers.minScores[l] = swipeMembers.aligns[l].score1;
        }
        swipeMembers.aligner.alignBanded(laneCount, swipeMembers.targets.data(), swipeMembers.laneLengths.data(), swipeMembers.diagonals.data(),
                                         swipeMembers.aligns.data(), swipeMembers.backtraces.data(), swipeMembers.columnEnds.data(), swipeMembers.minScores.data());
    }
    for (size_t g = 0; g < swipeMembers.gapped.size(); g++) {
        MemberCheck &check = memberChecks.checks[memberChecks.gappedMemberChecks[g]];
        if (check.isPassing == false) {
            continue;
        }
        const size_t i = swipeMembers.gapped[g];
        const DBKeyType elementKey = check.memberKeys[i];
        const size_t elementId = cluSeqDbr->getId(elementKey);
        element.mapSequence(elementId, elementKey, check.memberData.empty() ? cluSeqDbr->getData(elementId, threadIdx) : check.memberData[i], cluSeqDbr->getSeqLen(elementId));
        const size_t l = gappedMemberBatchIndices[g];
        Matcher::result_t res_element;
        if (l == SIZE_MAX || swipeMembers.aligns[l].word >= 2) {
            res_element = matcher.getSWResult(&element, 0, false, par.covMode, par.covThr, par.evalThr, swMode, par.seqIdMode, false);
        } else {
            res_element = createSwipeResult(par, swipeMembers.evaluer, elementKey, swipeMembers.minScores[l], swipeMembers.aligns[l], query.L, element.L, swipeMembers.backtraces[l]);
        }
        if (Alignment::checkCriteria(res_element, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr) == false) {
            check.isPassing = false;
            continue;
        }
        if (withRecords) {
            size_t len = Matcher::resultToBuffer(alnLineBuffer, res_element, par.addBacktrace);
            check.records[i].assign(alnLineBuffer, len);
        }
    }
}

// false if a query of queryLength cannot cover a member of memberLength
bool canCoverMember(const Parameters &par, int queryLength, int memberLength) {
    int minDiag, maxDiag;
    return Util::canBeCovered(par.covThr, par.covMode, queryLength, memberLength)
           && (par.diagonalMode != Parameters::DIAGONAL_MODE_UNGAPPED || canBeCoveredOnDiagonals(par, queryLength, memberLength, minDiag, maxDiag));
}

// the order of the members of the checks, the most different lengths first; the checks with a member that cannot be covered fail
void prepareMemberChecks(const Parameters &par, Sequence &query, DBReader<DBKeyType> *cluSeqDbr, bool withRecords, MemberChecks &memberChecks, size_t checkBegin, size_t checkEnd) {
    const int queryLength = query.L;
    for (size_t g = checkBegin; g < checkEnd; g++) {
        MemberCheck &check = memberChecks.checks[g];
        check.nextMember = 0;
        check.isPassing = true;
        check.order.resize(check.memberKeys.size());
        check.lengths.resize(check.memberKeys.size());
        for (size_t i = 0; i < check.memberKeys.size(); i++) {
            check.order[i] = i;
            check.lengths[i] = static_cast<int>(cluSeqDbr->getSeqLen(cluSeqDbr->getId(check.memberKeys[i])));
            if (canCoverMember(par, query.L, check.lengths[i]) == false) {
                check.isPassing = false;
                break;
            }
        }
        if (check.isPassing == false) {
            continue;
        }
        const std::vector<int> &memberLengths = check.lengths;
        std::stable_sort(check.order.begin(), check.order.end(), [&memberLengths, queryLength](size_t first, size_t second) {
            return std::abs(memberLengths[first] - queryLength) > std::abs(memberLengths[second] - queryLength);
        });
        if (withRecords) {
            check.records.assign(check.memberKeys.size(), std::string());
        }
    }
}

// the next batchSize members of the checks against the query; false if no check had any left
bool alignNextMembers(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                      DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer, MemberChecks &memberChecks,
                      size_t checkBegin, size_t checkEnd, size_t batchSize) {
    const char *querySequence = query.getSeqData();
    SwipeBatch *swipeMembers = memberChecks.swipeMembers;
    bool hasOpenCheck = false;
    for (size_t g = checkBegin; g < checkEnd; g++) {
        MemberCheck &check = memberChecks.checks[g];
        if (check.isPassing == false || check.nextMember == check.memberKeys.size()) {
            continue;
        }
        hasOpenCheck = true;
        const size_t batchEnd = std::min(check.memberKeys.size(), check.nextMember + batchSize);
        check.currentMemberCount = batchEnd - check.nextMember;
        for (size_t o = check.nextMember; o < batchEnd && check.isPassing; o++) {
            const size_t i = check.order[o];
            const DBKeyType elementKey = check.memberKeys[i];
            const size_t elementId = cluSeqDbr->getId(elementKey);
            const char *elementSequence = check.memberData.empty() ? cluSeqDbr->getData(elementId, threadIdx) : check.memberData[i];
            size_t elementLength = cluSeqDbr->getSeqLen(elementId);
            short elementDiagonal = check.startDiagonal;

            // 1. ungapped alignment
            element.mapSequence(elementId, elementKey, elementSequence, elementLength);
            if (Util::canBeCovered(par.covThr, par.covMode, query.L, element.L) == false) {
                check.isPassing = false;
                break;
            }
            if (par.diagonalMode == Parameters::DIAGONAL_MODE_UNGAPPED) {
                int minDiag, maxDiag;
                if (canBeCoveredOnDiagonals(par, query.L, element.L, minDiag, maxDiag) == false) {
                    check.isPassing = false;
                    break;
                }
                elementDiagonal = blockAligner.computeBestDiagonal(&element, minDiag, maxDiag);
            }
            BlockAligner::UngappedAln_res elementUngappedAlignment = blockAligner.ungappedAlign(&element, elementDiagonal);

            // 2. check the criteria
            bool elementHasEvalue = (elementUngappedAlignment.eval <= par.evalThr);
            bool elementHasAlnLen = (elementUngappedAlignment.alnLen >= par.alnLenThr);
            bool elementHasCoverage = Util::hasCoverage(par.covThr, par.covMode, elementUngappedAlignment.qcov, elementUngappedAlignment.tcov);
            int elementIdenticalCount = 0;
            if (elementHasEvalue && elementUngappedAlignment.qStart >= 0 && elementUngappedAlignment.tStart >= 0
                && elementUngappedAlignment.qEnd >= elementUngappedAlignment.qStart) {
                for (int q = elementUngappedAlignment.qStart; q <= elementUngappedAlignment.qEnd; q++) {
                    char queryLetter = querySequence[q] & static_cast<unsigned char>(~0x20);
                    char elementLetter = elementSequence[elementUngappedAlignment.tStart + (q - elementUngappedAlignment.qStart)] & static_cast<unsigned char>(~0x20);
                    elementIdenticalCount += (queryLetter == elementLetter) ? 1 : 0;
                }
            }
            float elementSeqId = Util::computeSeqId(par.seqIdMode, elementIdenticalCount, query.L, elementLength, elementUngappedAlignment.alnLen);
            bool elementHasSeqId = elementSeqId >= (par.seqIdThr - std::numeric_limits<float>::epsilon());

            if (!(elementHasAlnLen && elementHasCoverage && elementHasSeqId && elementHasEvalue)) {
                // 3. gapped alignment, after the round with swipe
                if (memberChecks.swipeTargets != NULL) {
                    int seedQuery, seedTarget;
                    if (findMatchSeed(querySequence, elementSequence, elementUngappedAlignment, seedQuery, seedTarget)) {
                        // the key of a member extension is its position in extendedMemberChecks
                        memberChecks.swipeTargets->addTarget(query, element, memberChecks.extendedMemberChecks.size(), seedQuery, seedTarget);
                    }
                    memberChecks.extendedMemberChecks.push_back(g);
                    memberChecks.gappedMembers.push_back(i);
                    continue;
                }
                if (swipeMembers != NULL) {
                    swipeMembers->add(element);
                    swipeMembers->gapped.push_back(i);
                    memberChecks.gappedMemberChecks.push_back(g);
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
                    check.isPassing = false;
                    break;
                }
                // stage member alignment (flushed only if all members pass)
                if (withRecords) {
                    appendAlignmentResult(check.records[i], alnLineBuffer, res_element, par.addBacktrace);
                }
            } else if (withRecords) {
                // member passed ungapped: gap-free (all 'M') record
                std::string elementBacktrace = par.addBacktrace ? std::string(elementUngappedAlignment.alnLen, 'M') : std::string();
                Matcher::result_t elementResult(elementKey, elementUngappedAlignment.score, elementUngappedAlignment.qcov,
                    elementUngappedAlignment.tcov, elementSeqId, elementUngappedAlignment.eval, elementUngappedAlignment.alnLen,
                    elementUngappedAlignment.qStart, elementUngappedAlignment.qEnd, query.L,
                    elementUngappedAlignment.tStart, elementUngappedAlignment.tEnd, elementLength, elementBacktrace);
                appendAlignmentResult(check.records[i], alnLineBuffer, elementResult, par.addBacktrace);
            }
        }
        check.nextMember = batchEnd;
    }
    return hasOpenCheck;
}

// whether all --filter-cludb-file members of each check meet the criteria, the ones that cannot be covered first and the most different lengths next;
// with SWIPE the checks go in rounds of 1, 2, 4, ... members
void alignClusterMembers(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                         DBReader<DBKeyType> *cluSeqDbr, unsigned int swMode, unsigned int threadIdx, bool withRecords, char *alnLineBuffer,
                         MemberChecks &memberChecks) {
    SwipeBatch *swipeMembers = memberChecks.swipeMembers;
    prepareMemberChecks(par, query, cluSeqDbr, withRecords, memberChecks, 0, memberChecks.checkCount);
    for (size_t batchSize = 1; ; batchSize *= (swipeMembers != NULL) ? 2 : 1) {
        if (swipeMembers != NULL) {
            swipeMembers->clear();
            memberChecks.gappedMemberChecks.clear();
        }
        if (alignNextMembers(par, blockAligner, matcher, matcherInitialized, query, element, cluSeqDbr, swMode, threadIdx, withRecords, alnLineBuffer, memberChecks,
                             0, memberChecks.checkCount, batchSize) == false) {
            break;
        }
        if (swipeMembers != NULL && swipeMembers->gapped.empty() == false) {
            alignMembersSwipe(par, matcher, matcherInitialized, memberChecks, query, element, cluSeqDbr, swMode, threadIdx, withRecords, alnLineBuffer);
        }
    }
}

// the member check of the target of alignTarget: checked now, or queued for the rounds of the rep and passed for now with isMemberCheckDeferred
static bool alignTargetMembers(const Parameters &par, BlockAligner &blockAligner, Matcher &matcher, bool &matcherInitialized, Sequence &query, Sequence &element,
                               DBReader<DBKeyType> *cluDbr, DBReader<DBKeyType> *cluSeqDbr, DBKeyType targetKey, size_t targetId, short startDiagonal,
                               unsigned int swMode, unsigned int threadIdx, std::string *pendingMemberAln, char *alnLineBuffer, MemberChecks &memberChecks) {
    if (memberChecks.isMemberCheckDeferred == false) {
        memberChecks.checkCount = 0;
    }
    MemberCheck &check = memberChecks.addCheck(targetId, startDiagonal);
    getClusterMembers(cluDbr, targetKey, threadIdx, check.memberKeys);
    if (pendingMemberAln != NULL) {
        pendingMemberAln->clear();
    }
    if (memberChecks.isMemberCheckDeferred) {
        return true;
    }
    alignClusterMembers(par, blockAligner, matcher, matcherInitialized, query, element, cluSeqDbr, swMode, threadIdx, pendingMemberAln != NULL, alnLineBuffer, memberChecks);
    if (check.isPassing && pendingMemberAln != NULL) {
        for (size_t r = 0; r < check.records.size(); r++) {
            *pendingMemberAln += check.records[r];
        }
    }
    return check.isPassing;
}

// whether the target meets the clustering criteria against the query of blockAligner; diagonal becomes the one its --filter-cludb-file members start from
bool alignTarget(const Parameters &par, BlockAligner &blockAligner, bool &blockAlignerInitialized, Sequence &query, Sequence &target,
                 float scorePerColThreshold, unsigned short &diagonal, const ClusterAssignment *assignedCluster, size_t targetId, Matcher::result_t *result,
                 SwipeTargets *swipeTargets) {
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
        diagonal = blockAligner.computeBestDiagonal(&target, minDiag, maxDiag);
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

    int newQueryStartPos, newTargetStartPos;
    if (isAssigned() || findMatchSeed(querySequence, targetSequence, ungappedAlignment, newQueryStartPos, newTargetStartPos) == false) {
        return false;
    }
    // the gapped alignment in the SWIPE lanes, with the other queued targets of the rep
    if (swipeTargets != NULL) {
        swipeTargets->addTarget(query, target, target.getDbKey(), newQueryStartPos, newTargetStartPos);
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
        assignSetCoverCandidates(setCoverCandidates, setCoverMemberPool, setCoverLiveMemberCount, allCalculationsDone ? 0 : currentPrefSize,
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
        const bool isSwipeEnabled = (par.align2clustAlignerMode != Parameters::ALIGNER_MODE_FARRAR && par.gapOpen.values.aminoacid() >= par.gapExtend.values.aminoacid());
        SwipeBatch swipeMembers(par, scoring.subMat, scoring.evaluer, db_maxseqlen);
        // the gapped targets and the member checks of a rep wait for its last target and share the SWIPE lanes, unless the alignments are written in target order
        const bool isTargetSwipeEnabled = isSwipeEnabled && alnWriter == nullptr;
        SwipeTargets swipeTargets(par, scoring.subMat, scoring.evaluer, db_maxseqlen);
        SwipeTargets memberTargets(par, scoring.subMat, scoring.evaluer, db_maxseqlen, true);
        MemberChecks memberChecks(isSwipeEnabled ? &swipeMembers : NULL, isTargetSwipeEnabled && cluDbr != NULL, isTargetSwipeEnabled ? &memberTargets : NULL);
        std::vector<ClusterPair> swipePairs;
        std::vector<size_t> repMemberCheckOffsets(2, 0);
        std::vector<DBKeyType> repKeys(1);
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
            swipeMembers.startRep();
            swipeTargets.startRep(queryKey);

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
                                              includeAlignFiles ? &targetResult : NULL, isTargetSwipeEnabled ? &swipeTargets : NULL);
                if (isAccepted && cluDbr != NULL) {
                    isAccepted = alignTargetMembers(par, blockAligner, matcher, matcherInitialized, query, element, cluDbr, cluSeqDbr, targetKey, targetId,
                                                    diagonal, swMode, threadIdx, includeAlignFiles ? &pendingMemberAln : NULL, alnLineBuffer.data(), memberChecks);
                }
                if (isAccepted) {
                    if (includeAlignFiles) {
                        appendAlignmentResult(alnResultBuffer, alnLineBuffer.data(), targetResult, par.addBacktrace);
                        alnResultBuffer += pendingMemberAln;
                    }
                    clusterResult.memberIds.push_back(targetId);
                }
            }
            swipePairs.clear();
            alignTargetsSwipe(par, swipeTargets, [&](DBKeyType key) { return loadAssignedCluster(assignedCluster, seqDbr->getId(key)) != DB_LOCAL_ID_INVALID; }, swipePairs);
            swipeTargets.clearTargets();
            for (size_t p = 0; p < swipePairs.size(); p++) {
                const DBKeyType targetKey = swipePairs[p].memberKey;
                const size_t targetId = seqDbr->getId(targetKey);
                if (cluDbr == NULL || alignTargetMembers(par, blockAligner, matcher, matcherInitialized, query, element, cluDbr, cluSeqDbr, targetKey, targetId,
                                                         0, swMode, threadIdx, NULL, NULL, memberChecks)) {
                    clusterResult.memberIds.push_back(targetId);
                }
            }
            if (memberChecks.isMemberCheckDeferred && memberChecks.checkCount > 0) {
                repMemberCheckOffsets[1] = memberChecks.checkCount;
                repKeys[0] = queryKey;
                alignMemberChecksSwipe(par, blockAligner, matcher, query, element, cluSeqDbr, swMode, threadIdx, memberChecks, repMemberCheckOffsets, repKeys,
                                       [](Sequence &, BlockAligner &, DBKeyType) {});
                for (size_t g = 0; g < memberChecks.checkCount; g++) {
                    if (memberChecks.checks[g].isPassing == false) {
                        clusterResult.memberIds.erase(std::find(clusterResult.memberIds.begin(), clusterResult.memberIds.end(), memberChecks.checks[g].targetId));
                    }
                }
                memberChecks.checkCount = 0;
            }

            if (includeAlignFiles) {
                alnWriter->writeData(alnResultBuffer.c_str(), alnResultBuffer.length(), queryKey, threadIdx);
            }
            pushClusterResult(std::move(clusterResult));
        }
    }
    if (par.align2clustAlignerMode != Parameters::ALIGNER_MODE_FARRAR) {
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

int align2clust(int argc, const char **argv, const Command &command) {
    MMseqsNode::init();
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);
    if (par.filterCluDBFile.empty() != par.filterSeqDBFile.empty()) {
        Debug(Debug::ERROR)<< "Error: Both filterCluDBFile and filterSeqDBFile should be provided together.\n";
        EXIT(EXIT_FAILURE);
    }
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(FileUtil::parseDbType(par.db1.c_str())) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity) {
        if (par.clusteringMode != Parameters::SET_COVER && par.clusteringMode != Parameters::GREEDY && par.clusteringMode != Parameters::GREEDY_MEM) {
            Debug(Debug::ERROR) << "align2clust supports only --cluster-mode " << Parameters::SET_COVER << ", " << Parameters::GREEDY << " and " << Parameters::GREEDY_MEM
                                << " for linfinity databases (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ")\n";
            EXIT(EXIT_FAILURE);
        }
        const int prefDbtype = FileUtil::parseDbType(par.db2.c_str());
        if (Parameters::isEqualDbtype(prefDbtype, Parameters::DBTYPE_PREFILTER_RES) == false
            || (DBReader<DBKeyType>::getExtendedDbtype(prefDbtype) & Parameters::DBTYPE_EXTENDED_LINFINITY) == 0) {
            Debug(Debug::ERROR) << par.db2 << " is not a linfinity prefilter result (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ")\n";
            EXIT(EXIT_FAILURE);
        }
        if (par.compressed) {
            Debug(Debug::WARNING) << "--compressed cannot be combined with linfinity databases (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ")\n";
            Debug(Debug::WARNING) << "We recompute with --compressed 0\n";
            par.compressed = 0;
        }
        if (par.includeAlignFiles) {
            Debug(Debug::WARNING) << "--include-align-files cannot be combined with linfinity databases (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ")\n";
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
