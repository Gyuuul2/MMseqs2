#include "Util.h"
#include "Parameters.h"
#include "Matcher.h"
#include "Debug.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "QueryMatcher.h"
#include "FastSort.h"
#include "BlockAligner.h"
#include "Alignment.h"
#include <atomic>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

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
    size_t size;

    static bool compareBySizeAndId(const PrefInfo &first, const PrefInfo &second){
        if(first.size > second.size)
            return true;
        if(second.size > first.size)
            return false;
        if(first.id < second.id)
            return true;
        if(second.id < first.id)
            return false;
        return false;
    }
};

// Lightweight entry stored in the set-cover ready queue. Instead of carrying the
// member id vector (as ClusterResult does) it only references the members, which
// are kept in a single shared pool (setCoverMemberPool), by offset and count.
struct SetCoverCandidate {
    DBLocalId representativeId;
    size_t memberCount;
    size_t memberOffset;
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
// Reorders out-of-order worker results back to sequenceIdx order for the consumer,
// indexed by sequenceIdx % reorderCapacity. The next-in-order slot is always free,
// so producers never deadlock.
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

// Compact setCoverMemberPool when over half is dead. Only offsets change, so heap order holds.
static const size_t ALIGN2CLUST_MIN_COMPACTION_DEAD_MEMBERS = 16 * 1024 * 1024 / sizeof(DBLocalId);

static void compactSetCoverMemberPool() {
    const size_t deadMemberCount = setCoverMemberPool.size() - setCoverLiveMemberCount;
    if (setCoverMemberPool.empty() ||
        deadMemberCount < ALIGN2CLUST_MIN_COMPACTION_DEAD_MEMBERS ||
        deadMemberCount * 2 <= setCoverMemberPool.size()) {
        return;
    }
    std::vector<DBLocalId> compactedPool;
    compactedPool.reserve(setCoverLiveMemberCount);
    for (SetCoverCandidate &candidate : setCoverCandidates) {
        const size_t newOffset = compactedPool.size();
        compactedPool.insert(compactedPool.end(),
                             setCoverMemberPool.begin() + candidate.memberOffset,
                             setCoverMemberPool.begin() + candidate.memberOffset + candidate.memberCount);
        candidate.memberOffset = newOffset;
    }
    setCoverMemberPool.swap(compactedPool);
}

// memory-derived capacity holds 100Ms of pending member vectors and buys no throughput over this
static const size_t ALIGN2CLUST_DEFAULT_REORDER_LIMIT = 5 * 1000 * 1000;

// Env override for the reorder-buffer size; 0 (unset/invalid) means the default cap above.
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

// Size the reorder buffer from free memory: subtract the resident per-sequence arrays,
// keep 10% headroom, divide by the worst-case result size. Capped by
// MMSEQS_ALIGN2CLUST_REORDER_LIMIT and by the number of results produced.
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
    const size_t userCap = getReorderBufferLimitFromEnv();  // 0 == use the default cap
    capacity = std::min(capacity, userCap != 0 ? userCap : ALIGN2CLUST_DEFAULT_REORDER_LIMIT);
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
            return idx < currentProcessPosition + reorderCapacity;
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

// mirrors Clustering::writeData, but reads members through a local-id permutation
static void writeClustering(DBWriter *dbWriter, DBReader<DBKeyType> *seqDbr,
                            const ClusterAssignment *assignedCluster, const DBLocalId *memberOrder, size_t dbSize) {
    std::string resultString;
    resultString.reserve(1024 * 1024);
    char buffer[32];
    DBKeyType previousRepresentativeKey = DB_KEY_INVALID;

    for (size_t i = 0; i < dbSize; i++) {
        const DBLocalId memberId = memberOrder[i];
        const DBKeyType currentRepresentativeKey = seqDbr->getDbKey(loadAssignedCluster(assignedCluster, memberId));

        if (previousRepresentativeKey != currentRepresentativeKey) {
            if (previousRepresentativeKey != DB_KEY_INVALID) {
                dbWriter->writeData(resultString.c_str(), resultString.length(), previousRepresentativeKey);
            }
            resultString.clear();
            char *outPos = Itoa::u64toa_sse2(static_cast<uint64_t>(currentRepresentativeKey), buffer);
            resultString.append(buffer, (outPos - buffer - 1));
            resultString.push_back('\n');
        }

        const DBKeyType memberKey = seqDbr->getDbKey(memberId);
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

// every getId here must hit: a miss returns DB_ENTRY_NOT_FOUND and the caller would index out of bounds
static size_t requireId(size_t id, const char *dbName, DBKeyType key) {
    if (id == DB_ENTRY_NOT_FOUND) {
        Debug(Debug::ERROR) << dbName << " has no entry for key " << key << "\n";
        EXIT(EXIT_FAILURE);
    }
    return id;
}

// length-only gate over a cluster's members: index reads only, no sequence data touched
static bool clusterMembersCanBeCovered(DBReader<DBKeyType> *cluSeqDbr, const Parameters &par,
                                       char *cluData, DBKeyType targetKey, int queryLen) {
    char buffer[1024];
    char *scan = cluData;
    while (*scan != '\0') {
        Util::parseKey(scan, buffer);
        const DBKeyType memberKey = Util::fast_atoi<DBKeyType>(buffer);
        if (memberKey != targetKey) {
            const size_t memberId = requireId(cluSeqDbr->getId(memberKey), "Filter sequence DB", memberKey);
            if (Util::canBeCovered(par.covThr, par.covMode, queryLen, cluSeqDbr->getSeqLen(memberId)) == false) {
                return false;
            }
        }
        scan = Util::skipLine(scan);
    }
    return true;
}

static void (*clusterThreadFunc)(ClusterAssignment*) = nullptr;

void clusterThreadFuncSetcover(ClusterAssignment* assignedCluster) {
    while (true) {
        size_t drained = 0;
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
            drained++;
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

        // Compaction only touches consumer-private structures (setCoverCandidates,
        // setCoverMemberPool, setCoverLiveMemberCount), never shared state, so release
        // the mutex during the copy so worker threads can keep pushing results.
        lock.unlock();
        // one wake per drained round, issued while the lock is free
        if (drained > 0) {
            reorderSpaceCondition.notify_all();
        }
        compactSetCoverMemberPool();
        lock.lock();

        if (allCalculationsDone &&
            reorderBufferedCount == 0 &&
            setCoverCandidates.empty()) {
            break;
        }
    }
}

void clusterThreadFuncGreedy(ClusterAssignment* assignedCluster) {
    // consumer-private scratch, so both are reused instead of allocated once per result
    std::vector<ClusterResult> drainedResults;
    std::vector<DBLocalId> validMemberIds;
    while (true) {
        size_t drained = 0;
        bool lastRound = false;
        drainedResults.clear();
        {
            std::unique_lock<std::mutex> lock(clusterMutex);

            clusterCondition.wait(lock, [] {
                return reorderFilled[currentProcessPosition % reorderCapacity] != 0 ||
                       allCalculationsDone;
            });

            lastRound = (allCalculationsDone && reorderBufferedCount == 0);

            // the mutex guards the ring, so hold it only for the moves out of it
            while (reorderFilled[currentProcessPosition % reorderCapacity] != 0) {
                const size_t slot = currentProcessPosition % reorderCapacity;
                drainedResults.push_back(std::move(reorderSlots[slot]));
                reorderFilled[slot] = 0;
                reorderBufferedCount--;
                currentProcessPosition++;
                drained++;
            }
        }

        // one wake per drained round, issued while the lock is free
        if (drained > 0) {
            reorderSpaceCondition.notify_all();
        }

        // only this thread writes assignedCluster while the producers run, so no mutex is needed
        for (ClusterResult &result : drainedResults) {
            if (loadAssignedCluster(assignedCluster, result.representativeId) != DB_LOCAL_ID_INVALID) {
                continue;
            }

            validMemberIds.clear();
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

        if (lastRound) {
            break;
        }
    }
}

// keyed access over a file-ordered layout, where the kernel's guess readaheads a window per miss
static void adviseRandom(DBReader<DBKeyType> *reader) {
    if (reader == NULL) {
        return;
    }
    for (size_t fileIdx = 0; fileIdx < reader->getDataFileCnt(); fileIdx++) {
        Util::madviseLogged(reader->getDataForFile(fileIdx), reader->getDataSizeForFile(fileIdx),
                            POSIX_MADV_RANDOM, "align2clust");
    }
}

// the prefilter db is read once per representative, so its cache only competes with the bodies
static void dropCache(DBReader<DBKeyType> *reader) {
#ifdef HAVE_POSIX_FADVISE
    if (reader == NULL) {
        return;
    }
    std::vector<std::string> names = reader->getDataFileNames();
    for (size_t fileIdx = 0; fileIdx < names.size(); fileIdx++) {
        int fd = ::open(names[fileIdx].c_str(), O_RDONLY);
        if (fd < 0) {
            continue;
        }
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
#endif
}

// one fault per scattered entry, so ask for the whole prefilter list at once and let them overlap
static void prefetchBody(DBReader<DBKeyType> *reader, size_t id) {
    static const size_t pageSize = Util::getPageSize();
    char *data = reader->getDataUncompressed(id);
    const size_t len = reader->getEntryLen(id);
    const uintptr_t start = reinterpret_cast<uintptr_t>(data) & ~(pageSize - 1);
    posix_madvise(reinterpret_cast<void *>(start),
                  len + (reinterpret_cast<uintptr_t>(data) - start), POSIX_MADV_WILLNEED);
}

// each drop walks the cached extent, so bound how many the run pays for
static const size_t ALIGN2CLUST_PREF_DROPS = 256;

int doAlign2clust(Parameters &par, DBWriter &resultWriter, DBReader<DBKeyType> &alnDbr, DBWriter *alnWriter) {
    DBReader<DBKeyType> *seqDbr = new DBReader<DBKeyType>(
        par.db1.c_str(), par.db1Index.c_str(), par.threads, 
        DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX
    );
    // SORT_BY_LENGTH costs id2local plus local2id; neither mode needs them once lengthOrder exists
    seqDbr->open(DBReader<DBKeyType>::NOSORT);
 
    DBReader<DBKeyType> *cluDbr = nullptr;
    DBReader<DBKeyType> *cluSeqDbr = nullptr;
    if (par.filterCluDBFile.empty()== false && par.filterSeqDBFile.empty()== false) {
        std::string cluIndex = par.filterCluDBFile + ".index";
        cluDbr = new DBReader<DBKeyType>(
            par.filterCluDBFile.c_str(), cluIndex.c_str(), par.threads, 
            DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX
        );
        // NOSORT: only ever read by key lookup, so the LINEAR_ACCCESS id mapping is dead weight
        cluDbr->open(DBReader<DBKeyType>::NOSORT);

        std::string cluSeqIndex = par.filterSeqDBFile + ".index";
        cluSeqDbr = new DBReader<DBKeyType>(
            par.filterSeqDBFile.c_str(), cluSeqIndex.c_str(), par.threads, 
            DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX
        );
            
        cluSeqDbr->open(DBReader<DBKeyType>::NOSORT);
    } else if (par.filterCluDBFile.empty() != par.filterSeqDBFile.empty()) {
        Debug(Debug::ERROR)<< "Error: Both filterCluDBFile and filterSeqDBFile should be provided together.\n";
        EXIT(EXIT_FAILURE);
    }


    const size_t dbSize = seqDbr->getSize();

    BaseMatrix *subMat = new SubstitutionMatrix(
        par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, 0.0
    );
    SubstitutionMatrix::FastMatrix fastMatrix = SubstitutionMatrix::createAsciiSubMat(*subMat);

    std::string libraryString = (par.covMode == Parameters::COV_MODE_BIDIRECTIONAL)
                                    ? getCovSeqidQscPercMinDiag()
                                    : getCovSeqidQscPercMinDiagTargetCov();
                                    
    float scorePerColThreshold = parsePrecisionLib(libraryString, par.seqIdThr, par.covThr, 0.99);
    Debug(Debug::INFO) << "Score per column threshold for filtering: " << scorePerColThreshold << "\n";
    
    EvalueComputation evaluer(seqDbr->getAminoAcidDBSize(), subMat);
    int32_t xDrop = (MIN_SIZE * par.gapExtend.values.aminoacid() + par.gapOpen.values.aminoacid());
    
    ClusterAssignment *assignedCluster = new(std::nothrow) ClusterAssignment[dbSize];
    Util::checkAllocation(assignedCluster, "Can not allocate assignedCluster memory in Align2Clust");
    // one thread would first-touch the whole array, which on a multi socket box also lands every page
    // on a single node for the rest of the run
#pragma omp parallel for schedule(static)
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
        delete[] fastMatrix.matrix;
        delete[] fastMatrix.matrixData;
        delete subMat;
        seqDbr->close();
        delete seqDbr;
        return EXIT_FAILURE;
    }

    // Ring size = out-of-order window, sized from the memory budget (OOM-aware) and
    // capped by the result count. sequenceIdx runs over [0, endRange); every index
    // publishes exactly one result.
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
        // NOSORT drops the reader's length order, so materialise it from a fused (length, id) array
        std::pair<unsigned int, DBLocalId> *sortForLength =
            new(std::nothrow) std::pair<unsigned int, DBLocalId>[dbSize];
        Util::checkAllocation(sortForLength, "Can not allocate sortForLength memory in Align2Clust");
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dbSize; i++) {
            sortForLength[i] = std::make_pair(static_cast<unsigned int>(seqDbr->getEntryLen(i)),
                                              static_cast<DBLocalId>(i));
        }
        // byte-for-byte DBReader::comparePairBySeqLength: entry length descending, local id ascending
        SORT_PARALLEL(sortForLength, sortForLength + dbSize,
                      [](const std::pair<unsigned int, DBLocalId> &first,
                         const std::pair<unsigned int, DBLocalId> &second) {
                          if (first.first != second.first) {
                              return first.first > second.first;
                          }
                          return first.second < second.second;
                      });
        lengthOrder = new(std::nothrow) DBLocalId[dbSize];
        Util::checkAllocation(lengthOrder, "Can not allocate lengthOrder memory in Align2Clust");
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dbSize; i++) {
            lengthOrder[i] = sortForLength[i].second;
        }
        delete[] sortForLength;
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
                const size_t alnId = requireId(alnDbr.getId(clusterId), "Alignment DB", clusterId);
                const char *data = alnDbr.getData(alnId, thread_idx);
                const size_t dataSize = alnDbr.getEntryLen(alnId);
                prefRepSizePair[i].id = i;
                prefRepSizePair[i].size = (*data == '\0') ? 1 : Util::countLines(data, dataSize);
            }
        }
        SORT_PARALLEL(prefRepSizePair, prefRepSizePair + dbSize, PrefInfo::compareBySizeAndId);
    }

    timer.reset();

    size_t endRange = (mode == Parameters::SET_COVER) ? dbSize : alnDbr.getSize();
    // lengthOrder is indexed by i, so replace the bounds check getDbKey(i) used to do
    if (lengthOrder != nullptr && endRange > dbSize) {
        Debug(Debug::ERROR) << "Alignment DB has " << endRange << " entries but the sequence DB has " << dbSize << "\n";
        EXIT(EXIT_FAILURE);
    }
    unsigned int swMode = Alignment::initSWMode(par.alignmentMode, par.covThr, par.seqIdThr);
    // while the dbs still fit, readahead is free and every hint below only costs what it saves
    const bool cacheStarved = seqDbr->getDataSize() + alnDbr.getDataSize()
                              > Util::computeMemory(par.splitMemoryLimit);
    const size_t prefDropInterval = std::max<size_t>(1, endRange / ALIGN2CLUST_PREF_DROPS);
    if (cacheStarved) {
        adviseRandom(seqDbr);
        adviseRandom(&alnDbr);
        adviseRandom(cluSeqDbr);
        adviseRandom(cluDbr);
    }
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
        Matcher matcher(Parameters::DBTYPE_AMINO_ACIDS, db_maxseqlen, subMat, &evaluer, 
                       par.compBiasCorrection, par.compBiasCorrectionScale, 
                       par.gapOpen.values.aminoacid(), par.gapExtend.values.aminoacid(), 
                       0.0, par.zdrop);
        Sequence query(db_maxseqlen, Parameters::DBTYPE_AMINO_ACIDS, subMat, 0, false, par.compBiasCorrection);
        Sequence target(db_maxseqlen, Parameters::DBTYPE_AMINO_ACIDS, subMat, 0, false, par.compBiasCorrection);
        Sequence element(db_maxseqlen, Parameters::DBTYPE_AMINO_ACIDS, subMat, 0, false, par.compBiasCorrection);
        BlockAligner blockAligner(Parameters::DBTYPE_AMINO_ACIDS, db_maxseqlen, subMat, &fastMatrix, 
                                 &evaluer, par.compBiasCorrection, par.compBiasCorrectionScale, 
                                 -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid());
        std::vector<std::pair<size_t, unsigned short>> targetsWithDiagonal;
        targetsWithDiagonal.reserve(1000);

        const bool includeAlignFiles = (alnWriter != nullptr);
        std::string queryCopy;
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
            // dropping consumed prefilter pages hands the cache to the bodies
            if (cacheStarved && i > 0 && (i % prefDropInterval) == 0) {
                dropCache(&alnDbr);
            }
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
                representativeId = lengthOrder[i];
                queryKey = seqDbr->getDbKey(representativeId);
                clusterResult.prefSize = 0;                         // greedy has no currentPrefSize gate
            }
            clusterResult.representativeId = representativeId;

            // Representative already assigned to another cluster: this cluster is discarded
            // by the cluster thread anyway, so skip parsing and aligning it entirely. prefSize
            // is already set (precomputed for set-cover), so the currentPrefSize gate stays
            // correct.
            if (loadAssignedCluster(assignedCluster, representativeId) != DB_LOCAL_ID_INVALID) {
                pushClusterResult(std::move(clusterResult));
                continue;
            }

            const size_t alignmentId = requireId(alnDbr.getId(queryKey), "Alignment DB", queryKey);
            char *alignmentData = alnDbr.getData(alignmentId, threadIdx);
            size_t queryId = representativeId;
            // index-only, so it is known without faulting the body in; Sequence::mapSequence sets L to it
            const int queryLength = static_cast<int>(seqDbr->getSeqLen(queryId));
            // the body is loaded lazily below, on the first target that actually reaches alignment
            const char *querySequence = nullptr;

            size_t prefSize = 0;
            while (*alignmentData != '\0') {
                hit_t hit = QueryMatcher::parsePrefilterHit(alignmentData);
                const size_t targetId = requireId(seqDbr->getId(hit.seqId), "Sequence DB", hit.seqId);
                if (mode == Parameters::SET_COVER) {
                    targetsWithDiagonal.push_back(std::make_pair(targetId, hit.diagonal));
                } else {
                    if (loadAssignedCluster(assignedCluster, targetId) == DB_LOCAL_ID_INVALID) {
                        targetsWithDiagonal.push_back(std::make_pair(targetId, hit.diagonal));
                    }
                }
                alignmentData = Util::skipLine(alignmentData);
                prefSize++;
            }
            clusterResult.prefSize = prefSize;   // exact parsed count for the aligned path

            if (cacheStarved) {
                // the loop below reads a body only past the assigned and coverage gates, so asking for
                // the rest of the list evicts the pages that are actually wanted
                bool anyTargetSurvives = false;
                for (size_t targetIdx = 0; targetIdx < targetsWithDiagonal.size(); targetIdx++) {
                    const size_t targetId = targetsWithDiagonal[targetIdx].first;
                    if (loadAssignedCluster(assignedCluster, targetId) != DB_LOCAL_ID_INVALID) {
                        continue;
                    }
                    if (Util::canBeCovered(par.covThr, par.covMode, queryLength,
                                           seqDbr->getSeqLen(targetId)) == false) {
                        continue;
                    }
                    prefetchBody(seqDbr, targetId);
                    anyTargetSurvives = true;
                }
                if (anyTargetSurvives) {
                    prefetchBody(seqDbr, queryId);
                }
            }

            for (size_t targetIdx = 0; targetIdx < targetsWithDiagonal.size(); targetIdx++) {
                // Representative assigned meanwhile: the cluster thread discards clusters
                // whose representative is assigned, so this result (partial or empty) is
                // never used; stop aligning the rest. Safe in set-cover too: prefSize is
                // already fully counted, so this is just the k-th-target form of the
                // (k=0) rep-skip above.
                if (loadAssignedCluster(assignedCluster, representativeId) != DB_LOCAL_ID_INVALID) {
                    break;
                }

                const size_t targetId = targetsWithDiagonal[targetIdx].first;
                const unsigned short diagonal = targetsWithDiagonal[targetIdx].second;
                const DBKeyType targetKey = seqDbr->getDbKey(targetId);

                const bool isIdentity = (queryKey == targetKey);
                if (isIdentity) {
                    clusterResult.memberIds.push_back(queryId);
                    if (includeAlignFiles) {
                        // Identity hit: no alignment needed. mmseqs forces coverage/seqId to
                        // 1.0 for identity (see Alignment.cpp), so emit a full-length self
                        // record directly instead of running Smith-Waterman.
                        std::string backtrace = par.addBacktrace ? std::string(queryLength, 'M') : std::string();
                        Matcher::result_t selfResult(queryKey, queryLength, 1.0f, 1.0f, 1.0f, 0.0,
                            queryLength, 0, queryLength - 1, queryLength, 0, queryLength - 1, queryLength, backtrace);
                        appendAlignmentResult(alnResultBuffer, alnLineBuffer.data(), selfResult, par.addBacktrace);
                    }
                    continue;
                }

                // Skip the (expensive) alignment if the target was assigned meanwhile.
                // Safe in set-cover too: an assigned target is monotonic, so it would be
                // dropped by the cluster thread's re-evaluation anyway.
                if (loadAssignedCluster(assignedCluster, targetId) != DB_LOCAL_ID_INVALID) {
                    continue;
                }

                // the length alone decides coverage, so test it before faulting in the sequence
                const size_t targetLength = seqDbr->getSeqLen(targetId);
                if (Util::canBeCovered(par.covThr, par.covMode, queryLength, targetLength) == false) {
                    continue;
                }

                // first target to survive filtering: now the query body is worth its random read
                if (querySequence == nullptr) {
                    // getData may hand back a shared per-thread buffer, so copy before the first target read
                    queryCopy.assign(seqDbr->getData(queryId, threadIdx), queryLength);
                    querySequence = queryCopy.c_str();
                    query.mapSequence(queryId, queryKey, querySequence, queryLength);
                    blockAligner.initQuery(&query);
                    matcher.initQuery(&query);
                }

                char *targetSequence = seqDbr->getData(targetId, threadIdx);
                target.mapSequence(targetId, targetKey, targetSequence, targetLength);

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
                    seqId = Util::computeSeqId(par.seqIdMode, identicalCount, queryLength, target.L, ungappedAlignment.alnLen);
                }
                
                bool hasSeqId = seqId >= (par.seqIdThr - std::numeric_limits<float>::epsilon());
                if (loadAssignedCluster(assignedCluster, targetId) != DB_LOCAL_ID_INVALID) continue;

                if (hasAlnLen && hasCoverage && hasSeqId && hasEvalue) {
                    if (loadAssignedCluster(assignedCluster, targetId) != DB_LOCAL_ID_INVALID) continue;
                    if (par.filterCluDBFile.empty()== false && par.filterSeqDBFile.empty()== false){
                        // check all the member from filtering file
                        const size_t cluId = requireId(cluDbr->getId(targetKey), "Filter cluster DB", targetKey);
                        char *cluData = cluDbr->getData(cluId, threadIdx);
                        const size_t cluDataSize = cluDbr->getEntryLen(cluId);
                        size_t numClu = Util::countLines(cluData, cluDataSize);
                        bool allpass = true;
                        char buffer[1024];
                        if (includeAlignFiles) {
                            pendingMemberAln.clear();
                        }
                        if (numClu > 1) {
                            allpass = clusterMembersCanBeCovered(cluSeqDbr, par, cluData, targetKey, queryLength);
                        }
                        if (allpass && numClu > 1) { // if not singleton
                            while (*cluData != '\0') {
                                Util::parseKey(cluData, buffer);

                                const DBKeyType elementKey = Util::fast_atoi<DBKeyType>(buffer);
                                if (elementKey == targetKey) {
                                    cluData = Util::skipLine(cluData);
                                    continue;
                                }
                                const size_t elementId = requireId(cluSeqDbr->getId(elementKey), "Filter sequence DB", elementKey);
                                char *elementSequence = cluSeqDbr->getData(elementId, threadIdx);
                                size_t elementLength = cluSeqDbr->getSeqLen(elementId);
                                short elementDiagonal = diagonal;

                                // 1. ungapped alignment
                                element.mapSequence(elementId, elementKey, elementSequence, elementLength);
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
                                float elementSeqId = Util::computeSeqId(par.seqIdMode, elementIdenticalCount, queryLength, elementLength, elementUngappedAlignment.alnLen);
                                bool elementHasSeqId = elementSeqId >= (par.seqIdThr - std::numeric_limits<float>::epsilon());
                                
                                if (!(elementHasAlnLen && elementHasCoverage && elementHasSeqId && elementHasEvalue)) {
                                    // 3. gapped alignment
                                    Matcher::result_t res_element = matcher.getSWResult(&element, static_cast<int>(elementDiagonal), false, par.covMode, par.covThr, par.evalThr,
                                                                        swMode, par.seqIdMode, false);
                                    if (Alignment::checkCriteria(res_element, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr) == false) {
                                        allpass = false;
                                        break;
                                    }
                                    // stage member alignment (flushed only if allpass holds)
                                    if (includeAlignFiles) {
                                        appendAlignmentResult(pendingMemberAln, alnLineBuffer.data(), res_element, par.addBacktrace);
                                    }
                                } else if (includeAlignFiles) {
                                    // member passed ungapped: gap-free (all 'M') record
                                    std::string elementBacktrace = par.addBacktrace ? std::string(elementUngappedAlignment.alnLen, 'M') : std::string();
                                    Matcher::result_t elementResult(elementKey, elementUngappedAlignment.score, elementUngappedAlignment.qcov,
                                        elementUngappedAlignment.tcov, elementSeqId, elementUngappedAlignment.eval, elementUngappedAlignment.alnLen,
                                        elementUngappedAlignment.qStart, elementUngappedAlignment.qEnd, queryLength,
                                        elementUngappedAlignment.tStart, elementUngappedAlignment.tEnd, elementLength, elementBacktrace);
                                    appendAlignmentResult(pendingMemberAln, alnLineBuffer.data(), elementResult, par.addBacktrace);
                                }
                                cluData = Util::skipLine(cluData);
                            }
                        }
                        if (allpass == false) {
                            continue;
                        }
                    }
                    if (includeAlignFiles) {
                        std::string backtrace = par.addBacktrace ? std::string(ungappedAlignment.alnLen, 'M') : std::string();
                        Matcher::result_t ungappedResult(targetKey, ungappedAlignment.score, ungappedAlignment.qcov,
                            ungappedAlignment.tcov, seqId, ungappedAlignment.eval, ungappedAlignment.alnLen,
                            ungappedAlignment.qStart, ungappedAlignment.qEnd, queryLength,
                            ungappedAlignment.tStart, ungappedAlignment.tEnd, targetLength, backtrace);
                        appendAlignmentResult(alnResultBuffer, alnLineBuffer.data(), ungappedResult, par.addBacktrace);
                        // flush staged member alignments (empty unless filter gate ran)
                        alnResultBuffer += pendingMemberAln;
                    }
                    clusterResult.memberIds.push_back(targetId);
                    continue;
                }

                float currentScorePerCol = static_cast<float>(ungappedAlignment.score) / static_cast<float>(ungappedAlignment.diagonalLen);
                if (currentScorePerCol < scorePerColThreshold) {
                    continue;
                }
                
                int alignmentLength = ungappedAlignment.alnLen;
                int queryStartPos = ungappedAlignment.qStart;
                int targetStartPos = ungappedAlignment.tStart;
                int newQueryStartPos = queryStartPos;
                int newTargetStartPos = targetStartPos;
                
                if (queryStartPos == -1 || targetStartPos == -1 || alignmentLength < 3) {
                    continue;
                }

                if (loadAssignedCluster(assignedCluster, targetId) != DB_LOCAL_ID_INVALID) continue;

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
                
                if (foundConsecutiveMatchSeed) {
                    std::string gappedBacktrace;

                    s_align gappedAlignment = blockAligner.bandedalign(&target, newQueryStartPos, newTargetStartPos,
                                                                       gappedBacktrace, xDrop, par.covThr, par.covMode);
                    // bandedalign signals failure/no-coverage with evalue < 0 and an empty backtrace;
                    // skip before computeSeqId, which would divide by alnLen == 0 in SEQ_ID_ALN_LEN mode.
                    if (gappedAlignment.evalue < 0 || gappedBacktrace.empty()) {
                        continue;
                    }
                    unsigned int gappedAlnLength = gappedBacktrace.size();
                    double gappedSeqId = Util::computeSeqId(par.seqIdMode, gappedAlignment.identicalAACnt,
                                                           queryLength, targetLength, gappedAlnLength);
                    Matcher::result_t result = Matcher::result_t(
                        targetKey, gappedAlignment.score1, gappedAlignment.qCov, gappedAlignment.tCov, 
                        gappedSeqId, gappedAlignment.evalue, gappedAlnLength,
                        gappedAlignment.qStartPos1, gappedAlignment.qEndPos1, queryLength,
                        gappedAlignment.dbStartPos1, gappedAlignment.dbEndPos1, targetLength, gappedBacktrace
                    );
                    if (Alignment::checkCriteria(result, isIdentity, par.evalThr, par.seqIdThr, 
                                                par.alnLenThr, par.covMode, par.covThr)) {
                        if (loadAssignedCluster(assignedCluster, targetId) != DB_LOCAL_ID_INVALID) continue;
                        if (par.filterCluDBFile.empty()== false && par.filterSeqDBFile.empty()== false){
                            // check all the member from filtering file
                            const size_t cluId = requireId(cluDbr->getId(targetKey), "Filter cluster DB", targetKey);
                            char *cluData = cluDbr->getData(cluId, threadIdx);
                            const size_t cluDataSize = cluDbr->getEntryLen(cluId);
                            size_t numClu = Util::countLines(cluData, cluDataSize);
                            bool allpass = true;
                            char buffer[1024];
                            if (includeAlignFiles) {
                                pendingMemberAln.clear();
                            }
                            if (numClu > 1) {
                                allpass = clusterMembersCanBeCovered(cluSeqDbr, par, cluData, targetKey, queryLength);
                            }
                            if (allpass && numClu > 1) { // if not singleton
                                while (*cluData != '\0') {
                                    Util::parseKey(cluData, buffer);
                                    const DBKeyType elementKey = Util::fast_atoi<DBKeyType>(buffer);
                                    if (elementKey == targetKey) {
                                        cluData = Util::skipLine(cluData);
                                        continue;
                                    }
                                    const size_t elementId = requireId(cluSeqDbr->getId(elementKey), "Filter sequence DB", elementKey);
                                    char *elementSequence = cluSeqDbr->getData(elementId, threadIdx);
                                    size_t elementLength = cluSeqDbr->getSeqLen(elementId);
                                    short elementDiagonal = 0;

                                    // 1. ungapped alignment
                                    element.mapSequence(elementId, elementKey, elementSequence, elementLength);
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
                                    float elementSeqId = Util::computeSeqId(par.seqIdMode, elementIdenticalCount, queryLength, elementLength, elementUngappedAlignment.alnLen);
                                    bool elementHasSeqId = elementSeqId >= (par.seqIdThr - std::numeric_limits<float>::epsilon());
                                    
                                    if (!(elementHasAlnLen && elementHasCoverage && elementHasSeqId && elementHasEvalue)) {
                                        // 3. gapped alignment
                                        Matcher::result_t res_element = matcher.getSWResult(&element, static_cast<int>(elementDiagonal), false, par.covMode, par.covThr, par.evalThr,
                                                                            swMode, par.seqIdMode, false);
                                        if (Alignment::checkCriteria(res_element, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode, par.covThr) == false) {
                                            allpass = false;
                                            break;
                                        }
                                        // stage member alignment (flushed only if allpass holds)
                                        if (includeAlignFiles) {
                                            appendAlignmentResult(pendingMemberAln, alnLineBuffer.data(), res_element, par.addBacktrace);
                                        }
                                    } else if (includeAlignFiles) {
                                        // member passed ungapped: gap-free (all 'M') record
                                        std::string elementBacktrace = par.addBacktrace ? std::string(elementUngappedAlignment.alnLen, 'M') : std::string();
                                        Matcher::result_t elementResult(elementKey, elementUngappedAlignment.score, elementUngappedAlignment.qcov,
                                            elementUngappedAlignment.tcov, elementSeqId, elementUngappedAlignment.eval, elementUngappedAlignment.alnLen,
                                            elementUngappedAlignment.qStart, elementUngappedAlignment.qEnd, queryLength,
                                            elementUngappedAlignment.tStart, elementUngappedAlignment.tEnd, elementLength, elementBacktrace);
                                        appendAlignmentResult(pendingMemberAln, alnLineBuffer.data(), elementResult, par.addBacktrace);
                                    }
                                    cluData = Util::skipLine(cluData);
                                }
                            }
                            if (allpass == false) {
                                continue;
                            }
                        }
                        if (includeAlignFiles) {
                            appendAlignmentResult(alnResultBuffer, alnLineBuffer.data(), result, par.addBacktrace);
                            // flush staged member alignments (empty unless filter gate ran)
                            alnResultBuffer += pendingMemberAln;
                        }
                        clusterResult.memberIds.push_back(targetId);
                    }
                }
            }

            if (includeAlignFiles) {
                alnWriter->writeData(alnResultBuffer.c_str(), alnResultBuffer.length(), queryKey, threadIdx);
            }
            pushClusterResult(std::move(clusterResult));
        }
    }

    // nothing past the producer loop reads the visit order, so drop it before memberOrder is sized
    if (lengthOrder != nullptr) {
        delete[] lengthOrder;
        lengthOrder = nullptr;
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

    // ring and set-cover pool are dead once the consumer joins; free them before memberOrder is sized
    std::vector<ClusterResult>().swap(reorderSlots);
    std::vector<unsigned char>().swap(reorderFilled);
    std::vector<SetCoverCandidate>().swap(setCoverCandidates);
    std::vector<DBLocalId>().swap(setCoverMemberPool);

    // the visit order and the alignment db are read only inside the producer loop, so the output phase
    // does not have to compete with 16 byte per sequence and a whole reader index for the page cache
    if (prefRepSizePair != nullptr) {
        delete[] prefRepSizePair;
        prefRepSizePair = nullptr;
    }
    alnDbr.close();

    for (size_t i = 0; i < dbSize; ++i) {
        if (loadAssignedCluster(assignedCluster, i) == DB_LOCAL_ID_INVALID) {
            storeAssignedCluster(assignedCluster, i, i);
        }
    }

    // group members by representative through a local-id permutation: 8 byte per sequence
    // instead of the 16 byte (representative key, member key) pair array
    DBLocalId *memberOrder = new(std::nothrow) DBLocalId[dbSize];
    Util::checkAllocation(memberOrder, "Can not allocate memberOrder memory in Align2Clust");
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dbSize; i++) {
        memberOrder[i] = static_cast<DBLocalId>(i);
    }

    // comparator touches only the two arrays, so no getDbKey call per comparison
    SORT_PARALLEL(memberOrder, memberOrder + dbSize, [assignedCluster](DBLocalId first, DBLocalId second) {
        const DBLocalId firstRep = loadAssignedCluster(assignedCluster, first);
        const DBLocalId secondRep = loadAssignedCluster(assignedCluster, second);
        if (firstRep != secondRep) {
            return firstRep < secondRep;
        }
        return first < second;
    });

    size_t clusterCount = 0;
    DBLocalId previousRep = DB_LOCAL_ID_INVALID;
    for (size_t i = 0; i < dbSize; i++) {
        const DBLocalId rep = loadAssignedCluster(assignedCluster, memberOrder[i]);
        clusterCount += (rep != previousRep);
        previousRep = rep;
    }

    Debug(Debug::INFO) << "Size of the alignment database: " << dbSize << "\n";
    Debug(Debug::INFO) << "Number of clusters: " << clusterCount << "\n";

    writeClustering(&resultWriter, seqDbr, assignedCluster, memberOrder, dbSize);

    delete[] memberOrder;
    delete[] assignedCluster;
    delete[] fastMatrix.matrix;
    delete[] fastMatrix.matrixData;
    delete subMat;
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
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);
    
    Timer timer;
    timer.reset();
    
    DBReader<DBKeyType> alnDbr(par.db2.c_str(), par.db2Index.c_str(), par.threads,
                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA);
    // read only by key, and LINEAR_ACCCESS would advise SEQUENTIAL on a db this pass reads out of order
    alnDbr.open(DBReader<DBKeyType>::NOSORT);
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

    return status;
}
