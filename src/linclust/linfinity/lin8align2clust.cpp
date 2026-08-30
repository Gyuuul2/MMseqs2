#include "Lin8Db.h"
#include "Lin8DbReader.h"
#include "Parameters.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"
#include "Timer.h"
#include "NodePlacement.h"
#include "Alignment.h"
#include "Matcher.h"

#include <sstream>
#include <sched.h>
#include "Sequence.h"
#include "SubstitutionMatrix.h"
#include "EvalueComputation.h"
#include "BlockAligner.h"
#include "FastSort.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

static const size_t STREAM_ROWS = 1u << 16;
// enough that a fork and a join are lost in the aligning, few enough that a batch is megabytes
static const size_t BATCH_ROWS = 1u << 18;
static const size_t ARENA_BYTES = 64u << 20;

static const size_t MEMBERS_PER_ALIGN_BATCH = 512;

// one representative's members, cut so a large group is shared rather than owned
struct MemberBatch {
    size_t group;
    size_t from;
    size_t count;
};

// one representative and a batch of its members, so a big group becomes many of them
static void readPipelineShape(const std::string &path, unsigned int &nodes, size_t &repRankBlocks, uint64_t &ranks) {
    FILE *in = fopen(path.c_str(), "r");
    if (in == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << path << ". Run lin8pref first\n";
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
                            << " repRankBlocks, which is not a set of prefilter pairs\n";
        EXIT(EXIT_FAILURE);
    }
}

class RepRankBlockReader {
public:
    RepRankBlockReader(const std::string &prefix, unsigned int prefNodes, size_t repRankBlock, size_t bufferRows,
                uint64_t skipRows)
        : buffer(bufferRows), at(0), filled(0), stopped(false) {
        const std::string path =
            prefix + "." + SSTR(repRankBlock % prefNodes) + "." + SSTR(repRankBlock);
        file = fopen(path.c_str(), "r");
        if (file == NULL) {
            Debug(Debug::ERROR) << "Cannot open " << path << ", which node " << (repRankBlock % prefNodes)
                                << " should have written for repRankBlock " << repRankBlock << "\n";
            EXIT(EXIT_FAILURE);
        }
        // off_t, not long: a repRankBlock is two hundred gigabytes at a trillion sequences
        const off_t skip = static_cast<off_t>(skipRows * PairRecord::DISK_BYTES);
        if (skip > 0 && fseeko(file, skip, SEEK_SET) != 0) {
            Debug(Debug::ERROR) << "Cannot seek " << skip << " byte into " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        refill();
    }

    ~RepRankBlockReader() {
        if (file != NULL) {
            fclose(file);
        }
    }

    bool next(PairRecord &row) {
        if (at >= filled) {
            refill();
            if (filled == 0) {
                return false;
            }
        }
        row = buffer[at++];
        return true;
    }

    // stops at the last change of representative, carrying the rows of the one it stopped inside
    bool fillBatch(std::vector<PairRecord> &into, size_t want, uint64_t until) {
        into.clear();
        into.swap(carry);
        PairRecord row;
        while (stopped == false && into.size() < want) {
            if (next(row) == false || row.rep() >= until) {
                stopped = true;
                break;
            }
            into.push_back(row);
        }
        if (stopped == false && into.empty() == false) {
            const uint64_t last = into.back().rep();
            size_t keep = into.size();
            while (keep > 0 && into[keep - 1].rep() == last) {
                keep--;
            }
            if (keep == 0) {
                // one representative is larger than a batch, so its rows are read to the end of it
                while (true) {
                    if (next(row) == false || row.rep() >= until) {
                        stopped = true;
                        break;
                    }
                    if (row.rep() != last) {
                        carry.push_back(row);
                        break;
                    }
                    into.push_back(row);
                }
            } else {
                carry.assign(into.begin() + keep, into.end());
                into.resize(keep);
            }
        }
        return into.empty() == false;
    }

private:
    RepRankBlockReader(const RepRankBlockReader &);
    RepRankBlockReader &operator=(const RepRankBlockReader &);

    void refill() {
        filled = readRecords(buffer.data(), buffer.size(), file);
        at = 0;
    }

    std::vector<PairRecord> buffer;
    FILE *file;
    size_t at;
    size_t filled;
    bool stopped;
    std::vector<PairRecord> carry;
};

struct Candidates {
    std::vector<uint64_t> members;
    std::vector<int> diagonals;
    std::vector<uint64_t> kept;

    void clear() {
        members.clear();
        diagonals.clear();
        kept.clear();
    }
};

struct AlignWorker {
    AlignWorker(size_t maxLen, SubstitutionMatrix &subMat,
                SubstitutionMatrix::FastMatrix &fastMatrix, EvalueComputation &evaluer,
                const Parameters &par)
        : query(maxLen, Parameters::DBTYPE_AMINO_ACIDS, &subMat, 0, false, par.compBiasCorrection),
          target(maxLen, Parameters::DBTYPE_AMINO_ACIDS, &subMat, 0, false, par.compBiasCorrection),
          aligner(Parameters::DBTYPE_AMINO_ACIDS, maxLen, &subMat, &fastMatrix, &evaluer,
                  par.compBiasCorrection, par.compBiasCorrectionScale,
                  -par.gapOpen.values.aminoacid(), -par.gapExtend.values.aminoacid(),
                  BlockAligner::CLUSTER_MAX_BAND) {}
    Sequence query;
    Sequence target;
    BlockAligner aligner;
};

// one shared counter, so every batch is drawn exactly once and no thread waits on another
static size_t draw(size_t &counter) {
    size_t mine = 0;
#pragma omp atomic capture
    mine = counter++;
    return mine;
}

struct GateCounts {
    GateCounts() : seen(0), rejected(0), rescued(0), kept(0), stale(0) {}
    uint64_t seen;
    uint64_t rejected;
    uint64_t rescued;   // turned away and worth a gapped alignment
    uint64_t kept;      // and the gapped alignment held up
    uint64_t stale;     // read, then found already in a cluster, so the read was wasted
};

static float parsePrecisionLib(const std::string &table, double seqId, double cov,
                                  double precision) {
    std::stringstream in(table);
    std::string line;
    const int hundredths = static_cast<int>((seqId + 0.0001) * 100);
    const float wantSeqId = static_cast<float>(hundredths - hundredths % 5) / 100;
    const float wantCov = static_cast<float>(static_cast<int>((cov + 0.0001) * 10)) / 10;
    while (std::getline(in, line)) {
        const std::vector<std::string> values = Util::split(line, " ");
        if (values.size() < 4) {
            continue;
        }
        if (MathUtil::AreSame((float) strtod(values[0].c_str(), NULL), wantCov)
            && MathUtil::AreSame((float) strtod(values[1].c_str(), NULL), wantSeqId)
            && strtod(values[3].c_str(), NULL) >= precision) {
            return (float) strtod(values[2].c_str(), NULL);
        }
    }
    Debug(Debug::WARNING) << "No calibrated score per column for coverage " << wantCov
                          << " and identity " << wantSeqId << ", so nothing is rescued\n";
    return 0;
}


static bool rescueWithGaps(uint64_t member, uint32_t queryLen, uint32_t targetLen,
                           const char *querySeq, const char *targetSeq,
                           const BlockAligner::UngappedAln_res &hit, const ClusterAssignmentBitmap &assignedCluster,
                           Sequence &target, BlockAligner &aligner, const Parameters &par,
                           float scorePerColThreshold, int xDrop, GateCounts &gate) {
    if (hit.diagonalLen <= 0
        || (float) hit.score / (float) hit.diagonalLen < scorePerColThreshold) {
        return false;
    }
    if (hit.qStart < 0 || hit.tStart < 0 || hit.alnLen < 3 || assignedCluster.isAssigned(member)) {
        return false;
    }
    int queryFrom = -1;
    int targetFrom = -1;
    for (int at = 0; at <= hit.alnLen - 3; at++) {
        const int q = hit.qStart + at;
        const int t = hit.tStart + at;
        if (querySeq[q] == targetSeq[t] && querySeq[q + 1] == targetSeq[t + 1]
            && querySeq[q + 2] == targetSeq[t + 2]) {
            queryFrom = q + 1;
            targetFrom = t + 1;
            break;
        }
    }
    if (queryFrom < 0) {
        return false;
    }
    gate.rescued++;
    std::string backtrace;
    const s_align gapped = aligner.bandedalign(&target, (size_t) queryFrom, (size_t) targetFrom,
                                               backtrace, xDrop, par.covThr, par.covMode);
    if (gapped.evalue < 0 || backtrace.empty()) {
        return false;
    }
    const unsigned int alnLen = static_cast<unsigned int>(backtrace.size());
    const float seqId = Util::computeSeqId(par.seqIdMode, gapped.identicalAACnt, queryLen, targetLen,
                                           alnLen);
    Matcher::result_t result(0, gapped.score1, gapped.qCov, gapped.tCov, seqId, gapped.evalue, alnLen,
                             gapped.qStartPos1, gapped.qEndPos1, queryLen, gapped.dbStartPos1,
                             gapped.dbEndPos1, targetLen, backtrace);
    if (Alignment::checkCriteria(result, false, par.evalThr, par.seqIdThr, par.alnLenThr, par.covMode,
                                 par.covThr) == false) {
        return false;
    }
    gate.kept++;
    return true;
}

// puts the read in flight, so the caller can spend the time it takes on the batch it already has
static size_t startMemberBatch(const RunDbReader &reader, uint64_t rep, const PairRecord *rows,
                               size_t count, const ClusterAssignmentBitmap &assignedCluster, const Parameters &par,
                               unsigned int thread, unsigned int lane, Candidates &candidates) {
    candidates.clear();
    // assigned since the work list was drawn: a whole group of reads assignCluster would throw away
    if (assignedCluster.isAssigned(rep)) {
        return 0;
    }
    const uint32_t queryLen = reader.getSeqLen(rep);
    for (size_t i = 0; i < count; i++) {
        const uint64_t member = rows[i].member();
        if (member == rep || assignedCluster.isAssigned(member)) {
            continue;
        }
        if (Util::canBeCovered(par.covThr, par.covMode, queryLen, reader.getSeqLen(member)) == false) {
            continue;
        }
        candidates.members.push_back(member);
        candidates.diagonals.push_back(rows[i].diagonal());
    }
    if (candidates.members.empty()) {
        return 0;
    }
    return reader.startBatch(rep, candidates.members.data(), candidates.members.size(), thread, lane);
}

// a lane that could not hold every candidate finishes the rest through itself, and that path waits
static void alignMemberBatch(const RunDbReader &reader, uint64_t rep, size_t got,
                       const ClusterAssignmentBitmap &assignedCluster, Sequence &query, Sequence &target,
                       BlockAligner &aligner, const Parameters &par, unsigned int thread,
                       unsigned int lane, Candidates &candidates,
                       std::vector<uint64_t> &out, GateCounts &gate, float scorePerColThreshold,
                       int xDrop) {
    const uint32_t queryLen = reader.getSeqLen(rep);
    size_t from = 0;
    while (from < candidates.members.size()) {
        reader.awaitBatch(thread, lane);
        const char *querySeq = reader.batchQueryAt(thread, lane);
        query.mapSequence(0, 0, (char *) querySeq, queryLen);
        aligner.initQuery(&query);
        for (size_t k = 0; k < got; k++) {
            const uint64_t member = candidates.members[from + k];
            // assigned while in flight: the read is spent, the alignment is not
            if (assignedCluster.isAssigned(member)) {
                gate.stale++;
                continue;
            }
            const uint32_t targetLen = reader.getSeqLen(member);
            const char *targetSeq = reader.batchAt(thread, lane, k);
            target.mapSequence(0, 0, (char *) targetSeq, targetLen);
            const BlockAligner::UngappedAln_res hit = aligner.ungappedAlign(
                &target, static_cast<unsigned short>(candidates.diagonals[from + k]));
            gate.seen++;
            if (hit.eval > par.evalThr || hit.alnLen < par.alnLenThr
                || Util::hasCoverage(par.covThr, par.covMode, hit.qcov, hit.tcov) == false) {
                gate.rejected++;
                if (rescueWithGaps(member, queryLen, targetLen, querySeq, targetSeq, hit, assignedCluster,
                                   target, aligner, par, scorePerColThreshold, xDrop, gate)) {
                    out.push_back(member);
                }
                continue;
            }
            int identical = 0;
            for (int q = hit.qStart; q <= hit.qEnd; q++) {
                const char a = querySeq[q] & static_cast<unsigned char>(~0x20);
                const char b =
                    targetSeq[hit.tStart + (q - hit.qStart)] & static_cast<unsigned char>(~0x20);
                identical += (a == b);
            }
            // the same comparison upstream makes, epsilon included, so the boundary agrees
            if (Util::computeSeqId(par.seqIdMode, identical, queryLen, targetLen, hit.alnLen)
                < par.seqIdThr - FLT_EPSILON) {
                gate.rejected++;
                if (rescueWithGaps(member, queryLen, targetLen, querySeq, targetSeq, hit, assignedCluster,
                                   target, aligner, par, scorePerColThreshold, xDrop, gate)) {
                    out.push_back(member);
                }
                continue;
            }
            out.push_back(member);
        }
        from += got;
        if (from < candidates.members.size()) {
            got = reader.startBatch(rep, candidates.members.data() + from,
                                    candidates.members.size() - from, thread, lane);
        }
    }
}

int lin8align2clust(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    FileUtil::fixRlimitNoFile();
    const NodePlacement node = NodePlacement::resolve(par);
    unsigned int writerNodes = 0;
    size_t repRankBlocks = 0;
    uint64_t ranks = 0;
    readPipelineShape(par.db2, writerNodes, repRankBlocks, ranks);
    requireEveryNodeDone(par.db2, writerNodes);

    RunDbReader reader(par.db1);
    reader.open();
    if (reader.getSize() != ranks) {
        Debug(Debug::ERROR) << "The database holds " << reader.getSize()
                            << " sequences and the pairs were made for " << ranks << "\n";
        EXIT(EXIT_FAILURE);
    }
    const unsigned int threads = par.threads;
    reader.openBatch(threads, ARENA_BYTES, Util::computeMemory(par.splitMemoryLimit));

    SubstitutionMatrix subMat(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, par.scoreBias);
    SubstitutionMatrix::FastMatrix fastMatrix = SubstitutionMatrix::createAsciiSubMat(subMat);
    EvalueComputation evaluer(reader.getTotalBytes(), &subMat);
    const size_t maxLen = std::max<size_t>(reader.getRunTable().maxSeqLen(), 1);

    size_t firstRepRankBlock = par.lin8RepRankBlock < 0 ? 0 : (size_t) par.lin8RepRankBlock;
    const size_t lastRepRankBlock = par.lin8RepRankBlock < 0 ? repRankBlocks : std::min(firstRepRankBlock + 1, repRankBlocks);

    const bool decideHere = node.count == 1;
    if (decideHere && par.lin8RepRankBlock < 0) {
        while (firstRepRankBlock < lastRepRankBlock
               && FileUtil::fileExists((par.db4 + ".0." + SSTR(firstRepRankBlock)).c_str())) {
            firstRepRankBlock++;
        }
        if (firstRepRankBlock > 0) {
            Debug(Debug::INFO) << "Resuming at repRankBlock " << firstRepRankBlock << ", the earlier ones are decided\n";
        }
    }
    Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes its share of "
                       << (lastRepRankBlock - firstRepRankBlock) << " of " << repRankBlocks << " repRankBlocks\n";

    // this run's progress rather than the database's, so it sits beside the accepted pairs
    ClusterAssignmentBitmap assignedCluster;
    assignedCluster.open(par.db4 + ".align_assigned_" + SSTR(node.index), ranks);
    assignedCluster.catchUpTo(par.db4, firstRepRankBlock);
    if (decideHere == false) {
        assignedCluster.save(firstRepRankBlock);
    }
    const BucketCounts prefCounts(par.db2, writerNodes, PairRecord::REP_RANK_SUB_BLOCKS, repRankBlocks);

    Timer timer;
    uint64_t aligned = 0;
    uint64_t passed = 0;
    std::vector<PairRecord> batch;
    std::vector<size_t> starts;
    std::vector<std::vector<uint64_t> > survivors;
    std::vector<MemberBatch> work;
    std::vector<std::vector<uint64_t> > batchSurvivors;
    // where the wall clock of this pass actually goes, so a slow run says which part was slow
    double spentReading = 0, spentAligning = 0, spentDeciding = 0, spentWriting = 0;
    Debug::Progress progress(lastRepRankBlock - firstRepRankBlock);
    std::vector<PairRecord> rows;
    std::vector<std::vector<Candidates> > candidates(threads,
                                                     std::vector<Candidates>(RunDbReader::LANES));
    std::vector<PairRecord> outBuffer;
    std::vector<AlignWorker *> workers(threads, NULL);
    std::vector<GateCounts> gate(threads);
    const float scorePerColThreshold =
        parsePrecisionLib(par.covMode == Parameters::COV_MODE_BIDIRECTIONAL
                                 ? getCovSeqidQscPercMinDiag()
                                 : getCovSeqidQscPercMinDiagTargetCov(),
                             par.seqIdThr, par.covThr, 0.99);
    const int xDrop = (int) BlockAligner::MIN_BAND * par.gapExtend.values.aminoacid()
                      + par.gapOpen.values.aminoacid();

    uint64_t clusters = 0;
    uint64_t assigned = 0;

    std::vector<std::pair<std::string, std::string> > pendingOut;
    size_t pendingFirst = firstRepRankBlock;
    uint64_t pendingBytes = 0;
    for (size_t repRankBlock = firstRepRankBlock; repRankBlock < lastRepRankBlock; repRankBlock++) {
        const uint64_t blockFirstRank = PairRecord::firstRankOf(repRankBlock, ranks, repRankBlocks);
        const uint64_t blockLastRank = PairRecord::firstRankOf(repRankBlock + 1, ranks, repRankBlocks);
        const uint64_t span = blockLastRank - blockFirstRank;
        const uint64_t myFrom = blockFirstRank + span * node.index / node.count;
        const uint64_t myUntil = blockFirstRank + span * (node.index + 1) / node.count;

        const std::string outPath = decideHere ? par.db4 + ".0." + SSTR(repRankBlock)
                                              : par.db3 + "." + SSTR(node.index) + "." + SSTR(repRankBlock);
        const std::string outTmp = outPath + ".tmp";
        FILE *out = FileUtil::openAndDelete(outTmp.c_str(), "w");
        outBuffer.clear();

        uint64_t skipRows = 0;
        if (node.count > 1) {
            const size_t mySub = PairRecord::repRankSubBlockOf(myFrom, ranks, repRankBlocks);
            const std::vector<uint64_t> counts = prefCounts.of(repRankBlock, repRankBlock % writerNodes);
            for (size_t sub = 0; sub < mySub; sub++) {
                skipRows += counts[sub];
            }
        }
        RepRankBlockReader stream(par.db2, writerNodes, repRankBlock, STREAM_ROWS, skipRows);

        // only the deciding is ordered; assignCluster rechecks, so a stale view only wastes work
        while (true) {
            double mark = omp_get_wtime();
            const bool more = stream.fillBatch(batch, BATCH_ROWS, myUntil);
            spentReading += omp_get_wtime() - mark;
            if (more == false) {
                break;
            }
            starts.clear();
            starts.push_back(0);
            for (size_t i = 1; i < batch.size(); i++) {
                if (batch[i].rep() != batch[i - 1].rep()) {
                    starts.push_back(i);
                }
            }
            starts.push_back(batch.size());
            const size_t groups = starts.size() - 1;
            if (survivors.size() < groups) {
                survivors.resize(groups);
            }

            // groups are six rows on average and thousands in the tail, so the tail is shared
            work.clear();
            for (size_t g = 0; g < groups; g++) {
                survivors[g].clear();
                const uint64_t rep = batch[starts[g]].rep();
                if (rep < myFrom || rep >= myUntil || assignedCluster.isAssigned(rep)) {
                    continue;
                }
                const size_t rows = starts[g + 1] - starts[g];
                for (size_t from = 0; from < rows; from += MEMBERS_PER_ALIGN_BATCH) {
                    MemberBatch item;
                    item.group = g;
                    item.from = from;
                    item.count = std::min(MEMBERS_PER_ALIGN_BATCH, rows - from);
                    work.push_back(item);
                }
            }
            if (batchSurvivors.size() < work.size()) {
                batchSurvivors.resize(work.size());
            }

            mark = omp_get_wtime();
            // a thread draws the next batch before aligning this one, so its reads are in flight
            size_t drawn = 0;
#pragma omp parallel num_threads(threads)
            {
                unsigned int thread = 0;
#ifdef OPENMP
                thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
                if (workers[thread] == NULL) {
                    workers[thread] = new AlignWorker(maxLen, subMat, fastMatrix, evaluer, par);
                }
                AlignWorker &worker = *workers[thread];
                unsigned int lane = 0;
                size_t here = draw(drawn);
                size_t got = 0;
                if (here < work.size()) {
                    const MemberBatch &item = work[here];
                    const size_t at = starts[item.group];
                    got = startMemberBatch(reader, batch[at].rep(), &batch[at + item.from],
                                           item.count, assignedCluster, par, thread, lane,
                                           candidates[thread][lane]);
                }
                while (here < work.size()) {
                    const size_t next = draw(drawn);
                    const unsigned int nextLane = lane ^ 1u;
                    size_t nextGot = 0;
                    if (next < work.size()) {
                        const MemberBatch &item = work[next];
                        const size_t at = starts[item.group];
                        nextGot = startMemberBatch(reader, batch[at].rep(), &batch[at + item.from],
                                                   item.count, assignedCluster, par, thread, nextLane,
                                                   candidates[thread][nextLane]);
                    }
                    const MemberBatch &item = work[here];
                    const size_t at = starts[item.group];
                    batchSurvivors[here].clear();
                    alignMemberBatch(reader, batch[at].rep(), got, assignedCluster, worker.query,
                                     worker.target, worker.aligner, par, thread, lane,
                                     candidates[thread][lane], batchSurvivors[here],
                                     gate[thread], scorePerColThreshold, xDrop);
                    here = next;
                    lane = nextLane;
                    got = nextGot;
                }
            }
            // joined in the order the serial version made them
            for (size_t w = 0; w < work.size(); w++) {
                std::vector<uint64_t> &into = survivors[work[w].group];
                into.insert(into.end(), batchSurvivors[w].begin(), batchSurvivors[w].end());
            }
            spentAligning += omp_get_wtime() - mark;
            mark = omp_get_wtime();
            for (size_t g = 0; g < groups; g++) {
                const uint64_t rep = batch[starts[g]].rep();
                if (rep < myFrom || rep >= myUntil) {
                    continue;
                }
                aligned += starts[g + 1] - starts[g];
                passed += survivors[g].size();
                if (decideHere) {
                    clusters += assignCluster(rep, survivors[g].data(), survivors[g].size(), assignedCluster,
                                            outBuffer, assigned) ? 1 : 0;
                } else {
                    PairRecord line;
                    line.set(rep, rep, 0);
                    outBuffer.push_back(line);
                    for (size_t k = 0; k < survivors[g].size(); k++) {
                        line.set(rep, survivors[g][k], 0);
                        outBuffer.push_back(line);
                    }
                }
                if (outBuffer.size() >= STREAM_ROWS) {
                    const double put = omp_get_wtime();
                    if (writeRecords(outBuffer.data(), outBuffer.size(), out) != outBuffer.size()) {
                        Debug(Debug::ERROR) << "Cannot write " << outTmp << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    outBuffer.clear();
                    spentWriting += omp_get_wtime() - put;
                }
            }
            spentDeciding += omp_get_wtime() - mark;
        }

        if (outBuffer.empty() == false
            && writeRecords(outBuffer.data(), outBuffer.size(), out)
                   != outBuffer.size()) {
            Debug(Debug::ERROR) << "Cannot write " << outTmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        const uint64_t outBytes = static_cast<uint64_t>(ftello(out));
        if (fclose(out) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << outTmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (decideHere) {
            // renamed in repRankBlock order after a batched flush, so a crash leaves a whole decided prefix
            pendingOut.push_back(std::make_pair(outTmp, outPath));
            pendingBytes += outBytes;
            if (pendingBytes >= PUBLISH_BATCH_BYTES || pendingOut.size() >= PUBLISH_BATCH_FILES
                || repRankBlock + 1 == lastRepRankBlock) {
                publishAllAtomically(pendingOut, threads);
                // these repRankBlocks are decided and published, so what they read is read by nobody again
                if (par.removeTmpFiles) {
                    dropConsumed(par.db2, writerNodes, pendingFirst, repRankBlock + 1, 1);
                }
                pendingFirst = repRankBlock + 1;
                pendingBytes = 0;
            }
        } else {
            FileUtil::publishAtomically(outTmp, outPath);
            markNodeDone(par.db3 + "." + SSTR(repRankBlock), node.index);
        }
        progress.updateProgress();
    }

    if (decideHere) {
        // every repRankBlock this run decided is published, so the cache can name them all
        assignedCluster.save(lastRepRankBlock);
    }

    const std::string shapeTmp = (decideHere ? par.db4 : par.db3)
                                 + "." + SSTR(node.index) + ".shape.tmp";
    FILE *shape = FileUtil::openAndDelete(shapeTmp.c_str(), "w");
    if (decideHere) {
        fprintf(shape, "repRankBlocks\t%zu\nranks\t%zu\n", repRankBlocks, (size_t) ranks);
    } else {
        fprintf(shape, "nodes\t%u\nrepRankBlocks\t%zu\nranks\t%zu\n", node.count, repRankBlocks, (size_t) ranks);
    }
    if (fclose(shape) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << shapeTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(shapeTmp, decideHere ? par.db4 : par.db3);

    GateCounts all;
    for (size_t i = 0; i < gate.size(); i++) {
        all.seen += gate[i].seen;
        all.rejected += gate[i].rejected;
        all.rescued += gate[i].rescued;
        all.kept += gate[i].kept;
        all.stale += gate[i].stale;
    }
    Debug(Debug::INFO) << "The gate saw " << all.seen << " pairs and turned away " << all.rejected
                       << ", of which " << all.rescued << " were worth a gapped alignment and "
                       << all.kept << " held up\n";
    Debug(Debug::INFO) << all.stale << " of the sequences it read were already in a cluster by the "
                       << "time it looked, so those reads were spent for nothing\n";
    Debug(Debug::INFO) << "Where the time went: reading " << (uint64_t) spentReading << "s, aligning "
                       << (uint64_t) spentAligning << "s, deciding " << (uint64_t) spentDeciding
                       << "s, writing " << (uint64_t) spentWriting << "s\n";
    Debug(Debug::INFO) << "Aligned " << aligned << " candidates, " << passed << " passed, in "
                       << timer.lap() << "\n";
    if (decideHere) {
        Debug(Debug::INFO) << "Made " << clusters << " clusters holding " << (clusters + assigned)
                           << " sequences\n";
    }
    for (size_t i = 0; i < workers.size(); i++) {
        delete workers[i];
    }
    reader.close();
    return EXIT_SUCCESS;
}
