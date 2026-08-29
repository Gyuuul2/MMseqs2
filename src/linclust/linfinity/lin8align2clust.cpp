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

struct RingSlot {
    RingSlot() : state(0), rep(0) {}
    volatile int state;   // 0 empty, 1 filled, 2 aligned
    uint64_t rep;
    std::vector<PairRecord> rows;
    std::vector<uint64_t> survivors;
};

static const size_t STREAM_ROWS = 1u << 16;
static const size_t RING_SLOTS = 4096;
static const size_t ARENA_BYTES = 64u << 20;

static const size_t MEMBERS_PER_SLICE = 512;

// one representative and a slice of its members, so a big group becomes many pieces
static void readShape(const std::string &path, unsigned int &nodes, size_t &ranges, uint64_t &ranks) {
    FILE *in = fopen(path.c_str(), "r");
    if (in == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << path << ". Run lin8pref first\n";
        EXIT(EXIT_FAILURE);
    }
    nodes = 0;
    ranges = 0;
    size_t seen = 0;
    const bool read = fscanf(in, "nodes\t%u\nranges\t%zu\nranks\t%zu", &nodes, &ranges, &seen) == 3;
    fclose(in);
    ranks = seen;
    if (read == false || nodes == 0 || ranges == 0 || ranges > PairRecord::MAX_RANGE_COUNT) {
        Debug(Debug::ERROR) << path << " names " << nodes << " nodes and " << ranges
                            << " ranges, which is not a set of folded pairs\n";
        EXIT(EXIT_FAILURE);
    }
}

class RangeReader {
public:
    RangeReader(const std::string &prefix, unsigned int foldNodes, size_t range, size_t bufferRows,
                uint64_t skipRows)
        : buffer(bufferRows), at(0), filled(0) {
        const std::string path =
            prefix + "." + SSTR(range % foldNodes) + "." + SSTR(range);
        file = fopen(path.c_str(), "r");
        if (file == NULL) {
            Debug(Debug::ERROR) << "Cannot open " << path << ", which node " << (range % foldNodes)
                                << " should have written for range " << range << "\n";
            EXIT(EXIT_FAILURE);
        }
        // off_t, not long: a range is two hundred gigabytes at a trillion sequences
        const off_t skip = static_cast<off_t>(skipRows * PairRecord::DISK_BYTES);
        if (skip > 0 && fseeko(file, skip, SEEK_SET) != 0) {
            Debug(Debug::ERROR) << "Cannot seek " << skip << " byte into " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        refill();
    }

    ~RangeReader() {
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

private:
    RangeReader(const RangeReader &);
    RangeReader &operator=(const RangeReader &);

    void refill() {
        filled = readRecords(buffer.data(), buffer.size(), file);
        at = 0;
    }

    std::vector<PairRecord> buffer;
    FILE *file;
    size_t at;
    size_t filled;
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

struct GateCounts {
    GateCounts() : seen(0), rejected(0), rescued(0), kept(0) {}
    uint64_t seen;
    uint64_t rejected;
    uint64_t rescued;   // turned away and worth a gapped alignment
    uint64_t kept;      // and the gapped alignment held up
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
                           const BlockAligner::UngappedAln_res &hit, const RankBitmap &taken,
                           Sequence &target, BlockAligner &aligner, const Parameters &par,
                           float scorePerColThreshold, int xDrop, GateCounts &gate) {
    if (hit.diagonalLen <= 0
        || (float) hit.score / (float) hit.diagonalLen < scorePerColThreshold) {
        return false;
    }
    if (hit.qStart < 0 || hit.tStart < 0 || hit.alnLen < 3 || taken.taken(member)) {
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

static void alignSlice(const RunDbReader &reader, uint64_t rep, const PairRecord *rows, size_t count,
                       const RankBitmap &taken, Sequence &query, Sequence &target,
                       BlockAligner &aligner, const Parameters &par, unsigned int thread,
                       Candidates &candidates, std::vector<uint64_t> &out, GateCounts &gate,
                       float scorePerColThreshold, int xDrop) {
    candidates.clear();
    const uint32_t queryLen = reader.getSeqLen(rep);
    const char *querySeq = reader.getData(rep);
    query.mapSequence(0, 0, (char *) querySeq, queryLen);
    aligner.initQuery(&query);

    for (size_t i = 0; i < count; i++) {
        const uint64_t member = rows[i].member();
        if (member == rep || taken.taken(member)) {
            continue;
        }
        if (Util::canBeCovered(par.covThr, par.covMode, queryLen, reader.getSeqLen(member)) == false) {
            continue;
        }
        candidates.members.push_back(member);
        candidates.diagonals.push_back(rows[i].diagonal());
    }

    size_t from = 0;
    while (from < candidates.members.size()) {
        const size_t got = reader.loadBatch(candidates.members.data() + from,
                                            candidates.members.size() - from, thread);
        if (got == 0) {
            Debug(Debug::ERROR) << "A single sequence does not fit the read arena\n";
            EXIT(EXIT_FAILURE);
        }
        for (size_t k = 0; k < got; k++) {
            const uint64_t member = candidates.members[from + k];
            const uint32_t targetLen = reader.getSeqLen(member);
            const char *targetSeq = reader.batchAt(thread, k);
            target.mapSequence(0, 0, (char *) targetSeq, targetLen);
            const BlockAligner::UngappedAln_res hit = aligner.ungappedAlign(
                &target, static_cast<unsigned short>(candidates.diagonals[from + k]));
            gate.seen++;
            if (hit.eval > par.evalThr || hit.alnLen < par.alnLenThr
                || Util::hasCoverage(par.covThr, par.covMode, hit.qcov, hit.tcov) == false) {
                gate.rejected++;
                if (rescueWithGaps(member, queryLen, targetLen, querySeq, targetSeq, hit, taken,
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
                if (rescueWithGaps(member, queryLen, targetLen, querySeq, targetSeq, hit, taken,
                                   target, aligner, par, scorePerColThreshold, xDrop, gate)) {
                    out.push_back(member);
                }
                continue;
            }
            out.push_back(member);
        }
        from += got;
    }
}

int lin8align(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    FileUtil::fixRlimitNoFile();
    const NodePlacement node = NodePlacement::resolve(par);
    unsigned int writerNodes = 0;
    size_t ranges = 0;
    uint64_t ranks = 0;
    readShape(par.db2, writerNodes, ranges, ranks);
    requireEveryNodeDone(par.db2, writerNodes);

    RunDbReader reader(par.db1, par.linclusthashValid);
    reader.open();
    if (reader.getSize() != ranks) {
        Debug(Debug::ERROR) << "The database holds " << reader.getSize()
                            << " sequences and the pairs were made for " << ranks << "\n";
        EXIT(EXIT_FAILURE);
    }
    const unsigned int threads = std::max<unsigned int>(par.threads, 3);
    reader.openBatch(threads, ARENA_BYTES);

    SubstitutionMatrix subMat(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, par.scoreBias);
    SubstitutionMatrix::FastMatrix fastMatrix = SubstitutionMatrix::createAsciiSubMat(subMat);
    EvalueComputation evaluer(reader.getTotalBytes(), &subMat);
    const size_t maxLen = std::max<size_t>(reader.getRunTable().maxSeqLen(), 1);

    size_t firstRange = par.linclustRange < 0 ? 0 : (size_t) par.linclustRange;
    const size_t lastRange = par.linclustRange < 0 ? ranges : std::min(firstRange + 1, ranges);

    const bool decideHere = node.count == 1;
    if (decideHere && par.linclustDecided.empty()) {
        Debug(Debug::ERROR) << "One machine decides as it aligns and needs --decided to write to\n";
        EXIT(EXIT_FAILURE);
    }
    if (decideHere && par.linclustRange < 0) {
        while (firstRange < lastRange
               && FileUtil::fileExists((par.linclustDecided + ".0." + SSTR(firstRange)).c_str())) {
            firstRange++;
        }
        if (firstRange > 0) {
            Debug(Debug::INFO) << "Resuming at range " << firstRange << ", the earlier ones are decided\n";
        }
    }
    Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes its share of "
                       << (lastRange - firstRange) << " of " << ranges << " ranges\n";

    RankBitmap taken;
    taken.open(par.linclustTaken, ranks);
    if (par.linclustDecided.empty() == false) {
        taken.catchUpTo(par.linclustDecided, firstRange);
        if (decideHere == false) {
            taken.save(firstRange);
        }
    }
    const BucketCounts foldCounts(par.db2, writerNodes, PairRecord::SUB_RANGE_COUNT, ranges);

    Timer timer;
    uint64_t aligned = 0;
    uint64_t passed = 0;
    Debug::Progress progress(lastRange - firstRange);
    std::vector<PairRecord> rows;
    std::vector<Candidates> candidates(threads);
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
    size_t pendingFirst = firstRange;
    uint64_t pendingBytes = 0;
    for (size_t range = firstRange; range < lastRange; range++) {
        const uint64_t rangeFrom = PairRecord::firstRankOf(range, ranks, ranges);
        const uint64_t rangeUntil = PairRecord::firstRankOf(range + 1, ranks, ranges);
        const uint64_t span = rangeUntil - rangeFrom;
        const uint64_t myFrom = rangeFrom + span * node.index / node.count;
        const uint64_t myUntil = rangeFrom + span * (node.index + 1) / node.count;

        const std::string outPath = decideHere ? par.linclustDecided + ".0." + SSTR(range)
                                              : par.db3 + "." + SSTR(node.index) + "." + SSTR(range);
        const std::string outTmp = outPath + ".tmp";
        FILE *out = FileUtil::openAndDelete(outTmp.c_str(), "w");
        outBuffer.clear();

        uint64_t skipRows = 0;
        if (node.count > 1) {
            const size_t mySub = PairRecord::subRangeOf(myFrom, ranks, ranges);
            const std::vector<uint64_t> counts = foldCounts.of(range, range % writerNodes);
            for (size_t sub = 0; sub < mySub; sub++) {
                skipRows += counts[sub];
            }
        }
        RangeReader stream(par.db2, writerNodes, range, STREAM_ROWS, skipRows);

        std::vector<RingSlot> ring(RING_SLOTS);
        volatile uint64_t filled = 0;      // slots the reader has produced
        volatile uint64_t claimed = 0;     // slots an aligner has taken
        volatile uint64_t decided = 0;     // slots the decider has drained
        volatile bool ended = false;       // the reader saw the end of this machine's share

#pragma omp parallel num_threads(threads)
        {
            unsigned int thread = 0;
#ifdef OPENMP
            thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
            if (thread == 0) {
                // reader: one representative a slot, in rank order
                PairRecord row;
                bool more = stream.next(row);
                uint64_t at = 0;
                while (more) {
                    while (at - decided >= RING_SLOTS) {
                        sched_yield();
                    }
                    RingSlot &slot = ring[at % RING_SLOTS];
                    slot.rows.clear();
                    slot.survivors.clear();
                    slot.rep = row.rep();
                    while (more && row.rep() == slot.rep) {
                        slot.rows.push_back(row);
                        more = stream.next(row);
                    }
                    if (more && row.rep() >= myUntil) {
                        more = false;
                    }
                    __sync_synchronize();
                    slot.state = 1;
                    at++;
                    filled = at;
                }
                ended = true;
            } else if (thread == 1) {
                // decider: strictly in index order, the only writer of the bitmap
                uint64_t at = 0;
                while (true) {
                    while (at >= filled) {
                        if (ended && at >= filled) {
                            break;
                        }
                        sched_yield();
                    }
                    if (at >= filled && ended) {
                        break;
                    }
                    RingSlot &slot = ring[at % RING_SLOTS];
                    while (slot.state != 2) {
                        sched_yield();
                    }
                    __sync_synchronize();
                    if (slot.rows.empty() == false) {
                        aligned += slot.rows.size();
                        passed += slot.survivors.size();
                        if (decideHere) {
                            clusters += takeCluster(slot.rep, slot.survivors.data(),
                                                    slot.survivors.size(), taken, outBuffer,
                                                    assigned) ? 1 : 0;
                        } else {
                            PairRecord line;
                            line.set(slot.rep, slot.rep, 0);
                            outBuffer.push_back(line);
                            for (size_t k = 0; k < slot.survivors.size(); k++) {
                                line.set(slot.rep, slot.survivors[k], 0);
                                outBuffer.push_back(line);
                            }
                        }
                        if (outBuffer.size() >= STREAM_ROWS) {
                            if (writeRecords(outBuffer.data(), outBuffer.size(), out)
                                != outBuffer.size()) {
                                Debug(Debug::ERROR) << "Cannot write " << outTmp << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                            outBuffer.clear();
                        }
                    }
                    slot.state = 0;
                    at++;
                    decided = at;
                }
            } else {
                // aligners: claim the next slot and align it
                while (true) {
                    uint64_t at = __sync_fetch_and_add((uint64_t *) &claimed, 1);
                    while (at >= filled) {
                        if (ended && at >= filled) {
                            break;
                        }
                        sched_yield();
                    }
                    if (at >= filled && ended) {
                        break;
                    }
                    RingSlot &slot = ring[at % RING_SLOTS];
                    while (slot.state != 1) {
                        sched_yield();
                    }
                    __sync_synchronize();
                    if (workers[thread] == NULL) {
                        workers[thread] = new AlignWorker(maxLen, subMat, fastMatrix, evaluer, par);
                    }
                    AlignWorker &worker = *workers[thread];
                    slot.survivors.clear();
                    if (slot.rep >= myFrom && slot.rep < myUntil && taken.taken(slot.rep) == false) {
                        for (size_t from = 0; from < slot.rows.size(); from += MEMBERS_PER_SLICE) {
                            const size_t until =
                                std::min(from + MEMBERS_PER_SLICE, slot.rows.size());
                            alignSlice(reader, slot.rep, &slot.rows[from], until - from, taken,
                                       worker.query, worker.target, worker.aligner, par, thread,
                                       candidates[thread], slot.survivors, gate[thread],
                                       scorePerColThreshold, xDrop);
                        }
                    } else {
                        slot.rows.clear();
                    }
                    __sync_synchronize();
                    slot.state = 2;
                }
            }
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
            // renamed in range order after a batched flush, so a crash leaves a whole decided prefix
            pendingOut.push_back(std::make_pair(outTmp, outPath));
            pendingBytes += outBytes;
            if (pendingBytes >= PUBLISH_BATCH_BYTES || pendingOut.size() >= PUBLISH_BATCH_FILES
                || range + 1 == lastRange) {
                publishAllAtomically(pendingOut, threads);
                // these ranges are decided and published, so what they read is read by nobody again
                if (par.removeTmpFiles) {
                    dropConsumed(par.db2, writerNodes, pendingFirst, range + 1, 1);
                }
                pendingFirst = range + 1;
                pendingBytes = 0;
            }
        } else {
            FileUtil::publishAtomically(outTmp, outPath);
            markNodeDone(par.db3 + "." + SSTR(range), node.index);
        }
        progress.updateProgress();
    }

    if (decideHere) {
        // every range this run decided is published, so the cache can name them all
        taken.save(lastRange);
    }

    const std::string shapeTmp = (decideHere ? par.linclustDecided : par.db3)
                                 + "." + SSTR(node.index) + ".shape.tmp";
    FILE *shape = FileUtil::openAndDelete(shapeTmp.c_str(), "w");
    if (decideHere) {
        fprintf(shape, "ranges\t%zu\nranks\t%zu\n", ranges, (size_t) ranks);
    } else {
        fprintf(shape, "nodes\t%u\nranges\t%zu\nranks\t%zu\n", node.count, ranges, (size_t) ranks);
    }
    if (fclose(shape) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << shapeTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(shapeTmp, decideHere ? par.linclustDecided : par.db3);

    GateCounts all;
    for (size_t i = 0; i < gate.size(); i++) {
        all.seen += gate[i].seen;
        all.rejected += gate[i].rejected;
        all.rescued += gate[i].rescued;
        all.kept += gate[i].kept;
    }
    Debug(Debug::INFO) << "The gate saw " << all.seen << " pairs and turned away " << all.rejected
                       << ", of which " << all.rescued << " were worth a gapped alignment and "
                       << all.kept << " held up\n";
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
