#include "Lin8Db.h"
#include "Lin8DbReader.h"
#include "Parameters.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"
#include "Timer.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static void readPipelineShape(const std::string &path, unsigned int &nodes, size_t &repRankBlocks, uint64_t &ranks) {
    FILE *in = fopen(path.c_str(), "r");
    if (in == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << path << ". Run lin8align first\n";
        EXIT(EXIT_FAILURE);
    }
    nodes = 0;
    repRankBlocks = 0;
    size_t seen = 0;
    const bool read = fscanf(in, "nodes\t%u\nrepRankBlocks\t%zu\nranks\t%zu", &nodes, &repRankBlocks, &seen) == 3;
    fclose(in);
    ranks = seen;
    if (read == false || nodes == 0 || repRankBlocks == 0) {
        Debug(Debug::ERROR) << path << " does not name a set of aligned repRankBlocks\n";
        EXIT(EXIT_FAILURE);
    }
}

static void readRepRankBlock(const std::string &prefix, unsigned int nodes, size_t repRankBlock, size_t budget,
                      std::vector<PairRecord> &rows) {
    rows.clear();
    size_t bytes = 0;
    for (unsigned int node = 0; node < nodes; node++) {
        bytes += FileUtil::getFileSize(prefix + "." + SSTR(node) + "." + SSTR(repRankBlock));
    }
    requireArena("Representative rank block " + SSTR(repRankBlock), bytes / PairRecord::DISK_BYTES * sizeof(PairRecord),
                 budget, "raise --rep-rank-blocks");
    std::vector<PairRecord> buffer(1u << 16);
    for (unsigned int node = 0; node < nodes; node++) {
        const std::string path = prefix + "." + SSTR(node) + "." + SSTR(repRankBlock);
        FILE *in = fopen(path.c_str(), "r");
        if (in == NULL) {
            continue;
        }
        size_t read = 0;
        while ((read = readRecords(buffer.data(), buffer.size(), in)) > 0) {
            rows.insert(rows.end(), buffer.begin(), buffer.begin() + read);
        }
        if (ferror(in) != 0) {
            Debug(Debug::ERROR) << "Cannot read " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        fclose(in);
    }
}

int lin8cluster(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    FileUtil::fixRlimitNoFile();
    unsigned int alignNodes = 0;
    size_t repRankBlocks = 0;
    uint64_t ranks = 0;
    readPipelineShape(par.db1, alignNodes, repRankBlocks, ranks);
    if (par.lin8RepRankBlock < 0 || (size_t) par.lin8RepRankBlock >= repRankBlocks) {
        Debug(Debug::ERROR) << "--repRankBlock must name one of the " << repRankBlocks << " repRankBlocks\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t firstRepRankBlock = (size_t) par.lin8RepRankBlock;
    const size_t lastRepRankBlock = firstRepRankBlock + 1;
    for (size_t repRankBlock = firstRepRankBlock; repRankBlock < lastRepRankBlock; repRankBlock++) {
        requireEveryNodeDone(par.db1 + "." + SSTR(repRankBlock), alignNodes);
    }

    RankBitmap taken;
    taken.open(par.linclustTaken, ranks);
    taken.catchUpTo(par.db2, firstRepRankBlock);

    const std::string outPath = par.db2 + ".0." + SSTR(firstRepRankBlock);
    const std::string outTmp = outPath + ".tmp";
    FILE *out = FileUtil::openAndDelete(outTmp.c_str(), "w");

    const size_t budget = Util::computeMemory(par.splitMemoryLimit);
    Timer timer;
    uint64_t clusters = 0;
    uint64_t assigned = 0;
    std::vector<PairRecord> rows;
    std::vector<PairRecord> outBuffer;
    std::vector<uint64_t> members;
    Debug::Progress progress(lastRepRankBlock - firstRepRankBlock);

    for (size_t repRankBlock = firstRepRankBlock; repRankBlock < lastRepRankBlock; repRankBlock++) {
        readRepRankBlock(par.db1, alignNodes, repRankBlock, budget, rows);
        size_t at = 0;
        uint64_t lastRep = 0;
        while (at < rows.size()) {
            const uint64_t rep = rows[at].rep();
            if (rep < lastRep) {
                Debug(Debug::ERROR) << "Representative rank block " << repRankBlock << " goes back from representative "
                                    << lastRep << " to " << rep
                                    << ". The aligning pass did not cut it in rank order\n";
                EXIT(EXIT_FAILURE);
            }
            lastRep = rep;
            size_t end = at;
            while (end < rows.size() && rows[end].rep() == rep) {
                end++;
            }
            members.clear();
            for (size_t i = at; i < end; i++) {
                members.push_back(rows[i].member());
            }
            clusters += takeCluster(rep, members.data(), members.size(), taken, outBuffer, assigned)
                        ? 1 : 0;
            at = end;
            if (outBuffer.size() >= (1u << 16)) {
                if (writeRecords(outBuffer.data(), outBuffer.size(), out)
                    != outBuffer.size()) {
                    Debug(Debug::ERROR) << "Cannot write " << outTmp << "\n";
                    EXIT(EXIT_FAILURE);
                }
                outBuffer.clear();
            }
        }
        progress.updateProgress();
    }
    if (outBuffer.empty() == false
        && writeRecords(outBuffer.data(), outBuffer.size(), out) != outBuffer.size()) {
        Debug(Debug::ERROR) << "Cannot write " << outTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(out) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << outTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(outTmp, outPath);
    // the decisions are on disk now, so the cache may name the repRankBlocks that made them
    taken.save(lastRepRankBlock);
    if (par.linclustPref.empty() == false && par.removeTmpFiles) {
        for (size_t repRankBlock = firstRepRankBlock; repRankBlock < lastRepRankBlock; repRankBlock++) {
            dropConsumed(par.linclustPref, alignNodes, repRankBlock, repRankBlock + 1, 1);
        }
    }

    const std::string shapeTmp = par.db2 + "." + SSTR(firstRepRankBlock) + ".shape.tmp";
    FILE *shape = FileUtil::openAndDelete(shapeTmp.c_str(), "w");
    fprintf(shape, "repRankBlocks\t%zu\nranks\t%zu\n", repRankBlocks, (size_t) ranks);
    if (fclose(shape) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << shapeTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(shapeTmp, par.db2);

    Debug(Debug::INFO) << "Made " << clusters << " clusters holding " << (clusters + assigned)
                       << " sequences in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}
