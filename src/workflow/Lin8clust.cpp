#include "Parameters.h"
#include "CommandCaller.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"

#include <vector>

#include "NodePlacement.h"

#include "lin8clust.sh.h"

void setLinclustOneshotWorkflowDefaults(Parameters *p) {
    p->covThr = 0.8;
    p->seqIdThr = 0.9;
    p->covMode = Parameters::COV_MODE_BIDIRECTIONAL;
    p->maskMode = 0;
    p->spacedKmer = false;
    p->clusteringMode = Parameters::GREEDY;
    // on by default here, unlike linclust: at a trillion metagenomic sequences the exact duplicates
    // are most of the input, and folding them away is what the passes after it are sized for
    p->clustHash = true;
}

// The script passes these itself: which machine this is, where it is in the wave, and the paths the
// passes hand to each other. mmseqs refuses a flag given twice, so they are dropped from the strings.
static bool scriptOwns(const Parameters &par, const MMseqsParameter *p) {
    const int owned[] = {par.PARAM_LINCLUSTERDB_NODE_LIST.uniqid, par.PARAM_LINCLUSTERDB_NODE_ID.uniqid,
                         par.PARAM_LINCLUSTERDB_NODE_COUNT.uniqid, par.PARAM_LINCLUSTHASH_VALID.uniqid,
                         par.PARAM_LINCLUST_RANGE.uniqid, par.PARAM_LINCLUST_RANGES.uniqid,
                         par.PARAM_LINCLUST_TAKEN.uniqid, par.PARAM_LINCLUST_DECIDED.uniqid,
                         par.PARAM_LINCLUST_PREF.uniqid, par.PARAM_THREADS.uniqid,
                         par.PARAM_MIN_SEQ_ID.uniqid, par.PARAM_C.uniqid, par.PARAM_COV_MODE.uniqid,
                         par.PARAM_CLUST_HASH.uniqid, par.PARAM_TSV.uniqid};
    for (size_t i = 0; i < sizeof(owned) / sizeof(owned[0]); i++) {
        if (p->uniqid == owned[i]) {
            return true;
        }
    }
    return false;
}

static std::string passOn(Parameters &par, const std::vector<MMseqsParameter *> &all) {
    std::vector<MMseqsParameter *> mine;
    for (size_t i = 0; i < all.size(); i++) {
        if (scriptOwns(par, all[i]) == false) {
            mine.push_back(all[i]);
        }
    }
    // only what the caller actually asked for: a pass reads a zero as "decide this yourself", and
    // handing it the global default instead would answer a question the pass was meant to answer
    return par.createParameterString(mine, true);
}

int lin8clust(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    setLinclustOneshotWorkflowDefaults(&par);
    par.overrideParameterDescription(par.PARAM_LINCLUSTERDB_NODE_LIST,
                                     "Every machine this runs on, in order. A machine finds itself "
                                     "in it by host name and that position is its number",
                                     NULL, par.PARAM_LINCLUSTERDB_NODE_LIST.category);
    par.overrideParameterDescription(par.PARAM_LINCLUSTERDB_NODE_ID,
                                     "Take this position in --node-list rather than the one the host "
                                     "name finds, for a scheduler that renames machines or a test "
                                     "putting several on one", NULL, par.PARAM_LINCLUSTERDB_NODE_ID.category);
    par.overrideParameterDescription(par.PARAM_TSV,
                                 "Name the clustering as a two column tsv beside the cluster database",
                                 NULL, par.PARAM_TSV.category);
    par.overrideParameterDescription(par.PARAM_CLUST_HASH,
                                     "Fold exact duplicates away before the k-mer passes and put "
                                     "them back at the end", NULL, par.PARAM_CLUST_HASH.category);
    par.parseParameters(argc, argv, command, true, Parameters::PARSE_VARIADIC, 0);

    // The wave is greedy by rank: a representative may only take what no lower rank took, and the
    // range order is that rank order, so the other modes have no order to run in.
    if (par.clusteringMode != Parameters::GREEDY && par.clusteringMode != Parameters::GREEDY_MEM) {
        Debug(Debug::ERROR) << "lin8clust clusters greedily by length, so --cluster-mode must "
                            << "be " << Parameters::GREEDY << " or " << Parameters::GREEDY_MEM << "\n";
        EXIT(EXIT_FAILURE);
    }

    const std::string tmpDir = par.filenames.back();
    par.filenames.pop_back();
    const std::string out = par.filenames.back();
    par.filenames.pop_back();
    if (FileUtil::directoryExists(tmpDir.c_str()) == false
        && FileUtil::makeDir(tmpDir.c_str()) == false) {
        Debug(Debug::ERROR) << "Cannot create tmp directory " << tmpDir << "\n";
        EXIT(EXIT_FAILURE);
    }

    // Every machine runs this same command with the same arguments and its own --node-id, so the
    // script is handed the node pair rather than reading it from the environment.
    CommandCaller cmd;
    cmd.addVariable("REMOVE_TMP", par.removeTmpFiles ? "TRUE" : NULL);
    // the same resolution the passes do, so a node list places the script as it places them
    const NodePlacement node = NodePlacement::resolve(par);
    cmd.addVariable("NODES", SSTR(node.count).c_str());
    cmd.addVariable("NODE", SSTR(node.index).c_str());
    cmd.addVariable("RANGES", SSTR(par.linclustRanges).c_str());
    cmd.addVariable("THREADS", SSTR(par.threads).c_str());
    cmd.addVariable("SEQID", SSTR(par.seqIdThr).c_str());
    cmd.addVariable("COV", SSTR(par.covThr).c_str());
    cmd.addVariable("COVMODE", SSTR(par.covMode).c_str());
    cmd.addVariable("CLUSTHASH", par.clustHash ? "1" : "0");
    cmd.addVariable("TSV", par.tsvOut ? "1" : "0");
    cmd.addVariable("REPSEQ", par.fastaSplits > 0 ? "1" : "0");
    cmd.addVariable("VERBOSITY", par.createParameterString(par.onlyverbosity).c_str());
    cmd.addVariable("CREATEDB_PAR", passOn(par, par.lin8createdb).c_str());
    cmd.addVariable("HASH_PAR", passOn(par, par.lin8clusthash).c_str());
    cmd.addVariable("EXTRACT_PAR", passOn(par, par.lin8kmers).c_str());
    cmd.addVariable("GROUP_PAR", passOn(par, par.lin8pairs).c_str());
    cmd.addVariable("FOLD_PAR", passOn(par, par.lin8pref).c_str());
    cmd.addVariable("ALIGN_PAR", passOn(par, par.lin8align).c_str());
    cmd.addVariable("ASSIGN_PAR", passOn(par, par.lin8cluster).c_str());
    cmd.addVariable("EXPAND_PAR", passOn(par, par.lin8expand).c_str());
    cmd.addVariable("CLUSTERDB_PAR", passOn(par, par.lin8merge).c_str());
    cmd.addVariable("TSV_PAR", passOn(par, par.lin8createtsv).c_str());
    cmd.addVariable("REPSEQ_PAR", passOn(par, par.lin8repseq).c_str());

    std::string program = tmpDir + "/lin8clust.sh";
    FileUtil::writeFile(program, lin8clust_sh, lin8clust_sh_len);
    par.filenames.push_back(out);
    par.filenames.push_back(tmpDir);
    cmd.execProgram(program.c_str(), par.filenames);

    return EXIT_SUCCESS;
}
