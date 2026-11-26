#include "Parameters.h"
#include "Util.h"
#include "CommandCaller.h"
#include "Debug.h"
#include "FileUtil.h"
#include "DBReader.h"

#include "lin8clust.sh.h"

#include <cassert>

extern void setLinclustWorkflowDefaults(Parameters *p);

int lin8clust(int argc, const char **argv, const Command& command) {
    Parameters& par = Parameters::getInstance();
    setLinclustWorkflowDefaults(&par);
    par.clustHash = true;
    par.PARAM_ADD_BACKTRACE.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_ALT_ALIGNMENT.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_ZDROP.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_RESCORE_MODE.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_MAX_REJECTED.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_MAX_ACCEPT.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.overrideParameterDescription(par.PARAM_S, "Sensitivity will be automatically determined but can be adjusted", NULL, par.PARAM_S.category | MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_INCLUDE_ONLY_EXTENDABLE.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_NODE_RUNNER.removeCategory(MMseqsParameter::COMMAND_EXPERT);

    par.parseParameters(argc, argv, command, true, 0, 0);
    std::string tmpDir = par.db3;
    std::string hash = SSTR(par.hashParameter(command.databases, par.filenames, par.lin8clustworkflow));
    if (par.reuseLatest) {
        hash = FileUtil::getHashFromSymLink(tmpDir + "/latest");
    }
    tmpDir = FileUtil::createTemporaryDirectory(tmpDir, hash);
    par.filenames.pop_back();
    par.filenames.push_back(tmpDir);

    int dbType = FileUtil::parseDbType(par.db1.c_str());
    if ((DBReader<DBKeyType>::getExtendedDbtype(dbType) & Parameters::DBTYPE_EXTENDED_LINFINITY) == 0) {
        Debug(Debug::ERROR) << "lin8clust clusters linfinity databases (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ") only. "
                            << "Use createdb --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << " or easy-lin8clust.\n";
        EXIT(EXIT_FAILURE);
    }
    // align2clust leaves out the singletons of subsets
    if (FileUtil::fileExists((par.db1 + ".bits.0").c_str())) {
        Debug(Debug::ERROR) << "lin8clust clusters whole linfinity databases (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << "), not subsets such as clusthashfast or createsubdb output\n";
        EXIT(EXIT_FAILURE);
    }

    bool nonSymetric = (par.covMode == Parameters::COV_MODE_TARGET || par.covMode == Parameters::COV_MODE_QUERY);
    if (par.PARAM_CLUSTER_MODE.wasSet == false) {
        if (nonSymetric) {
            par.clusteringMode = Parameters::GREEDY_MEM;
        } else {
            par.clusteringMode = Parameters::SET_COVER;
        }
        std::string cluMode = (par.clusteringMode == Parameters::GREEDY_MEM) ? "GREEDY MEM" : "SET COVER";
        Debug(Debug::INFO) << "Set cluster mode " << cluMode << ".\n";
    }
    if (par.clusteringMode != Parameters::SET_COVER && par.clusteringMode != Parameters::GREEDY && par.clusteringMode != Parameters::GREEDY_MEM) {
        Debug(Debug::ERROR) << "align2clust supports only --cluster-mode " << Parameters::SET_COVER << ", " << Parameters::GREEDY << " and " << Parameters::GREEDY_MEM
                            << " for linfinity databases (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ")\n";
        EXIT(EXIT_FAILURE);
    }
    if (par.PARAM_ALPH_SIZE.wasSet == false) {
        par.alphabetSize = MultiParam<NuclAA<int>>(NuclAA<int>(Parameters::CLUST_LINEAR_DEFAULT_ALPH_SIZE, 5));
    }
    par.includeCountTable = false;
    par.countTableIteration = 0;
    if (par.PARAM_INCLUDE_ADJACENCY.wasSet == false && par.PARAM_NUM_ADJACENCY.wasSet == false) {
        if (nonSymetric) {
            par.adjIteration = (par.seqIdThr < 0.5f) ? 7 : Parameters::CLUST_LINEAR2_NUM_ADJACENCY;
        } else {
            // more adjacency to make up for no count table
            par.adjIteration = (par.seqIdThr < 0.5f) ? 6 : 4;
        }
    }
    // like linclust2
    if (par.seqIdThr >= 0.7f && par.PARAM_LINCLUST_ITERATIONS.wasSet == false) {
        par.linclustIterations = 1;
    }

    // let punchHole warn before the run if it is not supported
    std::string punchFile = tmpDir + "/punch_hole_check";
    FILE *punchHandle = FileUtil::openAndDelete(punchFile.c_str(), "w");
    std::vector<char> page(Util::getPageSize(), 0);
    if (fwrite(page.data(), 1, page.size(), punchHandle) != page.size() || fflush(punchHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << punchFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::punchHole(fileno(punchHandle), 0, page.size());
    fclose(punchHandle);
    FileUtil::remove(punchFile.c_str());

    CommandCaller cmd;
    cmd.addVariable("MERGECLU_PAR", par.createParameterString(par.mergeclusters).c_str());
    cmd.addVariable("REMOVE_TMP", par.removeTmpFiles ? "TRUE" : NULL);
    cmd.addVariable("RUNNER", par.nodeRunner.c_str());
    cmd.addVariable("CLUSTHASH", par.clustHash ? "TRUE" : NULL);
    cmd.addVariable("KMERMATCHER_PAR", par.createParameterString(par.kmermatcher).c_str());
    cmd.addVariable("ALIGN2CLUST_PAR", par.createParameterString(par.align2clust).c_str());
    cmd.addVariable("VERBOSITY", par.createParameterString(par.onlyverbosity).c_str());
    cmd.addVariable("LINCLUST_ITERATIONS", SSTR(par.linclustIterations).c_str());
    // second iteration as in linclust
    if (par.linclustIterations > 1) {
        if (par.PARAM_SPACED_KMER_MODE.wasSet == false) {
            par.spacedKmer = true;
        }
        par.kmersPerSequenceScale.values.first += 0.1f;
        par.kmersPerSequenceScale.values.second += 0.1f;
        if (par.seqIdThr <= 0.5f && par.PARAM_KMER_SCORE_MODE.wasSet == false) {
            par.kmerScoreMode = 1;
        }
        if (par.seqIdThr <= 0.5f && par.PARAM_SEED_MEMBER_PER_REP.wasSet == false) {
            par.seedMemberPerRep = (par.seqIdThr < 0.5f) ? 4 : 3;
        }
        par.subDbMode = Parameters::SUBDB_MODE_SOFT;
        cmd.addVariable("CREATESUBDB_PAR", par.createParameterString(par.createsubdb).c_str());
        cmd.addVariable("KMERMATCHER_PAR2", par.createParameterString(par.kmermatcher).c_str());
    }

    // # 0. clust hash
    par.seqIdThr = std::max(0.9f, par.seqIdThr);
    par.alphabetSize = MultiParam<NuclAA<int>>(NuclAA<int>(Parameters::CLUST_HASH_DEFAULT_ALPH_SIZE, 5));
    cmd.addVariable("CLUSTHASHFAST_PAR", par.createParameterString(par.clusthashfast).c_str());

    std::string program = tmpDir + "/lin8clust.sh";
    FileUtil::writeFile(program, lin8clust_sh, lin8clust_sh_len);
    cmd.execProgram(program.c_str(), par.filenames);

    // Unreachable
    assert(false);
    return 0;
}
