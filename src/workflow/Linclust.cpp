#include "Parameters.h"
#include "Util.h"
#include "DBWriter.h"
#include "CommandCaller.h"
#include "Debug.h"
#include "FileUtil.h"

#include "linclust.sh.h"

#include <cassert>

void setLinclustWorkflowDefaults(Parameters *p) {
    p->spacedKmer = false;
    p->covThr = 0.8;
    p->maskMode = 0;
    p->evalThr = 0.001;
    p->seqIdThr = 0.9;
    p->alignmentMode = Parameters::ALIGNMENT_MODE_SCORE_COV_SEQID; // set alignmentmode 3 as a default in linclust2
    p->clustHash = false;
}

int linclust(int argc, const char **argv, const Command& command) {
    Parameters& par = Parameters::getInstance();
    setLinclustWorkflowDefaults(&par);
    par.PARAM_ADD_BACKTRACE.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_ALT_ALIGNMENT.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_ZDROP.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_RESCORE_MODE.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_MAX_REJECTED.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_MAX_ACCEPT.addCategory(MMseqsParameter::COMMAND_EXPERT);
    par.overrideParameterDescription(par.PARAM_S, "Sensitivity will be automatically determined but can be adjusted", NULL, par.PARAM_S.category | MMseqsParameter::COMMAND_EXPERT);
    par.PARAM_INCLUDE_ONLY_EXTENDABLE.addCategory(MMseqsParameter::COMMAND_EXPERT);

    par.parseParameters(argc, argv, command, true, 0, 0);
    if (par.linclustVersion != Parameters::LINCLUST_VERSION2
        && par.kmerMatcherMode == Parameters::KMERMATCHER_MODE_LOCAL) {
        Debug(Debug::ERROR) << "--kmermatcher-mode 2 requires --linclust-version 2; "
                            << "linclust1 passes kmermatcher results to generic rescore/alignment commands. "
                            << "Use --kmermatcher-mode 1 with --linclust-version 1.\n";
        EXIT(EXIT_FAILURE);
    }
    std::string tmpDir = par.db3;
    std::string hash = SSTR(par.hashParameter(command.databases, par.filenames, par.linclustworkflow));
    if (par.reuseLatest) {
        hash = FileUtil::getHashFromSymLink(tmpDir + "/latest");
    }
    tmpDir = FileUtil::createTemporaryDirectory(tmpDir, hash);
    par.filenames.pop_back();
    par.filenames.push_back(tmpDir);

    CommandCaller cmd;
    cmd.addVariable("REMOVE_TMP", par.removeTmpFiles ? "TRUE" : NULL);
    cmd.addVariable("RUNNER", par.runner.c_str());
    // both versions call mergeclusters, and nothing below changes threads, compression or verbosity
    cmd.addVariable("MERGECLU_PAR", par.createParameterString(par.threadsandcompression).c_str());
    if (par.linclustVersion == 1) {
        cmd.addVariable("LINCLUST_MODULE", "linclust1");
    } else if (par.linclustVersion == 2) {
        cmd.addVariable("LINCLUST_MODULE", "linclust2");
    }

    // Optionally switch to profile-consensus representatives after clustering.
    // This reuses the representative-to-member alignments, so force their creation.
    const bool writeAlnFiles = par.PARAM_INCLUDE_ALIGN_FILES.wasSet;
    if (par.switchConsensusRep && par.linclustVersion != Parameters::LINCLUST_VERSION2) {
        Debug(Debug::WARNING) << "--switch-consensus-rep requires --linclust-version 2; ignoring.\n";
        par.switchConsensusRep = false;
    }
    if (par.switchConsensusRep) {
        par.includeAlignFiles = true;
        par.addBacktrace = true;
    }
    // save some values to restore them later
    MultiParam<NuclAA<int>>alphabetSize = par.alphabetSize;
    size_t kmerSize = par.kmerSize;
    // each of these is in par.linclustworkflow, so wasSet is exactly what the uniqid scan computed
    const bool kmerSizeWasSet = par.PARAM_K.wasSet;
    const bool alphabetSizeWasSet = par.PARAM_ALPH_SIZE.wasSet;
    const bool clusterModeSet = par.PARAM_CLUSTER_MODE.wasSet;
    const bool includeCountTableSet = par.PARAM_INCLUDE_COUNTTABLE.wasSet || par.PARAM_NUM_COUNTS.wasSet;
    const bool includeAdjacencySet = par.PARAM_INCLUDE_ADJACENCY.wasSet || par.PARAM_NUM_ADJACENCY.wasSet;

    const bool nonSymetric = (par.covMode == Parameters::COV_MODE_TARGET || par.covMode == Parameters::COV_MODE_QUERY);
    if (clusterModeSet == false){
        if (nonSymetric) {
            par.clusteringMode = Parameters::GREEDY_MEM;
        } else {
            par.clusteringMode = Parameters::SET_COVER;
        }
        std::string cluMode = (par.clusteringMode==Parameters::GREEDY_MEM) ? "GREEDY MEM" : "SET COVER";
        Debug(Debug::INFO) << "Set cluster mode " << cluMode << ".\n";
    }

    if (includeCountTableSet == false) {
        if (nonSymetric) {
            par.includeCountTable = false;
            par.countTableIteration = 0;
        } else {
            par.includeCountTable = true;
        }
    }
    Util::resolveIncludeIterationPair(par.PARAM_INCLUDE_COUNTTABLE.wasSet, par.includeCountTable,
                                par.PARAM_NUM_COUNTS.wasSet, par.countTableIteration,
                                "--include-count-table", "--num-count-table");

    if (includeAdjacencySet == false) {
        par.adjIteration = nonSymetric ? Parameters::CLUST_LINEAR_DEFAULT_NUM_ADJACENCY
                                       : Parameters::CLUST_LINEAR_SYMMETRIC_NUM_ADJACENCY;
    }
    Util::resolveIncludeIterationPair(par.PARAM_INCLUDE_ADJACENCY.wasSet, par.includeAdjacency,
                                par.PARAM_NUM_ADJACENCY.wasSet, par.adjIteration,
                                "--include-adjacency", "--num-adjacency");

    if (kmerSizeWasSet == false) {
        par.kmerSize = Parameters::CLUST_LINEAR_DEFAULT_K;
    }
    if (alphabetSizeWasSet == false) {
        par.alphabetSize = MultiParam<NuclAA<int>>(NuclAA<int>(Parameters::CLUST_LINEAR_DEFAULT_ALPH_SIZE, 5));
    }

    const int dbType = FileUtil::parseDbType(par.db1.c_str());
    const bool isUngappedMode = par.alignmentMode == Parameters::ALIGNMENT_MODE_UNGAPPED;
    if (isUngappedMode && Parameters::isEqualDbtype(dbType, Parameters::DBTYPE_HMM_PROFILE)) {
        par.printUsageMessage(command, MMseqsParameter::COMMAND_ALIGN|MMseqsParameter::COMMAND_PREFILTER);
        Debug(Debug::ERROR) << "Cannot use ungapped alignment mode with profile databases.\n";
        EXIT(EXIT_FAILURE);
    }
    
    if (par.linclustVersion == 1) {
        cmd.addVariable("ALIGN_MODULE", isUngappedMode ? "rescorediagonal" : "align");
        // filter by diagonal in case of AA (do not filter for nucl, profiles, ...)
        cmd.addVariable("FILTER", Parameters::isEqualDbtype(dbType, Parameters::DBTYPE_AMINO_ACIDS) ? "1" : NULL);
        cmd.addVariable("KMERMATCHER_PAR", par.createParameterString(par.kmermatcher).c_str());
        cmd.addVariable("VERBOSITY", par.createParameterString(par.onlyverbosity).c_str());
        cmd.addVariable("VERBOSITYANDCOMPRESS", par.createParameterString(par.threadsandcompression).c_str());
        
        par.alphabetSize = alphabetSize;
        par.kmerSize = kmerSize;
        // # 2. Hamming distance pre-clustering
        par.rescoreMode = Parameters::RESCORE_MODE_HAMMING;
        par.filterHits = false;
        float prevSeqId = par.seqIdThr;
        // hamming distance does not work well with seq. id < 0.5 since it does not have an e-value criteria
        par.seqIdThr = std::max(0.5f, par.seqIdThr);
        // also coverage should not be under 0.5
        float prevCov = par.covThr;
        par.covThr = std::max(0.5f, par.covThr);
        cmd.addVariable("HAMMING_PAR", par.createParameterString(par.rescorediagonal).c_str());
        // set it back to old value
        par.covThr = prevCov;
        par.seqIdThr = prevSeqId;
        par.rescoreMode = Parameters::RESCORE_MODE_SUBSTITUTION;

        // # 3. Ungapped alignment filtering
        par.filterHits = true;
        cmd.addVariable("UNGAPPED_ALN_PAR", par.createParameterString(par.rescorediagonal).c_str());

        // # 4. Local gapped sequence alignment.
        if (isUngappedMode) {
            const int originalRescoreMode = par.rescoreMode;
            par.rescoreMode = Parameters::RESCORE_MODE_ALIGNMENT;
            cmd.addVariable("ALIGNMENT_PAR", par.createParameterString(par.rescorediagonal).c_str());
            par.rescoreMode = originalRescoreMode;
        } else {
            cmd.addVariable("ALIGNMENT_PAR", par.createParameterString(par.align).c_str());
        }
        // # 5. Clustering using greedy set cover.
        cmd.addVariable("CLUSTER_PAR", par.createParameterString(par.clust).c_str());

    } else if (par.linclustVersion == 2) {
        par.alphabetSize = alphabetSize;
        par.kmerSize = kmerSize;
        bool prevspacedKmer = par.spacedKmer;
        bool prevmaskMode = par.maskMode;
        MultiParam<NuclAA<float>> prevKmersPerSequenceScale = par.kmersPerSequenceScale;
        const int previousKmerMatcherMode = par.kmerMatcherMode;
        if (par.PARAM_KMERMATCHER_MODE.wasSet == false) {
            par.kmerMatcherMode = Parameters::KMERMATCHER_MODE_LOCAL;
        }
        par.spacedKmer = false;
        par.maskMode = false;
        cmd.addVariable("KMERMATCHER_PAR", par.createParameterString(par.kmermatcher).c_str());
        
        cmd.addVariable("VERBOSITY", par.createParameterString(par.onlyverbosity).c_str());
        cmd.addVariable("ALIGN2CLUST_PAR", par.createParameterString(par.align2clust).c_str());
        cmd.addVariable("REFINE_ROUND", par.linclust2Iter >= 2 ? "TRUE" : NULL);
        cmd.addVariable("CLUSTER_PAR", par.createParameterString(par.clust).c_str());
        
        par.spacedKmer = true;
        par.kmersPerSequenceScale = 0.1;
        cmd.addVariable("KMERMATCHER_PAR2", par.createParameterString(par.kmermatcher).c_str());
        
        par.spacedKmer = prevspacedKmer;
        par.maskMode = prevmaskMode;
        // the 0.1 above is intended for KMERMATCHER_PAR2 only, so it must not leak into later strings
        par.kmersPerSequenceScale = prevKmersPerSequenceScale;
        par.kmerMatcherMode = previousKmerMatcherMode;
    }
    float prevSeqId = par.seqIdThr;
    // # 0. clust hash
    par.seqIdThr = std::max(0.9f, par.seqIdThr);
    par.alphabetSize = MultiParam<NuclAA<int>>(NuclAA<int>(Parameters::CLUST_HASH_DEFAULT_ALPH_SIZE, 5));
    cmd.addVariable("CLUSTHASH", par.clustHash ? "TRUE" : NULL);
    cmd.addVariable("CLUSTHASHFAST_PAR", par.createParameterString(par.clusthashfast).c_str());
    par.seqIdThr = prevSeqId;
    par.alphabetSize = alphabetSize;

    cmd.addVariable("SWITCH_CONSENSUS_REP", par.switchConsensusRep ? "TRUE" : NULL);
    cmd.addVariable("KEEP_SWITCH_ALN", (par.switchConsensusRep && writeAlnFiles) ? "TRUE" : NULL);
    cmd.addVariable("PICKREP_PAR", par.createParameterString(par.pickrepprofile).c_str());

    std::string program = tmpDir + "/linclust.sh";
    FileUtil::writeFile(program, linclust_sh, linclust_sh_len);
    cmd.execProgram(program.c_str(), par.filenames);

    // Unreachable
    assert(false);
    return 0;
}
