#include "Parameters.h"
#include "FileUtil.h"
#include "CommandCaller.h"
#include "Debug.h"
#include "Util.h"

#include <cassert>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "batch_clustering.sh.h"

namespace {

void setBatchLinclustDefaults(Parameters *p) {
    p->spacedKmer = false;
    p->covThr = 0.8f;
    p->maskMode = 0;
    p->evalThr = 0.001;
    p->seqIdThr = 0.9f;
    p->alignmentMode = Parameters::ALIGNMENT_MODE_SCORE_COV_SEQID;
    p->linclustVersion = Parameters::LINCLUST_VERSION2;
    p->clustHash = false;
    p->createdbMode = Parameters::SEQUENCE_SPLIT_MODE_SOFT;   // soft-link: skip DB data copy; batch materializes single-line FASTA first
    p->removeTmpFiles = true;   // batch scale accumulates per-chunk tmp; clean by default
    p->batchDeleteSourceChunk = true;  // reclaim each chunk as it is consumed; disk is the bottleneck
}

int parseRequestedThreads(int argc, const char **argv) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--threads") == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return -1;
}

void restoreRequestedThreads(Parameters &par, int requestedThreads) {
    if (requestedThreads > 0 && par.PARAM_THREADS.wasSet) {
        par.threads = requestedThreads;
    }
}

int mergeDefaultThreads(const Parameters &par) {
    return (par.threads > 0) ? par.threads : 1;
}

int automaticMergeSplits(const Parameters &par) {
    int splits = mergeDefaultThreads(par);   // always >= 1
    if (splits > 256) {
        splits = 256;
    }
    return splits;
}

int automaticMergeSplitJobs(const Parameters &par, int splits) {
    int jobs = mergeDefaultThreads(par) / 8;
    if (jobs < 1) {
        jobs = 1;
    }
    if (jobs > 16) {
        jobs = 16;
    }
    if (jobs > splits) {
        jobs = splits;
    }
    return jobs;
}

void resolveBatchMergeParallelism(Parameters &par) {
    // default is 0, so "== 0" already covers "unset"; 0 means auto (explicit or defaulted)
    const bool autoSplits = (par.batchMergeSplits == 0);
    const bool autoJobs = (par.batchMergeSplitJobs == 0);

    if (autoSplits) {
        par.batchMergeSplits = automaticMergeSplits(par);
        if (autoJobs == false && par.batchMergeSplits < par.batchMergeSplitJobs) {
            par.batchMergeSplits = par.batchMergeSplitJobs;
        }
    }
    if (autoJobs) {
        par.batchMergeSplitJobs = automaticMergeSplitJobs(par, par.batchMergeSplits);
    }
    if (par.batchMergeSplitJobs > par.batchMergeSplits) {
        par.batchMergeSplitJobs = par.batchMergeSplits;
    }
}

void setBatchClusterDefaults(Parameters *p) {
    p->spacedKmer = true;
    p->covThr = 0.8f;
    p->evalThr = 0.001;
    p->alignmentMode = Parameters::ALIGNMENT_MODE_SCORE_COV_SEQID;
    p->maxResListLen = 20;
    p->clusterVersion = Parameters::CLUSTER_VERSION2;
    p->createdbMode = Parameters::SEQUENCE_SPLIT_MODE_SOFT;   // soft-link: skip DB data copy; batch materializes single-line FASTA first
    p->removeTmpFiles = true;   // batch scale accumulates per-chunk tmp; clean by default
    p->batchDeleteSourceChunk = true;  // reclaim each chunk as it is consumed; disk is the bottleneck
}

bool isNonSymmetricCovMode(const Parameters &par) {
    return (par.covMode == Parameters::COV_MODE_TARGET ||
            par.covMode == Parameters::COV_MODE_QUERY);
}

std::string trimString(const std::string &value) {
    size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin]))) {
        begin++;
    }
    size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(begin, end - begin);
}

void appendItemsFromList(std::vector<std::string> &items, const std::string &list) {
    size_t begin = 0;
    while (begin <= list.size()) {
        size_t end = list.find(',', begin);
        if (end == std::string::npos) {
            end = list.size();
        }
        std::string item = trimString(list.substr(begin, end - begin));
        if (item.empty() == false) {
            items.push_back(item);
        }
        if (end == list.size()) {
            break;
        }
        begin = end + 1;
    }
}

size_t getSlurmNodeCount(const Parameters &par) {
    std::vector<std::string> nodes;
    appendItemsFromList(nodes, par.batchSlurmNodelist);
    return nodes.size();
}

bool hasSlurmParameters(const Parameters &par) {
    return par.PARAM_BATCH_SLURM_NODELIST.wasSet ||
           par.PARAM_BATCH_SLURM_PARTITION.wasSet ||
           par.PARAM_BATCH_SLURM_TIME.wasSet ||
           par.PARAM_BATCH_SLURM_MEM.wasSet ||
           par.PARAM_BATCH_SLURM_EXTRA.wasSet ||
           par.PARAM_BATCH_ROUND0_SLURM_NODELIST.wasSet ||
           par.PARAM_BATCH_ROUND0_SLURM_PARTITION.wasSet ||
           par.PARAM_BATCH_ROUND0_SLURM_TIME.wasSet ||
           par.PARAM_BATCH_ROUND0_SLURM_MEM.wasSet ||
           par.PARAM_BATCH_ROUND0_SLURM_EXTRA.wasSet;
}

bool isS3Uri(const std::string &path) {
    return path.compare(0, 5, "s3://") == 0;
}

float batchClusterAutomaticSensitivity(float seqId) {
    if (seqId <= 0.3f) {
        return 6.0f;
    }
    if (seqId > 0.8f) {
        return 1.0f;
    }
    return 1.0f + (0.7f - seqId) * 10.0f;
}

int batchClusterAutomaticIterations(float sensitivity) {
    return (sensitivity <= 2.0f) ? 1 : 3;
}

void applyBatchLinclustAutomagic(Parameters &par) {
    const bool nonSymmetric = isNonSymmetricCovMode(par);

    if (par.PARAM_CLUSTER_MODE.wasSet == false) {
        par.clusteringMode = nonSymmetric ? Parameters::GREEDY_MEM : Parameters::SET_COVER;
    }

    const bool includeCountTableSet =
        (par.PARAM_INCLUDE_COUNTTABLE.wasSet || par.PARAM_NUM_COUNTS.wasSet);
    if (includeCountTableSet == false) {
        if (nonSymmetric) {
            par.includeCountTable = false;
            par.countTableIteration = 0;
        } else {
            par.includeCountTable = true;
        }
    }
    Util::resolveIncludeIterationPair(par.PARAM_INCLUDE_COUNTTABLE.wasSet, par.includeCountTable,
                                par.PARAM_NUM_COUNTS.wasSet, par.countTableIteration,
                                "--include-count-table", "--num-count-table");

    if ((par.PARAM_INCLUDE_ADJACENCY.wasSet || par.PARAM_NUM_ADJACENCY.wasSet) == false) {
        par.adjIteration = nonSymmetric ? Parameters::CLUST_LINEAR_DEFAULT_NUM_ADJACENCY
                                        : Parameters::CLUST_LINEAR_SYMMETRIC_NUM_ADJACENCY;
    }
    Util::resolveIncludeIterationPair(par.PARAM_INCLUDE_ADJACENCY.wasSet, par.includeAdjacency,
                                par.PARAM_NUM_ADJACENCY.wasSet, par.adjIteration,
                                "--include-adjacency", "--num-adjacency");
}

void applyBatchClusterAutomagic(Parameters &par) {
    if (par.PARAM_S.wasSet == false) {
        par.sensitivity = batchClusterAutomaticSensitivity(par.seqIdThr);
    }

    const bool nonSymmetric = isNonSymmetricCovMode(par);
    if (par.PARAM_CLUSTER_MODE.wasSet == false) {
        par.clusteringMode = nonSymmetric ? Parameters::GREEDY_MEM : Parameters::SET_COVER;
    }

    if (par.PARAM_INCLUDE_COUNTTABLE.wasSet == false) {
        par.includeCountTable = (nonSymmetric == false);
        if (par.includeCountTable == false) {
            par.countTableIteration = 0;
        }
    }
    Util::resolveIncludeIterationPair(par.PARAM_INCLUDE_COUNTTABLE.wasSet, par.includeCountTable,
                                par.PARAM_NUM_COUNTS.wasSet, par.countTableIteration,
                                "--include-count-table", "--num-count-table");

    Util::resolveIncludeIterationPair(par.PARAM_INCLUDE_ADJACENCY.wasSet, par.includeAdjacency,
                                par.PARAM_NUM_ADJACENCY.wasSet, par.adjIteration,
                                "--include-adjacency", "--num-adjacency");

    if (par.PARAM_CLUSTER_STEPS.wasSet == false) {
        par.clusterSteps = batchClusterAutomaticIterations(par.sensitivity);
    }
}

void validateBatchCreatedbMode(const Parameters &par) {
    if (par.createdbMode != Parameters::SEQUENCE_SPLIT_MODE_HARD &&
        par.createdbMode != Parameters::SEQUENCE_SPLIT_MODE_SOFT &&
        par.createdbMode != Parameters::SEQUENCE_SPLIT_MODE_LENGTH_DESC) {
        Debug(Debug::ERROR) << "Batch clustering supports --createdb-mode 0, 1 or 3 only. "
                            << "--createdb-mode " << par.createdbMode
                            << " changes the createdb layout and can change clustering results.\n";
        EXIT(EXIT_FAILURE);
    }
}

void validateBatchServerBackend(Parameters &par, const Command &command) {
    validateBatchCreatedbMode(par);

    if (par.batchBackend == "single-node") {
        if (isS3Uri(par.db2) || isS3Uri(par.db3)) {
            Debug(Debug::ERROR) << "--backend single-node requires local <resultDir> and <tmpDir>. Use mmseqs "
                                << command.cmd << "-aws for S3 paths.\n";
            EXIT(EXIT_FAILURE);
        }
        if (hasSlurmParameters(par)) {
            Debug(Debug::ERROR) << "--slurm-* parameters are only valid with --backend multi-node.\n";
            EXIT(EXIT_FAILURE);
        }
        return;
    }

    if (par.batchBackend == "multi-node") {
        if (isS3Uri(par.db2) || isS3Uri(par.db3)) {
            Debug(Debug::ERROR) << "--backend multi-node requires shared local <resultDir> and <tmpDir>. Use mmseqs "
                                << command.cmd << "-aws for S3 paths.\n";
            EXIT(EXIT_FAILURE);
        }
        size_t nodeCount = getSlurmNodeCount(par);
        if (nodeCount == 0) {
            Debug(Debug::ERROR) << "--backend multi-node requires --slurm-nodelist.\n";
            EXIT(EXIT_FAILURE);
        }
        if (par.batchNodeWorkDir.empty()) {
            Debug(Debug::ERROR) << "--backend multi-node requires --node-work-dir (a per-node LOCAL disk path, e.g. /mnt/scratch). Without it each chunk's createdb/cluster tmp is written under the shared <tmpDir>, hammering the shared filesystem.\n";
            EXIT(EXIT_FAILURE);
        }
        if (isS3Uri(par.batchNodeWorkDir)) {
            Debug(Debug::ERROR) << "--node-work-dir must be a LOCAL disk path, not an s3:// URI.\n";
            EXIT(EXIT_FAILURE);
        }
        if (par.batchRound0NodeWorkDir.empty() == false && isS3Uri(par.batchRound0NodeWorkDir)) {
            Debug(Debug::ERROR) << "--round0-node-work-dir must be a LOCAL disk path, not an s3:// URI.\n";
            EXIT(EXIT_FAILURE);
        }
        if (par.batchNodeWorkDir == par.db3 ||
            par.batchNodeWorkDir.compare(0, par.db3.size() + 1, par.db3 + "/") == 0) {
            Debug(Debug::WARNING) << "--node-work-dir is inside the shared <tmpDir> (" << par.db3
                                  << "); this defeats its purpose -- per-chunk tmp will still hit the shared filesystem. Point it at per-node local disk.\n";
        }
        return;
    }

    if (par.batchBackend == "aws-batch") {
        Debug(Debug::ERROR) << "The aws-batch backend moved to its own command. Use mmseqs "
                            << command.cmd << "-aws (same arguments, without --backend).\n";
        EXIT(EXIT_FAILURE);
    }

    Debug(Debug::ERROR) << "Invalid --backend " << par.batchBackend
                        << ". Valid values are single-node and multi-node; for AWS Batch use mmseqs "
                        << command.cmd << "-aws.\n";
    EXIT(EXIT_FAILURE);
}

void validateBatchAwsBackend(Parameters &par, const Command &command) {
    validateBatchCreatedbMode(par);

    if (par.batchNodeWorkDir.empty()) {
        Debug(Debug::ERROR) << "mmseqs " << command.cmd << " requires --node-work-dir (a container-local disk path). Without it each chunk's createdb/cluster tmp defaults to the container /tmp, which is often too small at batch scale.\n";
        EXIT(EXIT_FAILURE);
    }
    if (isS3Uri(par.batchNodeWorkDir)) {
        Debug(Debug::ERROR) << "--node-work-dir must be a container-LOCAL disk path, not an s3:// URI.\n";
        EXIT(EXIT_FAILURE);
    }
    if (par.batchRound0NodeWorkDir.empty() == false && isS3Uri(par.batchRound0NodeWorkDir)) {
        Debug(Debug::ERROR) << "--round0-node-work-dir must be a container-LOCAL disk path, not an s3:// URI.\n";
        EXIT(EXIT_FAILURE);
    }
}

std::string getBatchSubmitTmpBase() {
    const char *tmp = std::getenv("TMPDIR");
    std::string base = (tmp != NULL && tmp[0] != '\0') ? tmp : "/tmp";
    while (base.size() > 1 && base[base.size() - 1] == '/') {
        base.erase(base.size() - 1);
    }
    return base + "/mmseqs-batch-submit";
}

std::string createBatchSubmitDirectory(const std::string &hash) {
    std::string base = getBatchSubmitTmpBase();
    if (FileUtil::directoryExists(base.c_str()) == false &&
        FileUtil::makeDir(base.c_str()) == false) {
        Debug(Debug::ERROR) << "Cannot create temporary folder " << base << ".\n";
        EXIT(EXIT_FAILURE);
    }

    std::string dir = base + "/" + hash;
    if (FileUtil::directoryExists(dir.c_str()) == false &&
        FileUtil::makeDir(dir.c_str()) == false) {
        Debug(Debug::ERROR) << "Cannot create temporary subfolder " << dir << ".\n";
        EXIT(EXIT_FAILURE);
    }
    return dir;
}

std::string buildCreatedbPar(const Parameters &par) {
    // the per-chunk .lookup is unused, and shuffling would renumber ids and change tie-breaks
    std::string createdbPar = std::string("--shuffle 0 --write-lookup 0") +
           " --createdb-mode " + SSTR(par.createdbMode) +
           " --threads " + SSTR(par.threads) + " -v " + SSTR(par.verbosity);
    // only when set, so an absent flag keeps createdb's own default exactly as before
    if (par.PARAM_SHUFFLE_SPLITS.wasSet) {
        createdbPar += " --shuffle-splits " + SSTR(par.shuffleSplits);
    }
    return createdbPar;
}

std::string buildCreatetsvPar(const Parameters &par) {
    return "--threads " + SSTR(par.threads) + " -v " + SSTR(par.verbosity);
}

const std::vector<MMseqsParameter*>& innerClusterParameters(Parameters &par,
                                                           const std::string &clusterCmd) {
    return (clusterCmd == "linclust") ? par.linclustbatchinner : par.clusterbatchinner;
}

std::string buildInnerClusterParFromCurrent(Parameters &par, const std::string &clusterCmd) {
    const int inheritedKmerMatcherMode = par.kmerMatcherMode;
    if (par.PARAM_KMERMATCHER_MODE.wasSet == false
        && par.linclustVersion == Parameters::LINCLUST_VERSION2) {
        par.kmerMatcherMode = Parameters::KMERMATCHER_MODE_LOCAL;
    }
    const std::string parameters = par.createParameterString(innerClusterParameters(par, clusterCmd));
    par.kmerMatcherMode = inheritedKmerMatcherMode;
    return parameters;
}

std::string buildInnerClusterPar(Parameters &par, const std::string &clusterCmd) {
    if (clusterCmd == "linclust") {
        applyBatchLinclustAutomagic(par);
    } else {
        applyBatchClusterAutomagic(par);
    }
    return buildInnerClusterParFromCurrent(par, clusterCmd);
}

void applyRound0ClusterDefaults(Parameters &par, const std::string &clusterCmd) {
    // round0 only reduces redundancy, so keep it light unless the user overrides it
    par.kmersPerSequence = 21;
    par.includeCountTable = false;
    par.countTableIteration = 0;
    par.includeAdjacency = false;
    par.adjIteration = 0;
    if (clusterCmd == "linclust") {
        par.clustHash = false;
    }
}

// explicit --round0-* flags only, never automagic state; contradictory toggle and count is an error
void resolveRound0InclNum(bool includeSet, bool includeValue, bool numSet, int numValue,
                          int parameterDefault, const std::string &what,
                          bool &includeOut, int &iterationOut) {
    if (includeSet && numSet) {
        if (includeValue && numValue == 0) {
            Debug(Debug::ERROR) << "--round0-include-" << what << " 1 conflicts with --round0-num-"
                                << what << " 0. Enable with a positive count, or disable both.\n";
            EXIT(EXIT_FAILURE);
        }
        if (includeValue == false && numValue > 0) {
            Debug(Debug::ERROR) << "--round0-include-" << what << " 0 conflicts with --round0-num-"
                                << what << " " << numValue << ". Disable with count 0, or enable both.\n";
            EXIT(EXIT_FAILURE);
        }
        includeOut = includeValue;
        iterationOut = numValue;
        return;
    }
    if (includeSet) {
        includeOut = includeValue;
        iterationOut = includeValue ? parameterDefault : 0;
        return;
    }
    if (numSet) {
        iterationOut = numValue;
        includeOut = (numValue > 0);
    }
}

void applyRound0ClusterOverrides(Parameters &par) {
    if (par.PARAM_BATCH_ROUND0_MIN_SEQ_ID.wasSet) {
        par.seqIdThr = par.batchRound0SeqIdThr;
    }
    if (par.PARAM_BATCH_ROUND0_C.wasSet) {
        par.covThr = par.batchRound0CovThr;
    }
    if (par.PARAM_BATCH_ROUND0_COV_MODE.wasSet) {
        par.covMode = par.batchRound0CovMode;
    }
    if (par.PARAM_BATCH_ROUND0_CLUSTER_MODE.wasSet) {
        par.clusteringMode = par.batchRound0ClusteringMode;
    }
    if (par.PARAM_BATCH_ROUND0_KMER_PER_SEQ.wasSet) {
        par.kmersPerSequence = par.batchRound0KmersPerSequence;
    }
    const bool nonSymmetric = isNonSymmetricCovMode(par);
    resolveRound0InclNum(par.PARAM_BATCH_ROUND0_INCLUDE_COUNTTABLE.wasSet, par.batchRound0IncludeCountTable,
                         par.PARAM_BATCH_ROUND0_NUM_COUNTS.wasSet, par.batchRound0CountTableIteration,
                         Parameters::CLUST_LINEAR_DEFAULT_NUM_COUNT_TABLE, "count-table",
                         par.includeCountTable, par.countTableIteration);
    resolveRound0InclNum(par.PARAM_BATCH_ROUND0_INCLUDE_ADJACENCY.wasSet, par.batchRound0IncludeAdjacency,
                         par.PARAM_BATCH_ROUND0_NUM_ADJACENCY.wasSet, par.batchRound0AdjIteration,
                         nonSymmetric ? Parameters::CLUST_LINEAR_DEFAULT_NUM_ADJACENCY
                                      : Parameters::CLUST_LINEAR_SYMMETRIC_NUM_ADJACENCY,
                         "adjacency", par.includeAdjacency, par.adjIteration);
    if (par.PARAM_BATCH_ROUND0_CLUST_HASH.wasSet) {
        par.clustHash = par.batchRound0ClustHash;
    }
    if (par.PARAM_BATCH_ROUND0_SPLIT_MEMORY_LIMIT.wasSet) {
        par.splitMemoryLimit = par.batchRound0SplitMemoryLimit;
    }
    if (par.PARAM_BATCH_ROUND0_PRELOAD_MODE.wasSet) {
        par.preloadMode = par.batchRound0PreloadMode;
    }
    if (par.PARAM_BATCH_ROUND0_LINCLUST2_ITER.wasSet) {
        par.linclust2Iter = par.batchRound0Linclust2Iter;
    }
}

void applyBatchClusterAutomagic(Parameters &par, const std::string &clusterCmd) {
    if (clusterCmd == "linclust") {
        applyBatchLinclustAutomagic(par);
    } else {
        applyBatchClusterAutomagic(par);
    }
}

std::string buildRound0ClusterPar(Parameters &par, const std::string &clusterCmd, bool forceClustHash = false) {
    const float prevSeqIdThr = par.seqIdThr;
    const float prevCovThr = par.covThr;
    const int prevCovMode = par.covMode;
    const int prevClusteringMode = par.clusteringMode;
    const int prevKmersPerSequence = par.kmersPerSequence;
    const bool prevIncludeCountTable = par.includeCountTable;
    const int prevCountTableIteration = par.countTableIteration;
    const bool prevIncludeAdjacency = par.includeAdjacency;
    const int prevAdjIteration = par.adjIteration;
    const bool prevClustHash = par.clustHash;
    const size_t prevSplitMemoryLimit = par.splitMemoryLimit;
    const int prevPreloadMode = par.preloadMode;
    const float prevSensitivity = par.sensitivity;
    const int prevClusterSteps = par.clusterSteps;

    // explicit values first so automagic derives from them, then defaults and overrides again so the user still wins
    applyRound0ClusterOverrides(par);
    applyBatchClusterAutomagic(par, clusterCmd);
    applyRound0ClusterDefaults(par, clusterCmd);
    applyRound0ClusterOverrides(par);
    if (forceClustHash) {
        par.clustHash = true;
    }

    std::string round0Par = buildInnerClusterParFromCurrent(par, clusterCmd);

    par.seqIdThr = prevSeqIdThr;
    par.covThr = prevCovThr;
    par.covMode = prevCovMode;
    par.clusteringMode = prevClusteringMode;
    par.kmersPerSequence = prevKmersPerSequence;
    par.includeCountTable = prevIncludeCountTable;
    par.countTableIteration = prevCountTableIteration;
    par.includeAdjacency = prevIncludeAdjacency;
    par.adjIteration = prevAdjIteration;
    par.clustHash = prevClustHash;
    par.splitMemoryLimit = prevSplitMemoryLimit;
    par.preloadMode = prevPreloadMode;
    par.sensitivity = prevSensitivity;
    par.clusterSteps = prevClusterSteps;

    return round0Par;
}

void addBatchEngineVariables(CommandCaller &cmd, const Parameters &par,
                             const std::string &clusterCmd,
                             const std::string &clusterPar,
                             const std::string &round0ClusterPar) {
    const std::string threads = SSTR(par.threads);
    const std::string chunkMaxBytes = SSTR(par.batchChunkMaxBytes);
    const std::string chunkMaxSeqs = SSTR(par.batchChunkMaxSeqs);
    const std::string round0ChunkMaxBytes = SSTR(par.batchRound0ChunkMaxBytes);
    const std::string round0ChunkMaxSeqs = SSTR(par.batchRound0ChunkMaxSeqs);
    const std::string chunkDiskBudget = SSTR(par.batchChunkDiskBudget);
    const std::string round0ChunkDiskBudget = SSTR(par.batchRound0ChunkDiskBudget);
    const std::string maxRounds = SSTR(par.batchMaxRounds);
    const std::string minReductionRatio = SSTR(par.batchMinReductionRatio);
    const std::string convergencePatience = SSTR(par.batchConvergencePatience);
    const std::string minReductionCount = SSTR(par.batchMinReductionCount);
    const std::string maxChunkAttempts = SSTR(par.batchMaxChunkAttempts);
    const std::string compressBatchOutputs = par.batchCompressOutputs ? "1" : "0";
    const std::string mergeSplits = SSTR(par.batchMergeSplits);
    const std::string mergeSplitJobs = SSTR(par.batchMergeSplitJobs);
    const std::string createdbPar = buildCreatedbPar(par);
    const std::string createtsvPar = buildCreatetsvPar(par);

    cmd.addVariable("CLUSTER_CMD", clusterCmd.c_str());
    cmd.addVariable("ROUND0_CLUSTER_CMD", clusterCmd == "cluster" ? "linclust" : NULL);
    cmd.addVariable("CLUSTER_PAR", clusterPar.c_str());
    cmd.addVariable("ROUND0_CLUSTER_PAR", round0ClusterPar.empty() ? NULL : round0ClusterPar.c_str());
    cmd.addVariable("CREATEDB_PAR", createdbPar.c_str());
    cmd.addVariable("CREATETSV_PAR", createtsvPar.c_str());
    cmd.addVariable("THREADS", threads.c_str());
    cmd.addVariable("CHUNK_MAX_BYTES", chunkMaxBytes.c_str());
    cmd.addVariable("CHUNK_MAX_SEQS", chunkMaxSeqs.c_str());
    // round_cluster_par picks the spill per round; an explicit --kmer-write-to-disk must win
    cmd.addVariable("BATCH_KMER_WRITE_TO_DISK", par.PARAM_KMER_WRITE_TO_DISK.wasSet ? (par.kmerWriteToDisk ? "1" : "0") : NULL);
    cmd.addVariable("ROUND0_CHUNK_MAX_BYTES", par.PARAM_BATCH_ROUND0_CHUNK_MAX_BYTES.wasSet ? round0ChunkMaxBytes.c_str() : NULL);
    cmd.addVariable("ROUND0_CHUNK_MAX_SEQS", par.PARAM_BATCH_ROUND0_CHUNK_MAX_SEQS.wasSet ? round0ChunkMaxSeqs.c_str() : NULL);
    cmd.addVariable("CHUNK_DISK_BUDGET", chunkDiskBudget.c_str());
    cmd.addVariable("ROUND0_CHUNK_DISK_BUDGET", par.PARAM_BATCH_ROUND0_CHUNK_DISK_BUDGET.wasSet ? round0ChunkDiskBudget.c_str() : NULL);
    cmd.addVariable("DISK_POLL_SEC", SSTR(par.batchDiskPollInterval).c_str());
    cmd.addVariable("RAM_POLL_SEC", SSTR(par.batchRamPollInterval).c_str());
    cmd.addVariable("MERGE_SPLITS", mergeSplits.c_str());
    cmd.addVariable("MERGE_SPLIT_JOBS", mergeSplitJobs.c_str());
    cmd.addVariable("BATCH_REP_FASTA_SPLITS", SSTR(par.batchRepFastaSplits).c_str());
    cmd.addVariable("ROUND0_THREADS", par.PARAM_BATCH_ROUND0_THREADS.wasSet ? SSTR(par.batchRound0Threads).c_str() : NULL);
    cmd.addVariable("CREATEDB_SHUFFLE_SPLITS", par.PARAM_SHUFFLE_SPLITS.wasSet ? SSTR(par.shuffleSplits).c_str() : NULL);
    cmd.addVariable("ROUND0_CREATEDB_SHUFFLE_SPLITS", par.PARAM_BATCH_ROUND0_SHUFFLE_SPLITS.wasSet ? SSTR(par.batchRound0ShuffleSplits).c_str() : NULL);
    cmd.addVariable("ROUND0_BATCH_REP_FASTA_SPLITS", par.PARAM_BATCH_ROUND0_REP_FASTA_SPLITS.wasSet ? SSTR(par.batchRound0RepFastaSplits).c_str() : NULL);
    cmd.addVariable("MAX_ROUNDS", maxRounds.c_str());
    cmd.addVariable("MIN_REDUCTION_RATIO", minReductionRatio.c_str());
    cmd.addVariable("CONVERGENCE_PATIENCE", convergencePatience.c_str());
    cmd.addVariable("MIN_REDUCTION_COUNT", minReductionCount.c_str());
    cmd.addVariable("MAX_CHUNK_ATTEMPTS", maxChunkAttempts.c_str());
    cmd.addVariable("COMPRESS_BATCH_OUTPUTS", compressBatchOutputs.c_str());
    cmd.addVariable("BATCH_BACKEND", par.batchBackend.c_str());
    cmd.addVariable("REMOVE_TMP", par.removeTmpFiles ? "TRUE" : NULL);
    cmd.addVariable("BATCH_SLURM_NODELIST", par.batchSlurmNodelist.empty() ? NULL : par.batchSlurmNodelist.c_str());
    cmd.addVariable("BATCH_SLURM_PARTITION", par.batchSlurmPartition.empty() ? NULL : par.batchSlurmPartition.c_str());
    cmd.addVariable("BATCH_SLURM_TIME", par.batchSlurmTime.empty() ? NULL : par.batchSlurmTime.c_str());
    cmd.addVariable("BATCH_SLURM_MEM", par.batchSlurmMem.empty() ? NULL : par.batchSlurmMem.c_str());
    cmd.addVariable("BATCH_SLURM_EXTRA", par.batchSlurmExtra.empty() ? NULL : par.batchSlurmExtra.c_str());
    cmd.addVariable("NODE_WORK_DIR", par.batchNodeWorkDir.empty() ? NULL : par.batchNodeWorkDir.c_str());
    cmd.addVariable("ROUND0_BATCH_SLURM_NODELIST", par.batchRound0SlurmNodelist.empty() ? NULL : par.batchRound0SlurmNodelist.c_str());
    cmd.addVariable("ROUND0_BATCH_SLURM_PARTITION", par.batchRound0SlurmPartition.empty() ? NULL : par.batchRound0SlurmPartition.c_str());
    cmd.addVariable("ROUND0_BATCH_SLURM_TIME", par.batchRound0SlurmTime.empty() ? NULL : par.batchRound0SlurmTime.c_str());
    cmd.addVariable("ROUND0_BATCH_SLURM_MEM", par.batchRound0SlurmMem.empty() ? NULL : par.batchRound0SlurmMem.c_str());
    cmd.addVariable("ROUND0_BATCH_SLURM_EXTRA", par.batchRound0SlurmExtra.empty() ? NULL : par.batchRound0SlurmExtra.c_str());
    cmd.addVariable("ROUND0_NODE_WORK_DIR", par.batchRound0NodeWorkDir.empty() ? NULL : par.batchRound0NodeWorkDir.c_str());
    if (par.PARAM_BATCH_AWS_MACHINE.wasSet) {
        cmd.addVariable("BATCH_AWS_MACHINE", par.batchAwsMachine.c_str());
    }
    if (par.PARAM_BATCH_AWS_JOB_QUEUE.wasSet) {
        cmd.addVariable("BATCH_AWS_JOB_QUEUE", par.batchAwsJobQueue.c_str());
    }
    if (par.PARAM_BATCH_AWS_JOB_DEFINITION.wasSet) {
        cmd.addVariable("BATCH_AWS_JOB_DEFINITION", par.batchAwsJobDefinition.c_str());
    }
    if (par.PARAM_BATCH_ROUND0_AWS_MACHINE.wasSet) {
        cmd.addVariable("ROUND0_BATCH_AWS_MACHINE", par.batchRound0AwsMachine.c_str());
    }
    if (par.PARAM_BATCH_ROUND0_AWS_JOB_QUEUE.wasSet) {
        cmd.addVariable("ROUND0_BATCH_AWS_JOB_QUEUE", par.batchRound0AwsJobQueue.c_str());
    }
    if (par.PARAM_BATCH_ROUND0_AWS_JOB_DEFINITION.wasSet) {
        cmd.addVariable("ROUND0_BATCH_AWS_JOB_DEFINITION", par.batchRound0AwsJobDefinition.c_str());
    }
    if (par.PARAM_BATCH_AWS_MACHINE_TAG_KEY.wasSet) {
        cmd.addVariable("BATCH_AWS_MACHINE_TAG_KEY", par.batchAwsMachineTagKey.c_str());
    }

    // former environment-only knobs; empty keeps the shell's derive-it path
    cmd.addVariable("ROUND0_MMSEQS", par.batchRound0Mmseqs.c_str());
    cmd.addVariable("ROUND0_CREATEDB_MODE", SSTR(par.batchRound0CreatedbMode).c_str());
    cmd.addVariable("COMPRESS_RATIO", SSTR(par.batchCompressRatio).c_str());
    cmd.addVariable("BATCH_DELETE_SOURCE_CHUNK", par.batchDeleteSourceChunk ? "1" : "0");
    cmd.addVariable("SORT_TMP", par.batchSortTmpDir.c_str());
    // the shell tests for unset to derive it, so only export a real override
    if (par.batchSortBufferSize.empty() == false) {
        cmd.addVariable("SORT_BUFFER_SIZE", par.batchSortBufferSize.c_str());
    }
    cmd.addVariable("BATCH_AWS_MMSEQS", par.batchAwsMmseqs.c_str());
    cmd.addVariable("ROUND0_BATCH_AWS_MMSEQS", par.batchRound0AwsMmseqs.c_str());
    cmd.addVariable("BATCH_AWS_JOB_PREFIX", par.batchAwsJobPrefix.c_str());
    cmd.addVariable("BATCH_AWS_LOCAL_DIR", par.batchAwsLocalDir.c_str());
    cmd.addVariable("BATCH_AWS_TIMEOUT", SSTR(par.batchAwsTimeout).c_str());
    cmd.addVariable("BATCH_AWS_ALLOW_NONS3_INPUT", par.batchAwsAllowNonS3Input ? "1" : "0");
    cmd.addVariable("BATCH_AWS_DRY_RUN", par.batchAwsDryRun ? "1" : NULL);
    // derived from the work prefix when empty
    if (par.batchAwsScriptUri.empty() == false) {
        cmd.addVariable("BATCH_AWS_SCRIPT_URI", par.batchAwsScriptUri.c_str());
    }
    if (par.batchAwsChunkPrefix.empty() == false) {
        cmd.addVariable("S3_CHUNK_PREFIX", par.batchAwsChunkPrefix.c_str());
    }
    cmd.addVariable("BATCH_AWS_WORKER_ATTEMPTS",
                    par.batchAwsWorkerAttempts > 0 ? SSTR(par.batchAwsWorkerAttempts).c_str() : NULL);
}

int execBatchEngine(Parameters &par, const std::string &programDir,
                    const std::string &mode, const std::vector<std::string> &modeArgs,
                    const std::string &clusterCmd, const std::string &clusterPar,
                    const std::string &round0ClusterPar) {
    resolveBatchMergeParallelism(par);
    CommandCaller cmd;
    addBatchEngineVariables(cmd, par, clusterCmd, clusterPar, round0ClusterPar);
    if (mode == "cluster-chunk") {
        cmd.addVariable("BATCH_WORKER_DISPATCH", "1");
    }

    if (FileUtil::directoryExists(programDir.c_str()) == false &&
        FileUtil::makeDir(programDir.c_str()) == false) {
        Debug(Debug::ERROR) << "Cannot create temporary folder " << programDir << ".\n";
        EXIT(EXIT_FAILURE);
    }

    std::string program = programDir + "/batch_clustering.sh";
    FileUtil::writeFile(program, batch_clustering_sh, batch_clustering_sh_len);

    std::vector<std::string> args;
    args.push_back(mode);
    args.insert(args.end(), modeArgs.begin(), modeArgs.end());
    cmd.execProgram(program.c_str(), args);

    assert(false);
    return 0;
}

// several inputs are written into one manifest, so the engine keeps its single-manifest contract
std::string resolveBatchInputManifest(const Parameters &par, const std::string &tmpDir) {
    if (par.filenames.size() < 2) {
        return par.db1;
    }
    std::string content;
    for (size_t i = 0; i < par.filenames.size(); ++i) {
        content.append(par.filenames[i]);
        content.append("\n");
    }
    std::string manifest = tmpDir + "/input.manifest";
    FileUtil::writeFile(manifest, reinterpret_cast<const unsigned char *>(content.c_str()), content.size());
    Debug(Debug::INFO) << "Input: " << par.filenames.size() << " file(s) written to " << manifest << "\n";
    return manifest;
}

std::string createBatchSharedTmp(Parameters &par, const Command &command,
                                 const std::vector<MMseqsParameter*> &paramList) {
    std::string tmpDir = par.db3;
    std::string hash = SSTR(par.hashParameter(command.databases, par.filenames, paramList));
    if (par.reuseLatest) {
        hash = FileUtil::getHashFromSymLink(tmpDir + "/latest");
    }
    return FileUtil::createTemporaryDirectory(tmpDir, hash);
}

int runBatchSingleNode(Parameters &par, const Command &command,
                       const std::vector<MMseqsParameter*> &paramList,
                       const std::string &clusterCmd, const std::string &clusterPar,
                       const std::string &round0ClusterPar) {
    std::string tmpDir = createBatchSharedTmp(par, command, paramList);

    std::vector<std::string> args;
    args.push_back(resolveBatchInputManifest(par, tmpDir));
    args.push_back(tmpDir);
    args.push_back(par.db2);
    return execBatchEngine(par, tmpDir, "run-single-node", args, clusterCmd, clusterPar, round0ClusterPar);
}

int runBatchMultiNode(Parameters &par, const Command &command,
                      const std::vector<MMseqsParameter*> &paramList,
                      const std::string &clusterCmd, const std::string &clusterPar,
                      const std::string &round0ClusterPar) {
    std::string tmpDir = createBatchSharedTmp(par, command, paramList);

    std::vector<std::string> args;
    args.push_back(resolveBatchInputManifest(par, tmpDir));
    args.push_back(tmpDir);
    args.push_back(par.db2);
    return execBatchEngine(par, tmpDir, "run-multi-node", args, clusterCmd, clusterPar, round0ClusterPar);
}

int runBatchAwsBatch(Parameters &par, const Command &command,
                     const std::vector<MMseqsParameter*> &paramList,
                     const std::string &clusterCmd, const std::string &clusterPar,
                     const std::string &round0ClusterPar) {
    if (isS3Uri(par.db2) == false || isS3Uri(par.db3) == false) {
        Debug(Debug::ERROR)
            << "mmseqs " << command.cmd << " requires S3 prefixes for <resultDir> and <tmpDir>.\n"
            << "Example: mmseqs " << command.cmd
            << " s3://bucket/input.manifest s3://bucket/results/run1 s3://bucket/work/run1\n";
        EXIT(EXIT_FAILURE);
    }

    std::string hash = SSTR(par.hashParameter(command.databases, par.filenames, paramList));
    std::string programDir = createBatchSubmitDirectory(hash);

    std::vector<std::string> args;
    args.push_back(par.db1);
    args.push_back(par.db3);
    args.push_back(par.db2);
    return execBatchEngine(par, programDir, "aws-submit", args, clusterCmd, clusterPar, round0ClusterPar);
}

int runBatchClustering(Parameters &par, const Command &command,
                       const std::vector<MMseqsParameter*> &paramList,
                       const std::string &clusterCmd, const std::string &clusterPar,
                       const std::string &round0ClusterPar) {
    if (par.removeTmpFiles == false) {
        Debug(Debug::WARNING) << "--remove-tmp-files 0: per-chunk temporary files are KEPT and "
                                 "accumulate across all chunks and rounds. At batch scale this can "
                                 "fill the working disk. Use only for debugging a small input.\n";
    }
    if (par.batchBackend == "aws-batch") {
        return runBatchAwsBatch(par, command, paramList, clusterCmd, clusterPar, round0ClusterPar);
    }
    if (par.batchBackend == "multi-node") {
        return runBatchMultiNode(par, command, paramList, clusterCmd, clusterPar, round0ClusterPar);
    }

    return runBatchSingleNode(par, command, paramList, clusterCmd, clusterPar, round0ClusterPar);
}

void setBatchClusteringDescriptions(Parameters &par) {
    par.overrideParameterDescription(
        par.PARAM_THREADS,
        "CPU threads per chunk task. In multi-node mode this is the per-node CPU request",
        NULL,
        par.PARAM_THREADS.category);
    par.overrideParameterDescription(
        par.PARAM_CREATEDB_MODE,
        "Batch createdb mode: 0 copies FASTA/.zst into a compact MMseqs DB, 1 soft-links plain single-line FASTA, 3 copies it ordered by descending length. Mode 2/GPU DB layout is not supported by batch clustering",
        NULL,
        par.PARAM_CREATEDB_MODE.category);
    par.overrideParameterDescription(
        par.PARAM_COMPRESS_KMER_TMP_FILES,
        "Compress kmermatcher spill files during inner linclust/cluster. Separate from --compressed and from --compress-batch-outputs",
        NULL,
        par.PARAM_COMPRESS_KMER_TMP_FILES.category);
    par.overrideParameterDescription(
        par.PARAM_BATCH_COMPRESS_OUTPUTS,
        "Compress per-round and final batch FASTA/TSV shard files as .zst. This does not compress createdb sequence DBs",
        NULL,
        par.PARAM_BATCH_COMPRESS_OUTPUTS.category);
}

void setBatchAwsDescriptions(Parameters &par) {
    par.overrideParameterDescription(
        par.PARAM_BATCH_NODE_WORK_DIR,
        "Container-LOCAL disk for chunk tasks (createdb/cluster tmp + sort spill). REQUIRED; mount a real volume at this path in the job definition",
        NULL,
        par.PARAM_BATCH_NODE_WORK_DIR.category);
}

} // namespace

int linclustbatch(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchLinclustDefaults(&par);
    setBatchClusteringDescriptions(par);
    par.parseParameters(argc, argv, command, false, Parameters::PARSE_VARIADIC, 0);
    // the last two arguments are always the result dir and the shared tmp dir; the rest are inputs
    par.db3 = par.filenames.back(); par.filenames.pop_back();
    par.db2 = par.filenames.back(); par.filenames.pop_back();
    par.db1 = par.filenames.front();
    restoreRequestedThreads(par, requestedThreads);
    resolveBatchMergeParallelism(par);
    validateBatchServerBackend(par, command);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust");
    std::string clusterPar = buildInnerClusterPar(par, "linclust");
    par.printParameters(command.cmd, argc, argv, *command.params);
    return runBatchClustering(par, command, par.linclustbatch, "linclust", clusterPar, round0ClusterPar);
}

int linclustbatchaws(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchLinclustDefaults(&par);
    setBatchClusteringDescriptions(par);
    setBatchAwsDescriptions(par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
    resolveBatchMergeParallelism(par);
    par.batchBackend = "aws-batch";
    validateBatchAwsBackend(par, command);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust");
    std::string clusterPar = buildInnerClusterPar(par, "linclust");
    par.printParameters(command.cmd, argc, argv, *command.params);
    return runBatchClustering(par, command, par.linclustbatchaws, "linclust", clusterPar, round0ClusterPar);
}

int clusterbatch(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchClusterDefaults(&par);
    setBatchClusteringDescriptions(par);
    par.parseParameters(argc, argv, command, false, Parameters::PARSE_VARIADIC, 0);
    // the last two arguments are always the result dir and the shared tmp dir; the rest are inputs
    par.db3 = par.filenames.back(); par.filenames.pop_back();
    par.db2 = par.filenames.back(); par.filenames.pop_back();
    par.db1 = par.filenames.front();
    restoreRequestedThreads(par, requestedThreads);
    resolveBatchMergeParallelism(par);
    validateBatchServerBackend(par, command);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust", true);
    std::string clusterPar = buildInnerClusterPar(par, "cluster");
    par.printParameters(command.cmd, argc, argv, *command.params);
    return runBatchClustering(par, command, par.clusterbatch, "cluster", clusterPar, round0ClusterPar);
}

int clusterbatchaws(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchClusterDefaults(&par);
    setBatchClusteringDescriptions(par);
    setBatchAwsDescriptions(par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
    resolveBatchMergeParallelism(par);
    par.batchBackend = "aws-batch";
    validateBatchAwsBackend(par, command);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust", true);
    std::string clusterPar = buildInnerClusterPar(par, "cluster");
    par.printParameters(command.cmd, argc, argv, *command.params);
    return runBatchClustering(par, command, par.clusterbatchaws, "cluster", clusterPar, round0ClusterPar);
}

int linclustbatchworker(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchLinclustDefaults(&par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
    resolveBatchMergeParallelism(par);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust");
    std::string clusterPar = buildInnerClusterPar(par, "linclust");
    par.printParameters(command.cmd, argc, argv, *command.params);

    std::vector<std::string> args;
    args.push_back(par.db1);
    args.push_back(par.db2);
    args.push_back(par.db3);
    return execBatchEngine(par, par.db3, "cluster-chunk", args, "linclust", clusterPar, round0ClusterPar);
}

int clusterbatchworker(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchClusterDefaults(&par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
    resolveBatchMergeParallelism(par);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust", true);
    std::string clusterPar = buildInnerClusterPar(par, "cluster");
    par.printParameters(command.cmd, argc, argv, *command.params);

    std::vector<std::string> args;
    args.push_back(par.db1);
    args.push_back(par.db2);
    args.push_back(par.db3);
    return execBatchEngine(par, par.db3, "cluster-chunk", args, "cluster", clusterPar, round0ClusterPar);
}

int batchclusteringprepare(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchLinclustDefaults(&par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
    resolveBatchMergeParallelism(par);
    par.printParameters(command.cmd, argc, argv, *command.params);

    std::vector<std::string> args;
    args.push_back(par.db1);
    args.push_back(par.db2);
    args.push_back(par.db3);
    return execBatchEngine(par, FileUtil::dirName(par.db3), "prepare", args, "linclust", "", "");
}

int batchclusteringmerge(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchLinclustDefaults(&par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
    resolveBatchMergeParallelism(par);
    par.printParameters(command.cmd, argc, argv, *command.params);

    std::vector<std::string> args;
    args.push_back(par.db1);
    args.push_back(par.db2);
    args.push_back(par.db3);
    args.push_back(par.db4);
    return execBatchEngine(par, par.db4, "propagate", args, "linclust", "", "");
}
