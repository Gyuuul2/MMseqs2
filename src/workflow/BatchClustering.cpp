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
    p->createdbMode = Parameters::SEQUENCE_SPLIT_MODE_HARD;
    p->removeTmpFiles = true;   // batch scale accumulates per-chunk tmp; clean by default
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

void setBatchClusterDefaults(Parameters *p) {
    p->spacedKmer = true;
    p->covThr = 0.8f;
    p->evalThr = 0.001;
    p->alignmentMode = Parameters::ALIGNMENT_MODE_SCORE_COV_SEQID;
    p->maxResListLen = 20;
    p->clusterVersion = Parameters::CLUSTER_VERSION1;
    p->createdbMode = Parameters::SEQUENCE_SPLIT_MODE_HARD;
    p->removeTmpFiles = true;   // batch scale accumulates per-chunk tmp; clean by default
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

    if (par.PARAM_NUM_ADJACENCY.wasSet && par.adjIteration == 0 &&
        par.PARAM_INCLUDE_ADJACENCY.wasSet == false) {
        par.includeAdjacency = false;
    }
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
    }

    if (par.PARAM_NUM_ADJACENCY.wasSet && par.adjIteration == 0 &&
        par.PARAM_INCLUDE_ADJACENCY.wasSet == false) {
        par.includeAdjacency = false;
    }

    if (par.PARAM_CLUSTER_STEPS.wasSet == false) {
        par.clusterSteps = batchClusterAutomaticIterations(par.sensitivity);
    }
}

void validateBatchBackend(Parameters &par) {
    if (par.createdbMode != Parameters::SEQUENCE_SPLIT_MODE_HARD &&
        par.createdbMode != Parameters::SEQUENCE_SPLIT_MODE_SOFT) {
        Debug(Debug::ERROR) << "Batch clustering supports --createdb-mode 0 or 1 only. "
                            << "--createdb-mode " << par.createdbMode
                            << " changes the createdb layout and can change clustering results.\n";
        EXIT(EXIT_FAILURE);
    }

    if (par.batchBackend == "single-node") {
        if (isS3Uri(par.db2) || isS3Uri(par.db3)) {
            Debug(Debug::ERROR) << "--backend single-node requires local <resultDir> and <tmpDir>. Use --backend aws-batch for S3 paths.\n";
            EXIT(EXIT_FAILURE);
        }
        if (par.PARAM_BATCH_SLURM_NODELIST.wasSet || par.PARAM_BATCH_SLURM_PARTITION.wasSet ||
            par.PARAM_BATCH_SLURM_TIME.wasSet || par.PARAM_BATCH_SLURM_MEM.wasSet ||
            par.PARAM_BATCH_SLURM_EXTRA.wasSet) {
            Debug(Debug::ERROR) << "--slurm-* parameters are only valid with --backend multi-node.\n";
            EXIT(EXIT_FAILURE);
        }
        return;
    }

    if (par.batchBackend == "multi-node") {
        if (isS3Uri(par.db2) || isS3Uri(par.db3)) {
            Debug(Debug::ERROR) << "--backend multi-node requires shared local <resultDir> and <tmpDir>. Use --backend aws-batch for S3 paths.\n";
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
        if (par.PARAM_BATCH_SLURM_NODELIST.wasSet || par.PARAM_BATCH_SLURM_PARTITION.wasSet ||
            par.PARAM_BATCH_SLURM_TIME.wasSet || par.PARAM_BATCH_SLURM_MEM.wasSet ||
            par.PARAM_BATCH_SLURM_EXTRA.wasSet) {
            Debug(Debug::ERROR) << "--slurm-* parameters are only valid with --backend multi-node.\n";
            EXIT(EXIT_FAILURE);
        }
        if (par.batchNodeWorkDir.empty()) {
            Debug(Debug::ERROR) << "--backend aws-batch requires --node-work-dir (a container-local disk path). Without it each chunk's createdb/cluster tmp defaults to the container /tmp, which is often too small at batch scale.\n";
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
        return;
    }

    Debug(Debug::ERROR) << "Invalid --backend " << par.batchBackend
                        << ". Valid values are single-node, multi-node and aws-batch.\n";
    EXIT(EXIT_FAILURE);
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
    // --write-lookup 0: the per-chunk .lookup is unused here. The inner clust reads .lookup only
    // in set-mode (clusteringSetMode), which the batch entry points never enable; accessions come
    // from the _h header DB (createtsv/convert2fasta), so the .lookup only costs time and disk.
    // --createdb-mode 0: let createdb read FASTA/.zst natively and write the compact sequence DB.
    // Mode 1 is left as an explicit opt-in only; in batch it requires materializing a plain FASTA
    // first, which is slower and does not reduce node-local peak disk for compressed chunks.
    return "--shuffle 0 --write-lookup 0 --createdb-mode " + SSTR(par.createdbMode) + " --threads " +
           SSTR(par.threads) + " -v " + SSTR(par.verbosity);
}

std::string buildCreatetsvPar(const Parameters &par) {
    return "--threads " + SSTR(par.threads) + " -v " + SSTR(par.verbosity);
}

std::vector<MMseqsParameter*> innerClusterParameters(Parameters &par, const std::string &clusterCmd) {
    std::vector<MMseqsParameter*> inner;
    inner.push_back(&par.PARAM_C);
    inner.push_back(&par.PARAM_COV_MODE);
    inner.push_back(&par.PARAM_MIN_SEQ_ID);
    inner.push_back(&par.PARAM_CLUSTER_MODE);
    inner.push_back(&par.PARAM_KMER_PER_SEQ);
    inner.push_back(&par.PARAM_INCLUDE_COUNTTABLE);
    inner.push_back(&par.PARAM_NUM_COUNTS);
    inner.push_back(&par.PARAM_INCLUDE_ADJACENCY);
    inner.push_back(&par.PARAM_NUM_ADJACENCY);
    inner.push_back(&par.PARAM_SWITCH_CONSENSUS_REP);
    inner.push_back(&par.PARAM_REMOVE_TMP_FILES);
    inner.push_back(&par.PARAM_THREADS);
    inner.push_back(&par.PARAM_SPLIT_MEMORY_LIMIT);
    inner.push_back(&par.PARAM_PRELOAD_MODE);
    inner.push_back(&par.PARAM_COMPRESSED);
    inner.push_back(&par.PARAM_V);

    if (clusterCmd == "linclust") {
        inner.push_back(&par.PARAM_COMPRESS_KMER_TMP_FILES);
        inner.push_back(&par.PARAM_CLUST_HASH);
        inner.push_back(&par.PARAM_LINCLUST_VERSION);
    } else {
        inner.push_back(&par.PARAM_CLUSTER_VERSION);
        inner.push_back(&par.PARAM_LINCLUST_VERSION);
        inner.push_back(&par.PARAM_CASCADED);
        inner.push_back(&par.PARAM_CLUSTER_STEPS);
        inner.push_back(&par.PARAM_CLUSTER_REASSIGN);
        inner.push_back(&par.PARAM_MAX_SEQS);
        inner.push_back(&par.PARAM_S);
    }

    return inner;
}

std::string buildInnerClusterParFromCurrent(Parameters &par, const std::string &clusterCmd) {
    std::vector<MMseqsParameter*> inner = innerClusterParameters(par, clusterCmd);
    return par.createParameterString(inner);
}

std::string buildInnerClusterPar(Parameters &par, const std::string &clusterCmd) {
    if (clusterCmd == "linclust") {
        applyBatchLinclustAutomagic(par);
    } else {
        applyBatchClusterAutomagic(par);
    }
    return buildInnerClusterParFromCurrent(par, clusterCmd);
}

bool hasRound0ClusterOverride(const Parameters &par) {
    return par.PARAM_BATCH_ROUND0_MIN_SEQ_ID.wasSet ||
           par.PARAM_BATCH_ROUND0_C.wasSet ||
           par.PARAM_BATCH_ROUND0_COV_MODE.wasSet ||
           par.PARAM_BATCH_ROUND0_CLUSTER_MODE.wasSet ||
           par.PARAM_BATCH_ROUND0_KMER_PER_SEQ.wasSet ||
           par.PARAM_BATCH_ROUND0_INCLUDE_COUNTTABLE.wasSet ||
           par.PARAM_BATCH_ROUND0_NUM_COUNTS.wasSet ||
           par.PARAM_BATCH_ROUND0_NUM_ADJACENCY.wasSet ||
           par.PARAM_BATCH_ROUND0_CLUST_HASH.wasSet ||
           par.PARAM_BATCH_ROUND0_SPLIT_MEMORY_LIMIT.wasSet ||
           par.PARAM_BATCH_ROUND0_PRELOAD_MODE.wasSet;
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
    if (par.PARAM_BATCH_ROUND0_INCLUDE_COUNTTABLE.wasSet) {
        par.includeCountTable = par.batchRound0IncludeCountTable;
    }
    if (par.PARAM_BATCH_ROUND0_NUM_COUNTS.wasSet) {
        par.countTableIteration = par.batchRound0CountTableIteration;
    }
    if (par.PARAM_BATCH_ROUND0_NUM_ADJACENCY.wasSet) {
        par.adjIteration = par.batchRound0AdjIteration;
        par.includeAdjacency = (par.batchRound0AdjIteration > 0);
    }
    if (par.PARAM_BATCH_ROUND0_CLUST_HASH.wasSet) {
        par.clustHash = par.batchRound0ClustHash;
    }
    if (par.PARAM_BATCH_ROUND0_SPLIT_MEMORY_LIMIT.wasSet) {
        par.splitMemoryLimit = par.batchRound0SplitMemoryLimit;
    }
    if (par.PARAM_BATCH_ROUND0_PRELOAD_MODE.wasSet) {
        par.preloadMode = par.batchRound0PreloadMode;
    }
}

void applyBatchClusterAutomagic(Parameters &par, const std::string &clusterCmd) {
    if (clusterCmd == "linclust") {
        applyBatchLinclustAutomagic(par);
    } else {
        applyBatchClusterAutomagic(par);
    }
}

std::string buildRound0ClusterPar(Parameters &par, const std::string &clusterCmd) {
    if (hasRound0ClusterOverride(par) == false) {
        return "";
    }

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

    // Apply round0 values before automagic so derived settings (e.g. cov-mode -> cluster-mode)
    // are recomputed for round0, then apply explicit round0 overrides once more so they win.
    applyRound0ClusterOverrides(par);
    applyBatchClusterAutomagic(par, clusterCmd);
    applyRound0ClusterOverrides(par);

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
    const std::string maxRounds = SSTR(par.batchMaxRounds);
    const std::string minReductionRatio = SSTR(par.batchMinReductionRatio);
    const std::string convergencePatience = SSTR(par.batchConvergencePatience);
    const std::string minReductionCount = SSTR(par.batchMinReductionCount);
    const std::string maxChunkAttempts = SSTR(par.batchMaxChunkAttempts);
    const std::string compressBatchOutputs = par.batchCompressOutputs ? "1" : "0";
    const std::string mergeBuckets = SSTR(par.batchMergeBuckets);
    const std::string createdbPar = buildCreatedbPar(par);
    const std::string createtsvPar = buildCreatetsvPar(par);

    cmd.addVariable("CLUSTER_CMD", clusterCmd.c_str());
    cmd.addVariable("CLUSTER_PAR", clusterPar.c_str());
    cmd.addVariable("ROUND0_CLUSTER_PAR", round0ClusterPar.empty() ? NULL : round0ClusterPar.c_str());
    cmd.addVariable("CREATEDB_PAR", createdbPar.c_str());
    cmd.addVariable("CREATETSV_PAR", createtsvPar.c_str());
    cmd.addVariable("THREADS", threads.c_str());
    cmd.addVariable("CHUNK_MAX_BYTES", chunkMaxBytes.c_str());
    cmd.addVariable("CHUNK_MAX_SEQS", chunkMaxSeqs.c_str());
    cmd.addVariable("ROUND0_CHUNK_MAX_BYTES", par.PARAM_BATCH_ROUND0_CHUNK_MAX_BYTES.wasSet ? round0ChunkMaxBytes.c_str() : NULL);
    cmd.addVariable("ROUND0_CHUNK_MAX_SEQS", par.PARAM_BATCH_ROUND0_CHUNK_MAX_SEQS.wasSet ? round0ChunkMaxSeqs.c_str() : NULL);
    cmd.addVariable("MERGE_BUCKETS", mergeBuckets.c_str());
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
    cmd.addVariable("ROUND0_SLURM_NODELIST", par.batchRound0SlurmNodelist.empty() ? NULL : par.batchRound0SlurmNodelist.c_str());
    cmd.addVariable("ROUND0_NODE_WORK_DIR", par.batchRound0NodeWorkDir.empty() ? NULL : par.batchRound0NodeWorkDir.c_str());
}

int execBatchEngine(Parameters &par, const std::string &programDir,
                    const std::string &mode, const std::vector<std::string> &modeArgs,
                    const std::string &clusterCmd, const std::string &clusterPar,
                    const std::string &round0ClusterPar) {
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
    args.push_back(par.db1);
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
    args.push_back(par.db1);
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
            << "--backend aws-batch requires S3 prefixes for <resultDir> and <tmpDir>.\n"
            << "Example: mmseqs " << command.cmd
            << " s3://bucket/input.manifest s3://bucket/results/run1 s3://bucket/work/run1 --backend aws-batch\n";
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
        "Batch createdb mode: 0 copies FASTA/.zst into a compact MMseqs DB, 1 soft-links plain single-line FASTA. Mode 2/GPU DB layout is not supported by batch clustering",
        NULL,
        par.PARAM_CREATEDB_MODE.category);
}

} // namespace

int linclustbatch(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchLinclustDefaults(&par);
    setBatchClusteringDescriptions(par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
    validateBatchBackend(par);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust");
    std::string clusterPar = buildInnerClusterPar(par, "linclust");
    par.printParameters(command.cmd, argc, argv, *command.params);
    return runBatchClustering(par, command, par.linclustbatch, "linclust", clusterPar, round0ClusterPar);
}

int clusterbatch(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchClusterDefaults(&par);
    setBatchClusteringDescriptions(par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
    validateBatchBackend(par);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "cluster");
    std::string clusterPar = buildInnerClusterPar(par, "cluster");
    par.printParameters(command.cmd, argc, argv, *command.params);
    return runBatchClustering(par, command, par.clusterbatch, "cluster", clusterPar, round0ClusterPar);
}

int linclustbatchworker(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    int requestedThreads = parseRequestedThreads(argc, argv);
    setBatchLinclustDefaults(&par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    restoreRequestedThreads(par, requestedThreads);
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
    std::string round0ClusterPar = buildRound0ClusterPar(par, "cluster");
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
    par.printParameters(command.cmd, argc, argv, *command.params);

    std::vector<std::string> args;
    args.push_back(par.db1);
    args.push_back(par.db2);
    args.push_back(par.db3);
    args.push_back(par.db4);
    return execBatchEngine(par, par.db4, "propagate", args, "linclust", "", "");
}
