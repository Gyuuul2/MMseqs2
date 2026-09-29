#include "Parameters.h"
#include "FileUtil.h"
#include "CommandCaller.h"
#include "Debug.h"
#include "Util.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "batch_clustering.sh.h"


static void setBatchLinclustDefaults(Parameters *p) {
    p->covThr = 0.8f;
    p->maskMode = 0;
    p->seqIdThr = 0.9f;
    p->linclustVersion = Parameters::LINCLUST_VERSION2;
    p->clustHash = false;
    // the chunks are already single line FASTA, so the db can soft link them instead of copying
    p->createdbMode = Parameters::SEQUENCE_SPLIT_MODE_SOFT;
    p->removeTmpFiles = true;
}

static void setBatchClusterDefaults(Parameters *p) {
    p->covThr = 0.8f;
    p->maskMode = 0;
    p->maxResListLen = 20;
    p->clusterVersion = Parameters::CLUSTER_VERSION2;
    // the chunks are already single line FASTA, so the db can soft link them instead of copying
    p->createdbMode = Parameters::SEQUENCE_SPLIT_MODE_SOFT;
    p->removeTmpFiles = true;
}

// values the inner linclust/cluster would not arrive at on its own, so they are passed even at default
static void setBatchMustPassAlong(Parameters *p, bool cascaded) {
    p->PARAM_C.wasSet = true;
    p->PARAM_MIN_SEQ_ID.wasSet = true;
    p->PARAM_REMOVE_TMP_FILES.wasSet = true;
    p->PARAM_THREADS.wasSet = true;
    p->PARAM_V.wasSet = true;
    if (cascaded) {
        p->PARAM_CLUSTER_VERSION.wasSet = true;
    } else {
        p->PARAM_LINCLUST_VERSION.wasSet = true;
    }
}

static bool isNonSymmetricCovMode(const Parameters &par) {
    return (par.covMode == Parameters::COV_MODE_TARGET ||
            par.covMode == Parameters::COV_MODE_QUERY);
}

static bool hasSlurmParameters(const Parameters &par) {
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

static bool isS3Uri(const std::string &path) {
    return Util::startWith("s3://", path);
}



static void validateBatchCreatedbMode(const Parameters &par) {
    // mode 2 stores numeric codes and a GPU padded layout, which changes clustering results
    if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_GPU) {
        Debug(Debug::ERROR) << "Batch clustering supports --createdb-mode 0, 1 or 3. "
                            << "Mode 2 stores a GPU database layout and can change clustering results.\n";
        EXIT(EXIT_FAILURE);
    }
}

static std::string withoutTrailingSlash(const std::string &path) {
    size_t end = path.size();
    while (end > 1 && path[end - 1] == '/') {
        end--;
    }
    return path.substr(0, end);
}

// a bucket root has no key to scope the recursive cleanup to, so it would take the bucket
static void rejectS3BucketRoot(const std::string &path, const char *what) {
    if (isS3Uri(path) == false) {
        return;
    }
    const std::string rest = withoutTrailingSlash(path.substr(strlen("s3://")));
    const size_t slash = rest.find('/');
    if (slash == std::string::npos || slash == 0) {
        Debug(Debug::ERROR) << what << " " << path << " is a bucket root; give s3://bucket/prefix\n";
        EXIT(EXIT_FAILURE);
    }
}

static void validateBatchTmpPolicy(const Parameters &par) {
    rejectS3BucketRoot(par.db2, "<resultDir>");
    rejectS3BucketRoot(par.db3, "<tmpDir>");
    const std::string resultDir = withoutTrailingSlash(par.db2);
    const std::string tmpDir = withoutTrailingSlash(par.db3);
    if (resultDir == tmpDir || Util::startWith(tmpDir + "/", resultDir)) {
        Debug(Debug::ERROR) << "<resultDir> " << par.db2 << " is inside <tmpDir> " << par.db3
                            << ". Cleaning up the work area would take the result with it; give them separate paths.\n";
        EXIT(EXIT_FAILURE);
    }
    if (par.removeTmpFiles == false) {
        Debug(Debug::WARNING) << "--remove-tmp-files 0: per-chunk temporary files are KEPT and "
                                 "accumulate across all chunks and rounds. At batch scale this can "
                                 "fill the working disk. Use only for debugging a small input.\n";
    }
}

static void validateBatchServerBackend(Parameters &par, const Command &command) {
    validateBatchCreatedbMode(par);
    validateBatchTmpPolicy(par);

    // the nodelist is what submits the run to SLURM; without one the work runs in this process
    par.batchBackend = par.batchSlurmNodelist.empty() ? "single-node" : "multi-node";
    if (par.batchSlurmNodelist.empty()) {
        if (isS3Uri(par.db2) || isS3Uri(par.db3)) {
            Debug(Debug::ERROR) << "<resultDir> and <tmpDir> must be local paths. Use mmseqs "
                                << command.cmd << "-aws for S3 paths.\n";
            EXIT(EXIT_FAILURE);
        }
        if (hasSlurmParameters(par)) {
            Debug(Debug::ERROR) << "--slurm-* parameters need --slurm-nodelist, which is what submits the run to SLURM.\n";
            EXIT(EXIT_FAILURE);
        }
        return;
    }

    if (isS3Uri(par.db2) || isS3Uri(par.db3)) {
        Debug(Debug::ERROR) << "--slurm-nodelist requires shared local <resultDir> and <tmpDir>. Use mmseqs "
                            << command.cmd << "-aws for S3 paths.\n";
        EXIT(EXIT_FAILURE);
    }
    if (par.batchNodeWorkDir.empty()) {
        Debug(Debug::ERROR) << "--slurm-nodelist requires --node-work-dir (a per-node LOCAL disk path, e.g. /mnt/scratch). Without it each chunk's createdb/cluster tmp is written under the shared <tmpDir>, hammering the shared filesystem.\n";
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
        Util::startWith(par.db3 + "/", par.batchNodeWorkDir)) {
        Debug(Debug::WARNING) << "--node-work-dir is inside the shared <tmpDir> (" << par.db3
                              << "); this defeats its purpose -- per-chunk tmp will still hit the shared filesystem. Point it at per-node local disk.\n";
    }
}

static void validateBatchAwsBackend(Parameters &par, const Command &command) {
    validateBatchCreatedbMode(par);
    validateBatchTmpPolicy(par);

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

// the submit side has no shared tmp of its own; <resultDir> and <tmpDir> are s3://
static std::string createBatchSubmitDirectory(const std::string &hash) {
    const char *tmp = std::getenv("TMPDIR");
    std::string base = (tmp != NULL && tmp[0] != '\0') ? tmp : "/tmp";
    return FileUtil::createTemporaryDirectory(base + "/mmseqs-batch-submit", hash);
}

static std::string buildCreatedbPar(Parameters &par) {
    // the per-chunk .lookup is unused, and shuffling would renumber ids and change tie-breaks
    par.shuffleDatabase = false;
    par.writeLookup = false;
    return par.createParameterString(par.createdb);
}

static std::string buildCreatetsvPar(Parameters &par) {
    return par.createParameterString(par.createtsv);
}

static const std::vector<MMseqsParameter*>& innerClusterParameters(Parameters &par,
                                                           const std::string &clusterCmd) {
    return (clusterCmd == "linclust") ? par.linclustbatchinner : par.clusterbatchinner;
}

static std::string buildInnerClusterParFromCurrent(Parameters &par, const std::string &clusterCmd) {
    return par.createParameterString(innerClusterParameters(par, clusterCmd), true);
}



static void applyRound0ClusterDefaults(Parameters &par, const std::string &clusterCmd) {
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

static void applyRound0IncludePair(bool includeSet, bool includeValue, bool numSet, int numValue,
                            int iterationDefault, const char *includeName, const char *numName,
                            bool &includeOut, int &iterationOut) {
    if (includeSet) {
        includeOut = includeValue;
        if (includeValue) {
            iterationOut = iterationDefault;
        }
    }
    if (numSet) {
        iterationOut = numValue;
    }
    Util::resolveIncludeIterationPair(includeSet, includeOut, numSet, iterationOut, includeName, numName);
}

// an override only reaches the inner command if its own parameter is marked as given
static void applyRound0ClusterOverrides(Parameters &par) {
    if (par.PARAM_BATCH_ROUND0_MIN_SEQ_ID.wasSet) {
        par.seqIdThr = par.batchRound0SeqIdThr;
    }
    if (par.PARAM_BATCH_ROUND0_C.wasSet) {
        par.covThr = par.batchRound0CovThr;
    }
    if (par.PARAM_BATCH_ROUND0_COV_MODE.wasSet) {
        par.covMode = par.batchRound0CovMode;
        par.PARAM_COV_MODE.wasSet = true;
    }
    if (par.PARAM_BATCH_ROUND0_CLUSTER_MODE.wasSet) {
        par.clusteringMode = par.batchRound0ClusteringMode;
        par.PARAM_CLUSTER_MODE.wasSet = true;
    }
    if (par.PARAM_BATCH_ROUND0_KMER_PER_SEQ.wasSet) {
        par.kmersPerSequence = par.batchRound0KmersPerSequence;
    }
    const bool nonSymmetric = isNonSymmetricCovMode(par);
    applyRound0IncludePair(par.PARAM_BATCH_ROUND0_INCLUDE_COUNTTABLE.wasSet, par.batchRound0IncludeCountTable,
                           par.PARAM_BATCH_ROUND0_NUM_COUNTS.wasSet, par.batchRound0CountTableIteration,
                           Parameters::CLUST_LINEAR_DEFAULT_NUM_COUNT_TABLE,
                           "--round0-include-count-table", "--round0-num-count-table",
                           par.includeCountTable, par.countTableIteration);
    applyRound0IncludePair(par.PARAM_BATCH_ROUND0_INCLUDE_ADJACENCY.wasSet, par.batchRound0IncludeAdjacency,
                           par.PARAM_BATCH_ROUND0_NUM_ADJACENCY.wasSet, par.batchRound0AdjIteration,
                           nonSymmetric ? Parameters::CLUST_LINEAR_DEFAULT_NUM_ADJACENCY
                                        : Parameters::CLUST_LINEAR_SYMMETRIC_NUM_ADJACENCY,
                           "--round0-include-adjacency", "--round0-num-adjacency",
                           par.includeAdjacency, par.adjIteration);
    if (par.PARAM_BATCH_ROUND0_CLUST_HASH.wasSet) {
        par.clustHash = par.batchRound0ClustHash;
    }
    if (par.PARAM_BATCH_ROUND0_SPLIT_MEMORY_LIMIT.wasSet) {
        par.splitMemoryLimit = par.batchRound0SplitMemoryLimit;
        par.PARAM_SPLIT_MEMORY_LIMIT.wasSet = true;
    }
    if (par.PARAM_BATCH_ROUND0_PRELOAD_MODE.wasSet) {
        par.preloadMode = par.batchRound0PreloadMode;
        par.PARAM_PRELOAD_MODE.wasSet = true;
    }
}


static std::string buildRound0ClusterPar(Parameters &par, const std::string &clusterCmd, bool forceClustHash) {
    applyRound0ClusterDefaults(par, clusterCmd);
    // round 0 pins what it forces, so these ride along even at their default
    par.PARAM_KMER_PER_SEQ.wasSet = true;
    par.PARAM_INCLUDE_COUNTTABLE.wasSet = true;
    par.PARAM_NUM_COUNTS.wasSet = true;
    par.PARAM_INCLUDE_ADJACENCY.wasSet = true;
    par.PARAM_NUM_ADJACENCY.wasSet = true;
    par.PARAM_CLUST_HASH.wasSet = true;
    applyRound0ClusterOverrides(par);
    if (forceClustHash) {
        par.clustHash = true;
    }
    return buildInnerClusterParFromCurrent(par, clusterCmd);
}

static void addBatchEngineVariables(CommandCaller &cmd, Parameters &par,
                             const std::string &clusterCmd,
                             const std::string &clusterPar,
                             const std::string &round0ClusterPar) {
    cmd.addVariable("CLUSTER_CMD", clusterCmd.c_str());
    cmd.addVariable("ROUND0_CLUSTER_CMD", clusterCmd == "cluster" ? "linclust" : NULL);
    cmd.addVariable("CLUSTER_PAR", clusterPar.c_str());
    cmd.addVariable("ROUND0_CLUSTER_PAR", round0ClusterPar.empty() ? NULL : round0ClusterPar.c_str());
    cmd.addVariable("CREATEDB_PAR", buildCreatedbPar(par).c_str());
    cmd.addVariable("CREATETSV_PAR", buildCreatetsvPar(par).c_str());
    cmd.addVariable("THREADS", SSTR(par.threads).c_str());
    cmd.addVariable("CHUNK_MAX_BYTES", SSTR(par.batchChunkMaxBytes).c_str());
    cmd.addVariable("CHUNK_MAX_SEQS", SSTR(par.batchChunkMaxSeqs).c_str());
    cmd.addVariable("ROUND0_CHUNK_MAX_BYTES", par.PARAM_BATCH_ROUND0_CHUNK_MAX_BYTES.wasSet ? SSTR(par.batchRound0ChunkMaxBytes).c_str() : NULL);
    cmd.addVariable("ROUND0_CHUNK_MAX_SEQS", par.PARAM_BATCH_ROUND0_CHUNK_MAX_SEQS.wasSet ? SSTR(par.batchRound0ChunkMaxSeqs).c_str() : NULL);
    cmd.addVariable("CHUNK_DISK_BUDGET", SSTR(par.batchChunkDiskBudget).c_str());
    cmd.addVariable("ROUND0_CHUNK_DISK_BUDGET", par.PARAM_BATCH_ROUND0_CHUNK_DISK_BUDGET.wasSet ? SSTR(par.batchRound0ChunkDiskBudget).c_str() : NULL);
    cmd.addVariable("MERGE_SPLITS", SSTR(par.batchMergeSplits).c_str());
    cmd.addVariable("MERGE_SPLIT_JOBS", SSTR(par.batchMergeSplitJobs).c_str());
    cmd.addVariable("MERGE_NODES", SSTR(par.batchMergeNodes).c_str());
    cmd.addVariable("BATCH_REP_FASTA_SPLITS", SSTR(par.batchRepFastaSplits).c_str());
    cmd.addVariable("ROUND0_THREADS", par.PARAM_BATCH_ROUND0_THREADS.wasSet ? SSTR(par.batchRound0Threads).c_str() : NULL);
    cmd.addVariable("ROUND0_BATCH_REP_FASTA_SPLITS", par.PARAM_BATCH_ROUND0_REP_FASTA_SPLITS.wasSet ? SSTR(par.batchRound0RepFastaSplits).c_str() : NULL);
    cmd.addVariable("MAX_ROUNDS", SSTR(par.batchMaxRounds).c_str());
    cmd.addVariable("MIN_REDUCTION_RATIO", SSTR(par.batchMinReductionRatio).c_str());
    cmd.addVariable("CONVERGENCE_PATIENCE", SSTR(par.batchConvergencePatience).c_str());
    cmd.addVariable("MAX_CHUNK_ATTEMPTS", SSTR(par.batchMaxChunkAttempts).c_str());
    cmd.addVariable("COMPRESS_BATCH_OUTPUTS", par.batchCompressOutputs ? "1" : "0");
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
    cmd.addVariable("SORT_TMP", par.batchSortTmpDir.c_str());
    // the shell tests for unset to derive it, so only export a real override
    cmd.addVariable("BATCH_AWS_TIMEOUT", SSTR(par.batchAwsTimeout).c_str());
    cmd.addVariable("BATCH_AWS_ALLOW_NONS3_INPUT", par.batchAwsAllowNonS3Input ? "1" : "0");
    cmd.addVariable("BATCH_AWS_DRY_RUN", par.batchAwsDryRun ? "1" : NULL);
    // derived from the work prefix when empty
    cmd.addVariable("BATCH_AWS_WORKER_ATTEMPTS",
                    par.batchAwsWorkerAttempts > 0 ? SSTR(par.batchAwsWorkerAttempts).c_str() : NULL);
}

static int execBatchEngine(Parameters &par, const std::string &programDir,
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

    // the work directory is named after the parameter hash, so its name identifies this run and
    // keeps two runs that share a node from colliding in node-local scratch
    cmd.addVariable("RUN_ID", FileUtil::baseName(programDir).c_str());

    std::string program = programDir + "/batch_clustering.sh";
    FileUtil::writeFile(program, batch_clustering_sh, batch_clustering_sh_len);

    std::vector<std::string> args;
    args.push_back(mode);
    args.insert(args.end(), modeArgs.begin(), modeArgs.end());
    cmd.execProgram(program.c_str(), args);

    // Unreachable
    assert(false);
    return 0;
}

// mmseqs createdb's file list convention: one path per line, optional second column = sequence count
static size_t appendBatchInputList(std::string &content, const std::string &tsv) {
    FILE *file = FileUtil::openFileOrDie(tsv.c_str(), "r", true);
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    size_t rows = 0;
    size_t counted = 0;
    while ((read = getline(&line, &len, file)) != -1) {
        std::string row(line, (read > 0 && line[read - 1] == '\n') ? (size_t)(read - 1) : (size_t)read);
        std::string path = row;
        std::string count;
        size_t tab = row.find('\t');
        if (tab != std::string::npos) {
            path = row.substr(0, tab);
            count = row.substr(tab + 1);
        }
        const char *blank = " \t\r\n";
        size_t begin = path.find_first_not_of(blank);
        if (begin == std::string::npos) {
            continue;
        }
        path = path.substr(begin, path.find_last_not_of(blank) + 1 - begin);
        if (path[0] == '#') {
            continue;
        }
        begin = count.find_first_not_of(blank);
        count = (begin == std::string::npos) ? "" : count.substr(begin, count.find_last_not_of(blank) + 1 - begin);
        content.append(path);
        if (count.empty() == false) {
            if (count.find_first_not_of("0123456789") != std::string::npos) {
                Debug(Debug::ERROR) << "Invalid sequence count \"" << count << "\" for " << path << " in " << tsv << "\n";
                EXIT(EXIT_FAILURE);
            }
            // the manifest's second column is the file size, measured later; the count is its third
            content.append("\t0\t");
            content.append(count);
            counted++;
        }
        content.append("\n");
        rows++;
    }
    free(line);
    fclose(file);
    if (rows == 0) {
        Debug(Debug::ERROR) << "No input file found in " << tsv << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (counted != 0 && counted != rows) {
        Debug(Debug::ERROR) << "The tsv " << tsv << " mixes rows with and without a sequence count: all rows need one, or none\n";
        EXIT(EXIT_FAILURE);
    }
    return rows;
}

// the engine reads one path per line, so any number of inputs is normalized into one manifest
static std::string resolveBatchInputManifest(const Parameters &par, const std::string &tmpDir) {
    std::string content;
    size_t inputs = par.filenames.size();
    if (Util::endsWith(".tsv", par.filenames[0])) {
        if (par.filenames.size() > 1) {
            Debug(Debug::ERROR) << "Only one tsv file can be given\n";
            EXIT(EXIT_FAILURE);
        }
        inputs = appendBatchInputList(content, par.filenames[0]);
    } else {
        for (size_t i = 0; i < par.filenames.size(); ++i) {
            content.append(par.filenames[i]);
            content.append("\n");
        }
    }
    std::string manifest = tmpDir + "/input.manifest";
    FileUtil::writeFile(manifest + ".tmp", reinterpret_cast<const unsigned char *>(content.c_str()), content.size());
    FileUtil::publishAtomically(manifest + ".tmp", manifest);
    Debug(Debug::INFO) << "Input: " << inputs << " file(s) written to " << manifest << "\n";
    return manifest;
}

static int runBatchClustering(Parameters &par, const Command &command, const std::string &clusterCmd,
                       const std::string &clusterPar, const std::string &round0ClusterPar,
                       std::string hash) {
    std::string mode = "run-single-node";
    std::string workDir;
    if (par.batchBackend == "aws-batch") {
        if (isS3Uri(par.db2) == false || isS3Uri(par.db3) == false) {
            Debug(Debug::ERROR)
                << "mmseqs " << command.cmd << " requires S3 prefixes for <resultDir> and <tmpDir>.\n"
                << "Example: mmseqs " << command.cmd
                << " s3://bucket/part_00.fa.zst s3://bucket/results/run1 s3://bucket/work/run1\n";
            EXIT(EXIT_FAILURE);
        }
        mode = "aws-submit";
        workDir = createBatchSubmitDirectory(hash);
    } else {
        if (par.batchBackend == "multi-node") {
            mode = "run-multi-node";
        }
        if (par.reuseLatest) {
            hash = FileUtil::getHashFromSymLink(par.db3 + "/latest");
        }
        workDir = FileUtil::createTemporaryDirectory(par.db3, hash);
    }

    std::vector<std::string> args;
    args.push_back(resolveBatchInputManifest(par, workDir));
    args.push_back(mode == "aws-submit" ? par.db3 : workDir);
    args.push_back(par.db2);
    return execBatchEngine(par, workDir, mode, args, clusterCmd, clusterPar, round0ClusterPar);
}

static void setBatchClusteringDescriptions(Parameters &par) {
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
        par.PARAM_KMER_PER_SEQ,
        "k-mers per sequence for rounds 1 and up. Round 0 only removes redundancy and always runs light (21 k-mers, no count table, no adjacency); use --round0-kmer-per-seq to change it",
        NULL,
        par.PARAM_KMER_PER_SEQ.category);
    par.overrideParameterDescription(
        par.PARAM_BATCH_COMPRESS_OUTPUTS,
        "Compress per-round and final batch FASTA/TSV shard files as .zst. This does not compress createdb sequence DBs",
        NULL,
        par.PARAM_BATCH_COMPRESS_OUTPUTS.category);
}

static void setBatchAwsDescriptions(Parameters &par) {
    par.overrideParameterDescription(
        par.PARAM_BATCH_NODE_WORK_DIR,
        "Container-LOCAL disk for chunk tasks (createdb/cluster tmp + sort spill). REQUIRED; mount a real volume at this path in the job definition",
        NULL,
        par.PARAM_BATCH_NODE_WORK_DIR.category);
}


static int dobatchclustering(int argc, const char **argv, const Command &command, bool cascaded, bool aws) {
    Parameters &par = Parameters::getInstance();
    if (cascaded) {
        setBatchClusterDefaults(&par);
    } else {
        setBatchLinclustDefaults(&par);
    }
    setBatchClusteringDescriptions(par);
    if (aws) {
        setBatchAwsDescriptions(par);
    }
    par.parseParameters(argc, argv, command, false, Parameters::PARSE_VARIADIC, 0);
    setBatchMustPassAlong(&par, cascaded);
    // the last two arguments are always the result dir and the shared tmp dir; the rest are inputs
    par.db3 = par.filenames.back(); par.filenames.pop_back();
    par.db2 = par.filenames.back(); par.filenames.pop_back();
    if (aws) {
        par.batchBackend = "aws-batch";
        validateBatchAwsBackend(par, command);
    } else {
        validateBatchServerBackend(par, command);
    }

    const std::string clusterCmd = cascaded ? "cluster" : "linclust";
    std::string clusterPar = buildInnerClusterParFromCurrent(par, clusterCmd);
    par.printParameters(command.cmd, argc, argv, *command.params);
    // taken before round 0 rewrites par, so the work directory names what the user asked for
    const std::string hash = SSTR(par.hashParameter(command.databases, par.filenames, *command.params));
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust", cascaded);
    return runBatchClustering(par, command, clusterCmd, clusterPar, round0ClusterPar, hash);
}

int linclustbatch(int argc, const char **argv, const Command &command) {
    return dobatchclustering(argc, argv, command, false, false);
}

int linclustbatchaws(int argc, const char **argv, const Command &command) {
    return dobatchclustering(argc, argv, command, false, true);
}

int clusterbatch(int argc, const char **argv, const Command &command) {
    return dobatchclustering(argc, argv, command, true, false);
}

int clusterbatchaws(int argc, const char **argv, const Command &command) {
    return dobatchclustering(argc, argv, command, true, true);
}

static int dobatchclusteringworker(int argc, const char **argv, const Command &command, bool cascaded) {
    Parameters &par = Parameters::getInstance();
    if (cascaded) {
        setBatchClusterDefaults(&par);
    } else {
        setBatchLinclustDefaults(&par);
    }
    par.parseParameters(argc, argv, command, false, 0, 0);
    setBatchMustPassAlong(&par, cascaded);

    const std::string clusterCmd = cascaded ? "cluster" : "linclust";
    std::string clusterPar = buildInnerClusterParFromCurrent(par, clusterCmd);
    par.printParameters(command.cmd, argc, argv, *command.params);
    std::string round0ClusterPar = buildRound0ClusterPar(par, "linclust", cascaded);

    std::vector<std::string> args;
    args.push_back(par.db1);
    args.push_back(par.db2);
    args.push_back(par.db3);
    return execBatchEngine(par, par.db3, "cluster-chunk", args, clusterCmd, clusterPar, round0ClusterPar);
}

int linclustbatchworker(int argc, const char **argv, const Command &command) {
    return dobatchclusteringworker(argc, argv, command, false);
}

int clusterbatchworker(int argc, const char **argv, const Command &command) {
    return dobatchclusteringworker(argc, argv, command, true);
}

// prepare and propagate move files, so they pass no clustering parameters along
static int dobatchclusteringstep(int argc, const char **argv, const Command &command,
                          const std::string &mode, bool fourArgs) {
    Parameters &par = Parameters::getInstance();
    setBatchLinclustDefaults(&par);
    par.parseParameters(argc, argv, command, false, 0, 0);
    par.printParameters(command.cmd, argc, argv, *command.params);

    std::vector<std::string> args;
    args.push_back(par.db1);
    args.push_back(par.db2);
    args.push_back(par.db3);
    if (fourArgs) {
        args.push_back(par.db4);
    }
    const std::string programDir = fourArgs ? par.db4 : FileUtil::dirName(par.db3);
    return execBatchEngine(par, programDir, mode, args, "linclust", "", "");
}

int batchclusteringprepare(int argc, const char **argv, const Command &command) {
    return dobatchclusteringstep(argc, argv, command, "prepare", false);
}

int batchclusteringmerge(int argc, const char **argv, const Command &command) {
    return dobatchclusteringstep(argc, argv, command, "propagate", true);
}
