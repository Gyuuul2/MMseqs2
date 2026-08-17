#!/usr/bin/env bash
# bash, not sh: arrays, [[ ]], printf %q and process substitution are all required here
set -euo pipefail
export LC_ALL=C

fail() {
    echo "Error: $*" >&2
    exit 1
}

# xargs abandons its remaining input only on exit 255, so fanned-out workers fail with this
fail_hard() {
    echo "Error: $*" >&2
    exit 255
}

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

usage() {
    cat >&2 <<'EOF'
Usage:
  batch_clustering.sh prepare <input_manifest> <chunk_dir> <chunk_manifest>
  batch_clustering.sh cluster-chunk <chunk_uri> <result_prefix> <work_dir> [chunk_id] [round] [expected_seqs]
  batch_clustering.sh propagate <child_tsv_manifest> <parent_tsv_manifest> <out_manifest> <work_dir>
  batch_clustering.sh finalize <mapping_manifest> <rep_manifest> <result_prefix> <work_dir> [mark_final]
  batch_clustering.sh run-single-node <input_manifest> <work_dir> <result_dir>
  batch_clustering.sh run-multi-node <input_manifest> <work_dir> <result_dir>   (submits the SLURM event chain, then returns)
  batch_clustering.sh slurm-driver <input_manifest> <work_dir> <result_dir> <round>
  batch_clustering.sh slurm-worker <chunk_manifest> <result_prefix> <round_work_dir> <round> <worker_index> <num_workers>
  batch_clustering.sh slurm-merge <input_manifest> <work_dir> <result_dir> <round>
  batch_clustering.sh aws-submit <input_manifest> <work_s3_prefix> <result_s3_prefix>
  batch_clustering.sh aws-driver <input_manifest> <work_s3_prefix> <result_s3_prefix> <round>
  batch_clustering.sh aws-worker <chunk_manifest_s3> <result_s3_prefix> <round>
  batch_clustering.sh aws-merge <input_manifest> <work_s3_prefix> <result_s3_prefix> <round>

Environment (normally exported by mmseqs from the linclust-batch/cluster-batch command line,
or their -aws variants for the AWS vars; set them directly only when running this script standalone):
  MMSEQS ROUND0_MMSEQS THREADS ROUND0_THREADS CHUNK_MAX_BYTES CHUNK_MAX_SEQS MERGE_SPLITS MERGE_SPLIT_JOBS COMPRESS_RATIO
  CHUNK_DISK_BUDGET ROUND0_CHUNK_DISK_BUDGET DISK_POLL_SEC RAM_POLL_SEC
  CLUSTER_CMD ROUND0_CLUSTER_CMD CLUSTER_PAR ROUND0_CLUSTER_PAR CLUSTER_COV_MODE CREATEDB_PAR CREATEDB_SHUFFLE_SPLITS ROUND0_CREATEDB_SHUFFLE_SPLITS BATCH_COMPRESS_KMER_TMP_FILES CREATETSV_PAR COMPRESS_BATCH_OUTPUTS SORT_BUFFER_SIZE SORT_TMP
  MAX_ROUNDS MIN_REDUCTION_RATIO MIN_REDUCTION_COUNT CONVERGENCE_PATIENCE MAX_CHUNK_ATTEMPTS BATCH_REP_FASTA_SPLITS ROUND0_BATCH_REP_FASTA_SPLITS
  REMOVE_TMP NODE_WORK_DIR ROUND0_NODE_WORK_DIR
  BATCH_SLURM_NODELIST ROUND0_BATCH_SLURM_NODELIST BATCH_SLURM_PARTITION ROUND0_BATCH_SLURM_PARTITION
  BATCH_SLURM_TIME ROUND0_BATCH_SLURM_TIME BATCH_SLURM_MEM ROUND0_BATCH_SLURM_MEM BATCH_SLURM_EXTRA ROUND0_BATCH_SLURM_EXTRA
  BATCH_AWS_MACHINE ROUND0_BATCH_AWS_MACHINE BATCH_AWS_MACHINE_TAG_KEY
  BATCH_AWS_JOB_QUEUE BATCH_AWS_JOB_DEFINITION ROUND0_BATCH_AWS_JOB_QUEUE ROUND0_BATCH_AWS_JOB_DEFINITION
  BATCH_AWS_JOB_PREFIX BATCH_AWS_WORKER_ATTEMPTS BATCH_AWS_ALLOW_NONS3_INPUT
  BATCH_AWS_MMSEQS ROUND0_BATCH_AWS_MMSEQS BATCH_AWS_SCRIPT_URI BATCH_AWS_LOCAL_DIR BATCH_AWS_TIMEOUT BATCH_AWS_DRY_RUN
  BATCH_DELETE_SOURCE_CHUNK ROUND0_CREATEDB_MODE S3_CHUNK_PREFIX BATCH_AWS_ALLOW_NONS3_INPUT
  (AWS env var names must NOT start with 'AWS_BATCH' -- that prefix is reserved by the AWS Batch service.)
  All of the above also exist as command line parameters; the AWS ones only on the -aws commands
  (see 'mmseqs linclust-batch -h' and 'mmseqs linclust-batch-aws -h').
EOF
    exit 1
}

# INVARIANT: every ${VAR:-default} below must match Parameters::setDefaults / addBatchEngineVariables.
MMSEQS=${MMSEQS:-mmseqs}
# round0 may run on a different architecture than later rounds; empty uses $MMSEQS everywhere
ROUND0_MMSEQS=${ROUND0_MMSEQS:-}
THREADS=${THREADS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)}
# Optional round0-only thread count: round0 often runs on a smaller machine class than round1+.
ROUND0_THREADS=${ROUND0_THREADS:-}
[[ -z "$ROUND0_THREADS" || "$ROUND0_THREADS" =~ ^[1-9][0-9]*$ ]] || fail "ROUND0_THREADS must be a positive integer (got '$ROUND0_THREADS')"
CHUNK_MAX_BYTES=${CHUNK_MAX_BYTES:-21474836480}   # = 20*1024^3; must match BatchClustering.cpp/Parameters.cpp batchChunkMaxBytes so a standalone run chunks identically to the mmseqs-injected run
CHUNK_MAX_SEQS=${CHUNK_MAX_SEQS:-0}
ROUND0_CHUNK_MAX_BYTES=${ROUND0_CHUNK_MAX_BYTES:-}
ROUND0_CHUNK_MAX_SEQS=${ROUND0_CHUNK_MAX_SEQS:-}
CHUNK_DISK_BUDGET=${CHUNK_DISK_BUDGET:-0}   # bytes under a chunk's work dir; 0 = no guard
ROUND0_CHUNK_DISK_BUDGET=${ROUND0_CHUNK_DISK_BUDGET:-}
DISK_POLL_SEC=${DISK_POLL_SEC:-60}
RAM_POLL_SEC=${RAM_POLL_SEC:-2}
[[ "$DISK_POLL_SEC" =~ ^[1-9][0-9]*$ ]] || fail "DISK_POLL_SEC must be a positive integer (got '$DISK_POLL_SEC')"
[[ "$RAM_POLL_SEC" =~ ^[1-9][0-9]*$ ]] || fail "RAM_POLL_SEC must be a positive integer (got '$RAM_POLL_SEC')"
S3_CHUNK_PREFIX=${S3_CHUNK_PREFIX:-}
COMPRESS_BATCH_OUTPUTS=${COMPRESS_BATCH_OUTPUTS:-0}
[[ "$COMPRESS_BATCH_OUTPUTS" =~ ^[01]$ ]] || fail "COMPRESS_BATCH_OUTPUTS must be 0 or 1 (got '$COMPRESS_BATCH_OUTPUTS')"
# --createdb-mode 3 sorts by length, which createdb's parallel parse and align2clust's visit order both need
CREATEDB_SHUFFLE_SPLITS=${CREATEDB_SHUFFLE_SPLITS:-$THREADS}
ROUND0_CREATEDB_SHUFFLE_SPLITS=${ROUND0_CREATEDB_SHUFFLE_SPLITS:-}
[[ -z "$ROUND0_CREATEDB_SHUFFLE_SPLITS" || "$ROUND0_CREATEDB_SHUFFLE_SPLITS" =~ ^[1-9][0-9]*$ ]] || fail "ROUND0_CREATEDB_SHUFFLE_SPLITS must be a positive integer (got '$ROUND0_CREATEDB_SHUFFLE_SPLITS')"
CREATEDB_PAR=${CREATEDB_PAR:---write-lookup 0 --createdb-mode 3 --shuffle-splits ${CREATEDB_SHUFFLE_SPLITS}}
createdb_mode_from_par() {
    local par=" ${CREATEDB_PAR} "
    if [[ "$par" =~ [[:space:]]--createdb-mode=([0-9]+) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    elif [[ "$par" =~ [[:space:]]--createdb-mode[[:space:]]+([0-9]+) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    else
        printf '0'
    fi
}
validate_createdb_par() {
    local mode
    mode=$(createdb_mode_from_par)
    case "$mode" in
        0|1|3) ;;
        *) fail "batch clustering supports CREATEDB_PAR --createdb-mode 0, 1 or 3 only (got $mode). Mode 2/GPU stores numeric codes and can change clustering results." ;;
    esac
    case "${ROUND0_CREATEDB_MODE:-0}" in
        0|1|3) ;;
        *) fail "ROUND0_CREATEDB_MODE must be 0, 1 or 3 (got ${ROUND0_CREATEDB_MODE})" ;;
    esac
}
validate_createdb_par
CLUSTER_CMD=${CLUSTER_CMD:-linclust}
ROUND0_CLUSTER_CMD=${ROUND0_CLUSTER_CMD:-}
CLUSTER_COV_MODE=${CLUSTER_COV_MODE:-1}
if [[ -z "${CLUSTER_PAR+x}" ]]; then
    if [[ "$CLUSTER_CMD" == "cluster" ]]; then
        CLUSTER_PAR="-c 0.8 --cov-mode ${CLUSTER_COV_MODE} --cluster-version 1 --threads $THREADS"
    elif [[ "$CLUSTER_COV_MODE" -eq 0 ]]; then
        CLUSTER_PAR="--linclust-version 2 -c 0.8 --cov-mode 0 --cluster-mode 0 --num-adjacency 3 --num-count-table 2 --min-seq-id 0.9 --threads $THREADS"
    else
        CLUSTER_PAR="--linclust-version 2 -c 0.8 --cov-mode ${CLUSTER_COV_MODE} --cluster-mode 3 --num-adjacency 3 --include-count-table 0 --min-seq-id 0.9 --threads $THREADS"
    fi
fi
ROUND0_CLUSTER_PAR=${ROUND0_CLUSTER_PAR:-}
CREATETSV_PAR=${CREATETSV_PAR:---threads ${THREADS}}
MAX_ROUNDS=${MAX_ROUNDS:-32}
MIN_REDUCTION_RATIO=${MIN_REDUCTION_RATIO:-0.02}
CONVERGENCE_PATIENCE=${CONVERGENCE_PATIENCE:-1}
MIN_REDUCTION_COUNT=${MIN_REDUCTION_COUNT:-0}
MAX_CHUNK_ATTEMPTS=${MAX_CHUNK_ATTEMPTS:-1}
[[ "$MAX_CHUNK_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_CHUNK_ATTEMPTS must be a positive integer (got '$MAX_CHUNK_ATTEMPTS')"
COMPRESS_RATIO=${COMPRESS_RATIO:-3}
# representative FASTA shards per chunk; above --shuffle-splits it only helps the shuffle, not the parse
BATCH_REP_FASTA_SPLITS=${BATCH_REP_FASTA_SPLITS:-32}
[[ "$BATCH_REP_FASTA_SPLITS" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_REP_FASTA_SPLITS must be a positive integer (got '$BATCH_REP_FASTA_SPLITS')"
ROUND0_BATCH_REP_FASTA_SPLITS=${ROUND0_BATCH_REP_FASTA_SPLITS:-}
[[ -z "$ROUND0_BATCH_REP_FASTA_SPLITS" || "$ROUND0_BATCH_REP_FASTA_SPLITS" =~ ^[1-9][0-9]*$ ]] || fail "ROUND0_BATCH_REP_FASTA_SPLITS must be a positive integer (got '$ROUND0_BATCH_REP_FASTA_SPLITS')"

merge_default_threads() {
    if [[ "${THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s' "$THREADS"
    else
        printf '1'
    fi
}

auto_merge_splits() {
    local splits
    splits=$(merge_default_threads)   # always >= 1
    [[ "$splits" -gt 256 ]] && splits=256
    printf '%s' "$splits"
}

auto_merge_split_jobs() {
    local threads splits jobs
    threads=$(merge_default_threads)
    splits="${MERGE_SPLITS:-1}"
    jobs=$((threads / 8))
    [[ "$jobs" -lt 1 ]] && jobs=1
    [[ "$jobs" -gt 16 ]] && jobs=16
    [[ "$jobs" -gt "$splits" ]] && jobs="$splits"
    printf '%s' "$jobs"
}

if [[ -z "${MERGE_SPLITS:-}" || "$MERGE_SPLITS" == "0" ]]; then
    MERGE_SPLITS=$(auto_merge_splits)
fi
[[ "$MERGE_SPLITS" =~ ^[1-9][0-9]*$ ]] || fail "MERGE_SPLITS must be a positive integer (got '$MERGE_SPLITS')"
[[ -z "${ROUND0_MERGE_SPLITS:-}" ]] || fail "ROUND0_MERGE_SPLITS is not supported: round TSVs are written pre-split and propagate joins parent/child split by split, so a per-round split count would break the join and lose cluster members; use one global --merge-splits"
if [[ -z "${MERGE_SPLIT_JOBS:-}" || "$MERGE_SPLIT_JOBS" == "0" ]]; then
    MERGE_SPLIT_JOBS=$(auto_merge_split_jobs)
fi
[[ "$MERGE_SPLIT_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "MERGE_SPLIT_JOBS must be a positive integer (got '$MERGE_SPLIT_JOBS')"
[[ "$MERGE_SPLIT_JOBS" -gt "$MERGE_SPLITS" ]] && MERGE_SPLIT_JOBS="$MERGE_SPLITS"

# createtsv and the propagate fragment writer each hold one open file per merge split
check_split_fd_budget() {
    local fd_limit; fd_limit=$(ulimit -n 2>/dev/null || echo 256)
    [[ "$MERGE_SPLITS" -le $((fd_limit - 64)) ]] ||
        fail "--merge-splits ${MERGE_SPLITS} exceeds this node's safe open-file budget (ulimit -n=${fd_limit}); lower --merge-splits or raise 'ulimit -n'"
}
check_split_fd_budget
BATCH_BACKEND=${BATCH_BACKEND:-single-node}
NODE_WORK_DIR=${NODE_WORK_DIR:-}
ROUND0_NODE_WORK_DIR=${ROUND0_NODE_WORK_DIR:-}
BATCH_SLURM_NODELIST=${BATCH_SLURM_NODELIST:-}
ROUND0_BATCH_SLURM_NODELIST=${ROUND0_BATCH_SLURM_NODELIST:-}
BATCH_SLURM_PARTITION=${BATCH_SLURM_PARTITION:-}
ROUND0_BATCH_SLURM_PARTITION=${ROUND0_BATCH_SLURM_PARTITION:-}
BATCH_SLURM_TIME=${BATCH_SLURM_TIME:-}
ROUND0_BATCH_SLURM_TIME=${ROUND0_BATCH_SLURM_TIME:-}
BATCH_SLURM_MEM=${BATCH_SLURM_MEM:-}
ROUND0_BATCH_SLURM_MEM=${ROUND0_BATCH_SLURM_MEM:-}
BATCH_SLURM_EXTRA=${BATCH_SLURM_EXTRA:-}
ROUND0_BATCH_SLURM_EXTRA=${ROUND0_BATCH_SLURM_EXTRA:-}
BATCH_AWS_MACHINE=${BATCH_AWS_MACHINE:-}
ROUND0_BATCH_AWS_MACHINE=${ROUND0_BATCH_AWS_MACHINE:-}
BATCH_AWS_MACHINE_TAG_KEY=${BATCH_AWS_MACHINE_TAG_KEY:-mmseqs:machine}
BATCH_SCRIPT="$(cd -- "$(dirname -- "$0")" >/dev/null 2>&1 && pwd -P)/$(basename -- "$0")"

# mmseqs rejects a repeated flag, so drop it from the base before appending the round's value.
par_without_flag() {
    local par="$1" flag="$2"
    printf '%s' "$par" \
        | sed -E "s/(^|[[:space:]])${flag}([[:space:]]+[^[:space:]]+|=[^[:space:]]+)/\1/g"
}

round_cluster_par() {
    local round="$1"
    local base
    if [[ "$round" -eq 0 && -n "${ROUND0_CLUSTER_PAR:-}" ]]; then
        base="$ROUND0_CLUSTER_PAR"
    else
        base="$CLUSTER_PAR"
    fi
    # the spill defaults to round1+ because round0 chunks usually fit in memory; --kmer-write-to-disk overrides it
    local spill=1
    [[ "$round" -eq 0 ]] && spill=0
    [[ -n "${BATCH_KMER_WRITE_TO_DISK:-}" ]] && spill="$BATCH_KMER_WRITE_TO_DISK"
    base=$(par_without_flag "$base" --kmer-write-to-disk)
    base=$(par_without_flag "$base" --compress-kmer-tmp-files)
    base=$(round_par_threads "$round" "$base")
    printf '%s --kmer-write-to-disk %s --compress-kmer-tmp-files %s' \
        "$base" "$spill" "${BATCH_COMPRESS_KMER_TMP_FILES:-0}"
}

round_cluster_cmd() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_CLUSTER_CMD:-}" ]]; then
        printf '%s' "$ROUND0_CLUSTER_CMD"
    else
        printf '%s' "$CLUSTER_CMD"
    fi
}

round_chunk_max_bytes() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_CHUNK_MAX_BYTES:-}" ]]; then
        printf '%s' "$ROUND0_CHUNK_MAX_BYTES"
    else
        printf '%s' "$CHUNK_MAX_BYTES"
    fi
}

round_chunk_max_seqs() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_CHUNK_MAX_SEQS:-}" ]]; then
        printf '%s' "$ROUND0_CHUNK_MAX_SEQS"
    else
        printf '%s' "$CHUNK_MAX_SEQS"
    fi
}

round_chunk_disk_budget() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_CHUNK_DISK_BUDGET:-}" ]]; then
        printf '%s' "$ROUND0_CHUNK_DISK_BUDGET"
    else
        printf '%s' "$CHUNK_DISK_BUDGET"
    fi
}

round_node_work_dir() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_NODE_WORK_DIR:-}" ]]; then
        printf '%s' "$ROUND0_NODE_WORK_DIR"
    else
        printf '%s' "${NODE_WORK_DIR:-}"
    fi
}

round_slurm_nodelist() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_SLURM_NODELIST:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_SLURM_NODELIST"
    else
        printf '%s' "$BATCH_SLURM_NODELIST"
    fi
}

round_slurm_partition() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_SLURM_PARTITION:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_SLURM_PARTITION"
    else
        printf '%s' "$BATCH_SLURM_PARTITION"
    fi
}

round_slurm_time() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_SLURM_TIME:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_SLURM_TIME"
    else
        printf '%s' "$BATCH_SLURM_TIME"
    fi
}

round_slurm_mem() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_SLURM_MEM:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_SLURM_MEM"
    else
        printf '%s' "$BATCH_SLURM_MEM"
    fi
}

round_slurm_extra() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_SLURM_EXTRA:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_SLURM_EXTRA"
    else
        printf '%s' "$BATCH_SLURM_EXTRA"
    fi
}

# every mmseqs call goes through cluster_chunk, which knows its round, so this covers all backends
round_mmseqs() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_MMSEQS:-}" ]]; then
        printf '%s' "$ROUND0_MMSEQS"
    else
        printf '%s' "$MMSEQS"
    fi
}

round_threads() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_THREADS:-}" ]]; then
        printf '%s' "$ROUND0_THREADS"
    else
        printf '%s' "$THREADS"
    fi
}

round_rep_fasta_splits() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_REP_FASTA_SPLITS:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_REP_FASTA_SPLITS"
    else
        printf '%s' "$BATCH_REP_FASTA_SPLITS"
    fi
}

# rewrites --threads in a PAR string only when round0 overrides it, so absent overrides keep today's flags
round_par_threads() {
    local round="$1" par="$2"
    if [[ "$round" -eq 0 && -n "${ROUND0_THREADS:-}" ]]; then
        par="$(par_without_flag "$par" --threads) --threads $ROUND0_THREADS"
    fi
    printf '%s' "$par"
}

prepare_round() {
    local round="$1"
    shift
    local old_chunk_max_bytes="$CHUNK_MAX_BYTES"
    local old_chunk_max_seqs="$CHUNK_MAX_SEQS"
    local old_threads="$THREADS"
    local rc
    CHUNK_MAX_BYTES=$(round_chunk_max_bytes "$round")
    CHUNK_MAX_SEQS=$(round_chunk_max_seqs "$round")
    THREADS=$(round_threads "$round")
    set +e
    prepare "$@"
    rc=$?
    set -e
    CHUNK_MAX_BYTES="$old_chunk_max_bytes"
    CHUNK_MAX_SEQS="$old_chunk_max_seqs"
    THREADS="$old_threads"
    return "$rc"
}

with_round_node_work_dir() {
    local round="$1"
    shift
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    NODE_WORK_DIR="$node_work_dir" NODE_SCRATCH_SCOPE="round${round}" "$@"
}

# per-sort memory target for the merge, GNU sort only; --merge-splits sizes the input, this the memory
if [[ -z "${SORT_BUFFER_SIZE+x}" ]]; then
    if [[ "$MERGE_SPLIT_JOBS" -gt 1 ]]; then
        sort_pct=$((25 / MERGE_SPLIT_JOBS))
        [[ "$sort_pct" -lt 1 ]] && sort_pct=1
        SORT_BUFFER_SIZE="${sort_pct}%"
    else
        SORT_BUFFER_SIZE="25%"
    fi
fi
SORT_PARALLEL_OPT=""
if sort --version 2>/dev/null | grep -q GNU; then
    MERGE_SORT_THREADS="$THREADS"
    if [[ "$MERGE_SPLIT_JOBS" -gt 1 ]]; then
        MERGE_SORT_THREADS=$((THREADS / MERGE_SPLIT_JOBS))
        [[ "$MERGE_SORT_THREADS" -lt 1 ]] && MERGE_SORT_THREADS=1
    fi
    SORT_PARALLEL_OPT="--parallel=$MERGE_SORT_THREADS --buffer-size=$SORT_BUFFER_SIZE"
fi

# node-local scratch, NVMe when --node-work-dir is set, per-run so co-located runs cannot collide
resolve_node_scratch() {
    local work_dir="$1" name="${2:-}"
    if [[ -n "${NODE_WORK_DIR:-}" ]]; then
        # NODE_WORK_DIR already carries the user, and the round is what has to stay distinct
        printf '%s/mmseqs-batch/%s/%s' "$NODE_WORK_DIR" "${NODE_SCRATCH_SCOPE:-run}" "$name"
    else
        printf '%s/%s' "$work_dir" "$name"
    fi
}

# where GNU sort spills its external-sort runs (-T); SORT_TMP overrides the node-local default
resolve_sort_tmp() {
    local work_dir="$1"
    if [[ -n "${SORT_TMP:-}" ]]; then
        printf '%s' "$SORT_TMP"
    else
        resolve_node_scratch "$work_dir" sort-tmp
    fi
}

is_s3() {
    [[ "$1" == s3://* ]]
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

uri_exists() {
    local uri="$1"
    if is_s3 "$uri"; then
        [[ -n "${BATCH_AWS_DRY_RUN:-}" ]] && return 1
        need_cmd aws
        # head-object is an exact-key test; 'aws s3 ls' is a prefix match that reports chunk-1.done for chunk-10.done
        aws s3api head-object --bucket "$(s3_bucket "$uri")" --key "$(s3_key_prefix "$uri")" >/dev/null 2>&1
    else
        [[ -e "$uri" ]]
    fi
}

done_exists() {
    uri_exists "$1"
}

mark_done() {
    local done_uri="$1"
    local tmp_done="$2"
    printf 'done\n' > "$tmp_done"
    copy_out "$tmp_done" "$done_uri"
}

normalize_s3_prefix() {
    local p="$1"
    [[ -z "$p" ]] && return 0
    [[ "$p" == */ ]] && printf '%s' "$p" || printf '%s/' "$p"
}

join_uri() {
    local prefix
    prefix=$(normalize_s3_prefix "$1")
    printf '%s%s' "$prefix" "$2"
}

s3_bucket() {
    local rest="${1#s3://}"
    printf '%s' "${rest%%/*}"
}

s3_key_prefix() {
    local rest="${1#s3://}"
    if [[ "$rest" == */* ]]; then
        printf '%s' "${rest#*/}"
    fi
}

basename_no_compression() {
    local b
    b=$(basename "$1")
    b=${b%.zst}
    b=${b%.gz}
    b=${b%.fasta}
    b=${b%.fa}
    printf '%s' "$b"
}

copy_in() {
    local src="$1"
    local dst="$2"
    [[ "$src" == "$dst" ]] && return 0
    if is_s3 "$src"; then
        need_cmd aws
        aws s3 cp "$src" "$dst" --no-progress
    else
        cp "$src" "$dst"
    fi
}

copy_out() {
    local src="$1"
    local dst="$2"
    [[ "$src" == "$dst" ]] && return 0
    if is_s3 "$dst"; then
        need_cmd aws
        aws s3 cp "$src" "$dst" --no-progress
    else
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
    fi
}

append_raw_uri() {
    local src="$1"
    local dst="$2"
    if is_s3 "$src"; then
        need_cmd aws
        aws s3 cp "$src" - --no-progress >> "$dst"
    else
        cat "$src" >> "$dst"
    fi
}

stream_uri() {
    local uri="$1"
    if is_s3 "$uri"; then
        need_cmd aws
        case "$uri" in
            *.zst) need_cmd pzstd; aws s3 cp "$uri" - --no-progress | pzstd -dc ;;
            *.gz)  aws s3 cp "$uri" - --no-progress | gzip -dc ;;
            *)     aws s3 cp "$uri" - --no-progress ;;
        esac
    else
        case "$uri" in
            *.zst) need_cmd pzstd; pzstd -dc "$uri" ;;
            *.gz)  gzip -dc "$uri" ;;
            *)     cat "$uri" ;;
        esac
    fi
}

stream_manifest() {
    local manifest="$1"
    stream_uri "$manifest"
}

s3_list_prefix() {
    local prefix="$1"
    need_cmd aws
    local bucket key
    bucket=$(s3_bucket "$prefix")
    key=$(s3_key_prefix "$prefix")
    # list-objects-v2 returns exact keys and auto-paginates; stderr stays visible so a 403 aborts instead of reading as none-done
    aws s3api list-objects-v2 --bucket "$bucket" --prefix "$key" \
        --query 'Contents[].Key' --output text \
        | tr '\t' '\n' | awk -v base="s3://${bucket}/" 'NF && $0 != "None" { print base $0 }'
}

compress_to_zst() {
    local src="$1"
    local dst="$2"
    need_cmd pzstd
    # pzstd writes a multi-frame .zst so it can later be decompressed in parallel (pzstd -dc).
    pzstd -p "$THREADS" -3 -c "$src" > "$dst"
}

compress_batch_outputs_enabled() {
    [[ "${COMPRESS_BATCH_OUTPUTS:-0}" == "1" ]]
}

batch_compression_suffix() {
    if compress_batch_outputs_enabled; then
        printf '.zst'
    fi
}

final_cluster_file_name() {
    printf 'final_cluster_manifest.txt'
}

write_batch_output() {
    local src="$1"
    local dst="$2"
    if compress_batch_outputs_enabled; then
        compress_to_zst "$src" "$dst"
    else
        [[ "$src" == "$dst" ]] && return 0
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
    fi
}

# round0 is gated on ROUND0_CREATEDB_MODE (default 0), round1+ follows CREATEDB_PAR (default 0)
round_createdb_mode() {
    local round="$1"
    if [[ "$round" -eq 0 ]]; then
        printf '%s' "${ROUND0_CREATEDB_MODE:-0}"
    else
        createdb_mode_from_par
    fi
}

# only mode 1 needs one plain FASTA staged first; mode 0 and 3 read FASTA/.zst directly
round_createdb_softlink() {
    [[ "$(round_createdb_mode "$1")" == "1" ]]
}

round_createdb_par() {
    local round="$1" mode par
    mode=$(round_createdb_mode "$round")
    par=$(par_without_flag "$CREATEDB_PAR" --createdb-mode)
    # mode 3 warns and forces --shuffle 1, so drop the flag rather than let it read as if it applied
    [[ "$mode" == "3" ]] && par=$(par_without_flag "$par" --shuffle)
    if [[ "$round" -eq 0 && -n "${ROUND0_CREATEDB_SHUFFLE_SPLITS:-}" ]]; then
        par="$(par_without_flag "$par" --shuffle-splits) --shuffle-splits $ROUND0_CREATEDB_SHUFFLE_SPLITS"
    fi
    par=$(round_par_threads "$round" "$par")
    printf '%s --createdb-mode %s' "$par" "$mode"
}

append_singleline_fasta() {
    local src="$1"
    local dst="$2"
    # softlink mode needs single-line FASTA, so check the first record and normalize only if it is wrapped
    stream_uri "$src" | awk '
        function clean_line(line) {
            gsub(/[[:space:]]/, "", line)
            return line
        }
        function emit_first_raw(    i) {
            print first_header
            for (i = 1; i <= first_seq_n; i++) {
                print first_seq[i]
            }
        }
        function emit_norm_record() {
            if (norm_seen) {
                printf "\n"
            }
            print norm_header
            printf "%s", norm_seq
            norm_seen = 1
        }
        function enter_raw_mode() {
            mode = "raw"
            emit_first_raw()
        }
        function enter_norm_mode() {
            mode = "norm"
            norm_header = first_header
            norm_seq = ""
            for (i = 1; i <= first_seq_n; i++) {
                norm_seq = norm_seq clean_line(first_seq[i])
            }
            emit_norm_record()
            norm_header = ""
            norm_seq = ""
        }
        /^>/ {
            if (!have_first) {
                first_header = $0
                first_seq_n = 0
                have_first = 1
                next
            }
            if (mode == "") {
                if (first_seq_n == 1) {
                    enter_raw_mode()
                } else {
                    enter_norm_mode()
                }
            }
            if (mode == "raw") {
                print
            } else {
                if (norm_header != "") {
                    emit_norm_record()
                }
                norm_header = $0
                norm_seq = ""
            }
            next
        }
        {
            if (!have_first) {
                next
            }
            if (mode == "") {
                if (length($0) > 0) {
                    first_seq[++first_seq_n] = $0
                }
                next
            }
            if (mode == "raw") {
                print
            } else {
                line = clean_line($0)
                if (length(line) > 0) {
                    norm_seq = norm_seq line
                }
            }
        }
        END {
            if (!have_first) {
                exit
            }
            if (mode == "") {
                if (first_seq_n == 1) {
                    enter_raw_mode()
                } else {
                    enter_norm_mode()
                }
            } else if (mode == "norm" && norm_header != "") {
                emit_norm_record()
            }
            if (mode == "norm") {
                printf "\n"
            }
        }
    ' >> "$dst"
}

materialize_softlink_fasta() {
    local src="$1"
    local dst="$2"
    local tmp="${dst}.tmp.$$"
    # left by a previous attempt; mv below is atomic, so an existing dst is complete
    [[ -s "$dst" ]] && return 0
    mkdir -p "$(dirname "$dst")"
    : > "$tmp"
    append_singleline_fasta "$src" "$tmp"
    [[ -s "$tmp" ]] || fail "materialized FASTA is empty for input: $src"
    mv "$tmp" "$dst"
}

materialize_softlink_filelist() {
    local filelist="$1"
    local dst="$2"
    local tmp="${dst}.tmp.$$"
    [[ -s "$dst" ]] && return 0
    local mem cnt count=0
    mkdir -p "$(dirname "$dst")"
    : > "$tmp"
    # counted filelists carry "path<TAB>seqs"; only the path matters here
    while IFS=$'\t' read -r mem cnt || [[ -n "${mem:-}" ]]; do
        [[ -z "${mem:-}" || "$mem" =~ ^[[:space:]]*# ]] && continue
        append_singleline_fasta "$mem" "$tmp"
        count=$((count + 1))
    done < <(stream_manifest "$filelist")
    [[ "$count" -gt 0 ]] || fail "empty group filelist: $filelist"
    [[ -s "$tmp" ]] || fail "materialized FASTA is empty for group filelist: $filelist"
    mv "$tmp" "$dst"
}

write_chunk_manifest_line() {
    local manifest="$1"
    local chunk_id="$2"
    local uri="$3"
    local seqs="$4"
    local bytes="$5"
    printf '%s\t%s\t%s\t%s\n' "$chunk_id" "$uri" "$seqs" "$bytes" >> "$manifest"
}

# On-disk size in bytes (portable across GNU/BSD stat, and S3).
file_size_bytes() {
    local uri="$1"
    if is_s3 "$uri"; then
        need_cmd aws
        # || true: a missing or denied head must not abort the caller, this is a size estimate
        { aws s3api head-object --bucket "$(s3_bucket "$uri")" --key "$(s3_key_prefix "$uri")" \
            --query ContentLength --output text 2>/dev/null || true; } | awk 'END { print ($1 ~ /^[0-9]+$/) ? $1 : 0 }'
    else
        stat -c %s "$uri" 2>/dev/null || stat -f %z "$uri" 2>/dev/null || echo 0
    fi
}

# must be exact, not estimated: a different number per backend bin-packs round 1 differently
uncompressed_bytes() {
    local uri="$1" sz n
    if ! is_s3 "$uri"; then
        case "$uri" in
            *.zst)
                if command -v zstd >/dev/null 2>&1; then
                    # `|| true`: zstd -lv exits non-zero when there is no embedded content size
                    n=$(zstd -lv "$uri" 2>/dev/null | awk -F'[()]' '/Decompressed Size:/ {split($2, a, " "); print a[1]; exit}' || true)
                    if [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]]; then
                        printf '%s' "$n"
                        return
                    fi
                    log "sizing $uri by decompressing it: the frame header carries no decompressed size"
                    if command -v pzstd >/dev/null 2>&1; then
                        n=$(pzstd -dc -p "${THREADS:-1}" "$uri" 2>/dev/null | wc -c || true)
                    else
                        n=$(zstd -dc "$uri" 2>/dev/null | wc -c || true)
                    fi
                    if [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]]; then
                        printf '%s' "$n"
                        return
                    fi
                fi
                ;;
            *.gz)
                if command -v gzip >/dev/null 2>&1; then
                    log "sizing $uri by decompressing it: gzip stores no reliable size above 4 GiB"
                    n=$(gzip -dc "$uri" 2>/dev/null | wc -c || true)
                    if [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]]; then
                        printf '%s' "$n"
                        return
                    fi
                fi
                ;;
        esac
    fi
    sz=$(file_size_bytes "$uri")
    case "$uri" in
        *.zst|*.gz) printf '%s' "$(( sz * COMPRESS_RATIO ))" ;;
        *)          printf '%s' "$sz" ;;
    esac
}

# size and count one input; a pure function of the file, so prepare_group measures these in parallel
measure_input() {
    [[ "$#" -eq 1 ]] || usage
    local idx path bytes seqs
    IFS=$'\t' read -r idx path bytes seqs <<< "$1"
    [[ -n "${path:-}" ]] || fail_hard "measure-input got no path"
    [[ -z "${bytes:-}" || "$bytes" == "0" ]] && bytes=$(uncompressed_bytes "$path")
    if [[ "${bytes:-0}" -gt "$CHUNK_MAX_BYTES" ]]; then
        fail_hard "input file '$path' has an estimated uncompressed size of $(( bytes / 1048576 )) MiB, exceeding the per-chunk limit --chunk-max-bytes=$(( CHUNK_MAX_BYTES / 1048576 )) MiB. Grouping cannot split within a single file: either raise --chunk-max-bytes to fit your machine memory, or pre-split '$path' into smaller pieces."
    fi
    if [[ "${CHUNK_MAX_SEQS:-0}" -gt 0 ]]; then
        # count only when the manifest didn't already supply a count (3-column rep manifests do)
        if [[ -z "${seqs:-}" || "$seqs" == "0" ]]; then
            seqs=$(stream_uri "$path" | awk '/^>/ {n++} END {print n + 0}')
        fi
        # oversize check runs whether the count was supplied or measured (grouping can't split a file)
        if [[ "${seqs:-0}" -gt "$CHUNK_MAX_SEQS" ]]; then
            fail_hard "input file '$path' has ${seqs} sequences, exceeding the per-chunk limit --chunk-max-seqs=$CHUNK_MAX_SEQS. Grouping cannot split within a single file: raise --chunk-max-seqs or pre-split '$path' into smaller pieces."
        fi
    fi
    printf '%s\t%s\t%s\t%s\n' "$idx" "$path" "${bytes:-0}" "${seqs:-0}"
}

# group-mode prepare bin-packs already-split inputs by size without reading or rewriting any file
prepare_group() {
    local input_manifest="$1" chunk_dir="$2" chunk_manifest="$3"
    local done_file="${chunk_manifest}.done"
    if [[ -s "$chunk_manifest" ]] && done_exists "$done_file"; then
        log "prepare(group): reusing completed chunk manifest $chunk_manifest"
        return 0
    fi
    rm -rf "${chunk_dir:?}"
    mkdir -p "$chunk_dir" "$(dirname "$chunk_manifest")"
    # guard so a missing tool cannot silently fall back to a ratio estimate and shift chunk boundaries
    need_cmd awk
    if stream_manifest "$input_manifest" | awk 'NF && $0 !~ /^[[:space:]]*#/ && $1 ~ /\.zst([[:space:]]|$)/ { found = 1 } END { exit !found }'; then
        need_cmd zstd
    fi
    local s3_prefix
    s3_prefix=$(normalize_s3_prefix "$S3_CHUNK_PREFIX")

    local local_manifest="${chunk_manifest}.local"
    local chunk_manifest_tmp="${chunk_manifest}.tmp"
    : > "$local_manifest"
    : > "$chunk_manifest_tmp"

    # resolve each input to path/bytes/seqs then bin-pack; --chunk-max-seqs forces a full decompress pass to count
    need_cmd xargs
    if [[ "${CHUNK_MAX_SEQS:-0}" -gt 0 ]]; then
        log "prepare(group): --chunk-max-seqs=$CHUNK_MAX_SEQS set; counting sequences in every input (one full pass over the data, ${THREADS:-1} at a time)"
    fi
    stream_manifest "$input_manifest" \
    | awk -F'\t' 'NF && $0 !~ /^[[:space:]]*#/ { printf "%d\t%s\t%s\t%s%c", ++i, $1, $2 + 0, $3 + 0, 0 }' \
    | xargs -0 -n 1 -P "${THREADS:-1}" env BATCH_WORKER_DISPATCH=1 bash "$BATCH_SCRIPT" measure-input \
    | sort -k1,1n | cut -f2- \
    | awk -F'\t' -v dir="$chunk_dir" -v manifest="$local_manifest" \
          -v max_bytes="$CHUNK_MAX_BYTES" -v max_seqs="$CHUNK_MAX_SEQS" '
        function open_chunk() { cid++; flist = sprintf("%s/chunk-%08d.inputs.list", dir, cid); cb = 0; cs = 0; n = 0; counted = 1 }
        function close_chunk(    i) {
            if (n == 0) { return }
            # per-file counts ride along only when every member has one: createdb takes all rows counted, or none
            for (i = 0; i < n; i++) { print (counted ? paths[i] "\t" cnts[i] : paths[i]) > flist }
            printf "chunk-%08d\t%s\t%d\t%d\n", cid, flist, cs, cb >> manifest; close(manifest); close(flist)
        }
        BEGIN { cid = -1; flist = ""; cb = 0; cs = 0; n = 0 }
        {
            b = $2 + 0; s = $3 + 0
            if (flist == "") open_chunk()
            # new chunk when adding this file would exceed a limit; an oversized file gets its own chunk
            if (n > 0 && ((max_bytes > 0 && cb + b > max_bytes) ||
                          (max_seqs  > 0 && s > 0 && cs + s > max_seqs))) { close_chunk(); open_chunk() }
            paths[n] = $1; cnts[n] = s
            if (s <= 0) { counted = 0 }
            cb += b; cs += s; n++
        }
        END { close_chunk() }
    '
    [[ -s "$local_manifest" ]] || fail "no input files found in manifest: $input_manifest"

    # S3 runs need the filelist on S3 too, since worker containers cannot see the driver's local files
    local chunk_id flist uri
    while IFS=$'\t' read -r chunk_id flist seqs bytes || [[ -n "${chunk_id:-}" ]]; do
        [[ -z "${chunk_id:-}" ]] && continue
        uri="$flist"
        if [[ -n "$s3_prefix" ]]; then
            uri="${s3_prefix}$(basename "$flist")"
            copy_out "$flist" "$uri"
            [[ -n "${REMOVE_TMP:-}" ]] && rm -f "$flist"
        fi
        write_chunk_manifest_line "$chunk_manifest_tmp" "$chunk_id" "$uri" "$seqs" "$bytes"
    done < "$local_manifest"
    rm -f "$local_manifest"
    mv "$chunk_manifest_tmp" "$chunk_manifest"
    printf 'done\n' > "$done_file"
    log "chunk manifest (group) written: $chunk_manifest"
}

prepare() {
    [[ "$#" -eq 3 ]] || usage
    local input_manifest="$1"
    local chunk_dir="$2"
    local chunk_manifest="$3"
    # one input file is repartitioned into equal chunks, several are bin-packed by size
    local nfiles
    nfiles=$(stream_manifest "$input_manifest" | awk 'NF && $0 !~ /^[[:space:]]*#/' | wc -l | tr -d '[:space:]')
    if [[ "${nfiles:-0}" -gt 1 ]]; then
        # grouping bin-packs by byte size, so a positive byte limit is required
        [[ "${CHUNK_MAX_BYTES:-0}" -gt 0 ]] || fail "multiple input files ($nfiles): grouping needs a byte-based chunk limit. Set --chunk-max-bytes (a sequence-count limit cannot be applied without reading every file)."
        prepare_group "$@"
        return
    fi
    local s3_prefix
    s3_prefix=$(normalize_s3_prefix "$S3_CHUNK_PREFIX")
    local done_file="${chunk_manifest}.done"

    if [[ -s "$chunk_manifest" ]] && done_exists "$done_file"; then
        log "prepare: reusing completed chunk manifest $chunk_manifest"
        return 0
    fi

    rm -rf "${chunk_dir:?}"
    mkdir -p "$chunk_dir" "$(dirname "$chunk_manifest")"
    rm -f "$chunk_manifest" "$done_file" "${chunk_manifest}.tmp" "${chunk_manifest}.local"

    need_cmd awk
    # the chunks follow the input: a compressed input keeps them compressed, which holds the prepare peak down
    local chunk_suffix=".fa" compress_chunks=0 first_input
    first_input=$(stream_manifest "$input_manifest" | awk -F'\t' 'NF && $0 !~ /^[[:space:]]*#/ { print $1; exit }')
    case "$first_input" in
        *.zst|*.gz) chunk_suffix=".fa.zst"; compress_chunks=1; need_cmd zstd ;;
    esac
    local local_manifest="${chunk_manifest}.local"
    local chunk_manifest_tmp="${chunk_manifest}.tmp"
    : > "$local_manifest"
    : > "$chunk_manifest_tmp"

    stream_input_manifest() {
        local uri _rest
        # tolerate a 3-column "path<TAB>bytes<TAB>seqs" manifest (take field 1) as well as bare paths
        while IFS=$'\t' read -r uri _rest || [[ -n "$uri" ]]; do
            [[ -z "$uri" || "$uri" =~ ^[[:space:]]*# ]] && continue
            log "streaming input: $uri"
            stream_uri "$uri"
        done < <(stream_manifest "$input_manifest")
    }

    stream_input_manifest | awk \
        -v chunk_dir="$chunk_dir" \
        -v chunk_manifest="$local_manifest" \
        -v max_bytes="$CHUNK_MAX_BYTES" \
        -v max_seqs="$CHUNK_MAX_SEQS" \
        -v suffix="$chunk_suffix" \
        -v compress="$compress_chunks" \
        -v zthreads="$THREADS" '
            BEGIN {
                chunk_id = -1
                out = ""
                cmd = ""
                chunk_seqs = 0
                chunk_bytes = 0
            }
            function open_chunk() {
                chunk_id++
                # stream each chunk straight into its writer, so the whole input is never materialized
                out = sprintf("%s/chunk-%08d%s", chunk_dir, chunk_id, suffix)
                if (compress) {
                    cmd = "zstd -q -3 -T" zthreads " -c > \"" out "\""
                } else {
                    cmd = "cat > \"" out "\""
                }
                chunk_seqs = 0
                chunk_bytes = 0
            }
            function close_chunk() {
                if (cmd != "" && chunk_seqs > 0) {
                    if (close(cmd) != 0) {
                        printf "chunk writer failed while writing %s\n", out > "/dev/stderr"
                        exit 2
                    }
                    printf "chunk-%08d\t%s\t%d\t%d\n", chunk_id, out, chunk_seqs, chunk_bytes >> chunk_manifest
                    close(chunk_manifest)
                }
                out = ""
                cmd = ""
                chunk_seqs = 0
                chunk_bytes = 0
            }
            function emit_record() {
                if (!have) {
                    return
                }
                if (cmd == "") {
                    open_chunk()
                }
                rec_bytes = length(header) + length(seq) + 2
                if (chunk_seqs > 0 &&
                    ((max_seqs > 0 && chunk_seqs + 1 > max_seqs) ||
                     (max_bytes > 0 && chunk_bytes + rec_bytes > max_bytes))) {
                    close_chunk()
                    open_chunk()
                }
                print header | cmd
                print seq | cmd
                chunk_seqs++
                chunk_bytes += rec_bytes
                have = 0
                header = ""
                seq = ""
            }
            /^>/ {
                emit_record()
                header=$0
                seq=""
                have=1
                next
            }
            have {
                gsub(/[[:space:]]/, "", $0)
                seq = seq $0
            }
            END {
                emit_record()
                close_chunk()
            }
        '

    [[ -s "$local_manifest" ]] || fail "no FASTA records found in input manifest: $input_manifest"

    local chunk_id raw_chunk seqs bytes uri
    while IFS=$'\t' read -r chunk_id raw_chunk seqs bytes || [[ -n "${chunk_id:-}" ]]; do
        [[ -z "${chunk_id:-}" ]] && continue
        # raw_chunk is already in its final form from the streaming awk above, compressed or not
        uri="$raw_chunk"
        if [[ -n "$s3_prefix" ]]; then
            uri="${s3_prefix}$(basename "$raw_chunk")"
            copy_out "$raw_chunk" "$uri"
            if [[ -n "${REMOVE_TMP:-}" ]]; then
                rm -f "$raw_chunk"
            fi
        fi
        write_chunk_manifest_line "$chunk_manifest_tmp" "$chunk_id" "$uri" "$seqs" "$bytes"
        log "prepared ${chunk_id}: seqs=${seqs} bytes=${bytes} uri=${uri}"
    done < "$local_manifest"
    rm -f "$local_manifest"
    mv "$chunk_manifest_tmp" "$chunk_manifest"
    printf 'done\n' > "$done_file"
    log "chunk manifest written: $chunk_manifest"
}

# createdb sorts inputs by basename with a non-stable sort, so duplicate basenames must be renamed
resolve_chunk_filelist() {
    local filelist="$1" work_dir="$2" out_list="$3"
    local mem cnt i=0 local_ref base ext
    : > "$out_list"
    while IFS=$'\t' read -r mem cnt || [[ -n "${mem:-}" ]]; do
        [[ -z "${mem:-}" || "$mem" =~ ^[[:space:]]*# ]] && continue
        base=$(basename "$mem")
        ext=""
        case "$base" in
            *.zst) ext=".zst"; base="${base%.zst}" ;;
            *.gz)  ext=".gz";  base="${base%.gz}" ;;
        esac
        # keep the source's real extension (input-00000000.fa, input-00000001.fa.zst, ...)
        [[ "$base" == *.* ]] && ext=".${base##*.}${ext}"
        printf -v local_ref '%s/input-%08d%s' "$work_dir" "$i" "$ext"
        if is_s3 "$mem"; then
            copy_in "$mem" "$local_ref"      # download, keep .zst/.gz compression
        else
            # the link sits in another directory, so a relative path would resolve against that one
            ln -sf "$(cd -- "$(dirname -- "$mem")" && pwd -P)/$(basename -- "$mem")" "$local_ref"
        fi
        # a declared count rides along so createdb can parallelise its parse on 32-bit keys too
        if [[ -n "${cnt:-}" ]]; then
            printf '%s\t%s\n' "$local_ref" "$cnt" >> "$out_list"
        else
            printf '%s\n' "$local_ref" >> "$out_list"
        fi
        i=$((i + 1))
    done < <(stream_manifest "$filelist")
    [[ -s "$out_list" ]] || fail "empty input list for chunk: $filelist"
}

# the worker entry points exec this script from inside the work dir, so keep the script and the materialized chunk
scrub_chunk_workdir() {
    local d="$1" f
    [[ -n "$d" && -d "$d" ]] || return 0
    rm -rf "$d/cluster-db" "$d/cluster-tmp" "$d/output"
    for f in "$d"/*; do
        [[ -e "$f" || -L "$f" ]] || continue
        case "$f" in
            "$d"/input-*) ;;
            *.fa | "$BATCH_SCRIPT") continue ;;
        esac
        rm -rf "$f"
    done
    rmdir "$d" 2>/dev/null || true
}

# apparent bytes under a path; du walks inodes only (no data reads), so this is cheap even at TB scale
du_bytes() {
    local path="$1"
    [[ -e "$path" || -L "$path" ]] || { printf '0\n'; return 0; }
    du -sb "$path" 2>/dev/null | awk '{printf "%.0f\n", $1}'
}

df_avail() {
    if df -B1 --output=avail "$1" >/dev/null 2>&1; then
        df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
    else
        df -k -P "$1" 2>/dev/null | awk 'NR==2{printf "%.0f", $4*1024}'
    fi
}

# summed RSS (KiB) of a pid and its descendants; stays fork-safe
rss_tree_kib() {
    local root="$1"
    ps -e -o pid=,ppid=,rss= 2>/dev/null | awk -v r="$root" '
        { pid=$1; parent[pid]=$2; rss[pid]=$3 }
        END {
            keep[r]=1; changed=1
            while (changed) {
                changed=0
                for (pid in parent) {
                    if (!(pid in keep) && (parent[pid] in keep)) { keep[pid]=1; changed=1 }
                }
            }
            for (pid in keep) sum += rss[pid]
            printf "%.0f\n", sum+0
        }'
}

# DEBUG-METRICS: wall time and peak RSS are for tuning only; the disk guard below is not
ram_poll() {
    local root="$1" out="$2" iv="$3" max=0 cur
    printf '0\n' > "$out"
    while kill -0 "$root" 2>/dev/null; do
        cur=$(rss_tree_kib "$root" || true)
        [[ "${cur:-0}" -gt "$max" ]] && max=$cur
        printf '%s\n' "$max" > "$out"
        sleep "$iv"
    done
    printf '%s\n' "$max" > "$out"
}

# tracks peak bytes under dir; when budget > 0 and exceeded, records the size and kills the stage
disk_poll() {
    local root="$1" out="$2" dir="$3" budget="$4" killflag="$5" iv="$6" max=0 cur
    printf '0\n' > "$out"
    while kill -0 "$root" 2>/dev/null; do
        cur=$(du_bytes "$dir" || true)
        cur=${cur:-0}
        [[ "$cur" -gt "$max" ]] && max=$cur
        printf '%s\n' "$max" > "$out"
        if [[ "$budget" -gt 0 && "$cur" -ge "$budget" ]]; then
            printf '%s\n' "$cur" > "$killflag"
            kill -TERM "$root" 2>/dev/null
            sleep 5
            kill -KILL "$root" 2>/dev/null
            break
        fi
        sleep "$iv"
    done
    printf '%s\n' "$max" > "$out"
}

# run_stage <stage> <chunk_id> <round> -- <command...>: record wall/peak-RSS/peak-disk and enforce the round's disk budget
run_stage() {
    local stage="$1" chunk_id="$2" round="$3"
    shift 3
    [[ "${1:-}" == "--" ]] && shift
    local poll="$CHUNK_WORK_DIR/poll-state"
    mkdir -p "$poll"
    local rf="$poll/$stage.rss" df_="$poll/$stage.disk" tf="$poll/$stage.time" kf="$poll/$stage.kill"
    rm -f "$rf" "$df_" "$tf" "$kf"
    local t0 t1 pid rc=0
    local -a pollers=()
    t0=$(date +%s)
    # /usr/bin/time gives an exact max RSS where it exists; it cannot exec a shell function
    if [[ "${HAVE_GNU_TIME:-0}" == "1" ]] && ! declare -F "$1" >/dev/null; then
        /usr/bin/time -v -o "$tf" "$@" &
    else
        "$@" &
    fi
    pid=$!
    ram_poll "$pid" "$rf" "$RAM_POLL_SEC" & pollers+=("$!")
    disk_poll "$pid" "$df_" "$CHUNK_WORK_DIR" "$CHUNK_DISK_BUDGET_BYTES" "$kf" "$DISK_POLL_SEC" & pollers+=("$!")
    wait "$pid" || rc=$?
    kill "${pollers[@]}" 2>/dev/null || true
    wait "${pollers[@]}" 2>/dev/null || true
    t1=$(date +%s)
    local rss trss disk final avail
    rss=$(head -1 "$rf" 2>/dev/null || true); rss=${rss:-0}
    if [[ -s "$tf" ]]; then
        trss=$(awk -F: '/Maximum resident set size/ { gsub(/[^0-9]/, "", $2); print $2; exit }' "$tf" 2>/dev/null || true)
        [[ "${trss:-0}" -gt "$rss" ]] && rss=$trss
    fi
    disk=$(head -1 "$df_" 2>/dev/null || true); disk=${disk:-0}
    final=$(du_bytes "$CHUNK_WORK_DIR" || true); final=${final:-0}
    [[ "$final" -gt "$disk" ]] && disk=$final
    avail=$(df_avail "$CHUNK_WORK_DIR" || true); avail=${avail:-0}
    STAGE_METRIC_LINES+=("$(printf 'stage\t%s\t%s\t%s\t%s' "$stage" "$((t1 - t0))" "$rss" "$disk")")
    [[ "$rss" -gt "$CHUNK_PEAK_RSS_KIB" ]] && CHUNK_PEAK_RSS_KIB=$rss
    [[ "$disk" -gt "$CHUNK_PEAK_DISK_BYTES" ]] && CHUNK_PEAK_DISK_BYTES=$disk
    CHUNK_FS_AVAIL_BYTES=$avail
    log "metrics ${chunk_id} ${stage}: wall_s=$((t1 - t0)) peak_rss_kib=${rss} peak_disk_bytes=${disk} work_fs_avail_bytes=${avail}"
    if [[ -e "$kf" ]]; then
        local seen; seen=$(head -1 "$kf" 2>/dev/null || true)
        fail "disk budget exceeded (chunk ${chunk_id}, round ${round}, stage ${stage}): observed ${seen:-?} bytes under ${CHUNK_WORK_DIR} >= budget ${CHUNK_DISK_BUDGET_BYTES} bytes; stage killed, no done-marker written (work fs avail: ${avail} bytes)"
    fi
    return "$rc"
}

# one stage covering both representative steps, matching the 'representatives' log line
representatives_stage() {
    local mmseqs_bin="$1" clu="$2" db="$3" rep="$4" rep_fa="$5" rep_fasta_splits="$6"
    "$mmseqs_bin" createsubdb "$clu" "$db" "$rep" --subdb-mode 1 || return 1
    "$mmseqs_bin" convert2fasta "$rep" "$rep_fa" --fasta-splits "$rep_fasta_splits"
}

cluster_chunk() {
    [[ "$#" -ge 3 && "$#" -le 6 ]] || usage
    local chunk_uri="$1"
    local result_prefix="$2"
    local work_dir="$3"
    local chunk_id
    chunk_id="${4:-$(basename_no_compression "$chunk_uri")}"
    local round="${5:-${BATCH_WORKER_ROUND:-0}}"
    # round must be numeric: the round_* helpers' arithmetic test would silently treat anything else as 0
    [[ "$round" =~ ^[0-9]+$ ]] || fail "cluster-chunk: round must be a non-negative integer (got '$round')"
    local expected_seqs="${6:-${BATCH_EXPECTED_SEQS:-0}}"
    local prefix
    prefix=$(normalize_s3_prefix "$result_prefix")
    local done_uri="${prefix}done/${chunk_id}.done"

    if done_exists "$done_uri"; then
        log "cluster-chunk: reusing completed ${chunk_id}"
        return 0
    fi

    # leftovers here are from a failed previous attempt, so scrub before starting
    scrub_chunk_workdir "$work_dir"
    mkdir -p "$work_dir/cluster-db" "$work_dir/cluster-tmp" "$work_dir/output"
    # scrub on any exit so a chunk that gives up never leaves scratch behind
    CLUSTER_CHUNK_WORKDIR="$work_dir"
    [[ -n "${REMOVE_TMP:-}" ]] && trap 'scrub_chunk_workdir "$CLUSTER_CHUNK_WORKDIR"' EXIT INT TERM HUP

    # mode 0 reads FASTA/.zst directly, mode 1 materializes one node-local FASTA first
    local createdb_input
    case "$chunk_uri" in
        # *.filelist.tsv is the pre-rename spelling, kept so an old run's chunk manifest still resumes
        *.inputs.list|*.filelist.tsv)
            if round_createdb_softlink "$round"; then
                createdb_input="$work_dir/${chunk_id}.fa"
                materialize_softlink_filelist "$chunk_uri" "$createdb_input"
            else
                # createdb only reads a file-of-files when it ends in .tsv (createdb.cpp), so the suffix is load-bearing
                createdb_input="$work_dir/inputs.list.tsv"
                resolve_chunk_filelist "$chunk_uri" "$work_dir" "$createdb_input"
            fi
            ;;
        *)
            if round_createdb_softlink "$round"; then
                createdb_input="$work_dir/${chunk_id}.fa"
                materialize_softlink_fasta "$chunk_uri" "$createdb_input"
            else
                case "$chunk_uri" in
                    s3://*)
                        createdb_input="$work_dir/$(basename "$chunk_uri")"
                        copy_in "$chunk_uri" "$createdb_input"
                        ;;
                    *)
                        createdb_input="$chunk_uri"
                        ;;
                esac
            fi
            ;;
    esac

    local db="$work_dir/cluster-db/seq"
    local clu="$work_dir/cluster-db/clu"
    local rep="$work_dir/cluster-db/rep"
    local tsv_prefix="$work_dir/output/${chunk_id}.cluster"
    local rep_fa="$work_dir/output/${chunk_id}.rep.fa"
    local metrics="$work_dir/output/${chunk_id}.metrics.tsv"

    # resolve the round's mmseqs binary, so a round0 node pool of another architecture runs its own
    local mmseqs_bin
    mmseqs_bin=$(round_mmseqs "$round")
    need_cmd "$mmseqs_bin"
    # cluster-chunk runs as its own process, so every $THREADS use below can follow the round's budget
    THREADS=$(round_threads "$round")
    CHUNK_WORK_DIR="$work_dir"
    CHUNK_DISK_BUDGET_BYTES=$(round_chunk_disk_budget "$round")
    [[ "$CHUNK_DISK_BUDGET_BYTES" =~ ^[0-9]+$ ]] || fail "chunk disk budget must be an integer byte count (got '$CHUNK_DISK_BUDGET_BYTES')"
    CHUNK_PEAK_RSS_KIB=0
    CHUNK_PEAK_DISK_BYTES=0
    CHUNK_FS_AVAIL_BYTES=0
    STAGE_METRIC_LINES=()
    HAVE_GNU_TIME=0
    if [[ -x /usr/bin/time ]] && /usr/bin/time -v true >/dev/null 2>&1; then
        HAVE_GNU_TIME=1
    fi
    log "createdb ${chunk_id}"
    # shellcheck disable=SC2046,SC2086
    run_stage createdb "$chunk_id" "$round" -- "$mmseqs_bin" createdb "$createdb_input" "$db" $(round_createdb_par "$round") || fail "createdb failed (chunk ${chunk_id}, rc=$?)"
    local actual_seqs
    actual_seqs=$(wc -l < "${db}.index" | tr -d ' ')
    if [[ "${expected_seqs:-0}" =~ ^[1-9][0-9]*$ && "$actual_seqs" -ne "$expected_seqs" ]]; then
        fail "createdb sequence-count mismatch for ${chunk_id}: manifest=${expected_seqs}, db.index=${actual_seqs}. Input may be truncated or corrupt."
    fi
    if round_createdb_softlink "$round" && [[ ! -L "$db" || ! -L "${db}_h" ]]; then
        fail "createdb-mode 1 fell back to a copied DB for ${chunk_id}. Batch softlink mode requires plain single-line FASTA; refusing silent NVMe expansion."
    fi
    # staged inputs feed only createdb (the DB is a full copy here); reclaim them so S3 downloads stop charging the disk budget
    if [[ "$createdb_input" == "$work_dir/inputs.list.tsv" && -f "$createdb_input" ]]; then
        local staged _cnt
        while IFS=$'\t' read -r staged _cnt || [[ -n "${staged:-}" ]]; do
            # only ever unlink what resolve_chunk_filelist staged: a source path here would delete the user's input
            if [[ "$staged" == "$work_dir"/input-* ]]; then rm -f "$staged"; fi
        done < "$createdb_input"
        rm -f "$createdb_input"
    fi
    local cluster_cmd
    cluster_cmd=$(round_cluster_cmd "$round")
    log "${cluster_cmd} ${chunk_id}"
    # shellcheck disable=SC2046,SC2086
    run_stage "$cluster_cmd" "$chunk_id" "$round" -- "$mmseqs_bin" "$cluster_cmd" "$db" "$clu" "$work_dir/cluster-tmp" $(round_cluster_par "$round") || fail "${cluster_cmd} failed (chunk ${chunk_id}, rc=$?)"

    log "createtsv ${chunk_id}"
    # round0 TSVs feed the merge join as child (key = rep, col 1); round1+ TSVs as parent (key = member, col 2)
    local split_col=2
    [[ "$round" -eq 0 ]] && split_col=1
    local createtsv_par
    createtsv_par=$(par_without_flag "$CREATETSV_PAR" --tsv-splits)
    createtsv_par=$(par_without_flag "$createtsv_par" --tsv-split-column)
    createtsv_par=$(round_par_threads "$round" "$createtsv_par")
    # shellcheck disable=SC2086
    run_stage createtsv "$chunk_id" "$round" -- "$mmseqs_bin" createtsv "$db" "$db" "$clu" "$tsv_prefix" ${createtsv_par} --tsv-splits "$MERGE_SPLITS" --tsv-split-column "$split_col" || fail "createtsv failed (chunk ${chunk_id}, rc=$?)"

    log "representatives ${chunk_id}"
    # one index line per cluster, and one representative per cluster, so this needs no fasta scan
    local rep_total
    rep_total=$(wc -l < "${clu}.index" | tr -d ' ')
    # cap the shard count at the rep count so no shard is empty
    local rep_fasta_splits
    rep_fasta_splits=$(round_rep_fasta_splits "$round")
    [[ "$rep_fasta_splits" -gt "$rep_total" ]] && rep_fasta_splits="$rep_total"
    [[ "$rep_fasta_splits" -ge 1 ]] || rep_fasta_splits=1
    run_stage representatives "$chunk_id" "$round" -- representatives_stage "$mmseqs_bin" "$clu" "$db" "$rep" "$rep_fa" "$rep_fasta_splits" || fail "representatives (createsubdb/convert2fasta) failed (chunk ${chunk_id}, rc=$?)"

    local batch_suffix b split_tsv tsv_out
    batch_suffix=$(batch_compression_suffix)

    local k shard shard_out shard_bytes shard_seqs rep_bytes_total=0
    local -a shard_outs=() shard_metric_lines=()
    for ((k = 0; k < rep_fasta_splits; k++)); do
        printf -v shard '%s/output/%s.rep.split%05d.fa' "$work_dir" "$chunk_id" "$k"
        [[ -s "$shard" ]] || fail "convert2fasta left no shard $shard (chunk ${chunk_id})"
        shard_bytes=$(wc -c < "$shard" | tr -d ' ')
        # rep DB entry i lands in shard i % rep_fasta_splits, so the per-shard count is arithmetic
        shard_seqs=$(( (rep_total - 1 - k) / rep_fasta_splits + 1 ))
        rep_bytes_total=$((rep_bytes_total + shard_bytes))
        shard_out="${shard}${batch_suffix}"
        if compress_batch_outputs_enabled; then
            write_batch_output "$shard" "$shard_out"
        fi
        shard_outs+=("$shard_out")
        shard_metric_lines+=("$(printf 'rep_shard\t%s\t%s\t%s' "$(basename "$shard_out")" "$shard_bytes" "$shard_seqs")")
    done

    {
        printf 'chunk_id\t%s\n' "$chunk_id"
        printf 'input_uri\t%s\n' "$chunk_uri"
        printf 'seq_count\t%s\n' "$actual_seqs"
        printf 'input_bytes\t%s\n' "$(wc -c < "$db" 2>/dev/null | tr -d ' ' || echo 0)"
        printf 'rep_count\t%s\n' "$rep_total"
        printf 'rep_bytes\t%s\n' "$rep_bytes_total"
        printf 'cluster_tsv\t%s\n' "$(basename "$tsv_prefix").split*.tsv${batch_suffix}"
        printf 'tsv_splits\t%s\n' "$MERGE_SPLITS"
        printf 'rep_splits\t%s\n' "$rep_fasta_splits"
        printf '%s\n' "${shard_metric_lines[@]}"
        printf 'round\t%s\n' "$round"
        printf 'disk_budget_bytes\t%s\n' "$CHUNK_DISK_BUDGET_BYTES"
        # DEBUG-METRICS: this key and the stage lines below go when the tuning is done
        printf 'peak_rss_kib\t%s\n' "$CHUNK_PEAK_RSS_KIB"
        printf 'peak_disk_bytes\t%s\n' "$CHUNK_PEAK_DISK_BYTES"
        printf 'work_fs_avail_bytes\t%s\n' "$CHUNK_FS_AVAIL_BYTES"
        printf '%s\n' "${STAGE_METRIC_LINES[@]}"
    } > "$metrics"

    for ((b = 0; b < MERGE_SPLITS; b++)); do
        printf -v split_tsv '%s.split%05d.tsv' "$tsv_prefix" "$b"
        [[ -e "$split_tsv" ]] || fail "createtsv left no split file $split_tsv (chunk ${chunk_id})"
        tsv_out="${split_tsv}${batch_suffix}"
        if compress_batch_outputs_enabled; then
            write_batch_output "$split_tsv" "$tsv_out"
        fi
        copy_out "$tsv_out" "${prefix}tsv/$(basename "$tsv_out")"
    done
    for shard_out in "${shard_outs[@]}"; do
        copy_out "$shard_out" "${prefix}rep/$(basename "$shard_out")"
    done
    copy_out "$metrics" "${prefix}metrics/$(basename "$metrics")"
    mark_done "$done_uri" "$work_dir/output/${chunk_id}.done"

    # reclaim the shared source chunk once its done-marker is written, so the chunk dir shrinks as the round runs
    if [[ -n "${REMOVE_TMP:-}" && "${BATCH_DELETE_SOURCE_CHUNK:-0}" == "1" && "$chunk_uri" != s3://* && -f "$chunk_uri" ]]; then
        rm -f "$chunk_uri"
    fi

    # free this chunk's working set before the next chunk's createdb; with REMOVE_TMP off the dir is kept
    [[ -n "${REMOVE_TMP:-}" ]] && rm -rf "${work_dir:?}"
    # rmdir, never rm -rf: it fails on a non-empty parent, so a sibling chunk still running is safe
    if [[ -n "${REMOVE_TMP:-}" ]]; then
        rmdir "$(dirname "$work_dir")" 2>/dev/null && rmdir "$(dirname "$(dirname "$work_dir")")" 2>/dev/null
    fi
    log "cluster-chunk complete: ${chunk_id}"
}


# compose child(rep->member) with parent(rep->member) on parent.member == child.rep

# byte-hash of a column into [0, B); must stay byte-identical to tsvSplitOfColumn in createtsv.cpp
SPLIT_HASH_AWK='
    function split_of(key,   i, h, n) {
        h = 0; n = length(key)
        for (i = 1; i <= n; i++) h = (h * 131 + ord[substr(key, i, 1)]) % B
        return h
    }
    BEGIN { for (i = 0; i < 256; i++) ord[sprintf("%c", i)] = i }
'

# A listed split index >= MERGE_SPLITS means the TSVs were written with MORE splits; joining with fewer would silently drop them.
check_manifest_split_count() {
    local manifest="$1" splits="$2" bad
    bad=$(stream_manifest "$manifest" | awk -F'\t' -v B="$splits" '
        {
            n = split($1, parts, /\.split/)
            if (n < 2) next
            idx = parts[n]
            sub(/\.tsv(\.zst|\.gz)?$/, "", idx)
            if (idx ~ /^[0-9]+$/ && idx + 0 >= B) { print idx + 0; exit }
        }')
    [[ -z "$bad" ]] || fail "manifest $manifest lists TSV split ${bad} but this run uses --merge-splits ${splits}: the TSVs were written with more splits, and joining with fewer would silently drop cluster members (one split count must cover the whole run)"
}

# Print the manifest entries of one split (*.split<bb>.tsv[.zst|.gz]); zero matches = split-count mismatch.
split_manifest_files() {
    local manifest="$1" bb="$2" uri _rest found=0
    while IFS=$'\t' read -r uri _rest || [[ -n "${uri:-}" ]]; do
        [[ -z "${uri:-}" || "$uri" =~ ^[[:space:]]*# ]] && continue
        case "$(basename "$uri")" in
            *".split${bb}.tsv"|*".split${bb}.tsv.zst"|*".split${bb}.tsv.gz") printf '%s\n' "$uri"; found=1 ;;
        esac
    done < <(stream_manifest "$manifest")
    [[ "$found" -eq 1 ]] || fail "no split ${bb} files listed in manifest $manifest (merge-splits mismatch with the run that wrote it?)"
}

# Concatenate one split's rows from every file the manifest lists for it.
stream_split_files() {
    local manifest="$1" bb="$2" list uri
    list=$(split_manifest_files "$manifest" "$bb")
    while IFS= read -r uri; do
        stream_uri "$uri"
    done <<< "$list"
}

run_split_jobs() {
    local label="$1" splits="$2" jobs="$3" worker="$4"
    shift 4
    [[ "$jobs" -gt "$splits" ]] && jobs="$splits"
    [[ "$jobs" -lt 1 ]] && jobs=1

    local pids="" active=0
    local b pid rc=0
    for ((b = 0; b < splits; b++)); do
        "$worker" "$@" "$b" &
        pids="${pids}$! "
        active=$((active + 1))
        if [[ "$active" -ge "$jobs" ]]; then
            for pid in $pids; do
                if ! wait "$pid"; then
                    rc=1
                fi
            done
            pids=""
            active=0
            [[ "$rc" -eq 0 ]] || fail "${label}: one or more split jobs failed"
        fi
    done
    for pid in $pids; do
        if ! wait "$pid"; then
            rc=1
        fi
    done
    [[ "$rc" -eq 0 ]] || fail "${label}: one or more split jobs failed"
}

finalize_emit_shard_job() {
    local sorted_dir="$1" shard_prefix="$2" work_dir="$3" split_idx="$4"
    local bb batch_suffix split_file shard_out shard_tmp
    printf -v bb '%05d' "$split_idx"
    batch_suffix=$(batch_compression_suffix)
    printf -v split_file '%s/final.split%s.tsv' "$sorted_dir" "$bb"
    [[ -e "$split_file" ]] || fail "finalize: missing sorted split $split_file"
    shard_out="${shard_prefix}final.split${bb}.tsv${batch_suffix}"
    shard_tmp="$(resolve_node_scratch "$work_dir")/final.split${bb}.tsv${batch_suffix}.out.tmp.$$"
    write_batch_output "$split_file" "$shard_tmp"
    copy_out "$shard_tmp" "$shard_out"
    rm -f "$shard_tmp"
}

# merge_join <child_manifest> <parent_manifest> <frag_dir> <sort_tmp> <split>: join parent.member == child.rep, routing output by hash(new rep) so it leaves already split.
merge_join() {
    local child_manifest="$1" parent_manifest="$2" frag_dir="$3" sort_tmp="$4" split_idx="$5"
    local splits="${MERGE_SPLITS:-1}"
    local bb; printf -v bb '%05d' "$split_idx"
    sort_tmp="$sort_tmp/split${bb}"
    mkdir -p "$frag_dir" "$sort_tmp"
    local child_sorted="$frag_dir/.child.split${bb}.by_rep"
    local parent_sorted="$frag_dir/.parent.split${bb}.by_member"
    local joined_cnt="$frag_dir/.joined.split${bb}.count"
    # shellcheck disable=SC2086
    stream_split_files "$child_manifest" "$bb" | sort -T "$sort_tmp" -t $'\t' -k1,1 -o "$child_sorted" $SORT_PARALLEL_OPT
    # shellcheck disable=SC2086
    stream_split_files "$parent_manifest" "$bb" | sort -T "$sort_tmp" -t $'\t' -k2,2 -o "$parent_sorted" $SORT_PARALLEL_OPT
    join -t $'\t' -1 2 -2 1 -o '1.1 2.2' "$parent_sorted" "$child_sorted" \
        | awk -F'\t' -v B="$splits" -v pfx="$frag_dir/joined.b${bb}.k" -v cnt="$joined_cnt" \
              "$SPLIT_HASH_AWK"'
        BEGIN { for (k = 0; k < B; k++) printf "" > (pfx sprintf("%05d.tsv", k)) }
        { print > (pfx sprintf("%05d.tsv", split_of($1))) }
        END { print NR > cnt }
    '

    # every child rep finds its parent exactly once, because child carries a self-link for each previous rep
    local child_count joined_count
    child_count=$(awk 'END { print NR + 0 }' "$child_sorted")
    read -r joined_count < "$joined_cnt"
    [[ "$joined_count" -eq "$child_count" ]] ||
        fail "merge_join split ${bb} lost cluster members: child=${child_count} joined=${joined_count}"
    rm -f "$child_sorted" "$parent_sorted" "$joined_cnt"
    log "merge_join split ${bb}: ${joined_count} members"
}

# merge_emit_shard <frag_dir> <out_dir> <split>: concatenate this rep-split's join fragments into one shard.
merge_emit_shard() {
    local frag_dir="$1" out_dir="$2" split_idx="$3"
    local splits="${MERGE_SPLITS:-1}"
    local kk; printf -v kk '%05d' "$split_idx"
    local out
    out="$out_dir/propagated.split${kk}.tsv$(batch_compression_suffix)"
    local tmp="${out}.tmp.$$"
    local b frag
    local -a frags=()
    for ((b = 0; b < splits; b++)); do
        printf -v frag '%s/joined.b%05d.k%s.tsv' "$frag_dir" "$b" "$kk"
        [[ -e "$frag" ]] || fail "propagate: missing joined fragment $frag"
        frags+=("$frag")
    done
    # temp + rename: a crash mid-write must never leave a truncated shard that looks complete
    mkdir -p "$out_dir"
    if compress_batch_outputs_enabled; then
        need_cmd pzstd
        cat "${frags[@]}" | pzstd -p "$THREADS" -3 -c > "$tmp"
    else
        cat "${frags[@]}" > "$tmp"
    fi
    mv -f "$tmp" "$out"
}

# finalize_sort_split <mapping_manifest> <out_dir> <sort_tmp> <split>: the rep's self-link leads its cluster
finalize_sort_split() {
    local mapping_manifest="$1" out_dir="$2" sort_tmp="$3" split_idx="$4"
    local bb; printf -v bb '%05d' "$split_idx"
    local out="$out_dir/final.split${bb}.tsv"
    sort_tmp="$sort_tmp/split${bb}"
    mkdir -p "$out_dir" "$sort_tmp"
    # shellcheck disable=SC2086
    stream_split_files "$mapping_manifest" "$bb" \
        | awk 'BEGIN {FS=OFS="\t"} { print $1, ($1 == $2 ? 0 : 1), $2 }' \
        | sort -T "$sort_tmp" -t $'\t' -k1,1 -k2,2n -k3,3 $SORT_PARALLEL_OPT \
        | awk 'BEGIN {FS=OFS="\t"} { print $1, $3 }' \
        > "$out"
}

# propagate <child_manifest> <parent_manifest> <out_manifest> <work_dir>
propagate() {
    [[ "$#" -eq 4 ]] || usage
    local child_manifest="$1"
    local parent_manifest="$2"
    local out_manifest="$3"
    local work_dir="$4"
    local done_uri="${out_manifest}.done"
    local splits="${MERGE_SPLITS:-1}"
    local sort_tmp; sort_tmp=$(resolve_sort_tmp "$work_dir")

    if [[ -s "$out_manifest" ]] && done_exists "$done_uri"; then
        log "propagate: reusing completed $out_manifest"
        return 0
    fi

    # fragments are node-local scratch; shards are this round's output and must survive on shared storage
    local frag; frag=$(resolve_node_scratch "$work_dir" fragments)
    local shards="$work_dir/shards"
    rm -rf "${frag:?}" "${shards:?}" "${sort_tmp:?}"
    mkdir -p "$work_dir" "$sort_tmp" "$frag" "$shards"

    check_manifest_split_count "$child_manifest" "$splits"
    check_manifest_split_count "$parent_manifest" "$splits"
    log "propagate: joining ${splits} split(s), ${MERGE_SPLIT_JOBS} at a time"
    local b
    run_split_jobs "propagate join" "$splits" "$MERGE_SPLIT_JOBS" merge_join "$child_manifest" "$parent_manifest" "$frag" "$sort_tmp"

    log "propagate: assembling ${splits} shard(s), ${MERGE_SPLIT_JOBS} at a time"
    run_split_jobs "propagate shard write" "$splits" "$MERGE_SPLIT_JOBS" merge_emit_shard "$frag" "$shards"

    local split_file
    : > "${out_manifest}.tmp"
    for ((b = 0; b < splits; b++)); do
        printf -v split_file '%s/propagated.split%05d.tsv%s' "$shards" "$b" "$(batch_compression_suffix)"
        [[ -e "$split_file" ]] || fail "propagate: missing split output $split_file"
        printf '%s\n' "$split_file" >> "${out_manifest}.tmp"
    done
    mv "${out_manifest}.tmp" "$out_manifest"
    mark_done "$done_uri" "$work_dir/propagate.done"
    if [[ -n "${REMOVE_TMP:-}" ]]; then
        rm -rf "$frag" "$sort_tmp"
    fi
    log "propagated clusters written: ${splits} shard(s) -> $out_manifest"
}

# filenames are deterministic, so the expected list needs no directory listing
make_manifest_from_chunk_ids() {
    local chunk_manifest="$1"
    local dir="$2"
    local suffix="$3"
    local out="$4"
    : > "$out"
    local cid _rest
    while IFS=$'\t' read -r cid _rest || [[ -n "${cid:-}" ]]; do
        [[ -z "${cid:-}" || "$cid" =~ ^# ]] && continue
        printf '%s/%s%s\n' "$dir" "$cid" "$suffix" >> "$out"
    done < "$chunk_manifest"
}

# make_manifest_from_chunk_ids for cluster TSVs: each chunk lists its MERGE_SPLITS <chunk_id>.cluster.split%05d.tsv[.zst] files.
make_split_tsv_manifest() {
    local chunk_manifest="$1"
    local dir="$2"
    local out="$3"
    local suffix; suffix=$(batch_compression_suffix)
    : > "$out"
    local cid _rest b
    while IFS=$'\t' read -r cid _rest || [[ -n "${cid:-}" ]]; do
        [[ -z "${cid:-}" || "$cid" =~ ^# ]] && continue
        for ((b = 0; b < MERGE_SPLITS; b++)); do
            printf '%s/%s.cluster.split%05d.tsv%s\n' "$dir" "$cid" "$b" "$suffix" >> "$out"
        done
    done < "$chunk_manifest"
}

# one entry per rep shard as path/bytes/seqs, so the next round's prepare needs no size probe
make_rep_manifest_with_sizes() {
    local chunk_manifest="$1"
    local clustered="$2"
    local out="$3"
    local round="$4"
    local plain="${out}.plain.$$"
    : > "$plain"
    local cid _rest repf metricsf bytes seqs rep_suffix
    rep_suffix=".rep.fa$(batch_compression_suffix)"
    while IFS=$'\t' read -r cid _rest || [[ -n "${cid:-}" ]]; do
        [[ -z "${cid:-}" || "$cid" =~ ^# ]] && continue
        metricsf="$clustered/metrics/${cid}.metrics.tsv"
        # one manifest entry per rep shard; each shard's exact bytes/seqs come from the metrics
        if awk -F'\t' '$1 == "rep_shard" { found = 1 } END { exit !found }' "$metricsf" 2>/dev/null; then
            awk -F'\t' -v dir="$clustered/rep" '$1 == "rep_shard" { printf "%s/%s\t%s\t%s\n", dir, $2, $3 + 0, $4 + 0 }' "$metricsf" >> "$plain"
            continue
        fi
        # single-file metrics written before rep sharding: resume support for old work areas
        repf="$clustered/rep/${cid}${rep_suffix}"
        bytes=$(awk -F'\t' '$1 == "rep_bytes" { print $2 + 0; exit }' "$metricsf" 2>/dev/null || true)
        seqs=$(awk -F'\t' '$1 == "rep_count" { print $2 + 0; exit }' "$metricsf" 2>/dev/null || true)
        # only pre-rep_bytes metrics need this: it decompresses the rep, or estimates from --threads-dependent compressed size
        [[ "$bytes" =~ ^[0-9]+$ && "$bytes" -gt 0 ]] || bytes=$(uncompressed_bytes "$repf")
        [[ "$seqs" =~ ^[0-9]+$ ]] || seqs=0
        printf '%s\t%s\t%s\n' "$repf" "${bytes:-0}" "$seqs" >> "$plain"
    done < "$chunk_manifest"
    interleave_manifest_file "$plain" "$out" "$round"
    rm -f "$plain"
}

# take bytes and seqs from the per-chunk metrics instead of stat-ing the rep FASTAs
make_rep_manifest_from_metrics() {
    local metrics_manifest="$1" rep_dir="$2" out="$3" round="$4"
    local plain="${out}.plain.$$"
    : > "$plain"
    local uri
    while IFS= read -r uri || [[ -n "${uri:-}" ]]; do
        [[ -z "${uri:-}" || "$uri" =~ ^# ]] && continue
        stream_uri "$uri" | awk -F'\t' -v dir="$rep_dir" '
            $1 == "rep_shard" { printf "%s/%s\t%s\t%s\n", dir, $2, $3 + 0, $4 + 0; shards++ }
            $1 == "rep_fasta" { fa = $2 }
            $1 == "rep_bytes" { b  = $2 }
            $1 == "rep_count" { s  = $2 }
            END {
                if (shards > 0) { exit }
                # single-file metrics written before rep sharding: resume support for old work areas
                if (fa == "") { print "batch: metrics file has no rep_shard or rep_fasta line" > "/dev/stderr"; exit 1 }
                printf "%s/%s\t%s\t%s\n", dir, fa, (b == "" ? 0 : b), (s == "" ? 0 : s)
            }' >> "$plain"
    done < <(stream_manifest "$metrics_manifest")
    interleave_manifest_file "$plain" "$out" "$round"
    rm -f "$plain"
}

# stride-deal permutation, a pure function of (entry count, round): same-chunk shards land ~N/stride apart, and the stride moves with the round so successive rounds pack different shards together
interleave_manifest_file() {
    local input="$1"
    local out="$2"
    local round="${3:?interleave_manifest_file needs a round}"
    awk -v round="$round" '
        { entries[NR] = $0 }
        END {
            if (NR == 0) { exit }
            stride = int(sqrt(NR)) + round
            stride = (NR > 1) ? ((stride - 1) % (NR - 1)) + 1 : 1
            for (b = 0; b < stride; b++) {
                for (i = b + 1; i <= NR; i += stride) { print entries[i] }
            }
        }
    ' "$input" > "$out"
}

count_manifest_rows() {
    stream_manifest "$1" | awk 'NF && $1 !~ /^#/ { n++ } END { print n + 0 }'
}

validate_clustered_outputs() {
    local chunk_manifest="$1"
    local clustered="$2"
    # check per chunk_id, not file counts: every chunk needs its done-marker
    local missing
    missing=$(list_missing_chunks "$chunk_manifest" "$clustered")
    [[ -z "$missing" ]] ||
        fail "round output incomplete; chunk(s) missing outputs: $(printf '%s' "$missing" | tr '\n' ' ')"
}


count_reps_from_metrics_manifest() {
    local manifest="$1"
    while IFS= read -r uri || [[ -n "${uri:-}" ]]; do
        [[ -z "${uri:-}" || "$uri" =~ ^# ]] && continue
        stream_uri "$uri"
    done < <(stream_manifest "$manifest") \
        | awk -F'\t' '$1 == "rep_count" { s += $2 } END { print s + 0 }'
}

write_state_file() {
    local out="$1"
    local prev_reps="$2"
    local low_benefit_rounds="$3"
    {
        printf 'PREV_REPS=%s\n' "$prev_reps"
        printf 'LOW_BENEFIT_ROUNDS=%s\n' "$low_benefit_rounds"
    } > "$out"
}

read_counter_uri() {
    local uri="$1"
    if done_exists "$uri"; then
        stream_uri "$uri" | awk 'NR == 1 { print $1 + 0; found = 1; exit } END { if (!found) print 0 }'
    else
        printf '0\n'
    fi
}

write_counter_file() {
    local out="$1"
    local value="$2"
    printf '%s\n' "$value" > "$out"
}

# TSVs are written pre-split, so a run must keep ONE split count: pin on first use, adopt on resume.
pin_merge_splits() {
    local store="$1" pinned tmp
    # a pre-rename work area pinned merge_buckets.txt and wrote .bkt TSVs; re-deriving a count here would silently break its join
    local legacy="${store%merge_splits.txt}merge_buckets.txt"
    if ! done_exists "$store" && done_exists "$legacy"; then
        fail "this work area was pinned by an older version ($legacy, .bkt file naming); finish it with that version or start a fresh work dir"
    fi
    pinned=$(read_counter_uri "$store")
    if [[ "$pinned" -gt 0 ]]; then
        if [[ "$pinned" != "$MERGE_SPLITS" ]]; then
            log "merge-splits pinned to ${pinned} by an earlier run of this work area (this run derived ${MERGE_SPLITS}); using ${pinned}"
            MERGE_SPLITS="$pinned"
            [[ "$MERGE_SPLIT_JOBS" -gt "$MERGE_SPLITS" ]] && MERGE_SPLIT_JOBS="$MERGE_SPLITS"
            check_split_fd_budget
        fi
    else
        tmp=$(mktemp)
        write_counter_file "$tmp" "$MERGE_SPLITS"
        copy_out "$tmp" "$store"
        rm -f "$tmp"
    fi
    export MERGE_SPLITS MERGE_SPLIT_JOBS
}

trim_field() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

# expand a SLURM nodelist, comma list or super[001-005] range, into SLURM_NODE_ARRAY
SLURM_NODE_ARRAY=()
build_slurm_node_array() {
    local round="${1:-1}"
    SLURM_NODE_ARRAY=()
    local list
    list=$(round_slurm_nodelist "$round")
    [[ -n "$list" ]] || return 0
    if [[ "$list" == *"["* ]]; then
        command -v scontrol >/dev/null 2>&1 || \
            fail "--slurm-nodelist uses bracket syntax ('$list') but 'scontrol' is unavailable to expand it; pass a comma-separated node list"
        local n
        while IFS= read -r n; do
            [[ -n "$n" ]] && SLURM_NODE_ARRAY+=("$n")
        done < <(scontrol show hostnames "$list")
        return 0
    fi
    local -a raw
    IFS=',' read -r -a raw <<< "$list"
    local n
    for n in "${raw[@]}"; do
        n=$(trim_field "$n")
        [[ -n "$n" ]] && SLURM_NODE_ARRAY+=("$n")
    done
}

first_slurm_node_for_round() {
    local round="$1"
    build_slurm_node_array "$round"
    [[ "${#SLURM_NODE_ARRAY[@]}" -gt 0 ]] || fail "empty node array for round ${round}"
    printf '%s' "${SLURM_NODE_ARRAY[0]}"
}

make_chunk_work_dir() {
    local round_work_dir="$1"
    local chunk_id="$2"
    local round="${3:-1}"
    local base
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    if [[ -n "$node_work_dir" ]]; then
        base="$node_work_dir/mmseqs-batch/round${round}"
    else
        base="$round_work_dir/chunk-work"
    fi
    printf '%s/%s' "$base" "$chunk_id"
}

# finalize <mapping_manifest> <rep_manifest> <result_prefix> <work_dir> [mark_final]
finalize_outputs() {
    [[ "$#" -ge 4 && "$#" -le 5 ]] || usage
    local mapping_manifest="$1"
    local rep_manifest="$2"
    local result_prefix="$3"
    local work_dir="$4"
    local mark_final="${5:-1}"
    local splits="${MERGE_SPLITS:-1}"
    local sort_tmp; sort_tmp=$(resolve_sort_tmp "$work_dir")
    local prefix
    prefix=$(normalize_s3_prefix "$result_prefix")

    mkdir -p "$work_dir"
    local batch_suffix
    batch_suffix=$(batch_compression_suffix)
    local final_cluster_manifest="${prefix}final_cluster_manifest.txt"
    local final_cluster_shard_prefix="${prefix}final_cluster_shards/"
    local final_rep="${prefix}final_rep_seq.fasta${batch_suffix}"
    local final_rep_manifest="${prefix}final_rep_seq_manifest.txt"
    local final_done="${prefix}final.done"
    local final_result
    final_result="${prefix}$(final_cluster_file_name)"

    if [[ "$mark_final" == "1" ]] && done_exists "$final_done" && done_exists "$final_result"; then
        log "finalize: reusing completed result $final_result"
        return 0
    fi

    # rep-sorted is transient, so keep it on node-local scratch instead of the shared FS
    local sorted; sorted=$(resolve_node_scratch "$work_dir" rep-sorted)
    rm -rf "${sorted:?}" "${sort_tmp:?}"
    mkdir -p "$sorted" "$sort_tmp"

    check_manifest_split_count "$mapping_manifest" "$splits"
    log "finalize: sorting ${splits} split(s) representative-first, ${MERGE_SPLIT_JOBS} at a time"
    local b
    run_split_jobs "finalize sort" "$splits" "$MERGE_SPLIT_JOBS" finalize_sort_split "$mapping_manifest" "$sorted" "$sort_tmp"

    local manifest_tmp="$work_dir/final_cluster_manifest.txt.tmp.$$"
    : > "$manifest_tmp"
    log "finalize: writing ${splits} final shard(s), ${MERGE_SPLIT_JOBS} at a time"
    run_split_jobs "finalize shard write" "$splits" "$MERGE_SPLIT_JOBS" finalize_emit_shard_job "$sorted" "$final_cluster_shard_prefix" "$work_dir"
    local shard_out bb
    for ((b = 0; b < splits; b++)); do
        printf -v bb '%05d' "$b"
        shard_out="${final_cluster_shard_prefix}final.split${bb}.tsv${batch_suffix}"
        printf '%s\n' "$shard_out" >> "$manifest_tmp"
    done
    copy_out "$manifest_tmp" "$final_cluster_manifest"
    rm -f "$manifest_tmp"

    if [[ -n "$rep_manifest" ]]; then
        local rep_tmp
        rep_tmp="$(resolve_node_scratch "$work_dir")/final_rep_seq.fasta${batch_suffix}.tmp.$$"
        : > "$rep_tmp"
        # rep_manifest may be 3-column or a bare path per line; take field 1 either way
        while IFS=$'\t' read -r repf _ || [[ -n "${repf:-}" ]]; do
            [[ -z "${repf:-}" || "$repf" =~ ^# ]] && continue
            append_raw_uri "$repf" "$rep_tmp"
        done < <(stream_manifest "$rep_manifest")
        copy_out "$rep_tmp" "$final_rep"
        rm -f "$rep_tmp"

        local rep_manifest_tmp="$work_dir/final_rep_seq_manifest.txt.tmp.$$"
        printf '%s\n' "$final_rep" > "$rep_manifest_tmp"
        copy_out "$rep_manifest_tmp" "$final_rep_manifest"
        rm -f "$rep_manifest_tmp"
        log "final representative FASTA: $final_rep"
    fi

    # The rep-sort temp (and the node-local sort spill) are transient; drop them.
    [[ -n "${REMOVE_TMP:-}" ]] && rm -rf "$sorted" "${sort_tmp:?}"

    if [[ "$mark_final" == "1" ]]; then
        mark_done "$final_done" "$work_dir/final.done"
        log "final clusters written: $final_result"
    else
        log "partial clusters written without final marker: $final_result"
    fi
}

chunk_done_uri() {
    local result_prefix="$1"
    local chunk_id="$2"
    local prefix
    prefix=$(normalize_s3_prefix "$result_prefix")
    printf '%sdone/%s.done' "$prefix" "$chunk_id"
}

list_missing_chunks() {
    local chunk_manifest="$1"
    local result_prefix="$2"
    while IFS=$'\t' read -r chunk_id _ || [[ -n "${chunk_id:-}" ]]; do
        [[ -z "${chunk_id:-}" || "$chunk_id" =~ ^[[:space:]]*# ]] && continue
        if ! done_exists "$(chunk_done_uri "$result_prefix" "$chunk_id")"; then
            printf '%s\n' "$chunk_id"
        fi
    done < "$chunk_manifest"
}

# list the done/ prefix once and diff against the manifest, instead of one aws s3 ls per chunk
s3_missing_chunks() {
    local chunk_manifest="$1" done_prefix="$2" present="$3"
    s3_list_prefix "$done_prefix" | awk -F/ '{ print $NF }' > "$present"
    awk -F'\t' -v pf="$present" '
        BEGIN { while ((getline l < pf) > 0) { seen[l] = 1 } }
        /^[[:space:]]*(#|$)/ { next }
        { if (!seen[$1 ".done"]) print $1 }
    ' "$chunk_manifest"
}

cluster_manifest_single() {
    local chunk_manifest="$1"
    local result_prefix="$2"
    local round_work_dir="$3"
    local round="${4:-0}"

    mkdir -p "$round_work_dir"

    local attempt=1
    while [[ "$attempt" -le "$MAX_CHUNK_ATTEMPTS" ]]; do
        local failed=0
        while IFS=$'\t' read -r chunk_id chunk_uri seqs _ || [[ -n "${chunk_id:-}" ]]; do
            [[ -z "${chunk_id:-}" || "$chunk_id" =~ ^[[:space:]]*# ]] && continue
            if done_exists "$(chunk_done_uri "$result_prefix" "$chunk_id")"; then
                continue
            fi
            local task_work
            task_work=$(make_chunk_work_dir "$round_work_dir" "$chunk_id" "$round")
            if ! env BATCH_WORKER_DISPATCH=1 bash "$BATCH_SCRIPT" cluster-chunk "$chunk_uri" "$result_prefix" "$task_work" "$chunk_id" "$round" "${seqs:-0}"; then
                failed=1
            fi
        done < "$chunk_manifest"

        local missing_file="$round_work_dir/missing_chunks.attempt${attempt}.txt"
        list_missing_chunks "$chunk_manifest" "$result_prefix" > "$missing_file"
        if [[ "$failed" -eq 0 && ! -s "$missing_file" ]]; then
            return 0
        fi

        log "chunk attempt ${attempt}/${MAX_CHUNK_ATTEMPTS} incomplete; retrying missing chunks"
        attempt=$((attempt + 1))
    done

    local dead_letter="$round_work_dir/DEAD_LETTER.txt"
    list_missing_chunks "$chunk_manifest" "$result_prefix" > "$dead_letter"
    fail "chunk workers did not complete after ${MAX_CHUNK_ATTEMPTS} attempt(s); missing chunks written to $dead_letter"
}

write_shell_export() {
    local out="$1"
    local name="$2"
    local value="$3"
    printf 'export %s=%q\n' "$name" "$value" >> "$out"
}

write_batch_exports() {
    local out="$1"
    local name
    for name in \
        MMSEQS ROUND0_MMSEQS THREADS ROUND0_THREADS CHUNK_MAX_BYTES CHUNK_MAX_SEQS ROUND0_CHUNK_MAX_BYTES ROUND0_CHUNK_MAX_SEQS \
        CHUNK_DISK_BUDGET ROUND0_CHUNK_DISK_BUDGET DISK_POLL_SEC RAM_POLL_SEC \
        S3_CHUNK_PREFIX COMPRESS_BATCH_OUTPUTS \
        CREATEDB_PAR ROUND0_CREATEDB_MODE BATCH_DELETE_SOURCE_CHUNK CLUSTER_CMD ROUND0_CLUSTER_CMD CLUSTER_COV_MODE CLUSTER_PAR ROUND0_CLUSTER_PAR CREATETSV_PAR SORT_TMP \
        MAX_ROUNDS MIN_REDUCTION_RATIO CONVERGENCE_PATIENCE MIN_REDUCTION_COUNT \
        MAX_CHUNK_ATTEMPTS COMPRESS_RATIO MERGE_SPLITS MERGE_SPLIT_JOBS BATCH_BACKEND REMOVE_TMP \
        BATCH_KMER_WRITE_TO_DISK BATCH_COMPRESS_KMER_TMP_FILES CREATEDB_SHUFFLE_SPLITS ROUND0_CREATEDB_SHUFFLE_SPLITS \
        BATCH_REP_FASTA_SPLITS ROUND0_BATCH_REP_FASTA_SPLITS \
        NODE_WORK_DIR ROUND0_NODE_WORK_DIR BATCH_SLURM_NODELIST ROUND0_BATCH_SLURM_NODELIST \
        BATCH_SLURM_PARTITION ROUND0_BATCH_SLURM_PARTITION BATCH_SLURM_TIME ROUND0_BATCH_SLURM_TIME \
        BATCH_SLURM_MEM ROUND0_BATCH_SLURM_MEM BATCH_SLURM_EXTRA ROUND0_BATCH_SLURM_EXTRA SORT_BUFFER_SIZE \
        BATCH_AWS_JOB_QUEUE BATCH_AWS_JOB_DEFINITION ROUND0_BATCH_AWS_JOB_QUEUE ROUND0_BATCH_AWS_JOB_DEFINITION \
        BATCH_AWS_MACHINE ROUND0_BATCH_AWS_MACHINE BATCH_AWS_MACHINE_TAG_KEY BATCH_AWS_SCRIPT_URI \
        BATCH_AWS_JOB_PREFIX BATCH_AWS_LOCAL_DIR BATCH_AWS_MMSEQS ROUND0_BATCH_AWS_MMSEQS \
        BATCH_AWS_TIMEOUT BATCH_AWS_WORKER_ATTEMPTS BATCH_AWS_DRY_RUN AWS_RETRY_MODE AWS_MAX_ATTEMPTS
    do
        write_shell_export "$out" "$name" "${!name:-}"
    done
}

write_slurm_wrapper() {
    local wrapper="$1"
    shift
    {
        printf '#!/usr/bin/env bash\n'
        printf 'set -euo pipefail\n'
        printf 'export LC_ALL=C\n'
    } > "$wrapper"
    write_batch_exports "$wrapper"
    printf 'bash %q %s\n' "$BATCH_SCRIPT" "$(shell_join "$@")" >> "$wrapper"
    chmod +x "$wrapper"
}

# submit_slurm_job <job_name> <log_dir> <wrapper> <node> <round> [depends]: pinned to one host
submit_slurm_job() {
    local job_name="$1"
    local log_dir="$2"
    local wrapper="$3"
    local node="$4"
    local round="$5"
    local depends_on="${6:-}"

    need_cmd sbatch
    mkdir -p "$log_dir"

    local slurm_partition slurm_time slurm_mem slurm_extra
    slurm_partition=$(round_slurm_partition "$round")
    slurm_time=$(round_slurm_time "$round")
    slurm_mem=$(round_slurm_mem "$round")
    slurm_extra=$(round_slurm_extra "$round")

    local -a cmd=(
        sbatch
        --parsable
        --job-name "$job_name"
        --nodes 1
        --ntasks 1
        --cpus-per-task "$(round_threads "$round")"
        --output "$log_dir/%x-%j.out"
        --error "$log_dir/%x-%j.err"
        # a requeued driver or merge would submit its workers twice, and two workers would share a work dir
        --no-requeue
    )
    [[ -n "$node" ]] && cmd+=(--nodelist "$node")
    # afterany, not afterok: the dependent runs even after a failure so it can reconcile and retry
    [[ -n "$depends_on" ]] && cmd+=(--dependency "afterany:${depends_on}" --kill-on-invalid-dep=yes)
    [[ -n "$slurm_partition" ]] && cmd+=(--partition "$slurm_partition")
    [[ -n "$slurm_time" ]] && cmd+=(--time "$slurm_time")
    [[ -n "$slurm_mem" ]] && cmd+=(--mem "$slurm_mem")
    if [[ -n "$slurm_extra" ]]; then
        local -a extra
        read -r -a extra <<< "$slurm_extra"
        cmd+=("${extra[@]}")
    fi
    cmd+=("$wrapper")

    local out rc
    set +e
    out=$("${cmd[@]}")
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] || return "$rc"
    printf '%s\n' "$out" | tail -n 1 | cut -d';' -f1
}

slurm_worker() {
    [[ "$#" -eq 6 ]] || usage
    local chunk_manifest="$1"
    local result_prefix="$2"
    local round_work_dir="$3"
    local round="$4"
    local shard="$5"
    local num_shards="$6"
    [[ "$shard" =~ ^[0-9]+$ && "$num_shards" =~ ^[1-9][0-9]*$ ]] || fail "slurm-worker needs numeric <worker_index> <num_workers>"

    # process this shard's chunks (line index % num_shards == shard) that are not yet done
    local idx=0 chunk_id chunk_uri seqs rest
    while IFS=$'\t' read -r chunk_id chunk_uri seqs rest || [[ -n "${chunk_id:-}" ]]; do
        [[ -z "${chunk_id:-}" ]] && continue
        if [[ $((idx % num_shards)) -eq "$shard" ]] && ! done_exists "$(chunk_done_uri "$result_prefix" "$chunk_id")"; then
            local task_work rc=0
            task_work=$(make_chunk_work_dir "$round_work_dir" "$chunk_id" "$round")
            log "slurm-worker round ${round}, worker ${shard} of ${num_shards}: clustering ${chunk_id}"
            set +e
            env BATCH_WORKER_DISPATCH=1 bash "$BATCH_SCRIPT" cluster-chunk "$chunk_uri" "$result_prefix" "$task_work" "$chunk_id" "$round" "${seqs:-0}"
            rc=$?
            set -e
            [[ "$rc" -eq 0 ]] || log "slurm-worker: chunk ${chunk_id} FAILED (rc=$rc); left no done-marker, so the merge will retry it"
        fi
        idx=$((idx + 1))
    done < "$chunk_manifest"
}

aws_resolve_batch_resource_by_tag() {
    local kind="$1"
    local tag_key="$2"
    local tag_value="$3"
    # resolve queue and definition in one Resource Groups Tagging call, with the tags as the query
    local label rtype arns matches
    case "$kind" in
        queue)          label="job queue";      rtype="batch:job-queue" ;;
        job-definition) label="job definition"; rtype="batch:job-definition" ;;
        *) printf 'unsupported AWS Batch resource kind: %s\n' "$kind" >&2; return 2 ;;
    esac

    arns=$(aws resourcegroupstaggingapi get-resources \
        --resource-type-filters "$rtype" \
        --tag-filters "Key=${tag_key},Values=${tag_value}" \
        --query 'ResourceTagMappingList[].ResourceARN' --output text) || return 1
    arns=$(printf '%s' "$arns" | tr '[:space:]' '\n' | sed '/^$/d')

    if [[ "$kind" == "queue" ]]; then
        # ARNs are .../job-queue/NAME; keep only ENABLED+VALID queues (never route to a disabled one)
        if [[ -n "$arns" ]]; then
            # shellcheck disable=SC2016,SC2086  # $arns splits intentionally; backticks are JMESPath literals
            matches=$(aws batch describe-job-queues --job-queues $arns \
                --query 'jobQueues[?state==`ENABLED`&&status==`VALID`].jobQueueName' \
                --output text) || return 1
            matches=$(printf '%s' "$matches" | tr '[:space:]' '\n' | sed '/^$/d')
        else
            matches=""
        fi
    else
        # ARNs are .../job-definition/NAME:REVISION; keep the max revision per name
        matches=$(printf '%s\n' "$arns" | sed 's#.*/##' \
            | awk -F: 'NF==2 { if ($2+0 > m[$1]+0) m[$1]=$2 } END { for (k in m) print k ":" m[k] }')
    fi

    matches=$(printf '%s\n' "$matches" | LC_ALL=C sort -u | sed '/^$/d')
    local count
    count=$(printf '%s' "$matches" | grep -c . || true)
    if [[ "$count" -eq 1 ]]; then
        printf '%s' "$matches"
        return 0
    fi
    if [[ "$count" -eq 0 ]]; then
        if [[ "$kind" == "queue" && -n "$arns" ]]; then
            printf 'AWS Batch job queue(s) tagged %s=%s exist but none are ENABLED/VALID\n' "$tag_key" "$tag_value" >&2
        else
            printf 'no AWS Batch %s tagged %s=%s\n' "$label" "$tag_key" "$tag_value" >&2
        fi
        return 1
    fi
    printf 'multiple AWS Batch %ss tagged %s=%s: %s\n' \
        "$label" "$tag_key" "$tag_value" "$(printf '%s' "$matches" | tr '\n' ' ')" >&2
    return 1
}

aws_resolve_machine_env() {
    local tag_key="${BATCH_AWS_MACHINE_TAG_KEY:-mmseqs:machine}"
    local needs_lookup=0
    if [[ -n "${BATCH_AWS_MACHINE:-}" ]]; then
        [[ -n "${BATCH_AWS_JOB_QUEUE:-}" && -n "${BATCH_AWS_JOB_DEFINITION:-}" ]] || needs_lookup=1
    fi
    if [[ -n "${ROUND0_BATCH_AWS_MACHINE:-}" ]]; then
        [[ -n "${ROUND0_BATCH_AWS_JOB_QUEUE:-}" && -n "${ROUND0_BATCH_AWS_JOB_DEFINITION:-}" ]] || needs_lookup=1
    fi
    if [[ "$needs_lookup" -eq 1 ]]; then
        need_cmd aws
    fi

    if [[ -n "${BATCH_AWS_MACHINE:-}" ]]; then
        if [[ -z "${BATCH_AWS_JOB_QUEUE:-}" ]]; then
            BATCH_AWS_JOB_QUEUE=$(aws_resolve_batch_resource_by_tag queue "$tag_key" "$BATCH_AWS_MACHINE")
            log "aws-resolve: resolved --aws-machine ${BATCH_AWS_MACHINE} to queue ${BATCH_AWS_JOB_QUEUE}"
        fi
        if [[ -z "${BATCH_AWS_JOB_DEFINITION:-}" ]]; then
            BATCH_AWS_JOB_DEFINITION=$(aws_resolve_batch_resource_by_tag job-definition "$tag_key" "$BATCH_AWS_MACHINE")
            log "aws-resolve: resolved --aws-machine ${BATCH_AWS_MACHINE} to job definition ${BATCH_AWS_JOB_DEFINITION}"
        fi
    fi

    if [[ -n "${ROUND0_BATCH_AWS_MACHINE:-}" ]]; then
        if [[ -z "${ROUND0_BATCH_AWS_JOB_QUEUE:-}" ]]; then
            ROUND0_BATCH_AWS_JOB_QUEUE=$(aws_resolve_batch_resource_by_tag queue "$tag_key" "$ROUND0_BATCH_AWS_MACHINE")
            log "aws-resolve: resolved --round0-aws-machine ${ROUND0_BATCH_AWS_MACHINE} to queue ${ROUND0_BATCH_AWS_JOB_QUEUE}"
        fi
        if [[ -z "${ROUND0_BATCH_AWS_JOB_DEFINITION:-}" ]]; then
            ROUND0_BATCH_AWS_JOB_DEFINITION=$(aws_resolve_batch_resource_by_tag job-definition "$tag_key" "$ROUND0_BATCH_AWS_MACHINE")
            log "aws-resolve: resolved --round0-aws-machine ${ROUND0_BATCH_AWS_MACHINE} to job definition ${ROUND0_BATCH_AWS_JOB_DEFINITION}"
        fi
    fi

    export BATCH_AWS_MACHINE ROUND0_BATCH_AWS_MACHINE BATCH_AWS_MACHINE_TAG_KEY
    export BATCH_AWS_JOB_QUEUE BATCH_AWS_JOB_DEFINITION
    export ROUND0_BATCH_AWS_JOB_QUEUE ROUND0_BATCH_AWS_JOB_DEFINITION
}

aws_require_submit_env() {
    aws_resolve_machine_env
    [[ -n "${BATCH_AWS_JOB_QUEUE:-}" ]] || fail "BATCH_AWS_JOB_QUEUE is required for AWS Batch mode (or pass --aws-machine to resolve it by tag)"
    [[ -n "${BATCH_AWS_JOB_DEFINITION:-}" ]] || fail "BATCH_AWS_JOB_DEFINITION is required for AWS Batch mode (or pass --aws-machine to resolve it by tag)"
    if [[ -n "${BATCH_AWS_WORKER_ATTEMPTS:-}" ]]; then
        [[ "$BATCH_AWS_WORKER_ATTEMPTS" =~ ^([1-9]|10)$ ]] || fail "BATCH_AWS_WORKER_ATTEMPTS must be 1-10 (got '$BATCH_AWS_WORKER_ATTEMPTS')"
    fi
    if [[ -z "${BATCH_AWS_DRY_RUN:-}" ]]; then
        need_cmd aws
    fi
    # let the aws CLI retry throttled SubmitJob calls, so a chain is never left half-built
    export AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}"
    export AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-10}"
}

# the queue selects the instance type and the definition selects the container image, so set both
aws_queue_for_round() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_AWS_JOB_QUEUE:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_AWS_JOB_QUEUE"
    else
        printf '%s' "$BATCH_AWS_JOB_QUEUE"
    fi
}

# safe without :- because aws_require_submit_env asserts these are set first
aws_definition_for_round() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_AWS_JOB_DEFINITION:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_AWS_JOB_DEFINITION"
    else
        printf '%s' "$BATCH_AWS_JOB_DEFINITION"
    fi
}

# drop the S3 work prefix, the mirror of SLURM's rm -rf on a final result
aws_cleanup_work_prefix() {
    local work_prefix="$1"
    [[ -n "${REMOVE_TMP:-}" ]] || return 0
    is_s3 "$work_prefix" || return 0
    need_cmd aws
    log "aws-merge: removing S3 work prefix $work_prefix (REMOVE_TMP set)"
    aws s3 rm --recursive "$work_prefix" >/dev/null 2>&1 || log "aws-merge: warning: cleanup of $work_prefix failed (non-fatal; result is already written)"
}

shell_join() {
    local out=""
    local arg
    for arg in "$@"; do
        printf -v arg '%q' "$arg"
        out="${out} ${arg}"
    done
    printf '%s' "${out# }"
}

aws_bootstrap_command() {
    local script_uri="$1"
    shift
    # the run config lives next to the script; each container downloads and sources it
    local config_uri="${script_uri%/*}/config.env"
    local args
    args=$(shell_join "$@")
    printf 'aws s3 cp %q /tmp/mmseqs-batch.sh --no-progress && aws s3 cp %q /tmp/mmseqs-batch.env --no-progress && chmod +x /tmp/mmseqs-batch.sh && . /tmp/mmseqs-batch.env && /tmp/mmseqs-batch.sh %s' "$script_uri" "$config_uri" "$args"
}

aws_container_overrides() {
    local bootstrap="$1"
    # containerOverrides carries only the command; config vars are sourced from config.env in the container
    local esc="$bootstrap"
    esc="${esc//\\/\\\\}"
    esc="${esc//\"/\\\"}"
    printf '{"command":["bash","-lc","%s"]}' "$esc"
}

aws_submit_batch_job() {
    [[ "$#" -ge 5 ]] || fail "aws_submit_batch_job needs <job_name> <array_size> <depends_on_job_id> <script_uri> <script_args...>"
    local job_name="$1"
    local array_size="$2"
    local depends_on="$3"
    local script_uri="$4"
    shift 4
    local subcommand="${1:-}"   # first script arg = aws-driver | aws-worker | aws-merge
    # the target round is always the last script arg, and it picks the round-specific queue and definition
    local target_round="${*: -1}"
    [[ "$target_round" =~ ^[0-9]+$ ]] || target_round=1

    aws_require_submit_env

    local bootstrap overrides timeout
    bootstrap=$(aws_bootstrap_command "$script_uri" "$@")
    overrides=$(aws_container_overrides "$bootstrap")
    timeout=${BATCH_AWS_TIMEOUT:-43200}

    local cmd=(
        aws batch submit-job
        --job-name "$job_name"
        --job-queue "$(aws_queue_for_round "$target_round")"
        --job-definition "$(aws_definition_for_round "$target_round")"
        --container-overrides "$overrides"
        --timeout "attemptDurationSeconds=${timeout}"
        --query jobId
        --output text
    )
    if [[ "$array_size" -gt 1 ]]; then
        cmd+=(--array-properties "size=${array_size}")
    fi
    if [[ -n "$depends_on" ]]; then
        cmd+=(--depends-on "jobId=${depends_on}")
    fi

    # workers only: an infra kill is retried on a fresh instance so the array still reaches SUCCEEDED
    if [[ "$subcommand" == "aws-worker" ]]; then
        local worker_attempts="${BATCH_AWS_WORKER_ATTEMPTS:-2}"   # value already validated in aws_require_submit_env
        cmd+=(--retry-strategy "attempts=${worker_attempts}")
    fi

    if [[ -n "${BATCH_AWS_DRY_RUN:-}" ]]; then
        printf 'DRY-RUN:' >&2
        printf ' %q' "${cmd[@]}" >&2
        printf '\n' >&2
        printf 'dryrun-%s-%s\n' "$job_name" "$RANDOM"
        return 0
    fi

    "${cmd[@]}"
}

# idempotent submit: adopt the job id at marker_uri instead of forking a second chain
aws_submit_batch_job_once() {
    local marker_uri="$1"; shift
    if done_exists "$marker_uri"; then
        local existing
        existing=$(stream_uri "$marker_uri" | awk 'NR == 1 { print $1; exit }')
        if [[ -n "$existing" ]]; then
            log "adopting already-submitted job ${existing} (${marker_uri})"
            printf '%s\n' "$existing"
            return 0
        fi
    fi
    local jid
    jid=$(aws_submit_batch_job "$@") || return 1
    if [[ -z "${BATCH_AWS_DRY_RUN:-}" ]]; then
        local marker_local
        marker_local=$(mktemp)
        printf '%s\n' "$jid" > "$marker_local"
        copy_out "$marker_local" "$marker_uri" 2>/dev/null \
            || log "warning: could not write submit marker ${marker_uri} (a re-invocation could double-submit this step)"
        rm -f "$marker_local"
    fi
    printf '%s\n' "$jid"
}

aws_round_prefix() {
    local work_prefix="$1"
    local round="$2"
    join_uri "$work_prefix" "round${round}/"
}

aws_clustered_prefix() {
    local work_prefix="$1"
    local round="$2"
    join_uri "$(aws_round_prefix "$work_prefix" "$round")" "clustered/"
}

aws_submit() {
    [[ "$#" -eq 3 ]] || usage
    local input_manifest="$1"
    local work_prefix
    local result_prefix
    work_prefix=$(normalize_s3_prefix "$2")
    result_prefix=$(normalize_s3_prefix "$3")

    is_s3 "$work_prefix" || fail "aws-submit requires an s3:// work prefix"
    is_s3 "$result_prefix" || fail "aws-submit requires an s3:// result prefix"
    # AWS cannot cheaply detect a live chain, so never launch a second aws-submit on one work prefix
    if done_exists "${result_prefix}final.done"; then
        log "aws-submit: reusing completed result ${result_prefix}$(final_cluster_file_name)"
        return 0
    fi
    aws_require_submit_env
    # dry-run performs no S3 writes, so it cannot pin; the pinned value rides in config.env below
    [[ -n "${BATCH_AWS_DRY_RUN:-}" ]] || pin_merge_splits "$(join_uri "$work_prefix" "merge_splits.txt")"

    # the round0 override is a pair: setting only the queue or only the definition mismatches arch and image
    if [[ -n "${ROUND0_BATCH_AWS_JOB_QUEUE:-}" && -z "${ROUND0_BATCH_AWS_JOB_DEFINITION:-}" ]]; then
        log "warning: ROUND0_BATCH_AWS_JOB_QUEUE set without ROUND0_BATCH_AWS_JOB_DEFINITION; round0 runs the BASE container image on the round0 queue (arch mismatch if the queue is a different CPU arch)"
    fi
    if [[ -z "${ROUND0_BATCH_AWS_JOB_QUEUE:-}" && -n "${ROUND0_BATCH_AWS_JOB_DEFINITION:-}" ]]; then
        log "warning: ROUND0_BATCH_AWS_JOB_DEFINITION set without ROUND0_BATCH_AWS_JOB_QUEUE; round0 runs the round0 image on the BASE queue"
    fi

    # every FASTA path must be reachable from a worker container, so s3:// or a shared mount
    if [[ "${BATCH_AWS_ALLOW_NONS3_INPUT:-0}" != "1" ]]; then
        local nons3_paths
        nons3_paths=$(stream_manifest "$input_manifest" | awk 'NF && $1 !~ /^#/ && $1 !~ /^s3:\/\// { if (++n <= 5) print $1 }')
        if [[ -n "$nons3_paths" ]]; then
            log "aws-submit: input manifest has non-s3:// FASTA path(s) worker containers cannot read: $(printf '%s' "$nons3_paths" | tr '\n' ' ')"
            fail "aws-batch input manifest must use s3:// paths (or set BATCH_AWS_ALLOW_NONS3_INPUT=1 for a shared mount visible to every worker)"
        fi
    fi

    local script_uri="${BATCH_AWS_SCRIPT_URI:-${work_prefix}scripts/batch_clustering.sh}"
    if [[ -n "${BATCH_AWS_DRY_RUN:-}" ]]; then
        log "dry-run: would upload $0 to $script_uri"
    else
        copy_out "$0" "$script_uri"
    fi
    export BATCH_AWS_SCRIPT_URI="$script_uri"
    # In the container the mmseqs binary comes from the image, not the submit host's PATH.
    export MMSEQS="${BATCH_AWS_MMSEQS:-mmseqs}"
    export ROUND0_MMSEQS="${ROUND0_BATCH_AWS_MMSEQS:-}"   # round0-image binary path (not the submit host's)

    # serialize the run config to S3 as %q-quoted export lines for every container to source
    local config_uri="${script_uri%/*}/config.env"
    if [[ -n "${BATCH_AWS_DRY_RUN:-}" ]]; then
        log "dry-run: would upload run config to $config_uri"
    else
        local config_local
        config_local=$(mktemp)
        printf 'export LC_ALL=C\n' > "$config_local"
        write_batch_exports "$config_local"
        copy_out "$config_local" "$config_uri"
        rm -f "$config_local"
        log "uploaded run config to $config_uri"
    fi

    local remote_manifest="$input_manifest"
    if ! is_s3 "$input_manifest"; then
        remote_manifest="${work_prefix}input.manifest"
        if [[ -n "${BATCH_AWS_DRY_RUN:-}" ]]; then
            log "dry-run: would upload local input manifest to $remote_manifest"
        else
            copy_out "$input_manifest" "$remote_manifest"
            log "uploaded local input manifest to $remote_manifest"
        fi
    fi

    local job_prefix="${BATCH_AWS_JOB_PREFIX:-mmseqs-batch}"
    local driver_job
    driver_job=$(aws_submit_batch_job "${job_prefix}-round0-driver" 1 "" "$script_uri" \
        aws-driver "$remote_manifest" "$work_prefix" "$result_prefix" 0)
    log "submitted AWS Batch driver round 0: $driver_job"
    printf '%s\n' "$driver_job"
}

aws_driver() {
    [[ "$#" -eq 4 ]] || usage
    local input_manifest="$1"
    local work_prefix
    local result_prefix
    local round="$4"
    work_prefix=$(normalize_s3_prefix "$2")
    result_prefix=$(normalize_s3_prefix "$3")

    is_s3 "$work_prefix" || fail "aws-driver requires an s3:// work prefix"
    is_s3 "$result_prefix" || fail "aws-driver requires an s3:// result prefix"
    [[ "$round" -le "$MAX_ROUNDS" ]] || fail "round $round exceeds MAX_ROUNDS=$MAX_ROUNDS"
    # adopt the run's pinned split count, so no round can write TSVs with a different one
    [[ -n "${BATCH_AWS_DRY_RUN:-}" ]] || pin_merge_splits "$(join_uri "$work_prefix" "merge_splits.txt")"

    local script_uri="${BATCH_AWS_SCRIPT_URI:-${work_prefix}scripts/batch_clustering.sh}"
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    local local_root="${node_work_dir:-${BATCH_AWS_LOCAL_DIR:-/tmp/mmseqs-batch}}/${AWS_BATCH_JOB_ID:-manual}-round${round}-driver"
    rm -rf "$local_root"
    mkdir -p "$local_root"

    local round_prefix chunk_s3_prefix chunk_manifest_s3 chunk_manifest_local chunk_dir
    round_prefix=$(aws_round_prefix "$work_prefix" "$round")
    chunk_s3_prefix=$(join_uri "$round_prefix" "chunks/")
    chunk_manifest_s3=$(join_uri "$round_prefix" "chunks.tsv")
    chunk_manifest_local="$local_root/chunks.tsv"
    chunk_dir="$local_root/chunks"

    if done_exists "${chunk_manifest_s3}.done"; then
        log "aws-driver round ${round}: reusing prepared chunks $chunk_manifest_s3"
    else
        log "aws-driver round ${round}: preparing chunks"
        S3_CHUNK_PREFIX="$chunk_s3_prefix" REMOVE_TMP=TRUE prepare_round "$round" "$input_manifest" "$chunk_dir" "$chunk_manifest_local"
        copy_out "$chunk_manifest_local" "$chunk_manifest_s3"
        copy_out "${chunk_manifest_local}.done" "${chunk_manifest_s3}.done"
    fi

    local chunk_count
    chunk_count=$(count_manifest_rows "$chunk_manifest_s3")
    [[ "$chunk_count" -gt 0 ]] || fail "round ${round}: no chunks found in $chunk_manifest_s3"
    [[ "$chunk_count" -le 10000 ]] || fail "round ${round}: AWS Batch array size ${chunk_count} exceeds 10000; increase --chunk-max-bytes or add array sharding"

    # seed the retry budget unconditionally so a re-driven round gets its full MAX_CHUNK_ATTEMPTS
    local attempt_uri attempt_local
    attempt_uri=$(join_uri "$round_prefix" "chunk_attempts.txt")
    attempt_local="$local_root/chunk_attempts.txt"
    write_counter_file "$attempt_local" 1
    copy_out "$attempt_local" "$attempt_uri"

    local job_prefix="${BATCH_AWS_JOB_PREFIX:-mmseqs-batch}"
    local clustered_prefix worker_job merge_job
    clustered_prefix=$(aws_clustered_prefix "$work_prefix" "$round")
    worker_job=$(aws_submit_batch_job_once "$(join_uri "$round_prefix" "submitted_workers.txt")" "${job_prefix}-round${round}-worker" "$chunk_count" "" "$script_uri" \
        aws-worker "$chunk_manifest_s3" "$clustered_prefix" "$round")
    log "aws-driver round ${round}: submitted worker array $worker_job (${chunk_count} chunks)"

    merge_job=$(aws_submit_batch_job_once "$(join_uri "$round_prefix" "submitted_merge.txt")" "${job_prefix}-round${round}-merge" 1 "$worker_job" "$script_uri" \
        aws-merge "$input_manifest" "$work_prefix" "$result_prefix" "$round")
    log "aws-driver round ${round}: submitted merge job $merge_job depending on $worker_job"

    if [[ -n "${REMOVE_TMP:-}" ]]; then
        rm -rf "$local_root"
    fi
}

aws_worker() {
    [[ "$#" -eq 3 ]] || usage
    local chunk_manifest_s3="$1"
    local result_prefix="$2"
    local round="$3"
    local index="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    local local_root="${node_work_dir:-${BATCH_AWS_LOCAL_DIR:-/tmp/mmseqs-batch}}/${AWS_BATCH_JOB_ID:-manual}-round${round}-worker-${index}"
    rm -rf "$local_root"
    mkdir -p "$local_root"

    # a worker must never fail its array child: dependsOn fires only on SUCCEEDED, retries go by done-marker
    AWS_WORKER_LOCAL_ROOT="$local_root"
    trap 'rc=$?; [[ "$rc" -eq 0 ]] || log "aws-worker soft-fail (rc=$rc): left no done-marker, so the merge will retry it"; [[ -n "${REMOVE_TMP:-}" ]] && rm -rf "${AWS_WORKER_LOCAL_ROOT:-}"; exit 0' EXIT

    local manifest_local="$local_root/chunks.tsv"
    copy_in "$chunk_manifest_s3" "$manifest_local"

    local line chunk_id chunk_uri seqs bytes
    # select the row with the same predicate that sized the array, so selection and sizing cannot diverge
    line=$(awk -v want="$((index + 1))" 'NF && $1 !~ /^#/ { if (++c == want) { print; exit } }' "$manifest_local")
    [[ -n "$line" ]] || fail "array index $index not present in $chunk_manifest_s3"
    IFS=$'\t' read -r chunk_id chunk_uri seqs bytes <<< "$line"
    [[ -n "$chunk_id" && -n "$chunk_uri" ]] || fail "malformed chunk manifest line for array index $index: $line"

    log "aws-worker round ${round} index ${index}: ${chunk_id} ${chunk_uri}"
    local chunk_rc=0
    set +e
    env BATCH_WORKER_DISPATCH=1 bash "$BATCH_SCRIPT" cluster-chunk \
        "$chunk_uri" "$result_prefix" "$local_root/work" "$chunk_id" "$round" "${seqs:-0}"
    chunk_rc=$?
    set -e
    if [[ "$chunk_rc" -ne 0 ]]; then
        log "aws-worker round ${round} index ${index}: chunk ${chunk_id} FAILED (rc=${chunk_rc}); left no done-marker, so the merge will retry it"
    fi
    # cleanup + exit 0 are handled by the EXIT trap above (so they also run on an early set -e failure).
}

aws_merge() {
    [[ "$#" -eq 4 ]] || usage
    local input_manifest="$1"
    local work_prefix
    local result_prefix
    local round="$4"
    work_prefix=$(normalize_s3_prefix "$2")
    result_prefix=$(normalize_s3_prefix "$3")

    is_s3 "$work_prefix" || fail "aws-merge requires an s3:// work prefix"
    is_s3 "$result_prefix" || fail "aws-merge requires an s3:// result prefix"
    # adopt the run's pinned split count, so retries and the propagate join stay on one count
    [[ -n "${BATCH_AWS_DRY_RUN:-}" ]] || pin_merge_splits "$(join_uri "$work_prefix" "merge_splits.txt")"

    local script_uri="${BATCH_AWS_SCRIPT_URI:-${work_prefix}scripts/batch_clustering.sh}"
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    local local_root="${node_work_dir:-${BATCH_AWS_LOCAL_DIR:-/tmp/mmseqs-batch}}/${AWS_BATCH_JOB_ID:-manual}-round${round}-merge"
    rm -rf "$local_root"
    mkdir -p "$local_root"
    # aws-merge has several early returns, so scrub scratch on any exit
    AWS_MERGE_LOCAL_ROOT="$local_root"
    [[ -n "${REMOVE_TMP:-}" ]] && trap 'rm -rf "${AWS_MERGE_LOCAL_ROOT:-}"' EXIT

    local round_prefix clustered_prefix chunk_manifest_s3 tsv_manifest rep_manifest metrics_manifest
    round_prefix=$(aws_round_prefix "$work_prefix" "$round")
    clustered_prefix=$(aws_clustered_prefix "$work_prefix" "$round")
    chunk_manifest_s3=$(join_uri "$round_prefix" "chunks.tsv")
    tsv_manifest=$(join_uri "$clustered_prefix" "tsv_manifest.txt")
    rep_manifest=$(join_uri "$clustered_prefix" "rep_manifest.txt")
    metrics_manifest=$(join_uri "$clustered_prefix" "metrics_manifest.txt")

    local chunk_manifest_local="$local_root/chunks.tsv"
    copy_in "$chunk_manifest_s3" "$chunk_manifest_local"

    local chunk_count cur_reps
    chunk_count=$(count_manifest_rows "$chunk_manifest_local")

    # a chunk is complete iff its done-marker exists, which is written last
    local missing_chunk_ids
    missing_chunk_ids=$(s3_missing_chunks "$chunk_manifest_local" "$(join_uri "$clustered_prefix" "done/")" "$local_root/done_present.txt")
    if [[ -n "$missing_chunk_ids" ]]; then
        local attempt_uri attempts missing_file dead_letter attempt_local job_prefix retry_job retry_merge n_missing
        n_missing=$(printf '%s\n' "$missing_chunk_ids" | awk 'NF { n++ } END { print n + 0 }')
        attempt_uri=$(join_uri "$round_prefix" "chunk_attempts.txt")
        attempts=$(read_counter_uri "$attempt_uri")
        [[ "$attempts" -gt 0 ]] || attempts=1
        missing_file="$local_root/missing_chunks.txt"
        printf '%s\n' "$missing_chunk_ids" > "$missing_file"

        if [[ "$attempts" -ge "$MAX_CHUNK_ATTEMPTS" ]]; then
            dead_letter="$local_root/DEAD_LETTER.txt"
            {
                printf 'round\t%s\n' "$round"
                printf 'expected_chunks\t%s\n' "$chunk_count"
                printf 'missing_chunks\t%s\n' "$n_missing"
                printf 'attempts\t%s\n' "$attempts"
                printf 'missing_chunk_ids\n'
                cat "$missing_file"
            } > "$dead_letter"
            copy_out "$dead_letter" "$(join_uri "$round_prefix" "DEAD_LETTER.txt")"
            fail "round ${round}: ${n_missing}/${chunk_count} chunk(s) incomplete after ${attempts} attempt(s); wrote DEAD_LETTER.txt"
        fi

        attempts=$((attempts + 1))
        attempt_local="$local_root/chunk_attempts.txt"
        write_counter_file "$attempt_local" "$attempts"
        copy_out "$attempt_local" "$attempt_uri"
        copy_out "$missing_file" "$(join_uri "$round_prefix" "missing_chunks_attempt${attempts}.txt")"

        # submit an array sized to the missing chunks only, not to the full chunk count
        local missing_manifest_local missing_manifest_s3
        missing_manifest_local="$local_root/missing_chunks.manifest.tsv"
        awk -F'\t' 'NR==FNR { want[$1]=1; next } ($1 in want)' "$missing_file" "$chunk_manifest_local" > "$missing_manifest_local"
        missing_manifest_s3=$(join_uri "$round_prefix" "missing_chunks_attempt${attempts}.manifest.tsv")
        copy_out "$missing_manifest_local" "$missing_manifest_s3"

        job_prefix="${BATCH_AWS_JOB_PREFIX:-mmseqs-batch}"
        # the retry counter is a read-modify-write on S3, so these two cannot be made idempotent by a marker
        retry_job=$(aws_submit_batch_job "${job_prefix}-round${round}-worker-retry${attempts}" "$n_missing" "" "$script_uri" \
            aws-worker "$missing_manifest_s3" "$clustered_prefix" "$round")
        retry_merge=$(aws_submit_batch_job "${job_prefix}-round${round}-merge-retry${attempts}" 1 "$retry_job" "$script_uri" \
            aws-merge "$input_manifest" "$work_prefix" "$result_prefix" "$round")
        log "aws-merge round ${round}: ${n_missing}/${chunk_count} chunk(s) missing; retrying only those (array size ${n_missing}); worker $retry_job merge $retry_merge"
        return 0
    fi

    # build the manifests from chunk_ids, never from a listing, so stale objects are never ingested
    make_split_tsv_manifest "$chunk_manifest_local" "$(join_uri "$clustered_prefix" "tsv")" "$local_root/tsv_manifest.txt"
    make_manifest_from_chunk_ids "$chunk_manifest_local" "$(join_uri "$clustered_prefix" "metrics")" ".metrics.tsv" "$local_root/metrics_manifest.txt"
    make_rep_manifest_from_metrics "$local_root/metrics_manifest.txt" "$(join_uri "$clustered_prefix" "rep")" "$local_root/rep_manifest.txt" "$round"
    copy_out "$local_root/tsv_manifest.txt" "$tsv_manifest"
    copy_out "$local_root/rep_manifest.txt" "$rep_manifest"
    copy_out "$local_root/metrics_manifest.txt" "$metrics_manifest"

    cur_reps=$(awk -F'\t' '{ s += $3 } END { print s + 0 }' "$local_root/rep_manifest.txt")
    log "aws-merge round ${round}: ${cur_reps} representatives across ${chunk_count} chunk(s)"
    if [[ "$round" -eq 0 ]]; then
        if [[ "$chunk_count" -le 1 ]]; then
            log "aws-merge round 0: input fit in one chunk; finalizing"
            with_round_node_work_dir "$round" finalize_outputs "$tsv_manifest" "$rep_manifest" "$result_prefix" "$local_root/final"
            aws_cleanup_work_prefix "$work_prefix"
            return 0
        fi

        write_state_file "$local_root/state.env" "$cur_reps" 0
        copy_out "$local_root/state.env" "$(join_uri "$round_prefix" "state.env")"

        local next_round=1
        local job_prefix="${BATCH_AWS_JOB_PREFIX:-mmseqs-batch}"
        local next_driver
        next_driver=$(aws_submit_batch_job_once "$(join_uri "$round_prefix" "submitted_next_driver.txt")" "${job_prefix}-round${next_round}-driver" 1 "" "$script_uri" \
            aws-driver "$rep_manifest" "$work_prefix" "$result_prefix" "$next_round")
        log "aws-merge round 0: submitted next driver $next_driver"
        return 0
    fi

    local child_manifest parent_manifest propagated_manifest prev_state prev_reps low_benefit_rounds
    if [[ "$round" -eq 1 ]]; then
        child_manifest=$(join_uri "$(aws_clustered_prefix "$work_prefix" 0)" "tsv_manifest.txt")
    else
        child_manifest=$(join_uri "$(aws_round_prefix "$work_prefix" "$((round - 1))")" "propagated_manifest.txt")
    fi
    parent_manifest="$tsv_manifest"
    propagated_manifest=$(join_uri "$round_prefix" "propagated_manifest.txt")

    # upload each split shard so the next round's driver container can read it
    if done_exists "$propagated_manifest"; then
        log "aws-merge round ${round}: reusing propagated manifest $propagated_manifest"
    else
        local local_prop_manifest="$local_root/propagated_manifest.local.txt"
        with_round_node_work_dir "$round" propagate "$child_manifest" "$parent_manifest" "$local_prop_manifest" "$local_root/propagate"
        : > "$local_root/propagated_manifest.txt"
        local shard s3shard
        while IFS= read -r shard || [[ -n "${shard:-}" ]]; do
            [[ -z "${shard:-}" || "$shard" =~ ^# ]] && continue
            s3shard=$(join_uri "$round_prefix" "propagated/$(basename "$shard")")
            copy_out "$shard" "$s3shard"
            printf '%s\n' "$s3shard" >> "$local_root/propagated_manifest.txt"
        done < "$local_prop_manifest"
        copy_out "$local_root/propagated_manifest.txt" "$propagated_manifest"
    fi

    prev_state="$local_root/prev_state.env"
    copy_in "$(join_uri "$(aws_round_prefix "$work_prefix" "$((round - 1))")" "state.env")" "$prev_state"
    # shellcheck disable=SC1090
    source "$prev_state"
    prev_reps="${PREV_REPS:-0}"
    low_benefit_rounds="${LOW_BENEFIT_ROUNDS:-0}"

    local reduction=0
    if [[ "$prev_reps" -gt "$cur_reps" ]]; then
        reduction=$((prev_reps - cur_reps))
    fi

    local low_benefit=0
    if awk -v c="$cur_reps" -v p="$prev_reps" -v r="$MIN_REDUCTION_RATIO" \
        'BEGIN { exit !(p > 0 && ((p - c) / p) < r) }'; then
        low_benefit=1
    fi
    if [[ "$MIN_REDUCTION_COUNT" -gt 0 && "$reduction" -lt "$MIN_REDUCTION_COUNT" ]]; then
        low_benefit=1
    fi
    if [[ "$low_benefit" -eq 1 ]]; then
        low_benefit_rounds=$((low_benefit_rounds + 1))
        log "aws-merge round ${round}: low-benefit round ${low_benefit_rounds}/${CONVERGENCE_PATIENCE} (removed ${reduction} representatives)"
    else
        low_benefit_rounds=0
    fi

    local converged=0
    local mark_final=1
    if [[ "$chunk_count" -le 1 ]]; then
        converged=1
        log "aws-merge round ${round}: representatives fit in one chunk"
    elif [[ "$low_benefit_rounds" -ge "$CONVERGENCE_PATIENCE" ]]; then
        converged=1
        log "aws-merge round ${round}: representatives converged by low-benefit threshold"
    elif [[ "$round" -ge "$MAX_ROUNDS" ]]; then
        converged=1
        mark_final=0
        log "aws-merge round ${round}: reached MAX_ROUNDS=${MAX_ROUNDS}; writing partial clustering"
    fi

    if [[ "$converged" -eq 1 ]]; then
        with_round_node_work_dir "$round" finalize_outputs "$propagated_manifest" "$rep_manifest" "$result_prefix" "$local_root/final" "$mark_final"
        # only clean on a final result; a partial MAX_ROUNDS result keeps the work prefix so a re-run can resume
        [[ "$mark_final" -eq 1 ]] && aws_cleanup_work_prefix "$work_prefix"
        return 0
    fi

    write_state_file "$local_root/state.env" "$cur_reps" "$low_benefit_rounds"
    copy_out "$local_root/state.env" "$(join_uri "$round_prefix" "state.env")"

    local next_round=$((round + 1))
    local job_prefix="${BATCH_AWS_JOB_PREFIX:-mmseqs-batch}"
    local next_driver
    next_driver=$(aws_submit_batch_job_once "$(join_uri "$round_prefix" "submitted_next_driver.txt")" "${job_prefix}-round${next_round}-driver" 1 "" "$script_uri" \
        aws-driver "$rep_manifest" "$work_prefix" "$result_prefix" "$next_round")
    log "aws-merge round ${round}: submitted next driver $next_driver"

    if [[ -n "${REMOVE_TMP:-}" ]]; then
        rm -rf "$local_root"
    fi
}

# multi-node (SLURM) event chain: driver, worker and merge submitted via sbatch --dependency

# Submit one event-chain job. submit_event_step <job_name> <slurm_dir> <depends_on> <pin_node> <subcommand> <args...>
submit_event_step() {
    local job_name="$1" slurm_dir="$2" depends_on="$3" pin_node="$4"; shift 4
    mkdir -p "$slurm_dir"
    local wrapper="$slurm_dir/${job_name}.sh"
    write_slurm_wrapper "$wrapper" "$@"
    local target_round="${*: -1}"
    [[ "$target_round" =~ ^[0-9]+$ ]] || target_round=1
    local jid rc
    set +e
    jid=$(submit_slurm_job "$job_name" "$slurm_dir" "$wrapper" "$pin_node" "$target_round" "$depends_on")
    rc=$?
    set -e
    [[ "$rc" -eq 0 && -n "$jid" ]] || fail "failed to submit SLURM job ${job_name}"
    printf '%s\n' "$jid"
}

# per-run token derived from work_dir, embedded in every job name so a re-run detects a live chain
run_token() {
    printf '%s' "$1" | cksum | awk '{ print $1 }'
}

# one worker per node that still owns an unfinished chunk; a single sbatch failure is logged, not fatal
submit_round_workers() {
    local chunk_manifest="$1" clustered="$2" round_dir="$3" round="$4" slurm_dir="$5" tok="$6"
    local nn="${#SLURM_NODE_ARRAY[@]}"

    # Which shards still own a not-done chunk? (idle nodes are not submitted.)
    local -a has_work=()
    local s
    for ((s = 0; s < nn; s++)); do has_work[s]=0; done
    local idx=0 cid _rest
    while IFS=$'\t' read -r cid _rest || [[ -n "${cid:-}" ]]; do
        [[ -z "${cid:-}" ]] && continue
        done_exists "$(chunk_done_uri "$clustered" "$cid")" || has_work[idx % nn]=1
        idx=$((idx + 1))
    done < "$chunk_manifest"

    local -a ids=()
    local wrapper jid rc
    for ((s = 0; s < nn; s++)); do
        [[ "${has_work[$s]}" -eq 1 ]] || continue
        wrapper="$slurm_dir/mmseqs-${CLUSTER_CMD}-${tok}-round${round}-worker${s}.sh"
        write_slurm_wrapper "$wrapper" slurm-worker "$chunk_manifest" "$clustered" "$round_dir" "$round" "$s" "$nn"
        set +e
        jid=$(submit_slurm_job "mmseqs-${CLUSTER_CMD}-${tok}-round${round}-worker${s}" "$slurm_dir" "$wrapper" "${SLURM_NODE_ARRAY[$s]}" "$round")
        rc=$?
        set -e
        if [[ "$rc" -eq 0 && -n "$jid" ]]; then
            ids+=("$jid")
        else
            log "WARNING: failed to submit worker shard ${s} (round ${round}); the merge will reconcile and retry it"
        fi
    done
    (IFS=:; printf '%s' "${ids[*]}")
}

# Entry point for the multi-node backend: submit the first driver and return immediately.
slurm_submit() {
    [[ "$#" -eq 3 ]] || usage
    local input_manifest="$1" work_dir="$2" result_dir="$3"
    mkdir -p "$work_dir" "$result_dir"
    local final_cluster_name
    final_cluster_name=$(final_cluster_file_name)
    if [[ -s "$result_dir/$final_cluster_name" && -f "$result_dir/final.done" ]]; then
        log "multi-node: reusing completed result $result_dir/$final_cluster_name"
        return 0
    fi
    pin_merge_splits "$work_dir/merge_splits.txt"
    build_slurm_node_array 0
    [[ "${#SLURM_NODE_ARRAY[@]}" -gt 0 ]] || fail "--backend multi-node requires --slurm-nodelist"
    need_cmd sbatch
    need_cmd squeue
    local slurm_dir="$work_dir/logs"
    mkdir -p "$slurm_dir"

    # refuse a second chain for the same work_dir while one is live, so two job trees cannot race
    local tok
    tok=$(run_token "$work_dir")
    if squeue -h -o '%j' 2>/dev/null | grep -q -- "-${tok}-"; then
        fail "a batch-clustering chain for this work_dir (token ${tok}) is already active in the queue; cancel those jobs or wait for them to finish before re-running"
    fi

    # pin the chain to an immutable snapshot of this script, so deferred jobs cannot pick up a re-staged one
    local snap="$slurm_dir/batch_clustering.sh"
    cp -f "$BATCH_SCRIPT" "$snap"
    BATCH_SCRIPT="$snap"

    local jid
    jid=$(submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-round0-driver" "$slurm_dir" "" "${SLURM_NODE_ARRAY[0]}" \
        slurm-driver "$input_manifest" "$work_dir" "$result_dir" 0)
    log "submitted batch-clustering driver (round 0) as SLURM job ${jid}."
    log "The run now proceeds on the compute nodes as a self-submitting job chain; the submitting host does no further work (no need to keep this session open). Track with: squeue -u \"\$USER\""
    log "job logs: $slurm_dir/<job-name>-<jobid>.out and .err (job names carry round/worker/attempt, e.g. mmseqs-${CLUSTER_CMD}-${tok}-round0-worker0)"
    printf '%s\n' "$jid"
}

# One round's driver: prepare chunks, submit workers, submit the merge (depends on workers), exit.
slurm_driver() {
    [[ "$#" -eq 4 ]] || usage
    local input_manifest="$1" work_dir="$2" result_dir="$3" round="$4"
    [[ "$round" -le "$MAX_ROUNDS" ]] || fail "round $round exceeds MAX_ROUNDS=$MAX_ROUNDS"
    build_slurm_node_array "$round"
    local nn="${#SLURM_NODE_ARRAY[@]}"
    [[ "$nn" -gt 0 ]] || fail "driver: empty node array"

    local round_dir="$work_dir/round${round}"
    local clustered="$round_dir/clustered"
    local slurm_dir="$work_dir/logs"
    local chunks="$round_dir/chunks"
    local chunk_manifest="$round_dir/chunks.tsv"
    mkdir -p "$round_dir" "$clustered" "$slurm_dir"
    # adopt the work area's pinned split count, so no round can write TSVs with a different one
    pin_merge_splits "$work_dir/merge_splits.txt"

    if [[ -s "$chunk_manifest" ]] && done_exists "${chunk_manifest}.done"; then
        log "driver round ${round}: reusing prepared chunks ($chunk_manifest)"
    else
        prepare_round "$round" "$input_manifest" "$chunks" "$chunk_manifest"
    fi
    local chunk_count
    chunk_count=$(count_manifest_rows "$chunk_manifest")
    [[ "$chunk_count" -gt 0 ]] || fail "driver round ${round}: no chunks in $chunk_manifest"

    # only seed it: a re-submitted driver would otherwise reset MAX_CHUNK_ATTEMPTS every time
    [[ -s "$round_dir/chunk_attempts.txt" ]] || write_counter_file "$round_dir/chunk_attempts.txt" 1
    local tok dep mjid
    tok=$(run_token "$work_dir")
    # always submit the merge, so a partial worker submit can never orphan the round
    dep=$(submit_round_workers "$chunk_manifest" "$clustered" "$round_dir" "$round" "$slurm_dir" "$tok")
    mjid=$(submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-round${round}-merge" "$slurm_dir" "$dep" "${SLURM_NODE_ARRAY[0]}" \
        slurm-merge "$input_manifest" "$work_dir" "$result_dir" "$round")
    log "driver round ${round}: ${chunk_count} chunk(s); submitted worker(s) and merge ${mjid}"
}

# one round's merge: reconcile, propagate, decide convergence, then finalize or submit the next driver
slurm_merge() {
    [[ "$#" -eq 4 ]] || usage
    local input_manifest="$1" work_dir="$2" result_dir="$3" round="$4"
    build_slurm_node_array "$round"
    local nn="${#SLURM_NODE_ARRAY[@]}"
    [[ "$nn" -gt 0 ]] || fail "merge: empty node array"

    local round_dir="$work_dir/round${round}"
    local clustered="$round_dir/clustered"
    local slurm_dir="$work_dir/logs"
    local chunk_manifest="$round_dir/chunks.tsv"
    mkdir -p "$clustered" "$slurm_dir"
    # adopt the work area's pinned split count, so retries and the propagate join stay on one count
    pin_merge_splits "$work_dir/merge_splits.txt"

    local tok
    tok=$(run_token "$work_dir")

    # a chunk is complete iff its done-marker exists, which cluster_chunk writes last
    local chunk_count missing_chunks
    chunk_count=$(count_manifest_rows "$chunk_manifest")
    missing_chunks=$(list_missing_chunks "$chunk_manifest" "$clustered")
    if [[ -n "$missing_chunks" ]]; then
        local n_missing attempt_file="$round_dir/chunk_attempts.txt" attempts
        n_missing=$(printf '%s\n' "$missing_chunks" | awk 'NF { n++ } END { print n + 0 }')
        attempts=$(read_counter_uri "$attempt_file")
        [[ "$attempts" -gt 0 ]] || attempts=1
        if [[ "$attempts" -ge "$MAX_CHUNK_ATTEMPTS" ]]; then
            printf '%s\n' "$missing_chunks" > "$round_dir/DEAD_LETTER.txt"
            fail "round ${round}: ${n_missing}/${chunk_count} chunk(s) incomplete after ${attempts} attempt(s); see $round_dir/DEAD_LETTER.txt"
        fi
        attempts=$((attempts + 1))
        write_counter_file "$attempt_file" "$attempts"
        log "merge round ${round}: incomplete (${n_missing}/${chunk_count} chunk(s) missing); retry attempt ${attempts}"
        local dep
        dep=$(submit_round_workers "$chunk_manifest" "$clustered" "$round_dir" "$round" "$slurm_dir" "$tok")
        submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-round${round}-merge-attempt${attempts}" "$slurm_dir" "$dep" "${SLURM_NODE_ARRAY[0]}" \
            slurm-merge "$input_manifest" "$work_dir" "$result_dir" "$round" >/dev/null
        return 0
    fi

    # build the round manifests from chunk_ids, never a directory find, so stale outputs are not ingested
    local tsv_manifest="$clustered/tsv_manifest.txt"
    local rep_manifest="$clustered/rep_manifest.txt"
    local metrics_manifest="$clustered/metrics_manifest.txt"
    make_split_tsv_manifest "$chunk_manifest" "$clustered/tsv" "$tsv_manifest"
    make_rep_manifest_with_sizes "$chunk_manifest" "$clustered" "$rep_manifest" "$round"
    make_manifest_from_chunk_ids "$chunk_manifest" "$clustered/metrics" '.metrics.tsv' "$metrics_manifest"

    local cur_reps
    cur_reps=$(count_reps_from_metrics_manifest "$metrics_manifest")
    log "merge round ${round}: ${cur_reps} representatives across ${chunk_count} chunk(s)"

    if [[ "$round" -eq 0 ]]; then
        if [[ "$chunk_count" -le 1 ]]; then
            log "merge round 0: input fit in one chunk; finalizing"
            with_round_node_work_dir "$round" finalize_outputs "$tsv_manifest" "$rep_manifest" "$result_dir" "$round_dir/final" 1
            # final result is in result_dir; the round scratch is now safe to drop.
            [[ -n "${REMOVE_TMP:-}" ]] && rm -rf "$work_dir"/round*
            return 0
        fi
        write_state_file "$round_dir/state.env" "$cur_reps" 0
        local nd next_driver_node
        next_driver_node=$(first_slurm_node_for_round 1)
        nd=$(submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-round1-driver" "$slurm_dir" "" "$next_driver_node" \
            slurm-driver "$rep_manifest" "$work_dir" "$result_dir" 1)
        log "merge round 0: ${cur_reps} reps across ${chunk_count} chunks; submitted next driver ${nd}"
        return 0
    fi

    local child_manifest parent_manifest propagated_manifest
    if [[ "$round" -eq 1 ]]; then
        child_manifest="$work_dir/round0/clustered/tsv_manifest.txt"
    else
        child_manifest="$work_dir/round$((round - 1))/propagated_manifest.txt"
    fi
    parent_manifest="$tsv_manifest"
    propagated_manifest="$round_dir/propagated_manifest.txt"

    with_round_node_work_dir "$round" propagate "$child_manifest" "$parent_manifest" "$propagated_manifest" "$round_dir/propagate"

    local prev_reps low_benefit_rounds
    # shellcheck disable=SC1090
    source "$work_dir/round$((round - 1))/state.env"
    prev_reps="${PREV_REPS:-0}"
    low_benefit_rounds="${LOW_BENEFIT_ROUNDS:-0}"

    local reduction=0
    [[ "$prev_reps" -gt "$cur_reps" ]] && reduction=$((prev_reps - cur_reps))
    local low_benefit=0
    if awk -v c="$cur_reps" -v p="$prev_reps" -v r="$MIN_REDUCTION_RATIO" \
        'BEGIN { exit !(p > 0 && ((p - c) / p) < r) }'; then
        low_benefit=1
    fi
    [[ "$MIN_REDUCTION_COUNT" -gt 0 && "$reduction" -lt "$MIN_REDUCTION_COUNT" ]] && low_benefit=1
    if [[ "$low_benefit" -eq 1 ]]; then
        low_benefit_rounds=$((low_benefit_rounds + 1))
        log "merge round ${round}: low-benefit round ${low_benefit_rounds}/${CONVERGENCE_PATIENCE} (removed ${reduction} representatives)"
    else
        low_benefit_rounds=0
    fi

    local converged=0 mark_final=1
    if [[ "$chunk_count" -le 1 ]]; then
        converged=1; log "merge round ${round}: representatives fit in one chunk"
    elif [[ "$low_benefit_rounds" -ge "$CONVERGENCE_PATIENCE" ]]; then
        converged=1; log "merge round ${round}: representatives converged by low-benefit threshold"
    elif [[ "$round" -ge "$MAX_ROUNDS" ]]; then
        converged=1; mark_final=0; log "merge round ${round}: reached MAX_ROUNDS=${MAX_ROUNDS}; writing partial clustering"
    fi

    if [[ "$converged" -eq 1 ]]; then
        with_round_node_work_dir "$round" finalize_outputs "$propagated_manifest" "$rep_manifest" "$result_dir" "$round_dir/final" "$mark_final"
        # only drop the round scratch on a converged result, so a re-run with a larger --max-rounds can resume
        [[ "$mark_final" -eq 1 && -n "${REMOVE_TMP:-}" ]] && rm -rf "$work_dir"/round*
        return 0
    fi

    write_state_file "$round_dir/state.env" "$cur_reps" "$low_benefit_rounds"
    local next_round=$((round + 1)) nd next_driver_node
    next_driver_node=$(first_slurm_node_for_round "$next_round")
    nd=$(submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-round${next_round}-driver" "$slurm_dir" "" "$next_driver_node" \
        slurm-driver "$rep_manifest" "$work_dir" "$result_dir" "$next_round")
    log "merge round ${round}: submitted next driver ${nd}"
}

# single-node driver: prepare, cluster chunks, propagate rounds, finalize, all in process
run_workflow() {
    [[ "$#" -eq 3 ]] || usage
    local input_manifest="$1"
    local work_dir="$2"
    local result_dir="$3"

    mkdir -p "$work_dir" "$result_dir"
    local final_cluster_name
    final_cluster_name=$(final_cluster_file_name)
    if [[ -s "$result_dir/$final_cluster_name" && -f "$result_dir/final.done" ]]; then
        log "single-node: reusing completed result $result_dir/$final_cluster_name"
        return 0
    fi
    pin_merge_splits "$work_dir/merge_splits.txt"

    local round=0
    local chunks="$work_dir/round${round}/chunks"
    local chunk_manifest="$work_dir/round${round}/chunks.tsv"
    mkdir -p "$work_dir/round${round}"
    prepare_round "$round" "$input_manifest" "$chunks" "$chunk_manifest"

    cluster_manifest_single "$chunk_manifest" "$work_dir/round${round}/clustered" "$work_dir/round${round}" "$round"

    local current_mapping_manifest="$work_dir/round${round}/clustered/tsv_manifest.txt"
    local current_rep_manifest="$work_dir/round${round}/clustered/rep_manifest.txt"
    local current_metrics_manifest="$work_dir/round${round}/clustered/metrics_manifest.txt"
    validate_clustered_outputs "$chunk_manifest" "$work_dir/round${round}/clustered"
    make_split_tsv_manifest "$chunk_manifest" "$work_dir/round${round}/clustered/tsv" "$current_mapping_manifest"
    make_rep_manifest_with_sizes "$chunk_manifest" "$work_dir/round${round}/clustered" "$current_rep_manifest" "$round"
    make_manifest_from_chunk_ids "$chunk_manifest" "$work_dir/round${round}/clustered/metrics" '.metrics.tsv' "$current_metrics_manifest"

    local initial_chunk_count
    initial_chunk_count=$(count_manifest_rows "$chunk_manifest")
    local converged=0
    if [[ "$initial_chunk_count" -le 1 ]]; then
        converged=1
        log "input fit in one chunk; no representative merge rounds needed"
    fi

    local prev_reps
    prev_reps=$(count_reps_from_metrics_manifest "$current_metrics_manifest")
    log "round 0: ${prev_reps} representatives across ${initial_chunk_count} chunk(s)"

    local low_benefit_rounds=0
    local current_round=0
    for ((round=1; converged == 0 && round<=MAX_ROUNDS; round++)); do
        current_round="$round"
        log "starting representative round ${round}"
        chunks="$work_dir/round${round}/chunks"
        chunk_manifest="$work_dir/round${round}/chunks.tsv"
        mkdir -p "$work_dir/round${round}"

        prepare_round "$round" "$current_rep_manifest" "$chunks" "$chunk_manifest"
        local rep_chunk_count
        rep_chunk_count=$(count_manifest_rows "$chunk_manifest")
        cluster_manifest_single "$chunk_manifest" "$work_dir/round${round}/clustered" "$work_dir/round${round}" "$round"

        local parent_manifest="$work_dir/round${round}/clustered/tsv_manifest.txt"
        current_rep_manifest="$work_dir/round${round}/clustered/rep_manifest.txt"
        local round_metrics_manifest="$work_dir/round${round}/clustered/metrics_manifest.txt"
        validate_clustered_outputs "$chunk_manifest" "$work_dir/round${round}/clustered"
        make_split_tsv_manifest "$chunk_manifest" "$work_dir/round${round}/clustered/tsv" "$parent_manifest"
        make_rep_manifest_with_sizes "$chunk_manifest" "$work_dir/round${round}/clustered" "$current_rep_manifest" "$round"
        make_manifest_from_chunk_ids "$chunk_manifest" "$work_dir/round${round}/clustered/metrics" '.metrics.tsv' "$round_metrics_manifest"

        # propagate dispatches its split tasks per backend and emits MERGE_SPLITS mapping shards
        local propagated_manifest="$work_dir/round${round}/propagated_manifest.txt"
        with_round_node_work_dir "$round" propagate "$current_mapping_manifest" "$parent_manifest" "$propagated_manifest" "$work_dir/round${round}/propagate"
        current_mapping_manifest="$propagated_manifest"

        local cur_reps
        cur_reps=$(count_reps_from_metrics_manifest "$round_metrics_manifest")
        log "round ${round}: ${prev_reps} -> ${cur_reps} representatives (${rep_chunk_count} chunk(s))"

        if [[ "$rep_chunk_count" -le 1 ]]; then
            converged=1
            log "representatives fit in one chunk; finishing"
        else
            local reduction=0
            if [[ "$prev_reps" -gt "$cur_reps" ]]; then
                reduction=$((prev_reps - cur_reps))
            fi
            local low_benefit=0
            if awk -v c="$cur_reps" -v p="$prev_reps" -v r="$MIN_REDUCTION_RATIO" \
                'BEGIN { exit !(p > 0 && ((p - c) / p) < r) }'; then
                low_benefit=1
            fi
            if [[ "$MIN_REDUCTION_COUNT" -gt 0 && "$reduction" -lt "$MIN_REDUCTION_COUNT" ]]; then
                low_benefit=1
            fi
            if [[ "$low_benefit" -eq 1 ]]; then
                low_benefit_rounds=$((low_benefit_rounds + 1))
                log "round ${round}: low-benefit round ${low_benefit_rounds}/${CONVERGENCE_PATIENCE} (removed ${reduction} representatives)"
            else
                low_benefit_rounds=0
            fi
            if [[ "$low_benefit_rounds" -ge "$CONVERGENCE_PATIENCE" ]]; then
                converged=1
                log "representatives converged after ${low_benefit_rounds} low-benefit round(s)"
            fi
        fi
        prev_reps="$cur_reps"
    done

    if [[ "$converged" -ne 1 ]]; then
        log "WARNING: reached MAX_ROUNDS=${MAX_ROUNDS} while representatives were still reducing; emitting current global clustering (raise MAX_ROUNDS to continue)"
    fi

    local mark_final=0
    [[ "$converged" -eq 1 ]] && mark_final=1
    # finalize dispatches its rep-first-sort splits per backend and marks final.done only when mark_final=1
    with_round_node_work_dir "$current_round" finalize_outputs "$current_mapping_manifest" "$current_rep_manifest" "$result_dir" "$work_dir/finalize" "$mark_final"

    if [[ "$converged" -eq 1 ]]; then
        if [[ -n "${REMOVE_TMP:-}" ]]; then
            rm -rf "$work_dir"/round* "$work_dir"/sort-tmp "$work_dir"/finalize
            rm -f "$work_dir/batch_clustering.sh"
            case "$BATCH_SCRIPT" in
                "$work_dir"/*|/private"$work_dir"/*) rm -f "$BATCH_SCRIPT" ;;
            esac
        fi
        log "${BATCH_BACKEND} complete: $result_dir/$(final_cluster_file_name)"
    else
        log "${BATCH_BACKEND} stopped at MAX_ROUNDS=${MAX_ROUNDS} WITHOUT convergence."
        log "Wrote a PARTIAL clustering to $result_dir/$(final_cluster_file_name) (NOT marked final)."
        log "Re-run the same command with a larger --max-rounds to resume from the completed rounds kept in $work_dir."
    fi
}

run_single_node() {
    BATCH_BACKEND=single-node run_workflow "$@"
}

run_multi_node() {
    # multi-node runs everything, including the merge, through the SLURM event chain
    BATCH_BACKEND=multi-node
    slurm_submit "$@"
}

main() {
    [[ "$#" -ge 1 ]] || usage
    local mode="$1"
    shift
    case "$mode" in
        prepare)       prepare "$@" ;;
        cluster-chunk)
            [[ "${BATCH_WORKER_DISPATCH:-}" == "1" ]] || \
                fail "cluster-chunk received no dispatch environment (BATCH_WORKER_DISPATCH unset); refusing to cluster with default parameters. This subcommand is launched by the driver, not run directly."
            cluster_chunk "$@" ;;
        measure-input)
            [[ "${BATCH_WORKER_DISPATCH:-}" == "1" ]] || \
                fail "measure-input received no dispatch environment (BATCH_WORKER_DISPATCH unset). This subcommand is launched by prepare, not run directly."
            measure_input "$@" ;;
        propagate)     propagate "$@" ;;
        finalize)      finalize_outputs "$@" ;;
        run-single-node) run_single_node "$@" ;;
        run-multi-node)  run_multi_node "$@" ;;
        slurm-driver)  slurm_driver "$@" ;;
        slurm-worker)  slurm_worker "$@" ;;
        slurm-merge)   slurm_merge "$@" ;;
        aws-submit)    aws_submit "$@" ;;
        aws-driver)    aws_driver "$@" ;;
        aws-worker)    aws_worker "$@" ;;
        aws-merge)     aws_merge "$@" ;;
        -h|--help|help) usage ;;
        *) fail "unknown mode: $mode" ;;
    esac
}

main "$@"
