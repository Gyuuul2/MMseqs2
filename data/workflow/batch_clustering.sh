#!/usr/bin/env bash
# Intentional bash (not /bin/sh like the linear step-runner peers): this is a distributed
# orchestrator (single-node / SLURM / AWS Batch) relying on arrays, [[ ]], printf %q, process
# substitution and traps. Do not "port" to POSIX sh -- it would drop the %q path-quoting safety.
set -euo pipefail
export LC_ALL=C

fail() {
    echo "Error: $*" >&2
    exit 1
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
  batch_clustering.sh slurm-worker <chunk_manifest> <result_prefix> <round_work_dir> <round> <shard> <num_shards>
  batch_clustering.sh slurm-merge <input_manifest> <work_dir> <result_dir> <round>
  batch_clustering.sh aws-submit <input_manifest> <work_s3_prefix> <result_s3_prefix>
  batch_clustering.sh aws-driver <input_manifest> <work_s3_prefix> <result_s3_prefix> <round>
  batch_clustering.sh aws-worker <chunk_manifest_s3> <result_s3_prefix> <round>
  batch_clustering.sh aws-merge <input_manifest> <work_s3_prefix> <result_s3_prefix> <round>

Environment (normally exported by mmseqs from the linclust-batch/cluster-batch command line;
set them directly only when running this script standalone):
  MMSEQS ROUND0_MMSEQS THREADS CHUNK_MAX_BYTES CHUNK_MAX_SEQS MERGE_BUCKETS MERGE_BUCKET_JOBS COMPRESS_RATIO
  CLUSTER_CMD CLUSTER_PAR ROUND0_CLUSTER_PAR CLUSTER_COV_MODE CREATEDB_PAR CREATETSV_PAR COMPRESS_BATCH_OUTPUTS SORT_BUFFER_SIZE SORT_TMP
  MAX_ROUNDS MIN_REDUCTION_RATIO MIN_REDUCTION_COUNT CONVERGENCE_PATIENCE MAX_CHUNK_ATTEMPTS
  REMOVE_TMP NODE_WORK_DIR ROUND0_NODE_WORK_DIR
  BATCH_SLURM_NODELIST ROUND0_BATCH_SLURM_NODELIST BATCH_SLURM_PARTITION ROUND0_BATCH_SLURM_PARTITION
  BATCH_SLURM_TIME ROUND0_BATCH_SLURM_TIME BATCH_SLURM_MEM ROUND0_BATCH_SLURM_MEM BATCH_SLURM_EXTRA ROUND0_BATCH_SLURM_EXTRA
  BATCH_AWS_MACHINE ROUND0_BATCH_AWS_MACHINE BATCH_AWS_MACHINE_TAG_KEY
  BATCH_AWS_JOB_QUEUE BATCH_AWS_JOB_DEFINITION ROUND0_BATCH_AWS_JOB_QUEUE ROUND0_BATCH_AWS_JOB_DEFINITION
  BATCH_AWS_JOB_PREFIX BATCH_AWS_WORKER_ATTEMPTS BATCH_AWS_ALLOW_NONS3_INPUT
  BATCH_AWS_MMSEQS ROUND0_BATCH_AWS_MMSEQS BATCH_AWS_SCRIPT_URI BATCH_AWS_LOCAL_DIR BATCH_AWS_TIMEOUT BATCH_AWS_DRY_RUN
  BATCH_DELETE_SOURCE_CHUNK ROUND0_CREATEDB_MODE S3_CHUNK_PREFIX BATCH_AWS_ALLOW_NONS3_INPUT
  (AWS env var names must NOT start with 'AWS_BATCH' -- that prefix is reserved by the AWS Batch service.)
  All of the above also exist as command line parameters (see 'mmseqs linclust-batch -h').
EOF
    exit 1
}

# INVARIANT: every ${VAR:-default} below must match Parameters::setDefaults / addBatchEngineVariables.
MMSEQS=${MMSEQS:-mmseqs}
# Optional round0-specific mmseqs binary (multi-node): when round0 runs on a different-architecture
# node pool than round1+ (e.g. ARM round0 / x86 round1+ on a heterogeneous SLURM cluster), a single
# binary path cannot serve both. Empty = use $MMSEQS for every round (the common same-arch case).
ROUND0_MMSEQS=${ROUND0_MMSEQS:-}
THREADS=${THREADS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)}
CHUNK_MAX_BYTES=${CHUNK_MAX_BYTES:-21474836480}   # = 20*1024^3; must match BatchClustering.cpp/Parameters.cpp batchChunkMaxBytes so a standalone run chunks identically to the mmseqs-injected run
CHUNK_MAX_SEQS=${CHUNK_MAX_SEQS:-0}
ROUND0_CHUNK_MAX_BYTES=${ROUND0_CHUNK_MAX_BYTES:-}
ROUND0_CHUNK_MAX_SEQS=${ROUND0_CHUNK_MAX_SEQS:-}
S3_CHUNK_PREFIX=${S3_CHUNK_PREFIX:-}
COMPRESS_BATCH_OUTPUTS=${COMPRESS_BATCH_OUTPUTS:-0}
[[ "$COMPRESS_BATCH_OUTPUTS" =~ ^[01]$ ]] || fail "COMPRESS_BATCH_OUTPUTS must be 0 or 1 (got '$COMPRESS_BATCH_OUTPUTS')"
# --write-lookup 0: the per-chunk .lookup is unused here. The inner clust reads .lookup only in
# set-mode (never enabled from batch entry points); accessions come from the _h header DB, not it.
# --createdb-mode 0: createdb reads FASTA/.zst natively and writes the compact sequence DB. Mode 1
# is explicit opt-in only; batch must materialize plain FASTA first, which is slower for .zst chunks.
CREATEDB_PAR=${CREATEDB_PAR:---shuffle 0 --write-lookup 0 --createdb-mode 1}
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
        0|1) ;;
        *) fail "batch clustering supports CREATEDB_PAR --createdb-mode 0 or 1 only (got $mode). Mode 2/GPU changes DB layout and can change clustering results." ;;
    esac
}
validate_createdb_par
CLUSTER_CMD=${CLUSTER_CMD:-linclust}
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
MAX_CHUNK_ATTEMPTS=${MAX_CHUNK_ATTEMPTS:-2}
[[ "$MAX_CHUNK_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_CHUNK_ATTEMPTS must be a positive integer (got '$MAX_CHUNK_ATTEMPTS')"
COMPRESS_RATIO=${COMPRESS_RATIO:-3}

merge_default_threads() {
    if [[ "${THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s' "$THREADS"
    else
        printf '1'
    fi
}

auto_merge_buckets() {
    local buckets
    buckets=$(merge_default_threads)   # always >= 1
    [[ "$buckets" -gt 256 ]] && buckets=256
    printf '%s' "$buckets"
}

auto_merge_bucket_jobs() {
    local threads buckets jobs
    threads=$(merge_default_threads)
    buckets="${MERGE_BUCKETS:-1}"
    jobs=$((threads / 8))
    [[ "$jobs" -lt 1 ]] && jobs=1
    [[ "$jobs" -gt 16 ]] && jobs=16
    [[ "$jobs" -gt "$buckets" ]] && jobs="$buckets"
    printf '%s' "$jobs"
}

if [[ -z "${MERGE_BUCKETS:-}" || "$MERGE_BUCKETS" == "0" ]]; then
    MERGE_BUCKETS=$(auto_merge_buckets)
fi
[[ "$MERGE_BUCKETS" =~ ^[1-9][0-9]*$ ]] || fail "MERGE_BUCKETS must be a positive integer (got '$MERGE_BUCKETS')"
if [[ -z "${MERGE_BUCKET_JOBS:-}" || "$MERGE_BUCKET_JOBS" == "0" ]]; then
    MERGE_BUCKET_JOBS=$(auto_merge_bucket_jobs)
fi
[[ "$MERGE_BUCKET_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "MERGE_BUCKET_JOBS must be a positive integer (got '$MERGE_BUCKET_JOBS')"
[[ "$MERGE_BUCKET_JOBS" -gt "$MERGE_BUCKETS" ]] && MERGE_BUCKET_JOBS="$MERGE_BUCKETS"
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
    # round0 chunks are usually sized to fit in memory, where the spill is a no-op, so it defaults
    # to round1+ only; an explicit --kmer-write-to-disk overrides that, because with a small
    # --split-memory-limit round0 chunks do split and the caller may want the spill there too.
    local spill=1
    [[ "$round" -eq 0 ]] && spill=0
    [[ -n "${BATCH_KMER_WRITE_TO_DISK:-}" ]] && spill="$BATCH_KMER_WRITE_TO_DISK"
    printf '%s --kmer-write-to-disk %s' "$(par_without_flag "$base" --kmer-write-to-disk)" "$spill"
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

# Round0-specific mmseqs binary (see ROUND0_MMSEQS above). All mmseqs invocations happen in
# cluster_chunk, which knows its round -- so this one helper covers every backend (single/slurm/aws).
# On AWS the per-round binary is normally the round's container image, so ROUND0_MMSEQS stays empty.
round_mmseqs() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_MMSEQS:-}" ]]; then
        printf '%s' "$ROUND0_MMSEQS"
    else
        printf '%s' "$MMSEQS"
    fi
}

prepare_round() {
    local round="$1"
    shift
    local old_chunk_max_bytes="$CHUNK_MAX_BYTES"
    local old_chunk_max_seqs="$CHUNK_MAX_SEQS"
    local rc
    CHUNK_MAX_BYTES=$(round_chunk_max_bytes "$round")
    CHUNK_MAX_SEQS=$(round_chunk_max_seqs "$round")
    set +e
    prepare "$@"
    rc=$?
    set -e
    CHUNK_MAX_BYTES="$old_chunk_max_bytes"
    CHUNK_MAX_SEQS="$old_chunk_max_seqs"
    return "$rc"
}

with_round_node_work_dir() {
    local round="$1"
    shift
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    NODE_WORK_DIR="$node_work_dir" "$@"
}

# GNU-sort acceleration for the merge: parallel workers + an in-memory buffer before spilling to -T.
# SORT_BUFFER_SIZE is the per-sort memory target; it is NOT related to bucket count. It defaults to a
# modest fraction of node RAM (auto-scales to the machine, no manual tuning) so that the round
# mapping -- one row per ORIGINAL sequence, hundreds of GB and unrelated to chunk size -- stays in
# external (disk-backed) mode. GNU sort's -S is a target, not a hard cap (the parallel merge can
# exceed it somewhat), but at ~25% of RAM the headroom is large enough that a single sort will not
# realistically OOM. --merge-buckets shrinks each sort's INPUT; this only sizes its memory.
# BSD/macOS sort lacks these flags, so they are only added when GNU sort is detected.
if [[ -z "${SORT_BUFFER_SIZE+x}" ]]; then
    if [[ "$MERGE_BUCKET_JOBS" -gt 1 ]]; then
        sort_pct=$((25 / MERGE_BUCKET_JOBS))
        [[ "$sort_pct" -lt 1 ]] && sort_pct=1
        SORT_BUFFER_SIZE="${sort_pct}%"
    else
        SORT_BUFFER_SIZE="25%"
    fi
fi
SORT_PARALLEL_OPT=""
if sort --version 2>/dev/null | grep -q GNU; then
    MERGE_SORT_THREADS="$THREADS"
    if [[ "$MERGE_BUCKET_JOBS" -gt 1 ]]; then
        MERGE_SORT_THREADS=$((THREADS / MERGE_BUCKET_JOBS))
        [[ "$MERGE_SORT_THREADS" -lt 1 ]] && MERGE_SORT_THREADS=1
    fi
    SORT_PARALLEL_OPT="--parallel=$MERGE_SORT_THREADS --buffer-size=$SORT_BUFFER_SIZE"
fi

# Where GNU sort spills its external-sort runs (-T). Prefer node-local NVMe (--node-work-dir) so the
# large merge/finalize sort does not read/write the shared filesystem repeatedly; fall back to the
# work_dir (already node-local on single-node). SORT_TMP overrides everything.
# Node-local scratch dir for a merge/finalize step (partition files, sort spill). Prefers NVMe
# (--node-work-dir) with a per-run component (like make_chunk_work_dir) so co-located runs never
# collide -- otherwise one run's cleanup could rm -rf another run's scratch mid-sort. Falls back to
# the work_dir (already node-local on single-node). Only the merge OUTPUT (shards/ for the next
# round) and the final output must live on shared storage; these intermediates should not.
resolve_node_scratch() {
    local work_dir="$1" name="$2"
    if [[ -n "${NODE_WORK_DIR:-}" ]]; then
        printf '%s/mmseqs-batch/%s/%s/%s' "$NODE_WORK_DIR" "${USER:-user}" "$(sanitize_path "$work_dir")" "$name"
    else
        printf '%s/%s' "$work_dir" "$name"
    fi
}

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
        # head-object is an EXACT-key existence test; `aws s3 ls <key>` is a prefix match that can
        # false-positive (e.g. report chunk-1.done present because chunk-10.done exists). Existence
        # here gates the done-marker reconcile, so an exact test is required for correctness.
        # NOTE: head-object needs s3:GetObject on the key (workers already need it to fetch chunks);
        # a 403 is indistinguishable from a 404 here, so a role lacking GetObject on the done/ path
        # would see every marker as absent and resubmit endlessly.
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
    # list-objects-v2 returns exact keys (no `aws s3 ls` column parsing, robust to any object name)
    # and the CLI auto-paginates, so all keys under the prefix are returned. A null result prints
    # "None" with --output text; drop it. Keys are tab- or newline-separated; normalize to one/line.
    # stderr is NOT suppressed: this listing drives the done-marker reconcile, so a throttle/403/network
    # failure must surface and abort the merge rather than be silently read as "no chunks done".
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

createdb_softlink_enabled() {
    [[ " ${CREATEDB_PAR} " =~ (^|[[:space:]])--createdb-mode(=|[[:space:]]+)1($|[[:space:]]) ]]
}

# createdb mode is round-dependent. round0 chunks are single-frame zstd (no parallel pzstd -dc), so
# soft-link mode 1 only adds materialize + reindex I/O -> use native mode 0. Set ROUND0_CREATEDB_MODE=1
# once round0 chunks are multi-frame. round1+ representatives are pzstd multi-frame, so mode 1 (soft-link
# + parallel decompress) pays off; it follows CREATEDB_PAR's mode (default 1).
round_createdb_softlink() {
    local round="$1"
    if [[ "$round" -eq 0 ]]; then
        [[ "${ROUND0_CREATEDB_MODE:-0}" == "1" ]]
    else
        createdb_softlink_enabled
    fi
}

round_createdb_par() {
    local round="$1"
    local mode=0
    round_createdb_softlink "$round" && mode=1
    printf '%s --createdb-mode %s' "$(par_without_flag "$CREATEDB_PAR" --createdb-mode)" "$mode"
}

append_singleline_fasta() {
    local src="$1"
    local dst="$2"
    # createdb softlink mode needs plain single-line FASTA. Inspect the first record while
    # streaming: if it is already single-line, append the input verbatim; otherwise normalize all
    # records to one sequence line. The first-record check assumes a consistent wrapping style per
    # file, matching the common Prodigal/MMseqs output cases.
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
    local mem count=0
    mkdir -p "$(dirname "$dst")"
    : > "$tmp"
    while IFS= read -r mem || [[ -n "${mem:-}" ]]; do
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
        # `|| true`: a missing key / denied head exits non-zero and under pipefail would abort the
        # caller (this is a best-effort size estimate, not a hard existence check). `END` always
        # emits an integer even on empty input, preserving the "always returns a number" contract.
        { aws s3api head-object --bucket "$(s3_bucket "$uri")" --key "$(s3_key_prefix "$uri")" \
            --query ContentLength --output text 2>/dev/null || true; } | awk 'END { print ($1 ~ /^[0-9]+$/) ? $1 : 0 }'
    else
        stat -c %s "$uri" 2>/dev/null || stat -f %z "$uri" 2>/dev/null || echo 0
    fi
}

# Uncompressed-size estimate for bin-packing. A local .zst carries its exact decompressed size in the
# frame header (zstd -lv, header-only, no decompression); use it when present so representative FASTAs
# -- which compress poorly and would be badly over-estimated by a fixed ratio, wrongly tripping the
# oversized-file guard -- are sized correctly. Otherwise fall back to an on-disk-size x COMPRESS_RATIO
# estimate (S3, .gz, or a stream-compressed .zst whose header lacks the content size).
uncompressed_bytes() {
    local uri="$1" sz n
    if [[ "$uri" == *.zst ]] && ! is_s3 "$uri" && command -v zstd >/dev/null 2>&1; then
        # `|| true`: zstd -lv exits non-zero for a .zst without an embedded content size; under
        # `pipefail` that would otherwise fail this assignment. On non-zero we simply fall through
        # to the ratio estimate below.
        n=$(zstd -lv "$uri" 2>/dev/null | awk -F'[()]' '/Decompressed Size:/ {split($2, a, " "); print a[1]; exit}' || true)
        if [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]]; then
            printf '%s' "$n"
            return
        fi
    fi
    sz=$(file_size_bytes "$uri")
    case "$uri" in
        *.zst|*.gz) printf '%s' "$(( sz * COMPRESS_RATIO ))" ;;
        *)          printf '%s' "$sz" ;;
    esac
}

# group-mode prepare: bin-pack already-split input files into chunks by size (and optional seq
# count) without reading or rewriting any file. Each chunk is a filelist read directly by createdb.
prepare_group() {
    local input_manifest="$1" chunk_dir="$2" chunk_manifest="$3"
    local done_file="${chunk_manifest}.done"
    if [[ -s "$chunk_manifest" ]] && done_exists "$done_file"; then
        log "prepare(group): reusing completed chunk manifest $chunk_manifest"
        return 0
    fi
    rm -rf "${chunk_dir:?}"
    mkdir -p "$chunk_dir" "$(dirname "$chunk_manifest")"
    # group sizing/counting relies on these; guard so a missing tool cannot silently fall back to a
    # ratio estimate and produce different chunk boundaries (a determinism hazard) on a re-run.
    need_cmd awk
    if stream_manifest "$input_manifest" | awk 'NF && $0 !~ /^[[:space:]]*#/ && $1 ~ /\.zst([[:space:]]|$)/ { found = 1 } END { exit !found }'; then
        need_cmd zstd
    fi
    local s3_prefix
    s3_prefix=$(normalize_s3_prefix "$S3_CHUNK_PREFIX")

    # Fail fast if any single input is larger than one chunk: grouping never splits within a
    # file, so an oversized file would produce an oversized chunk (OOM at cluster time). Uses the
    # cheap size estimate (stat x COMPRESS_RATIO for compressed inputs); no file is read.
    local gpath gbytes
    while IFS=$'\t' read -r gpath gbytes _ || [[ -n "${gpath:-}" ]]; do
        [[ -z "${gpath:-}" || "$gpath" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${gbytes:-}" || "$gbytes" == "0" ]] && gbytes=$(uncompressed_bytes "$gpath")
        if [[ "${gbytes:-0}" -gt "$CHUNK_MAX_BYTES" ]]; then
            fail "input file '$gpath' has an estimated uncompressed size of $(( gbytes / 1048576 )) MiB, exceeding the per-chunk limit --chunk-max-bytes=$(( CHUNK_MAX_BYTES / 1048576 )) MiB. Grouping cannot split within a single file: either raise --chunk-max-bytes to fit your machine memory, or pre-split '$gpath' into smaller pieces."
        fi
    done < <(stream_manifest "$input_manifest")

    local local_manifest="${chunk_manifest}.local"
    local chunk_manifest_tmp="${chunk_manifest}.tmp"
    : > "$local_manifest"
    : > "$chunk_manifest_tmp"

    # Resolve each input to "path <tab> bytes <tab> seqs" (fill missing bytes from file size),
    # then greedily bin-pack into chunk filelists (each listing the member input URIs verbatim).
    # When --chunk-max-seqs is set, group mode must know per-file sequence counts, which it can only
    # get by reading each input once (a full decompress pass). This is opt-in via --chunk-max-seqs;
    # without it, group packing is byte-only and reads nothing.
    local count_seqs=0
    if [[ "${CHUNK_MAX_SEQS:-0}" -gt 0 ]]; then
        count_seqs=1
        log "prepare(group): --chunk-max-seqs=$CHUNK_MAX_SEQS set; counting sequences in every input (one full pass over the data)"
    fi
    local path bytes seqs
    while IFS=$'\t' read -r path bytes seqs || [[ -n "${path:-}" ]]; do
        [[ -z "${path:-}" || "$path" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${bytes:-}" || "$bytes" == "0" ]] && bytes=$(uncompressed_bytes "$path")
        if [[ "$count_seqs" -eq 1 ]]; then
            # count only when the manifest didn't already supply a count (3-column rep manifests do)
            if [[ -z "${seqs:-}" || "$seqs" == "0" ]]; then
                seqs=$(stream_uri "$path" | awk '/^>/ {n++} END {print n + 0}')
            fi
            # oversize check runs whether the count was supplied or measured (grouping can't split a file)
            if [[ "${seqs:-0}" -gt "$CHUNK_MAX_SEQS" ]]; then
                fail "input file '$path' has ${seqs} sequences, exceeding the per-chunk limit --chunk-max-seqs=$CHUNK_MAX_SEQS. Grouping cannot split within a single file: raise --chunk-max-seqs or pre-split '$path' into smaller pieces."
            fi
        fi
        printf '%s\t%s\t%s\n' "$path" "${bytes:-0}" "${seqs:-0}"
    done < <(stream_manifest "$input_manifest") \
    | awk -F'\t' -v dir="$chunk_dir" -v manifest="$local_manifest" \
          -v max_bytes="$CHUNK_MAX_BYTES" -v max_seqs="$CHUNK_MAX_SEQS" '
        function open_chunk() { cid++; flist = sprintf("%s/chunk-%08d.filelist.tsv", dir, cid); cb = 0; cs = 0; n = 0 }
        function close_chunk() {
            if (n > 0) { printf "chunk-%08d\t%s\t%d\t%d\n", cid, flist, cs, cb >> manifest; close(manifest); close(flist) }
        }
        BEGIN { cid = -1; flist = ""; cb = 0; cs = 0; n = 0 }
        {
            b = $2 + 0; s = $3 + 0
            if (flist == "") open_chunk()
            # new chunk when adding this file would exceed a limit; a single oversized file gets its
            # own chunk (assumes one input file fits one machine, so no intra-file split).
            if (n > 0 && ((max_bytes > 0 && cb + b > max_bytes) ||
                          (max_seqs  > 0 && s > 0 && cs + s > max_seqs))) { close_chunk(); open_chunk() }
            print $1 > flist
            cb += b; cs += s; n++
        }
        END { close_chunk() }
    '
    [[ -s "$local_manifest" ]] || fail "no input files found in manifest: $input_manifest"

    # Publish each chunk filelist. For S3 runs the filelist must live on S3 too (worker containers
    # cannot see the driver's local files), so upload it and record the S3 URI; the member URIs
    # inside stay as-is (already S3/local paths the worker resolves via resolve_chunk_filelist).
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
    # Auto-select the chunking strategy from the input file COUNT (no reading needed): a single
    # input file must be split into equal chunks (repartition); multiple files are bin-packed
    # as-is by size (group).
    local nfiles
    nfiles=$(stream_manifest "$input_manifest" | awk 'NF && $0 !~ /^[[:space:]]*#/' | wc -l | tr -d '[:space:]')
    if [[ "${nfiles:-0}" -gt 1 ]]; then
        # Grouping bin-packs by byte size (sequence counts are unknown without reading every
        # file), so a positive byte limit is required.
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
    need_cmd zstd
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
                # Stream each chunk straight into its own zstd, so NO uncompressed .fa is ever
                # written to disk -- prepare peak = compressed chunks only, not the full input.
                out = sprintf("%s/chunk-%08d.fa.zst", chunk_dir, chunk_id)
                cmd = "zstd -q -3 -T" zthreads " -c > \"" out "\""
                chunk_seqs = 0
                chunk_bytes = 0
            }
            function close_chunk() {
                if (cmd != "" && chunk_seqs > 0) {
                    if (close(cmd) != 0) {
                        printf "zstd failed while writing %s\n", out > "/dev/stderr"
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
        # raw_chunk is already a compressed chunk (chunk-*.fa.zst) from the streaming awk above --
        # no separate compression step, and no uncompressed .fa was ever written to disk.
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

# Resolve a group filelist into a local createdb .tsv for hard-copy mode. Every member is exposed
# under a UNIQUE, order-preserving basename "member-<NNN><ext>". This matters for determinism --
# createdb sorts inputs by basename with a non-stable sort (createdb.cpp), so passing original paths
# with duplicate basenames would give implementation-defined DB key / representative tie-breaks.
resolve_chunk_filelist() {
    local filelist="$1" work_dir="$2" out_tsv="$3"
    local mem i=0 local_ref ext
    : > "$out_tsv"
    while IFS= read -r mem || [[ -n "${mem:-}" ]]; do
        [[ -z "${mem:-}" || "$mem" =~ ^[[:space:]]*# ]] && continue
        case "$mem" in
            *.zst) ext=".zst" ;;
            *.gz)  ext=".gz" ;;
            *)     ext="" ;;
        esac
        printf -v local_ref '%s/member-%08d%s' "$work_dir" "$i" "$ext"
        if is_s3 "$mem"; then
            copy_in "$mem" "$local_ref"      # download, keep .zst/.gz compression
        else
            ln -sf "$mem" "$local_ref"       # symlink: no copy, unique ordered basename
        fi
        printf '%s\n' "$local_ref" >> "$out_tsv"
        i=$((i + 1))
    done < <(stream_manifest "$filelist")
    [[ -s "$out_tsv" ]] || fail "empty filelist for chunk: $filelist"
}

# Remove a chunk's scratch. Normally the whole dedicated work dir goes; but the hidden
# linclust-batch-worker / cluster-batch-worker entry points exec the workflow script from INSIDE
# their work dir (programDir == work_dir), so there keep the script and scrub only what it created.
# Keeps *.fa: after the source chunk is reclaimed it is the only input a retry can rebuild from.
scrub_chunk_workdir() {
    local d="$1" f
    [[ -n "$d" && -d "$d" ]] || return 0
    rm -rf "$d/db" "$d/tmp" "$d/result"
    for f in "$d"/*; do
        [[ -e "$f" ]] || continue
        case "$f" in
            *.fa | "$BATCH_SCRIPT") continue ;;
        esac
        rm -rf "$f"
    done
    rmdir "$d" 2>/dev/null || true
}

cluster_chunk() {
    [[ "$#" -ge 3 && "$#" -le 6 ]] || usage
    local chunk_uri="$1"
    local result_prefix="$2"
    local work_dir="$3"
    local chunk_id
    chunk_id="${4:-$(basename_no_compression "$chunk_uri")}"
    local round="${5:-${BATCH_WORKER_ROUND:-0}}"
    # round drives round_cluster_par AND round_mmseqs (binary selection); a non-numeric value would
    # be treated as 0 by the round_* helpers' arithmetic test and silently pick round0's binary/params.
    [[ "$round" =~ ^[0-9]+$ ]] || fail "cluster-chunk: round must be a non-negative integer (got '$round')"
    local expected_seqs="${6:-${BATCH_EXPECTED_SEQS:-0}}"
    local prefix
    prefix=$(normalize_s3_prefix "$result_prefix")
    local done_uri="${prefix}done/${chunk_id}.done"

    if done_exists "$done_uri"; then
        log "cluster-chunk: reusing completed ${chunk_id}"
        return 0
    fi

    # Not-done (the done-check above returned early otherwise), so any leftovers in this chunk's
    # dedicated work dir are from a previous failed attempt -- scrub before starting so materialized
    # inputs (member-* symlinks/downloads) can't pile up.
    scrub_chunk_workdir "$work_dir"
    mkdir -p "$work_dir/db" "$work_dir/tmp" "$work_dir/result"
    # With cleanup enabled, scrub on ANY exit (failure/dead-letter, or a SLURM walltime / OOM
    # SIGTERM) so a chunk that gives up never leaves large scratch behind. cluster-chunk runs as its
    # own process; the trap fires after this function returns (work_dir out of local scope), so pass
    # the path via a global. SIGKILL cannot be trapped -- the start-of-run scrub covers that on retry.
    CLUSTER_CHUNK_WORKDIR="$work_dir"
    [[ -n "${REMOVE_TMP:-}" ]] && trap 'scrub_chunk_workdir "$CLUSTER_CHUNK_WORKDIR"' EXIT INT TERM HUP

    # createdb input: by default mode 0 reads FASTA/.zst directly. Explicit mode 1 materializes one
    # node-local FASTA first, avoiding multi-file softlink DBs whose header DB is not safe for
    # createsubdb/convert2fasta.
    local createdb_input
    case "$chunk_uri" in
        *.filelist.tsv)
            if round_createdb_softlink "$round"; then
                createdb_input="$work_dir/${chunk_id}.fa"
                materialize_softlink_filelist "$chunk_uri" "$createdb_input"
            else
                createdb_input="$work_dir/${chunk_id}.filelist.tsv"
                resolve_chunk_filelist "$chunk_uri" "$work_dir" "$createdb_input"
            fi
            ;;
        *)
            if round_createdb_softlink "$round"; then
                createdb_input="$work_dir/${chunk_id}.fa"
                materialize_softlink_fasta "$chunk_uri" "$createdb_input"
                # free the local source chunk now; a retry rebuilds from the materialized FASTA
                if [[ -n "${REMOVE_TMP:-}" && "${BATCH_DELETE_SOURCE_CHUNK:-0}" == "1" && "$chunk_uri" != s3://* && -f "$chunk_uri" ]]; then
                    rm -f "$chunk_uri"
                fi
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

    local db="$work_dir/db/seq"
    local clu="$work_dir/db/clu"
    local rep="$work_dir/db/rep"
    local tsv="$work_dir/result/${chunk_id}.cluster.tsv"
    local rep_fa="$work_dir/result/${chunk_id}.rep.fa"
    local metrics="$work_dir/result/${chunk_id}.metrics.tsv"

    # Resolve the round's mmseqs binary (ROUND0_MMSEQS for round0 when set). All mmseqs calls below
    # use it, so a round0 node pool of a different architecture runs its own binary.
    local mmseqs_bin
    mmseqs_bin=$(round_mmseqs "$round")
    need_cmd "$mmseqs_bin"
    log "createdb ${chunk_id}"
    # shellcheck disable=SC2046,SC2086
    "$mmseqs_bin" createdb "$createdb_input" "$db" $(round_createdb_par "$round") || fail "createdb failed (chunk ${chunk_id}, rc=$?)"
    local actual_seqs
    actual_seqs=$(wc -l < "${db}.index" | tr -d ' ')
    if [[ "${expected_seqs:-0}" =~ ^[1-9][0-9]*$ && "$actual_seqs" -ne "$expected_seqs" ]]; then
        fail "createdb sequence-count mismatch for ${chunk_id}: manifest=${expected_seqs}, db.index=${actual_seqs}. Input may be truncated or corrupt."
    fi
    if round_createdb_softlink "$round" && [[ ! -L "$db" || ! -L "${db}_h" ]]; then
        fail "createdb-mode 1 fell back to a copied DB for ${chunk_id}. Batch softlink mode requires plain single-line FASTA; refusing silent NVMe expansion."
    fi

    log "${CLUSTER_CMD} ${chunk_id}"
    # shellcheck disable=SC2046,SC2086
    "$mmseqs_bin" ${CLUSTER_CMD} "$db" "$clu" "$work_dir/tmp" $(round_cluster_par "$round") || fail "${CLUSTER_CMD} failed (chunk ${chunk_id}, rc=$?)"

    log "createtsv ${chunk_id}"
    # shellcheck disable=SC2086
    "$mmseqs_bin" createtsv "$db" "$db" "$clu" "$tsv" ${CREATETSV_PAR} || fail "createtsv failed (chunk ${chunk_id}, rc=$?)"

    log "representatives ${chunk_id}"
    "$mmseqs_bin" createsubdb "$clu" "$db" "$rep" --subdb-mode 1 || fail "createsubdb failed (chunk ${chunk_id}, rc=$?)"
    "$mmseqs_bin" convert2fasta "$rep" "$rep_fa" || fail "convert2fasta failed (chunk ${chunk_id}, rc=$?)"

    local batch_suffix tsv_out rep_out
    batch_suffix=$(batch_compression_suffix)
    tsv_out="${tsv}${batch_suffix}"
    rep_out="${rep_fa}${batch_suffix}"
    if compress_batch_outputs_enabled; then
        write_batch_output "$tsv" "$tsv_out"
        write_batch_output "$rep_fa" "$rep_out"
    fi

    {
        printf 'chunk_id\t%s\n' "$chunk_id"
        printf 'input_uri\t%s\n' "$chunk_uri"
        printf 'seq_count\t%s\n' "$actual_seqs"
        printf 'input_bytes\t%s\n' "$(wc -c < "$db" 2>/dev/null | tr -d ' ' || echo 0)"
        printf 'rep_count\t%s\n' "$(grep -c '^>' "$rep_fa" || true)"
        printf 'rep_bytes\t%s\n' "$(wc -c < "$rep_fa" 2>/dev/null | tr -d ' ' || echo 0)"
        printf 'cluster_tsv\t%s\n' "$(basename "$tsv_out")"
        printf 'rep_fasta\t%s\n' "$(basename "$rep_out")"
    } > "$metrics"

    copy_out "$tsv_out" "${prefix}tsv/$(basename "$tsv_out")"
    copy_out "$rep_out" "${prefix}rep/$(basename "$rep_out")"
    copy_out "$metrics" "${prefix}metrics/$(basename "$metrics")"
    mark_done "$done_uri" "$work_dir/result/${chunk_id}.done"

    # Reclaim the shared source chunk now that its result is durably out (done-marker written), so the
    # shared chunk dir shrinks as the round progresses instead of holding every chunk to round end.
    # Safe: later rounds cluster representatives (not this round's chunks) and the reconcile keys on
    # done-markers, not chunk presence -- a retry past this point reuses the marker. LOCAL prepare-
    # generated chunk only (repartition .fa.zst / group filelist); never an s3:// URI or user input.
    if [[ -n "${REMOVE_TMP:-}" && "${BATCH_DELETE_SOURCE_CHUNK:-0}" == "1" && "$chunk_uri" != s3://* && -f "$chunk_uri" ]]; then
        rm -f "$chunk_uri"
    fi

    # Free this chunk's working set before the next chunk's createdb (gated, MMseqs style).
    # With REMOVE_TMP off the dir is kept for inspection; note it then accumulates across chunks.
    [[ -n "${REMOVE_TMP:-}" ]] && rm -rf "${work_dir:?}"
    log "cluster-chunk complete: ${chunk_id}"
}


# Distributed representative merge (hash-bucketed shuffle join).
# Composes child(rep->member) with parent(rep->member) on parent.member == child.rep
# to emit (new rep, original member). Sharded by hashing the join key into MERGE_BUCKETS
# with the SAME hash on both sides, so each bucket join is independent and the union
# equals the single-node join. The (rep,member) MAPPING is identical regardless of
# MERGE_BUCKETS or node count; the row ORDER of the final file matches a single global
# sort only for a fixed --merge-buckets (see finalize_outputs).

# Deterministic byte-hash of a chosen column into [0, B); LC_ALL=C fixes byte order.
BUCKET_HASH_AWK='
    function bucket(key,   i, h, n) {
        h = 0; n = length(key)
        for (i = 1; i <= n; i++) h = (h * 131 + ord[substr(key, i, 1)]) % B
        return h
    }
    BEGIN { for (i = 0; i < 256; i++) ord[sprintf("%c", i)] = i }
'

# Shuffle an input manifest into B per-bucket files by hashing a key column.
# merge_partition <child|parent|rep> <input_manifest> <out_dir> <buckets>
# child/rep hash column 1 (rep); parent hashes column 2 (member).
merge_partition() {
    local mode="$1" input_manifest="$2" out_dir="$3" buckets="$4"
    local keycol
    case "$mode" in
        child|rep) keycol=1 ;;
        parent)    keycol=2 ;;
        *) fail "merge_partition mode must be child|parent|rep" ;;
    esac
    # awk keeps one output file open per bucket for the whole pass; keep B well under the
    # process fd limit so routing never dies mid-shuffle with "too many open files".
    local fd_limit; fd_limit=$(ulimit -n 2>/dev/null || echo 256)
    [[ "$buckets" -le $((fd_limit - 64)) ]] ||
        fail "--merge-buckets ${buckets} exceeds this node's safe open-file budget (ulimit -n=${fd_limit}); lower --merge-buckets or raise 'ulimit -n'"
    mkdir -p "$out_dir"

    # Pre-create all B files so an empty bucket still yields a (zero-line) file for the join.
    local b bkt
    for ((b = 0; b < buckets; b++)); do
        printf -v bkt '%s/%s.bkt%05d.tsv' "$out_dir" "$mode" "$b"
        : > "$bkt"
    done

    # Stream every file in the manifest and route each line to its bucket by hash(key)%B. Manifest
    # lines are either "<uri>", "<uri>\tbytes[\tseqs]", or "<id>\t<uri>\t..." (chunk manifests).
    local first second third rest uri
    {
        while IFS=$'\t ' read -r first second third rest || [[ -n "${first:-}" ]]; do
            [[ -z "${first:-}" || "$first" =~ ^# ]] && continue
            if [[ -n "${second:-}" && "$second" =~ ^[0-9]+$ &&
                  ( -z "${third:-}" || "$third" =~ ^[0-9]+$ ) ]]; then
                uri="$first"
            elif [[ -n "${second:-}" && "$first" != s3://* && "$first" != /* && "$first" != ./* ]]; then
                uri="$second"
            else
                uri="$first"
            fi
            stream_uri "$uri"
        done < <(stream_manifest "$input_manifest")
    } | awk -F'\t' -v B="$buckets" -v kc="$keycol" -v pfx="$out_dir/${mode}.bkt" \
          "$BUCKET_HASH_AWK"'
        $kc == "" { next }
        { print > (pfx sprintf("%05d.tsv", bucket($kc))) }
    '
}

run_bucket_jobs() {
    local label="$1" buckets="$2" jobs="$3" worker="$4"
    shift 4
    [[ "$jobs" -gt "$buckets" ]] && jobs="$buckets"
    [[ "$jobs" -lt 1 ]] && jobs=1

    local pids="" active=0
    local b pid rc=0
    for ((b = 0; b < buckets; b++)); do
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
            [[ "$rc" -eq 0 ]] || fail "${label}: one or more bucket jobs failed"
        fi
    done
    for pid in $pids; do
        if ! wait "$pid"; then
            rc=1
        fi
    done
    [[ "$rc" -eq 0 ]] || fail "${label}: one or more bucket jobs failed"
}

merge_join_job() {
    local part_dir="$1" out_dir="$2" sort_tmp="$3" bucket="$4"
    merge_join "$part_dir" "$out_dir" "$bucket" "$sort_tmp"
}

finalize_sort_bucket_job() {
    local part_dir="$1" out_dir="$2" sort_tmp="$3" bucket="$4"
    finalize_sort_bucket "$part_dir" "$out_dir" "$bucket" "$sort_tmp"
}

finalize_emit_shard_job() {
    local sorted_dir="$1" shard_prefix="$2" work_dir="$3" bucket="$4"
    local bb batch_suffix bkt shard_out shard_tmp
    printf -v bb '%05d' "$bucket"
    batch_suffix=$(batch_compression_suffix)
    printf -v bkt '%s/final.bkt%s.tsv' "$sorted_dir" "$bb"
    [[ -e "$bkt" ]] || fail "finalize: missing sorted bucket $bkt"
    shard_out="${shard_prefix}final.bkt${bb}.tsv${batch_suffix}"
    shard_tmp="$work_dir/final.bkt${bb}.tsv${batch_suffix}.out.tmp.$$"
    write_batch_output "$bkt" "$shard_tmp"
    copy_out "$shard_tmp" "$shard_out"
    rm -f "$shard_tmp"
}

# Join one bucket: parent.member == child.rep -> (parent.rep, child.member). One shard out.
# merge_join <part_dir> <out_dir> <bucket> <sort_tmp>  (child/parent bucket files live in part_dir)
merge_join() {
    local part_dir="$1" out_dir="$2" bucket="$3" sort_tmp="$4"
    local bb; printf -v bb '%05d' "$bucket"
    local out
    out="$out_dir/propagated.bkt${bb}.tsv$(batch_compression_suffix)"
    if done_exists "$out"; then
        log "merge_join: reusing bucket ${bb}"
        return 0
    fi
    sort_tmp="$sort_tmp/bkt${bb}"
    mkdir -p "$out_dir" "$sort_tmp"
    local child="$part_dir/child.bkt${bb}.tsv"
    local parent="$part_dir/parent.bkt${bb}.tsv"
    local child_sorted="$part_dir/.child.bkt${bb}.by_rep"
    local parent_sorted="$part_dir/.parent.bkt${bb}.by_member"
    local joined="$part_dir/.joined.bkt${bb}"
    # shellcheck disable=SC2086
    sort -T "$sort_tmp" -t $'\t' -k1,1 "$child" -o "$child_sorted" $SORT_PARALLEL_OPT
    # shellcheck disable=SC2086
    sort -T "$sort_tmp" -t $'\t' -k2,2 "$parent" -o "$parent_sorted" $SORT_PARALLEL_OPT
    join -t $'\t' -1 2 -2 1 -o '1.1 2.2' "$parent_sorted" "$child_sorted" > "$joined"

    # Every child (rep,member) must find its rep as a parent member exactly once -- guaranteed
    # because child carries a self-link for every previous-round rep, so parent.member covers
    # every child.rep. A shortfall means lost members.
    local child_count joined_count
    child_count=$(awk 'END { print NR + 0 }' "$child")
    joined_count=$(awk 'END { print NR + 0 }' "$joined")
    [[ "$joined_count" -eq "$child_count" ]] ||
        fail "merge_join bucket ${bb} lost cluster members: child=${child_count} joined=${joined_count}"

    # Write via a temp then rename so a crash mid-write never leaves a truncated shard that
    # the existence-based done-check would mistake for complete.
    write_batch_output "$joined" "${out}.tmp.$$"
    mv -f "${out}.tmp.$$" "$out"
    rm -f "$child_sorted" "$parent_sorted" "$joined"
    log "merge_join bucket ${bb}: ${joined_count} members"
}

# Rep-first sort of one final-output bucket (the rep's self-link leads its cluster, then members).
# finalize_sort_bucket <part_dir> <out_dir> <bucket> <sort_tmp>  (reads part_dir/rep.bkt<b>.tsv)
finalize_sort_bucket() {
    local part_dir="$1" out_dir="$2" bucket="$3" sort_tmp="$4"
    local bb; printf -v bb '%05d' "$bucket"
    local out="$out_dir/final.bkt${bb}.tsv"
    sort_tmp="$sort_tmp/bkt${bb}"
    mkdir -p "$out_dir" "$sort_tmp"
    # shellcheck disable=SC2086
    awk 'BEGIN {FS=OFS="\t"} { print $1, ($1 == $2 ? 0 : 1), $2 }' "$part_dir/rep.bkt${bb}.tsv" \
        | sort -T "$sort_tmp" -t $'\t' -k1,1 -k2,2n -k3,3 $SORT_PARALLEL_OPT \
        | awk 'BEGIN {FS=OFS="\t"} { print $1, $3 }' \
        > "$out"
}

# propagate <child_manifest> <parent_manifest> <out_manifest> <work_dir>
# Composes child.rep == parent.member on one node, sharded into MERGE_BUCKETS independent
# sort+joins (smaller sorts, no correctness change). Writes out_manifest listing the shard
# shards (order is irrelevant; the next round re-partitions). B=1 is the plain single join.
propagate() {
    [[ "$#" -eq 4 ]] || usage
    local child_manifest="$1"
    local parent_manifest="$2"
    local out_manifest="$3"
    local work_dir="$4"
    local done_uri="${out_manifest}.done"
    local buckets="${MERGE_BUCKETS:-1}"
    local sort_tmp; sort_tmp=$(resolve_sort_tmp "$work_dir")

    if [[ -s "$out_manifest" ]] && done_exists "$done_uri"; then
        log "propagate: reusing completed $out_manifest"
        return 0
    fi

    # partition = node-local scratch (transient, big); shards = shared FS (this round's OUTPUT, read
    # by the next round's merge, so it must survive on shared storage).
    local part; part=$(resolve_node_scratch "$work_dir" partition)
    local shards="$work_dir/shards"
    rm -rf "${part:?}" "${shards:?}" "${sort_tmp:?}"
    mkdir -p "$work_dir" "$sort_tmp" "$part" "$shards"

    log "propagate: partitioning child/parent into ${buckets} bucket(s)"
    merge_partition child "$child_manifest" "$part" "$buckets"
    merge_partition parent "$parent_manifest" "$part" "$buckets"

    log "propagate: joining ${buckets} bucket(s) with ${MERGE_BUCKET_JOBS} concurrent bucket job(s)"
    local b
    run_bucket_jobs "propagate join" "$buckets" "$MERGE_BUCKET_JOBS" merge_join_job "$part" "$shards" "$sort_tmp"

    local bkt
    : > "${out_manifest}.tmp"
    for ((b = 0; b < buckets; b++)); do
        printf -v bkt '%s/propagated.bkt%05d.tsv%s' "$shards" "$b" "$(batch_compression_suffix)"
        [[ -e "$bkt" ]] || fail "propagate: missing bucket output $bkt"
        printf '%s\n' "$bkt" >> "${out_manifest}.tmp"
    done
    mv "${out_manifest}.tmp" "$out_manifest"
    mark_done "$done_uri" "$work_dir/propagate.done"
    if [[ -n "${REMOVE_TMP:-}" ]]; then
        rm -rf "$part" "$sort_tmp"
    fi
    log "propagated clusters written: ${buckets} shard(s) -> $out_manifest"
}

# Build a manifest listing the EXPECTED artifact for every chunk_id in the chunk manifest.
# Filenames are deterministic (<dir>/<chunk_id><suffix>), so -- unlike a directory find -- this
# never ingests stale outputs left by a previous attempt that used different (positional) chunk ids.
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

# Build the representative manifest as "path<TAB>bytes<TAB>seqs" (interleaved in chunk order) so the
# NEXT round's prepare gets per-file sizes AND sequence counts for FREE -- no FASTA rescan and, when
# --chunk-max-seqs is used, no decompress-and-count pass (the biggest cost at Logan scale):
#   bytes = exact size for plain rep FASTA, or an uncompressed-size estimate for compressed input
#   seqs  = rep_count from this chunk's metrics.tsv (== #sequences in the rep FASTA)
# prepare_group already parses this 3-column form; finalize reads only field 1 (the path).
make_rep_manifest_with_sizes() {
    local chunk_manifest="$1"
    local clustered="$2"
    local out="$3"
    local plain="${out}.plain.$$"
    : > "$plain"
    local cid _rest repf metricsf bytes seqs rep_suffix
    rep_suffix=".rep.fa$(batch_compression_suffix)"
    while IFS=$'\t' read -r cid _rest || [[ -n "${cid:-}" ]]; do
        [[ -z "${cid:-}" || "$cid" =~ ^# ]] && continue
        repf="$clustered/rep/${cid}${rep_suffix}"
        metricsf="$clustered/metrics/${cid}.metrics.tsv"
        bytes=$(uncompressed_bytes "$repf")
        seqs=$(awk -F'\t' '$1 == "rep_count" { print $2 + 0; exit }' "$metricsf" 2>/dev/null || true)
        [[ "$seqs" =~ ^[0-9]+$ ]] || seqs=0
        printf '%s\t%s\t%s\n' "$repf" "${bytes:-0}" "$seqs" >> "$plain"
    done < "$chunk_manifest"
    interleave_manifest_file "$plain" "$out"
    rm -f "$plain"
}

# Build the "rep_path<TAB>bytes<TAB>seqs" manifest (interleaved, chunk order) from the per-chunk
# metrics files instead of stat-ing the rep FASTAs. bytes/seqs come from the metrics' EXACT
# rep_bytes/rep_count (recorded by cluster_chunk from the uncompressed rep FASTA), so the next round's
# chunk boundaries never depend on a COMPRESS_RATIO estimate or on --threads (compressed size), and no
# S3 head-object per rep is needed. Reads metrics via stream_uri (URI-agnostic: local or s3://).
# rep_dir is where the rep FASTAs live (no trailing slash, as returned by join_uri).
make_rep_manifest_from_metrics() {
    local metrics_manifest="$1" rep_dir="$2" out="$3"
    local plain="${out}.plain.$$"
    : > "$plain"
    local uri
    while IFS= read -r uri || [[ -n "${uri:-}" ]]; do
        [[ -z "${uri:-}" || "$uri" =~ ^# ]] && continue
        stream_uri "$uri" | awk -F'\t' -v dir="$rep_dir" '
            $1 == "rep_fasta" { fa = $2 }
            $1 == "rep_bytes" { b  = $2 }
            $1 == "rep_count" { s  = $2 }
            END {
                if (fa == "") { print "batch: metrics file has no rep_fasta line" > "/dev/stderr"; exit 1 }
                printf "%s/%s\t%s\t%s\n", dir, fa, (b == "" ? 0 : b), (s == "" ? 0 : s)
            }' >> "$plain"
    done < <(stream_manifest "$metrics_manifest")
    interleave_manifest_file "$plain" "$out"
    rm -f "$plain"
}

interleave_manifest_file() {
    local input="$1"
    local out="$2"
    awk '
        { entries[NR] = $0 }
        END {
            half = int((NR + 1) / 2)
            for (i = 1; i <= half; i++) {
                if (entries[i] != "") print entries[i]
                if (entries[i + half] != "") print entries[i + half]
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
    # ($3 tsv_manifest, $4 rep_manifest kept for call-site compatibility.)
    # Per-chunk_id check, not file counts: every chunk in the manifest must have its done-marker,
    # which cluster_chunk writes LAST (after tsv/rep/metrics are copied out), so a present marker
    # implies all four artifacts. Counting files could be satisfied by stale outputs from a previous
    # attempt while a current chunk is missing (the round manifests are built from chunk_ids too, so
    # such stale files are never ingested either).
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

trim_field() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

# Expand a round-specific SLURM nodelist (comma list or "super[001-005]" bracket range)
# into SLURM_NODE_ARRAY.
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

sanitize_path() {
    printf '%s' "$1" | sed 's#[^A-Za-z0-9._-]#_#g'
}

make_chunk_work_dir() {
    local round_work_dir="$1"
    local chunk_id="$2"
    local round="${3:-1}"
    local base
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    if [[ -n "$node_work_dir" ]]; then
        base="$node_work_dir/mmseqs-batch/${USER:-user}/$(sanitize_path "$round_work_dir")"
    else
        base="$round_work_dir/node-work"
    fi
    printf '%s/%s' "$base" "$chunk_id"
}

# finalize <mapping_manifest> <rep_manifest> <result_prefix> <work_dir> [mark_final]
# Hash-partitions the final mapping by rep into MERGE_BUCKETS buckets, rep-first sorts each,
# and writes final_cluster_shards/final.bkt*.tsv[.zst]. final_cluster_manifest.txt lists those
# shards and is the final cluster output. Each cluster is contiguous and rep-led. With B=1 this
# is a global rep sort; with B>1 the clusters are ordered by (bucket, rep), so only inter-cluster
# row order differs.
finalize_outputs() {
    [[ "$#" -ge 4 && "$#" -le 5 ]] || usage
    local mapping_manifest="$1"
    local rep_manifest="$2"
    local result_prefix="$3"
    local work_dir="$4"
    local mark_final="${5:-1}"
    local buckets="${MERGE_BUCKETS:-1}"
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

    # rep-partition and rep-sorted are transient (the final output is copy_out'd to result_prefix), so
    # keep them on node-local scratch (NVMe) instead of the shared FS.
    local part; part=$(resolve_node_scratch "$work_dir" rep-partition)
    local sorted; sorted=$(resolve_node_scratch "$work_dir" rep-sorted)
    rm -rf "${part:?}" "${sorted:?}" "${sort_tmp:?}"
    mkdir -p "$part" "$sorted" "$sort_tmp"

    log "finalize: partitioning final mapping by rep into ${buckets} bucket(s)"
    merge_partition rep "$mapping_manifest" "$part" "$buckets"
    log "finalize: rep-first sorting ${buckets} bucket(s) with ${MERGE_BUCKET_JOBS} concurrent bucket job(s)"
    local b
    run_bucket_jobs "finalize sort" "$buckets" "$MERGE_BUCKET_JOBS" finalize_sort_bucket_job "$part" "$sorted" "$sort_tmp"

    local manifest_tmp="$work_dir/final_cluster_manifest.txt.tmp.$$"
    : > "$manifest_tmp"
    log "finalize: writing ${buckets} final shard(s) with ${MERGE_BUCKET_JOBS} concurrent bucket job(s)"
    run_bucket_jobs "finalize shard write" "$buckets" "$MERGE_BUCKET_JOBS" finalize_emit_shard_job "$sorted" "$final_cluster_shard_prefix" "$work_dir"
    local shard_out bb
    for ((b = 0; b < buckets; b++)); do
        printf -v bb '%05d' "$b"
        shard_out="${final_cluster_shard_prefix}final.bkt${bb}.tsv${batch_suffix}"
        printf '%s\n' "$shard_out" >> "$manifest_tmp"
    done
    copy_out "$manifest_tmp" "$final_cluster_manifest"
    rm -f "$manifest_tmp"

    if [[ -n "$rep_manifest" ]]; then
        local rep_tmp="$work_dir/final_rep_seq.fasta${batch_suffix}.tmp.$$"
        : > "$rep_tmp"
        # rep_manifest may be 3-column (path<TAB>bytes<TAB>seqs, from make_rep_manifest_with_sizes)
        # or a bare path per line; take field 1 either way.
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

    # The rep-partition/sort temp (and the node-local sort spill) are transient; drop them.
    [[ -n "${REMOVE_TMP:-}" ]] && rm -rf "$part" "$sorted" "${sort_tmp:?}"

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

# S3 variant of list_missing_chunks: list the done/ prefix ONCE and diff against the manifest,
# instead of one `aws s3 ls` per chunk (which at the 10k-chunk ceiling is thousands of serial
# round-trips per merge). Writes the present-set to $3. Prints chunk_ids with no <chunk_id>.done.
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
        MMSEQS ROUND0_MMSEQS THREADS CHUNK_MAX_BYTES CHUNK_MAX_SEQS ROUND0_CHUNK_MAX_BYTES ROUND0_CHUNK_MAX_SEQS \
        S3_CHUNK_PREFIX COMPRESS_BATCH_OUTPUTS \
        CREATEDB_PAR ROUND0_CREATEDB_MODE BATCH_DELETE_SOURCE_CHUNK CLUSTER_CMD CLUSTER_COV_MODE CLUSTER_PAR ROUND0_CLUSTER_PAR CREATETSV_PAR SORT_TMP \
        MAX_ROUNDS MIN_REDUCTION_RATIO CONVERGENCE_PATIENCE MIN_REDUCTION_COUNT \
        MAX_CHUNK_ATTEMPTS COMPRESS_RATIO MERGE_BUCKETS MERGE_BUCKET_JOBS BATCH_BACKEND REMOVE_TMP \
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

# Submit one job pinned to a single host. submit_slurm_job <job_name> <log_dir> <wrapper> <node> <round> [depends]
# (pin to one node, never the whole nodelist, or --nodes 1 would grab every listed host).
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
        --cpus-per-task "$THREADS"
        --output "$log_dir/%x-%j.out"
        --error "$log_dir/%x-%j.err"
    )
    [[ -n "$node" ]] && cmd+=(--nodelist "$node")
    # afterany (not afterok): the dependent runs even if a predecessor failed, so the merge/driver
    # can reconcile and retry -- the same self-healing the AWS backend relies on.
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
    [[ "$shard" =~ ^[0-9]+$ && "$num_shards" =~ ^[1-9][0-9]*$ ]] || fail "slurm-worker needs numeric <shard> <num_shards>"

    # process this shard's chunks (line index % num_shards == shard) that are not yet done
    local idx=0 chunk_id chunk_uri seqs rest
    while IFS=$'\t' read -r chunk_id chunk_uri seqs rest || [[ -n "${chunk_id:-}" ]]; do
        [[ -z "${chunk_id:-}" ]] && continue
        if [[ $((idx % num_shards)) -eq "$shard" ]] && ! done_exists "$(chunk_done_uri "$result_prefix" "$chunk_id")"; then
            local task_work rc=0
            task_work=$(make_chunk_work_dir "$round_work_dir" "$chunk_id" "$round")
            log "slurm-worker round ${round} shard ${shard}/${num_shards}: ${chunk_id}"
            set +e
            env BATCH_WORKER_DISPATCH=1 bash "$BATCH_SCRIPT" cluster-chunk "$chunk_uri" "$result_prefix" "$task_work" "$chunk_id" "$round" "${seqs:-0}"
            rc=$?
            set -e
            [[ "$rc" -eq 0 ]] || log "slurm-worker: chunk ${chunk_id} FAILED (rc=$rc); left no done-marker for retry/dead-letter"
        fi
        idx=$((idx + 1))
    done < "$chunk_manifest"
}

aws_resolve_batch_resource_by_tag() {
    local kind="$1"
    local tag_key="$2"
    local tag_value="$3"
    # Resolve the AWS Batch job queue / job definition tagged tag_key=tag_value in ONE Resource Groups
    # Tagging API call (tags ARE the query -- no per-resource ListTags N+1 scan, and no throttle that
    # could mask an ambiguous match). An API error is FATAL (no `|| true`): the caller's $(...) then
    # fails under set -e instead of silently mis-resolving. Prints the single match to stdout; none /
    # multiple / bad-kind -> stderr + non-zero. IAM: tag:GetResources (+ batch:DescribeJobQueues for
    # the queue ENABLED/VALID check). bash-3.2 safe: no associative arrays (awk does the reduction).
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
    # Let the aws CLI itself retry throttled API calls (SubmitJob is rate-limited, and a chain fans out
    # submit-job + s3 cp across many containers). Adaptive retry avoids a half-built chain from a single
    # throttle under set -e. Exported (and forwarded via the names list) so the containers inherit it.
    export AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}"
    export AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-10}"
}

# Round0-specific AWS Batch queue/definition (multi-machine plan: e.g. x2gd round0 / i4i round1+).
# The QUEUE selects the compute environment (instance type); the DEFINITION selects the container
# image (and thus the architecture, ARM vs x86). Empty ROUND0_* => base queue/definition for all
# rounds. Round1+ always uses the base BATCH_AWS_JOB_QUEUE / BATCH_AWS_JOB_DEFINITION.
aws_queue_for_round() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_AWS_JOB_QUEUE:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_AWS_JOB_QUEUE"
    else
        printf '%s' "$BATCH_AWS_JOB_QUEUE"
    fi
}

# else-branch derefs the base vars without :- ; safe because aws_require_submit_env (which asserts
# they are set) always runs before aws_submit_batch_job builds the command.
aws_definition_for_round() {
    local round="$1"
    if [[ "$round" -eq 0 && -n "${ROUND0_BATCH_AWS_JOB_DEFINITION:-}" ]]; then
        printf '%s' "$ROUND0_BATCH_AWS_JOB_DEFINITION"
    else
        printf '%s' "$BATCH_AWS_JOB_DEFINITION"
    fi
}

# Mirror of SLURM's `rm -rf "$work_dir"/round*` on a final result: drop the S3 work prefix (chunks,
# per-round clustered/tsv/rep/metrics, propagated shards, the uploaded script). The final result lives
# under result_prefix and a re-run short-circuits via ${result_prefix}final.done, so this is safe.
# Guarded by REMOVE_TMP; non-fatal (the result is already written). Only called on a FINAL result.
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
    # The run config lives next to the script; each container downloads and sources it (%q-quoted
    # `export` lines from write_batch_exports) instead of receiving ~40 vars via container-overrides.
    local config_uri="${script_uri%/*}/config.env"
    local args
    args=$(shell_join "$@")
    printf 'aws s3 cp %q /tmp/mmseqs-batch.sh --no-progress && aws s3 cp %q /tmp/mmseqs-batch.env --no-progress && chmod +x /tmp/mmseqs-batch.sh && . /tmp/mmseqs-batch.env && /tmp/mmseqs-batch.sh %s' "$script_uri" "$config_uri" "$args"
}

aws_container_overrides() {
    local bootstrap="$1"
    # containerOverrides carries ONLY the command; every config var is sourced from config.env inside
    # the container (see aws_bootstrap_command + write_batch_exports), so there is no per-value JSON
    # escaping and no reserved-name hazard. The bootstrap is a single line, so escaping \ and " (via
    # bash parameter expansion -- deterministic across shells, unlike awk gsub) yields valid JSON.
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
    # The target round is ALWAYS the last script arg -- aws-driver/worker/merge all end with <round>.
    # It picks the round-specific queue/definition. Default to round1 (base queue) if not numeric.
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

    # Worker arrays get an AWS-native retry so an infra kill (spot reclaim / OOM / attempt timeout)
    # that exits a child non-zero is retried on a fresh instance -- the array then still reaches
    # SUCCEEDED, so the merge dependency fires and reconciles. This is the closest AWS Batch has to
    # SLURM's `afterany`. Application failures already exit 0 (reconciled by the merge), so they never
    # consume a retry. Only workers get this: retrying a driver/merge would double-submit its
    # downstream jobs. A chunk that hard-kills on every attempt still fails the array (there is no
    # native afterany); fully closing that residual gap would require an EventBridge trigger.
    #
    # This is a SEPARATE layer from MAX_CHUNK_ATTEMPTS (the merge-level reconcile that re-drives
    # missing chunks): BATCH_AWS_WORKER_ATTEMPTS is per-container infra placement, MAX_CHUNK_ATTEMPTS
    # is logical clustering attempts. They compound, so a chunk can run up to attempts x
    # MAX_CHUNK_ATTEMPTS times, and each attempt gets its own attemptDurationSeconds (a hung chunk
    # burns attempts x BATCH_AWS_TIMEOUT before the merge reconciles). Keep both small (default 2).
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

# Idempotent submit: record the submitted job id at marker_uri and, if that marker already exists,
# ADOPT the recorded id instead of submitting again. This makes re-invoking a driver/merge (a manual
# recovery of a stalled chain, or an EventBridge rule you add) safe -- it cannot fork a second chain or
# a duplicate downstream job. The marker write is best-effort (non-fatal): if it fails the job is still
# submitted, we just lose the dedup guarantee for that one step (the pre-marker behavior). Note we do
# NOT give drivers/merges AWS-native `--retry-strategy attempts>1`: a retried MERGE would re-read the
# incremented chunk_attempts counter and could dead-letter prematurely, so unattended auto-recovery
# still needs an external (EventBridge) trigger -- but that trigger is now safe because of this marker.
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
    # Reuse a completed run (parity with single-node/slurm entry points). NOTE: unlike SLURM,
    # AWS cannot cheaply detect a still-in-flight chain, so do not launch a second aws-submit
    # against the same work prefix while one is running (the chains would race S3 state).
    if done_exists "${result_prefix}final.done"; then
        log "aws-submit: reusing completed result ${result_prefix}$(final_cluster_file_name)"
        return 0
    fi
    aws_require_submit_env

    # Round0 machine override is a pair: queue (instance type) + definition (container image = arch).
    # Setting only one is almost always a mistake -- e.g. round0 on an ARM queue with the base x86
    # image gives "exec format error" only after the instance provisions. Warn, don't fail (a same-arch
    # different-instance-type setup legitimately sets only the queue).
    if [[ -n "${ROUND0_BATCH_AWS_JOB_QUEUE:-}" && -z "${ROUND0_BATCH_AWS_JOB_DEFINITION:-}" ]]; then
        log "warning: ROUND0_BATCH_AWS_JOB_QUEUE set without ROUND0_BATCH_AWS_JOB_DEFINITION; round0 runs the BASE container image on the round0 queue (arch mismatch if the queue is a different CPU arch)"
    fi
    if [[ -z "${ROUND0_BATCH_AWS_JOB_QUEUE:-}" && -n "${ROUND0_BATCH_AWS_JOB_DEFINITION:-}" ]]; then
        log "warning: ROUND0_BATCH_AWS_JOB_DEFINITION set without ROUND0_BATCH_AWS_JOB_QUEUE; round0 runs the round0 image on the BASE queue"
    fi

    # Workers run in separate containers with no view of the submit host's filesystem, so every FASTA
    # path in the manifest must be reachable from a worker -- i.e. s3:// (or a shared mount visible to
    # every worker, e.g. EFS, opted in via BATCH_AWS_ALLOW_NONS3_INPUT=1). Validate at submit so a bad
    # path fails now, not after every worker dies hours later on an unreadable local path.
    # (BATCH_AWS_ALLOW_NONS3_INPUT is used only here on the submit host; it is deliberately NOT written
    # into config.env, so workers never re-validate.)
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

    # Serialize the run config to S3 as %q-quoted `export` lines; every driver/worker/merge container
    # downloads and sources it (write_batch_exports -- shared with the SLURM wrapper, the single source
    # of truth for the config var list). This replaces forwarding ~40 vars through the submit-job
    # container-overrides JSON: overrides now carry only the command, so there is no per-value JSON
    # escaping, no reserved-name hazard, and no 8 KiB override ceiling. The config also lands in S3
    # beside the run, which is what the reproducibility requirement actually wants.
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
    driver_job=$(aws_submit_batch_job "${job_prefix}-driver-r0" 1 "" "$script_uri" \
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

    local script_uri="${BATCH_AWS_SCRIPT_URI:-${work_prefix}scripts/batch_clustering.sh}"
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    local local_root="${node_work_dir:-${BATCH_AWS_LOCAL_DIR:-/tmp/mmseqs-batch}}/${AWS_BATCH_JOB_ID:-manual}-driver-r${round}"
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

    # Seed this round's retry budget unconditionally (matching slurm_driver), so a re-driven
    # round always gets its full MAX_CHUNK_ATTEMPTS rather than inheriting a stale count.
    local attempt_uri attempt_local
    attempt_uri=$(join_uri "$round_prefix" "chunk_attempts.txt")
    attempt_local="$local_root/chunk_attempts.txt"
    write_counter_file "$attempt_local" 1
    copy_out "$attempt_local" "$attempt_uri"

    local job_prefix="${BATCH_AWS_JOB_PREFIX:-mmseqs-batch}"
    local clustered_prefix worker_job merge_job
    clustered_prefix=$(aws_clustered_prefix "$work_prefix" "$round")
    worker_job=$(aws_submit_batch_job_once "$(join_uri "$round_prefix" "submitted_workers.txt")" "${job_prefix}-worker-r${round}" "$chunk_count" "" "$script_uri" \
        aws-worker "$chunk_manifest_s3" "$clustered_prefix" "$round")
    log "aws-driver round ${round}: submitted worker array $worker_job (${chunk_count} chunks)"

    merge_job=$(aws_submit_batch_job_once "$(join_uri "$round_prefix" "submitted_merge.txt")" "${job_prefix}-merge-r${round}" 1 "$worker_job" "$script_uri" \
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
    local local_root="${node_work_dir:-${BATCH_AWS_LOCAL_DIR:-/tmp/mmseqs-batch}}/${AWS_BATCH_JOB_ID:-manual}-worker-r${round}-${index}"
    rm -rf "$local_root"
    mkdir -p "$local_root"

    # A worker must NEVER fail its array child: the merge's dependsOn fires only when the array reaches
    # SUCCEEDED, and reconcile/retry is driven by the chunk's done-marker (not the child exit code).
    # copy_in / manifest parsing / fail() run under `set -e`, so a transient S3 error would otherwise
    # fail the child -> array FAILED -> merge never runs -> chain dies silently. Force exit 0 on ANY
    # exit (an untrappable SIGKILL from spot/OOM still exits non-zero -> the --retry-strategy handles
    # that at the AWS level).
    AWS_WORKER_LOCAL_ROOT="$local_root"
    trap 'rc=$?; [[ "$rc" -eq 0 ]] || log "aws-worker soft-fail (rc=$rc): left no done-marker; merge reconcile will retry/dead-letter"; [[ -n "${REMOVE_TMP:-}" ]] && rm -rf "${AWS_WORKER_LOCAL_ROOT:-}"; exit 0' EXIT

    local manifest_local="$local_root/chunks.tsv"
    copy_in "$chunk_manifest_s3" "$manifest_local"

    local line chunk_id chunk_uri seqs bytes
    # Select the (index+1)-th DATA row with the SAME predicate count_manifest_rows uses to size the
    # array (skip blank / '#'-comment lines), so row selection and array sizing can never disagree; a
    # plain `sed -n Np` on the physical line would silently shift every worker onto the wrong chunk if
    # a comment/blank line ever appeared in chunks.tsv.
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
        log "aws-worker round ${round} index ${index}: chunk ${chunk_id} FAILED (rc=${chunk_rc}); left no done-marker for the merge reconcile to retry/dead-letter"
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

    local script_uri="${BATCH_AWS_SCRIPT_URI:-${work_prefix}scripts/batch_clustering.sh}"
    local node_work_dir
    node_work_dir=$(round_node_work_dir "$round")
    local local_root="${node_work_dir:-${BATCH_AWS_LOCAL_DIR:-/tmp/mmseqs-batch}}/${AWS_BATCH_JOB_ID:-manual}-merge-r${round}"
    rm -rf "$local_root"
    mkdir -p "$local_root"
    # aws-merge has several early returns (retry, round-0, converged); scrub scratch on any exit.
    # The trap fires AFTER aws_merge returns, when the local $local_root is out of scope -- under
    # `set -u` that would abort the trap and make the whole merge process exit non-zero (AWS Batch
    # then marks every merge, including the final one, FAILED). Pass the path via a global, mirroring
    # cluster_chunk's CLUSTER_CHUNK_WORKDIR.
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

    # Reconcile per chunk_id using a SINGLE done/ listing (not one aws-ls per chunk): a chunk is
    # complete iff its <chunk_id>.done exists (written last, implies all artifacts). Stale objects
    # from a previous attempt neither satisfy this nor get ingested -- the manifests below are built
    # from chunk_ids, never from a directory listing.
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

        # Retry ONLY the missing chunks: submit an array of size n_missing over a manifest of just the
        # missing rows, not size=chunk_count over the full manifest (which would boot chunk_count
        # containers -- e.g. 10000 for 3 stragglers -- each only to skip its already-done chunk).
        local missing_manifest_local missing_manifest_s3
        missing_manifest_local="$local_root/missing_chunks.manifest.tsv"
        awk -F'\t' 'NR==FNR { want[$1]=1; next } ($1 in want)' "$missing_file" "$chunk_manifest_local" > "$missing_manifest_local"
        missing_manifest_s3=$(join_uri "$round_prefix" "missing_chunks_attempt${attempts}.manifest.tsv")
        copy_out "$missing_manifest_local" "$missing_manifest_s3"

        job_prefix="${BATCH_AWS_JOB_PREFIX:-mmseqs-batch}"
        # NOTE: these two are NOT wrapped in aws_submit_batch_job_once. The retry counter is a
        # read-modify-write on S3, so a re-invoked merge cannot be made idempotent by a marker keyed on
        # it (it would re-read the bumped counter). Re-invocation safety here would need the counter
        # decoupled from this function; for now let a retry finish rather than re-invoking mid-retry.
        retry_job=$(aws_submit_batch_job "${job_prefix}-worker-r${round}-retry${attempts}" "$n_missing" "" "$script_uri" \
            aws-worker "$missing_manifest_s3" "$clustered_prefix" "$round")
        retry_merge=$(aws_submit_batch_job "${job_prefix}-merge-r${round}-retry${attempts}" 1 "$retry_job" "$script_uri" \
            aws-merge "$input_manifest" "$work_prefix" "$result_prefix" "$round")
        log "aws-merge round ${round}: ${n_missing}/${chunk_count} chunk(s) missing; retrying only those (array size ${n_missing}); worker $retry_job merge $retry_merge"
        return 0
    fi

    # All chunks complete -- build the manifests from chunk_ids (deterministic S3 URIs), never a
    # listing, so stale objects are never ingested. The rep manifest is the 3-column sized form
    # (path<TAB>bytes<TAB>seqs) built from the per-chunk metrics' EXACT rep_bytes/rep_count, so the
    # next round's chunk boundaries do not depend on a COMPRESS_RATIO size estimate and need no S3
    # head-object per rep; the same metrics pass also yields cur_reps (sum of the seqs column).
    make_manifest_from_chunk_ids "$chunk_manifest_local" "$(join_uri "$clustered_prefix" "tsv")" ".cluster.tsv$(batch_compression_suffix)" "$local_root/tsv_manifest.txt"
    make_manifest_from_chunk_ids "$chunk_manifest_local" "$(join_uri "$clustered_prefix" "metrics")" ".metrics.tsv" "$local_root/metrics_manifest.txt"
    make_rep_manifest_from_metrics "$local_root/metrics_manifest.txt" "$(join_uri "$clustered_prefix" "rep")" "$local_root/rep_manifest.txt"
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
        next_driver=$(aws_submit_batch_job_once "$(join_uri "$round_prefix" "submitted_next_driver.txt")" "${job_prefix}-driver-r${next_round}" 1 "" "$script_uri" \
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

    # propagate emits local bucket shards; upload each to S3 so the next round's driver
    # (a separate container) can read them, and record their S3 URIs in the manifest.
    # Idempotency: a re-driven merge (retry, or an infra-killed merge re-run) must not redo the whole
    # hash-bucket join. If a previous attempt already uploaded the S3 propagated_manifest, reuse it.
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
        # Only clean on a FINAL (mark_final) result; a partial MAX_ROUNDS result keeps the work prefix
        # so a re-run with a larger --max-rounds can resume (matches slurm_merge).
        [[ "$mark_final" -eq 1 ]] && aws_cleanup_work_prefix "$work_prefix"
        return 0
    fi

    write_state_file "$local_root/state.env" "$cur_reps" "$low_benefit_rounds"
    copy_out "$local_root/state.env" "$(join_uri "$round_prefix" "state.env")"

    local next_round=$((round + 1))
    local job_prefix="${BATCH_AWS_JOB_PREFIX:-mmseqs-batch}"
    local next_driver
    next_driver=$(aws_submit_batch_job_once "$(join_uri "$round_prefix" "submitted_next_driver.txt")" "${job_prefix}-driver-r${next_round}" 1 "" "$script_uri" \
        aws-driver "$rep_manifest" "$work_prefix" "$result_prefix" "$next_round")
    log "aws-merge round ${round}: submitted next driver $next_driver"

    if [[ -n "${REMOVE_TMP:-}" ]]; then
        rm -rf "$local_root"
    fi
}

# ---------------------------------------------------------------------------
# Multi-node (SLURM) event chain -- the AWS backend's driver/worker/merge model,
# expressed with `sbatch --dependency` on a shared filesystem instead of AWS Batch.
#
# Nothing stays alive on the submitting host: it submits one driver job and returns.
# Each job is short and hands off to the next via a dependency, so no job ever blocks
# holding a node's cores while waiting for another (which would deadlock a co-located
# worker). State between rounds is passed through round<r>/state.env on the shared FS.
# The merge runs on a single node per round (like AWS); MERGE_BUCKETS still splits that
# node's sort/join for RAM/disk headroom.
# ---------------------------------------------------------------------------

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

# Deterministic per-run token derived from the work_dir, embedded in every job name so a
# re-run can detect a still-live chain for the same run (and only that run) in the queue.
run_token() {
    printf '%s' "$1" | cksum | awk '{ print $1 }'
}

# Submit one worker per node that still owns an unfinished chunk (round-robin shard, skipping
# already-done chunks). Best-effort: a single sbatch failure is logged, not fatal, so the merge
# is still created and reconciles the gap -- never leaving workers orphaned with no merge.
# Prints a colon-joined list of the worker job ids that were submitted (may be empty).
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
        wrapper="$slurm_dir/mmseqs-${CLUSTER_CMD}-${tok}-r${round}-w${s}.sh"
        write_slurm_wrapper "$wrapper" slurm-worker "$chunk_manifest" "$clustered" "$round_dir" "$round" "$s" "$nn"
        set +e
        jid=$(submit_slurm_job "mmseqs-${CLUSTER_CMD}-${tok}-r${round}-w${s}" "$slurm_dir" "$wrapper" "${SLURM_NODE_ARRAY[$s]}" "$round")
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
    build_slurm_node_array 0
    [[ "${#SLURM_NODE_ARRAY[@]}" -gt 0 ]] || fail "--backend multi-node requires --slurm-nodelist"
    need_cmd sbatch
    need_cmd squeue
    local slurm_dir="$work_dir/event"
    mkdir -p "$slurm_dir"

    # Refuse to start a second chain for the same work_dir while one is still live in the queue,
    # so two job trees cannot race on the shared round state.
    local tok
    tok=$(run_token "$work_dir")
    if squeue -h -o '%j' 2>/dev/null | grep -q -- "-${tok}-"; then
        fail "a batch-clustering chain for this work_dir (token ${tok}) is already active in the queue; cancel those jobs or wait for them to finish before re-running"
    fi

    # Pin the whole chain to an immutable snapshot of this script, so deferred jobs cannot pick up
    # a re-staged or removed $BATCH_SCRIPT between submit and run.
    local snap="$slurm_dir/batch_clustering.sh"
    cp -f "$BATCH_SCRIPT" "$snap"
    BATCH_SCRIPT="$snap"

    local jid
    jid=$(submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-driver-r0" "$slurm_dir" "" "${SLURM_NODE_ARRAY[0]}" \
        slurm-driver "$input_manifest" "$work_dir" "$result_dir" 0)
    log "submitted batch-clustering driver (round 0) as SLURM job ${jid}."
    log "The run now proceeds on the compute nodes as a self-submitting job chain; the submitting host does no further work (no need to keep this session open). Track with: squeue -u \"\$USER\""
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
    local slurm_dir="$work_dir/event"
    local chunks="$round_dir/chunks"
    local chunk_manifest="$round_dir/chunks.tsv"
    mkdir -p "$round_dir" "$clustered" "$slurm_dir"

    if [[ -s "$chunk_manifest" ]] && done_exists "${chunk_manifest}.done"; then
        log "driver r${round}: reusing prepared chunks"
    else
        prepare_round "$round" "$input_manifest" "$chunks" "$chunk_manifest"
    fi
    local chunk_count
    chunk_count=$(count_manifest_rows "$chunk_manifest")
    [[ "$chunk_count" -gt 0 ]] || fail "driver r${round}: no chunks in $chunk_manifest"

    write_counter_file "$round_dir/chunk_attempts.txt" 1
    local tok dep mjid
    tok=$(run_token "$work_dir")
    # Always submit the merge (afterany on whatever workers were submitted, or immediately if none),
    # so a partial worker submit can never orphan the round -- the merge reconciles and retries.
    dep=$(submit_round_workers "$chunk_manifest" "$clustered" "$round_dir" "$round" "$slurm_dir" "$tok")
    mjid=$(submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-merge-r${round}" "$slurm_dir" "$dep" "${SLURM_NODE_ARRAY[0]}" \
        slurm-merge "$input_manifest" "$work_dir" "$result_dir" "$round")
    log "driver r${round}: ${chunk_count} chunk(s); submitted worker(s) and merge ${mjid}"
}

# One round's merge: reconcile chunk outputs (retry if incomplete), propagate the mapping
# (single node), decide convergence, then finalize or submit the next driver, and exit.
slurm_merge() {
    [[ "$#" -eq 4 ]] || usage
    local input_manifest="$1" work_dir="$2" result_dir="$3" round="$4"
    build_slurm_node_array "$round"
    local nn="${#SLURM_NODE_ARRAY[@]}"
    [[ "$nn" -gt 0 ]] || fail "merge: empty node array"

    local round_dir="$work_dir/round${round}"
    local clustered="$round_dir/clustered"
    local slurm_dir="$work_dir/event"
    local chunk_manifest="$round_dir/chunks.tsv"
    mkdir -p "$clustered" "$slurm_dir"

    local tok
    tok=$(run_token "$work_dir")

    # Reconcile per chunk_id (not file counts): a chunk is complete iff its done-marker exists, which
    # cluster_chunk writes LAST (after tsv/rep/metrics), so the marker implies all artifacts. Counting
    # files could be satisfied by stale outputs from a previous attempt while a current chunk is
    # missing. If any chunk is missing, retry workers + merge (afterany) up to MAX_CHUNK_ATTEMPTS.
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
        log "merge r${round}: incomplete (${n_missing}/${chunk_count} chunk(s) missing); retry attempt ${attempts}"
        local dep
        dep=$(submit_round_workers "$chunk_manifest" "$clustered" "$round_dir" "$round" "$slurm_dir" "$tok")
        submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-merge-r${round}-a${attempts}" "$slurm_dir" "$dep" "${SLURM_NODE_ARRAY[0]}" \
            slurm-merge "$input_manifest" "$work_dir" "$result_dir" "$round" >/dev/null
        return 0
    fi

    # All chunks complete -- build the round manifests from chunk_ids (deterministic filenames),
    # never a directory find, so stale outputs from a previous attempt are never ingested.
    local tsv_manifest="$clustered/tsv_manifest.txt"
    local rep_manifest="$clustered/rep_manifest.txt"
    local metrics_manifest="$clustered/metrics_manifest.txt"
    make_manifest_from_chunk_ids "$chunk_manifest" "$clustered/tsv" ".cluster.tsv$(batch_compression_suffix)" "$tsv_manifest"
    make_rep_manifest_with_sizes "$chunk_manifest" "$clustered" "$rep_manifest"
    make_manifest_from_chunk_ids "$chunk_manifest" "$clustered/metrics" '.metrics.tsv' "$metrics_manifest"

    local cur_reps
    cur_reps=$(count_reps_from_metrics_manifest "$metrics_manifest")
    log "merge r${round}: ${cur_reps} representatives across ${chunk_count} chunk(s)"

    if [[ "$round" -eq 0 ]]; then
        if [[ "$chunk_count" -le 1 ]]; then
            log "merge r0: input fit in one chunk; finalizing"
            with_round_node_work_dir "$round" finalize_outputs "$tsv_manifest" "$rep_manifest" "$result_dir" "$round_dir/final" 1
            # final result is in result_dir; the round scratch is now safe to drop.
            [[ -n "${REMOVE_TMP:-}" ]] && rm -rf "$work_dir"/round*
            return 0
        fi
        write_state_file "$round_dir/state.env" "$cur_reps" 0
        local nd next_driver_node
        next_driver_node=$(first_slurm_node_for_round 1)
        nd=$(submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-driver-r1" "$slurm_dir" "" "$next_driver_node" \
            slurm-driver "$rep_manifest" "$work_dir" "$result_dir" 1)
        log "merge r0: ${cur_reps} reps across ${chunk_count} chunks; submitted next driver ${nd}"
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
        log "merge r${round}: low-benefit round ${low_benefit_rounds}/${CONVERGENCE_PATIENCE} (removed ${reduction} representatives)"
    else
        low_benefit_rounds=0
    fi

    local converged=0 mark_final=1
    if [[ "$chunk_count" -le 1 ]]; then
        converged=1; log "merge r${round}: representatives fit in one chunk"
    elif [[ "$low_benefit_rounds" -ge "$CONVERGENCE_PATIENCE" ]]; then
        converged=1; log "merge r${round}: representatives converged by low-benefit threshold"
    elif [[ "$round" -ge "$MAX_ROUNDS" ]]; then
        converged=1; mark_final=0; log "merge r${round}: reached MAX_ROUNDS=${MAX_ROUNDS}; writing partial clustering"
    fi

    if [[ "$converged" -eq 1 ]]; then
        with_round_node_work_dir "$round" finalize_outputs "$propagated_manifest" "$rep_manifest" "$result_dir" "$round_dir/final" "$mark_final"
        # Only drop the round scratch on a final (converged) result; a partial (MAX_ROUNDS) result
        # is left intact so a re-run with a larger --max-rounds can resume.
        [[ "$mark_final" -eq 1 && -n "${REMOVE_TMP:-}" ]] && rm -rf "$work_dir"/round*
        return 0
    fi

    write_state_file "$round_dir/state.env" "$cur_reps" "$low_benefit_rounds"
    local next_round=$((round + 1)) nd next_driver_node
    next_driver_node=$(first_slurm_node_for_round "$next_round")
    nd=$(submit_event_step "mmseqs-${CLUSTER_CMD}-${tok}-driver-r${next_round}" "$slurm_dir" "" "$next_driver_node" \
        slurm-driver "$rep_manifest" "$work_dir" "$result_dir" "$next_round")
    log "merge r${round}: submitted next driver ${nd}"
}

# Single-node driver: prepare -> cluster chunks -> propagate rounds -> finalize, all in
# process. (Multi-node uses the SLURM event chain; AWS uses aws-driver/worker/merge.)
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
    make_manifest_from_chunk_ids "$chunk_manifest" "$work_dir/round${round}/clustered/tsv" ".cluster.tsv$(batch_compression_suffix)" "$current_mapping_manifest"
    make_rep_manifest_with_sizes "$chunk_manifest" "$work_dir/round${round}/clustered" "$current_rep_manifest"
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
        make_manifest_from_chunk_ids "$chunk_manifest" "$work_dir/round${round}/clustered/tsv" ".cluster.tsv$(batch_compression_suffix)" "$parent_manifest"
        make_rep_manifest_with_sizes "$chunk_manifest" "$work_dir/round${round}/clustered" "$current_rep_manifest"
        make_manifest_from_chunk_ids "$chunk_manifest" "$work_dir/round${round}/clustered/metrics" '.metrics.tsv' "$round_metrics_manifest"

        # propagate dispatches its bucket tasks per backend (single-node loop / multi-node
        # fan-out); it emits MERGE_BUCKETS mapping shards listed in propagated_manifest.
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
    # finalize dispatches its rep-first-sort buckets per backend and marks final.done only
    # when mark_final=1, so a partial (MAX_ROUNDS) result is never mistaken for complete.
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
    # Multi-node uses the SLURM event chain (self-submitting driver/worker/merge jobs),
    # not the shared blocking loop -- so all work, including the merge, runs on the
    # specified compute nodes and the submitting host only fires the first job.
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
