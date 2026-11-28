#!/bin/sh -e
fail() {
    echo "Error: $1"
    exit 1
}

# true if none of the files exists, pass later outputs too
notExists() {
    for FILE in "$@"; do
        [ -f "$FILE" ] && return 1
    done
    return 0
}

# also check the refinement outputs with two iterations
notExistsBeforeRefinement() {
    notExists "$@" || return 1
    [ "$LINCLUST_ITERATIONS" -ge 2 ] || return 0
    notExists "${TMP_PATH}/clu_merged.dbtype" "${TMP_PATH}/input_rep.dbtype" "${TMP_PATH}/pref_rep.dbtype" "${TMP_PATH}/clu_rep.dbtype"
}

# rm -f of many node files, printf is a shell builtin and does not hit ARG_MAX
removeFiles() {
    printf '%s\0' "$@" | xargs -0 rm -f
}

# data and index files of a linfinity cluster DB, one per node
removeClusterDb() {
    removeFiles "$1" "$1.dbtype" "$1.index" "$1.index."[0-9] "$1.index."[0-9][0-9] "$1.index."[0-9][0-9][0-9] "$1."[0-9] "$1."[0-9][0-9] "$1."[0-9][0-9][0-9]
}

# end all tasks of a srun step if one fails
export SLURM_KILL_BAD_EXIT=1

[ "$#" -ne 3 ] && echo "Please provide <sequenceDB> <outDB> <tmp>" && exit 1;
# check if files exist
[ ! -f "$1.dbtype" ] && echo "$1.dbtype not found!" && exit 1;
[   -f "$2.dbtype" ] && echo "$2.dbtype exists already!" && exit 1;
[ ! -d "$3" ] && echo "tmp directory $3 not found!" && mkdir -p "$3";

INPUT="$1"
TMP_PATH="$3"
SOURCE="$INPUT"
PIDS=""

if [ -n "$CLUSTHASH" ] || [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
    CLUDB="${TMP_PATH}/clu"
else
    CLUDB="$2"
fi
# a finished step is skipped but keeps its number
STEP=0
STEPS=2
if [ -n "$CLUSTHASH" ]; then
    STEPS=$((STEPS + 2))
fi
if [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
    STEPS=$((STEPS + 4))
fi
# 0. clusthash
if [ -n "$CLUSTHASH" ]; then
    STEP=$((STEP + 1))
    if notExistsBeforeRefinement "${TMP_PATH}/input_clusthash_clust.dbtype"; then
        echo "Clustering step $STEP of $STEPS: group nearly identical sequences (clusthashfast)"
        removeFiles "${TMP_PATH}/input_clusthash_clust"*
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" clusthashfast "$INPUT" "${TMP_PATH}/input_clusthash_clust" ${CLUSTHASHFAST_PAR} \
            || fail "clusthashfast died"
    fi

    INPUT="${TMP_PATH}/input_clusthash_clust_redundancy"
fi

# 1. k-mer matching, align2clust removes .partitions before the pref files
STEP=$((STEP + 1))
if { notExists "${TMP_PATH}/pref.dbtype" || notExists "${TMP_PATH}/pref.partitions"; } && notExistsBeforeRefinement "${CLUDB}.dbtype"; then
    echo "Clustering step $STEP of $STEPS: find pairs of sequences that share k-mers (kmermatcher)"
    removeFiles "${TMP_PATH}/pref" "${TMP_PATH}/pref."*
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" kmermatcher "$INPUT" "${TMP_PATH}/pref" ${KMERMATCHER_PAR} \
        || fail "kmermatcher died"
fi

STEP=$((STEP + 1))
if notExistsBeforeRefinement "${CLUDB}.dbtype"; then
    echo "Clustering step $STEP of $STEPS: align the pairs and cluster the sequences (align2clust)"
    removeClusterDb "${CLUDB}"
    removeFiles "${CLUDB}.memory" "${CLUDB}.memory."* "${CLUDB}.merged" "${CLUDB}.merged."* "${CLUDB}_accepted."[0-9]* "${CLUDB}_assigned_keys" "${CLUDB}_assigned_keys."* "${CLUDB}_clustered."[0-9]* "${CLUDB}_clustered_pairsets."[0-9]* "${CLUDB}_clustered_sets."[0-9]* "${CLUDB}_clustered_large."[0-9]* "${CLUDB}_cluster_pairs."* "${CLUDB}_sequence_requests."[0-9]* "${CLUDB}_served_sequences."[0-9]*
    rm -rf "${CLUDB}_sequence_requests" "${CLUDB}_served_sequences"
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" align2clust "$INPUT" "${TMP_PATH}/pref" "$CLUDB" ${ALIGN2CLUST_PAR} \
        || fail "align2clust died"
    if [ -n "$REMOVE_TMP" ]; then
        # shellcheck disable=SC2086
        MMSEQS_NUM_THREADS=1 "$MMSEQS" rmdb "${TMP_PATH}/pref" ${VERBOSITY} &
        PIDS="$PIDS $!"
    fi
fi

# 2. Merge clusthash pre-clusters with alignment clusters
if [ -n "$CLUSTHASH" ]; then
    if [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
        MERGED="${TMP_PATH}/clu_merged"
    else
        MERGED="$2"
    fi
    STEP=$((STEP + 1))
    if notExistsBeforeRefinement "${MERGED}.dbtype"; then
        echo "Clustering step $STEP of $STEPS: add the nearly identical sequences to the clusters (mergeclusters)"
        removeClusterDb "${MERGED}"
        removeFiles "${MERGED}.cluster_key_offsets" "${MERGED}.cluster_key_offsets."* "${MERGED}.merged" "${MERGED}.merged."* "${MERGED}_by_member."* "${MERGED}_by_rep."*
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" mergeclusters "$SOURCE" "$MERGED" \
            "${TMP_PATH}/input_clusthash_clust" "$CLUDB" ${MERGECLU_PAR} \
            || fail "mergeclusters died"
        if [ -n "$REMOVE_TMP" ]; then
            # shellcheck disable=SC2086
            MMSEQS_NUM_THREADS=1 "$MMSEQS" rmdb "${TMP_PATH}/input_clusthash_clust" ${VERBOSITY} &
            PIDS="$PIDS $!"
            # shellcheck disable=SC2086
            MMSEQS_NUM_THREADS=1 "$MMSEQS" rmdb "${TMP_PATH}/input_clusthash_clust_redundancy" ${VERBOSITY} &
            PIDS="$PIDS $!"
            # shellcheck disable=SC2086
            MMSEQS_NUM_THREADS=1 "$MMSEQS" rmdb "$CLUDB" ${VERBOSITY} &
            PIDS="$PIDS $!"
        fi
    fi
    CLUDB="$MERGED"
fi

# 3. Refinement pass: re-cluster representative sequences
if [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
    STEP=$((STEP + 1))
    # a rerun kmermatcher needs the seed member requests again
    if notExists "${TMP_PATH}/input_rep.dbtype" "${TMP_PATH}/pref_rep.dbtype" "${TMP_PATH}/clu_rep.dbtype" \
        || notExists "${TMP_PATH}/pref_rep.partitions" "${TMP_PATH}/clu_rep.dbtype" "${TMP_PATH}/input_rep.seed_member_requests.0"; then
        echo "Clustering step $STEP of $STEPS: collect the representative sequences for a second round (createsubdb)"
        removeFiles "${TMP_PATH}/input_rep" "${TMP_PATH}/input_rep."* "${TMP_PATH}/input_rep_h"*
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" createsubdb "$CLUDB" "$SOURCE" "${TMP_PATH}/input_rep" ${CREATESUBDB_PAR} \
            || fail "createsubdb (representatives) died"
    fi

    STEP=$((STEP + 1))
    if { notExists "${TMP_PATH}/pref_rep.dbtype" || notExists "${TMP_PATH}/pref_rep.partitions"; } && notExists "${TMP_PATH}/clu_rep.dbtype"; then
        echo "Clustering step $STEP of $STEPS: find pairs of representatives that share k-mers (kmermatcher)"
        removeFiles "${TMP_PATH}/pref_rep" "${TMP_PATH}/pref_rep."*
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" kmermatcher "${TMP_PATH}/input_rep" "${TMP_PATH}/pref_rep" ${KMERMATCHER_PAR2} \
            || fail "kmermatcher (representatives) died"
        if [ -n "$REMOVE_TMP" ]; then
            removeFiles "${TMP_PATH}/input_rep.seed_member_requests."*
        fi
    fi

    STEP=$((STEP + 1))
    if notExists "${TMP_PATH}/clu_rep.dbtype"; then
        echo "Clustering step $STEP of $STEPS: align these pairs and cluster the representatives (align2clust)"
        removeFiles "${TMP_PATH}/clu_rep" "${TMP_PATH}/clu_rep."* "${TMP_PATH}/clu_rep_accepted"* "${TMP_PATH}/clu_rep_assigned_keys"* "${TMP_PATH}/clu_rep_cluster"* "${TMP_PATH}/clu_rep_member"*
        rm -rf "${TMP_PATH}/clu_rep_sequence_requests"* "${TMP_PATH}/clu_rep_served_sequences"*
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" align2clust "${TMP_PATH}/input_rep" "${TMP_PATH}/pref_rep" "${TMP_PATH}/clu_rep" \
            ${ALIGN2CLUST_PAR} \
            --filter-cludb-file "$CLUDB" \
            --filter-seqdb-file "$SOURCE" \
            || fail "align2clust (representatives) died"
        if [ -n "$REMOVE_TMP" ]; then
            # shellcheck disable=SC2086
            MMSEQS_NUM_THREADS=1 "$MMSEQS" rmdb "${TMP_PATH}/pref_rep" ${VERBOSITY} &
            PIDS="$PIDS $!"
            # shellcheck disable=SC2086
            MMSEQS_NUM_THREADS=1 "$MMSEQS" rmdb "${TMP_PATH}/input_rep" ${VERBOSITY} &
            PIDS="$PIDS $!"
        fi
    fi

    STEP=$((STEP + 1))
    if notExists "$2.dbtype"; then
        echo "Clustering step $STEP of $STEPS: merge both rounds into the final clusters (mergeclusters)"
        removeClusterDb "$2"
        removeFiles "${2}.cluster_key_offsets" "${2}.cluster_key_offsets."* "${2}.merged" "${2}.merged."* "${2}_by_member."* "${2}_by_rep."*
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" mergeclusters "$SOURCE" "$2" \
            "$CLUDB" "${TMP_PATH}/clu_rep" ${MERGECLU_PAR} \
            || fail "mergeclusters died"
        if [ -n "$REMOVE_TMP" ]; then
            # shellcheck disable=SC2086
            MMSEQS_NUM_THREADS=1 "$MMSEQS" rmdb "${TMP_PATH}/clu_rep" ${VERBOSITY} &
            PIDS="$PIDS $!"
            # shellcheck disable=SC2086
            MMSEQS_NUM_THREADS=1 "$MMSEQS" rmdb "$CLUDB" ${VERBOSITY} &
            PIDS="$PIDS $!"
        fi
    fi
fi

# wait for the background rmdb
for PID in $PIDS; do
    wait "$PID"
done

if [ -n "$REMOVE_TMP" ]; then
    rm -f "${TMP_PATH}/lin8clust.sh"
fi
