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
LOCAL_DB_ACTIVE=""

cleanupLocalDb() {
    [ -n "$LOCAL_DB_ACTIVE" ] || return 0
    LOCAL_DB_ACTIVE=""
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" rmdb "$LOCAL_DB_NAME" ${LOCAL_DB_REMOVE_PAR}
}

if [ -n "$LOCAL_DB_NAME" ]; then
    LOCAL_DB_ACTIVE=1
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" cpdb "$SOURCE" "$LOCAL_DB_NAME" ${LOCAL_DB_COPY_PAR} \
        || fail "copying the sequence database to --local-db-dir died"
fi

if [ -n "$CLUSTHASH" ] || [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
    CLUDB="${TMP_PATH}/clu"
else
    CLUDB="$2"
fi
# 0. clusthash
if [ -n "$CLUSTHASH" ]; then
    if notExistsBeforeRefinement "${TMP_PATH}/input_clusthash_clust.dbtype"; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" clusthashfast "$INPUT" "${TMP_PATH}/input_clusthash_clust" ${CLUSTHASHFAST_PAR} \
            || fail "clusthashfast died"
    fi

    INPUT="${TMP_PATH}/input_clusthash_clust_redundancy"
fi

# 1. k-mer matching
if notExistsBeforeRefinement "${TMP_PATH}/pref.dbtype" "${CLUDB}.dbtype"; then
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" kmermatcher "$INPUT" "${TMP_PATH}/pref" ${KMERMATCHER_PAR} \
        || fail "kmermatcher died"
fi

if notExistsBeforeRefinement "${CLUDB}.dbtype"; then
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" align2clust "$INPUT" "${TMP_PATH}/pref" "$CLUDB" ${ALIGN2CLUST_PAR} \
        || fail "align2clust died"
    if [ -n "$REMOVE_TMP" ]; then
        # shellcheck disable=SC2086
        "$MMSEQS" rmdb "${TMP_PATH}/pref" ${VERBOSITY} &
        PIDS="$PIDS $!"
    fi
fi

# 2. Merge clusthash pre-clusters with alignment clusters
MERGECLU_PAR1="$MERGECLU_PAR"
if [ "$LINCLUST_ITERATIONS" -lt 2 ]; then
    cleanupLocalDb || fail "removing the local sequence database died"
else
    # The sequence copy stays on tmpfs until the refinement alignment ends.
    MERGECLU_PAR1="$MERGECLU_LOCAL_PAR"
fi
if [ -n "$CLUSTHASH" ]; then
    if [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
        MERGED="${TMP_PATH}/clu_merged"
    else
        MERGED="$2"
    fi
    if notExistsBeforeRefinement "${MERGED}.dbtype"; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" mergeclusters "$SOURCE" "$MERGED" \
            "${TMP_PATH}/input_clusthash_clust" "$CLUDB" ${MERGECLU_PAR1} \
            || fail "mergeclusters died"
        if [ -n "$REMOVE_TMP" ]; then
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/input_clusthash_clust" ${VERBOSITY} &
            PIDS="$PIDS $!"
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/input_clusthash_clust_redundancy" ${VERBOSITY} &
            PIDS="$PIDS $!"
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "$CLUDB" ${VERBOSITY} &
            PIDS="$PIDS $!"
        fi
    fi
    CLUDB="$MERGED"
fi

# 3. Refinement pass: re-cluster representative sequences
if [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
    if notExists "${TMP_PATH}/input_rep.dbtype" "${TMP_PATH}/pref_rep.dbtype" "${TMP_PATH}/clu_rep.dbtype"; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" createsubdb "$CLUDB" "$SOURCE" "${TMP_PATH}/input_rep" ${CREATESUBDB_PAR} \
            || fail "createsubdb (representatives) died"
    fi

    if notExists "${TMP_PATH}/pref_rep.dbtype" "${TMP_PATH}/clu_rep.dbtype"; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" kmermatcher "${TMP_PATH}/input_rep" "${TMP_PATH}/pref_rep" ${KMERMATCHER_PAR2} \
            || fail "kmermatcher (representatives) died"
    fi

    if notExists "${TMP_PATH}/clu_rep.dbtype"; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" align2clust "${TMP_PATH}/input_rep" "${TMP_PATH}/pref_rep" "${TMP_PATH}/clu_rep" \
            ${ALIGN2CLUST_PAR} \
            --filter-cludb-file "$CLUDB" \
            --filter-seqdb-file "$SOURCE" \
            || fail "align2clust (representatives) died"
        if [ -n "$REMOVE_TMP" ]; then
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/pref_rep" ${VERBOSITY} &
            PIDS="$PIDS $!"
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/input_rep" ${VERBOSITY} &
            PIDS="$PIDS $!"
        fi
    fi

    cleanupLocalDb || fail "removing the local sequence database died"

    if notExists "$2.dbtype"; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" mergeclusters "$SOURCE" "$2" \
            "$CLUDB" "${TMP_PATH}/clu_rep" ${MERGECLU_PAR} \
            || fail "mergeclusters died"
        if [ -n "$REMOVE_TMP" ]; then
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/clu_rep" ${VERBOSITY} &
            PIDS="$PIDS $!"
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "$CLUDB" ${VERBOSITY} &
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
