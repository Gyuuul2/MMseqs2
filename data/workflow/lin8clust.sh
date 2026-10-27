#!/bin/sh -e
fail() {
    echo "Error: $1"
    exit 1
}

# true if none of the files exists
notExists() {
    for FILE in "$@"; do
        [ -f "$FILE" ] && return 1
    done
    return 0
}

[ "$#" -ne 3 ] && echo "Please provide <sequenceDB> <outDB> <tmp>" && exit 1;
# check if files exist
[ ! -f "$1.dbtype" ] && echo "$1.dbtype not found!" && exit 1;
[   -f "$2.dbtype" ] && echo "$2.dbtype exists already!" && exit 1;
[ ! -d "$3" ] && echo "tmp directory $3 not found!" && mkdir -p "$3";

INPUT="$1"
TMP_PATH="$3"
SOURCE="$INPUT"

if [ -n "$CLUSTHASH" ] || [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
    CLUDB="${TMP_PATH}/clu"
else
    CLUDB="$2"
fi
ITER2_PRODUCTS=""
if [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
    ITER2_PRODUCTS="${TMP_PATH}/clu_merged.dbtype ${TMP_PATH}/input_rep.dbtype ${TMP_PATH}/pref_rep.dbtype ${TMP_PATH}/clu_rep.dbtype"
fi

# 0. clusthash
if [ -n "$CLUSTHASH" ]; then
    # shellcheck disable=SC2086
    if notExists "${TMP_PATH}/input_clusthash_clust.dbtype" ${ITER2_PRODUCTS}; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" clusthashfast "$INPUT" "${TMP_PATH}/input_clusthash_clust" ${CLUSTHASHFAST_PAR} \
            || fail "clusthashfast died"
    fi

    INPUT="${TMP_PATH}/input_clusthash_clust_redundancy"
fi

# 1. k-mer matching
# shellcheck disable=SC2086
if notExists "${TMP_PATH}/pref.dbtype" "${CLUDB}.dbtype" ${ITER2_PRODUCTS}; then
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" kmermatcher "$INPUT" "${TMP_PATH}/pref" ${KMERMATCHER_PAR} \
        || fail "kmermatcher died"
fi

# shellcheck disable=SC2086
if notExists "${CLUDB}.dbtype" ${ITER2_PRODUCTS}; then
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" align2clust "$INPUT" "${TMP_PATH}/pref" "$CLUDB" ${ALIGN2CLUST_PAR} \
        || fail "align2clust died"
    if [ -n "$REMOVE_TMP" ]; then
        # shellcheck disable=SC2086
        "$MMSEQS" rmdb "${TMP_PATH}/pref" ${VERBOSITY}
        if [ -n "$CLUSTHASH" ]; then
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/input_clusthash_clust_redundancy" ${VERBOSITY}
        fi
    fi
fi

# 2. Merge clusthash pre-clusters with alignment clusters
if [ -n "$CLUSTHASH" ]; then
    if [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
        MERGED="${TMP_PATH}/clu_merged"
    else
        MERGED="$2"
    fi
    # shellcheck disable=SC2086
    if notExists "${MERGED}.dbtype" ${ITER2_PRODUCTS}; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" mergeclusters "$SOURCE" "$MERGED" \
            "${TMP_PATH}/input_clusthash_clust" "$CLUDB" ${MERGECLU_PAR} \
            || fail "mergeclusters died"
        if [ -n "$REMOVE_TMP" ]; then
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/input_clusthash_clust" ${VERBOSITY}
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "$CLUDB" ${VERBOSITY}
        fi
    fi
    CLUDB="$MERGED"
fi

# 3. Refinement pass: re-cluster representative sequences
if [ "$LINCLUST_ITERATIONS" -ge 2 ]; then
    if notExists "${TMP_PATH}/input_rep.dbtype" "${TMP_PATH}/pref_rep.dbtype" "${TMP_PATH}/clu_rep.dbtype"; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" createsubdb "$CLUDB" "$SOURCE" "${TMP_PATH}/input_rep" ${VERBOSITY} --subdb-mode 1 \
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
            "$MMSEQS" rmdb "${TMP_PATH}/pref_rep" ${VERBOSITY}
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/input_rep" ${VERBOSITY}
        fi
    fi

    if notExists "$2.dbtype"; then
        # shellcheck disable=SC2086
        $RUNNER "$MMSEQS" mergeclusters "$SOURCE" "$2" \
            "$CLUDB" "${TMP_PATH}/clu_rep" ${MERGECLU_PAR} \
            || fail "mergeclusters died"
        if [ -n "$REMOVE_TMP" ]; then
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "${TMP_PATH}/clu_rep" ${VERBOSITY}
            # shellcheck disable=SC2086
            "$MMSEQS" rmdb "$CLUDB" ${VERBOSITY}
        fi
    fi
fi

if [ -n "$REMOVE_TMP" ]; then
    rm -f "${TMP_PATH}/lin8clust.sh"
fi
