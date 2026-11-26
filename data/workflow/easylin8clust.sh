#!/bin/sh -e
fail() {
    echo "Error: $1"
    exit 1
}

notExists() {
   [ ! -f "$1" ]
}

# end all tasks of a srun step if one fails
export SLURM_KILL_BAD_EXIT=1

if notExists "${TMP_PATH}/input.dbtype"; then
    echo "Step 1 of 4: convert the input sequences into a database (createdb)"
    rm -f "${TMP_PATH}/input" "${TMP_PATH}/input."* "${TMP_PATH}/input_h"* "${TMP_PATH}/input_sent_"*
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" createdb "$@" "${TMP_PATH}/input" ${CREATEDB_PAR} \
        || fail "createdb died"
fi

if notExists "${TMP_PATH}/clu.dbtype"; then
    echo "Step 2 of 4: cluster the sequences (lin8clust)"
    # shellcheck disable=SC2086
    "$MMSEQS" lin8clust "${TMP_PATH}/input" "${TMP_PATH}/clu" "${TMP_PATH}/clu_tmp" ${CLUSTER_PAR} \
        || fail "lin8clust died"
fi

# a killed step can leave some node files, mark finished steps
if notExists "${TMP_PATH}/createtsv.done"; then
    echo "Step 3 of 4: write the cluster table (createtsv)"
    rm -f "${RESULTS}_cluster.tsv".* "${TMP_PATH}/clu_tsv"*
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" createtsv "${TMP_PATH}/input" "${TMP_PATH}/input" "${TMP_PATH}/clu" "${RESULTS}_cluster.tsv" ${CREATETSV_PAR} \
        || fail "createtsv died"
    touch "${TMP_PATH}/createtsv.done"
fi

if notExists "${TMP_PATH}/result2flat.done"; then
    echo "Step 4 of 4: write the representative sequences (result2flat)"
    rm -f "${RESULTS}_rep_seq.fasta".* "${TMP_PATH}/clu_fasta"*
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" result2flat "${TMP_PATH}/input" "${TMP_PATH}/input" "${TMP_PATH}/clu" "${RESULTS}_rep_seq.fasta" ${RESULT2FLAT_PAR} \
        || fail "result2flat died"
    touch "${TMP_PATH}/result2flat.done"
fi

rm -f "${TMP_PATH}/createtsv.done" "${TMP_PATH}/result2flat.done"

if [ -n "${REMOVE_TMP}" ]; then
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${TMP_PATH}/input" ${VERBOSITY_PAR}
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${TMP_PATH}/input_h" ${VERBOSITY_PAR}
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${TMP_PATH}/clu" ${VERBOSITY_PAR}
    # keep the .nodes.log files of the clustering steps
    find "${TMP_PATH}/clu_tmp" ! -type d ! -name '*.nodes.log' -exec rm -f {} +
    rm -f "${TMP_PATH}/easylin8clust.sh"
fi
