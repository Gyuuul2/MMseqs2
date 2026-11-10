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
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" createdb "$@" "${TMP_PATH}/input" ${CREATEDB_PAR} \
        || fail "createdb died"
fi

if notExists "${TMP_PATH}/clu.dbtype"; then
    # shellcheck disable=SC2086
    "$MMSEQS" lin8clust "${TMP_PATH}/input" "${TMP_PATH}/clu" "${TMP_PATH}/clu_tmp" ${CLUSTER_PAR} \
        || fail "lin8clust died"
fi

# a killed step can leave some node files, mark finished steps
if notExists "${TMP_PATH}/createtsv.done"; then
    rm -f "${RESULTS}_cluster.tsv".*
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" createtsv "${TMP_PATH}/input" "${TMP_PATH}/input" "${TMP_PATH}/clu" "${TMP_PATH}/cluster.tsv" ${CREATETSV_PAR} \
        || fail "createtsv died"
    touch "${TMP_PATH}/createtsv.done"
fi

if notExists "${TMP_PATH}/result2flat.done"; then
    rm -f "${RESULTS}_rep_seq.fasta".*
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" result2flat "${TMP_PATH}/input" "${TMP_PATH}/input" "${TMP_PATH}/clu" "${TMP_PATH}/rep_seq.fasta" ${RESULT2FLAT_PAR} \
        || fail "result2flat died"
    touch "${TMP_PATH}/result2flat.done"
fi

for FILE in "${TMP_PATH}/cluster.tsv".* "${TMP_PATH}/rep_seq.fasta".*; do
    [ -e "$FILE" ] || continue
    NAME="${FILE##*/}"
    mv "$FILE" "${RESULTS}_${NAME}"
done
rm -f "${TMP_PATH}/createtsv.done" "${TMP_PATH}/result2flat.done"

if [ -n "${REMOVE_TMP}" ]; then
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${TMP_PATH}/input" ${VERBOSITY_PAR}
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${TMP_PATH}/input_h" ${VERBOSITY_PAR}
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${TMP_PATH}/clu" ${VERBOSITY_PAR}
    rm -rf "${TMP_PATH}/clu_tmp"
    rm -f "${TMP_PATH}/easylin8clust.sh"
fi
