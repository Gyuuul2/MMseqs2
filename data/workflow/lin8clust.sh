#!/bin/sh -e
# One shot linear clustering. Run it once on every machine with the same NODES and its own NODE.
# Guards are on the per machine markers, not on the output every machine publishes into.
fail() { echo "$1" >&2; exit 1; }
notExists() { [ ! -f "$1" ]; }

# waits for every machine's marker, because a machine that has not started looks like one with nothing
waitForAll() {
    _path="$1"; _n="$2"; _waited=0
    while true; do
        _have=0; _i=0
        while [ "$_i" -lt "$_n" ]; do
            [ -f "$_path.$_i.done" ] && _have=$((_have + 1))
            _i=$((_i + 1))
        done
        [ "$_have" -eq "$_n" ] && return 0
        [ "$_waited" -ge "$WAIT_LIMIT" ] && fail "waited ${WAIT_LIMIT}s for $_path, $_have of $_n machines finished"
        sleep 1
        _waited=$((_waited + 1))
    done
}

# arguments when called as a command, IN/OUT/TMP in the environment when called on its own
if [ "$#" -ge 3 ]; then
    OUT="$(eval echo "\${$(($# - 1))}")"
    TMP="$(eval echo "\${$#}")"
    IN=""
    _i=1
    while [ "$_i" -le "$(($# - 2))" ]; do
        IN="$IN $(eval echo "\${$_i}")"
        _i=$((_i + 1))
    done
fi
[ -z "$IN" ] && fail "no input sequence file given"
[ -z "$OUT" ] && fail "no output cluster database given"
[ -z "$TMP" ] && fail "no tmp directory given"
[ ! -d "$TMP" ] && mkdir -p "$TMP"
NODES="${NODES:-1}"
NODE="${NODE:-0}"
REP_RANK_BLOCKS="${REP_RANK_BLOCKS:-1024}"
THREADS="${THREADS:-1}"
SEQID="${SEQID:-0.9}"
# the redundancy pass may fold at any threshold at or above the target, and 0.9 is the loosest
# the original allows; the workflow sets this, so this is for a script run on its own
HASHSEQID="${HASHSEQID:-$(awk -v s="$SEQID" 'BEGIN{print (s>0.9)?s:0.9}')}"
COV="${COV:-0.8}"
COVMODE="${COVMODE:-0}"
REPSEQ="${REPSEQ:-0}"
WAIT_LIMIT="${WAIT_LIMIT:-86400}"

mkdir -p "$TMP/kmer" "$TMP/pairs" "$TMP/pref" "$TMP/aln" "$TMP/clu_accepted"

# two rounds: every machine writes a histogram, then places sequences, so this is a retry loop
_waited=0
while notExists "$TMP/db.$NODE.done" || notExists "$TMP/db.dbtype"; do
    # shellcheck disable=SC2086
    "$MMSEQS" lin8-createdb $IN "$TMP/db" --node-count "$NODES" --node-id "$NODE" \
        --threads "$THREADS" ${CREATEDB_PAR}
    if notExists "$TMP/db.$NODE.done" || notExists "$TMP/db.dbtype"; then
        [ "$_waited" -ge "$WAIT_LIMIT" ] && fail "waited ${WAIT_LIMIT}s for the database"
        sleep 1
        _waited=$((_waited + 1))
    fi
done

# the same two rounds, ending in the bitmap
if [ -n "$CLUSTHASH" ]; then
    _waited=0
    while notExists "$TMP/db.clusthash_kept"; do
        # shellcheck disable=SC2086
        "$MMSEQS" lin8-clusthash "$TMP/db" "$TMP/hash" --node-count "$NODES" --node-id "$NODE" \
            --threads "$THREADS" --min-seq-id "$HASHSEQID" ${HASH_PAR}
        if notExists "$TMP/db.clusthash_kept"; then
            [ "$_waited" -ge "$WAIT_LIMIT" ] && fail "waited ${WAIT_LIMIT}s for the redundancy pass"
            sleep 1
            _waited=$((_waited + 1))
        fi
    done

    [ "$NODES" -gt 1 ] && waitForAll "$TMP/hash" "$NODES"
fi

# shellcheck disable=SC2086
notExists "$TMP/kmer/out.$NODE.done" && \
    "$MMSEQS" lin8-extractkmers "$TMP/db" "$TMP/kmer/out" --node-count "$NODES" --node-id "$NODE" \
        --threads "$THREADS" --min-seq-id "$SEQID" ${EXTRACT_PAR}

[ "$NODES" -gt 1 ] && waitForAll "$TMP/kmer/out" "$NODES"

# shellcheck disable=SC2086
notExists "$TMP/pairs/pairs.$NODE.done" && \
    "$MMSEQS" lin8-assignedpairs "$TMP/db" "$TMP/kmer/out" "$TMP/pairs/pairs" \
        --node-count "$NODES" --node-id "$NODE" --threads "$THREADS" --rep-rank-blocks "$REP_RANK_BLOCKS" \
        -c "$COV" --cov-mode "$COVMODE" ${GROUP_PAR}

[ "$NODES" -gt 1 ] && waitForAll "$TMP/pairs/pairs" "$NODES"

# shellcheck disable=SC2086
notExists "$TMP/pref/pref.$NODE.done" && \
    "$MMSEQS" lin8-pref "$TMP/pairs/pairs" "$TMP/db" "$TMP/pref/pref" \
        --node-count "$NODES" --node-id "$NODE" --threads "$THREADS" ${PREF_PAR}

[ "$NODES" -gt 1 ] && waitForAll "$TMP/pref/pref" "$NODES"

# The wave: block k only needs what was decided up to k-1, and deciding is one thread because a
# greedy assignment is one order. One machine fuses the two halves; many machines barrier here.
if [ "$NODES" -eq 1 ]; then
    # shellcheck disable=SC2086
    notExists "$TMP/clu_accepted/clu_accepted.0.$((REP_RANK_BLOCKS - 1))" && \
        "$MMSEQS" lin8-align2clust "$TMP/db" "$TMP/pref/pref" "$TMP/aln/aln" \
            "$TMP/clu_accepted/clu_accepted" --node-count 1 --node-id 0 \
            --threads "$THREADS" --min-seq-id "$SEQID" -c "$COV" --cov-mode "$COVMODE" \
            ${ALIGN_PAR}
    R="$REP_RANK_BLOCKS"
else
    R=0
fi
while [ "$R" -lt "$REP_RANK_BLOCKS" ]; do
    if notExists "$TMP/clu_accepted/clu_accepted.0.$R"; then
        # shellcheck disable=SC2086
        notExists "$TMP/aln/aln.$R.$NODE.done" && \
            "$MMSEQS" lin8-align2clust "$TMP/db" "$TMP/pref/pref" "$TMP/aln/aln" \
                "$TMP/clu_accepted/clu_accepted" \
                --node-count "$NODES" --node-id "$NODE" --rep-rank-block "$R" \
                --threads "$THREADS" --min-seq-id "$SEQID" -c "$COV" --cov-mode "$COVMODE" \
                ${ALIGN_PAR}
        if [ "$NODE" -eq 0 ]; then
            [ "$NODES" -gt 1 ] && waitForAll "$TMP/aln/aln.$R" "$NODES"
            # shellcheck disable=SC2086
            "$MMSEQS" lin8-align2clustmulti "$TMP/aln/aln" "$TMP/pref/pref" \
                "$TMP/clu_accepted/clu_accepted" --rep-rank-block "$R" ${ASSIGN_PAR}
        else
            # the next block needs this one's bitmap
            _waited=0
            while notExists "$TMP/clu_accepted/clu_accepted.0.$R"; do
                [ "$_waited" -ge "$WAIT_LIMIT" ] && fail "waited ${WAIT_LIMIT}s for repRankBlock $R to be decided"
                sleep 1
                _waited=$((_waited + 1))
            done
        fi
    fi
    R=$((R + 1))
done

# the redundant sequences go back in on the pair stream, and the clustering database is made from that
if [ "$NODE" -eq 0 ]; then
    CLUDB="$TMP/clu_accepted/clu_accepted"
    if [ -n "$CLUSTHASH" ]; then
        mkdir -p "$TMP/clu_accepted_plus_redundant"
        # shellcheck disable=SC2086
        notExists "$TMP/clu_accepted_plus_redundant/clu_accepted_plus_redundant" && \
            "$MMSEQS" lin8-mergehashredundancy "$CLUDB" "$TMP/hash" \
                "$TMP/clu_accepted_plus_redundant/clu_accepted_plus_redundant" ${EXPAND_PAR}
        CLUDB="$TMP/clu_accepted_plus_redundant/clu_accepted_plus_redundant"
    fi
    # shellcheck disable=SC2086
    notExists "$OUT.dbtype" && \
        "$MMSEQS" lin8-createclusterdb "$CLUDB" "$OUT" ${CLUSTERDB_PAR}

    # naming is its own pass, and it runs the way linclust names its answer: always
    # shellcheck disable=SC2086
    notExists "$OUT.tsv" && \
        "$MMSEQS" lin8-createtsv "$TMP/db" "$OUT" "$OUT.tsv" ${TSV_PAR}

    # shellcheck disable=SC2086
    [ "$REPSEQ" -eq 1 ] && notExists "$OUT.rep.fasta" && \
        "$MMSEQS" lin8-createrepseqfasta "$TMP/db" "$OUT" "$OUT.rep.fasta" ${REPSEQ_PAR}
fi

# what is left here is what a rerun would have needed; every machine clears its own
if [ -n "$REMOVE_TMP" ]; then
    rm -f "$TMP/clu_accepted/clu_accepted.align_assigned_$NODE" "$TMP/kmer/out.$NODE."* "$TMP/pairs/pairs.$NODE."* \
          "$TMP/pref/pref.$NODE."* "$TMP/aln/aln."*".$NODE"*
    if [ "$NODE" -eq 0 ]; then
        rm -rf "$TMP/kmer" "$TMP/pairs" "$TMP/pref" "$TMP/aln" "$TMP/clu_accepted" \
               "$TMP/clu_accepted_plus_redundant"
        # the database stays until the clustering is named, because a cluster is named by rank
        rm -f "$TMP/hash" "$TMP/hash."* "$TMP/lin8clust.sh"
        [ "$REPSEQ" -eq 0 ] && rm -f "$TMP/db" "$TMP/db".[0-9]* "$TMP/db.hist."* "$TMP/db.runs"* \
            "$TMP/db.files" "$TMP/db.dbtype" "$TMP/db_h."*
    fi
fi
exit 0
