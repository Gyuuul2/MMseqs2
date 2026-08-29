#!/bin/sh -e
# One shot linear clustering. Every pass is a command of its own, so a pass can be rerun on its own,
# and the barrier between the waves is held here rather than inside a module, as elsewhere in mmseqs.
#
# Run this once on every machine with the same NODES and its own NODE. A pass leaves a marker of its
# own per machine, and the guards are on those markers rather than on the output: the output of a
# pass is one path every machine publishes, so guarding on it would make the machine that finished
# second skip its own work.
fail() { echo "$1" >&2; exit 1; }
notExists() { [ ! -f "$1" ]; }

# Waits for every machine to leave its marker. Every pass reads what all of them wrote, and a machine
# that has not started is indistinguishable from one that wrote nothing, so the check inside the
# modules refuses rather than waits and the waiting is here.
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

# Called either as a command, which hands the inputs in as arguments, or on its own with IN, OUT and
# TMP in the environment. Every machine gets the same arguments and its own NODE.
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
COV="${COV:-0.8}"
COVMODE="${COVMODE:-0}"
TSV="${TSV:-0}"
REPSEQ="${REPSEQ:-0}"
CLUSTHASH="${CLUSTHASH:-1}"
WAIT_LIMIT="${WAIT_LIMIT:-86400}"

mkdir -p "$TMP/kmer" "$TMP/pairs" "$TMP/pref" "$TMP/aligned" "$TMP/decided"

# The database is built in two rounds: a machine writes its own histogram, and once every machine
# has, the next run of it places the sequences and whichever machine finds every part written
# publishes the database. So this is a retry loop, which is what the pass says to do.
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

# the same two rounds: a machine writes its shard, and once every machine has, the next run of it
# merges them and publishes the bitmap
_waited=0
while notExists "$TMP/hash.valid"; do
    notExists "$TMP/hash.$NODE.done" || [ "$NODES" -eq 1 ] && :
    # shellcheck disable=SC2086
    "$MMSEQS" lin8-clusthash "$TMP/db" "$TMP/hash" --node-count "$NODES" --node-id "$NODE" \
        --clust-hash "$CLUSTHASH" \
        --threads "$THREADS" ${HASH_PAR}
    if notExists "$TMP/hash.valid"; then
        [ "$_waited" -ge "$WAIT_LIMIT" ] && fail "waited ${WAIT_LIMIT}s for the redundancy pass"
        sleep 1
        _waited=$((_waited + 1))
    fi
done

[ "$NODES" -gt 1 ] && waitForAll "$TMP/hash" "$NODES"

VALID="--valid $TMP/hash.valid"

# shellcheck disable=SC2086
notExists "$TMP/kmer/out.$NODE.done" && \
    "$MMSEQS" lin8-kmers "$TMP/db" "$TMP/kmer/out" --node-count "$NODES" --node-id "$NODE" \
        --threads "$THREADS" --min-seq-id "$SEQID" $VALID ${EXTRACT_PAR}

[ "$NODES" -gt 1 ] && waitForAll "$TMP/kmer/out" "$NODES"

# shellcheck disable=SC2086
notExists "$TMP/pairs/pairs.$NODE.done" && \
    "$MMSEQS" lin8-pairs "$TMP/db" "$TMP/kmer/out" "$TMP/pairs/pairs" \
        --node-count "$NODES" --node-id "$NODE" --threads "$THREADS" --rep-rank-blocks "$REP_RANK_BLOCKS" \
        -c "$COV" --cov-mode "$COVMODE" $VALID ${GROUP_PAR}

[ "$NODES" -gt 1 ] && waitForAll "$TMP/pairs/pairs" "$NODES"

# shellcheck disable=SC2086
notExists "$TMP/pref/pref.$NODE.done" && \
    "$MMSEQS" lin8-pref "$TMP/pairs/pairs" "$TMP/db" "$TMP/pref/pref" \
        --node-count "$NODES" --node-id "$NODE" --threads "$THREADS" $VALID ${FOLD_PAR}

[ "$NODES" -gt 1 ] && waitForAll "$TMP/pref/pref" "$NODES"

# The wave. A representative can only be taken by one of a lower rank and the blocks ascend with that
# rank, so aligning block k only needs what was decided up to the end of block k-1. Deciding is one
# thread, because a greedy assignment is one order over the whole database.
#
# One machine is its own whole share of every repRankBlock, so there is nothing to wait for: it walks the
# repRankBlocks in one process, deciding each group as the aligner hands it over. The bitmap is then a
# variable rather than a file, and the pairs the deciding pass would have read never reach the disk.
#
# Many machines cannot do that: block k is not decided until every one of them has aligned its share
# of it, so they keep a process a repRankBlock and this loop holds the barrier between the two halves. The
# guard is per repRankBlock and guards both halves together: the bitmap the deciding pass hands on is only
# consistent with the repRankBlocks it has already decided, so redoing a decided repRankBlock would find every
# representative in it already taken and write an empty repRankBlock over a full one.
if [ "$NODES" -eq 1 ]; then
    # shellcheck disable=SC2086
    notExists "$TMP/decided/decided.0.$((REP_RANK_BLOCKS - 1))" && \
        "$MMSEQS" lin8-align "$TMP/db" "$TMP/pref/pref" "$TMP/aligned/aligned" \
            --node-count 1 --node-id 0 --taken "$TMP/taken.0" \
            --decided "$TMP/decided/decided" \
            --threads "$THREADS" --min-seq-id "$SEQID" -c "$COV" --cov-mode "$COVMODE" \
            $VALID ${ALIGN_PAR}
    R="$REP_RANK_BLOCKS"
else
    R=0
fi
while [ "$R" -lt "$REP_RANK_BLOCKS" ]; do
    if notExists "$TMP/decided/decided.0.$R"; then
        # shellcheck disable=SC2086
        notExists "$TMP/aligned/aligned.$R.$NODE.done" && \
            "$MMSEQS" lin8-align "$TMP/db" "$TMP/pref/pref" "$TMP/aligned/aligned" \
                --node-count "$NODES" --node-id "$NODE" --repRankBlock "$R" --taken "$TMP/taken.$NODE" \
                --decided "$TMP/decided/decided" \
                        --threads "$THREADS" --min-seq-id "$SEQID" -c "$COV" --cov-mode "$COVMODE" \
                $VALID ${ALIGN_PAR}
        if [ "$NODE" -eq 0 ]; then
            [ "$NODES" -gt 1 ] && waitForAll "$TMP/aligned/aligned.$R" "$NODES"
            # shellcheck disable=SC2086
            "$MMSEQS" lin8-cluster "$TMP/aligned/aligned" "$TMP/decided/decided" \
                --repRankBlock "$R" --taken "$TMP/taken.assign" \
                --pref "$TMP/pref/pref" ${ASSIGN_PAR}
        else
            # the next repRankBlock needs the bitmap this repRankBlock produced, so wait for it
            _waited=0
            while notExists "$TMP/decided/decided.0.$R"; do
                [ "$_waited" -ge "$WAIT_LIMIT" ] && fail "waited ${WAIT_LIMIT}s for repRankBlock $R to be decided"
                sleep 1
                _waited=$((_waited + 1))
            done
        fi
    fi
    R=$((R + 1))
done

# The sequences the redundancy pass folded away go back in on the pair stream, which is the form
# that scales, and the clustering database is made from that. A database has an index entry a
# representative, which at a trillion sequences is tens of terabytes of index that nothing here
# looks anything up in, so at that size the pair stream is the output and this last step is skipped.
if [ "$NODE" -eq 0 ]; then
    mkdir -p "$TMP/expanded"
    # shellcheck disable=SC2086
    notExists "$TMP/expanded/expanded" && \
        "$MMSEQS" lin8-expand "$TMP/decided/decided" "$TMP/hash" "$TMP/expanded/expanded" \
            ${EXPAND_PAR}
    # shellcheck disable=SC2086
    notExists "$OUT.dbtype" && \
        "$MMSEQS" lin8-merge "$TMP/expanded/expanded" "$OUT" ${CLUSTERDB_PAR}

    # createtsv cannot read a run table database, which has no per entry index to open, so the naming
    # is its own pass. At a trillion sequences the names are tens of terabytes nothing looks up, so it
    # is asked for rather than assumed.
    # shellcheck disable=SC2086
    [ "$TSV" -eq 1 ] && notExists "$OUT.tsv" && \
        "$MMSEQS" lin8-createtsv "$TMP/db" "$OUT" "$OUT.tsv" ${TSV_PAR}

    # shellcheck disable=SC2086
    [ "$REPSEQ" -eq 1 ] && notExists "$OUT.rep.fasta" && \
        "$MMSEQS" lin8-repseq "$TMP/db" "$OUT" "$OUT.rep.fasta" ${REPSEQ_PAR}
fi

# The passes drop what they have finished reading as they go, so what is left here is what a rerun
# would have needed. Every machine clears its own, and the last of them takes the directories.
if [ -n "$REMOVE_TMP" ]; then
    rm -f "$TMP/taken.$NODE" "$TMP/kmer/out.$NODE."* "$TMP/pairs/pairs.$NODE."* \
          "$TMP/pref/pref.$NODE."* "$TMP/aligned/aligned."*".$NODE"*
    if [ "$NODE" -eq 0 ]; then
        rm -f "$TMP/taken.assign"
        rm -rf "$TMP/kmer" "$TMP/pairs" "$TMP/pref" "$TMP/aligned" "$TMP/decided" "$TMP/expanded"
        # The sequence database stays unless the clustering has already been named: a cluster is named
        # by rank, so what maps a rank to a name is part of the answer until the answer carries it.
        rm -f "$TMP/hash" "$TMP/hash."* "$TMP/lin8clust.sh"
        [ "$TSV" -eq 1 ] && [ "$REPSEQ" -eq 0 ] && rm -f "$TMP/db" "$TMP/db".[0-9]* "$TMP/db.hist."* "$TMP/db.runs"* \
            "$TMP/db.files" "$TMP/db.dbtype" "$TMP/db_h."*
    fi
fi
exit 0
