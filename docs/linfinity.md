# linfinity: multi-node linear clustering

Design of the linfinity modules for the developer who maintains them; commit subjects are one line, the
reasons live here. Checked against the code of the commit that adds this file; anything not confirmed in the
code is marked "(unverified)".

## 1. What linfinity is

linfinity clusters protein sets of 1 M to 1 T sequences on 1 to about 100 nodes with the same code. It is
linclust2 (`linclust --linclust-version 2`) with every step made node-parallel. It is used through

```
mmseqs easy-lin8clust in.fasta[.zst] res tmp --node-runner "srun --kill-on-bad-exit=1 -N 8 --ntasks-per-node=1 -c 64" [options]
mmseqs createdb in.fasta db --createdb-mode 3; mmseqs lin8clust db clu tmp --node-runner "..."
```

`--node-runner` starts one process per node for every multi-node step. Without it each step runs as one node.

| Step | Module | Work |
|---|---|---|
| 0 | `createdb --createdb-mode 3` | reads the input once, writes a length-descending linfinity DB, one data file per node |
| 1 | `clusthashfast` | hash pre-clustering of near-identical sequences; writes cluster parts and representative bits (`_redundancy`) |
| 2 | `kmermatcher` | k-mer exchange between nodes, writes the prefilter DB `pref` by target split |
| 3 | `align2clust` | aligns rep → target pairs on the align nodes, the master clusters the accepted pairs |
| 4 | `mergeclusters` | merges the clusthash clusters with the alignment clusters |
| 5 | `createsubdb`, `kmermatcher`, `align2clust --filter-cludb-file --filter-seqdb-file`, `mergeclusters` | iteration 2 on the representatives (`--linclust-iterations 2`, default) |
| 6 | `createtsv`, `result2flat` | per-node `res_cluster.tsv.<r>` and `res_rep_seq.fasta.<r>` (easy-lin8clust) |

Scripts: `data/workflow/lin8clust.sh`, `data/workflow/easylin8clust.sh`; drivers `src/workflow/Lin8clust.cpp`,
`EasyLin8clust.cpp`. `lin8clust` refuses non-linfinity DBs and subsets (a DB with `.bits.0`).

### Relation to linclust2

- Keys are assigned in length-descending order (ties in input order), so greedy incremental clustering in key
  order is "longest sequence first", as linclust2 `--cluster-mode 2` on a length-sorted DB.
- `--cluster-mode`: 0 (set cover), 2 and 3 (greedy) are supported, anything else stops. Without it lin8clust
  picks 3 for `--cov-mode 1/2` and 0 otherwise (`Lin8clust.cpp`).
- lin8clust forces `--diagonal-mode` ungapped, `--include-count-table 0` and the clusthash pre-step
  (`--min-seq-id` at least 0.9 for clusthashfast); alphabet and adjacency default to the linclust values unless
  given. Iteration 2 uses spaced k-mers with a tenth of the k-mers per sequence. It uses its own k-mer record
  (no diagonal) and its own alignment scheduling, so results are similar to, not identical with, linclust2
  (unverified: no current md5 comparison).
- `--createdb-mode 3` drops entries without residues (counted in a WARNING); headers keep only the first word.

### Core invariant

**For a given input and parameters, the final clusters are byte-identical for any node count, thread count,
memory limit and timing.** Intermediate layouts may differ (DB file boundaries, windows, which node took
which work item), but no decision that changes a cluster may depend on them. The checks in §8 compare the
sorted TSV pairs of runs with different P and T. Every change must keep this.

How the code keeps it:
- All ordering is by key; every node computes the same layout from the same published numbers.
- Work queues (§2) decide only who does an item, never what the item produces.
- The master clusters accepted pairs in global order key order, and only up to the smallest key every node
  has passed (§5.6). Align nodes may drop work early using the master's published assignments, but the master
  re-checks every pair; dropping is never a decision.

## 2. Node model

- `MMseqsNode::init()` reads rank and count from `OMPI_COMM_WORLD_RANK/SIZE`, `PMI_RANK/SIZE` or
  `SLURM_PROCID/SLURM_STEP_NUM_TASKS` (first pair found). No MPI library is used.
- Rank 0 is the master. With P ≥ 2 the master does not align (`getFirstAlignNode() == 1`); it clusters,
  gathers and writes the small shared files. With P = 1 the one node does both.
- Every step of a pipeline runs with the same P and the same `--threads` on every node; kmermatcher checks
  `--threads` in its memory barrier. Memory may differ per node: sizes that must agree use the smallest
  limit (`getSmallestMemoryLimit`), per-node buffers use the node's own `--split-memory-limit`
  (`MMseqsNode::getMemoryLimit` = `Util::computeMemory`, default 90 % of RAM; set it when several ranks share a
  machine or a cgroup).
- Nodes communicate only through files in the shared tmp folder, so all nodes must mount one POSIX file system
  (NFS, Lustre, GPFS; on AWS e.g. FSx for Lustre, EFS or an EBS volume exported over NFS, since an EBS volume alone
  is attached to one instance):
  - Run files: `publishRunFile` writes `<file>.tmp`, fsync, rename. The content starts with a run id built from
    `SLURM_JOB_ID`, `SLURM_STEP_ID`, `SLURM_RESTART_COUNT`, `OMPI_MCA_ess_base_jobid`, `PMIX_NAMESPACE`,
    `MMSEQS_RUN_ID`, `MMSEQS_STEP_ID`; a reader ignores files of other runs, so leftovers of a killed run are
    never read.
  - `barrier(file, content)`: each node publishes `<file>.<r>`, the master gathers them in node order and
    publishes `<file>`; every node returns the gathered content. One barrier is O(P) small files.
  - Logs: a node appends to a file and publishes `<log>.progress` (`key \t size`) after fsync; readers read
    only up to the published size (`publishLog`, `readLog`, `waitForLog`).
  - Exchange files: one file per sending node, records sorted by destination, a column-major offset table at
    the end, sizes gathered by one barrier on `<exchange>.index` (`ClusterPairWriter`, `closeExchange`,
    `openExchange`, `readExchangeRecordsUpTo`). Records for the own node stay in memory. Readers punch holes
    behind what they read (`FileUtil::punchHole`) so exchange disk shrinks while it is consumed.
  - Work queues (`MMseqsNode::WorkQueue`): item `i` is claimed by creating `<name>.<i>` with `O_EXCL`. A node
    first takes its own items (`i ≡ rank mod P`), then steals from the others from the back.
- NFS: a missing name may be cached for up to 60 s. Polled run files are opened with `O_CREAT` on NFS and the
  directory is reopened between polls (an `ENOENT` for 60 s stops the node); other shared opens retry a
  cached `ENOENT` and list the directory at most once a second (`FileUtil::needsDirectoryRefresh`, decided by `statfs` type; unknown storage
  is treated as NFS).
- Data files and exchange files are closed without fsync; run files and logs are fsynced.
- Rerun rule: a failed step is rerun, never repaired. `lin8clust.sh` skips a step whose output `.dbtype`
  exists; `easylin8clust.sh` uses `createtsv.done` / `result2flat.done`, because these outputs have no dbtype.
  The master writes the `.dbtype` last. Output names carry no job id.
- `SLURM_KILL_BAD_EXIT=1` is exported by the scripts: one failed rank ends the whole step.

## 3. Data formats

### linfinity sequence DB (`createdb --createdb-mode 3`)

| File | Content |
|---|---|
| `db.<f>` | sequences `seq\n\0` of a contiguous key range, f = 0 … P−1 (empty files allowed) |
| `db.index` | one text row per length range: `firstKey offset entryLength count file`; offset is local to `db.<file>`, entryLength = L + 2 |
| `db.dbtype` | amino acids with `DBTYPE_EXTENDED_LINFINITY` (64) |
| `db.file_keys` | `firstKey count` per file; it is the index of the header DB |
| `db_h.<f>` | first word of each header, `name\n\0`, key order |
| `db_h.index.<f>` | binary uint64 start offsets of the names, count + 1 values |
| `db_h.dbtype` | generic with the linfinity bit; with `--compress-headers 1` also the compressed bit, then `db_h.<f>` is zstd seekable format (1 MiB frames, seek table at the end) |
| `db.source` | input file names |

There is no per-sequence `.index` and no `.lookup`. key → sequence is arithmetic on the range row:
`offset + (key − firstKey) × entryLength` in `db.<file>`. Totals (size, residues, max length) are sums over rows,
so they are global on every node.

Readers open a linfinity DB with `DBReader::USE_LINFINITY` and `NOSORT`, `HARDNOSORT` or `LINEAR_ACCCESS`; any
other tool or access mode stops in `DBReader::open`. Key-range reads go through `loadData(keyBegin, keyEnd)`,
`prefetchData`, `loadData(keys)`, with `BatchReader` (io_uring when available, else pread threads). Node key
ranges come from `getFileKeyOffsets(P)` (the data files; stops if their count ≠ P) or
`getEvenNodeKeyOffsets(P)` (equal key counts, used by mergeclusters, createtsv, align2clust output).

Subsets (`_redundancy` of clusthashfast, `input_rep` of createsubdb) link the data files of the source DB and
add `<sub>.bits.<f>` (header `{firstKey, keyCount}`, one bit per key) and a sixth index column with the
representative count of each range (used for k and e-values).

### linfinity cluster DB

`clu.<r>` holds entries `rep\nmember\n…\0`; `clu.index.<r>` is binary uint64 `(key, offset)` pairs ending with
`(DB_KEY_INVALID, data size)`; dbtype `CLUSTER_RES` with the linfinity bit. There is no global index; readers
binary search the per-node index. The first line of an entry is the representative.
- clusthashfast output has no singleton entries; a singleton is a key with its bit set in `_redundancy.bits`
  and no entry.
- align2clust output on a subset with bits omits singletons too; mergeclusters brings them back. On a whole DB
  (no bits) align2clust writes the singletons from the master's assigned bits.
- align2clust and the first mergeclusters write by balanced key ranges (`ClusterPairWriter` counts pairs in
  1024 key bins per node), so `clu.<r>` holds about equal line counts. The second mergeclusters keeps the node
  ranges of its first clustering; createtsv balances its output by cluster bytes (`getBalancedClusterKeyOffsets`).

### Prefilter DB (`pref`)

- One align node: `pref.<r>` / `pref.index.<r>` per writing node, entries `rep: target gaps` varint coded.
- Two or more align nodes: written by target split. The DB blocks (16 MiB runs of keys inside one data file,
  `DBReader::setBlockCacheSize`) are cut into at most `8 × align nodes per group` target splits. A writing node
  r writes `pref.<r·S + s>` for target split s; an entry holds rep key gap, size class, target gaps, 0.
- `pref.partitions` (master): line 1 the align group count, line 2 the first block of every target split, then
  one row per written piece `firstKey entry prefBytes cost node`, sorted by key. align2clust builds its windows
  from these rows.

### Exchange and control file names

| Module | Files (all `.<rank>`, tables in `.index`) |
|---|---|
| createdb | `<db>.splitplan.<r>`, `<db>_sent_entries`, `<db>_sent_edges`, `--local-tmp` spool `_sent_sequences` |
| kmermatcher | `<pref>.kmers*`, `<pref>.pairs*`, queues `.send_splits.<i>`, `.hash_ranges.<i>`, `.merge_splits.<i>`, `<pref>.memory`, `.progress.<r>` |
| align2clust | `<clu>_accepted.<r>` (log), `<clu>_assigned_keys` (+ `.bits`), `<clu>_clustered.<r>`, set cover spills `<clu>_clustered_pairsets/_sets/_large`, dirs `<clu>_sequence_requests/`, `<clu>_served_sequences/`, `<clu>_member_requests`, `<clu>_members`, `<clu>_cluster_pairs`, `<clu>.memory`, `<clu>.cluster_key_offsets` |
| mergeclusters | `<out>_by_member`, `<out>_by_rep`, `<out>.cluster_key_offsets` |
| createtsv | `<clu>_tsv_by_member`, `<clu>_tsv_by_rep`, `<clu>_tsv.cluster_key_offsets` |

## 4. Modules

### createdb `--createdb-mode 3` (`createdbLinfinity`, `createdb.cpp`)

1. Plan (master): input files in upstream order are cut into input splits of `total / (P × T)` bytes. Plain
   FASTA is cut at any byte (an entry belongs to the split its `>` falls in); a `.zst` is cut only if it starts
   with a pzstd skippable frame (a split holds the frames that start in it); gz, bz2, FASTQ and single-frame zst
   are one split each (WARNING). Splits go to nodes in contiguous runs of about equal bytes,
   published as `<db>.splitplan.<r>`.
2. Read once: threads read splits; entries collect in a buffer, are counted-sorted by length at each flush,
   zstd-compressed in 64 KiB units and appended to `<db>_sent_entries.<r>` with a per-length row table. If all
   nodes have local flash with room, sequences go to a `--local-tmp` spool instead.
3. Layout: one barrier carries each node's length histogram. Every node computes the same layout: keys by
   length descending, then input split order; file boundaries minimize the largest file
   (weight = sequence + header + 8 B per key; a length is never split; binary search on the maximum). The master
   checks free space and publishes `db.index` and `db.file_keys`; every node `ftruncate`s its three files to
   the final size, so file-size limits fail at once.
4. Write (entry buffers `0.95 × limit / (2 T)` per thread; the received length rows must fit memory): each node reads its length rows from all senders' exchange files and writes `db.<r>`, `db_h.<r>`,
   `db_h.index.<r>` with `BatchWriter`. With a spool, senders write whole pages of their sequence runs into
   the target `db.<f>` directly; partial edge pages go through `<db>_sent_edges` to the owner, so every page
   has one writer.

### clusthashfast (`clusthashfast.cpp`)

Node r takes the key range of `db.<r>`, no communication. Lengths are grouped into splits that fit memory;
per split: hash each sequence on the reduced alphabet, sort `(hash, id)`, representative = smallest key of a
hash run, members checked against `--min-seq-id` (lin8clust passes at least 0.9) in `findRepresentative`. A length larger than
memory is processed in hash ranges with read chunks. Output: cluster part `<out>.<r>` (+ binary index),
`<out>_redundancy` = links to the data files, `.bits.<r>` and the 6-column index (via
`writeRepresentativeBits`, gathered by the master).

### kmermatcher (`kmermatcherLinfinity`, `src/linclust/kmerexchange.cpp`)

Three phases, each over a work queue, so a slow node takes fewer items:
1. Send: `8 × P` send splits (1 for P = 1) of about equal cost (k-mers × 12 B + sequence bytes). A node loads a send
   split, extracts the k-mers of its representatives (bits) and writes them to k-mer file groups by hash range.
   Some own hash ranges stay in memory.
2. Receive: `max(8 × P, 2 × T)` hash ranges; a node reads all senders' records of a range in passes that fit
   the smallest memory limit, groups by k-mer and emits (rep, target) pairs into pair file groups by key
   partition.
3. Merge: per send split, merge the pair partitions into prefilter entries; with ≥ 2 align nodes write them
   by target split (§3) and report the `.partitions` rows.
The align group count is decided here (`--align2clust-split`, 0 = ⌊√(align nodes)⌋, fewer if a group node's
share of the sequences does not fit a quarter of the smallest memory limit) and stored in `.partitions`.
`--local-tmp` may hold the last exchange file groups before they are copied to the shared folder.

### mergeclusters (`mergeclusters.cpp`)

`mergeclusters SOURCE out C1 C2` with C1 the lower level. Two exchanges with `sendClusterPairs`: (member,
rep) pairs of C2 to the member's node; then every C1 pair, rewritten to its final representative, to the
representative's node; the receiving node writes its entries in key order with singletons from the bits.
Only two clusterings are merged (the workflow never passes more).

### createsubdb, createtsv, result2flat

- createsubdb (`input_rep`): sets a bit for every representative of the cluster DB in the node's file range;
  no data copied.
- createtsv: (member, rep) pairs to the member's node; that node reads its names once in key order and sends
  (rep, member, name) to the rep's node; the rep's node writes `rep\tmember` lines in key order to `<tsv>.<r>`.
  The name is `Util::parseFastaHeader` of the stored first word.
- result2flat: each node writes the representatives of its file key range to `<out>.<r>`, in splits whose
  sequences and names fit an eighth of the memory limit.
- Outputs stay per node; concatenating them in node order gives the rep-key-ordered result.

## 5. align2clust in depth (`src/alignment/Align2clustLinfinity.cpp`)

### 5.1 Align groups and target splits

The align nodes (ranks 1 … P−1 for P ≥ 2) form `G` align groups (`.partitions` line 1). Every group holds all
target splits; inside a group the splits go to the nodes back and forth (0 1 2 2 1 0 …) and
`getBlockAlignNode(block)` returns the node of a block's target split. A group aligns every window assigned to
it, so a node sees only the targets in its own splits plus the reps it fetches.

### 5.2 Windows

Windows are runs of consecutive `.partitions` rows of one writing node, cut at equal alignment cost; their
number is `ceil(prefBytes / (nodes per group × prefilter read size))`, the read size being the pref bytes
that fill a prefilter buffer of `smallestMemoryLimit / 32` (`getMaxPrefReadSize`). Windows go in key order to the group with the least cost per node, so the groups
advance through the keys side by side. A node reads the entries of its target splits of its group's windows
into three rotating prefilter buffers (`TargetSplitReader`), each sorted by order key.

### 5.3 Fetches

A fetch (`FetchedSequences`) is a run of prefilter records (rep → targets) plus pending pairs (iteration 2).
The I/O threads gather its sequences: own keys from `seqDbr`, keys of other align nodes through requests
(`_sequence_requests/`, `_served_sequences/`). Fetch size rule: the first fetch is `min(maxFetchSize / 64,
64 MiB)`, each next one doubles, capped by half a second of aligning at the planned-bytes rate of the last
aligned fetch (`READ_SECONDS / 2 × plannedBytesPerSecond`). More fetches start while the planned bytes in
flight are below one second of aligning and fetch slots and memory remain. `MMSEQS_ALIGN_FETCH_BYTES` fixes the
limit for measurements.

### 5.4 Serving

Targets are served by their target-split node. Reps inside a fetch's rep key range are served round the group
by block (`alignGroupNodeBegin + block % groupSize`, `getFetchSource`), which spreads the rep reads. A served
pass packs the unassigned entries (5-bit alphabet, assigned keys sent as zeros) into one served file per pass
for all requesters; requesters poll a progress file and directory listings. Reps of other nodes' blocks are
read from the shared DB into a separate `repDbr` block cache when memory allows.

### 5.5 Block cache and `--local-tmp`

Memory besides fixed buffers becomes a block cache of whole 16 MiB blocks (`setBlockCacheSize`). If the cache
is too small to beat two packed copies per miss, keys are read one by one ("read directly"). With `--local-tmp`:
- if the cache can hold the node's share of the sequences, or cannot hold the largest served block (the copy
  buffer), nothing is copied;
- else the node reserves room for its served blocks in key order and the I/O threads copy them block by block
  while aligning (`copyServedBlock`; with no I/O thread before aligning); a block is read locally once fully
  written. A write error stops copying, the rest is read from the shared folder.
- `--local-tmp` must be local flash on another device than the DB, else `--local-tmp-fallback 0` stops and 1
  (default) warns and reads from the shared folder. Iteration 2 first copies the node's cluster part, whole or
  not at all.

### 5.6 Clustering by the master, and the gate

- Align nodes append accepted pairs to `<clu>_accepted.<r>` in order key order: a rep word (bit 63 set) then
  member keys. Order key = key for greedy; for set cover = size class pass × padded DB size + key, largest
  classes first. A node publishes its done rep key with the log progress.
- The master (`AcceptedPairClustering::step`) reads every log up to the smallest done key over all nodes (the
  slowest node gates the step), merges the reps in order key order, drops pairs whose rep or member is
  assigned, assigns (greedy: a free rep takes its free members; set cover: sets by size with spill files in
  `--local-tmp` when possible) and publishes the step's assigned keys, Rice-coded, in `<clu>_assigned_keys`.
- Align nodes read the assigned keys at least once a second and after publishing their log (every 0.125 s while
  gated) and skip assigned reps and targets.
- Gate: a fetch is requested only while its first rep key is at most the master's clustered key or this node's
  unclustered accepted bytes are below its share of a master step (`getStepWords / P` words). This bounds how
  far a node runs ahead and the master's memory.
- At the end every node turns the master's `_clustered.<r>` result into cluster pairs and writes the clusters
  by balanced key ranges.

### 5.7 Iteration 2: cluster members

With `--filter-cludb-file` an accepted pair (rep2, rep1) becomes a pending pair: rep1's iteration-1 members
must pass against rep2 too. Each node answers member lookups for its own cluster part
(`ClusterMemberExchange`, `_member_requests` / `_members`), preloads the entries of its own target splits,
and reads member sequences from `--filter-seqdb-file`. A pair whose members do not fit one fetch is checked in
parts; it is accepted only if all parts pass.

### 5.8 Threads

- A node never runs more threads than `--threads` T (io_uring kernel workers and BatchReader read threads are
  not counted). I/O threads I = min(2, T − 1) if sequences or members cross nodes, else min(1, T − 1); the
  OpenMP team is T − I. With I = 0 the main thread runs the I/O tasks between batches.
- An I/O pass runs one bounded step of each task: serve members, cluster step (single node), serve sequences,
  receive sequences, copy own keys ahead, preload members, one prefilter fill step; a block copy only if nothing
  else worked.
- Prefilter fill: the I/O threads fill buffer n + 1 in steps (start a pref read, an entry or entry part, the
  sort). If the main thread needs that buffer first, it takes over and finishes it with its idle team.
- Entries are decoded in parts of about `1 / T` of a pref read, so one large entry does not serialize the fill.
- The master of P ≥ 2 clusters on the main thread's team; the assigned-key writer and the member server use
  the spare I/O threads.

### 5.9 Memory

Per node: block cache = limit − (3 × 2 prefilter buffers of limit / 32, fetch buffers, served file buffers,
member lookup, copy buffer) (`getReservedMemory`). The master holds one bit per key (`dbSize / 8` bytes,
independent of P) plus a 16 MiB log buffer per node, and clustering memory: its whole master limit for set
cover, a quarter for greedy. The master limit is half the node limit when the master also aligns (P = 1).
Every feasibility check (memory, open files via `checkRlimitNoFile`, local disk) runs before the first
alignment.

## 6. Failure handling

- Transient shared-FS errors are retried: `MMseqsNode::waitBeforeRetry` retries once at once, then sleeps
  20 ms doubling to 1.28 s, and gives up when the same operation failed at least 100 times and for at least
  60 s (`hasFailedTooLong`). Covered: `openSharedFd`/`openSharedFile` (EIO, ESTALE, NFS-cached ENOENT),
  `statSharedFile` (EIO, ESTALE; a missing name returns false at once),
  `FileUtil::preadAll` (EIO, ESTALE, EAGAIN, short read; reopens the fd), `BatchReader::retryRead`, log reads,
  polled run files, exchange index lines.
- Fatal: write errors (`pwriteAll`, no space), close errors of written files, corrupt formats (wrong magic,
  bad seek table, incomplete index), memory or fd limits found at setup.
- Waits: `WaitWarning` prints an ERROR after 600 s of one wait and again at doubling intervals ("Check that no
  node has died…"). `MMSEQS_WAIT_TIMEOUT=<s>` stops the node after one wait of that length. A dead node is
  normally handled by the runner (`--kill-on-bad-exit`).
- The exchange files of kmermatcher (file groups) and align2clust (requests, served files, member files) carry
  a 16-byte header (magic mixed with the run id, file number) that the reader checks together with the
  published size (`tryOpenExchangeFile`), so a stale or half-visible file is polled again, not read as data.
- gz and bz2 input are read without retry; a read error stops createdb (use plain or pzstd input at scale).

## 7. Reading a run log

Each module prints `[node r] name: value` lines (no prefix for P = 1) at the end of its phases
(`MMseqsNode::printPhaseValues`, INFO level, every node). "Time for X" lines are phase times. Useful align2clust lines:

| Line | Meaning |
|---|---|
| `Target splits`, `Align group`, `Target split entries` | this node's splits and sequence bytes, its group, windows |
| `Block cache` | cache bytes vs coverable and own sequence bytes; "read directly" if too small |
| `Local tmp` | what `--local-tmp` holds or reserves, or why it is not used |
| `Local block copy` | blocks copied while / after aligning |
| `aligning the fetches` | pure aligning time; compare with the waits below |
| `waiting for the sequences / the prefilter / the master` | main-thread waits; master = gate or done-key wait |
| `Prefilter fill` | buffers, how many the main thread finished, read/parse/sort seconds, main blocked seconds |
| `Fetches in flight`, `Fetches from N planned bytes` | fetch pipeline depth and per-size buckets |
| `Gate slack` | how far the node's front rep key ran ahead of the master (keys, windows) |
| `Target pairs` | examined, skipped by rep / target, uncoverable, aligned |
| `SWIPE … alignments`, `SWIPE extensions in 8 bit lanes` | lane fill of the SIMD aligner |
| `Process threads` | sampled thread count incl. io_uring workers vs `--threads` |
| `Window decile k` | per tenth of the node's windows: cells, pairs, aligning vs other seconds |
| `Sequence exchange bytes`, `Served bytes per requesting node`, `service: …` | serving cost and balance |
| `Align node status after N s` | once a minute: done rep key, waits, idle reasons, gate slack |

Master: `clustering the accepted pairs`, `waiting for the aligning nodes`, `Logged pairs dropped`.
kmermatcher: `sending k-mers`, `receiving k-mers`, `merging pairs` with byte counters.

Environment variables for measurement: `MMSEQS_IO_MODE` (auto|buffered|direct), `MMSEQS_IO_CONCURRENCY`
(reads in flight; defaults: parallel FS 128, NFS 16, local disk 1, flash T), `MMSEQS_ALIGN_FETCH_BYTES`,
`MMSEQS_LOCAL_COPY_BYTES` (caps the local copy), `MMSEQS_ALIGNED_PAIR_LOG=<dir>` (aligned pairs per node),
`MMSEQS_WAIT_TIMEOUT`. O_DIRECT is used automatically only for
files larger than 10 % of RAM on local devices.

## 8. Verifying a change

Build with `-DMMSEQS_INT64_IDS=1` (linfinity assumes 64-bit ids). Compare results only by sorted TSV pairs:

```
cat res_cluster.tsv.* | cut -f1,2 | LC_ALL=C sort | md5sum | cut -c1-12
```

Simulate P ranks on one machine with a node runner that starts the module P times on the same tmp folder:

```sh
# rankN.sh
for ((r = 0; r < RANKS; r++)); do SLURM_PROCID=$r SLURM_STEP_NUM_TASKS=$RANKS "$@" & pids="$pids $!"; done
rc=0; for p in $pids; do wait "$p" || rc=1; done; exit $rc

RANKS=5 mmseqs easy-lin8clust uniprot_64.fasta res tmp --threads 4 --node-runner ./rankN.sh <options>
```

Expected md5 on `uniprot_64.fasta` (3,956,337 TSV lines), `--min-seq-id 0.5 --kmer-per-seq 80
--split-memory-limit 2G` (greedy with 5 ranks × 4 threads, set cover with 17 ranks × 1 thread, on this tree):

| Config | md5 |
|---|---|
| `--cov-mode 1 --cluster-mode 2` (greedy) | `3823f64cd1bd` |
| `--cov-mode 0 --cluster-mode 0` (set cover) | `0adb403f73d9` |

The same md5 must come out for 1, 3, 5 and 17 ranks, for several `--threads`, with `--align2clust-split 1`
and with a small `--split-memory-limit`.
A change that cannot change results needs one md5 run of the combined tree; a change that can needs the set.

## 9. Known limits and open measurements

- Scaling curves (P = 3, 5, 9, …; 100 M to 10 B) are not yet measured for the current tree.
- Prefilter fill at large T (e.g. 192 threads): the main-thread takeover and entry parts are unmeasured there.
- The `--local-tmp` copy while aligning re-reads cached blocks from the shared folder; whether it wins is
  unmeasured where the cache is smaller than the node's share.
- Served progress reads and request listings grow with the align group size; groups larger than 16 nodes are
  untested (unverified). Not enforced in code.
- The master's assigned bits are `N / 8` bytes regardless of P (125 GB at 1 T); the master's clustering
  rate is a serial ceiling (measure before 1 T).
- File-size limits (16 TiB on ext4/Lustre defaults) for `db.<f>` at small P; createdb fails at `ftruncate`.
- Without fsync, a successful close on Lustre/GPFS does not prove the data reached disk; zeros inside a file
  of the expected size are not detected.
- `openAndDelete` and `FileUtil::remove` are not retried.
