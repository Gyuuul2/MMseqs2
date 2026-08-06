# Scaling clusthashfast / align2clust / kmermatcher to 10-20 billion sequences

Working notes for the 1 TB single node target. Everything here is either measured on super004
(128 threads, 995 GB, NVMe RAID) or on spark003 (20 cores, 121 GB), or is arithmetic over those
measurements. Estimates are labelled as such.

## 0. The dominant constraint, checked

The user's framing was "20B sequences means 1 byte per sequence = 20 GB". Confirmed, and it is the
right first-order metric:

| | 10B | 20B |
|---|---|---|
| 1 byte/sequence | 10 GB | 20 GB |
| budget at 90% of 995 GB | 89.5 B/seq | 44.8 B/seq |

So at 20B the entire machine affords **under 45 bytes per sequence**, and that has to cover the
reader index, the working arrays, the page cache and the output buffers at the same time.

Measured shape of the 10B input: index tail key `9999999999`, last offset `1,388,439,065,206`, so
exactly 10,000,000,000 sequences over 1.388 TB of data = **138.8 B/sequence** on disk.

## 1. Per-sequence byte budget by phase

`DBReader::Index` is `{DBKeyType id; size_t offset; unsigned int length}`. Under `MMSEQS_INT64_IDS`
(which 10e9 ids force) that is **24 B/sequence**, not 16.

### clusthashfast

| structure | B/seq | live during |
|---|---|---|
| reader index | 24 | whole run |
| `HashEntry[]` (hash + id) | 16 | hashing, sort, clustering |
| member store bitmap | 0.125 | staging onward |
| member bodies staged | 56.5 (measured: 4.07B of 10B in runs >= 2) | clustering |

Peak is the clustering phase: 24 + 16 = **40 B/seq of anonymous memory**, plus whatever page cache
the sequence data and the staged file occupy.

- 10B: 400 GB anon, 495 GB left for cache against 1.388 TB of data.
- 20B: 800 GB anon of 995 GB. **This is the wall.** Not the algorithm - the reader index.

Hard ceiling with zero cache: 995 GB / 40 B = 24.4B sequences. Practical ceiling leaving 300 GB of
cache: about 17B. So 20B in one shot needs the index itself to shrink; 10B has room.

### align2clust (round 0)

| structure | B/seq | note |
|---|---|---|
| sequence db index | 24 | |
| prefilter db index | 24 | round 0 pref has one entry per sequence |
| `assignedCluster` | 8 | atomic per sequence |
| `lengthOrder` | 8 | greedy modes |
| reorder ring | capped at 5M results | not per-sequence |

= **64 B/seq**, i.e. 640 GB at 10B, leaving 355 GB of cache against 2.7 TB of data (sequence bodies
1.388 TB + prefilter 1.35 TB). That 13% cache ratio is the whole story of this module's I/O.

## 2. Passes over the sequence db

### clusthashfast
1. hashing: one **sequential** pass, 1.388 TB. Measured 4 h -> 15 min on the server once it got
   `POSIX_MADV_SEQUENTIAL` plus OpenMP static blocks ~64 MB wide (so neighbouring threads stop
   faulting the same readahead folios). Largest verified win in the module.
2. sort: no db access, but streams the 160 GB `HashEntry` array about four times (ips4o,
   `kLogBuckets = 8`, so log256(1e10/256) ~ 4 levels).
3. clustering: reads bodies in **hash order**, i.e. randomly. 4.07B sequences in runs >= 2, each a
   138.8 B body costing a 4 kB fault = **29x amplification = 16.3 TB**, 3.5 h at 1.3 GB/s.
   Verified on the server: 8.37 MB/fault before `MADV_RANDOM`, 4 kB/fault after - same total, but
   the "before" number was the readahead throwing away 2000x what it used.

### align2clust
1. prefilter prefix pass (SET_COVER only): sequential scan of the whole pref db to count lines.
2. producer loop: prefilter entries read once per representative (random by key), sequence bodies
   read once per surviving target (random). 4.85 TB estimated at 10B.

## 3. What is actually the bottleneck, per module

Ranked by measured or arithmetic size at 10B, largest first.

1. **align2clust random reads** - was hours with no bound. Fixed this session: 3.30x measured on a
   63.3M-query real input, 160,475,604 -> 250,602 major faults. See section 5.
2. **clusthashfast clustering pass** - 16.3 TB / 3.5 h. `MemberStore` attacks the right quantity
   (amplification, not queue depth) but as written stages all 565 GB into one file while only
   ~495 GB can stay cached, so ~12% still misses. Slicing the sorted array at run boundaries and
   staging one slice at a time is the fix; determinism holds because slices are contiguous ranges of
   an already `(hash, id)` sorted array. **Not implemented - see open items.**
3. **clusthashfast sort** - 1.28 TB of traffic over 4 levels. Minutes, not hours. Not worth
   replacing ips4o.
4. **Hamming distance** - already SIMD (`simdi8_eq` + movemask + popcount). Total work at 10B is
   558 GB of byte compares, i.e. seconds across 128 threads. **Not a bottleneck**; a faster kernel
   would be invisible.

## 4. Things that look like wins and are not

- **Target-level parallelism in align2clust.** The outer loop has 3.7B queries for 128 threads, so
  outer parallelism is saturated 30-million-fold, and a query averages ~9-20 targets (measured:
  156.1 B/entry over 17 B/hit on the u4 round-0 pref). `ungappedprefilter.cpp:356` splits targets
  because there the target loop is the whole db; here it is a handful of entries. Standard OpenMP
  guidance is to use all threads at the outer level when it has enough work.
- **`MADV_HUGEPAGE` on the 160 GB `HashEntry` array.** 41.9M pages, 256 entries per 4 kB page, ~4
  ips4o levels, so ~3.1e8 TLB misses at 20-100 ns = 6-31 s of *CPU* = well under a second of wall
  across 128 threads. `new[]` already routes 160 GB through mmap and does not pre-zero a POD. Against
  that, `MADV_HUGEPAGE` over 160 GB with 400+ GB already committed invites direct compaction stalls.
- **`numactl --interleave=all`.** Plausible for the sort, but it is a run-time flag, not code -
  mmseqs does not link libnuma. Worth trying on the server: `numactl --interleave=all mmseqs ...`.
- **Byte-balanced dynamic ranges for the hashing pass.** Measured neutral-to-2%-worse locally
  (33.42/34.63 vs 33.53/35.32 min/mean over 4 reps), and `schedule(static, chunk)` already assigns
  block i to thread i mod T, so each thread samples the whole length distribution at an 8 GB stride
  and systematic byte imbalance averages out. Reverted.

## 5. Changes made this session

### align2clust page-cache policy (Align2clust.cpp, +66; Parameters.cpp, +1)

Root cause: none of the four readers in this module set any madvise, so every keyed random read paid
a full kernel readahead window per miss, and with many threads the shared `mmap_miss` counter never
reaches its back-off.

Three hints behind one predicate:

```cpp
const bool cacheStarved = seqDbr->getDataSize() + alnDbr.getDataSize()
                          > Util::computeMemory(par.splitMemoryLimit);
```

- `POSIX_MADV_RANDOM` on all four readers.
- periodic `POSIX_FADV_DONTNEED` on the prefilter db, which is read exactly once per representative
  so its cache can never be reused. Bounded to 256 drops because each drop walks the cached extent.
- batched `POSIX_MADV_WILLNEED` over the query and target bodies of a prefilter list, issued after
  the list is parsed and before any body is read.

Measured on the u4 round-0 input (63.3M queries, 9.2 hits/query, seq 22 GB + pref 9.3 GB, 20 threads,
cgroup cap reproducing the server's cache ratio, digest identical in all cells):

| cap | binary | wall | major faults |
|---|---|---|---|
| 60G | base | 749.30 s | 324,799 |
| 60G | patched | 745.84 s | 318,291 |
| 20G | base | 4375.10 s | 160,475,604 |
| 20G | patched | **1323.81 s** | **250,602** |

So parity when the data fits, 3.30x when it does not. The gate matters: an earlier version applied
the hints unconditionally and was **14% slower** at 60G with 25x the major faults, because dropping a
cache that would have been reused forces re-reads and `MADV_RANDOM` gives up free readahead.

Also found while unifying the gate: issuing `MADV_RANDOM` *before* the SET_COVER prefix pass poisons
that pass, which is a legitimate sequential full scan of the pref db. Moving the advice after it
recovered 480 s and 28M major faults (1803 s/28.6M -> 1324 s/0.25M).

`--split-memory-limit` was registered for align2clust because `computeReorderCapacity` already read
`par.splitMemoryLimit` but the flag could not be passed on the command line.

### clusthashfast cleanups

- One definition of the hash (`hashOf`), because the array pass and the partitioned pass had
  copy-pasted it and were drifting.
- Bucket write buffers get a fixed byte budget divided by the bucket count. A fixed depth of 8192
  entries would have needed 8192 * 4096 * 128 * 16 B = **68.7 GB**, i.e. the buffers alone would have
  consumed 43% of the deficit that partitioning exists to close.
- `ok` in `hashIntoBuckets` was a plain `bool` written by every thread with no synchronisation; a
  lost `false` is a silently truncated partition. Now a thread-local flag combined with
  `reduction(&&:ok)`.
- `hashBucketCount` keeps a floor under its budget. A large reader index can leave
  `Util::computeMemory` with almost nothing, and the loop below would then ask for the maximum
  partition count on an ordinary run.

## 6. Open items, with the reasoning needed to decide them

1. **Slice the sorted array for staging (largest remaining win, est. 3.5 h -> ~30 min at 10B).**
   After `SORT_PARALLEL` the array is `(hash, id)` ordered and runs are contiguous. Cut at run
   boundaries so each slice's member bytes fit a staging budget (K=2 at 10B, 283 GB each), then
   stage and cluster one slice at a time. Slices in ascending order visit runs in exactly today's
   order and each run is still scanned by ascending id, so the representative election - and
   therefore the output - is unchanged by construction. Better still: one sweep scattering bodies
   into K slice files rather than K sweeps, which keeps I/O at 1.388 TB sequential read + 565 GB
   write + 565 GB read regardless of K.
2. **Gate `MemberStore` staging on the data not being cacheable.** `canStage` only checks file count
   and compression, so on a db that is already fully cached staging is pure added work - that is the
   local 11.03 s -> 19.38 s regression. The test already exists in this file at the preload site
   (`reader.getDataSize() < Util::getTotalSystemMemory() / 2`).
3. **Delete the hash-bucket partitioning path (~135 lines).** It only engages when
   `16n > 0.9*995e9 - 24n - 4e9`, i.e. **n > 22.3e9** - beyond even the 20B target - so it is
   unreachable by default. It costs 320 GB of extra I/O at 10B, and `buildMemberStore` is called
   inside the per-partition loop with `idSpace = dbSize`, so every partition re-marks the full
   10e9-bit space and redoes a 156M-word prefix sum. Item 1 is a strict superset of what it provides.
   Deferred only because it removes a capability rather than fixing a defect.
4. ~~**`open(HARDNOSORT)` instead of `NOSORT`.**~~ **Rejected - it changes the output.** Fable
   recommended this as a one-word diff on the grounds that the tool never looks anything up by key,
   which a grep confirms (`getId`/`getDataByDBKey` appear zero times; every access is
   `getSeqLen`/`getData`/`getDbKey` on a local id). But determinism here rests on sorting by
   `(hash, id)` and electing the first unclaimed entry of each run, and `id` is a **local** id whose
   meaning depends on the index order. `NOSORT` leaves the index in key order, `HARDNOSORT` leaves it
   in file order, so the same run's members get different local ids, the ascending-id scan visits them
   in a different order, and a different member is elected representative - different representative
   key and different member list. Identical only when the index is already id-sorted, which is true
   for `createdb` output but false for a subdb with sparse keys. Violates invariant 1, so not applied.

   The related hazard Fable raised **is** real but is about performance, not correctness:
   `sortedByOffset` is computed in `readIndex` (DBReader.cpp:202) and never recomputed, so
   `hashScanIsSequential()` can report true on an index that `NOSORT` has just permuted into key
   order. The hashing pass would then request sequential advice for reads that are not sequential and
   derive its block width from an entry that is no longer last in the file. Worth fixing inside this
   module (recompute the offset-monotonicity after open) rather than in DBReader.
5. **`readMmapedDataInMemory()` at startup.** Touches up to 500 GB that `dropDataCache` then
   discards, and the hashing pass is already a sequential stream with sequential advice. Belongs to
   the original commit `5a4eaa11`, so it is the author's call; worth measuring with and without.
6. ~~**`posix_madvise(POSIX_MADV_DONTNEED)` may be a no-op under glibc**~~ **Confirmed and fixed.**
   A three-call test program straced on spark003 (aarch64, glibc) shows only two `madvise` syscalls:
   `posix_madvise(POSIX_MADV_SEQUENTIAL)` becomes `madvise(MADV_SEQUENTIAL)` and a direct
   `madvise(MADV_DONTNEED)` appears, but `posix_madvise(POSIX_MADV_DONTNEED)` **issues no syscall at
   all** and still returns 0. POSIX requires `POSIX_MADV_DONTNEED` not to change semantics, while
   Linux's `MADV_DONTNEED` destroys the contents of anonymous mappings, so glibc deliberately drops
   it. Consequence: the madvise half of `dropDataCache` never ran, and because the pages stayed
   mapped `invalidate_mapping_pages` (what `POSIX_FADV_DONTNEED` uses) skipped them too, so the whole
   function was close to inert. Now calls `madvise(MADV_DONTNEED)` directly, which is safe here
   because `MemoryMapped.cpp:299` maps the data `PROT_READ, MAP_SHARED` - dropping the PTEs of a
   read-only file mapping loses nothing and the next access re-faults from the file. Verified on
   aarch64; glibc's stub is in generic sysdeps so x86_64 should behave the same, worth one strace on
   super004 to be sure.
7. **A pathological hash run is serial and quadratic.** One thread owns a whole run and the claim
   scan is O(runSize^2). Millions of near-identical short reads would hit this. There is no
   determinism-preserving parallel fix, so the mitigation is monitoring, not code.
8. **Identity-only fast path.** If `seqIdThr >= 1.0`, folding a second strong hash of the raw body
   bytes into the sort key removes all 16.3 TB of scattered reads (128 bits over 1e10 items gives a
   collision probability ~1.5e-19). Conditional, not general: `Linclust.cpp:188` calls this tool with
   `--min-seq-id max(0.9, seqIdThr)`, and the strong hash must be forward-only because the bucketing
   hash uses `min(forward, revcomp)` while the Hamming check compares forward bytes only.

## 7. Verification protocol used

Results are compared by **content, not bytes** - the physical layout of a result db depends on thread
scheduling and always will. The digest is the xor of per-entry md5 over
`representative_key \t sorted(member_keys)`, which is order independent, so one streaming pass suffices
for 60M entries. Invariance is checked across `--threads` and, where applicable, across
`--split-memory-limit` and partition counts. All eight cells of the clusthashfast refactor
(base/patched x threads 20/4 x partitions 1/4) returned `5f9fa7e4fd47 7710591` on u32, and both
binaries returned `14f2f66600df 58657450` on u4.
