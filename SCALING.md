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

| structure | B/seq | line | live during |
|---|---|---|---|
| sequence db index | 24 | | whole run |
| prefilter db index | 24 | | whole run; round 0 pref has one entry per sequence |
| `assignedCluster` | 8 | 546 | producer + consumer |
| `sortForLength` | 16 | 602 | length sort only (greedy) |
| `lengthOrder` | 8 | 619 | producer loop (greedy) |
| `prefRepSizePair` | 16 | 629 | producer loop (set-cover) |
| `memberOrder` | 8 | 1109 | writeClustering |
| reorder ring | 5M results, ~240 MB | 580 | producer + consumer |

The producer loop holds 64 B/seq, but **the peak is the length sort at lines 600-626**, where
`lengthOrder` is allocated at 619 while `sortForLength` is still alive until 625:

    24 + 24 + 8 + 16 + 8 = 80 B/seq = 800 GB at 10B

on a 995 GB machine with 29 GB of swap. That leaves the page cache about 195 GB against 2.7 TB of
data, and it is the most likely explanation for the 10B run appearing to make no progress after the
banner. The producer loop's own 64 B/seq (640 GB, 355 GB of cache) is the steady state, not the peak.

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
2. **clusthashfast clustering pass.** The figures I first wrote here (4.07B members, 565 GB, 16.3 TB,
   100% miss) were estimates; the live 10B run says **5,299,290,541 members and a 694.4 GB staged
   file**, and vmstat pins the page cache at 651 GB with 6.2 GB free, so resident anon is ~407 GB and
   the cache available during clustering is ~661 GB. Redone with those:

   | | random 4 kB faults | bytes from disk |
   |---|---|---|
   | no staging (661 GB cache vs 1.388 TB) | 5.299e9 x 52.4% = 2.78e9 | 11.4 TB |
   | staging as first written | 5.299e9 x 57-68% = 3.0-3.6e9 | 12.4-14.8 TB |
   | staging + dropping consumed db pages | 5.299e9 x ~4.8% = 2.5e8 | 1.05 TB |

   The middle row is the finding: **staging was a net loss.** The staging pass streams 1.388 TB of db
   pages through the same cache and never marks them dead, and the LRU cannot tell them from the staged
   file, so the file kept only 694/(1388+694) = 33% of the cache - more faults than reading the db
   directly, plus a 2.08 TB pass to get there. Fixed by dropping each thread's consumed byte range as
   it goes (committed): `schedule(static)` gives a thread one contiguous word range, so with an offset
   sorted index its byte range is contiguous and disjoint from every other thread's.

   Also visible in the same vmstat: `wa=0` with 100+ runnable threads, `cs` only 20k/s and `in`
   0.5-1.1M/s means nobody waits on I/O - with 6.2 GB free every fault enters direct reclaim, and
   reclaiming a *mapped* file page needs an rmap walk plus a TLB flush IPI to every CPU. That is the
   interrupt storm, and it is why `us=0, sy=94%`.
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
- **Lowering `BlockAligner`'s `MAX_SIZE`.** Measured, and it is not a tuning constant - it decides how
  far the adaptive band may grow, so it changes which alignments clear the threshold. On the u8 round-0
  input at 20 threads, two reps each, every cell reproducible:

  | MAX_SIZE | wall | maxRSS | clusters | digest |
  |---|---|---|---|---|
  | 4096 | 325.97 / 355.68 s | 17.41 GB | 24,196,835 | `eb31987eb2f2` twice |
  | 1024 | 313.62 / 314.39 s | 17.29 GB | 24,196,837 | `93e6448e4f31` twice |
  | 256 | 309.61 / 309.25 s | 17.24 GB | 24,196,856 | `cd530f02234a` twice |

  Smaller is faster and smaller, monotonically, and the cluster count climbs monotonically with it
  because a narrower band leaves borderline pairs unmerged. 5% of wall and 1% of memory for a changed
  result is invariant 1, so it stays at 4096. The `//change` comment on the define was warning about
  exactly this. Note `StripedSmithWaterman.cpp:37` has an independent `MAX_SIZE 4096` for the striped
  path, not measured here.
- **Replacing mmap with pread.** `getData` hands back a raw pointer into the mapping and callers parse
  in place, so pread needs a per-thread buffer at every site. The reason not to is that it does not
  fix the cost that matters: a 138.8 B body read at random costs one 4 kB page either way, because a
  plain pread still populates a full page in the page cache before copying out - 29x amplification
  before and after. `MADV_RANDOM` already removed the readahead waste on top of that (8.37 MB/fault
  measured down to 4 kB/fault). Only `O_DIRECT` would cut the 29x, to about 3.7x on a 512 B sector,
  and it bypasses the cache - which is wrong here, because a sequence body is the target of many
  queries and genuinely gets reused. A pread path was written and md5-verified for clusthashfast in an
  earlier session and measured **23% slower** locally. The structural fixes for the 29x are staging
  bodies in visit order (which is what `MemberStore` does) or `O_DIRECT`; for reused data staging wins.
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
3. ~~**Delete the hash-bucket partitioning path (~135 lines).**~~ **Won't do - it is reachable.**
   `hashPartitionCount` is passed `halfBudget`, which is `computeMemory / 2`, so one partition needs
   `56n <= 0.9*T`, not `40n <= 0.9*T`. The crossover is **n > 16.4e9**, inside the 20B target, where
   the count becomes 2. `--split-memory-limit L` reaches it independently at `n > L/56`, so
   `--split-memory-limit 200G` partitions a 5e9 run. It is also the oracle that validates the
   in-memory path: identical digests across partitions 1/4/8 and threads 1/3/8 in both key builds is
   what proves the output is partition invariant, and `MMSEQS_CLUSTHASHFAST_PARTITIONS` is the only
   handle that exercises it. Item 1 still subsumes it, so it should not be optimised either -
   mirroring kmermatcher's per-(thread,partition) sink would open 4096 x 128 files.

   The regression this replaced: 18fa6f98 routed the single-partition case through the staged path,
   costing 113 s of the 5B hashing phase (37.7 s -> 151 s) for an 80 GB write and read-back behind one
   mutex and one FILE, plus a single-threaded `fread` that first-touched all 80 GB on one NUMA node
   before a 128-thread sort ran over it. Restored in-memory; measured 4.2x on the round trip in
   isolation at 200M entries.
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

   The related hazard Fable raised is **not** real, checked: `DBReader::open` calls
   `sortIndex(isSortedById)` at line 199 and then recomputes `sortedByOffset` over the final array at
   lines 201-206, after it. `hashScanIsSequential()` is accurate as written. No fix needed.
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
