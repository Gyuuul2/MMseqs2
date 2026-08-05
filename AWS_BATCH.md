# `mmseqs {linclust,cluster}-batch --backend aws-batch` — AWS setup contract

This backend runs the batch clustering workflow as a self-submitting chain of AWS Batch jobs: one
**driver** per round splits the input into chunks and submits a **worker** array (each worker does
`createdb` + one round of `linclust`/`cluster` on its chunk), then a **merge** job reconciles the
chunk results (hash-bucket propagate) and submits the next round's driver, until convergence. All
state lives in S3 under the run's work prefix; there is no long-lived driver process.

Because AWS Batch runs your container, **you** own the image, the queues/compute environments, the
job definitions, and the IAM roles. This page is the contract those must satisfy. (Manifest format,
output layout, scale knobs and known failure modes: see §6–§9 below.)

---

## 1. Image contract

Build the AWS image with [`Dockerfile.aws`](Dockerfile.aws) (layers on your mmseqs image). The image
**must** provide, on `PATH`:

| tool | why |
|------|-----|
| `bash` | the container command is `bash -lc "<bootstrap>"` |
| `aws` (CLI **v2**) | bootstrap `aws s3 cp`; drivers/merges call `aws batch submit-job` from inside the container |
| `zstd` | chunk FASTAs and rep FASTAs are `.zst` |
| `mmseqs` | the clustering itself |
| GNU `awk`(gawk), `grep`, `sort`, `join`, `sed`, `tr` | manifest/merge shell logic |

and **must not** have an `ENTRYPOINT` that prepends `mmseqs` (or anything) to the command — AWS Batch
`containerOverrides.command` replaces CMD, not ENTRYPOINT, so a swallowing entrypoint turns
`bash -lc "…"` into `mmseqs bash -lc "…"`. `Dockerfile.aws` resets `ENTRYPOINT []` for this reason.

> Config is delivered via S3, **not** the container-overrides env: the driver writes
> `<work>/scripts/config.env` (%q-quoted `export` lines) once, and every container downloads and
> sources it. So the job definition needs **no** baked-in mmseqs parameters.

Build one image per CPU architecture you route to (round0 may run ARM `x2gd`, round1+ x86 `i4i`);
`Dockerfile.aws` selects the matching awscli build via `uname -m`.

## 2. Compute environment + job queue contract

- One **compute environment** per machine type, `minvCpus=0` so idle instances scale to zero
  (this is what makes round0's queue turn *off* once round0 finishes and round1+ starts on another).
- One **job queue** per compute environment.
- Point the workflow at them one of two ways:
  - **explicit (recommended, simplest):** `--aws-job-queue <arn|name> --aws-job-definition <name:rev>`
    (+ `--round0-aws-job-queue` / `--round0-aws-job-definition` for a different round0 machine).
  - **by tag:** tag each queue *and* its job definition with `mmseqs:machine=<value>` (key override:
    `--aws-machine-tag-key`) and pass `--aws-machine <value>` / `--round0-aws-machine <value>`. The
    workflow resolves the queue+definition by tag at submit time.

## 3. Job definition contract

- References the **Dockerfile.aws image** (per architecture).
- `vcpus`/`memory` sized for one chunk's `createdb`+`linclust` (chunk size is set by
  `--chunk-max-bytes`/`--chunk-max-seqs`). Pass `--threads` to match the definition's vCPUs
  (the workflow pins `--threads` into every container for determinism; it does **not** auto-detect).
- A **host volume mounted** at the path you pass as `--node-work-dir` (container-local scratch for
  createdb/cluster tmp + sort spill). Without a real mount this lands on the container's small overlay
  and fills the OS disk. `--backend aws-batch` **requires** `--node-work-dir`.
- Leave `command` to be overridden (the workflow sets it). A `retryStrategy` on the definition is
  additive with the workflow's own `--retry-strategy attempts=N` for workers.

## 4. IAM contract

Two distinct principals — note the second is unusual:

- **Submitting principal** (whoever runs `mmseqs … --backend aws-batch`): `batch:SubmitJob`, and for
  `--aws-machine` tag resolution `tag:GetResources` (resolves the queue+definition by tag in one
  Resource Groups Tagging API call) plus `batch:DescribeJobQueues` (to keep only ENABLED/VALID
  queues). Plus `s3:PutObject` on the work prefix (uploads the script + `config.env` + input manifest).
  With explicit `--aws-job-queue`/`--aws-job-definition` no tag/describe permissions are needed.
- **Job role** (`jobRoleArn` on the job definition — the role the *containers* assume):
  - `s3:GetObject`/`PutObject`/`ListBucket`/`DeleteObject` on the **work** and **result** prefixes
    (workers read chunks + write results; merge reads/writes markers; cleanup deletes tmp).
  - **`batch:SubmitJob`** — because drivers and merges submit the next round's jobs *from inside the
    container*. Outside users do not expect a job role to need SubmitJob; without it the chain
    stalls after round 0.

> A missing `s3:GetObject` on the work prefix is especially bad: existence checks (done-markers) read
> a 403 as "absent", so the merge can resubmit forever. Grant the job role read on the whole work
> prefix.

## 5. S3 prefixes

Pass **disjoint** work and result prefixes, and never a bucket root:

```
mmseqs linclust-batch s3://bucket/input.manifest s3://bucket/result/run1 s3://bucket/work/run1 \
    --backend aws-batch --aws-machine i4i.metal --round0-aws-machine x2gd.metal \
    --node-work-dir /scratch/mmseqs --threads 64
```

The final merge deletes the work prefix when `--remove-tmp-files 1` (default). Keep the result prefix
outside the work prefix so results are never in the deletion scope. Set an S3 lifecycle rule to abort
incomplete multipart uploads on the bucket.

## 6. Input manifest & output

**Input manifest** (`<input>` argument): a text file, one input per line, `#` comments allowed.
Each line is an `s3://…` FASTA path (`.fa`, `.fa.zst`, `.fa.gz`); a line may optionally carry
`path<TAB>bytes<TAB>seqs` (bin-packing then needs no size probe). Every path must be readable by the
job role (see §4).

**Output** (under `<resultDir>`): the clustering is written as
- `final_cluster_manifest.txt` — lists the cluster shards,
- `final_cluster_shards/final.bkt*.tsv[.zst]` — the `representative<TAB>member` assignments,
- `final_rep_seq.fasta[.zst]` (+ `final_rep_seq_manifest.txt`) — the representative sequences,
- `final.done` — written last; its presence means the run is complete (a re-run short-circuits on it).

## 7. Scale knobs

| knob | effect |
|------|--------|
| `--chunk-max-bytes` / `--chunk-max-seqs` | round-0 chunk size (fit one chunk's createdb+cluster in the job definition's memory) |
| `--round0-chunk-max-bytes` / `--round0-chunk-max-seqs` | round-0-only chunk size override |
| `--threads` | per-container threads — **pass explicitly**; it is pinned into every container (not auto-detected) |
| `--max-rounds` | representative-round cap |
| `--merge-buckets` / `--merge-bucket-jobs` | hash-bucket count / concurrent bucket jobs on the merge |
| env `BATCH_AWS_TIMEOUT` | per-job `attemptDurationSeconds` (default 43200 = 12 h) |
| env `BATCH_AWS_WORKER_ATTEMPTS` | AWS-native worker retries for spot/OOM (default 2) |
| env `MAX_CHUNK_ATTEMPTS` | merge-level re-drive of missing chunks before dead-lettering (default 2) |

Determinism: the clustering result is independent of chunk count, `--threads`, and worker parallelism
(chunk boundaries use exact uncompressed sizes recorded per chunk, not a compression estimate).

## 8. Failure & recovery

- **Worker chunk failure** self-heals: the merge reconciles missing chunks (done-markers) and re-drives
  ONLY the missing ones, up to `MAX_CHUNK_ATTEMPTS`, then writes `round<N>/DEAD_LETTER.txt` and fails.
- **Driver/merge loss** (spot reclaim, OOM, or exceeding `BATCH_AWS_TIMEOUT`) does **not** auto-heal:
  AWS Batch `dependsOn` fires only on SUCCEEDED, and drivers/merges are submitted with a single attempt
  (a retried merge would misread the chunk-attempt counter). The chain then stalls with no further jobs.
  **Re-invocation is safe on the main path**: the round→round submissions (a driver's worker+merge, a
  merge's next-round driver) each record their job id in a `round<N>/submitted_*.txt` marker, so
  re-invoking a stalled `aws-driver`/`aws-merge` — manually, or from an EventBridge Batch "job FAILED"
  rule you attach to the queue — adopts the job it already submitted instead of forking a second chain.
  The **missing-chunk retry** path is the exception: it advances a read-modify-write attempt counter, so
  do not re-invoke a merge that is mid-retry (let the retry finish). An EventBridge rule on the driver
  and the round-boundary merges is the recommended unattended safety net.
- **Never launch a second `aws-submit`** against the same work prefix while one chain is live (AWS
  cannot cheaply detect it); the two would race shared S3 state.

## 9. Testing before a live run

- **Dry run** (no AWS calls): set `BATCH_AWS_DRY_RUN=1`. Each `aws batch submit-job` is printed instead
  of executed, so you can review the exact queue/definition/command/dependency plan first. It needs no
  AWS credentials when queues/definitions are given explicitly.
- **Offline smoke test** (CI): `bash data/workflow/batch_clustering_smoke.sh` validates that the
  container-overrides JSON is well-formed and that the run config round-trips — no AWS or mmseqs needed.
- **Cross-architecture check** (only if round0 and round1+ run different CPU arch, e.g. ARM `x2gd` +
  x86 `i4i`): cluster one identical chunk set on each queue and diff the TSVs before trusting a
  split-architecture plan for a determinism-critical (paper) run.
