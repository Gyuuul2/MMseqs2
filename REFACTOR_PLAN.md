# batch_clustering 리팩터 계획

## 왜

```
                     줄 수   [[ ]]   local   함수   awk 줄
upstream 최대(databases.sh)   540      0       0      -      -
linclust.sh                   324      0       0      0      -
batch_clustering.sh          3675    354     525     70   ~1100
```

업스트림 워크플로는 전부 `#!/bin/sh -e` POSIX 선형 스크립트이고 함수가 없다. 모듈을 부르고
`notExists` 로 재개성을 지킬 뿐이다. 지금 파일은 bash 로 쓰인 3,675 줄짜리 응용프로그램이다.

사용자 지시(2026-09-26): **전부 한다** — awk 를 C++ 모듈로 올리고, 스크립트를 쪼개고,
POSIX 로 내리고, 가다가 보이는 결함도 고친다(각각 따로 기록해 되돌릴 수 있게).

## 순서 — 모듈 추출이 먼저다

POSIX 로 먼저 내리면 곧 지울 코드를 손보게 된다. 지우는 양이 가장 큰 것부터 간다.

### 1단계. 데이터 경로 awk → mmseqs 모듈

| 대상 셸 함수 | 줄 | 하는 일 | 갈 곳 |
|---|---|---|---|
| `propagate` `merge_join` `merge_emit_shard` `merge_join_node` `merge_emit_node` `merge_emit_bucket` `final_join_assemble` | ~250 | 해시 분할된 TSV 두 벌의 equi-join + 출력 키로 재분할 | **새 `propagateclusters`** (`mergeclusters` 를 본떠서) |
| `rebucket_manifest` `rebucket_route_split` | ~55 | 열 하나를 해시해 샤드 재배치 | `createtsv --tsv-splits` 와 같은 함수 — 모듈로 흡수 |
| `finalize_sort_split` | ~35 | 대표 우선 정렬 | 위 모듈의 출력 단계 |
| `append_singleline_fasta` | 100 | 한 줄 FASTA 이어붙이기 | `convert2fasta` 가 이미 한다 |
| `make_rep_manifest_*` | ~60 | 메트릭에서 매니페스트 조립 | 셸에 남김(작다) |

나머지 awk 30여 곳은 행 세기·숫자 파싱이라 셸로 둔다.

### 2단계. 스크립트 분리

    batch_clustering.sh        청크·클러스터·라운드·머지 (단일 노드)
    batch_clustering_slurm.sh  sbatch 체인
    batch_clustering_aws.sh    AWS Batch 체인

업스트림 선례: 워크플로 하나에 스크립트 하나.

### 3단계. POSIX 화

`#!/bin/sh -e`, `[[ ]]`→`[ ]`, `local` 제거, 배열 제거, `(( ))` 제거, 한 번만 불리는 헬퍼는 인라인.

## 불변식 (매 단계 후 확인)

1. AWS 하네스: 클린 런과 `FAULT_REPEAT=driver` 런의 최종 샤드 md5 가 같고 잡 수가 같다.
2. 2,000,000 서열 단일 노드 A/B: 리팩터 전후 `createtsv → sort → md5` 동일.
3. A/A 대조군을 함께 돌린다 — 두 바이너리 모두 20 스레드에서 0/2,000,000 재현이 확인돼 있다.

## 기준선 (리팩터 전, 2026-09-26)

    port linclust 2M, --createdb-mode 3, cm0, -c 0.8, id 0.9, adj 3
        클러스터 1,834,498   행 md5 f66752a7f2acb468
    AWS 하네스 small.fa, chunk 15000
        최종 샤드 md5 39aceb26fbfc3fa3   잡 3개

## 진행 기록

### 2026-09-26 — `propagate` → `propagateclusters` (1단계 첫 항목, 완료)

새 모듈 `src/util/propagateclusters.cpp` (160줄)가 셸의 sort/join/awk 파이프라인을 대체한다.
`Util::tsvSplitOfColumn` 은 `createtsv.cpp` 의 static 에서 commons 로 옮겨 두 명령이 한 정의를
공유한다(해시가 두 곳에 있으면 한쪽이 조용히 어긋난다 — 같은 유형의 버그를 이미 다섯 번 봤다).

지운 셸: `merge_emit_shard`(24줄), `propagate` 본문 51→30줄.
남긴 셸: `merge_join` — **다중 노드 최종 조인이 아직 쓴다.** 두 경로를 한 모듈로 합치는 게 다음 항목.

검증 (모듈이 실제로 돌았음을 로그의 `Propagated 200000 cluster member(s)` 로 확인):

```
셸 propagate 입력을 그대로 모듈에 먹여 비교   200,000 행 중 다른 행 0, 샤드 4개 각각 동일
end-to-end easy-linclust2-batch 20만 서열     최종 샤드 4개 전부 바이트 동일
                                              md5 4ea0a3080e2636c5 (리팩터 전후 같음)
AWS 하네스 (클린 / FAULT_REPEAT=driver)       md5 39aceb26fbfc3fa3, 기준선과 같음
속도                                          모듈 44 ms (셸은 sort 2회 + join + awk)
```

**함정 둘, 기록해 둔다.**
1. `data/workflow/*.sh` 는 빌드 때 바이너리에 박힌다. `.sh` 만 고치고 재빌드 없이 돌리면 옛
   스크립트가 돈다 — 한 번 속아서 "동일" 을 잘못 읽었다. 이후 `strings build/src/mmseqs | grep` 로
   새 호출이 박혔는지 먼저 확인한다.
2. 이 스크립트는 `set -u` 다(업스트림과 다르다). 없는 변수를 쓰면 즉시 죽는다.

### 다음
- `merge_join`/`merge_join_node`/`merge_emit_node`/`merge_pack_bucket` 를 모듈로 흡수(다중 노드)
- `rebucket_manifest` + `rebucket_route_split` → 같은 해시를 쓰므로 모듈로
- `finalize_sort_split` → 모듈 출력 단계로

### 리뷰 반영 (2026-09-26 06:10)

에이전트 두 명이 각각 FAIL 을 냈다(정확성 11건, 설계 20건). 고친 것:

**막힌 것이었던 3건**
- **AWS 경로가 죽어 있었다.** 옛 propagate 는 `stream_uri` 로 `s3://` 를 스트리밍했는데 모듈은
  `std::ifstream` 뿐이었다. 하네스가 통과한 건 그 입력이 한 청크에 들어가 **propagate 자체가
  안 돌았기** 때문 — "검증 입력이 밟지 않는 경로는 검증되지 않았다" 를 또 밟았다.
  → `stage_local_manifest` 로 원격 샤드만 내려받고, 여러 청크가 나오는 입력으로 다시 검증했다.
- **압축 샤드에서 하드 실패.** `--compress-batch-outputs 1` 이면 샤드가 `.tsv.zst` 인데 파서가
  거부했다. → 모듈이 zstd 를 직접 읽고 쓴다(`--compressed`).
- **메모리 무제한.** `MERGE_SPLITS = min(THREADS,256)` 에 `--threads` 를 그대로 넘겨 모든 split 이
  동시에 떠 있었다. → `createtsv` 와 같은 버퍼 예산으로 임계값 flush.

**중복 제거 (설계 리뷰 F5~F8)** — `splitName`, 버퍼 상수, lock/fwrite/clear 블록이 `createtsv` 와
겹쳤다. 해시와 같은 이유로 전부 commons 로 올렸다: `Util::tsvSplitOfKey`, `Util::tsvSplitName`,
`Util::flushSplitBuffer`, `Util::TSV_SPLIT_*`. `createtsv` 도 이걸 쓴다.

**`.rows` sidecar 삭제 (F14)** — 업스트림에 선례가 없고, split 마다 `joined == child` 를 강제하므로
전체 합 검사는 구조적으로 발동할 수 없었다. `expected_rows` 인자도 호출부 6곳에서 뺐다.

그 외: `fwrite` 반환 검사(조용한 truncation 이 보존 검사를 통과하던 구멍), `Util::isNumber` 로
split 번호 검증, `.split` 없는 줄은 조용히 버리지 않고 실패, `--tsv-splits` 는 1~99999 강제,
`COMMAND_EXPERT`, 긴 설명 `NULL`, 주석 11줄 삭제, stale 주석 2곳.

**재검증 — 세 경로 전부 같은 md5 `4ea0a3080e2636c5` (기준선과 동일)**
```
평문        easy-linclust2-batch 20만 서열, 4 splits      200,000 행
압축        --compress-batch-outputs 1                    200,000 행
AWS         하네스, 10 청크 2 라운드로 propagate 실제 실행  200,000 행
```

**안 고친 것**
- 이름 `propagateclusters` 유지. 리뷰는 `mergeclusterstsv` 를 제안했으나(blocker 아님이라 명시),
  `mergeclusters`·`pickconsensusrep` 와 같은 동사+명사 꼴이고 셸 함수명과 1:1 로 대응한다.
- 다중 노드 통합(F15), 셸 POSIX 화(F19), `PROPAGATECLUSTERS_PAR`(F20) — 다음 단계 항목.
- `EXIT()` 를 OpenMP 영역에서 부르는 것(정확성 E) — MMseqs 전반의 관용구라 단독으로 안 바꾼다.
- **선존**: 보존 검사의 상쇄 구멍(한 rep 이 초과, 다른 rep 이 부족해 합이 맞으면 통과). master
  트리에도 있다. 고치려면 매치된 rep 집합 크기도 봐야 한다 — 사용자 확인 대기.

## 계획 정정 (2026-09-26) — 2·3단계는 이 스크립트에 적용되지 않는다

계획의 2단계(3분할)와 3단계(POSIX 화)를 실제로 하려다 근거를 확인했고, **둘 다 틀렸습니다.**

**2단계(3분할)가 배포를 깬다.** `aws_submit:2584` 가 `copy_out "$0" "$script_uri"` 로
**스크립트 파일 하나를 통째로 S3 에 올리고**, 모든 컨테이너가 거기서 부트스트랩한다
(`aws-driver`/`aws-worker`/`aws-merge`). 파일을 셋으로 나누면 부트스트랩이 셋을 받아야 하고
버전이 어긋날 여지가 생긴다. 단일 파일은 취향이 아니라 배포 요건이다.

**3단계(`local` 제거)가 안전하지 않다.** `local` 이 501곳이고, 이 스크립트는 함수를 **이름으로
간접 호출**한다(`run_split_jobs` → `run_split_jobs_strided` → 콜백 이름). `local` 을 빼면
호출자와 콜백이 같은 이름의 변수를 공유한다. 업스트림 워크플로에 `local` 이 0인 이유는 함수가
0이기 때문이지, POSIX 라서가 아니다.

**근본 이유: 이 스크립트는 파이프라인이 아니라 오케스트레이터다.** 업스트림 워크플로는 모듈을
순서대로 부르고 끝난다. 이건 백엔드 3개, 재시도 예산, 배리어, 자기 재호출을 가진 프로그램이다.
업스트림 관용구(함수 없음·선형·POSIX)가 적용될 대상이 아니다.

## 남은 awk 46곳 분류 — 옮길 것은 이미 옮겼다

| 성격 | 곳 | 판단 |
|---|---|---|
| 클러스터 행 처리 | `finalize_sort_split` 2 | **남긴다.** GNU `sort` 가 `-T` 로 디스크 스필하는 외부 정렬이다. 1e12 에서 샤드 하나가 40억 행이라 in-memory 정렬로 못 바꾼다. awk 두 줄은 "자기 자신 먼저" 키를 만드는 것뿐 |
| 클러스터 행 처리 | `merge_join` 2 | 다중 노드 전용, 릴리즈 뒤 모듈로 |
| 청크 빈 패킹·파일 크기 | `prepare`/`prepare_group` 8 | **남긴다.** `aws s3api head-object`/`stat` 로 전송 계층을 부른다. C++ 로 옮기면 S3 클라이언트가 필요하다 |
| 매니페스트·카운터·du/df | 나머지 ~34 | **남긴다.** 업스트림도 셸로 쓸 글루다 |

**데이터 처리 awk 는 propagate 와 rebucket 둘이었고 둘 다 옮겼다.** 남은 건 오케스트레이션이다.

스크립트 크기를 더 줄이는 실질적 수단은 이제 모듈 추출이 아니라 **다중 노드 최종 조인 삭제
(약 200줄, 릴리즈 뒤)** 와 주석 정리다.

## `expected_rows` 를 지워도 최종 정렬은 결정적이다 (확인)

설계 리뷰는 "최종 매핑의 member(col2)가 전역 유일하므로 정렬 키가 전순서" 라고 했고, 그 유일성의
근거로 `expected_rows == input_seqs` 검사를 들었다. **그 검사를 내가 지웠으므로** 근거가 사라졌다.
다시 따져 보면 결론은 유지된다:

```
finalize_sort_split 의 정렬 키   (col1, ($1==$2 ? 0 : 1), col2)
출력                              (col1, col2)
```
flag 는 `(col1, col2)` 의 함수다. 따라서 **키가 같다 ⇒ `(col1,col2)` 가 같다 ⇒ 출력 줄이 바이트
동일**하다. 동률 두 행의 순서가 무엇이든 출력 바이트가 같으므로 `sort` 의 안정성과 무관하게
결정적이다. 유일성 가정이 필요 없다.

실측으로도 같은 md5 `4ea0a3080e2636c5` 가 평문·압축·AWS·AWS+장애주입에서 재현된다.

## `propagateclusters` 메모리 — 세 번 고쳤고, 세 번 다 리뷰/피어가 찾았다

| 무엇이 틀렸나 | 왜 틀렸나 | 실측 |
|---|---|---|
| 해시를 **child** 에 걸었다 | child 는 서열마다 1행, parent 는 대표마다 1행. 해시 조인은 작은 쪽을 올려야 한다 | 1,114 MB |
| `unordered_map` | 행당 141 B 실측(추정 104 B 보다 큼). 정렬 배열 + `lower_bound` 는 64 B 이고 짧은 accession 은 heap 0 | 619 MB |
| 예산이 빡빡하면 **패스**를 늘렸다 | 패스는 child 를 다시 읽는다 = 큰 쪽을 다시 읽는다. 1e12 에서 split 당 42패스면 21 TB 가 880 TB 가 된다. 동시성을 줄이면 child 는 1회 | 동시성 유도로 대체 |
| 출력 버퍼가 예산 밖 | `max(4 KB, 256 MB/(threads×splits))` 는 `threads×splits > 65,536` 에서 바닥이 이겨 총량이 선형 증가. 128×9,948 이면 4.86 GB(명목 256 MB) | 601 MB |

```
uniprot_32, child 7,913,508행 / parent 7,598,457행, 20 splits, 20 threads

무제한                        601 MB   동시 20개, 측정 생략
--split-memory-limit 200M     170 MB   동시  6개
--split-memory-limit 10M       16 MB   동시  1개
전부 7,913,508 행, 다른 행 0
```
기본 경로 **46% 감소**, 예산을 주면 원래의 **1/67**.

**설계로 남은 것 두 가지**

1. **동시성은 데이터에서 유도하지만 `splits` 는 아니다.** `auto_merge_splits = min(threads, 256)` 이라 1e12 에서 splits 가 너무 적으면 동시성이 1로 떨어져 직렬에 가까워진다(완주는 한다). `splits` 를 데이터에서 유도하려면 `pin_merge_splits` 가 `prepare` 보다 먼저 돌아 **그 시점에 입력 서열 수를 모르는** 문제를 풀어야 한다(매니페스트 `seqs` 열은 선언됐거나 `measure-input` 이 채운 경우만). 바이트로 유도하면 레코드 최소 크기라는 매직 상수가 필요하다. **사용자 판단 대기.**
2. **정수 키가 10.4배를 준다.** 행당 83.4 B(실측) → 8 B 면 1e12 에서 필요 splits 9,948 → 954. `MERGE_DESIGN §3.2`.

**보존 검사가 완전한 이유**(주석에 한 줄로 있음): child 행 하나가 `lower_bound` 를 한 번 하고 최대 한 행을 내므로 `joinedRows <= childRows` 가 루프 구조만으로 성립한다. 따라서 "한 rep 초과 + 다른 rep 누락" 상쇄가 불가능하다. **중복 검사와는 무관하다** — 그건 한 member 가 두 클러스터에 지명된 입력을 거부하는 별개 장치다(뒤집기 전 child 측 해시는 값이 `vector` 라 초과가 가능했고, 그때는 합계 검사가 실제로 취약했다).
