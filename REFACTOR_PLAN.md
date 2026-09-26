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
