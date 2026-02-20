# KURE Evaluation System Guide

한국어 임베딩 모델 평가 시스템 가이드

## 목차
- [빠른 시작](#빠른-시작)
- [여러 모델 평가하기](#여러-모델-평가하기)
- [프롬프트 자동 적용](#프롬프트-자동-적용)
- [커스텀 프롬프트 설정](#커스텀-프롬프트-설정)
- [MRL (Matryoshka) 지원](#mrl-matryoshka-지원)
- [bf16 & Flash Attention](#bf16--flash-attention)
- [GPU 분배 및 큐 시스템](#gpu-분배-및-큐-시스템)
- [평가 태스크](#평가-태스크)
- [결과 구조](#결과-구조)
- [문제 해결](#문제-해결)

---

## 빠른 시작

```bash
# 단일 모델 평가 (모든 태스크, 모든 GPU)
python eval/evaluate.py --model BAAI/bge-m3

# 특정 태스크만 평가
python eval/evaluate.py --model BAAI/bge-m3 --tasks BelebeleRetrieval NanoFEVERKo

# 특정 GPU에서만 평가
python eval/evaluate.py --model BAAI/bge-m3 --gpu 0

# 사용 가능한 태스크 목록
python eval/evaluate.py --list_tasks

# 설정된 모델 목록
python eval/evaluate.py --list_models
```

---

## 여러 모델 평가하기

### 방법 1: 쉘 스크립트 (권장)

```bash
#!/bin/bash
# evaluate_models.sh

MODELS=(
    "BAAI/bge-m3"
    "nlpai-lab/KoE5"
    "jinaai/jina-embeddings-v3"
    "Qwen/Qwen3-Embedding-0.6B"
)

for model in "${MODELS[@]}"; do
    echo "========== Evaluating: $model =========="
    python eval/evaluate.py --model "$model"
    echo "========== Completed: $model =========="
done
```

### 방법 2: Python 스크립트

```python
# evaluate_multiple.py
import subprocess

models = [
    "BAAI/bge-m3",
    "nlpai-lab/KoE5",
    "jinaai/jina-embeddings-v3",
    "Qwen/Qwen3-Embedding-0.6B",
]

for model in models:
    print(f"Evaluating: {model}")
    subprocess.run([
        "python", "eval/evaluate.py",
        "--model", model,
    ])
```

### 방법 3: 병렬 실행 (여러 GPU로 서로 다른 모델)

```bash
# GPU 0에서 bge-m3 평가 (백그라운드)
python eval/evaluate.py --model BAAI/bge-m3 --gpu 0 &

# GPU 1에서 KoE5 평가 (백그라운드)
python eval/evaluate.py --model nlpai-lab/KoE5 --gpu 1 &

# GPU 2에서 jina 평가 (백그라운드)
python eval/evaluate.py --model jinaai/jina-embeddings-v3 --gpu 2 &

wait  # 모든 프로세스 완료 대기
```

---

## 프롬프트 자동 적용

### MTEB 등록 모델 (자동 처리)

MTEB에 등록된 모델은 프롬프트가 **자동으로 적용**됩니다:

```python
# 내부적으로 mteb.get_model()이 호출되면 자동 처리
model = mteb.get_model("BAAI/bge-m3", device=device)
# → 프롬프트가 모델 config에서 자동 로드됨
```

**자동 처리되는 모델 예시:**
- `BAAI/bge-m3` - 프롬프트 자동
- `intfloat/multilingual-e5-large-instruct` - instruction 자동
- `Alibaba-NLP/gte-Qwen2-7B-instruct` - instruction 자동

### 시스템에 등록된 커스텀 프롬프트

MTEB에 등록되지 않은 모델은 `eval/models/config.py`에서 관리합니다:

```python
# eval/models/config.py에 이미 등록된 모델들

"nlpai-lab/KoE5": ModelConfig(
    name="nlpai-lab/KoE5",
    custom_prompts={
        PromptType.query.value: "query: ",      # 쿼리 앞에 붙음
        PromptType.document.value: "passage: ", # 문서 앞에 붙음
    },
),

"Snowflake/snowflake-arctic-embed-l-v2.0": ModelConfig(
    custom_prompts={
        PromptType.query.value: "query: ",  # 쿼리만 프롬프트
    },
),

"FronyAI/frony-embed-medium-arctic-ko-v2.5": ModelConfig(
    custom_prompts={
        PromptType.query.value: "<Q>",
        PromptType.document.value: "<P>",
    },
),
```

---

## 커스텀 프롬프트 설정

### 방법 1: config.py에 모델 추가 (권장)

새 모델의 프롬프트를 영구적으로 설정하려면 `eval/models/config.py`를 수정합니다:

```python
# eval/models/config.py

from mteb.types import PromptType

MODEL_CONFIGS: dict[str, ModelConfig] = {
    # ... 기존 모델들 ...

    # 새 모델 추가
    "my-org/my-custom-model": ModelConfig(
        name="my-org/my-custom-model",
        batch_size=64,                    # 배치 사이즈
        supports_bf16=True,               # bf16 지원
        supports_flash_attn=False,        # BERT 계열은 False
        custom_prompts={
            PromptType.query.value: "질문: ",      # 쿼리 프롬프트
            PromptType.document.value: "문서: ",   # 문서 프롬프트
        },
    ),
}
```

### 방법 2: 코드에서 직접 래핑

일회성으로 프롬프트를 적용하려면:

```python
from sentence_transformers import SentenceTransformer
from mteb import SentenceTransformerEncoderWrapper
from mteb.types import PromptType

# 모델 로드
model = SentenceTransformer("my-org/my-model", device="cuda:0")

# 프롬프트 래핑
wrapped = SentenceTransformerEncoderWrapper(
    model=model,
    model_prompts={
        PromptType.query.value: "query: ",
        PromptType.document.value: "passage: ",
    },
)

# MTEB 평가 실행
from mteb import MTEB
evaluation = MTEB(tasks=["BelebeleRetrieval"])
evaluation.run(wrapped, output_folder="results/my-model")
```

### 프롬프트 타입 정리

| PromptType | 용도 | 예시 |
|------------|------|------|
| `query` | 검색 쿼리 | `"query: "`, `"질문: "` |
| `document` | 문서/패시지 | `"passage: "`, `"문서: "` |

**Retrieval 태스크 흐름:**
```
Query: "서울의 인구는?"
  → "query: 서울의 인구는?"  (query 프롬프트 적용)

Document: "서울특별시는 대한민국의 수도로..."
  → "passage: 서울특별시는 대한민국의 수도로..."  (document 프롬프트 적용)
```

---

## MRL (Matryoshka) 지원

Matryoshka Representation Learning을 지원하는 모델은 차원 축소가 가능합니다.

### 지원 모델

| 모델 | 기본 차원 | 지원 차원 |
|------|----------|----------|
| `jinaai/jina-embeddings-v3` | 1024 | 32, 64, 128, 256, 512, 768, 1024 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 64, 128, 256, 512, 768 |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | 1024 | 256, 512, 768, 1024 |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | 128, 256, 512, 768, 1024 |
| `Qwen/Qwen3-Embedding-4B` | 2560 | 128, 256, 512, 1024, 1536, 2048, 2560 |
| `Qwen/Qwen3-Embedding-8B` | 4096 | 128, 256, 512, 1024, 2048, 3072, 4096 |

### 사용법

```bash
# 256 차원으로 평가
python eval/evaluate.py --model jinaai/jina-embeddings-v3 --dim 256

# 결과 저장 위치: eval/results/jinaai_jina-embeddings-v3_dim256/
```

### 여러 차원 비교 평가

```bash
#!/bin/bash
MODEL="jinaai/jina-embeddings-v3"

for dim in 128 256 512 768 1024; do
    python eval/evaluate.py --model "$MODEL" --dim $dim
done
```

---

## bf16 & Flash Attention

### 자동 최적화

모델 아키텍처에 따라 자동으로 최적화가 적용됩니다:

| 아키텍처 | bf16 | Flash Attention |
|----------|------|-----------------|
| **Qwen** (GTE-Qwen, Qwen3-Embedding) | ✅ | ✅ |
| **Mistral** (e5-mistral) | ✅ | ✅ |
| **Gemma** (bge-gemma2) | ✅ | ✅ |
| **LLaMA** 계열 | ✅ | ✅ |
| **BERT/RoBERTa** (mE5, KoE5, bge-m3) | ✅ | ❌ (SDPA 사용) |

### 수동 비활성화

```bash
# bf16 비활성화
python eval/evaluate.py --model BAAI/bge-m3 --no_bf16

# Flash Attention 비활성화
python eval/evaluate.py --model Qwen/Qwen3-Embedding-0.6B --no_flash_attn

# 둘 다 비활성화
python eval/evaluate.py --model my-model --no_bf16 --no_flash_attn
```

---

## GPU 분배 및 큐 시스템

### 큐 기반 평가 방식

```
GPU 0: Task1 → Task2 → Task3 → ... (순차)
GPU 1: Task1 → Task2 → Task3 → ... (순차)  ← 병렬 실행
GPU 2: Task1 → Task2 → Task3 → ... (순차)
```

- **모델 1회 로딩**: 각 GPU에서 모델을 한 번만 로드
- **태스크 순차 처리**: 큐에서 하나씩 꺼내서 평가
- **GPU 간 병렬**: 여러 GPU가 동시에 각자의 큐 처리

### 기본 GPU 분배

```
GPU 0 (9 tasks): BelebeleRetrieval, XPQARetrieval, Ko-StrategyQA,
                 AutoRAGRetrieval, PublicHealthQA, NanoArguAnaKo,
                 NanoClimateFEVERKo, NanoDBPediaKo, NanoFEVERKo

GPU 1 (6 tasks): MIRACLRetrieval, NanoFiQA2018Ko, NanoHotpotQAKo,
                 NanoMSMARCOKo, NanoNFCorpusKo, NanoNQKo

GPU 2 (6 tasks): MrTidyRetrieval, MultiLongDocRetrieval, NanoQuoraRetrievalKo,
                 NanoSCIDOCSKo, NanoSciFactKo, NanoTouche2020Ko
```

### GPU 분배 커스터마이징

`eval/evaluate.py`의 `TASK_GPU_MAPPING`을 수정:

```python
TASK_GPU_MAPPING = {
    0: ["BelebeleRetrieval", "XPQARetrieval"],
    1: ["MIRACLRetrieval", "MrTidyRetrieval"],
    2: ["NanoFEVERKo", "NanoMSMARCOKo"],
    3: ["MultiLongDocRetrieval"],  # GPU 추가 가능
}
```

### 완료된 태스크 스킵

기본적으로 이미 결과 파일이 있는 태스크는 스킵됩니다:

```bash
# 기본: 완료된 태스크 스킵
python eval/evaluate.py --model BAAI/bge-m3

# 강제 재평가 (스킵 안 함)
python eval/evaluate.py --model BAAI/bge-m3 --no_skip
```

---

## 평가 태스크

### MTEB Korean Retrieval (8개)

| 태스크 | 설명 |
|--------|------|
| `BelebeleRetrieval` | 다국어 독해 기반 검색 |
| `XPQARetrieval` | 교차 언어 QA 검색 |
| `MultiLongDocRetrieval` | 긴 문서 검색 |
| `Ko-StrategyQA` | 전략적 질의응답 |
| `AutoRAGRetrieval` | 자동 RAG 벤치마크 |
| `PublicHealthQA` | 공중보건 QA |
| `MIRACLRetrieval` | 대규모 다국어 검색 |
| `MrTidyRetrieval` | 한국어 검색 벤치마크 |

### NanoBEIR-ko (13개)

| 태스크 | 원본 BEIR 데이터셋 |
|--------|-------------------|
| `NanoArguAnaKo` | ArguAna |
| `NanoClimateFEVERKo` | Climate-FEVER |
| `NanoDBPediaKo` | DBPedia |
| `NanoFEVERKo` | FEVER |
| `NanoFiQA2018Ko` | FiQA-2018 |
| `NanoHotpotQAKo` | HotpotQA |
| `NanoMSMARCOKo` | MS MARCO |
| `NanoNFCorpusKo` | NFCorpus |
| `NanoNQKo` | Natural Questions |
| `NanoQuoraRetrievalKo` | Quora |
| `NanoSCIDOCSKo` | SCIDOCS |
| `NanoSciFactKo` | SciFact |
| `NanoTouche2020Ko` | Touché-2020 |

---

## 결과 구조

### 디렉토리 구조

```
eval/results/
├── BAAI_bge-m3/                          # 기본 차원
│   ├── BelebeleRetrieval.json
│   ├── NanoFEVERKo.json
│   └── ...
├── jinaai_jina-embeddings-v3/            # 기본 차원 (1024)
│   └── ...
├── jinaai_jina-embeddings-v3_dim256/     # MRL 256 차원
│   └── ...
└── jinaai_jina-embeddings-v3_dim512/     # MRL 512 차원
    └── ...
```

### 결과 파일 형식

```json
{
  "dataset_revision": "...",
  "evaluation_time": 45.2,
  "kg_co2_emissions": null,
  "mteb_version": "2.7.30",
  "scores": {
    "test": [
      {
        "hf_subset": "default",
        "languages": ["kor-Hang"],
        "main_score": 0.7234,
        "map_at_1": 0.6521,
        "map_at_10": 0.7102,
        "mrr_at_1": 0.6521,
        "mrr_at_10": 0.7102,
        "ndcg_at_1": 0.6521,
        "ndcg_at_10": 0.7234,
        "ndcg_at_100": 0.7456,
        ...
      }
    ]
  }
}
```

---

## 문제 해결

### Flash Attention 오류

```
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
```

**해결:** `--no_flash_attn` 플래그 사용

```bash
python eval/evaluate.py --model my-model --no_flash_attn
```

### CUDA Out of Memory

**해결 1:** 배치 사이즈 줄이기

```bash
python eval/evaluate.py --model my-model --batch_size 8
```

**해결 2:** `config.py`에서 모델별 배치 사이즈 조정

```python
"my-org/my-large-model": ModelConfig(
    name="my-org/my-large-model",
    batch_size=4,  # 더 작은 배치
),
```

### 프롬프트가 적용되지 않음

1. `--list_models`로 모델이 등록되어 있는지 확인
2. 등록되지 않은 경우 `config.py`에 추가
3. MTEB 등록 모델인 경우 자동 처리됨

### 태스크 실패 후 재시작

실패한 태스크만 다시 실행됩니다 (자동 스킵):

```bash
# 이미 완료된 태스크는 스킵하고 실패한 것만 재시도
python eval/evaluate.py --model BAAI/bge-m3
```

### 특정 태스크만 재평가

```bash
# 특정 태스크 강제 재평가
python eval/evaluate.py --model BAAI/bge-m3 --tasks NanoFEVERKo --no_skip
```

---

## CLI 옵션 요약

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 평가할 모델 (필수) | - |
| `--tasks` | 평가할 태스크들 | 전체 |
| `--gpu` | 사용할 GPU ID | 전체 GPU |
| `--dim` | MRL 차원 | 기본 차원 |
| `--batch_size` | 배치 사이즈 오버라이드 | 모델별 설정 |
| `--output_dir` | 결과 저장 경로 | `eval/results` |
| `--no_bf16` | bf16 비활성화 | False |
| `--no_flash_attn` | Flash Attention 비활성화 | False |
| `--no_skip` | 완료된 태스크 재평가 | False |
| `--list_tasks` | 태스크 목록 출력 | - |
| `--list_models` | 모델 설정 출력 | - |
