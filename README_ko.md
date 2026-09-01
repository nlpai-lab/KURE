# 🔎 KURE: Korea University Retrieval Embedding models

[English](README.md) | [한국어](README_ko.md)

**KURE**는 고려대학교 [NLP & AI 연구실](http://nlp.korea.ac.kr/)과 [HIAI 연구소](http://hiai.korea.ac.kr)가 개발한 한국어-영어 검색 임베딩 모델 시리즈입니다.

## Update Logs

- **2026.08.29**: [🤗 KURE-v2](https://huggingface.co/nlpai-lab/KURE-v2) 공개: 한국어-영어 이중 언어 **Late-Interaction (Multi-Vector)** 모델, MTEB(kor, v2) 검색 벤치마크 최고 성능. 최신 `mteb` 기반으로 리더보드 재구축.
- 2024.12.21: [🤗 KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1), MTEB-ko-retrieval 리더보드 공개
- 2024.10.02: [🤗 KoE5](https://huggingface.co/nlpai-lab/KoE5), [🤗 ko-triplet-v1.0](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0) 공개

## Models

| Model | Type | Params | Base model |
|---|---|---|---|
| [KURE-v2](https://huggingface.co/nlpai-lab/KURE-v2) | Late-interaction (multi-vector) | 154M | [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) |
| [KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) | Dense (single-vector) | 568M | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| [KoE5](https://huggingface.co/nlpai-lab/KoE5) | Dense (single-vector) | 560M | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) |

KURE-v2는 모든 토큰을 128차원 벡터로 인코딩하고 질의-문서 유사도를 MaxSim으로 계산하여, Single-Vector 모델이 압축 과정에서 잃는 토큰 수준의 의미를 보존합니다. 최대 8,192 토큰의 문서를 지원하며 별도의 instruction prefix가 필요 없습니다.

## Environment

[uv](https://docs.astral.sh/uv/)로 환경을 구성합니다.

```bash
uv sync
```

## Usage

### KURE-v2 (Late-Interaction)

<details>
<summary><b>PyLate</b></summary>

```bash
uv add pylate
```

```python
from pylate import indexes, models, retrieve

model = models.ColBERT(model_name_or_path="nlpai-lab/KURE-v2")

index = indexes.PLAID(index_folder="pylate-index", index_name="index", override=True)

documents_ids = ["1", "2", "3"]
documents = [
    "세종대왕은 1443년에 훈민정음을 창제하고 1446년에 이를 반포하였다.",
    "김치는 배추나 무를 소금에 절인 뒤 고춧가루와 젓갈을 넣어 발효시킨 음식이다.",
    "한라산은 해발 1,947m로 남한에서 가장 높은 산이며 제주도 중앙에 자리한다.",
]

documents_embeddings = model.encode(documents, is_query=False)
index.add_documents(documents_ids=documents_ids, documents_embeddings=documents_embeddings)

retriever = retrieve.ColBERT(index=index)
queries_embeddings = model.encode(["훈민정음은 언제 만들어졌나요?"], is_query=True)
scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=10)
```

</details>

<details>
<summary><b>Sentence-Transformers</b></summary>

```bash
uv add sentence-transformers
```

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("nlpai-lab/KURE-v2")

query = "훈민정음은 언제 만들어졌나요?"
documents = [
    "세종대왕은 1443년에 훈민정음을 창제하고 1446년에 이를 반포하였다.",
    "김치는 배추나 무를 소금에 절인 뒤 고춧가루와 젓갈을 넣어 발효시킨 음식이다.",
]

query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)

# MaxSim 후기 상호작용 점수 (높을수록 관련성 높음)
scores = model.similarity(query_embeddings, document_embeddings)
```

</details>

### KURE-v1 / KoE5 (Dense, Single-Vector)

<details>
<summary><b>Sentence-Transformers</b></summary>

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nlpai-lab/KURE-v1")
# model = SentenceTransformer("nlpai-lab/KoE5")  # "query: " / "passage: " prefix 필요

embeddings = model.encode(["첫 번째 문장", "두 번째 문장"])
similarities = model.similarity(embeddings, embeddings)
```

</details>

## Evaluation

최신 [mteb](https://github.com/embeddings-benchmark/mteb)(v2.x)로 **MTEB(kor, v2) Retrieval** 9개 태스크를 평가하며, 한국어 subset만 사용합니다 (Belebele: `kor_Hang-kor_Hang`; MLDR은 dev/test 평균):

- [Ko-StrategyQA](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA): 한국어 ODQA multi-hop 검색 (StrategyQA 번역)
- [AutoRAGRetrieval](https://huggingface.co/datasets/yjoonjang/markers_bm): 금융·공공·의료·법률·커머스 분야 PDF 파싱 문서 검색
- [MIRACLRetrieval](https://huggingface.co/datasets/miracl/miracl): Wikipedia 기반 검색, 약 150만 문서
- [PublicHealthQA](https://huggingface.co/datasets/xhluca/publichealth-qa): 의료·공중보건 도메인 검색
- [BelebeleRetrieval](https://huggingface.co/datasets/facebook/belebele): FLORES-200 기반 검색
- [MrTidyRetrieval](https://huggingface.co/datasets/mteb/mrtidy): Wikipedia 기반 검색, 약 150만 문서
- [MultiLongDocRetrieval](https://huggingface.co/datasets/Shitao/MLDR): 다양한 도메인의 장문 검색
- [LawIRKo](https://huggingface.co/datasets/on-and-on/lawgov_ir-ko): 한국어 법령·판례 검색
- [SQuADKorV1Retrieval](https://huggingface.co/datasets/yjoonjang/squad_kor_v1): KorQuAD v1 문단 검색

후기 상호작용 모델은 mteb + PLAID 검색으로 직접 측정하고, 밀집 모델 행은 공식 [MTEB results 저장소](https://github.com/embeddings-benchmark/results)의 점수를 사용합니다. 직접 평가하려면 모델명과 paradigm을 넘기면 됩니다:

```bash
uv run python eval/evaluate.py nlpai-lab/KURE-v2 multi-vector
uv run python eval/evaluate.py nlpai-lab/KURE-v1 single-vector
```

리더보드를 뒷받침하는 태스크별 결과 파일은 [`eval/results`](eval/results)에 있으며(출처는 해당 폴더에 문서화), 아래 표는 그 파일들로부터 재생성됩니다:

```bash
uv run python eval/make_leaderboard.py
```

### MTEB-ko-retrieval Leaderboard

9개 태스크 평균입니다. 태스크별 상세 결과는 공식 [MTEB 리더보드](https://mteb-leaderboard.hf.space/benchmark/MTEB(kor%2C%20v2)?types=Retrieval&s.summary=meanTask&d.summary=desc)에서 확인할 수 있습니다.

| Model | Type | Params | Avg nDCG@10 | Avg Recall@10 |
|---|---|---:|---:|---:|
| **[nlpai-lab/KURE-v2](https://huggingface.co/nlpai-lab/KURE-v2)** | Late-interaction | 154M | **0.8160** | **0.8921** |
| yjoonjang/colbert-ko-en-v2 | Late-interaction | 149M | 0.8063 | 0.8827 |
| sionic-ai/comsat-embed-ko-8b-preview | Dense | 7.6B | 0.7927 | 0.8876 |
| lightonai/mLateOn | Late-interaction | 307M | 0.7906 | 0.8740 |
| Qwen/Qwen3-Embedding-8B | Dense | 7.6B | 0.7826 | 0.8815 |
| Qwen/Qwen3-Embedding-4B | Dense | 4.0B | 0.7737 | 0.8753 |
| microsoft/harrier-oss-v1-27b | Dense | 27.0B | 0.7667 | 0.8624 |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko | Dense | 568M | 0.7653 | 0.8609 |
| codefuse-ai/F2LLM-v2-8B | Dense | 7.6B | 0.7638 | 0.8593 |
| telepix/PIXIE-Rune-v1.5 | Dense | 568M | 0.7618 | 0.8596 |
| **[nlpai-lab/KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1)** | Dense | 568M | 0.7616 | 0.8629 |
| dragonkue/BGE-m3-ko | Dense | 568M | 0.7547 | 0.8513 |
| BAAI/bge-m3 | Dense | 568M | 0.7509 | 0.8588 |
| perplexity-ai/pplx-embed-v1-late-0.6b | Late-interaction | 596M | 0.7381 | 0.8328 |
| **[nlpai-lab/KoE5](https://huggingface.co/nlpai-lab/KoE5)** | Dense | 560M | 0.7337 | 0.8300 |
| **[nlpai-lab/KURE-v2-unsupervised](https://huggingface.co/nlpai-lab/KURE-v2-unsupervised)** | Late-interaction | 154M | 0.7283 | 0.8268 |
| dragonkue/colbert-ko-0.1b | Late-interaction | 149M | 0.6776 | 0.7723 |
| yjoonjang/colbert-ko-v1 | Late-interaction | 149M | 0.6282 | 0.7212 |

## Training Details

### KURE-v2

전체 2단계 학습 코드는 [`train/late-interaction`](train/late-interaction)에 있습니다.

- [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) 위에 다층 투영 헤드를 얹은 구조(토큰당 128차원, MaxSim 점수)이며 [PyLate](https://github.com/lightonai/pylate)로 학습했습니다.
- **1단계 (PFT)**: 약한 관련성의 한국어·영어 2,070만 쌍에 대한 거대 배치 대조 학습. 이 단계만 거친 모델을 [KURE-v2-unsupervised](https://huggingface.co/nlpai-lab/KURE-v2-unsupervised)로 공개합니다.
- **2단계 (SFT)**: 하드 네거티브와 위음성 필터링을 적용한 삼중항 303만 개에 대해 대조 학습과 리랭커 교사 점수의 KL 증류를 결합합니다.

### KURE-v1

학습 코드는 [`train/dense`](train/dense)에 있습니다.

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)를 한국어 query-document-hard_negative(5개) 약 200만 쌍으로 fine-tuning.
- CachedGISTEmbedLoss, 배치 4,096, 학습률 2e-5, 1 에폭.

### KoE5

- [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)를 [ko-triplet-v1.0](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0)(약 70만+ 쌍)으로 fine-tuning.
- CachedMultipleNegativesRankingLoss, 배치 512, 학습률 1e-5, 1 에폭. 사용 시 `query: ` / `passage: ` prefix가 필요합니다.

## Serving

KURE-v2는 후기 상호작용 모델로, 문서를 토큰 벡터의 집합으로 저장하기 때문에 배포에서는 색인 크기와 검색 비용이 실질적인 관건입니다. 9개 한국어 MTEB 검색 태스크에서 KURE-v2를 여러 ANN 백엔드·압축 기법으로, [faiss HNSW](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSW.html)로 서빙되는 단일 벡터 5종과 함께 벤치마크했습니다. 모든 수치는 종단간(배치 1 질의 인코딩 + 색인 검색)이며 A100 80GB 1대에서 직렬로 측정했습니다. 아래 구성들을 직접 실행해볼 수 있는 코드는 [`deploy/late-interaction`](deploy/late-interaction)에 있습니다 (`uv run deploy/late-interaction/run.py --index plaid`).

<p align="center">
  <img src="assets/deploy_overview.png" width="100%" alt="색인 크기 대비(좌) / 종단간 QPS 대비(우) 평균 nDCG@10">
</p>

그림이 보여주는 두 가지:

- Hierarchical Token Pooling(x2)은 0.04의 nDCG 하락으로 색인을 절반으로 줄입니다. Asymmetric Binary Quantization (문서 토큰 1-bit, 질의 bf16)는 1.05 하락으로 9.4배 축소합니다. 둘을 결합하면(pooling x3 + Asym. Binary) 9개 코퍼스 전체 색인이 **1.7 GB로, 모든 Single-Vector HNSW 색인(13.1-50.0 GB)보다 작으면서도** 최고 Single-Vector 모델보다 높은 품질(79.57 대 79.07)을 유지합니다.
- 실제 query는 텍스트로 도착합니다: 4B-8B Single-Vector 모델은 query 인코딩에만 38-40 ms를 써서 색인이 아무리 빨라도 약 25 QPS에 고정됩니다. KURE-v2는 13.8 ms(154M)에 인코딩하므로 MUVERA를 제외한 모든 구성이 **43-55 QPS로, 8B Single-Vector 모델의 약 2배 처리량을 더 높은 품질로** 제공합니다.

### 대규모 문서 집합에서의 성능, 지연, 인덱스

<p align="center">
  <img src="assets/bigcorpus_miracl.png" width="70%" alt="MIRACL(150만 문서): 품질, 종단간 p95 지연, 색인 크기">
</p>

가장 큰 코퍼스(MIRACL, 약 150만 문서)에서 전수 1-bit 스캔의 비용은 문서 수에 비례합니다: p95가 156 ms까지 오르고, 토큰을 3배 줄여도 74 ms에 그칩니다. 같은 1-bit 색인 위에서 faiss [BinaryIVF](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexBinaryIVF.html) (해밍 검색)로 후보를 생성하고 정확한 비대칭 MaxSim으로 재순위화하면 **같은 2.2 GB 색인에서 p95가 38 ms로, 4B-8B 단일 벡터(43 ms)보다 짧은 꼬리 지연을 더 높은 nDCG로** 달성합니다. 대규모 컬렉션에서는 전수 스캔이 아니라 후보 생성형 색인(PLAID 또는 BinaryIVF)을 사용하세요.

<details>
<summary><b>측정 상세</b></summary>

- **하드웨어**: NVIDIA A100 80GB 1대, AMD EPYC 7513 2개(64코어), RAM 1.2 TB.
- **소프트웨어**: faiss-cpu 1.15.0, fast-plaid 1.6.0, sentence-transformers 6.0.0, PyTorch 2.8.0.
- **프로토콜**: 배치 1, 직렬. 색인 검색 지연: 워밍업 10회 후 태스크 전체 질의를 각 1회 측정(QPS = 평균 지연의 역수). 질의 인코딩 지연: 워밍업 5회 후 50회 측정. 종단간 = 인코딩 + 검색.
- **정밀도**: 인코딩은 bf16; 색인은 방법별 저장 형식(HNSW fp32, PLAID 4-bit 잔차, 이진화 1-bit).
- **색인 크기**: 디스크에 직렬화된 색인 전체(벡터·그래프·코드북 포함, 외부 문서 ID 매핑 제외).
- **태스크**: 9개 한국어 MTEB 검색 태스크; MLDR은 dev/test 평균; nDCG@10 x100.
- **HNSW**: `IndexHNSWFlat`(정규화 임베딩의 내적), M=32, efConstruction=200, efSearch=64.
- **PLAID**: nbits=4, 나머지는 fast-plaid 기본값(kmeans_niters=4, n_ivf_probe=8, n_full_scores=4096). nbits=2/1은 27.0/17.0 GB, 81.25/81.09 nDCG.
- **MUVERA**: num_repetitions=10, num_simhash_projections=6, final_projection_dimension=8192, 상위 1,000개 정확 MaxSim 재순위화.
- **BinaryIVF**: nlist=floor(sqrt(총 토큰 수)), 상한 65,536, nprobe=32, 질의 토큰당 상위 128개 해밍 토큰, 상위 1,000개 문서 정확 비대칭 MaxSim 재순위화.
- **토큰 풀링**: 계층적(Ward linkage), pool_factor 2-3, 문서에만 적용.
</details>

## License

```MIT```

## Citation

모델이 도움이 되었다면 다음을 인용해 주세요:

```bibtex
@misc{kure-v2,
  title  = {KURE-v2: a Korean-English bilingual late-interaction retriever},
  author = {Youngjoon Jang, Junyoung Son, Taemin Lee, Seongtae Hong, Chanjun Park, Heuiseok Lim},
  year   = {2026},
  url    = {https://huggingface.co/nlpai-lab/KURE-v2},
}
```

```bibtex
@inproceedings{jang2025kure,
  title={KURE: Embedding Model for Korean-Specific Retrieval},
  author={Jang, Youngjoon and Son, Junyoung and Lee, Taemin and Hong, Seongtae and Park, JeongBae and Lim, Heuiseok},
  booktitle={Annual Conference on Human and Language Technology},
  pages={129--134},
  year={2025},
  organization={Human and Language Technology}
}
```

```bibtex
@inproceedings{jang2024koe5,
  title={KoE5: A New Dataset and Model for Improving Korean Embedding Performance},
  author={Jang, Youngjoon and Son, Junyoung and Park, Chanjun and Choi, Soonwoo and Lee, Byeonggoo and Lee, Taemin and Lim, Heuiseok},
  booktitle={Annual Conference on Human and Language Technology},
  pages={239--244},
  year={2024},
  organization={Human and Language Technology}
}
```
