# 🔎 KURE: Korea University Retrieval Embedding models

<p align="center" width="100%">
<img src="assets/kure_logo.png" alt="KURE Logo" style="width: 50%;">
</p>

[English](README.md) | [한국어](README_ko.md)

**KURE** is a family of Korean-English retrieval embedding models developed by the [NLP & AI Lab](http://nlp.korea.ac.kr/) and the [HIAI Institute](http://hiai.korea.ac.kr) at Korea University.

## Update Logs

- 2026.10.01: [🤗 KURE-Reranker-base](https://huggingface.co/nlpai-lab/KURE-Reranker-base) and [🤗 KURE-Reranker-nano](https://huggingface.co/nlpai-lab/KURE-Reranker-nano) released.
- 2026.08.29: [🤗 KURE-v2](https://huggingface.co/nlpai-lab/KURE-v2) released.
- 2024.12.21: [🤗 KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) released with the MTEB-ko-retrieval leaderboard.
- 2024.10.02: [🤗 KoE5](https://huggingface.co/nlpai-lab/KoE5) and [🤗 ko-triplet-v1.0](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0) released.

## Models

| Model | Type | Params | Base model |
|---|---|---|---|
| [KURE-Reranker-base](https://huggingface.co/nlpai-lab/KURE-Reranker-base) | Reranker (cross-encoder) | 1.7B | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| [KURE-Reranker-nano](https://huggingface.co/nlpai-lab/KURE-Reranker-nano) | Reranker (cross-encoder) | 149M | [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) |
| [KURE-v2](https://huggingface.co/nlpai-lab/KURE-v2) | Late-interaction (multi-vector) | 154M | [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) |
| [KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) | Dense (single-vector) | 568M | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| [KoE5](https://huggingface.co/nlpai-lab/KoE5) | Dense (single-vector) | 560M | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) |


## Environment

We use [uv](https://docs.astral.sh/uv/) to manage the environment.

```bash
uv sync                                                # dense, sparse and cross-encoder models
uv sync --extra late-interaction --no-default-groups   # KURE-v2 with PyLate
```

PyLate pins `sentence-transformers==5.3.0`, while the KURE-Reranker models need `>=5.6.1`, so the late-interaction stack installs into its own environment.

## Usage

### KURE-Reranker

<details>
<summary><b>Sentence-Transformers</b></summary>

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("nlpai-lab/KURE-Reranker-nano", max_length=8192)
# model = CrossEncoder("nlpai-lab/KURE-Reranker-base", max_length=8192)

query = "훈민정음은 언제 만들어졌나요?"
documents = [
    "세종대왕은 1443년에 훈민정음을 창제하고 1446년에 이를 반포하였다.",
    "김치는 배추나 무를 소금에 절인 뒤 고춧가루와 젓갈을 넣어 발효시킨 음식이다.",
    "한라산은 해발 1,947m로 남한에서 가장 높은 산이며 제주도 중앙에 자리한다.",
]

# Higher is more relevant
results = model.rank(query, documents, return_documents=True)
```

</details>

### KURE-v2

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

# MaxSim late-interaction scoring (higher is more relevant)
scores = model.similarity(query_embeddings, document_embeddings)
```

</details>

### KURE-v1 / KoE5

<details>
<summary><b>Sentence-Transformers</b></summary>

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nlpai-lab/KURE-v1")
# model = SentenceTransformer("nlpai-lab/KoE5")  # needs "query: " / "passage: " prefixes

embeddings = model.encode(["첫 번째 문장", "두 번째 문장"])
similarities = model.similarity(embeddings, embeddings)
```

</details>

## Evaluation

We evaluate on the nine **MTEB(kor, v2) Retrieval** tasks with the latest [mteb](https://github.com/embeddings-benchmark/mteb) (v2.x), using the Korean subsets only (Belebele: `kor_Hang-kor_Hang`; MLDR is the mean of its dev/test splits):

- [Ko-StrategyQA](https://huggingface.co/datasets/taeminlee/Ko-StrategyQA): Korean ODQA multi-hop retrieval (translated StrategyQA)
- [AutoRAGRetrieval](https://huggingface.co/datasets/yjoonjang/markers_bm): document retrieval over parsed PDFs in finance, public sector, healthcare, law, and commerce
- [MIRACLRetrieval](https://huggingface.co/datasets/miracl/miracl): Wikipedia-based retrieval, ~1.5M documents
- [PublicHealthQA](https://huggingface.co/datasets/xhluca/publichealth-qa): medical / public-health domain retrieval
- [BelebeleRetrieval](https://huggingface.co/datasets/facebook/belebele): FLORES-200-based retrieval
- [MrTidyRetrieval](https://huggingface.co/datasets/mteb/mrtidy): Wikipedia-based retrieval, ~1.5M documents
- [MultiLongDocRetrieval](https://huggingface.co/datasets/Shitao/MLDR): long-document retrieval across domains
- [LawIRKo](https://huggingface.co/datasets/on-and-on/lawgov_ir-ko): Korean statute / case-law retrieval
- [SQuADKorV1Retrieval](https://huggingface.co/datasets/yjoonjang/squad_kor_v1): KorQuAD v1 passage retrieval

Late-interaction models are measured directly with mteb + PLAID retrieval; dense rows come from the official [MTEB results repository](https://github.com/embeddings-benchmark/results). To evaluate a model yourself, pass its name and paradigm:

```bash
uv run --extra late-interaction --no-default-groups python eval/dense/evaluate.py nlpai-lab/KURE-v2 multi-vector
uv run python eval/dense/evaluate.py nlpai-lab/KURE-v1 single-vector
```

The per-task result files backing the leaderboard are in [`eval/dense/results`](eval/dense/results), and the table below is regenerated from them:

```bash
uv run python eval/dense/make_leaderboard.py
```

### MTEB-ko-retrieval Leaderboard

Average over the nine tasks. Full per-task results are on the official [MTEB Leaderboard](https://mteb-leaderboard.hf.space/benchmark/MTEB(kor%2C%20v2)?types=Retrieval&s.summary=meanTask&d.summary=desc).

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

### Reranker Leaderboard

Rerankers are evaluated on the same nine tasks with the protocol of [reranker-simple-benchmark](https://github.com/instructkr/reranker-simple-benchmark): each query's candidates are all of its gold documents plus the BM25 top-50, every model runs in bf16 at up to 8,192 tokens, and we report nDCG@10 with throughput in pairs per second (PPS), measured on one NVIDIA RTX A6000 48GB. To evaluate a reranker yourself:

```bash
bash eval/cross-encoder/fetch_eval_assets.sh   # once: the BM25 candidate pools
uv run python eval/cross-encoder/evaluate.py nlpai-lab/KURE-Reranker-nano --speed
```

The per-task result files are in [`eval/cross-encoder/results`](eval/cross-encoder/results), and the table below is regenerated from them:

```bash
uv run python eval/cross-encoder/make_leaderboard.py
```

Mean over the nine tasks. Per-task nDCG@10 and PPS are in the [KURE-Reranker-base model card](https://huggingface.co/nlpai-lab/KURE-Reranker-base#korean-reranking-evaluation).

<p align="center">
  <img src="assets/reranker_pps_vs_ndcg.png" width="80%" alt="Mean nDCG@10 vs. mean PPS of rerankers on the nine Korean tasks">
</p>

| Model | Params | Mean nDCG@10 | Mean PPS |
|---|---:|---:|---:|
| Qwen/Qwen3-Reranker-8B | 8.2B | 0.9030 | 15.2 |
| KaLM-Embedding/KaLM-Reranker-V1-Large-R2 | 7.5B | 0.8960 | 25.3 |
| Qwen/Qwen3-Reranker-4B | 4.0B | 0.8957 | 24.3 |
| **[nlpai-lab/KURE-Reranker-base](https://huggingface.co/nlpai-lab/KURE-Reranker-base)** | 1.7B | **0.8849** | 55.1 |
| **[nlpai-lab/KURE-Reranker-nano](https://huggingface.co/nlpai-lab/KURE-Reranker-nano)** | 149M | **0.8808** | **473.7** |
| zeroentropy/zerank-2-reranker | 4.0B | 0.8695 | 29.4 |
| lightonai/LightOn-rerank-PW-4B | 4.5B | 0.8664 | 15.0 |
| mixedbread-ai/mxbai-rerank-large-v2 | 1.5B | 0.8661 | 65.2 |
| BAAI/bge-reranker-v2-m3 | 568M | 0.8586 | 404.1 |
| Qwen/Qwen3-Reranker-0.6B | 596M | 0.8577 | 100.2 |
| nvidia/llama-nemotron-rerank-1b-v2 | 1.2B | 0.8522 | 127.3 |
| nlpai-lab/LAMAR-600m | 568M | 0.8406 | 408.8 |
| dragonkue/bge-reranker-v2-m3-ko | 568M | 0.8263 | 401.5 |
| BAAI/bge-reranker-v2-gemma | 2.5B | 0.8186 | 58.7 |
| upskyy/ko-reranker-8k | 568M | 0.8085 | 404.0 |
| Dongjin-kr/ko-reranker | 560M | 0.7950 | 482.8 |
| telepix/PIXIE-Spell-Reranker-Preview-0.6B | 596M | 0.7806 | 99.6 |

## Training Details

### KURE-Reranker

Training code in [`train/cross-encoder`](train/cross-encoder).

- Distilled from a Qwen3-Reranker teacher with pointwise MSE on the KURE-v2 SFT data flattened to 33.3M query-document pairs, 1 epoch.
- **KURE-Reranker-base**: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) scored as logit("yes") − logit("no") over the Qwen3-Reranker template; up to 8,192 tokens.
- **KURE-Reranker-nano**: [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) with a single-logit head; up to 8,192 tokens.

### KURE-v2

The full two-stage training code is in [`train/late-interaction`](train/late-interaction).

- Built on [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) with a multi-layer projection head (128-d per token, MaxSim scoring); trained with [PyLate](https://github.com/lightonai/pylate).
- **Stage 1 (PFT)**: large in-batch contrastive learning on 20.7M weakly related Korean/English pairs, released as [KURE-v2-unsupervised](https://huggingface.co/nlpai-lab/KURE-v2-unsupervised).
- **Stage 2 (SFT)**: contrastive learning + KL distillation from a reranker teacher on 3.03M triplets with hard negatives and false-negative filtering.

### KURE-v1

Training code in [`train/dense`](train/dense).

- Fine-tuned from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) on ~2M Korean query-document-hard-negative(5) pairs.
- CachedGISTEmbedLoss, batch size 4,096, lr 2e-5, 1 epoch.

### KoE5

- Fine-tuned from [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) on [ko-triplet-v1.0](https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0) (~700K+ examples).
- CachedMultipleNegativesRankingLoss, batch size 512, lr 1e-5, 1 epoch. Requires `query: ` / `passage: ` prefixes.

## Serving

Index-size and latency benchmarks of KURE-v2 across ANN backends, token pooling and binary quantization, with runnable configurations, are in [`deploy/late-interaction`](deploy/late-interaction).

## License

```MIT```

## Citation

If you find our models helpful, please consider citing:

```bibtex
@misc{kure-reranker,
  title  = {KURE-Reranker: Korean-English bilingual reranking models},
  author = {Jang, Youngjoon and Hong, Seongtae and Son, Junyoung and Lee, Taemin and Lim, Heuiseok},
  year   = {2026},
  url    = {https://huggingface.co/nlpai-lab/KURE-Reranker-base},
}
```

```bibtex
@misc{kure-v2,
  title  = {KURE-v2: a Korean-English bilingual late-interaction retriever},
  author = {Jang, Youngjoon and Son, Junyoung and Lee, Taemin and Hong, Seongtae and Lim, Heuiseok},
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
