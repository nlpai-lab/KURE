# Training KURE-Reranker (cross-encoder)

The recipe that produced [nlpai-lab/KURE-Reranker-base](https://huggingface.co/nlpai-lab/KURE-Reranker-base) and [nlpai-lab/KURE-Reranker-nano](https://huggingface.co/nlpai-lab/KURE-Reranker-nano): a cross-encoder distilled from a Qwen3-Reranker teacher with pointwise MSE on the teacher's raw yes/no logit (the [ettin-reranker](https://huggingface.co/blog/ettin-reranker) recipe), trained with [sentence-transformers](https://github.com/UKPLab/sentence-transformers) in bf16.

```
train.py                  training entrypoint (CrossEncoderTrainer + MSELoss)
train.sh                  the released settings: `train.sh base|nano <sft-data-root> [num-gpus]`
src/data.py               11-way SFT rows -> (query, doc, label) pointwise rows
src/qwen_backbone.py      wires a base Qwen3 causal LM as a Qwen3-Reranker-format CrossEncoder
src/length_grouping.py    length-grouped batches for training at max_seq_length 8192
src/utils.py              logging setup
```

In-training evaluation reranks five small Korean tasks with the benchmark protocol of [`eval/cross-encoder`](../../eval/cross-encoder/evaluate.py).

## Setup

```bash
uv sync --extra train
uv pip install flash-attn --no-build-isolation
bash eval/cross-encoder/fetch_eval_assets.sh   # BM25 candidate pools for the in-training evaluator
```

## Quickstart

```bash
bash train/cross-encoder/train.sh base /path/to/sft 8   # KURE-Reranker-base
bash train/cross-encoder/train.sh nano /path/to/sft 8   # KURE-Reranker-nano
```

The final model is saved to `output/kure-reranker-<base|nano>/final`. Set `REPORT_TO=wandb` to log to Weights & Biases.

## Data

The same 11-way SFT format as [KURE-v2](../late-interaction/README.md#data), a `datasets.save_to_disk` directory per language:

| Path | Columns |
|---|---|
| `<sft-data-root>/sft_<lang>` | `query`, `positive`, `negative_0`..`negative_9`, `label` |

`label` is the teacher reranker's yes/no logit margin for `[positive, negative_0, ..., negative_9]`, in that order. The loader flattens every row into 11 `(query, doc, label)` pairs, so the 1.46M Korean + 1.57M English rows become 33.3M training pairs.

| Model | Teacher (`label`) | Korean data |
|---|---|---|
| KURE-Reranker-base | [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | KURE-v2 SFT data, with long documents from the [MLDR](https://huggingface.co/datasets/Shitao/MLDR) Korean train split |
| KURE-Reranker-nano | [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | KURE-v2 SFT data |

## Recipe

One stage, 1 epoch, pointwise MSE between the student's score and the teacher's raw logit margin. The score stays un-squashed (identity activation), so the student learns the teacher's scale, not only its ranking.

| | KURE-Reranker-base | KURE-Reranker-nano |
|---|---|---|
| Backbone | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) |
| Scoring | logit("yes") − logit("no") over the Qwen3-Reranker template | a single-logit classification head |
| Max length | 8,192 | 8,192 |
| Global batch | 1,024 | 1,024 |
| Learning rate | 2e-5, cosine | 2e-5, linear |
| Warmup | 10% | 10% |

**Causal backbone (base).** A base Qwen3 ships none of the reranker wiring the Qwen3-Reranker repos have, so `src/qwen_backbone.py` installs the Qwen3-Reranker chat template (`<Instruct>`, `<Query>`, `<Document>`) and a yes/no `LogitScore` head. The student therefore reads exactly the input the teacher scored when it produced `label`, with the teacher's default instruction (*"Given a web search query, retrieve relevant passages that answer the query"*), which also ships with the released model.

**Extended length (base).** `--extended_length` batches rows by length (homogeneous batches, shuffled in blocks of one per GPU so the ranks stay in step) and runs a batch over `--exlen_chunk_tokens` tokens a few rows at a time. Pointwise MSE has no in-batch terms, so the summed chunk gradients equal the whole-batch gradient exactly; the longest block runs first as a memory probe. KURE-Reranker-base was trained this way on 4 GPUs (per-device batch 16, accumulation 16). KURE-Reranker-nano was trained with plain batches on 8 GPUs (per-device batch 64, accumulation 2).
