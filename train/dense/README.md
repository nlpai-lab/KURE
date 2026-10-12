# Training KURE-v3 (dense)

The recipe that produced [nlpai-lab/KURE-v3](https://huggingface.co/nlpai-lab/KURE-v3): a dense retriever distilled from a teacher's scores over each query's candidate documents, trained with [sentence-transformers](https://github.com/UKPLab/sentence-transformers) on top of [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16).

```
train.py                training entrypoint (SentenceTransformerTrainer)
train.sh                the released settings: `train.sh <sft-data-root> [num-gpus]`
src/losses.py           min-max KL distillation, Global Orthogonal Regularization, per-dimension loss logging
src/length_grouping.py  length-grouped batches and token-budget row chunks for training at 8,192 tokens
src/data.py, src/evaluators.py, src/arguments.py, src/utils.py
```

In-training evaluation scores five small Korean tasks of MTEB(kor, v2) on rank 0.

## Setup

```bash
uv sync --extra train
uv pip install flash-attn --no-build-isolation
```

## Quickstart

```bash
bash train/dense/train.sh /path/to/sft 8
```

The model is saved to `output/kure-v3/final`. Set `REPORT_TO=wandb` to log to Weights & Biases. The token budget in the script is sized for 48 GB GPUs; raise `--exlen_chunk_tokens` on larger cards (it sets row-chunk sizes only, so the loss is unchanged).

## Data

The same `datasets.save_to_disk` directories as [KURE-v2](../late-interaction/README.md#data), one per language:

| Stage | Path | Columns |
|---|---|---|
| SFT | `<sft-data-root>/sft_<lang>` | `query`, `positive`, `negative_0`..`negative_9`, `label` |

`label` is the teacher's score for `[positive, negative_0, ..., negative_9]`. KURE-v3 uses the KURE-v2 SFT data (1.46M Korean + 1.57M English rows), with long documents from the [MLDR](https://huggingface.co/datasets/Shitao/MLDR) Korean train split. Keep one teacher per run: different teachers score on different scales.

## Recipe

**Objective.** For each query, the student's scores for its 11 candidates and the teacher's are both min-max normalized to [0, 1] and turned into distributions with a softmax at temperature 0.2, and the student minimizes the KL divergence to the teacher's. Normalizing both sides makes the loss match the shape of the teacher's ranking rather than its scale. There are no in-batch negatives, so the batch composition does not change the loss.

**Matryoshka and regularization.** The loss is applied at 2,048, 1,024, 512 and 256 dimensions, so truncated vectors stay usable. A Global Orthogonal Regularization term (weight 0.01) is added at every dimension.

**Extended length.** `--extended_length` batches rows by length (homogeneous batches, shuffled in blocks of one per GPU so the ranks stay in step, the longest block first as a memory probe). A batch over `--exlen_chunk_tokens` tokens runs through the step a few rows at a time, which is exact because the loss has no in-batch terms.

**Setup.** One epoch at a global batch of 1,024 and up to 8,192 tokens, learning rate 2e-5 with 10% warmup and cosine decay. Weights are kept in fp32 and the forward runs under bf16 autocast: with bf16 weights most lr-sized AdamW steps round away.
