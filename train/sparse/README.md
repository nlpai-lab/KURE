# Training KURE-Sparse (learned sparse)

The recipes that produced [nlpai-lab/KURE-Sparse-base](https://huggingface.co/nlpai-lab/KURE-Sparse-base) and [nlpai-lab/KURE-Sparse-nano](https://huggingface.co/nlpai-lab/KURE-Sparse-nano): SPLADE-style retrievers distilled from a Qwen3-Reranker teacher with an in-batch contrastive term, trained with [sentence-transformers](https://github.com/UKPLab/sentence-transformers) `SparseEncoder`.

```
train.py              training entrypoint (SparseEncoderTrainer); both stages, selected by --distill_loss
train.sh              SFT with the released settings: `train.sh base|nano <sft-data-root> [num-gpus] [pft-checkpoint]`
pft.sh                KURE-Sparse-nano stage 1: contrastive pre-finetuning of the grafted backbone
graft_mlm_head.py     KURE-Sparse-nano backbone: the A.X-Encoder-base MLM head on the KURE-v2-unsupervised body
src/losses.py         contrastive + min-max KL loss, L1 regularizer, token-budget GradCache for long batches
src/pooling.py        SPLADE pooling with a logit shift, muted prefix tokens and vocabulary canonicalization
src/milco_encoder.py  MILCO (pivot + LexEcho source view) as a SparseEncoder module
src/length_grouping.py  length-grouped batches for training at 8,192 tokens
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
# KURE-Sparse-base: one stage from MILCO
bash train/sparse/train.sh base /path/to/sft 8

# KURE-Sparse-nano: graft, pre-finetune, then SFT
uv run python train/sparse/graft_mlm_head.py output/ax-kurev2unsup-mlm
bash train/sparse/pft.sh output/ax-kurev2unsup-mlm /path/to/pft 8
bash train/sparse/train.sh nano /path/to/sft 8 output/kure-sparse-nano-pft/final
```

Models are saved to `output/<run>/final`. Set `REPORT_TO=wandb` to log to Weights & Biases. The EXLEN token budgets in the scripts are sized for 48 GB GPUs; raise `--exlen_chunk_tokens` on larger cards (it sets GradCache chunk sizes only, so the loss is unchanged).

## Data

The same `datasets.save_to_disk` directories as [KURE-v2](../late-interaction/README.md#data), one per language:

| Stage | Path | Columns |
|---|---|---|
| PFT (nano only) | `<pft-data-root>/pft_<lang>` | `anchor`, `positive` |
| SFT | `<sft-data-root>/sft_<lang>` | `query`, `positive`, `negative_0`..`negative_9`, `label` |

`label` is the teacher's yes/no logit margin for `[positive, negative_0, ..., negative_9]`. Both models use [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) labels on the KURE-v2 SFT data (1.46M Korean + 1.57M English rows), with long documents from the [MLDR](https://huggingface.co/datasets/Shitao/MLDR) Korean train split.

## Recipe

**SFT (both models).** One epoch over the SFT rows at a global batch of 1,024 and up to 8,192 tokens. The loss is an in-batch contrastive term (every other row's documents are negatives, gathered across GPUs) plus a KL term that matches each query's distribution over its own 11 candidates to the teacher's, with both score vectors min-max normalized per row (T = 0.2). Rows whose hardest negative outscores the positive under the teacher are dropped as likely false negatives. Sparsity comes from an L1 penalty on queries and documents: unlike FLOPS, which barely touches a dimension only one long document uses, L1 pushes every active weight equally. Weights are kept in fp32 and the forward runs under bf16 autocast; with bf16 weights most lr-sized AdamW steps round away.

| | KURE-Sparse-base | KURE-Sparse-nano |
|---|---|---|
| Initialization | [omai-research/milco-650m](https://huggingface.co/omai-research/milco-650m) | stage-1 checkpoint of `pft.sh` |
| Output space | 30,522 English pivot terms + 250,002 source tokens | 36,282 canonical A.X terms |
| Contrastive temperature | 0.6 | 6 |
| L1 weight (query / document) | 1e-3 / 1e-5 | 3e-4 / 1e-4 |
| Learning rate | 1e-5, cosine | 2e-5, cosine |
| Prompts | none | `[Q] ` / `[D] ` |

The contrastive temperature follows the score scale: MILCO's dot products are about 10x smaller than the A.X head's.

**Extended length.** `--extended_length` batches rows by length (homogeneous batches, shuffled in blocks of one per GPU so the ranks stay in step, the longest block first as a memory probe). SPLADE batches are padded, and both the contrastive term and the regularizer read the whole batch, so a batch over `--exlen_chunk_tokens` padded tokens is not split in the trainer but goes through GradCache in chunks (`src/losses.py::TokenBudgetSpladeLoss`), which is exact.

**KURE-Sparse-nano stage 0 and 1.** KURE-v2-unsupervised is A.X-Encoder-base after PyLate contrastive pre-finetuning, so it has no MLM head; `graft_mlm_head.py` puts A.X's head back on (the SparseUp recipe). On this head the logits sit far above zero, so `src/pooling.py` shifts them by 10 before the ReLU, never pools the `[Q] `/`[D] ` prefix tokens, and folds WordPiece variants (`계약` and `##계약`) into one output term, which turns the 50,002-token vocabulary into 36,282 terms. `pft.sh` then pre-finetunes on the 20.7M PFT pairs with in-batch negatives only, at a global batch of 16,384 and 512 tokens, lr 1e-4, with a light FLOPS penalty (5e-6 / 3e-6) so rare domain vocabulary survives.
