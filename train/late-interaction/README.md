# Training KURE-v2 (late-interaction)

The two-stage recipe that produced [nlpai-lab/KURE-v2](https://huggingface.co/nlpai-lab/KURE-v2): a ColBERT built from [skt/A.X-Encoder-base](https://huggingface.co/skt/A.X-Encoder-base) with a multi-layer projection head (128-d per token), trained with [PyLate](https://github.com/lightonai/pylate) in bf16.

```
1_pft.py          stage 1 (PFT): large-batch contrastive learning on weakly related pairs
2_sft.py          stage 2 (SFT): contrastive + KL distillation from a reranker teacher
run.sh            both stages in sequence
modules.py        loss / collator / callback, extracted from lightonai's mdenseon-mlateon
                  training code (Apache-2.0)
dev_evaluator.py  three cheap Korean retrieval dev tasks for in-training monitoring
```

## Setup

```bash
uv sync --extra train
```

## Quickstart

Both stages in sequence, with the KURE-v2 settings:

```bash
bash train/late-interaction/run.sh /path/to/pft /path/to/sft 8
```

## Data

Both stages read `datasets.save_to_disk` directories, one per language:

| Stage | Path | Columns |
|---|---|---|
| PFT | `<data-root>/pft_<lang>` | `anchor`, `positive` |
| SFT | `<data-root>/sft_<lang>` | `query`, `positive`, `negative_0`..`negative_9`, `label` |

`label` is the teacher reranker's scores for `[positive, negative_0, ..., negative_9]`, in that order. KURE-v2 used 10.35M Korean + 10.35M English PFT pairs and 1.46M Korean + 1.57M English SFT triplets scored by [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B).

## Stage 1: pre-finetuning (PFT)

Contrastive learning (pylate `CachedContrastive`) where the only negatives are the other pairs in a very large batch. `--batch-size` is the *global* in-batch negative pool (split across devices), so the GPU count changes wall clock, not the loss. Batches are language-pure so the pool is never separable by language alone.

```bash
uv run torchrun --nproc_per_node 8 train/late-interaction/1_pft.py \
    --data-root /path/to/pft --head multi --run-name kure-v2-pft
```

KURE-v2 settings are the defaults: global batch 16,384, lr 1e-4, temperature 0.02, query/document length 64/256, 1 epoch. The output `output/kure-v2-pft/final` is what we released as [KURE-v2-unsupervised](https://huggingface.co/nlpai-lab/KURE-v2-unsupervised).

## Stage 2: supervised fine-tuning (SFT)

Contrastive learning plus KL distillation of the teacher's ranking (`CachedContrastiveKLDiv`). A sharp student temperature (0.001 vs teacher 0.1) distills the ranking structure rather than absolute margins. Rows whose hardest negative outscores the positive under the teacher are dropped as likely false negatives.

```bash
uv run torchrun --nproc_per_node 8 train/late-interaction/2_sft.py \
    --model-name output/kure-v2-pft/final --data-root /path/to/sft \
    --run-name kure-v2-sft
```

KURE-v2 settings are the defaults: global batch 256, lr 2e-5, 7 negatives sampled per step from the stored 10, contrastive temperature 0.02, document length 1,024, 1 epoch. The stage-1 checkpoint already carries the multi-layer head, so the default `--head single --embedding-size 0` loads its module stack as-is; pass `--head multi` only when starting stage 2 from a raw encoder (it builds a fresh head).
