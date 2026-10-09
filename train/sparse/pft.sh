#!/usr/bin/env bash
# Stage 1 of KURE-Sparse-nano: contrastive pre-finetuning of the grafted backbone (graft_mlm_head.py) on
# (anchor, positive) pairs with in-batch negatives only, at a global batch of 16,384, 1 epoch.
#
#   bash train/sparse/pft.sh <backbone> <pft-data-root> [num-gpus]
#
# <pft-data-root> contains pft_<lang> (anchor, positive), the same pairs as KURE-v2's stage 1. The batch is
# global and gathered across ranks, never accumulated: accumulation would add steps, not negatives. The FLOPS
# weights are a tenth of the usual SPLADE values, so rare domain vocabulary survives this stage.
set -euo pipefail

USAGE="usage: pft.sh <backbone> <pft-data-root> [num-gpus]"
BACKBONE=${1:?$USAGE}
DATA=${2:?$USAGE}
NPROC=${3:-8}
DIR=$(cd "$(dirname "$0")" && pwd)
GLOBAL_BATCH_SIZE=16384
if (( GLOBAL_BATCH_SIZE % NPROC != 0 )); then
    echo "global batch $GLOBAL_BATCH_SIZE is not a multiple of $NPROC GPUs" >&2; exit 1
fi

# SparseUp pooling on the grafted head: shift the logits by 10 before the ReLU, pool "[Q] "/"[D] " nowhere,
# and fold WordPiece "##" variants into one output term. Contrastive temperature 6 on the raw SPLADE dots.
# EXLEN chunks are sized for a 48 GB A6000 (8 GPUs); raise --exlen_chunk_tokens on larger cards.
uv run --extra train torchrun --nproc_per_node "$NPROC" "$DIR/train.py" \
    --model_name_or_path "$BACKBONE" \
    --train_dataset_path "$DATA/pft_ko" "$DATA/pft_en" \
    --distill_loss contrastive \
    --splade_logit_shift 10 \
    --splade_exclude_token_ids 50000 50001 \
    --splade_canonical_min_chars 1 \
    --query_prompt "[Q] " \
    --document_prompt "[D] " \
    --max_seq_length 512 \
    --contrastive_scale 0.166667 \
    --gather_across_devices true \
    --query_regularizer_weight 5e-6 \
    --document_regularizer_weight 3e-6 \
    --extended_length true \
    --exlen_chunk_tokens 24000 \
    --use_cached_splade true \
    --mini_batch_size 256 \
    --per_device_train_batch_size $((GLOBAL_BATCH_SIZE / NPROC)) \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --dtype bfloat16 \
    --bf16 \
    --eval_strategy steps --eval_steps 0.1 \
    --save_strategy steps --save_steps 0.25 \
    --logging_steps 5 \
    --dataloader_drop_last true \
    --dataloader_num_workers 8 \
    --ddp_find_unused_parameters false \
    --ddp_timeout 3600 \
    --output_dir output/kure-sparse-nano-pft \
    --run_name kure-sparse-nano-pft \
    --report_to "${REPORT_TO:-none}"
