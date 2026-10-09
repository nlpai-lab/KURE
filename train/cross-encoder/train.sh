#!/usr/bin/env bash
# Train KURE-Reranker-base or -nano with the released settings.
#
#   bash train/cross-encoder/train.sh base <sft-data-root> [num-gpus]
#   bash train/cross-encoder/train.sh nano <sft-data-root> [num-gpus]
#
# <sft-data-root> contains sft_<lang> (query, positive, negative_0..9, label), the label being the
# teacher's yes/no logit margin for [positive, negative_0..9]. Both models use a global batch of 1024;
# the accumulation follows from the GPU count. Run eval/cross-encoder/fetch_eval_assets.sh once first:
# the in-training evaluator reranks the benchmark's BM25 pools.
set -euo pipefail

USAGE="usage: train.sh <base|nano> <sft-data-root> [num-gpus]"
MODEL=${1:?$USAGE}
DATA=${2:?$USAGE}
NPROC=${3:-8}
DIR=$(cd "$(dirname "$0")" && pwd)
GLOBAL_BATCH_SIZE=1024

case "$MODEL" in
    # Qwen3-1.7B as a yes/no-logit reranker, at 8192 with length-grouped batches.
    base) BACKBONE=Qwen/Qwen3-1.7B; PER_DEVICE=16; SCHEDULER=cosine
          EXTRA=(--extended_length true --exlen_chunk_tokens 16000) ;;
    nano) BACKBONE=skt/A.X-Encoder-base; PER_DEVICE=64; SCHEDULER=linear; EXTRA=() ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
if (( GLOBAL_BATCH_SIZE % (PER_DEVICE * NPROC) != 0 )); then
    echo "global batch $GLOBAL_BATCH_SIZE is not a multiple of $PER_DEVICE x $NPROC GPUs" >&2; exit 1
fi
ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE * NPROC)))

uv run --extra train torchrun --nproc_per_node "$NPROC" "$DIR/train.py" \
    --model_name_or_path "$BACKBONE" \
    --train_dataset_path "$DATA/sft_ko" "$DATA/sft_en" \
    --max_seq_length 8192 \
    --per_device_train_batch_size "$PER_DEVICE" \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps "$ACCUM" \
    --learning_rate 2e-5 \
    --lr_scheduler_type "$SCHEDULER" \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --bf16 \
    --eval_strategy steps --eval_steps 0.05 \
    --save_strategy steps --save_steps 0.1 \
    --logging_steps 10 \
    --dataloader_drop_last true \
    --dataloader_num_workers 8 \
    --ddp_find_unused_parameters false \
    --ddp_timeout 3600 \
    --output_dir "output/kure-reranker-$MODEL" \
    --run_name "kure-reranker-$MODEL" \
    --report_to "${REPORT_TO:-none}" \
    "${EXTRA[@]}"
