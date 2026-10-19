#!/usr/bin/env bash
# Train KURE-v3 with the released settings.
#
#   bash train/dense/train.sh <sft-data-root> [num-gpus]
#
# <sft-data-root> contains sft_<lang> (query, positive, negative_0..9, label), the label being the teacher's score
# for [positive, negative_0..9]. The student is nvidia/Nemotron-3-Embed-1B-BF16 with its own "query: " / "passage: "
# prompts. Min-max KL (T 0.2) against the teacher at every Matryoshka dimension (2048, 1024, 512, 256) plus Global
# Orthogonal Regularization (0.01), 8,192 tokens, global batch 1,024, 1 epoch, fp32 master weights under bf16
# autocast.
set -euo pipefail

USAGE="usage: train.sh <sft-data-root> [num-gpus]"
DATA=${1:?$USAGE}
NPROC=${2:-8}
DIR=$(cd "$(dirname "$0")" && pwd)
GLOBAL_BATCH_SIZE=1024
PER_DEVICE=32
if (( GLOBAL_BATCH_SIZE % (PER_DEVICE * NPROC) != 0 )); then
    echo "global batch $GLOBAL_BATCH_SIZE is not a multiple of $PER_DEVICE x $NPROC GPUs" >&2; exit 1
fi
ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE * NPROC)))

# Extended length: batches are grouped by length, and one over --exlen_chunk_tokens tokens runs through the step a
# few rows at a time (exact: the KL has no in-batch terms). 80,000 is sized for a 48 GB A6000 with fp32 master
# weights; raise it on larger cards.
uv run --extra train torchrun --nproc_per_node "$NPROC" "$DIR/train.py" \
    --model_name_or_path nvidia/Nemotron-3-Embed-1B-BF16 \
    --train_dataset_path "$DATA/sft_ko" "$DATA/sft_en" \
    --num_negatives 10 \
    --query_prompt "query: " \
    --document_prompt "passage: " \
    --distill_loss kldiv_minmax \
    --kldiv_temperature 0.2 \
    --matryoshka_dims 2048 1024 512 256 \
    --gor_weight 0.01 \
    --max_seq_length 8192 \
    --extended_length true \
    --exlen_chunk_tokens 80000 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size "$PER_DEVICE" \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps "$ACCUM" \
    --gradient_checkpointing true \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --dtype float32 \
    --bf16 \
    --eval_strategy steps --eval_steps 0.05 \
    --save_strategy steps --save_steps 0.1 \
    --logging_steps 10 \
    --dataloader_drop_last true \
    --dataloader_num_workers 8 \
    --ddp_find_unused_parameters false \
    --ddp_timeout 3600 \
    --output_dir output/kure-v3 \
    --run_name kure-v3 \
    --report_to "${REPORT_TO:-none}"
