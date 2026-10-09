#!/usr/bin/env bash
# Train KURE-Sparse-base or -nano (SFT) with the released settings.
#
#   bash train/sparse/train.sh base <sft-data-root> [num-gpus]
#   bash train/sparse/train.sh nano <sft-data-root> [num-gpus] [pft-checkpoint]
#
# <sft-data-root> contains sft_<lang> (query, positive, negative_0..9, label), the label being the teacher
# reranker's yes/no logit margin for [positive, negative_0..9]. base starts from omai-research/milco-650m; nano
# from the stage-1 checkpoint of pft.sh (default output/kure-sparse-nano-pft/final). Both: in-batch contrastive
# + min-max KL (T 0.2) against the teacher, L1 on both sides, 8,192 tokens, global batch 1,024, 1 epoch, fp32
# master weights under bf16 autocast.
set -euo pipefail

USAGE="usage: train.sh <base|nano> <sft-data-root> [num-gpus] [pft-checkpoint]"
MODEL=${1:?$USAGE}
DATA=${2:?$USAGE}
NPROC=${3:-8}
DIR=$(cd "$(dirname "$0")" && pwd)
GLOBAL_BATCH_SIZE=1024
PER_DEVICE=4

case "$MODEL" in
    # MILCO's scores are ~10x smaller than the A.X head's, so its contrastive temperature is 10x lower.
    base) INIT=omai-research/milco-650m; LR=1e-5; CTEMP_SCALE=1.66667; QREG=1e-3; DREG=1e-5; CHUNK=24000 ;;
    nano) INIT=${4:-output/kure-sparse-nano-pft/final}; LR=2e-5; CTEMP_SCALE=0.166667; QREG=3e-4; DREG=1e-4
          CHUNK=30000 ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
if (( GLOBAL_BATCH_SIZE % (PER_DEVICE * NPROC) != 0 )); then
    echo "global batch $GLOBAL_BATCH_SIZE is not a multiple of $PER_DEVICE x $NPROC GPUs" >&2; exit 1
fi
ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE * NPROC)))

# The nano checkpoint carries its "[Q] "/"[D] " prompts and its pooling; MILCO takes no prompts.
PROMPTS=()
[ "$MODEL" = nano ] && PROMPTS=(--query_prompt "[Q] " --document_prompt "[D] ")

# EXLEN chunks are sized for a 48 GB A6000 (8 GPUs); raise --exlen_chunk_tokens on larger cards.
uv run --extra train torchrun --nproc_per_node "$NPROC" "$DIR/train.py" \
    --model_name_or_path "$INIT" \
    --train_dataset_path "$DATA/sft_ko" "$DATA/sft_en" \
    --num_negatives 10 \
    "${PROMPTS[@]}" \
    --distill_loss contrastive_kldiv \
    --contrastive_weight 1.0 \
    --kldiv_weight 1.0 \
    --contrastive_scale "$CTEMP_SCALE" \
    --kldiv_temperature 0.2 \
    --normalize_student true \
    --normalize_teacher true \
    --drop_false_negatives true \
    --false_negative_margin 0.0 \
    --gather_across_devices true \
    --splade_regularizer l1 \
    --query_regularizer_weight "$QREG" \
    --document_regularizer_weight "$DREG" \
    --max_seq_length 8192 \
    --extended_length true \
    --exlen_chunk_tokens "$CHUNK" \
    --eval_max_active_dims 0 \
    --per_device_train_batch_size "$PER_DEVICE" \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps "$ACCUM" \
    --learning_rate "$LR" \
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
    --ddp_timeout 3600 \
    --output_dir "output/kure-sparse-$MODEL" \
    --run_name "kure-sparse-$MODEL" \
    --report_to "${REPORT_TO:-none}"
