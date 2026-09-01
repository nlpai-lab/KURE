#!/usr/bin/env bash
# Two-stage KURE-v2 training: PFT then SFT from the PFT checkpoint.
#
#   bash train/late-interaction/run.sh <pft-data-root> <sft-data-root> [num-gpus]
#
# <pft-data-root> contains pft_<lang> (anchor, positive);
# <sft-data-root> contains sft_<lang> (query, positive, negative_0..9, label).
set -euo pipefail

USAGE="usage: run.sh <pft-data-root> <sft-data-root> [num-gpus]"
PFT_DATA=${1:?$USAGE}
SFT_DATA=${2:?$USAGE}
NPROC=${3:-8}
DIR=$(cd "$(dirname "$0")" && pwd)

uv run torchrun --nproc_per_node "$NPROC" "$DIR/1_pft.py" \
    --data-root "$PFT_DATA" \
    --head multi \
    --run-name kure-v2-pft

uv run torchrun --nproc_per_node "$NPROC" "$DIR/2_sft.py" \
    --model-name output/kure-v2-pft/final \
    --data-root "$SFT_DATA" \
    --run-name kure-v2-sft
