#!/usr/bin/env bash
set -euo pipefail

if [ -f config.sh ]; then
  source config.sh
fi

WANDB_ENABLE="${WANDB_ENABLE:-1}"
if [ "${WANDB_ENABLE}" != "0" ]; then
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY is not set. Put it in config.sh or export it before running."
    exit 1
  fi
  python -m wandb login "$WANDB_API_KEY"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"

# Override any of these with environment variables before running this script.
MUON_LR="${MUON_LR:-0.025}"
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.025}"
MUON_MU="${MUON_MU:-0.95}"
MUON_NESTEROV="${MUON_NESTEROV:-true}"

export MUON_CONFIG="{\"lr\":${MUON_LR},\"weight_decay\":${MUON_WEIGHT_DECAY},\"mu\":${MUON_MU},\"nesterov\":${MUON_NESTEROV}}"
export MATRIX_OPT="muon"
export WANDB_NOTES="muon baseline"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt_simple.py "$@"
