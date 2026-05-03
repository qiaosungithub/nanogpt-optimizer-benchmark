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
FOOF_LR=0.05                                   # default 0.025
FOOF_WEIGHT_DECAY=0.015                         # default 0.025
FOOF_BETA=0.95                                  # default 0.95
FOOF_FW_STEPS=4                                 # default 4
FOOF_ALPHA_MULT=1.0                             # default 1.0
FOOF_NESTEROV=true                              # default true
FOOF_EPS=1e-12                                  # default 1e-12

export FOOF_CONFIG="{\"lr\":${FOOF_LR},\"weight_decay\":${FOOF_WEIGHT_DECAY},\"beta\":${FOOF_BETA},\"fw_steps\":${FOOF_FW_STEPS},\"alpha_mult\":${FOOF_ALPHA_MULT},\"nesterov\":${FOOF_NESTEROV},\"eps\":${FOOF_EPS}}"
export MATRIX_OPT="foof"
export WANDB_NOTES="FooF baseline, lr 0.015, wd 0.025, FW 4 iters"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt_simple.py "$@"
