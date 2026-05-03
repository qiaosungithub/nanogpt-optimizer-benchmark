#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"

# Override any of these with environment variables before running this script.
FOOF_LR="${FOOF_LR:-0.025}"
FOOF_WEIGHT_DECAY="${FOOF_WEIGHT_DECAY:-0.025}"
FOOF_BETA="${FOOF_BETA:-0.95}"
FOOF_FW_STEPS="${FOOF_FW_STEPS:-4}"
FOOF_ALPHA_MULT="${FOOF_ALPHA_MULT:-1.0}"
FOOF_NESTEROV="${FOOF_NESTEROV:-true}"
FOOF_EPS="${FOOF_EPS:-1e-12}"

export FOOF_CONFIG="{\"lr\":${FOOF_LR},\"weight_decay\":${FOOF_WEIGHT_DECAY},\"beta\":${FOOF_BETA},\"fw_steps\":${FOOF_FW_STEPS},\"alpha_mult\":${FOOF_ALPHA_MULT},\"nesterov\":${FOOF_NESTEROV},\"eps\":${FOOF_EPS}}"
export MATRIX_OPT="foof"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt_simple.py
