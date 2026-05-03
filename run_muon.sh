#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"

# Override any of these with environment variables before running this script.
MUON_LR="${MUON_LR:-0.025}"
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.025}"
MUON_MU="${MUON_MU:-0.95}"
MUON_NESTEROV="${MUON_NESTEROV:-true}"

export MUON_CONFIG="{\"lr\":${MUON_LR},\"weight_decay\":${MUON_WEIGHT_DECAY},\"mu\":${MUON_MU},\"nesterov\":${MUON_NESTEROV}}"
export MATRIX_OPT="muon"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt_simple.py
