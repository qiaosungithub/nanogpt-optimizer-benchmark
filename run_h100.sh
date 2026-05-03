#!/usr/bin/env bash
set -euo pipefail

if [ -f config.sh ]; then
  source config.sh
fi

WANDB_ENABLE="${WANDB_ENABLE:-1}"
if [ "${WANDB_ENABLE}" != "0" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set. Put it in config.sh or export it before running."
  exit 1
fi

OPTIMIZER="${MATRIX_OPT:-foof}"
if [ "${1:-}" = "foof" ] || [ "${1:-}" = "muon" ]; then
  OPTIMIZER="$1"
  shift
fi

if [ "$OPTIMIZER" != "foof" ] && [ "$OPTIMIZER" != "muon" ]; then
  echo "Unsupported optimizer: $OPTIMIZER (expected foof or muon)"
  exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-20000}"
  AVAILABLE_CUDA_DEVICES=$(
    nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v min_free="$GPU_FREE_MEM_MB" '{free=$2-$3; if (free >= min_free) print $1}' \
      | paste -sd, -
  )
  if [ -z "$AVAILABLE_CUDA_DEVICES" ]; then
    echo "No GPU has at least ${GPU_FREE_MEM_MB}MB free memory."
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$AVAILABLE_CUDA_DEVICES"
fi

NUM_CUDA_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ "$NUM_CUDA_DEVICES" -le 0 ]; then
  echo "Failed to parse CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  exit 1
fi

STAGE_ROOT="${STAGE_ROOT:-/data/scratch-oc40/$USER/stage/nanogpt_optimizer_benchmark}"
NOW=$(date +%Y%m%d_%H%M%S)
RND_STR=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6 ; echo '')
GIT_COMMIT=$(git rev-parse --short HEAD || echo 'no_git')
STAGE_NAME="${NOW}_${RND_STR}_${GIT_COMMIT}"
STAGE_DIR="$STAGE_ROOT/$STAGE_NAME"
mkdir -p "$STAGE_DIR"
rsync -av . "$STAGE_DIR" --exclude='.git' --exclude='*.pyc' --exclude='__pycache__' --exclude='tmp'

if [ -n "${DATA_ROOT:-}" ]; then
  mkdir -p "$STAGE_DIR/data"
  if [ -d "$DATA_ROOT/fineweb10B" ]; then
    ln -sfn "$DATA_ROOT/fineweb10B" "$STAGE_DIR/data/fineweb10B"
  elif compgen -G "$DATA_ROOT/fineweb_train_*.bin" > /dev/null; then
    ln -sfn "$DATA_ROOT" "$STAGE_DIR/data/fineweb10B"
  else
    echo "DATA_ROOT does not contain FineWeb shards: $DATA_ROOT"
    exit 1
  fi
fi

LOG_DIR="$STAGE_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Staging done to $STAGE_DIR"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using optimizer=$OPTIMIZER"
echo "Logs: $LOG_DIR/output.log"

cd "$STAGE_DIR"
export NPROC_PER_NODE="$NUM_CUDA_DEVICES"
export MASTER_PORT=$((10000 + RANDOM % 20000))
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/scratch-oc40/$USER/triton_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/scratch-oc40/$USER/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/data/scratch-oc40/$USER/compile_cache}"

echo "Starting at $(date)"
"./run_${OPTIMIZER}.sh" "$@" 2>&1 | tee -a "$LOG_DIR/output.log"
echo "Finished at $(date)"
echo "check logs at $LOG_DIR/output.log"
