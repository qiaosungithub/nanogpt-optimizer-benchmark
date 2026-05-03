#!/usr/bin/env bash
set -euo pipefail

USE_PARTITION="${1:-}"
if [ -z "$USE_PARTITION" ]; then
  echo "Usage: $0 <he|csail> [foof|muon] [train_args...]"
  exit 1
fi
shift

if [ "$USE_PARTITION" = "he" ]; then
  ACCOUNT="${ACCOUNT:-vision-he}"
  PARTITION="${PARTITION:-vision-he-h200}"
  QOS="${QOS:-vision-he-main}"
  TIMELIMIT="${SBATCH_TIME:-47:59:59}"
elif [ "$USE_PARTITION" = "csail" ]; then
  ACCOUNT="${ACCOUNT:-csail}"
  PARTITION="${PARTITION:-csail-shared-h200}"
  QOS="${QOS:-shared-if-available}"
  TIMELIMIT="${SBATCH_TIME:-23:59:59}"
else
  echo "Unknown USE_PARTITION: $USE_PARTITION"
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

if [ -f config.sh ]; then
  source config.sh
fi

WANDB_ENABLE="${WANDB_ENABLE:-1}"
if [ "${WANDB_ENABLE}" != "0" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set. Put it in config.sh or export it before running."
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

mkdir -p "/data/scratch-oc40/$USER/slurm"

if [ "$#" -gt 0 ]; then
  CMD_ARGS_STR=$(printf "%q " "$@")
else
  CMD_ARGS_STR=""
fi
SBATCH_MEM="${SBATCH_MEM:-800G}"
SBATCH_CPUS="${SBATCH_CPUS:-32}"
SBATCH_GPUS="${SBATCH_GPUS:-8}"
CUDA_VISIBLE_DEVICES_LIST=$(seq -s, 0 $((SBATCH_GPUS - 1)))
DOWN_NODES="${DOWN_NODES:-}"
EXCLUDE_LINE=""
if [ -n "$DOWN_NODES" ]; then
  EXCLUDE_LINE="#SBATCH --exclude=$DOWN_NODES"
fi

SLURM_SCRIPT_FILE="$STAGE_DIR/slurm"
printf "%s" "#!/bin/bash
#
#SBATCH --job-name=nanogpt_opt
$EXCLUDE_LINE
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --time=$TIMELIMIT
#SBATCH --output=/data/scratch-oc40/$USER/slurm/%j.log
#SBATCH --error=/data/scratch-oc40/$USER/slurm/%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:$SBATCH_GPUS
#SBATCH --mem=$SBATCH_MEM
#SBATCH --cpus-per-task=$SBATCH_CPUS

set -euo pipefail

echo \"My Job ID is \$SLURM_JOB_ID\"
echo \"The time is \$(date)\"
echo \"This job is running on \$(hostname)\"
nvidia-smi

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_LIST
export NPROC_PER_NODE=$SBATCH_GPUS
export MASTER_PORT=\$((10000 + RANDOM % 20000))
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/data/scratch-oc40/$USER/triton_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data/scratch-oc40/$USER/.cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/data/scratch-oc40/$USER/compile_cache}

LOG_DIR=$STAGE_DIR/log_\$(hostname)
mkdir -p \$LOG_DIR
echo \"Log dir: \$LOG_DIR\"

cd $STAGE_DIR
echo \"Starting at \$(date)\"
./run_${OPTIMIZER}.sh $CMD_ARGS_STR 2>&1 | tee -a \$LOG_DIR/output.log
echo \"Finished at \$(date)\"
echo \"check logs at \$LOG_DIR/output.log\"
" > "$SLURM_SCRIPT_FILE"

JOB_ID=$(sbatch "$SLURM_SCRIPT_FILE")
JOB_ID=${JOB_ID##* }
echo "Submitted job $JOB_ID"

sleep 3
echo
echo "--------------------------------------------------------------------"
echo
tail -f "/data/scratch-oc40/$USER/slurm/$JOB_ID.log" || echo -e "Job is in queue\n$(squeue | grep "$USER")\nPlease wait"
