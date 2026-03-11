#!/bin/bash
#SBATCH -J moka_train_infer_ave
#SBATCH --partition=overcap
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:a40:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o /nethome/rkhan96/flash/MokA/AudioVisualText/slurm-train-infer-%j.out
#SBATCH -e /nethome/rkhan96/flash/MokA/AudioVisualText/slurm-train-infer-%j.err

set -euo pipefail

source /nethome/rkhan96/flash/miniconda3/etc/profile.d/conda.sh
conda activate moka_torch25

cd /nethome/rkhan96/flash/MokA/AudioVisualText
export PYTHONUNBUFFERED=1
export NPROC_PER_NODE=1
export DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-deepspeed/stage2-offload-torch25.json}

# Run identity / config (override via sbatch --export=ALL,...)
export MODEL_NAME=${MODEL_NAME:-llama2}
export DATASET=${DATASET:-ave}
export PRECISION=${PRECISION:-bf16}
export CROSS_ATTN_KV_MODE=${CROSS_ATTN_KV_MODE:-question}
export CROSS_MODAL_MODE=${CROSS_MODAL_MODE:-trilinear}   # pairwise|trilinear
export BLC_WEIGHT=${BLC_WEIGHT:-1}
export NUM_EPOCHS=${NUM_EPOCHS:-3}
export TRILINEAR_PACK_TOKENS=${TRILINEAR_PACK_TOKENS:-False}
export SEED=${SEED:-42}

# Optional label in folder name (e.g. kernel/packed)
export ABLATION_TAG=${ABLATION_TAG:-}

# If OUTPUT_DIR is not provided, create a deterministic one here so inference can use it.
if [ -z "${OUTPUT_DIR:-}" ]; then
  RUN_TAG="${MODEL_NAME}_${DATASET}_${PRECISION}_${CROSS_ATTN_KV_MODE}"
  if [ "${CROSS_MODAL_MODE}" = "pairwise" ]; then
    RUN_TAG="${RUN_TAG}_moka"
  else
    RUN_TAG="${RUN_TAG}_trilinear"
  fi
  if [ -n "${ABLATION_TAG}" ]; then
    RUN_TAG="${RUN_TAG}_${ABLATION_TAG}"
  fi
  RUN_TAG="${RUN_TAG}_ep${NUM_EPOCHS}_bw${BLC_WEIGHT}_$(date +%Y%m%d_%H%M%S)"
  export OUTPUT_DIR="results/finetune/${RUN_TAG}"
fi

echo "OUTPUT_DIR=${OUTPUT_DIR}"

echo "=== TRAIN ==="
bash scripts/finetune/ft_ave.sh

echo "=== INFER + EVAL ==="
export YOUR_CKPT_PARH="${OUTPUT_DIR}"

# Ensure inference uses the same fusion mode
if [ "${CROSS_MODAL_MODE}" = "pairwise" ]; then
  export CROSS_MODAL_MODE=pairwise
else
  export CROSS_MODAL_MODE=trilinear
fi

bash scripts/finetune/infer_ave.sh

echo "Done. Metrics should be in: ${YOUR_CKPT_PARH}/inference_results/metrics.jsonl"

