#!/bin/bash
#SBATCH -J moka_eval_ave
#SBATCH -p overcap
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /nethome/rkhan96/flash/MokA/AudioVisualText/slurm-eval-%j.out
#SBATCH -e /nethome/rkhan96/flash/MokA/AudioVisualText/slurm-eval-%j.err

set -euo pipefail

source /nethome/rkhan96/flash/miniconda3/etc/profile.d/conda.sh
conda activate moka_torch25

cd /nethome/rkhan96/flash/MokA/AudioVisualText
export PYTHONUNBUFFERED=1

# Required: pass RUN_DIR via --export=ALL,RUN_DIR=results/finetune/<run_dir>
export RUN_DIR
if [ -z "${RUN_DIR:-}" ]; then
    echo "RUN_DIR is required. Example: sbatch --export=ALL,RUN_DIR=results/finetune/llama2_ave_bf16_20250305_143022 run_eval_ave.sbatch"
    exit 1
fi

python scripts/ablation/ave_eval_flexible.py \
  --inference-jsonl "$RUN_DIR/inference_results/inference_ave.jsonl" \
  --annotations scripts/evaluation/Annotations.txt \
  --out-json "$RUN_DIR/inference_results/metrics_ave.json"
