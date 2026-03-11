# AVE Ablation Workflow (Mq vs Full-Text KV)

This doc explains how to run and log AVE ablations after adding `cross_attn_kv_mode`.

## What changed

- New training/inference arg: `--cross_attn_kv_mode`
  - `question` (default): old behavior, KV from question mask (`M_q`)
  - `full_text`: ablation, KV from all text tokens
- Added flexible evaluator:
  - `scripts/ablation/ave_eval_flexible.py`
  - Accepts explicit input/output paths for automated experiment tracking
- Added SLURM wrappers:
  - `run_infer_ave.sbatch`
  - `run_eval_ave.sbatch`

## Key files

- Config arg definition:
  - `configs/unified_config.py`
- Arg wired into LoRA setup:
  - `scripts/finetune/finetune.py`
  - `scripts/finetune/inference_cut.py`
- Core behavior switch (question vs full-text KV):
  - `peft_hyper/tuners/lora.py`
- Flexible evaluator:
  - `scripts/ablation/ave_eval_flexible.py`
- SLURM wrappers:
  - `run_ft_ave.sbatch`
  - `run_infer_ave.sbatch`
  - `run_eval_ave.sbatch`

## Run directory naming (automatic)

Run dir is built from env vars and ends with date_time:  
`<model>_<dataset>_<precision>_<question|full_text>_[gradsense_]bw<blc>_<YYYYMMDD>_<HHMMSS>`

Examples:
- `llama2_ave_bf16_question_bw1_20250305_143022` — bf16, question (Mq), blc=1
- `llama2_ave_bf16_full_text_bw0.5_20250305_150000` — bf16, full_text, blc=0.5
- `llama2_ave_bf16_question_gradsense_bw0.5_20250305_160000` — + gradsens

Training prints the output directory at start (check slurm-*.out). Use that path for inference/eval.

## Output locations

Training output directory:
- `results/finetune/<auto_run_name>/` (see naming convention above)

Important files:
- `saved_config.json` (full run args)
- `trainer_state.json` (log history incl. training loss)
- `adapter_model.bin`
- `non_lora_trainables.bin`

Inference output directory:
- `results/finetune/<RUN_NAME>/inference_results/`

Inference output file:
- `inference_ave.jsonl`

Evaluation output (recommended):
- `results/finetune/<RUN_NAME>/inference_results/metrics_ave.json`

## Training (SLURM)

From `AudioVisualText/`. Run dir is built from env vars (defaults: MODEL_NAME=llama2, DATASET=ave, PRECISION=bf16, CROSS_ATTN_KV_MODE=question, GRAD_SENS_RUN=0, BLC_WEIGHT=1).

```bash
# Mq baseline (defaults)
sbatch run_ft_ave.sbatch

# Full-text KV ablation
sbatch --export=ALL,CROSS_ATTN_KV_MODE=full_text run_ft_ave.sbatch

# With gradsens and blc=0.5
sbatch --export=ALL,GRAD_SENS_RUN=1,BLC_WEIGHT=0.5 run_ft_ave.sbatch
```

Check slurm-*.out for the printed output directory, then use that path for inference/eval.

## Inference (SLURM)

Pass the exact run directory (from training log). Match CROSS_ATTN_KV_MODE, PRECISION, and BLC_WEIGHT to the trained run.

```bash
# Example: run dir llama2_ave_bf16_question_bw1_20250305_143022
sbatch --export=ALL,YOUR_CKPT_PARH=results/finetune/llama2_ave_bf16_question_bw1_20250305_143022,CROSS_ATTN_KV_MODE=question run_infer_ave.sbatch

# FP32 run: pass PRECISION=fp32 (and BLC_WEIGHT if not 1)
sbatch --export=ALL,YOUR_CKPT_PARH=results/finetune/llama2_ave_fp32_question_bw1_20250305_120000,PRECISION=fp32 run_infer_ave.sbatch
```

Outputs:
- `<YOUR_CKPT_PARH>/inference_results/inference_ave.jsonl`

## Evaluation (SLURM)

Pass the same run directory used for inference.

```bash
sbatch --export=ALL,RUN_DIR=results/finetune/llama2_ave_bf16_question_bw1_20250305_143022 run_eval_ave.sbatch
```

Or direct (no SLURM):

```bash
python scripts/ablation/ave_eval_flexible.py \
  --inference-jsonl results/finetune/llama2_ave_bf16_question_bw1_20250305_143022/inference_results/inference_ave.jsonl \
  --annotations scripts/evaluation/Annotations.txt \
  --out-json results/finetune/llama2_ave_bf16_question_bw1_20250305_143022/inference_results/metrics_ave.json
```

## Quick comparison checklist

For each run, record:
- `cross_attn_kv_mode`
- `blc_weight`
- final train loss (from `trainer_state.json`)
- `frame_accuracy`
- `valid_prediction_format_samples`

Keep all other hyperparameters fixed for fair ablation.
