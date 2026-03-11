# AudioVisualText Local Changes (Run/Eval/Analysis)

This document summarizes the practical changes made to run AVE finetuning locally, evaluate results, and add gradient-sensitivity analysis.

## 1) Runtime/Path Updates

- Updated local checkpoint paths in finetune and inference scripts:
  - `AudioVisualText/scripts/finetune/ft_ave.sh`
  - `AudioVisualText/scripts/finetune/infer_ave.sh`
- Added dynamic GPU process detection:
  - `NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}`
  - Fallback to `1` when detection returns `0`.
- Set AVE scripts to use local weights under:
  - `/nethome/rkhan96/flash/weights/...`

## 2) Precision/Compatibility Fixes

- Training and inference were aligned to FP32 (`bf16=False`) to avoid dtype mismatch issues observed with BF16 on this setup.
- `AudioVisualText/deepspeed/stage2-offload.json`
  - `bf16.enabled` set to `false`.

## 3) Gradient Sensitivity Instrumentation

### Config Flags

- Added training flags in:
  - `AudioVisualText/configs/unified_config.py`
- New fields:
  - `grad_sensitivity_enable`
  - `grad_sensitivity_include_projectors`

### Trainer Logging

- Extended `UnifiedTrainer` in:
  - `AudioVisualText/trainer.py`
- Added per-step logging for:
  - `lora_A_text`, `lora_A_visual`, `lora_A_audio`, `lora_B_shared`
  - optional `vl_projector`, `al_projector`
- Logged metrics:
  - `*_grad_norm`
  - `*_param_norm`
  - `*_relative_grad_norm`
  - `*_num_params`
- Output file:
  - `<output_dir>/grad_sensitivity.jsonl`

### DeepSpeed/ZeRO Reliability Fix

- Initial implementation used `param.grad` and produced near-zero gradients in logs.
- Updated implementation to use parameter backward hooks to accumulate grad norms reliably under DeepSpeed.

## 4) Script Controls for Clean Runs

- Updated `AudioVisualText/scripts/finetune/ft_ave.sh` run naming:
  - If `RUN_NAME` is set, use it.
  - Else if `GRAD_SENS_RUN=1`, use `llama_ave_gradsens`.
  - Else use `llama_ave`.
- This avoids accidental resume collisions with existing checkpoint directories.

## 5) Evaluation Summary (AVE)

- Finetune result (3 epochs) was reproduced near paper-level:
  - AVE accuracy: **77.24%**
  - Reported reference: **77.06%**
- Parse-valid samples differed due to format strictness in evaluator:
  - Local run: `394/402`
  - Reference: `397/402`

## 6) Gradient Analysis Artifacts

### Analysis Script

- Added:
  - `AudioVisualText/scripts/analysis/plot_grad_sensitivity.py`
- Script outputs:
  - `grad_sensitivity_long.csv`
  - `grad_sensitivity_summary.csv`
  - PNG plots (if `matplotlib` is installed):
    - `lora_grad.png`
    - `lora_rel.png`
    - `projector_rel.png`

### Current Analysis Output Location

- `AudioVisualText/results/finetune/llama_ave_gradsens_v2/analysis/`

## 7) Notes on Job Interruptions

- One long run was preempted by scheduler, but partial gradient logs were captured.
- The partial `grad_sensitivity.jsonl` still confirms non-zero gradient signals after hook-based fix.

