#!/bin/bash
#SBATCH --partition=overcap
#SBATCH --requeue
#SBATCH --open-mode=append

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MASTER_PORT=${MASTER_PORT:-6666}
RANK=0

if [ "$NPROC_PER_NODE" -le 0 ]; then
    NPROC_PER_NODE=1
fi

llama_ckpt_path=/nethome/rkhan96/flash/weights/Llama-2-7b-chat-hf

# Training Arguments
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune
export CROSS_ATTN_KV_MODE=${CROSS_ATTN_KV_MODE:-question}
export CROSS_MODAL_MODE=${CROSS_MODAL_MODE:-trilinear}
MODEL_NAME=${MODEL_NAME:-llama2}
DATASET=${DATASET:-ave}
PRECISION=${PRECISION:-bf16}
GRAD_SENS_RUN=${GRAD_SENS_RUN:-0}
BLC_WEIGHT=${BLC_WEIGHT:-1}
TRILINEAR_PACK_TOKENS=${TRILINEAR_PACK_TOKENS:-False}
LOGGING_STEPS=${LOGGING_STEPS:-10}

# Map fusion mode to a short tag for naming:
#   - pairwise  -> moka
#   - trilinear -> trilinear
if [ "$CROSS_MODAL_MODE" = "pairwise" ]; then
    CROSS_MODAL_TAG="moka"
else
    CROSS_MODAL_TAG="trilinear"
    # Triton trilinear uses more GPU memory; reduce batch size unless overridden.
    LOCAL_BATCH_SIZE=${TRILINEAR_BATCH:-2}
fi

# Epochs (default 3). Override via NUM_EPOCHS=1 for fast ablations.
NUM_EPOCHS=${NUM_EPOCHS:-3}

# Run dir: model_dataset_precision_question|full_text_<moka|trilinear>_[ablation_]epN_bwN_date_time
# ABLATION_TAG: optional extra suffix such as kernel or packed
RUN_NAME="${MODEL_NAME}_${DATASET}_${PRECISION}_${CROSS_ATTN_KV_MODE}_${CROSS_MODAL_TAG}"
[ -n "${ABLATION_TAG:-}" ] && RUN_NAME="${RUN_NAME}_${ABLATION_TAG}"
[ "$GRAD_SENS_RUN" = "1" ] && RUN_NAME="${RUN_NAME}_gradsense"
RUN_NAME="${RUN_NAME}_ep${NUM_EPOCHS}_bw${BLC_WEIGHT}_$(date +%Y%m%d_%H%M%S)"
export RUN_NAME
OUTP_DIR=results
OUTPUT_DIR=${OUTPUT_DIR:-$OUTP_DIR/$WANDB_PROJECT/$RUN_NAME}
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Mirror Slurm stdout/stderr into the run directory only when the scheduler
# is not already writing directly to those files. This avoids duplicate lines.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    desired_stdout="$(readlink -f "$OUTPUT_DIR/slurm-${SLURM_JOB_ID}.out" 2>/dev/null || printf '%s' "$OUTPUT_DIR/slurm-${SLURM_JOB_ID}.out")"
    desired_stderr="$(readlink -f "$OUTPUT_DIR/slurm-${SLURM_JOB_ID}.err" 2>/dev/null || printf '%s' "$OUTPUT_DIR/slurm-${SLURM_JOB_ID}.err")"
    current_stdout="$(readlink -f "/proc/$$/fd/1" 2>/dev/null || true)"
    current_stderr="$(readlink -f "/proc/$$/fd/2" 2>/dev/null || true)"

    if [ "$current_stdout" != "$desired_stdout" ]; then
        exec > >(tee -a "$OUTPUT_DIR/slurm-${SLURM_JOB_ID}.out")
    fi
    if [ "$current_stderr" != "$desired_stderr" ]; then
        exec 2> >(tee -a "$OUTPUT_DIR/slurm-${SLURM_JOB_ID}.err" >&2)
    fi
fi

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-deepspeed/stage2-offload.json}
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
SEED=${SEED:-42}

torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/finetune.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --llm_name llama \
    --reserved_modality None \
    --loramethod train \
    --cross_attn_kv_mode $CROSS_ATTN_KV_MODE \
    --cross_modal_mode $CROSS_MODAL_MODE \
    --model_name_or_path $llama_ckpt_path \
    --exp_desc "baseline" \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 444 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --blc_weight $BLC_WEIGHT \
    --blc_alpha 1 \
    --trilinear_pack_tokens $TRILINEAR_PACK_TOKENS \
    --bf16 $([ "$PRECISION" = "bf16" ] && echo True || echo False) \
    --tf32 False \
    --fp16 False \
    --avqa_task False \
    --ave_task True \
    --save_modules vl_projector,al_projector,lora \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path /nethome/rkhan96/flash/weights/clip-vit-large-patch14 \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /nethome/rkhan96/flash/weights/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_find_unused_parameters True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 150 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps $LOGGING_STEPS \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --dataloader_num_workers 4 \
    --grad_sensitivity_enable $([ "$GRAD_SENS_RUN" = "1" ] && echo True || echo False) \
    --grad_sensitivity_include_projectors $([ "$GRAD_SENS_RUN" = "1" ] && echo True || echo False) \
    --report_to tensorboard \
