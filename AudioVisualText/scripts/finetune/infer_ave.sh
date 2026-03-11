#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MASTER_PORT=6668
RANK=0

llama_ckpt_path=/nethome/rkhan96/flash/weights/Llama-2-7b-chat-hf
YOUR_CKPT_PARH=${YOUR_CKPT_PARH:?Set YOUR_CKPT_PARH to the run directory (e.g. results/finetune/llama2_ave_bf16_question_moka_bw0.5_... or ..._question_trilinear_bw0.5_...)}
export CROSS_ATTN_KV_MODE=${CROSS_ATTN_KV_MODE:-question}
export CROSS_MODAL_MODE=${CROSS_MODAL_MODE:-trilinear}
PRECISION=${PRECISION:-bf16}
BLC_WEIGHT=${BLC_WEIGHT:-1}

if [ "$NPROC_PER_NODE" -le 0 ]; then
    NPROC_PER_NODE=1
fi

if [ -z "$YOUR_CKPT_PARH" ]; then
    echo "No checkpoint found. Set YOUR_CKPT_PARH manually."
    exit 1
fi

# Training Arguments
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune
RUN_NAME=test
OUTP_DIR=results
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'


torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/inference_cut.py \
    --llm_name llama \
    --reserved_modality None \
    --loramethod test \
    --cross_attn_kv_mode $CROSS_ATTN_KV_MODE \
    --cross_modal_mode $CROSS_MODAL_MODE \
    --model_name_or_path $llama_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 444 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --blc_weight $BLC_WEIGHT \
    --blc_alpha 1 \
    --bf16 $([ "$PRECISION" = "bf16" ] && echo True || echo False) \
    --tf32 False \
    --fp16 False \
    --ckpt_dir $YOUR_CKPT_PARH \
    --avqa_task False \
    --ave_task True \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path /nethome/rkhan96/flash/weights/clip-vit-large-patch14 \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /nethome/rkhan96/flash/weights/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir 'not_used'

# After inference, automatically run AVE evaluation and write metrics next to predictions.
python scripts/evaluation/ave_eval.py \
    --jsonl_path "$YOUR_CKPT_PARH/inference_results/inference_ave.jsonl" \
    --annotations_path "AVE_data/annotations.txt" \
    --output_metrics_path "$YOUR_CKPT_PARH/inference_results/metrics.jsonl" \
