#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_PORT=6668
RANK=0

llama_ckpt_path=/dockerdata/Llama-2-7b-chat-hf
YOUR_CKPT_PARH=AVQA_checkpoint

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
    --model_name_or_path $llama_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 444 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --blc_weight 1 \
    --blc_alpha 1 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --ckpt_dir $YOUR_CKPT_PARH \
    --avqa_task True \
    --ave_task False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path /dockerdata/clip-vit-large-patch14 \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /dockerdata/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir 'not_used' \

