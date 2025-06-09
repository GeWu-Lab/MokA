#!/bin/bash



# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_PORT=6666
RANK=0

llama_ckpt_path=llama2-7b-chat-hf

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
export CUDA_VISIBLE_DEVICES='0'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'


########### Arguments ###########
#################################

# lora_r: 444 means three LoRA_A with rank $4$. If there are two modalities, please set as "44".
# blc_weight: control the weight of cross-attention ouput, range [0,1].
# ckpt_dir: the path of the checkpoint to be evaluated.
# vit_ckpt_path: the path of the pre-trained ViT checkpoint.
# BEATs_ckpt_path: the path of the pre-trained BEATs checkpoint.
# avqa_task: set True to train on AVQA task

#################################



python scripts/finetune/inference_cut.py \
    --llm_name llama \
    --reserved_modality None \
    --loramethod test \
    --cut_folds 0 \
    --model_name_or_path $llama_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 444 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --blc_weight 1 \
    --blc_alpha 1 \
    --bf16 False \
    --tf32 False \
    --fp16 False \
    --ckpt_dir music_avqa/checkpoint-675 \
    --avqa_task True \
    --ave_task False \
    --device cuda:0 \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path clip-vit-large-patch14 \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir 'not_used' \

