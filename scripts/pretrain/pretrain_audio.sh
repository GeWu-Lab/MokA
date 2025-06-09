#!/bin/bash


echo "pretrain"

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=4
MASTER_PORT=6666
RANK=0




llama_ckpt_path=llama2-7b-chat-hf

# Training Arguments
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=pretrain
RUN_NAME=audio_pretrain
OUTP_DIR=results
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"



########### Arguments ###########
#################################

# vit_ckpt_path: the path of the pre-trained ViT checkpoint.
# BEATs_ckpt_path: the path of the pre-trained BEATs checkpoint.


#################################



torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/pretrain/pretrain.py \
    --deepspeed deepspeed/stage2-offload.json \
    --llm_name llama \
    --model_name_or_path $llama_ckpt_path \
    --freeze_backbone True \
    --lora_enable False \
    --bits 32 \
    --bf16 False \
    --tf32 False \
    --fp16 False \
    --visual_branch False \
    --image_caption_task False \
    --video_caption_task False \
    --video_frame_nums 8 \
    --vit_ckpt_path clip-vit-large-patch14 \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --audio_caption_task True \
    --BEATs_ckpt_path BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --output_dir $OUTP_DIR/$WANDB_PROJECT/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_find_unused_parameters True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.33 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --dataloader_num_workers 4 \
    --report_to tensorboard