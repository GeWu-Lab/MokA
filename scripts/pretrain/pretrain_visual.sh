#!/bin/bash


echo "pretrain"



llama2_ckpt_path=llama2-7b-chat-hf
qwen2_ckpt_path=Qwen2-7B-Instruct   

# Training Arguments
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# Log Arguments
RUN_NAME=qwen_visual_qformer
OUTP_DIR=results
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

torchrun \
    scripts/pretrain/pretrain.py \
    --deepspeed deepspeed/stage2-offload.json \
    --llm_name qwen \
    --model_name_or_path $qwen2_ckpt_path \
    --freeze_backbone True \
    --lora_enable False \
    --bits 32 \
    --bf16 False \
    --tf32 False \
    --fp16 False \
    --visual_branch True \
    --image_caption_task True \
    --video_caption_task True \
    --video_frame_nums 8 \
    --vit_ckpt_path clip-vit-large-patch14 \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch False \
    --audio_caption_task False \
    --BEATs_ckpt_path BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --seg_branch False \
    --segmentation_task False \
    --prompt_embed_dim 256 \
    --mask_decoder_transformer_depth 2 \
    --low_res_mask_size 128 \
    --image_scale_nums 2 \
    --token_nums_per_scale 3 \
    --ce_loss_weight 1.0 \
    --dice_loss_weight 0.5 \
    --bce_loss_weight 2.0 \
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
    --report_to tensorboard >> "${OUTPUT_LOG}" 2>&1

