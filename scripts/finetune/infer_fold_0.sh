#!/bin/bash

llama2_ckpt_path=llama2-7b-chat-hf
qwen2_ckpt_path=Qwen2-7B-Instruct   

# Training Arguments
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
RUN_NAME=test
OUTP_DIR=results

export CUDA_VISIBLE_DEVICES='0'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export NCCL_P2P_DISABLE=NVL
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python scripts/finetune/inference_cut.py \
    --llm_name qwen \
    --reserved_modality None \
    --loramethod test \
    --cut_folds 0 \
    --model_name_or_path $qwen2_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 44 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 False \
    --tf32 False \
    --fp16 False \
    --ckpt_dir moka/checkpoint-1198 \
    --avvp_task True \
    --adapter_ckpt_path none \
    --test_name test \
    --device cuda:0 \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path clip-vit-large-patch14 \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch False \
    --BEATs_ckpt_path BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --seg_branch False \
    --prompt_embed_dim 256 \
    --mask_decoder_transformer_depth 2 \
    --low_res_mask_size 128 \
    --image_scale_nums 2 \
    --token_nums_per_scale 3 \
    --avs_query_num 128 \
    --num_classes 1 \
    --query_generator_num_layers 2 \
    --output_dir 'results'

