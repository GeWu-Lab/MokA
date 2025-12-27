
torchrun --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port=${MASTER_PORT} \
  pope.py \
  --model_path ckpt \
  --name_tag "final_test" \
  --deepspeed "zero_stage2_config.json" \
  --report_to wandb \
  --run_name $BASE_RUN_NAME \
  --remove_unused_columns False