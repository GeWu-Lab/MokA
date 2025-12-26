set -x

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_IB_TIMEOUT=24
export NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME=bond1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8192

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400   # 4H waitting
export TORCH_NCCL_ENABLE_MONITORING=0   # waitting for Infinity 

export ACCELERATE_BACKEND=nccl
export NCCL_TIMEOUT_MINS=30



export PYTHONPATH="${PYTHONPATH}:$(pwd)"



WANDB_API_KEY="your_wandb_api_key"
wandb login --relogin $WANDB_API_KEY


export WANDB_PROJECT=QWEN_MLLM_LORA
time_stamp=$(date +%Y%m%d)

BASE_RUN_NAME="moka_vl_${time_stamp}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

mkdir -p /tmp/wandb
export WANDB_CACHE_DIR=/tmp/wandb


OUTPUT_DIR=./checkpoints/moka_vl/${BASE_RUN_NAME}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


torchrun --nproc_per_node=8 \
  --nnodes=${WORLD_SIZE} \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port=${MASTER_PORT} \
  train/train.py \
  --output_dir ${OUTPUT_DIR} \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 0.1 \
  --save_total_limit 10 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --do_train True \
  --deepspeed "zero_stage2_config.json" \
  --report_to wandb \
  --run_name $BASE_RUN_NAME \
  --remove_unused_columns False