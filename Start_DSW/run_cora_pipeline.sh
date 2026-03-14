#!/bin/bash
# Pipeline: Distill -> SFT -> RL (cora 数据集，8 GPU)
# 用法: bash Start_DSW/run_cora_pipeline.sh（需在 Visual_Graph/ 根目录运行）

set -eo pipefail

DATASET="cora"

echo "========================================================"
echo "[${DATASET}] 开始 Distill -> SFT -> RL 全链路训练"
echo "========================================================"

if [ -f ".env" ]; then set -a; source .env; set +a; fi
if [ -z "${WANDB_API_KEY}" ]; then echo "[ERROR] WANDB_API_KEY 未设置"; exit 1; fi
if [ -z "${DASHSCOPE_API_KEY}" ]; then echo "[ERROR] DASHSCOPE_API_KEY 未设置"; exit 1; fi

# ==========================================
# 1. 蒸馏阶段
# ==========================================
echo "[INFO] 步骤 1/3: 开始数据蒸馏（hard=不限, easy/class=100）..."
python distill/distill_data.py \
    --dataset ${DATASET} \
    --num_tasks 100000 \
    --dataset_dir datasets \
    --max_hard_per_class 0 \
    --max_easy_per_class 100 \
    --trajectories_per_node 3 \
    --max_attempts 15

# ==========================================
# 2. SFT 阶段
# ==========================================
echo "[INFO] 步骤 2/3: 开始 SFT 训练..."
bash distill/run_sft_${DATASET}.sh

# ==========================================
# 3. 查找最新 SFT checkpoint
# ==========================================
SFT_CKPT=$(ls -d ./checkpoints/sft_${DATASET}/global_step_* 2>/dev/null | sort -V | tail -1)
if [ -z "${SFT_CKPT}" ]; then
    echo "[ERROR] 未找到 SFT checkpoint，请检查 SFT 训练是否成功。"
    exit 1
fi
echo "[INFO] 使用 SFT checkpoint: ${SFT_CKPT}"

# ==========================================
# 3. RL 阶段（接续 SFT 权重）
# ==========================================
echo "[INFO] 步骤 3/3: 开始 RL 训练（GRPO）..."

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VERL_PPO_ASYNC_ROLLOUT=1
export MASTER_ADDRESS=127.0.0.1
export WORLD_SIZE=8
export WANDB_ENTITY="zzy_szsh"

ENGINE=vllm
train_data_size=64
val_data_size=256
group_size=8
ppo_mini_batch_size=256
micro_batch_size=2
log_prob_batch_size=2
num_cpus_per_env_worker=1.0

SCRIPT_NAME="run_${DATASET}_Graph_4B_Thinking"
EXP_DATE=$(date +%m%d%H)
EXPERIMENT_NAME="${SCRIPT_NAME}_sft2rl_${EXP_DATE}"

TRAIN_FILE="./datasets/${DATASET}_train_slim.parquet"
VAL_FILE="./datasets/${DATASET}_test_slim.parquet"
NODE_TEXT_PATH="./datasets/${DATASET}_text.json"
LOG_FILE="log_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_CKPT \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=True \
    env.env_name=graph_search/GraphSearchEnv \
    env.seed=42 \
    env.max_steps=10 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.node_text_path=$NODE_TEXT_PATH \
    env.dataset_name=$DATASET \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_graph_node_prediction' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=50 \
    trainer.val_before_train=false \
    ray_init.num_cpus=64 \
    trainer.resume_mode=disable \
    actor_rollout_ref.rollout.dtype=bfloat16 2>&1 | tee >(sed -u -E 's/\x1b\[[0-9;]*m//g; s/\((WorkerDict|TaskRunner) pid=[0-9]*\)//g' > "$LOG_FILE")

echo "========================================================"
echo "[${DATASET}] Pipeline 完成！"
echo "========================================================"
