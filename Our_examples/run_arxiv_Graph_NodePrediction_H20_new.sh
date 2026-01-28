set -x
# --- 1. 基础环境设置 ---
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VERL_PPO_ASYNC_ROLLOUT=1 

export WANDB_API_KEY="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"
export MASTER_ADDRESS=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+')
export WORLD_SIZE=8

ENGINE=${1:-vllm}
num_cpus_per_env_worker=1.0

# --- H20 优化参数 ---
train_data_size=32
val_data_size=256
group_size=8
MODEL_PATH="./models/Qwen3-VL-4B-Instruct"

ppo_mini_batch_size=1024
# 保持为 4，关闭 remove_padding 后显存压力变大，4 是 H20 的安全水位
micro_batch_size=2

# --- 2. 脚本信息获取 ---
SCRIPT_NAME=$(basename "$0" .sh)
EXP_DATE=$(date +%m%d%H)
EXPERIMENT_NAME="${SCRIPT_NAME}_${EXP_DATE}"

# --- 3. 动态检测数据集 ---
KEYWORDS=("pubmed" "cora" "arxiv")
MATCH_COUNT=0
DETECTED_DATASET=""
for KEY in "${KEYWORDS[@]}"; do
    if [[ "$SCRIPT_NAME" == *"$KEY"* ]]; then
        DETECTED_DATASET="$KEY"
        ((MATCH_COUNT++))
    fi
done

if [ $MATCH_COUNT -eq 0 ]; then
    echo "错误: 数据集名称不匹配。"
    exit 1
fi

# --- 4. 路径设置 ---
NODE_TEXT_PATH="./datasets/${DETECTED_DATASET}_text.json"
TRAIN_FILE="./datasets/${DETECTED_DATASET}_train_slim.parquet"
VAL_FILE="./datasets/${DETECTED_DATASET}_test_slim.parquet"

# --- 5. 日志设置 ---
LOG_FILE="log_$(date +%Y%m%d_%H%M%S).log"

# --- 6. 执行训练 ---
set +x
set -o pipefail 

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=5120 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=graph_search/GraphSearchEnv \
    env.seed=42 \
    env.max_steps=10 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.node_text_path=$NODE_TEXT_PATH \
    env.dataset_name=$DETECTED_DATASET \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_graph_node_prediction' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.val_before_train=false \
    ray_init.num_cpus=64 \
    actor_rollout_ref.rollout.dtype=bfloat16 2>&1 | tee >(sed -u -E 's/\x1b\[[0-9;]*m//g; s/\((WorkerDict|TaskRunner) pid=[0-9]*\)//g' > "$LOG_FILE")