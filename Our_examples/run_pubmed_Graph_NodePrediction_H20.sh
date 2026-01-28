set -x
# --- 1. 显存碎片优化 (保持开启) ---
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export WANDB_API_KEY="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"
export MASTER_ADDRESS=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+')
export WORLD_SIZE=8

ENGINE=${1:-vllm}
num_cpus_per_env_worker=0.5

# --- H20 优化参数 ---
train_data_size=64
val_data_size=256
group_size=8
MODEL_PATH="./models/Qwen3-VL-4B-Instruct"

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
    echo "错误: 脚本文件名 '$SCRIPT_NAME' 中未包含指定的数据集名称 (pubmed, cora, arxiv)。"
    exit 1
elif [ $MATCH_COUNT -gt 1 ]; then
    echo "错误: 脚本文件名 '$SCRIPT_NAME' 中包含多个数据集名称，请只保留一个。"
    exit 1
fi

echo "✅ 检测到数据集: $DETECTED_DATASET"

# --- 4. 路径设置 ---
NODE_TEXT_PATH="./datasets/${DETECTED_DATASET}_text.json"
TRAIN_FILE="./datasets/${DETECTED_DATASET}_train_slim.parquet"
VAL_FILE="./datasets/${DETECTED_DATASET}_test_slim.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=graph_search/GraphSearchEnv \
    env.seed=0 \
    env.max_steps=50 \
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
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=false \
    ray_init.num_cpus=64 \
    actor_rollout_ref.rollout.dtype=bfloat16