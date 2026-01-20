set -x
# --- 1. 显存碎片优化 (保持开启) ---
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
# 建议开启异步数据预取，减少 GPU 等待
export VERL_PPO_ASYNC_ROLLOUT=1 

export WANDB_API_KEY="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"
export MASTER_ADDRESS=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+')
export WORLD_SIZE=8

ENGINE=${1:-vllm}
num_cpus_per_env_worker=0.5

# --- H20 激进优化参数 ---
# 增加每轮采集的 Prompt 数量 (64 -> 256)，让8张卡有肉吃
train_data_size=256
val_data_size=256
group_size=8
MODEL_PATH="./models/Qwen3-VL-4B-Instruct"

# PPO 参数调整
# 全局 Batch = 256 * 8 = 2048 个样本
# Mini Batch 设为 512，则每次 Update 包含 2048/512 = 4 个 Mini Batches
ppo_mini_batch_size=512
# 单卡微批次从 8 提升到 16 或 24 (4B模型在H20上完全可以更大)
micro_batch_size=16

# ... (中间的脚本信息获取与数据集检测部分保持不变) ...

# --- 4. 路径设置 ---
NODE_TEXT_PATH="./datasets/${DETECTED_DATASET}_text.json"
TRAIN_FILE="./datasets/${DETECTED_DATASET}_train_slim.parquet"
VAL_FILE="./datasets/${DETECTED_DATASET}_test_slim.parquet"

# --- 5. 日志文件名生成 (log_年月日_时分秒.log) ---
LOG_FILE="log_$(date +%Y%m%d_%H%M%S).log"
echo "日志将输出到: $LOG_FILE"

# --- 6. 执行训练并同时输出到控制台和文件 ---
# set -o pipefail 确保如果 python 运行失败，脚本也会返回错误代码
set -o pipefail 

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
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
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
    actor_rollout_ref.rollout.dtype=bfloat16 2>&1 | tee "$LOG_FILE"