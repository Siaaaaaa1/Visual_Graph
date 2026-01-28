#!/bin/bash
set -x

# =========================================================
# 1. 环境清理与保护 (对应您成功脚本中的 trap 和清理逻辑)
# =========================================================
# 停止旧的 Ray 进程，防止端口冲突
ray stop --force 2>/dev/null
pkill -f "raylet"

# =========================================================
# 2. 核心修复：限制 Ray 资源 (借鉴您成功脚本的配置)
# =========================================================
# 【关键】防止 Ray 在 100+ 核的机器上自动从那个启动过多进程导致崩溃
# 根据您的成功脚本，设置为 64 是安全的
export RAY_NUM_CPUS=64

# 显存保护：给 Ray 留出显存空间，防止 vLLM 占满导致 OOM
export VLLM_GPU_MEMORY_UTILIZATION=0.6

# =========================================================
# 3. 核心修复：网络与存储 (解决 CephFS 卡住问题)
# =========================================================
# 屏蔽代理，确保 Ray 本地通信畅通
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export NO_PROXY=localhost,127.0.0.1,::1

# 【关键】将 Ray 临时目录指向本地磁盘 /tmp，而不是 CephFS
# CephFS 不支持 Socket 文件，这是导致“卡住”的常见原因
export RAY_TEMP_DIR="/tmp/ray_results_$(date +%s)"
mkdir -p "$RAY_TEMP_DIR"
echo "Ray temporary directory set to: $RAY_TEMP_DIR"

# =========================================================
# 4. 任务参数配置
# =========================================================
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="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"
export VLLM_LOGGING_LEVEL=INFO

# 确保 Checkpoint 目录存在
mkdir -p ./checkpoints

num_cpus_per_env_worker=0.5
train_data_size=16
val_data_size=128
group_size=8

# =========================================================
# 5. 启动训练
# =========================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./datasets/pubmed_train_slim.parquet \
    data.val_files=./datasets/pubmed_test_slim.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=./models/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=graph_search/GraphSearchEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_graph_search' \
    trainer.experiment_name='graph_search_qwen2.5_7b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=50 \
    trainer.total_epochs=150 \
    trainer.val_before_train=false \
    env.node_text_path=./datasets/pubmed_text.json \
    env.dataset_name='pubmed' \
    ray_init.num_cpus=128 