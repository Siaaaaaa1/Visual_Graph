set -x ENGINE=${1:-vllm} 
# ===== 基础环境 =====
 #export VLLM_ATTENTION_BACKEND=XFORMERS 
export VLLM_ATTENTION_BACKEND=FLASH_ATTN 
export TOKENIZERS_PARALLELISM=false 
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
 # ===== 资源与规模（A800 安全配置）===== 
num_cpus_per_env_worker=0.1 
train_data_size=1 
val_data_size=4 
group_size=1 
# # ===== 数据准备（只是元信息占位，不会吃 image）===== 
# python3 -m examples.data_preprocess.prepare \ 
# --mode 'text' \ 
# --train_data_size $train_data_size \ 
# --val_data_size $val_data_size 
# ===== 正式训练 ===== 
python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
\
data.train_files=./gra2img/datasets/pubmed_train_slim.parquet \
data.val_files=./gra2img/datasets/pubmed_test_slim.parquet \
data.train_batch_size=$train_data_size \
data.val_batch_size=$val_data_size \
data.max_prompt_length=1024 \
data.max_response_length=128 \
data.filter_overlong_prompts=True \
data.truncation='right' \
data.return_raw_chat=True \
\
actor_rollout_ref.model.path=./model/Qwen3-VL-4B-Instruct \
+actor_rollout_ref.model.model_type=vision_language \
actor_rollout_ref.rollout.max_model_len=2048 \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
\
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.rollout.free_cache_engine=True \
+actor_rollout_ref.actor.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=1 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
+actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.01 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.use_invalid_action_penalty=True \
actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
\
actor_rollout_ref.rollout.name=vllm \
+actor_rollout_ref.actor.fsdp.cpu_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.enable_chunked_prefill=True \
actor_rollout_ref.rollout.enforce_eager=True \
actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
\
algorithm.use_kl_in_reward=False \
\
env.env_name=graph_search/GraphSearchEnv \
env.seed=0 \
env.max_steps=5 \
+env.node_text_path=./gra2img/origin_datasets/pubmed_text.json \
+env.dataset_name=pubmed \
+env.dataset_dir=./gra2img/datasets \
env.history_length=3 \
env.rollout.n=$group_size \
env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
\
trainer.logger=['console'] \
trainer.project_name='verl_graph_search_debug' \
trainer.experiment_name='graph_search_qwen3_vl4b' \
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
trainer.save_freq=-1 \
trainer.total_epochs=5 \
trainer.test_freq=-1 \
trainer.val_before_train=False \