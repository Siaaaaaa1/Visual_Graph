#!/bin/bash
# 用法: bash distill/run_sft_training.sh 8 distill/output_dir
# 需在项目根目录（Visual_Graph/）下运行

nproc_per_node=$1
save_path=$2

# ================= 参数校验 =================
if [ -z "$nproc_per_node" ] || [ -z "$save_path" ]; then
    echo "用法: bash distill/run_sft_training.sh <nproc_per_node> <save_path>"
    exit 1
fi

# ================= 从 .env 加载凭据 =================
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

if [ -z "${WANDB_API_KEY}" ]; then
    echo "[ERROR] WANDB_API_KEY 未设置。请在项目根目录 .env 中写入 WANDB_API_KEY=your_key"
    exit 1
fi

export WANDB_ENTITY="zzy_szsh"

# 在 verl-agent 根目录下寻找 distill 文件夹中的 parquet 文件
TRAIN_FILES=$(ls distill/*_training.parquet 2>/dev/null | tr '\n' ',' | sed 's/,$//')

if [ -z "$TRAIN_FILES" ]; then
    echo "[ERROR] 未在 distill 目录下找到任何 *_training.parquet 数据集文件。"
    exit 1
fi

echo "[INFO] 训练数据集: ${TRAIN_FILES}"

# 注意：蒸馏数据的 Parquet 列为 messages（OpenAI 多轮对话格式），无独立 target 列
# verl fsdp_sft_trainer 在 prompt_key=messages 时自动将最后一条 assistant 消息作为训练目标
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="[${TRAIN_FILES}]" \
    data.prompt_key=messages \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=4096 \
    model.partial_pretrain=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=graph-search-distill \
    trainer.experiment_name=multimodal-graph-agent \
    trainer.total_epochs=3 \
    trainer.logger=['console','wandb'] \
    optim.lr=2e-5 $@