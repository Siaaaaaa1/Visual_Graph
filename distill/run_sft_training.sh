#!/bin/bash
# 用法: bash run_sft_training.sh 8 ./output_dir

nproc_per_node=$1
save_path=$2

# 自动合并当前目录下所有的蒸馏 parquet 文件
TRAIN_FILES=$(ls *_training.parquet | tr '\n' ',' | sed 's/,$//')

echo "[INFO] 训练数据集: ${TRAIN_FILES}"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="[${TRAIN_FILES}]" \
    data.prompt_key=messages \
    data.response_key=target \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=4096 \
    model.partial_pretrain=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=graph-search-distill \
    trainer.experiment_name=multimodal-graph-agent \
    trainer.total_epochs=3 \
    trainer.logger=['console','wandb'] \
    optim.lr=2e-5 $@