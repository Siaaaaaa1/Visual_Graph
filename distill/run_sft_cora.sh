#!/bin/bash
# SFT 训练 —— cora 数据集
# 策略：全量使用（cora 本身较小），不做难度过滤
# 早停：连续 3 次验证集 loss 无改善时自动停止
# 用法: bash distill/run_sft_cora.sh（需在 Visual_Graph/ 根目录运行）

set -e
DATASET="cora"
PARQUET="distill/${DATASET}_vgraph_training.parquet"
VAL_FILE="datasets/${DATASET}_test_slim.parquet"
PROCESSED="/tmp/${DATASET}_sft_ready.parquet"

if [ -f ".env" ]; then set -a; source .env; set +a; fi
if [ -z "${WANDB_API_KEY}" ]; then echo "[ERROR] WANDB_API_KEY 未设置"; exit 1; fi
if [ ! -f "${PARQUET}" ]; then echo "[ERROR] 未找到 ${PARQUET}，请先运行蒸馏。"; exit 1; fi

export WANDB_ENTITY="zzy_szsh"

echo "[${DATASET}] 准备训练数据..."
python3 - <<EOF
import pandas as pd

df = pd.read_parquet("${PARQUET}")
print(f"原始数据: {len(df)} 条")
print(f"类别分布:\n{df['node_class'].value_counts().to_string()}")
print(f"难度分布: mean={df['difficulty_score'].mean():.3f}, "
      f"hard(>0.33)={( df['difficulty_score'] > 0.33).sum()} 条")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_parquet("${PROCESSED}", index=False)
print(f"最终训练集: {len(df)} 条 → ${PROCESSED}")
EOF

echo "[${DATASET}] 启动 SFT 训练（含早停监控）..."
python distill/early_stop_monitor.py --patience 3 --min_delta 0.001 -- \
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="[${PROCESSED}]" \
        data.val_files="[${VAL_FILE}]" \
        data.prompt_key=messages \
        data.micro_batch_size_per_gpu=2 \
        data.max_length=4096 \
        model.partial_pretrain=./models/Qwen3-VL-Plus-Instruct \
        trainer.default_local_dir=./checkpoints/sft_${DATASET} \
        trainer.project_name=graph-search-distill \
        trainer.experiment_name=sft-${DATASET} \
        trainer.total_epochs=10 \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.logger=['console','wandb'] \
        optim.lr=2e-5
