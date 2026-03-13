#!/bin/bash

# 遇到任何非 0 退出码的错误立即中断整个流水线
set -e

echo "========================================================"
echo "[START] 开始执行全链路训练 Pipeline (Distill -> SFT -> RL)"
echo "========================================================"

# 1. 激活 Conda 环境
echo "[INFO] 激活 Conda 环境: verl-agent..."
source /mnt/workspace/haowengao/miniconda3/bin/activate verl-agent

# 2. 蒸馏阶段
echo "--------------------------------------------------------"
echo "[INFO] 步骤 1: 开始执行全局数据蒸馏..."
bash distill/run_distill_pipeline.sh

# 3. Arxiv 数据集 (SFT + RL)
echo "--------------------------------------------------------"
echo "[INFO] 步骤 2: 开始 Arxiv 数据集训练..."
bash distill/run_sft_arxiv.sh
bash Our_examples/run_arxiv_Graph_4B_Thinking.sh

# 4. Cora 数据集 (SFT + RL)
echo "--------------------------------------------------------"
echo "[INFO] 步骤 3: 开始 Cora 数据集训练..."
bash distill/run_sft_cora.sh
bash Our_examples/run_cora_Graph_4B_Thinking.sh

# 5. Pubmed 数据集 (SFT + RL)
echo "--------------------------------------------------------"
echo "[INFO] 步骤 4: 开始 Pubmed 数据集训练..."
bash distill/run_sft_pubmed.sh
bash Our_examples/run_pubmed_Graph_4B_Thinking.sh

echo "========================================================"
echo "[SUCCESS] 所有数据集的 Pipeline 已全部顺利执行完毕！"
echo "========================================================"