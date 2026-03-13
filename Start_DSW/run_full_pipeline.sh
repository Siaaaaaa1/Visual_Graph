#!/bin/bash

# 遇到任何非 0 退出码的错误立即中断整个流水线，并捕获管道中的错误
set -eo pipefail

echo "========================================================"
echo "[START] 开始执行全链路训练 Pipeline (Distill -> SFT -> RL)"
echo "========================================================"

# ==========================================
# 0. 环境准备
# ==========================================
echo "[INFO] 准备环境: 激活 Conda 环境 verl-agent..."
source /mnt/workspace/haowengao/miniconda3/bin/activate verl-agent
cd /mnt/workspace/haowengao/Visual_Graph

# ==========================================
# 1. 蒸馏阶段 (Distill)
# ==========================================
echo "--------------------------------------------------------"
echo "[INFO] 步骤 1/3: 开始执行全局数据蒸馏..."
bash distill/run_distill_pipeline.sh

# ==========================================
# 2. 监督微调阶段 (SFT)
# ==========================================
echo "--------------------------------------------------------"
echo "[INFO] 步骤 2/3: 开始执行 SFT 阶段 (Arxiv, Pubmed, Cora)..."
bash distill/run_sft_arxiv.sh
bash distill/run_sft_pubmed.sh
bash distill/run_sft_cora.sh

# ==========================================
# 3. 强化学习阶段 (RL / Thinking)
# ==========================================
echo "--------------------------------------------------------"
echo "[INFO] 步骤 3/3: 开始执行 RL 阶段 (Arxiv, Cora, Pubmed)..."
bash Our_examples/run_arxiv_Graph_4B_Thinking.sh
bash Our_examples/run_cora_Graph_4B_Thinking.sh
bash Our_examples/run_pubmed_Graph_4B_Thinking.sh

echo "========================================================"
echo "[SUCCESS] 所有数据集的 Pipeline 已全部顺利执行完毕！"
echo "========================================================"