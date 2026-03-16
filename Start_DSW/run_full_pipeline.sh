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
echo "[INFO] 步骤 2-3/3: 开始执行 SFT -> RL 全链路 (Arxiv, Pubmed, Cora)..."
# 蒸馏数据已在步骤 1 生成，各 pipeline 脚本会自动跳过蒸馏阶段
# 每个数据集独立完成 SFT（最优 checkpoint 按 gen_accuracy 选取）→ RL
bash Start_DSW/run_arxiv_pipeline.sh
bash Start_DSW/run_pubmed_pipeline.sh
bash Start_DSW/run_cora_pipeline.sh

echo "========================================================"
echo "[SUCCESS] 所有数据集的 Pipeline 已全部顺利执行完毕！"
echo "========================================================"