#!/bin/bash
# 蒸馏流水线：使用 DashScope 云端 API（qwen3-vl-plus）对三个数据集依次执行蒸馏
# 凭据从项目根目录的 .env 文件读取（distill_data.py 内 load_dotenv() 自动加载）
# 用法: bash distill/run_distill_pipeline.sh（需在 Visual_Graph/ 根目录下运行）

set -e  # 任意步骤失败立即退出

# 加载 .env（校验 DashScope 凭据是否就绪）
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

if [ -z "${DASHSCOPE_API_KEY}" ]; then
    echo "[ERROR] DASHSCOPE_API_KEY 未设置。请在项目根目录 .env 中写入 DASHSCOPE_API_KEY=your_key"
    exit 1
fi

DATASET_DIR="datasets"

# -------- 蒸馏参数 --------
# trajectories_per_node  : 每个节点最多收集的成功轨迹数（多条 = 多种推理路径 = 更高多样性）
# max_attempts           : 每个节点的总尝试上限，超过后放弃并写入 debug_failures.jsonl
#
# max_hard_per_class     : 困难样本（首次成功 attempt > 5）每类上限（0=不限，尽量全收）
#                          困难样本是模型能力提升的核心，不应被过度截断
# max_easy_per_class     : 简单样本（首次成功 attempt <= 5）每类上限
#                          保留适量简单样本防止遗忘基础技能，但不能淹没困难样本
#
# 数据量估算（trajectories_per_node=3）：
#   cora  (~7 类,  ~2708 节点): 困难样本不限 + 简单 100/类  → 困难优先全量收集
#   pubmed (~3 类, ~19717 节点): 困难样本不限 + 简单 500/类  → 困难优先全量收集
#   arxiv (~40 类, ~169343 节点): 困难样本不限 + 简单 200/类 → 困难优先全量收集

TRAJECTORIES_PER_NODE=3
MAX_ATTEMPTS=15

# 困难样本不设上限（0=不限）
CORA_MAX_HARD_PER_CLASS=0
PUBMED_MAX_HARD_PER_CLASS=0
ARXIV_MAX_HARD_PER_CLASS=0

# 简单样本适量保留（约为预期困难样本量的 30-50%）
CORA_MAX_EASY_PER_CLASS=100
PUBMED_MAX_EASY_PER_CLASS=500
ARXIV_MAX_EASY_PER_CLASS=200

echo "========================================================"
echo "[START] 启动异步蒸馏 (执行路径: $(pwd))"
echo "  策略：困难样本不限量全收，简单样本适量保留"
echo "========================================================"

# 1. 蒸馏 cora（小数据集，困难样本全量，简单样本每类 ${CORA_MAX_EASY_PER_CLASS} 条）
echo "--------------------------------------------------------"
echo "[PROCESS] 1/3: 蒸馏 cora（hard=不限, easy/class=${CORA_MAX_EASY_PER_CLASS}, traj/node=${TRAJECTORIES_PER_NODE}）"
python distill/distill_data.py \
    --dataset cora \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_hard_per_class ${CORA_MAX_HARD_PER_CLASS} \
    --max_easy_per_class ${CORA_MAX_EASY_PER_CLASS} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

# 2. 蒸馏 pubmed
echo "--------------------------------------------------------"
echo "[PROCESS] 2/3: 蒸馏 pubmed（hard=不限, easy/class=${PUBMED_MAX_EASY_PER_CLASS}, traj/node=${TRAJECTORIES_PER_NODE}）"
python distill/distill_data.py \
    --dataset pubmed \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_hard_per_class ${PUBMED_MAX_HARD_PER_CLASS} \
    --max_easy_per_class ${PUBMED_MAX_EASY_PER_CLASS} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

# 3. 蒸馏 arxiv（大数据集，困难样本不限，简单样本每类 ${ARXIV_MAX_EASY_PER_CLASS} 条）
echo "--------------------------------------------------------"
echo "[PROCESS] 3/3: 蒸馏 arxiv（hard=不限, easy/class=${ARXIV_MAX_EASY_PER_CLASS}, traj/node=${TRAJECTORIES_PER_NODE}）"
python distill/distill_data.py \
    --dataset arxiv \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_hard_per_class ${ARXIV_MAX_HARD_PER_CLASS} \
    --max_easy_per_class ${ARXIV_MAX_EASY_PER_CLASS} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

echo "========================================================"
echo "[FINISH] 所有数据集蒸馏完成！样本图片已保存至 distill/*_samples/"
echo "生成文件: distill/{cora,pubmed,arxiv}_vgraph_training.parquet"
