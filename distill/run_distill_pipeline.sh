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
# hard_ratio      : hard 总量 / easy 总量的上限比例
#                   cora/arxiv=1.0（1:1），pubmed=0.5（hard 为 easy 的一半）
#                   配合 stale_stop，所有配额满后自动取消剩余任务
# max_easy_per_class : 每类简单样本上限
# stale_stop      : 连续 N 个成功节点全被拒绝时触发 early-stop（安全网）
#
# 数据量估算（trajectories_per_node=3）：
#   cora  (~7 类,  ~2708 节点): easy=700  hard≤700  total≤1400
#   pubmed (~3 类, ~19717 节点): easy=1500 hard≤750  total≤2250
#   arxiv (~40 类, ~169343 节点): easy=8000 hard≤8000 total≤16000

TRAJECTORIES_PER_NODE=3
MAX_ATTEMPTS=15

CORA_MAX_EASY_PER_CLASS=100
PUBMED_MAX_EASY_PER_CLASS=500
ARXIV_MAX_EASY_PER_CLASS=200

CORA_HARD_RATIO=1.0
PUBMED_HARD_RATIO=0.5
ARXIV_HARD_RATIO=1.0

echo "========================================================"
echo "[START] 启动异步蒸馏 (执行路径: $(pwd))"
echo "  策略：easy 每类限额，hard 按全局比例限制，超额自动停止"
echo "========================================================"

# 1. 蒸馏 cora
echo "--------------------------------------------------------"
echo "[PROCESS] 1/3: 蒸馏 cora（hard_ratio=${CORA_HARD_RATIO}, easy/class=${CORA_MAX_EASY_PER_CLASS}）"
python distill/distill_data.py \
    --dataset cora \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_easy_per_class ${CORA_MAX_EASY_PER_CLASS} \
    --hard_ratio ${CORA_HARD_RATIO} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

# 2. 蒸馏 pubmed
echo "--------------------------------------------------------"
echo "[PROCESS] 2/3: 蒸馏 pubmed（hard_ratio=${PUBMED_HARD_RATIO}, easy/class=${PUBMED_MAX_EASY_PER_CLASS}）"
python distill/distill_data.py \
    --dataset pubmed \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_easy_per_class ${PUBMED_MAX_EASY_PER_CLASS} \
    --hard_ratio ${PUBMED_HARD_RATIO} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

# 3. 蒸馏 arxiv
echo "--------------------------------------------------------"
echo "[PROCESS] 3/3: 蒸馏 arxiv（hard_ratio=${ARXIV_HARD_RATIO}, easy/class=${ARXIV_MAX_EASY_PER_CLASS}）"
python distill/distill_data.py \
    --dataset arxiv \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_easy_per_class ${ARXIV_MAX_EASY_PER_CLASS} \
    --hard_ratio ${ARXIV_HARD_RATIO} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

echo "========================================================"
echo "[FINISH] 所有数据集蒸馏完成！样本图片已保存至 distill/*_samples/"
echo "生成文件: distill/{cora,pubmed,arxiv}_vgraph_training.parquet"
