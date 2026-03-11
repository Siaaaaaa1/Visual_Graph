#!/bin/bash
# 蒸馏流水线：使用 DashScope 云端 API（qwen-vl-max）对三个数据集依次执行蒸馏
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
# trajectories_per_node : 每个节点最多收集的成功轨迹数（多条 = 多种推理路径 = 更高多样性）
# max_attempts          : 每个节点的总尝试上限，超过后放弃并写入 debug_failures.jsonl
# max_per_class         : 每个类别标签最多保留的轨迹总条数（0=不限，用于控制类别平衡和总体量）
#
# 数据量估算：
#   cora  ~7  类 × 200 条/类 × ~3 条/节点 → 约  1400 条
#   pubmed ~3 类 × 500 条/类 × ~3 条/节点 → 约  1500 条
#   arxiv ~40 类 × 300 条/类 × ~3 条/节点 → 约 12000 条

TRAJECTORIES_PER_NODE=3
MAX_ATTEMPTS=15

CORA_MAX_PER_CLASS=200
PUBMED_MAX_PER_CLASS=500
ARXIV_MAX_PER_CLASS=300

echo "========================================================"
echo "[START] 启动异步蒸馏 (执行路径: $(pwd))"
echo "========================================================"

# 1. 蒸馏 cora（小数据集，全量节点，按类配额）
echo "--------------------------------------------------------"
echo "[PROCESS] 1/3: 蒸馏 cora（max_per_class=${CORA_MAX_PER_CLASS}, traj/node=${TRAJECTORIES_PER_NODE}）"
python distill/distill_data.py \
    --dataset cora \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_per_class ${CORA_MAX_PER_CLASS} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

# 2. 蒸馏 pubmed
echo "--------------------------------------------------------"
echo "[PROCESS] 2/3: 蒸馏 pubmed（max_per_class=${PUBMED_MAX_PER_CLASS}, traj/node=${TRAJECTORIES_PER_NODE}）"
python distill/distill_data.py \
    --dataset pubmed \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_per_class ${PUBMED_MAX_PER_CLASS} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

# 3. 蒸馏 arxiv（大数据集，按类配额严格控制总量）
echo "--------------------------------------------------------"
echo "[PROCESS] 3/3: 蒸馏 arxiv（max_per_class=${ARXIV_MAX_PER_CLASS}, traj/node=${TRAJECTORIES_PER_NODE}）"
python distill/distill_data.py \
    --dataset arxiv \
    --num_tasks 100000 \
    --dataset_dir ${DATASET_DIR} \
    --max_per_class ${ARXIV_MAX_PER_CLASS} \
    --trajectories_per_node ${TRAJECTORIES_PER_NODE} \
    --max_attempts ${MAX_ATTEMPTS}

echo "========================================================"
echo "[FINISH] 所有数据集蒸馏完成！样本图片已保存至 distill/*_samples/"
echo "生成文件: distill/{cora,pubmed,arxiv}_vgraph_training.parquet"
