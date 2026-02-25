#!/bin/bash

# ================= 配置区 =================
# 全局相对路径（以 verl-agent 为根目录）
MODEL_PATH="Qwen3-VL-235B-A22B-Thinking"
DATASET_DIR="datasets"
SERVER_LOG="distill/vllm_server.log"

TP_SIZE=8           # GPU 数量
PORT=8080
DATASETS=("cora" "pubmed" "arxiv") # 需要处理的数据集列表

echo "========================================================"
echo "[START] 启动数据蒸馏全自动 Pipeline (Root: verl-agent)"
echo "========================================================"

# 1. 后台启动服务
# 提前设置环境变量，优化多进程并发内存分配
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn

echo "[STEP 1] 启动 vLLM 后端服务 (已开启极速加载模式)..."
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --tensor-parallel-size ${TP_SIZE} \
    --port ${PORT} \
    --max-model-len 16384 \
    --served-model-name qwen3-vl-teacher \
    --enforce-eager \
    --distributed-executor-backend mp > ${SERVER_LOG} 2>&1 &

VLLM_PID=$!
trap "kill -9 $VLLM_PID; echo '[EXIT] 服务已强制关闭'; exit" SIGINT SIGTERM EXIT

# 2. 健康检查轮询
echo "[STEP 2] 轮询健康检查接口..."
while true; do
    HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:${PORT}/v1/models)
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[INFO] vLLM 服务已就绪！"
        break
    else
        echo "[WAIT] 等待服务响应 (Status: $HTTP_STATUS)... 30s 后重试"
        sleep 30
    fi
done

# 3. 循环处理各个数据集
for DS in "${DATASETS[@]}"; do
    echo "--------------------------------------------------------"
    echo "[PROCESS] 正在蒸馏数据集: ${DS}"
    echo "--------------------------------------------------------"
    # 调用位于 distill 下的脚本，传入数据集相对路径
    python distill/distill_data.py --dataset ${DS} --num_tasks 500 --dataset_dir ${DATASET_DIR}
done

echo "========================================================"
echo "[FINISH] 所有数据集蒸馏任务已完成！"
echo "========================================================"

# 4. 正常关闭
kill $VLLM_PID
trap - SIGINT SIGTERM EXIT