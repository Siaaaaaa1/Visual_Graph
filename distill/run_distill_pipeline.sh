#!/bin/bash

# ================= 配置区 =================
MODEL_PATH="Qwen/Qwen3-VL-235B-A22B-Thinking"
TP_SIZE=8           # GPU 数量
PORT=8000
DATASETS=("cora" "pubmed" "arxiv") # 需要处理的数据集列表
SERVER_LOG="vllm_server.log"

echo "========================================================"
echo "[START] 启动数据蒸馏全自动 Pipeline"
echo "========================================================"

# 1. 后台启动服务
echo "[STEP 1] 启动 vLLM 后端服务 (235B 模型加载较慢，请稍后)..."
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --tensor-parallel-size ${TP_SIZE} \
    --port ${PORT} \
    --max-model-len 16384 \
    --served-model-name qwen3-vl-teacher > ${SERVER_LOG} 2>&1 &

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
    # 调用 Python 脚本，传入数据集名称
    python distill_data.py --dataset ${DS} --num_tasks 500
done

echo "========================================================"
echo "[FINISH] 所有数据集蒸馏任务已完成！"
echo "========================================================"

# 4. 正常关闭
kill $VLLM_PID
trap - SIGINT SIGTERM EXIT