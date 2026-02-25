#!/bin/bash

# ================= 配置区 =================
MODEL_PATH="Qwen3-VL-235B-A22B-Thinking"
DATASET_DIR="datasets"
SERVER_LOG="distill/vllm_server.log"

TP_SIZE=8           
PORT=8080

echo "========================================================"
echo "[START] 启动高并发异步全自动蒸馏 (执行路径: $(pwd))"
echo "========================================================"

# 1. 启动服务 (注入高并发吞吐量参数)
echo "[STEP 1] 启动 vLLM 后端服务 (Qwen3-VL-235B)..."
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --tensor-parallel-size ${TP_SIZE} \
    --port ${PORT} \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --served-model-name qwen3-vl-teacher > ${SERVER_LOG} 2>&1 &

VLLM_PID=$!
trap "kill -9 $VLLM_PID; echo '[EXIT] 服务已强制关闭'; exit" SIGINT SIGTERM EXIT

# 2. 健康检查
while true; do
    HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:${PORT}/v1/models)
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[INFO] vLLM 服务已就绪！可以开始打满并发！"
        break
    else
        echo "[WAIT] 等待服务响应... 30s 后重试"
        sleep 30
    fi
done

# 3. 严格按顺序处理，执行特定采样策略
echo "--------------------------------------------------------"
echo "[PROCESS] 1/3: 蒸馏数据集 cora (目标: 3条成功轨迹/样本)"
python distill/distill_data.py --dataset cora --num_tasks 100000 --dataset_dir ${DATASET_DIR} --target_successes 3

echo "--------------------------------------------------------"
echo "[PROCESS] 2/3: 蒸馏数据集 pubmed (目标: 3条成功轨迹/样本)"
python distill/distill_data.py --dataset pubmed --num_tasks 100000 --dataset_dir ${DATASET_DIR} --target_successes 3

echo "--------------------------------------------------------"
echo "[PROCESS] 3/3: 蒸馏数据集 arxiv (目标: 1条成功轨迹/样本)"
python distill/distill_data.py --dataset arxiv --num_tasks 100000 --dataset_dir ${DATASET_DIR} --target_successes 1

echo "========================================================"
echo "[FINISH] 所有数据集蒸馏任务已完成！"
kill $VLLM_PID
trap - SIGINT SIGTERM EXIT