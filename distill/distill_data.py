import os
import sys
import json
import base64
import logging
import argparse
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
import random
from dotenv import load_dotenv

# ================= 1. 环境与配置加载 =================
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_URL = f"{DASHSCOPE_BASE_URL}/chat/completions"

# ================= 2. 路径挂载与组件导入 =================
sys.path.insert(0, os.path.abspath("."))

# 导入你提供的组件
from agent_system.environments.env_package.graph_search.envs import GraphSearchEnv
from agent_system.environments.env_package.graph_search.graph_visualizer import GraphVisualizer
# 从 env_manager 中获取最新的系统提示词
from agent_system.environments.env_manager_graph_search import V_GRAPH_AGENT_INSTRUCTION

# ================= 3. 参数解析与日志 =================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["cora", "pubmed", "arxiv"])
parser.add_argument("--num_tasks", type=int, default=100000)
parser.add_argument("--dataset_dir", type=str, default="datasets")
args = parser.parse_args()

# 蒸馏门控常量
INITIAL_MAX_ATTEMPTS = 5
EXTENDED_MAX_ATTEMPTS = 15
TARGET_SUCCESS_COUNT = 3

os.makedirs("distill", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"distill/distill_vgraph_{args.dataset}.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 并发控制：针对 qwen-vl-max 建议保持在 8-10 左右
MAX_CONCURRENT_REQUESTS = 8   

# ================= 4. 辅助函数 =================
def ndarray_to_bytes(img_array: np.ndarray) -> bytes:
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def img_to_b64(img_array):
    return base64.b64encode(ndarray_to_bytes(img_array)).decode("utf-8")

# 正则匹配
ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

# ================= 5. 数据预加载 =================
def prepare_shared_assets(dataset_name, dataset_dir):
    logger.info(f"预加载数据集资产: {dataset_name}")
    
    test_parquet_path = os.path.join(dataset_dir, f"{dataset_name}_test_slim.parquet")
    test_ids = set()
    if os.path.exists(test_parquet_path):
        test_df = pd.read_parquet(test_parquet_path)
        if 'center_id' in test_df.columns:
            test_ids = set(test_df['center_id'].astype(int).tolist())
        logger.info(f"🛡️ 过滤测试集节点数: {len(test_ids)}")

    json_path = os.path.join(dataset_dir, f"{dataset_name}.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    all_tasks = []
    for node in raw.get("nodes", []):
        nid = int(node["id"])
        if nid in test_ids: continue
        ans = node.get("label") or node.get("proxy_info", {}).get("top1") or "Unknown"
        all_tasks.append({"center_id": nid, "answer": ans})
        
    text_path = os.path.join(dataset_dir, f"{dataset_name}_text.json")
    with open(text_path, 'r', encoding='utf-8') as f:
        text_db = json.load(f)

    # 初始化 Visualizer 获取共享数据
    g_data, r_adj, c_map = GraphVisualizer.load_graph_data(dataset_name, dataset_dir)
    temp_viz = GraphVisualizer(dataset_name=dataset_name, dataset_dir=dataset_dir, shared_data=(g_data, r_adj, c_map, None))
    shared_payload = (g_data, r_adj, c_map, temp_viz.feat_matrix)
    
    return all_tasks, text_db, shared_payload

# ================= 6. 异步 API 请求 (指定使用 qwen-vl-max) =================
async def fetch_completion(session: aiohttp.ClientSession, msgs: list, temp: float, sem: asyncio.Semaphore):
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-vl-max",
        "messages": msgs,
        "temperature": temp,
        "stop": ["</action>"]
    }
    async with sem:
        timeout = aiohttp.ClientTimeout(total=600, connect=60, sock_read=300)
        async with session.post(API_URL, json=payload, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")
            data = await response.json()
            return data["choices"][0]["message"]["content"] + "</action>"

# ================= 7. 核心任务蒸馏逻辑 =================
async def run_task_async(task, text_db, shared_payload, session, sem, executor):
    tid = task["center_id"]
    collected_trajectories = []
    loop = asyncio.get_running_loop()
    
    # 逻辑：前5次只要有1次成功，就目标凑齐3条，上限15次
    for attempt_idx in range(1, EXTENDED_MAX_ATTEMPTS + 1):
        # 门控检查：如果前5次结束了且一条成功的都没有，直接退出
        if attempt_idx > INITIAL_MAX_ATTEMPTS and len(collected_trajectories) == 0:
            break
        # 目标达成检查
        if len(collected_trajectories) >= TARGET_SUCCESS_COUNT:
            break

        # 动态调节温度增加探索性
        current_temp = 0.6 if attempt_idx <= 5 else 0.8
        
        # 初始化环境
        env = await loop.run_in_executor(
            executor, 
            lambda: GraphSearchEnv(
                max_steps=10, node_text_db=text_db, dataset_name=args.dataset,
                dataset_dir=args.dataset_dir, shared_graph_data=shared_payload
            )
        )
        obs_text, obs_img, info = await loop.run_in_executor(executor, env.reset, task)
        
        messages = [{"role": "system", "content": V_GRAPH_AGENT_INSTRUCTION}] # 使用最新的 Instruction
        raw_traj_steps = []
        done = False
        api_error = False
        final_reward = 0.0
        
        while not done:
            user_content = [{"type": "text", "text": obs_text}]
            if obs_img is not None:
                b64_img = await loop.run_in_executor(executor, img_to_b64, obs_img)
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
            
            messages.append({"role": "user", "content": user_content})
            
            try:
                ans_text = await fetch_completion(session, messages, current_temp, sem)
            except Exception as e:
                logger.error(f"Task {tid} API Error: {repr(e)}")
                api_error = True
                break

            messages.append({"role": "assistant", "content": ans_text})
            
            # 解析 Action 并推进行进
            action_match = ACTION_PATTERN.search(ans_text)
            action_str = action_match.group(1).strip() if action_match else "ERROR"
            
            next_obs_text, next_obs_img, r, done, s_info = await loop.run_in_executor(executor, env.step, action_str)
            
            raw_traj_steps.append({
                "user": user_content,
                "assistant": ans_text
            })
            
            obs_text, obs_img = next_obs_text, next_obs_img
            if done: final_reward = r

        if not api_error and final_reward >= 1.0: # 环境中 reward=1.0 代表预测正确
            collected_trajectories.append({
                "task_id": tid,
                "traj": raw_traj_steps
            })

    return collected_trajectories if collected_trajectories else None

# ================= 8. 异步主函数 =================
async def main_async():
    if not DASHSCOPE_API_KEY:
        logger.error("DASHSCOPE_API_KEY 缺失！")
        return

    all_tasks, text_db, shared_payload = prepare_shared_assets(args.dataset, args.dataset_dir)
    
    stats_path = f"distill/{args.dataset}_vgraph_stats.jsonl"
    training_data_path = f"distill/{args.dataset}_vgraph_training.jsonl"
    
    completed_tids = set()
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed_tids.add(data["tid"])
                    except: pass
                        
    pending_tasks = [t for t in all_tasks if t["center_id"] not in completed_tids]
    random.shuffle(pending_tasks)
    tasks_to_run = pending_tasks[:args.num_tasks]
    
    logger.info(f"🚀 启动蒸馏。待执行任务数: {len(tasks_to_run)}")
    
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    executor = ThreadPoolExecutor(max_workers=32)
    
    saved_traj_count = 0
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks_coroutines = [
                run_task_async(t, text_db, shared_payload, session, sem, executor) 
                for t in tasks_to_run
            ]
            
            for coro in tqdm.as_completed(tasks_coroutines, total=len(tasks_to_run), desc="蒸馏进度"):
                trajs = await coro
                if trajs:
                    with open(stats_path, 'a', encoding='utf-8') as f_stats, \
                         open(training_data_path, 'a', encoding='utf-8') as f_data:
                        
                        f_stats.write(json.dumps({"tid": trajs[0]["task_id"], "count": len(trajs)}) + "\n")
                        
                        for entry in trajs:
                            # 构造符合 SFT 格式的消息流
                            sft_messages = [{"role": "system", "content": V_GRAPH_AGENT_INSTRUCTION}]
                            for step in entry["traj"]:
                                sft_messages.append({"role": "user", "content": step["user"]})
                                sft_messages.append({"role": "assistant", "content": step["assistant"]})
                            
                            f_data.write(json.dumps({
                                "dataset": args.dataset,
                                "messages": sft_messages
                            }, ensure_ascii=False) + "\n")
                            saved_traj_count += 1
                            
    except KeyboardInterrupt:
        logger.info("用户中断执行。")
    finally:
        executor.shutdown(wait=True)
        logger.info(f"蒸馏结束。共保存成功轨迹数: {saved_traj_count}")

    # 自动转换 Parquet
    if os.path.exists(training_data_path):
        try:
            df = pd.read_json(training_data_path, lines=True)
            df.to_parquet(f"distill/{args.dataset}_vgraph_training.parquet", index=False)
            logger.info("✅ 训练集 Parquet 已生成。")
        except Exception as e:
            logger.error(f"❌ Parquet 转换失败: {e}")

if __name__ == "__main__":
    asyncio.run(main_async())