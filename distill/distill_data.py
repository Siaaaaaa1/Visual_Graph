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

# ================= 路径动态挂载 =================
sys.path.append("agent_system/environments/env_package")

from graph_search.envs import GraphSearchEnv
from graph_search.graph_visualizer import GraphVisualizer

# ================= 配置与日志 =================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["cora", "pubmed", "arxiv"])
parser.add_argument("--num_tasks", type=int, default=100000)
parser.add_argument("--dataset_dir", type=str, default="datasets")
parser.add_argument("--target_successes", type=int, default=1, help="每个样本需要收集的成功轨迹数")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(f"distill/distill_{args.dataset}.log", mode='w')]
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8080/v1/chat/completions"
MAX_CONCURRENT_REQUESTS = 128  # 压榨 GPU 的核心并发数控制
MAX_RETRIES = 10               # 10次未成功则丢弃

# ================= CPU 密集型辅助函数 =================
def ndarray_to_bytes(img_array: np.ndarray) -> bytes:
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def img_to_b64(img_array):
    return base64.b64encode(ndarray_to_bytes(img_array)).decode("utf-8")

# ================= 数据预加载 =================
def prepare_shared_assets(dataset_name, dataset_dir):
    logger.info(f"预加载数据集资产: {dataset_name}")
    json_path = os.path.join(dataset_dir, f"{dataset_name}.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    all_tasks = []
    for node in raw.get("nodes", []):
        nid = int(node["id"])
        ans = node.get("label") or node.get("proxy_info", {}).get("top1") or "Unknown"
        all_tasks.append({"center_id": nid, "answer": ans})
    
    text_path = os.path.join(dataset_dir, f"make_{dataset_name}_text.json")
    if not os.path.exists(text_path):
        text_path = os.path.join(dataset_dir, "node_text_db.json")
    with open(text_path, 'r', encoding='utf-8') as f:
        text_db = json.load(f)

    g_data, r_adj, c_map = GraphVisualizer.load_graph_data(dataset_name, dataset_dir)
    temp_viz = GraphVisualizer(dataset_name=dataset_name, dataset_dir=dataset_dir, shared_data=(g_data, r_adj, c_map, None))
    shared_payload = (g_data, r_adj, c_map, temp_viz.feat_matrix)
    
    return all_tasks, text_db, shared_payload

# ================= 异步 API 请求 =================
async def fetch_completion(session: aiohttp.ClientSession, msgs: list, temp: float, sem: asyncio.Semaphore):
    payload = {
        "model": "qwen3-vl-teacher",
        "messages": msgs,
        "temperature": temp,
        "stop": ["</action>"]
    }
    async with sem:
        async with session.post(API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["message"]["content"] + "</action>"

# ================= 单个任务并发执行逻辑 =================
async def run_task_async(task, text_db, shared_payload, session, sem, executor):
    tid = task["center_id"]
    successful_trajectories = []
    loop = asyncio.get_running_loop()
    
    # 针对需要多条轨迹的要求进行循环
    for needed_idx in range(args.target_successes):
        success = False
        
        for attempt in range(MAX_RETRIES):
            # 10 次内动态调温：0.6 -> 0.96
            current_temp = min(0.6 + (attempt * 0.04), 1.0)
            tau_val = random.choice([0.2, 0.4, 0.6])
            
            # 使用后台线程池执行 CPU 密集型的环境初始化
            env = await loop.run_in_executor(
                executor, 
                lambda: GraphSearchEnv(
                    max_steps=10, node_text_db=text_db, dataset_name=args.dataset,
                    dataset_dir=args.dataset_dir, shared_graph_data=shared_payload, tau=tau_val
                )
            )
            obs_text, obs_img, info = await loop.run_in_executor(executor, env.reset, task)
            
            raw_traj = []
            msgs = [{"role": "system", "content": "You are a graph reasoning expert. Think step-by-step in <think> tags and act in <action> tags."}]
            
            step_count, process_rewards, final_reward = 0, 0.0, 0.0
            done = False
            api_error = False
            
            while not done:
                user_content = [{"type": "text", "text": obs_text}]
                if obs_img is not None:
                    b64_img = await loop.run_in_executor(executor, img_to_b64, obs_img)
                    user_content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
                    img_bytes = await loop.run_in_executor(executor, ndarray_to_bytes, obs_img)
                else:
                    img_bytes = None

                msgs.append({"role": "user", "content": user_content})
                raw_traj.append({"role": "user", "content": [{"type": "text", "text": obs_text}, {"type": "image", "bytes": img_bytes}]})
                
                try:
                    ans_text = await fetch_completion(session, msgs, current_temp, sem)
                except Exception as e:
                    logger.error(f"Task {tid} API Error on attempt {attempt+1}: {e}")
                    api_error = True
                    break

                msgs.append({"role": "assistant", "content": ans_text})
                raw_traj.append({"role": "assistant", "content": ans_text})
                
                action_match = re.search(r"<action>(.*?)</action>", ans_text, re.S | re.I)
                action_str = action_match.group(1).strip() if action_match else "ERROR"
                
                obs_text, obs_img, r, done, s_info = await loop.run_in_executor(executor, env.step, action_str)
                step_count += 1
                if 0 < r < 1.0: process_rewards += r
                if done: final_reward = r

            if api_error:
                continue

            # 质量门控校验
            is_valid = True
            if final_reward < 1.0 or step_count > 8: is_valid = False
            if info["mode"] == "System2" and process_rewards <= 0: is_valid = False
            
            if is_valid:
                successful_trajectories.append({
                    "task_id": tid, "mode": info["mode"], "steps": step_count, "traj": raw_traj
                })
                success = True
                break  # 成功拿到当前要求的轨迹，跳出重试循环，进入下一条收集
                
        # 如果 10 次尝试都未能拿到这一条轨迹，放弃该任务的后续收集
        if not success:
            logger.warning(f"Task {tid} 连续 {MAX_RETRIES} 次尝试失败，放弃该样本。")
            break

    # 只有当收集到的成功轨迹数量严格满足要求时，才认为该任务完全成功
    if len(successful_trajectories) == args.target_successes:
        return successful_trajectories
    return None

# ================= 异步主函数 =================
async def main_async():
    all_tasks, text_db, shared_payload = prepare_shared_assets(args.dataset, args.dataset_dir)
    random.shuffle(all_tasks)
    tasks = all_tasks[:args.num_tasks]
    
    results = []
    training_data = []
    
    # 控制并发量：限制 aiohttp 连接池大小与并发信号量
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 建立单独的后台线程池供环境图片渲染使用
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # 创建所有任务的协程
        tasks_coroutines = [
            run_task_async(t, text_db, shared_payload, session, sem, executor) 
            for t in tasks
        ]
        
        # 使用 tqdm.asyncio 展现并发进度条
        for coro in tqdm.as_completed(tasks_coroutines, total=len(tasks), desc=f"蒸馏 {args.dataset} (目标:{args.target_successes}条/样本)"):
            trajectories = await coro
            if trajectories:
                for res in trajectories:
                    results.append({"tid": res["task_id"], "steps": res["steps"], "mode": res["mode"]})
                    t_seq = res["traj"]
                    for i in range(0, len(t_seq), 2):
                        training_data.append({
                            "dataset": args.dataset,
                            "messages": t_seq[:i+1],
                            "target": t_seq[i+1]
                        })
                        
    executor.shutdown(wait=True)

    if training_data:
        parquet_path = f"distill/{args.dataset}_training.parquet"
        stats_path = f"distill/{args.dataset}_stats.json"
        pd.DataFrame(training_data).to_parquet(parquet_path)
        with open(stats_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[SUCCESS] {args.dataset} 完工! 生成高质量微调样本数: {len(training_data)} (保存于 distill/)")

if __name__ == "__main__":
    asyncio.run(main_async())