import os
import sys
import json
import base64
import copy
import logging
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import re
import random

# ================= 路径动态挂载 =================
# 因为在 verl-agent 根目录下运行，直接使用相对路径添加包
sys.path.append("agent_system/environments/env_package")

# 导入环境组件
from graph_search.envs import GraphSearchEnv
from graph_search.graph_visualizer import GraphVisualizer

# ================= 配置与日志 =================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["cora", "pubmed", "arxiv"])
parser.add_argument("--num_tasks", type=int, default=200)
parser.add_argument("--dataset_dir", type=str, default="datasets")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    # 将日志强制写入 distill 目录
    handlers=[logging.FileHandler(f"distill/distill_{args.dataset}.log", mode='w')]
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8080/v1")

# ================= 辅助函数 =================
def ndarray_to_bytes(img_array: np.ndarray) -> bytes:
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def img_to_b64(img_array):
    return base64.b64encode(ndarray_to_bytes(img_array)).decode("utf-8")

# ================= 数据预加载与任务提取 =================
def prepare_shared_assets(dataset_name, dataset_dir):
    logger.info(f"预加载数据集资产: {dataset_name}")
    
    # 1. 提取真实任务 (center_id 和真实 answer)
    json_path = os.path.join(dataset_dir, f"{dataset_name}.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    all_tasks = []
    for node in raw.get("nodes", []):
        nid = int(node["id"])
        # 优先取 label，没有则取 proxy_info 中的 top1
        ans = node.get("label") or node.get("proxy_info", {}).get("top1") or "Unknown"
        all_tasks.append({"center_id": nid, "answer": ans})
    
    # 2. 预加载文本库
    text_path = os.path.join(dataset_dir, f"make_{dataset_name}_text.json")
    if not os.path.exists(text_path):
        text_path = os.path.join(dataset_dir, "node_text_db.json") # Fallback
    with open(text_path, 'r', encoding='utf-8') as f:
        text_db = json.load(f)

    # 3. 构建共享 Payload 避免内存崩溃
    g_data, r_adj, c_map = GraphVisualizer.load_graph_data(dataset_name, dataset_dir)
    temp_viz = GraphVisualizer(dataset_name=dataset_name, dataset_dir=dataset_dir, shared_data=(g_data, r_adj, c_map, None))
    shared_payload = (g_data, r_adj, c_map, temp_viz.feat_matrix)
    
    return all_tasks, text_db, shared_payload

# ================= 蒸馏 Episode 逻辑 =================
def run_episode(task, text_db, shared_payload, max_retries=30):
    tid = task["center_id"]
    
    for attempt in range(max_retries):
        # 动态调整温度：每次失败稍微提高温度 (从0.6开始，每次+0.02，最高不超过1.0)
        # 这样在模型卡住时，后续尝试会有更高的探索性和多样性
        current_temp = min(0.6 + (attempt * 0.02), 1.0)
        
        tau_val = random.choice([0.2, 0.4, 0.6]) # 每次尝试也可以换不同的多样性参数
        
        env = GraphSearchEnv(
            max_steps=10,
            node_text_db=text_db,
            dataset_name=args.dataset,
            dataset_dir=args.dataset_dir,
            shared_graph_data=shared_payload,
            tau=tau_val
        )
        
        obs_text, obs_img, info = env.reset(task)
        raw_traj = []
        msgs = [{"role": "system", "content": "You are a graph reasoning expert. Think step-by-step in <think> tags and act in <action> tags."}]
        
        step_count, process_rewards, final_reward = 0, 0.0, 0.0
        done = False
        api_error = False
        
        while not done:
            # 组装消息
            user_content = [{"type": "text", "text": obs_text}]
            if obs_img is not None:
                b64_img = img_to_b64(obs_img)
                user_content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
                img_bytes = ndarray_to_bytes(obs_img)
            else:
                img_bytes = None

            msgs.append({"role": "user", "content": user_content})
            raw_traj.append({"role": "user", "content": [{"type": "text", "text": obs_text}, {"type": "image", "bytes": img_bytes}]})
            
            try:
                # 使用动态调整的温度
                res = client.chat.completions.create(model="qwen3-vl-teacher", messages=msgs, temperature=current_temp, stop=["</action>"])
                ans_text = res.choices[0].message.content + "</action>"
            except Exception as e:
                logger.error(f"Task {tid} API Error on attempt {attempt+1}: {e}")
                api_error = True
                break

            msgs.append({"role": "assistant", "content": ans_text})
            raw_traj.append({"role": "assistant", "content": ans_text})
            
            action = re.search(r"<action>(.*?)</action>", ans_text, re.S | re.I)
            action_str = action.group(1).strip() if action else "ERROR"
            
            obs_text, obs_img, r, done, s_info = env.step(action_str)
            step_count += 1
            if 0 < r < 1.0: process_rewards += r
            if done: final_reward = r

        # 如果是因为 API 错误中断，直接进入下一次尝试
        if api_error:
            continue

        # ====== 质量门控校验 ======
        is_valid = True
        if final_reward < 1.0 or step_count > 8: 
            is_valid = False
        if info["mode"] == "System2" and process_rewards <= 0: 
            is_valid = False
        
        # 如果成功通过校验，直接返回结果退出循环
        if is_valid:
            if attempt > 0:
                logger.info(f"Task {tid} 成功通过 (耗费尝试次数: {attempt + 1}, 最终温度: {current_temp:.2f})")
            return {"task_id": tid, "mode": info["mode"], "steps": step_count, "traj": raw_traj}
            
    # ====== 如果 30 次尝试全部失败 ======
    logger.warning(f"Task {tid} 尝试了 {max_retries} 次均未能成功通过质量门控，最终放弃。")
    return None

# ================= 主循环 =================
def main():
    all_tasks, text_db, shared_payload = prepare_shared_assets(args.dataset, args.dataset_dir)
    
    random.shuffle(all_tasks)
    tasks = all_tasks[:args.num_tasks]
    
    results = []
    training_data = []
    
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_episode, t, text_db, shared_payload) for t in tasks]
        for f in tqdm(futures, desc=f"蒸馏 {args.dataset}"):
            res = f.result()
            if res:
                results.append({"tid": res["task_id"], "steps": res["steps"], "mode": res["mode"]})
                # 轨迹展开
                t = res["traj"]
                for i in range(0, len(t), 2):
                    training_data.append({
                        "dataset": args.dataset,
                        "messages": t[:i+1],
                        "target": t[i+1]
                    })
    
    # 强制落盘到 distill 目录
    if training_data:
        parquet_path = f"distill/{args.dataset}_training.parquet"
        stats_path = f"distill/{args.dataset}_stats.json"
        pd.DataFrame(training_data).to_parquet(parquet_path)
        with open(stats_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[SUCCESS] {args.dataset} 完成! 有效任务数: {len(results)}, 样本数: {len(training_data)} (保存于 distill/)")

if __name__ == "__main__":
    main()