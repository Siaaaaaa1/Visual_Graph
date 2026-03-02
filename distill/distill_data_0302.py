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

# ================= 路径动态挂载与组件导入 =================
# 将项目根目录加入环境变量，确保能顺利导入 agent_system 下的所有模块
sys.path.insert(0, os.path.abspath("."))

# 根据真实目录结构导入组件
from agent_system.memory import FullSequenceSearchMemory
from agent_system.environments.env_package.graph_search.envs import GraphSearchEnv
from agent_system.environments.env_package.graph_search.graph_visualizer import GraphVisualizer

# 导入 RL 环境管理器中一模一样的 Prompt 模板
from agent_system.environments.env_manager_graph_search import (
    SYSTEM1_TASK_INSTRUCTION, SYSTEM1_FEW_SHOT,
    SYSTEM2_TASK_INSTRUCTION, SYSTEM2_FEW_SHOT,
    TEMPLATE_NO_HIS, TEMPLATE_WITH_HIS
)

# ================= 配置与日志 =================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["cora", "pubmed", "arxiv"])
parser.add_argument("--num_tasks", type=int, default=100000)
parser.add_argument("--dataset_dir", type=str, default="datasets")
parser.add_argument("--target_successes", type=int, default=1, help="每个样本需要收集的成功轨迹数")
args = parser.parse_args()

os.makedirs("distill", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"distill/distill_{args.dataset}.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8080/v1/chat/completions"
MAX_CONCURRENT_REQUESTS = 32   # GPU 并发数
MAX_RETRIES = 10               # 最大重试次数
HISTORY_LENGTH = 10            # 记忆追溯长度

# ================= 预编译正则 =================
SUMMARY_PATTERN = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

# ================= CPU 密集型辅助函数 =================
def ndarray_to_bytes(img_array: np.ndarray) -> bytes:
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def img_to_b64(img_array):
    return base64.b64encode(ndarray_to_bytes(img_array)).decode("utf-8")

# ================= 数据预加载与测试集过滤 =================
def prepare_shared_assets(dataset_name, dataset_dir):
    logger.info(f"预加载数据集资产: {dataset_name}")
    
    test_parquet_path = os.path.join(dataset_dir, f"{dataset_name}_test_slim.parquet")
    test_ids = set()
    if os.path.exists(test_parquet_path):
        test_df = pd.read_parquet(test_parquet_path)
        if 'center_id' in test_df.columns:
            test_ids = set(test_df['center_id'].astype(int).tolist())
        logger.info(f"🛡️ 成功加载测试集名单，共计 {len(test_ids)} 个测试节点将在蒸馏中被强行过滤！")
    else:
        logger.warning(f"⚠️ 未找到测试集文件 {test_parquet_path}，无法执行测试集屏蔽！")

    json_path = os.path.join(dataset_dir, f"{dataset_name}.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    all_tasks = []
    for node in raw.get("nodes", []):
        nid = int(node["id"])
        
        if nid in test_ids:
            continue
            
        ans = node.get("label") or node.get("proxy_info", {}).get("top1") or "Unknown"
        all_tasks.append({"center_id": nid, "answer": ans})
        
    logger.info(f"✅ 过滤后剩余可用节点数 (非Test集): {len(all_tasks)}")
    
    text_path = os.path.join(dataset_dir, f"{dataset_name}_text.json")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"找不到节点文本文件：{text_path}")
        
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
        timeout = aiohttp.ClientTimeout(total=3600, connect=300, sock_read=3600)
        async with session.post(API_URL, json=payload, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")
                
            data = await response.json()
            return data["choices"][0]["message"]["content"] + "</action>"

# ================= 单个任务并发执行逻辑 (完全复刻 RL Env Manager) =================
async def run_task_async(task, text_db, shared_payload, session, sem, executor):
    tid = task["center_id"]
    successful_trajectories = []
    loop = asyncio.get_running_loop()
    
    for needed_idx in range(args.target_successes):
        success = False
        
        for attempt in range(MAX_RETRIES):
            current_temp = min(0.6 + (attempt * 0.04), 1.0)
            tau_val = random.choice([0.2, 0.4, 0.6])
            
            # 初始化 Memory (每次尝试均重置)
            memory = FullSequenceSearchMemory()
            memory.reset(batch_size=1)
            
            env = await loop.run_in_executor(
                executor, 
                lambda: GraphSearchEnv(
                    max_steps=10, node_text_db=text_db, dataset_name=args.dataset,
                    dataset_dir=args.dataset_dir, shared_graph_data=shared_payload, tau=tau_val
                )
            )
            obs_text, obs_img, info = await loop.run_in_executor(executor, env.reset, task)
            
            # 保存初始状态给 Prompt 模板用
            initial_state = obs_text 
            raw_traj = []
            
            step_count, process_rewards, final_reward = 0, 0.0, 0.0
            done = False
            api_error = False
            
            while not done:
                # 1. 判断并获取对应的任务指令和 few shot
                mode = info["mode"]
                task_inst = SYSTEM1_TASK_INSTRUCTION if mode == "System1" else SYSTEM2_TASK_INSTRUCTION
                few_shot = SYSTEM1_FEW_SHOT if mode == "System1" else SYSTEM2_FEW_SHOT
                
                # 2. 完全模拟 RL 的 Prompt 构建逻辑
                if step_count == 0:
                    prompt_text = TEMPLATE_NO_HIS.format(
                        task_instruction=task_inst,
                        few_shot=few_shot,
                        initial_state=initial_state
                    )
                else:
                    memory_ctx, _ = memory.fetch(
                        HISTORY_LENGTH,
                        obs_key="information",
                        action_key="search",
                        summary_key="summary"
                    )
                    prompt_text = TEMPLATE_WITH_HIS.format(
                        task_instruction=task_inst,
                        few_shot=few_shot,
                        initial_state=initial_state,
                        memory_context=memory_ctx[0],
                        step_count=step_count + 1
                    )
                
                # 3. 构造真正的 API Message
                # 此处 user_content 仅包含本轮拼接好的大段文本（内含纯文本历史记录）以及唯一的一张最新图
                user_content = [{"type": "text", "text": prompt_text}]
                
                if obs_img is not None:
                    b64_img = await loop.run_in_executor(executor, img_to_b64, obs_img)
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

                # API 调用永远是单轮请求，彻底斩断历史图片堆积导致的 HTTP 400 OOM 风险
                api_msgs = [{"role": "user", "content": user_content}]
                
                try:
                    ans_text = await fetch_completion(session, api_msgs, current_temp, sem)
                except Exception as e:
                    err_msg = repr(e)
                    logger.error(f"Task {tid} API Error on attempt {attempt+1}: {err_msg}")
                    api_error = True
                    break

                # 4. 解析模型的返回内容
                s_match = SUMMARY_PATTERN.search(ans_text)
                summary = s_match.group(1).strip() if s_match else "No summary provided."
                
                action_match = ACTION_PATTERN.search(ans_text)
                action_str = action_match.group(1).strip() if action_match else "ERROR"
                
                # 5. 环境前进一步，获取反馈
                next_obs_text, next_obs_img, r, done, s_info = await loop.run_in_executor(executor, env.step, action_str)
                
                # 6. 将操作存入 Memory，供下一轮使用
                memory.store({
                    "search": [action_str],
                    "information": [next_obs_text],
                    "summary": [summary]
                })
                
                # 7. 存入专门供 SFT 使用的 Trajectory (单轮格式，文本已包含完整上下文)
                raw_traj.append({
                    "messages": [{"role": "user", "content": user_content}],
                    "target": {"role": "assistant", "content": ans_text}
                })
                
                # 更新循环变量
                obs_text = next_obs_text
                obs_img = next_obs_img
                step_count += 1
                if 0 < r < 1.0: process_rewards += r
                if done: final_reward = r

            if api_error:
                await asyncio.sleep(2)
                continue

            # ================= 质量门控 =================
            is_valid = True
            if final_reward < 1.0 or step_count > 8: is_valid = False
            if info["mode"] == "System2" and process_rewards <= 0: is_valid = False
            
            if is_valid:
                successful_trajectories.append({
                    "task_id": tid, "mode": info["mode"], "steps": step_count, "traj": raw_traj
                })
                success = True
                break
                
        if not success:
            logger.warning(f"Task {tid} 连续 {MAX_RETRIES} 次尝试失败，放弃该样本。")
            break

    if len(successful_trajectories) == args.target_successes:
        return successful_trajectories
    return None

# ================= 异步主函数 =================
async def main_async():
    all_tasks, text_db, shared_payload = prepare_shared_assets(args.dataset, args.dataset_dir)
    
    stats_path = f"distill/{args.dataset}_stats.jsonl"
    training_data_path = f"distill/{args.dataset}_training.jsonl"
    
    completed_tids = set()
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed_tids.add(data["tid"])
                    except Exception:
                        pass
                        
    logger.info(f"🔄 检测到本地历史进度，跳过已完成的任务数: {len(completed_tids)}")
    
    pending_tasks = [t for t in all_tasks if t["center_id"] not in completed_tids]
    random.shuffle(pending_tasks)
    
    tasks_to_run = pending_tasks[:max(0, args.num_tasks - len(completed_tids))]
    
    if not tasks_to_run:
        logger.info("🎉 所有指定数量的任务已全部完成，无需继续执行！")
    else:
        logger.info(f"🚀 本次实际启动蒸馏任务数: {len(tasks_to_run)}")
        
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
        
        saved_count = 0
        
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks_coroutines = [
                    run_task_async(t, text_db, shared_payload, session, sem, executor) 
                    for t in tasks_to_run
                ]
                
                for coro in tqdm.as_completed(tasks_coroutines, total=len(tasks_to_run), desc=f"蒸馏 {args.dataset}"):
                    trajectories = await coro
                    if trajectories:
                        with open(stats_path, 'a', encoding='utf-8') as f_stats, \
                             open(training_data_path, 'a', encoding='utf-8') as f_data:
                             
                            for res in trajectories:
                                stats_record = {"tid": res["task_id"], "steps": res["steps"], "mode": res["mode"]}
                                f_stats.write(json.dumps(stats_record) + "\n")
                                
                                t_seq = res["traj"]
                                # 写入 SFT 格式
                                for step_data in t_seq:
                                    train_record = {
                                        "dataset": args.dataset,
                                        "messages": step_data["messages"] + [step_data["target"]]
                                    }
                                    f_data.write(json.dumps(train_record) + "\n")
                                
                                saved_count += 1
        except KeyboardInterrupt:
            logger.info("\n🛑 检测到键盘中断 (Ctrl+C)，正在安全停止并进入 Parquet 转换阶段...")
        finally:
            executor.shutdown(wait=True)
            print(f"\n[SUCCESS] 蒸馏流程结束! 本次新增成功轨迹数: {saved_count} (数据已写入 jsonl)")

    # ================= 最终步骤：自动生成 Parquet 供训练使用 =================
    logger.info("正在将完整的 JSONL 数据转换为 Parquet 格式以供后续 SFT 训练...")
    if os.path.exists(training_data_path):
        try:
            df = pd.read_json(training_data_path, lines=True)
            parquet_path = f"distill/{args.dataset}_training.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"✅ 成功生成 Parquet 训练集: {parquet_path}，共包含 {len(df)} 条对话样本！")
            logger.info(f"👉 现在可以直接运行: bash distill/run_sft_training.sh 8 distill/output_dir")
        except Exception as e:
            logger.error(f"❌ JSONL 转换 Parquet 失败: {e}")
    else:
        logger.warning(f"⚠️ 未找到数据文件 {training_data_path}，跳过 Parquet 转换。")

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass