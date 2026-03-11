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
parser.add_argument("--max_hard_per_class", type=int, default=0,
                    help="每个类别困难样本（首次成功 attempt > INITIAL_MAX_ATTEMPTS）的保留上限（0=不限）。"
                         "建议设为 0 以尽量保留所有困难样本，提升模型解决难题的能力。")
parser.add_argument("--max_easy_per_class", type=int, default=200,
                    help="每个类别简单样本（首次成功 attempt <= INITIAL_MAX_ATTEMPTS）的保留上限。"
                         "设得过高会导致简单样本淹没困难样本；建议为 max_hard_per_class 的 30-50%%。")
parser.add_argument("--trajectories_per_node", type=int, default=3,
                    help="每个节点最多收集的成功轨迹数。多条轨迹 = 同一问题的多种推理路径 = 更高多样性。")
parser.add_argument("--max_attempts", type=int, default=15,
                    help="每个节点的最大总尝试次数。超过后若仍无成功则放弃，并写入 debug 文件。")
args = parser.parse_args()

# 蒸馏门控常量
INITIAL_MAX_ATTEMPTS = 5
EXTENDED_MAX_ATTEMPTS = 15
TARGET_SUCCESS_COUNT = 3

os.makedirs("distill", exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,   # 根 logger 设 WARNING，屏蔽所有第三方库噪音
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"distill/distill_vgraph_{args.dataset}.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
# 只有自己的 logger 开 DEBUG，精确定位卡点
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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
        "model": "qwen3-vl-plus",
        "messages": msgs,
        "temperature": temp,
        "stop": ["</action>"]
    }
    async with sem:
        timeout = aiohttp.ClientTimeout(total=120, connect=15, sock_read=90)
        async with session.post(API_URL, json=payload, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text[:300]}")
            data = await response.json()
            return data["choices"][0]["message"]["content"] + "</action>"

# ================= 7. 核心任务蒸馏逻辑 =================
async def run_task_async(task, text_db, shared_payload, session, sem, executor,
                         trajectories_per_node: int, max_attempts: int):
    """
    对单个节点执行多轮蒸馏。

    - trajectories_per_node : 每个节点最多收集的成功轨迹数（多条 = 多样性）
    - max_attempts           : 总尝试上限，超过后若仍无成功则放弃该节点
    - 门控逻辑               : 前 INITIAL_MAX_ATTEMPTS 次若 0 成功 → 提前放弃
    - 返回                   : (collected_trajectories, failure_info)
                               failure_info 仅在完全失败时非 None
    """
    tid = task["center_id"]
    collected_trajectories = []
    loop = asyncio.get_running_loop()

    last_attempt_steps = []   # 最后一次尝试的完整步骤，供 debug 用
    last_failure_type = "no_correct_answer"
    actual_attempts = 0

    for attempt_idx in range(1, max_attempts + 1):
        actual_attempts = attempt_idx

        # 门控：前 INITIAL_MAX_ATTEMPTS 次结束后仍 0 成功 → 提前放弃（困难节点耗尽探索预算）
        if attempt_idx > INITIAL_MAX_ATTEMPTS and len(collected_trajectories) == 0:
            last_failure_type = "gated_out_no_early_success"
            break
        # 目标达成
        if len(collected_trajectories) >= trajectories_per_node:
            break

        # 动态调节温度：前期保守探索，后期加大随机性
        current_temp = 0.6 if attempt_idx <= 5 else 0.8

        # 初始化环境
        logger.debug(f"[Task {tid}] attempt={attempt_idx} — 创建 env...")
        env = await loop.run_in_executor(
            executor,
            lambda: GraphSearchEnv(
                max_steps=10, node_text_db=text_db, dataset_name=args.dataset,
                dataset_dir=args.dataset_dir, shared_graph_data=shared_payload
            )
        )
        logger.debug(f"[Task {tid}] attempt={attempt_idx} — env.reset (渲染图片)...")
        obs_text, obs_img, info = await loop.run_in_executor(executor, env.reset, task)
        logger.debug(f"[Task {tid}] attempt={attempt_idx} — env.reset 完成，开始对话循环")

        messages = [{"role": "system", "content": V_GRAPH_AGENT_INSTRUCTION}]
        raw_traj_steps = []
        done = False
        api_error = False
        final_reward = 0.0
        step_idx = 0

        while not done:
            user_content = [{"type": "text", "text": obs_text}]
            if obs_img is not None:
                b64_img = await loop.run_in_executor(executor, img_to_b64, obs_img)
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})

            messages.append({"role": "user", "content": user_content})

            logger.info(f"[Task {tid}] attempt={attempt_idx} step={step_idx} — → API请求中 (msgs={len(messages)}条)...")
            try:
                ans_text = await fetch_completion(session, messages, current_temp, sem)
                logger.info(f"[Task {tid}] attempt={attempt_idx} step={step_idx} — ← API返回 {len(ans_text)} 字符")
            except Exception as e:
                logger.error(f"Task {tid} API Error: {repr(e)}")
                api_error = True
                last_failure_type = "api_error"
                break
            step_idx += 1

            messages.append({"role": "assistant", "content": ans_text})

            action_match = ACTION_PATTERN.search(ans_text)
            action_str = action_match.group(1).strip() if action_match else "ERROR"

            next_obs_text, next_obs_img, r, done, s_info = await loop.run_in_executor(executor, env.step, action_str)

            raw_traj_steps.append({
                "user": user_content,
                "assistant": ans_text
            })

            obs_text, obs_img = next_obs_text, next_obs_img
            if done:
                final_reward = r

        last_attempt_steps = raw_traj_steps  # 始终保留最后一次尝试用于 debug

        if not api_error and final_reward >= 1.0:
            collected_trajectories.append({
                "task_id": tid,
                "node_class": task["answer"],
                "first_success_attempt": attempt_idx,
                "traj": raw_traj_steps
            })

    # ---- 返回结果 ----
    if collected_trajectories:
        # 将节点级统计写入每条轨迹，供调用方计算成功率
        node_successes = len(collected_trajectories)
        for traj in collected_trajectories:
            traj["node_attempts"] = actual_attempts
            traj["node_successes"] = node_successes
        return collected_trajectories, None

    # 完全失败：构造 debug 信息（截取最后两步的 assistant 输出，避免 base64 图片膨胀体积）
    debug_steps = []
    for step in last_attempt_steps[-2:]:
        debug_steps.append({
            "assistant_text": step.get("assistant", "")[:1000],
            "user_text": step["user"][0].get("text", "") if step.get("user") else ""
        })

    failure_info = {
        "task_id": tid,
        "answer": task["answer"],
        "total_attempts": actual_attempts,
        "failure_type": last_failure_type,
        "last_steps_summary": debug_steps,
    }
    return None, failure_info

# ================= 8. 样本图片保存（用于人工验证） =================
def _save_sample_images(training_data_path: str, dataset_name: str, n: int = 5):
    """
    从 JSONL 训练文件中随机抽取 n 条轨迹，将每条轨迹所有步骤的图片
    （含步骤序号和轨迹 idx）保存到 distill/{dataset}_samples/ 目录下。

    - 每条轨迹以子目录 traj_{idx}/ 存放，每步图片命名为 step_{step_idx}.png
    - 由于每步图片完全相同（雷达图不更新），可通过目视确认一致性
    """
    if not os.path.exists(training_data_path):
        logger.warning("_save_sample_images: 训练文件不存在，跳过。")
        return

    sample_dir = os.path.join("distill", f"{dataset_name}_samples")
    os.makedirs(sample_dir, exist_ok=True)

    try:
        with open(training_data_path, 'r', encoding='utf-8') as f:
            all_entries = [json.loads(line) for line in f if line.strip()]

        if not all_entries:
            logger.warning("_save_sample_images: 训练文件为空，跳过。")
            return

        samples = random.sample(all_entries, min(n, len(all_entries)))
        logger.info(f"💾 保存 {len(samples)} 条随机样本图片 → {sample_dir}/")

        for traj_idx, entry in enumerate(samples):
            traj_dir = os.path.join(sample_dir, f"traj_{traj_idx:02d}")
            os.makedirs(traj_dir, exist_ok=True)

            messages = entry.get("messages", [])
            step_idx = 0
            for msg in messages:
                if msg["role"] != "user":
                    continue
                content = msg["content"]
                if not isinstance(content, list):
                    step_idx += 1
                    continue
                for part in content:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        img = Image.open(BytesIO(img_bytes))
                        save_path = os.path.join(traj_dir, f"step_{step_idx:02d}.png")
                        img.save(save_path)
                        break
                step_idx += 1

            saved_steps = len(list(os.scandir(traj_dir)))
            logger.info(f"  traj_{traj_idx:02d}: 保存了 {saved_steps} 张图片（dataset={entry.get('dataset', '?')}）")

    except Exception as e:
        logger.error(f"_save_sample_images 失败: {e}")


# ================= 9. 异步主函数 =================
async def _check_api_connectivity():
    """启动前快速验证 DashScope API 是否可达，失败直接报错退出"""
    test_payload = {
        "model": "qwen3-vl-plus",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=15, connect=10)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, json=test_payload, headers=headers, timeout=timeout) as resp:
                if resp.status in (200, 400):  # 400 也说明服务可达（参数问题）
                    logger.info(f"✅ DashScope API 连通性验证通过 (HTTP {resp.status})")
                    return True
                text = await resp.text()
                logger.error(f"❌ DashScope API 返回异常: HTTP {resp.status} — {text[:200]}")
                return False
    except asyncio.TimeoutError:
        logger.error("❌ DashScope API 连接超时（10s），请检查网络或 DASHSCOPE_BASE_URL 是否正确")
        return False
    except Exception as e:
        logger.error(f"❌ DashScope API 连通性检查失败: {repr(e)}")
        return False


async def main_async():
    if not DASHSCOPE_API_KEY:
        logger.error("DASHSCOPE_API_KEY 缺失！")
        return

    logger.info(f"🔌 正在验证 DashScope API 连通性... ({API_URL})")
    if not await _check_api_connectivity():
        return

    # 用真实多模态请求测试（文字+图片），确认 VL 推理链路可用
    logger.info("🖼️  正在测试多模态 API（含图片）...")
    try:
        _dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        _b64 = img_to_b64(_dummy_img)
        _test_msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What color is this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64}"}}
        ]}]
        _headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
        _payload = {"model": "qwen3-vl-plus", "messages": _test_msgs, "max_tokens": 10}
        _timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        async with aiohttp.ClientSession() as _s:
            async with _s.post(API_URL, json=_payload, headers=_headers, timeout=_timeout) as _r:
                _body = await _r.json()
                if _r.status == 200:
                    _reply = _body["choices"][0]["message"]["content"]
                    logger.info(f"✅ 多模态 API 测试通过，模型回复: {_reply[:80]}")
                else:
                    logger.error(f"❌ 多模态 API 返回错误 HTTP {_r.status}: {str(_body)[:300]}")
                    return
    except asyncio.TimeoutError:
        logger.error("❌ 多模态 API 超时（30s），图片推理链路不通，请检查模型名称或网络")
        return
    except Exception as _e:
        logger.error(f"❌ 多模态 API 测试失败: {repr(_e)}")
        return

    all_tasks, text_db, shared_payload = prepare_shared_assets(args.dataset, args.dataset_dir)
    
    stats_path = f"distill/{args.dataset}_vgraph_stats.jsonl"
    training_data_path = f"distill/{args.dataset}_vgraph_training.jsonl"
    failures_path = f"distill/{args.dataset}_debug_failures.jsonl"
    
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
    
    # 预热 matplotlib 字体缓存（只在第一次渲染时扫描字体，之后所有 env 直接复用）
    logger.info("🎨 预热 matplotlib 字体缓存...")
    try:
        from agent_system.environments.env_package.graph_search.graph_visualizer import GraphVisualizer
        _warmup_viz = GraphVisualizer(
            dataset_name=args.dataset, dataset_dir=args.dataset_dir,
            shared_data=shared_payload
        )
        _warmup_center = all_tasks[0]["center_id"] if all_tasks else 0
        _warmup_viz.draw_vgraph_radar_layout(_warmup_center)
        logger.info("🎨 字体缓存预热完成")
    except Exception as e:
        logger.warning(f"字体预热失败（不影响蒸馏）: {e}")

    logger.info(f"🚀 启动蒸馏。待执行任务数: {len(tasks_to_run)}")
    
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    executor = ThreadPoolExecutor(max_workers=32)
    
    saved_traj_count = 0
    failed_node_count = 0
    hard_class_counts: dict = {}   # {class_name: hard_saved_count}
    easy_class_counts: dict = {}   # {class_name: easy_saved_count}

    logger.info(f"配置: trajectories_per_node={args.trajectories_per_node}, "
                f"max_attempts={args.max_attempts}, "
                f"max_hard_per_class={args.max_hard_per_class or '不限'}, "
                f"max_easy_per_class={args.max_easy_per_class}")

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks_coroutines = [
                run_task_async(t, text_db, shared_payload, session, sem, executor,
                               trajectories_per_node=args.trajectories_per_node,
                               max_attempts=args.max_attempts)
                for t in tasks_to_run
            ]

            for coro in tqdm.as_completed(tasks_coroutines, total=len(tasks_to_run), desc="蒸馏进度"):
                trajs, failure_info = await coro

                # 完全失败的节点 → 写入 debug 文件
                if failure_info is not None:
                    failed_node_count += 1
                    with open(failures_path, 'a', encoding='utf-8') as f_fail:
                        f_fail.write(json.dumps(failure_info, ensure_ascii=False) + "\n")

                if trajs:
                    with open(stats_path, 'a', encoding='utf-8') as f_stats, \
                         open(training_data_path, 'a', encoding='utf-8') as f_data:

                        f_stats.write(json.dumps({"tid": trajs[0]["task_id"], "count": len(trajs)}) + "\n")

                        for entry in trajs:
                            node_class = entry.get("node_class", "unknown")

                            # 难度判定：基于节点成功率（成功次数 / 总尝试次数）
                            # 成功率 < 0.5 → 困难（超过一半尝试都失败）
                            # 成功率 ≥ 0.5 → 简单
                            node_attempts  = entry.get("node_attempts", args.max_attempts)
                            node_successes = entry.get("node_successes", 1)
                            success_rate   = node_successes / node_attempts if node_attempts > 0 else 1.0
                            difficulty_score = round(1.0 - success_rate, 4)   # 越高越难
                            is_hard = success_rate < 0.5

                            # 困难样本：独立配额门控（0 = 不限制，尽量全保留）
                            if is_hard:
                                if args.max_hard_per_class > 0:
                                    if hard_class_counts.get(node_class, 0) >= args.max_hard_per_class:
                                        continue
                            else:
                                # 简单样本：严格限制，避免淹没困难样本
                                if easy_class_counts.get(node_class, 0) >= args.max_easy_per_class:
                                    continue

                            # 构造符合 SFT 格式的消息流
                            sft_messages = [{"role": "system", "content": V_GRAPH_AGENT_INSTRUCTION}]
                            for step in entry["traj"]:
                                sft_messages.append({"role": "user", "content": step["user"]})
                                sft_messages.append({"role": "assistant", "content": step["assistant"]})

                            f_data.write(json.dumps({
                                "dataset": args.dataset,
                                "node_class": node_class,
                                "difficulty_score": difficulty_score,
                                "is_hard": is_hard,
                                "messages": sft_messages
                            }, ensure_ascii=False) + "\n")

                            if is_hard:
                                hard_class_counts[node_class] = hard_class_counts.get(node_class, 0) + 1
                            else:
                                easy_class_counts[node_class] = easy_class_counts.get(node_class, 0) + 1
                            saved_traj_count += 1
                            
    except KeyboardInterrupt:
        logger.info("用户中断执行。")
    finally:
        executor.shutdown(wait=True)
        total_hard = sum(hard_class_counts.values())
        total_easy = sum(easy_class_counts.values())
        logger.info(f"蒸馏结束。成功轨迹: {saved_traj_count} 条（困难: {total_hard} | 简单: {total_easy}），"
                    f"失败节点: {failed_node_count} 个")
        if failed_node_count > 0:
            logger.info(f"  失败节点详情 → {failures_path}")

    # 自动转换 Parquet
    if os.path.exists(training_data_path):
        try:
            df = pd.read_json(training_data_path, lines=True)
            df.to_parquet(f"distill/{args.dataset}_vgraph_training.parquet", index=False)
            logger.info("✅ 训练集 Parquet 已生成。")
        except Exception as e:
            logger.error(f"❌ Parquet 转换失败: {e}")

    # 随机抽取 5 条轨迹，将每条第一步的图片保存到 distill/{dataset}_samples/ 供人工校验
    _save_sample_images(training_data_path, args.dataset, n=5)

if __name__ == "__main__":
    asyncio.run(main_async())