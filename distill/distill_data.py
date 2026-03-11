import os
import sys
import json
import matplotlib as _mpl
_mpl.rcParams['font.family'] = 'DejaVu Sans'
_mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
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

TASK_CONCURRENCY = 10   # 同时运行的轨迹任务数（env + API 全链路并发上限）

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
THINK_PATTERN  = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)

# 中文标点检测（出现在 action 内则格式无效）
_CN_PUNCT = re.compile(r"[，。；：""''（）【】《》、！？]")

def _is_valid_step(assistant_text: str) -> bool:
    """
    校验单步 assistant 输出格式是否合法：
    1. 必须包含 <thinking>...</thinking>
    2. 必须包含 <action>...</action>
    3. <action> 内不得出现中文标点
    """
    if not THINK_PATTERN.search(assistant_text):
        return False
    action_match = ACTION_PATTERN.search(assistant_text)
    if not action_match:
        return False
    action_content = action_match.group(1)
    if _CN_PUNCT.search(action_content):
        return False
    return True

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
        # qwen3 系列默认 enable_thinking=True：思考内容在 reasoning_content，
        # content 只放最终答案，当模型没有输出 action 时 content="" (0字符)。
        # 设为 False 强制全部输出（含 <thinking> 标签）走 content，
        # 与 prompt 要求的 "<thinking>...</thinking><action>...</action>" 格式一致。
        "enable_thinking": False,
    }
    async with sem:  # sem 由调用方传入，与 TASK_CONCURRENCY 对齐
        timeout = aiohttp.ClientTimeout(total=120, connect=15, sock_read=90)
        async with session.post(API_URL, json=payload, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text[:300]}")
            data = await response.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            # 防御：若 content 仍为空但 reasoning_content 有值（模型忽略了 enable_thinking=False），
            # 将思考内容包回 <thinking> 标签，保证后续正则能正常解析
            if not content:
                reasoning = msg.get("reasoning_content") or ""
                if reasoning:
                    content = f"<thinking>{reasoning}</thinking>"
            return content

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
                # 打印 API 返回内容（截取前 300 字符，避免 base64 图片膨胀日志）
                preview = ans_text.replace("\n", " ")[:300]
                logger.info(f"[Task {tid}] attempt={attempt_idx} step={step_idx} — ← API返回 ({len(ans_text)}字符): {preview}")
            except Exception as e:
                logger.error(f"[Task {tid}] attempt={attempt_idx} step={step_idx} — API ERROR: {repr(e)}")
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
            # 格式校验：每一步都必须有合法的 <thinking> + <action>（英文标点）
            bad_steps = [i for i, s in enumerate(raw_traj_steps)
                         if not _is_valid_step(s.get("assistant", ""))]
            if bad_steps:
                logger.debug(f"[Task {tid}] attempt={attempt_idx} 格式不合法（步骤 {bad_steps} 含中文标点或缺 think），丢弃本次轨迹")
                last_failure_type = "invalid_format"
            else:
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
        "max_tokens": 5,
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

    # ── 1. API 连通性检查 ──
    logger.info(f"🔌 验证 DashScope API ({API_URL})...")
    if not await _check_api_connectivity():
        return

    # ── 2. 多模态链路测试（含图片） ──
    logger.info("🖼️  测试多模态 API（含图片）...")
    try:
        _dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        _b64 = img_to_b64(_dummy_img)
        _test_msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What color is this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64}"}}
        ]}]
        _headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
        _payload  = {"model": "qwen3-vl-plus", "messages": _test_msgs, "max_tokens": 10}
        _timeout  = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        async with aiohttp.ClientSession() as _s:
            async with _s.post(API_URL, json=_payload, headers=_headers, timeout=_timeout) as _r:
                _body = await _r.json()
                if _r.status == 200:
                    logger.info(f"✅ 多模态 API 通过，回复: {str(_body['choices'][0]['message']['content'])[:80]}")
                else:
                    logger.error(f"❌ 多模态 API 错误 HTTP {_r.status}: {str(_body)[:300]}")
                    return
    except asyncio.TimeoutError:
        logger.error("❌ 多模态 API 超时（30s），请检查模型名称或网络")
        return
    except Exception as _e:
        logger.error(f"❌ 多模态 API 测试失败: {repr(_e)}")
        return

    # ── 3. 数据准备 ──
    all_tasks, text_db, shared_payload = prepare_shared_assets(args.dataset, args.dataset_dir)

    stats_path         = f"distill/{args.dataset}_vgraph_stats.jsonl"
    training_data_path = f"distill/{args.dataset}_vgraph_training.jsonl"
    failures_path      = f"distill/{args.dataset}_debug_failures.jsonl"

    completed_tids = set()
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        completed_tids.add(json.loads(line)["tid"])
                    except Exception:
                        pass

    pending_tasks = [t for t in all_tasks if t["center_id"] not in completed_tids]
    random.shuffle(pending_tasks)
    tasks_to_run = pending_tasks[:args.num_tasks]
    total = len(tasks_to_run)

    logger.info(f"🚀 启动蒸馏 | 并发={TASK_CONCURRENCY} | 待处理={total} 个节点")
    logger.info(f"   trajectories_per_node={args.trajectories_per_node} | "
                f"max_attempts={args.max_attempts} | "
                f"max_hard/class={args.max_hard_per_class or '不限'} | "
                f"max_easy/class={args.max_easy_per_class}")

    # ── 4. 并发控制与线程池 ──
    task_sem  = asyncio.Semaphore(TASK_CONCURRENCY)          # 任务级并发上限
    api_sem   = asyncio.Semaphore(TASK_CONCURRENCY)          # API 调用并发（与任务对齐）
    connector = aiohttp.TCPConnector(limit=TASK_CONCURRENCY)
    executor  = ThreadPoolExecutor(max_workers=TASK_CONCURRENCY * 2)

    saved_traj_count = 0
    failed_node_count = 0
    hard_class_counts: dict = {}
    easy_class_counts: dict = {}

    def _save_entry(entry: dict, f_data) -> bool:
        """将单条轨迹写入训练文件，返回是否实际保存（配额未满）"""
        node_class     = entry.get("node_class", "unknown")
        node_attempts  = entry.get("node_attempts", args.max_attempts)
        node_successes = entry.get("node_successes", 1)
        success_rate   = node_successes / node_attempts if node_attempts > 0 else 1.0
        difficulty_score = round(1.0 - success_rate, 4)
        is_hard = success_rate < 0.5

        if is_hard:
            if args.max_hard_per_class > 0 and hard_class_counts.get(node_class, 0) >= args.max_hard_per_class:
                return False
        else:
            if easy_class_counts.get(node_class, 0) >= args.max_easy_per_class:
                return False

        sft_messages = [{"role": "system", "content": V_GRAPH_AGENT_INSTRUCTION}]
        for step in entry["traj"]:
            sft_messages.append({"role": "user",      "content": step["user"]})
            sft_messages.append({"role": "assistant",  "content": step["assistant"]})

        f_data.write(json.dumps({
            "dataset":          args.dataset,
            "node_class":       node_class,
            "difficulty_score": difficulty_score,
            "is_hard":          is_hard,
            "messages":         sft_messages,
        }, ensure_ascii=False) + "\n")

        if is_hard:
            hard_class_counts[node_class] = hard_class_counts.get(node_class, 0) + 1
        else:
            easy_class_counts[node_class] = easy_class_counts.get(node_class, 0) + 1
        return True

    try:
        async with aiohttp.ClientSession(connector=connector) as session:

            # 有界并发包装：每个任务在 task_sem 内运行，完成即释放槽位
            async def run_bounded(t):
                async with task_sem:
                    return await run_task_async(
                        t, text_db, shared_payload, session, api_sem, executor,
                        trajectories_per_node=args.trajectories_per_node,
                        max_attempts=args.max_attempts,
                    )

            futures = [asyncio.create_task(run_bounded(t)) for t in tasks_to_run]
            done_count = 0

            for fut in asyncio.as_completed(futures):
                trajs, failure_info = await fut
                done_count += 1

                # ── 失败节点 ──
                if failure_info is not None:
                    failed_node_count += 1
                    with open(failures_path, 'a', encoding='utf-8') as f_fail:
                        f_fail.write(json.dumps(failure_info, ensure_ascii=False) + "\n")
                    logger.info(
                        f"[{done_count}/{total}] ❌ node={failure_info['task_id']} "
                        f"class={failure_info['answer']} | {failure_info['failure_type']} "
                        f"(尝试 {failure_info['total_attempts']} 次)"
                    )

                # ── 成功节点 ──
                if trajs:
                    saved_this = 0
                    node_id    = trajs[0]["task_id"]
                    node_class = trajs[0]["node_class"]
                    sr         = trajs[0].get("node_successes", 1) / max(trajs[0].get("node_attempts", 1), 1)
                    difficulty = "困难" if sr < 0.5 else "简单"

                    with open(stats_path, 'a', encoding='utf-8') as f_stats, \
                         open(training_data_path, 'a', encoding='utf-8') as f_data:
                        f_stats.write(json.dumps({"tid": node_id, "count": len(trajs)}) + "\n")
                        for entry in trajs:
                            if _save_entry(entry, f_data):
                                saved_this     += 1
                                saved_traj_count += 1

                    logger.info(
                        f"[{done_count}/{total}] ✅ node={node_id} class={node_class} "
                        f"({difficulty} sr={sr:.2f}) | 本节点保存 {saved_this}/{len(trajs)} 条 "
                        f"| 累计 {saved_traj_count} 条 (失败节点 {failed_node_count})"
                    )

    except KeyboardInterrupt:
        logger.info("用户中断执行。")
    finally:
        executor.shutdown(wait=False)
        total_hard = sum(hard_class_counts.values())
        total_easy = sum(easy_class_counts.values())
        logger.info(
            f"\n{'='*60}\n蒸馏结束 | 成功轨迹={saved_traj_count} (困难={total_hard} 简单={total_easy}) "
            f"| 失败节点={failed_node_count}\n{'='*60}"
        )

    # ── 5. 转换 Parquet ──
    if os.path.exists(training_data_path):
        try:
            df = pd.read_json(training_data_path, lines=True)
            df.to_parquet(f"distill/{args.dataset}_vgraph_training.parquet", index=False)
            logger.info("✅ Parquet 已生成。")
        except Exception as e:
            logger.error(f"❌ Parquet 转换失败: {e}")

    # ── 6. 样本图片 ──
    _save_sample_images(training_data_path, args.dataset, n=5)

if __name__ == "__main__":
    asyncio.run(main_async())