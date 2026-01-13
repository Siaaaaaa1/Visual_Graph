import json
import random
import io
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from .graph_visualizer import GraphVisualizer

# ============================================================
# 单回合图搜索环境 (Single-Turn Graph Search Environment)
# ============================================================

class GraphSearchEnv:
    def __init__(self, 
                 max_steps: int, 
                 node_text_db: Dict[str, str], 
                 dataset_name: str, 
                 dataset_dir: str,
                 shared_graph_data: Optional[Tuple[Dict, Dict]] = None):
        """
        初始化图搜索环境。
        
        Args:
            shared_graph_data: 预加载的图数据，用于多环境共享内存，减少开销。
        """
        self.max_steps = max_steps
        self.node_text_db = node_text_db
        
        # 初始化可视化引擎 (数据仅加载一次，或使用共享数据)
        self.visualizer = GraphVisualizer(
            dataset_name=dataset_name, 
            dataset_dir=dataset_dir,
            shared_data=shared_graph_data
        )
        self._reset_internal()

    def _reset_internal(self):
        """重置内部状态变量。"""
        self.step_count = 0
        self.seen_nodes = set()
        self.done = False
        self.current_image = None
        # 生成本局唯一的颜色种子，确保同一 Episode 内类别颜色映射保持一致，避免 Agent 混淆
        self.episode_color_seed = random.randint(0, 1000000)

    def _format_legend(self, legend_dict: Dict[str, str]) -> str:
        """格式化图例字符串，确保中心节点 (Black) 始终置顶。"""
        items = [f"{color}: {cls}" for color, cls in legend_dict.items()]
        # 排序策略：带有 "Black" (Center Node) 的项排在最前，其余按默认顺序
        items.sort(key=lambda x: 0 if "Black" in x else 1)
        return "; ".join(items)

    def reset(self, kwargs: Dict[str, Any]) -> str:
        """
        重置环境并开始新的 Episode。
        
        Args:
            kwargs: 必须包含 'center_id' 和 'answer'，可选 'center_text'。
        """
        self._reset_internal()

        self.center_id = kwargs["center_id"]
        # 优先从 kwargs 获取文本，若无则查库
        self.center_text = kwargs.get(
            "center_text", 
            self.node_text_db.get(str(self.center_id), "No text available.")
        )
        self.answer = kwargs["answer"]
        
        # 1. 获取中心节点的拓扑统计信息
        stats = self.visualizer.get_node_degree_info(self.center_id)
        
        # 2. 绘制初始视图 (默认 1-hop, 传入 seed 锁定配色)
        img_bytes, legend_dict = self.visualizer.draw_subgraph(
            self.center_id, 
            mode="1-hop", 
            max_nodes=10,
            color_seed=self.episode_color_seed 
        )
        
        infos = {
            "center_id": self.center_id,
            "answer": self.answer,
            "step": self.step_count
        }
        # 转换为 Numpy 数组供 VLM 处理
        self.current_image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        legend_str = self._format_legend(legend_dict)

        # 构建初始文本观察 (Observation)
        obs = (
            f"Current Agent Task: Classify Node {self.center_id}.\n"
            f"Center Node Info:\n"
            f"- Text: {self.center_text}\n"
            f"- In-Degree: {stats['in_degree']}, Out-Degree: {stats['out_degree']}\n"
            f"- 1-Hop Neighbors (Undirected): {stats['neighbor_count_1hop']}\n"
            f"- 2-Hop Neighbors (Undirected): {stats['neighbor_count_2hop']}\n\n"
            f"Initial Graph View (1-hop, max 10 nodes):\n{legend_str}\n"
        )

        return obs, self.current_image, infos

    def step(self, action: str):
        if self.done:
            return "", 0, True, {}

        self.step_count += 1
        reward = 0
        done = False
        obs = ""
        
        # --- 动作处理逻辑 ---
        
        # 动作 A: check_graph (调整图视图)
        if action.startswith("check_graph:"):
            try:
                # 预期格式: check_graph:hop_mode,rank_mode,max_nodes
                # 示例: check_graph:2-hop,sim,20
                params = action.split(":", 1)[1].strip().split(",")

                hop_mode = params[0].strip()
                # 兼容性处理：若参数缺失，提供默认值
                rank_mode = params[1].strip() if len(params) > 1 else "hop"
                max_nodes = int(params[2].strip()) if len(params) > 2 else 15

                img_bytes, legend_dict = self.visualizer.draw_subgraph(
                    self.center_id,
                    mode=hop_mode,
                    rank_mode=rank_mode,
                    max_nodes=max_nodes,
                    color_seed=self.episode_color_seed # 保持本局配色一致
                )
                
                self.current_image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                legend_str = self._format_legend(legend_dict)
                
                obs = f"Graph view updated (Mode: {hop_mode}, Rank: {rank_mode}, Max: {max_nodes}).\nLegend: {legend_str}"
            except Exception as e:
                obs = f"Error in check_graph: {str(e)}. Use format: check_graph:hop_mode,rank_mode,max_nodes"

        # 动作 B: check_node / check_nodes (查阅节点文本)
        elif action.startswith("check_node:") or action.startswith("check_nodes:"):
            node_ids = []
            try:
                # 统一格式处理，支持: check_node:123 和 check_nodes:[123, 456]
                content_str = action.split(":", 1)[1].strip()
                content_str = content_str.replace("[", "").replace("]", "")
                
                parts = content_str.split(",")
                for p in parts:
                    if p.strip().isdigit():
                        node_ids.append(int(p.strip()))
            except:
                node_ids = []

            if node_ids:
                texts = []
                # 上下文保护：限制单步最多读取 5 个节点文本
                for node_id in node_ids[:5]:
                    self.seen_nodes.add(node_id)
                    text = self.node_text_db.get(str(node_id), "No text available.")
                    texts.append(f"Node {node_id} Text:\n{text}")
                obs = "\n\n".join(texts)
            else:
                obs = "Invalid node ID format."

        # 动作 C: final (提交答案)
        elif action.startswith("final:"):
            pred = action.split(":", 1)[1].strip()
            obs = "Final answer submitted."
            done = True
            self.done = True
            # 评估：简单的大小写不敏感匹配
            if pred.lower() == self.answer.lower():
                reward = 1
        else:
            obs = "Invalid action."

        # 超时检查
        if not done and self.step_count >= self.max_steps:
            done = True
            self.done = True

        info = {
            "step": self.step_count,
            "seen_nodes": list(self.seen_nodes),
            "won": bool(reward)
        }

        return obs, self.current_image, reward, done, info


# ============================================================
# 批量环境包装器 (Batch Wrapper)
# ============================================================

def build_graph_search_envs(
    seed: int,
    env_num: int,
    group_n: int,
    is_train: bool,
    env_config
):
    """
    构建批量图搜索环境，支持 Ray/Verl 等分布式框架。
    关键优化：在主进程一次性预加载图数据，避免每个子环境重复 IO。
    """
    batch_size = env_num * group_n
    max_steps = env_config.max_steps
    dataset_name = getattr(env_config, "dataset_name", "cora")
    dataset_dir = getattr(env_config, "dataset_dir", "./datasets")

    # 1. 加载文本数据库
    with open(env_config.node_text_path, "r", encoding="utf-8") as f:
        node_text_db = json.load(f)

    # 2. 全局预加载图结构数据 (内存优化)
    print(f"[build_envs] Pre-loading graph data for {dataset_name}...")
    shared_graph_data = GraphVisualizer.load_graph_data(dataset_name, dataset_dir)
    print(f"[build_envs] Load complete.")

    # 3. 实例化多个环境
    envs = [
        GraphSearchEnv(
            max_steps=max_steps, 
            node_text_db=node_text_db, 
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            shared_graph_data=shared_graph_data
        )
        for _ in range(batch_size)
    ]

    class BatchGraphSearchEnv:
        def __init__(self):
            self.num_envs = batch_size

        def reset(self, kwargs):
            text_obs, image_obs, infos = [], [], []
            for env, kw in zip(envs, kwargs):
                obs = env.reset(kw)
                text_obs.append(obs)
                image_obs.append(env.current_image)
                infos.append({})
            return text_obs, image_obs, infos

        def step(self, actions: List[str]):
            text_obs, image_obs, rewards, dones, infos = [], [], [], [], []
            for env, act in zip(envs, actions):
                obs, img, r, d, info = env.step(act)
                text_obs.append(obs)
                image_obs.append(img)
                rewards.append(r)
                dones.append(d)
                infos.append(info)
            return text_obs, image_obs, rewards, dones, infos

        def close(self):
            pass

    return BatchGraphSearchEnv()