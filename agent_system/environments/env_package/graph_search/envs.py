import json
import random
from typing import List, Dict, Any, Tuple, Optional
from .graph_visualizer import GraphVisualizer

# ============================================================
# 单回合图搜索环境
# ============================================================

class GraphSearchEnv:
    def __init__(self, 
                 max_steps: int, 
                 node_text_db: Dict[str, str], 
                 dataset_name: str, 
                 dataset_dir: str,
                 shared_graph_data: Optional[Tuple[Dict, Dict]] = None):
        
        self.max_steps = max_steps
        self.node_text_db = node_text_db
        
        # 初始化 Visualizer (数据只加载一次)
        self.visualizer = GraphVisualizer(
            dataset_name=dataset_name, 
            dataset_dir=dataset_dir,
            shared_data=shared_graph_data
        )
        self._reset_internal()

    def _reset_internal(self):
        self.step_count = 0
        self.seen_nodes = set()
        self.done = False
        self.current_image_bytes = None
        # ✅ 新增: 每个 Episode 生成一个随机 Seed，用于控制本局的颜色风格
        self.episode_color_seed = random.randint(0, 1000000)

    def _format_legend(self, legend_dict: Dict[str, str]) -> str:
        items = [f"{color}: {cls}" for color, cls in legend_dict.items()]
        # 把 Black (Center Node) 排在最前面
        items.sort(key=lambda x: 0 if "Black" in x else 1)
        return "; ".join(items)

    def reset(self, kwargs: Dict[str, Any]) -> str:
        self._reset_internal()

        self.center_id = kwargs["center_id"]
        # 允许从 kwargs 传入 text，也可以查表
        self.center_text = kwargs.get(
            "center_text", 
            self.node_text_db.get(str(self.center_id), "No text available.")
        )
        self.answer = kwargs["answer"]
        
        # 1. 获取图统计信息
        stats = self.visualizer.get_node_degree_info(self.center_id)
        
        # 2. 绘制初始图 (传入 color_seed)
        img_bytes, legend_dict = self.visualizer.draw_subgraph(
            self.center_id, 
            mode="1-hop", 
            max_nodes=10,
            color_seed=self.episode_color_seed # ✅ 保持一致性
        )
        self.current_image_bytes = img_bytes
        legend_str = self._format_legend(legend_dict)

        obs = (
            f"Current Agent Task: Classify Node {self.center_id}.\n"
            f"Center Node Info:\n"
            f"- Text: {self.center_text}\n"
            f"- In-Degree: {stats['in_degree']}, Out-Degree: {stats['out_degree']}\n"
            f"- 1-Hop Neighbors (Undirected): {stats['neighbor_count_1hop']}\n"
            f"- 2-Hop Neighbors (Undirected): {stats['neighbor_count_2hop']}\n\n"
            f"Initial Graph View (1-hop, max 10 nodes):\n{legend_str}\n"
        )

        return obs

    def step(self, action: str):
        if self.done:
            return "", 0, True, {}

        self.step_count += 1
        reward = 0
        done = False
        obs = ""
        
        # 动作 1: check_graph
        if action.startswith("check_graph:"):
            try:
                # 格式: check_graph:mode,max_nodes
                # 例如: check_graph:2-hop,20 或 check_graph:1-hop
                params = action.split(":", 1)[1].strip().split(",")
                mode = params[0].strip()
                max_nodes = int(params[1].strip()) if len(params) > 1 else 15
                
                img_bytes, legend_dict = self.visualizer.draw_subgraph(
                    self.center_id,
                    mode=mode,
                    max_nodes=max_nodes,
                    color_seed=self.episode_color_seed # ✅ 使用相同的 seed
                )
                self.current_image_bytes = img_bytes
                legend_str = self._format_legend(legend_dict)
                
                obs = f"Graph view updated (Mode: {mode}, Max: {max_nodes}).\nLegend: {legend_str}"
            except Exception as e:
                obs = f"Error in check_graph: {str(e)}. Use format: check_graph:mode,max_nodes"

        # 动作 2: check_node / check_nodes
        elif action.startswith("check_node:") or action.startswith("check_nodes:"):
            node_ids = []
            try:
                # 统一清理格式，支持 check_node:123 和 check_nodes:[123, 456]
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
                # 限制一次最多读取 5 个，防止 Context 爆炸
                for node_id in node_ids[:5]:
                    self.seen_nodes.add(node_id)
                    text = self.node_text_db.get(str(node_id), "No text available.")
                    texts.append(f"Node {node_id} Text:\n{text}")
                obs = "\n\n".join(texts)
            else:
                obs = "Invalid node ID format."

        # 动作 3: final
        elif action.startswith("final:"):
            pred = action.split(":", 1)[1].strip()
            obs = "Final answer submitted."
            done = True
            self.done = True
            # 简单的大小写不敏感匹配
            if pred.lower() == self.answer.lower():
                reward = 1
        else:
            obs = "Invalid action."

        if not done and self.step_count >= self.max_steps:
            done = True
            self.done = True

        info = {
            "step": self.step_count,
            "seen_nodes": list(self.seen_nodes),
            "won": bool(reward)
        }

        return obs, reward, done, info


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
    batch_size = env_num * group_n
    max_steps = env_config.max_steps
    dataset_name = getattr(env_config, "dataset_name", "cora")
    dataset_dir = getattr(env_config, "dataset_dir", "./datasets")

    # 加载文本数据库
    with open(env_config.node_text_path, "r", encoding="utf-8") as f:
        node_text_db = json.load(f)

    # 全局预加载图数据
    print(f"[build_envs] Pre-loading graph data for {dataset_name}...")
    shared_graph_data = GraphVisualizer.load_graph_data(dataset_name, dataset_dir)
    print(f"[build_envs] Load complete.")

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
                image_obs.append(env.current_image_bytes)
                infos.append({})
            return text_obs, image_obs, infos

        def step(self, actions: List[str]):
            text_obs, image_obs, rewards, dones, infos = [], [], [], [], []
            for env, act in zip(envs, actions):
                obs, r, d, info = env.step(act)
                text_obs.append(obs)
                image_obs.append(env.current_image_bytes)
                rewards.append(r)
                dones.append(d)
                infos.append(info)
            return text_obs, image_obs, rewards, dones, infos

        def close(self):
            pass

    return BatchGraphSearchEnv()