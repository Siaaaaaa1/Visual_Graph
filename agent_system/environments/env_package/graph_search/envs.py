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
        """
        self.max_steps = max_steps
        self.node_text_db = node_text_db
        
        # 初始化可视化引擎
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
        self.episode_color_seed = random.randint(0, 1000000)

    def _format_legend(self, legend_dict: Dict[str, str]) -> str:
        items = [f"{color}: {cls}" for color, cls in legend_dict.items()]
        items.sort(key=lambda x: 0 if "Black" in x else 1)
        return "; ".join(items)

    def reset(self, kwargs: Dict[str, Any]) -> str:
        """
        重置环境。
        初始状态：生成并返回中心节点图像。
        """
        self._reset_internal()

        self.center_id = kwargs["center_id"]
        self.center_text = kwargs.get(
            "center_text", 
            self.node_text_db.get(str(self.center_id), "No text available.")
        )
        self.answer = kwargs["answer"]
        
        # 1. 统计信息
        stats = self.visualizer.get_node_degree_info(self.center_id)
        
        # 2. 候选类别
        candidates = self.visualizer.get_candidate_classes(self.center_id, top_k=100)
        candidates_str = ", ".join(candidates)
        
        # 3. 初始绘图
        img_bytes, legend_dict = self.visualizer.draw_subgraph(
            self.center_id, 
            view_mode="center",
            max_nodes=1,
            color_seed=self.episode_color_seed 
        )
        
        infos = {
            "center_id": self.center_id,
            "answer": self.answer,
            "step": self.step_count
        }
        
        # 强制 Resize 固定分辨率
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pil_img = pil_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        self.current_image = np.array(pil_img)
        
        # Reset 必须包含 <image>
        obs = (
            f"Current Agent Task: Classify Node {self.center_id}.\n"
            f"Center Node Info:\n"
            f"- Text: {self.center_text}\n"
            f"- In-Degree: {stats['in_degree']}, Out-Degree: {stats['out_degree']}\n"
            f"- 1-Hop Neighbors: {stats['neighbor_count_1hop']}\n\n"
            f"Candidate Categories: {candidates_str}\n\n"
            f"Current View: Center Node Only. Use 'check_graph' to see neighbors.\n"
            f"<image>" 
        )

        return obs, self.current_image, infos

    def step(self, action: str):
        if self.done:
            # ✅ 修复 1: 返回 5 个值 (obs, img, reward, done, info)
            # ✅ 修复 2: 即使 Done 也要返回有效图像，防止 Batch Collate 报错
            if self.current_image is not None:
                img_ret = self.current_image.copy()
            else:
                img_ret = np.zeros((1024, 1024, 3), dtype=np.uint8)
            
            return "", img_ret, 0, True, {}

        self.step_count += 1
        reward = 0
        done = False
        obs = ""
        
        # --- 动作处理逻辑 ---
        
        if action.startswith("check_graph:"):
            try:
                params = action.split(":", 1)[1].strip().split(",")
                view_mode = params[0].strip()
                max_nodes = int(params[1].strip())

                img_bytes, legend_dict = self.visualizer.draw_subgraph(
                    self.center_id,
                    view_mode=view_mode,
                    max_nodes=max_nodes,
                    color_seed=self.episode_color_seed 
                )
                
                # 更新 current_image (Resize)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_img = pil_img.resize((1024, 1024), Image.Resampling.LANCZOS)
                self.current_image = np.array(pil_img)
                
                legend_str = self._format_legend(legend_dict)
                obs = f"Graph view updated (Mode: {view_mode}, Max: {max_nodes}).\nLegend: {legend_str}"
            except Exception as e:
                obs = f"Error in check_graph: {str(e)}. Use format: check_graph:view_mode,max_nodes"

        elif action.startswith("check_node:") or action.startswith("check_nodes:"):
            # 不更新图像，仅查文本
            node_ids = []
            try:
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
                for node_id in node_ids[:5]:
                    self.seen_nodes.add(node_id)
                    text = self.node_text_db.get(str(node_id), "No text available.")
                    texts.append(f"Node {node_id} Text:\n{text}")
                obs = "\n\n".join(texts)
            else:
                obs = "Invalid node ID format."

        elif action.startswith("final:"):
            pred = action.split(":", 1)[1].strip()
            obs = "Final answer submitted."
            done = True
            self.done = True
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
        
        # 始终在 obs 末尾追加 <image>
        obs += "\n<image>"

        # 始终返回 current_image 的副本 (不能是 None)
        step_image = self.current_image.copy()

        return obs, step_image, reward, done, info


def build_graph_search_envs(
    seed: int,
    env_num: int,
    group_n: int,
    is_train: bool,
    env_config
):
    """
    构建批量图搜索环境。
    """
    batch_size = env_num * group_n
    max_steps = env_config.max_steps
    dataset_name = getattr(env_config, "dataset_name", "cora")
    dataset_dir = getattr(env_config, "dataset_dir", "./datasets")

    with open(env_config.node_text_path, "r", encoding="utf-8") as f:
        node_text_db = json.load(f)

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
                obs, img, info = env.reset(kw)
                text_obs.append(obs)
                image_obs.append(img.copy())
                infos.append(info)
            return text_obs, image_obs, infos

        def step(self, actions: List[str]):
            text_obs, image_obs, rewards, dones, infos = [], [], [], [], []
            for env, act in zip(envs, actions):
                # 这里解包 5 个值，与 GraphSearchEnv.step 对应
                obs, img, r, d, info = env.step(act)
                
                text_obs.append(obs)
                
                # img 现在永远是 np.array，不为 None
                if img is not None:
                    image_obs.append(img.copy()) 
                else:
                    image_obs.append(None)
                
                rewards.append(r)
                dones.append(d)
                infos.append(info)
            return text_obs, image_obs, rewards, dones, infos

        def close(self):
            pass

    return BatchGraphSearchEnv()