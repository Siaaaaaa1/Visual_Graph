import json
import random
import io
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from .graph_visualizer import GraphVisualizer
import re

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
        self.max_steps = max_steps
        self.node_text_db = node_text_db
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
        self.current_image = None
        self.episode_color_seed = random.randint(0, 1000000)

    def _format_legend(self, legend_dict: Dict[str, str]) -> str:
        items = [f"{color}: {cls}" for color, cls in legend_dict.items()]
        items.sort(key=lambda x: 0 if "Black" in x else 1)
        return "; ".join(items)

    def reset(self, kwargs: Dict[str, Any]) -> str:
        self._reset_internal()

        self.center_id = kwargs["center_id"]
        self.center_text = kwargs.get(
            "center_text", 
            self.node_text_db.get(str(self.center_id), "No text available.")
        )
        self.answer = kwargs["answer"]
        
        stats = self.visualizer.get_node_degree_info(self.center_id)
        # 优化点 1: 候选类别逻辑在 visualizer 中已增强
        candidates = self.visualizer.get_candidate_classes(self.center_id, top_k=100)
        candidates_str = "\n".join(candidates)
        
        # 优化点 5: 初始图设定为 2-hop, 20节点
        # img_bytes, legend_dict = self.visualizer.draw_subgraph(
        #     self.center_id, 
        #     view_mode="center",
        #     max_nodes=1,
        #     color_seed=self.episode_color_seed 
        # )
        img_bytes, legend_dict = self.visualizer.draw_subgraph(
            self.center_id, 
            view_mode="2-hop",  # 修改：从 center 改为 2-hop
            max_nodes=20,       # 修改：从 1 改为 20
            color_seed=self.episode_color_seed 
        )
        
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pil_img = pil_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        self.current_image = np.array(pil_img)
        
        infos = {
            "center_id": self.center_id,
            "answer": self.answer,
            "step": self.step_count
        }
        legend_str = self._format_legend(legend_dict)
        # Obs 纯文本
        obs = (
            f"Current Agent Task: Classify Node {self.center_id}.\n"
            f"Center Node Info:\n"
            f"- Text: {self.center_text}\n"
            f"- In-Degree: {stats['in_degree']}, Out-Degree: {stats['out_degree']}\n"
            f"- 1-Hop Neighbors: {stats['neighbor_count_1hop']}\n\n"
            f"Candidate Categories: {candidates_str}\n\n"
            f"Current View: Initial 2-hop subgraph (Max 20 nodes). Use 'check_graph' to update view."
            f"legend: {legend_str}"
        )

        return obs, self.current_image, infos

    def step(self, raw_input: str):
        """
        raw_input: 这是一个混合容错接口。
        1. 它可以是 projection 清洗过的纯命令 (例如 'check_node:123')
        2. 也可以是包含 XML 标签的原始字符串 (例如 '<action>check_node:123</action>')
        """
        
        # ------------------- 容错解析逻辑 -------------------
        current_action = ""
        think_content = None
        summary_content = None

        # 1. 优先尝试正则提取 (应对带标签的情况)
        # re.DOTALL 让 . 匹配换行符
        action_match = re.search(r"<action>(.*?)</action>", raw_input, re.DOTALL | re.IGNORECASE)
        
        if action_match:
            # Case A: 即使传进来的是 XML，我们也能提取
            current_action = action_match.group(1).strip()
            
            # 顺便提取一下 think/summary (如果有的话)，方便 info 记录
            t_match = re.search(r"<think>(.*?)</think>", raw_input, re.DOTALL | re.IGNORECASE)
            if t_match: think_content = t_match.group(1).strip()
            
            s_match = re.search(r"<summary>(.*?)</summary>", raw_input, re.DOTALL | re.IGNORECASE)
            if s_match: summary_content = s_match.group(1).strip()
            
        else:
            # Case B: 没有标签，假设输入本身就是纯命令 (应对 Projection 后的情况)
            current_action = raw_input.strip()
        
        # 2. 空值检查
        # 如果 projection 失败传了空字符串，或者 xml 提取出来是空的
        if not current_action:
            err_obs = "Error: Invalid action format. Could not parse command. Please check your output format."
            if self.current_image is not None:
                img_ret = self.current_image.copy()
            else:
                img_ret = np.zeros((1024, 1024, 3), dtype=np.uint8)
            
            info = {
                "step": self.step_count,
                "seen_nodes": list(self.seen_nodes),
                "won": False,
                "parsed_think": think_content, 
                "parsed_summary": summary_content,
                "parsed_action": "PARSE_ERROR"
            }
            return err_obs, img_ret, 0, False, info

        # ------------------------------------------------------------------

        if self.done:
            if self.current_image is not None:
                img_ret = self.current_image.copy()
            else:
                img_ret = np.zeros((1024, 1024, 3), dtype=np.uint8)
            return "", img_ret, 0, True, {}

        self.step_count += 1
        reward = 0
        done = False
        obs = ""
        
        if current_action.startswith("check_graph:"):
            try:
                # 兼容性处理：无论上游是否处理过，这里再做一次分割是最安全的
                params = current_action.split(":", 1)[1].strip().split(",")
                view_mode = params[0].strip()
                max_nodes = int(params[1].strip())

                img_bytes, legend_dict = self.visualizer.draw_subgraph(
                    self.center_id,
                    view_mode=view_mode,
                    max_nodes=max_nodes,
                    color_seed=self.episode_color_seed 
                )
                
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pil_img = pil_img.resize((1024, 1024), Image.Resampling.LANCZOS)
                self.current_image = np.array(pil_img)
                
                legend_str = self._format_legend(legend_dict)
                obs = f"Graph view updated (Mode: {view_mode}, Max: {max_nodes}).\nLegend: {legend_str}"
            except Exception as e:
                obs = f"Error in check_graph: {str(e)}. Use format: check_graph:view_mode,max_nodes"

        elif current_action.startswith("check_node:") or current_action.startswith("check_nodes:"):
            node_ids = []
            try:
                content_str = current_action.split(":", 1)[1].strip()
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
                    texts.append(f"Node {node_id} Text:\n{text[:400]}")
                obs = "\n\n".join(texts)
            else:
                obs = "Invalid node ID format."

        elif current_action.startswith("final:"):
            pred = current_action.split(":", 1)[1].strip()
            obs = "Final answer submitted."
            done = True
            self.done = True
            if pred.lower().strip().strip(".'\"") == self.answer.lower().strip().strip(".'\""):
                reward = 1
        else:
            # 这里的 raw_input 可能很长，info 里只记录截断的，避免日志爆炸
            display_act = current_action[:50] + "..." if len(current_action) > 50 else current_action
            obs = f"Invalid action: '{display_act}' is not a valid command."

        if not done and self.step_count >= self.max_steps:
            done = True
            self.done = True

        info = {
            "step": self.step_count,
            "seen_nodes": list(self.seen_nodes),
            "won": bool(reward),
            "parsed_think": think_content, 
            "parsed_summary": summary_content, 
            "parsed_action": current_action
        }
        
        step_image = self.current_image.copy()

        return obs, step_image, reward, done, info


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
                obs, img, r, d, info = env.step(act)
                text_obs.append(obs)
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