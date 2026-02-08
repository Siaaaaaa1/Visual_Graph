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
                 graph_setting: str = "standard_20", 
                 # 核心：支持接收完整的 4元组 (包含矩阵)
                 shared_graph_data: Optional[Tuple[Dict, Dict, Dict, Any]] = None):
        self.max_steps = max_steps
        self.node_text_db = node_text_db
        self.graph_setting = graph_setting
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
        
        # 解析 Graph Setting
        show_center_text = True
        init_view_mode = "center"
        init_max_nodes = 1
        mask_neighbors_init = False
        
        setting = self.graph_setting.lower()
        
        if "no_text" in setting or "zero_shot" in setting:
            show_center_text = False
            init_view_mode = "center"
            init_max_nodes = 1
        elif "center_only" in setting or "start_from_center" in setting:
            show_center_text = True
            init_view_mode = "center"
            init_max_nodes = 1
        else:
            show_center_text = True
            init_view_mode = "1-hop+sim"
            if "50" in setting:
                init_max_nodes = 50
            else:
                init_max_nodes = 20
            if "masked" in setting or "no_color" in setting:
                mask_neighbors_init = True
            else:
                mask_neighbors_init = False

        if show_center_text:
            raw_text = self.node_text_db.get(str(self.center_id), "No text available.")
            self.center_text = raw_text
        else:
            self.center_text = "[Text Hidden. Explore to read.]"

        self.answer = kwargs["answer"]
        
        stats = self.visualizer.get_node_degree_info(self.center_id)
        candidates = self.visualizer.get_candidate_classes(self.center_id, top_k=100)
        candidates_str = "\n".join(candidates)
        
        img_bytes, legend_dict = self.visualizer.draw_subgraph(
            self.center_id, 
            view_mode=init_view_mode,
            max_nodes=init_max_nodes,
            color_seed=self.episode_color_seed,
            mask_neighbors=mask_neighbors_init
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
        
        if init_view_mode == "center":
            view_desc = "Center Node Only (Initial State)"
        else:
            view_desc = f"Initial Subgraph (Mode: {init_view_mode}, Max: {init_max_nodes} nodes, Colors: {'Hidden' if mask_neighbors_init else 'Shown'})"

        obs = (
            f"Current Agent Task: Classify Node {self.center_id}.\n"
            f"Center Node Info:\n"
            f"- Text: {self.center_text}\n"
            f"- In-Degree: {stats['in_degree']}, Out-Degree: {stats['out_degree']}\n"
            f"- 1-Hop Neighbors: {stats['neighbor_count_1hop']}\n\n"
            f"Candidate Categories: {candidates_str}\n\n"
            f"Current View: {view_desc}. Use 'check_graph' to update view."
            f"legend: {legend_str}"
        )

        return obs, self.current_image, infos

    def step(self, raw_input: str):
        # 容错解析逻辑
        current_action = ""
        think_content = None
        summary_content = None

        action_match = re.search(r"<action>(.*?)</action>", raw_input, re.DOTALL | re.IGNORECASE)
        
        if action_match:
            current_action = action_match.group(1).strip()
            t_match = re.search(r"<think>(.*?)</think>", raw_input, re.DOTALL | re.IGNORECASE)
            if t_match: think_content = t_match.group(1).strip()
            s_match = re.search(r"<summary>(.*?)</summary>", raw_input, re.DOTALL | re.IGNORECASE)
            if s_match: summary_content = s_match.group(1).strip()
        else:
            current_action = raw_input.strip()
        
        if not current_action:
            err_obs = "Error: Invalid action format."
            img_ret = self.current_image.copy() if self.current_image is not None else np.zeros((1024, 1024, 3), dtype=np.uint8)
            info = {
                "step": self.step_count,
                "seen_nodes": list(self.seen_nodes),
                "won": False,
                "parsed_think": think_content, 
                "parsed_summary": summary_content,
                "parsed_action": "PARSE_ERROR"
            }
            return err_obs, img_ret, 0, False, info

        if self.done:
            img_ret = self.current_image.copy() if self.current_image is not None else np.zeros((1024, 1024, 3), dtype=np.uint8)
            return "", img_ret, 0, True, {}

        self.step_count += 1
        reward = 0
        done = False
        obs = ""
        
        if current_action.startswith("check_graph:"):
            try:
                params = current_action.split(":", 1)[1].strip().split(",")
                view_mode = params[0].strip()
                max_nodes = int(params[1].strip())

                img_bytes, legend_dict = self.visualizer.draw_subgraph(
                    self.center_id,
                    view_mode=view_mode,
                    max_nodes=max_nodes,
                    color_seed=self.episode_color_seed,
                    mask_neighbors=False # 主动check_graph默认显示颜色
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
        
        step_image = self.current_image.copy() if self.current_image is not None else None

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
    graph_setting = getattr(env_config, "graph_setting", "standard_20")

    with open(env_config.node_text_path, "r", encoding="utf-8") as f:
        node_text_db = json.load(f)

    # =========================================================================
    # 核心内存优化：在主进程加载一次数据并构建大矩阵
    # =========================================================================
    print(f"[build_envs] Pre-loading graph data and building feature matrix for {dataset_name}...")
    
    # 1. 加载原始 JSON 数据
    g_data, r_adj, c_map = GraphVisualizer.load_graph_data(dataset_name, dataset_dir)
    
    # 2. 构建特征矩阵 (利用临时 Visualizer 实例，避免手动复制代码逻辑)
    # shared_data 的第4个元素传 None，触发 Visualizer 内部的构建逻辑
    temp_viz = GraphVisualizer(
        dataset_name=dataset_name, 
        dataset_dir=dataset_dir,
        shared_data=(g_data, r_adj, c_map, None) 
    )
    feat_matrix = temp_viz.feat_matrix
    
    print(f"[build_envs] Matrix built successfully. Shape: {feat_matrix.shape}, "
          f"Size: {feat_matrix.nbytes / 1024**2:.2f} MB")
    
    # 3. 打包共享数据 (4元组)
    # 传递这个 payload 给所有 worker，它们将引用同一块内存 (只读)
    shared_payload = (g_data, r_adj, c_map, feat_matrix)
    
    print(f"[build_envs] Shared payload ready. Graph Setting: {graph_setting}")
    # =========================================================================

    envs = [
        GraphSearchEnv(
            max_steps=max_steps, 
            node_text_db=node_text_db, 
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            graph_setting=graph_setting, 
            shared_graph_data=shared_payload # 传入共享数据
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