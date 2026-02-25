import json
import random
import io
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from .graph_visualizer import GraphVisualizer # 请确保正确导入
import re

class GraphSearchEnv:
    """
    图搜索智能体环境类。
    用于管理单个大模型智能体在图数据上的探索、动作交互、奖励计算以及视觉图像的生成。
    """
    def __init__(self, 
                 max_steps: int, 
                 node_text_db: Dict[str, str], 
                 dataset_name: str, 
                 dataset_dir: str,
                 graph_setting: str = "standard_20", 
                 shared_graph_data: Optional[Tuple[Dict, Dict, Dict, Any]] = None,
                 tau: float = 0.6,
                 epsilon: float = 0.15): 
        self.max_steps = max_steps
        self.node_text_db = node_text_db
        self.graph_setting = graph_setting
        self.tau = tau
        self.epsilon = epsilon
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
        self.paint_history = {} 
        self.painted_nodes = {} 
        self.mode = "System1"
        self.unlock_target = 0
        self.available_node_types = {'1-hop': 0, 'high_out': 0, 'high_in': 0} 
        self.hindsight_rewards = None
        self.current_node_shapes = {} # [新增] 缓存画面上节点的真实形状

    def _format_legend(self, legend_dict: Dict[str, str]) -> str:
        items = [f"{color}: {cls}" for color, cls in legend_dict.items()]
        items.sort(key=lambda x: 0 if "Black" in x else 1)
        return "; ".join(items)

    def reset(self, kwargs: Dict[str, Any]) -> str:
        self._reset_internal()
        self.center_id = kwargs["center_id"]
        self.answer = kwargs["answer"]
        is_train = kwargs.get("is_train", True)

        center_info = self.visualizer._get_node_info(self.center_id)
        center_proxy_cls = center_info["pred_class"]
        
        neighbors_1hop = set(self.visualizer._get_neighbors(self.center_id, undirected=True))
        all_neighbors_1_2_hop = set(neighbors_1hop)
        for nb in neighbors_1hop:
            all_neighbors_1_2_hop.update(self.visualizer._get_neighbors(nb, undirected=True))
            
        if self.center_id in all_neighbors_1_2_hop:
            all_neighbors_1_2_hop.remove(self.center_id)
        
        same_class_count = 0
        valid_nb_count = 0
        for nb in all_neighbors_1_2_hop:
            nb_proxy_cls = self.visualizer._get_node_info(nb)["pred_class"]
            if nb_proxy_cls == center_proxy_cls:
                same_class_count += 1
            valid_nb_count += 1
            
        h_hat = (same_class_count / valid_nb_count) if valid_nb_count > 0 else 0.0

        if h_hat > self.tau:
            if is_train and random.random() < self.epsilon:
                self.mode = "System2"
            else:
                self.mode = "System1"
        else:
            self.mode = "System2"

        init_view_mode = "2-hop+sim"
        init_max_nodes = 30

        if self.mode == "System1":
            mask_neighbors_init = False
            self.center_text = self.node_text_db.get(str(self.center_id), "No text available.")
            self.unlock_target = 0
        else:
            mask_neighbors_init = True
            self.center_text = "[Text Hidden by Fog of War. Explore to read.]"
            target_raw = max(2, int(5 * (1.0 - h_hat)))
            self.unlock_target = target_raw # 暂存

        stats = self.visualizer.get_node_degree_info(self.center_id)
        candidates_str = "\n".join(self.visualizer.get_candidate_classes(self.center_id, top_k=100))
        
        # [修改] 接收三个返回值，获取实际分配的形状
        img_bytes, legend_dict, self.current_node_shapes = self.visualizer.draw_subgraph(
            self.center_id, 
            view_mode=init_view_mode,
            max_nodes=init_max_nodes,
            color_seed=self.episode_color_seed,
            mask_neighbors=mask_neighbors_init,
            painted_nodes=self.painted_nodes
        )
        
        # 基于渲染结果校准解锁门槛
        self.available_node_types = {'1-hop': 0, 'high_out': 0, 'high_in': 0}
        if self.mode == "System2":
            for shape in self.current_node_shapes.values():
                self.available_node_types[shape] += 1
                
            total_available = sum(self.available_node_types.values())
            self.unlock_target = min(self.unlock_target, total_available)
            if self.unlock_target == 0 and total_available > 0:
                self.unlock_target = 1

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        self.current_image = np.array(pil_img)
        
        mode_desc = "CLEAR WEATHER (System 1)" if self.mode == "System1" else "FOG OF WAR (System 2)"
        obs = (
            f"=== Environment Mode: {mode_desc} ===\n"
            f"Current Agent Task: Classify Node {self.center_id}.\n"
            f"Center Node Info:\n"
            f"- Text: {self.center_text}\n"
            f"- In-Degree: {stats['in_degree']}, Out-Degree: {stats['out_degree']}\n\n"
            f"Candidate Categories: {candidates_str}\n\n"
            f"Legend: {self._format_legend(legend_dict)}\n"
            f"Visual Shapes: ◯ = 1-hop, ▼ = High Out-degree, ▲ = High In-degree"
        )

        infos = {"center_id": self.center_id, "answer": self.answer, "step": self.step_count, "mode": self.mode, "won": False}
        return obs, self.current_image, infos

    def step(self, raw_input: str):
        current_action = ""
        think_content, summary_content = None, None

        action_match = re.search(r"<action>(.*?)</action>", raw_input, re.DOTALL | re.IGNORECASE)
        if action_match:
            current_action = action_match.group(1).strip()
            s_match = re.search(r"<summary>(.*?)</summary>", raw_input, re.DOTALL | re.IGNORECASE)
            if s_match: summary_content = s_match.group(1).strip()
            t_match = re.search(r"<think>(.*?)</think>", raw_input, re.DOTALL | re.IGNORECASE)
            if t_match: think_content = t_match.group(1).strip()
        else:
            current_action = raw_input.strip()

        if not current_action or self.done:
            img_ret = self.current_image.copy() if self.current_image is not None else np.zeros((1024,1024,3), dtype=np.uint8)
            return ("Invalid format" if not self.done else ""), img_ret, 0, self.done, {
                "parsed_action": "ERROR", 
                "won": False, 
                "step": self.step_count
            }
        
        self.step_count += 1
        reward = 0
        done = False
        obs = ""
        
        if current_action.startswith("check_node:") or current_action.startswith("check_nodes:"):
            if self.mode == "System1":
                obs = "System 1 Violation: You are not allowed to explore in Clear Weather. Directly submit the answer."
            else:
                node_ids = [int(p) for p in re.findall(r"\d+", current_action)]
                texts = []
                for node_id in node_ids[:5]:
                    self.seen_nodes.add(node_id)
                    texts.append(f"Node {node_id} Text:\n{self.node_text_db.get(str(node_id), 'No text available.')[:400]}")
                obs = "\n\n".join(texts)

        elif current_action.startswith("paint:"):
            if self.mode == "System1":
                obs = "System 1 Violation: Paint is disabled in Clear Weather."
            else:
                parts = current_action.split(":", 1)[1].split(",", 1)
                nid, cls = int(parts[0].strip()), parts[1].strip()
                if nid not in self.painted_nodes:
                    self.paint_history[self.step_count] = (nid, cls)
                    self.painted_nodes[nid] = cls
                    obs = f"[DELAYED FEEDBACK] Node {nid} painted as '{cls}'. Map updated. Correctness will be evaluated at the end."
                else:
                    obs = f"[INVALID] Node {nid} is already painted. Prevent Farming constraint triggered. Action yields 0 reward."
                
                img_bytes, legend_dict, self.current_node_shapes = self.visualizer.draw_subgraph(
                    self.center_id, view_mode="2-hop+sim", max_nodes=30, 
                    color_seed=self.episode_color_seed, mask_neighbors=True, painted_nodes=self.painted_nodes
                )
                self.current_image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((1024, 1024)))
                obs += f"\nLegend: {self._format_legend(legend_dict)}"

        elif current_action.startswith("check_graph:"):
            if self.mode == "System1":
                obs = "System 1 Violation: Graph manipulation disabled. Submit directly."
            else:
                params = current_action.split(":", 1)[1].strip().split(",")
                v_mode, max_n = params[0].strip(), int(params[1].strip())
                img_bytes, legend_dict, self.current_node_shapes = self.visualizer.draw_subgraph(
                    self.center_id, view_mode=v_mode, max_nodes=max_n, 
                    color_seed=self.episode_color_seed, mask_neighbors=True, painted_nodes=self.painted_nodes
                )
                self.current_image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((1024, 1024)))
                obs = f"Graph view updated. Legend: {self._format_legend(legend_dict)}"

        elif current_action.startswith("final:") or current_action.startswith("submit:"):
            pred = current_action.split(":", 1)[1].strip()
            is_correct = (pred.lower().strip().strip(".'\"") == self.answer.lower().strip().strip(".'\""))
            
            if self.mode == "System1":
                done = True
                self.done = True
                reward = 1.0 if is_correct else -0.5
                obs = "System 1 Final answer submitted."
            else:
                # =========================================================
                # 【修改：开门条件放宽】只看涂色数量和种类是否齐全，不强校验对错
                # =========================================================
                total_paints = len(self.paint_history)
                painted_hub = 0
                painted_1hop = 0
                
                for p_nid, p_cls in self.paint_history.values():
                    # 判断种类齐全度
                    shape = self.current_node_shapes.get(p_nid, '1-hop')
                    if shape in ['high_out', 'high_in']:
                        painted_hub += 1
                    else:
                        painted_1hop += 1
                
                has_hub_req = (self.available_node_types['high_out'] > 0 or self.available_node_types['high_in'] > 0)
                has_1hop_req = (self.available_node_types['1-hop'] > 0)
                
                # 解锁逻辑：数量达标
                is_unlocked = (total_paints >= self.unlock_target)
                if has_hub_req and painted_hub == 0: is_unlocked = False
                if has_1hop_req and painted_1hop == 0: is_unlocked = False
                
                if is_unlocked:
                    done = True
                    self.done = True
                    if is_correct:
                        base_reward = 1.0 
                        obs = f"Gate UNLOCKED ({total_paints}/{self.unlock_target}). Final answer CORRECT!"
                    else:
                        base_reward = -0.5
                        obs = f"Gate UNLOCKED ({total_paints}/{self.unlock_target}). Final answer WRONG."
                    
                    # 幕后清算：分配历史动作的奖励
                    hindsight_rewards = [0.0] * self.step_count
                    rewarded_paints = 0
                    for step_idx, (p_nid, p_cls) in self.paint_history.items():
                        gt_cls = self.visualizer._get_node_info(p_nid)["true_class"]
                        if p_cls.lower() == gt_cls.lower():
                            if rewarded_paints < self.unlock_target:
                                hindsight_rewards[step_idx - 1] = 0.02 
                                rewarded_paints += 1
                            else:
                                hindsight_rewards[step_idx - 1] = 0.0  
                        else:
                            hindsight_rewards[step_idx - 1] = -0.01
                            
                    hindsight_rewards[-1] += base_reward
                    reward = base_reward
                    self.hindsight_rewards = hindsight_rewards
                else:
                    done = False
                    reward = -0.01
                    missing = []
                    if total_paints < self.unlock_target:
                        missing.append(f"{self.unlock_target - total_paints} more node(s) to paint")
                    if has_hub_req and painted_hub == 0:
                        missing.append("at least 1 High-degree (Triangle) node")
                    if has_1hop_req and painted_1hop == 0:
                        missing.append("at least 1 1-hop (Circle) node")
                        
                    obs = f"Action Failed: Logic Gate Locked. Graph Homophily is low. Based on the current available topology, you still need to fulfill: {', '.join(missing)}."
        else:
            obs = f"Invalid action command."

        if not done and self.step_count >= self.max_steps:
            done = True
            self.done = True
            if self.mode == "System2" and self.hindsight_rewards is None:
                hindsight_rewards = [0.0] * self.step_count
                rewarded_paints = 0
                for step_idx, (p_nid, p_cls) in self.paint_history.items():
                    gt_cls = self.visualizer._get_node_info(p_nid)["true_class"]
                    if p_cls.lower() == gt_cls.lower():
                        if rewarded_paints < self.unlock_target:
                            hindsight_rewards[step_idx - 1] = 0.02
                            rewarded_paints += 1
                        else:
                            hindsight_rewards[step_idx - 1] = 0.0
                    else:
                        hindsight_rewards[step_idx - 1] = -0.01
                hindsight_rewards[-1] += -0.5 
                self.hindsight_rewards = hindsight_rewards

        info = {
            "step": self.step_count,
            "won": bool(reward > 0) if done else False,
            "parsed_think": think_content, 
            "parsed_action": current_action
        }
        
        if self.done and self.mode == "System2":
            info["hindsight_rewards"] = self.hindsight_rewards

        return obs, self.current_image.copy() if self.current_image is not None else None, reward, done, info


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

    print(f"[build_envs] Pre-loading graph data and building feature matrix for {dataset_name}...")
    
    g_data, r_adj, c_map = GraphVisualizer.load_graph_data(dataset_name, dataset_dir)
    
    temp_viz = GraphVisualizer(
        dataset_name=dataset_name, 
        dataset_dir=dataset_dir,
        shared_data=(g_data, r_adj, c_map, None) 
    )
    feat_matrix = temp_viz.feat_matrix
    
    print(f"[build_envs] Matrix built successfully. Shape: {feat_matrix.shape}, "
          f"Size: {feat_matrix.nbytes / 1024**2:.2f} MB")
    
    shared_payload = (g_data, r_adj, c_map, feat_matrix)
    
    print(f"[build_envs] Shared payload ready. Graph Setting: {graph_setting}")

    envs = [
        GraphSearchEnv(
            max_steps=max_steps, 
            node_text_db=node_text_db, 
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            graph_setting=graph_setting, 
            shared_graph_data=shared_payload 
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