import json
import random
import io
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from .graph_visualizer import GraphVisualizer
import re
from collections import Counter

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
                 tau: float = 0.4,
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
        self.painted_groups = {}
        self.mode = "System1"
        self.required_classes = set()
        self.anon_map = {}
        self.color_mapping = {}

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
        
        neighbors_1hop = set(self.visualizer._get_neighbors(self.center_id, undirected=True))
        all_neighbors_1_2_hop = set(neighbors_1hop)
        for nb in neighbors_1hop:
            all_neighbors_1_2_hop.update(self.visualizer._get_neighbors(nb, undirected=True))
            
        if self.center_id in all_neighbors_1_2_hop:
            all_neighbors_1_2_hop.remove(self.center_id)
        
        # 使用 Margin of Victory (断层优势) 计算同质性
        neighbor_classes = []
        for nb in all_neighbors_1_2_hop:
            tc = self.visualizer._get_node_info(nb)["true_class"]
            neighbor_classes.append(tc)
            
        if len(neighbor_classes) > 0:
            class_counts = Counter(neighbor_classes)
            top_classes = class_counts.most_common(2)
            
            top1_ratio = top_classes[0][1] / len(neighbor_classes)
            top2_ratio = top_classes[1][1] / len(neighbor_classes) if len(top_classes) > 1 else 0.0
            
            margin = top1_ratio - top2_ratio
        else:
            margin = 0.0 # 孤立节点直接进 System 2

        if margin >= self.tau:
            if is_train and random.random() < self.epsilon:
                self.mode = "System2"
            else:
                self.mode = "System1"
        else:
            self.mode = "System2"

        init_view_mode = "2-hop+sim"
        init_max_nodes = 30

        # 获取画面节点
        view_nodes = self.visualizer._select_nodes(self.center_id, init_view_mode, init_max_nodes, undirected=True)
        
        # 解析相对高度数的 Top-3
        node_degrees = {}
        for nid in view_nodes:
            if nid == self.center_id: continue
            deg = self.visualizer.get_node_degree_info(nid)
            node_degrees[nid] = deg['out_degree'] + deg['in_degree']
        sorted_by_deg = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        self.top3_hubs = set([nid for nid, _ in sorted_by_deg[:3]])

        # 统计可视范围内的真实类别分布
        valid_view_nodes = [n for n in view_nodes if n != self.center_id]
        class_counts_view = {}
        for nid in valid_view_nodes:
            tc = self.visualizer._get_node_info(nid)["true_class"]
            class_counts_view[tc] = class_counts_view.get(tc, 0) + 1
            
        prioritized_classes = sorted(class_counts_view.keys(), key=lambda tc: class_counts_view[tc], reverse=True)
        
        all_classes = self.visualizer.get_all_candidate_classes()
        all_classes_set = set(all_classes)
        for tc in prioritized_classes:
            all_classes_set.add(tc)
        all_classes = sorted(list(all_classes_set))

        self.color_mapping = self.visualizer._get_color_map_for_episode(
            all_classes, 
            self.episode_color_seed,
            prioritized_classes=prioritized_classes
        )
        
        self.anon_map = {}
        for i, cls in enumerate(prioritized_classes):
            self.anon_map[cls] = f"Group {i+1}"
            
        curr_group_id = len(prioritized_classes) + 1
        for cls in all_classes:
            if cls not in self.anon_map:
                self.anon_map[cls] = f"Group {curr_group_id}"
                curr_group_id += 1

        gate_instruction = ""
        if self.mode == "System1":
            mask_neighbors_init = False
            self.center_text = self.node_text_db.get(str(self.center_id), "No text available.")
        else:
            mask_neighbors_init = True
            self.center_text = "[Text Hidden by Fog of War. Colors group similar nodes, but legend labels are anonymized. Explore to deduce meaning.]"
            
            threshold = len(valid_view_nodes) * 0.1
            top4_qualified = []
            for tc in prioritized_classes:
                if class_counts_view[tc] >= threshold:
                    top4_qualified.append(tc)
                    if len(top4_qualified) == 4:
                        break
                        
            self.required_classes = set(top4_qualified)
            
            for tc in self.required_classes:
                if tc not in self.anon_map:
                    self.anon_map[tc] = f"Group {len(self.anon_map) + 1}"
                    
            req_groups = sorted([self.anon_map[tc] for tc in self.required_classes])
            
            if req_groups:
                needed_count = max(1, len(self.required_classes) // 2 + (len(self.required_classes) % 2))
                gate_instruction = f"🚨 GATE REQUIREMENT: You must correctly paint AT LEAST {needed_count} of these Major Clusters: [{', '.join(req_groups)}]. (Groups <10% or beyond top 4 are ignored)."
            else:
                gate_instruction = f"🚨 GATE REQUIREMENT: Highly fragmented graph. Paint ANY 1 correct Group to unlock the gate."

        stats = self.visualizer.get_node_degree_info(self.center_id)
        
        visible_classes_set = set(class_counts_view.keys())
        center_tc = self.visualizer._get_node_info(self.center_id)["true_class"]
        visible_classes_set.add(center_tc)
        
        visible_classes = sorted(list(visible_classes_set))
        candidates_str = ", ".join(visible_classes) if visible_classes else "Unknown"
        
        img_bytes, legend_dict = self.visualizer.draw_subgraph(
            self.center_id, 
            view_mode=init_view_mode,
            max_nodes=init_max_nodes,
            color_seed=self.episode_color_seed,
            mask_neighbors=mask_neighbors_init,
            painted_nodes={},
            color_mapping=self.color_mapping,
            anon_map=self.anon_map
        )
        
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        self.current_image = np.array(pil_img)
        
        mode_desc = "CLEAR WEATHER (System 1)" if self.mode == "System1" else "FOG OF WAR: ANONYMOUS COLORS (System 2)"

        neighbor_text_block = ""
        if self.mode == "System1":
            class_to_nodes = {}
            for nid in view_nodes:
                if nid == self.center_id: continue
                tc = self.visualizer._get_node_info(nid)["true_class"]
                if tc not in class_to_nodes:
                    class_to_nodes[tc] = []
                class_to_nodes[tc].append(nid)

            selected_nbs = []
            selected_nids_set = set()
            
            for idx, tc in enumerate(prioritized_classes):
                if tc not in class_to_nodes:
                    continue
                nids = class_to_nodes[tc]
                
                if idx == 0:
                    max_quota = 3     
                elif idx in [1, 2]:
                    max_quota = 2     
                else:
                    max_quota = 1     
                
                sorted_nids = sorted(nids, key=lambda x: (x not in neighbors_1hop, -node_degrees.get(x, 0)))
                
                rep_count = min(max_quota, len(sorted_nids))
                for i in range(rep_count):
                    nid = sorted_nids[i]
                    hop_str = "1-hop" if nid in neighbors_1hop else "2-hop/Sim"
                    label = f"Representative for {tc} ({hop_str})"
                    selected_nbs.append((nid, label))
                    selected_nids_set.add(nid)
                    
            for nid in self.top3_hubs:
                if nid not in selected_nids_set and nid != self.center_id:
                    hop_str = "1-hop" if nid in neighbors_1hop else "2-hop/Sim"
                    label = f"Global High-Degree Hub ({hop_str})"
                    selected_nbs.append((nid, label))
                    selected_nids_set.add(nid)
                
            if selected_nbs:
                neighbor_text_block = "--- Supplementary Neighbors Context (Fast Thinking View) ---\n"
                for nid, n_type in selected_nbs:
                    txt = self.node_text_db.get(str(nid), "No text available.")[:300].replace('\n', ' ')
                    neighbor_text_block += f"- [{n_type} | Node {nid}]: {txt}...\n"
                neighbor_text_block += "------------------------------------------------------------\n\n"

        obs = (
            f"=== Environment Mode: {mode_desc} ===\n"
            f"Current Agent Task: Classify the Target Center Node {self.center_id}.\n\n"
            f"🎯 TARGET CENTER NODE {self.center_id} INFO:\n"
            f"============================================================\n"
            f"{self.center_text}\n"
            f"============================================================\n"
            f"Topology -> In-Degree: {stats['in_degree']}, Out-Degree: {stats['out_degree']}\n\n"
            f"{neighbor_text_block}"
            f"{gate_instruction}\n\n"
            f"Candidate Categories: {candidates_str}\n\n"
            f"Legend: {self._format_legend(legend_dict)}\n"
            f"Visual Shapes: ◯ = 1-hop/Other, ▼ = High Out-degree (Top 3), ▲ = High In-degree (Top 3)"
        )

        infos = {"center_id": self.center_id, "answer": self.answer, "step": self.step_count, "mode": self.mode, "won": False}
        return obs, self.current_image, infos

    def step(self, raw_input: str):
        self.step_count += 1
        reward = 0.0
        done = False
        obs = ""
        current_action = raw_input.strip()

        if not current_action or self.done:
            img_ret = self.current_image.copy() if self.current_image is not None else np.zeros((1024,1024,3), dtype=np.uint8)
            return ("Invalid format" if not self.done else ""), img_ret, -0.1 if not self.done else 0.0, self.done, {
                "parsed_action": "ERROR", 
                "won": False, 
                "step": self.step_count
            }

        if current_action.startswith("check_node:") or current_action.startswith("check_nodes:"):
            if self.mode == "System1":
                obs = "System 1 Violation: You are not allowed to explore in Clear Weather. Directly submit the answer."
                reward = -0.1 
            else:
                try:
                    node_ids = [int(p) for p in re.findall(r"\d+", current_action)]
                    texts = []
                    for node_id in node_ids[:5]:
                        self.seen_nodes.add(node_id)
                        texts.append(f"Node {node_id} Text:\n{self.node_text_db.get(str(node_id), 'No text available.')[:400]}")
                    obs = "\n\n".join(texts)
                    reward = 0.0 
                except Exception:
                    obs = "Error parsing check_node ids."
                    reward = -0.1

        elif current_action.startswith("paint:"):
            if self.mode == "System1":
                obs = "System 1 Violation: Paint is disabled in Clear Weather."
                reward = -0.1
            else:
                try:
                    parts = current_action.split(":", 1)[1].split(",", 1)
                    target_group = parts[0].strip()
                    cls = parts[1].strip()
                    
                    valid_group = None
                    gt_cls = None
                    for tc, anon in self.anon_map.items():
                        if anon.lower() == target_group.lower():
                            valid_group = anon
                            gt_cls = tc
                            break
                            
                    if valid_group:
                        norm_pred = cls.lower().strip().strip(".'\"")
                        norm_gt = gt_cls.lower().strip().strip(".'\"")
                        
                        # 获取该组当前的涂色状态
                        current_paint = self.painted_groups.get(valid_group)
                        current_norm = current_paint.lower().strip().strip(".'\"") if current_paint else None
                        
                        # 【核心优化】：如果本次涂色和当前已有的涂色完全一致，则拦截防刷分
                        if current_norm == norm_pred:
                            obs = f"[INVALID] {valid_group} is already painted as '{cls}'. Prevent Farming constraint triggered."
                            reward = -0.1 
                        else:
                            # 允许涂色（首次涂色 或 纠错覆盖）
                            action_word = "repainted" if current_paint else "painted"
                            self.paint_history[self.step_count] = (valid_group, cls)
                            self.painted_groups[valid_group] = cls # 更新/覆盖最新状态
                            obs = f"[DELAYED FEEDBACK] {valid_group} {action_word} as '{cls}'. Map updated. Correctness hidden."
                            
                            if norm_pred == norm_gt:
                                reward = 0.1
                            else:
                                reward = -0.1
                    else:
                        obs = f"Invalid group name '{target_group}'. Please use names like 'Group 1'."
                        reward = -0.1

                    legend_dict = {}
                    for tc, c_conf in self.color_mapping.items():
                        color_name = c_conf["name"]
                        if self.mode == "System2":
                            anon_name = self.anon_map.get(tc, "Unknown Group")
                            if anon_name in self.painted_groups:
                                pred_class = self.painted_groups[anon_name]
                                legend_dict[color_name] = f"Painted: {pred_class}"
                            else:
                                legend_dict[color_name] = f"Anonymous: {anon_name}"
                        else:
                            legend_dict[color_name] = tc
                    
                    obs += f"\nLegend: {self._format_legend(legend_dict)}"

                except Exception:
                    obs = "Invalid paint format. Use paint:Group Name,Category"
                    reward = -0.1

        elif current_action.startswith("check_graph:"):
            if self.mode == "System1":
                obs = "System 1 Violation: Graph manipulation disabled. Submit directly."
                reward = -0.1
            else:
                try:
                    params = current_action.split(":", 1)[1].strip().split(",")
                    v_mode, max_n = params[0].strip(), int(params[1].strip())
                    img_bytes, legend_dict = self.visualizer.draw_subgraph(
                        self.center_id, view_mode=v_mode, max_nodes=max_n, 
                        color_seed=self.episode_color_seed, mask_neighbors=True, 
                        painted_nodes={}, 
                        color_mapping=self.color_mapping, anon_map=self.anon_map
                    )
                    self.current_image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((1024, 1024)))
                    obs = f"Graph view updated. Legend: {self._format_legend(legend_dict)}"
                    reward = 0.0
                except Exception:
                    obs = "Invalid check_graph format."
                    reward = -0.1
                    
        elif current_action.startswith("final:") or current_action.startswith("submit:"):
            try:
                pred = current_action.split(":", 1)[1].strip()
                is_correct = (pred.lower().strip().strip(".'\"") == self.answer.lower().strip().strip(".'\""))
                
                if self.mode == "System1":
                    done = True
                    self.done = True
                    reward = 1.0 if is_correct else -1.0
                    obs = "System 1 Final answer submitted."
                else:
                    painted_tcs = set()
                    for grp, pred_class in self.painted_groups.items():
                        for tc, anon in self.anon_map.items():
                            if anon.lower() == grp.lower():
                                norm_pred = pred_class.lower().strip().strip(".'\"")
                                norm_tc = tc.lower().strip().strip(".'\"")
                                if norm_pred == norm_tc:
                                    painted_tcs.add(tc)
                                break
                    
                    if len(self.required_classes) == 0:
                        is_unlocked = len(painted_tcs) >= 1
                    else:
                        required_count = len(self.required_classes)
                        needed_count = max(1, required_count // 2 + (required_count % 2)) 
                        correct_required = len(painted_tcs.intersection(self.required_classes))
                        
                        is_unlocked = correct_required >= needed_count
                    
                    if is_unlocked:
                        done = True
                        self.done = True
                        if is_correct:
                            reward = 1.0 
                            obs = f"Gate UNLOCKED. Final answer CORRECT!"
                        else:
                            reward = -1.0
                            obs = f"Gate UNLOCKED. Final answer WRONG."
                    else:
                        done = False
                        reward = -0.1 
                        missing_classes = self.required_classes - painted_tcs
                        if len(self.required_classes) == 0:
                            missing_str = "ANY 1 correct Group"
                            obs = f"Action Failed: Logic Gate Locked. You still need to paint: [{missing_str}]."
                        else:
                            missing_groups = sorted([self.anon_map[tc] for tc in missing_classes])
                            missing_str = ", ".join(missing_groups)
                            needed_more = needed_count - correct_required
                            obs = f"Action Failed: Logic Gate Locked. You need to correctly paint at least {needed_more} more from the remaining required groups: [{missing_str}]."
            except Exception:
                obs = "Invalid submit format."
                reward = -0.1
        else:
            obs = f"Invalid action command."
            reward = -0.1

        # ---------------------------------------------------------
        # [新增逻辑]：判定失败原因 (Failure Reason Tracking)
        # ---------------------------------------------------------
        if not done and self.step_count >= self.max_steps:
            done = True
            self.done = True
            reward = -1.0
            
        failure_reason = "Success"
        if done and reward < 1.0: # 如果是失败的结局
            is_final_action = current_action.startswith("final:") or current_action.startswith("submit:")
            
            if self.mode == "System1":
                if is_final_action:
                    failure_reason = "Sys1_Wrong_Answer"
                else:
                    failure_reason = "Sys1_Timeout"
            else: # System 2
                # 判断当前门是否处于解锁状态 (复用你之前的软匹配逻辑判定)
                painted_tcs = set()
                for grp, pred_class in self.painted_groups.items():
                    for tc, anon in self.anon_map.items():
                        if anon.lower() == grp.lower():
                            if pred_class.lower().strip().strip(".'\"") == tc.lower().strip().strip(".'\""):
                                painted_tcs.add(tc)
                            break
                is_unlocked = False
                if len(self.required_classes) == 0:
                    is_unlocked = len(painted_tcs) >= 1
                else:
                    needed_count = max(1, len(self.required_classes) // 2 + (len(self.required_classes) % 2))
                    correct_required = len(painted_tcs.intersection(self.required_classes))
                    is_unlocked = correct_required >= needed_count

                if is_final_action:
                    if not is_unlocked:
                        failure_reason = "Sys2_Premature_Submit" # 没解锁就抢答
                    else:
                        failure_reason = "Sys2_Wrong_Answer"     # 解锁了但答错
                else:
                    if is_unlocked:
                        failure_reason = "Sys2_Timeout_Unlocked" # 解锁了但没步数提交了
                    else:
                        failure_reason = "Sys2_Timeout_Locked"   # 超时且没解锁
        # ---------------------------------------------------------

        info = {
            "step": self.step_count,
            "won": bool(reward == 1.0) if done else False,
            "parsed_action": current_action,
            "failure_reason": failure_reason  # 将失败原因传给上层
        }

        return obs, self.current_image.copy() if self.current_image is not None else None, reward, done, info
    
def build_graph_search_envs(seed: int, env_num: int, group_n: int, is_train: bool, env_config):
    batch_size = env_num * group_n
    max_steps = env_config.max_steps
    dataset_name = getattr(env_config, "dataset_name", "cora")
    dataset_dir = getattr(env_config, "dataset_dir", "./datasets")
    
    with open(env_config.node_text_path, "r", encoding="utf-8") as f:
        node_text_db = json.load(f)

    g_data, r_adj, c_map = GraphVisualizer.load_graph_data(dataset_name, dataset_dir)
    temp_viz = GraphVisualizer(dataset_name=dataset_name, dataset_dir=dataset_dir, shared_data=(g_data, r_adj, c_map, None))
    shared_payload = (g_data, r_adj, c_map, temp_viz.feat_matrix)

    envs = [
        GraphSearchEnv(max_steps=max_steps, node_text_db=node_text_db, dataset_name=dataset_name, 
                       dataset_dir=dataset_dir, shared_graph_data=shared_payload)
        for _ in range(batch_size)
    ]

    class BatchGraphSearchEnv:
        def __init__(self): 
            self.num_envs = batch_size
            self.executor = ThreadPoolExecutor(max_workers=min(batch_size, 32))

        def reset(self, kwargs):
            def _reset_env(env, kw):
                kw["is_train"] = is_train
                return env.reset(kw)
                
            results = list(self.executor.map(_reset_env, envs, kwargs))
            text_obs, image_obs, infos = zip(*results)
            return list(text_obs), list(image_obs), list(infos)

        def step(self, actions: List[str]):
            def _step_env(env, act):
                return env.step(act)
                
            results = list(self.executor.map(_step_env, envs, actions))
            text_obs, image_obs, rewards, dones, infos = zip(*results)
            return list(text_obs), list(image_obs), list(rewards), list(dones), list(infos)

        def close(self): 
            self.executor.shutdown(wait=True)

    return BatchGraphSearchEnv()