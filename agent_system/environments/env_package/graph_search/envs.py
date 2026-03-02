import json
import random
import io
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from .graph_visualizer import GraphVisualizer
import re

class GraphSearchEnv:
    def __init__(self, max_steps: int, node_text_db: Dict[str, str], dataset_name: str, dataset_dir: str, shared_graph_data: Optional[Any] = None): 
        self.max_steps = max_steps
        self.node_text_db = node_text_db
        self.visualizer = GraphVisualizer(dataset_name=dataset_name, dataset_dir=dataset_dir, shared_data=shared_graph_data)
        # [移除] max_budget 变量，不设置全局强行截断
        self._reset_internal()

    def _reset_internal(self):
        self.step_count = 0
        self.check_node_count = 0 # 全局累计查阅次数统计
        self.done = False
        self.current_image = None
        self.valid_nodes_in_view = set()
        
        self.is_won = False
        self.failure_reason = "Timeout"
        
    def _get_title_and_abstract(self, raw_text: str) -> Tuple[str, str]:
        parts = raw_text.split("Abstract:", 1)
        title_part = parts[0].replace("Title:", "").strip()
        abstract_part = parts[1].strip() if len(parts) > 1 else "No abstract."
        return title_part, abstract_part

    def reset(self, kwargs: Dict[str, Any]) -> str:
        self._reset_internal()
        self.center_id = kwargs["center_id"]
        self.answer = kwargs["answer"]

        # 生成零文本同心圆雷达图
        img_bytes, node_catalog_info, all_classes = self.visualizer.draw_vgraph_radar_layout(int(self.center_id))
        self.current_image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS))
        
        for real_id in node_catalog_info.keys():
            self.valid_nodes_in_view.add(int(real_id))
            
        center_raw_text = self.node_text_db.get(str(self.center_id), "Title: Unknown\nAbstract: None")
        center_title, _ = self._get_title_and_abstract(center_raw_text)
        
        proxy_data = self.visualizer._get_node_info(self.center_id).get("proxy_info", {})
        center_proxy = proxy_data.get("top1", "Unknown")

        candidates_str = ", ".join(all_classes)
        
        obs = (
            f"🎯 任务：请预测目标中心节点 **{self.center_id}** 的准确类别。\n\n"
            f"【中心节点基础先验】\n"
            f"* Title: {center_title}\n"
            f"* Initial Proxy Prediction (60% confidence): {center_proxy}\n\n"
            f"【操作指南】\n"
            f"图上展示了其局部结构的同心圆雷达图（内圈1-hop，外圈Hubs），颜色代表真实文本特征相似度，形状代表拓扑度数。\n"
            f"你可以自由结合视觉发现，调用查阅动作：`<action>check_nodes([ID1, ID2, ...])</action>`。\n"
            f"注意：单次动作最多允许带 5 个 ID。总查阅次数没有硬性限制，但前两免费，从查阅第 3 个节点起每个节点都会产生递增的微小惩罚。如果不查阅盲猜失败将扣 -2.0 分。\n\n"
            f"可选类别: [{candidates_str}]\n"
        )

        infos = {"center_id": self.center_id, "answer": self.answer, "step": self.step_count, "mode": "V-GraphAgent"}
        return obs, self.current_image, infos

    def step(self, raw_input: str):
        self.step_count += 1
        reward = 0.0
        obs = ""
        current_action = raw_input.strip()
        img_ret = self.current_image.copy() if self.current_image is not None else np.zeros((1024,1024,3), dtype=np.uint8)

        if self.done:
             return "Episode already finished.", img_ret, 0.0, True, {
                 "parsed_action": "ERROR", 
                 "won": self.is_won, 
                 "failure_reason": self.failure_reason
             }
             
        if not current_action:
            return "[System] Invalid format. Wrap action in <action>...</action>.", img_ret, -0.1, False, {
                "parsed_action": "ERROR", 
                "won": False, 
                "failure_reason": "Format_Error"
            }

        check_match = re.search(r"check_nodes?\(\[?([\d,\s]+)\]?\)", current_action, re.IGNORECASE)
        final_match = re.search(r"(?:final|submit)\((.+?)\)", current_action, re.IGNORECASE)

        if check_match:
            ids_str = check_match.group(1)
            target_ids = [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]
            
            # [恢复限制]：单次请求 ID 个数不得超过 5 个，防止上下文一次性撑爆
            if len(target_ids) > 5:
                target_ids = target_ids[:5] 

            obs_lines = []
            for tid in target_ids:
                if tid in self.valid_nodes_in_view:
                    raw_text = self.node_text_db.get(str(tid), "Title: Unknown\nAbstract: not found.")
                    title, abstract = self._get_title_and_abstract(raw_text)
                    obs_lines.append(f"--- Node {tid} ---\nTitle: {title}\nAbstract: {abstract}")
                    
                    # 动态探索阶梯成本：全局查阅超过 2 个后，开始持续累加软性惩罚 (-0.05)
                    self.check_node_count += 1
                    if self.check_node_count > 2:
                        reward += -0.05 
                else:
                    obs_lines.append(f"错误：节点 {tid} 不在当前雷达视野内。")
                    reward += -0.1
            
            obs = "\n\n".join(obs_lines)

        elif final_match:
            pred = final_match.group(1).strip()
            is_correct = (pred.lower() == self.answer.lower())
            self.done = True
            
            if is_correct:
                reward = 1.0
                obs = "预测正确！"
            else:
                if self.check_node_count == 0:
                    reward = -2.0  # 致命的盲目自信惩罚
                    obs = "预测错误！致命惩罚：未查阅任何视图证据即盲听先验。"
                else:
                    reward = -1.0
                    obs = "预测错误。"
        else:
            obs = "无效动作格式，请检查拼写。"
            reward = -0.1

        failure_reason = "Success"
        if self.done and reward < 1.0:
            if final_match:
                failure_reason = "Wrong_Answer_Blind" if self.check_node_count == 0 else "Wrong_Answer"
            else:
                failure_reason = "Timeout"
        elif not self.done and self.step_count >= self.max_steps:
            # 环境通过回合最大步数自然结束，而非预算截断
            self.done = True; reward = -1.0; failure_reason = "Timeout"

        if self.done:
            self.is_won = (reward == 1.0)
            self.failure_reason = failure_reason

        info = {
            "step": self.step_count, 
            "won": (reward == 1.0), 
            "parsed_action": current_action, 
            "failure_reason": failure_reason
        }
        return obs, img_ret, reward, self.done, info

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