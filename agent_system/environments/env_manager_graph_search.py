import re
import os
import time
import json
from typing import Any, Dict, List, Tuple
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import FullSequenceSearchMemory

V_GRAPH_AGENT_INSTRUCTION = """
# V-GraphAgent: RL-Driven Multimodal Graph Exploration

你是一个多模态图智能体。你的目标是预测视觉图中中心节点（方形节点）的所属类别。

## 视觉与文本对照规则
1. **视觉宏观导航**：你将看到一张以中心节点为原点的局部图。节点上的颜色代表语义特征相似度（暖色/深红色代表高度相似，冷色/浅灰色代表异质或偏移），形状代表拓扑重要性（★ 为宏观聚类中心，▼/▲ 为高出入度 Hub，● 为普通节点，■ 为中心节点）。图上直接标有真实的节点数字 ID。
2. **零文本探索机制**：为了锻炼你的视觉与推理能力，初始状态**不提供**任何邻居节点的文本摘要。你只能通过观察图上的“形状”和“颜色”寻找高价值的证据节点。

## 探索与决策机制
- **彻底打破纯文本依赖**：如果仅靠中心节点的初始先验无法确信，你**必须**结合视觉拓扑调用工具批量查阅完整摘要。
- **注意非线性成本**：探索预算极度有限！单次动作最多允许查阅 5 个节点。累计查阅超出最大预算将强制判定失败；如果不查阅任何视图证据就盲猜且猜错，将受到 -2.0 的致命惩罚！

## 动作空间
- 批量查阅节点摘要：`<action>check_nodes([ID1, ID2, ...])</action>` （例如：<action>check_nodes([1024, 2048])</action>）
- 提交最终答案：`<action>final(Category)</action>` （Category必须严格来自可选类别列表）

## 强制思考格式 (Metacognitive CoT)
在采取动作前，必须按以下模板进行显式的自我审视：
<think>
[当前假设]：结合中心节点先验与局部视觉拓扑（如深红色节点的分布），最可能的类别是什么？
[置信度评估]：把握大吗？（高/中/低）
[批量动作决策]：如果不确定，根据视图上的颜色和重要形状，我该查阅哪几个真实 ID 的节点及其原因？如果确定，直接提交答案。
</think>
<action>...</action>
"""

class GraphSearchEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = FullSequenceSearchMemory()
        super().__init__(envs, projection_f, config)
        self._think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        self._action_pattern = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
        
        self.initial_modes = []

        self.log_dir = "./Check_Log"
        os.makedirs(self.log_dir, exist_ok=True)
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_run_log_file = os.path.join(self.log_dir, f"run_{run_timestamp}_pid{os.getpid()}.jsonl")
        
        self.episode_trajectories = {}
        self.conversations = [] # 维护原生的多轮对话

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        text_obs, image_obs, infos = self.envs.reset(kwargs=kwargs)
        self.initial_states = text_obs
        
        self.initial_modes = [info.get("mode", "System2") for info in infos]
        self.memory.reset(batch_size=len(text_obs))

        # =========================================================
        # 初始化原生多轮对话列表：System Role + User <image>
        # =========================================================
        self.conversations = []
        for i in range(len(text_obs)):
            mode = self.initial_modes[i]
            
            # 统一使用最新版的 V_GRAPH_AGENT_INSTRUCTION
            sys_inst = V_GRAPH_AGENT_INSTRUCTION

            # 将 Prompt 完美融合进 ChatML 原生格式
            conv = [
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": f"=== Initial State ===\n{text_obs[i]}\n\nCurrent Visual View: <image>"}
            ]
            self.conversations.append(conv)

        for i in range(len(text_obs)):
            self.episode_trajectories[i] = {
                "center_id": infos[i].get("center_id", "Unknown"),
                "answer": infos[i].get("answer", "Unknown"),
                "mode": self.initial_modes[i],
                "steps": [],
                "initial_obs": text_obs[i]
            }

        observations = {
            "text": [conv.copy() for conv in self.conversations], 
            "image": image_obs, 
            "anchor": text_obs.copy(), 
        }

        for i, prompt in enumerate(observations["text"]):
             if i in self.episode_trajectories:
                 self.episode_trajectories[i]["initial_prompt"] = str(prompt)

        return observations, infos

    def step(self, text_actions: List[str]):
        thinks = []
        
        for raw_text in text_actions:
            t_match = self._think_pattern.search(raw_text)
            if t_match:
                thinks.append(t_match.group(1).strip())
            else:
                # 如果没有匹配到 <think>，但遇到了 <action>，则截取 action 之前的内容作为 think
                pre_action = raw_text.split("<action>")[0].strip()
                thinks.append(pre_action if pre_action else None)

        actions, valids = self.projection_f(text_actions)
        next_text_obs, next_image_obs, rewards, dones, infos = self.envs.step(actions)

        # =========================================================
        # [核心修复] 多轮对话状态追加 (坚决不碰历史记录中的 <image>)
        # =========================================================
        for i, act_text in enumerate(text_actions):

            if i not in self.episode_trajectories:
                continue

            # 1. 拼接模型的动作
            self.conversations[i].append({"role": "assistant", "content": act_text})
            
            # 2. 拼接环境的纯文本新反馈 (加入 Key 防护机制)
            if i in self.episode_trajectories:
                step_num = len(self.episode_trajectories[i]['steps']) + 1
            else:
                step_num = "N/A (Finished)"
                
            feedback_content = f"=== Step {step_num} Environment Feedback ===\n{next_text_obs[i]}"
            self.conversations[i].append({"role": "user", "content": feedback_content})

        self.memory.store({
            "search": actions,
            "information": next_text_obs
        })

        next_observations = {
            "text": [conv.copy() for conv in self.conversations], 
            "image": next_image_obs, # 将新图传入，Processor 会自动去寻找第一轮的 <image> 占位符并替换
            "anchor": next_text_obs.copy(),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["parsed_think"] = thinks[i]
            a_match = self._action_pattern.search(text_actions[i])
            parsed_act_content = a_match.group(1).strip() if a_match else "No Action Found"
            info["parsed_action_content"] = parsed_act_content

            if i in self.episode_trajectories:
                step_record = {
                    "step_idx": len(self.episode_trajectories[i]["steps"]) + 1,
                    "model_raw_output": text_actions[i],
                    "parsed_think": thinks[i],
                    "parsed_action": parsed_act_content,
                    "env_executed_action": actions[i],
                    "is_valid_format": bool(valids[i]),
                    "env_feedback_obs": next_text_obs[i],
                    "step_reward": float(rewards[i]),
                    "done": bool(dones[i]),
                    "next_prompt": str(next_observations["text"][i]) if not dones[i] else "N/A"
                }
                self.episode_trajectories[i]["steps"].append(step_record)

                if dones[i]:
                    self.episode_trajectories[i]["final_reward"] = float(rewards[i])
                    self.episode_trajectories[i]["won"] = info.get("won", False)
                    self.episode_trajectories[i]["failure_reason"] = info.get("failure_reason", "Unknown")
                    self._save_trajectory_to_disk(i)

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def _save_trajectory_to_disk(self, env_idx: int):
        if env_idx not in self.episode_trajectories:
            return
            
        traj_data = self.episode_trajectories.pop(env_idx)
        
        center_id = traj_data.get("center_id", "Unknown")
        mode = traj_data.get("mode", "Unknown")
        won = traj_data.get("won", False)
        status = "WIN" if won else "LOSE"
        traj_data["episode_status"] = status
        
        try:
            with open(self.current_run_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(traj_data, ensure_ascii=False) + '\n')
                
            print(f"[Check_Log] Appended to log | Mode: {mode} | Center: {center_id} | Result: {status} | Steps: {len(traj_data['steps'])} | Reward: {traj_data.get('final_reward', 0.0)}")
            
        except Exception as e:
            print(f"[Check_Log] Error saving trajectory for env {env_idx}: {e}")