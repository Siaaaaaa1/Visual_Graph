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
1. **视觉圈层导航**：内圈 (1-hop) 为直接邻居，外圈为宏观拓扑背景。
2. **节点属性映射**：
   - **颜色 (语义)**：代表与中心节点的特征相似度（暖色/红高度相似，冷色/蓝异质）。注意：这是动态对比度，最红的节点即当前视野中最相似的。
   - **形状 (拓扑重要性)**：
     - **■ (中心节点)**：你的预测目标。
     - **★ (宏观学科锚点 Macro Anchor)**：极度重要的情报节点！系统会在初始提示中直接告诉你这些 ★ 节点分别代表哪个学科（可能是你的先验假设，也可能是易混淆的姐妹学科，甚至可能是**用来骗你的“假红”跨界节点**）。
     - **▲ (高入度 In-Hub)**：被广泛引用的基础权威。
     - **▼ (高出度 Out-Hub)**：广泛引用的综述性节点。
     - **● (普通节点 Normal)**：常规邻居。

## 探索与决策机制
- **破除颜色与先验迷信**：系统为你提供的 ★ 锚点中，可能存在颜色极红但实际类别完全无关的“视觉陷阱”。你**必须**调用工具 `check_nodes` 查阅这些关键节点和内圈邻居的摘要（含出入度），用逻辑证伪！
- **软性成本博弈**：单次最多查阅 5 个。前 2 批次免费，第 3 批次起产生微小惩罚。盲猜失败扣 -2.0 分。

## 动作空间
- 查阅节点摘要及度数：`<action>check_nodes([ID1, ID2, ...])</action>`
- 提交最终答案：`<action>final(Category)</action>` 

## 强制思考格式 (Metacognitive CoT)
在 `<action>` 前必须进行 `<think>`：

### 【第一轮思考逻辑】（未查阅节点时）
<think>
[初始视觉与靶点锁定]：中心节点先验是什么？图上那些已知身份的 ★ 锚点，哪个呈现出高度相似的红色？哪个呈现冷色？有没有内圈的 ▲ 或 1-hop 节点颜色特别异常？
[多类别发散]：这会不会是一个交叉学科？我最容易把它和哪个类别搞混？
[初始信念分布]：按严格 JSON 格式输出预估概率。
预估分布：{"类别A": 0.6, "类别B": 0.3, "类别C": 0.1}
[探索策略]：我应该优先查阅图上哪几个具体的 ID？（提示：优先核对那些颜色最红的 ★ 锚点，以及红色的内圈邻居，去粉碎或验证假设）。
</think>

### 【后续轮次思考逻辑】（已查阅节点后）
<think>
[新证据融合]：查阅返回的入度/出度和摘要说了什么？那个红色的 ★ 锚点真的是同类，还是只是某些方法学上相似的跨界坑？
[自由逻辑推演]：灵活的交叉比对推演。
[信念更新]：按严格 JSON 格式更新概率。
预估分布：{"类别A": 0.85, "类别C": 0.15}
[当前轮次决策]：是否已收敛至单一明确类别？是则 final，否则继续 check_nodes。
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
        self.conversations = [] 

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        text_obs, image_obs, infos = self.envs.reset(kwargs=kwargs)
        self.initial_states = text_obs
        
        self.initial_modes = [info.get("mode", "System2") for info in infos]
        self.memory.reset(batch_size=len(text_obs))

        self.conversations = []
        for i in range(len(text_obs)):
            mode = self.initial_modes[i]
            sys_inst = V_GRAPH_AGENT_INSTRUCTION

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
                pre_action = raw_text.split("<action>")[0].strip()
                thinks.append(pre_action if pre_action else None)

        actions, valids = self.projection_f(text_actions)
        next_text_obs, next_image_obs, rewards, dones, infos = self.envs.step(actions)

        for i, act_text in enumerate(text_actions):
            if i not in self.episode_trajectories:
                continue
            self.conversations[i].append({"role": "assistant", "content": act_text})
            
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
            "image": next_image_obs, 
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
        except Exception as e:
            pass