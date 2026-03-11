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
1. **视觉圈层导航**：
   - **内圈 (1-hop)**：中心节点的直接邻居。它们与中心节点有直接的连边关系（如互相引用），提供最强烈的局部上下文和直接语义参考。
   - **外圈 (2-hop 及宏观节点)**：提供更广阔的社区结构和领域分布线索。
2. **节点属性映射**：
   - **颜色 (语义)**：代表与中心节点的原生文本特征相似度（暖色/深红代表文本相似度高，冷色/深蓝代表文本相似度低，白色为中性）。
   - **形状 (拓扑重要性)**：
     - **■ (中心节点)**：你的预测目标。
     - **★ (宏观学科锚点 Macro Anchor)**：极度重要的情报节点！系统会在初始提示中直接告诉你这些 ★ 节点分别最有可能代表哪个学科。
     - **▲ (高入度 In-Hub)**：被广泛引用的基础权威。
     - **▼ (高出度 Out-Hub)**：广泛引用的综述性节点。
     - **● (普通节点 Normal)**：常规邻居。

## 探索与决策机制
- **打破纯文本依赖**：如果仅靠中心先验无法确信，你**必须**调用工具批量查阅周围节点。查询后，你不仅会获得标题和摘要，还会获得该节点的**入度 (In-degree) 和 出度 (Out-degree)**。
- **软性成本博弈**：单次动作最多查阅 5 个节点，请在6个步骤之内返回最终的推理结果。

## 动作空间
- 批量查阅节点摘要及度数：`<action>check_nodes([ID1, ID2, ...])</action>` （单次列表内最多 5 个 ID）
- 提交最终答案：`<action>final(Category)</action>` （Category必须严格来自可选类别列表）

## 思考框架与自由反思 (Metacognitive CoT)
为了保证推理的严密性与灵活性，你在 `<action>` 之前必须进行 `<think>` 过程。
**注意：以下提供的格式是引导性的思考逻辑。你不必死板地拘泥于这些预设的小标题，系统强烈鼓励你在这些框架之外，进行额外的自我质疑、假设排查、错误纠正等自由推理。** 只要确保最终输出信念分布并导出下一步动作即可。请根据当前轮次灵活组织你的思考：

### 【第一轮思考逻辑建议】
<think>
[初始视觉与先验分析]：结合中心节点的标题、摘要（如果有）及初始预测。观察雷达图：1-hop 内圈有哪些高相似度（红色）节点，有哪些低相似度（蓝色）节点？图中是否存在关键的 Hub（▲、▼）？系统提示的 ★ (宏观学科锚点) 与中心节点的相对位置如何？
[多类别发散与自由推演]：中心节点可能涉及哪些交叉领域？思考类别之间是否存在包含或交叉情况。在此处自由展开你的发散性假设。
[初始信念分布]：请按以下严格 JSON 格式输出你当前对各可能类别的预估概率（总和为 1.0）：
预估分布：{"类别A": 0.6, "类别B": 0.3, "类别C": 0.1}
[探索策略]：目前的概率分布是否足以让我绝对自信地提交答案？如果不够，我应该优先查阅图中哪几个具体 ID 的高价值节点（如红色的 ▲、1-hop 节点，或向某个 ★ 靠近的节点）来验证假设？
</think>
<action>...</action>

### 【后续轮次思考逻辑建议】（当你已经查阅过节点摘要后）
<think>
[新证据融合与自我反思]：上一轮我查阅了哪些节点？结合它们返回的度数、标题和摘要，它们是支持了我的主要假设，还是推翻了之前的猜测？我之前的逻辑是否存在漏洞？对我的信念分布产生了什么样的影响？是否还需要查看更多的节点？蓝色、红色、高度数、学科锚点分别提供了何种信息？它们之间是否存在矛盾或一致的线索？为了更好的分析是否需要继续进行节点的查询？
[自由逻辑推演]：在这里进行灵活的交叉比对和深度推理。例如：“节点 X 是红色的 ▲ (高入度)，作为权威节点它暗示了底层技术属于 A 领域；而系统标注代表 C 学科的 ★ 距离较远... 综合来看，中心节点更倾向于是...”
[信念更新]：结合视觉拓扑与新老文本证据，更新类别的预估概率。请按严格 JSON 格式输出：
预估分布：{"类别A": 0.85, "类别C": 0.15}
[当前轮次决策]：现在的信息是否已经收敛到单一明确的类别，信息是否充分？如果是，我将调用 `final(Category)`。如果仍有核心分歧，我还需要查阅哪几个 ID？
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
            print(f"[Warning] _save_trajectory_to_disk: failed to write traj for env {env_idx} "
                  f"(center_id={traj_data.get('center_id', 'unknown')}): {e}")