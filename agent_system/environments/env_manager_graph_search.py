import re
import os
import time
import json
from typing import Any, Dict, List, Tuple

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import FullSequenceSearchMemory

# =================================================================
# 1. System 1 (Fast Thinking) Prompts
# =================================================================

SYSTEM1_TASK_INSTRUCTION = """
# SYSTEM 1: FAST THINKING (CLEAR WEATHER MODE)

You are a graph reasoning agent. The environment has analyzed the current Ego-graph and determined it has extremely high homophily (the surrounding neighborhood is highly consistent and simple). 
Therefore, you have been routed to **System 1 Mode (Clear Weather)**.

## 1. Initial State & Context
* **Full Visibility:** There is NO Fog of War. You can visually see the predicted categories (colors) of all neighbor nodes in the graph legend, and you can directly read the center node's text in your observation.
* **Objective:** Directly predict the correct semantic category of the Center Node based on the obvious visual homophily and the center node's text.

## 2. Constraints & Rules
* **NO EXPLORATION ALLOWED:** To save test-time computation and prevent hallucination, you are strictly forbidden from using exploration actions (`check_node`, `check_graph`, `paint`).
* **Direct Answer Only:** You must synthesize the text and visual graph immediately in one step.

## 3. Interaction Actions
At each step, you must choose **exactly one** action:
* **Submit Answer:** `<action>final:Category Name</action>`

## 4. Response Format
You must strictly follow this cognitive sequence:
1. **<think>...** Briefly analyze the center text's keywords and the dominant category of the neighborhood.
2. **<summary>...** State your final conclusion based on the clear, unambiguous evidence.
3. **<action>...** Output your final action.
"""

SYSTEM1_FEW_SHOT = """
---
## 5. Demonstration Example (System 1)

**Context:** Center Node 38491. The text states "Attention is all you need... We propose the Transformer...". The visual graph legend shows 90% of neighbors are Blue (Natural Language Processing).

**Step 1:**
<think>
I am in System 1 (Clear Weather). I have full visibility.
1. Self-evidence: The center node text is clearly about the Transformer architecture, a foundational NLP model.
2. Contextual-evidence: The visual graph confirms this. The vast majority of connected neighbors are labeled "Natural Language Processing".
There is no ambiguity. I do not need to explore or paint. I can confidently submit the answer right now to save compute.
</think>
<summary>Textual evidence from the center node and overwhelming topological homophily from the neighbors unanimously confirm the category is Natural Language Processing.</summary>
<action>final:Natural Language Processing</action>
"""

# =================================================================
# 2. System 2 (Slow Thinking) Prompts
# =================================================================

SYSTEM2_TASK_INSTRUCTION = """
# SYSTEM 2: SLOW THINKING (FOG OF WAR MODE)

You are a graph reasoning agent. The environment has determined this graph is deceptive, heterogeneous, or sits on a boundary. 
Therefore, you have been routed to **System 2 Mode (Fog of War)**.

## 1. Fog of War Mechanics
* **Masked State:** The center node's text is hidden. The categories (colors) of all neighbor nodes are masked (Black). 
* **Delayed Feedback POMDP:** You must actively gather evidence step-by-step. The environment will NOT tell you if you are right or wrong until the very end.

## 2. Visual Topology & Shape Semantics (CRITICAL)
Since colors are hidden, you MUST rely on topological shapes to prioritize your exploration:
* **1-Hop Nodes (Circles ◯):** Your immediate neighbors. Represent direct local context.
* **High Out-Degree Nodes (Downward Triangles ▼):** Hub nodes pointing to many others. Meaning: Foundational methods or core algorithms. 
* **High In-Degree Nodes (Upward Triangles ▲):** Hub nodes heavily cited by others. Meaning: Popular applications or datasets.

## 3. The Homophily-Driven Logic Lock
You CANNOT guess the final answer. You must unlock the physical gate by successfully `paint`ing nodes.
* **Requirement:** You must paint a mix of different nodes (max 5 nodes needed). If you submit prematurely, the environment will reject your action.

## 4. Interaction Actions (CRITICAL: MAX 10 STEPS)
You ONLY have 10 steps to complete this task. Efficiency is paramount. Use batching!
At each step, choose **exactly one** action:
* **Batch Inspect Text:** `<action>check_nodes:[ID1, ID2, ID3]</action>` (Reveal text of up to 5 nodes at once. USE THIS to save steps!)
* **Single Inspect Text:** `<action>check_node:ID</action>`
* **Paint Node:** `<action>paint:ID,Category Name</action>`. Paint ONE node per step to unlock the gate.
* **Submit Answer:** `<action>final:Category Name</action>` (Use immediately after the logic gate is unlocked).

## 5. Response Format
1. **<think>...** Detailed Chain of Thought. Analyze shapes, formulate hypotheses, and plan.
2. **<summary>...** **[MEMORY PASSING]** Summarize actual knowledge derived (e.g., "Node 15 is a Bio paper") for your short-term memory.
3. **<action>...** One valid command.
"""

SYSTEM2_FEW_SHOT = """
---
## 6. Demonstration Example (Fast Execution within 10 Steps)

**Context:** Center Node 8712. Text is hidden. The image shows Node 15 (Hub ▼), Node 22 (1-hop ◯). Max steps: 10.

**Step 1: (Batch Exploration)**
<think>
I must unlock the logic gate quickly. I see Hub Node 15 and local Node 22. 
To save steps, I will check all their texts simultaneously in this single step.
</think>
<summary>Initiating batch exploration. Checking Hub 15 and local 22 to gather all necessary evidence at once.</summary>
<action>check_nodes:[15, 22]</action>

**Observation:** Node 15: "Graph Neural Networks survey..." | Node 22: "Bioinformatics application..."

**Step 2: (Paint Hub)**
<think>
Node 15 is clearly a 'Machine Learning' survey. Node 22 is 'Biomolecular Engineering'.
I will start unlocking the gate by painting the Hub Node 15 first.
</think>
<summary>Evidence confirms Node 15 is ML and Node 22 is Bio. Painting Hub Node 15 as Machine Learning.</summary>
<action>paint:15,Machine Learning</action>

**Observation:** [DELAYED FEEDBACK] Node 15 painted as 'Machine Learning'.

**Step 3: (Paint Local & Fulfill Quota)**
<think>
Hub is painted. Now I paint a 1-hop node (Node 22) to fulfill the mixed-node quota and unlock the gate.
</think>
<summary>Hub 15 painted. Now painting local Node 22 as Biomolecular Engineering to fulfill the unlocking quota.</summary>
<action>paint:22,Biomolecular Engineering</action>

**Observation:** [DELAYED FEEDBACK] Node 22 painted. Gate UNLOCKED (2/2).

**Step 4: (Check Center & Finalize)**
<think>
The gate is unlocked. Since Node 22 (Bio) connects to it, the center is likely Bio. Let me verify quickly.
</think>
<summary>Gate unlocked. Checking the center node 8712's text to make the final determination.</summary>
<action>check_node:8712</action>

**Observation:** Node 8712: "We utilize GNNs for fast drug discovery..."

**Step 5: (Submit - Task completed in 5 steps)**
<think>
Center text confirms drug discovery. Category is Biomolecular Engineering.
</think>
<summary>Center text confirms Biomolecular Engineering. Submitting final answer.</summary>
<action>final:Biomolecular Engineering</action>
"""

# =================================================================
# 3. Prompt Assembly Templates
# =================================================================

TEMPLATE_NO_HIS = """{task_instruction}
{few_shot}

=== Initial State ===
{initial_state}

=== Current Observation ===
Current Visual View: <image>

Current step: 1

Response Format:
1. First, analyze the current state. Output your thinking naturally inside <think>...</think>.
2. Next, consolidate your reasoning into a detailed summary inside <summary>...</summary>.
3. Finally, on a new line, output the chosen action wrapped in <action>...</action> tags.
"""

TEMPLATE_WITH_HIS = """{task_instruction}
{few_shot}

=== Initial State (Reference) ===
{initial_state}

=== History ===
(The following is a log of your previous actions and observations):
{memory_context}

=== Current Observation ===
Current Visual View (Snapshot after your last action): <image>

Current step: {step_count}

Response Format:
1. First, review the history and analyze the current state inside <think>...</think>.
2. Next, consolidate your reasoning into a detailed summary inside <summary>...</summary>.
3. Finally, on a new line, output the chosen action wrapped in <action>...</action> tags.
"""

# =================================================================
# 4. Environment Manager
# =================================================================

class GraphSearchEnvironmentManager(EnvironmentManagerBase):
    """
    Manager for the Graph Search Environment supporting Ada-Fog v4.0.
    Dynamically routes to System 1 or System 2 prompts based on environment state.
    """

    def __init__(self, envs, projection_f, config):
        self.memory = FullSequenceSearchMemory()
        super().__init__(envs, projection_f, config)
        
        # Regex patterns to parse the model output (Strict format enforcement)
        self._think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        self._summary_pattern = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
        self._action_pattern = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
        
        # Track mode per environment in batch
        self.initial_modes = []

        # =========================================================
        # [NEW] Check_Log 跟踪系统初始化 (修改为单个 JSONL 文件)
        # =========================================================
        self.log_dir = "/mnt/cephfs/haowengao/Visual_Graph/Check_Log"
        os.makedirs(self.log_dir, exist_ok=True)
        # 为当前 run 创建一个单一的 .jsonl 文件
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_run_log_file = os.path.join(self.log_dir, f"run_{run_timestamp}.jsonl")
        
        # 内部状态，用来记录每个 batch 维度的当前 episode 轨迹
        self.episode_trajectories = {}
        # =========================================================

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        # Call environment reset
        text_obs, image_obs, infos = self.envs.reset(kwargs=kwargs)
        self.initial_states = text_obs
        
        # Store the routing mode (System1 or System2) determined by env
        self.initial_modes = [info.get("mode", "System2") for info in infos]
        
        self.memory.reset(batch_size=len(text_obs))

        # =========================================================
        # [NEW] Check_Log: 重置当前轨迹记录
        # =========================================================
        for i in range(len(text_obs)):
            self.episode_trajectories[i] = {
                "center_id": infos[i].get("center_id", "Unknown"),
                "answer": infos[i].get("answer", "Unknown"),
                "mode": self.initial_modes[i],
                "steps": [],
                "initial_obs": text_obs[i]
            }
        # =========================================================

        observations = {
            "text": self.build_text_obs(init=True), 
            "image": image_obs, 
            "anchor": text_obs.copy(), 
        }

        # [NEW] Check_Log: 记录初始的 Prompt
        for i, prompt in enumerate(observations["text"]):
             if i in self.episode_trajectories:
                 self.episode_trajectories[i]["initial_prompt"] = prompt

        return observations, infos

    def step(self, text_actions: List[str]):
        summaries = []
        thinks = []
        
        for raw_text in text_actions:
            # Extract Summary
            s_match = self._summary_pattern.search(raw_text)
            if s_match:
                summaries.append(s_match.group(1).strip())
            else:
                summaries.append("No summary provided.")
            
            # Extract Think
            t_match = self._think_pattern.search(raw_text)
            if t_match:
                thinks.append(t_match.group(1).strip())
            else:
                # Fallback: if no explicit think tags, treat pre-summary text as thought
                if s_match:
                    pre_summary = raw_text.split("<summary>")[0].strip()
                    thinks.append(pre_summary if pre_summary else None)
                else:
                    thinks.append(None)

        # Map actions to environment specific format using projection.py
        actions, valids = self.projection_f(text_actions)

        # Execute step in environment
        next_text_obs, next_image_obs, rewards, dones, infos = self.envs.step(actions)

        # Store experience in memory
        self.memory.store({
            "search": actions,
            "information": next_text_obs,
            "summary": summaries 
        })

        # Build next observation
        next_observations = {
            "text": self.build_text_obs(init=False), 
            "image": next_image_obs, 
            "anchor": next_text_obs.copy(),
        }

        # Update Info dicts for monitoring and PPO/GRPO updates
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["parsed_think"] = thinks[i]
            info["parsed_summary"] = summaries[i]
            a_match = self._action_pattern.search(text_actions[i])
            parsed_act_content = a_match.group(1).strip() if a_match else "No Action Found"
            info["parsed_action_content"] = parsed_act_content

            # =========================================================
            # [NEW] Check_Log: 记录每一步的详细信息
            # =========================================================
            if i in self.episode_trajectories:
                step_record = {
                    "step_idx": len(self.episode_trajectories[i]["steps"]) + 1,
                    "model_raw_output": text_actions[i],
                    "parsed_think": thinks[i],
                    "parsed_summary": summaries[i],
                    "parsed_action": parsed_act_content,
                    "env_executed_action": actions[i],
                    "is_valid_format": bool(valids[i]),
                    "env_feedback_obs": next_text_obs[i],
                    "step_reward": float(rewards[i]),
                    "done": bool(dones[i]),
                    "next_prompt": next_observations["text"][i] if not dones[i] else "N/A (Episode Finished)"
                }
                self.episode_trajectories[i]["steps"].append(step_record)

                # 如果环境结束了，将这条轨迹追加写入 jsonl 日志文件并从内存清理
                if dones[i]:
                    self.episode_trajectories[i]["final_reward"] = float(rewards[i])
                    self.episode_trajectories[i]["won"] = info.get("won", False)
                    # 处理 System2 的 hindsight rewards 记录
                    if "hindsight_rewards" in info:
                        self.episode_trajectories[i]["hindsight_rewards"] = info["hindsight_rewards"]
                    
                    self._save_trajectory_to_disk(i)
            # =========================================================

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    # =========================================================
    # [NEW] Check_Log: 改为以追加模式写入单独的 JSONL 文件
    # =========================================================
    def _save_trajectory_to_disk(self, env_idx: int):
        if env_idx not in self.episode_trajectories:
            return
            
        # 提取并同时从内存字典中删除该轨迹
        traj_data = self.episode_trajectories.pop(env_idx)
        
        # 添加一些基础信息便于在单行里检索
        center_id = traj_data.get("center_id", "Unknown")
        mode = traj_data.get("mode", "Unknown")
        won = traj_data.get("won", False)
        status = "WIN" if won else "LOSE"
        traj_data["episode_status"] = status
        
        try:
            # 使用 'a' 追加模式，一行写一个 JSON 对象（JSON Lines 格式）
            with open(self.current_run_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(traj_data, ensure_ascii=False) + '\n')
                
            # 在控制台打印简短的统计信息
            print(f"[Check_Log] Appended to log | Mode: {mode} | Center: {center_id} | Result: {status} | Steps: {len(traj_data['steps'])} | Reward: {traj_data.get('final_reward', 0.0)}")
            
        except Exception as e:
            print(f"[Check_Log] Error saving trajectory for env {env_idx}: {e}")
    # =========================================================

    def build_text_obs(self, init: bool) -> List[str]:
        batch_size = len(self.initial_states)
        rendered_prompts: List[str] = []

        if not init:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search",
                summary_key="summary"
            )
        else:
            memory_ctx = [""] * batch_size

        for i in range(batch_size):
            # Dynamic Prompt Routing based on Epsilon-Greedy Prior
            current_mode = self.initial_modes[i]
            
            if current_mode == "System1":
                task_inst = SYSTEM1_TASK_INSTRUCTION
                few_shot = SYSTEM1_FEW_SHOT
            else:
                task_inst = SYSTEM2_TASK_INSTRUCTION
                few_shot = SYSTEM2_FEW_SHOT

            if init:
                prompt = TEMPLATE_NO_HIS.format(
                    task_instruction=task_inst,
                    few_shot=few_shot,
                    initial_state=self.initial_states[i]
                )
            else:
                prompt = TEMPLATE_WITH_HIS.format(
                    task_instruction=task_inst,
                    few_shot=few_shot,
                    initial_state=self.initial_states[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]) + 1
                )
            
            rendered_prompts.append(prompt)

        return rendered_prompts