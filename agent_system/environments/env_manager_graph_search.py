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

You are a graph reasoning agent. The environment has analyzed the current Ego-graph and determined it has an **Extreme Margin of Victory** (the dominant category overwhelmingly outnumbers any other category). 
Therefore, you have been routed to **System 1 Mode (Clear Weather)**.

## 1. Initial State & Context
* **Full Visibility:** There is NO Fog of War. You can visually see the predicted categories (colors) of all neighbor nodes.
* **Context Provided:** You are provided with the text of the Center Node, PLUS the texts of representative nodes scaled by cluster size (up to 3 for the dominant category, 2 for secondary, and 1 for minor categories; 1-hop preferred) and key global hubs from your neighborhood.
* **Candidate Categories:** A list of valid categories present in the current view is provided.

## 2. Constraints & Rules
* **NO EXPLORATION ALLOWED:** You are strictly forbidden from using exploration actions (`check_node`, `check_graph`, `paint`).
* **The "Majority Trap":** Although the environment exhibits high homophily, **the topological majority is NOT ALWAYS the Ground Truth**. Sometimes a node applies a method to a new domain, connecting to many method nodes but belonging to the application domain.
* **Action Required:** You MUST logically align the center node's text with the provided representative texts. If the center text aligns with a minority representative, you must override the visual majority. If it aligns with the majority, follow it.
* **Strict Vocabulary:** You MUST choose your final answer exactly from the provided **Candidate Categories** list. Do not invent new category names.

## 3. Interaction Actions
At each step, you must choose **exactly one** action:
* **Submit Answer:** `<action>final:Category Name</action>`

## 4. Response Format
1. **<think>...** Analyze the center text against the representative texts. Decide whether to follow or override the topological majority.
2. **<summary>...** State your final conclusion and textual justification.
3. **<action>...** Output your final action using a category from the Candidate Categories.
"""

SYSTEM1_FEW_SHOT = """
---
## 5. Demonstration Examples (System 1)

### Example 5.1: Overriding the Majority (The "Majority Trap")
**Context:** Center Node 38491. Text states: "We propose a novel Transformer architecture for medical image tumor segmentation...". 
Visual graph legend shows 80% Blue (Natural Language Processing) and 20% Red (Computer Vision).
Candidate Categories: Natural Language Processing, Computer Vision.
Supplementary Neighbors Context:
- [Representative for Natural Language Processing (1-hop) | Node 12]: "Attention is all you need for language translation tasks..."
- [Representative for Computer Vision (1-hop) | Node 55]: "Image classification using convolutional neural networks..."

**Step 1:**
<think>
I am in System 1 (Clear Weather). I must evaluate the texts without exploring.
1. Topology Analysis: The dominant majority is Natural Language Processing (80%).
2. Textual Verification (CRITICAL): The center node proposes a "Transformer architecture" (often NLP) but explicitly applies it to "medical image tumor segmentation".
3. Representative Matching: The NLP representative discusses "language translation", which does not match. The Computer Vision representative discusses "Image classification", which perfectly aligns with the center node's visual task.
Conclusion: The high NLP neighbor count is a citation artifact. The semantic Ground Truth of the center node is Computer Vision. Both categories are in the Candidate Categories list. I must NOT blindly follow the 80% majority.
</think>
<summary>Despite the 80% NLP topological majority, textual verification proves the center node applies transformers to images. It perfectly aligns with the Computer Vision representative text, thus overriding the misleading topological prior.</summary>
<action>final:Computer Vision</action>

### Example 5.2: Aligning with the Majority (Homophily Holds)
**Context:** Center Node 1024. Text states: "This paper introduces a new message passing scheme for Graph Neural Networks to improve node classification...". 
Visual graph legend shows 90% Green (Graph Learning) and 10% Yellow (Optimization).
Candidate Categories: Graph Learning, Optimization.
Supplementary Neighbors Context:
- [Representative for Graph Learning (1-hop) | Node 8]: "A comprehensive survey on Graph Neural Networks and their applications..."
- [Representative for Optimization (1-hop) | Node 23]: "Stochastic gradient descent with momentum..."

**Step 1:**
<think>
I am in System 1 (Clear Weather). I must evaluate the texts.
1. Topology Analysis: The dominant majority is Graph Learning (90%).
2. Textual Verification: The center node introduces a "message passing scheme for Graph Neural Networks".
3. Representative Matching: The Graph Learning representative explicitly discusses "Graph Neural Networks", matching the center node perfectly. The Optimization representative is secondary.
Conclusion: The topological majority perfectly aligns with the center node's semantic content. I will confidently follow the majority.
</think>
<summary>The center node discusses message passing in GNNs, which perfectly aligns with the 90% Graph Learning topological majority and its representative text. The majority is valid and correct.</summary>
<action>final:Graph Learning</action>
"""

# =================================================================
# 2. System 2 (Slow Thinking) Prompts
# =================================================================

SYSTEM2_TASK_INSTRUCTION = """
# SYSTEM 2: SLOW THINKING (FOG OF WAR: ANONYMOUS COLORS)

You are a graph reasoning agent. The environment has determined this graph lacks a clear margin of victory (it is deceptive, heterogeneous, or sits on a boundary). 
Therefore, you have been routed to **System 2 Mode (Fog of War)**.

## 1. Fog of War Mechanics (Anonymous Color Mapping)
* **Text Masked:** The center node's text is hidden.
* **Colors Visible, Semantics Masked:** You CAN see the colors of neighbor nodes. Nodes sharing the same color belong to the same category. However, the actual semantic names are anonymized (e.g., "Group 1", "Group 2").
* **Candidate Categories:** You are provided with a list of ALL valid categories present in your current view. Your job is to map the anonymous groups to these candidate categories.

## 2. Visual Topology & Shape Semantics
Rely on topological shapes to prioritize exploration:
* **1-Hop Nodes (Circles ◯):** Immediate neighbors. Represent direct local context.
* **High Out-Degree Nodes (Downward Triangles ▼):** Hub nodes. Foundational methods. 
* **High In-Degree Nodes (Upward Triangles ▲):** Hub nodes. Popular applications.

## 3. The Evidence-Gated Logic Lock & Category Identification
* **Major Clusters Rule:** You MUST successfully `paint` the correct category for **EACH** of the Major Clusters (Anonymous Groups) listed in your observation gate requirement.
* **Strict Vocabulary:** You MUST choose the category name for your `paint` and `final` actions strictly from the provided **Candidate Categories** list.
* **NO IMMEDIATE FEEDBACK:** When you use the `paint` action, the environment will NOT immediately confirm if your semantic mapping is perfectly correct. It will only say "[DELAYED FEEDBACK] Group 'X' painted...". 

## 4. Interaction Actions (CRITICAL: MAX 10 Steps)
You ONLY have 10 steps to complete this task. Efficiency is paramount. Use batching!
At each step, choose **exactly one** action:
* **Batch Inspect Text:** `<action>check_nodes:[ID1, ID2, ID3]</action>` (Reveal text of up to 5 nodes at once. USE THIS to save steps!)
* **Single Inspect Text:** `<action>check_node:ID</action>`
* **Paint Category:** `<action>paint:Group Name,Category Name</action>`. Submit your category identification for an anonymous group. Example: `<action>paint:Group 1,Machine Learning</action>`.
* **Submit Answer:** `<action>final:Category Name</action>` (Use immediately after the logic gate is unlocked).

## 5. Response Format
1. **<think>...** Detailed Chain of Thought. Analyze shapes, autonomously sample nodes to build evidence, verify hypotheses, and map clusters to the Candidate Categories.
2. **<summary>...** Summarize your current findings and progress to ground your next action.
3. **<action>...** One valid command.
"""

SYSTEM2_FEW_SHOT = """
---
## 6. Demonstration Example (Autonomous Sampling & Category Identification)

**Context:** Center Node 8712. Text is hidden. 
Observation states: "GATE REQUIREMENT: You must correctly paint the category for at least ONE node from EACH of these Major Clusters: [Group 1, Group 2]."
Candidate Categories: Machine Learning, Biomolecular Engineering, Data Mining
Image shows 8 Blue nodes (Group 1), 6 Red nodes (Group 2).

**Step 1: (Autonomous Batch Sampling)**
<think>
I need to unlock the gate by identifying and painting the categories for Group 1 (Blue) and Group 2 (Red). 
I must gather enough evidence to prove my hypothesis about what these groups represent, which will also help me understand the center node's context.
I will autonomously sample a mix of high-degree Hubs and 1-hop nodes: Blue Nodes (15, 22) for Group 1, and Red Nodes (8, 30) for Group 2.
</think>
<summary>Initiating batch exploration. Checking Blue nodes 15 and 22, and Red nodes 8 and 30 to build sufficient evidence for category identification.</summary>
<action>check_nodes:[15, 22, 8, 30]</action>

**Observation:** Node 15: "Graph Neural Networks survey..." | Node 22: "Optimizing GNN layers..." | Node 8: "Bioinformatics application of neural networks..." | Node 30: "Molecular property prediction..."

**Step 2: (Hypothesis Proven & Paint Category for Group 1)**
<think>
Both Blue nodes (15, 22) consistently discuss algorithms and GNNs. Looking at the Candidate Categories, "Machine Learning" is the perfect semantic match for 'Group 1'.
I will fulfill the Group 1 requirement by painting the category 'Machine Learning' for Group 1.
</think>
<summary>Evidence is sufficient: Group 1 maps to Machine Learning. Painting the 'Machine Learning' category for Group 1 to satisfy the requirement.</summary>
<action>paint:Group 1,Machine Learning</action>

**Observation:** [DELAYED FEEDBACK] Group 1 painted as 'Machine Learning'. Correctness hidden. Legend updated.

**Step 3: (Paint Category for Group 2)**
<think>
Group 1 requirement met. Now I must paint the category for Group 2. My evidence confirmed it is Biomolecular Engineering.
</think>
<summary>Group 1 fulfilled. Now painting the verified 'Biomolecular Engineering' category for Group 2 to fulfill the quota.</summary>
<action>paint:Group 2,Biomolecular Engineering</action>

**Observation:** [DELAYED FEEDBACK] Group 2 painted as 'Biomolecular Engineering'. Correctness hidden. Gate UNLOCKED.

**Step 4: (Check Center & Finalize)**
<think>
The environment says Gate UNLOCKED. The center node connects to ML methods (Group 1) and Bio applications (Group 2). Let me quickly check the center node's text to finalize the prediction.
</think>
<summary>Gate unlocked. Checking the center node 8712's text to make the final determination.</summary>
<action>check_node:8712</action>

**Observation:** Node 8712: "We utilize GNNs for fast drug discovery..."

**Step 5: (Submit - Task completed in 5 steps)**
<think>
The center text confirms it uses the computational method for an application (drug discovery). Between the candidate categories, this clearly belongs to Biomolecular Engineering.
</think>
<summary>Center text confirms Biomolecular Engineering. Submitting final answer.</summary>
<action>final:Biomolecular Engineering</action>
"""

# =================================================================
# 3. Environment Manager
# =================================================================

class GraphSearchEnvironmentManager(EnvironmentManagerBase):
    """
    Graph Search 场景的全局管理器（Manager）。
    协调底层物理环境群(BatchGraphSearchEnv)、智能体内存系统和提示词构建器（Prompt Routing）。
    """

    def __init__(self, envs, projection_f, config):
        self.memory = FullSequenceSearchMemory()
        super().__init__(envs, projection_f, config)
        
        self._think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        self._summary_pattern = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
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
            if mode == "System1":
                sys_inst = SYSTEM1_TASK_INSTRUCTION + "\n" + SYSTEM1_FEW_SHOT
            else:
                sys_inst = SYSTEM2_TASK_INSTRUCTION + "\n" + SYSTEM2_FEW_SHOT

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
        summaries = []
        thinks = []
        
        for raw_text in text_actions:
            s_match = self._summary_pattern.search(raw_text)
            if s_match:
                summaries.append(s_match.group(1).strip())
            else:
                summaries.append("No summary provided.")
            
            t_match = self._think_pattern.search(raw_text)
            if t_match:
                thinks.append(t_match.group(1).strip())
            else:
                if s_match:
                    pre_summary = raw_text.split("<summary>")[0].strip()
                    thinks.append(pre_summary if pre_summary else None)
                else:
                    thinks.append(None)

        actions, valids = self.projection_f(text_actions)
        next_text_obs, next_image_obs, rewards, dones, infos = self.envs.step(actions)

        # =========================================================
        # [核心修复] 多轮对话状态追加 (坚决不碰历史记录中的 <image>)
        # =========================================================
        for i, act_text in enumerate(text_actions):
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
            "information": next_text_obs,
            "summary": summaries 
        })

        next_observations = {
            "text": [conv.copy() for conv in self.conversations], 
            "image": next_image_obs, # 将新图传入，Processor 会自动去寻找第一轮的 <image> 占位符并替换
            "anchor": next_text_obs.copy(),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["parsed_think"] = thinks[i]
            info["parsed_summary"] = summaries[i]
            a_match = self._action_pattern.search(text_actions[i])
            parsed_act_content = a_match.group(1).strip() if a_match else "No Action Found"
            info["parsed_action_content"] = parsed_act_content

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
                    "next_prompt": str(next_observations["text"][i]) if not dones[i] else "N/A"
                }
                self.episode_trajectories[i]["steps"].append(step_record)

                if dones[i]:
                    self.episode_trajectories[i]["final_reward"] = float(rewards[i])
                    self.episode_trajectories[i]["won"] = info.get("won", False)
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