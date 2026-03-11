import re
import os
import time
import json
from typing import Any, Dict, List, Tuple
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import FullSequenceSearchMemory

V_GRAPH_AGENT_INSTRUCTION = """

# V-GraphAgent: RL-Driven Multimodal Graph Exploration

You are a multimodal graph intelligence agent. Your goal is to predict the category of the center node (square node) in the visual graph.

## Visual-Text Reference Rules
1. **Visual Ring Navigation**:
   - **Inner ring (1-hop)**: Direct neighbors of the center node. They share direct edges (e.g., mutual citations) and provide the strongest local context and semantic reference.
   - **Outer ring (2-hop & macro nodes)**: Provide broader community structure and domain distribution clues.
2. **Node Attribute Mapping**:
   - **Color (semantics)**: Represents text feature similarity to the center node (warm/deep red = high similarity, cool/deep blue = low similarity, white = neutral).
   - **Shape (topological importance)**:
     - **■ (Center node)**: Your prediction target.
     - **★ (Macro Anchor)**: Critically important intelligence nodes! The system prompt will directly tell you which academic field each ★ node most likely represents.
     - **▲ (High In-degree Hub)**: Widely cited foundational authority.
     - **▼ (High Out-degree Hub)**: Survey-like node that cites broadly.
     - **● (Normal node)**: Regular neighbor.

## Exploration & Decision Mechanism
- **Break pure text dependency**: If the center node's prior alone is insufficient, you **must** call tools to inspect surrounding nodes. Queries return title, abstract, in-degree, and out-degree.
- **Step budget**: Each action can query at most 5 nodes. Return your final answer within 6 steps.

## Action Space
- Query node summaries and degrees: `<action>check_nodes([ID1, ID2, ID3])</action>` (max 5 IDs per call)
- Submit final answer: `<action>final(Category)</action>` (Category must exactly match one from the candidate list)

**Format rules (violations are treated as invalid actions):**
- Every reply MUST contain a complete `<thinking>...</thinking>` block followed by `<action>...</action>`.
- All content inside `<action>` tags must use standard English punctuation: parentheses `()`, brackets `[]`, comma `,`.
- Correct: `<action>check_nodes([42, 163, 565])</action>`

## Reasoning Framework (Metacognitive CoT)
You MUST produce a `<thinking>` block before every `<action>`. The sub-headings below are suggested guides — feel free to deviate, add self-questioning, hypothesis testing, or error correction beyond these templates. Just ensure you output a belief distribution and derive the next action.

### [Round 1 — Suggested Reasoning]
<thinking>
[Initial Visual & Prior Analysis]: Combine the center node's title/abstract (if available) and initial prediction. Observe the radar chart: which 1-hop inner nodes are high-similarity (red)? Which are low-similarity (blue)? Are there key Hubs (▲, ▼)? How are the ★ macro anchors positioned relative to the center?
[Multi-category Divergence]: What cross-domain areas might the center node touch? Are there overlapping or hierarchical relationships between candidate categories? Freely expand divergent hypotheses here.
[Initial Belief Distribution]: Output your current probability estimate in strict JSON format (sum = 1.0):
Belief: {"CategoryA": 0.6, "CategoryB": 0.3, "CategoryC": 0.1}
[Exploration Strategy]: Is the current distribution confident enough to submit a final answer? If not, which specific high-value node IDs should I query next (e.g., red ▲ in 1-hop, or nodes near a ★ anchor)?
</thinking>
<action>...</action>

### [Subsequent Rounds — Suggested Reasoning]
<thinking>
[New Evidence Integration & Self-Reflection]: Which nodes did I query last round? Do their degrees, titles, and abstracts support or contradict my main hypothesis? Are there flaws in my previous logic? How does this update my belief distribution? Do I need more queries? What do blue/red colors, high-degree hubs, and ★ anchors each contribute? Are there contradictions or consistent signals?
[Free Logical Reasoning]: Perform flexible cross-referencing and deep inference. E.g., "Node X is a red ▲ (high in-degree), suggesting the underlying technology belongs to domain A; the ★ anchor representing domain C is far away... Overall, the center node leans toward..."
[Belief Update]: Update category probabilities based on visual topology and accumulated text evidence. Output in strict JSON format:
Belief: {"CategoryA": 0.85, "CategoryC": 0.15}
[Current Round Decision]: Has the information converged to a single clear category? If yes, call `final(Category)`. If there is still meaningful uncertainty, which IDs should I query next?
</thinking>
<action>...</action>
"""

class GraphSearchEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = FullSequenceSearchMemory()
        super().__init__(envs, projection_f, config)
        self._think_pattern = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
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