from typing import Any, Dict, List, Tuple

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SearchMemory

# =========================
# 1. 固定任务指令 (System / Instruction)
# =========================

GRAPH_SEARCH_TASK_INSTRUCTION = """You are a graph reasoning agent.

You will be given:
- a CENTER node (ID + text),
- statistical info about the center node (degrees, etc.),
- a list of CANDIDATE CLASSES (from surrounding nodes).

**Important:** Initially, you do NOT have a view of the graph neighbors. You see only the center node. You must actively query the graph view to see neighbors.

Your goal:
Predict the category of the CENTER node.

You have 3 options to gather information and solve the task:

1. **Check Node Text**: Inspect the text content of specific nodes.
2. **Check Graph View**: Render a subgraph view centered on the target node. You can specify the 'View Mode' and 'Max Nodes'.
   - **View Modes**:
       * `1-hop`: Show only immediate neighbors. (Truncates by similarity if exceeding max).
       * `2-hop`: Show 1-hop and 2-hop neighbors. (Truncates by similarity).
       * `sim`: Show nodes with highest semantic similarity to center, regardless of hop distance.
       * `1-hop+sim`: Priority to 1-hop neighbors. If count < max, fill with high-similarity nodes.
       * `2-hop+sim`: Priority to 1-hop and 2-hop. If count < max, fill with high-similarity nodes.
   - **Max Nodes**: Integer limit (e.g., 10, 20).
3. **Submit Answer**: Submit the final category prediction.

When submitting the final answer, use the EXACT category name from the provided list or legend (case-insensitive).

Follow the interaction rules strictly.
"""


# =========================
# 2. Prompt 模板 (CoT via <think> tags)
# =========================

GRAPH_SEARCH_TEMPLATE_NO_HIS = """{task_instruction}

Initial state:
{initial_state}

You may choose EXACTLY ONE action from the following list:

1. Inspect Node Text:
   <action>check_node:<node_id></action>
   <action>check_nodes:[<node_id_1>,<node_id_2>,...]</action>

2. Update Graph Visualization:
   <action>check_graph:<view_mode>,<max_nodes></action>
   Example: <action>check_graph:1-hop+sim,15</action>
   (Valid modes: 1-hop, 2-hop, sim, 1-hop+sim, 2-hop+sim)

3. Submit Answer:
   <action>final:<category_name></action>

Response Format:
1. First, analyze the current information (text or graph) to decide your next step. Wrap your reasoning inside <think>...</think> tags.
2. Then, on a new line, output the chosen action wrapped in <action>...</action> tags.

Example:
<think>I see the center node text is about ML. I need to see its direct neighbors to infer the class.</think>
<action>check_graph:1-hop,10</action>
"""


GRAPH_SEARCH_TEMPLATE_WITH_HIS = """{task_instruction}

Initial state:
{initial_state}

History (previous actions and observations):
{memory_context}

Current step: {step_count}

You may choose EXACTLY ONE action from the following list:

1. Inspect Node Text:
   <action>check_node:<node_id></action>
   <action>check_nodes:[<node_id_1>,<node_id_2>,...]</action>

2. Update Graph Visualization:
   <action>check_graph:<view_mode>,<max_nodes></action>
   Example: <action>check_graph:1-hop,20</action>
   (Valid modes: 1-hop, 2-hop, sim, 1-hop+sim, 2-hop+sim)

3. Submit Answer:
   <action>final:<category_name></action>

Response Format:
1. First, review the history and analyze the current state. Wrap your reasoning inside <think>...</think> tags.
2. Then, on a new line, output the chosen action wrapped in <action>...</action> tags.

Example:
<think>The 1-hop neighbors are mixed. I want to see if there are any high-similarity nodes further away.</think>
<action>check_graph:sim,10</action>
"""


# =========================
# 3. GraphSearchEnvironmentManager
# =========================

class GraphSearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for GraphSearch.
    """

    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        # text_obs 是初始状态描述，image_obs 是 list of bytes
        text_obs, image_obs, infos = self.envs.reset(kwargs=kwargs)

        self.initial_states = text_obs
        self.memory.reset(batch_size=len(text_obs))

        observations = {
            "text": self.build_text_obs(init=True),
            "image": image_obs,
            "anchor": text_obs.copy(),
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        # 1) Projection
        actions, valids = self.projection_f(text_actions)

        # 2) Environment Step
        next_text_obs, next_image_obs, rewards, dones, infos = self.envs.step(actions)

        # 3) Memory Store
        self.memory.store({
            "search": actions,
            "information": next_text_obs,
        })

        # 4) Build Next Observation
        next_observations = {
            "text": self.build_text_obs(init=False),
            "image": next_image_obs,
            "anchor": next_text_obs.copy(),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, init: bool) -> List[str]:
        batch_size = len(self.initial_states)
        rendered_prompts: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search",
            )
        else:
            memory_ctx = [""] * batch_size

        for i in range(batch_size):
            if init or self.config.env.history_length <= 0:
                prompt = GRAPH_SEARCH_TEMPLATE_NO_HIS.format(
                    task_instruction=GRAPH_SEARCH_TASK_INSTRUCTION,
                    initial_state=self.initial_states[i],
                )
            else:
                prompt = GRAPH_SEARCH_TEMPLATE_WITH_HIS.format(
                    task_instruction=GRAPH_SEARCH_TASK_INSTRUCTION,
                    initial_state=self.initial_states[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            rendered_prompts.append(prompt)

        return rendered_prompts