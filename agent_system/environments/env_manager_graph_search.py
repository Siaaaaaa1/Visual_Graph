from typing import Any, Dict, List, Tuple

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SearchMemory

# =========================
# 1. 固定任务指令 (System / Instruction)
# =========================

GRAPH_SEARCH_TASK_INSTRUCTION = """You are a graph reasoning agent.

You will be given:
- a CENTER node (ID + text),
- a rendered subgraph image (<image>),
- a legend and color distribution describing neighbor categories.

Your goal:
Predict the category of the CENTER node.

You have the following tools to gather information:

1. **Check Node Text**: Inspect the text content of specific nodes.
2. **Check Graph View**: Update the visual graph to focus on different aspects.
   - Hop Mode:
       * '1-hop': only immediate neighbors
       * '2-hop': include both 1-hop and 2-hop neighbors
   - Rank Mode:
       * 'hop': prioritize closer hops (1-hop before 2-hop)
       * 'sim': prioritize nodes by semantic similarity to the center node
   - Max Nodes: Integer limit for the number of nodes to draw (e.g., 10, 20).

When submitting the final answer, use the EXACT category name shown in the legend (case-insensitive).
Do NOT paraphrase or modify the category name.

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
   <action>check_nodes:[<node_id_1>,<node_id_2>,...,<node_id_k>]</action>  (k ≤ 5)

2. Update Graph Visualization:
   <action>check_graph:<hop_mode>,<rank_mode>,<max_nodes></action>
   Example: <action>check_graph:2-hop,sim,20</action>
   (Valid hop_mode: 1-hop, 2-hop; Valid rank_mode: hop, sim)

3. Submit Answer:
   <action>final:<category_name></action>
   (Use the EXACT category name from the legend.)

Response Format:
1. First, analyze the image and text to decide your next step. Wrap your reasoning inside <think>...</think> tags.
2. Then, on a new line, output the chosen action wrapped in <action>...</action> tags.

Example:
<think>The center node is connected to several blue nodes, but I need to verify their text content to be sure about the category.</think>
<action>check_nodes:[12, 45]</action>
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
   <action>check_nodes:[<node_id_1>,<node_id_2>,...,<node_id_k>]</action>  (k ≤ 5)

2. Update Graph Visualization:
   <action>check_graph:<hop_mode>,<rank_mode>,<max_nodes></action>
   Example: <action>check_graph:2-hop,sim,20</action>
   (Valid hop_mode: 1-hop, 2-hop; Valid rank_mode: hop, sim)

3. Submit Answer:
   <action>final:<category_name></action>
   (Use the EXACT category name from the legend.)

Response Format:
1. First, review the history and analyze the current state. Wrap your reasoning inside <think>...</think> tags.
2. Then, on a new line, output the chosen action wrapped in <action>...</action> tags.

Example:
<think>I have already checked the text of node 12 in the previous step. Now I need to see a broader view of the graph structure.</think>
<action>check_graph:2-hop,sim,20</action>
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
            "image": image_obs,  # 直接透传 bytes 给 VLM Processor
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