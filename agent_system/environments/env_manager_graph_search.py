from typing import Any, Dict, List, Tuple

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SearchMemory


# =========================
# 1. 固定任务指令（System / Instruction）
# =========================

GRAPH_SEARCH_TASK_INSTRUCTION = """You are a graph reasoning agent.

You will be given:
- a CENTER node (ID + text),
- a rendered subgraph image (<image>),
- a legend and color distribution describing neighbor categories.

Your goal:
Predict the category of the CENTER node.

You may:
- directly answer if you are confident, or
- inspect one or multiple neighbor node texts per step to gather more evidence.

Follow the interaction rules strictly.
"""


# =========================
# 2. Prompt 模板（无历史 / 有历史）
# =========================

GRAPH_SEARCH_TEMPLATE_NO_HIS = """{task_instruction}

Initial state:
{initial_state}

You may choose EXACTLY ONE action:

<action>view_node:<node_id></action>
<action>view_nodes:[<node_id_1>,<node_id_2>,...,<node_id_k>]</action>  (k ≤ 5)
<action>final:<category_name></action>

You must output ONLY the action.

You may reason internally before acting, but you MUST output ONLY the final action
wrapped in <action>...</action> tags.

Do NOT include your reasoning or any other text outside the action tags.
"""


GRAPH_SEARCH_TEMPLATE_WITH_HIS = """{task_instruction}

Initial state:
{initial_state}

History (previous actions and observations):
{memory_context}

Current step: {step_count}

You may choose EXACTLY ONE action:

<action>view_node:<node_id></action>
<action>view_nodes:[<node_id_1>,<node_id_2>,...,<node_id_k>]</action>  (k ≤ 5)
<action>final:<category_name></action>

You must output ONLY the action.

You may reason internally before acting, but you MUST output ONLY the final action
wrapped in <action>...</action> tags.

Do NOT include your reasoning or any other text outside the action tags.
"""


# =========================
# 3. GraphSearchEnvironmentManager
# =========================

class GraphSearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for GraphSearch.

    Responsibilities:
    - maintain initial state (from env.reset)
    - maintain multi-step memory (actions + observations)
    - render prompts for the agent
    - bridge model outputs <-> env actions

    This class DOES NOT:
    - define world dynamics (env does)
    - define reward (env does)
    - perform reasoning (model does)
    """

    def __init__(self, envs, projection_f, config):
        # 与 Search 对齐：使用 SearchMemory
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    # -------------------------
    # reset：episode 起点
    # -------------------------
    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Reset environments with a batch of samples.

        Expected env.reset return:
          text_obs  : List[str]   (initial state per sample)
          image_obs : List[bytes] (graph image per sample)
          infos     : List[Dict]
        """
        text_obs, image_obs, infos = self.envs.reset(kwargs=kwargs)

        # ★ 语义纠正：这是「初始状态」，不是任务描述
        self.initial_states = text_obs

        # 清空历史记忆（batch 对齐）
        self.memory.reset(batch_size=len(text_obs))

        observations = {
            "text": self.build_text_obs(init=True),
            "image": image_obs,                   # 直接透传给 VLM
            "anchor": text_obs.copy(),            # GiGPO / logging 用
        }

        return observations, infos

    # -------------------------
    # step：多轮交互
    # -------------------------
    def step(self, text_actions: List[str]):
        """
        text_actions: raw model outputs (strings)
        """
        # 1) projection：抽取 env action
        actions, valids = self.projection_f(text_actions)

        # 2) 执行环境 step
        next_text_obs, next_image_obs, rewards, dones, infos = self.envs.step(actions)

        # 3) 记录历史（Search 风格）
        self.memory.store({
            "search": actions,
            "information": next_text_obs,
        })

        next_observations = {
            "text": self.build_text_obs(init=False),
            "image": next_image_obs,
            "anchor": next_text_obs.copy(),
        }

        # 把 action 是否合法写入 info（与 Search 对齐）
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    # -------------------------
    # Prompt 构建（核心）
    # -------------------------
    def build_text_obs(self, init: bool) -> List[str]:
        """
        Render prompts for each environment in the batch.

        init=True  : 无历史（step 0）
        init=False : 含 history
        """
        batch_size = len(self.initial_states)
        rendered_prompts: List[str] = []

        # 拉取历史（仅在非 init 且 history_length > 0 时）
        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search",
            )

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
