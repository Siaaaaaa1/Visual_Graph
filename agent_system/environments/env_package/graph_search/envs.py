import json
from typing import List, Dict, Any


# ============================================================
# Single-episode Graph Search Environment
# (Search-style semantics: terminal reward only, no shaping)
# ============================================================

class GraphSearchEnv:
    """
    One episode of graph-based active information search.

    Reward semantics:
      - reward is a scalar encoding of task success
      - intermediate steps do NOT shape reward
      - only final action may produce reward (success = 1, else 0)
    """

    def __init__(self, max_steps: int, node_text_db: Dict[str, str]):
        self.max_steps = max_steps
        self.node_text_db = node_text_db  # global, read-only
        self._reset_internal()

    def _reset_internal(self):
        self.step_count = 0
        self.seen_nodes = set()
        self.done = False

    def reset(self, kwargs: Dict[str, Any]) -> str:
        """
        Reset environment using one parquet sample.
        """
        self._reset_internal()

        self.center_id = kwargs["center_id"]
        self.center_text = kwargs["center_text"]
        self.image_bytes = kwargs["image_bytes"]

        self.legend = kwargs["legend"]
        #self.color_distribution = kwargs["color_distribution"]
        color_dist = kwargs["color_distribution"]
        if isinstance(color_dist, str):
            color_dist = json.loads(color_dist)
        self.color_distribution = color_dist

        inspectable = kwargs["inspectable_nodes"]

        if isinstance(inspectable, str):
            inspectable = json.loads(inspectable)


        
        self.inspectable_nodes = (
            set(inspectable.get("1hop", []))
            | set(inspectable.get("2hop", []))
        )

        self.answer = kwargs["answer"]

        # Initial observation:
        # - includes center node ID as an anchor
        # - does NOT include neighbor texts
        obs = (
            f"Center node ID: {self.center_id}\n"
            f"Center node text:\n{self.center_text}\n\n"
            f"Legend:\n{self.legend}\n\n"
            f"Color distribution:\n{self.color_distribution}"
        )

        return obs

    def step(self, action: str):
        """
        Execute one action.

        Returns:
          obs   : textual observation
          reward: scalar success indicator (Search-style)
          done  : episode termination flag
          info  : auxiliary info (no learning semantics)
        """
        # If already finished, behave like SearchEnv: no-op
        if self.done:
            return "", 0, True, {}

        self.step_count += 1

        reward = 0
        done = False

        # --------------------------------------------------
        # view_node action (information acquisition)
        # --------------------------------------------------
        if action.startswith("view_node:"):
            try:
                node_id = int(action.split(":", 1)[1])
            except Exception:
                obs = "Invalid node id format."
            else:
                if node_id not in self.inspectable_nodes:
                    obs = f"Node {node_id} is not inspectable."
                elif node_id in self.seen_nodes:
                    obs = f"Node {node_id} has already been inspected."
                else:
                    self.seen_nodes.add(node_id)
                    text = self.node_text_db.get(str(node_id), "")
                    obs = f"Text of node {node_id}:\n{text}"

        # --------------------------------------------------
        # final action (task completion attempt)
        # --------------------------------------------------
        elif action.startswith("final:"):
            pred = action.split(":", 1)[1].strip()
            obs = "Final answer submitted."
            done = True
            self.done = True

            # success is encoded numerically, not shaped
            if pred == self.answer:
                reward = 1

        # --------------------------------------------------
        # invalid / unparsable action
        # --------------------------------------------------
        else:
            obs = "Invalid action."

        # --------------------------------------------------
        # timeout (Search-style: no explicit penalty)
        # --------------------------------------------------
        if not done and self.step_count >= self.max_steps:
            done = True
            self.done = True

        info = {
            "step": self.step_count,
            "seen_nodes": list(self.seen_nodes),
        }

        return obs, reward, done, info


# ============================================================
# Batch wrapper (aligns with Search / AlfWorld env interface)
# ============================================================

def build_graph_search_envs(
    seed: int,          # ← 新增，但可以不用
    env_num: int,
    group_n: int,
    is_train: bool,
    env_config
):
    """
    Build batched graph-search environments.

    This function mirrors the role of build_search_envs:
    - dataset-driven reset
    - batched step/reset
    - reward semantics delegated to single env
    """
    batch_size = env_num * group_n
    max_steps = env_config.max_steps

    # Load node text database ONCE (Search-style design)
    with open(env_config.node_text_path, "r", encoding="utf-8") as f:
        node_text_db = json.load(f)

    envs = [
        GraphSearchEnv(
            max_steps=max_steps,
            node_text_db=node_text_db
        )
        for _ in range(batch_size)
    ]

    class BatchGraphSearchEnv:
        # def reset(self, kwargs_list: List[Dict[str, Any]]):
        #     text_obs, image_obs, infos = [], [], []
        def __init__(self):
            self.num_envs = batch_size
        def reset(self, kwargs):
            kwargs_list = kwargs
            text_obs, image_obs, infos = [], [], []
            for env, kw in zip(envs, kwargs_list):
                obs = env.reset(kw)
                text_obs.append(obs)
                image_obs.append(kw["image_bytes"])
                infos.append({})

            return text_obs, image_obs, infos

        def step(self, actions: List[str]):
            text_obs, image_obs = [], []
            rewards, dones, infos = [], [], []

            for env, act in zip(envs, actions):
                obs, r, d, info = env.step(act)
                text_obs.append(obs)
                image_obs.append(env.image_bytes)
                rewards.append(r)
                dones.append(d)
                infos.append(info)

            return text_obs, image_obs, rewards, dones, infos

        def close(self):
            return

    return BatchGraphSearchEnv()
