import json
from typing import List, Dict, Any
import io
import numpy as np
from PIL import Image

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
        img = Image.open(io.BytesIO(self.image_bytes)).convert("RGB")
        #img = img.resize((64, 64), resample=Image.BICUBIC)
        self.image = np.array(img)  # shape: (H, W, 3), dtype: uint8
        self.legend = kwargs["legend"]

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

        # Initial observation (NO neighbor texts)
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
        obs = ""

        # --------------------------------------------------
        # view_node / view_nodes action (information acquisition)
        # --------------------------------------------------
        node_ids = None

        if action.startswith("view_node:"):
            try:
                node_ids = [int(action.split(":", 1)[1])]
            except Exception:
                node_ids = None

        elif action.startswith("view_nodes:"):
            try:
                content = action.split(":", 1)[1].strip()
                content = content.lstrip("[").rstrip("]")
                node_ids = [int(x.strip()) for x in content.split(",") if x.strip()]
            except Exception:
                node_ids = None

        if node_ids is not None:
            texts = []

            for node_id in node_ids:
                if node_id not in self.inspectable_nodes:
                    texts.append(f"Node {node_id} is not inspectable.")
                elif node_id in self.seen_nodes:
                    texts.append(f"Node {node_id} has already been inspected.")
                else:
                    self.seen_nodes.add(node_id)
                    text = self.node_text_db.get(str(node_id), "")
                    texts.append(f"Text of node {node_id}:\n{text}")

            obs = "\n\n".join(texts)

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
        info["won"] = bool(done)

        return obs, reward, done, info


# ============================================================
# Batch wrapper (aligns with Search / AlfWorld env interface)
# ============================================================

def build_graph_search_envs(
    seed: int,
    env_num: int,
    group_n: int,
    is_train: bool,
    env_config
):
    """
    Build batched graph-search environments.
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
        def __init__(self):
            self.num_envs = batch_size

        def reset(self, kwargs):
            kwargs_list = kwargs
            text_obs, image_obs, infos = [], [], []

            for env, kw in zip(envs, kwargs_list):
                obs = env.reset(kw)
                text_obs.append(obs)
                #image_obs.append(kw["image_bytes"])
                image_obs.append(env.image)
                infos.append({})

            return text_obs, image_obs, infos

        def step(self, actions: List[str]):
            text_obs, image_obs = [], []
            rewards, dones, infos = [], [], []

            for env, act in zip(envs, actions):
                obs, r, d, info = env.step(act)
                text_obs.append(obs)
                #image_obs.append(env.image_bytes)
                image_obs.append(env.image)
                rewards.append(r)
                dones.append(d)
                infos.append(info)

            return text_obs, image_obs, rewards, dones, infos

        def close(self):
            return

    return BatchGraphSearchEnv()
