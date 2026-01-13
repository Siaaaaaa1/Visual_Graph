import io
import os
import json
import random
import hashlib
import networkx as nx
import matplotlib
# 非交互后端，适合服务器端批量渲染
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Set, Optional


class GraphVisualizer:
    @staticmethod
    def load_graph_data(dataset_name: str, dataset_dir: str) -> Tuple[Dict, Dict]:
        file_path = os.path.join(dataset_dir, f"{dataset_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        graph_data = {}
        reverse_adj = {}

        nodes_list = raw_data.get("nodes", [])
        for node in nodes_list:
            nid = str(node["id"])
            proxy = node.get("proxy_info", {})
            pred_class = proxy.get("top1") or "Unknown"

            graph_data[nid] = {
                "neighbors": node.get("neighbors", []),
                "pred_class": pred_class,
                "feature": node.get("feature", None),
            }

            for nb in node.get("neighbors", []):
                reverse_adj.setdefault(str(nb), []).append(int(nid))

        return graph_data, reverse_adj

    def __init__(
        self,
        dataset_name: str,
        dataset_dir: str = "./datasets",
        shared_data: Optional[Tuple[Dict, Dict]] = None,
    ):
        # === 原始参数（保持不变） ===
        self.BASE_FIG_SIZE = 10
        self.NODE_SCALE_FACTOR = 0.05

        # 色板
        self.cmap10 = plt.get_cmap("tab10")
        self.colors10 = self.cmap10.colors
        self.names10 = [
            "Blue", "Orange", "Green", "Red", "Purple",
            "Brown", "Pink", "Gray", "Olive", "Cyan"
        ]

        self.cmap20 = plt.get_cmap("tab20")
        self.colors20 = self.cmap20.colors
        self.names20 = [
            "Blue", "LightBlue", "Orange", "LightOrange",
            "Green", "LightGreen", "Red", "LightRed",
            "Purple", "LightPurple", "Brown", "LightBrown",
            "Pink", "LightPink", "Gray", "LightGray",
            "Olive", "LightOlive", "Cyan", "LightCyan"
        ]

        if shared_data is not None:
            self.graph_data, self.reverse_adj = shared_data
        else:
            self.graph_data, self.reverse_adj = self.load_graph_data(dataset_name, dataset_dir)

    # ============================================================
    # ★ 像素预算函数（稍微放松版）
    # ============================================================

    def _compute_fig_size(self, N: int) -> float:
        return float(
            np.clip(
                1.2
                + 0.18 * N
                + 0.015 * N * N
                + 0.6 * np.log1p(N),   
                1.8,
                11.0                  
            )
        )
        # ------------------------------------------------------------

    def _get_node_info(self, node_id: int) -> Dict:
        return self.graph_data.get(str(node_id), {"neighbors": [], "pred_class": "Unknown"})

    def _get_neighbors(self, node_id: int, undirected: bool):
        out_nbs = self._get_node_info(node_id)["neighbors"]
        if not undirected:
            return out_nbs
        in_nbs = self.reverse_adj.get(str(node_id), [])
        return list(set(out_nbs) | set(in_nbs))

    def _get_similarity(self, center_id: int, node_id: int) -> float:
        f1 = np.array(self.graph_data[str(center_id)]["feature"], dtype=np.float32)
        f2 = np.array(self.graph_data[str(node_id)]["feature"], dtype=np.float32)
        denom = np.linalg.norm(f1) * np.linalg.norm(f2)
        if denom == 0:
            return 0.0
        return float(np.dot(f1, f2) / denom)

    def _get_color_map_for_episode(self, active_classes: List[str], seed: int):
        if len(active_classes) <= 10:
            base = list(zip(self.colors10, self.names10))
        else:
            base = list(zip(self.colors20, self.names20))

        rng = random.Random(seed)
        rng.shuffle(base)

        cmap = {}
        for cls in active_classes:
            idx = int(hashlib.md5(cls.encode()).hexdigest(), 16) % len(base)
            color, name = base[idx]
            cmap[cls] = {"color": color, "name": name}
        return cmap

    # ------------------------------------------------------------

    def draw_subgraph(
        self,
        center_id: int,
        mode: str = "1-hop",
        max_nodes: int = 50,
        color_seed: int = 42,
        rank_mode: str = "hop",
        undirected: bool = True,
    ) -> Tuple[bytes, Dict[str, str]]:

        # ---------- 节点候选 ----------
        neighbors_1hop = self._get_neighbors(center_id, undirected)
        candidates = list(neighbors_1hop)

        if "2-hop" in mode:
            for nb in neighbors_1hop:
                candidates.extend(self._get_neighbors(nb, undirected))

        unique = []
        seen = set()
        for n in candidates:
            if n != center_id and n not in seen:
                unique.append(n)
                seen.add(n)

        sim = {n: self._get_similarity(center_id, n) for n in unique}

        if rank_mode == "sim":
            unique.sort(key=lambda n: sim[n], reverse=True)
        else:
            one = [n for n in unique if n in neighbors_1hop]
            two = [n for n in unique if n not in neighbors_1hop]
            one.sort(key=lambda n: sim[n], reverse=True)
            two.sort(key=lambda n: sim[n], reverse=True)
            unique = one + two

        final_nodes = [center_id] + unique[: max_nodes - 1]
        final_set = set(final_nodes)
        one_hop_set = set(neighbors_1hop) & final_set

        # ---------- 2-hop 树化 ----------
        two_hop_anchor = {}
        for n in final_nodes:
            if n == center_id or n in one_hop_set:
                continue
            parents = [p for p in self._get_neighbors(n, undirected) if p in one_hop_set]
            if parents:
                two_hop_anchor[n] = max(parents, key=lambda x: sim.get(x, 0.0))

        # ---------- 建图 ----------
        G = nx.Graph()
        G.add_nodes_from(final_nodes)

        for n in one_hop_set:
            G.add_edge(center_id, n)

        for c, p in two_hop_anchor.items():
            G.add_edge(p, c)

        # ---------- 布局（不使用 k） ----------
        pos_init = {center_id: np.array([0.0, 0.0])}
        rng = np.random.RandomState(42)

        for n in one_hop_set:
            s = (sim[n] + 1) / 2
            r = 0.6 + (1 - s) * 1.2
            theta = rng.uniform(0, 2 * np.pi)
            pos_init[n] = np.array([r * np.cos(theta), r * np.sin(theta)])

        pos = nx.spring_layout(
            G,
            pos=pos_init,
            seed=42,
            iterations=100
        )

        center_xy = pos[center_id].copy()
        for k in pos:
            pos[k] -= center_xy

        # ---------- 颜色 ----------
        active_classes = set()
        for n in final_nodes:
            if n != center_id:
                active_classes.add(self._get_node_info(n)["pred_class"])

        color_map = self._get_color_map_for_episode(list(active_classes), color_seed)

        edgecolors = []
        sizes = []
        legend = {}

        for n in final_nodes:
            if n == center_id:
                edgecolors.append("black")
                sizes.append(1500)
                legend["Black"] = "Center Node"
            else:
                cls = self._get_node_info(n)["pred_class"]
                conf = color_map.get(cls, {"color": "gray", "name": "Gray"})
                edgecolors.append(conf["color"])
                sizes.append(1000)
                legend.setdefault(conf["name"], cls)

        # ============================================================
        # ★ 像素大小仅由节点数决定
        # ============================================================
        N = len(final_nodes)
        fig_size = self._compute_fig_size(N)
        fig = plt.figure(figsize=(fig_size, fig_size))

        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_nodes(
            G, pos,
            node_color="white",
            edgecolors=edgecolors,
            linewidths=3.5,
            node_size=sizes
        )
        nx.draw_networkx_labels(G, pos, font_size=9)

        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        return buf.getvalue(), legend
