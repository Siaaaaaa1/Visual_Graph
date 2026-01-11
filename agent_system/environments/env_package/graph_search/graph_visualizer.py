import io
import os
import json
import random
import hashlib
import networkx as nx
import matplotlib
# 强制使用非交互式后端，防止内存泄露和并发错误
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Set, Optional

class GraphVisualizer:
    @staticmethod
    def load_graph_data(dataset_name: str, dataset_dir: str) -> Tuple[Dict, Dict]:
        """
        静态方法：只加载和处理数据一次。
        返回: (graph_data, reverse_adj)
        """
        file_path = os.path.join(dataset_dir, f"{dataset_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        graph_data = {}
        reverse_adj = {}
        
        nodes_list = raw_data.get("nodes", [])
        for node in nodes_list:
            feature = node.get("feature", None)
            nid = str(node["id"])
            proxy = node.get("proxy_info", {})
            pred_class = proxy.get("top1") or "Unknown"
            neighbors = node.get("neighbors", [])
            
            graph_data[nid] = {
                "neighbors": neighbors,
                "pred_class": pred_class,
                "feature": feature
            }
            
            # 构建反向边
            for nb in neighbors:
                nb_str = str(nb)
                if nb_str not in reverse_adj:
                    reverse_adj[nb_str] = []
                reverse_adj[nb_str].append(int(nid))
                
        return graph_data, reverse_adj

    def __init__(self, 
                 dataset_name: str, 
                 dataset_dir: str = "./datasets", 
                 shared_data: Optional[Tuple[Dict, Dict]] = None):
        
        self.BASE_FIG_SIZE = 10
        self.NODE_SCALE_FACTOR = 0.05
        
        # 1. Tab10 (<=10 classes)
        self.cmap10 = plt.get_cmap("tab10")
        self.colors10 = self.cmap10.colors
        self.names10 = [
            "Blue", "Orange", "Green", "Red", "Purple",
            "Brown", "Pink", "Gray", "Olive", "Cyan"
        ]

        # 2. Tab20 (>10 classes)
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

    def _get_node_info(self, node_id: int) -> Dict:
        return self.graph_data.get(str(node_id), {"neighbors": [], "pred_class": "Unknown"})

    def _get_color_map_for_episode(self, active_classes: List[str], seed: int) -> Dict[str, Dict]:
        """
        核心逻辑：
        1. 使用 seed 随机打乱颜色列表（保证不同 episode 整体配色方案不同）。
        2. 使用 MD5 哈希将类名映射到颜色索引（保证同一 episode 内，Neural_Net 始终对应同一个颜色）。
        """
        num_classes = len(active_classes)
        
        # 选择色盘
        if num_classes <= 10:
            base_colors = list(zip(self.colors10, self.names10))
        else:
            base_colors = list(zip(self.colors20, self.names20))
            
        # ✅ 使用 seed 随机打乱色盘顺序 (Episode 级别的随机性)
        rng = random.Random(seed)
        shuffled_colors = list(base_colors)
        rng.shuffle(shuffled_colors)
        
        color_map = {}
        for cls_name in active_classes:
            # ✅ 使用稳定哈希确定索引 (Step 级别的稳定性)
            # 只要 shuffled_colors 不变 (即 seed 不变)，同一个 cls_name 永远拿到同一个颜色
            hash_val = int(hashlib.md5(cls_name.encode()).hexdigest(), 16)
            idx = hash_val % len(shuffled_colors)
            
            color_tuple, color_name = shuffled_colors[idx]
            color_map[cls_name] = {
                "color": color_tuple,
                "name": color_name
            }
        return color_map

    def _get_neighbors(self, node_id: int, undirected: bool):
        """根据 undirected 参数，返回邻居集合"""
        out_nbs = self._get_node_info(node_id)["neighbors"]
        if not undirected:
            return out_nbs

        # 无向：出边 + 入边
        in_nbs = self.reverse_adj.get(str(node_id), [])
        return list(set(out_nbs) | set(in_nbs))

    def _get_similarity(self, center_id: int, node_id: int) -> float:
        f1 = np.array(self.graph_data[str(center_id)]["feature"], dtype=np.float32)
        f2 = np.array(self.graph_data[str(node_id)]["feature"], dtype=np.float32)
        denom = (np.linalg.norm(f1) * np.linalg.norm(f2))
        if denom == 0:
            return 0.0
        return float(np.dot(f1, f2) / denom)


    def get_node_degree_info(self, node_id: int) -> Dict[str, int]:
        node_str = str(node_id)
        info = self._get_node_info(node_id)
        neighbors_1hop = info["neighbors"]
        out_degree = len(neighbors_1hop)
        
        in_neighbors = self.reverse_adj.get(node_str, [])
        in_degree = len(in_neighbors)
        
        undirected_1hop = set(neighbors_1hop) | set(in_neighbors)
        if node_id in undirected_1hop:
            undirected_1hop.remove(node_id)
            
        # 2-hop 计算
        undirected_2hop = set()
        for nb in undirected_1hop:
            nb_out = self.graph_data.get(str(nb), {}).get("neighbors", [])
            nb_in = self.reverse_adj.get(str(nb), [])
            nb_neighbors = set(nb_out) | set(nb_in)
            for nb2 in nb_neighbors:
                if nb2 != node_id and nb2 not in undirected_1hop:
                    undirected_2hop.add(nb2)
                    
        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "neighbor_count_1hop": len(undirected_1hop),
            "neighbor_count_2hop": len(undirected_2hop)
        }

    def draw_subgraph(
        self,
        center_id: int,
        mode: str = "1-hop",
        max_nodes: int = 50,
        color_seed: int = 42,  # ✅ 接收外部传入的 Seed
        rank_mode: str = "hop_then_sim", #sim_only or hop_then_sim
        undirected: bool = True
    ) -> Tuple[bytes, Dict[str, str]]:

        
        # 1. 节点筛选 (BFS)
        center_info = self._get_node_info(center_id)

        #neighbors_1hop = center_info["neighbors"]
        neighbors_1hop = self._get_neighbors(center_id, undirected)
        
        candidates = []
        if mode == "1-hop":
            candidates = neighbors_1hop
        elif "2-hop" in mode:
            candidates.extend(neighbors_1hop)
            # for nb in neighbors_1hop:
            #     nb_info = self._get_node_info(nb)
            #     candidates.extend(nb_info["neighbors"])
            for nb in neighbors_1hop:
                candidates.extend(self._get_neighbors(nb, undirected))
        
        # 去重、排除中心、截断
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen and c != center_id:
                seen.add(c)
                unique_candidates.append(c)

        one_hop_set = set(neighbors_1hop)

        sim_cache = {
            nid: self._get_similarity(center_id, nid)
            for nid in unique_candidates
        }

        if rank_mode == "sim_only":
            # 模式 A：完全按相似度排序
            unique_candidates = sorted(
                unique_candidates,
                key=lambda nid: sim_cache[nid],
                reverse=True
            )

        elif rank_mode == "hop_then_sim":
            # 模式 B：1-hop 优先，其次 2-hop；同一 hop 内按相似度
            one_hop = [n for n in unique_candidates if n in one_hop_set]
            two_hop = [n for n in unique_candidates if n not in one_hop_set]

            one_hop = sorted(
                one_hop,
                key=lambda nid: sim_cache[nid],
                reverse=True
            )
            two_hop = sorted(
                two_hop,
                key=lambda nid: sim_cache[nid],
                reverse=True
            )

            unique_candidates = one_hop + two_hop

        else:
            raise ValueError(f"Unknown rank_mode: {rank_mode}")
        
        budget = max(0, max_nodes - 1)
        final_nodes = [center_id] + unique_candidates[:budget]
        final_nodes_set = set(final_nodes)
        one_hop_set = set(u for u in one_hop_set if u in final_nodes_set)

        # =========================================================
        # 【STEP A】2-hop 强制树化：为每个 2-hop 选唯一 anchor（1-hop）
        # 插入位置：final_nodes 确定之后，建图之前
        # =========================================================

        two_hop_anchor = {}

        for nid in final_nodes:
            if nid == center_id:
                continue

            # 只处理 2-hop
            if nid in one_hop_set:
                continue

            # 找它连接的 1-hop（基于真实图结构）
            candidates = [
                nb for nb in self._get_neighbors(nid, undirected)
                if nb in one_hop_set
            ]

            if not candidates:
                continue

            # 用与你现有逻辑一致的规则：与 center 最相似的 1-hop
            anchor = max(
                candidates,
                key=lambda x: sim_cache.get(x, 0.0)
            )

            two_hop_anchor[nid] = anchor

        # =========================================================
        # 2. 建图（方案 A：严格树结构）
        # =========================================================

        # ① 初始化图（只做一次）
        G = nx.Graph()
        G.add_nodes_from(final_nodes)

        # ② 统计 active_classes（只做一次）
        active_classes = set()
        for u in final_nodes:
            if u != center_id:
                active_classes.add(
                    self._get_node_info(u)["pred_class"] or "Unknown"
                )

        # ③ center -> 1-hop
        for u in one_hop_set:
            if u in final_nodes_set:
                G.add_edge(center_id, u)

        # ④ 1-hop -> 自己的 2-hop（唯一 anchor）
        for child, anchor in two_hop_anchor.items():
            if child in final_nodes_set and anchor in final_nodes_set:
                G.add_edge(anchor, child)

        TWO_HOP_EDGE_WEIGHT = 0.2   # 很关键：一定要小

        for u in final_nodes:
            # 只处理 2-hop
            if u == center_id or u in one_hop_set:
                continue

            # 原图中的邻居
            for v in self._get_neighbors(u, undirected):
                if v == center_id:
                    continue
                if v not in final_nodes_set:
                    continue
                if v in one_hop_set:
                    continue          # 不要回连 1-hop
                if u >= v:
                    continue          # 防止重复加边

                # ✅ 只在“同一层的 2-hop 之间”加弱边
                G.add_edge(u, v, weight=TWO_HOP_EDGE_WEIGHT)


        # 3. 配色 (传入 seed)
        sorted_classes = sorted(list(active_classes))
        color_mapping = self._get_color_map_for_episode(sorted_classes, color_seed)

        # === 构造 spring_layout 的初始位置（只 bias 1-hop） ===
        pos_init = {}
        pos_init[center_id] = np.array([0.0, 0.0])

        rng = np.random.RandomState(42)

        # 1-hop：相似度 → 半径
        R_MIN, R_MAX = 0.6, 1.8   # 很保守的范围，不会炸
        for u in one_hop_set:
            sim = sim_cache.get(u, 0.0)
            sim = (sim + 1.0) / 2.0          # [-1,1] → [0,1]
            sim = np.clip(sim, 0.0, 1.0)

            # 相似度高 → 半径小
            r = R_MIN + (1.0 - sim) * (R_MAX - R_MIN)

            theta = rng.uniform(0, 2 * np.pi)
            pos_init[u] = np.array([
                r * np.cos(theta),
                r * np.sin(theta)
            ])

        pos = nx.spring_layout(
            G,
            weight="weight",
            pos=pos_init,     # ✅ 关键：传入初始位置
            seed=42,
            iterations=100
        )

        center_xy = pos[center_id].copy()
        for k in pos:
            pos[k] = pos[k] - center_xy

        edgecolors = []
        sizes = []
        legend_dict = {} 
        
        for nid in final_nodes:
            if nid == center_id:
                edgecolors.append("black")
                sizes.append(1500)
                legend_dict["Black"] = "Center Node"
            else:
                pred_class = self._get_node_info(nid)["pred_class"]
                c_conf = color_mapping.get(pred_class, {"color": "gray", "name": "Gray"})
                edgecolors.append(c_conf["color"])
                sizes.append(1000)
                
                c_name = c_conf["name"]
                if c_name not in legend_dict:
                    legend_dict[c_name] = pred_class

        # 4. 绘图
        fig_size = self.BASE_FIG_SIZE + len(final_nodes) * self.NODE_SCALE_FACTOR
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


        all_pos = np.array(list(pos.values()))

        xmin, ymin = all_pos.min(axis=0)
        xmax, ymax = all_pos.max(axis=0)

        pad = 0.05   # 少量留白，防止节点/文字被裁
        dx = xmax - xmin
        dy = ymax - ymin

        # 防止极端情况下 dx / dy 为 0
        if dx == 0:
            dx = 1e-3
        if dy == 0:
            dy = 1e-3

        plt.xlim(xmin - pad * dx, xmax + pad * dx)
        plt.ylim(ymin - pad * dy, ymax + pad * dy)


        # 5. 保存
        buf = io.BytesIO()
        #plt.savefig(buf, format="png", bbox_inches="tight")
        #plt.savefig(buf, format="png")
        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0
        )
        
        plt.close(fig)
        
        buf.seek(0)
        image_bytes = buf.getvalue()

        return image_bytes, legend_dict