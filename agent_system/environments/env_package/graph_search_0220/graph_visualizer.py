import io
import os
import json
import random
import hashlib
import networkx as nx
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Set, Optional

class GraphVisualizer:
    @staticmethod
    def load_graph_data(dataset_name: str, dataset_dir: str) -> Tuple[Dict, Dict, Dict]:
        file_path = os.path.join(dataset_dir, f"{dataset_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        graph_data = {}
        reverse_adj = {}
        
        nodes_list = raw_data.get("nodes", [])
        class_map = raw_data.get("class_map", {}) 
        
        for node in nodes_list:
            feature = node.get("feature", None)
            nid = str(node["id"])
            proxy = node.get("proxy_info", {})
            pred_class = proxy.get("top1") or "Unknown"
            # [致命Bug修复]：你的原代码忘记提取 true_class 了，我在这里补上，防止最后结算断言报错
            true_class = node.get("label", pred_class)
            neighbors = node.get("neighbors", [])
            
            graph_data[nid] = {
                "neighbors": neighbors,
                "pred_class": pred_class,
                "true_class": true_class, 
                "feature": feature
            }
            
            for nb in neighbors:
                nb_str = str(nb)
                if nb_str not in reverse_adj:
                    reverse_adj[nb_str] = []
                reverse_adj[nb_str].append(int(nid))
                
        return graph_data, reverse_adj, class_map

    def __init__(self, 
                 dataset_name: str, 
                 dataset_dir: str = "./datasets", 
                 shared_data: Optional[Tuple[Dict, Dict, Dict, Any]] = None):
        
        self.BASE_FIG_SIZE = 10
        self.NODE_SCALE_FACTOR = 0.05
        
        self.cmap10 = plt.get_cmap("tab10")
        self.colors10 = self.cmap10.colors
        self.names10 = ["Blue", "Orange", "Green", "Red", "Purple", "Brown", "Pink", "Gray", "Olive", "Cyan"]

        self.cmap20 = plt.get_cmap("tab20")
        self.colors20 = self.cmap20.colors
        self.names20 = [
            "Blue", "LightBlue", "Orange", "LightOrange", "Green", "LightGreen", "Red", "LightRed",
            "Purple", "LightPurple", "Brown", "LightBrown", "Pink", "LightPink", "Gray", "LightGray",
            "Olive", "LightOlive", "Cyan", "LightCyan"
        ]
        
        if shared_data is not None:
            if len(shared_data) == 4:
                self.graph_data, self.reverse_adj, self.class_map, matrix_candidate = shared_data
                if matrix_candidate is not None:
                    self.feat_matrix = matrix_candidate
                else:
                    self.feat_matrix = self._build_feat_matrix()
            else:
                self.graph_data, self.reverse_adj, self.class_map = shared_data[:3]
                self.feat_matrix = self._build_feat_matrix()
        else:
            self.graph_data, self.reverse_adj, self.class_map = self.load_graph_data(dataset_name, dataset_dir)
            self.feat_matrix = self._build_feat_matrix()
        
        self.all_node_ids = list(self.graph_data.keys())
        self.id_to_idx = {nid: i for i, nid in enumerate(self.all_node_ids)}

    def _build_feat_matrix(self):
        print("Building feature matrix (Process Local)...")
        all_ids = list(self.graph_data.keys())
        features_list = []
        for nid in all_ids:
            raw_feat = self.graph_data[nid]["feature"]
            arr = np.array(raw_feat, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 1e-9: arr = arr / norm
            else: arr = np.zeros_like(arr)
            features_list.append(arr)
        return np.stack(features_list)

    def _get_node_info(self, node_id: int) -> Dict:
        return self.graph_data.get(str(node_id), {"neighbors": [], "pred_class": "Unknown", "true_class": "Unknown"})

    def _get_color_map_for_episode(self, active_classes: List[str], seed: int) -> Dict[str, Dict]:
        num_classes = len(active_classes)
        base_colors = list(zip(self.colors10, self.names10)) if num_classes <= 10 else list(zip(self.colors20, self.names20))
        rng = random.Random(seed)
        shuffled_colors = list(base_colors)
        rng.shuffle(shuffled_colors)
        
        color_map = {}
        for cls_name in active_classes:
            hash_val = int(hashlib.md5(cls_name.encode()).hexdigest(), 16)
            idx = hash_val % len(shuffled_colors)
            color_tuple, color_name = shuffled_colors[idx]
            color_map[cls_name] = {"color": color_tuple, "name": color_name}
        return color_map

    def _get_neighbors(self, node_id: int, undirected: bool):
        out_nbs = self._get_node_info(node_id)["neighbors"]
        if not undirected: return out_nbs
        in_nbs = self.reverse_adj.get(str(node_id), [])
        return list(set(out_nbs) | set(in_nbs))

    def _get_similarity(self, center_id: int, node_id: int) -> float:
        c_str, n_str = str(center_id), str(node_id)
        if c_str in self.id_to_idx and n_str in self.id_to_idx:
            idx1 = self.id_to_idx[c_str]
            idx2 = self.id_to_idx[n_str]
            return float(np.dot(self.feat_matrix[idx1], self.feat_matrix[idx2]))
        
        f1 = np.array(self.graph_data[str(center_id)]["feature"], dtype=np.float32)
        f2 = np.array(self.graph_data[str(node_id)]["feature"], dtype=np.float32)
        denom = (np.linalg.norm(f1) * np.linalg.norm(f2))
        return float(np.dot(f1, f2) / denom) if denom != 0 else 0.0

    def get_candidate_classes(self, center_id: int, top_k: int = 100) -> List[str]:
        candidates = set()
        queue = [center_id]
        seen = {center_id}
        while queue and len(candidates) < top_k:
            curr = queue.pop(0)
            nbs = self._get_neighbors(curr, undirected=True)
            for nb in nbs:
                if nb not in seen:
                    seen.add(nb)
                    candidates.add(nb)
                    queue.append(nb)
                    if len(candidates) >= top_k: break
        
        c_str = str(center_id)
        if c_str in self.id_to_idx:
            c_idx = self.id_to_idx[c_str]
            sim_scores = self.feat_matrix @ self.feat_matrix[c_idx]
            sim_scores[c_idx] = -10.0 
            check_k = min(20, len(sim_scores))
            top_k_indices = np.argpartition(sim_scores, -check_k)[-check_k:]
            for idx in top_k_indices: candidates.add(int(self.all_node_ids[idx]))

        classes = {self._get_node_info(nid)["pred_class"] for nid in candidates}
        center_pred = self._get_node_info(center_id)["pred_class"]
        if center_pred: classes.add(center_pred)
        return sorted(list(classes))

    def get_node_degree_info(self, node_id: int) -> Dict[str, int]:
        node_str = str(node_id)
        info = self._get_node_info(node_id)
        neighbors_1hop = info["neighbors"]
        in_neighbors = self.reverse_adj.get(node_str, [])
        undirected_1hop = set(neighbors_1hop) | set(in_neighbors)
        if node_id in undirected_1hop: undirected_1hop.remove(node_id)
        return {
            "in_degree": len(in_neighbors),
            "out_degree": len(neighbors_1hop),
            "neighbor_count_1hop": len(undirected_1hop)
        }

    # ========================================================
    # 【修改】：严格遵从 1-hop -> 2-hop -> sim 顺序截断 30 个点
    # ========================================================
    def _select_nodes(self, center_id: int, view_mode: str, max_nodes: int, undirected: bool) -> Tuple[List[int], Dict[int, Dict]]:
        if view_mode == "center":
            return [center_id], {center_id: {'hop': 0, 'sim': 1.0}}

        def get_sim(nid):
            return self._get_similarity(center_id, nid)

        # 1-hop
        hop1 = set(self._get_neighbors(center_id, undirected))
        if center_id in hop1: hop1.remove(center_id)

        # 2-hop
        hop2 = set()
        if "2-hop" in view_mode:
            for n in hop1:
                hop2.update(self._get_neighbors(n, undirected))
            hop2 -= hop1
            if center_id in hop2: hop2.remove(center_id)

        # sim
        sim_nodes = set()
        if "sim" in view_mode and len(hop1) + len(hop2) < max_nodes - 1:
            c_str = str(center_id)
            if c_str in self.id_to_idx:
                c_idx = self.id_to_idx[c_str]
                sim_scores = self.feat_matrix @ self.feat_matrix[c_idx]
                top_indices = np.argsort(sim_scores)[::-1]
                for idx in top_indices:
                    nid = int(self.all_node_ids[idx])
                    if nid != center_id and nid not in hop1 and nid not in hop2:
                        sim_nodes.add(nid)
                        if len(hop1) + len(hop2) + len(sim_nodes) >= max_nodes - 1:
                            break

        # 优先级：跳数升序，相似度降序
        pool = []
        for n in hop1: pool.append((n, 1, get_sim(n)))
        for n in hop2: pool.append((n, 2, get_sim(n)))
        for n in sim_nodes: pool.append((n, 3, get_sim(n)))
        pool.sort(key=lambda x: (x[1], -x[2]))
        
        selected_pool = pool[:max_nodes - 1]
        final_nodes = [center_id] + [x[0] for x in selected_pool]
        
        node_meta = {center_id: {'hop': 0, 'sim': 1.0}}
        for n, h, s in selected_pool:
            node_meta[n] = {'hop': h, 'sim': s}
            
        return final_nodes, node_meta

    # ========================================================
    # 【新增】：4个上限与跳数裁决
    # ========================================================
    def _assign_node_shapes(self, nodes_to_draw: List[int], center_id: int, node_meta: Dict) -> Dict[int, str]:
        candidates = [n for n in nodes_to_draw if n != center_id]
        metrics = []
        for n in candidates:
            deg = self.get_node_degree_info(n)
            metrics.append({
                'id': n,
                'max_deg': max(deg['out_degree'], deg['in_degree']),
                'out_degree': deg['out_degree'],
                'in_degree': deg['in_degree'],
                'hop': node_meta[n]['hop']
            })
            
        # 仲裁规则：最大度数降序，同度数则跳数升序
        metrics.sort(key=lambda x: (-x['max_deg'], x['hop']))
        
        shapes = {}
        hub_count = 0
        for m in metrics:
            if hub_count < 4 and m['max_deg'] > 1:
                if m['out_degree'] > m['in_degree']: shapes[m['id']] = 'high_out'
                else: shapes[m['id']] = 'high_in'
                hub_count += 1
            else:
                shapes[m['id']] = '1-hop'
        return shapes

    def draw_subgraph(
        self,
        center_id: int,
        view_mode: str = "1-hop",
        max_nodes: int = 10,
        color_seed: int = 42,
        undirected: bool = True,
        mask_neighbors: bool = False,
        painted_nodes: Optional[Dict[int, str]] = None 
    ) -> Tuple[bytes, Dict[str, str], Dict[int, str]]:
        
        if painted_nodes is None: painted_nodes = {}
        final_nodes, node_meta = self._select_nodes(center_id, view_mode, max_nodes, undirected)
        
        G = nx.Graph()
        G.add_nodes_from(final_nodes)

        # ========================================================
        # 【修改】：去除你的 edge sparsification，只画图里真实存在的拓扑跳数边
        # ========================================================
        edges = set()
        final_nodes_set = set(final_nodes)
        for u in final_nodes:
            u_nbs = self._get_neighbors(u, undirected)
            for v in u_nbs:
                if v in final_nodes_set and u < v:
                    edges.add((u, v))
        G.add_edges_from(list(edges))
        
        nodes_to_draw = list(G.nodes())
        active_classes = set()
        for u in nodes_to_draw:
            if u != center_id:
                active_classes.add(self._get_node_info(u)["pred_class"] or "Unknown")
                if u in painted_nodes:
                    active_classes.add(painted_nodes[u])

        color_mapping = self._get_color_map_for_episode(sorted(list(active_classes)), color_seed)

        # ---------------- 你的优秀物理排版引擎保留 ----------------
        pos_init = {center_id: np.array([0.0, 0.0])}
        rng = np.random.RandomState(42)
        sim_cache = {u: node_meta[u]['sim'] for u in nodes_to_draw if u != center_id}
        R_MIN, R_MAX = 0.5, 2.0

        for u in nodes_to_draw:
            if u == center_id: continue
            sim = sim_cache.get(u, 0.0)
            dist = R_MIN + (1.0 - sim) * (R_MAX - R_MIN)
            theta = rng.uniform(0, 2 * np.pi)
            pos_init[u] = np.array([dist * np.cos(theta), dist * np.sin(theta)])

        n = len(nodes_to_draw)
        k_val = max(0.35, 1.0 / np.sqrt(n)) if n > 0 else 1.0

        DISCONNECTED_OK = {"sim", "1-hop+sim", "2-hop+sim"}
        if view_mode in DISCONNECTED_OK:
            pos = {center_id: np.array([0.0, 0.0], dtype=np.float32)}
            golden = 2.399963229728653 
            nodes_others = [nid for nid in nodes_to_draw if nid != center_id]
            for idx, u in enumerate(nodes_others):
                sim = float(sim_cache.get(u, 0.0))
                # 越相似的 sim 节点拉得越近
                dist = R_MIN + (1.0 - sim) * (R_MAX - R_MIN)
                theta = (idx + 1) * golden
                pos[u] = np.array([dist * np.cos(theta), dist * np.sin(theta)], dtype=np.float32)
        else:
            pos = nx.spring_layout(G, pos=pos_init, seed=42, k=k_val, iterations=80, fixed=[center_id])

        if center_id in pos:
            c_xy = pos[center_id]
            for node in pos: pos[node] = pos[node] - c_xy

        # 防碰撞循环
        MIN_R = 0.5 
        for node, xy in pos.items():
            if node == center_id: continue
            r = float(np.linalg.norm(xy))
            if r < MIN_R:
                if r < 1e-8:
                    theta = rng.uniform(0, 2 * np.pi)
                    xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                    r = 1.0
                pos[node] = xy * (MIN_R / r)
        
        MIN_NODE_DIST = 0.22 
        COLLISION_ITERS = 30
        nodes_others = [n for n in nodes_to_draw if n != center_id]

        for _ in range(COLLISION_ITERS):
            moved = False
            for i in range(len(nodes_others)):
                for j in range(i + 1, len(nodes_others)):
                    a, b = nodes_others[i], nodes_others[j]
                    va, vb = pos[a], pos[b]
                    delta = va - vb
                    dist = float(np.linalg.norm(delta))
                    if dist < 1e-8:
                        theta = rng.uniform(0, 2 * np.pi)
                        delta = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                        dist = 1e-4
                    if dist < MIN_NODE_DIST:
                        push = (MIN_NODE_DIST - dist) / dist * 0.5
                        shift = delta * push
                        pos[a] = va + shift
                        pos[b] = vb - shift
                        moved = True
            for node in nodes_others:
                xy = pos[node]
                r = float(np.linalg.norm(xy))
                if r < MIN_R:
                    if r < 1e-8:
                        theta = rng.uniform(0, 2 * np.pi)
                        xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                        r = 1.0
                    pos[node] = xy * (MIN_R / r)
            if not moved: break
        # -------------------------------------------------------------

        edgecolors = []
        sizes = []
        legend_dict = {}
        COLOR_CENTER = "black"
        COLOR_MASKED = "black" 

        for nid in nodes_to_draw:
            if nid == center_id:
                edgecolors.append(COLOR_CENTER)
                sizes.append(1500)
                legend_dict["Black"] = "Center Node"
            else:
                if mask_neighbors:
                    if nid in painted_nodes:
                        pred_class = painted_nodes[nid]
                        c_conf = color_mapping.get(pred_class, {"color": "gray", "name": "Gray"})
                        edgecolors.append(c_conf["color"])
                        sizes.append(1000)
                        legend_dict[c_conf["name"]] = f"Painted: {pred_class}"
                    else:
                        edgecolors.append(COLOR_MASKED)
                        sizes.append(800)
                        if "Masked Nodes" not in legend_dict:
                             legend_dict["Masked Nodes"] = "Unlabeled Neighbors (Fog)"
                else:
                    pred_class = self._get_node_info(nid)["pred_class"]
                    c_conf = color_mapping.get(pred_class, {"color": "gray", "name": "Gray"})
                    edgecolors.append(c_conf["color"])
                    sizes.append(800)
                    c_name = c_conf["name"]
                    if c_name not in legend_dict:
                        legend_dict[c_name] = pred_class

        # 获取形状分配
        node_shapes = self._assign_node_shapes(final_nodes, center_id, node_meta)
        nodes_1hop = [n for n, s in node_shapes.items() if s == '1-hop']
        nodes_high_out = [n for n, s in node_shapes.items() if s == 'high_out']
        nodes_high_in = [n for n, s in node_shapes.items() if s == 'high_in']

        fig_size = self.BASE_FIG_SIZE + (len(nodes_to_draw) * 0.02)
        fig = plt.figure(figsize=(fig_size, fig_size))

        other_edges = [(u, v) for (u, v) in G.edges() if (u != center_id and v != center_id)]
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, alpha=0.5, width=1.6, edge_color="black")
        center_edges = [(u, v) for (u, v) in G.edges() if (u == center_id or v == center_id)]
        nx.draw_networkx_edges(G, pos, edgelist=center_edges, alpha=1, width=1.8, edge_color="black")

        # ========================================================
        # 【修改】：按照不同形状分别绘制节点
        # ========================================================
        def get_styles(nlist):
             return [edgecolors[nodes_to_draw.index(n)] for n in nlist], [sizes[nodes_to_draw.index(n)] for n in nlist]

        c_c, c_s = get_styles([center_id])
        nx.draw_networkx_nodes(G, pos, nodelist=[center_id], node_color="white", edgecolors=c_c, linewidths=3.0, node_size=c_s, node_shape="s")
        
        if nodes_1hop:
            c_e, c_s = get_styles(nodes_1hop)
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_1hop, node_color="white", edgecolors=c_e, linewidths=3.0, node_size=c_s, node_shape="o")
            
        if nodes_high_out:
            c_e, c_s = get_styles(nodes_high_out)
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_high_out, node_color="white", edgecolors=c_e, linewidths=3.0, node_size=c_s, node_shape="v")
            
        if nodes_high_in:
            c_e, c_s = get_styles(nodes_high_in)
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_high_in, node_color="white", edgecolors=c_e, linewidths=3.0, node_size=c_s, node_shape="^")

        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.axis("off")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        
        buf.seek(0)
        return buf.getvalue(), legend_dict, node_shapes