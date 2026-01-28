import io
import os
import json
import random
import hashlib
import networkx as nx
import matplotlib
# 强制使用非交互式后端
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Set, Optional

class GraphVisualizer:
    @staticmethod
    def load_graph_data(dataset_name: str, dataset_dir: str) -> Tuple[Dict, Dict, Dict]:
        """
        加载图数据 (采用文件2的逻辑，包含了 class_map)
        """
        file_path = os.path.join(dataset_dir, f"{dataset_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        graph_data = {}
        reverse_adj = {}
        
        nodes_list = raw_data.get("nodes", [])
        class_map = raw_data.get("class_map", {}) # 文件2特有
        
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
            
            for nb in neighbors:
                nb_str = str(nb)
                if nb_str not in reverse_adj:
                    reverse_adj[nb_str] = []
                reverse_adj[nb_str].append(int(nid))
                
        return graph_data, reverse_adj, class_map

    def __init__(self, 
                 dataset_name: str, 
                 dataset_dir: str = "./datasets", 
                 shared_data: Optional[Tuple[Dict, Dict, Dict]] = None):
        
        self.BASE_FIG_SIZE = 10
        self.NODE_SCALE_FACTOR = 0.05
        
        # 1. 颜色配置 (共用)
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
        
        # 2. 数据加载
        if shared_data is not None:
            self.graph_data, self.reverse_adj, self.class_map = shared_data
        else:
            self.graph_data, self.reverse_adj, self.class_map = self.load_graph_data(dataset_name, dataset_dir)
        
        self.all_node_ids = list(self.graph_data.keys())
        
        # 3. 构建特征矩阵 (文件2的核心优化逻辑)
        # 为了支持 _select_nodes 中的向量化计算，这里预处理 feature matrix
        print("Building feature matrix for vectorized similarity...")
        self.id_to_idx = {nid: i for i, nid in enumerate(self.all_node_ids)}
        
        # 预先计算 L2 归一化的特征矩阵
        features_list = []
        for nid in self.all_node_ids:
            raw_feat = self.graph_data[nid]["feature"]
            arr = np.array(raw_feat, dtype=np.float32)
            norm = np.linalg.norm(arr)
            # 避免除零
            if norm > 1e-9:
                arr = arr / norm
            else:
                arr = np.zeros_like(arr)
            features_list.append(arr)
            
        self.feat_matrix = np.stack(features_list) # Shape: [N, D]

    def _get_node_info(self, node_id: int) -> Dict:
        return self.graph_data.get(str(node_id), {"neighbors": [], "pred_class": "Unknown"})

    def _get_color_map_for_episode(self, active_classes: List[str], seed: int) -> Dict[str, Dict]:
        num_classes = len(active_classes)
        if num_classes <= 10:
            base_colors = list(zip(self.colors10, self.names10))
        else:
            base_colors = list(zip(self.colors20, self.names20))
            
        rng = random.Random(seed)
        shuffled_colors = list(base_colors)
        rng.shuffle(shuffled_colors)
        
        color_map = {}
        for cls_name in active_classes:
            hash_val = int(hashlib.md5(cls_name.encode()).hexdigest(), 16)
            idx = hash_val % len(shuffled_colors)
            color_tuple, color_name = shuffled_colors[idx]
            color_map[cls_name] = {
                "color": color_tuple,
                "name": color_name
            }
        return color_map

    def _get_neighbors(self, node_id: int, undirected: bool):
        out_nbs = self._get_node_info(node_id)["neighbors"]
        if not undirected:
            return out_nbs
        in_nbs = self.reverse_adj.get(str(node_id), [])
        return list(set(out_nbs) | set(in_nbs))

    def _get_similarity(self, center_id: int, node_id: int) -> float:
        """
        计算单点相似度。
        虽然有矩阵，但在处理单个边权重时（draw_subgraph中），
        直接用原始数据点积有时比查索引再切片更方便，
        或者为了保持接口兼容性，保留此函数。
        """
        # 注意：因为 __init__ 里已经做了归一化矩阵，如果想极速，可以用矩阵查。
        # 但为了稳健性（防止 id 找不到 idx），这里保留原始的点积逻辑，
        # 且复用 __init__ 里的归一化思路。
        
        # 简单直接计算 (利用已归一化的矩阵加速)
        c_str, n_str = str(center_id), str(node_id)
        if c_str in self.id_to_idx and n_str in self.id_to_idx:
            idx1 = self.id_to_idx[c_str]
            idx2 = self.id_to_idx[n_str]
            return float(np.dot(self.feat_matrix[idx1], self.feat_matrix[idx2]))
        
        # Fallback (如果没有构建矩阵)
        f1 = np.array(self.graph_data[str(center_id)]["feature"], dtype=np.float32)
        f2 = np.array(self.graph_data[str(node_id)]["feature"], dtype=np.float32)
        denom = (np.linalg.norm(f1) * np.linalg.norm(f2))
        if denom == 0:
            return 0.0
        return float(np.dot(f1, f2) / denom)

    def get_candidate_classes(self, center_id: int, top_k: int = 100) -> List[str]:
        """
        (采用文件2的优化逻辑)
        除了 BFS 邻居，还加入 Top-20 全局相似节点的类别。
        """
        # 1. BFS 邻居
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
                    if len(candidates) >= top_k:
                        break
        
        # 2. 增加 Top-20 全局相似 (利用矩阵加速)
        c_str = str(center_id)
        if c_str in self.id_to_idx:
            c_idx = self.id_to_idx[c_str]
            # 计算全量相似度
            sim_scores = self.feat_matrix @ self.feat_matrix[c_idx]
            sim_scores[c_idx] = -10.0 # 排除自己
            
            # 取 Top 20
            check_k = min(20, len(sim_scores))
            top_k_indices = np.argpartition(sim_scores, -check_k)[-check_k:]
            
            for idx in top_k_indices:
                candidates.add(int(self.all_node_ids[idx]))

        classes = set()
        for nid in candidates:
            cls = self._get_node_info(nid)["pred_class"]
            classes.add(cls)
            
        # 确保中心类也在
        center_pred = self._get_node_info(center_id)["pred_class"]
        if center_pred:
            classes.add(center_pred)
            
        return sorted(list(classes))

    def get_all_candidate_classes(self) -> List[str]:
        return list(self.class_map.values())
    
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

    def _select_nodes(self, center_id: int, view_mode: str, max_nodes: int, undirected: bool) -> List[int]:
        """
        (采用文件2的优化逻辑)
        利用矩阵运算 (feat_matrix) 快速进行节点筛选，替代 Python 循环。
        """
        if view_mode == "center":
            return [center_id]
        
        neighbors_1hop = set(self._get_neighbors(center_id, undirected))
        if center_id in neighbors_1hop: neighbors_1hop.remove(center_id)
        
        neighbors_2hop = set()
        if "2-hop" in view_mode:
            for nb in neighbors_1hop:
                nbs = self._get_neighbors(nb, undirected)
                for nb2 in nbs:
                    if nb2 != center_id and nb2 not in neighbors_1hop:
                        neighbors_2hop.add(nb2)

        selected = []
        budget = max_nodes
        
        # 准备矩阵索引
        c_str = str(center_id)
        if c_str not in self.id_to_idx:
            return [center_id] + list(neighbors_1hop)[:budget] # Fallback
        c_idx = self.id_to_idx[c_str]

        # 计算所有节点到中心的相似度 (向量化)
        # sim_scores: shape [N]
        sim_scores = self.feat_matrix @ self.feat_matrix[c_idx]

        if view_mode in ["1-hop", "2-hop", "1-hop+sim", "2-hop+sim"]:
            pool = list(neighbors_1hop)
            if "2-hop" in view_mode:
                pool.extend(list(neighbors_2hop))
            
            # 对 pool 里的节点按相似度排序
            # 使用列表推导 + sort，因为 pool 通常不大，这样做比全量 argpartition 更灵活
            pool_scores = []
            for nid in pool:
                n_str = str(nid)
                if n_str in self.id_to_idx:
                    s = sim_scores[self.id_to_idx[n_str]]
                else:
                    s = -1.0
                pool_scores.append((s, nid))
            
            pool_scores.sort(key=lambda x: x[0], reverse=True)
            sorted_pool = [x[1] for x in pool_scores]
            
            if view_mode in ["1-hop", "2-hop"]:
                selected = sorted_pool[:budget]
            else: 
                # +sim 模式
                if len(sorted_pool) >= budget:
                    selected = sorted_pool[:budget]
                else:
                    selected = sorted_pool[:]
                    remain = budget - len(selected)
                    
                    if remain > 0:
                        # 全局补全
                        # 排除已选中的、中心点、以及 pool 中的点
                        exclude_indices = {self.id_to_idx[str(n)] for n in selected if str(n) in self.id_to_idx}
                        exclude_indices.add(c_idx)
                        
                        # 把要排除的分数设极低
                        # 这里为了不破坏原始 sim_scores (虽然下面重新计算也没事)，拷贝一份 mask
                        temp_scores = sim_scores.copy()
                        # 需要排除的点位置设为 -10
                        # 注意：这里需要先把 indices 转为 list
                        temp_scores[list(exclude_indices)] = -10.0
                        
                        # Top-K
                        # argpartition 找到最大的 remain 个
                        top_k_indices = np.argpartition(temp_scores, -remain)[-remain:]
                        # 排序这 remain 个
                        top_k_indices = top_k_indices[np.argsort(temp_scores[top_k_indices])[::-1]]
                        
                        for idx in top_k_indices:
                            if temp_scores[idx] > -9.0: # 只要没被 mask 掉
                                selected.append(int(self.all_node_ids[idx]))

        elif view_mode == "sim":
            # 纯相似度模式，全图搜索
            temp_scores = sim_scores.copy()
            temp_scores[c_idx] = -10.0 # 排除自己
            
            top_k_indices = np.argpartition(temp_scores, -budget)[-budget:]
            top_k_indices = top_k_indices[np.argsort(temp_scores[top_k_indices])[::-1]]
            
            selected = [int(self.all_node_ids[idx]) for idx in top_k_indices]
        
        return [center_id] + selected

    def draw_subgraph(
        self,
        center_id: int,
        view_mode: str = "1-hop",
        max_nodes: int = 10,
        color_seed: int = 42,
        undirected: bool = True
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        (采用文件1的视觉优化逻辑)
        包含：边缘稀疏化(只保留重要边)、物理防碰撞布局、中心点/中心边特殊绘制。
        """
        
        final_nodes = self._select_nodes(center_id, view_mode, max_nodes, undirected)
        final_nodes_set = set(final_nodes)
        
        G = nx.Graph()
        G.add_nodes_from(final_nodes)

        # ========= Edge Sparsification (文件1特有逻辑) =========
        # 只保留中心边 + 节点间最重要的 Top-2 相似边，避免连线过于杂乱
        edges = set()

        # 1) 中心边：center -> 其在 final_nodes 内的邻居 (全保留)
        center_nbs = set(self._get_neighbors(center_id, undirected))
        for v in final_nodes:
            if v == center_id:
                continue
            if v in center_nbs:
                a, b = (center_id, v) if center_id < v else (v, center_id)
                edges.add((a, b))

        # 2) 其他节点：只保留与它们最相似的 2 个邻居 (如果也在图中)
        TOP_K_EDGES = 2
        for u in final_nodes:
            if u == center_id:
                continue

            cand = []
            u_nbs = self._get_neighbors(u, undirected)
            for v in u_nbs:
                if v == u or v == center_id:
                    continue
                if v in final_nodes_set:
                    # 使用 _get_similarity (这里利用矩阵加速版)
                    score = self._get_similarity(u, v)
                    cand.append((score, v))

            cand.sort(key=lambda x: x[0], reverse=True)
            for score, v in cand[:TOP_K_EDGES]:
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))

        G.clear_edges()
        G.add_edges_from(list(edges))
        # ========= End Edge Sparsification =========

        # ========= Connectivity policy (文件1逻辑) =========
        # 如果是 sim 模式，允许不连通；如果是 hop 模式，尽量保持连通性
        DISCONNECTED_OK = {"sim", "1-hop+sim", "2-hop+sim"}

        if G.number_of_nodes() > 0 and center_id in G:
            keep = set(nx.node_connected_component(G, center_id))
        else:
            keep = {center_id}

        if view_mode in DISCONNECTED_OK:
            # 允许孤立点，只剔除不在 final_nodes 里的边(其实上面构建时已经保证了)
            # 这里主要是为了只画 "与中心相连" 的边，或者上面稀疏化后的边
            pass
        else:
            # hop 模式下，如果断开了，做个兜底，变成星型图连回去，防止图太难看
            G = G.subgraph(keep).copy()
            MIN_KEEP_NODES = 3
            if G.number_of_nodes() < MIN_KEEP_NODES:
                keep_nodes = {center_id}
                center_nbs = set(self._get_neighbors(center_id, undirected))
                for v in final_nodes:
                    if v != center_id and v in center_nbs:
                        keep_nodes.add(v)

                G = nx.Graph()
                G.add_nodes_from(sorted(list(keep_nodes)))
                for v in keep_nodes:
                    if v != center_id:
                        G.add_edge(center_id, v)
        # ========= End connectivity policy =========

        nodes_to_draw = list(G.nodes())
        
        # 准备颜色
        active_classes = set()
        for u in nodes_to_draw:
            if u != center_id:
                active_classes.add(self._get_node_info(u)["pred_class"] or "Unknown")

        sorted_classes = sorted(list(active_classes))
        color_mapping = self._get_color_map_for_episode(sorted_classes, color_seed)

        # ========= Layout Logic (文件1逻辑：防碰撞优化) =========
        pos_init = {}
        pos_init[center_id] = np.array([0.0, 0.0])
        rng = np.random.RandomState(42)

        # 计算所有点到中心的距离 (基于相似度)
        sim_cache = {u: self._get_similarity(center_id, u) for u in nodes_to_draw if u != center_id}
        R_MIN, R_MAX = 0.5, 2.0

        for u in nodes_to_draw:
            if u == center_id: continue
            sim = sim_cache.get(u, 0.0)
            dist = R_MIN + (1.0 - sim) * (R_MAX - R_MIN)
            theta = rng.uniform(0, 2 * np.pi)
            pos_init[u] = np.array([dist * np.cos(theta), dist * np.sin(theta)])

        n = len(nodes_to_draw)
        k_val = max(0.35, 1.0 / np.sqrt(n)) if n > 0 else 1.0

        if view_mode in DISCONNECTED_OK:
            # sim 模式下使用径向布局，严格反映相似度距离
            pos = {}
            pos[center_id] = np.array([0.0, 0.0], dtype=np.float32)
            golden = 2.399963229728653 
            nodes_others = [nid for nid in nodes_to_draw if nid != center_id]

            for idx, u in enumerate(nodes_others):
                sim = float(sim_cache.get(u, 0.0))
                dist = R_MIN + (1.0 - sim) * (R_MAX - R_MIN)
                theta = (idx + 1) * golden
                pos[u] = np.array([dist * np.cos(theta), dist * np.sin(theta)], dtype=np.float32)
        else:
            # 其他模式使用 Spring Layout
            pos = nx.spring_layout(
                G,
                pos=pos_init,
                seed=42,
                k=k_val,
                iterations=80,
                fixed=[center_id]
            )

        # 平移归零
        if center_id in pos:
            c_xy = pos[center_id]
            for node in pos:
                pos[node] = pos[node] - c_xy

        # -----------------------------------------------
        # 物理防碰撞迭代 (文件1的核心视觉优化)
        # -----------------------------------------------
        MIN_R = 0.5 # 中心禁区
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
            
            # 再次强制中心禁区
            for node in nodes_others:
                xy = pos[node]
                r = float(np.linalg.norm(xy))
                if r < MIN_R:
                    if r < 1e-8:
                        theta = rng.uniform(0, 2 * np.pi)
                        xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                        r = 1.0
                    pos[node] = xy * (MIN_R / r)

            if not moved:
                break
        # -----------------------------------------------

        # 准备样式
        edgecolors = []
        sizes = []
        legend_dict = {}

        for nid in nodes_to_draw:
            if nid == center_id:
                edgecolors.append("black")
                sizes.append(1500)
                legend_dict["Black"] = "Center Node"
            else:
                pred_class = self._get_node_info(nid)["pred_class"]
                c_conf = color_mapping.get(pred_class, {"color": "gray", "name": "Gray"})
                edgecolors.append(c_conf["color"])
                sizes.append(800)
                
                c_name = c_conf["name"]
                if c_name not in legend_dict:
                    legend_dict[c_name] = pred_class

        # 绘图
        fig_size = self.BASE_FIG_SIZE + (len(nodes_to_draw) * 0.02)
        fig = plt.figure(figsize=(fig_size, fig_size))

        # 分开画中心边和其他边 (文件1的视觉优化)
        other_edges = [(u, v) for (u, v) in G.edges() if (u != center_id and v != center_id)]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=other_edges,
            alpha=0.5,
            width=1.6,
            edge_color="black",
        )

        center_edges = [(u, v) for (u, v) in G.edges() if (u == center_id or v == center_id)]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=center_edges,
            alpha=1,  
            width=1.8,  
            edge_color="black",
        )

        nx.draw_networkx_nodes(
            G, pos,
            node_color="white",
            edgecolors=edgecolors,
            linewidths=3.0,
            node_size=sizes
        )
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.axis("off")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        
        buf.seek(0)
        return buf.getvalue(), legend_dict