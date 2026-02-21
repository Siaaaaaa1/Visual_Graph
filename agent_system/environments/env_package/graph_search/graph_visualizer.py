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
import time

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
        
        self.cmap10 = plt.get_cmap("tab10")
        self.colors10 = self.cmap10.colors
        self.names10 = ["Blue", "Orange", "Green", "Red", "Purple", "Brown", "Pink", "Gray", "Olive", "Cyan"]

        self.cmap20 = plt.get_cmap("tab20")
        self.colors20 = self.cmap20.colors
        self.names20 = ["Blue", "LightBlue", "Orange", "LightOrange", "Green", "LightGreen", "Red", "LightRed",
            "Purple", "LightPurple", "Brown", "LightBrown", "Pink", "LightPink", "Gray", "LightGray",
            "Olive", "LightOlive", "Cyan", "LightCyan"]
        
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

    def get_candidate_classes(self, center_id: int, top_k: int = 100) -> List[str]:
        return list(self.class_map.values())
    
    def get_node_degree_info(self, node_id: int) -> Dict[str, int]:
        node_str = str(node_id)
        info = self._get_node_info(node_id)
        out_degree = len(info["neighbors"])
        in_degree = len(self.reverse_adj.get(node_str, []))
        undirected_1hop = set(info["neighbors"]) | set(self.reverse_adj.get(node_str, []))
        if node_id in undirected_1hop: undirected_1hop.remove(node_id)
            
        return {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "neighbor_count_1hop": len(undirected_1hop),
        }

    def _select_nodes(self, center_id: int, view_mode: str, max_nodes: int, undirected: bool) -> List[int]:
        if view_mode == "center": return [center_id]
        
        # 1. 采集候选节点：提取 1-hop 和 2-hop 邻居
        candidates = set()
        neighbors_1hop = set(self._get_neighbors(center_id, undirected))
        candidates.update(neighbors_1hop)
        
        if "2-hop" in view_mode:
            for nb in neighbors_1hop:
                candidates.update(self._get_neighbors(nb, undirected))
                
        if center_id in candidates: candidates.remove(center_id)
        
        pool = list(candidates)
        
        # 2. 特征相似度过滤 (Top-K)
        if "sim" in view_mode and len(pool) > max_nodes:
            center_idx = self.id_to_idx.get(str(center_id))
            if center_idx is not None and self.feat_matrix is not None:
                center_feat = self.feat_matrix[center_idx]
                pool_indices = [self.id_to_idx.get(str(nid)) for nid in pool]
                
                valid_pool = []
                sims = []
                for nid, idx in zip(pool, pool_indices):
                    if idx is not None:
                        # 计算余弦相似度 (假设矩阵在初始化时已经 L2 Normalize)
                        sim = np.dot(center_feat, self.feat_matrix[idx])
                        valid_pool.append(nid)
                        sims.append(sim)
                    else:
                        valid_pool.append(nid)
                        sims.append(-1.0)
                
                # 按相似度降序排序并截断
                sorted_pairs = sorted(zip(valid_pool, sims), key=lambda x: x[1], reverse=True)
                pool = [p[0] for p in sorted_pairs]
                
        selected = pool[:max_nodes]

        # 3. 视觉防噪截断：限制高出入度节点数量 (最多 3 个)
        high_deg_count = 0
        final_selected = []
        for n in selected:
            deg = self.get_node_degree_info(n)
            if deg['out_degree'] > 5 or deg['in_degree'] > 5:
                if high_deg_count < 3: 
                    final_selected.append(n)
                    high_deg_count += 1
            else:
                final_selected.append(n)

        return [center_id] + final_selected

    def draw_subgraph(
        self,
        center_id: int,
        view_mode: str = "1-hop",
        max_nodes: int = 10,
        color_seed: int = 42,
        undirected: bool = True,
        mask_neighbors: bool = False,
        painted_nodes: Optional[Dict[int, str]] = None 
    ) -> Tuple[bytes, Dict[str, str]]:
        
        if painted_nodes is None: painted_nodes = {}
        final_nodes = self._select_nodes(center_id, view_mode, max_nodes, undirected)
        G = nx.Graph()
        G.add_nodes_from(final_nodes)

        for v in final_nodes:
            if v != center_id: G.add_edge(center_id, v)

        nodes_to_draw = list(G.nodes())
        
        active_classes = set()
        for u in nodes_to_draw:
            if u != center_id:
                active_classes.add(self._get_node_info(u)["true_class"] or "Unknown")
                if u in painted_nodes:
                    active_classes.add(painted_nodes[u])

        color_mapping = self._get_color_map_for_episode(sorted(list(active_classes)), color_seed)
        pos = nx.spring_layout(G, seed=42)

        edgecolors_dict = {}
        sizes_dict = {}
        legend_dict = {}

        for nid in nodes_to_draw:
            if nid == center_id:
                edgecolors_dict[nid] = "black"
                sizes_dict[nid] = 1500
                legend_dict["Black"] = "Center Node"
            else:
                if mask_neighbors:
                    if nid in painted_nodes:
                        pred_class = painted_nodes[nid]
                        c_conf = color_mapping.get(pred_class, {"color": "gray", "name": "Gray"})
                        edgecolors_dict[nid] = c_conf["color"]
                        sizes_dict[nid] = 1000
                        legend_dict[c_conf["name"]] = f"Painted: {pred_class}"
                    else:
                        edgecolors_dict[nid] = "black"
                        sizes_dict[nid] = 800
                        # 【修正】修复图例键值冲突，避免覆盖 "Black": "Center Node"
                        if "Masked Nodes" not in legend_dict:
                            legend_dict["Masked Nodes"] = "Unlabeled Neighbors (Fog)"
                else:
                    true_class = self._get_node_info(nid)["true_class"]
                    c_conf = color_mapping.get(true_class, {"color": "gray", "name": "Gray"})
                    edgecolors_dict[nid] = c_conf["color"]
                    sizes_dict[nid] = 800
                    if c_conf["name"] not in legend_dict:
                        legend_dict[c_conf["name"]] = true_class

        # 【核心修改】形状解耦
        nodes_1hop, nodes_high_out, nodes_high_in = [], [], []
        for nid in nodes_to_draw:
            if nid == center_id: continue
            deg = self.get_node_degree_info(nid)
            if deg['out_degree'] > 5: nodes_high_out.append(nid)
            elif deg['in_degree'] > 5: nodes_high_in.append(nid)
            else: nodes_1hop.append(nid)

        fig = plt.figure(figsize=(self.BASE_FIG_SIZE, self.BASE_FIG_SIZE))
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="black")
        
        # 绘制中心节点 (正方形)
        nx.draw_networkx_nodes(G, pos, nodelist=[center_id], node_color="white", edgecolors=[edgecolors_dict[center_id]], linewidths=3.0, node_size=[sizes_dict[center_id]], node_shape="s")
        
        # 绘制 1-hop (圆形 o)
        if nodes_1hop:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_1hop, node_color="white", edgecolors=[edgecolors_dict[n] for n in nodes_1hop], linewidths=3.0, node_size=[sizes_dict[n] for n in nodes_1hop], node_shape="o")
        
        # 绘制 高出度 (倒三角 v)
        if nodes_high_out:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_high_out, node_color="white", edgecolors=[edgecolors_dict[n] for n in nodes_high_out], linewidths=3.0, node_size=[sizes_dict[n] for n in nodes_high_out], node_shape="v")
        
        # 绘制 高入度 (正三角 ^)
        if nodes_high_in:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_high_in, node_color="white", edgecolors=[edgecolors_dict[n] for n in nodes_high_in], linewidths=3.0, node_size=[sizes_dict[n] for n in nodes_high_in], node_shape="^")

        # 【补充】绘制节点序号 (显示节点的ID)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        # 提取图片字节数据
        img_bytes = buf.getvalue()
        
        # 【新增】以概率保存图片，最多保存 self.max_save_figs 张
        if self.saved_fig_count < self.max_save_figs and random.random() < self.save_prob:
            timestamp = int(time.time() * 1000)
            save_path = os.path.join(self.save_dir, f"graph_c{center_id}_m{view_mode}_{timestamp}.png")
            
            # 直接将内存中的字节写入文件，最高效
            with open(save_path, "wb") as f:
                f.write(img_bytes)
                
            self.saved_fig_count += 1
            print(f"[GraphVisualizer] Saved debug fig to {save_path} ({self.saved_fig_count}/{self.max_save_figs})")

        buf.seek(0)
        return img_bytes, legend_dict