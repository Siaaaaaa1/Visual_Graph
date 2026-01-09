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
            nid = str(node["id"])
            proxy = node.get("proxy_info", {})
            pred_class = proxy.get("top1", "Unknown")
            neighbors = node.get("neighbors", [])
            
            graph_data[nid] = {
                "neighbors": neighbors,
                "pred_class": pred_class
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
        color_seed: int = 42  # ✅ 接收外部传入的 Seed
    ) -> Tuple[bytes, Dict[str, str]]:
        
        # 1. 节点筛选 (BFS)
        center_info = self._get_node_info(center_id)
        neighbors_1hop = center_info["neighbors"]
        
        candidates = []
        if mode == "1-hop":
            candidates = neighbors_1hop
        elif "2-hop" in mode:
            candidates.extend(neighbors_1hop)
            for nb in neighbors_1hop:
                nb_info = self._get_node_info(nb)
                candidates.extend(nb_info["neighbors"])
        
        # 去重、排除中心、截断
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen and c != center_id:
                seen.add(c)
                unique_candidates.append(c)
        
        budget = max(0, max_nodes - 1)
        final_nodes = [center_id] + unique_candidates[:budget]
        final_nodes_set = set(final_nodes)

        # 2. 建图
        G = nx.Graph()
        G.add_nodes_from(final_nodes)
        
        active_classes = set()
        for u in final_nodes:
            if u != center_id:
                active_classes.add(self._get_node_info(u)["pred_class"])
            
            nbs = self._get_node_info(u)["neighbors"]
            for v in nbs:
                if v in final_nodes_set:
                    if u < v: 
                        G.add_edge(u, v)

        # 3. 配色 (传入 seed)
        sorted_classes = sorted(list(active_classes))
        color_mapping = self._get_color_map_for_episode(sorted_classes, color_seed)

        pos = nx.spring_layout(G, seed=42)

        edgecolors = []
        sizes = []
        legend_dict = {} 
        
        for nid in final_nodes:
            if nid == center_id:
                edgecolors.append("black")
                sizes.append(1200)
                legend_dict["Black"] = "Center Node"
            else:
                pred_class = self._get_node_info(nid)["pred_class"]
                c_conf = color_mapping.get(pred_class, {"color": "gray", "name": "Gray"})
                edgecolors.append(c_conf["color"])
                sizes.append(800)
                
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
            linewidths=2,
            node_size=sizes
        )
        nx.draw_networkx_labels(G, pos, font_size=7)
        plt.axis("off")

        # 5. 保存
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        
        plt.close(fig)
        
        buf.seek(0)
        image_bytes = buf.getvalue()

        return image_bytes, legend_dict