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
    def load_graph_data(dataset_name: str, dataset_dir: str) -> Tuple[Dict, Dict]:
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
            
        self.all_node_ids = list(self.graph_data.keys())

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
        f1 = np.array(self.graph_data[str(center_id)]["feature"], dtype=np.float32)
        f2 = np.array(self.graph_data[str(node_id)]["feature"], dtype=np.float32)
        denom = (np.linalg.norm(f1) * np.linalg.norm(f2))
        if denom == 0:
            return 0.0
        return float(np.dot(f1, f2) / denom)

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
                    if len(candidates) >= top_k:
                        break
        
        classes = set()
        for nid in candidates:
            cls = self._get_node_info(nid)["pred_class"]
            classes.add(cls)
            
        return sorted(list(classes))

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

        def get_sim(nid):
            return self._get_similarity(center_id, nid)

        selected = []
        budget = max_nodes

        if view_mode == "1-hop":
            pool = list(neighbors_1hop)
            pool.sort(key=get_sim, reverse=True)
            selected = pool[:budget]

        elif view_mode == "2-hop":
            pool = list(neighbors_1hop | neighbors_2hop)
            pool.sort(key=get_sim, reverse=True)
            selected = pool[:budget]

        elif view_mode == "sim":
            all_nodes = [int(n) for n in self.all_node_ids if int(n) != center_id]
            all_nodes.sort(key=get_sim, reverse=True)
            selected = all_nodes[:budget]

        elif view_mode == "1-hop+sim":
            pool_1 = list(neighbors_1hop)
            pool_1.sort(key=get_sim, reverse=True)
            if len(pool_1) >= budget:
                selected = pool_1[:budget]
            else:
                selected = pool_1[:]
                remain = budget - len(selected)
                all_others = [int(n) for n in self.all_node_ids 
                              if int(n) != center_id and int(n) not in neighbors_1hop]
                all_others.sort(key=get_sim, reverse=True)
                selected.extend(all_others[:remain])

        elif view_mode == "2-hop+sim":
            pool_local = list(neighbors_1hop | neighbors_2hop)
            pool_local.sort(key=get_sim, reverse=True)
            if len(pool_local) >= budget:
                selected = pool_local[:budget]
            else:
                selected = pool_local[:]
                remain = budget - len(selected)
                local_set = set(pool_local)
                all_others = [int(n) for n in self.all_node_ids 
                              if int(n) != center_id and int(n) not in local_set]
                all_others.sort(key=get_sim, reverse=True)
                selected.extend(all_others[:remain])
        
        return [center_id] + selected

    def draw_subgraph(
        self,
        center_id: int,
        view_mode: str = "1-hop",
        max_nodes: int = 10,
        color_seed: int = 42,
        undirected: bool = True
    ) -> Tuple[bytes, Dict[str, str]]:
        
        final_nodes = self._select_nodes(center_id, view_mode, max_nodes, undirected)
        final_nodes_set = set(final_nodes)
        
        G = nx.Graph()
        G.add_nodes_from(final_nodes)

        active_classes = set()
        for u in final_nodes:
            if u != center_id:
                active_classes.add(
                    self._get_node_info(u)["pred_class"] or "Unknown"
                )

        for u in final_nodes:
            nbs = self._get_neighbors(u, undirected)
            for v in nbs:
                if v in final_nodes_set and u < v:
                     G.add_edge(u, v)
        
        sorted_classes = sorted(list(active_classes))
        color_mapping = self._get_color_map_for_episode(sorted_classes, color_seed)

        pos_init = {}
        pos_init[center_id] = np.array([0.0, 0.0])
        rng = np.random.RandomState(42)

        sim_cache = {u: self._get_similarity(center_id, u) for u in final_nodes if u != center_id}
        R_MIN, R_MAX = 0.5, 2.0

        for u in final_nodes:
            if u == center_id: continue
            sim = sim_cache.get(u, 0.0)
            dist = R_MIN + (1.0 - sim) * (R_MAX - R_MIN)
            theta = rng.uniform(0, 2 * np.pi)
            pos_init[u] = np.array([dist * np.cos(theta), dist * np.sin(theta)])

        k_val = 1.0 / np.sqrt(len(final_nodes)) if len(final_nodes) > 0 else 1.0
        pos = nx.spring_layout(G, pos=pos_init, seed=42, k=k_val, iterations=50)

        if center_id in pos:
            c_xy = pos[center_id]
            for n in pos:
                pos[n] -= c_xy

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
                sizes.append(800)
                
                c_name = c_conf["name"]
                if c_name not in legend_dict:
                    legend_dict[c_name] = pred_class

        fig_size = self.BASE_FIG_SIZE + (len(final_nodes) * 0.02)
        fig = plt.figure(figsize=(fig_size, fig_size))

        nx.draw_networkx_edges(G, pos, alpha=0.2)
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