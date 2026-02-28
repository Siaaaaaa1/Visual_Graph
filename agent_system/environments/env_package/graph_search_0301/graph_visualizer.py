import io
import os
import json
import random
import networkx as nx
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Set, Optional
import time
import matplotlib

# 引入 matplotlib 路径效果库，用于文字描边
import matplotlib.patheffects as pe

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
        self.cmap10 = matplotlib.colormaps["tab10"]
        self.colors10 = self.cmap10.colors
        self.names10 = ["Blue", "Orange", "Green", "Red", "Purple", "Brown", "Pink", "Gray", "Olive", "Cyan"]

        self.cmap20 = matplotlib.colormaps["tab20"]
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

        self.max_save_figs = 100        
        self.saved_fig_count = 0        
        self.save_prob = 0         
        self.save_dir = "./debug_figs"  
        os.makedirs(self.save_dir, exist_ok=True) 

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

    def get_all_candidate_classes(self) -> List[str]:
        classes = set(self.class_map.values()) if self.class_map else set()
        for info in self.graph_data.values():
            if info.get("true_class") and info["true_class"] != "Unknown":
                classes.add(info["true_class"])
            if info.get("pred_class") and info["pred_class"] != "Unknown":
                classes.add(info["pred_class"])
        return sorted(list(classes))

    def _get_color_map_for_episode(self, active_classes: List[str], seed: int, prioritized_classes: Optional[List[str]] = None) -> Dict[str, Dict]:
        rng = random.Random(seed)
        
        pool10 = list(zip(self.colors10, self.names10))
        pool_light = [(self.colors20[i], self.names20[i]) for i in range(1, 20, 2)]
        
        rng.shuffle(pool10)
        rng.shuffle(pool_light)
        full_pool = pool10 + pool_light
        
        if prioritized_classes is None: prioritized_classes = []
            
        p_classes = []
        for c in prioritized_classes:
            if c in active_classes and c not in p_classes:
                p_classes.append(c)
                
        other_classes = sorted([c for c in active_classes if c not in p_classes])
        
        while len(full_pool) < len(active_classes):
            full_pool += full_pool 
            
        color_map = {}
        color_idx = 0
        for cls in p_classes:
            color_map[cls] = {"color": full_pool[color_idx][0], "name": full_pool[color_idx][1]}
            color_idx += 1
            
        for cls in other_classes:
            color_map[cls] = {"color": full_pool[color_idx][0], "name": full_pool[color_idx][1]}
            color_idx += 1
            
        return color_map

    def _get_neighbors(self, node_id: int, undirected: bool):
        out_nbs = self._get_node_info(node_id)["neighbors"]
        if not undirected: return out_nbs
        in_nbs = self.reverse_adj.get(str(node_id), [])
        return list(set(out_nbs) | set(in_nbs))

    def get_candidate_classes(self, center_id: int = -1, top_k: int = 100) -> List[str]:
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
        
        nbs_1hop = list(set(self._get_neighbors(center_id, undirected)))
        if center_id in nbs_1hop: nbs_1hop.remove(center_id)
        
        if len(nbs_1hop) >= max_nodes:
            center_idx = self.id_to_idx.get(str(center_id))
            if center_idx is not None and self.feat_matrix is not None:
                center_feat = self.feat_matrix[center_idx]
                valid_nbs = [n for n in nbs_1hop if str(n) in self.id_to_idx]
                idx_list = [self.id_to_idx[str(n)] for n in valid_nbs]
                if idx_list:
                    sims = self.feat_matrix[idx_list].dot(center_feat)
                    sorted_order = np.argsort(sims)[::-1]
                    return [center_id] + [valid_nbs[i] for i in sorted_order[:max_nodes]]
            return [center_id] + nbs_1hop[:max_nodes]
            
        pool = list(nbs_1hop)
        pool_set = set(pool)
        
        if "2-hop" in view_mode:
            nbs_2hop = set()
            for nb in pool:
                nbs_2hop.update(self._get_neighbors(nb, undirected))
            if center_id in nbs_2hop: nbs_2hop.remove(center_id)
            for n in pool:
                if n in nbs_2hop: nbs_2hop.remove(n)
                
            nbs_2hop = list(nbs_2hop)
            
            if len(pool) + len(nbs_2hop) > max_nodes:
                rem = max_nodes - len(pool)
                center_idx = self.id_to_idx.get(str(center_id))
                if center_idx is not None and self.feat_matrix is not None:
                    center_feat = self.feat_matrix[center_idx]
                    valid_2hop = [n for n in nbs_2hop if str(n) in self.id_to_idx]
                    idx_list = [self.id_to_idx[str(n)] for n in valid_2hop]
                    if idx_list:
                        sims = self.feat_matrix[idx_list].dot(center_feat)
                        sorted_order = np.argsort(sims)[::-1]
                        pool.extend([valid_2hop[i] for i in sorted_order[:rem]])
                else:
                    pool.extend(nbs_2hop[:rem])
            else:
                pool.extend(nbs_2hop)
                
        pool_set = set(pool)
        
        if "sim" in view_mode and len(pool) < max_nodes:
            center_idx = self.id_to_idx.get(str(center_id))
            if center_idx is not None and self.feat_matrix is not None:
                center_feat = self.feat_matrix[center_idx]
                all_sims = self.feat_matrix.dot(center_feat)
                sorted_all_indices = np.argsort(all_sims)[::-1]
                
                sim_added = 0
                for idx in sorted_all_indices:
                    if len(pool) >= max_nodes or sim_added >= 5:
                        break
                    nid = int(self.all_node_ids[idx])
                    if nid != center_id and nid not in pool_set:
                        pool.append(nid)
                        pool_set.add(nid)
                        sim_added += 1

        return [center_id] + pool

    def draw_subgraph(
        self,
        center_id: int,
        view_mode: str = "1-hop",
        max_nodes: int = 10,
        color_seed: int = 42,
        undirected: bool = True,
        mask_neighbors: bool = False,
        painted_nodes: Optional[Dict[int, str]] = None,
        color_mapping: Optional[Dict[str, Dict]] = None,
        anon_map: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, Dict[str, str]]:
        if painted_nodes is None: painted_nodes = {}
        final_nodes = self._select_nodes(center_id, view_mode, max_nodes, undirected)
        
        G = nx.Graph()
        G.add_nodes_from(final_nodes)
        final_nodes_set = set(final_nodes)
        for u in final_nodes:
            for v in self._get_neighbors(u, undirected):
                if v in final_nodes_set and u != v:
                    G.add_edge(u, v)

        nodes_to_draw = list(G.nodes())
        
        node_degrees = {}
        for nid in nodes_to_draw:
            if nid == center_id: continue
            deg = self.get_node_degree_info(nid)
            node_degrees[nid] = deg['out_degree'] + deg['in_degree']
            
        sorted_by_deg = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        top3_hubs = set([nid for nid, _ in sorted_by_deg[:3]])

        nodes_1hop, nodes_high_out, nodes_high_in = [], [], []
        for nid in nodes_to_draw:
            if nid == center_id: continue
            if nid in top3_hubs:
                deg = self.get_node_degree_info(nid)
                if deg['out_degree'] >= deg['in_degree']:
                    nodes_high_out.append(nid)
                else:
                    nodes_high_in.append(nid)
            else:
                nodes_1hop.append(nid)

        if color_mapping is None or anon_map is None:
            all_classes = self.get_all_candidate_classes()
            color_mapping = self._get_color_map_for_episode(all_classes, color_seed, prioritized_classes=all_classes)
            anon_map = {cls: f"Group {i+1}" for i, cls in enumerate(all_classes)}

        center_1hop = set(self._get_neighbors(center_id, undirected))
        hop1_nodes = [n for n in final_nodes if n != center_id and n in center_1hop]
        other_nodes = [n for n in final_nodes if n != center_id and n not in center_1hop]

        G_topo = G.copy()
        for n in G_topo.nodes():
            if n != center_id and not G_topo.has_edge(center_id, n):
                G_topo.add_edge(center_id, n, weight=0.1) 
        
        topo_pos = nx.spring_layout(G_topo, weight='weight', seed=42)
        cx, cy = topo_pos[center_id]
        
        node_angles = {}
        for n in G.nodes():
            if n == center_id: continue
            dx = topo_pos[n][0] - cx
            dy = topo_pos[n][1] - cy
            node_angles[n] = np.arctan2(dy, dx)
            
        hop1_nodes.sort(key=lambda n: node_angles[n])
        other_nodes.sort(key=lambda n: node_angles[n])

        node_sims = {}
        center_idx = self.id_to_idx.get(str(center_id))
        center_feat = self.feat_matrix[center_idx] if (center_idx is not None and self.feat_matrix is not None) else None
        
        for n in G.nodes():
            if n == center_id: continue
            idx = self.id_to_idx.get(str(n))
            if center_feat is not None and idx is not None:
                node_sims[n] = float(self.feat_matrix[idx].dot(center_feat))
            else:
                node_sims[n] = 0.0

        def get_dynamic_radii(nodes_list, r_min, r_max):
            if not nodes_list: return {}
            sims = [node_sims[n] for n in nodes_list]
            s_min, s_max = min(sims), max(sims)
            
            radii = {}
            if s_max - s_min < 1e-5:
                return {n: (r_min + r_max) / 2.0 for n in nodes_list}
            
            for n in nodes_list:
                norm_sim = (node_sims[n] - s_min) / (s_max - s_min)
                radii[n] = r_max - norm_sim * (r_max - r_min)
            return radii

        pos = {center_id: np.array([0.0, 0.0])}
        r1_base = max(3.5, len(hop1_nodes) * 0.35)  
        r1_min, r1_max = r1_base, r1_base + 1.0
        r_dict_1hop = get_dynamic_radii(hop1_nodes, r1_min, r1_max)
        
        r2_base = max(r1_max + 2.5, len(other_nodes) * 0.4)
        r2_min, r2_max = r2_base, r2_base + 1.2
        r_dict_other = get_dynamic_radii(other_nodes, r2_min, r2_max)

        if hop1_nodes:
            angle_step = 2 * np.pi / len(hop1_nodes)
            for i, n in enumerate(hop1_nodes):
                angle = i * angle_step
                r = r_dict_1hop[n]
                pos[n] = np.array([r * np.cos(angle), r * np.sin(angle)])
                
        if other_nodes:
            angle_step = 2 * np.pi / len(other_nodes)
            phase_shift = np.pi / len(other_nodes) if len(other_nodes) > 0 else 0
            for i, n in enumerate(other_nodes):
                angle = i * angle_step + phase_shift
                r = r_dict_other[n]
                pos[n] = np.array([r * np.cos(angle), r * np.sin(angle)])

        fill_colors_dict, edgecolors_dict, sizes_dict, legend_dict = {}, {}, {}, {}
        for nid in nodes_to_draw:
            if nid == center_id:
                fill_colors_dict[nid] = "white"
                edgecolors_dict[nid] = "black"
                sizes_dict[nid] = 2200  # 【优化】缩小中心方框尺寸（原为3000）
                legend_dict["Black"] = "Center Node"
            else:
                true_class = self._get_node_info(nid)["true_class"]
                c_conf = color_mapping.get(true_class, {"color": "gray", "name": "Gray"})
                
                fill_colors_dict[nid] = c_conf["color"]
                edgecolors_dict[nid] = "black"
                
                base_size = 2200 if nid in painted_nodes else 1600
                if nid in nodes_high_out or nid in nodes_high_in:
                    sizes_dict[nid] = int(base_size * 1.6) 
                else:
                    sizes_dict[nid] = base_size

                if mask_neighbors:
                    if nid in painted_nodes:
                        pred_class = painted_nodes[nid]
                        legend_dict[c_conf["name"]] = f"Painted: {pred_class}"
                    else:
                        anon_name = anon_map.get(true_class, "Unknown Group")
                        if c_conf["name"] not in legend_dict or "Painted:" not in legend_dict[c_conf["name"]]:
                            legend_dict[c_conf["name"]] = f"Anonymous: {anon_name}"
                else:
                    if c_conf["name"] not in legend_dict:
                        legend_dict[c_conf["name"]] = true_class

        fig = Figure(figsize=(self.BASE_FIG_SIZE, self.BASE_FIG_SIZE))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="gray", width=2.2, ax=ax)
        
        nx.draw_networkx_nodes(G, pos, nodelist=[center_id], node_color=[fill_colors_dict[center_id]], edgecolors=[edgecolors_dict[center_id]], linewidths=4.0, node_size=[sizes_dict[center_id]], node_shape="s", ax=ax)
        if nodes_1hop:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_1hop, node_color=[fill_colors_dict[n] for n in nodes_1hop], edgecolors=[edgecolors_dict[n] for n in nodes_1hop], linewidths=2.5, node_size=[sizes_dict[n] for n in nodes_1hop], node_shape="o", ax=ax)
        if nodes_high_out:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_high_out, node_color=[fill_colors_dict[n] for n in nodes_high_out], edgecolors=[edgecolors_dict[n] for n in nodes_high_out], linewidths=2.5, node_size=[sizes_dict[n] for n in nodes_high_out], node_shape="v", ax=ax)
        if nodes_high_in:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_high_in, node_color=[fill_colors_dict[n] for n in nodes_high_in], edgecolors=[edgecolors_dict[n] for n in nodes_high_in], linewidths=2.5, node_size=[sizes_dict[n] for n in nodes_high_in], node_shape="^", ax=ax)

        max_radius = r2_max if other_nodes else (r1_max if hop1_nodes else 1.0)
        padding = max_radius * 0.2
        limit = max_radius + padding
        
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.scatter([-limit, limit], [-limit, limit], alpha=0.0) 

        label_pos = {}
        y_offset = limit * 0.035 
        for nid, (x, y) in pos.items():
            if nid in nodes_high_out:
                label_pos[nid] = (x, y + y_offset)
            elif nid in nodes_high_in:
                label_pos[nid] = (x, y - y_offset)
            else:
                label_pos[nid] = (x, y)

        # 【优化】分离出中心节点以单独设置更大字号
        labels_center = {center_id: str(center_id)}
        labels_normal = {nid: str(nid) for nid in G.nodes() if nid != center_id and len(str(nid)) < 5}
        labels_small = {nid: str(nid) for nid in G.nodes() if nid != center_id and len(str(nid)) >= 5}

        # 创建白色描边特效
        outline_effect = [pe.withStroke(linewidth=3, foreground='white')]

        # 【优化】为不同分类应用不同的字号和特效
        texts_c = nx.draw_networkx_labels(G, label_pos, labels=labels_center, font_size=14, font_weight="bold", font_color="black", ax=ax)
        for t in texts_c.values():
            t.set_path_effects(outline_effect)

        if labels_normal:
            texts_n = nx.draw_networkx_labels(G, label_pos, labels=labels_normal, font_size=12, font_weight="bold", font_color="black", ax=ax)
            for t in texts_n.values():
                t.set_path_effects(outline_effect)
                
        if labels_small:
            texts_s = nx.draw_networkx_labels(G, label_pos, labels=labels_small, font_size=10, font_weight="bold", font_color="black", ax=ax)
            for t in texts_s.values():
                t.set_path_effects(outline_effect)

        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        
        img_bytes = buf.getvalue()
        
        if getattr(self, "saved_fig_count", 0) < getattr(self, "max_save_figs", 0) and random.random() < getattr(self, "save_prob", 0):
            timestamp = int(time.time() * 1000)
            save_path = os.path.join(getattr(self, "save_dir", "./debug_figs"), f"graph_c{center_id}_m{view_mode}_{timestamp}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(img_bytes)
                
            self.saved_fig_count += 1
            print(f"[GraphVisualizer] Saved debug fig to {save_path} ({self.saved_fig_count}/{self.max_save_figs})")

        buf.seek(0)
        return img_bytes, legend_dict