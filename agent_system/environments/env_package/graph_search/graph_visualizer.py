import io
import os
import json
import random
import networkx as nx
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional
import time
import matplotlib
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches

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
                "feature": feature,
                "proxy_info": proxy
            }
            
            for nb in neighbors:
                nb_str = str(nb)
                if nb_str not in reverse_adj:
                    reverse_adj[nb_str] = []
                reverse_adj[nb_str].append(int(nid))
                
        return graph_data, reverse_adj, class_map

    def __init__(self, dataset_name: str, dataset_dir: str = "./datasets", shared_data: Optional[Tuple[Dict, Dict, Dict, Any]] = None):
        self.BASE_FIG_SIZE = 12 # 稍微放大以容纳图例
        
        if shared_data is not None:
            self.graph_data, self.reverse_adj, self.class_map = shared_data[:3]
            self.feat_matrix = shared_data[3] if len(shared_data) == 4 and shared_data[3] is not None else self._build_feat_matrix()
        else:
            self.graph_data, self.reverse_adj, self.class_map = self.load_graph_data(dataset_name, dataset_dir)
            self.feat_matrix = self._build_feat_matrix()
        
        self.all_node_ids = list(self.graph_data.keys())
        self.id_to_idx = {nid: i for i, nid in enumerate(self.all_node_ids)}

        # 预计算全局 Hubs 以备“外圈虫洞”使用
        self.global_degrees = {nid: self.get_node_degree_info(int(nid)) for nid in self.all_node_ids}
        self.sorted_global_hubs = sorted(self.all_node_ids, key=lambda x: self.global_degrees[x]['in_degree'] + self.global_degrees[x]['out_degree'], reverse=True)

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
            if info.get("true_class") and info["true_class"] != "Unknown": classes.add(info["true_class"])
        return sorted(list(classes))

    def get_node_degree_info(self, node_id: int) -> Dict[str, int]:
        node_str = str(node_id)
        info = self._get_node_info(node_id)
        out_degree = len(info["neighbors"])
        in_degree = len(self.reverse_adj.get(node_str, []))
        return {"in_degree": in_degree, "out_degree": out_degree}

    def _get_neighbors(self, node_id: int, undirected: bool = True):
        out_nbs = self._get_node_info(node_id)["neighbors"]
        if not undirected: return out_nbs
        in_nbs = self.reverse_adj.get(str(node_id), [])
        return list(set(out_nbs) | set(in_nbs))

    def draw_vgraph_radar_layout(self, center_id: int, max_1hop: int = 15, max_2hop: int = 10, max_global: int = 5) -> Tuple[bytes, Dict[str, Dict], List[str]]:
        """V-GraphAgent 2.0: 纯正同心圆 + 余弦相似度热力图映射"""
        
        # 1. 挑选节点构建视图
        nbs_1hop = self._get_neighbors(center_id, undirected=True)
        if center_id in nbs_1hop: nbs_1hop.remove(center_id)
        
        node_degrees = {n: self.get_node_degree_info(n) for n in nbs_1hop}
        inner_circle_nodes = sorted(nbs_1hop, key=lambda x: node_degrees[x]['out_degree'] + node_degrees[x]['in_degree'], reverse=True)[:max_1hop]
        
        nbs_2hop = set()
        for n in inner_circle_nodes: nbs_2hop.update(self._get_neighbors(n, undirected=True))
        nbs_2hop.difference_update(set(inner_circle_nodes))
        nbs_2hop.discard(center_id)
        
        node_degrees_2hop = {n: self.get_node_degree_info(n) for n in nbs_2hop}
        # 过滤出 2-hop 中的绝对 Hub
        outer_circle_2hop = sorted([n for n in nbs_2hop if node_degrees_2hop[n]['out_degree'] > 3 or node_degrees_2hop[n]['in_degree'] > 3], 
                                   key=lambda x: node_degrees_2hop[x]['out_degree'] + node_degrees_2hop[x]['in_degree'], reverse=True)[:max_2hop]
        
        # 补充全局异质虫洞 (★)
        global_hubs_to_add = []
        for n_str in self.sorted_global_hubs:
            nid = int(n_str)
            if nid != center_id and nid not in inner_circle_nodes and nid not in outer_circle_2hop:
                global_hubs_to_add.append(nid)
            if len(global_hubs_to_add) >= max_global: break
                
        outer_circle_nodes = outer_circle_2hop + global_hubs_to_add
        nodes_to_draw = [center_id] + inner_circle_nodes + outer_circle_nodes
        
        # 2. 计算特征余弦相似度并映射到 CoolWarm 颜色空间
        center_idx = self.id_to_idx.get(str(center_id))
        center_feat = self.feat_matrix[center_idx]
        
        node_colors = {}
        node_sims = {}
        cmap = cm.coolwarm
        
        for nid in nodes_to_draw:
            if nid == center_id:
                sim = 1.0
            else:
                idx = self.id_to_idx.get(str(nid))
                sim = float(self.feat_matrix[idx].dot(center_feat)) if idx is not None else 0.0
            
            node_sims[nid] = sim
            # 将 [-1, 1] 映射到 [0, 1] 获取颜色
            norm_sim = max(0.0, min(1.0, (sim + 1.0) / 2.0))
            rgba = cmap(norm_sim)
            node_colors[nid] = rgba

        # 3. 组装拓扑连线与完美同心圆坐标
        G = nx.Graph()
        G.add_nodes_from(nodes_to_draw)
        final_nodes_set = set(nodes_to_draw)
        
        for u in nodes_to_draw:
            for v in self._get_neighbors(u, undirected=True):
                if v in final_nodes_set and u != v:
                    G.add_edge(u, v)
                    
        pos = {center_id: np.array([0.0, 0.0])}
        
        # 完美内圈
        r_inner = 4.0
        if inner_circle_nodes:
            angle_step = 2 * np.pi / len(inner_circle_nodes)
            for i, n in enumerate(inner_circle_nodes):
                pos[n] = np.array([r_inner * np.cos(i * angle_step), r_inner * np.sin(i * angle_step)])
                
        # 完美外圈
        r_outer = 7.5
        if outer_circle_nodes:
            angle_step = 2 * np.pi / len(outer_circle_nodes)
            for i, n in enumerate(outer_circle_nodes):
                pos[n] = np.array([r_outer * np.cos(i * angle_step), r_outer * np.sin(i * angle_step)])

        # 4. 绘图渲染
        fig = Figure(figsize=(self.BASE_FIG_SIZE, self.BASE_FIG_SIZE))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # 绘制不同类型的虚实连线
        edges_1hop = [(u, v) for u, v in G.edges() if u == center_id or v == center_id]
        edges_other = [(u, v) for u, v in G.edges() if u != center_id and v != center_id]
        
        nx.draw_networkx_edges(G, pos, edgelist=edges_1hop, alpha=0.6, edge_color="gray", width=2.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges_other, alpha=0.3, edge_color="lightgray", style="dashed", width=1.5, ax=ax)

        # 区分形状绘制
        shapes_dict = {"s": [], "o": [], "^": [], "v": [], "*": []}
        for nid in nodes_to_draw:
            if nid == center_id:
                shapes_dict["s"].append(nid)
            elif nid in global_hubs_to_add:
                shapes_dict["*"].append(nid)
            else:
                deg = self.get_node_degree_info(nid)
                if deg["in_degree"] > 10 and deg["in_degree"] >= deg["out_degree"]:
                    shapes_dict["^"].append(nid)
                elif deg["out_degree"] > 10 and deg["out_degree"] > deg["in_degree"]:
                    shapes_dict["v"].append(nid)
                else:
                    shapes_dict["o"].append(nid)

        node_catalog_info = {}
        for shape_marker, nlist in shapes_dict.items():
            if not nlist: continue
            colors = [node_colors[n] for n in nlist]
            size = 3500 if shape_marker == "s" else (2200 if shape_marker in ["*", "^", "v"] else 1800)
            nx.draw_networkx_nodes(G, pos, nodelist=nlist, node_color=colors, edgecolors="black", linewidths=2.0, node_size=size, node_shape=shape_marker, ax=ax)
            
            # 记录 Catalog，供纯文本核对（不含摘要）
            role_map = {"s": "Center", "*": "Macro Hub", "^": "In-Hub", "v": "Out-Hub", "o": "Normal"}
            for n in nlist:
                node_catalog_info[str(n)] = {"role": role_map[shape_marker], "similarity": f"{node_sims[n]:.2f}"}

        # 绘制描边 ID
        labels_dict = {n: str(n) for n in nodes_to_draw}
        outline_effect = [pe.withStroke(linewidth=3, foreground='white')]
        texts = nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=11, font_weight="bold", font_color="black", ax=ax)
        for t in texts.values():
            t.set_path_effects(outline_effect)

        # 5. 绘制视觉说明图例 (Legend) - 帮助大模型理解颜色和形状
        legend_elements = [
            mpatches.Patch(facecolor=cmap(0.9), edgecolor='k', label='High Semantic Sim (Red)'),
            mpatches.Patch(facecolor=cmap(0.5), edgecolor='k', label='Neutral Sim (White)'),
            mpatches.Patch(facecolor=cmap(0.1), edgecolor='k', label='Low Semantic Sim (Blue)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', title="Color: Similarity", fontsize=10, title_fontsize=12)

        ax.text(0.95, 0.95, 'Shapes:\n■ Center\n★ Macro Cluster\n▲/▼ Topology Hubs\n● Normal Node', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        limit = r_outer * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axis("off")
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        img_bytes = buf.getvalue()
        buf.seek(0)
        
        return img_bytes, node_catalog_info, self.get_all_candidate_classes()