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
from collections import defaultdict

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
        self.BASE_FIG_SIZE = 12 
        
        if shared_data is not None:
            self.graph_data, self.reverse_adj, self.class_map = shared_data[:3]
            self.feat_matrix = shared_data[3] if len(shared_data) == 4 and shared_data[3] is not None else self._build_feat_matrix()
        else:
            self.graph_data, self.reverse_adj, self.class_map = self.load_graph_data(dataset_name, dataset_dir)
            self.feat_matrix = self._build_feat_matrix()
        
        self.all_node_ids = list(self.graph_data.keys())
        self.id_to_idx = {nid: i for i, nid in enumerate(self.all_node_ids)}

        self.global_degrees = {nid: self.get_node_degree_info(int(nid)) for nid in self.all_node_ids}
        
        # [新增] 1. 优先校准全局特征相似度的真实分布范围
        self.global_sim_min, self.global_sim_max = self._calibrate_global_sim()
        
        self.proxy_class_anchors, self.global_confusion_matrix = self._build_robust_anchors_and_confusion()

    def _calibrate_global_sim(self, sample_size=1000) -> Tuple[float, float]:
        """随机抽样全局节点对，寻找特征相似度的真实有效分布区间，用于色谱拉伸"""
        print("[GraphVisualizer] 正在校准全局特征相似度分布以优化色谱渲染...")
        num_nodes = len(self.feat_matrix)
        if num_nodes < 2: 
            return 0.0, 1.0
        
        idx1 = np.random.randint(0, num_nodes, sample_size)
        idx2 = np.random.randint(0, num_nodes, sample_size)
        sims = np.sum(self.feat_matrix[idx1] * self.feat_matrix[idx2], axis=1)
        
        # 截取 4% 和 96% 分位数，去除极端离群值，让主体分布撑满红蓝两极
        global_min = float(np.percentile(sims, 4))
        global_max = float(np.percentile(sims, 96))
        
        if global_max - global_min < 1e-5:
            global_min -= 0.1
            global_max += 0.1
            
        print(f"[GraphVisualizer] 色谱拉伸校准完毕: 纯蓝(2%)={global_min:.4f}, 纯红(98%)={global_max:.4f}")
        return global_min, global_max

    def _build_robust_anchors_and_confusion(self):
        print("[GraphVisualizer] 正在离线构建鲁棒类锚点与全局混淆矩阵...")
        class_candidates = defaultdict(list)
        confusion_matrix = defaultdict(int)
        
        for nid_str, info in self.graph_data.items():
            nid = int(nid_str)
            deg = self.global_degrees[nid_str]['in_degree'] + self.global_degrees[nid_str]['out_degree']
            ranked = info.get("proxy_info", {}).get("ranked_labels", [])
            
            for i in range(len(ranked)):
                for j in range(i + 1, len(ranked)):
                    c1, c2 = ranked[i], ranked[j]
                    confusion_matrix[(c1, c2)] += 1
                    confusion_matrix[(c2, c1)] += 1
            
            for rank_idx, cls_name in enumerate(ranked):
                weight = 1.0 / (rank_idx + 1)
                score = deg * weight
                class_candidates[cls_name].append((nid, score))
                
        anchors = {}
        for cls_name, candidates in class_candidates.items():
            if candidates:
                best_node = max(candidates, key=lambda x: x[1])[0]
                anchors[cls_name] = best_node
                
        return anchors, confusion_matrix

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

    def draw_vgraph_radar_layout(self, center_id: int, max_1hop: int = 15, max_2hop: int = 10, max_global: int = 5) -> Tuple[bytes, Dict[str, Dict], List[str], Dict[int, str]]:
        
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
        outer_circle_2hop = sorted([n for n in nbs_2hop if node_degrees_2hop[n]['out_degree'] > 3 or node_degrees_2hop[n]['in_degree'] > 3], 
                                   key=lambda x: node_degrees_2hop[x]['out_degree'] + node_degrees_2hop[x]['in_degree'], reverse=True)[:max_2hop]
        
        # 补充全局异质虫洞 (★) - 三路困难负样本召回策略
        global_hubs_to_add = []
        added_classes = set()
        anchor_mapping = {}
        
        center_info = self._get_node_info(center_id)
        ranked_labels = center_info.get("proxy_info", {}).get("ranked_labels", [])
        
        def _add_anchor(cls_name):
            if cls_name in self.proxy_class_anchors and cls_name not in added_classes:
                anchor_id = self.proxy_class_anchors[cls_name]
                if anchor_id != center_id and anchor_id not in inner_circle_nodes and anchor_id not in outer_circle_2hop:
                    if anchor_id not in global_hubs_to_add:
                        global_hubs_to_add.append(anchor_id)
                        added_classes.add(cls_name)
                        anchor_mapping[anchor_id] = cls_name

        # 【第一路】本地假设锚点
        for label in ranked_labels:
            if len(global_hubs_to_add) >= max_global: break
            _add_anchor(label)
            
        # 【第二路】全局混淆锚点
        if len(global_hubs_to_add) < max_global and len(ranked_labels) > 0:
            top1_cls = ranked_labels[0]
            confused_pairs = []
            if hasattr(self, 'global_confusion_matrix'):
                for (c1, c2), count in self.global_confusion_matrix.items():
                    if c1 == top1_cls and c2 not in added_classes:
                        confused_pairs.append((c2, count))
                confused_pairs.sort(key=lambda x: x[1], reverse=True)
                for cls_name, _ in confused_pairs:
                    if len(global_hubs_to_add) >= max_global - 1: break
                    _add_anchor(cls_name)

        # 【第三路】纯视觉欺骗锚点
        if len(global_hubs_to_add) < max_global and hasattr(self, 'proxy_class_anchors'):
            center_idx = self.id_to_idx.get(str(center_id))
            center_feat = self.feat_matrix[center_idx]
            feature_negatives = []
            for cls_name, anchor_id in self.proxy_class_anchors.items():
                if cls_name in added_classes: continue
                a_idx = self.id_to_idx.get(str(anchor_id))
                if a_idx is not None:
                    sim = float(self.feat_matrix[a_idx].dot(center_feat))
                    feature_negatives.append((cls_name, sim))
            feature_negatives.sort(key=lambda x: x[1], reverse=True)
            for cls_name, _ in feature_negatives:
                if len(global_hubs_to_add) >= max_global: break
                _add_anchor(cls_name)
                
        outer_circle_nodes = outer_circle_2hop + global_hubs_to_add
        nodes_to_draw = [center_id] + inner_circle_nodes + outer_circle_nodes
        
        # [核心修复] 2. 基于“全局分布校准”计算颜色映射，根治表示退化与局部失真
        center_idx = self.id_to_idx.get(str(center_id))
        center_feat = self.feat_matrix[center_idx]
        
        node_colors = {}
        node_sims = {}
        cmap = cm.coolwarm
        
        for nid in nodes_to_draw:
            if nid == center_id:
                sim = 1.0
                norm_sim = 1.0 # 中心节点绝对红
            else:
                idx = self.id_to_idx.get(str(nid))
                sim = float(self.feat_matrix[idx].dot(center_feat)) if idx is not None else 0.0
                
                # 使用我们在 __init__ 中探明的全局真实上下界拉伸
                norm_sim = (sim - self.global_sim_min) / (self.global_sim_max - self.global_sim_min)
                norm_sim = max(0.0, min(1.0, norm_sim)) # 超出 98% 都是纯红，低于 2% 都是纯蓝
            
            node_sims[nid] = sim
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
        
        r_inner = 4.0
        if inner_circle_nodes:
            angle_step = 2 * np.pi / len(inner_circle_nodes)
            for i, n in enumerate(inner_circle_nodes):
                pos[n] = np.array([r_inner * np.cos(i * angle_step), r_inner * np.sin(i * angle_step)])
                
        r_outer = 7.5
        if outer_circle_nodes:
            angle_step = 2 * np.pi / len(outer_circle_nodes)
            for i, n in enumerate(outer_circle_nodes):
                pos[n] = np.array([r_outer * np.cos(i * angle_step), r_outer * np.sin(i * angle_step)])

        # 4. 绘图渲染
        fig = Figure(figsize=(self.BASE_FIG_SIZE, self.BASE_FIG_SIZE))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        edges_1hop = [(u, v) for u, v in G.edges() if u == center_id or v == center_id]
        edges_other = [(u, v) for u, v in G.edges() if u != center_id and v != center_id]
        
        # ---------- 修改后的代码 ----------
        # 1. 中心辐射实线：加深为 dimgray，透明度提高到 0.85，确保作为视觉锚点足够清晰
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=edges_1hop, 
            alpha=0.85, 
            edge_color="dimgray", 
            width=2.5, 
            ax=ax
        )
        
        # 2. 其他上下文虚线：颜色改为 gray，透明度提升至 0.65，既能看清走向，又不会盖过中心线的风头
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=edges_other, 
            alpha=0.65, 
            edge_color="gray", 
            style="dashed", 
            width=1.5, 
            ax=ax
        )

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
            
            role_map = {"s": "Center", "*": "Macro Hub", "^": "In-Hub", "v": "Out-Hub", "o": "Normal"}
            for n in nlist:
                node_catalog_info[str(n)] = {"role": role_map[shape_marker], "similarity": f"{node_sims[n]:.2f}"}

        labels_dict = {n: str(n) for n in nodes_to_draw}
        outline_effect = [pe.withStroke(linewidth=3, foreground='white')]
        texts = nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=11, font_weight="bold", font_color="black", ax=ax)
        for t in texts.values():
            t.set_path_effects(outline_effect)

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
        
        return img_bytes, node_catalog_info, self.get_all_candidate_classes(), anchor_mapping