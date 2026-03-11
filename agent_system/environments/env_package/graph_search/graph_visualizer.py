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
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
from collections import defaultdict
import threading

# 显式指定 matplotlib 自带字体，完全跳过系统字体扫描
# DejaVu Sans 随 matplotlib 安装，100% 存在，无需 findfont 搜索
import matplotlib as _mpl
_mpl.rcParams['font.family'] = 'DejaVu Sans'
_mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']

class GraphVisualizer:
    # [优化] 类级别内存缓存
    _GLOBAL_DATA_CACHE = {}
    _CACHE_LOCK = threading.Lock()
    # matplotlib 非线程安全，所有渲染操作必须串行化
    _RENDER_LOCK = threading.Lock()

    # [新增] 磁盘缓存配置
    CACHE_DIR = "./graph_cache"

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
        self.dataset_name = dataset_name
        
        # 确保磁盘缓存目录存在
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)

        # ── Step 1: 只在锁内做一次快速读取，不做任何计算 ──
        with self._CACHE_LOCK:
            mem_cache = GraphVisualizer._GLOBAL_DATA_CACHE.get(dataset_name)

        # ── Step 2: 内存缓存完整命中（无 shared_data 覆盖）→ 直接返回，零计算 ──
        if mem_cache is not None and shared_data is None:
            self._load_from_memory_cache(dataset_name)
            return

        # ── Step 3: 加载图数据（锁外执行，无竞争） ──
        if shared_data is not None:
            self.graph_data, self.reverse_adj, self.class_map = shared_data[:3]
            self.feat_matrix = (shared_data[3] if len(shared_data) == 4 and shared_data[3] is not None
                                else self._build_feat_matrix())
        else:
            self.graph_data, self.reverse_adj, self.class_map = self.load_graph_data(dataset_name, dataset_dir)
            self.feat_matrix = self._build_feat_matrix()

        # ── Step 4: 拓扑/统计数据 —— 有缓存直接复用，不重算 ──
        if mem_cache is not None:
            # 内存缓存已有全部结果，直接赋值，不持锁、不重算
            self.all_node_ids            = mem_cache['all_node_ids']
            self.id_to_idx               = mem_cache['id_to_idx']
            self.global_degrees          = mem_cache['global_degrees']
            self.global_sim_min          = mem_cache['sim_min']
            self.global_sim_max          = mem_cache['sim_max']
            self.proxy_class_anchors     = mem_cache['anchors']
            self.global_confusion_matrix = mem_cache['confusion']
            return  # shared_data 的图数据已在上面加载，统计直接复用缓存

        # ── Step 5: 真正的首次计算（仅执行一次，后续全走缓存） ──
        self.all_node_ids   = list(self.graph_data.keys())
        self.id_to_idx      = {nid: i for i, nid in enumerate(self.all_node_ids)}
        self.global_degrees = {nid: self.get_node_degree_info(int(nid)) for nid in self.all_node_ids}

        if not self._load_from_disk_cache(dataset_name):
            self.global_sim_min, self.global_sim_max = self._calibrate_global_sim()
            self.proxy_class_anchors, self.global_confusion_matrix = self._build_robust_anchors_and_confusion()
            self._save_to_disk_cache(dataset_name)

        # ── Step 6: 回填内存缓存（加锁保护写操作） ──
        with self._CACHE_LOCK:
            self._update_memory_cache(dataset_name)

    def _get_cache_file_path(self, dataset_name: str) -> str:
        return os.path.join(self.CACHE_DIR, f"{dataset_name}_stats.json")

    def _save_to_disk_cache(self, dataset_name: str):
        """持久化存储全局分布和锚点"""
        cache_path = self._get_cache_file_path(dataset_name)
        # JSON 不支持元组键，需要将 (c1, c2) 转换为 "c1|c2"
        serialized_confusion = {f"{k[0]}|{k[1]}": v for k, v in self.global_confusion_matrix.items()}
        
        cache_data = {
            "sim_min": self.global_sim_min,
            "sim_max": self.global_sim_max,
            "anchors": self.proxy_class_anchors,
            "confusion": serialized_confusion
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"[GraphVisualizer] 离线统计数据已保存: {cache_path}")
        except Exception as e:
            print(f"[Warning] 磁盘缓存保存失败: {e}")

    def _load_from_disk_cache(self, dataset_name: str) -> bool:
        """从磁盘读取缓存，成功返回 True"""
        cache_path = self._get_cache_file_path(dataset_name)
        if not os.path.exists(cache_path):
            return False
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.global_sim_min = float(data["sim_min"])
            self.global_sim_max = float(data["sim_max"])
            self.proxy_class_anchors = data["anchors"]
            
            # 还原混淆矩阵的元组键
            self.global_confusion_matrix = defaultdict(int)
            for k, v in data["confusion"].items():
                if "|" in k:
                    c1, c2 = k.split("|", 1)
                    self.global_confusion_matrix[(c1, c2)] = v
            
            print(f"[GraphVisualizer] 成功从磁盘载入 '{dataset_name}' 的离线校准值。")
            return True
        except Exception as e:
            print(f"[Warning] 磁盘缓存加载异常 (可能是格式版本不匹配): {e}")
            return False

    def _load_from_memory_cache(self, dataset_name: str):
        cache = GraphVisualizer._GLOBAL_DATA_CACHE[dataset_name]
        self.graph_data = cache['graph_data']
        self.reverse_adj = cache['reverse_adj']
        self.class_map = cache['class_map']
        self.feat_matrix = cache['feat_matrix']
        self.all_node_ids = cache['all_node_ids']
        self.id_to_idx = cache['id_to_idx']
        self.global_degrees = cache['global_degrees']
        self.global_sim_min = cache['sim_min']
        self.global_sim_max = cache['sim_max']
        self.proxy_class_anchors = cache['anchors']
        self.global_confusion_matrix = cache['confusion']

    def _update_memory_cache(self, dataset_name: str):
        GraphVisualizer._GLOBAL_DATA_CACHE[dataset_name] = {
            'graph_data': self.graph_data,
            'reverse_adj': self.reverse_adj,
            'class_map': self.class_map,
            'feat_matrix': self.feat_matrix,
            'all_node_ids': self.all_node_ids,
            'id_to_idx': self.id_to_idx,
            'global_degrees': self.global_degrees,
            'sim_min': self.global_sim_min,
            'sim_max': self.global_sim_max,
            'anchors': self.proxy_class_anchors,
            'confusion': self.global_confusion_matrix
        }

    def _calibrate_global_sim(self, sample_size=800, num_rounds=7) -> Tuple[float, float]:
        """
        多轮随机采样后取均值，避免单次采样偏差导致色谱校准结果不稳定后被缓存固化。
        每轮独立采样 sample_size 对节点，共 num_rounds 轮，取各轮 4%/96% 分位数的均值。
        """
        print(f"[GraphVisualizer] 正在校准 '{self.dataset_name}' 的全局特征相似度分布"
              f"（{num_rounds} 轮 × {sample_size} 对）...")
        num_nodes = len(self.feat_matrix)
        if num_nodes < 2:
            return 0.0, 1.0

        round_mins, round_maxs = [], []
        for _ in range(num_rounds):
            idx1 = np.random.randint(0, num_nodes, sample_size)
            idx2 = np.random.randint(0, num_nodes, sample_size)
            sims = np.einsum('ij,ij->i', self.feat_matrix[idx1], self.feat_matrix[idx2])
            r_min, r_max = np.percentile(sims, [4, 96])
            round_mins.append(r_min)
            round_maxs.append(r_max)

        # 多轮均值，消除单轮随机误差，结果稳定后再保存到磁盘
        global_min = float(np.mean(round_mins))
        global_max = float(np.mean(round_maxs))

        if global_max - global_min < 1e-5:
            global_min -= 0.1
            global_max += 0.1

        print(f"[GraphVisualizer] 色谱拉伸校准完毕: "
              f"纯蓝(4%)={global_min:.4f} ± {np.std(round_mins):.4f}, "
              f"纯红(96%)={global_max:.4f} ± {np.std(round_maxs):.4f}")
        return global_min, global_max

    def _build_robust_anchors_and_confusion(self):
        print(f"[GraphVisualizer] 正在构建 '{self.dataset_name}' 的鲁棒类锚点与全局混淆矩阵...")
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
        
        # 补充全局异质虫洞逻辑 
        global_hubs_to_add = []
        added_classes = set()
        anchor_mapping = {}
        
        center_info = self._get_node_info(center_id)
        ranked_labels = center_info.get("proxy_info", {}).get("ranked_labels", [])
        
        def _add_anchor(cls_name):
            if hasattr(self, 'proxy_class_anchors') and cls_name in self.proxy_class_anchors and cls_name not in added_classes:
                anchor_id = self.proxy_class_anchors[cls_name]
                if anchor_id != center_id and anchor_id not in inner_circle_nodes and anchor_id not in outer_circle_2hop:
                    if anchor_id not in global_hubs_to_add:
                        global_hubs_to_add.append(anchor_id)
                        added_classes.add(cls_name)
                        anchor_mapping[anchor_id] = cls_name

        for label in ranked_labels:
            if len(global_hubs_to_add) >= max_global: break
            _add_anchor(label)
            
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
        
        # 2. 节点颜色映射 
        center_idx = self.id_to_idx.get(str(center_id))
        center_feat = self.feat_matrix[center_idx]
        
        node_colors = {}
        node_sims = {}
        cmap = cm.coolwarm
        
        global_sim_min = getattr(self, 'global_sim_min', 0.0)
        global_sim_max = getattr(self, 'global_sim_max', 1.0)
        
        for nid in nodes_to_draw:
            if nid == center_id:
                sim = 1.0
                norm_sim = 1.0
            else:
                idx = self.id_to_idx.get(str(nid))
                sim = float(self.feat_matrix[idx].dot(center_feat)) if idx is not None else 0.0
                denom = (global_sim_max - global_sim_min)
                if denom < 1e-5:
                    norm_sim = 0.5
                else:
                    norm_sim = (sim - global_sim_min) / denom
                norm_sim = max(0.0, min(1.0, norm_sim)) 
            
            node_sims[nid] = sim
            rgba = cmap(norm_sim)
            node_colors[nid] = rgba

        # 3. 组装拓扑连线
        G = nx.Graph()
        G.add_nodes_from(nodes_to_draw)
        final_nodes_set = set(nodes_to_draw)
        
        for u in nodes_to_draw:
            for v in self._get_neighbors(u, undirected=True):
                if v in final_nodes_set and u != v:
                    G.add_edge(u, v)

        # ==========================================
        # 【关键布局】纯角度均匀排布，彻底避免紧贴问题
        # ==========================================
        hop1_nodes = inner_circle_nodes
        other_nodes = outer_circle_nodes
        
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
        
        r1_base = max(4.0, len(hop1_nodes) * 0.4) 
        r1_min, r1_max = r1_base, r1_base + 1.5
        r_dict_1hop = get_dynamic_radii(hop1_nodes, r1_min, r1_max)
        
        r2_base = max(r1_max + 3.0, len(other_nodes) * 0.45)
        r2_min, r2_max = r2_base, r2_base + 2.0
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

        # 4. 边分组 + 节点分类（纯 Python，无 matplotlib，不需要锁）
        edges_1hop  = [(u, v) for u, v in G.edges() if u == center_id or v == center_id]
        edges_other = [(u, v) for u, v in G.edges() if u != center_id and v != center_id]

        shapes_dict = {"s": [], "o": [], "^": [], "v": [], "*": []}
        shapes_dict["s"].append(center_id)
        shapes_dict["*"].extend([n for n in global_hubs_to_add if n in nodes_to_draw])

        local_nodes = [n for n in nodes_to_draw if n != center_id and n not in global_hubs_to_add]
        potential_in_hubs, potential_out_hubs = [], []
        for nid in local_nodes:
            deg_info = self.get_node_degree_info(nid)
            in_d = deg_info.get("in_degree", 0)
            out_d = deg_info.get("out_degree", 0)
            if in_d > 5 and in_d >= out_d: potential_in_hubs.append((nid, in_d))
            elif out_d > 5 and out_d > in_d: potential_out_hubs.append((nid, out_d))
        potential_in_hubs.sort(key=lambda x: x[1], reverse=True)
        potential_out_hubs.sort(key=lambda x: x[1], reverse=True)
        top_in_hubs  = set(x[0] for x in potential_in_hubs[:2])
        top_out_hubs = set(x[0] for x in potential_out_hubs[:2])
        for nid in local_nodes:
            if nid in top_in_hubs: shapes_dict["^"].append(nid)
            elif nid in top_out_hubs: shapes_dict["v"].append(nid)
            else: shapes_dict["o"].append(nid)

        node_catalog_info = {}
        role_map = {"s": "Center", "*": "Macro Hub", "^": "In-Hub", "v": "Out-Hub", "o": "Normal"}
        for shape_marker, nlist in shapes_dict.items():
            for n in nlist:
                node_catalog_info[str(n)] = {"role": role_map[shape_marker], "similarity": f"{node_sims[n]:.2f}"}

        max_radius = r2_max if other_nodes else (r1_max if hop1_nodes else 1.0)
        limit = max_radius * 1.2
        y_offset = limit * 0.035
        label_pos = {}
        for nid, (x, y) in pos.items():
            if nid in top_out_hubs: label_pos[nid] = (x, y + y_offset)
            elif nid in top_in_hubs: label_pos[nid] = (x, y - y_offset)
            else: label_pos[nid] = (x, y)

        labels_center = {center_id: str(center_id)}
        labels_normal = {nid: str(nid) for nid in G.nodes() if nid != center_id and len(str(nid)) < 5}
        labels_small  = {nid: str(nid) for nid in G.nodes() if nid != center_id and len(str(nid)) >= 5}

        # 5. 绘图渲染（加锁：matplotlib 字体查找等全局状态非线程安全）
        with GraphVisualizer._RENDER_LOCK:
            fig = Figure(figsize=(self.BASE_FIG_SIZE, self.BASE_FIG_SIZE))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            nx.draw_networkx_edges(G, pos, edgelist=edges_1hop, alpha=0.85, edge_color="dimgray", width=2.5, ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=edges_other, alpha=0.65, edge_color="gray", style="dashed", width=1.5, ax=ax)

            for shape_marker, nlist in shapes_dict.items():
                if not nlist: continue
                colors = [node_colors[n] for n in nlist]
                size = {"*": 3000, "s": 2200, "^": 2500, "v": 2500}.get(shape_marker, 2000)
                nx.draw_networkx_nodes(G, pos, nodelist=nlist, node_color=colors,
                                       edgecolors="black", linewidths=2.0,
                                       node_size=size, node_shape=shape_marker, ax=ax)

            outline_effect = [pe.withStroke(linewidth=3, foreground='white')]
            texts_c = nx.draw_networkx_labels(G, label_pos, labels=labels_center, font_size=14, font_weight="bold", font_color="black", ax=ax)
            for t in texts_c.values(): t.set_path_effects(outline_effect)
            if labels_normal:
                texts_n = nx.draw_networkx_labels(G, label_pos, labels=labels_normal, font_size=12, font_weight="bold", font_color="black", ax=ax)
                for t in texts_n.values(): t.set_path_effects(outline_effect)
            if labels_small:
                texts_s = nx.draw_networkx_labels(G, label_pos, labels=labels_small, font_size=10, font_weight="bold", font_color="black", ax=ax)
                for t in texts_s.values(): t.set_path_effects(outline_effect)

            legend_elements = [
                mpatches.Patch(facecolor=cmap(0.9), edgecolor='k', label='High Semantic Sim (Red)'),
                mpatches.Patch(facecolor=cmap(0.5), edgecolor='k', label='Neutral Sim (White)'),
                mpatches.Patch(facecolor=cmap(0.1), edgecolor='k', label='Low Semantic Sim (Blue)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', title="Color: Similarity", fontsize=12, title_fontsize=14)
            ax.text(0.95, 0.95, 'Shapes:\n■ Center\n★ Macro Cluster\n▲/▼ Topology Hubs\n● Normal Node',
                    transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.axis("off")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
            img_bytes = buf.getvalue()

        return img_bytes, node_catalog_info, self.get_all_candidate_classes(), anchor_mapping