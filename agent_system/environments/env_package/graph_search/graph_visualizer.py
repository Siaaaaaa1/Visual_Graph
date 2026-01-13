import io
import os
import json
import random
import hashlib
import networkx as nx
import matplotlib
# 配置 Matplotlib 后端为 'Agg' (非交互模式)，适用于服务器端批量渲染，避免内存泄漏
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Set, Optional

class GraphVisualizer:
    @staticmethod
    def load_graph_data(dataset_name: str, dataset_dir: str) -> Tuple[Dict, Dict]:
        """
        静态方法：加载原始图数据并建立索引。
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
            
            # 构建反向邻接表，支持无向图查询
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
        
        # --- [新增] 分辨率控制参数 ---
        # 限制画布尺寸（英寸），防止节点过多导致图片无限变大
        self.MIN_FIG_SIZE = 8.0   
        self.MAX_FIG_SIZE = 20.0  
        # 固定 DPI (Dots Per Inch)，控制像素密度
        # 最终像素 = FIG_SIZE * DPI。例如 Max 20 * 100 = 2000px 宽/高
        self.DPI = 100            
        
        # 1. Tab10 色板 (适用于类别数 <= 10)
        self.cmap10 = plt.get_cmap("tab10")
        self.colors10 = self.cmap10.colors
        self.names10 = [
            "Blue", "Orange", "Green", "Red", "Purple",
            "Brown", "Pink", "Gray", "Olive", "Cyan"
        ]

        # 2. Tab20 色板 (适用于类别数 > 10)
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
        生成当前 Episode 的类别颜色映射。
        
        策略:
        1. 随机性 (Seed): 随机打乱色盘，使不同 Episode 的整体配色方案不同，防止模型过拟合特定颜色。
        2. 稳定性 (Hash): 使用 MD5 哈希将类名映射到打乱后的索引。只要 Seed 不变，同一类名永远对应同一颜色。
        """
        num_classes = len(active_classes)
        
        # 选择合适的色盘
        if num_classes <= 10:
            base_colors = list(zip(self.colors10, self.names10))
        else:
            base_colors = list(zip(self.colors20, self.names20))
            
        # 基于 seed 随机打乱色盘顺序
        rng = random.Random(seed)
        shuffled_colors = list(base_colors)
        rng.shuffle(shuffled_colors)
        
        color_map = {}
        for cls_name in active_classes:
            # 使用稳定哈希确定索引
            hash_val = int(hashlib.md5(cls_name.encode()).hexdigest(), 16)
            idx = hash_val % len(shuffled_colors)
            
            color_tuple, color_name = shuffled_colors[idx]
            color_map[cls_name] = {
                "color": color_tuple,
                "name": color_name
            }
        return color_map

    def _get_neighbors(self, node_id: int, undirected: bool):
        """根据是否为无向图模式，获取节点的邻居集合。"""
        out_nbs = self._get_node_info(node_id)["neighbors"]
        if not undirected:
            return out_nbs

        # 无向模式：合并出边和入边
        in_nbs = self.reverse_adj.get(str(node_id), [])
        return list(set(out_nbs) | set(in_nbs))

    def _get_similarity(self, center_id: int, node_id: int) -> float:
        """计算节点特征与中心节点的余弦相似度。"""
        f1 = np.array(self.graph_data[str(center_id)]["feature"], dtype=np.float32)
        f2 = np.array(self.graph_data[str(node_id)]["feature"], dtype=np.float32)
        denom = (np.linalg.norm(f1) * np.linalg.norm(f2))
        if denom == 0:
            return 0.0
        return float(np.dot(f1, f2) / denom)

    def get_node_degree_info(self, node_id: int) -> Dict[str, int]:
        """计算节点的出入度及 1-hop/2-hop 邻居数量统计。"""
        node_str = str(node_id)
        info = self._get_node_info(node_id)
        neighbors_1hop = info["neighbors"]
        out_degree = len(neighbors_1hop)
        
        in_neighbors = self.reverse_adj.get(node_str, [])
        in_degree = len(in_neighbors)
        
        undirected_1hop = set(neighbors_1hop) | set(in_neighbors)
        if node_id in undirected_1hop:
            undirected_1hop.remove(node_id)
            
        # 统计 2-hop 邻居
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
        color_seed: int = 42,
        rank_mode: str = "hop", # 'sim' (纯相似度) 或 'hop' (层级优先)
        undirected: bool = True
    ) -> Tuple[bytes, Dict[str, str]]:

        # 1. 节点候选集获取 (BFS)
        center_info = self._get_node_info(center_id)
        neighbors_1hop = self._get_neighbors(center_id, undirected)
        
        candidates = []
        if mode == "1-hop":
            candidates = neighbors_1hop
        elif "2-hop" in mode:
            candidates.extend(neighbors_1hop)
            for nb in neighbors_1hop:
                candidates.extend(self._get_neighbors(nb, undirected))
        
        # 去重、排除中心节点
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen and c != center_id:
                seen.add(c)
                unique_candidates.append(c)

        one_hop_set = set(neighbors_1hop)

        # 预计算相似度缓存
        sim_cache = {
            nid: self._get_similarity(center_id, nid)
            for nid in unique_candidates
        }

        # 2. 节点截断策略
        if rank_mode == "sim":
            # 模式 A：全局按特征相似度排序
            unique_candidates = sorted(
                unique_candidates,
                key=lambda nid: sim_cache[nid],
                reverse=True
            )

        elif rank_mode == "hop":
            # 模式 B：层级优先 (1-hop > 2-hop)，同层级内按相似度排序
            one_hop = [n for n in unique_candidates if n in one_hop_set]
            two_hop = [n for n in unique_candidates if n not in one_hop_set]

            one_hop = sorted(one_hop, key=lambda nid: sim_cache[nid], reverse=True)
            two_hop = sorted(two_hop, key=lambda nid: sim_cache[nid], reverse=True)

            unique_candidates = one_hop + two_hop

        else:
            raise ValueError(f"Unknown rank_mode: {rank_mode}")
        
        budget = max(0, max_nodes - 1)
        final_nodes = [center_id] + unique_candidates[:budget]
        final_nodes_set = set(final_nodes)
        one_hop_set = set(u for u in one_hop_set if u in final_nodes_set)

        # =========================================================
        # 【STEP A】2-hop 强制树化 (Visual Tree-ification)
        # =========================================================

        two_hop_anchor = {}

        for nid in final_nodes:
            if nid == center_id or nid in one_hop_set:
                continue

            # 查找其在 1-hop 中的潜在父节点
            candidates = [
                nb for nb in self._get_neighbors(nid, undirected)
                if nb in one_hop_set
            ]

            if not candidates:
                continue

            # 选取最相似的 1-hop 作为唯一连线对象
            anchor = max(
                candidates,
                key=lambda x: sim_cache.get(x, 0.0)
            )
            two_hop_anchor[nid] = anchor

        # =========================================================
        # 3. 建图 (NetworkX)
        # =========================================================

        G = nx.Graph()
        G.add_nodes_from(final_nodes)

        # 统计当前视图涉及的类别，用于分配颜色
        active_classes = set()
        for u in final_nodes:
            if u != center_id:
                active_classes.add(self._get_node_info(u)["pred_class"] or "Unknown")

        # 边 A: Center -> 1-hop
        for u in one_hop_set:
            if u in final_nodes_set:
                G.add_edge(center_id, u)

        # 边 B: 1-hop -> 2-hop (通过 Anchor 连接)
        for child, anchor in two_hop_anchor.items():
            if child in final_nodes_set and anchor in final_nodes_set:
                G.add_edge(anchor, child)

        # 边 C: 弱连接 (可选)
        TWO_HOP_EDGE_WEIGHT = 0.2   
        for u in final_nodes:
            if u == center_id or u in one_hop_set: continue
            for v in self._get_neighbors(u, undirected):
                if v == center_id: continue
                if v not in final_nodes_set: continue
                if v in one_hop_set: continue          # 不回连 1-hop
                if u >= v: continue                    # 防止重复

                G.add_edge(u, v, weight=TWO_HOP_EDGE_WEIGHT)


        # 4. 布局与渲染
        # 获取颜色映射 (Episode 级别一致)
        sorted_classes = sorted(list(active_classes))
        color_mapping = self._get_color_map_for_episode(sorted_classes, color_seed)

        # --- 布局初始化 (Bias Initialization) ---
        pos_init = {}
        pos_init[center_id] = np.array([0.0, 0.0])

        rng = np.random.RandomState(42)

        R_MIN, R_MAX = 0.6, 1.8 
        for u in one_hop_set:
            sim = sim_cache.get(u, 0.0)
            sim = (sim + 1.0) / 2.0          # 归一化 [-1,1] -> [0,1]
            sim = np.clip(sim, 0.0, 1.0)

            # 视觉编码：相似度越高，距离中心越近 (半径越小)
            r = R_MIN + (1.0 - sim) * (R_MAX - R_MIN)
            theta = rng.uniform(0, 2 * np.pi)
            pos_init[u] = np.array([r * np.cos(theta), r * np.sin(theta)])

        # 使用 Fruchterman-Reingold 算法微调布局
        pos = nx.spring_layout(
            G,
            weight="weight",
            pos=pos_init,
            seed=42,
            iterations=100
        )

        # 归一化坐标中心
        center_xy = pos[center_id].copy()
        for k in pos:
            pos[k] = pos[k] - center_xy

        # 准备绘图属性
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

        # =========================================================
        # 【修改点】 图像尺寸控制
        # =========================================================
        
        # 1. 计算基于节点数量的理论尺寸
        calculated_size = self.BASE_FIG_SIZE + len(final_nodes) * self.NODE_SCALE_FACTOR
        
        # 2. 强制限制在 [MIN, MAX] 范围内
        final_fig_size = np.clip(calculated_size, self.MIN_FIG_SIZE, self.MAX_FIG_SIZE)
        
        # 3. 设置 Figure 尺寸和 DPI
        fig = plt.figure(figsize=(final_fig_size, final_fig_size), dpi=self.DPI)

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

        # 自动调整视口范围
        all_pos = np.array(list(pos.values()))
        xmin, ymin = all_pos.min(axis=0)
        xmax, ymax = all_pos.max(axis=0)
        pad = 0.05
        dx, dy = xmax - xmin, ymax - ymin
        if dx == 0: dx = 1e-3
        if dy == 0: dy = 1e-3
        plt.xlim(xmin - pad * dx, xmax + pad * dx)
        plt.ylim(ymin - pad * dy, ymax + pad * dy)

        # 5. 保存并输出
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=self.DPI  # 确保保存时使用指定的 DPI
        )
        plt.close(fig)
        
        buf.seek(0)
        image_bytes = buf.getvalue()

        return image_bytes, legend_dict