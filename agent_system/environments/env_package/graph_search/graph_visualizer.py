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
            
        # 缓存所有节点ID，用于 sim 搜索
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
        """根据 undirected 参数，返回邻居集合"""
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
        """获取中心节点周围（或相似）节点的类别列表，用于 Prompt 提示"""
        # 这里简单使用 BFS 扩展 100 个节点，或者直接用 1-hop + 2-hop
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

    def _select_nodes(self, center_id: int, view_mode: str, max_nodes: int, undirected: bool) -> List[int]:
        """
        根据 view_mode 选择节点列表。
        策略：
        - center: 仅中心
        - 1-hop: 仅1阶。若超限，按 sim 截断。
        - 2-hop: 1阶+2阶。若超限，按 sim 截断。
        - sim: 全局相似度最高的节点。
        - 1-hop+sim: 优先1阶，不足填 sim。
        - 2-hop+sim: 优先1阶+2阶，不足填 sim。
        """
        if view_mode == "center":
            return [center_id]
        
        # 1. 基础集合获取
        neighbors_1hop = set(self._get_neighbors(center_id, undirected))
        if center_id in neighbors_1hop: neighbors_1hop.remove(center_id)
        
        neighbors_2hop = set()
        if "2-hop" in view_mode:
            for nb in neighbors_1hop:
                nbs = self._get_neighbors(nb, undirected)
                for nb2 in nbs:
                    if nb2 != center_id and nb2 not in neighbors_1hop:
                        neighbors_2hop.add(nb2)

        # 2. 预计算相似度 (Lazy calc could be better but dataset is usually small enough)
        # 为了 'sim' 模式和填充逻辑，我们可能需要更广的候选集
        # 如果是纯 hop 模式，只计算局部相似度即可
        
        candidates = set()
        if view_mode in ["1-hop", "2-hop"]:
            candidates.update(neighbors_1hop)
            candidates.update(neighbors_2hop)
            global_search = False
        else:
            # 涉及 sim 填充，简单起见，我们在当前图的所有节点中搜（如果图太大，需优化为随机采样或 TopK 索引）
            # 这里假设 Cora/Pubmed 规模，全量计算尚可，或者取一个大范围 BFS
            # 为了性能，这里我们取 "1-hop + 2-hop + Random Sample" 或者如果图小就全量
            # 稳妥起见，我们暂且认为候选集是 1hop + 2hop + 3hop 或者 全量
            # 这里简化：若需要 sim 补充，则对所有节点计算 sim (注意性能)
            global_search = True

        def get_sim(nid):
            return self._get_similarity(center_id, nid)

        selected = []
        budget = max_nodes

        # --- 核心筛选逻辑 ---

        if view_mode == "1-hop":
            pool = list(neighbors_1hop)
            pool.sort(key=get_sim, reverse=True)
            selected = pool[:budget]

        elif view_mode == "2-hop":
            pool = list(neighbors_1hop | neighbors_2hop)
            pool.sort(key=get_sim, reverse=True)
            selected = pool[:budget]

        elif view_mode == "sim":
            # 全局 Top K Sim
            # 优化：只对所有节点算一次，或者采样。此处全量扫描。
            all_nodes = [int(n) for n in self.all_node_ids if int(n) != center_id]
            # 为了速度，如果是非常大的图，建议预先算好 embedding 索引。
            # 这里简单实现：全量排序
            # 如果太慢，可以只搜索 3-hop 内
            all_nodes.sort(key=get_sim, reverse=True)
            selected = all_nodes[:budget]

        elif view_mode == "1-hop+sim":
            # 优先 1-hop
            pool_1 = list(neighbors_1hop)
            pool_1.sort(key=get_sim, reverse=True)
            
            if len(pool_1) >= budget:
                selected = pool_1[:budget]
            else:
                selected = pool_1[:]
                remain = budget - len(selected)
                # 补充 sim
                all_others = [int(n) for n in self.all_node_ids 
                              if int(n) != center_id and int(n) not in neighbors_1hop]
                all_others.sort(key=get_sim, reverse=True)
                selected.extend(all_others[:remain])

        elif view_mode == "2-hop+sim":
            # 优先 1+2 hop
            pool_local = list(neighbors_1hop | neighbors_2hop)
            pool_local.sort(key=get_sim, reverse=True)
            
            if len(pool_local) >= budget:
                selected = pool_local[:budget]
            else:
                selected = pool_local[:]
                remain = budget - len(selected)
                local_set = set(pool_local)
                # 补充 sim
                all_others = [int(n) for n in self.all_node_ids 
                              if int(n) != center_id and int(n) not in local_set]
                all_others.sort(key=get_sim, reverse=True)
                selected.extend(all_others[:remain])
        
        return [center_id] + selected

    def draw_subgraph(
        self,
        center_id: int,
        view_mode: str = "1-hop",  # 1-hop, 2-hop, sim, 1-hop+sim, 2-hop+sim, center
        max_nodes: int = 10,
        color_seed: int = 42,
        undirected: bool = True
    ) -> Tuple[bytes, Dict[str, str]]:
        
        # 1. 筛选节点
        final_nodes = self._select_nodes(center_id, view_mode, max_nodes, undirected)
        final_nodes_set = set(final_nodes)
        
        # 2. 建图 (Induced Subgraph + 必要的视觉辅助边)
        G = nx.Graph()
        G.add_nodes_from(final_nodes)

        # 统计类别
        active_classes = set()
        for u in final_nodes:
            if u != center_id:
                active_classes.add(
                    self._get_node_info(u)["pred_class"] or "Unknown"
                )

        # 加边策略：
        # 为了让图看起来连贯，我们添加所有选中节点之间的真实边
        # 如果 view_mode 是 sim，可能会有孤立点，这是符合预期的（说明虽然相似但无连接）
        
        # 优化：只遍历 final_nodes 中的节点对，或者遍历 final_nodes 的邻居是否在 final_nodes 中
        for u in final_nodes:
            nbs = self._get_neighbors(u, undirected)
            for v in nbs:
                if v in final_nodes_set and u < v: # 避免重复
                     G.add_edge(u, v)
        
        # 3. 布局与配色
        sorted_classes = sorted(list(active_classes))
        color_mapping = self._get_color_map_for_episode(sorted_classes, color_seed)

        # 计算 Layout 初始位置 (基于相似度)
        pos_init = {}
        pos_init[center_id] = np.array([0.0, 0.0])
        rng = np.random.RandomState(42)

        sim_cache = {u: self._get_similarity(center_id, u) for u in final_nodes if u != center_id}
        R_MIN, R_MAX = 0.5, 2.0

        for u in final_nodes:
            if u == center_id: continue
            
            sim = sim_cache.get(u, 0.0)
            # 简单的极坐标映射：相似度高 -> 距离近
            dist = R_MIN + (1.0 - sim) * (R_MAX - R_MIN)
            theta = rng.uniform(0, 2 * np.pi)
            pos_init[u] = np.array([dist * np.cos(theta), dist * np.sin(theta)])

        # 使用 spring_layout 调整，保留拓扑结构特征
        # k 参数控制节点间距，节点越少间距应越大
        k_val = 1.0 / np.sqrt(len(final_nodes)) if len(final_nodes) > 0 else 1.0
        pos = nx.spring_layout(G, pos=pos_init, seed=42, k=k_val, iterations=50)

        # 中心归零
        if center_id in pos:
            c_xy = pos[center_id]
            for n in pos:
                pos[n] -= c_xy

        # 4. 绘图样式
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

        # 动态画布大小
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
        
        # 裁切与保存
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        
        buf.seek(0)
        return buf.getvalue(), legend_dict