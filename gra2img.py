import torch
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import os
import multiprocessing
import random
import hashlib
import traceback
import time
import datetime
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from torch_geometric.utils import (
    k_hop_subgraph,
    subgraph,
    to_undirected,
    remove_self_loops
)

# ================= 配置 =================

WORK_DIR = "./"
ORIGIN_DIR = WORK_DIR
OUTPUT_DIR = os.path.join(WORK_DIR, "dataset")
PT_FILE_PATH = os.path.join(ORIGIN_DIR, "graph_data_all.pt")

TARGET_DATASETS = ["pubmed"]
NUM_PROCESSES = 72
BATCH_WRITE_SIZE = 100

MAX_GRAPH_NEIGHBORS = 50
MAX_CENTER_TOKENS = 500
HARD_NEIGHBOR_LIMIT = 800

BASE_FIG_SIZE = 10
NODE_SCALE_FACTOR = 0.05

global_data = {}

# ================= 工具 =================

def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def stable_hash(s):
    if s is None:
        s = "Unknown"
    if not isinstance(s, str):
        s = str(s)
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

# ================= 数据加载 =================

def load_dataset_splits(pt_path, dataset):
    raw = torch.load(pt_path, weights_only=False)
    data = raw[dataset] if isinstance(raw, dict) else raw

    x = data.x
    y = data.y.squeeze(-1).long()

    edge_index = data.edge_index
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)

    def get_idx(mask):
        if not hasattr(data, mask):
            return None
        m = getattr(data, mask)
        return m.nonzero(as_tuple=False).view(-1)

    splits = {
        "train": get_idx("train_mask"),
        "val": get_idx("val_mask"),
        "test": get_idx("test_mask")
    }
    return x, edge_index, y, splits

# ================= Worker 初始化 =================

def init_worker(x, edge_index, y, tm, lm, proxy_map, split_name):
    global global_data
    global_data.update({
        "x": x,
        "edge_index": edge_index,
        "y": y,
        "text_map": tm,
        "label_map": lm,
        "proxy_label_map": proxy_map,
        "split_name": split_name
    })

    # 固定调色板（tab10 顺序稳定）
    cmap = plt.get_cmap("tab10")
    global_data["cmap"] = cmap

    # 与 tab10 对齐的颜色名（不包含中心节点颜色）
    global_data["color_names"] = [
        "Blue", "Orange", "Green", "Red", "Purple",
        "Brown", "Pink", "Gray", "Olive", "Cyan"
    ]

# ================= 文本 / 标签 =================

def get_node_text(node_id, max_tokens=None):
    tm = global_data["text_map"]
    text = tm.get(str(node_id), "No text.")
    text = text.replace("\n", " ")
    if max_tokens:
        toks = text.split()
        if len(toks) > max_tokens:
            text = " ".join(toks[:max_tokens]) + "..."
    return text

def get_label_text(label_id):
    lm = global_data["label_map"]
    return lm.get(str(int(label_id)), f"Label_{int(label_id)}")

def get_proxy_label_text(node_id):
    pm = global_data["proxy_label_map"]
    entry = pm.get(str(node_id))
    if not isinstance(entry, dict):
        return "Unknown"
    v = entry.get("top1")
    return v if isinstance(v, str) else "Unknown"

# ================= 图像生成 =================

def generate_image_bytes(center_id, nodes, edge_index):
    """
    nodes: Tensor of node ids (not relabeled)
    edge_index: Tensor [2, E] with node ids aligned to original graph
    """
    G = nx.Graph()
    node_list = nodes.tolist()
    G.add_nodes_from(node_list)

    row, col = edge_index
    # edge_index here is shape [2, E] in PyG style; after subgraph we keep [2, E]
    # but in our code we sometimes pass as (row, col) already
    # ensure row/col are 1D tensors
    for i in range(row.size(0)):
        u = row[i].item()
        v = col[i].item()
        if u in G and v in G:
            G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42)

    fig = plt.figure(
        figsize=(BASE_FIG_SIZE + len(nodes) * NODE_SCALE_FACTOR,
                 BASE_FIG_SIZE + len(nodes) * NODE_SCALE_FACTOR)
    )

    cmap = global_data["cmap"]
    color_names = global_data["color_names"]

    edgecolors = []
    sizes = []
    legend = {}
    counter = Counter()

    for nid in node_list:
        if nid == center_id:
            # ✅ 修复点 B：中心节点用 Black，避免与 tab10 的 Red 冲突
            edgecolors.append("black")
            sizes.append(1200)
            legend["Black"] = "Center Node"
            continue

        proxy_label = get_proxy_label_text(nid)
        c_idx = stable_hash(proxy_label) % cmap.N
        color = cmap(c_idx)
        color_name = color_names[c_idx]

        edgecolors.append(color)
        sizes.append(800)

        # 颜色名 -> 语义类（proxy）
        if color_name not in legend:
            legend[color_name] = proxy_label

        counter[proxy_label] += 1

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

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    total = sum(counter.values())
    # ✅ 避免除 0（极端情况下邻居为 0）
    if total == 0:
        color_distribution = {}
    else:
        color_distribution = {k: round(v / total, 4) for k, v in counter.items()}

    legend_str = "; ".join([f"{k}: {v}" for k, v in legend.items()])
    return buf.getvalue(), legend_str, json.dumps(color_distribution)

# ================= 单节点处理 =================

def process_single_node(node_id: int):
    try:
        edge_index = global_data["edge_index"]
        y = global_data["y"]

        # =========================================================
        # Step 1) 1-hop 必须从“原始 edge_index”计算（不受任何截断影响）
        # =========================================================
        row0, col0 = edge_index
        one_hop_full = set(
            torch.cat([col0[row0 == node_id], row0[col0 == node_id]]).tolist()
        ) - {node_id}

        # =========================================================
        # Step 2) 取 2-hop 子图（不 relabel）
        # =========================================================
        subset_2hop, _, _, _ = k_hop_subgraph(
            node_id, 2, edge_index, relabel_nodes=False
        )
        subset_set = set(subset_2hop.tolist())

        # 只保留在 2-hop 范围内的 1-hop（理论上都在，但保险）
        one_hop = [n for n in one_hop_full if n in subset_set]

        # =========================================================
        # ✅ 修复点 A：如果 1-hop 本身就超过预算，必须裁掉到 budget 内
        # =========================================================
        max_one_hop = HARD_NEIGHBOR_LIMIT - 1
        if len(one_hop) > max_one_hop:
            # 保证 1-hop 仍然非空且不会爆预算
            one_hop = random.sample(one_hop, max_one_hop)

        # =========================================================
        # Step 3) 截断：强制保留 center + (裁过的) 1-hop，再补 2-hop 其它
        # =========================================================
        if subset_2hop.size(0) > HARD_NEIGHBOR_LIMIT:
            remaining_budget = HARD_NEIGHBOR_LIMIT - 1 - len(one_hop)
            remaining_budget = max(0, remaining_budget)

            others = [n for n in subset_2hop.tolist() if n != node_id and n not in set(one_hop)]
            sampled_others = random.sample(others, min(len(others), remaining_budget))

            subset = torch.tensor([node_id] + list(one_hop) + sampled_others, dtype=torch.long)
        else:
            subset = subset_2hop

        # 最终 subset 构边（此时 center 的 1-hop 边不会“被自己截断剪掉”）
        sub_edge, _ = subgraph(subset, edge_index, relabel_nodes=False)

        # =========================================================
        # Step 4) 在最终 sub_edge 上标记 is_1hop（用于 action space）
        # =========================================================
        row, col = sub_edge
        one_hop_final = set(
            torch.cat([col[row == node_id], row[col == node_id]]).tolist()
        ) - {node_id}

        # =========================================================
        # Step 5) 选择可视化/可检索节点：强制 1-hop 优先
        # =========================================================
        candidates = []
        for nid in subset.tolist():
            if nid == node_id:
                continue
            candidates.append({"id": nid, "is_1hop": (nid in one_hop_final)})

        # 1-hop 优先，其它在后；同类内部按 id 稳定排序，便于复现
        candidates.sort(key=lambda c: (not c["is_1hop"], c["id"]))

        top_c = candidates[:MAX_GRAPH_NEIGHBORS]

        final_nodes = torch.tensor([node_id] + [c["id"] for c in top_c], dtype=torch.long)
        final_edge, _ = subgraph(final_nodes, edge_index, relabel_nodes=False)

        img_b, legend_str, color_dist = generate_image_bytes(
            node_id, final_nodes, final_edge
        )

        inspectable = {
            "1hop": [c["id"] for c in top_c if c["is_1hop"]],
            "2hop": [c["id"] for c in top_c if not c["is_1hop"]]
        }

        return {
            "center_id": int(node_id),
            "center_text": get_node_text(node_id, MAX_CENTER_TOKENS),
            "image_bytes": img_b,
            "legend": legend_str,
            "color_distribution": color_dist,
            "inspectable_nodes": json.dumps(inspectable),
            "answer": get_label_text(y[node_id]),
            "split": global_data["split_name"]
        }

    except Exception:
        print(f"Error at node {node_id}")
        traceback.print_exc()
        return None

# ================= 主流程 =================

def process_and_save(dataset, split, indices, x, edge_index, y, tm, lm, proxy_map):
    if indices is None:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{dataset}_{split}.parquet")

    schema = pa.schema([
        ("center_id", pa.int64()),
        ("center_text", pa.string()),
        ("image_bytes", pa.binary()),
        ("legend", pa.string()),
        ("color_distribution", pa.string()),
        ("inspectable_nodes", pa.string()),
        ("answer", pa.string()),
        ("split", pa.string())
    ])

    writer = pq.ParquetWriter(out_path, schema)

    pool = multiprocessing.Pool(
        NUM_PROCESSES,
        initializer=init_worker,
        initargs=(x, edge_index, y, tm, lm, proxy_map, split)
    )

    batch = []
    total = len(indices)
    count = 0
    start_time = time.time()

    for res in pool.imap_unordered(process_single_node, indices.tolist()):
        count += 1
        if res:
            batch.append(res)

        if len(batch) >= BATCH_WRITE_SIZE:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            batch.clear()

        if count % 50 == 0 or count == total:
            elapsed = time.time() - start_time
            speed = count / elapsed if elapsed > 0 else 0.0
            eta = (total - count) / speed if speed > 0 else 0.0
            print(
                f"\r[{dataset} | {split}] "
                f"{count}/{total} "
                f"({count/total*100:.1f}%) | "
                f"{speed:.2f} it/s | "
                f"ETA {eta/60:.1f} min",
                end="",
                flush=True
            )

    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))

    print()
    writer.close()
    pool.close()
    pool.join()

def main():
    for dataset in TARGET_DATASETS:
        x, edge_index, y, splits = load_dataset_splits(PT_FILE_PATH, dataset)

        with open(f"{dataset}_text.json", "r", encoding="utf-8") as f:
            tm = json.load(f)
        with open(f"{dataset}_label_id_to_text.json", "r", encoding="utf-8") as f:
            lm = json.load(f)
        with open(f"{dataset}_proxy_class.json", "r", encoding="utf-8") as f:
            proxy_map = json.load(f)

        # 多进程 spawn 下：共享内存避免重复拷贝
        x = x.share_memory_()
        edge_index = edge_index.share_memory_()
        y = y.share_memory_()

        for split in ["train", "val", "test"]:
            process_and_save(
                dataset, split, splits[split],
                x, edge_index, y, tm, lm, proxy_map
            )

    log("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
