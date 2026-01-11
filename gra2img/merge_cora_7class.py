import torch
import json
import os
from tqdm import tqdm

# ================= 配置 =================
PT_PATH = "./origin_datasets/cora.pt"
PROXY_PATH = "./origin_datasets/cora_proxy_class.json"
OUTPUT_DIR = "./datasets"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cora.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 工具函数 =================
def load_json_safe(path):
    if path is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ================= 主逻辑 =================
def main():
    # ---------- 1. 读取数据 ----------
    data = torch.load(PT_PATH)
    proxy_map = load_json_safe(PROXY_PATH)

    num_nodes = data.num_nodes
    node_features = data.x.detach().cpu().to(torch.float16)

    # ---------- 2. 构建 split ----------
    node_splits = ["none"] * num_nodes

    if hasattr(data, "train_mask"):
        for i in data.train_mask.nonzero(as_tuple=False).view(-1).tolist():
            node_splits[i] = "train"

    if hasattr(data, "val_mask"):
        for i in data.val_mask.nonzero(as_tuple=False).view(-1).tolist():
            node_splits[i] = "val"

    if hasattr(data, "test_mask"):
        for i in data.test_mask.nonzero(as_tuple=False).view(-1).tolist():
            node_splits[i] = "test"

    # ---------- 3. 构建邻接表 ----------
    adj = {i: [] for i in range(num_nodes)}
    edge_index = data.edge_index.long().cpu()
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        adj[u].append(v)

    # ---------- 4. classes_map（严格保持原结构） ----------
    # 等价于原来的 label_map.json
    label_map = {str(i): name for i, name in enumerate(data.label_texts)}

    # ---------- 5. 构建 nodes_list（结构完全一致） ----------
    nodes_list = []

    for i in tqdm(range(num_nodes), desc="Processing nodes"):
        label_id = int(data.y[i])
        label_text = data.label_texts[label_id]
        str_id = str(i)

        node_item = {
            "id": i,
            "split": node_splits[i],
            "label_id": label_id,
            "label_name": label_text,
            # 保持字段名为 text（数据来源改为 pt 内）
            "text": data.raw_texts[i],
            "proxy_info": proxy_map.get(str_id, {}),
            "feature": node_features[i].tolist(),
            "neighbors": adj[i]
        }

        nodes_list.append(node_item)

    # ---------- 6. 构建最终结构（严格一致） ----------
    final_output = {
        "dataset": "cora",
        "total_nodes": num_nodes,
        "classes_map": label_map,
        "nodes": nodes_list
    }

    # ---------- 7. 保存 ----------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_PATH}")

# ================= 入口 =================
if __name__ == "__main__":
    main()
