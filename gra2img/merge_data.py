import torch
import json
import os
from tqdm import tqdm

# ================= 配置路径 =================
# 假设 .pt 文件名为 graph_data_all.pt，如果不是请修改
PT_FILE_PATH = "./origin_datasets/graph_data_all.pt" 

# 目标处理的数据集名称
# 注意：截图里是 Industrial，但你要求处理 arxiv。
# 请确保 .pt 文件里包含 'arxiv' 这个 key，或者根据实际情况修改列表。
TARGET_DATASETS = ["cora", "pubmed", "arxiv"] 

OUTPUT_DIR = "./datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def tensor_to_py(obj):
    """辅助函数：将 Tensor 转换为 Python 原生类型"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    return obj

def load_json_safe(path):
    """安全加载 JSON，如果文件不存在返回空字典"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"Warning: {path} not found.")
    return {}

def process_dataset(dataset_name, data_obj):
    print(f"Processing {dataset_name}...")
    
    # 1. 加载辅助 JSON 文件
    text_map = load_json_safe(f"./origin_datasets/{dataset_name}_text.json")
    label_map = load_json_safe(f"./origin_datasets/{dataset_name}_label_id_to_text.json")
    proxy_map = load_json_safe(f"./origin_datasets/{dataset_name}_proxy_class.json")
    
    # 2. 获取基础数据
    # PyG data object usually stores x, y, edge_index, masks
    # 注意：为了减小 JSON 体积，通常不保存 float32 的 feature (x)
    # 除非 Agent 明确需要读取浮点向量，否则这里只存语义信息
    
    num_nodes = data_obj.num_nodes if hasattr(data_obj, 'num_nodes') else data_obj.x.shape[0]
    
    # 处理 Mask (Train/Val/Test)
    # 创建一个 split 映射: index -> "train" | "val" | "test" | "none"
    node_splits = ["none"] * num_nodes
    
    # 辅助函数处理 mask
    def apply_mask(mask_tensor, split_name):
        if mask_tensor is not None:
            # 获取为 True 的索引
            indices = mask_tensor.nonzero(as_tuple=False).view(-1).tolist()
            for idx in indices:
                node_splits[idx] = split_name

    if hasattr(data_obj, 'train_mask'): apply_mask(data_obj.train_mask, "train")
    if hasattr(data_obj, 'val_mask'):   apply_mask(data_obj.val_mask, "val")
    if hasattr(data_obj, 'test_mask'):  apply_mask(data_obj.test_mask, "test")

    # 3. 处理边信息 (Edge Index -> Adjacency List)
    # 对于 Agent 来说，知道“邻居是谁”比知道全局 Edge Index 更重要
    print(f"  - Building adjacency list for {num_nodes} nodes...")
    adj = {i: [] for i in range(num_nodes)}
    if hasattr(data_obj, 'edge_index') and data_obj.edge_index is not None:
        # 确保 edge_index 是 long 类型并在 CPU 上
        ei = data_obj.edge_index.long().cpu()
        src, dst = ei[0].tolist(), ei[1].tolist()
        for u, v in zip(src, dst):
            adj[u].append(v)
            # 如果是无向图且 .pt 里只存了单向，可能需要反向添加，
            # 但通常 PyG 的 undirected edge_index 已经包含了双向。
    
    # 4. 组装节点数据
    nodes_list = []
    labels_tensor = data_obj.y if hasattr(data_obj, 'y') else None
    
    print(f"  - Assembling node details...")
    for i in tqdm(range(num_nodes), desc=f"Nodes ({dataset_name})"):
        str_id = str(i)
        
        # 获取标签 ID 和 文本名称
        label_id = None
        label_text = "Unknown"
        if labels_tensor is not None:
            label_id = tensor_to_py(labels_tensor[i])
            # 尝试从 label_map 映射，注意 map 的 key 可能是 string
            label_text = label_map.get(str(label_id), str(label_id))
            
        node_item = {
            "id": i,
            "split": node_splits[i],
            "label_id": label_id,
            "label_name": label_text,
            # 从 text.json 获取文本
            "text": text_map.get(str_id, ""),
            # 从 proxy.json 获取额外信息
            "proxy_info": proxy_map.get(str_id, {}),
            # 邻居列表
            "neighbors": adj[i]
        }
        nodes_list.append(node_item)
        
    # 5. 构建最终结构
    final_output = {
        "dataset": dataset_name,
        "total_nodes": num_nodes,
        "classes_map": label_map,
        "nodes": nodes_list
    }
    
    # 6. 保存
    out_file = os.path.join(OUTPUT_DIR, f"{dataset_name}.json")
    print(f"  - Saving to {out_file}...")
    with open(out_file, 'w', encoding='utf-8') as f:
        # indent=2 为了可读性，如果文件太大（如arxiv），建议改为 indent=None
        json.dump(final_output, f, ensure_ascii=False, indent=2) 
    print(f"Finished {dataset_name}.")

def main():
    if not os.path.exists(PT_FILE_PATH):
        print(f"Error: {PT_FILE_PATH} not found.")
        return

    print("Loading PT file (this might take a while)...")
    # weights_only=False 是为了兼容旧版 PyTorch 保存的复杂结构
    # 如果报错，尝试去掉 weights_only 参数
    try:
        all_data = torch.load(PT_FILE_PATH, weights_only=False)
    except TypeError:
        all_data = torch.load(PT_FILE_PATH)

    # 检查 .pt 是字典还是单个 Data 对象
    # 截图显示包含多个数据集，应该是 Dict[str, Data]
    
    for ds in TARGET_DATASETS:
        if isinstance(all_data, dict):
            if ds not in all_data:
                print(f"Dataset '{ds}' not found in .pt file keys: {list(all_data.keys())}")
                # 兼容截图情况：如果用户想要处理 Industrial 也可以加进去
                continue
            data_obj = all_data[ds]
        else:
            # 如果 .pt 里只有一个数据对象
            print("PT file seems to contain a single dataset.")
            data_obj = all_data
            
        process_dataset(ds, data_obj)

if __name__ == "__main__":
    main()