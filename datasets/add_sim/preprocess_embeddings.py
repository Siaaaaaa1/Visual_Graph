import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---- 网络下载配置 ----
# 优先使用 HF_ENDPOINT 环境变量；若未设置，自动切换到国内镜像（hf-mirror.com）
# 若已将模型下载到本地，将 LOCAL_MODEL_PATH 设置为本地目录路径可完全跳过网络
# 手动预下载命令（只需执行一次）:
#   pip install -U huggingface_hub
#   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-base-en-v1.5 --local-dir ./models/bge-base-en-v1.5
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def main():
    # ================= 配置区 =================
    # 本地路径优先：若下载过模型，填写本地目录；否则留空，自动从镜像下载
    LOCAL_MODEL_PATH = ""   # 例: "./models/bge-base-en-v1.5"
    MODEL_NAME = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH else 'BAAI/bge-base-en-v1.5'
    DATASET_DIR = "./datasets/add_sim"
    OUTPUT_DIR = "./datasets"
    DATASETS = ["cora", "pubmed", "arxiv"]
    BATCH_SIZE = 256 # 如果显存不够（如 Mac 的 MPS），可以调小到 64 或 128
    # ==========================================

    # 1. 自动检测计算设备 (支持 Nvidia CUDA, Apple Silicon MPS, 或 CPU)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"🚀 [Step 1] Loading Embedding Model [{MODEL_NAME}] on [{device.upper()}]...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # 仅在 CUDA 下开启 FP16 加速
    if device == 'cuda':
        model.half()

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 遍历处理每个数据集
    for dataset_name in DATASETS:
        input_file = os.path.join(DATASET_DIR, f"{dataset_name}.json")
        output_file = os.path.join(OUTPUT_DIR, f"{dataset_name}.json")
        
        if not os.path.exists(input_file):
            print(f"⚠️ [Warning] Dataset file not found: {input_file}. Skipping...")
            continue

        print(f"\n📂 [Step 2] Processing dataset: {dataset_name.upper()}")
        print(f"   Reading from: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        nodes = dataset.get("nodes", [])
        if not nodes:
            print(f"   ❌ Error: No nodes found in {dataset_name}.")
            continue

        # 提取需要编码的文本
        texts_to_encode = []
        for node in nodes:
            # 你的数据集中文本字段名为 "text"，包含了 title 和 abstract
            raw_text = node.get("text", "")
            texts_to_encode.append(raw_text)

        print(f"   🧠 Encoding {len(texts_to_encode)} nodes...")
        
        # 编码并归一化 (L2 Normalize 保证后续可以直接算余弦相似度)
        embeddings = model.encode(
            texts_to_encode, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )

        print(f"   💾 Injecting features and saving to {output_file}...")
        # 覆盖原有的 feature
        for i, node in enumerate(nodes):
            node["feature"] = embeddings[i].tolist()

        # 写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 保持紧凑格式，防止文件过大
            json.dump(dataset, f, ensure_ascii=False)

        print(f"   ✅ {dataset_name.upper()} processing finished!")

    print("\n🎉 All datasets have been successfully processed with BGE embeddings!")

if __name__ == "__main__":
    main()