from datasets import load_dataset

# ======================
# 路径配置
# ======================
INPUT_PARQUET = "./pubmed_train_with_prompt.parquet"
OUTPUT_PARQUET = "./pubmed_train_with_all.parquet"

# ======================
# 占位 prompt（只用于骗过 RLDataset）
# ⚠️ 不会被真正送进 VLM
# ======================
PLACEHOLDER_PROMPT = [
    {
        "role": "user",
        "content": "<image>\nYou are an agent."
    }
]

# ======================
# data_source（给 trainer / protocol 用）
# ======================
DATA_SOURCE = "graph_search"


def process_example(example):
    """
    对每一行样本做最小、必要的修正
    """

    # 1️⃣ 删掉无用字段（避免歧义）
    example.pop("split", None)

    # 2️⃣ 补 prompt（RLDataset 必须）
    if "prompt" not in example or example["prompt"] is None:
        example["prompt"] = PLACEHOLDER_PROMPT

    # 3️⃣ 补 data_source（Trainer / protocol 必须）
    if "data_source" not in example or example["data_source"] is None:
        example["data_source"] = DATA_SOURCE

    return example


def main():
    # 读取 parquet（当成一个 Dataset，用 train split 即可）
    dataset = load_dataset(
        "parquet",
        data_files=INPUT_PARQUET,
        split="train"
    )

    # 处理字段
    dataset = dataset.map(
        process_example,
        desc="Cleaning fields & adding prompt/data_source"
    )

    # 保存新 parquet
    dataset.to_parquet(OUTPUT_PARQUET)
    print(f"✅ Saved cleaned parquet to: {OUTPUT_PARQUET}")

    # 可选：打印一条样本确认
    print("\n🔍 Sample row after processing:")
    sample = dataset[0]
    for k in sample.keys():
        if k in ("image_bytes",):
            print(f"{k}: <bytes>")
        else:
            print(f"{k}: {sample[k]}")


if __name__ == "__main__":
    main()
