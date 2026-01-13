import pyarrow.parquet as pq
import pandas as pd

def inspect_parquet(file_path, num_rows=5):
    print(f"=== 正在检查文件: {file_path} ===")
    
    # 1. 读取元数据（不加载内容，瞬间完成）
    parquet_file = pq.ParquetFile(file_path)
    print(f"总行数: {parquet_file.metadata.num_rows}")
    print(f"Row Groups数量: {parquet_file.num_row_groups}")
    
    # 2. 打印 Schema (非常重要，用于检查数据类型是否符合 DataProto 要求)
    print("\n--- 列信息 (Schema) ---")
    print(parquet_file.schema)

    # 3. 只读取第一批数据 (内存安全)
    # iter_batches 会返回一个迭代器，next() 取出第一块
    first_batch = next(parquet_file.iter_batches(batch_size=num_rows))
    df = first_batch.to_pandas()
    
    print(f"\n--- 前 {num_rows} 行预览 ---")
    # 设置 pandas 显示选项，避免列过多被折叠
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df)

# 替换成你的文件路径
file_path = "/Users/sianeko/vscode/verl-agent/datasets/pubmed_test_slim.parquet" 
try:
    inspect_parquet(file_path)
except Exception as e:
    print(f"读取失败: {e}")