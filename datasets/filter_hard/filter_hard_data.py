import pandas as pd
import json
import os

def filter_middle_range_datasets():
    # 1. 定义基础路径
    base_input_dir = './datasets/filter_hard/'
    base_output_dir = './datasets/'
    
    # 需要处理的数据集列表
    dataset_names = ['arxiv', 'cora', 'pubmed']
    
    # 确保输出目录存在
    os.makedirs(base_output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"开始处理: 剔除简单(Easy)和困难(Hard)样本，仅保留中间(Middle)样本")
    print(f"处理数据集: {', '.join(dataset_names)}")
    print(f"{'='*60}\n")

    for dataset in dataset_names:
        print(f"正在处理数据集: [{dataset.upper()}]")
        
        # ---------------------------------------------------------
        # 1. 构建文件路径
        # ---------------------------------------------------------
        setting_path = os.path.join(base_input_dir, f'{dataset}_filter_setting.json')
        metrics_path = os.path.join(base_input_dir, f'{dataset}_feature_metrics.csv')
        input_parquet_path = os.path.join(base_input_dir, f'{dataset}_train_slim.parquet')
        output_parquet_path = os.path.join(base_output_dir, f'{dataset}_train_slim.parquet')

        # 检查文件是否存在
        if not os.path.exists(setting_path):
            print(f"  [跳过] 缺少配置文件: {setting_path}")
            continue
        if not os.path.exists(metrics_path):
            print(f"  [跳过] 缺少指标文件: {metrics_path}")
            continue
        if not os.path.exists(input_parquet_path):
            print(f"  [跳过] 缺少数据文件: {input_parquet_path}")
            continue

        # ---------------------------------------------------------
        # 2. 读取配置 (Settings)
        # ---------------------------------------------------------
        try:
            with open(setting_path, 'r') as f:
                settings = json.load(f)
            
            easy_min = settings.get("easy_homophily_min", 0.7)
            hard_max = settings.get("hard_homophily_max", 0)
            
            print(f"  -> 读取配置: 剔除 >= {easy_min} (简单) 和 <= {hard_max} (困难)")
            print(f"  -> 保留区间: ({hard_max} < Value < {easy_min})")
            
        except Exception as e:
            print(f"  [错误] 配置文件读取失败: {e}")
            continue

        # ---------------------------------------------------------
        # 3. 处理 CSV 指标，筛选出"中间态"节点
        # ---------------------------------------------------------
        try:
            metrics_df = pd.read_csv(metrics_path)
            
            col_1hop = 'hom_Full (1-hop)'
            col_limit20 = 'hom_Limit 20'
            
            if col_1hop not in metrics_df.columns or col_limit20 not in metrics_df.columns:
                print(f"  [错误] CSV缺少列: {col_1hop} 或 {col_limit20}")
                continue

            # === 核心逻辑修改 ===
            # 我们要保留的是：既不简单也不难的数据
            # 即：值必须严格大于 hard_max 且 严格小于 easy_min
            def is_middle_range(series):
                return (series > hard_max) & (series < easy_min)

            # 1. 检查 1-hop 是否在中间区间
            mask_1hop = is_middle_range(metrics_df[col_1hop])
            
            # 2. 检查 Limit 20 是否在中间区间
            mask_limit20 = is_middle_range(metrics_df[col_limit20])
            
            # 3. 取交集 (AND)：两个指标都必须是"中间态"才保留
            # 如果某节点在 1-hop 是简单的，但 Limit 20 是中间的 -> 剔除 (因为它包含简单特征)
            final_mask = mask_1hop & mask_limit20
            
            # 获取合法的 node_id
            valid_nodes = set(metrics_df.loc[final_mask, 'node_id'].unique())
            
            kept_ratio = len(valid_nodes) / len(metrics_df) * 100
            print(f"  -> 指标筛选结果: 保留 {len(valid_nodes)} / {len(metrics_df)} 个节点 ({kept_ratio:.2f}%)")
            
        except Exception as e:
            print(f"  [错误] CSV处理失败: {e}")
            continue

        # ---------------------------------------------------------
        # 4. 过滤 Parquet 数据并保存
        # ---------------------------------------------------------
        try:
            df_data = pd.read_parquet(input_parquet_path)
            
            if 'center_id' not in df_data.columns:
                print(f"  [错误] Parquet缺少 'center_id' 列")
                continue
            
            # 过滤
            df_filtered = df_data[df_data['center_id'].isin(valid_nodes)]
            
            # 保存
            df_filtered.to_parquet(output_parquet_path, index=False)
            
            print(f"  -> 数据保存至: {output_parquet_path}")
            print(f"  -> 最终数据量: {len(df_filtered)} (原数据: {len(df_data)})")
            print("-" * 40)

        except Exception as e:
            print(f"  [错误] Parquet处理失败: {e}")

    print("\n所有任务完成。")

if __name__ == "__main__":
    filter_middle_range_datasets()