import pandas as pd
import json
import os

def process_and_validate_cora():
    # 1. 定义文件路径
    file_path_test = '/mnt/cephfs/haowengao/Visual_Graph/datasets/all_test/cora_test_slim.parquet'
    file_path_train = '/mnt/cephfs/haowengao/Visual_Graph/datasets/filter_hard/cora_train_slim.parquet'
    output_path = '/mnt/cephfs/haowengao/Visual_Graph/datasets/cora_text.json'

    print(f"正在读取文件...")
    try:
        # 读取 Parquet 文件
        df_test = pd.read_parquet(file_path_test)
        df_train = pd.read_parquet(file_path_train)

        # 合并 DataFrame
        df_all = pd.concat([df_test, df_train], ignore_index=True)
        
        # 提取 center_id 并转换为整数用于计算
        # 假设原始数据中 center_id 是数字类型，如果是字符串会自动转换
        all_ids = df_all['center_id'].astype(int).tolist()
        total_rows = len(all_ids)
        
        print(f"--- 数据分析报告 ---")
        print(f"1. 数据总量: 读取到 {total_rows} 条数据")

        # 2. 检查是否有重复 ID
        # 在转换为 JSON 字典前，必须确认 ID 唯一，否则会发生覆盖
        unique_ids = set(all_ids)
        duplicate_count = total_rows - len(unique_ids)
        
        if duplicate_count > 0:
            print(f"⚠️ 警告: 发现 {duplicate_count} 个重复的 center_id！")
            print("   (注意: 生成 JSON 时，重复 ID 的数据会被最后出现的覆盖)")
        else:
            print("2. 唯一性检查: 所有 center_id 均唯一，无重复。")

        # 3. 检查 ID 连续性和缺失值
        if len(unique_ids) > 0:
            min_id = min(unique_ids)
            max_id = max(unique_ids)
            
            # 理论上应该有的所有 ID 集合 (从 min 到 max)
            expected_ids = set(range(min_id, max_id + 1))
            
            # 计算缺失的 ID (理论集合 - 实际集合)
            missing_ids = sorted(list(expected_ids - unique_ids))
            missing_count = len(missing_ids)

            print(f"3. ID 范围: {min_id} 到 {max_id}")
            
            if missing_count == 0:
                print(f"4. 连续性结果: ✅ ID 是完全连续的 (共 {len(unique_ids)} 个 key)")
            else:
                print(f"4. 连续性结果: ❌ ID 不连续")
                print(f"   - 缺失数量: {missing_count} 个")
                
                # 如果缺失数量较少，打印所有；如果较多，打印前 20 个
                if missing_count <= 20:
                    print(f"   - 缺失的 ID 为: {missing_ids}")
                else:
                    print(f"   - 缺失的 ID 示例 (前20个): {missing_ids[:20]} ...")
        else:
            print("错误: 数据集为空，无法检查连续性。")

        # 4. 导出 JSON
        print(f"\n正在写入 JSON 文件: {output_path} ...")
        
        # 将 center_id 转为字符串作为 Key，center_text 作为 Value
        # dict() 构造时如果 key 重复，通过 zip 生成的字典会保留最后遇到的那个值
        result_dict = dict(zip(df_all['center_id'].astype(str), df_all['center_text']))
        
        # 确保目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False)
            
        print(f"完成。最终 JSON 包含 {len(result_dict)} 个 Key。")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_and_validate_cora()