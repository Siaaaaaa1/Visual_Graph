import os
import json
import numpy as np
from PIL import Image
import random

# ==========================================
# 1. 导入你的环境组件 (请根据你的实际路径适当调整)
# ==========================================
from agent_system.environments.env_package.graph_search.envs import build_graph_search_envs
from agent_system.environments.env_manager_graph_search import GraphSearchEnvironmentManager
from agent_system.environments.env_package.graph_search.projection import graph_search_projection

# ==========================================
# 2. 模拟配置类 (Mock Configs)
# ==========================================
class MockEnvConfig:
    max_steps = 10
    dataset_name = "arxiv"
    dataset_dir = "./datasets"
    node_text_path = "./datasets/arxiv_text.json"

class MockManagerConfig:
    class EnvConfig:
        history_length = 5
    env = EnvConfig()

def main():
    print("🚀 [Init] 正在加载环境与图数据集...")
    env_config = MockEnvConfig()
    manager_config = MockManagerConfig()

    # 自动从 arxiv.json 获取真实的测试节点
    arxiv_path = os.path.join(env_config.dataset_dir, f"{env_config.dataset_name}.json")
    if not os.path.exists(arxiv_path):
        print(f"❌ 找不到文件: {arxiv_path}，请确保路径正确！")
        return

    with open(arxiv_path, "r", encoding="utf-8") as f:
        arxiv_data = json.load(f)
        
        # 随机采样 30 个节点 (如果不足30个则全取)
        nodes_list = arxiv_data.get("nodes", [])
        sample_size = min(30, len(nodes_list))
        test_nodes = random.sample(nodes_list, sample_size)
    
    print(f"🎯 [Init] 成功随机采样 {sample_size} 个节点，准备进行绘图与测试。")

    # ==========================================
    # 3. 实例化底层环境群和主控 Manager
    # ==========================================
    envs = build_graph_search_envs(
        seed=42, 
        env_num=1, 
        group_n=1, 
        is_train=False, 
        env_config=env_config
    )
    
    manager = GraphSearchEnvironmentManager(
        envs=envs, 
        projection_f=graph_search_projection, 
        config=manager_config
    )

    # ==========================================
    # 4. 遍历 30 个节点进行绘图，并仅对第 1 个节点交互
    # ==========================================
    for idx, test_node in enumerate(test_nodes):
        center_id = int(test_node["id"])
        answer = test_node.get("label", test_node.get("proxy_info", {}).get("top1", "Unknown"))
        
        print(f"\n[{idx+1}/{sample_size}] 正在处理节点 ID: {center_id} | 真实答案: {answer}")

        # 初始化 (Reset) 环境到当前节点
        reset_kwargs = [{"center_id": center_id, "answer": answer}]
        observations, infos = manager.reset(reset_kwargs)

        mode = infos[0].get("mode", "Unknown")
        print(f"🔮 [Env] 判定模式: {mode} (由同质性决定)")
        
        # 保存每个节点的初始视野图像 (绘图)
        init_img_array = observations["image"][0]
        if init_img_array is not None:
            img_filename = f"agent_view_node_{center_id}.png"
            Image.fromarray(init_img_array).save(img_filename)
            print(f"🖼️  [Vision] 节点 {center_id} 的初始视野图像已保存至 {img_filename}")

        # 只对第一个节点开启对话交互，其余节点仅做环境重置和出图
        if idx == 0:
            print("\n==============================================")
            print(f"📜 [Prompt] 节点 {center_id} 的初始 Prompt 如下：")
            print("==============================================")
            print(observations["text"][0])
            print("==============================================\n")

            print("💡 提示：现在的脚本支持真实的、包含多行文本的大模型输出！")
            print("请直接将带有 <thinking>...</thinking>, <summary>...</summary>, <action>...</action> 的完整内容粘贴进来。")
            print("完成输入/粘贴后，请在新的一行输入 'EOF' 并回车提交。输入 'QUIT' 退出测试。\n")

            # 5. 真实的交互式测试循环 (Interactive Loop)
            while True:
                print("👉 请粘贴大模型的完整输出 (结束请在新行输入 EOF)：")
                lines = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break # 处理 ctrl+d 等直接结束输入的情况
                    
                    if line.strip().upper() == "QUIT":
                        print("🚪 退出当前节点的交互测试，继续渲染剩余的测试节点图像...")
                        break
                    if line.strip().upper() == "EOF":
                        break
                    
                    lines.append(line)
                
                # 如果刚才内层循环是因为 QUIT 退出，外层也退出
                if line.strip().upper() == "QUIT":
                    break

                raw_llm_response = "\n".join(lines)
                if not raw_llm_response.strip():
                    continue

                print("\n⏳ [Manager] 正在解析原始文本并向环境提交请求...")
                
                # 直接将完整的、包含了各种标签的纯文本交给 Manager
                next_obs, rewards, dones, step_infos = manager.step([raw_llm_response])

                reward = rewards[0]
                done = dones[0]
                info = step_infos[0]
                env_feedback = next_obs["anchor"][0] # 底层环境返回的原始文本观测

                print("\n================ 环境反馈 ================")
                print(f"🧠 解析到的 Think:\n{info.get('parsed_think')}")
                print(f"📝 解析到的 Summary:\n{info.get('parsed_summary')}")
                print(f"⚡ 解析到的 Action: {info.get('parsed_action_content')}")
                print(f"\n🔍 环境返回给模型的观测 (Next Obs):\n{env_feedback}")
                print(f"\n💰 单步奖励: {reward}")
                
                # 保存交互后的视野图像
                next_img_array = next_obs["image"][0]
                if next_img_array is not None:
                    step_img_filename = f"agent_view_node_{center_id}_step.png"
                    Image.fromarray(next_img_array).save(step_img_filename)
                    print(f"🖼️  视野图像已更新 -> {step_img_filename}")
                print("==========================================")

                if done:
                    final_reward = info.get("final_reward", reward)
                    won = info.get("won", False)
                    print(f"\n🎉 当前节点交互结束！最终状态: {'[胜利 WIN]' if won else '[失败 LOSE]'} | 最终奖励: {final_reward}")
                    if hasattr(manager, "current_run_log_file"):
                        print(f"📂 轨迹记录已追加至: {manager.current_run_log_file}")
                    break

if __name__ == "__main__":
    main()