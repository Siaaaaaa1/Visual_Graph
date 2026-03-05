from verl import DataProto
import torch
import numpy as np
import re

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length
        
        # [修复 2] 正则匹配：只强制校验 <action> 闭环是否存在，允许模型跳过 <think> 直接行动
        self.format_pattern = re.compile(
            r"<think>.*?</think>\s*<action>.*?</action>",
            re.DOTALL | re.IGNORECASE
        )
                
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        
        # [防刷分机制]：只设定惩罚项
        FORMAT_PENALTY_COEF = -0.2 

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch['data_source']

            is_filtered = data_item.non_tensor_batch.get('is_filtered', False)
            
            if is_filtered:
                score = 0.0
            else:
                # 提取整条轨迹的总奖励
                episode_rewards = data_item.non_tensor_batch.get('episode_rewards', 0.0)
                
                # [核心修复]：放弃使用单步的 step_reward，直接让所有中间步共享 episode_rewards。
                # 这解决了 GRPO 算法下无 Value Network 时，历史正确动作无法获得奖励正反馈的问题。
                if self.normalize_by_length:
                    # 如果开启归一化，则将总奖励平摊到每一步上
                    score = episode_rewards / data_item.non_tensor_batch.get('episode_lengths', 1)
                else:
                    # 不开启归一化，则每一步都完整获得该 Trajectory 的最终结果
                    score = episode_rewards
                
            # 格式惩罚必须全局强制生效（过滤样本已被抹平，此处仅对正常样本强制惩罚）
            if not is_filtered and not self.format_pattern.search(response_str):
                score += FORMAT_PENALTY_COEF

            # 将计算好的 score 放置在模型当前步 response 的最后一个 token 处
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # 随机采样打印，用于观察 Reward 的分配是否符合预期
            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor