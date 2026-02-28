from verl import DataProto
import torch
import numpy as np
import re  # 新增引用

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length
        # 预编译正则，提高效率 (宽松匹配：允许标签间有换行或空白)
        self.format_pattern = re.compile(
            r"(?:<think>)?.*?</think>.*<summary>.*?</summary>.*<action>.*?</action>", 
            re.DOTALL | re.IGNORECASE
        )
                
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        
        # [防刷分机制]：只设定惩罚项，不设正向格式奖励。
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

            # --- [核心修改]：绝对过滤逻辑执行 ---
            # 检查来自 rollout_loop.py 的绝对过滤标记
            is_filtered = data_item.non_tensor_batch.get('is_filtered', False)
            
            if is_filtered:
                # 如果被过滤，强制设为 0，不进行任何奖励或惩罚计算
                score = 0.0
            else:
                episode_rewards = data_item.non_tensor_batch['episode_rewards']
                episode_lengths = data_item.non_tensor_batch['episode_lengths']

                # 优先读取单步事后分配的真实 Reward
                step_reward = data_item.non_tensor_batch.get('step_reward', episode_rewards)

                # 计算基础分数
                if self.normalize_by_length:
                    score = step_reward / episode_lengths
                else:
                    score = step_reward
                
                # 只有非过滤样本才进行格式校验和惩罚
                if not self.format_pattern.search(response_str):
                    score += FORMAT_PENALTY_COEF 
            # ---------------------------------

            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

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