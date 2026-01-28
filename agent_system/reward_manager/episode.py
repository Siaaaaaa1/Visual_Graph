# File 3: agent_system/reward_manager.py

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
        # 这里的 .*? 是非贪婪匹配，re.DOTALL 允许 . 匹配换行符
        self.format_pattern = re.compile(r"<think>.*?</think>.*<summary>.*?</summary>.*<action>.*?</action>", re.DOTALL | re.IGNORECASE)

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
        
        # 定义格式奖励系数 (根据你的奖励尺度调整，这里设为 +/- 0.1)
        FORMAT_REWARD_COEF = 0.1 

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # 修改点：将 valid_prompt_ids 改为 prompt_ids (利用切片获取有效长度)
            # prompt_ids 在上方已定义
            prompt_str = self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch['data_source']

            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            # --- 计算基础分数 ---
            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards
            
            # --- [新增] 格式奖励逻辑 ---
            # 检查 response_str 是否符合格式
            if self.format_pattern.search(response_str):
                score += FORMAT_REWARD_COEF
            else:
                score -= FORMAT_REWARD_COEF # 惩罚非法格式
            # -------------------------

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