from verl import DataProto
import torch
import numpy as np
import re

class EpisodeRewardManager:
    """The reward manager.
    """

    # 移除了 normalize_by_length 参数
    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        
        # 用于匹配完整的闭环格式
        self.format_pattern = re.compile(
            r"<think>.*?</think>\s*<action>.*?</action>",
            re.DOTALL | re.IGNORECASE
        )
                
    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        
        FORMAT_PENALTY_COEF = -0.2 

        for i in range(len(data)):
            data_item = data[i] 

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch['data_source']
            is_filtered = data_item.non_tensor_batch.get('is_filtered', False)
            
            if is_filtered:
                score = 0.0
            else:
                # 【修改点 1】：彻底废除 episode_lengths 平摊逻辑，直接拿总奖励
                score = float(data_item.non_tensor_batch.get('episode_rewards', 0.0))
                
            # 【修改点 2】：强化格式与长度校验
            if not is_filtered:
                think_match = re.search(r"<think>(.*?)</think>", response_str, re.DOTALL | re.IGNORECASE)
                action_match = re.search(r"<action>(.*?)</action>", response_str, re.DOTALL | re.IGNORECASE)
                
                if not (think_match and action_match):
                    # 连基本格式都没有，扣分
                    score += FORMAT_PENALTY_COEF
                # else:
                #     # 思考内容不足 50 个字符，视为偷懒作弊，扣分
                #     think_content = think_match.group(1).strip()
                #     if len(think_content) < 50:
                #         score += FORMAT_PENALTY_COEF

            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][score]", score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        else:
            return reward_tensor