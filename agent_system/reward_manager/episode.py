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
        # 预编译正则，提高效率 (宽松匹配：允许标签间有换行或空白)
        self.format_pattern = re.compile(
            r"(?:<think>)?.*?</think>.*<action>.*?</action>", 
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

            # =========================================================
            # [核心修改]：绝对过滤与多轮 GRPO 优势奖励整合逻辑
            # =========================================================
            is_filtered = data_item.non_tensor_batch.get('is_filtered', False)
            
            if is_filtered:
                score = 0.0
            else:
                episode_rewards = data_item.non_tensor_batch.get('episode_rewards', 0.0)
                step_reward = data_item.non_tensor_batch.get('step_reward', episode_rewards)
                
                # 获取在 rollout_loop 中计算的轨迹级别 GRPO Advantage
                traj_advantage = data_item.non_tensor_batch.get('traj_advantage', 0.0)

                if self.normalize_by_length:
                    step_reward = step_reward / data_item.non_tensor_batch.get('episode_lengths', 1)

                # 核心合并：将组内相对优势叠加单步 reward 分发给 Token
                score = traj_advantage + step_reward
                
            # 格式惩罚必须全局强制生效（过滤样本已被抹平，此处仅对正常样本强制惩罚）
            if not is_filtered and not self.format_pattern.search(response_str):
                score += FORMAT_PENALTY_COEF
            # =========================================================

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