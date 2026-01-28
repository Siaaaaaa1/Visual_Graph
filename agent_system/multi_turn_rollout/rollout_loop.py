# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from verl import DataProto
import time
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import os
from PIL import Image
import json
import re 

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        step: int = 0,
    ):
        """
        Process a single observation sample.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # --- 图片可视化抽样保存逻辑 ---
        if is_multi_modal and step > 0 and np.random.random() < 0.01:
            try:
                debug_dir = os.path.join(os.getcwd(), 'debug_images')
                os.makedirs(debug_dir, exist_ok=True)
                
                img_to_save = None
                if isinstance(obs_image, Image.Image):
                    img_to_save = obs_image
                elif isinstance(obs_image, (np.ndarray, torch.Tensor)):
                    img_to_save = process_image(obs_image)
                
                if img_to_save:
                    timestamp = int(time.time() * 1000)
                    filename = f"step{step}_{timestamp}_{item}_{data_source}.png"
                    save_path = os.path.join(debug_dir, filename)
                    img_to_save.save(save_path)
                    print(f"[Debug] Saved sampled image (Step {step}) to {save_path}")
            except Exception as e:
                print(f"[Warning] Failed to save debug image: {e}")
        # -----------------------------------

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        if is_multi_modal:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            ) 
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)] 
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        # --- 截断内容捕获逻辑 ---
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        original_len = len(raw_prompt_ids)
        truncated_text = ""
        is_truncated = False

        if original_len > self.config.data.max_prompt_length:
            is_truncated = True
            if self.config.data.truncation == "left":
                truncated_ids = raw_prompt_ids[:-self.config.data.max_prompt_length]
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                truncated_ids = raw_prompt_ids[self.config.data.max_prompt_length:]
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                truncated_ids = raw_prompt_ids[left_half:-right_half]
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")
            
            if is_truncated and truncated_ids:
                truncated_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=False)

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source,
            'final_prompt_text': prompt_with_chat_template,
            'is_truncated': is_truncated,
            'truncated_text': truncated_text
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict,
        step: int = 0,
    ) -> DataProto:
        """
        Process a batch of observation samples.
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        for item in range(batch_size):
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                step=step,
            )
            processed_samples.append(processed)
        
        batch = collate_fn(processed_samples)
        
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data.
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_lengths'] = episode_lengths[bs]
                    data['tool_callings'] = tool_callings[bs]
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment loop.
        """

        batch_size = len(gen_batch.batch)

        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
            
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        summary_pattern = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
        action_pattern = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs, step=_step)

            input_prompts = batch.non_tensor_batch.get('final_prompt_text', [""] * batch_size)
            truncation_flags = batch.non_tensor_batch.get('is_truncated', [False] * batch_size)
            truncated_texts = batch.non_tensor_batch.get('truncated_text', [""] * batch_size)
            
            # 获取 image_inputs (这是一个 batch 数组，每个元素是一个 dict)
            image_inputs = batch.non_tensor_batch.get('multi_modal_inputs', None)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "final_prompt_text", "is_truncated", "truncated_text"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            
            # --- Model Interaction ---
            model_start_time = time.time()
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            model_duration = time.time() - model_start_time
            
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            # 将模型的原始文本响应写入 Batch
            batch.non_tensor_batch['model_response_text'] = np.array(text_actions, dtype=object)

            # --- Env Interaction ---
            env_start_time = time.time()
            next_obs, rewards, dones, infos = envs.step(text_actions)
            env_duration = time.time() - env_start_time

            print(f"\n{'='*40} STEP {_step} DETAILS {'='*40}")
            input_len_batch = batch_input.batch['input_ids'].shape[1]
            output_len_batch = batch_output.batch['responses'].shape[1]
            print(f"Stats: ModelTime={model_duration:.2f}s, EnvTime={env_duration:.2f}s, InputLen={input_len_batch}, OutputLen={output_len_batch}")

            for i in range(batch_size):
                if i >= 1: break 
                
                print(f"\n--- [Sample {i}] ---")
                
                valid_input_len = batch_input.batch['attention_mask'][i].sum().item()
                img_info = "No Image"
                
                # --- [修正逻辑 Start] ---
                # 1. 显式检查 is not None，避免 numpy array 的 truth value ambiguous 错误
                if image_inputs is not None:
                    # 2. image_inputs 是一个 Object Array，先用 [i] 取出当前样本的字典
                    sample_inputs = image_inputs[i]
                    
                    if isinstance(sample_inputs, dict):
                        if 'image_grid_thw' in sample_inputs:
                             # 3. 直接访问字典内的值，不需要再加 [i]
                             grid = sample_inputs['image_grid_thw']
                             img_shape = grid.tolist() if hasattr(grid, 'tolist') else str(grid)
                             img_info = f"Grid Shape: {img_shape}"
                        elif 'pixel_values' in sample_inputs:
                             pv = sample_inputs['pixel_values']
                             if isinstance(pv, list):
                                 img_info = f"Pixel Values: {len(pv)} images"
                             else:
                                 img_info = f"Pixel Values Shape: {pv.shape}"
                # --- [修正逻辑 End] ---
                
                print(f"[Input Stats] Text Len: {valid_input_len} | Image Info: {img_info}")

                print(f"[Input Prompt] (Start): {input_prompts[i][:200]} ...")
                print(f"[Input Prompt] (End): ... {input_prompts[i][-500:]}")
                
                if truncation_flags[i]:
                    print(f"\n[!!! WARNING: TRUNCATED !!!]")
                    print(f"[Truncated Content]: {truncated_texts[i]}")
                
                response_text = text_actions[i]
                print(f"\n[Full Model Response]:\n{response_text}")

                think = "N/A"
                summary = "N/A"
                action = "N/A"

                if infos and i < len(infos):
                    info = infos[i]
                    think = info.get('parsed_think', None)
                    summary = info.get('parsed_summary', None)
                    action = info.get('parsed_action_content', None)
                
                if not think: 
                    m = think_pattern.search(response_text)
                    think = m.group(1).strip() if m else "Not Found"
                if not summary:
                    m = summary_pattern.search(response_text)
                    summary = m.group(1).strip() if m else "Not Found"
                if not action or action == "No Action Found":
                    m = action_pattern.search(response_text)
                    action = m.group(1).strip() if m else "Not Found"

                print(f"\n[Parsed Structure]")
                print(f"  > Think: {think[:300]}..." if len(str(think)) > 300 else f"  > Think: {think}")
                print(f"  > Summary: {summary}")
                print(f"  > Action: {action}")

                feedback = next_obs['text'][i]
                print(f"\n[Environment Feedback]:\n{feedback[:1000]}..." if len(feedback) > 1000 else f"\n[Environment Feedback]:\n{feedback}")
                print("-" * 60)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            # 将解析后的结构化数据 (Think/Summary/Action) 写入 Batch
            if infos:
                batch.non_tensor_batch['parsed_think'] = np.array([info.get('parsed_think', '') for info in infos], dtype=object)
                batch.non_tensor_batch['parsed_summary'] = np.array([info.get('parsed_summary', '') for info in infos], dtype=object)
                batch.non_tensor_batch['parsed_action'] = np.array([info.get('parsed_action_content', '') for info in infos], dtype=object)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            is_done = np.logical_or(is_done, dones)
            obs = next_obs

            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
            
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        if self.config.algorithm.filter_groups.enable and is_train:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs)
        else:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs)
        
        assert len(total_batch_list) == len(total_episode_rewards)

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        # --- 优化后的保存逻辑 Start ---
        REMOVE_KEYS = ['pixel_values', 'image_grid_thw'] 
        PARENT_KEY = 'multi_modal_inputs' 
        # 指定需要排到最后的字段列表
        END_KEYS = ['model_response_text', 'anchor_obs', 'final_prompt_text', 'raw_prompt', 'parsed_think']

        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, (np.ndarray, np.generic)):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj.item()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(x) for x in obj]
            elif isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8')
                except:
                    return str(obj)
            return obj

        output_dir = os.path.join(os.getcwd(), 'test')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'guiji.jsonl')

        non_tensor_data = gen_batch_output.non_tensor_batch
        if non_tensor_data:
            all_keys = list(non_tensor_data.keys())
            batch_size = len(non_tensor_data[all_keys[0]])

            # 1. 筛选出中间字段：排除掉 index 和 需要放到最后的 END_KEYS
            # 注意：step_in_traj 是我们在循环中生成的，不属于原始 key
            middle_keys = [k for k in all_keys if k != 'index' and k not in END_KEYS]
            
            # 2. 筛选出数据中实际存在的 END_KEYS (防止硬编码的 key 不存在报错)
            valid_end_keys = [k for k in END_KEYS if k in all_keys]

            # 用于计算 step_in_traj 的状态变量
            last_traj_uid = None
            step_counter = 0

            with open(output_file, 'a', encoding='utf-8') as f:
                for i in range(batch_size):
                    row = {} # Python 3.7+ 字典保证插入顺序

                    # --- A. 头部字段: index ---
                    if 'index' in non_tensor_data:
                        row['index'] = make_serializable(non_tensor_data['index'][i])
                    
                    # --- B. 头部字段: step_in_traj (轨迹内序号) ---
                    # 获取当前样本的 traj_uid
                    current_traj_uid = str(non_tensor_data['traj_uid'][i]) if 'traj_uid' in non_tensor_data else "unknown"
                    
                    # 如果 traj_uid 变了，说明是新的一条轨迹，重置计数器
                    # (gather_rollout_data 保证了同一轨迹的数据是连续的)
                    if last_traj_uid != current_traj_uid:
                        step_counter = 0
                        last_traj_uid = current_traj_uid
                    else:
                        step_counter += 1
                    
                    row['step_in_traj'] = step_counter

                    # --- C. 中间字段 (动态处理，包含 action 等任意字段) ---
                    for key in middle_keys:
                        raw_val = non_tensor_data[key][i]
                        # 特殊清理逻辑：清理 multi_modal_inputs
                        if key == PARENT_KEY and isinstance(raw_val, dict):
                            raw_val = {k: v for k, v in raw_val.items() if k not in REMOVE_KEYS}
                        
                        row[key] = make_serializable(raw_val)
                    
                    # --- D. 尾部字段 (大文本) ---
                    for key in valid_end_keys:
                        raw_val = non_tensor_data[key][i]
                        row[key] = make_serializable(raw_val)

                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        # --- 优化后的保存逻辑 End ---
        
        return gen_batch_output