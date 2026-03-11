import os
import json
import uuid
import re
from datetime import datetime
import numpy as np
from filelock import FileLock

class SmartLogger:
    """
    工业级 GRPO/RLHF 训练日志与自动诊断器
    提供高并发安全、轨迹去重、多模态清洗、GRPO 细粒度校验与训练风险预警。

    日志格式：
    - 高频轨迹数据写入 JSONL 文件（追加模式，无需读回整文件）
    - 低频诊断报告写入独立 JSON 文件（体积小，读写开销可忽略）
    """
    _instance = None

    def __new__(cls, log_dir="./log"):
        if cls._instance is None:
            cls._instance = super(SmartLogger, cls).__new__(cls)
            cls._instance._init_logger(log_dir)
        return cls._instance

    def _init_logger(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 依赖环境变量确保 Ray 集群/多进程共享同一个 Log File ID
        self.run_id = os.environ.get("SMART_LOGGER_RUN_ID")
        if not self.run_id:
            self.run_id = str(uuid.uuid4())[:4]
            os.environ["SMART_LOGGER_RUN_ID"] = self.run_id

        self.date_str = datetime.now().strftime("%m%d")

        # 高频轨迹/指标 → JSONL（追加写，不需要读全文件）
        self.log_file = os.path.join(self.log_dir, f"log_{self.run_id}_{self.date_str}.jsonl")
        # 低频诊断报告 → 独立 JSON（体积小，可安全读写全文件）
        self.report_file = os.path.join(self.log_dir, f"report_{self.run_id}_{self.date_str}.json")
        self.lock_file = self.log_file + ".lock"

        self.history_lengths = []

        # 初始化诊断报告 JSON 骨架
        with FileLock(self.lock_file):
            if not os.path.exists(self.report_file):
                with open(self.report_file, 'w', encoding='utf-8') as f:
                    json.dump({"analysis_reports": []}, f, indent=2)

    def _clean_multimodal_tokens(self, text: str) -> str:
        """多模态 Token 清理：精准剔除视觉占位符"""
        if not isinstance(text, str):
            return str(text)
        pattern = r'<image>|<patch_\d+>|<\|image_pad\|>|<\|vision_start\|>|<\|vision_end\|>'
        return re.sub(pattern, '', text)

    def _process_trajectories(self, total_batch_list, is_agent: bool):
        """处理轨迹，角色拆分，Agent 历史去重"""
        processed_trajs = []
        for batch_idx, traj_steps in enumerate(total_batch_list):
            traj_record = {"trajectory_id": batch_idx, "roles": []}

            for step_idx, step_data in enumerate(traj_steps):
                if not step_data.get('active_masks', True):
                    continue

                raw_prompt = str(step_data.get('final_prompt_text', ''))
                env_obs = str(step_data.get('anchor_obs', raw_prompt))
                response = str(step_data.get('model_response_text', ''))

                env_obs = self._clean_multimodal_tokens(env_obs)
                response = self._clean_multimodal_tokens(response)

                env_obs_literal = env_obs.replace('\n', '\\n')
                response_literal = response.replace('\n', '\\n')

                if is_agent:
                    traj_record["roles"].append({
                        "step": step_idx,
                        "Environment_Observation": env_obs_literal,
                        "Agent_Action": response_literal,
                        "Step_Reward": float(step_data.get('step_reward', 0.0)),
                        "Format_Penalty": float(step_data.get('format_penalty', 0.0))
                    })
                else:
                    traj_record["roles"].append({
                        "step": step_idx,
                        "User_Prompt": raw_prompt.replace('\n', '\\n'),
                        "Model_Response": response_literal,
                        "Reward": float(step_data.get('step_reward', 0.0))
                    })
            processed_trajs.append(traj_record)
        return processed_trajs

    def log_rollout_data(self, step_idx: int, total_batch_list: list, group_size: int,
                         rewards: np.ndarray, format_penalties: np.ndarray = None, is_agent: bool = True):
        """记录 Rollout 轨迹数据与 GRPO 打组及奖励校验（追加写 JSONL，无读取开销）"""
        trajectories = self._process_trajectories(total_batch_list, is_agent)

        grpo_groups = []
        batch_size = len(total_batch_list)
        invalid_reward_flags = 0
        success_count = 0

        if rewards is not None and batch_size > 0:
            rewards_flat = rewards.flatten()
            for i in range(0, batch_size, group_size):
                group_outcomes = rewards_flat[i : i + group_size]
                success_count += np.sum(group_outcomes >= 1.0)
                grpo_groups.append({
                    "group_idx": i // group_size,
                    "group_mean_reward": float(np.mean(group_outcomes)),
                    "group_std_reward": float(np.std(group_outcomes)),
                    "member_trajectories": [j for j in range(i, min(i + group_size, batch_size))]
                })

            for traj in total_batch_list:
                for s_data in traj:
                    if not s_data.get('active_masks', True) and s_data.get('step_reward', 0.0) != 0.0:
                        invalid_reward_flags += 1

        step_record = {
            "type": "rollout",
            "global_step": step_idx,
            "timestamp": datetime.now().isoformat(),
            "grpo_stats": {
                "total_groups": len(grpo_groups),
                "success_rate": float(success_count / batch_size) if batch_size > 0 else 0.0,
                "invalid_reward_allocations_detected": invalid_reward_flags,
                "groups": grpo_groups
            },
            "trajectories": trajectories,
        }

        # 追加写 JSONL，O(1) 写入，不随训练轮数退化
        with FileLock(self.lock_file):
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(step_record, ensure_ascii=False) + '\n')

    def log_rl_metrics_and_diagnose(self, step_idx: int, metrics: dict):
        """注入 PPO/RL 训练核心指标并执行自动诊断（指标追加写 JSONL，报告写小型 JSON）"""
        metrics_record = {
            "type": "metrics",
            "global_step": step_idx,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }

        risk_reports = self._auto_diagnose(metrics)

        with FileLock(self.lock_file):
            # 指标追加写 JSONL
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_record, ensure_ascii=False) + '\n')

            # 诊断报告体积小，维持原有 JSON 读写模式
            if risk_reports:
                with open(self.report_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                report_entry = {
                    "step": step_idx,
                    "timestamp": datetime.now().isoformat(),
                    "warnings": risk_reports
                }
                data["analysis_reports"].insert(0, report_entry)
                with open(self.report_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

    def _auto_diagnose(self, metrics: dict) -> list:
        reports = []

        curr_len = metrics.get('response_length/mean', None)
        if curr_len is not None:
            self.history_lengths.append(curr_len)
            if len(self.history_lengths) > 6:
                self.history_lengths.pop(0)

        if curr_len is not None and len(self.history_lengths) >= 2:
            initial_len = self.history_lengths[0]
            mean_last_5 = np.mean(self.history_lengths[:-1])
            if curr_len < mean_last_5 * 0.6 or curr_len < initial_len * 0.5:
                reports.append(f"[CRITICAL] Response length collapsed from ~{mean_last_5:.1f} to {curr_len:.1f}. Potential Reward Hacking detected: Agent might be 'shutting up' to avoid penalties.")

        entropy = metrics.get('actor/entropy_loss', None)
        if entropy is not None and entropy < 0.05:
            reports.append(f"[WARNING] Mode Collapse Risk: actor/entropy_loss is {entropy:.4f} (< 0.05). The model is losing output diversity.")

        grad_norm = metrics.get('actor/grad_norm', None)
        if grad_norm is not None and grad_norm > 30.0:
            reports.append(f"[WARNING] Policy Collapse Risk: actor/grad_norm spiked to {grad_norm:.2f} (> 30.0). Gradients are exploding.")

        ppo_kl = metrics.get('actor/ppo_kl', None)
        if ppo_kl is not None and ppo_kl > 0.15:
            reports.append(f"[CRITICAL] Divergence Risk: actor/ppo_kl is {ppo_kl:.4f} (> 0.15). The policy has deviated significantly from the Reference Model and might start generating gibberish.")

        return reports
