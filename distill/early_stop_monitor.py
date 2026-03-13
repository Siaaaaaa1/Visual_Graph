#!/usr/bin/env python3
"""
早停监控器：包装 torchrun 训练命令，实时解析 stdout 中的验证集 loss。
当连续 --patience 次评估无改善时自动发送 SIGTERM 停止训练。

修复亮点：
1. 彻底清理显存：使用进程组 (PGID) 级别终止，确保 torchrun 拉起的所有子进程都被干净杀掉。
2. Pipeline 友好：如果是主动触发的早停，监控器会返回状态码 0，确保外层 bash 的 set -e 不会中断后续的 RL 阶段。
3. 实时无缓冲：强制注入 PYTHONUNBUFFERED=1，防止 Python 缓冲 stdout 导致监控延迟。

用法:
    python distill/early_stop_monitor.py [--patience N] [--min_delta D] -- <torchrun 命令>
"""

import argparse
import re
import signal
import subprocess
import sys
import os

# verl / HuggingFace Trainer 可能输出的 val loss 格式（正则列表，按优先级匹配）
VAL_LOSS_PATTERNS = [
    r"'val/loss'[:\s]+([0-9]+\.[0-9]+)",      # verl 格式: 'val/loss': 0.312
    r"val[_/]loss[:\s=]+([0-9]+\.[0-9]+)",    # val_loss: 0.312 / val/loss=0.312
    r"validation[_\s]loss[:\s=]+([0-9]+\.[0-9]+)",
    r"eval[_/]loss[:\s=]+([0-9]+\.[0-9]+)",
    r"\"val_loss\"[:\s]+([0-9]+\.[0-9]+)",    # JSON 格式
]

def parse_val_loss(line: str):
    for pattern in VAL_LOSS_PATTERNS:
        m = re.search(pattern, line, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="SFT 训练早停监控器")
    parser.add_argument("--patience", type=int, default=3,
                        help="连续多少次评估无改善后停止训练（默认 3）")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="判定为「改善」的最小 loss 下降幅度（默认 0.001）")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="训练命令（在 -- 之后传入）")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        print("[EarlyStop] 错误：未提供训练命令。", file=sys.stderr)
        sys.exit(1)

    print(f"[EarlyStop] 启动训练，patience={args.patience}, min_delta={args.min_delta}")
    print(f"[EarlyStop] 命令: {' '.join(cmd)}\n")

    # 强制开启无缓冲输出，保证实时获取 loss 信息
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # 启动子进程：使用 start_new_session=True 为 torchrun 及其子进程分配独立的 PGID
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True  
    )

    best_val_loss = float("inf")
    no_improve_count = 0
    eval_count = 0
    triggered_early_stop = False  # 标记是否为主动早停

    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            val_loss = parse_val_loss(line)
            if val_loss is None:
                continue

            eval_count += 1
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                no_improve_count = 0
                print(f"\n[EarlyStop] ✅ 改善 (eval #{eval_count}): val_loss={val_loss:.4f}", flush=True)
            else:
                no_improve_count += 1
                print(
                    f"\n[EarlyStop] ⚠️  无改善 {no_improve_count}/{args.patience} "
                    f"(eval #{eval_count}): val_loss={val_loss:.4f}, best={best_val_loss:.4f}",
                    flush=True,
                )
                if no_improve_count >= args.patience:
                    print(
                        f"\n[EarlyStop] 🛑 达到早停条件（连续 {args.patience} 次无改善）。"
                        f"最优 val_loss={best_val_loss:.4f}，正在停止训练并释放显存...",
                        flush=True,
                    )
                    triggered_early_stop = True
                    # 杀掉整个进程组，彻底清理 torchrun 和底层的所有 DDP 进程
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    break

    except KeyboardInterrupt:
        print("\n[EarlyStop] 用户中断，转发 SIGINT...", flush=True)
        # 用户手动 Ctrl+C 时，同样清理整个进程组
        os.killpg(os.getpgid(process.pid), signal.SIGINT)

    # 等待进程完全退出
    process.wait()
    print(f"\n[EarlyStop] 训练结束。共评估 {eval_count} 次，最优 val_loss={best_val_loss:.4f}")
    
    # 核心修复：如果是监控器主动触发的早停，返回 0 以保证 bash pipeline 继续执行 RL 阶段
    if triggered_early_stop:
        return 0
        
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())