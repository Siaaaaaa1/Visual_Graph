import re
from typing import List, Tuple

_ACTION_BLOCK = re.compile(
    r"<action>(.*?)</action>",
    re.IGNORECASE | re.DOTALL
)

_ACTION_TAG = re.compile(r"<action>", re.IGNORECASE)


def graph_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    输入：
      actions: LLM 原始输出（batch）

    输出：
      results: 抽取出的 action 字符串（如 'view_node:28'）
      valids : 0/1，表示该 action 是否合法
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    for i, raw in enumerate(actions):
        # 1. 必须且只能有一个 <action>
        if len(_ACTION_TAG.findall(raw)) != 1:
            results.append("")
            valids[i] = 0
            continue

        # 2. 抽取 action 内容
        m = _ACTION_BLOCK.search(raw)
        if not m:
            results.append("")
            valids[i] = 0
            continue

        action = m.group(1).strip()

        # 3. 基本合法性校验（语法层，不是语义层）
        if ":" not in action:
            results.append("")
            valids[i] = 0
            continue

        results.append(action)

    return results, valids
