import re
from typing import List, Tuple

_ACTION_BLOCK = re.compile(
    r"<action>(.*?)</action>",
    re.IGNORECASE | re.DOTALL
)

_ACTION_TAG = re.compile(r"<action>", re.IGNORECASE)

# 新增：multi-node 解析用的正则
_VIEW_NODE_RE = re.compile(r"^view_node:(\d+)$")
_VIEW_NODES_RE = re.compile(r"^view_nodes:\[(.*?)\]$")

MAX_NODES_PER_STEP = 5


def graph_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    输入：
      actions: LLM 原始输出（batch）

    输出：
      results: 抽取出的 action 字符串（如 'view_node:28' 或 'view_nodes:[28,41]'）
      valids : 0/1，表示该 action 是否合法（语法 + 基本约束）
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

        # 3. view_node:<id>
        m_single = _VIEW_NODE_RE.match(action)
        if m_single:
            results.append(action)
            continue

        # 4. view_nodes:[id1,id2,...]
        m_multi = _VIEW_NODES_RE.match(action)
        if m_multi:
            content = m_multi.group(1).strip()
            if not content:
                results.append("")
                valids[i] = 0
                continue

            try:
                node_ids = [int(x.strip()) for x in content.split(",")]
            except ValueError:
                results.append("")
                valids[i] = 0
                continue

            if len(node_ids) == 0 or len(node_ids) > MAX_NODES_PER_STEP:
                results.append("")
                valids[i] = 0
                continue

            results.append(action)
            continue

        # 5. final:<category>
        if action.startswith("final:"):
            results.append(action)
            continue

        # 6. 其他情况：非法
        results.append("")
        valids[i] = 0

    return results, valids
