import re
from typing import List, Tuple

_ACTION_BLOCK = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_ACTION_TAG = re.compile(r"<action>", re.IGNORECASE)

# V-GraphAgent 升级版动作空间：支持批量节点查阅
_CHECK_NODES_RE = re.compile(r"^check_nodes?\(\[?([\d,\s]+)\]?\)$", re.IGNORECASE)
_FINAL_RE = re.compile(r"^(?:final|submit)\((.+?)\)$", re.IGNORECASE)

def graph_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    for i, raw in enumerate(actions):
        if len(_ACTION_TAG.findall(raw)) != 1:
            results.append(""); valids[i] = 0; continue

        m = _ACTION_BLOCK.search(raw)
        if not m:
            results.append(""); valids[i] = 0; continue

        action = m.group(1).strip()

        # 1. 批量查阅摘要 (如 check_nodes([1024, 2048]))
        if _CHECK_NODES_RE.match(action):
            results.append(action)
            continue

        # 2. 提交最终答案
        if _FINAL_RE.match(action):
            results.append(action)
            continue

        # 未命中合法动作
        results.append(""); valids[i] = 0

    return results, valids