import re
from typing import List, Tuple

_ACTION_BLOCK = re.compile(
    r"<action>(.*?)</action>",
    re.IGNORECASE | re.DOTALL
)

_ACTION_TAG = re.compile(r"<action>", re.IGNORECASE)

_CHECK_NODE_RE = re.compile(r"^check_node:(\d+)$")
_CHECK_NODES_RE = re.compile(r"^check_nodes:\[(.*?)\]$")

# 允许 view_mode 包含字母、数字、连字符和加号 (例如: 1-hop+sim)
_CHECK_GRAPH_RE = re.compile(r"^check_graph:(.+?)$") 

MAX_NODES_PER_STEP = 5
VALID_VIEW_MODES = {"1-hop", "2-hop", "sim", "1-hop+sim", "2-hop+sim"}

def graph_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    for i, raw in enumerate(actions):
        # 1. 唯一性检查
        if len(_ACTION_TAG.findall(raw)) != 1:
            results.append("")
            valids[i] = 0
            continue

        m = _ACTION_BLOCK.search(raw)
        if not m:
            results.append("")
            valids[i] = 0
            continue

        action = m.group(1).strip()

        # 2. check_node:<id>
        if _CHECK_NODE_RE.match(action):
            results.append(action)
            continue

        # 3. check_nodes:[id1, id2]
        m_multi = _CHECK_NODES_RE.match(action)
        if m_multi:
            content = m_multi.group(1).strip()
            if not content:
                results.append("")
                valids[i] = 0
                continue
            try:
                # 允许空格
                node_ids = [int(x.strip()) for x in content.split(",") if x.strip()]
                if 0 < len(node_ids) <= MAX_NODES_PER_STEP:
                    results.append(action)
                    continue
            except ValueError:
                pass
            results.append("")
            valids[i] = 0
            continue

        # 4. check_graph:<view_mode>,<max_nodes>
        if action.startswith("check_graph:"):
            try:
                params = action.split(":", 1)[1].split(",")

                # 必须正好 2 个参数
                if len(params) != 2:
                    raise ValueError

                view_mode = params[0].strip()
                max_nodes = int(params[1].strip())

                if view_mode not in VALID_VIEW_MODES:
                    raise ValueError
                if max_nodes <= 0:
                    raise ValueError

                results.append(action)
                continue
            except:
                results.append("")
                valids[i] = 0
                continue

        # 5. final:<category>
        if action.startswith("final:"):
            results.append(action)
            continue

        # 非法
        results.append("")
        valids[i] = 0

    return results, valids