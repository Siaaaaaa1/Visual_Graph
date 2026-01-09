import re
from typing import List, Tuple

_ACTION_BLOCK = re.compile(
    r"<action>(.*?)</action>",
    re.IGNORECASE | re.DOTALL
)

_ACTION_TAG = re.compile(r"<action>", re.IGNORECASE)

_CHECK_NODE_RE = re.compile(r"^check_node:(\d+)$")
_CHECK_NODES_RE = re.compile(r"^check_nodes:\[(.*?)\]$")

_CHECK_GRAPH_RE = re.compile(r"^check_graph:(.+?)$") 

MAX_NODES_PER_STEP = 5

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

        # 4. check_graph:<mode>[,<max>]
        if action.startswith("check_graph:"):
            # 简单校验格式
            try:
                params = action.split(":", 1)[1].split(",")
                mode = params[0].strip()
                if mode in ["1-hop", "2-hop", "sim", "1-hop+sim"]:
                    results.append(action)
                    continue
            except:
                pass
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