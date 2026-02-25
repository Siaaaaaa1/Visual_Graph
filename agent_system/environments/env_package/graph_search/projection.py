import re
from typing import List, Tuple

_ACTION_BLOCK = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_ACTION_TAG = re.compile(r"<action>", re.IGNORECASE)

_CHECK_NODE_RE = re.compile(r"^check_node:(\d+)$")
_CHECK_NODES_RE = re.compile(r"^check_nodes:\[(.*?)\]$")
_CHECK_GRAPH_RE = re.compile(r"^check_graph:(.+?)$") 
_PAINT_RE = re.compile(r"^paint:(\d+),\s*(.+)$") 

MAX_NODES_PER_STEP = 5
VALID_VIEW_MODES = {"1-hop", "2-hop", "sim", "1-hop+sim", "2-hop+sim"}

def graph_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    将模型生成的原始输出映射到标准的合法环境动作中。
    检查语法合法性，若不合法则阻断并标记。
    
    Args:
        actions (List[str]): 模型批量生成的原始文本列表。
        
    Returns:
        Tuple[List[str], List[int]]: (格式化后的合法动作字符串列表, 是否合法的 1/0 标志位列表)
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    for i, raw in enumerate(actions):
        # 必须仅包含一个 <action> 标签块，防止模型幻觉多输出
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

        if _CHECK_NODE_RE.match(action):
            results.append(action)
            continue

        # 解析 check_nodes:[1,2,3] 的批量检查语法
        m_multi = _CHECK_NODES_RE.match(action)
        if m_multi:
            content = m_multi.group(1).strip()
            if not content:
                results.append(""); valids[i] = 0; continue
            try:
                node_ids = [int(x.strip()) for x in content.split(",") if x.strip()]
                # 限制每次查询的节点上限（防止大模型白嫖无脑探索所有节点）
                if 0 < len(node_ids) <= MAX_NODES_PER_STEP:
                    results.append(action)
                    continue
            except ValueError: pass
            results.append(""); valids[i] = 0; continue

        # 解析 check_graph:2-hop+sim,30 语法
        if action.startswith("check_graph:"):
            try:
                # 兼容中文逗号
                content = action[len("check_graph:"):].strip().replace("，", ",")
                params = content.split(",")
                if len(params) != 2: raise ValueError
                view_mode = params[0].strip()
                num_match = re.search(r"\d+", params[1].strip())
                if not num_match: raise ValueError
                max_nodes = int(num_match.group(0))
                # 严格校验传入的 view_mode
                if view_mode not in VALID_VIEW_MODES or max_nodes <= 0: raise ValueError
                results.append(f"check_graph:{view_mode},{max_nodes}")
                continue
            except:
                results.append(""); valids[i] = 0; continue

        # 解析 paint:节点ID,类别
        m_paint = _PAINT_RE.match(action)
        if m_paint:
            nid = m_paint.group(1).strip()
            cls = m_paint.group(2).strip()
            results.append(f"paint:{nid},{cls}")
            continue

        if action.startswith("final:") or action.startswith("submit:"):
            results.append(action)
            continue

        # 没有命中任何合法的预设动作规范
        results.append("")
        valids[i] = 0

    return results, valids