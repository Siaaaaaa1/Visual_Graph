import re
from typing import List, Tuple

# 匹配 <action> 标签块
_ACTION_BLOCK = re.compile(
    r"<action>(.*?)</action>",
    re.IGNORECASE | re.DOTALL
)

_ACTION_TAG = re.compile(r"<action>", re.IGNORECASE)

# 动作特定正则
_CHECK_NODE_RE = re.compile(r"^check_node:(\d+)$")
_CHECK_NODES_RE = re.compile(r"^check_nodes:\[(.*?)\]$")
_CHECK_GRAPH_RE = re.compile(r"^check_graph:(.+?)$") 

MAX_NODES_PER_STEP = 5

def graph_search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    将模型原始输出投影为合法的环境动作字符串。
    
    Args:
        actions: 模型输出的字符串列表。
        
    Returns:
        results: 清洗后的动作字符串列表。
        valids: 掩码列表 (1 表示有效动作，0 表示非法)。
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    for i, raw in enumerate(actions):
        # 1. 唯一性与格式检查: 必须且只能包含一个 <action> 块
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

        # 2. 匹配: check_node:<id> (单节点查询)
        if _CHECK_NODE_RE.match(action):
            results.append(action)
            continue

        # 3. 匹配: check_nodes:[id1, id2] (多节点批量查询)
        m_multi = _CHECK_NODES_RE.match(action)
        if m_multi:
            content = m_multi.group(1).strip()
            if not content:
                results.append("")
                valids[i] = 0
                continue
            try:
                # 解析 ID 列表，并根据 MAX_NODES_PER_STEP 进行截断/校验
                node_ids = [int(x.strip()) for x in content.split(",") if x.strip()]
                if 0 < len(node_ids) <= MAX_NODES_PER_STEP:
                    results.append(action)
                    continue
            except ValueError:
                pass
            results.append("")
            valids[i] = 0
            continue

        # 4. 匹配: check_graph:hop_mode,rank_mode,max_nodes
        # 注意：此处解析逻辑比 Env 中严格，强制要求提供所有3个参数
        if action.startswith("check_graph:"):
            try:
                params = action.split(":", 1)[1].split(",")

                if len(params) != 3:
                    raise ValueError

                hop_mode = params[0].strip()     # e.g., "1-hop"
                rank_mode = params[1].strip()    # e.g., "sim"
                max_nodes = int(params[2].strip())

                if hop_mode not in ["1-hop", "2-hop"]:
                    raise ValueError
                if rank_mode not in ["hop", "sim"]:
                    raise ValueError
                if max_nodes <= 0:
                    raise ValueError

                results.append(action)
                continue
            except:
                results.append("")
                valids[i] = 0
                continue


        # 5. 匹配: final:<category>
        if action.startswith("final:"):
            results.append(action)
            continue

        # 无法匹配任何已知模式
        results.append("")
        valids[i] = 0

    return results, valids