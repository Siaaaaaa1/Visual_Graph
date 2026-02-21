import re
from typing import Any, Dict, List, Tuple

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import FullSequenceSearchMemory

# =================================================================
# 1. Task Instructions
# =================================================================

GRAPH_SEARCH_TASK_INSTRUCTION = """
# GRAPH SEARCH TASK INSTRUCTION

You are a graph reasoning agent. Your task is to predict the correct semantic category of a provided center node.

---

## 1. Initial State & Context
* **Input:** You are provided with a **center node** (ID), its statistical metadata, and an **initial observation** (which may include text, a graph view, or be restricted based on the current setting).
* **History & Updates:** From step 2 onwards, you will additionally see the summaries and actions from previous steps, observations from the most recent steps, and the graph visualization updated according to your previous requests.

## 2. Reasoning Guidelines
Your reasoning must be **holistic** and integrate multiple dimensions:
* **Textual Content:** The raw text of the center node (if available) and its neighbors.
* **Graph Topology:** Distribution and connectivity of categories.
* **Neighboring Node Information:** The text, predicted categories, and statistical context of surrounding nodes.
The Center Node may relate to multiple categories; therefore, you must carefully synthesize the Center Node's text, graph topology, and neighboring node information to make a final judgment. There is only one correct final answer.

## 3. Objective & Accuracy
* **Goal:** Submit the correct category. Rewards are strictly based on **accuracy**. Minimize exploration steps while maintaining accuracy. You must submit the final answer within 10 steps.
* **Verification First:** You are a fact-verification agent. Do not guess. 
* **Grounding:** If the information is insufficient, utilize `check_node` to read actual text or `check_graph` to expand your scope.

## 4. Interaction Actions
At each step, you must choose **exactly one** of the following actions:

### A. Inspect Node Text
Read the actual content of specific nodes.
* **Command:** `<action>check_node:ID</action>` or `<action>check_nodes:[ID1, ID2, ...]</action>`

### B. Update Graph Visualization
Render a subgraph centered on the target node.
* **Command:** `<action>check_graph:view_mode,max_nodes</action>`
* **Valid View Modes:** `1-hop`, `2-hop`, `sim`, `1-hop+sim`, `2-hop+sim`.

### C. Submit Answer
Submit the final category when confident.
* **Command:** `<action>final:Category Name</action>`

---

## 5. Response Format
You must strictly follow this sequence in your output:

1. **Reasoning Process:** First, output your natural stream of consciousness to analyze the current state, verify hypotheses, and plan the next step.
2. **Knowledge Synthesis:** Consolidate your reasoning into a detailed summary. **This should not be a simple status update.** It must capture the **new knowledge derived**, **logical inferences made**, and **reflection on evidence quality**. Wrap inside `<summary>...</summary>`. 
3. **Action:** Output your chosen action on a new line wrapped in `<action>...</action>`.
"""

# =================================================================
# 2. Few-Shot Examples (Refined for Deep Summary)
# =================================================================

GRAPH_SEARCH_FEW_SHOT = """
---
## 6. Demonstration Examples

### Example 1: Simple Case (High Homophily, Requires Interpretability)
**Context:** Center Node 38491 (Category Unknown). Initial 1-hop view shows 8 neighbors: 6 labeled "Natural Language Processing" (NLP), 2 labeled "Information Retrieval" (IR).

**Step 1:**
<think>
Currently, I cannot see the text of the center node 38491.
The initial view shows a cluster dominated by "Natural Language Processing" (NLP), but there are two "Information Retrieval" (IR) nodes mixed in.
Based solely on neighbor distribution, I am 75% confident it is NLP, but I cannot rule out IR.
To confirm, I must read the text to see if the center node is about foundational NLP research or an IR application.
I also need to understand why the IR nodes are connected. If they are citing the center node as a base technology, the center node is likely NLP.
Plan:
1. Read text of center node 38491 (Critical).
2. Read one typical NLP neighbor (9341) as a baseline.
3. Read one outlier IR neighbor (1205) to analyze the citation motivation.
</think>
<summary>While the majority of neighbors (75%) suggest NLP, the presence of IR nodes creates ambiguity. I hypothesize the center node provides a foundational technique used by IR. To validate this, I need to verify the specific textual relationship: is the center node the 'method' (NLP) or the 'application' (IR)?</summary>
<action>check_nodes:[38491, 9341, 1205]</action>

**Observation:**
- 38491 (Center): "Attention is all you need... We propose a new simple network architecture, the Transformer, based solely on attention mechanisms..."
- 9341 (NLP): "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding..."
- 1205 (IR): "Deep relevance ranking using enhanced Transformer representations for web search..."

**Step 2:**
<think>
The evidence is now very clear:
1. **Center Node (38491):** The text explicitly proposes the "Transformer" architecture and "Attention mechanisms". This is a foundational model innovation in Deep Learning/NLP.
2. **NLP Neighbor (9341):** Cites the center node for pre-training language models, aligning with the NLP community.
3. **IR Outlier Neighbor (1205):** Text mentions "using enhanced Transformer representations". This indicates the IR node is *applying* the technology from the center node.
Citation Chain: IR Node (1205) -> Cites -> NLP Foundation Node (38491).
This "Application layer citing Foundation layer" relationship explains the heterogeneous connection and confirms the center node belongs to the upstream NLP category.
Conclusion: Center node is definitely NLP.
</think>
<summary>The text confirms the center node is the foundational paper for 'Transformer' architectures. The heterogeneous IR neighbor (1205) is citing it as a tool for ranking, while the NLP neighbor (9341) builds upon it. The causality flow (Tool -> Application) proves the center node belongs to the method-providing category: Natural Language Processing.</summary>
<action>final:Natural Language Processing</action>


### Example 2: Complex Case (Deep Reasoning: Wide Sampling & Multi-Angle Verification)
**Context:** Center Node 8712 (Unknown). Initial 1-hop view is extremely sparse, showing only 3 neighbors: 2 "Machine Learning" (ML), 1 "Biomolecular Engineering" (Bio).

**Step 1:**
<think>
The initial information is very limited and potentially misleading. The center node is sandwiched between two distinct fields: ML and Bio. The sample size (3 neighbors) is too small for statistical significance.
Since I cannot see the center node's text, any guess based on micro-neighbors is unreliable.
I need to establish a "Global View" first. I need to know if this node sits in a larger ML cluster or a Bio cluster in the broader academic network.
Strategy: Expand to 2-hop range combined with similarity view.
</think>
<summary>The local neighborhood is too sparse (N=3) and mixed to be reliable. I hypothesize the node acts as a bridge between ML and Bio. To avoid sampling bias, I must shift from local inspection to global topology analysis using a 2-hop view to determine the dominant community structure.</summary>
<action>check_graph:2-hop+sim,50</action>

**Observation:** Graph updated. View zooms out, showing a massive "Machine Learning" cluster on the left and a smaller "Biomolecular Engineering" cluster on the right. Center node 8712 acts as a "bridge", but connection density seems slightly biased towards the ML cluster.

**Step 2:**
<think>
The global view implies a topological affinity to ML, but this is just a vague visual guess.
To verify this, I need to zoom back in and identify the specific key neighbors connecting it to these clusters so I can read their text.
Strategy: Switch to 1-hop view to get IDs of key neighbors.
</think>
<summary>Macro-topology suggests a slight ML bias, but visual proximity is not factual evidence. I need to convert this topological hypothesis into textual verification. The immediate next step requires acquiring specific neighbor IDs to check if the 'bridge' nature is due to algorithmic contribution (ML) or biological application (Bio).</summary>
<action>check_graph:1-hop,10</action>

**Observation:** 1-hop view confirms 3 key neighbors: Node 33902 (ML), Node 4410 (ML), Node 7114 (Bio). Also identifies boundary nodes 2201 (ML) and 9982 (Unknown).

**Step 3:**
<think>
Key is text evidence. I need to build a hypothesis.
Since this is a hard case, just checking one node is insufficient. I need to check the center node and its conflicting neighbors (ML vs Bio).
To ensure diversity, I will check 4 nodes.
Plan:
1. Center Node 8712
2. ML Neighbor 33902
3. ML Neighbor 4410
4. Bio Neighbor 7114
</think>
<summary>I have isolated the conflicting neighbors. To resolve the 'Bridge Node' ambiguity, I must perform high-density sampling. I will compare the center node's self-description against the citation contexts of both ML and Bio neighbors to distinguish between 'methodological innovation' vs 'domain application'.</summary>
<action>check_nodes:[8712, 33902, 4410, 7114]</action>

**Observation:**
- 8712 (Center): "Graph Neural Networks for Molecule Property Prediction... We introduce a novel message passing architecture invariant to rotation..."
- 33902 (ML): "Benchmarking Equivariant GNNs... We compare with [8712]..."
- 4410 (ML): "Theoretical analysis of message passing... [8712] proposes a practical implementation..."
- 7114 (Bio): "High-throughput screening... We utilize the architecture from [8712]..."

**Step 4:**
<think>
Preliminary Analysis:
- Center (8712) proposes a "novel architecture" (Algorithmic contribution).
- ML neighbors (33902, 4410) treat it as a benchmark/theory subject (Peer perspective).
- Bio neighbor (7114) uses it as a tool (Downstream perspective).
Evidence points strongly to ML.
However, to avoid "survivorship bias", I must check the boundary nodes I saw earlier.
Plan: Check boundary nodes and one more heterogenous node to confirm consensus.
</think>
<summary>Textual evidence strongly supports ML: the center node claims algorithmic novelty (architecture), which ML neighbors critique theoretically, while Bio neighbors merely apply it. However, to ensure this isn't a local anomaly, I must validate this pattern against boundary nodes and other outliers.</summary>
<action>check_nodes:[2201, 9982, 1005]</action>

**Observation:**
- 2201 (ML): "Geometric Deep Learning extension..." (Cites center as foundation)
- 9982 (Chem): "Computational methods..." (Lists center as an ML method)
- 1005 (Bio): "Predicting protein interactions..." (Applies the center's algorithm)

**Step 5:**
<think>
I have now checked 7 neighbors + center.
All ML nodes treat it as an algorithm; all Bio/Chem nodes treat it as a tool.
To be absolutely safe (since this is a hard case), I will check "Authority" nodes in the 2-hop range, specifically Survey papers.
I saw Node 5521 (ML Survey) and 6632 (Chem Survey) in Step 1.
</think>
<summary>Micro-level consistency is achieved: 7 different nodes confirm the 'Algorithm vs Tool' distinction. To reach 100% confidence and rule out any niche sub-field exceptions, I will seek macro-level confirmation via domain-specific survey papers (Authority Nodes).</summary>
<action>check_nodes:[5521, 6632]</action>

**Observation:**
- 5521 (ML Survey): "A survey on geometric deep learning... We categorize [8712] as a rotation-invariant GNN method."
- 6632 (Chem Survey): "Machine learning methods in quantum chemistry... [8712] is a popular ML technique."

**Step 6:**
<think>
Final Review:
1. **Self Text:** Contribution is Rotation Invariant Architecture (Algorithm).
2. **Micro Neighbors:** Peers (ML) analyze it; Users (Bio) apply it.
3. **Macro Consensus:** Surveys from both fields categorize it as ML.
Evidence is closed-loop and consistent.
Conclusion: Probability of Machine Learning is near 100%.
</think>
<summary>Evidence is conclusive across all levels: 1. Self-claims (Architecture), 2. Neighbor citations (Theory vs Application), and 3. Authority surveys (Categorization). The node is a Machine Learning method applied to Biology, not a biological discovery itself.</summary>
<action>final:Machine Learning</action>


### Example 3: Counter-Intuitive Case (The Minority Report)
**Context:** Center Node 5021 (Unknown). Initial 1-hop view shows 10 neighbors: 8 "Databases" (DB), 2 "Artificial Intelligence" (AI).

**Step 1:**
<think>
This is a deceptive initial state. 80% of neighbors are "Databases" (DB). Simple voting would suggest DB.
However, there are 2 "Artificial Intelligence" (AI) minority nodes.
In academic networks, the majority is not always the source. Sometimes, a seminal AI paper is heavily cited by the DB community (e.g., for optimization).
I must not blindly follow the majority.
Key: Read text to identify who are the "peers" and who are the "appliers".
Plan:
1. Read Center Node 5021 (Core).
2. Read a typical DB neighbor 6001 (Majority).
3. Read an AI neighbor 7005 (Minority).
</think>
<summary>The overwhelming majority (80% DB) suggests a Database classification, but the minority AI nodes raise a 'Cross-Domain' red flag. I hypothesize the majority might represent 'impact' rather than 'origin'. I must strictly compare the textual depth of the Majority (DB) vs Minority (AI) to find the true academic home.</summary>
<action>check_nodes:[5021, 6001, 7005]</action>

**Observation:**
- 5021 (Center): "Learned Index Structures... We replace B-Trees with neural network models to improve lookup speed..."
- 6001 (DB): "Optimizing database query performance using learned indexes... We implement [5021] in PostgreSQL..."
- 7005 (AI): "Neural architecture search for efficient index models... We improve the training of [5021]..."

**Step 2:**
<think>
Analysis:
1. **Center (5021):** Core idea is replacing B-Trees with "Neural Networks". This is "AI for DB".
2. **DB Neighbor (6001):** *Implements* it in a DB system (Downstream application).
3. **AI Neighbor (7005):** *Improves* the model training (Upstream algorithm).
Classification dilemma: Application (DB) vs Method (AI)?
Usually, if the core contribution is a Neural Network model, it is AI/ML, even if applied to DB.
The high DB neighbor count is because the DB community is testing this new tool.
I need to check if the DB neighbors discuss system architecture (DB) or treat the AI as a black box.
Plan: Check more neighbors to verify depth of discussion.
</think>
<summary>The center node proposes a Neural Network model (Method) to solve a B-Tree problem (Application). Initial checks show DB neighbors implementing it and AI neighbors refining it. The core contribution seems to be the *model* itself. I need to verify if the DB community treats this as a 'black box' tool, which would confirm AI as the source domain.</summary>
<action>check_nodes:[6002, 7006]</action>

**Observation:**
- 6002 (DB): "A survey of machine learning for database systems... [5021] is a pioneering work in ML-based indexing."
- 7006 (AI): "Gradient descent optimization for non-differentiable index structures..."

**Step 3:**
<think>
Deep Reasoning:
- DB neighbor (6002) calls it "ML-based indexing".
- AI neighbor (7006) discusses "Gradient descent" (Pure optimization).
The core contribution is the *Neural Network Model*.
Most importantly, I need to check the venue. Is it NeurIPS/ICML (AI) or SIGMOD/VLDB (DB)?
I will check the 2-hop view for venue nodes or authorities.
</think>
<summary>The conceptual evidence is solid: DB papers label it as 'ML-based', and AI papers discuss its mathematical optimization. The intellectual ownership belongs to AI. To make the final decision irrefutable, I will search for 'Venue Nodes' in the 2-hop graph to pinpoint the exact publication community (e.g., NeurIPS vs SIGMOD).</summary>
<action>check_graph:2-hop,10</action>

**Observation:** 2-hop view shows connection to Node 8802 ("NeurIPS Proceedings").

**Step 4:**
<think>
The 2-hop view provides the decisive evidence:
The node is connected to 8802 ("NeurIPS Proceedings"), a top-tier AI conference.
This confirms its academic root is AI/ML.
Although it heavily impacted the DB community (hence 80% DB neighbors), it is an AI paper.
I must reject the majority vote (DB) and support the minority truth (AI).
</think>
<summary>Decisive evidence found: The paper is linked to 'NeurIPS Proceedings', a top-tier AI venue. This confirms that despite the high volume of Database citations (Application), the paper's origin and methodology are rooted in Artificial Intelligence. I am overruling the majority neighbor count.</summary>
<action>final:Artificial Intelligence</action>
"""

# =================================================================
# 3. Prompt Templates
# =================================================================

GRAPH_SEARCH_TEMPLATE_NO_HIS = """{task_instruction}
{few_shot}

=== Initial State ===
{initial_state}

=== Current Observation ===
Current Visual View: <image>

Current step: 1

Response Format:
1. First, analyze the current initial information (center node text/stats and graph view) to decide your first step.
2. Next, consolidate your reasoning into a detailed summary. **This should not be a simple status update.** It must capture the **new knowledge derived**, **logical inferences made**, and **reflection on evidence quality**. Wrap inside <summary>...</summary> tags.
3. Finally, on a new line, output the chosen action wrapped in <action>...</action> tags.
"""

GRAPH_SEARCH_TEMPLATE_WITH_HIS = """{task_instruction}
{few_shot}

=== Initial State (Reference) ===
{initial_state}

=== History ===
(The following is a log of your previous actions and observations):
{memory_context}

=== Current Observation ===
Current Visual View (Snapshot after your last action): <image>

Current step: {step_count}

Response Format:
1. First, review the history and analyze the current state. Output your thinking naturally.
2. Next, consolidate your reasoning into a detailed summary. **This should not be a simple status update.** It must capture the **new knowledge derived**, **logical inferences made**, and **reflection on evidence quality**. Wrap inside <summary>...</summary> tags.
3. Finally, on a new line, output the chosen action wrapped in <action>...</action> tags.
"""

# =================================================================
# 4. Environment Manager
# =================================================================

class GraphSearchEnvironmentManager(EnvironmentManagerBase):
    """
    Manager for the Graph Search Environment.
    """

    def __init__(self, envs, projection_f, config):
        self.memory = FullSequenceSearchMemory()
        super().__init__(envs, projection_f, config)
        
        # Regex patterns to parse the model output
        self._think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        self._summary_pattern = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
        self._action_pattern = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        text_obs, image_obs, infos = self.envs.reset(kwargs=kwargs)
        self.initial_states = text_obs
        self.memory.reset(batch_size=len(text_obs))

        observations = {
            "text": self.build_text_obs(init=True), 
            "image": image_obs, 
            "anchor": text_obs.copy(), 
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        summaries = []
        thinks = []
        
        for raw_text in text_actions:
            # Extract Summary
            s_match = self._summary_pattern.search(raw_text)
            if s_match:
                summaries.append(s_match.group(1).strip())
            else:
                summaries.append("No summary provided.")
            
            # Extract Think
            t_match = self._think_pattern.search(raw_text)
            if t_match:
                thinks.append(t_match.group(1).strip())
            else:
                # Fallback: if no explicit think tags, but summary exists, treat pre-summary text as thought
                if s_match:
                    pre_summary = raw_text.split("<summary>")[0].strip()
                    thinks.append(pre_summary)
                else:
                    thinks.append(None)

        # Map actions to environment specific format
        actions, valids = self.projection_f(text_actions)

        # Execute step in environment
        next_text_obs, next_image_obs, rewards, dones, infos = self.envs.step(actions)

        # Store experience in memory
        self.memory.store({
            "search": actions,
            "information": next_text_obs,
            "summary": summaries 
        })

        # Build next observation
        next_observations = {
            "text": self.build_text_obs(init=False), 
            "image": next_image_obs, 
            "anchor": next_text_obs.copy(),
        }

        # Update Info dicts
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["parsed_think"] = thinks[i]
            info["parsed_summary"] = summaries[i]
            a_match = self._action_pattern.search(text_actions[i])
            info["parsed_action_content"] = a_match.group(1).strip() if a_match else "No Action Found"

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, init: bool) -> List[str]:
        batch_size = len(self.initial_states)
        rendered_prompts: List[str] = []

        if not init:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search",
                summary_key="summary"
            )
        else:
            memory_ctx = [""] * batch_size

        for i in range(batch_size):
            if init:
                prompt = GRAPH_SEARCH_TEMPLATE_NO_HIS.format(
                    task_instruction=GRAPH_SEARCH_TASK_INSTRUCTION,
                    few_shot=GRAPH_SEARCH_FEW_SHOT,
                    initial_state=self.initial_states[i],
                )
            else:
                prompt = GRAPH_SEARCH_TEMPLATE_WITH_HIS.format(
                    task_instruction=GRAPH_SEARCH_TASK_INSTRUCTION,
                    few_shot=GRAPH_SEARCH_FEW_SHOT,
                    initial_state=self.initial_states[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            rendered_prompts.append(prompt)

        return rendered_prompts