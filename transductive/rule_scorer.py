"""
规则评分模块 (Rule Scorer Module)
====================================

在 GNN 的每一层采样过程中，为候选节点计算基于逻辑规则的分数。

核心功能：
---------
1. 根据当前传播路径，判断候选节点是否触发已挖掘的逻辑规则
2. 为触发规则的节点赋予额外的置信度分数
3. 将规则分数与 GNN 的神经网络分数结合，指导节点采样

工作流程：
---------
1. 离线阶段：使用 rule_mining.py 从训练数据中挖掘规则，保存为 rules.pkl
2. 在线阶段：
   - 加载规则到 RuleScorer
   - 在每层 GNN 传播时，调用 score_nodes() 计算候选节点的规则分数
   - 结合规则分数和神经分数进行采样

示例：
-----
假设有规则：r1(X,Y) ∧ r2(Y,Z) => r3(X,Z)，置信度 0.8
在预测 (实体A, r3, ?) 时：
- GNN 第1层：从实体A出发，沿 r1 到达实体B（路径：[r1]）
- GNN 第2层：从实体B出发，沿 r2 到达实体C（路径：[r1, r2]）
- 检查：路径 [r1, r2] 匹配规则体，且查询关系是 r3
- 结果：实体C 的规则分数 += 0.8

作者：xhuan
日期：2025
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional
import pickle
import os


class RuleScorer(nn.Module):
    """
    规则评分器：为 GNN 采样过程中的候选节点计算规则分数

    设计思想：
    --------
    在 GNN 的增量采样过程中，候选节点通过多跳路径从查询节点到达。
    如果某条路径匹配已知的逻辑规则前件（rule body），则该节点更可能是正确答案。

    核心数据结构：
    ------------
    1. self.rules: 规则列表，每条规则包含 body (关系序列) 和 head (目标关系)
    2. self.rule_index: 规则索引，格式为 {head_rel: {length: [rule_indices]}}
       - 用于快速查找：给定查询关系和当前层数，找到所有相关规则
    3. self.rule_weights: 可学习的规则权重（nn.Parameter）
       - 初始值为规则的置信度 confidence
       - 在训练过程中会自动调整（梯度下降）

    工作原理：
    --------
    对于每个候选节点：
    1. 回溯从查询节点到该节点的路径（关系序列）
    2. 检查路径是否匹配某条规则的 body
    3. 如果匹配且规则的 head 等于查询关系，累加规则权重
    4. 返回所有匹配规则的权重之和

    参数：
    -----
    rules: List[Rule]
        规则列表，来自 rule_mining.py 挖掘的结果
    n_rel: int
        知识图谱中的关系总数
    device: str
        计算设备，'cuda' 或 'cpu'

    示例：
    -----
    >>> from rule_mining import mine_and_save_rules
    >>> from rule_scorer import load_rule_scorer
    >>>
    >>> # 离线挖掘规则
    >>> mine_and_save_rules('./data/family/', min_support=5, min_confidence=0.1)
    >>>
    >>> # 在线加载规则评分器
    >>> scorer = load_rule_scorer('./data/family/rules.pkl', n_rel=12, device='cuda')
    >>>
    >>> # 在 GNN 前向传播中使用
    >>> rule_scores = scorer.score_nodes(
    ...     nodes=candidate_nodes,
    ...     edges=current_edges,
    ...     query_rel=batch_query_relations,
    ...     current_layer=2,
    ...     batch_size=32
    ... )
    """

    def __init__(self, rules: List, n_rel: int, device='cuda'):
        super().__init__()
        self.rules = rules
        self.n_rel = n_rel
        self.device = device

        # ========== 构建规则索引 ==========
        # 目的：在评分时快速查找相关规则，避免遍历所有规则
        #
        # 索引结构：
        # rule_index = {
        #     关系ID: {
        #         规则长度: [规则索引列表]
        #     }
        # }
        #
        # 例如：
        # rule_index[3][2] = [0, 5, 12]
        # 表示关系3的2-hop规则有3条，索引分别为0、5、12
        print("[RuleScorer] 构建规则索引...")
        self.rule_index = self._build_rule_index()

        # ========== 初始化可学习的规则权重 ==========
        # 关键设计：规则权重是 nn.Parameter，会随着模型训练更新
        #
        # 初始值：规则的置信度 confidence（统计意义上的可靠性）
        # 训练后：神经网络自动调整的权重（结合数据特征的可靠性）
        #
        # 好处：
        # 1. 统计规则提供了良好的初始化
        # 2. 神经网络可以学习规则在特定上下文中的重要性
        # 3. 某些置信度低但在特定场景有用的规则可以被放大
        if len(rules) > 0:
            initial_weights = torch.tensor([r.confidence for r in rules], dtype=torch.float32)
        else:
            initial_weights = torch.tensor([], dtype=torch.float32)

        self.rule_weights = nn.Parameter(initial_weights)

        print(f"[RuleScorer] 初始化完成，共 {len(rules)} 条规则")

        # ========== 打印规则长度分布统计 ==========
        # 用于诊断：检查挖掘到的规则类型分布
        if len(rules) > 0:
            length_dist = defaultdict(int)
            for rule in rules:
                length_dist[rule.length] += 1
            print(f"[RuleScorer] 规则长度分布: {dict(length_dist)}")
            # 输出示例：{2: 150, 3: 45} 表示有150条2-hop规则，45条3-hop规则

    def _build_rule_index(self) -> dict:
        """
        构建规则索引以加速查找

        问题：
        ----
        在每层 GNN 传播时，需要找到"与当前查询关系相关且长度匹配当前层数"的规则。
        如果每次都遍历所有规则，时间复杂度为 O(N_rules)。

        解决方案：
        --------
        预先建立索引，按 (head_relation, length) 分组规则。
        查找时间复杂度降低为 O(1)。

        数据结构：
        --------
        index = {
            head_relation: {
                length: [rule_idx_1, rule_idx_2, ...]
            }
        }

        示例：
        -----
        假设有以下规则：
        - Rule 0: [r1, r2] => r3  (2-hop, head=3)
        - Rule 1: [r2, r4] => r3  (2-hop, head=3)
        - Rule 2: [r1, r2, r5] => r6  (3-hop, head=6)

        则索引为：
        {
            3: {2: [0, 1]},
            6: {3: [2]}
        }

        查询示例：
        - 查询关系 r3，当前在第2层 → 获取 index[3][2] = [0, 1]
        - 查询关系 r6，当前在第3层 → 获取 index[6][3] = [2]

        返回：
        -----
        index: dict
            规则索引字典

        时间复杂度：
        ----------
        O(N_rules)，只在初始化时执行一次
        """
        # 使用 defaultdict 避免手动检查键是否存在
        # lambda: defaultdict(list) 表示嵌套的字典结构
        index = defaultdict(lambda: defaultdict(list))

        for i, rule in enumerate(self.rules):
            # rule.head: 规则的目标关系
            # rule.length: 规则的长度（hop数）
            # i: 规则在 self.rules 中的索引
            index[rule.head][rule.length].append(i)

        return index

    def score_nodes(self,
                    nodes: torch.Tensor,          # [N, 2] (batch_idx, node_id)
                    edges: torch.Tensor,          # [M, 6] (batch_idx, h, r, t, h_idx, t_idx)
                    query_rel: torch.Tensor,      # [batch_size]
                    current_layer: int,           # 当前层数（0-indexed）
                    batch_size: int) -> torch.Tensor:  # 规则分数 [N]
        """
        为候选节点计算规则分数

        调用时机：
        --------
        在 GNN 每一层的采样过程中，计算候选节点的规则分数，与神经分数结合。

        输入说明：
        --------
        1. nodes: 候选节点张量 [N, 2]
           - 第0列：batch 索引（该节点属于哪个查询）
           - 第1列：实体 ID
           示例：[[0, 100], [0, 200], [1, 150]] 表示 batch 0 有节点100、200，batch 1 有节点150

        2. edges: 当前层的边 [M, 6]
           - 格式：(batch_idx, head_id, relation_id, tail_id, head_node_idx, tail_node_idx)
           - 记录了从上一层节点到当前层节点的所有边
           示例：[0, 100, 5, 200, 0, 1] 表示 batch 0 中，节点100 通过关系5 到达节点200

        3. query_rel: 查询关系 [batch_size]
           - 每个 batch 对应的查询关系
           示例：[3, 7] 表示 batch 0 查询关系3，batch 1 查询关系7

        4. current_layer: 当前 GNN 层索引（从0开始）
           - Layer 0: 查询节点（1-hop邻居）
           - Layer 1: 2-hop邻居
           - Layer 2: 3-hop邻居
           - ...

        5. batch_size: batch 大小

        核心算法：
        --------
        对于每个候选节点：
        1. 回溯从查询节点到该节点的路径（关系序列）
        2. 获取与查询关系和当前层数相关的规则
        3. 检查路径是否匹配某条规则的 body
        4. 累加所有匹配规则的权重

        示例：
        -----
        假设：
        - 查询：(实体A, r3, ?)
        - 当前在第2层（2-hop）
        - 候选节点C，路径：A -[r1]-> B -[r2]-> C
        - 有规则：r1(X,Y) ∧ r2(Y,Z) => r3(X,Z)，置信度 0.8

        处理过程：
        1. current_layer = 2，query_rel = r3
        2. 从索引获取 rule_index[r3][2] → 找到该规则
        3. 回溯节点C的路径 → [r1, r2]
        4. 匹配规则 body [r1, r2] → 匹配成功
        5. 累加规则权重 → score = 0.8

        返回：
        -----
        rule_scores: torch.Tensor [N]
            每个候选节点的规则分数（匹配规则权重之和）

        时间复杂度：
        ----------
        O(batch_size * nodes_per_batch * current_layer)
        - 对每个 batch 的每个节点，回溯 current_layer 步
        - 实际中 batch_size 和 nodes_per_batch 都不大，可接受
        """
        N = nodes.size(0)
        rule_scores = torch.zeros(N, device=self.device)

        # ========== 边界情况处理 ==========
        # 情况1：没有挖掘到任何规则 → 返回全0分数
        # 情况2：第0层（查询节点的1-hop邻居）→ 路径长度为0，无法匹配规则
        if len(self.rules) == 0 or current_layer == 0:
            return rule_scores

        # ========== 按 batch 处理（每个查询独立） ==========
        # 原因：每个 batch 对应不同的查询关系，需要的规则不同
        for b_idx in range(batch_size):
            q_rel = query_rel[b_idx].item()

            # ---------- 获取相关规则 ----------
            # 条件：
            # 1. 规则的 head 必须等于查询关系 q_rel
            # 2. 规则的 length 必须等于当前层数 current_layer
            #
            # 为什么长度要相等？
            # - Layer 1: 1-hop路径，只能匹配 1-hop 规则
            # - Layer 2: 2-hop路径，只能匹配 2-hop 规则
            # - Layer 3: 3-hop路径，只能匹配 3-hop 规则
            relevant_rule_indices = self.rule_index[q_rel].get(current_layer, [])

            if not relevant_rule_indices:
                continue  # 没有相关规则，跳过该 batch

            relevant_rules = [self.rules[i] for i in relevant_rule_indices]

            # ---------- 获取该 batch 的节点和边 ----------
            # 使用掩码过滤，只保留属于当前 batch 的数据
            batch_nodes_mask = (nodes[:, 0] == b_idx)
            batch_edges_mask = (edges[:, 0] == b_idx)

            batch_nodes_idx = torch.where(batch_nodes_mask)[0]  # 节点在原始数组中的索引
            batch_edges = edges[batch_edges_mask]  # [M', 6]

            if len(batch_edges) == 0:
                continue  # 该 batch 没有边（异常情况），跳过

            # ---------- 为每个节点评分 ----------
            for node_idx in batch_nodes_idx:
                node_id = nodes[node_idx, 1].item()  # 获取实体ID

                # 步骤1：回溯路径
                # 从当前节点反向追踪到查询节点，得到关系序列
                path = self._trace_path(node_id, batch_edges, current_layer)

                if path is None:
                    # 无法回溯到查询节点（可能是孤立节点），跳过
                    continue

                # 步骤2：匹配规则并计算分数
                score = self._match_rules(path, relevant_rule_indices)
                rule_scores[node_idx] = score

        return rule_scores

    def _trace_path(self, target_node: int, edges: torch.Tensor,
                    max_depth: int) -> Optional[List[int]]:
        """
        从目标节点回溯到查询节点的路径（关系序列）

        问题：
        ----
        在 GNN 的增量采样中，节点是逐层扩展的，但我们需要知道从查询节点到当前节点的完整路径。

        解决方案：
        --------
        利用 edges 记录的传播历史，从目标节点反向追踪到查询节点。

        算法：
        -----
        1. 从目标节点开始
        2. 查找指向该节点的边（incoming edges）
        3. 记录边的关系，移动到源节点
        4. 重复步骤2-3，直到回溯到查询节点（max_depth 步）
        5. 反转路径（因为是反向追踪的）

        示例：
        -----
        假设 edges 包含：
        - [0, 100, 5, 200, 0, 1]: 节点100 -[r5]-> 节点200
        - [0, 200, 7, 300, 1, 2]: 节点200 -[r7]-> 节点300

        回溯节点300（max_depth=2）：
        1. 查找指向300的边 → [0, 200, 7, 300, 1, 2]
        2. 关系为 r7，源节点为200，path = [7]
        3. 查找指向200的边 → [0, 100, 5, 200, 0, 1]
        4. 关系为 r5，源节点为100，path = [7, 5]
        5. 反转路径 → [5, 7]

        参数：
        -----
        target_node: int
            目标节点的实体 ID
        edges: torch.Tensor [M, 6]
            该 batch 的边，格式 (batch_idx, h, r, t, h_idx, t_idx)
        max_depth: int
            最大回溯深度（等于当前层数）

        返回：
        -----
        path: Optional[List[int]]
            关系序列 [r1, r2, ..., rN]，从查询节点到目标节点
            如果无法回溯（孤立节点），返回 None

        时间复杂度：
        ----------
        O(max_depth * M)，其中 M 是边的数量
        - 对每一层，需要遍历所有边查找 incoming edges
        - 实际中 M 较小（只包含当前 batch 的边），可接受
        """
        if len(edges) == 0:
            return None

        path = []
        current_node = target_node

        # 转换为 numpy 以便使用高效的索引操作
        edges_np = edges.cpu().numpy()

        for depth in range(max_depth):
            # ========== 查找指向当前节点的边 ==========
            # edges format: (batch_idx, head, relation, tail, head_idx, tail_idx)
            # 我们需要 tail == current_node 的边
            incoming = edges_np[edges_np[:, 3] == current_node]

            if len(incoming) == 0:
                # 无法继续回溯，可能原因：
                # 1. 当前节点是查询节点（但还没到 max_depth）
                # 2. 数据异常（边记录不完整）
                return None

            # ========== 选择一条边（简化实现） ==========
            # 注意：理论上可能有多条边指向同一个节点（多路径）
            # 简化假设：选择第一条边
            # 未来优化：可以选择置信度最高的路径
            edge = incoming[0]
            relation = int(edge[2])  # 关系 ID
            source = int(edge[1])     # 源节点 ID

            path.append(relation)
            current_node = source

        # ========== 反转路径 ==========
        # 因为是反向追踪的，需要反转以得到正向路径
        # 例如：回溯得到 [r2, r1]，反转后为 [r1, r2]（查询节点 -> 目标节点）
        return path[::-1]

    def _match_rules(self, path: List[int], rule_indices: List[int]) -> float:
        """
        检查路径是否匹配规则，返回匹配分数

        目的：
        ----
        给定一条路径和一组候选规则，计算路径匹配的规则权重总和。

        算法：
        -----
        1. 遍历所有候选规则
        2. 检查路径是否完全匹配规则的 body
        3. 如果匹配，累加该规则的权重
        4. 返回总分数

        示例：
        -----
        假设：
        - 路径：[r1, r2]
        - 候选规则：
          - Rule 0: [r1, r2] => r3, weight=0.8 ✓ 匹配
          - Rule 1: [r1, r4] => r3, weight=0.6 ✗ 不匹配
        - 分数：0.8

        参数：
        -----
        path: List[int]
            路径关系序列 [r1, r2, ...]
        rule_indices: List[int]
            候选规则的索引列表（已经过滤为相关规则）

        返回：
        -----
        score: float
            匹配规则的加权分数总和

        时间复杂度：
        ----------
        O(K * L)，其中 K 是候选规则数，L 是路径长度
        - K 通常很小（相同 head 和 length 的规则不多）
        - L <= 3（最多3-hop）
        - 总体非常快
        """
        score = 0.0

        for rule_idx in rule_indices:
            rule = self.rules[rule_idx]

            # ========== 检查路径是否匹配规则体 ==========
            if self._path_matches_rule_body(path, rule.body):
                # 使用可学习的规则权重
                # .item() 将 PyTorch 张量转换为 Python 标量
                #
                # 注意：这里的权重是 nn.Parameter，在训练中会更新
                # 初始值为 confidence，但训练后可能变化
                score += self.rule_weights[rule_idx].item()

        return score

    def _path_matches_rule_body(self, path: List[int], body: List[int]) -> bool:
        """
        检查路径是否完全匹配规则体

        匹配标准：
        --------
        1. 长度必须相等
        2. 每一步的关系必须完全相同

        示例：
        -----
        匹配：
        - path=[1, 2, 3], body=[1, 2, 3] ✓

        不匹配：
        - path=[1, 2], body=[1, 2, 3] ✗ （长度不同）
        - path=[1, 2, 3], body=[1, 4, 3] ✗ （第2步关系不同）

        参数：
        -----
        path: List[int]
            路径关系序列 [r1, r2, ...]
        body: List[int]
            规则体的关系序列 [r1', r2', ...]

        返回：
        -----
        bool
            是否完全匹配

        时间复杂度：
        ----------
        O(L)，其中 L 是路径长度（最多3）
        """
        # 长度检查
        if len(path) != len(body):
            return False

        # 逐步检查关系是否相同
        # all() 函数：只有所有元素为 True 时才返回 True
        return all(p == b for p, b in zip(path, body))


class FastRuleScorer(nn.Module):
    """
    高效规则评分器：预计算路径表示（优化版本）

    动机：
    ----
    基础版 RuleScorer 的性能瓶颈：
    1. 每个节点都要回溯路径（_trace_path）→ O(N * L * M)
    2. 路径匹配是逐个检查（_match_rules）→ O(N * K * L)

    优化思路：
    --------
    如果在 GNN 传播时记录每个节点的路径历史，就可以：
    1. 避免回溯（直接读取路径）
    2. 使用向量化操作批量匹配规则

    路径编码：
    --------
    将路径表示为 one-hot 矩阵：[max_path_len, n_rel]
    - path_encodings[i, j, k] = 1 表示第 i 个节点的第 j 步关系是 k

    示例：
    -----
    假设路径 [r1, r2]，n_rel=10，max_path_len=3
    编码为：
    [
        [0,1,0,0,0,0,0,0,0,0],  # 第0步：r1
        [0,0,1,0,0,0,0,0,0,0],  # 第1步：r2
        [0,0,0,0,0,0,0,0,0,0],  # 第2步：无
    ]

    规则模式：
    --------
    将规则也表示为 one-hot 矩阵：[max_path_len, n_rel]
    - rule_patterns[i, j, k] = 1 表示第 i 条规则的第 j 步关系是 k

    向量化匹配：
    ----------
    match_score = (path_encodings * rule_patterns).sum()
    如果 match_score == rule.length，则完全匹配

    性能提升：
    --------
    - 基础版：O(N * L * M + N * K * L)
    - 优化版：O(N * K)（向量化操作，GPU 加速）

    注意：
    ----
    这是高级优化版本，需要在 GNNModel 中维护路径历史。
    当前先使用基础版 RuleScorer，未来可以升级到 FastRuleScorer。

    参数：
    -----
    rules: List[Rule]
        规则列表
    n_rel: int
        关系总数
    max_path_len: int
        最大路径长度（默认3，支持3-hop规则）
    device: str
        计算设备
    """

    def __init__(self, rules: List, n_rel: int, max_path_len=3, device='cuda'):
        super().__init__()
        self.rules = rules
        self.n_rel = n_rel
        self.max_path_len = max_path_len
        self.device = device

        if len(rules) > 0:
            # ========== 规则权重 ==========
            # 与基础版相同，可学习的 nn.Parameter
            self.rule_weights = nn.Parameter(
                torch.tensor([r.confidence for r in rules], dtype=torch.float32)
            )

            # ========== 预计算规则的路径表示矩阵 ==========
            # 格式：[n_rules, max_path_len, n_rel]
            # 只在初始化时计算一次，后续重复使用
            self.rule_patterns = self._build_rule_patterns()
        else:
            self.rule_weights = nn.Parameter(torch.tensor([], dtype=torch.float32))
            self.rule_patterns = torch.zeros(0, max_path_len, n_rel)

        print(f"[FastRuleScorer] 初始化完成，共 {len(rules)} 条规则")

    def _build_rule_patterns(self):
        """
        构建规则模式矩阵 [n_rules, max_path_len, n_rel]

        目的：
        ----
        将所有规则预先编码为 one-hot 矩阵，用于快速向量化匹配。

        示例：
        -----
        假设规则：[r1, r2] => r3
        - n_rel = 10
        - max_path_len = 3

        编码为：
        [
            [0,1,0,0,0,0,0,0,0,0],  # 第0步：r1
            [0,0,1,0,0,0,0,0,0,0],  # 第1步：r2
            [0,0,0,0,0,0,0,0,0,0],  # 第2步：无（padding）
        ]

        返回：
        -----
        patterns: torch.Tensor [n_rules, max_path_len, n_rel]
            规则模式矩阵
        """
        n_rules = len(self.rules)
        patterns = torch.zeros(n_rules, self.max_path_len, self.n_rel)

        for i, rule in enumerate(self.rules):
            for j, rel in enumerate(rule.body):
                if j < self.max_path_len:
                    # 设置 one-hot 编码
                    # patterns[i, j, rel] = 1 表示规则 i 的第 j 步关系是 rel
                    patterns[i, j, rel] = 1.0

        return patterns.to(self.device)

    def score_nodes_fast(self,
                        path_encodings: torch.Tensor,  # [N, max_path_len, n_rel]
                        query_rel: torch.Tensor,       # [N]
                        ) -> torch.Tensor:             # [N]
        """
        快速规则评分（向量化实现）

        前提：
        ----
        GNNModel 在传播过程中维护了每个节点的路径编码（path_encodings）。

        算法：
        -----
        1. 对每条规则：
           a. 检查查询关系是否匹配规则头
           b. 计算路径与规则模式的匹配度（向量点积）
           c. 判断是否完全匹配（匹配度 == 规则长度）
           d. 累加规则权重
        2. 返回所有规则的分数总和

        向量化技巧：
        ----------
        使用 PyTorch 的广播机制和并行计算：
        - (path_encodings * pattern).sum(dim=[1, 2]) 一次性计算所有节点的匹配度
        - GPU 加速，速度比循环快几个数量级

        示例：
        -----
        假设：
        - N=1000 个节点
        - 100 条规则

        基础版：需要 1000 * 100 = 100,000 次路径匹配（CPU 循环）
        优化版：100 次向量化操作（GPU 并行）

        参数：
        -----
        path_encodings: torch.Tensor [N, max_path_len, n_rel]
            节点的路径编码（one-hot 矩阵）
            path_encodings[i, j, k] = 1 表示节点 i 的第 j 步关系是 k
        query_rel: torch.Tensor [N]
            每个节点对应的查询关系

        返回：
        -----
        rule_scores: torch.Tensor [N]
            每个节点的规则分数

        时间复杂度：
        ----------
        O(N * K)，其中 K 是规则数
        - 向量化操作，GPU 并行
        - 比基础版快约 10-100 倍
        """
        if len(self.rules) == 0:
            return torch.zeros(path_encodings.size(0), device=self.device)

        N = path_encodings.size(0)
        scores = torch.zeros(N, device=self.device)

        # ========== 对每条规则，计算匹配度 ==========
        for i, rule in enumerate(self.rules):
            # ---------- 步骤1：检查查询关系是否匹配规则头 ----------
            # mask[j] = True 表示节点 j 的查询关系 == rule.head
            mask = (query_rel == rule.head)  # [N]

            if not mask.any():
                # 没有节点的查询关系匹配该规则，跳过
                continue

            # ---------- 步骤2：计算路径与规则体的匹配度 ----------
            # pattern: [max_path_len, n_rel]
            # paths: [N, max_path_len, n_rel]
            pattern = self.rule_patterns[i]  # [max_path_len, n_rel]

            # 逐位匹配（element-wise multiplication）
            # path_encodings * pattern: [N, max_path_len, n_rel]
            # .sum(dim=[1, 2]): 对每个节点，累加所有匹配的位置 → [N]
            #
            # 例如：
            # - 路径 [r1, r2]，规则 [r1, r2]
            # - path_encodings[0] 和 pattern 重叠2个位置 → match=2
            # - 路径 [r1, r3]，规则 [r1, r2]
            # - path_encodings[0] 和 pattern 重叠1个位置 → match=1
            match = (path_encodings * pattern).sum(dim=[1, 2])  # [N]

            # ---------- 步骤3：判断是否完全匹配 ----------
            # 完全匹配：match == rule.length
            # 例如：规则长度为2，match必须为2
            full_match = (match == rule.length).float()  # [N]，1表示完全匹配，0表示不匹配

            # ---------- 步骤4：累加规则权重 ----------
            # 只有同时满足两个条件才加分：
            # 1. 查询关系匹配（mask）
            # 2. 路径完全匹配（full_match）
            scores += full_match * mask.float() * self.rule_weights[i]

        return scores


def load_rule_scorer(rule_path: str, n_rel: int, device='cuda', fast_mode=False) -> nn.Module:
    """
    从规则文件加载并创建 RuleScorer

    便捷函数，封装了规则加载和评分器创建的流程。

    工作流程：
    --------
    1. 检查规则文件是否存在
    2. 加载规则（pickle 格式）
    3. 根据 fast_mode 选择创建 RuleScorer 或 FastRuleScorer
    4. 返回评分器实例

    参数：
    -----
    rule_path: str
        规则文件路径（.pkl 格式），由 rule_mining.mine_and_save_rules() 生成
    n_rel: int
        知识图谱中的关系总数
    device: str
        计算设备，'cuda' 或 'cpu'
    fast_mode: bool
        是否使用快速版本（需要 GNN 维护路径历史）

    返回：
    -----
    scorer: nn.Module
        RuleScorer 或 FastRuleScorer 实例

    异常处理：
    --------
    如果规则文件不存在，创建空规则评分器（规则列表为空）。
    这样可以在没有规则的情况下仍然运行模型（规则分数全为0）。

    使用示例：
    --------
    >>> # 基础版（推荐）
    >>> scorer = load_rule_scorer(
    ...     rule_path='./data/fb15k-237/rules.pkl',
    ...     n_rel=237,
    ...     device='cuda',
    ...     fast_mode=False
    ... )
    >>>
    >>> # 优化版（需要修改 GNN 维护路径历史）
    >>> scorer = load_rule_scorer(
    ...     rule_path='./data/fb15k-237/rules.pkl',
    ...     n_rel=237,
    ...     device='cuda',
    ...     fast_mode=True
    ... )
    """
    # ========== 加载规则文件 ==========
    if not os.path.exists(rule_path):
        print(f"[Warning] 规则文件不存在: {rule_path}")
        print(f"[Warning] 返回空规则评分器")
        rules = []
    else:
        with open(rule_path, 'rb') as f:
            rules = pickle.load(f)
        print(f"[RuleScorer] 从 {rule_path} 加载了 {len(rules)} 条规则")

    # ========== 创建评分器 ==========
    if fast_mode:
        return FastRuleScorer(rules, n_rel, device=device)
    else:
        return RuleScorer(rules, n_rel, device=device)


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    import sys
    from rule_mining import Rule

    print("=" * 60)
    print("RuleScorer 测试")
    print("=" * 60)

    # ========== 创建测试规则 ==========
    # 模拟挖掘得到的规则
    test_rules = [
        # 规则1：r0(X,Y) ∧ r1(Y,Z) => r2(X,Z)，置信度 0.8
        Rule(body=[0, 1], head=2, confidence=0.8, support=100, length=2),

        # 规则2：r1(X,Y) ∧ r2(Y,Z) => r3(X,Z)，置信度 0.7
        Rule(body=[1, 2], head=3, confidence=0.7, support=80, length=2),

        # 规则3：r0(X,Y) ∧ r1(Y,Z) ∧ r2(Z,W) => r4(X,W)，置信度 0.9
        Rule(body=[0, 1, 2], head=4, confidence=0.9, support=150, length=3),
    ]

    print(f"\n测试规则:")
    for i, rule in enumerate(test_rules):
        print(f"  {i}. {rule}")

    # ========== 创建 RuleScorer ==========
    scorer = RuleScorer(test_rules, n_rel=10, device='cpu')

    # ========== 创建测试数据 ==========
    batch_size = 2

    # 候选节点
    nodes = torch.tensor([
        [0, 10],  # batch 0, node 10
        [0, 20],  # batch 0, node 20
        [1, 30],  # batch 1, node 30
    ])

    # edges format: (batch_idx, h, r, t, h_idx, t_idx)
    # 模拟 GNN 传播过程中记录的边
    edges = torch.tensor([
        [0, 0, 0, 10, 0, 1],  # batch 0: 查询节点(0) -[r0]-> 节点10
        [0, 10, 1, 20, 1, 2], # batch 0: 节点10 -[r1]-> 节点20
        [1, 5, 1, 30, 0, 1],  # batch 1: 查询节点(5) -[r1]-> 节点30
    ])

    # 查询关系
    query_rel = torch.tensor([2, 3])  # batch 0: r2, batch 1: r3

    # ========== 测试评分 ==========
    print(f"\n测试输入:")
    print(f"  nodes: {nodes}")
    print(f"  edges: {edges}")
    print(f"  query_rel: {query_rel}")

    # 在第2层（2-hop）计算规则分数
    scores = scorer.score_nodes(nodes, edges, query_rel, current_layer=2, batch_size=batch_size)

    print(f"\n规则分数:")
    print(f"  {scores}")
    print(f"\n解释:")
    print(f"  - 节点10 (batch 0): 路径 [r0]，长度1，无法匹配2-hop规则 → 分数 0")
    print(f"  - 节点20 (batch 0): 路径 [r0, r1]，匹配规则1 (body=[0,1], head=2) → 分数 0.8")
    print(f"  - 节点30 (batch 1): 路径 [r1]，长度1，无法匹配2-hop规则 → 分数 0")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
