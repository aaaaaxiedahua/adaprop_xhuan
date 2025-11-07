"""
规则挖掘模块 (Rule Mining Module)

实现 SimplifiedAMIE 算法，从知识图谱训练数据中挖掘 Horn 子句规则。

核心思想：
    通过统计知识图谱中的路径模式，挖掘出形如 "r1(X,Y) ∧ r2(Y,Z) => r3(X,Z)" 的规则。
    这些规则描述了实体间的推理模式，例如：
        born_in(X,Y) ∧ located_in(Y,Z) => nationality(X,Z)
    表示"如果X出生在Y，Y位于Z，则X的国籍可能是Z"。

算法流程：
    1. 构建KG索引：head -> [(relation, tail), ...]
    2. 统计路径模式：遍历所有2-hop和3-hop路径
    3. 生成规则：计算置信度和支持度，过滤低质量规则
    4. 保存规则：序列化到 rules.pkl 文件

参考文献：
    AMIE+: "Fast Rule Mining in Ontological Knowledge Bases with AMIE+" (VLDB 2015)
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Set
from dataclasses import dataclass
import pickle
import os


@dataclass
class Rule:
    """
    表示一条Horn规则 (Horn Clause)

    Horn规则是一阶逻辑中的子句，形式为：B1 ∧ B2 ∧ ... ∧ Bn => H
    其中 B1, ..., Bn 是规则体（body），H 是规则头（head）

    Attributes:
        body: 规则体的关系序列，例如 [r1, r2] 表示 2-hop 路径
        head: 规则头的关系ID
        confidence: 置信度，范围 [0, 1]，表示规则的可靠性
                   计算方式：support / count(body)
        support: 支持度，表示规则在KG中被触发的次数
        length: 规则长度（hop数），即 body 的长度

    Example:
        Rule(
            body=[1, 2],    # 关系ID: r1, r2
            head=5,         # 关系ID: r5
            confidence=0.82,
            support=1523,
            length=2
        )
        表示: r1(X,Y) ∧ r2(Y,Z) => r5(X,Z)
        含义: 如果存在路径 X -[r1]-> Y -[r2]-> Z，
             则有 82% 的概率存在 X -[r5]-> Z
    """
    body: List[int]          # 规则体的关系序列 [r1, r2, ...]
    head: int                # 规则头的关系 r_head
    confidence: float        # 置信度 (confidence)
    support: int             # 支持度 (support)
    length: int              # 规则长度（hop数）

    def __hash__(self):
        """允许 Rule 作为 dict 的 key 或 set 的元素"""
        return hash((tuple(self.body), self.head))

    def __repr__(self):
        """友好的字符串表示，用于打印和调试"""
        body_str = " ∧ ".join([f"r{r}" for r in self.body])
        return f"{body_str} => r{self.head} (conf={self.confidence:.2f}, sup={self.support})"


class SimplifiedAMIE:
    """
    简化版 AMIE+ 规则挖掘算法

    仅挖掘 2-hop 和 3-hop 路径规则（平衡效果与效率）
    完整的 AMIE+ 算法可以挖掘任意长度的规则，但计算复杂度极高。
    我们的简化版本只挖掘最常见的 2-hop 和 3-hop 规则，这已经足以捕获
    大部分有价值的推理模式。

    Args:
        min_support: 最小支持度阈值（规则必须至少出现多少次才被保留）
        min_confidence: 最小置信度阈值（规则的可靠性至少要达到多少）
        max_length: 最大规则长度（hop数），推荐 2 或 3
    """

    def __init__(self, min_support=10, min_confidence=0.1, max_length=3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_length = max_length

    def mine_rules(self, train_triples: np.ndarray, n_rel: int) -> List[Rule]:
        """
        从训练三元组中挖掘规则（主入口函数）

        Args:
            train_triples: [N, 3] 数组，每行 (head, relation, tail)
            n_rel: 关系总数（用于统计）

        Returns:
            List[Rule]: 按置信度降序排列的规则列表
        """
        print("[Rule Mining] 开始挖掘规则...")
        print(f"[Rule Mining] 训练三元组数量: {len(train_triples)}")
        print(f"[Rule Mining] 关系数量: {n_rel}")
        print(f"[Rule Mining] 参数: min_support={self.min_support}, "
              f"min_confidence={self.min_confidence}, max_length={self.max_length}")

        # 步骤1: 构建知识图谱索引
        print("[Rule Mining] 构建KG索引...")
        kg_index = self._build_kg_index(train_triples)
        print(f"[Rule Mining] KG索引构建完成，节点数: {len(kg_index)}")

        # 步骤2: 统计路径模式
        print("[Rule Mining] 统计路径模式...")
        path_counts = self._count_paths(train_triples, kg_index)
        print(f"[Rule Mining] 路径模式统计完成，模式数: {len(path_counts)}")

        # 步骤3: 生成规则
        print("[Rule Mining] 生成规则...")
        rules = self._generate_rules(path_counts, train_triples)

        print(f"[Rule Mining] 完成！共挖掘到 {len(rules)} 条规则")
        return rules

    def _build_kg_index(self, triples: np.ndarray) -> dict:
        """
        构建 KG 索引：head -> [(relation, tail), ...]

        为了快速查找从某个节点出发的所有边，我们构建一个字典索引。
        这样可以在 O(1) 时间内获取节点的所有出边。

        Args:
            triples: [N, 3] 数组，每行 (head, relation, tail)
                    例如：[[0, 1, 2], [0, 3, 4]] 表示两条边：
                         0 -[r1]-> 2 和 0 -[r3]-> 4

        Returns:
            dict: {head_id: [(relation, tail), ...]}
                 例如：{0: [(1, 2), (3, 4)]} 表示节点0有两条出边

        Time Complexity: O(N) where N = len(triples)
        """
        kg = defaultdict(list)  # 使用 defaultdict 避免 KeyError
        for h, r, t in triples:
            # 将每条边 (h, r, t) 转换为 h -> (r, t) 的映射
            kg[int(h)].append((int(r), int(t)))
        return kg

    def _count_paths(self, triples: np.ndarray, kg_index: dict) -> dict:
        """
        统计所有 2-hop 和 3-hop 路径

        核心思想：
            对于每条边 h -[r1]-> t1，我们寻找从 t1 出发的路径，
            看是否能通过多跳到达某个节点 t2，同时 h 到 t2 存在直接边。
            这样就形成了一条规则：(r1, r2, ...) => r_direct

        算法示例（2-hop）：
            已知边：A -[r1]-> B, B -[r2]-> C, A -[r3]-> C
            可以推导规则：r1 ∧ r2 => r3 (置信度取决于统计频次)

        Returns:
            path_counts: {
                (r1, r2): {r3: count},      # 2-hop 规则
                (r1, r2, r3): {r4: count},  # 3-hop 规则
            }
            其中 count 表示路径 (r1, r2, ...) 和直接边 r_head 同时出现的次数

        Time Complexity:
            2-hop: O(N * avg_degree^2) where N = len(triples)
            3-hop: O(sample_size * avg_degree^3)
        """
        path_counts = defaultdict(lambda: defaultdict(int))

        # ========== 统计 2-hop 路径 ==========
        print("  统计 2-hop 路径...")
        count_2hop = 0
        for h, r1, t1 in triples:
            h, r1, t1 = int(h), int(r1), int(t1)
            if t1 not in kg_index:
                continue

            # 从 t1 出发的所有边：t1 -[r2]-> t2
            for r2, t2 in kg_index[t1]:
                # 现在我们有路径: h -[r1]-> t1 -[r2]-> t2

                # 检查是否存在直接边 h -[r3]-> t2
                # 如果存在，说明 (r1, r2) 可能推导出 r3
                if h in kg_index:
                    for r3, t3 in kg_index[h]:
                        if t3 == t2:
                            # 找到一个匹配：h -[r1]-> t1 -[r2]-> t2 且 h -[r3]-> t2
                            # 记录：路径 (r1, r2) 可能推导 r3
                            path_counts[(r1, r2)][r3] += 1
                            count_2hop += 1

        print(f"    发现 {count_2hop} 条 2-hop 路径")

        # ========== 统计 3-hop 路径 ==========
        if self.max_length >= 3:
            print("  统计 3-hop 路径...")

            # 性能优化：3-hop路径的组合数是O(N * avg_degree^3)，非常大
            # 因此我们采样一部分三元组进行统计（最多10000条）
            # 这样可以在合理时间内完成，同时仍能捕获主要的规则模式
            sample_size = min(len(triples), 10000)
            sampled_indices = np.random.choice(len(triples), sample_size, replace=False)
            sampled_triples = triples[sampled_indices]

            count_3hop = 0
            for h, r1, t1 in sampled_triples:
                h, r1, t1 = int(h), int(r1), int(t1)
                if t1 not in kg_index:
                    continue

                # 第二跳：t1 -[r2]-> t2
                for r2, t2 in kg_index[t1]:
                    if t2 not in kg_index:
                        continue

                    # 第三跳：t2 -[r3]-> t3
                    for r3, t3 in kg_index[t2]:
                        # 现在我们有路径: h -[r1]-> t1 -[r2]-> t2 -[r3]-> t3

                        # 检查是否存在直接边 h -[r4]-> t3
                        # 如果存在，说明 (r1, r2, r3) 可能推导出 r4
                        if h in kg_index:
                            for r4, t4 in kg_index[h]:
                                if t4 == t3:
                                    # 找到匹配：h -[r1]-> t1 -[r2]-> t2 -[r3]-> t3 且 h -[r4]-> t3
                                    # 记录：路径 (r1, r2, r3) 可能推导 r4
                                    path_counts[(r1, r2, r3)][r4] += 1
                                    count_3hop += 1

            print(f"    发现 {count_3hop} 条 3-hop 路径")

        return path_counts

    def _generate_rules(self, path_counts: dict, triples: np.ndarray) -> List[Rule]:
        """
        从路径统计生成规则

        将路径模式转换为 Horn 规则，计算置信度和支持度，过滤低质量规则。

        置信度计算：
            confidence(r1 ∧ r2 => r3) = support / count(r1 ∧ r2)
            表示：在所有 r1 ∧ r2 路径中，有多少比例同时存在 r3 边

        Args:
            path_counts: 路径模式统计 {(r1, r2, ...): {r_head: count}}
            triples: 训练三元组（用于统计关系频次）

        Returns:
            List[Rule]: 按置信度降序排列的规则列表
        """
        rules = []

        # 统计每个关系的出现次数（备用，当前未使用）
        relation_counts = Counter(triples[:, 1].astype(int))

        # 遍历所有路径模式
        for body, head_counts in path_counts.items():
            # body: (r1, r2, ...) 路径关系序列
            # head_counts: {r_head: count} 该路径可以推导出的关系及其出现次数

            # 计算 body 的总出现次数
            # 例如：路径 (r1, r2) 总共出现了 100 次
            body_count = sum(head_counts.values())

            # 为每个可能的 head 生成规则
            for head, support in head_counts.items():
                # support: 路径 (r1, r2, ...) 和 head 同时出现的次数

                # 过滤1: 支持度过低的规则
                # 如果规则出现次数太少，可能是偶然的，不可靠
                if support < self.min_support:
                    continue

                # 计算置信度
                # confidence = support / body_count
                # 表示：在所有 body 路径中，有多少比例同时存在 head 边
                confidence = support / body_count if body_count > 0 else 0

                # 过滤2: 置信度过低的规则
                # 如果规则的可靠性太低，不具有预测价值
                if confidence < self.min_confidence:
                    continue

                # 创建规则对象
                rule = Rule(
                    body=list(body),         # 规则体：关系序列
                    head=int(head),          # 规则头：目标关系
                    confidence=confidence,   # 置信度
                    support=support,         # 支持度
                    length=len(body)         # 规则长度（hop数）
                )
                rules.append(rule)

        # 按置信度降序排序（高置信度的规则更可靠）
        rules.sort(key=lambda r: r.confidence, reverse=True)

        return rules

    def save_rules(self, rules: List[Rule], save_path: str):
        """
        保存规则到文件

        使用 pickle 序列化，可以保留 Rule 对象的所有信息。

        Args:
            rules: 规则列表
            save_path: 保存路径（推荐使用 .pkl 后缀）
        """
        with open(save_path, 'wb') as f:
            pickle.dump(rules, f)
        print(f"[Rule Mining] 规则已保存至 {save_path}")

    def load_rules(self, load_path: str) -> List[Rule]:
        """
        从文件加载规则

        Args:
            load_path: 规则文件路径

        Returns:
            List[Rule]: 规则列表
        """
        with open(load_path, 'rb') as f:
            rules = pickle.load(f)
        print(f"[Rule Mining] 从 {load_path} 加载了 {len(rules)} 条规则")
        return rules


def mine_and_save_rules(data_path: str, min_support=10, min_confidence=0.1, max_length=3):
    """
    便捷函数：挖掘并保存规则

    这是一个高层接口，封装了从数据加载到规则保存的完整流程。

    流程：
        1. 加载 train.txt 文件
        2. 创建 SimplifiedAMIE 实例
        3. 挖掘规则
        4. 保存到 rules.pkl
        5. 打印统计信息

    Args:
        data_path: 数据集路径（包含 train.txt 文件）
        min_support: 最小支持度（推荐：小数据集5-10，大数据集20-50）
        min_confidence: 最小置信度（推荐：0.1-0.3）
        max_length: 最大规则长度（推荐：2或3）

    Returns:
        List[Rule]: 挖掘到的规则列表

    Usage:
        from rule_mining import mine_and_save_rules

        # 示例1：Family 数据集（小）
        rules = mine_and_save_rules('./data/family/', min_support=5, min_confidence=0.1)

        # 示例2：FB15k-237 数据集（大）
        rules = mine_and_save_rules('./data/fb15k-237/', min_support=20, min_confidence=0.15)
    """
    # 加载训练数据
    train_file = os.path.join(data_path, 'train.txt')
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件不存在: {train_file}")

    print(f"[Rule Mining] 加载训练数据: {train_file}")
    train_triples = []
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                train_triples.append([int(h), int(r), int(t)])

    train_triples = np.array(train_triples)
    print(f"[Rule Mining] 加载了 {len(train_triples)} 条训练三元组")

    # 获取关系数（关系ID从0开始编号，所以最大ID+1就是总数）
    n_rel = int(train_triples[:, 1].max()) + 1

    # 挖掘规则
    miner = SimplifiedAMIE(min_support=min_support, min_confidence=min_confidence, max_length=max_length)
    rules = miner.mine_rules(train_triples, n_rel)

    # 保存到文件
    save_path = os.path.join(data_path, 'rules.pkl')
    miner.save_rules(rules, save_path)

    # 打印统计信息（帮助用户了解挖掘结果）
    print("\n" + "=" * 60)
    print("规则统计：")
    print(f"  总数: {len(rules)}")
    if rules:
        # 按规则长度分组统计
        print(f"  2-hop: {sum(1 for r in rules if r.length == 2)}")
        print(f"  3-hop: {sum(1 for r in rules if r.length == 3)}")

        # 质量统计
        print(f"  平均置信度: {np.mean([r.confidence for r in rules]):.3f}")
        print(f"  平均支持度: {np.mean([r.support for r in rules]):.1f}")

        # 打印置信度最高的10条规则（供用户参考）
        print("\nTop-10 规则（按置信度）：")
        for i, rule in enumerate(rules[:10]):
            print(f"  {i+1}. {rule}")
    print("=" * 60)

    return rules


# ==================== 测试代码 ====================
# 可以直接运行此脚本来测试规则挖掘功能
if __name__ == "__main__":
    import sys

    # 支持命令行参数指定数据集路径
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # 默认：挖掘 Family 数据集的规则
        data_path = './data/family/'

    print(f"数据集路径: {data_path}")

    # 挖掘规则
    # 对于 Family 数据集（小），使用较低的阈值
    rules = mine_and_save_rules(
        data_path=data_path,
        min_support=5,           # 支持度至少5次
        min_confidence=0.1,      # 置信度至少10%
        max_length=3             # 最多3-hop规则
    )
