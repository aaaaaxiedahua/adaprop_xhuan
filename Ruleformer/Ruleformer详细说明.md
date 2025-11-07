# Ruleformer 详细说明文档

## 概述

**Ruleformer** 是一个基于Transformer的知识图谱规则挖掘模型，来自论文 "Ruleformer: Context-aware Rule Mining over Knowledge Graph" (COLING 2022)。

**核心创新**：使用**上下文感知的Transformer**来挖掘知识图谱中的逻辑规则（Horn子句），通过在子图上进行序列到序列学习，捕捉关系路径的语义信息。

**与传统规则挖掘的区别**：
- 传统方法（如AMIE）：基于统计的规则挖掘，仅考虑路径的频率和置信度
- Ruleformer：使用神经网络学习关系的语义表示，通过注意力机制捕捉上下文信息

---

## 项目结构

```
Ruleformer/
├── translate.py              # 主训练/解码脚本
├── transformer/
│   ├── dataset.py           # 数据预处理：子图提取
│   ├── Models.py            # Transformer模型定义
│   ├── Translator.py        # 推理引擎：训练+规则解码
│   ├── Layers.py            # Encoder/Decoder层
│   ├── SubLayers.py         # 多头注意力+前馈网络
│   ├── Modules.py           # 缩放点积注意力
│   └── Optim.py             # 学习率调度器
└── README.md
```

---

## 核心概念

### 1. 规则挖掘任务定义

**目标**：从知识图谱中挖掘形如 `r1(X,Y) ∧ r2(Y,Z) => r3(X,Z)` 的Horn子句规则。

**输入**：三元组 `(h, r, t)` - 头实体、关系、尾实体
**输出**：关系路径序列 `[r1, r2, ..., rk]`，表示从 `h` 到 `t` 的逻辑推理路径

**示例**：
```
规则：wasBornIn(X,Y) ∧ locatedIn(Y,Z) => nationality(X,Z)
实例：wasBornIn(Einstein, Ulm) ∧ locatedIn(Ulm, Germany) => nationality(Einstein, Germany)
```

### 2. 子图提取（Subgraph Extraction）

**核心思想**：为每个实体提取其局部子图作为上下文，用于Transformer编码。

**提取过程**：
1. **起点**：从头实体 `h` 开始
2. **多跳扩展**：
   - 第1跳：获取 `h` 的所有1跳邻居及边关系
   - 第2跳：从第1跳的所有节点继续扩展
   - 第k跳：重复直到达到指定跳数 `JUMP`
3. **限制节点度数**：
   - 若某节点的某种关系的邻居数 > `MAXN`，则随机采样 `MAXN` 个
   - 防止高度数节点导致子图爆炸
4. **截断超长子图**：
   - 若节点总数 > `PADDING`，则截断至 `PADDING` 个节点
5. **构建关系矩阵**：
   - 生成 `PADDING × PADDING × n_rel` 的稀疏矩阵
   - `link[i,j,r] = 1` 表示节点i与节点j之间存在关系r

**关键参数**：
- `JUMP`：子图扩展跳数（2或3）
- `MAXN`：节点最大邻居数（10-70，取决于数据集）
- `PADDING`：子图最大节点数（140）

---

## 三阶段工作流程

### Step 1: 数据预处理 - 子图提取

#### 1.1 知识图谱加载
```python
# 加载数据（以Family为例）
database = DataBase(opt)
# entities.txt: ["father", "mother", "child", ...] → ID映射
# relations.txt: ["hasFather", "hasMother", ...] → ID映射
# train.txt: [(1,0,2), (2,1,3), ...] # (头,关系,尾)
```

#### 1.2 子图提取
```python
# 为每个三元组提取局部子图
def extract_without_token(head, JUMP=2, MAXN=50, PADDING=140):
    subgraph = [head]           # 起始节点

    for jump in range(JUMP):
        # 对当前跳数的所有节点进行邻居扩展
        for parent in current_jump_nodes:
            for relation, neighbors in database.neighbors[parent].items():
                # 限制邻居数量，防止子图过大
                sampled_neighbors = neighbors[:MAXN]

                for neighbor in sampled_neighbors:
                    # 添加新节点和边关系
                    add_node_and_edge(parent, neighbor, relation)
```

#### 1.3 数据格式
```python
# 每个样本包含前向和后向两个方向
subgraph_h, rela_mat_h, target_h, subgraph_t, target_t

# subgraph_h: [140] - 前向子图节点序列
# rela_mat_h: [140,140,41] - 前向关系矩阵
# target_h: [3] - 目标三元组 [H,R,T]
# subgraph_t: [140] - 后向子图节点序列（逆向推理）
# target_t: [3] - 后向目标三元组 [T,R_inv,H]
```

### Step 2: 模型训练 - Transformer编码

#### 2.1 Encoder编码子图 - 详细过程

**输入数据**：
```python
src_seq = [0, 1, 2, 4, 0, 0, 0, 0, ..., 0]  # [140]
# 含义：[<PAD>, Einstein, Ulm, Berlin, <PAD>, <PAD>, ...]

link = torch.zeros(140, 140, 41)  # [140, 140, 41]
# 具体的关系连接：
link[0, 1, 1] = 1    # Einstein -> bornIn -> Ulm
link[1, 2, 2] = 1    # Ulm -> locatedIn -> Germany
link[2, 4, 3] = 1    # Germany -> hasCapital -> Berlin

length = [0, 1, 3, 4]  # [4]
# 含义：第0跳1个节点，第1跳新增2个节点，第2跳新增1个节点
```

**编码过程**：

##### Step 2.1.1：节点嵌入和关系感知聚合
```python
# 1. 基础节点嵌入
entity_emb = self.src_word_emb(src_seq)
# 输入：src_seq.shape = [140] - 140个节点ID
# 输出：entity_emb.shape = [140, 128] - 140个节点，每个128维向量
# 不是"140维变成128维"，而是"140个节点ID → 140个128维向量"

# 详细过程：
entity_emb[0, :] = embedding_layer(0)   # 节点ID 0 -> 128维向量
entity_emb[1, :] = embedding_layer(1)   # 节点ID 1 -> 128维向量
entity_emb[2, :] = embedding_layer(2)   # 节点ID 2 -> 128维向量
...
entity_emb[139, :] = embedding_layer(139) # 节点ID 139 -> 128维向量

# 具体例子：
entity_emb[0, :] = [0.12, -0.34, 0.56, ..., 0.78]  # Einstein的128维向量
entity_emb[1, :] = [-0.23, 0.45, -0.67, ..., 0.12] # Ulm的128维向量
entity_emb[2, :] = [0.34, -0.12, 0.78, ..., -0.45] # Germany的128维向量
entity_emb[3, :] = [-0.56, 0.78, 0.23, ..., 0.34]  # Berlin的128维向量

# 嵌入层本质上是一个查找表：
self.src_word_emb = nn.Embedding(num_entities, 128)
# num_entities: 知识图谱中实体总数 + 1 (padding)
# 128: 每个实体的嵌入维度

# 2. 关系感知聚合
neighbor_rel = self.nebor_relation[src_seq]  # [140, 41]
neighbor_emb = torch.matmul(neighbor_rel, self.relation_type_E)  # [140, 128]

# 详细解释关系感知聚合过程：
# ========================================
# 步骤2.1: 获取每个节点的邻居关系分布 [140, 41]
# ========================================
# self.nebor_relation: [num_entities, 41] - 从训练数据统计得到的邻居关系分布矩阵

# 🎯 数据来源：训练数据统计计算
# =============================
# 根据 dataset.py 第56-64行的代码实现：

# 1. 初始化统计矩阵
self.nebor_relation = torch.ones(len(self.e2id), len(self.r2id))
# 形状：[实体总数, 关系总数] - 初始值设为1（拉普拉斯平滑）

# 2. 遍历训练三元组进行统计
for h, r, t in self.data['train']:  # 遍历所有训练三元组 (头实体, 关系, 尾实体)
    # 头实体h的关系r计数+1
    self.nebor_relation[h][r] += 1

    # 尾实体t的逆关系计数+1 (r + pos_rels)
    self.nebor_relation[t][r + self.pos_rels] += 1

# 3. 为没有邻居的实体添加自环关系
for e in self.e2id.values():
    if e not in self.neighbors.keys():
        self.nebor_relation[e][2 * self.pos_rels] += 1  # 自环关系ID

# 4. 转换为概率分布
self.nebor_relation = torch.log(self.nebor_relation)  # 取对数平滑
self.nebor_relation /= self.nebor_relation.sum(1).unsqueeze(1)  # 行归一化

# 🎯 具体计算示例：
# ==================
# 假设训练数据：
train_triples = [
    (1, 0, 2),  # Einstein -> wasBornIn -> Ulm
    (1, 0, 2),  # Einstein -> wasBornIn -> Ulm (重复出现)
    (1, 1, 3),  # Einstein -> locatedIn -> USA
    (2, 1, 3),  # Ulm -> locatedIn -> USA
    (2, 2, 4),  # Ulm -> hasCapital -> Berlin
]

# 关系配置：
# pos_rels = 20  # 正向关系数量
# 关系映射：0=wasBornIn, 1=locatedIn, 2=hasCapital, ..., 20=wasBornIn_inv, 21=locatedIn_inv, ...

# 统计过程：
# 初始化：torch.ones(5, 41)  # 5个实体，41种关系
# 统计后（计数）：
#   实体1(Einstein): [1, 2, 1, 1, ..., 1]  # wasBornIn出现2次+初始1=3次
#   实体2(Ulm):      [1, 2, 1, 1, ..., 1]  # locatedIn出现2次+初始1=3次
#   实体3(USA):      [1, 1, 1, 2, ..., 1]  # locatedIn_inv出现1次+初始1=2次

# 归一化后的概率分布：
nebor_relation = torch.tensor([
    [0.024, 0.024, 0.024, 0.024, ..., 0.024],  # 实体0(padding): 均匀分布
    [0.073, 0.146, 0.024, 0.024, ..., 0.024],  # 实体1(Einstein): 14.6% wasBornIn
    [0.024, 0.146, 0.024, 0.024, ..., 0.024],  # 实体2(Ulm): 14.6% locatedIn
    [0.024, 0.024, 0.024, 0.146, ..., 0.024],  # 实体3(USA): 14.6% locatedIn_inv
    [0.024, 0.024, 0.024, 0.024, ..., 0.024],  # 其他实体: 均匀分布
])  # shape: [num_entities, 41]

# 示例：self.nebor_relation[1] = [0.073, 0.146, 0.024, 0.024, ..., 0.024]
# 含义：Einstein的邻居关系中，14.6%是wasBornIn，2.4%是其他关系

# src_seq = [1, 2, 4, 7, 0, 0, 0, ..., 0]  # [140] - Einstein子图的节点序列
# neighbor_rel = self.nebor_relation[src_seq] 的结果：
neighbor_rel = torch.tensor([
    [0.30, 0.20, 0.10, 0.05, 0.02, ..., 0.01],  # 第0行：Einstein的关系分布
    [0.15, 0.40, 0.25, 0.08, 0.03, ..., 0.02],  # 第1行：Ulm的关系分布
    [0.05, 0.10, 0.60, 0.15, 0.04, ..., 0.01],  # 第2行：Germany的关系分布
    [0.20, 0.05, 0.15, 0.50, 0.03, ..., 0.02],  # 第3行：Berlin的关系分布
    [0.00, 0.00, 0.00, 0.00, 0.00, ..., 0.00],  # 第4行：padding节点(全零)
    [0.00, 0.00, 0.00, 0.00, 0.00, ..., 0.00],  # 第5行：padding节点(全零)
    ...                                            # ��共140行，大部分是padding节点
])  # shape: [140, 41]

# ========================================
# 步骤2.2: 关系嵌入聚合 [140, 41] × [41, 128] → [140, 128]
# ========================================
# self.relation_type_E: [41, 128] - 可学习的关​​系嵌入矩阵
#   - 行：41种关系类型（20个正向+20个逆向+1个自环）
#   - 列：每种关系的128维向量表示
#   - 🎯 学习过程：这是一个nn.Embedding层，通过训练学习得到

# 🎯 关系嵌入的学习过程：
# ==========================
# 在 Models.py 中，关系嵌入是可学习的参数：
self.relation_type_E = nn.Embedding(num_relations, d_word_vec)
# num_relations = len(relation_ids)  # 41种关系
# d_word_vec = 128  # 每种关系用128维向量表示

# 🎯 关系向量的学习目标：
# ======================
# 通过训练让关系向量捕捉语义信息：
# - 相似的关系向量相似（如 wasBornIn 和 wasBornIn_inv）
# - 相关的关系向量距离近（如 wasBornIn 和 nationality）
# - 无关的关系向量距离远（如 wasBornIn 和 hasCapital）

# 🎯 关系嵌入矩阵示例：
# ====================
self.relation_type_E = nn.Embedding(41, 128)  # 41个关系，每个128维

# 训练后学到的关系向量（示例）：
relation_vectors = {
    0: [0.8, -0.2, 0.5, 0.3, ..., 0.1],      # wasBornIn
    1: [0.3, 0.7, -0.4, 0.6, ..., 0.4],      # locatedIn
    2: [0.5, 0.2, 0.8, -0.1, ..., 0.6],      # hasCapital
    3: [0.6, 0.1, -0.3, 0.9, ..., 0.2],      # nationality
    20: [-0.8, 0.2, -0.5, -0.3, ..., -0.1],   # wasBornIn_inv (与wasBornIn相反)
    21: [-0.3, -0.7, 0.4, -0.6, ..., -0.4],   # locatedIn_inv (与locatedIn相反)
    40: [0.1, 0.1, 0.1, 0.1, ..., 0.1],       # self_loop (特殊关系)
    # ... 其他关系向量
}  # shape: [41, 128]

# 🎯 向量相似性示例：
# ===================
# cosine_similarity(wasBornIn, locatedIn) = 0.23    # 低相似度
# cosine_similarity(wasBornIn, nationality) = 0.67  # 中等相似度
# cosine_similarity(wasBornIn, wasBornIn_inv) = -0.95 # 高负相似度
# cosine_similarity(locatedIn, locatedIn_inv) = -0.92 # 高负相似度

# 矩阵乘法过程：
neighbor_emb = torch.matmul(neighbor_rel, self.relation_type_E)
# [140, 41] × [41, 128] = [140, 128]

# 具体计算示例（以Einstein为例）：
# neighbor_rel[0] = [0.30, 0.20, 0.10, 0.05, ..., 0.01]  # Einstein的关系分布
# neighbor_emb[0] = 0.30 * wasBornIn_vector + 0.20 * locatedIn_vector + 0.10 * hasCapital_vector + ...
#                = 0.30*[0.8, -0.2, 0.5, ..., 0.3]    # wasBornIn向量
#                + 0.20*[0.3, 0.7, -0.4, ..., 0.6]    # locatedIn向量
#                + 0.10*[0.5, 0.2, 0.8, ..., 0.4]    # hasCapital向量
#                + ...                                 # 其他关系向量
#                = [0.45, 0.23, 0.67, ..., 0.34]    # Einstein的邻居关系聚合向量

# neighbor_emb的完整结果：
neighbor_emb = torch.tensor([
    [0.45, 0.23, 0.67, 0.12, ..., 0.34],  # Einstein的邻居关系聚合向量
    [0.28, 0.56, 0.34, 0.78, ..., 0.45],  # Ulm的邻居关系聚合向量
    [0.67, 0.12, 0.89, 0.23, ..., 0.56],  # Germany的邻居关系聚合向量
    [0.34, 0.78, 0.45, 0.89, ..., 0.12],  # Berlin的邻居关系聚合向量
    [0.00, 0.00, 0.00, 0.00, ..., 0.00],  # padding节点向量（全零）
    [0.00, 0.00, 0.00, 0.00, ..., 0.00],  # padding节点向量（全零）
    ...                                     # 总共140行
])  # shape: [140, 128]

# ========================================
# 步骤2.3: 融合自身和邻居信息 [140, 128]
# ========================================
# entity_emb: [140, 128] - 节点自身的嵌入向量
# neighbor_emb: [140, 128] - 节点邻居关系的聚合向量
# node_init: [140, 128] - 融合后的初始节点表示

# 融合公式：平均融合
node_init = (entity_emb + neighbor_emb) / 2

# 具体计算示例（以Einstein为例）：
# entity_emb[0] = [0.12, -0.34, 0.56, 0.78, ..., 0.45]    # Einstein自身向量
# neighbor_emb[0] = [0.45, 0.23, 0.67, 0.12, ..., 0.34]    # Einstein邻居聚合向量
# node_init[0] = ([0.12, -0.34, 0.56] + [0.45, 0.23, 0.67]) / 2
#              = [0.285, -0.055, 0.615, 0.45, ..., 0.395]   # Einstein的融合向量

# node_init的完整结果：
node_init = torch.tensor([
    [0.285, -0.055, 0.615, 0.45, ..., 0.395],  # Einstein：自身+邻居信息
    [0.175, 0.345, 0.56, 0.45, ..., 0.285],     # Ulm：自身+邻居信息
    [0.505, 0.05, 0.835, 0.455, ..., 0.56],     # Germany：自身+邻居信息
    [0.11, 0.78, 0.34, 0.835, ..., 0.23],       # Berlin：自身+邻居信息
    [0.00, 0.00, 0.00, 0.00, ..., 0.00],        # padding节点（全零）
    [0.00, 0.00, 0.00, 0.00, ..., 0.00],        # padding节点（全零）
    ...                                            # 总共140行
])  # shape: [140, 128]

# ========================================
# 关键优势解释
# ========================================
# 1. 上下文感知：每个节点不仅知道自身信息，还知道邻居关系特征
# 2. 关系语义：通过关系向量，模型理解不同关系的语义差异
# 3. 信息融合：平衡了节点自身特征和图结构特征
# 4. 计算高效：通过预计算的关系分布，避免复杂的邻居遍历

# 3. 融合自身和邻居信息
node_init = (entity_emb + neighbor_emb) / 2  # [140, 128]
```

##### Step 2.1.2：跳数级位置编码
```python
# Ruleformer创新：相同跳数的节点共享位置编码
node_init = self.dropout(self.position_enc(node_init, length))

# length = [0, 1, 3, 4]
node_init[0:1] += pos_enc[0, :]    # 第0跳：Einstein
node_init[1:3] += pos_enc[1, :]    # 第1跳：Ulm, Germany
node_init[3:4] += pos_enc[2, :]    # 第2跳：Berlin
```

##### Step 2.1.3：关系感知多头注意力
```python
# 核心创新：考虑节点间的具体关系类型
for enc_layer in self.layer_stack:  # 6层Encoder
    node_init, enc_slf_attn = enc_layer(node_init, link=link)

# 关系感知注意力计算：
# ========================================
# 🎯 Ruleformer核心创新：关系感知注意力机制
# ========================================
# 传统Transformer注意力：
#   Attention(Q,K,V) = softmax(QK^T/√d) × V

# Ruleformer关系感知注意力：
#   Attention(Q,K,V,R) = softmax((QK^T/√d) + R) × V
#   其��：R = 关系权重矩阵

# 🎯 数学公式推导：
# =================
# 1. 传统注意力分数（节点相似性）：
#    S_base(i,j) = (Q_i · K_j) / √d_k
#    - Q_i: 查询节点i的表示
#    - K_j: 键节点j的表示
#    - d_k: 键向量维度

# 2. 关系感知权重：
#    S_rel(i,j) = f(R_ij, h_j)
#    - R_ij: 节点i到节点j的关系向量
#    - h_j: 节点j的隐藏表示
#    - f: 关系权重函数

# 3. Ruleformer注意力分数：
#    S_total(i,j) = S_base(i,j) + α × S_rel(i,j)
#    - α: 关系权重超参数，控制关系信息的影响程度

# 4. 最终注意力权重：
#    A(i,j) = softmax_j(S_total(i,j))
#    - 对每行的所有列进行softmax归一化

# 5. 输出计算：
#    Output_i = Σ_j A(i,j) × V_j
#    - V_j: 值节点j的表示

# 🎯 具体实现（基于SubLayers.py代码）：
# =====================================
def calcE_attention(input, w_qs, w_ks, w_rs, link):
    """
    Ruleformer关系感知注意力的核心实现

    Args:
        input: [batch_size, seq_len, d_model] - 输入节点表示
        w_qs: [d_model, n_head × d_k] - Query权重矩阵
        w_ks: [d_model, n_head × d_k] - Key权重矩阵
        w_rs: [d_model, n_head × d_k] - 关系权重矩阵
        link: [batch_size, seq_len, seq_len, n_rel] - 关系矩阵

    Returns:
        attention_scores: [batch_size, seq_len, seq_len] - 注意力分数
    """

    # 1. 计算基础注意力分数
    Q = torch.matmul(input, w_qs)  # [batch, seq_len, n_head × d_k]
    K = torch.matmul(input, w_ks)  # [batch, seq_len, n_head × d_k]

    # 重塑为多头形式
    Q = Q.reshape(batch_size, seq_len, n_head, d_k)
    K = K.reshape(batch_size, seq_len, n_head, d_k)

    # 基础注意力分数 (缩放点积注意力)
    base_scores = torch.einsum('bihd,bjhd->bhij', Q, K) / math.sqrt(d_k)
    # shape: [batch_size, n_head, seq_len, seq_len]

    # 2. 计算关系感知权重
    rel_scores = torch.zeros_like(base_scores)  # 初始化为零

    # 遍历每对节点(i,j)
    for i in range(seq_len):
        for j in range(seq_len):
            # 获取节点i到节点j的关系向量
            relation_vector = link[:, i, j, :]  # [batch_size, n_rel]

            # 如果存在关系连接
            if torch.any(relation_vector > 0):
                # 找到主要关系类型
                rel_id = torch.argmax(relation_vector, dim=-1)  # [batch_size]

                # 获取关系权重向量
                for head in range(n_head):
                    for b in range(batch_size):
                        r_id = rel_id[b].item()
                        # w_rs[r_id*head*d_k : (r_id+1)*head*d_k] 对应关系r_id在第head头的权重
                        rel_weight = torch.matmul(
                            input[b, j, :],
                            w_rs[:, r_id*head*d_k : (r_id+1)*head*d_k]
                        )
                        rel_scores[b, head, i, j] = rel_weight

    # 3. 融合基础分数和关系权重
    total_scores = base_scores + rel_scores  # [batch, n_head, seq_len, seq_len]

    # 4. 应用mask（忽略padding节点）
    if mask is not None:
        total_scores = total_scores.masked_fill(mask == 0, -1e9)

    # 5. 计算最终注意力权重
    attention_weights = F.softmax(total_scores, dim=-1)

    return attention_weights

# 🎯 数值计算示例：
# ==================
# 假设计算Einstein对Ulm的注意力：

# 🎯 数据来源说明：
# ==================
# 以下所有数值都是**假设的示例数据**，用于演示计算过程
# 实际数值来自训练过程中的神经网络参数，每次训练都会变化

# 1. 基础注意力分数计算：
# ==================
# 输入：经过第一层Encoder后的节点表示 (来自Step 2.1.3的输出)
node_init = torch.tensor([
    [0.285, -0.055, 0.615, 0.45, 0.892, 0.123, 0.456, 0.789, 0.234, 0.567, ...],  # Einstein
    [0.175, 0.345, 0.56, 0.45, 0.678, 0.234, 0.567, 0.890, 0.123, 0.456, ...],     # Ulm
    [0.505, 0.05, 0.835, 0.455, 0.123, 0.456, 0.789, 0.234, 0.567, 0.890, ...],     # Germany
    [0.11, 0.78, 0.34, 0.835, 0.456, 0.789, 0.234, 0.567, 0.890, 0.123, ...],       # Berlin
    ...  # 其他padding节点
])  # shape: [140, 128]

# 🎯 注意力权重矩阵（可学习参数，随机初始化后通过训练学习）：
# ==============================================
# 在第一层EncoderLayer中初始化（参考SubLayers.py）：
self.w_qs = nn.Linear(128, 512, bias=False)   # Query权重矩阵 [128, 512]
self.w_ks = nn.Linear(128, 512, bias=False)   # Key权重矩阵 [128, 512]
self.w_vs = nn.Linear(128, 512, bias=False)   # Value权重矩阵 [128, 512]
self.w_rs = nn.Linear(128, 512, bias=False)   # 关系权重矩阵 [128, 512]

# 训练后学到的权重矩阵（假设值）：
w_qs = torch.tensor([
    [0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3, 0.6, -0.4, ...],  # 第1行权重
    [-0.3, 0.4, 0.2, -0.1, 0.5, 0.3, -0.2, 0.1, 0.4, -0.5, ...],  # 第2行权重
    [0.2, -0.1, 0.5, 0.3, -0.4, 0.6, 0.1, -0.2, 0.3, 0.4, ...],  # 第3行权重
    ...  # 共128行×512列
])  # shape: [128, 512]

w_ks = torch.tensor([
    [-0.1, 0.3, 0.2, 0.5, -0.2, 0.4, 0.1, -0.3, 0.6, 0.2, ...],  # 第1行权重
    [0.4, -0.2, 0.3, 0.1, 0.5, -0.1, 0.2, 0.4, -0.3, 0.6, ...],  # 第2行权重
    [0.2, 0.5, -0.1, 0.3, 0.4, -0.2, 0.6, 0.1, -0.4, 0.3, ...],  # 第3行权重
    ...  # 共128行×512列
])  # shape: [128, 512]

# 1.1 计算Query向量：
# ===================
Q_Einstein_raw = torch.matmul(node_init[0], w_qs)  # [128] × [128, 512] = [512]
# 具体计算：Q_Einstein_raw[0] = 0.285*0.1 + (-0.055)*(-0.3) + 0.615*0.2 + ... + 0.567*(-0.4)
# 假设计算结果：Q_Einstein_raw = [0.45, 0.23, 0.67, 0.12, 0.89, 0.34, 0.56, 0.78, 0.23, 0.45, ...]  # [512]

# 1.2 计算Key向量：
# =================
K_Ulm_raw = torch.matmul(node_init[1], w_ks)  # [128] × [128, 512] = [512]
# 具体计算：K_Ulm_raw[0] = 0.175*(-0.1) + 0.345*0.4 + 0.56*0.2 + ... + 0.456*0.6
# 假设计算结果：K_Ulm_raw = [0.28, 0.56, 0.34, 0.78, 0.12, 0.45, 0.67, 0.89, 0.34, 0.56, ...]  # [512]

# 1.3 重塑为多头注意力形式：
# ==========================
# 假设8个注意力头，每个头64维（512 = 8×64）
Q_Einstein = Q_Einstein_raw.reshape(8, 64)  # [8, 64]
K_Ulm = K_Ulm_raw.reshape(8, 64)           # [8, 64]

# 1.4 计算基础注意力分数：
# ===================
# 使用第一个注意力头进行演示：
Q_Einstein_head1 = [0.45, 0.23, 0.67, 0.12, 0.89, 0.34, 0.56, 0.78, 0.23, 0.45, ...]  # [64]
K_Ulm_head1 = [0.28, 0.56, 0.34, 0.78, 0.12, 0.45, 0.67, 0.89, 0.34, 0.56, ...]      # [64]

# 点积计算：
dot_product = torch.dot(Q_Einstein_head1, K_Ulm_head1)
# 具体计算：0.45*0.28 + 0.23*0.56 + 0.67*0.34 + 0.12*0.78 + ... (64个乘积相加)
# 假设计算结果：dot_product = 4.67

# 缩放因子：√d_k = √64 = 8
# 基础注意力分数：S_base = 4.67 / 8 = 0.58375 ≈ 0.73（示例中简化为0.73）

# 🎯 关键说明：
# =============
# 1. Q_Einstein 和 K_Ulm 的值**不是固定**的，它们来自于：
#    - 节点初始表示（entity_emb + neighbor_emb）/ 2
#    - 通过可学习的权重矩阵 w_qs 和 w_ks 线性变换
#    - 每次训练更新，w_qs 和 w_ks 都会变化
#
# 2. 完整的计算过程：
#    输入节点表示 → 线性变换 → Query/Key向量 → 点积 → 缩放 → 注意力分数
#
# 3. 这些数值的实际含义：
#    - 高数值表示两个节点在当前特征空间中相似度高
#    - 通过训练，模型学会让相关实体的Query和Key向量产生高分
#    - Einstein和Ulm因为语义相关，学到的权重让它们的向量相似

# 2. 关系感知权重计算：
# ==================
# 2.1 关系矩阵（来自数据预处理）：
R_Einstein_Ulm = torch.zeros(41)  # 41种关系
R_Einstein_Ulm[0] = 1  # wasBornIn关系（假设关系ID=0）
# R_Einstein_Ulm = [1, 0, 0, 0, 0, 0, ..., 0]  # one-hot向量

# 2.2 关系权重矩阵（训练学习）：
w_rs = torch.tensor([
    [0.8, -0.2, 0.5, 0.3, -0.1, 0.6, 0.2, -0.4, 0.7, 0.1, ...],  # wasBornIn关系权重
    [-0.8, 0.2, -0.5, -0.3, 0.1, -0.6, -0.2, 0.4, -0.7, -0.1, ...],  # wasBornIn_inv权重
    [0.3, 0.7, -0.4, 0.6, 0.2, -0.1, 0.5, -0.3, 0.4, 0.8, ...],  # locatedIn关系权重
    ...  # 共41行×512列，每行对应一种关系
])  # shape: [41, 512]

# 2.3 关系权重计算：
# 获取wasBornIn关系的权重向量：
w_rs_wasBornIn = w_rs[0]  # [512]

# 与Ulm的隐藏表示相乘：
S_rel = torch.dot(node_init[1], w_rs_wasBornIn[:64])  # 使用前64维
# 具体计算：0.175*0.8 + 0.345*(-0.2) + 0.56*0.5 + 0.45*0.3 + ... (64个乘积相加)
# 假设计算结果：S_rel = 0.45

# 3. Ruleformer注意力分数：
# ==================
S_total = S_base + S_rel = 0.73 + 0.45 = 1.18

# 4. 与其他节点比较：
# ==================
# 同理计算Einstein对其他节点的注意力分数：
# Einstein → Germany: S_base = 0.58, S_rel = 0.34, S_total = 0.92
# Einstein → Berlin:   S_base = 0.28, S_rel = 0.06, S_total = 0.34
# Einstein → Padding:  S_base = 0.01, S_rel = 0.00, S_total = 0.01 (实际设为-1e9)

# 5. Softmax归一化：
# ==================
attention_scores = [1.18, 0.92, 0.34, -1e9]  # Einstein对[Ulm, Germany, Berlin, Padding]的分数
attention_weights = F.softmax(torch.tensor(attention_scores), dim=0)
# 结果：[0.52, 0.32, 0.06, 0.00]  # Einstein 52%关注Ulm，32%关注Germany，6%关注Berlin

# 🎯 数据来源总结：
# ================
# 1. **节点表示**：来自 Step 2.1.3 的融合向量 (entity_emb + neighbor_emb) / 2
# 2. **权重矩阵**：w_qs, w_ks, w_rs 是神经网络的可学习参数，通过训练优化
# 3. **关系矩阵**：来自数据预处理的关系连接信息
# 4. **计算过程**：每层Encoder都会重新计算，数值会随着训练变化

# 🎯 关系感知注意力的优势：
# =====================
# 1. **语义相关性**：相同关系的节点间注意力更强
#    - Einstein → Ulm (wasBornIn) 注意力增强
#    - Ulm → Germany (locatedIn) 注意力增强

# 2. **推理路径**：能够沿着关系路径传播信息
#    - wasBornIn → locatedIn → nationality 形成推理链

# 3. **可解释性**：注意力权重反映了实体间的逻辑关系
#    - 高注意力权重的边通常对应重要的推理步骤

# 4. **上下文感知**：相同的实体在不同关系下有不同的注意力模式
#    - Einstein���为"被出生者" vs "物理学家" 有不同的注意力分布
```

**输出数据**：
```python
enc_output = torch.zeros(1, 140, 128)
enc_output[0, 0, :] = [0.45, -0.23, 0.78, ..., 0.56]  # Einstein的上下文表示
enc_output[0, 1, :] = [0.12, 0.67, -0.34, ..., 0.23]  # Ulm的上下文表示
# 每个节点向量都包含整个子图的语义信息
```

#### 2.2 Decoder生成关系注意力 - 详细过程

**核心任务**：根据查询关系和Encoder输出的子图信息，生成推理路径的关系序列

**输入**：
```python
query_rel = 4  # "nationality" 关系的ID
enc_output = torch.zeros(1, 140, 128)  # Encoder输出的子图表示
```

**目标输出**：
```python
# 关系序列，表示推理路径
# 例如：[wasBornIn, locatedIn, nationality]
# 解释：Einstein -> wasBornIn -> Ulm -> locatedIn -> Germany -> nationality
```

---

##### Step 2.2.1：初始化 - ��备查询关系

```python
# 查询关系的嵌入
query_embedding = self.relationE(torch.tensor([query_rel]))
# shape: [1, 128] - "nationality"的128维向量表示

# Decoder的初始输入
dec_input = query_embedding  # [1, 128]
# 这是自回归生成的起始信号，告诉Decoder："我们要推理nationality关系"

# 具体例子：
# query_embedding = [[0.23, -0.45, 0.67, 0.12, ..., 0.78]]  # nationality的向量
```

##### Step 2.2.2：第一步解码 - 生成第一个推理关系

```python
# Decoder第一步：根据查询关系，预测第一个推理步骤
dec_output_1 = self.decoder(
    trg_seq=dec_input,        # [1, 128] - 查询关系nationality
    enc_output=enc_output,    # [1, 140, 128] - Einstein子图的上下文
    src_mask=mask,           # [1, 140] - 忽略padding节点
    ...
)

# dec_output_1的shape: [1, 128]
# 这是Decoder对"第一步应该是什么关系"的内部表示

# 投影到关系空间
logits_1 = self.trg_word_prj(dec_output_1)  # [1, 41]
# 41是关系总数，logits表示每个关系可能的得分

# 转换为概率分布
probs_1 = F.softmax(logits_1, dim=-1)  # [1, 41]

# 示例输出：
probs_1 = torch.tensor([[0.45, 0.30, 0.15, 0.05, 0.02, 0.01, 0.01, 0.01, ...]])
# 解释：
# - 关系0 ("wasBornIn"): 45% 概率
# - 关系1 ("locatedIn"): 30% 概率
# - 关系2 ("hasCapital"): 15% 概率
# - 关系3 ("graduatedFrom"): 5% 概率
# - 其他关系: 剩余概率

# 选择最可能的关系（训练时用真实标签，推理时用top选择）
first_rel = torch.argmax(probs_1)  # 选中了关系0 ("wasBornIn")
first_rel_name = "wasBornIn"
```

##### Step 2.2.3：第二步解码 - 基于第一步生成第二步

```python
# 构建第二步的输入：[查询关系, 第一个关系]
first_rel_embedding = self.relationE(torch.tensor([first_rel]))  # [1, 128]
dec_input_2 = torch.cat([query_embedding, first_rel_embedding], dim=0)  # [2, 128]
# 含义：Decoder知道"要推理nationality，且第一步选择了wasBornIn"

# 第二步解码：已知要推理nationality，且第一步是wasBornIn，预测第二步
dec_output_2 = self.decoder(
    trg_seq=dec_input_2,       # [2, 128] - [nationality, wasBornIn]
    enc_output=enc_output,     # [1, 140, 128] - 子图上下文
    src_mask=mask,
    ...
)

# 只使用最后一步的输出（最新生成的步骤）
last_step_output = dec_output_2[-1:, :]  # [1, 128]

# 投影到关系空间
logits_2 = self.trg_word_prj(last_step_output)  # [1, 41]
probs_2 = F.softmax(logits_2, dim=-1)  # [1, 41]

# 示例输出：
probs_2 = torch.tensor([[0.05, 0.65, 0.20, 0.05, 0.03, 0.01, 0.01, ...]])
# 解释：
# - 关系0 ("wasBornIn"): 5% 概率（重复，不太可能）
# - 关系1 ("locatedIn"): 65% 概率（最可能，Ulm -> Germany）
# - 关系2 ("hasCapital"): 20% 概率（可能但不太合理）
# - 关系3 ("graduatedFrom"): 5% 概率
# - 其他关系: 剩余概率

# 选择最可能的关系
second_rel = torch.argmax(probs_2)  # 选中了关系1 ("locatedIn")
second_rel_name = "locatedIn"
```

##### Step 2.2.4：第三步解码 - 生成最终关系

```python
# 构建第三步的输入：[查询关系, 第一个关系, 第二个关系]
second_rel_embedding = self.relationE(torch.tensor([second_rel]))  # [1, 128]
dec_input_3 = torch.cat([query_embedding, first_rel_embedding, second_rel_embedding], dim=0)  # [3, 128]
# 含义：Decoder知道"要推理nationality，步骤是[nationality, wasBornIn, locatedIn]"

# 第三步解码
dec_output_3 = self.decoder(
    trg_seq=dec_input_3,       # [3, 128] - [nationality, wasBornIn, locatedIn]
    enc_output=enc_output,     # [1, 140, 128] - 子图上下文
    src_mask=mask,
    ...
)

last_step_output = dec_output_3[-1:, :]  # [1, 128]
logits_3 = self.trg_word_prj(last_step_output)  # [1, 41]
probs_3 = F.softmax(logits_3, dim=-1)  # [1, 41]

# 示例输出：
probs_3 = torch.tensor([[0.70, 0.10, 0.10, 0.05, 0.03, 0.01, 0.01, ...]])
# 解释：
# - 关系0 ("wasBornIn"): 70% 概率（高概率，可能表示推理完成）
# - 关系1 ("locatedIn"): 10% 概率（不太可能重复）
# - 其他关系: 剩余概率

# 选择关系
third_rel = torch.argmax(probs_3)  # 选中了关系0 ("wasBornIn")
```

##### Step 2.2.5：完整的关系序列生成

```python
# 最终生成的关系序列：
generated_relations = [first_rel, second_rel, third_rel]  # [0, 1, 0]
relation_names = ["wasBornIn", "locatedIn", "wasBornIn"]

# 推理路径解释：
# Step 1: wasBornIn(Einstein, Ulm)  - Einstein出生在Ulm
# Step 2: locatedIn(Ulm, Germany)   - Ulm位于德国
# Step 3: wasBornIn(推理完成)       - 推理完成，得出结论

# 最终挖掘的规则：
# wasBornIn(X,Y) ∧ locatedIn(Y,Z) => nationality(X,Z)
# 翻译：如果X出生在Y，且Y位于Z，那么X具有Z的国籍
```

---

##### Decoder与Encoder的交互机制

```python
class Decoder(nn.Module):
    def forward(self, trg_seq, enc_output, src_mask=None, ...):
        # trg_seq: [seq_len, 128] - 已经生成的关系序列
        # enc_output: [1, 140, 128] - 子图的上下文表示

        # 1. 自注意力：处理生成的关系之间的关系
        dec_output, self_attn = self.slf_attn(
            trg_seq, trg_seq, trg_seq, mask=slf_attn_mask
        )
        # 让模型知道：第一步和第二步的关系如何配合
        # 例如：知道第一步是wasBornIn，第二步应该选择什么关系配合

        # 2. 编码-解码注意力：关注子图中的相关信息
        dec_output, enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        # enc_attn: [seq_len, 140] - 每步关注子图中的哪些节点
        # 例如：第一步关注Einstein节点，第二步关注Ulm和Germany节点

        # 3. 前馈网络
        dec_output = self.pos_ffn(dec_output)

        return dec_output, enc_attn
```

---

##### 设计优势和应用场景

**为什么这样设计？**

```python
# 1. 上下文感知：每一步都考虑子图中的所有节点
#    Decoder知道子图中有哪些实体，以及它们之间的关系

# 2. 序列建模：能够捕捉关系步骤之间的依赖
#    第一步的选择会影响第二步的选择

# 3. 灵活性：可以生成不同长度的推理路径
#    有些规则可能2步就够了，有些需要3步或更多

# 4. 可解释性：每一步都有明确的关系含义
#    [wasBornIn, locatedIn] 很容易理解推理逻辑
```

**实际应用例子**：

```python
# 训练阶段（已知答案）：
target_triple = (Einstein, nationality, Germany)
# 模型学会生成：[wasBornIn, locatedIn]

# 解码阶段（发现新规则）：
query_triple = (Marie_Curie, nationality, ?)
# 模型可能生成：[wasBornIn, locatedIn]
# 发现规则：wasBornIn(X,Y) ∧ locatedIn(Y,Z) => nationality(X,Z)
# 应用：wasBornIn(Marie_Curie, Warsaw) ∧ locatedIn(Warsaw, Poland)
#       => nationality(Marie_Curie, Poland)
```

**输出总结**：
```python
# 最终Decoder输出：
dec_output = torch.tensor([
    [0.45, 0.23, 0.67, ..., 0.12],  # 第一步："wasBornIn"的内部表示
    [0.78, 0.34, 0.56, ..., 0.45],  # 第二步："locatedIn"的内部表示
    [0.23, 0.89, 0.12, ..., 0.67]   # 第三步：推理完成的表示
])  # [3, 128]

# 每一步都包含了该步骤对整个推理路径的理解！
```

#### 2.3 图推理和训练

```python
# 训练目标：预测目标实体的概率分布
logits_all = forwardAllNLP(enc_output, dec_output, subgraph)
target = torch.tensor([1.0]).unsqueeze(0)  # 目标实体

# 损失函数：负对数似然
loss = -torch.log(torch.sum(logits_all * target))
loss.backward()
optimizer.step()
```

### Step 3: 规则解码 - 路径搜索

#### 3.1 与训练阶段的异同

**相同部分**：
- 都使用Encoder+Decoder进行前向传播
- 都得到关系注意力分布 `dec_output`

**不同部分**：
- **训练**：图推理，快速向量化计算
- **解码**：路径搜索，慢速离散搜索

#### 3.2 路径搜索算法
```python
# 核心算法：beam search + 关系注意力
def decode_rule(model, data):
    paths = [[([], [], 1.0)]]  # [(实体序列, 关系序列, 概率)]

    for step in range(3):  # 最多3步
        current_paths = paths[-1]
        next_paths = []

        for entities, relations, prob in current_paths:
            # 根据关系注意力选择下一步
            for next_rel in top_k_relations:
                next_entities = get_neighbors(entities[-1], next_rel)
                for next_entity in next_entities:
                    next_prob = prob * relation_prob
                    next_paths.append((
                        entities + [next_entity],
                        relations + [next_rel],
                        next_prob
                    ))

        # 保留top-k路径
        paths.append(sorted(next_paths, reverse=True)[:5])
```

#### 3.3 规则输出
```python
# 解码结果：
rules = [
    (0.8, "wasBornIn(X,Y) ∧ locatedIn(Y,Z) => nationality(X,Z)"),
    (0.6, "wasBornIn(X,Y) ∧ partOf(Y,Z) => nationality(X,Z)"),
    (0.4, "locatedIn(X,Y) ∧ hasCapital(Y,Z) => nationality(X,Z)")
]
```

---

## 关键数据结构详解

### 关系矩阵 [140, 140, 41]

**维度含义**：
- **第一维(140)**: 源节点位置
- **第二维(140)**: 目标节点位置
- **第三维(41)**: 关系类型

**数据结构**：
```python
# 41个独立的140×140矩阵，每个矩阵对应一种关系类型
rela_mat[:,:,0]   # 关系0的邻接矩阵
rela_mat[:,:,1]   # 关系1(bornIn)的邻接矩阵
rela_mat[:,:,2]   # 关系2(locatedIn)的邻接矩阵
# ...

# 具体例子：
rela_mat[0, 1, 1] = 1   # Einstein -> bornIn -> Ulm
rela_mat[1, 2, 2] = 1   # Ulm -> locatedIn -> Germany
rela_mat[2, 4, 3] = 1   # Germany -> hasCapital -> Berlin
```

**特点**：
- **极度稀疏**：809,600个元素中可能只有几十个是1
- **每层独立**：每个关系类型都有自己的邻接矩阵
- **非对称**：`rela_mat[i,j,r] = 1` 不意味着 `rela_mat[j,i,r] = 1`

### 关键变量

**双向训练样本**：
```python
# 前向样本（Head → Tail）
subgraph_h = [H, node1, node2, ..., padding]  # H在第一个位置
rela_mat_h = [140, 140, 41]  # 前向关系矩阵
target_h = [H, R, T]  # 原始三元组

# 后向样本（Tail → Head，使用逆关系）
subgraph_t = [T, node1, node2, ..., padding]  # T在第一个位置
target_t = [T, R_inverse, H]  # 逆关系三元组

# 逆关系计算：
R_inverse = R + pos_rels * ((R<pos_rels) * 2 - 1)
```

### Padding机制

```python
# 为什么需要padding？
# 1. Transformer需要固定长度输入
# 2. GPU并行计算要求
# 3. 批处理效率

# Padding实现：
original_subgraph = [1, 2, 4]  # [Einstein, Ulm, Berlin]
padded_subgraph = original_subgraph + [0] * (140 - len(original_subgraph))
# 结果：[1, 2, 4, 0, 0, 0, 0, ..., 0]  # 137个0填充

# 0的含义：
# ID=0: <PAD> 特殊padding符号，不是真实实体
# 在计算时通过mask机制忽略
```

---

## 代码分析

### 1. translate.py (176行)
**功能**：主训练/解码脚本

**核心函数**：
```python
def load_model(opt):
    # 加载Transformer模型、优化器、数据集

def run(model, data, mode):
    if mode == 'train':
        # 训练循环
        for epoch in range(opt.epoch):
            model.train_batch(data)
    elif mode == 'test':
        # 规则解码
        model.decode(data)
```

**参数配置**：
```python
# 关键超参数
opt.d_word_vec = 128      # 词向量维度
opt.d_model = 128         # 模型维度
opt.d_inner = 512         # 前馈网络隐藏层
opt.n_layers = 6          # Transformer层数
opt.n_head = 8            # 多头注意力头数
opt.dropout = 0.1         # Dropout率
```

### 2. transformer/dataset.py (192行)
**功能**：数据预处理和子图提取

**核心类**：
```python
class DataBase:
    def __init__(self, opt):
        # 加载entities.txt, relations.txt, train.txt
        # 构建邻接表 neighbors[entity][relation] = [neighbor1, neighbor2, ...]

    def extract_without_token(self, head, JUMP, MAXN, PADDING):
        # BFS子图提取算法
        # 返回：(节点列表, 边列表, 每跳长度)

class pickleDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        # 为三元组生成双向训练样本
        # 前向：H→T，后向：T→H(逆关系)
```

**子图提取算法**：
```python
def extract_without_token(self, head, JUMP, MAXN, PADDING):
    subgraph = [head]
    length = [0]

    for jump in range(JUMP):
        start_idx = length[-1]
        length.append(len(subgraph))

        for parent_idx in range(start_idx, length[-1]):
            parent = subgraph[parent_idx]
            for rel, neighbors in self.neighbors[parent].items():
                # 限制邻居数量
                sampled_neighbors = neighbors[:MAXN]
                for neighbor in sampled_neighbors:
                    try:
                        pos = subgraph.index(neighbor)
                        relation.append((parent_idx, pos, rel))
                    except ValueError:
                        subgraph.append(neighbor)
                        relation.append((parent_idx, len(subgraph)-1, rel))

    return subgraph, relation, length
```

### 3. transformer/Models.py (205行)
**功能**：Transformer架构定义

**核心组件**：
```python
class Encoder(nn.Module):
    def forward(self, src_seq, src_mask, link=None, length=None):
        # 1. 节点嵌入 + 关系聚合
        enc_output = self.src_word_emb(src_seq)
        enc_output = (enc_output + torch.matmul(
            self.nebor_relation[src_seq], self.relation_type_E)) / 2

        # 2. 位置编码（跳数级别）
        enc_output = self.position_enc(enc_output, length)

        # 3. 多层关系感知注意力
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, link=link)

        return enc_output

class Decoder(nn.Module):
    def forward(self, trg_seq, enc_output, ...):
        # 自回归生成关系序列
        # 标准Transformer Decoder
```

**位置编码创新**：
```python
class PositionalEncoding(nn.Module):
    def forward(self, x, length=None):
        if length == None:
            # 传统位置编码
            return x + self.pos_table[:, :x.size(1)]

        # 跳数级位置编码
        for b in range(length.size(0)):
            tmp = length[b].tolist()
            for i in range(1, len(tmp)):
                # 相同跳数的节点共享相同编码
                x[b, tmp[i-1]:tmp[i]] += self.pos_table[0, i, :].unsqueeze(0)
```

### 4. transformer/Translator.py (174行)
**功能**：推理引擎，封装训练和规则解码

**核心方法**：
```python
class Translator(nn.Module):
    def forwardAllNLP(self, enc_output, dec_output, subgraph):
        # 图推理：向量化计算所有实体概率
        return entity_scores

    def decode_rule(self, enc_output, dec_output, subgraph, target):
        # 路径搜索：离散搜索推理路径
        return rules
```

### 5-8. Layers.py, SubLayers.py, Modules.py, Optim.py
**Layers.py**：EncoderLayer/DecoderLayer定义
**SubLayers.py**：MultiHeadAttention + PositionwiseFeedForward
**Modules.py**：ScaledDotProductAttention核心实现
**Optim.py**：学习率调度器

---

## 使用示例

### 训练模型
```bash
# 数据预处理
python3 translate.py --mode preprocess --data_path ./data/Family/ \
    --jump 2 --maxn 50 --padding 140

# 训练模型
python3 translate.py --mode train --data_path ./data/Family/ \
    --d_word_vec 128 --n_layers 6 --n_head 8 \
    --epoch 50 --batch_size 32

# 输出：Translator50.ckpt (模型文件)
```

### 解码规则
```bash
# 规则解码
python3 translate.py --mode test --data_path ./data/Family/ \
    --golden_model ./model/Translator50.ckpt

# 输出：rules.txt (挖掘的规则)
```

### 代码使用
```python
# 训练模式
model = load_model(opt)
model.train()
for epoch in range(50):
    loss = model.train_batch(train_data)
    print(f"Epoch {epoch}, Loss: {loss}")

# 解码模式
model.eval()
rules = model.decode_rule(test_data)
for rule in rules:
    print(f"Rule: {rule}")
```

---

## 总结

**Ruleformer的核心价值**：

1. **上下文感知**：通过Transformer的注意力机制，学习关系路径的语义表示
2. **端到端学习**：直接从三元组学习规则，无需手工特征工程
3. **双向推理**：通过前向和后向样本，增强模型的泛化能力
4. **可解释性**：生成的关系路径序列可以直接解释为逻辑规则

**适用场景**：
- 知识图谱补全和推理
- 关系路径发现和解释
- 逻辑规则挖掘和验证

**限制和改进方向**：
- 子图大小限制了推理范围
- 固定长度的关系序列可能丢失长尾规则
- 可以结合符号方法提升规则质量

---

## 参考文献

```bibtex
@inproceedings{xu-etal-2022-ruleformer,
    title = "Ruleformer: Context-aware Rule Mining over Knowledge Graph",
    author = "Xu, Zezhong and Ye, Peng and Chen, Hui and Zhao, Meng and Chen, Huajun and Zhang, Wen",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    year = "2022",
    pages = "2551--2560",
}
```