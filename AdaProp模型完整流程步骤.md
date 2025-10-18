# AdaProp模型完整流程步骤

## 🎯 文档目标

本文档详细讲解AdaProp模型从输入查询到输出答案的**完整流程**，包括每一步的：
- 输入输出
- 数据维度变化
- 核心操作
- 代码位置

---

## 📚 目录

1. [整体流程概览](#1-整体流程概览)
2. [步骤0：输入准备](#2-步骤0输入准备)
3. [步骤1：初始化](#3-步骤1初始化)
4. [步骤2：图扩展（获取邻居）](#4-步骤2图扩展获取邻居)
5. [步骤3：注意力机制打分](#5-步骤3注意力机制打分)
6. [步骤4：边采样（可选）⭐ Gumbel Softmax + 软硬采样](#6-步骤4边采样可选-gumbel-softmax--软硬采样)
7. [步骤5：消息传播](#7-步骤5消息传播)
8. [步骤6：节点采样 ⭐ Gumbel Softmax + 软硬采样](#8-步骤6节点采样-gumbel-softmax--软硬采样)
9. [步骤7：GRU门控更新](#9-步骤7gru门控更新)
10. [步骤8：多层迭代](#10-步骤8多层迭代)
11. [步骤9：最终预测](#11-步骤9最终预测)
12. [步骤10：训练与评估](#12-步骤10训练与评估)
13. [步骤11：反向传播详解 ⭐](#13-步骤11反向传播详解-)
14. [完整示例](#14-完整示例)
15. [数据流转总览](#15-数据流转总览)

---

## 1. 整体流程概览

### 🎨 一句话总结

> **AdaProp从查询实体出发，通过多层图神经网络逐步扩展和筛选候选节点，最终预测答案实体。**

### 📊 流程图

```
输入: (姚明, 工作地, ?)
    ↓
┌─────────────────────────────────────┐
│ 步骤1: 初始化                        │
│ - 创建初始节点表示                   │
│ - hidden = zeros [1, 64]            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 第1层 GNN                            │
│ ┌───────────────────────────────┐   │
│ │ 步骤2: 图扩展                  │   │
│ │ - 获取邻居边和节点             │   │
│ │ - nodes: [1] → [100]          │   │
│ │ - edges: [350条]              │   │
│ └───────────────────────────────┘   │
│ ┌───────────────────────────────┐   │
│ │ 步骤3: 注意力打分              │   │
│ │ - 计算每条边的重要性alpha      │   │
│ │ - alpha: [350, 1]             │   │
│ └───────────────────────────────┘   │
│ ┌───────────────────────────────┐   │
│ │ 步骤4: 边采样（可选）          │   │
│ │ - 从350条边选Top-K=200条      │   │
│ └───────────────────────────────┘   │
│ ┌───────────────────────────────┐   │
│ │ 步骤5: 消息传播                │   │
│ │ - 聚合邻居信息                 │   │
│ │ - hidden: [100, 64]           │   │
│ └───────────────────────────────┘   │
│ ┌───────────────────────────────┐   │
│ │ 步骤6: 节点采样                │   │
│ │ - 从100个节点选Top-K=20个     │   │
│ │ - nodes: [100] → [20]         │   │
│ └───────────────────────────────┘   │
│ ┌───────────────────────────────┐   │
│ │ 步骤7: GRU更新                 │   │
│ │ - 融合历史信息                 │   │
│ │ - hidden: [20, 64]            │   │
│ └───────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 第2层 ~ 第L层                        │
│ (重复步骤2-7)                        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 步骤9: 最终预测                      │
│ - scores = W_final(hidden)          │
│ - 选择得分最高的实体                 │
└─────────────────────────────────────┘
    ↓
输出: 休斯顿 (score=8.92)
```

### 🔑 关键点

1. **多层结构**：通常8层，每层逐步探索更远的邻居
2. **双重采样**：边采样（减少边）+ 节点采样（减少节点）
3. **注意力机制**：不是所有信息都同等重要
4. **增量传播**：每层只处理新增的节点，提高效率

---

## 2. 步骤0：输入准备

### 📥 输入

```python
# 查询格式: (subject, relation, ?)
query = (姚明, 工作地, ?)

# 转换为ID
sub_id = 12345   # 姚明的实体ID
rel_id = 67      # "工作地"关系的ID
```

### 🎯 目标

回答：**"姚明在哪里工作？"**

### 📍 代码位置

```python
# base_model.py 第71-82行
for iteration, batch in enumerate(train_data):
    subs, rels, objs = batch[:,0], batch[:,1], batch[:,2]

    # 前向传播
    scores = model(subs, rels, mode='train')
```

---

## 3. 步骤1：初始化

### 🎯 目的

创建初始状态，准备开始图探索。

### 📥 输入

```python
subs = [12345]        # 姚明的ID
rels = [67]           # 工作地关系ID
batchsize = 1
```

### ⚙️ 操作

```python
# models.py 第154-159行
n = len(subs)  # batchsize = 1
q_sub = torch.LongTensor(subs).cuda()      # [1]
q_rel = torch.LongTensor(rels).cuda()      # [1]

# 初始化隐藏状态（全0）
h0 = torch.zeros((1, n, self.hidden_dim)).cuda()  # [1, 1, 64]
hidden = torch.zeros(n, self.hidden_dim).cuda()   # [1, 64]

# 初始节点：(batch_idx, entity_id)
nodes = torch.cat([
    torch.arange(n).unsqueeze(1).cuda(),  # batch索引 [1, 1]
    q_sub.unsqueeze(1)                     # 实体ID [1, 1]
], 1)  # [1, 2]
```

### 📤 输出

```python
nodes = [[0, 12345]]  # (batch=0, entity=姚明)
         # ↑批次  ↑实体ID

hidden = [[0, 0, 0, ..., 0]]  # 64个0
         # ↑64维全0向量

h0 = [[[0, 0, 0, ..., 0]]]  # GRU初始状态
```

### 📊 维度

| 变量 | 维度 | 含义 |
|------|------|------|
| `q_sub` | [1] | 查询实体ID |
| `q_rel` | [1] | 查询关系ID |
| `nodes` | [1, 2] | 当前节点列表 (batch_idx, entity_id) |
| `hidden` | [1, 64] | 节点的表示向量 |
| `h0` | [1, 1, 64] | GRU隐藏状态 |

---

## 4. 步骤2：图扩展（获取邻居）

### 🎯 目的

从当前节点出发，获取所有邻居节点和连接边。

### 📥 输入

```python
nodes = [[0, 12345]]  # 姚明
n = 1  # batchsize
mode = 'train'
```

### ⚙️ 操作

```python
# models.py 第166行
nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(
    nodes.data.cpu().numpy(),
    n,
    mode=mode
)
```

**`get_neighbors` 做了什么？**

1. 查询知识图谱，找到姚明的所有邻居
2. 返回三元组：(姚明, 关系, 邻居)

### 📤 输出

```python
# 假设姚明有100个邻居节点，350条边（包括双向）

nodes = [
    [0, 12345],   # 姚明（旧节点）
    [0, 54321],   # 休斯顿（新节点）
    [0, 11111],   # 火箭队（新节点）
    [0, 22222],   # 上海（新节点）
    ...           # 共100个节点
]  # [100, 2]

edges = [
    [0, 12345, 67,  54321, 0, 1],  # 姚明-[工作地]->休斯顿
    [0, 12345, 12,  22222, 0, 3],  # 姚明-[出生地]->上海
    [0, 12345, 156, 11111, 0, 2],  # 姚明-[效力]->火箭队
    ...                             # 共350条边
]  # [350, 6]

# 边的格式: [batch_idx, head_entity, relation, tail_entity, head_idx, tail_idx]
#                ↑          ↑           ↑         ↑           ↑         ↑
#              批次       头实体ID    关系ID    尾实体ID    头索引    尾索引

old_nodes_new_idx = [0]  # 姚明在新nodes中的索引是0
```

### 📊 维度

| 变量 | 维度 | 含义 |
|------|------|------|
| `nodes` | [100, 2] | 100个节点 (batch_idx, entity_id) |
| `edges` | [350, 6] | 350条边 (batch, head, rel, tail, head_idx, tail_idx) |
| `old_nodes_new_idx` | [1] | 旧节点在新nodes中的索引 |

### 🎨 形象理解

```
第0层:
  姚明

第1层:
  姚明 ──工作地──> 休斯顿
  姚明 ──出生地──> 上海
  姚明 ──效力───> 火箭队
  姚明 ──队友───> 麦迪
  ... (100个邻居)
```

---

## 5. 步骤3：注意力机制打分

### 🎯 目的

计算每条边的重要性，决定信息传递的权重。

### 📥 输入

```python
edges = [350, 6]   # 350条边
hidden = [100, 64] # 100个节点的表示（扩展后包含新节点的0向量）
q_rel = [1]        # 查询关系ID
```

### ⚙️ 操作

```python
# models.py 第45-51行
# 提取边的信息
sub = edges[:,4]  # 源节点索引 [350]
rel = edges[:,2]  # 关系ID [350]
obj = edges[:,5]  # 目标节点索引 [350]

# 获取表示向量
hs = hidden[sub]           # 源节点表示 [350, 64]
hr = self.rela_embed(rel)  # 关系表示 [350, 64]
r_idx = edges[:,0]         # batch索引 [350]
h_qr = self.rela_embed(q_rel)[r_idx]  # 查询关系表示 [350, 64]
```

```python
# models.py 第57行或第66行
# 计算注意力分数
alpha = self.w_alpha(
    nn.ReLU()(
        self.Ws_attn(hs) +      # 源节点特征 [350, attn_dim]
        self.Wr_attn(hr) +      # 关系特征 [350, attn_dim]
        self.Wqr_attn(h_qr)     # 查询特征 [350, attn_dim]
    )
)  # [350, 1]

alpha = torch.sigmoid(alpha)  # 转换到[0,1]范围
```

### 📤 输出

```python
alpha = [
    [0.92],  # 姚明-[工作地]->休斯顿 (很重要！)
    [0.78],  # 姚明-[效力]->火箭队
    [0.45],  # 姚明-[队友]->麦迪
    [0.23],  # 姚明-[出生地]->上海
    [0.08],  # 姚明-[喜欢吃]->火锅 (不重要)
    ...
]  # [350, 1]
```

### 📊 维度

| 变量 | 维度 | 含义 |
|------|------|------|
| `hs` | [350, 64] | 源节点表示 |
| `hr` | [350, 64] | 关系表示 |
| `h_qr` | [350, 64] | 查询关系表示 |
| `alpha` | [350, 1] | 注意力分数（0-1之间） |

### 🎨 形象理解

```
注意力机制就像给每条边打分：

姚明 --[工作地]--> 休斯顿    alpha=0.92 (非常相关！✅)
姚明 --[出生地]--> 上海      alpha=0.23 (不太相关)
姚明 --[喜欢吃]--> 火锅      alpha=0.08 (几乎无关)
```

### 💡 为什么考虑三个因素？

- **源节点特征 (hs)**：姚明是谁
- **关系特征 (hr)**：这条边是什么关系
- **查询特征 (h_qr)**：我们要找什么关系

三者结合，判断这条边对回答问题的重要性！

---

## 6. 步骤4：边采样（可选）⭐ Gumbel Softmax + 软硬采样

### 🎯 目的

从大量边中选择最重要的Top-K条，减少计算量。

**⭐ 这里使用了 Gumbel Softmax 和软硬采样技术！**

### 📥 输入

```python
alpha = [350, 1]     # 注意力分数
n_edge_topk = 200    # 只保留200条边
mode = 'train'       # 训练模式或测试模式
```

### ⚙️ 操作

```python
# models.py 第56-63行
if self.n_edge_topk > 0:
    # 步骤1: ⭐ 使用Gumbel Softmax计算采样概率（软采样）
    edge_prob = F.gumbel_softmax(alpha, tau=1, hard=False)  # [350]

    # 步骤2: 选择Top-K条边
    topk_index = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]

    # 步骤3: 创建硬选择（硬采样：0或1）
    edge_prob_hard = torch.zeros((alpha.shape[0])).cuda()
    edge_prob_hard[topk_index] = 1  # Top-K条边设为1，其他为0

    # 步骤4: ⭐ Straight-through Estimator（前向硬采样，反向软采样）
    alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
```

---

### 🌟 技术详解1：Gumbel Softmax

**📍 代码位置：** `models.py` 第58行

```python
edge_prob = F.gumbel_softmax(alpha, tau=1, hard=False)
```

#### 什么是Gumbel Softmax？

**Gumbel Softmax 是一种可微分的采样技术**，让离散的选择操作（选或不选）变得可以训练。

#### 为什么需要它？

**问题：** 普通的Top-K选择不可微分
```python
# 普通Top-K（不可微分❌）
topk_index = torch.topk(alpha, k=200)  # 硬选择，无法反向传播梯度
```

**解决：** Gumbel Softmax（可微分✅）
```python
# Gumbel Softmax（可微分）
edge_prob = F.gumbel_softmax(alpha, tau=1, hard=False)  # 软选择，可以训练
```

#### Gumbel Softmax如何工作？

**数学公式：**
```
gumbel_noise = -log(-log(uniform(0,1)))  # Gumbel分布噪声
edge_prob = softmax((alpha + gumbel_noise) / tau)
```

**参数说明：**
- `tau`（温度）：控制采样的"硬度"
  - `tau → 0`：接近硬选择（one-hot）
  - `tau → ∞`：接近均匀分布
  - `tau = 1`：平衡的选择

#### 具体例子

```python
# 输入：注意力分数
alpha = [0.92, 0.78, 0.45, 0.23, 0.08]  # 5条边的注意力分数

# 步骤1：添加Gumbel噪声（增加随机性）
gumbel_noise = [-0.23, 0.15, -0.67, 0.42, -1.34]  # 随机噪声
logits = alpha + gumbel_noise
       = [0.69, 0.93, -0.22, 0.65, -1.26]

# 步骤2：Softmax归一化
edge_prob = softmax(logits / tau)
          = [0.32, 0.41, 0.13, 0.12, 0.02]  # 软概率（可微分！）

# 对比：如果没有Gumbel噪声，只用普通softmax
normal_prob = softmax(alpha)
            = [0.42, 0.36, 0.16, 0.05, 0.01]  # 确定性的，探索性差
```

#### 为什么加入随机噪声？

1. **训练时的探索**：避免每次都选相同的边
2. **防止过拟合**：增加模型的泛化能力
3. **梯度流动**：让所有边都有机会被选中，梯度可以流向所有参数

---

### 🌟 技术详解2：软采样 vs 硬采样

#### 什么是软采样和硬采样？

| 类型 | 输出 | 可微分？ | 何时使用 |
|------|------|---------|---------|
| **软采样** | 概率值 [0.32, 0.41, 0.13, 0.12, 0.02] | ✅ 是 | 反向传播（训练） |
| **硬采样** | 0/1值 [0, 1, 0, 0, 0] | ❌ 否 | 前向传播（推理） |

#### 具体例子

**软采样（Soft Sampling）：**
```python
# Gumbel Softmax的输出
edge_prob = [0.32, 0.41, 0.13, 0.12, 0.02]  # 软概率

# 使用软采样：每条边都有权重
message_1 = 0.32 * edge_1  # 第1条边贡献32%
message_2 = 0.41 * edge_2  # 第2条边贡献41%
message_3 = 0.13 * edge_3  # 第3条边贡献13%
...
```

**硬采样（Hard Sampling）：**
```python
# 选择Top-2条边
topk_index = [1, 0]  # 索引1和0的边被选中

# 硬选择：0或1
edge_prob_hard = [1, 1, 0, 0, 0]  # 只有Top-2是1，其他是0

# 使用硬采样：只保留选中的边
message_1 = 1.0 * edge_1  # 第1条边贡献100%
message_2 = 1.0 * edge_2  # 第2条边贡献100%
message_3 = 0.0 * edge_3  # 第3条边不贡献 ❌
...
```

#### 为什么需要两种采样？

**前向传播（Forward）：使用硬采样**
- 推理时需要明确的决策（选或不选）
- 减少计算量（只处理选中的边）
- 符合实际应用场景

**反向传播（Backward）：使用软采样**
- 需要梯度能够流向所有边
- 让模型学习哪些边应该被选中
- 避免梯度为0的问题

---

### 🌟 技术详解3：Straight-through Estimator

**📍 代码位置：** `models.py` 第62行

```python
alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
```

#### 什么是Straight-through Estimator？

**一个神奇的技巧**：前向传播用硬采样，反向传播用软采样！

#### 数学原理

```python
# 魔法公式
output = edge_prob_hard - edge_prob.detach() + edge_prob
       = edge_prob_hard + (edge_prob - edge_prob.detach())
```

**分解理解：**

**前向传播：**
```python
edge_prob.detach() == edge_prob  # detach()切断梯度，但值相同

output = edge_prob_hard - edge_prob + edge_prob
       = edge_prob_hard  # 前向时使用硬选择！
```

**反向传播：**
```python
∂output/∂edge_prob = ∂(edge_prob_hard - edge_prob.detach() + edge_prob)/∂edge_prob
                   = 0 - 0 + 1  # detach()梯度为0，edge_prob_hard是常数
                   = 1  # 反向时梯度直接传给edge_prob！
```

#### 具体例子

```python
# 假设Top-2是索引1和0
edge_prob = [0.32, 0.41, 0.13, 0.12, 0.02]  # 软概率
edge_prob_hard = [1, 1, 0, 0, 0]            # 硬选择

# 前向计算（使用硬选择）
output = [1, 1, 0, 0, 0] - [0.32, 0.41, 0.13, 0.12, 0.02] + [0.32, 0.41, 0.13, 0.12, 0.02]
       = [1, 1, 0, 0, 0]  # 前向：只保留Top-2条边

# 反向传播（梯度传给软概率）
∂loss/∂edge_prob = ∂loss/∂output * ∂output/∂edge_prob
                  = ∂loss/∂output * 1
                  # 梯度完整传递！不会因为硬选择而中断
```

#### 为什么有效？

1. **前向**：使用硬选择 → 高效推理，明确决策
2. **反向**：梯度传给软概率 → 模型可以学习
3. **结果**：既享受硬选择的效率，又享受可微分的训练性

---

### 🎨 训练模式 vs 测试模式

#### 训练模式（mode='train'）

```python
# models.py 第33-34行
if self.training and self.tau > 0:
    self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
```

**特点：**
- 使用 Gumbel Softmax（加噪声）
- 软采样 + 硬采样结合（Straight-through）
- 探索性强，避免过拟合

**流程：**
```
alpha → [Gumbel Softmax] → edge_prob (软概率)
                          ↓
                    选Top-K → edge_prob_hard (硬选择)
                          ↓
            [Straight-through] → 前向用hard，反向用soft
```

#### 测试模式（mode='test'）

```python
# models.py 第36行
else:
    self.softmax = lambda x: F.softmax(x, dim=1)
```

**特点：**
- 使用普通 Softmax（无噪声）
- 确定性选择
- 稳定的预测结果

**流程：**
```
alpha → [普通 Softmax] → edge_prob (确定性概率)
                       ↓
                 选Top-K → edge_prob_hard
                       ↓
                    前向传播（无反向）
```

---

### 📤 输出

```python
# alpha被修改：只有Top-200条边的alpha非零
alpha = [
    [0.92],  # 姚明-[工作地]->休斯顿 ✅ 保留
    [0.78],  # 姚明-[效力]->火箭队 ✅ 保留
    ...      # Top-200条边保留
    [0.00],  # 某条边被丢弃 ❌
    [0.00],  # 某条边被丢弃 ❌
    ...
]  # [350, 1] (但只有200个非零)
```

### 📊 效果

```
原始: 350条边 → 计算量 = 350
采样后: 200条有效边 → 计算量 = 200 (减少43%)

同时保持可微分：梯度可以反向传播，模型可以学习！✅
```

### 🎨 形象理解

```
海选环节（但是可以学习的海选！）：

普通海选（不可微）：
  350个选手 → 直接选Top-200 → 被淘汰的选手无法改进 ❌

Gumbel Softmax海选（可微）：
  350个选手 → Gumbel采样 → 选Top-200
              ↓
  反向传播时，所有选手都能收到反馈 ✅
  下次选拔时，表现会更好！
```

### 🔑 核心要点

1. **⭐ Gumbel Softmax 用在这里！**
   - `F.gumbel_softmax(alpha, tau=1, hard=False)`
   - 让离散选择变得可微分

2. **⭐ 软硬采样结合！**
   - 软采样：反向传播，学习参数
   - 硬采样：前向传播，高效推理
   - Straight-through：无缝衔接两者

3. **训练 vs 测试差异**
   - 训练：Gumbel Softmax（探索）
   - 测试：普通 Softmax（确定性）

### ⚠️ 注意

如果 `n_edge_topk <= 0`，则跳过此步骤，保留所有边。

---

## 7. 步骤5：消息传播

### 🎯 目的

通过边传递信息，更新节点的表示向量。

### 📥 输入

```python
edges = [350, 6]     # 边（但只有200条有效）
hidden = [100, 64]   # 旧的节点表示
alpha = [350, 1]     # 注意力权重
```

### ⚙️ 操作

```python
# models.py 第53行
# 构造消息：节点表示 + 关系表示
message = hs + hr  # [350, 64]

# models.py 第69行
# 用注意力权重调节消息
message = alpha * message  # [350, 64]

# models.py 第70行
# 聚合消息到目标节点
message_agg = scatter(
    message,           # 要聚合的消息 [350, 64]
    index=obj,         # 目标节点索引 [350]
    dim=0,             # 在第0维聚合
    dim_size=n_node,   # 输出大小 = 100
    reduce='sum'       # 求和
)  # [100, 64]

# models.py 第71行
# 通过线性变换和激活函数
hidden_new = self.act(self.W_h(message_agg))  # [100, 64]
```

### 📤 输出

```python
hidden_new = [
    [0.00, 0.00, ..., 0.00],  # 姚明（没有接收消息）
    [0.83, -0.45, 0.62, ...],  # 休斯顿（接收了高质量消息）
    [0.71, -0.32, 0.54, ...],  # 火箭队
    [0.22, -0.08, 0.15, ...],  # 上海
    ...
]  # [100, 64]
```

### 📊 维度

| 变量 | 维度 | 含义 |
|------|------|------|
| `message` | [350, 64] | 每条边的消息 |
| `message_agg` | [100, 64] | 聚合后的消息 |
| `hidden_new` | [100, 64] | 更新后的节点表示 |

### 🎨 形象理解

```
消息传播就像信息流动：

姚明 --[工作地, α=0.92]--> 休斯顿
     传递消息: 0.92 * (姚明表示 + 工作地关系表示)

休斯顿接收到的总消息 = 从所有指向它的边收集消息并求和

休斯顿的新表示 = 变换(接收到的总消息)
```

### 💡 关键技术：torch_scatter

```python
scatter(message, index=obj, reduce='sum')
```

**作用：** 高效地将消息聚合到目标节点

```
边:
  姚明(idx=0) --msg1--> 休斯顿(idx=1)
  姚明(idx=0) --msg2--> 火箭队(idx=2)
  火箭队(idx=2) --msg3--> 休斯顿(idx=1)

聚合:
  休斯顿(idx=1) = msg1 + msg3
  火箭队(idx=2) = msg2
```

---

## 8. 步骤6：节点采样 ⭐ Gumbel Softmax + 软硬采样

### 🎯 目的

从候选节点中选择Top-K个最有希望的，控制节点数量。

**⭐ 这里也使用了 Gumbel Softmax 和软硬采样技术！**（与边采样类似，但应用于节点）

### 📥 输入

```python
hidden_new = [100, 64]  # 所有节点的新表示
nodes = [100, 2]        # 所有节点
n_node_topk = 20        # 只保留20个节点
old_nodes_new_idx = [1] # 旧节点索引（姚明）
mode = 'train'          # 训练模式或测试模式
```

### ⚙️ 操作

**第1步：识别新节点**

```python
# models.py 第80-83行
# 创建一个mask，标记哪些是新节点
tmp_diff_node_idx = torch.ones(n_node)  # [100]，全是1
tmp_diff_node_idx[old_nodes_new_idx] = 0  # 旧节点设为0
bool_diff_node_idx = tmp_diff_node_idx.bool()  # 转为布尔

# 提取新节点
diff_node = nodes[bool_diff_node_idx]  # [99, 2] (除了姚明，其他99个都是新节点)
```

**第2步：计算节点分数**

```python
# models.py 第86行
# 对每个新节点计算重要性分数
diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1)
# [99] (99个新节点的分数)

# models.py 第89-90行
# 把分数映射到完整的实体空间
node_scores = torch.ones((batchsize, self.n_ent)).cuda() * float('-inf')
# [1, 40943] (假设有40943个实体)，初始化为-∞

node_scores[diff_node[:,0], diff_node[:,1]] = diff_node_logit
# 把99个新节点的分数填入对应位置
```

**第3步：⭐ 使用Gumbel Softmax选择Top-K节点**

```python
# models.py 第95-99行
# ⭐ 转换为概率（训练时用Gumbel Softmax，测试时用普通Softmax）
node_scores = self.softmax(node_scores)  # [1, 40943]

# 选择Top-20个节点
topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
# [20] (20个实体ID)

topk_batchidx = torch.arange(batchsize).repeat(self.n_node_topk, 1).T.reshape(-1)
# [20] (都是0，因为batchsize=1)

batch_topk_nodes = torch.zeros((batchsize, self.n_ent)).cuda()  # [1, 40943]
batch_topk_nodes[topk_batchidx, topk_index] = 1
# 标记Top-20节点为1，其他为0（硬采样）
```

**第4步：⭐ 应用Straight-through Estimator**

```python
# models.py 第107-109行
# 软概率（可微分）
diff_node_prob = node_scores[diff_node[:,0], diff_node[:,1]]  # [99]

# 硬选择（0或1）
diff_node_prob_hard = batch_topk_nodes[diff_node[:,0], diff_node[:,1]]  # [99]

# ⭐ Straight-through: 前向用hard，反向用soft
hidden_new[bool_diff_node_idx] *= (
    diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob
).unsqueeze(-1)
```

**第5步：提取保留的节点**

```python
# models.py 第102-104行
# 创建最终的节点mask
bool_sampled_diff_nodes_idx = batch_topk_nodes[diff_node[:,0], diff_node[:,1]].bool()
bool_same_node_idx = ~bool_diff_node_idx.cuda()  # 先取反（全False）
bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes_idx  # 填入采样结果

# models.py 第112-113行
# 提取保留的节点和表示
new_nodes = nodes[bool_same_node_idx]      # [21, 2] (1个旧节点 + 20个新节点)
hidden_new = hidden_new[bool_same_node_idx]  # [21, 64]
```

---

### 🌟 技术详解1：节点采样中的Gumbel Softmax

**📍 代码位置：** `models.py` 第29-36行（train方法）+ 第95行

```python
# 第29-36行：根据训练/测试模式选择softmax函数
def train(self, mode=True):
    if self.training and self.tau > 0:
        self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
    else:
        self.softmax = lambda x: F.softmax(x, dim=1)

# 第95行：使用softmax转换节点分数
node_scores = self.softmax(node_scores)
```

#### 训练模式 vs 测试模式

**训练模式（mode='train'）：**
```python
# 使用Gumbel Softmax（加随机噪声）
node_scores = F.gumbel_softmax(node_scores, tau=self.tau, hard=False)

# 例子：
原始分数: [2.3, 1.8, 1.2, 0.5, 0.1, ...]  # 99个新节点
         ↓ 加Gumbel噪声
带噪声:  [2.1, 2.0, 0.8, 0.9, -0.3, ...]  # 随机性！
         ↓ Softmax
软概率:  [0.25, 0.23, 0.07, 0.08, 0.02, ...]  # 可微分的概率
```

**测试模式（mode='test'）：**
```python
# 使用普通Softmax（确定性）
node_scores = F.softmax(node_scores, dim=1)

# 例子：
原始分数: [2.3, 1.8, 1.2, 0.5, 0.1, ...]
         ↓ 普通Softmax（无噪声）
确定性概率: [0.28, 0.21, 0.12, 0.06, 0.04, ...]  # 每次都一样
```

#### 为什么节点采样也需要Gumbel Softmax？

原因与边采样相同：

1. **可微分训练**：让模型学习哪些节点重要
2. **探索性**：训练时尝试不同的节点组合
3. **防止过拟合**：避免每次都选相同的节点

---

### 🌟 技术详解2：节点采样中的软硬采样

#### 软采样（Soft Sampling）

```python
# Gumbel Softmax的输出（软概率）
node_scores = [0.25, 0.23, 0.07, 0.08, 0.02, ...]  # [1, 40943]

# 对应节点
休斯顿:     0.25  # 25%概率
火箭队:     0.23  # 23%概率
得克萨斯州: 0.07  # 7%概率
...
```

**用途：反向传播时使用**
- 所有节点都有概率值
- 梯度可以传递给所有节点
- 模型可以学习调整节点分数

#### 硬采样（Hard Sampling）

```python
# 选择Top-20
topk_index = torch.topk(node_scores, k=20).indices
# topk_index = [54321, 11111, 33333, ...]  # 20个实体ID

# 创建硬选择（0或1）
batch_topk_nodes = torch.zeros((1, 40943))
batch_topk_nodes[0, topk_index] = 1

# 结果
休斯顿:     1  # 选中 ✅
火箭队:     1  # 选中 ✅
得克萨斯州: 1  # 选中 ✅
...（共20个）
其他节点:   0  # 未选中 ❌
```

**用途：前向传播时使用**
- 明确的选择（选中或未选中）
- 减少计算量（只处理20个节点）
- 高效推理

---

### 🌟 技术详解3：节点采样中的Straight-through Estimator

**📍 代码位置：** `models.py` 第107-109行

```python
hidden_new[bool_diff_node_idx] *= (
    diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob
).unsqueeze(-1)
```

#### 具体例子

假设有99个新节点，我们选Top-20：

```python
# 软概率（Gumbel Softmax输出）
diff_node_prob = [
    0.25,  # 休斯顿（新节点1）
    0.23,  # 火箭队（新节点2）
    0.18,  # 得克萨斯州（新节点3）
    ...    # Top-20
    0.02,  # 某节点（新节点21，未入选）
    0.01,  # 某节点（新节点22，未入选）
    ...
]  # [99]

# 硬选择（0或1）
diff_node_prob_hard = [
    1,  # 休斯顿 ✅ 选中
    1,  # 火箭队 ✅ 选中
    1,  # 得克萨斯州 ✅ 选中
    ...  # Top-20都是1
    0,  # 某节点 ❌ 未选中
    0,  # 某节点 ❌ 未选中
    ...
]  # [99]

# Straight-through计算
scale_factor = diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob
             = [1, 1, 1, ..., 0, 0, ...] - [0.25, 0.23, 0.18, ..., 0.02, 0.01, ...] + [0.25, 0.23, 0.18, ..., 0.02, 0.01, ...]
             = [1, 1, 1, ..., 0, 0, ...]  # 前向：使用硬选择

# 更新节点表示
hidden_new[新节点] *= scale_factor.unsqueeze(-1)
```

**前向传播：**
```python
# 选中的节点：scale = 1，表示保持不变
休斯顿_hidden *= 1  → 保留完整表示 ✅
火箭队_hidden *= 1  → 保留完整表示 ✅

# 未选中的节点：scale = 0，表示丢弃
某节点_hidden *= 0  → 清零，不参与后续计算 ❌
```

**反向传播：**
```python
# 梯度完整传递（detach()切断了hard的梯度）
∂loss/∂diff_node_prob = ∂loss/∂output * 1

# 所有节点都能收到梯度反馈
∂loss/∂W_samp  # 采样权重可以学习
```

---

### 🌟 技术详解4：为什么节点采样比边采样更复杂？

#### 边采样 vs 节点采样

| 对比项 | 边采样 | 节点采样 |
|-------|-------|---------|
| **采样对象** | 边（当前层的所有边） | 节点（只采样新节点） |
| **空间大小** | 通常几百条边 | 完整实体空间（40943个实体） |
| **Softmax应用** | 直接对边分数 | 需要映射到实体空间 |
| **旧节点处理** | 不涉及 | 必须保留旧节点 |
| **复杂度** | 相对简单 | 更复杂 |

#### 节点采样的额外步骤

1. **识别新旧节点**：
   - 旧节点（姚明）必须保留
   - 只对新节点进行采样

2. **映射到实体空间**：
   - 新节点只有99个
   - 但Softmax在40943维空间上计算
   - 目的：确保概率归一化正确

3. **合并结果**：
   - 旧节点（1个）+ 采样的新节点（20个）
   - 最终得到21个节点

---

### 🎨 完整流程可视化

```
步骤1: 识别新节点
  所有节点[100] → 分离 → 旧节点[1] + 新节点[99]
                          ↓保留     ↓采样

步骤2: 计算节点分数
  新节点[99] → W_samp → logit[99]

步骤3: 映射到实体空间
  logit[99] → 映射 → node_scores[1, 40943]
  (其他位置填-∞)

步骤4: ⭐ Gumbel Softmax（训练模式）
  node_scores → [加Gumbel噪声] → softmax → 软概率[1, 40943]

步骤5: 选择Top-K
  软概率 → topk(k=20) → 20个实体ID

步骤6: 创建硬选择
  20个ID → 标记为1 → hard_selection[1, 40943]

步骤7: ⭐ Straight-through Estimator
  软概率 + 硬选择 → 前向用hard，反向用soft

步骤8: 更新节点表示
  hidden[新节点] *= scale_factor
  (选中的×1，未选中的×0)

步骤9: 合并节点
  旧节点[1] + 选中的新节点[20] = 最终节点[21]
```

---

### 📤 输出

```python
new_nodes = [
    [0, 12345],  # 姚明（旧节点，保留）
    [0, 54321],  # 休斯顿（新节点，Top-1）
    [0, 11111],  # 火箭队（新节点，Top-2）
    [0, 33333],  # 得克萨斯州（新节点，Top-3）
    ...          # 共20个新节点
]  # [21, 2]

hidden_new = [
    [0.00, 0.00, ..., 0.00],     # 姚明
    [0.83, -0.45, 0.62, ...],    # 休斯顿
    [0.71, -0.32, 0.54, ...],    # 火箭队
    ...
]  # [21, 64]
```

### 📊 维度变化

| 阶段 | nodes维度 | hidden维度 |
|------|----------|-----------|
| 输入 | [100, 2] | [100, 64] |
| 输出 | [21, 2] | [21, 64] |
| **变化** | **100 → 21** | **100 → 21** |

### 🎨 形象理解

```
节点采样就像晋级赛：

100个候选 → 评分 → 选Top-20

保留：
✅ 姚明（旧节点，必须保留）
✅ 休斯顿（score=0.85，排名1）
✅ 火箭队（score=0.72，排名2）
✅ 得克萨斯州（score=0.68，排名3）
...
✅ 第20名

淘汰：
❌ 第21名
❌ 第22名
...
❌ 第99名
```

### 💡 为什么要保留旧节点？

旧节点（姚明）必须保留，因为：
1. 它是查询的起点
2. 后续层需要从它继续扩展
3. 保持图的连通性

---

## 9. 步骤7：GRU门控更新

### 🎯 目的

融合历史信息和当前层信息，更新节点表示。

### 📥 输入

```python
hidden = [21, 64]   # 当前层的节点表示
h0 = [1, 1, 64]     # 上一层的隐藏状态（对应姚明）
old_nodes_new_idx   # 旧节点在当前nodes中的索引
sampled_nodes_idx   # 采样保留的节点索引
```

### ⚙️ 操作

```python
# models.py 第174-178行

# 步骤1: 扩展h0以匹配所有节点（采样前的大小）
h0 = torch.zeros(1, n_node, hidden.size(1)).cuda()  # [1, 100, 64]，全0
h0 = h0.index_copy_(1, old_nodes_new_idx, h0)       # 把旧节点的h0复制进去

# 步骤2: 只保留采样后的节点
h0 = h0[0, sampled_nodes_idx, :].unsqueeze(0)  # [1, 21, 64]

# 步骤3: Dropout
hidden = self.dropout(hidden)  # [21, 64]

# 步骤4: GRU更新
hidden, h0 = self.gate(
    hidden.unsqueeze(0),  # [1, 21, 64] 当前输入
    h0                    # [1, 21, 64] 历史状态
)

# 步骤5: 去掉batch维度
hidden = hidden.squeeze(0)  # [21, 64]
```

### 📤 输出

```python
hidden = [
    [0.15, 0.23, ..., -0.18],  # 姚明（融合了历史信息）
    [0.89, -0.52, 0.71, ...],  # 休斯顿（融合了历史信息）
    [0.76, -0.38, 0.61, ...],  # 火箭队
    ...
]  # [21, 64]

h0 = [[[same as hidden]]]  # [1, 21, 64] 更新后的隐藏状态
```

### 📊 维度

| 变量 | 维度 | 含义 |
|------|------|------|
| `hidden` (输入) | [21, 64] | 当前层表示 |
| `h0` (输入) | [1, 1, 64] | 上一层状态 |
| `hidden` (输出) | [21, 64] | 融合后的表示 |
| `h0` (输出) | [1, 21, 64] | 更新后的状态 |

### 🎨 形象理解

**GRU就像一个信息过滤器：**

```
当前层信息: "休斯顿是答案！"（强信号）
历史信息: "上海是出生地"（旧信息）

GRU决定：
- 保留80%的当前信息（休斯顿很重要）
- 保留20%的历史信息（上海作为背景）

融合结果: 休斯顿的表示更强，上海的表示减弱
```

### 💡 GRU的作用

1. **记忆管理**：决定记住什么，忘记什么
2. **信息融合**：平衡新旧信息
3. **梯度流动**：缓解梯度消失问题

### 🔧 GRU公式

```
r_t = σ(W_r * [h_{t-1}, x_t])      # 重置门
z_t = σ(W_z * [h_{t-1}, x_t])      # 更新门
h̃_t = tanh(W * [r_t ⊙ h_{t-1}, x_t])  # 候选状态
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # 最终状态
```

---

## 10. 步骤8：多层迭代

### 🎯 目的

重复步骤2-7，逐层扩展图结构，探索更远的邻居。

### ⚙️ 操作

```python
# models.py 第161-178行
for i in range(self.n_layer):  # 假设n_layer=8
    # 步骤2: 获取邻居
    nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(...)

    # 步骤3-6: GNN层处理
    hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](...)

    # 步骤7: GRU更新
    h0 = torch.zeros(1, n_node, hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
    h0 = h0[0, sampled_nodes_idx, :].unsqueeze(0)
    hidden = self.dropout(hidden)
    hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
    hidden = hidden.squeeze(0)
```

### 📊 每层的节点数变化

假设 `n_node_topk = 20`：

```
第0层: 1个节点（姚明）

第1层:
  扩展 → 100个节点
  采样 → 21个节点（1旧 + 20新）

第2层:
  扩展 → 300个节点（21个节点各扩展邻居）
  采样 → 41个节点（21旧 + 20新）

第3层:
  扩展 → 600个节点
  采样 → 61个节点（41旧 + 20新）

第4层:
  扩展 → 900个节点
  采样 → 81个节点（61旧 + 20新）

...

第8层:
  扩展 → 很多节点
  采样 → 161个节点（141旧 + 20新）
```

### 🎨 可视化

```
第0层:           [姚明]
                   ↓
第1层:    [姚明, 休斯顿, 火箭队, ..., 20个节点]
                   ↓
第2层:    [上述21个 + 新的20个] = 41个节点
                   ↓
第3层:    [上述41个 + 新的20个] = 61个节点
                   ↓
                  ...
                   ↓
第8层:    [最终161个节点]
```

### 💡 关键点

1. **增量扩展**：每层只扩展当前节点，不重复扩展旧节点
2. **控制规模**：通过节点采样，每层只增加K个新节点
3. **深度探索**：8层可以探索到距离查询实体8跳的节点
4. **信息积累**：每层通过GRU积累和融合信息

---

## 11. 步骤9：最终预测

### 🎯 目的

根据最终的节点表示，预测每个实体是答案的得分。

### 📥 输入

```python
hidden = [161, 64]  # 第8层后的节点表示
nodes = [161, 2]    # 节点列表 (batch_idx, entity_id)
n = 1               # batchsize
n_ent = 40943       # 总实体数
```

### ⚙️ 操作

```python
# models.py 第182-186行

# 步骤1: 对每个节点计算分数
scores = self.W_final(hidden).squeeze(-1)  # [161]

# 步骤2: 创建完整的分数张量（大部分是0）
scores_all = torch.zeros((n, self.loader.n_ent)).cuda()  # [1, 40943]

# 步骤3: 把有分数的节点填入对应位置
scores_all[[nodes[:,0], nodes[:,1]]] = scores
# nodes[:,0]: batch索引 [161]
# nodes[:,1]: entity索引 [161]
# scores: 分数 [161]
```

### 📤 输出

```python
scores_all = [
    [
        0.00,    # entity_0: 未访问，分数为0
        0.00,    # entity_1: 未访问，分数为0
        ...
        8.92,    # entity_54321 (休斯顿): 最高分！
        ...
        6.34,    # entity_11111 (火箭队)
        ...
        1.05,    # entity_12345 (姚明)
        ...
        0.00,    # entity_40942: 未访问，分数为0
    ]
]  # [1, 40943]
```

### 📊 维度

| 变量 | 维度 | 含义 |
|------|------|------|
| `hidden` | [161, 64] | 最终节点表示 |
| `scores` | [161] | 161个访问节点的分数 |
| `scores_all` | [1, 40943] | 所有实体的分数 |

### 🎨 Top-10预测结果

```
排名  实体          分数    是否是答案
1     休斯顿        8.92    ✅ 正确答案！
2     火箭队        6.34    ✅ 也算对（公司/机构）
3     得克萨斯州    4.78
4     NBA           3.21
5     美国          2.87
6     丰田中心      2.45
7     篮球          1.92
8     中国          1.45
9     麦迪          0.83
10    姚明          1.05
...
40783 其他实体      0.00    (未访问)
```

### 💡 为什么未访问的实体分数是0？

因为它们：
1. 没有在图探索中被访问到
2. 没有表示向量（hidden）
3. 不在最终的候选集中

**这是合理的**：如果一个实体在8层探索中都没有被访问到，它不太可能是答案。

---

## 12. 步骤10：训练与评估

### 🎯 目的

- **训练**：优化模型参数，使正确答案的分数最高
- **评估**：计算模型的性能指标（MRR, Hits@K）

### 📥 输入

```python
scores_all = [1, 40943]  # 预测分数
triple = [[12345, 67, 54321]]  # (姚明, 工作地, 休斯顿)
                                # (sub,  rel,  obj)
```

---

### 🔧 A. 训练模式

#### 操作

```python
# base_model.py 第88-91行

# 步骤1: 提取正确答案的分数
pos_scores = scores_all[[
    torch.arange(len(scores_all)).cuda(),  # batch索引 [1]
    torch.LongTensor(triple[:,2]).cuda()   # 正确答案ID [1]
]]  # [1] -> [8.92] (休斯顿的分数)

# 步骤2: 计算损失（Log-Sum-Exp技巧）
max_n = torch.max(scores_all, 1, keepdim=True)[0]  # [1, 1] -> [8.92]
loss = torch.sum(
    -pos_scores +                                    # -8.92
    max_n +                                          # +8.92
    torch.log(torch.sum(torch.exp(scores_all - max_n), 1))  # log(sum(exp(...)))
)

# 步骤3: 反向传播
loss.backward()

# 步骤4: 更新参数
optimizer.step()
```

#### 损失函数解释

**目标：** 最大化正确答案的分数，最小化其他候选的分数

**数学形式：**
```
loss = -log(exp(score_correct) / sum(exp(score_all)))
     = -score_correct + log(sum(exp(score_all)))
```

**为什么用Log-Sum-Exp技巧？**
- 直接计算 `exp(score)` 可能溢出（如果score很大）
- 减去最大值 `max_n` 保证数值稳定

**例子：**
```
scores_all = [8.92, 6.34, 4.78, 3.21, 0, 0, ...]
pos_score = 8.92 (休斯顿，正确答案)

loss = -8.92 + log(exp(8.92-8.92) + exp(6.34-8.92) + ... )
     = -8.92 + log(exp(0) + exp(-2.58) + exp(-4.14) + ...)
     = -8.92 + log(1 + 0.076 + 0.016 + ...)
     = -8.92 + log(1.092)
     ≈ -8.92 + 0.088
     ≈ -8.83

如果模型完美：
  pos_score >> 其他scores
  loss → 0

如果模型糟糕：
  pos_score ≈ 其他scores
  loss → 很大
```

---

### 🔧 B. 评估模式

#### 操作

```python
# base_model.py 第125-146行

with torch.no_grad():  # 不计算梯度
    for batch in test_data:
        subs, rels, objs = batch[:,0], batch[:,1], batch[:,2]

        # 前向传播
        scores = model(subs, rels, mode='test')

        # Filtered评估：移除已知的其他正确答案
        for i in range(len(subs)):
            # 获取该查询的所有已知答案
            known_answers = all_answers[(subs[i], rels[i])]
            # 除了当前测试答案，其他已知答案的分数设为-∞
            for ans in known_answers:
                if ans != objs[i]:
                    scores[i, ans] = -float('inf')

        # 对分数排序
        sorted_indices = torch.argsort(scores, dim=1, descending=True)

        # 找到正确答案的排名
        for i in range(len(objs)):
            rank = (sorted_indices[i] == objs[i]).nonzero()[0].item() + 1

            # 计算指标
            mrr += 1.0 / rank  # Mean Reciprocal Rank
            if rank <= 1:
                hits1 += 1
            if rank <= 3:
                hits3 += 1
            if rank <= 10:
                hits10 += 1
```

#### 评估指标

**1. MRR (Mean Reciprocal Rank)**

```
MRR = (1/rank_1 + 1/rank_2 + ... + 1/rank_n) / n

例子：
  查询1: 休斯顿排名1 → 1/1 = 1.00
  查询2: 火箭队排名2 → 1/2 = 0.50
  查询3: 上海排名5 → 1/5 = 0.20

  MRR = (1.00 + 0.50 + 0.20) / 3 = 0.567
```

**2. Hits@K**

```
Hits@1: 正确答案在Top-1的比例
Hits@3: 正确答案在Top-3的比例
Hits@10: 正确答案在Top-10的比例

例子（100个查询）：
  50个查询的答案排名1 → Hits@1 = 50/100 = 50%
  70个查询的答案排名≤3 → Hits@3 = 70/100 = 70%
  85个查询的答案排名≤10 → Hits@10 = 85/100 = 85%
```

#### Filtered vs Raw评估

**Raw评估：** 直接对所有实体排序
- 问题：已知的其他正确答案会降低排名

**Filtered评估：** 移除其他已知答案
- 只评估是否能找到当前测试答案
- 更公平

**例子：**

```
查询: (姚明, 工作地, ?)
测试答案: 休斯顿

已知答案: {休斯顿, 火箭队, 丰田中心}

Raw排序:
  1. 休斯顿 ✅
  2. 火箭队 ← 也是正确答案，但在测试时算错
  3. 丰田中心 ← 也是正确答案，但在测试时算错
  4. 其他...

Filtered排序:
  1. 休斯顿 ✅
  2. 其他...
  (火箭队和丰田中心被移除，不参与排名)
```

---

## 13. 步骤11：反向传播详解 ⭐

### 🎯 目的

理解梯度如何从损失函数反向传播到所有模型参数，更新权重以优化模型。

**⭐ 这是理解整个训练过程的关键！**

### 📍 反向传播的触发时机

**触发位置：** `base_model.py` 第96-97行

```python
# 步骤3: 反向传播
loss.backward()  # ⭐ 这里触发反向传播！

# 步骤4: 更新参数
optimizer.step()
```

**什么时候发生？**
- 在每个训练批次的前向传播完成后
- 计算出loss之后立即执行
- 只在训练模式下发生，测试模式不进行反向传播

---

### 🌊 完整的反向传播流程

#### 阶段划分

```
前向传播（Forward Pass）
  ↓
计算损失（Compute Loss）
  ↓
⭐ 反向传播（Backward Pass）← 我们在这里！
  ↓
参数更新（Update Parameters）
```

---

### 🔄 反向传播的逐层梯度流

#### 步骤1：从损失函数开始

**📍 代码：** `base_model.py` 第88-94行

```python
# 计算损失
pos_scores = scores_all[[...]]  # 正确答案的分数 [1] -> 8.92
max_n = torch.max(scores_all, 1, keepdim=True)[0]  # [1, 1] -> 8.92
loss = torch.sum(
    -pos_scores + max_n +
    torch.log(torch.sum(torch.exp(scores_all - max_n), 1))
)  # 标量，例如: loss = 0.088

# ⭐ 触发反向传播
loss.backward()
```

**梯度起点：**
```python
∂loss/∂loss = 1  # 损失对自身的梯度恒为1
```

---

#### 步骤2：传播到最终预测分数

**📍 代码：** `models.py` 第182行

```python
scores = self.W_final(hidden).squeeze(-1)  # [161]
```

**梯度计算：**
```python
# 链式法则
∂loss/∂scores = ∂loss/∂scores_all * ∂scores_all/∂scores

# 对于正确答案（休斯顿）：
∂loss/∂score_houston = -1 + exp(score_houston - max) / sum(exp(...))
                       ≈ -1 + exp(0) / 1.092
                       ≈ -1 + 0.916
                       ≈ -0.084  # 负梯度，说明要增大这个分数

# 对于其他候选（火箭队）：
∂loss/∂score_rockets = 0 + exp(score_rockets - max) / sum(exp(...))
                      ≈ exp(-2.58) / 1.092
                      ≈ 0.076 / 1.092
                      ≈ 0.070  # 正梯度，说明要减小这个分数
```

**反向传播到W_final：**
```python
∂loss/∂W_final = ∂loss/∂scores * ∂scores/∂W_final
               = ∂loss/∂scores * hidden.T  # [64, 161] @ [161] -> [64]
```

---

#### 步骤3：传播到节点表示（第L层）

**📍 代码：** 从最后一层的hidden反向传播

```python
# models.py 第177-178行
hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
hidden = hidden.squeeze(0)  # [161, 64]
```

**梯度计算：**
```python
∂loss/∂hidden_L = ∂loss/∂scores * ∂scores/∂hidden_L
                = ∂loss/∂scores * W_final.T  # [161] @ [64] -> [161, 64]

# 例子（休斯顿节点）：
∂loss/∂hidden_houston = [-0.084 * W_final[0], -0.084 * W_final[1], ...]
                       = [-0.0042, 0.0035, -0.0021, ...]  # 64维梯度
```

---

#### 步骤4：传播通过GRU

**📍 代码：** `models.py` 第706-709行

```python
hidden, h0 = self.gate(
    hidden.unsqueeze(0),  # 当前输入
    h0                     # 历史状态
)
```

**GRU反向传播：**

GRU有复杂的内部结构，梯度需要分流到多个部分：

```python
# 梯度分流到：
1. ∂loss/∂hidden_input  # 传播到当前层输入
2. ∂loss/∂h0            # 传播到历史状态
3. ∂loss/∂W_r, ∂loss/∂W_z, ∂loss/∂W  # 传播到GRU权重
```

**数学表达（简化）：**
```python
# GRU前向：
r_t = σ(W_r * [h_{t-1}, x_t])
z_t = σ(W_z * [h_{t-1}, x_t])
h̃_t = tanh(W * [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

# GRU反向：
∂loss/∂x_t = ... (复杂的链式法则)
∂loss/∂h_{t-1} = ∂loss/∂h_t * (1 - z_t) + ...  # 梯度传递到上一层
```

---

#### 步骤5：传播通过节点采样 ⭐

**📍 代码：** `models.py` 第107-109行

```python
hidden_new[bool_diff_node_idx] *= (
    diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob
)
```

**⭐ Straight-through Estimator的反向传播：**

这是关键！前面我们讲过，这里使用了软硬采样结合。

**前向传播（复习）：**
```python
output = diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob
       = diff_node_prob_hard  # 前向使用硬采样
```

**反向传播：**
```python
∂output/∂diff_node_prob = ∂(diff_node_prob_hard - diff_node_prob.detach() + diff_node_prob)/∂diff_node_prob
                        = 0 - 0 + 1  # detach()切断梯度，hard是常数
                        = 1

∂loss/∂diff_node_prob = ∂loss/∂output * 1  # 梯度完整传递！
```

**具体例子：**
```python
# 假设休斯顿节点的梯度
∂loss/∂hidden_houston = [-0.0042, 0.0035, ...]  # [64]

# Straight-through反向传播
∂loss/∂diff_node_prob_houston = ∂loss/∂hidden_houston * hidden_houston
                                # 标量梯度

# 梯度继续传播到W_samp
∂loss/∂W_samp = ∂loss/∂diff_node_prob * ∂diff_node_prob/∂node_scores * ∂node_scores/∂W_samp
```

**为什么所有节点都能收到梯度？**
```python
# 即使某个节点被硬采样丢弃（hard = 0），
# 它的软概率 diff_node_prob 仍然参与梯度计算

# 未选中节点（例如某个节点）：
diff_node_prob_hard = 0  # 前向被丢弃
diff_node_prob = 0.02    # 但梯度仍然传递
∂loss/∂diff_node_prob = ...  # 能收到梯度！

# 这让模型学习：下次应该给这个节点更高/更低的分数
```

---

#### 步骤6：传播通过Gumbel Softmax ⭐

**📍 代码：** `models.py` 第95行（通过train方法设置）

```python
node_scores = self.softmax(node_scores)  # Gumbel Softmax
```

**Gumbel Softmax的反向传播：**

```python
# 前向（复习）：
node_scores = softmax((node_logit + gumbel_noise) / tau)

# 反向传播：
∂loss/∂node_logit = ∂loss/∂node_scores * ∂node_scores/∂node_logit
```

**Softmax的梯度：**
```python
# Softmax的雅可比矩阵（Jacobian）
∂softmax_i/∂logit_j = {
    softmax_i * (1 - softmax_i)  if i == j  # 对角元素
    -softmax_i * softmax_j       if i ≠ j  # 非对角元素
}

# 因此梯度会在所有logit之间分布
∂loss/∂node_logit = node_scores ⊙ (∂loss/∂node_scores - <∂loss/∂node_scores, node_scores>)
```

**关键insight：**
```python
# 即使某个节点的node_score很低（比如0.02），
# 它仍然能收到梯度反馈

# 例子：
node_scores = [0.25, 0.23, ..., 0.02, 0.01, ...]
∂loss/∂node_scores = [...]

# Softmax反向传播后：
∂loss/∂node_logit = [...]  # 所有logit都有梯度！

# 这使得Gumbel Softmax可微分 ✅
```

---

#### 步骤7：传播到采样权重W_samp

**📍 代码：** `models.py` 第86行

```python
diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1)
```

**梯度计算：**
```python
∂loss/∂W_samp = ∂loss/∂node_logit * ∂node_logit/∂W_samp
              = ∂loss/∂node_logit * hidden_new.T  # [99, 64].T @ [99] -> [64, 1]

# W_samp会被更新：
W_samp_new = W_samp - learning_rate * ∂loss/∂W_samp
```

---

#### 步骤8：传播到消息传播

**📍 代码：** `models.py` 第69-71行

```python
message = alpha * message  # [350, 64]
message_agg = scatter(message, index=obj, ...)  # [100, 64]
hidden_new = self.act(self.W_h(message_agg))  # [100, 64]
```

**反向传播顺序（逆向）：**

```python
# 1. 通过W_h反向传播
∂loss/∂message_agg = ∂loss/∂hidden_new * ∂hidden_new/∂message_agg
                    = ∂loss/∂hidden_new * act'() * W_h.T

# 2. 通过scatter反向传播（gather操作）
∂loss/∂message = gather(∂loss/∂message_agg, index=obj)  # [350, 64]

# 3. 通过alpha反向传播
∂loss/∂alpha = ∂loss/∂message * message / alpha  # [350, 1]
∂loss/∂message_原始 = ∂loss/∂message * alpha
```

---

#### 步骤9：传播通过边采样 ⭐

**📍 代码：** `models.py` 第56-63行（边采样的Straight-through）

```python
alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
```

**反向传播（类似节点采样）：**
```python
∂alpha_out/∂edge_prob = 1  # Straight-through的魔法

∂loss/∂edge_prob = ∂loss/∂alpha_out * 1

# 然后通过Gumbel Softmax反向传播
∂loss/∂alpha_原始 = ∂loss/∂edge_prob * ∂edge_prob/∂alpha_原始
```

**所有边都能收到梯度：**
```python
# 即使某条边被硬采样丢弃
edge_prob_hard[i] = 0  # 前向被丢弃
edge_prob[i] = 0.08    # 但仍有软概率
∂loss/∂edge_prob[i] = ...  # 能收到梯度！✅
```

---

#### 步骤10：传播到注意力机制

**📍 代码：** `models.py` 第57行和66行

```python
alpha = self.w_alpha(nn.ReLU()(
    self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)
))
alpha = torch.sigmoid(alpha)
```

**反向传播：**
```python
# 1. 通过sigmoid反向传播
∂loss/∂alpha_logit = ∂loss/∂alpha * alpha * (1 - alpha)

# 2. 通过w_alpha反向传播
∂loss/∂attn_sum = ∂loss/∂alpha_logit * w_alpha.T

# 3. 通过ReLU反向传播（ReLU'(x) = 1 if x>0 else 0）
∂loss/∂attn_sum = ∂loss/∂attn_sum * (attn_sum > 0)

# 4. 分流到三个分支
∂loss/∂Ws_attn(hs) = ∂loss/∂attn_sum
∂loss/∂Wr_attn(hr) = ∂loss/∂attn_sum
∂loss/∂Wqr_attn(h_qr) = ∂loss/∂attn_sum

# 5. 更新注意力权重
∂loss/∂Ws_attn = ∂loss/∂Ws_attn(hs) * hs.T
∂loss/∂Wr_attn = ∂loss/∂Wr_attn(hr) * hr.T
∂loss/∂Wqr_attn = ∂loss/∂Wqr_attn(h_qr) * h_qr.T
```

---

#### 步骤11：传播到嵌入层

**📍 代码：** `models.py` 第49行

```python
hr = self.rela_embed(rel)  # 关系嵌入 [350, 64]
```

**反向传播：**
```python
∂loss/∂rela_embed[rel] = ∂loss/∂hr

# 嵌入层的梯度更新（稀疏更新）
# 只有被使用的关系ID会收到梯度
rela_embed[rel] -= learning_rate * ∂loss/∂rela_embed[rel]
```

---

### 📊 完整的反向传播路径图

```
Loss (标量)
  ↓
∂loss/∂scores_all [1, 40943]
  ↓
∂loss/∂scores [161]
  ↓
∂loss/∂W_final [64, 1]
  ↓
∂loss/∂hidden_L [161, 64]
  ↓
┌───────────────────────────────────────┐
│ 第L层反向传播                          │
│                                       │
│  ∂loss/∂hidden → [GRU] → ∂loss/∂hidden_input │
│         │                      ↓              │
│         └─→ [Dropout] → [W_h] → [scatter] → [alpha * message]  │
│                                              │
│  ⭐ 节点采样的Straight-through:              │
│  ∂loss/∂diff_node_prob (软概率，可微分)     │
│         ↓                                    │
│  [Gumbel Softmax反向] → ∂loss/∂node_logit   │
│         ↓                                    │
│  ∂loss/∂W_samp [64, 1]                      │
│                                              │
│  ⭐ 边采样的Straight-through:                │
│  ∂loss/∂edge_prob (软概率，可微分)          │
│         ↓                                    │
│  [Gumbel Softmax反向] → ∂loss/∂alpha_原始   │
│         ↓                                    │
│  [注意力机制反向]                            │
│         ↓                                    │
│  ∂loss/∂Ws_attn, ∂loss/∂Wr_attn, ∂loss/∂Wqr_attn  │
│         ↓                                    │
│  ∂loss/∂rela_embed (关系嵌入)               │
│         ↓                                    │
│  ∂loss/∂hidden_{L-1} (传播到上一层)         │
└───────────────────────────────────────┘
  ↓
第L-1层、第L-2层...第1层
(重复上述过程)
  ↓
所有参数的梯度都计算完成！
```

---

### 🔄 参数更新

**📍 代码：** `base_model.py` 第97行

```python
optimizer.step()  # 更新所有参数
```

**更新公式（SGD为例）：**
```python
for param in model.parameters():
    param.data -= learning_rate * param.grad

# 具体例子：
W_final -= learning_rate * ∂loss/∂W_final
W_samp -= learning_rate * ∂loss/∂W_samp
Ws_attn -= learning_rate * ∂loss/∂Ws_attn
rela_embed -= learning_rate * ∂loss/∂rela_embed
...
```

**Adam优化器（实际使用）：**
```python
# Adam维护动量和二阶矩估计
m_t = β1 * m_{t-1} + (1 - β1) * grad
v_t = β2 * v_{t-1} + (1 - β2) * grad^2
param -= learning_rate * m_t / (sqrt(v_t) + ε)
```

---

### 🌟 关键技术点回顾

#### 1. 为什么Gumbel Softmax是可微分的？

```python
# 普通Top-K（不可微分）：
topk = hard_select(scores)  # 离散操作，∂topk/∂scores = 0 ❌

# Gumbel Softmax（可微分）：
soft_prob = softmax((scores + gumbel_noise) / tau)
# ∂soft_prob/∂scores ≠ 0  ✅ 可以训练！
```

#### 2. 为什么Straight-through Estimator有效？

```python
# 前向：使用硬选择（高效）
output_forward = hard_selection  # 0或1

# 反向：梯度传给软概率（可学习）
∂loss/∂soft_prob = ∂loss/∂output * 1  # 梯度流畅

# 结果：鱼和熊掌兼得！
- 前向推理高效 ✅
- 反向训练可微 ✅
```

#### 3. 采样中的梯度分配

```python
# 所有候选都能收到梯度反馈，不只是被选中的

# 被选中的节点：
∂loss/∂node_i = ...  # 较大的梯度，鼓励继续好的表现

# 未被选中的节点：
∂loss/∂node_j = ...  # 较小但非零的梯度，指导改进

# 这样模型可以学习：
- 哪些节点应该得分更高（被选中）
- 哪些节点应该得分更低（被淘汰）
```

---

### 💡 形象理解：反向传播就像反馈循环

```
学生考试类比：

前向传播 = 考试答题
  学生: 姚明在哪里工作？
  答案: 休斯顿 (score=8.92)

计算损失 = 批改试卷
  正确答案: 休斯顿
  得分: -0.088 (接近0，说明答对了)

⭐ 反向传播 = 老师反馈
  "你选了休斯顿，正确！(∂loss = -0.084)"
  "但还需要把休斯顿的分数再提高一点"
  "火箭队的分数太高了 (∂loss = +0.070)，要降低"

  反馈逐层传递：
  - W_final: "你最后的判断权重需要调整"
  - GRU: "你融合信息的方式可以改进"
  - 采样: "下次选择更好的节点"
  - 注意力: "关注更相关的边"
  - 嵌入: "关系的表示需要优化"

参数更新 = 学生改进
  下次遇到类似问题，表现会更好！
```

---

### 📊 数值例子：完整的梯度流

假设简化的例子：

```python
# 前向传播结果
scores_all = [8.92, 6.34, 4.78, 0, 0, ...]  # 休斯顿=8.92是正确答案
loss = 0.088

# 步骤1: 损失对scores的梯度
∂loss/∂scores = [
    -0.084,   # 休斯顿（正确答案，负梯度→增大分数）
    +0.070,   # 火箭队（错误答案，正梯度→减小分数）
    +0.012,   # 得克萨斯州
    +0.001,   # 其他
    +0.001,   # 其他
    ...
]

# 步骤2: scores对hidden的梯度
∂scores/∂hidden = W_final.T  # [161, 64]
∂loss/∂hidden = ∂loss/∂scores @ W_final.T
              = [-0.084, 0.070, ...] @ W_final.T
              # 每个节点的64维梯度

# 步骤3: hidden对W_final的梯度
∂loss/∂W_final = hidden.T @ ∂loss/∂scores
                = [161, 64].T @ [161]
                = [64, 1]
# W_final将被更新

# 步骤4: 通过GRU反向传播
∂loss/∂hidden_before_GRU = GRU.backward(∂loss/∂hidden)

# 步骤5: 通过节点采样反向传播（Straight-through）
∂loss/∂diff_node_prob = ∂loss/∂hidden_before_GRU * ...
# 所有节点（包括未选中的）都收到梯度

# 步骤6: 通过Gumbel Softmax反向传播
∂loss/∂node_logit = Gumbel_Softmax.backward(∂loss/∂diff_node_prob)

# 步骤7: node_logit对W_samp的梯度
∂loss/∂W_samp = hidden.T @ ∂loss/∂node_logit
# W_samp将被更新

# ... 继续向前传播到所有参数
```

---

### 🔑 总结：反向传播的关键点

1. **触发时机**：每个训练批次的前向传播后，`loss.backward()`
2. **梯度流向**：从loss → scores → hidden → 采样 → 消息传播 → 注意力 → 嵌入
3. **Gumbel Softmax的作用**：让离散的采样操作变得可微分
4. **Straight-through的魔法**：前向用硬采样（高效），反向用软采样（可学习）
5. **所有参数都更新**：W_final, W_samp, W_h, Ws_attn, Wr_attn, Wqr_attn, w_alpha, rela_embed, GRU权重
6. **梯度分配**：即使未被选中的边/节点，也能收到梯度反馈，指导模型改进

---

### ⚠️ 常见问题

**Q1: 为什么未被选中的节点还能收到梯度？**
A: 因为Straight-through Estimator！前向虽然被丢弃（hard=0），但反向梯度传给了软概率（soft_prob），所有节点的soft_prob都能收到梯度。

**Q2: Gumbel噪声在反向传播时怎么处理？**
A: Gumbel噪声是前向采样时加的，反向传播时梯度仍然通过softmax传递，不需要特殊处理噪声。

**Q3: 多层GNN如何反向传播？**
A: 从最后一层开始，逐层反向。每层的梯度都会传播到上一层的hidden，形成梯度流。

**Q4: 为什么需要Log-Sum-Exp技巧？**
A: 防止数值溢出。直接计算exp(score)可能溢出，减去max保证数值稳定，但梯度计算不受影响。

---

## 14. 完整示例

### 🎯 查询："姚明在哪里工作？"

让我们完整地走一遍流程，包括所有数据维度。

---

### 📥 输入

```python
sub = 12345  # 姚明
rel = 67     # 工作地
obj = ?      # 要预测
```

---

### 🔄 步骤0-1：初始化

```python
nodes = [[0, 12345]]         # [1, 2]
hidden = [[0, 0, ..., 0]]    # [1, 64]
h0 = [[[0, 0, ..., 0]]]      # [1, 1, 64]
```

---

### 🔄 第1层

**步骤2: 图扩展**
```python
nodes: [1, 2] → [100, 2]
edges: [350, 6]
old_nodes_new_idx: [1] (姚明的索引)
```

**步骤3: 注意力打分**
```python
alpha: [350, 1]
例: [0.92, 0.78, 0.45, 0.23, 0.08, ...]
```

**步骤4: 边采样**
```python
保留Top-200条边
alpha: [350, 1] (但只有200个非零)
```

**步骤5: 消息传播**
```python
hidden_new: [100, 64]
例: 休斯顿 = [0.83, -0.45, 0.62, ...]
```

**步骤6: 节点采样**
```python
nodes: [100, 2] → [21, 2] (1旧 + 20新)
hidden: [100, 64] → [21, 64]
```

**步骤7: GRU更新**
```python
hidden: [21, 64] (融合历史信息)
h0: [1, 21, 64] (更新状态)
```

---

### 🔄 第2层

从21个节点继续扩展...

```python
输入 nodes: [21, 2]
输出 nodes: [41, 2] (21旧 + 20新)
输出 hidden: [41, 64]
```

---

### 🔄 第3-8层

重复上述过程...

```python
第3层: nodes [61, 2], hidden [61, 64]
第4层: nodes [81, 2], hidden [81, 64]
第5层: nodes [101, 2], hidden [101, 64]
第6层: nodes [121, 2], hidden [121, 64]
第7层: nodes [141, 2], hidden [141, 64]
第8层: nodes [161, 2], hidden [161, 64]
```

---

### 🎯 步骤9: 最终预测

```python
scores = W_final(hidden)  # [161]

scores_all: [1, 40943]
例:
  scores_all[0, 54321] = 8.92  # 休斯顿
  scores_all[0, 11111] = 6.34  # 火箭队
  scores_all[0, 33333] = 4.78  # 得克萨斯州
  ...
  scores_all[0, 其他] = 0.00
```

**排序后的Top-10：**
```
1. 休斯顿 (8.92) ← 答案！
2. 火箭队 (6.34)
3. 得克萨斯州 (4.78)
4. NBA (3.21)
5. 美国 (2.87)
6. 丰田中心 (2.45)
7. 篮球 (1.92)
8. 中国 (1.45)
9. 姚明 (1.05)
10. 麦迪 (0.83)
```

---

### 📊 训练

```python
pos_score = 8.92  # 休斯顿的分数
loss = -8.92 + log(sum(exp(scores_all - 8.92)))
     ≈ -8.83

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

### 📊 评估

```python
正确答案: 休斯顿
预测排名: 1

MRR = 1/1 = 1.00
Hits@1 = 1 (正确！)
Hits@3 = 1 (正确！)
Hits@10 = 1 (正确！)
```

---

## 14. 数据流转总览

### 📊 完整的维度变化表

| 阶段 | 变量 | 维度 | 说明 |
|-----|------|------|------|
| **输入** | `subs, rels` | [1], [1] | 查询实体和关系 |
| **初始化** | `nodes` | [1, 2] | 初始节点（姚明） |
| | `hidden` | [1, 64] | 初始表示（全0） |
| | `h0` | [1, 1, 64] | GRU初始状态 |
| **第1层** | | | |
| 图扩展 | `nodes` | [100, 2] | 扩展到100个节点 |
| | `edges` | [350, 6] | 350条边 |
| 注意力 | `alpha` | [350, 1] | 每条边的重要性 |
| 边采样 | `alpha` | [350, 1] | 只有200个非零 |
| 消息传播 | `hidden_new` | [100, 64] | 更新节点表示 |
| 节点采样 | `nodes` | [21, 2] | 保留21个节点 |
| | `hidden` | [21, 64] | 对应表示 |
| GRU更新 | `hidden` | [21, 64] | 融合后的表示 |
| | `h0` | [1, 21, 64] | 更新后的状态 |
| **第2层** | `nodes` | [41, 2] | 21旧 + 20新 |
| | `hidden` | [41, 64] | |
| **...** | | | |
| **第8层** | `nodes` | [161, 2] | 最终节点 |
| | `hidden` | [161, 64] | 最终表示 |
| **最终预测** | `scores` | [161] | 161个节点的分数 |
| | `scores_all` | [1, 40943] | 所有实体的分数 |
| **输出** | `answer` | 标量 | 休斯顿 (entity_54321) |

---

### 🎨 可视化数据流

```
输入: (姚明, 工作地, ?)
  ↓
[1个节点] → 初始化 → hidden:[1,64]
  ↓
第1层 GNN:
  [1,2] --图扩展--> [100,2]
  [100,64] --消息传播--> [100,64]
  [100,64] --节点采样--> [21,64]
  [21,64] --GRU--> [21,64]
  ↓
第2层 GNN:
  [21,2] --图扩展--> [300,2]
  [300,64] --消息传播--> [300,64]
  [300,64] --节点采样--> [41,64]
  [41,64] --GRU--> [41,64]
  ↓
... (第3-8层)
  ↓
第8层后:
  [161,2], [161,64]
  ↓
最终预测:
  [161,64] --W_final--> [161]
  [161] --填充--> [1, 40943]
  ↓
排序:
  1. 休斯顿 (8.92)
  2. 火箭队 (6.34)
  ...
  ↓
输出: 休斯顿 ✅
```

---

## 📚 总结

### 🎯 核心流程（10步）

1. **输入准备**：查询 (sub, rel, ?)
2. **初始化**：创建初始节点和表示
3. **图扩展**：获取邻居节点和边
4. **注意力打分**：计算边的重要性
5. **边采样**：选择Top-K条边（可选）
6. **消息传播**：通过边传递信息
7. **节点采样**：选择Top-K个节点
8. **GRU更新**：融合历史信息
9. **多层迭代**：重复步骤3-8，L次
10. **最终预测**：计算每个实体的得分

### 🔑 关键技术

1. **增量采样**：每层只增加K个新节点，控制计算量
2. **双重采样**：边采样 + 节点采样
3. **注意力机制**：query-aware，考虑查询关系
4. **GRU门控**：融合多层信息
5. **Gumbel Softmax**：可微分采样
6. **Straight-through Estimator**：前向硬，反向软

### 📊 数据维度规律

- **节点数**：每层增加K个，最终约 L*K+1 个
- **边数**：取决于图结构，通常数百到数千
- **表示维度**：始终是64维（hidden_dim）

### 💡 设计哲学

1. **效率优先**：通过采样控制计算量
2. **质量保证**：通过注意力保证信息质量
3. **平衡探索**：深度（8层）+ 广度（每层K个新节点）
4. **端到端**：可微分训练，直接优化答案排名

---

## 🔗 相关文档

- `AdaProp_详细流程说明.md` - 更详细的技术解释
- `AdaProp打分机制详解_小白版.md` - 四个打分环节详解
- `核心概念详解_小白版.md` - 基础概念
- `CLAUDE.md` - 代码使用指南

---

**希望这份文档帮助你理解AdaProp模型的完整流程！** 🎉
