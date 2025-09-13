# AdaProp模型详细分析

## 目录
- [1. 模型概述](#1-模型概述)
- [2. 整体架构](#2-整体架构)
- [3. 自适应传播机制](#3-自适应传播机制)
- [4. Transductive vs Inductive设置](#4-transductive-vs-inductive设置)
- [5. 采样策略详解](#5-采样策略详解)
- [6. 训练过程](#6-训练过程)
- [7. 评估机制](#7-评估机制)
- [8. 关键创新点](#8-关键创新点)
- [9. 代码实现细节](#9-代码实现细节)
- [10. 使用方法](#10-使用方法)

## 1. 模型概述

AdaProp（Adaptive Propagation）是一个基于图神经网络的知识图谱推理模型，发表在KDD 2023上。该模型的核心创新是**学习自适应传播路径**，解决了传统GNN方法中实体数量爆炸性增长的问题。

### 1.1 问题背景
- 传统GNN方法使用固定的传播路径，忽略了实体与查询关系的相关性
- 在大规模传播步骤中，涉及的实体数量会爆炸性增长
- 需要一种机制来过滤不相关实体，同时保留有希望的目标实体

### 1.2 核心思想
- **增量采样机制**：保持线性复杂度，同时保留附近目标和层级连接
- **学习式采样分布**：识别语义相关的实体
- **查询感知传播**：根据具体查询关系调整传播过程

## 2. 整体架构

### 2.1 模型组件层次结构

```
AdaProp
├── GNNModel (主模型)
│   ├── GNNLayer × n_layer (图神经网络层)
│   │   ├── 关系嵌入层 (rela_embed)
│   │   ├── 注意力机制 (Ws_attn, Wr_attn, Wqr_attn, w_alpha)
│   │   ├── 消息传递 (W_h)
│   │   └── 采样网络 (W_samp)
│   ├── GRU门控单元 (gate)
│   └── 最终输出层 (W_final)
└── BaseModel (训练封装)
    ├── 优化器 (Adam)
    ├── 学习率调度器 (ExponentialLR)
    └── 评估函数
```

### 2.2 数据流程

```
输入查询 (subject, relation) 
    ↓
初始化节点集合 {query_subject}
    ↓
第1层传播：获取邻居 → 消息传递 → 注意力加权 → 节点采样
    ↓
第2层传播：获取邻居 → 消息传递 → 注意力加权 → 节点采样
    ↓
...
    ↓
第n层传播：获取邻居 → 消息传递 → 注意力加权
    ↓
最终预测：计算所有候选实体的分数
```

## 3. 自适应传播机制

### 3.1 查询感知的注意力机制

AdaProp的核心创新之一是查询感知的注意力机制：

```python
# 构建消息组件
hs = hidden[sub]                    # 源节点特征 [n_edges, hidden_dim]
hr = self.rela_embed(rel)          # 关系嵌入 [n_edges, hidden_dim]
h_qr = self.rela_embed(q_rel)[r_idx]  # 查询关系嵌入 [n_edges, hidden_dim]

# 计算注意力权重
alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(
    self.Ws_attn(hs) +      # 源节点注意力
    self.Wr_attn(hr) +      # 关系注意力
    self.Wqr_attn(h_qr)     # 查询关系注意力
)))

# 加权消息聚合
message = alpha * (hs + hr)
message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
```

**关键特点**：
- **三元注意力**：同时考虑源节点、边关系、查询关系
- **查询相关性**：注意力权重根据具体查询任务动态调整
- **可学习权重**：所有注意力组件都是可学习的线性变换

### 3.2 消息传递过程

```python
def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize):
    # 1. 提取边信息
    sub = edges[:,4]  # 源节点索引
    rel = edges[:,2]  # 关系索引  
    obj = edges[:,5]  # 目标节点索引
    
    # 2. 构建消息
    hs = hidden[sub]
    hr = self.rela_embed(rel)
    message = hs + hr  # 基础消息：节点特征 + 关系特征
    
    # 3. 计算注意力并加权
    alpha = self.compute_attention(hs, hr, q_rel, edges)
    message = alpha * message
    
    # 4. 聚合消息到目标节点
    message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
    
    # 5. 更新节点表示
    hidden_new = self.act(self.W_h(message_agg))
    
    return hidden_new
```

## 4. Transductive vs Inductive设置

### 4.1 设置对比

| 特征 | Transductive | Inductive |
|------|-------------|-----------|
| **测试实体** | 训练时见过 | 训练时未见过 |
| **实体空间** | 单一实体空间 | 训练/测试分离空间 |
| **采样复杂度** | 节点+边双重采样 | 简化的Top-K采样 |
| **模型复杂度** | 更复杂的GNNLayer | 相对简化的结构 |
| **应用场景** | 已知KG上的推理 | 新实体的零样本推理 |

### 4.2 Transductive实现

```python
# transductive/models.py - 复杂的双重采样
class GNNLayer(torch.nn.Module):
    def __init__(self, ..., n_node_topk=-1, n_edge_topk=-1, ...):
        # 支持节点和边的采样参数
        
    def forward(self, ...):
        # 边采样
        if self.n_edge_topk > 0:
            alpha = self.w_alpha(...)
            edge_prob = F.gumbel_softmax(alpha, tau=1, hard=False)
            topk_index = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]
            
        # 节点采样
        if self.n_node_topk > 0:
            diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx])
            node_scores = self.softmax(node_scores)
            topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices
```

### 4.3 Inductive实现

```python
# inductive/models.py - 简化的采样机制
class GNNLayer(torch.nn.Module):
    def forward(self, ...):
        # 只有基础的消息传递，没有复杂采样
        message = mess2 * alpha_2
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        
# 在GNNModel中实现soft_to_hard采样
def soft_to_hard(self, i, hidden, nodes, n_ent, batch_size, old_nodes_new_idx):
    # 简化的Top-K选择
    _, argtopk = torch.topk(soft_all, k=self.topk, dim=-1)
```

### 4.4 数据处理差异

```python
# Inductive数据加载器需要处理两个实体空间
class DataLoader:
    def __init__(self, task_dir, n_batch=32):
        # 训练实体空间
        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            
        # 测试实体空间  
        with open(os.path.join(self.ind_dir, 'entities.txt')) as f:
            self.entity2id_ind = dict()
            
        self.n_ent = len(self.entity2id)      # 训练实体数
        self.n_ent_ind = len(self.entity2id_ind)  # 测试实体数
```

## 5. 采样策略详解

### 5.1 增量采样机制

AdaProp的核心创新是增量采样，它解决了传统GNN中节点数量指数增长的问题：

```python
def incremental_sampling_process():
    """
    增量采样的完整流程
    """
    # 第0层：初始查询节点
    nodes_0 = [query_subject]  # [batch_size, 2] (batch_idx, entity_id)
    
    for layer in range(n_layers):
        # 1. 获取当前层的所有邻居
        all_neighbors = get_neighbors(nodes_i)  # 可能有大量邻居
        
        # 2. 区分新旧节点
        old_nodes = nodes_i  # 上一层保留的节点
        new_nodes = all_neighbors - old_nodes  # 新发现的邻居节点
        
        # 3. 对新节点进行采样
        if layer < n_layers - 1:  # 非最后一层需要采样
            sampled_new_nodes = adaptive_sampling(new_nodes, query_relation)
            nodes_i+1 = old_nodes + sampled_new_nodes
        else:  # 最后一层保留所有节点用于预测
            nodes_i+1 = old_nodes + new_nodes
```

### 5.2 自适应节点采样

```python
def adaptive_node_sampling(self, hidden, nodes, old_nodes_new_idx):
    """
    基于学习的节点采样机制
    """
    n_node = nodes.size(0)
    
    # 1. 区分新旧节点
    bool_diff_node_idx = torch.ones(n_node).bool()
    bool_diff_node_idx[old_nodes_new_idx] = False  # 旧节点标记为False
    diff_nodes = nodes[bool_diff_node_idx]  # 新节点
    
    # 2. 计算新节点的采样分数
    diff_node_logits = self.W_samp(hidden[bool_diff_node_idx]).squeeze(-1)
    
    # 3. 将分数映射到固定大小的张量
    node_scores = torch.ones((batch_size, n_ent)) * float('-inf')
    node_scores[diff_nodes[:,0], diff_nodes[:,1]] = diff_node_logits
    
    # 4. 使用Gumbel Softmax进行可微分采样
    if self.training:
        node_scores = F.gumbel_softmax(node_scores, tau=self.tau, hard=False)
    else:
        node_scores = F.softmax(node_scores, dim=-1)
    
    # 5. Top-K选择
    _, topk_indices = torch.topk(node_scores, k=self.n_node_topk, dim=-1)
    
    # 6. 创建硬采样掩码
    batch_topk_nodes = torch.zeros((batch_size, n_ent))
    topk_batch_idx = torch.arange(batch_size).repeat(self.n_node_topk, 1).T.reshape(-1)
    batch_topk_nodes[topk_batch_idx, topk_indices.reshape(-1)] = 1
    
    # 7. 直通估计器：前向传播用硬采样，反向传播用软采样
    diff_node_prob_hard = batch_topk_nodes[diff_nodes[:,0], diff_nodes[:,1]]
    diff_node_prob_soft = node_scores[diff_nodes[:,0], diff_nodes[:,1]]
    
    # 更新节点嵌入
    hidden[bool_diff_node_idx] *= (
        diff_node_prob_hard - diff_node_prob_soft.detach() + diff_node_prob_soft
    ).unsqueeze(-1)
    
    # 8. 返回采样后的节点和嵌入
    sampled_mask = batch_topk_nodes[diff_nodes[:,0], diff_nodes[:,1]].bool()
    final_mask = ~bool_diff_node_idx  # 保留旧节点
    final_mask[bool_diff_node_idx] = sampled_mask  # 添加采样的新节点
    
    return hidden[final_mask], nodes[final_mask]
```

### 5.3 边采样机制（仅Transductive）

```python
def adaptive_edge_sampling(self, edges, hs, hr, h_qr):
    """
    基于注意力的边采样机制
    """
    if self.n_edge_topk <= 0:
        return edges, torch.ones(edges.size(0))  # 不采样，返回所有边
    
    # 1. 计算边的重要性分数
    edge_attention = self.w_alpha(nn.ReLU()(
        self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)
    )).squeeze(-1)
    
    # 2. Gumbel Softmax采样
    edge_prob_soft = F.gumbel_softmax(edge_attention, tau=1, hard=False)
    
    # 3. Top-K选择
    _, topk_indices = torch.topk(edge_prob_soft, k=self.n_edge_topk)
    
    # 4. 创建硬采样掩码
    edge_prob_hard = torch.zeros_like(edge_attention)
    edge_prob_hard[topk_indices] = 1
    
    # 5. 直通估计器
    edge_weights = edge_prob_hard - edge_prob_soft.detach() + edge_prob_soft
    
    return edges[topk_indices], edge_weights[topk_indices]
```

### 5.4 采样策略的优势

1. **线性复杂度**：相比于指数级的邻居扩展，采样保持了线性复杂度
2. **语义感知**：采样分数基于节点的语义表示，能够选择相关实体
3. **可微分性**：使用Gumbel Softmax和直通估计器实现端到端训练
4. **查询相关**：采样过程间接考虑了查询关系的信息

## 6. 训练过程

### 6.1 训练流程

```python
def training_pipeline():
    """
    完整的训练流程
    """
    for epoch in range(max_epochs):
        epoch_loss = 0
        model.train()
        
        # 1. 批次训练
        for batch_idx in range(n_batches):
            # 1.1 获取训练批次
            triples = loader.get_batch(batch_idx)  # [batch_size, 3] (h,r,t)
            
            # 1.2 前向传播
            scores = model(triples[:,0], triples[:,1])  # [batch_size, n_entities]
            
            # 1.3 提取正样本分数
            pos_scores = scores[torch.arange(len(scores)), triples[:,2]]
            
            # 1.4 计算损失（数值稳定版本）
            max_scores = torch.max(scores, dim=1, keepdim=True)[0]
            log_sum_exp = torch.log(torch.sum(torch.exp(scores - max_scores), dim=1))
            loss = torch.sum(-pos_scores + max_scores.squeeze() + log_sum_exp)
            
            # 1.5 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 1.6 数值稳定性处理
            handle_nan_parameters()
            
            epoch_loss += loss.item()
        
        # 2. 学习率调度
        scheduler.step()
        
        # 3. 验证评估
        if epoch % eval_interval == 0:
            val_results = evaluate_on_validation()
            test_results = evaluate_on_test()
            
            # 4. 早停和模型保存
            if val_results['mrr'] > best_val_mrr:
                best_val_mrr = val_results['mrr']
                save_model_checkpoint()
```

### 6.2 损失函数设计

AdaProp使用负对数似然损失，但采用了数值稳定的实现：

```python
def compute_loss(scores, targets):
    """
    数值稳定的负对数似然损失
    
    标准公式: -log(exp(s_pos) / sum(exp(s_i)))
    数值稳定版本: -s_pos + log(sum(exp(s_i - s_max))) + s_max
    """
    # 1. 提取正样本分数
    batch_size = scores.size(0)
    pos_scores = scores[torch.arange(batch_size), targets]
    
    # 2. 数值稳定的log-sum-exp计算
    max_scores = torch.max(scores, dim=1, keepdim=True)[0]  # 防止溢出
    shifted_scores = scores - max_scores  # 平移到最大值为0
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_scores), dim=1))
    
    # 3. 计算最终损失
    loss = torch.sum(-pos_scores + max_scores.squeeze() + log_sum_exp)
    
    return loss
```

### 6.3 关键训练技巧

#### 6.3.1 NaN处理

```python
def handle_nan_parameters(model):
    """
    处理训练过程中可能出现的NaN参数
    """
    for param in model.parameters():
        if param.data is not None:
            # 检测NaN
            nan_mask = param.data != param.data
            if nan_mask.any():
                print(f"发现NaN参数，进行修复")
                # 用小随机数替换NaN
                param.data[nan_mask] = torch.randn_like(param.data[nan_mask]) * 0.01
```

#### 6.3.2 GRU门控机制

```python
def apply_gru_gating(self, current_hidden, previous_hidden):
    """
    使用GRU门控机制融合当前层和历史信息
    """
    # current_hidden: [n_nodes, hidden_dim] 当前层的节点表示
    # previous_hidden: [1, n_nodes_prev, hidden_dim] 上一层的节点表示
    
    # 1. 调整previous_hidden的维度以匹配当前节点
    h0_expanded = torch.zeros(1, current_hidden.size(0), current_hidden.size(1))
    h0_expanded = h0_expanded.index_copy_(1, old_nodes_indices, previous_hidden)
    
    # 2. 应用GRU门控
    current_hidden = self.dropout(current_hidden)
    gated_hidden, new_h0 = self.gru(
        current_hidden.unsqueeze(0),  # [1, n_nodes, hidden_dim]
        h0_expanded                   # [1, n_nodes, hidden_dim]
    )
    
    return gated_hidden.squeeze(0), new_h0
```

#### 6.3.3 温度参数调节

```python
class GNNLayer(torch.nn.Module):
    def train(self, mode=True):
        """
        根据训练/测试模式调整采样策略
        """
        self.training = mode
        if self.training and self.tau > 0:
            # 训练时使用Gumbel Softmax进行可微分采样
            self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            # 测试时使用标准Softmax
            self.softmax = lambda x: F.softmax(x, dim=1)
        
        for module in self.children():
            module.train(mode)
        return self
```

## 7. 评估机制

### 7.1 评估指标

AdaProp使用知识图谱推理的标准评估指标：

- **MRR (Mean Reciprocal Rank)**: 平均倒数排名
- **Hit@1**: Top-1准确率
- **Hit@10**: Top-10准确率

### 7.2 过滤评估

```python
def filtered_evaluation(scores, true_targets, filters):
    """
    过滤评估：排除训练集中已知的正确答案
    """
    batch_size, n_entities = scores.shape
    rankings = []
    
    for i in range(batch_size):
        # 1. 获取当前查询的所有已知答案
        known_answers = filters[i]  # 训练/验证集中的已知正确答案
        
        # 2. 创建过滤掩码
        filter_mask = np.zeros(n_entities)
        filter_mask[known_answers] = 1  # 已知答案位置标记为1
        
        # 3. 计算排名
        # 原始排名：基于所有实体的分数
        full_ranks = rankdata(-scores[i], method='ordinal')
        
        # 过滤排名：只考虑未知实体的分数
        filtered_scores = scores[i] * (1 - filter_mask)  # 已知答案分数置0
        filtered_ranks = rankdata(-filtered_scores, method='min')
        
        # 4. 计算最终排名
        true_target = true_targets[i]
        final_rank = full_ranks[true_target] - filtered_ranks[true_target] + 1
        rankings.append(final_rank)
    
    return rankings

def compute_metrics(rankings):
    """
    计算评估指标
    """
    rankings = np.array(rankings)
    
    # MRR: 倒数排名的平均值
    mrr = np.mean(1.0 / rankings)
    
    # Hit@K: 排名在前K的比例
    hit_1 = np.mean(rankings <= 1)
    hit_10 = np.mean(rankings <= 10)
    
    return {
        'mrr': mrr,
        'hit_1': hit_1, 
        'hit_10': hit_10
    }
```

### 7.3 评估流程

```python
def evaluate_model(model, data_loader, mode='test'):
    """
    完整的模型评估流程
    """
    model.eval()
    all_rankings = []
    
    with torch.no_grad():
        for batch_idx in range(data_loader.n_batches):
            # 1. 获取评估批次
            subjects, relations, objects = data_loader.get_batch(batch_idx, data=mode)
            
            # 2. 模型预测
            scores = model(subjects, relations, mode=mode)
            scores = scores.cpu().numpy()
            
            # 3. 准备过滤信息
            batch_filters = []
            for i in range(len(subjects)):
                # 获取(subject, relation)对应的所有已知目标实体
                known_objects = data_loader.filters[(subjects[i], relations[i])]
                filter_mask = np.zeros(data_loader.n_entities)
                filter_mask[known_objects] = 1
                batch_filters.append(filter_mask)
            
            # 4. 计算过滤排名
            batch_rankings = cal_ranks(scores, objects, np.array(batch_filters))
            all_rankings.extend(batch_rankings)
    
    # 5. 计算最终指标
    metrics = cal_performance(all_rankings)
    return metrics
```

## 8. 关键创新点

### 8.1 技术创新总结

#### 8.1.1 自适应传播路径学习
- **问题**：传统GNN使用固定传播路径，忽略查询相关性
- **解决方案**：学习查询感知的动态传播路径
- **实现**：查询关系嵌入参与注意力计算和采样决策

#### 8.1.2 增量采样机制
- **问题**：GNN传播中实体数量指数增长
- **解决方案**：每层只对新发现的实体进行采样，保留重要的历史实体
- **优势**：线性复杂度，保持语义连贯性

#### 8.1.3 可微分离散采样
- **问题**：离散采样不可微分，无法端到端训练
- **解决方案**：Gumbel Softmax + 直通估计器
- **效果**：训练时软采样，推理时硬采样

### 8.2 架构设计亮点

#### 8.2.1 多层次注意力机制
```python
# 三元注意力：节点 + 关系 + 查询
attention_score = W_attn(ReLU(
    W_s(node_features) +      # 节点注意力
    W_r(relation_features) +  # 关系注意力  
    W_qr(query_relation)      # 查询注意力
))
```

#### 8.2.2 门控信息融合
```python
# 类似ResNet的跳跃连接，保持长期依赖
hidden_new, h_state = GRU(hidden_current, h_previous)
```

#### 8.2.3 双重采样策略
- **节点采样**：选择语义相关的实体
- **边采样**：选择重要的关系连接

### 8.3 工程实现优势

#### 8.3.1 数值稳定性
- Log-sum-exp技巧防止数值溢出
- NaN检测和修复机制
- 梯度裁剪和权重约束

#### 8.3.2 模块化设计
- 清晰的层次结构
- 可配置的采样策略
- 支持多种激活函数和优化器

#### 8.3.3 多设置支持
- Transductive和Inductive设置
- 不同规模数据集的参数配置
- 灵活的评估模式

## 9. 代码实现细节

### 9.1 关键文件结构

```
AdaProp/
├── transductive/          # 传导式设置
│   ├── models.py          # 复杂的双重采样GNN模型
│   ├── base_model.py      # 训练和评估封装
│   ├── load_data.py       # 数据加载器
│   ├── train.py           # 训练脚本
│   └── utils.py           # 工具函数
├── inductive/             # 归纳式设置  
│   ├── models.py          # 简化的GNN模型
│   ├── base_model.py      # 训练和评估封装
│   ├── load_data.py       # 双实体空间数据加载器
│   ├── train.py           # 训练脚本
│   ├── reproduce.sh       # 复现脚本
│   └── utils.py           # 工具函数
└── data/                  # 数据集
    ├── transductive/      # 传导式数据
    └── inductive/         # 归纳式数据
```

### 9.2 核心类详解

#### 9.2.1 GNNLayer类

```python
class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, n_ent, 
                 n_node_topk=-1, n_edge_topk=-1, tau=1.0, act=lambda x:x):
        super(GNNLayer, self).__init__()
        
        # 基础参数
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.n_rel = n_rel
        self.n_ent = n_ent
        self.act = act
        
        # 采样参数
        self.n_node_topk = n_node_topk  # 节点采样数量
        self.n_edge_topk = n_edge_topk  # 边采样数量
        self.tau = tau                  # Gumbel Softmax温度
        
        # 网络组件
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)     # 关系嵌入
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False) # 源节点注意力
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False) # 关系注意力
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)           # 查询注意力
        self.w_alpha = nn.Linear(attn_dim, 1)                 # 注意力输出
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)     # 消息变换
        self.W_samp = nn.Linear(in_dim, 1, bias=False)        # 采样网络
```

#### 9.2.2 GNNModel类

```python
class GNNModel(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNNModel, self).__init__()
        
        # 模型参数
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_ent = params.n_ent
        self.n_rel = params.n_rel
        
        # 采样参数
        self.n_node_topk = params.n_node_topk
        self.n_edge_topk = params.n_edge_topk
        
        # 构建多层GNN
        self.gnn_layers = nn.ModuleList([
            GNNLayer(...) for _ in range(self.n_layer)
        ])
        
        # 其他组件
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
```

#### 9.2.3 BaseModel类

```python
class BaseModel(object):
    def __init__(self, args, loader):
        # 模型初始化
        self.model = GNNModel(args, loader)
        self.model.cuda()
        
        # 训练组件
        self.optimizer = Adam(self.model.parameters(), 
                            lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        
        # 数据和评估
        self.loader = loader
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
```

### 9.3 数据处理流程

#### 9.3.1 数据加载

```python
class DataLoader:
    def __init__(self, task_dir):
        # 1. 加载实体和关系映射
        self.entity2id = self.load_entities(task_dir + '/entities.txt')
        self.relation2id = self.load_relations(task_dir + '/relations.txt')
        
        # 2. 加载训练/验证/测试数据
        self.train_triples = self.read_triples(task_dir + '/train.txt')
        self.valid_triples = self.read_triples(task_dir + '/valid.txt')
        self.test_triples = self.read_triples(task_dir + '/test.txt')
        
        # 3. 构建知识图谱和索引
        self.KG, self.adjacency_matrix = self.build_kg(self.train_triples)
        
        # 4. 准备过滤器（用于评估）
        self.filters = self.build_filters()
    
    def get_neighbors(self, nodes, batch_size, mode='train'):
        """
        获取给定节点的邻居节点和边
        """
        # 1. 提取当前层的所有节点
        current_entities = nodes[:, 1]  # 节点ID
        
        # 2. 查找所有邻居
        neighbors = []
        edges = []
        
        for i, entity in enumerate(current_entities):
            # 从邻接矩阵中获取邻居
            entity_neighbors = self.adjacency_matrix[entity].nonzero()[1]
            
            # 构建边信息：[batch_idx, head, relation, tail, head_idx, tail_idx]
            for neighbor in entity_neighbors:
                relation = self.get_relation(entity, neighbor)
                edge = [nodes[i,0], entity, relation, neighbor, i, len(neighbors)]
                edges.append(edge)
                neighbors.append([nodes[i,0], neighbor])
        
        # 3. 去重和索引处理
        unique_neighbors = self.remove_duplicates(neighbors)
        edge_array = np.array(edges)
        
        return torch.LongTensor(unique_neighbors), torch.LongTensor(edge_array)
```

#### 9.3.2 批次处理

```python
def get_batch(self, batch_indices, data='train'):
    """
    获取训练/验证/测试批次
    """
    if data == 'train':
        triples = self.train_triples[batch_indices]
    elif data == 'valid':
        triples = self.valid_triples[batch_indices]  
    else:
        triples = self.test_triples[batch_indices]
    
    # 分离头实体、关系、尾实体
    subjects = triples[:, 0]
    relations = triples[:, 1] 
    objects = triples[:, 2]
    
    # 转换为one-hot格式（用于评估）
    if data != 'train':
        object_labels = np.zeros((len(triples), self.n_entities))
        for i, obj in enumerate(objects):
            object_labels[i, obj] = 1
        return subjects, relations, object_labels
    
    return triples
```

## 10. 使用方法

### 10.1 环境配置

```bash
# 依赖安装
pip install torch==1.12.1
pip install torch_scatter==2.0.9  
pip install numpy==1.21.6
pip install scipy==1.10.1
```

### 10.2 Transductive设置使用

#### 10.2.1 训练命令

```bash
# Family数据集
python3 train.py --data_path ./data/family/ --train --topk 100 --layers 8 --fact_ratio 0.90 --gpu 0

# WN18RR数据集  
python3 train.py --data_path ./data/WN18RR/ --train --topk 1000 --layers 8 --fact_ratio 0.96 --gpu 0

# FB15k-237数据集
python3 train.py --data_path ./data/fb15k-237/ --train --topk 2000 --layers 7 --fact_ratio 0.99 --remove_1hop_edges --gpu 0
```

#### 10.2.2 评估命令

```bash
# 使用保存的模型进行评估
python3 train.py --data_path ./data/family/ --eval --topk 100 --layers 8 --gpu 0 --weight ./data/family/8-layers-best.pt
```

#### 10.2.3 参数说明

- `--data_path`: 数据集路径
- `--topk`: 每层采样的节点数量
- `--layers`: GNN层数
- `--fact_ratio`: 事实保留比例
- `--gpu`: GPU设备ID
- `--train`: 训练模式
- `--eval`: 评估模式
- `--weight`: 预训练模型路径

### 10.3 Inductive设置使用

#### 10.3.1 训练命令

```bash
# WN18RR v1数据集
python3 train.py --data_path ./data/WN18RR_v1

# FB237 v1数据集  
python3 train.py --data_path ./data/fb237_v1

# NELL v1数据集
python3 train.py --data_path ./data/nell_v1
```

#### 10.3.2 批量复现

```bash
# 运行所有inductive实验
bash reproduce.sh
```

### 10.4 自定义数据集

#### 10.4.1 数据格式

每个数据集需要包含以下文件：

```
your_dataset/
├── entities.txt      # 实体列表：entity_name \t entity_id
├── relations.txt     # 关系列表：relation_name \t relation_id  
├── train.txt         # 训练三元组：head \t relation \t tail
├── valid.txt         # 验证三元组：head \t relation \t tail
└── test.txt          # 测试三元组：head \t relation \t tail
```

#### 10.4.2 Inductive数据格式

```
your_dataset/         # 训练数据（transductive）
├── entities.txt
├── relations.txt
├── train.txt
├── valid.txt  
└── test.txt

your_dataset_ind/     # 测试数据（inductive）
├── entities.txt      # 包含新实体
├── train.txt         # 空文件或包含新实体的训练数据
├── valid.txt         # 涉及新实体的验证数据
└── test.txt          # 涉及新实体的测试数据
```

### 10.5 超参数调优

#### 10.5.1 关键超参数

```python
# 模型结构参数
hidden_dim = 64        # 隐藏维度
attn_dim = 5          # 注意力维度  
n_layer = 8           # GNN层数
dropout = 0.02        # Dropout率

# 采样参数
topk = 1000           # 节点采样数量
tau = 1.0             # Gumbel Softmax温度

# 训练参数
lr = 0.003            # 学习率
decay_rate = 0.994    # 学习率衰减
weight_decay = 0.00014 # 权重衰减
batch_size = 50       # 批次大小
```

#### 10.5.2 数据集特定配置

不同数据集需要不同的超参数配置，代码中已经为常用数据集提供了最优配置：

```python
# transductive/train.py中的配置示例
if dataset == 'family':
    opts.lr = 0.0036
    opts.hidden_dim = 48
    opts.n_node_topk = [opts.topk] * opts.layers
    
elif dataset == 'WN18RR':
    opts.lr = 0.0030
    opts.hidden_dim = 64
    opts.n_node_topk = [opts.topk] * opts.layers
```

### 10.6 结果分析

#### 10.6.1 性能指标

模型在多个数据集上的性能表现：

| 数据集 | Test MRR | Test Hit@1 | Test Hit@10 |
|--------|----------|------------|-------------|
| Family | 0.9882   | 0.9866     | 0.9903      |
| UMLS   | 0.9698   | 0.9558     | 0.9945      |
| WN18RR | 0.5622   | 0.5086     | 0.6608      |
| FB15k-237| 0.4173 | 0.3312     | 0.5848      |
| NELL995| 0.5542   | 0.4933     | 0.6526      |
| YAGO3-10| 0.5738  | 0.5105     | 0.6865      |

#### 10.6.2 效率分析

- **时间复杂度**：O(L × K × d)，其中L是层数，K是采样数量，d是特征维度
- **空间复杂度**：O(K × d)，与采样数量线性相关
- **相比传统GNN**：避免了指数级的邻居扩展

---

## 总结

AdaProp是一个精心设计的知识图谱推理模型，其核心创新在于自适应传播机制和增量采样策略。通过学习查询感知的传播路径，模型能够在保持线性复杂度的同时，选择语义相关的实体进行推理。

模型的主要优势包括：
1. **高效性**：线性复杂度，适用于大规模知识图谱
2. **准确性**：查询感知的传播提高了推理准确性  
3. **通用性**：支持transductive和inductive两种设置
4. **可扩展性**：模块化设计，易于扩展和改进

该实现提供了完整的训练和评估流程，包含了丰富的工程细节和优化技巧，是图神经网络在知识图谱推理领域的优秀实践。
