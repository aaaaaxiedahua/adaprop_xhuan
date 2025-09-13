# AdaProp数据流程与核心创新详解

## 目录
- [1. 完整数据流程](#1-完整数据流程)
- [2. 核心创新点详解](#2-核心创新点详解)
- [3. 三种采样机制](#3-三种采样机制)
- [4. 关键文件作用](#4-关键文件作用)

## 1. 完整数据流程

### 1.1 数据预处理阶段

#### 步骤1: 原始数据加载
**文件**: `load_data.py` - `DataLoader.__init__()`
**位置**: `transductive/load_data.py:47` 和 `inductive/load_data.py:36`

```python
# 1. 加载实体映射
with open(os.path.join(task_dir, 'entities.txt')) as f:
    self.entity2id = dict()
    for line in f:
        entity, eid = line.strip().split()
        self.entity2id[entity] = int(eid)  # 字符串实体 -> 数字ID

# 2. 加载关系映射  
with open(os.path.join(task_dir, 'relations.txt')) as f:
    self.relation2id = dict()
    for line in f:
        relation, rid = line.strip().split()
        self.relation2id[relation] = int(rid)  # 字符串关系 -> 数字ID
```

**数据格式转换**:
```
原始: ("Barack_Obama", "born_in", "Hawaii")
转换后: (0, 5, 1247)  # (头实体ID, 关系ID, 尾实体ID)
```

#### 步骤2: 三元组预处理
**文件**: `load_data.py` - `read_triples()`

```python
def read_triples(self, directory, filename, mode='transductive'):
    triples = []
    with open(os.path.join(directory, filename)) as f:
        for line in f:
            h, r, t = line.strip().split()
            # 转换为数字ID
            if mode == 'transductive':
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
            else:
                h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]
            
            # 添加正向和反向三元组
            triples.append([h, r, t])           # 正向: h -r-> t
            triples.append([t, r+self.n_rel, h])  # 反向: t -r_inv-> h
    return triples
```

**关键处理**:
- 每个三元组生成正向和反向两个版本
- 反向关系ID = 原关系ID + 关系总数

#### 步骤3: 知识图谱构建
**文件**: `load_data.py` - `load_graph()`

```python
def load_graph(self, triples, mode='transductive'):
    n_ent = self.n_ent if mode=='transductive' else self.n_ent_ind
    
    KG = np.array(triples)  # 所有三元组
    
    # 添加自环边 (每个实体指向自己)
    idd = np.concatenate([
        np.expand_dims(np.arange(n_ent), 1),      # 头实体: 0,1,2,...
        2*self.n_rel*np.ones((n_ent, 1)),        # 自环关系ID
        np.expand_dims(np.arange(n_ent), 1)      # 尾实体: 0,1,2,...
    ], 1)
    
    KG = np.concatenate([KG, idd], 0)  # 合并原三元组和自环
    
    # 构建邻接矩阵 (用于快速邻居查找)
    n_fact = KG.shape[0]
    M_sub = csr_matrix(
        (np.ones((n_fact,)), (np.arange(n_fact), KG[:,0])), 
        shape=(n_fact, n_ent)
    )
    return KG, M_sub
```

**为什么添加自环边？**
- **保持自身信息**: 每个实体都需要保留自身的特征信息
- **数学稳定性**: 确保每个节点在消息传递时至少有一条边
- **特殊关系ID**: 自环使用关系ID = 2*n_rel，区别于正向和反向关系

### 1.2 训练数据准备阶段

#### 步骤4: 批次数据获取
**文件**: `base_model.py` - `train_batch()`
**调用**: `loader.get_batch(batch_idx)`

```python
def get_batch(self, batch_idx):
    # 获取一个批次的训练三元组
    triple = self.loader.get_batch(batch_idx)  # [batch_size, 3]
    # triple[:, 0] = 头实体IDs
    # triple[:, 1] = 关系IDs  
    # triple[:, 2] = 尾实体IDs (正确答案)
    return triple
```

### 1.3 模型推理阶段

#### 步骤5: 模型前向传播开始
**文件**: `models.py` - `GNNModel.forward()`
**输入**: `subs`(头实体列表), `rels`(关系列表)

```python
def forward(self, subs, rels, mode='train'):
    n = len(subs)  # 批次大小
    q_sub = torch.LongTensor(subs).cuda()   # [batch_size] 查询头实体
    q_rel = torch.LongTensor(rels).cuda()   # [batch_size] 查询关系
    
    # 初始化
    nodes = torch.cat([
        torch.arange(n).unsqueeze(1).cuda(),  # 批次索引
        q_sub.unsqueeze(1)                    # 实体ID
    ], 1)  # [batch_size, 2] = [(batch_idx, entity_id), ...]
    
    # 🔥关键初始化🔥
    hidden = torch.zeros(n, self.hidden_dim).cuda()  # 初始节点嵌入 [batch_size, hidden_dim]
    h0 = torch.zeros((1, n, self.hidden_dim)).cuda() # GRU初始状态 [1, batch_size, hidden_dim]
```

**初始节点嵌入详解**:
- **为什么用零初始化**: 查询实体没有预训练嵌入，从零开始学习
- **维度含义**: `[batch_size, hidden_dim]` 表示每个查询实体的初始特征向量
- **学习过程**: 通过GNN层的消息传递逐渐学习到有意义的表示

#### 步骤6: 逐层传播 (核心循环)

```python
for i in range(self.n_layer):  # 对每一层GNN
    # 6.1 获取邻居节点和边
    nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(
        nodes.data.cpu().numpy(), n, mode=mode
    )
    
    # 6.2 GNN层前向传播
    hidden, nodes, sampled_nodes_idx = self.gnn_layers[i](
        q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, n
    )
    
    # 6.3 GRU门控更新 (🔥关键步骤🔥)
    # 6.3.1 调整h0维度以匹配当前层节点数
    h0 = torch.zeros(1, n_node, hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
    
    # 6.3.2 只保留采样后节点的历史状态
    h0 = h0[0, sampled_nodes_idx, :].unsqueeze(0)  # [1, sampled_nodes, hidden_dim]
    
    # 6.3.3 应用dropout防止过拟合
    hidden = self.dropout(hidden)
    
    # 6.3.4 🔥GRU门控融合🔥: 结合当前层特征和历史信息
    hidden, h0 = self.gate(
        hidden.unsqueeze(0),  # 当前层特征 [1, sampled_nodes, hidden_dim]
        h0                    # 历史状态     [1, sampled_nodes, hidden_dim]
    )
    # 输出: hidden [1, sampled_nodes, hidden_dim], h0 [1, sampled_nodes, hidden_dim]
    
    # 6.3.5 移除batch维度
    hidden = hidden.squeeze(0)  # [sampled_nodes, hidden_dim]
```

#### 步骤7: 邻居获取详解
**文件**: `load_data.py` - `get_neighbors()`

```python
def get_neighbors(self, nodes, batchsize, mode='train'):
    """
    输入: nodes = [[batch_idx, entity_id], ...] 当前层的节点
    输出: 
    - new_nodes: 扩展后的所有节点 
    - edges: 连接这些节点的所有边
    - old_nodes_new_idx: 原节点在新节点列表中的索引
    """
    
    # 1. 为每个节点查找所有邻居
    all_neighbors = []
    all_edges = []
    
    for i, (batch_idx, entity_id) in enumerate(nodes):
        # 从知识图谱中找到该实体的所有邻居
        entity_edges = self.KG[self.KG[:, 0] == entity_id]  # 以entity_id为头的边
        
        for edge in entity_edges:
            head, relation, tail = edge
            all_neighbors.append([batch_idx, tail])  # 邻居节点
            all_edges.append([batch_idx, head, relation, tail, i, len(all_neighbors)-1])
    
    # 2. 去重并重新索引
    unique_neighbors, old_to_new_mapping = self.deduplicate_nodes(all_neighbors)
    
    # 3. 更新边的索引
    for edge in all_edges:
        edge[4] = old_to_new_mapping[edge[4]]  # 更新头节点索引
        edge[5] = old_to_new_mapping[edge[5]]  # 更新尾节点索引
    
    return torch.LongTensor(unique_neighbors), torch.LongTensor(all_edges), old_nodes_new_idx
```

#### 步骤8: GNN层处理详解
**文件**: `models.py` - `GNNLayer.forward()`

```python
def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize):
    # 8.1 提取边信息
    sub = edges[:,4]  # 源节点在当前层节点列表中的索引
    rel = edges[:,2]  # 关系ID
    obj = edges[:,5]  # 目标节点在当前层节点列表中的索引
    
    # 8.2 构建消息
    hs = hidden[sub]                          # 源节点特征
    hr = self.rela_embed(rel)                # 关系嵌入
    h_qr = self.rela_embed(q_rel)[edges[:,0]] # 查询关系嵌入
    
    # 8.3 计算注意力权重 (创新点1: 查询感知注意力)
    alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(
        self.Ws_attn(hs) +      # 源节点注意力
        self.Wr_attn(hr) +      # 关系注意力  
        self.Wqr_attn(h_qr)     # 查询关系注意力 ⭐核心创新⭐
    )))
    
    # 8.4 加权消息聚合
    message = alpha * (hs + hr)
    message_agg = scatter(message, index=obj, dim=0, dim_size=nodes.shape[0], reduce='sum')
    
    # 8.5 更新节点表示
    hidden_new = self.act(self.W_h(message_agg))
    
    # 8.6 节点采样 (创新点2: 自适应采样)
    if self.n_node_topk > 0 and not是最后一层:
        hidden_new, nodes, sampled_idx = self.adaptive_node_sampling(
            hidden_new, nodes, old_nodes_new_idx, batchsize
        )
        return hidden_new, nodes, sampled_idx
    
    return hidden_new, nodes, torch.arange(nodes.shape[0])
```

### 1.4 最终预测阶段

#### 步骤9: 计算实体分数
**文件**: `models.py` - `GNNModel.forward()` 最后部分

```python
# 9.1 计算每个候选实体的分数
scores = self.W_final(hidden).squeeze(-1)  # [num_final_nodes]

# 9.2 将分数映射到完整实体空间
scores_all = torch.zeros((n, self.loader.n_ent)).cuda()  # [batch_size, num_entities]
scores_all[[nodes[:,0], nodes[:,1]]] = scores  # 只有访问过的实体有非零分数

return scores_all  # [batch_size, num_entities]
```

#### 步骤10: 损失计算和反向传播
**文件**: `base_model.py` - `train_batch()`

```python
# 10.1 获取正样本分数
scores = self.model(triple[:,0], triple[:,1])  # [batch_size, num_entities]
pos_scores = scores[torch.arange(len(scores)), triple[:,2]]  # 正确答案的分数

# 10.2 计算负对数似然损失
max_n = torch.max(scores, 1, keepdim=True)[0]  # 数值稳定技巧
loss = torch.sum(-pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))

# 10.3 反向传播
loss.backward()
self.optimizer.step()
```

### 1.5 GNNLayer vs GNNModel 详细区别

#### 1.5.1 架构层次对比

```
GNNModel (整个模型)
├── 控制多层传播流程
├── 处理节点扩展和采样
├── 管理GRU门控状态
└── 最终预测分数计算
    ↓ 调用
GNNLayer (单层GNN)  
├── 单层消息传递
├── 注意力权重计算
├── 边采样 (可选)
└── 节点采样 (可选)
```

#### 1.5.2 功能职责对比

| 方面 | GNNModel.forward() | GNNLayer.forward() |
|------|-------------------|-------------------|
| **主要职责** | 整个模型的控制器 | 单层GNN的执行器 |
| **输入** | 查询三元组 (subs, rels) | 当前层的节点和边 |
| **输出** | 所有实体的预测分数 | 更新后的节点嵌入 |
| **处理层数** | 多层循环控制 | 单层处理 |
| **邻居获取** | ✅ 调用get_neighbors() | ❌ 接收已获取的邻居 |
| **节点扩展** | ✅ 管理节点集合的增长 | ❌ 处理给定的节点 |
| **GRU门控** | ✅ 跨层状态管理 | ❌ 不涉及 |
| **采样决策** | ✅ 决定何时采样 | ✅ 执行具体采样 |
| **最终预测** | ✅ 计算预测分数 | ❌ 不涉及 |

#### 1.5.3 详细执行流程对比

**GNNModel.forward() 的执行流程**:
```python
def GNNModel_forward_detailed():
    """
    GNNModel.forward() 的完整执行流程
    """
    # 🔥第1步: 初始化🔥
    nodes = [(0, query_entity_1), (1, query_entity_2), ...]  # 初始查询节点
    hidden = zero_embeddings  # 零初始化嵌入
    h0 = zero_gru_state      # 零初始化GRU状态
    
    # 🔥第2步: 多层传播循环🔥
    for layer_i in range(n_layers):
        # 2.1 获取邻居 (GNNModel负责)
        nodes, edges, old_indices = loader.get_neighbors(nodes)
        
        # 2.2 调用单层GNN (委托给GNNLayer)
        hidden, nodes, sampled_indices = gnn_layers[layer_i].forward(
            query_subs, query_rels, hidden, edges, nodes, old_indices, batch_size
        )
        
        # 2.3 GRU门控更新 (GNNModel负责)
        h0 = adjust_gru_state_dimensions(h0, old_indices, sampled_indices)
        hidden, h0 = gru_gate(hidden, h0)
    
    # 🔥第3步: 最终预测🔥
    scores = final_layer(hidden)
    full_scores = map_to_all_entities(scores, nodes)
    return full_scores
```

**GNNLayer.forward() 的执行流程**:
```python
def GNNLayer_forward_detailed():
    """
    GNNLayer.forward() 的完整执行流程
    """
    # 🔥第1步: 消息构建🔥
    source_features = hidden[edge_sources]      # 源节点特征
    relation_features = relation_embed(edges)   # 关系嵌入
    query_features = query_relation_embed()     # 查询关系嵌入
    
    # 🔥第2步: 注意力计算🔥 (核心创新1)
    attention_weights = attention_network(
        source_features + relation_features + query_features
    )
    
    # 🔥第3步: 边采样🔥 (可选, 仅transductive)
    if enable_edge_sampling:
        selected_edges, edge_weights = edge_sampling(attention_weights)
    
    # 🔥第4步: 消息传递🔥
    messages = attention_weights * (source_features + relation_features)
    aggregated_messages = scatter_sum(messages, target_nodes)
    updated_embeddings = transform(aggregated_messages)
    
    # 🔥第5步: 节点采样🔥 (可选, 核心创新2)
    if enable_node_sampling and not_final_layer:
        # 5.1 区分新旧节点
        old_nodes, new_nodes = separate_old_new(nodes, old_indices)
        
        # 5.2 计算新节点采样分数
        new_node_scores = sampling_network(new_nodes_embeddings)
        
        # 5.3 Gumbel Softmax采样
        selected_new_nodes = gumbel_softmax_sampling(new_node_scores)
        
        # 5.4 组合最终节点
        final_nodes = old_nodes + selected_new_nodes
        final_embeddings = embeddings[final_nodes]
        
        return final_embeddings, final_nodes, selected_indices
    
    return updated_embeddings, nodes, all_indices
```

#### 1.5.4 数据流转示例

假设有一个2层GNN处理查询 `("Obama", "born_in", ?)`：

```python
# GNNModel.forward() 开始执行
batch_size = 1
query_sub = ["Obama"]  # ID: 0
query_rel = ["born_in"]  # ID: 5

# === 第0层初始化 ===
nodes_0 = [(0, 0)]  # [(batch_idx, entity_id)] = [(0, Obama_id)]
hidden_0 = [[0, 0, 0, ..., 0]]  # [1, 64] 零初始化
h0_0 = [[[0, 0, 0, ..., 0]]]   # [1, 1, 64] GRU零状态

# === 第1层传播 ===
# GNNModel调用: get_neighbors()
nodes_1, edges_1, old_idx_1 = loader.get_neighbors(nodes_0)
# 结果: 
# nodes_1 = [(0,0), (0,1247), (0,1248), ...]  # Obama + 其邻居
# edges_1 = [(0,0,5,1247,0,1), (0,0,7,1248,0,2), ...]  # Obama的所有边

# GNNModel调用: gnn_layers[0].forward()
hidden_1, nodes_1, sampled_1 = gnn_layers[0].forward(
    [0], [5], hidden_0, edges_1, nodes_1, [0], 1
)
# GNNLayer内部执行:
# - 计算注意力权重 (考虑query_rel=5)
# - 消息传递和聚合
# - 节点采样 (保留top-k个最相关邻居)

# GNNModel执行: GRU门控
h0_1 = adjust_h0_dimensions(h0_0, old_idx_1, sampled_1)
hidden_1, h0_1 = gru_gate(hidden_1, h0_1)

# === 第2层传播 ===
# 重复上述过程...

# === 最终预测 ===
scores = W_final(hidden_final)  # 计算每个候选实体分数
full_scores = map_to_full_space(scores, final_nodes)
return full_scores  # [1, n_entities] 所有实体的预测分数
```

#### 1.5.5 关键设计原则

**GNNModel的设计原则**:
- **全局控制**: 管理整个推理过程的状态和流程
- **资源管理**: 控制内存使用和计算复杂度
- **状态持续**: 通过GRU保持跨层的长期依赖
- **灵活扩展**: 支持不同层数和采样策略

**GNNLayer的设计原则**:
- **单一职责**: 专注于单层的消息传递和特征更新
- **可复用性**: 同一层可以处理不同的输入
- **可配置性**: 支持不同的采样和注意力策略
- **计算效率**: 优化单层的计算性能

#### 1.5.6 GRU门控机制详解

**为什么需要GRU门控？**

```python
# 🔥问题: 没有门控的多层传播🔥
def without_gru_gating():
    """
    没有门控机制时，深层GNN容易出现的问题
    """
    # 第1层: 学习到"Obama"的基本特征
    hidden_1 = gnn_layer_1(query_features)  # [nodes, dim]
    
    # 第2层: 完全覆盖第1层的信息
    hidden_2 = gnn_layer_2(hidden_1)  # 可能丢失重要的初始信息
    
    # 第3层: 继续覆盖，梯度消失
    hidden_3 = gnn_layer_3(hidden_2)  # 查询信息可能完全丢失
    
    # ❌ 问题: 
    # 1. 梯度消失 - 深层网络难以训练
    # 2. 信息丢失 - 重要的查询信息被覆盖
    # 3. 过平滑 - 所有节点特征趋于相同

# 🔥解决方案: GRU门控机制🔥  
def with_gru_gating():
    """
    GRU门控机制保持长期依赖和梯度流动
    """
    h0 = zero_state  # 初始GRU状态
    
    for layer in range(n_layers):
        # 当前层的新信息
        current_info = gnn_layer(previous_hidden)
        
        # GRU门控融合: 决定保留多少历史信息和新信息
        updated_hidden, h0 = gru_gate(
            current_info,  # 新信息 (当前层学到的特征)
            h0            # 历史信息 (之前层的重要信息)
        )
        
        # ✅ 优势:
        # 1. 梯度流动 - 类似ResNet的跳跃连接
        # 2. 信息保持 - 重要的历史信息不会丢失  
        # 3. 自适应融合 - 学习最优的信息组合比例
```

**GRU门控的数学原理**:
```python
def gru_gate_mathematics():
    """
    GRU门控的数学公式详解
    """
    # 输入:
    # x_t: 当前层的节点特征 [nodes, hidden_dim]
    # h_{t-1}: 上一层的GRU状态 [nodes, hidden_dim]
    
    # 1. 重置门 (Reset Gate): 决定遗忘多少历史信息
    r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1} + b_r)
    
    # 2. 更新门 (Update Gate): 决定更新多少新信息
    z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)
    
    # 3. 候选状态: 基于重置后的历史信息计算新状态
    h_tilde = tanh(W_h @ x_t + U_h @ (r_t ⊙ h_{t-1}) + b_h)
    
    # 4. 最终状态: 在历史信息和新信息间进行插值
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde
    #     ↑保留历史信息    ↑融入新信息
    
    return h_t
```

**GRU在AdaProp中的具体作用**:
```python
# transductive/models.py:174-178 的详细解释
def adaprop_gru_usage():
    """
    AdaProp中GRU门控的具体应用
    """
    # 第1层后: h0包含查询实体的初始信息
    h0_layer1 = gru_state_containing_query_info
    
    # 第2层: 获得了1跳邻居的信息
    current_layer2 = one_hop_neighbor_features
    
    # GRU融合: 保留查询信息 + 融入邻居信息
    hidden_layer2, h0_layer2 = gru_gate(current_layer2, h0_layer1)
    # 结果: 既有查询的语义，又有邻居的上下文
    
    # 第3层: 获得了2跳邻居的信息  
    current_layer3 = two_hop_neighbor_features
    
    # GRU融合: 保留重要的历史路径 + 融入新发现
    hidden_layer3, h0_layer3 = gru_gate(current_layer3, h0_layer2)
    # 结果: 多层次的语义理解，从查询到多跳邻居
```

**为什么不用其他机制？**
- **简单相加**: `hidden = hidden_old + hidden_new` → 无法学习最优组合
- **注意力机制**: 计算复杂度高，且不专门处理序列信息
- **LSTM**: 比GRU复杂，在这个任务中GRU已足够
- **ResNet连接**: 只是简单相加，不如GRU的门控机制灵活

## 2. 核心创新点详解

### 2.1 创新点1: 查询感知的自适应传播

#### 2.1.1 传统方法的问题
```python
# 传统GNN只考虑源节点和关系
traditional_attention = W_attn(W_s(source_node) + W_r(relation))
```

#### 2.1.2 AdaProp的解决方案
```python
# AdaProp同时考虑查询关系，实现查询感知
adaprop_attention = W_attn(
    W_s(source_node) +     # 源节点特征
    W_r(relation) +        # 边关系特征
    W_qr(query_relation)   # 🔥查询关系特征🔥
)
```

#### 2.1.3 具体实现位置
**文件**: `models.py` - `GNNLayer.forward()` 第55-66行 (transductive) / 第35行 (inductive)

```python
# transductive/models.py:55-66
alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(
    self.Ws_attn(hs) +      # 源节点: "Barack_Obama"的嵌入
    self.Wr_attn(hr) +      # 边关系: "born_in"的嵌入
    self.Wqr_attn(h_qr)     # 查询关系: "born_in"的嵌入 ⭐关键创新⭐
)))

# inductive/models.py:35
alpha_2 = torch.sigmoid(self.W_attn(nn.ReLU()(
    self.Ws_attn(mess1) + 
    self.Wr_attn(hr) + 
    self.Wqr_attn(h_qr)     # ⭐查询感知⭐
)))
```

#### 2.1.4 创新的意义
- **问题**: 传统GNN对所有查询使用相同的传播模式
- **解决**: 不同查询关系(如"出生地"vs"工作地")会产生不同的注意力权重
- **效果**: 模型能够根据具体查询任务调整传播路径

### 2.2 创新点2: 增量自适应采样

#### 2.2.1 传统方法的问题
```
传统GNN传播:
Layer 0: 1个节点 (查询实体)
Layer 1: 100个节点 (1跳邻居)  
Layer 2: 10,000个节点 (2跳邻居) ❌指数爆炸❌
Layer 3: 1,000,000个节点 ❌内存不够❌
```

#### 2.2.2 AdaProp的解决方案
```
AdaProp增量采样:
Layer 0: 1个节点 (查询实体)
Layer 1: 100个节点 (1跳邻居) → 采样保留20个最相关的
Layer 2: 20个旧节点 + 新发现的邻居 → 采样保留20个最相关的  
Layer 3: 20个旧节点 + 新发现的邻居 → 采样保留20个最相关的
✅线性复杂度✅
```

#### 2.2.3 具体实现位置
**文件**: `models.py` - `GNNLayer.forward()` 第79-115行 (transductive)

```python
# transductive/models.py:79-115 节点采样实现
def forward(self, ...):
    # ... 消息传递 ...
    
    if self.n_node_topk > 0:  # 如果需要节点采样
        # 1. 区分新旧节点
        tmp_diff_node_idx = torch.ones(n_node)
        tmp_diff_node_idx[old_nodes_new_idx] = 0  # 旧节点标记为0
        bool_diff_node_idx = tmp_diff_node_idx.bool()
        diff_node = nodes[bool_diff_node_idx]  # 新发现的节点
        
        # 2. 计算新节点的重要性分数
        diff_node_logit = self.W_samp(hidden_new[bool_diff_node_idx]).squeeze(-1)
        
        # 3. 映射到固定大小张量并采样
        node_scores = torch.ones((batchsize, self.n_ent)).cuda() * float('-inf')
        node_scores[diff_node[:,0], diff_node[:,1]] = diff_node_logit
        
        # 4. Gumbel Softmax采样 (训练时) / 标准Softmax (测试时)
        node_scores = self.softmax(node_scores)
        topk_index = torch.topk(node_scores, self.n_node_topk, dim=1).indices
        
        # 5. 直通估计器: 前向传播硬采样，反向传播软采样
        # ... 复杂的采样逻辑 ...
```

**文件**: `models.py` - `GNNModel.soft_to_hard()` (inductive版本)

```python
# inductive/models.py:71-94 简化版采样
def soft_to_hard(self, i, hidden, nodes, n_ent, batch_size, old_nodes_new_idx):
    # 1. 区分新旧节点
    bool_diff_node_idx = torch.ones(n_node).bool().cuda()
    bool_diff_node_idx[old_nodes_new_idx] = False
    
    # 2. 计算采样分数
    diff_node_logits = self.Ws_layers[i](hidden[bool_diff_node_idx].detach()).squeeze(-1)
    
    # 3. Softmax + Top-K采样
    soft_all = torch.ones((batch_size, n_ent)) * float('-inf')
    soft_all[diff_nodes[:,0], diff_nodes[:,1]] = diff_node_logits
    soft_all = F.softmax(soft_all, dim=-1)
    
    _, argtopk = torch.topk(soft_all, k=self.topk, dim=-1)
    
    # 4. 更新节点嵌入
    # ... 采样逻辑 ...
```

## 3. 三种采样机制

### 3.1 采样机制1: 增量节点采样 (Incremental Node Sampling)

#### 3.1.1 什么是增量节点采样
每一层GNN传播时，只对**新发现的节点**进行采样，**保留所有历史重要节点**。

#### 3.1.2 使用位置
- **Transductive**: `models.py` - `GNNLayer.forward()` 第75-115行
- **Inductive**: `models.py` - `GNNModel.soft_to_hard()` 第71-94行

#### 3.1.3 详细实现流程

```python
def incremental_node_sampling_detailed():
    """
    增量节点采样的详细步骤
    """
    # 假设当前是第2层，有以下节点:
    # 旧节点(第1层保留): ["Obama", "Hawaii", "USA"] 
    # 新节点(第2层发现): ["Honolulu", "Pacific", "Island", "State", "America", ...]
    
    # 步骤1: 标记新旧节点
    old_nodes = ["Obama", "Hawaii", "USA"]  # 来自上一层，必须保留
    new_nodes = ["Honolulu", "Pacific", "Island", "State", "America"]  # 新发现的
    
    # 步骤2: 只对新节点计算采样分数
    new_node_scores = sampling_network(new_node_embeddings)
    # 结果: [0.8, 0.3, 0.1, 0.9, 0.6] 对应上面5个新节点
    
    # 步骤3: Top-K选择 (假设K=2)
    selected_new_nodes = ["Honolulu", "State"]  # 分数最高的2个
    
    # 步骤4: 组合结果
    final_nodes = old_nodes + selected_new_nodes
    # = ["Obama", "Hawaii", "USA", "Honolulu", "State"]
    
    # 步骤5: 传递到下一层
    return final_nodes
```

#### 3.1.4 为什么有效
- **保持连贯性**: 重要的历史节点不会丢失
- **控制复杂度**: 每层只采样固定数量的新节点
- **语义相关**: 采样网络学习选择与查询相关的节点

### 3.2 采样机制2: 边采样 (Edge Sampling) - 仅Transductive

#### 3.2.1 什么是边采样
在消息传递之前，先对边进行重要性评估和采样，只保留最重要的边进行消息传递。

#### 3.2.2 使用位置
**仅在Transductive设置中**: `transductive/models.py` - `GNNLayer.forward()` 第56-64行

#### 3.2.3 详细实现

```python
# transductive/models.py:56-64
if self.n_edge_topk > 0:  # 如果启用边采样
    # 1. 计算每条边的重要性分数
    alpha = self.w_alpha(nn.ReLU()(
        self.Ws_attn(hs) +      # 源节点特征
        self.Wr_attn(hr) +      # 关系特征
        self.Wqr_attn(h_qr)     # 查询关系特征
    )).squeeze(-1)
    
    # 2. Gumbel Softmax采样
    edge_prob = F.gumbel_softmax(alpha, tau=1, hard=False)
    
    # 3. Top-K边选择
    topk_index = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]
    
    # 4. 直通估计器
    edge_prob_hard = torch.zeros((alpha.shape[0])).cuda()
    edge_prob_hard[topk_index] = 1
    alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
    
    # 5. 只使用采样后的边进行消息传递
    alpha = torch.sigmoid(alpha).unsqueeze(-1)
```

#### 3.2.4 边采样的例子

```python
# 假设当前有以下边需要处理:
edges = [
    ("Obama", "born_in", "Hawaii"),      # 边1: 高相关性
    ("Obama", "friend_of", "Biden"),     # 边2: 低相关性  
    ("Obama", "lived_in", "Chicago"),    # 边3: 中等相关性
    ("Obama", "married_to", "Michelle"), # 边4: 低相关性
    ("Obama", "president_of", "USA")     # 边5: 高相关性
]

# 查询: ("Obama", "born_in", ?)

# 边重要性分数 (由注意力网络计算):
edge_scores = [0.9, 0.1, 0.5, 0.2, 0.8]

# Top-2边采样结果:
selected_edges = [
    ("Obama", "born_in", "Hawaii"),    # 分数0.9，最相关
    ("Obama", "president_of", "USA")   # 分数0.8，次相关
]

# 只有这2条边参与消息传递，其他边被过滤
```

### 3.3 采样机制3: Gumbel Softmax采样 (可微分离散采样)

#### 3.3.1 什么是Gumbel Softmax采样
一种使离散采样过程可微分的技术，结合了Gumbel分布和Softmax函数。

#### 3.3.2 使用位置
- **节点采样**: `transductive/models.py` 第94行
- **边采样**: `transductive/models.py` 第58行
- **训练/测试切换**: `transductive/models.py` 第33-39行

#### 3.3.3 详细实现

```python
class GNNLayer(torch.nn.Module):
    def train(self, mode=True):
        """根据训练/测试模式切换采样策略"""
        self.training = mode
        if self.training and self.tau > 0:
            # 训练时: 使用Gumbel Softmax (可微分)
            self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            # 测试时: 使用标准Softmax (确定性)
            self.softmax = lambda x: F.softmax(x, dim=1)
```

#### 3.3.4 Gumbel Softmax vs 标准采样

```python
# 标准Top-K采样 (不可微分):
def standard_sampling(scores, k):
    _, indices = torch.topk(scores, k)
    mask = torch.zeros_like(scores)
    mask[indices] = 1  # 硬采样: 0或1
    return mask  # 梯度无法传播 ❌

# Gumbel Softmax采样 (可微分):
def gumbel_softmax_sampling(scores, k, tau=1.0):
    # 1. 添加Gumbel噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores)))
    noisy_scores = (scores + gumbel_noise) / tau
    
    # 2. Softmax (软采样)
    soft_mask = F.softmax(noisy_scores, dim=-1)  # 连续值: 0~1
    
    # 3. Top-K硬采样 (前向传播)
    _, indices = torch.topk(soft_mask, k)
    hard_mask = torch.zeros_like(scores)
    hard_mask[indices] = 1
    
    # 4. 直通估计器 (关键技巧)
    # 前向传播: 使用hard_mask (离散)
    # 反向传播: 使用soft_mask的梯度 (连续)
    final_mask = hard_mask - soft_mask.detach() + soft_mask
    return final_mask  # 梯度可以传播 ✅
```

#### 3.3.5 直通估计器 (Straight-Through Estimator)

```python
# 这是Gumbel Softmax的关键技巧
final_output = hard_sampling - soft_sampling.detach() + soft_sampling

# 分解理解:
# 1. hard_sampling: 离散的采样结果 (0或1)
# 2. soft_sampling.detach(): 软采样结果，但阻断梯度
# 3. soft_sampling: 软采样结果，保留梯度

# 前向传播时:
# final_output = hard_sampling - soft_sampling + soft_sampling = hard_sampling

# 反向传播时:
# ∂final_output/∂input = ∂soft_sampling/∂input (因为hard_sampling和detach项梯度为0)
```

### 3.4 三种采样机制的协同工作

#### 3.4.1 完整采样流程 (Transductive)

```python
def complete_sampling_process():
    """
    三种采样机制的完整协同过程
    """
    # 输入: 当前层的所有边和节点
    
    # 🔥采样1: 边采样🔥
    if enable_edge_sampling:
        # 计算边重要性分数
        edge_scores = attention_network(edges)
        # Gumbel Softmax采样选择重要边
        selected_edges = gumbel_softmax_sampling(edge_scores, top_k_edges)
    else:
        selected_edges = all_edges
    
    # 消息传递 (只在选中的边上进行)
    messages = message_passing(selected_edges)
    updated_node_embeddings = aggregate_messages(messages)
    
    # 🔥采样2: 增量节点采样🔥  
    if not_final_layer:
        # 区分新旧节点
        old_nodes, new_nodes = separate_old_new_nodes(all_nodes)
        
        # 只对新节点计算采样分数
        new_node_scores = sampling_network(new_nodes_embeddings)
        
        # 🔥采样3: Gumbel Softmax采样🔥
        selected_new_nodes = gumbel_softmax_sampling(new_node_scores, top_k_nodes)
        
        # 组合最终节点集合
        final_nodes = old_nodes + selected_new_nodes
    else:
        final_nodes = all_nodes  # 最后一层保留所有节点用于预测
    
    return final_nodes, updated_node_embeddings
```

#### 3.4.2 采样机制的优势对比

| 采样类型 | 解决问题 | 主要优势 | 使用场景 |
|---------|----------|----------|----------|
| **增量节点采样** | 节点数量爆炸 | 线性复杂度，保持语义连贯性 | 所有层 (除最后层) |
| **边采样** | 边数量过多，计算效率 | 过滤无关边，专注重要连接 | 仅Transductive设置 |
| **Gumbel Softmax** | 离散采样不可微分 | 端到端训练，梯度传播 | 所有采样过程 |

## 4. 关键文件作用

### 4.1 数据处理相关文件

#### 4.1.1 `load_data.py`
**作用**: 数据预处理和邻居查找的核心文件

**主要功能**:
- `__init__()`: 加载原始数据，构建实体/关系映射
- `read_triples()`: 解析三元组文件，添加反向关系
- `load_graph()`: 构建知识图谱和邻接矩阵
- `get_neighbors()`: 🔥核心函数🔥 - 为给定节点查找所有邻居
- `get_batch()`: 获取训练/测试批次数据

**调用关系**:
```
train.py → BaseModel → DataLoader.__init__() → 数据预处理
训练循环 → GNNModel.forward() → DataLoader.get_neighbors() → 邻居查找
```

#### 4.1.2 数据文件格式
```
data/
├── entities.txt          # 实体映射: "Barack_Obama\t0"
├── relations.txt         # 关系映射: "born_in\t5" 
├── train.txt            # 训练三元组: "Barack_Obama\tborn_in\tHawaii"
├── valid.txt            # 验证三元组
└── test.txt             # 测试三元组
```

### 4.2 模型结构相关文件

#### 4.2.1 `models.py`
**作用**: 核心GNN模型实现

**主要类**:
- `GNNLayer`: 单层GNN，实现消息传递和采样
  - `forward()`: 🔥核心方法🔥 - 消息传递 + 注意力 + 采样
- `GNNModel`: 多层GNN模型
  - `forward()`: 模型主入口，控制多层传播流程
  - `soft_to_hard()`: (仅inductive) 采样实现

**关键创新实现位置**:
- 查询感知注意力: `GNNLayer.forward()` 注意力计算部分
- 增量采样: `GNNLayer.forward()` 节点采样部分
- Gumbel Softmax: `train()` 方法中的采样策略切换

#### 4.2.2 `base_model.py`
**作用**: 训练和评估的封装类

**主要功能**:
- `__init__()`: 模型初始化，优化器设置
- `train_batch()`: 🔥训练核心🔥 - 前向传播 + 损失计算 + 反向传播
- `evaluate()`: 模型评估，计算MRR/Hit@K指标
- `saveModelToFiles()` / `loadModel()`: 模型保存和加载

**调用关系**:
```
train.py → BaseModel.__init__() → 模型初始化
训练循环 → BaseModel.train_batch() → GNNModel.forward() → 前向传播
评估 → BaseModel.evaluate() → GNNModel.forward() → 性能评估
```

### 4.3 训练控制相关文件

#### 4.3.1 `train.py`
**作用**: 训练脚本主入口

**主要功能**:
- 参数解析和配置
- 数据集特定的超参数设置
- 训练循环控制
- 模型保存和性能记录

**执行流程**:
```python
# 1. 解析命令行参数
args = parser.parse_args()

# 2. 设置数据集特定参数
if dataset == 'family':
    opts.lr = 0.0036
    opts.n_node_topk = [opts.topk] * opts.layers
    
# 3. 初始化数据和模型
loader = DataLoader(opts)
model = BaseModel(opts, loader)

# 4. 训练循环
for epoch in range(opts.epoch):
    model.train_batch()                    # 训练一个epoch
    result_dict, out_str = model.evaluate() # 评估性能
    
    # 5. 保存最佳模型
    if v_mrr > best_v_mrr:
        model.saveModelToFiles(...)
```

#### 4.3.2 `utils.py`
**作用**: 工具函数集合

**主要功能**:
- `cal_ranks()`: 计算过滤排名
- `cal_performance()`: 计算MRR/Hit@K指标
- `checkPath()`: 创建目录
- `select_gpu()`: (inductive) 自动GPU选择

### 4.4 文件间调用关系图

```
训练启动:
train.py
    ↓
    创建DataLoader(load_data.py) → 数据预处理
    ↓
    创建BaseModel(base_model.py) → 包装GNNModel(models.py)
    ↓
    训练循环:
        BaseModel.train_batch()
            ↓
            GNNModel.forward()
                ↓
                逐层循环:
                    DataLoader.get_neighbors() → 查找邻居
                    ↓
                    GNNLayer.forward() → 消息传递+采样
                        ↓
                        🔥查询感知注意力🔥
                        🔥增量节点采样🔥  
                        🔥Gumbel Softmax采样🔥
            ↓
            损失计算 + 反向传播
    ↓
    BaseModel.evaluate() → 性能评估
        ↓
        utils.cal_ranks() → 排名计算
        ↓
        utils.cal_performance() → 指标计算
```

### 4.5 Transductive vs Inductive 文件差异

#### 4.5.1 主要差异对比

| 文件 | Transductive版本 | Inductive版本 | 主要差异 |
|------|-----------------|---------------|----------|
| `models.py` | 复杂双重采样GNNLayer | 简化版GNNLayer | 采样复杂度不同 |
| `load_data.py` | 单一实体空间 | 双实体空间处理 | 数据结构不同 |
| `train.py` | 详细参数配置 | 预设参数配置 | 配置方式不同 |
| `base_model.py` | 复杂评估逻辑 | 简化评估逻辑 | 评估复杂度不同 |

#### 4.5.2 选择使用指南

**使用Transductive版本当**:
- 测试实体在训练时已知
- 需要最高的推理性能
- 计算资源充足

**使用Inductive版本当**:
- 需要对新实体进行零样本推理
- 计算资源有限
- 快速原型开发

---

## 总结

AdaProp通过三个核心创新解决了传统GNN在知识图谱推理中的问题:

1. **查询感知的自适应传播**: 让传播过程根据具体查询任务进行调整
2. **增量采样机制**: 解决节点数量爆炸问题，保持线性复杂度
3. **可微分离散采样**: 使用Gumbel Softmax实现端到端训练

这些创新通过精心设计的采样策略协同工作，在保持高效性的同时提供了强大的推理能力。完整的数据流程从原始文本数据开始，经过预处理、多层传播、自适应采样，最终输出每个候选实体的预测分数，整个过程都是可微分和端到端训练的。
