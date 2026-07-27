# Research: 混合检索 - BM25关键词检索

## 技术决策与依据

### 1. BM25库选型

- **Decision**: 使用 `rank_bm25` 库的 `BM25Okapi` 实现
- **Rationale**:
  - 纯Python实现，零外部依赖，与项目现有的轻量架构一致
  - 提供5种BM25变体（BM25Okapi / BM25L / BM25+ / BM25-Adpt / BM25T），Okapi是行业标准
  - API简洁：`BM25Okapi(tokenized_corpus)` → `get_scores()` / `get_top_n()`
  - PyPI下载量高，社区验证充分
- **Alternatives considered**:
  - `whoosh`: 功能更全（索引持久化），但依赖较重，本项目BM25索引可内存重建
  - `elasticsearch`: 分布式搜索引擎，超出当前规模需求
  - 自行实现BM25: 重复造轮子，无必要

### 2. 中文分词方案

- **Decision**: 使用 `jieba` 的 `lcut_for_search` 模式进行分词
- **Rationale**:
  - `jieba` 是Python中文分词的事实标准，成熟稳定
  - `lcut_for_search` 模式比 `lcut` 更适合检索场景：会生成更细粒度的词组合，提高短查询召回率
  - 支持自定义词典 (`jieba.load_userdict()`)，可用于领域专业术语
- **Alternatives considered**:
  - `pkuseg`: 北京大学分词器，精度更高但速度慢
  - `hanlp`: 功能全面但依赖重
  - 字符级n-gram: 无需分词但BM25效果差（MTEB评测显示分词BM25 NDCG@10 0.641 vs 字符级 0.567）

### 3. 融合策略

- **Decision**: 默认使用RRF (Reciprocal Rank Fusion) 融合策略，k=60
- **Rationale**:
  - RRF只依赖排位不依赖绝对分数，规避了BM25分数和向量相似度量纲不统一的问题
  - 行业内公认的零配置融合方案（"gold industry standard"）
  - 实现简单、可解释性强
  - k=60 是业界的推荐默认值，有效平滑top排名的优势
- **Alternatives considered**:
  - 加权平均融合: 需要对两路分数分别归一化，引入额外的归一化开销和精度损失，作为备选方案提供
  - Convex融合: 需要训练数据调参，复杂度高
  - 仅intersection/union: 丢失排序信息

### 4. 索引管理策略

- **Decision**: BM25索引全在内存中维护，启动时从ChromaDB全量重建
- **Rationale**:
  - 当前预估知识库规模 < 10万文档块，rank_bm25的内存索引完全可接受
  - 避免持久化逻辑的复杂性
  - ChromaDB已有的文档块是单一数据源，保证一致性
  - 增量更新：添加文档时同步更新BM25索引
- **Alternatives considered**:
  - 序列化BM25索引到磁盘: 加快启动速度，但增加了持久化逻辑的复杂度
  - 使用Elasticsearch: 适合超大规模，但引入额外的运维复杂性

### 5. BM25参数

- **Decision**: 使用默认参数 k1=1.5, b=0.75
- **Rationale**:
  - `k1=1.5`: 控制词频饱和，值越大高频词对分数影响越大。默认值适合大多数场景
  - `b=0.75`: 文档长度归一化，惩罚过长文档。0.75是经验最佳值
- **备选**: 留出配置接口，后续可根据实际效果微调

## 实施指引

### API设计参考

```
BM25索引构建:
  输入: List[str] (文档块列表)
  过程: jieba.lcut_for_search → BM25Okapi.fit()
  输出: BM25Okapi实例

混合检索:
  输入: query, top_k, fusion_strategy, alpha
  过程: 
    1. BM25召回 → 获取rankings
    2. 向量召回 → 获取rankings  
    3. RRF/加权融合
  输出: List[(doc_id, content, fusion_score)]

索引更新:
  add_document(document_chunks):
    分词后调用 BM25Okapi 无直接增量API
    方案: 维护已分词语料列表，每次新增时重新构造BM25Okapi
  remove_document(doc_id):
    过滤语料列表后重建BM25Okapi
```

### ChromaDB集成

`knowledge_base.py`当前使用ChromaDB查询：
```python
chroma_collection.query(query_texts=[query], n_results=top_k)
```
混合检索策略：
1. 混合检索时，同时执行 `chroma_collection.query()`（向量搜索）和 `bm25.get_scores()`（BM25搜索）
2. 两路结果通过RRF融合
3. 返回融合后的Top-K结果

### 依赖清单

```
rank_bm25==0.2.2    # BM25算法实现
jieba==0.42.1       # 中文分词
```

*注: 验证实际的最新版本号后更新requirements.txt*

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| BM25索引内存占用随文档增长 | 监控内存使用，设置告警阈值；备选方案：Elasticsearch |
| jieba分词对专业术语效果差 | 支持自定义词典加载 |
| 重建索引期间影响检索 | 延迟重建：新索引构建完成后再原子替换引用 |
| 两路检索速度不一致 | 向量检索是主要的性能瓶颈，BM25检索效率高，整体延迟可控 |
