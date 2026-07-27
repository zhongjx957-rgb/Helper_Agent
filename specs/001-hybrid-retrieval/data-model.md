# Data Model: 混合检索 - BM25关键词检索

## 核心实体

### BM25Index

BM25索引层，对知识库中所有文档块构建的倒排索引。

| 字段 | 类型 | 描述 |
|------|------|------|
| `corpus` | `List[str]` | 原始文档块文本列表，与ChromaDB中存储的内容一一对应 |
| `tokenized_corpus` | `List[List[str]]` | 经jieba分词后的文档块Token列表 |
| `bm25` | `BM25Okapi` | rank_bm25库的BM25Okapi实例 |
| `k1` | `float` | BM25词频饱和参数 (默认1.5) |
| `b` | `float` | BM25长度归一化参数 (默认0.75) |
| `doc_ids` | `List[str]` | 对应ChromaDB中每个文档块的ID |
| `is_built` | `bool` | 索引是否已构建 |

**关系**:
- BM25Index直接对应ChromaDB `knowledge_base` collection中的所有文档块
- `doc_ids` 数组索引位置与ChromaDB的ID顺序一致，用于两路结果对齐

**构建流程**:
```
ChromaDB所有文档块 → jieba.lcut_for_search分词 → BM25Okapi初始化 → 索引就绪
```

**增量更新**:
- `add_documents(chunks)`: 新块分词后追加到 `tokenized_corpus`，`doc_ids`扩展，重新构造BM25Okapi
- `remove_document(doc_id)`: 对应位置从列表中移除后重建

---

### HybridSearchResult

混合检索返回的单个结果项。

| 字段 | 类型 | 描述 |
|------|------|------|
| `title` | `str` | 文档标题 |
| `content` | `str` | 文档片段内容 |
| `bm25_score` | `float` | BM25原始得分 |
| `vector_score` | `float` | 向量检索相似度得分 (1.0 - ChromaDB distance) |
| `fusion_score` | `float` | 融合后的最终得分 (RRF或加权) |
| `chunk` | `int` | 文档块索引 |
| `source` | `str` | 检索来源: "bm25" / "vector" / "hybrid" |

---

### FusionConfig

融合策略配置。

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `strategy` | `FusionStrategy` | `RRF` | 融合策略: RRF / WEIGHTED |
| `rrf_k` | `int` | `60` | RRF平滑常数 |
| `alpha` | `float` | `0.5` | 加权融合时向量检索的权重 (1-alpha为BM25权重) |

```python
from enum import Enum

class FusionStrategy(str, Enum):
    RRF = "rrf"           # Reciprocal Rank Fusion (默认)
    WEIGHTED = "weighted" # 加权平均融合
```

---

## 状态流转

### 索引生命周期

```
[系统启动]
    │
    ├─── 加载ChromaDB → 读取所有文档块
    │
    ├─── BM25Index.build(corpus, tokenizer)
    │        │
    │        ├── jieba.lcut_for_search → tokenized_corpus
    │        ├── BM25Okapi(tokenized_corpus)
    │        └── 状态: READY
    │
    ├─── [文档添加]
    │        new_chunks → jieba分词 → 追加语料 → 重建BM25Okapi
    │
    └─── [文档删除]
             过滤语料 → 重建BM25Okapi
```

### 混合检索流程

```
[用户查询]
    │
    ├─── BM25检索 ─────────────────────────┐
    │    query → jieba分词                  │
    │    → BM25Okapi.get_scores()          │
    │    → BM25排名列表                     │
    │                                      │
    ├─── 向量检索 ─────────────────────────┤
    │    query → ChromaDB.query()          │
    │    → 语义相似度排名列表                │
    │                                      │
    ├─── 融合 ─────────────────────────────┤
    │    两路排名 → FusionStrategy.apply() │
    │    (RRF / Weighted)                  │
    │    → 融合排名列表                     │
    │                                      │
    └─── 返回 Top-K ──────────────────────┘
```

---

## 验证规则

| 规则 | 说明 |
|------|------|
| k1 > 0 | BM25词频饱和参数必须为正数 |
| 0 ≤ b ≤ 1 | BM25长度归一化参数必须在 [0, 1] 范围 |
| 0 ≤ alpha ≤ 1 | 加权融合权重必须在 [0, 1] 范围 |
| rrf_k ≥ 1 | RRF平滑常数必须 ≥ 1 |
| top_k ≥ 1 | 返回结果数必须 ≥ 1 |
| corpus非空 | BM25索引至少需要一个文档块 |
| 分词一致性 | 查询和文档必须使用相同的分词器 |

---

## 与现有实体的关系

```
KnowledgeBase (existing)
    │
    ├── ChromaDB collection "knowledge_base"
    │       ├── 文档存储 (documents, metadatas, embeddings)
    │       └── 向量检索: query() → 语义匹配
    │
    └── BM25Index (新增)
            ├── tokenized_corpus (分词后的文档块)
            ├── BM25Okapi实例
            └── 关键词检索: get_scores() → BM25匹配

MCPToolManager (existing)
    │
    ├── search_with_rewrite()
    │       └── 内部调用KnowledgeBase.hybrid_search() → RRF融合
    │
    └── _rerank()
            └── 可选：对混合检索结果进一步用LLM重排
```
