# Contract: KnowledgeBase 类接口

> 位置: `mcp/knowledge_base.py`

## 新增方法

### `build_bm25_index()`

构建/重建BM25倒排索引。

```python
def build_bm25_index(self) -> None:
    """
    从ChromaDB中读取所有文档块，使用jieba分词后构建BM25Okapi索引。
    
    副作用: 设置 self._bm25_index 和 self._bm25_doc_ids
    在 KnowledgeBase.__init__() 末尾自动调用。
    """
```

**输入**: 无（从ChromaDB所有文档重建）

**输出**: 无（设置内部状态）

**错误处理**:
- ChromaDB查询失败 → `logger.error` 并标记索引未就绪 (`self._bm25_built = False`)
- jieba分词失败 → 跳过该文档块，不影响其他块

---

### `hybrid_search()`

混合检索：同时执行BM25关键词检索和向量检索，融合后返回结果。

```python
def hybrid_search(
    self,
    query: str,
    top_k: int = 5,
    strategy: str = "rrf",
    rrf_k: int = 60,
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    混合检索：BM25关键词检索 + ChromaDB向量检索 → RRF/加权融合。
    
    Args:
        query:      用户查询
        top_k:      返回Top-K结果
        strategy:   融合策略 "rrf" | "weighted"
        rrf_k:      RRF平滑常数（仅RRF模式有效）
        alpha:      向量检索权重（仅加权模式有效，0-1）
    
    Returns:
        List[Dict]: [
            {
                "title":       str,   # 文档标题
                "content":     str,   # 文档片段
                "bm25_score":  float, # BM25得分
                "vector_score": float, # 向量相似度
                "fusion_score": float, # 融合得分
                "chunk":       int,   # 块索引
                "source":      str,   # "hybrid"
            },
            ...
        ]
    
    Raises:
        RuntimeError: BM25索引未构建时调用
    """
```

**行为**:
1. BM25索引未构建 → 抛出 `RuntimeError`
2. BM25检索: `query` 经 `jieba.lcut_for_search` 分词 → `bm25.get_scores()` → 全量排名
3. 向量检索: `self._collection.query(query_texts=[query], n_results=top_k * 2)` → 向量排名
4. RRF融合: 两路排名列表通过RRF计算融合分
5. 返回融合后Top-K

**RRF融合逻辑**:
```python
def _rrf_fuse(
    self,
    bm25_rankings: List[int],         # BM25检索结果索引列表（按得分降序）
    vector_rankings: List[int],       # 向量检索结果索引列表（按相似度降序）
    top_k: int,
    k: int = 60,
) -> List[Tuple[int, float]]:          # (doc_index, fusion_score)
```

**加权融合逻辑**:
```python
def _weighted_fuse(
    self,
    bm25_scores: np.ndarray,          # BM25原始得分
    vector_scores: np.ndarray,         # 向量检索原始相似度
    top_k: int,
    alpha: float = 0.5,               # 向量权重
) -> List[Tuple[int, float]]:
    # 1. BM25分数归一化到 [0, 1]
    # 2. 向量相似度已在 [0, 1] 范围
    # 3. 融合 = alpha * vector_norm + (1-alpha) * bm25_norm
```

---

### `search_handler()` 变更

```python
async def search_handler(
    self,
    params: Dict[str, Any],
    context: Any
) -> List[Dict]:
    """
    变更: 支持通过参数 hybrid=True 启用混合检索。
    
    新增参数:
        params["hybrid"]:    bool (默认False) — 是否启用混合检索
        params["strategy"]:  str (默认"rrf") — 融合策略
        params["rrf_k"]:     int (默认60)
        params["alpha"]:     float (默认0.5)
    """
    query = params.get("query", "")
    top_k = params.get("top_k", 5)
    hybrid = params.get("hybrid", False)
    if hybrid:
        return self.hybrid_search(
            query=query,
            top_k=top_k,
            strategy=params.get("strategy", "rrf"),
            rrf_k=params.get("rrf_k", 60),
            alpha=params.get("alpha", 0.5),
        )
    return self.search(query, top_k=top_k)
```

## 无变更方法

- `add_documents()` — 内部追加BM25索引更新逻辑，对外接口不变
- `search()` — 保持纯向量检索，供不需要混合检索的场景使用
- `doc_count` — 无变更
- `_chunk_text()` — 无变更
- `_load_default_docs()` — 无变更

## 新增内部属性

```python
class KnowledgeBase:
    # ... 现有属性 ...
    
    # ── BM25 相关（新增） ──
    _bm25_built: bool = False
    _bm25_index: Optional[BM25Okapi] = None        # rank_bm25实例
    _bm25_corpus: List[str] = []                    # 原始文档块列表
    _bm25_tokenized: List[List[str]] = []           # 分词后文档块列表
    _bm25_doc_ids: List[str] = []                   # ChromaDB文档ID（对应关系）
    _bm25_config: Dict[str, Any] = field(default_factory=lambda: {
        "k1": 1.5,
        "b": 0.75,
    })
```

## 依赖

```python
import jieba
from rank_bm25 import BM25Okapi
```
