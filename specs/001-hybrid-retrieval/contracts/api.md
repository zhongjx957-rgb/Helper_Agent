# Contract: API 接口

> 位置: `api/main.py`

## 新增/变更端点

### GET `/knowledge/stats` — 变更

新增混合检索状态信息到返回数据中。

**响应变更**:
```json
{
    "total_documents": 6,
    "bm25_index_built": true,
    "bm25_corpus_size": 42,
    "bm25_config": {
        "k1": 1.5,
        "b": 0.75
    }
}
```

**新增字段**:
| 字段 | 类型 | 描述 |
|------|------|------|
| `bm25_index_built` | `bool` | BM25索引是否已构建 |
| `bm25_corpus_size` | `int` | BM25索引中的文档块数量 |
| `bm25_config` | `object` | BM25索引配置参数 |

---

### POST `/search` — 变更

支持混合检索参数。

**请求体变更**:
```json
{
    "query": "退款流程",
    "top_k": 5,
    "hybrid": true,
    "strategy": "rrf",
    "rrf_k": 60,
    "alpha": 0.5
}
```

**新增字段**:
| 字段 | 类型 | 必填 | 默认值 | 描述 |
|------|------|------|--------|------|
| `hybrid` | `bool` | 否 | `false` | 启用混合检索 |
| `strategy` | `str` | 否 | `"rrf"` | 融合策略: "rrf" / "weighted" |
| `rrf_k` | `int` | 否 | `60` | RRF平滑常数 |
| `alpha` | `float` | 否 | `0.5` | 加权融合权重 |

**响应体**（无变更，内容格式与纯向量检索一致，但返回结果由混合检索产生）:
```json
{
    "results": [
        {
            "title": "退款政策",
            "content": "退款政策说明。用户在购买后...",
            "score": 0.9213,
            "chunk": 0
        }
    ]
}
```

**注意**: 现有 `knowledge_search` Tool 的 JSON Schema 也需要同步更新，新增 `hybrid`、`strategy`、`rrf_k`、`alpha` 字段定义。

---

## 内部函数变更

### `_build_knowledge_context()` — 变更

调用混合检索替代纯向量检索：

```python
async def _build_knowledge_context(message: str) -> str:
    """构建知识库上下文 - 使用混合检索提升检索质量"""
    result = await kb.search_handler(
        {"query": message, "top_k": 3, "hybrid": True},
        context=None,
    )
    # ... 其余逻辑不变 ...
```

或通过 `search_with_rewrite` 管道集成：

```python
# 在 MCPToolManager.search_with_rewrite 内部
# 将 tool.call("knowledge_search", {"query": q, "top_k": recall_k, "hybrid": True})
```
