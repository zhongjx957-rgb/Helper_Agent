# Contract: MCPToolManager 集成

> 位置: `mcp/tool_manager.py`

## 变更点

### Tool Schema 更新

在 `api/main.py` 中注册 `knowledge_search` Tool 时，JSON Schema 新增字段：

```python
Tool(
    name="knowledge_search",
    description="搜索知识库，支持混合检索（BM25+向量）",
    handler=kb.search_handler,
    supports_rerank=True,
    cache_ttl=300,  # 5分钟内相同参数命中缓存
    schema={
        "type": "object",
        "properties": {
            "query":     {"type": "string",  "description": "搜索查询"},
            "top_k":     {"type": "integer", "description": "返回Top-K结果", "default": 5},
            "hybrid":    {"type": "boolean", "description": "启用混合检索（BM25+向量RRF融合）", "default": False},
            "strategy":  {"type": "string",  "description": "融合策略: rrf / weighted", "default": "rrf"},
            "rrf_k":     {"type": "integer", "description": "RRF平滑常数", "default": 60},
            "alpha":     {"type": "number",  "description": "加权融合中向量的权重", "default": 0.5},
        },
        "required": ["query"],
    },
)
```

### search_with_rewrite 调整

在 `search_with_rewrite()` 中调用 `knowledge_search` 时传递 `hybrid=True`：

```python
async def search_with_rewrite(
    self,
    tool_name: str,
    query: str,
    top_k: int = 5,
    context: Optional[Dict[str, Any]] = None,
    hybrid: bool = True,              # 新增参数
    strategy: str = "rrf",           # 新增参数
) -> ToolResult:
    # ... 查询改写逻辑不变 ...
    
    # 并行召回时传递混合检索参数
    tasks = [
        self.call(tool_name, {
            "query": q,
            "top_k": recall_k,
            "hybrid": hybrid,
            "strategy": strategy,
        }, context, use_cache=True)
        for q in sub_queries
    ]
    
    # ... 合并去重 + 重排逻辑不变 ...
```

### 缓存注意

混合检索结果缓存的cache key应包含 `hybrid`、`strategy` 等参数（由于在 `_cache_key()` 中使用 `json.dumps(params, sort_keys=True)` 自动包含，无需改动）。
