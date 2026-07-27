# Quickstart: 混合检索功能验证指南

## 前提条件

- Python 3.12+ 环境
- 项目依赖已安装（含新依赖）
- 项目可正常启动（FastAPI 服务运行中）

## 安装新增依赖

```bash
pip install rank_bm25 jieba
```

或更新 `requirements.txt`：

```bash
echo "rank_bm25==0.2.2" >> requirements.txt
echo "jieba==0.42.1" >> requirements.txt
pip install -r requirements.txt
```

## 验证场景

### 场景1: BM25关键词精确匹配

验证BM25能否检索到包含特定关键词的文档。

1. **启动服务**:
   ```bash
   uvicorn api.main:app --reload
   ```

2. **添加测试文档**（包含唯一编号）:
   ```bash
   curl -X POST http://localhost:8000/knowledge/add \
     -H "Content-Type: application/json" \
     -d '{
       "documents": [
         {"title": "订单ORD-2024-001", "content": "订单ORD-2024-001的状态是已发货。该订单包含商品：iPhone 15 Pro Max。预计2024-03-15送达。"}
       ]
     }'
   ```
   **预期结果**: 返回 `{"added": 1}`

3. **BM25混合检索 - 按精确编号搜索**:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "ORD-2024-001",
       "top_k": 5,
       "hybrid": true
     }'
   ```
   **预期结果**: 结果列表中包含"订单ORD-2024-001"文档，且排位靠前。
   **对比测试**: 设置 `"hybrid": false` 运行相同查询，纯向量检索可能无法精确匹配编号。

4. **BM25检索 - 按关键词搜索**:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "变压器故障排查流程",
       "top_k": 5,
       "hybrid": true
     }'
   ```
   **预期结果**: 虽然知识库中可能没有"变压器"相关内容，但混合检索不应崩溃，返回空结果或基于向量匹配的结果。

### 场景2: 混合检索 - RRF融合效果

验证BM25和向量检索能互为补充。

1. **添加对比测试文档**:
   ```bash
   curl -X POST http://localhost:8000/knowledge/add \
     -H "Content-Type: application/json" \
     -d '{
       "documents": [
         {"title": "API错误码说明", "content": "错误码401表示未授权访问。错误码404表示资源不存在。错误码500表示服务器内部错误。开发者可以根据错误码快速定位问题。"},
         {"title": "HTTP状态码", "content": "HTTP状态码分为五类：1xx信息、2xx成功、3xx重定向、4xx客户端错误、5xx服务器错误。常见的状态码有200、301、401、404、500等。"},
         {"title": "系统运维手册", "content": "服务器日常维护包括：检查磁盘空间、查看系统日志、监控CPU使用率、备份数据库。建议每周执行一次完整巡检。"}
       ]
     }'
   ```

2. **搜索"错误码 排查"**（混合模式）:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "错误码 排查",
       "top_k": 5,
       "hybrid": true
     }'
   ```
   **预期结果**:
   - "API错误码说明"（关键词"错误码"匹配）
   - "HTTP状态码"（语义相关，文中含"错误码"）
   - 两篇都会出现在结果中，RRF融合后合理排序

3. **对比纯向量检索**:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "错误码 排查",
       "top_k": 5,
       "hybrid": false
     }'
   ```
   **预期对比**: 混合检索对含明确关键词的文档（如"API错误码说明"）排名应高于纯向量检索。

### 场景3: BM25索引自动构建与更新

验证索引在启动和添加文档时自动维护。

1. **检查知识库状态**:
   ```bash
   curl http://localhost:8000/knowledge/stats
   ```
   **预期结果**:
   ```json
   {
     "total_documents": 6,
     "bm25_index_built": true,
     "bm25_corpus_size": 42,
     ...
   }
   ```
   `bm25_index_built` 为 `true` 表示索引已构建。
   `bm25_corpus_size` 显示文档块数量。

2. **添加新文档后再次检查**:
   ```bash
   curl -X POST http://localhost:8000/knowledge/add \
     -H "Content-Type: application/json" \
     -d '{
       "documents": [{"title": "新测试", "content": "这是新增的测试文档，包含特殊关键词BM25_TEST_KEYWORD。"}]
     }'
   ```
   ```bash
   curl http://localhost:8000/knowledge/stats
   ```
   **预期结果**: `bm25_corpus_size` 增加。

3. **验证新文档可被关键词检索**:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "BM25_TEST_KEYWORD",
       "top_k": 5,
       "hybrid": true
     }'
   ```
   **预期结果**: 新文档出现在结果中。

### 场景4: 融合策略切换

验证RRF和加权融合两种策略的不同效果。

1. **RRF融合**:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "错误码 401",
       "top_k": 5,
       "hybrid": true,
       "strategy": "rrf",
       "rrf_k": 60
     }'
   ```

2. **加权融合（向量权重0.7）**:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "错误码 401",
       "top_k": 5,
       "hybrid": true,
       "strategy": "weighted",
       "alpha": 0.7
     }'
   ```

3. **加权融合（BM25权重0.7）**:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "query": "错误码 401",
       "top_k": 5,
       "hybrid": true,
       "strategy": "weighted",
       "alpha": 0.3
     }'
   ```
   **预期结果**: 三种策略结果排序可能不同。alpha=0.3时（BM25权重大），关键词匹配的文档排名更高。

## 边界条件验证

### 空知识库测试

1. 清空知识库数据目录（注意备份）
2. 重启服务
3. 执行混合检索:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "测试", "top_k": 5, "hybrid": true}'
   ```
   **预期结果**: 返回空结果列表 `{"results": []}`，不崩溃。

### 超长查询测试

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "这是一个超长的查询语句，包含非常多的词汇来测试BM25在面对超长文本时的表现，BM25应该能够正确处理这种长查询，分词器会将每个词分离，然后计算每个词的TF-IDF值...（此处省略2000字）",
    "top_k": 3,
    "hybrid": true
  }'
```
**预期结果**: 正常返回结果，无超时或崩溃。

### 无匹配查询测试

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "xyznonexistentkeyword12345", "top_k": 5, "hybrid": true}'
```
**预期结果**: BM25返回空，fallback到纯向量检索结果或返回空列表。

## 配置参考

- BM25参数: `k1=1.5`, `b=0.75`（在`knowledge_base.py`中配置）
- RRF常数: `k=60`（默认推荐值）
- 融合策略: `"rrf"`（默认），可选 `"weighted"`

## 相关文档

- [Data Model](data-model.md) — 数据实体定义
- [KnowledgeBase Contract](contracts/knowledge_base.md) — BM25索引和混合检索接口
- [API Contract](contracts/api.md) — API端点变更
- [MCPTool Contract](contracts/mcp_tool.md) — MCP工具框架集成
