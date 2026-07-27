# Implementation Plan: 混合检索 - BM25关键词检索

**Branch**: `001-hybrid-retrieval` | **Date**: 2026-07-25 | **Spec**: [spec.md](spec.md)

**Input**: User requested hybrid retrieval in RAG using BM25 for keyword search

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

在现有ChromaDB向量检索的基础上，增加BM25关键词检索能力，实现向量+关键词的混合检索。核心思路是使用`rank_bm25`库构建倒排索引，`jieba`做中文分词，通过RRF(Reciprocal Rank Fusion)融合两种检索结果。扩展`KnowledgeBase`类集成混合检索能力，并在`MCPToolManager`的检索管道中使用。

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**: 新增 `rank_bm25` (纯Python BM25实现)、`jieba` (中文分词)、现有 `chromadb==0.5.23`、`anthropic==0.40.0`、`fastapi==0.115.5`

**Storage**: ChromaDB (向量存储 + 文档块存储)、BM25索引维护在内存中，每次启动时从ChromaDB重建

**Testing**: pytest (项目现有框架)。需要单元测试(BM25索引构建/检索/增量更新)、集成测试(混合检索管道端到端评估)

**Target Platform**: Linux服务器 (Docker部署，当前项目架构)

**Project Type**: Web服务 (FastAPI) + 智能Agent系统 (MCP架构)

**Performance Goals**: BM25索引构建耗时：1000个文档块 < 5秒；单次混合检索 < 200ms；内存占用 < 200MB (BM25倒排索引)

**Constraints**: BM25索引全量在内存中维护（非持久化），应用重启时从ChromaDB重建；需支持增量更新（添加文档后同步更新索引）

**Scale/Scope**: 当前知识库规模预估 < 10万文档块；BM25索引内存方案在可接受范围内（若未来规模超出预期，可考虑Elasticsearch方案）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### 初始评估 (Phase 0 前)

根据项目的`.specify/memory/constitution.md`，当前constitution为模板占位符，未设置具体约束门禁。默认通过，无违规项。

- **Gate 1 - 技术选型**: ✅ `rank_bm25` + `jieba` 组合是Python生态中最轻量的BM25中文方案
- **Gate 2 - 架构衔接**: ✅ 扩展`KnowledgeBase`类，新增`hybrid_search`方法，与现有`search`方法并存，不影响已有功能
- **Gate 3 - 性能**: ✅ BM25全内存索引，性能可接受；RRF融合计算轻量

### 设计后重新评估 (Phase 1 后)

| 门禁 | 状态 | 说明 |
|------|------|------|
| 技术选型 | ✅ 通过 | research.md确认了`rank_bm25` + `jieba`是最佳方案，MTEB评测支持jieba分词BM25在中文检索上的有效性 |
| 架构衔接 | ✅ 通过 | 设计采用`hybrid_search`作为独立方法，不修改`search()`的现有行为；contracts明确定义了接口变更范围，向后兼容 |
| 性能 | ✅ 通过 | BM25索引构建 O(N) 复杂度（N=文档块数），RRF融合 O(NlogN)，均在可接受范围内 |
| 数据模型完整性 | ✅ 通过 | data-model.md明确定义了BM25Index、HybridSearchResult、FusionConfig三大实体及状态流转 |
| 接口完整性 | ✅ 通过 | contracts/覆盖了KnowledgeBase类、API端点、MCPTool集成三个层面的接口变更，无遗漏 |

**结论**: ✅ 所有门禁通过，设计合规，可以进入实现阶段。

## Project Structure

### Documentation (this feature)

```text
specs/001-hybrid-retrieval/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
# Single project layout — extends existing files, no structural changes needed
mcp/
├── knowledge_base.py    # 主要修改: 新增BM25索引构建、hybrid_search方法
├── tool_manager.py      # 修改: search_with_rewrite管道中集成混合检索
api/
├── main.py              # 修改: 新增/扩展统计端点以显示混合检索状态
requirements.txt         # 修改: 添加 rank_bm25, jieba
```

**Structure Decision**: 本项目为单项目布局。所有修改集中在`mcp/knowledge_base.py`（核心BM25逻辑）、`mcp/tool_manager.py`（检索管道集成）和`api/main.py`（API暴露）。无需新增目录或独立服务。

## Complexity Tracking

当前Constitution无约束门禁，无需复杂度跟踪。
