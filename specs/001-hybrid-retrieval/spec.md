# Feature Specification: 混合检索 - BM25关键词检索

**Feature Branch**: `001-hybrid-retrieval`

**Created**: 2026-07-25

**Status**: Draft

**Input**: User description: "我现在想要实现rag检索中的混合检索，使用bm25实现关键词检索"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 关键词精准匹配检索 (Priority: P1)

作为用户，我希望能通过关键词精确匹配检索到知识库中的文档，包括对特定术语、编号、代码片段的搜索，覆盖纯向量检索在这些场景下的不足。

**Why this priority**: BM25关键词检索是混合检索的基础能力，没有它就谈不上混合。这是最核心的功能。

**Independent Test**: 可以向知识库添加包含特定编号（如"ORD-2024-001"）的文档，然后通过搜索该编号验证BM25能精确匹配，而纯向量搜索可能失效。

**Acceptance Scenarios**:

1. **Given** 知识库中存在包含唯一编号"ORD-2024-001"的文档，**When** 用户搜索"ORD-2024-001"，**Then** BM25检索能返回该文档作为top结果
2. **Given** 知识库中存在包含特定术语"变压器故障排查流程"的文档，**When** 用户搜索"变压器"，**Then** BM25检索返回包含该术语的文档

---

### User Story 2 - 向量+关键词混合检索 (Priority: P1)

作为用户，我希望能同时利用语义相似度和关键词匹配进行检索，获得更全面的搜索结果。

**Why this priority**: 这是混合检索的核心价值 - 结合两种方法的优势。与Story 1同为最高优先级。

**Independent Test**: 在知识库中同时存入语义相关但关键词不同的文档，和关键词匹配但语义不同的文档，验证混合检索能同时召回两者。

**Acceptance Scenarios**:

1. **Given** 知识库中有文档A(语义相关/关键词不同)和文档B(关键词匹配/语义不同)，**When** 用户搜索关键词，**Then** 混合检索结果中同时包含A和B
2. **Given** 混合检索配置了RRF融合策略，**When** 执行搜索，**Then** BM25和向量检索的结果通过RRF算法融合排序

---

### User Story 3 - BM25索引自动维护 (Priority: P2)

作为系统管理员，我希望能自动维护BM25倒排索引，在添加/删除文档时索引同步更新。

**Why this priority**: 索引一致性确保检索结果始终反映最新的知识库状态。在核心检索功能完成后实现。

**Independent Test**: 添加新文档后，立即搜索该文档中的独特关键词，验证它能被BM25检索到。

**Acceptance Scenarios**:

1. **Given** BM25索引已构建，**When** 向知识库添加新文档，**Then** BM25索引自动增量更新，新文档可被关键词检索
2. **Given** BM25索引已构建，**When** 用户搜索文档中的关键词，**Then** BM25返回正确的相关度分数

---

### User Story 4 - 混合检索参数可配置 (Priority: P3)

作为开发者，我希望能配置混合检索的融合策略和参数，以适应不同场景。

**Why this priority**: 参数可配置性属于优化和灵活性增强，在核心功能完成后补充。

**Independent Test**: 修改融合权重参数后，执行相同搜索，验证返回结果顺序与修改前不同。

**Acceptance Scenarios**:

1. **Given** 系统提供混合检索配置接口，**When** 设置BM25权重为0.7、向量权重为0.3，**Then** 搜索时按该权重融合排序
2. **Given** 系统支持RRF和加权平均两种融合策略，**When** 切换策略，**Then** 搜索结果排序相应变化

### Edge Cases

- 当知识库为空时，BM25检索应返回空结果而不崩溃
- 当搜索查询为超长文本时，BM25应能正确处理
- 当查询词在所有文档中都不存在时，BM25返回空结果，fallback到纯向量检索
- 中文分词：BM25基于词频，中文需要正确的分词策略
- BM25索引重建过程中不影响正在进行的检索请求

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统MUST支持基于BM25算法的关键词检索，能倒排索引所有文档块
- **FR-002**: 系统MUST支持向量检索与BM25关键词检索的混合融合，默认使用RRF(Reciprocal Rank Fusion)策略
- **FR-003**: 系统MUST在知识库初始化时自动构建BM25索引
- **FR-004**: 系统MUST在添加新文档后自动更新BM25索引
- **FR-005**: 系统MUST提供一个可配置的混合检索接口，支持切换融合策略和调整权重
- **FR-006**: 系统SHOULD使用jieba分词对中文内容进行合理分词
- **FR-007**: 系统MUST在KnowledgeBase的search方法中集成混合检索能力
- **FR-008**: 系统SHOULD在MCPToolManager的search_with_rewrite管道中使用混合检索

### Key Entities

- **BM25Index**: 倒排索引数据结构，包含文档频率、词频统计、文档集合统计等信息
- **HybridSearchResult**: 混合检索结果，包含BM25得分、向量得分、融合得分和原始文档内容
- **FusionStrategy**: 融合策略配置（RRF、加权平均）
- **ChineseAnalyzer**: 中文分词器，基于jieba分词，支持中英文混合文本

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 混合检索在包含精确编号/代码的查询上的Recall@10比纯向量检索提升至少30%
- **SC-002**: BM25索引构建时间在1000个文档块内不超过5秒
- **SC-003**: 混合检索的平均响应时间不超过纯向量检索的2倍
- **SC-004**: 中文关键词检索能正确分词并返回相关结果

## Assumptions

- BM25索引维护在内存中，知识库规模在可接受范围内（预估<10万文档块）
- 使用`rank_bm25`作为BM25实现的依赖库（纯Python，无外部依赖）
- `jieba`用于中文分词（已广泛使用的Python中文分词库）
- 对于超大规模知识库，未来可考虑引入Elasticsearch等专业搜索引擎
- BM25索引在应用启动时从ChromaDB全量重建
