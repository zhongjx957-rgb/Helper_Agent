---

description: "Task list for hybrid retrieval with BM25 implementation"
---

# Tasks: 混合检索 - BM25关键词检索

**Input**: Design documents from `specs/001-hybrid-retrieval/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization — install new dependencies

- [X] T001 Add `rank_bm25>=0.2.2` and `jieba>=0.42.1` to `requirements.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core BM25 infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 [P] Implement BM25 index building and Chinese tokenizer in `mcp/knowledge_base.py` — add `_init_bm25_index()` method that reads all ChromaDB documents, tokenizes with jieba, and builds `BM25Okapi` index
- [X] T003 [P] Implement BM25 search method in `mcp/knowledge_base.py` — add `_bm25_search()` internal method that tokenizes query, calls `bm25.get_scores()`, and returns ranked results with doc_ids and scores
- [X] T004 [P] Implement BM25 config storage in `mcp/knowledge_base.py` — add `_bm25_built` flag, `_bm25_index`, `_bm25_corpus`, `_bm25_tokenized`, `_bm25_doc_ids` attributes, and `_bm25_config` dict with k1=1.5, b=0.75
- [X] T005 Call `_init_bm25_index()` at end of `KnowledgeBase.__init__()` in `mcp/knowledge_base.py` so BM25 index auto-builds on startup

**Checkpoint**: Foundation ready — BM25 index builds, retrieves keyword results, content is accessible

---

## Phase 3: User Story 1 — 关键词精准匹配检索 (Priority: P1) 🎯 MVP

**Goal**: BM25关键词检索的核心能力：用户可通过关键词精确匹配检索到知识库中的文档，包括特定术语、编号、代码片段

**Independent Test**: 向知识库添加包含唯一编号"ORD-2024-001"的文档，搜索该编号，验证BM25能精确匹配返回。与纯向量检索对比，混合检索结果中应包含该编号文档且排名更靠前。

### Implementation for User Story 1

- [X] T006 [P] [US1] Add imports for `jieba` and `BM25Okapi` at top of `mcp/knowledge_base.py`
- [X] T007 [P] [US1] Implement RRF fusion function `_rrf_fuse()` in `mcp/knowledge_base.py` — accepts two ranked lists and k constant, returns fused ranking
- [X] T008 [P] [US1] Implement weighted fusion function `_weighted_fuse()` in `mcp/knowledge_base.py` — accepts normalized scores and alpha, returns fused ranking
- [X] T009 [US1] Implement `hybrid_search()` method in `mcp/knowledge_base.py` — orchestrates BM25 search + vector search + fusion, returns unified `HybridSearchResult` list with bm25_score, vector_score, fusion_score fields
- [X] T010 [US1] Update `search_handler()` in `mcp/knowledge_base.py` to accept and pass `hybrid`, `strategy`, `rrf_k`, `alpha` parameters with defaults
- [X] T011 [US1] Update `knowledge_search` Tool schema in `api/main.py` — add `hybrid`, `strategy`, `rrf_k`, `alpha` fields to JSON Schema properties

**Checkpoint**: User Story 1 complete — BM25 keyword search works standalone, can retrieve documents by exact keyword match

---

## Phase 4: User Story 2 — 向量+关键词混合检索 (Priority: P1)

**Goal**: 同时利用语义相似度和关键词匹配进行检索，通过RRF融合获取更全面的搜索结果

**Independent Test**: 存入语义相关但关键词不同的文档A（如"苹果发布新手机"）和关键词匹配但语义不同的文档B（如"水果苹果的营养价值"），搜索"苹果手机"验证混合检索同时召回A和B

### Implementation for User Story 2

- [X] T012 [P] [US2] Integrate hybrid search into `search_with_rewrite()` in `mcp/tool_manager.py` — pass `hybrid=True` and `strategy="rrf"` when calling `knowledge_search` tool in parallel recall tasks
- [X] T013 [US2] Update `_build_knowledge_context()` in `api/main.py` to use hybrid search (hybrid=True) for chat context retrieval
- [X] T014 [US2] Update `/search` endpoint in `api/main.py` — add `hybrid`, `strategy`, `rrf_k`, `alpha` query/body parameters to the search request model
- [X] T015 [US2] Add BM25 stats to `/knowledge/stats` endpoint in `api/main.py` — expose `bm25_index_built`, `bm25_corpus_size`, `bm25_config` fields

**Checkpoint**: User Story 2 complete — hybrid retrieval works end-to-end, chat uses BM25+vector fusion, stats endpoint shows BM25 status

---

## Phase 5: User Story 3 — BM25索引自动维护 (Priority: P2)

**Goal**: 在添加文档时，BM25倒排索引自动增量更新，确保新文档立即可被关键词检索到

**Independent Test**: 添加含关键词"BM25_TEST_KEYWORD"的新文档，立即搜索该关键词，验证新文档出现在结果中

### Implementation for User Story 3

- [X] T016 [US3] Add BM25 index update logic inside `add_documents()` in `mcp/knowledge_base.py` — after ChromaDB insert, reinitialize BM25 index from updated corpus
- [X] T017 [US3] Handle empty corpus gracefully in `mcp/knowledge_base.py` — `_init_bm25_index()` should check if corpus is empty and set `_bm25_built=False` without crashing

**Checkpoint**: User Story 3 complete — BM25 index automatically reflects new documents, edge cases handled

---

## Phase 6: User Story 4 — 混合检索参数可配置 (Priority: P3)

**Goal**: 开发者可通过参数配置融合策略（RRF/加权平均）和权重，适应不同检索场景

**Independent Test**: 设置BM25权重0.7、向量权重0.3执行搜索，与默认RRF策略的搜索结果排序进行对比，验证排序结果不同

### Implementation for User Story 4

- [X] T018 [US4] Add `FusionStrategy` enum (`RRF` / `WEIGHTED`) in `mcp/knowledge_base.py`
- [X] T019 [US4] Wire up strategy switching in `hybrid_search()` — route to `_rrf_fuse()` or `_weighted_fuse()` based on strategy parameter
- [X] T020 [US4] Normalize BM25 scores to [0,1] range in `_weighted_fuse()` in `mcp/knowledge_base.py` using min-max normalization

**Checkpoint**: User Story 4 complete — fusion strategy and weights are configurable at search time

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T021 [P] Add error handling for BM25 index rebuild failures in `mcp/knowledge_base.py` — wrap rebuild in try/except, log errors, set `_bm25_built=False`
- [X] T022 [P] Add Chinese custom dictionary support path in `mcp/knowledge_base.py` — placeholder `jieba.load_userdict()` call with configurable dict path
- [X] T023 Handle fallback: when BM25 returns no matches in `hybrid_search()`, return pure vector search results instead of empty list in `mcp/knowledge_base.py`
- [X] T024 Log BM25 index stats on startup and on each `hybrid_search()` call in `mcp/knowledge_base.py` — log corpus size, top scores, fusion strategy
- [X] T025 Run [quickstart.md](quickstart.md) validation — execute all scenarios to verify end-to-end correctness

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - US1 (Phase 3) and US2 (Phase 4) can proceed in parallel (different files)
  - US3 (Phase 5) depends on BM25 core (US1)
  - US4 (Phase 6) depends on hybrid_search methods (US1, US2)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **US1 - 关键词精准匹配检索 (P1)**: Can start after Foundation — No dependency on other stories
- **US2 - 混合检索 (P1)**: Depends on US1 (needs `hybrid_search()`) — but `search_with_rewrite` integration can proceed in parallel
- **US3 - 索引自动维护 (P2)**: Depends on US1 (needs `_init_bm25_index()`)
- **US4 - 参数可配置 (P3)**: Depends on US1 (needs `hybrid_search()` and fusion functions)

### Within Each User Story

- Core infrastructure before integration
- Story complete before moving to next priority

### Parallel Opportunities

- T002, T003, T004 (Phase 2 Foundational) can run in parallel
- T006, T007, T008 (Phase 3 US1) can run in parallel
- T012 (US2 tool_manager.py) can run in parallel with T013-T015 (US2 api/main.py)
- T021, T022 (Phase 7) can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all BM25 index constructors together:
Task: "Implement BM25 index building in KnowledgeBase._init_bm25_index()"
Task: "Implement BM25 search in KnowledgeBase._bm25_search()"
Task: "Implement BM25 config storage attributes"

# Launch fusion functions together:
Task: "Implement RRF fusion _rrf_fuse()"
Task: "Implement weighted fusion _weighted_fuse()"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 3: User Story 1 (关键词精准匹配检索)
4. **STOP and VALIDATE**: Test User Story 1 independently
   - Add test doc with "ORD-2024-001" → search → verify BM25 match
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready (BM25 builds on startup)
2. Add US1 (关键词精准匹配) → Test independently → **MVP achieved!**
3. Add US2 (混合检索) → Test independently → RRF fusion works end-to-end
4. Add US3 (索引自动维护) → Test independently → Index auto-updates on add
5. Add US4 (参数可配置) → Test independently → Strategy/weights configurable
6. Phase 7 (Polish) → Error handling, logging, custom dict support

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
