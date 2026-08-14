# 实现计划：弹性工具箱（超时 / 重试 / 循环守卫）

- 基于设计：`docs/superpowers/specs/2026-08-02-resilience-design.md`
- 目标：新建 `core/resilience.py` 提供超时 + 重试 + 循环守卫，接入全部 6 处裸 LLM 调用与工具链路，不改变现有降级语义。

## 文件清单

| 动作 | 文件 |
|---|---|
| 新建 | `core/resilience.py` |
| 新建 | `tests/test_resilience.py` |
| 修改 | `agents/agent_orchestrator.py`（`BaseAgent._call_llm` + `run` 链路预算） |
| 修改 | `core/intent_recognizer.py`（`_llm_recognize`、`_extract_entities`） |
| 修改 | `mcp/tool_manager.py`（`call`、`rewrite_query`、`_rerank`、`ToolStats.retried`） |
| 修改 | `evaluation/evaluator.py`（`LLMJudge.judge`） |

## 任务分解（按依赖排序）

### T1 新建 `core/resilience.py`（无依赖，最先做）

内容（零依赖，仅 `asyncio`/`random`/`logging` + `anthropic` 异常引用）：

```python
LLM_TIMEOUT = 20.0                # 单次 LLM 调用超时
REQUEST_TOTAL_TIMEOUT = 60.0      # 整条链路预算

RETRYABLE_EXCEPTIONS = (
    asyncio.TimeoutError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

class ResilienceError(Exception): ...

async def call_with_timeout(coro_factory, timeout=LLM_TIMEOUT): ...
#   return await asyncio.wait_for(coro_factory(), timeout=timeout)

async def with_retry(coro_factory, *, attempts=3, base_delay=0.5,
                     max_delay=8.0, on_failure=None, timeout=LLM_TIMEOUT): ...
#   for attempt in range(attempts):
#       try:      return await call_with_timeout(coro_factory, timeout=timeout)
#       except RETRYABLE_EXCEPTIONS as ex:
#           on_failure(ex, attempt) 若提供
#           attempts 耗尽 → raise ResilienceError
#           delay = min(base_delay * 2**attempt, max_delay) * (0.5 + random.random())
#           await asyncio.sleep(delay)
#       except asyncio.CancelledError: raise        # 绝不吞
#       except Exception: raise                     # 永久错误不重试

class ToolLoopGuard:
    # max_iterations=10, max_same_call=3
    # begin() -> bool 是否允许新一轮
    # should_stop() -> bool
    # record_call(tool_name, params_hash) -> bool  False=同参重复超限
```

### T2 新建 `tests/test_resilience.py`（可与 T1 同批做，T1 完成后可跑）

用 pytest + unittest.mock。构造 anthropic 异常用 `_FakeResponse(status_code, headers={}, request=None)`：
- **瞬态重试**：handler 抛 `RateLimitError` 两次、第三次返回 "ok" → mock `asyncio.sleep` → 断言 handler 恰被调 3 次、返回 "ok"。
- **永久不重试**：handler 抛 `BadRequestError` → 断言只调 1 次、异常原样抛出。
- **退避递增**：mock sleep 记录间隔 → 断言 `d1 < d2`。
- **超时**：handler sleep 1s、timeout=0.01 → `call_with_timeout` 抛 `asyncio.TimeoutError`；`with_retry` 抛 `ResilienceError`。
- **CancelledError 穿透**：handler 抛 `asyncio.CancelledError` → 断言原样抛出、未重试。
- **ToolLoopGuard**：同参第 4 次 `record_call` 返回 False；`iterations` 达上限后 `should_stop()==True`。

### T3 接入 `BaseAgent._call_llm`（agent_orchestrator.py）

把 `resp = await self._client.messages.create(...)` 换成：

```python
resp = await with_retry(lambda: self._client.messages.create(...))
```

`with_retry` 抛 `ResilienceError` → 被 `BaseAgent.handle` 现有 `except Exception` 捕获 → 走失败 AgentResponse → orchestrator 降级 GeneralAgent。无需改动 handle。

### T4 接入 `core/intent_recognizer.py`（2 处）

- `_llm_recognize`：`self.client.messages.create` 外包 `with_retry`。`ResilienceError` 被现有 `except Exception` 捕获 → 返回 `{failed: True, ...}` → embedding/pattern 兜底接管（现状已如此）。
- `_extract_entities`：同样外包；失败返回空实体字典（现状已如此）。

### T5 接入 `mcp/tool_manager.py`（3 处 + 统计）

- `ToolStats` 新增字段 `retried: int = 0`。
- `call`：把 `asyncio.wait_for(tool.handler(...), timeout=tool.timeout_s)` 替换为：

```python
data = await with_retry(
    lambda: tool.handler(params, context),
    timeout=tool.timeout_s,
    on_failure=lambda ex, attempt: (tool.breaker.record_failure(),
                                     setattr(tool.stats, "retried", tool.stats.retried + 1)),
)
```

  - 原 `except asyncio.TimeoutError` 分支变为死代码（timeout 已内聚进 with_retry），删除。
  - `ResilienceError` 被现有 `except Exception` 捕获 → `_fallback_result` 降级（语义不变）。
  - 熔断衔接：重试期 `on_failure` 每次累加 `record_failure()`，耗尽后再走 except 补记一次，确保持续失败必然触发 OPEN。
- `rewrite_query`、`_rerank`：`messages.create` 外包 `with_retry`，失败沿用现有降级（原始查询 / 原始顺序）。

### T6 接入 `evaluation/evaluator.py` `LLMJudge.judge`

`messages.create` 外包 `with_retry`；`ResilienceError` 被现有 `except Exception` 捕获 → `judge_failed=True` 回退 0.5 分（语义不变）。

### T7 `AgentOrchestrator.run` 链路预算

- 把现有 `run` 主体搬入私有 `async def _run(self, req)`。
- `run` 变为：

```python
try:
    return await asyncio.wait_for(self._run(req), timeout=REQUEST_TOTAL_TIMEOUT)
except asyncio.TimeoutError:
    return OrchestratorResult(req.request_id, "抱歉，处理超时，请稍后重试。",
                              AgentType.GENERAL, req.intent, escalated=True,
                              latency_ms=REQUEST_TOTAL_TIMEOUT * 1000)
except asyncio.CancelledError:
    raise
```

### T8 全量验证

- `python -m pytest tests/ -q`（新增 6 用例 + 现有 test_intent / test_debug 不回归）
- `python -m pyflakes core/ agents/ mcp/ evaluation/`（如已装；否则 `python -m compileall`）
- 手工冒烟：`python api/main.py --cli` 走通一次对话 + `POST /eval/run`（若环境可连 API）

## 验证步骤（端到端）

1. `pytest tests/test_resilience.py` 全绿。
2. `pytest tests/` 全量不回归。
3. CLI 冒烟：正常对话返回；`/health` 的 agent stats 正常。

## 风险与回滚

- **风险**：`with_retry` 误把非瞬态异常当瞬态反复重试 → 通过 `RETRYABLE_EXCEPTIONS` 白名单收敛，只列 5 类已知瞬态。
- **风险**：重试放大延迟（最坏 0.5+1+2 秒 ≈ 3.5s + 请求体时长 ×3）→ 链路预算 `REQUEST_TOTAL_TIMEOUT=60s` 兜底。
- **回滚**：每个文件改动彼此独立，逐个 `git revert` 即可，T1 模块不影响其他文件编译。
