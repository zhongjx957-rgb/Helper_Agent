# EchoMind 弹性工具箱设计（超时 / 重试 / 循环守卫）

- 日期：2026-08-02
- 范围：`core/resilience.py` 及其在 agent / intent / tool / judge 链路的接入
- 定位：学习/作品集项目 —— 零依赖、自研、可讲解

## 背景与动机

EchoMind 的故障处理目前只有两块：

1. **熔断器（CircuitBreaker）**—— 位于 `mcp/tool_manager.py`，处理"持续故障防雪崩"。
2. **降级（Fallback）**—— 工具 fallback、Agent 降级到 GeneralAgent、意图识别 embedding/pattern 兜底。

缺口：

- **全部 6 处 LLM 调用无超时**（`messages.create()` 可能永久挂起）：
  `BaseAgent._call_llm`、`IntentRecognizer._llm_recognize`、`IntentRecognizer._extract_entities`、
  `MCPToolManager.rewrite_query`、`MCPToolManager._rerank`、`LLMJudge.judge`。
- **零重试**：一次瞬态网络抖动直接整单失败，且会累计 `consecutive_fails` 误触熔断。
- **无工具循环守卫**：模型反复调用工具不收敛（未来 function calling 场景）。

熔断器只覆盖四分之一。它不是重复的轮子，而是与下列机制形成故障处理流水线：

```
调用发出 → 【超时】判断"这次到底失败没？"   ← 没有它，挂了也感知不到
         → 【重试】判断"要不要再试一次？"     ← 吸收瞬态，不让小抖变事故
         → 【熔断器】判断"这个下游还值得发吗？" ← 已有✅，拦持续故障防雪崩
         → 【循环守卫】判断"模型在收敛吗？"     ← 限制总轮次，防空转烧钱
```

## 架构

### 新增文件：`core/resilience.py`

纯标准库（`asyncio` / `random` / `logging`），零依赖。

```
core/resilience.py
├── 常量
│   LLM_TIMEOUT = 20.0            # 单次 LLM 调用超时（秒）
│   REQUEST_TOTAL_TIMEOUT = 60.0  # 整条链路预算（秒）
├── async def call_with_timeout(coro_factory, timeout) -> Any
│   # 用 asyncio.wait_for 包一层，超时抛 asyncio.TimeoutError
├── async def with_retry(coro_factory, *, attempts=3, base_delay=0.5,
│                        max_delay=8.0, on_failure=None) -> Any
│   # 指数退避+抖动，只重试瞬态错误；on_failure 用于向熔断器报数
├── class ToolLoopGuard            # 写好但暂不接入
│   def __init__(self, max_iterations=10, max_same_call=3)
│   def record_call(self, tool_name, params_hash) -> bool  # False=重复
│   def should_stop(self) -> bool
└── class ResilienceError(Exception)  # 重试耗尽后的统一异常
```

关键设计点：

- **`coro_factory` 用工厂而非协程对象**：`messages.create()` 每次重试都要重新构造请求，
  协程对象只能 await 一次。
- **`call_with_timeout` 与 `with_retry` 组合**：`with_retry(lambda: call_with_timeout(...))`，
  超时被归类为瞬态错误参与重试。

## 重试规则：只重试瞬态错误

| 异常类型 | 重试? | 说明 |
|---|---|---|
| `asyncio.TimeoutError` | ✅ | 挂起/超时是典型的瞬态 |
| `anthropic.APIConnectionError` | ✅ | 网络抖动 |
| `anthropic.APITimeoutError` | ✅ | SDK 自身读超时 |
| `anthropic.RateLimitError` | ✅ | 限流，退避后重试最有效 |
| `anthropic.InternalServerError` | ✅ | 5xx |
| `anthropic.AuthenticationError` | ❌ | 永久错误 |
| `anthropic.BadRequestError` | ❌ | 参数/请求体错误 |
| 其他 `APIStatusError`(4xx) | ❌ | 同上 |

退避公式：

```python
delay = min(base_delay * 2 ** attempt, max_delay) * (0.5 + random.random())
```

指数增长 + 0.5~1.5 抖动，避免同一时刻重试的请求再次撞车。

## 接入点

| 接入位置 | 做法 |
|---|---|
| `BaseAgent._call_llm`（agent_orchestrator.py:148） | `messages.create` 外包 `with_retry(call_with_timeout(...))` |
| `IntentRecognizer._llm_recognize`（intent_recognizer.py:230） | 同上；重试耗尽返回 `failed=True`，embedding/pattern 兜底接管 |
| `IntentRecognizer._extract_entities`（intent_recognizer.py:328） | 同上；失败返回空实体 |
| `MCPToolManager.rewrite_query`（tool_manager.py:333） | 同上；失败降级为原始查询 |
| `MCPToolManager._rerank`（tool_manager.py:422） | 同上；失败降级为原始顺序 |
| `MCPToolManager.call`（tool_manager.py:190） | 在 `wait_for` 外层加重试，`on_failure=tool.breaker.record_failure` —— 重试喂养熔断器的衔接点 |
| `LLMJudge.judge`（evaluator.py:125） | 同上；失败回退 `judge_failed=True` / 0.5 分 |
| `AgentOrchestrator.run`（agent_orchestrator.py:261） | 整个 `run` 包 `REQUEST_TOTAL_TIMEOUT`，链路级兜底 |

原则：**所有失败都走现有降级路径**，新增代码不改变行为语义，只改变"失败前的耐受力"。

## 错误处理

- 超时 → 归为瞬态 → 重试；重试耗尽 → 抛 `ResilienceError` → 上层按现有逻辑降级。
- `asyncio.CancelledError` 绝不吞掉，原样重新抛出（`wait_for` 取消依赖它）。
- `MCPToolManager` 重试次数记入 `tool.stats`（新增 `retried` 计数），Monitor 可观测。

## 测试：`tests/test_resilience.py`

1. 瞬态失败两次、第三次成功 → 断言 handler 恰好被调 3 次、结果正确。
2. 永久错误（`BadRequestError`）→ 断言只调 1 次、不重试。
3. 退避延迟递增 → mock `asyncio.sleep` 记录间隔。
4. 超时 → handler 睡过头 → 断言走 fallback。
5. `ToolLoopGuard`：同参第 4 次调用返回重复标记；轮次超 `max_iterations` 时 `should_stop()==True`。

## 范围外（YAGNI）

- `ToolLoopGuard` 写好但暂不接入 agent 链路（当前无 function calling）。
- 不收纳/不重构现有 `CircuitBreaker` 与 TTL 缓存，保持 `tool_manager.py` 原地不动。
- 不使用第三方库（tenacity 等），保持零依赖。
