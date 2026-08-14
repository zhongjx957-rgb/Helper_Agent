"""
亮点：弹性工具箱（超时 / 重试 / 循环守卫）

核心问题：LLM 调用无超时、零重试、未来 function calling 无循环守卫，边界异常怎么处理？

与熔断器的分工：
  熔断器（mcp/tool_manager.py）拦"持续故障防雪崩"，只统计失败。
  本模块处理它看不到的三块：
    1. call_with_timeout —— 把"永远等不到"变成"可计数的失败"（超时是熔断器能感知的前提）
    2. with_retry       —— 吸收瞬时故障（熔断器的哲学是"别再发了"，重试是"再发一次"）
    3. ToolLoopGuard    —— 限制模型-工具循环总轮次（熔断器对"成功但空转"无感）

原则：
  - 零第三方依赖，仅标准库 + anthropic 异常引用。
  - 只重试瞬态错误（超时/网络/限流/5xx），4xx/鉴权直接抛出。
  - asyncio.CancelledError 绝不吞掉，原样重新抛出（asyncio.wait_for 取消依赖它）。
"""
import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Optional

import anthropic

logger = logging.getLogger(__name__)


# ── 超时常量 ──────────────────────────────────────────────────────────────────

LLM_TIMEOUT = 20.0               # 单次 LLM 调用超时（秒）
REQUEST_TOTAL_TIMEOUT = 60.0     # 整条链路预算（秒），在 AgentOrchestrator.run 兜底


# ── 可重试的瞬态错误白名单 ────────────────────────────────────────────────────
# 只有这些异常值得重试；AuthenticationError / BadRequestError 等永久错误绝不重试。

RETRYABLE_EXCEPTIONS: tuple = (
    asyncio.TimeoutError,              # 我们的 wait_for 超时
    anthropic.APIConnectionError,      # 网络抖动/连接失败
    anthropic.APITimeoutError,         # SDK 自身读超时
    anthropic.RateLimitError,          # 限流，退避后重试最有效
    anthropic.InternalServerError,     # 5xx
)


class ResilienceError(Exception):
    """重试耗尽后的统一异常，由上层按现有降级路径处理。"""


# ── 超时 ──────────────────────────────────────────────────────────────────────

async def call_with_timeout(
    coro_factory: Callable[[], Awaitable[Any]],
    timeout: float = LLM_TIMEOUT,
) -> Any:
    """
    单次调用超时。

    用 asyncio.wait_for 包一层，超时抛 asyncio.TimeoutError（被 with_retry 当作瞬态重试）。

    coro_factory 是工厂而非协程对象：重试需要每次重新构造请求，协程只能 await 一次。
    """
    return await asyncio.wait_for(coro_factory(), timeout=timeout)


# ── 重试 ──────────────────────────────────────────────────────────────────────

async def with_retry(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    on_failure: Optional[Callable[[Exception, int], Any]] = None,
    timeout: float = LLM_TIMEOUT,
) -> Any:
    """
    指数退避 + 抖动重试，只重试瞬态错误。

    attempts 为总尝试次数（含首次），默认 3 = 首次 + 2 次重试。
    on_failure(exception, attempt) 可选回调，用于向熔断器报数（重试喂养熔断器）。

    退避公式：min(base_delay * 2**attempt, max_delay) * (0.5 + random.random())，
    指数增长 + 0.5~1.5 抖动，避免同一时刻重试的请求再次撞车。
    """
    for attempt in range(attempts):
        try:
            return await call_with_timeout(coro_factory, timeout=timeout)
        except RETRYABLE_EXCEPTIONS as ex:
            if on_failure is not None:
                on_failure(ex, attempt)
            if attempt == attempts - 1:
                raise ResilienceError(f"重试 {attempts} 次后仍失败: {ex}") from ex
            delay = min(base_delay * 2 ** attempt, max_delay) * (0.5 + random.random())
            logger.warning(f"第 {attempt + 1}/{attempts} 次失败，{delay:.2f}s 后重试: {ex}")
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            # 绝不吞取消：asyncio.wait_for 的超时/取消依赖 CancelledError 正常传播
            raise
        except Exception:
            # 永久错误（鉴权/参数/业务异常）：不重试，原样抛出
            raise


# ── 工具循环守卫（写好但暂不接入，当前无 function calling）───────────────────

class ToolLoopGuard:
    """
    工具循环守卫：限制模型-工具循环不收敛。

    熔断器只统计失败，对"模型反复调用工具、每次调用都成功但不收敛"完全无感。
    本守卫解决两类循环：
      1. 总轮次上限 —— 超过 max_iterations 强制终止（防空转烧钱）
      2. 同参重复调用 —— 同一 (tool, params) 调用超过 max_same_call 次视为病态重复

    未来接入 function calling 时，在 agent 循环里每个工具调用前检查：
        if not guard.begin():  # 或 guard.should_stop()
            停止循环，返回当前最佳回答
        ok = guard.record_call(name, params_hash)
        if not ok:  # 同参重复，直接返回上轮结果
            不再执行工具，把上轮结果原样返回
    """

    def __init__(self, max_iterations: int = 10, max_same_call: int = 3):
        self.max_iterations = max_iterations
        self.max_same_call = max_same_call
        self.iterations = 0
        self._call_counts: dict = {}  # (tool_name, params_hash) -> count

    def begin(self) -> bool:
        """新一轮工具调用；返回 False 表示已达轮次上限，应停止循环。"""
        self.iterations += 1
        return self.iterations <= self.max_iterations

    def should_stop(self) -> bool:
        """轮次是否已达上限。"""
        return self.iterations >= self.max_iterations

    def record_call(self, tool_name: str, params_hash: str) -> bool:
        """
        记录一次调用；返回 False 表示同参重复超限（应返回上轮结果而非再次执行）。
        """
        key = (tool_name, params_hash)
        count = self._call_counts.get(key, 0) + 1
        self._call_counts[key] = count
        return count <= self.max_same_call
