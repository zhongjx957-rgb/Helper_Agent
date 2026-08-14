"""弹性工具箱单元测试：超时 / 重试 / 循环守卫。

零依赖：使用标准库 unittest（IsolatedAsyncioTestCase 支持 async 测试），不引入 pytest。
运行：python -m unittest tests.test_resilience -v
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic

from core.resilience import (
    ResilienceError,
    ToolLoopGuard,
    call_with_timeout,
    with_retry,
)


class _FakeResponse:
    status_code = 200
    headers = {}
    request = None


def _make_anthropic_error(cls, status_code: int):
    resp = _FakeResponse()
    resp.status_code = status_code
    return cls(f"fake HTTP {status_code}", response=resp, body={})


class WithRetryTest(unittest.IsolatedAsyncioTestCase):
    """重试语义测试。"""

    async def test_transient_retries_then_succeeds(self):
        """瞬态错误前两次失败、第三次成功 → 恰好调用 3 次，返回正确结果。"""
        calls = 0

        async def flaky_handler():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _make_anthropic_error(anthropic.RateLimitError, 429)
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await with_retry(flaky_handler, attempts=3, base_delay=0.1)

        self.assertEqual(result, "ok")
        self.assertEqual(calls, 3)
        self.assertEqual(mock_sleep.await_count, 2)  # 两次失败后各睡一次

    async def test_permanent_error_not_retried(self):
        """永久错误（BadRequestError）→ 只调 1 次，异常原样抛出（不包 ResilienceError）。"""
        calls = 0

        async def bad_handler():
            nonlocal calls
            calls += 1
            raise _make_anthropic_error(anthropic.BadRequestError, 400)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with self.assertRaises(anthropic.BadRequestError):
                await with_retry(bad_handler, attempts=3)

        self.assertEqual(calls, 1)
        mock_sleep.assert_not_awaited()

    async def test_backoff_delay_increases(self):
        """指数退避：两次失败后的等待间隔应递增。"""
        delays = []
        attempts_left = 3

        async def flaky_handler():
            nonlocal attempts_left
            attempts_left -= 1
            if attempts_left > 0:
                raise _make_anthropic_error(anthropic.RateLimitError, 429)
            return "ok"

        async def fake_sleep(delay):
            delays.append(delay)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await with_retry(flaky_handler, attempts=3, base_delay=0.5, max_delay=8.0)

        self.assertEqual(len(delays), 2)
        # 抖动因子在 0.5~1.5 之间，指数基 0.5、1.0 的抖动区间上界不重叠
        self.assertGreater(delays[1], delays[0])

    async def test_timeout_raises_timeout_then_resilience(self):
        """超时：call_with_timeout 抛 asyncio.TimeoutError；with_retry 耗尽后抛 ResilienceError。

        注意：不能 patch asyncio.sleep，否则会把 slow_handler 内部的 sleep 也 mock 掉、
        导致它瞬间返回而不再超时。
        """
        async def slow_handler():
            await asyncio.sleep(1)
            return "too late"

        with self.assertRaises(asyncio.TimeoutError):
            await call_with_timeout(slow_handler, timeout=0.01)

        with self.assertRaises(ResilienceError):
            await with_retry(
                slow_handler, attempts=2, timeout=0.01,
                base_delay=0.01, max_delay=0.01,
            )

    async def test_cancelled_error_propagates_not_retried(self):
        """CancelledError 绝不吞掉，也不参与重试。"""
        calls = 0

        async def cancelling_handler():
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with self.assertRaises(asyncio.CancelledError):
                await with_retry(cancelling_handler, attempts=3)

        self.assertEqual(calls, 1)
        mock_sleep.assert_not_awaited()

    async def test_on_failure_callback_receives_each_failure(self):
        """on_failure 回调每次失败都触发（重试喂养熔断器的衔接点）。"""
        calls = 0
        notified = []

        async def flaky_handler():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _make_anthropic_error(anthropic.RateLimitError, 429)
            return "ok"

        async def fake_sleep(delay):
            pass

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await with_retry(
                flaky_handler,
                attempts=3,
                base_delay=0.01,
                on_failure=lambda ex, attempt: notified.append(attempt),
            )

        self.assertEqual(notified, [0, 1])


class CallWithTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_result_within_timeout(self):
        async def fast():
            return 42

        self.assertEqual(await call_with_timeout(fast, timeout=1.0), 42)


class ToolLoopGuardTest(unittest.TestCase):
    """循环守卫测试（同步，无需事件循环）。"""

    def test_same_call_exceeds_limit_returns_false(self):
        guard = ToolLoopGuard(max_iterations=10, max_same_call=3)
        h = "hash-123"
        self.assertTrue(guard.record_call("tool_a", h))
        self.assertTrue(guard.record_call("tool_a", h))
        self.assertTrue(guard.record_call("tool_a", h))
        self.assertFalse(guard.record_call("tool_a", h))  # 第 4 次同参 → 重复超限
        # 不同参数不受影响
        self.assertTrue(guard.record_call("tool_a", "hash-456"))
        self.assertTrue(guard.record_call("tool_b", h))

    def test_iteration_limit_stops(self):
        guard = ToolLoopGuard(max_iterations=3, max_same_call=3)
        self.assertTrue(guard.begin())
        self.assertTrue(guard.begin())
        self.assertTrue(guard.begin())
        self.assertFalse(guard.begin())            # 第 4 轮被拦
        self.assertTrue(guard.should_stop())

    def test_max_same_call_is_inclusive(self):
        guard = ToolLoopGuard(max_iterations=10, max_same_call=1)
        self.assertTrue(guard.record_call("tool_a", "h1"))
        self.assertFalse(guard.record_call("tool_a", "h1"))


if __name__ == "__main__":
    unittest.main()
