"""快速测试意图识别"""
import asyncio
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from core.intent_recognizer import IntentRecognizer

async def main():
    r = IntentRecognizer(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        model=os.getenv("ANTHROPIC_MODEL", "deepseek-chat"),
    )
    messages = ["你好", "为什么连续扣了我两次款","我很急","我把我密码给忘了","草泥马你这什么服务啊"]
    results = await asyncio.gather(*[r.recognize(msg) for msg in messages])
    for msg, result in zip(messages, results):
        print(f"消息: {msg}")
        print(f"  意图: {result.intent.value}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  紧急度: {result.urgency.name}")
        print(f"  耗时: {result.latency_ms:.0f}ms")
        print(f"  推理: {result.reasoning[:80]}")
        print()

asyncio.run(main())
