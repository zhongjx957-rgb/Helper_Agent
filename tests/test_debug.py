"""调试：分别查看三路各自的结果"""
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

    msg = "为什么连续扣了我两次款"

    # 三路分别跑
    llm = await r._llm_recognize(msg, None)
    emb = await r._embedding_recognize(msg) if r._embedding_enabled else {"error": "embedding 被禁用"}
    pat = r._pattern_recognize(msg)

    print(f"消息: {msg}")
    print(f"  Embedding 启用: {r._embedding_enabled}")
    print(f"  LLM:      意图={llm.get('intent', '?' )} 置信度={llm.get('confidence', 0):.3f}")
    print(f"  Embedding:意图={emb.get('intent', '?')} 置信度={emb.get('confidence', 0):.3f}")
    print(f"  Pattern:  意图={pat.get('intent', '?')} 置信度={pat.get('confidence', 0):.3f}")

    # 最终投票
    result = await r.recognize(msg)
    print(f"  最终:   意图={result.intent.value} 置信度={result.confidence:.3f}")

asyncio.run(main())
