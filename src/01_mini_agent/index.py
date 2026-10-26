"""最小 agent：只做天气查询与对比。

核心：模型无状态，所谓"记忆"就是每次重发的 messages 数组。
    模型说要用工具 -> 我们执行 -> 结果塞回 messages -> 再请求一次
"""

import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODEL = os.environ.get("AGENT_MODEL", "deepseek-v4-pro")
MAX_STEPS = 10

client = Anthropic(
    auth_token=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url=os.environ["ANTHROPIC_BASE_URL"],
)

TOOLS = [
    {
        "name": "get_weather",
        "description": "查询某个城市的当前天气。",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，例如 北京"}},
            "required": ["city"],
        },
    },
]


def run_tool(args: dict) -> str:
    return f"{args['city']} 晴，26℃"


def main(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    for step in range(1, MAX_STEPS + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        # 整块追加，别只取 text，否则丢掉 tool_use 块接不上
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            print(f"[step {step}] 调用 {block.name}({block.input})")
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": run_tool(block.input),
                }
            )

        # 同一次响应的所有结果，放进同一条 user 消息
        messages.append({"role": "user", "content": results})

    return f"达到最大步数 {MAX_STEPS}，已中止。"


if __name__ == "__main__":
    print(main("北京和上海今天的天气怎么样？对比一下"))
