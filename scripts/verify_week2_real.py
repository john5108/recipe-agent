"""Real-model Week 2 acceptance test. This consumes model quota."""

import asyncio
import os
import sys
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app.agent import create_recipe_agent  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402


async def run():
    agent = create_recipe_agent()
    session_id = f"week2-real-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}
    state = {
        "session_id": session_id,
        "ingredients": "请调用工具查询菜品库中包含土豆的历史菜品，只返回菜名列表，不要编造。",
        "intent": "greeting",
        "recipe_ideas": [],
        "selected_recipe": None,
        "final_recipe": None,
        "image_url": None,
        "image_data": "",
        "messages": [],
    }

    result = await agent.ainvoke(state, config=config)
    messages = result.get("messages", [])
    tool_calls = [
        call
        for message in messages
        for call in (getattr(message, "tool_calls", None) or [])
    ]
    tool_messages = [
        message for message in messages
        if isinstance(message, ToolMessage)
    ]

    print("TOOL_CALLS=", [
        (call.get("name"), call.get("args"))
        for call in tool_calls
    ])
    print("TOOL_MESSAGE_COUNT=", len(tool_messages))

    last_ai = next(
        (
            message for message in reversed(messages)
            if getattr(message, "type", "") == "ai" and message.content
        ),
        None,
    )
    print("FINAL_ANSWER=", last_ai.content if last_ai else "")

    if not any(
        call.get("name") == "query_dishes_by_ingredient"
        for call in tool_calls
    ):
        raise SystemExit("REAL_WEEK2_FAIL: 模型没有调用数据库查询工具")
    if not tool_messages:
        raise SystemExit("REAL_WEEK2_FAIL: 没有工具执行结果")
    print("REAL_WEEK2_PASS")


if __name__ == "__main__":
    asyncio.run(run())
