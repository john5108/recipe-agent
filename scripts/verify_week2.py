"""Offline Week 2 acceptance test: real SQLite tool + fake model tool call."""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app import agent as agent_module  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402


class ToolCallingFakeLLM:
    """Emits a real tool_call, then a final answer after the ToolMessage."""

    bound_tools = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def bind_tools(self, tools):
        type(self).bound_tools = tools
        return self

    def invoke(self, messages):
        last_message = messages[-1] if messages else None
        if (
            isinstance(last_message, HumanMessage)
            and isinstance(last_message.content, str)
            and "意图编号" in last_message.content
        ):
            return AIMessage(content="2")

        last_human_index = max(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage)
            ),
            default=-1,
        )
        has_tool_result_for_current_turn = any(
            isinstance(message, ToolMessage)
            for message in messages[last_human_index + 1:]
        )

        if has_tool_result_for_current_turn:
            return AIMessage(content="已根据菜品库返回查询结果。")

        user_text = ""
        if last_human_index >= 0:
            user_text = str(messages[last_human_index].content)
        ingredient = "牛肉" if "牛肉" in user_text else "土豆"

        return AIMessage(
            content="",
            tool_calls=[{
                "name": "query_dishes_by_ingredient",
                "args": {"ingredient": ingredient, "limit": 5},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }],
        )


def state_for(session_id, text, messages):
    return {
        "session_id": session_id,
        "ingredients": text,
        "intent": "greeting",
        "recipe_ideas": [],
        "selected_recipe": None,
        "final_recipe": None,
        "image_url": None,
        "image_data": "",
        "messages": messages,
    }


async def run():
    original_llm = agent_module.ChatOpenAI
    agent_module.ChatOpenAI = ToolCallingFakeLLM
    try:
        agent = agent_module.create_recipe_agent()
        session_id = f"offline-week2-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": session_id}}

        first = await agent.ainvoke(
            state_for(session_id, "查询包含土豆的历史菜品", []),
            config=config,
        )
        first_tool_messages = [
            message for message in first["messages"]
            if isinstance(message, ToolMessage)
        ]
        assert len(first_tool_messages) == 1
        first_payload = json.loads(first_tool_messages[0].content)
        assert first_payload["count"] >= 1
        print(
            "PASS: 第一轮工具调用返回",
            first_payload["count"],
            "条含土豆的历史菜品",
        )

        snapshot = await agent.aget_state(config)
        previous_messages = list(snapshot.values.get("messages", []))
        assert len(previous_messages) >= 2

        second = await agent.ainvoke(
            state_for(session_id, "那牛肉呢？", previous_messages),
            config=config,
        )
        second_tool_messages = [
            message for message in second["messages"]
            if isinstance(message, ToolMessage)
        ]
        assert len(second_tool_messages) == 2
        second_payload = json.loads(second_tool_messages[-1].content)
        assert second_payload["query"] == "牛肉"

        contents = [message.content for message in second["messages"]]
        assert contents.count("查询包含土豆的历史菜品") == 1
        assert "那牛肉呢？" in contents
        print(
            "PASS: 第二轮 checkpoint 带上了第一轮历史，且数据库工具再次调用成功",
        )
        print("PASS: 第二轮查询食材 =", second_payload["query"])
        print("OFFLINE_WEEK2_VERIFY_PASS")
    finally:
        agent_module.ChatOpenAI = original_llm


if __name__ == "__main__":
    asyncio.run(run())
