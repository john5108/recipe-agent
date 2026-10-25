"""
测试脚本，验证智能体各组件是否正常工作
"""
import os
import sys
from unittest.mock import patch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 设置模拟环境变量
os.environ.setdefault("TAVILY_API_KEY", "dummy_key_for_test")
os.environ.setdefault("OPENAI_API_KEY", "dummy_key_for_test")
os.environ.setdefault("ALIBABA_CLOUD_ACCESS_KEY_ID", "dummy_id_for_test")
os.environ.setdefault("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "dummy_secret_for_test")
os.environ.setdefault("OSS_BUCKET_NAME", "dummy_bucket_for_test")

def test_imports():
    """测试导入"""
    print("Testing imports...")
    try:
        import chainlit
        import langchain
        import langgraph
        import tavily
        import openai
        import oss2
        import requests
        from PIL import Image
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    return True

def test_config():
    """测试配置加载"""
    print("\nTesting config...")
    try:
        from config.settings import TAVILY_API_KEY, MODEL_API_KEY, WANX_MODEL
        print(f"✅ Config loaded, TAVILY_API_KEY exists: {TAVILY_API_KEY is not None}")
        assert WANX_MODEL.startswith(("qwen-image", "wanx")), WANX_MODEL
        print(f"✅ Image model configured: {WANX_MODEL}")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    return True

def test_tools():
    """测试工具模块"""
    print("\nTesting tools...")
    try:
        from tools.recipe_tools import tools
        print(f"✅ Tools loaded: {[tool.name for tool in tools]}")
    except Exception as e:
        print(f"❌ Tools error: {e}")
        return False
    return True

def test_db_tools():
    """测试 SQLite 查询工具，不修改项目真实数据库"""
    print("\nTesting database tools...")
    import tempfile
    from utils import dish_tracker
    from tools.recipe_tools import query_dishes_by_ingredient, query_recent_dishes

    old_tracker = dish_tracker._tracker_instance
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = dish_tracker.DishDevelopmentTracker(
                os.path.join(tmpdir, "test_week2.db")
            )
            dish_tracker._tracker_instance = tracker
            tracker.save_session("test-session")
            tracker.save_dish("test-session", {
                "dish_name": "土豆牛肉测试菜",
                "ingredients": ["土豆 200g", "牛肉 150g"],
                "recipe": {"steps": ["切块", "炖煮"]},
                "cost_estimation": "8元",
                "suggested_price": "22元",
                "marketing_copy": {"title": "土豆牛肉"},
            })

            by_ingredient = query_dishes_by_ingredient.invoke({
                "ingredient": "土豆",
                "limit": 5,
            })
            recent = query_recent_dishes.invoke({
                "session_id": "test-session",
                "limit": 5,
            })
            assert by_ingredient["count"] == 1, by_ingredient
            assert recent["count"] == 1, recent
            print("✅ Database query tools returned the inserted dish")
    except Exception as e:
        print(f"❌ Database tools error: {e}")
        return False
    finally:
        dish_tracker._tracker_instance = old_tracker
    return True

def test_tool_binding_and_checkpointer():
    """测试 bind_tools 和 LangGraph checkpointer 是否真的接上"""
    print("\nTesting tool binding and checkpointer...")
    from app import agent as agent_module
    from langchain_core.messages import AIMessage

    class FakeLLM:
        bound_tools = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def bind_tools(self, tools):
            type(self).bound_tools = tools
            return self

        def invoke(self, messages):
            return AIMessage(content="test")

    original_llm = agent_module.ChatOpenAI
    try:
        agent_module.ChatOpenAI = FakeLLM
        agent = agent_module.create_recipe_agent()
        tool_names = [tool.name for tool in FakeLLM.bound_tools or []]
        assert "query_dishes_by_ingredient" in tool_names, tool_names
        assert "query_recent_dishes" in tool_names, tool_names
        assert agent.checkpointer is not None, "checkpointer is missing"
        print(f"✅ Tools bound: {tool_names}")
        print("✅ LangGraph checkpointer is enabled")
    except Exception as e:
        print(f"❌ Tool binding/checkpointer error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        agent_module.ChatOpenAI = original_llm
    return True

def test_recipe_schema():
    """测试模型返回的菜品 JSON 是否能被 Pydantic 校验"""
    print("\nTesting recipe output schema...")
    try:
        from pydantic import ValidationError
        from app.schemas import RecipeOutput

        parsed = RecipeOutput.model_validate({
            "dishes": [{
                "dish_name": "土豆牛肉测试菜",
                "ingredients": ["土豆", "牛肉"],
            }]
        })
        assert parsed.dishes[0].dish_name == "土豆牛肉测试菜"
        try:
            RecipeOutput.model_validate({"dishes": []})
        except ValidationError:
            pass
        else:
            raise AssertionError("empty dishes should fail validation")
        print("✅ Recipe schema accepts valid dishes and rejects empty output")
    except Exception as e:
        print(f"❌ Recipe schema error: {e}")
        return False
    return True


def test_intent_routing_rules():
    """测试清晰意图不再依赖模型分类。"""
    print("\nTesting intent routing rules...")
    try:
        from app.agent import _classify_image_intent, _classify_intent_by_rules

        assert _classify_intent_by_rules("你好，你能做什么") == "greeting"
        assert _classify_intent_by_rules("牛肉黄瓜能做什么菜") == "recipe"
        assert _classify_intent_by_rules("上传冰箱里的食材图片") == "recipe"
        assert _classify_intent_by_rules("查询包含土豆的历史菜品") == "query"
        assert _classify_intent_by_rules("那牛肉呢？") == "query"
        assert _classify_image_intent("里面有几种食材分别是什么") == "ingredient_scan"
        assert _classify_image_intent("用这些食材做几道菜") == "recipe"
        assert _classify_image_intent("") == "ingredient_scan"
        print("✅ Text and image intents route without an LLM call")
    except Exception as e:
        print(f"❌ Intent routing error: {e}")
        return False
    return True


def test_recipe_parser():
    """测试带 Markdown 包裹的菜品 JSON 能被解析和校验。"""
    print("\nTesting recipe parser...")
    try:
        from app.agent import _parse_recipe_output

        parsed = _parse_recipe_output(
            '方案如下：\n```json\n'
            '{"dishes":[{"dish_name":"黄瓜炒肉","ingredients":["黄瓜","牛肉"]}]}'
            '\n```'
        )
        assert parsed and parsed["dishes"][0]["dish_name"] == "黄瓜炒肉"
        assert _parse_recipe_output("没有结构化数据") is None
        print("✅ Recipe parser validates JSON and rejects prose-only output")
    except Exception as e:
        print(f"❌ Recipe parser error: {e}")
        return False
    return True


def test_optional_recipe_state():
    """测试空菜谱状态和上下文预算。"""
    print("\nTesting optional state and context budget...")
    try:
        from app.agent import _messages_for_llm, normalize_final_recipe
        from app.ui import normalize_dish_markdown
        from config.settings import MAX_CONTEXT_CHARS
        from langchain_core.messages import AIMessage, HumanMessage

        assert normalize_final_recipe(None) == {}
        assert normalize_final_recipe({"dishes": []}) == {"dishes": []}
        messages = []
        for index in range(20):
            messages.append(HumanMessage(content=f"{index}:" + "食材" * 1000))
            messages.append(AIMessage(content=f"{index}:" + "方案" * 1000))
        selected = _messages_for_llm(messages, "系统提示")
        selected_chars = sum(
            len(message.content) if isinstance(message.content, str) else 0
            for message in selected
        )
        assert len(selected) <= 13
        assert selected_chars <= MAX_CONTEXT_CHARS + 10

        rendered = normalize_dish_markdown(
            "5. 最后一步。\n"
            "**⏱️ 出品信息：**\n"
            "• 烹饪时间：8分钟\n"
            "**📝 上架文案：**\n"
            "📌 **菜品标题**：测试菜"
        )
        assert "最后一步。\n\n**出品信息**" in rendered
        assert "**上架文案**" in rendered
        assert "⏱️" not in rendered and "📝" not in rendered
        print("✅ None recipe state is safe and context is character-bounded")
    except Exception as e:
        print(f"❌ Recipe state error: {e}")
        return False
    return True


def test_workflow_branches():
    """测试闲聊和缺少 JSON 的菜谱分支不会生成假菜品。"""
    print("\nTesting workflow branches...")
    import asyncio
    from app import agent as agent_module
    from langchain_core.messages import AIMessage, SystemMessage

    class BranchFakeLLM:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            if isinstance(messages[-1].content, list):
                return AIMessage(content="生菜, 番茄, 黄瓜, 卷心菜")
            system_text = "\n".join(
                str(message.content)
                for message in messages
                if isinstance(message, SystemMessage)
            )
            if "严格的数据整理器" in system_text:
                return AIMessage(content=(
                    '{"dishes":[{"dish_name":"黄瓜炒牛肉",'
                    '"ingredients":["黄瓜","牛肉"],"steps":["切片","快炒"]}]}'
                ))
            if "空回复测试" in str(messages[-1].content):
                return AIMessage(content="")
            if "外卖菜品研发与上架顾问" in system_text:
                return AIMessage(content=(
                    "### 黄瓜炒牛肉\n"
                    "食材清单：黄瓜、牛肉\n"
                    "制作步骤：切片后快炒\n"
                    "成本与售价：待测算"
                ))
            return AIMessage(content="你好，我可以帮你根据食材研发外卖菜品。")

    original_llm = agent_module.ChatOpenAI
    original_image_search = agent_module.search_dish_images_tavily
    try:
        agent_module.ChatOpenAI = BranchFakeLLM
        agent_module.search_dish_images_tavily = lambda *args, **kwargs: {
            "images": [],
            "error": "test skip",
            "error_type": "test",
        }
        agent = agent_module.create_recipe_agent()

        async def run_cases():
            greeting = await agent.ainvoke(
                {
                    "session_id": "test-greeting",
                    "user_input": "你好",
                    "ingredients": "你好",
                    "intent": "greeting",
                    "recipe_ideas": [],
                    "selected_recipe": None,
                    "final_recipe": None,
                    "image_url": None,
                    "image_data": "",
                    "messages": [],
                },
                config={"configurable": {"thread_id": "test-greeting"}},
            )
            assert greeting.get("final_recipe") is None

            recipe = await agent.ainvoke(
                {
                    "session_id": "test-recipe",
                    "user_input": "黄瓜和牛肉能做什么菜",
                    "ingredients": "黄瓜和牛肉能做什么菜",
                    "intent": "recipe",
                    "recipe_ideas": [],
                    "selected_recipe": None,
                    "final_recipe": None,
                    "image_url": None,
                    "image_data": "",
                    "messages": [],
                },
                config={"configurable": {"thread_id": "test-recipe"}},
            )
            dish = recipe["final_recipe"]["dishes"][0]
            assert dish["dish_name"] == "黄瓜炒牛肉"
            assert "黄瓜和牛肉能做什么菜" not in dish["dish_name"]

            empty_reply = await agent.ainvoke(
                {
                    "session_id": "test-empty",
                    "user_input": "空回复测试菜品",
                    "ingredients": "空回复测试菜品",
                    "intent": "recipe",
                    "recipe_ideas": [],
                    "selected_recipe": None,
                    "final_recipe": None,
                    "image_url": None,
                    "image_data": "",
                    "messages": [],
                },
                config={"configurable": {"thread_id": "test-empty"}},
            )
            assert empty_reply.get("final_recipe") is None

            image_scan = await agent.ainvoke(
                {
                    "session_id": "test-image-scan",
                    "user_input": "里面有几种食材分别是什么",
                    "ingredients": "",
                    "intent": "greeting",
                    "recipe_ideas": [],
                    "selected_recipe": None,
                    "final_recipe": None,
                    "image_url": None,
                    "image_data": "fake-image-data",
                    "messages": [],
                },
                config={"configurable": {"thread_id": "test-image-scan"}},
            )
            assert image_scan.get("final_recipe") is None
            assert "识别到以下食材" in image_scan["messages"][-1].content
            assert "番茄" in image_scan["ingredients"]

            image_followup = await agent.ainvoke(
                {
                    "session_id": "test-image-scan",
                    "user_input": "用这些食材做几道菜",
                    "ingredients": image_scan["ingredients"],
                    "intent": "greeting",
                    "recipe_ideas": [],
                    "selected_recipe": None,
                    "final_recipe": None,
                    "image_url": None,
                    "image_data": "",
                    "messages": list(image_scan["messages"]),
                },
                config={"configurable": {"thread_id": "test-image-scan"}},
            )
            assert "番茄" in image_followup["ingredients"]
            assert image_followup["final_recipe"]["dishes"]

        asyncio.run(run_cases())
        print("✅ Image scan stays inventory-only; the next turn can reuse it")
    except Exception as e:
        print(f"❌ Workflow branch error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        agent_module.ChatOpenAI = original_llm
        agent_module.search_dish_images_tavily = original_image_search
    return True


def test_agent():
    """测试智能体"""
    print("\nTesting agent...")
    try:
        # 模拟ChatOpenAI的调用
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key"
        }):
            from app.agent import create_recipe_agent
            agent = create_recipe_agent()
            print("✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Agent error: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

def main():
    """运行所有测试"""
    print("🔍 Running tests for Recipe Agent...")
    print("="*50)

    tests = [
        test_imports,
        test_config,
        test_tools,
        test_db_tools,
        test_tool_binding_and_checkpointer,
        test_recipe_schema,
        test_intent_routing_rules,
        test_recipe_parser,
        test_optional_recipe_state,
        test_workflow_branches,
        test_agent
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("🎉 All tests passed! The recipe agent is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
