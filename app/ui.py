import sys
import os
# 添加项目根目录到 Python 路径，解决 chainlit run 时的模块导入问题
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Python 3.14 兼容修复：sniffio 1.3.1 在 nest-asyncio 嵌套事件循环下无法检测异步上下文
# 导致 anyio.to_thread.run_sync → sniffio.current_async_library() 抛出 AsyncLibraryNotFoundError
# 这里做一个安全降级：如果 sniffio 检测失败，默认返回 "asyncio"
import sniffio
_original_sniffio = sniffio.current_async_library
def _patched_sniffio():
    try:
        return _original_sniffio()
    except sniffio.AsyncLibraryNotFoundError:
        return "asyncio"
sniffio.current_async_library = _patched_sniffio

import chainlit as cl
from app.agent import (
    create_recipe_agent,
    RecipeAgentState,
    _generate_dish_image,
    normalize_final_recipe,
)
from utils.dish_tracker import get_tracker, get_dish_history, save_current_dish
from tools.image_utils import analyze_dish_visuals_for_prompt
from config.settings import WANX_API_KEY
import asyncio
import re
import base64

# 智能体改为懒加载，避免模块导入失败导致白屏
_agent = None


def normalize_dish_markdown(dish_text: str) -> str:
    """Keep generated dish sections visually separated in Chainlit Markdown."""
    replacements = {
        "**✅ 核心卖点：**": "**核心卖点**",
        "**🥘 食材清单：**": "**食材清单**",
        "**👨‍🍳 制作步骤：**": "**制作步骤**",
        "**⏱️ 出品信息：**": "**出品信息**",
        "**📝 上架文案：**": "**上架文案**",
        "**💪 营养价值：**": "**营养价值**",
        "**🖼️ 图片建议：**": "**图片建议**",
        "📌 **菜品标题**": "**菜品标题**",
        "📄 **详情描述**": "**详情描述**",
        "🏷️ **标签**": "**标签**",
        "### 🍽️": "###",
    }
    for old, new in replacements.items():
        dish_text = dish_text.replace(old, new)
    dish_text = dish_text.replace("• ", "- ")
    dish_text = dish_text.replace(" · ", " / ")
    dish_text = re.sub(
        r"\n(?=\*\*(?:核心卖点|食材清单|制作步骤|出品信息|上架文案|营养价值|图片建议)\*\*)",
        "\n\n",
        dish_text,
    )
    return re.sub(r"\n{3,}", "\n\n", dish_text).strip()


def get_agent():
    """懒加载智能体实例"""
    global _agent
    if _agent is None:
        _agent = create_recipe_agent()
    return _agent

def get_session_id() -> str:
    """获取Chainlit当前会话ID，用于持久化"""
    try:
        return cl.user_session.get("id") or "default_session"
    except Exception:
        return "default_session"


@cl.action_callback("generate_wanx_images")
async def on_generate_wanx_images(action: cl.Action):
    """用户点击「生成外卖优化图」按钮时触发，调用通义万相生图"""
    if not WANX_API_KEY:
        await cl.Message(content="⚠️ 通义万相 API Key 未配置，无法生成优化图。请在 .env 中设置 QWEN_API_KEY。").send()
        return

    final_recipe = normalize_final_recipe(cl.user_session.get("current_final_recipe"))
    if not final_recipe:
        await cl.Message(content="⚠️ 未找到当前菜谱数据，请先研发菜品。").send()
        return

    dishes = final_recipe.get("dishes", [])
    if not dishes:
        await cl.Message(content="⚠️ 当前没有菜品可生成图片。").send()
        return

    status_msg = cl.Message(content="🖼️ 正在为每道菜生成外卖优化图，请稍候...")
    await status_msg.send()
    generation_errors = []

    for i, dish in enumerate(dishes):
        dish_name = dish.get("dish_name", f"菜品{i+1}")
        ingredients = dish.get("ingredients", [])
        dish["generated_image_error"] = ""

        # 优化1：取 Tavily 参考图，多模态分析视觉特征，用于优化生图 Prompt
        visual_hints = ""
        ref_images = dish.get("tavily_reference_images", [])
        if ref_images:
            try:
                visual_hints = await asyncio.to_thread(
                    analyze_dish_visuals_for_prompt,
                    dish_name, ingredients, ref_images
                )
            except Exception as e:
                print(f"[wanx_action] [{dish_name}] 视觉分析失败: {e}")

        try:
            img_url = await asyncio.to_thread(_generate_dish_image, dish_name, ingredients, visual_hints)
            dish["generated_image_url"] = img_url
        except Exception as e:
            print(f"[wanx_action] [{dish_name}] 生图失败: {e}")
            dish["generated_image_url"] = ""
            dish["generated_image_error"] = str(e)
            generation_errors.append(f"{dish_name}：{e}")

        if i < len(dishes) - 1:
            await asyncio.sleep(0.5)

    # 更新 session 状态
    cl.user_session.set("current_final_recipe", final_recipe)

    # 构建图片展示
    img_elements = []
    for dish in dishes:
        gen_url = dish.get("generated_image_url", "")
        if gen_url:
            img_elements.append(
                cl.Image(
                    name=f"wanx-{dish.get('dish_name', '')}",
                    display="inline",
                    url=gen_url,
                    size="large"
                )
            )

    if img_elements:
        await cl.Message(
            content="✅ 外卖优化图生成完成！",
            elements=img_elements
        ).send()
    else:
        detail = generation_errors[0] if generation_errors else "请检查模型配置和账户额度。"
        await cl.Message(
            content=f"⚠️ 所有菜品图片生成失败。\n\n原因：{detail}"
        ).send()

    await status_msg.remove()

@cl.on_chat_start
async def start():
    """聊天开始时的初始化"""
    try:
        # 初始化会话状态
        session_id = get_session_id()
        cl.user_session.set("state", RecipeAgentState(
            user_input="",
            ingredients="",
            intent="greeting",
            recipe_ideas=[],
            selected_recipe=None,
            final_recipe=None,
            image_url=None,
            image_data="",
            messages=[],
            session_id=session_id
        ))

        # 保存会话到数据库
        tracker = get_tracker()
        tracker.save_session(session_id)

        # 检查是否有历史记录
        history = get_dish_history(session_id, limit=1)

        # 发送欢迎消息
        if history:
            last_dish = history[0]
            welcome_msg = f"""
    👋 欢迎回来！上次研发的菜品是 **{last_dish['dish_name']}**

    我可以帮助您继续开发新菜品：
    - 上传食材图片，自动识别并推荐菜谱
    - 根据现有食材结合外卖爆款趋势研发新品
    - 一键生成上架文案和菜品图

    请问这次想做什么菜呢？
            """
        else:
            welcome_msg = """
    👋 欢迎使用外卖菜品研发与上架辅助助手！

    我可以帮助您：
    - 上传食材图片，自动识别食材并推荐菜谱
    - 搜索外卖爆款趋势，研发不拍脑袋
    - 一键生成上架文案和菜品主图

    请告诉我您有哪些食材，我来帮您研发新菜品！
            """

        await cl.Message(content=welcome_msg).send()
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"on_chat_start error: {error_detail}")
        await cl.Message(
            content=f"⚠️ 初始化出错：{str(e)}\n\n请检查 .env 配置文件中的 API Key 是否正确。"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """处理用户消息"""
    session_id = get_session_id()

    # ========== 图片检测与提取 ==========
    image_b64 = ""
    print(f"[ui] 收到的文字：{message.content[:100] if message.content else '(空)'}")
    print(f"[ui] 收到的附件数：{len(message.elements) if message.elements else 0}")

    if message.elements:
        for elem in message.elements:
            mime = getattr(elem, 'mime', '')
            if mime and mime.startswith("image/"):
                try:
                    # Chainlit 上传文件：优先从 path 读取，其次从 content 读取
                    if hasattr(elem, 'path') and elem.path:
                        with open(elem.path, 'rb') as f:
                            img_bytes = f.read()
                    elif hasattr(elem, 'content') and elem.content:
                        img_bytes = elem.content if isinstance(elem.content, bytes) else elem.content.encode()
                    else:
                        continue

                    image_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    print(f"[ui] ✅ 检测到图片，mime: {mime}，大小: {len(img_bytes)} 字节")
                    break  # 只取第一张图片
                except Exception as e:
                    print(f"[ui] 图片读取失败: {e}")

    try:
        # ✅ 坑1修复：用 ainvoke 异步调用，不阻塞事件循环
        agent = get_agent()
        config = {"configurable": {"thread_id": session_id}}

        # 从 LangGraph checkpoint 恢复同一会话的历史消息
        previous_messages = []
        previous_ingredients = ""
        try:
            snapshot = await agent.aget_state(config)
            if snapshot and snapshot.values:
                previous_messages = list(snapshot.values.get("messages", []))
                previous_ingredients = snapshot.values.get("ingredients", "") or ""
                print(f"[ui] 从 checkpoint 恢复 {len(previous_messages)} 条历史消息")
        except Exception as checkpoint_error:
            print(f"[ui] checkpoint 读取失败，按新会话处理: {checkpoint_error}")

        state = RecipeAgentState(
            user_input=message.content or "",
            ingredients=previous_ingredients,
            intent="greeting",  # classify_intent 节点会重新判定
            recipe_ideas=[],
            selected_recipe=None,
            final_recipe=None,
            image_url=None,
            image_data=image_b64,
            messages=previous_messages,
            session_id=session_id
        )
        result = await agent.ainvoke(state, config=config)

        cl.user_session.set("state", result)

        # 存储 final_recipe 供 Wanxiang 按钮回调使用
        final_recipe = normalize_final_recipe(result.get("final_recipe"))
        cl.user_session.set("current_final_recipe", final_recipe)
        dishes = final_recipe.get("dishes", [])

        # 兼容旧版单道菜格式：没有 dishes 但有 dish_name
        if not dishes and final_recipe.get("dish_name"):
            dishes = [final_recipe]

        for dish in dishes:
            if dish.get("dish_name"):
                dish_data = {
                    "dish_name": dish.get("dish_name", ""),
                    "ingredients": dish.get("ingredients", []),
                    "recipe": {
                        "steps": dish.get("steps", []),
                        "time": dish.get("cooking_time", ""),
                        "difficulty": dish.get("difficulty_level", "")
                    },
                    "cost_estimation": dish.get("cost_estimation", ""),
                    "suggested_price": dish.get("suggested_price", ""),
                    "marketing_copy": {
                        "title": dish.get("dish_name", ""),
                        "description": dish.get("marketing_description", ""),
                        "tags": dish.get("tags", []),
                        "nutritional_benefits": dish.get("nutritional_benefits", [])
                    },
                    "image_url": result.get("image_url", ""),
                    "dish_source": "ai"
                }
                save_current_dish(get_session_id(), dish_data)
                print(f"✅ 菜品 [{dish.get('dish_name')}] 已保存到菜品库")

        # ========== 构建响应 ==========
        response_parts = []

        if dishes:
            for idx, dish in enumerate(dishes):
                dish_parts = []
                dish_num = f"【第{idx + 1}道】" if len(dishes) > 1 else ""
                dish_parts.append(f"### 🍽️ {dish_num} {dish.get('dish_name', '推荐菜品')}")

                selling_points = dish.get('core_selling_points', [])
                if selling_points:
                    dish_parts.append("**✅ 核心卖点：**")
                    for point in selling_points:
                        dish_parts.append(f"• {point}")

                dish_parts.append("**🥘 食材清单：**")
                for ingredient in dish.get('ingredients', []):
                    dish_parts.append(f"• {ingredient}")

                dish_parts.append("**👨🍳 制作步骤：**")
                for i, step in enumerate(dish.get('steps', []), 1):
                    dish_parts.append(f"{i}. {step}")

                time_info = dish.get('cooking_time', '')
                diff_info = dish.get('difficulty_level', '')
                cost_info = dish.get('cost_estimation', '')
                price_info = dish.get('suggested_price', '')

                dish_parts.append("**⏱️ 出品信息：**")
                if time_info:
                    dish_parts.append(f"• 烹饪时间：{time_info}")
                if diff_info:
                    dish_parts.append(f"• 难度等级：{diff_info}")
                if cost_info:
                    dish_parts.append(f"• 预估成本：{cost_info}")
                if price_info:
                    dish_parts.append(f"• 建议售价：{price_info}")

                dish_parts.append("**📝 上架文案：**")
                title = dish.get('marketing_title', '')
                if title:
                    dish_parts.append(f"📌 **菜品标题**：{title}")
                desc = dish.get('marketing_description', '')
                if desc:
                    dish_parts.append(f"📄 **详情描述**：{desc}")
                tags = dish.get('tags', [])
                if tags:
                    dish_parts.append(f"🏷️ **标签**：{' · '.join(tags)}")

                nutrition = dish.get('nutritional_benefits', [])
                if nutrition:
                    dish_parts.append("**💪 营养价值：**")
                    for benefit in nutrition:
                        dish_parts.append(f"• {benefit}")

                img_suggestion = dish.get('image_suggestion', '')
                if img_suggestion:
                    dish_parts.append(f"**🖼️ 图片建议：** {img_suggestion}")

                dish_text = "\n".join(dish_parts)
                label_replacements = {
                    "**✅ 核心卖点：**": "**核心卖点**",
                    "**🥘 食材清单：**": "**食材清单**",
                    "**👨‍🍳 制作步骤：**": "**制作步骤**",
                    "**⏱️ 出品信息：**": "**出品信息**",
                    "**📝 上架文案：**": "**上架文案**",
                    "**💪 营养价值：**": "**营养价值**",
                }
                for old_label, new_label in label_replacements.items():
                    dish_text = dish_text.replace(old_label, new_label)
                    dish_text = re.sub(
                        rf"\n(?={re.escape(new_label)})",
                        "\n\n",
                        dish_text,
                    )
                dish_text = dish_text.replace("• ", "- ")
                dish_text = dish_text.replace("📌 **菜品标题**", "**菜品标题**")
                dish_text = dish_text.replace("📄 **详情描述**", "**详情描述**")
                dish_text = dish_text.replace("🏷️ **标签**", "**标签**")
                dish_text = dish_text.replace("**🖼️ 图片建议：**", "**图片建议**：")
                dish_text = dish_text.replace(" · ", " / ")
                dish_text = re.sub(r"\n{3,}", "\n\n", dish_text).strip()
                response_parts.append(normalize_dish_markdown(dish_text))

        if not dishes:
            # 闲聊/问候场景：没有菜谱，直接展示 AI 的实际回复
            ai_reply = ""
            messages = result.get("messages", [])
            for m in reversed(messages):
                if hasattr(m, 'type') and m.type == "ai" and hasattr(m, 'content'):
                    ai_reply = m.content
                    break
            ai_reply = re.sub(r'```json\s*[\s\S]*?\s*```', '', str(ai_reply)).strip()
            response_parts.append(
                ai_reply
                if ai_reply
                else "模型没有返回可展示内容。请重试一次；如果连续出现，请检查模型额度和可选搜索配置。"
            )

        # 为每道菜发送对应生成的图片
        response_content = "\n\n---\n\n".join(response_parts)
        msg = cl.Message(content="")
        await msg.send()
        await msg.stream_token(response_content)
        await msg.update()

        # 展示 Tavily 搜索的参考图（每道菜 2~3 张 base64）
        if dishes:
            reference_errors = []
            for dish_idx, dish in enumerate(dishes):
                ref_imgs = dish.get("tavily_reference_images", [])
                if ref_imgs:
                    label = f"📸 {dish.get('dish_name', '')} 参考图" if len(dishes) > 1 else "📸 参考图预览"
                    img_elements = [
                        cl.Image(
                            name=f"ref-{dish.get('dish_name', '')}-{j}",
                            display="inline",
                            url=img_data,
                            size="medium"
                        )
                        for j, img_data in enumerate(ref_imgs)
                    ]
                    await cl.Message(content=label, elements=img_elements).send()
                elif dish.get("reference_image_error"):
                    reference_errors.append(dish["reference_image_error"])

            if reference_errors:
                await cl.Message(
                    content=f"⚠️ 参考图未生成：{reference_errors[0]}"
                ).send()

            # 「生成外卖优化图」按钮
            if WANX_API_KEY:
                await cl.Message(
                    content="",
                    actions=[
                        cl.Action(
                            name="generate_wanx_images",
                            label="🖼️ 生成外卖优化图（通义万相）",
                            description="调用通义万相为每道菜生成高清外卖主图",
                            payload={}
                        )
                    ]
                ).send()

    except Exception as e:
        import traceback
        traceback.print_exc()
        await cl.Message(content=f"处理请求时发生错误：{str(e)}").send()

# 启动应用的函数
def run_app():
    """运行Chainlit应用"""
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # 这个函数主要用于开发调试
    pass
