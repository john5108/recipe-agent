from typing import Dict, TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from pydantic import ValidationError

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from app.schemas import RecipeOutput
from tools.recipe_tools import tools
from tools.image_utils import search_dish_images_tavily, analyze_dish_visuals_for_prompt
from config.settings import (
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_MESSAGES,
    MAX_TOOL_RESULT_CHARS,
    MAX_TOKENS,
    MULTIMODAL_MODEL,
    TEXT_MODEL,
    TEMPERATURE,
    MODEL_ENABLE_THINKING,
    MODEL_API_KEY,
    MODEL_API_BASE,
    WANX_API_KEY,
    WANX_MODEL,
)
import json
import re
import time

# 通义万相文生图 SDK
try:
    import dashscope
    from dashscope import ImageSynthesis, MultiModalConversation
    dashscope.api_key = WANX_API_KEY
    _wanx_available = bool(WANX_API_KEY)
except ImportError:
    _wanx_available = False
    print("[agent] dashscope 未安装，文生图功能不可用。请执行: pip install dashscope")


class ImageGenerationError(RuntimeError):
    """Raised when the configured image model cannot return an image URL."""


def normalize_final_recipe(value: object) -> Dict:
    """Keep optional recipe state safe for callers that use mapping methods."""
    return value if isinstance(value, dict) else {}


def _classify_intent_by_rules(text: str) -> Optional[str]:
    """Handle clear greeting and menu requests without spending an LLM call."""
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized:
        return "greeting"

    if re.match(r"^(那|那么|还有|其他).*(呢|吗)[？?]?$", normalized):
        return "query"

    query_keywords = (
        "历史",
        "查询",
        "搜索",
        "趋势",
        "爆款",
        "参考图",
        "价格参考",
        "以前做过",
    )
    if any(keyword in normalized for keyword in query_keywords):
        return "query"

    recipe_keywords = (
        "食材",
        "菜谱",
        "菜单",
        "菜品",
        "道菜",
        "做什么菜",
        "能做什么菜",
        "怎么搭配",
        "研发",
        "上新",
        "冰箱",
        "做菜",
        "做成菜",
        "来一道",
        "出一道",
    )
    if any(keyword in normalized for keyword in recipe_keywords):
        return "recipe"

    greeting_patterns = (
        "你好",
        "您好",
        "在吗",
        "嗨",
        "哈喽",
        "hello",
        "hi",
        "你能做什么",
        "介绍一下",
    )
    if any(normalized.lower().startswith(pattern) for pattern in greeting_patterns):
        return "greeting"
    return None


def _classify_image_intent(text: str) -> str:
    """Decide whether an image request asks for inventory only or recipes."""
    normalized = re.sub(r"\s+", "", text or "")
    recipe_terms = (
        "做菜",
        "做什么",
        "做什么菜",
        "菜谱",
        "菜单",
        "菜品",
        "研发",
        "推荐",
        "怎么搭配",
        "出餐",
        "上新",
        "做几道",
    )
    if any(term in normalized for term in recipe_terms):
        return "recipe"
    return "ingredient_scan"


def _parse_recipe_output(content: object) -> Optional[Dict]:
    """Parse and validate a recipe payload from model text."""
    if not isinstance(content, str) or not content.strip():
        return None

    candidates = [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            content,
            re.IGNORECASE,
        )
    ]

    first_object = content.find("{")
    if first_object >= 0:
        try:
            decoded, _ = json.JSONDecoder().raw_decode(content[first_object:])
            candidates.append(json.dumps(decoded, ensure_ascii=False))
        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        try:
            recipe_data = json.loads(candidate)
            if isinstance(recipe_data, list):
                dishes_payload = recipe_data
            elif isinstance(recipe_data, dict):
                dishes_payload = recipe_data.get("dishes", [recipe_data])
            else:
                continue

            validated = RecipeOutput.model_validate({"dishes": dishes_payload})
            dishes = [dish.model_dump() for dish in validated.dishes]
            if dishes:
                return {"dishes": dishes}
        except (json.JSONDecodeError, ValidationError):
            continue
    return None


def _generate_dish_image(dish_name: str, ingredients: List[str], visual_hints: str = "") -> str:
    """
    调用通义万相 API 生成外卖菜品主图。

    Args:
        dish_name: 菜品名称
        ingredients: 食材列表
        visual_hints: 从参考图分析出的视觉特征，用于优化生图 Prompt

    Returns:
        生成的图片 URL（有效期 24 小时），失败返回空字符串
    """
    if not _wanx_available:
        raise ImageGenerationError(
            "通义万相不可用：请检查 QWEN_API_KEY 和 dashscope 安装。"
        )

    ingredient_hint = "、".join(ingredients[:3]) if ingredients else dish_name
    prompt = (
        f"高清美食摄影图，{dish_name}，主要食材有{ingredient_hint}"
    )
    if visual_hints:
        prompt += f"，参考风格：{visual_hints}"
    prompt += (
        "，外卖平台菜品主图视角，暖光氛围，精致摆盘，食欲感拉满，"
        "干净简约背景，俯拍45度角，4K画质，商业美食摄影风格"
    )

    print(f"[wanx] 正在生成图片: {dish_name}...")
    if visual_hints:
        print(f"[wanx] 视觉参考: {visual_hints[:60]}...")

    try:
        if WANX_MODEL.startswith("qwen-image"):
            rsp = MultiModalConversation.call(
                api_key=WANX_API_KEY,
                model=WANX_MODEL,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt}],
                }],
                prompt_extend=True,
            )
            if rsp.status_code == 200 and rsp.output and rsp.output.choices:
                content = rsp.output.choices[0].message.content
                if content and isinstance(content[0], dict):
                    img_url = content[0].get("image", "")
                    if img_url:
                        print(f"[wanx] 图片生成成功: {img_url[:60]}...")
                        return img_url
        else:
            rsp = ImageSynthesis.call(
                model=WANX_MODEL,
                prompt=prompt,
                n=1,
                size="1024*1024",
                watermark=False,
            )
            if rsp.status_code == 200 and rsp.output and rsp.output.results:
                img_url = rsp.output.results[0].url
                print(f"[wanx] 图片生成成功: {img_url[:60]}...")
                return img_url

        code = getattr(rsp, "code", "") or ""
        message = getattr(rsp, "message", "") or "未返回图片 URL"
        error = f"code={code}, message={message}".strip(", ")
        print(f"[wanx] [{dish_name}] 生成失败: {error}")
        raise ImageGenerationError(error)
    except ImageGenerationError:
        raise
    except Exception as e:
        print(f"[wanx] [{dish_name}] 异常: {e}")
        raise ImageGenerationError(str(e)) from e


# ========== 状态定义 ==========
def _messages_for_llm(messages: List[BaseMessage], system_prompt: str) -> List[BaseMessage]:
    """保留唯一系统提示，并按消息数和字符预算裁剪上下文。"""
    non_system_messages = [
        message for message in messages
        if not isinstance(message, SystemMessage)
    ]

    def content_size(message: BaseMessage) -> int:
        content = message.content
        if isinstance(content, str):
            return len(content)
        return len(json.dumps(content, ensure_ascii=False, default=str))

    def clip_message(message: BaseMessage, limit: int) -> BaseMessage:
        content = message.content
        if isinstance(content, str):
            if len(content) <= limit:
                return message
            clipped = content[:limit] + "\n[历史内容已截断]"
        else:
            serialized = json.dumps(content, ensure_ascii=False, default=str)
            if len(serialized) <= limit:
                return message
            clipped = serialized[:limit] + "\n[历史内容已截断]"
        return message.model_copy(update={"content": clipped})

    recent_messages = non_system_messages[-max(1, MAX_CONTEXT_MESSAGES):]
    selected_reversed = []
    used_chars = len(system_prompt)

    for message in reversed(recent_messages):
        per_message_limit = (
            MAX_TOOL_RESULT_CHARS
            if isinstance(message, ToolMessage)
            else max(1000, MAX_CONTEXT_CHARS // 2)
        )
        candidate = clip_message(message, per_message_limit)
        candidate_size = content_size(candidate)
        remaining = MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break
        if candidate_size > remaining:
            if selected_reversed:
                break
            candidate = clip_message(candidate, max(1, remaining))
            candidate_size = content_size(candidate)
        selected_reversed.append((candidate, message))
        used_chars += candidate_size

    selected_pairs = list(reversed(selected_reversed))
    selected = [candidate for candidate, _ in selected_pairs]

    # 工具结果不能脱离产生它的 AI tool_call 单独发送。
    if selected and isinstance(selected[0], ToolMessage):
        first_original = selected_pairs[0][1]
        first_index = next(
            (index for index, message in enumerate(recent_messages) if message is first_original),
            0,
        )
        for index in range(first_index - 1, -1, -1):
            previous = recent_messages[index]
            if isinstance(previous, AIMessage) and previous.tool_calls:
                selected.insert(0, previous)
                used_chars += content_size(previous)
                break

    print(
        f"[context] messages={len(selected)}/{MAX_CONTEXT_MESSAGES}, "
        f"chars={used_chars}/{MAX_CONTEXT_CHARS}"
    )
    return [SystemMessage(content=system_prompt)] + selected


class RecipeAgentState(TypedDict):
    """智能体状态定义"""
    session_id: str
    user_input: str
    ingredients: str
    intent: str  # "greeting" | "ingredient_scan" | "recipe" | "query"
    recipe_ideas: List[Dict]
    selected_recipe: Optional[Dict]
    final_recipe: Optional[Dict]
    image_url: Optional[str]
    image_data: Optional[str]
    messages: List[BaseMessage]

def create_recipe_agent():
    """创建菜谱推荐智能体"""
    # 初始化大模型
    # 根据 settings 自动选择 OpenAI / Qwen / DeepSeek
    llm_kwargs = {
        "model": TEXT_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    multimodal_kwargs = {
        "model": MULTIMODAL_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    if MODEL_API_BASE:
        llm_kwargs["openai_api_key"] = MODEL_API_KEY
        llm_kwargs["base_url"] = MODEL_API_BASE
        multimodal_kwargs["openai_api_key"] = MODEL_API_KEY
        multimodal_kwargs["base_url"] = MODEL_API_BASE
        if "dashscope" in MODEL_API_BASE or TEXT_MODEL.startswith("qwen"):
            llm_kwargs["extra_body"] = {"enable_thinking": MODEL_ENABLE_THINKING}
            multimodal_kwargs["extra_body"] = {"enable_thinking": MODEL_ENABLE_THINKING}

    llm = ChatOpenAI(**llm_kwargs)
    multimodal_llm = ChatOpenAI(**multimodal_kwargs)
    llm_with_tools = llm.bind_tools(tools)

    def classify_intent(state: RecipeAgentState):
        """轻量意图分类：判断用户是闲聊还是需要生成菜谱（<0.5s，极低 token 消耗）"""
        image_data = state.get("image_data", "")
        user_text = state.get("user_input") or state.get("ingredients", "")

        # 有图片：只有明确要求研发时才直接出菜谱。
        if image_data:
            intent = _classify_image_intent(user_text)
            print(f"[classify_intent] 检测到图片 → 意图: {intent}")
            return {"intent": intent}

        rule_intent = _classify_intent_by_rules(user_text)
        if rule_intent:
            print(f"[classify_intent] 规则命中 → 意图: {rule_intent}")
            return {"intent": rule_intent}

        # 纯文字 → 用极小 prompt 做快速分类
        classify_prompt = """判断用户输入属于以下哪种意图，只输出数字编号，不要任何其他内容：
1 - 闲聊问候（打招呼、随便聊天、问功能）
2 - 根据食材研发菜品
3 - 查询历史菜品、搜索趋势或参考信息、查询菜品库数据

用户输入：{user_input}
意图编号："""

        try:
            response = llm.invoke([
                HumanMessage(content=classify_prompt.format(user_input=user_text))
            ])
            result_text = response.content.strip()
            print(f"[classify_intent] 用户输入: {user_text[:60]}, LLM 返回: {result_text}")

            # 只接受独立的 1/2/3，避免把 "2026" 之类的文本误判为菜谱。
            match = re.search(r"(?<!\d)([123])(?!\d)", result_text)
            selected = match.group(1) if match else "1"
            intent = {"1": "greeting", "2": "recipe", "3": "query"}[selected]
        except Exception as e:
            print(f"[classify_intent] 分类失败: {e}，默认走菜谱流程")
            intent = "recipe"

        print(f"[classify_intent] 最终意图: {intent}")
        return {"intent": intent}

    def greeting_response(state: RecipeAgentState):
        """闲聊回复节点：对问候/闲聊做简短友好回应，引导用户提供食材"""
        user_text = state.get("user_input") or state.get("ingredients", "")
        messages = state.get("messages", [])

        greeting_prompt = """你是友好的外卖菜品研发助手。用户正在打招呼或闲聊。
请简短友好地回复（2-3句话），介绍你能帮他做菜谱研发，引导他提供食材或上传食材图片。"""

        user_message = HumanMessage(
            content=user_text.strip() if user_text.strip() else "你好"
        )
        recent_history = [
            message for message in messages
            if not isinstance(message, SystemMessage)
        ][-6:]
        greeting_messages = [
            SystemMessage(content=greeting_prompt)
        ] + recent_history + [user_message]

        try:
            response = llm.invoke(greeting_messages)
            messages.append(user_message)
            messages.append(response)
            print(f"[greeting_response] 回复长度: {len(response.content or '')}")
        except Exception as e:
            print(f"[greeting_response] LLM 调用失败: {e}")
            messages.append(user_message)
            messages.append(AIMessage(content="你好！有什么食材方面的需要帮忙的吗？"))

        return {"messages": messages}

    def recognize_ingredients_from_image(state: RecipeAgentState):
        """食材图片识别节点：调用多模态模型识别图片中的食材，同时作为统一入口构建首条人类消息"""
        image_data = state.get("image_data", "")
        user_text = str(state.get("user_input") or "").strip()
        existing_ingredients = str(state.get("ingredients") or "").strip()
        intent = state.get("intent", "recipe")
        messages = state.get("messages", [])

        if image_data:
            print(f"[recognize] 收到图片数据，大小: {len(image_data)} 字符 (base64)")

            try:
                # 按千问多模态标准格式构建请求
                response = multimodal_llm.invoke([
                    HumanMessage(content=[
                        {
                            "type": "text",
                            "text": "识别图片中所有可烹饪的食材，只输出食材名称，用中文逗号分隔，不要其他内容。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ])
                ])

                content = response.content
                if isinstance(content, list):
                    recognized = "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    ).strip()
                else:
                    recognized = str(content or "").strip()
                print(f"[recognize] 识别结果: {recognized}")

                if not recognized:
                    raise ValueError("多模态模型没有返回食材名称")

                user_message = HumanMessage(
                    content=user_text or "请识别图片中的食材"
                )
                messages.append(user_message)

                if intent == "ingredient_scan":
                    ingredient_names = [
                        item.strip()
                        for item in re.split(r"[，,、\n;；]+", recognized)
                        if item.strip()
                    ]
                    reply = "识别到以下食材：\n" + "\n".join(
                        f"- {item}" for item in ingredient_names
                    )
                    messages.append(AIMessage(content=reply))
                    return {
                        "ingredients": recognized,
                        "image_data": "",
                        "messages": messages,
                    }

                full_input = (
                    f"{user_text}\n" if user_text else ""
                ) + (
                    f"（图片中识别到的食材：{recognized}）\n"
                    "请基于这些食材设计外卖菜品。"
                )
                messages[-1] = HumanMessage(content=full_input)
                return {
                    "ingredients": recognized,
                    "image_data": "",
                    "messages": messages
                }

            except Exception as e:
                import traceback
                print(f"[recognize] 图片识别失败: {e}")
                traceback.print_exc()

                if intent == "ingredient_scan":
                    messages.append(HumanMessage(
                        content=user_text or "请识别图片中的食材"
                    ))
                    messages.append(AIMessage(
                        content="图片识别失败，请换一张更清晰、光线更亮的食材照片。"
                    ))
                    return {
                        "ingredients": existing_ingredients,
                        "image_data": "",
                        "messages": messages,
                    }

                # 识别失败，回退到文字输入
                user_message = HumanMessage(content=user_text or "请帮我推荐一些外卖菜品")
                messages.append(user_message)

                return {
                    "ingredients": user_text,
                    "image_data": "",
                    "messages": messages
                }
        else:
            # 无图片：纯文字输入入口
            current_input = user_text or existing_ingredients
            print(f"[recognize] 无图片，使用文字输入: {current_input[:50] if current_input else '(空)'}")
            model_input = current_input or "请帮我推荐一些外卖菜品"
            if existing_ingredients and existing_ingredients not in model_input:
                model_input += f"\n（当前会话已有食材：{existing_ingredients}）"
            user_message = HumanMessage(content=model_input)
            messages.append(user_message)

            return {
                "ingredients": existing_ingredients or current_input,
                "messages": messages,
            }

    def call_model(state: RecipeAgentState):
        """调用语言模型获取菜谱建议"""
        import traceback
        messages = state.get("messages", [])
        ingredients = state.get("ingredients", "")

        # 调试日志：打印用户输入和初始食材状态
        user_input = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "human":
                user_input = msg.content
                break
        print("🔍 接收到的用户输入：", user_input)
        print("🔍 初始状态食材字段：", repr(ingredients) if ingredients else "空")

        # 普通研发和工具查询使用不同输出协议，避免重复生成正文和 JSON。
        if state.get("intent") == "query":
            system_prompt = """
你是外卖菜品研发与上架顾问，当前任务是查询历史、趋势、价格参考或菜品库信息。
需要事实数据时必须调用可用工具，不允许编造历史记录或搜索结果。
工具失败时直接说明失败原因，并给出不依赖该工具的简短建议。
使用简洁中文回答，不要输出 JSON，不要声称已经完成未实际执行的操作。
"""
        else:
            system_prompt = """
你是专业的外卖菜品研发与上架顾问，服务于外卖中小商家。
根据用户提供的食材、预算、口味和经营约束，设计 3~5 道可落地、差异明显的外卖菜品。

【工作原则】
1. 菜品要出餐快、耐运输、成本可控、适合标准化。
2. 多食材时尽量采用不同组合和烹饪方式，不要所有菜使用同一组食材。
3. 只在用户现有食材基础上补充常见辅料，不虚构主料。
4. 步骤控制在 5 步以内，适合外卖后厨操作。
5. 成本和售价给出合理估算区间，不把估算说成真实采购数据。
6. 上架文案要短、清楚、符合一般外卖平台表达习惯，不做医疗功效承诺。

【唯一输出格式】
只输出一个合法 JSON 对象，不要 Markdown 代码块，不要解释，不要输出 JSON 之外的文字。
顶层必须是 {"dishes": [...]}，每道菜必须包含以下字段：
{
  "dish_name": "菜名",
  "ingredients": ["主料和用量", "辅料和用量"],
  "steps": ["步骤1", "步骤2"],
  "cooking_time": "预计出餐时间",
  "difficulty_level": "简单/中等/困难",
  "cost_estimation": "单份成本估算",
  "suggested_price": "建议售价",
  "nutritional_benefits": ["营养特点"],
  "main_image_url": "",
  "core_selling_points": ["卖点1", "卖点2", "卖点3"],
  "marketing_title": "20字以内上架标题",
  "marketing_description": "60-100字详情描述",
  "tags": ["标签1", "标签2", "标签3"],
  "image_suggestion": "外卖主图拍摄建议"
}
确保 JSON 完整闭合。不要输出历史查询结果，不要调用工具。
"""

        # 历史里只保留一个系统提示，并限制送入模型的上下文消息数
        messages = [
            message for message in messages
            if not isinstance(message, SystemMessage)
        ]
        llm_messages = _messages_for_llm(messages, system_prompt)

        # 调用模型
        try:
            # 调用LLM前，打印完整的输入内容
            print(f"[call_model] Invoking LLM with {len(llm_messages)} messages...")
            selected_llm = llm_with_tools if state.get("intent") == "query" else llm
            response = selected_llm.invoke(llm_messages)
            print(f"[call_model] LLM response received, content length: {len(response.content) if response.content else 0}")
        except Exception as e:
            print(f"[call_model] LLM invocation FAILED: {e}")
            traceback.print_exc()
            raise

        # 更新消息历史
        messages.append(response)

        return {"messages": messages}

    def handle_tool_calls(state: RecipeAgentState):
        """处理工具调用"""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # 执行工具调用
            for tool_call in last_message.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                # 查找对应的工具
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    try:
                        # 执行工具
                        tool_response = tool.invoke(tool_args)
                        if not isinstance(tool_response, str):
                            tool_response = json.dumps(
                                tool_response,
                                ensure_ascii=False,
                                default=str,
                            )
                        if len(tool_response) > MAX_TOOL_RESULT_CHARS:
                            tool_response = (
                                tool_response[:MAX_TOOL_RESULT_CHARS]
                                + "\n[工具结果已截断]"
                            )

                        # 添加工具响应到消息历史（ToolMessage + tool_call_id）
                        tool_message = ToolMessage(
                            content=tool_response,
                            name=tool_name,
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(tool_message)
                    except Exception as e:
                        error_msg = f"Tool {tool_name} failed: {str(e)}"
                        tool_message = ToolMessage(
                            content=error_msg,
                            name=tool_name,
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(tool_message)
                else:
                    messages.append(ToolMessage(
                        content=f"Tool {tool_name} not found",
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    ))

        return {"messages": messages}

    def extract_recipe_info(state: RecipeAgentState):
        """从模型回复中提取菜谱；格式不完整时做一次严格 JSON 修复。"""
        messages = state.get("messages", [])

        # 获取最后一条 AI 消息的完整内容
        ai_content = ""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == "ai" and hasattr(msg, 'content'):
                ai_content = msg.content
                break

        if not isinstance(ai_content, str) or not ai_content.strip():
            print("[extract_recipe_info] 模型回复为空，跳过结构化提取")
            return {
                "final_recipe": None,
                "image_url": None,
                "messages": messages,
            }

        final_recipe = _parse_recipe_output(ai_content)
        if final_recipe:
            dishes = final_recipe.get("dishes", [])
            print(f"[extract_recipe_info] 从 AI 回复直接解析到 {len(dishes)} 道菜品")
        else:
            print("[extract_recipe_info] 未找到有效 JSON，请求模型整理为严格 JSON")
            schema = json.dumps(
                RecipeOutput.model_json_schema(),
                ensure_ascii=False,
            )
            repair_prompt = f"""请把下面的菜品方案整理成严格 JSON。
只输出 JSON，不要 Markdown 代码块，不要补充解释。
必须符合以下 JSON Schema：
{schema}

菜品方案：
{str(ai_content)[:12000]}
"""
            try:
                repair_response = llm.invoke([
                    SystemMessage(content="你是严格的数据整理器，只输出合法 JSON。"),
                    HumanMessage(content=repair_prompt),
                ])
                final_recipe = _parse_recipe_output(repair_response.content)
            except Exception as e:
                print(f"[extract_recipe_info] JSON 修复调用失败: {e}")

        if final_recipe:
            dishes = final_recipe.get("dishes", [])
            print(f"[extract_recipe_info] 结构化菜谱就绪，共 {len(dishes)} 道菜品")
            return {
                "final_recipe": final_recipe,
                "image_url": dishes[0].get("main_image_url") if dishes else None,
                "messages": messages,
            }

        # 不再用整段用户输入拼接假菜名。保留模型原文，由 UI 正常展示。
        print("[extract_recipe_info] 无法得到有效结构化菜谱，跳过菜品保存和配图")
        return {
            "final_recipe": None,
            "image_url": None,
            "messages": messages,
        }

    def search_reference_images(state: RecipeAgentState):
        """
        Tavily 参考图搜索节点：为每道菜搜索 2~3 张真实美食图片，
        后端代理转 base64 后存储，解决前端防盗链问题。
        """
        final_recipe = normalize_final_recipe(state.get("final_recipe"))
        dishes = final_recipe.get("dishes", [])

        if not dishes:
            print("[search_images] 无菜品，跳过图片搜索")
            return {}

        import time as _time
        auth_failed = False
        for i, dish in enumerate(dishes):
            dish_name = dish.get("dish_name", f"菜品{i+1}")
            print(f"[search_images] 搜索第 {i+1}/{len(dishes)} 道菜参考图: {dish_name}")

            if auth_failed:
                dish["tavily_reference_images"] = []
                dish["reference_image_error"] = "Tavily API Key 无效，已跳过后续搜索。"
                continue

            search_result = search_dish_images_tavily(dish_name, max_images=3)
            dish["tavily_reference_images"] = search_result.get("images", [])
            if search_result.get("error"):
                dish["reference_image_error"] = search_result["error"]
                auth_failed = search_result.get("error_type") == "auth"

            if i < len(dishes) - 1:
                _time.sleep(0.3)

        total = sum(len(d.get("tavily_reference_images", [])) for d in dishes)
        print(f"[search_images] 完成！共获取 {total} 张参考图")
        return {"final_recipe": final_recipe}

    def optimize_dish_image(state: RecipeAgentState):
        """
        图片生成节点：遍历每道菜，调用通义万相 API 生成外卖风格菜品主图。
        生成结果写入每道菜的 generated_image_url 字段。
        """
        final_recipe = normalize_final_recipe(state.get("final_recipe"))
        if not final_recipe:
            return {"image_url": None, "final_recipe": final_recipe}

        dishes = final_recipe.get("dishes", [])
        if not dishes:
            return {"image_url": None, "final_recipe": final_recipe}

        if not _wanx_available:
            print("[optimize_image] 通义万相不可用（未安装 dashscope 或未配置 API Key），跳过图片生成")
            return {"final_recipe": final_recipe}

        first_url = ""
        for i, dish in enumerate(dishes):
            dish_name = dish.get("dish_name", f"菜品{i+1}")
            ingredients = dish.get("ingredients", [])

            # 优化：取 Tavily 参考图，多模态分析视觉特征，优化生图 Prompt
            visual_hints = ""
            ref_images = dish.get("tavily_reference_images", [])
            if ref_images:
                try:
                    visual_hints = analyze_dish_visuals_for_prompt(
                        dish_name, ingredients, ref_images
                    )
                except Exception as e:
                    print(f"[optimize_image] [{dish_name}] 视觉分析失败: {e}")

            print(f"[optimize_image] 生成第 {i+1}/{len(dishes)} 道菜图片: {dish_name}")
            try:
                img_url = _generate_dish_image(dish_name, ingredients, visual_hints)
            except ImageGenerationError as e:
                img_url = ""
                dish["generated_image_error"] = str(e)
            dish["generated_image_url"] = img_url

            if img_url and not first_url:
                first_url = img_url

            # 多道菜时加间隔，避免 API 限流
            if i < len(dishes) - 1:
                time.sleep(0.5)

        print(f"[optimize_image] 完成！共生成 {len(dishes)} 道菜图片")
        return {"final_recipe": final_recipe, "image_url": first_url}


    # 创建状态图
    workflow = StateGraph(RecipeAgentState)

    # 添加节点
    workflow.add_node("classify_intent", classify_intent)         # 入口：意图分类
    workflow.add_node("greeting_response", greeting_response)     # 闲聊回复
    workflow.add_node("recognize_image", recognize_ingredients_from_image)  # 识图/文字处理
    workflow.add_node("model", call_model)
    workflow.add_node("tools", handle_tool_calls)
    workflow.add_node("extract", extract_recipe_info)
    workflow.add_node("search_images", search_reference_images)   # Tavily 参考图搜索（默认分支）
    workflow.add_node("optimize_image", optimize_dish_image)      # 通义万相生图（用户手动触发）

    # 定义边
    workflow.set_entry_point("classify_intent")

    # classify_intent → 条件分流
    def route_intent(state: RecipeAgentState) -> str:
        intent = state.get("intent", "recipe")
        if intent == "greeting":
            print("🔀 意图分流: → greeting_response")
            return "greeting"
        print(f"🔀 意图分流: → recognize_image（{intent}）")
        return "recipe"

    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "greeting": "greeting_response",
            "recipe": "recognize_image"
        }
    )

    workflow.add_edge("greeting_response", "__end__")

    def route_after_recognize(state: RecipeAgentState) -> str:
        if state.get("intent") == "ingredient_scan":
            print("🔀 识图完成：只返回食材清单，不生成菜谱")
            return "__end__"
        return "model"

    workflow.add_conditional_edges(
        "recognize_image",
        route_after_recognize,
        {
            "model": "model",
            "__end__": "__end__",
        },
    )
    workflow.add_edge("tools", "model")
    workflow.add_edge("extract", "search_images")     # 菜谱提取后 → Tavily 搜索参考图
    workflow.add_edge("search_images", "__end__")     # 默认分支：参考图搜索完成 → 返回前端
    # optimize_image 不连接默认边，由用户手动触发（UI 按钮调用）

    # 条件边：model 节点输出后
    #   1. 有工具调用 → 执行工具
    #   2. AI 回复包含菜谱内容 → 进入提取节点
    #   3. AI 回复是闲聊/问候 → 直接结束
    def should_continue(state: RecipeAgentState) -> str:
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if (hasattr(last_message, 'tool_calls') and
                last_message.tool_calls):
                return "tools"

            if hasattr(last_message, 'content') and last_message.content:
                if state.get("intent") == "recipe":
                    print("🔀 意图识别：普通研发 → 进入结构化提取")
                    return "extract"
                print("🔀 意图识别：工具查询 → 直接结束")
                return "__end__"
            else:
                print("🔀 意图识别：模型回复为空 → 直接结束")
        return "__end__"

    workflow.add_conditional_edges(
        "model",
        should_continue,
        {
            "tools": "tools",
            "extract": "extract",
            "__end__": "__end__"
        }
    )

    return workflow.compile(checkpointer=MemorySaver())
