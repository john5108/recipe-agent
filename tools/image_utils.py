import requests
import base64
from typing import Any, Dict, Optional, List


def url_to_base64(img_url: str, timeout: int = 8) -> Optional[str]:
    """
    后端代理下载网络图片 → 转 base64 Data URI，彻底绕过浏览器防盗链。
    搭配 Chainlit 的 cl.Image(content=data_uri) 使用即可正常渲染。

    Args:
        img_url: 网络图片 URL
        timeout: 下载超时秒数

    Returns:
        data:image/xxx;base64,... 格式字符串，失败返回 None
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(img_url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            b64 = base64.b64encode(resp.content).decode("utf-8")
            mime = resp.headers.get("Content-Type", "image/jpeg")
            return f"data:{mime};base64,{b64}"
    except Exception:
        pass
    return None


def search_dish_images_tavily(dish_name: str, max_images: int = 3) -> Dict[str, Any]:
    """
    通过 Tavily 搜索菜品的参考图片，返回 base64 Data URI 列表。

    Args:
        dish_name: 菜品名称（用于搜索关键词）
        max_images: 最多返回图片数量

    Returns:
        包含 images、error 和 error_type 的字典
    """
    from config.settings import TAVILY_API_KEY

    if not TAVILY_API_KEY:
        print("[tavily_img] TAVILY_API_KEY 未配置，跳过图片搜索")
        return {
            "images": [],
            "error": "Tavily API Key 未配置。",
            "error_type": "configuration",
        }

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        # 加限定词：美食 成品 实拍，过滤步骤图、表情包、logo
        results = client.search(
            query=f"{dish_name} 美食 成品 实拍",
            search_depth="advanced",
            include_images=True,
            max_results=5,
        )

        image_urls = results.get("images", [])
        if not image_urls:
            print(f"[tavily_img] [{dish_name}] 未搜索到图片")
            return {
                "images": [],
                "error": "未搜索到可用参考图。",
                "error_type": "not_found",
            }

        print(f"[tavily_img] [{dish_name}] 搜索到 {len(image_urls)} 张图片，开始转 base64...")

        valid_images = []
        for img_url in image_urls:
            if isinstance(img_url, dict):
                img_url = img_url.get("url", "")
            if not img_url:
                continue
            b64 = url_to_base64(img_url)
            if b64:
                valid_images.append(b64)
                if len(valid_images) >= max_images:
                    break

        print(f"[tavily_img] [{dish_name}] 成功转换 {len(valid_images)} 张")
        if valid_images:
            return {"images": valid_images, "error": "", "error_type": ""}
        return {
            "images": [],
            "error": "参考图下载失败，可能是网络或图片防盗链限制。",
            "error_type": "download",
        }

    except ImportError:
        print("[tavily_img] tavily-python 未安装")
        return {
            "images": [],
            "error": "tavily-python 未安装。",
            "error_type": "dependency",
        }
    except Exception as e:
        print(f"[tavily_img] [{dish_name}] 搜索失败: {e}")
        error_text = str(e)
        is_auth_error = "unauthorized" in error_text.lower() or "invalid api key" in error_text.lower()
        return {
            "images": [],
            "error": (
                "Tavily API Key 无效或已过期，请更新 .env 中的 TAVILY_API_KEY。"
                if is_auth_error
                else f"Tavily 参考图搜索失败：{error_text}"
            ),
            "error_type": "auth" if is_auth_error else "request",
        }


def analyze_dish_visuals_for_prompt(
    dish_name: str,
    ingredients: List[str],
    reference_images_b64: List[str]
) -> str:
    """
    分析 Tavily 搜到的参考图，提取菜品视觉特征（摆盘、颜色、食材呈现），
    用于优化通义万相的生图 Prompt，让生成图更贴合真实菜品。

    只取前 2 张参考图发给多模态模型做分析，分析结果 50 字以内。

    Args:
        dish_name: 菜品名称
        ingredients: 食材列表
        reference_images_b64: Tavily 搜图转的 base64 Data URI 列表

    Returns:
        视觉特征描述字符串，失败返回空字符串
    """
    if not reference_images_b64:
        return ""

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from config.settings import (
            MULTIMODAL_MODEL,
            TEMPERATURE,
            MODEL_ENABLE_THINKING,
            MODEL_API_KEY,
            MODEL_API_BASE,
        )

        llm_kwargs = {"model": MULTIMODAL_MODEL, "temperature": TEMPERATURE}
        if MODEL_API_BASE:
            llm_kwargs["openai_api_key"] = MODEL_API_KEY
            llm_kwargs["base_url"] = MODEL_API_BASE
            if "dashscope" in MODEL_API_BASE or MULTIMODAL_MODEL.startswith("qwen"):
                llm_kwargs["extra_body"] = {"enable_thinking": MODEL_ENABLE_THINKING}

        multimodal_llm = ChatOpenAI(**llm_kwargs)

        # 构建多模态消息：文字指令 + 前 2 张参考图
        content_parts = [
            {
                "type": "text",
                "text": (
                    f'分析以下"{dish_name}"（食材：{"、".join(ingredients[:4])}）的参考图片，'
                    f'简要总结（50字以内）：摆盘风格、汤汁/酱料颜色、主打食材的视觉呈现方式。'
                    f'输出格式：摆盘：xxx，色系：xxx，呈现：xxx'
                )
            }
        ]
        for img_b64 in reference_images_b64[:2]:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })

        print(f"[visual_analyze] 分析 [{dish_name}] 的 {len(reference_images_b64[:2])} 张参考图...")
        response = multimodal_llm.invoke([HumanMessage(content=content_parts)])
        summary = response.content.strip()
        print(f"[visual_analyze] [{dish_name}] 视觉总结: {summary[:80]}...")
        return summary

    except ImportError:
        print("[visual_analyze] langchain_openai 不可用，跳过视觉分析")
        return ""
    except Exception as e:
        print(f"[visual_analyze] [{dish_name}] 分析失败: {e}")
        return ""
