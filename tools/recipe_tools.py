import oss2
import requests
from config.settings import (
    TAVILY_API_KEY,
    ALIBABA_CLOUD_ACCESS_KEY_ID,
    ALIBABA_CLOUD_ACCESS_KEY_SECRET,
    OSS_ENDPOINT,
    OSS_BUCKET_NAME
)
from utils.dish_tracker import get_tracker
from langchain.tools import tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def tavily_search(query: str) -> dict:
    """
    使用Tavily API搜索网络信息，特别适合查找食谱相关内容。

    Args:
        query: 搜索查询，例如"最佳家常菜食谱"或"健康素食料理"

    Returns:
        包含搜索结果的字典
    """
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY 未配置，请在 .env 中填写有效密钥。"}

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth="advanced",
            include_images=True,
            include_answer=True,
            max_results=5
        )

        return response
    except Exception as e:
        logger.error(f"Tavily search failed: {str(e)}")
        error_text = str(e)
        if "unauthorized" in error_text.lower() or "invalid api key" in error_text.lower():
            return {"error": "Tavily API Key 无效或已过期，请更新 .env 中的 TAVILY_API_KEY。"}
        return {"error": f"Tavily 搜索失败: {error_text}"}

@tool
def upload_image_to_oss(image_url: str, filename: str) -> str:
    """
    将图片从URL上传到阿里云OSS

    Args:
        image_url: 图片的URL地址
        filename: 保存到OSS的文件名

    Returns:
        上传后的图片访问URL
    """
    try:
        # 初始化OSS客户端
        auth = oss2.Auth(ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)

        # 下载图片内容
        response = requests.get(image_url)
        response.raise_for_status()

        # 上传到OSS
        result = bucket.put_object(filename, response.content)

        if result.status == 200:
            # 返回OSS中的图片URL
            oss_url = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT.replace('https://', '')}/{filename}"
            return oss_url
        else:
            raise Exception(f"Upload failed with status: {result.status}")
    except Exception as e:
        logger.error(f"OSS upload failed: {str(e)}")
        return f"Upload failed: {str(e)}"

@tool
def get_top_delicious_recipes(ingredients: str) -> dict:
    """
    获取使用指定食材的最佳菜谱

    Args:
        ingredients: 食材列表，例如"鸡肉，土豆，胡萝卜"

    Returns:
        包含菜谱和相关图片的字典
    """
    search_query = f"使用{ingredients}制作的最受欢迎菜谱及精美图片"
    search_result = tavily_search.invoke({"query": search_query})

    if "error" in search_result:
        return search_result

    # 提取最有价值的信息
    recipes = []
    if "results" in search_result:
        for result in search_result["results"]:
            if "image" in result:
                recipe_info = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "image_url": result["image"]
                }
                recipes.append(recipe_info)

    return {
        "recipes": recipes[:3],  # 只返回前3个最佳结果
        "query": search_query
    }

@tool
def query_dishes_by_ingredient(ingredient: str, limit: int = 10) -> dict:
    """
    查询菜品研发历史库，找出包含指定食材的历史菜品。

    Args:
        ingredient: 食材名称，例如"土豆"或"牛肉"
        limit: 最多返回多少条记录，默认 10

    Returns:
        包含查询食材、命中数量和菜品记录的字典
    """
    ingredient = ingredient.strip()
    if not ingredient:
        return {"query": ingredient, "count": 0, "dishes": [], "error": "食材不能为空"}

    try:
        dishes = get_tracker().find_dishes_by_ingredient(ingredient)[:max(1, limit)]
        return {
            "query": ingredient,
            "count": len(dishes),
            "dishes": dishes,
        }
    except Exception as e:
        logger.error("Query dishes by ingredient failed: %s", e)
        return {"query": ingredient, "count": 0, "dishes": [], "error": str(e)}


@tool
def query_recent_dishes(session_id: str, limit: int = 10) -> dict:
    """
    查询某个会话最近研发的菜品记录。

    Args:
        session_id: Chainlit 会话 ID
        limit: 最多返回多少条记录，默认 10

    Returns:
        包含会话 ID、命中数量和菜品记录的字典
    """
    if not session_id.strip():
        return {"session_id": session_id, "count": 0, "dishes": [], "error": "session_id 不能为空"}

    try:
        dishes = get_tracker().get_session_history(
            session_id.strip(),
            limit=max(1, limit),
        )
        return {
            "session_id": session_id,
            "count": len(dishes),
            "dishes": dishes,
        }
    except Exception as e:
        logger.error("Query recent dishes failed: %s", e)
        return {"session_id": session_id, "count": 0, "dishes": [], "error": str(e)}

# Initialize tools list
tools = [
    tavily_search,
    upload_image_to_oss,
    get_top_delicious_recipes,
    query_dishes_by_ingredient,
    query_recent_dishes,
]
