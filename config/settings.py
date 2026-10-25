import os
from dotenv import load_dotenv

load_dotenv()

# ========================
# 通用配置
# ========================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 阿里云OSS配置
ALIBABA_CLOUD_ACCESS_KEY_ID = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME")

# ========================
# 模型配置 - 支持三种提供商
# ========================
MULTIMODAL_MODEL = os.getenv("MULTIMODAL_MODEL", "gpt-4-vision-preview")
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-3.5-turbo")

# 自动识别模型提供商，设置对应的 API Key 和 API Base
def _detect_model_config():
    """
    自动检测使用的是哪种模型提供商，返回 (api_key, api_base) 元组。
    优先级：QWEN > DEEPSEEK > OPENAI
    """
    qwen_key = os.getenv("QWEN_API_KEY")
    if qwen_key:
        return qwen_key, os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return deepseek_key, os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    # 默认使用 OpenAI
    return os.getenv("OPENAI_API_KEY", ""), None

MODEL_API_KEY, MODEL_API_BASE = _detect_model_config()

# ========================
# 其他配置
# ========================
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "12"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "8000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MODEL_ENABLE_THINKING = os.getenv("MODEL_ENABLE_THINKING", "false").lower() in {
    "1", "true", "yes", "on",
}

# ========================
# 通义万相文生图配置
# ========================
WANX_API_KEY = os.getenv("QWEN_API_KEY", "")  # 与千问共用同一个 DashScope API Key
_DEFAULT_WANX_MODEL = "qwen-image-2.1-turbo"
WANX_MODEL = os.getenv("WANX_MODEL", _DEFAULT_WANX_MODEL)
if not WANX_MODEL.startswith(("qwen-image", "wanx")):
    print(
        f"[settings] WANX_MODEL={WANX_MODEL!r} 不是文生图模型，"
        f"已回退到 {_DEFAULT_WANX_MODEL}"
    )
    WANX_MODEL = _DEFAULT_WANX_MODEL
