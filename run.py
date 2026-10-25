#!/usr/bin/env python3
"""
智能菜谱推荐助手启动脚本
"""

import subprocess
import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

def check_dependencies():
    """检查依赖项"""
    try:
        import chainlit
        import langchain
        import langgraph
        import tavily
        import openai
        import oss2
        print("✅ 所有依赖项已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖项: {e}")
        print("请先运行: pip install -r requirements.txt")
        return False

def check_env_vars():
    """检查环境变量"""
    env_file = ENV_FILE
    if not env_file.exists():
        print("⚠️  未检测到 .env 文件，请复制 .env.example 并填入正确的API密钥")
        return False

    values = {}
    with open(env_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")

    model_keys = ("QWEN_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")
    valid_model_keys = [
        key for key in model_keys
        if values.get(key) and "your_" not in values[key].lower()
    ]
    if not valid_model_keys:
        print("❌ .env 至少需要一个有效模型密钥：QWEN_API_KEY、DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        return False

    optional_features = {
        "Tavily 参考图搜索": "TAVILY_API_KEY",
        "阿里云 OSS": "OSS_BUCKET_NAME",
    }
    for label, key in optional_features.items():
        value = values.get(key, "")
        if not value or "your_" in value.lower():
            print(f"⚠️  {label}未配置，对应功能会自动跳过")

    print(f"✅ 环境配置可用，模型密钥: {', '.join(valid_model_keys)}")
    return True

def run_chainlit():
    """运行Chainlit应用"""
    try:
        # 运行chainlit
        subprocess.run([
            sys.executable, "-m", "chainlit", "run", "app/ui.py",
            "--host", "127.0.0.1", "--port", "8001",
        ], check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
    except KeyboardInterrupt:
        print("\n👋 已停止服务")

def main():
    """主函数"""
    os.chdir(PROJECT_ROOT)
    print("🍽️  智能菜谱推荐助手")
    print("=" * 50)

    if not check_dependencies():
        sys.exit(1)

    if not check_env_vars():
        print("\n💡 提示: 请确保正确配置了环境变量后再运行")
        sys.exit(1)

    print("\n🚀 正在启动服务...")
    print("🌐 访问地址: http://127.0.0.1:8001")
    print("⌨️  按 Ctrl+C 停止服务\n")

    run_chainlit()

if __name__ == "__main__":
    main()
