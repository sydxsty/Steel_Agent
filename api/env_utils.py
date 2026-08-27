from dotenv import load_dotenv
import os
from pathlib import Path

# 强制指定 .env 文件路径（解决 VSCode 工作目录不在项目根导致找不到 .env 的问题）
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.8-max")
