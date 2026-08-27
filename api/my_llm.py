"""
my_llm.py - DeepSeek / Qwen 大语言模型配置模块
=============================================

项目只保留 DeepSeek V4 Flash 和 Qwen 3.8 Max 两个模型配置。
"""

from langchain.chat_models import init_chat_model

from env_utils import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
)


deepseek_Llm = init_chat_model(
    model=DEEPSEEK_MODEL,
    model_provider="openai",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    streaming=True,
    # 最终报告可能需要较长时间才返回第一个流式片段；禁用 LangChain 默认的
    # 120 秒首分块超时，避免管线钢与风电共用报告链路被提前中断。
    stream_chunk_timeout=None,
    temperature=0,
)

qwen_Llm = init_chat_model(
    model=QWEN_MODEL,
    model_provider="openai",
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    streaming=True,
    temperature=0,
)
