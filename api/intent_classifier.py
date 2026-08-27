"""
intent_classifier.py — 通用 RAG + 意图分类 可复用模块
======================================================

提供 classify_with_rag() 方法，传入系统提示词、用户输入、会话ID和期望的JSON输出样例，
内部自动完成：
  1. RAG 检索（从指定数据库检索相关文档）
  2. 上下文拼接（检索结果 + 系统提示词 + JSON输出格式）
  3. 链式执行（带多轮对话记忆管理）
  4. 返回结构化 JSON 键值对结果

使用示例:
    from intent_classifier import classify_with_rag

    result = classify_with_rag(
        system_prompt="你是钢材设计意图识别助手...",
        user_message="Q460钢的热处理工艺如何优化？",
        session_id="user-001",
        json_schema={"intent": "DESIGN"},
        db_name="Nb_KnowBase_db",
    )
    # 返回: {"intent": "DESIGN"}
"""

import json
import re
import socket
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from my_llm import deepseek_Llm

# 导入会话持久化模块
from session_store import SessionStore, PersistentChatMessageHistory, register_for_cleanup

# RAG 检索超时（秒），避免 HuggingFace 下载模型卡住
_RAG_TIMEOUT = 5

# 最大保留轮数
_MAX_HISTORY_TURNS = 20

# ============================================================
# 会话历史存储 — 持久化（数据库 + 内存缓存）
# ============================================================
_intent_store = SessionStore(
    session_type="intent_classifier",
    max_turns=_MAX_HISTORY_TURNS,
    ttl=3600,
)
# 注册到后台定时清理
register_for_cleanup(_intent_store)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取指定 session_id 的持久化对话历史。

    Args:
        session_id: 会话唯一标识

    Returns:
        PersistentChatMessageHistory 实例（兼容 LangChain BaseChatMessageHistory 接口）
    """
    return PersistentChatMessageHistory(_intent_store, session_id)


def _parse_json_response(text: str, json_schema: dict) -> dict:
    """
    从 LLM 返回的文本中解析 JSON 结果。

    优先匹配 ```json ... ``` 代码块，否则尝试直接解析整个文本，
    最后用正则提取 {} 包围的 JSON。

    Args:
        text:       LLM 返回的原始文本
        json_schema: 期望的 JSON 样例（用于获取 key 列表做 fallback 校验）

    Returns:
        dict: 解析后的 JSON 键值对
    """
    # 1. 尝试提取 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. 尝试直接解析整个文本
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 3. 用正则提取第一个 {} 包围的 JSON 对象
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Fallback: 用期望的 key 列表构建默认结果
    default = {}
    for key in json_schema.keys():
        default[key] = "UNKNOWN"
    return default


# ============================================================
# 公开 API
# ============================================================

def _doc_value(doc, key: str, default=None):
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _doc_metadata(doc) -> dict:
    metadata = _doc_value(doc, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def retrieve_classify_rag_docs(
    user_message: str,
    db_name: str = "Nb_KnowBase_db",
    db_collection: str = "documents",
    top_k: int = 5,
) -> list[dict]:
    """Run the RAG retrieval used by classify_with_rag and return raw docs."""
    docs = []
    try:
        import os as _os

        _os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(_RAG_TIMEOUT))
        _os.environ.setdefault("REQUESTS_TIMEOUT", str(_RAG_TIMEOUT))

        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(_RAG_TIMEOUT)

        try:
            from hybrid_retriever import hybrid_search

            rag_executor = ThreadPoolExecutor(max_workers=1)
            future = rag_executor.submit(
                hybrid_search,
                query=user_message,
                k=top_k,
                db_name=db_name,
                db_collection=db_collection,
            )
            try:
                docs = future.result(timeout=_RAG_TIMEOUT)
            finally:
                rag_executor.shutdown(wait=False, cancel_futures=True)
        except (FutureTimeoutError, TimeoutError):
            print(f"[classify_with_rag] RAG检索超时({_RAG_TIMEOUT}s)，跳过知识库检索")
            docs = []
        finally:
            socket.setdefaulttimeout(old_timeout)
    except Exception as e:
        print(f"[classify_with_rag] RAG检索跳过: {e}")
        docs = []

    return docs


def format_rag_context(docs: list[dict]) -> str:
    """Format retrieved docs for classify_with_rag prompt context."""
    if not docs:
        return ""
    return "\n\n---\n\n".join([
        (
            f"[来源: {_doc_value(d, 'source') or _doc_metadata(d).get('source') or 'unknown'}]\n"
            f"{_doc_value(d, 'content') or _doc_value(d, 'page_content') or ''}"
        )
        for d in docs
    ])


def classify_with_rag(
    system_prompt: str,
    user_message: str,
    session_id: str,
    json_schema: dict,
    db_name: str = "Nb_KnowBase_db",
    db_collection: str = "documents",
    top_k: int = 5,
    retrieval_docs: Optional[list[dict]] = None,
) -> dict:
    """
    通用 RAG + 意图分类方法。

    自动执行 RAG 检索 → 上下文拼接 → 链式执行（带历史记忆）→ 返回结构化JSON。

    Args:
        system_prompt:  系统提示词，定义分类任务、意图类别和判断规则
        user_message:   用户输入的消息文本
        session_id:     会话唯一标识，用于多轮对话记忆管理
        json_schema:    期望输出的 JSON 样例，如 {"intent": "DESIGN"}
        db_name:        RAG 检索的目标数据库名，默认 "Nb_KnowBase_db"
        db_collection:  PGVector 集合名，默认 "documents"
        top_k:          RAG 检索返回的文档数量，默认 5

    Returns:
        dict: JSON 键值对结果，key 与 json_schema 一致

    Example:
        >>> classify_with_rag(
        ...     system_prompt="判断用户意图：钢材设计问题输出DESIGN，否则输出CHAT。",
        ...     user_message="Q460钢的热处理工艺如何优化？",
        ...     session_id="user-session-1",
        ...     json_schema={"intent": "DESIGN"},
        ...     db_name="Nb_KnowBase_db",
        ... )
        {'intent': 'DESIGN'}
    """
    # ==========================================================
    # Step 1: RAG 检索 — 暂时关闭，只交给 LLM 做语义意图识别。
    # 如需恢复，打开下方被注释的检索逻辑并把 rag_context 加回提示词。
    # ==========================================================
    rag_context = ""
    docs = []
    # try:
    #     import os as _os
    #
    #     # 强制设置 HuggingFace 下载超时（秒）
    #     _os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(_RAG_TIMEOUT))
    #     _os.environ.setdefault("REQUESTS_TIMEOUT", str(_RAG_TIMEOUT))
    #
    #     # 设置全局 socket 超时
    #     old_timeout = socket.getdefaulttimeout()
    #     socket.setdefaulttimeout(_RAG_TIMEOUT)
    #
    #     try:
    #         from hybrid_retriever import hybrid_search
    #
    #         with ThreadPoolExecutor(max_workers=1) as rag_executor:
    #             future = rag_executor.submit(
    #                 hybrid_search,
    #                 query=user_message,
    #                 k=top_k,
    #                 db_name=db_name,
    #                 db_collection=db_collection,
    #             )
    #             docs = future.result(timeout=_RAG_TIMEOUT)
    #     except (FutureTimeoutError, TimeoutError):
    #         print(f"[classify_with_rag] RAG检索超时({_RAG_TIMEOUT}s)，跳过知识库检索")
    #         docs = []
    #     finally:
    #         socket.setdefaulttimeout(old_timeout)
    #
    #     if docs:
    #         rag_context = "\n\n---\n\n".join([
    #             f"[来源: {d.get('source', 'unknown')}]\n{d.get('content', '')}"
    #             for d in docs
    #         ])
    #         print(f"[classify_with_rag] RAG检索命中 {len(docs)} 条文档")
    # except Exception as e:
    #     print(f"[classify_with_rag] RAG检索跳过: {e}")
    #     rag_context = "(知识库不可用)"

    # ==========================================================
    # Step 2: 构建完整的系统提示词（含RAG上下文+输出格式）
    # ==========================================================
    json_format_str = json.dumps(json_schema, ensure_ascii=False)
    keys_desc = "、".join([f'"{k}"' for k in json_schema.keys()])
    value_example = "、".join([f'"{v}"' for v in json_schema.values()])

    full_system_prompt = f"""{system_prompt}

## 输出格式要求（极其重要！）
你的回复必须且只能是一个JSON对象，不要输出任何其他内容。
- JSON键: {keys_desc}
- 正确输出示例: {json_format_str}
- 错误输出示例: "根据分析，我认为这是{value_example}" ← 这是错误的！不要输出任何解释！
- 可选值: {value_example}"""

    # ==========================================================
    # Step 3: 构建链式执行管道
    #   chain = prompt_template | llm | output_parser
    #   chain_with_history = RunnableWithMessageHistory(chain, ...)
    # ==========================================================
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt_template | deepseek_Llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # ==========================================================
    # Step 4: 执行链式调用
    # ==========================================================
    # 注意：裁剪逻辑现在由 SessionStore 内部处理（通过 PersistentChatMessageHistory.messages 属性）
    session_hist = get_session_history(session_id)

    try:
        raw_result = chain_with_history.invoke(
            {
                "system_prompt": full_system_prompt,
                "input": user_message,
            },
            config={"configurable": {"session_id": session_id}},
        )

        # Step 5: 解析 LLM 返回为 JSON
        result = _parse_json_response(raw_result, json_schema)

        # 追加用户消息和 AI 回复到历史（持久化存储）
        session_hist.add_message(HumanMessage(content=user_message))
        session_hist.add_message(AIMessage(content=json.dumps(result, ensure_ascii=False)))

        return result

    except Exception as e:
        print(f"[classify_with_rag] 链式执行失败: {e}")
        # Fallback: 返回默认结果
        default = {}
        for key, value in json_schema.items():
            default[key] = value
        return default


def classify_with_preloaded_rag(
    system_prompt: str,
    user_message: str,
    session_id: str,
    json_schema: dict,
    retrieval_docs: list[dict],
) -> dict:
    """Classify with docs that were already retrieved and streamed to the UI."""
    json_format_str = json.dumps(json_schema, ensure_ascii=False)
    keys_desc = "、".join([f'"{k}"' for k in json_schema.keys()])
    value_example = "、".join([f'"{v}"' for v in json_schema.values()])
    rag_context = format_rag_context(retrieval_docs)

    full_system_prompt = f"""{system_prompt}

## 参考知识库信息
{rag_context}

## 输出格式要求（极其重要！）
你的回复必须且只能是一个JSON对象，不要输出任何其他内容。
- JSON键: {keys_desc}
- 正确输出示例: {json_format_str}
- 错误输出示例: "根据分析，我认为这是{value_example}" -> 这是错误的！不要输出任何解释。
- 可选值: {value_example}"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt_template | deepseek_Llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    session_hist = get_session_history(session_id)

    try:
        raw_result = chain_with_history.invoke(
            {
                "system_prompt": full_system_prompt,
                "input": user_message,
            },
            config={"configurable": {"session_id": session_id}},
        )
        result = _parse_json_response(raw_result, json_schema)
        session_hist.add_message(HumanMessage(content=user_message))
        session_hist.add_message(AIMessage(content=json.dumps(result, ensure_ascii=False)))
        return result
    except Exception as e:
        print(f"[classify_with_preloaded_rag] 链式执行失败: {e}")
        return {key: value for key, value in json_schema.items()}


def clear_session(session_id: str) -> None:
    """清除指定会话的历史记录（内存缓存 + 数据库）"""
    _intent_store.clear(session_id)


def clear_all_sessions() -> None:
    """清除所有 intent_classifier 类型会话的历史记录（内存缓存）"""
    _intent_store.clear_all()



