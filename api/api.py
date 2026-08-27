"""
api.py - FastAPI 流式聊天后端服务（带服务端会话管理）
=====================================================
功能 / Features:
  1. POST /chat - 接收用户消息 + session_id，流式返回 LLM 回复chat
  2. GET /      - 重定向到前端页面
  3. CORS 支持  - 允许前端跨域请求
  4. 会话管理   - 服务端存储对话历史（内存），支持 50 轮记忆

会话机制 / Session Mechanism:
  - 前端打开页面时生成随机 session_id
  - 每次请求携带 session_id，服务端查找/创建对应会话
  - 服务端自动累积对话历史，限制最近 50 轮（100 条消息）
  - 关闭网页 → session_id 丢失 → 下次打开为新会话
  - 闲置超过 1 小时的会话自动清理

启动方式 / How to run:
  python api.py
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

# ============================================================
# 【必须在所有 import 之前】强制 HuggingFace 离线 + 网络超时
# 确保后续任何工具的 import 链中触发的 HuggingFace 操作都不会联网
# ============================================================
import os as _os
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5"

# ============================================================
# 全局 print 时间戳前缀：所有 Python print 输出统一加上 [yyyy-MM-dd HH:mm:ss]
# 注意：C# DLL 直接写到 stdout 的内容不是 Python print，不会被加前缀。
# ============================================================
import builtins as _builtins
from datetime import datetime as _datetime, timedelta as _timedelta

_orig_print = _builtins.print


def _ts_print(*args, **kwargs):
    """给每行 print 自动加上 [yyyy-MM-dd HH:mm:ss] 时间戳前缀。"""
    ts = _datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _orig_print(f"[{ts}]", *args, **kwargs)


_builtins.print = _ts_print

from network_timeout import enforce_timeout
enforce_timeout(timeout=5)

import json
import asyncio
import re
import sys
import threading
import hashlib
import time
import base64
import copy
import shutil
import subprocess
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse, Response, FileResponse

# 聊天附件使用独立临时目录和独立解析进程，不进入知识库向量化目录。
from attachment_service import (
    AttachmentServiceError,
    MAX_FILE_BYTES,
    attachment_manager,
)

# 设计版本控制层独立保存最终 matched_result，不向固定业务JSON中增加字段。
from design_versioning import (
    DesignRevisionValidationError,
    DesignTaskNormalizationError,
    build_normalized_design_task,
    build_resolved_design_request,
    design_snapshot_store,
    extract_target_grade as extract_version_target_grade,
    resolve_design_reference,
    validate_revision_constraints,
)

# LangChain 消息类型 — 用于构建对话历史
# LangChain message types — for building conversation history
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 导入已配置流式输出的 LLM 模型实例
from my_llm import deepseek_Llm, qwen_Llm
from official_llm_client import (
    official_deepseek_sync,
    official_deepseek_async,
    official_qwen_sync,
)
from pipeline_agents import (
    CompositionRefinementValidationError,
    CompositionRefinementDependencies,
    DesignChangeAssessmentDependencies,
    DesignChangeAssessmentError,
    ProcessAgentDependencies,
    RequirementParsingDependencies,
    RequirementParsingError,
    WindPowerDesignValidationError,
    assess_design_change,
    build_unified_design_user_message,
    parse_design_requirement,
    refine_composition_process_performance,
    refine_reheat_process,
    refine_rolling_process,
    refine_cooling_process,
)
from prompt import (
    INTENT_SYSTEM_PROMPT,
    PURPOSE_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    KNOWLEDGE_BASE_TOOL_ROUTING_PROMPT,
    KNOWLEDGE_BASE_TOOL_SELECTION_SYSTEM_PROMPT,
    QWEN_REPORT_REVIEW_SYSTEM_PROMPT,
    PIPELINE_REHEAT_AGENT_SYSTEM_PROMPT,
    PIPELINE_ROLL_AGENT_SYSTEM_PROMPT,
    PIPELINE_COOLING_AGENT_SYSTEM_PROMPT,
    PIPELINE_REPORT_RISK_SYSTEM_PROMPT,
    PIPELINE_DESIGN_PREVIEW_SYSTEM_PROMPT,
    WIND_POWER_DESIGN_PREVIEW_SYSTEM_PROMPT,
    WEAR_STEEL_DESIGN_PREVIEW_SYSTEM_PROMPT,
    LIMITED_CHAT_SYSTEM_PROMPT,
    PIPELINE_REPORT_SYSTEM_PROMPT,
    build_wind_power_report_system_prompt,
    get_wind_power_material_label,
    WIND_POWER_REFINEMENT_PROCESS_RULE,
    WIND_POWER_PROMPT_CONTEXT_TAG,
    REPORT_TEMPLATE_CONTEXT_PREFIX,
    REPORT_TEMPLATE_CONTEXT_SUFFIX,
    PIPELINE_REFINEMENT_USER_PROMPT,
    _build_qwen_report_review_user_prompt,
    _build_pipeline_agent_repair_prompt,
    build_pipeline_reheat_agent_user_prompt_text,
    build_pipeline_roll_agent_user_prompt_text,
    build_pipeline_cooling_agent_user_prompt_text,
    build_unstrict_refinement_prompt,
    build_oracle_expand_spec_prompt,
    build_pipeline_refinement_prompt,
    build_pipeline_expand_spec_prompt,
    build_flash_design_preview_user_prompt,
    build_pipeline_report_risk_user_prompt,
    build_pipeline_report_user_prompt,
    build_cross_route_context_system_prompt,
    build_report_history_user_prompt,
    build_pipeline_refinement_repair_prompt,
    build_wind_power_standard_redesign_instruction_text,
    PIPELINE_REFINEMENT_COMPOSITION_REPAIR_SCOPE_PROMPT,
    PIPELINE_REFINEMENT_ROLLING_REPAIR_SCOPE_PROMPT,
    PIPELINE_REFINEMENT_ALL_REPAIR_SCOPE_PROMPT,
)
# 导入通用 RAG + 意图分类方法
from intent_classifier import (
    classify_with_rag,
)

# 导入会话持久化工具
from session_store import (
    SessionStore,
    PersistentChatMessageHistory,
    init_session_db,
    register_for_cleanup,
    _periodic_db_cleanup,
)

# ============================================================
# 意图分类配置
# ============================================================



INTENT_JSON_SCHEMA = {"intent": "CHAT"}

# 钢材用途分类系统提示词

PURPOSE_JSON_SCHEMA = {"purpose": "其他聊天"}

# 聊天助手系统提示词


# 聊天会话历史存储 — 持久化（数据库 + 内存缓存，服务器重启后数据不丢失）
agent_chat_store = SessionStore(session_type="agent_chat", max_turns=50, ttl=3600)

# 报告生成上下文存储 — 保存报告 LLM 的输入和正文输出，不保存末尾 base64 图片区。
report_session_store = SessionStore(session_type="report", max_turns=10, ttl=3600)


class _FrontendComputationTask:
    """保存一次会话计算的后台任务状态及可重放 NDJSON 事件。"""

    def __init__(self, session_id: str, user_message: str):
        self.session_id = session_id
        self.user_message = user_message
        self.status = "running"
        self.events: list[str] = []
        self.condition = asyncio.Condition()
        self.worker: asyncio.Task | None = None
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.completed_at: float | None = None
        self.error = ""
        self.answer_done_emitted = False


# 浏览器连接只是后台计算任务的订阅者。关闭、刷新或切换页面只断开订阅，
# 不会取消这里保存的计算任务；同一 session_id 重新进入页面后可重放事件。
_FRONTEND_COMPUTATION_TASKS: dict[str, _FrontendComputationTask] = {}
_FRONTEND_COMPUTATION_TASK_RETENTION_SECONDS = 2 * 3600

# 多候选设计确认期间临时保留本轮已拼接附件的有效提示词。前端只拿随机令牌，
# 不会接触附件全文；选择方案后令牌一次性消费并继续原始用户请求。
_PENDING_DESIGN_REFERENCE_REQUESTS: dict[str, dict] = {}
_PENDING_DESIGN_REFERENCE_TTL_SECONDS = 30 * 60


def _store_pending_design_reference_request(
    session_id: str,
    original_user_message: str,
    effective_user_message: str,
) -> str:
    token = str(uuid.uuid4())
    _PENDING_DESIGN_REFERENCE_REQUESTS[token] = {
        "session_id": session_id,
        "original_user_message": original_user_message,
        "effective_user_message": effective_user_message,
        "created_at": time.time(),
    }
    return token


def _consume_pending_design_reference_request(
    token: str,
    session_id: str,
    original_user_message: str,
) -> str | None:
    record = _PENDING_DESIGN_REFERENCE_REQUESTS.pop(str(token or ""), None)
    if not record:
        return None
    if time.time() - float(record.get("created_at") or 0) > _PENDING_DESIGN_REFERENCE_TTL_SECONDS:
        return None
    if record.get("session_id") != session_id:
        return None
    if record.get("original_user_message") != original_user_message:
        return None
    return str(record.get("effective_user_message") or "")


def _ndjson_contains_event(chunk: str, event_name: str) -> bool:
    """判断一个可能含多行的 NDJSON 块中是否已经包含指定事件。"""
    for line in str(chunk or "").splitlines():
        try:
            if json.loads(line).get("event") == event_name:
                return True
        except (json.JSONDecodeError, AttributeError):
            continue
    return False


async def _append_frontend_task_event(
    state: _FrontendComputationTask,
    event_text: str,
) -> None:
    text = str(event_text or "")
    if not text:
        return
    if not text.endswith("\n"):
        text += "\n"
    async with state.condition:
        state.events.append(text)
        state.updated_at = time.time()
        if _ndjson_contains_event(text, "answer_done"):
            state.answer_done_emitted = True
        state.condition.notify_all()


async def _run_frontend_computation_task(
    state: _FrontendComputationTask,
    stream_factory,
) -> None:
    """独立消费原业务流并缓存结果，不受任何单个浏览器连接取消影响。"""
    try:
        async for chunk in stream_factory():
            text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            await _append_frontend_task_event(state, text)
        state.status = "completed"
    except asyncio.CancelledError:
        state.status = "cancelled"
        state.error = "后端服务关闭，计算任务已取消"
        raise
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        print(f"[后台计算任务] session={state.session_id[:8]}... 执行失败: {exc}")
        await _append_frontend_task_event(
            state,
            _ndjson_event("error", message=f"生成过程异常: {exc}"),
        )
    finally:
        if not state.answer_done_emitted:
            await _append_frontend_task_event(state, _ndjson_event("answer_done"))
        state.completed_at = time.time()
        state.updated_at = state.completed_at
        async with state.condition:
            state.condition.notify_all()


def _cleanup_frontend_computation_tasks() -> None:
    """清理已结束且超过保留时间的任务；运行中的任务永不在此处删除。"""
    now = time.time()
    expired = [
        session_id
        for session_id, state in _FRONTEND_COMPUTATION_TASKS.items()
        if state.status != "running"
        and state.completed_at is not None
        and now - state.completed_at > _FRONTEND_COMPUTATION_TASK_RETENTION_SECONDS
    ]
    for session_id in expired:
        _FRONTEND_COMPUTATION_TASKS.pop(session_id, None)
    expired_reference_tokens = [
        token
        for token, record in _PENDING_DESIGN_REFERENCE_REQUESTS.items()
        if now - float(record.get("created_at") or 0) > _PENDING_DESIGN_REFERENCE_TTL_SECONDS
    ]
    for token in expired_reference_tokens:
        _PENDING_DESIGN_REFERENCE_REQUESTS.pop(token, None)


def _start_or_get_frontend_computation_task(
    session_id: str,
    user_message: str,
    stream_factory,
) -> _FrontendComputationTask:
    """同一会话已有运行任务时直接复用，否则创建一项新的后台计算。"""
    _cleanup_frontend_computation_tasks()
    existing = _FRONTEND_COMPUTATION_TASKS.get(session_id)
    if existing is not None and existing.status == "running":
        return existing

    state = _FrontendComputationTask(session_id, user_message)
    _FRONTEND_COMPUTATION_TASKS[session_id] = state
    state.worker = asyncio.create_task(
        _run_frontend_computation_task(state, stream_factory)
    )
    return state


async def _subscribe_frontend_computation_task(
    state: _FrontendComputationTask,
    start_index: int = 0,
):
    """从指定事件序号开始重放，并持续订阅尚未完成的后台计算。"""
    index = max(0, int(start_index or 0))
    while True:
        async with state.condition:
            while index >= len(state.events) and state.status == "running":
                await state.condition.wait()
            available = state.events[index:]
            index = len(state.events)
            finished = state.status != "running"

        for event_text in available:
            yield event_text
        if finished and index >= len(state.events):
            break

# ============================================================
# FastAPI 生命周期管理：初始化/清理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务器生命周期管理：
    - 启动时：初始化会话持久化数据库 + 启动后台清理任务
    - 关闭时：（预留）清理资源
    """
    # 启动
    init_session_db()
    design_snapshot_store.initialize()
    register_for_cleanup(chat_session_store)
    register_for_cleanup(agent_chat_store)
    register_for_cleanup(report_session_store)
    cleanup_task = asyncio.create_task(_periodic_db_cleanup(interval_seconds=1800))

    async def periodic_attachment_cleanup():
        """定期清除用户未发送、页面关闭后遗留的临时附件。"""
        while True:
            await asyncio.sleep(1800)
            await attachment_manager.cleanup_stale()

    await attachment_manager.cleanup_stale()
    attachment_cleanup_task = asyncio.create_task(periodic_attachment_cleanup())
    print("[启动] 会话持久化数据库已就绪，后台清理任务已启动")
    yield
    # 关闭
    running_frontend_tasks = [
        state.worker
        for state in _FRONTEND_COMPUTATION_TASKS.values()
        if state.worker is not None and not state.worker.done()
    ]
    for worker in running_frontend_tasks:
        worker.cancel()
    if running_frontend_tasks:
        await asyncio.gather(*running_frontend_tasks, return_exceptions=True)
    await attachment_manager.shutdown()
    attachment_cleanup_task.cancel()
    try:
        await attachment_cleanup_task
    except asyncio.CancelledError:
        pass
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    print("[关闭] 后台清理任务已停止")


# ============================================================
# FastAPI 应用初始化
# ============================================================
app = FastAPI(
    title="Steel Multi-Agent System (SMAS) API",
    description="Streaming chat API with server-side session memory (50 turns)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 临时聊天附件接口
# ============================================================
@app.post("/attachments/upload")
async def upload_chat_attachment(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """分块接收单个附件；超过10 MiB时立即停止写入并清理临时目录。"""
    try:
        record = attachment_manager.reserve(session_id, file.filename or "")
    except AttachmentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    destination = record.task_dir / "source" / record.stored_name
    size = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise AttachmentServiceError("单个附件不能超过 10 MiB")
                output.write(chunk)
        if size <= 0:
            raise AttachmentServiceError("不能上传空文件")
        attachment_manager.finish_upload(record, size)
        return {
            "attachment_id": record.attachment_id,
            "name": record.original_name,
            "size": record.size,
            "status": record.status,
        }
    except AttachmentServiceError as exc:
        await attachment_manager.cancel(record.attachment_id, record.session_id)
        status_code = 413 if size > MAX_FILE_BYTES else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        await attachment_manager.cancel(record.attachment_id, record.session_id)
        raise HTTPException(status_code=500, detail=f"附件上传失败：{exc}") from exc
    finally:
        await file.close()


@app.post("/attachments/{attachment_id}/parse")
async def parse_chat_attachment(attachment_id: str, request: Request):
    """把附件加入当前会话的串行解析队列。"""
    try:
        try:
            body = await request.json()
        except Exception as exc:
            raise AttachmentServiceError("请求体必须是有效 JSON") from exc
        session_id = str(body.get("session_id", "")).strip()
        if not session_id:
            raise AttachmentServiceError("session_id 不能为空")
        record = await attachment_manager.enqueue(attachment_id, session_id)
        return attachment_manager.status_payload(record.attachment_id, session_id)
    except AttachmentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/attachments/{attachment_id}/status")
async def get_chat_attachment_status(attachment_id: str, session_id: str):
    """返回上传/排队/解析/完成状态及阶段进度。"""
    try:
        return attachment_manager.status_payload(attachment_id, session_id)
    except AttachmentServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/attachments/{attachment_id}")
async def delete_chat_attachment(attachment_id: str, session_id: str):
    """随时取消解析；重复删除同一附件也按成功处理。"""
    try:
        await attachment_manager.cancel(attachment_id, session_id)
        return {"ok": True}
    except AttachmentServiceError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


# ============================================================
# 服务端会话存储 / Server-Side Session Store
# ============================================================
# 使用持久化存储（PostgreSQL + 内存缓存），服务器重启后数据不丢失。
# SessionStore 内部管理会话的 CRUD、过期清理和裁剪。
# ============================================================

MAX_TURNS = 50         # 最多保留 50 轮对话 / Max 50 conversation turns
SESSION_TTL = 3600     # 会话过期时间（秒），1 小时无活动即清理 / 1 hour TTL

# 聊天会话存储 — 持久化（数据库 + 内存缓存，服务器重启后数据不丢失）
chat_session_store = SessionStore(session_type="chat", max_turns=MAX_TURNS, ttl=SESSION_TTL)


def cleanup_expired_sessions():
    """
    清理过期的会话 — 持久化版本
    Clean up expired sessions — persistent version

    委托给 chat_session_store，同时清理内存缓存和数据库。
    """
    chat_session_store.cleanup_expired()


def get_or_create_session(session_id: str) -> dict:
    """
    获取或创建会话 / Get or create a session

    使用持久化存储（数据库 + 内存缓存），重启后数据不丢失。

    Args:
        session_id: 前端生成的唯一会话 ID

    Returns:
        session 字典，包含 "messages" 列表和 "last_active" 时间戳
    """
    # SessionStore.get_or_create() 内部已包含过期清理
    return chat_session_store.get_or_create(session_id)

#裁剪对话历史到指定的轮数
def trim_history(messages: list, max_turns: int = MAX_TURNS) -> list:
    """
    裁剪对话历史到指定的轮数 / Trim conversation history to max turns

    每轮 = 1 条用户消息 + 1 条 AI 消息 = 2 条 LangChain 消息
    保留最近的 max_turns 轮，即最多 max_turns * 2 条消息

    Args:
        messages: LangChain 消息列表 [HumanMessage, AIMessage, ...]
        max_turns: 最大保留轮数

    Returns:
        裁剪后的消息列表（不修改原列表，返回新切片）
    """
    max_messages = max_turns * 2  # 每轮 2 条消息
    if len(messages) > max_messages:
        return messages[-max_messages:]
    return messages


# ============================================================
# 耐磨钢规格匹配 Oracle 实绩数据
# ============================================================

# Oracle 连接参数从 api/.env 读取，下面仅保留非敏感默认值。
ORACLE_HOST = _os.environ.get("ORACLE_HOST", "localhost")
ORACLE_PORT = int(_os.environ.get("ORACLE_PORT", "1521"))
ORACLE_SERVICE_NAME = _os.environ.get("ORACLE_SERVICE_NAME", "ORCL")
ORACLE_USER = _os.environ.get("ORACLE_USER", "")
ORACLE_PASSWORD = _os.environ.get("ORACLE_PASSWORD", "")
ORACLE_TABLE = _os.environ.get("ORACLE_TABLE", "match_process_valid")
ORACLE_CLIENT_LIB_DIR = _os.environ.get("ORACLE_CLIENT_LIB_DIR") or (
    r"D:\software\app\CX\product\11.2.0\dbhome_1\BIN"
)
ORACLE_TNS_ADMIN = _os.environ.get("ORACLE_TNS_ADMIN") or str(
    _os.path.join(_os.path.dirname(__file__), "oracle_network")
)
IMAGE_GENERATOR_BIN_DIR = _os.path.join(
    _os.path.dirname(__file__),
    "DLL",
    "涟钢热处理形性面演变预测系统",
    "bin",
    "Debug",
)
IMAGE_GENERATOR_DLL_PATH = _os.path.join(IMAGE_GENERATOR_BIN_DIR, "ImageGeneratorLib.dll")

PIPELINE_IMAGE_GENERATOR_BIN_DIR = _os.path.join(
    _os.path.dirname(__file__),
    "FoundationModel_Deno_New",
    "HotColdDataBase",
    "bin",
    "Debug",
)
PIPELINE_IMAGE_GENERATOR_DLL_PATH = _os.path.join(
    PIPELINE_IMAGE_GENERATOR_BIN_DIR,
    "ANSTEEL_ImageGeneratorLib.dll",
)
PIPELINE_REHEAT_IMAGE_GENERATOR_DLL_PATH = _os.path.join(
    PIPELINE_IMAGE_GENERATOR_BIN_DIR,
    "ANSTEEL_ReheatImageGeneratorLib.dll",
)
PIPELINE_ROLL_IMAGE_GENERATOR_DLL_PATH = _os.path.join(
    PIPELINE_IMAGE_GENERATOR_BIN_DIR,
    "ANSTEEL_RollImageGeneratorLib.dll",
)
PIPELINE_COOLING_IMAGE_GENERATOR_DLL_PATH = _os.path.join(
    PIPELINE_IMAGE_GENERATOR_BIN_DIR,
    "ANSTEEL_CoolingImageGeneratorLib.dll",
)
PIPELINE_PRECIPITATE_IMAGE_PROCESSOR_DLL_PATH = _os.path.join(
    PIPELINE_IMAGE_GENERATOR_BIN_DIR,
    "ANSTEEL_PrecipitateImageProcessorLib.dll",
)

# DLL 内部依赖当前工作目录和相对路径，必须串行调用并在 finally 中恢复 cwd。
IMAGE_GENERATOR_CALL_LOCK = threading.Lock()

# os.add_dll_directory 返回的句柄需要长期保存，否则依赖 DLL 搜索路径可能失效。
IMAGE_GENERATOR_DLL_DIRECTORY_HANDLES = []

# 报告图片只在前端使用 URL 引用，避免把大 PNG 转为 base64 塞进回答文本导致浏览器 OOM。
GENERATED_IMAGE_REGISTRY: dict[str, str] = {}
GENERATED_IMAGE_REGISTRY_SCAN_CACHE: dict[str, str] = {}


def _register_generated_image(image_path: str) -> str | None:
    try:
        abs_path = _os.path.abspath(image_path)
        allowed_roots = [
            _os.path.abspath(IMAGE_GENERATOR_BIN_DIR),
            _os.path.abspath(PIPELINE_IMAGE_GENERATOR_BIN_DIR),
        ]
        if not abs_path.lower().endswith(".png"):
            return None
        if not _os.path.isfile(abs_path):
            return None
        if not any(_os.path.commonpath([root, abs_path]) == root for root in allowed_roots):
            return None
        token = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()
        GENERATED_IMAGE_REGISTRY[token] = abs_path
        return token
    except Exception as exc:
        print(f"[报告生成] 注册图片失败: {exc}")
        return None


def _resolve_generated_image_path(token: str) -> str | None:
    """根据图片 token 找到本地 PNG；服务重启后注册表为空时，按同一 hash 规则兜底扫描。"""
    image_path = GENERATED_IMAGE_REGISTRY.get(token) or GENERATED_IMAGE_REGISTRY_SCAN_CACHE.get(token)
    if image_path and _os.path.isfile(image_path):
        return image_path

    allowed_roots = [
        _os.path.abspath(IMAGE_GENERATOR_BIN_DIR),
        _os.path.abspath(PIPELINE_IMAGE_GENERATOR_BIN_DIR),
    ]
    for root in allowed_roots:
        if not _os.path.isdir(root):
            continue
        for current_dir, dir_names, file_names in _os.walk(root):
            if _os.path.basename(current_dir) not in {"Image", "Images"}:
                # 绝大部分报告图都在 Image 目录，跳过其它深层目录，避免接口扫描过慢。
                continue
            for file_name in file_names:
                if not file_name.lower().endswith(".png"):
                    continue
                candidate_path = _os.path.abspath(_os.path.join(current_dir, file_name))
                candidate_token = hashlib.sha256(candidate_path.encode("utf-8")).hexdigest()
                GENERATED_IMAGE_REGISTRY_SCAN_CACHE[candidate_token] = candidate_path
                if candidate_token == token:
                    return candidate_path
            dir_names[:] = []
    return None


def _pipeline_image_process_support_analysis(image_name: str) -> str:
    """按管线钢仿真图片名称生成工艺支持分析，追加在每张图片后面。"""
    analysis_map = {
        "加热Ⅱ温度.png": "该图用于验证第二加热段温度场是否稳定，为奥氏体均匀化、微合金元素固溶和后续均热过程提供升温路径依据。",
        "均热温度.png": "该图用于验证均热段温度场均匀性，为全固溶温度达成、元素固溶和异常晶粒长大控制提供工艺依据。",
        "粗轧入口温度.png": "该图用于判断粗轧入口温度是否满足再结晶区变形需求，支撑粗轧阶段晶粒破碎和初始奥氏体组织均匀化。",
        "精轧入口温度.png": "该图用于判断精轧入口温度是否进入适宜控轧窗口，支撑未再结晶区变形累积和后续晶粒细化。",
        "终轧温度.png": "该图用于验证终轧温度与目标 TMCP 路径是否匹配，支撑形变奥氏体向细小铁素体/贝氏体组织转变。",
        "冷却温度.png": "该图用于分析轧后冷却路径是否稳定，支撑入水、冷却终止和返红温度对相变组织的控制。",
        "温度场曲线.png": "该图用于综合验证加热、轧制、冷却全流程温度连续性，支撑成分-工艺-组织协同设计的热历程合理性。",
        "晶粒长大.png": "该图用于评估加热和高温停留过程中晶粒长大风险，支撑均热温度和均热时长是否需要保守控制。",
        "晶粒尺寸分布.png": "该图用于判断最终晶粒尺寸分布是否均匀，支撑方案是否具备细晶强化和韧性稳定性基础。",
        "轧制力.png": "该图用于核对各道次轧制负荷与压下量、板宽、变形温度和平均变形抗力的匹配关系，支撑粗精轧负荷分配及设备能力校核。",
        "扭矩.png": "该图用于核对各道次传动扭矩与轧制力、接触弧长和轧制速度的协同关系，支撑主传动负荷及轧制规程可执行性判断。",
        "摩擦系数.png": "该图用于分析各道次轧辊与轧件界面的摩擦条件变化，支撑咬入稳定性、轧制力和扭矩计算依据的合理性判断。",
        "各道次晶粒尺寸.png": "该图用于验证各轧制道次晶粒演化是否连续合理，支撑道次压下、轧制温度、速度和轧制力分配的协同有效性。",
        "粗轧出口奥氏体晶粒尺寸.png": "该图用于观察粗轧出口奥氏体晶粒形貌和均匀性，支撑粗轧再结晶、晶粒破碎及后续精轧初始组织条件的合理性判断。",
        "精轧出口奥氏体晶粒尺寸.png": "该图用于观察精轧出口奥氏体晶粒细化与形变状态，支撑未再结晶区累积变形和最终组织细化效果的合理性判断。",
        "氧化铁皮厚度.png": "该图用于综合评估加热、轧制和冷却全过程的氧化铁皮控制水平，支撑温度制度、除鳞效果及最终表面质量判断。",
        "软化率.png": "该图用于分析轧制过程中再结晶与软化行为，支撑控轧温度窗口和道次间隔是否有利于变形累积。",
        "析出动力学.png": "该图用于评估微合金碳氮化物析出时机和强化贡献，支撑 Nb、V、Ti 等元素与控轧控冷工艺的匹配性。",
        "RPTT.png": "该图用于分析轧制过程中再结晶与析出的温度-时间耦合关系，支撑控轧温度窗口、道次间隔和微合金析出时序的合理性判断。",
        "CCT.png": "该图用于判断连续冷却过程中的相变路径，支撑冷却制度对铁素体、针状铁素体或贝氏体组织比例的控制。",
        "相组成.png": "该图用于验证冷却后组织相比例是否合理，支撑返红温度和冷却路径对最终强韧性匹配的贡献。",
        "析出形貌.png": "该图用于观察冷却后微合金析出相的空间分布和形貌特征，支撑析出强化效果及成分—控轧控冷工艺匹配关系的判断。",
        "强化机制.png": "该图用于核对固溶强化、析出强化、细晶强化、位错强化及其他强化贡献与最终屈服强度的一致性，支撑成分、组织和性能协同关系判断。",
    }
    return analysis_map.get(
        image_name,
        "该图用于从仿真结果角度补充验证当前管线钢工艺方案，支撑成分、轧制、冷却与最终组织性能之间的协同判断。",
    )


def _save_pipeline_stage_matched_result(stage_name: str, matched_result: dict) -> None:
    """将工艺智能体的阶段最终结果覆盖保存到 api.py 所在目录。

    每个阶段使用独立文件，后一次设计会原子替换上一次结果。使用 UTF-8 BOM，
    便于直接通过 Windows 记事本查看中文，同时避免服务中断时留下半截 JSON。
    保存失败只记录后端日志，不能影响后续智能体或最终报告生成。
    """
    stage_files = {
        "reheat": "加热工艺最终结果.txt",
        "roll": "轧制工艺最终结果.txt",
        "cooling": "冷却工艺最终结果.txt",
    }
    file_name = stage_files.get(stage_name)
    if not file_name:
        print(f"[工艺智能体结果保存] 未知阶段，跳过保存: {stage_name!r}")
        return
    if not isinstance(matched_result, dict):
        print(f"[工艺智能体结果保存] {stage_name} 的 matched_result 不是对象，跳过保存")
        return

    target_path = _os.path.join(_os.path.dirname(__file__), file_name)
    temporary_path = (
        target_path
        + f".{_os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(temporary_path, "w", encoding="utf-8-sig", newline="\n") as result_file:
            json.dump(matched_result, result_file, ensure_ascii=False, indent=2)
            result_file.write("\n")
        _os.replace(temporary_path, target_path)
        print(f"[工艺智能体结果保存] 已覆盖保存: {target_path}")
    except Exception as exc:
        try:
            if _os.path.isfile(temporary_path):
                _os.remove(temporary_path)
        except OSError:
            pass
        print(
            f"[工艺智能体结果保存] 保存失败: stage={stage_name}, "
            f"{type(exc).__name__}: {exc}"
        )


def _pipeline_exit_grain_steel_dir_name(
    matched_result: dict,
    target_context: str = "",
) -> str | None:
    """按当前目标牌号映射 bigmodel_Picture_aus.py 所在的参考模型目录。"""
    _target_grade, reference_grade = _resolve_pipeline_dll_grade(
        matched_result,
        target_context,
    )
    if reference_grade == "X80NG":
        return "Physical_Metallurgy_X80NG"
    if reference_grade == "X70":
        return "Physical_Metallurgy_X70"
    if reference_grade == "X65":
        return "Physical_Metallurgy_X65"
    return None


def _read_pipeline_exit_grain_pass_numbers(pass_number_path: str) -> tuple[int, int]:
    """读取首条有效记录的前两列，依次返回精轧出口、粗轧出口道次号。"""
    with open(pass_number_path, "r", encoding="utf-8-sig") as pass_file:
        for raw_line in pass_file:
            columns = raw_line.strip().split()
            if len(columns) < 2:
                continue
            try:
                finish_pass = int(columns[0])
                rough_pass = int(columns[1])
            except ValueError:
                continue
            if finish_pass > 0 and rough_pass > 0:
                return finish_pass, rough_pass
    raise ValueError("Pass number.txt 中没有找到至少包含两个正整数的有效记录")


def _terminate_pipeline_drawing_processes(processes: list[subprocess.Popen]) -> None:
    """终止绘图 Python 进程及其 microstructpy 子进程，不等待绘图自然结束。"""
    running = [process for process in processes if process and process.poll() is None]
    if not running:
        return

    if _os.name == "nt":
        command = ["taskkill"]
        for process in running:
            command.extend(["/PID", str(process.pid)])
        command.extend(["/T", "/F"])
        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except Exception as exc:
            print(f"[轧制出口晶粒绘图] 终止子进程树失败: {exc}")

    for process in running:
        if process.poll() is not None:
            continue
        try:
            process.kill()
        except Exception:
            pass


class _PipelineExitGrainDrawingJob:
    """管理粗轧/精轧出口奥氏体晶粒图的两个后台绘图任务。"""

    TARGET_NAMES = (
        "粗轧出口奥氏体晶粒尺寸.png",
        "精轧出口奥氏体晶粒尺寸.png",
    )

    def __init__(
        self,
        coil_id: str,
        steel_dir: str,
        finish_pass: int,
        rough_pass: int,
    ):
        self.coil_id = coil_id
        self.steel_dir = steel_dir
        self.started_at = time.time()
        self.cancel_event = threading.Event()
        self.process_lock = threading.Lock()
        self.processes: list[subprocess.Popen] = []
        self.executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"pipeline-exit-grain-{coil_id}",
        )
        self.image_dir = _os.path.join(
            PIPELINE_IMAGE_GENERATOR_BIN_DIR,
            "ModelManage",
            coil_id,
            "Image",
        )
        self.source_dir = _os.path.join(
            PIPELINE_IMAGE_GENERATOR_BIN_DIR,
            "ZHB",
            steel_dir,
            "Temp_Physical_Metallurgy",
            coil_id,
            "Microstruct",
            "BefDef",
        )
        self.script_path = _os.path.join(
            PIPELINE_IMAGE_GENERATOR_BIN_DIR,
            "ZHB",
            steel_dir,
            "bigmodel_Picture_aus.py",
        )
        self.futures = [
            self.executor.submit(
                self._draw_one,
                rough_pass,
                "粗轧出口奥氏体晶粒尺寸.png",
            ),
            self.executor.submit(
                self._draw_one,
                finish_pass,
                "精轧出口奥氏体晶粒尺寸.png",
            ),
        ]

    def _draw_one(self, pass_number: int, target_name: str) -> bool:
        """调用单个道次绘图脚本；成功后把新图片原子复制到统一 Image 目录。"""
        if self.cancel_event.is_set():
            return False

        request_json = json.dumps(
            {"coilid": self.coil_id, "num": pass_number},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if _os.name == "nt" else 0
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, self.script_path, request_json],
                cwd=_os.path.dirname(self.script_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creation_flags,
            )
            with self.process_lock:
                self.processes.append(process)

            if self.cancel_event.is_set():
                _terminate_pipeline_drawing_processes([process])
                return False

            stdout_text, stderr_text = process.communicate()
            if process.returncode != 0:
                error_text = (stderr_text or stdout_text or "").strip()
                print(
                    f"[轧制出口晶粒绘图] {target_name} 绘制失败: "
                    f"returncode={process.returncode}, detail={error_text[-500:]}"
                )
                return False
            if self.cancel_event.is_set():
                return False

            source_path = _os.path.join(self.source_dir, f"F{pass_number}.png")
            if not _os.path.isfile(source_path):
                print(f"[轧制出口晶粒绘图] 未找到输出图片: {source_path}")
                return False
            if _os.path.getmtime(source_path) + 1 < self.started_at:
                print(f"[轧制出口晶粒绘图] 拒绝使用本轮启动前的旧图片: {source_path}")
                return False

            _os.makedirs(self.image_dir, exist_ok=True)
            target_path = _os.path.join(self.image_dir, target_name)
            temporary_path = target_path + f".{threading.get_ident()}.tmp"
            shutil.copyfile(source_path, temporary_path)
            if self.cancel_event.is_set():
                try:
                    _os.remove(temporary_path)
                except OSError:
                    pass
                return False
            _os.replace(temporary_path, target_path)
            print(
                f"[轧制出口晶粒绘图] 绘制完成: num={pass_number}, "
                f"target={target_path}"
            )
            return True
        except Exception as exc:
            print(
                f"[轧制出口晶粒绘图] {target_name} 后台任务异常: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def finish_without_waiting(self) -> list[str]:
        """保留已经完成的图片，取消未完成任务并立即释放后台执行器。"""
        self.cancel_event.set()
        with self.process_lock:
            processes = list(self.processes)
        _terminate_pipeline_drawing_processes(processes)

        completed_images = []
        for target_name in self.TARGET_NAMES:
            target_path = _os.path.join(self.image_dir, target_name)
            if _os.path.isfile(target_path) and _os.path.getmtime(target_path) + 1 >= self.started_at:
                completed_images.append(target_name)
        self.executor.shutdown(wait=False, cancel_futures=True)
        print(
            "[轧制出口晶粒绘图] 后台任务收尾完成: "
            f"可用于报告的图片={completed_images or '无'}"
        )
        return completed_images


def _start_pipeline_exit_grain_drawing(
    matched_result: dict,
    target_context: str = "",
) -> _PipelineExitGrainDrawingJob | None:
    """解析当前板坯的道次信息，并静默启动粗轧/精轧出口晶粒图并发任务。"""
    coil_id = str(matched_result.get("strCoil") or "").strip()
    steel_dir = _pipeline_exit_grain_steel_dir_name(matched_result, target_context)
    if not coil_id or not steel_dir:
        print(
            "[轧制出口晶粒绘图] 缺少受支持的板坯号或钢级，跳过后台绘图: "
            f"coil_id={coil_id!r}, strSteel={matched_result.get('strSteel')!r}"
        )
        return None

    pass_number_path = _os.path.join(
        PIPELINE_IMAGE_GENERATOR_BIN_DIR,
        "ModelManage",
        coil_id,
        "Physical Metallurgy Results",
        "Pass number.txt",
    )
    script_path = _os.path.join(
        PIPELINE_IMAGE_GENERATOR_BIN_DIR,
        "ZHB",
        steel_dir,
        "bigmodel_Picture_aus.py",
    )
    try:
        finish_pass, rough_pass = _read_pipeline_exit_grain_pass_numbers(pass_number_path)
        if not _os.path.isfile(script_path):
            raise FileNotFoundError(f"未找到绘图脚本: {script_path}")

        image_dir = _os.path.join(
            PIPELINE_IMAGE_GENERATOR_BIN_DIR,
            "ModelManage",
            coil_id,
            "Image",
        )
        _os.makedirs(image_dir, exist_ok=True)
        for target_name in _PipelineExitGrainDrawingJob.TARGET_NAMES:
            target_path = _os.path.join(image_dir, target_name)
            if _os.path.isfile(target_path):
                _os.remove(target_path)

        print(
            "[轧制出口晶粒绘图] 启动两个后台任务: "
            f"coil_id={coil_id}, 精轧道次={finish_pass}, 粗轧道次={rough_pass}"
        )
        return _PipelineExitGrainDrawingJob(
            coil_id=coil_id,
            steel_dir=steel_dir,
            finish_pass=finish_pass,
            rough_pass=rough_pass,
        )
    except Exception as exc:
        print(f"[轧制出口晶粒绘图] 无法启动后台任务: {type(exc).__name__}: {exc}")
        return None


def _copy_pipeline_image_atomically(source_path: str, target_path: str) -> None:
    """把单张图片原子复制到报告目录，避免报告读取到半写入文件。"""
    _os.makedirs(_os.path.dirname(target_path), exist_ok=True)
    temporary_path = target_path + f".{_os.getpid()}.{threading.get_ident()}.tmp"
    try:
        shutil.copyfile(source_path, temporary_path)
        _os.replace(temporary_path, target_path)
    finally:
        if _os.path.isfile(temporary_path):
            try:
                _os.remove(temporary_path)
            except OSError:
                pass


def _draw_pipeline_precipitate_morphology(
    matched_result: dict,
    target_context: str = "",
) -> bool:
    """
    同步绘制并后处理析出形貌图。

    任一步失败都只返回 False 并记录日志；调用方继续生成其它报告内容。开始前
    删除统一 Image 目录中的旧析出形貌图，确保本轮失败时不会误用历史图片。
    """
    coil_id = str(matched_result.get("strCoil") or "").strip()
    steel_dir = _pipeline_exit_grain_steel_dir_name(matched_result, target_context)
    if not coil_id or not steel_dir:
        print(
            "[析出形貌绘图] 缺少受支持的板坯号或钢级，跳过: "
            f"coil_id={coil_id!r}, strSteel={matched_result.get('strSteel')!r}"
        )
        return False

    script_dir = _os.path.join(
        PIPELINE_IMAGE_GENERATOR_BIN_DIR,
        "ZHB",
        steel_dir,
    )
    script_path = _os.path.join(script_dir, "bigmodel_Picture_pre.py")
    source_microstructure_dir = _os.path.join(
        script_dir,
        "Temp_Physical_Metallurgy",
        coil_id,
        "Microstruct",
    )
    model_manage_dir = _os.path.join(
        PIPELINE_IMAGE_GENERATOR_BIN_DIR,
        "ModelManage",
        coil_id,
    )
    target_microstructure_dir = _os.path.join(model_manage_dir, "Microstruct")
    target_image_dir = _os.path.join(model_manage_dir, "Image")
    report_image_path = _os.path.join(target_image_dir, "析出形貌.png")

    # 先移除上一轮报告图。失败时报告图片扫描自然跳过该图，不影响其它报告。
    try:
        if _os.path.isfile(report_image_path):
            _os.remove(report_image_path)
    except OSError as exc:
        print(f"[析出形貌绘图] 无法清理旧报告图，跳过本轮: {exc}")
        return False

    started_at = time.time()
    try:
        if not _os.path.isfile(script_path):
            raise FileNotFoundError(f"未找到析出形貌绘图脚本: {script_path}")
        if not _os.path.isfile(PIPELINE_PRECIPITATE_IMAGE_PROCESSOR_DLL_PATH):
            raise FileNotFoundError(
                "未找到析出形貌处理DLL: "
                + PIPELINE_PRECIPITATE_IMAGE_PROCESSOR_DLL_PATH
            )

        # 清除本轮会混淆时效判断的临时结果；保留 P.xml 输入文件。
        generated_dir = _os.path.join(source_microstructure_dir, "P")
        if _os.path.isdir(generated_dir):
            shutil.rmtree(generated_dir)
        for file_name in ("P.png", "P2.png", "P_processed.png", "P3.png"):
            stale_path = _os.path.join(source_microstructure_dir, file_name)
            if _os.path.isfile(stale_path):
                _os.remove(stale_path)

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if _os.name == "nt" else 0
        process = subprocess.Popen(
            [sys.executable, script_path, coil_id],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )
        try:
            stdout_text, stderr_text = process.communicate(timeout=900)
        except subprocess.TimeoutExpired as exc:
            _terminate_pipeline_drawing_processes([process])
            raise TimeoutError("bigmodel_Picture_pre.py 绘制超过900秒") from exc
        if process.returncode != 0:
            detail = (stderr_text or stdout_text or "").strip()
            raise RuntimeError(
                f"bigmodel_Picture_pre.py 返回 {process.returncode}: {detail[-800:]}"
            )

        raw_image_path = _os.path.join(source_microstructure_dir, "P.png")
        if not _os.path.isfile(raw_image_path):
            raise FileNotFoundError(f"析出形貌脚本未生成 P.png: {raw_image_path}")
        if _os.path.getmtime(raw_image_path) + 1 < started_at:
            raise RuntimeError("拒绝使用本轮绘图启动前的旧 P.png")

        # pythonnet 调用专用 C# DLL，严格复用 ANSTEEL_BYQ 的三步图片处理逻辑。
        with IMAGE_GENERATOR_CALL_LOCK:
            _prepare_pipeline_image_generator_runtime()
            import clr
            clr.AddReference(PIPELINE_PRECIPITATE_IMAGE_PROCESSOR_DLL_PATH)
            from ANSTEEL_PrecipitateImageProcessorLib import PrecipitateImageProcessor

            old_cwd = _os.getcwd()
            try:
                _os.chdir(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
                processor = PrecipitateImageProcessor()
                processor.Process(
                    source_microstructure_dir,
                    _os.path.join(model_manage_dir, "Fv.txt"),
                )
            finally:
                _os.chdir(old_cwd)

        processed_names = ("P.png", "P2.png", "P_processed.png", "P3.png")
        for file_name in processed_names:
            source_path = _os.path.join(source_microstructure_dir, file_name)
            if not _os.path.isfile(source_path):
                raise FileNotFoundError(f"析出形貌后处理缺少图片: {source_path}")

        for file_name in processed_names:
            _copy_pipeline_image_atomically(
                _os.path.join(source_microstructure_dir, file_name),
                _os.path.join(target_microstructure_dir, file_name),
            )
        _copy_pipeline_image_atomically(
            _os.path.join(source_microstructure_dir, "P3.png"),
            report_image_path,
        )
        print(
            "[析出形貌绘图] 绘制、DLL后处理及复制完成: "
            f"coil_id={coil_id}, report_image={report_image_path}"
        )
        return True
    except Exception as exc:
        try:
            if _os.path.isfile(report_image_path):
                _os.remove(report_image_path)
        except OSError:
            pass
        print(
            "[析出形貌绘图] 本轮失败，跳过析出形貌图并继续生成报告: "
            f"coil_id={coil_id}, {type(exc).__name__}: {exc}"
        )
        return False

def _ensure_oracle_network_config() -> None:
    """准备项目级 sqlnet.ora，禁用 NTS，避免 ORA-12638 身份证明检索失败。"""
    try:
        _os.makedirs(ORACLE_TNS_ADMIN, exist_ok=True)
        sqlnet_path = _os.path.join(ORACLE_TNS_ADMIN, "sqlnet.ora")
        if not _os.path.exists(sqlnet_path):
            with open(sqlnet_path, "w", encoding="ascii") as f:
                f.write("SQLNET.AUTHENTICATION_SERVICES=(NONE)\n")
                f.write("NAMES.DIRECTORY_PATH=(TNSNAMES,EZCONNECT)\n")
        _os.environ.setdefault("TNS_ADMIN", ORACLE_TNS_ADMIN)
    except Exception as exc:
        print(f"[Oracle匹配] 准备项目级 sqlnet.ora 失败: {exc}")

# 成分字段与 Oracle 表字段同名，直接按规格上下限过滤。
COMPONENT_FIELDS = [
    "C", "SI", "MN", "P", "S", "N", "NB", "V", "TI",
    "AL", "ALS", "CU", "CR", "NI", "CO", "MO", "B",
]

COMPONENT_FIELD_SET = set(COMPONENT_FIELDS)
COMPONENT_MENTION_ALIASES = {
    "C": ["碳", "含碳量", "C"],
    "SI": ["硅", "Si", "SI"],
    "MN": ["锰", "Mn", "MN"],
    "P": ["磷", "P"],
    "S": ["硫", "S"],
    "N": ["氮", "N"],
    "NB": ["铌", "Nb", "NB"],
    "V": ["钒", "V"],
    "TI": ["钛", "Ti", "TI"],
    "AL": ["铝", "Al", "AL"],
    "ALS": ["酸溶铝", "Als", "ALS"],
    "CU": ["铜", "Cu", "CU"],
    "CR": ["铬", "Cr", "CR"],
    "NI": ["镍", "Ni", "NI"],
    "CO": ["钴", "Co", "CO"],
    "MO": ["钼", "Mo", "MO"],
    "B": ["硼", "B"],
}

# 工艺字段：规格 JSON 字段前缀 -> Oracle 实绩字段。
PROCESS_FIELD_MAP = {
    "SOAKING_TEMP": "SOAK_T",
    "FET": "F1_ET_AVG",
    "FDT": "F7_RT_AVG",
    "CT": "CT_AVG",
    "QUENCHING_TEMP": "AUSTENITIZING_TEMP_Q",
    "TEMPERING_TEMP": "TEMPERING_TEMP_T",
}

# 性能字段：规格 JSON 字段前缀 -> Oracle 实绩字段。
PERFORMANCE_FIELD_MAP = {
    "YS": "YS",
    "TS": "TS",
    "EL": "BREAK_EL",
}

# LLM 微调时只允许修改这些成分、厚度和温度工艺字段，避免破坏 C# DLL 依赖的固定结构。
REFINABLE_COMPONENT_FIELDS = set(COMPONENT_FIELDS)
REFINABLE_PROCESS_FIELD_TO_SPEC = {
    "SOAK_T": "SOAKING_TEMP",
    "F1_ET_AVG": "FET",
    "F7_RT_AVG": "FDT",
    "CT_AVG": "CT",
    "AUSTENITIZING_TEMP_Q": "QUENCHING_TEMP",
    "TEMPERING_TEMP_T": "TEMPERING_TEMP",
    # 设定/平均温度字段允许跟随同一工艺边界微调，但不允许改道次分配。
    "F1_ET_SET": "FET",
    "F7_RT_SET": "FDT",
    "AUSTENITIZING_TEMP_AVG_Q": "QUENCHING_TEMP",
    "TEMPERING_TEMP_AVG_T": "TEMPERING_TEMP",
}
ROLLING_SCHEDULE_FIELDS = {f"F{i}_DH_AVG" for i in range(1, 8)}
REFINABLE_THICKNESS_FIELDS = {"AIM_HEIGHT", "MAT_ACT_THICK_RCL", *ROLLING_SCHEDULE_FIELDS}
REFINABLE_FIELDS = REFINABLE_COMPONENT_FIELDS | set(REFINABLE_PROCESS_FIELD_TO_SPEC) | REFINABLE_THICKNESS_FIELDS

SENSITIVE_MATCHED_RESULT_FIELDS = {
    "COIL_ID",
    "COIL_CREATETIME",
    "STEELGRADE",
    "MEL_NO",
    "RSLAB_ID",
    "RSLAB_TYPE",
    "RSLAB_CREATETIME",
    "PACK_NO",
    "IN_MAT_NO_RCL",
    "STEEL_SIGN",
    "SLAB_ID",
    "FURNACE_ID",
    "HEAT_FURNACE_ID",
    "SLAB_REMARK",
    "MAT_ID",
    "PLATE_ID",
    "PROD_SLAB_DATE",
    "PROD_PLATE_DATE",
    "MAT_NO",
    "TENSILE_TEST_MARK",
    "SAMPLE_NO",
}
SENSITIVE_MATCHED_RESULT_TOP_LEVEL_FIELDS = {"strCoil", "strSteel"}


def _to_float(value):
    """将规格值安全转为 float；无法转换时返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_effective_min(value) -> bool:
    """min=0 视为规格默认下限，不生成 SQL 条件。"""
    number = _to_float(value)
    return number is not None and number > 0


def _is_effective_max(value) -> bool:
    """max=9999 视为规格默认上限，不生成 SQL 条件。"""
    number = _to_float(value)
    return number is not None and number < 9999


def _user_mentioned_component_fields(user_message: str) -> set[str]:
    """从用户原文中识别被强调的成分字段。"""
    mentioned = set()
    message = user_message or ""
    for field, aliases in COMPONENT_MENTION_ALIASES.items():
        for alias in aliases:
            if re.fullmatch(r"[A-Za-z]+", alias):
                if re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", message, re.IGNORECASE):
                    mentioned.add(field)
                    break
            elif alias in message:
                mentioned.add(field)
                break
    return mentioned


def _component_label_from_context(field: str, user_message: str) -> str:
    """优先使用用户原文里的成分叫法；未提及时用字段名本身。"""
    message = user_message or ""
    for alias in COMPONENT_MENTION_ALIASES.get(field, []):
        if re.fullmatch(r"[A-Za-z]+", alias):
            match = re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", message, re.IGNORECASE)
            if match:
                return match.group(0)
        elif alias in message:
            return f"{alias} {field}" if alias.upper() != field else field
    return field


def _report_component_fields(spec_result: dict, user_message: str) -> list[str]:
    """报告成分显示项：来自 spec_result 有效边界，并补充用户提示词强调的成分。"""
    fields = []
    seen = set()

    def add_field(field: str) -> None:
        if field in COMPONENT_FIELD_SET and field not in seen:
            fields.append(field)
            seen.add(field)

    if isinstance(spec_result, dict):
        for key, value in spec_result.items():
            if not isinstance(key, str) or not (key.endswith("_min") or key.endswith("_max")):
                continue
            field = key.rsplit("_", 1)[0].upper()
            if field not in COMPONENT_FIELD_SET:
                continue
            if key.endswith("_min") and _is_effective_min(value):
                add_field(field)
            elif key.endswith("_max") and _is_effective_max(value):
                add_field(field)

    for field in COMPONENT_FIELDS:
        if field in _user_mentioned_component_fields(user_message):
            add_field(field)

    return fields


def _sanitize_matched_result_for_llm(value):
    """移除 matched_result 中不允许进入报告 prompt/上下文的身份追溯字段。"""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in SENSITIVE_MATCHED_RESULT_TOP_LEVEL_FIELDS:
                continue
            if key_text.upper() in SENSITIVE_MATCHED_RESULT_FIELDS:
                continue
            if key_text.upper() in ROLLING_SCHEDULE_FIELDS:
                continue
            sanitized[key] = _sanitize_matched_result_for_llm(item)
        return sanitized
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            sanitized_item = _sanitize_matched_result_for_llm(item)
            if sanitized_item == {}:
                continue
            sanitized_items.append(sanitized_item)
        return sanitized_items
    return value


def _collect_sensitive_matched_terms(matched_result: dict) -> list[str]:
    """收集本次 matched_result 中敏感字段的值，供流式输出兜底过滤。"""
    terms = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                if key_text in SENSITIVE_MATCHED_RESULT_TOP_LEVEL_FIELDS or key_text.upper() in SENSITIVE_MATCHED_RESULT_FIELDS:
                    item_text = str(item).strip()
                    if item_text:
                        terms.append(item_text)
                    continue
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(matched_result)
    return _filter_sensitive_redaction_terms(terms)


def _filter_sensitive_redaction_terms(terms: list[str]) -> list[str]:
    """过滤过短或过泛的脱敏词，避免把字段名/数值中的普通字符删掉。"""
    filtered = []
    seen = set()
    for term in terms or []:
        text = str(term or "").strip()
        if not text or text in seen:
            continue
        if len(text) < 3:
            continue
        if re.fullmatch(r"[A-Za-z]", text):
            continue
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text) and len(text) < 6:
            continue
        filtered.append(text)
        seen.add(text)
    return filtered


def _fact_table_unit_for_field(field_name: str) -> str:
    upper_name = str(field_name).upper()
    if upper_name in COMPONENT_FIELD_SET or upper_name in PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC:
        return "wt%"
    if "TEMP" in upper_name or upper_name in {
        "SOAK_T", "F1_ET_AVG", "F7_RT_AVG", "F1_ET_SET", "F7_RT_SET", "CT_AVG",
        "FET", "FDT", "TEMP_ENTR", "SELF_TEMP", "FURNACE_EXIT_TEMP", "RET",
    }:
        return "℃"
    if (
        "THICK" in upper_name
        or "HEIGHT" in upper_name
        or upper_name.endswith("_DH_AVG")
        or upper_name.endswith("_DH_CAL")
        or upper_name in {"AIM_THICK", "AIM_HEIGHT", "FEH", "SLAB_THICK"}
    ):
        return "mm"
    if "WIDTH" in upper_name or upper_name.endswith("_DW_CAL"):
        return "mm"
    if "LEN" in upper_name:
        return "mm"
    if "SPD" in upper_name or "SPEED" in upper_name:
        return "m/s"
    if "FORCE" in upper_name:
        return "kN"
    if upper_name in {"YS", "TS"}:
        return "MPa"
    if upper_name in {"EL", "BREAK_EL"}:
        return "%"
    if upper_name == "AKV":
        return "J"
    if "TIME" in upper_name:
        return "min"
    return ""


def _fact_table_category_for_field(field_name: str) -> str:
    upper_name = str(field_name).upper()
    if upper_name in COMPONENT_FIELD_SET or upper_name in PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC:
        return "成分"
    if upper_name in {"YS", "TS", "EL", "BREAK_EL", "AKV"}:
        return "力学性能"
    if upper_name.startswith("F") and upper_name.endswith("_DH_AVG"):
        return "轧制道次"
    if re.fullmatch(r"N\d+_(DH_CAL|DT_CAL|DW_CAL|FORCE|GAP|SPD)", upper_name):
        return "轧制道次"
    return "工艺"


def _build_full_fact_table_from_matched_result(matched_result: dict) -> list[dict]:
    """从 matched_result.arrBody 构造完整事实表，排除身份追溯字段但保留工艺/道次字段。"""
    fact_table = []
    for item in matched_result.get("arrBody", []) if isinstance(matched_result, dict) else []:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        field, value = next(iter(item.items()))
        field_name = str(field)
        upper_name = field_name.upper()
        if upper_name in SENSITIVE_MATCHED_RESULT_FIELDS:
            continue
        if value is None or str(value).strip() == "":
            continue
        fact_table.append({
            "类别": _fact_table_category_for_field(field_name),
            "项目": field_name,
            "字段": field_name,
            "数值": _format_oracle_value(value),
            "单位": _fact_table_unit_for_field(field_name),
        })
    return fact_table


def _redact_sensitive_text(text: str, terms: list[str]) -> str:
    """从上下文文本中移除敏感字段名和值。"""
    redacted = text or ""
    for term in _filter_sensitive_redaction_terms(terms):
        redacted = redacted.replace(term, "")
    return redacted






async def _collect_llm_text(llm, messages: list, forbidden_terms: list[str], log_prefix: str) -> str:
    chunks = []
    async for chunk in llm.astream(messages):
        if not chunk.content:
            continue
        safe_content = chunk.content
        for term in forbidden_terms:
            safe_content = safe_content.replace(term, "")
        if safe_content:
            chunks.append(safe_content)
    text = "".join(chunks)
    print(f"[{log_prefix}] 生成完成，正文长度={len(text)}")
    return text


def _fact_table_component_values(fact_table: list[dict]) -> dict[str, str]:
    """从 fact_table 中提取成分项原值，供最终报告做确定性数值校正。"""
    aliases = {
        "C": ("C", "碳"),
        "S": ("S", "硫"),
        "P": ("P", "磷"),
        "MN": ("MN", "Mn", "锰"),
        "SI": ("SI", "Si", "硅"),
        "ALT": ("ALT", "Alt", "全铝"),
        "ALS": ("ALS", "Als", "酸溶铝"),
    }
    values = {}
    for row in fact_table or []:
        if not isinstance(row, dict) or str(row.get("类别", "")).strip() != "成分":
            continue
        label = str(row.get("项目", "")).strip()
        value = str(row.get("数值", "")).strip()
        if not label or not value:
            continue
        for field, names in aliases.items():
            if any(re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", label, re.IGNORECASE) for name in names):
                values[field] = value
                break
    return values


def _enforce_fact_table_component_values(text: str, fact_table: list[dict]) -> str:
    """把最终报告中的关键成分数值强制校正为 fact_table 原值。"""
    corrected = text or ""
    component_values = _fact_table_component_values(fact_table)
    for field, value in component_values.items():
        names = {
            "C": ("C", "碳"),
            "S": ("S", "硫"),
            "P": ("P", "磷"),
            "MN": ("MN", "Mn", "锰"),
            "SI": ("SI", "Si", "硅"),
            "ALT": ("ALT", "Alt", "全铝"),
            "ALS": ("ALS", "Als", "酸溶铝"),
        }[field]
        for name in names:
            if re.fullmatch(r"[A-Za-z]+", name):
                pattern = rf"(?<![A-Za-z])({re.escape(name)})(?![A-Za-z])(\s*(?:含量|实测值|为|=|:|：)?\s*)([0-9]+(?:\.[0-9]+)?)"
            else:
                pattern = rf"({re.escape(name)})(\s*(?:含量|实测值|为|=|:|：)?\s*)([0-9]+(?:\.[0-9]+)?)"

            def replace_if_different(match):
                old_value = match.group(3)
                if old_value == value:
                    return match.group(0)
                return f"{match.group(1)}{match.group(2)}{value}"

            corrected = re.sub(pattern, replace_if_different, corrected, flags=re.IGNORECASE)
    return corrected


def _write_report_review_debug_context(
    *,
    user_message: str,
    report_user_prompt: str,
    qwen_user_prompt: str,
    draft_report_text: str,
    qwen_report_text: str,
    corrected_report_text: str,
    matched_result_for_llm: dict,
    fact_table: list[dict],
    spec_result: dict,
    knowledge_context: str,
) -> None:
    """把最终报告复核上下文写入 text.text，方便排查数值漂移来源。"""
    debug_path = _os.path.join(_os.path.dirname(__file__), "text.text")
    content = "\n\n".join([
        "==== 用户原始需求 ====\n" + str(user_message or ""),
        "==== fact_table ====\n" + json.dumps(fact_table, ensure_ascii=False, indent=2),
        "==== matched_result_for_llm ====\n" + json.dumps(matched_result_for_llm, ensure_ascii=False, indent=2),
        "==== spec_result ====\n" + json.dumps(spec_result, ensure_ascii=False, indent=2),
        "==== knowledge_context ====\n" + str(knowledge_context or ""),
        "==== report_user_prompt（初稿报告上下文） ====\n" + str(report_user_prompt or ""),
        "==== 初稿待复核报告 ====\n" + str(draft_report_text or ""),
        "==== Qwen 用户提示词 ====\n" + str(qwen_user_prompt or ""),
        "==== Qwen 原始最终报告 ====\n" + str(qwen_report_text or ""),
        "==== 程序按 fact_table 校正后的最终报告 ====\n" + str(corrected_report_text or ""),
    ])
    with open(debug_path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"[Qwen报告复核] 上下文已写入: {debug_path}")


async def _stream_qwen_reviewed_report(
    *,
    user_message: str,
    report_user_prompt: str,
    draft_report_text: str,
    matched_result_for_llm: dict,
    fact_table: list[dict],
    spec_result: dict,
    knowledge_context: str,
    forbidden_terms: list[str],
    report_table_markdown: str,
    history: PersistentChatMessageHistory,
    report_table_sent: bool = False,
):
    qwen_user_prompt = _build_qwen_report_review_user_prompt(
        user_message=user_message,
        draft_report_text=draft_report_text,
        fact_table=fact_table,
    )
    qwen_response_chunks = []
    messages = [
        SystemMessage(content=QWEN_REPORT_REVIEW_SYSTEM_PROMPT),
        HumanMessage(content=qwen_user_prompt),
    ]
    qwen_raw_chunks = []
    async for chunk in qwen_Llm.astream(messages):
        if not chunk.content:
            continue
        safe_content = chunk.content
        for term in forbidden_terms:
            safe_content = safe_content.replace(term, "")
        if not safe_content:
            continue
        qwen_raw_chunks.append(safe_content)

    qwen_report_text = "".join(qwen_raw_chunks)
    corrected_report_text = _enforce_fact_table_component_values(qwen_report_text, fact_table)
    _write_report_review_debug_context(
        user_message=user_message,
        report_user_prompt=report_user_prompt,
        qwen_user_prompt=qwen_user_prompt,
        draft_report_text=draft_report_text,
        qwen_report_text=qwen_report_text,
        corrected_report_text=corrected_report_text,
        matched_result_for_llm=matched_result_for_llm,
        fact_table=fact_table,
        spec_result=spec_result,
        knowledge_context=knowledge_context,
    )
    if corrected_report_text:
        output_text = report_table_markdown + corrected_report_text
        report_table_sent = True
        qwen_response_chunks.append(corrected_report_text)
        yield _ndjson_event("answer_replace", content=output_text)

    history.add_message(HumanMessage(content=qwen_user_prompt))
    if corrected_report_text:
        history.add_message(AIMessage(content=corrected_report_text))
    return


def _numeric_expr(column_name: str) -> str:
    """Oracle 数值字段表达式。match_process_valid 中范围过滤列均为 NUMBER。"""
    return column_name


def _numeric_guard(column_name: str) -> str:
    """NUMBER 字段只需判空，避免空值参与范围比较。"""
    return f"{column_name} IS NOT NULL"


def _append_range_condition(where_parts: list, params: dict, column_name: str, prefix: str, min_value, max_value) -> None:
    """把一个字段的上下限转换为 Oracle WHERE 条件。"""
    expr = _numeric_expr(column_name)
    parts = [_numeric_guard(column_name)]
    if _is_effective_min(min_value):
        param_name = f"{prefix}_min"
        parts.append(f"{expr} >= :{param_name}")
        params[param_name] = _to_float(min_value)
    if _is_effective_max(max_value):
        param_name = f"{prefix}_max"
        parts.append(f"{expr} <= :{param_name}")
        params[param_name] = _to_float(max_value)
    if len(parts) > 1:
        where_parts.append("(" + " AND ".join(parts) + ")")


def _append_thickness_condition(where_parts: list, params: dict, spec_result: dict) -> None:
    """厚度优先用 MAT_ACT_THICK_RCL/1000，若该字段不可用则允许 AIM_HEIGHT 命中。"""
    min_value = spec_result.get("THK_min")
    max_value = spec_result.get("THK_max")
    if not (_is_effective_min(min_value) or _is_effective_max(max_value)):
        return

    params["thk_min"] = _to_float(min_value) if _is_effective_min(min_value) else -1e18
    params["thk_max"] = _to_float(max_value) if _is_effective_max(max_value) else 1e18
    mat_thick = f"{_numeric_expr('MAT_ACT_THICK_RCL')} / 1000"
    aim_height = _numeric_expr("AIM_HEIGHT")
    where_parts.append(
        "("
        f"({_numeric_guard('MAT_ACT_THICK_RCL')} AND {mat_thick} BETWEEN :thk_min AND :thk_max)"
        " OR "
        f"({_numeric_guard('AIM_HEIGHT')} AND {aim_height} BETWEEN :thk_min AND :thk_max)"
        ")"
    )


def _append_simulation_ready_condition(where_parts: list) -> None:
    """限制 Oracle 匹配结果必须具备 DLL 仿真所需的关键热处理字段。"""
    required_positive_fields = [
        "TEMPERING_TEMP_T",
        "TEMPERING_TIME_T",
        "F7_RT_AVG",
        "CT_AVG",
        "MAT_ACT_THICK_RCL",
    ]
    required_parts = [
        f"({_numeric_guard(field)} AND {field} > 0)"
        for field in required_positive_fields
    ]
    # LG700T/LG800T 在 DLL 中不需要淬火参数；其他钢种必须有有效淬火温度和时间。
    required_parts.append(
        "("
        "STEELGRADE LIKE '%LG700T%' OR STEELGRADE LIKE '%LG800T%' OR "
        "((AUSTENITIZING_TEMP_Q IS NOT NULL AND AUSTENITIZING_TEMP_Q > 0) "
        "AND (AUSTENITIZING_TIME_Q IS NOT NULL AND AUSTENITIZING_TIME_Q > 0))"
        ")"
    )
    where_parts.append("(" + " AND ".join(required_parts) + ")")


def _build_match_where(spec_result: dict, include_process: bool = True, include_performance: bool = True) -> tuple[str, dict]:
    """根据规格 JSON 生成 WHERE 子句和绑定参数。"""
    where_parts = []
    params = {}

    _append_simulation_ready_condition(where_parts)
    _append_thickness_condition(where_parts, params, spec_result)

    # 成分条件始终参与前三个查询阶段，是钢种匹配的核心条件。
    for field in COMPONENT_FIELDS:
        _append_range_condition(
            where_parts,
            params,
            field,
            field.lower(),
            spec_result.get(f"{field}_min"),
            spec_result.get(f"{field}_max"),
        )

    # 工艺条件可在第三阶段放开。
    if include_process:
        for spec_prefix, column_name in PROCESS_FIELD_MAP.items():
            _append_range_condition(
                where_parts,
                params,
                column_name,
                spec_prefix.lower(),
                spec_result.get(f"{spec_prefix}_min"),
                spec_result.get(f"{spec_prefix}_max"),
            )

    # 性能条件最先放开，因为历史实绩中性能数据可能缺失或滞后。
    if include_performance:
        for spec_prefix, column_name in PERFORMANCE_FIELD_MAP.items():
            _append_range_condition(
                where_parts,
                params,
                column_name,
                spec_prefix.lower(),
                spec_result.get(f"{spec_prefix}_min"),
                spec_result.get(f"{spec_prefix}_max"),
            )

    return (" AND ".join(where_parts) if where_parts else "1=1"), params


def _load_oracle_driver():
    """优先使用 cx_Oracle 适配本机 11g 客户端；不可用时回退 python-oracledb。"""
    _ensure_oracle_network_config()
    try:
        import cx_Oracle
        return cx_Oracle, "cx_Oracle"
    except ModuleNotFoundError:
        pass

    import oracledb
    # 老版本 Oracle 服务端不支持 python-oracledb thin 模式，尝试启用 thick 模式。
    if oracledb.is_thin_mode() and ORACLE_CLIENT_LIB_DIR and _os.path.exists(_os.path.join(ORACLE_CLIENT_LIB_DIR, "oci.dll")):
        try:
            oracledb.init_oracle_client(lib_dir=ORACLE_CLIENT_LIB_DIR)
            print(f"[Oracle匹配] 已启用 thick 模式: {ORACLE_CLIENT_LIB_DIR}")
        except oracledb.ProgrammingError:
            # Oracle Client 已初始化时会抛出 ProgrammingError，可安全忽略。
            pass
    return oracledb, "oracledb"


def _query_first_oracle_row(
    spec_result: dict,
    include_process: bool = True,
    include_performance: bool = True,
    stage_name: str = "Oracle查询",
):
    """执行一次 Oracle 查询，按 COIL_CREATETIME 倒序返回首条记录。"""
    oracle_driver, driver_name = _load_oracle_driver()

    where_sql, params = _build_match_where(
        spec_result,
        include_process=include_process,
        include_performance=include_performance,
    )
    condition_count = 0 if where_sql == "1=1" else len(params)
    print(
        f"[Oracle匹配] {stage_name}: 条件数={condition_count}, "
        f"参数={sorted(params.keys())}, 驱动={driver_name}"
    )
    if params:
        print(
            f"[Oracle匹配] {stage_name} 参数值: "
            f"{json.dumps(params, ensure_ascii=False, sort_keys=True)}"
        )
    dsn = oracle_driver.makedsn(ORACLE_HOST, ORACLE_PORT, service_name=ORACLE_SERVICE_NAME)
    # Oracle 11g 不支持 FETCH FIRST，使用 ROWNUM 外层过滤取倒序第一条。
    sql = f"""
        SELECT *
        FROM (
            SELECT *
            FROM {ORACLE_TABLE}
            WHERE {where_sql}
            ORDER BY COIL_CREATETIME DESC
        )
        WHERE ROWNUM = 1
    """
    with oracle_driver.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if not row:
                return None
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))


def _query_first_oracle_row_without_filters(stage_name: str = "Oracle无筛选兜底查询"):
    """去掉所有筛选条件，优先按 COIL_CREATETIME 倒序返回首条 Oracle 实绩。

    如果排序兜底仍没有取到数据，再完全不排序取表中第一条。
    这样只要 Oracle 实绩表中存在任意记录，最终匹配函数就不会返回空 arrBody。
    """
    oracle_driver, driver_name = _load_oracle_driver()
    print(f"[Oracle匹配] {stage_name}: 条件数=0, 驱动={driver_name}")
    dsn = oracle_driver.makedsn(ORACLE_HOST, ORACLE_PORT, service_name=ORACLE_SERVICE_NAME)
    ordered_sql = f"""
        SELECT *
        FROM (
            SELECT *
            FROM {ORACLE_TABLE}
            ORDER BY COIL_CREATETIME DESC
        )
        WHERE ROWNUM = 1
    """
    unordered_sql = f"""
        SELECT *
        FROM (
            SELECT *
            FROM {ORACLE_TABLE}
        )
        WHERE ROWNUM = 1
    """
    with oracle_driver.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(ordered_sql)
            row = cur.fetchone()
            if row:
                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, row))

            print(f"[Oracle匹配] {stage_name}: 按 COIL_CREATETIME 排序未取到数据，改为无排序取第一条")
            cur.execute(unordered_sql)
            row = cur.fetchone()
            if not row:
                return None
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))


def _format_match_error(stage_name: str, exc: Exception) -> str:
    """格式化 Oracle 匹配异常，不包含密码等敏感信息。"""
    return f"{stage_name}: {type(exc).__name__}: {exc}"


def _format_oracle_value(value) -> str:
    """Oracle 值统一转字符串；None 返回空字符串。"""
    if value is None:
        return ""
    return str(value)


def _build_match_response(
    row: dict | None,
    is_state: bool,
    session_id: str,
    message: str | None = None,
    error: str | None = None,
) -> dict:
    """把 Oracle 首行记录转换为前端约定的 JSON 包装结构。"""
    if not row:
        response = {
            "isState": False,
            "strCoil": "",
            "strSteel": "",
            "session_key": session_id,
            "arrBody": [],
        }
        if message:
            response["message"] = message
        if error:
            response["error"] = error
        return response

    normalized = {str(k).upper(): v for k, v in row.items()}
    response = {
        "isState": bool(is_state),
        "strCoil": _format_oracle_value(normalized.get("IN_MAT_NO_RCL")),
        "strSteel": _format_oracle_value(normalized.get("STEELGRADE")),
        "session_key": session_id,
        "arrBody": [
            {key: _format_oracle_value(value)}
            for key, value in normalized.items()
        ],
    }
    if message:
        response["message"] = message
    if error:
        response["error"] = error
    return response


def _prepare_image_generator_runtime() -> None:
    """准备 pythonnet 和 DLL 搜索路径，确保 ImageGeneratorLib 的依赖能被找到。"""
    if not _os.path.exists(IMAGE_GENERATOR_DLL_PATH):
        raise FileNotFoundError(f"未找到绘图DLL: {IMAGE_GENERATOR_DLL_PATH}")

    # DLL 目录放入 sys.path，便于 pythonnet 解析同目录下的 .NET 程序集依赖。
    if IMAGE_GENERATOR_BIN_DIR not in sys.path:
        sys.path.insert(0, IMAGE_GENERATOR_BIN_DIR)

    # Windows 下把 bin/Debug 加入原生 DLL 搜索路径，供 OxyPlot/Oracle 等依赖加载。
    if hasattr(_os, "add_dll_directory"):
        handle = _os.add_dll_directory(IMAGE_GENERATOR_BIN_DIR)
        IMAGE_GENERATOR_DLL_DIRECTORY_HANDLES.append(handle)

    # pythonnet 3 支持显式选择 netfx；如果运行时已初始化，会抛 RuntimeError，可忽略。
    try:
        from pythonnet import load
        try:
            load("netfx")
        except RuntimeError:
            pass
    except ModuleNotFoundError:
        # 老版本 pythonnet 可能没有 pythonnet.load，但只要 import clr 可用即可。
        pass


def _matched_result_body_to_row(matched_result: dict) -> dict[str, str]:
    """把 matched_result.arrBody 转为大写字段名映射，便于 DLL 入参诊断。"""
    row = {}
    for item in matched_result.get("arrBody", []):
        if isinstance(item, dict) and len(item) == 1:
            key, value = next(iter(item.items()))
            row[str(key).upper()] = _format_oracle_value(value)
    return row


def _load_hot_rolling_report_template() -> str:
    """每次生成报告时读取同目录模板；读取失败只降级格式参考，不中断报告。"""
    template_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "热轧工艺设计报告模板.md",
    )
    try:
        with open(template_path, "r", encoding="utf-8") as template_file:
            template_text = template_file.read()
        if not template_text.strip():
            print(f"[管线钢报告生成] 热轧报告模板为空，使用内置格式规则: {template_path}")
            return ""
        print(
            f"[管线钢报告生成] 已读取热轧报告模板: {template_path}, "
            f"长度={len(template_text)}"
        )
        return template_text
    except (OSError, UnicodeError) as exc:
        print(f"[管线钢报告生成] 热轧报告模板读取失败，使用内置格式规则: {exc}")
        return ""


def _build_rag_references_markdown(knowledge_docs: list[dict]) -> str:
    """从本轮 RAG 召回片段中提取并去重文献名，生成报告末尾参考文献。"""
    reference_names = []
    seen_names = set()
    source_pattern = re.compile(r"(?m)^\[来源\s*[:：]\s*(.+?)\]\s*$")

    for doc in knowledge_docs or []:
        if not isinstance(doc, dict):
            continue
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        content = str(doc.get("content") or "")
        source_match = source_pattern.search(content)
        source = (
            (source_match.group(1) if source_match else "")
            or str(doc.get("title") or "")
            or str(metadata.get("title") or "")
            or str(metadata.get("source") or "")
            or str(doc.get("source") or "")
        ).strip()
        if not source or source.casefold() in {"unknown", "knowledge_base_agent"}:
            continue
        if source.startswith("search_") and source.endswith("_knowledge_base"):
            continue

        # 只展示文献文件名，禁止把服务器目录、URL 查询参数或知识库内部路径写入报告。
        normalized_source = source.split("?", 1)[0].split("#", 1)[0].replace("\\", "/")
        document_name = normalized_source.rsplit("/", 1)[-1].strip()
        document_name = re.sub(
            r"\.(?:pdf|docx?|txt|md|html?|pptx?)$",
            "",
            document_name,
            flags=re.IGNORECASE,
        ).strip()
        if not document_name:
            continue
        dedupe_key = document_name.casefold()
        if dedupe_key in seen_names:
            continue
        seen_names.add(dedupe_key)
        reference_names.append(document_name)

    lines = ["## 8. 参考文献", ""]
    if reference_names:
        lines.extend(
            f"[{index}] {document_name}"
            for index, document_name in enumerate(reference_names, start=1)
        )
    else:
        lines.append("本轮 RAG 检索未召回可列出的参考文献。")
    return "\n".join(lines) + "\n"


def _read_pipeline_torque_values_for_report(matched_result: dict) -> list[str]:
    """读取本轮物理冶金计算输出的 MF.txt，按文件行序返回各道次扭矩。"""
    coil_id = str(matched_result.get("strCoil") or "").strip()
    if not coil_id:
        return []

    # 报告优先使用 DLL 已复制到统一 ModelManage 目录的本轮结果。
    candidate_paths = [
        _os.path.join(
            PIPELINE_IMAGE_GENERATOR_BIN_DIR,
            "ModelManage",
            coil_id,
            "Physical Metallurgy Results",
            "MF.txt",
        )
    ]
    # 兼容手工运行物理冶金程序、尚未复制到 ModelManage 的调试场景。
    zhb_root = _os.path.join(PIPELINE_IMAGE_GENERATOR_BIN_DIR, "ZHB")
    if _os.path.isdir(zhb_root):
        for steel_dir_name in ("Physical_Metallurgy_X65", "Physical_Metallurgy_X70", "Physical_Metallurgy_X80NG"):
            candidate_paths.append(
                _os.path.join(
                    zhb_root,
                    steel_dir_name,
                    "Temp_Physical_Metallurgy",
                    coil_id,
                    "Physical Metallurgy Results",
                    "MF.txt",
                )
            )

    torque_path = next((path for path in candidate_paths if _os.path.isfile(path)), None)
    if not torque_path:
        print(f"[管线钢报告生成] 未找到道次扭矩来源 MF.txt: coil_id={coil_id}")
        return []

    values = []
    try:
        with open(torque_path, "r", encoding="utf-8-sig") as torque_file:
            for line_number, raw_line in enumerate(torque_file, start=1):
                normalized_line = raw_line.strip().replace(",", " ")
                if not normalized_line:
                    continue
                token = normalized_line.split()[0]
                try:
                    float(token)
                    # 报告必须逐字采用仿真文件结果，因此校验为数值后仍保留原始精度。
                    values.append(token)
                except ValueError:
                    print(
                        "[管线钢报告生成] MF.txt 存在非数值行，已跳过: "
                        f"path={torque_path}, line={line_number}, value={raw_line.strip()!r}"
                    )
        print(
            f"[管线钢报告生成] 已读取道次扭矩来源 MF.txt: "
            f"path={torque_path}, values={len(values)}"
        )
        return values
    except OSError as exc:
        print(f"[管线钢报告生成] 读取 MF.txt 失败: path={torque_path}, error={exc}")
        return []


def _build_pipeline_rolling_schedule_markdown(matched_result: dict) -> str:
    """根据最终 matched_result 与 MFS 仿真结果生成报告中的轧制规程表。"""
    row = _matched_result_body_to_row(matched_result)
    torque_values = _read_pipeline_torque_values_for_report(matched_result)

    def as_int(value) -> int:
        try:
            return max(0, int(float(value)))
        except (TypeError, ValueError):
            return 0

    def as_number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def display(value, digits: int = 2) -> str:
        number = as_number(value)
        if number is None:
            return "-"
        return f"{number:.{digits}f}"

    def display_speed_m_per_second(value) -> str:
        """matched_result 的道次速度单位为 m/min，报告表统一换算为 m/s。"""
        speed_m_per_minute = as_number(value)
        if speed_m_per_minute is None:
            return "-"
        return f"{speed_m_per_minute / 60.0:.2f}"

    rough_passes = as_int(row.get("R_PASS_ACT"))
    finish_passes = as_int(row.get("F_PASS_ACT"))
    expected_passes = min(30, rough_passes + finish_passes)

    # 道次计数字段异常时，按连续正厚度字段保守识别实际有效道次。
    if expected_passes <= 0:
        for pass_index in range(1, 31):
            thickness = as_number(row.get(f"N{pass_index}_DH_CAL"))
            if thickness is None or thickness <= 0:
                break
            expected_passes = pass_index

    lines = [
        '<div class="report-table-caption"><strong>表3　设计轧制规程</strong></div>',
        "",
        "| 道次 | 阶段 | 出口厚度/mm | 道次压下量/mm | 变形温度/℃ | 出口宽度/mm | 轧制速度/(m/s) | 轧制力/kN | 扭矩/(kN·m) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    previous_thickness = as_number(row.get("SLAB_THICK"))
    emitted = 0
    for pass_index in range(1, expected_passes + 1):
        thickness = as_number(row.get(f"N{pass_index}_DH_CAL"))
        if thickness is None or thickness <= 0:
            break
        reduction = (
            previous_thickness - thickness
            if previous_thickness is not None and previous_thickness > thickness
            else None
        )
        stage = "粗轧" if pass_index <= rough_passes else "精轧"
        lines.append(
            "| {pass_no} | {stage} | {thickness} | {reduction} | {temperature} | "
            "{width} | {speed} | {force} | {torque} |".format(
                pass_no=pass_index,
                stage=stage,
                thickness=display(thickness),
                reduction=display(reduction),
                temperature=display(row.get(f"N{pass_index}_DT_CAL")),
                width=display(row.get(f"N{pass_index}_DW_CAL")),
                speed=display_speed_m_per_second(row.get(f"N{pass_index}_SPD")),
                force=display(row.get(f"N{pass_index}_FORCE")),
                torque=(
                    torque_values[pass_index - 1]
                    if pass_index <= len(torque_values)
                    else "-"
                ),
            )
        )
        previous_thickness = thickness
        emitted += 1

    if emitted == 0:
        return ""
    return "\n".join(lines) + "\n\n"


def _build_pipeline_performance_standard_markdown(spec_result: dict) -> str:
    """根据最终规格边界生成报告第一部分末尾的力学性能标准表。

    参数:
        spec_result: 规格提取、标准知识库和硬编码兜底合并后的最终范围 JSON。

    返回:
        表1的 Markdown。YS、TS、EL、AKV 始终按固定顺序展示；默认无约束边界
        ``0/9999`` 显示为 ``-``，不会被误写成真实产品标准。
    """
    performance_fields = [
        ("屈服强度 YS", "YS", "MPa"),
        ("抗拉强度 TS", "TS", "MPa"),
        ("断后伸长率 EL", "EL", "%"),
        ("冲击功 AKV", "AKV", "J"),
    ]

    def display_bound(value, *, is_minimum: bool) -> str:
        effective = _is_effective_min(value) if is_minimum else _is_effective_max(value)
        return str(value) if effective else "-"

    lines = [
        '<div class="report-table-caption"><strong>表1　力学性能标准</strong></div>',
        "",
        "| 力学性能项目 | 标准下限 | 标准上限 | 单位 |",
        "|:---:|:---:|:---:|:---:|",
    ]
    for label, field, unit in performance_fields:
        lines.append(
            f"| {label} | "
            f"{display_bound(spec_result.get(f'{field}_min'), is_minimum=True)} | "
            f"{display_bound(spec_result.get(f'{field}_max'), is_minimum=False)} | "
            f"{unit} |"
        )
    return "\n".join(lines) + "\n\n"


def _invalid_positive_fields(row: dict[str, str], fields: list[str]) -> list[str]:
    """检查字段是否存在且为正数，返回缺失或无效字段描述。"""
    invalid = []
    for field in fields:
        raw_value = row.get(field)
        number = _to_float(raw_value)
        if number is None or number <= 0:
            invalid.append(f"{field}={raw_value!r}")
    return invalid


def _diagnose_image_generator_input(matched_result: dict) -> list[str]:
    """返回 DLL 绘图入参不满足计算条件的原因列表。"""
    reasons = []
    str_coil = matched_result.get("strCoil")
    arr_body = matched_result.get("arrBody")
    if not str_coil:
        reasons.append("缺少 strCoil")
    if not arr_body:
        reasons.append("缺少 arrBody")
        return reasons

    row = _matched_result_body_to_row(matched_result)
    steelgrade = str(row.get("STEELGRADE") or matched_result.get("strSteel") or "")
    required_fields = [
        "C", "SI", "MN", "P", "S", "CR", "MO", "NB", "V", "TI", "N", "ALS", "CU", "NI", "CO", "B",
        "F7_RT_AVG", "CT_AVG", "TEMPERING_TEMP_T", "TEMPERING_TIME_T", "MAT_ACT_THICK_RCL",
        "SLAB_HEIGHT", "SLAB_WIDTH", "AIM_HEIGHT", "AIM_WIDTH",
    ]
    missing_required = [field for field in required_fields if str(row.get(field, "")).strip() == ""]
    if missing_required:
        reasons.append("DLL基础字段缺失: " + ", ".join(missing_required))

    invalid_positive = _invalid_positive_fields(
        row,
        ["F7_RT_AVG", "CT_AVG", "TEMPERING_TEMP_T", "TEMPERING_TIME_T", "MAT_ACT_THICK_RCL"],
    )
    if invalid_positive:
        reasons.append("DLL基础正数校验失败: " + ", ".join(invalid_positive))

    if not ("LG700T" in steelgrade or "LG800T" in steelgrade):
        invalid_quench = _invalid_positive_fields(row, ["AUSTENITIZING_TEMP_Q", "AUSTENITIZING_TIME_Q"])
        if invalid_quench:
            reasons.append("非LG700T/LG800T钢种必须具备有效淬火参数: " + ", ".join(invalid_quench))
    return reasons


def _generate_images_with_dll(matched_result: dict) -> str | None:
    """同步调用 C# ImageGeneratorLib.dll 绘图，返回 DLL 原始结果；失败只打印日志。"""
    if not isinstance(matched_result, dict):
        print("[绘图DLL] matched_result 不是 JSON 对象，跳过绘图")
        return None
    if not matched_result.get("strCoil") or not matched_result.get("arrBody"):
        print(
            "[绘图DLL] matched_result 缺少 strCoil 或 arrBody，跳过绘图；"
            f"strCoil={matched_result.get('strCoil')!r}, "
            f"arrBody长度={len(matched_result.get('arrBody') or [])}"
        )
        return None

    row_for_log = _matched_result_body_to_row(matched_result)
    diagnosis_reasons = _diagnose_image_generator_input(matched_result)
    print(
        "[绘图DLL] 准备调用: "
        f"strCoil={matched_result.get('strCoil')}, "
        f"strSteel={matched_result.get('strSteel')}, "
        f"STEELGRADE={row_for_log.get('STEELGRADE')}, "
        f"arrBody字段数={len(row_for_log)}"
    )
    if diagnosis_reasons:
        print("[绘图DLL] 入参预检发现风险: " + "；".join(diagnosis_reasons))

    # 传给 C# DLL 的必须是原始 matched_result JSON 字符串，不做 Markdown 包装。
    json_input = json.dumps(matched_result, ensure_ascii=False)

    try:
        with IMAGE_GENERATOR_CALL_LOCK:
            _prepare_image_generator_runtime()
            old_cwd = _os.getcwd()
            try:
                # DLL 内部大量使用 .\RLZ、.\Physical_Matel_RCL 等相对路径，必须切到 bin/Debug。
                _os.chdir(IMAGE_GENERATOR_BIN_DIR)
                import clr
                clr.AddReference(IMAGE_GENERATOR_DLL_PATH)
                from ImageGeneratorLib import ImageGenerator

                result = str(ImageGenerator.GenerateAllImages(json_input))
                if result == "true":
                    print(f"[绘图DLL] 绘图成功: strCoil={matched_result.get('strCoil')}")
                else:
                    print(
                        "[绘图DLL] DLL未完成仿真计算，返回原因: "
                        f"{result}; strCoil={matched_result.get('strCoil')}, "
                        f"STEELGRADE={row_for_log.get('STEELGRADE')}, "
                        f"预检风险={'；'.join(diagnosis_reasons) if diagnosis_reasons else '无'}"
                    )
                return result
            finally:
                _os.chdir(old_cwd)
    except Exception as exc:
        print(
            f"[绘图DLL] 调用异常，未完成仿真计算: {type(exc).__name__}: {exc}; "
            f"strCoil={matched_result.get('strCoil')}, STEELGRADE={row_for_log.get('STEELGRADE')}"
        )
        return None


def _prepare_pipeline_image_generator_runtime() -> None:
    """准备管线钢 ANSTEEL_ImageGeneratorLib 的运行时和依赖搜索路径。"""
    if not _os.path.exists(PIPELINE_IMAGE_GENERATOR_DLL_PATH):
        raise FileNotFoundError(f"未找到管线钢绘图DLL: {PIPELINE_IMAGE_GENERATOR_DLL_PATH}")

    if PIPELINE_IMAGE_GENERATOR_BIN_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_IMAGE_GENERATOR_BIN_DIR)

    if hasattr(_os, "add_dll_directory"):
        handle = _os.add_dll_directory(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
        IMAGE_GENERATOR_DLL_DIRECTORY_HANDLES.append(handle)

    try:
        from pythonnet import load
        try:
            load("netfx")
        except RuntimeError:
            pass
    except ModuleNotFoundError:
        pass


def _prepare_pipeline_mysql_runtime() -> None:
    """准备管线钢 MySql.Data.dll 的运行时和依赖搜索路径。"""
    mysql_data_path = _os.path.join(PIPELINE_IMAGE_GENERATOR_BIN_DIR, "MySql.Data.dll")
    if not _os.path.exists(mysql_data_path):
        raise FileNotFoundError(f"未找到 MySql.Data.dll: {mysql_data_path}")

    if PIPELINE_IMAGE_GENERATOR_BIN_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_IMAGE_GENERATOR_BIN_DIR)

    if hasattr(_os, "add_dll_directory"):
        handle = _os.add_dll_directory(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
        IMAGE_GENERATOR_DLL_DIRECTORY_HANDLES.append(handle)

    try:
        from pythonnet import load
        try:
            load("netfx")
        except RuntimeError:
            pass
    except ModuleNotFoundError:
        pass


def _diagnose_pipeline_image_generator_input(matched_result: dict) -> list[str]:
    """返回管线钢 DLL 绘图入参不满足计算条件的原因列表。"""
    reasons = []
    str_coil = matched_result.get("strCoil")
    arr_body = matched_result.get("arrBody")
    if not str_coil:
        reasons.append("缺少 strCoil")
    if not arr_body:
        reasons.append("缺少 arrBody")
        return reasons

    row = _matched_result_body_to_row(matched_result)
    required_fields = [
        "STEEL_SIGN", "C", "SI", "MN", "P", "S", "N", "NB", "V", "TI", "ALS",
        "CU", "CR", "NI", "CO", "MO", "B", "FDT", "FET", "TEMP_ENTR",
        "SELF_TEMP", "AIM_THICK", "SLAB_THICK", "SLAB_WIDTH", "AIM_WIDTH",
        "SLAB_FURNACE_ENT_TEMP", "PRE_HEAT_TEMP", "PRE_HEAT_TIME", "HEAT_TEMP1",
        "HEAT_TIME1", "HEAT_TEMP2", "HEAT_TIME2", "HEAT_TEMP3", "HEAT_TIME3",
        "SOAK_TEMP", "SOAK_TIME", "FURNACE_EXIT_TEMP", "R_PASS_ACT", "F_PASS_ACT",
    ]
    missing_required = [field for field in required_fields if str(row.get(field, "")).strip() == ""]
    if missing_required:
        reasons.append("管线钢DLL基础字段缺失: " + ", ".join(missing_required))

    invalid_positive = _invalid_positive_fields(
        row,
        ["FDT", "FET", "TEMP_ENTR", "SELF_TEMP", "AIM_THICK"],
    )
    if invalid_positive:
        reasons.append("管线钢DLL基础正数校验失败: " + ", ".join(invalid_positive))
    return reasons


def _generate_pipeline_images_with_dll(matched_result: dict) -> str | None:
    """同步调用管线钢 ANSTEEL_ImageGeneratorLib.dll 绘图，返回 DLL 原始结果；失败只打印日志。"""
    if not isinstance(matched_result, dict):
        print("[管线钢绘图DLL] matched_result 不是 JSON 对象，跳过绘图")
        return None
    if not matched_result.get("strCoil") or not matched_result.get("arrBody"):
        print(
            "[管线钢绘图DLL] matched_result 缺少 strCoil 或 arrBody，跳过绘图；"
            f"strCoil={matched_result.get('strCoil')!r}, "
            f"arrBody长度={len(matched_result.get('arrBody') or [])}"
        )
        return None

    row_for_log = _matched_result_body_to_row(matched_result)
    diagnosis_reasons = _diagnose_pipeline_image_generator_input(matched_result)
    print(
        "[管线钢绘图DLL] 准备调用: "
        f"strCoil={matched_result.get('strCoil')}, "
        f"strSteel={matched_result.get('strSteel')}, "
        f"STEEL_SIGN={row_for_log.get('STEEL_SIGN')}, "
        f"arrBody字段数={len(row_for_log)}"
    )
    if diagnosis_reasons:
        print("[管线钢绘图DLL] 入参预检发现风险: " + "；".join(diagnosis_reasons))

    json_input = json.dumps(matched_result, ensure_ascii=False)

    try:
        with IMAGE_GENERATOR_CALL_LOCK:
            _prepare_pipeline_image_generator_runtime()
            old_cwd = _os.getcwd()
            try:
                # ANSTEEL DLL 内部使用 .\ModelManage 等相对路径，必须切到 HotColdDataBase\bin\Debug。
                _os.chdir(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
                import clr
                clr.AddReference(PIPELINE_IMAGE_GENERATOR_DLL_PATH)
                from ANSTEEL_ImageGeneratorLib import ImageGenerator

                result = str(ImageGenerator.GenerateAllImagesDLL(json_input))
                if result == "true":
                    print(f"[管线钢绘图DLL] 绘图成功: strCoil={matched_result.get('strCoil')}")
                else:
                    print(
                        "[管线钢绘图DLL] DLL未完成仿真计算，返回原因: "
                        f"{result}; strCoil={matched_result.get('strCoil')}, "
                        f"STEEL_SIGN={row_for_log.get('STEEL_SIGN')}, "
                        f"预检风险={'；'.join(diagnosis_reasons) if diagnosis_reasons else '无'}"
                    )
                return result
            finally:
                _os.chdir(old_cwd)
    except Exception as exc:
        print(
            f"[管线钢绘图DLL] 调用异常，未完成仿真计算: {type(exc).__name__}: {exc}; "
            f"strCoil={matched_result.get('strCoil')}, STEEL_SIGN={row_for_log.get('STEEL_SIGN')}"
        )
        return None


def _prepare_pipeline_reheat_image_generator_runtime() -> None:
    """准备管线钢加热智能体 ANSTEEL_ReheatImageGeneratorLib 的运行时和依赖搜索路径。"""
    if not _os.path.exists(PIPELINE_REHEAT_IMAGE_GENERATOR_DLL_PATH):
        raise FileNotFoundError(f"未找到管线钢加热绘图DLL: {PIPELINE_REHEAT_IMAGE_GENERATOR_DLL_PATH}")

    # 加热 DLL 与现有管线钢绘图 DLL 位于同一 bin\Debug 目录，复用相同的依赖搜索路径。
    if PIPELINE_IMAGE_GENERATOR_BIN_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_IMAGE_GENERATOR_BIN_DIR)

    # Windows 原生依赖 DLL 需要通过 add_dll_directory 暴露给 .NET/Python 运行时。
    if hasattr(_os, "add_dll_directory"):
        handle = _os.add_dll_directory(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
        IMAGE_GENERATOR_DLL_DIRECTORY_HANDLES.append(handle)

    # pythonnet 3 可以显式选择 .NET Framework；如果运行时已经初始化，忽略 RuntimeError。
    try:
        from pythonnet import load
        try:
            load("netfx")
        except RuntimeError:
            pass
    except ModuleNotFoundError:
        pass


def _retrieve_pipeline_reheat_rag_context(context: str = "") -> str:
    """进入加热智能体时只检索一次文献依据，后续三轮循环复用同一份结果。"""
    is_wind = "[[WIND_POWER_STEEL_X70_REFERENCE]]" in str(context or "")
    material_label = get_wind_power_material_label(context) if is_wind else "管线钢"
    query = (
        f"{material_label} 加热炉 固溶温度 固溶时间 加热后奥氏体最终晶粒尺寸 晶粒长大 温度均匀性 焊接性相关研究结论"
        if is_wind else "检索所有加热炉内固溶温度固溶时间及加热后的最终晶粒尺寸等研究内容"
    )
    try:
        from hybrid_retriever import hybrid_search

        docs = hybrid_search(
            query,
            k=8,
            db_name="jgyg_Know_db" if is_wind else "gxg_Know_db",
            db_collection="documents",
        )
        if not docs:
            return f"（未检索到{material_label}加热炉固溶温度、固溶时间和晶粒尺寸相关文献，请根据材料学知识保守判断。）"

        def _doc_text(doc) -> str:
            if isinstance(doc, dict):
                source = doc.get("source") or (doc.get("metadata") or {}).get("source") or "unknown"
                content = doc.get("content") or doc.get("page_content") or ""
            else:
                metadata = getattr(doc, "metadata", {}) or {}
                source = getattr(doc, "source", None) or metadata.get("source") or "unknown"
                content = getattr(doc, "content", None) or getattr(doc, "page_content", "") or ""
            return f"[来源: {source}]\n{content}"

        print(f"[管线钢加热智能体] RAG检索命中 {len(docs)} 条文献")
        return "\n\n---\n\n".join(_doc_text(doc) for doc in docs)
    except Exception as exc:
        print(f"[管线钢加热智能体] RAG检索失败: {exc}")
        return "（RAG检索失败，请根据已有上下文、模拟结果和材料学知识保守判断。）"


def _generate_pipeline_reheat_images_with_dll(
    matched_result: dict,
    target_context: str = "",
) -> str | None:
    """同步调用管线钢加热 DLL，入参格式与现有管线钢绘图 DLL 完全一致。"""
    if not isinstance(matched_result, dict):
        print("[管线钢加热智能体] matched_result 不是 JSON 对象，跳过加热 DLL")
        return None
    if not matched_result.get("strCoil") or not matched_result.get("arrBody"):
        print(
            "[管线钢加热智能体] matched_result 缺少 strCoil 或 arrBody，跳过加热 DLL；"
            f"strCoil={matched_result.get('strCoil')!r}, "
            f"arrBody长度={len(matched_result.get('arrBody') or [])}"
        )
        return None

    dll_result, target_grade, reference_grade = _build_pipeline_dll_matched_result(
        matched_result,
        target_context,
    )
    row_for_log = _matched_result_body_to_row(dll_result)
    json_input = json.dumps(dll_result, ensure_ascii=False)
    print(
        "[管线钢加热智能体] DLL模型映射: "
        f"target={target_grade or '未识别'}, reference={reference_grade or '沿用原值'}"
    )
    try:
        with IMAGE_GENERATOR_CALL_LOCK:
            _prepare_pipeline_reheat_image_generator_runtime()
            old_cwd = _os.getcwd()
            try:
                # DLL 内部使用 .\ModelManage 等相对路径，必须切到 HotColdDataBase\bin\Debug。
                _os.chdir(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
                import clr
                clr.AddReference(PIPELINE_REHEAT_IMAGE_GENERATOR_DLL_PATH)
                from ANSTEEL_ReheatImageGeneratorLib import ImageGenerator

                result = str(ImageGenerator.GenerateAllImagesDLL(json_input))
                if result == "true":
                    print(f"[管线钢加热智能体] 加热 DLL 计算成功: strCoil={matched_result.get('strCoil')}")
                else:
                    print(
                        "[管线钢加热智能体] 加热 DLL 未完成计算，返回原因: "
                        f"{result}; strCoil={matched_result.get('strCoil')}, "
                        f"STEEL_SIGN={row_for_log.get('STEEL_SIGN')}"
                    )
                return result
            finally:
                _os.chdir(old_cwd)
    except Exception as exc:
        print(
            f"[管线钢加热智能体] 加热 DLL 调用异常: {type(exc).__name__}: {exc}; "
            f"strCoil={matched_result.get('strCoil')}, STEEL_SIGN={row_for_log.get('STEEL_SIGN')}"
        )
        return None


def _find_first_file_under_dir(root_dir: str, file_name: str) -> str | None:
    """在指定目录下递归查找第一个目标文件，兼容 DLL 子目录名称变化。"""
    if not _os.path.isdir(root_dir):
        return None
    target_name = str(file_name).casefold()
    for current_dir, dir_names, file_names in _os.walk(root_dir):
        actual_name = next(
            (name for name in file_names if str(name).casefold() == target_name),
            None,
        )
        if actual_name:
            return _os.path.join(current_dir, actual_name)
        # Image 目录通常只保存图片，不再继续深挖，减少无意义扫描。
        if _os.path.basename(current_dir).lower() == "image":
            dir_names[:] = []
    return None


def _read_text_file_for_prompt(file_path: str | None, label: str) -> str:
    """读取文本结果；缺失或编码异常时返回明确说明，避免提示词中出现空洞字段。"""
    if not file_path or not _os.path.exists(file_path):
        return f"未读取到 {label}"
    for encoding in ("utf-8", "gbk", "gb2312"):
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read().strip() or f"{label} 文件为空"
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            return f"读取 {label} 失败: {exc}"
    try:
        with open(file_path, "rb") as file:
            return file.read().decode("utf-8", errors="ignore").strip() or f"{label} 文件为空"
    except OSError as exc:
        return f"读取 {label} 失败: {exc}"


def _read_image_base64_for_prompt(file_path: str | None, label: str) -> str:
    """读取 PNG 图片并转成 base64；缺失时返回文本说明，供 Qwen 保守判断。"""
    if not file_path or not _os.path.exists(file_path):
        return f"未读取到 {label}"
    try:
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("ascii")
    except OSError as exc:
        return f"读取 {label} 失败: {exc}"


def _is_valid_image_base64(value: str) -> bool:
    """判断字符串是否是可用的 PNG base64 数据（失败说明文本会被判为 False）。"""
    if not value or len(value) < 100:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=]+", value) is not None


def _build_qwen_vision_content(text_prompt: str, images: list[tuple[str, str]]) -> list[dict]:
    """构造多模态 content：先放文本提示，再把可用图片作为 image_url 传入。

    images 中每项为 (base64_or_text, label)：base64 有效时作为图片传入，
    无效（缺失说明文本）时作为文本提示告诉模型该图片缺失。
    """
    content: list[dict] = [{"type": "text", "text": text_prompt}]
    for b64, label in images:
        if _is_valid_image_base64(b64):
            content.append({"type": "text", "text": f"【{label}】"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        else:
            content.append({"type": "text", "text": f"【{label}】{b64}"})
    return content


PIPELINE_AGENT_JSON_REPAIR_RETRIES = 2
PIPELINE_AGENT_MAX_COMPLETION_TOKENS = 32768
PIPELINE_AGENT_REQUEST_TIMEOUT_SECONDS = 240.0
PIPELINE_PERFORMANCE_FIELDS = ("YS", "TS", "EL", "AKV")
_PIPELINE_PERFORMANCE_BASELINE_CACHE: dict[str, dict[str, str]] = {}
_PIPELINE_PERFORMANCE_SPEC_CACHE: dict[str, dict] = {}


def _pipeline_performance_cache_key(matched_result: dict) -> str:
    """用会话键保存后置微调形成的合格性能基线，板坯号仅作为兼容兜底。"""
    return str(
        matched_result.get("session_key")
        or matched_result.get("strCoil")
        or ""
    ).strip()


def _pipeline_performance_values(matched_result: dict) -> dict[str, str]:
    """从单键字典列表提取四项力学性能。"""
    row = _matched_result_body_to_row(matched_result)
    return {
        field: str(row.get(field, ""))
        for field in PIPELINE_PERFORMANCE_FIELDS
    }


def _cache_pipeline_performance_baseline(
    matched_result: dict,
    spec_result: dict,
) -> None:
    """缓存后置微调后的合格性能值和本轮性能标准，供三个工艺智能体共同使用。"""
    cache_key = _pipeline_performance_cache_key(matched_result)
    if not cache_key:
        return
    if len(_PIPELINE_PERFORMANCE_BASELINE_CACHE) >= 2048:
        oldest_key = next(iter(_PIPELINE_PERFORMANCE_BASELINE_CACHE))
        _PIPELINE_PERFORMANCE_BASELINE_CACHE.pop(oldest_key, None)
        _PIPELINE_PERFORMANCE_SPEC_CACHE.pop(oldest_key, None)
    _PIPELINE_PERFORMANCE_BASELINE_CACHE[cache_key] = _pipeline_performance_values(matched_result)
    _PIPELINE_PERFORMANCE_SPEC_CACHE[cache_key] = copy.deepcopy(spec_result)


def _pipeline_agent_performance_context(matched_result: dict) -> str:
    """生成工艺智能体提示词使用的性能标准和合格基线 JSON。"""
    cache_key = _pipeline_performance_cache_key(matched_result)
    spec_result = _PIPELINE_PERFORMANCE_SPEC_CACHE.get(cache_key, {})
    baseline = _PIPELINE_PERFORMANCE_BASELINE_CACHE.get(
        cache_key,
        _pipeline_performance_values(matched_result),
    )
    payload = {
        field: {
            "min": spec_result.get(f"{field}_min"),
            "max": spec_result.get(f"{field}_max"),
            "baseline": baseline.get(field, ""),
        }
        for field in PIPELINE_PERFORMANCE_FIELDS
    }
    return json.dumps(payload, ensure_ascii=False)


def _pipeline_agent_attempted_performance_change(original: dict, candidate: dict) -> bool:
    """四项性能均为数值且至少一项变化时，才视为完成同步性能预测。"""
    original_values = _pipeline_performance_values(original)
    candidate_values = _pipeline_performance_values(candidate)
    if any(_to_float(candidate_values.get(field)) is None for field in PIPELINE_PERFORMANCE_FIELDS):
        return False
    return any(
        str(candidate_values.get(field, "")).strip()
        != str(original_values.get(field, "")).strip()
        for field in PIPELINE_PERFORMANCE_FIELDS
    )


def _resolve_pipeline_agent_performance_value(
    original: dict,
    field_name: str,
    candidate_value,
):
    """采纳范围内的性能预测；越界时回退后置微调形成的合格基线。"""
    cache_key = _pipeline_performance_cache_key(original)
    spec_result = _PIPELINE_PERFORMANCE_SPEC_CACHE.get(cache_key, {})
    baseline = _PIPELINE_PERFORMANCE_BASELINE_CACHE.get(cache_key, {})
    if _pipeline_value_within_spec_bounds(field_name, candidate_value, spec_result):
        formatted = _format_pipeline_refined_value(field_name, candidate_value)
        if (
            formatted is not None
            and _pipeline_formatted_value_within_spec_bounds(field_name, formatted, spec_result)
        ):
            return formatted
    return baseline.get(field_name, _pipeline_performance_values(original).get(field_name, ""))


def _invoke_pipeline_qwen_json(
    system_prompt: str,
    user_prompt: str,
    images: list[tuple[str, str]],
    stage_label: str,
) -> dict:
    """调用工艺智能体判断模型，并返回正文、解析结果和完整诊断信息。"""
    content = _build_qwen_vision_content(user_prompt, images)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]
    raw = None
    call_error = ""
    try:
        raw = official_qwen_sync.invoke(
            messages,
            timeout=PIPELINE_AGENT_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
            response_format={"type": "json_object"},
            max_completion_tokens=PIPELINE_AGENT_MAX_COMPLETION_TOKENS,
            extra_body={"enable_thinking": False},
        )
    except Exception as exc:
        call_error = f"{type(exc).__name__}: {exc}"
        print(f"[{stage_label}] Qwen 模型调用失败: {call_error}")

    if raw is None:
        return {
            "candidate": None,
            "judgement": {},
            "reasoning": "",
            "text": "",
            "diagnostic": {
                "error": call_error or "Qwen 判断模型未返回响应",
                "json_valid": False,
            },
        }

    reasoning = raw.reasoning_content or ""
    text = str(raw.content or "")
    parsed_response = _parse_json_object(text)
    candidate, judgement = _extract_qwen_agent_response(parsed_response)
    metadata = dict(raw.raw_metadata or {})
    arr_body = candidate.get("arrBody") if isinstance(candidate, dict) else None
    diagnostic = {
        "model": metadata.get("model"),
        "finish_reason": metadata.get("finish_reason"),
        "usage": metadata.get("usage"),
        "content_chars": len(text),
        "json_valid": isinstance(parsed_response, dict),
        "arrBody_len": len(arr_body) if isinstance(arr_body, list) else None,
        "error": "" if isinstance(parsed_response, dict) else "模型正文不是合法JSON对象",
    }
    print(
        f"[{stage_label}] 模型响应诊断: model={diagnostic['model']}, "
        f"finish_reason={diagnostic['finish_reason']}, usage={diagnostic['usage']}, "
        f"content_chars={diagnostic['content_chars']}, json_valid={diagnostic['json_valid']}, "
        f"arrBody_len={diagnostic['arrBody_len']}"
    )
    return {
        "candidate": candidate,
        "judgement": judgement,
        "reasoning": reasoning,
        "text": text,
        "diagnostic": diagnostic,
    }


def _describe_pipeline_agent_structure_error(original: dict, candidate) -> str:
    """把模型结构错误转换成下一次同轮修复调用可直接理解的中文说明。"""
    if not isinstance(candidate, dict):
        return "返回正文无法解析为包含 matched_result 的合法 JSON 对象"
    if list(candidate.keys()) != list(original.keys()):
        return (
            "matched_result 顶层字段或顺序不一致；"
            f"期望={list(original.keys())}，实际={list(candidate.keys())}"
        )
    original_body = original.get("arrBody")
    candidate_body = candidate.get("arrBody")
    if not isinstance(candidate_body, list):
        return "matched_result.arrBody 不是数组"
    expected_length = len(original_body) if isinstance(original_body, list) else 0
    if len(candidate_body) != expected_length:
        return (
            "matched_result.arrBody 长度不一致；"
            f"期望={expected_length}，实际={len(candidate_body)}"
        )
    for index, (original_item, candidate_item) in enumerate(
        zip(original_body, candidate_body),
        start=1,
    ):
        original_key = _get_arrbody_key(original_item)
        candidate_key = _get_arrbody_key(candidate_item)
        if not original_key or original_key != candidate_key:
            return (
                f"matched_result.arrBody 第{index}项字段名或顺序不一致；"
                f"期望={original_key}，实际={candidate_key}"
            )
    return "返回结果未通过当前工艺智能体的字段白名单或结构校验"




def _pipeline_agent_has_meaningful_process_change(before: dict, after: dict) -> bool:
    """判断工艺字段是否发生工程值变化，性能同步变化不计作新一轮仿真依据。"""
    before_body = before.get("arrBody") if isinstance(before, dict) else None
    after_body = after.get("arrBody") if isinstance(after, dict) else None
    if not isinstance(before_body, list) or not isinstance(after_body, list):
        return False
    for before_item, after_item in zip(before_body, after_body):
        field_name = str(_get_arrbody_key(before_item) or "").upper()
        if field_name in PIPELINE_PERFORMANCE_FIELDS:
            continue
        before_value = _get_arrbody_value(before_item)
        after_value = _get_arrbody_value(after_item)
        before_number = _to_float(before_value)
        after_number = _to_float(after_value)
        if before_number is not None and after_number is not None:
            if abs(before_number - after_number) > 1e-9:
                return True
        elif str(before_value).strip() != str(after_value).strip():
            return True
    return False


def _pipeline_stage_simulation_input_changed(before: dict, after: dict, stage: str) -> bool:
    """判断智能体最终结果是否改变了对应 DLL 的有效输入。"""
    before_row = _matched_result_body_to_row(before)
    after_row = _matched_result_body_to_row(after)
    if stage == "reheat":
        fields = {
            "HEAT_TEMP1", "HEAT_TEMP2", "HEAT_TEMP3",
            "SOAK_TEMP", "SOAK_TIME", "FURNACE_EXIT_TEMP",
        }
    elif stage == "roll":
        fields = {
            "FET", "FDT", "R_PASS_ACT", "F_PASS_ACT",
            "WIDTH_ROLL_START_REMARK", "WIDTH_ROLL_END_REMARK",
        }
        fields.update(
            field_name
            for field_name in set(before_row) | set(after_row)
            if re.fullmatch(
                r"N(?:[1-9]|[12]\d|30)_(?:DH_CAL|DT_CAL|DW_CAL|FORCE|SPD|ENTR_DATE)",
                field_name,
            )
        )
    elif stage == "cooling":
        fields = {"TIME_ENTR", "TEMP_ENTR", "SELF_TEMP", "YS", "TS", "EL", "AKV"}
    else:
        return False

    for field_name in fields:
        before_value = before_row.get(field_name)
        after_value = after_row.get(field_name)
        before_number = _to_float(before_value)
        after_number = _to_float(after_value)
        if before_number is not None and after_number is not None:
            if abs(before_number - after_number) > 1e-9:
                return True
        elif str(before_value or "").strip() != str(after_value or "").strip():
            return True
    return False


def _remember_pipeline_agent_accepted_response(
    reasoning_key: str | None,
    invocation: dict,
    matched_result: dict,
) -> None:
    """每个仿真轮只缓存最终通过校验的响应，避免把格式重试显示成新工艺轮次。"""
    if not reasoning_key:
        return
    reasoning = _sanitize_pipeline_agent_reference_text(
        str(invocation.get("reasoning") or ""),
        matched_result,
    )
    text = str(invocation.get("text") or "")
    judgement = invocation.get("judgement")
    if isinstance(judgement, dict):
        judgement = {
            key: _sanitize_pipeline_agent_reference_text(str(value or ""), matched_result)
            for key, value in judgement.items()
        }
    _LLM_REASONING_CONTENT_CACHE[reasoning_key] = reasoning
    _LLM_JUDGEMENT_CONTENT_CACHE.setdefault(reasoning_key, []).append(text)
    _LLM_JUDGEMENT_REASONING_CACHE.setdefault(reasoning_key, []).append(reasoning)
    _LLM_JUDGEMENT_VISIBLE_CACHE.setdefault(reasoning_key, []).append(
        judgement if isinstance(judgement, dict) else {}
    )
    print(f"[reasoning_content] key={reasoning_key}, reasoning_len={len(reasoning)}")


def _resolve_pipeline_agent_round(
    *,
    invoke_func,
    sanitize_func,
    current_result: dict,
    base_prompt: str,
    images: list[tuple[str, str]],
    reasoning_key: str | None,
    progress_callback,
    stage: str,
    stage_label: str,
    simulation_attempt: int,
) -> dict | None:
    """在同一次 DLL 仿真结果上完成初次判断及最多两次结构修复调用。"""
    last_error = ""
    calls_made = 0
    terminal_timeout = False
    total_calls = PIPELINE_AGENT_JSON_REPAIR_RETRIES + 1
    for response_attempt in range(1, total_calls + 1):
        calls_made = response_attempt
        prompt = (
            base_prompt
            if response_attempt == 1
            else _build_pipeline_agent_repair_prompt(
                base_prompt,
                last_error,
                response_attempt - 1,
            )
        )
        if response_attempt > 1:
            print(
                f"[{stage_label}] 第 {simulation_attempt} 轮复用既有 DLL 结果，"
                f"开始第 {response_attempt - 1}/{PIPELINE_AGENT_JSON_REPAIR_RETRIES} 次模型格式修复"
            )

        invocation = invoke_func(prompt, images)
        candidate = invocation.get("candidate")
        if not isinstance(candidate, dict):
            diagnostic = invocation.get("diagnostic") or {}
            last_error = str(
                diagnostic.get("error")
                or "返回正文无法解析为合法 JSON"
            )
            normalized_error = last_error.lower()
            terminal_timeout = any(marker in normalized_error for marker in (
                "timeout",
                "timed out",
                "apitimeouterror",
                "readtimeout",
            ))
        else:
            sanitize_result = sanitize_func(current_result, candidate)
            sanitize_error = ""
            if (
                isinstance(sanitize_result, tuple)
                and len(sanitize_result) == 2
            ):
                sanitized, sanitize_error = sanitize_result
            else:
                sanitized = sanitize_result
            if sanitized is None:
                last_error = str(sanitize_error or "").strip() or (
                    _describe_pipeline_agent_structure_error(
                        current_result,
                        candidate,
                    )
                )
            elif (
                sanitized.get("isState") is False
                and not _pipeline_agent_has_meaningful_process_change(
                    current_result,
                    sanitized,
                )
            ):
                last_error = (
                    "isState=false，但允许修改的工艺字段没有任何工程值变化；"
                    "相同参数不能进入下一轮 DLL 仿真"
                )
            else:
                _remember_pipeline_agent_accepted_response(
                    reasoning_key,
                    invocation,
                    current_result,
                )
                return sanitized

        print(
            f"[{stage_label}] 第 {simulation_attempt} 轮第 {response_attempt}/{total_calls} 次"
            f"模型响应未采纳: {last_error}"
        )
        if terminal_timeout:
            print(
                f"[{stage_label}] 判断模型单次请求超过 "
                f"{PIPELINE_AGENT_REQUEST_TIMEOUT_SECONDS:.0f} 秒，停止本轮格式修复重试"
            )
            break

    if terminal_timeout:
        failure_message = (
            f"第 {simulation_attempt} 轮判断模型请求超过 "
            f"{PIPELINE_AGENT_REQUEST_TIMEOUT_SECONDS:.0f} 秒，已停止等待且不再自动重试："
            f"{last_error}。已保留进入本轮前的 matched_result，未重复运行 DLL。"
        )
    else:
        failure_message = (
            f"第 {simulation_attempt} 轮模型结果连续 {calls_made} 次未通过结构或调整校验："
            f"{last_error}。已保留进入本轮前的 matched_result，未重复运行 DLL。"
        )
    print(f"[{stage_label}] {failure_message}")
    if progress_callback:
        try:
            progress_callback({
                "event_type": "agent_error",
                "attempt": simulation_attempt,
                "stage": stage,
                "message": failure_message,
            })
        except Exception as exc:
            print(f"[{stage_label}] 前端失败通知发送异常: {exc}")
    return None


_PIPELINE_AGENT_DLL_ONLY_IDENTITY_FIELDS = {"STEEL_SIGN", "SLAB_ID"}


_PIPELINE_DLL_GRADE_ALIASES = {
    "L290": "X42",
    "L320": "X46",
    "L360": "X52",
    "L390": "X56",
    "L415": "X60",
    "L450": "X65",
    "L485": "X70",
    "L555": "X80",
    "L625": "X90",
    "L690": "X100",
    "L830": "X120",
}


def _extract_pipeline_target_grade(text: str) -> str | None:
    """从用户需求或上下文提取目标管线钢牌号，并统一为 X 系列牌号。"""
    merged = str(text or "").upper()
    for grade in ("X120", "X100", "X90", "X80", "X70", "X65", "X60", "X56", "X52", "X46", "X42"):
        if re.search(rf"(?<![A-Z0-9]){grade}(?:[A-Z-]*)?(?![A-Z0-9])", merged):
            return grade
    for alias, grade in _PIPELINE_DLL_GRADE_ALIASES.items():
        if re.search(rf"(?<![A-Z0-9]){alias}(?:[A-Z-]*)?(?![A-Z0-9])", merged):
            return grade
    return None


def _select_pipeline_dll_reference_grade(target_grade: str | None) -> str | None:
    """按现有物理冶金模型能力，把目标牌号映射到 X65/X70/X80NG。"""
    normalized = _extract_pipeline_target_grade(str(target_grade or ""))
    if normalized in {"X42", "X46", "X52", "X56", "X60", "X65"}:
        # return "X65"
        return "X70"
    if normalized == "X70":
        return "X70"
    if normalized == "X80":
        # return "X80NG"
        return "X70"
    if normalized in {"X90", "X100", "X120"}:
        return "X70"
    return None


def _resolve_pipeline_dll_grade(
    matched_result: dict,
    target_context: str = "",
) -> tuple[str | None, str | None]:
    """优先按当前用户目标选择DLL模型；没有明确目标时沿用历史实绩牌号。"""
    # 风电塔筒钢板当前没有独立 DLL。风电分支统一将 X70 作为趋势参考模型，
    # 且绝不读取历史 strSteel/STEEL_SIGN 来决定模型目录。
    if "[[WIND_POWER_STEEL_X70_REFERENCE]]" in str(target_context or ""):
        return "风电用钢", "X70"
    target_grade = _extract_pipeline_target_grade(target_context)
    if target_grade is None:
        target_grade = _extract_pipeline_target_grade(matched_result.get("strSteel"))
    if target_grade is None:
        for item in matched_result.get("arrBody") or []:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            key, value = next(iter(item.items()))
            if str(key).upper() == "STEEL_SIGN":
                target_grade = _extract_pipeline_target_grade(value)
                break
    return target_grade, _select_pipeline_dll_reference_grade(target_grade)


def _build_pipeline_dll_matched_result(
    matched_result: dict,
    target_context: str = "",
) -> tuple[dict, str | None, str | None]:
    """构建仅供DLL使用的副本；模型路由字段变化不会污染最终设计结果。"""
    dll_result = copy.deepcopy(matched_result)
    target_grade, reference_grade = _resolve_pipeline_dll_grade(
        matched_result,
        target_context,
    )
    if reference_grade is None:
        return dll_result, target_grade, None

    dll_result["strSteel"] = reference_grade
    dll_body = []
    for item in dll_result.get("arrBody") or []:
        if not isinstance(item, dict) or len(item) != 1:
            dll_body.append(item)
            continue
        key, value = next(iter(item.items()))
        dll_body.append(
            {key: reference_grade}
            if str(key).upper() == "STEEL_SIGN"
            else {key: value}
        )
    dll_result["arrBody"] = dll_body
    return dll_result, target_grade, reference_grade


def _pipeline_agent_identity_terms(matched_result: dict) -> tuple[set[str], set[str]]:
    """提取仅供 DLL 使用的身份值及其牌号变体，供 LLM 输入和可见正文脱敏。

    strSteel/STEEL_SIGN 决定现有 DLL 的钢级运行目录，strCoil/SLAB_ID 决定
    计算结果目录；它们都不是工艺判断依据，不能进入三个智能体的判断语境。
    """
    identity_values: set[str] = set()
    steel_variants: set[str] = set()

    for top_key in ("strCoil", "strSteel"):
        value = str(matched_result.get(top_key) or "").strip()
        if value:
            identity_values.add(value)
            if top_key == "strSteel":
                steel_variants.add(value)

    for item in matched_result.get("arrBody") or []:
        key = _get_arrbody_key(item)
        if not key or key.upper() not in _PIPELINE_AGENT_DLL_ONLY_IDENTITY_FIELDS:
            continue
        value = str(_get_arrbody_value(item) or "").strip()
        if value:
            identity_values.add(value)
            if key.upper() == "STEEL_SIGN":
                steel_variants.add(value)

    # 数据库牌号常带生产后缀，例如 X65MS-1。模型可能把它概括成 X65MS
    # 或 X65，因此一并登记为不可展示的历史牌号变体。
    for value in list(steel_variants):
        upper_value = value.upper()
        grade_match = re.search(r"(?<![A-Z0-9])(X\d{2,3}[A-Z]*)(?![A-Z0-9])", upper_value)
        if grade_match:
            full_grade = grade_match.group(1)
            steel_variants.add(full_grade)
            base_match = re.match(r"X\d{2,3}", full_grade)
            if base_match:
                steel_variants.add(base_match.group(0))

    return identity_values, steel_variants


def _sanitize_pipeline_agent_reference_text(text: str, matched_result: dict) -> str:
    """从智能体上下文/RAG/可见判断中移除 DLL 身份信息和历史牌号。"""
    sanitized = str(text or "")
    identity_values, steel_variants = _pipeline_agent_identity_terms(matched_result)

    # 先替换较长内容，避免 X65MS-1 先被 X65 截断后残留后缀。
    for value in sorted(identity_values - steel_variants, key=len, reverse=True):
        sanitized = re.sub(re.escape(value), "", sanitized, flags=re.IGNORECASE)
    for value in sorted(steel_variants, key=len, reverse=True):
        sanitized = re.sub(re.escape(value), "当前目标热轧钢板", sanitized, flags=re.IGNORECASE)

    # 即使模型主动复述字段名，也不允许这些 DLL 身份字段进入前端思维链。
    sanitized = re.sub(
        r"(?i)\b(?:STEEL_SIGN|SLAB_ID|strSteel|strCoil)\b",
        "",
        sanitized,
    )
    return sanitized


WIND_POWER_DLL_CONTEXT_MARKER = "[[WIND_POWER_STEEL_X70_REFERENCE]]"


def _is_wind_power_context(text: str) -> bool:
    """判断上下文是否属于风电分支；DLL 映射标记只允许在后端内部流转。"""
    value = str(text or "")
    return (
        WIND_POWER_DLL_CONTEXT_MARKER in value
        or WIND_POWER_PROMPT_CONTEXT_TAG in value
    )


def _filter_wind_power_session_context(text: str) -> str:
    """隔离历史管线钢会话，避免旧牌号和油气场景污染风电设计提示词。"""
    retained_lines = []
    legacy_pattern = re.compile(
        r"管线|油气|输送管|API\s*5L|\bL\d{3}[A-Z]*\b|\bX(?:42|46|52|56|60|65|70|80|90|100|120)[A-Z0-9-]*\b",
        flags=re.IGNORECASE,
    )
    for line in str(text or "").splitlines():
        if not legacy_pattern.search(line):
            retained_lines.append(line)
    filtered = "\n".join(retained_lines).strip()
    return filtered or "（已隔离与当前风电设计无关的历史会话内容）"


def _wind_power_agent_prompt(text: str) -> str:
    """将复用的智能体提示词转换为纯风电语境，不泄露 DLL 参考钢级。"""
    value = str(text or "")
    if not _is_wind_power_context(value):
        return value

    material_label = get_wind_power_material_label(value)
    replacements = {
        WIND_POWER_DLL_CONTEXT_MARKER: WIND_POWER_PROMPT_CONTEXT_TAG,
        "管线钢": material_label,
        "管线用钢": material_label,
        "油气输送管": "风电塔筒",
        "油气管线": "风电塔筒",
        "输送管线": "风电塔筒",
        "API 管线": "GB/T 1591",
        "API管线": "GB/T 1591",
        "X系列": "目标钢级系列",
        "L系列": "目标钢级系列",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)

    # 复用提示词中可能遗留的管线钢级举例不属于风电设计依据，统一移除。
    value = re.sub(
        r"\b(?:X(?:42|46|52|56|60|65|70|80|90|100|120)[A-Z0-9-]*|L\d{3}[A-Z]*)\b",
        "目标钢级",
        value,
        flags=re.IGNORECASE,
    )
    return value


def _build_pipeline_agent_llm_view(matched_result: dict) -> dict:
    """构建判断模型专用副本；字段结构不变，仅清空 DLL 身份字段的值。"""
    llm_view = copy.deepcopy(matched_result)
    if "strCoil" in llm_view:
        llm_view["strCoil"] = ""
    if "strSteel" in llm_view:
        llm_view["strSteel"] = ""

    masked_body = []
    for item in llm_view.get("arrBody") or []:
        key = _get_arrbody_key(item)
        if key and key.upper() in _PIPELINE_AGENT_DLL_ONLY_IDENTITY_FIELDS:
            masked_body.append({key: ""})
        else:
            masked_body.append(item)
    llm_view["arrBody"] = masked_body
    return llm_view


def _prepare_pipeline_agent_llm_prompt_data(
    context: str,
    rag_context: str,
    matched_result: dict,
) -> tuple[str, str, dict]:
    """统一准备三个工艺智能体的无身份判断输入，避免各阶段出现实现偏差。"""
    return (
        _sanitize_pipeline_agent_reference_text(context, matched_result),
        _sanitize_pipeline_agent_reference_text(rag_context, matched_result),
        _build_pipeline_agent_llm_view(matched_result),
    )


def _build_pipeline_reheat_agent_user_prompt(
    context: str,
    rag_context: str,
    matched_result: dict,
    tas_text: str,
) -> str:
    """构建每轮传给 Qwen 的动态用户提示词，当前轮 matched_result 和模拟结果会更新。

    均热温度.png、晶粒长大.png、晶粒尺寸分布.png 作为多模态图片随消息一起传入。
    """
    context, rag_context, llm_matched_result = _prepare_pipeline_agent_llm_prompt_data(
        context,
        rag_context,
        matched_result,
    )
    matched_result_json = json.dumps(llm_matched_result, ensure_ascii=False)
    return build_pipeline_reheat_agent_user_prompt_text(
        context, rag_context, matched_result_json, tas_text
    )




def _coerce_reheat_is_state(value):
    """把模型返回的 true/false 字符串兼容转换成布尔值，无法识别时返回原值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return value


def _sanitize_pipeline_reheat_agent_result(original: dict, candidate: dict) -> dict | None:
    """校验加热结果；只采纳加热工艺变化，性能字段始终恢复原值。"""
    if not isinstance(original, dict) or not isinstance(candidate, dict):
        return None
    if list(candidate.keys()) != list(original.keys()):
        print("[管线钢加热智能体] Qwen返回顶层键不一致，放弃本轮结果")
        return None

    original_body = original.get("arrBody")
    candidate_body = candidate.get("arrBody")
    if not isinstance(original_body, list) or not isinstance(candidate_body, list):
        return None
    if len(original_body) != len(candidate_body):
        print("[管线钢加热智能体] Qwen返回 arrBody 长度不一致，放弃本轮结果")
        return None

    sanitized = copy.deepcopy(original)
    candidate_state = _coerce_reheat_is_state(candidate.get("isState", original.get("isState")))
    sanitized["isState"] = candidate_state if isinstance(candidate_state, bool) else original.get("isState")
    candidate_soak_temp = None
    for original_item, candidate_item in zip(original_body, candidate_body):
        original_key = _get_arrbody_key(original_item)
        candidate_key = _get_arrbody_key(candidate_item)
        if not original_key or original_key != candidate_key:
            print("[管线钢加热智能体] Qwen返回 arrBody 字段顺序或字段名不一致，放弃本轮结果")
            return None
        if original_key.upper() == "SOAK_TEMP":
            candidate_soak_temp = _get_arrbody_value(candidate_item)

    sanitized_body = []
    for original_item, candidate_item in zip(original_body, candidate_body):
        original_key = _get_arrbody_key(original_item)
        original_value = _get_arrbody_value(original_item)
        candidate_value = _get_arrbody_value(candidate_item)
        field_name = original_key.upper()
        if field_name in {"SOAK_TEMP", "SOAK_TIME", "HEAT_TEMP1", "HEAT_TEMP2", "HEAT_TEMP3"}:
            sanitized_body.append({original_key: candidate_value})
        elif field_name == "FURNACE_EXIT_TEMP" and candidate_soak_temp is not None:
            synchronized_value = (
                str(candidate_soak_temp)
                if isinstance(original_value, str)
                else candidate_soak_temp
            )
            sanitized_body.append({original_key: synchronized_value})
        else:
            sanitized_body.append({original_key: original_value})

    sanitized["arrBody"] = sanitized_body
    original_row = _matched_result_body_to_row(original)
    sanitized_row = _matched_result_body_to_row(sanitized)
    original_soak_temp = _to_float(original_row.get("SOAK_TEMP"))
    sanitized_soak_temp = _to_float(sanitized_row.get("SOAK_TEMP"))
    soak_temp_changed = (
        original_soak_temp is not None
        and sanitized_soak_temp is not None
        and abs(original_soak_temp - sanitized_soak_temp) > 1e-9
    ) or (
        (original_soak_temp is None or sanitized_soak_temp is None)
        and str(original_row.get("SOAK_TEMP", "")).strip()
        != str(sanitized_row.get("SOAK_TEMP", "")).strip()
    )
    if soak_temp_changed:
        heating_temperature_names = ("HEAT_TEMP1", "HEAT_TEMP2", "HEAT_TEMP3", "SOAK_TEMP")
        heating_temperatures = [
            _to_float(sanitized_row.get(field_name))
            for field_name in heating_temperature_names
        ]
        if all(value is not None for value in heating_temperatures):
            # 保留模型确定的 SOAK_TEMP，从 HEAT_TEMP3 向前逐级修正。相邻温差
            # 超过20℃时仅削减超出部分，不要求温度单调递增，也不拒绝本轮结果。
            adjusted_temperatures = list(heating_temperatures)
            for index in range(len(adjusted_temperatures) - 2, -1, -1):
                current = adjusted_temperatures[index]
                next_temperature = adjusted_temperatures[index + 1]
                difference = current - next_temperature
                if difference > 20.0:
                    adjusted_temperatures[index] = next_temperature + 20.0
                elif difference < -20.0:
                    adjusted_temperatures[index] = next_temperature - 20.0

            if adjusted_temperatures != heating_temperatures:
                corrected_values = {
                    field_name: _format_pipeline_refined_value(field_name, value)
                    for field_name, value in zip(
                        heating_temperature_names,
                        adjusted_temperatures,
                    )
                }
                sanitized["arrBody"] = [
                    {
                        _get_arrbody_key(item): corrected_values.get(
                            str(_get_arrbody_key(item) or "").upper(),
                            _get_arrbody_value(item),
                        )
                    }
                    for item in sanitized["arrBody"]
                ]
                print(
                    "[管线钢加热智能体] 已自动修正三级加热温度，使相邻温度绝对差不超过20℃: "
                    f"{dict(zip(heating_temperature_names, adjusted_temperatures))}"
                )
    return sanitized


def _invoke_qwen_reheat_agent(user_prompt: str, images: list[tuple[str, str]]) -> dict:
    """调用加热判断模型；结构修复和结果缓存由外层同轮重试逻辑负责。"""
    print(f"[管线钢加热智能体] 开始调用判断模型，prompt长度={len(user_prompt)}, 图片数={len(images)}")
    return _invoke_pipeline_qwen_json(
        PIPELINE_REHEAT_AGENT_SYSTEM_PROMPT.replace("管线钢", get_wind_power_material_label(user_prompt))
        if WIND_POWER_PROMPT_CONTEXT_TAG in user_prompt else PIPELINE_REHEAT_AGENT_SYSTEM_PROMPT,
        user_prompt,
        images,
        "管线钢加热智能体",
    )


def _collect_pipeline_reheat_simulation_context(matched_result: dict) -> tuple[str, str, str, str]:
    """读取全固溶温度 Tas.txt、均热温度场和两张加热阶段晶粒图片。

    全固溶温度固定优先取：
    ModelManage/{卷号}/Physical Metallurgy Results/Tas.txt。
    只有该标准输出路径不存在时，才递归查找同卷号目录下的 Tas.txt 作为兼容兜底。
    """
    coil_id = str(matched_result.get("strCoil", "")).strip()
    coil_dir = _os.path.join(PIPELINE_IMAGE_GENERATOR_BIN_DIR, "ModelManage", coil_id)
    full_solution_tas_path = _os.path.join(coil_dir, "Physical Metallurgy Results", "Tas.txt")
    if _os.path.isfile(full_solution_tas_path):
        tas_path = full_solution_tas_path
        print(f"[管线钢加热智能体] 读取全固溶温度文件: {tas_path}")
    else:
        tas_path = _find_first_file_under_dir(coil_dir, "Tas.txt")
        print(f"[管线钢加热智能体] 未找到标准全固溶温度文件，使用递归 Tas.txt 兜底: {tas_path or '未找到'}")

    grain_growth_path = _find_first_file_under_dir(coil_dir, "晶粒长大.png")
    grain_size_distribution_path = _find_first_file_under_dir(coil_dir, "晶粒尺寸分布.png")
    soaking_temperature_path = _find_first_file_under_dir(coil_dir, "均热温度.png")

    tas_text = _read_text_file_for_prompt(tas_path, "全固溶温度 Tas.txt")
    soaking_temperature_base64 = _read_image_base64_for_prompt(soaking_temperature_path, "均热温度.png")
    grain_growth_base64 = _read_image_base64_for_prompt(grain_growth_path, "晶粒长大.png")
    grain_size_distribution_base64 = _read_image_base64_for_prompt(
        grain_size_distribution_path,
        "晶粒尺寸分布.png",
    )
    return tas_text, soaking_temperature_base64, grain_growth_base64, grain_size_distribution_base64


def _build_process_agent_dependencies() -> ProcessAgentDependencies:
    """汇集三段工艺智能体依赖；具体循环流程位于 pipeline_agents.py。"""
    return ProcessAgentDependencies(
        resolve_agent_round=_resolve_pipeline_agent_round,
        stage_input_changed=_pipeline_stage_simulation_input_changed,
        input_cache=_LLM_JUDGEMENT_INPUT_CACHE,
        reasoning_cache=_LLM_JUDGEMENT_REASONING_CACHE,
        visible_cache=_LLM_JUDGEMENT_VISIBLE_CACHE,
        wind_power_prompt=_wind_power_agent_prompt,
        retrieve_reheat_rag=_retrieve_pipeline_reheat_rag_context,
        generate_reheat_images=_generate_pipeline_reheat_images_with_dll,
        collect_reheat_context=_collect_pipeline_reheat_simulation_context,
        build_reheat_prompt=_build_pipeline_reheat_agent_user_prompt,
        invoke_reheat=_invoke_qwen_reheat_agent,
        sanitize_reheat=_sanitize_pipeline_reheat_agent_result,
        retrieve_roll_rag=_retrieve_pipeline_roll_rag_context,
        generate_roll_images=_generate_pipeline_roll_images_with_dll,
        collect_roll_context=_collect_pipeline_roll_simulation_context,
        build_roll_prompt=_build_pipeline_roll_agent_user_prompt,
        invoke_roll=_invoke_qwen_roll_agent,
        sanitize_roll=_sanitize_pipeline_roll_agent_result,
        require_valid_roll_result=_require_valid_pipeline_roll_result,
        retrieve_cooling_rag=_retrieve_pipeline_cooling_rag_context,
        generate_cooling_images=_generate_pipeline_cooling_images_with_dll,
        collect_cooling_context=_collect_pipeline_cooling_simulation_context,
        build_cooling_prompt=_build_pipeline_cooling_agent_user_prompt,
        invoke_cooling=_invoke_qwen_cooling_agent,
        sanitize_cooling=_sanitize_pipeline_cooling_agent_result,
        user_requests_high_self_temp=_pipeline_user_explicitly_requests_high_self_temp,
        set_arrbody_field=_set_pipeline_arrbody_field,
        body_to_row=_matched_result_body_to_row,
        to_float=_to_float,
        stabilize_cooling_timing=_stabilize_pipeline_cooling_timing,
        require_valid_cooling_timing=_require_valid_pipeline_cooling_timing,
    )


def _refine_pipeline_reheat_process_with_agent(
    matched_result: dict,
    context: str,
    reasoning_key_prefix: str | None = None,
    progress_callback=None,
) -> dict:
    """兼容原调用名；加热智能体业务已迁移到 pipeline_agents.py。"""
    return refine_reheat_process(
        matched_result,
        context,
        reasoning_key_prefix,
        progress_callback=progress_callback,
        dependencies=_build_process_agent_dependencies(),
    )


def _prepare_pipeline_roll_image_generator_runtime() -> None:
    """准备管线钢轧制智能体 ANSTEEL_RollImageGeneratorLib 的运行时和依赖搜索路径。"""
    if not _os.path.exists(PIPELINE_ROLL_IMAGE_GENERATOR_DLL_PATH):
        raise FileNotFoundError(f"未找到管线钢轧制绘图DLL: {PIPELINE_ROLL_IMAGE_GENERATOR_DLL_PATH}")

    # 轧制 DLL 与管线钢主绘图 DLL 位于同一 bin\Debug 目录，依赖搜索路径保持一致。
    if PIPELINE_IMAGE_GENERATOR_BIN_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_IMAGE_GENERATOR_BIN_DIR)

    # .NET 程序集及其原生依赖需要通过 add_dll_directory 暴露给运行时。
    if hasattr(_os, "add_dll_directory"):
        handle = _os.add_dll_directory(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
        IMAGE_GENERATOR_DLL_DIRECTORY_HANDLES.append(handle)

    # pythonnet 3 可显式选择 .NET Framework；运行时已初始化时忽略 RuntimeError。
    try:
        from pythonnet import load
        try:
            load("netfx")
        except RuntimeError:
            pass
    except ModuleNotFoundError:
        pass


def _retrieve_pipeline_roll_rag_context(context: str = "") -> str:
    """进入轧制智能体时只检索一次轧制晶粒尺寸文献依据，后续循环复用。"""
    is_wind = "[[WIND_POWER_STEEL_X70_REFERENCE]]" in str(context or "")
    material_label = get_wind_power_material_label(context) if is_wind else "管线钢"
    query = (
        (f"{material_label} " if is_wind else "管线钢 ")
        + "TMCP 控轧过程 奥氏体再结晶区 未再结晶区 轧制温度 FET FDT "
        "道次压下量 道次出口厚度 轧制速度 轧制力 对最终晶粒尺寸 晶粒细化 "
        "晶粒尺寸分布 异常粗晶 的影响规律和研究结论"
    )
    try:
        from hybrid_retriever import hybrid_search

        docs = hybrid_search(
            query,
            k=8,
            db_name="jgyg_Know_db" if is_wind else "gxg_Know_db",
            db_collection="documents",
        )
        if not docs:
            return (
                f"（未检索到{material_label}轧制过程晶粒尺寸相关文献，请根据 TMCP 控轧、再结晶、"
                "未再结晶区变形、道次压下和晶粒细化的材料学知识保守判断。）"
            )

        def _doc_text(doc) -> str:
            if isinstance(doc, dict):
                source = doc.get("source") or (doc.get("metadata") or {}).get("source") or "unknown"
                content = doc.get("content") or doc.get("page_content") or ""
            else:
                metadata = getattr(doc, "metadata", {}) or {}
                source = getattr(doc, "source", None) or metadata.get("source") or "unknown"
                content = getattr(doc, "content", None) or getattr(doc, "page_content", "") or ""
            return f"[来源: {source}]\n{content}"

        print(f"[管线钢轧制智能体] RAG检索命中 {len(docs)} 条文献")
        return "\n\n---\n\n".join(_doc_text(doc) for doc in docs)
    except Exception as exc:
        print(f"[管线钢轧制智能体] RAG检索失败: {exc}")
        return "（RAG检索失败，请根据已有上下文、轧制模拟结果和材料学知识保守判断。）"


def _generate_pipeline_roll_images_with_dll(
    matched_result: dict,
    target_context: str = "",
) -> str | None:
    """同步调用管线钢轧制 DLL，入参为完整 matched_result JSON 字符串。"""
    if not isinstance(matched_result, dict):
        print("[管线钢轧制智能体] matched_result 不是 JSON 对象，跳过轧制 DLL")
        return None
    if not matched_result.get("strCoil") or not matched_result.get("arrBody"):
        print(
            "[管线钢轧制智能体] matched_result 缺少 strCoil 或 arrBody，跳过轧制 DLL；"
            f"strCoil={matched_result.get('strCoil')!r}, "
            f"arrBody长度={len(matched_result.get('arrBody') or [])}"
        )
        return None

    dll_result, target_grade, reference_grade = _build_pipeline_dll_matched_result(
        matched_result,
        target_context,
    )
    row_for_log = _matched_result_body_to_row(dll_result)
    json_input = json.dumps(dll_result, ensure_ascii=False)
    print(
        "[管线钢轧制智能体] DLL模型映射: "
        f"target={target_grade or '未识别'}, reference={reference_grade or '沿用原值'}"
    )
    try:
        with IMAGE_GENERATOR_CALL_LOCK:
            _prepare_pipeline_roll_image_generator_runtime()
            old_cwd = _os.getcwd()
            try:
                # 轧制 DLL 内部使用 .\ModelManage 等相对路径，必须切到 HotColdDataBase\bin\Debug。
                _os.chdir(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
                import clr
                clr.AddReference(PIPELINE_ROLL_IMAGE_GENERATOR_DLL_PATH)
                from ANSTEEL_RollImageGeneratorLib import ImageGenerator

                result = str(ImageGenerator.GenerateAllImagesDLL(json_input))
                if result == "true":
                    print(f"[管线钢轧制智能体] 轧制 DLL 计算成功: strCoil={matched_result.get('strCoil')}")
                else:
                    print(
                        "[管线钢轧制智能体] 轧制 DLL 未完成计算，返回原因: "
                        f"{result}; strCoil={matched_result.get('strCoil')}, "
                        f"STEEL_SIGN={row_for_log.get('STEEL_SIGN')}"
                    )
                return result
            finally:
                _os.chdir(old_cwd)
    except Exception as exc:
        print(
            f"[管线钢轧制智能体] 轧制 DLL 调用异常: {type(exc).__name__}: {exc}; "
            f"strCoil={matched_result.get('strCoil')}, STEEL_SIGN={row_for_log.get('STEEL_SIGN')}"
        )
        return None


def _build_pipeline_roll_agent_user_prompt(
    context: str,
    rag_context: str,
    matched_result: dict,
    historical_roll_reference_markdown: str = "",
) -> str:
    """构建每轮传给 Qwen 的轧制智能体用户提示词。

    各道次晶粒尺寸.png 不再以 base64 文本拼入，而是作为多模态图片随消息一起传入。
    """
    context, rag_context, llm_matched_result = _prepare_pipeline_agent_llm_prompt_data(
        context,
        rag_context,
        matched_result,
    )
    matched_result_json = json.dumps(llm_matched_result, ensure_ascii=False)
    return build_pipeline_roll_agent_user_prompt_text(
        context, rag_context, historical_roll_reference_markdown, matched_result_json
    )




def _collect_pipeline_roll_simulation_context(matched_result: dict) -> str:
    """从 ModelManage/{卷号} 递归读取各道次晶粒尺寸图片并转成 base64。"""
    coil_id = str(matched_result.get("strCoil", "")).strip()
    coil_dir = _os.path.join(PIPELINE_IMAGE_GENERATOR_BIN_DIR, "ModelManage", coil_id)
    pass_grain_size_path = _find_first_file_under_dir(coil_dir, "各道次晶粒尺寸.png")
    return _read_image_base64_for_prompt(pass_grain_size_path, "各道次晶粒尺寸.png")


def _invoke_qwen_roll_agent(user_prompt: str, images: list[tuple[str, str]]) -> dict:
    """调用轧制判断模型；结构修复和结果缓存由外层同轮重试逻辑负责。"""
    print(f"[管线钢轧制智能体] 开始调用判断模型，prompt长度={len(user_prompt)}, 图片数={len(images)}")
    return _invoke_pipeline_qwen_json(
        PIPELINE_ROLL_AGENT_SYSTEM_PROMPT.replace("管线钢", get_wind_power_material_label(user_prompt))
        if WIND_POWER_PROMPT_CONTEXT_TAG in user_prompt else PIPELINE_ROLL_AGENT_SYSTEM_PROMPT,
        user_prompt,
        images,
        "管线钢轧制智能体",
    )


def _find_pipeline_final_pass_thickness_field(matched_result: dict) -> str | None:
    """识别最终有效道次出口厚度字段，用于强制保护最终成品厚度不变。"""
    row = _matched_result_body_to_row(matched_result)
    rough_pass = _to_float(row.get("R_PASS_ACT"))
    finish_pass = _to_float(row.get("F_PASS_ACT"))
    if rough_pass is not None and finish_pass is not None and rough_pass + finish_pass > 0:
        pass_index = int(rough_pass + finish_pass)
        field_name = f"N{pass_index}_DH_CAL"
        if _to_float(row.get(field_name)) is not None:
            return field_name

    for pass_index in range(30, 0, -1):
        field_name = f"N{pass_index}_DH_CAL"
        value = _to_float(row.get(field_name))
        if value is not None and value > 0:
            return field_name
    return None


def _is_roll_agent_editable_pass_field(field_name: str) -> bool:
    """判断字段是否属于轧制智能体可重排的道次参数。"""
    return bool(re.fullmatch(r"N\d+_(DH_CAL|DT_CAL|DW_CAL|FORCE|SPD|ENTR_DATE)", field_name))


def _parse_pipeline_process_datetime(value) -> _datetime | None:
    """解析数据库和模型可能返回的轧制过程时间格式。"""
    text = str(value or "").strip()
    if not text:
        return None
    for date_format in (
        "%Y%m%d%H%M%S.%f",
        "%Y%m%d%H%M%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return _datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def _format_pipeline_dll_datetime(value: _datetime) -> str:
    """输出冷却 DLL 要求的紧凑日期时间格式。"""
    return value.strftime("%Y%m%d%H%M%S")


def _validate_pipeline_dll_time_encodings(
    matched_result: dict,
    *,
    include_cooling_start: bool,
) -> str:
    """校验 DLL 时间字段编码，并生成可直接反馈给模型的修复说明。"""
    row = _matched_result_body_to_row(matched_result)
    errors: list[str] = []

    def _compact_time_error(field_name: str, value) -> None:
        text = str(value or "").strip()
        if not re.fullmatch(r"\d{14}", text):
            errors.append(
                f"时间格式错误：{field_name}={value!r}；该字段必须使用 "
                "yyyyMMddHHmmss 格式，例如 20260318125601，"
                "不能使用 yyyy-MM-dd HH:mm:ss.fff 格式"
            )
            return
        try:
            _datetime.strptime(text, "%Y%m%d%H%M%S")
        except ValueError:
            errors.append(
                f"时间格式错误：{field_name}={value!r}；该字段必须是有效的 "
                "yyyyMMddHHmmss 时间，例如 20260318125601"
            )

    def _pass_time_error(field_name: str, value) -> None:
        text = str(value or "").strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", text):
            errors.append(
                f"时间格式错误：{field_name}={value!r}；该字段必须使用 "
                "yyyy-MM-dd HH:mm:ss.fff 格式，例如 2026-03-18 12:56:01.000，"
                "不能使用 yyyyMMddHHmmss 格式"
            )
            return
        try:
            _datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            errors.append(
                f"时间格式错误：{field_name}={value!r}；该字段必须是有效的 "
                "yyyy-MM-dd HH:mm:ss.fff 时间"
            )

    _compact_time_error("FURNACE_EXIT_TIME", row.get("FURNACE_EXIT_TIME"))
    rough_passes = _to_float(row.get("R_PASS_ACT"))
    finish_passes = _to_float(row.get("F_PASS_ACT"))
    if (
        rough_passes is not None
        and finish_passes is not None
        and abs(rough_passes - round(rough_passes)) <= 1e-9
        and abs(finish_passes - round(finish_passes)) <= 1e-9
    ):
        total_passes = int(round(rough_passes + finish_passes))
        if 1 <= total_passes <= 30:
            for pass_index in range(1, total_passes + 1):
                field_name = f"N{pass_index}_ENTR_DATE"
                _pass_time_error(field_name, row.get(field_name))
    if include_cooling_start:
        _compact_time_error("TIME_ENTR", row.get("TIME_ENTR"))
    return "；".join(errors)


_PIPELINE_MIN_ROUGH_PASSES = 5
_PIPELINE_MIN_FINISH_PASSES = 3


def _normalize_pipeline_declared_pass_tail(matched_result: dict) -> dict:
    """按实际连续正值道次反推总道次数，并清空最终道次后的历史残留值。

    模型经常已经在某一连续正值道次达到 AIM_THICK 并把后续字段归零，却
    忘记同步缩小 R_PASS_ACT/F_PASS_ACT。这里以“从 N1 开始连续且五类参数
    均为正值”的区间为基础；若区间内首次出现 AIM_THICK，则该道次直接作为
    最终有效道次。后端同步修正粗精轧计数并清空尾部，不生成有效区间参数。
    """
    normalized = copy.deepcopy(matched_result)
    body = normalized.get("arrBody")
    if not isinstance(body, list):
        return normalized

    field_items: dict[str, tuple[int, str, object]] = {}
    for item_index, item in enumerate(body):
        key = _get_arrbody_key(item)
        if key:
            field_items[str(key).upper()] = (item_index, str(key), _get_arrbody_value(item))

    numeric_suffixes = ("DH_CAL", "DT_CAL", "DW_CAL", "FORCE", "SPD")
    aim_item = field_items.get("AIM_THICK")
    aim_thickness = _to_float(aim_item[2]) if aim_item else None
    contiguous_passes = 0
    target_pass = None
    pass_temperatures: dict[int, float] = {}
    for pass_index in range(1, 31):
        values = {
            suffix: _to_float(field_items.get(f"N{pass_index}_{suffix}", (None, None, None))[2])
            for suffix in numeric_suffixes
        }
        if not all(value is not None and value > 0 for value in values.values()):
            break
        contiguous_passes = pass_index
        pass_temperatures[pass_index] = values["DT_CAL"]
        if (
            target_pass is None
            and aim_thickness is not None
            and abs(values["DH_CAL"] - aim_thickness) <= 0.01
        ):
            target_pass = pass_index

    total_passes = target_pass or contiguous_passes
    if total_passes < 2:
        return normalized

    rough_item = field_items.get("R_PASS_ACT")
    finish_item = field_items.get("F_PASS_ACT")
    rough_number = _to_float(rough_item[2]) if rough_item else None
    finish_number = _to_float(finish_item[2]) if finish_item else None
    rough_passes = None
    if rough_number is not None and abs(rough_number - round(rough_number)) <= 1e-9:
        candidate_rough_passes = int(round(rough_number))
        if (
            candidate_rough_passes >= _PIPELINE_MIN_ROUGH_PASSES
            and total_passes - candidate_rough_passes >= _PIPELINE_MIN_FINISH_PASSES
        ):
            rough_passes = candidate_rough_passes

    # 当模型仍保留历史粗轧道次数且已超出新的有效总道次数时，根据 FET 与
    # 各道次温度最接近的位置反推精轧起始道次；FET 对应 N{R+1}。
    if rough_passes is None:
        fet_item = field_items.get("FET")
        fet = _to_float(fet_item[2]) if fet_item else None
        finish_start_candidates = [
            pass_index for pass_index in range(2, total_passes + 1)
            if pass_index in pass_temperatures
        ]
        if fet is not None and finish_start_candidates:
            finish_start = min(
                finish_start_candidates,
                key=lambda pass_index: abs(pass_temperatures[pass_index] - fet),
            )
            rough_passes = finish_start - 1
    if (
        rough_passes is None
        or rough_passes < _PIPELINE_MIN_ROUGH_PASSES
        or total_passes - rough_passes < _PIPELINE_MIN_FINISH_PASSES
    ):
        return normalized

    finish_passes = total_passes - rough_passes
    original_rough = rough_item[2] if rough_item else None
    original_finish = finish_item[2] if finish_item else None
    if rough_item:
        item_index, original_key, _ = rough_item
        body[item_index] = {original_key: str(rough_passes)}
    if finish_item:
        item_index, original_key, _ = finish_item
        body[item_index] = {original_key: str(finish_passes)}

    for pass_index in range(total_passes + 1, 31):
        for suffix in numeric_suffixes:
            item = field_items.get(f"N{pass_index}_{suffix}")
            if item:
                item_index, original_key, _ = item
                body[item_index] = {original_key: "0"}
        time_item = field_items.get(f"N{pass_index}_ENTR_DATE")
        if time_item:
            item_index, original_key, _ = time_item
            body[item_index] = {original_key: ""}
    original_total = None
    if rough_number is not None and finish_number is not None:
        original_total = int(round(rough_number)) + int(round(finish_number))
    if (
        original_total != total_passes
        or _to_float(original_rough) != float(rough_passes)
        or _to_float(original_finish) != float(finish_passes)
    ):
        print(
            "[轧制道次计数后校验] 已按连续正值道次修正: "
            f"R_PASS_ACT={original_rough!r}->{rough_passes}, "
            f"F_PASS_ACT={original_finish!r}->{finish_passes}, "
            f"有效总道次={total_passes}, AIM_THICK所在道次={target_pass}"
        )
    return normalized


_PIPELINE_ROLL_WIDTH_TOLERANCE_MM = 0.01


def _get_pipeline_roll_width_change_passes(
    matched_result: dict,
) -> tuple[list[int], int | None, int | None]:
    """统计有效轧制道次之间的两次转钢宽度变化位置。

    N1 是轧后宽度序列的基准，不把 ``SLAB_WIDTH -> N1_DW_CAL`` 的变化当作
    转钢动作。返回的道次号表示该道次宽度相对上一有效道次发生变化。
    """
    row = _matched_result_body_to_row(matched_result)
    rough_number = _to_float(row.get("R_PASS_ACT"))
    finish_number = _to_float(row.get("F_PASS_ACT"))
    if (
        rough_number is None
        or finish_number is None
        or abs(rough_number - round(rough_number)) > 1e-9
        or abs(finish_number - round(finish_number)) > 1e-9
    ):
        return [], None, None

    rough_passes = int(round(rough_number))
    total_passes = rough_passes + int(round(finish_number))
    if rough_passes < 1 or total_passes < 2 or total_passes > 30:
        return [], rough_passes, total_passes

    first_width = _to_float(row.get("N1_DW_CAL"))
    if first_width is None or first_width <= 0:
        return [], rough_passes, total_passes
    previous_width = first_width
    change_passes: list[int] = []
    for pass_index in range(2, total_passes + 1):
        current_width = _to_float(row.get(f"N{pass_index}_DW_CAL"))
        if current_width is None or current_width <= 0:
            return [], rough_passes, total_passes
        if abs(current_width - previous_width) > _PIPELINE_ROLL_WIDTH_TOLERANCE_MM:
            change_passes.append(pass_index)
        previous_width = current_width
    return change_passes, rough_passes, total_passes


def _stabilize_pipeline_roll_turn_widths(matched_result: dict) -> dict:
    """模型重试耗尽后确定性整理转钢标记和多余宽度变化。

    展宽/转钢不是阻断整个材料设计流程的核心门禁：没有宽度变化时使用
    ``0/0``，只有一次变化时使用 ``N/N``；两次及以上变化时保留前两次，
    并把第二次变化后的全部有效道次宽度统一为第二次变化道次宽度。该兜底
    不改厚度、温度、速度、轧制力、道次数或时间。
    """
    stabilized = copy.deepcopy(matched_result)
    change_passes, rough_passes, total_passes = _get_pipeline_roll_width_change_passes(stabilized)
    if rough_passes is None or total_passes is None or total_passes < 1:
        return stabilized

    if not change_passes:
        turn_start = turn_end = 0
    elif len(change_passes) == 1:
        turn_start = turn_end = change_passes[0]
    else:
        turn_start, turn_end = change_passes[:2]
        row = _matched_result_body_to_row(stabilized)
        second_turn_width = row.get(f"N{turn_end}_DW_CAL")
        for pass_index in range(turn_end + 1, total_passes + 1):
            _set_pipeline_arrbody_field(
                stabilized,
                f"N{pass_index}_DW_CAL",
                second_turn_width,
            )

    _set_pipeline_arrbody_field(
        stabilized,
        "WIDTH_ROLL_START_REMARK",
        str(turn_start),
    )
    _set_pipeline_arrbody_field(
        stabilized,
        "WIDTH_ROLL_END_REMARK",
        str(turn_end),
    )
    print(
        "[轧制转钢宽度确定性兜底] "
        f"原宽度变化道次={change_passes or '无'}，"
        f"转钢标记设为 {turn_start}/{turn_end}；"
        + (
            f"N{turn_end} 后有效道次宽度已统一"
            if len(change_passes) > 2 else "未改动有效道次宽度"
        )
    )
    return stabilized


def _is_pipeline_turn_width_validation_error(error: str) -> bool:
    """识别可在轧制智能体重试耗尽后降级放行的转钢/展宽错误。"""
    text = str(error or "")
    return text.startswith((
        "转钢道次标识无效",
        "转钢宽度变化次数无效",
        "转钢宽度变化位置无效",
        "转钢标记与宽度变化不一致",
    ))


def _collect_pipeline_deformation_pass_errors(
    matched_result: dict,
    validate_timing: bool = False,
    validate_cooling_timing: bool = True,
) -> list[str]:
    """一次收集整套轧制规程错误。

    ``validate_timing`` 校验有效道次时间格式和严格递增；只有
    ``validate_cooling_timing`` 同时为真时才读取 ``TIME_ENTR`` 并比较中间坯
    待温与终轧到开冷时间。后置微调和轧制阶段尚未确定开冷时刻，因此只做
    前一种校验，跨阶段关系留给冷却智能体收敛。
    """
    errors: list[str] = []
    if not isinstance(matched_result, dict):
        return ["matched_result 不是 JSON 对象"]
    body = matched_result.get("arrBody")
    if not isinstance(body, list):
        return ["matched_result.arrBody 不是数组"]

    field_items: dict[str, tuple[int, str, object]] = {}
    for item_index, item in enumerate(body):
        key = _get_arrbody_key(item)
        if not key:
            errors.append(f"arrBody 第 {item_index + 1} 项不是单键对象")
            continue
        field_items[str(key).upper()] = (item_index, str(key), _get_arrbody_value(item))

    required_suffixes = ("DH_CAL", "DT_CAL", "DW_CAL", "FORCE", "SPD")
    missing_fields = [
        f"N{pass_index}_{suffix}"
        for pass_index in range(1, 31)
        for suffix in required_suffixes
        if f"N{pass_index}_{suffix}" not in field_items
    ]
    if missing_fields:
        errors.append("缺少道次字段: " + ", ".join(missing_fields))

    rough_item = field_items.get("R_PASS_ACT")
    finish_item = field_items.get("F_PASS_ACT")
    if not rough_item:
        errors.append("缺少 R_PASS_ACT")
    if not finish_item:
        errors.append("缺少 F_PASS_ACT")
    rough_number = _to_float(rough_item[2]) if rough_item else None
    finish_number = _to_float(finish_item[2]) if finish_item else None
    rough_passes = None
    finish_passes = None
    if rough_number is None or abs(rough_number - round(rough_number)) > 1e-9:
        errors.append(f"R_PASS_ACT={rough_item[2] if rough_item else None!r} 不是整数")
    else:
        rough_passes = int(round(rough_number))
    if finish_number is None or abs(finish_number - round(finish_number)) > 1e-9:
        errors.append(f"F_PASS_ACT={finish_item[2] if finish_item else None!r} 不是整数")
    else:
        finish_passes = int(round(finish_number))

    total_passes = None
    if rough_passes is not None and finish_passes is not None:
        total_passes = rough_passes + finish_passes
        if rough_passes < _PIPELINE_MIN_ROUGH_PASSES:
            errors.append(
                f"R_PASS_ACT={rough_passes} 不满足粗轧最少道次数要求："
                f"R_PASS_ACT 必须大于或等于 {_PIPELINE_MIN_ROUGH_PASSES}；"
                "请重新分配完整粗轧/精轧规程及全部有效道次参数"
            )
        if finish_passes < _PIPELINE_MIN_FINISH_PASSES:
            errors.append(
                f"F_PASS_ACT={finish_passes} 不满足精轧最少道次数要求："
                f"F_PASS_ACT 必须大于或等于 {_PIPELINE_MIN_FINISH_PASSES}；"
                "请重新分配完整粗轧/精轧规程及全部有效道次参数"
            )
        if total_passes > 30:
            errors.append(f"R_PASS_ACT+F_PASS_ACT={total_passes} 超过 N1-N30 字段容量")
        if total_passes < 2:
            errors.append(f"R_PASS_ACT+F_PASS_ACT={total_passes} 无法同时包含粗轧和精轧")

    if total_passes is None or total_passes < 2 or total_passes > 30 or missing_fields:
        return errors

    # 转钢起止道次属于当前轧制规程，而不是不可变的历史身份信息。模型每次
    # 重排道次时必须同步返回一对合法标识；后端在这里统一校验，错误会原样
    # 反馈给模型进行同轮修复，避免旧的转钢区间落入新的粗轧/精轧分界中。
    turn_start_item = field_items.get("WIDTH_ROLL_START_REMARK")
    turn_end_item = field_items.get("WIDTH_ROLL_END_REMARK")
    if not turn_start_item:
        errors.append("缺少 WIDTH_ROLL_START_REMARK")
    if not turn_end_item:
        errors.append("缺少 WIDTH_ROLL_END_REMARK")
    turn_start_number = _to_float(turn_start_item[2]) if turn_start_item else None
    turn_end_number = _to_float(turn_end_item[2]) if turn_end_item else None
    turn_start = None
    turn_end = None
    if turn_start_number is None or abs(turn_start_number - round(turn_start_number)) > 1e-9:
        errors.append(
            f"WIDTH_ROLL_START_REMARK={turn_start_item[2] if turn_start_item else None!r} 不是整数"
        )
    else:
        turn_start = int(round(turn_start_number))
    if turn_end_number is None or abs(turn_end_number - round(turn_end_number)) > 1e-9:
        errors.append(
            f"WIDTH_ROLL_END_REMARK={turn_end_item[2] if turn_end_item else None!r} 不是整数"
        )
    else:
        turn_end = int(round(turn_end_number))
    if turn_start is not None and turn_end is not None and rough_passes is not None:
        marker_is_none = turn_start == 0 and turn_end == 0
        marker_is_single = 2 <= turn_start == turn_end < rough_passes
        marker_is_double = 2 <= turn_start < turn_end < rough_passes
        if not (marker_is_none or marker_is_single or marker_is_double):
            errors.append(
                "转钢道次标识无效：无转钢使用0/0，一次转钢使用N/N，"
                "两次转钢使用START/END；非零标记必须从N2开始且早于粗轧末道，"
                f"当前为 {turn_start}, {turn_end}, R_PASS_ACT={rough_passes}"
            )

    thicknesses: list[float | None] = []
    temperatures: list[float | None] = []
    widths: list[float | None] = []
    for pass_index in range(1, total_passes + 1):
        pass_values: dict[str, float | None] = {}
        for suffix in required_suffixes:
            field_name = f"N{pass_index}_{suffix}"
            item = field_items.get(field_name)
            value = _to_float(item[2]) if item else None
            pass_values[suffix] = value
            if value is None or value <= 0:
                errors.append(f"有效道次 {field_name}={item[2] if item else None!r} 必须为正数")
        thicknesses.append(pass_values["DH_CAL"])
        temperatures.append(pass_values["DT_CAL"])
        widths.append(pass_values["DW_CAL"])

    if all(width is not None and width > 0 for width in widths):
        width_change_passes: list[int] = []
        previous_width = widths[0]
        for pass_index, current_width in enumerate(widths[1:], start=2):
            if abs(current_width - previous_width) > _PIPELINE_ROLL_WIDTH_TOLERANCE_MM:
                width_change_passes.append(pass_index)
            previous_width = current_width

        if len(width_change_passes) > 2:
            errors.append(
                "转钢宽度变化次数无效：从 N1_DW_CAL 开始比较全部有效道次宽度，"
                f"最多允许变化2次，当前变化{len(width_change_passes)}次，"
                f"变化道次={width_change_passes}；请将第二次变化后的宽度统一"
            )
        else:
            expected_start = width_change_passes[0] if width_change_passes else 0
            expected_end = width_change_passes[-1] if width_change_passes else 0
            if any(
                pass_index < 2 or pass_index >= rough_passes
                for pass_index in width_change_passes
            ):
                errors.append(
                    "转钢宽度变化位置无效：宽度变化必须从N2开始，"
                    "并在粗轧最后一道次之前完成，"
                    f"当前变化道次={width_change_passes}, R_PASS_ACT={rough_passes}"
                )
            if turn_start != expected_start or turn_end != expected_end:
                errors.append(
                    "转钢标记与宽度变化不一致："
                    f"N*_DW_CAL 的变化道次={width_change_passes or '无'}，"
                    "WIDTH_ROLL_START_REMARK/WIDTH_ROLL_END_REMARK "
                    f"应为 {expected_start}/{expected_end}，当前为 {turn_start}/{turn_end}"
                )

    slab_item = field_items.get("SLAB_THICK")
    slab_thickness = _to_float(slab_item[2]) if slab_item else None
    if slab_thickness is not None and thicknesses and thicknesses[0] is not None:
        first_drop = slab_thickness - thicknesses[0]
        if first_drop <= 1.0 + 1e-9:
            errors.append(
                f"首道次 N1_DH_CAL={thicknesses[0]} 与 SLAB_THICK={slab_thickness} "
                "的厚度差未严格大于1mm"
            )

    for pass_index in range(2, total_passes + 1):
        previous = thicknesses[pass_index - 2]
        current = thicknesses[pass_index - 1]
        if previous is None or current is None:
            continue
        drop = previous - current
        if drop <= 1.0 + 1e-9:
            errors.append(
                f"N{pass_index}_DH_CAL={current} 与上一道次厚度 {previous} "
                f"的差值={drop:.4f}mm，未连续递减且严格大于1mm"
            )

    aim_item = field_items.get("AIM_THICK")
    aim_thickness = _to_float(aim_item[2]) if aim_item else None
    if aim_thickness is None:
        errors.append(f"AIM_THICK={aim_item[2] if aim_item else None!r} 不是有效数值")
    else:
        for pass_index, thickness in enumerate(thicknesses[:-1], start=1):
            if thickness is not None and thickness <= aim_thickness + 1e-9:
                errors.append(
                    f"非最终道次 N{pass_index}_DH_CAL={thickness} 已小于或等于 "
                    f"AIM_THICK={aim_thickness}，整套道次必须重新设计"
                )
        final_thickness = thicknesses[-1] if thicknesses else None
        if final_thickness is None or abs(final_thickness - aim_thickness) > 0.01:
            errors.append(
                f"最终有效道次 N{total_passes}_DH_CAL={final_thickness} "
                f"与 AIM_THICK={aim_thickness} 不一致"
            )

    fdt_item = field_items.get("FDT")
    fdt = _to_float(fdt_item[2]) if fdt_item else None
    final_temperature = temperatures[-1] if temperatures else None
    if fdt is None:
        errors.append(f"FDT={fdt_item[2] if fdt_item else None!r} 不是有效数值")
    elif final_temperature is None:
        errors.append(f"N{total_passes}_DT_CAL 不是有效数值")
    elif abs(final_temperature - fdt) > 5.0 + 1e-9:
        errors.append(
            f"末道次温度校验失败：N{total_passes}_DT_CAL={final_temperature:.2f}℃，"
            f"FDT={fdt:.2f}℃，绝对偏差={abs(final_temperature - fdt):.2f}℃，超过5℃"
        )

    if validate_timing:
        active_time_fields = [f"N{index}_ENTR_DATE" for index in range(1, total_passes + 1)]
        required_time_fields = [*active_time_fields]
        if validate_cooling_timing:
            required_time_fields.append("TIME_ENTR")
        parsed_times: dict[str, _datetime | None] = {}
        for field_name in required_time_fields:
            item = field_items.get(field_name)
            if not item:
                errors.append(f"缺少轧制时间字段 {field_name}")
                parsed_times[field_name] = None
                continue
            parsed_time = _parse_pipeline_process_datetime(item[2])
            parsed_times[field_name] = parsed_time
            if parsed_time is None:
                errors.append(f"轧制时间格式无效: {field_name}={item[2]!r}")

        for previous_field, current_field in zip(active_time_fields, active_time_fields[1:]):
            previous_time = parsed_times.get(previous_field)
            current_time = parsed_times.get(current_field)
            if previous_time is not None and current_time is not None and current_time <= previous_time:
                errors.append(
                    f"轧制道次时间顺序无效：{current_field} 必须晚于 {previous_field}"
                )

        if (
            validate_cooling_timing
            and rough_passes is not None
            and rough_passes >= _PIPELINE_MIN_ROUGH_PASSES
            and total_passes - rough_passes >= _PIPELINE_MIN_FINISH_PASSES
        ):
            rough_end_field = f"N{rough_passes}_ENTR_DATE"
            finish_start_field = f"N{rough_passes + 1}_ENTR_DATE"
            # matched_result 只保存真实变形道次，DLL 的末端空过由 DLL 内部
            # 增加，因此终轧时刻必须引用 R_PASS_ACT+F_PASS_ACT 对应的末道。
            cooling_reference_field = f"N{total_passes}_ENTR_DATE"
            rough_end = parsed_times.get(rough_end_field)
            finish_start = parsed_times.get(finish_start_field)
            cooling_reference = parsed_times.get(cooling_reference_field)
            cooling_start = parsed_times.get("TIME_ENTR")
            if all(value is not None for value in (rough_end, finish_start, cooling_reference, cooling_start)):
                intermediate_wait = abs((finish_start - rough_end).total_seconds())
                finish_to_cooling = abs((cooling_start - cooling_reference).total_seconds())
                if intermediate_wait + 1e-9 < finish_to_cooling:
                    errors.append(
                        f"中间坯待温时间={intermediate_wait:.3f}s，小于终轧到开冷时间="
                        f"{finish_to_cooling:.3f}s"
                    )
    return errors


def _pipeline_roll_errors_require_global_redesign(errors: list[str]) -> bool:
    """厚度、道次数或断道错误必须触发整套规程重写，不能逐字段修补。"""
    global_markers = (
        "DH_CAL", "厚度", "有效道次", "R_PASS_ACT", "F_PASS_ACT", "断道",
        "轧制道次时间顺序", "轧制时间格式", "中间坯待温时间",
    )
    return any(marker in error for error in errors for marker in global_markers)


def _normalize_pipeline_deformation_passes(
    matched_result: dict,
    stage_label: str,
    validate_timing: bool = False,
    validate_cooling_timing: bool = True,
    tolerate_turn_width_errors: bool = False,
) -> tuple[dict | None, str]:
    """校验真实变形道次并修正粗精轧计数，DLL 固定空过不计入 matched_result。

    有效道次必须从 N1 连续开始，且 DH_CAL、DT_CAL、DW_CAL、FORCE、SPD
    五项均为正数。第一个非完整道次之后如果再次出现完整正值道次，说明道次
    中间断开，不能自动猜测，应退回模型修复。连续有效区间之后的残留字段会
    全部归零；F_PASS_ACT 按“有效总道次-R_PASS_ACT”重新计算。
    """
    matched_result = _normalize_pipeline_declared_pass_tail(matched_result)
    collected_errors = _collect_pipeline_deformation_pass_errors(
        matched_result,
        validate_timing=validate_timing,
        validate_cooling_timing=validate_cooling_timing,
    )
    if tolerate_turn_width_errors:
        ignored_errors = [
            error for error in collected_errors
            if _is_pipeline_turn_width_validation_error(error)
        ]
        collected_errors = [
            error for error in collected_errors
            if not _is_pipeline_turn_width_validation_error(error)
        ]
        if ignored_errors:
            print(
                f"[{stage_label}] 转钢/展宽非阻断兜底已放行: "
                + "；".join(ignored_errors)
            )
    if collected_errors:
        return None, "；".join(collected_errors)

    if not isinstance(matched_result, dict):
        return None, "matched_result 不是 JSON 对象"
    body = matched_result.get("arrBody")
    if not isinstance(body, list):
        return None, "matched_result.arrBody 不是数组"

    field_items: dict[str, tuple[int, str, object]] = {}
    for item_index, item in enumerate(body):
        key = _get_arrbody_key(item)
        if not key:
            return None, f"arrBody 第 {item_index + 1} 项不是单键对象"
        field_items[str(key).upper()] = (item_index, str(key), _get_arrbody_value(item))

    required_suffixes = ("DH_CAL", "DT_CAL", "DW_CAL", "FORCE", "SPD")
    missing_fields = [
        f"N{pass_index}_{suffix}"
        for pass_index in range(1, 31)
        for suffix in required_suffixes
        if f"N{pass_index}_{suffix}" not in field_items
    ]
    if missing_fields:
        return None, f"缺少道次字段: {', '.join(missing_fields[:5])}"

    active_passes: list[int] = []
    active_thicknesses: list[float] = []
    reached_tail = False
    for pass_index in range(1, 31):
        values = [
            _to_float(field_items[f"N{pass_index}_{suffix}"][2])
            for suffix in required_suffixes
        ]
        is_active = all(value is not None and value > 0 for value in values)
        if is_active:
            if reached_tail:
                return None, f"N{pass_index} 在零值或不完整道次之后重新出现正值，形成中间断道"
            thickness = values[0]
            if active_thicknesses and active_thicknesses[-1] - thickness <= 1.0 + 1e-9:
                return None, (
                    f"N{pass_index}_DH_CAL={thickness} 与上一道次厚度"
                    f" {active_thicknesses[-1]} 的差值未严格大于1mm"
                )
            active_passes.append(pass_index)
            active_thicknesses.append(thickness)
        else:
            reached_tail = True

    if not active_passes:
        return None, "没有识别到从 N1 开始连续且五类参数均为正数的真实变形道次"

    active_count = active_passes[-1]
    rough_item = field_items.get("R_PASS_ACT")
    finish_item = field_items.get("F_PASS_ACT")
    if not rough_item or not finish_item:
        return None, "缺少 R_PASS_ACT 或 F_PASS_ACT"
    rough_number = _to_float(rough_item[2])
    if rough_number is None or abs(rough_number - round(rough_number)) > 1e-9:
        return None, f"R_PASS_ACT={rough_item[2]!r} 不是整数"
    rough_passes = int(round(rough_number))
    if rough_passes < _PIPELINE_MIN_ROUGH_PASSES:
        return None, (
            f"R_PASS_ACT={rough_passes} 不满足粗轧最少道次数要求："
            f"R_PASS_ACT 必须大于或等于 {_PIPELINE_MIN_ROUGH_PASSES}；"
            "请重新分配完整粗轧/精轧规程及全部有效道次参数"
        )
    if rough_passes >= active_count:
        return None, (
            f"R_PASS_ACT={rough_passes} 与真实有效总道次 {active_count} 不匹配，"
            f"无法得到至少 {_PIPELINE_MIN_FINISH_PASSES} 道精轧变形道次"
        )
    finish_passes = active_count - rough_passes
    if finish_passes < _PIPELINE_MIN_FINISH_PASSES:
        return None, (
            f"F_PASS_ACT={finish_passes} 不满足精轧最少道次数要求："
            f"F_PASS_ACT 必须大于或等于 {_PIPELINE_MIN_FINISH_PASSES}；"
            "请重新分配完整粗轧/精轧规程及全部有效道次参数"
        )

    intermediate_wait_seconds = None
    finish_to_cooling_seconds = None
    if validate_timing:
        # 粗轧结束至精轧开始的时间定义为中间坯待温时间；终轧至开冷时间使用
        # 真实最后一道变形（总有效道次数）对应的 ENTR_DATE。后置微调和轧制
        # 阶段只检查道次时间，冷却阶段确定 TIME_ENTR 后再执行跨阶段比较。
        rough_end_time_field = f"N{rough_passes}_ENTR_DATE"
        finish_start_time_field = f"N{rough_passes + 1}_ENTR_DATE"
        finish_to_cooling_reference_field = f"N{active_count}_ENTR_DATE"
        active_time_fields = [
            f"N{pass_index}_ENTR_DATE"
            for pass_index in range(1, active_count + 1)
        ]
        required_time_fields = (
            (*active_time_fields, "TIME_ENTR")
            if validate_cooling_timing
            else tuple(active_time_fields)
        )
        missing_time_fields = [
            field_name for field_name in required_time_fields
            if field_name not in field_items
        ]
        if missing_time_fields:
            return None, f"缺少轧制时间字段: {', '.join(missing_time_fields)}"

        parsed_times = {
            field_name: _parse_pipeline_process_datetime(field_items[field_name][2])
            for field_name in required_time_fields
        }
        invalid_time_fields = [
            f"{field_name}={field_items[field_name][2]!r}"
            for field_name, parsed_time in parsed_times.items()
            if parsed_time is None
        ]
        if invalid_time_fields:
            return None, "轧制时间格式无效: " + ", ".join(invalid_time_fields)

        for previous_field, current_field in zip(active_time_fields, active_time_fields[1:]):
            if parsed_times[current_field] <= parsed_times[previous_field]:
                return None, (
                    "轧制道次时间顺序无效："
                    f"{current_field}={field_items[current_field][2]!r} 必须晚于 "
                    f"{previous_field}={field_items[previous_field][2]!r}；"
                    "请按有效道次顺序重新设计 N*_ENTR_DATE"
                )

        if validate_cooling_timing:
            intermediate_wait_seconds = abs(
                (
                    parsed_times[finish_start_time_field]
                    - parsed_times[rough_end_time_field]
                ).total_seconds()
            )
            finish_to_cooling_seconds = abs(
                (
                    parsed_times["TIME_ENTR"]
                    - parsed_times[finish_to_cooling_reference_field]
                ).total_seconds()
            )
            if intermediate_wait_seconds + 1e-9 < finish_to_cooling_seconds:
                return None, (
                    "中间坯待温时间校验失败："
                    f"|{finish_start_time_field}-{rough_end_time_field}|="
                    f"{intermediate_wait_seconds:.3f}s，小于终轧到开冷时间 "
                    f"|TIME_ENTR-{finish_to_cooling_reference_field}|="
                    f"{finish_to_cooling_seconds:.3f}s；该关系应由冷却阶段调整 "
                    "TIME_ENTR 后满足"
                )

    aim_item = field_items.get("AIM_THICK")
    aim_thickness = _to_float(aim_item[2]) if aim_item else None
    if aim_thickness is not None and abs(active_thicknesses[-1] - aim_thickness) > 0.01:
        return None, (
            f"最终有效道次 N{active_count}_DH_CAL={active_thicknesses[-1]} "
            f"与 AIM_THICK={aim_thickness} 不一致"
        )

    # 终轧温度必须与真实最后一道变形温度一致。该检查放在有效道次识别之后，
    # 防止错误地使用 DLL 固定空过或已归零道次的温度进行比较。
    fdt_item = field_items.get("FDT")
    if not fdt_item:
        return None, "缺少终轧温度字段 FDT，无法校验末道次温度"
    fdt = _to_float(fdt_item[2])
    final_temperature_field = f"N{active_count}_DT_CAL"
    final_temperature = _to_float(field_items[final_temperature_field][2])
    if fdt is None:
        return None, f"FDT={fdt_item[2]!r} 不是有效数值"
    if final_temperature is None:
        return None, (
            f"{final_temperature_field}="
            f"{field_items[final_temperature_field][2]!r} 不是有效数值"
        )
    final_temperature_deviation = abs(final_temperature - fdt)
    if final_temperature_deviation > 5.0 + 1e-9:
        return None, (
            f"末道次温度校验失败：{final_temperature_field}={final_temperature:.2f}℃，"
            f"FDT={fdt:.2f}℃，绝对偏差={final_temperature_deviation:.2f}℃，"
            "超过允许的5℃；请重新设计 FDT 和末道次 DT_CAL，使偏差不超过5℃"
        )

    normalized = copy.deepcopy(matched_result)
    normalized_body = normalized.get("arrBody")
    rough_index, rough_key, _ = rough_item
    finish_index, finish_key, finish_original = finish_item
    normalized_body[rough_index] = {rough_key: str(rough_passes)}
    normalized_body[finish_index] = {finish_key: str(finish_passes)}

    # matched_result 只保存真实变形道次。模型残留值和 DLL 所需的末端空过
    # 都不能计入这里；DLL 内部仍按其既有逻辑自行增加一个空过道次。
    for pass_index in range(active_count + 1, 31):
        for suffix in required_suffixes:
            item_index, original_key, _ = field_items[f"N{pass_index}_{suffix}"]
            normalized_body[item_index] = {original_key: "0"}
        time_item = field_items.get(f"N{pass_index}_ENTR_DATE")
        if time_item:
            item_index, original_key, _ = time_item
            normalized_body[item_index] = {original_key: ""}

    original_finish = _to_float(finish_original)
    if original_finish is None or abs(original_finish - finish_passes) > 1e-9:
        print(
            f"[{stage_label}] 道次后校验已修正: R_PASS_ACT={rough_passes}, "
            f"F_PASS_ACT={finish_original!r}->{finish_passes}, 有效总道次={active_count}；"
            "DLL 固定增加的空过道次不计入 matched_result"
        )
    else:
        timing_summary = (
            f", 中间坯待温={intermediate_wait_seconds:.3f}s, "
            f"终轧到开冷={finish_to_cooling_seconds:.3f}s"
            if intermediate_wait_seconds is not None and finish_to_cooling_seconds is not None
            else ""
        )
        print(
            f"[{stage_label}] 道次后校验通过: R_PASS_ACT={rough_passes}, "
            f"F_PASS_ACT={finish_passes}, 有效总道次={active_count}"
            f"{timing_summary}"
        )
    return normalized, ""


def _sanitize_pipeline_roll_agent_result(
    original: dict,
    candidate: dict,
) -> dict | tuple[None, str] | None:
    """校验轧制结果；只采纳轧制工艺变化，性能字段始终恢复原值。"""
    if not isinstance(original, dict) or not isinstance(candidate, dict):
        return None
    if list(candidate.keys()) != list(original.keys()):
        print("[管线钢轧制智能体] Qwen返回顶层键不一致，放弃本轮结果")
        return None

    original_body = original.get("arrBody")
    candidate_body = candidate.get("arrBody")
    if not isinstance(original_body, list) or not isinstance(candidate_body, list):
        return None
    if len(original_body) != len(candidate_body):
        print("[管线钢轧制智能体] Qwen返回 arrBody 长度不一致，放弃本轮结果")
        return None

    sanitized = copy.deepcopy(original)
    candidate_state = _coerce_reheat_is_state(candidate.get("isState", original.get("isState")))
    sanitized["isState"] = candidate_state if isinstance(candidate_state, bool) else original.get("isState")
    sanitized_body = []
    for original_item, candidate_item in zip(original_body, candidate_body):
        original_key = _get_arrbody_key(original_item)
        candidate_key = _get_arrbody_key(candidate_item)
        if not original_key or original_key != candidate_key:
            print("[管线钢轧制智能体] Qwen返回 arrBody 字段顺序或字段名不一致，放弃本轮结果")
            return None

        original_value = _get_arrbody_value(original_item)
        candidate_value = _get_arrbody_value(candidate_item)
        field_name = original_key.upper()

        if field_name in {"AIM_THICK", "FET", "FDT", "FURNACE_EXIT_TIME"}:
            sanitized_body.append({original_key: candidate_value})
        elif field_name in PIPELINE_REFINABLE_PASS_COUNT_FIELDS:
            sanitized_body.append({original_key: candidate_value})
        elif field_name in PIPELINE_REFINABLE_TURN_FIELDS:
            sanitized_body.append({original_key: candidate_value})
        elif _is_roll_agent_editable_pass_field(field_name):
            sanitized_body.append({original_key: candidate_value})
        else:
            # 非允许字段全部恢复原值；道次增删通过固定字段归零/赋值表达。
            sanitized_body.append({original_key: original_value})

    sanitized["arrBody"] = sanitized_body
    normalized, validation_error = _normalize_pipeline_deformation_passes(
        sanitized,
        "管线钢轧制智能体",
        # 模型的厚度、温度、宽度、速度和轧制力方案先按工程门禁采纳；
        # ENTR_DATE 缺失、格式或顺序问题统一在智能体出口确定性重建，避免
        # 因单个时间字段错误丢弃整套已经合格的轧制规程。
        validate_timing=False,
        validate_cooling_timing=False,
    )
    if normalized is None:
        print(f"[管线钢轧制智能体] 道次后校验未通过: {validation_error}")
        return None, validation_error
    if normalized.get("isState") is True:
        time_encoding_error = _validate_pipeline_dll_time_encodings(
            normalized,
            include_cooling_start=False,
        )
        if time_encoding_error:
            print(f"[管线钢轧制智能体] 时间编码校验未通过: {time_encoding_error}")
            return None, time_encoding_error
    return normalized


class PipelineRollValidationError(RuntimeError):
    """轧制智能体最终规程未通过硬门禁，禁止进入冷却和报告阶段。"""


def _require_valid_pipeline_roll_result(
    matched_result: dict,
    stage_label: str = "管线钢轧制智能体最终门禁",
) -> dict:
    """校验轧制智能体最终设计，任何错误都不能用历史规程静默兜底。

    模型每次返回后已经执行同一套校验并在当前轮内带错误重试。此函数位于
    轧制智能体出口，专门堵住“重试耗尽后返回进入轮次前旧结果”的路径；
    尤其保证最后有效道次 DH_CAL 必须与锁定的 AIM_THICK 完全一致。
    """
    normalized, validation_error = _normalize_pipeline_deformation_passes(
        matched_result,
        stage_label,
        validate_timing=True,
        validate_cooling_timing=False,
    )
    if normalized is None:
        # 该入口仅在轧制智能体模型重试耗尽或最终返回时执行。先硬编码整理转钢
        # 与宽度，再放行仍残留的纯展宽告警；核心厚度、温度、道次结构错误不放行。
        repaired = _stabilize_pipeline_roll_turn_widths(matched_result)
        normalized, validation_error = _normalize_pipeline_deformation_passes(
            repaired,
            stage_label,
            validate_timing=True,
            validate_cooling_timing=False,
            tolerate_turn_width_errors=True,
        )
        if normalized is None and _is_pipeline_pass_time_only_error(validation_error):
            repaired = _stabilize_pipeline_roll_pass_times(repaired)
            normalized, validation_error = _normalize_pipeline_deformation_passes(
                repaired,
                stage_label,
                validate_timing=True,
                validate_cooling_timing=False,
                tolerate_turn_width_errors=True,
            )
    if normalized is None:
        raise PipelineRollValidationError(
            "轧制规程设计后校验未通过，禁止进入最终报告："
            + (validation_error or "未知轧制规程错误")
        )
    return normalized


def _is_pipeline_pass_time_only_error(validation_error: str) -> bool:
    """判断轧制门禁错误是否仅由 ENTR_DATE 缺失、格式或顺序引起。"""
    text = str(validation_error or "")
    if not text:
        return False
    time_markers = ("ENTR_DATE", "轧制时间", "道次时间顺序")
    non_time_markers = (
        "DH_CAL", "厚度", "FDT", "末道次温度", "有效道次", "断道",
        "R_PASS_ACT", "F_PASS_ACT", "DW_CAL", "FORCE", "SPD",
    )
    return any(marker in text for marker in time_markers) and not any(
        marker in text for marker in non_time_markers
    )


def _stabilize_pipeline_roll_pass_times(matched_result: dict) -> dict:
    """只重建有效轧制道次时间轴，保持全部轧制参数和道次数不变。"""
    stabilized = copy.deepcopy(matched_result)
    row = _matched_result_body_to_row(stabilized)
    rough_number = _to_float(row.get("R_PASS_ACT"))
    finish_number = _to_float(row.get("F_PASS_ACT"))
    if rough_number is None or finish_number is None:
        return stabilized
    if (
        abs(rough_number - round(rough_number)) > 1e-9
        or abs(finish_number - round(finish_number)) > 1e-9
    ):
        return stabilized
    rough_passes = int(round(rough_number))
    total_passes = rough_passes + int(round(finish_number))
    if rough_passes < 1 or total_passes <= rough_passes or total_passes > 30:
        return stabilized

    parsed_times = [
        _parse_pipeline_process_datetime(row.get(f"N{index}_ENTR_DATE"))
        for index in range(1, total_passes + 1)
    ]
    anchor = next((value for value in parsed_times if value is not None), None)
    if anchor is None:
        cooling_start = _parse_pipeline_process_datetime(row.get("TIME_ENTR"))
        anchor = cooling_start - _timedelta(seconds=5 * total_passes + 60) \
            if cooling_start is not None else _datetime.now().replace(microsecond=0)
    current_time = anchor
    for pass_index in range(1, total_passes + 1):
        if pass_index > 1:
            current_time += _timedelta(
                seconds=60.0 if pass_index == rough_passes + 1 else 5.0
            )
        _set_pipeline_arrbody_field(
            stabilized,
            f"N{pass_index}_ENTR_DATE",
            _format_pipeline_process_datetime(current_time),
        )
    print(
        "[轧制时序确定性修复] 模型道次时间缺失、格式错误或逆序，"
        f"已按最终 R_PASS_ACT={rough_passes}, F_PASS_ACT={total_passes - rough_passes} "
        "重建严格递增时间轴"
    )
    return stabilized


def _refine_pipeline_roll_process_with_agent(
    matched_result: dict,
    context: str,
    reasoning_key_prefix: str | None = None,
    progress_callback=None,
    historical_roll_reference_markdown: str = "",
) -> dict:
    """兼容原调用名；轧制智能体业务已迁移到 pipeline_agents.py。"""
    return refine_rolling_process(
        matched_result,
        context,
        reasoning_key_prefix,
        progress_callback=progress_callback,
        historical_roll_reference_markdown=historical_roll_reference_markdown,
        dependencies=_build_process_agent_dependencies(),
    )


def _prepare_pipeline_cooling_image_generator_runtime() -> None:
    """准备管线钢冷却智能体 ANSTEEL_CoolingImageGeneratorLib 的运行时和依赖搜索路径。"""
    if not _os.path.exists(PIPELINE_COOLING_IMAGE_GENERATOR_DLL_PATH):
        raise FileNotFoundError(f"未找到管线钢冷却绘图DLL: {PIPELINE_COOLING_IMAGE_GENERATOR_DLL_PATH}")

    # 冷却 DLL 与现有管线钢绘图 DLL 位于同一 bin\Debug 目录，复用相同依赖搜索路径。
    if PIPELINE_IMAGE_GENERATOR_BIN_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_IMAGE_GENERATOR_BIN_DIR)

    # Windows 原生依赖需要加入 DLL 搜索目录；句柄保存到全局列表避免被提前释放。
    if hasattr(_os, "add_dll_directory"):
        handle = _os.add_dll_directory(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
        IMAGE_GENERATOR_DLL_DIRECTORY_HANDLES.append(handle)

    # pythonnet 3 支持显式选择 .NET Framework；运行时已初始化时忽略 RuntimeError。
    try:
        from pythonnet import load
        try:
            load("netfx")
        except RuntimeError:
            pass
    except ModuleNotFoundError:
        pass


def _retrieve_pipeline_cooling_rag_context(context: str = "") -> str:
    """进入冷却智能体时只检索一次冷却相变和铁素体晶粒尺寸文献依据。"""
    is_wind = "[[WIND_POWER_STEEL_X70_REFERENCE]]" in str(context or "")
    material_label = get_wind_power_material_label(context) if is_wind else "管线钢"
    query = (
        (f"{material_label} " if is_wind else "管线钢 ")
        + "TMCP 加速冷却 层流冷却 冷却速度 终轧温度 FDT 入水温度 TEMP_ENTR 返红温度 SELF_TEMP "
        "铁素体晶粒尺寸 针状铁素体 贝氏体 相变比例 相组成 组织分数 "
        "冷却后最终铁素体尺寸与相比例分数 的影响规律和研究结论"
    )
    try:
        from hybrid_retriever import hybrid_search

        docs = hybrid_search(
            query,
            k=8,
            db_name="jgyg_Know_db" if is_wind else "gxg_Know_db",
            db_collection="documents",
        )
        if not docs:
            return (
                f"（未检索到{material_label}冷却后铁素体晶粒尺寸和相变比例相关文献，请根据 TMCP 控冷、"
                "相变动力学、返红温度、冷速与铁素体/贝氏体组织控制的材料学知识保守判断。）"
            )

        def _doc_text(doc) -> str:
            if isinstance(doc, dict):
                source = doc.get("source") or (doc.get("metadata") or {}).get("source") or "unknown"
                content = doc.get("content") or doc.get("page_content") or ""
            else:
                metadata = getattr(doc, "metadata", {}) or {}
                source = getattr(doc, "source", None) or metadata.get("source") or "unknown"
                content = getattr(doc, "content", None) or getattr(doc, "page_content", "") or ""
            return f"[来源: {source}]\n{content}"

        print(f"[管线钢冷却智能体] RAG检索命中 {len(docs)} 条文献")
        return "\n\n---\n\n".join(_doc_text(doc) for doc in docs)
    except Exception as exc:
        print(f"[管线钢冷却智能体] RAG检索失败: {exc}")
        return "（RAG检索失败，请根据已有上下文、冷却模拟结果和材料学知识保守判断。）"


def _generate_pipeline_cooling_images_with_dll(
    matched_result: dict,
    target_context: str = "",
) -> str | None:
    """同步调用管线钢冷却 DLL，入参为完整 matched_result JSON 字符串。"""
    if not isinstance(matched_result, dict):
        print("[管线钢冷却智能体] matched_result 不是 JSON 对象，跳过冷却 DLL")
        return None
    if not matched_result.get("strCoil") or not matched_result.get("arrBody"):
        print(
            "[管线钢冷却智能体] matched_result 缺少 strCoil 或 arrBody，跳过冷却 DLL；"
            f"strCoil={matched_result.get('strCoil')!r}, "
            f"arrBody长度={len(matched_result.get('arrBody') or [])}"
        )
        return None

    dll_result, target_grade, reference_grade = _build_pipeline_dll_matched_result(
        matched_result,
        target_context,
    )
    row_for_log = _matched_result_body_to_row(dll_result)
    json_input = json.dumps(dll_result, ensure_ascii=False)
    print(
        "[管线钢冷却智能体] DLL模型映射: "
        f"target={target_grade or '未识别'}, reference={reference_grade or '沿用原值'}"
    )
    try:
        with IMAGE_GENERATOR_CALL_LOCK:
            _prepare_pipeline_cooling_image_generator_runtime()
            old_cwd = _os.getcwd()
            try:
                # 冷却 DLL 内部依赖 .\ModelManage 等相对路径，必须切到 HotColdDataBase\bin\Debug。
                _os.chdir(PIPELINE_IMAGE_GENERATOR_BIN_DIR)
                import clr
                clr.AddReference(PIPELINE_COOLING_IMAGE_GENERATOR_DLL_PATH)
                from ANSTEEL_CoolingImageGeneratorLib import ImageGenerator

                result = str(ImageGenerator.GenerateAllImagesDLL(json_input))
                if result == "true":
                    print(f"[管线钢冷却智能体] 冷却 DLL 计算成功: strCoil={matched_result.get('strCoil')}")
                else:
                    print(
                        "[管线钢冷却智能体] 冷却 DLL 未完成计算，返回原因: "
                        f"{result}; strCoil={matched_result.get('strCoil')}, "
                        f"STEEL_SIGN={row_for_log.get('STEEL_SIGN')}"
                    )
                return result
            finally:
                _os.chdir(old_cwd)
    except Exception as exc:
        print(
            f"[管线钢冷却智能体] 冷却 DLL 调用异常: {type(exc).__name__}: {exc}; "
            f"strCoil={matched_result.get('strCoil')}, STEEL_SIGN={row_for_log.get('STEEL_SIGN')}"
        )
        return None


def _build_pipeline_cooling_timing_context(matched_result: dict) -> str:
    """提取最后有效轧制道次时刻，供冷却智能体动态设计开冷时间。

    优先使用 R_PASS_ACT+F_PASS_ACT 定位末道；计数字段异常时，再按从 N1
    开始连续为正的出口厚度识别有效道次。这里只生成判断上下文，不修改轧制
    道次时间，避免冷却阶段越权改写已经校验通过的轧制规程。
    """
    row = _matched_result_body_to_row(matched_result)

    def _as_positive_int(value) -> int:
        number = _to_float(value)
        if number is None or number <= 0 or abs(number - round(number)) > 1e-9:
            return 0
        return int(round(number))

    declared_total = (
        _as_positive_int(row.get("R_PASS_ACT"))
        + _as_positive_int(row.get("F_PASS_ACT"))
    )
    active_count = 0
    if 1 <= declared_total <= 30:
        declared_thickness = _to_float(row.get(f"N{declared_total}_DH_CAL"))
        if declared_thickness is not None and declared_thickness > 0:
            active_count = declared_total

    if active_count == 0:
        for pass_index in range(1, 31):
            thickness = _to_float(row.get(f"N{pass_index}_DH_CAL"))
            if thickness is None or thickness <= 0:
                break
            active_count = pass_index

    if active_count == 0:
        return "未识别到有效轧制道次；请保守保持 TIME_ENTR，并在 judgement 中说明依据。"

    last_pass_field = f"N{active_count}_ENTR_DATE"
    last_pass_value = str(row.get(last_pass_field) or "").strip()
    cooling_start_value = str(row.get("TIME_ENTR") or "").strip()
    last_pass_time = _parse_pipeline_process_datetime(last_pass_value)
    cooling_start_time = _parse_pipeline_process_datetime(cooling_start_value)
    if last_pass_time is None or cooling_start_time is None:
        interval_text = "当前时间字段无法解析，调整时必须保持原日期时间格式"
    else:
        interval_seconds = (cooling_start_time - last_pass_time).total_seconds()
        interval_text = f"当前轧后至开冷间隔={interval_seconds:.3f}秒"

    return (
        f"最后有效轧制道次=N{active_count}；"
        f"{last_pass_field}={last_pass_value or '（空）'}；"
        f"当前 TIME_ENTR={cooling_start_value or '（空）'}；{interval_text}。"
    )


def _build_pipeline_cooling_agent_user_prompt(
    context: str,
    rag_context: str,
    matched_result: dict,
) -> str:
    """构建每轮传给 Qwen 的冷却智能体用户提示词。

    相组成.png、CCT.png 和强化机制.PNG 不再以 base64 文本拼入，而是作为
    多模态图片随消息一起传入。
    """
    performance_context = _pipeline_agent_performance_context(matched_result)
    context, rag_context, llm_matched_result = _prepare_pipeline_agent_llm_prompt_data(
        context,
        rag_context,
        matched_result,
    )
    timing_context = _build_pipeline_cooling_timing_context(matched_result)
    matched_result_json = json.dumps(llm_matched_result, ensure_ascii=False)
    return build_pipeline_cooling_agent_user_prompt_text(
        context, rag_context, matched_result_json, timing_context, performance_context
    )


def _pipeline_user_explicitly_requests_high_self_temp(context: str) -> bool:
    """只从本轮用户原始需求判断是否明确要求返红温度达到或超过500℃。"""
    text = str(context or "")
    original_match = re.search(
        r"【用户原始需求】\s*(.*?)(?=\n\s*【|\Z)",
        text,
        flags=re.DOTALL,
    )
    if original_match:
        text = original_match.group(1)

    self_temp_label = r"(?:(?:返红|反红)温度|SELF_TEMP|SELF_TRMP)"
    # “不低于500℃”同时含有“低于”字样，因此明确的高温下限必须优先判断。
    comparator_pattern = (
        self_temp_label + r"[^。；\n]{0,16}"
        r"(?:大于或等于|大于等于|高于或等于|不低于|至少|达到|≥|>)\s*500(?:\.0+)?\s*(?:℃|度)?"
    )
    if re.search(comparator_pattern, text, flags=re.IGNORECASE):
        return True

    # “500℃以下/不超过500℃”是上限要求，不能被数值500误识别为高温例外。
    upper_limit_pattern = (
        self_temp_label + r"[^。；\n]{0,16}"
        r"(?:小于|低于|不高于|不超过|至多|上限|以下|<|≤)\s*500(?:\.0+)?\s*(?:℃|度)?"
    )
    if re.search(upper_limit_pattern, text, flags=re.IGNORECASE):
        return False

    assignment_pattern = (
        self_temp_label + r"\s*"
        r"(?:(?:设置|设定|调整)?\s*(?:为|到|至)|控制在|范围(?:为|是)?|[=:：])?\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:℃|度)?"
    )
    for value in re.findall(assignment_pattern, text, flags=re.IGNORECASE):
        number = _to_float(value)
        if number is not None and number >= 500.0:
            return True
    return False


def _set_pipeline_arrbody_field(matched_result: dict, field_name: str, value) -> bool:
    """在不改变 arrBody 长度、顺序和字段名的前提下覆盖一个已有字段。"""
    target = str(field_name or "").upper()
    for item in matched_result.get("arrBody") or []:
        key = _get_arrbody_key(item)
        if key and str(key).upper() == target:
            item[key] = value
            return True
    return False


def _format_pipeline_process_datetime(value: _datetime) -> str:
    """按智能体约定格式输出毫秒级工艺时刻。"""
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _stabilize_pipeline_cooling_timing(matched_result: dict, context: str = "") -> dict:
    """在冷却阶段确定性收敛末道至开冷时间关系。

    轧制智能体负责道次和道次时间，冷却智能体负责 ``TIME_ENTR``。本函数先
    使用最终的 R/F 道次数定位真实末道；道次时间完整时尽量保持原值，仅修正
    不可行的开冷时刻。时间缺失、逆序或粗精轧分界无有效待温时，才以首个可用
    时刻为锚点重建全部有效道次时间。任何情况下都保持中间坯待温时间不小于
    终轧到开冷时间，且不修改其他工艺字段。
    """
    stabilized = copy.deepcopy(matched_result)
    row = _matched_result_body_to_row(stabilized)
    rough_number = _to_float(row.get("R_PASS_ACT"))
    finish_number = _to_float(row.get("F_PASS_ACT"))
    if rough_number is None or finish_number is None:
        return stabilized
    if (
        abs(rough_number - round(rough_number)) > 1e-9
        or abs(finish_number - round(finish_number)) > 1e-9
    ):
        return stabilized
    rough_passes = int(round(rough_number))
    total_passes = rough_passes + int(round(finish_number))
    if rough_passes < 1 or total_passes <= rough_passes or total_passes > 30:
        return stabilized

    time_fields = [f"N{index}_ENTR_DATE" for index in range(1, total_passes + 1)]
    parsed_times = [_parse_pipeline_process_datetime(row.get(field)) for field in time_fields]
    times_are_increasing = all(
        previous is not None and current is not None and current > previous
        for previous, current in zip(parsed_times, parsed_times[1:])
    )
    has_all_times = all(value is not None for value in parsed_times)
    rough_end = parsed_times[rough_passes - 1] if has_all_times else None
    finish_start = parsed_times[rough_passes] if has_all_times else None
    intermediate_wait = (
        (finish_start - rough_end).total_seconds()
        if rough_end is not None and finish_start is not None
        else 0.0
    )

    # 珠光体钢倾向较长轧后等待；其他高强风电/管线设计默认按贝氏体类短等待
    # 处理。无论目标组织如何，最终开冷间隔都不能超过中间坯待温时间。
    context_text = str(context or "")
    pearlitic = "珠光体" in context_text and not any(
        marker in context_text for marker in ("贝氏体", "针状铁素体", "粒状贝氏体", "板条贝氏体")
    )
    desired_wait_seconds = 120.0 if pearlitic else 60.0

    if not has_all_times or not times_are_increasing or intermediate_wait <= 0.0:
        anchor = next((value for value in parsed_times if value is not None), None)
        if anchor is None:
            cooling_anchor = _parse_pipeline_process_datetime(row.get("TIME_ENTR"))
            anchor = cooling_anchor - _timedelta(seconds=desired_wait_seconds + 5 * total_passes) \
                if cooling_anchor is not None else _datetime.now().replace(microsecond=0)
        rebuilt_times: list[_datetime] = []
        current_time = anchor
        for pass_index in range(1, total_passes + 1):
            if pass_index > 1:
                gap_seconds = desired_wait_seconds if pass_index == rough_passes + 1 else 5.0
                current_time += _timedelta(seconds=gap_seconds)
            rebuilt_times.append(current_time)
            _set_pipeline_arrbody_field(
                stabilized,
                f"N{pass_index}_ENTR_DATE",
                _format_pipeline_process_datetime(current_time),
            )
        parsed_times = rebuilt_times
        intermediate_wait = desired_wait_seconds
        print(
            "[冷却时序确定性修复] 已按最终 R_PASS_ACT/F_PASS_ACT 重建有效道次时间，"
            f"中间坯待温={intermediate_wait:.3f}s"
        )

    last_pass_time = parsed_times[-1]
    existing_cooling_start = _parse_pipeline_process_datetime(row.get("TIME_ENTR"))
    existing_delay = (
        (existing_cooling_start - last_pass_time).total_seconds()
        if existing_cooling_start is not None and last_pass_time is not None
        else -1.0
    )
    organization_delay_ok = existing_delay > 100.0 if pearlitic else 0.0 <= existing_delay < 100.0
    if (
        existing_cooling_start is None
        or existing_delay < 0.0
        or existing_delay > intermediate_wait + 1e-9
        or not organization_delay_ok
    ):
        if pearlitic and intermediate_wait > 100.0:
            cooling_delay = min(desired_wait_seconds, intermediate_wait)
        else:
            cooling_delay = min(60.0, intermediate_wait)
        cooling_delay = max(0.0, cooling_delay)
        corrected_cooling_start = last_pass_time + _timedelta(seconds=cooling_delay)
        _set_pipeline_arrbody_field(
            stabilized,
            "TIME_ENTR",
            _format_pipeline_dll_datetime(corrected_cooling_start),
        )
        print(
            "[冷却时序确定性修复] 已按真实末道次修正 TIME_ENTR: "
            f"末道=N{total_passes}, 终轧到开冷={cooling_delay:.3f}s, "
            f"中间坯待温={intermediate_wait:.3f}s"
        )
    elif existing_cooling_start is not None:
        # 时刻关系已经合理时也必须规范为冷却 DLL 的紧凑编码；此前保留
        # yyyy-MM-dd HH:mm:ss.fff 会在 DLL 的 ParseExact 处直接终止。
        _set_pipeline_arrbody_field(
            stabilized,
            "TIME_ENTR",
            _format_pipeline_dll_datetime(existing_cooling_start),
        )
    return stabilized


def _collect_pipeline_strict_cooling_gate_errors(matched_result: dict) -> list[dict]:
    """返回微调和最终冷却共用的字段级严格门禁错误。"""
    row = _matched_result_body_to_row(matched_result)
    errors: list[dict] = []
    fdt = _to_float(row.get("FDT"))
    temp_entr = _to_float(row.get("TEMP_ENTR"))
    self_temp = _to_float(row.get("SELF_TEMP"))
    current_temperatures = {
        "FDT": row.get("FDT"),
        "TEMP_ENTR": row.get("TEMP_ENTR"),
        "SELF_TEMP": row.get("SELF_TEMP"),
    }
    if fdt is None or temp_entr is None or self_temp is None:
        missing = [
            name for name, value in (
                ("FDT", fdt), ("TEMP_ENTR", temp_entr), ("SELF_TEMP", self_temp)
            ) if value is None
        ]
        errors.append({
            "module": "cooling",
            "field": missing[0] if len(missing) == 1 else "FDT/TEMP_ENTR/SELF_TEMP",
            "rule": "FDT > TEMP_ENTR > SELF_TEMP",
            "status": "FAIL",
            "current_values": current_temperatures,
            "message": "冷却温度字段缺失或不是合法数值：" + "、".join(missing),
        })
    else:
        if fdt <= temp_entr:
            errors.append({
                "module": "cooling",
                "field": "TEMP_ENTR",
                "rule": "FDT > TEMP_ENTR > SELF_TEMP",
                "status": "FAIL",
                "current_values": current_temperatures,
                "message": "入水温度必须严格低于终轧温度",
            })
        if temp_entr <= self_temp:
            errors.append({
                "module": "cooling",
                "field": "SELF_TEMP",
                "rule": "FDT > TEMP_ENTR > SELF_TEMP",
                "status": "FAIL",
                "current_values": current_temperatures,
                "message": "返红温度必须严格低于入水温度",
            })

    rough = _to_float(row.get("R_PASS_ACT"))
    finish = _to_float(row.get("F_PASS_ACT"))
    total = (
        int(round(rough)) + int(round(finish))
        if rough is not None and finish is not None
        else 0
    )
    last_field = f"N{total}_ENTR_DATE" if 1 <= total <= 30 else ""
    last_time = _parse_pipeline_process_datetime(row.get(last_field)) if last_field else None
    cooling_time = _parse_pipeline_process_datetime(row.get("TIME_ENTR"))
    if last_time is None or cooling_time is None or cooling_time <= last_time:
        errors.append({
            "module": "cooling",
            "field": "TIME_ENTR",
            "rule": "TIME_ENTR > last active pass ENTR_DATE",
            "status": "FAIL",
            "current_values": {
                "TIME_ENTR": row.get("TIME_ENTR"),
                "last_pass_field": last_field,
                "last_pass_time": row.get(last_field) if last_field else None,
            },
            "message": "开冷时刻必须采用合法编码并严格晚于最后有效轧制道次",
        })
    return errors


def _require_valid_pipeline_cooling_timing(matched_result: dict) -> dict:
    """冷却出口硬门禁：使用最终 TIME_ENTR 校验完整轧制/开冷时间关系。"""
    normalized, validation_error = _normalize_pipeline_deformation_passes(
        matched_result,
        "控制冷却智能体最终时序门禁",
        validate_timing=True,
        validate_cooling_timing=True,
    )
    if normalized is None:
        raise PipelineRollValidationError(
            "冷却时序确定性修复后仍未通过最终门禁："
            + (validation_error or "未知轧制/开冷时间错误")
        )
    strict_errors = _collect_pipeline_strict_cooling_gate_errors(normalized)
    if strict_errors:
        raise PipelineRollValidationError(
            "最终冷却严格门禁未通过："
            + json.dumps(strict_errors, ensure_ascii=False)
        )
    return normalized




def _collect_pipeline_cooling_simulation_context(matched_result: dict) -> tuple[str, str, str]:
    """读取相组成、动态 CCT 和强化机制图片，并转换为多模态输入。"""
    coil_id = str(matched_result.get("strCoil", "")).strip()
    coil_dir = _os.path.join(PIPELINE_IMAGE_GENERATOR_BIN_DIR, "ModelManage", coil_id)
    phase_composition_path = _find_first_file_under_dir(coil_dir, "相组成.png")
    cct_path = _find_first_file_under_dir(coil_dir, "CCT.png")
    strengthening_mechanism_path = _find_first_file_under_dir(coil_dir, "强化机制.PNG")
    phase_composition_base64 = _read_image_base64_for_prompt(phase_composition_path, "相组成.png")
    cct_base64 = _read_image_base64_for_prompt(cct_path, "CCT.png")
    strengthening_mechanism_base64 = _read_image_base64_for_prompt(
        strengthening_mechanism_path,
        "强化机制.PNG",
    )
    return phase_composition_base64, cct_base64, strengthening_mechanism_base64


def _invoke_qwen_cooling_agent(user_prompt: str, images: list[tuple[str, str]]) -> dict:
    """调用冷却判断模型；结构修复和结果缓存由外层同轮重试逻辑负责。"""
    print(f"[管线钢冷却智能体] 开始调用判断模型，prompt长度={len(user_prompt)}, 图片数={len(images)}")
    return _invoke_pipeline_qwen_json(
        PIPELINE_COOLING_AGENT_SYSTEM_PROMPT.replace("管线钢", get_wind_power_material_label(user_prompt))
        if WIND_POWER_PROMPT_CONTEXT_TAG in user_prompt else PIPELINE_COOLING_AGENT_SYSTEM_PROMPT,
        user_prompt,
        images,
        "管线钢冷却智能体",
    )


def _extract_revision_parent_baseline_from_context(context: str) -> dict:
    """从智能体上下文读取父方案关键值，不依赖身份字段或完整历史报告。"""
    marker = "【父方案关键成分、规格与性能基线】"
    text = str(context or "")
    marker_index = text.find(marker)
    if marker_index < 0:
        return {}
    json_start = text.find("{", marker_index + len(marker))
    if json_start < 0:
        return {}
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[json_start:])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _sanitize_pipeline_cooling_agent_result(
    original: dict,
    candidate: dict,
    context: str = "",
) -> dict | tuple[None, str] | None:
    """校验冷却结果；开冷时刻或温度变化时同步性能并保护硬约束。"""
    if not isinstance(original, dict) or not isinstance(candidate, dict):
        return None
    if list(candidate.keys()) != list(original.keys()):
        print("[管线钢冷却智能体] Qwen返回顶层键不一致，放弃本轮结果")
        return None

    original_body = original.get("arrBody")
    candidate_body = candidate.get("arrBody")
    if not isinstance(original_body, list) or not isinstance(candidate_body, list):
        return None
    if len(original_body) != len(candidate_body):
        print("[管线钢冷却智能体] Qwen返回 arrBody 长度不一致，放弃本轮结果")
        return None

    sanitized = copy.deepcopy(original)
    candidate_state = _coerce_reheat_is_state(candidate.get("isState", original.get("isState")))
    sanitized["isState"] = candidate_state if isinstance(candidate_state, bool) else original.get("isState")
    attempted_performance_change = _pipeline_agent_attempted_performance_change(original, candidate)

    sanitized_body = []
    for original_item, candidate_item in zip(original_body, candidate_body):
        original_key = _get_arrbody_key(original_item)
        candidate_key = _get_arrbody_key(candidate_item)
        if not original_key or original_key != candidate_key:
            print("[管线钢冷却智能体] Qwen返回 arrBody 字段顺序或字段名不一致，放弃本轮结果")
            return None

        original_value = _get_arrbody_value(original_item)
        candidate_value = _get_arrbody_value(candidate_item)
        field_name = original_key.upper()
        if field_name in {"TIME_ENTR", "TEMP_ENTR", "SELF_TEMP", "FURNACE_EXIT_TIME"}:
            sanitized_body.append({original_key: candidate_value})
        elif re.fullmatch(r"N(?:[1-9]|[12]\d|30)_ENTR_DATE", field_name):
            # 冷却阶段只接受后端时间编码校验触发的格式修正；道次时间关系
            # 仍由轧制智能体和最终时序门禁负责校验。
            sanitized_body.append({original_key: candidate_value})
        elif field_name in PIPELINE_PERFORMANCE_FIELDS and sanitized["isState"] is False:
            sanitized_body.append({
                original_key: _resolve_pipeline_agent_performance_value(
                    original,
                    field_name,
                    candidate_value,
                )
            })
        else:
            sanitized_body.append({original_key: original_value})

    sanitized["arrBody"] = sanitized_body

    # 续改任务要求“性能不降低”时，父方案四项性能会以精简JSON进入上下文。
    # 在同一轮结构校验中返回具体错误，外层会复用当前DLL图片要求模型修复，
    # 不会把不合格结果带到最终报告，也不会回退SQL历史性能。
    revision_parent = _extract_revision_parent_baseline_from_context(context)
    if revision_parent:
        current_performance = _matched_result_body_to_row(sanitized)
        relative_errors = []
        for field_name in ("YS", "TS", "EL", "AKV"):
            parent_value = _to_float(revision_parent.get(field_name))
            current_value = _to_float(current_performance.get(field_name))
            if parent_value is not None and (
                current_value is None or current_value < parent_value - 1e-9
            ):
                relative_errors.append(
                    f"{field_name}必须不低于父方案{parent_value:g}，"
                    f"当前为{current_performance.get(field_name)!r}"
                )
        if relative_errors:
            return None, "；".join(relative_errors)

    # 500℃是默认冷却工艺上限。当前用户没有明确指定高温返红时，模型不得
    # 通过历史值或误判把 SELF_TEMP>=500℃ 带入下一轮和最终报告。
    sanitized_row = _matched_result_body_to_row(sanitized)
    if not _pipeline_user_explicitly_requests_high_self_temp(context):
        self_temp = _to_float(sanitized_row.get("SELF_TEMP"))
        if self_temp is None or self_temp >= 500.0:
            print(
                "[管线钢冷却智能体] SELF_TEMP后校验失败: "
                f"当前用户未明确要求高温返红，模型返回 SELF_TEMP={sanitized_row.get('SELF_TEMP')!r}"
            )
            return None

    strict_cooling_errors = _collect_pipeline_strict_cooling_gate_errors(sanitized)
    if strict_cooling_errors:
        error_json = json.dumps(strict_cooling_errors, ensure_ascii=False)
        print(f"[管线钢冷却智能体] 严格冷却门禁未通过: {error_json}")
        return None, error_json

    # TIME_ENTR 允许由冷却智能体修改，但不得早于最后一个有效轧制道次。
    # 只有模型实际修改 TIME_ENTR 时才执行该格式与顺序门禁，避免历史数据格式
    # 差异阻断原本无需调整开冷时刻的合格方案。
    sanitized_row = _matched_result_body_to_row(sanitized)
    original_row = _matched_result_body_to_row(original)
    original_time_entr = str(original_row.get("TIME_ENTR") or "").strip()
    candidate_time_entr = str(sanitized_row.get("TIME_ENTR") or "").strip()
    if candidate_time_entr != original_time_entr:
        rough_passes = _to_float(sanitized_row.get("R_PASS_ACT"))
        finish_passes = _to_float(sanitized_row.get("F_PASS_ACT"))
        if rough_passes is None or finish_passes is None:
            print("[管线钢冷却智能体] TIME_ENTR调整校验失败: 缺少有效道次数")
            return None
        total_passes = rough_passes + finish_passes
        if (
            total_passes < 1
            or total_passes > 30
            or abs(total_passes - round(total_passes)) > 1e-9
        ):
            print(
                "[管线钢冷却智能体] TIME_ENTR调整校验失败: "
                f"R_PASS_ACT+F_PASS_ACT={total_passes!r} 不是1至30的整数"
            )
            return None
        last_pass_index = int(round(total_passes))
        last_pass_field = f"N{last_pass_index}_ENTR_DATE"
        last_pass_time = _parse_pipeline_process_datetime(sanitized_row.get(last_pass_field))
        cooling_start_time = _parse_pipeline_process_datetime(candidate_time_entr)
        if last_pass_time is None or cooling_start_time is None:
            print(
                "[管线钢冷却智能体] TIME_ENTR调整校验失败: "
                f"{last_pass_field}={sanitized_row.get(last_pass_field)!r}, "
                f"TIME_ENTR={candidate_time_entr!r}"
            )
            return None
        if cooling_start_time < last_pass_time:
            print(
                "[管线钢冷却智能体] TIME_ENTR调整校验失败: "
                f"TIME_ENTR={candidate_time_entr!r} 早于 "
                f"{last_pass_field}={sanitized_row.get(last_pass_field)!r}"
            )
            return None

    if (
        sanitized["isState"] is False
        and _pipeline_agent_has_meaningful_process_change(original, sanitized)
        and not attempted_performance_change
    ):
        print("[管线钢冷却智能体] 本轮调整了冷却工艺但未同步更新性能，放弃本轮结果")
        return None
    if sanitized.get("isState") is True:
        time_encoding_error = _validate_pipeline_dll_time_encodings(
            sanitized,
            include_cooling_start=True,
        )
        if time_encoding_error:
            print(f"[管线钢冷却智能体] 时间编码校验未通过: {time_encoding_error}")
            return None, time_encoding_error
    return sanitized


def _refine_pipeline_cooling_process_with_agent(
    matched_result: dict,
    context: str,
    reasoning_key_prefix: str | None = None,
    progress_callback=None,
) -> dict:
    """兼容原调用名；冷却智能体业务已迁移到 pipeline_agents.py。"""
    return refine_cooling_process(
        matched_result,
        context,
        reasoning_key_prefix,
        progress_callback=progress_callback,
        dependencies=_build_process_agent_dependencies(),
    )


def _list_png_images(image_dir: str) -> dict[str, str]:
    """返回目录中的 PNG 文件名到绝对路径映射。"""
    if not _os.path.isdir(image_dir):
        return {}
    return {
        image_name: _os.path.join(image_dir, image_name)
        for image_name in _os.listdir(image_dir)
        if image_name.lower().endswith(".png")
    }


def _image_dir_mtime(image_dir: str) -> float:
    """取图片目录及 PNG 文件的最新修改时间，用于判断是否属于本次生成。"""
    mtimes = []
    if _os.path.isdir(image_dir):
        mtimes.append(_os.path.getmtime(image_dir))
        for image_path in _list_png_images(image_dir).values():
            try:
                mtimes.append(_os.path.getmtime(image_path))
            except OSError:
                continue
    return max(mtimes) if mtimes else 0.0


def _find_generated_image_dir(expected_image_dir: str, generated_after: float) -> str | None:
    """优先使用匹配卷号目录；若 DLL 写到其他新目录，则查找本次生成后的 Image 目录。"""
    if _list_png_images(expected_image_dir):
        return expected_image_dir

    root_dir = _os.path.join(IMAGE_GENERATOR_BIN_DIR, "Physical_Matel_RCL")
    if not _os.path.isdir(root_dir):
        return None

    recent_dirs = []
    for current_dir, dir_names, _file_names in _os.walk(root_dir):
        if _os.path.basename(current_dir) != "Image":
            continue
        png_images = _list_png_images(current_dir)
        if not png_images:
            continue
        latest_mtime = _image_dir_mtime(current_dir)
        if latest_mtime >= generated_after - 5:
            recent_dirs.append((latest_mtime, current_dir))
        dir_names[:] = []

    if not recent_dirs:
        return None
    recent_dirs.sort(reverse=True)
    fallback_dir = recent_dirs[0][1]
    print(f"[报告生成] 使用本次生成图片兜底目录: {fallback_dir}")
    return fallback_dir


def _find_pipeline_generated_image_dir(expected_image_dir: str, generated_after: float) -> str | None:
    """优先使用管线钢匹配号目录；若 DLL 写到其他新目录，则查找 ModelManage 下本次生成后的 Image 目录。"""
    if _list_png_images(expected_image_dir):
        return expected_image_dir

    root_dir = _os.path.join(PIPELINE_IMAGE_GENERATOR_BIN_DIR, "ModelManage")
    if not _os.path.isdir(root_dir):
        return None

    recent_dirs = []
    for current_dir, dir_names, _file_names in _os.walk(root_dir):
        if _os.path.basename(current_dir) != "Image":
            continue
        png_images = _list_png_images(current_dir)
        if not png_images:
            continue
        latest_mtime = _image_dir_mtime(current_dir)
        if latest_mtime >= generated_after - 5:
            recent_dirs.append((latest_mtime, current_dir))
        dir_names[:] = []

    if not recent_dirs:
        return None
    recent_dirs.sort(reverse=True)
    fallback_dir = recent_dirs[0][1]
    print(f"[管线钢报告生成] 使用本次生成图片兜底目录: {fallback_dir}")
    return fallback_dir


def _parse_json_object(text: str) -> dict | None:
    """从 LLM 文本中提取 JSON 对象。"""
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidate = match.group(1).strip() if match else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _markdown_code_fence_closer(markdown_text: str) -> str:
    """如果报告正文存在未闭合的 ``` 代码块，追加关闭围栏，避免后续图片被当代码显示。"""
    if not markdown_text:
        return ""
    fence_count = len(re.findall(r"(^|\n)\s*```", markdown_text))
    return "\n\n```\n" if fence_count % 2 == 1 else ""


def _get_recent_session_context(session_id: str) -> str:
    """读取当前会话最近几条消息，供 LLM 扩大范围时参考。"""
    try:
        session = chat_session_store.get_or_create(session_id)
        messages = session.get("messages", [])[-6:]
        return "\n".join([
            getattr(message, "content", "")
            for message in messages
            if getattr(message, "content", "")
        ])
    except Exception:
        return ""


def _format_recent_store_context(store: SessionStore, session_id: str, limit: int = 6) -> str:
    """把指定上下文池最近消息整理成短文本，供跨路由追问使用。"""
    try:
        messages = store.get_messages(session_id)[-limit:]
    except Exception:
        return ""
    lines = []
    for message in messages:
        content = getattr(message, "content", "")
        if not content:
            continue
        content = _drop_sensitive_context_lines(content)
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _drop_sensitive_context_lines(text: str) -> str:
    """丢弃历史上下文中包含身份追溯字段名的行。"""
    sensitive_keys = {
        "strCoil",
        "strSteel",
        *SENSITIVE_MATCHED_RESULT_FIELDS,
        *SENSITIVE_MATCHED_RESULT_TOP_LEVEL_FIELDS,
    }
    safe_lines = []
    for line in (text or "").splitlines():
        if any(key in line for key in sensitive_keys):
            continue
        safe_lines.append(line)
    return "\n".join(safe_lines)


def _build_cross_route_context(session_id: str) -> str:
    """汇总文本框普通对话、Chat Agent 和报告生成上下文。"""
    context_parts = []
    chat_context = _format_recent_store_context(chat_session_store, session_id)
    if chat_context:
        context_parts.append(f"【文本框普通对话上下文】\n{chat_context}")
    agent_context = _format_recent_store_context(agent_chat_store, session_id)
    if agent_context:
        context_parts.append(f"【Chat Agent 会话上下文】\n{agent_context}")
    report_context = _format_recent_store_context(report_session_store, session_id, limit=4)
    if report_context:
        context_parts.append(f"【报告生成上下文】\n{report_context}")
    return "\n\n".join(context_parts)


def _is_context_lookup_chat_request(user_message: str) -> bool:
    """识别“不要重新设计，只从当前上下文取结果”的追问。"""
    message = user_message or ""
    no_redesign = any(keyword in message for keyword in [
        "不设计", "不重新设计", "不要重新设计", "无需重新设计", "别重新设计", "不要再设计",
    ])
    context_ref = any(keyword in message for keyword in [
        "当前设计", "这条设计", "刚刚设计", "刚才设计", "上下文", "已有设计", "上一次设计",
    ])
    rolling_schedule = any(keyword in message for keyword in [
        "轧制规程", "各道次", "道次轧制", "道次规程", "压下", "轧制力", "轧制速度",
    ])
    return rolling_schedule and (no_redesign or context_ref)


def _deterministic_intent_override(user_message: str) -> str | None:
    """对含义明确的设计请求和知识问答做稳定路由，避免分类接口异常导致误判。"""
    message = re.sub(r"\s+", "", str(user_message or ""))
    if not message:
        return None
    if _is_context_lookup_chat_request(message):
        return "CHAT"

    # “设计一组/一种/一套”是明确的方案生成命令；这里不把“成分设计差异”
    # 这类作为名词使用的“设计”误判为设计任务。
    explicit_design_patterns = (
        r"(?:请|帮我|请帮我)?设计(?:一组|一种|一套|一个)",
        r"(?:重新设计|从新设计|再设计)(?:一组|一种|一套|一个|成分|工艺|方案)",
        r"(?:制定|生成|给出)(?:一组|一种|一套|一个)?[^。？！?]{0,20}(?:成分工艺|工艺方案|设计方案)",
        r"如何调整成分",
        r"什么工艺能达到",
        r"(?:模拟|预测)[^。？！?]{0,12}(?:性能|组织|工艺)",
    )
    if any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in explicit_design_patterns):
        return "DESIGN"

    # 明确询问差异、机理或参数协同关系时属于知识问答。该规则专门保护
    # 前端第4、5个快捷问题，避免“成分设计”“协同控制”触发 DESIGN。
    knowledge_question_patterns = (
        r"(?:差异|区别)(?:有)?(?:哪些|什么|在哪里|为何|为什么)",
        r"(?:如何|怎么|怎样)(?:协同|配合|共同)(?:控制|作用|影响)",
        r"(?:作用|机理|原理|影响规律)(?:是|有|为|有哪些|是什么|如何)",
    )
    if any(re.search(pattern, message, flags=re.IGNORECASE) for pattern in knowledge_question_patterns):
        return "CHAT"
    return None


def _deterministic_purpose_override(user_message: str) -> str | None:
    """明确用途的设计请求直接进入对应分支，避免二级分类断连降级。"""
    message = re.sub(r"\s+", "", str(user_message or ""))
    pipeline_markers = (
        "管线钢", "管线用钢", "油气输送管", "油气管线", "输送管线",
        "API管线", "TMCP管线",
    )
    if any(marker.lower() in message.lower() for marker in pipeline_markers):
        return "管线钢"
    wind_markers = (
        "风电用钢", "风电钢", "风电塔筒", "风机塔架", "风电塔架", "风电塔筒钢板",
    )
    if any(marker.lower() in message.lower() for marker in wind_markers):
        return "风电用钢"
    return None


_CHAT_TEXT_TOOL_ALIASES = {
    "engineering_machinery_wear_steel": "search_engineering_machinery_wear_steel_knowledge_base",
    "wear_steel": "search_engineering_machinery_wear_steel_knowledge_base",
    "工程机械耐磨钢": "search_engineering_machinery_wear_steel_knowledge_base",
    "pipeline_steel": "search_pipeline_steel_knowledge_base",
    "管线钢": "search_pipeline_steel_knowledge_base",
    "offshore_steel": "search_offshore_steel_knowledge_base",
    "海工钢": "search_offshore_steel_knowledge_base",
    "building_steel": "search_building_steel_knowledge_base",
    "建筑钢": "search_building_steel_knowledge_base",
    "structural_steel": "search_structural_steel_knowledge_base",
    "结构钢": "search_structural_steel_knowledge_base",
    "wind_power_steel": "search_wind_power_steel_knowledge_base",
    "wind_steel": "search_wind_power_steel_knowledge_base",
    "风电用钢": "search_wind_power_steel_knowledge_base",
    "风电钢": "search_wind_power_steel_knowledge_base",
    "automotive_steel": "search_automotive_steel_knowledge_base",
    "汽车钢": "search_automotive_steel_knowledge_base",
    "bridge_steel": "search_bridge_steel_knowledge_base",
    "桥梁钢": "search_bridge_steel_knowledge_base",
}


def _parse_chat_json_tool_call(
    text: str,
    user_message: str,
    tool_map: dict,
) -> tuple[str, str] | None:
    """兼容模型以正文 JSON 返回工具选择的情况，并统一为真实 LangChain 工具名。"""
    payload = _parse_json_object(str(text or ""))
    if not isinstance(payload, dict):
        return None
    raw_tool_name = str(
        payload.get("tool")
        or payload.get("tool_name")
        or payload.get("name")
        or ""
    ).strip()
    if not raw_tool_name:
        return None
    tool_name = _CHAT_TEXT_TOOL_ALIASES.get(raw_tool_name, raw_tool_name)
    if tool_name not in tool_map:
        return None
    query = str(
        payload.get("query")
        or payload.get("input")
        or payload.get("question")
        or user_message
        or ""
    ).strip()
    return tool_name, query or str(user_message or "")


def _is_context_based_design_modification_request(user_message: str) -> bool:
    """识别“基于上一轮/上下文设计继续调整”的重新设计请求。"""
    message = user_message or ""
    context_ref = any(keyword in message for keyword in [
        "基于以上", "基于上述", "以上设计", "上述设计", "以上方案", "上述方案",
        "上一轮", "上一次", "前一轮", "前一次", "刚刚设计", "刚才设计",
        "当前设计", "这条设计", "已有设计", "参考上下文", "上下文",
    ])
    modification = any(keyword in message for keyword in [
        "调整", "微调", "调低", "调高", "降低", "提高", "下调", "上调",
        "减少", "增加", "优化", "修改", "重新设计", "从新设计", "再设计",
    ])
    return context_ref and modification


def _get_arrbody_key(item: dict) -> str | None:
    """读取 arrBody 中单键对象的字段名；格式不正确时返回 None。"""
    if not isinstance(item, dict) or len(item) != 1:
        return None
    return next(iter(item.keys()))


def _get_arrbody_value(item: dict):
    """读取 arrBody 中单键对象的字段值；格式不正确时返回 None。"""
    if not isinstance(item, dict) or len(item) != 1:
        return None
    return next(iter(item.values()))


def _spec_bounds_for_matched_field(field_name: str, spec_result: dict) -> tuple[float | None, float | None]:
    """把 Oracle 字段名映射到 spec_result 中对应的上下限。"""
    upper_name = str(field_name).upper()
    if upper_name in ROLLING_SCHEDULE_FIELDS:
        return None, None
    if upper_name in {"AIM_HEIGHT", "MAT_ACT_THICK_RCL"}:
        return _to_float(spec_result.get("THK_min")), _to_float(spec_result.get("THK_max"))
    if upper_name in REFINABLE_COMPONENT_FIELDS:
        spec_prefix = upper_name
    else:
        spec_prefix = REFINABLE_PROCESS_FIELD_TO_SPEC.get(upper_name)
    if not spec_prefix:
        return None, None
    return (
        _to_float(spec_result.get(f"{spec_prefix}_min")),
        _to_float(spec_result.get(f"{spec_prefix}_max")),
    )


def _value_within_spec_bounds(field_name: str, value, spec_result: dict) -> bool:
    """判断 LLM 修改后的成分/工艺值是否仍在 spec_result 边界内。"""
    min_value, max_value = _spec_bounds_for_matched_field(field_name, spec_result)
    number = _to_float(value)
    if number is None:
        return False
    if _is_effective_min(min_value) and number < min_value:
        return False
    if _is_effective_max(max_value) and number > max_value:
        return False
    return True


def _extract_target_thickness_from_text(user_message: str) -> float | None:
    """从用户原始需求中提取明确的目标厚度，单位按 mm 处理。"""
    if not user_message:
        return None
    patterns = [
        r"(?:厚度|目标厚度|成品厚度|钢板厚度|板厚)\D{0,12}(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)\D{0,12}(?:厚度|目标厚度|成品厚度|钢板厚度|板厚)",
        # 覆盖“25mm厚的钢板”“25毫米厚钢”等工程需求中最常见的简写。
        # 板坯厚度短语已在上方先移除，因此不会把板坯尺寸误识别为成品厚度。
        r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)\s*厚(?:的)?",
        r"厚\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(user_message), flags=re.IGNORECASE)
        if match:
            return _to_float(match.group(1))
    return None


def _format_target_thickness(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _format_engineering_target_thickness_field(field_name: str, target_thickness: float) -> str:
    if str(field_name).upper() == "MAT_ACT_THICK_RCL":
        return _format_target_thickness(target_thickness * 1000)
    return _format_target_thickness(target_thickness)


def _sanitize_refined_matched_result(
    original: dict,
    candidate: dict,
    spec_result: dict,
    target_thickness: float | None = None,
    context_modification_override: bool = False,
) -> dict:
    """校验并修正 LLM 返回的 matched_result，确保结构和边界安全。"""
    if not isinstance(original, dict) or not isinstance(candidate, dict):
        return original
    if list(candidate.keys()) != list(original.keys()):
        print("[Oracle匹配] LLM微调结果顶层键不一致，已回退原始匹配结果")
        return original
    if candidate.get("isState") != original.get("isState"):
        print("[Oracle匹配] LLM微调结果修改了 isState，已回退原始匹配结果")
        return original

    original_body = original.get("arrBody")
    candidate_body = candidate.get("arrBody")
    if not isinstance(original_body, list) or not isinstance(candidate_body, list):
        return original
    if len(original_body) != len(candidate_body):
        print("[Oracle匹配] LLM微调结果 arrBody 长度不一致，已回退原始匹配结果")
        return original

    sanitized = {
        key: original[key]
        for key in original
        if key != "arrBody"
    }
    sanitized_body = []

    for original_item, candidate_item in zip(original_body, candidate_body):
        original_key = _get_arrbody_key(original_item)
        candidate_key = _get_arrbody_key(candidate_item)
        if not original_key or original_key != candidate_key:
            print("[Oracle匹配] LLM微调结果 arrBody 字段顺序或字段名不一致，已回退原始匹配结果")
            return original

        original_value = _get_arrbody_value(original_item)
        candidate_value = _get_arrbody_value(candidate_item)
        field_name = original_key.upper()

        # 非白名单字段一律恢复原值，保护性能、钢卷、速度、轧制力等数据。
        if field_name not in REFINABLE_FIELDS:
            sanitized_body.append({original_key: original_value})
            continue

        if target_thickness is not None and field_name in {"AIM_HEIGHT", "MAT_ACT_THICK_RCL", "F7_DH_AVG"}:
            sanitized_body.append({
                original_key: _format_engineering_target_thickness_field(field_name, target_thickness)
            })
            continue

        # 白名单字段即使在 isState=true 且原值非空时也允许微调，但必须满足 spec_result 边界；否则恢复原值。
        if _value_within_spec_bounds(field_name, candidate_value, spec_result):
            sanitized_body.append({original_key: _format_oracle_value(candidate_value)})
        else:
            print(f"[Oracle匹配] LLM微调字段 {field_name} 超出规格边界或非数值，已恢复原值")
            sanitized_body.append({original_key: original_value})

    sanitized["arrBody"] = sanitized_body
    return sanitized


def _build_refinement_rag_context(spec_result: dict, user_message: str, db_name: str) -> str:
    """使用 gcjxyg_Know_db 检索微调成分和工艺所需的参考资料。"""
    try:
        from hybrid_retriever import hybrid_search

        query = (
            f"{user_message} {spec_result.get('用途', '')} "
            f"厚度范围 {spec_result.get('THK_min')} {spec_result.get('THK_max')} "
            "耐磨钢 工程机械用钢 成分 工艺 温度 均热 FET FDT CT 淬火 回火 "
            "C SI MN P S N NB V TI AL ALS CU CR NI CO MO B "
            "性能要求 屈服强度 抗拉强度 断后伸长率"
        )
        docs = hybrid_search(query, k=8, db_name=db_name)
        if not docs:
            return ""
        print(f"[Oracle匹配] 微调RAG命中 {len(docs)} 条文档，db={db_name}")
        return "\n\n---\n\n".join([
            f"[来源: {doc.get('source', 'unknown')}]\n{doc.get('content', '')}"
            for doc in docs
        ])
    except Exception as exc:
        print(f"[Oracle匹配] 微调RAG失败: {exc}")
        return ""


def _refine_unstrict_matched_result_with_llm(
    spec_result: dict,
    matched_result: dict,
    user_message: str,
    session_id: str,
    db_name: str = "gcjxyg_Know_db",
) -> dict:
    """已有实绩时，让 LLM 在固定 JSON 结构内统一微调/格式化成分、厚度和工艺。"""
    if not matched_result.get("arrBody"):
        return matched_result

    target_thickness = _extract_target_thickness_from_text(user_message)
    context_modification_override = _is_context_based_design_modification_request(user_message)
    rag_context = _build_refinement_rag_context(spec_result, user_message, db_name)
    session_context = _build_cross_route_context(session_id) or _get_recent_session_context(session_id)
    prompt = build_unstrict_refinement_prompt(
        user_message, session_context, rag_context, spec_result, matched_result
    )
    try:
        raw = deepseek_Llm.invoke(prompt)
        text = getattr(raw, "content", raw)
        candidate = _parse_json_object(str(text))
        if not isinstance(candidate, dict):
            print("[Oracle匹配] LLM微调未返回JSON对象，已保留原始匹配结果")
            result = matched_result
        else:
            result = _sanitize_refined_matched_result(
                matched_result,
                candidate,
                spec_result,
                target_thickness=target_thickness,
                context_modification_override=context_modification_override,
            )
    except Exception as exc:
        print(f"[Oracle匹配] LLM微调失败: {exc}")
        result = matched_result

    # isState=false 时，strCoil 追加 _yyyyMMddHHmmssfff 时间戳，arrBody 中 IN_MAT_NO_RCL 同步更新
    if result.get("isState") is False and result.get("arrBody"):
        _t = time.time()
        ts = time.strftime("%Y%m%d%H%M%S", time.localtime(_t)) + f"{int(_t * 1000) % 1000:03d}"
        original_coil = result.get("strCoil", "")
        new_coil = f"{original_coil}_{ts}" if original_coil else ts
        result["strCoil"] = new_coil
        for item in result["arrBody"]:
            if isinstance(item, dict) and "IN_MAT_NO_RCL" in item:
                item["IN_MAT_NO_RCL"] = new_coil
                break
        print(f"[Oracle匹配] isState=false，strCoil 已追加时间戳: {new_coil}")

    return result


def _expand_spec_with_llm(spec_result: dict, user_message: str, session_id: str, attempt: int, last_spec: dict) -> dict | None:
    """让 LLM 在原规格基础上适度扩大范围，用于最后阶段迭代查询。"""
    session_context = _get_recent_session_context(session_id)
    prompt = build_oracle_expand_spec_prompt(
        user_message, session_context, spec_result, last_spec, attempt
    )
    try:
        raw = deepseek_Llm.invoke(prompt)
        text = getattr(raw, "content", raw)
        expanded = _parse_json_object(str(text))
        if not isinstance(expanded, dict):
            return None
        merged = dict(last_spec)
        component_range_keys = {
            f"{prefix}_{bound}"
            for prefix in PIPELINE_COMPONENT_FIELD_MAP
            for bound in ("min", "max")
        }
        for key in component_range_keys:
            if key in merged and key in expanded:
                merged[key] = expanded[key]
        return merged
    except Exception as exc:
        print(f"[Oracle匹配] LLM扩大规格失败: {exc}")
        return None


def match_engineering_steel_process(spec_result: dict, user_message: str, session_id: str) -> dict:
    """耐磨钢规格匹配入口：严格查询、逐级放宽、LLM迭代，最终返回包装 JSON。"""
    query_errors = []

    # 阶段1：严格使用厚度、成分、工艺、性能全部条件；命中则 isState=true。
    try:
        row = _query_first_oracle_row(
            spec_result,
            include_process=True,
            include_performance=True,
            stage_name="严格查询",
        )
        if row:
            matched_result = _build_match_response(row, is_state=True, session_id=session_id)
            return matched_result
    except Exception as exc:
        error = _format_match_error("严格查询失败", exc)
        query_errors.append(error)
        print(f"[Oracle匹配] {error}")

    # 阶段2：放开性能条件，保留厚度、成分、工艺；命中则 isState=false。
    try:
        row = _query_first_oracle_row(
            spec_result,
            include_process=True,
            include_performance=False,
            stage_name="放开性能查询",
        )
        if row:
            matched_result = _build_match_response(row, is_state=False, session_id=session_id)
            return matched_result
    except Exception as exc:
        error = _format_match_error("放开性能查询失败", exc)
        query_errors.append(error)
        print(f"[Oracle匹配] {error}")

    # 阶段3：继续放开工艺条件，仅保留厚度和成分；命中则 isState=false。
    try:
        row = _query_first_oracle_row(
            spec_result,
            include_process=False,
            include_performance=False,
            stage_name="放开工艺查询",
        )
        if row:
            matched_result = _build_match_response(row, is_state=False, session_id=session_id)
            return matched_result
    except Exception as exc:
        error = _format_match_error("放开工艺查询失败", exc)
        query_errors.append(error)
        print(f"[Oracle匹配] {error}")

    # 如果基础查询都因为驱动、连接或 SQL 异常失败，则不要伪装成无数据。
    if query_errors:
        return _build_match_response(
            None,
            is_state=False,
            session_id=session_id,
            message=f"Oracle匹配失败: {query_errors[0]}",
            error="; ".join(query_errors),
        )

    # 阶段4：LLM 逐轮扩大范围，最多 5 次；命中则 isState=false。
    current_spec = dict(spec_result)
    llm_errors = []
    for attempt in range(1, 6):
        expanded = _expand_spec_with_llm(spec_result, user_message, session_id, attempt, current_spec)
        if not expanded:
            continue
        current_spec = expanded
        llm_stage_queries = [
            (True, True, f"LLM第{attempt}轮扩大查询"),
            (True, False, f"LLM第{attempt}轮扩大后放开性能查询"),
            (False, False, f"LLM第{attempt}轮扩大后放开工艺查询"),
        ]
        for include_process, include_performance, stage_name in llm_stage_queries:
            try:
                row = _query_first_oracle_row(
                    current_spec,
                    include_process=include_process,
                    include_performance=include_performance,
                    stage_name=stage_name,
                )
                if row:
                    matched_result = _build_match_response(row, is_state=False, session_id=session_id)
                    return matched_result
            except Exception as exc:
                error = _format_match_error(f"{stage_name}失败", exc)
                llm_errors.append(error)
                print(f"[Oracle匹配] {error}")

    if llm_errors:
        return _build_match_response(
            None,
            is_state=False,
            session_id=session_id,
            message=f"Oracle匹配失败: {llm_errors[0]}",
            error="; ".join(llm_errors),
        )

    # 所有策略都没有命中时，去掉全部筛选条件取首条实绩，避免返回空匹配。
    try:
        row = _query_first_oracle_row_without_filters(stage_name="最终无筛选兜底查询")
        if row:
            matched_result = _build_match_response(row, is_state=False, session_id=session_id)
            return matched_result
    except Exception as exc:
        error = _format_match_error("最终无筛选兜底查询失败", exc)
        print(f"[Oracle匹配] {error}")
        return _build_match_response(
            None,
            is_state=False,
            session_id=session_id,
            message=f"Oracle匹配失败: {error}",
            error=error,
        )

    return _build_match_response(
        None,
        is_state=False,
        session_id=session_id,
        message="最终无筛选兜底未返回首条 Oracle 实绩，请检查实绩表是否为空",
    )


# ============================================================
# 管线钢规格匹配 MySQL 实绩数据
# ============================================================

PIPELINE_MYSQL_HOST = _os.environ.get("PIPELINE_MYSQL_HOST", "127.0.0.1")
PIPELINE_MYSQL_PORT = int(_os.environ.get("PIPELINE_MYSQL_PORT", "3306"))
PIPELINE_MYSQL_DATABASE = _os.environ.get("PIPELINE_MYSQL_DATABASE", "nansteel")
PIPELINE_MYSQL_USER = _os.environ.get("PIPELINE_MYSQL_USER", "")
PIPELINE_MYSQL_PASSWORD = _os.environ.get("PIPELINE_MYSQL_PASSWORD", "")
PIPELINE_MYSQL_TABLE = _os.environ.get(
    "PIPELINE_MYSQL_TABLE", "table_x80ng_read_valid"
)
PIPELINE_SORT_FIELD = "PROD_PLATE_DATE"

PIPELINE_COMPONENT_FIELD_MAP = {
    "C": "C", "SI": "Si", "MN": "Mn", "P": "P", "S": "S", "N": "N",
    "NB": "Nb", "V": "V", "TI": "Ti", "AL": "Alt", "ALS": "Als",
    "CU": "Cu", "CR": "Cr", "NI": "Ni", "CO": "Co", "MO": "Mo", "B": "B",
}
PIPELINE_PROCESS_FIELD_MAP = {
    "FET": "FET",
    "FDT": "FDT",
    "TEMP_ENTR": "TEMP_ENTR",
    "FEH": "FEH",
    "SELF_TEMP": "SELF_TEMP",
    "FURNACE_EXIT_TEMP": "FURNACE_EXIT_TEMP",
    "SLAB_THICK": "SLAB_THICK",
    "AIM_THICK": "AIM_THICK",
}
PIPELINE_PERFORMANCE_FIELD_MAP = {
    "YS": "YS",
    "TS": "TS",
    "EL": "EL",
    "AKV": "AKV",
}
PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC = {
    column_name.upper(): spec_prefix
    for spec_prefix, column_name in PIPELINE_COMPONENT_FIELD_MAP.items()
}
PIPELINE_REFINABLE_PROCESS_FIELD_TO_SPEC = {
    column_name.upper(): spec_prefix
    for spec_prefix, column_name in PIPELINE_PROCESS_FIELD_MAP.items()
}
PIPELINE_REFINABLE_PERFORMANCE_FIELD_TO_SPEC = {
    column_name.upper(): spec_prefix
    for spec_prefix, column_name in PIPELINE_PERFORMANCE_FIELD_MAP.items()
}
PIPELINE_REFINABLE_PASS_THICKNESS_FIELDS = {f"N{index}_DH_CAL" for index in range(1, 31)}
PIPELINE_REFINABLE_PASS_TEMPERATURE_FIELDS = {f"N{index}_DT_CAL" for index in range(1, 31)}
PIPELINE_REFINABLE_PASS_WIDTH_FIELDS = {f"N{index}_DW_CAL" for index in range(1, 31)}
PIPELINE_REFINABLE_PASS_FORCE_FIELDS = {f"N{index}_FORCE" for index in range(1, 31)}
PIPELINE_REFINABLE_PASS_SPEED_FIELDS = {f"N{index}_SPD" for index in range(1, 31)}
PIPELINE_REFINABLE_PASS_TIME_FIELDS = {f"N{index}_ENTR_DATE" for index in range(1, 31)}
PIPELINE_REFINABLE_PASS_COUNT_FIELDS = {
    "R_PASS_ACT", "F_PASS_ACT",
}
PIPELINE_REFINABLE_TURN_FIELDS = {
    "WIDTH_ROLL_START_REMARK", "WIDTH_ROLL_END_REMARK",
}
PIPELINE_REFINABLE_PASS_FIELDS = (
    PIPELINE_REFINABLE_PASS_THICKNESS_FIELDS
    | PIPELINE_REFINABLE_PASS_TEMPERATURE_FIELDS
    | PIPELINE_REFINABLE_PASS_WIDTH_FIELDS
    | PIPELINE_REFINABLE_PASS_FORCE_FIELDS
    | PIPELINE_REFINABLE_PASS_SPEED_FIELDS
    | PIPELINE_REFINABLE_PASS_TIME_FIELDS
)
PIPELINE_REFINABLE_ROLL_FIELDS = (
    {"FET", "FDT", "FURNACE_EXIT_TIME"}
    | PIPELINE_REFINABLE_PASS_COUNT_FIELDS
    | PIPELINE_REFINABLE_TURN_FIELDS
    | PIPELINE_REFINABLE_PASS_FIELDS
)
PIPELINE_REFINABLE_INITIAL_COOLING_FIELDS = {
    "TIME_ENTR", "TEMP_ENTR", "SELF_TEMP",
}
# 后置 Agent 负责成分、力学性能、轧制规程和冷却初值的协同设计。用户明确的
# AIM_THICK/SLAB_THICK 仍由后端锁定；加热参数继续交给加热智能体，冷却智能体
# 对这里形成的 TIME_ENTR/TEMP_ENTR/SELF_TEMP 保留最终调整权。
PIPELINE_REFINABLE_FIELDS = (
    set(PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC)
    | set(PIPELINE_REFINABLE_PERFORMANCE_FIELD_TO_SPEC)
    | PIPELINE_REFINABLE_ROLL_FIELDS
    | PIPELINE_REFINABLE_INITIAL_COOLING_FIELDS
)
PIPELINE_REFINABLE_TEMPERATURE_FIELDS = (
    {"FET", "FDT", "TEMP_ENTR", "SELF_TEMP", "FURNACE_EXIT_TEMP"}
    | PIPELINE_REFINABLE_PASS_TEMPERATURE_FIELDS
)
PIPELINE_REFINABLE_THICKNESS_FIELDS = (
    {"SLAB_THICK", "AIM_THICK"}
    | PIPELINE_REFINABLE_PASS_THICKNESS_FIELDS
    | PIPELINE_REFINABLE_PASS_WIDTH_FIELDS
    | PIPELINE_REFINABLE_PASS_SPEED_FIELDS
)
PIPELINE_REFINABLE_INTEGER_FIELDS = (
    PIPELINE_REFINABLE_PASS_FORCE_FIELDS
    | PIPELINE_REFINABLE_PASS_COUNT_FIELDS
    | PIPELINE_REFINABLE_TURN_FIELDS
)
PIPELINE_REFINABLE_PERFORMANCE_INTEGER_FIELDS = {"YS", "TS", "AKV"}
PIPELINE_REFINABLE_PERCENT_FIELDS = {"EL"}

PIPELINE_SELECT_COLUMNS = [
    "SLAB_ID", "STEEL_SIGN",
    "SLAB_THICK", "SLAB_WIDTH", "SLAB_LEN", "AIM_THICK", "AIM_WIDTH",
    # 转钢起止道次标识同时进入主匹配 matched_result 和10组历史轧制参考。
    # 两个查询共用本列清单，因此只需在这里维护一次，避免两条数据链不一致。
    "WIDTH_ROLL_START_REMARK", "WIDTH_ROLL_END_REMARK",
    "C", "Si", "Mn", "P", "S", "N", "Nb", "V", "Ti", "Alt", "Als",
    "Cu", "Cr", "Ni", "Co", "Mo", "B", "SLAB_FURNACE_ENT_TEMP",
    "PRE_HEAT_TEMP", "PRE_HEAT_TIME", "HEAT_TEMP1",
    "HEAT_TIME1", "HEAT_TEMP2", "HEAT_TIME2", "HEAT_TEMP3", "HEAT_TIME3",
    "SOAK_TEMP", "SOAK_TIME", "FURNACE_EXIT_TEMP", "FURNACE_EXIT_TIME",
    "TEMP_ENTR", "TIME_ENTR", "COOL_TYPE", "SELF_TEMP",
    "SPEED", "R_PASS_ACT", "F_PASS_ACT", "FET", "FDT",
    "TS", "EL", "YS", "AKV",
]
PIPELINE_SELECT_COLUMNS.extend([f"N{index}_DESCALING_REMARK_A" for index in range(1, 31)])
PIPELINE_SELECT_COLUMNS.extend([f"N{index}_DH_CAL" for index in range(1, 31)])
PIPELINE_SELECT_COLUMNS.extend([f"N{index}_DT_CAL" for index in range(1, 31)])
PIPELINE_SELECT_COLUMNS.extend([f"N{index}_DW_CAL" for index in range(1, 31)])
PIPELINE_SELECT_COLUMNS.extend([f"N{index}_FORCE" for index in range(1, 31)])
PIPELINE_SELECT_COLUMNS.extend([f"N{index}_SPD" for index in range(1, 31)])
PIPELINE_SELECT_COLUMNS.extend([f"N{index}_ENTR_DATE" for index in range(1, 31)])
PIPELINE_HISTORICAL_ROLL_REFERENCE_LIMIT = 10


def _mysql_identifier(name: str) -> str:
    return f"`{str(name).replace('`', '``')}`"


def _append_range_condition_mysql(where_parts: list, params: list, column_name: str, min_value, max_value) -> None:
    parts = [f"{_mysql_identifier(column_name)} IS NOT NULL"]
    if _is_effective_min(min_value):
        parts.append(f"{_mysql_identifier(column_name)} >= %s")
        params.append(_to_float(min_value))
    if _is_effective_max(max_value):
        parts.append(f"{_mysql_identifier(column_name)} <= %s")
        params.append(_to_float(max_value))
    if len(parts) > 1:
        where_parts.append("(" + " AND ".join(parts) + ")")


def _build_pipeline_mysql_where(
    spec_result: dict,
    include_process: bool = True,
    include_performance: bool = True,
    include_slab_thickness: bool = True,
    steel_sign_like: str | None = None,
) -> tuple[str, list]:
    where_parts = []
    params = []

    if steel_sign_like:
        where_parts.append(f"{_mysql_identifier('STEEL_SIGN')} LIKE %s")
        params.append(steel_sign_like)

    _append_range_condition_mysql(
        where_parts,
        params,
        "AIM_THICK",
        spec_result.get("AIM_THICK_min"),
        spec_result.get("AIM_THICK_max"),
    )
    for spec_prefix, column_name in PIPELINE_COMPONENT_FIELD_MAP.items():
        _append_range_condition_mysql(
            where_parts,
            params,
            column_name,
            spec_result.get(f"{spec_prefix}_min"),
            spec_result.get(f"{spec_prefix}_max"),
        )
    if include_process:
        for spec_prefix, column_name in PIPELINE_PROCESS_FIELD_MAP.items():
            if spec_prefix == "AIM_THICK":
                continue
            if spec_prefix == "SLAB_THICK" and not include_slab_thickness:
                continue
            _append_range_condition_mysql(
                where_parts,
                params,
                column_name,
                spec_result.get(f"{spec_prefix}_min"),
                spec_result.get(f"{spec_prefix}_max"),
            )
    if include_performance:
        for spec_prefix, column_name in PIPELINE_PERFORMANCE_FIELD_MAP.items():
            _append_range_condition_mysql(
                where_parts,
                params,
                column_name,
                spec_result.get(f"{spec_prefix}_min"),
                spec_result.get(f"{spec_prefix}_max"),
            )
    return (" AND ".join(where_parts) if where_parts else "1=1"), params


def _load_mysql_driver():
    try:
        import pymysql
        return "pymysql", pymysql
    except ModuleNotFoundError:
        pass
    return "dotnet-mysql", None


def _convert_mysql_placeholders_for_dotnet(sql: str, params: list) -> tuple[str, list[str]]:
    param_names = []
    converted = sql
    for index, _value in enumerate(params):
        param_name = f"@p{index}"
        converted = converted.replace("%s", param_name, 1)
        param_names.append(param_name)
    return converted, param_names


def _format_mysql_log_value(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "''") + "'"


def _render_mysql_sql_for_log(sql: str, params: list) -> str:
    rendered = sql
    for value in params:
        rendered = rendered.replace("%s", _format_mysql_log_value(value), 1)
    return rendered


def _query_first_pipeline_mysql_row_with_dotnet(sql: str, params: list):
    _prepare_pipeline_mysql_runtime()
    converted_sql, param_names = _convert_mysql_placeholders_for_dotnet(sql, params)
    import clr
    clr.AddReference(_os.path.join(PIPELINE_IMAGE_GENERATOR_BIN_DIR, "MySql.Data.dll"))
    from MySql.Data.MySqlClient import MySqlConnection, MySqlCommand, MySqlDataAdapter
    from System.Data import DataTable

    connstr = (
        f"server={PIPELINE_MYSQL_HOST};"
        "Allow User Variables=True;"
        f"user={PIPELINE_MYSQL_USER};"
        f"database={PIPELINE_MYSQL_DATABASE};"
        f"port={PIPELINE_MYSQL_PORT};"
        f"password={PIPELINE_MYSQL_PASSWORD};"
        "Connect Timeout=1200;"
        "SslMode=None;"
    )
    dt = DataTable()
    conn = MySqlConnection(connstr)
    try:
        conn.Open()
        cmd = MySqlCommand(converted_sql, conn)
        try:
            cmd.CommandTimeout = 300
            for param_name, value in zip(param_names, params):
                cmd.Parameters.AddWithValue(param_name, value)
            adapter = MySqlDataAdapter(cmd)
            try:
                adapter.Fill(dt)
            finally:
                adapter.Dispose()
        finally:
            cmd.Dispose()
    finally:
        conn.Close()
        conn.Dispose()

    if dt.Rows.Count == 0:
        return None
    row = dt.Rows[0]
    return {
        str(column.ColumnName): "" if row[column] is None else str(row[column])
        for column in dt.Columns
    }


def _query_pipeline_mysql_rows_with_dotnet(sql: str, params: list) -> list[dict]:
    """通过 MySql.Data 执行多行查询，供相近厚度历史轧制规程检索使用。"""
    _prepare_pipeline_mysql_runtime()
    converted_sql, param_names = _convert_mysql_placeholders_for_dotnet(sql, params)
    import clr
    clr.AddReference(_os.path.join(PIPELINE_IMAGE_GENERATOR_BIN_DIR, "MySql.Data.dll"))
    from MySql.Data.MySqlClient import MySqlConnection, MySqlCommand, MySqlDataAdapter
    from System.Data import DataTable

    connstr = (
        f"server={PIPELINE_MYSQL_HOST};"
        "Allow User Variables=True;"
        f"user={PIPELINE_MYSQL_USER};"
        f"database={PIPELINE_MYSQL_DATABASE};"
        f"port={PIPELINE_MYSQL_PORT};"
        f"password={PIPELINE_MYSQL_PASSWORD};"
        "Connect Timeout=1200;"
        "SslMode=None;"
    )
    dt = DataTable()
    conn = MySqlConnection(connstr)
    try:
        conn.Open()
        cmd = MySqlCommand(converted_sql, conn)
        try:
            cmd.CommandTimeout = 300
            for param_name, value in zip(param_names, params):
                cmd.Parameters.AddWithValue(param_name, value)
            adapter = MySqlDataAdapter(cmd)
            try:
                adapter.Fill(dt)
            finally:
                adapter.Dispose()
        finally:
            cmd.Dispose()
    finally:
        conn.Close()
        conn.Dispose()

    return [
        {
            str(column.ColumnName): "" if row[column] is None else str(row[column])
            for column in dt.Columns
        }
        for row in dt.Rows
    ]


def _resolve_pipeline_history_target_thickness(
    spec_result: dict,
    user_message: str,
) -> tuple[float | None, float | None, float | None]:
    """确定历史规程检索使用的目标值与有效厚度区间。"""
    explicit_target = _extract_pipeline_target_thickness_from_text(user_message)
    lower = _to_float(spec_result.get("AIM_THICK_min"))
    upper = _to_float(spec_result.get("AIM_THICK_max"))
    lower = lower if _is_effective_min(lower) else None
    upper = upper if _is_effective_max(upper) else None

    target = explicit_target
    if target is None and lower is not None and upper is not None:
        target = (lower + upper) / 2.0
    elif target is None:
        target = lower if lower is not None else upper
    return target, lower, upper


def _query_nearest_pipeline_historical_rows(
    spec_result: dict,
    user_message: str,
    limit: int = PIPELINE_HISTORICAL_ROLL_REFERENCE_LIMIT,
) -> list[dict]:
    """查询成品厚度最接近用户目标区间的历史实绩，供轧制规程设计参考。"""
    driver_name, mysql_driver = _load_mysql_driver()
    target, lower, upper = _resolve_pipeline_history_target_thickness(
        spec_result,
        user_message,
    )
    safe_limit = max(1, min(int(limit), PIPELINE_HISTORICAL_ROLL_REFERENCE_LIMIT))
    selected_columns = ", ".join(
        _mysql_identifier(column) for column in PIPELINE_SELECT_COLUMNS
    )
    thickness_column = _mysql_identifier("AIM_THICK")
    order_parts = []
    params: list = []

    # 先优先选择落入目标范围的记录，再按与明确目标值的距离排序。
    if lower is not None and upper is not None:
        order_parts.append(
            f"CASE WHEN {thickness_column} < %s THEN %s - {thickness_column} "
            f"WHEN {thickness_column} > %s THEN {thickness_column} - %s ELSE 0 END ASC"
        )
        params.extend([lower, lower, upper, upper])
    if target is not None:
        order_parts.append(f"ABS({thickness_column} - %s) ASC")
        params.append(target)
    order_parts.append(f"{_mysql_identifier(PIPELINE_SORT_FIELD)} DESC")

    sql = (
        f"SELECT {selected_columns} FROM {_mysql_identifier(PIPELINE_MYSQL_TABLE)} "
        f"WHERE {thickness_column} IS NOT NULL AND {thickness_column} > 0 "
        f"ORDER BY {', '.join(order_parts)} LIMIT {safe_limit}"
    )
    print(
        "[历史轧制规程参考] 查询相近厚度实绩: "
        f"target={target}, range=({lower}, {upper}), limit={safe_limit}, driver={driver_name}"
    )
    print(f"[历史轧制规程参考] SQL预览: {_render_mysql_sql_for_log(sql, params)}")

    if driver_name == "pymysql":
        with mysql_driver.connect(
            host=PIPELINE_MYSQL_HOST,
            port=PIPELINE_MYSQL_PORT,
            user=PIPELINE_MYSQL_USER,
            password=PIPELINE_MYSQL_PASSWORD,
            database=PIPELINE_MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=mysql_driver.cursors.DictCursor,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall() or [])

    if driver_name == "dotnet-mysql":
        return _query_pipeline_mysql_rows_with_dotnet(sql, params)

    conn = mysql_driver.connect(
        host=PIPELINE_MYSQL_HOST,
        port=PIPELINE_MYSQL_PORT,
        user=PIPELINE_MYSQL_USER,
        password=PIPELINE_MYSQL_PASSWORD,
        database=PIPELINE_MYSQL_DATABASE,
    )
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(sql, params)
            return list(cur.fetchall() or [])
        finally:
            cur.close()
    finally:
        conn.close()


def _query_first_pipeline_mysql_row(
    spec_result: dict,
    include_process: bool = True,
    include_performance: bool = True,
    include_slab_thickness: bool = True,
    stage_name: str = "MySQL查询",
    steel_sign_like: str | None = None,
):
    # 加载当前可用的 MySQL 驱动：优先 pymysql，其次 .NET MySql.Data，最后兼容 mysql-connector。
    driver_name, mysql_driver = _load_mysql_driver()
    # 根据模型提取/兜底后的规格边界生成 WHERE 条件；include_* 控制是否纳入工艺和性能约束。
    where_sql, params = _build_pipeline_mysql_where(
        spec_result,
        include_process=include_process,
        include_performance=include_performance,
        include_slab_thickness=include_slab_thickness,
        steel_sign_like=steel_sign_like,
    )
    # 记录本阶段的约束数量，便于排查“条件过严导致无结果”或“放宽查询是否生效”。
    condition_count = 0 if where_sql == "1=1" else len(params)
    print(
        f"[管线钢MySQL匹配] {stage_name}: 条件数={condition_count}, "
        f"牌号条件={steel_sign_like or '无'}, 驱动={driver_name}, 排序={PIPELINE_SORT_FIELD}"
    )
    # 只查询后续匹配响应、DLL 仿真和报告生成需要的字段，避免返回整表造成上下文污染。
    selected_columns = ", ".join(_mysql_identifier(column) for column in PIPELINE_SELECT_COLUMNS)
    # 按生产日期倒序取第一条，表示在满足规格范围的实绩中优先使用最新记录。
    sql = (
        f"SELECT {selected_columns} FROM {_mysql_identifier(PIPELINE_MYSQL_TABLE)} "
        f"WHERE {where_sql} ORDER BY {_mysql_identifier(PIPELINE_SORT_FIELD)} DESC LIMIT 1"
    )
    # 打印渲染后的 SQL 预览，仅用于调试匹配条件，不影响实际参数化执行。
    print(f"[管线钢MySQL匹配] {stage_name} SQL预览: {_render_mysql_sql_for_log(sql, params)}")
    if driver_name == "pymysql":
        # pymysql 分支：使用 DictCursor，直接返回字段名到字段值的字典。
        with mysql_driver.connect(
            host=PIPELINE_MYSQL_HOST,
            port=PIPELINE_MYSQL_PORT,
            user=PIPELINE_MYSQL_USER,
            password=PIPELINE_MYSQL_PASSWORD,
            database=PIPELINE_MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=mysql_driver.cursors.DictCursor,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()

    if driver_name == "dotnet-mysql":
        # .NET MySql.Data 分支：复用 DLL 运行环境，适配 Windows 环境下的 MySQL 访问。
        return _query_first_pipeline_mysql_row_with_dotnet(sql, params)

    # mysql-connector 兼容分支：显式关闭 cursor/connection，避免长时间服务中连接泄漏。
    conn = mysql_driver.connect(
        host=PIPELINE_MYSQL_HOST,
        port=PIPELINE_MYSQL_PORT,
        user=PIPELINE_MYSQL_USER,
        password=PIPELINE_MYSQL_PASSWORD,
        database=PIPELINE_MYSQL_DATABASE,
    )
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            cur.close()
    finally:
        conn.close()


def _query_first_pipeline_mysql_row_without_filters(stage_name: str = "MySQL无筛选兜底查询"):
    """去掉所有筛选条件，优先按生产日期倒序取首条；取不到时再无排序取第一条。"""
    driver_name, mysql_driver = _load_mysql_driver()
    print(f"[管线钢MySQL匹配] {stage_name}: 条件数=0, 驱动={driver_name}, 排序={PIPELINE_SORT_FIELD}")
    selected_columns = ", ".join(_mysql_identifier(column) for column in PIPELINE_SELECT_COLUMNS)
    ordered_sql = (
        f"SELECT {selected_columns} FROM {_mysql_identifier(PIPELINE_MYSQL_TABLE)} "
        f"ORDER BY {_mysql_identifier(PIPELINE_SORT_FIELD)} DESC LIMIT 1"
    )
    unordered_sql = f"SELECT {selected_columns} FROM {_mysql_identifier(PIPELINE_MYSQL_TABLE)} LIMIT 1"
    print(f"[管线钢MySQL匹配] {stage_name} SQL预览: {ordered_sql}")
    params = []

    if driver_name == "pymysql":
        with mysql_driver.connect(
            host=PIPELINE_MYSQL_HOST,
            port=PIPELINE_MYSQL_PORT,
            user=PIPELINE_MYSQL_USER,
            password=PIPELINE_MYSQL_PASSWORD,
            database=PIPELINE_MYSQL_DATABASE,
            charset="utf8mb4",
            cursorclass=mysql_driver.cursors.DictCursor,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(ordered_sql, params)
                row = cur.fetchone()
                if row:
                    return row
                print(f"[管线钢MySQL匹配] {stage_name}: 按 {PIPELINE_SORT_FIELD} 排序未取到数据，改为无排序取第一条")
                cur.execute(unordered_sql, params)
                return cur.fetchone()

    if driver_name == "dotnet-mysql":
        row = _query_first_pipeline_mysql_row_with_dotnet(ordered_sql, params)
        if row:
            return row
        print(f"[管线钢MySQL匹配] {stage_name}: 按 {PIPELINE_SORT_FIELD} 排序未取到数据，改为无排序取第一条")
        return _query_first_pipeline_mysql_row_with_dotnet(unordered_sql, params)

    conn = mysql_driver.connect(
        host=PIPELINE_MYSQL_HOST,
        port=PIPELINE_MYSQL_PORT,
        user=PIPELINE_MYSQL_USER,
        password=PIPELINE_MYSQL_PASSWORD,
        database=PIPELINE_MYSQL_DATABASE,
    )
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(ordered_sql, params)
            row = cur.fetchone()
            if row:
                return row
            print(f"[管线钢MySQL匹配] {stage_name}: 按 {PIPELINE_SORT_FIELD} 排序未取到数据，改为无排序取第一条")
            cur.execute(unordered_sql, params)
            return cur.fetchone()
        finally:
            cur.close()
    finally:
        conn.close()


def _build_pipeline_match_response(
    row: dict | None,
    is_state: bool,
    session_id: str,
    message: str | None = None,
    error: str | None = None,
) -> dict:
    # 未查到数据或上游查询失败时，返回统一的空匹配结构，方便前端和后续逻辑稳定处理。
    if not row:
        response = {
            "isState": False,
            "strCoil": "",
            "strSteel": "",
            "session_key": session_id,
            "arrBody": [],
        }
        if message:
            response["message"] = message
        if error:
            response["error"] = error
        return response

    # MySQL 字段名可能大小写不一致，这里统一转成大写，便于提取卷号/牌号等固定字段。
    normalized = {str(k).upper(): v for k, v in row.items()}
    arr_body = []
    for key, value in row.items():
        arr_body.append({str(key): _format_oracle_value(value)})
        if str(key).upper() == "SLAB_ID":
            # 加热炉号不再读取数据库，统一使用当前生产线固定炉号。
            arr_body.append({"HEAT_FURNACE_ID": "1"})
    # 返回结构保持与原有耐磨钢匹配接口一致：
    # isState 表示是否严格命中；strCoil/strSteel 供内部流程使用；arrBody 保存完整字段列表。
    response = {
        "isState": bool(is_state),
        # strCoil 取板坯号，用于 DLL 图片目录和内部追溯，不应写入最终报告。
        "strCoil": _format_oracle_value(normalized.get("SLAB_ID")),
        # strSteel 是数据库中的牌号/钢种标识，仅用于匹配过程，最终报告会通过脱敏规则禁止输出。
        "strSteel": _format_oracle_value(normalized.get("STEEL_SIGN")),
        "session_key": session_id,
        # arrBody 保留精简后的查询字段顺序，并在 SLAB_ID 后插入固定炉号。
        "arrBody": arr_body,
    }
    # message/error 是可选诊断信息，主要用于放宽查询、异常查询或空结果场景。
    if message:
        response["message"] = message
    if error:
        response["error"] = error
    return response


def _pipeline_spec_bounds_for_matched_field(field_name: str, spec_result: dict) -> tuple[float | None, float | None]:
    """把管线钢 MySQL 字段名映射到 spec_result 中对应的上下限。"""
    upper_name = str(field_name).upper()
    if upper_name in PIPELINE_REFINABLE_PASS_FIELDS:
        return None, None
    spec_prefix = (
        PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC.get(upper_name)
        or PIPELINE_REFINABLE_PROCESS_FIELD_TO_SPEC.get(upper_name)
        or PIPELINE_REFINABLE_PERFORMANCE_FIELD_TO_SPEC.get(upper_name)
    )
    if not spec_prefix:
        return None, None
    return (
        _to_float(spec_result.get(f"{spec_prefix}_min")),
        _to_float(spec_result.get(f"{spec_prefix}_max")),
    )


def _pipeline_value_within_spec_bounds(field_name: str, value, spec_result: dict) -> bool:
    """判断管线钢 LLM 修改后的成分、性能或工艺值是否仍在 spec_result 边界内。"""
    min_value, max_value = _pipeline_spec_bounds_for_matched_field(field_name, spec_result)
    number = _to_float(value)
    if number is None:
        return False
    if _is_effective_min(min_value) and number < min_value:
        return False
    if _is_effective_max(max_value) and number > max_value:
        return False
    return True


def _build_pipeline_refinement_bound_error(
    field_name: str,
    candidate_value,
    spec_result: dict,
    *,
    formatted: bool = False,
) -> str:
    """生成可直接交给 LLM 重设计的字段边界反馈。

    后置微调在风电等严格分支中会拒绝任何越界成分。仅记录“MO 超界”会让
    模型不知道应回填到哪个标准区间，因此这里把本轮候选值、规格最小/最大值
    和字段单位一起写入校验错误；下一轮修复提示词会原样携带该信息。
    """
    upper_name = str(field_name or "").upper()
    minimum, maximum = _pipeline_spec_bounds_for_matched_field(upper_name, spec_result)
    unit = _fact_table_unit_for_field(upper_name)

    range_parts: list[str] = []
    if _is_effective_min(minimum):
        range_parts.append(f"{upper_name} >= {minimum:g}")
    if _is_effective_max(maximum):
        range_parts.append(f"{upper_name} <= {maximum:g}")
    allowed_range = "，".join(range_parts) or "未提取到有效上下限"
    unit_text = f" {unit}" if unit else ""
    value_text = repr(candidate_value)
    stage_text = "格式化后" if formatted else ""
    return (
        f"LLM微调字段 {upper_name}={value_text} {stage_text}超出规格边界或不是合法数值；"
        f"本轮标准允许范围：{allowed_range}{unit_text}。"
        f"请重新设计 {upper_name}，使其严格落入上述范围；"
        "同时仅在必要范围内联动调整相关成分、工艺和性能，不能用历史越界值替代。"
    )


def _round_to_last_digit_0_or_5(number: float, decimals: int) -> float:
    """按指定小数位格式化前，先把最后一位就近归到 0 或 5。"""
    scale = 10 ** decimals
    scaled = int(number * scale + 0.5) if number >= 0 else int(number * scale - 0.5)
    quotient = scaled / 5
    rounded_scaled = (int(quotient + 0.5) if quotient >= 0 else int(quotient - 0.5)) * 5
    return rounded_scaled / scale


def _format_pipeline_refined_value(field_name: str, value) -> str | None:
    """按管线钢微调规则格式化白名单字段；无法转数值时返回 None。"""
    number = _to_float(value)
    if number is None:
        return None
    upper_name = str(field_name).upper()
    if upper_name in PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC:
        return f"{_round_to_last_digit_0_or_5(number, 4):.4f}"
    if upper_name in PIPELINE_REFINABLE_PERFORMANCE_INTEGER_FIELDS:
        return f"{_round_to_last_digit_0_or_5(number, 0):.0f}"
    if upper_name in PIPELINE_REFINABLE_PERCENT_FIELDS:
        return f"{_round_to_last_digit_0_or_5(number, 2):.2f}"
    if upper_name in PIPELINE_REFINABLE_TEMPERATURE_FIELDS:
        return f"{_round_to_last_digit_0_or_5(number, 0):.0f}"
    if upper_name in PIPELINE_REFINABLE_PASS_COUNT_FIELDS:
        return str(int(round(number)))
    if upper_name in PIPELINE_REFINABLE_INTEGER_FIELDS:
        return f"{_round_to_last_digit_0_or_5(number, 0):.0f}"
    if upper_name in PIPELINE_REFINABLE_THICKNESS_FIELDS:
        return f"{_round_to_last_digit_0_or_5(number, 2):.2f}"
    return _format_oracle_value(value)


def _pipeline_formatted_value_within_spec_bounds(field_name: str, formatted_value, spec_result: dict) -> bool:
    """判断格式化后的管线钢白名单字段是否仍满足 spec_result 边界。"""
    return _pipeline_value_within_spec_bounds(field_name, formatted_value, spec_result)


def _enforce_pipeline_performance_standard(
    matched_result: dict,
    spec_result: dict,
) -> dict:
    """保证四项性能满足规格，并为模型失败场景生成确定性的合格结果。

    后置 LLM 正常返回时保留其范围内数值；模型漏填、返回非数值或越界时，
    按规格边界夹紧。该兜底只作用于性能字段，绝不恢复历史数据库中的越界值。
    """
    result = copy.deepcopy(matched_result)
    body = result.get("arrBody")
    if not isinstance(body, list):
        return result

    corrected_body = []
    for item in body:
        field_key = _get_arrbody_key(item)
        field_name = str(field_key or "").upper()
        value = _get_arrbody_value(item)
        if field_name not in PIPELINE_PERFORMANCE_FIELDS:
            corrected_body.append(item)
            continue

        number = _to_float(value)
        min_value = _to_float(spec_result.get(f"{field_name}_min"))
        max_value = _to_float(spec_result.get(f"{field_name}_max"))
        has_min = min_value is not None and _is_effective_min(min_value)
        has_max = max_value is not None and _is_effective_max(max_value)

        if number is None:
            number = min_value if has_min else max_value
        if number is not None and has_min and number < min_value:
            number = min_value
        if number is not None and has_max and number > max_value:
            number = max_value

        formatted = _format_pipeline_refined_value(field_name, number)
        if (
            formatted is None
            or not _pipeline_formatted_value_within_spec_bounds(field_name, formatted, spec_result)
        ):
            # 无有效上下限时保留已有值；正常规格下此分支只作为防御性保护。
            formatted = value
        corrected_body.append({field_key: formatted})

    result["arrBody"] = corrected_body
    return result


def _extract_pipeline_target_thickness_from_text(user_message: str) -> float | None:
    """提取明确的目标成品厚度，板坯厚度必须排除。"""
    if not user_message:
        return None
    text = str(user_message)
    # 先移除板坯厚度短语，防止其中的“厚度320mm”被误判为成品厚度。
    text = re.sub(
        r"(?:板坯厚度|连铸坯厚度)\D{0,12}\d+(?:\.\d+)?\s*(?:mm|毫米)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\d+(?:\.\d+)?\s*(?:mm|毫米)\D{0,12}(?:板坯厚度|连铸坯厚度)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    patterns = [
        r"(?:厚度|目标厚度|成品厚度|钢板厚度|板厚)\D{0,12}(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)\D{0,12}(?:厚度|目标厚度|成品厚度|钢板厚度|板厚)",
        # 覆盖“25mm厚的钢板”“25毫米厚钢”等常见简写。板坯厚度短语已在
        # 上方移除，故这里不会把板坯尺寸误识别为最终成品厚度。
        r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)\s*厚(?:的)?",
        r"厚\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _to_float(match.group(1))
    return None


def _extract_pipeline_target_slab_thickness_from_text(user_message: str) -> float | None:
    """从用户原始需求中独立提取明确的目标板坯厚度，单位按 mm 处理。"""
    if not user_message:
        return None
    patterns = [
        r"(?:板坯厚度|连铸坯厚度)\D{0,12}(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)\D{0,12}(?:板坯厚度|连铸坯厚度)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(user_message), flags=re.IGNORECASE)
        if match:
            return _to_float(match.group(1))
    return None


WIND_POWER_SOFT_PROCESS_FIELD_LABELS = {
    "FET": r"(?:FET|精轧开轧温度|精轧入口温度)",
    "FDT": r"(?:FDT|精轧终轧温度|终轧温度)",
    "TEMP_ENTR": r"(?:TEMP_ENTR|入水温度|开冷温度)",
    "FEH": r"(?:FEH|中间坯厚度)",
    "SELF_TEMP": r"(?:SELF_TEMP|返红温度)",
    "FURNACE_EXIT_TEMP": r"(?:FURNACE_EXIT_TEMP|出炉温度)",
}


def _extract_explicit_wind_process_value(user_message: str, field_name: str) -> float | None:
    """只识别用户明确给出的风电工艺单值，不从文献摘要猜测工艺窗口。"""
    text = str(user_message or "")
    label = WIND_POWER_SOFT_PROCESS_FIELD_LABELS.get(field_name)
    if not text or not label:
        return None
    patterns = (
        rf"{label}\D{{0,12}}(-?\d+(?:\.\d+)?)\s*(?:℃|°C|摄氏度)?",
        rf"(-?\d+(?:\.\d+)?)\s*(?:℃|°C|摄氏度)?\D{{0,12}}{label}",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _to_float(match.group(1))
    return None


def _normalize_wind_power_process_spec(spec_result: dict, user_message: str) -> dict:
    """将风电产品标准中未规定的厂内 TMCP 参数恢复为非强制范围。

    GB/T 1591 的成分、性能、CEV/Pcm 等仍由专用标准上下文严格校验；
    FET/FDT 等只有在用户明确给出具体值时才作为单值要求，否则使用
    0～9999 的开放范围，防止 RAG 文献数字误入 MySQL 硬筛选条件。
    """
    normalized = dict(spec_result or {})
    for field_name in WIND_POWER_SOFT_PROCESS_FIELD_LABELS:
        explicit_value = _extract_explicit_wind_process_value(user_message, field_name)
        if explicit_value is None:
            normalized[f"{field_name}_min"] = 0.0
            normalized[f"{field_name}_max"] = 9999.0
        else:
            normalized[f"{field_name}_min"] = explicit_value
            normalized[f"{field_name}_max"] = explicit_value
    return normalized


def _format_pipeline_target_thickness(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _lock_explicit_pipeline_thickness_targets(
    matched_result: dict,
    target_thickness: float | None,
    target_slab_thickness: float | None,
) -> dict:
    """把用户明确提出的厚度写入字段骨架，并交由后续工艺智能体沿用。

    后置 LLM 可以修改轧制道次，但不能修改 AIM_THICK 和 SLAB_THICK。
    因此必须在调用模型前由后端确定性写入用户明确值，后续道次设计以该
    AIM_THICK 为末道厚度硬约束。未明确提出的厚度继续沿用数据库匹配值。
    """
    locked = copy.deepcopy(matched_result)
    replacements = {}
    if target_thickness is not None:
        replacements["AIM_THICK"] = _format_pipeline_target_thickness(target_thickness)
    if target_slab_thickness is not None:
        replacements["SLAB_THICK"] = _format_pipeline_target_thickness(target_slab_thickness)
    if not replacements:
        return locked

    locked_body = []
    for item in locked.get("arrBody") or []:
        key = _get_arrbody_key(item)
        replacement = replacements.get(str(key or "").upper())
        locked_body.append({key: replacement} if replacement is not None else item)
    locked["arrBody"] = locked_body
    return locked


def _restore_pipeline_arrbody_fields(
    candidate: dict,
    source: dict,
    field_names: set[str],
) -> dict:
    """把指定字段恢复到已通过另一类校验的版本，实现成分与轧制分阶段重试。"""
    restored = copy.deepcopy(candidate)
    source_values = {
        str(_get_arrbody_key(item) or "").upper(): _get_arrbody_value(item)
        for item in (source.get("arrBody") or [])
        if _get_arrbody_key(item)
    }
    restored_body = []
    for item in restored.get("arrBody") or []:
        key = _get_arrbody_key(item)
        upper_key = str(key or "").upper()
        if upper_key in field_names and upper_key in source_values:
            restored_body.append({key: source_values[upper_key]})
        else:
            restored_body.append(item)
    restored["arrBody"] = restored_body
    return restored


def _prepare_pipeline_full_roll_redesign_baseline(matched_result: dict) -> dict:
    """清空整套轧制设计值，阻止模型继续逐道移动目标厚度做局部修补。"""
    baseline = copy.deepcopy(matched_result)
    baseline_body = []
    for item in baseline.get("arrBody") or []:
        key = _get_arrbody_key(item)
        upper_key = str(key or "").upper()
        if upper_key in (
            {"FET", "FDT"}
            | PIPELINE_REFINABLE_PASS_COUNT_FIELDS
            | PIPELINE_REFINABLE_TURN_FIELDS
        ):
            baseline_body.append({key: ""})
        elif upper_key in PIPELINE_REFINABLE_PASS_TIME_FIELDS:
            baseline_body.append({key: ""})
        elif upper_key in PIPELINE_REFINABLE_PASS_FIELDS:
            baseline_body.append({key: "0"})
        else:
            baseline_body.append(item)
    baseline["arrBody"] = baseline_body
    return baseline


def _build_pipeline_historical_roll_reference(matched_result: dict) -> str:
    """提取紧凑历史轧制参考，供全量重设计时映射相似道次和轧制力。"""
    values = {
        str(_get_arrbody_key(item) or "").upper(): _get_arrbody_value(item)
        for item in (matched_result.get("arrBody") or [])
        if _get_arrbody_key(item)
    }
    rough = _to_float(values.get("R_PASS_ACT"))
    finish = _to_float(values.get("F_PASS_ACT"))
    total = None
    if rough is not None and finish is not None:
        total = int(round(rough)) + int(round(finish))
    if total is None or total < 1 or total > 30:
        total = 30

    passes = []
    for pass_index in range(1, total + 1):
        pass_data = {"index": pass_index}
        for suffix in ("DH_CAL", "DT_CAL", "DW_CAL", "FORCE", "SPD", "ENTR_DATE"):
            pass_data[suffix] = values.get(f"N{pass_index}_{suffix}")
        passes.append(pass_data)
    reference = {
        "SLAB_THICK": values.get("SLAB_THICK"),
        "SLAB_WIDTH": values.get("SLAB_WIDTH"),
        "SLAB_LEN": values.get("SLAB_LEN"),
        "AIM_THICK": values.get("AIM_THICK"),
        "AIM_WIDTH": values.get("AIM_WIDTH"),
        "WIDTH_ROLL_START_REMARK": values.get("WIDTH_ROLL_START_REMARK"),
        "WIDTH_ROLL_END_REMARK": values.get("WIDTH_ROLL_END_REMARK"),
        "FET": values.get("FET"),
        "FDT": values.get("FDT"),
        "R_PASS_ACT": values.get("R_PASS_ACT"),
        "F_PASS_ACT": values.get("F_PASS_ACT"),
        "passes": passes,
    }
    return json.dumps(reference, ensure_ascii=False)


def _build_pipeline_historical_roll_markdown(rows: list[dict]) -> str:
    """把相近厚度历史实绩压缩为仅含轧制设计信息的 Markdown。"""
    if not rows:
        return ""

    def cell(value) -> str:
        text = "" if value is None else str(value).strip()
        return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ") or "-"

    def pass_count(value) -> int:
        number = _to_float(value)
        if number is None:
            return 0
        return max(0, min(30, int(round(number))))

    normalized_rows = [
        {str(key).upper(): value for key, value in row.items()}
        for row in rows
        if isinstance(row, dict)
    ]
    if not normalized_rows:
        return ""

    lines = [
        "### 相近厚度历史轧制实绩概览",
        "",
        "|样本|板坯厚度(mm)|板坯宽度(mm)|板坯长度(mm)|成品厚度(mm)|目标宽度(mm)|转钢开始道次|转钢结束道次|粗轧道次数|精轧道次数|FET(℃)|FDT(℃)|",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    pass_rows: list[str] = []
    for sample_index, values in enumerate(normalized_rows, start=1):
        rough_count = pass_count(values.get("R_PASS_ACT"))
        finish_count = pass_count(values.get("F_PASS_ACT"))
        declared_total = rough_count + finish_count

        effective_indices = []
        for pass_index in range(1, 31):
            thickness = _to_float(values.get(f"N{pass_index}_DH_CAL"))
            if thickness is None or thickness <= 0:
                if effective_indices:
                    break
                continue
            effective_indices.append(pass_index)
        if declared_total <= 0 or declared_total > 30:
            declared_total = len(effective_indices)
        effective_indices = effective_indices[:declared_total]

        lines.append(
            "|{sample}|{slab_thick}|{slab_width}|{slab_len}|{aim}|{aim_width}|{turn_start}|{turn_end}|{rough}|{finish}|{fet}|{fdt}|".format(
                sample=sample_index,
                slab_thick=cell(values.get("SLAB_THICK")),
                slab_width=cell(values.get("SLAB_WIDTH")),
                slab_len=cell(values.get("SLAB_LEN")),
                aim=cell(values.get("AIM_THICK")),
                aim_width=cell(values.get("AIM_WIDTH")),
                turn_start=cell(values.get("WIDTH_ROLL_START_REMARK")),
                turn_end=cell(values.get("WIDTH_ROLL_END_REMARK")),
                rough=rough_count,
                finish=finish_count,
                fet=cell(values.get("FET")),
                fdt=cell(values.get("FDT")),
            )
        )

        for ordinal, pass_index in enumerate(effective_indices, start=1):
            stage = "粗轧" if rough_count > 0 and ordinal <= rough_count else "精轧"
            pass_rows.append(
                "|{sample}|N{pass_index}|{stage}|{thickness}|{temperature}|{width}|"
                "{speed}|{force}|{entry_time}|{descaling}|".format(
                    sample=sample_index,
                    pass_index=pass_index,
                    stage=stage,
                    thickness=cell(values.get(f"N{pass_index}_DH_CAL")),
                    temperature=cell(values.get(f"N{pass_index}_DT_CAL")),
                    width=cell(values.get(f"N{pass_index}_DW_CAL")),
                    speed=cell(values.get(f"N{pass_index}_SPD")),
                    force=cell(values.get(f"N{pass_index}_FORCE")),
                    entry_time=cell(values.get(f"N{pass_index}_ENTR_DATE")),
                    descaling=cell(values.get(f"N{pass_index}_DESCALING_REMARK_A")),
                )
            )

    lines.extend([
        "",
        "### 相近厚度历史实绩逐道次参数",
        "",
        "|样本|道次|阶段|出口厚度(mm)|变形温度(℃)|出口宽度(mm)|轧制速度(m/min)|轧制力(kN)|入口时间|除鳞标记|",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|:---|:---:|",
        *pass_rows,
    ])
    return "\n".join(lines)


def _is_pipeline_pass_reallocation_request(user_message: str) -> bool:
    """识别用户是否明确要求重新分配、增加或删除轧制道次。"""
    text = str(user_message or "")
    patterns = (
        r"(?:重新设计|重排|重新分配|优化|调整).{0,8}(?:所有|全部|各)?道次",
        r"(?:增加|新增|减少|删除|取消|增删).{0,6}道次",
        r"道次数.{0,6}(?:调整|优化|增加|减少|重新设计)",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _sanitize_pipeline_refined_matched_result(
    original: dict,
    candidate: dict,
    spec_result: dict,
    target_thickness: float | None = None,
    target_slab_thickness: float | None = None,
    context_modification_override: bool = False,
    pass_reallocation_override: bool = False,
    strict_no_restore: bool = False,
    soft_process_bounds: bool = False,
    validation_errors: list[str] | None = None,
) -> dict | None:
    """校验并修正 LLM 微调结果，确保结构和边界安全。

    风电分支启用 ``strict_no_restore`` 后，任何可设计字段的非法值都会使
    当前候选整体失效并返回明确错误，绝不再静默恢复历史数据库值。
    ``soft_process_bounds`` 用于区分产品标准与工艺参考窗口：GB/T 1591
    约束成分、焊接性和性能，并不把 FET/FDT 等厂内 TMCP 参数规定为
    产品验收硬边界，这些参数只要求是合法数值，后续由工艺智能体校验。
    """
    def reject(message: str) -> None:
        if validation_errors is not None:
            validation_errors.append(message)
        print(f"[管线钢MySQL匹配] {message}")

    if not isinstance(original, dict) or not isinstance(candidate, dict):
        reject("LLM微调结果不是合法 matched_result JSON 对象")
        return None
    original_top_keys = list(original.keys())
    candidate_top_keys = list(candidate.keys())
    if set(candidate_top_keys) != set(original_top_keys):
        missing_keys = [key for key in original_top_keys if key not in candidate]
        extra_keys = [key for key in candidate_top_keys if key not in original]
        reject(
            "LLM微调结果顶层字段集合不一致，拒绝本次结果；"
            f"缺少={missing_keys}，新增={extra_keys}"
        )
        return None
    if candidate.get("isState") != original.get("isState"):
        reject("LLM微调结果修改了 isState，拒绝本次结果")
        return None
    for identity_key in ("strCoil", "strSteel", "session_key"):
        if candidate.get(identity_key) != original.get(identity_key):
            reject(f"LLM微调结果修改了 {identity_key}，拒绝本次结果")
            return None

    original_body = original.get("arrBody")
    candidate_body = candidate.get("arrBody")
    if not isinstance(original_body, list) or not isinstance(candidate_body, list):
        reject("LLM微调结果 arrBody 不是数组")
        return None
    if len(original_body) != len(candidate_body):
        original_field_names = [
            _get_arrbody_key(item) for item in original_body if _get_arrbody_key(item)
        ]
        candidate_field_names = [
            _get_arrbody_key(item) for item in candidate_body if _get_arrbody_key(item)
        ]
        candidate_field_set = set(candidate_field_names)
        original_field_set = set(original_field_names)
        missing_field_names = [
            field_name for field_name in original_field_names
            if field_name not in candidate_field_set
        ]
        extra_field_names = [
            field_name for field_name in candidate_field_names
            if field_name not in original_field_set
        ]
        reject(
            "LLM微调结果 arrBody 长度不一致，"
            f"期望 {len(original_body)} 项，实际 {len(candidate_body)} 项；"
            f"缺少字段={missing_field_names}；新增字段={extra_field_names}"
        )
        return None

    # 预先保留 arrBody 占位，后续替换其值时不会改变顶层键顺序。JSON 对象
    # 本身不依赖键顺序，但固定输出顺序可降低下一轮模型复制结构时的随机性。
    sanitized = {
        key: ([] if key == "arrBody" else original[key])
        for key in original_top_keys
    }
    sanitized_body = []

    for original_item, candidate_item in zip(original_body, candidate_body):
        original_key = _get_arrbody_key(original_item)
        candidate_key = _get_arrbody_key(candidate_item)
        if not original_key or original_key != candidate_key:
            reject(
                "LLM微调结果 arrBody 字段顺序或字段名不一致，"
                f"期望 {original_key!r}，实际 {candidate_key!r}"
            )
            return None

        original_value = _get_arrbody_value(original_item)
        candidate_value = _get_arrbody_value(candidate_item)
        field_name = original_key.upper()

        # 非白名单字段一律恢复原值，保护板坯号、板号等追溯数据。
        if field_name not in PIPELINE_REFINABLE_FIELDS:
            sanitized_body.append({original_key: original_value})
            continue

        original_is_empty = original_value is None or str(original_value).strip() == ""
        strict_target_thickness_override = original.get("isState") is True and (
            # AIM_THICK 及整套轧制规程始终按当前用户需求和规格边界重新设计，
            # 不受严格命中时的历史非空值保护。
            field_name in {"AIM_THICK", "FET", "FDT"}
            or field_name in PIPELINE_REFINABLE_INITIAL_COOLING_FIELDS
            or field_name in PIPELINE_REFINABLE_PASS_FIELDS
            or field_name in PIPELINE_REFINABLE_PASS_COUNT_FIELDS
            or field_name in PIPELINE_REFINABLE_TURN_FIELDS
            or (
                target_thickness is not None
                and (
                    field_name == "AIM_THICK"
                    or field_name in PIPELINE_REFINABLE_PASS_FIELDS
                    or (
                        field_name in PIPELINE_REFINABLE_PROCESS_FIELD_TO_SPEC
                        and field_name != "SLAB_THICK"
                    )
                )
            )
            or (target_slab_thickness is not None and field_name == "SLAB_THICK")
            or (
                pass_reallocation_override
                and (
                    field_name in PIPELINE_REFINABLE_PASS_FIELDS
                    or field_name in PIPELINE_REFINABLE_PASS_COUNT_FIELDS
                    or field_name in PIPELINE_REFINABLE_TURN_FIELDS
                )
            )
        )
        if field_name == "AIM_THICK" and target_thickness is not None:
            candidate_value = target_thickness
        if field_name == "SLAB_THICK" and target_slab_thickness is not None:
            candidate_value = target_slab_thickness

        if (
            original.get("isState") is True
            and not original_is_empty
            and not strict_target_thickness_override
            and not context_modification_override
            and not strict_no_restore
            and field_name not in PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC
            and field_name not in PIPELINE_REFINABLE_PERFORMANCE_FIELD_TO_SPEC
            and field_name not in {"FURNACE_EXIT_TIME", "TIME_ENTR"}
        ):
            formatted_original = _format_pipeline_refined_value(field_name, original_value)
            if (
                formatted_original is not None
                and _pipeline_formatted_value_within_spec_bounds(field_name, formatted_original, spec_result)
            ):
                sanitized_body.append({original_key: formatted_original})
            else:
                sanitized_body.append({original_key: original_value})
            continue

        # 炉出时刻属于上游既成事实，只能修正编码而不能改变实际时刻。
        if field_name == "FURNACE_EXIT_TIME":
            candidate_time = str(candidate_value or "").strip()
            original_time = _parse_pipeline_process_datetime(original_value)
            parsed_candidate_time = _parse_pipeline_process_datetime(candidate_time)
            if not candidate_time or parsed_candidate_time is None:
                reject(
                    f"LLM微调时间字段 {field_name}={candidate_value!r} 格式无效，"
                    "请仅修正为要求的时间编码"
                )
                return None
            if original_time is not None and parsed_candidate_time != original_time:
                reject(
                    f"LLM微调不得改变 {field_name} 的实际时刻，只允许修正时间格式"
                )
                return None
            sanitized_body.append({original_key: candidate_time})
            continue

        # 开冷时刻由微调 Agent 基于本轮末道时间和冷却初始方案重新设计；这里只
        # 校验 DLL 编码，严格的“晚于末道”关系在统一冷却门禁中检查。
        if field_name == "TIME_ENTR":
            candidate_time = str(candidate_value or "").strip()
            if not candidate_time or _parse_pipeline_process_datetime(candidate_time) is None:
                reject(
                    f"LLM微调时间字段 TIME_ENTR={candidate_value!r} 格式无效，"
                    "请使用 yyyyMMddHHmmss"
                )
                return None
            sanitized_body.append({original_key: candidate_time})
            continue

        if field_name == "AIM_THICK" and target_thickness is not None:
            sanitized_body.append({original_key: _format_pipeline_target_thickness(target_thickness)})
            continue
        if field_name == "AIM_THICK":
            # 未明确给定单值厚度时，AIM_THICK 也必须来自本轮 LLM 基于规格与
            # 性能范围给出的设计结果。无效或越界时拒绝整份响应并触发模型重试，
            # 禁止静默恢复数据库历史 AIM_THICK。
            if not _pipeline_value_within_spec_bounds(field_name, candidate_value, spec_result):
                reject("LLM未返回合法的新 AIM_THICK，拒绝本次微调结果")
                return None
            formatted_aim_thickness = _format_pipeline_refined_value(field_name, candidate_value)
            if (
                formatted_aim_thickness is None
                or not _pipeline_formatted_value_within_spec_bounds(
                    field_name,
                    formatted_aim_thickness,
                    spec_result,
                )
            ):
                reject("LLM返回的新 AIM_THICK 格式化后越界，拒绝本次微调结果")
                return None
            sanitized_body.append({original_key: formatted_aim_thickness})
            continue
        if field_name == "SLAB_THICK" and target_slab_thickness is not None:
            sanitized_body.append({original_key: _format_pipeline_target_thickness(target_slab_thickness)})
            continue

        # 道次时间不是数值字段，不能套用成分/温度的数值格式化逻辑。允许模型
        # 为有效道次返回受支持的时间格式，也允许将停用道次时间清空；有效道次
        # 是否缺时、是否严格递增以及待温时间约束统一由后置道次校验判断。
        if field_name in PIPELINE_REFINABLE_PASS_TIME_FIELDS:
            candidate_time = str(candidate_value or "").strip()
            if candidate_time and _parse_pipeline_process_datetime(candidate_time) is None:
                reject(
                    f"LLM微调道次时间字段 {field_name}={candidate_value!r} 格式无效，"
                    "请使用 YYYY-MM-DD HH:MM:SS.mmm"
                )
                return None
            sanitized_body.append({original_key: candidate_time})
            continue

        # 风电产品标准不规定 FET/FDT 等厂内 TMCP 参数的统一强制窗口。
        # 规格提取结果中的这些范围只作为设计参考，不能据此把模型新值替换成
        # 历史实绩；这里只检查数值和格式，工艺合理性由后续智能体判断。
        if (
            soft_process_bounds
            and field_name in PIPELINE_REFINABLE_PROCESS_FIELD_TO_SPEC
            and field_name not in {"AIM_THICK", "SLAB_THICK"}
        ):
            if _to_float(candidate_value) is None:
                reject(f"LLM微调工艺字段 {field_name} 不是合法数值，拒绝本次结果")
                return None
            formatted_candidate = _format_pipeline_refined_value(field_name, candidate_value)
            if formatted_candidate is None:
                reject(f"LLM微调工艺字段 {field_name} 无法格式化，拒绝本次结果")
                return None
            sanitized_body.append({original_key: formatted_candidate})
            continue

        if _pipeline_value_within_spec_bounds(field_name, candidate_value, spec_result):
            formatted_candidate = _format_pipeline_refined_value(field_name, candidate_value)
            if (
                formatted_candidate is not None
                and _pipeline_formatted_value_within_spec_bounds(field_name, formatted_candidate, spec_result)
            ):
                sanitized_body.append({original_key: formatted_candidate})
            else:
                if (
                    field_name in PIPELINE_PERFORMANCE_FIELDS
                    or field_name in PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC
                ):
                    reject(
                        _build_pipeline_refinement_bound_error(
                            field_name,
                            formatted_candidate,
                            spec_result,
                            formatted=True,
                        )
                    )
                    return None
                if strict_no_restore:
                    reject(
                        _build_pipeline_refinement_bound_error(
                            field_name,
                            formatted_candidate,
                            spec_result,
                            formatted=True,
                        )
                    )
                    return None
                print(f"[管线钢MySQL匹配] LLM微调字段 {field_name} 格式化后超出规格边界，已恢复原值")
                sanitized_body.append({original_key: original_value})
        else:
            if (
                field_name in PIPELINE_PERFORMANCE_FIELDS
                or field_name in PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC
            ):
                reject(
                    _build_pipeline_refinement_bound_error(
                        field_name,
                        candidate_value,
                        spec_result,
                    )
                )
                return None
            if strict_no_restore:
                reject(
                    _build_pipeline_refinement_bound_error(
                        field_name,
                        candidate_value,
                        spec_result,
                    )
                )
                return None
            print(f"[管线钢MySQL匹配] LLM微调字段 {field_name} 超出规格边界或非数值，已恢复原值")
            sanitized_body.append({original_key: original_value})

    sanitized["arrBody"] = sanitized_body
    # 这里仅采纳成分、性能和轧制规程白名单，继续保留顶层结构、arrBody
    # 字段顺序、身份字段以及锁定厚度，供后续三个工艺智能体按固定结构读取。
    return sanitized


def _build_pipeline_refinement_rag_context(spec_result: dict, user_message: str) -> str:
    """由知识库 Agent 自主选择一个用途知识库，为后置微调提供参考资料。"""
    del spec_result  # 路由只依据当前用户提示词，数据库由被选中的工具固定。
    routed = _route_steel_knowledge_base_tool(
        user_message,
        stage_label="后置成分/工艺微调",
    )
    return str(routed.get("content") or "")


def _extract_wind_user_pcm_max_with_llm(user_message: str) -> float | None:
    """用轻量结构化 LLM 调用提取用户明确提出的 Pcm 上限。

    此提取只服务风电分支的第二道 Pcm 校验，不会替代 GB/T 1591 标准值。
    接口异常或模型未返回合法 JSON 时回退到正则识别，避免一个辅助提取失败
    阻断整套设计流程。
    """
    message = str(user_message or "").strip()
    if not message:
        return None

    extraction_prompt = (
        "你只负责提取用户是否明确给出了焊接裂纹敏感性参数 Pcm 的上限。"
        "仅输出 JSON：{\"pcm_user_max\": 数值或 null}。"
        "只有 Pcm/PCM 后出现 <、<=、≤、＜、不高于、不大于、小于或小于等于等"
        "明确上限表达时才返回数值；不要根据材料常识补造数值。"
    )
    try:
        raw = official_qwen_sync.invoke(
            [
                SystemMessage(content=extraction_prompt),
                HumanMessage(content=message),
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=128,
            extra_body={"enable_thinking": False},
        )
        content = str(getattr(raw, "content", "") or "")
        parsed = _parse_json_object(content)
        value = _to_float((parsed or {}).get("pcm_user_max"))
        if value is not None and value >= 0:
            print(f"[风电Pcm提取] LLM识别用户 Pcm 上限={value:.4f}")
            return value
        print("[风电Pcm提取] LLM未识别到用户明确 Pcm 上限")
    except Exception as exc:
        print(f"[风电Pcm提取] LLM提取失败，尝试规则兜底: {type(exc).__name__}: {exc}")

    # 兜底只覆盖明确的上限表达；不能把普通文本中的任意小数误作 Pcm 限值。
    patterns = (
        r"(?:Pcm|PCM)\s*(?:<|<=|≤|＜|不高于|不大于|小于等于|小于)\s*(\d+(?:\.\d+)?)",
        r"(?:<|<=|≤|＜|不高于|不大于|小于等于|小于)\s*(\d+(?:\.\d+)?)\s*(?:的)?\s*(?:Pcm|PCM)",
    )
    for pattern in patterns:
        matched = re.search(pattern, message, flags=re.IGNORECASE)
        if matched:
            value = _to_float(matched.group(1))
            if value is not None and value >= 0:
                print(f"[风电Pcm提取] 规则兜底识别用户 Pcm 上限={value:.4f}")
                return value
    return None


def _effective_wind_chemistry_limits(
    standard_context: dict,
    spec_result: dict | None = None,
) -> dict:
    """合并风电国标与本轮规格提取的有效成分边界。

    GB/T 1591 负责提供缺失字段的硬边界；RAG/用户需求已在 ``spec_result``
    中形成有效上下限时优先使用它。schema 默认的 ``min=0``、``max=9999``
    视为“未提取到约束”，不能覆盖国标最小值或最大值。
    """
    effective = dict((standard_context or {}).get("chemistry") or {})
    if not isinstance(spec_result, dict):
        return effective

    for prefix in PIPELINE_COMPONENT_FIELD_MAP:
        minimum = _to_float(spec_result.get(f"{prefix}_min"))
        maximum = _to_float(spec_result.get(f"{prefix}_max"))
        if _is_effective_min(minimum):
            effective[f"{prefix}_min"] = minimum
        if _is_effective_max(maximum):
            effective[f"{prefix}_max"] = maximum
    return effective


def _validate_wind_power_matched_result(
    matched_result: dict,
    standard_context: dict,
    spec_result: dict | None = None,
) -> str:
    """校验风电塔筒钢板最终成分、性能和焊接性指标。

    历史 MySQL 管线钢记录可供 Agent 比较成分、性能和工艺规律，但该校验仍以
    GB/T 1591 及本轮用户约束为准，防止历史牌号或数值覆盖风电钢设计目标。
    """
    row = _matched_result_body_to_row(matched_result)
    errors: list[str] = []

    field_aliases = {
        "C": "C", "SI": "SI", "MN": "MN", "P": "P", "S": "S", "N": "N",
        "NB": "NB", "V": "V", "TI": "TI", "AL": "AL", "ALS": "ALS",
        "CU": "CU", "CR": "CR", "NI": "NI", "CO": "CO", "MO": "MO", "B": "B",
    }
    values: dict[str, float] = {}
    for standard_name, row_name in field_aliases.items():
        actual_key = next(
            (key for key in row if str(key).upper() in {row_name, "ALT" if row_name == "AL" else row_name}),
            None,
        )
        numeric_value = _to_float(row.get(actual_key)) if actual_key else None
        if numeric_value is not None:
            values[standard_name] = numeric_value

    for key, limit in _effective_wind_chemistry_limits(
        standard_context,
        spec_result,
    ).items():
        prefix, _, bound = str(key).partition("_")
        actual = values.get(prefix)
        if actual is None:
            errors.append(f"缺少化学成分 {prefix}")
        elif bound == "max" and actual > float(limit) + 1e-9:
            errors.append(f"{prefix}={actual:.4f} 超过本轮允许上限 {float(limit):.4f}")
        elif bound == "min" and actual < float(limit) - 1e-9:
            errors.append(f"{prefix}={actual:.4f} 低于本轮允许下限 {float(limit):.4f}")

    refiner_limits = standard_context.get("grain_refiner_requirement") or {}
    refiner_ok = any(
        values.get(prefix, 0.0) >= float(limit) - 1e-9
        for key, limit in refiner_limits.items()
        for prefix, _, bound in [str(key).partition("_")]
        if bound == "min"
    )
    if refiner_limits and not refiner_ok:
        errors.append("Al、Nb、V、Ti 中至少一种未达到 GB/T 1591 细晶元素最低要求")

    tensile = standard_context.get("tensile") or {}
    for key, limit in tensile.items():
        prefix, _, bound = str(key).partition("_")
        actual = _to_float(row.get(prefix))
        if actual is None:
            errors.append(f"缺少力学性能 {prefix}")
        elif bound == "min" and actual < float(limit) - 1e-9:
            errors.append(f"{prefix}={actual:.2f} 低于标准下限 {float(limit):.2f}")
        elif bound == "max" and actual > float(limit) + 1e-9:
            errors.append(f"{prefix}={actual:.2f} 超过标准上限 {float(limit):.2f}")

    impact = standard_context.get("impact") or {}
    akv = _to_float(row.get("AKV"))
    if akv is None:
        errors.append("缺少冲击功 AKV")
    elif impact and akv < float(impact.get("longitudinal", 0.0)) - 1e-9:
        errors.append(
            f"AKV={akv:.2f} 低于 {impact.get('temperature')}℃纵向冲击要求 "
            f"{float(impact.get('longitudinal')):.2f} J"
        )

    if all(component in values for component in ("C", "MN", "CR", "MO", "V", "NI", "CU")):
        cev = (
            values["C"] + values["MN"] / 6.0
            + (values["CR"] + values["MO"] + values["V"]) / 5.0
            + (values["NI"] + values["CU"]) / 15.0
        )
        if cev > float(standard_context.get("CEV_max", 9999.0)) + 1e-9:
            errors.append(f"计算 CEV={cev:.4f} 超过标准上限 {float(standard_context['CEV_max']):.4f}")
    else:
        errors.append("无法计算 CEV，缺少 C/Mn/Cr/Mo/V/Ni/Cu 成分")

    if all(component in values for component in ("C", "SI", "MN", "CU", "CR", "NI", "MO", "V", "B")):
        pcm = (
            values["C"] + values["SI"] / 30.0
            + (values["MN"] + values["CU"] + values["CR"]) / 20.0
            + values["NI"] / 60.0 + values["MO"] / 15.0
            + values["V"] / 10.0 + 5.0 * values["B"]
        )
        standard_pcm_limit = standard_context.get(
            "Pcm_standard_max",
            standard_context.get("Pcm_max"),
        )
        if standard_pcm_limit is not None and pcm > float(standard_pcm_limit) + 1e-9:
            errors.append(
                f"计算 Pcm={pcm:.4f} 超过 GB/T 1591 标准上限 "
                f"{float(standard_pcm_limit):.4f}"
            )
        # 用户明示的 Pcm 要求是独立于标准的第二道硬门禁。即使标准允许，
        # 用户给出更低上限时仍必须反馈模型联动重设计成分。
        user_pcm_limit = standard_context.get("Pcm_user_max")
        if user_pcm_limit is not None and pcm > float(user_pcm_limit) + 1e-9:
            errors.append(
                f"计算 Pcm={pcm:.4f} 超过用户要求上限 "
                f"{float(user_pcm_limit):.4f}"
            )
    else:
        errors.append("无法计算 Pcm，缺少 C/Si/Mn/Cu/Cr/Ni/Mo/V/B 成分")

    return "；".join(errors)


def _calculate_wind_power_pcm(matched_result: dict) -> float | None:
    """按风电分支统一公式计算本轮结果的 Pcm，供报告展示已验证的焊接性约束。"""
    row = _matched_result_body_to_row(matched_result)
    required_fields = ("C", "SI", "MN", "CU", "CR", "NI", "MO", "V", "B")
    values = {field: _to_float(row.get(field)) for field in required_fields}
    if any(value is None for value in values.values()):
        return None
    return (
        values["C"] + values["SI"] / 30.0 + (values["MN"] + values["CU"] + values["CR"]) / 20.0
        + values["NI"] / 60.0 + values["MO"] / 15.0 + values["V"] / 10.0 + 5.0 * values["B"]
    )


def _project_wind_power_result_to_standard(
    matched_result: dict,
    standard_context: dict,
    spec_result: dict | None = None,
) -> dict:
    """把风电成分和性能候选投影到 GB/T 1591 约束可行域。

    该函数只处理可由确定性公式验证的标准上下限、CEV 和 Pcm，不设计轧制
    规程，也不回填历史值。先裁剪单项边界，再按碳当量公式逐步降低仍有余量
    的合金元素；每一步均保留标准下限和至少一种细晶元素要求。
    """
    projected = copy.deepcopy(matched_result)
    row = _matched_result_body_to_row(projected)
    actual_keys = {str(key).upper(): key for key in row}

    def field_key(name: str) -> str | None:
        upper = str(name).upper()
        if upper == "AL":
            return actual_keys.get("AL") or actual_keys.get("ALT")
        return actual_keys.get(upper)

    def numeric(name: str) -> float | None:
        key = field_key(name)
        return _to_float(row.get(key)) if key else None

    def assign(name: str, value: float, *, performance: bool = False) -> None:
        key = field_key(name)
        if not key:
            return
        formatted = (
            _format_pipeline_refined_value(name, value)
            if performance
            else f"{max(0.0, value):.4f}"
        )
        if formatted is None:
            return
        _set_pipeline_arrbody_field(projected, key, formatted)
        row[key] = formatted

    # 投影必须沿用与前置 LLM 结果校验相同的本轮 spec_result 边界。例如
    # spec_result 给出 MO=0.10~0.30 时，Pcm 修正绝不能把 Mo 降为 0。
    chemistry = _effective_wind_chemistry_limits(standard_context, spec_result)
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    for constraint, raw_limit in chemistry.items():
        element, _, bound = str(constraint).upper().partition("_")
        limit = _to_float(raw_limit)
        if limit is None:
            continue
        if bound == "MIN":
            lower_bounds[element] = limit
        elif bound == "MAX":
            upper_bounds[element] = limit

    for element in set(lower_bounds) | set(upper_bounds):
        current = numeric(element)
        if current is None:
            continue
        bounded = max(lower_bounds.get(element, current), current)
        bounded = min(upper_bounds.get(element, bounded), bounded)
        assign(element, bounded)

    refiner_limits = standard_context.get("grain_refiner_requirement") or {}
    refiner_candidates: list[tuple[str, float]] = []
    for constraint, raw_limit in refiner_limits.items():
        element, _, bound = str(constraint).upper().partition("_")
        limit = _to_float(raw_limit)
        if bound == "MIN" and limit is not None and field_key(element):
            refiner_candidates.append((element, limit))
    if refiner_candidates and not any(
        (numeric(element) or 0.0) >= limit - 1e-9
        for element, limit in refiner_candidates
    ):
        # 优先使用 Al 作为最低要求承载元素，若结构中没有 Al 再选择标准列出的
        # 第一个元素；不同时抬高多种微合金元素。
        element, limit = next(
            ((name, value) for name, value in refiner_candidates if name == "AL"),
            refiner_candidates[0],
        )
        assign(element, limit)
        lower_bounds[element] = max(lower_bounds.get(element, 0.0), limit)

    tensile = standard_context.get("tensile") or {}
    for constraint, raw_limit in tensile.items():
        performance, _, bound = str(constraint).upper().partition("_")
        limit = _to_float(raw_limit)
        current = numeric(performance)
        if limit is None or current is None:
            continue
        corrected = max(current, limit) if bound == "MIN" else min(current, limit)
        assign(performance, corrected, performance=True)
    impact = standard_context.get("impact") or {}
    impact_min = _to_float(impact.get("longitudinal"))
    current_akv = numeric("AKV")
    if impact_min is not None and current_akv is not None and current_akv < impact_min:
        assign("AKV", impact_min, performance=True)

    def reduce_formula_excess(coefficients: dict[str, float], limit_value) -> None:
        limit = _to_float(limit_value)
        if limit is None:
            return
        formula_value = sum((numeric(name) or 0.0) * coefficient for name, coefficient in coefficients.items())
        excess = formula_value - limit
        if excess <= 1e-9:
            return
        # 优先降低残余合金和强碳当量贡献元素，C、Mn 最后调整。每次保留
        # 0.002 的公式裕量，避免四位小数格式化后再次压线越界。
        for name in ("B", "MO", "CR", "CU", "NI", "V", "SI", "MN", "C"):
            coefficient = coefficients.get(name)
            current = numeric(name)
            if coefficient is None or coefficient <= 0 or current is None:
                continue
            floor = lower_bounds.get(name, 0.0)
            available = max(0.0, current - floor)
            needed = (excess + 0.002) / coefficient
            reduction = min(available, needed)
            if reduction <= 0:
                continue
            assign(name, current - reduction)
            excess -= reduction * coefficient
            if excess <= -0.001:
                break

    reduce_formula_excess(
        {"C": 1.0, "MN": 1 / 6, "CR": 1 / 5, "MO": 1 / 5, "V": 1 / 5,
         "NI": 1 / 15, "CU": 1 / 15},
        standard_context.get("CEV_max"),
    )
    reduce_formula_excess(
        {"C": 1.0, "SI": 1 / 30, "MN": 1 / 20, "CU": 1 / 20,
         "CR": 1 / 20, "NI": 1 / 60, "MO": 1 / 15, "V": 1 / 10, "B": 5.0},
        standard_context.get("Pcm_standard_max", standard_context.get("Pcm_max")),
    )
    reduce_formula_excess(
        {"C": 1.0, "SI": 1 / 30, "MN": 1 / 20, "CU": 1 / 20,
         "CR": 1 / 20, "NI": 1 / 60, "MO": 1 / 15, "V": 1 / 10, "B": 5.0},
        standard_context.get("Pcm_user_max"),
    )
    return projected


def _build_wind_power_standard_redesign_instruction(
    standard_context: dict,
    validation_error: str = "",
    spec_result: dict | None = None,
) -> str:
    """把风电标准校验结果转换成后置微调模型可直接执行的成分重设计约束。"""
    # 提示模型的单元素范围必须和投影、校验共用同一来源；否则 MO 等字段
    # 会出现“后端使用 spec_result 下限、模型只看到国标默认下限”的信息断层。
    chemistry = _effective_wind_chemistry_limits(standard_context, spec_result)
    constraints: list[str] = []
    zero_upper_fields: list[str] = []

    for field_key, limit in chemistry.items():
        element, _, bound = str(field_key).partition("_")
        try:
            numeric_limit = float(limit)
        except (TypeError, ValueError):
            continue
        if bound == "max":
            constraints.append(f"{element} <= {numeric_limit:.4f}")
            if abs(numeric_limit) <= 1e-12:
                zero_upper_fields.append(element)
        elif bound == "min":
            constraints.append(f"{element} >= {numeric_limit:.4f}")

    weldability_targets: list[str] = []
    cev_limit = _to_float(standard_context.get("CEV_max"))
    if cev_limit is not None:
        weldability_targets.append(
            f"CEV 后端公式计算值必须 <= {cev_limit:.4f}；为避免四位小数格式化后压线超限，"
            f"设计目标应保守控制在 <= {max(0.0, cev_limit - 0.01):.4f}"
        )
    standard_pcm_limit = _to_float(
        standard_context.get("Pcm_standard_max", standard_context.get("Pcm_max"))
    )
    if standard_pcm_limit is not None:
        weldability_targets.append(
            f"GB/T 1591 Pcm 后端公式计算值必须 <= {standard_pcm_limit:.4f}；建议保守控制在 "
            f"<= {max(0.0, standard_pcm_limit - 0.01):.4f}"
        )
    user_pcm_limit = _to_float(standard_context.get("Pcm_user_max"))
    if user_pcm_limit is not None:
        weldability_targets.append(
            f"用户明确要求的 Pcm 后端公式计算值必须 <= {user_pcm_limit:.4f}；建议保守控制在 "
            f"<= {max(0.0, user_pcm_limit - 0.01):.4f}"
        )
    return build_wind_power_standard_redesign_instruction_text(
        constraints,
        zero_upper_fields,
        validation_error,
        weldability_targets,
    )




def _build_langchain_qwen_agent_model():
    """为工具调用 Agent 构建非流式 Qwen 副本，不改变全局聊天模型。"""
    model_kwargs = dict(getattr(qwen_Llm, "model_kwargs", {}) or {})
    model_kwargs.update({
        "max_completion_tokens": PIPELINE_AGENT_MAX_COMPLETION_TOKENS,
        "extra_body": {"enable_thinking": False},
    })
    return qwen_Llm.model_copy(update={
        "streaming": False,
        "disable_streaming": "tool_calling",
        "model_kwargs": model_kwargs,
    })


def _build_requirement_parsing_dependencies() -> RequirementParsingDependencies:
    """构造需求解析 Agent 依赖。

    需求解析与设计变更评估、成分工艺微调复用同一个非流式 Qwen 模型配置，
    但不向解析 Agent 开放任何 RAG、MySQL 或候选校验工具。解析阶段只允许读取
    当前用户提示词和必要会话上下文，避免把知识库/历史实绩数值误写成用户要求。
    """
    return RequirementParsingDependencies(
        agent_model=_build_langchain_qwen_agent_model(),
    )


def _retrieve_design_agent_product_knowledge(is_wind: bool, query: str) -> str:
    """只开放当前产品对应的一个 RAG 工具，阻止跨钢种资料污染。"""
    from rag_tools import (
        search_pipeline_steel_knowledge_base,
        search_wind_power_steel_knowledge_base,
    )
    selected = (
        search_wind_power_steel_knowledge_base
        if is_wind else search_pipeline_steel_knowledge_base
    )
    return str(selected.invoke({"query": str(query or "").strip()}) or "")


_AGENT_HISTORY_FIELDS = {
    "SLAB_THICK", "SLAB_WIDTH", "SLAB_LEN", "AIM_THICK", "AIM_WIDTH",
    "WIDTH_ROLL_START_REMARK", "WIDTH_ROLL_END_REMARK", "R_PASS_ACT",
    "F_PASS_ACT", "FET", "FDT", "TEMP_ENTR", "SELF_TEMP",
    *{f"N{index}_{suffix}" for index in range(1, 31) for suffix in (
        "DH_CAL", "DT_CAL", "DW_CAL", "FORCE", "SPD", "ENTR_DATE",
    )},
}
_PIPELINE_AGENT_HISTORY_COMPOSITION_PERFORMANCE_FIELDS = {
    "C", "SI", "MN", "P", "S", "N", "NB", "V", "TI", "ALT", "AL", "ALS",
    "CU", "CR", "NI", "CO", "MO", "B", "YS", "TS", "EL", "AKV",
}


def _build_design_agent_history_payload(
    rows: list[dict],
    *,
    spec_result: dict,
    user_message: str,
    is_wind: bool,
) -> dict:
    """将当前目标历史实绩压缩为无身份 JSON，并声明风电数据用途边界。"""
    target, lower, upper = _resolve_pipeline_history_target_thickness(
        spec_result,
        user_message,
    )
    samples = []
    # 管线钢与风电用钢使用同一份结构化历史字段。风电分支不再在代码层删除
    # 成分/性能；其“只参考规律、不直接复用数值”的边界由系统提示词规定。
    allowed_fields = (
        set(_AGENT_HISTORY_FIELDS)
        | _PIPELINE_AGENT_HISTORY_COMPOSITION_PERFORMANCE_FIELDS
    )
    for row in rows or []:
        normalized = {str(key).upper(): value for key, value in row.items()}
        samples.append({
            key: normalized.get(key)
            for key in sorted(allowed_fields)
            if normalized.get(key) not in (None, "")
        })
    return {
        "status": "ok" if samples else "unavailable",
        "target_thickness_mm": target,
        "target_range_mm": [lower, upper],
        "sample_count": len(samples),
        "usage_policy": "CURRENT_TARGET_ENGINEERING_REFERENCE",
        "samples": samples,
    }


def _build_design_assessment_summary(matched_result: dict | None) -> dict:
    row = _matched_result_body_to_row(matched_result or {})
    summary_fields = (
        "STEEL_SIGN", "AIM_THICK", "SLAB_THICK", "C", "SI", "MN", "P", "S",
        "N", "NB", "V", "TI", "AL", "ALS", "CU", "CR", "NI", "CO", "MO",
        "B", "YS", "TS", "EL", "AKV", "FET", "FDT", "TEMP_ENTR", "SELF_TEMP",
        "R_PASS_ACT", "F_PASS_ACT",
    )
    return {
        field: row.get(field)
        for field in summary_fields
        if row.get(field) not in (None, "")
    }


def _structured_agent_error(
    *, module: str, field: str, rule: str, message: str, current_values: dict | None = None,
) -> dict:
    return {
        "module": module,
        "field": field,
        "rule": rule,
        "status": "FAIL",
        "current_values": current_values or {},
        "message": message,
    }


def _validate_refinement_agent_candidate(
    original: dict,
    candidate: dict,
    spec_result: dict,
    is_wind: bool,
    engineering_standard_context: dict | None = None,
) -> list[dict]:
    """给微调 Agent 的快速工具校验；最终后置门禁仍完整执行。"""
    errors: list[dict] = []
    if not isinstance(candidate, dict):
        return [_structured_agent_error(
            module="structure", field="matched_result", rule="JSON object",
            message="候选结果不是 JSON 对象",
        )]
    if set(candidate) != set(original):
        errors.append(_structured_agent_error(
            module="structure", field="matched_result", rule="same top-level keys",
            current_values={
                "missing": sorted(set(original) - set(candidate)),
                "extra": sorted(set(candidate) - set(original)),
            },
            message="顶层字段集合必须与输入完全一致",
        ))
        return errors
    original_body = original.get("arrBody") or []
    candidate_body = candidate.get("arrBody") or []
    original_keys = [_get_arrbody_key(item) for item in original_body]
    candidate_keys = [_get_arrbody_key(item) for item in candidate_body]
    if original_keys != candidate_keys:
        errors.append(_structured_agent_error(
            module="structure", field="arrBody", rule="same field order",
            message="arrBody长度、字段名或顺序与输入不一致",
        ))
        return errors

    row = _matched_result_body_to_row(candidate)
    for field_name in (
        set(PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC)
        | set(PIPELINE_REFINABLE_PERFORMANCE_FIELD_TO_SPEC)
    ):
        if field_name in row and not _pipeline_value_within_spec_bounds(
            field_name,
            row.get(field_name),
            spec_result,
        ):
            errors.append(_structured_agent_error(
                module=("performance" if field_name in PIPELINE_PERFORMANCE_FIELDS else "composition"),
                field=field_name,
                rule="spec_result bounds",
                current_values={field_name: row.get(field_name)},
                message=_build_pipeline_refinement_bound_error(
                    field_name,
                    row.get(field_name),
                    spec_result,
                ),
            ))
    for message in _remove_refinement_turn_width_errors_for_agent(
        _collect_pipeline_deformation_pass_errors(
            candidate,
            validate_timing=False,
            validate_cooling_timing=False,
        )
    ):
        match = re.search(r"\b([A-Z][A-Z0-9_]*)\b", str(message))
        errors.append(_structured_agent_error(
            module="rolling",
            field=match.group(1) if match else "rolling_schedule",
            rule="rolling schedule consistency",
            message=str(message),
        ))
    errors.extend(_collect_pipeline_strict_cooling_gate_errors(candidate))
    if is_wind and engineering_standard_context:
        wind_error = _validate_wind_power_matched_result(
            candidate,
            engineering_standard_context,
            spec_result,
        )
        if wind_error:
            errors.append(_structured_agent_error(
                module="composition",
                field="wind_standard",
                rule="GB/T 1591 and CEV/Pcm",
                message=wind_error,
            ))
    return errors


def _remove_refinement_turn_width_errors_for_agent(errors: list[str]) -> list[str]:
    prefixes = (
        "转钢道次标识无效", "转钢宽度变化次数无效", "转钢宽度变化位置无效",
        "转钢标记与宽度变化不一致", "WIDTH_ROLL_START_REMARK=",
        "WIDTH_ROLL_END_REMARK=",
    )
    return [error for error in errors if not str(error or "").startswith(prefixes)]


def _build_composition_refinement_dependencies(
    *,
    is_wind: bool = False,
    historical_rows: list[dict] | None = None,
    spec_result: dict | None = None,
    user_message: str = "",
    engineering_standard_context: dict | None = None,
) -> CompositionRefinementDependencies:
    """汇集后置微调所需依赖；业务实现位于 pipeline_agents.py。"""
    return CompositionRefinementDependencies(
        extract_target_thickness=_extract_pipeline_target_thickness_from_text,
        extract_target_slab_thickness=_extract_pipeline_target_slab_thickness_from_text,
        lock_explicit_thickness_targets=_lock_explicit_pipeline_thickness_targets,
        is_context_modification_request=_is_context_based_design_modification_request,
        build_refinement_rag_context=_build_pipeline_refinement_rag_context,
        build_cross_route_context=_build_cross_route_context,
        get_recent_session_context=_get_recent_session_context,
        filter_wind_session_context=_filter_wind_power_session_context,
        component_fields=frozenset(PIPELINE_REFINABLE_COMPONENT_FIELD_TO_SPEC),
        performance_fields=frozenset(PIPELINE_REFINABLE_PERFORMANCE_FIELD_TO_SPEC),
        roll_fields=frozenset(PIPELINE_REFINABLE_ROLL_FIELDS),
        get_arrbody_key=_get_arrbody_key,
        build_historical_roll_reference=_build_pipeline_historical_roll_reference,
        build_wind_standard_redesign_instruction=_build_wind_power_standard_redesign_instruction,
        reasoning_cache=_LLM_REASONING_CONTENT_CACHE,
        invoke_qwen=official_qwen_sync.invoke,
        parse_json_object=_parse_json_object,
        extract_qwen_agent_response=_extract_qwen_agent_response,
        restore_arrbody_fields=_restore_pipeline_arrbody_fields,
        sanitize_refined_result=_sanitize_pipeline_refined_matched_result,
        validate_wind_result=_validate_wind_power_matched_result,
        project_wind_result=_project_wind_power_result_to_standard,
        extract_wind_user_pcm_max=_extract_wind_user_pcm_max_with_llm,
        normalize_declared_pass_tail=_normalize_pipeline_declared_pass_tail,
        collect_deformation_pass_errors=_collect_pipeline_deformation_pass_errors,
        roll_errors_require_global_redesign=_pipeline_roll_errors_require_global_redesign,
        prepare_full_roll_redesign_baseline=_prepare_pipeline_full_roll_redesign_baseline,
        normalize_deformation_passes=_normalize_pipeline_deformation_passes,
        validate_dll_time_encodings=lambda result, include_cooling_start: (
            _validate_pipeline_dll_time_encodings(
                result,
                include_cooling_start=include_cooling_start,
            )
        ),
        enforce_performance_standard=_enforce_pipeline_performance_standard,
        cache_performance_baseline=_cache_pipeline_performance_baseline,
        performance_values=_pipeline_performance_values,
        max_completion_tokens=PIPELINE_AGENT_MAX_COMPLETION_TOKENS,
        agent_model=_build_langchain_qwen_agent_model(),
        retrieve_agent_knowledge=lambda query: _retrieve_design_agent_product_knowledge(
            is_wind,
            query,
        ),
        retrieve_agent_history=lambda: _build_design_agent_history_payload(
            historical_rows or [],
            spec_result=spec_result or {},
            user_message=user_message,
            is_wind=is_wind,
        ),
        validate_agent_candidate=lambda original, candidate, current_spec, current_is_wind: (
            _validate_refinement_agent_candidate(
                original,
                candidate,
                current_spec,
                current_is_wind,
                engineering_standard_context,
            )
        ),
        validate_initial_cooling=_collect_pipeline_strict_cooling_gate_errors,
    )


def _build_design_change_assessment_dependencies(
    *,
    is_wind: bool,
    historical_rows: list[dict],
    spec_result: dict,
    user_message: str,
) -> DesignChangeAssessmentDependencies:
    return DesignChangeAssessmentDependencies(
        agent_model=_build_langchain_qwen_agent_model(),
        retrieve_product_knowledge=lambda query: _retrieve_design_agent_product_knowledge(
            is_wind,
            query,
        ),
        retrieve_current_target_history=lambda: _build_design_agent_history_payload(
            historical_rows,
            spec_result=spec_result,
            user_message=user_message,
            is_wind=is_wind,
        ),
    )


def _refine_pipeline_matched_result_with_llm(
    spec_result: dict,
    matched_result: dict,
    user_message: str,
    session_id: str,
    material_name: str = "管线钢",
    engineering_standard_context: dict | None = None,
    historical_roll_reference_markdown: str = "",
) -> dict:
    """兼容原调用名；后置微调业务已迁移到 pipeline_agents.py。"""
    return refine_composition_process_performance(
        spec_result,
        matched_result,
        user_message,
        session_id,
        material_name=material_name,
        engineering_standard_context=engineering_standard_context,
        historical_roll_reference_markdown=historical_roll_reference_markdown,
        dependencies=_build_composition_refinement_dependencies(
            is_wind=bool(engineering_standard_context),
            spec_result=spec_result,
            user_message=user_message,
            engineering_standard_context=engineering_standard_context,
        ),
    )


def _refine_pipeline_unstrict_matched_result_with_llm(
    spec_result: dict,
    matched_result: dict,
    user_message: str,
    session_id: str,
) -> dict:
    """兼容旧调用名；实际微调逻辑已统一迁移到后置通用函数。"""
    return _refine_pipeline_matched_result_with_llm(
        spec_result,
        matched_result,
        user_message,
        session_id,
    )


def _expand_pipeline_spec_with_llm(
    spec_result: dict,
    user_message: str,
    session_id: str,
    attempt: int,
    last_spec: dict,
    material_name: str = "管线钢",
) -> dict | None:
    """让 LLM 在原规格基础上适度扩大成分范围，用于最后阶段迭代查询。"""
    session_context = _get_recent_session_context(session_id)
    if material_name == "陆上风电塔筒用TMCP钢板":
        session_context = _filter_wind_power_session_context(session_context)
    prompt = build_pipeline_expand_spec_prompt(
        material_name, PIPELINE_MYSQL_TABLE, user_message, session_context,
        spec_result, last_spec, attempt,
    )
    try:
        raw = deepseek_Llm.invoke(prompt)
        text = getattr(raw, "content", raw)
        expanded = _parse_json_object(str(text))
        if not isinstance(expanded, dict):
            return None
        merged = dict(last_spec)
        for key in merged:
            if key in expanded:
                merged[key] = expanded[key]
        return merged
    except Exception as exc:
        print(f"[管线钢MySQL匹配] LLM扩大规格失败: {exc}")
        return None


def _pipeline_mysql_grade_scope(user_message: str) -> tuple[str | None, str | None, str | None]:
    """把用户目标牌号映射为优先检索的历史实绩牌号族。"""
    target_grade = _extract_pipeline_target_grade(user_message)
    reference_grade = _select_pipeline_dll_reference_grade(target_grade)
    if reference_grade == "X80NG":
        return target_grade, reference_grade, "%X80%"
    if reference_grade in {"X65", "X70"}:
        return target_grade, reference_grade, f"%{reference_grade}%"
    return target_grade, reference_grade, None


def match_pipeline_steel_process(
    spec_result: dict,
    user_message: str,
    session_id: str,
    material_name: str = "管线钢",
) -> dict:
    """优先在映射牌号族内逐级匹配，三轮成分扩展后再解除牌号限制。"""
    query_errors = []
    target_grade, reference_grade, steel_sign_like = _pipeline_mysql_grade_scope(user_message)
    scope_label = (
        f"目标{target_grade}→{reference_grade}实绩族"
        if target_grade and reference_grade
        else "未指定牌号"
    )
    print(
        f"[管线钢MySQL匹配] 牌号优先策略: {scope_label}, "
        f"STEEL_SIGN LIKE {steel_sign_like or '%%'}"
    )

    def _run_query_plan(current_spec: dict, query_plan: list[tuple], grade_like: str | None):
        for include_process, include_performance, include_slab_thickness, is_state, stage_name in query_plan:
            try:
                row = _query_first_pipeline_mysql_row(
                    current_spec,
                    include_process=include_process,
                    include_performance=include_performance,
                    include_slab_thickness=include_slab_thickness,
                    stage_name=stage_name,
                    steel_sign_like=grade_like,
                )
                if row:
                    return _build_pipeline_match_response(
                        row,
                        is_state=is_state,
                        session_id=session_id,
                    )
            except Exception as exc:
                error = _format_match_error(f"{stage_name}失败", exc)
                query_errors.append(error)
                print(f"[管线钢MySQL匹配] {error}")
        return None

    # 第一轮：按目标牌号映射后的 X65/X70/X80 实绩族，依次放宽板坯厚度、性能和工艺。
    initial_plan = [
        (True, True, True, True, f"{scope_label}严格查询"),
        (True, True, False, False, f"{scope_label}仅放开板坯厚度查询"),
        (True, False, False, False, f"{scope_label}继续放开性能查询"),
        (False, False, False, False, f"{scope_label}继续放开工艺查询"),
    ]
    response = _run_query_plan(spec_result, initial_plan, steel_sign_like)
    if response:
        return response
    if query_errors:
        return _build_pipeline_match_response(
            None,
            is_state=False,
            session_id=session_id,
            message=f"MySQL匹配失败: {query_errors[0]}",
            error="; ".join(query_errors),
        )

    # 在牌号族内只保留成分和成品厚度约束，最多三次逐步扩大成分范围。
    current_spec = dict(spec_result)
    for attempt in range(1, 4):
        expanded = _expand_pipeline_spec_with_llm(
            spec_result,
            user_message,
            session_id,
            attempt,
            current_spec,
            material_name=material_name,
        )
        if not expanded:
            continue
        current_spec = expanded
        response = _run_query_plan(
            current_spec,
            [(False, False, False, False, f"{scope_label}第{attempt}轮成分范围扩大查询")],
            steel_sign_like,
        )
        if response:
            return response
        if query_errors:
            return _build_pipeline_match_response(
                None,
                is_state=False,
                session_id=session_id,
                message=f"MySQL匹配失败: {query_errors[0]}",
                error="; ".join(query_errors),
            )

    # 第二轮：只有映射牌号族始终无结果时才取消 STEEL_SIGN 条件，并以最后扩大后的
    # 成分范围重新执行一次“严格→放开板坯厚度→放开性能→放开工艺”逻辑。
    if steel_sign_like:
        unrestricted_plan = [
            (True, True, True, False, "放开牌号限制后严格条件查询"),
            (True, True, False, False, "放开牌号限制后仅放开板坯厚度查询"),
            (True, False, False, False, "放开牌号限制后继续放开性能查询"),
            (False, False, False, False, "放开牌号限制后继续放开工艺查询"),
        ]
        response = _run_query_plan(current_spec, unrestricted_plan, None)
        if response:
            return response
        if query_errors:
            return _build_pipeline_match_response(
                None,
                is_state=False,
                session_id=session_id,
                message=f"MySQL匹配失败: {query_errors[0]}",
                error="; ".join(query_errors),
            )

    # 映射牌号、三轮成分扩展和解除牌号限制均无命中时，最终去掉全部筛选条件取首条。
    try:
        row = _query_first_pipeline_mysql_row_without_filters(stage_name="最终无筛选兜底查询")
        if row:
            return _build_pipeline_match_response(row, is_state=False, session_id=session_id)
    except Exception as exc:
        error = _format_match_error("最终无筛选兜底查询失败", exc)
        print(f"[管线钢MySQL匹配] {error}")
        return _build_pipeline_match_response(
            None,
            is_state=False,
            session_id=session_id,
            message=f"MySQL匹配失败: {error}",
            error=error,
        )

    return _build_pipeline_match_response(
        None,
        is_state=False,
        session_id=session_id,
        message="最终无筛选兜底未返回首条 MySQL 实绩，请检查实绩表是否为空",
    )


def match_wind_power_steel_process(spec_result: dict, user_message: str, session_id: str) -> dict:
    """为风电塔筒钢板取得完整管线钢历史实绩参考。

    现有 MySQL 表仅有管线钢历史实绩。风电分支与管线钢使用相同查询和完整字段，
    由 Agent 参考其成分、性能和工艺规律，但不得直接复用或高度仿照具体数值；
    后续仍由风电系统提示词、GB/T 1591、CEV/Pcm 和确定性校验约束最终方案。
    """
    response = match_pipeline_steel_process(
        spec_result,
        user_message,
        session_id,
        material_name="陆上风电塔筒用TMCP钢板",
    )
    if isinstance(response, dict):
        response["isState"] = False
        response["message"] = (
            "已取得管线钢热轧实绩参考；成分和性能只用于规律判断，"
            "不得直接复用或高度仿照，最终以风电用钢标准和用户要求为准。"
        )
    return response


# ============================================================
# 流式生成函数（接收消息列表，支持多轮对话上下文）
# ============================================================

async def stream_async(messages: list):
    """
    使用 astream 进行异步流式生成（首选方案）
    Use astream for async streaming (preferred approach)

    LangChain 的 astream 接收消息列表作为输入，
    模型会根据完整历史上下文生成下一个 token。

    Args:
        messages: LangChain 消息列表 [HumanMessage, AIMessage, ...]
                  包含完整的对话历史（已被裁剪到 50 轮以内）

    Yields:
        NDJSON 格式字符串: {"content": "文本块"}\n
    """
    try:
        # astream 接收消息列表，模型基于完整上下文逐 token 生成
        # astream accepts message list, model generates token-by-token with full context
        async for chunk in deepseek_Llm.astream(messages):
            if chunk.content:
                data = json.dumps(
                    {"content": chunk.content},
                    ensure_ascii=False
                )
                yield f"{data}\n"
    except Exception as e:
        error_data = json.dumps(
            {"error": f"生成回复时出错: {str(e)}"},
            ensure_ascii=False,
        )
        yield f"{error_data}\n"


def stream_sync(messages: list):
    """
    使用 stream 进行同步流式生成（备用方案）
    Use sync stream for streaming (fallback approach)

    Args:
        messages: LangChain 消息列表（同 stream_async）

    Yields:
        与 stream_async 相同的 NDJSON 格式
    """
    try:
        for chunk in deepseek_Llm.stream(messages):
            if chunk.content:
                data = json.dumps(
                    {"content": chunk.content},
                    ensure_ascii=False,
                )
                yield f"{data}\n"
    except Exception as e:
        error_data = json.dumps(
            {"error": f"生成回复时出错: {str(e)}"},
            ensure_ascii=False,
        )
        yield f"{error_data}\n"


async def generate_stream(messages: list):
    """
    智能流式生成器 — 优先异步，失败时降级到同步
    Smart streaming generator — prefer async, fallback to sync

    Args:
        messages: LangChain 消息列表（含完整对话历史）
    """
    try:
        async for line in stream_async(messages):
            yield line
    except Exception as async_error:
        print(f"[警告] astream 失败，降级到同步流式: {async_error}")
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            sync_gen = stream_sync(messages)

            def next_chunk():
                try:
                    return next(sync_gen)
                except StopIteration:
                    return None

            loop = asyncio.get_running_loop()
            while True:
                chunk = await loop.run_in_executor(executor, next_chunk)
                if chunk is None:
                    break
                yield chunk

            executor.shutdown(wait=False)
        except Exception as sync_error:
            error_data = json.dumps(
                {
                    "error": (
                        f"流式生成完全失败。"
                        f"异步错误: {async_error}; 同步错误: {sync_error}"
                    )
                },
                ensure_ascii=False,
            )
            yield f"{error_data}\n"


# ============================================================
# API 路由，处理聊天请求流式处理
# ============================================================

@app.get("/generated-images/{token}")
async def get_generated_image(token: str):
    image_path = _resolve_generated_image_path(token)
    if not image_path or not _os.path.isfile(image_path):
        return JSONResponse({"error": "图片不存在或已过期"}, status_code=404)
    return FileResponse(
        image_path,
        media_type="image/png",
        filename=_os.path.basename(image_path),
    )


@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    POST /chat — 聊天接口（带会话管理），返回流式 NDJSON 响应
    Chat endpoint with session management, returns streaming NDJSON response

    请求格式 / Request format:
        Content-Type: application/json
        Body: {
            "message": "用户消息文本",
            "session_id": "550e8400-e29b-41d4-a716-446655440000"
        }

    会话流程 / Session Flow:
        1. 根据 session_id 查找或创建会话
        2. 将用户消息追加到会话历史
        3. 裁剪历史到 50 轮
        4. 将完整历史传给 LLM 进行流式生成
        5. 收集完整 AI 回复，追加到会话历史
        6. 同时流式返回给前端c
    """
    # 解析 JSON 请求体
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={"error": "无效的请求格式，请发送 JSON 数据"},
            status_code=400,
        )

    # 提取并验证消息和会话 ID
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse(
            content={"error": "消息不能为空"},
            status_code=400,
        )

    session_id = body.get("session_id", "").strip()
    if not session_id:
        return JSONResponse(
            content={"error": "缺少 session_id"},
            status_code=400,
        )

    # ==========================================================
    # 会话管理：获取或创建会话，构建 LLM 上下文
    # ==========================================================
    session = get_or_create_session(session_id)

    # 本地构建 LLM 消息列表（含当前用户消息），暂不持久化
    llm_messages = list(session["messages"]) + [HumanMessage(content=user_message)]
    llm_messages = trim_history(llm_messages, MAX_TURNS)

    # ==========================================================
    # 意图分类 — 同步阻塞执行，先用意图分类结果判断再决定后续逻辑
    # ==========================================================
    intent_result = None
    deterministic_intent = _deterministic_intent_override(user_message)
    if deterministic_intent:
        intent_result = {"intent": deterministic_intent}
        print(f"[意图分类] 明确语义规则命中: {user_message[:40]}... → {deterministic_intent}")
    else:
        try:
            intent_result = classify_with_rag(
                system_prompt=INTENT_SYSTEM_PROMPT,
                user_message=user_message,
                session_id=f"intent_{session_id}",
                json_schema=INTENT_JSON_SCHEMA,
                db_name="Nb_KnowBase_db",
            )
            print(f"[意图分类] {user_message[:40]}... → {intent_result}")
        except Exception as e:
            print(f"[意图分类] 失败: {e}")
            intent_result = {"intent": "CHAT"}

    # ==========================================================
    # 流式生成 + 收集完整回复
    # ==========================================================
    full_response_chunks = []

    async def response_generator():
        """
        包装流式生成器：
        0. 先返回意图分类结果（第一条 NDJSON）
        1. 然后流式返回聊天回复
        2. 收集完整回复，最后一次性持久化用户消息 + AI 回复
        """
        # 先发送意图分类结果
        yield json.dumps({"intent": intent_result.get("intent", "CHAT")}, ensure_ascii=False) + "\n"

        async for line in generate_stream(llm_messages):
            # 收集流式内容块
            try:
                data = json.loads(line.strip())
                if "content" in data and data["content"]:
                    full_response_chunks.append(data["content"])
            except json.JSONDecodeError:
                pass

            yield line

        # 流式完成后一次性持久化用户消息 + AI 回复
        ai_response_text = "".join(full_response_chunks)
        if ai_response_text:
            chat_session_store.add_messages_batch(
                session_id,
                [HumanMessage(content=user_message), AIMessage(content=ai_response_text)],
            )
        else:
            # 即使没有 AI 回复内容，也保存用户消息
            chat_session_store.add_message(session_id, HumanMessage(content=user_message))

    return StreamingResponse(
        response_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ============================================================
# API 路由，处理根路径重定向
## ============================================================
@app.get("/")
async def root():
    """
    GET / — 根路径重定向到前端页面
    """
    return RedirectResponse(url="/index.html")


# ============================================================
# 智能路由端点：意图分类 + 条件流式对话
# ============================================================
# ============================================================
# Chat Agent 辅助函数 — 提取复用，支持不同系统提示词
# ============================================================
def _build_chat_agent_response(
    session_id: str,
    user_message: str,
    system_prompt: str,
    persisted_user_message: str | None = None,
):
    """
    构建 Chat Agent 流式响应（@tool RAG + LLM自主判断 + astream）
    供 CHAT 路径和 DESIGN→其他聊天 路径复用

    使用 PersistentChatMessageHistory 持久化存储对话历史，
    服务器重启后数据不丢失。修复了 AI 回复未存储的 bug。
    """
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    from rag_tools import KNOWLEDGE_BASE_TOOLS, KNOWLEDGE_BASE_TOOL_MAP

    llm_with_tools = deepseek_Llm.bind_tools(KNOWLEDGE_BASE_TOOLS)

    # 使用持久化历史（替代原来的 InMemoryChatMessageHistory + _chat_histories dict）
    history = PersistentChatMessageHistory(agent_chat_store, session_id)
    cross_route_context = _build_cross_route_context(session_id)
    safe_history_messages = [
        message
        for message in history.messages
        if "DSML" not in str(getattr(message, "content", ""))
        and "tool_calls" not in str(getattr(message, "content", ""))
        and _parse_chat_json_tool_call(
            str(getattr(message, "content", "")),
            user_message,
            KNOWLEDGE_BASE_TOOL_MAP,
        ) is None
    ]

    msgs = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=KNOWLEDGE_BASE_TOOL_ROUTING_PROMPT),
        *([SystemMessage(content=build_cross_route_context_system_prompt(cross_route_context))]
          if cross_route_context else []),
        *safe_history_messages,
        HumanMessage(content=user_message),
    ]

    def parse_text_tool_call(text: str) -> tuple[str, str] | None:
        """兼容模型偶发输出的 JSON 或 DSML 文本工具调用格式。"""
        if not text:
            return None
        json_tool_call = _parse_chat_json_tool_call(
            text,
            user_message,
            KNOWLEDGE_BASE_TOOL_MAP,
        )
        if json_tool_call:
            return json_tool_call
        tool_name = next(
            (name for name in KNOWLEDGE_BASE_TOOL_MAP if name in text),
            None,
        )
        if not tool_name:
            return None
        match = re.search(
            r'parameter\s+name=["\']query["\'][^>]*>([\s\S]*?)'
            r'(?:</\s*parameter\s*>|<\s*[｜|]{0,2}\s*DSML|$)',
            text,
            flags=re.IGNORECASE,
        )
        if match:
            query = match.group(1).strip()
            if query:
                return tool_name, query
        return tool_name, user_message

    def is_internal_tool_markup(text: str) -> bool:
        return bool(text) and (
            "DSML" in text
            or "tool_calls" in text
            or _parse_chat_json_tool_call(text, user_message, KNOWLEDGE_BASE_TOOL_MAP) is not None
        )

    def format_exception_chain(exc: Exception) -> str:
        """展开 OpenAI/httpx 包装的异常链，避免日志中只剩 Connection error。"""
        parts = []
        current = exc
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            parts.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        return " <- ".join(parts)

    def is_retryable_connection_error(exc: Exception) -> bool:
        """仅对连接建立、连接重置和临时网络错误重试，不重试参数或鉴权错误。"""
        detail = format_exception_chain(exc).lower()
        markers = (
            "connection error",
            "apiconnectionerror",
            "connecterror",
            "connection reset",
            "connection aborted",
            "server disconnected",
            "remoteprotocolerror",
        )
        return any(marker in detail for marker in markers)

    async def invoke_with_connection_retry(call, stage_name: str):
        """DeepSeek 偶发断连时重试一次，并把底层异常链保留在服务端日志。"""
        for attempt in range(2):
            try:
                return await call()
            except Exception as exc:
                detail = format_exception_chain(exc)
                print(f"[Agent] {stage_name}第 {attempt + 1}/2 次调用失败: {detail}")
                if attempt >= 1 or not is_retryable_connection_error(exc):
                    raise
                await asyncio.sleep(1.0)

    async def generator():
        full_response_chunks = []  # 收集完整 AI 回复（修复 bug）
        try:
            first = await invoke_with_connection_retry(
                lambda: llm_with_tools.ainvoke(msgs),
                "工具判断",
            )
            should_generate_after_tool = False

            if getattr(first, "tool_calls", None):
                msgs.append(first)
                selected_tool_name = None
                for tc in first.tool_calls:
                    tool_name = str(tc.get("name", "") or "")
                    tool_item = KNOWLEDGE_BASE_TOOL_MAP.get(tool_name)
                    if tool_item is None:
                        tool_result = (
                            f"未执行未知知识库工具 {tool_name or 'unknown'}；"
                            "请直接依据专业知识回答，不得改用其他用途知识库。"
                        )
                    elif selected_tool_name is not None:
                        tool_result = (
                            f"未执行 {tool_name}：每次回答最多只允许调用一个知识库工具，"
                            f"本轮已选择 {selected_tool_name}。"
                        )
                    else:
                        tool_args = tc.get("args", {})
                        query = (
                            tool_args.get("query")
                            if isinstance(tool_args, dict)
                            else None
                        ) or user_message
                        tool_result = tool_item.invoke({"query": query})
                        selected_tool_name = tool_name
                    msgs.append(ToolMessage(content=tool_result, tool_call_id=tc.get("id", "unknown")))
                should_generate_after_tool = True
                print(f"[Agent] LangChain知识库工具调用完成: selected={selected_tool_name or 'none'}")
            else:
                first_content = getattr(first, "content", "")
                if not isinstance(first_content, str):
                    first_content = str(first_content)
                text_tool_call = parse_text_tool_call(first_content)
                if text_tool_call:
                    text_tool_name, text_query = text_tool_call
                    tool_result = KNOWLEDGE_BASE_TOOL_MAP[text_tool_name].invoke({"query": text_query})
                    msgs.append(SystemMessage(
                        content=(
                            f"模型请求调用 {text_tool_name}。以下是后端代为执行的检索结果；"
                            "请基于检索结果回答用户原始问题，不要输出任何工具调用标记。\n\n"
                            f"检索问题：{text_query}\n\n检索结果：\n{tool_result}"
                        )
                    ))
                    should_generate_after_tool = True
                    print(f"[Agent] 文本知识库工具调用兜底完成: selected={text_tool_name}")
                elif first_content and not is_internal_tool_markup(first_content):
                    full_response_chunks.append(first_content)
                    yield json.dumps({"content": first_content}, ensure_ascii=False) + "\n"
                    print("[Agent] 无需检索，直接回答")

            if should_generate_after_tool:
                reasoning_started = False
                answer_started = False
                for attempt in range(2):
                    try:
                        async for event in official_deepseek_async.astream(msgs):
                            if event['type'] == 'reasoning':
                                if not reasoning_started:
                                    yield json.dumps({"reasoning_start": True}, ensure_ascii=False) + "\n"
                                    reasoning_started = True
                                yield json.dumps({"reasoning": event['text']}, ensure_ascii=False) + "\n"
                            elif event['type'] == 'content':
                                if is_internal_tool_markup(event['text']):
                                    print("[Agent] 已过滤模型内部工具调用标记")
                                    continue
                                answer_started = True
                                full_response_chunks.append(event['text'])
                                yield json.dumps({"content": event['text']}, ensure_ascii=False) + "\n"
                        break
                    except Exception as exc:
                        detail = format_exception_chain(exc)
                        print(f"[Agent] 回答流第 {attempt + 1}/2 次调用失败: {detail}")
                        # 已输出部分回答时重试会造成正文重复，因此只允许在正文开始前重试。
                        if attempt >= 1 or answer_started or not is_retryable_connection_error(exc):
                            raise
                        await asyncio.sleep(1.0)

            # 修复：同时存储用户消息和完整的 AI 回复
            # 附件全文只参与当前模型调用；持久历史只记录用户原始问题。
            history.add_message(HumanMessage(content=persisted_user_message or user_message))
            ai_text = "".join(full_response_chunks)
            if ai_text:
                history.add_message(AIMessage(content=ai_text))
        except Exception as e:
            print(f"[Agent] 失败: {format_exception_chain(e)}")
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _remember_agent_context_turn(session_id: str, user_message: str, assistant_summary: str) -> None:
    """把非 CHAT 路由的用户可见轮次同步到普通聊天上下文，便于后续追问。"""
    try:
        history = PersistentChatMessageHistory(agent_chat_store, session_id)
        history.add_message(HumanMessage(content=user_message))
        if assistant_summary:
            history.add_message(AIMessage(content=assistant_summary))
    except Exception as exc:
        print(f"[上下文同步] 写入普通聊天上下文失败: {exc}")


def _safe_text_preview(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").replace("\x00", "").split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _doc_value(doc, key: str, default=None):
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _doc_metadata(doc) -> dict:
    metadata = _doc_value(doc, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def _serialize_retrieval_docs(docs: list[dict]) -> list[dict]:
    items = []
    for index, doc in enumerate(docs or [], start=1):
        metadata = _doc_metadata(doc)
        source = _doc_value(doc, "source") or metadata.get("source") or "unknown"
        content = (
            _doc_value(doc, "content_preview")
            or _doc_value(doc, "content")
            or _doc_value(doc, "page_content")
            or ""
        )
        title = (
            metadata.get("title")
            or metadata.get("document_title")
            or metadata.get("file_name")
            or _os.path.basename(str(source))
            or f"文档 {index}"
        )
        items.append({
            "rank": _doc_value(doc, "rank", index),
            "title": title,
            "summary": _safe_text_preview(content),
            "source": source,
            "section": metadata.get("section") or metadata.get("heading") or metadata.get("chapter") or "",
            "page": metadata.get("page") or metadata.get("page_number") or metadata.get("page_index") or "",
            "chunk_id": metadata.get("chunk_id") or metadata.get("doc_id") or "",
            "score": _doc_value(doc, "score"),
        })
    return items


def _format_retrieval_docs_markdown(docs: list[dict]) -> str:
    if not docs:
        return "未检索到耐磨钢知识库资料。"

    sections = []
    for item in _serialize_retrieval_docs(docs):
        metadata_lines = []
        if item["source"]:
            metadata_lines.append(f"来源：{item['source']}")
        if item["section"]:
            metadata_lines.append(f"section：{item['section']}")
        if item["page"]:
            metadata_lines.append(f"页码：{item['page']}")
        if item["chunk_id"]:
            metadata_lines.append(f"chunk_id：{item['chunk_id']}")

        meta = "  \n".join(metadata_lines)
        sections.append(
            f"{item['rank']}. **{item['title']}**\n\n"
            f"{item['summary'] or '（无片段内容）'}\n\n"
            f"{meta}"
        )
    return "\n\n---\n\n".join(sections)


def _format_retrieval_docs_for_prompt(docs: list[dict], max_chars_per_doc: int = 900) -> str:
    """把检索结果压缩为模型参考资料，不直接展示到前端。"""
    if not docs:
        return "（未检索到耐磨钢知识库参考资料，请基于材料学常识保守给出初步方案。）"

    sections = []
    for item in _serialize_retrieval_docs(docs):
        meta_parts = []
        if item["source"]:
            meta_parts.append(f"来源={item['source']}")
        if item["section"]:
            meta_parts.append(f"section={item['section']}")
        if item["page"]:
            meta_parts.append(f"页码={item['page']}")
        if item["chunk_id"]:
            meta_parts.append(f"chunk_id={item['chunk_id']}")
        meta = "；".join(meta_parts) if meta_parts else "无元数据"
        summary = (item["summary"] or "").strip()
        if len(summary) > max_chars_per_doc:
            summary = summary[:max_chars_per_doc].rstrip() + "..."
        sections.append(
            f"[{item['rank']}] 标题：{item['title']}\n"
            f"元数据：{meta}\n"
            f"片段：{summary or '（无片段内容）'}"
        )
    return "\n\n".join(sections)


def _build_flash_design_preview_messages(
    user_message: str,
    docs: list[dict],
    purpose: str = "耐磨钢",
) -> list:
    """构造 Qwen 用于前置输出的初步设计方案提示词。"""
    if purpose in {"管线钢", "风电用钢"}:
        material_label = get_wind_power_material_label(user_message) if purpose == "风电用钢" else "管线钢"
        flash_system_prompt = PIPELINE_DESIGN_PREVIEW_SYSTEM_PROMPT
        docs_label = "管线钢知识库参考资料"
        if purpose == "风电用钢":
            # 风电初步方案使用独立提示词，避免复用文本中的油气场景、X 系列举例
            # 或内部仿真参考信息进入模型上下文。
            flash_system_prompt = WIND_POWER_DESIGN_PREVIEW_SYSTEM_PROMPT
            docs_label = "风电用钢知识库参考资料"
    else:
        flash_system_prompt = WEAR_STEEL_DESIGN_PREVIEW_SYSTEM_PROMPT
        docs_label = "耐磨钢知识库参考资料"

    if docs:
        flash_user_prompt = build_flash_design_preview_user_prompt(
            user_message, docs_label, _format_retrieval_docs_for_prompt(docs)
        )
    else:
        flash_user_prompt = build_flash_design_preview_user_prompt(
            user_message, docs_label
        )
    return [
        SystemMessage(content=flash_system_prompt),
        HumanMessage(content=flash_user_prompt),
    ]


def _sanitize_flash_design_preview_text(text: str, purpose: str = "耐磨钢", user_message: str = "") -> str:
    """初步方案展示前移除牌号/钢级/内部钢种号，避免前端出现实绩或标准牌号信息。"""
    target_name = (
        f"目标{get_wind_power_material_label(user_message)}" if purpose == "风电用钢"
        else "目标管线钢" if purpose == "管线钢" else "目标耐磨钢"
    )
    sanitized_lines = []
    grade_line_pattern = re.compile(
        r"(本方案以|以.*为目标|目标.*钢级|钢级标识|牌号|钢种号|标准牌号|L\d{3,4}|X\d{2,3}|NM\d{3})",
        flags=re.IGNORECASE,
    )
    for line in str(text or "").splitlines():
        if grade_line_pattern.search(line):
            if any(keyword in line for keyword in ["本方案以", "为目标", "钢级标识", "牌号", "钢种号", "标准牌号"]):
                continue
        sanitized_lines.append(line)
    sanitized = "\n".join(sanitized_lines)
    replacements = [
        (r"(?<![A-Za-z0-9])X\d{2,3}[A-Z]{0,4}(?:-\d+)?(?![A-Za-z0-9])", target_name),
        (r"(?<![A-Za-z0-9])L\d{3,4}[A-Z]?(?![A-Za-z0-9])", target_name),
        (r"(?<![A-Za-z0-9])NM\d{3}(?:D/E|D|E)?(?![A-Za-z0-9])", target_name),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    if purpose == "风电用钢":
        material_label = get_wind_power_material_label(user_message)
        for source, replacement in (
            ("管线钢", material_label),
            ("管线用钢", material_label),
            ("油气输送管", "风电塔筒"),
            ("油气管线", "风电塔筒"),
        ):
            sanitized = sanitized.replace(source, replacement)
    sanitized = re.sub(r"（\s*" + re.escape(target_name) + r"\s*）", "", sanitized)
    sanitized = re.sub(r"\(\s*" + re.escape(target_name) + r"\s*\)", "", sanitized)
    # 初步方案中的 ~~内容~~ 不是删除建议，统一按普通文本展示，避免 marked 渲染为删除线。
    sanitized = re.sub(r"~~(.+?)~~", r"\1", sanitized)
    return _sanitize_visible_text(sanitized.strip())


async def _stream_flash_design_preview_to_queue(
    user_message: str,
    docs: list[dict],
    queue: asyncio.Queue,
    purpose: str = "耐磨钢",
    emit_lifecycle: bool = True,
) -> None:
    """将 Qwen 的初步方案流式写入队列，不阻塞后续计算任务。"""
    try:
        if emit_lifecycle:
            await queue.put(_ndjson_event("design_preview_start", message="正在生成材料设计初步方案..."))
        await queue.put(_ndjson_event("design_preview_delta", content="## 材料设计初步方案\n\n"))
        # 初步方案关闭 DeepSeek thinking，只展示流程摘要，不再向前端流式发送
        # reasoning_content 的完整内容。
        _LLM_REASONING_CONTENT_CACHE.pop("_visible:初步方案", None)
        await queue.put(_ndjson_event(
            "design_preview_delta",
            content=_format_reasoning_content(
                "初步方案",
                "",
                [
                    (
                        f"根据用户目标整理{get_wind_power_material_label(user_message)}成分设计、"
                        "TMCP工艺路线、组织控制和预期性能。"
                        if purpose == "风电用钢"
                        else "根据用户目标整理管线钢成分设计、TMCP工艺路线、组织控制和预期性能。"
                    ),
                    "本阶段仅形成初步设计思路，不把历史实绩或仿真结果作为确定结论。",
                ],
            ),
        ))
        await queue.put(_ndjson_event("design_preview_delta", content="\n\n---\n\n## 初步方案正文\n\n"))
        messages = _build_flash_design_preview_messages(user_message, docs, purpose=purpose)
        preview_chunks = []
        emitted_length = 0
        has_streamed_content = False
        async for event in official_deepseek_async.astream(
            messages,
            extra_body={"thinking": {"type": "disabled"}},
        ):
            if event['type'] == 'reasoning':
                # 即使上游兼容接口意外返回 reasoning_content，也不缓存、不展示。
                continue
            elif event['type'] == 'content':
                preview_chunks.append(event['text'])
                preview_text = _sanitize_flash_design_preview_text("".join(preview_chunks), purpose=purpose, user_message=user_message)
                if len(preview_text) > emitted_length:
                    await queue.put(_ndjson_event("design_preview_delta", content=preview_text[emitted_length:]))
                    emitted_length = len(preview_text)
                    has_streamed_content = True
            elif event['type'] == 'done':
                print(
                    "[Flash初步方案] thinking=disabled, "
                    f"full_content_len={len(event.get('full_content', ''))}"
                )
        preview_text = _sanitize_flash_design_preview_text("".join(preview_chunks), purpose=purpose, user_message=user_message)
        if preview_text and not has_streamed_content:
            await queue.put(_ndjson_event("design_preview_delta", content=preview_text))
        if emit_lifecycle:
            await queue.put(_ndjson_event("design_preview_done"))
    except Exception as exc:
        print(f"[Flash初步方案] 生成失败: {type(exc).__name__}: {exc}")
        await queue.put(_ndjson_event("error", message=f"初步方案生成失败: {exc}"))
    finally:
        await queue.put(None)


def _retrieve_engineering_knowledge_docs(user_message: str, top_k: int = 5) -> list[dict]:
    from hybrid_retriever import hybrid_search

    query = (
        f"{user_message} 耐磨钢 工程机械用钢 成分 工艺 组织演变 晶粒 析出 相变 CCT PTT "
        "FET FDT CT 淬火 回火 热处理 力学性能 屈服强度 抗拉强度 断后伸长率 "
        "成分对组织的影响 组织对性能的影响"
    )
    return hybrid_search(query, k=top_k, db_name="gcjxyg_Know_db")


def _route_steel_knowledge_base_tool(user_message: str, stage_label: str) -> dict:
    """使用与普通聊天相同的 LangChain 工具路由，让模型自主决定是否检索知识库。

    返回 selected_tool、query 和 content。模型未选择工具、工具名无效或调用失败时，
    content 为空字符串；调用方继续使用自身的专业知识和结构化数据，不中断主流程。
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from rag_tools import KNOWLEDGE_BASE_TOOLS, KNOWLEDGE_BASE_TOOL_MAP

    normalized_message = str(user_message or "").strip()
    if not normalized_message:
        return {"selected_tool": None, "query": "", "content": ""}

    routing_messages = [
        SystemMessage(content=KNOWLEDGE_BASE_TOOL_ROUTING_PROMPT),
        SystemMessage(content=KNOWLEDGE_BASE_TOOL_SELECTION_SYSTEM_PROMPT),
        HumanMessage(content=normalized_message),
    ]

    def _parse_text_tool_call(text: str) -> tuple[str, str] | None:
        if not text:
            return None
        json_tool_call = _parse_chat_json_tool_call(
            text,
            normalized_message,
            KNOWLEDGE_BASE_TOOL_MAP,
        )
        if json_tool_call:
            return json_tool_call
        tool_name = next(
            (name for name in KNOWLEDGE_BASE_TOOL_MAP if name in text),
            None,
        )
        if not tool_name:
            return None
        match = re.search(
            r'parameter\s+name=["\']query["\'][^>]*>([\s\S]*?)'
            r'(?:</\s*parameter\s*>|<\s*[｜|]{0,2}\s*DSML|$)',
            text,
            flags=re.IGNORECASE,
        )
        query = match.group(1).strip() if match else normalized_message
        return tool_name, query or normalized_message

    try:
        router = deepseek_Llm.bind_tools(KNOWLEDGE_BASE_TOOLS)
        response = router.invoke(routing_messages)
        selected_tool_name = None
        selected_query = ""

        for tool_call in getattr(response, "tool_calls", None) or []:
            candidate_name = str(tool_call.get("name") or "")
            if candidate_name not in KNOWLEDGE_BASE_TOOL_MAP:
                continue
            tool_args = tool_call.get("args") or {}
            selected_tool_name = candidate_name
            selected_query = (
                tool_args.get("query")
                if isinstance(tool_args, dict)
                else ""
            ) or normalized_message
            break

        if not selected_tool_name:
            response_content = getattr(response, "content", "")
            if not isinstance(response_content, str):
                response_content = str(response_content or "")
            text_tool_call = _parse_text_tool_call(response_content)
            if text_tool_call:
                selected_tool_name, selected_query = text_tool_call

        if not selected_tool_name:
            print(f"[知识库Agent路由] {stage_label}: 模型判断无需检索")
            return {"selected_tool": None, "query": "", "content": ""}

        tool_item = KNOWLEDGE_BASE_TOOL_MAP[selected_tool_name]
        tool_content = str(tool_item.invoke({"query": selected_query}) or "").strip()
        print(
            f"[知识库Agent路由] {stage_label}: selected={selected_tool_name}, "
            f"query={_safe_text_preview(selected_query, 160)!r}, content_len={len(tool_content)}"
        )
        return {
            "selected_tool": selected_tool_name,
            "query": selected_query,
            "content": tool_content,
        }
    except Exception as exc:
        print(
            f"[知识库Agent路由] {stage_label}失败: "
            f"{type(exc).__name__}: {exc}"
        )
        return {"selected_tool": None, "query": "", "content": ""}


def _retrieve_pipeline_knowledge_docs(user_message: str, top_k: int = 5) -> list[dict]:
    """由知识库 Agent 自主选择用途知识库；返回格式兼容现有最终报告逻辑。"""
    del top_k  # 七类知识库工具与普通聊天保持一致，内部固定使用 Top-5。
    routed = _route_steel_knowledge_base_tool(
        user_message,
        stage_label="管线钢设计前置知识检索",
    )
    content = str(routed.get("content") or "").strip()
    if not content:
        return []
    sections = [
        section.strip()
        for section in content.split("\n\n---\n\n")
        if section.strip()
    ]
    selected_tool = routed.get("selected_tool") or "knowledge_base_agent"
    selected_query = routed.get("query") or user_message
    return [
        {
            "source": selected_tool,
            "content": section,
            "query": selected_query,
        }
        for section in sections
    ]


def _sanitize_visible_text(text: str) -> str:
    """清理面向用户的模型品牌名称，内部日志和调用标识保持不变。"""
    rendered = str(text or "")
    replacements = (
        (r"(?i)qwen", "分析模型"),
        (r"(?i)deepseek", "报告生成模型"),
    )
    for pattern, replacement in replacements:
        rendered = re.sub(pattern, replacement, rendered)
    return rendered


def _ndjson_event(event: str, **payload) -> str:
    return json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"


_LLM_REASONING_CONTENT_CACHE: dict[str, str] = {}
_LLM_JUDGEMENT_CONTENT_CACHE: dict[str, list[str]] = {}
_LLM_JUDGEMENT_REASONING_CACHE: dict[str, list[str]] = {}
_LLM_JUDGEMENT_INPUT_CACHE: dict[str, list[dict]] = {}
_LLM_JUDGEMENT_VISIBLE_CACHE: dict[str, list[dict]] = {}


def _extract_qwen_agent_response(parsed_response: dict | None) -> tuple[dict | None, dict]:
    """拆分 Qwen 可见正文包装；最终业务结果仍只返回原结构 matched_result。"""
    if not isinstance(parsed_response, dict):
        return None, {}
    wrapped_result = parsed_response.get("matched_result")
    judgement = parsed_response.get("judgement")
    if isinstance(wrapped_result, dict):
        return wrapped_result, judgement if isinstance(judgement, dict) else {}
    # 兼容模型偶尔仍按旧协议直接返回 matched_result。
    return parsed_response, {}


def _extract_reasoning_content(ai_message) -> str:
    """从模型返回的 AIMessage 中提取可见 reasoning_content。"""
    additional_kwargs = getattr(ai_message, "additional_kwargs", {}) or {}
    reasoning = additional_kwargs.get("reasoning_content", "")
    if isinstance(reasoning, (list, tuple)):
        reasoning = "\n".join(str(item) for item in reasoning if item)
    return str(reasoning or "").strip()


def _remember_reasoning_content(key: str, ai_message) -> str:
    """按阶段缓存模型 reasoning_content，供前端“模型处理过程”展示。"""
    additional_kwargs = getattr(ai_message, "additional_kwargs", {}) or {}
    reasoning = _extract_reasoning_content(ai_message)
    print(
        f"[reasoning_content] key={key}, "
        f"additional_kwargs_keys={list(additional_kwargs.keys())}, "
        f"reasoning_len={len(reasoning)}"
    )
    if reasoning:
        _LLM_REASONING_CONTENT_CACHE[key] = reasoning
    return reasoning


def _pop_reasoning_content(key: str, fallback: str = "") -> str:
    """读取并移除指定阶段的 reasoning_content；为空时返回 fallback。"""
    reasoning = _LLM_REASONING_CONTENT_CACHE.pop(key, "")
    return reasoning or fallback


def _format_reasoning_content(title: str, reasoning: str, fallback_lines: list[str] | None = None) -> str:
    """把模型返回的 reasoning_content 包装成前端可展示 Markdown。"""
    text = str(reasoning or "").strip()
    if not text and fallback_lines:
        safe_lines = [str(line).strip() for line in fallback_lines if str(line).strip()]
        text = "\n".join(f"- {line}" for line in safe_lines)
    if not text:
        text = "- 本阶段模型未返回可展示的分析摘要。"
    return _sanitize_visible_text(f"\n\n### 思维链：{title}\n\n{text}\n\n")


def _pop_judgement_contents(key: str) -> list[str]:
    """读取并移除某个工艺智能体每一轮返回的正文。"""
    return _LLM_JUDGEMENT_CONTENT_CACHE.pop(key, [])


def _pop_judgement_reasonings(key: str) -> list[str]:
    """读取并移除某个工艺智能体每一轮返回的 reasoning_content。"""
    _LLM_REASONING_CONTENT_CACHE.pop(key, None)
    return _LLM_JUDGEMENT_REASONING_CACHE.pop(key, [])


def _pop_judgement_inputs(key: str) -> list[dict]:
    """读取并移除某个工艺智能体每一轮判断前的 matched_result。"""
    return _LLM_JUDGEMENT_INPUT_CACHE.pop(key, [])


def _pop_visible_judgements(key: str) -> list[dict]:
    """读取并移除每轮 Qwen 正文中显式返回的图片结论和专业理论依据。"""
    return _LLM_JUDGEMENT_VISIBLE_CACHE.pop(key, [])


def _summarize_visible_reasoning(reasoning: str, fallback_lines: list[str] | None = None) -> str:
    """把模型 reasoning_content 压缩成简短可见摘要，避免前端展示冗长推理过程。"""
    text = " ".join(str(reasoning or "").replace("\x00", "").split())
    if text:
        sentences = [item.strip() for item in re.split(r"(?<=[。！？；.!?;])\s*", text) if item.strip()]
        selected = []
        for sentence in sentences:
            if sentence in selected:
                continue
            selected.append(sentence[:220])
            if len(selected) >= 3:
                break
        if selected:
            return "\n".join(f"- {sentence}" for sentence in selected)

    safe_lines = [str(line).strip() for line in (fallback_lines or []) if str(line).strip()]
    if not safe_lines:
        safe_lines = ["本阶段模型未返回可用于摘要的 reasoning_content。"]
    return "\n".join(f"- {line}" for line in safe_lines)


def _format_qwen_judgement_result(
    title: str,
    judgement_reasonings: list[str],
    visible_judgements: list[dict],
    judgement_contents: list[str],
    judgement_inputs: list[dict],
    stage: str,
    fallback_lines: list[str] | None = None,
    round_numbers: list[int] | None = None,
    include_title: bool = True,
) -> str:
    """仅展示每轮图片判断、合法工艺调整和理论依据，不暴露完整 matched_result。"""
    stage_rules = {
        "reheat": {
            "fields": {"SOAK_TEMP", "SOAK_TIME"},
            "conclusion_keywords": ("晶粒", "固溶", "温度场", "粗化", "分布", "符合"),
            "theory_keywords": ("固溶", "均热", "晶界", "粗化", "文献", "析出"),
            "pass_text": "加热温度场、晶粒长大和晶粒尺寸分布满足当前加热工艺判断要求。",
            "fail_text": "加热温度场或晶粒组织结果尚未满足当前加热工艺判断要求。",
            "theory_text": "依据微合金元素固溶、奥氏体均匀化及抑制异常晶粒长大的协同方向调整均热温度和时长。",
        },
        "roll": {
            "fields": {"FET", "FDT"},
            "conclusion_keywords": ("各道次", "晶粒", "粗晶", "突变", "分布", "符合"),
            "theory_keywords": ("再结晶", "未再结晶", "压下", "细化", "软化", "析出"),
            "pass_text": "各道次晶粒尺寸及演化趋势满足当前控制轧制工艺判断要求。",
            "fail_text": "各道次晶粒尺寸、演化连续性或细化效果尚未满足当前控制轧制工艺判断要求。",
            "theory_text": "依据再结晶区与未再结晶区变形、累积压下和晶粒细化的协同方向调整轧制参数。",
        },
        "cooling": {
            "fields": {"TIME_ENTR", "TEMP_ENTR", "SELF_TEMP"},
            "conclusion_keywords": ("相组成", "铁素体", "贝氏体", "晶粒", "比例", "符合"),
            "theory_keywords": ("相变", "CCT", "冷速", "返红", "过冷", "形核"),
            "pass_text": "冷却后的铁素体晶粒尺寸和相组成比例满足当前控制冷却工艺判断要求。",
            "fail_text": "冷却后的铁素体晶粒尺寸或相组成比例尚未满足当前控制冷却工艺判断要求。",
            "theory_text": "依据最后有效轧制道次时刻、目标组织、轧后等待时间、终轧温度、连续冷却相变和过冷度协同调整开冷时刻、入水及返红温度。",
        },
    }
    rule = stage_rules[stage]
    sections = [f"\n\n### 思维链：{title}\n"] if include_title else []
    if not judgement_contents:
        fallback_summary = _summarize_visible_reasoning("", fallback_lines)
        sections.append(f"\n#### 思考摘要\n\n{fallback_summary}\n\n> 本阶段未返回判断正文。\n")
        return _sanitize_visible_text("".join(sections) + "\n")

    for index, raw_content in enumerate(judgement_contents, start=1):
        raw_text = str(raw_content or "").strip()
        candidate = _parse_json_object(raw_text) or {}
        before = judgement_inputs[index - 1] if index <= len(judgement_inputs) else {}
        visible_judgement = (
            visible_judgements[index - 1]
            if index <= len(visible_judgements) and isinstance(visible_judgements[index - 1], dict)
            else {}
        )
        before_row = _matched_result_body_to_row(before) if isinstance(before, dict) else {}
        candidate_row = _matched_result_body_to_row(candidate) if isinstance(candidate, dict) else {}

        allowed_fields = set(rule["fields"])
        if stage == "roll":
            allowed_fields.update(PIPELINE_REFINABLE_PASS_COUNT_FIELDS)
            allowed_fields.update(PIPELINE_REFINABLE_TURN_FIELDS)
            allowed_fields.update(
                field_name for field_name in set(before_row) | set(candidate_row)
                if re.fullmatch(
                    r"N(?:[1-9]|[12]\d|30)_(?:DH_CAL|DT_CAL|DW_CAL|FORCE|GAP|SPD)",
                    str(field_name).upper(),
                )
            )
        changes = []
        for field_name in sorted(allowed_fields):
            before_value = before_row.get(field_name)
            after_value = candidate_row.get(field_name)
            if str(before_value).strip() != str(after_value).strip():
                changes.append(f"`{field_name}`：{before_value} → {after_value}")
        adjustment_text = "；".join(changes) if changes else "本轮未调整允许修改的工艺参数。"

        conclusion_detail = str(
            visible_judgement.get("imageConclusion")
            or visible_judgement.get("图片判断结论")
            or ""
        ).strip()
        theory_detail = str(
            visible_judgement.get("adjustmentBasis")
            or visible_judgement.get("调整理论依据")
            or ""
        ).strip()
        is_state = candidate.get("isState") if isinstance(candidate, dict) else None
        conclusion = rule["pass_text"] if is_state is True else rule["fail_text"]
        if conclusion_detail:
            conclusion = f"{conclusion} {conclusion_detail}"
        theory = theory_detail or rule["theory_text"]

        displayed_round = (
            round_numbers[index - 1]
            if round_numbers and index <= len(round_numbers)
            else index
        )
        heading = (
            f"第 {displayed_round} 轮判断正文"
            if round_numbers or len(judgement_contents) > 1
            else "判断正文"
        )
        sections.append(
            f"\n#### {heading}\n\n"
            f"- **温度场/组织图片判断结论：** {conclusion}\n"
            f"- **工艺调整及具体值：** {adjustment_text}\n"
            f"- **调整的理论依据方向：** {theory}\n"
        )
    return _sanitize_visible_text("".join(sections) + "\n")


_PIPELINE_AGENT_ALL_MODULES_MARKDOWN = """系统可调用工具包括：

1. 加热过程温度场工具
2. 加热过程奥氏体晶粒长大工具
3. 加热过程奥氏体晶粒尺寸分布工具
4. 粗轧过程温度场工具
5. 精轧过程温度场工具
6. 软化率工具
7. 变形抗力计算工具
8. 轧制力计算工具
9. 扭矩计算工具
10. 摩擦系数计算工具
11. 氧化铁皮厚度计算工具
12. 析出热力学计算工具
13. RPTT工具
14. 析出动力学计算工具
15. 道次间隔期间奥氏体晶粒长大工具
16. 粗轧出口奥氏体晶粒组织工具
17. 精轧出口奥氏体晶粒组织工具
18. 冷却过程温度场工具
19. 动态CCT曲线工具
20. 铁素体相变工具
21. 珠光体相变工具
22. 针状铁素体相变工具
23. 粒状贝氏体相变工具
24. 板条贝氏体相变工具
25. 铁素体晶粒尺寸工具
26. 固溶强化工具
27. 细晶强化工具
28. 析出强化工具
29. 元素固溶工具"""


_PIPELINE_AGENT_MODULE_DECISIONS = {
    "reheat": {
        "heading": "加热过程智能体工具决策",
        "description": "智能体根据钢种成分、规格参数、生产工艺制度及目标任务，融合成分–工艺–组织–性能演变关系，对系统内全部计算工具进行智能识别与协同匹配，自动选择适用于当前工艺阶段的计算模型。",
        "task": "根据当前任务属于钢坯加热过程模拟与加热阶段组织演变模拟，智能体从上述全部计算工具中筛选出以下工具执行计算：",
        "selected": [
            "加热过程温度场工具",
            "加热过程奥氏体晶粒长大工具",
            "加热过程奥氏体晶粒尺寸分布工具",
            "元素固溶工具",
        ],
        "closing": "智能体完成工具选择，开始执行加热过程多物理场耦合计算。",
    },
    "roll": {
        "heading": "轧制过程智能体工具决策",
        "description": "智能体根据钢种成分、规格参数、轧制工艺制度及目标任务，融合成分–工艺–组织–性能演变关系，对系统内全部计算工具进行智能识别与协同匹配，自动选择适用于当前轧制阶段的计算模型。",
        "task": "根据当前任务属于钢板轧制过程模拟与轧制阶段组织演变预测，智能体从上述全部计算工具中筛选出以下工具执行计算：",
        "selected": [
            "粗轧过程温度场工具",
            "精轧过程温度场工具",
            "RPTT工具",
            "析出动力学计算工具",
            "软化率工具",
            "道次间隔期间奥氏体晶粒长大工具",
            "粗轧出口奥氏体晶粒组织工具",
            "精轧出口奥氏体晶粒组织工具",
        ],
        "closing": "智能体完成工具选择，开始执行轧制过程多物理场耦合计算。",
    },
    "cooling": {
        "heading": "控制冷却过程智能体工具决策",
        "description": "智能体根据钢种成分、规格参数、冷却工艺制度及目标任务，融合成分–工艺–组织–性能演变关系，对系统内全部计算工具进行智能识别与协同匹配，自动选择适用于当前控制冷却阶段的计算模型。",
        "task": "根据当前任务属于钢板控制冷却过程模拟与冷却阶段组织转变预测，智能体从上述全部计算工具中筛选出以下工具执行计算：",
        "selected": [
            "冷却过程温度场工具",
            "动态CCT曲线工具",
            "铁素体相变工具",
            "铁素体晶粒尺寸工具",
            "珠光体相变工具",
            "针状铁素体相变工具",
            "粒状贝氏体相变工具",
            "板条贝氏体相变工具",
            "固溶强化工具",
            "细晶强化工具",
            "析出强化工具",
        ],
        "closing": "智能体完成工具选择，开始执行控制冷却过程多物理场耦合计算。",
    },
}


def _format_pipeline_agent_module_decision(
    stage: str,
    attempt: int,
    title: str,
    include_title: bool,
) -> str:
    """生成每轮计算前固定展示的智能体工具选择说明。"""
    decision = _PIPELINE_AGENT_MODULE_DECISIONS[stage]
    title_markdown = f"\n\n### 思维链：{title}\n" if include_title else ""
    selected_markdown = "\n\n".join(f"✓ {name}" for name in decision["selected"])
    return _sanitize_visible_text(
        f"{title_markdown}\n#### 第 {attempt} 轮调用前工具决策\n\n"
        f"【{decision['heading']}】\n\n"
        f"{decision['description']}\n\n"
        f"{_PIPELINE_AGENT_ALL_MODULES_MARKDOWN}\n\n"
        f"【智能体工具选择结果】\n\n"
        f"{decision['task']}\n\n"
        f"{selected_markdown}\n\n"
        f"{decision['closing']}\n\n"
    )


async def _stream_pipeline_agent_execution(
    agent_func,
    matched_result: dict,
    context: str,
    session_id: str,
    stage: str,
    title: str,
    fallback_lines: list[str],
    agent_kwargs: dict | None = None,
):
    """在线程中运行同步工艺智能体，并把每轮判断通过异步队列即时送回流式接口。"""
    is_wind_power_design = _is_wind_power_context(context)
    progress_queue: asyncio.Queue = asyncio.Queue()
    event_loop = asyncio.get_running_loop()

    def progress_callback(payload: dict) -> None:
        # 该函数运行在 asyncio.to_thread 的工作线程中，必须通过主事件循环安全入队。
        event_loop.call_soon_threadsafe(progress_queue.put_nowait, payload)

    async def run_agent():
        try:
            return await asyncio.to_thread(
                agent_func,
                matched_result,
                context,
                session_id,
                progress_callback=progress_callback,
                **(agent_kwargs or {}),
            )
        finally:
            # None 是结束哨兵；保证智能体异常时前端等待循环也能正常退出。
            await progress_queue.put(None)

    agent_task = asyncio.create_task(run_agent())
    has_judgement_progress = False
    agent_failed = False
    include_title = True
    judgement_context_sections: list[str] = []
    while True:
        payload = await progress_queue.get()
        if payload is None:
            break
        attempt = int(payload.get("attempt") or 1)
        if payload.get("event_type") == "module_decision":
            markdown = _format_pipeline_agent_module_decision(
                stage,
                attempt,
                title,
                include_title,
            )
        elif payload.get("event_type") == "agent_error":
            agent_failed = True
            has_judgement_progress = True
            markdown = (
                ("\n\n" if include_title else "")
                + f"> **本阶段判断未完成：** {payload.get('message') or '模型结果未通过校验。'}\n\n"
            )
        elif payload.get("event_type") == "fallback_applied":
            # 确定性工程兜底已经形成可用于仿真和报告的最终结果，覆盖此前
            # 模型格式失败状态，避免前端误报“仍保留失败前工艺”。
            agent_failed = False
            has_judgement_progress = True
            markdown = (
                ("\n\n" if include_title else "")
                + f"> **确定性工艺兜底：** {payload.get('message') or '已应用冷却工艺兜底。'}\n\n"
            )
        else:
            has_judgement_progress = True
            markdown = _format_qwen_judgement_result(
                title,
                [str(payload.get("reasoning") or "")],
                [payload.get("judgement") or {}],
                [json.dumps(payload.get("after") or {}, ensure_ascii=False)],
                [payload.get("before") or {}],
                stage,
                fallback_lines,
                round_numbers=[attempt],
                include_title=include_title,
            )
        if is_wind_power_design:
            # 共享智能体的模型判断偶尔会沿用“管线钢”措辞。前端展示和后续
            # 报告上下文统一转换为风电塔筒钢板语境，避免用途名称串线。
            markdown = _wind_power_agent_prompt(markdown)
        if payload.get("event_type") != "module_decision":
            # 保存后端已采纳的精简判断正文，供后续智能体和最终报告参考。
            # 这里不保存完整 matched_result 或隐藏推理，只保存图片结论、调整值和理论依据。
            judgement_context_sections.append(markdown.strip())
        include_title = False
        # 同步接口会一次性返回正文，这里把精简 Markdown 切成小块逐段发送，
        # 让前端保持与其他分析阶段一致的流式追加和 Markdown 刷新效果。
        stream_chunk_size = 48
        for offset in range(0, len(markdown), stream_chunk_size):
            yield {
                "type": "progress",
                "markdown": markdown[offset:offset + stream_chunk_size],
            }
            await asyncio.sleep(0.01)

    result = await agent_task
    cache_key = f"{session_id}:{stage}"
    # 每轮正文已经实时展示，清理调用层缓存，避免阶段结束后重复输出或长期占用内存。
    _pop_judgement_reasonings(cache_key)
    _pop_visible_judgements(cache_key)
    _pop_judgement_contents(cache_key)
    _pop_judgement_inputs(cache_key)
    if not has_judgement_progress:
        fallback_markdown = _format_qwen_judgement_result(
            title,
            [],
            [],
            [],
            [],
            stage,
            fallback_lines,
            include_title=include_title,
        )
        for offset in range(0, len(fallback_markdown), 48):
            yield {
                "type": "progress",
                "markdown": fallback_markdown[offset:offset + 48],
            }
            await asyncio.sleep(0.01)
    yield {
        "type": "result",
        "result": result,
        "failed": agent_failed,
        "judgement_context": "\n\n".join(
            section for section in judgement_context_sections if section
        ),
    }




def _apply_pipeline_risk_adjustment_without_validation(
    original: dict,
    candidate: dict,
) -> tuple[dict, set[str]]:
    """风险复核结果不做字段和值校验，直接采用；这里只识别图片过滤所需的影响阶段。"""
    if not isinstance(candidate, dict):
        print("[管线钢报告风险评估] adjustedMatchedResult 不是JSON对象，无法替换当前结果")
        return copy.deepcopy(original), set()

    original_row = {
        str(_get_arrbody_key(item) or "").upper(): _get_arrbody_value(item)
        for item in original.get("arrBody", [])
        if _get_arrbody_key(item)
    }
    candidate_row = {
        str(_get_arrbody_key(item) or "").upper(): _get_arrbody_value(item)
        for item in candidate.get("arrBody", [])
        if _get_arrbody_key(item)
    }
    changed_fields = {
        field_name
        for field_name in set(original_row) | set(candidate_row)
        if original_row.get(field_name) != candidate_row.get(field_name)
    }

    adjusted_stages: set[str] = set()
    unknown_change = False
    for field_name in changed_fields:
        if (
            field_name.startswith(("PRE_HEAT", "HEAT_TEMP", "HEAT_TIME", "SOAK_", "FURNACE_"))
            or field_name == "SLAB_FURNACE_ENT_TEMP"
        ):
            adjusted_stages.add("reheat")
        elif re.fullmatch(r"N\d+_.*", field_name) or field_name in {
            "FET", "FDT", "R_PASS_ACT", "F_PASS_ACT",
        }:
            adjusted_stages.add("roll")
        elif (
            field_name.startswith("COOL_")
            or field_name in {
                "TEMP_ENTR", "TIME_ENTR", "SELF_TEMP", "SPEED",
            }
        ):
            adjusted_stages.add("cooling")
        else:
            unknown_change = True

    top_level_changed = any(
        original.get(key) != candidate.get(key)
        for key in set(original) | set(candidate)
        if key != "arrBody"
    )
    if unknown_change or top_level_changed:
        adjusted_stages.update({"reheat", "roll", "cooling"})
    return copy.deepcopy(candidate), adjusted_stages


def _collect_pipeline_risk_adjustments(original: dict, adjusted: dict) -> list[tuple[str, object, object]]:
    """统计风险模型直接返回的全部字段变化，仅用于前端进度和运行日志。"""
    changes = []
    for key in set(original) | set(adjusted):
        if key != "arrBody" and original.get(key) != adjusted.get(key):
            changes.append((str(key), original.get(key), adjusted.get(key)))
    original_body = original.get("arrBody") if isinstance(original, dict) else None
    adjusted_body = adjusted.get("arrBody") if isinstance(adjusted, dict) else None
    if not isinstance(original_body, list) or not isinstance(adjusted_body, list):
        return changes
    original_row = {
        str(_get_arrbody_key(item)): _get_arrbody_value(item)
        for item in original_body
        if _get_arrbody_key(item)
    }
    adjusted_row = {
        str(_get_arrbody_key(item)): _get_arrbody_value(item)
        for item in adjusted_body
        if _get_arrbody_key(item)
    }
    for field_name in set(original_row) | set(adjusted_row):
        if original_row.get(field_name) != adjusted_row.get(field_name):
            changes.append((field_name, original_row.get(field_name), adjusted_row.get(field_name)))
    return changes


def _invoke_pipeline_report_risk_assessment(
    source_report: str,
    matched_result: dict,
    fact_table: dict,
    user_message: str,
    report_model: str = "deepseek",
) -> dict:
    """使用与报告生成模型不同的模型复核报告；当前 DeepSeek 报告固定由 Qwen 评估。"""
    user_prompt = build_pipeline_report_risk_user_prompt(
        user_message, source_report, matched_result, fact_table
    )
    try:
        evaluator_name = "Qwen" if str(report_model).lower() == "deepseek" else "DeepSeek"
        print(
            f"[管线钢报告风险评估] 开始调用 {evaluator_name}，"
            f"源报告模型={report_model}, 报告长度={len(source_report)}"
        )
        messages = [
            SystemMessage(content=PIPELINE_REPORT_RISK_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        if str(report_model).lower() == "deepseek":
            raw = official_qwen_sync.invoke(
                messages,
                timeout=None,
                extra_body={"enable_thinking": False},
            )
            evaluator_name = "Qwen"
        else:
            raw = official_deepseek_sync.invoke(messages, timeout=None)
            evaluator_name = "DeepSeek"
        parsed = _parse_json_object(str(raw.content or ""))
        if not isinstance(parsed, dict):
            raise ValueError("风险评估模型未返回合法 JSON")
        parsed["_evaluator"] = evaluator_name
        print(
            f"[管线钢报告风险评估] {evaluator_name} 返回完成，"
            f"hasRisk={parsed.get('hasRisk')}, "
            f"声明调整阶段={parsed.get('adjustedStages')}"
        )
        return parsed
    except Exception as exc:
        print(f"[管线钢报告风险评估] 调用失败: {exc}")
        return {
            "hasRisk": False,
            "riskReport": f"## 工艺风险评估报告\n\n> 风险评估模型调用失败，已保留源报告和原工艺：{exc}",
            "adjustedMatchedResult": copy.deepcopy(matched_result),
            "adjustedStages": [],
            "finalReport": source_report,
            "_evaluator": "Qwen" if str(report_model).lower() == "deepseek" else "DeepSeek",
        }


def _render_pipeline_report_with_stage_images(
    report_text: str,
    image_blocks_by_stage: dict[str, list[str]],
    adjusted_stages: set[str],
) -> str:
    """把未调整阶段的原仿真图片插入最终报告；调整阶段不再引用旧图片。"""
    marker_stages = {
        "**加热工艺制度的解析**": ("reheat", "加热阶段"),
        "**控制轧制工艺制度的解析**": ("roll", "控制轧制阶段"),
        "**控制冷却工艺制度的解析**": ("cooling", "控制冷却阶段"),
    }
    rendered = str(report_text or "")
    missing_sections = []
    for marker, (stage_key, stage_name) in marker_stages.items():
        if stage_key in adjusted_stages:
            continue
        blocks = image_blocks_by_stage.get(stage_name) or []
        if not blocks:
            continue
        image_section = "\n\n" + "\n\n".join(blocks) + "\n\n"
        if marker in rendered:
            rendered = rendered.replace(marker, marker + image_section, 1)
        else:
            missing_sections.append(f"## {stage_name}仿真图像\n\n" + "\n\n".join(blocks))
    if missing_sections:
        rendered += "\n\n" + "\n\n".join(missing_sections)
    return rendered


def _strip_pipeline_risk_sections_from_report(report_text: str) -> str:
    """最终回答不展示风险评估过程；仅移除模型误写的独立风险章节。"""
    rendered = str(report_text or "")
    risk_heading_patterns = (
        r"\*\*(?:工艺风险与调整策略|(?:工艺)?风险评估(?:与调整策略|报告)?)\*\*",
        r"(?m)^#{1,6}\s*(?:工艺风险与调整策略|(?:工艺)?风险评估(?:与调整策略|报告)?)\s*$",
    )
    report_section_markers = (
        "**加热工艺制度的解析**",
        "**控制轧制工艺制度的解析**",
        "**控制冷却工艺制度的解析**",
    )
    for pattern in risk_heading_patterns:
        while True:
            match = re.search(pattern, rendered)
            if not match:
                break
            following_positions = [
                rendered.find(marker, match.end())
                for marker in report_section_markers
                if rendered.find(marker, match.end()) >= 0
            ]
            next_heading = re.search(r"(?m)^#{1,6}\s+\S", rendered[match.end():])
            if next_heading:
                following_positions.append(match.end() + next_heading.start())
            section_end = min(following_positions) if following_positions else len(rendered)
            rendered = rendered[:match.start()].rstrip() + "\n\n" + rendered[section_end:].lstrip()
    return rendered.strip()


def _visible_reasoning_content(title: str, lines: list[str]) -> str:
    """构造可展示的 reasoning_content；优先使用缓存中的原始思维链，无则用摘要。"""
    cached = _LLM_REASONING_CONTENT_CACHE.get(f"_visible:{title}", "").strip()
    if cached:
        return _sanitize_visible_text(f"\n\n### 思维链：{title}\n\n{cached}\n\n")
    safe_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not safe_lines:
        safe_lines = ["本阶段使用可见输入、规格边界和已生成结果做一致性检查。"]
    body = "\n".join(f"- {line}" for line in safe_lines)
    return _sanitize_visible_text(f"\n\n### 思维链：{title}\n\n{body}\n\n")


async def _relay_streaming_response(response: Response):
    async for chunk in response.body_iterator:
        yield chunk


async def _stream_with_heartbeat(source, interval_seconds: float = 15.0):
    """业务流静默时发送心跳，避免长时间仿真期间流式连接被当作空闲连接关闭。"""
    queue: asyncio.Queue = asyncio.Queue()

    async def produce():
        try:
            async for chunk in source:
                await queue.put(("chunk", chunk))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(("error", exc))
        finally:
            await queue.put(("done", None))

    producer = asyncio.create_task(produce())
    try:
        while True:
            try:
                item_type, payload = await asyncio.wait_for(
                    queue.get(),
                    timeout=interval_seconds,
                )
            except asyncio.TimeoutError:
                yield _ndjson_event("heartbeat")
                continue

            if item_type == "chunk":
                yield payload
            elif item_type == "error":
                raise payload
            else:
                break
    finally:
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass


def _wrap_event_stream(response: Response, prelude_events: list[dict]) -> StreamingResponse:
    async def generator():
        for event in prelude_events:
            yield json.dumps(event, ensure_ascii=False) + "\n"

        yield _ndjson_event("answer_start")

        pending = ""
        try:
            async for chunk in response.body_iterator:
                text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                pending += text
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        yield _ndjson_event("answer_delta", content=line)
                        continue
                    if data.get("event"):
                        yield json.dumps(data, ensure_ascii=False) + "\n"
                    elif data.get("reasoning_start") or "reasoning" in data:
                        # 普通对话的 DeepSeek 官方流仍使用 reasoning_start/reasoning
                        # 旧事件格式。这里必须原样透传给前端的“模型处理过程”区域；
                        # 如果落入下面的兜底分支，事件 JSON 会被当作回答正文，进而
                        # 在页面和导出的 PDF 中显示为连续的 {"reasoning": ...} 乱码。
                        yield json.dumps(data, ensure_ascii=False) + "\n"
                    elif data.get("content"):
                        yield _ndjson_event("answer_delta", content=data["content"])
                    elif data.get("error"):
                        yield _ndjson_event("error", message=data["error"])
                    else:
                        yield _ndjson_event("answer_delta", content=json.dumps(data, ensure_ascii=False))

            if pending.strip():
                yield _ndjson_event("answer_delta", content=pending.strip())
        except Exception as exc:
            print(f"[流式包装] 子流异常: {exc}")
            yield _ndjson_event("error", message=f"生成过程异常: {exc}")
        yield _ndjson_event("answer_done")

    return StreamingResponse(
        generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _wrap_plain_answer(content: str, prelude_events: list[dict]) -> StreamingResponse:
    async def generator():
        for event in prelude_events:
            yield json.dumps(event, ensure_ascii=False) + "\n"
        yield _ndjson_event("answer_start")
        yield _ndjson_event("answer_delta", content=content)
        yield _ndjson_event("answer_done")

    return StreamingResponse(
        generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/classify")
async def classify_endpoint(request: Request):
    """
    POST /classify — 智能路由接口

    - 意图 = DESIGN → 只返回 {"intent": "DESIGN"}
    - 意图 = CHAT   → 启动流式对话（大模型回复，不输出意图JSON）

    请求格式:
        {"message": "用户消息", "session_id": "xxx"}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "无效的 JSON 格式"}, status_code=400)

    original_user_message = body.get("message", "").strip()
    if not original_user_message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    session_id = body.get("session_id", "default").strip()
    # 前端仅回传当前会话正在展示的设计UUID；该值不进入用户消息或报告正文，
    # 只用于“以上方案、当前设计”等续改表达的确定性版本定位。
    active_design_id = str(body.get("active_design_id") or "").strip() or None
    # 多候选弹窗确认后回传的明确版本拥有最高优先级。它只参与版本定位，
    # 不拼入用户消息、上下文或最终报告。
    reference_design_id = str(body.get("reference_design_id") or "").strip() or None
    reference_resume_token = str(body.get("reference_resume_token") or "").strip() or None
    attachment_ids = body.get("attachment_ids") or []
    if not isinstance(attachment_ids, list):
        return JSONResponse({"error": "attachment_ids 必须是数组"}, status_code=400)
    if reference_resume_token:
        user_message = _consume_pending_design_reference_request(
            reference_resume_token,
            session_id,
            original_user_message,
        )
        if not user_message:
            return JSONResponse(
                {"error": "设计版本确认已过期或与当前会话不匹配，请重新发送原问题。"},
                status_code=409,
            )
    else:
        try:
            # effective user_message 只存在于本轮后台任务内，附件不会写入任何会话历史。
            user_message = await attachment_manager.build_prompt_and_consume(
                session_id,
                attachment_ids,
                original_user_message,
            )
        except AttachmentServiceError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)

    async def classify_stream():
        yield _ndjson_event("design_preview_start", message="模型处理过程")
        yield _ndjson_event(
            "design_preview_delta",
            content=(
                "## 模型处理过程\n\n"
                "> 这里展示模型返回的 reasoning_content；未返回时使用阶段说明兜底。\n\n"
            ),
        )
        # Step 1: 意图分类。这里必须在外层流里执行，才能先把检索状态推给前端。
        deterministic_intent = _deterministic_intent_override(user_message)
        if deterministic_intent:
            intent_result = {"intent": deterministic_intent}
            print(f"[意图分类] 明确语义规则命中: {user_message[:40]}... → {deterministic_intent}")
            yield _ndjson_event(
                "design_preview_delta",
                content=f"- 明确语义路由完成：`{deterministic_intent}`。\n",
            )
        else:
            try:
                yield _ndjson_event("design_preview_delta", content="- 正在调用 LLM 进行 DESIGN/CHAT 意图识别...\n")
                # Step 3: 用途分类
                intent_result = await asyncio.to_thread(
                    classify_with_rag,
                    system_prompt=INTENT_SYSTEM_PROMPT,
                    user_message=user_message,
                    session_id=f"intent_{session_id}",
                    json_schema=INTENT_JSON_SCHEMA,
                    db_name="Nb_KnowBase_db",
                )
                print(f"[意图分类] {user_message[:40]}... → {intent_result}")
                yield _ndjson_event(
                    "design_preview_delta",
                    content=f"- 意图识别完成：`{intent_result.get('intent', 'CHAT')}`。\n",
                )
            except Exception as e:
                print(f"[意图分类] 失败: {e}")
                yield _ndjson_event("error", message=f"意图分类检索失败: {e}")
                intent_result = {"intent": "CHAT"}
                yield _ndjson_event("design_preview_delta", content="- 意图识别失败，已降级为普通对话。\n")

        # Step 2: 根据意图路由判断式对话还是设计
        intent = intent_result.get("intent", "CHAT")

        if intent == "CHAT":
            yield _ndjson_event("design_preview_delta", content="- 正在调用 LLM 生成普通对话回答，最终内容将在报告/回答区域流式输出。\n")
            yield _ndjson_event("design_preview_done")
            async for chunk in _relay_streaming_response(
                _wrap_event_stream(
                    _build_chat_agent_response(
                        session_id,
                        user_message,
                        CHAT_SYSTEM_PROMPT,
                        persisted_user_message=original_user_message,
                    ),
                    [],
                )
            ):
                yield chunk
            return

        # ==========================================================
        # DESIGN 路径：二次分类（钢材用途）→ 路由
        # ==========================================================
        purpose_result = None
        deterministic_purpose = _deterministic_purpose_override(user_message)
        if deterministic_purpose:
            purpose_result = {"purpose": deterministic_purpose}
            print(f"[用途分类] 明确用途规则命中: {user_message[:40]}... → {deterministic_purpose}")
            yield _ndjson_event(
                "design_preview_delta",
                content=f"- 明确用途路由完成：`{deterministic_purpose}`。\n",
            )
        else:
            try:
                yield _ndjson_event("design_preview_delta", content="- 正在调用 LLM 判断材料用途设计分支...\n")
                # Step 3: 用途分类。
                # 一级 DESIGN/CHAT 意图识别保持原逻辑不变；这里只给二级材料用途识别补充
                # 当前会话上下文，使“以上方案、上一轮设计”等续改请求能够继承原材料用途。
                purpose_session_context = (
                    _build_cross_route_context(session_id)
                    or _get_recent_session_context(session_id)
                    or "（无最近会话上下文）"
                )
                purpose_user_message = (
                    f"【当前用户提示词】\n{user_message}\n\n"
                    f"【最近会话上下文】\n{purpose_session_context}"
                )
                purpose_result = await asyncio.to_thread(
                    classify_with_rag,
                    system_prompt=PURPOSE_SYSTEM_PROMPT,
                    user_message=purpose_user_message,
                    session_id=f"purpose_{session_id}",
                    json_schema=PURPOSE_JSON_SCHEMA,
                    db_name="Nb_KnowBase_db",
                )
                print(f"[用途分类] {user_message[:40]}... → {purpose_result}")
                yield _ndjson_event(
                    "design_preview_delta",
                    content=f"- 用途识别完成：`{purpose_result.get('purpose', '其他聊天')}`。\n",
                )
            except Exception as e:
                print(f"[用途分类] 失败: {e}")
                purpose_result = {"purpose": "其他聊天"}
                yield _ndjson_event("design_preview_delta", content="- 用途识别失败，已降级为普通对话。\n")

        purpose = purpose_result.get("purpose", "其他聊天")
        if purpose == "其他聊天":
            # 流式对话；当前仅保留管线钢设计分支，其他用途不再进入设计链路。
            LIMITED_PROMPT = LIMITED_CHAT_SYSTEM_PROMPT
            yield _ndjson_event("design_preview_delta", content="- 正在调用 LLM 生成普通对话回答，最终内容将在回答区域流式输出。\n")
            yield _ndjson_event("design_preview_done")
            async for chunk in _relay_streaming_response(
                _wrap_event_stream(
                    _build_chat_agent_response(
                        session_id,
                        user_message,
                        LIMITED_PROMPT,
                        persisted_user_message=original_user_message,
                    ),
                    [],
                )
            ):
                yield chunk
            return

        # DESIGN 且用途已确认后，先由统一需求解析 Agent 把自然语言转换成
        # Requirement JSON。这里刻意位于管线钢/风电钢 IF 分支之前，使两个
        # 产品分支获得完全一致的 USER_MESSAGE 结构。user_message_raw 保留本轮
        # 附件拼接后的原始有效提示词；后续解析失败、版本选择恢复或诊断日志均
        # 不会把已经增强过的 USER_MESSAGE 再次嵌套解析。
        user_message_raw = user_message
        requirement_session_context = (
            _build_cross_route_context(session_id)
            or _get_recent_session_context(session_id)
            or "（无最近会话上下文）"
        )
        try:
            yield _ndjson_event(
                "design_preview_delta",
                content="- 正在调用需求解析 Agent 生成统一 Requirement JSON。\n",
            )
            requirement_json = await asyncio.to_thread(
                parse_design_requirement,
                user_message=user_message_raw,
                purpose=purpose,
                session_context=requirement_session_context,
                dependencies=_build_requirement_parsing_dependencies(),
            )
        except RequirementParsingError as exc:
            print(f"[需求解析Agent] 结构化需求提取失败: {exc}")
            yield _ndjson_event(
                "error",
                message=f"需求解析未能形成有效的结构化 JSON，已停止本轮设计：{exc}",
            )
            yield _ndjson_event("design_preview_done")
            yield _ndjson_event("answer_done")
            return

        # 从这里开始，所有设计阶段都读取同一个增强 USER_MESSAGE：第一段保留
        # 用户原文，第二段提供经过 Pydantic 校验的结构化需求。原始消息仍由
        # original_user_message 单独保存，不改变会话显示、附件生命周期或追溯。
        user_message = build_unified_design_user_message(
            user_message_raw,
            requirement_json,
        )
        print(
            "[需求解析Agent] Requirement JSON生成完成: "
            + json.dumps(requirement_json, ensure_ascii=False)
        )
        yield _ndjson_event(
            "design_preview_delta",
            content="- 需求解析完成，后续设计智能体统一使用原始需求与结构化需求。\n",
        )

        if purpose in {"管线钢", "风电用钢"}:
            # 两个热轧设计分支共用同一条完整链路；风电分支仅替换标准、知识库和
            # X70 趋势参考模型，绝不跳过历史骨架、后置微调、三段仿真或最终报告。
            from steel_spec_extractor import extract_steel_specs, get_wind_power_standard_context

            is_wind_power_design = purpose == "风电用钢"
            design_material_label = get_wind_power_material_label(user_message) if is_wind_power_design else "管线钢"
            standard_db_name = "jgyg_db" if is_wind_power_design else "gxg_db"

            # 设计上下文控制层位于用途识别之后、任何RAG/MySQL/DLL之前。
            # 全新设计不改变原流程；续改设计先定位唯一成功快照，再提取本轮可改字段。
            reference_resolution = await asyncio.to_thread(
                resolve_design_reference,
                session_id,
                user_message,
                purpose,
                active_design_id,
                reference_design_id,
            )
            if reference_resolution.get("mode") == "clarification":
                candidates = reference_resolution.get("candidates") or []
                resume_token = _store_pending_design_reference_request(
                    session_id,
                    original_user_message,
                    user_message_raw,
                )
                yield _ndjson_event(
                    "design_reference_required",
                    message=str(
                        reference_resolution.get("message")
                        or "检测到多份可能的历史设计，请选择本轮需要修改的方案。"
                    ),
                    candidates=candidates,
                    original_prompt=original_user_message,
                    resume_token=resume_token,
                )
                yield _ndjson_event("design_preview_delta", content="- 需要先确认本次续改所引用的设计版本。\n")
                yield _ndjson_event("design_preview_done")
                yield _ndjson_event("answer_done")
                return

            design_mode = str(reference_resolution.get("mode") or "new")
            reference_snapshot = reference_resolution.get("snapshot") if design_mode == "modify" else None
            normalized_design_task = None
            design_user_message = user_message
            if reference_snapshot:
                try:
                    normalized_design_task = await asyncio.to_thread(
                        build_normalized_design_task,
                        user_message,
                        reference_snapshot,
                    )
                except DesignTaskNormalizationError as exc:
                    yield _ndjson_event(
                        "error",
                        message=f"历史设计续改任务未能可靠解析，已停止计算：{exc}",
                    )
                    yield _ndjson_event("design_preview_done")
                    yield _ndjson_event("answer_done")
                    return
                design_user_message = build_resolved_design_request(normalized_design_task)
                yield _ndjson_event(
                    "design_preview_delta",
                    content=(
                        f"- 已定位续改基准：`方案V{reference_snapshot.get('version_no')}`；"
                        "已继承牌号、成品厚度和板坯厚度，并进入完整再设计流程。\n"
                    ),
                )

            # 1. 使用与普通聊天相同的七工具 Agent 路由，由模型根据当前提示词判断是否检索。
            yield _ndjson_event("search_start", message="正在判断是否需要调用材料知识库检索工具...")
            await asyncio.sleep(0.05)
            pipeline_knowledge_docs = await asyncio.to_thread(
                _retrieve_pipeline_knowledge_docs,
                design_user_message,
            )
            yield _ndjson_event("search_done", message="知识库工具判断完成，正在生成初步方案...")
            await asyncio.sleep(0.05)

            async def compute_pipeline_design_result():
                # 2. 后台计算任务：由 RAG/LLM 提取规格边界，再按规格匹配管线钢历史实绩。
                extraction_result = await asyncio.to_thread(
                    extract_steel_specs,
                    user_message=design_user_message,
                    session_id=f"spec_{session_id}",
                    purpose=purpose,
                    db_name=standard_db_name,
                    return_range_stages=True,
                )
                spec, rag_range_spec, final_range_spec = extraction_result
                if is_wind_power_design:
                    # 风电产品标准不规定统一 FET/FDT 等厂内 TMCP 数值。
                    # 保留原始 RAG 范围用于前端溯源展示，但最终查询/微调范围只
                    # 接受用户明确工艺单值，否则开放，避免文献数字污染硬筛选。
                    spec = _normalize_wind_power_process_spec(spec, design_user_message)
                    final_range_spec = _normalize_wind_power_process_spec(
                        final_range_spec,
                        design_user_message,
                    )
                if reference_snapshot:
                    # 继承规格在标准提取后再次确定性收口，防止RAG范围或兜底值
                    # 把父方案的牌号、成品厚度、板坯厚度改掉。
                    inherited = (normalized_design_task or {}).get("inherited_constraints") or {}
                    inherited_aim = _to_float(inherited.get("product_thickness_mm"))
                    inherited_slab = _to_float(inherited.get("slab_thickness_mm"))
                    if inherited_aim is not None:
                        spec["AIM_THICK_min"] = inherited_aim
                        spec["AIM_THICK_max"] = inherited_aim
                        final_range_spec["AIM_THICK_min"] = inherited_aim
                        final_range_spec["AIM_THICK_max"] = inherited_aim
                    if inherited_slab is not None:
                        spec["SLAB_THICK_min"] = inherited_slab
                        spec["SLAB_THICK_max"] = inherited_slab
                        final_range_spec["SLAB_THICK_min"] = inherited_slab
                        final_range_spec["SLAB_THICK_max"] = inherited_slab

                # 全新设计和续改设计都正常执行SQL历史匹配。续改时该结果只作为
                # 生产参考，真正传入后置微调的主骨架始终是父DesignSnapshot。
                sql_matched_reference = await asyncio.to_thread(
                    match_wind_power_steel_process if is_wind_power_design else match_pipeline_steel_process,
                    spec_result=spec,
                    user_message=design_user_message,
                    session_id=session_id,
                )
                if reference_snapshot:
                    matched = copy.deepcopy(reference_snapshot.get("matched_result") or {})
                    if not matched.get("arrBody"):
                        raise RuntimeError("引用方案缺少有效 matched_result，无法继续完整再设计")
                else:
                    matched = sql_matched_reference
                try:
                    historical_roll_rows = await asyncio.to_thread(
                        _query_nearest_pipeline_historical_rows,
                        spec,
                        design_user_message,
                        PIPELINE_HISTORICAL_ROLL_REFERENCE_LIMIT,
                    )
                    historical_roll_reference_markdown = _build_pipeline_historical_roll_markdown(
                        historical_roll_rows
                    )
                    print(
                        "[历史轧制规程参考] 已生成参考 Markdown: "
                        f"样本数={len(historical_roll_rows)}, "
                        f"字符数={len(historical_roll_reference_markdown)}"
                    )
                except Exception as exc:
                    historical_roll_rows = []
                    historical_roll_reference_markdown = ""
                    print(
                        "[历史轧制规程参考] 查询或 Markdown 转换失败，"
                        f"本轮继续使用当前 matched_result: {type(exc).__name__}: {exc}"
                    )
                wind_standard_context = (
                    get_wind_power_standard_context(f"spec_{session_id}")
                    if is_wind_power_design else {}
                )
                return (
                    spec,
                    matched,
                    rag_range_spec,
                    final_range_spec,
                    wind_standard_context,
                    historical_roll_reference_markdown,
                    historical_roll_rows,
                    sql_matched_reference,
                )

            # 3. 初步方案流式输出与规格提取/实绩匹配并行执行，先保证用户能看到前置方案。
            flash_preview_queue = asyncio.Queue()
            flash_preview_task = asyncio.create_task(
                _stream_flash_design_preview_to_queue(
                    design_user_message,
                    [],
                    flash_preview_queue,
                    purpose=purpose,
                    emit_lifecycle=False,
                )
            )
            yield _ndjson_event("design_preview_delta", content=f"- 正在调用 LLM 生成{design_material_label}材料设计初步方案...\n\n")
            compute_task = asyncio.create_task(compute_pipeline_design_result())

            while True:
                flash_event = await flash_preview_queue.get()
                if flash_event is None:
                    break
                yield flash_event

            await flash_preview_task
            yield _ndjson_event("design_preview_delta", content="\n\n- 初步方案流式生成完成；等待规格提取和历史实绩匹配结果。\n")
            (
                spec_result,
                matched_result,
                rag_range_spec,
                final_range_spec,
                wind_standard_context,
                historical_roll_reference_markdown,
                historical_roll_rows,
                sql_matched_reference,
            ) = await compute_task
            if is_wind_power_design and wind_standard_context.get("error"):
                # 明确的交货状态、钢级或厚度适用范围不被标准覆盖时，不能降级为
                # 管线钢或沿用历史实绩继续计算，必须在完整设计链路前停止。
                yield _ndjson_event("error", message=wind_standard_context["error"])
                yield _ndjson_event("design_preview_done")
                return
            yield _ndjson_event(
                "design_preview_delta",
                content=_visible_reasoning_content(
                    "本地知识库检索",
                    [
                        f"知识库 Agent 已召回 {len(pipeline_knowledge_docs or [])} 条参考片段。",
                        "召回结果将作为最终报告的材料学参考；未调用工具时按0条显示。",
                    ],
                ),
            )
            yield _ndjson_event(
                "design_preview_delta",
                content=(
                    "\n#### RAG 提取范围 JSON\n\n"
                    "```json\n"
                    f"{json.dumps(rag_range_spec, ensure_ascii=False, indent=2)}\n"
                    "```\n\n"
                ),
            )
            await asyncio.sleep(0.01)
            yield _ndjson_event(
                "design_preview_delta",
                content=(
                    "#### 兜底后最终范围 JSON\n\n"
                    "```json\n"
                    f"{json.dumps(final_range_spec, ensure_ascii=False, indent=2)}\n"
                    "```\n\n"
                ),
            )
            await asyncio.sleep(0.01)
            yield _ndjson_event(
                "design_preview_delta",
                content=_visible_reasoning_content(
                    "规格提取与历史实绩匹配",
                    [
                        "规格提取已把用户需求转换为成分、厚度、性能和工艺边界。",
                        "历史实绩匹配已返回完整 matched_result，后续微调以该结构为基础。",
                        f"当前匹配状态 isState={matched_result.get('isState') if isinstance(matched_result, dict) else '未知'}。",
                    ],
                ),
            )
            yield _ndjson_event("design_preview_delta", content="- 规格提取与历史实绩匹配完成。\n")
            # 在任何成分/工艺数值重设计之前，先由真正的 LangChain Agent 调用
            # 当前产品 RAG 和当前目标厚度历史工具，形成五模块结构化变更评估。
            yield _ndjson_event(
                "design_preview_delta",
                content="- 正在调用设计变更评估 Agent 检索资料并判断各模块调整范围。\n",
            )
            assessment_session_context = (
                _build_cross_route_context(session_id)
                or _get_recent_session_context(session_id)
                or "（无）"
            )
            if is_wind_power_design:
                assessment_session_context = _filter_wind_power_session_context(
                    assessment_session_context
                )
            target_thickness = _extract_pipeline_target_thickness_from_text(
                design_user_message
            )
            if target_thickness is None:
                lower = _to_float(spec_result.get("AIM_THICK_min"))
                upper = _to_float(spec_result.get("AIM_THICK_max"))
                if lower is not None and upper is not None and abs(lower - upper) <= 1e-9:
                    target_thickness = lower
            target_slab_thickness = _extract_pipeline_target_slab_thickness_from_text(
                design_user_message
            )
            target_summary = {
                "steel_grade": extract_version_target_grade(design_user_message) or "",
                "thickness_mm": target_thickness,
                "slab_thickness_mm": target_slab_thickness,
            }
            reference_summary = _build_design_assessment_summary(
                (reference_snapshot or {}).get("matched_result")
                if reference_snapshot else None
            )
            if reference_snapshot:
                reference_summary.update({
                    "design_id": reference_snapshot.get("design_id"),
                    "version_no": reference_snapshot.get("version_no"),
                    "steel_grade": reference_snapshot.get("steel_grade") or "",
                })
            try:
                design_change_assessment = await asyncio.to_thread(
                    assess_design_change,
                    material_name=design_material_label,
                    user_message=user_message,
                    session_context=assessment_session_context,
                    spec_result=spec_result,
                    reference_summary=reference_summary,
                    target_summary=target_summary,
                    engineering_standard_context=(
                        wind_standard_context if is_wind_power_design else {}
                    ),
                    matched_result_summary=_build_design_assessment_summary(
                        matched_result
                    ),
                    dependencies=_build_design_change_assessment_dependencies(
                        is_wind=is_wind_power_design,
                        historical_rows=historical_roll_rows,
                        spec_result=spec_result,
                        user_message=design_user_message,
                    ),
                )
            except DesignChangeAssessmentError as exc:
                yield _ndjson_event(
                    "error",
                    message=f"设计变更评估 Agent 连续两次未完成结构化判断，已停止计算：{exc}",
                )
                yield _ndjson_event("design_preview_done")
                yield _ndjson_event("answer_done")
                return
            print(
                "[设计变更评估Agent] 结构化结论: "
                + json.dumps(design_change_assessment, ensure_ascii=False)
            )
            yield _ndjson_event(
                "design_preview_delta",
                content=_visible_reasoning_content(
                    "设计变更评估 Agent",
                    [
                        "Agent 已自主调用当前产品知识库和当前目标厚度历史实绩工具。",
                        "已分别形成成分、加热、轧制、冷却及性能要求的继承/重评估/重设计结论。",
                        "后续每个专业智能体只读取属于自己的评估模块。",
                    ],
                ),
            )
            # 智能体阶段 1/4：后置成分、性能及初步轧制规程微调。
            # 此步骤必须在初步方案流结束后执行；输入当前规格边界、历史匹配结果、
            # 用户原始需求、会话标识、工程标准和十组相近厚度轧制实绩。
            yield _ndjson_event("design_preview_delta", content="- 正在调用 LangChain 微调 Agent 对 matched_result 做成分/工艺协同设计。\n")
            try:
                matched_result = await asyncio.to_thread(
                    refine_composition_process_performance,
                    spec_result,
                    matched_result,
                    design_user_message,
                    session_id,
                    material_name=design_material_label,
                    engineering_standard_context=wind_standard_context if is_wind_power_design else None,
                    historical_roll_reference_markdown=historical_roll_reference_markdown,
                    normalized_design_task=normalized_design_task,
                    reference_snapshot=reference_snapshot,
                    sql_match_reference=sql_matched_reference,
                    design_change_assessment=design_change_assessment,
                    dependencies=_build_composition_refinement_dependencies(
                        is_wind=is_wind_power_design,
                        historical_rows=historical_roll_rows,
                        spec_result=spec_result,
                        user_message=design_user_message,
                        engineering_standard_context=(
                            wind_standard_context if is_wind_power_design else None
                        ),
                    ),
                )
            except DesignRevisionValidationError as exc:
                yield _ndjson_event(
                    "error",
                    message=(
                        "历史方案续改经过多轮完整重新设计后仍未满足继承规格、"
                        f"微合金减量或相对性能约束，已停止后续仿真：{exc}"
                    ),
                )
                yield _ndjson_event("design_preview_done")
                yield _ndjson_event("answer_done")
                return
            except CompositionRefinementValidationError as exc:
                yield _ndjson_event(
                    "error",
                    message=(
                        "成分、性能、轧制规程或冷却初值经过多轮 Agent 重新设计后仍未通过"
                        f"确定性校验，已停止后续仿真：{exc}"
                    ),
                )
                yield _ndjson_event("design_preview_done")
                yield _ndjson_event("answer_done")
                return
            except WindPowerDesignValidationError as exc:
                # 风电标准与后置轧制规程校验是进入三段工艺智能体的强制门禁。
                # 多轮重设计仍不合格时正常结束 NDJSON 流，不回退历史字段。
                yield _ndjson_event(
                    "error",
                    message=(
                        "风电用钢成分、性能或轧制规程经过多轮重新设计后仍未通过"
                        f"标准与工艺校验，已停止后续仿真：{exc}"
                    ),
                )
                yield _ndjson_event("design_preview_done")
                yield _ndjson_event("answer_done")
                return
            if is_wind_power_design:
                wind_validation_error = _validate_wind_power_matched_result(
                    matched_result,
                    wind_standard_context,
                )
                if wind_validation_error:
                    yield _ndjson_event(
                        "error",
                        message=(
                            "风电用钢成分/性能设计未通过 GB/T 1591 标准及 CEV/Pcm 校验："
                            + wind_validation_error
                        ),
                    )
                    yield _ndjson_event("design_preview_done")
                    yield _ndjson_event("answer_done")
                    return
            yield _ndjson_event(
                "design_preview_delta",
                content=_visible_reasoning_content(
                    "后置成分/工艺微调",
                    [
                        "后端已锁定用户明确的成品厚度和板坯厚度；本阶段检查成分、性能规格边界及轧制规程。",
                        "LangChain Agent 在 matched_result 固定结构内设计成分、YS/TS/EL/AKV、轧制道次和冷却初值。",
                        "Agent 返回结果先检查结构、成分性能边界、道次连续性、时间关系和严格冷却温度顺序；后续专业智能体继续结合仿真结果校核。",
                    ],
                ),
            )
            yield _ndjson_event(
                "design_preview_delta",
                content=_format_reasoning_content(
                    "后置微调结果校验",
                    _pop_reasoning_content(f"{session_id}:pipeline_refine"),
                    [
                        "已得到 LangChain 微调 Agent 输出的 matched_result。",
                        f"微调后 isState={matched_result.get('isState') if isinstance(matched_result, dict) else '未知'}。",
                        "后续进入加热、轧制、冷却三个工艺智能体进行仿真与工艺一致性校验。",
                    ],
                ),
            )
            yield _ndjson_event("design_preview_delta", content="- 后置 LangChain Agent 成分/性能/工艺微调完成。\n")
            # 5. 后置 LLM 微调完成后，按加热 -> 轧制 -> 冷却顺序调用工艺智能体。
            #    三个智能体内部都会调用各自 DLL 生成仿真图片，因此这里不再额外调用总绘图 DLL。
            pipeline_image_started_at = time.time()
            dll_target_context = (
                WIND_POWER_DLL_CONTEXT_MARKER
                if is_wind_power_design
                else design_user_message
            )
            target_grade, dll_reference_grade = _resolve_pipeline_dll_grade(
                matched_result,
                dll_target_context,
            )
            dll_mapping_context = (
                WIND_POWER_DLL_CONTEXT_MARKER + "\n"
                + f"【本轮设计对象】{design_material_label}\n"
                "【风电用钢工程标准】\n"
                + json.dumps(wind_standard_context, ensure_ascii=False)
                + "\n历史实绩中的牌号、成分和性能仅用于固定字段结构，不得作为本轮设计或工艺判断依据。"
                if is_wind_power_design else (
                "【DLL仿真模型映射】\n"
                f"目标设计牌号为 {target_grade}；当前现有DLL采用 {dll_reference_grade} 参考模型进行近似仿真。"
                "必须继续按照用户目标牌号的成分、性能标准和工艺目标判断，"
                "不得把DLL参考模型解释为最终设计牌号。"
                if target_grade and dll_reference_grade
                else "")
            )
            recent_agent_context = (
                _build_cross_route_context(session_id)
                or _get_recent_session_context(session_id)
                or "（无）"
            )
            if is_wind_power_design:
                recent_agent_context = _filter_wind_power_session_context(recent_agent_context)
            revision_agent_context = ""
            if reference_snapshot and normalized_design_task:
                task_for_agent = copy.deepcopy(normalized_design_task)
                task_for_agent.pop("target_design_id", None)
                parent_row = _matched_result_body_to_row(
                    reference_snapshot.get("matched_result") or {}
                )
                parent_reference_fields = {
                    field: parent_row.get(field)
                    for field in (
                        "AIM_THICK", "SLAB_THICK", "C", "SI", "MN", "P", "S", "N",
                        "NB", "V", "TI", "AL", "ALS", "CU", "CR", "NI", "CO", "MO", "B",
                        "YS", "TS", "EL", "AKV",
                    )
                    if parent_row.get(field) is not None
                }
                current_row = _matched_result_body_to_row(matched_result)
                changed_composition_fields = [
                    field
                    for field in (
                        "C", "SI", "MN", "P", "S", "N", "NB", "V", "TI", "AL",
                        "ALS", "CU", "CR", "NI", "CO", "MO", "B",
                    )
                    if str(parent_row.get(field, "")).strip()
                    != str(current_row.get(field, "")).strip()
                ]
                revision_agent_context = (
                    "【历史方案续改标准任务】\n"
                    + json.dumps(task_for_agent, ensure_ascii=False, indent=2)
                    + "\n【父方案关键成分、规格与性能基线】\n"
                    + json.dumps(parent_reference_fields, ensure_ascii=False, indent=2)
                    + "\n【成分变更后的工艺再优化】\n"
                    + "本轮相对父方案实际变化的成分字段："
                    + ("、".join(changed_composition_fields) if changed_composition_fields else "无")
                    + "。三个工艺智能体必须使用本轮 DLL 结果重新判断，不能直接沿用父方案结论。\n"
                    + "成分变化后，加热、轧制、冷却智能体均可按仿真结果调整工艺；"
                    "牌号、成品厚度和板坯厚度不得改变，四项性能不得低于父方案。"
                )
            pipeline_agent_base_context = "\n\n".join([
                f"【用户原始需求】\n{user_message}",
                f"【最近会话上下文】\n{recent_agent_context}",
                *([revision_agent_context] if revision_agent_context else []),
                *([dll_mapping_context] if dll_mapping_context else []),
            ])
            pipeline_agent_judgement_contexts: list[tuple[str, str]] = []

            def build_pipeline_agent_context(stage_name: str) -> str:
                """只向当前专业智能体传递属于该阶段的结构化评估模块。"""
                assessment_module = (
                    (design_change_assessment.get("change_assessment") or {}).get(stage_name)
                    or {}
                )
                stage_context = (
                    pipeline_agent_base_context
                    + "\n\n【设计变更评估 Agent：当前模块】\n"
                    + json.dumps(assessment_module, ensure_ascii=False, indent=2)
                    + "\n执行语义：INHERIT 可用父方案为初值但仍须运行当前 DLL；"
                    "REASSESS 必须按当前成分、规格和 DLL 重新判断；"
                    "REDESIGN 必须重新设计该模块。"
                )
                if not pipeline_agent_judgement_contexts:
                    return stage_context
                prior_judgements = "\n\n".join(
                    f"### {stage_label}\n{judgement_text}"
                    for stage_label, judgement_text in pipeline_agent_judgement_contexts
                    if judgement_text
                )
                return (
                    stage_context
                    + "\n\n【本轮前序工艺智能体判断正文】\n"
                    + prior_judgements
                )

            def remember_pipeline_agent_judgement(stage_label: str, agent_event: dict) -> None:
                judgement_context = str(agent_event.get("judgement_context") or "").strip()
                if judgement_context:
                    pipeline_agent_judgement_contexts.append((stage_label, judgement_context))

            # 智能体阶段 2/4：加热工艺仿真与微调。
            # 必须承接阶段1的 matched_result；每轮先运行加热 DLL，再由判断模型校验，
            # 每轮进度仍通过原异步队列实时传给前端。
            reheat_agent_failed = False
            async for agent_event in _stream_pipeline_agent_execution(
                refine_reheat_process,
                matched_result,
                build_pipeline_agent_context("heating"),
                session_id,
                "reheat",
                "工艺智能体判断和加热工艺校验",
                [
                    "加热智能体已调用加热 DLL 生成模拟结果，并读取全固溶温度、均热温度场和晶粒相关图片作为参考。",
                    "分析模型根据均热温度场均匀性、均热温度、均热时长和晶粒尺寸判断加热工艺是否符合文献结论。",
                    "后端只采纳 isState、SOAK_TEMP、SOAK_TIME、HEAT_TEMP1/2/3 的合法调整，并将 FURNACE_EXIT_TEMP 自动同步为 SOAK_TEMP；修改均热温度后若相邻温度绝对差超过20℃，后端保留 SOAK_TEMP 并从 HEAT_TEMP3 向前自动削减超出部分，不要求单调递增。",
                ],
                agent_kwargs={
                    "dependencies": _build_process_agent_dependencies(),
                },
            ):
                if agent_event["type"] == "progress":
                    yield _ndjson_event("design_preview_delta", content=agent_event["markdown"])
                else:
                    matched_result = agent_event["result"]
                    reheat_agent_failed = bool(agent_event.get("failed"))
                    remember_pipeline_agent_judgement("加热工艺智能体判断", agent_event)
            _save_pipeline_stage_matched_result("reheat", matched_result)
            yield _ndjson_event(
                "design_preview_delta",
                content=(
                    "- 加热工艺智能体未完成判断：模型结果连续未通过结构或调整校验，已保留进入失败轮次前的工艺。\n"
                    if reheat_agent_failed
                    else "- 加热工艺智能体完成：已完成加热仿真、工艺判断和加热工艺校验。\n"
                ),
            )
            # 智能体阶段 3/4：控制轧制仿真、道次重设计及最终硬门禁。
            # 输入必须是加热阶段结果，并附带十组相近厚度历史轧制实绩；校验不通过时
            # 保持原异常语义，停止冷却和报告生成。
            roll_agent_failed = False
            try:
                async for agent_event in _stream_pipeline_agent_execution(
                    refine_rolling_process,
                    matched_result,
                    build_pipeline_agent_context("rolling"),
                    session_id,
                    "roll",
                    "工艺智能体判断和道次工艺校验",
                    [
                        "轧制智能体已调用轧制 DLL 生成各道次晶粒尺寸相关模拟结果。",
                        "分析模型判断 FET、FDT、道次厚度、速度、轧制力、温度分布和道次间隔时间是否支持晶粒细化与工艺衔接。",
                        "后端校验有效道次连续性、最终厚度、末道次温度和时间顺序，并强制中间坯待温时间不小于终轧到开冷时间。",
                    ],
                    agent_kwargs={
                        "historical_roll_reference_markdown": historical_roll_reference_markdown,
                        "dependencies": _build_process_agent_dependencies(),
                    },
                ):
                    if agent_event["type"] == "progress":
                        yield _ndjson_event("design_preview_delta", content=agent_event["markdown"])
                    else:
                        matched_result = agent_event["result"]
                        roll_agent_failed = bool(agent_event.get("failed"))
                        remember_pipeline_agent_judgement("轧制工艺智能体判断", agent_event)
            except PipelineRollValidationError as exc:
                print(f"[管线钢轧制智能体] 最终硬门禁终止设计流程: {exc}")
                yield _ndjson_event(
                    "error",
                    message=(
                        "轧制智能体重新设计后仍未通过最终厚度及道次一致性校验，"
                        f"已停止冷却仿真和最终报告生成：{exc}"
                    ),
                )
                yield _ndjson_event("design_preview_done")
                yield _ndjson_event("answer_done")
                return
            _save_pipeline_stage_matched_result("roll", matched_result)
            yield _ndjson_event(
                "design_preview_delta",
                content=(
                    "- 轧制工艺智能体未完成判断：模型结果连续未通过结构或调整校验，已保留进入失败轮次前的工艺。\n"
                    if roll_agent_failed
                    else "- 轧制工艺智能体完成：已完成轧制仿真、工艺判断和道次工艺校验。\n"
                ),
            )
            # 轧制智能体结束后静默启动粗轧/精轧出口奥氏体晶粒图任务。
            # 这里不发送任何前端事件；两个后台任务与后续冷却智能体并发执行。
            exit_grain_drawing_job = _start_pipeline_exit_grain_drawing(
                matched_result,
                build_pipeline_agent_context("rolling"),
            )
            # 智能体阶段 4/4：控制冷却仿真与最终性能校验。
            # 该阶段只在轧制硬门禁通过后执行，并继续复用原来的每轮流式进度、
            # 返红温度兜底和最终参数变化后补仿真的行为。
            cooling_agent_failed = False
            try:
                async for agent_event in _stream_pipeline_agent_execution(
                    refine_cooling_process,
                    matched_result,
                    build_pipeline_agent_context("cooling"),
                    session_id,
                    "cooling",
                    "工艺智能体判断和控制冷却工艺校验",
                    [
                        "冷却智能体已调用冷却 DLL 生成相组成模拟结果。",
                        "分析模型判断冷却后铁素体晶粒尺寸与相比例分数是否符合文献结论。",
                        "后端只保留 isState、TIME_ENTR、TEMP_ENTR、SELF_TEMP 及随工艺变化同步更新的 YS、TS、EL、AKV；TIME_ENTR 参考最后有效轧制道次时刻和目标组织动态调整；SELF_TEMP 默认严格小于500℃，仅当前用户明确要求大于或等于500℃时允许突破；模型调整全部失败时将 SELF_TEMP 确定性设置为485℃并补做最终冷却仿真；FDT 仅作为终轧温度参考且保持原值，越界性能恢复后置微调合格基线，其他冷却、轧制、加热和成分字段保持原值。",
                    ],
                    agent_kwargs={
                        "dependencies": _build_process_agent_dependencies(),
                    },
                ):
                    if agent_event["type"] == "progress":
                        yield _ndjson_event("design_preview_delta", content=agent_event["markdown"])
                    else:
                        matched_result = agent_event["result"]
                        cooling_agent_failed = bool(agent_event.get("failed"))
                        remember_pipeline_agent_judgement("控制冷却工艺智能体判断", agent_event)
                _save_pipeline_stage_matched_result("cooling", matched_result)
                yield _ndjson_event(
                    "design_preview_delta",
                    content=(
                        "- 冷却工艺智能体未完成判断：模型结果连续未通过结构或调整校验，已保留进入失败轮次前的工艺。\n"
                        if cooling_agent_failed
                        else "- 冷却工艺智能体完成：已完成冷却仿真、工艺判断和控制冷却工艺校验。\n"
                    ),
                )
                if is_wind_power_design:
                    final_wind_error = _validate_wind_power_matched_result(
                        matched_result,
                        wind_standard_context,
                        spec_result,
                    )
                    if final_wind_error:
                        yield _ndjson_event(
                            "error",
                            message=(
                                "冷却智能体最终结果未通过风电用钢 GB/T 1591、CEV/Pcm "
                                "及性能复核，已停止报告生成：" + final_wind_error
                            ),
                        )
                        yield _ndjson_event("design_preview_done")
                        yield _ndjson_event("answer_done")
                        return
            except PipelineRollValidationError as exc:
                yield _ndjson_event(
                    "error",
                    message=f"控制冷却最终严格门禁未通过，已停止报告生成：{exc}",
                )
                yield _ndjson_event("design_preview_done")
                yield _ndjson_event("answer_done")
                return
            finally:
                # 报告准备阶段绝不等待后台绘图；保留已成功复制的图片，并终止未完成进程树。
                if exit_grain_drawing_job is not None:
                    exit_grain_drawing_job.finish_without_waiting()

            if reference_snapshot:
                # 三段智能体均可按仿真需要调整。全部结束后统一检查继承规格、
                # 末道厚度、模型自主选择的微合金总量及四项相对性能。
                revision_errors = validate_revision_constraints(
                    matched_result,
                    reference_snapshot,
                    normalized_design_task,
                    spec_result,
                    require_final_pass=True,
                )
                if revision_errors:
                    yield _ndjson_event(
                        "error",
                        message=(
                            "历史方案续改最终强约束未通过，未保存新版本且不生成报告："
                            + "；".join(revision_errors)
                        ),
                    )
                    yield _ndjson_event("design_preview_done")
                    yield _ndjson_event("answer_done")
                    return

            # 冷却及全部最终约束通过后，同步绘制析出形貌。该图片是可选报告证据：
            # 绘图、DLL 后处理或复制失败时只跳过该图，其它图片和报告继续生成。
            precipitate_image_ready = await asyncio.to_thread(
                _draw_pipeline_precipitate_morphology,
                matched_result,
                build_pipeline_agent_context("cooling"),
            )
            if not precipitate_image_ready:
                yield _ndjson_event(
                    "design_preview_delta",
                    content="- 析出形貌绘制或图片后处理失败，本轮已跳过该图并继续生成其它报告内容。\n",
                )

            # 6. arrBody 是单键字典列表，这里摊平成字段字典，后续表格统一从 matched_row 取值。
            matched_row = {}
            for item in matched_result.get("arrBody", []):
                if isinstance(item, dict) and len(item) == 1:
                    key, value = next(iter(item.items()))
                    matched_row[str(key).upper()] = value

            def format_pipeline_table_value(value, value_type: str) -> str:
                # 7. 只做展示格式化；报告正文必须继续使用 fact_table 中的最终字符串值。
                number = _to_float(value)
                if number is None:
                    return str(value)
                if value_type == "component":
                    return f"{number:.4f}"
                if value_type == "temperature":
                    return f"{number:.0f}"
                if value_type == "thickness":
                    return f"{number:.2f}"
                return str(value)

            # 8. 成分字段按用户需求/spec 关注项动态选择，避免把无关元素塞进报告。
            report_table_rows = []
            for field in _report_component_fields(spec_result, user_message):
                value = matched_row.get(field)
                if value is None or str(value).strip() == "":
                    continue
                report_table_rows.append((
                    "成分",
                    _component_label_from_context(field, user_message),
                    format_pipeline_table_value(value, "component"),
                    "wt%",
                ))

            # 9. 工艺与性能字段固定来自匹配到的实绩数据，作为最终报告唯一事实表的一部分。
            pipeline_report_fields = [
                ("工艺", "板坯厚度", "SLAB_THICK", "mm", "thickness"),
                ("工艺", "板坯宽度", "SLAB_WIDTH", "mm", "thickness"),
                ("工艺", "板坯长度", "SLAB_LEN", "mm", "thickness"),
                ("工艺", "成品厚度", "AIM_THICK", "mm", "thickness"),
                ("工艺", "均热温度", "SOAK_TEMP", "℃", "temperature"),
                ("工艺", "精轧开轧温度 FET", "FET", "℃", "temperature"),
                ("工艺", "精轧终轧温度 FDT", "FDT", "℃", "temperature"),
                ("工艺", "入水温度 TEMP_ENTR", "TEMP_ENTR", "℃", "temperature"),
                ("工艺", "返红温度 SELF_TEMP", "SELF_TEMP", "℃", "temperature"),
                ("力学性能", "屈服强度 YS", "YS", "MPa", "raw"),
                ("力学性能", "抗拉强度 TS", "TS", "MPa", "raw"),
                ("力学性能", "断后伸长率 EL", "EL", "%", "raw"),
                ("力学性能", "冲击功 AKV", "AKV", "J", "raw"),
            ]
            for category, label, field, unit, value_type in pipeline_report_fields:
                value = matched_row.get(field)
                if value is None or str(value).strip() == "":
                    continue
                report_table_rows.append((
                    category,
                    label,
                    format_pipeline_table_value(value, value_type),
                    unit,
                ))

            # 10. 同时生成前端展示用 Markdown 表格和传给 LLM 的结构化 fact_table。
            report_table_lines = [
                '<div class="report-table-caption"><strong>表2　成分、工艺与性能参数</strong></div>',
                "",
                "| 类别 | 项目 | 数值 | 单位 |",
                "|:---:|:---:|:---:|:---:|",
            ]
            for category, label, value, unit in report_table_rows:
                report_table_lines.append(f"| {category} | {label} | {value} | {unit} |")
            report_table_markdown = "\n".join(report_table_lines) + "\n\n"
            performance_standard_markdown = _build_pipeline_performance_standard_markdown(
                spec_result
            )
            report_document_header = (
                f"# {design_material_label}热轧工艺设计报告\n\n"
                if is_wind_power_design else "# 管线钢热轧工艺设计报告\n\n"
            )
            report_table_reference = [
                {"类别": category, "项目": label, "数值": value, "单位": unit}
                for category, label, value, unit in report_table_rows
            ]
            full_fact_table_reference = _build_full_fact_table_from_matched_result(matched_result)
            rolling_schedule_markdown = _build_pipeline_rolling_schedule_markdown(matched_result)

            # 工艺链已完成并形成权威 fact_table，此时才分配版本号。中间失败任务
            # 不会执行到这里，因此不会占用 V 序号。UUID仅通过协议回传，不写入报告。
            snapshot_target_grade = (
                extract_version_target_grade(design_user_message, purpose)
                or (reference_snapshot or {}).get("target_grade")
                or target_grade
            )
            saved_design_snapshot = None
            if not (reheat_agent_failed or roll_agent_failed or cooling_agent_failed):
                saved_design_snapshot = await asyncio.to_thread(
                    design_snapshot_store.save_snapshot,
                    session_id=session_id,
                    material_purpose=purpose,
                    target_grade=snapshot_target_grade,
                    aim_thick=_to_float(matched_row.get("AIM_THICK")),
                    slab_thick=_to_float(matched_row.get("SLAB_THICK")),
                    user_request=original_user_message,
                    change_plan=normalized_design_task or {"mode": "new"},
                    spec_result=spec_result,
                    matched_result=matched_result,
                    fact_table=full_fact_table_reference,
                    parent_design_id=(reference_snapshot or {}).get("design_id"),
                )
                yield _ndjson_event(
                    "design_context",
                    design_id=saved_design_snapshot.get("design_id"),
                    version=f"V{saved_design_snapshot.get('version_no')}",
                    parent_design_id=saved_design_snapshot.get("parent_design_id"),
                    mode=design_mode,
                )
            else:
                print("[设计快照] 本轮存在未完成的工艺智能体判断，不占用设计版本号")

            def rebuild_pipeline_report_facts(current_result: dict):
                """风险评估调整工艺后，重新生成权威简表和完整 fact_table。"""
                current_row = _matched_result_body_to_row(current_result)
                current_rows = []
                for component_field in _report_component_fields(spec_result, user_message):
                    component_value = current_row.get(component_field)
                    if component_value is None or str(component_value).strip() == "":
                        continue
                    current_rows.append((
                        "成分",
                        _component_label_from_context(component_field, user_message),
                        format_pipeline_table_value(component_value, "component"),
                        "wt%",
                    ))
                for category, label, field, unit, value_type in pipeline_report_fields:
                    field_value = current_row.get(field)
                    if field_value is None or str(field_value).strip() == "":
                        continue
                    current_rows.append((
                        category,
                        label,
                        format_pipeline_table_value(field_value, value_type),
                        unit,
                    ))
                current_table_lines = [
                    '<div class="report-table-caption"><strong>表2　成分、工艺与性能参数</strong></div>',
                    "",
                    "| 类别 | 项目 | 数值 | 单位 |",
                    "|:---:|:---:|:---:|:---:|",
                ]
                for category, label, value, unit in current_rows:
                    current_table_lines.append(f"| {category} | {label} | {value} | {unit} |")
                return (
                    current_row,
                    "\n".join(current_table_lines) + "\n\n",
                    [
                        {"类别": category, "项目": label, "数值": value, "单位": unit}
                        for category, label, value, unit in current_rows
                    ],
                    _build_full_fact_table_from_matched_result(current_result),
                )

            # 11. 注册 DLL 输出的图片文件，LLM 只拿图片名称说明，前端展示使用受控的图片 URL。
            image_references_for_llm = []
            image_markdown_blocks = []
            image_markdown_blocks_by_stage = {
                "加热阶段": [],
                "控制轧制阶段": [],
                "控制冷却阶段": [],
            }
            pipeline_image_dir = _os.path.join(
                PIPELINE_IMAGE_GENERATOR_BIN_DIR,
                "ModelManage",
                str(matched_result.get("strCoil", "")),
                "Image",
            )
            resolved_pipeline_image_dir = _find_pipeline_generated_image_dir(
                pipeline_image_dir,
                pipeline_image_started_at,
            )
            pipeline_image_display_order = [
                "均热温度.png",
                "晶粒长大.png",
                "晶粒尺寸分布.png",
                "粗轧入口温度.png",
                "精轧入口温度.png",
                "终轧温度.png",
                "轧制力.png",
                "扭矩.png",
                "摩擦系数.png",
                "各道次晶粒尺寸.png",
                "软化率.png",
                "RPTT.png",
                "析出动力学.png",
                "粗轧出口奥氏体晶粒尺寸.png",
                "精轧出口奥氏体晶粒尺寸.png",
                "温度场曲线.png",
                "CCT.png",
                "相组成.png",
                "析出形貌.png",
                "强化机制.png",
                "氧化铁皮厚度.png",
            ]
            pipeline_image_stage_map = {
                "均热温度.png": "加热阶段",
                "粗轧入口温度.png": "控制轧制阶段",
                "精轧入口温度.png": "控制轧制阶段",
                "终轧温度.png": "控制轧制阶段",
                "轧制力.png": "控制轧制阶段",
                "扭矩.png": "控制轧制阶段",
                "摩擦系数.png": "控制轧制阶段",
                "温度场曲线.png": "控制冷却阶段",
                "晶粒长大.png": "加热阶段",
                "晶粒尺寸分布.png": "加热阶段",
                "各道次晶粒尺寸.png": "控制轧制阶段",
                "软化率.png": "控制轧制阶段",
                "析出动力学.png": "控制轧制阶段",
                "RPTT.png": "控制轧制阶段",
                "粗轧出口奥氏体晶粒尺寸.png": "控制轧制阶段",
                "精轧出口奥氏体晶粒尺寸.png": "控制轧制阶段",
                "CCT.png": "控制冷却阶段",
                "相组成.png": "控制冷却阶段",
                "析出形貌.png": "控制冷却阶段",
                "强化机制.png": "控制冷却阶段",
                "氧化铁皮厚度.png": "控制冷却阶段",
            }
            # 兜底扩展时也要排除的图片：仅保留报告需要展示的仿真图，避免无关力能/析出细节图进入前端。
            pipeline_excluded_image_names = {
                "加热Ⅱ温度.png",
                "加热温度.png",
                "加热段温度.png",
                "冷却温度.png",
                "组织形貌.png",
                "析出相尺寸分布.png",
                "析出相体积分数.png",
                "析出尺寸.png",
                "析出分数.png",
                "驱动力钉轧力.png",
                "CCP.png",
            }
            try:
                if resolved_pipeline_image_dir:
                    available_images = _list_png_images(resolved_pipeline_image_dir)
                    available_by_casefold = {
                        image_name.casefold(): (image_name, image_path)
                        for image_name, image_path in available_images.items()
                    }
                    ordered_images = []
                    ordered_casefold_names = set()
                    for canonical_name in pipeline_image_display_order:
                        available_item = available_by_casefold.get(canonical_name.casefold())
                        if not available_item:
                            continue
                        ordered_images.append((canonical_name, available_item[1]))
                        ordered_casefold_names.add(canonical_name.casefold())
                    excluded_casefold_names = {
                        image_name.casefold() for image_name in pipeline_excluded_image_names
                    }
                    for actual_name, image_path in sorted(available_images.items()):
                        folded_name = actual_name.casefold()
                        if (
                            folded_name in ordered_casefold_names
                            or folded_name in excluded_casefold_names
                        ):
                            continue
                        ordered_images.append((actual_name, image_path))
                        ordered_casefold_names.add(folded_name)

                    # 连续图号必须与最终页面展示顺序一致。稳定排序确保额外图片也会
                    # 归入所属阶段，而不会在控制冷却图片之后才显示较小图号。
                    stage_order = {
                        "加热阶段": 0,
                        "控制轧制阶段": 1,
                        "控制冷却阶段": 2,
                    }
                    ordered_images = [
                        image_item
                        for _, image_item in sorted(
                            enumerate(ordered_images),
                            key=lambda indexed_item: (
                                stage_order.get(
                                    pipeline_image_stage_map.get(
                                        indexed_item[1][0],
                                        "控制轧制阶段",
                                    ),
                                    1,
                                ),
                                indexed_item[0],
                            ),
                        )
                    ]

                    figure_number = 0
                    for image_name, image_path in ordered_images:
                        if not _os.path.isfile(image_path):
                            continue
                        image_token = _register_generated_image(image_path)
                        if not image_token:
                            continue
                        figure_number += 1
                        image_url = f"http://localhost:8000/generated-images/{image_token}"
                        image_stage = pipeline_image_stage_map.get(image_name, "控制轧制阶段")
                        image_caption = image_name.rsplit(".", 1)[0]
                        image_references_for_llm.append({
                            "name": image_name,
                            "figure_number": figure_number,
                            "description": image_name.replace(".png", ""),
                            "stage": image_stage,
                            "process_support": _pipeline_image_process_support_analysis(image_name),
                        })
                        process_support_analysis = _pipeline_image_process_support_analysis(image_name)
                        if is_wind_power_design:
                            process_support_analysis = process_support_analysis.replace(
                                "管线钢", "陆上风电塔筒用TMCP钢板"
                            )
                        image_markdown_block = (
                            f'![{image_name}]({image_url})\n\n'
                            f'<div class="report-figure-caption"><strong>'
                            f'图{figure_number}　{image_caption}'
                            f'</strong></div>\n\n'
                            f'　　**工艺支持分析：** {process_support_analysis}'
                        )
                        image_markdown_blocks.append(image_markdown_block)
                        image_markdown_blocks_by_stage[image_stage].append(image_markdown_block)
                    print(
                        f"[管线钢报告生成] 图片读取完成: 展示图片数={len(image_markdown_blocks)}, "
                        f"图片目录={resolved_pipeline_image_dir}"
                    )
                else:
                    print(f"[管线钢报告生成] 图片目录不存在: {pipeline_image_dir}")
            except Exception as exc:
                print(f"[管线钢报告生成] 读取仿真图片失败: {exc}")

            # 11. 将前置检索结果整理成报告可读的知识库上下文。
            report_rag_context = "\n\n---\n\n".join([
                f"[来源: {doc.get('source', 'unknown')}]\n{doc.get('content', '')}"
                for doc in pipeline_knowledge_docs
            ]) if pipeline_knowledge_docs else ""
            report_references_markdown = _build_rag_references_markdown(
                pipeline_knowledge_docs
            )
            report_date_text = _datetime.now().strftime("%Y年%m月%d日")
            report_signature_markdown = (
                '<div class="report-signature">'
                '<strong>Steel Multi-Agent System (SMAS)</strong><br>'
                f'{report_date_text}'
                '</div>'
            )
            final_report_session_context = _sanitize_pipeline_agent_reference_text(
                _build_cross_route_context(session_id)
                or _get_recent_session_context(session_id)
                or "（无最近会话上下文）",
                matched_result,
            )
            if is_wind_power_design:
                final_report_session_context = _filter_wind_power_session_context(
                    final_report_session_context
                )
            final_report_agent_judgements = _sanitize_pipeline_agent_reference_text(
                "\n\n".join(
                    f"### {stage_label}\n{judgement_text}"
                    for stage_label, judgement_text in pipeline_agent_judgement_contexts
                    if judgement_text
                ) or "（本轮工艺智能体未返回可用判断正文）",
                matched_result,
            )

            # 12. 身份追溯字段和数据库敏感字段禁止进入最终报告，避免暴露卷号/牌号/板坯号。
            sensitive_terms = [
                "strCoil", "strSteel",
                (reference_snapshot or {}).get("design_id"),
                (saved_design_snapshot or {}).get("design_id"),
                *SENSITIVE_MATCHED_RESULT_FIELDS,
                *SENSITIVE_MATCHED_RESULT_TOP_LEVEL_FIELDS,
                *_collect_sensitive_matched_terms(matched_result),
            ]
            sensitive_terms = [term for term in sensitive_terms if term]
            report_template_text = _load_hot_rolling_report_template()
            report_system_prompt = PIPELINE_REPORT_SYSTEM_PROMPT
            if is_wind_power_design:
                report_system_prompt = build_wind_power_report_system_prompt(
                    report_system_prompt, design_user_message
                )
            if report_template_text:
                report_system_prompt += (
                    REPORT_TEMPLATE_CONTEXT_PREFIX
                    + report_template_text
                    + REPORT_TEMPLATE_CONTEXT_SUFFIX
                )
            # 13. 最终报告 prompt 只传本轮权威 fact_table、规格边界、知识库与图片说明；
            #     不拼接历史会话和 matched_result 摘要，防止旧错误数值污染新报告。
            report_wind_standard_context = dict(wind_standard_context or {})
            if is_wind_power_design:
                calculated_pcm = _calculate_wind_power_pcm(matched_result)
                if calculated_pcm is not None:
                    # Pcm 是后端按照与校验相同的确定性公式得出的派生指标，
                    # 允许最终报告在用户明确提出焊接Pcm目标时如实展示与比较。
                    report_wind_standard_context["Pcm_calculated"] = round(calculated_pcm, 6)
            report_user_prompt = build_pipeline_report_user_prompt(
                design_user_message, final_report_session_context, final_report_agent_judgements,
                full_fact_table_reference, spec_result, rolling_schedule_markdown,
                report_rag_context, design_material_label, report_wind_standard_context,
                is_wind_power_design, image_references_for_llm,
            )

            # 14. 工艺智能体完成后，直接流式输出最终参数表和报告正文；不再进入额外风险评估链路。
            yield _ndjson_event(
                "design_preview_delta",
                content=_visible_reasoning_content(
                    "最终报告生成",
                    [
                        "加热、控制轧制和控制冷却智能体的结果已汇总为本轮权威事实表。",
                        "成分、工艺与性能参数表将与报告正文进入同一最终回答流连续输出。",
                        "报告正文将结合本地知识库资料、规格边界和仿真图片进行分析。",
                    ],
                ),
            )
            yield _ndjson_event("design_preview_done")

            async def final_report_generator():
                """连续转发最终报告，并在兼容的章节标题后插入对应阶段图片。"""
                history = PersistentChatMessageHistory(report_session_store, session_id)
                overview_title = "设计目标与方案概述"
                composition_title = "化学成分设计与作用"
                stage_titles = {
                    "加热工艺制度的解析": ("加热阶段", 3),
                    "控制轧制工艺制度的解析": ("控制轧制阶段", 4),
                    "控制冷却工艺制度的解析": ("控制冷却阶段", 5),
                }
                title_pattern = re.compile(
                    r"(?m)^(?:\s*#{1,6}\s*)?(?:\s*\*\*)?\s*"
                    r"(?:\d+(?:\.\d+)*[.、]\s*)?"
                    r"(设计目标与方案概述|化学成分设计与作用|加热工艺制度的解析|控制轧制工艺制度的解析|控制冷却工艺制度的解析)"
                    # 流式缓冲区末尾的章节标题可能暂时只有开头的 **。此处必须等到
                    # 闭合 ** 或完整换行到达后再匹配，避免把图片插入两个 ** 之间。
                    r"(?:\s*\*\*\s*(?:\r?\n|$)|\s*\r?\n)"
                )
                inserted_stages = set()
                pending = ""
                final_chunks = []
                document_started = False
                overview_emitted = False
                performance_table_emitted = False
                parameter_table_emitted = False
                image_summary = {
                    stage_name: [
                        caption_match.group(0)
                        for block in blocks
                        if (caption_match := re.search(r"图\d+　[^<\n]+", block))
                    ]
                    for stage_name, blocks in image_markdown_blocks_by_stage.items()
                }
                print(f"[管线钢最终报告] 各阶段待插入图片: {image_summary}")

                def sanitize_report_content(content: str) -> str:
                    safe_content = str(content or "")
                    for term in sensitive_terms:
                        safe_content = safe_content.replace(term, "")
                    return _sanitize_visible_text(safe_content)

                def emit_performance_table() -> str | None:
                    """在第一章首段之后插入 Word 模板中的表1力学性能标准。"""
                    nonlocal document_started, performance_table_emitted
                    if performance_table_emitted:
                        return None
                    performance_table_emitted = True
                    rendered_table = sanitize_report_content(performance_standard_markdown)
                    if not rendered_table:
                        return None
                    prefix = ""
                    if not document_started:
                        document_started = True
                        prefix = report_document_header
                        final_chunks.append(prefix)
                    final_chunks.append(rendered_table)
                    return prefix + rendered_table

                def emit_parameter_table() -> str | None:
                    """在第一章末尾插入 Word 模板中的表2成分、工艺与性能参数。"""
                    nonlocal document_started, parameter_table_emitted
                    if parameter_table_emitted:
                        return None
                    parameter_table_emitted = True
                    rendered_table = sanitize_report_content(report_table_markdown)
                    if not rendered_table:
                        return None
                    prefix = ""
                    if not document_started:
                        document_started = True
                        prefix = report_document_header
                        final_chunks.append(prefix)
                    final_chunks.append(rendered_table)
                    return prefix + rendered_table

                def emit_intro_tables() -> str | None:
                    """标题缺失等兜底场景下，补齐尚未插入的表1和表2。"""
                    rendered = []
                    performance_table = emit_performance_table()
                    if performance_table:
                        rendered.append(performance_table)
                    parameter_table = emit_parameter_table()
                    if parameter_table:
                        rendered.append(parameter_table)
                    return "".join(rendered) or None

                def emit_report_content(
                    content: str,
                    *,
                    allow_before_table: bool = False,
                ) -> str | None:
                    nonlocal document_started, overview_emitted
                    rendered = sanitize_report_content(content)
                    if not rendered:
                        return None
                    prefix = ""
                    if not document_started:
                        document_started = True
                        prefix = report_document_header
                        final_chunks.append(prefix)

                    final_chunks.append(rendered)
                    return prefix + rendered

                def is_possible_stage_title_prefix(fragment: str) -> bool:
                    """判断未换行尾部是否可能是第1章或三个工艺阶段标题的开头。"""
                    candidate = str(fragment or "").lstrip()
                    if not candidate:
                        return True

                    # 兼容模型输出 Markdown 标题（# / ##）或加粗标题（**）。
                    if candidate.startswith("#"):
                        heading_marks = len(candidate) - len(candidate.lstrip("#"))
                        if heading_marks > 6:
                            return False
                        candidate = candidate[heading_marks:].lstrip()
                        if not candidate:
                            return True
                    elif candidate.startswith("*"):
                        if candidate in {"*", "**"}:
                            return True
                        if not candidate.startswith("**"):
                            return False
                        candidate = candidate[2:].lstrip()
                        if not candidate:
                            return True

                    # 章节号可能与标题分属不同流式 chunk；数字前缀未完整时继续缓存。
                    if re.fullmatch(r"\d+(?:\.\d+)*[.、]?\s*", candidate):
                        return True
                    candidate = re.sub(r"^\d+(?:\.\d+)*[.、]\s*", "", candidate)
                    all_insertable_titles = (
                        overview_title,
                        composition_title,
                        *stage_titles.keys(),
                    )
                    return any(title.startswith(candidate) for title in all_insertable_titles)

                def drain_pending(force: bool = False) -> list[str]:
                    """识别跨 chunk 标题；普通正文立即转发，只缓存可能的标题前缀。"""
                    nonlocal pending, overview_emitted
                    emitted = []
                    while pending:
                        title_match = title_pattern.search(pending)
                        if title_match:
                            # 无论模型输出格式是否稳定，都规范为固定的编号二级标题。
                            title_text = title_match.group(1)
                            if title_text == overview_title:
                                title_markdown = pending[:title_match.start()]
                                if not overview_emitted:
                                    overview_emitted = True
                                    title_markdown += "## 1. 设计目标与方案概述\n\n"
                                pending = pending[title_match.end():]
                                rendered_title = emit_report_content(
                                    title_markdown,
                                    allow_before_table=True,
                                )
                                if rendered_title:
                                    emitted.append(rendered_title)
                                continue

                            if title_text == composition_title:
                                # 表1固定落在第一章首段之后；表2固定落在第一章末尾。
                                first_section_tail = pending[:title_match.start()]
                                pending = pending[title_match.end():]
                                if not performance_table_emitted:
                                    paragraph_end = re.search(
                                        r"\r?\n[ \t\u3000]*\r?\n",
                                        first_section_tail,
                                    )
                                    split_at = (
                                        paragraph_end.end()
                                        if paragraph_end
                                        else len(first_section_tail)
                                    )
                                    first_paragraph = first_section_tail[:split_at]
                                    remaining_overview = first_section_tail[split_at:]
                                    rendered_first_paragraph = emit_report_content(
                                        first_paragraph,
                                        allow_before_table=True,
                                    )
                                    if rendered_first_paragraph:
                                        emitted.append(rendered_first_paragraph)
                                    rendered_performance_table = emit_performance_table()
                                    if rendered_performance_table:
                                        emitted.append(rendered_performance_table)
                                    rendered_tail = emit_report_content(
                                        remaining_overview,
                                        allow_before_table=True,
                                    )
                                else:
                                    rendered_tail = emit_report_content(
                                        first_section_tail,
                                        allow_before_table=True,
                                    )
                                if rendered_tail:
                                    emitted.append(rendered_tail)
                                rendered_parameter_table = emit_parameter_table()
                                if rendered_parameter_table:
                                    emitted.append(rendered_parameter_table)
                                rendered_composition_title = emit_report_content(
                                    "## 2. 化学成分设计与作用\n\n",
                                    allow_before_table=True,
                                )
                                if rendered_composition_title:
                                    emitted.append(rendered_composition_title)
                                continue

                            stage_name, section_number = stage_titles[title_text]
                            # 模型漏写第2章时，在第一个工艺阶段标题前补齐首章两张表，
                            # 避免表格被拖到报告末尾或图片之后。
                            if not (
                                performance_table_emitted and parameter_table_emitted
                            ):
                                section_tail = pending[:title_match.start()]
                                pending = pending[title_match.start():]
                                rendered_tail = emit_report_content(
                                    section_tail,
                                    allow_before_table=True,
                                )
                                if rendered_tail:
                                    emitted.append(rendered_tail)
                                rendered_tables = emit_intro_tables()
                                if rendered_tables:
                                    emitted.append(rendered_tables)
                                continue
                            title_markdown = (
                                pending[:title_match.start()]
                                + f"## {section_number}. {title_text}\n\n"
                            )
                            pending = pending[title_match.end():]
                            rendered_title = emit_report_content(title_markdown)
                            if rendered_title:
                                emitted.append(rendered_title)

                            if stage_name not in inserted_stages:
                                inserted_stages.add(stage_name)
                                if stage_name == "控制轧制阶段" and rolling_schedule_markdown:
                                    rendered_schedule = emit_report_content(
                                        "\n" + rolling_schedule_markdown
                                    )
                                    if rendered_schedule:
                                        emitted.append(rendered_schedule)
                                blocks = image_markdown_blocks_by_stage.get(stage_name) or []
                                if blocks:
                                    rendered_images = emit_report_content(
                                        "\n" + "\n\n".join(blocks) + "\n\n"
                                    )
                                    if rendered_images:
                                        emitted.append(rendered_images)
                                else:
                                    print(f"[管线钢最终报告] {stage_name} 未收集到可插入图片")
                            continue

                        # Word 模板把表1放在第一章首个完整自然段之后。这里仅缓存
                        # 首段，收到空行后立即转发正文和表1，后续第一章正文继续流式输出。
                        if overview_emitted and not performance_table_emitted:
                            paragraph_end = re.search(
                                r"\r?\n[ \t\u3000]*\r?\n",
                                pending,
                            )
                            if paragraph_end:
                                first_paragraph = pending[:paragraph_end.end()]
                                pending = pending[paragraph_end.end():]
                                rendered_first_paragraph = emit_report_content(
                                    first_paragraph,
                                    allow_before_table=True,
                                )
                                if rendered_first_paragraph:
                                    emitted.append(rendered_first_paragraph)
                                rendered_performance_table = emit_performance_table()
                                if rendered_performance_table:
                                    emitted.append(rendered_performance_table)
                                continue

                        if force:
                            rendered_tail = emit_report_content(pending)
                            pending = ""
                            if rendered_tail:
                                emitted.append(rendered_tail)
                        else:
                            # 完整行不可能再与后续 token 拼成标题，可立即输出；只保留
                            # 最后一个未完成行，继续判断它是否是阶段标题的流式前缀。
                            newline_index = pending.rfind("\n")
                            if newline_index >= 0:
                                flush_end = newline_index + 1
                                rendered_prefix = emit_report_content(pending[:flush_end])
                                pending = pending[flush_end:]
                                if rendered_prefix:
                                    emitted.append(rendered_prefix)
                                continue

                            if is_possible_stage_title_prefix(pending):
                                break

                            rendered_prefix = emit_report_content(pending)
                            pending = ""
                            if rendered_prefix:
                                emitted.append(rendered_prefix)
                    return emitted

                try:
                    async for chunk in deepseek_Llm.astream([
                        SystemMessage(content=report_system_prompt),
                        HumanMessage(content=report_user_prompt),
                    ]):
                        if not chunk.content:
                            continue
                        pending += str(chunk.content)
                        for rendered_piece in drain_pending():
                            yield json.dumps({"content": rendered_piece}, ensure_ascii=False) + "\n"

                    for rendered_piece in drain_pending(force=True):
                        yield json.dumps({"content": rendered_piece}, ensure_ascii=False) + "\n"

                    # 极端情况下模型没有输出第2章及三个工艺标题，仍补出两张权威表格；
                    # 正常固定七章报告会在第2章标题出现时更早完成插入。
                    rendered_intro_tables = emit_intro_tables()
                    if rendered_intro_tables:
                        yield json.dumps(
                            {"content": rendered_intro_tables},
                            ensure_ascii=False,
                        ) + "\n"

                    missing_stages = [
                        stage_name
                        for stage_name, _ in stage_titles.values()
                        if stage_name not in inserted_stages
                    ]
                    for stage_name in missing_stages:
                        blocks = image_markdown_blocks_by_stage.get(stage_name) or []
                        schedule_block = (
                            rolling_schedule_markdown
                            if stage_name == "控制轧制阶段"
                            else ""
                        )
                        if not blocks and not schedule_block:
                            print(f"[管线钢最终报告] 缺失章节 {stage_name} 且无可用图片")
                            continue
                        print(f"[管线钢最终报告] 未识别 {stage_name} 章节标题，追加该阶段图片")
                        fallback_section_number = next(
                            section_number
                            for mapped_stage, section_number in stage_titles.values()
                            if mapped_stage == stage_name
                        )
                        fallback_section_title = next(
                            title
                            for title, (mapped_stage, _) in stage_titles.items()
                            if mapped_stage == stage_name
                        )
                        fallback_images = (
                            f"\n\n## {fallback_section_number}. {fallback_section_title}\n\n"
                            + schedule_block
                            + ("\n\n".join(blocks) if blocks else "")
                            + "\n\n"
                        )
                        rendered_images = emit_report_content(fallback_images)
                        if rendered_images:
                            yield json.dumps({"content": rendered_images}, ensure_ascii=False) + "\n"

                    # 参考文献必须位于模型正文和所有兜底补图之后，并且只采用
                    # 本轮 RAG 实际召回的文献名，避免模型补造或泄露本地文件路径。
                    rendered_references = emit_report_content(
                        "\n\n" + report_references_markdown
                    )
                    if rendered_references:
                        yield json.dumps(
                            {"content": rendered_references},
                            ensure_ascii=False,
                        ) + "\n"

                    # 落款始终位于参考文献之后，由后端生成软件名称和当天日期，
                    # 避免模型漏写、重复或输出不符合 yyyy年MM月dd日 的日期。
                    rendered_signature = emit_report_content(
                        "\n\n" + report_signature_markdown
                    )
                    if rendered_signature:
                        yield json.dumps(
                            {"content": rendered_signature},
                            ensure_ascii=False,
                        ) + "\n"

                    final_report_text = "".join(final_chunks).strip()
                    if not final_report_text:
                        fallback = "## 管线钢工艺分析报告\n\n> 最终报告正文生成失败。"
                        rendered_fallback = emit_report_content(fallback)
                        if rendered_fallback:
                            yield json.dumps({"content": rendered_fallback}, ensure_ascii=False) + "\n"
                        final_report_text = fallback
                    print(
                        "[管线钢最终报告] 阶段图片插入完成: "
                        f"已插入={sorted(inserted_stages)}, 缺失标题补图={missing_stages}"
                    )
                    history.add_message(HumanMessage(content=build_report_history_user_prompt(
                        original_user_message, full_fact_table_reference
                    )))
                    history.add_message(AIMessage(content=final_report_text))
                except Exception as exc:
                    print(f"[管线钢最终报告] 流式生成失败: {exc}")
                    fallback = "\n\n> 最终报告生成失败，请稍后重试。"
                    rendered_fallback = emit_report_content(fallback)
                    if rendered_fallback:
                        yield json.dumps({"content": rendered_fallback}, ensure_ascii=False) + "\n"
                finally:
                    _remember_agent_context_turn(
                        session_id,
                        original_user_message,
                        "已完成管线钢设计和最终报告输出。",
                    )

            async for chunk in _relay_streaming_response(
                _wrap_event_stream(
                    StreamingResponse(
                        final_report_generator(),
                        media_type="application/x-ndjson",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    ),
                    [],
                )
            ):
                yield chunk
            return

    computation_state = _start_or_get_frontend_computation_task(
        session_id,
        original_user_message,
        lambda: _stream_with_heartbeat(classify_stream()),
    )
    return StreamingResponse(
        _subscribe_frontend_computation_task(computation_state),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Computation-Session": session_id,
        },
    )


@app.get("/computation-status/{session_id}")
async def get_computation_status(session_id: str):
    """供前端每5秒查询当前会话是否仍有后台计算。"""
    _cleanup_frontend_computation_tasks()
    state = _FRONTEND_COMPUTATION_TASKS.get(session_id)
    if state is None:
        return {
            "exists": False,
            "status": "idle",
            "event_count": 0,
        }
    return {
        "exists": True,
        "status": state.status,
        "event_count": len(state.events),
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "completed_at": state.completed_at,
        "error": state.error,
    }


@app.get("/computation-stream/{session_id}")
async def resume_computation_stream(session_id: str, from_event: int = 0):
    """重放并继续订阅当前会话的后台计算结果。"""
    state = _FRONTEND_COMPUTATION_TASKS.get(session_id)
    if state is None:
        return JSONResponse(
            {"error": "当前会话没有可恢复的后台计算任务"},
            status_code=404,
        )
    return StreamingResponse(
        _subscribe_frontend_computation_task(state, from_event),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Computation-Session": session_id,
        },
    )

# ============================================================
# 调试端点（可选）：查看会话状态
# ============================================================
@app.get("/sessions")
async def list_sessions():
    """
    GET /sessions — 查看当前活跃会话数量（调试用）
    List active sessions count (for debugging)

    现在从持久化存储查询（数据库 + 内存缓存），而非仅内存 dict。
    """
    chat_info = chat_session_store.get_active_sessions_info()
    agent_info = agent_chat_store.get_active_sessions_info()
    report_info = report_session_store.get_active_sessions_info()

    # 合并多种类型的会话信息
    all_sessions = {}
    for sid, info in chat_info.items():
        all_sessions[f"chat:{sid[:8]}"] = info
    for sid, info in agent_info.items():
        all_sessions[f"agent:{sid[:8]}"] = info
    for sid, info in report_info.items():
        all_sessions[f"report:{sid[:8]}"] = info

    return {
        "active_sessions": len(all_sessions),
        "sessions": all_sessions,
    }


# ============================================================
# 应用入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    import os as _os
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except OSError as e:
        if "10048" in str(e) or "bind" in str(e).lower():
            print(f"\n[错误] 端口 8000 已被占用。")
            print("请先在终端执行以下命令释放端口后重试：")
            print("  powershell -c \"Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id `$_.OwningProcess -Force }\"")
            _os._exit(0)
        raise
