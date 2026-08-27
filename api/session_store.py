"""
session_store.py — 会话持久化存储模块
======================================

提供 SessionStore 类，将对话历史持久化到 PostgreSQL 数据库，
支持服务器重启后自动恢复会话。采用双层架构：

  - 内存缓存层：快速读写（与原内存 dict 行为一致）
  - 数据库持久层：服务器重启后数据不丢失

写穿策略（Write-Through）：每次写操作同步写入内存缓存和数据库。
读策略：优先查内存缓存，miss 时查数据库（懒加载）。

使用方式:
    from session_store import SessionStore, PersistentChatMessageHistory, init_session_db

    # FastAPI startup 时调用一次
    init_session_db()

    # 创建存储实例
    store = SessionStore(session_type="chat", max_turns=50, ttl=3600)

    # 获取或创建会话
    session = store.get_or_create(session_id)
    messages = store.get_messages(session_id)

    # 追加消息
    store.add_message(session_id, HumanMessage(content="你好"))

    # 适配 LangChain RunnableWithMessageHistory
    history = PersistentChatMessageHistory(store, session_id)
"""

import json
import time
import threading
import asyncio
import os
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

import psycopg2
from psycopg2 import sql
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from psycopg2.errors import DuplicateDatabase, InvalidCatalogName

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ============================================================
# 数据库连接配置 — 复用 hybrid_retriever.py 中的参数
# ============================================================
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_NAME = os.getenv("SESSION_DB_NAME", "metal_sessions_db")

# ============================================================
# 连接池管理（单例 + 双重检查锁）
# ============================================================
_pool: Optional[ThreadedConnectionPool] = None
_pool_lock = threading.Lock()

# 当数据库不可用时设置为 True，SessionStore 回退到纯内存模式
_db_unavailable = False


def _get_pool() -> Optional[ThreadedConnectionPool]:
    """获取全局连接池（懒初始化，线程安全）"""
    global _pool, _db_unavailable
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool
        if _db_unavailable:
            return None
        try:
            _pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=5,
            )
            print(f"[session_store] 连接池已创建 (min=2, max=10) → {DB_HOST}:{DB_PORT}/{DB_NAME}")
        except Exception as e:
            print(f"[session_store] 连接池创建失败: {e}")
            print("[session_store] 将回退到纯内存模式（重启后会话丢失）")
            _db_unavailable = True
            _pool = None
        return _pool


def _put_conn(conn):
    """安全归还连接到池"""
    global _pool
    if _pool is not None and conn is not None:
        try:
            _pool.putconn(conn)
        except Exception:
            pass


def _close_all_connections():
    """关闭连接池中所有连接"""
    global _pool
    if _pool is not None:
        try:
            _pool.closeall()
        except Exception:
            pass
        _pool = None


# ============================================================
# 数据库和表初始化
# ============================================================

def _execute_sql(sql_str: str, params: tuple = None, autocommit: bool = False):
    """执行一条 SQL，返回结果。用于管理与 psycopg2 连接的完整生命周期。"""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname="postgres",  # 先连默认库
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=5,
    )
    try:
        conn.autocommit = autocommit
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_str, params)
            if cur.description:
                return cur.fetchall()
            return None
    finally:
        conn.close()


def init_session_db():
    """
    幂等初始化会话持久化数据库和表。

    步骤:
    1. 连接到 postgres 默认数据库
    2. 检查/创建 metal_sessions_db 数据库
    3. 连接到 metal_sessions_db，创建 persistent_sessions 表
    4. 初始化连接池

    在 FastAPI startup 事件中调用。
    """
    global _db_unavailable

    # ---- Step 1: 创建数据库（如果不存在） ----
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=5,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            # 检查数据库是否存在
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (DB_NAME,),
            )
            if cur.fetchone() is None:
                cur.execute(sql.SQL("CREATE DATABASE {} WITH ENCODING 'UTF8'").format(
                    sql.Identifier(DB_NAME)
                ))
                print(f"[session_store] 数据库 '{DB_NAME}' 已创建")
            else:
                print(f"[session_store] 数据库 '{DB_NAME}' 已存在")
        conn.close()
    except Exception as e:
        print(f"[session_store] 数据库创建检查失败: {e}")
        _db_unavailable = True
        return

    # ---- Step 2: 创建表（如果不存在） ----
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=5,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS persistent_sessions (
                    id              SERIAL PRIMARY KEY,
                    session_id      VARCHAR(255) NOT NULL,
                    session_type    VARCHAR(64)  NOT NULL DEFAULT 'chat',
                    messages        JSONB        NOT NULL DEFAULT '[]'::jsonb,
                    last_active     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    CONSTRAINT uq_session_id_type UNIQUE (session_id, session_type)
                );
            """)
            # 索引（幂等创建）
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ps_last_active
                    ON persistent_sessions(last_active);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ps_sid_type
                    ON persistent_sessions(session_id, session_type);
            """)
        conn.close()
        print(f"[session_store] 表 'persistent_sessions' 已就绪")
    except Exception as e:
        print(f"[session_store] 表创建失败: {e}")
        _db_unavailable = True
        return

    # ---- Step 3: 初始化连接池 ----
    _get_pool()
    print(f"[session_store] 会话持久化数据库已就绪 → {DB_HOST}:{DB_PORT}/{DB_NAME}")


# ============================================================
# 序列化 / 反序列化
# ============================================================

def serialize_messages(messages: list) -> list[dict]:
    """
    将 LangChain 消息列表转为 JSON-safe 的 dict 列表。

    支持 HumanMessage (type='human') 和 AIMessage (type='ai')。
    其他类型会以 type='unknown' 保留。

    Args:
        messages: LangChain BaseMessage 列表

    Returns:
        list[dict]: 每项包含 {"type": str, "content": str}
    """
    result = []
    for msg in messages:
        msg_type = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        # 确保 content 是字符串
        if isinstance(content, (list, tuple)):
            content = "".join(str(c) for c in content)
        elif not isinstance(content, str):
            content = str(content)
        result.append({"type": msg_type, "content": content})
    return result


def deserialize_messages(data: list) -> list:
    """
    将 JSON 数据反序列化为 LangChain 消息列表。

    按 type 字段重建消息对象：
      - "human" → HumanMessage
      - "ai"    → AIMessage

    Args:
        data: list[dict]，每项包含 {"type": str, "content": str}

    Returns:
        list[BaseMessage]: HumanMessage 或 AIMessage 实例
    """
    type_map = {
        "human": HumanMessage,
        "ai": AIMessage,
    }
    result = []
    for item in data:
        msg_type = item.get("type", "human")
        content = item.get("content", "")
        constructor = type_map.get(msg_type, HumanMessage)
        result.append(constructor(content=content))
    return result


# ============================================================
# SessionStore — 核心持久化类
# ============================================================

class SessionStore:
    """
    会话持久化存储封装。

    双层架构:
      - 内存缓存层：快速读写（与原内存 dict 行为一致）
      - 数据库持久层：服务器重启后数据不丢失

    写穿策略（Write-Through）：每次写操作同步写入内存缓存和数据库。
    读策略：优先查内存缓存，miss 时查数据库（懒加载）。
    容错：数据库不可用时自动回退到纯内存模式。

    线程安全：所有公共方法通过内部锁保护。
    """

    def __init__(
        self,
        session_type: str,
        max_turns: int = 50,
        ttl: float = 3600.0,
    ):
        """
        Args:
            session_type: 会话类型标识
                          'chat' | 'agent_chat' | 'intent_classifier' | 'spec_extractor'
            max_turns:    最大保留轮数（每轮 = 1用户 + 1AI = 2条消息）
            ttl:          会话过期时间（秒），超过此时间未活动的会话将被清理
        """
        self._session_type = session_type
        self._max_turns = max_turns
        self._ttl = ttl
        self._cache: dict[str, dict] = {}  # {session_id: {messages: [...], last_active: float}}
        self._lock = threading.RLock()

    # ---- 内部方法 ----

    def _db_load(self, session_id: str) -> Optional[dict]:
        """从数据库加载会话。返回 {messages: [...], last_active: float} 或 None"""
        pool = _get_pool()
        if pool is None:
            return None
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT messages, last_active FROM persistent_sessions "
                    "WHERE session_id = %s AND session_type = %s",
                    (session_id, self._session_type),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "messages": deserialize_messages(row["messages"]),
                        "last_active": row["last_active"],
                    }
                return None
        except Exception as e:
            print(f"[session_store] DB加载失败 ({self._session_type}/{session_id[:8]}...): {e}")
            return None
        finally:
            _put_conn(conn)

    def _db_save(self, session_id: str, messages: list, last_active: float):
        """保存会话到数据库（UPSERT）"""
        pool = _get_pool()
        if pool is None:
            return
        conn = None
        try:
            serialized = json.dumps(serialize_messages(messages), ensure_ascii=False)
            conn = pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO persistent_sessions (session_id, session_type, messages, last_active, updated_at)
                    VALUES (%s, %s, %s::jsonb, %s, NOW())
                    ON CONFLICT (session_id, session_type)
                    DO UPDATE SET messages = EXCLUDED.messages,
                                  last_active = EXCLUDED.last_active,
                                  updated_at = NOW()
                    """,
                    (session_id, self._session_type, serialized, last_active),
                )
            conn.commit()
        except Exception as e:
            print(f"[session_store] DB保存失败 ({self._session_type}/{session_id[:8]}...): {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
        finally:
            _put_conn(conn)

    def _db_delete(self, session_id: str):
        """从数据库删除指定会话"""
        pool = _get_pool()
        if pool is None:
            return
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM persistent_sessions WHERE session_id = %s AND session_type = %s",
                    (session_id, self._session_type),
                )
            conn.commit()
        except Exception as e:
            print(f"[session_store] DB删除失败 ({self._session_type}/{session_id[:8]}...): {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
        finally:
            _put_conn(conn)

    def _db_cleanup_expired(self, now: float) -> int:
        """数据库层面清理过期会话，返回清理数量"""
        pool = _get_pool()
        if pool is None:
            return 0
        conn = None
        try:
            conn = pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM persistent_sessions WHERE session_type = %s AND last_active < %s",
                    (self._session_type, now - self._ttl),
                )
                deleted = cur.rowcount
            conn.commit()
            if deleted > 0:
                print(f"[session_store] DB清理了 {deleted} 条过期会话 (type={self._session_type})")
            return deleted
        except Exception as e:
            print(f"[session_store] DB清理失败: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return 0
        finally:
            _put_conn(conn)

    def _db_get_all_active(self) -> dict:
        """从数据库获取所有活跃会话信息（用于 /sessions 调试端点）"""
        pool = _get_pool()
        if pool is None:
            return {}
        conn = None
        try:
            conn = pool.getconn()
            now = time.time()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT session_id, messages, last_active FROM persistent_sessions "
                    "WHERE session_type = %s AND last_active >= %s",
                    (self._session_type, now - self._ttl),
                )
                rows = cur.fetchall()
                result = {}
                for row in rows:
                    msgs = deserialize_messages(row["messages"])
                    result[row["session_id"]] = {
                        "messages": msgs,
                        "last_active": row["last_active"],
                    }
                return result
        except Exception as e:
            print(f"[session_store] DB查询活跃会话失败: {e}")
            return {}
        finally:
            _put_conn(conn)

    # ---- 公共方法 ----

    def get_or_create(self, session_id: str) -> dict:
        """
        获取或创建会话。

        先查内存缓存，再查数据库，都不存在则创建新会话。
        返回格式: {"messages": [...], "last_active": float}
        """
        with self._lock:
            # 被动清理过期缓存
            self._cleanup_cache_expired()

            # 1. 先查缓存
            if session_id in self._cache:
                self._cache[session_id]["last_active"] = time.time()
                return self._cache[session_id]

            # 2. 缓存 miss → 查数据库（懒加载）
            db_data = self._db_load(session_id)
            if db_data is not None:
                self._cache[session_id] = db_data
                print(f"[session_store] 从DB加载会话 (type={self._session_type}, sid={session_id[:8]}..., "
                      f"共{len(db_data['messages'])}条消息)")
                return self._cache[session_id]

            # 3. 都不存在 → 创建新会话
            new_session = {
                "messages": [],
                "last_active": time.time(),
            }
            self._cache[session_id] = new_session
            print(f"[session_store] 新会话 (type={self._session_type}, sid={session_id[:8]}...)")
            return new_session

    def get_messages(self, session_id: str) -> list:
        """
        获取会话的 LangChain 消息列表（已裁剪到 max_turns）。

        Args:
            session_id: 会话唯一标识

        Returns:
            list[BaseMessage]: HumanMessage/AIMessage 列表
        """
        session = self.get_or_create(session_id)
        return self._trim(session["messages"])

    def add_message(self, session_id: str, message) -> None:
        """
        追加一条消息到会话，自动裁剪并持久化。

        对应 InMemoryChatMessageHistory.add_message() 的接口。

        Args:
            session_id: 会话唯一标识
            message:    LangChain BaseMessage (HumanMessage 或 AIMessage)
        """
        with self._lock:
            session = self.get_or_create(session_id)
            session["messages"].append(message)
            session["messages"] = self._trim(session["messages"])
            session["last_active"] = time.time()
            # 写穿到数据库
            self._db_save(session_id, session["messages"], session["last_active"])

    def add_messages_batch(self, session_id: str, messages: list) -> None:
        """
        批量追加多条消息，最后一次性裁剪和持久化。

        Args:
            session_id: 会话唯一标识
            messages:   LangChain BaseMessage 列表
        """
        with self._lock:
            session = self.get_or_create(session_id)
            session["messages"].extend(messages)
            session["messages"] = self._trim(session["messages"])
            session["last_active"] = time.time()
            self._db_save(session_id, session["messages"], session["last_active"])

    def save(self, session_id: str, messages: list):
        """
        直接保存消息列表（覆盖模式），自动裁剪。

        Args:
            session_id: 会话唯一标识
            messages:   LangChain 消息列表
        """
        with self._lock:
            session = self.get_or_create(session_id)
            session["messages"] = self._trim(messages)
            session["last_active"] = time.time()
            self._db_save(session_id, session["messages"], session["last_active"])

    def clear(self, session_id: str):
        """
        清除指定会话（内存 + 数据库）。

        Args:
            session_id: 会话唯一标识
        """
        with self._lock:
            self._cache.pop(session_id, None)
            self._db_delete(session_id)

    def clear_all(self):
        """清除所有此类型的会话（仅内存缓存，数据库通过 TTL 自动过期）"""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        清理过期会话（内存缓存 + 数据库），返回清理数量。

        在每次 get_or_create() 调用时被动触发，
        也可手动调用进行主动清理。
        """
        with self._lock:
            cache_count = self._cleanup_cache_expired()
            db_count = self._db_cleanup_expired(time.time())
            return cache_count + db_count

    def get_active_sessions_info(self) -> dict:
        """
        获取所有活跃会话信息（合并缓存和数据库）。

        用于 /sessions 调试端点。
        Returns:
            dict: {session_id: {message_count, last_active, preview}}
        """
        with self._lock:
            self._cleanup_cache_expired()
            # 合并缓存和数据库结果
            all_sessions = self._db_get_all_active()
            # 缓存中的会话可能比 DB 更新
            for sid, s in self._cache.items():
                all_sessions[sid] = s

            return {
                sid: {
                    "message_count": len(s["messages"]),
                    "last_active": s["last_active"],
                    "preview": (
                        s["messages"][0].content[:50] + "..."
                        if s["messages"] else "(空)"
                    ),
                }
                for sid, s in all_sessions.items()
            }

    def _cleanup_cache_expired(self) -> int:
        """清理内存缓存中过期的会话，返回清理数量"""
        now = time.time()
        expired = [
            sid for sid, s in self._cache.items()
            if now - s["last_active"] > self._ttl
        ]
        for sid in expired:
            del self._cache[sid]
        if expired:
            print(f"[session_store] 缓存清理了 {len(expired)} 个过期会话 (type={self._session_type})")
        return len(expired)

    def _trim(self, messages: list) -> list:
        """
        裁剪消息列表到 max_turns 轮。

        每轮 = 1条 HumanMessage + 1条 AIMessage。
        保留最近的消息。

        Args:
            messages: LangChain 消息列表

        Returns:
            裁剪后的消息列表（新列表，不修改原列表）
        """
        max_msgs = self._max_turns * 2
        if len(messages) > max_msgs:
            return messages[-max_msgs:]
        return messages


# ============================================================
# PersistentChatMessageHistory — LangChain 适配器
# ============================================================

class PersistentChatMessageHistory(BaseChatMessageHistory):
    """
    实现 LangChain BaseChatMessageHistory 接口，
    底层使用 SessionStore 持久化。

    用于 RunnableWithMessageHistory 的 get_session_history 回调。
    兼容 intent_classifier.py 和 steel_spec_extractor.py 中的用法。

    使用示例:
        store = SessionStore("intent_classifier")
        history = PersistentChatMessageHistory(store, session_id)
        history.add_message(HumanMessage(content="..."))
        print(history.messages)  # 返回裁剪后的消息列表
    """

    def __init__(self, store: SessionStore, session_id: str):
        self._store = store
        self._session_id = session_id

    @property
    def messages(self) -> list:
        """返回会话中已裁剪的消息列表"""
        return self._store.get_messages(self._session_id)

    def add_message(self, message) -> None:
        """追加一条消息并持久化"""
        self._store.add_message(self._session_id, message)

    def clear(self) -> None:
        """清除此会话"""
        self._store.clear(self._session_id)

    def __repr__(self) -> str:
        return (
            f"PersistentChatMessageHistory("
            f"type={self._store._session_type}, "
            f"sid={self._session_id[:8]}...)"
        )


# ============================================================
# 后台定时清理任务
# ============================================================

_db_cleanup_stores: list[SessionStore] = []


def register_for_cleanup(store: SessionStore):
    """注册 SessionStore 实例以参与后台定时清理"""
    global _db_cleanup_stores
    if store not in _db_cleanup_stores:
        _db_cleanup_stores.append(store)


async def _periodic_db_cleanup(interval_seconds: int = 1800):
    """
    后台协程：每30分钟清理一次数据库中过期的会话。

    在 FastAPI startup 事件中通过 asyncio.create_task 启动。
    """
    while True:
        await asyncio.sleep(interval_seconds)
        total = 0
        for store in _db_cleanup_stores:
            total += store.cleanup_expired()
        if total > 0:
            print(f"[session_store] 后台定时清理完成，共清理 {total} 条过期记录")


def db_cleanup_all_types():
    """手动清理所有已注册 SessionStore 的过期会话（用于独立脚本/手动运维）"""
    total = 0
    for store in _db_cleanup_stores:
        total += store.cleanup_expired()
    print(f"[session_store] 手动清理完成，共清理 {total} 条过期记录")
    return total
