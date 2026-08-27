#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索脚本
===========
结合向量相似度检索（PGVector）与关键词检索（BM25），
使用 Reciprocal Rank Fusion (RRF) 算法进行加权混合排序。

实现原理:
    1. 向量检索器 — 基于 PGVector 存储的文档嵌入向量，
       使用余弦相似度查找语义相近的文档块。
    2. BM25 检索器 — 基于词频-逆文档频率的经典信息检索算法，
       从本地 JSON 缓存中重建文档块进行关键词匹配。
    3. RRF 融合 — 对两种检索器的排序结果进行加权融合，
       score = Σ weight_i / (k + rank_i)
       其中 k=60 是平滑常数，避免高排名结果过度主导。

依赖安装:
    pip install psycopg2-binary pgvector "sentence-transformers>=2.2.0" langchain langchain-community

使用前请先运行 store_vectors.py 生成向量数据和 chunks_cache.json。

运行方式:
    python hybrid_retriever.py "人工智能的发展趋势"
    python hybrid_retriever.py                        # 使用默认测试查询
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

# ============================================================
# 【关键】在所有 HuggingFace 相关 import 之前，强制拦截 requests 超时
# 解决 HuggingFace 下载模型时网络不可达导致无限阻塞的问题
# ============================================================
from network_timeout import enforce_timeout
enforce_timeout(timeout=5)

# 设置 stdout 为 UTF-8，避免 Windows GBK 控制台编码错误
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---- 必须在所有 HuggingFace 相关 import 之前设置 ----
# 强制离线模式：只从本地缓存加载模型，不尝试联网下载
# （如首次使用需先手动下载模型放到缓存目录，或临时注释这两行）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 使用国内镜像下载模型 (解决 huggingface.co 不可达问题)
# 如能直接访问 huggingface.co，注释掉下面两行即可
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 设置 HuggingFace 下载超时（秒）
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")
# 设置全局 socket 超时
import socket
socket.setdefaulttimeout(5)

# LangChain 组件
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# ============================================================
# 配置常量
# ============================================================

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = "gcjxyg_db"
# DB_NAME = "gqdujgg_db"
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_COLLECTION = "documents"

CACHE_FILE = Path(__file__).parent / f"chunks_cache_{DB_NAME}_{DB_COLLECTION}.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 混合检索权重 (可调整)
VECTOR_WEIGHT = 0.5   # 向量语义检索权重
BM25_WEIGHT = 0.5     # BM25 关键词检索权重

# RRF 平滑常数 k (值越大，排名差异影响越小)
RRF_K = 60

# 是否启用向量检索（需要联网下载 HuggingFace 模型，超时拦截模块会保证网络不可达时快速失败）
USE_VECTOR = True

# 默认返回结果数
DEFAULT_K = 5

CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================================
# 文档加载与缓存
# ============================================================

def _build_cache_path(db_name: str = None, db_collection: str = None) -> Path:
    """根据数据库名和集合名构建缓存文件路径"""
    name = db_name or DB_NAME
    coll = db_collection or DB_COLLECTION
    return Path(__file__).parent / f"chunks_cache_{name}_{coll}.json"


def _build_connection_string(db_name: str = None) -> str:
    """根据数据库名构建 PostgreSQL 连接字符串"""
    name = db_name or DB_NAME
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{name}"


def load_chunks_from_cache(cache_file: Path = None) -> list[Document]:
    """
    从 JSON 缓存文件加载文档块，重建 LangChain Document 对象。

    Args:
        cache_file: 缓存文件路径，默认使用模块常量 CACHE_FILE

    Returns:
        list[Document]: 文档块列表

    Raises:
        FileNotFoundError: 缓存文件不存在时抛出
    """
    cf = cache_file or CACHE_FILE
    if not cf.exists():
        raise FileNotFoundError(
            f"缓存文件 {cf} 不存在。\n"
            f"请先运行 store_vectors.py 生成向量数据和文本缓存。"
        )

    with open(cf, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    documents = []
    for item in cache_data:
        doc = Document(
            page_content=item["content"],
            metadata=item.get("metadata", {}),
        )
        documents.append(doc)

    print(f"[INFO] 从缓存加载 {len(documents)} 个文档块")
    return documents


# ============================================================
# 检索器构建
# ============================================================

def build_vector_retriever(connection_string: str = None, collection_name: str = None):
    """
    构建基于 PGVector 的向量检索器。

    Args:
        connection_string: PostgreSQL 连接字符串，默认使用模块常量 CONNECTION_STRING
        collection_name:   集合名称，默认使用模块常量 DB_COLLECTION

    Returns:
        tuple: (PGVector store or None, HuggingFaceEmbeddings or None)
               向量存储不可用时返回 (None, None)
    """
    cs = connection_string or CONNECTION_STRING
    cn = collection_name or DB_COLLECTION
    print(f"[INFO] 连接 PostgreSQL PGVector: {DB_HOST}:{DB_PORT}/{cs.split('/')[-1]}")
    try:
        # 初始化嵌入模型（需要联网下载，可能超时）
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # 初始化向量存储
        vector_store = PGVector(
            connection_string=cs,
            embedding_function=embeddings,
            collection_name=cn,
        )
        print("[INFO] 向量检索器已就绪 (PGVector)")
        return vector_store, embeddings
    except Exception as e:
        print(f"[警告] 向量检索器初始化失败: {e}")
        print("[警告] 将仅使用 BM25 关键词检索")
        return None, None


def build_bm25_retriever(documents: list[Document], k: int = DEFAULT_K) -> BM25Retriever:
    """
    基于文档块列表构建 BM25 关键词检索器。

    Args:
        documents: LangChain Document 列表
        k:         BM25 候选返回数量

    Returns:
        BM25Retriever
    """
    retriever = BM25Retriever.from_documents(documents, k=k)
    print(f"[INFO] BM25 检索器已就绪 ({len(documents)} 个文档, k={k})")
    return retriever


# ============================================================
# RRF 混合检索算法
# ============================================================

def reciprocal_rank_fusion(
    vector_results: list[tuple[Document, float]],
    bm25_results: list[Document],
    k: int = DEFAULT_K,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    使用加权 Reciprocal Rank Fusion 算法融合两路检索结果。

    公式: RRF_score(d) = Σ weight_i / (rrf_k + rank_i(d))

    Args:
        vector_results: 向量检索结果，每项为 (Document, similarity_score)
        bm25_results:   BM25 检索结果，每项为 Document (score 在 metadata 中)
        k:              最终返回的结果数量
        vector_weight:  向量检索的 RRF 权重
        bm25_weight:    BM25 检索的 RRF 权重
        rrf_k:          RRF 平滑常数

    Returns:
        list[dict]: 排序后的结果，每项包含:
            - content:      完整文本内容
            - score:        RRF 融合分数
            - source:       来源文件名
            - content_preview: 文本预览 (前 200 字)
            - metadata:     原始元数据
            - vector_score: 向量相似度分数
            - bm25_score:   BM25 分数
    """
    # 以 doc_id 为键聚合分数
    scores: dict[str, dict] = {}

    # --- 向量检索结果 (附带余弦相似度分数) ---
    for rank, (doc, similarity) in enumerate(vector_results, start=1):
        doc_id = doc.metadata.get("doc_id", f"vec_{rank}")
        if doc_id not in scores:
            scores[doc_id] = {
                "doc": doc,
                "rrf": 0.0,
                "vector_score": similarity,
                "bm25_score": None,
            }
        scores[doc_id]["rrf"] += vector_weight / (rrf_k + rank)

    # --- BM25 检索结果 ---
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.metadata.get("doc_id", f"bm25_{rank}")
        bm25_score = doc.metadata.get("score", None)
        if doc_id not in scores:
            scores[doc_id] = {
                "doc": doc,
                "rrf": 0.0,
                "vector_score": None,
                "bm25_score": bm25_score,
            }
        else:
            # BM25 docs come from the latest local chunks cache, so prefer their
            # content when vector DB records are stale but doc_id is unchanged.
            scores[doc_id]["doc"] = doc
            scores[doc_id]["bm25_score"] = bm25_score
        scores[doc_id]["rrf"] += bm25_weight / (rrf_k + rank)

    # --- 按 RRF 分数降序排序 ---
    sorted_items = sorted(
        scores.items(), key=lambda x: x[1]["rrf"], reverse=True
    )

    # --- 格式化输出 ---
    results = []
    for rank, (doc_id, data) in enumerate(sorted_items[:k], start=1):
        doc = data["doc"]
        source = doc.metadata.get("source", "unknown")
        content_preview = doc.page_content[:200].replace("\n", " ").replace("﻿", "").strip()

        results.append({
            "rank": rank,
            "score": round(data["rrf"], 6),
            "source": source,
            "content_preview": content_preview,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "vector_score": (
                round(data["vector_score"], 6)
                if data["vector_score"] is not None
                else None
            ),
            "bm25_score": (
                round(data["bm25_score"], 6)
                if data["bm25_score"] is not None
                else None
            ),
        })

    return results


# ============================================================
# 公开 API
# ============================================================

def hybrid_search(
    query: str,
    k: int = DEFAULT_K,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    db_name: str = None,
    db_collection: str = None,
) -> list[dict]:
    """
    混合检索：结合向量语义相似度和 BM25 关键词匹配。

    使用 Reciprocal Rank Fusion (RRF) 算法融合两路检索排序，
    返回加权排序后的 Top-K 结果。

    Args:
        query (str):       查询字符串
        k (int):           返回结果数量 (默认 DEFAULT_K)
        vector_weight:     向量检索权重 (默认 0.5)
        bm25_weight:       BM25 检索权重 (默认 0.5)
        db_name:           数据库名称，默认使用模块常量 DB_NAME
        db_collection:     集合名称，默认使用模块常量 DB_COLLECTION

    Returns:
        list[dict]: 排序后的检索结果，每项包含:
            - rank:            排名 (1-based)
            - score:           RRF 融合分数
            - source:          来源文件路径
            - content_preview: 文本预览 (前200字)
            - content:         完整文本内容
            - metadata:        原始元数据
            - vector_score:    向量相似度分数 (若有)
            - bm25_score:      BM25 分数 (若有)
    """
    # 根据传入参数动态构建连接字符串和缓存路径
    conn_str = _build_connection_string(db_name) if db_name else CONNECTION_STRING
    coll_name = db_collection or DB_COLLECTION
    cache_path = _build_cache_path(db_name, db_collection) if db_name else CACHE_FILE

    # 1. 加载文档块 (BM25 需要)
    try:
        bm25_docs = load_chunks_from_cache(cache_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n请先运行 store_vectors.py 以生成向量数据和文本缓存。"
        ) from e
    cache_docs_by_id = {
        doc.metadata.get("doc_id"): doc
        for doc in bm25_docs
        if doc.metadata.get("doc_id")
    }

    # 2. 构建检索器（向量检索仅在 USE_VECTOR=True 时启用）
    vector_store = None
    if USE_VECTOR:
        vector_store, _ = build_vector_retriever(conn_str, coll_name)
    bm25_retriever = build_bm25_retriever(bm25_docs, k=max(k * 2, DEFAULT_K))

    # 3. 执行检索
    print(f"\n[INFO] 查询: \"{query}\"")

    # 向量检索（仅在向量存储可用时执行）
    vector_results = []
    if vector_store is not None:
        try:
            vector_results = vector_store.similarity_search_with_score(
                query, k=k * 2
            )
            refreshed_vector_results = []
            stale_count = 0
            for doc, score in vector_results:
                doc_id = doc.metadata.get("doc_id")
                if doc_id in cache_docs_by_id:
                    refreshed_vector_results.append((cache_docs_by_id[doc_id], score))
                else:
                    stale_count += 1
            vector_results = refreshed_vector_results
            if stale_count:
                print(f"[INFO] Skipped {stale_count} stale vector results not found in current chunks cache")
            print(f"[INFO] 向量检索返回 {len(vector_results)} 条候选")
        except Exception as e:
            print(f"[警告] 向量检索失败: {e}")
            vector_results = []
    else:
        print("[INFO] 向量检索跳过（存储不可用），仅使用 BM25")

    # BM25 检索
    bm25_results = bm25_retriever.invoke(query)
    print(f"[INFO] BM25  检索返回 {len(bm25_results)} 条候选")

    # 4. RRF 融合排序（纯 BM25 模式：仅按 BM25 排名）
    if vector_results:
        results = reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
    else:
        # 无向量结果时，直接使用 BM25 结果
        results = []
        for rank, doc in enumerate(bm25_results[:k], start=1):
            source = doc.metadata.get("source", "unknown")
            content_preview = doc.page_content[:200].replace("\n", " ").strip()
            results.append({
                "rank": rank,
                "score": 1.0,
                "source": source,
                "content_preview": content_preview,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "vector_score": None,
                "bm25_score": doc.metadata.get("score"),
            })

    print(f"[INFO] RRF 融合后返回 Top-{len(results)}")
    return results


# ============================================================
# 结果格式化输出
# ============================================================

def print_results(results: list[dict], query: str) -> None:
    """
    格式化打印混合检索结果。

    Args:
        results: hybrid_search() 返回的结果列表
        query:   原始查询字符串
    """
    print("\n" + "=" * 70)
    print(f'  查询: "{query}"')
    print(f"  返回: {len(results)} 条结果")
    print(f"  权重: 向量={VECTOR_WEIGHT}, BM25={BM25_WEIGHT}")
    print("=" * 70)

    if not results:
        print("\n  (无结果)")
        return

    for r in results:
        source_name = Path(r["source"]).name if r["source"] != "unknown" else "?"
        preview = r["content_preview"]

        print(f"\n  [{r['rank']:2d}] RRF={r['score']:.4f} | 来源: {source_name}")
        if r["vector_score"] is not None:
            print(f"      向量分={r['vector_score']:.4f}", end="")
        if r["bm25_score"] is not None:
            print(f"  BM25分={r['bm25_score']:.4f}", end="")
        if r["vector_score"] or r["bm25_score"]:
            print()
        print(f"      {preview}...")

    print("\n" + "-" * 70)


# ============================================================
# 主入口
# ============================================================

def main() -> None:

    query = "耐磨钢的屈服强度范围"
    results = vector_search(query)
    print_results(results, query)

def vector_search(query: str, db_name: str = None) -> list[dict]:
    try:
        results = hybrid_search(query, k=DEFAULT_K, db_name=db_name)
        return results
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 检索失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
