#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档向量存储脚本
===============
加载 ./docs 目录中的文档，分割成文本块，生成嵌入向量，存储到 PostgreSQL pgvector。

使用前请确保:
    1. PostgreSQL 已安装并运行 (端口 5432)
    2. pgvector 扩展已安装
    3. 安装依赖:
       pip install psycopg2-binary pgvector "sentence-transformers>=2.2.0" pypdf langchain langchain-community

运行方式:
    python store_vectors.py          # 默认: 跳过已存在的块
    python store_vectors.py --force  # 强制重新处理所有文档

数据库配置 (可修改下方常量):
    主机: localhost:5432
    数据库: 通过 DB_NAME 选择（不存在时自动创建）
    连接参数: 通过 api/.env 中的 POSTGRES_* 变量配置
"""

import os

# ---- 必须在所有 HuggingFace 相关 import 之前设置 ----
# VPN 已连接，直连 huggingface.co
# 国内网络无法访问时可取消注释下面一行：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import io
import json
import hashlib
import re
import shutil
import unicodedata
from pathlib import Path
from html import unescape
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

# LangChain 文档加载
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

# 图片 OCR (可选依赖 — 如未安装则跳过图片格式)
try:
    from PIL import Image
    import pytesseract

    # 自动检测 Tesseract 安装路径
    _TESSERACT_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    _tesseract_executable = shutil.which("tesseract")
    for _tp in _TESSERACT_PATHS:
        if Path(_tp).is_file():
            pytesseract.pytesseract.tesseract_cmd = _tp
            _tesseract_executable = _tp
            break
    _HAS_OCR = bool(_tesseract_executable)
except ImportError:
    _HAS_OCR = False

# PyMuPDF (用于 PDF OCR 回退 — 将页面渲染为图片)
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

# pdfplumber (PDF 表格提取 — 比 pypdf 更精准，保留表格结构)
try:
    import pdfplumber as _pdfplumber_lib
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

# MinerU (magic-pdf) — 高精度 PDF→Markdown，含 OCR+表格识别
try:
    from magic_pdf.tools.common import do_parse, prepare_env
    from magic_pdf.data.data_reader_writer import FileBasedDataReader
    _HAS_MINERU = True
except ImportError:
    _HAS_MINERU = False

# 嵌入模型 (本地运行, 无需 API Key)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# PGVector 向量存储。新版优先使用 langchain-postgres；未安装时兼容旧版。
try:
    from langchain_postgres import PGVector
    _PGVECTOR_BACKEND = "langchain_postgres"
except ImportError:
    from langchain_community.vectorstores import PGVector
    _PGVECTOR_BACKEND = "langchain_community"

# PostgreSQL 直连 (用于创建数据库和扩展)
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# ============================================================
# 配置常量
# ============================================================


# PostgreSQL 连接参数
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
# DB_NAME = "Nb_KnowBase_db"# 通用数据库
# DB_NAME = "gcjxyg_db"#耐磨钢标准
# DB_NAME = "gcjxyg_Know_db"#耐磨钢知识库
# DB_NAME = "gxg_db"#管线用钢标准
# DB_NAME = "gxg_Know_db"#管线用钢知识库
# DB_NAME = "hgg_db"#海工用钢标准
# DB_NAME = "hgg_Know_db"#海工用钢知识库
# DB_NAME = "jzyg_db"#建筑用钢用钢标准
# DB_NAME = "jzyg_Know_db"#建筑用钢用钢知识库
# DB_NAME = "jgyg_db"#结构用钢用钢标准
DB_NAME = "jgyg_Know_db"#结构用钢用钢知识库
# DB_NAME = "qcyg_db"#汽车用钢用钢标准
# DB_NAME = "qcyg_Know_db"#汽车用钢用钢知识库
# DB_NAME = "qlyg_db"#桥梁用钢用钢标准
# DB_NAME = "qlyg_Know_db"#桥梁用钢用钢知识库

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_COLLECTION = "documents"  # PGVector 中的表/集合名

DOCS_DIR = Path(__file__).parent / "docs"
CACHE_FILE = Path(__file__).parent / f"chunks_cache_{DB_NAME}_{DB_COLLECTION}.json"
# 按数据库分别保存 PDF 识别后的 Markdown 文件。
MARKDOWN_ROOT = Path(__file__).parent / "markdown"
MARKDOWN_DB_DIR = MARKDOWN_ROOT / DB_NAME
# PDF 默认优先使用 MinerU；显式设置为 0/false/off 时才关闭。
USE_MINERU = os.getenv("STORE_VECTORS_USE_MINERU", "1").lower() not in {"0", "false", "no", "off"}

# 文本分割参数（钢材标准表格密度高，块太小会切断表格数据）
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
QUALITY_REPORT_FILE = Path(__file__).parent / "extraction_report.json"

# 嵌入模型 (sentence-transformers 本地模型, 384 维)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# PGVector 连接字符串
CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# ============================================================
# 辅助函数
# ============================================================

def ensure_docs_directory() -> None:
    """确保 docs 目录存在；首次运行时创建并放入示例文档。"""
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True,exist_ok=True)
        # print(f"[INFO] 已创建 docs 目录: {DOCS_DIR}")

        # sample_file = DOCS_DIR / "sample.txt"
        # sample_content = (
        #     "人工智能（Artificial Intelligence，AI）是计算机科学的一个重要分支，\n"
        #     "旨在创建能够模拟人类智能的系统。这些系统可以执行诸如学习、推理、\n"
        #     "问题解决、感知和语言理解等任务。\n\n"
        #     "机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的\n"return
        #     "情况下从数据中学习。深度学习是机器学习的一种方法，使用多层神经\n"
        #     "网络来学习数据的层次化表示。\n\n"
        #     "自然语言处理（NLP）是AI的一个重要应用领域，涉及计算机与人类语言\n"
        #     "之间的交互。现代NLP系统使用大型语言模型（LLM）来理解和生成人类语言。\n\n"
        #     "向量数据库是一类专门用于存储和检索高维向量的数据库系统。在AI应用中，\n"
        #     "向量数据库常被用于存储文档嵌入，支持语义搜索和相似度匹配。\n\n"
        #     "pgvector是PostgreSQL的一个扩展，它添加了向量数据类型和相似度搜索功能。\n"
        #     "这使得PostgreSQL可以作为向量数据库使用，同时保留关系数据库的所有优势。\n"
        # )
        # sample_file.write_text(sample_content, encoding="utf-8")
        # print(f"[INFO] 已创建示例文档: {sample_file}")
    else:
        filife_count = len([f for f in DOCS_DIR.iterdir() if f.is_file()])
    if filife_count <= 0:
        print(f"没有需要加载的文档")
        isdir = False
    else:
        print(f"[INFO] docs 目录已存在，包含 {filife_count} 个文件")
        isdir = True
    return isdir


def _save_extracted_markdown(pdf_file: Path, markdown_content: str) -> Path | None:
    """保存到 markdown/<DB_NAME>/<原PDF同名>.md，供人工核查。"""
    content = (markdown_content or "").strip()
    if not content:
        return None
    MARKDOWN_DB_DIR.mkdir(parents=True, exist_ok=True)
    output_file = MARKDOWN_DB_DIR / f"{pdf_file.stem}.md"
    output_file.write_text(content + "\n", encoding="utf-8")
    print(f"[INFO] Markdown 已保存: {output_file}")
    return output_file


def _table_to_markdown(table: list[list]) -> str:
    """将 pdfplumber 提取的二维表格转为 Markdown 格式。"""
    if not table or len(table) == 0:
        return ""
    # 过滤空行
    rows = [[str(c).replace("\n", " ").strip() if c else "" for c in row] for row in table if any(row)]
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    # 补齐列
    for row in rows:
        while len(row) < max_cols:
            row.append("")
    # 构建 Markdown 表格
    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


_MINERU_OCR_MODEL = None


def _parse_span(attrs: str, name: str) -> int:
    match = re.search(rf'{name}\s*=\s*["\']?(\d+)', attrs or "", re.I)
    return int(match.group(1)) if match else 1


def _detect_table_grid(image) -> tuple[list[int], list[int]]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
    height, width = bw.shape

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 20), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 20)))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    y_pixels = np.where(h_lines.sum(axis=1) > 255 * width * 0.25)[0]
    x_pixels = np.where(v_lines.sum(axis=0) > 255 * height * 0.25)[0]

    def _centers(values) -> list[int]:
        if len(values) == 0:
            return []
        groups = []
        start = prev = int(values[0])
        for value in values[1:]:
            value = int(value)
            if value <= prev + 2:
                prev = value
            else:
                groups.append((start + prev) // 2)
                start = prev = value
        groups.append((start + prev) // 2)
        return groups

    return _centers(x_pixels), _centers(y_pixels)


def _get_mineru_ocr_model():
    global _MINERU_OCR_MODEL
    if _MINERU_OCR_MODEL is None:
        from magic_pdf.model.sub_modules.model_init import ocr_model_init

        _MINERU_OCR_MODEL = ocr_model_init(show_log=False, lang="ch")
    return _MINERU_OCR_MODEL


def _flatten_ocr_result(result) -> tuple[str, float]:
    texts = []
    confidence = 0.0

    def _walk(value):
        nonlocal confidence
        if value is None:
            return
        if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], str):
            texts.append(value[0])
            try:
                confidence = max(confidence, float(value[1]))
            except Exception:
                pass
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(result)
    return "".join(texts).strip(), confidence


def _ocr_table_cell(image) -> tuple[str, float]:
    import cv2

    ocr_model = _get_mineru_ocr_model()
    for scale in (1, 2, 3):
        crop = image
        if scale > 1:
            crop = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result = ocr_model.ocr(crop, det=True, rec=True)[0]
        text, confidence = _flatten_ocr_result(result)
        text = re.sub(r"\s+", "", text)
        text = text.replace("O", "0").replace("o", "0")
        if text and confidence >= 0.80 and _is_safe_ocr_cell_value(text):
            return text, confidence
    return "", 0.0


def _is_safe_ocr_cell_value(text: str) -> bool:
    """Only backfill numeric-like table cells; leave uncertain blanks unchanged."""
    if not re.search(r"\d", text):
        return False
    return bool(re.fullmatch(r"[0-9.,~\-+<>≤≥=×xX/%℃°]+", text))


def _repair_table_body_empty_cells(table_body: str, image_path: Path) -> tuple[str, int]:
    import cv2
    import numpy as np

    if not image_path.exists() or "<td" not in table_body or "></td>" not in table_body:
        return table_body, 0

    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return table_body, 0

    x_lines, y_lines = _detect_table_grid(image)
    if len(x_lines) < 2 or len(y_lines) < 2:
        return table_body, 0

    row_count = len(y_lines) - 1
    col_count = len(x_lines) - 1
    tr_matches = list(re.finditer(r"<tr\b[^>]*>(.*?)</tr>", table_body, re.I | re.S))
    if not tr_matches:
        return table_body, 0

    replacements = []
    active_until: dict[int, int] = {}

    for row_idx, tr_match in enumerate(tr_matches):
        if row_idx >= row_count:
            break
        row_html = tr_match.group(1)
        col_idx = 0
        for td_match in re.finditer(r"<td\b([^>]*)>(.*?)</td>", row_html, re.I | re.S):
            while active_until.get(col_idx, -1) >= row_idx:
                col_idx += 1

            attrs = td_match.group(1) or ""
            raw_text = td_match.group(2)
            text = unescape(re.sub(r"<[^>]+>", "", raw_text)).strip()
            colspan = _parse_span(attrs, "colspan")
            rowspan = _parse_span(attrs, "rowspan")
            cell_col = col_idx

            if rowspan > 1:
                for c in range(cell_col, cell_col + colspan):
                    active_until[c] = row_idx + rowspan - 1

            if (
                not text
                and colspan == 1
                and rowspan == 1
                and cell_col < col_count
            ):
                x1, x2 = x_lines[cell_col], x_lines[cell_col + 1]
                y1, y2 = y_lines[row_idx], y_lines[row_idx + 1]
                pad_x = max(4, int((x2 - x1) * 0.03))
                pad_y = max(4, int((y2 - y1) * 0.08))
                crop = image[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]
                if crop.size:
                    value, confidence = _ocr_table_cell(crop)
                    if value:
                        abs_start = tr_match.start(1) + td_match.start()
                        abs_end = tr_match.start(1) + td_match.end()
                        replacements.append((abs_start, abs_end, f"<td{attrs}>{value}</td>"))

            col_idx += colspan

    if not replacements:
        return table_body, 0

    repaired = table_body
    for start, end, value in sorted(replacements, reverse=True):
        repaired = repaired[:start] + value + repaired[end:]
    return repaired, len(replacements)


def _repair_mineru_markdown_tables(md_content: str, content_list_file: Path) -> str:
    try:
        content_items = json.loads(content_list_file.read_text(encoding="utf-8"))
    except Exception:
        return md_content

    base_dir = content_list_file.parent
    repaired_md = md_content
    repaired_count = 0

    for item in content_items:
        if item.get("type") != "table":
            continue
        table_body = item.get("table_body") or ""
        img_path = item.get("img_path") or ""
        if not table_body or not img_path:
            continue

        repaired_table, count = _repair_table_body_empty_cells(table_body, base_dir / img_path)
        if count and table_body in repaired_md:
            repaired_md = repaired_md.replace(table_body, repaired_table, 1)
            repaired_count += count

    if repaired_count:
        print(f"[INFO] MinerU table OCR repaired {repaired_count} empty cells")
    return repaired_md


def _mineru_content_list_to_page_documents(
    content_list_file: Path,
    pdf_file: Path,
) -> list[Document]:
    """将 MinerU content_list 按页还原，尽量保留正文和表格页码。"""
    try:
        items = json.loads(content_list_file.read_text(encoding="utf-8"))
    except Exception:
        return []

    page_parts: dict[int, list[str]] = {}
    for item in items if isinstance(items, list) else []:
        try:
            page_idx = int(item.get("page_idx", 0))
        except (TypeError, ValueError):
            page_idx = 0

        item_type = str(item.get("type") or "")
        if item_type == "table":
            content = item.get("table_body") or item.get("text") or ""
            img_path = item.get("img_path") or ""
            if content and img_path:
                content, _ = _repair_table_body_empty_cells(
                    str(content), content_list_file.parent / img_path
                )
        else:
            content = item.get("text") or item.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content if part)
        content = str(content).strip()
        if content:
            page_parts.setdefault(page_idx, []).append(content)

    documents = []
    for page_idx in sorted(page_parts):
        page_content = "\n\n".join(page_parts[page_idx]).strip()
        if page_content:
            documents.append(Document(
                page_content=page_content,
                metadata={
                    "source": str(pdf_file),
                    "page": page_idx + 1,
                    "page_start": page_idx + 1,
                    "page_end": page_idx + 1,
                    "type": "pdf_mineru",
                    "extractor": "mineru",
                },
            ))
    return documents


def _load_pdf_with_plumber(pdf_file: Path) -> list | None:
    """
    使用 pdfplumber 加载 PDF：提取正文 + 表格 → 生成 Markdown。

    仅对可提取文本的 PDF 有效；扫描件返回 None。
    """
    if not _HAS_PDFPLUMBER:
        return None

    pages = []
    tables_found = 0
    total_chars = 0
    total_page_count = 0

    with _pdfplumber_lib.open(str(pdf_file)) as pdf:
        total_page_count = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            md_parts = []
            if text.strip():
                md_parts.append(text)

            for table in tables:
                if table and any(any(c for c in row if c) for row in table):
                    md_table = _table_to_markdown(table)
                    if md_table:
                        md_parts.append(md_table)
                        tables_found += 1

            page_content = "\n\n".join(md_parts).strip()
            if page_content:
                total_chars += len(page_content)
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "source": str(pdf_file),
                        "page": page_num + 1,
                        "page_start": page_num + 1,
                        "page_end": page_num + 1,
                        "type": "pdf_text",
                        "extractor": "pdfplumber",
                    },
                )
                pages.append(doc)

    if total_chars == 0:
        return None  # 扫描件, 交 OCR 处理

    combined_markdown = "\n\n".join(
        f"<!-- page: {doc.metadata.get('page', index)} -->\n\n{doc.page_content}"
        for index, doc in enumerate(pages, start=1)
    )
    _save_extracted_markdown(pdf_file, combined_markdown)

    print(
        f"[INFO] PDF {pdf_file.name}: 加载 {len(pages)}/{total_page_count} 页"
        + (f" (表格: {tables_found}个)" if tables_found > 0 else "")
    )
    return pages


def _load_pdf_with_mineru(pdf_file: Path) -> list | None:
    """
    使用 MinerU (magic-pdf) 处理 PDF — 高精度 OCR + 版面分析 + 表格识别 → Markdown。

    需要先下载模型: python download_models.py
    MinerU 自动识别 PDF 类型（文本/扫描件），统一输出 Markdown。
    """
    if not _HAS_MINERU:
        return None

    models_dir = str(Path.home() / "magic-pdf-models")
    if not Path(models_dir).exists():
        return None  # 模型未下载，静默跳过

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            pdf_bytes = pdf_file.read_bytes()
            prepare_env(tmpdir, pdf_file.stem, "auto")
            do_parse(
                output_dir=tmpdir,
                pdf_file_name=pdf_file.stem,
                pdf_bytes_or_dataset=pdf_bytes,
                model_list=[],
                parse_method="auto",     # auto = 自动选择 txt/ocr
                lang="ch",
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_json=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,
                table_enable=True,
                formula_enable=False,
            )

            # 查找生成的 Markdown 文件
            md_files = list(Path(tmpdir).rglob(f"{pdf_file.stem}.md"))
            if not md_files:
                return None

            md_content = md_files[0].read_text(encoding="utf-8")
            content_list_files = list(Path(tmpdir).rglob(f"{pdf_file.stem}_content_list.json"))
            if content_list_files:
                md_content = _repair_mineru_markdown_tables(md_content, content_list_files[0])
            if not md_content.strip():
                return None
            _save_extracted_markdown(pdf_file, md_content)

            if content_list_files:
                page_documents = _mineru_content_list_to_page_documents(
                    content_list_files[0], pdf_file
                )
                # 部分 MinerU 版本的 content_list 不含完整正文，此时保留完整 Markdown。
                page_chars = sum(len(doc.page_content) for doc in page_documents)
                if page_documents and page_chars >= len(md_content) * 0.5:
                    print(
                        f"[INFO] PDF {pdf_file.name}: MinerU 转换成功 "
                        f"({len(page_documents)} 页, {page_chars} 字符)"
                    )
                    return page_documents

            doc = Document(
                page_content=md_content,
                metadata={
                    "source": str(pdf_file),
                    "page": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "type": "pdf_mineru",
                    "extractor": "mineru",
                },
            )
            print(
                f"[INFO] PDF {pdf_file.name}: MinerU 转换成功"
                f" ({len(md_content)} 字符, 含表格识别)"
            )
            return [doc]
        except Exception as e:
            print(f"[WARN] MinerU 处理失败 {pdf_file.name}: {e}")
            return None


def load_documents() -> list:
    """
    加载 docs 目录中所有支持的文档。

    支持的格式:
        - .txt  — 纯文本
        - .md   — Markdown
        - .pdf  — PDF
        - .docx — Word 文档
        - .png / .jpg / .bmp / .tiff — 图片 (OCR 文字提取，需安装 Tesseract)

    Returns:
        list[Document]: LangChain Document 对象列表
    """
    if not any(DOCS_DIR.iterdir()):
        print("[ERROR] docs 目录为空，请放入支持的文档文件后重试。")
        sys.exit(1)

    documents = []

    # ---- 加载 .txt 和 .md 文件 ----
    for pattern in ["**/*.txt", "**/*.md"]:
        try:
            loader = DirectoryLoader(
                str(DOCS_DIR),
                glob=pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=False,
            )
            docs = loader.load()
            if docs:
                documents.extend(docs)
                print(f"[INFO] {pattern}: 加载 {len(docs)} 个文档")
        except Exception as e:
            print(f"[WARN] 加载 {pattern} 失败: {e}")

    # ---- 加载 .pdf 文件（默认 pdfplumber → PyPDFLoader；--mineru 时优先 MinerU）----
    pdf_files = list(DOCS_DIR.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        # ---- 优先: MinerU (高精度 OCR + 版面分析 + 表格识别 → Markdown) ----
        if USE_MINERU and _HAS_MINERU:
            try:
                mineru_docs = _load_pdf_with_mineru(pdf_file)
                if mineru_docs:
                    documents.extend(mineru_docs)
                    continue
                print(f"[WARN] MinerU 未提取到内容，回退 pdfplumber: {pdf_file.name}")
            except Exception as exc:
                print(f"[WARN] MinerU 处理失败，回退 pdfplumber: {pdf_file.name}: {exc}")
        elif USE_MINERU:
            print(f"[WARN] MinerU 不可用，回退 pdfplumber: {pdf_file.name}")

        # ---- 备选: pdfplumber (表格提取 → Markdown) ----
        try:
            pdf_docs = _load_pdf_with_plumber(pdf_file)
            if pdf_docs:
                documents.extend(pdf_docs)
                continue
        except Exception as exc:
            print(f"[WARN] pdfplumber 处理失败，继续 PyPDFLoader: {pdf_file.name}: {exc}")

        # ---- 回退: PyPDFLoader + OCR (MinerU/pdfplumber 不可用) ----
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            empty_pages = sum(1 for d in docs if not d.page_content.strip())

            if empty_pages > 0 and _HAS_FITZ and _HAS_OCR:
                print(
                    f"[INFO] PDF {pdf_file.name}: {len(docs)} 页, "
                    f"其中 {empty_pages} 页文本为空, 使用 OCR 回退..."
                )
                pdf_doc = fitz.open(str(pdf_file))
                for i, page in enumerate(pdf_doc):
                    if i < len(docs) and not docs[i].page_content.strip():
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
                        if text.strip():
                            docs[i].page_content = text
                pdf_doc.close()
                valid = [d for d in docs if d.page_content.strip()]
                print(f"[INFO] PDF {pdf_file.name}: OCR 后有效 {len(valid)}/{len(docs)} 页")
                documents.extend(valid)
            elif empty_pages > 0 and (not _HAS_FITZ or not _HAS_OCR):
                valid = [d for d in docs if d.page_content.strip()]
                print(
                    f"[INFO] PDF {pdf_file.name}: {len(docs)} 页, "
                    f"其中 {empty_pages} 页文本为空 (OCR 不可用, 丢弃空页)"
                )
                if empty_pages == len(docs):
                    print("[HINT] 该 PDF 可能是扫描件, 请安装 OCR 依赖: pip install PyMuPDF pytesseract Pillow")
                documents.extend(valid)
            else:
                documents.extend(docs)
                print(f"[INFO] PDF {pdf_file.name}: 加载 {len(docs)} 页 (文本)")
        except Exception as e:
            print(f"[WARN] 加载 PDF 失败 {pdf_file.name}: {e}")

    # ---- 加载 .docx Word 文档 ----
    docx_files = list(DOCS_DIR.glob("**/*.docx"))
    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            docs = loader.load()
            documents.extend(docs)
            print(f"[INFO] DOCX {docx_file.name}: 加载 {len(docs)} 个文档")
        except Exception as e:
            print(f"[WARN] 加载 DOCX 失败 {docx_file.name}: {e}")

    # ---- 加载图片文件 (OCR) ----
    if _HAS_OCR:
        img_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")
        img_files = []
        for ext in img_extensions:
            img_files.extend(DOCS_DIR.glob(f"**/{ext}"))

        for img_file in img_files:
            try:
                img = Image.open(str(img_file))
                text = pytesseract.image_to_string(img, lang="chi_sim+eng")
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"source": str(img_file), "type": "image_ocr"},
                    )
                    documents.append(doc)
                    print(f"[INFO] OCR {img_file.name}: 提取 {len(text)} 个字符")
                else:
                    print(f"[WARN] OCR {img_file.name}: 未识别到文字")
            except Exception as e:
                print(f"[WARN] OCR 处理失败 {img_file.name}: {e}")
    else:
        # 检查是否有图片文件但 OCR 不可用
        has_images = any(
            DOCS_DIR.glob(f"**/{ext}")
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")
        )
        if has_images:
            print("[WARN] 检测到图片文件，但 OCR 依赖未安装。")
            print("[HINT]  安装: pip install pytesseract Pillow")
            print("[HINT]  还需安装 Tesseract-OCR: https://github.com/UB-Mannheim/tesseract/wiki")

    if not documents:
        print("[ERROR] 未能加载任何文档，请检查 docs 目录中的文件格式。")
        print("[HINT]  支持的格式: .txt, .md, .pdf, .docx, .png/.jpg (需OCR)")
        sys.exit(1)

    # 清理 NUL 字符 (OCR 文本中常见, PostgreSQL 不接受)
    nul_count = 0
    for doc in documents:
        if "\x00" in doc.page_content:
            nul_count += 1
            doc.page_content = doc.page_content.replace("\x00", "")
    if nul_count > 0:
        print(f"[INFO] 清理了 {nul_count} 个文档中的 NUL 字符")

    print(f"[INFO] 总共加载 {len(documents)} 个文档单元")
    return documents


_MOJIBAKE_MARKERS = ("锟", "鐨", "銆", "鈥", "犐犆犛")


def _assess_chunk_quality(text: str) -> tuple[bool, float, list[str]]:
    """对单个片段做保守质量检查，避免乱码参与嵌入和 BM25。"""
    text = (text or "").replace("\x00", "").strip()
    reasons = []
    non_space = [char for char in text if not char.isspace()]
    if len(non_space) < 20:
        reasons.append("有效内容不足20个字符")
        return False, 0.0, reasons

    replacement_count = text.count("\ufffd")
    cid_count = len(re.findall(r"\(cid:\d+\)", text, re.I))
    private_count = sum(unicodedata.category(char) == "Co" for char in text)
    control_count = sum(
        unicodedata.category(char) == "Cc" and char not in "\n\r\t"
        for char in text
    )
    if replacement_count:
        reasons.append(f"包含{replacement_count}个Unicode替换字符")
    if cid_count:
        reasons.append(f"包含{cid_count}个CID占位符")
    if private_count:
        reasons.append(f"包含{private_count}个私有区字符")
    if control_count:
        reasons.append(f"包含{control_count}个非法控制字符")

    useful_count = sum(
        char.isalnum() or "\u4e00" <= char <= "\u9fff"
        for char in non_space
    )
    useful_ratio = useful_count / len(non_space)
    if useful_ratio < 0.55:
        reasons.append(f"有效文字数字比例过低({useful_ratio:.1%})")

    meaningful_lines = [line.strip() for line in text.splitlines() if line.strip()]
    symbol_lines = 0
    for line in meaningful_lines:
        chars = [char for char in line if not char.isspace()]
        if len(chars) >= 8:
            symbol_ratio = sum(
                unicodedata.category(char).startswith(("P", "S"))
                for char in chars
            ) / len(chars)
            if symbol_ratio >= 0.8:
                symbol_lines += 1
    symbol_line_ratio = symbol_lines / max(1, len(meaningful_lines))
    if symbol_line_ratio > 0.30:
        reasons.append(f"纯符号行比例过高({symbol_line_ratio:.1%})")

    marker_count = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
    if marker_count >= 3 or "犐犆犛" in text:
        reasons.append("命中典型乱码字符序列")

    # 部分错误 PDF 字体映射会把大量正文映射到罕见的犬部汉字。
    han_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    animal_radical_chars = sum("\u7200" <= char <= "\u72ff" for char in han_chars)
    if len(han_chars) >= 20 and animal_radical_chars >= max(8, int(len(han_chars) * 0.25)):
        reasons.append("罕见犬部汉字异常集中，疑似PDF字体映射错误")

    score = 1.0
    score -= min(0.45, (replacement_count + cid_count + private_count + control_count) * 0.1)
    score -= max(0.0, 0.55 - useful_ratio)
    score -= min(0.3, symbol_line_ratio)
    if marker_count >= 3:
        score -= 0.35
    return not reasons, round(max(0.0, score), 4), reasons


def _split_markdown_table(
    table_text: str,
    metadata: dict,
    max_size: int = CHUNK_SIZE,
) -> list[Document]:
    """超长 Markdown 表格按行拆分，并为每个片段重复表头。"""
    max_size = max(200, max_size)

    def _table_document(content: str) -> Document:
        chunk_metadata = dict(metadata)
        if len(content) > max_size:
            chunk_metadata["structural_overflow"] = True
        return Document(page_content=content.strip(), metadata=chunk_metadata)

    if "<table" in table_text.lower() and "</table>" in table_text.lower():
        if len(table_text) <= max_size:
            return [_table_document(table_text)]
        table_open = re.search(r"<table\b[^>]*>", table_text, re.I)
        rows = re.findall(r"<tr\b[^>]*>.*?</tr>", table_text, re.I | re.S)
        if rows and table_open:
            thead_match = re.search(r"<thead\b[^>]*>.*?</thead>", table_text, re.I | re.S)
            tbody_open = re.search(r"<tbody\b[^>]*>", table_text, re.I)
            captions = re.findall(r"<(?:caption|colgroup)\b[^>]*>.*?</(?:caption|colgroup)>", table_text, re.I | re.S)
            tfoot_match = re.search(r"<tfoot\b[^>]*>.*?</tfoot>", table_text, re.I | re.S)
            body_source = table_text
            if thead_match:
                header_html = thead_match.group(0)
                body_source = body_source.replace(header_html, "", 1)
            else:
                header_html = rows[0]
                body_source = body_source.replace(header_html, "", 1)
            if tfoot_match:
                body_source = body_source.replace(tfoot_match.group(0), "", 1)
            data_rows = re.findall(r"<tr\b[^>]*>.*?</tr>", body_source, re.I | re.S)
            if not data_rows:
                return [_table_document(table_text)]

            preamble = table_open.group(0) + "".join(captions) + header_html
            body_open = tbody_open.group(0) if tbody_open else ""
            body_close = "</tbody>" if tbody_open else ""
            footer = tfoot_match.group(0) if tfoot_match else ""

            def _render_html_rows(data: list[str]) -> str:
                return f"{preamble}{body_open}{''.join(data)}{body_close}{footer}</table>"

            chunks = []
            current_rows = []
            for row in data_rows:
                candidate = _render_html_rows(current_rows + [row])
                if len(candidate) > max_size and current_rows:
                    chunks.append(_table_document(_render_html_rows(current_rows)))
                    current_rows = []
                current_rows.append(row)
            if current_rows:
                chunks.append(_table_document(_render_html_rows(current_rows)))
            return chunks

    lines = [line for line in table_text.splitlines() if line.strip()]
    if len(table_text) <= max_size or len(lines) <= 3:
        return [_table_document(table_text)]

    header = lines[:2]
    chunks = []
    current = list(header)
    for row in lines[2:]:
        candidate = "\n".join(current + [row])
        if len(candidate) > max_size and len(current) > len(header):
            chunks.append(_table_document("\n".join(current)))
            current = list(header)
        current.append(row)
    if len(current) > len(header):
        chunks.append(_table_document("\n".join(current)))
    return chunks


_ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_HEADING_RE = re.compile(r"^\s*(=+|-+)\s*$")
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-+*]\s+|\d+[.)、．]\s*|[（(][一二三四五六七八九十\d]+[）)]\s*|"
    r"[一二三四五六七八九十]+、\s*)"
)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_TECH_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,5})\s+(.{1,100})$")
_APPENDIX_HEADING_RE = re.compile(
    r"^附录\s*[A-Za-zＡ-Ｚａ-ｚ0-9一二三四五六七八九十]+(?:\s*[（(].*?[）)])?.{0,80}$"
)
_TECH_HEADING_UNIT_RE = re.compile(
    r"^(?:MPa|GPa|Pa|mm|cm|m|kg|g|℃|°C|J|kJ|Hz|kHz|%|年版)(?:\s|$)",
    re.I,
)
_SECTION_PREFIX_LIMIT = min(500, CHUNK_SIZE // 4)


def _is_fence_close(line: str, fence_char: str, minimum_length: int) -> bool:
    """判断代码围栏关闭行，关闭标记后只允许空白。"""
    return bool(re.fullmatch(
        rf"\s*{re.escape(fence_char)}{{{minimum_length},}}\s*",
        line,
    ))


def _looks_like_technical_heading(number: str, title: str) -> bool:
    """保守判断无 Markdown 标记的技术标准编号标题。"""
    root_number = int(number.split(".", 1)[0])
    if root_number > 99 or _TECH_HEADING_UNIT_RE.match(title):
        return False
    han_count = len(re.findall(r"[一-鿿]", title))
    latin_count = len(re.findall(r"[A-Za-z]", title))
    return han_count >= 2 or latin_count >= 3


def _heading_at(lines: list[str], index: int) -> tuple[int, str, int] | None:
    """识别当前位置的 Markdown 或技术标准标题。"""
    stripped = lines[index].strip()
    match = _ATX_HEADING_RE.match(stripped)
    if match:
        return len(match.group(1)), match.group(2).strip(), 1

    if index + 1 < len(lines) and stripped and _SETEXT_HEADING_RE.match(lines[index + 1]):
        marker = lines[index + 1].strip()
        return (1 if marker.startswith("=") else 2), stripped, 2

    if len(stripped) <= 120 and _APPENDIX_HEADING_RE.match(stripped):
        return 1, stripped, 1

    match = _TECH_HEADING_RE.match(stripped)
    if match and not re.search(r"[。！？；;]$", stripped):
        number, title = match.groups()
        if _looks_like_technical_heading(number, title.strip()):
            return min(number.count(".") + 1, 6), f"{number} {title.strip()}", 1
    return None


def _pipe_table_cells(line: str) -> list[str]:
    """拆分 Markdown 表格行，允许省略外侧管道符。"""
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _is_pipe_table_row(line: str) -> bool:
    return len(_pipe_table_cells(line)) >= 2 and "|" in line


def _is_pipe_table_at(lines: list[str], index: int) -> bool:
    """判断当前位置是否为连续的 Markdown 管道表格。"""
    if index + 1 >= len(lines):
        return False
    first = lines[index].strip()
    second = lines[index + 1].strip()
    if not (_is_pipe_table_row(first) and _is_pipe_table_row(second)):
        return False
    cells = _pipe_table_cells(second)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _section_prefix(section: tuple[tuple[int, str], ...]) -> str:
    """将标题路径渲染为可参与向量化和 BM25 的 Markdown。"""
    rendered = []
    remaining = _SECTION_PREFIX_LIMIT
    for level, title in section:
        prefix = f"{'#' * min(level, 6)} "
        available = remaining - len(prefix)
        if available <= 0:
            break
        safe_title = title if len(title) <= available else f"{title[:max(1, available - 1)]}…"
        line = f"{prefix}{safe_title}"
        rendered.append(line)
        remaining -= len(line) + 1
    return "\n".join(rendered)


def _parse_semantic_blocks(
    text: str,
    initial_section: tuple[tuple[int, str], ...] = (),
) -> tuple[list[dict], tuple[tuple[int, str], ...]]:
    """按标题、段落、列表、代码和表格解析文档结构。"""
    lines = (text or "").splitlines()
    blocks = []
    section_stack: list[tuple[int, str]] = list(initial_section)
    index = 0

    def _section() -> tuple[tuple[int, str], ...]:
        return tuple(section_stack)

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        heading = _heading_at(lines, index)
        if heading:
            level, title, consumed = heading
            section_stack[:] = [item for item in section_stack if item[0] < level]
            section_stack.append((level, title))
            index += consumed
            continue

        fence_match = _FENCE_RE.match(lines[index])
        if fence_match:
            marker = fence_match.group(1)
            fence_char = marker[0]
            fence_length = len(marker)
            block_lines = [lines[index]]
            index += 1
            while index < len(lines):
                closing = _is_fence_close(lines[index], fence_char, fence_length)
                block_lines.append(lines[index])
                index += 1
                if closing:
                    break
            blocks.append({"type": "code", "text": "\n".join(block_lines), "section": _section()})
            continue

        if "<table" in lines[index].lower():
            block_lines = [lines[index]]
            index += 1
            while index < len(lines) and "</table>" not in block_lines[-1].lower():
                block_lines.append(lines[index])
                index += 1
            blocks.append({"type": "table", "text": "\n".join(block_lines), "section": _section()})
            continue

        if _is_pipe_table_at(lines, index):
            block_lines = []
            while index < len(lines):
                stripped = lines[index].strip()
                if not _is_pipe_table_row(stripped):
                    break
                block_lines.append(lines[index])
                index += 1
            blocks.append({"type": "table", "text": "\n".join(block_lines), "section": _section()})
            continue

        if _LIST_ITEM_RE.match(lines[index]):
            block_lines = [lines[index]]
            index += 1
            while index < len(lines) and lines[index].strip():
                if _heading_at(lines, index) or _FENCE_RE.match(lines[index]):
                    break
                if "<table" in lines[index].lower() or _is_pipe_table_at(lines, index):
                    break
                block_lines.append(lines[index])
                index += 1
            blocks.append({"type": "list", "text": "\n".join(block_lines), "section": _section()})
            continue

        block_lines = [lines[index]]
        index += 1
        while index < len(lines) and lines[index].strip():
            if _heading_at(lines, index) or _FENCE_RE.match(lines[index]):
                break
            if "<table" in lines[index].lower() or _is_pipe_table_at(lines, index):
                break
            if _LIST_ITEM_RE.match(lines[index]):
                break
            block_lines.append(lines[index])
            index += 1
        blocks.append({"type": "paragraph", "text": "\n".join(block_lines), "section": _section()})

    return blocks, tuple(section_stack)


def _split_oversized_text(text: str, max_size: int, overlap: int, block_type: str) -> list[str]:
    """超长正文优先按完整句子或列表项拆分，并保留有限上下文。"""
    max_size = max(1, max_size)
    overlap = min(max(0, overlap), max_size - 1) if max_size > 1 else 0
    if block_type == "list":
        units = [line.rstrip() for line in text.splitlines() if line.strip()]
        separator = "\n"
    else:
        units = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            sentences = re.findall(r"[^。！？；.!?;]+(?:[。！？；.!?;]+|$)", line)
            units.extend(sentence.strip() for sentence in sentences if sentence.strip())
        separator = "\n"

    expanded_units = []
    step = max(1, max_size - overlap)
    for unit in units or [text.strip()]:
        if len(unit) <= max_size:
            expanded_units.append(unit)
            continue
        start = 0
        while start < len(unit):
            expanded_units.append(unit[start:start + max_size])
            if start + max_size >= len(unit):
                break
            start += step

    chunks = []
    current: list[str] = []
    for unit in expanded_units:
        candidate = separator.join(current + [unit])
        if current and len(candidate) > max_size:
            chunks.append(separator.join(current))
            trailing: list[str] = []
            for old_unit in reversed(current):
                possible = [old_unit] + trailing
                if len(separator.join(possible)) > overlap:
                    break
                trailing = possible
            while trailing and len(separator.join(trailing + [unit])) > max_size:
                trailing.pop(0)
            current = trailing
        current.append(unit)
    if current:
        chunks.append(separator.join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def _split_document_semantically(
    document: Document,
    initial_section: tuple[tuple[int, str], ...] = (),
) -> tuple[list[Document], tuple[tuple[int, str], ...]]:
    """按文档结构生成标题感知的语义片段，并返回最终标题上下文。"""
    chunks = []
    current_parts: list[str] = []
    current_section: tuple[tuple[int, str], ...] = ()
    current_types: set[str] = set()

    def _metadata(
        section: tuple[tuple[int, str], ...],
        semantic_type: str,
        structural_overflow: bool = False,
    ) -> dict:
        metadata = dict(document.metadata)
        if section:
            metadata["section_path"] = [title for _, title in section]
        metadata["semantic_type"] = semantic_type
        if structural_overflow:
            metadata["structural_overflow"] = True
        return metadata

    def _render(text: str, section: tuple[tuple[int, str], ...]) -> str:
        prefix = _section_prefix(section)
        return f"{prefix}\n\n{text.strip()}" if prefix else text.strip()

    def _flush_current() -> None:
        if not current_parts:
            return
        content = _render("\n\n".join(current_parts), current_section)
        semantic_type = next(iter(current_types)) if len(current_types) == 1 else "mixed_prose"
        chunks.append(Document(
            page_content=content,
            metadata=_metadata(current_section, semantic_type),
        ))
        current_parts.clear()
        current_types.clear()

    blocks, final_section = _parse_semantic_blocks(
        document.page_content or "",
        initial_section,
    )
    for block in blocks:
        block_type = block["type"]
        block_text = block["text"].strip()
        section = block["section"]
        prefix = _section_prefix(section)
        available_size = max(200, CHUNK_SIZE - len(prefix) - (2 if prefix else 0))

        if block_type == "table":
            _flush_current()
            table_metadata = _metadata(section, "table")
            table_chunks = _split_markdown_table(block_text, table_metadata, available_size)
            for table_chunk in table_chunks:
                table_chunk.page_content = _render(table_chunk.page_content, section)
                chunks.append(table_chunk)
            current_section = section
            continue

        if block_type == "code":
            _flush_current()
            content = _render(block_text, section)
            chunks.append(Document(
                page_content=content,
                metadata=_metadata(section, "code", len(content) > CHUNK_SIZE),
            ))
            current_section = section
            continue

        if current_parts and section != current_section:
            _flush_current()
        current_section = section
        candidate_parts = current_parts + [block_text]
        candidate = _render("\n\n".join(candidate_parts), section)
        if len(candidate) <= CHUNK_SIZE:
            current_parts.append(block_text)
            current_types.add(block_type)
            continue

        _flush_current()
        if len(_render(block_text, section)) <= CHUNK_SIZE:
            current_parts.append(block_text)
            current_types.add(block_type)
            continue

        for part in _split_oversized_text(
            block_text,
            available_size,
            min(CHUNK_OVERLAP, available_size - 1),
            block_type,
        ):
            chunks.append(Document(
                page_content=_render(part, section),
                metadata=_metadata(section, block_type),
            ))

    _flush_current()
    return chunks, final_section


def split_documents(documents: list) -> list:
    """按语义结构分块并逐片段过滤乱码，只保留可用于检索的片段。"""
    raw_chunks = []
    source_sections: dict[str, tuple[tuple[int, str], ...]] = {}
    for document in documents:
        source = str(document.metadata.get("source", "unknown"))
        document_chunks, final_section = _split_document_semantically(
            document,
            source_sections.get(source, ()),
        )
        raw_chunks.extend(document_chunks)
        source_sections[source] = final_section

    accepted_chunks = []
    rejected_entries = []
    source_counter: dict[str, int] = {}
    for raw_index, chunk in enumerate(raw_chunks):
        chunk.page_content = chunk.page_content.replace("\x00", "").strip()
        source = chunk.metadata.get("source", "unknown")
        accepted, score, reasons = _assess_chunk_quality(chunk.page_content)
        if not accepted:
            rejected_entries.append({
                "source": source,
                "page": chunk.metadata.get("page"),
                "raw_chunk_index": raw_index,
                "quality_score": score,
                "reasons": reasons,
                "preview": chunk.page_content[:160],
            })
            continue

        idx = source_counter.get(source, 0)
        page_start = chunk.metadata.get("page_start", chunk.metadata.get("page", 1))
        page_end = chunk.metadata.get("page_end", page_start)
        content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        chunk.metadata.update({
            "source_name": Path(source).name,
            "page_start": page_start,
            "page_end": page_end,
            "quality_score": score,
            "chunk_index": idx,
            "content_sha256": content_hash,
            "doc_id": hashlib.sha256(
                f"{DB_NAME}:{source}:{page_start}:{idx}:{content_hash}".encode("utf-8")
            ).hexdigest(),
        })
        source_counter[source] = idx + 1
        accepted_chunks.append(chunk)

    structural_overflows = [
        {
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page"),
            "semantic_type": chunk.metadata.get("semantic_type"),
            "length": len(chunk.page_content),
            "preview": chunk.page_content[:160],
        }
        for chunk in accepted_chunks
        if chunk.metadata.get("structural_overflow")
    ]
    report = {
        "database": DB_NAME,
        "documents": len(documents),
        "raw_chunks": len(raw_chunks),
        "accepted_chunks": len(accepted_chunks),
        "rejected_chunks": len(rejected_entries),
        "structural_overflow_chunks": len(structural_overflows),
        "structural_overflows": structural_overflows,
        "rejected": rejected_entries,
    }
    QUALITY_REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[INFO] 分割得到 {len(raw_chunks)} 个片段，质量检查通过 "
        f"{len(accepted_chunks)} 个，跳过 {len(rejected_entries)} 个"
    )
    print(f"[INFO] 片段质量报告: {QUALITY_REPORT_FILE}")
    return accepted_chunks


def ensure_database() -> None:
    """
    确保 vector_db 数据库存在且 pgvector 扩展已启用。

    1. 连接 postgres 默认库 → 创建 vector_db (如不存在)
    2. 连接 vector_db → 启用 vector 扩展
    """
    # --- 创建数据库 ---
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s;", (DB_NAME,)
        )
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{DB_NAME}";')
            print(f"[INFO] 已创建数据库: {DB_NAME}")
        else:
            print(f"[INFO] 数据库 {DB_NAME} 已存在")
        cur.close()
        conn.close()
    except psycopg2.OperationalError as e:
        print(f"[ERROR] 无法连接 PostgreSQL (创建数据库): {e}")
        print("[HINT]  请检查 PostgreSQL 服务是否运行，用户名/密码是否正确。")
        raise

    # --- 启用 pgvector 扩展 ---
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("[INFO] pgvector 扩展已就绪 (CREATE EXTENSION IF NOT EXISTS)")
        cur.close()
        conn.close()
    except psycopg2.OperationalError as e:
        print(f"[ERROR] 无法连接 vector_db: {e}")
        raise


# PGVector 元数据建表 SQL（与 langchain_community PGVector 兼容）
_CREATE_COLLECTION_TABLE = """
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    cmetadata JSONB
);
"""

_CREATE_EMBEDDING_TABLE = """
CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
    embedding vector,
    document VARCHAR,
    cmetadata JSONB
);
"""


def _ensure_pgvector_tables(conn) -> None:
    """检查并自动创建 PGVector 所需的元数据表与 collection 记录。"""
    with conn.cursor() as cur:
        cur.execute(_CREATE_COLLECTION_TABLE)
        cur.execute(_CREATE_EMBEDDING_TABLE)
        cur.execute(
            "INSERT INTO langchain_pg_collection (name) VALUES (%s) ON CONFLICT DO NOTHING;",
            (DB_COLLECTION,),
        )
    conn.commit()


def get_existing_doc_ids(store: PGVector) -> set:
    """
    查询 PGVector 中已存储的 doc_id 集合（用于去重）。

    如果 PGVector 元数据表尚不存在，则自动创建表结构与 collection 记录。

    Args:
        store: PGVector 向量存储实例

    Returns:
        set: 已有的 doc_id 集合
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME,
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT cmetadata->>'doc_id' FROM langchain_pg_embedding "
            "WHERE collection_id = ("
            "  SELECT uuid FROM langchain_pg_collection WHERE name = %s"
            ");",
            (DB_COLLECTION,),
        )
        existing = {row[0] for row in cur.fetchall() if row[0]}
        cur.close()
        conn.close()
        return existing
    except psycopg2.errors.UndefinedTable:
        # 表不存在 → 自动建表，返回空集合
        print("[INFO] 检测到 PGVector 元数据表未创建，自动建表中...")
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                dbname=DB_NAME,
            )
            _ensure_pgvector_tables(conn)
            conn.close()
            print("[INFO] PGVector 元数据表创建成功")
        except Exception as init_err:
            print(f"[WARN] 自动建表失败: {init_err}")
        return set()
    except psycopg2.Error as e:
        # 其他数据库错误（连接、权限等），降级返回空集合
        print(f"[WARN] 查询已有记录失败（数据库错误）: {e}")
        return set()
    except Exception as e:
        print(f"[WARN] 查询已有记录失败（未知错误）: {e}")
        return set()


def save_chunks_cache(chunks: list) -> None:
    """
    保存文本块到本地 JSON 缓存文件，供 hybrid_retriever.py 中的 BM25 使用。

    Args:
        chunks: 文档块列表
    """
    cache_data = []
    for chunk in chunks:
        cache_data.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        })

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 文本块缓存已保存到 {CACHE_FILE} ({len(cache_data)} 条)")


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    """主入口：加载 → 分割 → 嵌入 → 存储 → 缓存。"""
    force = "--force" in sys.argv
    global USE_MINERU
    USE_MINERU = USE_MINERU or "--mineru" in sys.argv

    print("=" * 60)
    print("  文档向量存储 — LangChain + PostgreSQL pgvector")
    print("=" * 60)

    # --------------------------------------------------
    # Step 1: 确保文档目录存在
    # --------------------------------------------------
    print("\n[STEP 1/6] 检查文档目录...")
    if not ensure_docs_directory():
        # 文档目录不存在，停止加载加载文档
        return


    # --------------------------------------------------
    # Step 2: 加载文档
    # --------------------------------------------------
    print("\n[STEP 2/6] 加载文档...")
    if USE_MINERU and _HAS_MINERU:
        print("[INFO] PDF 提取模式: MinerU 优先，失败时回退 pdfplumber")
    elif USE_MINERU:
        print("[WARN] MinerU 依赖不可用，PDF 将使用 pdfplumber")
    else:
        print("[INFO] PDF 提取模式: pdfplumber")
    documents = load_documents()

    # --------------------------------------------------
    # Step 3: 分割文档
    # --------------------------------------------------
    print("\n[STEP 3/6] 分割文档...")
    chunks = split_documents(documents)
    if not chunks:
        print("[ERROR] 没有任何片段通过质量检查，保留现有数据库和缓存不变。")
        sys.exit(1)

    # --------------------------------------------------
    # Step 4: 确保数据库和扩展就绪
    # --------------------------------------------------
    print("\n[STEP 4/6] 确保 PostgreSQL 数据库就绪...")
    try:
        ensure_database()
    except psycopg2.OperationalError:
        print("[HINT] 请确认:")
        print("  1. PostgreSQL 服务正在运行")
        print("  2. pg_hba.conf 允许本地连接")
        print(f"  3. 用户 '{DB_USER}' 密码正确")
        sys.exit(1)

    # --------------------------------------------------
    # Step 5: 生成嵌入向量并存储到 PGVector
    # --------------------------------------------------
    print("\n[STEP 5/6] 生成嵌入向量并存储...")
    print(f"[INFO] 加载嵌入模型: {EMBEDDING_MODEL}")
    print("[INFO] 首次运行将下载模型 (~90MB)，请耐心等待...")

    # 初始化嵌入模型 (CPU 运行，避免 CUDA 依赖)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"[ERROR] 嵌入模型加载失败: {e}")
        print("[HINT] 请检查网络连接，确保能下载 HuggingFace 模型。")
        sys.exit(1)

    # 连接到 PGVector
    try:
        if _PGVECTOR_BACKEND == "langchain_postgres":
            store = PGVector(
                connection=CONNECTION_STRING,
                embeddings=embeddings,
                collection_name=DB_COLLECTION,
            )
        else:
            store = PGVector(
                connection_string=CONNECTION_STRING,
                embedding_function=embeddings,
                collection_name=DB_COLLECTION,
            )
        print(f"[INFO] PGVector 连接成功 → {DB_HOST}:{DB_PORT}/{DB_NAME}")
    except Exception as e:
        print(f"[ERROR] PGVector 连接失败: {e}")
        sys.exit(1)

    # 去重：查询已存在的 doc_id
    chunks_to_store = chunks
    if not force:
        existing_ids = get_existing_doc_ids(store)
        if existing_ids:
            chunks_to_store = [
                c for c in chunks
                if c.metadata.get("doc_id") not in existing_ids
            ]
            skipped = len(chunks) - len(chunks_to_store)
            if skipped > 0:
                print(f"[INFO] 跳过 {skipped} 个已存在的块 (使用 --force 强制覆盖)")
        else:
            print("[INFO] 数据库中无已有记录，将全部写入")
    else:
        print("[INFO] --force 模式: 将覆盖所有已存在记录")
        try:
            store.delete_collection()
            print("[INFO] 已清空旧记录，重新创建集合")
            store.create_collection()
        except Exception as e:
            print(f"[WARN] 清空旧记录失败: {e}")

    if not chunks_to_store:
        print("[INFO] 所有块已存在，无需存储。")
        save_chunks_cache(chunks)
        print_summary(len(documents), len(chunks), 0)
        return

    # 批量写入向量
    try:
        ids = store.add_documents(chunks_to_store)
        print(f"[INFO] 成功写入 {len(ids)} 条向量记录")
    except Exception as e:
        print(f"[ERROR] 向量写入失败: {e}")
        sys.exit(1)

    # 验证嵌入维度
    try:
        test_dim = len(embeddings.embed_query("test"))
        print(f"[INFO] 嵌入向量维度: {test_dim}")
    except Exception:
        pass

    # --------------------------------------------------
    # Step 6: 保存文本块缓存 (供 BM25 使用)
    # --------------------------------------------------
    print("\n[STEP 6/6] 保存文本块缓存...")
    save_chunks_cache(chunks)

    # --------------------------------------------------
    # 完成
    # --------------------------------------------------
    print_summary(len(documents), len(chunks), len(chunks_to_store))


def print_summary(num_docs: int, num_chunks: int, num_new: int) -> None:
    """打印处理摘要。"""
    print("\n" + "=" * 60)
    print("  [OK] 处理完成!")
    print(f"  文档数:     {num_docs}")
    print(f"  总块数:     {num_chunks}")
    print(f"  新写入:     {num_new}")
    print(f"  数据库:     {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"  集合名:     {DB_COLLECTION}")
    print(f"  嵌入模型:   {EMBEDDING_MODEL}")
    print(f"  块大小:     {CHUNK_SIZE} (重叠: {CHUNK_OVERLAP})")
    print(f"  缓存文件:   {CACHE_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
