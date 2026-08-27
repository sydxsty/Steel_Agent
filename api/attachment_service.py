# -*- coding: utf-8 -*-
"""聊天附件的临时存储、串行解析、取消和提示词拼接服务。"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES_PER_SESSION = 5
MAX_PDFS_PER_SESSION = 2
MAX_PROMPT_ATTACHMENT_CHARS = 200_000
STALE_SECONDS = 3600
SUPPORTED_EXTENSIONS = {
    ".docx", ".xlsx", ".pdf", ".md", ".markdown", ".txt",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
}


class AttachmentServiceError(RuntimeError):
    """可安全返回前端的附件业务错误。"""


@dataclass
class AttachmentRecord:
    """单个附件在当前后端进程中的运行状态。"""

    attachment_id: str
    session_id: str
    original_name: str
    extension: str
    task_dir: Path
    stored_name: str
    size: int = 0
    status: str = "uploading"
    progress: int = 0
    message: str = "正在上传"
    created_at: float = field(default_factory=time.time)
    process: asyncio.subprocess.Process | None = None
    cancelled: bool = False


class AttachmentManager:
    """管理附件生命周期，并保证同一会话每次只解析一个文件。"""

    def __init__(self, root: Path, parser_script: Path):
        self.root = root.resolve()
        self.parser_script = parser_script.resolve()
        self.records: dict[str, AttachmentRecord] = {}
        self.session_queues: dict[str, asyncio.Queue[str]] = {}
        self.session_workers: dict[str, asyncio.Task] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_session_folder(session_id: str) -> str:
        """会话ID只作为可读目录前缀，不能形成路径穿越。"""
        cleaned = "".join(char for char in str(session_id) if char.isalnum() or char in "-_")
        return (cleaned[:80] or "default")

    def reserve(self, session_id: str, original_name: str) -> AttachmentRecord:
        """在接收文件内容前预留UUID目录并执行数量、格式校验。"""
        session_id = str(session_id or "default").strip() or "default"
        original_name = Path(str(original_name or "")).name.strip()
        extension = Path(original_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise AttachmentServiceError(
                "不支持该附件格式；仅支持 DOCX、XLSX、PDF、Markdown、TXT 和常见图片"
            )
        active = [
            record for record in self.records.values()
            if record.session_id == session_id and not record.cancelled
        ]
        if len(active) >= MAX_FILES_PER_SESSION:
            raise AttachmentServiceError(f"每次最多上传 {MAX_FILES_PER_SESSION} 个附件")
        if extension == ".pdf" and sum(item.extension == ".pdf" for item in active) >= MAX_PDFS_PER_SESSION:
            raise AttachmentServiceError(f"每次最多上传 {MAX_PDFS_PER_SESSION} 个 PDF")

        attachment_id = str(uuid.uuid4())
        task_dir = self.root / self._safe_session_folder(session_id) / attachment_id
        source_dir = task_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=False)
        stored_name = "upload" + extension
        record = AttachmentRecord(
            attachment_id=attachment_id,
            session_id=session_id,
            original_name=original_name,
            extension=extension,
            task_dir=task_dir,
            stored_name=stored_name,
        )
        self.records[attachment_id] = record
        return record

    def finish_upload(self, record: AttachmentRecord, size: int) -> None:
        """落盘上传元数据，并把附件切换为等待解析状态。"""
        record.size = int(size)
        record.status = "queued"
        record.progress = 0
        record.message = "等待解析"
        metadata = {
            "attachment_id": record.attachment_id,
            "session_id": record.session_id,
            "original_name": record.original_name,
            "stored_name": record.stored_name,
            "size": record.size,
            "created_at": record.created_at,
        }
        (record.task_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        self._write_status(record, "queued", 0, "等待解析")

    @staticmethod
    def _write_status(record: AttachmentRecord, status: str, progress: int, message: str) -> None:
        """主进程写入状态；解析子进程会使用相同结构更新该文件。"""
        payload = {"status": status, "progress": progress, "message": message}
        temporary = record.task_dir / "status.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, record.task_dir / "status.json")

    async def enqueue(self, attachment_id: str, session_id: str) -> AttachmentRecord:
        """将附件加入会话串行队列；重复请求不会启动第二个解析进程。"""
        record = self.require_record(attachment_id, session_id)
        if record.status in {"parsing", "ready"}:
            return record
        if record.status not in {"queued", "error"}:
            raise AttachmentServiceError("附件当前状态不能开始解析")
        record.status = "queued"
        record.message = "等待解析"
        queue = self.session_queues.setdefault(record.session_id, asyncio.Queue())
        await queue.put(record.attachment_id)
        worker = self.session_workers.get(record.session_id)
        if worker is None or worker.done():
            self.session_workers[record.session_id] = asyncio.create_task(
                self._run_session_queue(record.session_id)
            )
        return record

    async def _run_session_queue(self, session_id: str) -> None:
        """逐个执行当前会话附件，避免多个PDF/OCR同时占满CPU和内存。"""
        queue = self.session_queues[session_id]
        while not queue.empty():
            attachment_id = await queue.get()
            record = self.records.get(attachment_id)
            try:
                if record is None or record.cancelled:
                    continue
                record.status = "parsing"
                record.progress = 5
                record.message = "正在启动解析器"
                self._write_status(record, record.status, record.progress, record.message)
                creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                record.process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(self.parser_script),
                    "--task-dir",
                    str(record.task_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=creation_flags,
                )
                stdout, stderr = await record.process.communicate()
                if record.cancelled:
                    continue
                status = self._read_status_file(record)
                record.status = status.get("status", "error")
                record.progress = int(status.get("progress", 100))
                record.message = str(status.get("message", ""))
                if record.process.returncode != 0 and record.status != "error":
                    detail = stderr.decode("utf-8", errors="replace").strip()
                    record.status = "error"
                    record.message = detail[-500:] or "附件解析进程异常退出"
            except asyncio.CancelledError:
                if record is not None:
                    await self._terminate_process(record)
                raise
            except Exception as exc:
                if record is not None and not record.cancelled:
                    record.status = "error"
                    record.progress = 100
                    record.message = str(exc)
                    self._write_status(record, "error", 100, record.message)
            finally:
                if record is not None:
                    record.process = None
                queue.task_done()

    @staticmethod
    def _read_status_file(record: AttachmentRecord) -> dict:
        try:
            return json.loads((record.task_dir / "status.json").read_text(encoding="utf-8"))
        except Exception:
            return {
                "status": record.status,
                "progress": record.progress,
                "message": record.message,
            }

    def require_record(self, attachment_id: str, session_id: str) -> AttachmentRecord:
        """校验附件存在且属于当前会话，阻止跨会话读取。"""
        record = self.records.get(str(attachment_id))
        if record is None or record.session_id != str(session_id):
            raise AttachmentServiceError("附件不存在或不属于当前会话")
        return record

    def status_payload(self, attachment_id: str, session_id: str) -> dict:
        """合并内存状态与子进程状态文件，返回前端轮询数据。"""
        record = self.require_record(attachment_id, session_id)
        if record.status == "parsing" and not record.cancelled:
            disk_status = self._read_status_file(record)
            disk_state = str(disk_status.get("status", record.status))
            # 子进程先原子写出 attachment.md/status.json，再从操作系统退出。
            # 在退出确认前不向前端暴露 ready，避免发送动作与最后一次文件写入竞态。
            if disk_state == "ready" and record.process is not None and record.process.returncode is None:
                record.status = "parsing"
                record.progress = 99
                record.message = "正在完成解析"
            else:
                record.status = disk_state
                record.progress = int(disk_status.get("progress", record.progress))
                record.message = str(disk_status.get("message", record.message))
        return {
            "attachment_id": record.attachment_id,
            "name": record.original_name,
            "size": record.size,
            "status": record.status,
            "progress": record.progress,
            "message": record.message,
        }

    async def cancel(self, attachment_id: str, session_id: str) -> None:
        """取消附件并终止解析进程树，然后删除该附件全部临时文件。"""
        record = self.records.get(str(attachment_id))
        if record is None:
            return
        if record.session_id != str(session_id):
            raise AttachmentServiceError("附件不属于当前会话")
        record.cancelled = True
        record.status = "cancelled"
        await self._terminate_process(record)
        self.records.pop(record.attachment_id, None)
        await asyncio.to_thread(shutil.rmtree, record.task_dir, True)
        self._remove_empty_session_dir(record.task_dir.parent)

    @staticmethod
    def _remove_empty_session_dir(session_dir: Path) -> None:
        """附件清空后顺手删除空会话目录；目录非空时保持不动。"""
        try:
            session_dir.rmdir()
        except OSError:
            pass

    @staticmethod
    async def _terminate_process(record: AttachmentRecord) -> None:
        process = record.process
        if process is None or process.returncode is not None:
            return
        if os.name == "nt":
            await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def build_prompt_and_consume(
        self,
        session_id: str,
        attachment_ids: list[str],
        user_message: str,
    ) -> str:
        """读取就绪Markdown、按公平额度截断，并在当前轮使用后清理附件。"""
        ids = [str(item) for item in attachment_ids if str(item).strip()]
        if not ids:
            return user_message
        if len(ids) > MAX_FILES_PER_SESSION or len(set(ids)) != len(ids):
            raise AttachmentServiceError("附件ID数量无效或包含重复项")
        records = [self.require_record(item, session_id) for item in ids]
        not_ready = [record.original_name for record in records if record.status != "ready"]
        if not_ready:
            raise AttachmentServiceError("以下附件尚未解析完成：" + "、".join(not_ready))

        prefix = "\n\n【附件内容开始】\n\n"
        suffix = "\n\n【附件内容结束】"
        # 预留各附件区块之间的两个换行，确保最终附件区始终包含完整结束标记。
        fixed_length = len(prefix) + len(suffix) + max(0, len(records) - 1) * 2
        quota = max(1, (MAX_PROMPT_ATTACHMENT_CHARS - fixed_length) // len(records))
        sections: list[str] = []
        for index, record in enumerate(records, start=1):
            markdown_path = record.task_dir / "attachment.md"
            if not markdown_path.exists():
                raise AttachmentServiceError(f"附件解析结果已丢失：{record.original_name}")
            content = markdown_path.read_text(encoding="utf-8").strip()
            heading = f"## 附件 {index}：{record.original_name}\n\n"
            truncation = "\n\n> 附件内容过长，超过当前轮可用额度，后续内容已截断。"
            body_limit = max(1, quota - len(heading))
            if len(content) > body_limit:
                content = content[:max(1, body_limit - len(truncation))] + truncation
            sections.append(heading + content)

        attachment_block = prefix + "\n\n".join(sections) + suffix
        if len(attachment_block) > MAX_PROMPT_ATTACHMENT_CHARS:
            attachment_block = (
                attachment_block[:MAX_PROMPT_ATTACHMENT_CHARS - len(suffix)] + suffix
            )
        effective_message = user_message + attachment_block

        # 内容已经复制进当前后台计算任务，此后立即清理，附件不会进入下一轮。
        for record in records:
            self.records.pop(record.attachment_id, None)
            await asyncio.to_thread(shutil.rmtree, record.task_dir, True)
            self._remove_empty_session_dir(record.task_dir.parent)
        return effective_message

    async def cleanup_stale(self) -> None:
        """删除超过一小时的遗留任务；运行中的进程也会被可靠终止。"""
        cutoff = time.time() - STALE_SECONDS
        stale = [record for record in self.records.values() if record.created_at < cutoff]
        for record in stale:
            await self.cancel(record.attachment_id, record.session_id)
        for session_dir in self.root.iterdir() if self.root.exists() else []:
            if not session_dir.is_dir():
                continue
            for task_dir in session_dir.iterdir():
                try:
                    if task_dir.is_dir() and task_dir.stat().st_mtime < cutoff:
                        await asyncio.to_thread(shutil.rmtree, task_dir, True)
                except OSError:
                    continue

    async def shutdown(self) -> None:
        """服务退出时停止所有解析进程和队列工作任务。"""
        for record in list(self.records.values()):
            if record.process is not None:
                record.cancelled = True
                await self._terminate_process(record)
        workers = [worker for worker in self.session_workers.values() if not worker.done()]
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)


ATTACHMENT_ROOT = Path(__file__).resolve().parent / "attachment_workspace"
attachment_manager = AttachmentManager(
    ATTACHMENT_ROOT,
    Path(__file__).resolve().parent / "attachment_parser.py",
)
