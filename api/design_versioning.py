"""设计版本快照、历史方案引用解析与续改任务约束。

本模块位于现有两级意图路由和材料设计主流程之间。它不改变 matched_result
的固定 JSON 结构，只在独立快照表中保存 ``design_id -> matched_result`` 映射，
并把自然语言中的“以上方案、第一次设计、方案V2”等表达解析为明确版本。
引用解析只负责定位方案；成分和三段工艺仍由原设计链路完整重设计。
"""

from __future__ import annotations

import copy
import json
import re
import threading
import uuid
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from langchain_core.messages import HumanMessage, SystemMessage

from official_llm_client import official_deepseek_sync
from prompt import (
    DESIGN_REFERENCE_RESOLVER_SYSTEM_PROMPT,
    DESIGN_REVISION_EXECUTION_PROMPT_TEMPLATE,
    DESIGN_TASK_NORMALIZER_SYSTEM_PROMPT,
)
from session_store import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


# matched_result 中用于续改任务校验的完整成分与性能字段。
COMPONENT_FIELDS = {
    "C", "SI", "MN", "P", "S", "N", "NB", "V", "TI", "AL", "ALS",
    "CU", "CR", "NI", "CO", "MO", "B",
}
PERFORMANCE_FIELDS = {"YS", "TS", "EL", "AKV"}


def _json_object(text: str) -> dict | None:
    """从模型正文中提取一个 JSON 对象。"""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def _invoke_deepseek_json(system_prompt: str, user_prompt: str) -> dict | None:
    """使用现有 DeepSeek V4 Flash 官方客户端返回严格 JSON。"""
    try:
        result = official_deepseek_sync.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
            max_tokens=2048,
            temperature=0,
            max_retries=1,
        )
        return _json_object(result.content)
    except Exception as exc:
        print(f"[设计上下文控制] DeepSeek结构化解析失败: {type(exc).__name__}: {exc}")
        return None


class DesignSnapshotStore:
    """成功设计方案的持久化快照存储，数据库异常时回退到进程内存。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._db_ready = False
        self._memory: dict[str, list[dict]] = {}

    def initialize(self) -> None:
        """幂等创建快照表；失败时不影响原设计流程。"""
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
                    CREATE TABLE IF NOT EXISTS design_snapshots (
                        design_id          UUID PRIMARY KEY,
                        session_id         VARCHAR(255) NOT NULL,
                        version_no         INTEGER NOT NULL,
                        parent_design_id   UUID NULL,
                        material_purpose   VARCHAR(64) NOT NULL,
                        target_grade       VARCHAR(64) NULL,
                        aim_thick          DOUBLE PRECISION NULL,
                        slab_thick         DOUBLE PRECISION NULL,
                        user_request       TEXT NOT NULL,
                        change_plan        JSONB NOT NULL DEFAULT '{}'::jsonb,
                        spec_result        JSONB NOT NULL DEFAULT '{}'::jsonb,
                        matched_result     JSONB NOT NULL,
                        fact_table         JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT uq_design_session_version UNIQUE(session_id, version_no)
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_design_snapshots_session
                    ON design_snapshots(session_id, version_no);
                """)
            conn.close()
            self._db_ready = True
            print("[设计快照] design_snapshots 表已就绪")
        except Exception as exc:
            self._db_ready = False
            print(
                "[设计快照] 持久化数据库不可用，当前进程使用内存快照: "
                f"{type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _normalize_row(row: dict | None) -> dict | None:
        if not row:
            return None
        normalized = dict(row)
        for key in ("design_id", "parent_design_id"):
            if normalized.get(key) is not None:
                normalized[key] = str(normalized[key])
        return normalized

    def list_snapshots(self, session_id: str) -> list[dict]:
        """按版本号升序列出当前会话全部成功设计。"""
        if self._db_ready:
            try:
                conn = psycopg2.connect(
                    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                    user=DB_USER, password=DB_PASSWORD, connect_timeout=5,
                )
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM design_snapshots WHERE session_id=%s ORDER BY version_no",
                        (session_id,),
                    )
                    rows = [self._normalize_row(row) for row in cur.fetchall()]
                conn.close()
                return [row for row in rows if row]
            except Exception as exc:
                self._db_ready = False
                print(f"[设计快照] 查询数据库失败，改用内存快照: {exc}")
        with self._lock:
            return copy.deepcopy(self._memory.get(session_id, []))

    def get_snapshot(self, session_id: str, design_id: str) -> dict | None:
        """按会话和UUID读取一份快照，防止跨会话引用。"""
        target = str(design_id or "").strip()
        if not target:
            return None
        return next(
            (row for row in self.list_snapshots(session_id) if row.get("design_id") == target),
            None,
        )

    def save_snapshot(
        self,
        *,
        session_id: str,
        material_purpose: str,
        target_grade: str | None,
        aim_thick: float | None,
        slab_thick: float | None,
        user_request: str,
        change_plan: dict,
        spec_result: dict,
        matched_result: dict,
        fact_table: list[dict],
        parent_design_id: str | None = None,
        design_id: str | None = None,
    ) -> dict:
        """保存最终成功快照并分配当前会话连续版本号。"""
        snapshot_id = str(design_id or uuid.uuid4())
        with self._lock:
            existing_rows = self.list_snapshots(session_id)
            version_no = max([int(row.get("version_no") or 0) for row in existing_rows] or [0]) + 1
            record = {
                "design_id": snapshot_id,
                "session_id": session_id,
                "version_no": version_no,
                "parent_design_id": parent_design_id,
                "material_purpose": material_purpose,
                "target_grade": target_grade,
                "aim_thick": aim_thick,
                "slab_thick": slab_thick,
                "user_request": user_request,
                "change_plan": copy.deepcopy(change_plan or {}),
                "spec_result": copy.deepcopy(spec_result or {}),
                "matched_result": copy.deepcopy(matched_result),
                "fact_table": copy.deepcopy(fact_table or []),
            }
            if self._db_ready:
                try:
                    conn = psycopg2.connect(
                        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                        user=DB_USER, password=DB_PASSWORD, connect_timeout=5,
                    )
                    with conn.cursor() as cur:
                        # 同一会话只允许一项后台设计任务；事务锁仍可防御意外并发。
                        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
                        cur.execute(
                            "SELECT COALESCE(MAX(version_no), 0) + 1 "
                            "FROM design_snapshots WHERE session_id=%s",
                            (session_id,),
                        )
                        version_no = int(cur.fetchone()[0])
                        record["version_no"] = version_no
                        cur.execute("""
                            INSERT INTO design_snapshots (
                                design_id, session_id, version_no, parent_design_id,
                                material_purpose, target_grade, aim_thick, slab_thick,
                                user_request, change_plan, spec_result, matched_result, fact_table
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """, (
                            snapshot_id, session_id, version_no, parent_design_id,
                            material_purpose, target_grade, aim_thick, slab_thick,
                            user_request, Json(change_plan or {}), Json(spec_result or {}),
                            Json(matched_result), Json(fact_table or []),
                        ))
                    conn.commit()
                    conn.close()
                    return record
                except Exception as exc:
                    self._db_ready = False
                    print(f"[设计快照] 数据库保存失败，当前版本仅存内存: {exc}")
            self._memory.setdefault(session_id, []).append(copy.deepcopy(record))
            return record


design_snapshot_store = DesignSnapshotStore()


_REFERENCE_WORDS = (
    "以上设计", "上述设计", "以上方案", "上述方案", "当前设计", "当前方案",
    "该设计", "该方案", "这个设计", "这个方案", "上一轮", "上一次", "前一轮",
    "前一次", "第一次设计", "原设计", "原方案", "已有设计", "基于前述",
    "继续调整", "继续优化", "方案V",
)
_MODIFICATION_WORDS = (
    "调整", "微调", "降低", "提高", "减少", "增加", "优化", "修改", "重新设计",
    "从新设计", "再设计", "保持", "不变",
)
_IMPLICIT_ACTIVE_MODIFICATION_WORDS = (
    "调整", "微调", "降低", "提高", "减少", "增加", "优化当前", "修改",
    "其他不变", "保持不变", "性能不变", "继续调整", "继续优化",
)


def is_design_continuation_request(user_message: str) -> bool:
    """判断当前 DESIGN 请求是否明确引用一份已有方案。"""
    text = str(user_message or "")
    return (
        any(word in text for word in _REFERENCE_WORDS)
        and any(word in text for word in _MODIFICATION_WORDS)
    ) or bool(re.search(r"design[_-]?[0-9a-f-]{8,}|方案\s*V\d+", text, re.IGNORECASE)) or (
        any(word in text for word in _MODIFICATION_WORDS)
        and bool(re.search(r"(?:X(?:42|46|52|56|60|65|70|80|90|100|120)|Q(?:355|390|420|460|500|550|620|690))\S{0,8}(?:设计|方案)", text, re.IGNORECASE))
    )


def extract_target_grade(text: str, purpose: str = "") -> str | None:
    """从用户文本提取管线钢X级或风电Q级牌号。"""
    upper = str(text or "").upper()
    if purpose == "管线钢":
        match = re.search(
            r"(?<![A-Z0-9])(X(?:42|46|52|56|60|65|70|80|90|100|120)(?:M|NG)?)(?![A-Z0-9])",
            upper,
        )
    else:
        match = re.search(
            r"(?<![A-Z0-9])(Q(?:355|390|420|460|500|550|620|690)(?:M[A-F]?|[A-F])?)(?![A-Z0-9])",
            upper,
        )
    return match.group(1) if match else None


def _extract_named_thickness(text: str, slab: bool = False) -> float | None:
    label = r"(?:板坯厚度|连铸坯厚度)" if slab else r"(?:目标厚度|成品厚度|钢板厚度|板厚|厚度)"
    cleaned = str(text or "")
    if not slab:
        cleaned = re.sub(r"(?:板坯厚度|连铸坯厚度)\D{0,12}\d+(?:\.\d+)?\s*(?:mm|毫米)", "", cleaned, flags=re.I)
    for pattern in (
        rf"{label}\D{{0,12}}(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        rf"(\d+(?:\.\d+)?)\s*(?:mm|毫米)\D{{0,12}}{label}",
    ):
        match = re.search(pattern, cleaned, re.I)
        if match:
            return float(match.group(1))
    return None


def _is_explicit_spec_change(text: str, slab: bool = False) -> bool:
    """区分“引用某厚度方案”和“把当前方案厚度改为新值”。"""
    label = r"(?:板坯厚度|连铸坯厚度)" if slab else r"(?:目标厚度|成品厚度|钢板厚度|板厚|厚度)"
    action = r"(?:改为|改成|调整为|调整到|变为|设为|修改为)"
    value = r"\d+(?:\.\d+)?\s*(?:mm|毫米)"
    return bool(re.search(rf"(?:{label}\D{{0,8}}{action}\D{{0,8}}{value}|{action}\D{{0,8}}{value}\D{{0,8}}{label})", str(text or ""), re.I))


def compact_snapshot_catalog(rows: list[dict]) -> list[dict]:
    """仅向引用解析模型提供非敏感的方案索引，不发送完整工艺JSON。"""
    return [
        {
            "design_id": row.get("design_id"),
            "version": f"V{row.get('version_no')}",
            "material": row.get("material_purpose"),
            "grade": row.get("target_grade"),
            "aim_thick": row.get("aim_thick"),
            "slab_thick": row.get("slab_thick"),
            "request_summary": str(row.get("user_request") or "")[:240],
        }
        for row in rows
    ]


def resolve_design_reference(
    session_id: str,
    user_message: str,
    purpose: str,
    active_design_id: str | None = None,
    explicit_design_id: str | None = None,
) -> dict:
    """按固定优先级确定当前请求引用的唯一成功设计版本。"""
    rows = design_snapshot_store.list_snapshots(session_id)
    implicit_active_continuation = bool(active_design_id) and any(
        word in str(user_message or "") for word in _IMPLICIT_ACTIVE_MODIFICATION_WORDS
    )
    if (
        not explicit_design_id
        and not is_design_continuation_request(user_message)
        and not implicit_active_continuation
    ):
        return {"mode": "new", "snapshot": None, "confidence": 1.0, "candidates": []}
    if not rows:
        return {
            "mode": "clarification", "snapshot": None, "confidence": 0.0,
            "candidates": [], "message": "当前会话还没有可引用的成功设计方案，请明确提供设计条件。",
        }

    text = str(user_message or "")
    candidates = [row for row in rows if row.get("material_purpose") == purpose] or rows

    # 1. 前端弹窗选中的设计ID或用户显式输入的UUID拥有最高优先级。
    if explicit_design_id:
        exact = next(
            (row for row in candidates if row.get("design_id") == explicit_design_id),
            None,
        )
        if exact:
            return {"mode": "modify", "snapshot": exact, "confidence": 1.0, "candidates": []}
        return {
            "mode": "clarification",
            "snapshot": None,
            "confidence": 0.0,
            "candidates": compact_snapshot_catalog(candidates),
            "message": "指定的设计版本不存在或不属于当前会话，请重新选择。",
        }
    active_before_filter = next(
        (row for row in candidates if row.get("design_id") == active_design_id),
        None,
    )

    uuid_match = re.search(r"[0-9a-f]{8}-[0-9a-f-]{27,}", text, re.I)
    if uuid_match:
        exact = next((row for row in candidates if row.get("design_id") == uuid_match.group(0)), None)
        if exact:
            return {"mode": "modify", "snapshot": exact, "confidence": 1.0, "candidates": []}

    version_match = re.search(r"方案\s*V(\d+)", text, re.I)
    if version_match:
        version_no = int(version_match.group(1))
        exact = next((row for row in candidates if int(row.get("version_no") or 0) == version_no), None)
        if exact:
            return {"mode": "modify", "snapshot": exact, "confidence": 1.0, "candidates": []}

    grade = extract_target_grade(text, purpose)
    aim_thick = _extract_named_thickness(text, slab=False)
    slab_thick = _extract_named_thickness(text, slab=True)
    explicitly_filtered = False
    if grade:
        candidates = [row for row in candidates if str(row.get("target_grade") or "").upper() == grade]
        explicitly_filtered = True
    if aim_thick is not None:
        candidates = [row for row in candidates if row.get("aim_thick") is not None and abs(float(row["aim_thick"]) - aim_thick) < 1e-6]
        explicitly_filtered = True
    if slab_thick is not None:
        candidates = [row for row in candidates if row.get("slab_thick") is not None and abs(float(row["slab_thick"]) - slab_thick) < 1e-6]
        explicitly_filtered = True

    if (
        not candidates
        and active_before_filter
        and (
            _is_explicit_spec_change(text, slab=False)
            or _is_explicit_spec_change(text, slab=True)
        )
    ):
        return {
            "mode": "modify",
            "snapshot": active_before_filter,
            "confidence": 1.0,
            "candidates": [],
        }

    if len(candidates) == 1:
        return {"mode": "modify", "snapshot": candidates[0], "confidence": 1.0, "candidates": []}

    if "第一次设计" in text and candidates:
        return {"mode": "modify", "snapshot": candidates[0], "confidence": 1.0, "candidates": []}
    if any(word in text for word in ("上一轮", "上一次", "最近方案", "最近设计")) and candidates:
        return {"mode": "modify", "snapshot": candidates[-1], "confidence": 1.0, "candidates": []}

    active = next((row for row in candidates if row.get("design_id") == active_design_id), None)
    if active and not explicitly_filtered:
        return {"mode": "modify", "snapshot": active, "confidence": 1.0, "candidates": []}

    catalog = compact_snapshot_catalog(candidates or rows)
    # 没有明确规格过滤、也没有活跃方案时，引用解析模型只看精简指纹目录。
    # 置信度不足仍必须弹窗，模型不能替代用户选择。
    if not explicitly_filtered and catalog:
        parsed = _invoke_deepseek_json(
            DESIGN_REFERENCE_RESOLVER_SYSTEM_PROMPT,
            json.dumps(
                {
                    "original_user_prompt": text,
                    "active_design_id": active_design_id,
                    "design_fingerprints": catalog,
                },
                ensure_ascii=False,
            ),
        )
        selected_id = str((parsed or {}).get("design_id") or "")
        confidence = _to_float((parsed or {}).get("confidence")) or 0.0
        selected = next(
            (row for row in candidates if row.get("design_id") == selected_id),
            None,
        )
        if selected and confidence >= 0.85:
            return {
                "mode": "modify",
                "snapshot": selected,
                "confidence": confidence,
                "candidates": [],
            }

    # 其余多个候选直接交给前端确认，选择结果会作为 explicit_design_id
    # 再次进入本函数并继续同一条原始请求。
    return {
        "mode": "clarification", "snapshot": None, "confidence": 0.0,
        "candidates": catalog,
        "message": "当前描述可能对应多份历史设计，请明确选择具体方案版本。",
    }


class DesignTaskNormalizationError(RuntimeError):
    """续改任务无法稳定解析时终止设计，避免退回关键词白名单。"""


class DesignRevisionValidationError(RuntimeError):
    """父子方案强约束在重试后仍未满足。"""


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_normalized_design_task(user_message: str, reference_snapshot: dict) -> dict:
    """把“修改以上设计”补全为独立标准任务，不替换原始用户提示词。"""
    parent_values = matched_result_values(reference_snapshot.get("matched_result") or {})
    target_aim_thick = reference_snapshot.get("aim_thick")
    target_slab_thick = reference_snapshot.get("slab_thick")
    if _is_explicit_spec_change(user_message, slab=False):
        target_aim_thick = _extract_named_thickness(user_message, slab=False)
    if _is_explicit_spec_change(user_message, slab=True):
        target_slab_thick = _extract_named_thickness(user_message, slab=True)
    parsed = _invoke_deepseek_json(
        DESIGN_TASK_NORMALIZER_SYSTEM_PROMPT,
        json.dumps(
            {
                "original_user_prompt": user_message,
                "reference_fingerprint": {
                    "version": f"V{reference_snapshot.get('version_no')}",
                    "application": reference_snapshot.get("material_purpose"),
                    "steel_grade": reference_snapshot.get("target_grade"),
                    "product_thickness_mm": reference_snapshot.get("aim_thick"),
                    "slab_thickness_mm": reference_snapshot.get("slab_thick"),
                },
                "parent_composition": {
                    field: parent_values.get(field) for field in sorted(COMPONENT_FIELDS)
                },
                "parent_performance": {
                    field: parent_values.get(field) for field in sorted(PERFORMANCE_FIELDS)
                },
                "candidate_composition_fields": sorted(COMPONENT_FIELDS),
            },
            ensure_ascii=False,
        ),
    )
    if not isinstance(parsed, dict):
        raise DesignTaskNormalizationError("标准任务解析模型未返回合法JSON")

    # ``selected_microalloy_fields`` 只保留为模型对本轮优化重点的说明，不能再
    # 作为后端固定减量集合。微合金减量往往需要通过 Mn、Cr、Mo 等其他元素和
    # 工艺协同补偿，不能把设计空间人为收窄为某几个字段的简单求和。
    selected_fields = []
    for value in parsed.get("selected_microalloy_fields") or []:
        field = str(value or "").upper().strip()
        if field in COMPONENT_FIELDS and field not in selected_fields:
            selected_fields.append(field)

    wants_microalloy_decrease = (
        "微合金" in user_message
        and any(word in user_message for word in ("降低", "减少", "下调", "减量"))
    )
    performance_non_decrease = any(
        word in user_message
        for word in (
            "性能不降低", "性能不能降低", "保持性能", "力学性能不变",
            "不影响力学性能", "性能不变",
        )
    )
    editable_scopes = [
        scope for scope in (parsed.get("editable_scopes") or [])
        if scope in {"composition", "heating", "rolling", "cooling", "performance"}
    ]
    if "重新设计" in user_message or "从新设计" in user_message or "再设计" in user_message:
        editable_scopes = ["composition", "heating", "rolling", "cooling", "performance"]

    task = {
        "original_user_prompt": user_message,
        "target_design_id": reference_snapshot.get("design_id"),
        "inherited_constraints": {
            "steel_grade": reference_snapshot.get("target_grade"),
            "product_thickness_mm": target_aim_thick,
            "slab_thickness_mm": target_slab_thick,
            "application": reference_snapshot.get("material_purpose"),
        },
        "optimization_targets": {
            # 此标记表达用户优化方向，后续由成分/工艺模型综合成分、性能和仿真
            # 结果决定具体减量与补偿方案；不再触发字段总量的硬编码后校验。
            "microalloy_direction": "decrease" if wants_microalloy_decrease else "unchanged",
            "composition": str((parsed.get("optimization_targets") or {}).get("composition") or "redesign"),
            "process": str((parsed.get("optimization_targets") or {}).get("process") or "redesign"),
        },
        "editable_scopes": editable_scopes or [
            "composition", "heating", "rolling", "cooling", "performance"
        ],
        "locked_constraints": ["steel_grade", "product_thickness", "slab_thickness"],
        "relative_performance_constraints": (
            {field: f">= parent.{field}" for field in sorted(PERFORMANCE_FIELDS)}
            if performance_non_decrease else {}
        ),
        "selected_microalloy_fields": selected_fields,
        "summary": str(parsed.get("summary") or "基于指定历史方案执行完整续改设计"),
    }
    return task


def build_resolved_design_request(task: dict) -> str:
    """为标准提取/RAG/SQL补全关键规格；不传整份父报告或matched_result。"""
    inherited = task.get("inherited_constraints") or {}
    return (
        f"{task.get('original_user_prompt') or ''}\n\n"
        "【已解析的历史设计引用】\n"
        f"材料用途：{inherited.get('application')}\n"
        f"目标牌号：{inherited.get('steel_grade')}\n"
        f"成品厚度：{inherited.get('product_thickness_mm')} mm\n"
        f"板坯厚度：{inherited.get('slab_thickness_mm')} mm\n"
        "【继承约束】\n"
        "本轮未要求改变上述规格，标准提取、RAG和SQL匹配不得覆盖这些条件。\n"
        f"【本轮优化说明】\n{task.get('summary') or ''}"
    )


def build_revision_execution_prompt(
    task: dict,
    reference_snapshot: dict,
    sql_reference: dict | None,
) -> str:
    """构造后置完整微调提示，明确父快照主基准与SQL参考的优先级。"""
    return DESIGN_REVISION_EXECUTION_PROMPT_TEMPLATE.format(
        original_user_prompt=task.get("original_user_prompt") or "",
        normalized_task_json=json.dumps(task, ensure_ascii=False, indent=2),
        parent_matched_result_json=json.dumps(
            reference_snapshot.get("matched_result") or {}, ensure_ascii=False
        ),
        sql_reference_json=json.dumps(sql_reference or {}, ensure_ascii=False),
    )


def matched_result_values(matched_result: dict) -> dict[str, Any]:
    """把单键字典列表摊平成大写字段映射，仅供版本差异校验。"""
    values: dict[str, Any] = {}
    for item in matched_result.get("arrBody") or []:
        if isinstance(item, dict) and len(item) == 1:
            key, value = next(iter(item.items()))
            values[str(key).upper()] = value
    return values


def _last_effective_pass_thickness(values: dict[str, Any]) -> float | None:
    """读取最后一个正值道次厚度，供续改最终硬门禁使用。"""
    last_value = None
    for index in range(1, 31):
        value = _to_float(values.get(f"N{index}_DH_CAL"))
        if value is not None and value > 0:
            last_value = value
    return last_value


def validate_revision_constraints(
    candidate: dict,
    reference_snapshot: dict,
    task: dict,
    spec_result: dict | None = None,
    *,
    require_final_pass: bool = False,
) -> list[str]:
    """统一校验父子规格、相对性能及当前标准范围。

    微合金元素如何减量、是否由其他合金元素补偿属于本轮模型的协同设计职责，
    不再使用预设字段集合的总量比较进行后端回退或阻断。
    """
    errors: list[str] = []
    current = matched_result_values(candidate)
    parent = matched_result_values(reference_snapshot.get("matched_result") or {})
    inherited = task.get("inherited_constraints") or {}

    for field, task_key, label in (
        ("AIM_THICK", "product_thickness_mm", "成品厚度"),
        ("SLAB_THICK", "slab_thickness_mm", "板坯厚度"),
    ):
        expected = _to_float(inherited.get(task_key))
        actual = _to_float(current.get(field))
        if expected is None or actual is None or abs(actual - expected) > 1e-6:
            errors.append(f"{label}{field}必须为{expected}，当前为{current.get(field)}")

    if task.get("relative_performance_constraints"):
        for field in sorted(PERFORMANCE_FIELDS):
            parent_value = _to_float(parent.get(field))
            current_value = _to_float(current.get(field))
            if parent_value is None or current_value is None:
                errors.append(f"{field}缺少可比较的父子方案性能值")
            elif current_value < parent_value - 1e-9:
                errors.append(
                    f"{field}不得低于父方案{parent_value:g}，当前为{current_value:g}"
                )

    for field in sorted(PERFORMANCE_FIELDS):
        value = _to_float(current.get(field))
        minimum = _to_float((spec_result or {}).get(f"{field}_min"))
        maximum = _to_float((spec_result or {}).get(f"{field}_max"))
        if value is None:
            errors.append(f"{field}缺少有效值")
            continue
        if minimum is not None and value < minimum - 1e-9:
            errors.append(f"{field}={value:g}低于当前标准下限{minimum:g}")
        if maximum is not None and maximum < 9999 and value > maximum + 1e-9:
            errors.append(f"{field}={value:g}高于当前标准上限{maximum:g}")

    if require_final_pass:
        target = _to_float(inherited.get("product_thickness_mm"))
        last_thickness = _last_effective_pass_thickness(current)
        if target is None or last_thickness is None or abs(last_thickness - target) > 1e-6:
            errors.append(
                f"最后有效轧制道次厚度必须为{target}，当前为{last_thickness}"
            )
    return errors
