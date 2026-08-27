"""管线钢与风电用钢四类设计智能体的独立执行入口。

本模块只保存智能体流程，不负责 FastAPI 路由、前端 NDJSON 流式编排或最终报告生成。
所有项目级依赖均由 ``api.py`` 通过入口参数传入，避免本模块反向导入 ``api.py``
形成循环依赖，也便于使用可控替身验证迁移前后的执行效果。
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, MutableMapping

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from design_versioning import (
    DesignRevisionValidationError,
    build_revision_execution_prompt,
    validate_revision_constraints,
)

from prompt import (
    DESIGN_CHANGE_ASSESSMENT_SYSTEM_PROMPT,
    PIPELINE_REFINEMENT_ALL_REPAIR_SCOPE_PROMPT,
    PIPELINE_REFINEMENT_COMPOSITION_REPAIR_SCOPE_PROMPT,
    PIPELINE_REFINEMENT_ROLLING_REPAIR_SCOPE_PROMPT,
    PIPELINE_REFINEMENT_USER_PROMPT,
    REQUIREMENT_PARSING_SYSTEM_PROMPT,
    WIND_POWER_REFINEMENT_PROCESS_RULE,
    build_design_change_assessment_user_prompt,
    build_pipeline_refinement_prompt,
    build_pipeline_refinement_repair_prompt,
    build_requirement_parsing_user_prompt,
)


class WindPowerDesignValidationError(RuntimeError):
    """风电成分、性能或轧制规程多轮重设计仍未通过强制校验。"""


class DesignChangeAssessmentError(RuntimeError):
    """设计变更评估 Agent 未能返回完整、可验证的结构化结论。"""


class CompositionRefinementValidationError(RuntimeError):
    """微调 Agent 多轮设计后仍未通过确定性后校验。"""


class RequirementParsingError(RuntimeError):
    """需求解析 Agent 连续两次未返回符合协议的结构化需求。"""


class _RequirementPerformance(BaseModel):
    """用户明确给出的性能目标；空字段不会被写入统一需求 JSON。"""

    model_config = ConfigDict(extra="forbid")

    YS: float | None = None
    TS: float | None = None
    EL: float | None = None
    AKV: float | None = None
    impact_temperature: str = ""
    Pcm_max: float | None = None
    CEV_max: float | None = None
    corrosion_rate_max: str = ""
    other: list[str] = Field(default_factory=list)


class DesignRequirement(BaseModel):
    """管线钢与风电用钢后续设计链共用的统一用户需求结构。"""

    model_config = ConfigDict(extra="forbid")

    application: Literal[
        "pipeline", "offshore_wind", "onshore_wind", "wind_power"
    ]
    steel_grade: str = ""
    thickness_mm: float | None = None
    slab_thickness_mm: float | None = None
    performance: _RequirementPerformance = Field(
        default_factory=_RequirementPerformance
    )
    requirements: list[str] = Field(default_factory=list)
    composition_constraints: list[str] = Field(default_factory=list)
    process_constraints: list[str] = Field(default_factory=list)
    explicit_constraints: list[str] = Field(default_factory=list)
    other_constraints: list[str] = Field(default_factory=list)
    references_previous_design: bool = False


@dataclass(frozen=True)
class RequirementParsingDependencies:
    """需求解析 Agent 的可替换依赖，便于复用项目模型并进行隔离测试。"""

    agent_model: Any
    create_agent_fn: Callable[..., Any] = create_agent


def parse_design_requirement(
    *,
    user_message: str,
    purpose: str,
    session_context: str,
    dependencies: RequirementParsingDependencies,
) -> dict:
    """把自然语言设计请求解析为稳定的 Requirement JSON。

    调用位置固定在 DESIGN/CHAT 和材料用途分类完成之后、管线钢/风电用钢
    设计分支开始之前。该 Agent 不开放 RAG、MySQL 或校验工具，避免资料中的
    数值污染用户真实需求；最近会话只用于补足“以上方案”等承接语义。

    结构化协议由 Pydantic 强制校验。第一次输出无效时追加纠错指令重试一次；
    第二次仍失败则抛出 ``RequirementParsingError``，调用方必须停止本轮设计，
    不能用默认牌号、默认厚度或空 JSON 静默继续。
    """
    if purpose not in {"管线钢", "风电用钢"}:
        raise RequirementParsingError(f"不支持的设计用途：{purpose or '空'}")

    user_prompt = build_requirement_parsing_user_prompt(
        user_message=user_message,
        purpose=purpose,
        session_context=session_context,
    )
    last_error = ""
    for attempt in range(1, 3):
        retry_instruction = (
            ""
            if attempt == 1
            else "\n上一次结构化输出不符合协议。本次只提取输入中有依据的字段，禁止补造数值。"
        )
        try:
            agent = dependencies.create_agent_fn(
                model=dependencies.agent_model,
                tools=[],
                system_prompt=REQUIREMENT_PARSING_SYSTEM_PROMPT + retry_instruction,
                response_format=ToolStrategy(DesignRequirement),
                name="steel_design_requirement_parser",
            )
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_prompt}]},
                config={"recursion_limit": 6},
            )
            structured = (
                response.get("structured_response")
                if isinstance(response, dict)
                else None
            )
            requirement = (
                structured
                if isinstance(structured, DesignRequirement)
                else DesignRequirement.model_validate(structured)
            )
            expected_applications = (
                {"pipeline"}
                if purpose == "管线钢"
                else {"offshore_wind", "onshore_wind", "wind_power"}
            )
            if requirement.application not in expected_applications:
                last_error = (
                    f"用途不一致：purpose={purpose}, "
                    f"application={requirement.application}"
                )
                continue
            # exclude_none 保持 JSON 紧凑；空字符串和空数组保留，以明确区分
            # “用户未提出”与解析过程遗漏/失败。
            return requirement.model_dump(mode="json", exclude_none=True)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    raise RequirementParsingError(
        "需求解析 Agent 连续两次未返回有效 Requirement JSON：" + last_error
    )


def build_unified_design_user_message(
    user_message_raw: str,
    requirement: dict,
    *,
    resolved_reference_context: str = "",
) -> str:
    """生成供所有后续设计 Agent 共用的统一 ``USER_MESSAGE``。

    原始提示词不被结构化 JSON 替换，而是原样保留在首段，确保模型仍能读取
    比较符号、自然语言细节和附件文本。Requirement JSON 作为第二段提供稳定的
    用途、牌号、厚度与性能字段。续改流程完成父方案定位后，可通过第三段追加
    已解析的继承规格；该段只补充上下文，不覆盖前两段。
    """
    sections = [
        "USER_MESSAGE:",
        "用户需求:\n" + str(user_message_raw or "").strip(),
        "结构化需求:\n"
        + json.dumps(requirement or {}, ensure_ascii=False, indent=2),
    ]
    if str(resolved_reference_context or "").strip():
        sections.append(
            "已解析的续改上下文:\n"
            + str(resolved_reference_context).strip()
        )
    return "\n\n".join(sections)


class _AssessmentTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steel_grade: str = ""
    thickness_mm: float | None = None


class _ModuleAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["INHERIT", "REASSESS", "REDESIGN"]
    priority: Literal["LOW", "MEDIUM", "HIGH"]
    reasons: list[str] = Field(min_length=1)


class _ChangeAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    composition: _ModuleAssessment
    heating: _ModuleAssessment
    rolling: _ModuleAssessment
    cooling: _ModuleAssessment
    performance_requirement: _ModuleAssessment


class _AssessmentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rag: list[dict[str, Any]] = Field(default_factory=list)
    historical_data: list[dict[str, Any]] = Field(default_factory=list)


class DesignChangeAssessment(BaseModel):
    """管线钢与风电用钢共用的设计变更评估结构。"""

    model_config = ConfigDict(extra="forbid")

    task_type: str = Field(min_length=1)
    reference: _AssessmentTarget
    target: _AssessmentTarget
    change_assessment: _ChangeAssessment
    evidence: _AssessmentEvidence


@dataclass(frozen=True)
class DesignChangeAssessmentDependencies:
    """设计变更评估 Agent 的模型和两个受产品边界约束的检索入口。"""

    agent_model: Any
    retrieve_product_knowledge: Callable[[str], Any]
    retrieve_current_target_history: Callable[[], Any]
    create_agent_fn: Callable[..., Any] = create_agent


def _json_tool_result(value: Any, *, unavailable_message: str) -> str:
    if value is None or value == "" or value == []:
        return json.dumps(
            {"status": "unavailable", "message": unavailable_message},
            ensure_ascii=False,
        )
    if isinstance(value, str):
        return json.dumps(
            {"status": "ok", "content": value},
            ensure_ascii=False,
        )
    return json.dumps(
        {"status": "ok", "data": value},
        ensure_ascii=False,
        default=str,
    )


def assess_design_change(
    *,
    material_name: str,
    user_message: str,
    session_context: str,
    spec_result: dict,
    reference_summary: dict,
    target_summary: dict,
    engineering_standard_context: dict | None,
    matched_result_summary: dict,
    dependencies: DesignChangeAssessmentDependencies,
) -> dict:
    """运行真正的 LangChain Agent，并要求其完成两类检索后结构化评估。"""

    tool_usage = {"rag": False, "historical_data": False}

    @tool("search_product_knowledge")
    def search_product_knowledge(query: str) -> str:
        """检索当前产品专属知识库；不能切换到其他钢种知识库。"""
        tool_usage["rag"] = True
        try:
            return _json_tool_result(
                dependencies.retrieve_product_knowledge(query),
                unavailable_message="当前产品知识库未返回资料",
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "unavailable",
                    "message": f"当前产品知识库检索失败：{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )

    @tool("search_current_target_history")
    def search_current_target_history() -> str:
        """读取后端锁定的当前目标厚度历史实绩；不能传入或改写目标厚度。"""
        tool_usage["historical_data"] = True
        try:
            return _json_tool_result(
                dependencies.retrieve_current_target_history(),
                unavailable_message="当前目标规格未取得历史实绩",
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "unavailable",
                    "message": f"当前目标历史实绩检索失败：{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )

    user_prompt = build_design_change_assessment_user_prompt(
        material_name=material_name,
        user_message=user_message,
        session_context=session_context,
        spec_result=spec_result,
        reference_summary=reference_summary,
        target_summary=target_summary,
        engineering_standard_context=engineering_standard_context or {},
        matched_result_summary=matched_result_summary,
    )
    last_error = ""
    for attempt in range(1, 3):
        tool_usage["rag"] = False
        tool_usage["historical_data"] = False
        retry_instruction = (
            ""
            if attempt == 1
            else "\n上一次未完成两个必需工具调用或结构化输出无效；本次必须先调用两个工具再输出。"
        )
        try:
            agent = dependencies.create_agent_fn(
                model=dependencies.agent_model,
                tools=[search_product_knowledge, search_current_target_history],
                system_prompt=DESIGN_CHANGE_ASSESSMENT_SYSTEM_PROMPT + retry_instruction,
                response_format=ToolStrategy(DesignChangeAssessment),
                name="steel_design_change_assessment",
            )
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_prompt}]},
                config={"recursion_limit": 12},
            )
            structured = (
                response.get("structured_response")
                if isinstance(response, dict)
                else None
            )
            assessment = (
                structured
                if isinstance(structured, DesignChangeAssessment)
                else DesignChangeAssessment.model_validate(structured)
            )
            missing_tools = [name for name, used in tool_usage.items() if not used]
            if missing_tools:
                last_error = "未调用必需工具：" + "、".join(missing_tools)
                continue
            return assessment.model_dump(mode="json")
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    raise DesignChangeAssessmentError(
        "设计变更评估 Agent 连续两次未返回有效结论：" + (last_error or "未知错误")
    )


_TURN_WIDTH_ERROR_PREFIXES = (
    "转钢道次标识无效",
    "转钢宽度变化次数无效",
    "转钢宽度变化位置无效",
    "转钢标记与宽度变化不一致",
    "WIDTH_ROLL_START_REMARK=",
    "WIDTH_ROLL_END_REMARK=",
)


def _extract_boundary_repair_fields(
    validation_errors: list[str],
    component_performance_fields: set[str],
) -> set[str]:
    """从后端字段边界错误中识别必须回到成分/性能范围修复的字段。

    例如 ``MO`` 越界发生在上一轮轧制规程修复之后时，不能继续沿用“只改轧制”
    的范围，否则模型会被提示词禁止修改 MO 而无限重复同一错误。
    """
    invalid_fields: set[str] = set()
    for error in validation_errors:
        match = re.search(r"字段\s+([A-Za-z][A-Za-z0-9_]*)\s*=", str(error or ""))
        if match:
            field_name = match.group(1).upper()
            if field_name in component_performance_fields:
                invalid_fields.add(field_name)
    return invalid_fields


def _pin_boundary_repair_baseline(
    matched_result: dict,
    validation_errors: list[str],
    component_fields: set[str],
    performance_fields: set[str],
) -> dict:
    """把越界字段钉在最近合法边界，作为下一轮 LLM 的输入基线。

    仅把错误文字附在长提示词末尾时，模型可能继续照抄本轮无效值。这里从
    ``MO >= 0.1，MO <= 0.3`` 一类后端错误中读取边界：低于下限时写入下限，
    高于上限时写入上限。模型仍可在合法区间内重新设计，但不会再以历史越界值
    作为默认值反复提交。
    """
    repaired = copy.deepcopy(matched_result)
    values: dict[str, float] = {}
    for item in repaired.get("arrBody") or []:
        if isinstance(item, dict) and len(item) == 1:
            key, raw_value = next(iter(item.items()))
            try:
                values[str(key).upper()] = float(raw_value)
            except (TypeError, ValueError):
                continue

    bounds_pattern = re.compile(
        r"字段\s+([A-Za-z][A-Za-z0-9_]*)=.*?允许范围："
        r"(?:\1\s*>=\s*([-+]?\d+(?:\.\d+)?))?"
        r"(?:，\1\s*<=\s*([-+]?\d+(?:\.\d+)?))?",
    )
    repaired_values: dict[str, float] = {}
    for error in validation_errors:
        match = bounds_pattern.search(str(error or ""))
        if not match:
            continue
        field_name = match.group(1).upper()
        lower = float(match.group(2)) if match.group(2) is not None else None
        upper = float(match.group(3)) if match.group(3) is not None else None
        current = values.get(field_name)
        if current is None:
            safe_value = lower if lower is not None else upper
        elif lower is not None and current < lower:
            safe_value = lower
        elif upper is not None and current > upper:
            safe_value = upper
        else:
            safe_value = current
        if safe_value is not None:
            repaired_values[field_name] = safe_value

    if not repaired_values:
        return repaired

    repaired_body = []
    for item in repaired.get("arrBody") or []:
        if not isinstance(item, dict) or len(item) != 1:
            repaired_body.append(item)
            continue
        key, raw_value = next(iter(item.items()))
        field_name = str(key).upper()
        safe_value = repaired_values.get(field_name)
        if safe_value is None:
            repaired_body.append(item)
        elif field_name in component_fields:
            repaired_body.append({key: f"{safe_value:.4f}"})
        elif field_name in performance_fields:
            repaired_body.append({key: str(int(round(safe_value)))})
        else:
            repaired_body.append({key: raw_value})
    repaired["arrBody"] = repaired_body
    return repaired


def _remove_refinement_turn_width_errors(errors: list[str]) -> list[str]:
    """后置微调不校验转钢标记；该职责完整留给后续轧制智能体。"""
    return [
        error for error in errors
        if not str(error or "").startswith(_TURN_WIDTH_ERROR_PREFIXES)
    ]


@dataclass(frozen=True)
class CompositionRefinementDependencies:
    """成分、性能及初步轧制规程微调入口使用的项目依赖。

    字段均对应迁移前 ``api.py`` 中已经存在的函数、缓存或常量。入口只调用这些
    依赖，不改变其实现，因此数据库、模型、校验及缓存行为继续由原项目控制。
    """

    extract_target_thickness: Callable[[str], float | None]
    extract_target_slab_thickness: Callable[[str], float | None]
    lock_explicit_thickness_targets: Callable[[dict, float | None, float | None], dict]
    is_context_modification_request: Callable[[str], bool]
    build_refinement_rag_context: Callable[[dict, str], str]
    build_cross_route_context: Callable[[str], str]
    get_recent_session_context: Callable[[str], str]
    filter_wind_session_context: Callable[[str], str]
    component_fields: frozenset[str]
    performance_fields: frozenset[str]
    roll_fields: frozenset[str]
    get_arrbody_key: Callable[[dict], str | None]
    build_historical_roll_reference: Callable[[dict], str]
    build_wind_standard_redesign_instruction: Callable[..., str]
    reasoning_cache: MutableMapping[str, Any]
    invoke_qwen: Callable[..., Any]
    parse_json_object: Callable[[str], dict | None]
    extract_qwen_agent_response: Callable[[dict | None], tuple[dict | None, dict]]
    restore_arrbody_fields: Callable[[dict, dict, set[str] | frozenset[str]], dict]
    sanitize_refined_result: Callable[..., dict | None]
    validate_wind_result: Callable[..., str]
    normalize_declared_pass_tail: Callable[[dict], dict]
    collect_deformation_pass_errors: Callable[..., list[str]]
    roll_errors_require_global_redesign: Callable[[list[str]], bool]
    prepare_full_roll_redesign_baseline: Callable[[dict], dict]
    normalize_deformation_passes: Callable[..., tuple[dict | None, str]]
    validate_dll_time_encodings: Callable[[dict, bool], str]
    enforce_performance_standard: Callable[[dict, dict], dict]
    cache_performance_baseline: Callable[[dict, dict], None]
    performance_values: Callable[[dict], dict]
    max_completion_tokens: int
    project_wind_result: Callable[..., dict] | None = None
    # 风电专用：从当前用户原始需求提取额外 Pcm 上限。该值与 GB/T 1591
    # 标准上限分别校验；管线钢不提供该依赖，也不会进入此分支。
    extract_wind_user_pcm_max: Callable[[str], float | None] | None = None
    # 生产路径提供 LangChain 模型和受当前产品/目标约束的工具；旧单元测试可继续
    # 只注入 invoke_qwen，以验证迁移前的确定性清洗与重试逻辑。
    agent_model: Any | None = None
    retrieve_agent_knowledge: Callable[[str], Any] | None = None
    retrieve_agent_history: Callable[[], Any] | None = None
    validate_agent_candidate: Callable[[dict, dict, dict, bool], list[dict]] | None = None
    validate_initial_cooling: Callable[[dict], list[dict]] | None = None
    create_agent_fn: Callable[..., Any] = create_agent


@dataclass(frozen=True)
class ProcessAgentDependencies:
    """加热、轧制和冷却智能体共享的执行依赖。

    三个入口使用同一依赖容器，但只读取各自所需字段。字段保存迁移前 ``api.py``
    中的 RAG、DLL、提示词、图片读取、模型判断、结果校验和缓存对象。
    """

    resolve_agent_round: Callable[..., dict | None]
    stage_input_changed: Callable[[dict, dict, str], bool]
    input_cache: MutableMapping[str, list]
    reasoning_cache: MutableMapping[str, list]
    visible_cache: MutableMapping[str, list]
    wind_power_prompt: Callable[[str], str]
    retrieve_reheat_rag: Callable[[str], str] | None = None
    generate_reheat_images: Callable[[dict, str], Any] | None = None
    collect_reheat_context: Callable[[dict], tuple[str, str, str, str]] | None = None
    build_reheat_prompt: Callable[..., str] | None = None
    invoke_reheat: Callable[..., dict] | None = None
    sanitize_reheat: Callable[[dict, dict], dict | None] | None = None
    retrieve_roll_rag: Callable[[str], str] | None = None
    generate_roll_images: Callable[[dict, str], Any] | None = None
    collect_roll_context: Callable[[dict], str] | None = None
    build_roll_prompt: Callable[..., str] | None = None
    invoke_roll: Callable[..., dict] | None = None
    sanitize_roll: Callable[[dict, dict], dict | None] | None = None
    require_valid_roll_result: Callable[[dict], dict] | None = None
    retrieve_cooling_rag: Callable[[str], str] | None = None
    generate_cooling_images: Callable[[dict, str], Any] | None = None
    collect_cooling_context: Callable[[dict], tuple[str, str, str]] | None = None
    build_cooling_prompt: Callable[..., str] | None = None
    invoke_cooling: Callable[..., dict] | None = None
    sanitize_cooling: Callable[[dict, dict, str], dict | None] | None = None
    user_requests_high_self_temp: Callable[[str], bool] | None = None
    set_arrbody_field: Callable[[dict, str, Any], bool] | None = None
    body_to_row: Callable[[dict], dict] | None = None
    to_float: Callable[[Any], float | None] | None = None
    stabilize_cooling_timing: Callable[[dict, str], dict] | None = None
    require_valid_cooling_timing: Callable[[dict], dict] | None = None


def _agent_final_text(response: Any) -> str:
    """提取 LangGraph Agent 最后一条模型消息中的文本。"""
    if not isinstance(response, dict):
        return str(response or "")
    messages = response.get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                    text_parts.append(str(part.get("text") or ""))
            if any(part.strip() for part in text_parts):
                return "".join(text_parts)
    return ""


def _assessment_module_action(assessment: dict | None, module_name: str) -> str:
    module = ((assessment or {}).get("change_assessment") or {}).get(module_name) or {}
    return str(module.get("action") or "").upper()


def _restore_assessment_inherited_fields(
    candidate: dict,
    inheritance_source: dict,
    design_change_assessment: dict | None,
    component_fields: set[str] | frozenset[str],
    roll_fields: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """执行 Agent 已作出的 INHERIT 决策；不替 Agent 推断任何 action。"""
    inherited_fields: set[str] = set()
    if _assessment_module_action(design_change_assessment, "composition") == "INHERIT":
        inherited_fields.update(str(field).upper() for field in component_fields)
    if _assessment_module_action(design_change_assessment, "rolling") == "INHERIT":
        inherited_fields.update(str(field).upper() for field in roll_fields)
    if _assessment_module_action(design_change_assessment, "cooling") == "INHERIT":
        inherited_fields.update({"TIME_ENTR", "TEMP_ENTR", "SELF_TEMP"})
    if not inherited_fields:
        return candidate
    source_values = {
        str(next(iter(item))).upper(): next(iter(item.values()))
        for item in (inheritance_source.get("arrBody") or [])
        if isinstance(item, dict) and len(item) == 1
    }
    restored = copy.deepcopy(candidate)
    restored_body = []
    for item in restored.get("arrBody") or []:
        if not isinstance(item, dict) or len(item) != 1:
            restored_body.append(item)
            continue
        key, value = next(iter(item.items()))
        source_value = source_values.get(str(key).upper())
        if str(key).upper() in inherited_fields and source_value is not None:
            restored_body.append({key: source_value})
        else:
            restored_body.append({key: value})
    restored["arrBody"] = restored_body
    return restored


def _invoke_refinement_langchain_agent(
    *,
    dependencies: CompositionRefinementDependencies,
    system_prompt: str,
    original: dict,
    spec_result: dict,
    is_wind: bool,
) -> tuple[str, dict]:
    """用 LangChain Agent 执行一次微调；最终正文仍是原 matched_result JSON。"""
    tool_usage = {"validator": False}

    @tool("search_product_knowledge")
    def search_product_knowledge(query: str) -> str:
        """检索当前设计产品专属知识库，返回本轮成分与工艺设计依据。"""
        if dependencies.retrieve_agent_knowledge is None:
            return json.dumps({"status": "unavailable"}, ensure_ascii=False)
        try:
            return _json_tool_result(
                dependencies.retrieve_agent_knowledge(query),
                unavailable_message="当前产品知识库未返回资料",
            )
        except Exception as exc:
            return json.dumps(
                {"status": "unavailable", "message": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

    @tool("search_current_target_history")
    def search_current_target_history() -> str:
        """读取后端锁定的当前目标规格历史实绩，目标厚度不能由 Agent 改写。"""
        if dependencies.retrieve_agent_history is None:
            return json.dumps({"status": "unavailable"}, ensure_ascii=False)
        try:
            return _json_tool_result(
                dependencies.retrieve_agent_history(),
                unavailable_message="当前目标规格未取得历史实绩",
            )
        except Exception as exc:
            return json.dumps(
                {"status": "unavailable", "message": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

    @tool("validate_candidate_matched_result")
    def validate_candidate_matched_result(candidate_json: str) -> str:
        """在最终提交前校验候选 matched_result，返回字段级错误 JSON。"""
        tool_usage["validator"] = True
        try:
            candidate = json.loads(candidate_json)
        except Exception as exc:
            return json.dumps(
                [{
                    "module": "structure",
                    "field": "matched_result",
                    "rule": "valid JSON object",
                    "status": "FAIL",
                    "current_values": {},
                    "message": f"候选 JSON 无法解析：{type(exc).__name__}: {exc}",
                }],
                ensure_ascii=False,
            )
        if dependencies.validate_agent_candidate is None:
            return json.dumps({"status": "validator_unavailable"}, ensure_ascii=False)
        errors = dependencies.validate_agent_candidate(
            original,
            candidate,
            spec_result,
            is_wind,
        )
        return json.dumps(
            {"status": "PASS" if not errors else "FAIL", "errors": errors},
            ensure_ascii=False,
        )

    agent = dependencies.create_agent_fn(
        model=dependencies.agent_model,
        tools=[
            search_product_knowledge,
            search_current_target_history,
            validate_candidate_matched_result,
        ],
        system_prompt=(
            system_prompt
            + "\n\n你现在作为 LangChain 自主设计 Agent 工作。可以按需要调用产品知识库和当前目标历史实绩工具；"
            "在最终提交前必须调用候选校验工具，并修复工具指出的全部错误。最终回复只能是完整 matched_result JSON，"
            "顶层不得增加任何 Agent 包装字段。"
        ),
        name="steel_composition_process_refinement",
    )
    response = agent.invoke(
        {"messages": [{"role": "user", "content": PIPELINE_REFINEMENT_USER_PROMPT}]},
        config={"recursion_limit": 12},
    )
    if not tool_usage["validator"]:
        raise ValueError("微调 Agent 未调用必需的候选 matched_result 校验工具")
    return _agent_final_text(response), {"agent": "langchain", "model": "qwen"}


def refine_composition_process_performance(
    spec_result: dict,
    matched_result: dict,
    user_message: str,
    session_id: str,
    material_name: str = "管线钢",
    engineering_standard_context: dict | None = None,
    historical_roll_reference_markdown: str = "",
    normalized_design_task: dict | None = None,
    reference_snapshot: dict | None = None,
    sql_match_reference: dict | None = None,
    design_change_assessment: dict | None = None,
    *,
    dependencies: CompositionRefinementDependencies,
) -> dict:
    """在固定 JSON 结构内微调成分、性能和初步轧制规程。

    参数:
        spec_result: 规格提取及标准兜底后形成的成分、厚度、性能和工艺边界。
        matched_result: MySQL 历史实绩匹配得到的完整字段骨架。
        user_message: 当前用户原始设计需求，优先用于锁定成品和板坯厚度。
        session_id: 当前会话标识，用于读取上下文和维护模型结果缓存。
        material_name: 当前设计对象名称，默认是管线钢，风电分支传入专用名称。
        engineering_standard_context: 风电分支的 GB/T 1591 等工程标准上下文；
            管线钢分支传入 ``None``。
        historical_roll_reference_markdown: 最多十组相近厚度历史轧制实绩，供模型
            设计道次分配及道次参数时参考。
        normalized_design_task: 续改请求的标准化任务；全新设计传入 ``None``。
        reference_snapshot: 被引用成功方案的最终快照，是续改设计的主基准。
        sql_match_reference: 本轮SQL匹配结果，只作为生产实绩参考，不能覆盖父快照。
        dependencies: 由 ``api.py`` 注入的 RAG、LLM、校验、缓存和格式化依赖。

    返回:
        保持顶层结构和 ``arrBody`` 字段顺序不变的完整 ``matched_result``。

    异常:
        WindPowerDesignValidationError: 风电分支多轮设计仍未通过标准或轧制规程
            门禁时抛出，调用方沿用原流程停止后续三个工艺智能体。
    """
    if not matched_result.get("arrBody"):
        return matched_result

    # 风电标准上下文不能直接复用并原地修改：同一对象后续还会用于报告。
    # 将 LLM 从用户原始提示词提取到的 Pcm 目标保存为独立约束，先做国标
    # Pcm 校验，再做用户 Pcm 校验；二者均不写入 spec_result 的成分字段。
    if engineering_standard_context:
        engineering_standard_context = copy.deepcopy(engineering_standard_context)
        if dependencies.extract_wind_user_pcm_max:
            user_pcm_max = dependencies.extract_wind_user_pcm_max(user_message)
            if user_pcm_max is not None:
                engineering_standard_context["Pcm_user_max"] = user_pcm_max

    revision_mode = bool(normalized_design_task and reference_snapshot)

    target_thickness = dependencies.extract_target_thickness(user_message)
    target_slab_thickness = dependencies.extract_target_slab_thickness(user_message)
    matched_result = dependencies.lock_explicit_thickness_targets(
        matched_result,
        target_thickness,
        target_slab_thickness,
    )
    context_modification_override = dependencies.is_context_modification_request(user_message)
    rag_context = dependencies.build_refinement_rag_context(spec_result, user_message)
    session_context = (
        dependencies.build_cross_route_context(session_id)
        or dependencies.get_recent_session_context(session_id)
    )
    if engineering_standard_context:
        # 风电分支只继承同用途会话语义，保持迁移前的历史管线语境隔离行为。
        session_context = dependencies.filter_wind_session_context(session_context)

    # 管线钢与风电用钢使用一致的历史实绩输入结构。风电 Agent 可以读取管线钢
    # 成分和性能以归纳相近低碳微合金钢的变化规律，但具体数值能否采用完全由
    # 系统提示词、风电标准和后置校验共同决定，代码不再预先清空这些字段。
    llm_matched_result = copy.deepcopy(matched_result)

    llm_matched_result_json = json.dumps(llm_matched_result, ensure_ascii=False)
    historical_roll_reference = dependencies.build_historical_roll_reference(matched_result)
    standard_context_text = json.dumps(
        engineering_standard_context or {},
        ensure_ascii=False,
        indent=2,
    )
    wind_standard_redesign_instruction = (
        dependencies.build_wind_standard_redesign_instruction(
            engineering_standard_context,
            "",
            spec_result,
        )
        if engineering_standard_context else ""
    )
    wind_process_rule = (
        WIND_POWER_REFINEMENT_PROCESS_RULE if engineering_standard_context else ""
    )
    prompt = build_pipeline_refinement_prompt(
        material_name,
        wind_process_rule,
        user_message,
        session_context,
        rag_context,
        standard_context_text,
        engineering_standard_context,
        wind_standard_redesign_instruction,
        spec_result,
        historical_roll_reference,
        historical_roll_reference_markdown,
        llm_matched_result_json,
        design_change_assessment,
    )
    if revision_mode:
        prompt += "\n\n" + build_revision_execution_prompt(
            normalized_design_task,
            reference_snapshot,
            sql_match_reference,
        )

    try:
        # 保持原缓存键、结构化输出模式、token 上限和关闭 thinking 的调用参数。
        dependencies.reasoning_cache.pop(f"{session_id}:pipeline_refine", None)
        dependencies.reasoning_cache.pop("_visible:后置成分/工艺微调", None)
        result = None
        repair_note = ""
        last_validation_error = ""
        last_wind_standard_error = ""
        current_llm_matched_result = copy.deepcopy(llm_matched_result)
        repair_scope = "all"
        locked_component_source = None
        # 只能冻结已经完整通过道次校验的规程。绝不能把初始 MySQL 历史记录
        # 当作“合法轧制规程”写回重试输入，否则成分错误会把刚重写的道次覆盖掉。
        last_valid_rolling_source = None
        component_performance_fields = (
            set(dependencies.component_fields) | set(dependencies.performance_fields)
        )

        # 成分/性能和轧制规程沿用独立修复预算，确保风电先过标准再校验道次。
        standard_attempt_budget = 6 if engineering_standard_context else 3
        roll_attempt_budget = 4
        max_attempts = standard_attempt_budget + roll_attempt_budget
        for attempt in range(1, max_attempts + 1):
            last_wind_standard_error = ""
            current_matched_result_json = json.dumps(
                current_llm_matched_result,
                ensure_ascii=False,
            )
            attempt_prompt = prompt.replace(
                llm_matched_result_json,
                current_matched_result_json,
                1,
            )
            if dependencies.agent_model is not None:
                try:
                    text, metadata = _invoke_refinement_langchain_agent(
                        dependencies=dependencies,
                        system_prompt=attempt_prompt + repair_note,
                        original=matched_result,
                        spec_result=spec_result,
                        is_wind=bool(engineering_standard_context),
                    )
                except Exception as exc:
                    last_validation_error = (
                        "LangChain Agent本轮调用或工具循环失败："
                        f"{type(exc).__name__}: {exc}"
                    )
                    repair_scope = "all"
                    repair_note = build_pipeline_refinement_repair_prompt(
                        last_validation_error,
                        PIPELINE_REFINEMENT_ALL_REPAIR_SCOPE_PROMPT,
                        "",
                    )
                    print(
                        f"[{material_name}MySQL匹配] 第 {attempt} 轮 Agent 调用失败，"
                        f"进入下一轮：{last_validation_error}"
                    )
                    continue
                raw = None
            else:
                # 兼容旧测试替身；生产依赖始终提供 agent_model，不再走直接LLM调用。
                raw = dependencies.invoke_qwen(
                    [
                        SystemMessage(content=attempt_prompt + repair_note),
                        HumanMessage(content=PIPELINE_REFINEMENT_USER_PROMPT),
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=dependencies.max_completion_tokens,
                    extra_body={"enable_thinking": False},
                )
                text = raw.content
                metadata = dict(getattr(raw, "raw_metadata", {}) or {})
            parsed_candidate = dependencies.parse_json_object(str(text))
            candidate, _ = dependencies.extract_qwen_agent_response(parsed_candidate)
            if isinstance(candidate, dict):
                inheritance_source = (
                    reference_snapshot.get("matched_result")
                    if isinstance(reference_snapshot, dict)
                    else llm_matched_result
                ) or llm_matched_result
                candidate = _restore_assessment_inherited_fields(
                    candidate,
                    inheritance_source,
                    design_change_assessment,
                    dependencies.component_fields,
                    dependencies.roll_fields,
                )
                if repair_scope == "composition" and isinstance(last_valid_rolling_source, dict):
                    candidate = dependencies.restore_arrbody_fields(
                        candidate,
                        last_valid_rolling_source,
                        dependencies.roll_fields,
                    )
                elif repair_scope == "rolling" and isinstance(locked_component_source, dict):
                    candidate = dependencies.restore_arrbody_fields(
                        candidate,
                        locked_component_source,
                        component_performance_fields,
                    )

            print(
                f"[{material_name}MySQL匹配] LLM微调第 {attempt} 轮响应诊断: "
                f"model={metadata.get('model')}, finish_reason={metadata.get('finish_reason')}, "
                f"usage={metadata.get('usage')}, content_chars={len(str(text or ''))}, "
                f"parsed_top_keys={list(parsed_candidate.keys()) if isinstance(parsed_candidate, dict) else None}, "
                f"candidate_top_keys={list(candidate.keys()) if isinstance(candidate, dict) else None}"
            )
            if isinstance(candidate, dict):
                candidate_body = candidate.get("arrBody")
                current_body = current_llm_matched_result.get("arrBody")
                if (
                    set(candidate.keys()) == set(matched_result.keys())
                    and isinstance(candidate_body, list)
                    and isinstance(current_body, list)
                    and len(candidate_body) == len(current_body)
                    and all(
                        dependencies.get_arrbody_key(candidate_item)
                        == dependencies.get_arrbody_key(current_item)
                        for candidate_item, current_item in zip(candidate_body, current_body)
                    )
                ):
                    current_llm_matched_result = copy.deepcopy(candidate)

                sanitize_errors: list[str] = []
                result = dependencies.sanitize_refined_result(
                    matched_result,
                    candidate,
                    spec_result,
                    target_thickness=target_thickness,
                    target_slab_thickness=target_slab_thickness,
                    context_modification_override=context_modification_override,
                    strict_no_restore=bool(engineering_standard_context),
                    soft_process_bounds=bool(engineering_standard_context),
                    validation_errors=sanitize_errors,
                )
                if result is not None:
                    if revision_mode:
                        revision_errors = validate_revision_constraints(
                            result,
                            reference_snapshot,
                            normalized_design_task,
                            spec_result,
                            require_final_pass=False,
                        )
                        if revision_errors:
                            last_validation_error = "；".join(revision_errors)
                            current_llm_matched_result = copy.deepcopy(result)
                            repair_scope = "all"
                            result = None
                            print(
                                f"[{material_name}MySQL匹配] 历史方案续改约束未通过: "
                                f"{last_validation_error}"
                            )

                if result is not None:
                    current_llm_matched_result = copy.deepcopy(result)
                    if engineering_standard_context:
                        wind_validation_error = dependencies.validate_wind_result(
                            result,
                            engineering_standard_context,
                            spec_result,
                        )
                        # 标准上下限和 CEV/Pcm 属于确定性数值约束。模型给出的
                        # 候选仅有轻微越界时，先在当前候选上做约束投影并立即
                        # 复核，避免把同一数值错误交给模型重复十轮。投影失败或
                        # 仍不合格时才进入原有的成分重设计循环。
                        if wind_validation_error and dependencies.project_wind_result:
                            projected_result = dependencies.project_wind_result(
                                result,
                                engineering_standard_context,
                                spec_result,
                            )
                            projected_error = dependencies.validate_wind_result(
                                projected_result,
                                engineering_standard_context,
                                spec_result,
                            )
                            if not projected_error:
                                print(
                                    f"[{material_name}MySQL匹配] "
                                    "风电成分/性能确定性约束修正后通过标准校验"
                                )
                                result = projected_result
                                current_llm_matched_result = copy.deepcopy(result)
                                wind_validation_error = ""
                        if wind_validation_error:
                            last_wind_standard_error = wind_validation_error
                            last_validation_error = wind_validation_error
                            current_llm_matched_result = _pin_boundary_repair_baseline(
                                result,
                                [wind_validation_error],
                                set(dependencies.component_fields),
                                set(dependencies.performance_fields),
                            )
                            # 标准不合格时，仅在此前已经得到完整合法规程的前提下
                            # 才能冻结道次；否则成分和规程必须基于当前候选联动重写。
                            repair_scope = (
                                "composition"
                                if isinstance(last_valid_rolling_source, dict)
                                else "all"
                            )
                            if repair_scope == "composition":
                                current_llm_matched_result = (
                                    dependencies.restore_arrbody_fields(
                                        current_llm_matched_result,
                                        last_valid_rolling_source,
                                        dependencies.roll_fields,
                                    )
                                )
                            else:
                                current_llm_matched_result = (
                                    dependencies.prepare_full_roll_redesign_baseline(
                                        current_llm_matched_result
                                    )
                                )
                            print(
                                f"[{material_name}MySQL匹配] "
                                f"LLM微调标准校验未通过: {wind_validation_error}"
                            )
                            result = None

                    if result is not None:
                        locked_component_source = copy.deepcopy(result)
                        tail_normalized_result = dependencies.normalize_declared_pass_tail(result)
                        roll_errors = dependencies.collect_deformation_pass_errors(
                            tail_normalized_result,
                            validate_timing=False,
                            validate_cooling_timing=False,
                        )
                        roll_errors = _remove_refinement_turn_width_errors(roll_errors)
                        if roll_errors:
                            last_validation_error = "；".join(roll_errors)
                            global_redesign = dependencies.roll_errors_require_global_redesign(
                                roll_errors
                            )
                            current_llm_matched_result = (
                                dependencies.prepare_full_roll_redesign_baseline(
                                    tail_normalized_result
                                )
                                if global_redesign else copy.deepcopy(tail_normalized_result)
                            )
                            repair_scope = "rolling"
                            result = None
                            print(
                                f"[{material_name}MySQL匹配] LLM微调轧制规程校验未通过，"
                                f"global_redesign={global_redesign}: {last_validation_error}"
                            )
                        else:
                            normalized_result, roll_validation_error = (
                                dependencies.normalize_deformation_passes(
                                    tail_normalized_result,
                                    f"{material_name}后置成分/工艺微调",
                                    validate_timing=False,
                                    validate_cooling_timing=False,
                                    tolerate_turn_width_errors=True,
                                )
                            )
                            if normalized_result is None:
                                last_validation_error = roll_validation_error
                                current_llm_matched_result = copy.deepcopy(
                                    tail_normalized_result
                                )
                                repair_scope = "rolling"
                                result = None
                            else:
                                result = normalized_result
                                time_encoding_error = (
                                    dependencies.validate_dll_time_encodings(
                                        result,
                                        True,
                                    )
                                )
                                if time_encoding_error:
                                    last_validation_error = time_encoding_error
                                    current_llm_matched_result = copy.deepcopy(result)
                                    repair_scope = "rolling"
                                    result = None
                                    print(
                                        f"[{material_name}MySQL匹配] "
                                        "LLM微调 DLL 时间编码校验未通过: "
                                        f"{last_validation_error}"
                                    )
                                else:
                                    cooling_errors = (
                                        dependencies.validate_initial_cooling(result)
                                        if dependencies.validate_initial_cooling
                                        else []
                                    )
                                    if cooling_errors:
                                        last_validation_error = json.dumps(
                                            cooling_errors,
                                            ensure_ascii=False,
                                        )
                                        current_llm_matched_result = copy.deepcopy(result)
                                        repair_scope = "all"
                                        result = None
                                        print(
                                            f"[{material_name}MySQL匹配] "
                                            "LLM微调冷却初值门禁未通过: "
                                            f"{last_validation_error}"
                                        )
                                    else:
                                        current_llm_matched_result = copy.deepcopy(result)
                                        last_valid_rolling_source = copy.deepcopy(result)
                                        break
                else:
                    last_validation_error = (
                        "；".join(sanitize_errors)
                        or "候选值未通过字段结构或边界校验"
                    )
                    boundary_repair_fields = _extract_boundary_repair_fields(
                        sanitize_errors,
                        component_performance_fields,
                    )
                    if boundary_repair_fields:
                        # 成分/性能越界优先于上一轮的轧制修复范围。把越界值钉在
                        # 当前候选的最近合法边界，禁止回退初始 MySQL 历史记录。
                        # 只有此前已通过完整道次校验时，下一轮才能只修复成分；
                        # 否则需要在当前设计基础上连同完整规程一起重新输出。
                        current_llm_matched_result = _pin_boundary_repair_baseline(
                            current_llm_matched_result,
                            sanitize_errors,
                            set(dependencies.component_fields),
                            set(dependencies.performance_fields),
                        )
                        repair_scope = (
                            "composition"
                            if isinstance(last_valid_rolling_source, dict)
                            else "all"
                        )
                        if repair_scope == "composition":
                            current_llm_matched_result = (
                                dependencies.restore_arrbody_fields(
                                    current_llm_matched_result,
                                    last_valid_rolling_source,
                                    dependencies.roll_fields,
                                )
                            )
                        else:
                            current_llm_matched_result = (
                                dependencies.prepare_full_roll_redesign_baseline(
                                    current_llm_matched_result
                                )
                            )
                        print(
                            f"[{material_name}MySQL匹配] 成分/性能字段越界，"
                            f"切换为{('成分范围' if repair_scope == 'composition' else '成分与全规程')}重设计: "
                            f"{', '.join(sorted(boundary_repair_fields))}"
                        )
            else:
                last_validation_error = "模型未返回合法 matched_result JSON 对象"
                print(f"[管线钢MySQL匹配] LLM微调第 {attempt} 次未返回JSON对象")

            standard_repair_instruction = (
                dependencies.build_wind_standard_redesign_instruction(
                    engineering_standard_context,
                    last_wind_standard_error,
                    spec_result,
                )
                if engineering_standard_context and last_wind_standard_error else ""
            )
            if repair_scope == "composition":
                scope_instruction = PIPELINE_REFINEMENT_COMPOSITION_REPAIR_SCOPE_PROMPT
            elif repair_scope == "rolling":
                scope_instruction = PIPELINE_REFINEMENT_ROLLING_REPAIR_SCOPE_PROMPT
            else:
                scope_instruction = PIPELINE_REFINEMENT_ALL_REPAIR_SCOPE_PROMPT
            repair_note = build_pipeline_refinement_repair_prompt(
                last_validation_error,
                scope_instruction,
                standard_repair_instruction,
            )

        if result is None:
            if revision_mode:
                final_error = last_validation_error or "续改结果未通过父子方案强约束"
                print(
                    f"[{material_name}MySQL匹配] LLM微调 {max_attempts} 次均未通过续改校验，"
                    f"不回退SQL或父方案：{final_error}"
                )
                raise DesignRevisionValidationError(final_error)
            if engineering_standard_context:
                final_error = (
                    last_validation_error
                    or last_wind_standard_error
                    or "模型结果未通过结构或风电标准校验"
                )
                print(
                    f"[{material_name}MySQL匹配] LLM微调 {max_attempts} 次均未通过校验，"
                    "终止风电设计且不回退历史字段骨架: "
                    f"{final_error}"
                )
                raise WindPowerDesignValidationError(final_error)
            if dependencies.agent_model is not None:
                final_error = last_validation_error or "模型结果未通过结构或工艺校验"
                print(
                    f"[{material_name}MySQL匹配] LangChain Agent微调 {max_attempts} 次均未通过校验，"
                    f"不回退历史字段骨架: {final_error}"
                )
                raise CompositionRefinementValidationError(final_error)
            print(
                f"[{material_name}MySQL匹配] LLM微调 {max_attempts} 次均未通过校验，"
                "保留原始字段骨架"
            )
            result = matched_result
    except (
        WindPowerDesignValidationError,
        DesignRevisionValidationError,
        CompositionRefinementValidationError,
    ):
        raise
    except Exception as exc:
        print(f"[管线钢MySQL匹配] LLM微调失败: {exc}")
        if revision_mode:
            raise DesignRevisionValidationError(
                f"历史方案续改模型调用失败：{type(exc).__name__}: {exc}"
            ) from exc
        if engineering_standard_context:
            raise WindPowerDesignValidationError(
                f"风电成分/性能/轧制规程重设计调用失败：{type(exc).__name__}: {exc}"
            ) from exc
        if dependencies.agent_model is not None:
            raise CompositionRefinementValidationError(
                f"成分/性能/工艺微调 Agent 调用失败：{type(exc).__name__}: {exc}"
            ) from exc
        result = matched_result

    result = dependencies.enforce_performance_standard(result, spec_result)
    if revision_mode:
        final_policy_errors = validate_revision_constraints(
            result,
            reference_snapshot,
            normalized_design_task,
            spec_result,
            require_final_pass=False,
        )
        if final_policy_errors:
            raise DesignRevisionValidationError(
                "历史方案续改结果未满足约束：" + "；".join(final_policy_errors)
            )
    dependencies.cache_performance_baseline(result, spec_result)
    print(
        "[管线钢MySQL匹配] 后置微调合格性能基线: "
        f"{dependencies.performance_values(result)}"
    )
    return result


def refine_reheat_process(
    matched_result: dict,
    context: str,
    reasoning_key_prefix: str | None = None,
    progress_callback=None,
    *,
    dependencies: ProcessAgentDependencies,
) -> dict:
    """运行加热仿真并微调均热温度、时间及相邻加热温度。

    参数:
        matched_result: 后置成分/性能微调后的完整设计结果；字段结构保持不变。
        context: 用户需求、最近会话、DLL映射及前序智能体判断组成的外部上下文。
        reasoning_key_prefix: 当前会话标识，用于保存每轮输入、判断摘要和可见正文。
        progress_callback: 同步回调；每轮开始及模型判断完成后向 ``api.py`` 推送进度。
        dependencies: 由 ``api.py`` 注入的加热 RAG、DLL、图片、LLM和缓存依赖。

    返回:
        完整 ``matched_result``。最多执行三轮；判断通过时提前返回。最后一轮参数
        与最近仿真输入不同时，会补做一次只生成最终图片、不再调用模型的仿真。
    """
    if not isinstance(matched_result, dict) or not matched_result.get("arrBody"):
        print("[管线钢加热智能体] matched_result 无效，直接返回原值")
        return matched_result

    rag_context = dependencies.retrieve_reheat_rag(context)
    current_result = copy.deepcopy(matched_result)
    last_simulated_result = None

    def finalize(final_result: dict) -> dict:
        if (
            last_simulated_result is not None
            and dependencies.stage_input_changed(
                last_simulated_result,
                final_result,
                "reheat",
            )
        ):
            print("[管线钢加热智能体] 最终加热参数已变化，补做一次最终DLL仿真（不再调用判断模型）")
            dependencies.generate_reheat_images(final_result, context)
        return final_result

    for attempt in range(1, 4):
        if progress_callback:
            progress_callback({
                "event_type": "module_decision",
                "attempt": attempt,
                "stage": "reheat",
            })
        print(f"[管线钢加热智能体] 开始第 {attempt}/3 轮加热模拟与 Qwen 判断")

        last_simulated_result = copy.deepcopy(current_result)
        dependencies.generate_reheat_images(current_result, context)
        tas_text, soaking_image, grain_growth_image, grain_distribution_image = (
            dependencies.collect_reheat_context(current_result)
        )
        user_prompt = dependencies.wind_power_prompt(
            dependencies.build_reheat_prompt(
                context=context,
                rag_context=rag_context,
                matched_result=current_result,
                tas_text=tas_text,
            )
        )
        reasoning_key = (
            f"{reasoning_key_prefix}:reheat" if reasoning_key_prefix else None
        )
        if reasoning_key:
            dependencies.input_cache.setdefault(reasoning_key, []).append(
                copy.deepcopy(current_result)
            )
        sanitized = dependencies.resolve_agent_round(
            invoke_func=dependencies.invoke_reheat,
            sanitize_func=dependencies.sanitize_reheat,
            current_result=current_result,
            base_prompt=user_prompt,
            images=[
                (soaking_image, "均热温度.png"),
                (grain_growth_image, "晶粒长大.png"),
                (grain_distribution_image, "晶粒尺寸分布.png"),
            ],
            reasoning_key=reasoning_key,
            progress_callback=progress_callback,
            stage="reheat",
            stage_label="管线钢加热智能体",
            simulation_attempt=attempt,
        )
        if sanitized is None:
            print("[管线钢加热智能体] 同轮修复重试耗尽，结束智能体并返回当前结果")
            return finalize(current_result)

        if progress_callback:
            try:
                round_reasoning = (
                    dependencies.reasoning_cache.get(reasoning_key, [""])[-1]
                    if reasoning_key else ""
                )
                visible_judgement = (
                    dependencies.visible_cache.get(reasoning_key, [{}])[-1]
                    if reasoning_key else {}
                )
                progress_callback({
                    "attempt": attempt,
                    "stage": "reheat",
                    "before": copy.deepcopy(current_result),
                    "after": copy.deepcopy(sanitized),
                    "reasoning": round_reasoning,
                    "judgement": copy.deepcopy(visible_judgement),
                })
            except Exception as exc:
                print(f"[管线钢加热智能体] 第 {attempt} 轮前端进度通知失败: {exc}")

        current_result = sanitized
        if current_result.get("isState") is True:
            print(f"[管线钢加热智能体] 第 {attempt} 轮判断通过，返回 matched_result")
            return finalize(current_result)
        print(f"[管线钢加热智能体] 第 {attempt} 轮未通过，继续使用调整后的均热工艺进入下一轮")

    print("[管线钢加热智能体] 3轮后仍未通过，返回最终调整后的 matched_result")
    return finalize(current_result)


def refine_rolling_process(
    matched_result: dict,
    context: str,
    reasoning_key_prefix: str | None = None,
    progress_callback=None,
    historical_roll_reference_markdown: str = "",
    *,
    dependencies: ProcessAgentDependencies,
) -> dict:
    """运行轧制仿真并微调完整道次规程。

    参数:
        matched_result: 加热智能体完成后的完整设计结果。
        context: 用户需求、会话上下文以及加热智能体已采纳判断正文。
        reasoning_key_prefix: 当前会话标识，用于保存每轮轧制判断相关缓存。
        progress_callback: 同步回调；每轮开始及判断完成后即时向 ``api.py`` 报告。
        historical_roll_reference_markdown: 最多十组相近厚度历史轧制实绩，供模型
            选择相似道次并设计厚度、温度、宽度、速度和轧制力。
        dependencies: 由 ``api.py`` 注入的轧制 RAG、DLL、图片、LLM、校验和缓存依赖。

    返回:
        通过最终硬门禁的完整 ``matched_result``。最多执行三轮；末道厚度、道次
        连续性或道次时间顺序不合格时沿用原异常行为，不进入冷却及最终报告。
        TIME_ENTR 与末道次的跨阶段关系由随后拥有该字段修改权的冷却智能体
        确定性收敛，不在本阶段使用历史 TIME_ENTR 阻断轧制方案。
    """
    if not isinstance(matched_result, dict) or not matched_result.get("arrBody"):
        print("[管线钢轧制智能体] matched_result 无效，直接返回原值")
        return matched_result

    rag_context = dependencies.retrieve_roll_rag(context)
    current_result = copy.deepcopy(matched_result)
    last_simulated_result = None

    def finalize(final_result: dict) -> dict:
        final_result = dependencies.require_valid_roll_result(final_result)
        if (
            last_simulated_result is not None
            and dependencies.stage_input_changed(
                last_simulated_result,
                final_result,
                "roll",
            )
        ):
            print("[管线钢轧制智能体] 最终轧制参数已变化，补做一次最终DLL仿真（不再调用判断模型）")
            dependencies.generate_roll_images(final_result, context)
        return final_result

    for attempt in range(1, 4):
        if progress_callback:
            progress_callback({
                "event_type": "module_decision",
                "attempt": attempt,
                "stage": "roll",
            })
        print(f"[管线钢轧制智能体] 开始第 {attempt}/3 轮轧制模拟与 Qwen 判断")

        last_simulated_result = copy.deepcopy(current_result)
        dependencies.generate_roll_images(current_result, context)
        pass_grain_size_image = dependencies.collect_roll_context(current_result)
        user_prompt = dependencies.wind_power_prompt(
            dependencies.build_roll_prompt(
                context=context,
                rag_context=rag_context,
                matched_result=current_result,
                historical_roll_reference_markdown=historical_roll_reference_markdown,
            )
        )
        reasoning_key = f"{reasoning_key_prefix}:roll" if reasoning_key_prefix else None
        if reasoning_key:
            dependencies.input_cache.setdefault(reasoning_key, []).append(
                copy.deepcopy(current_result)
            )
        sanitized = dependencies.resolve_agent_round(
            invoke_func=dependencies.invoke_roll,
            sanitize_func=dependencies.sanitize_roll,
            current_result=current_result,
            base_prompt=user_prompt,
            images=[(pass_grain_size_image, "各道次晶粒尺寸.png")],
            reasoning_key=reasoning_key,
            progress_callback=progress_callback,
            stage="roll",
            stage_label="管线钢轧制智能体",
            simulation_attempt=attempt,
        )
        if sanitized is None:
            print("[管线钢轧制智能体] 同轮修复重试耗尽，结束智能体并返回当前结果")
            return finalize(current_result)

        if progress_callback:
            try:
                round_reasoning = (
                    dependencies.reasoning_cache.get(reasoning_key, [""])[-1]
                    if reasoning_key else ""
                )
                visible_judgement = (
                    dependencies.visible_cache.get(reasoning_key, [{}])[-1]
                    if reasoning_key else {}
                )
                progress_callback({
                    "attempt": attempt,
                    "stage": "roll",
                    "before": copy.deepcopy(current_result),
                    "after": copy.deepcopy(sanitized),
                    "reasoning": round_reasoning,
                    "judgement": copy.deepcopy(visible_judgement),
                })
            except Exception as exc:
                print(f"[管线钢轧制智能体] 第 {attempt} 轮前端进度通知失败: {exc}")

        current_result = sanitized
        if current_result.get("isState") is True:
            print(f"[管线钢轧制智能体] 第 {attempt} 轮判断通过，返回 matched_result")
            return finalize(current_result)
        print(f"[管线钢轧制智能体] 第 {attempt} 轮未通过，继续使用调整后的轧制工艺进入下一轮")

    print("[管线钢轧制智能体] 3轮后仍未通过，返回最终调整后的 matched_result")
    return finalize(current_result)


def refine_cooling_process(
    matched_result: dict,
    context: str,
    reasoning_key_prefix: str | None = None,
    progress_callback=None,
    *,
    dependencies: ProcessAgentDependencies,
) -> dict:
    """运行控制冷却仿真并微调开冷、入水、返红温度及最终性能。

    参数:
        matched_result: 轧制智能体最终硬门禁通过后的完整设计结果。
        context: 用户需求、会话上下文以及加热和轧制智能体已采纳判断正文。
        reasoning_key_prefix: 当前会话标识，用于保存每轮冷却判断相关缓存。
        progress_callback: 同步回调；每轮开始、判断完成及485℃兜底时即时通知
            ``api.py`` 的异步流式编排层。
        dependencies: 由 ``api.py`` 注入的冷却 RAG、DLL、图片、LLM、结果校验、
            字段读取和数值转换依赖。

    返回:
        完整 ``matched_result``。最多执行两轮；用户未明确要求高温返红且模型
        重试耗尽或末轮返红仍不合格时，保持原结构并将已有 ``SELF_TEMP`` 设置为
        485℃，随后补做与最终参数一致的冷却仿真。
    """
    if not isinstance(matched_result, dict) or not matched_result.get("arrBody"):
        print("[管线钢冷却智能体] matched_result 无效，直接返回原值")
        return matched_result

    rag_context = dependencies.retrieve_cooling_rag(context)
    current_result = copy.deepcopy(matched_result)
    # TIME_ENTR 属于冷却阶段字段。进入首轮冷却仿真前，先依据真实末道次和
    # 粗精轧分界生成可行的开冷时刻，消除历史实绩 TIME_ENTR 对新轧制规程
    # 的污染。该步骤只改时间，不改厚度、温度、速度或轧制力。
    if dependencies.stabilize_cooling_timing:
        current_result = dependencies.stabilize_cooling_timing(current_result, context)
    last_simulated_result = None
    allow_high_self_temp = dependencies.user_requests_high_self_temp(context)

    def finalize(final_result: dict) -> dict:
        if dependencies.stabilize_cooling_timing:
            final_result = dependencies.stabilize_cooling_timing(final_result, context)
        if dependencies.require_valid_cooling_timing:
            final_result = dependencies.require_valid_cooling_timing(final_result)
        if (
            last_simulated_result is not None
            and dependencies.stage_input_changed(
                last_simulated_result,
                final_result,
                "cooling",
            )
        ):
            print("[管线钢冷却智能体] 最终冷却或性能参数已变化，补做一次最终DLL仿真（不再调用判断模型）")
            dependencies.generate_cooling_images(final_result, context)
        return final_result

    def apply_self_temp_fallback(final_result: dict, reason: str) -> dict:
        if allow_high_self_temp:
            return finalize(final_result)
        fallback_result = copy.deepcopy(final_result)
        if not dependencies.set_arrbody_field(fallback_result, "SELF_TEMP", "485"):
            print("[管线钢冷却智能体] 未找到SELF_TEMP字段，无法应用485℃兜底")
            return finalize(final_result)
        fallback_result["isState"] = False
        print(
            "[管线钢冷却智能体] "
            f"{reason}；当前用户未明确要求高温返红，SELF_TEMP确定性兜底为485℃"
        )
        finalized_result = finalize(fallback_result)
        if progress_callback:
            progress_callback({
                "event_type": "fallback_applied",
                "attempt": 2,
                "stage": "cooling",
                "before": copy.deepcopy(final_result),
                "after": copy.deepcopy(finalized_result),
                "message": (
                    f"{reason}；当前用户未明确指定高温返红，"
                    "已将 SELF_TEMP 确定性设置为485℃并完成最终冷却仿真。"
                ),
            })
        return finalized_result

    for attempt in range(1, 3):
        if progress_callback:
            progress_callback({
                "event_type": "module_decision",
                "attempt": attempt,
                "stage": "cooling",
            })
        print(f"[管线钢冷却智能体] 开始第 {attempt}/2 轮冷却模拟与 Qwen 判断")

        last_simulated_result = copy.deepcopy(current_result)
        dependencies.generate_cooling_images(current_result, context)
        phase_image, cct_image, strengthening_image = (
            dependencies.collect_cooling_context(current_result)
        )
        user_prompt = dependencies.wind_power_prompt(
            dependencies.build_cooling_prompt(
                context=context,
                rag_context=rag_context,
                matched_result=current_result,
            )
        )
        reasoning_key = (
            f"{reasoning_key_prefix}:cooling" if reasoning_key_prefix else None
        )
        if reasoning_key:
            dependencies.input_cache.setdefault(reasoning_key, []).append(
                copy.deepcopy(current_result)
            )
        sanitized = dependencies.resolve_agent_round(
            invoke_func=dependencies.invoke_cooling,
            sanitize_func=lambda original, candidate: dependencies.sanitize_cooling(
                original,
                candidate,
                context,
            ),
            current_result=current_result,
            base_prompt=user_prompt,
            images=[
                (phase_image, "相组成.png"),
                (cct_image, "CCT.png"),
                (strengthening_image, "强化机制.PNG"),
            ],
            reasoning_key=reasoning_key,
            progress_callback=progress_callback,
            stage="cooling",
            stage_label="管线钢冷却智能体",
            simulation_attempt=attempt,
        )
        if sanitized is None:
            print("[管线钢冷却智能体] 同轮修复重试耗尽，结束智能体并返回当前结果")
            return apply_self_temp_fallback(current_result, "同轮模型修复重试耗尽")

        if progress_callback:
            try:
                round_reasoning = (
                    dependencies.reasoning_cache.get(reasoning_key, [""])[-1]
                    if reasoning_key else ""
                )
                visible_judgement = (
                    dependencies.visible_cache.get(reasoning_key, [{}])[-1]
                    if reasoning_key else {}
                )
                progress_callback({
                    "attempt": attempt,
                    "stage": "cooling",
                    "before": copy.deepcopy(current_result),
                    "after": copy.deepcopy(sanitized),
                    "reasoning": round_reasoning,
                    "judgement": copy.deepcopy(visible_judgement),
                })
            except Exception as exc:
                print(f"[管线钢冷却智能体] 第 {attempt} 轮前端进度通知失败: {exc}")

        current_result = sanitized
        if current_result.get("isState") is True:
            print(f"[管线钢冷却智能体] 第 {attempt} 轮判断通过，返回 matched_result")
            return finalize(current_result)
        print(
            f"[管线钢冷却智能体] 第 {attempt} 轮未通过，"
            "继续使用调整后的开冷时刻、入水和返红温度进入下一轮"
        )

    print("[管线钢冷却智能体] 2轮后仍未通过，返回最终调整后的 matched_result")
    if not allow_high_self_temp:
        final_self_temp = dependencies.to_float(
            dependencies.body_to_row(current_result).get("SELF_TEMP")
        )
        if final_self_temp is None or final_self_temp >= 500.0:
            return apply_self_temp_fallback(
                current_result,
                "两轮判断结束后返红温度仍不合格",
            )
    return finalize(current_result)
