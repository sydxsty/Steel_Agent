"""
steel_spec_extractor.py — 钢材成分/性能规格提取模块
======================================================

提供 extract_steel_specs() 方法，从知识库中提取钢材成分和性能的上下限，
返回结构化 JSON，包含所有元素的含量范围和力学性能指标。
"""

import json
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from my_llm import deepseek_Llm

# 导入会话持久化模块
from session_store import SessionStore, PersistentChatMessageHistory, register_for_cleanup

# ============================================================
# 会话存储 — 持久化（数据库 + 内存缓存）
# ============================================================
_MAX_TURNS = 20
_spec_store = SessionStore(
    session_type="spec_extractor",
    max_turns=_MAX_TURNS,
    ttl=3600,
)
# 注册到后台定时清理
register_for_cleanup(_spec_store)


def _get_spec_history(sid: str) -> BaseChatMessageHistory:
    """获取指定 session_id 的持久化历史（兼容 LangChain BaseChatMessageHistory 接口）"""
    return PersistentChatMessageHistory(_spec_store, sid)


# ============================================================
# JSON Schema — 所有字段，默认下限0上限9999
# ============================================================
STEEL_SPEC_SCHEMA = {
    "用途": "",
    "THK_max": 9999.0, "THK_min": 0.0,
    "C_max": 9999, "C_min": 0.0,
    "SI_max": 9999, "SI_min": 0.0,
    "MN_max": 9999, "MN_min": 0.0,
    "P_max": 9999, "P_min": 0.0,
    "S_max": 9999, "S_min": 0.0,
    "N_max": 9999, "N_min": 0.0,
    "NB_max": 9999, "NB_min": 0.0,
    "V_max": 9999, "V_min": 0.0,
    "TI_max": 9999, "TI_min": 0.0,
    "AL_max": 9999, "AL_min": 0.0,
    "ALS_max": 9999, "ALS_min": 0.0,
    "CU_max": 9999, "CU_min": 0.0,
    "CR_max": 9999, "CR_min": 0.0,
    "NI_max": 9999, "NI_min": 0.0,
    "CO_max": 9999, "CO_min": 0.0,
    "MO_max": 9999, "MO_min": 0.0,
    "B_max": 9999, "B_min": 0.0,
    "SOAKING_TEMP_max": 9999.0, "SOAKING_TEMP_min": 0.0,  # 均热温度
    "FET_max": 9999.0, "FET_min": 0.0,  # 精轧开轧温度
    "FDT_max": 9999.0, "FDT_min": 0.0,  # 终轧温度
    "CT_max": 9999.0, "CT_min": 0.0,  # 卷取温度
    "QUENCHING_TEMP_max": 9999.0, "QUENCHING_TEMP_min": 0.0,  # 淬火温度
    "TEMPERING_TEMP_max": 9999.0, "TEMPERING_TEMP_min": 0.0,  # 回火温度
    "YS_max": 9999.0, "YS_min": 0.0,
    "TS_max": 9999.0, "TS_min": 0.0,
    "EL_max": 9999.0, "EL_min": 0.0,
    "AKV_max": 9999.0, "AKV_min": 0.0,
    "TEMP_ENTR_max": 9999.0, "TEMP_ENTR_min": 0.0,
    "FEH_max": 9999.0, "FEH_min": 0.0,
    "SELF_TEMP_max": 9999.0, "SELF_TEMP_min": 0.0,
    "FURNACE_EXIT_TEMP_max": 9999.0, "FURNACE_EXIT_TEMP_min": 0.0,
    "SLAB_THICK_max": 9999.0, "SLAB_THICK_min": 0.0,
    "AIM_THICK_max": 9999.0, "AIM_THICK_min": 0.0,
}

STEEL_SPEC_FIELD_ALIASES = {
    "均热温度_max": "SOAKING_TEMP_max",
    "均热温度_min": "SOAKING_TEMP_min",
    "淬火温度_max": "QUENCHING_TEMP_max",
    "淬火温度_min": "QUENCHING_TEMP_min",
    "回火温度_max": "TEMPERING_TEMP_max",
    "回火温度_min": "TEMPERING_TEMP_min",
}

PIPELINE_EXCLUDED_SPEC_FIELDS = {
    "SOAKING_TEMP_max", "SOAKING_TEMP_min",
    "CT_max", "CT_min",
    "QUENCHING_TEMP_max", "QUENCHING_TEMP_min",
    "TEMPERING_TEMP_max", "TEMPERING_TEMP_min",
}

PIPELINE_SPEC_FIELD_ORDER = [
    "用途",
    "C_max", "C_min",
    "SI_max", "SI_min",
    "MN_max", "MN_min",
    "P_max", "P_min",
    "S_max", "S_min",
    "N_max", "N_min",
    "NB_max", "NB_min",
    "V_max", "V_min",
    "TI_max", "TI_min",
    "AL_max", "AL_min",
    "ALS_max", "ALS_min",
    "CU_max", "CU_min",
    "CR_max", "CR_min",
    "NI_max", "NI_min",
    "CO_max", "CO_min",
    "MO_max", "MO_min",
    "B_max", "B_min",
    "FET_max", "FET_min",
    "FDT_max", "FDT_min",
    "FURNACE_EXIT_TEMP_max", "FURNACE_EXIT_TEMP_min",
    "SLAB_THICK_max", "SLAB_THICK_min",
    "AIM_THICK_max", "AIM_THICK_min",
    "TEMP_ENTR_max", "TEMP_ENTR_min",
    "FEH_max", "FEH_min",
    "SELF_TEMP_max", "SELF_TEMP_min",
    "YS_max", "YS_min",
    "TS_max", "TS_min",
    "EL_max", "EL_min",
    "AKV_max", "AKV_min",
]

PIPELINE_STEEL_SPEC_SCHEMA = {
    key: STEEL_SPEC_SCHEMA[key]
    for key in PIPELINE_SPEC_FIELD_ORDER
    if key in STEEL_SPEC_SCHEMA
}

# 风电塔筒钢沿用热轧板工艺字段与管线钢相同的规格结构。目标钢级、质量
# 等级、冲击试验条件等不写入 matched_result，而是保存在会话标准上下文中，
# 防止被误当作历史实绩字段或传入现有 DLL。
WIND_POWER_STEEL_SPEC_SCHEMA = dict(PIPELINE_STEEL_SPEC_SCHEMA)
WIND_POWER_STEEL_PURPOSE = "风电用钢"
_WIND_POWER_STANDARD_CONTEXT_CACHE: dict[str, dict] = {}

PIPELINE_GBT9711_PSL1_CHEMISTRY_BY_GRADE = {
    # GB/T 9711-2023 表4，t <= 25.0 mm，PSL1 焊管行；脚注a适用 Cu/Ni/Cr/Mo，脚注g适用 B。
    # 表4的 V/Nb/Ti 单项没有独立上限，仅规定 Nb+V+Ti<=0.15；这里把
    # 0.15 作为每个单项的必要上限，组合约束仍由标准提示词约束模型。
    "X42": {"C_max": 0.26, "MN_max": 1.30, "P_max": 0.030, "S_max": 0.030, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.15, "B_max": 0.001},
    "X46": {"C_max": 0.26, "MN_max": 1.40, "P_max": 0.030, "S_max": 0.030, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.15, "B_max": 0.001},
    "X52": {"C_max": 0.26, "MN_max": 1.40, "P_max": 0.030, "S_max": 0.030, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.15, "B_max": 0.001},
    "X56": {"C_max": 0.26, "MN_max": 1.40, "P_max": 0.030, "S_max": 0.030, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.15, "B_max": 0.001},
    "X60": {"C_max": 0.26, "MN_max": 1.40, "P_max": 0.030, "S_max": 0.030, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.15, "B_max": 0.001},
    "X65": {"C_max": 0.26, "MN_max": 1.45, "P_max": 0.030, "S_max": 0.030, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.15, "B_max": 0.001},
    "X70": {"C_max": 0.26, "MN_max": 1.65, "P_max": 0.030, "S_max": 0.030, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.15, "B_max": 0.001},
}

PIPELINE_GBT9711_PSL1_TENSILE_BY_GRADE = {
    # GB/T 9711-2023 表6，PSL1 钢管管体拉伸性能。
    "X42": {"YS_min": 290.0, "TS_min": 415.0},
    "X46": {"YS_min": 320.0, "TS_min": 435.0},
    "X52": {"YS_min": 360.0, "TS_min": 460.0},
    "X56": {"YS_min": 390.0, "TS_min": 490.0},
    "X60": {"YS_min": 415.0, "TS_min": 520.0},
    "X65": {"YS_min": 450.0, "TS_min": 535.0},
    "X70": {"YS_min": 485.0, "TS_min": 570.0},
}

PIPELINE_GBT9711_PSL2_CHEMISTRY_BY_GRADE_CONDITION = {
    # GB/T 9711-2023 表5，t <= 25.0 mm，PSL2。脚注 h/i/l 的合金元素默认限制同步到 Cu/Ni/Cr/Mo/B。
    ("X42", "R"): {"C_max": 0.24, "SI_max": 0.40, "MN_max": 1.20, "P_max": 0.025, "S_max": 0.015, "V_max": 0.06, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X42", "N"): {"C_max": 0.24, "SI_max": 0.40, "MN_max": 1.20, "P_max": 0.025, "S_max": 0.015, "V_max": 0.06, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X46", "N"): {"C_max": 0.24, "SI_max": 0.40, "MN_max": 1.40, "P_max": 0.025, "S_max": 0.015, "V_max": 0.07, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X52", "N"): {"C_max": 0.24, "SI_max": 0.45, "MN_max": 1.40, "P_max": 0.025, "S_max": 0.015, "V_max": 0.10, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X56", "N"): {"C_max": 0.24, "SI_max": 0.45, "MN_max": 1.40, "P_max": 0.025, "S_max": 0.015, "V_max": 0.10, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X60", "N"): {"C_max": 0.24, "SI_max": 0.45, "MN_max": 1.40, "P_max": 0.025, "S_max": 0.015, "V_max": 0.10, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X42", "Q"): {"C_max": 0.18, "SI_max": 0.45, "MN_max": 1.40, "P_max": 0.025, "S_max": 0.015, "V_max": 0.05, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X46", "Q"): {"C_max": 0.18, "SI_max": 0.45, "MN_max": 1.40, "P_max": 0.025, "S_max": 0.015, "V_max": 0.05, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X52", "Q"): {"C_max": 0.18, "SI_max": 0.45, "MN_max": 1.50, "P_max": 0.025, "S_max": 0.015, "V_max": 0.05, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X56", "Q"): {"C_max": 0.18, "SI_max": 0.45, "MN_max": 1.50, "P_max": 0.025, "S_max": 0.015, "V_max": 0.07, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X60", "Q"): {"C_max": 0.18, "SI_max": 0.45, "MN_max": 1.70, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X65", "Q"): {"C_max": 0.18, "SI_max": 0.45, "MN_max": 1.70, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X70", "Q"): {"C_max": 0.18, "SI_max": 0.45, "MN_max": 1.80, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X80", "Q"): {"C_max": 0.18, "SI_max": 0.45, "MN_max": 1.90, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 1.00, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.004},
    ("X90", "Q"): {"C_max": 0.16, "SI_max": 0.45, "MN_max": 1.90, "P_max": 0.020, "S_max": 0.010, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 1.00, "CR_max": 0.55, "MO_max": 0.80, "B_max": 0.004},
    ("X100", "Q"): {"C_max": 0.16, "SI_max": 0.45, "MN_max": 1.90, "P_max": 0.020, "S_max": 0.010, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 1.00, "CR_max": 0.55, "MO_max": 0.80, "B_max": 0.004},
    ("X42", "M"): {"C_max": 0.22, "SI_max": 0.45, "MN_max": 1.30, "P_max": 0.025, "S_max": 0.015, "V_max": 0.05, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X46", "M"): {"C_max": 0.22, "SI_max": 0.45, "MN_max": 1.30, "P_max": 0.025, "S_max": 0.015, "V_max": 0.05, "NB_max": 0.05, "TI_max": 0.04, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X52", "M"): {"C_max": 0.22, "SI_max": 0.45, "MN_max": 1.40, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X56", "M"): {"C_max": 0.22, "SI_max": 0.45, "MN_max": 1.40, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.30, "CR_max": 0.30, "MO_max": 0.15, "B_max": 0.001},
    ("X60", "M"): {"C_max": 0.12, "SI_max": 0.45, "MN_max": 1.60, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X65", "M"): {"C_max": 0.12, "SI_max": 0.45, "MN_max": 1.60, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X70", "M"): {"C_max": 0.12, "SI_max": 0.45, "MN_max": 1.70, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 0.50, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X80", "M"): {"C_max": 0.12, "SI_max": 0.45, "MN_max": 1.85, "P_max": 0.025, "S_max": 0.015, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 1.00, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X90", "M"): {"C_max": 0.10, "SI_max": 0.55, "MN_max": 2.10, "P_max": 0.020, "S_max": 0.010, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 1.00, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.001},
    ("X100", "M"): {"C_max": 0.10, "SI_max": 0.55, "MN_max": 2.10, "P_max": 0.020, "S_max": 0.010, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 1.00, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.004},
    ("X120", "M"): {"C_max": 0.10, "SI_max": 0.55, "MN_max": 2.10, "P_max": 0.020, "S_max": 0.010, "V_max": 0.15, "NB_max": 0.15, "TI_max": 0.15, "CU_max": 0.50, "NI_max": 1.00, "CR_max": 0.50, "MO_max": 0.50, "B_max": 0.004},
}

PIPELINE_GBT9711_PSL2_TENSILE_BY_GRADE = {
    # GB/T 9711-2023 表7，PSL2 钢管管体拉伸性能。
    "X42": {"YS_min": 290.0, "YS_max": 495.0, "TS_min": 415.0, "TS_max": 655.0},
    "X46": {"YS_min": 320.0, "YS_max": 525.0, "TS_min": 435.0, "TS_max": 655.0},
    "X52": {"YS_min": 360.0, "YS_max": 530.0, "TS_min": 460.0, "TS_max": 760.0},
    "X56": {"YS_min": 390.0, "YS_max": 545.0, "TS_min": 490.0, "TS_max": 760.0},
    "X60": {"YS_min": 415.0, "YS_max": 565.0, "TS_min": 520.0, "TS_max": 760.0},
    "X65": {"YS_min": 450.0, "YS_max": 600.0, "TS_min": 535.0, "TS_max": 760.0},
    "X70": {"YS_min": 485.0, "YS_max": 635.0, "TS_min": 570.0, "TS_max": 760.0},
    "X80": {"YS_min": 555.0, "YS_max": 705.0, "TS_min": 625.0, "TS_max": 825.0},
    "X90": {"YS_min": 625.0, "YS_max": 775.0, "TS_min": 695.0, "TS_max": 915.0},
    "X100": {"YS_min": 690.0, "YS_max": 840.0, "TS_min": 760.0, "TS_max": 990.0},
    "X120": {"YS_min": 830.0, "YS_max": 1050.0, "TS_min": 915.0, "TS_max": 1145.0},
}

PIPELINE_GBT9711_GRADE_ALIASES = {
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

PIPELINE_SPEC_SYSTEM_PROMPT = """你是一个管线钢标准管理助手。请根据用户提示词和知识库检索结果，提取管线钢成分、工艺和性能标准的上下限。

## 管线钢规则

1. 只输出 JSON 对象，不要 Markdown，不要解释。
2. 输出字段必须严格使用下方“管线钢输出JSON格式”，不要新增、删除或重命名字段。
3. 当前用途固定为“管线钢”；适用场景包括管线钢、管线用钢、油气输送管线钢、API 管线钢以及各类 X 系列/L 系列管线钢牌号。
4. 必须先确定目标牌号或牌号系列，并在同一牌号或同一牌号系列内提取成分、工艺、性能；禁止跨牌号拼接。
5. 如果用户明确提出的牌号不在程序确定性兜底表中，必须由你依据知识库中该牌号的标准资料，按照当前固定 JSON 结构给出成分、工艺和性能标准范围；不得改用相邻牌号、较低牌号或其他用途钢种的数据补齐。
6. 成分字段为 C/SI/MN/P/S/N/NB/V/TI/AL/ALS/CU/CR/NI/CO/MO/B，单位为 wt%。
7. 工艺字段只使用管线钢生产字段：FET_min/max(精轧开轧温度)、FDT_min/max(精轧终轧温度)、FURNACE_EXIT_TEMP_min/max(出炉温度)、SLAB_THICK_min/max(板坯厚度)、AIM_THICK_min/max(成品厚度)、TEMP_ENTR_min/max(入水温度)、FEH_min/max(中间坯厚度)、SELF_TEMP_min/max(返红温度)。
8. 只按第7条列出的管线钢生产字段提取工艺；不要输出其他温度或热处理字段。
9. 性能字段为 YS_min/max、TS_min/max、EL_min/max、AKV_min/max；AKV 表示冲击功，单位 J。
10. 用户给出的单点目标值必须由你转换为上下限窗口，而不是直接输出标准整段范围：
   - 成分字段：如“C要在0.32左右”“C约0.32”“C目标0.32”，按目标值上下浮动约5%输出，例如 C_min=0.304、C_max=0.336；若下限小于0则取0。
   - 性能字段：如“YS在450MPa左右”“抗拉强度约600MPa”“冲击功目标120J”，按目标值上下浮动约5%输出到对应 min/max；若下限小于0则取0。
   - 板坯厚度字段：SLAB_THICK_min/max 只表示用户目标板坯厚度窗口；如“板坯厚度320mm”，按上下各2mm输出 SLAB_THICK_min=318.0、SLAB_THICK_max=322.0，绝对不能把板坯厚度写入 AIM_THICK。
   - 成品厚度字段：AIM_THICK_min/max 只表示用户目标成品厚度窗口；若用户只给单点成品厚度（如“厚度28mm”“28mm管线钢”“目标厚度28mm”），默认按目标厚度上下各2mm设置窗口，例如28mm应输出 AIM_THICK_min=26.0、AIM_THICK_max=30.0；若目标厚度小于2mm导致下限小于0，则 AIM_THICK_min 取0。
11. 用户明确给出上限/下限时，严格按用户边界输出，例如“C最大值0.2”只设置 C_max=0.2，“YS不低于450MPa”只设置 YS_min=450；用户明确要求固定值或上下限相同（如“AIM_THICK_min=28且AIM_THICK_max=28”“厚度上下限均为28mm”“固定厚度28mm”“精确厚度28mm”“C固定0.32”）时，才输出 min=max；用户明确给出范围（如“厚度26到30mm”“C 0.30到0.34”），则按用户范围输出。
12. 用户提示词中的成分、性能、厚度显式要求优先级高于知识库标准和国标兜底；知识库用于补齐用户没有指定的字段，不得覆盖用户明确给出的目标值、上下限或范围。
13. 如果知识库中没有某字段数据，该字段保持默认值 0/9999；不得用其他牌号数据补齐。
14. 当标准规定 Nb+V+Ti 的合计上限为0.15 wt%时，NB_max、V_max、TI_max 的单项值均不得超过0.15，后续成分设计还必须保证三者实际含量之和不超过0.15 wt%。
15. 字段顺序必须保持为：用途、厚度、成分、管线钢工艺、性能。

## 管线钢输出JSON格式
""" + json.dumps(PIPELINE_STEEL_SPEC_SCHEMA, ensure_ascii=False, indent=2) + """

## 管线钢字段含义说明
{
    '用途': '钢种用途，例如：管线钢',

    'C_max': 'C成分wt%上限',
    'C_min': 'C成分wt%下限',

    'SI_max': 'Si成分wt%上限',
    'SI_min': 'Si成分wt%下限',

    'MN_max': 'Mn成分wt%上限',
    'MN_min': 'Mn成分wt%下限',

    'P_max': 'P成分wt%上限',
    'P_min': 'P成分wt%下限',

    'S_max': 'S成分wt%上限',
    'S_min': 'S成分wt%下限',

    'N_max': 'N成分wt%上限',
    'N_min': 'N成分wt%下限',

    'NB_max': 'Nb成分wt%上限',
    'NB_min': 'Nb成分wt%下限',

    'V_max': 'V成分wt%上限',
    'V_min': 'V成分wt%下限',

    'TI_max': 'Ti成分wt%上限',
    'TI_min': 'Ti成分wt%下限',

    'AL_max': 'Al成分wt%上限',
    'AL_min': 'Al成分wt%下限',

    'ALS_max': 'Als酸溶铝成分wt%上限',
    'ALS_min': 'Als酸溶铝成分wt%下限',

    'CU_max': 'Cu成分wt%上限',
    'CU_min': 'Cu成分wt%下限',

    'CR_max': 'Cr成分wt%上限',
    'CR_min': 'Cr成分wt%下限',

    'NI_max': 'Ni成分wt%上限',
    'NI_min': 'Ni成分wt%下限',

    'CO_max': 'Co成分wt%上限',
    'CO_min': 'Co成分wt%下限',

    'MO_max': 'Mo成分wt%上限',
    'MO_min': 'Mo成分wt%下限',

    'B_max': 'B成分wt%上限',
    'B_min': 'B成分wt%下限',

    'FET_max': '精轧入口温度上限',
    'FET_min': '精轧入口温度下限',

    'FDT_max': '终轧温度上限',
    'FDT_min': '终轧温度下限',

    'FURNACE_EXIT_TEMP_max': '出炉温度上限',
    'FURNACE_EXIT_TEMP_min': '出炉温度下限',

    'SLAB_THICK_max': '板坯厚度上限，单位mm',
    'SLAB_THICK_min': '板坯厚度下限，单位mm',

    'AIM_THICK_max': '成品厚度上限，单位mm',
    'AIM_THICK_min': '成品厚度下限，单位mm',

    'TEMP_ENTR_max': '入水温度上限',
    'TEMP_ENTR_min': '入水温度下限',

    'FEH_max': '中间坯厚度上限，单位mm',
    'FEH_min': '中间坯厚度下限，单位mm',

    'SELF_TEMP_max': '返红温度上限',
    'SELF_TEMP_min': '返红温度下限',

    'YS_max': '屈服强度上限，单位MPa',
    'YS_min': '屈服强度下限，单位MPa',

    'TS_max': '抗拉强度上限，单位MPa',
    'TS_min': '抗拉强度下限，单位MPa',

    'EL_max': '延伸率上限，单位%',
    'EL_min': '延伸率下限，单位%',

    'AKV_max': '冲击功上限，单位J',
    'AKV_min': '冲击功下限，单位J'
}
"""

WIND_POWER_SPEC_SYSTEM_PROMPT = """你是风电塔筒用热机械轧制钢标准管理助手。请根据用户提示词和知识库检索结果，提取成分、热轧工艺和性能标准上下限。

必须严格遵守：
1. 只输出 JSON 对象，不要 Markdown、解释或代码块；字段必须严格使用给定 Schema，不能新增、删除或改名。
2. 当前用途固定为“风电用钢”，产品为用户明确的海上、陆上或通用风电塔筒用钢板，交货状态固定为热机械轧制 TMCP 的 M 级。不得将“海上风电”改写为“陆上风电”。
3. 用户未指定钢级时，只能在 Q355M、Q390M、Q420M、Q460M、Q500M、Q550M、Q620M、Q690M 中选择一个与厚度和性能匹配的目标钢级；不得混用不同钢级、不同质量等级或热轧/正火状态的数据。
4. 用户明确给出的钢级、成品厚度、板坯厚度、成分、性能和工艺要求优先级最高。用户指定 N、AR 等非 M 交货状态时，不得悄悄替换为 M 级。
5. 成分字段为 C/SI/MN/P/S/N/NB/V/TI/AL/ALS/CU/CR/NI/CO/MO/B，单位 wt%。工艺字段只使用 FET、FDT、FURNACE_EXIT_TEMP、SLAB_THICK、AIM_THICK、TEMP_ENTR、FEH、SELF_TEMP。
6. 性能字段为 YS、TS、EL、AKV。AKV 的试验温度和取样方向由质量等级决定，必须在后续上下文中说明，但不新增 Schema 字段。
7. 不得用管线钢 X 系列、海工钢或其他用途钢的标准替代风电塔筒钢标准。
8. 用户单点板坯厚度按上下各 2 mm 形成 SLAB_THICK 窗口；单点成品厚度按上下各 2 mm 形成 AIM_THICK 窗口。板坯厚度不得写入 AIM_THICK。
9. 用户未指定的字段可依据知识库补齐；资料不足时保守给范围，不得虚构疲劳、Z 向性能、腐蚀或焊接试验数值。

输出 Schema：
""" + json.dumps(WIND_POWER_STEEL_SPEC_SCHEMA, ensure_ascii=False, indent=2)

STEEL_SPEC_SYSTEM_PROMPT = """你是一个知识库标准管理助手。请根据用户提示词和知识库检索结果，提取钢材成分和性能标准的上下限。

## 规则

1. 严格按照下面的JSON格式输出，不得缺项
2. 所有元素含量的默认下限为0，默认上限为9999
3. 如果用户在提示词中提出某一项的上下限，则替换为用户要求的值
4. 如果用户提出成分、性能或工艺字段的具体目标值（如“C要在0.32左右”“YS在450MPa左右”“抗拉强度约600MPa”），则根据材料学经验设置合理的上下浮动范围，通常按目标值上下浮动约5%输出到对应 min/max；若下限小于0则取0。用户明确给出上限/下限时只设置对应边界；用户明确要求固定值或上下限相同时，才输出 min=max。
   厚度字段例外：用户给出的单点厚度不是固定上下限，也不是要求输出整段标准允许范围，而是目标设计厚度。除非用户明确要求固定 THK_min/THK_max，否则应围绕该目标厚度给出合理的设计厚度窗口。
5. 如果知识库中有相关规格数据，优先使用知识库中的数据
6. 成分元素(C/SI/MN/P/S/N/NB/V/TI/AL/ALS/CU/CR/NI/CO/MO/B)单位为质量百分比(wt%)
7. 成品厚度规格字段为THK_min/THK_max，单位为mm。若用户只给出单一目标厚度（如“5mm”“厚度规格5mm”“5mm厚度”“目标厚度5mm”），应由你根据目标厚度给出窄范围设计窗口，通常可按目标厚度上下浮动约2mm：例如目标厚度5mm可输出THK_min约3、THK_max约7。该窗口必须落在知识库厚度范围表允许范围内；如标准为“≤120”，只表示上边界不能超过120，不得直接输出0~120。若目标厚度过小导致下限小于0，则THK_min取0。只有用户明确写“THK_min=5且THK_max=5”“厚度上下限均为5mm”“固定输出厚度范围5~5mm”时，才输出THK_min=5、THK_max=5；若用户明确给出范围（如“厚度80到120mm”），则按用户范围输出。
8. 温度工艺字段单位均为℃：SOAKING_TEMP_min/max(均热温度)、FET_min/max(精轧开轧温度)、FDT_min/max(终轧温度)、CT_min/max(卷取温度)、QUENCHING_TEMP_min/max(淬火温度)、TEMPERING_TEMP_min/max(回火温度)。
9. 性能指标(YS屈服强度/TS抗拉强度)单位为MPa，(EL延伸率/EL断后伸长率)单位为%。
10. 拉伸性能表中“断后伸长率(A50mm) % 不小于”“断后伸长率”“伸长率”必须映射到EL_min；“抗拉强度(Rm) MPa 不小于”“抗拉强度”“拉伸强度”必须映射到TS_min。不要求表名必须是“横向拉伸性能”，只要检索结果中属于拉伸性能/力学性能的表格即可使用。
11. 输出字段顺序必须保持为：用途、成品厚度规格、化学成分、温度工艺、力学性能。
12. **牌号一致性要求：必须先根据用户目标厚度、用途和检索结果确定一个目标牌号（如NM450或NM450D/E），然后THK_min/THK_max、化学成分、温度工艺、力学性能必须全部从同一牌号或同一牌号系列对应的表格行提取。禁止厚度使用NM450、成分使用NM500、性能使用NM400这类跨牌号拼接。若某牌号在某张表中没有数据，该字段保持默认值或按规则说明缺失，不得改用其他牌号的数据补齐。**
13. **合并牌号行处理：如果拉伸性能表把普通牌号和质量等级牌号写在同一行（如“NM450 NM450D/E”），该性能值可同时适用于NM450和NM450D/E，但成分必须继续按最终选定的具体牌号去对应成分表取值。若最终选定普通NM450，则使用普通NM450成分行（例如C_max=0.35）；若最终选定NM450D/E，则使用NM450D/E成分行（例如C_max=0.30）。禁止把普通级和D/E级的成分交叉套用。**
14. **默认选牌策略：如果用户没有明确指定牌号、D/E质量等级、低温韧性、冲击性能或特殊成分要求，则优先选择普通级牌号（如NM450而不是NM450D/E）。只有用户明确要求D/E、低温韧性、冲击性能或指定D/E牌号时，才选择D/E质量等级牌号。随后厚度、成分、性能都按该同一具体牌号或该牌号所在的合并性能行提取；不得为了获得更大的成分范围而选择厚度范围不覆盖用户目标厚度的牌号。**
15. **关键要求：必须在知识库检索结果中仔细查找所有厚度、元素、温度工艺和性能的实际数值。不得将任何字段保留为默认值9999或0，除非知识库中确实没有该字段的数据。化学成分(C/SI/MN/P/S等)、温度工艺(均热/FET/FDT/CT/淬火/回火)和力学性能(YS/TS/EL)必须从检索结果中提取！**
16. "用途"字段：填写钢材的用途描述（如"耐磨钢"或"高强度结构钢"）

## 输出JSON格式（必须包含所有字段）
""" + json.dumps(STEEL_SPEC_SCHEMA, ensure_ascii=False, indent=2)

def _parse_spec_json(text: str) -> dict:
    """从 LLM 返回中解析钢材规格 JSON"""
    # 1. 提取 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. 直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 3. 正则提取最外层 {}
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Fallback
    return dict(STEEL_SPEC_SCHEMA)


def _is_pipeline_purpose(purpose: str) -> bool:
    return str(purpose).strip() == "管线钢"


def _is_wind_power_purpose(purpose: str) -> bool:
    return str(purpose).strip() == WIND_POWER_STEEL_PURPOSE


def _schema_for_purpose(purpose: str) -> dict:
    if _is_pipeline_purpose(purpose):
        return PIPELINE_STEEL_SPEC_SCHEMA
    if _is_wind_power_purpose(purpose):
        return WIND_POWER_STEEL_SPEC_SCHEMA
    return STEEL_SPEC_SCHEMA


def _system_prompt_for_purpose(purpose: str) -> str:
    if _is_pipeline_purpose(purpose):
        return PIPELINE_SPEC_SYSTEM_PROMPT
    if _is_wind_power_purpose(purpose):
        return WIND_POWER_SPEC_SYSTEM_PROMPT
    return STEEL_SPEC_SYSTEM_PROMPT


def _normalize_spec_result(result: dict, purpose: str) -> dict:
    """按 schema 顺序补齐字段，并固定用途。"""
    schema = _schema_for_purpose(purpose)
    normalized = dict(schema)
    if isinstance(result, dict):
        for key in normalized:
            if key in result:
                normalized[key] = result[key]
        for old_key, new_key in STEEL_SPEC_FIELD_ALIASES.items():
            if old_key in result and new_key in normalized:
                normalized[new_key] = result[old_key]
    if _is_pipeline_purpose(purpose) or _is_wind_power_purpose(purpose):
        for key in PIPELINE_EXCLUDED_SPEC_FIELDS:
            normalized.pop(key, None)
    normalized["用途"] = purpose
    return normalized


def _extract_pipeline_slab_thickness_bounds(user_message: str) -> dict:
    """确定性提取用户明确给出的板坯厚度；未命中时返回空字典。

    单点板坯厚度沿用管线钢厚度规格窗口规则，上下各放宽 2 mm。该方法只
    处理“板坯厚度/连铸坯厚度”，避免将其误写到成品厚度 AIM_THICK。
    """
    text = str(user_message or "")
    range_patterns = [
        r"(?:板坯厚度|连铸坯厚度)\D{0,12}(\d+(?:\.\d+)?)\s*(?:到|至|[-~～—–])\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        r"(\d+(?:\.\d+)?)\s*(?:到|至|[-~～—–])\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)\D{0,12}(?:板坯厚度|连铸坯厚度)",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            first = _to_float_or_none(match.group(1))
            second = _to_float_or_none(match.group(2))
            if first is not None and second is not None:
                return {
                    "SLAB_THICK_min": max(0.0, min(first, second)),
                    "SLAB_THICK_max": max(first, second),
                }

    single_patterns = [
        r"(?:板坯厚度|连铸坯厚度)\D{0,12}(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)\D{0,12}(?:板坯厚度|连铸坯厚度)",
    ]
    for pattern in single_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _to_float_or_none(match.group(1))
            if value is not None:
                return {
                    "SLAB_THICK_min": max(0.0, value - 2.0),
                    "SLAB_THICK_max": value + 2.0,
                }
    return {}


# ============================================================
# 公开 API
# ============================================================

def _to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _detect_pipeline_gbt9711_grade_in_text(text: str) -> str | None:
    merged = str(text or "").upper()
    for alias, grade in PIPELINE_GBT9711_GRADE_ALIASES.items():
        if re.search(rf"(?<![A-Z0-9]){alias}(?:[MQRNO]*)?(?![A-Z0-9])", merged):
            return grade
    # 必须从长牌号向短牌号识别，避免 X100/X120 被较短模式误判。
    for grade in ("X120", "X100", "X90", "X80", "X70", "X65", "X60", "X56", "X52", "X46", "X42"):
        if re.search(rf"(?<![A-Z0-9]){grade}(?:[MQRNO]*)?(?![A-Z0-9])", merged):
            return grade
    return None


def _detect_pipeline_gbt9711_grade(*texts: str) -> str | None:
    for text in texts:
        grade = _detect_pipeline_gbt9711_grade_in_text(text)
        if grade:
            return grade
    return None


def _explicit_pipeline_grade_without_deterministic_fallback(text: str) -> str | None:
    """返回用户明确提出、但当前确定性标准表未覆盖的管线钢牌号。"""
    merged = str(text or "").upper()
    x_match = re.search(r"(?<![A-Z0-9])(X\d{2,3})(?:[MQRNO]*)(?![A-Z0-9])", merged)
    if x_match:
        requested_grade = x_match.group(1)
        supported_grades = set(PIPELINE_GBT9711_GRADE_ALIASES.values())
        if requested_grade not in supported_grades:
            return requested_grade

    l_match = re.search(r"(?<![A-Z0-9])(L\d{3})(?:[MQRNO]*)(?![A-Z0-9])", merged)
    if l_match:
        requested_grade = l_match.group(1)
        if requested_grade not in PIPELINE_GBT9711_GRADE_ALIASES:
            return requested_grade
    return None


def _extract_explicit_pipeline_grade_token(text: str) -> str | None:
    """提取用户明确给出的完整管线钢牌号，保留 R/N/Q/M/O 等状态后缀。"""
    merged = str(text or "").upper()
    for pattern in (
        r"(?<![A-Z0-9])(X\d{2,3}[MQRNO]*)(?![A-Z0-9])",
        r"(?<![A-Z0-9])(L\d{3}[MQRNO]*)(?![A-Z0-9])",
    ):
        match = re.search(pattern, merged)
        if match:
            return match.group(1)
    return None


def _detect_pipeline_gbt9711_condition(*texts: str) -> str:
    merged = " ".join(str(text or "") for text in texts).upper()
    grade_tokens = "X42|X46|X52|X56|X60|X65|X70|X80|X90|X100|X120|L290|L320|L360|L390|L415|L450|L485|L555|L625|L690|L830"
    if re.search(rf"(?<![A-Z0-9])(?:{grade_tokens})Q[ON]?(?![A-Z0-9])", merged):
        return "Q"
    if re.search(rf"(?<![A-Z0-9])(?:{grade_tokens})M[ON]?(?![A-Z0-9])", merged):
        return "M"
    if re.search(rf"(?<![A-Z0-9])(?:{grade_tokens})N[ON]?(?![A-Z0-9])", merged):
        return "N"
    if re.search(rf"(?<![A-Z0-9])(?:{grade_tokens})R[ON]?(?![A-Z0-9])", merged):
        return "R"
    if any(keyword in merged for keyword in ("TMCP", "控轧", "控冷")):
        return "M"
    return "M"


def _detect_pipeline_gbt9711_condition_from_standard(*texts: str) -> str:
    merged = " ".join(str(text or "") for text in texts).upper()
    grade_tokens = "X42|X46|X52|X56|X60|X65|X70|X80|X90|X100|X120|L290|L320|L360|L390|L415|L450|L485|L555|L625|L690|L830"
    if re.search(rf"(?<![A-Z0-9])(?:{grade_tokens})Q[ON]?(?![A-Z0-9])", merged):
        return "Q"
    if re.search(rf"(?<![A-Z0-9])(?:{grade_tokens})M[ON]?(?![A-Z0-9])", merged):
        return "M"
    if re.search(rf"(?<![A-Z0-9])(?:{grade_tokens})N[ON]?(?![A-Z0-9])", merged):
        return "N"
    if re.search(rf"(?<![A-Z0-9])(?:{grade_tokens})R[ON]?(?![A-Z0-9])", merged):
        return "R"
    if any(keyword in merged for keyword in ("TMCP", "控轧", "控冷")):
        return "M"
    return ""


def _detect_pipeline_gbt9711_psl(user_message: str, rag_context: str, condition: str) -> str:
    user_text = str(user_message or "").upper()
    context_text = str(rag_context or "").upper()
    if "PSL2" in user_text or "PSL 2" in user_text or condition:
        return "PSL2"
    if "PSL1" in user_text or "PSL 1" in user_text:
        return "PSL1"
    if ("表5" in context_text or "表 5" in context_text or "表7" in context_text or "表 7" in context_text) and not (
        "表4" in context_text or "表 4" in context_text or "表6" in context_text or "表 6" in context_text
    ):
        return "PSL2"
    return "PSL1"


def _infer_pipeline_gbt9711_grade_from_result(result: dict) -> str | None:
    ys_min = _to_float_or_none(result.get("YS_min"))
    ts_min = _to_float_or_none(result.get("TS_min"))
    if ys_min is None and ts_min is None:
        return None
    for table in (PIPELINE_GBT9711_PSL1_TENSILE_BY_GRADE, PIPELINE_GBT9711_PSL2_TENSILE_BY_GRADE):
        for grade, limits in table.items():
            if ys_min == limits["YS_min"] or ts_min == limits["TS_min"]:
                return grade
    return None


def _is_default_spec_value(field_name: str, value) -> bool:
    number = _to_float_or_none(value)
    if number is None:
        return True
    if field_name.endswith("_min"):
        return number == 0.0
    if field_name.endswith("_max"):
        return number == 9999.0
    return False


def _fill_pipeline_defaults(target: dict, fallback: dict, preserve_existing_min_max_pair: bool = True) -> None:
    for key, value in fallback.items():
        if preserve_existing_min_max_pair and key.endswith("_max"):
            paired_min_key = f"{key[:-4]}_min"
            if paired_min_key in target and not _is_default_spec_value(paired_min_key, target.get(paired_min_key)):
                continue
        if key not in target or _is_default_spec_value(key, target.get(key)):
            target[key] = value


USER_SPEC_FIELD_ALIASES = {
    "C": ("C", "碳"),
    "SI": ("SI", "Si", "硅"),
    "MN": ("MN", "Mn", "锰"),
    "P": ("P", "磷"),
    "S": ("S", "硫"),
    "N": ("N", "氮"),
    "NB": ("NB", "Nb", "铌"),
    "V": ("V", "钒"),
    "TI": ("TI", "Ti", "钛"),
    "AL": ("AL", "Al", "铝"),
    "ALS": ("ALS", "Als", "酸溶铝"),
    "CU": ("CU", "Cu", "铜"),
    "CR": ("CR", "Cr", "铬"),
    "NI": ("NI", "Ni", "镍"),
    "CO": ("CO", "Co", "钴"),
    "MO": ("MO", "Mo", "钼"),
    "B": ("B", "硼"),
    "YS": ("YS", "屈服", "屈服强度"),
    "TS": ("TS", "抗拉", "抗拉强度"),
    "EL": ("EL", "延伸", "延伸率", "断后伸长率"),
    "AKV": ("AKV", "冲击功"),
}


def _user_explicit_spec_fields(user_message: str) -> set[str]:
    text = str(user_message or "")
    explicit = set()
    for prefix, aliases in USER_SPEC_FIELD_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias)
            max_patterns = [
                rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])\s*(?:_?max|最大值|最大|上限|不大于|小于等于|≤|<=)\s*[0-9]+(?:\.[0-9]+)?",
                rf"(?:最大值|最大|上限|不大于|小于等于|≤|<=)\s*[0-9]+(?:\.[0-9]+)?\s*(?:的)?\s*{escaped}",
            ]
            min_patterns = [
                rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])\s*(?:_?min|最小值|最小|下限|不小于|大于等于|≥|>=)\s*[0-9]+(?:\.[0-9]+)?",
                rf"(?:最小值|最小|下限|不小于|大于等于|≥|>=)\s*[0-9]+(?:\.[0-9]+)?\s*(?:的)?\s*{escaped}",
            ]
            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in max_patterns):
                explicit.add(f"{prefix}_max")
            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in min_patterns):
                explicit.add(f"{prefix}_min")
    return explicit


def _user_target_spec_ranges(user_message: str) -> dict:
    """Extract user target values like 'C要在0.32左右' into narrow min/max ranges."""
    text = str(user_message or "")
    ranges = {}
    for prefix, aliases in USER_SPEC_FIELD_ALIASES.items():
        if prefix in {"YS", "TS", "EL", "AKV"}:
            continue
        for alias in aliases:
            escaped = re.escape(alias)
            patterns = [
                rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])\s*(?:含量)?\s*(?:要在|需要在|希望在|控制在|目标为|目标|要|在|为)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:左右|附近|上下|约|大约)?",
                rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])\s*(?:含量)?\s*(?:约|大约)\s*([0-9]+(?:\.[0-9]+)?)",
                rf"([0-9]+(?:\.[0-9]+)?)\s*(?:左右|附近|上下)\s*(?:的)?\s*(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
                rf"(?:约|大约)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:的)?\s*(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
            ]
            matched_value = None
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    matched_value = _to_float_or_none(match.group(1))
                    break
            if matched_value is None:
                continue
            tolerance = max(abs(matched_value) * 0.05, 0.001)
            ranges[f"{prefix}_min"] = max(0.0, matched_value - tolerance)
            ranges[f"{prefix}_max"] = matched_value + tolerance
            break
    return ranges


def _apply_pipeline_standard_overrides(
    target: dict,
    values: dict,
    clear_keys: tuple[str, ...] = (),
    protected_keys: set[str] | None = None,
) -> None:
    protected_keys = protected_keys or set()
    for key in clear_keys:
        if key in protected_keys:
            continue
        if key in target and not _is_default_spec_value(key, target.get(key)):
            continue
        if key.endswith("_min"):
            target[key] = 0.0
        elif key.endswith("_max"):
            target[key] = 9999.0
    for key, value in values.items():
        if key in protected_keys:
            continue
        if key not in target or _is_default_spec_value(key, target.get(key)):
            target[key] = value


# 未收录牌号进入最终 LLM 兜底时，这些字段组必须给出有效范围。厚度仍由
# extract_steel_specs 末尾的确定性提取覆盖，避免模型混淆板坯厚度和成品厚度。
PIPELINE_UNKNOWN_GRADE_COMPONENT_PREFIXES = (
    "C", "SI", "MN", "P", "S", "N", "NB", "V", "TI",
    "AL", "ALS", "CU", "CR", "NI", "CO", "MO", "B",
)
PIPELINE_UNKNOWN_GRADE_PROCESS_PREFIXES = (
    "FET", "FDT", "FURNACE_EXIT_TEMP", "TEMP_ENTR", "FEH", "SELF_TEMP",
)
PIPELINE_UNKNOWN_GRADE_PERFORMANCE_PREFIXES = ("YS", "TS", "EL", "AKV")


def _missing_unknown_grade_spec_ranges(result: dict) -> list[str]:
    """检查未知牌号 LLM 是否真正补全了成分、工艺和性能范围。"""
    missing = []
    prefixes = (
        PIPELINE_UNKNOWN_GRADE_COMPONENT_PREFIXES
        + PIPELINE_UNKNOWN_GRADE_PROCESS_PREFIXES
        + PIPELINE_UNKNOWN_GRADE_PERFORMANCE_PREFIXES
    )
    for prefix in prefixes:
        min_key = f"{prefix}_min"
        max_key = f"{prefix}_max"
        if _is_default_spec_value(min_key, result.get(min_key)) and _is_default_spec_value(
            max_key, result.get(max_key)
        ):
            missing.append(prefix)
    return missing


def _llm_response_text(response) -> str:
    """兼容 LangChain 文本消息及分块 content，统一提取模型正文。"""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
        return "".join(parts)
    return str(content or "")


def _resolve_unsupported_pipeline_grade_with_llm(
    initial_result: dict,
    user_message: str,
    rag_context: str,
    unsupported_grade: str,
) -> dict:
    """使用 LLM 作为本地标准表未收录牌号的最后一级兜底。

    该方法只在用户明确给出、且 X65/X70/X80 等确定性标准表无法覆盖的
    牌号时调用。模型失败或字段仍不完整时保留最后一次候选结果，但绝不再
    反推、替换或套用任何相邻牌号标准。
    """
    grade_token = _extract_explicit_pipeline_grade_token(user_message) or unsupported_grade
    current = _normalize_spec_result(initial_result, "管线钢")
    target_ranges = _user_target_spec_ranges(user_message)
    if target_ranges:
        current.update(target_ranges)

    last_missing = _missing_unknown_grade_spec_ranges(current)
    for attempt in range(1, 3):
        repair_instruction = ""
        if attempt > 1:
            repair_instruction = (
                "\n上一次结果仍缺少以下有效范围："
                + "、".join(last_missing)
                + "。请本次逐项补齐，不能继续保留默认的 0/9999 范围。\n"
            )

        prompt = f"""你是管线钢牌号标准与生产工艺专家。当前程序的本地确定性标准表中没有牌号 {grade_token}，现在由你执行最后一级兜底。

必须严格遵守：
1. 只针对用户明确要求的 {grade_token} 给出范围，禁止替换为 X80、X70、X65、相邻牌号或较低牌号。
2. 优先依据知识库中的该牌号标准资料；资料不足时，结合该牌号强度等级和管线钢专业知识保守给出可用于数据库匹配与工艺设计的合理范围。
3. 必须完整给出化学成分、生产工艺和力学性能范围。化学成分包含 C、SI、MN、P、S、N、NB、V、TI、AL、ALS、CU、CR、NI、CO、MO、B；工艺包含 FURNACE_EXIT_TEMP、FET、FDT、TEMP_ENTR、FEH、SELF_TEMP；性能包含 YS、TS、EL、AKV。
4. 对确实不应添加的合金元素，可给出合理的零值或极低上限，但不得把整个字段继续保留为默认的 min=0、max=9999。
5. 用户明确提出的牌号、板坯厚度、成品厚度、成分目标、工艺目标和性能目标具有最高优先级。
6. 输出必须是一个完整 JSON 对象，字段、字段顺序和数量必须与给定 Schema 完全一致；不要输出 Markdown、解释或代码块。
7. 这是最后一级兜底。输出后程序不会再使用任何其他牌号标准替换你的结果。
{repair_instruction}
【用户要求】
{user_message}

【知识库检索结果】
{rag_context if rag_context else "（没有检索到该牌号的直接资料，请依据专业知识保守补全。）"}

【当前规格结果，仅供修正和补全】
{json.dumps(current, ensure_ascii=False, indent=2)}

【必须严格遵循的输出 Schema】
{json.dumps(PIPELINE_STEEL_SPEC_SCHEMA, ensure_ascii=False, indent=2)}
"""
        try:
            raw_response = deepseek_Llm.invoke(prompt)
            candidate = _parse_spec_json(_llm_response_text(raw_response))
            current = _normalize_spec_result(candidate, "管线钢")
            if target_ranges:
                current.update(target_ranges)
            last_missing = _missing_unknown_grade_spec_ranges(current)
            if not last_missing:
                print(
                    f"[钢规格提取] 牌号 {grade_token} 已由最终 LLM 兜底补全成分、工艺和性能范围"
                )
                return current
            print(
                f"[钢规格提取] 牌号 {grade_token} 最终 LLM 兜底第 {attempt} 次结果仍缺字段: "
                + ", ".join(last_missing)
            )
        except Exception as exc:
            print(f"[钢规格提取] 牌号 {grade_token} 最终 LLM 兜底第 {attempt} 次调用失败: {exc}")

    print(
        f"[钢规格提取] 牌号 {grade_token} 最终 LLM 兜底未完整补齐，"
        "保留最后结果且不再套用其他牌号标准"
    )
    return current


def _apply_pipeline_gbt9711_limits(
    result: dict,
    user_message: str,
    rag_context: str,
) -> dict:
    unsupported_grade = _explicit_pipeline_grade_without_deterministic_fallback(user_message)
    if unsupported_grade:
        # 只有本地确定性标准表未覆盖的明确牌号才调用这一级 LLM。返回后立即
        # 结束标准兜底流程，禁止再由性能值反推并套用其他牌号。
        return _resolve_unsupported_pipeline_grade_with_llm(
            result,
            user_message,
            rag_context,
            unsupported_grade,
        )

    grade = _detect_pipeline_gbt9711_grade(user_message, rag_context)
    if grade is None:
        grade = _infer_pipeline_gbt9711_grade_from_result(result)
    if grade is None:
        return result

    fixed = dict(result)
    protected_keys = _user_explicit_spec_fields(user_message)
    target_ranges = _user_target_spec_ranges(user_message)
    if target_ranges:
        fixed.update(target_ranges)
        protected_keys.update(target_ranges)
    condition = _detect_pipeline_gbt9711_condition_from_standard(user_message)
    psl = _detect_pipeline_gbt9711_psl(user_message, rag_context, condition)
    # GB/T 9711-2023 的 PSL1 主表最高到 X70；X80 及以上牌号只能按
    # PSL2 表5/表7应用，不能因用户未显式写 PSL2 而落入空的 PSL1 兜底。
    if grade in {"X80", "X90", "X100", "X120"}:
        psl = "PSL2"
    if psl == "PSL2":
        condition = condition or "M"
        chemistry = PIPELINE_GBT9711_PSL2_CHEMISTRY_BY_GRADE_CONDITION.get((grade, condition))
        tensile = PIPELINE_GBT9711_PSL2_TENSILE_BY_GRADE.get(grade)
        psl2_clear_keys = (
            "N_max", "AL_max", "ALS_max", "CO_max", "NB_max", "V_max", "TI_max",
        )
        if chemistry:
            _apply_pipeline_standard_overrides(fixed, chemistry, clear_keys=psl2_clear_keys, protected_keys=protected_keys)
        if tensile:
            _apply_pipeline_standard_overrides(fixed, tensile, protected_keys=protected_keys)
    else:
        chemistry = PIPELINE_GBT9711_PSL1_CHEMISTRY_BY_GRADE.get(grade)
        tensile = PIPELINE_GBT9711_PSL1_TENSILE_BY_GRADE.get(grade)
        psl1_clear_keys = (
            "SI_max", "N_max", "NB_max", "V_max", "TI_max", "AL_max",
            "YS_max", "TS_max",
        )
        if chemistry:
            _apply_pipeline_standard_overrides(fixed, chemistry, clear_keys=psl1_clear_keys, protected_keys=protected_keys)
        if tensile:
            _apply_pipeline_standard_overrides(fixed, tensile, protected_keys=protected_keys)
    return fixed


# GB/T 1591-2018 表5、表6、表10、表11。风电分支只允许 TMCP(M) 钢板，
# 因而不将热轧(AR)和正火(N)行混入同一套兜底数据。
WIND_TMCP_GRADES = ("Q355M", "Q390M", "Q420M", "Q460M", "Q500M", "Q550M", "Q620M", "Q690M")
WIND_TMCP_QUALITY_GRADES = {
    "Q355M": ("B", "C", "D", "E", "F"),
    "Q390M": ("B", "C", "D", "E"),
    "Q420M": ("B", "C", "D", "E"),
    "Q460M": ("C", "D", "E"),
    "Q500M": ("C", "D", "E"),
    "Q550M": ("C", "D", "E"),
    "Q620M": ("C", "D", "E"),
    "Q690M": ("C", "D", "E"),
}
WIND_TMCP_CHEMISTRY = {
    "Q355M": {"C_max": 0.14, "SI_max": 0.50, "MN_max": 1.60, "NB_min": 0.01, "NB_max": 0.05, "V_min": 0.01, "V_max": 0.10, "TI_min": 0.006, "TI_max": 0.05, "CR_max": 0.30, "NI_max": 0.50, "CU_max": 0.40, "MO_max": 0.10, "N_max": 0.015, "B_max": 0.0, "CO_max": 0.0, "ALS_min": 0.015},
    "Q390M": {"C_max": 0.15, "SI_max": 0.50, "MN_max": 1.70, "NB_min": 0.01, "NB_max": 0.05, "V_min": 0.01, "V_max": 0.12, "TI_min": 0.006, "TI_max": 0.05, "CR_max": 0.30, "NI_max": 0.50, "CU_max": 0.40, "MO_max": 0.10, "N_max": 0.015, "B_max": 0.0, "CO_max": 0.0, "ALS_min": 0.015},
    "Q420M": {"C_max": 0.16, "SI_max": 0.50, "MN_max": 1.70, "NB_min": 0.01, "NB_max": 0.05, "V_min": 0.01, "V_max": 0.12, "TI_min": 0.006, "TI_max": 0.05, "CR_max": 0.30, "NI_max": 0.80, "CU_max": 0.40, "MO_max": 0.20, "N_max": 0.025, "B_max": 0.0, "CO_max": 0.0, "ALS_min": 0.015},
    "Q460M": {"C_max": 0.16, "SI_max": 0.60, "MN_max": 1.70, "NB_min": 0.01, "NB_max": 0.05, "V_min": 0.01, "V_max": 0.12, "TI_min": 0.006, "TI_max": 0.05, "CR_max": 0.30, "NI_max": 0.80, "CU_max": 0.40, "MO_max": 0.20, "N_max": 0.025, "B_max": 0.0, "CO_max": 0.0, "ALS_min": 0.015},
    "Q500M": {"C_max": 0.18, "SI_max": 0.60, "MN_max": 1.80, "NB_min": 0.01, "NB_max": 0.11, "V_min": 0.01, "V_max": 0.12, "TI_min": 0.006, "TI_max": 0.05, "CR_max": 0.60, "NI_max": 0.80, "CU_max": 0.55, "MO_max": 0.20, "N_max": 0.025, "B_max": 0.004, "CO_max": 0.0, "ALS_min": 0.015},
    "Q550M": {"C_max": 0.18, "SI_max": 0.60, "MN_max": 2.00, "NB_min": 0.01, "NB_max": 0.11, "V_min": 0.01, "V_max": 0.12, "TI_min": 0.006, "TI_max": 0.05, "CR_max": 0.80, "NI_max": 0.80, "CU_max": 0.80, "MO_max": 0.30, "N_max": 0.025, "B_max": 0.004, "CO_max": 0.0, "ALS_min": 0.015},
    "Q620M": {"C_max": 0.18, "SI_max": 0.60, "MN_max": 2.00, "NB_min": 0.01, "NB_max": 0.11, "V_min": 0.01, "V_max": 0.12, "TI_min": 0.006, "TI_max": 0.05, "CR_max": 1.00, "NI_max": 0.80, "CU_max": 0.80, "MO_max": 0.30, "N_max": 0.025, "B_max": 0.004, "CO_max": 0.0, "ALS_min": 0.015},
    "Q690M": {"C_max": 0.18, "SI_max": 0.60, "MN_max": 2.00, "NB_min": 0.01, "NB_max": 0.11, "V_min": 0.01, "V_max": 0.12, "TI_min": 0.006, "TI_max": 0.05, "CR_max": 1.00, "NI_max": 0.80, "CU_max": 0.80, "MO_max": 0.30, "N_max": 0.025, "B_max": 0.004, "CO_max": 0.0, "ALS_min": 0.015},
}
# GB/T 1591-2018 表5脚注要求 Al、Nb、V、Ti 中至少一种达到对应下限，
# 并非全部同时达到。该组合约束保存到标准上下文，供后续成分结果校验使用。
WIND_TMCP_GRAIN_REFINER_REQUIREMENT = {
    "ALS_min": 0.015,
    "AL_min": 0.020,
    "NB_min": 0.010,
    "V_min": 0.010,
    "TI_min": 0.006,
}
WIND_TMCP_P_S_LIMITS = {
    "B": {"P_max": 0.035, "S_max": 0.035},
    "C": {"P_max": 0.030, "S_max": 0.030},
    "D": {"P_max": 0.030, "S_max": 0.025},
    "E": {"P_max": 0.025, "S_max": 0.020},
    "F": {"P_max": 0.020, "S_max": 0.010},
}
WIND_TMCP_TENSILE = {
    "Q355M": [(16, 355, 470, 630), (40, 345, 470, 630), (63, 335, 450, 610), (80, 325, 440, 600), (100, 325, 440, 600), (120, 320, 430, 590, 22)],
    "Q390M": [(16, 390, 490, 650), (40, 380, 490, 650), (63, 360, 480, 640), (80, 340, 470, 630), (100, 340, 460, 620), (120, 335, 450, 610, 20)],
    "Q420M": [(16, 420, 520, 680), (40, 400, 520, 680), (63, 390, 500, 660), (80, 380, 480, 640), (100, 370, 470, 630), (120, 365, 460, 620, 19)],
    "Q460M": [(16, 460, 540, 720), (40, 440, 540, 720), (63, 430, 530, 710), (80, 410, 510, 690), (100, 400, 500, 680), (120, 385, 490, 660, 17)],
    "Q500M": [(16, 500, 610, 770), (40, 490, 610, 770), (63, 480, 600, 760), (80, 460, 590, 750), (100, 450, 540, 730, 17)],
    "Q550M": [(16, 550, 670, 830), (40, 540, 670, 830), (63, 530, 620, 810), (80, 510, 600, 790), (100, 500, 590, 780, 16)],
    "Q620M": [(16, 620, 710, 880), (40, 610, 710, 880), (63, 600, 690, 880), (80, 580, 670, 860, 15)],
    "Q690M": [(16, 690, 770, 940), (40, 680, 770, 940), (63, 670, 750, 920), (80, 650, 730, 900, 14)],
}
WIND_TMCP_CEV_PCM = {
    "Q355M": [(16, 0.39), (40, 0.39), (63, 0.40), (120, 0.45, 0.20)],
    "Q390M": [(16, 0.41), (40, 0.43), (63, 0.44), (120, 0.46, 0.20)],
    "Q420M": [(16, 0.43), (40, 0.45), (63, 0.46), (120, 0.47, 0.20)],
    "Q460M": [(16, 0.45), (40, 0.46), (63, 0.47), (120, 0.48, 0.22)],
    "Q500M": [(16, 0.47), (40, 0.47), (63, 0.47), (120, 0.48, 0.25)],
    "Q550M": [(16, 0.47), (40, 0.47), (63, 0.47), (120, 0.48, 0.25)],
    "Q620M": [(16, 0.48), (40, 0.48), (63, 0.48), (120, 0.49, 0.25)],
    "Q690M": [(16, 0.49), (40, 0.49), (63, 0.49), (120, 0.49, 0.25)],
}
WIND_TMCP_IMPACT = {
    "B": {"temperature": 20, "longitudinal": 34, "transverse": 27},
    "C": {"temperature": 0, "longitudinal": 34, "transverse": 27},
    "D": {"temperature": -20, "longitudinal": 40, "transverse": 20},
    "E": {"temperature": -40, "longitudinal": 31, "transverse": 20},
    "F": {"temperature": -60, "longitudinal": 27, "transverse": 16},
}


def get_wind_power_standard_context(session_id: str) -> dict:
    """返回本轮风电钢标准上下文，供匹配、智能体和报告统一使用。"""
    return dict(_WIND_POWER_STANDARD_CONTEXT_CACHE.get(str(session_id), {}))


def _wind_power_grade_from_text(text: str) -> tuple[str | None, str | None, str | None]:
    """识别用户显式指定的风电塔筒钢级、质量等级及不支持的交货状态。"""
    # 不删除空格，避免把紧随其后的厚度数字拼接到牌号后缀上（如“Q460ME 60mm”）。
    merged = str(text or "").upper()
    match = re.search(r"(?<![A-Z0-9])(Q(?:355|390|420|460|500|550|620|690))([A-Z]*)(?![A-Z0-9])", merged)
    if not match:
        return None, None, None
    base_grade, suffix = match.groups()
    if suffix.startswith("AR") or suffix.startswith("N"):
        return None, None, f"用户指定 {base_grade}{suffix}，当前风电分支仅支持 TMCP 的 M 级钢。"
    if suffix and not suffix.startswith("M"):
        return None, None, f"用户指定 {base_grade}{suffix}，当前风电分支仅支持 TMCP 的 M 级钢。"
    quality = suffix[1:] if suffix.startswith("M") else ""
    if quality and quality not in {"B", "C", "D", "E", "F"}:
        return None, None, f"用户指定 {base_grade}{suffix} 的质量等级不受当前风电分支支持。"
    grade = f"{base_grade}M"
    if grade not in WIND_TMCP_GRADES:
        return None, None, f"用户指定钢级 {grade} 不在 GB/T 1591-2018 的 TMCP 风电兜底范围内。"
    return grade, quality or None, None


def _choose_wind_power_grade_with_llm(user_message: str, rag_context: str) -> tuple[str | None, str | None, str | None]:
    """用户未指定钢级时，交由 LLM 在 GB/T 1591 的 TMCP 牌号中选择。"""
    prompt = f"""你是风电塔筒用 TMCP 钢板选牌助手。根据用户要求选择唯一的 GB/T 1591-2018 TMCP 钢级与质量等级。

可选钢级仅限：Q355M、Q390M、Q420M、Q460M、Q500M、Q550M、Q620M、Q690M。
质量等级必须与钢级匹配：Q355M可选B/C/D/E/F；Q390M、Q420M可选B/C/D/E；Q460M及以上可选C/D/E。
不得选择N、AR、X系列管线钢或任何其他牌号。优先满足用户明确的屈服强度、厚度、低温韧性及焊接性要求；信息不足时选择保守且可满足要求的最低钢级。
只返回JSON：{{"grade":"QxxxM","quality":"B/C/D/E/F"}}。

用户要求：{user_message}

结构钢知识库参考：{rag_context[:6000] if rag_context else "（无直接检索资料）"}"""
    for attempt in range(1, 3):
        try:
            raw = deepseek_Llm.invoke(prompt)
            parsed = _parse_spec_json(_llm_response_text(raw))
            grade = str(parsed.get("grade") or "").upper().strip()
            quality = str(parsed.get("quality") or "").upper().strip()
            if grade in WIND_TMCP_GRADES and quality in WIND_TMCP_QUALITY_GRADES[grade]:
                return grade, quality, None
            print(f"[风电钢规格提取] LLM选牌第{attempt}次返回无效: {parsed}")
        except Exception as exc:
            print(f"[风电钢规格提取] LLM选牌第{attempt}次失败: {exc}")
    return None, None, "风电塔筒钢级与质量等级未能由模型可靠确定，无法继续按国标兜底。"


def _choose_wind_quality_with_llm(grade: str, user_message: str, rag_context: str) -> tuple[str | None, str | None]:
    """用户明确钢级但未给质量等级时，按用户韧性/服役要求选择唯一质量等级。"""
    allowed = "/".join(WIND_TMCP_QUALITY_GRADES[grade])
    prompt = f"""你是风电塔筒 TMCP 钢板质量等级选择助手。用户已明确钢级 {grade}，请只在 {allowed} 中选择一个质量等级。
选择应优先满足用户明确的冲击温度、韧性、寒冷环境和焊接要求；信息不足时选择该钢级允许范围内最保守的常规质量等级。
只返回 JSON：{{\"quality\":\"B/C/D/E/F\"}}。

用户要求：{user_message}
结构钢知识库参考：{rag_context[:4000] if rag_context else "（无直接检索资料）"}"""
    for attempt in range(1, 3):
        try:
            parsed = _parse_spec_json(_llm_response_text(deepseek_Llm.invoke(prompt)))
            quality = str(parsed.get("quality") or "").upper().strip()
            if quality in WIND_TMCP_QUALITY_GRADES[grade]:
                return quality, None
            print(f"[风电钢规格提取] LLM选质量等级第{attempt}次返回无效: {parsed}")
        except Exception as exc:
            print(f"[风电钢规格提取] LLM选质量等级第{attempt}次失败: {exc}")
    return None, f"用户已指定 {grade}，但质量等级未明确且模型未能可靠选择允许等级 {allowed}。"


def _wind_product_thickness(result: dict, user_message: str) -> float | None:
    text = re.sub(
        r"(?:板坯厚度|连铸坯厚度)\D{0,12}\d+(?:\.\d+)?\s*(?:mm|毫米)",
        "",
        str(user_message or ""),
        flags=re.IGNORECASE,
    )
    patterns = (
        r"(?:成品厚度|钢板厚度|板厚|厚度目标|目标厚度|厚度)\D{0,12}(\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)\D{0,12}(?:风电|塔筒|钢板|成品|目标)?(?:厚度|板厚)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _to_float_or_none(match.group(1))
            if value is not None:
                return value
    lower = _to_float_or_none(result.get("AIM_THICK_min"))
    upper = _to_float_or_none(result.get("AIM_THICK_max"))
    if lower is not None and upper is not None and lower >= 0 and upper < 9999:
        return (lower + upper) / 2.0
    return None


def _wind_table_row_by_thickness(rows: list[tuple], thickness: float | None) -> tuple | None:
    if thickness is None:
        return None
    for row in rows:
        if thickness <= float(row[0]):
            return row
    return None


def _wind_explicit_performance_bounds(user_message: str) -> dict:
    """确定性提取用户明确给出的风电钢性能边界，确保其不被历史或标准最低值覆盖。"""
    text = str(user_message or "")
    aliases = {
        "YS": r"(?:YS|屈服强度|屈服)",
        "TS": r"(?:TS|抗拉强度|抗拉)",
        "EL": r"(?:EL|断后伸长率|伸长率)",
        "AKV": r"(?:AKV|冲击功)",
    }
    bounds: dict[str, float] = {}
    for field, alias in aliases.items():
        minimum_patterns = (
            rf"{alias}\s*(?:≥|>=|不低于|大于等于|不少于|大于|达到|目标(?:为|是)?)\s*(\d+(?:\.\d+)?)",
            rf"(?:≥|>=|不低于|大于等于|不少于|大于)\s*(\d+(?:\.\d+)?)\s*(?:MPA|MPa|兆帕|J|%|％)?\s*(?:的)?\s*{alias}",
        )
        maximum_patterns = (
            rf"{alias}\s*(?:≤|<=|不高于|不大于|小于等于)\s*(\d+(?:\.\d+)?)",
            rf"(?:≤|<=|不高于|不大于|小于等于)\s*(\d+(?:\.\d+)?)\s*(?:MPA|MPa|兆帕|J|%|％)?\s*(?:的)?\s*{alias}",
        )
        for pattern in minimum_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = _to_float_or_none(match.group(1))
                if value is not None:
                    bounds[f"{field}_min"] = value
                break
        for pattern in maximum_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = _to_float_or_none(match.group(1))
                if value is not None:
                    bounds[f"{field}_max"] = value
                break
    return bounds


def _wind_explicit_pcm_max(user_message: str) -> float | None:
    """提取用户明确的 Pcm 上限，供风电标准上下文和后端公式校验共同使用。"""
    text = str(user_message or "")
    patterns = (
        r"(?:Pcm|PCM)\s*(?:<|<=|≤|＜|不高于|不大于|小于等于|小于)\s*(\d+(?:\.\d+)?)",
        r"(?:<|<=|≤|＜|不高于|不大于|小于等于|小于)\s*(\d+(?:\.\d+)?)\s*(?:的)?\s*(?:Pcm|PCM)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _to_float_or_none(match.group(1))
    return None


def _wind_standard_profile(result: dict, user_message: str, rag_context: str) -> dict:
    grade, quality, error = _wind_power_grade_from_text(user_message)
    if error:
        return {"error": error}
    if grade is None:
        grade, quality, error = _choose_wind_power_grade_with_llm(user_message, rag_context)
        if error:
            return {"error": error}
    elif quality is None:
        quality, error = _choose_wind_quality_with_llm(grade, user_message, rag_context)
        if error:
            return {"error": error}

    thickness = _wind_product_thickness(result, user_message)
    tensile_row = _wind_table_row_by_thickness(WIND_TMCP_TENSILE[grade], thickness)
    if tensile_row is None:
        max_thickness = WIND_TMCP_TENSILE[grade][-1][0]
        return {
            "error": (
                f"目标成品厚度 {thickness if thickness is not None else '未识别'} mm 超出 "
                f"GB/T 1591-2018 表10中 {grade} 钢板的 {max_thickness} mm 覆盖范围。"
            )
        }
    ce_rows = WIND_TMCP_CEV_PCM[grade]
    ce_row = _wind_table_row_by_thickness(ce_rows, thickness)
    pcm_max = ce_rows[-1][2] if len(ce_rows[-1]) > 2 else None
    if ce_row is None or pcm_max is None:
        return {"error": f"未找到 {grade} 在目标厚度下可用于钢板的 CEV/Pcm 标准限值。"}

    impact = dict(WIND_TMCP_IMPACT[quality])
    if grade in {"Q500M", "Q550M", "Q620M", "Q690M"}:
        impact = {
            "C": {"temperature": 0, "longitudinal": 55, "transverse": 34},
            "D": {"temperature": -20, "longitudinal": 47, "transverse": 27},
            "E": {"temperature": -40, "longitudinal": 31, "transverse": 20},
        }[quality]

    default_el = {"Q355M": 22, "Q390M": 20, "Q420M": 19, "Q460M": 17, "Q500M": 17, "Q550M": 16, "Q620M": 15, "Q690M": 14}[grade]
    chemistry = {**WIND_TMCP_CHEMISTRY[grade], **WIND_TMCP_P_S_LIMITS[quality]}
    for key in WIND_TMCP_GRAIN_REFINER_REQUIREMENT:
        chemistry.pop(key, None)
    explicit_pcm_max = _wind_explicit_pcm_max(user_message)
    return {
        "purpose": WIND_POWER_STEEL_PURPOSE,
        "grade": grade,
        "quality": quality,
        "delivery_state": "TMCP(M)",
        "product": "海上风电塔筒用钢板" if re.search(r"海上|海洋|近海|海工", str(user_message or "")) else (
            "陆上风电塔筒用钢板" if re.search(r"陆上|内陆", str(user_message or "")) else "风电塔筒用钢板"
        ),
        "thickness_mm": thickness,
        "tensile": {"YS_min": tensile_row[1], "TS_min": tensile_row[2], "TS_max": tensile_row[3], "EL_min": tensile_row[4] if len(tensile_row) > 4 else default_el},
        "chemistry": chemistry,
        "grain_refiner_requirement": dict(WIND_TMCP_GRAIN_REFINER_REQUIREMENT),
        "CEV_max": ce_row[1],
        # Pcm_max 始终保存 GB/T 1591 对应厚度的标准上限。用户额外提出的
        # Pcm 目标单独保存，后置微调会在国标校验之后再做一次用户约束校验，
        # 避免用户值覆盖标准字段后无法区分两类校验来源。
        "Pcm_max": pcm_max,
        "Pcm_standard_max": pcm_max,
        "Pcm_user_max": explicit_pcm_max,
        "impact": impact,
    }


def _apply_wind_power_gbt1591_limits(result: dict, user_message: str, rag_context: str) -> tuple[dict, dict]:
    """将 GB/T 1591-2018 的 TMCP 钢板标准应用到风电规格结果。"""
    profile = _wind_standard_profile(result, user_message, rag_context)
    if profile.get("error"):
        return dict(result), profile

    fixed = dict(result)
    protected_keys = _user_explicit_spec_fields(user_message)
    target_ranges = _user_target_spec_ranges(user_message)
    if target_ranges:
        fixed.update(target_ranges)
        protected_keys.update(target_ranges)
    explicit_performance = _wind_explicit_performance_bounds(user_message)
    if explicit_performance:
        fixed.update(explicit_performance)
        protected_keys.update(explicit_performance)

    chemistry_limits = dict(profile["chemistry"])
    for key in WIND_TMCP_GRAIN_REFINER_REQUIREMENT:
        chemistry_limits.pop(key, None)
    for key, value in {**chemistry_limits, **profile["tensile"]}.items():
        if key in protected_keys:
            # 与管线钢标准提取保持一致：用户明确提出的成分、工艺或性能
            # 范围具有更高优先级，国标数据作为未指定字段的补全依据，不能
            # 覆盖用户目标，更不能因用户下限低于自动选择牌号的下限而中断。
            continue
        fixed[key] = value

    impact_min = float(profile["impact"]["longitudinal"])
    if "AKV_min" in protected_keys:
        # 与管线钢分支相同，已明确指定的冲击目标不由标准兜底覆盖。
        pass
    else:
        fixed["AKV_min"] = impact_min
    return fixed, profile


def build_spec_search_query(user_message: str, purpose: str) -> str:
    if _is_pipeline_purpose(purpose):
        return (
            f"{purpose} {user_message} "
            "管线钢 X系列 L系列 API管线钢 油气输送管线 TMCP 控轧控冷 "
            "牌号 同一牌号 目标牌号 板坯厚度 SLAB_THICK 成品厚度 厚度窗口 AIM_THICK "
            "C SI MN P S N NB V TI AL ALS CU CR NI CO MO B 化学成分 成分范围 合金含量 "
            "FURNACE_EXIT_TEMP 出炉温度 FET 精轧开轧温度 FDT 精轧终轧温度 "
            "TEMP_ENTR 入水温度 FEH 中间坯厚度 SELF_TEMP 返红温度 "
            "YS 屈服强度 TS 抗拉强度 EL 断后伸长率 AKV 冲击功 冲击韧性 "
            "组织演变 晶粒 析出 相变 CCT PTT 成分对组织的影响 组织对性能的影响"
        )
    if _is_wind_power_purpose(purpose):
        return (
            f"{purpose} 风电 塔筒 钢板 海上或陆上服役场景 TMCP 热机械轧制 M级 GB/T 1591-2018 {user_message} "
            "Q355M Q390M Q420M Q460M Q500M Q550M Q620M Q690M 质量等级 B C D E F "
            "板坯厚度 SLAB_THICK 成品厚度 AIM_THICK C Si Mn P S N Nb V Ti Alt Als Cu Cr Ni Co Mo B "
            "CEV 碳当量 Pcm 焊接裂纹敏感性 屈服强度 YS 抗拉强度 TS 延伸率 EL 冲击功 AKV "
            "低温韧性 焊接性 疲劳风险 加热制度 FET FDT 入水温度 返红温度 TMCP 控轧控冷 "
            "晶粒细化 析出强化 相变组织 风电塔筒"
        )
    return (
        f"{purpose} 工程机械用钢 {user_message} "
        "牌号 同一牌号 目标牌号 默认选牌 强度等级 厚度 产品厚度 成品厚度 规格 尺寸 厚度范围 "
        "C SI MN P S N NB V TI AL ALS CU CR NI CO MO B 化学成分 成分范围 合金含量 "
        "均热温度 精轧开轧温度 FET 终轧温度 FDT 卷取温度 CT 淬火温度 回火温度 "
        "力学性能 拉伸性能 横向拉伸性能 屈服强度 抗拉强度 延伸率 断后伸长率"
    )


def retrieve_steel_spec_docs(
    user_message: str,
    purpose: str = "耐磨钢",
    db_name: str = "Nb_KnowBase_db",
    top_k: int = 18,
) -> list[dict]:
    from hybrid_retriever import hybrid_search

    search_query = build_spec_search_query(user_message, purpose)
    return hybrid_search(search_query, k=top_k, db_name=db_name)


def format_spec_rag_context(docs: list[dict]) -> str:
    if not docs:
        return ""
    return "\n\n---\n\n".join([
        f"[来源: {d.get('source', 'unknown')}]\n{d.get('content', '')}"
        for d in docs
    ])


def extract_steel_specs(
    user_message: str,
    session_id: str,
    purpose: str = "耐磨钢",
    db_name: str = "Nb_KnowBase_db",
    top_k: int = 18,
    retrieval_docs: list[dict] | None = None,
    return_range_stages: bool = False,
) -> dict | tuple[dict, dict, dict]:
    """
    从知识库提取钢材成分/性能规格，返回结构化 JSON。

    Args:
        user_message: 用户输入
        session_id:   会话ID
        purpose:      钢材用途标签 ("耐磨钢" 或 "高强度结构钢")
        db_name:      检索数据库名
        top_k:        RAG 返回文档数

    Returns:
        默认返回包含所有成分/性能字段的最终 JSON；return_range_stages=True 时
        返回 (最终范围JSON, RAG提取范围JSON, 兜底后最终范围JSON)。
    """
    if _is_wind_power_purpose(purpose):
        _WIND_POWER_STANDARD_CONTEXT_CACHE.pop(str(session_id), None)

    # Step 1: RAG 检索
    rag_context = ""
    try:
        docs = retrieval_docs if retrieval_docs is not None else retrieve_steel_spec_docs(
            user_message=user_message,
            purpose=purpose,
            db_name=db_name,
            top_k=top_k,
        )
        if docs:
            rag_context = format_spec_rag_context(docs)
            print(f"[钢规格提取] RAG命中 {len(docs)} 条文档")
    except Exception as e:
        print(f"[钢规格提取] RAG失败: {e}")
        rag_context = ""

    # Step 2: 构建 prompt
    full_prompt = f"""{_system_prompt_for_purpose(purpose)}

## 当前钢材用途
{purpose}

## 知识库检索结果
{rag_context if rag_context else "（无检索结果，请根据专业知识推断）"}

## 用户要求
{user_message}"""

    # Step 3: 链式执行
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt_template | deepseek_Llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        _get_spec_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 注意：裁剪逻辑现在由 SessionStore 内部处理（通过 PersistentChatMessageHistory）
    hist = _get_spec_history(session_id)

    try:
        raw = chain_with_history.invoke(
            {"system_prompt": full_prompt, "input": user_message},
            config={"configurable": {"session_id": session_id}},
        )
        result = _parse_spec_json(raw)

        # 确保字段完整、顺序稳定，且 "用途" 字段与 purpose 一致
        result = _normalize_spec_result(result, purpose)
        # 保存应用国标硬编码、未知牌号 LLM 兜底和确定性厚度覆盖之前的规格范围，
        # 供前端对比“RAG 提取范围”和“兜底后最终范围”。
        rag_range_result = dict(result)
        if _is_pipeline_purpose(purpose):
            result = _apply_pipeline_gbt9711_limits(result, user_message, rag_context)
            # 用户明确给出的板坯厚度最终由确定性提取覆盖模型结果；未命中时
            # Schema 中的 SLAB_THICK_min/max 保持默认 0/9999。
            slab_thickness_bounds = _extract_pipeline_slab_thickness_bounds(user_message)
            if slab_thickness_bounds:
                result.update(slab_thickness_bounds)
            else:
                result["SLAB_THICK_min"] = 0.0
                result["SLAB_THICK_max"] = 9999.0
        elif _is_wind_power_purpose(purpose):
            result, wind_standard_context = _apply_wind_power_gbt1591_limits(
                result,
                user_message,
                rag_context,
            )
            slab_thickness_bounds = _extract_pipeline_slab_thickness_bounds(user_message)
            if slab_thickness_bounds:
                result.update(slab_thickness_bounds)
            else:
                result["SLAB_THICK_min"] = 0.0
                result["SLAB_THICK_max"] = 9999.0
            _WIND_POWER_STANDARD_CONTEXT_CACHE[str(session_id)] = wind_standard_context

        # 存储历史（持久化）
        hist.add_message(HumanMessage(content=user_message))
        hist.add_message(AIMessage(content=json.dumps(result, ensure_ascii=False)))

        print(f"[钢规格提取] 完成，用途={purpose}")
        if return_range_stages:
            final_range_result = dict(result)
            return final_range_result, rag_range_result, final_range_result
        return result

    except Exception as e:
        print(f"[钢规格提取] 失败: {e}")
        fallback = dict(_schema_for_purpose(purpose))
        fallback["用途"] = purpose
        if _is_wind_power_purpose(purpose):
            _WIND_POWER_STANDARD_CONTEXT_CACHE[str(session_id)] = {
                "error": f"风电钢规格提取失败: {type(e).__name__}: {e}"
            }
        if return_range_stages:
            fallback_result = dict(fallback)
            return fallback_result, fallback_result, fallback_result
        return fallback



