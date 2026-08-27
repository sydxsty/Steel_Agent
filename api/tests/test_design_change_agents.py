from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from pipeline_agents import (  # noqa: E402
    DesignChangeAssessmentDependencies,
    DesignChangeAssessmentError,
    RequirementParsingDependencies,
    RequirementParsingError,
    _invoke_refinement_langchain_agent,
    _restore_assessment_inherited_fields,
    assess_design_change,
    build_unified_design_user_message,
    parse_design_requirement,
)


def _assessment_payload(composition_action: str = "REASSESS") -> dict:
    module = lambda action, priority: {  # noqa: E731
        "action": action,
        "priority": priority,
        "reasons": ["根据当前目标、参考方案和检索证据重新判断"],
    }
    return {
        "task_type": "NEW_DESIGN_WITH_REFERENCE",
        "reference": {"steel_grade": "Q550M", "thickness_mm": 25},
        "target": {"steel_grade": "Q550M", "thickness_mm": 50},
        "change_assessment": {
            "composition": module(composition_action, "HIGH"),
            "heating": module("REASSESS", "MEDIUM"),
            "rolling": module("REASSESS", "HIGH"),
            "cooling": module("REASSESS", "HIGH"),
            "performance_requirement": module("INHERIT", "HIGH"),
        },
        "evidence": {
            "rag": [{"status": "ok", "topic": "厚规格TMCP"}],
            "historical_data": [{"status": "ok", "target_thickness_mm": 50}],
        },
    }


class _FakeAssessmentAgent:
    def __init__(self, tools, payload, calls):
        self.tools = tools
        self.payload = payload
        self.calls = calls

    def invoke(self, inputs, config=None):
        self.calls.append((inputs, config))
        self.tools[0].invoke({"query": "Q550M 50mm TMCP厚度效应"})
        self.tools[1].invoke({})
        return {"structured_response": self.payload}


class DesignChangeAgentTests(unittest.TestCase):
    def test_requirement_agent_returns_validated_json_and_unified_message(self):
        payload = {
            "application": "offshore_wind",
            "steel_grade": "Q500M",
            "thickness_mm": 25,
            "performance": {
                "YS": 550,
                "AKV": 120,
                "impact_temperature": "-20℃",
            },
            "requirements": ["composition_design", "process_design"],
            "composition_constraints": [],
            "process_constraints": ["采用TMCP工艺"],
            "explicit_constraints": ["屈服强度不低于550MPa", "-20℃冲击功不低于120J"],
            "other_constraints": [],
            "references_previous_design": False,
        }
        calls = []

        class Agent:
            def invoke(self, inputs, config=None):
                calls.append((inputs, config))
                return {"structured_response": payload}

        def factory(**kwargs):
            self.assertEqual(kwargs["name"], "steel_design_requirement_parser")
            self.assertEqual(kwargs["tools"], [])
            return Agent()

        result = parse_design_requirement(
            user_message=(
                "请设计25mm厚海上风电Q500M钢，要求屈服强度550MPa，"
                "-20℃冲击功120J，设计成分和TMCP工艺"
            ),
            purpose="风电用钢",
            session_context="",
            dependencies=RequirementParsingDependencies(
                agent_model=object(),
                create_agent_fn=factory,
            ),
        )
        self.assertEqual(result["application"], "offshore_wind")
        self.assertEqual(result["steel_grade"], "Q500M")
        self.assertEqual(result["thickness_mm"], 25)
        self.assertEqual(result["performance"]["YS"], 550)
        self.assertNotIn("slab_thickness_mm", result)
        self.assertEqual(calls[0][1]["recursion_limit"], 6)

        unified = build_unified_design_user_message("原始用户提示词", result)
        self.assertTrue(unified.startswith("USER_MESSAGE:"))
        self.assertIn("用户需求:\n原始用户提示词", unified)
        self.assertIn("结构化需求:", unified)
        self.assertIn('"steel_grade": "Q500M"', unified)

    def test_requirement_agent_retries_and_rejects_cross_product_application(self):
        attempts = []

        class Agent:
            def invoke(self, inputs, config=None):
                attempts.append(True)
                return {
                    "structured_response": {
                        "application": "pipeline",
                        "steel_grade": "Q500M",
                        "performance": {},
                    }
                }

        with self.assertRaises(RequirementParsingError):
            parse_design_requirement(
                user_message="设计风电用钢",
                purpose="风电用钢",
                session_context="",
                dependencies=RequirementParsingDependencies(
                    agent_model=object(),
                    create_agent_fn=lambda **kwargs: Agent(),
                ),
            )
        self.assertEqual(len(attempts), 2)

    def test_requirement_parser_is_called_before_product_design_branch(self):
        source = (API_DIR / "api.py").read_text(encoding="utf-8")
        parser_call = source.index("requirement_json = await asyncio.to_thread(")
        product_branch = source.index(
            'if purpose in {"管线钢", "风电用钢"}:',
            parser_call,
        )
        self.assertLess(parser_call, product_branch)
        self.assertIn(
            "user_message = build_unified_design_user_message(",
            source[parser_call:product_branch],
        )

    def test_assessment_agent_calls_product_rag_and_current_target_history(self):
        rag_queries = []
        history_calls = []
        agent_calls = []

        def factory(**kwargs):
            self.assertEqual(kwargs["name"], "steel_design_change_assessment")
            self.assertEqual(len(kwargs["tools"]), 2)
            return _FakeAssessmentAgent(
                kwargs["tools"],
                _assessment_payload(),
                agent_calls,
            )

        result = assess_design_change(
            material_name="风电塔筒用TMCP钢板",
            user_message="基于以上25mm方案设计50mm，性能要求不变",
            session_context="上一轮为25mm方案",
            spec_result={"AIM_THICK_min": 50, "AIM_THICK_max": 50},
            reference_summary={"steel_grade": "Q550M", "AIM_THICK": 25},
            target_summary={"steel_grade": "Q550M", "thickness_mm": 50},
            engineering_standard_context={"Pcm_max": 0.23},
            matched_result_summary={"AIM_THICK": 25},
            dependencies=DesignChangeAssessmentDependencies(
                agent_model=object(),
                retrieve_product_knowledge=lambda query: rag_queries.append(query) or "RAG",
                retrieve_current_target_history=lambda: history_calls.append(True) or {
                    "target_thickness_mm": 50,
                    "samples": [],
                },
                create_agent_fn=factory,
            ),
        )

        self.assertEqual(result["target"]["thickness_mm"], 50)
        self.assertEqual(result["change_assessment"]["composition"]["action"], "REASSESS")
        self.assertEqual(len(rag_queries), 1)
        self.assertEqual(history_calls, [True])
        self.assertEqual(agent_calls[0][1]["recursion_limit"], 12)

    def test_assessment_without_required_tool_calls_retries_then_stops(self):
        attempts = []

        class Agent:
            def invoke(self, inputs, config=None):
                attempts.append(True)
                return {"structured_response": _assessment_payload()}

        dependencies = DesignChangeAssessmentDependencies(
            agent_model=object(),
            retrieve_product_knowledge=lambda query: "RAG",
            retrieve_current_target_history=lambda: {},
            create_agent_fn=lambda **kwargs: Agent(),
        )
        with self.assertRaises(DesignChangeAssessmentError):
            assess_design_change(
                material_name="管线钢",
                user_message="设计X70管线钢",
                session_context="",
                spec_result={},
                reference_summary={},
                target_summary={},
                engineering_standard_context={},
                matched_result_summary={},
                dependencies=dependencies,
            )
        self.assertEqual(len(attempts), 2)

    def test_composition_inherit_restores_reference_without_changing_structure(self):
        reference = {
            "isState": False,
            "arrBody": [{"C": "0.0800"}, {"Mn": "1.5000"}, {"FDT": "820"}],
        }
        candidate = {
            "isState": False,
            "arrBody": [{"C": "0.1000"}, {"Mn": "1.7000"}, {"FDT": "790"}],
        }
        result = _restore_assessment_inherited_fields(
            candidate,
            reference,
            _assessment_payload("INHERIT"),
            {"C", "MN"},
        )
        self.assertEqual(result["arrBody"], [
            {"C": "0.0800"}, {"Mn": "1.5000"}, {"FDT": "790"},
        ])
        self.assertEqual(list(result), list(candidate))

    def test_rolling_and_cooling_inherit_restore_only_their_fields(self):
        assessment = _assessment_payload("REASSESS")
        assessment["change_assessment"]["rolling"]["action"] = "INHERIT"
        assessment["change_assessment"]["cooling"]["action"] = "INHERIT"
        reference = {
            "arrBody": [
                {"C": "0.0800"}, {"FDT": "820"}, {"N1_DH_CAL": "100"},
                {"TIME_ENTR": "20260824100100"}, {"TEMP_ENTR": "780"},
                {"SELF_TEMP": "480"},
            ],
        }
        candidate = {
            "arrBody": [
                {"C": "0.0900"}, {"FDT": "790"}, {"N1_DH_CAL": "90"},
                {"TIME_ENTR": "20260824100200"}, {"TEMP_ENTR": "760"},
                {"SELF_TEMP": "460"},
            ],
        }
        result = _restore_assessment_inherited_fields(
            candidate,
            reference,
            assessment,
            {"C"},
            {"FDT", "N1_DH_CAL"},
        )
        self.assertEqual(result["arrBody"], [
            {"C": "0.0900"}, {"FDT": "820"}, {"N1_DH_CAL": "100"},
            {"TIME_ENTR": "20260824100100"}, {"TEMP_ENTR": "780"},
            {"SELF_TEMP": "480"},
        ])

    def test_refinement_agent_exposes_three_tools_and_returns_exact_json(self):
        candidate = {
            "isState": False,
            "strCoil": "coil",
            "arrBody": [{"FDT": "820"}, {"TEMP_ENTR": "780"}, {"SELF_TEMP": "480"}],
        }
        tool_names = []
        validator_calls = []

        class Agent:
            def __init__(self, tools):
                self.tools = tools

            def invoke(self, inputs, config=None):
                tool_names.extend(tool.name for tool in self.tools)
                self.tools[0].invoke({"query": "当前产品"})
                self.tools[1].invoke({})
                self.tools[2].invoke({"candidate_json": json.dumps(candidate)})
                return {"messages": [SimpleNamespace(content=json.dumps(candidate))]}

        dependencies = SimpleNamespace(
            agent_model=object(),
            retrieve_agent_knowledge=lambda query: "RAG",
            retrieve_agent_history=lambda: {"target_thickness_mm": 50},
            validate_agent_candidate=lambda original, value, spec, is_wind: (
                validator_calls.append((value, is_wind)) or []
            ),
            create_agent_fn=lambda **kwargs: Agent(kwargs["tools"]),
        )
        text, metadata = _invoke_refinement_langchain_agent(
            dependencies=dependencies,
            system_prompt="SYSTEM",
            original=candidate,
            spec_result={},
            is_wind=True,
        )
        self.assertEqual(json.loads(text), candidate)
        self.assertEqual(metadata["agent"], "langchain")
        self.assertEqual(tool_names, [
            "search_product_knowledge",
            "search_current_target_history",
            "validate_candidate_matched_result",
        ])
        self.assertTrue(validator_calls[0][1])

    def test_wind_history_payload_keeps_design_fields_with_trend_only_policy(self):
        import api as api_module

        payload = api_module._build_design_agent_history_payload(
            [{
                "SLAB_ID": "secret",
                "C": 0.08,
                "YS": 585,
                "AIM_THICK": 50,
                "FDT": 810,
                "N1_DH_CAL": 300,
            }],
            spec_result={"AIM_THICK_min": 50, "AIM_THICK_max": 50},
            user_message="设计50mm风电钢",
            is_wind=True,
        )
        sample = payload["samples"][0]
        self.assertNotIn("SLAB_ID", sample)
        self.assertEqual(sample["C"], 0.08)
        self.assertEqual(sample["YS"], 585)
        self.assertEqual(sample["AIM_THICK"], 50)
        self.assertEqual(payload["usage_policy"], "CURRENT_TARGET_ENGINEERING_REFERENCE")

        from prompt import (
            DESIGN_CHANGE_ASSESSMENT_SYSTEM_PROMPT,
            WIND_POWER_REFINEMENT_PROCESS_RULE,
        )
        for system_prompt in (
            DESIGN_CHANGE_ASSESSMENT_SYSTEM_PROMPT,
            WIND_POWER_REFINEMENT_PROCESS_RULE,
        ):
            self.assertIn("管线钢实绩", system_prompt)
            self.assertIn("严禁直接复制", system_prompt)
            self.assertIn("高度模仿", system_prompt)

    def test_pipeline_history_payload_keeps_design_fields_but_removes_identity(self):
        import api as api_module

        payload = api_module._build_design_agent_history_payload(
            [{"SLAB_ID": "secret", "C": 0.08, "YS": 585, "AIM_THICK": 50}],
            spec_result={"AIM_THICK_min": 50, "AIM_THICK_max": 50},
            user_message="设计50mm X70管线钢",
            is_wind=False,
        )
        sample = payload["samples"][0]
        self.assertNotIn("SLAB_ID", sample)
        self.assertEqual(sample["C"], 0.08)
        self.assertEqual(sample["YS"], 585)

    def test_strict_cooling_gate_reports_temperature_and_time_fields(self):
        import api as api_module

        invalid = {
            "arrBody": [
                {"R_PASS_ACT": "1"}, {"F_PASS_ACT": "1"},
                {"N2_ENTR_DATE": "2026-08-24 10:00:10.000"},
                {"TIME_ENTR": "20260824100010"},
                {"FDT": "800"}, {"TEMP_ENTR": "800"}, {"SELF_TEMP": "810"},
            ]
        }
        errors = api_module._collect_pipeline_strict_cooling_gate_errors(invalid)
        self.assertEqual(
            {error["field"] for error in errors},
            {"TEMP_ENTR", "SELF_TEMP", "TIME_ENTR"},
        )
        self.assertTrue(all(error["status"] == "FAIL" for error in errors))


if __name__ == "__main__":
    unittest.main()
