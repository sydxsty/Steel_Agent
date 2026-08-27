from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from pipeline_agents import (  # noqa: E402
    CompositionRefinementDependencies,
    ProcessAgentDependencies,
    _extract_boundary_repair_fields,
    _pin_boundary_repair_baseline,
    _remove_refinement_turn_width_errors,
    refine_composition_process_performance,
    refine_cooling_process,
    refine_reheat_process,
    refine_rolling_process,
)


def _set_single_body_value(result: dict, key: str, value) -> bool:
    for item in result.get("arrBody") or []:
        if isinstance(item, dict) and key in item:
            item[key] = value
            return True
    return False


def _body_to_row(result: dict) -> dict:
    row = {}
    for item in result.get("arrBody") or []:
        if isinstance(item, dict) and len(item) == 1:
            key, value = next(iter(item.items()))
            row[key] = value
    return row


class PipelineAgentMigrationTests(unittest.TestCase):
    """以隔离依赖验证迁移后的业务顺序，不调用真实 RAG、模型或 DLL。"""

    def test_composition_refinement_ignores_only_turn_width_errors(self):
        errors = [
            "转钢道次标识无效：当前为0/5",
            "转钢标记与宽度变化不一致：应为2/6",
            "WIDTH_ROLL_START_REMARK='x' 不是整数",
            "最终有效道次厚度与AIM_THICK不一致",
        ]
        self.assertEqual(
            _remove_refinement_turn_width_errors(errors),
            ["最终有效道次厚度与AIM_THICK不一致"],
        )

    def test_component_boundary_error_returns_to_composition_repair_scope(self):
        """MO 越界必须解除上一轮仅轧制修复的限制。"""
        fields = _extract_boundary_repair_fields(
            [
                "LLM微调字段 MO='0.0856' 超出规格边界或不是合法数值；"
                "本轮标准允许范围：MO >= 0.1，MO <= 0.3 wt%。"
            ],
            {"C", "MO", "NB", "YS", "TS", "EL", "AKV"},
        )
        self.assertEqual(fields, {"MO"})

    def test_component_boundary_repair_pins_next_baseline_to_limit(self):
        """低于 Mo 下限的候选不能继续作为下一轮模型的默认值。"""
        result = {
            "arrBody": [{"Mo": "0.0383"}, {"YS": "560"}, {"N1_DH_CAL": "200"}],
        }
        repaired = _pin_boundary_repair_baseline(
            result,
            [
                "LLM微调字段 MO='0.0383' 超出规格边界或不是合法数值；"
                "本轮标准允许范围：MO >= 0.1，MO <= 0.3 wt%。"
            ],
            {"MO"},
            {"YS"},
        )
        self.assertEqual(_body_to_row(repaired)["Mo"], "0.1000")

    def test_boundary_retry_keeps_current_design_not_original_history_schedule(self):
        """Mo 越界且规程未通过时，下一轮必须使用当前设计基线全规程重写。"""
        prompts = []
        original = {
            "isState": False,
            "strCoil": "coil",
            "arrBody": [
                {"Mo": "0.2000"},
                {"YS": "560"},
                {"AIM_THICK": "25"},
                {"N1_DH_CAL": "99"},
                {"N2_DH_CAL": "80"},
            ],
        }
        first_candidate = copy.deepcopy(original)
        _set_single_body_value(first_candidate, "Mo", "0.0017")
        _set_single_body_value(first_candidate, "N1_DH_CAL", "27")
        _set_single_body_value(first_candidate, "N2_DH_CAL", "25")
        second_candidate = copy.deepcopy(first_candidate)
        _set_single_body_value(second_candidate, "Mo", "0.1000")
        _set_single_body_value(second_candidate, "N1_DH_CAL", "28")
        _set_single_body_value(second_candidate, "N2_DH_CAL", "25")
        candidates = [first_candidate, second_candidate]

        def invoke(messages, **kwargs):
            prompts.append(messages[0].content)
            return SimpleNamespace(
                content=json.dumps(candidates.pop(0), ensure_ascii=False),
                raw_metadata={"model": "test", "finish_reason": "stop"},
            )

        def sanitize(source, refined, spec, *, validation_errors, **kwargs):
            if float(_body_to_row(refined)["Mo"]) < 0.1:
                validation_errors.append(
                    "LLM微调字段 MO='0.0017' 超出规格边界或不是合法数值；"
                    "本轮标准允许范围：MO >= 0.1，MO <= 0.3 wt%。"
                )
                return None
            return copy.deepcopy(refined)

        def clear_roll(result):
            cleared = copy.deepcopy(result)
            _set_single_body_value(cleared, "N1_DH_CAL", "0")
            _set_single_body_value(cleared, "N2_DH_CAL", "0")
            return cleared

        dependencies = CompositionRefinementDependencies(
            extract_target_thickness=lambda message: 25.0,
            extract_target_slab_thickness=lambda message: None,
            lock_explicit_thickness_targets=lambda result, target, slab: copy.deepcopy(result),
            is_context_modification_request=lambda message: False,
            build_refinement_rag_context=lambda spec, message: "",
            build_cross_route_context=lambda sid: "",
            get_recent_session_context=lambda sid: "",
            filter_wind_session_context=lambda context: context,
            component_fields=frozenset({"MO"}),
            performance_fields=frozenset({"YS"}),
            roll_fields=frozenset({"N1_DH_CAL", "N2_DH_CAL"}),
            get_arrbody_key=lambda item: next(iter(item), None),
            build_historical_roll_reference=lambda result: "HISTORY",
            build_wind_standard_redesign_instruction=lambda context, error="": "",
            reasoning_cache={},
            invoke_qwen=invoke,
            parse_json_object=json.loads,
            extract_qwen_agent_response=lambda parsed: (parsed, {}),
            restore_arrbody_fields=lambda candidate, source, fields: candidate,
            sanitize_refined_result=sanitize,
            validate_wind_result=lambda result, context: "",
            normalize_declared_pass_tail=lambda result: result,
            collect_deformation_pass_errors=lambda result, **kwargs: [],
            roll_errors_require_global_redesign=lambda errors: True,
            prepare_full_roll_redesign_baseline=clear_roll,
            normalize_deformation_passes=lambda result, label, **kwargs: (result, ""),
            validate_dll_time_encodings=lambda result, include_cooling_start: "",
            enforce_performance_standard=lambda result, spec: result,
            cache_performance_baseline=lambda result, spec: None,
            performance_values=lambda result: {"YS": "560"},
            max_completion_tokens=32768,
        )

        result = refine_composition_process_performance(
            {"MO_min": 0.1, "MO_max": 0.3},
            original,
            "设计25mm风电钢",
            "sid",
            dependencies=dependencies,
        )

        self.assertEqual(_body_to_row(result)["Mo"], "0.1000")
        self.assertEqual(len(prompts), 2)
        self.assertIn('"Mo": "0.1000"', prompts[1])
        self.assertIn('"N1_DH_CAL": "0"', prompts[1])
        self.assertNotIn('"N1_DH_CAL": "99"', prompts[1])

    def test_target_thickness_recognizes_mm_thick_shorthand(self):
        """“25mm厚的钢板”必须锁定为 25mm，而不是采用历史匹配厚度。"""
        import api as api_module

        self.assertEqual(
            api_module._extract_pipeline_target_thickness_from_text(
                "请设计25mm厚的海上风电塔筒用钢热轧成分和工艺"
            ),
            25.0,
        )
        self.assertEqual(
            api_module._extract_pipeline_target_thickness_from_text("设计厚25mm的管线钢"),
            25.0,
        )
        locked = api_module._lock_explicit_pipeline_thickness_targets(
            {"arrBody": [{"AIM_THICK": "27.5"}, {"SLAB_THICK": "300"}]},
            25.0,
            None,
        )
        self.assertEqual(_body_to_row(locked)["AIM_THICK"], "25")

    def test_composition_refinement_preserves_call_contract(self):
        trace = []
        reasoning_cache = {
            "sid:pipeline_refine": ["old"],
            "_visible:后置成分/工艺微调": ["old"],
        }
        original = {
            "isState": False,
            "strCoil": "coil",
            "arrBody": [{"C": "0.05"}, {"YS": "500"}, {"AIM_THICK": "22"}],
        }
        candidate = copy.deepcopy(original)
        candidate["isState"] = True
        _set_single_body_value(candidate, "C", "0.045")
        _set_single_body_value(candidate, "YS", "560")

        def invoke(messages, **kwargs):
            trace.append(("invoke", messages, kwargs))
            return SimpleNamespace(
                content=json.dumps(candidate, ensure_ascii=False),
                raw_metadata={"model": "test", "finish_reason": "stop"},
            )

        dependencies = CompositionRefinementDependencies(
            extract_target_thickness=lambda message: trace.append(("target", message)) or 22.0,
            extract_target_slab_thickness=lambda message: trace.append(("slab", message)) or None,
            lock_explicit_thickness_targets=lambda result, target, slab: trace.append(
                ("lock", target, slab)
            ) or copy.deepcopy(result),
            is_context_modification_request=lambda message: trace.append(
                ("override", message)
            ) or False,
            build_refinement_rag_context=lambda spec, message: trace.append(
                ("rag", spec, message)
            ) or "RAG",
            build_cross_route_context=lambda sid: trace.append(("cross", sid)) or "CONTEXT",
            get_recent_session_context=lambda sid: "UNUSED",
            filter_wind_session_context=lambda context: context,
            component_fields=frozenset({"C"}),
            performance_fields=frozenset({"YS"}),
            roll_fields=frozenset({"AIM_THICK"}),
            get_arrbody_key=lambda item: next(iter(item), None),
            build_historical_roll_reference=lambda result: trace.append(("history",)) or "HISTORY",
            build_wind_standard_redesign_instruction=lambda context, error="": "",
            reasoning_cache=reasoning_cache,
            invoke_qwen=invoke,
            parse_json_object=json.loads,
            extract_qwen_agent_response=lambda parsed: (parsed, {}),
            restore_arrbody_fields=lambda candidate_result, source, fields: candidate_result,
            sanitize_refined_result=lambda source, refined, spec, **kwargs: trace.append(
                ("sanitize", kwargs)
            ) or copy.deepcopy(refined),
            validate_wind_result=lambda result, context: "",
            normalize_declared_pass_tail=lambda result: trace.append(("tail",)) or result,
            collect_deformation_pass_errors=lambda result, **kwargs: trace.append(
                ("roll_errors", kwargs)
            ) or [],
            roll_errors_require_global_redesign=lambda errors: False,
            prepare_full_roll_redesign_baseline=lambda result: result,
            normalize_deformation_passes=lambda result, label, **kwargs: trace.append(
                ("normalize", label, kwargs)
            ) or (result, ""),
            validate_dll_time_encodings=lambda result, include_cooling_start: trace.append(
                ("time_encoding", include_cooling_start)
            ) or "",
            enforce_performance_standard=lambda result, spec: trace.append(("performance",)) or result,
            cache_performance_baseline=lambda result, spec: trace.append(("cache",)),
            performance_values=lambda result: trace.append(("values",)) or {"YS": "560"},
            max_completion_tokens=32768,
        )

        result = refine_composition_process_performance(
            {"YS_min": 550},
            original,
            "设计22mm管线钢",
            "sid",
            dependencies=dependencies,
        )

        self.assertEqual(result, candidate)
        self.assertNotIn("sid:pipeline_refine", reasoning_cache)
        self.assertNotIn("_visible:后置成分/工艺微调", reasoning_cache)
        self.assertEqual(sum(item[0] == "rag" for item in trace), 1)
        invoke_event = next(item for item in trace if item[0] == "invoke")
        self.assertEqual(invoke_event[2]["response_format"], {"type": "json_object"})
        self.assertEqual(invoke_event[2]["max_completion_tokens"], 32768)
        self.assertEqual(invoke_event[2]["extra_body"], {"enable_thinking": False})
        ordered_names = [item[0] for item in trace]
        self.assertLess(ordered_names.index("rag"), ordered_names.index("invoke"))
        self.assertLess(ordered_names.index("invoke"), ordered_names.index("sanitize"))
        self.assertLess(ordered_names.index("sanitize"), ordered_names.index("roll_errors"))
        self.assertLess(ordered_names.index("roll_errors"), ordered_names.index("time_encoding"))
        self.assertLess(ordered_names.index("time_encoding"), ordered_names.index("performance"))
        self.assertIn(("time_encoding", True), trace)
        self.assertLess(ordered_names.index("roll_errors"), ordered_names.index("performance"))

    def test_reheat_stage_order_and_final_simulation(self):
        trace = []
        progress = []
        input_cache = {}
        reasoning_cache = {"sid:reheat": ["加热判断摘要"]}
        visible_cache = {"sid:reheat": [{"conclusion": "通过"}]}
        original = {"isState": False, "arrBody": [{"SOAK_TEMP": "1100"}]}

        def dll(result, context):
            trace.append(("dll", _body_to_row(result)["SOAK_TEMP"], context))

        def resolve(**kwargs):
            trace.append(("resolve", kwargs["stage"], kwargs["images"]))
            result = copy.deepcopy(kwargs["current_result"])
            result["isState"] = True
            _set_single_body_value(result, "SOAK_TEMP", "1110")
            return result

        dependencies = ProcessAgentDependencies(
            resolve_agent_round=resolve,
            stage_input_changed=lambda before, after, stage: trace.append(
                ("changed", stage)
            ) or before != after,
            input_cache=input_cache,
            reasoning_cache=reasoning_cache,
            visible_cache=visible_cache,
            wind_power_prompt=lambda prompt: trace.append(("wind", prompt)) or prompt,
            retrieve_reheat_rag=lambda context: trace.append(("rag", context)) or "RAG",
            generate_reheat_images=dll,
            collect_reheat_context=lambda result: ("Tas", "soak", "growth", "distribution"),
            build_reheat_prompt=lambda **kwargs: trace.append(("prompt", kwargs)) or "PROMPT",
            invoke_reheat=lambda *args, **kwargs: {},
            sanitize_reheat=lambda original_result, candidate: candidate,
        )

        result = refine_reheat_process(
            original,
            "CTX",
            "sid",
            progress_callback=progress.append,
            dependencies=dependencies,
        )

        self.assertTrue(result["isState"])
        self.assertEqual([item[0] for item in trace].count("rag"), 1)
        self.assertEqual([item[0] for item in trace].count("dll"), 2)
        self.assertEqual(trace[0], ("rag", "CTX"))
        self.assertEqual(trace[1], ("dll", "1100", "CTX"))
        resolve_event = next(item for item in trace if item[0] == "resolve")
        self.assertEqual(
            [name for _path, name in resolve_event[2]],
            ["均热温度.png", "晶粒长大.png", "晶粒尺寸分布.png"],
        )
        self.assertEqual(progress[0], {
            "event_type": "module_decision",
            "attempt": 1,
            "stage": "reheat",
        })
        self.assertEqual(progress[1]["reasoning"], "加热判断摘要")
        self.assertEqual(len(input_cache["sid:reheat"]), 1)

    def test_rolling_stage_order_hard_gate_and_final_simulation(self):
        trace = []
        progress = []
        original = {"isState": False, "arrBody": [{"N1_DH_CAL": "100"}]}

        def resolve(**kwargs):
            trace.append(("resolve", kwargs["stage"], kwargs["images"]))
            result = copy.deepcopy(kwargs["current_result"])
            result["isState"] = True
            _set_single_body_value(result, "N1_DH_CAL", "90")
            return result

        dependencies = ProcessAgentDependencies(
            resolve_agent_round=resolve,
            stage_input_changed=lambda before, after, stage: trace.append(
                ("changed", stage)
            ) or before != after,
            input_cache={},
            reasoning_cache={"sid:roll": ["轧制判断摘要"]},
            visible_cache={"sid:roll": [{"conclusion": "通过"}]},
            wind_power_prompt=lambda prompt: trace.append(("wind", prompt)) or prompt,
            retrieve_roll_rag=lambda context: trace.append(("rag", context)) or "RAG",
            generate_roll_images=lambda result, context: trace.append(
                ("dll", _body_to_row(result)["N1_DH_CAL"], context)
            ),
            collect_roll_context=lambda result: "grain",
            build_roll_prompt=lambda **kwargs: trace.append(("prompt", kwargs)) or "PROMPT",
            invoke_roll=lambda *args, **kwargs: {},
            sanitize_roll=lambda original_result, candidate: candidate,
            require_valid_roll_result=lambda result: trace.append(("gate",)) or result,
        )

        result = refine_rolling_process(
            original,
            "CTX",
            "sid",
            progress_callback=progress.append,
            historical_roll_reference_markdown="HISTORY-10",
            dependencies=dependencies,
        )

        names = [item[0] for item in trace]
        self.assertTrue(result["isState"])
        self.assertEqual(names.count("rag"), 1)
        self.assertEqual(names.count("dll"), 2)
        self.assertLess(names.index("gate"), names.index("changed"))
        prompt_event = next(item for item in trace if item[0] == "prompt")
        self.assertEqual(
            prompt_event[1]["historical_roll_reference_markdown"],
            "HISTORY-10",
        )
        resolve_event = next(item for item in trace if item[0] == "resolve")
        self.assertEqual(resolve_event[2], [("grain", "各道次晶粒尺寸.png")])
        self.assertEqual(progress[0]["event_type"], "module_decision")
        self.assertEqual(progress[1]["reasoning"], "轧制判断摘要")

    def test_cooling_stage_order_and_final_simulation(self):
        trace = []
        progress = []
        original = {"isState": False, "arrBody": [{"SELF_TEMP": "470"}]}

        def resolve(**kwargs):
            trace.append(("resolve", kwargs["stage"], kwargs["images"]))
            result = copy.deepcopy(kwargs["current_result"])
            result["isState"] = True
            _set_single_body_value(result, "SELF_TEMP", "480")
            return result

        dependencies = ProcessAgentDependencies(
            resolve_agent_round=resolve,
            stage_input_changed=lambda before, after, stage: trace.append(
                ("changed", stage)
            ) or before != after,
            input_cache={},
            reasoning_cache={"sid:cooling": ["冷却判断摘要"]},
            visible_cache={"sid:cooling": [{"conclusion": "通过"}]},
            wind_power_prompt=lambda prompt: trace.append(("wind", prompt)) or prompt,
            retrieve_cooling_rag=lambda context: trace.append(("rag", context)) or "RAG",
            generate_cooling_images=lambda result, context: trace.append(
                ("dll", _body_to_row(result)["SELF_TEMP"], context)
            ),
            collect_cooling_context=lambda result: ("phase", "cct", "strength"),
            build_cooling_prompt=lambda **kwargs: trace.append(("prompt", kwargs)) or "PROMPT",
            invoke_cooling=lambda *args, **kwargs: {},
            sanitize_cooling=lambda original_result, candidate, context: candidate,
            user_requests_high_self_temp=lambda context: False,
            set_arrbody_field=_set_single_body_value,
            body_to_row=_body_to_row,
            to_float=lambda value: float(value) if value is not None else None,
        )

        result = refine_cooling_process(
            original,
            "CTX",
            "sid",
            progress_callback=progress.append,
            dependencies=dependencies,
        )

        self.assertTrue(result["isState"])
        self.assertEqual([item[0] for item in trace].count("rag"), 1)
        self.assertEqual([item[0] for item in trace].count("dll"), 2)
        resolve_event = next(item for item in trace if item[0] == "resolve")
        self.assertEqual(
            resolve_event[2],
            [("phase", "相组成.png"), ("cct", "CCT.png"), ("strength", "强化机制.PNG")],
        )
        self.assertEqual(progress[0]["event_type"], "module_decision")
        self.assertEqual(progress[1]["reasoning"], "冷却判断摘要")

    def test_cooling_invalid_model_result_uses_original_fallback(self):
        dll_values = []
        progress = []
        original = {"isState": False, "arrBody": [{"SELF_TEMP": "620"}]}
        dependencies = ProcessAgentDependencies(
            resolve_agent_round=lambda **kwargs: None,
            stage_input_changed=lambda before, after, stage: before != after,
            input_cache={},
            reasoning_cache={},
            visible_cache={},
            wind_power_prompt=lambda prompt: prompt,
            retrieve_cooling_rag=lambda context: "RAG",
            generate_cooling_images=lambda result, context: dll_values.append(
                _body_to_row(result)["SELF_TEMP"]
            ),
            collect_cooling_context=lambda result: (None, None, None),
            build_cooling_prompt=lambda **kwargs: "PROMPT",
            invoke_cooling=lambda *args, **kwargs: {},
            sanitize_cooling=lambda original_result, candidate, context: candidate,
            user_requests_high_self_temp=lambda context: False,
            set_arrbody_field=_set_single_body_value,
            body_to_row=_body_to_row,
            to_float=lambda value: float(value) if value is not None else None,
        )

        result = refine_cooling_process(
            original,
            "CTX",
            "sid",
            progress_callback=progress.append,
            dependencies=dependencies,
        )

        self.assertEqual(_body_to_row(result)["SELF_TEMP"], "485")
        self.assertFalse(result["isState"])
        self.assertEqual(dll_values, ["620", "485"])
        self.assertEqual(progress[-1]["event_type"], "fallback_applied")

    def test_composition_stage_does_not_use_historical_cooling_time_as_roll_gate(self):
        """后置微调只校验道次自身时间，不能被尚未设计的 TIME_ENTR 阻断。"""
        validate_flags = []
        original = {
            "isState": False,
            "strCoil": "coil",
            "arrBody": [{"C": "0.05"}, {"YS": "500"}, {"AIM_THICK": "80"}],
        }

        def invoke(messages, **kwargs):
            return SimpleNamespace(
                content=json.dumps(original, ensure_ascii=False),
                raw_metadata={"model": "test", "finish_reason": "stop"},
            )

        dependencies = CompositionRefinementDependencies(
            extract_target_thickness=lambda message: 80.0,
            extract_target_slab_thickness=lambda message: 420.0,
            lock_explicit_thickness_targets=lambda result, target, slab: copy.deepcopy(result),
            is_context_modification_request=lambda message: False,
            build_refinement_rag_context=lambda spec, message: "",
            build_cross_route_context=lambda sid: "",
            get_recent_session_context=lambda sid: "",
            filter_wind_session_context=lambda context: context,
            component_fields=frozenset({"C"}),
            performance_fields=frozenset({"YS"}),
            roll_fields=frozenset({"AIM_THICK"}),
            get_arrbody_key=lambda item: next(iter(item), None),
            build_historical_roll_reference=lambda result: "",
            build_wind_standard_redesign_instruction=lambda context, error="": "",
            reasoning_cache={},
            invoke_qwen=invoke,
            parse_json_object=json.loads,
            extract_qwen_agent_response=lambda parsed: (parsed, {}),
            restore_arrbody_fields=lambda candidate, source, fields: candidate,
            sanitize_refined_result=lambda source, refined, spec, **kwargs: copy.deepcopy(refined),
            validate_wind_result=lambda result, context: "",
            normalize_declared_pass_tail=lambda result: result,
            collect_deformation_pass_errors=lambda result, **kwargs: validate_flags.append(kwargs) or [],
            roll_errors_require_global_redesign=lambda errors: False,
            prepare_full_roll_redesign_baseline=lambda result: result,
            normalize_deformation_passes=lambda result, label, **kwargs: validate_flags.append(kwargs) or (result, ""),
            validate_dll_time_encodings=lambda result, include_cooling_start: "",
            enforce_performance_standard=lambda result, spec: result,
            cache_performance_baseline=lambda result, spec: None,
            performance_values=lambda result: {"YS": "500"},
            max_completion_tokens=32768,
        )

        refine_composition_process_performance(
            {}, original, "设计80mm风电钢", "sid", dependencies=dependencies
        )
        self.assertTrue(validate_flags)
        self.assertTrue(all(flag["validate_timing"] is False for flag in validate_flags))
        self.assertTrue(all(flag["validate_cooling_timing"] is False for flag in validate_flags))

    def test_cooling_stabilizer_runs_before_first_simulation_and_final_gate(self):
        trace = []
        original = {"isState": True, "arrBody": [{"TIME_ENTR": "historical"}]}

        def stabilize(result, context):
            trace.append(("stabilize", context))
            fixed = copy.deepcopy(result)
            _set_single_body_value(fixed, "TIME_ENTR", "fixed")
            return fixed

        dependencies = ProcessAgentDependencies(
            resolve_agent_round=lambda **kwargs: copy.deepcopy(kwargs["current_result"]),
            stage_input_changed=lambda before, after, stage: False,
            input_cache={}, reasoning_cache={}, visible_cache={},
            wind_power_prompt=lambda prompt: prompt,
            retrieve_cooling_rag=lambda context: "",
            generate_cooling_images=lambda result, context: trace.append(
                ("dll", _body_to_row(result)["TIME_ENTR"])
            ),
            collect_cooling_context=lambda result: (None, None, None),
            build_cooling_prompt=lambda **kwargs: "PROMPT",
            invoke_cooling=lambda *args, **kwargs: {},
            sanitize_cooling=lambda original_result, candidate, context: candidate,
            user_requests_high_self_temp=lambda context: True,
            set_arrbody_field=_set_single_body_value,
            body_to_row=_body_to_row,
            to_float=lambda value: float(value) if value is not None else None,
            stabilize_cooling_timing=stabilize,
            require_valid_cooling_timing=lambda result: trace.append(("gate",)) or result,
        )

        result = refine_cooling_process(original, "CTX", dependencies=dependencies)
        self.assertEqual(_body_to_row(result)["TIME_ENTR"], "fixed")
        self.assertEqual(trace[0], ("stabilize", "CTX"))
        self.assertEqual(trace[1], ("dll", "fixed"))
        self.assertIn(("gate",), trace)

    def test_roll_exit_repairs_invalid_pass_times_without_changing_process_values(self):
        """仅时间字段错误时，出口修复器应保留整套轧制规程并继续执行。"""
        import api as api_module

        fields = [
            {"R_PASS_ACT": "5"}, {"F_PASS_ACT": "3"},
            {"WIDTH_ROLL_START_REMARK": "2"},
            {"WIDTH_ROLL_END_REMARK": "3"},
            {"SLAB_THICK": "420"}, {"SLAB_WIDTH": "2000"},
            {"AIM_THICK": "80"}, {"AIM_WIDTH": "2450"}, {"FDT": "790"},
            {"TIME_ENTR": "2026-08-10 12:20:00.000"},
        ]
        thicknesses = [370, 320, 270, 220, 170, 130, 100, 80]
        temperatures = [1020, 995, 970, 945, 920, 870, 830, 790]
        widths = [2200, 2400, 2450, 2450, 2450, 2450, 2450, 2450]
        forces = [50000, 52000, 51000, 49000, 47000, 43000, 36000, 22000]
        speeds = [20, 23, 26, 29, 32, 36, 41, 45]
        for index in range(1, 31):
            active = index <= 8
            value_index = index - 1
            fields.extend([
                {f"N{index}_DH_CAL": str(thicknesses[value_index]) if active else "0"},
                {f"N{index}_DT_CAL": str(temperatures[value_index]) if active else "0"},
                {f"N{index}_DW_CAL": str(widths[value_index]) if active else "0"},
                {f"N{index}_FORCE": str(forces[value_index]) if active else "0"},
                {f"N{index}_SPD": str(speeds[value_index]) if active else "0"},
                {f"N{index}_ENTR_DATE": "bad-time" if active else ""},
            ])
        result = {"isState": False, "arrBody": fields}
        repaired = api_module._require_valid_pipeline_roll_result(result, "测试轧制门禁")
        repaired_row = api_module._matched_result_body_to_row(repaired)
        self.assertEqual(repaired_row["N8_DH_CAL"], "80")
        self.assertEqual(repaired_row["N8_FORCE"], "22000")
        self.assertEqual(repaired_row["WIDTH_ROLL_START_REMARK"], "2")
        self.assertEqual(repaired_row["WIDTH_ROLL_END_REMARK"], "3")
        self.assertIsNotNone(
            api_module._parse_pipeline_process_datetime(repaired_row["N8_ENTR_DATE"])
        )

    def test_roll_gate_allows_single_width_change_with_single_turn_marker(self):
        """一次宽度变化使用N/N标记时应直接放行，不阻断后续流程。"""
        import api as api_module

        fields = [
            {"R_PASS_ACT": "5"}, {"F_PASS_ACT": "3"},
            {"WIDTH_ROLL_START_REMARK": "2"},
            {"WIDTH_ROLL_END_REMARK": "2"},
            {"SLAB_THICK": "420"}, {"SLAB_WIDTH": "2000"},
            {"AIM_THICK": "80"}, {"AIM_WIDTH": "2450"}, {"FDT": "790"},
        ]
        thicknesses = [370, 320, 270, 220, 170, 130, 100, 80]
        temperatures = [1020, 995, 970, 945, 920, 870, 830, 790]
        # N2 相对 N1 只变化一次，后续全部保持不变。
        widths = [2200, 2450, 2450, 2450, 2450, 2450, 2450, 2450]
        for index in range(1, 31):
            active = index <= 8
            value_index = index - 1
            fields.extend([
                {f"N{index}_DH_CAL": str(thicknesses[value_index]) if active else "0"},
                {f"N{index}_DT_CAL": str(temperatures[value_index]) if active else "0"},
                {f"N{index}_DW_CAL": str(widths[value_index]) if active else "0"},
                {f"N{index}_FORCE": "30000" if active else "0"},
                {f"N{index}_SPD": "20" if active else "0"},
                {f"N{index}_ENTR_DATE": ""},
            ])

        errors = api_module._collect_pipeline_deformation_pass_errors(
            {"isState": False, "arrBody": fields},
            validate_timing=False,
            validate_cooling_timing=False,
        )
        self.assertFalse(errors)

    def test_roll_final_gate_allows_no_turn_after_deterministic_fallback(self):
        """模型重试耗尽后，没有转钢也应写成0/0并继续流程。"""
        import api as api_module

        result = self._build_roll_gate_result(
            widths=[2450, 2450, 2450, 2450, 2450],
            turn_start="2",
            turn_end="3",
        )
        finalized = api_module._require_valid_pipeline_roll_result(result, "无转钢兜底测试")
        row = api_module._matched_result_body_to_row(finalized)
        self.assertEqual(row["WIDTH_ROLL_START_REMARK"], "0")
        self.assertEqual(row["WIDTH_ROLL_END_REMARK"], "0")

    def test_roll_final_gate_allows_single_turn_after_deterministic_fallback(self):
        """模型重试耗尽后，一次转钢应写成同一道次N/N并继续流程。"""
        import api as api_module

        result = self._build_roll_gate_result(
            widths=[2200, 2450, 2450, 2450, 2450],
            turn_start="0",
            turn_end="0",
        )
        finalized = api_module._require_valid_pipeline_roll_result(result, "一次转钢兜底测试")
        row = api_module._matched_result_body_to_row(finalized)
        self.assertEqual(row["WIDTH_ROLL_START_REMARK"], "2")
        self.assertEqual(row["WIDTH_ROLL_END_REMARK"], "2")

    def test_roll_final_gate_flattens_widths_after_second_change(self):
        """多次转钢时只保留前两次，第二次变化后的宽度全部统一。"""
        import api as api_module

        result = self._build_roll_gate_result(
            widths=[2200, 2400, 2450, 2500, 2450],
            turn_start="2",
            turn_end="4",
        )
        finalized = api_module._require_valid_pipeline_roll_result(result, "多次转钢兜底测试")
        row = api_module._matched_result_body_to_row(finalized)
        self.assertEqual(row["WIDTH_ROLL_START_REMARK"], "2")
        self.assertEqual(row["WIDTH_ROLL_END_REMARK"], "3")
        self.assertEqual(row["N4_DW_CAL"], row["N3_DW_CAL"])
        self.assertEqual(row["N5_DW_CAL"], row["N3_DW_CAL"])

    def _build_roll_gate_result(self, widths, turn_start, turn_end):
        """构造仅转钢/宽度可能不合格、其他核心门禁均合格的测试规程。"""
        fields = [
            {"R_PASS_ACT": "5"}, {"F_PASS_ACT": "3"},
            {"WIDTH_ROLL_START_REMARK": turn_start},
            {"WIDTH_ROLL_END_REMARK": turn_end},
            {"SLAB_THICK": "420"}, {"SLAB_WIDTH": "2000"},
            {"AIM_THICK": "80"}, {"AIM_WIDTH": "2450"}, {"FDT": "790"},
        ]
        thicknesses = [370, 320, 270, 220, 170, 130, 100, 80]
        temperatures = [1020, 995, 970, 945, 920, 870, 830, 790]
        widths = [*widths, *([widths[-1]] * (8 - len(widths)))]
        times = [
            "2026-08-10 12:00:00.000",
            "2026-08-10 12:00:05.000",
            "2026-08-10 12:00:10.000",
            "2026-08-10 12:00:15.000",
            "2026-08-10 12:00:20.000",
            "2026-08-10 12:01:15.000",
            "2026-08-10 12:01:20.000",
            "2026-08-10 12:01:25.000",
        ]
        for index in range(1, 31):
            active = index <= 8
            value_index = index - 1
            fields.extend([
                {f"N{index}_DH_CAL": str(thicknesses[value_index]) if active else "0"},
                {f"N{index}_DT_CAL": str(temperatures[value_index]) if active else "0"},
                {f"N{index}_DW_CAL": str(widths[value_index]) if active else "0"},
                {f"N{index}_FORCE": "30000" if active else "0"},
                {f"N{index}_SPD": "20" if active else "0"},
                {f"N{index}_ENTR_DATE": times[value_index] if active else ""},
            ])
        return {"isState": False, "arrBody": fields}

    def test_roll_gate_reports_minimum_rough_and_finish_pass_errors(self):
        """粗轧不足5道或精轧不足3道时必须返回字段级明确原因。"""
        import api as api_module

        too_few_rough = self._build_roll_gate_result(
            widths=[2200, 2400, 2450, 2450, 2450],
            turn_start="2",
            turn_end="3",
        )
        _set_single_body_value(too_few_rough, "R_PASS_ACT", "4")
        _set_single_body_value(too_few_rough, "F_PASS_ACT", "4")
        rough_errors = api_module._collect_pipeline_deformation_pass_errors(
            too_few_rough,
            validate_timing=False,
            validate_cooling_timing=False,
        )
        self.assertTrue(any(
            "R_PASS_ACT=4" in error and "大于或等于 5" in error
            for error in rough_errors
        ))

        too_few_finish = self._build_roll_gate_result(
            widths=[2200, 2400, 2450, 2450, 2450],
            turn_start="2",
            turn_end="3",
        )
        _set_single_body_value(too_few_finish, "F_PASS_ACT", "2")
        finish_errors = api_module._collect_pipeline_deformation_pass_errors(
            too_few_finish,
            validate_timing=False,
            validate_cooling_timing=False,
        )
        self.assertTrue(any(
            "F_PASS_ACT=2" in error and "大于或等于 3" in error
            for error in finish_errors
        ))

    def test_x70_dll_reference_grade_is_not_empty(self):
        """X70 的 DLL 路由必须同步写入有效牌号，不能返回空字符串。"""
        import api as api_module

        self.assertEqual(api_module._select_pipeline_dll_reference_grade("X70"), "X70")
        source = {
            "isState": False,
            "strCoil": "test-coil",
            "strSteel": "X70",
            "arrBody": [{"STEEL_SIGN": "X70"}, {"AIM_THICK": "30"}],
        }
        dll_result, target_grade, reference_grade = api_module._build_pipeline_dll_matched_result(source)
        dll_row = api_module._matched_result_body_to_row(dll_result)
        self.assertEqual(target_grade, "X70")
        self.assertEqual(reference_grade, "X70")
        self.assertEqual(dll_result["strSteel"], "X70")
        self.assertEqual(dll_row["STEEL_SIGN"], "X70")

    def test_roll_gate_rejects_stale_turn_markers_after_pass_reallocation(self):
        """重新分配粗轧道次后，历史转钢标识超界时必须返回完整错误。"""
        import api as api_module

        fields = [
            {"R_PASS_ACT": "6"}, {"F_PASS_ACT": "2"},
            {"WIDTH_ROLL_START_REMARK": "2"},
            {"WIDTH_ROLL_END_REMARK": "9"},
            {"SLAB_THICK": "350"}, {"AIM_THICK": "30"}, {"FDT": "790"},
        ]
        thicknesses = [300, 250, 200, 150, 110, 80, 50, 30]
        temperatures = [1050, 1020, 990, 960, 930, 900, 840, 790]
        for index in range(1, 31):
            active = index <= len(thicknesses)
            value_index = index - 1
            fields.extend([
                {f"N{index}_DH_CAL": str(thicknesses[value_index]) if active else "0"},
                {f"N{index}_DT_CAL": str(temperatures[value_index]) if active else "0"},
                {f"N{index}_DW_CAL": "2450" if active else "0"},
                {f"N{index}_FORCE": "30000" if active else "0"},
                {f"N{index}_SPD": "20" if active else "0"},
                {f"N{index}_ENTR_DATE": ""},
            ])
        errors = api_module._collect_pipeline_deformation_pass_errors(
            {"isState": False, "arrBody": fields},
            validate_timing=False,
            validate_cooling_timing=False,
        )
        self.assertTrue(any("转钢道次标识无效" in error for error in errors))

    def test_api_main_flow_calls_new_entries_in_fixed_order(self):
        source = (API_DIR / "api.py").read_text(encoding="utf-8")
        anchors = [
            "# 智能体阶段 1/4：后置成分、性能及初步轧制规程微调。",
            "# 智能体阶段 2/4：加热工艺仿真与微调。",
            "# 智能体阶段 3/4：控制轧制仿真、道次重设计及最终硬门禁。",
            "# 智能体阶段 4/4：控制冷却仿真与最终性能校验。",
        ]
        positions = [source.index(anchor) for anchor in anchors]
        self.assertEqual(positions, sorted(positions))
        calls = [
            source.index("refine_composition_process_performance,", positions[0]),
            source.index("refine_reheat_process,", positions[1]),
            source.index("refine_rolling_process,", positions[2]),
            source.index("refine_cooling_process,", positions[3]),
        ]
        self.assertEqual(calls, sorted(calls))

    def test_all_four_entries_have_parameter_and_return_documentation(self):
        for function in (
            refine_composition_process_performance,
            refine_reheat_process,
            refine_rolling_process,
            refine_cooling_process,
        ):
            doc = function.__doc__ or ""
            self.assertIn("参数:", doc, function.__name__)
            self.assertIn("返回:", doc, function.__name__)


if __name__ == "__main__":
    unittest.main()
