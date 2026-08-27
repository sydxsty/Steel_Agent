from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from design_versioning import (  # noqa: E402
    DesignSnapshotStore,
    build_normalized_design_task,
    build_resolved_design_request,
    resolve_design_reference,
    validate_revision_constraints,
)
from pipeline_agents import (  # noqa: E402
    refine_cooling_process,
    refine_reheat_process,
    refine_rolling_process,
)


def _matched_result(
    *,
    nb: str = "0.050",
    mo: str = "0.020",
    aim_thick: str = "30",
    slab_thick: str = "350",
    ys: str = "650",
    ts: str = "720",
    el: str = "24",
    akv: str = "120",
) -> dict:
    return {
        "isState": True,
        "strCoil": "test",
        "strSteel": "X70",
        "arrBody": [
            {"C": "0.060"}, {"MN": "1.70"}, {"NB": nb}, {"MO": mo},
            {"V": "0.010"}, {"TI": "0.015"},
            {"AIM_THICK": aim_thick}, {"SLAB_THICK": slab_thick},
            {"N1_DH_CAL": "100"}, {"N2_DH_CAL": aim_thick},
            {"YS": ys}, {"TS": ts}, {"EL": el}, {"AKV": akv},
        ],
    }


class DesignVersioningTests(unittest.TestCase):
    """验证指代解析、标准任务和父子方案统一强约束。"""

    def setUp(self):
        self.store = DesignSnapshotStore()
        self.store._db_ready = False
        self.first = self.store.save_snapshot(
            session_id="session",
            material_purpose="管线钢",
            target_grade="X90",
            aim_thick=30,
            slab_thick=350,
            user_request="设计X90管线钢，成品30mm，板坯350mm",
            change_plan={"mode": "new"},
            spec_result={"YS_min": 625},
            matched_result=_matched_result(),
            fact_table=[],
        )
        self.second = self.store.save_snapshot(
            session_id="session",
            material_purpose="管线钢",
            target_grade="X90",
            aim_thick=80,
            slab_thick=350,
            user_request="设计X90管线钢，成品80mm，板坯350mm",
            change_plan={"mode": "new"},
            spec_result={"YS_min": 625},
            matched_result=_matched_result(aim_thick="80"),
            fact_table=[],
        )

    def _task(self):
        model_result = {
            "optimization_targets": {
                "microalloy_total": "decrease",
                "composition": "redesign",
                "process": "redesign",
            },
            "editable_scopes": [
                "composition", "heating", "rolling", "cooling", "performance",
            ],
            # 使用NB+MO证明后端没有把“微合金”硬编码成NB/V/TI。
            "selected_microalloy_fields": ["NB", "MO"],
            "summary": "降低模型自主选择的微合金总量并完整重设计工艺",
        }
        with patch("design_versioning._invoke_deepseek_json", return_value=model_result):
            return build_normalized_design_task(
                "降低以上设计的微合金元素，保持性能不降低重新设计成分工艺",
                self.first,
            )

    def test_versions_and_reference_priority(self):
        self.assertEqual(self.first["version_no"], 1)
        self.assertEqual(self.second["version_no"], 2)
        with patch("design_versioning.design_snapshot_store", self.store):
            active = resolve_design_reference(
                "session",
                "降低以上设计的微合金元素，保持性能不降低重新设计成分工艺",
                "管线钢",
                self.first["design_id"],
            )
            explicit = resolve_design_reference(
                "session",
                "继续调整这个方案",
                "管线钢",
                self.first["design_id"],
                self.second["design_id"],
            )
        self.assertEqual(active["snapshot"]["design_id"], self.first["design_id"])
        self.assertEqual(explicit["snapshot"]["design_id"], self.second["design_id"])

    def test_grade_only_reference_is_ambiguous(self):
        with patch("design_versioning.design_snapshot_store", self.store):
            resolved = resolve_design_reference(
                "session",
                "降低X90设计的微合金元素",
                "管线钢",
                None,
            )
        self.assertEqual(resolved["mode"], "clarification")
        self.assertEqual(len(resolved["candidates"]), 2)

    def test_normalized_task_keeps_original_prompt_and_inherits_specs(self):
        task = self._task()
        self.assertEqual(task["selected_microalloy_fields"], ["NB", "MO"])
        self.assertEqual(task["inherited_constraints"]["steel_grade"], "X90")
        self.assertEqual(task["inherited_constraints"]["product_thickness_mm"], 30)
        self.assertEqual(task["inherited_constraints"]["slab_thickness_mm"], 350)
        self.assertEqual(set(task["relative_performance_constraints"]), {"YS", "TS", "EL", "AKV"})
        resolved_prompt = build_resolved_design_request(task)
        self.assertIn(task["original_user_prompt"], resolved_prompt)
        self.assertIn("X90", resolved_prompt)
        self.assertIn("成品厚度：30 mm", resolved_prompt)

    def test_explicit_thickness_change_becomes_new_locked_target(self):
        model_result = {
            "optimization_targets": {"composition": "unchanged", "process": "redesign"},
            "editable_scopes": ["rolling", "cooling", "performance"],
            "selected_microalloy_fields": [],
            "summary": "调整目标厚度并重设计工艺",
        }
        with patch("design_versioning._invoke_deepseek_json", return_value=model_result):
            task = build_normalized_design_task(
                "将以上方案的成品厚度调整为35mm，重新设计工艺",
                self.first,
            )
        self.assertEqual(task["inherited_constraints"]["product_thickness_mm"], 35)
        self.assertEqual(task["inherited_constraints"]["slab_thickness_mm"], 350)

    def test_parent_child_constraints_keep_specs_and_performance_without_microalloy_sum_gate(self):
        task = self._task()
        valid = _matched_result(nb="0.045", mo="0.015", ys="660", ts="730", el="25", akv="130")
        errors = validate_revision_constraints(
            valid,
            self.first,
            task,
            {"YS_min": 625, "TS_min": 695, "EL_min": 18.5, "AKV_min": 40},
            require_final_pass=True,
        )
        self.assertEqual(errors, [])

        invalid = _matched_result(
            nb="0.055", mo="0.020", aim_thick="27.5", slab_thick="370",
            ys="640", ts="710", el="23", akv="110",
        )
        errors = validate_revision_constraints(
            invalid,
            self.first,
            task,
            {"YS_min": 625, "TS_min": 695, "EL_min": 18.5, "AKV_min": 40},
            require_final_pass=True,
        )
        joined = "；".join(errors)
        self.assertIn("成品厚度", joined)
        self.assertIn("板坯厚度", joined)
        self.assertNotIn("微合金元素总量必须严格降低", joined)
        self.assertIn("YS不得低于父方案", joined)
        self.assertIn("TS不得低于父方案", joined)

    def test_process_agents_no_longer_accept_stage_freezing_arguments(self):
        for func in (refine_reheat_process, refine_rolling_process, refine_cooling_process):
            parameters = inspect.signature(func).parameters
            self.assertNotIn("design_change_plan", parameters)
            self.assertNotIn("reference_snapshot", parameters)


if __name__ == "__main__":
    unittest.main()
