import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import api as api_module


class PrecipitateMorphologyTests(unittest.TestCase):
    def test_failure_removes_stale_report_image_and_returns_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "ModelManage" / "TEST_COIL" / "Image"
            report_dir.mkdir(parents=True)
            stale_image = report_dir / "析出形貌.png"
            stale_image.write_bytes(b"old-image")

            matched_result = {
                "strCoil": "TEST_COIL",
                "strSteel": "X70",
                "arrBody": [{"STEELGRADE": "X70"}],
            }
            with patch.object(
                api_module,
                "PIPELINE_IMAGE_GENERATOR_BIN_DIR",
                temp_dir,
            ), patch.object(
                api_module,
                "_pipeline_exit_grain_steel_dir_name",
                return_value="Physical_Metallurgy_X70",
            ):
                result = api_module._draw_pipeline_precipitate_morphology(
                    matched_result,
                    "管线钢 X70",
                )

            self.assertFalse(result)
            self.assertFalse(stale_image.exists())

    def test_report_places_precipitate_morphology_after_phase_composition(self):
        source = inspect.getsource(api_module)
        list_start = source.index("pipeline_image_display_order = [")
        list_end = source.index("pipeline_image_stage_map = {", list_start)
        order_block = source[list_start:list_end]
        phase_index = order_block.index('"相组成.png"')
        precipitate_index = order_block.index('"析出形貌.png"')
        strengthening_index = order_block.index('"强化机制.png"')
        self.assertLess(phase_index, precipitate_index)
        self.assertLess(precipitate_index, strengthening_index)

    def test_precipitate_dll_receives_current_coil_fv_file(self):
        source = inspect.getsource(
            api_module._draw_pipeline_precipitate_morphology
        )
        self.assertIn(
            '_os.path.join(model_manage_dir, "Fv.txt")',
            source,
        )

    def test_precipitate_dll_uses_standard_canvas_without_ocr(self):
        project_root = (
            Path(api_module.PIPELINE_IMAGE_GENERATOR_BIN_DIR).parents[2]
            / "ANSTEEL_PrecipitateImageProcessorLib"
        )
        processor_text = (project_root / "PrecipitateImageProcessor.cs").read_text(
            encoding="utf-8"
        )
        project_text = (
            project_root / "ANSTEEL_PrecipitateImageProcessorLib.csproj"
        ).read_text(encoding="utf-8")
        self.assertIn("OutputWidth = 595", processor_text)
        self.assertIn("OutputHeight = 534", processor_text)
        self.assertIn("DetectPlotRectangle", processor_text)
        self.assertNotIn("ChartNumberProcessor", processor_text)
        self.assertNotIn("Tesseract", project_text)

    def test_all_reference_scripts_accept_coil_id_directly(self):
        zhb_root = (
            Path(api_module.PIPELINE_IMAGE_GENERATOR_BIN_DIR)
            / "ZHB"
        )
        for steel_dir in (
            "Physical_Metallurgy_X65",
            "Physical_Metallurgy_X70",
            "Physical_Metallurgy_X80NG",
        ):
            script_path = zhb_root / steel_dir / "bigmodel_Picture_pre.py"
            script_text = script_path.read_text(encoding="utf-8")
            self.assertIn("_safe_coil_id(sys.argv[1])", script_text)
            self.assertNotIn("Picture pre PD.txt", script_text)
            self.assertNotIn("os.system", script_text)


if __name__ == "__main__":
    unittest.main()
