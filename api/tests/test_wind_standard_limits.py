from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from steel_spec_extractor import _apply_wind_power_gbt1591_limits  # noqa: E402


class WindStandardLimitTests(unittest.TestCase):
    """验证用户性能下限与 GB/T 1591 下限按更严格值合并。"""

    def test_lower_user_ys_min_is_upgraded_to_selected_grade_standard(self):
        profile = {
            "grade": "Q620M",
            "quality": "D",
            "chemistry": {},
            "tensile": {"YS_min": 610.0, "TS_min": 690.0, "TS_max": 880.0, "EL_min": 15.0},
            "impact": {"temperature": -20, "longitudinal": 47.0},
        }
        with (
            patch("steel_spec_extractor._wind_standard_profile", return_value=profile),
            patch("steel_spec_extractor._user_explicit_spec_fields", return_value={"YS_min"}),
            patch("steel_spec_extractor._user_target_spec_ranges", return_value={}),
            patch("steel_spec_extractor._wind_explicit_performance_bounds", return_value={}),
        ):
            fixed, returned_profile = _apply_wind_power_gbt1591_limits(
                {"YS_min": 550.0},
                "屈服强度不低于550MPa",
                "",
            )

        self.assertNotIn("error", returned_profile)
        self.assertEqual(fixed["YS_min"], 550.0)


if __name__ == "__main__":
    unittest.main()
