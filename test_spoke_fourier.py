import math
import tempfile
import unittest
from pathlib import Path

import polars as pl

from spoke_fourier import (
    attach_fail_indicator,
    attach_polar_coordinates,
    compute_spoke_signals,
    resolve_spectrum_bands,
    run_spoke_fourier,
    SpokeConfig,
)


class SpokeFourierScoreTests(unittest.TestCase):
    def test_default_360_bin_spectrum_bands(self) -> None:
        self.assertEqual(resolve_spectrum_bands(angular_bins=360), (72, 90))

    def test_radial_spoke_scores_above_full_ring(self) -> None:
        rows: list[dict[str, str]] = []
        for wafer_id, pattern in (("SPOKE", "spoke"), ("RING", "ring")):
            for x in range(-25, 26):
                for y in range(-25, 26):
                    radius = math.hypot(x, y) / 25.0
                    if radius > 1.0:
                        continue
                    theta = math.atan2(y, x)
                    is_fail = (
                        pattern == "spoke"
                        and radius >= 0.12
                        and abs(theta) <= math.radians(5.0)
                    ) or (
                        pattern == "ring"
                        and 0.62 <= radius <= 0.72
                    )
                    rows.append(
                        {
                            "root_lot_id": "LOT1",
                            "wafer_id": wafer_id,
                            "chip_x_pos": str(x),
                            "chip_y_pos": str(y),
                            "bin_no": "12" if is_fail else "1",
                        }
                    )

        raw_df = attach_fail_indicator(
            pl.DataFrame(rows),
            bin_col="bin_no",
            fail_bin_nos=("12",),
        )
        analysis_df, _ = attach_polar_coordinates(
            raw_df,
            group_cols=["root_lot_id", "wafer_id"],
            x_col="chip_x_pos",
            y_col="chip_y_pos",
        )
        result = compute_spoke_signals(
            analysis_df,
            group_cols=["root_lot_id", "wafer_id"],
            angular_bins=360,
            low_freq_max_harmonic=None,
            broadband_min_harmonic=None,
            sinc_width_min_deg=1.0,
            sinc_width_max_deg=45.0,
            sinc_width_step_deg=0.5,
            min_chips=20,
        )

        spoke = result.filter(pl.col("wafer_id") == "SPOKE").row(0, named=True)
        ring = result.filter(pl.col("wafer_id") == "RING").row(0, named=True)
        self.assertGreater(spoke["spoke_fourier_signal"], ring["spoke_fourier_signal"] * 100)
        self.assertGreater(spoke["low_freq_energy_ratio"], ring["low_freq_energy_ratio"])
        self.assertLess(spoke["broadband_energy_ratio"], ring["broadband_energy_ratio"])
        self.assertGreaterEqual(spoke["estimated_spoke_width_deg"], 7.0)
        self.assertLessEqual(spoke["estimated_spoke_width_deg"], 14.0)
        self.assertAlmostEqual(spoke["spoke_fail_rate"], spoke["fail_rate"], places=6)
        self.assertLess(ring["spoke_fail_rate"], ring["fail_rate"] * 0.01)
        self.assertLessEqual(abs(spoke["spoke_theta_center_deg"]), 3.0)

    def test_saved_result_contains_only_requested_columns(self) -> None:
        rows = [
            {
                "LOT": "LOT1",
                "WF": "W01",
                "X": str(x),
                "Y": str(y),
                "BIN": "12" if y == 0 and x >= 0 else "1",
            }
            for x in range(-2, 3)
            for y in range(-2, 3)
            if x * x + y * y <= 4
        ]
        expected_columns = [
            "root_lot_id",
            "wafer_id",
            "spoke_fail_rate",
            "estimated_spoke_width_deg",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.csv"
            output_path = Path(temp_dir) / "output.csv"
            pl.DataFrame(rows).write_csv(input_path)
            run = run_spoke_fourier(
                SpokeConfig(
                    input_path=input_path,
                    output_csv=output_path,
                    fail_bin_nos=[12],
                    group_cols=("LOT", "WF"),
                    x_col="X",
                    y_col="Y",
                    bin_col="BIN",
                    min_chips=1,
                )
            )

            self.assertEqual(run.result.columns, expected_columns)
            self.assertEqual(pl.read_csv(output_path).columns, expected_columns)


if __name__ == "__main__":
    unittest.main()
