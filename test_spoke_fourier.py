import math
import unittest

import polars as pl

from spoke_fourier import (
    attach_defect_indicator,
    attach_polar_coordinates,
    compute_spoke_signals,
    resolve_spectrum_bands,
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
                    is_defect = (
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
                            "bin_no": "12" if is_defect else "1",
                        }
                    )

        raw_df = attach_defect_indicator(
            pl.DataFrame(rows),
            bin_col="bin_no",
            defect_bin_nos=("12",),
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
        self.assertAlmostEqual(spoke["spoke_defect_rate"], spoke["defect_rate"], places=6)
        self.assertLess(ring["spoke_defect_rate"], ring["defect_rate"] * 0.01)
        self.assertLessEqual(abs(spoke["spoke_theta_center_deg"]), 3.0)


if __name__ == "__main__":
    unittest.main()
