import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyreadstat

from utils import (
    age_value_to_group,
    fetch_latest_stats_sa_population,
    load_stats_sa_population,
    prepare_analysis_data,
    read_sav_bytes,
    stats_sa_targets,
    weighted_sav_bytes,
)
from weighting_engine import (
    calibration_weights,
    comparison_table,
    missing_positive_raking_categories,
    poststratification_weights,
    rake_weights,
    raking_targets_from_target_table,
    restrict_raking_targets_to_observed,
    trim_and_normalize_weights,
    weighting_diagnostics,
)


class WeightingTests(unittest.TestCase):
    def test_age_values_are_grouped_using_stats_sa_age_bands(self):
        self.assertEqual(age_value_to_group(19), "15-19")
        self.assertEqual(age_value_to_group(80), "80+")
        self.assertIsNone(age_value_to_group(121))

    def test_poststratification_matches_joint_target_percentages(self):
        data = pd.DataFrame({"sex": ["Female"] * 2 + ["Male"] * 8})
        target = pd.DataFrame(
            {
                "sex": ["Female", "Male"],
                "target_percent": [50.0, 50.0],
            }
        )

        weighted = poststratification_weights(data, target, ["sex"])
        weighted_share = (
            weighted.groupby("sex")["weight"].sum()
            / weighted["weight"].sum()
            * 100
        )

        self.assertAlmostEqual(weighted_share["Female"], 50.0)
        self.assertAlmostEqual(weighted_share["Male"], 50.0)

    def test_poststratification_rejects_target_cells_missing_from_sample(self):
        data = pd.DataFrame({"sex": ["Female", "Male"]})
        target = pd.DataFrame(
            {
                "sex": ["Female", "Male", "Other"],
                "target_percent": [40.0, 40.0, 20.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "no respondents"):
            poststratification_weights(data, target, ["sex"])

    def test_raking_matches_multiple_margins(self):
        sample = pd.DataFrame(
            {
                "age": ["young"] * 6 + ["older"] * 4,
                "sex": [
                    "Female",
                    "Female",
                    "Female",
                    "Female",
                    "Male",
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                    "Male",
                ],
            }
        )
        targets = {
            "age": {"young": 50.0, "older": 50.0},
            "sex": {"Female": 50.0, "Male": 50.0},
        }

        weights, converged = rake_weights(sample, targets)

        self.assertTrue(converged)
        self.assertAlmostEqual(weights.sum(), 100.0, places=5)
        for column, target in targets.items():
            shares = sample.groupby(column).apply(
                lambda group: weights.loc[group.index].sum() / weights.sum() * 100
            )
            for category, target_percent in target.items():
                self.assertAlmostEqual(shares[category], target_percent, places=3)

    def test_target_table_builds_margins(self):
        target_data = pd.DataFrame(
            {
                "age": ["young", "young", "older", "older"],
                "sex": ["F", "M", "F", "M"],
                "target_percent": [20.0, 30.0, 10.0, 40.0],
            }
        )
        targets = raking_targets_from_target_table(
            target_data, ["age", "sex"]
        )

        self.assertEqual(targets["age"], {"older": 50.0, "young": 50.0})
        self.assertEqual(targets["sex"], {"F": 30.0, "M": 70.0})

    def test_missing_positive_raking_categories_are_reported(self):
        data = pd.DataFrame({"age": ["20-24", "25-29", "20-24"]})
        targets = {
            "age": {"15-19": 10.0, "20-24": 40.0, "25-29": 50.0},
        }

        self.assertEqual(
            missing_positive_raking_categories(data, targets),
            {"age": ["15-19"]},
        )

    def test_raking_can_renormalize_targets_for_observed_categories(self):
        data = pd.DataFrame(
            {
                "age": ["15-19", "20-24"],
                "sex": ["Male", "Male"],
            }
        )
        targets = {
            "age": {
                "0-4": 10.0,
                "15-19": 20.0,
                "20-24": 30.0,
                "25-29": 40.0,
            },
            "sex": {"Female": 50.0, "Male": 50.0},
        }

        adjusted, excluded = restrict_raking_targets_to_observed(data, targets)

        self.assertEqual(adjusted["age"], {"15-19": 40.0, "20-24": 60.0})
        self.assertEqual(adjusted["sex"], {"Male": 100.0})
        self.assertEqual(
            excluded,
            {"age": ["0-4", "25-29"], "sex": ["Female"]},
        )

    def test_raking_restriction_rejects_unmatched_sample_categories(self):
        data = pd.DataFrame({"sex": ["Female", "Other"]})
        targets = {"sex": {"Female": 50.0, "Male": 50.0}}

        with self.assertRaisesRegex(ValueError, "missing from the target"):
            restrict_raking_targets_to_observed(data, targets)

    def test_linear_calibration_matches_control_margin(self):
        data = pd.DataFrame({"sex": ["Female"] * 8 + ["Male"] * 2})
        targets = {"sex": {"Female": 50.0, "Male": 50.0}}

        weights = calibration_weights(
            data, targets, "Linear / GREG (Samplics)"
        )

        self.assertAlmostEqual(weights.sum(), len(data), places=5)
        self.assertAlmostEqual(weights[data["sex"] == "Female"].sum(), 5.0, places=5)
        self.assertAlmostEqual(weights[data["sex"] == "Male"].sum(), 5.0, places=5)

    def test_bounded_logit_calibration_matches_control_margin(self):
        data = pd.DataFrame({"sex": ["Female"] * 8 + ["Male"] * 2})
        targets = {"sex": {"Female": 50.0, "Male": 50.0}}

        weights = calibration_weights(
            data, targets, "Logit (bounded)", logit_bounds=(0.3, 3.0)
        )

        self.assertAlmostEqual(weights.sum(), len(data), places=5)
        self.assertAlmostEqual(weights[data["sex"] == "Female"].sum(), 5.0, places=5)
        self.assertAlmostEqual(weights[data["sex"] == "Male"].sum(), 5.0, places=5)
        self.assertGreaterEqual((weights / 1).min(), 0.3)
        self.assertLessEqual((weights / 1).max(), 3.0)

    def test_logit_rejects_zero_targets_with_positive_bounds(self):
        data = pd.DataFrame({"sex": ["Female", "Male"]})
        targets = {"sex": {"Female": 100.0, "Male": 0.0}}

        with self.assertRaisesRegex(ValueError, "zero-percent target"):
            calibration_weights(data, targets, "Logit (bounded)")

    def test_trimmed_weights_keep_caps_and_requested_sum(self):
        raw = pd.Series([0.1, 1.0, 5.0])

        weights, info = trim_and_normalize_weights(
            raw,
            target_sum=30.0,
            trim_method="Min/max caps",
            lower_bound=0.5,
            upper_bound=2.0,
        )

        self.assertAlmostEqual(weights.sum(), 30.0, places=7)
        self.assertGreaterEqual(weights.min(), 5.0 - 1e-7)
        self.assertLessEqual(weights.max(), 20.0 + 1e-7)
        self.assertEqual(info["trimmed_count"], 3)

    def test_percentile_trim_is_renormalized(self):
        weights, _ = trim_and_normalize_weights(
            pd.Series([1.0, 2.0, 3.0, 100.0]),
            target_sum=20.0,
            trim_method="Percentile caps",
            lower_percentile=25,
            upper_percentile=75,
        )

        self.assertAlmostEqual(weights.sum(), 20.0, places=7)
        self.assertTrue(np.isfinite(weights).all())

    def test_diagnostics_and_target_comparison(self):
        data = pd.DataFrame({"sex": ["Female", "Male"]})
        weights = pd.Series([1.0, 3.0])
        diagnostics = weighting_diagnostics(weights)
        comparison = comparison_table(
            data,
            weights,
            {"sex": {"Female": 25.0, "Male": 75.0}},
        )

        self.assertAlmostEqual(diagnostics["design_effect"], 1.25)
        self.assertAlmostEqual(diagnostics["effective_sample_size"], 1.6)
        self.assertEqual(
            comparison["Margin error (pp)"].abs().max(),
            0.0,
        )

    @patch("utils.parse_stats_sa_population_pdf")
    @patch("utils.urlopen")
    def test_stats_sa_fetch_skips_broken_pdf_and_uses_previous_year(
        self, mock_urlopen, mock_parse
    ):
        mock_urlopen.side_effect = [
            BytesIO(b"%PDF broken"),
            BytesIO(b"%PDF valid"),
        ]
        expected = pd.DataFrame({"year": [2025]})
        mock_parse.side_effect = [ValueError("truncated PDF"), expected]

        table, year, url, pdf_bytes = fetch_latest_stats_sa_population(
            current_year=2026
        )

        self.assertEqual(year, 2025)
        self.assertEqual(table["year"].tolist(), [2025])
        self.assertIn("P03022025.pdf", url)
        self.assertEqual(pdf_bytes, b"%PDF valid")

    @patch("utils.fetch_latest_stats_sa_population")
    def test_stats_sa_loader_uses_official_archive_when_live_fetch_fails(
        self, mock_fetch
    ):
        mock_fetch.side_effect = RuntimeError("Stats SA unavailable")

        population, year, url, archive_used, pdf_bytes = load_stats_sa_population()

        self.assertEqual(year, 2026)
        self.assertIn("P03022026.pdf", url)
        self.assertTrue(archive_used)
        self.assertEqual(len(population), 476)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_stats_sa_province_targets_use_national_provincial_rows(self):
        population = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "stats_sa_population_2026.csv",
            )
        )

        targets = stats_sa_targets(
            population, "South Africa", ["province"]
        )

        self.assertEqual(set(targets["province"]), set(population.loc[
            population["geography"] != "South Africa", "geography"
        ].unique()))
        self.assertAlmostEqual(targets["target_percent"].sum(), 100.0)

    def test_stats_sa_province_population_group_cross_tab_is_rejected(self):
        population = pd.read_csv(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "stats_sa_population_2026.csv",
            )
        )

        with self.assertRaisesRegex(ValueError, "does not provide"):
            stats_sa_targets(
                population,
                "South Africa",
                ["province", "population_group"],
            )

    def test_weighted_sav_preserves_labels_and_missing_ranges(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.sav")
            source = pd.DataFrame({"sex": [1.0, 2.0, 9.0]})
            pyreadstat.write_sav(
                source,
                input_path,
                column_labels={"sex": "Respondent sex"},
                variable_value_labels={"sex": {1.0: "Female", 2.0: "Male"}},
                missing_ranges={"sex": [{"lo": 9.0, "hi": 9.0}]},
            )
            with open(input_path, "rb") as input_file:
                data, metadata = read_sav_bytes(input_file.read())
            analysis_data, missing_masks = prepare_analysis_data(data, metadata)
            data["survey_weight"] = [1.0, 2.0, float("nan")]

            output_bytes = weighted_sav_bytes(
                data, metadata, "survey_weight"
            )
            restored, restored_metadata = pyreadstat.read_sav(
                BytesIO(output_bytes), user_missing=True
            )

        self.assertTrue(missing_masks["sex"].iloc[2])
        self.assertTrue(pd.isna(analysis_data["sex"].iloc[2]))
        self.assertEqual(
            restored_metadata.column_names_to_labels["sex"],
            "Respondent sex",
        )
        self.assertEqual(
            restored_metadata.column_names_to_labels["survey_weight"],
            "Final survey weight",
        )
        self.assertEqual(
            restored_metadata.variable_value_labels["sex"][1.0],
            "Female",
        )
        self.assertEqual(restored_metadata.missing_ranges["sex"][0]["lo"], 9.0)
        self.assertEqual(restored["sex"].iloc[2], 9.0)


if __name__ == "__main__":
    unittest.main()
