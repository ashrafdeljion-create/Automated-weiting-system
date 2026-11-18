import os
import tempfile
import unittest

import pandas as pd
import pyreadstat
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_custom_weighting_methods_render_diagnostics_and_exports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            survey_path = os.path.join(temp_dir, "survey.sav")
            pyreadstat.write_sav(
                pd.DataFrame(
                    {
                        "sex": ["Female"] * 8 + ["Male"] * 2,
                        "age_group": ["20-29"] * 5 + ["30-39"] * 5,
                    }
                ),
                survey_path,
                column_labels={
                    "sex": "Respondent sex",
                    "age_group": "Age group",
                },
            )
            with open(survey_path, "rb") as survey_file:
                survey_bytes = survey_file.read()

        methods = [
            "Post-stratification (joint cells)",
            "Raking / RIM (IPF)",
            "Linear / GREG calibration (Samplics)",
            "Logit calibration (bounded)",
        ]
        for method in methods:
            with self.subTest(method=method):
                app = AppTest.from_file("app.py", default_timeout=30).run()
                app.session_state["benchmark_source"] = "Enter target percentages"
                app.session_state["weighting_method"] = method
                app.session_state["custom_variables"] = ["sex"]
                app.file_uploader[0].set_value(
                    ("survey.sav", survey_bytes, "application/octet-stream")
                ).run()
                self.assertFalse(app.exception)

                app.button[0].click().run()

                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertTrue(app.success)
                self.assertGreaterEqual(len(app.metric), 6)
                self.assertEqual(len(app.download_button), 4)


if __name__ == "__main__":
    unittest.main()
