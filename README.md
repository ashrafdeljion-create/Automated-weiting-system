# Survey weighting and visualization

A Streamlit application for importing SPSS survey files, fitting survey weights
to demographic or custom targets, reviewing diagnostics, and exporting a
weighted SPSS file and reports.

## Project files

- `app.py` — Streamlit interface, target mapping, validation, and exports.
- `weighting_engine.py` — post-stratification, IPF raking, calibration, trimming,
  normalization, and diagnostics.
- `visuals.py` — interactive Plotly comparison, weight distribution, and
  residual-margin charts.
- `utils.py` — Stats SA benchmark loading, SPSS metadata handling, and report
  utilities.
- `data/` — archived official Stats SA benchmark CSV and PDF.
- `test_weighting.py` — unit tests for benchmark and weighting behavior.

## Run locally

Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, then drag a `.sav` file into the
uploader. The app reads variable labels, value labels, and SPSS user-missing
definitions. It preserves those supported metadata fields when writing the
weighted `.sav` output and adds a labeled weight variable.

Run the tests with:

```powershell
python -m unittest
```

## Weighting workflow

1. Select the weighting method and benchmark source.
2. Map survey categories to Stats SA categories, or select custom survey
   variables and enter/upload targets.
3. Review missing or unmapped responses. Choose whether incomplete records
   should be excluded; excluded records receive a blank weight.
4. Choose weight trimming and the normalization total. Trimming is followed by
   constrained renormalization so the final weights still sum to `n` or, when
   a complete Stats SA benchmark is retained, the Stats SA population total `N`.
5. Calculate and review the diagnostics, distribution charts, and residual
   margin errors before downloading the outputs.

### Methods

- **Post-stratification / cell weighting:** matches the exact joint target cells.
  Positive target cells with no sample respondents cannot be created by
  weighting; the app can optionally restrict the result to represented cells.
- **Raking / RIM:** uses `ipfn` to iteratively match multiple marginal
  distributions. It reports convergence and final margin errors.
- **Linear / GREG calibration:** uses the requested `samplics` calibration
  implementation. Linear calibration can produce non-positive weights for
  incompatible controls; in that case, revise the targets or use another
  estimator.
- **Bounded logit calibration:** fits marginal controls while constraining
  adjustment factors. Positive bounds cannot attain a zero-percent target.

Trimming caps are expressed as multiples of the requested final mean weight.
Percentile trimming uses the selected percentiles of the fitted weight
distribution. The final weights are rescaled subject to the selected bounds.
If the requested total is infeasible under chosen caps, the app reports an
error instead of silently violating the caps or total.

**Maintenance note:** Samplics is archived and no longer maintained. It is
included because the application requirements requested it; verify its
compatibility and estimator behavior before relying on it for a production
study. The application does not calculate complex-survey standard errors.

## Targets

The built-in Stats SA population estimates provide age, sex, and province
targets, plus national population-group targets. The provincial benchmark in
the archived table does not provide provincial population-group splits.
Household and personal income are not available in this benchmark; use targets
from a reputable, documented source rather than treating sample proportions as
population figures. The root folder also contains example raking and
post-stratification target CSVs.

Custom target CSV formats:

- **Raking and calibration:** `variable,category,target_percent`; each selected
  variable must have one row per included category and its percentages must sum
  to 100.
- **Post-stratification:** one column per selected survey variable plus
  `target_percent`; each joint cell must be unique and the percentages must sum
  to 100.

The app can download a template based on the included sample categories. Those
sample-based values are starting points only, not verified benchmarks.

The live Stats SA publication is fetched when available and cached for one day.
If it cannot be reached, the app falls back to the official benchmark archive
in `data/`. The source year and download links are shown in the app.

## Deploy on Streamlit Community Cloud

1. Push the project files to a GitHub repository.
2. In Streamlit Community Cloud, select that repository, branch, and `app.py`.
3. Deploy. The platform installs packages from `requirements.txt`.
4. Keep survey uploads and generated weighted files private; do not commit them.
   The `.gitignore` excludes `.sav` files and generated report files by default.

No secrets are required for the built-in benchmark fetch. For other integrations,
store credentials in Streamlit secrets and never commit `.streamlit/secrets.toml`.
