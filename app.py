"""Streamlit interface for survey weighting, diagnostics, and exports."""

import hashlib
import json
import re

import pandas as pd
import streamlit as st

from utils import (
    DIMENSIONS,
    PROVINCES,
    age_value_to_group,
    category_label,
    excel_report_bytes,
    labeled_category_label,
    load_stats_sa_population,
    make_sample_columns,
    prepare_analysis_data,
    read_sav_bytes,
    stats_sa_targets,
    suggested_benchmark_category,
    suggested_dataset_column,
    weighted_sav_bytes,
)
from visuals import comparison_figure, margin_error_figure, weight_distribution_figure
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


st.set_page_config(
    page_title="Survey weighting and visualization",
    page_icon=":material/analytics:",
    layout="wide",
)


@st.cache_data(ttl="1D", max_entries=1)
def load_benchmarks():
    return load_stats_sa_population()


def _variable_label(variable, metadata):
    label = (metadata.column_names_to_labels or {}).get(variable)
    return f"{variable} — {label}" if label else variable


def _map_stats_dimension(data, dimension, variable, categories, metadata, file_key):
    output_column = f"__benchmark_{dimension}"
    series = data[variable]
    continuous_age = (
        dimension == "age_group"
        and pd.api.types.is_numeric_dtype(series)
        and series.nunique(dropna=True) > 15
    )
    if continuous_age:
        mapped = series.map(age_value_to_group)
        invalid = series.notna() & mapped.isna()
        if invalid.any():
            raise ValueError(
                f"'{variable}' contains age values outside the supported "
                "0–120 range or non-integer ages."
            )
        data[output_column] = mapped
        st.caption(
            f"`{variable}` is grouped into the five-year age bands used by "
            "the Stats SA table."
        )
        return output_column

    values = series.dropna().unique().tolist()
    values = sorted(values, key=str)
    mapping = {}
    with st.expander(
        f"Map **{_variable_label(variable, metadata)}** to Stats SA",
        expanded=True,
    ):
        st.caption("Review every category mapping. Numeric codes are never guessed.")
        for value in values:
            raw_value_label = (metadata.variable_value_labels or {}).get(
                variable, {}
            ).get(value)
            display_value = labeled_category_label(
                variable, value, metadata.variable_value_labels or {}
            )
            suggestion = (
                suggested_benchmark_category(
                    dimension, raw_value_label, categories
                )
                if raw_value_label is not None
                else None
            )
            if suggestion is None:
                suggestion = suggested_benchmark_category(
                    dimension, display_value, categories
                )
            if suggestion is None:
                suggestion = suggested_benchmark_category(
                    dimension, value, categories
                )
            options = [None, *categories]
            selected_index = (
                categories.index(suggestion) + 1
                if suggestion in categories
                else 0
            )
            key_material = f"{file_key}|{dimension}|{variable}|{value!r}"
            key = "category_map_" + hashlib.sha256(
                key_material.encode("utf-8")
            ).hexdigest()[:12]
            selected = st.selectbox(
                display_value,
                options=options,
                index=selected_index,
                format_func=lambda item: (
                    "Select a benchmark category" if item is None else item
                ),
                key=key,
            )
            if selected is not None:
                mapping[value] = selected
    data[output_column] = series.map(mapping)
    return output_column


def _custom_target_template(data, variables, mapping, method):
    margin_method = method != "poststratification"
    if margin_method:
        rows = []
        for variable in variables:
            column = mapping[variable]
            counts = data[column].value_counts(dropna=False)
            for category, count in counts.items():
                rows.append(
                    {
                        "variable": variable,
                        "category": str(category),
                        "target_percent": float(count / len(data) * 100),
                    }
                )
        return pd.DataFrame(rows)

    columns = [mapping[variable] for variable in variables]
    joint = (
        data[columns]
        .groupby(columns, dropna=False, observed=True)
        .size()
        .rename("sample_count")
        .reset_index()
    )
    joint["target_percent"] = joint["sample_count"] / len(data) * 100
    return joint.rename(
        columns={mapping[variable]: variable for variable in variables}
    ).drop(columns="sample_count")


def _validate_target_percentages(target, dimensions):
    if target.empty or target[dimensions].isna().any().any():
        raise ValueError("Targets must contain categories and percentages.")
    target["target_percent"] = pd.to_numeric(
        target["target_percent"], errors="raise"
    )
    if (
        target["target_percent"].isna().any()
        or (target["target_percent"] < 0).any()
        or abs(target["target_percent"].sum() - 100) > 0.01
    ):
        raise ValueError("Target percentages must be non-negative and sum to 100%.")
    if target.duplicated(dimensions).any():
        raise ValueError("Target tables cannot contain duplicate category rows.")
    return target


def _targets_from_margin_table(table, variables, mapping, active_data):
    required = {"variable", "category", "target_percent"}
    if not required.issubset(table.columns):
        raise ValueError(
            "A margin target table needs variable, category, target_percent."
        )
    unknown = set(table["variable"].dropna().astype(str)) - set(variables)
    if unknown:
        raise ValueError(
            "Target rows include unselected variables: "
            + ", ".join(sorted(unknown))
        )

    targets = {}
    for variable in variables:
        rows = table.loc[table["variable"].astype(str) == variable].copy()
        if rows.empty:
            raise ValueError(f"No target rows were provided for '{variable}'.")
        rows["category"] = rows["category"].map(category_label)
        if rows["category"].duplicated().any():
            raise ValueError(f"Duplicate target categories for '{variable}'.")
        rows["target_percent"] = pd.to_numeric(
            rows["target_percent"], errors="raise"
        )
        observed = set(active_data[mapping[variable]].astype(str).unique())
        supplied = set(rows["category"])
        if observed != supplied:
            raise ValueError(
                f"Categories for '{variable}' must match the included sample "
                "exactly. Use '(Missing)' for missing responses."
            )
        total = rows["target_percent"].sum()
        if (
            rows["target_percent"].isna().any()
            or (rows["target_percent"] < 0).any()
            or abs(total - 100) > 0.01
        ):
            raise ValueError(
                f"Targets for '{variable}' must be non-negative and sum to 100%."
            )
        targets[mapping[variable]] = dict(
            zip(rows["category"], rows["target_percent"].astype(float))
        )
    return targets


def _joint_target_from_table(table, variables, mapping):
    required = set(variables) | {"target_percent"}
    if not required.issubset(table.columns):
        raise ValueError(
            "A joint target table needs one column per selected variable, "
            "plus target_percent."
        )
    target = table[list(variables) + ["target_percent"]].copy()
    for variable in variables:
        target[variable] = target[variable].map(category_label)
    target = target.rename(
        columns={variable: mapping[variable] for variable in variables}
    )
    return _validate_target_percentages(
        target, [mapping[variable] for variable in variables]
    )


def _read_custom_target_csv(uploaded_csv, variables, mapping, active_data, method):
    table = pd.read_csv(uploaded_csv, dtype=str)
    table.columns = [column.strip() for column in table.columns]
    if method == "poststratification":
        return {}, _joint_target_from_table(table, variables, mapping)
    return (
        _targets_from_margin_table(table, variables, mapping, active_data),
        None,
    )


def _missing_joint_cells(data, target, dimensions):
    positive = target.loc[target["target_percent"] > 0, dimensions]
    observed = data[dimensions].drop_duplicates()
    return positive.merge(
        observed, on=dimensions, how="left", indicator=True
    ).loc[lambda rows: rows["_merge"] == "left_only", dimensions]


def _restrict_joint_target(data, target, dimensions):
    observed = data[dimensions].drop_duplicates()
    unmatched_sample = observed.merge(
        target[dimensions], on=dimensions, how="left", indicator=True
    )
    if (unmatched_sample["_merge"] == "left_only").any():
        raise ValueError(
            "The target table does not include every observed sample cell."
        )
    restricted = target.merge(
        observed, on=dimensions, how="inner", validate="many_to_one"
    ).copy()
    total = restricted["target_percent"].sum()
    if total <= 0:
        raise ValueError("No positive target remains for observed joint cells.")
    excluded = _missing_joint_cells(data, target, dimensions)
    restricted["target_percent"] = restricted["target_percent"] / total * 100
    return restricted, excluded


def _result_signature(
    file_key,
    method,
    source,
    variables,
    targets,
    target_data,
    options,
):
    payload = {
        "file": file_key,
        "method": method,
        "source": source,
        "variables": variables,
        "targets": targets,
        "target_data": (
            target_data.to_json(orient="split", date_format="iso")
            if target_data is not None
            else None
        ),
        "options": options,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


st.title("Survey weighting and visualization")
st.caption(
    "Map survey variables to credible population targets, fit survey weights, "
    "inspect diagnostics, and export the results."
)

with st.sidebar:
    st.header("Survey data")
    uploaded_file = st.file_uploader(
        "Upload an SPSS file",
        type=["sav"],
        help="SPSS variable labels, value labels, and missing-value definitions are read.",
    )

if uploaded_file is None:
    st.info("Upload an SPSS `.sav` file to configure the weighting run.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_key = hashlib.sha256(file_bytes).hexdigest()[:16]
try:
    df, metadata = read_sav_bytes(file_bytes)
    analysis_data, missing_masks = prepare_analysis_data(df, metadata)
except Exception as exc:
    st.error(f"Could not read the SPSS file: {exc}")
    st.stop()

st.success(f"Loaded **{len(df):,} respondents** and **{len(df.columns):,} variables**.")
with st.expander("SPSS metadata and missing values"):
    value_label_counts = {
        variable: len(labels)
        for variable, labels in (metadata.variable_value_labels or {}).items()
    }
    missing_definitions = metadata.missing_ranges or {}
    metadata_table = pd.DataFrame(
        [
            {
                "Variable": variable,
                "SPSS variable label": (metadata.column_names_to_labels or {}).get(
                    variable, ""
                ),
                "Value-label count": value_label_counts.get(variable, 0),
                "User-missing definitions": len(
                    missing_definitions.get(variable, [])
                ),
                "Missing observations": int(missing_masks[variable].sum()),
            }
            for variable in df.columns
        ]
    )
    st.dataframe(metadata_table, hide_index=True)
    st.caption(
        "User-missing values are excluded from calculations only when you choose "
        "to exclude incomplete respondents; their original values and definitions "
        "remain in the exported SPSS file."
    )

method_labels = {
    "Post-stratification (joint cells)": "poststratification",
    "Raking / RIM (IPF)": "raking",
    "Linear / GREG calibration (Samplics)": "linear",
    "Logit calibration (bounded)": "logit",
}
with st.sidebar:
    st.header("Weighting configuration")
    method_label = st.selectbox(
        "Weighting method",
        list(method_labels),
        key="weighting_method",
    )
    method = method_labels[method_label]
    source = st.selectbox(
        "Benchmark source",
        [
            "Latest Stats SA population estimates",
            "Enter target percentages",
            "Upload target CSV",
        ],
        key="benchmark_source",
        help=(
            "Stats SA supplies age, sex, and national population-group benchmarks. "
            "It does not publish household or personal income margins in this table."
        ),
    )

if method in {"linear", "logit"}:
    st.warning(
        "Calibration can match marginal controls but does not guarantee a good "
        "joint distribution. Samplics is archived and no longer maintained; "
        "the requested linear/GREG estimator uses its calibration API."
    )

weight_data = pd.DataFrame(index=analysis_data.index)
weight_variables = []
display_names = {}
source_variables = []
source_to_internal = {}
target_data = None
targets = {}
benchmark_population_total = None
benchmark_year = None
geography = None
configured = False

if source == "Latest Stats SA population estimates":
    try:
        (
            population,
            benchmark_year,
            benchmark_url,
            archive_used,
            benchmark_pdf,
        ) = load_benchmarks()
    except Exception as exc:
        st.error(f"Could not load Stats SA benchmarks: {exc}")
        st.info("Use entered or uploaded targets if Stats SA is unavailable.")
        st.stop()

    if archive_used:
        st.warning(
            f"Stats SA could not be reached. Using the archived {benchmark_year} "
            "official benchmark; check whether a newer release is available."
        )
    else:
        st.caption(
            f"Using the latest Stats SA population estimate found ({benchmark_year}); "
            "the source is cached for up to 24 hours."
        )
    st.markdown(
        f"Source: [Stats SA Mid-year population estimates, {benchmark_year}]"
        f"({benchmark_url})"
    )
    download_col, report_col = st.columns(2)
    with download_col:
        st.download_button(
            "Download benchmark CSV",
            population.to_csv(index=False).encode("utf-8"),
            file_name=f"stats_sa_population_{benchmark_year}.csv",
            mime="text/csv",
        )
    with report_col:
        st.download_button(
            "Download official PDF",
            benchmark_pdf,
            file_name=f"stats_sa_population_{benchmark_year}.pdf",
            mime="application/pdf",
        )

    geography_options = ["South Africa", *PROVINCES]
    with st.sidebar:
        geography = st.selectbox("Benchmark geography", geography_options)
        available_dimensions = [
            (label, dimension)
            for label, dimension in DIMENSIONS.items()
            if geography == "South Africa"
            or dimension not in {"population_group", "province"}
        ]
        default_dimensions = [
            label
            for label, _ in available_dimensions
            if label in {"Age group", "Sex", "Race / population group"}
        ]
        selected_dimension_labels = st.multiselect(
            "Demographic dimensions",
            options=[label for label, _ in available_dimensions],
            default=default_dimensions,
        )
    selected_dimensions = [
        dimension
        for label, dimension in available_dimensions
        if label in selected_dimension_labels
    ]
    if not selected_dimensions:
        st.info("Select at least one Stats SA dimension.")
    else:
        try:
            target_data = stats_sa_targets(
                population, geography, selected_dimensions
            ).copy()
            target_data["target_percent"] = target_data["target_percent"].astype(float)
            for dimension in selected_dimensions:
                target_data[dimension] = target_data[dimension].astype(str)
                variable_label = next(
                    label for label, key in DIMENSIONS.items() if key == dimension
                )
                suggestion = suggested_dataset_column(
                    df.columns.tolist(),
                    dimension,
                    metadata.column_names_to_labels or {},
                )
                options = [None, *df.columns.tolist()]
                default_index = options.index(suggestion) if suggestion in options else 0
                with st.sidebar:
                    variable = st.selectbox(
                        f"Survey variable for {variable_label}",
                        options=options,
                        index=default_index,
                        format_func=lambda item: (
                            "Select a variable"
                            if item is None
                            else _variable_label(item, metadata)
                        ),
                        key=f"stats_variable_{file_key}_{dimension}",
                    )
                if variable is None:
                    continue
                if variable in source_variables:
                    st.error(
                        f"'{variable}' is mapped more than once. Select a different "
                        "survey variable for each benchmark dimension."
                    )
                    continue
                source_variables.append(variable)
                internal = _map_stats_dimension(
                    analysis_data,
                    dimension,
                    variable,
                    target_data[dimension].unique().tolist(),
                    metadata,
                    file_key,
                )
                source_to_internal[variable] = internal
                weight_variables.append(internal)
                display_names[internal] = variable_label

            if len(weight_variables) == len(selected_dimensions):
                target_data = target_data.rename(
                    columns={
                        dimension: f"__benchmark_{dimension}"
                        for dimension in selected_dimensions
                    }
                )
                weight_data = analysis_data[weight_variables].copy()
                benchmark_population_total = float(target_data["population"].sum())
                if method == "poststratification":
                    configured = True
                else:
                    targets = raking_targets_from_target_table(
                        target_data, weight_variables
                    )
                    configured = bool(targets)
            else:
                weight_data = pd.DataFrame(index=analysis_data.index)
                weight_variables = []
                source_variables = []
                source_to_internal = {}
                display_names = {}
        except Exception as exc:
            st.error(f"Could not configure Stats SA targets: {exc}")
            weight_data = pd.DataFrame(index=analysis_data.index)
            weight_variables = []
            source_variables = []
            source_to_internal = {}
            display_names = {}
            configured = False

    st.info(
        "Stats SA targets include age, sex, and national population group. "
        "For household or personal income, select entered or uploaded targets "
        "from a reputable source; do not treat sample proportions as benchmarks."
    )
else:
    with st.sidebar:
        selected_variables = st.multiselect(
            "Survey variables to weight",
            options=df.columns.tolist(),
            format_func=lambda item: _variable_label(item, metadata),
            help="Use categorical or already-grouped variables; group continuous measures first.",
            key="custom_variables",
        )
    source_variables = selected_variables
    if selected_variables:
        weight_data, source_to_internal = make_sample_columns(
            analysis_data, selected_variables, "custom"
        )
        weight_variables = [
            source_to_internal[variable] for variable in selected_variables
        ]
        display_names = {
            source_to_internal[variable]: _variable_label(variable, metadata)
            for variable in selected_variables
        }
    else:
        st.info("Select one or more survey variables.")

if weight_variables:
    if source == "Latest Stats SA population estimates":
        row_missing = weight_data[weight_variables].isna().any(axis=1)
        missing_description = "missing or unmapped"
    else:
        row_missing = analysis_data[source_variables].isna().any(axis=1)
        missing_description = "missing"

    missing_row_count = int(row_missing.sum())
    if missing_row_count:
        st.warning(
            f"{missing_row_count:,} respondents have {missing_description} "
            "weighting values."
        )
        exclude_incomplete = st.checkbox(
            "Exclude respondents with missing or unmapped weighting values",
            value=True,
            key=f"exclude_incomplete_{file_key}",
            help=(
                "Excluded rows keep a blank final weight. For custom targets, "
                "uncheck this to treat missing responses as a '(Missing)' category."
            ),
        )
    else:
        exclude_incomplete = True
    if source == "Latest Stats SA population estimates" and missing_row_count and not exclude_incomplete:
        st.error(
            "Unmapped or missing responses cannot match Stats SA categories. "
            "Complete the category mapping or exclude those respondents."
        )
        configured = False
    active_mask = ~row_missing if exclude_incomplete else pd.Series(
        True, index=analysis_data.index
    )
    active_data = weight_data.loc[active_mask, weight_variables].copy()
    if active_data.empty:
        st.error("No complete respondents remain for weighting.")
        configured = False
else:
    exclude_incomplete = True
    active_mask = pd.Series(False, index=analysis_data.index)
    active_data = pd.DataFrame(columns=weight_variables, index=analysis_data.index[:0])
    configured = False

if source != "Latest Stats SA population estimates" and weight_variables:
    st.caption(
        "Targets entered here are editable starting values based on the included "
        "sample, not external benchmarks. Replace them with verified target figures."
    )
    if not active_data.empty:
        template = _custom_target_template(
            active_data, source_variables, source_to_internal, method
        )
        if source == "Enter target percentages":
            if len(template) > 500:
                st.error(
                    "This target table has more than 500 rows. Use a target CSV "
                    "instead or group the selected categories."
                )
            else:
                edited = st.data_editor(
                    template,
                    key=f"targets_{file_key}_{method}_{'_'.join(source_variables)}",
                    disabled=[
                        column for column in template.columns
                        if column != "target_percent"
                    ],
                    num_rows="fixed",
                    hide_index=True,
                    column_config={
                        "target_percent": st.column_config.NumberColumn(
                            "Target (%)",
                            min_value=0.0,
                            max_value=100.0,
                            format="%.2f",
                        )
                    },
                )
                try:
                    if method == "poststratification":
                        target_data = _joint_target_from_table(
                            edited, source_variables, source_to_internal
                        )
                        targets = raking_targets_from_target_table(
                            target_data, weight_variables
                        )
                    else:
                        targets = _targets_from_margin_table(
                            edited,
                            source_variables,
                            source_to_internal,
                            active_data,
                        )
                    configured = bool(targets)
                except (ValueError, KeyError) as exc:
                    st.error(f"Target values need attention: {exc}")
                    configured = False
        else:
            st.download_button(
                "Download target CSV template",
                template.to_csv(index=False).encode("utf-8"),
                file_name="survey_target_template.csv",
                mime="text/csv",
            )
            if method == "poststratification":
                st.caption(
                    "Joint target CSV columns: one per selected variable, plus "
                    "`target_percent`."
                )
            else:
                st.caption(
                    "Margin target CSV columns: `variable`, `category`, and "
                    "`target_percent`."
                )
            target_csv = st.file_uploader(
                "Upload target CSV",
                type=["csv"],
                key=f"target_csv_{file_key}_{method}",
            )
            if target_csv is not None:
                try:
                    targets, target_data = _read_custom_target_csv(
                        target_csv,
                        source_variables,
                        source_to_internal,
                        active_data,
                        method,
                    )
                    if method == "poststratification":
                        targets = raking_targets_from_target_table(
                            target_data, weight_variables
                        )
                    configured = bool(target_data is not None) if (
                        method == "poststratification"
                    ) else bool(targets)
                except (ValueError, KeyError, pd.errors.ParserError) as exc:
                    st.error(f"Invalid target CSV: {exc}")
                    configured = False

if source == "Latest Stats SA population estimates" and method == "poststratification":
    if target_data is not None and weight_variables:
        targets = raking_targets_from_target_table(target_data, weight_variables)

effective_targets = targets
effective_target_data = target_data
excluded_categories = {}
use_observed_categories = False
missing_target_categories = {}
missing_joint = pd.DataFrame()
if configured and not active_data.empty:
    if method == "poststratification":
        missing_joint = _missing_joint_cells(
            active_data, target_data, weight_variables
        )
        sample_cells_without_targets = active_data[weight_variables].drop_duplicates().merge(
            target_data[weight_variables],
            on=weight_variables,
            how="left",
            indicator=True,
        )
        if (sample_cells_without_targets["_merge"] == "left_only").any():
            st.error("The joint target table does not cover every observed sample cell.")
            configured = False
    else:
        missing_target_categories = missing_positive_raking_categories(
            active_data, targets
        )

    missing_details = {}
    if missing_target_categories:
        missing_details = {
            display_names.get(column, column): categories
            for column, categories in missing_target_categories.items()
        }
    elif not missing_joint.empty:
        missing_details = {
            "Joint cells": [
                ", ".join(str(row[column]) for column in weight_variables)
                for _, row in missing_joint.iterrows()
            ]
        }

    if missing_details:
        details = "; ".join(
            f"{variable}: {', '.join(map(str, categories))}"
            for variable, categories in missing_details.items()
        )
        st.warning(
            "Positive benchmark categories/cells have no respondents: "
            f"{details}. Weighting cannot create respondents for those targets."
        )
        use_observed_categories = st.checkbox(
            "Use only categories/cells present in my data and rescale their targets",
            value=False,
            key=f"restrict_targets_{file_key}_{method}",
            help=(
                "Use only when the absent categories are outside the intended "
                "survey population, not merely absent from the sample."
            ),
        )
        if not use_observed_categories:
            configured = False
    if configured:
        try:
            if method == "poststratification":
                if use_observed_categories:
                    effective_target_data, excluded = _restrict_joint_target(
                        active_data, target_data, weight_variables
                    )
                    excluded_categories = {
                        "Joint cells": [
                            ", ".join(str(row[column]) for column in weight_variables)
                            for _, row in excluded.iterrows()
                        ]
                    }
                effective_targets = raking_targets_from_target_table(
                    effective_target_data, weight_variables
                )
            else:
                effective_targets, excluded_categories = (
                    restrict_raking_targets_to_observed(active_data, targets)
                )
                if excluded_categories and not use_observed_categories:
                    configured = False
        except ValueError as exc:
            st.error(f"Target categories do not match the sample: {exc}")
            configured = False

with st.sidebar:
    st.header("Output and quality controls")
    weight_column = st.text_input("Output weight variable", value="survey_weight")
    trim_method = st.selectbox(
        "Weight trimming",
        ["None", "Min/max caps", "Percentile caps"],
        help="The final weights are renormalized while respecting active caps.",
    )
    lower_cap = upper_cap = None
    lower_percentile = 1.0
    upper_percentile = 99.0
    if trim_method == "Min/max caps":
        lower_cap = st.number_input(
            "Minimum weight / mean",
            min_value=0.0,
            value=0.1,
            step=0.1,
            help="The cap is applied as a multiple of the requested final mean weight.",
        )
        upper_cap = st.number_input(
            "Maximum weight / mean",
            min_value=0.1,
            value=10.0,
            step=0.5,
        )
    elif trim_method == "Percentile caps":
        lower_percentile = st.number_input(
            "Lower percentile",
            min_value=0.0,
            max_value=49.0,
            value=1.0,
            step=1.0,
        )
        upper_percentile = st.number_input(
            "Upper percentile",
            min_value=51.0,
            max_value=100.0,
            value=99.0,
            step=1.0,
        )
    max_iterations = 500
    tolerance = 1e-8
    if method == "raking":
        max_iterations = st.number_input(
            "Maximum IPF iterations",
            min_value=10,
            max_value=5000,
            value=500,
            step=50,
        )
        tolerance = st.number_input(
            "IPF convergence tolerance",
            min_value=1e-10,
            max_value=1e-3,
            value=1e-8,
            format="%.1e",
        )
    logit_bounds = (0.3, 3.0)
    if method == "logit":
        lower_logit = st.number_input(
            "Logit lower g-factor",
            min_value=0.01,
            max_value=0.99,
            value=0.3,
            step=0.05,
        )
        upper_logit = st.number_input(
            "Logit upper g-factor",
            min_value=1.01,
            max_value=100.0,
            value=3.0,
            step=0.25,
        )
        logit_bounds = (lower_logit, upper_logit)

allow_population_total = (
    source == "Latest Stats SA population estimates"
    and benchmark_population_total is not None
    and not use_observed_categories
)
normalization_options = ["Sample size (n)"]
if allow_population_total:
    normalization_options.append("Benchmark population total (N)")
with st.sidebar:
    normalization = st.selectbox(
        "Normalize weights to",
        normalization_options,
        help=(
            "Population-total normalization is available only when the complete "
            "Stats SA benchmark categories are retained."
        ),
        key="normalization",
    )

valid_weight_name = bool(
    re.fullmatch(r"[A-Za-z@#$][A-Za-z0-9_@#$\.]{0,63}", weight_column)
)
if not valid_weight_name:
    st.error(
        "Use an SPSS-compatible variable name: start with a letter or @/#/$, "
        "then up to 63 letters, digits, underscores, @, #, $, or periods."
    )
elif weight_column in df.columns:
    st.error("Choose an output weight variable that does not already exist.")
    valid_weight_name = False

settings = {
    "Method": method_label,
    "Target source": source,
    "Benchmark year": benchmark_year or "",
    "Geography": geography or "",
    "Weight variable": weight_column,
    "Normalization": normalization,
    "Trimming": trim_method,
    "Included respondents": int(len(active_data)),
    "Excluded respondents": int(len(df) - len(active_data)),
}
options_signature = {
    "weight_column": weight_column,
    "normalization": normalization,
    "trim_method": trim_method,
    "lower_cap": lower_cap,
    "upper_cap": upper_cap,
    "lower_percentile": lower_percentile,
    "upper_percentile": upper_percentile,
    "max_iterations": max_iterations,
    "tolerance": tolerance,
    "logit_bounds": logit_bounds,
    "active_rows_hash": hashlib.sha256(
        active_mask.to_numpy(dtype="uint8").tobytes()
    ).hexdigest(),
    "restrict": use_observed_categories,
}
current_result_key = _result_signature(
    file_key,
    method,
    source,
    source_variables,
    effective_targets,
    effective_target_data,
    options_signature,
)
stored_result = st.session_state.get("weighting_result")
if stored_result is not None and stored_result.get("key") != current_result_key:
    st.session_state.pop("weighting_result", None)

calculate = st.button(
    "Calculate weights",
    type="primary",
    disabled=not configured or not valid_weight_name,
)
if calculate:
    st.session_state.pop("weighting_result", None)
    try:
        with st.spinner("Fitting weights and preparing diagnostics..."):
            ipf_converged = None
            if method == "poststratification":
                raw_weights = poststratification_weights(
                    active_data,
                    effective_target_data,
                    weight_variables,
                )["weight"]
            elif method == "raking":
                raw_weights, ipf_converged = rake_weights(
                    active_data,
                    effective_targets,
                    max_iterations=int(max_iterations),
                    tolerance=float(tolerance),
                )
            else:
                estimator = (
                    "Linear / GREG (Samplics)"
                    if method == "linear"
                    else "Logit (bounded)"
                )
                raw_weights = calibration_weights(
                    active_data,
                    effective_targets,
                    estimator,
                    logit_bounds=logit_bounds,
                )

            target_sum = (
                benchmark_population_total
                if normalization == "Benchmark population total (N)"
                else float(len(active_data))
            )
            final_weights, trim_info = trim_and_normalize_weights(
                raw_weights,
                float(target_sum),
                trim_method=trim_method,
                lower_bound=lower_cap,
                upper_bound=upper_cap,
                lower_percentile=float(lower_percentile),
                upper_percentile=float(upper_percentile),
            )
            full_weights = pd.Series(
                float("nan"), index=df.index, name=weight_column
            )
            full_weights.loc[active_data.index] = final_weights
            weighted_data = df.copy()
            weighted_data[weight_column] = full_weights

            target_label = (
                "Stats SA target (%)"
                if source == "Latest Stats SA population estimates"
                else "Target (%)"
            )
            comparison = comparison_table(
                active_data,
                final_weights,
                effective_targets,
                labels=display_names,
                target_label=target_label,
            )
            diagnostics = weighting_diagnostics(final_weights)
            max_margin_error = (
                float(comparison["Margin error (pp)"].abs().max())
                if not comparison.empty
                else 0.0
            )
            diagnostics["maximum_margin_error_pp"] = max_margin_error
            diagnostics["trimmed_weight_count"] = int(
                trim_info["trimmed_count"]
            )
            diagnostics["ipf_converged"] = (
                bool(ipf_converged) if ipf_converged is not None else True
            )

            final_settings = {
                **settings,
                "Benchmark total (N)": (
                    float(target_sum)
                    if normalization == "Benchmark population total (N)"
                    else ""
                ),
                "IPF converged": (
                    bool(ipf_converged) if ipf_converged is not None else ""
                ),
                "Trim lower bound": trim_info["lower_bound"],
                "Trim upper bound": trim_info["upper_bound"],
                "Trimmed observations": trim_info["trimmed_count"],
            }
            if trim_method == "None":
                final_settings["Trim lower bound"] = ""
                final_settings["Trim upper bound"] = ""
            sav_bytes = weighted_sav_bytes(
                weighted_data, metadata, weight_column
            )
            excel_bytes = excel_report_bytes(
                diagnostics, comparison, final_settings
            )
            csv_bytes = comparison.to_csv(index=False).encode("utf-8")
            diagnostics_csv = pd.DataFrame(
                [{"Metric": key, "Value": value} for key, value in diagnostics.items()]
            ).to_csv(index=False).encode("utf-8")
            st.session_state["weighting_result"] = {
                "key": current_result_key,
                "weighted_data": weighted_data,
                "weights": final_weights,
                "comparison": comparison,
                "diagnostics": diagnostics,
                "settings": final_settings,
                "sav_bytes": sav_bytes,
                "excel_bytes": excel_bytes,
                "csv_bytes": csv_bytes,
                "diagnostics_csv": diagnostics_csv,
                "ipf_converged": ipf_converged,
                "excluded_categories": excluded_categories,
            }
    except Exception as exc:
        st.session_state.pop("weighting_result", None)
        st.error(f"Could not calculate weights: {exc}")

result = st.session_state.get("weighting_result")
if result is not None and result.get("key") == current_result_key:
    diagnostics = result["diagnostics"]
    comparison = result["comparison"]
    st.success("Weighting completed.")
    if result["excluded_categories"]:
        st.warning(
            "Restricted-category weighting was used. Missing benchmark "
            "categories/cells were omitted and the remaining targets rescaled. "
            "These weights do not represent the full benchmark population."
        )
    if result["ipf_converged"] is False:
        st.warning(
            "IPF reached its iteration limit before meeting the convergence "
            "criterion. Review the residual margin chart before using the weights."
        )
    if diagnostics["maximum_margin_error_pp"] > 0.01:
        st.warning(
            "The final margin error exceeds 0.01 percentage points. Trimming, "
            "restricted targets, or incompatible controls can prevent exact matching."
        )

    metrics = [
        ("Design effect (DEFF)", f"{diagnostics['design_effect']:.3f}"),
        ("Effective sample size", f"{diagnostics['effective_sample_size']:,.0f}"),
        (
            "Weighting efficiency",
            f"{diagnostics['weighting_efficiency_percent']:.1f}%",
        ),
        ("Minimum weight", f"{diagnostics['minimum_weight']:.4f}"),
        ("Maximum weight", f"{diagnostics['maximum_weight']:.4f}"),
        ("Mean / median weight", f"{diagnostics['mean_weight']:.4f} / {diagnostics['median_weight']:.4f}"),
    ]
    with st.container(horizontal=True):
        for label, value in metrics:
            st.metric(label, value, border=True)
    st.caption(
        f"Final weight sum: {diagnostics['weight_sum']:,.4f} · "
        f"Maximum absolute margin error: "
        f"{diagnostics['maximum_margin_error_pp']:.4f} percentage points · "
        f"Trimmed observations: {diagnostics['trimmed_weight_count']:,}"
    )

    chart_col, distribution_col = st.columns(2)
    with chart_col:
        with st.container(border=True):
            st.subheader("Sample and target distributions")
            st.plotly_chart(
                comparison_figure(comparison),
                width="stretch",
                key="comparison_chart",
            )
    with distribution_col:
        with st.container(border=True):
            st.subheader("Weight distribution")
            st.plotly_chart(
                weight_distribution_figure(result["weights"]),
                width="stretch",
                key="weight_distribution_chart",
            )
    with st.container(border=True):
        st.subheader("Convergence and margin error")
        st.caption(
            "Residuals are final weighted shares minus target shares, in "
            "percentage points. Trimming can change the fitted margins."
        )
        st.plotly_chart(
            margin_error_figure(comparison),
            width="stretch",
            key="margin_error_chart",
        )
    with st.expander("Target comparison table"):
        st.dataframe(comparison, hide_index=True)

    download_cols = st.columns(4)
    with download_cols[0]:
        st.download_button(
            "Download weighted SPSS file",
            result["sav_bytes"],
            file_name="weighted_survey.sav",
            mime="application/octet-stream",
        )
    with download_cols[1]:
        st.download_button(
            "Download Excel report",
            result["excel_bytes"],
            file_name="weighting_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with download_cols[2]:
        st.download_button(
            "Download target comparison CSV",
            result["csv_bytes"],
            file_name="target_comparison.csv",
            mime="text/csv",
        )
    with download_cols[3]:
        st.download_button(
            "Download diagnostics CSV",
            result["diagnostics_csv"],
            file_name="weighting_diagnostics.csv",
            mime="text/csv",
        )
    st.caption(
        "Weighting aligns selected distributions; it does not by itself correct "
        "coverage or nonresponse bias, and it does not calculate complex-survey "
        "standard errors."
    )
