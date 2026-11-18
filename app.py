import hashlib
import io
import numbers
import tempfile
from pathlib import Path

import pandas as pd
import pyreadstat
import streamlit as st
import weightipy as wp

from weighting import (
    DIMENSIONS,
    PROVINCES,
    age_value_to_group,
    load_stats_sa_population as load_stats_sa_benchmarks,
    missing_positive_raking_categories,
    poststratification_weights,
    raking_targets_from_target_table,
    restrict_raking_targets_to_observed,
    stats_sa_targets,
)


st.set_page_config(
    page_title="Automated survey weighting",
    page_icon="🇿🇦",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: #eef3f8;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #d8e1ec;
    }
    [data-testid="stHeader"] {
        background: #173a69;
    }
    .app-header {
        background: #173a69;
        color: white;
        padding: 1rem 1.25rem 0.85rem 1.25rem;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        border-bottom: 1px solid #0f2d54;
    }
    .app-header h1 {
        color: white;
        font-size: 1.15rem;
        margin: 0;
        font-weight: 700;
    }
    .header-subtitle {
        color: #b7cde8;
        margin: 0.1rem 0 0;
        font-size: 0.92rem;
    }
    .ready-status {
        background: #dce6f2;
        color: #173a69;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 700;
        flex-shrink: 0;
        white-space: nowrap;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe3ed;
        border-radius: 8px;
        padding: 0.65rem;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #dbe3ed;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl="1d", max_entries=1)
def load_stats_sa_population(cache_schema_version):
    return load_stats_sa_benchmarks()


def category_label(value):
    if pd.isna(value):
        return "(Missing)"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if float(value).is_integer():
            return str(int(value))
    if isinstance(value, str) and "." in value:
        try:
            numeric_value = float(value)
            if numeric_value.is_integer():
                return str(int(numeric_value))
        except ValueError:
            pass
    return str(value)


def labeled_category_label(variable, value, variable_value_labels):
    labels = variable_value_labels.get(variable, {})
    value_label = labels.get(value)
    if value_label is None:
        value_label = next(
            (
                label
                for code, label in labels.items()
                if category_label(code) == category_label(value)
            ),
            None,
        )
    if value_label is None:
        return category_label(value)

    label = category_label(value_label)
    code = category_label(value)
    if normalized_label(label) == normalized_label(code):
        return label
    return f"{label} ({code})"


def normalized_label(value):
    return "".join(character for character in category_label(value).lower() if character.isalnum())


def suggested_benchmark_category(dimension, value, categories):
    normalized = normalized_label(value)
    for category in categories:
        if normalized_label(category) == normalized:
            return category

    aliases = {
        "sex": {
            "m": "Male",
            "man": "Male",
            "men": "Male",
            "f": "Female",
            "woman": "Female",
            "women": "Female",
        },
        "population_group": {
            "african": "Black African",
            "black": "Black African",
            "indian": "Indian/Asian",
            "asian": "Indian/Asian",
        },
    }
    suggestion = aliases.get(dimension, {}).get(normalized)
    return suggestion if suggestion in categories else None


def suggested_dataset_column(columns, dimension, variable_labels):
    indicators = {
        "age_group": ("age", "agegroup", "ageband"),
        "sex": ("sex", "gender"),
        "population_group": ("populationgroup", "race", "ethnicity"),
    }
    matches = []
    for column in columns:
        searchable = normalized_label(
            f"{column} {variable_labels.get(column, '')}"
        )
        score = max(
            (
                3 if searchable == indicator
                else 2 if searchable.startswith(indicator)
                else 1 if indicator in searchable
                else 0
                for indicator in indicators[dimension]
            ),
            default=0,
        )
        if score:
            matches.append((score, column))
    if not matches:
        return None
    best_score = max(score for score, _ in matches)
    best_matches = [column for score, column in matches if score == best_score]
    return best_matches[0] if len(best_matches) == 1 else None


def add_stats_benchmark_variable(data, dimension, variable, categories):
    output_column = f"__benchmark_{dimension}"
    series = data[variable]
    is_continuous_age = (
        dimension == "age_group"
        and pd.api.types.is_numeric_dtype(series)
        and series.nunique(dropna=True) > 15
    )

    if is_continuous_age:
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
    try:
        values = sorted(values)
    except TypeError:
        values = sorted(values, key=str)

    mapping = {}
    with st.expander(f"Map categories for **{variable}** to Stats SA", expanded=True):
        st.caption(
            "Check every mapping. Numeric codes such as 1/2 are not guessed."
        )
        for index, value in enumerate(values):
            suggestion = suggested_benchmark_category(
                dimension, value, categories
            )
            options = [None] + categories
            selected_index = (
                categories.index(suggestion) + 1 if suggestion in categories else 0
            )
            selected = st.selectbox(
                f"`{category_label(value)}`",
                options=options,
                index=selected_index,
                format_func=lambda item: "Select a category" if item is None else item,
                key=f"map_{dimension}_{variable}_{index}",
            )
            if selected is not None:
                mapping[value] = selected

    data[output_column] = series.map(mapping)
    return output_column


def make_sample_columns(data, variables, prefix):
    working = pd.DataFrame(index=data.index)
    mapping = {}
    for index, variable in enumerate(variables):
        output_column = f"__{prefix}_{index}"
        working[output_column] = data[variable].map(category_label)
        mapping[variable] = output_column
    return working, mapping


def make_manual_raking_targets(
    data, variables, mapping, variable_value_labels, column_names_to_labels
):
    targets = {}
    comparison_targets = {}
    st.caption(
        "Starting values are this file's sample percentages, not population "
        "benchmarks. Replace them with verified target values."
    )
    for variable in variables:
        output_column = mapping[variable]
        counts = data[output_column].value_counts(dropna=False)
        denominator = counts.sum()
        variable_targets = {}

        variable_label = column_names_to_labels.get(variable) or variable
        st.subheader(f"Targets for {variable_label}")
        if len(counts) > 100:
            st.warning(
                f"'{variable}' has {len(counts)} categories. Group or recode it "
                "before entering targets."
            )
            return {}, {}

        for index, (category, count) in enumerate(counts.items()):
            default_percent = float(count / denominator * 100)
            category_label_text = labeled_category_label(
                variable, category, variable_value_labels
            )
            col_a, col_b = st.columns([2, 1])
            with col_a:
                target_percent = st.number_input(
                    f"Target % for {category_label_text}",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_percent,
                    step=0.5,
                    key=f"manual_{variable}_{index}",
                )
            with col_b:
                st.metric("Sample count", f"{count:,}")
            variable_targets[category] = target_percent

        total_percent = sum(variable_targets.values())
        if abs(total_percent - 100) > 0.01:
            st.error(
                f"Targets for '{variable}' total {total_percent:.2f}%; they "
                "must sum to 100%."
            )
        targets[output_column] = variable_targets
        comparison_targets[output_column] = pd.Series(variable_targets)
    return targets, comparison_targets


def make_target_template(data, variables, method):
    working, mapping = make_sample_columns(data, variables, "template")
    if method == "Raking":
        rows = []
        for variable in variables:
            output_column = mapping[variable]
            counts = working[output_column].value_counts(dropna=False)
            for category, count in counts.items():
                rows.append(
                    {
                        "variable": variable,
                        "category": category,
                        "target_percent": count / len(working) * 100,
                    }
                )
        return pd.DataFrame(rows)

    columns = [mapping[variable] for variable in variables]
    joint = (
        working.groupby(columns, dropna=False, observed=True)
        .size()
        .rename("sample_count")
        .reset_index()
    )
    joint["target_percent"] = joint["sample_count"] / len(working) * 100
    return joint.rename(
        columns={mapping[variable]: variable for variable in variables}
    ).drop(columns="sample_count")


def read_custom_target_csv(uploaded_csv, method, variables, mapping, data):
    target_file = pd.read_csv(uploaded_csv, dtype=str)
    target_file.columns = [column.strip() for column in target_file.columns]

    if method == "Raking":
        required = {"variable", "category", "target_percent"}
        if not required.issubset(target_file.columns):
            raise ValueError(
                "Raking CSV needs columns: variable, category, target_percent."
            )
        targets = {}
        comparisons = {}
        for variable in variables:
            rows = target_file.loc[target_file["variable"] == variable].copy()
            if rows.empty:
                raise ValueError(f"No target rows were provided for '{variable}'.")
            rows["category"] = rows["category"].map(category_label)
            rows["target_percent"] = pd.to_numeric(
                rows["target_percent"], errors="raise"
            )
            if rows["category"].duplicated().any():
                raise ValueError(f"Duplicate target categories for '{variable}'.")
            observed = set(data[mapping[variable]].unique())
            supplied = set(rows["category"])
            if observed != supplied:
                raise ValueError(
                    f"CSV categories for '{variable}' must match every observed "
                    "category exactly. Use '(Missing)' for missing responses."
                )
            total = rows["target_percent"].sum()
            if (rows["target_percent"] < 0).any() or abs(total - 100) > 0.01:
                raise ValueError(
                    f"Target percentages for '{variable}' must be non-negative "
                    "and sum to 100."
                )
            targets[mapping[variable]] = dict(
                zip(rows["category"], rows["target_percent"])
            )
            comparisons[mapping[variable]] = pd.Series(
                targets[mapping[variable]]
            )
        return targets, comparisons, None

    required = set(variables) | {"target_percent"}
    if not required.issubset(target_file.columns):
        raise ValueError(
            "Post-stratification CSV needs one column per selected survey "
            "variable, plus target_percent."
        )
    target = target_file[list(variables) + ["target_percent"]].copy()
    for variable in variables:
        target[variable] = target[variable].map(category_label)
    target["target_percent"] = pd.to_numeric(
        target["target_percent"], errors="raise"
    )
    target = target.rename(
        columns={variable: mapping[variable] for variable in variables}
    )
    return {}, {}, target


st.markdown(
    """
    <div class="app-header">
        <div>
            <h1>Survey Weighting Engine Pro</h1>
            <p class="header-subtitle">Multi-method calibration suite</p>
        </div>
        <span class="ready-status">● System Ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Weighting configuration")
st.sidebar.caption("Data source (.sav)")
uploaded_file = st.sidebar.file_uploader("Upload an SPSS .sav file", type=["sav"])
if uploaded_file is None:
    st.info("Upload an SPSS `.sav` file in Weighting configuration to begin.")
    st.stop()

try:
    df, meta = pyreadstat.read_sav(io.BytesIO(uploaded_file.getvalue()))
except Exception as exc:
    st.error(f"Could not read the SPSS file: {exc}")
    st.stop()

file_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:10]
st.sidebar.success(f"{uploaded_file.name}\n\n{len(df):,} rows")
st.subheader("Calibration dashboard")

method = st.sidebar.selectbox(
    "Weighting method",
    ["Raking", "Post-stratification"],
    format_func=lambda value: {
        "Raking": "RIM weighting (Raking)",
        "Post-stratification": "Joint post-stratification",
    }[value],
    help=(
        "Raking matches separate target margins. Post-stratification matches "
        "the joint distribution of the selected categories."
    ),
)
source = st.sidebar.selectbox(
    "Benchmark / target source",
    [
        "Latest Stats SA population estimates",
        "Enter target percentages",
        "Upload target CSV",
    ],
)

configured = False
weight_data = df.copy()
weight_variables = []
targets = {}
target_data = None
comparison_targets = {}
display_names = {}

if source == "Latest Stats SA population estimates":
    try:
        (
            population,
            benchmark_year,
            benchmark_url,
            archive_used,
            benchmark_pdf,
        ) = load_stats_sa_population(cache_schema_version=2)
    except Exception as exc:
        st.error(f"Could not load current Stats SA benchmarks: {exc}")
        st.info("You can switch to manually entered targets or a target CSV.")
        st.stop()

    if archive_used:
        st.warning(
            f"Stats SA could not be reached. Using the archived official "
            f"population estimates from {benchmark_year}; these may be older "
            "than the latest release."
        )
    else:
        st.caption(
            f"Using the latest publication found ({benchmark_year}). "
            "The app fetches and caches it for up to 24 hours."
        )
    st.markdown(
        f"Source: [Stats SA Mid-year population estimates, {benchmark_year}]"
        f"({benchmark_url})"
    )
    with st.container(horizontal=True):
        st.download_button(
            "Download benchmark data (.csv)",
            population.to_csv(index=False).encode("utf-8"),
            file_name=f"stats_sa_population_{benchmark_year}.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            wrap=True,
        )
        st.download_button(
            "Download official report (.pdf)",
            benchmark_pdf,
            file_name=f"stats_sa_population_{benchmark_year}.pdf",
            mime="application/pdf",
            icon=":material/download:",
            on_click="ignore",
            wrap=True,
        )

    geography_options = ["South Africa"] + PROVINCES
    geography = st.sidebar.selectbox("Benchmark geography", geography_options)
    dimensions = ["age_group", "sex"]
    if geography == "South Africa":
        dimensions.append("population_group")
    selected_dimension_labels = st.sidebar.multiselect(
        "Demographic dimensions to weight",
        options=[label for label, key in DIMENSIONS.items() if key in dimensions],
        default=["Age group", "Sex"],
    )
    selected_dimensions = [
        DIMENSIONS[label] for label in selected_dimension_labels
    ]
    st.sidebar.caption("Map each benchmark dimension to a survey variable.")

    if selected_dimensions:
        try:
            target_data = stats_sa_targets(
                population, geography, selected_dimensions
            )
            mapping_complete = True
            variable_labels = meta.column_names_to_labels
            for dimension in selected_dimensions:
                label = next(
                    label for label, key in DIMENSIONS.items() if key == dimension
                )
                suggested_column = suggested_dataset_column(
                    df.columns.tolist(), dimension, variable_labels
                )
                column_options = [None] + df.columns.tolist()
                variable = st.sidebar.selectbox(
                    f"Dataset column for {label}",
                    options=column_options,
                    index=(
                        column_options.index(suggested_column)
                        if suggested_column in column_options
                        else 0
                    ),
                    format_func=lambda item: (
                        "Select a dataset column" if item is None else item
                    ),
                    key=f"stats_column_{dimension}",
                )
                if variable is None:
                    mapping_complete = False
                    continue
                categories = target_data[dimension].astype(str).unique().tolist()
                internal_column = add_stats_benchmark_variable(
                    weight_data, dimension, variable, categories
                )
                weight_variables.append(internal_column)
                display_names[internal_column] = label
                target_data[dimension] = target_data[dimension].astype(str)

            if mapping_complete:
                target_data = target_data.rename(
                    columns={
                        dimension: f"__benchmark_{dimension}"
                        for dimension in selected_dimensions
                    }
                )
                weight_variables = [
                    f"__benchmark_{dimension}" for dimension in selected_dimensions
                ]
                if weight_data[weight_variables].isna().any().any():
                    st.error(
                        "Some responses are missing or unmapped. Resolve the "
                        "category mappings before weighting."
                    )
                else:
                    if method == "Raking":
                        targets = raking_targets_from_target_table(
                            target_data, weight_variables
                        )
                        comparison_targets = {
                            column: pd.Series(distribution)
                            for column, distribution in targets.items()
                        }
                    configured = True
        except Exception as exc:
            st.error(f"Could not configure Stats SA targets: {exc}")
    st.warning(
        "Check that the selected Stats SA population matches your survey. "
        "The published age groups are five-year bands (including 15–19); "
        "they cannot be split accurately into custom ranges such as 18–24."
    )
else:
    variables = st.sidebar.multiselect(
        "Survey questions or categories to weight",
        options=df.columns.tolist(),
        help=(
            "Categorical questions can be weighted when you have a valid target "
            "distribution. Group continuous measures before weighting."
        ),
    )
    if variables:
        weight_data, mapping = make_sample_columns(df, variables, "custom")
        weight_variables = [mapping[variable] for variable in variables]
        display_names = {
            mapping[variable]: variable for variable in variables
        }

        if source == "Enter target percentages" and method == "Raking":
            st.caption(
                "Targets should come from the population or another valid "
                "reference source, not from the sample itself."
            )
            targets, comparison_targets = make_manual_raking_targets(
                weight_data,
                variables,
                mapping,
                meta.variable_value_labels,
                meta.column_names_to_labels,
            )
            configured = bool(targets) and all(
                abs(sum(values.values()) - 100) <= 0.01
                for values in targets.values()
            )
        elif source == "Enter target percentages":
            joint = make_target_template(df, variables, method)
            if len(joint) > 500:
                st.error(
                    "There are more than 500 observed joint cells. Upload a "
                    "target CSV instead of editing them in the app."
                )
            else:
                st.caption(
                    "Edit the target percentages. The starting values are "
                    "sample proportions, not population benchmarks."
                )
                edited = st.data_editor(
                    joint,
                    key=f"joint_targets_{file_key}_{'_'.join(variables)}",
                    disabled=[column for column in joint.columns if column != "target_percent"],
                    num_rows="fixed",
                    width="stretch",
                )
                target_data = edited.rename(
                    columns={variable: mapping[variable] for variable in variables}
                )
                configured = (
                    pd.to_numeric(
                        target_data["target_percent"], errors="coerce"
                    ).notna().all()
                    and abs(
                        pd.to_numeric(target_data["target_percent"]).sum() - 100
                    )
                    <= 0.01
                )
                if not configured:
                    st.error("Joint target percentages must sum to 100%.")
        else:
            template = make_target_template(df, variables, method)
            st.download_button(
                "Download target CSV template",
                template.to_csv(index=False).encode("utf-8"),
                file_name="target_template.csv",
                mime="text/csv",
            )
            st.caption(
                "Raking CSV columns: variable, category, target_percent. "
                "Post-stratification CSV: one column per selected question "
                "plus target_percent."
            )
            target_csv = st.file_uploader(
                "Upload target CSV",
                type=["csv"],
                key=f"target_csv_{file_key}_{method}",
            )
            if target_csv is not None:
                try:
                    targets, comparison_targets, target_data = (
                        read_custom_target_csv(
                            target_csv, method, variables, mapping, weight_data
                        )
                    )
                    if method == "Raking":
                        configured = all(
                            abs(sum(values.values()) - 100) <= 0.01
                            for values in targets.values()
                        )
                    else:
                        configured = True
                except Exception as exc:
                    st.error(f"Invalid target CSV: {exc}")

        if weight_data[weight_variables].isna().any().any():
            st.error("Selected weighting variables contain missing values.")
            configured = False
    else:
        st.info("Select one or more survey questions to continue.")

weight_column = st.sidebar.text_input(
    "Output weight column name", value="survey_weight"
)
if weight_column in df.columns:
    st.error("Choose a weight-column name that does not already exist.")
    configured = False

missing_categories = (
    missing_positive_raking_categories(weight_data, targets)
    if configured and method == "Raking"
    else {}
)
use_observed_categories = False
if missing_categories:
    missing_details = "; ".join(
        f"{display_names.get(column, column)}: {', '.join(categories)}"
        for column, categories in missing_categories.items()
    )
    st.warning(
        "Some positive target categories have no respondents: "
        f"{missing_details}. You can calculate using only categories present "
        "in the sample, but the result will represent that restricted group "
        "rather than the full benchmark population."
    )
    use_observed_categories = st.checkbox(
        "Use only categories present in my data and rescale their targets",
        key="use_observed_raking_categories",
        help=(
            "Enable only when the absent categories are outside the survey's "
            "intended population, not simply missing from the sample."
        ),
    )

if configured and st.button(
    "Calculate weights",
    type="primary",
    disabled=bool(missing_categories) and not use_observed_categories,
):
    effective_raking_targets = None
    excluded_categories = {}
    try:
        with st.spinner("Calculating and checking weights..."):
            if method == "Raking":
                missing_targets = set(weight_variables) - set(targets)
                if missing_targets:
                    raise ValueError(
                        "Raking target distributions are missing for: "
                        + ", ".join(sorted(missing_targets))
                    )
                effective_raking_targets = targets
                if missing_categories:
                    if not use_observed_categories:
                        raise ValueError(
                            "Enable the option to calculate using only "
                            "categories present in the sample."
                        )
                    effective_raking_targets, excluded_categories = (
                        restrict_raking_targets_to_observed(weight_data, targets)
                    )
                scheme = wp.scheme_from_dict(effective_raking_targets)
                weighted_subset = wp.weight_dataframe(
                    weight_data[weight_variables].copy(),
                    scheme,
                    weight_column=weight_column,
                )
                df_weighted = df.copy()
                df_weighted[weight_column] = weighted_subset[
                    weight_column
                ].reindex(df.index)
            else:
                if target_data is None:
                    raise ValueError("No joint target table is available.")
                weighted_subset = poststratification_weights(
                    weight_data,
                    target_data,
                    weight_variables,
                    weight_column,
                )
                df_weighted = df.copy()
                df_weighted[weight_column] = weighted_subset[
                    weight_column
                ].reindex(df.index)

            weight_values = df_weighted[weight_column]
            if (
                weight_values.isna().any()
                or not weight_values.map(pd.notna).all()
                or (weight_values < 0).any()
            ):
                raise ValueError("Weight calculation produced invalid weights.")
            efficiency = wp.weighting_efficiency(weight_values)

        st.success("Weighting completed.")
        if excluded_categories:
            st.warning(
                "Restricted-category weighting was used. Missing target "
                "categories were excluded and the remaining targets were "
                "rescaled to 100% within each dimension. These weights do not "
                "represent the full benchmark population."
            )
        with st.container(horizontal=True):
            st.metric("Weighting efficiency", f"{efficiency:.1f}%", border=True)
            st.metric(
                "Effective sample size",
                f"{len(df_weighted) * efficiency / 100:,.0f}",
                border=True,
            )
            st.metric("Mean weight", f"{weight_values.mean():.3f}", border=True)
            st.metric("Maximum weight", f"{weight_values.max():.3f}", border=True)

        with st.container(border=True):
            st.subheader("Weighted sample vs. target")
            fit_rows = []
            report_rows = []
            target_column = (
                "Stats SA target (%)"
                if source == "Latest Stats SA population estimates"
                else "Target (%)"
            )
            for column in weight_variables:
                unweighted = (
                    weight_data[column].value_counts(normalize=True) * 100
                ).rename("Unweighted sample (%)")
                weighted = (
                    df_weighted.groupby(weight_data[column], observed=True)[
                        weight_column
                    ].sum()
                    / weight_values.sum()
                    * 100
                ).rename("Weighted sample (%)")

                if method == "Raking":
                    expected = pd.Series(
                        effective_raking_targets[column]
                    ).rename("Target (%)")
                elif target_data is not None:
                    expected = (
                        target_data.groupby(column, observed=True)["target_percent"]
                        .sum()
                        .rename("Target (%)")
                    )
                else:
                    expected = comparison_targets[column].rename("Target (%)")
                comparison = pd.concat(
                    [unweighted, expected, weighted], axis=1
                ).fillna(0)
                label = display_names.get(column, column)
                max_gap = (
                    comparison["Weighted sample (%)"]
                    - comparison["Target (%)"]
                ).abs().max()
                fit_rows.append(
                    {
                        "variable": label,
                        "max_gap": float(max_gap),
                    }
                )
                for category, row in comparison.iterrows():
                    unweighted_percent = float(row["Unweighted sample (%)"])
                    target_percent = float(row["Target (%)"])
                    weighted_percent = float(row["Weighted sample (%)"])
                    applied_factor = (
                        weighted_percent / unweighted_percent
                        if unweighted_percent > 0
                        else 0
                    )
                    report_rows.append(
                        {
                            "Variable / category": (
                                f"{label} - {category_label(category)}"
                            ),
                            "Unweighted (%)": round(unweighted_percent, 2),
                            target_column: round(target_percent, 2),
                            "Applied weight factor": round(applied_factor, 2),
                            "Final weighted (%)": round(weighted_percent, 2),
                        }
                    )

            if report_rows:
                st.dataframe(
                    pd.DataFrame(report_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Applied weight factor": st.column_config.NumberColumn(
                            format="%.2f"
                        )
                    },
                )

        with st.container(border=True):
            st.subheader("Weighting process monitor")
            st.caption(
                "Maximum absolute gap between weighted and target category "
                "shares, in percentage points."
            )
            fit_columns = st.columns(min(4, len(fit_rows)))
            for column, fit in zip(fit_columns, fit_rows):
                column.metric(
                    f"{fit['variable']} max gap",
                    f"{fit['max_gap']:.2f} pp",
                )
            st.metric(
                "Effective sample size (NEFF)",
                f"{len(df_weighted) * efficiency / 100:,.0f}",
            )

        out_file = "weighted_data.sav"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / out_file
            column_labels = {
                column: meta.column_names_to_labels.get(column)
                for column in df.columns
            }
            column_labels[weight_column] = "Survey weight"
            pyreadstat.write_sav(
                df_weighted,
                str(output_path),
                file_label=meta.file_label or "",
                column_labels=column_labels,
                variable_value_labels=meta.variable_value_labels,
                compress=True,
            )
            output_data = output_path.read_bytes()
        st.download_button(
            "Download weighted .sav file",
            output_data,
            file_name=out_file,
            mime="application/octet-stream",
        )
        st.caption(
            "These weights align the selected distributions. They do not by "
            "themselves correct coverage bias or provide complex-survey "
            "standard errors."
        )
    except Exception as exc:
        st.error(f"Could not calculate weights: {exc}")
