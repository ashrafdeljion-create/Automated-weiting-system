"""Interactive Plotly visualizations for survey weighting results."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

TEAL = "#00727D"
NAVY = "#0B2437"
AMBER = "#E5A33B"
SLATE = "#4895B7"


def comparison_figure(comparison: pd.DataFrame):
    target_column = next(
        (
            column
            for column in comparison.columns
            if column.lower().endswith("target (%)")
        ),
        None,
    )
    if target_column is None or not {
        "Variable",
        "Category",
        "Unweighted sample (%)",
        "Weighted sample (%)",
    }.issubset(comparison.columns):
        raise ValueError("Comparison data is missing chart columns.")
    if comparison.empty:
        raise ValueError("There are no target comparisons to plot.")

    series_columns = {
        "Unweighted sample (%)": "Unweighted sample",
        "Weighted sample (%)": "Weighted sample",
        target_column: "Benchmark target",
    }
    long = comparison.melt(
        id_vars=["Variable", "Category"],
        value_vars=list(series_columns),
        var_name="Series",
        value_name="Percent",
    )
    long["Series"] = long["Series"].map(series_columns)
    figure = px.bar(
        long,
        x="Category",
        y="Percent",
        color="Series",
        barmode="group",
        facet_row="Variable" if comparison["Variable"].nunique() > 1 else None,
        category_orders={"Series": list(series_columns.values())},
        labels={"Percent": "Share (%)"},
        title="Sample and benchmark distributions",
        color_discrete_sequence=["#8CA4B3", TEAL, AMBER],
    )
    figure.update_layout(
        height=max(380, 260 * comparison["Variable"].nunique()),
        legend_title_text="",
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(title_text="")
    figure.update_yaxes(rangemode="tozero")
    return figure


def weight_distribution_figure(weights: pd.Series):
    if weights.empty:
        raise ValueError("There are no weights to plot.")
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.16,
        subplot_titles=("Weight distribution", "Weight spread"),
    )
    figure.add_trace(
        go.Histogram(
            x=weights,
            name="Weights",
            marker_color=TEAL,
            opacity=0.85,
            hovertemplate="Weight: %{x:.4f}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Box(
            x=weights,
            name="Weights",
            marker_color=AMBER,
            boxpoints="outliers",
            hovertemplate="Weight: %{x:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        height=520,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(title_text="Weight", row=2, col=1)
    figure.update_yaxes(title_text="Respondents", row=1, col=1)
    return figure


def margin_error_figure(comparison: pd.DataFrame):
    required = {"Variable", "Category", "Margin error (pp)"}
    if not required.issubset(comparison.columns):
        raise ValueError("Comparison data is missing margin-error columns.")
    if comparison.empty:
        raise ValueError("There are no margin errors to plot.")

    figure = px.bar(
        comparison,
        x="Category",
        y="Margin error (pp)",
        color="Variable",
        barmode="group",
        facet_row="Variable" if comparison["Variable"].nunique() > 1 else None,
        labels={"Margin error (pp)": "Weighted minus target (pp)"},
        title="Residual differences from benchmark margins",
        color_discrete_sequence=[TEAL, NAVY, AMBER, SLATE, "#8A6FB1"],
    )
    figure.add_hline(y=0, line_dash="dash", line_color=NAVY)
    figure.update_layout(
        height=max(360, 240 * comparison["Variable"].nunique()),
        showlegend=comparison["Variable"].nunique() == 1,
        legend_title_text="",
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(title_text="")
    return figure
