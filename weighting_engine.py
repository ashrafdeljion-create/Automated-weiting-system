"""Survey weighting estimators and diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from ipfn import ipfn
from scipy.optimize import root


def raking_targets_from_target_table(
    target_data: pd.DataFrame, dimensions: list[str]
) -> dict[str, dict[Any, float]]:
    if not dimensions:
        raise ValueError("Select at least one Raking dimension.")
    required = set(dimensions) | {"target_percent"}
    if not required.issubset(target_data.columns):
        raise ValueError("The target table is missing Raking dimensions.")

    target_percent = pd.to_numeric(target_data["target_percent"], errors="raise")
    if target_percent.isna().any() or (target_percent < 0).any():
        raise ValueError("Raking targets must be present and non-negative.")
    if abs(target_percent.sum() - 100) > 0.01:
        raise ValueError("Raking target percentages must sum to 100.")

    targets: dict[str, dict[Any, float]] = {}
    target_data = target_data.assign(target_percent=target_percent)
    for dimension in dimensions:
        if target_data[dimension].isna().any():
            raise ValueError(
                f"Raking target dimension '{dimension}' contains missing values."
            )
        margin = target_data.groupby(dimension, observed=True)["target_percent"].sum()
        targets[dimension] = margin.astype(float).to_dict()
    return targets


def missing_positive_raking_categories(
    data: pd.DataFrame, targets: dict[str, dict[Any, float]]
) -> dict[str, list[Any]]:
    missing: dict[str, list[Any]] = {}
    for column, distribution in targets.items():
        observed = set(data[column].dropna().unique())
        categories = [
            category
            for category, target_percent in distribution.items()
            if target_percent > 0 and category not in observed
        ]
        if categories:
            missing[column] = sorted(categories, key=str)
    return missing


def restrict_raking_targets_to_observed(
    data: pd.DataFrame, targets: dict[str, dict[Any, float]]
) -> tuple[dict[str, dict[Any, float]], dict[str, list[Any]]]:
    adjusted: dict[str, dict[Any, float]] = {}
    excluded: dict[str, list[Any]] = {}
    for column, distribution in targets.items():
        observed = set(data[column].dropna().unique())
        unmatched = observed - set(distribution)
        if unmatched:
            categories = ", ".join(sorted(map(str, unmatched)))
            raise ValueError(
                f"Sample categories for '{column}' are missing from the "
                f"target distribution: {categories}."
            )

        supported = {
            category: float(target_percent)
            for category, target_percent in distribution.items()
            if category in observed
        }
        total = sum(supported.values())
        if total <= 0:
            raise ValueError(
                f"No positive target remains for observed categories in "
                f"'{column}'."
            )
        adjusted[column] = {
            category: target_percent / total * 100
            for category, target_percent in supported.items()
        }

        missing = [
            category
            for category, target_percent in distribution.items()
            if target_percent > 0 and category not in observed
        ]
        if missing:
            excluded[column] = sorted(missing, key=str)

    return adjusted, excluded


def poststratification_weights(
    data: pd.DataFrame,
    target: pd.DataFrame,
    dimensions: list[str],
    weight_column: str = "weight",
) -> pd.DataFrame:
    if not dimensions:
        raise ValueError("Select at least one weighting variable.")
    if data[dimensions].isna().any().any():
        raise ValueError(
            "Post-stratification variables contain missing values. "
            "Resolve or exclude them before weighting."
        )

    target = target[dimensions + ["target_percent"]].copy()
    target["target_percent"] = pd.to_numeric(
        target["target_percent"], errors="raise"
    )
    if target["target_percent"].isna().any() or (
        target["target_percent"] < 0
    ).any():
        raise ValueError("Target percentages must be present and non-negative.")
    target_total = target["target_percent"].sum()
    if target_total <= 0 or abs(target_total - 100) > 0.01:
        raise ValueError(
            f"Joint target percentages must sum to 100 (currently "
            f"{target_total:.2f})."
        )
    if target.duplicated(dimensions).any():
        raise ValueError("The target file contains duplicate category cells.")
    target["target_share"] = target["target_percent"] / target_total

    sample = (
        data.groupby(dimensions, observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    sample["sample_share"] = sample["count"] / len(data)
    matched = target.merge(sample, on=dimensions, how="left", validate="one_to_one")

    missing_target_cells = matched.loc[
        (matched["target_share"] > 0) & matched["sample_share"].isna()
    ]
    if not missing_target_cells.empty:
        raise ValueError(
            "The sample has no respondents in one or more population target "
            "cells. Those cells cannot be created by weighting."
        )

    unmatched_sample = sample.merge(
        target[dimensions], on=dimensions, how="left", indicator=True
    )
    if (unmatched_sample["_merge"] == "left_only").any():
        raise ValueError(
            "The target file does not include every observed sample category cell."
        )

    matched["cell_weight"] = matched["target_share"] / matched["sample_share"]
    weights = data.merge(
        matched[dimensions + ["cell_weight"]],
        on=dimensions,
        how="left",
        validate="many_to_one",
    )
    result = data.copy()
    result[weight_column] = weights["cell_weight"].to_numpy()
    return result


def rake_weights(
    data: pd.DataFrame,
    targets: dict[str, dict[Any, float]],
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> tuple[pd.Series, bool]:
    if not targets:
        raise ValueError("Provide at least one Raking target distribution.")
    dimensions = list(targets)
    if data[dimensions].isna().any().any():
        raise ValueError("Raking variables contain missing values.")
    if data.empty:
        raise ValueError("There are no complete records to weight.")

    effective_targets: dict[str, dict[Any, float]] = {}
    for column, distribution in targets.items():
        if not distribution:
            raise ValueError(f"Raking target '{column}' has no categories.")
        values = np.asarray(list(distribution.values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Raking target '{column}' is invalid.")
        if not np.isclose(values.sum(), 100, atol=0.01):
            raise ValueError(f"Raking target '{column}' must sum to 100.")
        observed = set(data[column].unique())
        unmatched = observed - set(distribution)
        if unmatched:
            categories = ", ".join(sorted(map(str, unmatched)))
            raise ValueError(
                f"Sample categories for '{column}' are missing from the "
                f"target distribution: {categories}."
            )
        absent_positive = [
            category
            for category, target_percent in distribution.items()
            if target_percent > 0 and category not in observed
        ]
        if absent_positive:
            categories = ", ".join(sorted(map(str, absent_positive)))
            raise ValueError(
                f"Positive target categories for '{column}' have no sample "
                f"respondents: {categories}."
            )
        effective_targets[column] = {
            category: float(target_percent)
            for category, target_percent in distribution.items()
            if category in observed
        }

    sample_cells = (
        data.groupby(dimensions, observed=True)
        .size()
        .rename("total")
        .reset_index()
    )
    sample_cells["total"] = sample_cells["total"].astype(float)
    sample_count = sample_cells.rename(
        columns={"total": "sample_total"}
    ).copy()
    aggregates = [
        pd.Series(distribution, dtype=float)
        for distribution in effective_targets.values()
    ]
    fitter = ipfn.ipfn(
        sample_cells.copy(),
        aggregates,
        [[dimension] for dimension in dimensions],
        weight_col="total",
        convergence_rate=tolerance,
        max_iteration=max_iterations,
        verbose=1,
    )
    fitted_result = fitter.iteration()
    if not isinstance(fitted_result, tuple) or len(fitted_result) < 2:
        raise RuntimeError("The IPF estimator returned an unexpected result.")
    fitted_cells, converged = fitted_result[:2]

    fitted_cells = fitted_cells.rename(columns={"total": "fitted_total"})
    factors = fitted_cells.merge(
        sample_count, on=dimensions, how="left", validate="one_to_one"
    )
    if (factors["sample_total"] <= 0).any():
        raise RuntimeError("IPF returned a target cell with no sample records.")
    factors["factor"] = factors["fitted_total"] / factors["sample_total"]

    indexed = data.reset_index(drop=True).copy()
    indexed["__row_number"] = np.arange(len(indexed))
    weighted_rows = indexed.merge(
        factors[dimensions + ["factor"]],
        on=dimensions,
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("__row_number")
    weights = pd.Series(
        weighted_rows["factor"].to_numpy(dtype=float),
        index=data.index,
        name="weight",
    )
    if weights.isna().any() or not np.isfinite(weights).all():
        raise RuntimeError("IPF did not produce a finite weight for every row.")
    return weights, bool(converged)


def _calibration_matrix(
    data: pd.DataFrame,
    targets: dict[str, dict[Any, float]],
    total: float,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    columns = [np.ones(len(data), dtype=float)]
    controls = {"Total population": float(total)}
    column_names = ["Total population"]

    for variable, distribution in targets.items():
        target_values = np.asarray(list(distribution.values()), dtype=float)
        if (
            not target_values.size
            or not np.isfinite(target_values).all()
            or (target_values < 0).any()
            or not np.isclose(target_values.sum(), 100, atol=0.01)
        ):
            raise ValueError(
                f"Calibration targets for '{variable}' must be non-negative "
                "and sum to 100."
            )
        sample_categories = set(data[variable].unique())
        target_categories = set(distribution)
        if sample_categories != target_categories:
            absent = sorted(target_categories - sample_categories, key=str)
            unbenchmarked = sorted(sample_categories - target_categories, key=str)
            details = []
            if absent:
                details.append(f"not in sample: {absent}")
            if unbenchmarked:
                details.append(f"not in targets: {unbenchmarked}")
            raise ValueError(
                f"Calibration categories for '{variable}' do not match "
                f"({'; '.join(details)})."
            )
        if len(distribution) < 2:
            continue
        baseline = next(
            category
            for category, target_percent in distribution.items()
            if target_percent > 0
        )
        for category, target_percent in distribution.items():
            if category == baseline:
                continue
            label = f"{variable} = {category}"
            columns.append(data[variable].eq(category).to_numpy(dtype=float))
            controls[label] = total * float(target_percent) / 100
            column_names.append(label)

    return np.column_stack(columns), controls, column_names


def _logit_calibration(
    design: np.ndarray,
    controls: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    if not 0 < lower_bound < 1 < upper_bound:
        raise ValueError(
            "Logit bounds must satisfy 0 < lower bound < 1 < upper bound."
        )

    baseline_logit = np.log(
        (1 - lower_bound) / (upper_bound - 1)
    )

    def values(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        linear_predictor = np.clip(
            baseline_logit + design @ parameters, -700, 700
        )
        probability = 1 / (1 + np.exp(-linear_predictor))
        factors = lower_bound + (upper_bound - lower_bound) * probability
        derivative = (
            (upper_bound - lower_bound)
            * probability
            * (1 - probability)
        )
        return factors, derivative

    def residual(parameters: np.ndarray) -> np.ndarray:
        factors, _ = values(parameters)
        return design.T @ factors - controls

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        _, derivative = values(parameters)
        return design.T @ (design * derivative[:, None])

    solution = root(
        residual,
        np.zeros(design.shape[1], dtype=float),
        jac=jacobian,
    )
    if not solution.success or np.max(np.abs(residual(solution.x))) > 1e-6:
        raise ValueError(
            "Logit calibration did not converge to the requested controls. "
            "Check the targets or use another estimator."
        )
    factors, _ = values(solution.x)
    return factors


def calibration_weights(
    data: pd.DataFrame,
    targets: dict[str, dict[Any, float]],
    estimator: str,
    logit_bounds: tuple[float, float] = (0.3, 3.0),
) -> pd.Series:
    if not targets:
        raise ValueError("Provide at least one calibration target distribution.")
    if data.empty or data[list(targets)].isna().any().any():
        raise ValueError("Calibration variables must have complete records.")

    design, controls, control_names = _calibration_matrix(
        data, targets, float(len(data))
    )
    control_values = np.asarray(
        [controls[name] for name in control_names], dtype=float
    )
    if estimator == "Linear / GREG (Samplics)":
        from samplics.weighting import SampleWeight

        weights = SampleWeight().calibrate(
            samp_weight=np.ones(len(data), dtype=float),
            aux_vars=design,
            control=controls,
        )
    elif estimator == "Logit (bounded)":
        if any(
            target_percent == 0
            for distribution in targets.values()
            for target_percent in distribution.values()
        ):
            raise ValueError(
                "Bounded logit calibration cannot reach a zero-percent target. "
                "Use linear/GREG or revise the target."
            )
        lower_bound, upper_bound = logit_bounds
        factors = _logit_calibration(
            design,
            control_values,
            lower_bound,
            upper_bound,
        )
        weights = factors
    else:
        raise ValueError(f"Unsupported calibration estimator: {estimator}.")

    result = pd.Series(np.asarray(weights, dtype=float), index=data.index, name="weight")
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError(
            "Calibration produced a zero, negative, or non-finite weight. "
            "Try a different estimator or revise the target margins."
        )
    return result


def trim_and_normalize_weights(
    weights: pd.Series,
    target_sum: float,
    trim_method: str = "None",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    lower_percentile: float = 1,
    upper_percentile: float = 99,
) -> tuple[pd.Series, dict[str, float | int | str]]:
    values = weights.to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Weights must be a non-empty, finite, non-negative vector.")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("The requested weight total must be positive and finite.")
    if values.sum() <= 0:
        raise ValueError("Weights sum to zero and cannot be normalized.")

    if trim_method == "None":
        final_values = values * (target_sum / values.sum())
        info: dict[str, float | int | str] = {
            "method": trim_method,
            "lower_bound": float("-inf"),
            "upper_bound": float("inf"),
            "trimmed_count": 0,
        }
    elif trim_method == "Min/max caps":
        if lower_bound is None or upper_bound is None:
            raise ValueError("Enter both lower and upper weight caps.")
        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
            raise ValueError("Weight caps must be finite.")
        if lower_bound < 0 or upper_bound <= lower_bound:
            raise ValueError(
                "Weight caps must satisfy 0 <= minimum multiple < maximum multiple."
            )
        target_mean = target_sum / len(values)
        low = float(lower_bound) * target_mean
        high = float(upper_bound) * target_mean
        final_values = _bounded_normalize(values, target_sum, low, high)
        relative_values = values / values.mean()
        info = {
            "method": trim_method,
            "lower_bound": low,
            "upper_bound": high,
            "trimmed_count": int(
                ((relative_values < lower_bound) | (relative_values > upper_bound)).sum()
            ),
        }
    elif trim_method == "Percentile caps":
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError(
                "Percentiles must satisfy 0 <= lower < upper <= 100."
            )
        normalized_values = values / values.mean()
        target_mean = target_sum / len(values)
        low_ratio, high_ratio = np.percentile(
            normalized_values, [lower_percentile, upper_percentile]
        )
        low, high = low_ratio * target_mean, high_ratio * target_mean
        final_values = _bounded_normalize(
            values, target_sum, float(low), float(high)
        )
        info = {
            "method": trim_method,
            "lower_bound": float(low),
            "upper_bound": float(high),
            "trimmed_count": int(
                ((normalized_values < low_ratio) | (normalized_values > high_ratio)).sum()
            ),
        }
    else:
        raise ValueError(f"Unsupported trimming method: {trim_method}.")

    return (
        pd.Series(final_values, index=weights.index, name=weights.name),
        info,
    )


def _bounded_normalize(
    values: np.ndarray,
    target_sum: float,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    count = len(values)
    if target_sum < count * lower_bound or target_sum > count * upper_bound:
        raise ValueError(
            "The requested total is incompatible with the selected weight caps."
        )

    low_scale = 0.0
    high_scale = max(target_sum / values.sum(), 1.0)
    while np.clip(values * high_scale, lower_bound, upper_bound).sum() < target_sum:
        high_scale *= 2

    for _ in range(100):
        scale = (low_scale + high_scale) / 2
        scaled_total = np.clip(
            values * scale, lower_bound, upper_bound
        ).sum()
        if scaled_total < target_sum:
            low_scale = scale
        else:
            high_scale = scale

    normalized = np.clip(values * high_scale, lower_bound, upper_bound)
    if not np.isclose(normalized.sum(), target_sum, rtol=1e-10, atol=1e-8):
        raise RuntimeError("Weights could not be normalized within the selected caps.")
    return normalized


def weighting_diagnostics(weights: pd.Series) -> dict[str, float | int]:
    values = weights.to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Diagnostics require finite, non-negative weights.")
    mean = float(values.mean())
    if mean <= 0:
        raise ValueError("The mean weight must be positive.")
    coefficient_of_variation = float(values.std(ddof=0) / mean)
    design_effect = 1 + coefficient_of_variation**2
    return {
        "sample_size": int(values.size),
        "weight_sum": float(values.sum()),
        "design_effect": design_effect,
        "effective_sample_size": float(values.size / design_effect),
        "weighting_efficiency_percent": float(100 / design_effect),
        "minimum_weight": float(values.min()),
        "maximum_weight": float(values.max()),
        "mean_weight": mean,
        "median_weight": float(np.median(values)),
        "coefficient_of_variation": coefficient_of_variation,
    }


def comparison_table(
    data: pd.DataFrame,
    weights: pd.Series,
    targets: dict[str, dict[Any, float]],
    labels: dict[str, str] | None = None,
    target_label: str = "Target (%)",
) -> pd.DataFrame:
    if len(data) != len(weights):
        raise ValueError("Data and weights must have the same number of rows.")
    labels = labels or {}
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("Weights must have a positive sum.")

    rows = []
    for variable, distribution in targets.items():
        observed = data[variable].value_counts(dropna=False)
        categories = list(dict.fromkeys([*distribution, *observed.index]))
        for category in categories:
            mask = data[variable].eq(category)
            sample_count = int(mask.sum())
            category_weight = float(weights.loc[mask].sum())
            rows.append(
                {
                    "Variable": labels.get(variable, variable),
                    "Category": str(category),
                    "Unweighted sample (%)": sample_count / len(data) * 100,
                    "Weighted sample (%)": category_weight / total_weight * 100,
                    target_label: float(distribution.get(category, 0.0)),
                    "Margin error (pp)": (
                        category_weight / total_weight * 100
                        - float(distribution.get(category, 0.0))
                    ),
                }
            )
    return pd.DataFrame(rows)


def margin_error_table(
    data: pd.DataFrame,
    weights: pd.Series,
    targets: dict[str, dict[Any, float]],
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    comparison = comparison_table(data, weights, targets, labels)
    target_column = next(
        column
        for column in comparison.columns
        if column.lower().endswith("target (%)")
    )
    return comparison[
        [
            "Variable",
            "Category",
            target_column,
            "Weighted sample (%)",
            "Margin error (pp)",
        ]
    ].copy()
