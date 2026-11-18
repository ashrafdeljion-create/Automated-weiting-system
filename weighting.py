import re
from datetime import date
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from pypdf import PdfReader


PROVINCES = [
    "Eastern Cape",
    "Free State",
    "Gauteng",
    "KwaZulu-Natal",
    "Limpopo",
    "Mpumalanga",
    "Northern Cape",
    "North West",
    "Western Cape",
]

POPULATION_GROUPS = [
    "Black African",
    "Coloured",
    "Indian/Asian",
    "White",
    "RSA",
]

DIMENSIONS = {
    "Age group": "age_group",
    "Sex": "sex",
    "Population group": "population_group",
}

_AGE_ROW = re.compile(
    r"^\s*(\d{1,2})\s*(?:[-–—\ufffd]\s*(\d{1,2})|(\+))\s+(.*)$"
)

STATS_SA_ARCHIVE_YEAR = 2026
STATS_SA_ARCHIVE_URL = (
    "https://www.statssa.gov.za/publications/P0302/P03022026.pdf"
)


def fetch_latest_stats_sa_population(current_year=None, timeout=30):
    year = current_year or date.today().year
    errors = []

    for candidate_year in range(year, max(year - 5, 2019), -1):
        url = (
            "https://www.statssa.gov.za/publications/P0302/"
            f"P0302{candidate_year}.pdf"
        )
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://www.statssa.gov.za/",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                pdf_bytes = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                continue
            errors.append(f"{candidate_year}: HTTP {exc.code}")
            break
        except URLError as exc:
            errors.append(f"{candidate_year}: {exc.reason}")
            break

        if not pdf_bytes.startswith(b"%PDF"):
            errors.append(f"{candidate_year}: publication is not a PDF")
            continue

        try:
            table = parse_stats_sa_population_pdf(pdf_bytes, candidate_year)
        except Exception as exc:
            errors.append(f"{candidate_year}: could not read PDF ({exc})")
            continue
        return table, candidate_year, url, pdf_bytes

    detail = "; ".join(errors) if errors else "no recent publication was found"
    raise RuntimeError(f"Could not retrieve Stats SA population estimates: {detail}")


def load_stats_sa_population(current_year=None, timeout=30):
    try:
        population, year, url, pdf_bytes = fetch_latest_stats_sa_population(
            current_year=current_year, timeout=timeout
        )
        return population, year, url, False, pdf_bytes
    except RuntimeError as live_error:
        archive_path = (
            Path(__file__).parent
            / "data"
            / f"stats_sa_population_{STATS_SA_ARCHIVE_YEAR}.csv"
        )
        if not archive_path.is_file():
            raise RuntimeError(
                f"{live_error}; the archived Stats SA benchmark file is missing"
            ) from live_error
        archive_pdf_path = archive_path.with_suffix(".pdf")

        try:
            population = pd.read_csv(archive_path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise RuntimeError(
                f"Could not read the archived Stats SA benchmark file: {exc}"
            ) from exc
        try:
            pdf_bytes = archive_pdf_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"Could not read the archived Stats SA report: {exc}"
            ) from exc
        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError("The archived Stats SA report is not a valid PDF.")

        required_columns = {
            "year",
            "geography",
            "population_group",
            "age_group",
            "sex",
            "population",
        }
        if not required_columns.issubset(population.columns):
            raise ValueError(
                "The archived Stats SA benchmark file has an invalid schema."
            )
        if set(population["year"].dropna().unique()) != {
            STATS_SA_ARCHIVE_YEAR
        }:
            raise ValueError(
                "The archived Stats SA benchmark file has an unexpected year."
            )
        if population.duplicated(
            ["geography", "population_group", "age_group", "sex"]
        ).any():
            raise ValueError(
                "The archived Stats SA benchmark file contains duplicate rows."
            )

        return (
            population,
            STATS_SA_ARCHIVE_YEAR,
            STATS_SA_ARCHIVE_URL,
            True,
            pdf_bytes,
        )


def parse_stats_sa_population_pdf(pdf_bytes, year):
    reader = PdfReader(BytesIO(pdf_bytes))
    texts = [page.extract_text() or "" for page in reader.pages]

    national_page = next(
        (
            index
            for index, text in enumerate(texts)
            if "Table 5" in text
            and "by population group, age and sex" in text
            and "Black African" in text
        ),
        None,
    )
    if national_page is None:
        raise ValueError("the national age, sex and population-group table is missing")
    national_text = reader.pages[national_page].extract_text(
        extraction_mode="layout"
    )

    records = []
    for line in national_text.splitlines():
        match = _AGE_ROW.match(line.replace("\u00a0", " "))
        if not match:
            continue
        age_group, numbers = _age_group_and_numbers(match)
        if len(numbers) != 15:
            continue
        for group_index, population_group in enumerate(POPULATION_GROUPS):
            start = group_index * 3
            for sex, count in zip(
                ("Male", "Female"), (numbers[start], numbers[start + 1])
            ):
                records.append(
                    {
                        "year": year,
                        "geography": "South Africa",
                        "population_group": population_group,
                        "age_group": age_group,
                        "sex": sex,
                        "population": count,
                    }
                )
    if len(records) != 17 * len(POPULATION_GROUPS) * 2:
        raise ValueError(
            "unexpected number of age rows in the national table "
            f"({len(records)} category records)"
        )

    province_pages = [
        (index, text)
        for index, text in enumerate(texts)
        if ("Table 10 (a)" in text or "Table 10 (b)" in text)
        and "by age and sex" in text
    ]
    for page_index, _ in province_pages:
        text = reader.pages[page_index].extract_text(extraction_mode="layout")
        if "Table 10 (a)" in text:
            page_provinces = PROVINCES[:4]
        elif "Table 10 (b)" in text:
            page_provinces = PROVINCES[4:]
        else:
            continue

        for line in text.splitlines():
            match = _AGE_ROW.match(line.replace("\u00a0", " "))
            if not match:
                continue
            age_group, numbers = _age_group_and_numbers(match)
            if len(numbers) != len(page_provinces) * 3:
                continue
            for province_index, province in enumerate(page_provinces):
                start = province_index * 3
                for sex, count in zip(
                    ("Male", "Female"), (numbers[start], numbers[start + 1])
                ):
                    records.append(
                        {
                            "year": year,
                            "geography": province,
                            "population_group": "All population groups",
                            "age_group": age_group,
                            "sex": sex,
                            "population": count,
                        }
                    )

    province_rows = sum(
        row["geography"] != "South Africa" for row in records
    )
    if province_rows != 17 * len(PROVINCES) * 2:
        raise ValueError(
            "unexpected number of age rows in the provincial tables "
            f"({province_rows} category records)"
        )
    return pd.DataFrame.from_records(records)


def _age_group_and_numbers(match):
    start, end, plus, values = match.groups()
    age_group = f"{start}+" if plus else f"{start}-{end}"
    cells = re.split(r"\s{2,}", values.strip())
    numbers = [
        int(re.sub(r"[,\s]", "", value))
        for value in cells
        if re.search(r"\d", value)
    ]
    return age_group, numbers


def stats_sa_targets(population, geography, dimensions):
    if not dimensions:
        raise ValueError("Select at least one benchmark dimension.")

    if geography == "South Africa":
        rows = population.loc[population["geography"] == "South Africa"].copy()
        if "population_group" in dimensions:
            rows = rows.loc[rows["population_group"] != "RSA"]
        else:
            rows = rows.loc[rows["population_group"] == "RSA"]
    else:
        if "population_group" in dimensions:
            raise ValueError(
                "The selected Stats SA provincial table does not provide "
                "population-group targets."
            )
        rows = population.loc[population["geography"] == geography].copy()

    target = (
        rows.groupby(dimensions, observed=True, as_index=False)["population"]
        .sum()
    )
    if target.empty or target["population"].sum() <= 0:
        raise ValueError("Stats SA returned no population for this selection.")
    target["target_percent"] = (
        target["population"] / target["population"].sum() * 100
    )
    return target


def raking_targets_from_target_table(target_data, dimensions):
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

    targets = {}
    for dimension in dimensions:
        if target_data[dimension].isna().any():
            raise ValueError(
                f"Raking target dimension '{dimension}' contains missing values."
            )
        margin = (
            target_data.assign(target_percent=target_percent)
            .groupby(dimension, observed=True)["target_percent"]
            .sum()
        )
        targets[dimension] = margin.to_dict()
    return targets


def age_value_to_group(value):
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if not age.is_integer() or age < 0 or age > 120:
        return None
    age = int(age)
    if age >= 80:
        return "80+"
    start = (age // 5) * 5
    return f"{start}-{start + 4}"


def poststratification_weights(
    data, target, dimensions, weight_column
):
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
    if (target["target_percent"] < 0).any():
        raise ValueError("Target percentages cannot be negative.")
    target_total = target["target_percent"].sum()
    if target_total <= 0 or abs(target_total - 100) > 0.01:
        raise ValueError(
            f"Joint target percentages must sum to 100 (currently "
            f"{target_total:.2f})."
        )
    if target.duplicated(dimensions).any():
        raise ValueError("The target file contains duplicate category cells.")
    target["target_share"] = target["target_percent"] / target_total

    sample = data.groupby(dimensions, observed=True).size().rename("count").reset_index()
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
