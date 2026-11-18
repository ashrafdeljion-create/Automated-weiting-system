"""Stats SA benchmark and common survey data utilities."""

import re
from datetime import date
from io import BytesIO
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import pyreadstat
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
    "Race / population group": "population_group",
    "Province": "province",
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

    if "province" in dimensions:
        if geography != "South Africa":
            raise ValueError(
                "Select South Africa as the geography when weighting province."
            )
        if "population_group" in dimensions:
            raise ValueError(
                "The Stats SA provincial table does not provide population-group "
                "targets."
            )
        rows = population.loc[
            (population["geography"] != "South Africa")
            & (population["population_group"] == "All population groups")
        ].copy()
        rows["province"] = rows["geography"]
    elif geography == "South Africa":
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


def category_label(value):
    if pd.isna(value):
        return "(Missing)"
    if isinstance(value, Real) and not isinstance(value, bool):
        if float(value).is_integer():
            return str(int(value))
    if isinstance(value, str) and "." in value:
        try:
            numeric_value = float(value)
        except ValueError:
            return value
        if numeric_value.is_integer():
            return str(int(numeric_value))
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
    return "".join(
        character for character in category_label(value).lower() if character.isalnum()
    )


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
        "province": ("province", "region"),
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


def make_sample_columns(data, variables, prefix):
    working = pd.DataFrame(index=data.index)
    mapping = {}
    for index, variable in enumerate(variables):
        output_column = f"__{prefix}_{index}"
        working[output_column] = data[variable].map(category_label)
        mapping[variable] = output_column
    return working, mapping


def read_sav_bytes(file_bytes):
    return pyreadstat.read_sav(BytesIO(file_bytes), user_missing=True)


def missing_value_mask(data, variable, missing_ranges):
    series = data[variable]
    mask = series.isna()
    for definition in missing_ranges.get(variable, []):
        low = definition["lo"]
        high = definition["hi"]
        if low == high:
            mask |= series.eq(low)
        else:
            mask |= series.between(low, high)
    return mask


def prepare_analysis_data(data, metadata):
    analysis_data = data.copy()
    missing_masks = {}
    missing_ranges = metadata.missing_ranges or {}
    for variable in data.columns:
        mask = missing_value_mask(data, variable, missing_ranges)
        missing_masks[variable] = mask
        if mask.any():
            missing_value = (
                float("nan")
                if pd.api.types.is_numeric_dtype(analysis_data[variable])
                else pd.NA
            )
            analysis_data.loc[mask, variable] = missing_value
    return analysis_data, missing_masks


def weighted_sav_bytes(data, metadata, weight_column):
    output = data.copy()
    labels = dict(metadata.column_names_to_labels or {})
    labels[weight_column] = "Final survey weight"
    columns = set(output.columns)
    value_labels = {
        name: labels_for_values
        for name, labels_for_values in (metadata.variable_value_labels or {}).items()
        if name in columns
    }
    missing_ranges = {
        name: ranges
        for name, ranges in (metadata.missing_ranges or {}).items()
        if name in columns
    }
    display_widths = {
        name: width
        for name, width in (metadata.variable_display_width or {}).items()
        if name in columns
    }
    measures = {
        name: measure
        for name, measure in (metadata.variable_measure or {}).items()
        if name in columns
    }
    formats = {
        name: variable_format
        for name, variable_format in (
            metadata.original_variable_types or {}
        ).items()
        if name in columns
    }

    with TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "weighted_data.sav"
        pyreadstat.write_sav(
            output,
            str(output_path),
            file_label=metadata.file_label or "",
            column_labels={name: labels.get(name) for name in output.columns},
            variable_value_labels=value_labels,
            missing_ranges=missing_ranges,
            variable_display_width=display_widths,
            variable_measure=measures,
            variable_format=formats,
            note=metadata.notes,
            compress=True,
        )
        return output_path.read_bytes()


def excel_report_bytes(diagnostics, comparison, settings):
    summary = pd.DataFrame([{**settings, **diagnostics}])
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Diagnostics", index=False)
        comparison.to_excel(writer, sheet_name="Target comparison", index=False)
    return output.getvalue()
