import numpy as np
import pandas as pd
import pytest

from demand_platform.data.standardization import CANONICAL_COLUMNS
from demand_platform.data.validation import (
    _calendar_issues,
    _duplicate_keys,
    _sales_issues,
    validate,
)

MONDAYS = pd.to_datetime(["2021-04-12", "2021-04-19", "2021-04-26", "2021-05-03"])


def canonical(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype("float64")
    return df


def clean_frame() -> pd.DataFrame:
    return canonical(
        [("1", "A", d, 1.0) for d in MONDAYS] + [("2", "A", d, 0.0) for d in MONDAYS[1:]]
    )


def test_clean_frame_is_valid():
    report = validate(clean_frame(), "W-MON")
    assert report.is_valid
    assert (report.n_rows, report.n_series) == (7, 2)
    assert report.missing_periods == report.off_grid_dates == report.duplicate_keys == 0


def test_empty_frame_is_valid():
    report = validate(canonical([]), "W-MON")
    assert report.is_valid
    assert report.n_rows == report.n_series == report.missing_periods == 0


def test_wrong_columns_raise():
    with pytest.raises(ValueError):
        validate(clean_frame().rename(columns={"sales": "qty"}), "W-MON")
    with pytest.raises(ValueError):
        validate(clean_frame()[list(reversed(CANONICAL_COLUMNS))], "W-MON")


def test_input_not_mutated():
    df = clean_frame().iloc[::-1]
    before = df.copy()
    validate(df, "W-MON")
    pd.testing.assert_frame_equal(df, before)


def test_sales_issues_counts():
    sales = pd.Series([1.0, np.nan, np.inf, -np.inf, -1.0, 2.5, -0.0, 3.0])
    assert _sales_issues(sales) == (1, 2, 1, 1)


def test_sales_issues_empty():
    assert _sales_issues(pd.Series([], dtype="float64")) == (0, 0, 0, 0)


def test_sales_issues_returns_plain_ints():
    assert all(type(v) is int for v in _sales_issues(pd.Series([1.0, np.nan])))


def test_non_finite_sales_blocks():
    df = clean_frame()
    df.loc[0, "sales"] = np.inf
    report = validate(df, "W-MON")
    assert report.non_finite_sales == 1
    assert not report.is_valid


def test_negative_and_fractional_sales_are_warnings_only():
    df = clean_frame()
    df.loc[0, "sales"] = -1.0
    df.loc[1, "sales"] = 2.5
    report = validate(df, "W-MON")
    assert (report.negative_sales, report.non_integer_sales) == (1, 1)
    assert report.is_valid


def test_duplicate_keys_count_every_involved_row():
    df = pd.concat([clean_frame(), clean_frame().iloc[[0]]], ignore_index=True)
    assert _duplicate_keys(df) == 2
    assert not validate(df, "W-MON").is_valid


def test_nat_rows_are_not_duplicates():
    df = canonical([("1", "A", pd.NaT, 1.0), ("1", "A", pd.NaT, 2.0)])
    assert _duplicate_keys(df) == 0


def test_missing_dates_block():
    df = clean_frame()
    df.loc[0, "date"] = pd.NaT
    report = validate(df, "W-MON")
    assert report.missing_dates == 1
    assert not report.is_valid


def test_missing_sales_is_warning_only():
    df = clean_frame()
    df.loc[0, "sales"] = np.nan
    report = validate(df, "W-MON")
    assert report.missing_sales == 1
    assert report.is_valid


def test_gap_inside_series_is_counted():
    df = clean_frame().drop(index=1)  # series 1 loses 2021-04-19
    assert _calendar_issues(df, "W-MON") == (0, 1, 1)


def test_different_spans_are_not_gaps():
    # series 2 starts a week later than series 1; that is not a gap.
    assert _calendar_issues(clean_frame(), "W-MON") == (0, 0, 0)


def test_duplicates_do_not_hide_gaps():
    df = clean_frame().drop(index=1)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert _calendar_issues(df, "W-MON") == (0, 1, 1)


def test_off_grid_date_is_counted():
    df = canonical([("1", "A", MONDAYS[0], 1.0), ("1", "A", "2021-04-13", 1.0), ("1", "A", MONDAYS[1], 1.0)])
    off_grid, _, _ = _calendar_issues(df, "W-MON")
    assert off_grid == 1


def test_off_grid_first_date_is_counted():
    df = canonical([("1", "A", "2021-04-13", 1.0), ("1", "A", MONDAYS[1], 1.0)])
    off_grid, _, _ = _calendar_issues(df, "W-MON")
    assert off_grid == 1


def test_single_date_and_all_nat_series_have_no_gaps():
    df = canonical([("1", "A", MONDAYS[0], 1.0), ("2", "A", pd.NaT, 1.0)])
    assert _calendar_issues(df, "W-MON") == (0, 0, 0)


def test_no_frequency_skips_calendar_check():
    df = clean_frame().drop(index=1)
    report = validate(df, None)
    assert report.frequency is None
    assert report.missing_periods == report.off_grid_dates == report.series_with_gaps == 0
