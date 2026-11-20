from dataclasses import dataclass

import pandas as pd
import numpy as np 

from demand_platform.data.standardization import CANONICAL_COLUMNS, DATE_FORMAT

KEY_COLUMNS = ["store_id", "product_id", "date"]
SERIES_COLUMNS = ["store_id", "product_id"]


@dataclass(frozen=True)
class ValidationReport:
    n_rows: int
    n_series: int
    missing_dates: int
    missing_sales: int
    non_finite_sales: int
    negative_sales: int
    non_integer_sales: int
    duplicate_keys: int
    off_grid_dates: int
    missing_periods: int
    series_with_gaps: int
    frequency: str | None

    @property
    def is_valid(self) -> bool:
        """True when nothing blocks EDA; negatives, fractions, NaN sales and gaps are only warnings."""
        return self.duplicate_keys == 0 and self.missing_dates == 0 and self.non_finite_sales == 0


def _check_columns(df: pd.DataFrame) -> None:
    """Guard the canonical data contract.

    Args:
        df: Supposedly the output of ``standardize``.

    Raises:
        ValueError: if ``list(df.columns) != CANONICAL_COLUMNS`` (missing, extra or reordered).
    """
    if list(df.columns) != CANONICAL_COLUMNS:
        raise ValueError(f"columns {list(df.columns)} do not match with {CANONICAL_COLUMNS}")

    
def _sales_issues(sales: pd.Series) -> tuple[int, int, int, int]:
    """Count problems in the float ``sales`` column.

    Args:
        sales: float64 Series, may contain NaN and ±inf.

    Returns:
        ``(missing, non_finite, negative, non_integer)`` as plain ints.

    Edge cases:
        - NaN counts only as missing; it is never negative or non-integer.
        - ±inf counts as non_finite; -inf is NOT also counted as negative, and inf is never non-integer.
        - -0.0 is not negative.
        - Empty Series gives all zeros.
    """
    arr = pd.to_numeric(sales, errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    missing_mask = np.isnan(arr)
    non_finite_mask = np.isinf(arr)  # catches +inf and -inf (NaN is excluded here)
    finite_vals = arr[~missing_mask & ~non_finite_mask]

    negative_count = int(np.sum(finite_vals < 0))
    non_integer_count = int(np.sum(np.mod(finite_vals, 1.0) != 0))

    return (
        int(np.sum(missing_mask)),
        int(np.sum(non_finite_mask)),
        negative_count,
        non_integer_count,
    )
  

def _duplicate_keys(df: pd.DataFrame) -> int:
    """Count rows whose (store_id, product_id, date) key appears more than once.

    Args:
        df: Canonical table.

    Returns:
        Number of rows involved in a duplicate (``keep=False``), so two identical
        keys count as 2.

    Edge cases:
        Rows with NaT date are excluded; they are reported as missing_dates instead.
    """
  
    if df.empty:
        return 0 
    parsed_dates = pd.to_datetime(df['date'],format=DATE_FORMAT,errors='coerce')
    valid = df.loc[parsed_dates.notna(), KEY_COLUMNS].copy()
    valid['date'] = parsed_dates[parsed_dates.notna()]

    if valid.empty:
        return 0 

    return int(valid.duplicated(subset=KEY_COLUMNS,keep=False).sum())
  

def _calendar_issues(df: pd.DataFrame, frequency: str | None) -> tuple[int, int, int]:
    """Check each series' dates against the expected calendar.

    Args:
        df: Canonical table.
        frequency: pandas offset alias from the schema (e.g. "W-MON"), or None.

    Returns:
        ``(off_grid_dates, missing_periods, series_with_gaps)``:
        - off_grid_dates: unique (series, date) pairs not on the series' grid,
          e.g. a Tuesday under W-MON.
        - missing_periods: total grid dates absent between each series' first
          and last valid date.
        - series_with_gaps: series with at least one missing period.

    Edge cases:
        - frequency None -> (0, 0, 0); the caller reports the check as skipped.
        - NaT dates are ignored; a series with only NaT or one date has no gaps.
        - Use unique dates, so duplicate rows don't hide or create gaps.
        - Grid is ``pd.date_range(first, last, freq=frequency)``. If ``first`` is
          off-grid, date_range snaps forward, so the off-grid count catches it.
        - Gaps are only *inside* each series' own span; different start/end dates
          across series are not gaps.
    """
    if frequency is None:
        return (0, 0, 0)

    valid = df.loc[df["date"].notna(), KEY_COLUMNS].drop_duplicates()
    off_grid = missing_periods = series_with_gaps = 0
    # shortcut: per-series Python loop, vectorise if it gets slow on larger datasets.
    for _, dates in valid.groupby(SERIES_COLUMNS)["date"]:
        grid = pd.date_range(dates.min(), dates.max(), freq=frequency)
        off_grid += int((~dates.isin(grid)).sum())
        missing = int((~grid.isin(dates)).sum())
        missing_periods += missing
        series_with_gaps += missing > 0
    return (off_grid, missing_periods, series_with_gaps)


def validate(df: pd.DataFrame, frequency: str | None) -> ValidationReport:
    """Report data-quality issues in the canonical table without changing it.

    Args:
        df: Output of ``standardize``.
        frequency: ``SchemaProposal.frequency`` of the confirmed schema, may be None.

    Returns:
        ValidationReport with counts for every check; ``frequency`` passed through.

    Raises:
        ValueError: if ``df`` does not have exactly ``CANONICAL_COLUMNS``.

    Edge cases:
        - Empty frame: every count 0, report is valid.
        - ``df`` is never mutated, sorted or filtered in place.
    """
    _check_columns(df)
    missing_sales, non_finite, negative, non_integer = _sales_issues(df["sales"])
    off_grid, missing_periods, series_with_gaps = _calendar_issues(df, frequency)
    return ValidationReport(
        n_rows=len(df),
        n_series=len(df[SERIES_COLUMNS].drop_duplicates()),
        missing_dates=int(df["date"].isna().sum()),
        missing_sales=missing_sales,
        non_finite_sales=non_finite,
        negative_sales=negative,
        non_integer_sales=non_integer,
        duplicate_keys=_duplicate_keys(df),
        off_grid_dates=off_grid,
        missing_periods=missing_periods,
        series_with_gaps=series_with_gaps,
        frequency=frequency,
    )
