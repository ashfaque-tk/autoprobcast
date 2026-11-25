import numpy as np
import pandas as pd
import pytest

from demand_platform.analytics.eda import (
    PROFILE_COLUMNS,
    _seasonal_period,
    aggregate_trend,
    seasonal_profile,
    series_profile,
)
from demand_platform.data.standardization import CANONICAL_COLUMNS

NAN = np.nan


def s(*values) -> pd.Series:
    return pd.Series(values, dtype="float64")


def canonical(series: dict[tuple[str, str], list[float]], start: str = "2021-04-12") -> pd.DataFrame:
    frames = []
    for (store, product), sales in series.items():
        frames.append(pd.DataFrame({
            "store_id": store,
            "product_id": product,
            "date": pd.date_range(start, periods=len(sales), freq="W-MON"),
            "sales": pd.Series(sales, dtype="float64"),
        }))
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                             zip(CANONICAL_COLUMNS, ["object", "object", "datetime64[ns]", "float64"])})
    return pd.concat(frames, ignore_index=True)[CANONICAL_COLUMNS]


# --- helpers -----------------------------------------------------------------

def test_seasonal_period():
    assert _seasonal_period("W-MON") == 52
    assert _seasonal_period("D") == 7
    assert _seasonal_period("MS") == 12
    assert _seasonal_period(None) is None
    assert _seasonal_period("h") is None


# --- series_profile ----------------------------------------------------------

def profile_frame() -> pd.DataFrame:
    return canonical({
        ("1", "A"): [0, 0, 2, 2],
        ("1", "B"): [0, 0, 0, 0],
        ("2", "A"): [5, 0, 0, 5],
    })


def test_profile_shape_and_order():
    p = series_profile(profile_frame(), "W-MON")
    assert list(p.columns) == PROFILE_COLUMNS
    assert list(zip(p.store_id, p.product_id)) == [("1", "A"), ("1", "B"), ("2", "A")]


def test_profile_late_start_uses_active_span():
    row = series_profile(profile_frame(), "W-MON").iloc[0]
    assert row.effective_start == pd.Timestamp("2021-04-26")
    assert (row.n_periods, row.zero_share, row["mean"], row.adi, row.cv2) == (2, 0.0, 2.0, 1.0, 0.0)
    assert row.dispersion == 0.0
    assert row.total == 4.0


def test_profile_never_sold():
    row = series_profile(profile_frame(), "W-MON").iloc[1]
    assert pd.isna(row.effective_start)
    assert (row.n_periods, row.total, row.longest_zero_run, row.n_spikes) == (0, 0.0, 0, 0)
    assert np.isnan(row["mean"]) and np.isnan(row.dispersion) and np.isnan(row.adi)


def test_profile_intermittent_series():
    row = series_profile(profile_frame(), "W-MON").iloc[2]
    assert (row.adi, row.cv2, row.dispersion) == (2.0, 0.0, 2.5)
    assert (row.longest_zero_run, row.current_zero_run) == (2, 0)


def test_profile_seasonal_acf_needs_two_cycles():
    pattern = list(np.tile(1 + np.arange(52) % 5, 2).astype(float))
    p = series_profile(canonical({("1", "A"): pattern, ("1", "B"): pattern[:-1]}), "W-MON")
    assert p.seasonal_acf.iloc[0] == pytest.approx(1.0)
    assert np.isnan(p.seasonal_acf.iloc[1])


def test_profile_without_frequency_skips_seasonality():
    pattern = list(np.tile(1 + np.arange(52) % 5, 2).astype(float))
    p = series_profile(canonical({("1", "A"): pattern}), None)
    assert np.isnan(p.seasonal_acf.iloc[0])


def test_profile_empty():
    p = series_profile(canonical({}), "W-MON")
    assert p.empty and list(p.columns) == PROFILE_COLUMNS


def test_profile_input_not_mutated():
    df = profile_frame().iloc[::-1]
    before = df.copy()
    out = series_profile(df, "W-MON")
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(out, series_profile(profile_frame(), "W-MON"))


# --- aggregates --------------------------------------------------------------

def test_aggregate_trend():
    out = aggregate_trend(canonical({("1", "A"): [0, 2, 1], ("1", "B"): [3, NAN, 0]}))
    assert list(out.columns) == ["date", "total_sales", "active_series"]
    assert out.total_sales.tolist() == [3.0, 2.0, 1.0]
    assert out.active_series.tolist() == [1, 2, 2]


def test_aggregate_trend_all_nan_date_is_nan():
    out = aggregate_trend(canonical({("1", "A"): [1, NAN]}))
    assert np.isnan(out.total_sales.iloc[1])


def test_seasonal_profile_weekly():
    df = canonical({("1", "A"): [2, 2, 2], ("1", "B"): [0, 4, 4], ("1", "C"): [0, 0, 0]})
    out = seasonal_profile(df, "W-MON")
    assert list(out.columns) == ["position", "mean_index", "n_obs"]
    assert out.position.tolist() == [15, 16, 17]
    assert out.mean_index.tolist() == [1.0, 1.0, 1.0]
    assert out.n_obs.tolist() == [1, 2, 2]


def test_seasonal_profile_unsupported_frequency():
    out = seasonal_profile(canonical({("1", "A"): [1, 2]}), None)
    assert out.empty and list(out.columns) == ["position", "mean_index", "n_obs"]
