import numpy as np
import pandas as pd
import pytest

from demand_platform.analytics.capability import (
    CapabilityReport,
    assess_capability,
    distribution_capability,
    history_capability,
    seasonality_capability,
)
from demand_platform.analytics.eda import PROFILE_COLUMNS, series_profile
from demand_platform.data.validation import ValidationReport

NAN = np.nan


def profile(**columns) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(v, dtype="float64") for k, v in columns.items()})


def report(frequency="W-MON", negative=0, non_integer=0) -> ValidationReport:
    return ValidationReport(
        n_rows=0, n_series=0, missing_dates=0, missing_sales=0, non_finite_sales=0,
        negative_sales=negative, non_integer_sales=non_integer, duplicate_keys=0,
        off_grid_dates=0, missing_periods=0, series_with_gaps=0, frequency=frequency,
    )


def test_history_shares_count_never_sold_as_not_covered():
    out = history_capability(profile(n_periods=[0, 30, 60, 110]), 52)
    assert out == {
        "n_series": 4,
        "n_never_sold": 1,
        "median_periods": 60.0,
        "share_one_cycle": 0.5,
        "share_two_cycles": 0.25,
    }


def test_history_unknown_period_gives_nan_shares():
    out = history_capability(profile(n_periods=[10, 20]), None)
    assert out["median_periods"] == 15.0
    assert np.isnan(out["share_one_cycle"]) and np.isnan(out["share_two_cycles"])


def test_history_empty_profile():
    out = history_capability(profile(n_periods=[]), 52)
    assert out["n_series"] == 0 and out["n_never_sold"] == 0
    assert np.isnan(out["median_periods"]) and np.isnan(out["share_one_cycle"])


def test_seasonality_ignores_nan_keeps_zero_and_negative():
    out = seasonality_capability(profile(seasonal_acf=[NAN, 0.0, -0.2, 0.4]))
    assert out == {"n_seasonal_eligible": 3, "median_seasonal_acf": 0.0}


def test_seasonality_all_nan():
    out = seasonality_capability(profile(seasonal_acf=[NAN, NAN]))
    assert out["n_seasonal_eligible"] == 0 and np.isnan(out["median_seasonal_acf"])


def test_distribution_zero_share_is_pooled():
    p = profile(
        n_periods=[10, 100, 0],
        missing_sales=[0, 0, 0],
        zero_share=[0.9, 0.1, NAN],
        adi=[10.0, 1.0, NAN],
        dispersion=[3.0, 0.8, NAN],
    )
    out = distribution_capability(p, report())
    assert out["count_data"] is True
    assert out["zero_share"] == pytest.approx(19 / 110)
    assert out["median_adi"] == 5.5
    assert out["share_overdispersed"] == 0.5


def test_distribution_missing_sales_reduce_known_values():
    p = profile(n_periods=[10], missing_sales=[5], zero_share=[0.2], adi=[1.0], dispersion=[1.0])
    assert distribution_capability(p, report())["zero_share"] == pytest.approx(0.2)


def test_distribution_fractions_or_negatives_are_not_count_data():
    p = profile(n_periods=[2], missing_sales=[0], zero_share=[0.0], adi=[1.0], dispersion=[1.0])
    assert distribution_capability(p, report(non_integer=1))["count_data"] is False
    assert distribution_capability(p, report(negative=1))["count_data"] is False


def test_distribution_empty_profile():
    p = profile(n_periods=[], missing_sales=[], zero_share=[], adi=[], dispersion=[])
    out = distribution_capability(p, report())
    assert all(np.isnan(out[k]) for k in ["zero_share", "median_adi", "share_overdispersed"])


def test_assess_capability_end_to_end():
    weeks = 120
    df = pd.DataFrame({
        "store_id": "1",
        "product_id": np.repeat(["A", "B"], weeks),
        "date": np.tile(pd.date_range("2021-04-12", periods=weeks, freq="W-MON"), 2),
        "sales": np.concatenate([np.tile([0.0, 2.0], weeks // 2), np.zeros(weeks)]),
    })
    p = series_profile(df, "W-MON")
    assert list(p.columns) == PROFILE_COLUMNS
    out = assess_capability(p, report())
    assert isinstance(out, CapabilityReport)
    assert out.seasonal_period == 52 and out.n_series == 2 and out.n_never_sold == 1
    assert out.share_two_cycles == 0.5
    assert out.n_seasonal_eligible == 1
