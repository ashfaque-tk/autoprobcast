import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from demand_platform.forecasting.dataset import to_mlforecast
from demand_platform.forecasting.features import (
    ALL_GROUPS,
    build_forecaster,
    difference,
    feature_groups,
    fourier,
    nonzero_rate,
    periods_since_last_sale,
    periods_since_robust_spike,
    seasonal_correlation,
)
from demand_platform.frequency import DAILY, MONTHLY, WEEKLY

NAN = np.nan


def a(*values):
    return np.array(values, dtype=float)


def same(x, y):
    np.testing.assert_allclose(x, y, equal_nan=True)


# --- values -----------------------------------------------------------------------

def test_nonzero_rate_ignores_nan():
    same(nonzero_rate(a(NAN, 5, 5, 0, 5), 3), a(NAN, 1, 1, 2 / 3, 2 / 3))


def test_periods_since_last_sale():
    same(periods_since_last_sale(a(NAN, 0, 3, 0, NAN, 0)), a(NAN, NAN, 0, 1, 2, 3))


def test_robust_spike_judged_on_previous_window():
    same(periods_since_robust_spike(a(NAN, 1, 1, 1, 1, 10, 1, 1), 4, 3), a(NAN, NAN, NAN, NAN, NAN, 0, 1, 2))


def test_robust_spike_on_sparse_series_is_any_sale():
    same(periods_since_robust_spike(a(NAN, 0, 0, 0, 0, 2, 0, 0, 3), 4, 3), a(NAN, NAN, NAN, NAN, NAN, 0, 1, 2, 0))


def test_robust_spike_uses_mad():
    # previous 4 = [2, 4, 2, 4]: median 3, MAD 1 -> 5 is not a spike; [4, 2, 4, 5]: median 4, MAD 0.5 -> 30 is.
    same(periods_since_robust_spike(a(NAN, 2, 4, 2, 4, 5, 30), 4, 4), a(NAN, NAN, NAN, NAN, NAN, NAN, 0))


def test_difference():
    same(difference(a(NAN, 1, 3, 6), 1), a(NAN, NAN, 2, 3))
    same(difference(a(NAN, 1, 3, 6), 2), a(NAN, NAN, NAN, 5))
    assert np.isnan(difference(a(1, 2), 5)).all()


def test_seasonal_correlation_expanding():
    same(seasonal_correlation(a(NAN, 1, 5, 2, 6, 3), 2, 2), a(NAN, NAN, NAN, NAN, 1, 1))


def test_seasonal_correlation_matches_corrcoef_on_all_pairs():
    x = np.random.default_rng(3).poisson(3.0, 60).astype(float)
    x[0] = NAN
    out = seasonal_correlation(x, 7, 7)
    pairs = ~np.isnan(x[7:]) & ~np.isnan(x[:-7])
    expected = np.corrcoef(x[7:][pairs], x[:-7][pairs])[0, 1]
    assert out[-1] == pytest.approx(expected)


def test_seasonal_correlation_constant_is_nan():
    assert np.isnan(seasonal_correlation(np.full(10, 2.0), 2, 2)).all()


def test_fourier_year_and_week():
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-04"])  # Monday, Thursday
    same(fourier(dates, "year", 1, "sin")[:1], a(0.0))
    same(fourier(dates, "year", 1, "cos")[:1], a(1.0))
    same(fourier(dates, "week", 1, "sin"), a(0.0, np.sin(2 * np.pi * 3 / 7)))
    year = fourier(pd.date_range("2023-01-01", "2024-12-31", freq="D"), "year", 3, "cos")
    assert year.min() >= -1 and year.max() <= 1
    with pytest.raises(ValueError):
        fourier(dates, "year", 1, "tan")
    with pytest.raises(ValueError):
        fourier(dates, "month", 1, "sin")


# --- causality: out[:t] never depends on x[t:] ------------------------------------

TRANSFORMS = {
    "nonzero_rate": lambda x: nonzero_rate(x, 4),
    "periods_since_last_sale": periods_since_last_sale,
    "periods_since_robust_spike": lambda x: periods_since_robust_spike(x, 5, 3),
    "difference": lambda x: difference(x, 3),
    "seasonal_correlation": lambda x: seasonal_correlation(x, 5, 3),
}


@pytest.mark.parametrize("name", TRANSFORMS)
def test_transform_is_causal(name):
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, 40).astype(float)
    x[:2] = NAN
    x[[7, 8, 20]] = [0, 25, 0]
    f = TRANSFORMS[name]
    base = f(x)
    assert base.shape == x.shape
    for t in (3, 10, 25, 39):
        changed = x.copy()
        changed[t:] = rng.poisson(9.0, len(x) - t)
        same(f(changed)[:t], base[:t])


# --- wiring -----------------------------------------------------------------------

def test_weekly_groups():
    groups = feature_groups(WEEKLY)
    assert set(groups) == set(ALL_GROUPS)
    assert groups["lags"]["lags"] == [1, 2, 3, 4, 51, 52, 53]
    last_year = groups["seasonal"]["lag_transforms"][50]
    assert last_year[0].window_size == 5


def test_daily_groups_use_daily_windows_and_two_seasons():
    groups = feature_groups(DAILY)
    assert 364 in groups["lags"]["lags"]
    assert groups["seasonal"]["lag_transforms"][361][0].window_size == 7
    names = [f.__name__ for f in groups["calendar"]["date_features"] if callable(f)]
    assert "fourier_week_sin_2" in names and "fourier_year_cos_3" in names and "week_of_year" in names


def test_monthly_groups_have_no_week_of_year():
    names = [getattr(f, "__name__", f) for f in feature_groups(MONTHLY)["calendar"]["date_features"]]
    assert "week_of_year" not in names and "month" in names


def test_unsupported_frequency_keeps_only_frequency_free_features():
    groups = feature_groups(None)
    assert groups["lags"] == {} and groups["seasonal"] == {} and groups["calendar"] == {}


def test_build_forecaster_static_and_unknown_group():
    _, static = build_forecaster({"m": LGBMRegressor()}, "W-MON")
    assert static == ["store_id", "product_id"]
    _, static = build_forecaster({"m": LGBMRegressor()}, "W-MON", groups=("lags",))
    assert static == []
    with pytest.raises(ValueError, match="Unknown"):
        build_forecaster({"m": LGBMRegressor()}, "W-MON", groups=("lags", "magic"))


# --- end to end: no feature row sees the future -----------------------------------

def test_features_do_not_leak_future_sales():
    weeks, rng = 130, np.random.default_rng(1)
    dates = pd.date_range("2021-01-04", periods=weeks, freq="W-MON")
    canonical = pd.DataFrame({
        "store_id": "1", "product_id": np.repeat(["A", "B"], weeks), "date": np.tile(dates, 2),
        "sales": rng.poisson(2.0, 2 * weeks).astype(float),
    })
    canonical.loc[[0, weeks], "sales"] = 1.0
    cutoff = dates[110]
    changed = canonical.copy()
    changed.loc[changed.date > cutoff, "sales"] = 999.0

    def features(df):
        forecaster, static = build_forecaster({"m": LGBMRegressor()}, "W-MON")
        X = forecaster.preprocess(to_mlforecast(df), static_features=static, dropna=False)
        return X[X.ds <= cutoff].drop(columns="y").reset_index(drop=True)

    pd.testing.assert_frame_equal(features(canonical), features(changed))
