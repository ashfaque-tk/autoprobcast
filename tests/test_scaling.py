import numpy as np
import pandas as pd
import pytest
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

from demand_platform.forecasting.scaling import AnnualScale


def frame(series: dict[str, list[float]]) -> pd.DataFrame:
    return pd.concat([
        pd.DataFrame({"unique_id": uid, "ds": pd.date_range("2022-01-03", periods=len(y), freq="W-MON"), "y": y})
        for uid, y in series.items()
    ], ignore_index=True)


def fitted(df, **kwargs):
    t = AnnualScale(52, **kwargs)
    t.set_column_names("unique_id", "ds", "y")
    return t, t.fit_transform(df)


def test_default_warm_start_threshold():
    assert AnnualScale(52).min_obs == 44 and AnnualScale(364).min_obs == 309


def test_scale_uses_last_year_warm_start_and_clip():
    df = frame({"full": [2.0] * 52, "old_level": [100.0] * 8 + [1.0] * 52, "tiny": [0.01] * 10, "short": [3.0] * 10})
    t, out = fitted(df)
    assert t.scale_.to_dict() == pytest.approx({"full": 104.0, "old_level": 52.0, "tiny": 1.0, "short": 156.0})
    np.testing.assert_allclose(out[out.unique_id == "full"].y, 2.0 / 104)
    assert len(out) == len(df) and df.y.iloc[0] == 2.0  # input untouched


def test_nan_values_are_skipped_in_the_level():
    t, _ = fitted(frame({"a": [np.nan] * 2 + [2.0] * 50}), min_obs=10)
    assert t.scale_["a"] == pytest.approx(104.0)


def test_inverse_transform_scales_every_prediction_column():
    t, _ = fitted(frame({"a": [2.0] * 52, "b": [1.0] * 52}))
    preds = pd.DataFrame({"unique_id": ["b", "a"], "ds": pd.Timestamp("2023-01-02"), "m": [0.5, 0.5], "m-lo-80": [0.1, 0.1]})
    out = t.inverse_transform(preds)
    assert out.m.tolist() == [26.0, 52.0] and out["m-lo-80"].tolist() == pytest.approx([5.2, 10.4])
    assert out.unique_id.tolist() == ["b", "a"] and preds.m.tolist() == [0.5, 0.5]
    np.testing.assert_allclose(t.scale_for(["a", "b", "a"]), [104.0, 52.0, 104.0])


def test_mlforecast_forecasts_in_original_units():
    df = frame({"small": [2.0] * 60, "big": [80.0] * 60})
    scale = AnnualScale(52)
    mlf = MLForecast(models={"m": LinearRegression()}, freq="W-MON", lags=[1], target_transforms=[scale])
    mlf.fit(df)
    preds = mlf.predict(2)
    np.testing.assert_allclose(preds[preds.unique_id == "small"].m, 2.0)
    np.testing.assert_allclose(preds[preds.unique_id == "big"].m, 80.0)
