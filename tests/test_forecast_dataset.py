import numpy as np
import pandas as pd
import pytest

from demand_platform.forecasting.dataset import recency_weight, to_mlforecast


def canonical(rows):
    return pd.DataFrame(rows, columns=["store_id", "product_id", "date", "sales"]).assign(
        date=lambda d: pd.to_datetime(d["date"]), sales=lambda d: d["sales"].astype(float))


def test_columns_ids_and_dtypes():
    out = to_mlforecast(canonical([("1", "B", "2024-01-08", 2), ("1", "A", "2024-01-08", 1), ("1", "A", "2024-01-01", 3)]))
    assert list(out.columns) == ["unique_id", "ds", "y", "store_id", "product_id"]
    assert out.unique_id.tolist() == ["1|A", "1|A", "1|B"]
    assert out.ds.dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-08", "2024-01-08"]
    assert out.store_id.dtype == "category" and out.product_id.dtype == "category"


def test_trims_before_first_sale_and_drops_never_sold():
    out = to_mlforecast(canonical([
        ("1", "A", "2024-01-01", 0), ("1", "A", "2024-01-08", 0), ("1", "A", "2024-01-15", 4), ("1", "A", "2024-01-22", 0),
        ("1", "Z", "2024-01-01", 0), ("1", "Z", "2024-01-08", 0),
    ]))
    assert out.unique_id.tolist() == ["1|A", "1|A"]
    assert out.y.tolist() == [4.0, 0.0]


def test_nan_sales_raise():
    with pytest.raises(ValueError, match="NaN"):
        to_mlforecast(canonical([("1", "A", "2024-01-01", np.nan)]))


def test_separator_in_id_raises():
    with pytest.raises(ValueError, match=r"\|"):
        to_mlforecast(canonical([("1|2", "A", "2024-01-01", 1)]))


def test_recency_weight_by_yearly_block_per_series():
    df = pd.DataFrame({"unique_id": ["a"] * 120 + ["b"] * 10, "ds": 0, "y": 1.0})
    w = recency_weight(df, 52)
    assert w.iloc[:16].eq(0.25).all() and w.iloc[16:68].eq(0.5).all() and w.iloc[68:120].eq(1.0).all()
    assert w.iloc[120:].eq(1.0).all()
    assert w.index.equals(df.index)


def test_recency_weight_no_decay():
    df = pd.DataFrame({"unique_id": ["a"] * 120, "ds": 0, "y": 1.0})
    assert recency_weight(df, 52, decay=1.0).eq(1.0).all()
