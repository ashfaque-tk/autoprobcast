import pandas as pd
import pytest

from demand_platform.data.schema import SchemaProposal, confirm_mapping, interpret_schema
from demand_platform.data.standardization import CANONICAL_COLUMNS, standardize

WEEKS = ["2021-04-12", "2021-04-19", "2021-04-26"]


def wide_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": ["1", "007"],
            "Product": ["20", "10"],
            "Extra": ["x", "y"],
            WEEKS[0]: ["3.0", "1"],
            WEEKS[1]: ["abc", None],
            WEEKS[2]: ["0.0", "2.5"],
        },
        dtype=str,
    )


def long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": [WEEKS[1], WEEKS[0], "not-a-date"],
            "Store": ["1", "1", "2"],
            "Product": ["9", "9", "9"],
            "Units": ["4", "5.0", "x"],
            "Note": ["a", "b", "c"],
        },
        dtype=str,
    )


LONG_SCHEMA = SchemaProposal(
    layout="long",
    id_columns={"store_id": "Store", "product_id": "Product"},
    date_columns=["Date"],
    target_column="Units",
    unmapped_columns=["Note"],
    frequency=None,
)


def confirmed(df: pd.DataFrame) -> SchemaProposal:
    return confirm_mapping(interpret_schema(df), "Store", "Product")


def test_wide_melts_to_canonical_long():
    df = wide_frame()
    out = standardize(df, confirmed(df))
    assert list(out.columns) == CANONICAL_COLUMNS
    assert len(out) == 2 * len(WEEKS)
    assert out.index.equals(pd.RangeIndex(len(out)))
    assert out[["store_id", "product_id", "date"]].equals(
        out[["store_id", "product_id", "date"]].sort_values(["store_id", "product_id", "date"], ignore_index=True)
    )


def test_wide_coerces_sales_without_dropping_rows():
    df = wide_frame()
    out = standardize(df, confirmed(df)).set_index(["store_id", "date"])["sales"]
    assert out[("1", pd.Timestamp(WEEKS[0]))] == 3.0
    assert pd.isna(out[("1", pd.Timestamp(WEEKS[1]))])
    assert pd.isna(out[("007", pd.Timestamp(WEEKS[1]))])
    assert out[("007", pd.Timestamp(WEEKS[2]))] == 2.5


def test_types():
    df = wide_frame()
    out = standardize(df, confirmed(df))
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert pd.api.types.is_float_dtype(out["sales"])
    assert set(out["store_id"]) == {"1", "007"}
    assert set(out["product_id"]) == {"20", "10"}


def test_long_renames_and_drops_unmapped():
    df = long_frame()
    out = standardize(df, LONG_SCHEMA)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert len(out) == 3
    first = out.iloc[0]
    assert (first["store_id"], first["date"], first["sales"]) == ("1", pd.Timestamp(WEEKS[0]), 5.0)


def test_long_keeps_bad_values_as_missing():
    df = long_frame()
    out = standardize(df, LONG_SCHEMA)
    bad = out[out["store_id"] == "2"].iloc[0]
    assert pd.isna(bad["date"])
    assert pd.isna(bad["sales"])


def test_input_not_mutated():
    df = wide_frame()
    before = df.copy()
    standardize(df, confirmed(df))
    pd.testing.assert_frame_equal(df, before)


def test_unknown_layout_raises():
    df = pd.DataFrame({"Store": ["1"], "Product": ["2"]}, dtype=str)
    with pytest.raises(ValueError):
        standardize(df, interpret_schema(df))
