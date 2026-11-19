import pandas as pd

from demand_platform.data.schema import (
    _infer_frequency,
    _match_role,
    _parse_dates,
    interpret_schema,
)

WEEKS = ["2021-04-12", "2021-04-19", "2021-04-26"]


def wide(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([["1"] * len(columns)], columns=columns, dtype=str)


def long_frame(**extra: list[str]) -> pd.DataFrame:
    data = {
        "Date": WEEKS * 2,
        "Store": ["0"] * 3 + ["1"] * 3,
        "Product": ["126"] * 6,
        **extra,
    }
    return pd.DataFrame(data, dtype=str)


def test_parse_dates_is_strict():
    parsed = _parse_dates(["2021-04-12", "2021", "1234", "2021.5", "2021-04", "12/04/2021", "abc"])
    assert parsed[0] == pd.Timestamp("2021-04-12")
    assert parsed[1:].isna().all()


def test_match_role_normalises_header():
    assert _match_role(" Store ", {"store"})
    assert not _match_role("Shop", {"store"})


def test_infer_frequency():
    assert _infer_frequency(pd.DatetimeIndex(WEEKS)) == "W-MON"
    assert _infer_frequency(pd.DatetimeIndex(list(reversed(WEEKS)) + WEEKS)) == "W-MON"
    assert _infer_frequency(pd.DatetimeIndex(WEEKS[:2])) is None
    assert _infer_frequency(pd.DatetimeIndex(["2021-04-12", "2021-04-19", "2021-05-10"])) is None


def test_wide_vn2_shape():
    p = interpret_schema(wide(["Store", "Product", *WEEKS]))
    assert p.layout == "wide"
    assert p.id_columns == {"store_id": "Store", "product_id": "Product"}
    assert p.date_columns == WEEKS
    assert p.target_column is None
    assert p.unmapped_columns == []
    assert p.frequency == "W-MON"


def test_wide_keeps_file_order_of_dates():
    shuffled = [WEEKS[2], WEEKS[0], WEEKS[1]]
    p = interpret_schema(wide(["Store", "Product", *shuffled]))
    assert p.date_columns == shuffled
    assert p.frequency == "W-MON"


def test_wide_extra_column_is_unmapped():
    p = interpret_schema(wide(["Store", "Product", "Region", *WEEKS]))
    assert p.unmapped_columns == ["Region"]


def test_wide_unrecognised_id_is_unmapped():
    p = interpret_schema(wide(["Shop", "Product", *WEEKS]))
    assert "store_id" not in p.id_columns
    assert p.unmapped_columns == ["Shop"]


def test_duplicate_role_match_maps_neither():
    p = interpret_schema(wide(["Store", "Location", "Product", *WEEKS]))
    assert "store_id" not in p.id_columns
    assert set(p.unmapped_columns) == {"Store", "Location"}


def test_wide_irregular_dates_have_no_frequency():
    p = interpret_schema(wide(["Store", "Product", "2021-04-12", "2021-04-19", "2021-05-10"]))
    assert p.layout == "wide"
    assert p.frequency is None


def test_long_layout():
    p = interpret_schema(long_frame(Sales=["1", "0", "3", "2", "0", "5"]))
    assert p.layout == "long"
    assert p.date_columns == ["Date"]
    assert p.id_columns == {"store_id": "Store", "product_id": "Product"}
    assert p.target_column == "Sales"
    assert p.unmapped_columns == []
    assert p.frequency == "W-MON"


def test_long_date_column_with_missing_cells():
    df = long_frame(Sales=["1"] * 6)
    df.loc[0, "Date"] = None
    p = interpret_schema(df)
    assert p.layout == "long"
    assert p.date_columns == ["Date"]


def test_long_unrecognised_target_is_unmapped():
    p = interpret_schema(long_frame(Volume=["1"] * 6))
    assert p.target_column is None
    assert p.unmapped_columns == ["Volume"]


def test_long_two_date_columns_are_ambiguous():
    p = interpret_schema(long_frame(Sales=["1"] * 6, Promo_Date=WEEKS * 2))
    assert p.layout == "long"
    assert p.date_columns == ["Date", "Promo_Date"]
    assert p.frequency is None


def test_unknown_layout():
    df = pd.DataFrame({"Store": ["0", "1"], "Product": ["2021", "1234"], "12": ["5", "6"]}, dtype=str)
    p = interpret_schema(df)
    assert p.layout == "unknown"
    assert p.id_columns == {}
    assert p.date_columns == []
    assert p.target_column is None
    assert p.frequency is None
