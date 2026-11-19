import io

import pandas as pd
import pytest

from demand_platform.data.ingestion import IngestionError, load_dataset


def csv_bytes(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


def test_loads_csv():
    df = load_dataset(csv_bytes("store,product,sales\n1,10,5\n2,20,7\n"), "sales.csv")
    assert list(df.columns) == ["store", "product", "sales"]
    assert len(df) == 2


def test_loads_xlsx():
    buf = io.BytesIO()
    pd.DataFrame({"store": ["1", "2"], "sales": ["5", "7"]}).to_excel(buf, index=False)
    buf.seek(0)
    df = load_dataset(buf, "sales.xlsx")
    assert list(df.columns) == ["store", "sales"]
    assert len(df) == 2


def test_extension_is_case_insensitive():
    df = load_dataset(csv_bytes("a,b\n1,2\n"), "SALES.CSV")
    assert len(df) == 1


@pytest.mark.parametrize("filename", ["data.txt", "data.xls", "data"])
def test_rejects_unsupported_extension(filename):
    with pytest.raises(IngestionError):
        load_dataset(csv_bytes("a,b\n1,2\n"), filename)


def test_rejects_empty_csv():
    with pytest.raises(IngestionError):
        load_dataset(csv_bytes(""), "empty.csv")


def test_rejects_header_only_csv():
    with pytest.raises(IngestionError):
        load_dataset(csv_bytes("a,b\n"), "header.csv")


def test_rejects_corrupt_xlsx():
    with pytest.raises(IngestionError):
        load_dataset(io.BytesIO(b"not an excel file"), "broken.xlsx")


def test_preserves_raw_values():
    df = load_dataset(csv_bytes("store,sales\n001,5\n"), "sales.csv")
    assert df.loc[0, "store"] == "001"
