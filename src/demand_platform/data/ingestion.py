from pathlib import Path
from typing import BinaryIO
import zipfile

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

class IngestionError(ValueError):
    "Raised when uploaded data fails basic checks"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx",".xls"}


def load_dataset(file: BinaryIO, filename: str) -> pd.DataFrame:

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported extension '{ext or '<none>'}'. Expected one of {sorted(SUPPORTED_EXTENSIONS)}."
        )
    try:
        if ext == '.csv':
            df = pd.read_csv(file, dtype=str)
        elif ext in { '.xlsx' , '.xls'}:
            # shortcut: first sheet only, add sheet selection when a dataset needs it.
            df = pd.read_excel(file, dtype=str)

    except (EmptyDataError, ParserError, UnicodeDecodeError, zipfile.BadZipFile, ValueError) as exc:
        raise IngestionError(f"Failed to parse '{filename}': {exc}") from exc

    # empty check 
    if df.empty:
        raise IngestionError(f"Dataset '{filename}' contains zero rows.")

    return df