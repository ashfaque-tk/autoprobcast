from pathlib import Path
from typing import BinaryIO
import zipfile

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

class IngestionError(ValueError):
    "Raised when uploaded data fails basic checks"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def load_dataset(file: BinaryIO, filename: str) -> pd.DataFrame:
    # TODO 1: derive the lowercased extension from `filename`.
    # TODO 2: raise IngestionError if it is not in SUPPORTED_EXTENSIONS.
    # TODO 3: read with pd.read_csv / pd.read_excel, dtype=str.
    #         shortcut: read_excel reads the first sheet only, add sheet selection when a dataset needs it.
    # TODO 4: translate EmptyDataError, ParserError, UnicodeDecodeError,
    #         zipfile.BadZipFile and ValueError into IngestionError (raise ... from exc).
    # TODO 5: raise IngestionError if the frame has zero rows.
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported extension '{ext or '<none>'}'. Expected one of {sorted(SUPPORTED_EXTENSIONS)}."
        )
    try:
        if ext == '.csv':
            df = pd.read_csv(file,dtype=str,keep_default_na=False) 
        elif ext in { '.xlsx' , '.xls'}: 
            loaded = pd.read_excel(file,dtype=str,keep_default_na=False,sheet_name=None)
            if isinstance(loaded , dict):
                # Combine all non-empty sheets into one DataFrame
                non_empty = [
                    sheet_df 
                    for sheet_df in loaded.values()
                    if not sheet_df.empty
                ]
                df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
            else:
                df = loaded 

    except (EmptyDataError, ParserError, UnicodeDecodeError, zipfile.BadZipFile, ValueError) as exc:
        raise IngestionError(f"Failed to parse '{filename}': {exc}") from exc

    # empty check 
    if df.empty:
        raise IngestionError(f"Dataset '{filename}' contains zero rows.")

    return df