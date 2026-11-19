from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from pandas.tseries.offsets import BaseOffset

CANONICAL_ID_ROLES = ("store_id", "product_id")

# Strict on purpose: format="ISO8601" also accepts "2021" and "2021.5", which would turn 4-digit IDs into dates.
DATE_FORMAT = "%Y-%m-%d"

ROLE_SYNONYMS = {
    "store_id": {"store", "store_id", "location"},
    "product_id": {"product", "product_id", "item", "sku","item_id"},
}
TARGET_SYNONYMS = {"sales", "quantity", "qty", "demand", "units"}

MIN_DATES_FOR_FREQUENCY = 3

pd.set_option('future.infer_freq_returns_offset', True)
@dataclass(frozen=True)
class SchemaProposal:
    layout: str
    id_columns: dict[str, str]
    date_columns: list[str]
    target_column: str | None
    unmapped_columns: list[str]
    frequency: str | None


def _parse_dates(values: Iterable) -> pd.DatetimeIndex:

    return pd.DatetimeIndex(
        pd.to_datetime(list(values), format=DATE_FORMAT, errors="coerce")
    )
    

def _match_role(header: str, synonyms: set[str]) -> bool:

    return header.strip().lower() in synonyms
 


def _infer_frequency(dates: pd.DatetimeIndex) -> str | None:

    dates_unique  = dates.dropna().unique().sort_values()

    if len(dates_unique) < MIN_DATES_FOR_FREQUENCY:
        return None 
    try:
        frequency = pd.infer_freq(dates_unique)
        if frequency is None:
            return None
        if isinstance(frequency, BaseOffset):
            return frequency.freqstr
        return str(frequency)
        
    except ValueError:
        return None 


def interpret_schema(df: pd.DataFrame) -> SchemaProposal:

    parsed = _parse_dates(df.columns)
    date_columns = [str(col) for col, ts in zip(df.columns, parsed) if pd.notna(ts)]

    if date_columns:
        layout = 'wide'
        frequency = _infer_frequency(_parse_dates(date_columns))
        non_date_columns = [col for col in df.columns if str(col) not in date_columns]
        target_column = None 

    else: 
        #2. detect date columns , long format
        dates_cols = []
        for col in df.columns:
            values  = df[col].dropna()
            if not values.empty and _parse_dates(values).notna().all():
                dates_cols.append(col)

        if not dates_cols:
            return SchemaProposal(
                layout="unknown",
                id_columns={},
                date_columns=[],
                target_column=None,
                unmapped_columns=list(df.columns),
                frequency=None,
            )
        
        layout = "long"
        date_columns = dates_cols
        non_date_columns = [
            col for col in df.columns if col not in dates_cols
        ]

        frequency = (
            _infer_frequency(_parse_dates(df[dates_cols[0]].dropna()))
            if len(dates_cols) == 1
            else None
        )

        target_matches = [
            col for col in non_date_columns
            if _match_role(str(col), TARGET_SYNONYMS)
        ]
        target_column = (
            target_matches[0] if len(target_matches) == 1 else None
        )
    
    # 3. Map identifier columns.
    id_columns = {}

    for role, synonyms in ROLE_SYNONYMS.items():
        matches = [
            col for col in non_date_columns
            if _match_role(str(col), synonyms)
        ]

        if len(matches) == 1:
            id_columns[role] = matches[0]
    # 4. Identify unmapped columns.
    mapped_columns = set(id_columns.values())
    mapped_columns.update(date_columns)

    if target_column is not None:
        mapped_columns.add(target_column)

    unmapped_columns = [
        col for col in df.columns if col not in mapped_columns
    ]

    return SchemaProposal(
        layout=layout,
        id_columns=id_columns,
        date_columns=date_columns,
        target_column=target_column,
        unmapped_columns=unmapped_columns,
        frequency=frequency,
    )
 