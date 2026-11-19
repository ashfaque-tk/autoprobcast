from collections.abc import Iterable
from dataclasses import dataclass, replace

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
        # Scoped so the module doesn't change pandas behaviour process-wide.
        with pd.option_context("future.infer_freq_returns_offset", True):
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
 

def confirm_mapping(
    proposal: SchemaProposal,
    store_col: str | None,
    product_col: str | None,
    target_col: str | None = None,
) -> SchemaProposal:
    if proposal.layout == "unknown":
        raise ValueError("Cannot confirm a mapping for an unknown layout.")
    if store_col is None or product_col is None:
        raise ValueError("Both store and product columns are required.")
    if proposal.layout == "long" and target_col is None:
        raise ValueError("A sales column is required for long layout.")
    if proposal.layout == "wide" and target_col is not None:
        raise ValueError("Wide layout takes sales from the date columns; no sales column allowed.")

    chosen = [c for c in (store_col, product_col, target_col) if c is not None]
    if len(set(chosen)) != len(chosen):
        raise ValueError("Each column can be mapped to only one role.")

    # The proposal partitions every dataset column, so it is the full column list.
    all_columns = list(dict.fromkeys([
        *proposal.id_columns.values(),
        *([proposal.target_column] if proposal.target_column is not None else []),
        *proposal.unmapped_columns,
    ]))
    for col in chosen:
        if col in proposal.date_columns:
            raise ValueError(f"Column {col!r} is a date column.")
        if col not in all_columns:
            raise ValueError(f"Column {col!r} is not in the dataset.")

    return replace(
        proposal,
        id_columns={"store_id": store_col, "product_id": product_col},
        target_column=target_col,
        unmapped_columns=[c for c in all_columns if c not in chosen],
    )
