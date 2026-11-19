import pandas as pd

from demand_platform.data.schema import DATE_FORMAT, SchemaProposal

CANONICAL_COLUMNS = ["store_id", "product_id", "date", "sales"]


def _melt_wide(df: pd.DataFrame, schema: SchemaProposal) -> pd.DataFrame:
    """Reshape a wide table (one column per date) into canonical long rows.

    Args:
        df: Raw ingested table, all values str or NaN.
        schema: Confirmed mapping with ``layout == "wide"``.

    Returns:
        DataFrame with columns ``store_id, product_id, date, sales`` and
        ``len(df) * len(schema.date_columns)`` rows. Values are still str/NaN;
        type conversion happens in ``_coerce_types``.

    Edge cases:
        Unmapped columns are dropped. Empty cells stay NaN, never filled.
    """

    store_col,product_col = schema.id_columns['store_id'],schema.id_columns['product_id']
    df = df.melt(id_vars=[store_col,product_col],value_vars=schema.date_columns,var_name='date',value_name='sales')
    df = df.rename(columns={store_col:'store_id',product_col:'product_id'})
    return df


def _select_long(df: pd.DataFrame, schema: SchemaProposal) -> pd.DataFrame:
    """Select and rename the mapped columns of a long table (one row per date).

    Args:
        df: Raw ingested table, all values str or NaN.
        schema: Confirmed mapping with ``layout == "long"``, exactly one date
            column and a non-None ``target_column``.

    Returns:
        DataFrame with columns ``store_id, product_id, date, sales`` and the
        same rows as ``df``. Values are still str/NaN.

    Edge cases:
        Unmapped columns are dropped. Returns a new frame; ``df`` is not mutated.
    """

    mapping = {
        schema.id_columns['store_id']: 'store_id',
        schema.id_columns['product_id']: 'product_id',
        schema.date_columns[0]: 'date',
        schema.target_column: 'sales',
    }
    return df[list(mapping)].rename(columns=mapping)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical string columns to their analytical types.

    Args:
        df: Frame with columns ``store_id, product_id, date, sales`` as str/NaN.

    Returns:
        Same rows with ``date`` as datetime64 and ``sales`` as float64.
        ``store_id`` and ``product_id`` stay str (keeps leading zeros like "007").

    Edge cases:
        Unparseable dates become NaT and unparseable sales become NaN; no row is
        dropped, so validation can report them. Sales is float, not int, because
        values arrive as "3.0" and missing values must stay NaN.
    """

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'],format=DATE_FORMAT,errors='coerce')
    df['sales'] = pd.to_numeric(df['sales'],errors='coerce').astype('float64')
    df['product_id'] = df['product_id'].astype('object')
    df['store_id'] = df['store_id'].astype('object')
    return df 


def standardize(df: pd.DataFrame, schema: SchemaProposal) -> pd.DataFrame:
    """Turn a raw ingested table into the canonical long sales table.

    Args:
        df: Output of ``load_dataset`` (all columns str, missing cells NaN).
        schema: Mapping returned by ``confirm_mapping``.

    Returns:
        New DataFrame with exactly ``CANONICAL_COLUMNS`` in that order, sorted by
        ``store_id, product_id, date``, with a fresh ``RangeIndex``.

    Raises:
        ValueError: if ``schema.layout`` is not "wide" or "long", or a long
            layout has more than one date column.

    Edge cases:
        No deduplication, gap filling or row removal; that is validation's job.
        ``df`` is never mutated.
    """

    if schema.layout == 'wide':
        long_df = _melt_wide(df,schema=schema)
    elif schema.layout =='long':
        if len(schema.date_columns) != 1:
            raise ValueError(f"Long layout needs exactly one date column, got {schema.date_columns}.")
        long_df = _select_long(df,schema)
    else:
        raise ValueError(f"Unsupported layout: {schema.layout!r}")
    long_df = _coerce_types(long_df)
    return long_df.sort_values(['store_id', 'product_id', 'date'], ignore_index=True)[CANONICAL_COLUMNS]

