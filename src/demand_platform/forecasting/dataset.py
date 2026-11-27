import pandas as pd

ID_COL, TIME_COL, TARGET_COL = "unique_id", "ds", "y"
STATIC_COLUMNS = ["store_id", "product_id"]
ID_SEPARATOR = "|"


def to_mlforecast(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical sales table -> the long format mlforecast trains on.

    mlforecast expects one row per (series, period) with three fixed names: ``unique_id``
    (which series), ``ds`` (the timestamp) and ``y`` (the target). Any extra column is a
    candidate *static feature*: a value that never changes within a series and is simply
    repeated for future rows when forecasting.

    Args:
        df: Validated canonical table ``store_id, product_id, date, sales``.

    Returns:
        DataFrame ``unique_id, ds, y, store_id, product_id`` sorted by ``unique_id, ds``.
        - unique_id: ``"<store_id>|<product_id>"``.
        - store_id, product_id: pandas ``category`` dtype, which LightGBM uses natively as
          categorical splits (no one-hot encoding).
        - Rows before a series' first positive sale are dropped (the same active span as
          EDA), so pre-launch zeros don't teach the model "this product sells nothing".
          Series that never sold are dropped entirely.

    Raises:
        ValueError: if any sales value is NaN (mlforecast's lags would silently spread it,
            and LightGBM can't train on a NaN target), or an id contains ``ID_SEPARATOR``.

    Assumes no gaps inside a series: mlforecast lags count rows, not calendar periods.
    ``validate`` reports gaps; VN2 has none.
    """
    if df["sales"].isna().any():
        # shortcut: refuse NaN sales; decide on imputation when a dataset needs it.
        raise ValueError(f"{int(df['sales'].isna().sum())} NaN sales values; mlforecast needs a complete target.")
    ids = df[STATIC_COLUMNS].astype(str)
    if ids.apply(lambda c: c.str.contains(ID_SEPARATOR, regex=False)).any().any():
        raise ValueError(f"store_id or product_id contains '{ID_SEPARATOR}'.")

    out = pd.DataFrame({
        ID_COL: ids["store_id"] + ID_SEPARATOR + ids["product_id"],
        TIME_COL: df["date"],
        TARGET_COL: df["sales"].astype("float64"),
        "store_id": ids["store_id"].astype("category"),
        "product_id": ids["product_id"].astype("category"),
    }).sort_values([ID_COL, TIME_COL], ignore_index=True)
    first_sale = out[TIME_COL].where(out[TARGET_COL] > 0).groupby(out[ID_COL]).transform("min")
    return out[out[TIME_COL] >= first_sale].reset_index(drop=True)


# Weight of each older year relative to the next (1.0, 0.5, 0.25, …), as in the VN2 winner's report.
RECENCY_DECAY = 0.5


def recency_weight(df: pd.DataFrame, year_length: int, decay: float = RECENCY_DECAY) -> pd.Series:
    """Training weight per row: 1 for each series' most recent year, ``decay`` for the year before, …

    Retail behaviour drifts (trends, substitution, new ranges). Down-weighting old years lets
    the model follow recent behaviour while still learning seasonality from the full history.
    Pass the result as a column and ``fit(..., weight_col=...)``.

    Args:
        df: ``to_mlforecast`` output (sorted by ``unique_id, ds``), training rows only.
        year_length: Rows per year (``FrequencyProfile.year_length``).
        decay: Multiplier per year back; 1.0 means no weighting.

    Returns:
        float Series aligned with ``df.index``: ``decay ** (age // year_length)``, where
        ``age`` is the number of rows between that row and its series' last row (0 for the
        last row). Age is counted per series, as in the report, and in rows, so blocks have
        ``year_length`` rows whatever the frequency.

    Vectorised algorithm:
        1. ``age = df.groupby(ID_COL).cumcount(ascending=False)``.
        2. ``return decay ** (age // year_length)`` (as float).

    Readable alternative and its problem:
        Ages from dates, e.g. ``(df.ds.max() - df.ds).dt.days // 365``: it measures from the
        *global* last date, so a series that stopped selling a year ago gets every row
        down-weighted; and 365-day blocks don't line up with 52-week (364-day) blocks, so
        one row a year slips into the wrong block.
    """
    raise NotImplementedError
