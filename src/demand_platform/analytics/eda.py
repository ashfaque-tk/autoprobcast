import numpy as np
import pandas as pd

SERIES_COLUMNS = ["store_id", "product_id"]

SPIKE_STD = 3.0
MIN_SEASONAL_CYCLES = 2
# shortcut: one window in periods (a quarter for weekly data), make it frequency-aware when daily data arrives.
TREND_WINDOW = 13

# Base frequency code -> seasonal period in rows. Anything else is "unsupported", never guessed.
SEASONAL_PERIODS = {"W": 52, "D": 7, "M": 12, "ME": 12, "MS": 12}

PROFILE_COLUMNS = [
    "store_id",
    "product_id",
    "effective_start",
    "last_date",
    "n_periods",
    "missing_sales",
    "total",
    "zero_share",
    "mean",
    "std",
    "dispersion",
    "adi",
    "cv2",
    "longest_zero_run",
    "current_zero_run",
    "n_spikes",
    "periods_since_last_spike",
    "acf_1",
    "seasonal_acf",
    "trend_ratio",
]


def _seasonal_period(frequency: str | None) -> int | None:
    """Seasonal period in rows for a pandas frequency alias, e.g. "W-MON" -> 52.

    Returns None for None or any frequency not in ``SEASONAL_PERIODS``.
    """
    if frequency is None:
        return None
    return SEASONAL_PERIODS.get(frequency.split("-")[0])




def series_profile(df: pd.DataFrame, frequency: str | None) -> pd.DataFrame:
    """One row of descriptive statistics per series.

    Args:
        df: Validated canonical table (``store_id, product_id, date, sales``).
        frequency: Confirmed schema frequency; only used for ``seasonal_acf``.

    Returns:
        DataFrame with exactly ``PROFILE_COLUMNS``, sorted by store_id, product_id.
        Statistics use each series' active span: its rows from the first sale > 0 to the end.
        - effective_start / last_date: first positive sale date / last date of the series.
        - n_periods: rows in the active span; missing_sales: NaN in it.
        - total: sum over the whole series; zero_share, mean, std (ddof=0): over the
          non-NaN active span.
        - dispersion: variance (ddof=0) / mean of the active span; 1 is Poisson-like,
          above 1 over-dispersed. NaN when the mean is 0 or NaN.
        - adi: known periods / non-zero periods. cv2: (std / mean)² of non-zero values.
        - longest_zero_run / current_zero_run: longest and trailing run of zeros; NaN ends a run.
        - n_spikes: values > mean + SPIKE_STD * std; periods_since_last_spike: rows from the
          last spike to the end (NaN if none).
        - acf_1: autocorrelation at lag 1. seasonal_acf: at lag = seasonal period, only where
          n_periods >= MIN_SEASONAL_CYCLES * period; otherwise NaN.
        - trend_ratio: mean of the last TREND_WINDOW rows / mean of the TREND_WINDOW before;
          NaN if fewer than 2 * TREND_WINDOW rows or the earlier mean is 0.

    Edge cases:
        - Never-sold series: effective_start NaT, n_periods 0, total 0, statistics NaN,
          run/spike counts 0.
        - Unsupported or None frequency: seasonal_acf is NaN for every series.
        - Empty df: empty frame with ``PROFILE_COLUMNS``.
        - Undefined statistics (std 0, too few rows) are NaN, never 0.
        - ``df`` is never mutated and may arrive unsorted.
    """
    period = _seasonal_period(frequency)
    if df.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)

    data = df.sort_values([*SERIES_COLUMNS, "date"], ignore_index=True)
    # The only place the key columns are named: every series becomes one integer id.
    sid = data.groupby(SERIES_COLUMNS, sort=False).ngroup()

    def by_series(values: pd.Series):
        return values.groupby(sid)

    first_sale = by_series(data["date"].where(data["sales"] > 0)).transform("first")
    active = data["date"] >= first_sale  # NaT compares False, so never-sold rows are inactive
    sales = data["sales"].where(active)  # NaN outside the active span
    known = sales.notna()
    is_zero = sales.eq(0)  # False for NaN, so NaN ends a zero run

    streak = is_zero.ne(is_zero.shift()).cumsum()
    run = (is_zero.groupby([sid, streak]).cumcount() + 1).where(is_zero, 0)

    mean = by_series(sales).transform("mean")
    std = by_series(sales).transform("std", ddof=0)
    is_spike = sales > mean + SPIKE_STD * std
    pos = data.groupby(sid).cumcount()

    nonzero = sales.where(sales.ne(0))
    n_known = by_series(known).sum()
    n_nonzero = by_series(nonzero.notna()).sum()
    n_periods = by_series(active).sum()

    out = pd.DataFrame({
        "effective_start": by_series(first_sale).first(),
        "last_date": by_series(data["date"]).max(),
        "n_periods": n_periods,
        "missing_sales": by_series(active & data["sales"].isna()).sum(),
        "total": by_series(data["sales"]).sum(),
        "zero_share": by_series(is_zero).sum() / n_known.where(n_known > 0),
        "mean": by_series(sales).mean(),
        "std": by_series(sales).std(ddof=0),
        "adi": n_known / n_nonzero.where(n_nonzero > 0),
        "cv2": (by_series(nonzero).std(ddof=0) / by_series(nonzero).mean()) ** 2,
        "longest_zero_run": by_series(run).max(),
        "current_zero_run": by_series(run).last(skipna=False),
        "n_spikes": by_series(is_spike).sum(),
        "periods_since_last_spike": by_series(pos).max() - by_series(pos.where(is_spike)).max(),
        "acf_1": _acf(sales, sid, 1),
        "seasonal_acf": (
            _acf(sales, sid, period).where(n_periods >= MIN_SEASONAL_CYCLES * period) if period else np.nan
        ),
        "trend_ratio": _trend_ratio(sales, sid, data.groupby(sid).cumcount(ascending=False), TREND_WINDOW)
        .where(n_periods >= 2 * TREND_WINDOW),
    })
    out["dispersion"] = out["std"] ** 2 / out["mean"].where(out["mean"] != 0)

    keys = data.groupby(sid)[SERIES_COLUMNS].first()
    return keys.join(out).reset_index(drop=True)[PROFILE_COLUMNS]


# shortcut: lags and windows count rows, not calendar periods; regrid first if series have gaps.
def _acf(sales: pd.Series, sid: pd.Series, lag: int) -> pd.Series:
    """Pearson correlation of x[t] and x[t-lag] per series id; NaN for < 2 pairs or zero variance."""
    x, y = sales, sales.groupby(sid).shift(lag)
    pair = x.notna() & y.notna()
    x, y = x.where(pair), y.where(pair)
    dx = x - x.groupby(sid).transform("mean")
    dy = y - y.groupby(sid).transform("mean")
    num = (dx * dy).groupby(sid).sum()
    den = np.sqrt((dx**2).groupby(sid).sum() * (dy**2).groupby(sid).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    return r.where((pair.groupby(sid).sum() >= 2) & (den > 0))


def _trend_ratio(sales: pd.Series, sid: pd.Series, from_end: pd.Series, window: int) -> pd.Series:
    """Mean of the last ``window`` rows / mean of the ``window`` before, per series id (caller checks length)."""
    recent = sales.where(from_end < window).groupby(sid).mean()
    previous = sales.where((from_end >= window) & (from_end < 2 * window)).groupby(sid).mean()
    return recent / previous.where(previous != 0)


def aggregate_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Total sales and number of active series per date.

    Returns:
        DataFrame ``date, total_sales, active_series`` sorted by date.
        - total_sales: sum of sales on that date, NaN skipped (``min_count=1``, so a
          date with only NaN gives NaN, not 0).
        - active_series: series whose first positive sale is on or before that date.
          This separates real growth from new products entering.

    Edge cases:
        - NaT dates are ignored.
        - Empty df -> empty frame with these columns.
    """
    first_sale = df[df["sales"] > 0].groupby(SERIES_COLUMNS, observed=True)["date"].min()
    total_sales = df.groupby("date")["sales"].sum(min_count=1)  # groupby drops NaT dates
    # Every first-sale date is a date of df, so counting new entrants per date and summing them up is enough.
    active_series = first_sale.value_counts().reindex(total_sales.index, fill_value=0).cumsum()
    return pd.DataFrame({
        "date": total_sales.index,
        "total_sales": total_sales.to_numpy(),
        "active_series": active_series.to_numpy(dtype=int),
    })


def seasonal_profile(df: pd.DataFrame, frequency: str | None) -> pd.DataFrame:
    """Average normalised sales by position in the season.

    Position: ISO week of year for "W", day of week (0=Mon) for "D", month for "M"/"ME"/"MS".

    Returns:
        DataFrame ``position, mean_index, n_obs`` sorted by position. ``mean_index`` is
        the mean of sales / series mean (over each series' active span), so 1.0 means
        an average period and series of different sizes count equally.

    Edge cases:
        - Unsupported or None frequency -> empty frame with these columns.
        - Only active-span rows are used; never-sold series (mean 0 or NaN) are excluded.
        - ISO week 53 exists in some years; it is kept as its own position with fewer obs.
        - NaN sales are skipped.
    """
    columns = ["position", "mean_index", "n_obs"]
    base = frequency.split("-")[0] if frequency else None
    if base not in SEASONAL_PERIODS or df.empty:
        return pd.DataFrame(columns=columns)

    sid = df.groupby(SERIES_COLUMNS, sort=False).ngroup()
    first_sale = df["date"].where(df["sales"] > 0).groupby(sid).transform("min")
    sales = df["sales"].where(df["date"] >= first_sale)
    mean = sales.groupby(sid).transform("mean")
    index = sales / mean.where(mean > 0)

    dates = df["date"]
    if base == "W":
        position = dates.dt.isocalendar().week.astype("Int64")
    elif base == "D":
        position = dates.dt.dayofweek
    else:
        position = dates.dt.month

    out = index.groupby(position).agg(["mean", "count"])
    out = out[out["count"] > 0]
    return pd.DataFrame({
        "position": out.index.to_numpy(dtype=int),
        "mean_index": out["mean"].to_numpy(),
        "n_obs": out["count"].to_numpy(dtype=int),
    })

