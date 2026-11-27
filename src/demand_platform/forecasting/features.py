"""Sales-only features for the global model, expressed as mlforecast configuration.

How mlforecast builds features (verified on mlforecast 1.0.2):

- ``lags=[1, 2]`` adds ``lag1``, ``lag2``: the target shifted by 1 and 2 rows within each series.
- ``lag_transforms={1: [f, (g, 13)]}`` first shifts each series by 1 row, then calls
  ``f(x)`` and ``g(x, 13)`` on that shifted array, **one series at a time**. ``x[t]`` is
  therefore ``y[t-1]``, and the first ``lag`` values of ``x`` are NaN. The result must be an
  array of the same length; column names come from the function and its arguments
  (``nonzero_rate_lag1_window13``). Because of the shift, a transform may use ``x[t]``
  itself and still only sees the past.
- ``date_features=[func, "month"]`` calls ``func(dates)`` with a ``DatetimeIndex`` (or reads
  the pandas attribute) and names the column ``func.__name__``. These depend only on the
  calendar, so future values are known.
- Static features (``fit(df, static_features=[...])``) are repeated unchanged for future rows.
- ``target_transforms`` (see ``scaling.py``) run *before* any feature, so every lag and lag
  transform sees the scaled target.
- Recursive ``predict(h)``: for each step it appends the new value to every series and
  **calls each lag transform again on the full history**, keeping only the last element.
  So a transform runs ``h × n_paths`` times per series: it must be vectorised (no Python
  loop over t) and causal (``out[t]`` depends only on ``x[:t+1]``). Changing ``x[t:]`` must
  never change ``out[:t]``; ``tests/test_features.py`` checks this for every transform.

Every custom transform must treat NaN as unknown, never as zero: the leading NaN from the
shift (and, later, weeks masked as out of stock) would otherwise count as "no sale".

All windows and lags come from ``FrequencyProfile`` (``demand_platform/frequency.py``); the
constants below are the only frequency-free choices.

Design follows the VN2 winner's report (Szabłowski, 2026), adapted: features are computed on
the scaled target; momentum is a difference; spikes use a robust z-score; seasonality uses
seasonal lags, a last-year window, Fourier terms, week of year and a causal seasonal
correlation. Differences: we forecast recursively (joint sample paths) rather than with one
model per horizon, and keep LightGBM's native NaN handling instead of median imputation
(imputation is an ablation candidate).
"""

import operator

import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import Combine, ExponentiallyWeightedMean, RollingMean, RollingQuantile, RollingStd

from demand_platform.forecasting.dataset import STATIC_COLUMNS
from demand_platform.frequency import FrequencyProfile, get_profile

# Robust z-score above which a value is a spike. 3.5 is the usual cut-off for median/MAD
# z-scores (Iglewicz & Hoaglin); it is unit-free, so it doesn't depend on the frequency.
ROBUST_Z = 3.5
# Scales MAD to the standard deviation of a normal distribution, so ROBUST_Z reads like a z-score.
MAD_TO_STD = 1.4826
# Average calendar year; day-of-year Fourier terms divide by it so leap years stay on cycle.
DAYS_PER_YEAR = 365.25

ALL_GROUPS = ("lags", "rolling", "shape", "trend", "intermittency", "seasonal", "calendar", "ids")


# --- custom lag transforms (user implements) -----------------------------------------

def nonzero_rate(x: np.ndarray, window: int) -> np.ndarray:
    """Share of periods with a sale among the known values in the last ``window`` positions.

    Args:
        x: One series' target, already shifted by the lag (leading NaN).
        window: Number of positions ending at ``t`` (``x[t-window+1 : t+1]``).

    Returns:
        float array, same length. ``out[t]`` = (# known values > 0) / (# known values) in
        the window; NaN when the window holds no known value.

    Vectorised algorithm:
        1. ``known = ~np.isnan(x)``; ``sold = known & (x > 0)`` (comparisons with NaN warn,
           so compare on ``np.nan_to_num(x)`` or mask first).
        2. Cumulative sums with a leading 0: ``cs_sold = np.concatenate([[0], np.cumsum(sold)])``
           and likewise ``cs_known``.
        3. Window sums for every t at once: ``cs[t+1] - cs[max(t+1-window, 0)]``. With
           ``idx = np.arange(len(x))``: ``hi = idx + 1``, ``lo = np.maximum(hi - window, 0)``.
        4. ``out = n_sold / n_known`` where ``n_known > 0``, else NaN.

    Readable alternative and its problem:
        ``[np.mean(x[max(0, t-window+1):t+1] > 0) for t in range(len(x))]``. Two bugs and a
        cost: NaN counts as "no sale", so the first windows are biased toward 0; an empty
        known window gives 0 instead of NaN; and it is O(n·window) in Python per call, called
        again at every recursive step for every path.
        (``pd.Series(...).rolling(window, min_periods=1).mean()`` on a 0/1/NaN series is
        correct too and simpler; the cumsum version shows how rolling sums work.)
    """
    raise NotImplementedError


def periods_since_last_sale(x: np.ndarray) -> np.ndarray:
    """Positions since the most recent ``x > 0`` at or before ``t``.

    Returns:
        float array, same length. 0 where ``x[t] > 0``; NaN before the first sale (or for
        leading NaN). Unknown values (NaN) don't reset the count; they are skipped.

    Vectorised algorithm:
        1. ``idx = np.arange(len(x))``.
        2. ``last = np.where(x > 0, idx, -1)`` (mask NaN first), then
           ``last = np.maximum.accumulate(last)``: a running maximum carries the index of the
           latest sale forward.
        3. ``out = (idx - last).astype(float)``; ``out[last < 0] = np.nan``.

    Readable alternative and its problem:
        For each t, scan backwards until ``x[k] > 0``. Correct, but O(n²) per call in the
        worst case (long zero runs are exactly what intermittent data has), repeated at
        every recursive step. The running maximum does it in one O(n) pass.
    """
    raise NotImplementedError


def periods_since_robust_spike(x: np.ndarray, window: int, min_obs: int) -> np.ndarray:
    """Positions since the last spike, where a spike is judged against the previous ``window`` values.

    Spike at ``t``: ``x[t]`` is known, the previous ``window`` positions hold at least
    ``min_obs`` known values, and ``x[t] > med + ROBUST_Z * MAD_TO_STD * mad``, where ``med``
    is their median and ``mad`` the median of ``|value - med|`` (NaN ignored).

    Why median/MAD and not mean/std: a single huge week inflates the std and hides the next
    spike; the median barely moves. When most past values are 0, ``med`` and ``mad`` are 0
    and *any* sale counts as a spike. That is intended (as in the VN2 winner's report): for
    sparse series this feature becomes "time since a non-zero week", and ``nonzero_rate``
    tells sparse series apart from low but regular ones.

    Returns:
        float array, same length. 0 at a spike; NaN until the first spike.

    Vectorised algorithm:
        1. Previous values only: ``prev = np.concatenate([[np.nan], x[:-1]])``.
        2. All windows at once: pad ``prev`` with ``window - 1`` NaN at the front, then
           ``w = np.lib.stride_tricks.sliding_window_view(padded, window)``: shape
           ``(len(x), window)``, row t = ``prev[t-window+1 : t+1]``, i.e. ``x[t-window : t]``.
           It is a view: no copy.
        3. ``n = (~np.isnan(w)).sum(axis=1)``; ``med = np.nanmedian(w, axis=1)``;
           ``mad = np.nanmedian(np.abs(w - med[:, None]), axis=1)``
           (silence the all-NaN warning with ``warnings.catch_warnings()``).
        4. ``spike = known & (n >= min_obs) & (x > med + ROBUST_Z * MAD_TO_STD * mad)``.
        5. Running maximum of spike indices, as in ``periods_since_last_sale``.

    Readable alternative and its problem:
        A loop computing ``np.median`` of ``x[t-window:t]`` for each t: O(n·window·log window)
        in Python, called again at every recursive step. pandas' ``rolling().median()`` covers
        the median but has no rolling MAD, which is why the sliding-window view is used.
        Judging spikes with the whole series' mean/std (as EDA does) uses future values to
        label a past week, which leaks.
    """
    raise NotImplementedError


def difference(x: np.ndarray, k: int) -> np.ndarray:
    """Change over ``k`` positions: ``out[t] = x[t] - x[t-k]``; NaN if either is unknown.

    On the scaled target this is the report's momentum Δk, and Δk / k is its "slope proxy":
    the mean of the last k first differences telescopes to ``(x[t] - x[t-k]) / k``. Trees
    split the same way on Δk and Δk / k, so the slope needs no separate feature.

    Returns:
        float array, same length; the first ``k`` values are NaN.

    Vectorised algorithm:
        ``out = np.full(len(x), np.nan)``; ``out[k:] = x[k:] - x[:-k]`` (NaN propagates).
        Guard ``k >= len(x)``: everything NaN.

    Readable alternative and its problem:
        ``pd.Series(x).diff(k)`` is the same thing and fine. The trap is ``np.diff(x, k)``:
        its ``n`` argument means "apply the first difference n times", not "lag k", and the
        result is shorter than ``x``.
    """
    raise NotImplementedError


def seasonal_correlation(x: np.ndarray, season: int, min_pairs: int) -> np.ndarray:
    """Expanding Pearson correlation between ``x[t]`` and ``x[t-season]`` over all pairs up to ``t``.

    The report's "seasonality strength", made causal: at row t only pairs ending at or before
    t are used, so the value never depends on the future. The EDA ``seasonal_acf`` is the
    same statistic computed once over the whole series.

    Returns:
        float array, same length. NaN until ``min_pairs`` complete pairs exist, or when either
        side has zero variance so far.

    Vectorised algorithm:
        1. ``y = x`` shifted right by ``season`` (``np.full`` NaN, then ``y[season:] = x[:-season]``).
        2. ``pair = ~np.isnan(x) & ~np.isnan(y)``; set both to 0 where not a pair.
        3. Running sums over pairs: ``n = cumsum(pair)``, ``sx = cumsum(x)``, ``sy``,
           ``sxx = cumsum(x*x)``, ``syy``, ``sxy = cumsum(x*y)``.
        4. ``cov = sxy - sx*sy/n``; ``vx = sxx - sx²/n``; ``vy = syy - sy²/n``;
           ``r = cov / sqrt(vx * vy)``, inside ``np.errstate``.
        5. NaN where ``n < min_pairs`` or ``vx * vy <= 0``.

    Readable alternative and its problem:
        ``[np.corrcoef(x[season:t+1], x[:t+1-season])[0, 1] for t in ...]``: O(n²) and called
        at every recursive step; ``corrcoef`` also returns NaN as soon as one value is NaN.
        Computing it once on the full series (like EDA) and attaching it as a static feature
        is the quick version, but leaks: early rows would see later years.
        Numerical note: the running-sum formula can lose precision for huge values; on the
        scaled target (values around 1/season) it is safe.
    """
    raise NotImplementedError


# --- calendar (user implements) ------------------------------------------------------

def fourier(dates: pd.DatetimeIndex, season: str, order: int, kind: str) -> np.ndarray:
    """Fourier term ``sin`` or ``cos`` of ``2π · order · position``, position in [0, 1) within the season.

    Smooth calendar curves the model can combine into any seasonal shape. The position comes
    from the calendar, not from row counts, so the same code works for any frequency:
    - ``season="year"``: ``(day_of_year - 1) / DAYS_PER_YEAR``. No week-53 problem.
    - ``season="week"``: ``day_of_week / 7`` (0 = Monday); only meaningful for daily data.

    Args:
        dates: Dates mlforecast passes to a date feature.
        season: ``"year"`` or ``"week"``.
        order: Harmonic 1, 2, 3, … (higher means faster wiggles within the season).
        kind: ``"sin"`` or ``"cos"``.

    Returns:
        float array in [-1, 1], one value per date.

    Raises:
        ValueError: for an unknown ``season`` or ``kind``.

    Vectorised algorithm:
        ``angle = 2 * np.pi * order * position``, ``position`` from ``dates.day_of_year`` or
        ``dates.dayofweek`` as arrays, then ``np.sin(angle)`` or ``np.cos(angle)``.

    Readable alternative and its problem:
        ``[math.sin(2*math.pi*order*d.isocalendar().week/52) for d in dates]``: a Python loop
        (slow, but date features only run once per step); the real problem is the week
        number. 53-week years break the cycle, and it means nothing for daily data.
    """
    raise NotImplementedError


def week_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """ISO week number (1–53); captures sharp single-week peaks like VN2's week 51."""
    return dates.isocalendar().week.to_numpy(dtype=int)


def _date_feature(func, name: str, **kwargs):
    """Fix ``kwargs`` and give the callable a column name (mlforecast uses ``__name__``)."""
    def feature(dates):
        return func(dates, **kwargs)
    feature.__name__ = name
    return feature


# --- wiring --------------------------------------------------------------------------

def feature_groups(profile: FrequencyProfile | None) -> dict[str, dict]:
    """Feature groups as mlforecast arguments; ablation switches whole groups on or off.

    Each value has any of the keys ``lags``, ``lag_transforms``, ``date_features``,
    ``static``. With ``profile=None`` (unsupported frequency) only frequency-free groups
    remain ("intermittency" without its windowed parts, "ids"); everything that needs a
    window length is left out rather than guessed.

    Groups (p = profile):
        - lags: ``p.short_lags`` and ``p.seasonal_lags`` (e.g. 1–4 and 51, 52, 53 weekly).
        - rolling: mean, median and std over each ``p.rolling_windows``; exponentially
          weighted means with ``alpha = 2 / (span + 1)`` for each ``p.ewm_spans`` (the usual
          span → alpha conversion; a span of 5 averages roughly the last 5 periods).
        - shape: interquartile range over ``p.shape_window``.
        - trend: ``difference`` for each ``p.difference_lags``.
        - intermittency: ``nonzero_rate`` and ``periods_since_robust_spike`` over
          ``p.shape_window``, and ``periods_since_last_sale``.
        - seasonal: last year's window and ``seasonal_correlation`` per season. Last year's
          window uses only built-ins: registered at lag ``year - h`` (h = half-width),
          ``RollingMean(2h + 1)`` averages ``y[t-year-h … t-year+h]``, i.e. the same periods
          one year ago, ± h. ``seasonal_correlation`` needs one season of pairs
          (``min_pairs = season``), i.e. two seasons of history, like EDA.
        - calendar: ``week_of_year`` (weekly and daily data), month, and Fourier sin/cos up to
          ``p.fourier_orders[season]`` per season.
        - ids: ``store_id``, ``product_id`` as static categoricals.
    """
    if profile is None:
        return {"lags": {}, "rolling": {}, "shape": {}, "trend": {},
                "intermittency": {"lag_transforms": {1: [periods_since_last_sale]}},
                "seasonal": {}, "calendar": {}, "ids": {"static": list(STATIC_COLUMNS)}}
    p = profile
    h = p.last_year_half_width
    return {
        "lags": {"lags": [*p.short_lags, *p.seasonal_lags]},
        "rolling": {"lag_transforms": {1: [
            *(RollingMean(w) for w in p.rolling_windows),
            *(RollingQuantile(0.5, w) for w in p.rolling_windows),
            *(RollingStd(w) for w in p.rolling_windows),
            *(ExponentiallyWeightedMean(alpha=2 / (span + 1)) for span in p.ewm_spans),
        ]}},
        "shape": {"lag_transforms": {1: [
            Combine(RollingQuantile(0.75, p.shape_window), RollingQuantile(0.25, p.shape_window), operator.sub),
        ]}},
        "trend": {"lag_transforms": {1: [(difference, k) for k in p.difference_lags]}},
        "intermittency": {"lag_transforms": {1: [
            (nonzero_rate, p.shape_window),
            periods_since_last_sale,
            (periods_since_robust_spike, p.shape_window, p.spike_min_obs),
        ]}},
        "seasonal": {"lag_transforms": {
            p.year_length - h: [RollingMean(2 * h + 1)],
            1: [(seasonal_correlation, length, length) for length in p.seasons.values()],
        }},
        "calendar": {"date_features": [
            *([week_of_year] if p.unit in ("weeks", "days") else []),
            "month",
            *(_date_feature(fourier, f"fourier_{season}_{kind}_{k}", season=season, order=k, kind=kind)
              for season, order in p.fourier_orders.items()
              for k in range(1, order + 1) for kind in ("sin", "cos")),
        ]},
        "ids": {"static": list(STATIC_COLUMNS)},
    }


def build_forecaster(models: dict, frequency: str, groups=ALL_GROUPS,
                     target_transforms: list | None = None) -> tuple[MLForecast, list[str]]:
    """An ``MLForecast`` with the selected feature groups merged.

    Args:
        models: ``{"name": sklearn-style regressor}``, e.g. ``{"lgb": LGBMRegressor(objective="poisson")}``.
        frequency: Confirmed pandas frequency (e.g. ``"W-MON"``); sets mlforecast's ``freq``
            and, through ``get_profile``, every window and lag.
        groups: Names from ``ALL_GROUPS``.
        target_transforms: e.g. ``[AnnualScale(...)]`` from ``scaling.py``.

    Returns:
        ``(forecaster, static_features)``. Then call
        ``forecaster.fit(df, static_features=static_features, dropna=False, weight_col=...)``:
        - ``static_features`` is empty when "ids" is off, and the id columns must then be
          dropped from ``df`` (mlforecast treats any extra column as a feature).
        - ``dropna=False`` is required: the default drops every row with a NaN feature, and
          with lag 53 that silently removes about the first year of every series.

    Raises:
        ValueError: for an unknown group name.
    """
    available = feature_groups(get_profile(frequency))
    unknown = set(groups) - set(available)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    lags, lag_transforms, date_features, static = [], {}, [], []
    for name in groups:
        group = available[name]
        lags += group.get("lags", [])
        for lag, transforms in group.get("lag_transforms", {}).items():
            lag_transforms.setdefault(lag, []).extend(transforms)
        date_features += group.get("date_features", [])
        static += group.get("static", [])
    forecaster = MLForecast(
        models=models, freq=frequency, lags=sorted(set(lags)) or None,
        lag_transforms=lag_transforms or None, date_features=date_features or None,
        target_transforms=target_transforms,
    )
    return forecaster, static
