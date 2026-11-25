from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_platform.analytics.eda import MIN_SEASONAL_CYCLES, _seasonal_period
from demand_platform.data.validation import ValidationReport

# Poisson has var / mean = 1; above this a count model needs an extra dispersion parameter.
OVERDISPERSION = 1.0


@dataclass(frozen=True)
class CapabilityReport:
    frequency: str | None
    seasonal_period: int | None
    n_series: int
    n_never_sold: int
    median_periods: float
    share_one_cycle: float
    share_two_cycles: float
    n_seasonal_eligible: int
    median_seasonal_acf: float
    count_data: bool
    zero_share: float
    median_adi: float
    share_overdispersed: float


def history_capability(profile: pd.DataFrame, seasonal_period: int | None) -> dict:
    """How much history the dataset has, measured in seasonal cycles.

    Why: decides whether seasonal lag features (lag = period) can be fed to the global
    model, how long the training window can be, and how many series are cold starts.

    Args:
        profile: Output of ``series_profile`` (one row per series).
        seasonal_period: Rows per season (e.g. 52 for weekly), or None if unsupported.

    Returns:
        dict with keys:
        - n_series: rows in ``profile``.
        - n_never_sold: series with ``n_periods == 0`` (no positive sale ever).
        - median_periods: median ``n_periods`` over sold series (never-sold excluded,
          they would drag the median to 0); NaN if none are sold.
        - share_one_cycle: share of all series with ``n_periods >= seasonal_period``.
        - share_two_cycles: same with ``>= MIN_SEASONAL_CYCLES * seasonal_period``.
          Both shares use ``n_series`` as denominator, so never-sold series count as
          "not covered".

    Vectorised algorithm:
        1. ``n = profile["n_periods"]``.
        2. ``sold = n > 0``; ``n_never_sold = (~sold).sum()``.
        3. ``median_periods = n[sold].median()`` (pandas gives NaN for an empty series).
        4. If ``seasonal_period`` is None, both shares are NaN.
        5. Otherwise ``share_one_cycle = (n >= seasonal_period).mean()`` — the mean of a
           boolean Series is the share of True — and likewise for two cycles.

    Readable alternative and its problem:
        ``for _, row in profile.iterrows(): if row.n_periods >= period: covered += 1``
        then divide by ``len(profile)``. It reads like the definition, but ``iterrows``
        builds one Series per row (slow, and it upcasts ints to float), and an empty
        profile divides by zero and raises instead of giving NaN. ``(n >= p).mean()``
        on an empty Series returns NaN on its own.

    Edge cases:
        - Empty profile: n_series 0, n_never_sold 0, everything else NaN.
        - ``seasonal_period`` None: shares NaN, never 0 (unknown is not "no history").
    """
    n = profile["n_periods"]
    sold = n > 0
    if seasonal_period is None:
        share_one_cycle = share_two_cycles = np.nan
    else:
        share_one_cycle = (n >= seasonal_period).mean()
        share_two_cycles = (n >= MIN_SEASONAL_CYCLES * seasonal_period).mean()
    return {
        "n_series": len(profile),
        "n_never_sold": int((~sold).sum()),
        "median_periods": n[sold].median(),
        "share_one_cycle": share_one_cycle,
        "share_two_cycles": share_two_cycles,
    }


def seasonality_capability(profile: pd.DataFrame) -> dict:
    """Evidence of seasonality across series that are long enough to measure it.

    Why: a lag-``period`` feature only helps if sales actually repeat across seasons.
    ``series_profile`` already left ``seasonal_acf`` NaN where a series has fewer than
    ``MIN_SEASONAL_CYCLES`` cycles or the frequency is unsupported, so NaN here means
    "not measurable", not "no seasonality".

    Args:
        profile: Output of ``series_profile``.

    Returns:
        dict with keys:
        - n_seasonal_eligible: series with a non-NaN ``seasonal_acf``.
        - median_seasonal_acf: median over those series; NaN if there are none.

    Vectorised algorithm:
        1. ``acf = profile["seasonal_acf"].dropna()``.
        2. ``n_seasonal_eligible = len(acf)``; ``median_seasonal_acf = acf.median()``.

    Readable alternative and its problem:
        ``values = [r for r in profile.seasonal_acf if r]`` then ``statistics.median``.
        It looks like it filters out missing values, but ``bool(nan)`` is True, so every
        NaN survives the filter, and a genuine 0.0 correlation is dropped. The median
        then comes out NaN or biased. Missing values must be tested with ``isna``, never
        by truthiness or ``== np.nan`` (which is always False).

    Edge cases:
        - All NaN or empty profile: n_seasonal_eligible 0, median NaN.
        - Negative ACF values are kept; they are evidence too.
    """
    acf = profile["seasonal_acf"].dropna()
    return {"n_seasonal_eligible": len(acf), "median_seasonal_acf": acf.median()}


def distribution_capability(profile: pd.DataFrame, report: ValidationReport) -> dict:
    """Shape of demand, which picks the sampling distribution of the probabilistic model.

    Why: integer, non-negative sales allow a count likelihood (Poisson or negative
    binomial); many zeros and long gaps between sales (high ADI) mean the distribution
    must put real mass on zero; var/mean > 1 rules out plain Poisson.

    Args:
        profile: Output of ``series_profile``.
        report: ``ValidationReport`` of the same data.

    Returns:
        dict with keys:
        - count_data: ``report.negative_sales == 0 and report.non_integer_sales == 0``.
        - zero_share: pooled share of zeros over all known active-span values, i.e.
          zeros / known values across the whole dataset, not a mean of per-series shares.
        - median_adi: median ``adi`` over series where it is defined.
        - share_overdispersed: share of series with ``dispersion > OVERDISPERSION``,
          among series where ``dispersion`` is defined.

    Vectorised algorithm:
        1. Known values per series: ``known = n_periods - missing_sales``.
        2. Zeros per series: ``zeros = zero_share * known`` (``zero_share`` is NaN when
           ``known`` is 0, so use ``zero_share.fillna(0)`` here: that series adds 0 zeros
           and 0 known values).
        3. ``zero_share = zeros.sum() / known.sum()``; NaN if ``known.sum() == 0``.
        4. ``median_adi = profile["adi"].median()`` (pandas skips NaN).
        5. ``d = profile["dispersion"].dropna()``;
           ``share_overdispersed = (d > OVERDISPERSION).mean()``.

    Readable alternative and its problem:
        ``zero_share = profile["zero_share"].mean()`` is one short line, but it gives a
        series with 10 known weeks the same weight as one with 157. A short new product
        with 9 zeros out of 10 moves the result as much as a long, steady series.
        Pooling (step 1-3) weights every observed period equally, which is what the
        likelihood will actually see during training.
        For step 5, ``(profile["dispersion"] > OVERDISPERSION).mean()`` without
        ``dropna`` counts NaN as "not overdispersed" (``NaN > 1`` is False) and pulls the
        share down.

    Edge cases:
        - Empty profile: zero_share, median_adi, share_overdispersed NaN.
        - count_data comes only from the report; the profile cannot see fractions.
    """
    known = profile["n_periods"] - profile["missing_sales"]
    zeros = profile["zero_share"].fillna(0) * known
    total_known = known.sum()
    dispersion = profile["dispersion"].dropna()
    return {
        "count_data": report.negative_sales == 0 and report.non_integer_sales == 0,
        "zero_share": zeros.sum() / total_known if total_known > 0 else np.nan,
        "median_adi": profile["adi"].median(),
        "share_overdispersed": (dispersion > OVERDISPERSION).mean(),
    }


def assess_capability(profile: pd.DataFrame, report: ValidationReport) -> CapabilityReport:
    """Combine the three assessments into one report; the frequency comes from ``report``."""
    period = _seasonal_period(report.frequency)
    return CapabilityReport(
        frequency=report.frequency,
        seasonal_period=period,
        **history_capability(profile, period),
        **seasonality_capability(profile),
        **distribution_capability(profile, report),
    )
