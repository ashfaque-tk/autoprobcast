"""Per-series scaling so one global model learns patterns, not sizes.

Without scaling, a series selling 80 units a week has errors 40× larger than one selling 2,
so it dominates the loss and the model spends its capacity on level differences between
series. Dividing every series by its own annual demand puts them on one comparable scale:
a +5 uplift means something different at 2/week than at 80/week, and scaled it shows that.
(VN2 winner's report, "Scaling".)

How mlforecast uses a target transform (``BaseTargetTransform``, verified on 1.0.2):
- ``fit(df)`` calls ``fit_transform(df)`` on the training frame (``unique_id, ds, y``, sorted)
  *before* building features, so every lag and rolling feature is computed on the scaled y.
- ``predict`` / ``cross_validation`` call ``inverse_transform(preds)`` at the end, where
  ``preds`` has ``unique_id, ds`` and one column per model, all in scaled units.
- During recursive ``predict``, the ``after_predict_callback`` runs *before* the inverse, so
  it sees scaled values. A sampling callback must unscale (``× scale_for(ids)``), draw a
  count, and scale back.
- ``cross_validation`` re-runs ``fit_transform`` on each window's training data, so the
  scale never uses values after the forecast origin.

Difference from the report: the report recomputes the scale every week (a rolling 53-week
mean). Here it is fixed per series at fit time, from the last year before the origin. Simpler,
and recursive sampling needs one scale per series. Within the training rows, early weeks are
therefore scaled by a level measured later; this is a normaliser, not a feature, and the
backtest origin is never crossed. A dynamic scale is a candidate for later.
"""

import numpy as np
import pandas as pd
from mlforecast.target_transforms import BaseTargetTransform

# The report needs 45 in-stock observations out of a 53-week window before trusting the
# yearly mean (≈ 85 %); fewer than that uses the warm start.
WARM_START_SHARE = 45 / 53


class AnnualScale(BaseTargetTransform):
    """Divide each series by ``max(year_length × mean(last year), 1)``: its annual demand.

    Args:
        year_length: Rows per year (``FrequencyProfile.year_length``: 52 weekly, 364 daily).
        min_obs: Known values needed in the last year to use it; default
            ``round(WARM_START_SHARE * year_length)`` (44 for weekly data).

    Attributes after ``fit_transform``:
        scale_: ``pd.Series`` of scale factors indexed by ``unique_id``.
    """

    def __init__(self, year_length: int, min_obs: int | None = None):
        self.year_length = year_length
        self.min_obs = round(WARM_START_SHARE * year_length) if min_obs is None else min_obs

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn one scale per series from ``df`` and return a copy with ``y`` divided by it.

        Per series (rows are sorted by ``self.id_col``, ``self.time_col``):
            level = mean of the known ``y`` in the last ``year_length`` rows, if that window
                    holds at least ``min_obs`` known values;
                    otherwise the mean of all known ``y`` (warm start: new products).
            scale = max(year_length × level, 1).
        The clip at 1 avoids dividing by ~0 for series that barely sell (and NaN for a series
        with no known value).

        Use ``self.id_col``, ``self.time_col``, ``self.target_col`` (set by mlforecast via
        ``set_column_names``), never hard-coded names. Never modify ``df`` in place.

        Vectorised algorithm:
            1. ``g = df.groupby(id_col)[target_col]``.
            2. Last year per series: ``recent = df[g.cumcount(ascending=False) < year_length]``
               (cumcount from the end numbers each series' rows 0, 1, … backwards).
            3. ``recent_mean``, ``recent_n`` = mean and count of ``recent`` by id;
               ``all_mean`` = ``g.mean()``.
            4. ``level = recent_mean.where(recent_n >= min_obs, all_mean)``.
            5. ``self.scale_ = (year_length * level).clip(lower=1).fillna(1)``.
            6. Return ``df`` with ``y / df[id_col].map(self.scale_)``.

        Readable alternative and its problem:
            ``for uid, s in df.groupby(id_col): ...`` building a dict, then a second loop
            assigning ``df.loc[df.id == uid, "y"] /= scale``. Correct but O(series × rows):
            each ``.loc`` scan is over the whole frame, so 30k M5 series means 30k full scans.
            ``tail(year_length)`` per group is fine; ``iloc[-52:]`` on the *whole* frame is
            the classic bug (it takes the last 52 rows of the last series only).
        """
        raise NotImplementedError

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multiply every prediction column (all but ``id_col`` and ``time_col``) by the series' scale.

        Returns a copy; ``df`` is not modified.

        Vectorised algorithm:
            ``factor = df[id_col].map(self.scale_).to_numpy()``, then for each prediction
            column ``out[col] = df[col] * factor``. mlforecast may add interval columns
            (``model-lo-80``), so transform every non-key column, not a fixed list.

        Readable alternative and its problem:
            ``df.merge(scale_frame, on=id_col)`` then multiply: also correct, but a merge can
            reorder rows; mlforecast expects the output in the same row order as the input.
        """
        raise NotImplementedError

    def scale_for(self, ids) -> np.ndarray:
        """Scale factors for an array of ``unique_id`` values, in that order (for sampling callbacks)."""
        return pd.Series(ids).map(self.scale_).to_numpy(dtype=float)
