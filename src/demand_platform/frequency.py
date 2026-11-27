"""Every frequency-specific number in one place.

Windows and lags are counted in rows, so "a quarter" is 13 rows for weekly data but 91 for
daily. Each supported frequency gets an explicit profile; anything else gets ``None`` and
callers switch seasonal behaviour off instead of guessing.

Only the weekly profile is checked against data (VN2, and the VN2 winner's report). Daily and
monthly values are reasoned defaults until a dataset at that frequency is tested.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrequencyProfile:
    name: str
    unit: str
    seasons: dict[str, int]
    primary_season: str
    short_lags: tuple[int, ...]
    seasonal_lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    ewm_spans: tuple[int, ...]
    shape_window: int
    trend_window: int
    last_year_half_width: int
    difference_lags: tuple[int, ...]
    spike_min_obs: int
    fourier_orders: dict[str, int]

    @property
    def season_length(self) -> int:
        """Rows in the primary season: what EDA and capability call "the seasonal period"."""
        return self.seasons[self.primary_season]

    @property
    def year_length(self) -> int:
        """Rows in a year: the window for scaling and the block for recency weights."""
        return self.seasons["year"]


WEEKLY = FrequencyProfile(
    name="Weekly", unit="weeks",
    seasons={"year": 52}, primary_season="year",
    short_lags=(1, 2, 3, 4), seasonal_lags=(51, 52, 53),
    rolling_windows=(3, 5, 13, 26), ewm_spans=(5, 10),
    shape_window=13, trend_window=13, last_year_half_width=2,
    difference_lags=(1, 4, 5), spike_min_obs=8,
    fourier_orders={"year": 3},
)

# 364 = 52 weeks, so "one year ago" falls on the same weekday.
DAILY = FrequencyProfile(
    name="Daily", unit="days",
    seasons={"week": 7, "year": 364}, primary_season="week",
    short_lags=(1, 2, 3, 4, 5, 6, 7), seasonal_lags=(7, 14, 364),
    rolling_windows=(7, 14, 28, 91), ewm_spans=(7, 28),
    shape_window=28, trend_window=91, last_year_half_width=3,
    difference_lags=(1, 7, 28), spike_min_obs=28,
    fourier_orders={"week": 2, "year": 3},
)

MONTHLY = FrequencyProfile(
    name="Monthly", unit="months",
    seasons={"year": 12}, primary_season="year",
    short_lags=(1, 2, 3), seasonal_lags=(11, 12, 13),
    rolling_windows=(3, 6, 12), ewm_spans=(3, 6),
    shape_window=6, trend_window=3, last_year_half_width=1,
    difference_lags=(1, 3), spike_min_obs=6,
    fourier_orders={"year": 3},
)

PROFILES = {"W": WEEKLY, "D": DAILY, "M": MONTHLY, "ME": MONTHLY, "MS": MONTHLY}


def get_profile(frequency: str | None) -> FrequencyProfile | None:
    """Profile for a pandas frequency alias ("W-MON" -> WEEKLY); None if unsupported or None."""
    if frequency is None:
        return None
    return PROFILES.get(frequency.split("-")[0])
