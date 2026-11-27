import pytest

from demand_platform.frequency import DAILY, MONTHLY, WEEKLY, get_profile


def test_get_profile():
    assert get_profile("W-MON") is WEEKLY and get_profile("W-SUN") is WEEKLY
    assert get_profile("D") is DAILY
    assert get_profile("MS") is MONTHLY and get_profile("ME") is MONTHLY
    assert get_profile(None) is None and get_profile("h") is None and get_profile("QS") is None


def test_season_lengths_match_eda_contract():
    assert (WEEKLY.season_length, DAILY.season_length, MONTHLY.season_length) == (52, 7, 12)
    assert (WEEKLY.year_length, DAILY.year_length, MONTHLY.year_length) == (52, 364, 12)


@pytest.mark.parametrize("profile", [WEEKLY, DAILY, MONTHLY], ids=lambda p: p.name)
def test_profile_is_consistent(profile):
    year = profile.year_length
    assert "year" in profile.seasons and profile.primary_season in profile.seasons
    assert set(profile.fourier_orders) <= set(profile.seasons)
    for season, order in profile.fourier_orders.items():
        assert 1 <= order <= profile.seasons[season] // 2  # above season/2 the terms alias
    assert all(0 < lag < year for lag in profile.short_lags)
    assert any(lag == year for lag in profile.seasonal_lags)
    assert all(w <= year for w in (*profile.rolling_windows, *profile.ewm_spans, profile.shape_window))
    assert 2 * profile.trend_window <= year
    assert 0 < profile.last_year_half_width < year
    assert profile.spike_min_obs <= profile.shape_window
    assert all(0 < k < year for k in profile.difference_lags)
