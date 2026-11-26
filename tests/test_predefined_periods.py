"""Calendar periods and static availability for period-based predefined layers."""

import pytest

from gui.scripts.predefined.periods import (
    MONTHS,
    availability_error,
    availability_fields,
    available_years,
    lookback_days,
    lookback_years,
    month_index,
    months_spanned,
    period_needs,
    period_param,
    period_range,
    period_start,
    source_window,
    window_covered,
)

# MCD64A1 / VNP64A1 complete months, verified 2026-10-09.
BURN = {"modis": ("2000-11", "2026-07"), "viirs": ("2012-03", "2026-08")}


def test_period_param_shape():
    """Period param shape."""
    spec = period_param()
    assert spec["key"] == "period"
    assert spec["type"] == "choice"
    assert spec["default"] == "year"
    assert spec["options"] == ["year", *MONTHS]
    assert len(MONTHS) == 12
    assert spec["omit_default"] is True
    assert spec["advanced"] is False
    assert spec["suffix_prefix"] == ""
    assert spec["option_label_key_prefix"] == "vars.modal.period_"
    assert period_param(advanced=True)["advanced"] is True
    # Each entry gets its own list: mutating one must not leak into another.
    assert period_param()["options"] is not period_param()["options"]


@pytest.mark.parametrize(
    "year, period, expected",
    [
        (2020, "year", ("2020-01-01", "2021-01-01")),
        (2020, "jan", ("2020-01-01", "2020-02-01")),
        (2020, "feb", ("2020-02-01", "2020-03-01")),
        (2020, "dec", ("2020-12-01", "2021-01-01")),
    ],
)
def test_period_range_is_half_open(year, period, expected):
    """Period range is half open."""
    assert period_range(year, period) == expected


def test_period_start_and_month_index():
    """Period start and month index."""
    assert period_start(2024, "sep") == "2024-09-01"
    assert period_start(2024, "year") == "2024-01-01"
    assert month_index("year") is None
    assert month_index("jan") == 1
    assert month_index("dec") == 12
    with pytest.raises(ValueError):
        month_index("sept")


def test_lookback_days_ends_at_the_period_start():
    # 2024 is a leap year: 365 days before 2024-09-01 is 2023-09-02.
    """Lookback days ends at the period start."""
    assert lookback_days(2024, "sep", 365) == ("2023-09-02", "2024-09-01")
    assert lookback_days(2020, "year", 7) == ("2019-12-25", "2020-01-01")


def test_lookback_years_keeps_the_month():
    """Lookback years keeps the month."""
    assert lookback_years(2024, "sep", 10) == ("2014-09-01", "2024-09-01")
    assert lookback_years(2024, "year", 1) == ("2023-01-01", "2024-01-01")


def test_months_spanned_and_window_covered():
    """Months spanned and window covered."""
    assert months_spanned("2020-01-01", "2021-01-01") == ("2020-01", "2020-12")
    assert months_spanned("2023-09-02", "2024-09-01") == ("2023-09", "2024-08")
    assert window_covered(("2000-11", "2026-07"), ("2026-07", "2026-07"))
    assert not window_covered(("2000-11", "2026-07"), ("2026-01", "2026-12"))
    assert not window_covered(("2000-11", "2026-07"), ("2000-01", "2000-12"))


def test_source_window_picks_source_then_sensor_then_the_single_key():
    """Source window picks source then sensor then the single key."""
    assert source_window(BURN, {"sensor": "viirs"}) == BURN["viirs"]
    rain = {"era": ("1950-02", "2026-09"), "chirps": ("1981-01", "2026-08")}
    assert source_window(rain, {"source": "chirps"}) == rain["chirps"]
    single = {"era": ("1950-02", "2026-09")}
    assert source_window(single, {"period": "year"}) == single["era"]
    with pytest.raises(KeyError):
        source_window(BURN, {})


def test_available_years_stop_at_the_last_complete_month():
    """Available years stop at the last complete month."""

    def years(**params):
        return available_years(BURN, period_needs, params)

    assert years(period="year", sensor="modis") == list(range(2001, 2026))
    assert years(period="jul", sensor="modis")[-1] == 2026
    assert years(period="aug", sensor="modis")[-1] == 2025
    assert years(period="year", sensor="viirs")[0] == 2013
    assert years(period="mar", sensor="viirs")[0] == 2012


def test_available_years_for_a_lookback_window():
    """Available years for a lookback window."""

    def needs(year, params):
        return lookback_years(year, params["period"], 10)

    offered = available_years(BURN, needs, {"period": "year", "sensor": "modis"})
    # 2011 needs 2001-2010; 2026 needs 2016-2025 (both inside 2000-11..2026-07).
    assert offered == list(range(2011, 2027))


def test_availability_error_reports_the_gap():
    """Availability error reports the gap."""
    params = {"period": "year", "sensor": "modis"}
    assert availability_error(BURN, period_needs, params, 2025) is None
    assert availability_error(BURN, period_needs, params, None) is None
    assert availability_error(BURN, period_needs, params, 2026) == (
        "vars.modal.error_period_unavailable",
        {
            "year": 2026,
            "start": "2026-01",
            "end": "2026-12",
            "first": "2000-11",
            "last": "2026-07",
        },
    )


def test_availability_fields_bundle_reads_its_own_copy():
    """Availability fields bundle reads its own copy."""
    fields = availability_fields(BURN)
    assert fields["availability"] == BURN
    assert fields["availability"] is not BURN
    assert fields["years"]({"period": "jul", "sensor": "modis"})[-1] == 2026
    err = fields["validate"]({"period": "aug", "sensor": "modis"}, 2026)
    assert err[0] == "vars.modal.error_period_unavailable"
    # validate reads the entry's own window, so a bump is a one-place edit.
    fields["availability"]["modis"] = ("2000-11", "2026-08")
    assert fields["validate"]({"period": "aug", "sensor": "modis"}, 2026) is None
