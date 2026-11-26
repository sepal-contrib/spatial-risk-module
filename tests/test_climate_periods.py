"""Precipitation and temperature_2m by period, and the rainfall source.

Also freezes the pre-Part-D expressions their default names must keep building.
"""

import pytest
from _ee_fake import install

import gui.scripts.predefined.climate as climate
from gui.scripts.predefined import (
    build_predefined_name,
    entry_for_name,
    resolve_predefined,
    resolved_entry,
    validate_entry,
)

ERA_M = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
ERA_D = "ECMWF/ERA5_LAND/DAILY_AGGR"
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"

# The pre-Part-D builders' call chains, frozen from
# gui/scripts/predefined_variables.py at commit time of Part A's split.
OLD_PRECIPITATION_2020 = [
    ("ImageCollection", (ERA_M,)),
    ("filterDate", ("2020-01-01", "2021-01-01")),
    ("select", ("total_precipitation_sum",)),
    ("sum", ()),
    ("multiply", (1000,)),
    ("round", ()),
    ("toInt16", ()),
    ("clip", ("AOI",)),
    ("rename", ("B1",)),
]


def _old_temperature_2020(metric):
    return [
        ("ImageCollection", (ERA_M,)),
        ("filterDate", ("2020-01-01", "2021-01-01")),
        ("select", ("temperature_2m",)),
        (metric, ()),
        ("subtract", (273.15,)),
        ("round", ()),
        ("toInt16", ()),
        ("clip", ("AOI",)),
        ("rename", ("B1",)),
    ]


def test_default_precipitation_is_the_pre_part_d_expression(monkeypatch):
    """Default precipitation is the pre part d expression."""
    fake = install(monkeypatch, climate)
    key, params, entry = entry_for_name("precipitation")
    entry["get_image"]("AOI", 2020, **params)
    assert fake.calls == OLD_PRECIPITATION_2020


@pytest.mark.parametrize("metric", ["mean", "max", "min", "median"])
def test_default_temperature_is_the_pre_part_d_expression(monkeypatch, metric):
    """Default temperature is the pre part d expression."""
    fake = install(monkeypatch, climate)
    key, params, entry = entry_for_name(f"temperature_2m_{metric}")
    entry["get_image"]("AOI", 2020, **params)
    assert fake.calls == _old_temperature_2020(metric)


def test_monthly_precipitation_filters_the_month(monkeypatch):
    """Monthly precipitation filters the month."""
    fake = install(monkeypatch, climate)
    climate._get_precipitation("AOI", 2020, period="sep")
    assert fake.calls == [
        ("ImageCollection", (ERA_M,)),
        ("filterDate", ("2020-09-01", "2020-10-01")),
        ("select", ("total_precipitation_sum",)),
        ("sum", ()),
        ("multiply", (1000,)),
        ("round", ()),
        ("toInt16", ()),
        ("clip", ("AOI",)),
        ("rename", ("B1",)),
    ]


def test_chirps_precipitation_is_already_in_mm(monkeypatch):
    """Chirps precipitation is already in mm."""
    fake = install(monkeypatch, climate)
    climate._get_precipitation("AOI", 2020, period="dec", source="chirps")
    assert fake.calls == [
        ("ImageCollection", (CHIRPS,)),
        ("filterDate", ("2020-12-01", "2021-01-01")),
        ("select", ("precipitation",)),
        ("sum", ()),
        ("round", ()),
        ("toInt16", ()),
        ("clip", ("AOI",)),
        ("rename", ("B1",)),
    ]


def test_monthly_temperature_reduces_the_daily_means(monkeypatch):
    """Monthly temperature reduces the daily means."""
    fake = install(monkeypatch, climate)
    climate._get_temperature("AOI", 2020, aggregation="max", period="feb")
    assert fake.calls == [
        ("ImageCollection", (ERA_D,)),
        ("filterDate", ("2020-02-01", "2020-03-01")),
        ("select", ("temperature_2m",)),
        ("max", ()),
        ("subtract", (273.15,)),
        ("round", ()),
        ("toInt16", ()),
        ("clip", ("AOI",)),
        ("rename", ("B1",)),
    ]


def test_climate_names():
    """Climate names."""
    assert (
        build_predefined_name("precipitation", {"period": "year", "source": "era"})
        == "precipitation"
    )
    assert (
        build_predefined_name("precipitation", {"period": "sep", "source": "chirps"})
        == "precipitation_sep_chirps"
    )
    assert resolve_predefined("precipitation_chirps") == (
        "precipitation",
        {"period": "year", "source": "chirps"},
    )
    assert (
        build_predefined_name(
            "temperature_2m", {"aggregation": "median", "period": "year"}
        )
        == "temperature_2m_median"
    )
    assert resolve_predefined("temperature_2m_median_sep") == (
        "temperature_2m",
        {"aggregation": "median", "period": "sep"},
    )
    # The metric is required: a month alone is not a temperature name.
    assert resolve_predefined("temperature_2m_sep") == (None, {})
    # An explicit default segment is not canonical.
    assert resolve_predefined("precipitation_year") == (None, {})


def test_precipitation_years_and_scale_follow_period_and_source():
    """Precipitation years and scale follow period and source."""
    assert resolved_entry("precipitation", {})["years"] == list(range(2001, 2026))
    assert resolved_entry("precipitation", {})["default_scale"] == 11132
    assert (
        resolved_entry("precipitation", {"source": "chirps"})["default_scale"] == 5566
    )
    assert resolved_entry("precipitation", {"period": "sep"})["years"][-1] == 2026
    assert resolved_entry("precipitation", {"period": "oct"})["years"][-1] == 2025
    chirps_aug = {"period": "aug", "source": "chirps"}
    assert resolved_entry("precipitation", chirps_aug)["years"][-1] == 2026
    chirps_sep = {"period": "sep", "source": "chirps"}
    assert resolved_entry("precipitation", chirps_sep)["years"][-1] == 2025


def test_temperature_years_follow_the_period():
    """Temperature years follow the period."""
    assert resolved_entry("temperature_2m", {})["years"] == list(range(2001, 2026))
    assert resolved_entry("temperature_2m", {"period": "sep"})["years"][-1] == 2026


def test_validate_refuses_months_past_the_source():
    """Validate refuses months past the source."""
    err = validate_entry("precipitation", {"period": "oct"}, 2026)
    assert err[0] == "vars.modal.error_period_unavailable"
    assert err[1]["last"] == "2026-09"
    assert validate_entry("precipitation", {}, 2025) is None
    chirps = validate_entry(
        "precipitation", {"period": "sep", "source": "chirps"}, 2026
    )
    assert chirps[1]["last"] == "2026-08"
    assert validate_entry("temperature_2m", {"aggregation": "median"}, 2026) is not None
