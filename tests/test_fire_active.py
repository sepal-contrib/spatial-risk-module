"""active_fire: daily active-fire products summarised over a period."""

import pytest
from _ee_fake import install

import gui.scripts.predefined.fire as fire
from gui.scripts.predefined import (
    build_predefined_name,
    entry_for_name,
    resolve_predefined,
    resolved_entry,
    validate_entry,
)


def test_default_entry_is_a_categorical_fire_mask():
    """Default entry is a categorical fire mask."""
    key, params, entry = entry_for_name("active_fire")
    assert params == {
        "metric": "any",
        "period": "year",
        "confidence": "nominal",
        "source": "modis",
    }
    assert entry["raster_type"] == "categorical"
    assert entry["default_scale"] == 926
    assert entry["vis_params"]["min"] == 0 and entry["vis_params"]["max"] == 1
    assert entry["legend_class_keys"] == ("legend.class.no_fire", "legend.class.fire")


def test_metric_drives_type_palette_and_unit():
    """Metric drives type palette and unit."""
    days = resolved_entry("active_fire", {"metric": "days"})
    assert days["raster_type"] == "continuous"
    assert days["unit_key"] == "legend.unit.days_value"
    assert "min" not in days["vis_params"]
    frp = resolved_entry("active_fire", {"metric": "frp"})
    assert frp["raster_type"] == "continuous"
    assert frp["unit_key"] == "legend.unit.mw_value"


def test_names_round_trip():
    """Names round trip."""
    values = {
        "metric": "days",
        "period": "sep",
        "confidence": "high",
        "source": "viirs",
    }
    name = build_predefined_name("active_fire", values)
    assert name == "active_fire_days_sep_high_viirs"
    assert resolve_predefined(name) == ("active_fire", values)
    assert resolve_predefined("active_fire_firms") == (
        "active_fire",
        {"metric": "any", "period": "year", "confidence": "nominal", "source": "firms"},
    )


def test_any_detection_merges_terra_and_aqua(monkeypatch):
    """Any detection merges terra and aqua."""
    fake = install(monkeypatch, fire)
    fire._get_active_fire("AOI", 2024, period="sep")
    calls = fake.calls
    assert calls[:2] == [
        ("ImageCollection", ("MODIS/061/MOD14A1",)),
        ("ImageCollection", ("MODIS/061/MYD14A1",)),
    ]
    assert calls[2][0] == "merge"
    assert calls[3] == ("filterDate", ("2024-09-01", "2024-10-01"))
    assert ("select", ("FireMask",)) in calls
    assert ("gte", (8,)) in calls  # nominal = fire classes 8 and 9
    assert fake.methods()[-5:] == ["max", "unmask", "toByte", "clip", "rename"]


def test_firms_uses_the_confidence_band(monkeypatch):
    """Firms uses the confidence band."""
    fake = install(monkeypatch, fire)
    fire._get_active_fire("AOI", 2024, confidence="high", source="firms")
    assert fake.calls[0] == ("ImageCollection", ("FIRMS",))
    assert ("select", ("confidence",)) in fake.calls
    assert ("gte", (80,)) in fake.calls


def test_fire_days_flag_each_day_once(monkeypatch):
    """Terra and Aqua both see a fire on the same day: that counts one day."""
    fake = install(monkeypatch, fire)
    fire._get_active_fire("AOI", 2024, metric="days", period="feb")
    methods = fake.methods()
    assert ("Date", ("2024-02-01",)) in fake.calls
    assert ("Date", ("2024-03-01",)) in fake.calls
    assert "List.sequence" in methods
    # per day: filter that day, flag, add a zero image, keep the max
    assert methods.index("advance") < methods.index("max")
    assert "Image.constant" in methods
    assert methods[-4:] == ["sum", "toInt16", "clip", "rename"]


def test_frp_converts_modis_tenths_to_megawatts(monkeypatch):
    """Frp converts modis tenths to megawatts."""
    fake = install(monkeypatch, fire)
    fire._get_active_fire("AOI", 2024, metric="frp")
    assert ("select", ("MaxFRP",)) in fake.calls
    assert ("multiply", (0.1,)) in fake.calls
    assert fake.methods()[-5:] == ["max", "unmask", "toFloat", "clip", "rename"]


def test_frp_on_viirs_is_already_megawatts(monkeypatch):
    """Frp on viirs is already megawatts."""
    fake = install(monkeypatch, fire)
    fire._get_active_fire("AOI", 2024, metric="frp", source="viirs")
    assert fake.calls[0] == ("ImageCollection", ("NASA/VIIRS/002/VNP14A1",))
    assert ("multiply", (1,)) in fake.calls


def test_frp_with_firms_is_refused(monkeypatch):
    """Frp with firms is refused."""
    install(monkeypatch, fire)
    with pytest.raises(ValueError, match="FIRMS"):
        fire._get_active_fire("AOI", 2024, metric="frp", source="firms")
    assert validate_entry(
        "active_fire", {"metric": "frp", "source": "firms"}, 2024
    ) == (
        "vars.modal.error_frp_firms",
        {},
    )
    assert (
        validate_entry("active_fire", {"metric": "frp", "source": "viirs"}, 2024)
        is None
    )


def test_active_fire_years_follow_the_source():
    """Active fire years follow the source."""

    def years(**params):
        return resolved_entry("active_fire", params)["years"]

    assert years() == list(range(2001, 2026))
    assert years(period="aug")[-1] == 2026  # MODIS ends 2026-08
    assert years(period="sep")[-1] == 2025
    assert years(source="viirs")[0] == 2013  # VNP14A1 from 2012-02
    assert years(source="viirs", period="sep") == list(range(2012, 2027))
    assert (
        validate_entry("active_fire", {"period": "sep"}, 2026)[1]["last"] == "2026-08"
    )


def test_flags_unmask_outside_the_image_footprint(monkeypatch):
    """Default sameFootprint unmasking zeroes the MODIS composite (live check)."""
    fake = install(monkeypatch, fire)
    fire._get_active_fire("AOI", 2024)
    assert ("unmask", (0, False)) in fake.calls
