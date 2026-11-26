"""fire_history: how often / how recently a pixel burned before the period."""

from _ee_fake import install

import gui.scripts.predefined.fire as fire
from gui.scripts.predefined import (
    build_predefined_name,
    entry_for_name,
    resolve_predefined,
    resolved_entry,
)


def test_default_entry():
    """Default entry."""
    key, params, entry = entry_for_name("fire_history")
    assert params == {
        "metric": "count",
        "window": 10,
        "period": "year",
        "source": "burned",
    }
    assert entry["raster_type"] == "continuous"
    assert entry["default_scale"] == 463
    assert entry["unit_key"] == "legend.unit.years_value"
    assert resolved_entry("fire_history", {"source": "active"})["default_scale"] == 926


def test_names_round_trip():
    """Names round trip."""
    values = {"metric": "since", "window": 5, "period": "sep", "source": "active"}
    name = build_predefined_name("fire_history", values)
    assert name == "fire_history_since_y5_sep_active"
    assert resolve_predefined(name) == ("fire_history", values)


def _filter_dates(fake):
    return [args for name, args in fake.calls if name == "filterDate"]


def test_count_reads_one_slice_per_year_before_the_period(monkeypatch):
    """Count reads one slice per year before the period."""
    fake = install(monkeypatch, fire)
    fire._get_fire_history("AOI", 2024, window=3, period="sep")
    assert _filter_dates(fake) == [
        ("2023-09-01", "2024-09-01"),
        ("2022-09-01", "2023-09-01"),
        ("2021-09-01", "2022-09-01"),
    ]
    assert fake.methods().count("ImageCollection") == 4  # 3 slices + the stack
    assert fake.methods()[-4:] == ["sum", "toInt16", "clip", "rename"]


def test_since_is_the_nearest_burned_slice(monkeypatch):
    """Since is the nearest burned slice."""
    fake = install(monkeypatch, fire)
    fire._get_fire_history("AOI", 2024, metric="since", window=2)
    constants = [args for name, args in fake.calls if name == "Image.constant"]
    assert constants == [(0,), (1,)]  # 0 = burned in the 12 months before
    assert ("unmask", (2,)) in fake.calls  # never burned -> the window
    assert fake.methods()[-5:] == ["min", "unmask", "toInt16", "clip", "rename"]


def test_active_source_uses_modis_nominal_detections(monkeypatch):
    """Active source uses modis nominal detections."""
    fake = install(monkeypatch, fire)
    fire._get_fire_history("AOI", 2024, window=1, source="active")
    assert ("ImageCollection", ("MODIS/061/MOD14A1",)) in fake.calls
    assert ("gte", (8,)) in fake.calls
    assert ("ImageCollection", ("MODIS/061/MCD64A1",)) not in fake.calls


def test_never_reads_the_period_itself(monkeypatch):
    """Never reads the period itself."""
    fake = install(monkeypatch, fire)
    fire._get_fire_history("AOI", 2024, window=20, period="mar")
    assert all(end <= "2024-03-01" for _start, end in _filter_dates(fake))


def test_years_need_the_whole_window():
    """Years need the whole window."""

    def years(**params):
        return resolved_entry("fire_history", params)["years"]

    assert years() == list(range(2011, 2027))  # 10 years of MCD64A1 from 2000-11
    assert years(window=1, period="sep") == list(range(2002, 2026))
    assert years(window=1, period="sep", source="active") == list(range(2001, 2027))
