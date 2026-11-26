"""dry_spell: rain-based dryness measured at the period start, looking back."""

from _ee_fake import install

import gui.scripts.predefined.climate as climate
from gui.scripts.predefined import (
    active_param_specs,
    build_predefined_name,
    entry_for_name,
    resolve_predefined,
    resolved_entry,
    validate_entry,
)

CHIRPS = "UCSB-CHG/CHIRPS/DAILY"
ERA_D = "ECMWF/ERA5_LAND/DAILY_AGGR"
DEFAULTS = {
    "metric": "dsr",
    "threshold": 1,
    "window": 365,
    "period": "year",
    "source": "chirps",
}


def test_default_name_and_entry():
    """Default name and entry."""
    key, params, entry = entry_for_name("dry_spell")
    assert key == "dry_spell"
    assert params == DEFAULTS
    assert entry["raster_type"] == "continuous"
    assert entry["default_scale"] == 5566
    assert entry["unit_key"] == "legend.unit.days_value"
    assert entry["group"] == "climate"


def test_names_round_trip():
    """Names round trip."""
    values = {
        "metric": "drydays",
        "threshold": 5,
        "window": 90,
        "period": "sep",
        "source": "era",
    }
    name = build_predefined_name("dry_spell", values)
    assert name == "dry_spell_drydays_t5_w90_sep_era"
    assert resolve_predefined(name) == ("dry_spell", values)
    assert resolve_predefined("dry_spell_sep_era") == (
        "dry_spell",
        {**DEFAULTS, "period": "sep", "source": "era"},
    )


def test_threshold_is_dropped_for_raintotal():
    """Threshold is dropped for raintotal."""
    keys = [s["key"] for s in active_param_specs("dry_spell", {"metric": "raintotal"})]
    assert "threshold" not in keys
    values = {**DEFAULTS, "metric": "raintotal", "threshold": 5}
    assert build_predefined_name("dry_spell", values) == "dry_spell_raintotal"
    assert resolve_predefined("dry_spell_raintotal_t5") == (None, {})


def test_days_since_rain_counts_back_from_the_period_start(monkeypatch):
    """Days since rain counts back from the period start."""
    fake = install(monkeypatch, climate)
    climate._get_dry_spell("AOI", 2024, period="sep")
    calls = fake.calls
    # 365 days before 2024-09-01 (leap year) is 2023-09-02; the period itself
    # is never read.
    assert calls[:3] == [
        ("ImageCollection", (CHIRPS,)),
        ("filterDate", ("2023-09-02", "2024-09-01")),
        ("select", ("precipitation",)),
    ]
    assert ("Date", ("2024-09-01",)) in calls
    assert ("gte", (1,)) in calls
    assert ("subtract", (1,)) in calls
    methods = fake.methods()
    assert "Image.constant" in methods
    assert methods.index("min") < methods.index("unmask")
    assert ("unmask", (365,)) in calls  # never rained -> capped at the window
    # No observation at all stays masked instead of reading "never rained".
    assert methods[-6:] == ["count", "gt", "updateMask", "toInt16", "clip", "rename"]


def test_dry_days_on_era_convert_metres(monkeypatch):
    """Dry days on era convert metres."""
    fake = install(monkeypatch, climate)
    climate._get_dry_spell(
        "AOI",
        2024,
        metric="drydays",
        threshold=5,
        window=90,
        period="sep",
        source="era",
    )
    calls = fake.calls
    assert calls[:3] == [
        ("ImageCollection", (ERA_D,)),
        ("filterDate", ("2024-06-03", "2024-09-01")),
        ("select", ("total_precipitation_sum",)),
    ]
    assert ("multiply", (1000,)) in calls
    assert ("lt", (5,)) in calls
    assert "sum" in fake.methods()
    assert "Image.constant" not in fake.methods()


def test_rain_total_sums_the_window(monkeypatch):
    """Rain total sums the window."""
    fake = install(monkeypatch, climate)
    climate._get_dry_spell("AOI", 2020, metric="raintotal", window=30)
    assert fake.calls[:3] == [
        ("ImageCollection", (CHIRPS,)),
        ("filterDate", ("2019-12-02", "2020-01-01")),
        ("select", ("precipitation",)),
    ]
    assert fake.methods()[3:6] == ["sum", "multiply", "round"]


def test_never_reads_the_period_itself(monkeypatch):
    """Never reads the period itself."""
    for period, start in (("year", "2024-01-01"), ("mar", "2024-03-01")):
        fake = install(monkeypatch, climate)
        climate._get_dry_spell("AOI", 2024, period=period, window=999)
        ends = [args[1] for name, args in fake.calls if name == "filterDate"]
        assert ends and all(end <= start for end in ends), (period, ends)


def test_years_follow_source_and_period():
    """Years follow source and period."""

    def years(**params):
        return resolved_entry("dry_spell", params)["years"]

    assert years() == list(range(2001, 2027))
    assert years(period="sep")[-1] == 2026
    assert years(period="oct")[-1] == 2025  # CHIRPS ends 2026-08
    assert years(period="oct", source="era")[-1] == 2026  # ERA5-Land ends 2026-09


def test_validate_refuses_a_window_past_the_source():
    """Validate refuses a window past the source."""
    err = validate_entry("dry_spell", {"period": "oct"}, 2026)
    assert err[0] == "vars.modal.error_period_unavailable"
    assert err[1]["last"] == "2026-08"
    assert validate_entry("dry_spell", {"period": "oct", "source": "era"}, 2026) is None


def test_scale_unit_and_palette_follow_params():
    """Scale unit and palette follow params."""
    era = resolved_entry("dry_spell", {"source": "era"})
    assert era["default_scale"] == 11132
    total = resolved_entry("dry_spell", {"metric": "raintotal"})
    assert total["unit_key"] == "legend.unit.mm_value"
    assert total["vis_params"] != resolved_entry("dry_spell", {})["vis_params"]
