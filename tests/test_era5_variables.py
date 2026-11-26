"""ERA5-Land climate variables: recipes and catalogue entries.

The recipes are exercised against a recording stub (no Earth Engine session),
the same pattern as ``test_threshold_reaches_the_gee_expression`` in
test_predefined_params.py.
"""

import types

import pytest

import gui.scripts.predefined_variables as pv
from gui.scripts.predefined import resolved_entry


class _Rec:
    """Chainable ee stand-in that records every method call."""

    def __init__(self, calls):
        self._calls = calls

    def __getattr__(self, method):
        def _call(*args, **kwargs):
            self._calls.append((method, args))
            return self

        return _call


def _record_recipe(monkeypatch, func, *args, **kwargs):
    """Run a recipe against the recorder; return the [(method, args), ...] log."""
    calls = []

    def _image_collection(collection_id):
        calls.append(("ImageCollection", (collection_id,)))
        return _Rec(calls)

    import gui.scripts.predefined.climate as climate_mod

    monkeypatch.setattr(
        climate_mod, "ee", types.SimpleNamespace(ImageCollection=_image_collection)
    )
    func(*args, **kwargs)
    return calls


def test_precipitation_sums_the_year_in_mm(monkeypatch):
    """Annual total = sum of the 12 monthly totals, converted m -> mm."""
    calls = _record_recipe(monkeypatch, pv._get_precipitation, "AOI", 2020)

    assert ("ImageCollection", ("ECMWF/ERA5_LAND/MONTHLY_AGGR",)) in calls
    assert ("filterDate", ("2020-01-01", "2021-01-01")) in calls
    assert ("select", ("total_precipitation_sum",)) in calls
    assert ("sum", ()) in calls
    assert ("multiply", (1000,)) in calls  # metres of water -> mm
    # Whole mm, stored as int16 — float64 rasters waste 4x the disk space.
    assert calls.index(("round", ())) > calls.index(("multiply", (1000,)))
    assert ("toInt16", ()) in calls
    assert ("clip", ("AOI",)) in calls
    assert ("rename", ("B1",)) in calls


def test_precipitation_catalogue_entry():
    """Continuous, temporal 2001-2025, native ~11 km scale, dynamic stretch."""
    cat = resolved_entry("precipitation", {})

    assert cat["var_type"] == "GEEVar"
    assert cat["raster_type"] == "continuous"
    assert cat["temporal"] is True
    assert cat["years"] == list(range(2001, 2026))
    assert cat["default_scale"] == 11132
    assert cat["get_image"] is pv._get_precipitation
    assert [s["key"] for s in cat["params"]] == ["period", "source"]
    # Palette-only vis -> the AOI-dependent mm range is stretched dynamically.
    assert cat["vis_params"]["palette"]
    assert "min" not in cat["vis_params"] and "max" not in cat["vis_params"]
    assert cat["label_key"] == "vars.predefined.precipitation"
    assert cat["description_key"] == "vars.predefined_info.precipitation"


@pytest.mark.parametrize("metric", ["mean", "max", "min", "median"])
def test_temperature_uses_selected_reducer_in_celsius(monkeypatch, metric):
    """The chosen metric drives the collection reducer; Kelvin -> Celsius."""
    calls = _record_recipe(
        monkeypatch, pv._get_temperature, "AOI", 2020, aggregation=metric
    )

    assert ("ImageCollection", ("ECMWF/ERA5_LAND/MONTHLY_AGGR",)) in calls
    assert ("filterDate", ("2020-01-01", "2021-01-01")) in calls
    assert ("select", ("temperature_2m",)) in calls
    assert (metric, ()) in calls
    other_metrics = {"mean", "max", "min", "median"} - {metric}
    assert not other_metrics & {method for method, _ in calls}
    assert ("subtract", (273.15,)) in calls  # applied AFTER the reducer
    # Whole degrees C, stored as int16 (signed: temperatures go negative).
    assert calls.index(("round", ())) > calls.index(("subtract", (273.15,)))
    assert ("toInt16", ()) in calls
    assert ("clip", ("AOI",)) in calls
    assert ("rename", ("B1",)) in calls


def test_temperature_defaults_to_median(monkeypatch):
    """Omitting the kwarg must match the catalogue default (median)."""
    calls = _record_recipe(monkeypatch, pv._get_temperature, "AOI", 2020)

    assert ("median", ()) in calls


def test_temperature_catalogue_entry():
    """Continuous, temporal 2001-2025, ~11 km, one choice param."""
    cat = resolved_entry("temperature_2m", {})

    assert cat["var_type"] == "GEEVar"
    assert cat["raster_type"] == "continuous"
    assert cat["temporal"] is True
    assert cat["years"] == list(range(2001, 2026))
    assert cat["default_scale"] == 11132
    assert cat["get_image"] is pv._get_temperature
    assert cat["vis_params"]["palette"]
    assert "min" not in cat["vis_params"] and "max" not in cat["vis_params"]

    assert [s["key"] for s in cat["params"]] == ["aggregation", "period"]
    spec = cat["params"][0]
    assert spec["key"] == "aggregation"
    assert spec["type"] == "choice"
    assert spec["default"] == "median"
    assert spec["options"] == ["mean", "max", "min", "median"]
    assert spec["suffix_prefix"] == ""
    assert spec["option_label_key_prefix"] == "vars.modal.agg_"
    assert spec["label_key"] == "vars.modal.param_aggregation"
    assert spec["hint_key"] == "vars.modal.param_aggregation_hint"
