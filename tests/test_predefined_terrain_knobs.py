"""altitude / slope: DEM source and slope unit as advanced knobs.

Defaults must build the pre-B2 expressions verbatim (SRTM, degrees).
"""

import math

import pytest
from _catalogue_checks import assert_roundtrip, locale_messages
from _ee_expr import fake_aoi, patch_ee

import gui.scripts.predefined.terrain as terrain
from gui.scripts.predefined import (
    PREDEFINED_CATALOGUE,
    build_predefined_name,
    default_param_values,
    param_specs,
    resolve_predefined,
    resolved_entry,
)

AOI_CLIP = ".clip(ee.FeatureCollection('AOI'))"


def _golden_altitude(ee, aoi):
    """Pre-B2 ``_get_altitude`` body, verbatim."""
    return ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)


def _golden_slope(ee, aoi):
    """Pre-B2 ``_get_slope`` body, verbatim."""
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")
    return ee.Terrain.slope(elevation).clip(aoi)


def test_altitude_default_expression_is_unchanged(monkeypatch):
    """No kwargs and explicit defaults both equal today's SRTM expression."""
    fake = patch_ee(monkeypatch, terrain)
    aoi = fake_aoi(fake)
    golden = repr(_golden_altitude(fake, aoi))
    assert repr(terrain._get_altitude(aoi)) == golden
    defaults = default_param_values("altitude")
    assert repr(terrain._get_altitude(aoi, None, **defaults)) == golden


def test_slope_default_expression_is_unchanged(monkeypatch):
    """No kwargs and explicit defaults both equal today's SRTM slope."""
    fake = patch_ee(monkeypatch, terrain)
    aoi = fake_aoi(fake)
    golden = repr(_golden_slope(fake, aoi))
    assert repr(terrain._get_slope(aoi)) == golden
    defaults = default_param_values("slope")
    assert repr(terrain._get_slope(aoi, None, **defaults)) == golden


def test_terrain_params_are_advanced_and_omitted_by_default():
    """Both knobs sit in Advanced; defaults keep the bare names."""
    assert [s["key"] for s in param_specs("altitude")] == ["dem"]
    assert [s["key"] for s in param_specs("slope")] == ["dem", "unit"]
    assert default_param_values("slope") == {"dem": "srtm", "unit": "deg"}
    for spec in param_specs("slope"):
        assert spec["advanced"] is True
        assert spec["omit_default"] is True
    assert PREDEFINED_CATALOGUE["altitude"]["group"] == "terrain"
    assert PREDEFINED_CATALOGUE["slope"]["group"] == "terrain"


def test_terrain_names_roundtrip_exhaustively():
    """3 sources x 2 units: every name is distinct and resolves back."""
    names = set()
    for dem in terrain.DEM_SOURCES:
        names.add(assert_roundtrip("altitude", {"dem": dem}))
        for unit in ("deg", "pct"):
            names.add(assert_roundtrip("slope", {"dem": dem, "unit": unit}))
    assert len(names) == 9
    assert {
        "altitude",
        "altitude_copernicus",
        "altitude_nasadem",
        "slope",
        "slope_pct",
        "slope_nasadem_pct",
    } <= names


def test_legacy_terrain_names_resolve_to_defaults():
    """Saved altitude / slope variables keep resolving to their defaults."""
    assert resolve_predefined("altitude") == ("altitude", {})
    assert resolve_predefined("slope") == ("slope", {})
    assert build_predefined_name("slope", default_param_values("slope")) == "slope"


def test_nasadem_reads_its_elevation_band(monkeypatch):
    """NASADEM is a straight swap for SRTM (band ``elevation``)."""
    fake = patch_ee(monkeypatch, terrain)
    text = repr(terrain._get_altitude(fake_aoi(fake), dem="nasadem"))
    assert text == ("ee.Image('NASA/NASADEM_HGT/001').select('elevation')" + AOI_CLIP)


def test_copernicus_mosaic_gets_a_native_projection_before_slope(monkeypatch):
    """Review focus 3: a bare mosaic has a 1-degree projection -> null slope.

    Verified 2026-10-09: ee.Terrain.slope on the raw GLO-30 mosaic is all-null;
    pinning the first tile's projection gives the expected 3.9 deg mean.
    """
    fake = patch_ee(monkeypatch, terrain)
    text = repr(terrain._get_slope(fake_aoi(fake), dem="copernicus"))
    collection = "ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1')"
    assert text == (
        f"ee.Terrain.slope({collection}"
        ".filterBounds(ee.FeatureCollection('AOI').geometry().bounds())"
        ".select('DEM').mosaic()"
        f".setDefaultProjection({collection}.first().select('DEM').projection()))"
        + AOI_CLIP
    )


def test_percent_slope_is_tangent_times_hundred(monkeypatch):
    """Percent slope = 100 * tan(deg in radians), applied before the clip."""
    fake = patch_ee(monkeypatch, terrain)
    aoi = fake_aoi(fake)
    degrees = repr(_golden_slope(fake, aoi)).removesuffix(AOI_CLIP)
    factor = repr(round(math.pi / 180, 12))
    assert repr(terrain._get_slope(aoi, unit="pct")) == (
        f"{degrees}.multiply({factor}).tan().multiply(100)" + AOI_CLIP
    )


def test_unknown_dem_is_rejected(monkeypatch):
    """A typo never silently falls back to SRTM."""
    fake = patch_ee(monkeypatch, terrain)
    with pytest.raises(ValueError):
        terrain._get_altitude(fake_aoi(fake), dem="aster")


def test_slope_unit_drives_legend_unit_and_stretch():
    """Degrees keep today's vis/unit; percent gets its own unit and range."""
    degrees = resolved_entry("slope", {})
    assert degrees["unit_key"] == "legend.unit.deg_value"
    assert degrees["vis_params"] == {
        "palette": ["1a9850", "ffffbf", "d73027"],
        "min": 0,
        "max": 60,
    }
    percent = resolved_entry("slope", {"unit": "pct", "dem": "copernicus"})
    assert percent["unit_key"] == "legend.unit.pct_value"
    assert percent["vis_params"]["max"] == 100
    assert resolved_entry("altitude", {"dem": "nasadem"})["unit_key"] == (
        "legend.unit.m_value"
    )


def test_percent_unit_label_exists_in_every_locale():
    """The legend formatter key resolves everywhere, with its placeholder."""
    for locale in ("en", "es-ES", "fr", "pt-BR"):
        assert "{value}" in locale_messages(locale)["legend.unit.pct_value"]
