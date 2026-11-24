"""Terrain layers: altitude and slope, from a choice of elevation models."""

import math

import ee

from gui.scripts.predefined.aoi import _aoi_bounds

# Elevation sources (verified 2026-10-09). Bump an ID here when its source is
# re-released; the default (srtm) must keep reproducing the pre-B2 expression.
SRTM_ASSET = "USGS/SRTMGL1_003"  # band elevation, int16, 1 arc-second
NASADEM_ASSET = "NASA/NASADEM_HGT/001"  # band elevation, int16, 1 arc-second
# COPERNICUS/DEM/GLO30 is deprecated in favour of this release (identical values
# on the probe AOI). Band DEM is float32; 1x1 degree EPSG:4326 tiles at 1
# arc-second, coarser in longitude above 50 degrees (2" at 60 N), which the
# pinned projection below resamples onto the 1" grid.
COPERNICUS_DEM_ASSET = "COPERNICUS/DEM/GLO30_2024_1"
DEM_SOURCES = ("srtm", "copernicus", "nasadem")

SLOPE_PALETTE = ["1a9850", "ffffbf", "d73027"]
# Legend / stretch ceiling per unit: 60 degrees as before B2; 100 % = 45 degrees.
SLOPE_VIS_MAX = {"deg": 60, "pct": 100}


def _elevation(aoi, dem):
    """The (unclipped) elevation image of one source."""
    if dem == "srtm":
        return ee.Image(SRTM_ASSET).select("elevation")
    if dem == "nasadem":
        return ee.Image(NASADEM_ASSET).select("elevation")
    if dem == "copernicus":
        tiles = ee.ImageCollection(COPERNICUS_DEM_ASSET)
        # A mosaic carries the default 1-degree WGS84 projection and
        # ee.Terrain.slope on it is all-null (verified 2026-10-09): pin the
        # tiles' native 1-arc-second grid. The bounds pre-filter is only an
        # optimisation (oversized tile footprints keep a few hundred tiles);
        # the layer is clipped to the AOI afterwards.
        native = tiles.first().select("DEM").projection()
        return (
            tiles.filterBounds(_aoi_bounds(aoi))
            .select("DEM")
            .mosaic()
            .setDefaultProjection(native)
        )
    raise ValueError(f"Unknown elevation source: {dem!r}")


def _get_altitude(aoi, year=None, dem="srtm"):
    """Elevation in metres from the chosen source (default SRTM, as before)."""
    return _elevation(aoi, dem).clip(aoi)


def _get_slope(aoi, year=None, dem="srtm", unit="deg"):
    """Terrain slope in degrees, or percent (100 * tan) when ``unit == "pct"``."""
    slope = ee.Terrain.slope(_elevation(aoi, dem))
    if unit == "pct":
        slope = slope.multiply(math.pi / 180).tan().multiply(100)
    return slope.clip(aoi)


def _slope_vis(params):
    """Green (flat) -> red (steep), pinned to the unit's range."""
    return {
        "palette": list(SLOPE_PALETTE),
        "min": 0,
        "max": SLOPE_VIS_MAX[params["unit"]],
    }


def _slope_unit_key(params):
    """Legend value formatter for the chosen slope unit."""
    if params["unit"] == "pct":
        return "legend.unit.pct_value"
    return "legend.unit.deg_value"


def _dem_param():
    """The elevation-source knob (a fresh dict per entry)."""
    return {
        "key": "dem",
        "label_key": "vars.modal.param_dem",
        "hint_key": "vars.modal.param_dem_hint",
        "type": "choice",
        "default": "srtm",
        "options": list(DEM_SOURCES),
        "option_label_key_prefix": "vars.modal.dem_",
        "suffix_prefix": "",
        "advanced": True,
        "omit_default": True,
    }


def _slope_unit_param():
    """Degrees (default) or percent rise."""
    return {
        "key": "unit",
        "label_key": "vars.modal.param_slope_unit",
        "hint_key": "vars.modal.param_slope_unit_hint",
        "type": "choice",
        "default": "deg",
        "options": ["deg", "pct"],
        "option_label_key_prefix": "vars.modal.slope_unit_",
        "suffix_prefix": "",
        "advanced": True,
        "omit_default": True,
    }


ENTRIES = {
    "altitude": {
        "group": "terrain",
        "label_key": "vars.predefined.altitude",
        "description_key": "vars.predefined_info.altitude",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": False,
        "params": [_dem_param()],
        "get_image": _get_altitude,
        # Terrain ramp (green lowlands -> brown -> white peaks), stretched to AOI.
        "vis_params": {
            "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
        },
        "unit_key": "legend.unit.m_value",
    },
    "slope": {
        "group": "terrain",
        "label_key": "vars.predefined.slope",
        "description_key": "vars.predefined_info.slope",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": False,
        "params": [_dem_param(), _slope_unit_param()],
        "get_image": _get_slope,
        # Dynamic (Part D): the unit picks the stretch ceiling and legend unit.
        "vis_params": _slope_vis,
        "unit_key": _slope_unit_key,
    },
}
