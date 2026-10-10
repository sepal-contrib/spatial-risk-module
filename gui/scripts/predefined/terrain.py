"""Terrain layers: altitude and slope (SRTM)."""

import ee


def _get_altitude(aoi, year=None):
    """USGS SRTM 30m elevation."""
    return ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)


def _get_slope(aoi, year=None):
    """Terrain slope derived from SRTM elevation."""
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")
    return ee.Terrain.slope(elevation).clip(aoi)


ENTRIES = {
    "altitude": {
        "label_key": "vars.predefined.altitude",
        "description_key": "vars.predefined_info.altitude",
        "group": "terrain",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": False,
        "get_image": _get_altitude,
        # Terrain ramp (green lowlands -> brown -> white peaks), stretched to AOI.
        "vis_params": {
            "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
        },
        "unit_key": "legend.unit.m_value",
    },
    "slope": {
        "label_key": "vars.predefined.slope",
        "description_key": "vars.predefined_info.slope",
        "group": "terrain",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": False,
        "get_image": _get_slope,
        # Degrees: green (flat) -> yellow -> red (steep).
        "vis_params": {
            "palette": ["1a9850", "ffffbf", "d73027"],
            "min": 0,
            "max": 60,
        },
        "unit_key": "legend.unit.deg_value",
    },
}
