"""Water layers: OSM rivers."""

import ee

from gui.scripts.predefined.aoi import _aoi_bounds


def _get_rivers(aoi, year=None):
    """OSM water layer — binary mask (rivers/streams)."""
    return (
        ee.ImageCollection("projects/sat-io/open-datasets/OSM_waterLayer")
        .filterBounds(_aoi_bounds(aoi))
        .mosaic()
        .clip(aoi)
        .gte(2)
        .unmask()
        .clip(aoi)
        .toByte()
    )


ENTRIES = {
    "rivers": {
        "label_key": "vars.predefined.rivers",
        "description_key": "vars.predefined_info.rivers",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "get_image": _get_rivers,
        "vis_params": {"palette": ["ffffff", "2196f3"], "min": 0, "max": 1},
    },
}
