"""Infrastructure layers: protected areas, roads and towns."""

import ee

from gui.scripts.predefined.aoi import _aoi_bounds


def _get_protected_area(aoi, year=None):
    """WDPA protected areas — binary mask (1 = inside a designated protected area).

    Rasterized by painting the status-filtered polygons onto a 0 background, so it
    depends on no feature attribute. The previous implementation reduced on a
    ``WDPAID`` property that the current ``WCMC/WDPA/current/polygons`` schema no
    longer exposes (it was renamed to ``SITE_ID`` / ``SITE_PID``), which silently
    produced an all-zero raster. ``aoi`` may be a Geometry, Feature, or
    FeatureCollection; the pre-filter uses its bounds (see ``_aoi_bounds``)
    and ``clip`` takes the AOI as given.
    """
    wdpa = (
        ee.FeatureCollection("WCMC/WDPA/current/polygons")
        .filterBounds(_aoi_bounds(aoi))
        .filter(
            ee.Filter.inList(
                "STATUS", ["Designated", "Inscribed", "Established", "Proposed"]
            )
        )
    )
    return ee.Image(0).paint(wdpa, 1).clip(aoi).toByte()


def _get_roads(aoi, year=None):
    """OSM roads — binary mask."""
    return (
        ee.Image(
            "projects/ee-andyarnellgee/assets/crosscutting/infrastructure"
            "/roads_osm/roadsAllImageOSM"
        )
        .unmask()
        .clip(aoi)
        .toByte()
    )


def _get_towns(aoi, year):
    """JRC GHSL population + built surface — urban area binary mask."""
    epochs = list(range(1975, 2021, 5))
    epoch = min(epochs, key=lambda x: abs(x - year))
    pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{epoch}")
    built = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}").select("built_surface")
    return ee.Image(0).where(pop.gte(15).And(built.gte(90)), 1).clip(aoi)


ENTRIES = {
    "protected_area": {
        "label_key": "vars.predefined.protected_area",
        "description_key": "vars.predefined_info.protected_area",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "get_image": _get_protected_area,
        "vis_params": {"palette": ["ffffff", "4caf50"], "min": 0, "max": 1},
    },
    "roads": {
        "label_key": "vars.predefined.roads",
        "description_key": "vars.predefined_info.roads",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "get_image": _get_roads,
        "vis_params": {"palette": ["ffffff", "ff9800"], "min": 0, "max": 1},
    },
    "towns": {
        "label_key": "vars.predefined.towns",
        "description_key": "vars.predefined_info.towns",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        "years": list(range(1975, 2021, 5)),
        "get_image": _get_towns,
        "vis_params": {"palette": ["ffffff", "e91e63"], "min": 0, "max": 1},
    },
}
