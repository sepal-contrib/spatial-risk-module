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


# ---------------------------------------------------------------------------
# roads
# ---------------------------------------------------------------------------

# Default source: one `constant` band, EPSG:4326 at 0.00025° (~28 m), all OSM
# roads with no road-type breakdown (asset metadata). Saved ``roads`` layers
# depend on this exact expression — keep it byte-identical.
OSM_ROADS_ASSET = (
    "projects/ee-andyarnellgee/assets/crosscutting/infrastructure"
    "/roads_osm/roadsAllImageOSM"
)

# GRIP4 (Meijer et al. 2018, PBL/GLOBIO, CC BY 4.0) as published by the sat-io
# awesome-gee-community-catalog: one table per region. Painting all seven
# merged costs the same as picking the AOI's region (benchmark 2026-10-08/09)
# and keeps AOIs that straddle two regions whole (a Nogales 1° box hits
# North-America and Central-South-America). Bump here if the catalog
# republishes GRIP.
GRIP4_TABLES = tuple(
    f"projects/sat-io/open-datasets/GRIP4/{region}"
    for region in (
        "Africa",
        "Central-South-America",
        "Europe",
        "Middle-East-Central-Asia",
        "North-America",
        "Oceania",
        "South-East-Asia",
    )
)

# GRIP4 road type ``GP_RTP``. Verified 2026-10-09: all 25,758,453 features of
# the seven tables carry GP_RTP (Long) and only 1-5 occur, so there is no
# "unspecified" class to handle. ``code`` is the one-letter name segment.
ROAD_TYPES = (
    {"value": "highway", "code": "h", "gp_rtp": 1},
    {"value": "primary", "code": "p", "gp_rtp": 2},
    {"value": "secondary", "code": "s", "gp_rtp": 3},
    {"value": "tertiary", "code": "t", "gp_rtp": 4},
    {"value": "local", "code": "l", "gp_rtp": 5},
)
# A list (not a tuple) so it compares equal to the catalogue spec default;
# never mutated.
ROAD_TYPE_DEFAULT = [road_type["value"] for road_type in ROAD_TYPES]
_GP_RTP = {road_type["value"]: road_type["gp_rtp"] for road_type in ROAD_TYPES}


def _get_roads(aoi, year=None, source="osm", types=ROAD_TYPE_DEFAULT):
    """Binary road mask (1 = road, 0 = no road) clipped to the AOI.

    ``source="osm"`` (default) reads the OSM roads raster and ignores
    ``types`` — that asset has no road-type breakdown. ``source="grip"``
    paints the GRIP4 lines whose ``GP_RTP`` class is in ``types`` onto a 0
    image, so an AOI with none of the chosen types gives an all-zero layer,
    never a masked one (masked pixels would export as 255 nodata).

    ``paint`` without a width draws 1-pixel, 8-connected lines at the export
    scale (verified 2026-10-09 at 30 m). The OSM raster's lines are also one
    pixel wide but mostly 4-connected, so diagonal GRIP4 roads cover slightly
    fewer pixels per km. The kwarg defaults must equal the catalogue spec
    defaults (single source of truth: the ``roads`` entry's ``params``).
    """
    if source == "osm":
        return ee.Image(OSM_ROADS_ASSET).unmask().clip(aoi).toByte()
    if source != "grip":
        raise ValueError(f"Unknown roads source: {source!r}")
    if not types or any(value not in _GP_RTP for value in types):
        raise ValueError(f"Invalid GRIP4 road types: {list(types)!r}")
    classes = sorted({_GP_RTP[value] for value in types})
    roads = (
        ee.FeatureCollection([ee.FeatureCollection(table) for table in GRIP4_TABLES])
        .flatten()
        .filterBounds(_aoi_bounds(aoi))
        .filter(ee.Filter.inList("GP_RTP", classes))
    )
    return ee.Image(0).paint(roads, 1).clip(aoi).toByte()


def _roads_params():
    """``source`` and GRIP4 ``types`` — both advanced, omitted at default.

    Names: ``roads`` (OSM), ``roads_grip`` (all five types),
    ``roads_grip_r<codes>`` (a subset, codes in declared order, e.g.
    ``roads_grip_rhp`` = highways + primary). The ``r`` prefix keeps a types
    segment from ever parsing as a source value. ``types`` only applies to
    GRIP4 (``depends_on``); with OSM it is hidden, coerced to its default and
    therefore never written into the name.
    """
    return [
        {
            "key": "source",
            "label_key": "vars.modal.param_roads_source",
            "hint_key": "vars.modal.param_roads_source_hint",
            "type": "choice",
            "default": "osm",
            "options": ["osm", "grip"],
            "option_label_key_prefix": "vars.modal.roads_source_",
            "suffix_prefix": "",
            "advanced": True,
            "omit_default": True,
        },
        {
            "key": "types",
            "label_key": "vars.modal.param_road_types",
            "hint_key": "vars.modal.param_road_types_hint",
            "type": "multichoice",
            "default": list(ROAD_TYPE_DEFAULT),
            "options": [
                {"value": road_type["value"], "code": road_type["code"]}
                for road_type in ROAD_TYPES
            ],
            "option_label_key_prefix": "vars.modal.road_type_",
            "min_selected": 1,
            "suffix_prefix": "r",
            "advanced": True,
            "omit_default": True,
            "depends_on": {"source": ["grip"]},
        },
    ]


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
        "group": "infrastructure",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "get_image": _get_protected_area,
        "vis_params": {"palette": ["ffffff", "4caf50"], "min": 0, "max": 1},
    },
    "roads": {
        "label_key": "vars.predefined.roads",
        "description_key": "vars.predefined_info.roads",
        "group": "infrastructure",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "params": _roads_params(),
        "get_image": _get_roads,
        "vis_params": {"palette": ["ffffff", "ff9800"], "min": 0, "max": 1},
    },
    "towns": {
        "label_key": "vars.predefined.towns",
        "description_key": "vars.predefined_info.towns",
        "group": "infrastructure",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        "years": list(range(1975, 2021, 5)),
        "get_image": _get_towns,
        "vis_params": {"palette": ["ffffff", "e91e63"], "min": 0, "max": 1},
    },
}
