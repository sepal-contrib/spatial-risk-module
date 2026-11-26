"""Infrastructure layers: protected areas, roads and towns."""

import ee

from gui.scripts.predefined.aoi import _aoi_bounds

# --- protected areas ----------------------------------------------------------
# WDPA and WD-OECM polygons (UNEP-WCMC & IUCN, Protected Planet; "current" tracks
# the monthly release). Field values verified 2026-10-09:
#   STATUS      Designated, Established, Proposed, Inscribed, Adopted (34 sites)
#   SITE_TYPE   "PA" for every WDPA polygon; OECMs are a separate collection
#   IUCN_CAT    Ia Ib II III IV V VI, Not Reported / Not Assigned / Not Applicable
#   DESIG_TYPE  National, Regional, International, Not Applicable (68 sites)
#   REALM       Terrestrial, Coastal, Marine (there is no MARINE field)
#   STATUS_YR   0 (unknown) for ~6.7 % of sites, else 1068..2026
WDPA_POLYGONS = "WCMC/WDPA/current/polygons"
WDOECM_POLYGONS = "WCMC/WDOECM/current/polygons"
PA_STATUS = {
    "designated": "Designated",
    "inscribed": "Inscribed",
    "established": "Established",
    "proposed": "Proposed",
    "adopted": "Adopted",
}
PA_STATUS_CODES = {
    "designated": "d",
    "inscribed": "i",
    "established": "e",
    "proposed": "p",
    "adopted": "a",
}
PA_DEFAULT_STATUS = ["designated", "inscribed", "established", "proposed"]
PA_SITE_COLLECTIONS = {"pa": WDPA_POLYGONS, "oecm": WDOECM_POLYGONS}
PA_IUCN_GROUPS = {
    "strict": ("Ia", "Ib", "II"),
    "managed": ("III", "IV"),
    "sustainable": ("V", "VI"),
    "unassigned": ("Not Reported", "Not Assigned", "Not Applicable"),
}
PA_IUCN_CODES = {"strict": "s", "managed": "m", "sustainable": "u", "unassigned": "n"}
PA_DESIGNATIONS = {
    "national": "National",
    "international": "International",
    "regional": "Regional",
}
PA_DESIGNATION_CODES = {"national": "n", "international": "i", "regional": "r"}


def _wdpa_sites(aoi, status, site_types, iucn, designation, realm):
    """The selected WDPA / WD-OECM polygons around the AOI.

    The default selection builds exactly the pre-B2 chain (WDPA, bounds
    pre-filter, the four-status filter). A filter whose selection covers every
    value is not added at all, so the default stays identical and values the
    WDPA might add later are not dropped.
    """
    sources = [
        PA_SITE_COLLECTIONS[kind] for kind in PA_SITE_COLLECTIONS if kind in site_types
    ]
    if not sources:
        raise ValueError("Protected areas need at least one site type.")
    sites = ee.FeatureCollection(sources[0])
    for asset in sources[1:]:
        sites = sites.merge(ee.FeatureCollection(asset))
    sites = sites.filterBounds(_aoi_bounds(aoi)).filter(
        ee.Filter.inList("STATUS", [PA_STATUS[value] for value in status])
    )
    if set(iucn) != set(PA_IUCN_GROUPS):
        categories = [c for group in iucn for c in PA_IUCN_GROUPS[group]]
        sites = sites.filter(ee.Filter.inList("IUCN_CAT", categories))
    if set(designation) != set(PA_DESIGNATIONS):
        types = [PA_DESIGNATIONS[value] for value in designation]
        sites = sites.filter(ee.Filter.inList("DESIG_TYPE", types))
    if realm == "land":
        sites = sites.filter(ee.Filter.neq("REALM", "Marine"))
    return sites


def _get_protected_area(
    aoi,
    year=None,
    timing="current",
    status=PA_DEFAULT_STATUS,
    site_types=["pa"],
    iucn=["strict", "managed", "sustainable", "unassigned"],
    designation=["national", "international", "regional"],
    realm="all",
):
    """WDPA protected areas — binary mask (1 = inside a selected site).

    Rasterized by painting the filtered polygons onto a 0 background, so it
    depends on no feature attribute (the old ``WDPAID`` reduction silently gave
    an all-zero raster after the schema renamed it to ``SITE_ID``/``SITE_PID``).
    ``aoi`` may be a Geometry, Feature or FeatureCollection; the pre-filter
    uses its bounds (see ``_aoi_bounds``) and ``clip`` takes the AOI as given.

    ``timing="yearly"`` keeps the sites whose status year is <= ``year`` (the
    network on 31 December). Sites with an unknown status year (STATUS_YR == 0)
    pass ``lte`` and count in every year, deliberately: they exist today, so the
    latest year equals ``"current"`` with the same filters. Boundaries are
    today's.
    """
    sites = _wdpa_sites(aoi, status, site_types, iucn, designation, realm)
    if timing == "yearly":
        sites = sites.filter(ee.Filter.lte("STATUS_YR", year))
    return ee.Image(0).paint(sites, 1).clip(aoi).toByte()


def _protected_area_params():
    """Timing (current or by status year), then the advanced WDPA filters."""
    return [
        {
            "key": "timing",
            "label_key": "vars.modal.param_pa_timing",
            "hint_key": "vars.modal.param_pa_timing_hint",
            "type": "choice",
            "default": "current",
            "options": ["current", "yearly"],
            "option_label_key_prefix": "vars.modal.pa_timing_",
            "suffix_prefix": "",
            "advanced": False,
            "omit_default": True,
        },
        {
            "key": "status",
            "label_key": "vars.modal.param_pa_status",
            "hint_key": "vars.modal.param_pa_status_hint",
            "type": "multichoice",
            "default": list(PA_DEFAULT_STATUS),
            "options": [
                {"value": value, "code": PA_STATUS_CODES[value]} for value in PA_STATUS
            ],
            "option_label_key_prefix": "vars.modal.pa_status_",
            "min_selected": 1,
            "suffix_prefix": "s",
            "advanced": True,
            "omit_default": True,
        },
        {
            "key": "site_types",
            "label_key": "vars.modal.param_pa_site_types",
            "hint_key": "vars.modal.param_pa_site_types_hint",
            "type": "multichoice",
            "default": ["pa"],
            "options": [{"value": "pa", "code": "p"}, {"value": "oecm", "code": "o"}],
            "option_label_key_prefix": "vars.modal.pa_site_",
            "min_selected": 1,
            "suffix_prefix": "t",
            "advanced": True,
            "omit_default": True,
        },
        {
            "key": "iucn",
            "label_key": "vars.modal.param_pa_iucn",
            "hint_key": "vars.modal.param_pa_iucn_hint",
            "type": "multichoice",
            "default": list(PA_IUCN_GROUPS),
            "options": [
                {"value": value, "code": PA_IUCN_CODES[value]}
                for value in PA_IUCN_GROUPS
            ],
            "option_label_key_prefix": "vars.modal.pa_iucn_",
            "min_selected": 1,
            "suffix_prefix": "i",
            "advanced": True,
            "omit_default": True,
        },
        {
            "key": "designation",
            "label_key": "vars.modal.param_pa_designation",
            "hint_key": "vars.modal.param_pa_designation_hint",
            "type": "multichoice",
            "default": list(PA_DESIGNATIONS),
            "options": [
                {"value": value, "code": PA_DESIGNATION_CODES[value]}
                for value in PA_DESIGNATIONS
            ],
            "option_label_key_prefix": "vars.modal.pa_desig_",
            "min_selected": 1,
            "suffix_prefix": "d",
            "advanced": True,
            "omit_default": True,
        },
        {
            "key": "realm",
            "label_key": "vars.modal.param_pa_realm",
            "hint_key": "vars.modal.param_pa_realm_hint",
            "type": "choice",
            "default": "all",
            "options": ["all", "land"],
            "option_label_key_prefix": "vars.modal.pa_realm_",
            "suffix_prefix": "",
            "advanced": True,
            "omit_default": True,
        },
    ]


# Years offered by protected_area's "By year" timing. WDPA STATUS_YR reached
# 2026 on 2026-10-09 (monthly releases): raise the upper bound each January.
PA_YEARS = list(range(2000, 2027))


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


# --- towns --------------------------------------------------------------------
# GHSL P2023A (verified 2026-10-09): GHS_POP, GHS_BUILT_S and GHS_SMOD are
# ImageCollections with epochs 1975..2030 every 5 years (2025 and 2030 are
# model projections). POP/BUILT_S are 100 m; SMOD is 1 km World Mollweide,
# band smod_code: 10 water, 11 very low density rural, 12 low density rural,
# 13 rural cluster, 21 suburban/peri-urban, 22 semi-dense urban cluster,
# 23 dense urban cluster, 30 urban centre.
GHSL_EPOCHS = list(range(1975, 2021, 5))  # threshold method: pre-B2 list, unchanged
SMOD_EPOCHS = list(range(1975, 2031, 5))
SMOD_SCALE = 1000
SMOD_CLASS_CODES = {
    "centre": 30,
    "dense": 23,
    "semidense": 22,
    "suburban": 21,
    "ruralcluster": 13,
    "lowrural": 12,
    "verylowrural": 11,
}
SMOD_CLASS_LETTERS = {
    "centre": "c",
    "dense": "d",
    "semidense": "s",
    "suburban": "u",
    "ruralcluster": "r",
    "lowrural": "l",
    "verylowrural": "v",
}
SMOD_DEFAULT_CLASSES = ["centre", "dense", "semidense", "suburban"]  # urban domain
TOWNS_METHODS = ("ghsl", "smod")


def _nearest_epoch(year, epochs):
    return min(epochs, key=lambda x: abs(x - year))


def _get_towns(
    aoi,
    year,
    method="ghsl",
    population_threshold=15,
    built_threshold=90,
    smod_classes=SMOD_DEFAULT_CLASSES,
):
    """JRC GHSL urban-area binary mask for the epoch nearest ``year``.

    ``ghsl`` (default, the pre-B2 layer): population >= ``population_threshold``
    people and built-up surface >= ``built_threshold`` m2 per 100 m cell.
    ``smod``: 1 where the settlement-model class is one of ``smod_classes``.
    """
    if method not in TOWNS_METHODS:
        raise ValueError(f"Unknown towns method: {method!r}")
    if method == "smod":
        epoch = _nearest_epoch(year, SMOD_EPOCHS)
        codes = [SMOD_CLASS_CODES[name] for name in smod_classes]
        return (
            ee.Image(f"JRC/GHSL/P2023A/GHS_SMOD/{epoch}")
            .select("smod_code")
            .remap(codes, [1] * len(codes), 0)
            .unmask(0)
            .clip(aoi)
            .toByte()
        )
    epoch = _nearest_epoch(year, GHSL_EPOCHS)
    pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{epoch}")
    built = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}").select("built_surface")
    return (
        ee.Image(0)
        .where(pop.gte(population_threshold).And(built.gte(built_threshold)), 1)
        .clip(aoi)
    )


def _towns_years(params):
    """Epochs offered for the chosen method (Part D dynamic ``years``)."""
    return list(SMOD_EPOCHS) if params["method"] == "smod" else list(GHSL_EPOCHS)


def _towns_default_scale(params):
    """SMOD is 1 km: export at native scale; ghsl keeps the 30 m fallback."""
    return SMOD_SCALE if params["method"] == "smod" else None


def _towns_params():
    """Method choice plus the knobs of each method (depends_on, Part D)."""
    return [
        {
            "key": "method",
            "label_key": "vars.modal.param_towns_method",
            "hint_key": "vars.modal.param_towns_method_hint",
            "type": "choice",
            "default": "ghsl",
            "options": list(TOWNS_METHODS),
            "option_label_key_prefix": "vars.modal.towns_method_",
            "suffix_prefix": "",
            "advanced": True,
            "omit_default": True,
        },
        {
            "key": "population_threshold",
            "label_key": "vars.modal.param_towns_population",
            "hint_key": "vars.modal.param_towns_population_hint",
            "type": "int",
            "default": 15,
            "min": 1,
            "max": 999,
            "suffix_prefix": "p",
            "advanced": True,
            "omit_default": True,
            "depends_on": {"method": ["ghsl"]},
        },
        {
            "key": "built_threshold",
            "label_key": "vars.modal.param_towns_built",
            "hint_key": "vars.modal.param_towns_built_hint",
            "type": "int",
            "default": 90,
            "min": 1,
            "max": 999,
            "suffix_prefix": "b",
            "advanced": True,
            "omit_default": True,
            "depends_on": {"method": ["ghsl"]},
        },
        {
            "key": "smod_classes",
            "label_key": "vars.modal.param_towns_smod_classes",
            "hint_key": "vars.modal.param_towns_smod_classes_hint",
            "type": "multichoice",
            "default": list(SMOD_DEFAULT_CLASSES),
            "options": [
                {"value": value, "code": SMOD_CLASS_LETTERS[value]}
                for value in SMOD_CLASS_CODES
            ],
            "option_label_key_prefix": "vars.modal.towns_smod_",
            "min_selected": 1,
            "suffix_prefix": "u",
            "advanced": True,
            "omit_default": True,
            "depends_on": {"method": ["smod"]},
        },
    ]


ENTRIES = {
    "protected_area": {
        "group": "infrastructure",
        "label_key": "vars.predefined.protected_area",
        "description_key": "vars.predefined_info.protected_area",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        # "By year" asks for a year; "Current" is one map for every period.
        "temporal": lambda params: params["timing"] == "yearly",
        "years": list(PA_YEARS),
        "params": _protected_area_params(),
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
        "group": "infrastructure",
        "label_key": "vars.predefined.towns",
        "description_key": "vars.predefined_info.towns",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        # Dynamic (Part D): ghsl 1975-2020 as before; smod 1975-2030.
        "years": _towns_years,
        "default_scale": _towns_default_scale,
        "params": _towns_params(),
        "get_image": _get_towns,
        "vis_params": {"palette": ["ffffff", "e91e63"], "min": 0, "max": 1},
    },
}
