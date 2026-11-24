"""MapBiomas asset registry: verified asset ids, layouts and coverage.

Pure data plus lookups. There is deliberately no ``ee`` import, so the modal, the
validators and the tests can read the registry without Earth Engine.

Every record was probed read-only on 2026-10-09 (re-verified 2026-11-24) against
``projects/mapbiomas-public/assets/``. When MapBiomas publishes a new collection,
update the record here and the pinned facts in
``tests/test_mapbiomas_registry.py`` together.

Layouts:

- ``"bands"``: one ``ee.Image`` with one band per year. ``band`` is a format
  string containing ``{year}``.
- ``"year_month_collection"``: an ``ee.ImageCollection`` with one image per
  year-month, a fixed ``band`` and ``year``/``month`` properties. The month
  property is an int (``month_value="int"``) or a ``"01"``..``"12"`` string
  (``"zero_padded"``).
- ``"month_collection"``: 12 images keyed by an int ``month`` property, each
  with one band per year (``band`` contains ``{year}``).
"""

import copy
import re

ROOT = "projects/mapbiomas-public/assets/"
LICENSE = "CC-BY-SA 4.0"

MAPBIOMAS_REGIONS = (
    "brazil",
    "amazon",
    "bolivia",
    "colombia",
    "ecuador",
    "paraguay",
    "peru",
    "venezuela",
)

_REGION_NAMES = {
    "brazil": "Brazil",
    "amazon": "Pan-Amazon",
    "bolivia": "Bolivia",
    "colombia": "Colombia",
    "ecuador": "Ecuador",
    "paraguay": "Paraguay",
    "peru": "Peru",
    "venezuela": "Venezuela",
}

# Level-1 grouping of the MapBiomas LULC legend, which all the collections above
# share. Each code is in exactly one group. 0 (no data) and 27 (not observed)
# are nodata. Codes listed in LULC_INFERRED_CODES were placed by cross-tabulating
# against ESA WorldCover v200 because the legend site was unreachable on
# 2026-10-09; confirm them when it is reachable again.
LULC_GROUPS = ("forest", "natural", "farming", "bare", "water")
LULC_GROUP_CODES = {
    "forest": "f",
    "natural": "n",
    "farming": "a",
    "bare": "b",
    "water": "w",
}
LULC_GROUP_COLORS = {
    "forest": "1f8d49",
    "natural": "d6bc74",
    "farming": "ffefc3",
    "bare": "d4271e",
    "water": "2532e4",
}
LULC_CODE_GROUPS = {
    "forest": (1, 3, 4, 5, 6, 7, 49, 74),
    "natural": (10, 11, 12, 13, 29, 32, 50, 66, 70, 77, 81, 82, 84),
    "farming": (9, 14, 15, 18, 19, 20, 21, 35, 36, 39, 40, 41, 46, 47, 48, 62, 72),
    "bare": (22, 23, 24, 25, 30, 61, 68, 75),
    "water": (26, 31, 33, 34),
}
LULC_NODATA_CODES = (0, 27)
LULC_INFERRED_CODES = {
    7: "Brazil C11 flooded savanna (beta): WorldCover tree/wetland -> forest",
    74: "Ecuador C4: tree-dominated in WorldCover -> forest",
    77: "Brazil C11 renumbering of 13 (other non-forest formation) -> natural",
    81: "Andean grassland / bare (Ecuador, Bolivia) -> natural",
    84: "Brazil C11: WorldCover herbaceous wetland -> natural",
}
LULC_CODE_TO_GROUP = {
    code: group for group, codes in LULC_CODE_GROUPS.items() for code in codes
}
LULC_KNOWN_CODES = tuple(sorted(LULC_CODE_TO_GROUP))

# Deforestation / secondary-vegetation classes. These were verified in Brazil:
# class 4 in year Y means Y-1 was class 2 and Y+1 is class 1; class 6 in Y means
# Y-1 was class 3.
DEFO_CLASSES = {
    1: "anthropic",
    2: "primary_vegetation",
    3: "secondary_vegetation",
    4: "primary_suppression",
    5: "recovery_to_secondary",
    6: "secondary_suppression",
    7: "other",
}

# LULC codes seen in a 5 km histogram of the last year of each collection.
_LULC_OBSERVED = {
    "brazil": (
        3,
        4,
        5,
        6,
        7,
        9,
        11,
        12,
        15,
        20,
        21,
        23,
        24,
        25,
        29,
        30,
        31,
        32,
        33,
        35,
        39,
        40,
        41,
        46,
        47,
        48,
        49,
        50,
        62,
        75,
        77,
        84,
    ),
    "amazon": (
        3,
        4,
        5,
        6,
        9,
        11,
        12,
        13,
        15,
        18,
        21,
        23,
        24,
        25,
        27,
        29,
        30,
        33,
        34,
        35,
        68,
    ),
    "bolivia": (
        3,
        4,
        6,
        11,
        12,
        13,
        15,
        21,
        23,
        24,
        25,
        29,
        30,
        33,
        34,
        39,
        61,
        66,
        68,
        72,
        81,
        82,
    ),
    "colombia": (
        0,
        3,
        5,
        6,
        9,
        11,
        12,
        13,
        21,
        23,
        24,
        25,
        27,
        29,
        30,
        33,
        34,
        35,
        49,
        50,
        68,
    ),
    "ecuador": (3, 4, 5, 6, 9, 12, 13, 21, 23, 24, 25, 29, 31, 33, 34, 74, 81, 82),
    "paraguay": (3, 4, 6, 9, 11, 12, 15, 18, 25, 33),
    "peru": (
        3,
        4,
        5,
        6,
        9,
        11,
        12,
        13,
        15,
        21,
        23,
        24,
        25,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        40,
        61,
        66,
        68,
        70,
        72,
    ),
    "venezuela": (
        3,
        4,
        5,
        6,
        9,
        11,
        12,
        13,
        15,
        18,
        21,
        23,
        24,
        25,
        29,
        30,
        31,
        32,
        33,
        50,
        66,
        68,
    ),
}

# Products grouped under one catalogue entry. A family's region list is the
# union of its products' regions.
FAMILIES = {
    "fire": ("fire", "fire_severity", "fire_monitor"),
    "degradation": (
        "degradation_edge",
        "degradation_secondary",
        "degradation_logging",
        "degradation_canopy",
    ),
    "water": ("water_annual", "water_monthly"),
}


def _bands(path, band, first, last, collection):
    return {
        "asset": ROOT + path,
        "layout": "bands",
        "band": band,
        "years": (first, last),
        "collection": collection,
    }


def _months(path, layout, band, first, last, month_value, collection):
    record = {
        "asset": ROOT + path,
        "layout": layout,
        "band": band,
        "months": (first, last),
        "years": (int(first[:4]), int(last[:4])),
        "month_property": "month",
        "month_value": month_value,
        "collection": collection,
    }
    if layout == "year_month_collection":
        record["year_property"] = "year"
    return record


_CLS = "classification_{year}"
_DEG = "brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_"

_ASSETS = {
    # --- land use / land cover -------------------------------------------
    ("brazil", "lulc"): _bands(
        "brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3",
        _CLS,
        1985,
        2025,
        "Collection 11",
    ),
    ("amazon", "lulc"): _bands(
        "amazon/lulc/collection6/mapbiomas_collection60_integration_v1",
        _CLS,
        1985,
        2023,
        "Collection 6",
    ),
    ("bolivia", "lulc"): _bands(
        "bolivia/lulc/collection3/mapbiomas_bolivia_collection3_integration_v1",
        _CLS,
        1985,
        2024,
        "Collection 3",
    ),
    ("colombia", "lulc"): _bands(
        "colombia/collection2/mapbiomas_colombia_collection2_integration_v1",
        _CLS,
        1985,
        2023,
        "Collection 2",
    ),
    ("ecuador", "lulc"): _bands(
        "ecuador/lulc/collection4/mapbiomas_ecuador_collection4_coverage_v1",
        _CLS,
        1985,
        2025,
        "Collection 4",
    ),
    ("paraguay", "lulc"): _bands(
        "paraguay/lulc/collection3/mapbiomas_paraguay_collection3_coverage_v1",
        _CLS,
        1985,
        2025,
        "Collection 3",
    ),
    ("peru", "lulc"): _bands(
        "peru/collection3/mapbiomas_peru_collection3_integration_v1",
        _CLS,
        1985,
        2024,
        "Collection 3",
    ),
    ("venezuela", "lulc"): _bands(
        "venezuela/collection2/mapbiomas_venezuela_collection2_integration_v1",
        _CLS,
        1985,
        2023,
        "Collection 2",
    ),
    # --- deforestation / secondary vegetation --------------------------------
    ("brazil", "deforestation"): _bands(
        "brazil/lulc/collection11/mapbiomas_brazil_collection11_deforestation_secondary_vegetation_v5",
        _CLS,
        1987,
        2025,
        "Collection 11",
    ),
    ("bolivia", "deforestation"): _bands(
        "bolivia/lulc/collection3/mapbiomas_bolivia_collection3_deforestation_secondary_vegetation_v1",
        _CLS,
        1985,
        2024,
        "Collection 3",
    ),
    ("paraguay", "deforestation"): _bands(
        "paraguay/lulc/collection3/mapbiomas_paraguay_collection3_deforestation_secondary_vegetation_v1",
        _CLS,
        1985,
        2025,
        "Collection 3",
    ),
    ("peru", "deforestation"): _bands(
        "peru/collection3/mapbiomas_peru_collection3_deforestation_secondary_vegetation_v4",
        _CLS,
        1999,
        2024,
        "Collection 3",
    ),
    # --- fire ----------------------------------------------------------------
    ("brazil", "fire"): _bands(
        "brazil/fire/collection5_1/mapbiomas_fire_collection51_monthly_burned_v1",
        "burned_monthly_{year}",
        1985,
        2025,
        "Fire Collection 5.1",
    ),
    ("colombia", "fire"): _bands(
        "colombia/fire/collection1/mapbiomas_colombia_fire_collection1_monthly_burned_v1",
        "burned_monthly_{year}",
        2000,
        2025,
        "Fire Collection 1",
    ),
    ("paraguay", "fire"): _bands(
        "paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_monthly_burned_v1",
        "burned_monthly_{year}",
        1999,
        2024,
        "Fire Collection 1",
    ),
    ("peru", "fire"): _bands(
        "peru/fire/collection2/mapbiomas_peru_fire_collection2_monthly_burned_v1",
        "burned_monthly_{year}",
        1990,
        2025,
        "Fire Collection 2",
    ),
    ("brazil", "fire_severity"): _bands(
        "brazil/fire/collection5_1/mapbiomas_fire_collection51_severity_class_v1",
        "severity_class_{year}",
        1985,
        2025,
        "Fire Collection 5.1",
    ),
    ("brazil", "fire_monitor"): _months(
        "brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1",
        "year_month_collection",
        "FireMonth",
        "2019-01",
        "2026-10",
        "int",
        "Fire Monitor",
    ),
    # --- degradation (Brazil only) -------------------------------------------
    ("brazil", "degradation_edge"): _bands(
        _DEG + "edge_age_v1", "age_{year}", 1985, 2024, "Collection 10.1"
    ),
    ("brazil", "degradation_secondary"): _bands(
        _DEG + "secondary_vegetation_age_v1",
        "age_{year}",
        1985,
        2024,
        "Collection 10.1",
    ),
    ("brazil", "degradation_logging"): _bands(
        _DEG + "logging_v2", "logging_{year}", 1988, 2024, "Collection 10.1"
    ),
    ("brazil", "degradation_canopy"): _bands(
        _DEG + "canopy_disturbance_frequency_v2",
        "canopy_disturbance_frequency_{year}",
        1988,
        2024,
        "Collection 10.1",
    ),
    # --- surface water ---------------------------------------------------------
    ("brazil", "water_annual"): _bands(
        "brazil/water/collection5/mapbiomas_brazil_collection5_water_annual_v4",
        _CLS,
        1985,
        2025,
        "Water Collection 5",
    ),
    ("bolivia", "water_annual"): _bands(
        "bolivia/water/collection4/mapbiomas_bolivia_collection4_water_v1",
        _CLS,
        1985,
        2025,
        "Water Collection 4",
    ),
    ("peru", "water_annual"): _bands(
        "peru/water/collection3/mapbiomas_peru_collection3_water_v1",
        _CLS,
        1985,
        2024,
        "Water Collection 3",
    ),
    ("brazil", "water_monthly"): _months(
        "brazil/water/collection5/mapbiomas_brazil_collection5_water_monthly_v4",
        "year_month_collection",
        "water_monthly",
        "1985-01",
        "2025-12",
        "zero_padded",
        "Water Collection 5",
    ),
    ("bolivia", "water_monthly"): _months(
        "bolivia/water/collection4/mapbiomas_bolivia_collection4_water_monthly_v1",
        "year_month_collection",
        "water_monthly",
        "1985-01",
        "2025-12",
        "zero_padded",
        "Water Collection 4",
    ),
}


def has_product(region, product):
    """True when MapBiomas publishes ``product`` for ``region``."""
    return (region, product) in _ASSETS


def regions_for(name):
    """Regions offering a product, or any product of a family, in declared order."""
    products = FAMILIES.get(name, (name,))
    return tuple(
        r for r in MAPBIOMAS_REGIONS if any((r, p) in _ASSETS for p in products)
    )


def year_list(region, product):
    """Every year with data for ``(region, product)``, ascending (KeyError if none)."""
    first, last = _ASSETS[(region, product)]["years"]
    return list(range(first, last + 1))


def month_available(region, product, year, month):
    """True when the monthly record covers ``year``-``month`` (False when absent)."""
    record = _ASSETS.get((region, product))
    if record is None or "months" not in record:
        return False
    first, last = record["months"]
    return first <= f"{year}-{month:02d}" <= last


def mapbiomas_asset(region, product):
    """Deep copy of the record for ``(region, product)``.

    Adds ``region``, ``product`` and ``attribution``. LULC records also get
    ``class_table`` (code -> group), ``nodata_codes`` and ``observed_codes``;
    deforestation records get ``class_table`` (code -> class).
    Raises KeyError when MapBiomas has no such layer.
    """
    try:
        record = copy.deepcopy(_ASSETS[(region, product)])
    except KeyError:
        raise KeyError(f"MapBiomas has no {product!r} layer for {region!r}") from None
    record["region"] = region
    record["product"] = product
    record[
        "attribution"
    ] = f"MapBiomas {_REGION_NAMES[region]}, {record['collection']} ({LICENSE})"
    if product == "lulc":
        record["class_table"] = dict(LULC_CODE_TO_GROUP)
        record["nodata_codes"] = LULC_NODATA_CODES
        record["observed_codes"] = _LULC_OBSERVED[region]
    elif product == "deforestation":
        record["class_table"] = dict(DEFO_CLASSES)
    return record


# --- AOI -> region pre-selection (static; no Earth Engine) ---------------------

ISO3_REGIONS = {
    "BRA": "brazil",
    "BOL": "bolivia",
    "COL": "colombia",
    "ECU": "ecuador",
    "PRY": "paraguay",
    "PER": "peru",
    "VEN": "venezuela",
}
# Countries mapped by the Pan-Amazon (RAISG) collection.
PAN_AMAZON_ISO3 = ("BOL", "BRA", "COL", "ECU", "GUF", "GUY", "PER", "SUR", "VEN")
# pysepal admin AOIs name themselves "<ISO3>_<admin1>[_<admin2>]" (see
# pysepal.solara.components.aoi.admin.process_admin). Other methods carry no
# reliable country.
ADMIN_METHODS = ("ADMIN0", "ADMIN1", "ADMIN2")


def aoi_iso3(aoi_result):
    """ISO3 prefix of an admin AOI's name, or None (drawn/asset AOIs, no AOI).

    "UNK" is returned as-is; it simply maps to no region.
    """
    if aoi_result is None:
        return None
    if str(getattr(aoi_result, "method", "")).upper() not in ADMIN_METHODS:
        return None
    match = re.match(r"([A-Z]{3})(?:_|$)", getattr(aoi_result, "name", None) or "")
    return match.group(1) if match else None


def region_for_aoi(name, aoi_result):
    """Region of ``name`` (product or family) that should contain the AOI, or None.

    The country's own collection wins. A Pan-Amazon country falls back to
    ``amazon`` when that is an option. Anything else returns None, so the
    modal keeps the spec default rather than guessing a wrong country.
    """
    iso3 = aoi_iso3(aoi_result)
    if iso3 is None:
        return None
    options = regions_for(name)
    region = ISO3_REGIONS.get(iso3)
    if region in options:
        return region
    if iso3 in PAN_AMAZON_ISO3 and "amazon" in options:
        return "amazon"
    return None


def region_param(name):
    """The ``region`` choice spec for a product or family (A/D param schema).

    It is always written into the name, so the collection is visible. It is
    seeded from the AOI through ``aoi_default`` (see
    ``params.aoi_param_defaults``).
    """
    options = list(regions_for(name))
    return {
        "key": "region",
        "label_key": "vars.modal.param_mapbiomas_region",
        "hint_key": "vars.modal.param_mapbiomas_region_hint",
        "type": "choice",
        "options": options,
        "default": options[0],
        "option_label_key_prefix": "vars.mapbiomas.region_",
        "suffix_prefix": "",
        "advanced": False,
        "omit_default": False,
        "aoi_default": lambda aoi_result: region_for_aoi(name, aoi_result),
    }
