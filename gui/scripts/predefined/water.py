"""Water layers: OSM rivers and JRC surface water."""

import ee

from gui.scripts.predefined.aoi import _aoi_bounds

# OSM Water Layer (projects/sat-io/open-datasets/OSM_waterLayer, 90 m) class
# ids. The option order below is the canonical code order in variable names.
OSM_WATER_CLASSES = {"large": 2, "major": 3, "canal": 4, "stream": 5, "ocean": 1}
DEFAULT_WATER_CLASSES = ("large", "major", "canal", "stream")


def _get_rivers(aoi, year=None, classes=DEFAULT_WATER_CLASSES):
    """OSM water layer — binary mask (1 = one of ``classes``).

    Classes: 1 ocean, 2 large lake/river, 3 major river, 4 canal, 5 small
    stream. The default (every class but ocean) keeps the original ``gte(2)``
    expression verbatim so default downloads stay byte-identical; any other
    selection remaps the chosen ids to 1 and everything else to 0. The kwarg
    default must match the catalogue spec's ``default``.
    """
    water = (
        ee.ImageCollection("projects/sat-io/open-datasets/OSM_waterLayer")
        .filterBounds(_aoi_bounds(aoi))
        .mosaic()
        .clip(aoi)
    )
    if tuple(classes) == DEFAULT_WATER_CLASSES:
        selected = water.gte(2)
    else:
        ids = [OSM_WATER_CLASSES[c] for c in classes]
        selected = water.remap(ids, [1] * len(ids), 0)
    return selected.unmask().clip(aoi).toByte()


# JRC Global Surface Water v1.4 (Pekel et al. 2016, Nature 540: 418-422),
# 1984-2021 at 30 m. Only GSW1_4 exists (verified 2026-10-09); its
# YearlyHistory ends 2021-01 and MonthlyHistory 2021-12. Bump here when JRC
# publishes a newer release.
GSW_ASSET = "JRC/GSW1_4/GlobalSurfaceWater"
# ``transition`` classes (codes 1-10 all present, verified 2026-10-09; names
# from the EE catalogue, not stored in the asset): 1 permanent, 2 new
# permanent, 3 lost permanent, 4 seasonal, 5 new seasonal, 6 lost seasonal,
# 7 seasonal->permanent, 8 permanent->seasonal, 9 ephemeral permanent,
# 10 ephemeral seasonal. A type groups the classes whose *current* state it
# is; lost water (3, 6) is in no group and never counts.
GSW_TYPE_CLASSES = {
    "permanent": (1, 2, 7),
    "seasonal": (4, 5, 8),
    "ephemeral": (9, 10),
}
GSW_TYPE_CODES = {"permanent": "p", "seasonal": "s", "ephemeral": "e"}
SW_DEFAULT_TYPES = ["permanent", "seasonal"]


def _get_surface_water(aoi, year=None, occurrence=1, water_types=SW_DEFAULT_TYPES):
    """JRC surface water — binary mask (1 = water).

    A pixel counts when its water occurrence (% of valid 1984-2021
    observations) is >= ``occurrence`` and its transition class belongs to one
    of ``water_types``. Pixels that were never water are masked in GSW and
    become 0 through ``unmask(0)``.
    """
    gsw = ee.Image(GSW_ASSET)
    classes = [code for kind in water_types for code in GSW_TYPE_CLASSES[kind]]
    of_type = gsw.select("transition").remap(classes, [1] * len(classes), 0)
    water = gsw.select("occurrence").gte(occurrence).And(of_type)
    return water.unmask(0).clip(aoi).toByte()


def _surface_water_params():
    """Occurrence threshold and water types (advanced, omitted at defaults)."""
    return [
        {
            "key": "occurrence",
            "label_key": "vars.modal.param_sw_occurrence",
            "hint_key": "vars.modal.param_sw_occurrence_hint",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 100,
            "suffix_prefix": "o",
            "advanced": True,
            "omit_default": True,
        },
        {
            "key": "water_types",
            "label_key": "vars.modal.param_sw_types",
            "hint_key": "vars.modal.param_sw_types_hint",
            "type": "multichoice",
            "default": list(SW_DEFAULT_TYPES),
            "options": [
                {"value": value, "code": GSW_TYPE_CODES[value]}
                for value in GSW_TYPE_CLASSES
            ],
            "option_label_key_prefix": "vars.modal.sw_type_",
            "min_selected": 1,
            "suffix_prefix": "t",
            "advanced": True,
            "omit_default": True,
        },
    ]


ENTRIES = {
    "rivers": {
        "label_key": "vars.predefined.rivers",
        "description_key": "vars.predefined_info.rivers",
        "group": "water",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "params": [
            {
                "key": "classes",
                "label_key": "vars.modal.param_water_classes",
                "hint_key": "vars.modal.param_water_classes_hint",
                "type": "multichoice",
                "default": list(DEFAULT_WATER_CLASSES),
                "options": [
                    {"value": "large", "code": "l"},
                    {"value": "major", "code": "m"},
                    {"value": "canal", "code": "c"},
                    {"value": "stream", "code": "s"},
                    {"value": "ocean", "code": "o"},
                ],
                "option_label_key_prefix": "vars.modal.water_",
                "suffix_prefix": "w",
                "advanced": True,
            }
        ],
        "get_image": _get_rivers,
        "vis_params": {"palette": ["ffffff", "2196f3"], "min": 0, "max": 1},
    },
    "surface_water": {
        "group": "water",
        "label_key": "vars.predefined.surface_water",
        "description_key": "vars.predefined_info.surface_water",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "params": _surface_water_params(),
        "get_image": _get_surface_water,
        "vis_params": {"palette": ["ffffff", "1565c0"], "min": 0, "max": 1},
    },
}
