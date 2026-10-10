"""Water layers: OSM rivers."""

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
}
