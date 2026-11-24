"""Forest layers: Hansen GFC and JRC TMF forest cover."""

import ee

from gui.scripts.predefined.aoi import _aoi_bounds


def _get_forest_gfc(aoi, year, tree_cover_threshold=30):
    """Hansen Global Forest Change — forest cover at a given year.

    ``tree_cover_threshold`` is the minimum canopy cover percentage in 2000 that
    counts as forest. It is user-selectable from the Add Variable modal; the
    default here must match the catalogue's declared default (see ``params``
    on the ``forest_gfc`` entry), which is the single source of truth.
    """
    gfc = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").clip(aoi)
    forest2000 = gfc.select("treecover2000")
    forest2000_thr = (
        ee.Image(0).where(forest2000.gte(tree_cover_threshold), 1).clip(aoi)
    )
    loss = gfc.select("lossyear")
    return forest2000_thr.where(loss.lt(year - 2000), 0).rename("B1")


# JRC TMF v1_2024 AnnualChanges (bands Dec1990..Dec2024). Part C imports it.
TMF_ANNUAL_CHANGES = "projects/JRC/TMF/v1_2024/AnnualChanges"

# JRC TMF AnnualChanges classes: 1 undisturbed, 2 degraded, 3 deforested,
# 4 regrowth, 5 water, 6 other. Each definition lists the classes kept as
# forest; dict order is the option order in the modal.
TMF_FOREST_CLASSES = {
    "undisturbed": (1,),
    "degraded": (1, 2),
    "regrowth": (1, 2, 4),
}


def tmf_definition_param():
    """The TMF forest-definition param spec (fresh dict; shared with Part C)."""
    return {
        "key": "definition",
        "label_key": "vars.modal.param_tmf_definition",
        "hint_key": "vars.modal.param_tmf_definition_hint",
        "type": "choice",
        "default": "undisturbed",
        "options": list(TMF_FOREST_CLASSES),
        "option_label_key_prefix": "vars.modal.tmf_def_",
        "suffix_prefix": "",
        "advanced": True,
    }


def _get_forest_tmf(aoi, year, definition="undisturbed"):
    """JRC Tropical Moist Forest — forest cover at 1 January of ``year``.

    Reads the AnnualChanges ``Dec{year-1}`` band and reduces it to a binary
    forest mask (1 = forest, 0 = non-forest) using the classes of
    ``definition`` (see ``TMF_FOREST_CLASSES``). The default ``undisturbed``
    keeps the expression inherited from notebooks/1.variables_factory.ipynb
    verbatim — it reduces to class 1 only, because ``where(neq(1), 0)``
    reads the original band — so default downloads stay byte-identical. The
    kwarg default must match the catalogue spec's ``default``.
    """
    tmf = ee.ImageCollection(TMF_ANNUAL_CHANGES).filterBounds(_aoi_bounds(aoi)).mosaic()
    band = tmf.select("Dec" + str(year - 1))
    if definition == "undisturbed":
        forest = band.where(band.eq(2), 1).where(band.neq(1), 0)
    else:
        classes = list(TMF_FOREST_CLASSES[definition])
        forest = band.remap(classes, [1] * len(classes), 0)
    return forest.clip(aoi).rename("B1")


ENTRIES = {
    "forest_gfc": {
        "label_key": "vars.predefined.forest_gfc",
        "description_key": "vars.predefined_info.forest_gfc",
        "group": "forest",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        # Forest at 1 Jan of ``year`` needs loss years 1..year-2001, so the GFC
        # release bounds this list: v1.13 carries lossyear up to 25 (2025) and
        # therefore supports years up to 2026. Bump this alongside the asset ID
        # in ``_get_forest_gfc`` whenever a new GFC version is adopted.
        # 2001 is the first: Hansen records no loss in 2000, so a period from
        # 2000 would count one year with no data and understate the rate.
        "years": list(range(2001, 2027)),
        # User-selectable knobs, rendered generically by the Add Variable modal
        # (same shape as MODEL_REGISTRY["params"] in gui/tile/train_tile.py).
        # ``suffix_prefix`` makes the value part of the variable name
        # (forest_gfc_tc30) so two forest definitions coexist as separate
        # variables — see ``build_predefined_name`` / ``resolve_predefined``.
        #
        # CONSTRAINT: a parameter segment must never contain four consecutive
        # digits. Downstream parsing pulls years out of variable names with
        # ``re.findall(r"\d{4}")`` (spatialrisk.evaluation.interval_from_target),
        # so a four-digit run is read as a year. The ONLY four-digit runs a
        # catalogue name may carry are the trailing ``_{start}_{end}`` pair of an
        # interval layer (``year`` specs synthesised from ``interval_years`` by
        # params.interval_param_specs), e.g. deforestation_gfc_tc30_2015_2020,
        # which interval_from_target reads as 5 years. ``max`` below keeps every
        # int value at three digits or fewer; tests/test_predefined_interval.py
        # lints both rules for every entry.
        "params": [
            {
                "key": "tree_cover_threshold",
                "label_key": "vars.modal.param_tree_cover_threshold",
                "hint_key": "vars.modal.param_tree_cover_threshold_hint",
                "type": "int",
                "default": 30,
                "min": 1,
                "max": 100,
                "suffix_prefix": "tc",
            }
        ],
        "get_image": _get_forest_gfc,
        "vis_params": {"palette": ["ffffff", "2e7d32"], "min": 0, "max": 1},
        "legend_class_keys": ("legend.class.non_forest", "legend.class.forest"),
    },
    "forest_tmf": {
        "label_key": "vars.predefined.forest_tmf",
        "description_key": "vars.predefined_info.forest_tmf",
        "group": "forest",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        # Forest at 1 Jan of ``year`` reads AnnualChanges Dec{year-1}; TMF
        # v1_2024 carries Dec1990..Dec2024, so years run to 2025. Bump this
        # alongside the asset ID in ``_get_forest_tmf``.
        "years": list(range(2001, 2026)),
        "params": [tmf_definition_param()],
        "get_image": _get_forest_tmf,
        "vis_params": {"palette": ["ffffff", "2e7d32"], "min": 0, "max": 1},
        "legend_class_keys": ("legend.class.non_forest", "legend.class.forest"),
    },
}
