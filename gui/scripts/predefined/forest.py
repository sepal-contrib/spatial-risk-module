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


def _get_forest_tmf(aoi, year):
    """JRC Tropical Moist Forest — forest cover at a given year.

    Reads the AnnualChanges ``Dec{year-1}`` band and reduces it to a binary
    forest mask (1 = forest, 0 = non-forest), matching the temporal forest_gfc
    structure. The TMF AnnualChange classes are 1 = undisturbed, 2 = degraded,
    3 = deforested, 4 = regrowth, 5 = water, 6 = other. This expression keeps
    class 1 as forest: ``where(eq(2), 1)`` is overwritten by the subsequent
    ``where(neq(1), 0)`` (which references the original band), so degraded
    pixels resolve to 0 — preserving the behaviour of notebooks/
    1.variables_factory.ipynb verbatim.
    """
    tmf = (
        ee.ImageCollection("projects/JRC/TMF/v1_2024/AnnualChanges")
        .filterBounds(_aoi_bounds(aoi))
        .mosaic()
    )
    band = tmf.select("Dec" + str(year - 1))
    return band.where(band.eq(2), 1).where(band.neq(1), 0).clip(aoi).rename("B1")


ENTRIES = {
    "forest_gfc": {
        "label_key": "vars.predefined.forest_gfc",
        "description_key": "vars.predefined_info.forest_gfc",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        # Forest at 1 Jan of ``year`` needs loss years 1..year-2001, so the GFC
        # release bounds this list: v1.13 carries lossyear up to 25 (2025) and
        # therefore supports years up to 2026. Bump this alongside the asset ID
        # in ``_get_forest_gfc`` whenever a new GFC version is adopted.
        "years": list(range(2001, 2027)),
        # User-selectable knobs, rendered generically by the Add Variable modal
        # (same shape as MODEL_REGISTRY["params"] in gui/tile/train_tile.py).
        # ``suffix_prefix`` makes the value part of the variable name
        # (forest_gfc_tc30) so two forest definitions coexist as separate
        # variables — see ``build_predefined_name`` / ``resolve_predefined``.
        #
        # CONSTRAINT: a suffix value must never reach four digits. Downstream
        # name parsing pulls years out of variable names with
        # ``re.findall(r"\d{4}")`` (spatialrisk.evaluation.interval_from_target),
        # so four consecutive digits in a parameter suffix would be read as a
        # year and corrupt the derived change-layer interval. ``max`` below
        # keeps every value at three digits or fewer; keep it that way for any
        # param added here.
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
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        "years": list(range(2001, 2025)),
        "get_image": _get_forest_tmf,
        "vis_params": {"palette": ["ffffff", "2e7d32"], "min": 0, "max": 1},
        "legend_class_keys": ("legend.class.non_forest", "legend.class.forest"),
    },
}
