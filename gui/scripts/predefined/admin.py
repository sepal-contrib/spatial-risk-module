"""Administrative units: the ``subj`` layer (FAO GAUL 2024, sat-io catalogue)."""

import ee

# ``level`` param value -> GAUL level.
GAUL_LEVELS = {"province": 1, "district": 2}


def _subj_units(aoi, level):
    """(units, name attribute, numbered image) for the AOI at one GAUL level.

    Units are numbered 1..n by distinct *name* (``gee_rasterize_unique_values``),
    so two units sharing a name share a value: a pre-B2 quirk kept for parity
    (2 of 508 districts on a Legal-Amazon box, verified 2026-10-09). The image
    is clipped to the AOI and not yet cast.
    """
    from spatialrisk.gee.ee_fao_gaul import get_fao_gaul_subj
    from spatialrisk.gee.ee_rasterize_unique_values import gee_rasterize_unique_values

    if level not in GAUL_LEVELS:
        raise ValueError(f"Unknown administrative level: {level!r}")
    units, attribute = get_fao_gaul_subj(GAUL_LEVELS[level], aoi)
    numbered = ee.Image(gee_rasterize_unique_values(units, attribute)).clip(aoi)
    return units, attribute, numbered


def _get_subj(aoi, year=None, level="district"):
    """Administrative units as a categorical byte raster (default GAUL level 2)."""
    _units, _attribute, numbered = _subj_units(aoi, level)
    return numbered.toByte()


# Unit values run 1..n. The categorical export fills nodata with 255
# (GEEVar._resolve_export_nodata) and toByte() saturates rather than wraps
# (verified 2026-10-09: ee.Image(300).toByte() == 255), so a byte raster keeps
# at most 254 distinct units; every unit from 255 on would become nodata.
SUBJ_BYTE_MAX_UNITS = 254
SUBJ_WIDE_NODATA = 65535
# riskmapjnr encodes vulnerability_class * 1000 + subj_id
# (spatialrisk/rmj/vulnerability_map.py): JNR separates at most 999 units.
JNR_MAX_SUBJ_UNITS = 999


def _count_units(units, attribute):
    """Distinct unit names = the largest value the numbering assigns. Blocking."""
    return int(units.aggregate_array(attribute).distinct().size().getInfo())


def _prepare_subj_download(var, params):
    """Widen the export to uint16 when the AOI has more units than a byte holds.

    Runs on the download worker (one ``getInfo``, ~3 s for 500 districts),
    never in a Solara handler. Up to 254 units the byte layer is left exactly
    as built. Returns an (i18n key, kwargs) notice when it widened, else None.
    """
    units, attribute, numbered = _subj_units(var.aoi, params["level"])
    n_units = _count_units(units, attribute)
    if n_units <= SUBJ_BYTE_MAX_UNITS:
        return None
    var.gee_images = [numbered.toUint16()]
    var.export_nodata = SUBJ_WIDE_NODATA
    if n_units <= JNR_MAX_SUBJ_UNITS:
        key = "vars.subj_notice.wide"
    else:
        key = "vars.subj_notice.wide_jnr"
    return key, {"name": var.name, "n": n_units}


ENTRIES = {
    "subj": {
        "group": "admin",
        "label_key": "vars.predefined.subj",
        "description_key": "vars.predefined_info.subj",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "params": [
            {
                "key": "level",
                "label_key": "vars.modal.param_subj_level",
                "hint_key": "vars.modal.param_subj_level_hint",
                "type": "choice",
                "default": "district",
                "options": ["province", "district"],
                "option_label_key_prefix": "vars.modal.subj_level_",
                "suffix_prefix": "",
                "advanced": True,
                "omit_default": True,
            }
        ],
        "get_image": _get_subj,
        # Download-time widening above 254 units (download_hooks.py).
        "prepare_download": _prepare_subj_download,
        # Many units with arbitrary rasterized values -> random RGB.
        "random_visualizer": True,
    },
}
