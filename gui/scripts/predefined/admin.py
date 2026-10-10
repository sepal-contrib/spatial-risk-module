"""Administrative layers: FAO GAUL subjurisdictions."""

import ee


def _get_subj(aoi, year=None):
    """FAO GAUL level-2 subjurisdiction — categorical raster."""
    from spatialrisk.gee.ee_fao_gaul import get_fao_gaul_subj
    from spatialrisk.gee.ee_rasterize_unique_values import gee_rasterize_unique_values

    filtered_subj, _ = get_fao_gaul_subj(2, aoi)
    return (
        ee.Image(gee_rasterize_unique_values(filtered_subj, "gaul2_name"))
        .clip(aoi)
        .toByte()
    )


ENTRIES = {
    "subj": {
        "label_key": "vars.predefined.subj",
        "description_key": "vars.predefined_info.subj",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "get_image": _get_subj,
        # Many subjurisdictions with arbitrary rasterized values -> random RGB.
        "random_visualizer": True,
    },
}
