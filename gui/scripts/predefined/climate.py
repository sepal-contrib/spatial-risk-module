"""Climate layers: ERA5-Land precipitation and 2 m temperature."""

import ee


def _get_precipitation(aoi, year):
    """ERA5-Land annual precipitation (mm/year): sum of the 12 monthly totals.

    ``total_precipitation_sum`` is the monthly total in metres of water depth;
    the yearly sum is converted to millimetres and rounded to whole mm (int16 —
    sub-mm precision is meaningless at ~11 km pixels and float64 quadruples the
    exported raster size).
    """
    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select("total_precipitation_sum")
    )
    return col.sum().multiply(1000).round().toInt16().clip(aoi).rename("B1")


def _get_temperature(aoi, year, aggregation="median"):
    """ERA5-Land 2m temperature (°C): monthly means reduced by ``aggregation``.

    ``temperature_2m`` is the monthly-averaged air temperature in Kelvin. The
    12 monthly images are reduced per pixel with the chosen metric; the Kelvin
    offset is subtracted after the reduction (a constant shift commutes with
    mean/max/min/median, and one subtract builds a smaller EE expression).
    The kwarg default must match the catalogue spec's ``default`` — the
    catalogue is the single source of truth (same note as ``_get_forest_gfc``).
    """
    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select("temperature_2m")
    )
    reduced = {
        "mean": col.mean,
        "max": col.max,
        "min": col.min,
        "median": col.median,
    }[aggregation]()
    # Whole degrees C as int16 (signed — temperatures go negative); float64
    # decimals are noise at ~11 km pixels and quadruple the raster size.
    return reduced.subtract(273.15).round().toInt16().clip(aoi).rename("B1")


ENTRIES = {
    "precipitation": {
        "label_key": "vars.predefined.precipitation",
        "description_key": "vars.predefined_info.precipitation",
        "group": "climate",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": True,
        "years": list(range(2001, 2026)),
        "get_image": _get_precipitation,
        # ERA5-Land native resolution (~0.1 deg). Without this the download
        # path's ``default_scale or 30`` fallback would export ~11 km pixels
        # at 30 m. Picked up by _build_predefined (gui/tile/variables_tile.py).
        "default_scale": 11132,
        # mm/year, AOI-dependent range -> palette-only dynamic stretch.
        "vis_params": {"palette": ["ffffff", "c6dbef", "4292c6", "08306b"]},
    },
    "temperature_2m": {
        "label_key": "vars.predefined.temperature_2m",
        "description_key": "vars.predefined_info.temperature_2m",
        "group": "climate",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": True,
        "years": list(range(2001, 2026)),
        # The metric is a choice param baked into the variable name
        # (temperature_2m_median) so two metrics coexist as separate
        # variables. Options must stay digit-free — see the four-digit
        # suffix constraint on forest_gfc's params in forest.py.
        "params": [
            {
                "key": "aggregation",
                "label_key": "vars.modal.param_aggregation",
                "hint_key": "vars.modal.param_aggregation_hint",
                "type": "choice",
                "default": "median",
                "options": ["mean", "max", "min", "median"],
                # Modal option label: t(option_label_key_prefix + option).
                "option_label_key_prefix": "vars.modal.agg_",
                "suffix_prefix": "",
            }
        ],
        "get_image": _get_temperature,
        "default_scale": 11132,
        # °C -> thermal ramp, dynamic stretch.
        "vis_params": {"palette": ["313695", "74add1", "fee090", "f46d43", "a50026"]},
    },
}
