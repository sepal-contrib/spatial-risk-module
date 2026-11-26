"""Climate layers: precipitation, 2 m temperature and dry spells, by period."""

import ee

from gui.scripts.predefined.periods import (
    availability_fields,
    lookback_days,
    period_param,
    period_range,
)

# ERA5-Land (Copernicus Climate Change Service). Complete months verified
# 2026-10-09: the monthly and the daily aggregates both run 1950-02 -> 2026-09.
# Bump ERA_WINDOW's last month when a new month lands. The daily sums over a
# month equal the monthly total exactly (verified), so monthly precipitation
# keeps using MONTHLY_AGGR.
ERA_MONTHLY = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
ERA_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA_WINDOW = ("1950-02", "2026-09")

# CHIRPS daily rainfall (Climate Hazards Center, UCSB), mm/day, 50S-50N only:
# an AOI outside that band gets an all-nodata layer. Complete months verified
# 2026-10-09: 1981-01 -> 2026-08. Bump CHIRPS_WINDOW's last month as data lands.
CHIRPS_DAILY = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_WINDOW = ("1981-01", "2026-08")

#: Rainfall sources keyed by the ``source`` choice value. "era", not "era5":
#: choice values must be letters only (no digits in variable-name segments).
RAIN_AVAILABILITY = {"era": ERA_WINDOW, "chirps": CHIRPS_WINDOW}
#: Native pixel size per source, in metres (~0.1 deg / ~0.05 deg).
RAIN_SCALE = {"era": 11132, "chirps": 5566}


def _rain_source_param(default):
    """Advanced ``source`` choice shared by the rainfall-based layers."""
    return {
        "key": "source",
        "label_key": "vars.modal.param_rain_source",
        "hint_key": "vars.modal.param_rain_source_hint",
        "type": "choice",
        "default": default,
        "options": ["era", "chirps"],
        "option_label_key_prefix": "vars.modal.rain_source_",
        "suffix_prefix": "",
        "advanced": True,
    }


def _get_precipitation(aoi, year, period="year", source="era"):
    """Total precipitation (mm) over the period: the whole year or one month.

    ``era`` (default) sums the ERA5-Land monthly totals
    (``total_precipitation_sum``, metres of water) and converts to mm. At the
    defaults this is exactly the pre-period expression, so the legacy
    ``precipitation`` name keeps its raster. ``chirps`` sums CHIRPS daily
    rainfall, already in mm. Whole mm as int16: sub-mm precision is
    meaningless at these pixel sizes and float64 quadruples the export.
    The kwarg defaults must match the catalogue spec defaults.
    """
    start, end = period_range(year, period)
    if source == "chirps":
        col = (
            ee.ImageCollection(CHIRPS_DAILY)
            .filterDate(start, end)
            .select("precipitation")
        )
        return col.sum().round().toInt16().clip(aoi).rename("B1")
    col = (
        ee.ImageCollection(ERA_MONTHLY)
        .filterDate(start, end)
        .select("total_precipitation_sum")
    )
    return col.sum().multiply(1000).round().toInt16().clip(aoi).rename("B1")


def _get_temperature(aoi, year, aggregation="median", period="year"):
    """ERA5-Land 2 m air temperature (°C) over the period.

    A year reduces the 12 monthly means (MONTHLY_AGGR), exactly the
    pre-period expression; a month reduces its daily means (DAILY_AGGR), so
    the metric still means something for a single month. The Kelvin offset is
    subtracted after the reduction (a constant shift commutes with
    mean/max/min/median). Whole degrees as int16 (signed: temperatures go
    negative). The kwarg defaults must match the catalogue spec defaults.
    """
    start, end = period_range(year, period)
    collection_id = ERA_MONTHLY if period == "year" else ERA_DAILY
    col = (
        ee.ImageCollection(collection_id)
        .filterDate(start, end)
        .select("temperature_2m")
    )
    reduced = {
        "mean": col.mean,
        "max": col.max,
        "min": col.min,
        "median": col.median,
    }[aggregation]()
    return reduced.subtract(273.15).round().toInt16().clip(aoi).rename("B1")


def _dry_spell_needs(year, params):
    """Data interval of dry_spell: the look-back window before the period."""
    return lookback_days(year, params["period"], params["window"])


def _get_dry_spell(
    aoi, year, metric="dsr", threshold=1, window=365, period="year", source="chirps"
):
    """Dryness at the period start, from daily rain over the look-back window.

    The window is ``[period start - window days, period start)``: the period
    itself is never read, so a September fire target cannot leak into it.

    * ``dsr``: days since the last day with at least ``threshold`` mm, counted
      back from the period start (0 = it rained the day before), capped at
      ``window`` when it never rained in the window;
    * ``drydays``: days in the window with less than ``threshold`` mm;
    * ``raintotal``: rain over the window in mm (``threshold`` unused, so it
      is a conditional param that is hidden and omitted).

    ``chirps`` (default, mm/day, 50S-50N only) or ``era`` (ERA5-Land daily
    totals in metres). Pixels with no observation at all (sea for ERA5-Land,
    beyond 50 deg for CHIRPS) stay masked rather than reading as "never
    rained". Whole numbers as int16. The kwarg defaults must match the
    catalogue spec defaults.
    """
    start, end = lookback_days(year, period, window)
    if source == "chirps":
        daily = (
            ee.ImageCollection(CHIRPS_DAILY)
            .filterDate(start, end)
            .select("precipitation")
        )
        to_mm = 1
    else:
        daily = (
            ee.ImageCollection(ERA_DAILY)
            .filterDate(start, end)
            .select("total_precipitation_sum")
        )
        to_mm = 1000

    if metric == "raintotal":
        out = daily.sum().multiply(to_mm).round()
    elif metric == "drydays":
        out = daily.map(lambda img: img.multiply(to_mm).lt(threshold)).sum()
    else:
        ref = ee.Date(end)

        def _days_before_start(img):
            wet = img.multiply(to_mm).gte(threshold)
            lag = ref.difference(img.date(), "day").subtract(1)
            # One pixel type for the whole collection (EE refuses to reduce
            # a mix), masked where the day was dry.
            return ee.Image.constant(lag).toInt16().updateMask(wet).rename("B1")

        out = daily.map(_days_before_start).min().unmask(window)
    return out.updateMask(daily.count().gt(0)).toInt16().clip(aoi).rename("B1")


ENTRIES = {
    "precipitation": {
        "label_key": "vars.predefined.precipitation",
        "description_key": "vars.predefined_info.precipitation",
        "group": "climate",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": True,
        "params": [period_param(), _rain_source_param("era")],
        # availability / years / validate: years whose period the chosen
        # source fully covers (2001-2025 at the defaults, as before).
        **availability_fields(RAIN_AVAILABILITY),
        "get_image": _get_precipitation,
        # Native resolution of the chosen source. Without it the download
        # path's ``default_scale or 30`` fallback would export ~11 km pixels at
        # 30 m. Read through resolved_entry (gui/tile/variables_tile.py).
        "default_scale": lambda params: RAIN_SCALE[params["source"]],
        # mm, AOI-dependent range -> palette-only dynamic stretch.
        "vis_params": {"palette": ["ffffff", "c6dbef", "4292c6", "08306b"]},
    },
    "temperature_2m": {
        "label_key": "vars.predefined.temperature_2m",
        "description_key": "vars.predefined_info.temperature_2m",
        "group": "climate",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": True,
        # The metric stays a required segment (temperature_2m_median), so
        # existing names keep resolving; the period is omitted at "year".
        # Options must stay letters only (no digits, no underscores).
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
            },
            period_param(),
        ],
        **availability_fields({"era": ERA_WINDOW}),
        "get_image": _get_temperature,
        "default_scale": 11132,
        # °C -> thermal ramp, dynamic stretch.
        "vis_params": {"palette": ["313695", "74add1", "fee090", "f46d43", "a50026"]},
    },
    "dry_spell": {
        "label_key": "vars.predefined.dry_spell",
        "description_key": "vars.predefined_info.dry_spell",
        "group": "climate",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": True,
        # Antecedent indicator: measured at the period start, looking back.
        # Very source- and threshold-sensitive (verified 2026-10-09, Rondônia,
        # 2024-09-01, 365 d: CHIRPS mean 7.7 d since rain vs ERA5-Land 62.5 d).
        "params": [
            {
                "key": "metric",
                "label_key": "vars.modal.param_dry_metric",
                "hint_key": "vars.modal.param_dry_metric_hint",
                "type": "choice",
                "default": "dsr",
                "options": ["dsr", "drydays", "raintotal"],
                "option_label_key_prefix": "vars.modal.dry_metric_",
                "suffix_prefix": "",
                "omit_default": True,
            },
            {
                "key": "threshold",
                "label_key": "vars.modal.param_rain_threshold",
                "hint_key": "vars.modal.param_rain_threshold_hint",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 50,
                "suffix_prefix": "t",
                "omit_default": True,
                # Rain total has no threshold: hidden and left out of the name.
                "depends_on": {"metric": ["dsr", "drydays"]},
            },
            {
                "key": "window",
                "label_key": "vars.modal.param_dry_window",
                "hint_key": "vars.modal.param_dry_window_hint",
                "type": "int",
                "default": 365,
                "min": 7,
                "max": 999,
                "suffix_prefix": "w",
                "omit_default": True,
            },
            period_param(),
            _rain_source_param("chirps"),
        ],
        **availability_fields(RAIN_AVAILABILITY, _dry_spell_needs),
        "get_image": _get_dry_spell,
        "default_scale": lambda params: RAIN_SCALE[params["source"]],
        "vis_params": lambda params: (
            {"palette": ["ffffff", "c6dbef", "4292c6", "08306b"]}
            if params["metric"] == "raintotal"
            else {"palette": ["ffffcc", "fd8d3c", "800026"]}
        ),
        "unit_key": lambda params: (
            "legend.unit.mm_value"
            if params["metric"] == "raintotal"
            else "legend.unit.days_value"
        ),
    },
}
