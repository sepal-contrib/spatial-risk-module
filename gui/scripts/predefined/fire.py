"""Fire layers: burned area, active fire detections, fire history.

Every builder returns one band named ``B1``. Categorical layers are 0/1 bytes
(GEEVar exports them with nodata 255); continuous layers are int16 (nodata
-32768) or float (FRP). Availability windows are a static snapshot verified
2026-10-09 against the Earth Engine catalogue: bump the last month of each
window as new data lands (complete months only).
"""

import ee

from gui.scripts.predefined.periods import (
    availability_fields,
    lookback_years,
    period_param,
    period_range,
)

# MODIS Collection 6.1 burned area (500 m, monthly). Unburned pixels are
# MASKED, not 0, so the builders unmask to 0.
MCD64A1 = "MODIS/061/MCD64A1"
# VIIRS (Suomi NPP) Collection 2 burned area, same layout as MCD64A1. The old
# NOAA/VIIRS/001/VNP64A1 is deprecated (2014-2018 only): do not use it.
VNP64A1 = "NASA/VIIRS/002/VNP64A1"

#: ``sensor`` choice value -> (collection id, burn-date band).
BURN_PRODUCTS = {
    "modis": (MCD64A1, "BurnDate"),
    "viirs": (VNP64A1, "Burn_Date"),
}
#: Complete months per sensor (verified 2026-10-09). Bump as data lands.
BURN_AVAILABILITY = {
    "modis": ("2000-11", "2026-07"),
    "viirs": ("2012-03", "2026-08"),
}

# Daily active-fire products, all at ~926 m (verified: VNP14A1 is NOT 375 m).
# FireMask classes 7/8/9 = fire with low/nominal/high confidence. MaxFRP is in
# tenths of MW on MODIS (x 0.1) and already in MW on VIIRS. Terra (MOD14A1)
# starts 2000-03 and Aqua (MYD14A1) 2002-07: before 2002-07 "modis" is
# Terra-only, so detection counts are not comparable across that date.
MOD14 = "MODIS/061/MOD14A1"
MYD14 = "MODIS/061/MYD14A1"
VNP14 = "NASA/VIIRS/002/VNP14A1"
# NASA FIRMS (MODIS-based rasterised hotspots): non-fire pixels masked,
# ``confidence`` 0-100, NO fire radiative power band.
FIRMS = "FIRMS"

FIREMASK_MIN = {"low": 7, "nominal": 8, "high": 9}
FIRMS_CONFIDENCE_MIN = {"low": 0, "nominal": 30, "high": 80}
FRP_TO_MW = {"modis": 0.1, "viirs": 1}
#: Complete months per source (verified 2026-10-09). Bump as data lands.
ACTIVE_AVAILABILITY = {
    "modis": ("2000-03", "2026-08"),
    "viirs": ("2012-02", "2026-09"),
    "firms": ("2000-11", "2026-09"),
}


def _fire_collection(source, start, end):
    """Daily fire images of ``source`` over ``[start, end)``."""
    if source == "modis":
        col = ee.ImageCollection(MOD14).merge(ee.ImageCollection(MYD14))
    elif source == "viirs":
        col = ee.ImageCollection(VNP14)
    else:
        col = ee.ImageCollection(FIRMS)
    return col.filterDate(start, end)


def _fire_flag(source, confidence):
    """Per-image function: 1 where a fire of at least ``confidence``, else 0.

    Cast to int16 so every image of a collection has one pixel type (EE refuses
    to reduce a mix of Byte<0,1> and Short).
    """
    if source == "firms":
        band, level = "confidence", FIRMS_CONFIDENCE_MIN[confidence]
    else:
        band, level = "FireMask", FIREMASK_MIN[confidence]

    def _flag(img):
        # sameFootprint=False: with the default footprint the unmasked daily
        # MODIS tiles composite to all zeros (verified live 2026-11-24).
        flag = img.select(band).gte(level).unmask(0, False)
        return flag.toInt16().rename("B1")

    return _flag


def _fire_days(col, flag, start, end):
    """Number of days in ``[start, end)`` with at least one flagged image.

    Flags are combined per day first: Terra and Aqua both seeing one fire on
    the same day is one fire day, not two (verified on a probe pixel: 5
    detections on 4 days).
    """
    first = ee.Date(start)
    n_days = ee.Date(end).difference(first, "day")
    zero = ee.Image.constant(0).toInt16().rename("B1")

    def _one_day(offset):
        day = first.advance(offset, "day")
        flags = col.filterDate(day, day.advance(1, "day")).map(flag)
        # A day without any image still yields a 0 image.
        return flags.merge(ee.ImageCollection([zero])).max()

    days = ee.List.sequence(0, n_days.subtract(1)).map(_one_day)
    return ee.ImageCollection(days).sum()


def _get_active_fire(
    aoi, year, metric="any", period="year", confidence="nominal", source="modis"
):
    """Active fire over the period, from daily detections.

    * ``any``: 1 where at least one fire was detected (categorical);
    * ``days``: number of days with a detection (int16);
    * ``frp``: maximum fire radiative power in MW (float), 0 without fire.
      FIRMS has no FRP band, so ``frp`` + ``firms`` raises ValueError (the
      modal refuses it first through the entry's ``validate``).

    The kwarg defaults must match the catalogue spec defaults.
    """
    if metric == "frp" and source == "firms":
        raise ValueError(
            "NASA FIRMS has no fire radiative power band: use MODIS or VIIRS."
        )
    start, end = period_range(year, period)
    col = _fire_collection(source, start, end)
    if metric == "frp":
        level = FIREMASK_MIN[confidence]
        to_mw = FRP_TO_MW[source]

        def _frp(img):
            fires = img.select("FireMask").gte(level)
            return (
                img.select("MaxFRP")
                .multiply(to_mw)
                .updateMask(fires)
                .toFloat()
                .rename("B1")
            )

        return col.map(_frp).max().unmask(0).toFloat().clip(aoi).rename("B1")
    flag = _fire_flag(source, confidence)
    if metric == "days":
        return _fire_days(col, flag, start, end).toInt16().clip(aoi).rename("B1")
    return col.map(flag).max().unmask(0).toByte().clip(aoi).rename("B1")


_ACTIVE_FIELDS = availability_fields(ACTIVE_AVAILABILITY)


def _validate_active_fire(params, year):
    """FRP needs MODIS or VIIRS; otherwise the source's months must cover it."""
    if params["metric"] == "frp" and params["source"] == "firms":
        return ("vars.modal.error_frp_firms", {})
    return _ACTIVE_FIELDS["validate"](params, year)


def _fire_year(source, start, end):
    """0/1 int16: did ``source`` see fire anywhere in ``[start, end)``?"""
    if source == "burned":
        seen = (
            ee.ImageCollection(MCD64A1)
            .filterDate(start, end)
            .select("BurnDate")
            .max()
            .gt(0)
        )
    else:
        seen = (
            _fire_collection("modis", start, end)
            .map(_fire_flag("modis", "nominal"))
            .max()
            .gt(0)
        )
    return seen.unmask(0).toInt16().rename("B1")


def _history_needs(year, params):
    """Data interval of fire_history: the look-back window before the period."""
    return lookback_years(year, params["period"], params["window"])


def _get_fire_history(
    aoi, year, metric="count", window=10, period="year", source="burned"
):
    """Fire history over the ``window`` years before the period start.

    Slice k (k = 1..window) is the 12 months ending k-1 years before the period
    start, so ``window`` slices tile ``[start - window years, start)`` and the
    period itself is never read.

    * ``count``: number of slices with fire (0..window);
    * ``since``: years since the nearest slice with fire (0 = within the 12
      months before the period), ``window`` when none burned.

    ``burned`` uses MODIS burned area (MCD64A1); ``active`` uses MODIS
    nominal-confidence detections. int16 (continuous; nodata -32768). The
    kwarg defaults must match the catalogue spec defaults.
    """
    slices = [
        _fire_year(source, *lookback_years(int(year) - k + 1, period, 1))
        for k in range(1, int(window) + 1)
    ]
    if metric == "since":
        ages = [
            ee.Image.constant(k - 1).toInt16().updateMask(burned).rename("B1")
            for k, burned in enumerate(slices, start=1)
        ]
        out = ee.ImageCollection(ages).min().unmask(window)
    else:
        out = ee.ImageCollection(slices).sum()
    return out.toInt16().clip(aoi).rename("B1")


#: Complete months per fire_history source (verified 2026-10-09).
HISTORY_AVAILABILITY = {
    "burned": BURN_AVAILABILITY["modis"],
    "active": ACTIVE_AVAILABILITY["modis"],
}

_FIRE_MASK_VIS = {"palette": ["ffffff", "d84315"], "min": 0, "max": 1}
_FIRE_DAYS_VIS = {"palette": ["ffffcc", "fd8d3c", "bd0026"]}
_FIRE_FRP_VIS = {"palette": ["ffffcc", "fe9929", "cc4c02", "662506"]}


def _burn_sensor_param():
    return {
        "key": "sensor",
        "label_key": "vars.modal.param_burn_sensor",
        "hint_key": "vars.modal.param_burn_sensor_hint",
        "type": "choice",
        "default": "modis",
        "options": ["modis", "viirs"],
        "option_label_key_prefix": "vars.modal.burn_sensor_",
        "suffix_prefix": "",
        "advanced": True,
    }


def _get_burned_area(aoi, year, period="year", sensor="modis"):
    """1 where the product mapped a burn in the period, else 0.

    ``year`` = the 12 monthly maps of that year; a month = that month's map.
    The kwarg defaults must match the catalogue spec defaults.
    """
    collection_id, band = BURN_PRODUCTS[sensor]
    start, end = period_range(year, period)
    col = ee.ImageCollection(collection_id).filterDate(start, end).select(band)
    return col.max().gt(0).unmask(0).toByte().clip(aoi).rename("B1")


ENTRIES = {
    "burned_area": {
        "label_key": "vars.predefined.burned_area",
        "description_key": "vars.predefined_info.burned_area",
        "group": "fire",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": True,
        "params": [period_param(), _burn_sensor_param()],
        **availability_fields(BURN_AVAILABILITY),
        "get_image": _get_burned_area,
        # Native ~463 m sinusoidal pixels.
        "default_scale": 463,
        "vis_params": {"palette": ["ffffff", "d84315"], "min": 0, "max": 1},
        "legend_class_keys": ("legend.class.unburned", "legend.class.burned"),
    },
    "active_fire": {
        "label_key": "vars.predefined.active_fire",
        "description_key": "vars.predefined_info.active_fire",
        "group": "fire",
        "var_type": "GEEVar",
        # 0/1 detections are categorical; day counts and FRP are continuous.
        "raster_type": lambda params: (
            "categorical" if params["metric"] == "any" else "continuous"
        ),
        "temporal": True,
        "params": [
            {
                "key": "metric",
                "label_key": "vars.modal.param_fire_metric",
                "hint_key": "vars.modal.param_fire_metric_hint",
                "type": "choice",
                "default": "any",
                "options": ["any", "days", "frp"],
                "option_label_key_prefix": "vars.modal.fire_metric_",
                "suffix_prefix": "",
                "omit_default": True,
            },
            period_param(),
            {
                "key": "confidence",
                "label_key": "vars.modal.param_fire_confidence",
                "hint_key": "vars.modal.param_fire_confidence_hint",
                "type": "choice",
                "default": "nominal",
                "options": ["low", "nominal", "high"],
                "option_label_key_prefix": "vars.modal.fire_confidence_",
                "suffix_prefix": "",
                "advanced": True,
            },
            {
                "key": "source",
                "label_key": "vars.modal.param_fire_source",
                "hint_key": "vars.modal.param_fire_source_hint",
                "type": "choice",
                "default": "modis",
                "options": ["modis", "viirs", "firms"],
                "option_label_key_prefix": "vars.modal.fire_source_",
                "suffix_prefix": "",
                "advanced": True,
            },
        ],
        **_ACTIVE_FIELDS,
        "validate": _validate_active_fire,
        "get_image": _get_active_fire,
        # Native ~926 m for every source.
        "default_scale": 926,
        "vis_params": lambda params: {
            "any": _FIRE_MASK_VIS,
            "days": _FIRE_DAYS_VIS,
            "frp": _FIRE_FRP_VIS,
        }[params["metric"]],
        "unit_key": lambda params: {
            "any": "legend.unit.plain_value",
            "days": "legend.unit.days_value",
            "frp": "legend.unit.mw_value",
        }[params["metric"]],
        "legend_class_keys": ("legend.class.no_fire", "legend.class.fire"),
    },
    "fire_history": {
        "label_key": "vars.predefined.fire_history",
        "description_key": "vars.predefined_info.fire_history",
        "group": "fire",
        "var_type": "GEEVar",
        "raster_type": "continuous",
        "temporal": True,
        "params": [
            {
                "key": "metric",
                "label_key": "vars.modal.param_history_metric",
                "hint_key": "vars.modal.param_history_metric_hint",
                "type": "choice",
                "default": "count",
                "options": ["count", "since"],
                "option_label_key_prefix": "vars.modal.history_metric_",
                "suffix_prefix": "",
                "omit_default": True,
            },
            {
                "key": "window",
                "label_key": "vars.modal.param_history_window",
                "hint_key": "vars.modal.param_history_window_hint",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 20,
                "suffix_prefix": "y",
                "omit_default": True,
            },
            period_param(),
            {
                "key": "source",
                "label_key": "vars.modal.param_history_source",
                "hint_key": "vars.modal.param_history_source_hint",
                "type": "choice",
                "default": "burned",
                "options": ["burned", "active"],
                "option_label_key_prefix": "vars.modal.history_source_",
                "suffix_prefix": "",
                "advanced": True,
            },
        ],
        **availability_fields(HISTORY_AVAILABILITY, _history_needs),
        "get_image": _get_fire_history,
        "default_scale": lambda params: 463 if params["source"] == "burned" else 926,
        # count: more fire = darker; since: recent fire = darker.
        "vis_params": lambda params: (
            {"palette": ["ffffff", "fdae61", "d7191c", "67001f"]}
            if params["metric"] == "count"
            else {"palette": ["67001f", "d7191c", "fdae61", "ffffff"]}
        ),
        "unit_key": "legend.unit.years_value",
    },
}
