"""MapBiomas predefined layers (catalogue group ``"mapbiomas"``).

Asset facts live in ``mapbiomas_registry``, which is pure data. This module turns
them into Earth Engine expressions, catalogue entries, validators and the
pre-download footprint check. Tests patch ``ee`` on this module (see
``tests/mapbiomas_fake_ee.py``).

Two rules hold for every layer here:

- Builders only build lazy expressions. The single ``getInfo`` is in
  ``region_coverage``, which runs from the entries' ``prepare_download`` hook.
  Part B2's ``download_hooks.prepare_predefined_download`` dispatches that hook
  on the download worker, never in a Solara handler or render.
- Pixels outside the chosen region are nodata, never 0. MapBiomas images are
  masked outside their region. A product that is ``unmask(0)``-ed, so that
  "no event" becomes 0, is re-masked with the region's LULC footprint
  (``_inside``).
"""

import ee

from gui.scripts.predefined.forest_change import LOSS_PALETTE, check_interval
from gui.scripts.predefined.mapbiomas_registry import (
    LULC_CODE_TO_GROUP,
    LULC_GROUP_CODES,
    LULC_GROUP_COLORS,
    LULC_GROUPS,
    LULC_KNOWN_CODES,
    has_product,
    mapbiomas_asset,
    region_param,
    regions_for,
    year_list,
)
from gui.scripts.predefined.periods import month_index, period_param

# Share of the AOI that must lie inside the region's LULC footprint. A GAUL
# admin-1 polygon inside Brazil measured 0.9956 at the 100 m check scale
# (2026-10-09). Below 0.95 the edge of the export would be silent nodata.
MIN_REGION_COVERAGE = 0.95
_COVERAGE_SCALE = 100


def region_label(region):
    """Translated region name, used in error messages."""
    from gui.i18n import t

    return t(f"vars.mapbiomas.region_{region}")


def _region_mask(region):
    """1 inside the region (its LULC band-0 footprint); 0 or masked outside."""
    rec = mapbiomas_asset(region, "lulc")
    return ee.Image(rec["asset"]).select(0).mask().gt(0)


def _inside(image, region):
    return image.updateMask(_region_mask(region))


def _band(rec, year):
    return ee.Image(rec["asset"]).select(rec["band"].format(year=year))


def _month_value(rec, month):
    return month if rec["month_value"] == "int" else f"{month:02d}"


def _month_image(rec, year, month):
    """One month of a monthly record (layouts are documented in mapbiomas_registry)."""
    if rec["layout"] == "year_month_collection":
        return (
            ee.ImageCollection(rec["asset"])
            .filter(ee.Filter.eq(rec["year_property"], year))
            .filter(ee.Filter.eq(rec["month_property"], _month_value(rec, month)))
            .select(rec["band"])
            .max()
        )
    if rec["layout"] == "month_collection":
        image = (
            ee.ImageCollection(rec["asset"])
            .filter(ee.Filter.eq(rec["month_property"], _month_value(rec, month)))
            .first()
        )
        return ee.Image(image).select(rec["band"].format(year=year))
    raise ValueError(f"{rec['asset']} has no monthly layout")


def _finish(image, aoi):
    return image.clip(aoi).rename("B1")


def _aoi_geometry(aoi):
    return aoi if isinstance(aoi, ee.Geometry) else aoi.geometry()


def region_coverage(aoi, region):
    """Fraction (0-1) of the AOI inside ``region``'s footprint (blocking getInfo).

    Call it only from a worker thread. Pixels off the asset count as outside
    (``unmask(0)``). An AOI the region never touches reduces to None, which is
    returned as 0.0.
    """
    stats = (
        _region_mask(region)
        .unmask(0)
        .rename("inside")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=_aoi_geometry(aoi),
            scale=_COVERAGE_SCALE,
            bestEffort=True,
            maxPixels=1e7,
            tileScale=2,
        )
    )
    value = stats.get("inside").getInfo()
    return float(value) if value is not None else 0.0


def _footprint_check(aoi, params, year):
    """``(i18n_key, fmt)`` when the AOI is not inside the region, else None."""
    if aoi is None:
        return None
    region = params["region"]
    coverage = region_coverage(aoi, region)
    if coverage >= MIN_REGION_COVERAGE:
        return None
    return (
        "vars.mapbiomas.error_outside_region",
        {
            "region": region_label(region),
            # Floor (with float slack): 94.9 % must never read "95 %" next to
            # "needs 95 %".
            "pct": int(coverage * 100 + 1e-9),
            "min": int(round(MIN_REGION_COVERAGE * 100)),
        },
    )


def _prepare_mapbiomas_download(var, params):
    """Entry ``prepare_download`` (Part B2 hook): refuse an AOI outside the region.

    B2's ``download_hooks.prepare_predefined_download`` calls it on the download
    worker right before the export, and skips it when the file on disk is
    reused. B2 hooks normally return a warning notice; this one raises instead,
    because exporting a region that does not cover the AOI would silently
    produce a nodata edge. The ``ValueError`` carries the localized message,
    which ``tracked_job`` shows in the download-error toast.
    """
    problem = _footprint_check(
        getattr(var, "aoi", None), params, getattr(var, "year", None)
    )
    if problem is None:
        return None
    from gui.i18n import t

    key, fmt = problem
    raise ValueError(t(key, **fmt))


# --- validator helpers (entry "validate" callables build on these) ------------
# D's validate_entry runs them in Solara renders, so they read the static
# registry only and never call Earth Engine.


def _not_available(region):
    return ("vars.mapbiomas.error_not_available", {"region": region_label(region)})


def _check_year(region, year, years):
    if year is None or year in years:
        return None
    return (
        "vars.mapbiomas.error_year",
        {"region": region_label(region), "first": years[0], "last": years[-1]},
    )


def _check_period(region, product, year, period):
    """A monthly record covers ``year``/``period``; "year" needs all 12 months."""
    if not has_product(region, product):
        return _not_available(region)
    if year is None:
        return None
    first, last = mapbiomas_asset(region, product)["months"]
    month = month_index(period)
    if month is None:
        lo, hi = f"{year}-01", f"{year}-12"
    else:
        lo = hi = f"{year}-{month:02d}"
    if first <= lo and hi <= last:
        return None
    return (
        "vars.mapbiomas.error_period",
        {"region": region_label(region), "first": first, "last": last},
    )


def _require_valid(problem):
    """Builders refuse what the entry's ``validate`` refuses.

    The modal never submits such a combination; this guards hand-edited names
    and direct callers before any Earth Engine expression is built.
    """
    if problem is None:
        return
    from gui.i18n import t

    key, fmt = problem
    raise ValueError(t(key, **fmt))


# --- land use / land cover ------------------------------------------------------

_LULC_DEFAULT_GROUPS = ("forest",)

LULC_MODE_PARAM = {
    "key": "mode",
    "label_key": "vars.modal.param_lulc_mode",
    "hint_key": "vars.modal.param_lulc_mode_hint",
    "type": "choice",
    "options": ["mask", "groups"],
    "default": "mask",
    "option_label_key_prefix": "vars.modal.lulc_mode_",
    "suffix_prefix": "",
    "advanced": False,
    "omit_default": True,
}

LULC_GROUPS_PARAM = {
    "key": "groups",
    "label_key": "vars.modal.param_lulc_groups",
    "hint_key": "vars.modal.param_lulc_groups_hint",
    "type": "multichoice",
    "options": [{"value": g, "code": LULC_GROUP_CODES[g]} for g in LULC_GROUPS],
    "default": list(_LULC_DEFAULT_GROUPS),
    "min_selected": 1,
    "option_label_key_prefix": "vars.mapbiomas.group_",
    "suffix_prefix": "g",
    "advanced": False,
    "omit_default": True,
    "depends_on": {"mode": ["mask"]},
}


def _validate_lulc(params, year):
    region = params["region"]
    return _check_year(region, year, year_list(region, "lulc"))


def _get_mapbiomas_lulc(
    aoi, year, region="brazil", mode="mask", groups=_LULC_DEFAULT_GROUPS
):
    """MapBiomas LULC folded into the five level-1 groups of ``LULC_GROUPS``.

    ``mask``: 1 where the pixel's class belongs to one of ``groups``, else 0.
    ``groups``: 1-5, the group's position in ``LULC_GROUPS``.
    The remap has **no default**, so a code the table does not know (and the
    no-data codes 0 and 27) stays masked and exports as nodata, never as 0.
    """
    _require_valid(_validate_lulc({"region": region}, year))
    codes = list(LULC_KNOWN_CODES)
    if mode == "groups":
        values = [LULC_GROUPS.index(LULC_CODE_TO_GROUP[c]) + 1 for c in codes]
    else:
        chosen = set(groups)
        values = [1 if LULC_CODE_TO_GROUP[c] in chosen else 0 for c in codes]
    band = _band(mapbiomas_asset(region, "lulc"), year)
    return _finish(band.remap(codes, values).toByte(), aoi)


def _lulc_years(params):
    return year_list(params["region"], "lulc")


def _lulc_vis(params):
    if params["mode"] == "groups":
        return {
            "palette": [LULC_GROUP_COLORS[g] for g in LULC_GROUPS],
            "min": 1,
            "max": len(LULC_GROUPS),
        }
    first = next(g for g in LULC_GROUPS if g in params["groups"])
    return {"palette": ["ffffff", LULC_GROUP_COLORS[first]], "min": 0, "max": 1}


def _lulc_class_labels(params):
    if params["mode"] != "groups":
        return None  # the 0/1 mask uses legend_class_keys
    return [
        {
            "value": i + 1,
            "label_key": f"vars.mapbiomas.group_{g}",
            "color": LULC_GROUP_COLORS[g],
        }
        for i, g in enumerate(LULC_GROUPS)
    ]


_LULC_ENTRY = {
    "label_key": "vars.predefined.mapbiomas_lulc",
    "description_key": "vars.predefined_info.mapbiomas_lulc",
    "group": "mapbiomas",
    "var_type": "GEEVar",
    "raster_type": "categorical",
    "temporal": True,
    "years": _lulc_years,
    "default_scale": 30,
    "params": [region_param("lulc"), LULC_MODE_PARAM, LULC_GROUPS_PARAM],
    "get_image": _get_mapbiomas_lulc,
    "vis_params": _lulc_vis,
    "class_labels": _lulc_class_labels,
    "legend_class_keys": ("legend.class.absent", "legend.class.present"),
    "validate": _validate_lulc,
    "prepare_download": _prepare_mapbiomas_download,
}


# --- deforestation (interval target, Part C contract) ------------------------------

# value -> (baseline codes at start - 1, event codes in start .. end - 1).
# DEFO_CLASSES: 2 primary vegetation, 3 secondary vegetation,
# 4 primary suppression, 6 secondary suppression.
DEFO_VEGETATION = {"primary": ((2,), (4,)), "all": ((2, 3), (4, 6))}


def defo_interval_years(region):
    """Start/end years for ``region``: bands first..last -> first+1 .. last+1."""
    years = year_list(region, "deforestation")
    return list(range(years[0] + 1, years[-1] + 2))


# interval_years is static (C), so it is the union over the regions; the
# entry's validate narrows it to the chosen region (review focus 3).
DEFO_INTERVAL_YEARS = sorted(
    {y for r in regions_for("deforestation") for y in defo_interval_years(r)}
)

DEFO_VEGETATION_PARAM = {
    "key": "vegetation",
    "label_key": "vars.modal.param_defo_vegetation",
    "hint_key": "vars.modal.param_defo_vegetation_hint",
    "type": "choice",
    "options": ["primary", "all"],
    "default": "primary",
    "option_label_key_prefix": "vars.modal.defo_vegetation_",
    "suffix_prefix": "",
    "advanced": True,
    "omit_default": True,
}


def _validate_defo(params, year):
    region = params["region"]
    start, end = params.get("start_year"), params.get("end_year")
    if start is None or end is None:
        return None  # C's coerce reports the missing year
    years = defo_interval_years(region)
    if int(start) in years and int(end) in years:
        return None
    return (
        "vars.mapbiomas.error_interval_range",
        {"region": region_label(region), "first": years[0], "last": years[-1]},
    )


def _get_deforestation_mapbiomas(
    aoi,
    year=None,
    region="brazil",
    vegetation="primary",
    start_year=None,
    end_year=None,
):
    """MapBiomas vegetation loss in [start, end): 1 = cleared, 0 = standing.

    Masked = not (primary, or primary/secondary for "all") vegetation in
    ``start - 1``, i.e. on 1 January of ``start`` (C's year convention).
    Outside the region the bands are masked, so the output is too.
    ``year`` is unused (non-temporal interval entry).
    """
    start_year, end_year = check_interval(
        start_year, end_year, defo_interval_years(region)
    )
    rec = mapbiomas_asset(region, "deforestation")
    base_codes, event_codes = (list(c) for c in DEFO_VEGETATION[vegetation])
    baseline = _band(rec, start_year - 1).remap(base_codes, [1] * len(base_codes), 0)
    events = ee.Image(0)
    for y in range(start_year, end_year):
        events = events.Or(_band(rec, y).remap(event_codes, [1] * len(event_codes), 0))
    return _finish(events.updateMask(baseline).toByte(), aoi)


def _defo_forest_at_start(params):
    """LULC native vegetation (forest + other natural) the target pairs with."""
    return (
        "mapbiomas_lulc",
        {"region": params["region"], "mode": "mask", "groups": ["forest", "natural"]},
    )


_DEFO_ENTRY = {
    "label_key": "vars.predefined.deforestation_mapbiomas",
    "description_key": "vars.predefined_info.deforestation_mapbiomas",
    "group": "mapbiomas",
    "var_type": "GEEVar",
    "raster_type": "categorical",
    "temporal": False,
    "interval_years": DEFO_INTERVAL_YEARS,
    "default_scale": 30,
    "params": [region_param("deforestation"), DEFO_VEGETATION_PARAM],
    "get_image": _get_deforestation_mapbiomas,
    "vis_params": {"palette": LOSS_PALETTE, "min": 0, "max": 1},
    "legend_class_keys": ("legend.class.stable", "legend.class.loss"),
    "validate": _validate_defo,
    "forest_at_start": _defo_forest_at_start,
    "prepare_download": _prepare_mapbiomas_download,
}


# --- fire (begin) -----------------------------------------------------------------

FIRE_PRODUCT_PARAM = {
    "key": "product",
    "label_key": "vars.modal.param_fire_product",
    "hint_key": "vars.modal.param_fire_product_hint",
    "type": "choice",
    "options": ["burned", "severity", "since", "frequency"],
    "default": "burned",
    "option_label_key_prefix": "vars.modal.fire_product_",
    "suffix_prefix": "",
    "advanced": False,
    "omit_default": False,
}

FIRE_PERIOD_PARAM = {**period_param(), "depends_on": {"product": ["burned"]}}

FIRE_SOURCE_PARAM = {
    "key": "source",
    "label_key": "vars.modal.param_mapbiomas_fire_source",
    "hint_key": "vars.modal.param_mapbiomas_fire_source_hint",
    "type": "choice",
    "options": ["collection", "monitor"],
    "default": "collection",
    "option_label_key_prefix": "vars.modal.mapbiomas_fire_source_",
    "suffix_prefix": "",
    "advanced": True,
    "omit_default": True,
    "depends_on": {"product": ["burned"]},
}

_FIRE_HISTORY = ("since", "frequency")

# max 25: Paraguay and Colombia have 26 years of fire data, so every region
# still offers at least two years (first + window .. last + 1).
FIRE_WINDOW_PARAM = {
    "key": "window",
    "label_key": "vars.modal.param_fire_window",
    "hint_key": "vars.modal.param_fire_window_hint",
    "type": "int",
    "default": 10,
    "min": 1,
    "max": 25,
    "suffix_prefix": "w",
    "advanced": False,
    "omit_default": True,
    "depends_on": {"product": list(_FIRE_HISTORY)},
}

FIRE_SEVERITY_CLASS_LABELS = [
    {"value": n, "label_key": f"vars.mapbiomas.severity_{n}", "color": color}
    for n, color in enumerate(
        ["ffffff", "ffffb2", "fecc5c", "fd8d3c", "f03b20", "bd0026"]
    )
]


def _period_years(region, product, period):
    """Years whose ``period`` the monthly record covers ("year" = all 12 months)."""
    first, last = mapbiomas_asset(region, product)["years"]
    return [
        y
        for y in range(first, last + 1)
        if _check_period(region, product, y, period) is None
    ]


def _uses_monitor(params):
    return params["product"] == "burned" and params["source"] == "monitor"


def _fire_years(params):
    region, product = params["region"], params["product"]
    if _uses_monitor(params) and has_product(region, "fire_monitor"):
        return _period_years(region, "fire_monitor", params["period"])
    if product == "severity" and has_product(region, "fire_severity"):
        return year_list(region, "fire_severity")
    if product in _FIRE_HISTORY:
        years = year_list(region, "fire")
        return list(range(years[0] + int(params["window"]), years[-1] + 2))
    # A combination the region lacks keeps the collection's years; validate
    # reports it as not available.
    return year_list(region, "fire")


def _validate_fire(params, year):
    region = params["region"]
    if params["product"] == "severity" and not has_product(region, "fire_severity"):
        return _not_available(region)
    if _uses_monitor(params):
        return _check_period(region, "fire_monitor", year, params["period"])
    return _check_year(region, year, _fire_years(params))


def _burned(region, year, period, source):
    """1 = burned in the year/month; masked = not burned (caller unmasks)."""
    month = month_index(period)
    if source == "monitor":
        rec = mapbiomas_asset(region, "fire_monitor")
        if month is None:
            image = (
                ee.ImageCollection(rec["asset"])
                .filter(ee.Filter.eq(rec["year_property"], year))
                .select(rec["band"])
                .max()
            )
        else:
            image = _month_image(rec, year, month)
        return image.gt(0)
    # Collection band = month of the burn (1-12), masked where nothing burned.
    band = _band(mapbiomas_asset(region, "fire"), year)
    return band.gt(0) if month is None else band.eq(month)


def _fire_history(region, year, product, window):
    """Years since the last burn, or burned-year count, over year-window .. year-1."""
    rec = mapbiomas_asset(region, "fire")
    years = list(range(year - window, year))
    # Same band name on every image, so the collection reducers line up.
    flags = [_band(rec, y).gt(0).unmask(0).rename("b") for y in years]
    if product == "since":
        last = ee.ImageCollection(
            [f.multiply(y).toInt16() for f, y in zip(flags, years)]
        ).max()
        return ee.Image(year).subtract(last).where(last.eq(0), window + 1)
    return ee.ImageCollection([f.toInt16() for f in flags]).sum()


def _get_mapbiomas_fire(
    aoi,
    year,
    region="brazil",
    product="burned",
    period="year",
    source="collection",
    window=10,
):
    """MapBiomas Fire for the chosen region.

    burned: 0/1 for the year or month; severity: 0 = unburned, 1-5; since:
    years since the last burn (window + 1 = none in the window); frequency:
    burned years in the window. ``unmask(0)`` makes "nothing burned" 0, then
    ``_inside`` re-masks everything outside the region (nodata, not 0).
    """
    params = {
        "region": region,
        "product": product,
        "period": period,
        "source": source,
        "window": window,
    }
    _require_valid(_validate_fire(params, year))
    if product in _FIRE_HISTORY:
        image = _fire_history(region, year, product, int(window))
        return _finish(_inside(image, region).toInt16(), aoi)
    if product == "severity":
        image = _band(mapbiomas_asset(region, "fire_severity"), year)
    else:
        image = _burned(region, year, period, source)
    return _finish(_inside(image.unmask(0), region).toByte(), aoi)


def _fire_raster_type(params):
    return "continuous" if params["product"] in _FIRE_HISTORY else "categorical"


def _fire_vis(params):
    product, window = params["product"], int(params["window"])
    if product == "since":
        return {
            "palette": ["d73027", "fc8d59", "fee08b", "ffffbf"],
            "min": 1,
            "max": window + 1,
        }
    if product == "frequency":
        return {
            "palette": ["ffffff", "fdd49e", "fc8d59", "b30000"],
            "min": 0,
            "max": window,
        }
    if product == "severity":
        return {
            "palette": [c["color"] for c in FIRE_SEVERITY_CLASS_LABELS],
            "min": 0,
            "max": 5,
        }
    return {"palette": ["ffffff", "d84315"], "min": 0, "max": 1}


def _fire_unit_key(params):
    return {
        "since": "legend.unit.years_value",
        "frequency": "legend.unit.plain_value",
    }.get(params["product"])


def _fire_class_labels(params):
    return FIRE_SEVERITY_CLASS_LABELS if params["product"] == "severity" else None


_FIRE_ENTRY = {
    "label_key": "vars.predefined.mapbiomas_fire",
    "description_key": "vars.predefined_info.mapbiomas_fire",
    "group": "mapbiomas",
    "var_type": "GEEVar",
    "raster_type": _fire_raster_type,
    "temporal": True,
    "years": _fire_years,
    "default_scale": 30,
    "params": [
        region_param("fire"),
        FIRE_PRODUCT_PARAM,
        FIRE_PERIOD_PARAM,
        FIRE_SOURCE_PARAM,
        FIRE_WINDOW_PARAM,
    ],
    "get_image": _get_mapbiomas_fire,
    "vis_params": _fire_vis,
    "unit_key": _fire_unit_key,
    "class_labels": _fire_class_labels,
    "legend_class_keys": ("legend.class.unburned", "legend.class.burned"),
    "validate": _validate_fire,
    "prepare_download": _prepare_mapbiomas_download,
}

# --- fire (end) -------------------------------------------------------------------


# --- degradation (Brazil) -----------------------------------------------------------

DEGRADATION_VARIABLES = ("edge", "secondary", "logging", "canopy")

DEGRADATION_VARIABLE_PARAM = {
    "key": "variable",
    "label_key": "vars.modal.param_degradation_variable",
    "hint_key": "vars.modal.param_degradation_variable_hint",
    "type": "choice",
    "options": list(DEGRADATION_VARIABLES),
    "default": "edge",
    "option_label_key_prefix": "vars.modal.degradation_variable_",
    "suffix_prefix": "",
    "advanced": False,
    "omit_default": False,
}


def _degradation_years(params):
    return year_list(params["region"], f"degradation_{params['variable']}")


def _validate_degradation(params, year):
    return _check_year(params["region"], year, _degradation_years(params))


def _get_mapbiomas_degradation(aoi, year, region="brazil", variable="edge"):
    """MapBiomas Brazil degradation indicators (Collection 10.1).

    edge: edge age in years, left masked away from edges. secondary: the band
    holds ``age * 100 + class``, so the age is ``floor(code / 100)``; 0 = not
    secondary vegetation. logging: 0/1. canopy: disturbance count. The last
    three are ``unmask(0)`` then re-masked to Brazil (nodata outside).
    """
    params = {"region": region, "variable": variable}
    _require_valid(_validate_degradation(params, year))
    band = _band(mapbiomas_asset(region, f"degradation_{variable}"), year)
    if variable == "edge":
        return _finish(band.toInt16(), aoi)
    if variable == "secondary":
        image = _inside(band.divide(100).floor().unmask(0), region).toInt16()
    elif variable == "logging":
        image = _inside(band.gt(0).unmask(0), region).toByte()
    else:
        image = _inside(band.unmask(0), region).toInt16()
    return _finish(image, aoi)


def _degradation_raster_type(params):
    return "categorical" if params["variable"] == "logging" else "continuous"


def _degradation_vis(params):
    return {
        "edge": {"palette": ["fff7bc", "fec44f", "d95f0e"], "min": 0, "max": 40},
        "secondary": {"palette": ["edf8e9", "74c476", "006d2c"], "min": 0, "max": 40},
        "logging": {"palette": ["ffffff", "8e24aa"], "min": 0, "max": 1},
        "canopy": {
            "palette": ["ffffff", "fdae6b", "e6550d", "7f2704"],
            "min": 0,
            "max": 10,
        },
    }[params["variable"]]


def _degradation_unit_key(params):
    return {
        "edge": "legend.unit.years_value",
        "secondary": "legend.unit.years_value",
        "canopy": "legend.unit.plain_value",
    }.get(params["variable"])


_DEGRADATION_ENTRY = {
    "label_key": "vars.predefined.mapbiomas_degradation",
    "description_key": "vars.predefined_info.mapbiomas_degradation",
    "group": "mapbiomas",
    "var_type": "GEEVar",
    "raster_type": _degradation_raster_type,
    "temporal": True,
    "years": _degradation_years,
    "default_scale": 30,
    "params": [region_param("degradation"), DEGRADATION_VARIABLE_PARAM],
    "get_image": _get_mapbiomas_degradation,
    "vis_params": _degradation_vis,
    "unit_key": _degradation_unit_key,
    "legend_class_keys": ("legend.class.absent", "legend.class.present"),
    "validate": _validate_degradation,
    "prepare_download": _prepare_mapbiomas_download,
}


# --- surface water ----------------------------------------------------------------


def _water_years(params):
    region, period = params["region"], params["period"]
    if month_index(period) is not None and has_product(region, "water_monthly"):
        return _period_years(region, "water_monthly", period)
    # A month the region lacks keeps the annual years; validate reports it.
    return year_list(region, "water_annual")


def _validate_water(params, year):
    region, period = params["region"], params["period"]
    if month_index(period) is not None:
        return _check_period(region, "water_monthly", year, period)
    return _check_year(region, year, year_list(region, "water_annual"))


def _get_mapbiomas_water(aoi, year, region="brazil", period="year"):
    """MapBiomas Water: 1 = water in the year (or month), 0 = none, nodata outside."""
    _require_valid(_validate_water({"region": region, "period": period}, year))
    month = month_index(period)
    if month is None:
        image = _band(mapbiomas_asset(region, "water_annual"), year)
    else:
        image = _month_image(mapbiomas_asset(region, "water_monthly"), year, month)
    return _finish(_inside(image.gt(0).unmask(0), region).toByte(), aoi)


_WATER_ENTRY = {
    "label_key": "vars.predefined.mapbiomas_water",
    "description_key": "vars.predefined_info.mapbiomas_water",
    "group": "mapbiomas",
    "var_type": "GEEVar",
    "raster_type": "categorical",
    "temporal": True,
    "years": _water_years,
    "default_scale": 30,
    "params": [region_param("water"), period_param()],
    "get_image": _get_mapbiomas_water,
    "vis_params": {"palette": ["ffffff", "2532e4"], "min": 0, "max": 1},
    "legend_class_keys": ("legend.class.absent", "legend.class.present"),
    "validate": _validate_water,
    "prepare_download": _prepare_mapbiomas_download,
}


# Order = order in the layer dropdown under the "MapBiomas" header.
ENTRIES = {
    "mapbiomas_lulc": _LULC_ENTRY,
    "deforestation_mapbiomas": _DEFO_ENTRY,
    "mapbiomas_fire": _FIRE_ENTRY,
    "mapbiomas_degradation": _DEGRADATION_ENTRY,
    "mapbiomas_water": _WATER_ENTRY,
}
