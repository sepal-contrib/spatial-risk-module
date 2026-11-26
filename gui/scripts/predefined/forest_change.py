"""Forest-change layers: deforestation / degradation targets.

Interval entries (``interval_years``) are non-temporal: the period is part of
the name (``deforestation_gfc_tc30_2015_2020``, see
``params.interval_param_specs``), the storage key is the name, and the output
follows the change-layer convention of
``spatialrisk.processing.process_change_xarray``: 1 = event, 0 = stable,
masked (exported as 255 by ``GEEVar._resolve_export_nodata``) = not forest at
the start of the period.

Year convention, shared with ``forest_gfc`` / ``forest_tmf``: a layer "at year
Y" is the state on 1 January of Y, so the period ``start``-``end`` holds the
events of the calendar years ``start`` ... ``end - 1``.
"""

import ee

from gui.scripts.predefined.aoi import _aoi_bounds
from gui.scripts.predefined.forest import ENTRIES as _FOREST_ENTRIES
from gui.scripts.predefined.forest import (
    TMF_ANNUAL_CHANGES,
    TMF_FOREST_CLASSES,
    _get_forest_gfc,
    tmf_definition_param,
)

# Interval years follow the forest layer each target pairs with, so a
# forest-at-start layer exists for every selectable start, and a GFC version
# bump (asset ID + years in forest.py) stays a one-place edit.
GFC_YEARS = list(_FOREST_ENTRIES["forest_gfc"]["years"])

#: The Process step's loss look (gui/scripts/postprocess_styles.py "loss"):
#: stable = neutral grey so the extent reads, event = red.
LOSS_PALETTE = ["d9d9d9", "e31a1c"]

# TMF_ANNUAL_CHANGES (bands Dec1990..Dec2024, verified 2026-10-09) comes from
# forest.py so the asset id lives in one place. The selectable years are
# forest_tmf's (2001-2025), so a forest-at-start layer exists for every
# interval start; the data itself reaches back to Dec1990.
TMF_YEARS = list(_FOREST_ENTRIES["forest_tmf"]["years"])

# AnnualChanges classes. 0 is "no data" and is NOT masked in the asset.
TMF_NO_DATA = 0
TMF_UNDISTURBED = 1
TMF_DEGRADED = 2
# Deforested land, regrowth (only reachable through clearing) and new water
# (TMF records conversion to water as deforestation): together they agree
# with the DeforestationYear band on 99.98 % of events (verified 2026-10-09).
TMF_DEFORESTED_CLASSES = (3, 4, 5)

#: Degradation event colour (orange) on the stable grey of LOSS_PALETTE.
DEGRADATION_PALETTE = ["d9d9d9", "ff7f00"]


def check_interval(start_year, end_year, years):
    """``(start, end)`` as ints, or ``ValueError`` before any Earth Engine call.

    Both years must be in ``years`` and ``start < end``; the modal and
    ``coerce_param_values`` enforce the same rules, this guards direct callers.
    """
    if start_year is None or end_year is None:
        raise ValueError("Interval layers need both start_year and end_year.")
    start_year, end_year = int(start_year), int(end_year)
    if start_year not in years or end_year not in years:
        raise ValueError(
            f"Years must lie within {years[0]}-{years[-1]}; "
            f"got {start_year}-{end_year}."
        )
    if start_year >= end_year:
        raise ValueError(
            f"The end year must be later than the start year "
            f"(got {start_year}-{end_year})."
        )
    return start_year, end_year


def _get_deforestation_gfc(
    aoi, year=None, tree_cover_threshold=30, start_year=None, end_year=None
):
    """Hansen deforestation in [start, end): 1 = lost, 0 = kept, masked = not forest.

    Built from the very ``_get_forest_gfc`` expressions the Process step
    downloads for ``forest_gfc_tc<T>`` at ``start`` and ``end``, so it equals
    ``loss_forest_gfc_tc<T>_<start>_<end>`` pixel for pixel: forest at start
    with ``lossyear`` in ``[start - 2000, end - 2001]`` -> 1, forest at start
    never lost or lost from ``end`` on -> 0, everything else masked. See
    tests/test_forest_change_gfc.py for the truth table and the comparison
    with ``process_change_xarray``. ``year`` is unused (non-temporal entry).
    """
    start_year, end_year = check_interval(start_year, end_year, GFC_YEARS)
    forest_start = _get_forest_gfc(aoi, start_year, tree_cover_threshold)
    forest_end = _get_forest_gfc(aoi, end_year, tree_cover_threshold)
    return forest_end.Not().updateMask(forest_start.eq(1)).rename("B1")


def _tmf_annual_changes(aoi):
    """The AnnualChanges mosaic over the AOI's bounds (see ``_aoi_bounds``)."""
    return (
        ee.ImageCollection(TMF_ANNUAL_CHANGES).filterBounds(_aoi_bounds(aoi)).mosaic()
    )


def _in_classes(band, classes):
    """1 where ``band`` is one of ``classes``, else 0."""
    out = band.eq(classes[0])
    for value in classes[1:]:
        out = out.Or(band.eq(value))
    return out


def _get_deforestation_tmf(
    aoi, year=None, definition="undisturbed", start_year=None, end_year=None
):
    """TMF deforestation in [start, end): 1 = cleared, 0 = still standing.

    Forest at start = ``TMF_FOREST_CLASSES[definition]`` on ``Dec{start-1}``.
    Event = the class on ``Dec{end-1}`` is deforested land, regrowth or water
    AND differs from the start class (so "regrowth at start, regrowth at end"
    under the ``regrowth`` definition is not an event). Degradation alone is
    NOT deforestation: forest degraded during the period stays 0, which is why
    this layer differs from a Process-step ``loss_forest_tmf_*`` layer. Masked
    (255 on export): not forest at start, or no data (class 0) at the end.
    """
    start_year, end_year = check_interval(start_year, end_year, TMF_YEARS)
    changes = _tmf_annual_changes(aoi)
    start = changes.select(f"Dec{start_year - 1}")
    end = changes.select(f"Dec{end_year - 1}")
    forest_start = _in_classes(start, TMF_FOREST_CLASSES[definition])
    event = _in_classes(end, TMF_DEFORESTED_CLASSES).And(end.neq(start))
    valid = forest_start.And(end.neq(TMF_NO_DATA))
    return event.updateMask(valid).clip(aoi).rename("B1")


def _get_degradation_tmf(aoi, year=None, start_year=None, end_year=None):
    """TMF degradation in [start, end): 1 = degraded, 0 = still undisturbed.

    At risk = undisturbed (class 1) on ``Dec{start-1}``. Event = degraded
    (class 2) on any of ``Dec{start}``..``Dec{end-1}``, including forest
    degraded and then cleared (equals the DegradationYear band, verified
    2026-10-09). Stable = still undisturbed on ``Dec{end-1}``. Masked (255):
    everything else, including forest cleared without passing through
    degradation (a competing risk, not a "stayed undisturbed" pixel).
    """
    start_year, end_year = check_interval(start_year, end_year, TMF_YEARS)
    changes = _tmf_annual_changes(aoi)
    start = changes.select(f"Dec{start_year - 1}")
    end = changes.select(f"Dec{end_year - 1}")
    degraded = changes.select(f"Dec{start_year}").eq(TMF_DEGRADED)
    for y in range(start_year + 1, end_year):
        degraded = degraded.Or(changes.select(f"Dec{y}").eq(TMF_DEGRADED))
    keep = start.eq(TMF_UNDISTURBED).And(degraded.Or(end.eq(TMF_UNDISTURBED)))
    return degraded.updateMask(keep).clip(aoi).rename("B1")


def _tmf_forest_at_start(params):
    """deforestation_tmf pairs with forest_tmf under the same definition."""
    return "forest_tmf", {"definition": params["definition"]}


def _undisturbed_forest_at_start(params):
    """degradation_tmf's at-risk population is undisturbed forest_tmf."""
    return "forest_tmf", {"definition": "undisturbed"}


def _gfc_forest_at_start(params):
    """The forest layer a GFC deforestation target pairs with (evaluation)."""
    return "forest_gfc", {"tree_cover_threshold": params["tree_cover_threshold"]}


ENTRIES = {
    "deforestation_gfc": {
        "group": "forest_change",
        "label_key": "vars.predefined.deforestation_gfc",
        "description_key": "vars.predefined_info.deforestation_gfc",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        # Start/end year selects and the trailing _{start}_{end} name pair are
        # synthesised from this list (params.interval_param_specs).
        "interval_years": GFC_YEARS,
        # The forest_gfc knob, copied so the two entries never share a dict.
        "params": [dict(_FOREST_ENTRIES["forest_gfc"]["params"][0])],
        "get_image": _get_deforestation_gfc,
        "vis_params": {"palette": LOSS_PALETTE, "min": 0, "max": 1},
        "legend_class_keys": ("legend.class.stable", "legend.class.loss"),
        "forest_at_start": _gfc_forest_at_start,
    },
    "deforestation_tmf": {
        "group": "forest_change",
        "label_key": "vars.predefined.deforestation_tmf",
        "description_key": "vars.predefined_info.deforestation_tmf",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "interval_years": TMF_YEARS,
        # Same knob (and therefore the same name segment) as forest_tmf.
        "params": [tmf_definition_param()],
        "get_image": _get_deforestation_tmf,
        "vis_params": {"palette": LOSS_PALETTE, "min": 0, "max": 1},
        "legend_class_keys": ("legend.class.stable", "legend.class.loss"),
        "forest_at_start": _tmf_forest_at_start,
    },
    "degradation_tmf": {
        "group": "forest_change",
        "label_key": "vars.predefined.degradation_tmf",
        "description_key": "vars.predefined_info.degradation_tmf",
        "var_type": "GEEVar",
        "raster_type": "categorical",
        "temporal": False,
        "interval_years": TMF_YEARS,
        "get_image": _get_degradation_tmf,
        "vis_params": {"palette": DEGRADATION_PALETTE, "min": 0, "max": 1},
        "legend_class_keys": ("legend.class.stable", "legend.class.degraded"),
        "forest_at_start": _undisturbed_forest_at_start,
    },
}
