"""Value probes for the Variables modal: a 2-98 % stretch and the class list.

An Earth Engine asset is read over the AOI's bounding rectangle; a local
raster is read whole, decimated. Both read the first band, the one a variable
uses. Every probe blocks (``gee_interface.get_info`` or a file read): run it
on a worker thread. Errors are i18n keys, never translated here.

Ported from the deforestation alerts app (gui/scripts/vis_probes.py).
"""

from typing import Optional, Tuple

import numpy as np

from gui.scripts.variable_classes import MAX_CLASSES

PROBE_MAX_PIXELS = 10_000_000
FILE_PROBE_SIZE = 1024  # the longest side of a local raster's sample, in pixels
PERCENTILES = (2, 98)
_BAND = "b"  # the first band, renamed so the reducers' keys are known

ERR_NO_VALUES = "vars.modal.err_no_values"
ERR_NOT_INTEGER = "vars.modal.err_not_integer"
ERR_TOO_MANY = "vars.modal.err_too_many_classes"
ERR_NO_STRETCH = "vars.modal.err_no_stretch"
ERR_TOO_MANY_FEATURES = "vars.modal.err_too_many_features"


def _region(image, aoi) -> dict:
    """The ``reduceRegion`` arguments both Earth Engine probes share."""
    from gui.scripts.predefined_variables import _aoi_bounds

    return {
        # The rectangle, never the AOI itself: a heavy AOI geometry inlined
        # in the request fails it (see _aoi_bounds).
        "geometry": _aoi_bounds(aoi),
        "scale": image.projection().nominalScale(),
        "bestEffort": True,
        "maxPixels": PROBE_MAX_PIXELS,
    }


def gee_stretch(gee_interface, asset: str, aoi) -> dict:
    """``{"span": (p2, p98)}`` of the asset's first band over the AOI, or an error."""
    import ee

    image = ee.Image(asset).select([0]).rename(_BAND)
    info = gee_interface.get_info(
        image.reduceRegion(
            reducer=ee.Reducer.percentile(list(PERCENTILES)), **_region(image, aoi)
        )
    )
    return _span(info.get(f"{_BAND}_p2"), info.get(f"{_BAND}_p98"))


def gee_classes(gee_interface, asset: str, aoi) -> dict:
    """``{"values", "known"}`` of the asset's first band over the AOI, or ``{"error"}``.

    The histogram is trimmed on the server to its size and its first
    ``MAX_CLASSES + 1`` keys (a float band can hold millions). ``known`` is the
    band's ``*_class_*`` properties, when the asset carries them.
    """
    import ee

    source = ee.Image(asset)
    name = ee.String(source.bandNames().get(0))
    image = source.select([0]).rename(_BAND)
    found = image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(), **_region(image, aoi)
    ).get(_BAND)
    hist = ee.Dictionary(ee.Algorithms.If(found, found, {}))  # null when empty
    props = ee.List(["values", "palette", "names"]).map(
        lambda part: name.cat("_class_").cat(part)
    )
    info = gee_interface.get_info(
        ee.Dictionary(
            {
                "band": name,
                "n": hist.size(),
                "hist": hist.select(hist.keys().slice(0, MAX_CLASSES + 1)),
                "meta": source.toDictionary(props),
            }
        )
    )
    return classes_from_info(info)


def classes_from_info(info: dict) -> dict:
    """Read a ``gee_classes`` reply: ``{"values", "known"}`` or ``{"error"}``."""
    hist = info.get("hist") or {}
    checked = check_values([float(k) for k in hist])
    if "error" not in checked and info.get("n", len(hist)) > MAX_CLASSES:
        checked = {"error": ERR_TOO_MANY}  # the keys beyond the trimmed ones
    if "error" in checked:
        return checked
    meta, band = info.get("meta") or {}, info.get("band")
    parts = [meta.get(f"{band}_class_{p}") for p in ("values", "palette", "names")]
    known = None
    if all(isinstance(p, list) for p in parts) and len({len(p) for p in parts}) == 1:
        known = dict(zip(("values", "palette", "names"), parts))
    return {"values": checked["values"], "known": known}


def check_values(values) -> dict:
    """The categorical rule: some values, all whole, at most ``MAX_CLASSES``."""
    values = [float(v) for v in values]
    if not values:
        return {"error": ERR_NO_VALUES}
    if not all(v.is_integer() for v in values):  # NaN and infinities are not whole
        return {"error": ERR_NOT_INTEGER}
    ordered = sorted({int(v) for v in values})
    if len(ordered) > MAX_CLASSES:
        return {"error": ERR_TOO_MANY}
    return {"values": ordered}


def file_sample(path) -> np.ndarray:
    """The valid values of the raster's first band, read nearest and decimated.

    The longest side is at most ``FILE_PROBE_SIZE``; nodata and non-finite
    values are dropped.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        k = min(1.0, FILE_PROBE_SIZE / max(src.width, src.height))
        shape = (max(1, round(src.height * k)), max(1, round(src.width * k)))
        data = src.read(1, out_shape=shape, resampling=Resampling.nearest, masked=True)
    values = data.compressed()
    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]
    return values


def file_stretch(path) -> dict:
    """``{"span": (p2, p98)}`` of a local raster's first band, or ``{"error"}``."""
    values = file_sample(path)
    if values.size == 0:
        return {"error": ERR_NO_STRETCH}
    return _span(*np.percentile(values, PERCENTILES))


def file_classes(path) -> dict:
    """``{"values", "known"}`` of a local raster's first band, or ``{"error"}``."""
    checked = check_values(np.unique(file_sample(path)))
    if "error" in checked:
        return checked
    return {"values": checked["values"], "known": None}


def vector_classes(path) -> dict:
    """``{"values", "known"}`` of a vector rasterized as "unique", or an error.

    That method burns one id per feature, numbered from 1 in file order
    (``spatialrisk.processing.xr_rasterize``). The names are the file's first
    text column, when it has one.
    """
    import geopandas as gpd
    import pandas as pd
    from pandas.api.types import is_string_dtype

    from gui.scripts import variable_palettes as palettes

    table = gpd.read_file(path, ignore_geometry=True)
    n = len(table)
    if n == 0:
        return {"error": ERR_NO_VALUES}
    if n > MAX_CLASSES:
        return {"error": ERR_TOO_MANY_FEATURES}
    values = list(range(1, n + 1))
    column = next((c for c in table.columns if is_string_dtype(table[c])), None)
    if column is None:
        return {"values": values, "known": None}
    names = [str(i) if pd.isna(v) else str(v) for i, v in zip(values, table[column])]
    colors = palettes.qualitative_colors(palettes.DEFAULT_QUALITATIVE, n)
    return {
        "values": values,
        "known": {"values": values, "palette": colors, "names": names},
    }


def _span(lo, hi) -> dict:
    if lo is None or hi is None or not float(lo) < float(hi):
        return {"error": ERR_NO_STRETCH}
    return {"span": (float(lo), float(hi))}


def parse_span(vmin, vmax) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """``(valid, span)`` for the Min and Max fields.

    Both empty is valid with no span (the renderer stretches the ramp); two
    numbers with min below max are valid with that span; anything else is not.
    """
    lo, hi = str(vmin or "").strip(), str(vmax or "").strip()
    if not lo and not hi:
        return True, None
    try:
        span = (float(lo), float(hi))
    except ValueError:
        return False, None
    return (True, span) if span[0] < span[1] else (False, None)
