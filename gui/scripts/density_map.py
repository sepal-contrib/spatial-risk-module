"""Map display for the deforestation-density raster.

The density raster is continuous (Float64 ha/pixel/yr, nodata -9999), so the
categorical ramps in ``prediction_styles.py`` (which assume integer classes and
nodata 0) do not apply. Only the localtileserver/Colormap wiring is shared —
see the note in gui/scripts/prediction_map.py: get_leaflet_tile_layer rejects
dict colormaps, so a matplotlib Colormap object plus vmin/vmax is what pins a
value→colour range.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger("spatial_risk")

#: Ramp for "hectares lost per pixel per year": green (little or no loss)
#: through yellow to red (most loss). matplotlib's RdYlGn runs red→green, so
#: the reversed variant puts green at the low end.
DENSITY_CMAP_NAME = "RdYlGn_r"

#: Nodata of the density raster, mirrored from spatialrisk.allocation so this
#: module stays importable without pulling in the numeric core.
DENSITY_NODATA = -9999.0


def density_layer_key(run_key: str) -> str:
    """Namespaced map-layer key for one allocation run's density raster."""
    return f"density_{run_key}"


def density_colormap():
    """Matplotlib Colormap object for the density ramp."""
    import matplotlib

    return matplotlib.colormaps[DENSITY_CMAP_NAME]


def density_value_range(path) -> Tuple[float, float]:
    """(vmin, vmax) of the raster's valid pixels, nodata excluded."""
    import numpy as np
    from osgeo import gdal

    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr = band.ReadAsArray().astype("float64")
    ds = None
    valid = arr[arr != nodata] if nodata is not None else arr
    if valid.size == 0:
        return (0.0, 1.0)
    return (float(np.nanmin(valid)), float(np.nanmax(valid)))


def add_density_on_map(
    map_, path, *, key: str, layer_name: str, fit_bounds: bool = False, opacity=1.0
):
    """Add the density raster to *map_* with a continuous, pinned colour range.

    Mirrors ``prediction_map.add_prediction_on_map``: same TileClient/
    get_leaflet_tile_layer wiring and the same replace-by-key behaviour, but the
    colour range is stretched to this raster's own min/max because density is a
    continuous quantity with no fixed classification.

    Returns ``(layer, (vmin, vmax))`` — the caller needs the range to label the
    layer's legend, and re-deriving it would mean a second full band read.
    """
    from localtileserver import TileClient, get_leaflet_tile_layer

    path = str(path)
    # The density map is written in the same tiled layout as predictions, so it
    # needs the same overview pyramid to draw when zoomed out. Idempotent,
    # threshold-gated and best-effort.
    try:
        import spatialrisk.overviews as overviews

        overviews.ensure_overviews(path, min_pixels=overviews.OVERVIEW_MIN_PIXELS)
    except Exception:
        logger.exception("overview build failed for %s; adding un-optimised", path)
    vmin, vmax = density_value_range(path)
    client = TileClient(path)
    layer = get_leaflet_tile_layer(
        client,
        colormap=density_colormap(),
        vmin=vmin,
        vmax=vmax,
        nodata=DENSITY_NODATA,
        name=layer_name,
        opacity=opacity,
        max_zoom=20,
        max_native_zoom=20,
    )

    map_.remove_layer(key, none_ok=True)
    map_.add_layer(layer, key=key)

    if fit_bounds:
        map_.center = client.center()
        map_.zoom = client.default_zoom

    logger.info(
        "Density map on map: %s (%.4f to %.4f ha/px/yr)", layer_name, vmin, vmax
    )
    return layer, (vmin, vmax)
