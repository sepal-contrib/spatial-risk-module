"""Draw model predictions on the map with their QGIS-faithful colour ramps.

Kept separate from ``map_helpers`` so this feature's commits stay confined to
new files. Pins ``vmin/vmax`` to the qml classification range via
``localtileserver.get_leaflet_tile_layer`` (which, unlike ``SepalMap.add_raster``,
accepts ``colormap/vmin/vmax/nodata``), and optionally builds overviews first.
"""

import logging

logger = logging.getLogger("spatial_risk")


def add_prediction_on_map(
    map_,
    path,
    *,
    model_key,
    layer_name,
    key,
    fit_bounds=True,
    opacity=1.0,
    display_palette=None,
):
    """Draw a prediction raster with its QGIS-faithful, value-pinned palette.

    Unlike ``SepalMap.add_raster`` (which auto-stretches the palette to the
    raster's actual min/max), this pins ``vmin/vmax`` to the qml classification
    range via ``localtileserver.get_leaflet_tile_layer`` so colours land on the
    same data values as in QGIS. Replaces any layer already registered under
    ``key``.

    The external ``.ovr`` pyramid is built first, always: predictions are
    written as 256 px tiles, and a tiled raster with no pyramid makes every
    zoomed-out tile decode the whole file, which is what stops a country-scale
    prediction from ever drawing. The build is idempotent and threshold-gated,
    so it is a no-op for a raster that already has one or is small enough to
    decimate quickly, and best-effort: a failure is logged and display proceeds.

    ``map_`` is the first positional arg (not a keyword) so callers read
    naturally; everything after ``path`` is keyword-only.
    """
    from localtileserver import TileClient, get_leaflet_tile_layer

    from gui.scripts.prediction_styles import resolve_display_style

    path = str(path)

    try:
        import spatialrisk.overviews as overviews

        overviews.ensure_overviews(path, min_pixels=overviews.OVERVIEW_MIN_PIXELS)
    except Exception:
        logger.exception("overview build failed for %s; adding un-optimised", path)

    style = resolve_display_style(model_key, display_palette)
    client = TileClient(path)
    layer = get_leaflet_tile_layer(
        client,
        colormap=style["colormap"],
        vmin=style["vmin"],
        vmax=style["vmax"],
        nodata=style["nodata"],
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

    return layer
