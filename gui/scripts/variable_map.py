"""Draw a downloaded variable raster on the map with its catalogue palette.

Mirrors ``prediction_map`` (and is kept separate from ``map_helpers`` so this
fix stays confined to new files). ``SepalMap.add_raster`` auto-stretches a named
colormap and can't pin a custom palette to a value range — and actually raises on
a ``Colormap`` object (pysepal binds ``cmap`` only for ``str`` colormaps). So this
renders through ``localtileserver.get_leaflet_tile_layer`` directly, the same way
predictions do, passing the matplotlib ``Colormap`` object + ``vmin``/``vmax`` from
``variable_styles.resolve_variable_style``.
"""

import logging

logger = logging.getLogger("spatial_risk")


def add_raster_var_on_map(
    map_,
    path,
    *,
    var,
    layer_name,
    key,
    fit_bounds=False,
    opacity=1.0,
):
    """Draw a local variable raster with the palette it had as a GEE layer.

    Blocking (builds the overview pyramid on first use, a ``TileClient``, and
    reads the file) — call it from a worker thread, like the GEE / vector
    branches. Replaces any layer already registered under ``key`` so
    re-toggling doesn't stack duplicates.
    """
    from localtileserver import TileClient, get_leaflet_tile_layer

    from gui.scripts.variable_styles import resolve_variable_style

    path = str(path)
    # Same pyramid predictions get (spatialrisk.overviews): without it a
    # zoomed-out tile of a large raster decodes the whole file. Resampled with
    # nearest, not average — variables are often categorical, and averaging
    # would invent classes (3 and 5 -> 4) or fade a 0/1 mask away.
    try:
        import spatialrisk.overviews as overviews

        overviews.ensure_overviews(
            path, resampling="nearest", min_pixels=overviews.OVERVIEW_MIN_PIXELS
        )
    except Exception:
        logger.exception("overview build failed for %s; adding un-optimised", path)
    style = resolve_variable_style(var)
    # Post-process styles (postprocess_styles.resolve_postprocess_style) carry their
    # own "nodata" key and are the authority when present: distance rasters declare a
    # nodata tag on disk that does not match their actual fill value (see
    # postprocess_styles.POSTPROCESS_PALETTES["distance"]), so sniffing the file would
    # reproduce that bug. Catalogue / raster_type styles carry no "nodata" key, so
    # they fall back to sniffing the file exactly as before.
    nodata = style.get("nodata", _file_nodata(path))

    client = TileClient(path)
    layer = get_leaflet_tile_layer(
        client,
        colormap=style["colormap"],
        vmin=style["vmin"],
        vmax=style["vmax"],
        nodata=nodata,
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


def _file_nodata(path):
    """Best-effort read of a raster's nodata tag (so masked fill isn't painted).

    Downloaded layers carry a raster_type-dependent nodata (255 for categorical,
    -32768 for continuous — see ``GEEVar._resolve_export_nodata``); honouring the
    file's tag keeps the unmasked background from clamping to a palette colour.
    Returns ``None`` when it can't be read (localtileserver then falls back
    to the file's own tag / no masking).
    """
    try:
        import rasterio

        with rasterio.open(path) as src:
            return src.nodata
    except Exception:
        logger.debug("could not read nodata from %s", path, exc_info=True)
        return None
