"""Draw sample sets on the map as PMTiles vector tiles.

Scales to millions of points: the browser fetches only the tiles for the current
viewport/zoom. Solara-free (mirrors prediction_map.py). On SEPAL the tiles go
through jupyter-server-proxy (see gui/scripts/tile_proxy.py). Without a proxy
prefix (local voila) they fall back to the jupyter-loopback comm bridge, which
forwards Range requests and relays 206 responses as pmtiles.js needs.
"""
import logging

logger = logging.getLogger("spatial_risk")

# Must match the tippecanoe -l layer name in spatialrisk/pmtiles_convert.py.
_SOURCE_LAYER = "points"


def build_sample_circle_style(
    url, *, strata_field="strata", event_color="#d62728", forest_color="#2ca02c"
):
    """Style dict: one filtered circle layer per class.

    event (strata == 1) -> red, forest (everything else) -> green.

    ipyleaflet's PMTilesLayer renders through protomaps-leaflet's json_style,
    which understands only a subset of the MapLibre spec: paint values are
    copied verbatim into canvas styles (an expression like ["match", ...]
    silently renders black) and circle-opacity is ignored. Legacy layer
    filters (==, !=, in, all, ...) ARE supported, so data-driven coloring
    must be expressed as one layer per class with scalar paint values.
    """

    def circle(id_, filter_, color):
        return {
            "id": id_,
            "type": "circle",
            "source": "sample",
            "source-layer": _SOURCE_LAYER,
            "filter": filter_,
            "paint": {
                "circle-radius": 4,
                "circle-stroke-width": 1,
                "circle-color": color,
                "circle-stroke-color": color,
            },
        }

    return {
        "version": 8,
        "sources": {"sample": {"type": "vector", "url": url}},
        "layers": [
            circle("sample-event", ["==", strata_field, 1], event_color),
            circle("sample-forest", ["!=", strata_field, 1], forest_color),
        ],
    }


def _enable_loopback(client):
    """Route the tile server through the jupyter-loopback comm bridge.

    vectortileserver 0.2.2+ owns the bridge wiring (correct port + proxy-prefix
    probe; the server host is 127.0.0.1 now, which a hand-rolled
    ``intercept_localhost`` would miss). Best-effort: on non-SEPAL frontends
    (or if the bridge is missing) this is a harmless no-op and the layer still
    works where the origin is shared. A no-op on SEPAL, where
    ``VECTORTILESERVER_DISABLE_JUPYTER_LOOPBACK`` is set at startup.
    """
    try:
        client.enable_jupyter_loopback()
    except Exception:
        logger.warning(
            "jupyter-loopback bridge unavailable; PMTiles tiles "
            "may not reach the browser on SEPAL",
            exc_info=True,
        )


def add_sample_pmtiles_on_map(map_, pmtiles_path, name, key):
    """Add a sample's PMTiles as one circle layer styled by strata.

    Replaces any layer already registered under ``key``. We construct the
    ipyleaflet ``PMTilesLayer`` directly and use ``TileClient`` purely for its
    range-capable tile server + ``pmtiles_url`` — the package's own layer
    factory pulls in ``mapbox_vector_tile``/``geopandas`` extras this path
    doesn't need. (``vectortileserver`` is the PyPI rename of the git-era
    ``pyvectortiles``; the API surface used here is unchanged.)
    """
    from ipyleaflet import PMTilesLayer
    from vectortileserver.client import TileClient

    client = TileClient(str(pmtiles_path))
    _enable_loopback(client)
    style = build_sample_circle_style(client.pmtiles_url)
    layer = PMTilesLayer(url=client.pmtiles_url, style=style)
    try:
        layer.name = name
    except Exception:
        pass
    map_.remove_layer(key, none_ok=True)
    map_.add_layer(layer, key=key)
    return layer


def remove_sample_pmtiles_from_map(map_, key):
    """Remove the PMTiles layer registered under ``key``."""
    map_.remove_layer(key, none_ok=True)
