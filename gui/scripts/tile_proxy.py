"""Route the local tile servers through SEPAL's jupyter-server-proxy.

Both tile servers bind 127.0.0.1 inside the kernel, so the browser needs a
route that reaches them. SEPAL sets ``LOCALTILESERVER_CLIENT_PREFIX`` to a
generic jupyter-server-proxy template (``…/proxy/{port}``) that forwards any
port in the sandbox — so it carries PMTiles as well as raster tiles. But
``vectortileserver`` never autodetects a prefix (its default is the raw
loopback URL plus the comm bridge), so the prefix must be copied across before
any ``TileClient`` is built. Only the generic ``/proxy/{port}`` template form is
borrowed: a route namespaced to one concrete server would not forward the
vector port. Pattern proven on SEPAL by sepal-contrib/sbae-design.

Once a proxy prefix is set, the jupyter-loopback comm bridge is switched off
too. The bridge probes ``HEAD <prefix>/__probe__`` to decide between the proxy
and itself, and the tile server answers that unknown path with its own 404 —
so on SEPAL (voila, where the bridge mounts) it wrongly marked the working
proxy as broken and tunnelled every tile through 4 comm workers as ``blob:``
URLs (confirmed on test.sepal.io 2026-09-25).
"""

import os

_RASTER_PREFIX = "LOCALTILESERVER_CLIENT_PREFIX"
_VECTOR_PREFIX = "VECTORTILESERVER_CLIENT_PREFIX"
_RASTER_DISABLE = "LOCALTILESERVER_DISABLE_JUPYTER_LOOPBACK"
_VECTOR_DISABLE = "VECTORTILESERVER_DISABLE_JUPYTER_LOOPBACK"


def borrow_localtileserver_prefix(environ=os.environ):
    """Copy SEPAL's generic proxy prefix onto ``VECTORTILESERVER_CLIENT_PREFIX``.

    No-op when the raster prefix is absent or port-specific, or when a vector
    prefix is already set explicitly. Call at app startup, before any
    ``vectortileserver.TileClient`` is constructed.
    """
    raster_prefix = environ.get(_RASTER_PREFIX)
    if raster_prefix and "/proxy/{port}" in raster_prefix:
        environ.setdefault(_VECTOR_PREFIX, raster_prefix)


def prefer_http_proxy(environ=os.environ):
    """Turn the loopback bridge off for each tile server with a proxy prefix.

    A non-empty ``*_CLIENT_PREFIX`` means the operator configured an HTTP route,
    so tiles go straight through it. Explicit ``*_DISABLE_JUPYTER_LOOPBACK``
    values win. Both packages read the opt-out when a layer is built, so call
    this at app startup after :func:`borrow_localtileserver_prefix`.
    """
    for prefix_var, disable_var in (
        (_RASTER_PREFIX, _RASTER_DISABLE),
        (_VECTOR_PREFIX, _VECTOR_DISABLE),
    ):
        if environ.get(prefix_var):
            environ.setdefault(disable_var, "1")


def _is_disabled(value):
    # Same truthiness as localtileserver's / vectortileserver's _is_disabled.
    return (value or "").lower() not in ("", "0", "false", "no", "off")


def loopback_bridge_needed(environ=os.environ):
    """Whether either tile server still relies on the comm bridge."""
    return not (
        _is_disabled(environ.get(_RASTER_DISABLE))
        and _is_disabled(environ.get(_VECTOR_DISABLE))
    )
