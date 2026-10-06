"""Tests for borrowing SEPAL's jupyter-server-proxy prefix for vector tiles.

On SEPAL, ``LOCALTILESERVER_CLIENT_PREFIX`` holds a generic ``/proxy/{port}``
jupyter-server-proxy template that forwards any sandbox port — so it can carry
the PMTiles server too. ``vectortileserver`` never autodetects a prefix; it only
honors ``VECTORTILESERVER_CLIENT_PREFIX``. The helper copies the raster prefix
across (pattern proven by sepal-contrib/sbae-design).
"""

from gui.scripts.tile_proxy import (
    borrow_localtileserver_prefix,
    loopback_bridge_needed,
    prefer_http_proxy,
)

GENERIC = "https://sepal.io/api/sandbox/jupyter/proxy/{port}"


def test_borrows_generic_proxy_prefix():
    """Generic /proxy/{port} raster prefix is copied to the vector env var."""
    env = {"LOCALTILESERVER_CLIENT_PREFIX": GENERIC}
    borrow_localtileserver_prefix(env)
    assert env["VECTORTILESERVER_CLIENT_PREFIX"] == GENERIC


def test_does_not_overwrite_explicit_vector_prefix():
    """An explicit vector prefix wins over the borrowed raster one."""
    env = {
        "LOCALTILESERVER_CLIENT_PREFIX": GENERIC,
        "VECTORTILESERVER_CLIENT_PREFIX": "/custom/{port}",
    }
    borrow_localtileserver_prefix(env)
    assert env["VECTORTILESERVER_CLIENT_PREFIX"] == "/custom/{port}"


def test_ignores_missing_raster_prefix():
    """No raster prefix means loopback stays the default."""
    env = {}
    borrow_localtileserver_prefix(env)
    assert "VECTORTILESERVER_CLIENT_PREFIX" not in env


def test_ignores_port_specific_prefix():
    """A port-specific raster route is not borrowed."""
    # A route namespaced to one concrete server/port would not forward the
    # vector server's port — leave loopback in place.
    env = {"LOCALTILESERVER_CLIENT_PREFIX": "/proxy/8888/localtileserver"}
    borrow_localtileserver_prefix(env)
    assert "VECTORTILESERVER_CLIENT_PREFIX" not in env


def test_proxy_prefixes_turn_off_both_loopback_bridges():
    """Explicit HTTP proxy prefixes switch both tile servers off the bridge."""
    # On SEPAL the bridge's HEAD <prefix>/__probe__ gets the tile server's own
    # JSON 404, reads it as "proxy broken" and tunnels every tile through the
    # comm bridge even though the proxy route works.
    env = {"LOCALTILESERVER_CLIENT_PREFIX": GENERIC}
    borrow_localtileserver_prefix(env)
    prefer_http_proxy(env)
    assert env["LOCALTILESERVER_DISABLE_JUPYTER_LOOPBACK"] == "1"
    assert env["VECTORTILESERVER_DISABLE_JUPYTER_LOOPBACK"] == "1"
    assert not loopback_bridge_needed(env)


def test_no_prefix_keeps_the_bridge():
    """Without a configured proxy (local voila) the bridge stays on."""
    env = {}
    prefer_http_proxy(env)
    assert "LOCALTILESERVER_DISABLE_JUPYTER_LOOPBACK" not in env
    assert "VECTORTILESERVER_DISABLE_JUPYTER_LOOPBACK" not in env
    assert loopback_bridge_needed(env)


def test_empty_vector_prefix_keeps_the_bridge_for_pmtiles():
    """An empty vector prefix forces loopback, so the bridge must still mount."""
    env = {
        "LOCALTILESERVER_CLIENT_PREFIX": GENERIC,
        "VECTORTILESERVER_CLIENT_PREFIX": "",
    }
    prefer_http_proxy(env)
    assert env["LOCALTILESERVER_DISABLE_JUPYTER_LOOPBACK"] == "1"
    assert "VECTORTILESERVER_DISABLE_JUPYTER_LOOPBACK" not in env
    assert loopback_bridge_needed(env)


def test_explicit_disable_value_is_respected():
    """An operator's explicit opt-out value is never overwritten."""
    env = {
        "LOCALTILESERVER_CLIENT_PREFIX": GENERIC,
        "LOCALTILESERVER_DISABLE_JUPYTER_LOOPBACK": "0",
    }
    prefer_http_proxy(env)
    assert env["LOCALTILESERVER_DISABLE_JUPYTER_LOOPBACK"] == "0"
    assert loopback_bridge_needed(env)
