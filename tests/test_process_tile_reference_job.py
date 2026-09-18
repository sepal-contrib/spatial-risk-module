"""Setting the reference runs off the render thread and only once at a time."""

import inspect

from gui.scripts.inflight import InflightKeys
from gui.tile import process_tile


def test_on_set_base_does_not_call_set_base_raster_inline():
    """A full GDAL warp in a click handler freezes the websocket loop."""
    src = inspect.getsource(process_tile.ProcessTile)
    handler = src[src.index("def on_set_base") :]
    handler = handler[: handler.index("\n    def ", 1)]
    assert "spawn_in_context" in handler, "the warp must run on its own thread"


def test_a_second_claim_is_refused_while_one_is_in_flight():
    """InflightKeys is the real double-click guard; `disabled=` is cosmetic."""
    keys = InflightKeys(key="test_reference_inflight")
    assert keys.claim("reference") is True
    assert keys.claim("reference") is False
    keys.release("reference")
    assert keys.claim("reference") is True
