"""The Process tile tells the user how much work Run will actually do.

The status check is disk I/O, so it must live in a threaded use_task, never in
the render body (a blocking call there runs inside the websocket receive loop
and freezes the session).
"""

from pathlib import Path

from gui.i18n import t

SRC = Path("gui/tile/process_tile.py").read_text()


def _hint_block() -> str:
    """The `harmonization_hint` task body, up to the next sibling definition."""
    rest = SRC[SRC.index("async def harmonization_hint") :]
    ends = [
        i
        for i in (
            rest.find("\n    @solara", 1),
            rest.find("\n    def ", 1),
            rest.find("\n    with solara", 1),
        )
        if i != -1
    ]
    return rest[: min(ends)] if ends else rest


def test_hint_keys_resolve_in_english():
    """The three new i18n keys resolve and interpolate in English."""
    assert "3" in t("tiles.process.hint_pending", pending=3, total=10)
    assert "10" in t("tiles.process.hint_pending", pending=3, total=10)
    assert "10" in t("tiles.process.hint_all_current", total=10)
    assert t("tiles.process.checking_status") != "tiles.process.checking_status"


def test_status_is_computed_off_the_render_thread():
    """The status check is disk I/O — it must not run on the websocket loop."""
    assert "asyncio.to_thread(harmonization_status" in _hint_block()


def test_hint_task_is_threaded_and_keyed_on_scalars():
    """model_copy() compares equal, so the deps must be scalar, not the project."""
    idx = SRC.index("async def harmonization_hint")
    decorator = SRC[SRC.rindex("@solara.lab.use_task", 0, idx) : idx]
    assert "prefer_threaded=True" in decorator
    assert "dependencies=[hint_key]" in decorator

    key_start = SRC.index("hint_key = (")
    key_block = SRC[key_start : SRC.index("@solara.lab.use_task", key_start)]
    assert "base_raster_key(p)" in key_block
    assert "processing.value" in key_block
    assert "raw_variables" in key_block
    assert "processed_variables" in key_block
    # A use_task keyed on the project itself would never retrigger.
    assert "project.value" not in key_block


def test_hint_bails_out_while_a_run_is_in_flight():
    """No point checking the grid while the run is rewriting those very files."""
    block = _hint_block()
    guard = block.split("return await")[0]
    assert "processing.value" in guard
    assert "return None" in guard


def test_hint_is_rendered_under_the_run_button():
    """The hint reads as the explanation for a Run that will do nothing."""
    assert "harmonization_hint.pending" in SRC
    assert "tiles.process.hint_all_current" in SRC
    assert "tiles.process.hint_pending" in SRC
