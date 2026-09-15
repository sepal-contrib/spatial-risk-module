"""The Process tile tells the user how much work Run will actually do.

The status check is disk I/O, so it must live in a threaded use_task, never in
the render body (a blocking call there runs inside the websocket receive loop
and freezes the session).
"""

import threading
import time
from pathlib import Path

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t
from gui.tile import process_tile
from spatialrisk.harmonization import HarmonizationStatus
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar

SRC = Path("gui/tile/process_tile.py").read_text()

Project._ensure_model_schemas()

# Long enough that a real off-thread resolution completes well inside it, short
# enough that a regression which never resolves (e.g. a dropped `await`, or a
# use_task that never refires) fails the test instead of hanging the suite.
BLOCK_TIMEOUT = 5.0


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


# --- Behavioral coverage -----------------------------------------------------
#
# The tests above only look at source text: they catch a deleted `await threaded`
# or a `hint_key` that deletes `base_raster_key(p)`/`processing.value`, but they
# would still pass under two regressions that reintroduce the exact bugs this
# task exists to avoid:
#
#  - dropping the `await` in `return await asyncio.to_thread(harmonization_status,
#    p)`. `asyncio.to_thread(...)` is itself a coroutine function: calling it
#    without awaiting never schedules the pool thread, so `harmonization_status`
#    is never actually invoked and `.value` resolves to a bare, un-awaited
#    coroutine object instead of a HarmonizationStatus — the substring
#    `"asyncio.to_thread(harmonization_status"` survives untouched.
#  - keying `hint_key` on `project.value` itself, including via a trivial alias
#    (`pv = project.value; hint_key = (pv, ...)`) — `"project.value" not in
#    key_block` is dodged by construction, but the task then never refires: see
#    `test_harmonization_hint_refires_on_raw_variable_change_not_on_project_alias`.
#
# These mount the real ProcessTile through reacton (the precedent is
# tests/test_postprocess_tile_threading.py, which solves the same
# render-and-measure-the-thread problem for edge/dist) and observe the actually
# resolved, actually rendered outcome.


def _project_with_base(n_vars: int = 2) -> Project:
    """An in-memory project with a base raster and ``n_vars`` raw raster layers."""
    p = Project(project_name="hint-behavior")
    for i in range(n_vars):
        name = f"layer{i}"
        p.raw_variables[name] = LocalRasterVar.model_construct(
            name=name,
            data_type="raster",
            raster_type="continuous",
            path=None,
            project=p,
        )
    p.base_raster = LocalRasterVar.model_construct(
        name=next(iter(p.raw_variables)),
        data_type="raster",
        raster_type="continuous",
        path=None,
        project=p,
    )
    return p


def _render_process_tile(project):
    """Mount the real ProcessTile; caller is responsible for ``rc.close()``."""
    processing = solara.reactive(False)
    box, rc = reacton.render(
        process_tile.ProcessTile(project=project, processing=processing),
        handle_error=False,
    )
    return rc


def _hint_texts(rc):
    """Rendered ``solara.Text()`` contents (it renders as ``v.Html(tag="span")``)."""
    return [str(w.children[0]) for w in rc.find(vw.Html).widgets if w.children]


def _wait_until(predicate, timeout: float = BLOCK_TIMEOUT) -> bool:
    """Poll ``predicate`` until it is truthy or ``timeout`` elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def test_harmonization_hint_value_is_the_awaited_status_not_a_coroutine(monkeypatch):
    """A dropped `await` would leave harmonization_status uncalled, forever pending.

    ``result = asyncio.to_thread(harmonization_status, p); return result`` keeps
    the checked substring intact, but the inner coroutine is never scheduled —
    ``harmonization_status`` never runs, and ``.value`` would hold a bare
    coroutine object rather than a HarmonizationStatus. This mounts the real
    tile and checks the actually-resolved outcome instead of the source text.
    """
    calls = []

    def _stub(project):
        calls.append(threading.get_ident())
        return HarmonizationStatus(pending=["layer0"], current=["layer1"])

    monkeypatch.setattr(process_tile, "harmonization_status", _stub)

    main_ident = threading.get_ident()
    project = solara.reactive(_project_with_base(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        assert _wait_until(lambda: bool(calls)), "harmonization_status was never called"
        assert calls[0] != main_ident, (
            "harmonization_status ran on the render thread; a genuinely awaited "
            "asyncio.to_thread(...) always hands the call to a worker thread"
        )

        expected = t("tiles.process.hint_pending", pending=1, total=2)
        ok = _wait_until(lambda: expected in _hint_texts(rc))
        assert ok, f"expected {expected!r} among rendered text, got {_hint_texts(rc)}"
    finally:
        rc.close()


def test_harmonization_hint_refires_on_raw_variable_change_not_on_project_alias(
    monkeypatch,
):
    """hint_key must be scalars — keying it on the project would never retrigger.

    ``project.set(p.model_copy())`` republishes a copy taken *after* ``p`` was
    already mutated in place, so the old and new Project both hold the mutated
    dict and compare `==`. A ``hint_key`` built from (or aliasing) that object
    would see no change and never refire; this reproduces the app's own
    mutate-then-copy idiom (``on_set_base``, ``_do_remove``) and checks that the
    hint actually catches up.
    """
    seen_counts = []

    def _stub(project):
        seen_counts.append(len(project.raw_variables))
        return HarmonizationStatus(pending=[], current=list(project.raw_variables))

    monkeypatch.setattr(process_tile, "harmonization_status", _stub)

    project = solara.reactive(_project_with_base(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        assert _wait_until(lambda: bool(seen_counts)), "initial hint never computed"
        assert seen_counts[0] == 2

        p = project.value
        p.raw_variables["layer2"] = LocalRasterVar.model_construct(
            name="layer2",
            data_type="raster",
            raster_type="continuous",
            path=None,
            project=p,
        )
        project.set(p.model_copy())

        refired = _wait_until(lambda: len(seen_counts) >= 2 and seen_counts[-1] == 3)
        assert refired, (
            f"hint did not refire after raw_variables changed; calls seen="
            f"{seen_counts} — a hint_key keyed on (or aliasing) the project "
            "object would stall exactly like this"
        )
    finally:
        rc.close()
