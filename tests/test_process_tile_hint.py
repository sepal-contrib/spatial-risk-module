"""The Process tile tells the user how much work Run will actually do.

The status check is disk I/O, so it must live in a threaded use_task, never in
the render body (a blocking call there runs inside the websocket receive loop
and freezes the session).
"""

import inspect
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

# inspect, not a CWD-relative read(): the file path only resolves when pytest is
# invoked from the repo root, so a `pytest tests/test_process_tile_hint.py` run
# from anywhere else was a collection error. Matches the two sibling
# source-inspecting modules (test_process_tile_wiring, test_process_tile_run_guard).
SRC = inspect.getsource(process_tile.ProcessTile)

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


def test_status_keys_resolve_in_english():
    """Only the still-computing line survives; the rows carry the rest.

    The hint sentences and then the summary chips both restated what the
    list's own Status column already says, per variable and in place.
    """
    assert t("tiles.process.checking_status") != "tiles.process.checking_status"
    assert t("tiles.process.harmonize_all_button") == "Harmonize all"
    # Orphaned by the reference strip and the chip removal — missing-key
    # behavior is the assertion.
    for gone in (
        "hint_pending",
        "hint_all_current",
        "error_no_base",
        "chip_harmonized",
        "chip_pending",
        "chip_not_downloaded",
        "run_processing_button",
    ):
        assert t(f"tiles.process.{gone}") == f"tiles.process.{gone}"


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


def test_the_status_summary_is_the_rows_themselves():
    """No summary chips: every row already states its own status, in place.

    The counts they carried were derivable by reading the column beside them,
    so they cost a row of the panel to say nothing new.
    """
    from gui.tile import process_tile as mod

    assert not hasattr(mod, "_StatusChip")
    assert "chip_harmonized" not in SRC
    # The still-resolving line survives: no row can state a status yet.
    assert "harmonization_hint.pending" in SRC


def test_section_order_is_strip_then_list_then_harmonize_all():
    """Reference strip, the per-variable list, then Harmonize all underneath.

    The bulk button acts on the rows above it — the shape of Step 2's source
    list with Download-all beneath. The reference form is not in this order at
    all: it lives in a dialog, so the tile body is the title, the strip, the
    list and one button.
    """
    assert "DerivedVariableList" not in SRC
    strip = SRC.index("ReferenceStrip(")
    listing = SRC.index("HarmonizationVariableList(")
    run = SRC.index("tiles.process.harmonize_all_button")
    assert strip < listing < run
    # The form moved into CreationDialog, which renders after the body.
    assert SRC.index("BaseProjectionForm(") > run


def test_per_row_harmonize_runs_only_that_key():
    """The row button narrows the run to one raw key via ``keys=``."""
    assert "pending_harmonize" in SRC
    body = SRC[SRC.index("async def process_task") :]
    body = body[: body.index("\n    def ")]
    assert "process_actions.run_processing(p, keys=" in body


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


def _row_status_labels(rc):
    """Per-row Status labels — the resolved status's only rendered trace.

    With the summary chips gone this is where a resolved status becomes
    visible at all, which makes it the honest thing to assert on. Walked by
    hand rather than via ``_hint_texts``: ``rc.find`` does not descend into
    the nested ``rv.Html`` cells ProductTable builds its rows from (same
    reason as ``_texts`` in test_harmonization_variable_list).
    """
    labels = {
        t(f"widgets.product_table.status_{s}")
        for s in ("harmonized", "pending", "not_downloaded", "checking")
    }
    found = []

    def walk(w):
        for c in getattr(w, "children", None) or []:
            if isinstance(c, str):
                if c in labels:
                    found.append(c)
            else:
                walk(c)

    for root in rc.find(vw.Html).widgets:
        walk(root)
    return found


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

        # layer0 pending, layer1 current — the rows must say so once resolved.
        expected = t("widgets.product_table.status_pending")
        ok = _wait_until(lambda: expected in _row_status_labels(rc))
        assert ok, f"no row reads {expected!r}; rows say {_row_status_labels(rc)}"
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


def test_harmonization_hint_refires_on_an_in_place_variable_edit(monkeypatch):
    """An edit that keeps name+year moves neither key set — the hint must still refire.

    ``variables_tile.on_save`` re-registers the rebuilt variable under the same
    ``{name}_{year}`` key, so a ``hint_key`` built from the raw/processed key
    SETS sees nothing change. The task then never refires and the tile keeps
    rendering "All N layer(s) are already harmonized" about a layer the edit
    just invalidated — the UI telling the user not to press Run, which is what
    leaves the stale output in place.
    """
    seen_paths = []

    def _stub(project):
        seen_paths.append(
            sorted(
                str(getattr(v, "path", None)) for v in project.raw_variables.values()
            )
        )
        return HarmonizationStatus(pending=[], current=list(project.raw_variables))

    monkeypatch.setattr(process_tile, "harmonization_status", _stub)

    project = solara.reactive(_project_with_base(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        assert _wait_until(lambda: bool(seen_paths)), "initial hint never computed"

        # Exactly what on_save does: pop the key, rebuild the variable from the
        # modal entry, re-register it under the SAME key. Only `path` differs.
        p = project.value
        old = p.raw_variables.pop("layer0")
        p.raw_variables["layer0"] = LocalRasterVar.model_construct(
            name=old.name,
            year=old.year,
            data_type="raster",
            raster_type="continuous",
            path=Path("/nowhere/my_dem.tif"),
            project=p,
        )
        project.set(p.model_copy())

        refired = _wait_until(
            lambda: any(
                any("my_dem.tif" in path for path in paths) for paths in seen_paths
            )
        )
        assert refired, (
            f"hint did not refire after an in-place edit; paths seen={seen_paths} "
            "— a hint_key built from the key sets alone stalls exactly like this"
        )
    finally:
        rc.close()
