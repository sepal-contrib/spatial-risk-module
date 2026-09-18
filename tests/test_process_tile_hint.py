"""The Process tile tells the user how much work Run will actually do.

The *disk* half of the status check must live in a threaded use_task, never in
the render body (a blocking call there runs inside the websocket receive loop
and freezes the session). Since the grid signatures landed, the render body
decides the status in memory and the task is left with the entries that predate
the stamp — ``tests/test_process_tile_pure_status.py`` owns that split, the
merge, and what reaches disk. This module keeps the surrounding tile contract.
"""

import inspect
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


def test_hint_task_is_threaded_and_keyed_on_scalars():
    """model_copy() compares equal, so the deps must be scalar, not the project."""
    idx = SRC.index("async def harmonization_hint")
    decorator = SRC[SRC.rindex("@solara.lab.use_task", 0, idx) : idx]
    assert "prefer_threaded=True" in decorator
    assert "dependencies=[hint_key]" in decorator

    key_start = SRC.index("hint_key = (")
    key_block = SRC[key_start : SRC.index("@solara.lab.use_task", key_start)]
    # What the key holds is test_process_tile_pure_status's contract; what it
    # must never hold is this one.
    assert "unknown_keys" in key_block
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


# The dropped-`await` guard moved with the call it guards: the render body now
# resolves the pure status inline, and `asyncio.to_thread` survives only around
# `harmonization_status_from_disk`. Its successor is
# test_process_tile_pure_status.test_an_unstamped_layer_takes_the_disk_verdict_off_thread,
# which also pins the keys the disk check is restricted to.


def test_harmonization_hint_refires_on_raw_variable_change_not_on_project_alias(
    monkeypatch,
):
    """A republished project must re-resolve the status, not reuse the old one.

    ``project.set(p.model_copy())`` republishes a copy taken *after* ``p`` was
    already mutated in place, so the old and new Project both hold the mutated
    dict and compare `==`. Anything that memoised the status on that object —
    a ``use_memo``, or the ``hint_key`` this used to be keyed on — would see no
    change and never refire; this reproduces the app's own mutate-then-copy
    idiom (``on_set_base``, ``_do_remove``) and checks that the tile catches up.
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
            f"the status did not catch up after raw_variables changed; calls "
            f"seen={seen_counts} — anything keyed on (or aliasing) the project "
            "object would stall exactly like this"
        )
    finally:
        rc.close()


def test_harmonization_hint_refires_on_an_in_place_variable_edit(monkeypatch):
    """An edit that keeps name+year moves neither key set — the status must still move.

    ``variables_tile.on_save`` re-registers the rebuilt variable under the same
    ``{name}_{year}`` key, so anything keyed on the raw/processed key SETS sees
    nothing change. It then never re-resolves and the tile keeps calling a layer
    the edit just invalidated harmonized — the UI telling the user not to press
    Run, which is what leaves the stale output in place.
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
            f"the status did not catch up after an in-place edit; paths seen="
            f"{seen_paths} — a dependency built from the key sets alone stalls "
            "exactly like this"
        )
    finally:
        rc.close()


def _legacy_project(n_vars: int = 2) -> Project:
    """A project saved before the grid signatures: base and outputs unstamped.

    Every layer is ``unknown`` to the pure check, so what the rows say is
    entirely the cached disk verdict — the one state in which a ``hint_key``
    that carries no per-variable path can go stale without a trace.
    """
    p = _project_with_base(n_vars)
    for key, var in list(p.raw_variables.items()):
        p.processed_variables[key] = LocalRasterVar.model_construct(
            name=var.name,
            data_type="raster",
            raster_type="continuous",
            path=Path(f"/nowhere/{key}_harmonized.tif"),
            project=p,
        )
    return p


def test_a_legacy_project_drops_harmonized_after_an_in_place_edit(monkeypatch):
    """The same edit, on a project whose base carries no signature.

    Its sibling above stubs the pure check, so it no longer covers this path.
    Here the real one runs: with no base stamp every layer used to come back
    ``unknown``, the rows were 100% the cached disk verdict, and nothing in
    ``hint_key`` moved on an edit — ``on_save`` re-registers under the same
    ``{name}_{year}`` key, so neither the raw keys nor the unknown set change.
    The disk task never refired and the row kept reading "harmonized" with
    Harmonize-all dead: the UI telling the user not to press Run over the very
    raster their edit invalidated. "Never harmonized" needs no base signature,
    so it must be decided in memory — and the key then moves with it.
    """
    disk_calls = []

    def _disk(project, keys=None):
        wanted = sorted(keys if keys is not None else project.raw_variables)
        disk_calls.append(wanted)
        return HarmonizationStatus(pending=[], current=wanted)

    monkeypatch.setattr(process_tile, "harmonization_status_from_disk", _disk)

    project = solara.reactive(_legacy_project(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    harmonized = t("widgets.product_table.status_harmonized")
    try:
        assert _wait_until(
            lambda: _row_status_labels(rc).count(harmonized) == 2
        ), f"the legacy rows never took the disk verdict: {_row_status_labels(rc)}"

        # Exactly what on_save does: pop the key, rebuild the variable from the
        # modal entry, re-register it under the SAME key — and drop the
        # processed entry, because the edit invalidated that output.
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
        p.processed_variables.pop("layer0", None)
        project.set(p.model_copy())

        moved = _wait_until(lambda: _row_status_labels(rc).count(harmonized) < 2)
        assert moved, (
            f"the edited layer still reads harmonized; disk calls={disk_calls} — "
            "an unstamped base made every verdict the cached disk one, and "
            "nothing in hint_key moves on an in-place edit"
        )
        assert (
            _harmonize_all(rc).disabled is False
        ), "Harmonize all stayed dead over a layer the edit just invalidated"

        # Waited on, not merely allowed: the disk task has to refire at all
        # (that is the regression), restricted to the entry that is still
        # unstamped — and waiting here also drains it, so no queued call of
        # ours can land in another module's stub.
        assert _wait_until(
            lambda: len(disk_calls) >= 2
        ), f"the disk task never refired after the edit; calls={disk_calls}"
        assert disk_calls[-1] == ["layer1"], (
            f"the refire must be restricted to the still-unstamped entry, got "
            f"{disk_calls[-1]}"
        )
    finally:
        rc.close()


def _harmonize_all(rc):
    """The bulk button, found by its label."""

    def leaves(w):
        for c in getattr(w, "children", None) or []:
            if isinstance(c, str):
                yield c
            else:
                yield from leaves(c)

    label = t("tiles.process.harmonize_all_button")
    hits = [b for b in rc.find(vw.Btn).widgets if label in list(leaves(b))]
    assert len(hits) == 1, f"expected one {label!r} button, got {len(hits)}"
    return hits[0]


def test_harmonize_all_goes_dead_once_every_layer_is_current(monkeypatch):
    """Nothing pending means pressing it would rewrite nothing.

    The sentence that used to say so ("Running again will do nothing") is
    gone, so the button itself has to carry it — the same contract as
    Download-all, which is dead when no layer is still cloud-backed.
    """
    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=[], current=list(p.raw_variables)),
    )
    project = solara.reactive(_project_with_base(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        assert _wait_until(
            lambda: _harmonize_all(rc).disabled is True
        ), "Harmonize all stayed live with nothing left to harmonize"
    finally:
        rc.close()


def test_harmonize_all_stays_live_while_a_layer_is_pending(monkeypatch):
    """The disable must key off real pending work, not merely a resolved status."""
    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=["layer0"], current=["layer1"]),
    )
    project = solara.reactive(_project_with_base(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        # Wait for the status to land, then assert it did NOT disable.
        assert _wait_until(
            lambda: t("widgets.product_table.status_pending") in _row_status_labels(rc)
        ), "status never resolved"
        assert _harmonize_all(rc).disabled is False
    finally:
        rc.close()
