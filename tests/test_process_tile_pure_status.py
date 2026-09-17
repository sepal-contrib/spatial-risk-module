"""The display status is decided in memory; only unstamped layers reach disk.

Step 3's status used to cost ``1 + N`` file opens on every reference change,
all of them inside a threaded ``use_task`` because doing them in the render
body would block the session's websocket receive loop. The grid signatures
stamped on each output answer the same question from attributes alone, so the
render body now decides the status and the task is left with the entries that
predate the stamp — none, in a project harmonized by this version.
"""

import inspect
import threading
import time

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t
from gui.tile import process_tile
from spatialrisk.harmonization import HarmonizationStatus
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar

SRC = inspect.getsource(process_tile.ProcessTile)

Project._ensure_model_schemas()

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


def _render_body() -> str:
    """Everything from the component's `with solara.Column` onward."""
    return SRC[SRC.index("    with solara.Column(") :]


def _key_block() -> str:
    """The `hint_key` tuple, up to the task that depends on it."""
    start = SRC.index("hint_key = (")
    return SRC[start : SRC.index("@solara.lab.use_task", start)]


def test_render_body_uses_the_pure_status():
    """The in-memory check is what the rows are drawn from."""
    assert "harmonization_status(p)" in SRC


def test_render_body_never_calls_the_disk_check():
    """A file open in the render body runs inside the websocket receive loop."""
    assert "harmonization_status_from_disk" not in _render_body()


def test_the_disk_check_is_confined_to_the_threaded_task():
    """It stays off the render thread, restricted to the unstamped keys."""
    block = _hint_block()
    assert "harmonization_status_from_disk" in block
    assert "asyncio.to_thread" in block
    assert "unknown_keys" in block


def test_the_task_is_keyed_on_the_unstamped_set():
    """A stamped project yields an empty tuple, so the task does no I/O."""
    key_block = _key_block()
    assert "unknown_keys," in key_block
    assert "processing.value," in key_block


def test_the_key_discriminates_the_project_and_the_base():
    """`unknown_keys` alone aliases across projects and across base grids.

    Two projects, or one project against two references, can yield an identical
    tuple of unstamped keys — and ``last_status`` would then serve one state's
    disk verdict against the other's rows. The project name and the base
    signature are in the key to keep that tagging honest, not to retrigger.
    """
    key_block = _key_block()
    assert "project_name" in key_block
    assert "base_sig_of(p)" in key_block
    # A use_task keyed on the project itself would never retrigger.
    assert "project.value" not in key_block


# --- Behavioral coverage -----------------------------------------------------
#
# The source-text tests above pin the shape; these mount the real ProcessTile
# and assert the outcome, because the shape checks would survive the two
# regressions that matter: a merge that drops the disk verdict on the floor,
# and a `disabled=` that goes back to keying off `pending` alone.


def _unstamped_project(n_vars: int = 2) -> Project:
    """A project as saved before grid signatures existed: base and outputs bare.

    Every harmonizable layer is therefore ``unknown`` to the pure check, which
    is the only state in which the disk task has anything to do.
    """
    p = Project(project_name="pure-status")
    for i in range(n_vars):
        name = f"layer{i}"
        p.raw_variables[name] = LocalRasterVar.model_construct(
            name=name,
            data_type="raster",
            raster_type="continuous",
            path=None,
            project=p,
        )
        p.processed_variables[name] = LocalRasterVar.model_construct(
            name=name,
            data_type="raster",
            raster_type="continuous",
            path=None,
            project=p,
        )
    p.base_raster = LocalRasterVar.model_construct(
        name="layer0",
        data_type="raster",
        raster_type="continuous",
        path=None,
        project=p,
    )
    return p


def _render_process_tile(project, processing=None):
    """Mount the real ProcessTile; caller is responsible for ``rc.close()``."""
    # Explicit None check: `processing or ...` asks a Reactive for its truth
    # value, which raises.
    if processing is None:
        processing = solara.reactive(False)
    box, rc = reacton.render(
        process_tile.ProcessTile(project=project, processing=processing),
        handle_error=False,
    )
    return rc


def _row_status_labels(rc):
    """Per-row Status labels — where a resolved status becomes visible at all.

    Walked by hand: ``rc.find`` does not descend into the nested ``rv.Html``
    cells ProductTable builds its rows from.
    """
    labels = {
        t(f"widgets.product_table.status_{s}")
        for s in ("harmonized", "pending", "not_downloaded", "checking", "running")
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


def test_a_stamped_project_never_reaches_the_disk(monkeypatch):
    """The whole point: with every signature recorded the task does no I/O."""
    called = []
    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=["layer0"], current=["layer1"]),
    )
    monkeypatch.setattr(
        process_tile,
        "harmonization_status_from_disk",
        lambda p, keys=None: called.append(keys),
    )
    project = solara.reactive(_unstamped_project(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        expected = t("widgets.product_table.status_pending")
        assert _wait_until(
            lambda: expected in _row_status_labels(rc)
        ), f"rows never rendered the pure verdict; they say {_row_status_labels(rc)}"
        assert called == [], f"the disk check ran for a fully stamped project: {called}"
    finally:
        rc.close()


def test_an_unstamped_layer_takes_the_disk_verdict_off_thread(monkeypatch):
    """The unknown half is resolved on a worker thread and merged into the rows.

    A merge that dropped ``disk`` would leave both rows reading "checking"
    forever; one that ran the disk check inline would block the websocket loop.
    """
    threads = []

    def _stub(project, keys=None):
        threads.append((threading.get_ident(), tuple(keys or ())))
        return HarmonizationStatus(pending=["layer0"], current=["layer1"])

    monkeypatch.setattr(process_tile, "harmonization_status_from_disk", _stub)

    main_ident = threading.get_ident()
    project = solara.reactive(_unstamped_project(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        assert _wait_until(lambda: bool(threads)), "the disk check was never reached"
        ident, keys = threads[0]
        assert ident != main_ident, (
            "the disk check ran on the render thread; a genuinely awaited "
            "asyncio.to_thread(...) always hands the call to a worker thread"
        )
        assert keys == (
            "layer0",
            "layer1",
        ), f"the disk check must be restricted to the unstamped keys, got {keys}"
        pending = t("widgets.product_table.status_pending")
        ok = _wait_until(lambda: pending in _row_status_labels(rc))
        assert ok, (
            f"the disk verdict never reached the rows; they say "
            f"{_row_status_labels(rc)}"
        )
        # Waited, like its sibling above: "pending" can show up one render
        # pass before "harmonized" does, and a bare assertion here fails on a
        # loaded machine for no reason of its own.
        assert _wait_until(
            lambda: t("widgets.product_table.status_harmonized")
            in _row_status_labels(rc)
        ), f"the current layer never reached the rows: {_row_status_labels(rc)}"
    finally:
        rc.close()


def test_an_unresolved_unknown_row_reads_as_checking(monkeypatch):
    """Never "harmonized" before the check that decides has answered.

    The failure this guards is the worst one available: a layer claiming to be
    done while the verdict is still in flight, which is also the advice not to
    press Run.
    """
    blocked = threading.Event()

    def _stub(project, keys=None):
        blocked.wait(BLOCK_TIMEOUT)
        return HarmonizationStatus(pending=[], current=["layer0", "layer1"])

    monkeypatch.setattr(process_tile, "harmonization_status_from_disk", _stub)

    project = solara.reactive(_unstamped_project(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        checking = t("widgets.product_table.status_checking")
        assert _wait_until(
            lambda: checking in _row_status_labels(rc)
        ), f"an unstamped row did not read as checking: {_row_status_labels(rc)}"
        assert t("widgets.product_table.status_harmonized") not in _row_status_labels(
            rc
        ), "an unstamped row claimed to be harmonized before the disk check answered"
    finally:
        blocked.set()
        rc.close()


def test_harmonize_all_stays_live_for_an_unstamped_project(monkeypatch):
    """A legacy project is exactly the one that most needs a run.

    ``unknown`` sits in neither ``pending`` nor ``current``, so keying the
    disable off an empty ``pending`` alone leaves a dead button here.
    """
    blocked = threading.Event()

    def _stub(project, keys=None):
        blocked.wait(BLOCK_TIMEOUT)
        return HarmonizationStatus(pending=["layer0"], current=["layer1"])

    monkeypatch.setattr(process_tile, "harmonization_status_from_disk", _stub)

    project = solara.reactive(_unstamped_project(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project)
    try:
        checking = t("widgets.product_table.status_checking")
        assert _wait_until(lambda: checking in _row_status_labels(rc)), "never rendered"
        assert (
            _harmonize_all(rc).disabled is False
        ), "Harmonize all went dead on a project whose layers were never checked"
    finally:
        blocked.set()
        rc.close()


def test_an_unstamped_row_spins_during_a_bulk_run(monkeypatch):
    """A bulk run re-checks the unstamped layers too, so they must not read idle.

    ``run_processing`` computes its own disk status over the whole project, so
    an unknown row is genuinely under consideration; leaving it out of
    ``running_keys`` makes it read as untouched while the run works on it.
    """
    monkeypatch.setattr(
        process_tile,
        "harmonization_status_from_disk",
        lambda p, keys=None: None,
    )
    project = solara.reactive(_unstamped_project(2), equals=lambda a, b: a is b)
    rc = _render_process_tile(project, processing=solara.reactive(True))
    try:
        running = t("widgets.product_table.status_running")
        assert _wait_until(
            lambda: running in _row_status_labels(rc)
        ), f"an unstamped row sat idle during a bulk run: {_row_status_labels(rc)}"
    finally:
        rc.close()
