"""The Variables tile asks before a download lands on a file already on disk.

Driven through the tile's real ``on_download`` and the real dialog: what matters
is that a conflicting download does NOT start until the user has chosen, and
that the choice reaches the export as ``overwrite``.
"""

import threading

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.scripts import process_actions  # noqa: E402
from gui.tile import variables_tile  # noqa: E402
from spatialrisk import project as project_module  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.variables.gee_var import GEEVar  # noqa: E402
from spatialrisk.variables.models import DataType, RasterType  # noqa: E402

Project._ensure_model_schemas()


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _overwrite_dialog(box):
    """The overwrite dialog itself — identified by its Re-download button.

    The tile renders three ConfirmDialogs and each has a Cancel, so every lookup
    is scoped to this one rather than to the first match in the whole tree.
    """
    label = t("tiles.variables.confirm_overwrite_redownload")
    hits = [
        d
        for d in _find(box, vw.Dialog)
        if any(b.children == [label] for b in _find(d, vw.Btn))
    ]
    assert len(hits) == 1, f"expected exactly one overwrite dialog, got {len(hits)}"
    return hits[0]


def _click(box, label):
    button = [b for b in _find(_overwrite_dialog(box), vw.Btn) if b.children == [label]]
    assert button, f"no {label!r} button in the overwrite dialog"
    button[0].fire_event("click", {})


def _overwrite_dialog_open(box):
    """Is the overwrite dialog showing?

    Every ConfirmDialog is eager, so its widgets exist from the first render —
    only ``v_model`` says it is open.
    """
    return bool(_overwrite_dialog(box).v_model)


def _geevar(project, name, year=2020):
    return GEEVar.model_construct(
        name=name,
        year=year,
        project=project,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        gee_images=["img"],
    )


def _project(tmp_path, monkeypatch, on_disk=()):
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    (tmp_path / "proj" / "data_raw").mkdir(parents=True)
    p = Project(project_name="proj")
    for name in ("slope", "altitude"):
        p.raw_variables[f"{name}_2020"] = _geevar(p, name)
    for key in on_disk:
        path = p.raw_variables[key].expected_local_path
        path.write_bytes(b"stale")
    return p


def _mount(monkeypatch, project_reactive):
    captured = {}

    @solara.component
    def _StubList(**kwargs):
        captured.update(kwargs)
        solara.Text("list")

    @solara.component
    def _StubModal(**kwargs):
        solara.Text("modal")

    monkeypatch.setattr(variables_tile, "SourceVariableList", _StubList)
    monkeypatch.setattr(variables_tile, "VariableModal", _StubModal)
    box, rc = reacton.render(
        variables_tile.VariablesTile(project=project_reactive), handle_error=False
    )
    _OPEN_CONTEXTS.append(rc)
    return box, captured


def _record_downloads(monkeypatch):
    """Replace the real export with a recorder; returns (calls, finished event)."""
    calls, done = [], threading.Event()

    def _fake_materialize(
        project, keys=None, on_progress=None, overwrite=False, on_wait=None
    ):
        calls.append({"keys": keys, "overwrite": overwrite})
        done.set()
        return []

    monkeypatch.setattr(process_actions, "materialize_raw_layers", _fake_materialize)
    return calls, done


def test_a_download_with_nothing_in_the_way_starts_at_once(monkeypatch, tmp_path):
    """No file, no question — the download begins immediately."""
    p = _project(tmp_path, monkeypatch)
    calls, done = _record_downloads(monkeypatch)
    box, captured = _mount(monkeypatch, solara.reactive(p))

    captured["on_download"]("slope_2020")

    assert not _overwrite_dialog_open(box), "asked about a file that is not there"
    assert done.wait(timeout=10), "the download never ran"
    assert calls[0]["overwrite"] is False


def test_a_conflicting_download_asks_before_it_starts(monkeypatch, tmp_path):
    """A file in the way stops the download until the user chooses."""
    p = _project(tmp_path, monkeypatch, on_disk=["slope_2020"])
    calls, _done = _record_downloads(monkeypatch)
    box, captured = _mount(monkeypatch, solara.reactive(p))

    captured["on_download"]("slope_2020")

    assert _overwrite_dialog_open(box)
    assert calls == [], "the download started before the user chose"


def test_re_download_runs_the_export_with_overwrite(monkeypatch, tmp_path):
    """Re-download replaces the file on disk."""
    p = _project(tmp_path, monkeypatch, on_disk=["slope_2020"])
    calls, done = _record_downloads(monkeypatch)
    box, captured = _mount(monkeypatch, solara.reactive(p))

    captured["on_download"]("slope_2020")
    _click(box, t("tiles.variables.confirm_overwrite_redownload"))

    assert done.wait(timeout=10), "the download never ran"
    assert calls[0]["overwrite"] is True
    assert calls[0]["keys"] == ["slope_2020"]


def test_keep_existing_runs_the_export_without_overwrite(monkeypatch, tmp_path):
    """Keep existing is today's silent skip, now chosen out loud."""
    p = _project(tmp_path, monkeypatch, on_disk=["slope_2020"])
    calls, done = _record_downloads(monkeypatch)
    box, captured = _mount(monkeypatch, solara.reactive(p))

    captured["on_download"]("slope_2020")
    _click(box, t("tiles.variables.confirm_overwrite_keep"))

    assert done.wait(timeout=10), "the download never ran"
    assert calls[0]["overwrite"] is False


def test_cancelling_downloads_nothing(monkeypatch, tmp_path):
    """Cancel leaves both the file and the download alone."""
    p = _project(tmp_path, monkeypatch, on_disk=["slope_2020"])
    calls, _done = _record_downloads(monkeypatch)
    box, captured = _mount(monkeypatch, solara.reactive(p))

    captured["on_download"]("slope_2020")
    _click(box, t("common.cancel"))

    assert calls == []
    assert not _overwrite_dialog_open(box)


def test_download_all_asks_once_for_every_layer_in_the_way(monkeypatch, tmp_path):
    """One dialog for the batch, not one per layer."""
    p = _project(tmp_path, monkeypatch, on_disk=["slope_2020", "altitude_2020"])
    calls, done = _record_downloads(monkeypatch)
    box, captured = _mount(monkeypatch, solara.reactive(p))

    captured["on_download"](None)  # Download all

    assert _overwrite_dialog_open(box)  # one dialog for the whole batch
    assert calls == []

    _click(box, t("tiles.variables.confirm_overwrite_redownload"))
    assert done.wait(timeout=10)
    # Bulk sends its own key list (it skips rows already downloading),
    # so "unrestricted" means every pending layer, not None.
    assert set(calls[0]["keys"]) == {"slope_2020", "altitude_2020"}
    assert calls[0]["overwrite"] is True


_OPEN_CONTEXTS = []


@pytest.fixture(autouse=True)
def _close_mounted_tiles():
    """Close every tile this module mounts.

    A render context left open keeps its component's hook record alive. A
    later test that renders the same component behind a different set of
    stubs — ``use_notifications`` replaced by a plain lambda, which calls one
    ``use_memo`` fewer — then trips reacton's hook-count check. The error
    lands in that later test, far from the one that leaked the context, and
    moves with collection order.
    """
    yield
    while _OPEN_CONTEXTS:
        _OPEN_CONTEXTS.pop().close()
