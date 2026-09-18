"""Removing a variable can now take its files with it — driven through the tile.

The checkbox is the whole feature, so these mount the real VariablesTile, tick
the real checkbox in the real dialog and look at the filesystem afterwards. A
source-substring check would pass just as happily with a checkbox that never
reaches the browser or a handler that ignores it.
"""

import ipyvuetify as vw
import pytest
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.tile import variables_tile  # noqa: E402
from spatialrisk import project as project_module  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.variables.local_raster_var import LocalRasterVar  # noqa: E402
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


def _button(box, label):
    hits = [b for b in _find(box, vw.Btn) if b.children == [label]]
    assert hits, f"no button labelled {label!r}"
    return hits[0]


def _raster(project, name, path, year=None):
    return LocalRasterVar.model_construct(
        name=name,
        year=year,
        path=path,
        project=project,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        active=True,
    )


def _project(tmp_path, monkeypatch):
    """A project with one downloaded raster (plus its overview) on disk."""
    monkeypatch.setattr(project_module, "downloads_folder", tmp_path)
    raw = tmp_path / "proj" / "data_raw"
    raw.mkdir(parents=True)
    main, ovr = raw / "slope_2020.tif", raw / "slope_2020.tif.ovr"
    main.write_bytes(b"\0" * 2048)
    ovr.write_bytes(b"\0" * 128)

    p = Project(project_name="proj")
    p.raw_variables["slope_2020"] = _raster(p, "slope", main, year=2020)
    return p, main, ovr


def _mount(monkeypatch, project_reactive):
    """Mount the tile with its list and modal stubbed; return (box, captured)."""
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


def test_ticking_the_box_deletes_the_raster_and_its_overview(monkeypatch, tmp_path):
    """The tick reaches the filesystem, sidecar included."""
    p, main, ovr = _project(tmp_path, monkeypatch)
    project = solara.reactive(p)
    box, captured = _mount(monkeypatch, project)

    captured["on_remove"]("slope_2020")  # the row's trash icon
    _find(box, vw.Checkbox)[0].v_model = True
    _button(box, t("common.remove")).fire_event("click", {})

    assert not main.exists()
    assert not ovr.exists()
    assert "slope_2020" not in project.value.raw_variables


def test_leaving_the_box_unticked_keeps_the_files(monkeypatch, tmp_path):
    """The default is the behaviour the app has always had."""
    p, main, ovr = _project(tmp_path, monkeypatch)
    project = solara.reactive(p)
    box, captured = _mount(monkeypatch, project)

    captured["on_remove"]("slope_2020")
    _button(box, t("common.remove")).fire_event("click", {})

    assert main.exists() and ovr.exists()
    assert "slope_2020" not in project.value.raw_variables


def test_the_box_is_unticked_again_for_the_next_variable(monkeypatch, tmp_path):
    """A tick is a one-off decision, never a mode — the next remove starts safe."""
    p, main, _ovr = _project(tmp_path, monkeypatch)
    other = main.parent / "altitude.tif"
    other.write_bytes(b"\0" * 16)
    p.raw_variables["altitude"] = _raster(p, "altitude", other)
    project = solara.reactive(p)
    box, captured = _mount(monkeypatch, project)

    captured["on_remove"]("slope_2020")
    _find(box, vw.Checkbox)[0].v_model = True
    _button(box, t("common.cancel")).fire_event("click", {})

    captured["on_remove"]("altitude")

    assert _find(box, vw.Checkbox)[0].v_model is False


def test_a_file_outside_the_project_is_explained_not_offered(monkeypatch, tmp_path):
    """No checkbox for a file the app did not write."""
    p, _main, _ovr = _project(tmp_path, monkeypatch)
    outside = tmp_path / "downloads" / "my_dem.tif"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"\0" * 8)
    p.raw_variables["my_dem"] = _raster(p, "my_dem", outside)
    project = solara.reactive(p)
    box, captured = _mount(monkeypatch, project)

    captured["on_remove"]("my_dem")

    assert _find(box, vw.Checkbox) == []
    _button(box, t("common.remove")).fire_event("click", {})
    assert outside.exists()


def test_a_cloud_variable_shows_the_dialog_unchanged(monkeypatch, tmp_path):
    """Nothing on disk: no checkbox, no note, exactly the dialog of before."""
    from spatialrisk.variables.gee_var import GEEVar

    p, _main, _ovr = _project(tmp_path, monkeypatch)
    p.raw_variables["forest"] = GEEVar.model_construct(
        name="forest",
        project=p,
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
        gee_images=["img"],
    )
    project = solara.reactive(p)
    box, captured = _mount(monkeypatch, project)

    captured["on_remove"]("forest")

    assert _find(box, vw.Checkbox) == []


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
