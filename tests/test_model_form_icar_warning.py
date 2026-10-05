"""The New model dialog warns before a long iCAR run and names the culprit.

Rendered for real. The dataset's categorical layer carries the 120 categories
stored when it was harmonized, so the dialog never reads a raster: scanning
one on a big AOI took a minute, and the warning only showed after a reopen.
The formula prefill is still a short background task, so the tests poll.
"""

import time
import types

import ipyvuetify as vw
import pytest
import rasterio
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.widget.model_form_dialog import ModelFormDialog  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.sample import Sample  # noqa: E402

Project._ensure_model_schemas()


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _texts(widget, out=None):
    out = [] if out is None else out
    for child in getattr(widget, "children", []) or []:
        if isinstance(child, str):
            out.append(child)
        else:
            _texts(child, out)
    return out


def _warning_shown(box):
    return any("faster without it" in s for s in _texts(box))


def _wait(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


@pytest.fixture(autouse=True)
def _no_raster_reads(monkeypatch):
    """The stored categories are the whole input: opening a raster is a bug."""

    def _boom(*args, **kwargs):
        raise AssertionError("the dialog must not read a raster")

    monkeypatch.setattr(rasterio, "open", _boom)


def _project(tmp_path):
    p = Project(project_name="p")
    p.datasets["calib"] = types.SimpleNamespace(
        name="calib",
        year=None,
        target=types.SimpleNamespace(name="defor"),
        features=[
            types.SimpleNamespace(name="altitude", raster_type="continuous"),
            types.SimpleNamespace(
                name="subj",
                raster_type="categorical",
                path=str(tmp_path / "subj.tif"),
                categorical_levels=list(range(120)),
            ),
        ],
    )
    p.samples["s1"] = Sample(
        name="s1",
        raster_var_name="forest",
        strategy="random",
        n_samples=20_000,
        class_counts={"0": 10_000, "1": 10_000},
    )
    return p


def _render(tmp_path, model_key):
    box, rc = reacton.render(
        ModelFormDialog(
            project=solara.reactive(_project(tmp_path)),
            open_=solara.reactive(True),
            on_submit=lambda entry: None,
        ),
        handle_error=False,
    )
    model_select = _find(box, vw.Select)[0]
    model_select.v_model = model_key
    return box, rc


def _formula_area(box):
    return _find(box, vw.Textarea)[0]


def test_icar_with_a_many_level_categorical_warns(tmp_path):
    """An iCAR model with a 120-level layer on 20k samples shows the warning."""
    box, rc = _render(tmp_path, "icar")
    try:
        assert _wait(lambda: _warning_shown(box))
        text = next(s for s in _texts(box) if "faster without it" in s)
        assert "subj (120 categories" in text
    finally:
        rc.close()


def test_glm_does_not_warn(tmp_path):
    """GLM trains fast whatever the levels: no warning."""
    box, rc = _render(tmp_path, "glm")
    try:
        # The formula prefill finishing is the cue that the form has settled.
        assert _wait(lambda: "C(subj)" in (_formula_area(box).v_model or ""))
        time.sleep(0.3)
        assert not _warning_shown(box)
    finally:
        rc.close()


def test_deleting_the_categorical_from_the_formula_clears_the_warning(tmp_path):
    """Editing the formula re-judges the run live."""
    box, rc = _render(tmp_path, "icar")
    try:
        assert _wait(lambda: _warning_shown(box))
        area = _formula_area(box)
        area.v_model = area.v_model.replace(" + C(subj)", "")

        assert _wait(lambda: not _warning_shown(box))
    finally:
        rc.close()


def test_reopening_after_the_dataset_gained_a_categorical_recounts(tmp_path):
    """The dialog stays mounted between openings; a reopen must re-judge.

    The formula refills on every reopen, so the level counts must too --
    otherwise a layer added to the dataset meanwhile shows up as C(subj) in the
    formula but is counted as one column, and the warning never comes.
    """
    p = _project(tmp_path)
    full = p.datasets["calib"]
    p.datasets["calib"] = types.SimpleNamespace(
        name="calib", year=None, target=full.target, features=full.features[:1]
    )
    project = solara.reactive(p)
    open_ = solara.reactive(True)
    box, rc = reacton.render(
        ModelFormDialog(project=project, open_=open_, on_submit=lambda entry: None),
        handle_error=False,
    )
    try:
        _find(box, vw.Select)[0].v_model = "icar"
        assert _wait(lambda: "scale(altitude)" in (_formula_area(box).v_model or ""))
        assert not _warning_shown(box)

        next(b for b in _find(box, vw.Btn) if t("common.cancel") in _texts(b)).click()
        p.datasets["calib"] = full
        project.set(p.model_copy())
        open_.set(True)

        assert _wait(lambda: _warning_shown(box))
    finally:
        rc.close()
