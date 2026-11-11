"""Step 3 asks for a re-harmonize of layers saved without their categories.

Categorical layers now carry the categories read when they were harmonized
(``categorical_levels``); training refuses a layer without them. A project
harmonized by an older version has none, so the Harmonization step names those
layers and asks for them to be harmonized again -- which also counts them as
pending, so Harmonize all picks them up.
"""

import solara
from _notification_host import render_under_notifications

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

from gui.tile import process_tile  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.variables.local_raster_var import LocalRasterVar  # noqa: E402

Project._ensure_model_schemas()


def _layer(p, name, raster_type="categorical", levels=None):
    return LocalRasterVar.model_construct(
        name=name,
        data_type="raster",
        raster_type=raster_type,
        path=None,
        project=p,
        grid_signature="SIG",
        categorical_levels=levels,
    )


def _project(subj_levels):
    p = Project(project_name="categories-notice")
    for name, kind, levels in (
        ("subj", "categorical", subj_levels),
        ("altitude", "continuous", None),
    ):
        p.raw_variables[name] = _layer(p, name, kind)
        p.processed_variables[name] = _layer(p, name, kind, levels)
    p.base_raster = _layer(p, "altitude", "continuous")
    return p


def _texts(widget, out=None):
    out = [] if out is None else out
    for child in getattr(widget, "children", None) or []:
        if isinstance(child, str):
            out.append(child)
        else:
            _texts(child, out)
    return out


def _notice(box):
    marker = "Harmonize them again"
    return next((s for s in _texts(box) if marker in s), None)


def _render(p):
    return render_under_notifications(
        lambda: process_tile.ProcessTile(
            project=solara.reactive(p), processing=solara.reactive(False)
        ),
        handle_error=False,
    )


def test_layers_without_stored_categories_are_named():
    """The categorical layer missing its categories is listed; others are not."""
    box, rc = _render(_project(subj_levels=None))
    try:
        notice = _notice(box)
        assert notice is not None and "subj" in notice
        assert "altitude" not in notice
    finally:
        rc.close()


def test_no_notice_once_every_layer_has_its_categories():
    """A project harmonized by this version shows nothing."""
    box, rc = _render(_project(subj_levels=[1, 2, 3]))
    try:
        assert _notice(box) is None
    finally:
        rc.close()
