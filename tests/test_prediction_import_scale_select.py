"""Import mode of the Predict dialog: value scale is an explicit choice.

Every import is warped onto the project grid and put on the 1..65535 scale, so
the dialog no longer offers a display palette. Instead it asks which scale the
file is on (0..1 probabilities or 1..65535 risk), refuses to submit without an
answer, and tells the user what the import expects.
"""

import types

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.prediction_form_dialog import PredictionFormDialog  # noqa: E402


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _select(box, label):
    hits = [s for s in _find(box, vw.Select) if s.label == label]
    assert hits, f"no select labelled {label!r}"
    return hits[0]


def _texts(box):
    """Every user-visible string in the render.

    solara.Markdown renders into a VuetifyTemplate whose text lives in its
    ``template`` trait (it has no ``children`` at all), so the spec block is
    only visible to a helper that reads that trait.
    """
    parts = [
        str(c)
        for w in _find(box, vw.Html)
        for c in (w.children or [])
        if isinstance(c, str)
    ]
    parts += [getattr(w, "v_model", "") or "" for w in _find(box, vw.Alert)]
    parts += [str(w.template or "") for w in _find(box, vw.VuetifyTemplate)]
    return " ".join(parts)


def _project(tmp_path):
    return types.SimpleNamespace(
        models={},
        datasets={},
        processed_variables={},
        predictions={},
        filter_predictions=lambda **kw: [],
        base_raster=object(),
        folders=types.SimpleNamespace(project_folder=str(tmp_path)),
    )


def _render(tmp_path, prefill=None):
    submitted = []

    @solara.component
    def Host():
        project = solara.use_reactive(_project(tmp_path))
        PredictionFormDialog(
            project=project,
            open_=solara.use_reactive(True),
            on_submit=submitted.append,
            sepal_client=None,
            prefill=solara.use_reactive(prefill),
        )

    box, rc = reacton.render(Host(), handle_error=False)
    _select(box, t("tiles.inference.source_label")).v_model = "import"
    return box, rc, submitted


def test_import_mode_shows_value_scale_and_no_palette(tmp_path):
    """Import asks for the file's value scale and offers no display palette."""
    box, _, _ = _render(tmp_path)
    scale = _select(box, t("widgets.prediction_import_modal.label_scale"))
    assert scale.v_model in (None, "")
    assert {i["value"] for i in scale.items} == {"probability", "risk"}
    # Positive assertions, because t() returns the raw key for a key this task
    # deleted: "no select labelled t('...label_palette')" would hold whether or
    # not the palette select survived. Name every select instead, and refuse the
    # palette's option values wherever they might reappear.
    assert [s.label for s in _find(box, vw.Select)] == [
        t("tiles.inference.source_label"),
        t("widgets.prediction_import_modal.label_scale"),
    ]
    offered = {
        item.get("value")
        for s in _find(box, vw.Select)
        for item in (s.items or [])
        if isinstance(item, dict)
    }
    assert not offered & {"far", "stretch"}


def test_import_mode_shows_the_spec_block(tmp_path):
    """The dialog states the contract an imported raster must meet."""
    box, _, _ = _render(tmp_path)
    text = _texts(box)
    assert t("widgets.prediction_import_modal.spec_format") in text
    assert t("widgets.prediction_import_modal.spec_values") in text
    assert t("widgets.prediction_import_modal.spec_nodata") in text
    assert t("widgets.prediction_import_modal.spec_warp") in text


def test_import_prefill_restores_value_scale(tmp_path):
    """Re-editing a failed import brings its declared scale back."""
    entry = {
        "kind": "import",
        "name": "ext",
        "path": "/data/ext.tif",
        "value_scale": "risk",
    }
    box, _, _ = _render(tmp_path, prefill=entry)
    assert (
        _select(box, t("widgets.prediction_import_modal.label_scale")).v_model == "risk"
    )
