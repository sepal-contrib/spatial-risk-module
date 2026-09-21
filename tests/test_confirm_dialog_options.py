"""ConfirmDialog grew the three slots the file prompts need.

A checkbox (the opt-in "also delete the files"), a muted note (why a file will
be kept anyway) and a second action (Keep existing / Re-download). All three are
optional, so every existing caller renders exactly as before.

Rendered rather than grepped: a widget prop that never reaches the browser is
invisible to a source check, and this app has shipped that bug before.
"""

import ipyvuetify as vw
import reacton

from gui.i18n import t

# Warm the translator before the first render (see test_manage_projects_render).
t("common.cancel")

from gui.widget.confirm_dialog import ConfirmDialog  # noqa: E402


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _render(**kwargs):
    defaults = dict(
        open=True,
        on_cancel=lambda: None,
        on_confirm=lambda: None,
        title="Remove variable?",
        message="Remove 'slope_2020'?",
    )
    box, _rc = reacton.render(ConfirmDialog(**{**defaults, **kwargs}))
    return box


def _labels(box):
    return [b.children[0] for b in _find(box, vw.Btn) if b.children]


def test_plain_dialog_has_no_checkbox_and_two_buttons():
    """Without the new props the dialog is exactly what it always was."""
    box = _render()
    assert _find(box, vw.Checkbox) == []
    assert len(_find(box, vw.Btn)) == 2


def test_checkbox_renders_with_its_label():
    """The opt-in reaches the widget, labelled and unticked."""
    box = _render(checkbox_label="Also delete the files from disk (2 files, 241 MB)")
    boxes = _find(box, vw.Checkbox)
    assert len(boxes) == 1
    assert boxes[0].label == "Also delete the files from disk (2 files, 241 MB)"
    assert boxes[0].v_model is False


def test_ticking_the_checkbox_reports_the_new_value():
    """A tick calls back, so the caller can act on it."""
    ticked = []
    box = _render(
        checkbox_label="Also delete the files from disk",
        checkbox_value=False,
        on_checkbox=ticked.append,
    )
    _find(box, vw.Checkbox)[0].v_model = True
    assert ticked == [True]


def test_checkbox_shows_the_value_it_is_given():
    """The caller owns the state; the widget only displays it."""
    box = _render(checkbox_label="Also delete", checkbox_value=True)
    assert _find(box, vw.Checkbox)[0].v_model is True


def test_note_is_rendered_and_dimmed():
    """The note explains a kept file; ``text--secondary`` is unusable here."""
    box = _render(note="The file is also used by 'forest_2020_matched'.")
    texts = [
        w
        for w in _find(box, vw.Html)
        if "also used by" in "".join(str(c) for c in (w.children or []))
    ]
    assert texts, "the note never rendered"
    assert "opacity" in (texts[0].style_ or "")


def test_details_are_listed_one_per_line():
    """Every path the choice would touch is named on screen."""
    box = _render(details=["data_raw/slope_2020.tif", "data_raw/slope_2020.tif.ovr"])
    rendered = " ".join(
        "".join(str(c) for c in (w.children or [])) for w in _find(box, vw.Html)
    )
    assert "data_raw/slope_2020.tif" in rendered
    assert "data_raw/slope_2020.tif.ovr" in rendered


def test_secondary_action_renders_between_cancel_and_confirm():
    """Three buttons, in the order the user reads them."""
    box = _render(
        secondary_label="Keep existing",
        on_secondary=lambda: None,
        confirm_label="Re-download",
    )
    assert _labels(box) == ["Cancel", "Keep existing", "Re-download"]


def test_secondary_action_fires_its_own_callback():
    """The middle button is a third outcome, not a second confirm."""
    fired = []
    box = _render(
        secondary_label="Keep existing",
        on_secondary=lambda: fired.append("secondary"),
        confirm_label="Re-download",
    )
    keep = [b for b in _find(box, vw.Btn) if b.children == ["Keep existing"]][0]
    keep.fire_event("click", {})
    assert fired == ["secondary"]


def test_a_plain_dialog_keeps_its_narrow_box():
    """Existing callers must look exactly as they did."""
    assert _find(_render(), vw.Dialog)[0].max_width == "380px"


def test_a_dialog_listing_files_is_wider():
    """File paths get the wider box."""
    box = _render(details=["data_raw/slope_2020.tif"])
    assert _find(box, vw.Dialog)[0].max_width == "440px"
