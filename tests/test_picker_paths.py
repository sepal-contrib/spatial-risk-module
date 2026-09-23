"""SEPAL file-picker values get anchored to $HOME (mitigation for PR-2)."""

import inspect
from pathlib import Path

import reacton
from pysepal.sepalwidgets.file_input import FileInput

from gui.scripts.picker_paths import resolve_picked_path
from gui.widget.borders_picker import BordersPicker


def test_relative_with_client_is_home_anchored():
    """A relative value behind a sepal_client is anchored under $HOME."""
    assert (
        resolve_picked_path("downloads/x.tif", sepal_client=object())
        == Path.home() / "downloads/x.tif"
    )


def test_absolute_untouched():
    """An absolute value passes through untouched, client or not."""
    assert resolve_picked_path("/tmp/x.tif", sepal_client=object()) == Path(
        "/tmp/x.tif"
    )


def test_no_client_is_plain_path():
    """Without a sepal_client, a relative value stays relative (local picker)."""
    assert resolve_picked_path("rel/x.tif", sepal_client=None) == Path("rel/x.tif")


def test_empty_value_is_none():
    """An empty or missing value resolves to None, not Path('.')."""
    assert resolve_picked_path("", sepal_client=object()) is None
    assert resolve_picked_path(None, sepal_client=object()) is None


# --- render test ---------------------------------------------------------
# The unit tests above pin the resolver itself; this mounts a real picker
# site so a wiring mistake (resolver imported but not called, or called with
# the args swapped) fails here instead of in the app.


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def test_borders_picker_anchors_a_picked_path_under_home():
    """The FILE method's callback resolves the raw v_model through the picker.

    A plain ``object()`` stands in for a sepal_client: load_files logs and
    swallows the AttributeError it raises, so no real SEPAL API is needed.
    """
    seen = []
    box, _rc = reacton.render(
        BordersPicker(value=None, on_value=seen.append, sepal_client=object())
    )

    file_input = _find(box, FileInput)[0]
    file_input.v_model = "downloads/b.gpkg"

    assert seen[-1].file_path == str(Path.home() / "downloads/b.gpkg")


def test_every_picker_site_uses_the_resolver():
    """The four FileInputComponent sites all route through resolve_picked_path."""
    import gui.widget.allocation_form as allocation_form
    import gui.widget.borders_picker as borders_picker
    import gui.widget.prediction_form_dialog as prediction_form_dialog
    import gui.widget.variable_modal as variable_modal

    sites = (variable_modal, prediction_form_dialog, borders_picker, allocation_form)
    for module in sites:
        assert "resolve_picked_path" in inspect.getsource(module)
