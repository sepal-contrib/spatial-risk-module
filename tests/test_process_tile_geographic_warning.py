"""A geographic reference CRS warns inline and still lets the user proceed."""

import ipyvuetify as v
import reacton
import solara

from gui.tile.process_tile import BaseProjectionForm


def _alerts(widget, found=None):
    """Collect every v.Alert in a rendered widget tree."""
    found = [] if found is None else found
    if isinstance(widget, v.Alert):
        found.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, v.Alert):
            _alerts(child, found)
    return found


def _render(epsg):
    """Render the form with a fixed EPSG and return its root widget."""
    project = solara.reactive(None)
    box, _rc = reacton.render(
        BaseProjectionForm(
            project=project,
            base_key="",
            set_base_key=lambda _v: None,
            epsg=epsg,
            set_epsg=lambda _v: None,
            resolution="30",
            set_resolution=lambda _v: None,
            on_auto_utm=lambda: None,
            autofill_pending=False,
        )
    )
    return box


def test_geographic_crs_renders_a_warning_alert():
    """EPSG:4326 means the pixel size is degrees — the form must say so."""
    alerts = _alerts(_render("EPSG:4326"))
    assert any(a.type == "warning" for a in alerts)


def test_projected_crs_renders_no_alert():
    """UTM is the expected case and must stay quiet."""
    assert _alerts(_render("EPSG:32618")) == []


def test_incomplete_input_renders_no_alert():
    """A half-typed code is not a warning — it is just not finished."""
    assert _alerts(_render("326")) == []
