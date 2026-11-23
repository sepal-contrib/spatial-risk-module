"""Add/Edit Variable modal for ``roads``: GRIP4 road types follow the source.

The types select is shown only when the source is GRIP4. A value left in the
hidden field (narrowed or cleared, then source switched back to OSM) must not
block submit or leak into the name — Review Focus 1 of the B1 plan.
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_modal import VariableModal  # noqa: E402

ALL_TYPES = ["highway", "primary", "secondary", "tertiary", "local"]


def _render(params, on_add=None, editing_key=None):
    box, rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=on_add or (lambda entry: None),
            editing_key=editing_key,
            initial_entry={
                "source": "predefined",
                "predefined_key": "roads",
                "params": params,
            },
        ),
        handle_error=False,
    )
    return rc


def _select(rc, label_key):
    label = t(label_key)
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


def _source(rc):
    return _select(rc, "vars.modal.param_roads_source")


def _types(rc):
    return _select(rc, "vars.modal.param_road_types")


def _submit(rc):
    text = t("common.add")
    [b for b in rc.find(vw.Btn).widgets if text in str(b.children)][0].click()


def test_saved_grip_layer_reopens_with_types_shown():
    """Editing roads_grip_rhp: Advanced open, GRIP4 selected, list seeded."""
    rc = _render(
        {"source": "grip", "types": ["highway", "primary"]},
        editing_key="roads_grip_rhp",
    )
    try:
        assert _source(rc) is not None and _source(rc).v_model == "grip"
        road_types = _types(rc)
        assert road_types is not None
        assert road_types.v_model == ["highway", "primary"]
        assert road_types.multiple is True
        assert [item["value"] for item in road_types.items] == ALL_TYPES
        assert [item["text"] for item in road_types.items] == [
            t(f"vars.modal.road_type_{value}") for value in ALL_TYPES
        ]
    finally:
        rc.close()


def test_legacy_roads_opens_collapsed():
    """A plain ``roads`` layer shows no advanced fields until the user asks."""
    rc = _render(None)
    try:
        assert _source(rc) is None
        assert _types(rc) is None
    finally:
        rc.close()


def test_switching_to_osm_hides_types():
    """Types only apply to GRIP4; the source select itself stays visible."""
    rc = _render({"source": "grip", "types": ["highway", "primary"]})
    try:
        _source(rc).v_model = "osm"
        assert _source(rc) is not None
        assert _types(rc) is None
    finally:
        rc.close()


def test_switching_back_to_grip_shows_types_again():
    """Flipping the source back re-reveals a non-empty types selection."""
    rc = _render({"source": "grip", "types": ["highway", "primary"]})
    try:
        _source(rc).v_model = "osm"
        _source(rc).v_model = "grip"
        road_types = _types(rc)
        assert road_types is not None
        assert road_types.v_model and set(road_types.v_model) <= set(ALL_TYPES)
    finally:
        rc.close()


def test_osm_submit_ignores_narrowed_hidden_types():
    """GRIP4 → highways only → back to OSM submits the plain ``roads`` layer."""
    added = []
    rc = _render({"source": "grip", "types": ["highway"]}, on_add=added.append)
    try:
        _source(rc).v_model = "osm"
        _submit(rc)
        assert len(added) == 1
        assert added[0]["name"] == "roads"
        assert added[0]["params"] == {"source": "osm", "types": ALL_TYPES}
    finally:
        rc.close()


def test_osm_submit_ignores_cleared_hidden_types():
    """An emptied (invalid) hidden selection must not block the OSM submit."""
    added = []
    rc = _render({"source": "grip", "types": ["highway"]}, on_add=added.append)
    try:
        _types(rc).v_model = []
        _source(rc).v_model = "osm"
        _submit(rc)
        assert [entry["name"] for entry in added] == ["roads"]
    finally:
        rc.close()


def test_grip_with_no_types_is_blocked():
    """Zero road types with GRIP4 is an error naming the field."""
    added = []
    rc = _render({"source": "grip", "types": ["highway"]}, on_add=added.append)
    try:
        _types(rc).v_model = []
        _submit(rc)
        assert added == []
        alerts = rc.find(vw.Alert).widgets
        assert alerts and t("vars.modal.param_road_types") in str(alerts[0].children)
    finally:
        rc.close()


def test_grip_subset_submits_its_suffix():
    """Highways + primary → roads_grip_rhp with the list carried in params."""
    added = []
    rc = _render(
        {"source": "grip", "types": ["primary", "highway"]}, on_add=added.append
    )
    try:
        _submit(rc)
        assert [entry["name"] for entry in added] == ["roads_grip_rhp"]
        assert added[0]["params"] == {
            "source": "grip",
            "types": ["highway", "primary"],
        }
        assert added[0]["predefined_key"] == "roads"
    finally:
        rc.close()


def test_all_five_types_submit_plain_grip():
    """An explicit full selection is the default GRIP4 variable."""
    added = []
    rc = _render({"source": "grip", "types": ["highway"]}, on_add=added.append)
    try:
        _types(rc).v_model = list(reversed(ALL_TYPES))
        _submit(rc)
        assert [entry["name"] for entry in added] == ["roads_grip"]
    finally:
        rc.close()
