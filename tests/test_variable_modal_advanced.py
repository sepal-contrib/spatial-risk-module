"""The Add/Edit Variable modal's collapsed Advanced parameters section."""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_modal import VariableModal  # noqa: E402


def _render(initial_entry=None, on_add=None, on_save=None, editing_key=None):
    box, rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=on_add or (lambda entry: None),
            on_save=on_save,
            editing_key=editing_key,
            initial_entry=initial_entry,
        ),
        handle_error=False,
    )
    return box, rc


def _buttons(rc, text):
    return [b for b in rc.find(vw.Btn).widgets if text in str(b.children)]


def _sections(rc, text):
    """The collapsible panels whose header contains ``text``."""
    return [
        p
        for p in rc.find(vw.ExpansionPanels).widgets
        if text in str(p.children[0].children[0].children)
    ]


def _select(rc, label):
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


WATER = "vars.modal.param_water_classes"


def _is_open(rc):
    return _sections(rc, t("vars.modal.advanced_toggle"))[0].v_model == 0


def _rivers_entry(params):
    return {"source": "predefined", "predefined_key": "rivers", "params": params}


def test_advanced_section_is_collapsed_by_default():
    """A fresh layer shows the toggle but hides the advanced fields."""
    _box, rc = _render(_rivers_entry(None))
    try:
        assert _sections(rc, t("vars.modal.advanced_toggle"))
        assert not _is_open(rc)
    finally:
        rc.close()


def test_toggle_opens_a_multiple_select_seeded_with_defaults():
    """Opening the section shows a multiple select holding the defaults."""
    _box, rc = _render(_rivers_entry(None))
    try:
        _sections(rc, t("vars.modal.advanced_toggle"))[0].v_model = 0
        sel = _select(rc, t(WATER))
        assert sel is not None and sel.multiple is True
        assert sel.v_model == ["large", "major", "canal", "stream"]
        assert [i["value"] for i in sel.items] == [
            "large",
            "major",
            "canal",
            "stream",
            "ocean",
        ]
    finally:
        rc.close()


def test_edit_with_non_default_values_opens_and_counts_changes():
    """Editing non-default values auto-opens the section and counts them."""
    _box, rc = _render(
        _rivers_entry({"classes": ["large", "major", "canal"]}),
        editing_key="rivers_wlmc",
    )
    try:
        sel = _select(rc, t(WATER))
        assert _is_open(rc), "advanced section should auto-open"
        assert sel.v_model == ["large", "major", "canal"]  # a list, not "['large', …]"
        assert _sections(rc, t("vars.modal.advanced_changed", n=1))
        assert "rivers_wlmc" in " ".join(
            str(w.children) for w in rc.find(vw.Html).widgets
        )
    finally:
        rc.close()


def test_edit_legacy_bare_name_opens_collapsed_with_defaults():
    """A pre-change ``rivers`` variable (params {}) edits under the same key."""
    saved = []
    _box, rc = _render(
        _rivers_entry({}),
        editing_key="rivers",
        on_save=lambda key, entry: saved.append((key, entry)),
    )
    try:
        assert not _is_open(rc)
        _buttons(rc, t("vars.modal.submit_save"))[0].click()
        assert saved and saved[0][0] == "rivers"
        assert saved[0][1]["name"] == "rivers"
        assert saved[0][1]["params"] == {
            "classes": ["large", "major", "canal", "stream"]
        }
    finally:
        rc.close()


def test_edit_partial_params_seeds_missing_defaults_collapsed():
    """Saved params lacking ``classes`` take the default and stay collapsed.

    No catalogue layer has two params yet, so "partial" here is a non-empty
    dict missing the layer's only key (e.g. a stale or renamed knob).
    """
    _box, rc = _render(_rivers_entry({"stale_knob": 1}), editing_key="rivers")
    try:
        assert not _is_open(rc), "defaults only: stay collapsed"
        _sections(rc, t("vars.modal.advanced_toggle"))[0].v_model = 0
        sel = _select(rc, t(WATER))
        assert sel is not None
        assert sel.v_model == ["large", "major", "canal", "stream"]
    finally:
        rc.close()


def test_reset_restores_defaults():
    """Reset puts the catalogue defaults back."""
    _box, rc = _render(_rivers_entry({"classes": ["stream"]}), editing_key="rivers_ws")
    try:
        _buttons(rc, t("vars.modal.advanced_reset"))[0].click()
        assert _select(rc, t(WATER)).v_model == ["large", "major", "canal", "stream"]
    finally:
        rc.close()


def test_empty_selection_is_refused_with_a_message():
    """Submitting with nothing selected shows the multichoice error."""
    added = []
    _box, rc = _render(_rivers_entry({"classes": ["stream"]}), on_add=added.append)
    try:
        _select(rc, t(WATER)).v_model = []
        _buttons(rc, t("common.add"))[0].click()
        assert added == []
        alerts = rc.find(vw.Alert).widgets
        assert alerts and t(WATER) in str(alerts[0].children)
    finally:
        rc.close()


def test_selection_order_does_not_change_the_name():
    """Selections are stored and named in canonical order."""
    added = []
    _box, rc = _render(_rivers_entry({"classes": ["stream"]}), on_add=added.append)
    try:
        _select(rc, t(WATER)).v_model = ["canal", "large"]
        _buttons(rc, t("common.add"))[0].click()
        assert added[0]["name"] == "rivers_wlc"
        assert added[0]["params"] == {"classes": ["large", "canal"]}
    finally:
        rc.close()


def test_switching_layer_resets_advanced_state():
    """Switching layers collapses the section and re-seeds defaults."""
    _box, rc = _render(_rivers_entry({"classes": ["stream"]}))
    try:
        layer = _select(rc, t("vars.modal.predefined_variable_label"))
        layer.v_model = "forest_tmf"
        assert _select(rc, t(WATER)) is None
        assert not _is_open(rc)
        layer.v_model = "rivers"
        assert not _is_open(rc)  # collapsed again, defaults re-seeded
        _sections(rc, t("vars.modal.advanced_toggle"))[0].v_model = 0
        assert _select(rc, t(WATER)).v_model == ["large", "major", "canal", "stream"]
    finally:
        rc.close()


def test_layers_without_advanced_params_show_no_toggle():
    """Layers without advanced params render no toggle."""
    _box, rc = _render(
        {"source": "predefined", "predefined_key": "forest_gfc", "params": None}
    )
    try:
        assert not _sections(rc, t("vars.modal.advanced_toggle"))
    finally:
        rc.close()


def test_dropdown_items_are_grouped_in_declared_order():
    """Layer dropdown shows one header per theme, in GROUP_ORDER, above its layers."""
    from gui.scripts.predefined import GROUP_ORDER, PREDEFINED_CATALOGUE
    from gui.widget.variable_modal import _predefined_items

    items = _predefined_items()
    headers = [i["header"] for i in items if "header" in i]
    assert headers == [t(f"vars.predefined_group.{g}") for g in GROUP_ORDER]

    group = None
    for item in items:
        if "header" in item:
            group = GROUP_ORDER[headers.index(item["header"])]
            continue
        assert PREDEFINED_CATALOGUE[item["value"]]["group"] == group, item
    assert {i["value"] for i in items if "value" in i} == set(PREDEFINED_CATALOGUE)


def test_dropdown_keeps_same_named_layers_together():
    """Every source of one layer name sits together, in catalogue order."""
    from gui.widget.variable_modal import _predefined_items

    keys = [i["value"] for i in _predefined_items() if "value" in i]
    for run in (
        ["deforestation_gfc", "deforestation_tmf", "deforestation_mapbiomas"],
        ["burned_area", "burned_area_mapbiomas"],
        ["fire_history", "fire_history_mapbiomas"],
        ["surface_water", "mapbiomas_water"],
    ):
        start = keys.index(run[0])
        assert keys[start : start + len(run)] == run
