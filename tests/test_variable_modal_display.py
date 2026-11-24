"""The Display section of a predefined layer in the Add/Edit Variable modal.

It opens on the catalogue's own style, and the entry carries ``vis_params``
only when the user's style differs from that default.
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.scripts.predefined.forest_change import LOSS_PALETTE  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402

LAYER = "vars.modal.predefined_variable_label"
TOGGLE = "vars.modal.display_toggle"


def _render(initial_entry=None, added=None, saved=None, editing_key=None):
    _box, rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=(added if added is not None else []).append,
            on_save=lambda key, entry: (saved if saved is not None else []).append(
                entry
            ),
            editing_key=editing_key,
            initial_entry=initial_entry,
        ),
        handle_error=False,
    )
    return rc


def _buttons(rc, text):
    return [b for b in rc.find(vw.Btn).widgets if text in str(b.children)]


def _select(rc, label_key):
    label = t(label_key)
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


def _swatches(rc):
    return [b for b in rc.find(vw.Btn).widgets if "sr-class-swatch" in (b.class_ or "")]


def _changed(rc):
    label = f"{t(TOGGLE)} {t('vars.modal.display_changed')}"
    return bool(_buttons(rc, label))


def _open_display(rc):
    _buttons(rc, t(TOGGLE))[0].click()


def _recolour_last_class(rc, color="#123456"):
    rc.find(vw.ColorPicker).widgets[-1].v_model = color


def _predefined(key, params=None, **extra):
    return {"source": "predefined", "predefined_key": key, "params": params, **extra}


def test_display_opens_seeded_with_the_catalogue_default():
    """deforestation_gfc shows the loss palette's two classes."""
    rc = _render()
    try:
        _select(rc, LAYER).v_model = "deforestation_gfc"
        assert _swatches(rc) == []  # collapsed
        _open_display(rc)
        colors = [s.style_ for s in _swatches(rc)]
        assert len(colors) == 2
        assert all(c in s for c, s in zip(LOSS_PALETTE, colors))
    finally:
        rc.close()


def test_continuous_default_ramp_is_the_catalogue_palette():
    """The slope palette is "Default"; the altitude one is the terrain preset."""
    rc = _render()
    try:
        _select(rc, LAYER).v_model = "slope"
        _open_display(rc)
        ramp = _select(rc, "vars.modal.ramp_label")
        assert ramp.items[0]["value"] == "default" and ramp.v_model == "default"
        _select(rc, LAYER).v_model = "altitude"
        _open_display(rc)
        ramp = _select(rc, "vars.modal.ramp_label")
        assert "default" not in [i["value"] for i in ramp.items]
        assert ramp.v_model == "terrain"
    finally:
        rc.close()


def test_untouched_display_submits_no_override():
    """Add and a legacy edit (no vis_params) both submit ``vis_params`` None."""
    for key in ("rivers", "slope", "altitude"):
        added = []
        rc = _render(added=added)
        try:
            _select(rc, LAYER).v_model = key
            _open_display(rc)  # looking is not changing
            _buttons(rc, t("common.add"))[0].click()
            assert added and added[0]["vis_params"] is None, key
        finally:
            rc.close()
    saved = []
    rc = _render(_predefined("rivers", {}), saved=saved, editing_key="rivers")
    try:
        assert _swatches(rc) == []  # no saved style: collapsed
        _buttons(rc, t("vars.modal.submit_save"))[0].click()
        assert saved and saved[0]["vis_params"] is None
    finally:
        rc.close()


def test_recoloured_class_submits_the_override():
    """A class recoloured in the picker is saved as the whole class list."""
    added = []
    rc = _render(added=added)
    try:
        _select(rc, LAYER).v_model = "rivers"
        _open_display(rc)
        _recolour_last_class(rc)
        assert _changed(rc)
        _buttons(rc, t("common.add"))[0].click()
        classes = added[0]["vis_params"]["classes"]
        assert [c["value"] for c in classes] == [0, 1]
        assert classes[1]["color"] == "123456"
    finally:
        rc.close()


def test_reset_to_default_clears_the_override():
    """A saved override opens the section; Reset brings back no override."""
    saved = []
    override = {
        "classes": [
            {"value": 0, "color": "ffffff", "label": "a", "visible": True},
            {"value": 1, "color": "123456", "label": "b", "visible": True},
        ]
    }
    rc = _render(
        _predefined("rivers", {}, vis_params=override),
        saved=saved,
        editing_key="rivers",
    )
    try:
        swatches = _swatches(rc)
        assert swatches and "123456" in swatches[1].style_  # opened on the saved style
        _buttons(rc, t("vars.modal.display_reset"))[0].click()
        assert not _changed(rc)
        _buttons(rc, t("vars.modal.submit_save"))[0].click()
        assert saved and saved[0]["vis_params"] is None
    finally:
        rc.close()


def test_raster_type_flip_drops_the_class_override():
    """active_fire: recolour under "any", switch to fire days: no classes saved."""
    saved = []
    rc = _render(
        _predefined("active_fire", {"metric": "any"}, year=2024),
        saved=saved,
        editing_key="active_fire_2024",
    )
    try:
        _open_display(rc)
        _recolour_last_class(rc)
        _select(rc, "vars.modal.param_fire_metric").v_model = "days"
        assert _select(rc, "vars.modal.ramp_label") is not None  # reseeded
        _buttons(rc, t("vars.modal.submit_save"))[0].click()
        assert saved and "classes" not in (saved[0]["vis_params"] or {})
        assert saved[0]["vis_params"] is None
    finally:
        rc.close()


def test_random_layer_shows_a_note_not_fields():
    """The subj layer's colours are random: a note, no ramp and no class list."""
    added = []
    rc = _render(added=added)
    try:
        _select(rc, LAYER).v_model = "subj"
        _open_display(rc)
        assert _select(rc, "vars.modal.ramp_label") is None
        assert _swatches(rc) == []
        note = t("vars.modal.display_random_note")
        assert any(note in str(w.children) for w in rc.find(vw.Html).widgets)
        _buttons(rc, t("common.add"))[0].click()
        assert added[0]["vis_params"] is None
    finally:
        rc.close()


def test_legacy_palette_mask_opens_with_its_classes():
    """A mask restyled as ``["ffffff", colour]`` pinned to 0..1 shows two classes.

    They keep the saved colours and the catalogue's translated labels, and
    Save goes through.
    """
    saved = []
    legacy = {"palette": ["ffffff", "123456"], "min": 0, "max": 1}
    rc = _render(
        _predefined(
            "deforestation_gfc",
            {"tree_cover_threshold": 30, "start_year": 2015, "end_year": 2020},
            vis_params=legacy,
        ),
        saved=saved,
        editing_key="deforestation_gfc_tc30_2015_2020",
    )
    try:
        colors = [s.style_ for s in _swatches(rc)]
        assert len(colors) == 2
        assert "ffffff" in colors[0] and "123456" in colors[1]
        _buttons(rc, t("vars.modal.submit_save"))[0].click()
        assert saved, "Save was blocked"
        classes = saved[0]["vis_params"]["classes"]
        assert [c["color"] for c in classes] == ["ffffff", "123456"]
        assert [c.get("label_key") for c in classes] == [
            "legend.class.stable",
            "legend.class.loss",
        ]
    finally:
        rc.close()


def test_new_class_values_reseed_a_touched_display():
    """mapbiomas_lulc mask -> groups: the edited 0/1 classes give way to groups."""
    saved = []
    rc = _render(
        _predefined("mapbiomas_lulc", {"mode": "mask"}, year=2020),
        saved=saved,
        editing_key="mapbiomas_lulc_2020",
    )
    try:
        _open_display(rc)
        _recolour_last_class(rc)
        _select(rc, "vars.modal.param_lulc_mode").v_model = "groups"
        assert len(_swatches(rc)) == 5
        assert not _changed(rc)
        _buttons(rc, t("vars.modal.submit_save"))[0].click()
        assert saved and saved[0]["vis_params"] is None
    finally:
        rc.close()
