"""Add Variable modal: interval layers pick a start and an end year."""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_modal import VariableModal, _raw_params  # noqa: E402


def _render(initial_entry, on_add=None):
    _box, rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=on_add or (lambda entry: None),
            initial_entry=initial_entry,
        ),
        handle_error=False,
    )
    return rc


def _select(rc, label_key):
    label = t(label_key)
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


def _submit(rc):
    text = t("common.add")
    [b for b in rc.find(vw.Btn).widgets if text in str(b.children)][0].click()


def _alert_text(rc):
    return " ".join(str(a.children) for a in rc.find(vw.Alert).widgets)


def _texts(rc):
    return " ".join(str(w.children) for w in rc.find(vw.Html).widgets)


def _defo_gfc(params):
    return {
        "source": "predefined",
        "predefined_key": "deforestation_gfc",
        "params": params,
    }


PERIOD = {"tree_cover_threshold": 30, "start_year": 2015, "end_year": 2020}


def test_interval_layer_shows_start_and_end_selects_not_year():
    """Two year selects seeded from the entry; no single Year select."""
    rc = _render(_defo_gfc(PERIOD))
    try:
        assert _select(rc, "vars.modal.param_start_year").v_model == 2015
        end = _select(rc, "vars.modal.param_end_year")
        assert end.v_model == 2020
        assert end.items == list(range(2001, 2027))
        assert _select(rc, "vars.modal.predefined_year_label") is None
    finally:
        rc.close()


def test_submit_builds_the_trailing_pair_name_without_a_year():
    """Entry: name with _2015_2020, year None (storage key = name), int params."""
    added = []
    rc = _render(_defo_gfc(PERIOD), added.append)
    try:
        _submit(rc)
        (entry,) = added
        assert entry["name"] == "deforestation_gfc_tc30_2015_2020"
        assert entry["year"] is None
        assert entry["params"] == PERIOD
    finally:
        rc.close()


def test_untouched_year_selects_block_submit_with_a_year_message():
    """Defaults are None: the selects start blank and submit names the field."""
    added = []
    rc = _render(_defo_gfc(None), added.append)
    try:
        assert _select(rc, "vars.modal.param_start_year").v_model is None
        _submit(rc)
        assert added == []
        assert t(
            "vars.modal.error_param_year", label=t("vars.modal.param_start_year")
        ) in _alert_text(rc)
    finally:
        rc.close()


def test_reversed_years_show_the_interval_order_error():
    """A start >= end is refused with the dedicated message."""
    added = []
    rc = _render(
        _defo_gfc({**PERIOD, "start_year": 2020, "end_year": 2015}), added.append
    )
    try:
        _submit(rc)
        assert added == []
        assert t("vars.modal.error_interval_order") in _alert_text(rc)
    finally:
        rc.close()


def test_switching_from_a_temporal_layer_drops_the_stale_year():
    """A year picked for forest_gfc must not ride along into an interval layer."""
    added = []
    rc = _render(
        {
            "source": "predefined",
            "predefined_key": "forest_gfc",
            "params": {"tree_cover_threshold": 30},
            "year": 2020,
        },
        added.append,
    )
    try:
        _select(
            rc, "vars.modal.predefined_variable_label"
        ).v_model = "deforestation_gfc"
        _select(rc, "vars.modal.param_start_year").v_model = 2015
        _select(rc, "vars.modal.param_end_year").v_model = 2020
        # The saved-as preview has no trailing stale year either.
        assert "deforestation_gfc_tc30_2015_2020" in _texts(rc)
        assert "deforestation_gfc_tc30_2015_2020_2020" not in _texts(rc)
        _submit(rc)
        (entry,) = added
        assert entry["name"] == "deforestation_gfc_tc30_2015_2020"
        assert entry["year"] is None
    finally:
        rc.close()


def test_raw_params_blanks_none_and_keeps_lists():
    """Seeding: None -> "", ints -> str, multichoice lists stay lists."""
    assert _raw_params(
        {"start_year": None, "tree_cover_threshold": 30, "classes": ["large"]}
    ) == {"start_year": "", "tree_cover_threshold": "30", "classes": ["large"]}


def test_edit_round_trip_reopens_the_saved_period():
    """The _variable_to_entry -> modal path shows the saved years."""
    from gui.tile.variables_tile import _variable_to_entry

    name = "deforestation_tmf_degraded_2010_2018"
    var = type("GEEVar", (), {"name": name, "path": None, "year": None})()
    entry = _variable_to_entry(name, var, None)
    assert entry["year"] == ""
    rc = _render(entry)
    try:
        assert _select(rc, "vars.modal.param_start_year").v_model == 2010
        assert _select(rc, "vars.modal.param_end_year").v_model == 2018
    finally:
        rc.close()
