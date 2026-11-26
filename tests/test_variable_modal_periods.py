"""Add/Edit Variable modal with period layers.

Year lists that follow the source, inline availability errors, conditional
params, legacy entries.
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_modal import VariableModal  # noqa: E402
from spatialrisk.variables.models import RasterType  # noqa: E402

YEAR = "vars.modal.predefined_year_label"
PERIOD = "vars.modal.param_period"
METRIC_DRY = "vars.modal.param_dry_metric"
METRIC_FIRE = "vars.modal.param_fire_metric"
THRESHOLD = "vars.modal.param_rain_threshold"


def _entry(key, params, year=None):
    return {
        "source": "predefined",
        "predefined_key": key,
        "params": params,
        "year": year,
    }


def _render(initial_entry, on_add=None, on_save=None, editing_key=None):
    return reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=on_add or (lambda entry: None),
            on_save=on_save,
            editing_key=editing_key,
            initial_entry=initial_entry,
        ),
        handle_error=False,
    )


def _select(rc, label_key):
    label = t(label_key)
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


def _text_field(rc, label_key):
    label = t(label_key)
    matches = [f for f in rc.find(vw.TextField).widgets if f.label == label]
    return matches[0] if matches else None


def _click(rc, label_key):
    text = t(label_key)
    [b for b in rc.find(vw.Btn).widgets if text in str(b.children)][0].click()


def _sections(rc, text):
    """The collapsible panels whose header contains ``text``."""
    return [
        p
        for p in rc.find(vw.ExpansionPanels).widgets
        if text in str(p.children[0].children[0].children)
    ]


def _availability_alerts(rc):
    return [
        a
        for a in rc.find(vw.Alert).widgets
        if "sr-availability-error" in (a.class_ or "")
    ]


def test_years_follow_the_period():
    """Years follow the period."""
    _box, rc = _render(_entry("burned_area", None))
    try:
        # Newest first (#35).
        assert list(_select(rc, YEAR).items) == list(range(2025, 2000, -1))
        _select(rc, PERIOD).v_model = "jul"
        assert list(_select(rc, YEAR).items)[0] == 2026
    finally:
        rc.close()


def test_unavailable_year_shows_an_inline_error_and_blocks_submit():
    """Unavailable year shows an inline error and blocks submit."""
    added = []
    _box, rc = _render(
        _entry("burned_area", {"period": "jul"}, year=2026), on_add=added.append
    )
    try:
        assert not _availability_alerts(rc)
        _select(rc, PERIOD).v_model = "aug"  # MCD64A1 ends 2026-07
        alerts = _availability_alerts(rc)
        assert alerts
        expected = t(
            "vars.modal.error_period_unavailable",
            year=2026,
            start="2026-08",
            end="2026-08",
            first="2000-11",
            last="2026-07",
        )
        assert expected in str(alerts[0].children)
        _click(rc, "common.add")
        assert added == []
    finally:
        rc.close()


def test_no_availability_error_without_a_year():
    """No availability error without a year."""
    _box, rc = _render(_entry("burned_area", {"period": "aug"}))
    try:
        assert not _availability_alerts(rc)
    finally:
        rc.close()


def test_dry_spell_threshold_hides_for_raintotal():
    """Dry spell threshold hides for raintotal."""
    added = []
    _box, rc = _render(_entry("dry_spell", None, year=2024), on_add=added.append)
    try:
        field = _text_field(rc, THRESHOLD)
        assert field is not None
        field.v_model = ""  # stale and invalid: must not block once hidden
        _select(rc, METRIC_DRY).v_model = "raintotal"
        assert _text_field(rc, THRESHOLD) is None
        # The hidden field must not shift the Advanced toggle off its handler.
        _sections(rc, t("vars.modal.advanced_toggle"))[0].v_model = 0
        assert _select(rc, "vars.modal.param_rain_source") is not None
        _click(rc, "common.add")
        assert added and added[0]["name"] == "dry_spell_raintotal"
        assert added[0]["params"]["threshold"] == 1
    finally:
        rc.close()


def test_frp_with_firms_shows_an_inline_error():
    """Frp with firms shows an inline error."""
    added = []
    _box, rc = _render(
        _entry("active_fire", {"metric": "frp", "source": "firms"}, year=2024),
        on_add=added.append,
    )
    try:
        alerts = _availability_alerts(rc)
        assert alerts and t("vars.modal.error_frp_firms") in str(alerts[0].children)
        _click(rc, "common.add")
        assert added == []
    finally:
        rc.close()


def test_active_fire_days_is_submitted_as_continuous():
    """Active fire days is submitted as continuous."""
    added = []
    _box, rc = _render(_entry("active_fire", None, year=2024), on_add=added.append)
    try:
        _select(rc, METRIC_FIRE).v_model = "days"
        _click(rc, "common.add")
        assert added[0]["name"] == "active_fire_days"
        assert added[0]["raster_type"] == RasterType.continuous
    finally:
        rc.close()


def test_legacy_temperature_entry_resubmits_under_its_name():
    """Saved before ``period`` existed: params carry only the metric."""
    saved = []
    _box, rc = _render(
        _entry("temperature_2m", {"aggregation": "max"}, year=2020),
        editing_key="temperature_2m_max_2020",
        on_save=lambda key, entry: saved.append((key, entry)),
    )
    try:
        _click(rc, "vars.modal.submit_save")
        assert saved and saved[0][1]["name"] == "temperature_2m_max"
        assert saved[0][1]["params"] == {"aggregation": "max", "period": "year"}
    finally:
        rc.close()
