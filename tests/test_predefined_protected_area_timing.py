"""protected_area's timing: "Current" (one map) or "By year" (STATUS_YR <= year).

Sites with an unknown status year (STATUS_YR == 0, ~6.7 % of the WDPA,
verified 2026-10-09) count in every year, which ``lte`` does on its own; this
file pins that decision.
"""

import itertools

import ipyvuetify as vw
import reacton
import solara
from _catalogue_checks import assert_roundtrip, multichoice_subsets
from _ee_expr import fake_aoi, patch_ee

import gui.scripts.predefined.infrastructure as infra
from gui.i18n import t
from gui.scripts.predefined import (
    PREDEFINED_CATALOGUE,
    build_predefined_name,
    param_specs,
    resolve_predefined,
    resolved_entry,
)
from gui.scripts.predefined.aoi import _aoi_bounds

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_modal import VariableModal  # noqa: E402

YEARLY = {"timing": "yearly"}


def _golden_by_year(ee, aoi, year):
    """Pre-B2 protected_area chain plus the status-year filter."""
    wdpa = (
        ee.FeatureCollection("WCMC/WDPA/current/polygons")
        .filterBounds(_aoi_bounds(aoi))
        .filter(
            ee.Filter.inList(
                "STATUS", ["Designated", "Inscribed", "Established", "Proposed"]
            )
        )
        .filter(ee.Filter.lte("STATUS_YR", year))
    )
    return ee.Image(0).paint(wdpa, 1).clip(aoi).toByte()


def test_by_year_is_the_current_layer_plus_a_year_filter(monkeypatch):
    """Same sites as Current, then STATUS_YR <= year."""
    fake = patch_ee(monkeypatch, infra)
    aoi = fake_aoi(fake)
    built = infra._get_protected_area(aoi, 2015, timing="yearly")
    assert repr(built) == repr(_golden_by_year(fake, aoi, 2015))


def test_current_ignores_a_year(monkeypatch):
    """A year reaching the Current builder adds no status-year filter."""
    fake = patch_ee(monkeypatch, infra)
    assert "STATUS_YR" not in repr(infra._get_protected_area(fake_aoi(fake), 2015))


def test_unknown_status_year_counts_in_every_year(monkeypatch):
    """STATUS_YR == 0 passes ``lte`` and is never excluded."""
    fake = patch_ee(monkeypatch, infra)
    text = repr(infra._get_protected_area(fake_aoi(fake), 2000, timing="yearly"))
    assert ".filter(ee.Filter.lte('STATUS_YR', 2000))" in text
    assert text.count("STATUS_YR") == 1  # no gt / neq on 0 anywhere


def test_attribute_filters_apply_before_the_year(monkeypatch):
    """The year filter is the last one, after every advanced filter."""
    fake = patch_ee(monkeypatch, infra)
    text = repr(
        infra._get_protected_area(
            fake_aoi(fake), 2010, timing="yearly", iucn=["strict"], realm="land"
        )
    )
    assert text.index("'IUCN_CAT'") < text.index("'REALM'") < text.index("STATUS_YR")


def test_only_by_year_asks_for_a_year():
    """Current is static; By year offers 2000-2026 (WDPA STATUS_YR reaches 2026)."""
    assert "protected_area_by_year" not in PREDEFINED_CATALOGUE
    assert resolved_entry("protected_area", {})["temporal"] is False
    entry = resolved_entry("protected_area", YEARLY)
    assert entry["temporal"] is True
    assert entry["years"] == list(range(2000, 2027))


def test_names():
    """Current keeps the bare key; By year adds one segment before the filters."""
    assert build_predefined_name("protected_area", {}) == "protected_area"
    assert build_predefined_name("protected_area", YEARLY) == "protected_area_yearly"
    assert (
        build_predefined_name("protected_area", {**YEARLY, "status": ["designated"]})
        == "protected_area_yearly_sd"
    )
    key, values = resolve_predefined("protected_area_yearly")
    assert key == "protected_area" and values["timing"] == "yearly"
    assert resolve_predefined("protected_area_by_year") == (None, {})


def test_by_year_names_roundtrip_exhaustively():
    """19 530 By-year names resolve back with timing kept."""
    specs = {s["key"]: s for s in param_specs("protected_area")}
    names = set()
    for status, sites, iucn, designation, realm in itertools.product(
        multichoice_subsets(specs["status"]),
        multichoice_subsets(specs["site_types"]),
        multichoice_subsets(specs["iucn"]),
        multichoice_subsets(specs["designation"]),
        specs["realm"]["options"],
    ):
        values = {
            "timing": "yearly",
            "status": status,
            "site_types": sites,
            "iucn": iucn,
            "designation": designation,
            "realm": realm,
        }
        names.add(assert_roundtrip("protected_area", values))
    assert len(names) == 31 * 3 * 15 * 7 * 2
    assert all(n.startswith("protected_area_yearly") for n in names)


def _select(rc, label_key):
    label = t(label_key)
    matches = [s for s in rc.find(vw.Select).widgets if s.label == label]
    return matches[0] if matches else None


def _submit(rc):
    text = t("common.add")
    [b for b in rc.find(vw.Btn).widgets if text in str(b.children)][0].click()


def test_modal_year_follows_the_period():
    """Year select only for By year; switching back to Current drops the year."""
    added = []
    _box, rc = reacton.render(
        VariableModal(open_=solara.reactive(True), on_add=added.append),
        handle_error=False,
    )
    try:
        _select(rc, "vars.modal.predefined_variable_label").v_model = "protected_area"
        year = "vars.modal.predefined_year_label"
        assert _select(rc, year) is None
        _select(rc, "vars.modal.param_pa_timing").v_model = "yearly"
        _select(rc, year).v_model = 2015
        _select(rc, "vars.modal.param_pa_timing").v_model = "current"
        assert _select(rc, year) is None
        _submit(rc)
        assert len(added) == 1
        assert added[0]["name"] == "protected_area"
        assert not added[0]["year"]
    finally:
        rc.close()


def test_modal_by_year_saves_the_year():
    """By year + 2015 saves protected_area_yearly for 2015."""
    added = []
    _box, rc = reacton.render(
        VariableModal(open_=solara.reactive(True), on_add=added.append),
        handle_error=False,
    )
    try:
        _select(rc, "vars.modal.predefined_variable_label").v_model = "protected_area"
        _select(rc, "vars.modal.param_pa_timing").v_model = "yearly"
        _submit(rc)
        assert added == []  # no year yet
        _select(rc, "vars.modal.predefined_year_label").v_model = 2015
        _submit(rc)
        assert [(e["name"], str(e["year"])) for e in added] == [
            ("protected_area_yearly", "2015")
        ]
    finally:
        rc.close()
