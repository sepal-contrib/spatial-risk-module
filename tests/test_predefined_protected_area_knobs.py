"""protected_area: WDPA filters as advanced knobs.

Defaults reproduce the pre-B2 layer verbatim. A filter is added to the EE
expression only when its value differs from "everything", so the default build
stays identical and future WDPA codes are not dropped silently.
"""

import itertools

import pytest
from _catalogue_checks import assert_roundtrip, multichoice_subsets
from _ee_expr import fake_aoi, patch_ee

import gui.scripts.predefined.infrastructure as infra
from gui.scripts.predefined import (
    build_predefined_name,
    default_param_values,
    param_specs,
    resolve_predefined,
)
from gui.scripts.predefined.aoi import _aoi_bounds

DEFAULTS = {
    "timing": "current",
    "status": ["designated", "inscribed", "established", "proposed"],
    "site_types": ["pa"],
    "iucn": ["strict", "managed", "sustainable", "unassigned"],
    "designation": ["national", "international", "regional"],
    "realm": "all",
}


def _golden_protected_area(ee, aoi):
    """Pre-B2 ``_get_protected_area`` body, verbatim."""
    wdpa = (
        ee.FeatureCollection("WCMC/WDPA/current/polygons")
        .filterBounds(_aoi_bounds(aoi))
        .filter(
            ee.Filter.inList(
                "STATUS", ["Designated", "Inscribed", "Established", "Proposed"]
            )
        )
    )
    return ee.Image(0).paint(wdpa, 1).clip(aoi).toByte()


def _text(monkeypatch, **params):
    fake = patch_ee(monkeypatch, infra)
    image = infra._get_protected_area(fake_aoi(fake), None, **{**DEFAULTS, **params})
    return repr(image)


def test_defaults_are_declared_in_order():
    """Timing (a main field), then five advanced knobs; all omitted at defaults."""
    assert default_param_values("protected_area") == DEFAULTS
    assert [s["key"] for s in param_specs("protected_area")] == list(DEFAULTS)
    for spec in param_specs("protected_area"):
        assert spec["advanced"] is (spec["key"] != "timing")
        assert spec["omit_default"] is True


def test_default_expression_is_unchanged(monkeypatch):
    """No kwargs and explicit defaults both equal today's expression."""
    fake = patch_ee(monkeypatch, infra)
    aoi = fake_aoi(fake)
    golden = repr(_golden_protected_area(fake, aoi))
    assert repr(infra._get_protected_area(aoi)) == golden
    assert repr(infra._get_protected_area(aoi, None, **DEFAULTS)) == golden


def test_full_selections_add_no_filter(monkeypatch):
    """All IUCN groups / designation types / realms -> no filter at all."""
    text = _text(monkeypatch)
    for field in ("IUCN_CAT", "DESIG_TYPE", "REALM", "STATUS_YR"):
        assert field not in text


def test_status_filter_follows_the_selection(monkeypatch):
    """Adopted is offered (34 WDPA sites) but off by default."""
    text = _text(monkeypatch, status=["designated", "adopted"])
    assert ".filter(ee.Filter.inList('STATUS', ['Designated', 'Adopted']))" in text


def test_oecms_come_from_their_own_collection(monkeypatch):
    """WDPA polygons are all SITE_TYPE PA; OECMs live in WCMC/WDOECM."""
    both = _text(monkeypatch, site_types=["pa", "oecm"])
    assert both.startswith(
        "ee.Image(0).paint(ee.FeatureCollection('WCMC/WDPA/current/polygons')"
        ".merge(ee.FeatureCollection('WCMC/WDOECM/current/polygons'))"
        ".filterBounds("
    )
    only = _text(monkeypatch, site_types=["oecm"])
    assert only.startswith(
        "ee.Image(0).paint(ee.FeatureCollection('WCMC/WDOECM/current/polygons')"
        ".filterBounds("
    )
    assert "WCMC/WDPA/" not in only
    assert "SITE_TYPE" not in both


def test_iucn_groups_expand_to_wdpa_categories(monkeypatch):
    """Groups map to the verified IUCN_CAT strings, in group order."""
    text = _text(monkeypatch, iucn=["strict", "unassigned"])
    assert (
        ".filter(ee.Filter.inList('IUCN_CAT', ['Ia', 'Ib', 'II', "
        "'Not Reported', 'Not Assigned', 'Not Applicable']))"
    ) in text


def test_iucn_groups_cover_every_verified_category():
    """Selecting every group equals no filter only if nothing is left out."""
    verified = {
        "Ia",
        "Ib",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "Not Reported",
        "Not Assigned",
        "Not Applicable",
    }
    flat = [c for group in infra.PA_IUCN_GROUPS.values() for c in group]
    assert sorted(flat) == sorted(verified)
    assert len(flat) == len(set(flat))


def test_designation_filter(monkeypatch):
    """Designation types map to the DESIG_TYPE strings."""
    text = _text(monkeypatch, designation=["national", "regional"])
    assert ".filter(ee.Filter.inList('DESIG_TYPE', ['National', 'Regional']))" in text


def test_exclude_marine_drops_only_marine_realm(monkeypatch):
    """There is no MARINE field; coastal sites stay."""
    text = _text(monkeypatch, realm="land")
    assert text.endswith(
        ".filter(ee.Filter.neq('REALM', 'Marine')), 1)"
        ".clip(ee.FeatureCollection('AOI')).toByte()"
    )


def test_filters_follow_the_bounds_prefilter(monkeypatch):
    """The cheap bounds filter runs first, then the attribute filters."""
    text = _text(monkeypatch, iucn=["strict"], designation=["national"], realm="land")
    order = [
        text.index(".filterBounds("),
        text.index("'STATUS'"),
        text.index("'IUCN_CAT'"),
        text.index("'DESIG_TYPE'"),
        text.index("'REALM'"),
    ]
    assert order == sorted(order)


def test_empty_site_types_is_rejected(monkeypatch):
    """Coercion requires one site type; a direct call fails loudly too."""
    fake = patch_ee(monkeypatch, infra)
    with pytest.raises(ValueError):
        infra._get_protected_area(fake_aoi(fake), None, site_types=[])


def test_readable_names():
    """Each knob writes one segment; defaults keep the bare key."""
    cases = {
        "protected_area": DEFAULTS,
        "protected_area_sd": {**DEFAULTS, "status": ["designated"]},
        "protected_area_tpo": {**DEFAULTS, "site_types": ["pa", "oecm"]},
        "protected_area_ism": {**DEFAULTS, "iucn": ["strict", "managed"]},
        "protected_area_dn": {**DEFAULTS, "designation": ["national"]},
        "protected_area_land": {**DEFAULTS, "realm": "land"},
        "protected_area_sdiepa_to_is_dn_land": {
            "status": ["designated", "inscribed", "established", "proposed", "adopted"],
            "site_types": ["oecm"],
            "iucn": ["strict"],
            "designation": ["national"],
            "realm": "land",
        },
    }
    for name, values in cases.items():
        assert build_predefined_name("protected_area", values) == name


def test_names_roundtrip_exhaustively():
    """31 x 3 x 15 x 7 x 2 = 19 530 distinct names, each resolving back."""
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
            "timing": "current",
            "status": status,
            "site_types": sites,
            "iucn": iucn,
            "designation": designation,
            "realm": realm,
        }
        names.add(assert_roundtrip("protected_area", values))
    assert len(names) == 31 * 3 * 15 * 7 * 2


def test_legacy_name_resolves_to_defaults():
    """Saved protected_area variables keep resolving."""
    assert resolve_predefined("protected_area") == ("protected_area", {})
