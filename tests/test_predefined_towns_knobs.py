"""towns: GHSL population/built-up thresholds (default) or SMOD classes."""

import pytest
from _catalogue_checks import assert_roundtrip, multichoice_subsets
from _ee_expr import fake_aoi, patch_ee

import gui.scripts.predefined.infrastructure as infra
from gui.scripts.predefined import (
    active_param_specs,
    build_predefined_name,
    coerce_param_values,
    default_param_values,
    param_specs,
    resolve_predefined,
    resolved_entry,
)

SMOD_DEFAULT = ["centre", "dense", "semidense", "suburban"]
GHSL_DEFAULTS = {
    "method": "ghsl",
    "population_threshold": 15,
    "built_threshold": 90,
    "smod_classes": SMOD_DEFAULT,
}


def _golden_towns(ee, aoi, year):
    """Pre-B2 ``_get_towns`` body, verbatim."""
    epochs = list(range(1975, 2021, 5))
    epoch = min(epochs, key=lambda x: abs(x - year))
    pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{epoch}")
    built = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}").select("built_surface")
    return ee.Image(0).where(pop.gte(15).And(built.gte(90)), 1).clip(aoi)


@pytest.mark.parametrize("year", [1975, 1992, 2020, 2023])
def test_towns_default_expression_is_unchanged(monkeypatch, year):
    """Epoch-nearest logic and thresholds 15 / 90 exactly as before."""
    fake = patch_ee(monkeypatch, infra)
    aoi = fake_aoi(fake)
    golden = repr(_golden_towns(fake, aoi, year))
    assert repr(infra._get_towns(aoi, year)) == golden
    assert repr(infra._get_towns(aoi, year, **GHSL_DEFAULTS)) == golden


def test_thresholds_reach_the_expression(monkeypatch):
    """Both GHSL thresholds are user values now."""
    fake = patch_ee(monkeypatch, infra)
    text = repr(
        infra._get_towns(
            fake_aoi(fake), 2020, population_threshold=50, built_threshold=200
        )
    )
    assert ".gte(50).And(" in text
    assert ".select('built_surface').gte(200)" in text


def test_smod_maps_selected_classes_to_one(monkeypatch):
    """SMOD: remap the chosen class codes to 1, everything else to 0, byte."""
    fake = patch_ee(monkeypatch, infra)
    text = repr(
        infra._get_towns(
            fake_aoi(fake), 2030, method="smod", smod_classes=["centre", "dense"]
        )
    )
    assert text == (
        "ee.Image('JRC/GHSL/P2023A/GHS_SMOD/2030').select('smod_code')"
        ".remap([30, 23], [1, 1], 0).unmask(0)"
        ".clip(ee.FeatureCollection('AOI')).toByte()"
    )


def test_smod_default_is_the_ghsl_urban_domain():
    """Default classes are the urban domain: codes 30, 23, 22, 21."""
    assert default_param_values("towns")["smod_classes"] == SMOD_DEFAULT
    assert [infra.SMOD_CLASS_CODES[c] for c in SMOD_DEFAULT] == [30, 23, 22, 21]
    assert sorted(infra.SMOD_CLASS_CODES.values()) == [11, 12, 13, 21, 22, 23, 30]


def test_smod_epoch_is_the_nearest_declared_one(monkeypatch):
    """SMOD runs 1975-2030 (2025/2030 projected); off-grid years snap."""
    fake = patch_ee(monkeypatch, infra)
    text = repr(infra._get_towns(fake_aoi(fake), 2023, method="smod"))
    assert "GHS_SMOD/2025" in text


def test_unknown_method_is_rejected(monkeypatch):
    """A typo never silently falls back to the threshold method."""
    fake = patch_ee(monkeypatch, infra)
    with pytest.raises(ValueError):
        infra._get_towns(fake_aoi(fake), 2020, method="degurba")


def test_years_and_scale_follow_the_method():
    """GHSL keeps today's years and 30 m fallback; smod adds 2025/2030 at 1 km."""
    ghsl = resolved_entry("towns", {})
    assert ghsl["years"] == list(range(1975, 2021, 5))
    assert ghsl["default_scale"] is None
    smod = resolved_entry("towns", {"method": "smod"})
    assert smod["years"] == list(range(1975, 2031, 5))
    assert smod["default_scale"] == 1000


def test_active_specs_follow_the_method():
    """depends_on: thresholds only for ghsl, classes only for smod."""
    ghsl = [s["key"] for s in active_param_specs("towns", {"method": "ghsl"})]
    smod = [s["key"] for s in active_param_specs("towns", {"method": "smod"})]
    assert ghsl == ["method", "population_threshold", "built_threshold"]
    assert smod == ["method", "smod_classes"]


def test_switching_method_drops_hidden_values_from_the_name():
    """Review focus 2: values typed for the other method never leak.

    An inactive spec is coerced to its default, even when its raw text is
    invalid, so a stale field can neither block submit nor reach the name.
    """
    values, bad = coerce_param_values(
        "towns",
        {
            "method": "smod",
            "population_threshold": "abc",
            "built_threshold": "200",
            "smod_classes": ["centre"],
        },
    )
    assert bad is None
    assert values["population_threshold"] == 15
    assert values["built_threshold"] == 90
    assert build_predefined_name("towns", values) == "towns_smod_uc"

    values, bad = coerce_param_values(
        "towns",
        {
            "method": "ghsl",
            "population_threshold": "50",
            "built_threshold": "90",
            "smod_classes": ["centre"],
        },
    )
    assert bad is None
    assert values["smod_classes"] == SMOD_DEFAULT
    assert build_predefined_name("towns", values) == "towns_p50"


def test_towns_names_roundtrip():
    """Every canonical ghsl/smod combination round-trips; legacy names hold."""
    specs = {s["key"]: s for s in param_specs("towns")}
    names = set()
    for population in (1, 15, 16, 999):
        for built in (1, 90, 91, 999):
            values = {
                **GHSL_DEFAULTS,
                "population_threshold": population,
                "built_threshold": built,
            }
            names.add(assert_roundtrip("towns", values))
    for classes in multichoice_subsets(specs["smod_classes"]):
        values = {**GHSL_DEFAULTS, "method": "smod", "smod_classes": classes}
        names.add(assert_roundtrip("towns", values))
    assert len(names) == 16 + 127
    assert {"towns", "towns_p16_b91", "towns_smod", "towns_smod_ucd"} <= names
    p50_b200 = {**GHSL_DEFAULTS, "population_threshold": 50, "built_threshold": 200}
    assert build_predefined_name("towns", p50_b200) == "towns_p50_b200"
    assert resolve_predefined("towns") == ("towns", {})
    assert resolve_predefined("towns_2020") == (None, {})
