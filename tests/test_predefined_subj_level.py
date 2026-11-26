"""subj: GAUL district (default, as before) or province level."""

import pytest
from _catalogue_checks import assert_roundtrip
from _ee_expr import fake_aoi, patch_ee

import gui.scripts.predefined.admin as admin
import spatialrisk.gee.ee_fao_gaul as ee_fao_gaul
import spatialrisk.gee.ee_rasterize_unique_values as ee_rasterize
from gui.scripts.predefined import (
    PREDEFINED_CATALOGUE,
    default_param_values,
    param_specs,
    resolve_predefined,
)


@pytest.fixture
def fake_ee(monkeypatch):
    """One fake shared by the builder and the two spatialrisk helpers."""
    return patch_ee(monkeypatch, admin, ee_fao_gaul, ee_rasterize)


def _golden_subj(aoi):
    """Pre-B2 ``_get_subj`` body, verbatim (helpers resolve the patched ee)."""
    from spatialrisk.gee.ee_fao_gaul import get_fao_gaul_subj
    from spatialrisk.gee.ee_rasterize_unique_values import gee_rasterize_unique_values

    filtered_subj, _ = get_fao_gaul_subj(2, aoi)
    return (
        ee_rasterize.ee.Image(gee_rasterize_unique_values(filtered_subj, "gaul2_name"))
        .clip(aoi)
        .toByte()
    )


def test_subj_default_expression_is_unchanged(fake_ee):
    """No kwargs and explicit defaults both equal today's level-2 expression."""
    aoi = fake_aoi(fake_ee)
    golden = repr(_golden_subj(aoi))
    assert repr(admin._get_subj(aoi)) == golden
    assert repr(admin._get_subj(aoi, None, **default_param_values("subj"))) == golden


def test_province_level_reads_gaul_level_one(fake_ee):
    """Province level -> GAUL_2024_L1 only."""
    text = repr(admin._get_subj(fake_aoi(fake_ee), level="province"))
    assert "GAUL_2024_L1" in text
    assert "GAUL_2024_L2" not in text
    assert text.endswith(".clip(ee.FeatureCollection('AOI')).toByte()")


def test_subj_units_exposes_the_uncast_image(fake_ee):
    """Task 7 needs the numbering before the byte cast."""
    units, attribute, numbered = admin._subj_units(fake_aoi(fake_ee), "province")
    assert attribute == "gaul1_name"
    assert "GAUL_2024_L1" in repr(units)
    assert repr(numbered).endswith(".clip(ee.FeatureCollection('AOI'))")


def test_unknown_level_is_rejected(fake_ee):
    """A typo never silently falls back to districts."""
    with pytest.raises(ValueError):
        admin._get_subj(fake_aoi(fake_ee), level="county")


def test_level_param_is_advanced_choice():
    """One advanced choice; districts by default, omitted from the name."""
    (spec,) = param_specs("subj")
    assert spec["key"] == "level"
    assert spec["options"] == ["province", "district"]
    assert spec["default"] == "district"
    assert spec["advanced"] is True
    assert spec["omit_default"] is True
    assert PREDEFINED_CATALOGUE["subj"]["random_visualizer"] is True


def test_subj_names_roundtrip():
    """Names: subj (bare, legacy) and subj_province."""
    assert assert_roundtrip("subj", {"level": "district"}) == "subj"
    assert assert_roundtrip("subj", {"level": "province"}) == "subj_province"
    assert resolve_predefined("subj") == ("subj", {})
