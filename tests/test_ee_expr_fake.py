"""The expression-recording ee fake used by the layer-knob builder tests."""

import pytest
from _ee_expr import Expr, fake_aoi, make_fake_ee, patch_ee


def test_constructor_and_method_chain_render_as_call_text():
    """A chain renders exactly as written, string arguments quoted."""
    ee = make_fake_ee()
    image = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(fake_aoi(ee))
    assert repr(image) == (
        "ee.Image('USGS/SRTMGL1_003').select('elevation')"
        ".clip(ee.FeatureCollection('AOI'))"
    )


def test_static_calls_kwargs_and_lists_render_deterministically():
    """Class-level calls record too; kwargs are sorted by name."""
    ee = make_fake_ee()
    join_filter = ee.Filter.intersects(rightField=".geo", leftField=".geo", maxError=10)
    assert repr(join_filter) == (
        "ee.Filter.intersects(leftField='.geo', maxError=10, rightField='.geo')"
    )
    assert repr(ee.Filter.inList("STATUS", ["A", "B"])) == (
        "ee.Filter.inList('STATUS', ['A', 'B'])"
    )
    assert repr(ee.Image(1).remap([1, 2], [1, 1], 0)) == (
        "ee.Image(1).remap([1, 2], [1, 1], 0)"
    )


def test_floats_are_rounded_and_callables_hidden():
    """Float noise never breaks parity; mapped functions render as <fn>."""
    ee = make_fake_ee()
    assert repr(ee.Image(1).multiply(0.1 + 0.2)) == "ee.Image(1).multiply(0.3)"
    mapped = ee.FeatureCollection("x").map(lambda feature: feature)
    assert repr(mapped) == "ee.FeatureCollection('x').map(<fn>)"


def test_fake_types_support_isinstance_and_equality():
    """Builders branch on isinstance(aoi, ee.FeatureCollection) and friends."""
    ee = make_fake_ee()
    aoi = fake_aoi(ee)
    assert isinstance(aoi, ee.FeatureCollection)
    assert isinstance(aoi, Expr)
    assert not isinstance(aoi, ee.Geometry)
    assert ee.Image(0) == ee.Image(0)
    assert ee.Image(0) != ee.Image(1)


def test_get_info_is_forbidden():
    """A unit test that reaches getInfo is a bug: it would need Earth Engine."""
    ee = make_fake_ee()
    with pytest.raises(AssertionError):
        ee.Number(1).getInfo()


def test_patch_ee_shares_one_fake_across_modules(monkeypatch):
    """Every patched module sees the same fake, so expressions compose."""
    import types

    first = types.ModuleType("first")
    second = types.ModuleType("second")
    fake = patch_ee(monkeypatch, first, second)
    assert first.ee is fake
    assert second.ee is fake


def test_roundtrip_helper_returns_the_name():
    """The shared round-trip helper builds, resolves and hands back the name."""
    from _catalogue_checks import assert_roundtrip

    assert assert_roundtrip("forest_gfc", {"tree_cover_threshold": 30}) == (
        "forest_gfc_tc30"
    )
