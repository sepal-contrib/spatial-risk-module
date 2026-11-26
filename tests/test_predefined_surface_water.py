"""surface_water: JRC Global Surface Water v1.4 (static, 1984-2021) binary mask."""

import itertools

from _catalogue_checks import assert_roundtrip, multichoice_subsets
from _ee_expr import fake_aoi, patch_ee

import gui.scripts.predefined.water as water
from gui.scripts.predefined import (
    PREDEFINED_CATALOGUE,
    PREDEFINED_NAMES,
    default_param_values,
    param_specs,
)

GSW = "ee.Image('JRC/GSW1_4/GlobalSurfaceWater')"
TAIL = ".unmask(0).clip(ee.FeatureCollection('AOI')).toByte()"


def test_default_expression(monkeypatch):
    """Occurrence >= 1 and transition in permanent + seasonal classes."""
    fake = patch_ee(monkeypatch, water)
    text = repr(water._get_surface_water(fake_aoi(fake)))
    assert text == (
        f"{GSW}.select('occurrence').gte(1)"
        f".And({GSW}.select('transition')"
        ".remap([1, 2, 7, 4, 5, 8], [1, 1, 1, 1, 1, 1], 0))" + TAIL
    )


def test_threshold_and_types_reach_the_expression(monkeypatch):
    """Both knobs land in the EE expression."""
    fake = patch_ee(monkeypatch, water)
    text = repr(
        water._get_surface_water(
            fake_aoi(fake), occurrence=50, water_types=["ephemeral"]
        )
    )
    assert text == (
        f"{GSW}.select('occurrence').gte(50)"
        f".And({GSW}.select('transition').remap([9, 10], [1, 1], 0))" + TAIL
    )


def test_lost_water_never_counts():
    """Transition 3 / 6 (lost permanent / seasonal) are in no type group."""
    flat = [c for classes in water.GSW_TYPE_CLASSES.values() for c in classes]
    assert sorted(flat + [3, 6]) == list(range(1, 11))
    assert len(flat) == len(set(flat))


def test_entry_metadata():
    """Static categorical layer in the water group, right after rivers."""
    entry = PREDEFINED_CATALOGUE["surface_water"]
    assert entry["group"] == "water"
    assert entry["temporal"] is False
    assert entry["raster_type"] == "categorical"
    assert "years" not in entry
    assert entry.get("default_scale") is None  # native 30 m = download fallback
    names = list(PREDEFINED_NAMES)
    assert names.index("surface_water") == names.index("rivers") + 1
    assert default_param_values("surface_water") == {
        "occurrence": 1,
        "water_types": ["permanent", "seasonal"],
    }


def test_names_roundtrip_exhaustively():
    """100 thresholds x 7 type sets, every name distinct and resolving back."""
    specs = {s["key"]: s for s in param_specs("surface_water")}
    names = set()
    for occurrence, types in itertools.product(
        range(1, 101), multichoice_subsets(specs["water_types"])
    ):
        values = {"occurrence": occurrence, "water_types": types}
        names.add(assert_roundtrip("surface_water", values))
    assert len(names) == 700
    assert {"surface_water", "surface_water_o50", "surface_water_tpse"} <= names
