"""Entry fields that depend on the param values, read through the resolvers."""

import pytest

from gui.scripts.predefined import (
    DYNAMIC_ATTRS,
    PREDEFINED_CATALOGUE,
    entry_for_name,
    resolved_entry,
    validate_entry,
)

KEY = "demo_layer"


def _image(aoi, year=None, level=5, mode="plain"):
    return None


DEMO = {
    "label_key": "vars.predefined.precipitation",
    "description_key": "vars.predefined_info.precipitation",
    "var_type": "GEEVar",
    "temporal": True,
    "group": "climate",
    "raster_type": lambda p: "categorical" if p["mode"] == "plain" else "continuous",
    "years": lambda p: [2001, 2002] if p["mode"] == "plain" else [2003],
    "default_scale": 30,
    "vis_params": {"palette": ["000000", "ffffff"]},
    "unit_key": lambda p: f"legend.unit.{p['mode']}",
    "validate": lambda p, year: (
        ("x.err", {"mode": p["mode"], "year": year}) if year == 1999 else None
    ),
    "get_image": _image,
    "params": [
        {
            "key": "level",
            "label_key": "x",
            "hint_key": "x",
            "type": "int",
            "default": 5,
            "min": 1,
            "max": 20,
            "suffix_prefix": "lv",
        },
        {
            "key": "mode",
            "label_key": "x",
            "hint_key": "x",
            "type": "choice",
            "default": "plain",
            "options": ["plain", "fancy"],
            "option_label_key_prefix": "x.",
            "suffix_prefix": "",
            "advanced": True,
        },
    ],
}


@pytest.fixture(autouse=True)
def demo(monkeypatch):
    """Demo."""
    monkeypatch.setitem(PREDEFINED_CATALOGUE, KEY, DEMO)


def test_dynamic_attrs_contract():
    """Dynamic attrs contract."""
    assert DYNAMIC_ATTRS == (
        "years",
        "raster_type",
        "default_scale",
        "vis_params",
        "unit_key",
        "legend_class_keys",
    )


def test_resolved_entry_evaluates_callables_with_full_params():
    """Resolved entry evaluates callables with full params."""
    entry = resolved_entry(KEY, {"mode": "fancy"})
    assert entry["raster_type"] == "continuous"
    assert entry["years"] == [2003]
    assert entry["unit_key"] == "legend.unit.fancy"
    # Static fields are copied as they are.
    assert entry["default_scale"] == 30
    assert entry["vis_params"] == {"palette": ["000000", "ffffff"]}
    # Non-dynamic callables are left alone.
    assert entry["get_image"] is _image
    assert callable(entry["validate"])


def test_resolved_entry_fills_defaults():
    """Resolved entry fills defaults."""
    assert resolved_entry(KEY, {})["raster_type"] == "categorical"
    assert resolved_entry(KEY, None)["years"] == [2001, 2002]


def test_resolved_entry_is_a_copy():
    """Resolved entry is a copy."""
    entry = resolved_entry(KEY, {})
    entry["raster_type"] = "changed"
    assert callable(PREDEFINED_CATALOGUE[KEY]["raster_type"])


def test_a_static_entry_resolves_to_an_equal_copy():
    """A static entry resolves to an equal copy."""
    entry = resolved_entry("altitude", {})
    assert entry == PREDEFINED_CATALOGUE["altitude"]
    assert entry is not PREDEFINED_CATALOGUE["altitude"]


def test_resolved_entry_raises_for_an_unknown_key():
    """Resolved entry raises for an unknown key."""
    with pytest.raises(KeyError):
        resolved_entry("not_a_layer", {})


def test_entry_for_name_returns_full_params():
    """Entry for name returns full params."""
    key, params, entry = entry_for_name("demo_layer_lv7_fancy")
    assert key == KEY
    assert params == {"level": 7, "mode": "fancy"}
    assert entry["raster_type"] == "continuous"


def test_entry_for_name_fills_the_bare_key():
    """A bare key resolves to (key, {}); entry_for_name fills the defaults."""
    key, params, entry = entry_for_name("demo_layer")
    assert key == KEY
    assert params == {"level": 5, "mode": "plain"}
    assert entry["raster_type"] == "categorical"


@pytest.mark.parametrize("name", ["loss_forest_gfc_tc30_2015_2020", "", None])
def test_entry_for_name_misses(name):
    """Entry for name misses."""
    assert entry_for_name(name) == (None, {}, None)


def test_validate_entry():
    """Validate entry."""
    assert validate_entry(KEY, {}, 2001) is None
    assert validate_entry(KEY, {"mode": "fancy"}, 1999) == (
        "x.err",
        {"mode": "fancy", "year": 1999},
    )
    assert validate_entry("altitude", {}, 2020) is None  # no hook declared
    assert validate_entry("not_a_layer", {}, 2020) is None
