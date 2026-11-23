"""The catalogue lives in gui.scripts.predefined; the old module re-exports it."""

import importlib

EXPECTED_KEYS = {
    "altitude",
    "slope",
    "protected_area",
    "rivers",
    "roads",
    "forest_gfc",
    "forest_tmf",
    "towns",
    "precipitation",
    "temperature_2m",
    "dry_spell",
    "burned_area",
    "active_fire",
    "fire_history",
    "subj",
}

THEME_OF = {
    "altitude": "terrain",
    "slope": "terrain",
    "protected_area": "infrastructure",
    "roads": "infrastructure",
    "towns": "infrastructure",
    "rivers": "water",
    "forest_gfc": "forest",
    "forest_tmf": "forest",
    "precipitation": "climate",
    "temperature_2m": "climate",
    "dry_spell": "climate",
    "burned_area": "fire",
    "active_fire": "fire",
    "fire_history": "fire",
    "subj": "admin",
}


def test_package_exposes_the_full_catalogue():
    """Every pre-split layer is still in the catalogue."""
    from gui.scripts.predefined import PREDEFINED_CATALOGUE, PREDEFINED_NAMES

    assert set(PREDEFINED_CATALOGUE) == EXPECTED_KEYS
    assert set(PREDEFINED_NAMES) == EXPECTED_KEYS


def test_each_builder_lives_in_its_theme_module():
    """Builders moved to one module per theme (tests patch ``ee`` there)."""
    from gui.scripts.predefined import PREDEFINED_CATALOGUE

    for key, theme in THEME_OF.items():
        builder = PREDEFINED_CATALOGUE[key]["get_image"]
        assert builder.__module__ == f"gui.scripts.predefined.{theme}", key


def test_theme_modules_have_disjoint_entries():
    """``catalogue.py`` merges theme ENTRIES; no key may be declared twice."""
    seen = set()
    for theme in sorted(set(THEME_OF.values())):
        module = importlib.import_module(f"gui.scripts.predefined.{theme}")
        overlap = seen & set(module.ENTRIES)
        assert not overlap, (theme, overlap)
        seen |= set(module.ENTRIES)
    assert seen == EXPECTED_KEYS


def test_shim_reexports_the_same_objects():
    """Old import path keeps working and shares state with the package."""
    import gui.scripts.predefined as pkg
    import gui.scripts.predefined_variables as pv

    assert pv.PREDEFINED_CATALOGUE is pkg.PREDEFINED_CATALOGUE
    for name in (
        "resolve_predefined",
        "build_predefined_name",
        "coerce_param_values",
        "default_param_values",
        "param_specs",
        "resolve_aoi_ee",
        "get_aoi_ee_feature",
    ):
        assert getattr(pv, name) is getattr(pkg, name), name
    for builder in (
        "_get_altitude",
        "_get_slope",
        "_get_protected_area",
        "_get_rivers",
        "_get_roads",
        "_get_forest_gfc",
        "_get_forest_tmf",
        "_get_towns",
        "_get_precipitation",
        "_get_temperature",
        "_get_subj",
        "_aoi_bounds",
    ):
        assert callable(getattr(pv, builder)), builder
