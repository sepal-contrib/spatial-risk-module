"""rivers water classes and forest_tmf forest definition."""

import types

import pytest

from gui.scripts.predefined import (
    build_predefined_name,
    coerce_param_values,
    default_param_values,
    resolve_predefined,
)


class _Rec:
    """Chainable ee stand-in recording (method, args)."""

    def __init__(self, calls):
        self._calls = calls

    def __getattr__(self, method):
        def _call(*args, **kwargs):
            self._calls.append((method, args))
            return self

        return _call


def _fake_ee(calls):
    def _factory(name):
        def _make(*args, **kwargs):
            calls.append((name, args))
            return _Rec(calls)

        return _make

    class _Geometry:
        def bounds(self):
            return "BOUNDS"

    return types.SimpleNamespace(
        Image=_factory("Image"),
        ImageCollection=_factory("ImageCollection"),
        Geometry=_Geometry,
    )


class _Aoi:
    def geometry(self):
        class _G:
            def bounds(self):
                return "BOUNDS"

        return _G()


# --------------------------------------------------------------------- rivers


def test_rivers_default_classes_and_bare_name():
    """Default classes are every class but ocean, and keep the bare name."""
    assert default_param_values("rivers") == {
        "classes": ["large", "major", "canal", "stream"]
    }
    assert build_predefined_name("rivers", default_param_values("rivers")) == "rivers"


@pytest.mark.parametrize(
    "classes, name",
    [
        (["large", "major", "canal"], "rivers_wlmc"),
        (["large", "major", "canal", "stream", "ocean"], "rivers_wlmcso"),
        (["stream"], "rivers_ws"),
    ],
)
def test_rivers_names_roundtrip(classes, name):
    """A non-default selection builds a coded name that resolves back."""
    assert build_predefined_name("rivers", {"classes": classes}) == name
    assert resolve_predefined(name) == ("rivers", {"classes": classes})


def test_rivers_requires_one_class():
    """An empty selection is rejected on the classes key."""
    values, bad = coerce_param_values("rivers", {"classes": []})
    assert values == {} and bad["key"] == "classes"


def test_rivers_default_keeps_todays_gte2_expression(monkeypatch):
    """Byte parity with the pre-change layer: the default path is ``gte(2)``."""
    import gui.scripts.predefined.water as water

    calls = []
    monkeypatch.setattr(water, "ee", _fake_ee(calls))
    water._get_rivers(_Aoi())

    methods = [m for m, _ in calls]
    assert ("gte", (2,)) in calls
    assert "remap" not in methods
    assert methods[-1] == "toByte"


def test_rivers_selection_remaps_class_ids(monkeypatch):
    """A non-default selection remaps the chosen OSM ids to 1."""
    import gui.scripts.predefined.water as water

    calls = []
    monkeypatch.setattr(water, "ee", _fake_ee(calls))
    water._get_rivers(_Aoi(), classes=["large", "major", "ocean"])

    assert ("remap", ([2, 3, 1], [1, 1, 1], 0)) in calls
    assert ("gte", (2,)) not in calls
    assert ("unmask", ()) in calls


# ----------------------------------------------------------------- forest_tmf


def test_tmf_default_definition_and_bare_name():
    """The default definition is undisturbed and keeps the bare name."""
    assert default_param_values("forest_tmf") == {"definition": "undisturbed"}
    assert build_predefined_name("forest_tmf", {"definition": "undisturbed"}) == (
        "forest_tmf"
    )


@pytest.mark.parametrize("definition", ["degraded", "regrowth"])
def test_tmf_definition_names_roundtrip(definition):
    """A non-default definition is suffixed to the name and resolves back."""
    name = build_predefined_name("forest_tmf", {"definition": definition})
    assert name == f"forest_tmf_{definition}"
    assert resolve_predefined(name) == ("forest_tmf", {"definition": definition})


def test_tmf_years_reach_2025():
    """TMF v1_2024 (Dec2024 band) supports forest at 1 Jan 2025."""
    from gui.scripts.predefined import PREDEFINED_CATALOGUE

    years = PREDEFINED_CATALOGUE["forest_tmf"]["years"]
    assert years[0] == 2001 and years[-1] == 2025


def test_tmf_definition_param_is_shared_shape():
    """Part C reuses this factory for deforestation_tmf."""
    from gui.scripts.predefined.forest import TMF_FOREST_CLASSES, tmf_definition_param

    spec = tmf_definition_param()
    assert spec["options"] == list(TMF_FOREST_CLASSES)
    assert spec["default"] == "undisturbed" and spec["advanced"] is True
    assert tmf_definition_param() is not tmf_definition_param()  # fresh dicts


def test_tmf_undisturbed_keeps_todays_expression(monkeypatch):
    """The default definition emits the inherited where/eq/neq expression."""
    import gui.scripts.predefined.forest as forest

    calls = []
    monkeypatch.setattr(forest, "ee", _fake_ee(calls))
    forest._get_forest_tmf(_Aoi(), 2020)

    assert ("select", ("Dec2019",)) in calls
    assert ("eq", (2,)) in calls and ("neq", (1,)) in calls
    assert "remap" not in [m for m, _ in calls]


@pytest.mark.parametrize(
    "definition, classes",
    [("degraded", [1, 2]), ("regrowth", [1, 2, 4])],
)
def test_tmf_definitions_remap_their_classes(monkeypatch, definition, classes):
    """Non-default definitions remap their forest classes to 1, else 0."""
    import gui.scripts.predefined.forest as forest

    calls = []
    monkeypatch.setattr(forest, "ee", _fake_ee(calls))
    forest._get_forest_tmf(_Aoi(), 2025, definition=definition)

    assert ("select", ("Dec2024",)) in calls
    assert ("remap", (classes, [1] * len(classes), 0)) in calls


# ------------------------------------------------------------ describe_params


def test_describe_params_is_empty_for_defaults_and_others():
    """Default settings and non-catalogue names describe to an empty string."""
    from gui.scripts.predefined import describe_params

    for name in ("rivers", "forest_tmf", "forest_gfc_tc30", "altitude", "custom_x", ""):
        assert describe_params(name) == "", name


def test_describe_params_lists_changed_advanced_values():
    """Changed advanced values are listed with translated labels."""
    from gui.i18n import t
    from gui.scripts.predefined import describe_params

    text = describe_params("rivers_wlmc")
    assert text == (
        f"{t('vars.modal.param_water_classes')}: "
        f"{t('vars.modal.water_large')}, {t('vars.modal.water_major')}, "
        f"{t('vars.modal.water_canal')}"
    )
    assert describe_params("forest_tmf_degraded") == (
        f"{t('vars.modal.param_tmf_definition')}: {t('vars.modal.tmf_def_degraded')}"
    )
