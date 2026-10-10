"""Name grammar for advanced params, exercised on a synthetic catalogue entry.

A synthetic entry keeps these tests independent of which real layers use which
features. It is injected with ``monkeypatch.setitem`` so it vanishes after
each test.
"""

import pytest

from gui.scripts.predefined import PREDEFINED_CATALOGUE
from gui.scripts.predefined.params import (
    advanced_param_specs,
    basic_param_specs,
    build_predefined_name,
    changed_advanced_count,
    coerce_param_values,
    default_param_values,
    parse_segment,
    resolve_predefined,
    spec_segments,
)

DEMO = {
    "label_key": "vars.predefined.rivers",
    "description_key": "vars.predefined_info.rivers",
    "var_type": "GEEVar",
    "raster_type": "categorical",
    "temporal": False,
    "group": "water",
    "get_image": lambda aoi, year=None, **params: None,
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
            "key": "kinds",
            "label_key": "x",
            "hint_key": "x",
            "type": "multichoice",
            "default": ["alpha", "beta"],
            "options": [
                {"value": "alpha", "code": "a"},
                {"value": "beta", "code": "b"},
                {"value": "gamma", "code": "g"},
            ],
            "option_label_key_prefix": "x.",
            "suffix_prefix": "k",
            "advanced": True,
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


@pytest.fixture
def demo(monkeypatch):
    """Inject the synthetic entry for one test."""
    monkeypatch.setitem(PREDEFINED_CATALOGUE, "demo", DEMO)
    return "demo"


def test_basic_and_advanced_split(demo):
    """Specs split on the ``advanced`` flag, keeping declared order."""
    assert [s["key"] for s in basic_param_specs(demo)] == ["level"]
    assert [s["key"] for s in advanced_param_specs(demo)] == ["kinds", "mode"]


def test_defaults_omit_advanced_segments(demo):
    """Advanced params at their defaults leave no trace in the name."""
    values = default_param_values(demo)
    assert values == {"level": 5, "kinds": ["alpha", "beta"], "mode": "plain"}
    assert build_predefined_name(demo, values) == "demo_lv5"


def test_missing_omittable_values_take_their_default(demo):
    """A missing omit_default value counts as its default."""
    assert build_predefined_name(demo, {"level": 5}) == "demo_lv5"


@pytest.mark.parametrize(
    "values, name",
    [
        ({"level": 5, "kinds": ["alpha"], "mode": "plain"}, "demo_lv5_ka"),
        ({"level": 5, "kinds": ["alpha", "beta"], "mode": "fancy"}, "demo_lv5_fancy"),
        (
            {"level": 7, "kinds": ["beta", "gamma"], "mode": "fancy"},
            "demo_lv7_kbg_fancy",
        ),
        (
            {"level": 7, "kinds": ["alpha", "beta", "gamma"], "mode": "plain"},
            "demo_lv7_kabg",
        ),
    ],
)
def test_build_resolve_roundtrip(demo, values, name):
    """Changed advanced values write segments that resolve back."""
    assert build_predefined_name(demo, values) == name
    assert resolve_predefined(name) == (demo, values)


def test_resolve_fills_defaults_for_omitted_segments(demo):
    """Omitted segments resolve to their defaults."""
    assert resolve_predefined("demo_lv5") == (
        demo,
        {"level": 5, "kinds": ["alpha", "beta"], "mode": "plain"},
    )


def test_bare_key_keeps_the_legacy_empty_params(demo):
    """A bare catalogue key resolves to ``(key, {})`` as before."""
    assert resolve_predefined("demo") == (demo, {})


def test_multichoice_order_is_canonical(demo):
    """Any pick order gives the declared-order name."""
    a = build_predefined_name(demo, {"level": 5, "kinds": ["gamma", "alpha"]})
    b = build_predefined_name(demo, {"level": 5, "kinds": ["alpha", "gamma"]})
    assert a == b == "demo_lv5_kag"


@pytest.mark.parametrize(
    "name",
    [
        "demo_lv5_kga",  # codes out of declared order
        "demo_lv5_kaa",  # duplicated code
        "demo_lv5_kab",  # explicit default segment (canonical form omits it)
        "demo_lv5_plain",  # explicit default choice
        "demo_lv05",  # zero-padded int
        "demo_lv5_kx",  # unknown code
        "demo_lv5_k",  # empty selection
        "demo_lv5_fancy_ka",  # segments out of spec order
        "demo_ka",  # required basic segment missing
        "demo_lv5_",  # trailing empty segment
        "demo_lv5_fancy_extra",  # leftover segment
    ],
)
def test_resolve_rejects_non_canonical_names(demo, name):
    """Only the name the builder emits resolves."""
    assert resolve_predefined(name) == (None, {})


def test_coerce_multichoice_canonicalises(demo):
    """Multichoice values come back deduplicated in declared order."""
    values, bad = coerce_param_values(
        demo, {"level": "5", "kinds": ["gamma", "alpha", "alpha"], "mode": "plain"}
    )
    assert bad is None
    assert values == {"level": 5, "kinds": ["alpha", "gamma"], "mode": "plain"}


@pytest.mark.parametrize("kinds", [[], None, "alpha", ["alpha", "zeta"]])
def test_coerce_multichoice_rejects(demo, kinds):
    """Empty, non-list or unknown multichoice values fail validation."""
    values, bad = coerce_param_values(
        demo, {"level": "5", "kinds": kinds, "mode": "plain"}
    )
    assert values == {}
    assert bad["key"] == "kinds"


def test_parse_segment_reports_match_and_value(demo):
    """``parse_segment`` returns ``(matched, value)`` per spec type."""
    kinds = DEMO["params"][1]
    assert parse_segment(kinds, "kbg") == (True, ["beta", "gamma"])
    assert parse_segment(kinds, "kgb") == (False, None)
    assert parse_segment(DEMO["params"][0], "lv12") == (True, 12)
    assert parse_segment(DEMO["params"][2], "fancy") == (True, "fancy")


def test_spec_segments_enumerates_every_emittable_segment(demo):
    """Every emittable segment is listed, multichoice in canonical form."""
    kinds = DEMO["params"][1]
    assert spec_segments(kinds) == ["ka", "kb", "kg", "kab", "kag", "kbg", "kabg"]
    assert spec_segments(DEMO["params"][2]) == ["plain", "fancy"]
    assert spec_segments(DEMO["params"][0])[:2] == ["lv1", "lv2"]
    assert len(spec_segments(DEMO["params"][0])) == 20


def test_changed_advanced_count_accepts_typed_and_raw_values(demo):
    """Only advanced params differing from their default count."""
    assert changed_advanced_count(demo, default_param_values(demo)) == 0
    assert changed_advanced_count(demo, {"kinds": ["beta", "alpha"]}) == 0
    assert changed_advanced_count(demo, {"kinds": ["alpha"], "mode": "fancy"}) == 2
    # basic params never count, even when changed
    assert changed_advanced_count(demo, {"level": "9"}) == 0
