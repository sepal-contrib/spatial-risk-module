"""Interval layers: the trailing _{start}_{end} pair (Part C contract).

Names keep the shape spatialrisk.evaluation.interval_from_target reads; the
pair is the only place a catalogue name may carry four-digit runs.
"""

import itertools
import re

import pytest

from gui.scripts.predefined import (
    PREDEFINED_CATALOGUE,
    build_predefined_name,
    coerce_param_values,
    default_param_values,
    param_specs,
    resolve_predefined,
    spec_segments,
)
from spatialrisk.evaluation import interval_from_target

INTERVAL_ENTRIES = sorted(
    k for k, e in PREDEFINED_CATALOGUE.items() if e.get("interval_years")
)
FOUR = re.compile(r"[0-9]{4}")


def _own_values(spec):
    """Every value a non-year spec accepts (multichoice: every allowed subset)."""
    kind = spec.get("type", "int")
    if kind == "int":
        return list(range(spec["min"], spec["max"] + 1))
    if kind == "choice":
        return list(spec["options"])
    if kind == "multichoice":
        values = [o["value"] for o in spec["options"]]
        low = spec.get("min_selected", 1)
        return [
            list(combo)
            for size in range(low, len(values) + 1)
            for combo in itertools.combinations(values, size)
        ]
    raise AssertionError(f"unexpected spec type {kind}")


def _all_values(key):
    own = [s for s in param_specs(key) if s["type"] != "year"]
    years = PREDEFINED_CATALOGUE[key]["interval_years"]
    for combo in itertools.product(*[_own_values(s) for s in own]):
        base = {s["key"]: v for s, v in zip(own, combo)}
        for start, end in itertools.combinations(years, 2):
            yield {**base, "start_year": start, "end_year": end}


def test_deforestation_gfc_has_the_two_trailing_year_specs():
    """Own params first, then start_year / end_year (type year, required)."""
    specs = param_specs("deforestation_gfc")
    assert [s["key"] for s in specs] == [
        "tree_cover_threshold",
        "start_year",
        "end_year",
    ]
    for spec in specs[1:]:
        assert spec["type"] == "year"
        assert (
            spec["options"]
            == PREDEFINED_CATALOGUE["deforestation_gfc"]["interval_years"]
        )
        assert spec["omit_default"] is False and spec["advanced"] is False
        assert spec["suffix_prefix"] == ""
        assert spec["default"] is None


def test_non_interval_entries_get_no_year_specs():
    """forest_gfc keeps its one param; nothing else grows year selects."""
    assert [s["key"] for s in param_specs("forest_gfc")] == ["tree_cover_threshold"]
    for key, entry in PREDEFINED_CATALOGUE.items():
        if not entry.get("interval_years"):
            assert all(s.get("type") != "year" for s in param_specs(key)), key


def test_contract_example_name():
    """The index's example builds and parses back exactly."""
    values = {"tree_cover_threshold": 30, "start_year": 2015, "end_year": 2020}
    name = build_predefined_name("deforestation_gfc", values)
    assert name == "deforestation_gfc_tc30_2015_2020"
    assert resolve_predefined(name) == ("deforestation_gfc", values)
    assert interval_from_target(name) == 5


@pytest.mark.parametrize("key", INTERVAL_ENTRIES)
def test_exhaustive_round_trip_and_interval(key):
    """Every own-param value x every start<end pair: inverse + interval + digits."""
    count = 0
    for values in _all_values(key):
        name = build_predefined_name(key, values)
        assert resolve_predefined(name) == (key, values), name
        start, end = values["start_year"], values["end_year"]
        assert name.endswith(f"_{start}_{end}"), name
        assert FOUR.findall(name) == [str(start), str(end)], name
        assert interval_from_target(name) == end - start, name
        count += 1
    assert count > 0


@pytest.mark.parametrize(
    "name",
    [
        "deforestation_gfc_tc30_2020_2015",  # reversed
        "deforestation_gfc_tc30_2015_2015",  # equal
        "deforestation_gfc_tc30_2015",  # missing end
        "deforestation_gfc_tc30_2010_2015_2020",  # three years
        "deforestation_gfc_2015_2020",  # required tc segment missing
        "deforestation_gfc_tc30_2000_2005",  # start outside interval_years
        "deforestation_gfc_tc30_2015_2027",  # end outside interval_years
        "deforestation_gfc_tc30_02015_2020",  # five digits
        "deforestation_gfc_tc30_2015_2020_dist",  # post-process output
        "deforestation_gfc_tc30_2015_2020_2020",  # storage key shape
        "loss_forest_gfc_tc30_2015_2020",  # Process-step change layer
        "forest_gfc_tc30_2015",  # storage key of a temporal layer
        "deforestation_tmf_2015_2020_degraded",  # param after the years
        "deforestation_tmf_banana_2015_2020",  # unknown definition
        "deforestation_tmf_2000_2005",  # outside forest_tmf's years
        "degradation_tmf_undisturbed_2015_2020",  # degradation takes no params
        "degradation_tmf_2020_2015",  # reversed
    ],
)
def test_resolve_rejects_malformed_interval_names(name):
    """Hand-edited or derived names fall through to custom styling, never crash."""
    assert resolve_predefined(name) == (None, {})


def test_bare_interval_key_keeps_the_bare_key_rule():
    """A bare catalogue key resolves with no params, as for every entry."""
    assert resolve_predefined("deforestation_gfc") == ("deforestation_gfc", {})


def test_default_values_leave_the_years_unset():
    """No silent default period: the modal seeds blank selects."""
    assert default_param_values("deforestation_gfc") == {
        "tree_cover_threshold": 30,
        "start_year": None,
        "end_year": None,
    }


def test_coerce_accepts_form_strings_and_ints():
    """Selects hand back ints, prefilled state holds strings: both coerce."""
    raw = {"tree_cover_threshold": "30", "start_year": "2015", "end_year": 2020}
    values, bad = coerce_param_values("deforestation_gfc", raw)
    assert bad is None
    assert values == {"tree_cover_threshold": 30, "start_year": 2015, "end_year": 2020}


@pytest.mark.parametrize(
    "text", ["", "  ", None, "None", "abc", "2015.0", "1999", "02015"]
)
def test_coerce_rejects_a_missing_or_invalid_start(text):
    """Blank, stringified None, non-numeric or out-of-range -> start_year spec."""
    raw = {"tree_cover_threshold": "30", "start_year": text, "end_year": "2020"}
    values, bad = coerce_param_values("deforestation_gfc", raw)
    assert values == {}
    assert bad["key"] == "start_year" and "error_key" not in bad


def test_coerce_reports_a_missing_end():
    """The end select is checked after the start select."""
    raw = {"tree_cover_threshold": "30", "start_year": "2015"}
    values, bad = coerce_param_values("deforestation_gfc", raw)
    assert values == {} and bad["key"] == "end_year"


@pytest.mark.parametrize("start,end", [("2020", "2015"), ("2015", "2015")])
def test_coerce_rejects_a_non_increasing_interval(start, end):
    """A start >= end -> the end_year spec carrying the interval-order error key."""
    raw = {"tree_cover_threshold": "30", "start_year": start, "end_year": end}
    values, bad = coerce_param_values("deforestation_gfc", raw)
    assert values == {}
    assert bad["key"] == "end_year"
    assert bad["error_key"] == "vars.modal.error_interval_order"


def test_year_specs_are_last_and_only_on_non_temporal_interval_entries():
    """Lint: the pair is trailing, ordered, and the entry is non-temporal."""
    for key in PREDEFINED_CATALOGUE:
        entry = PREDEFINED_CATALOGUE[key]
        kinds = [s.get("type") for s in param_specs(key)]
        if not entry.get("interval_years"):
            assert "year" not in kinds, key
            continue
        assert entry["temporal"] is False, key
        assert "years" not in entry, key
        assert kinds[-2:] == ["year", "year"] and "year" not in kinds[:-2], key
        assert [s["key"] for s in param_specs(key)[-2:]] == ["start_year", "end_year"]
        years = entry["interval_years"]
        assert len(years) >= 2 and years == sorted(set(years)), key
        assert all(1000 <= y <= 9999 for y in years), key


def test_only_year_segments_carry_four_digit_runs():
    """Lint: no other segment any entry can emit holds four consecutive digits."""
    for key in PREDEFINED_CATALOGUE:
        for spec in param_specs(key):
            segments = spec_segments(spec)
            if spec.get("type") == "year":
                assert all(FOUR.fullmatch(s) for s in segments), key
            else:
                assert not any(FOUR.search(s) for s in segments), (key, spec["key"])


def test_param_rows_list_the_period_one_field_each():
    """Details dialog: start and end year show as their own rows, as plain years."""
    from gui.i18n import t
    from gui.scripts.predefined import describe_params, param_rows

    values = {"tree_cover_threshold": 30, "start_year": 2015, "end_year": 2020}
    assert param_rows("deforestation_gfc", values) == [
        (t("vars.modal.param_tree_cover_threshold"), "30"),
        (t("vars.modal.param_start_year"), "2015"),
        (t("vars.modal.param_end_year"), "2020"),
    ]
    # The years are basic params, never part of the advanced summary.
    assert describe_params("deforestation_gfc_tc30_2015_2020") == ""


def test_tmf_names_resolve_with_the_definition_filled_in():
    """The default definition is omitted from the name and filled on resolve."""
    assert resolve_predefined("deforestation_tmf_2015_2020") == (
        "deforestation_tmf",
        {"definition": "undisturbed", "start_year": 2015, "end_year": 2020},
    )
    assert resolve_predefined("deforestation_tmf_degraded_2015_2020") == (
        "deforestation_tmf",
        {"definition": "degraded", "start_year": 2015, "end_year": 2020},
    )
    assert resolve_predefined("degradation_tmf_2015_2020") == (
        "degradation_tmf",
        {"start_year": 2015, "end_year": 2020},
    )
