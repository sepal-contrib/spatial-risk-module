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
    resolved_entry,
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
    """Every own-param combination under which the pair is in use x start < end.

    Combinations with the pair inactive (logging_mapbiomas By year) are covered
    by the inactive-pair tests below.
    """
    own = [s for s in param_specs(key) if s["type"] != "year"]
    years = PREDEFINED_CATALOGUE[key]["interval_years"]
    for combo in itertools.product(*[_own_values(s) for s in own]):
        base = {s["key"]: v for s, v in zip(own, combo)}
        if resolved_entry(key, base).get("interval_years"):
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
        controllers = entry.get("interval_depends_on")
        if controllers:
            # The pair is in use only under these values; there the entry is
            # non-temporal.
            period = {c: allowed[0] for c, allowed in controllers.items()}
            assert resolved_entry(key, period)["temporal"] is False, key
        else:
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


# --- an interval pair that depends on a param (logging_mapbiomas) ----------------

LOG = "logging_mapbiomas"
YEARLY = {"region": "brazil", "timing": "yearly", "start_year": None, "end_year": None}
PERIOD = {"region": "brazil", "timing": "period", "start_year": 2015, "end_year": 2020}


def test_interval_depends_on_is_copied_onto_both_year_specs():
    """``interval_depends_on`` becomes the pair's ``depends_on``; others get none."""
    for spec in param_specs(LOG)[-2:]:
        assert spec["type"] == "year"
        assert spec["depends_on"] == {"timing": ["period"]}
    for spec in param_specs("deforestation_gfc")[-2:]:
        assert "depends_on" not in spec


def test_inactive_pair_is_never_written_into_the_name():
    """By year writes no years (no ``_None_None``), even if stale ones are passed."""
    assert build_predefined_name(LOG, YEARLY) == "logging_mapbiomas_brazil"
    stale = {**YEARLY, "start_year": 2015, "end_year": 2020}
    assert build_predefined_name(LOG, stale) == "logging_mapbiomas_brazil"
    assert resolve_predefined("logging_mapbiomas_brazil") == (LOG, YEARLY)


def test_active_pair_round_trips():
    """Over a period: the pair trails the timing segment."""
    name = build_predefined_name(LOG, PERIOD)
    assert name == "logging_mapbiomas_brazil_period_2015_2020"
    assert resolve_predefined(name) == (LOG, PERIOD)
    assert interval_from_target(name) == 5


def test_coerce_skips_an_inactive_pair():
    """By year: blank year selects neither block submit nor reach the values."""
    raw = {"region": "brazil", "timing": "yearly", "start_year": "", "end_year": "x"}
    assert coerce_param_values(LOG, raw) == (YEARLY, None)


def test_coerce_still_requires_an_active_pair():
    """Over a period: a blank start names the start select; reversed -> order."""
    raw = {"region": "brazil", "timing": "period", "start_year": "", "end_year": "2020"}
    values, bad = coerce_param_values(LOG, raw)
    assert values == {} and bad["key"] == "start_year"
    raw = {**raw, "start_year": "2020", "end_year": "2015"}
    values, bad = coerce_param_values(LOG, raw)
    assert bad["error_key"] == "vars.modal.error_interval_order"


def test_resolved_entry_empties_interval_years_when_the_pair_is_inactive():
    """Consumers asking "is this an interval target" read the resolved entry."""
    assert resolved_entry(LOG, YEARLY)["interval_years"] == []
    assert resolved_entry(LOG, PERIOD)["interval_years"] == list(range(1988, 2026))
    assert (
        resolved_entry("deforestation_gfc", {})["interval_years"]
        == PREDEFINED_CATALOGUE["deforestation_gfc"]["interval_years"]
    )


def test_param_rows_hide_an_inactive_pair():
    """Details dialog: By year lists no start/end rows."""
    from gui.i18n import t
    from gui.scripts.predefined import changed_advanced_count, param_rows

    yearly = param_rows(LOG, YEARLY)
    assert [label for label, _ in yearly] == [
        t("vars.modal.param_mapbiomas_region"),
        t("vars.modal.param_logging_timing"),
    ]
    assert yearly[1][1] == t("vars.modal.logging_timing_yearly")
    assert param_rows(LOG, PERIOD)[-2:] == [
        (t("vars.modal.param_start_year"), "2015"),
        (t("vars.modal.param_end_year"), "2020"),
    ]
    assert changed_advanced_count(LOG, YEARLY) == 0
    assert changed_advanced_count(LOG, PERIOD) == 0


def test_interval_target_checks_follow_the_period():
    """Inference mask and evaluation: By year is not an interval target."""
    from types import SimpleNamespace as NS

    from gui.scripts.inference_runner import _is_interval_target

    assert _is_interval_target(NS(name="logging_mapbiomas_brazil")) is False
    period = NS(name="logging_mapbiomas_brazil_period_2015_2020")
    assert _is_interval_target(period) is True
    assert interval_from_target("logging_mapbiomas_brazil") is None
