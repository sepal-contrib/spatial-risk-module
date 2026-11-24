"""Interval targets (deforestation_*/degradation_*) through the downstream steps."""

import inspect
from types import SimpleNamespace as NS

import pytest

from gui.scripts import inference_runner as ir
from gui.scripts.dataset_validation import variable_choices
from gui.tile import evaluation_helpers as h
from gui.widget.variable_list import derived_source_key
from spatialrisk.evaluation import interval_from_target

DEFO = "deforestation_gfc_tc30_2015_2020"


def _var(name, year=None, data_type="raster"):
    return NS(name=name, year=year, path=f"/data/{name}.tif", data_type=data_type)


def _project(**variables):
    return NS(processed_variables=dict(variables), predictions={})


# --- time interval / dataset ---------------------------------------------------


@pytest.mark.parametrize(
    "name,years",
    [
        (DEFO, 5),
        ("deforestation_gfc_tc100_2001_2026", 25),
        ("deforestation_tmf_degraded_2010_2018", 8),
        ("degradation_tmf_2023_2025", 2),
    ],
)
def test_interval_from_target_reads_the_trailing_pair(name, years):
    """Train/JNR/MW/evaluation all derive time_interval this way."""
    assert interval_from_target(name) == years


def test_dataset_offers_an_interval_target_once_as_a_static_choice():
    """Non-temporal: no year to pick, the choice value is the bare name."""

    class _P:
        def list_unique_variable_names(self, source):
            return [DEFO, "forest_gfc_tc30"]

        def is_temporal(self, name, source):
            return name == "forest_gfc_tc30"

        def get_variable_years(self, name, source):
            return [2015, 2020]

    defo = [c for c in variable_choices(_P()) if c["name"] == DEFO]
    assert defo == [{"value": DEFO, "text": DEFO, "name": DEFO, "year": None}]


# --- evaluation ------------------------------------------------------------------


def test_parse_interval_for_an_interval_target():
    """The interval field auto-fills from the target name."""
    assert h.parse_interval(_project(**{DEFO: _var(DEFO)}), DEFO) == 5


@pytest.mark.parametrize(
    "truth,expected",
    [
        (DEFO, ("forest_gfc_tc30", 2015)),
        ("deforestation_tmf_2012_2016", ("forest_tmf", 2012)),
        ("deforestation_tmf_degraded_2012_2016", ("forest_tmf_degraded", 2012)),
        ("degradation_tmf_2012_2016", ("forest_tmf", 2012)),
        ("loss_forest_gfc_tc30_2015_2020", None),
        ("my_truth", None),
    ],
)
def test_expected_forest_for_truth(truth, expected):
    """Each interval target names the forest layer + year it starts from."""
    assert h.expected_forest_for_truth(truth) == expected


def test_forest_key_for_truth_picks_the_matching_threshold_and_year():
    """Not the first forest_gfc: the one with the target's tc and start year."""
    proj = _project(
        forest_gfc_tc75_2015=_var("forest_gfc_tc75", 2015),
        forest_gfc_tc30_2010=_var("forest_gfc_tc30", 2010),
        forest_gfc_tc30_2015=_var("forest_gfc_tc30", 2015),
        **{DEFO: _var(DEFO)},
    )
    assert h.default_forest_key(proj) == "forest_gfc_tc75_2015"  # unchanged
    assert h.forest_key_for_truth(proj, DEFO) == "forest_gfc_tc30_2015"
    assert h.missing_forest_for_truth(proj, DEFO) is None


def test_missing_forest_for_truth_names_what_to_add():
    """No matching layer: the dialog can say which one to add."""
    proj = _project(**{DEFO: _var(DEFO)})
    assert h.forest_key_for_truth(proj, DEFO) is None
    assert h.missing_forest_for_truth(proj, DEFO) == ("forest_gfc_tc30", 2015)
    assert h.missing_forest_for_truth(proj, "") is None


def test_default_forest_key_ignores_interval_targets():
    """deforestation_gfc resolves to its own key, never to forest_gfc."""
    assert h.default_forest_key(_project(**{DEFO: _var(DEFO)})) is None


def test_truth_cannot_double_as_forest_at_start():
    """Forest = the 1/0/255 target would count only deforested pixels as forest."""
    spec, err = h.build_truth_spec(_project(**{DEFO: _var(DEFO)}), DEFO, DEFO, "5")
    assert spec is None and err


def test_dialog_seeds_the_paired_forest_and_shows_the_hint():
    """Source inspection (handler closures): truth change re-seeds the forest."""
    from gui.widget import evaluation_form_dialog as mod

    src = inspect.getsource(mod.EvaluationFormDialog)
    assert "forest_key_for_truth(" in src
    assert "missing_forest_for_truth(" in src
    assert "tiles.evaluation.forest_for_truth_hint" in src


# --- Predict mask --------------------------------------------------------------


def _mask_project():
    return _project(
        forest_gfc_tc30_2015=_var("forest_gfc_tc30", 2015),
        **{
            DEFO: _var(DEFO),
            "deforestation_tmf_2015_2020": _var("deforestation_tmf_2015_2020"),
        },
    )


def test_interval_targets_are_not_mask_candidates():
    """1 = event / 0 = stable as a 1 = keep mask would drop every stable pixel."""
    assert ir.mask_layer_candidates(_mask_project()) == ["forest_gfc_tc30_2015"]


def test_suggested_mask_is_still_the_sole_forest_layer():
    """deforestation_gfc must not count as a second Hansen forest layer."""
    assert ir.suggested_mask_layer(_mask_project()) == "forest_gfc_tc30_2015"


def test_resolve_mask_rejects_an_interval_target():
    """The runner refuses what the dialog never offers."""
    with pytest.raises(ValueError):
        ir._resolve_mask(_mask_project(), DEFO)


# --- derived-layer source -------------------------------------------------------


def test_derived_source_prefers_the_longest_token_bounded_match():
    """Definition variants share a prefix; the longer source name must win."""
    raw = {
        "deforestation_tmf_2015_2020": NS(
            name="deforestation_tmf_2015_2020", year=None
        ),
        "deforestation_tmf_degraded_2015_2020": NS(
            name="deforestation_tmf_degraded_2015_2020", year=None
        ),
        "forest_gfc_tc3_2015": NS(name="forest_gfc_tc3", year=2015),
        "forest_gfc_tc30_2015": NS(name="forest_gfc_tc30", year=2015),
    }
    p = NS(raw_variables=raw)
    assert (
        derived_source_key(p, "deforestation_tmf_degraded_2015_2020_dist", "?")
        == "deforestation_tmf_degraded_2015_2020"
    )
    assert (
        derived_source_key(p, "deforestation_tmf_2015_2020_edge", "?")
        == "deforestation_tmf_2015_2020"
    )
    assert derived_source_key(p, "forest_gfc_tc30_dist", "?", year=2015) == (
        "forest_gfc_tc30_2015"
    )
    assert derived_source_key(p, "loss_forest_gfc_tc3_2015_2020", "?") == (
        "forest_gfc_tc3_2015"
    )


# --- Part D period guards (MW / JNR / Evaluate / Allocate) ---------------------


@pytest.mark.parametrize(
    "target",
    [DEFO, "deforestation_tmf_2015_2020", "degradation_tmf_2015_2020"],
)
def test_interval_targets_pass_the_period_guards(target):
    """Every consumer that refuses a target without _YYYY_YYYY accepts these."""
    from gui.scripts.allocation_runner import _no_interval_caveat
    from gui.scripts.model_registry import MODEL_KEYS, interval_target_error

    for model_key in MODEL_KEYS:
        assert interval_target_error(model_key, target) is None, model_key
    pred = NS(dataset_name="ds", model_key="glm")
    project = NS(get_dataset=lambda name: NS(target=NS(name=target)))
    assert _no_interval_caveat(project, pred) is None
    proj = _project(**{target: _var(target)})
    assert h.parse_interval(proj, target) == 5


def test_non_interval_target_is_still_refused():
    """Control: the guard itself still bites a single-period target."""
    from gui.scripts.model_registry import interval_target_error

    assert interval_target_error("jnr", "burned_area") is not None
