"""JRC TMF interval targets: truth tables against AnnualChanges class sequences.

Semantics verified live 2026-10-09 (see the Part C plan): deforestation = end
class in {3, 4, 5} and different from the start class (agrees with TMF's
DeforestationYear on 99.98 % of events); degradation = class 2 in any year of
the period (equals DegradationYear exactly).
"""

import inspect
import itertools

import numpy as np
import pytest
from np_ee import FakeAoi, NpImage, export, fake_ee

import gui.scripts.predefined.forest_change as fc
from gui.scripts.predefined import PREDEFINED_CATALOGUE
from gui.scripts.predefined.forest import ENTRIES as FOREST_ENTRIES
from gui.scripts.predefined.forest import TMF_FOREST_CLASSES, tmf_definition_param

S, E = 2016, 2019  # bands Dec2015 (start state) .. Dec2018 (end state)


def _scene(monkeypatch, sequences, inside):
    """One pixel per class sequence over Dec{S-1}..Dec{E-1}."""
    bands = {
        f"Dec{S - 1 + i}": np.array([[seq[i] for seq in sequences]])
        for i in range(E - S + 1)
    }
    fake = fake_ee((1, len(sequences)), collections={"tmf": NpImage(bands)})
    monkeypatch.setattr(fc, "ee", fake)
    return FakeAoi(np.array([inside])), fake


def _defo_expected(start, end, inside, definition):
    forest = TMF_FOREST_CLASSES[definition]
    if not inside or start not in forest or end == 0:
        return 255
    if end in (3, 4, 5) and end != start:
        return 1
    return 0


@pytest.mark.parametrize("definition", sorted(TMF_FOREST_CLASSES))
def test_deforestation_tmf_truth_table(monkeypatch, definition):
    """Every start x end class (0..6), inside and outside the AOI."""
    rows = list(itertools.product(range(7), range(7), (True, False)))
    sequences = [(s,) * (E - S) + (e,) for s, e, _ in rows]
    aoi, _fake = _scene(monkeypatch, sequences, [r[2] for r in rows])

    got = export(
        fc._get_deforestation_tmf(aoi, definition=definition, start_year=S, end_year=E),
        aoi,
    )

    want = [_defo_expected(s, e, inside, definition) for s, e, inside in rows]
    np.testing.assert_array_equal(got[0], np.array(want, dtype=np.uint8))


def test_degradation_alone_is_not_deforestation(monkeypatch):
    """Undisturbed -> degraded stays 0 (stable) for the deforestation target."""
    aoi, _ = _scene(monkeypatch, [(1, 1, 2, 2)], [True])
    got = export(fc._get_deforestation_tmf(aoi, start_year=S, end_year=E), aoi)
    assert got.tolist() == [[0]]


def _deg_expected(sequence, inside):
    start, end = sequence[0], sequence[-1]
    if not inside or start != 1:
        return 255
    if 2 in sequence[1:]:
        return 1
    if end == 1:
        return 0
    return 255


def test_degradation_tmf_truth_table(monkeypatch):
    """Exhaustive over classes 0..4 for the 4 states of a 3-year period."""
    rows = [
        (seq, inside)
        for seq in itertools.product(range(5), repeat=E - S + 1)
        for inside in (True, False)
    ]
    aoi, _ = _scene(monkeypatch, [r[0] for r in rows], [r[1] for r in rows])

    got = export(fc._get_degradation_tmf(aoi, start_year=S, end_year=E), aoi)

    want = [_deg_expected(seq, inside) for seq, inside in rows]
    np.testing.assert_array_equal(got[0], np.array(want, dtype=np.uint8))


def test_degraded_then_cleared_counts_as_degradation(monkeypatch):
    """1 -> 2 -> 3 within the period: degradation event (DegradationYear agrees)."""
    aoi, _ = _scene(monkeypatch, [(1, 2, 3, 3), (1, 1, 3, 3)], [True, True])
    got = export(fc._get_degradation_tmf(aoi, start_year=S, end_year=E), aoi)
    assert got.tolist() == [[1, 255]]  # direct clearing is excluded, not "stable"


@pytest.mark.parametrize("builder", ["_get_deforestation_tmf", "_get_degradation_tmf"])
def test_tmf_builders_prefilter_with_the_aoi_bounds(monkeypatch, builder):
    """The filterBounds prefilter gets the AOI bounds, never the AOI itself."""
    aoi, fake = _scene(monkeypatch, [(1, 1, 1, 1)], [True])
    getattr(fc, builder)(aoi, start_year=S, end_year=E)
    assert fake.filter_bounds == ["BOUNDS"]


@pytest.mark.parametrize("builder", ["_get_deforestation_tmf", "_get_degradation_tmf"])
@pytest.mark.parametrize(
    "start,end", [(None, 2020), (2020, 2015), (2000, 2005), (2015, 2026)]
)
def test_tmf_builders_reject_a_bad_interval(monkeypatch, builder, start, end):
    """Checked against forest_tmf's years (2001-2025) before any EE call."""
    aoi, _ = _scene(monkeypatch, [(1, 1, 1, 1)], [True])
    with pytest.raises(ValueError):
        getattr(fc, builder)(aoi, start_year=start, end_year=end)


def test_tmf_entries():
    """Non-temporal interval entries over forest_tmf's years; definition reused."""
    years = FOREST_ENTRIES["forest_tmf"]["years"]
    defo = PREDEFINED_CATALOGUE["deforestation_tmf"]
    deg = PREDEFINED_CATALOGUE["degradation_tmf"]
    for entry in (defo, deg):
        assert entry["temporal"] is False
        assert entry["interval_years"] == years
        assert entry["raster_type"] == "categorical"
        assert entry["group"] == "forest_change"
    assert defo["params"] == [tmf_definition_param()]
    assert "params" not in deg
    assert defo["forest_at_start"]({"definition": "degraded"}) == (
        "forest_tmf",
        {"definition": "degraded"},
    )
    assert deg["forest_at_start"]({}) == ("forest_tmf", {"definition": "undisturbed"})


def test_tmf_kwarg_defaults_match_the_catalogue():
    """The definition default == tmf_definition_param()'s; the years have none."""
    sig = inspect.signature(fc._get_deforestation_tmf)
    assert sig.parameters["definition"].default == tmf_definition_param()["default"]
    for builder in (fc._get_deforestation_tmf, fc._get_degradation_tmf):
        params = inspect.signature(builder).parameters
        assert params["start_year"].default is None
        assert params["end_year"].default is None
