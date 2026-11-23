"""A single-period target (burned_area_sep) has no ``_YYYY_YYYY`` interval.

Every workflow that needs a period length must say so clearly instead of
crashing late; ML prediction must keep working.
"""

import re
import types
from pathlib import Path

import pytest

TARGET = "burned_area_sep"


# --- Predict ------------------------------------------------------------------


class _RecordingModel:
    def __init__(self):
        self.apply_calls = []

    def apply(self, *args, **kwargs):
        self.apply_calls.append((args, kwargs))
        return Path("/tmp/out.tif")


def _inference_project(tmp_path, model, model_key):
    target = types.SimpleNamespace(name=TARGET, path=tmp_path / "t.tif")
    dataset = types.SimpleNamespace(name="calibration", target=target, features=[])
    folders = types.SimpleNamespace(
        **{
            k: tmp_path / k
            for k in ("glm_model", "rf_model", "icar_model", "rmj_bm", "rmj_mw")
        }
    )
    return types.SimpleNamespace(
        models={model_key: model},
        get_dataset=lambda name: dataset,
        processed_variables={},
        folders=folders,
    )


def test_jnr_prediction_refuses_a_single_period_target(tmp_path):
    """Jnr prediction refuses a single period target."""
    from gui.scripts.inference_runner import run_inference

    model = _RecordingModel()
    project = _inference_project(tmp_path, model, "jnr_v1")
    message = f"Cannot derive time_interval from target '{TARGET}'"
    with pytest.raises(ValueError, match=re.escape(message)):
        run_inference(project, "jnr_v1", "calibration")
    assert model.apply_calls == []


def test_glm_prediction_takes_a_single_period_target(tmp_path):
    """Glm prediction takes a single period target."""
    from gui.scripts.inference_runner import run_inference

    model = _RecordingModel()
    project = _inference_project(tmp_path, model, "glm_v1")
    run_inference(project, "glm_v1", "calibration")
    assert len(model.apply_calls) == 1


# --- Evaluate -----------------------------------------------------------------


def test_library_evaluation_raises_before_writing(monkeypatch, tmp_path):
    """Library evaluation raises before writing."""
    import spatialrisk.evaluation as ev

    monkeypatch.setattr(
        ev,
        "resolve_layers",
        lambda project, pred: {
            "defor_file": "d.tif",
            "forest_file": "f.tif",
            "riskmap_file": "r.tif",
            "time_interval": None,
            "period": "dataset_1",
        },
    )

    def _must_not_run(**kwargs):
        raise AssertionError("must not run without a time interval")

    monkeypatch.setattr(ev, "_defrate_per_cat", _must_not_run)
    monkeypatch.setattr(ev, "validate_two_layer", _must_not_run)
    project = types.SimpleNamespace(
        folders=types.SimpleNamespace(project_folder=str(tmp_path))
    )
    pred = types.SimpleNamespace(
        model_key="glm_v1", window=None, dataset_name="dataset_1"
    )
    with pytest.raises(ValueError, match="start and end year"):
        ev.evaluate_prediction(project, pred)
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize("interval", [None, "", "  "])
def test_truth_spec_explains_a_blank_interval(interval):
    """Truth spec explains a blank interval."""
    from gui.tile.evaluation_helpers import build_truth_spec

    spec, err = build_truth_spec(None, "truth", "forest", interval)
    assert spec is None
    assert "could not be read" in err


# --- Allocate -----------------------------------------------------------------


def _alloc_project(tmp_path, target_name, model_key="rf_v1"):
    pred = types.SimpleNamespace(
        model_key=model_key,
        dataset_name="dataset_1",
        window=None,
        path=tmp_path / "prob.tif",
        defrate_path=None,
        run_params={},
        model_snapshot={},
    )
    dataset = types.SimpleNamespace(
        name="dataset_1",
        target=types.SimpleNamespace(name=target_name, path=str(tmp_path / "t.tif")),
        features=[],
    )
    return types.SimpleNamespace(
        predictions={"run": pred},
        processed_variables={},
        datasets={"dataset_1": dataset},
        get_dataset=lambda name: dataset,
        allocations={},
    )


def _form(tmp_path, **overrides):
    from gui.scripts.allocation_runner import AllocationForm, BordersSelection

    borders = tmp_path / "borders.gpkg"
    borders.write_bytes(b"")
    fields = dict(
        name="run1",
        prediction_key="run",
        user_defrate_path=None,
        borders=BordersSelection(method="FILE", file_path=str(borders)),
        mask_file=None,
        defor_juris_ha=100.0,
        years_forecast=10.0,
    )
    fields.update(overrides)
    return AllocationForm(**fields)


def test_allocation_preview_reports_unavailable(tmp_path):
    """Allocation preview reports unavailable."""
    from gui.scripts.allocation_runner import preview_defrate_source

    src = preview_defrate_source(_alloc_project(tmp_path, TARGET), "run")
    assert src.provenance == "unavailable"
    assert TARGET in src.caveat and "start and end year" in src.caveat
    ok = preview_defrate_source(
        _alloc_project(tmp_path, "forest_loss_2010_2019"), "run"
    )
    assert ok.provenance == "computed"


def test_allocation_form_is_blocked_with_the_reason(tmp_path):
    """Allocation form is blocked with the reason."""
    from gui.scripts.allocation_runner import validate_form

    err = validate_form(_alloc_project(tmp_path, TARGET), _form(tmp_path))
    assert err and "start and end year" in err


def test_a_manual_rate_table_lifts_the_block(tmp_path):
    """A manual rate table lifts the block."""
    from gui.scripts.allocation_runner import validate_form

    table = tmp_path / "rates.csv"
    table.write_text("cat\n1\n")
    form = _form(tmp_path, user_defrate_path=str(table))
    assert validate_form(_alloc_project(tmp_path, TARGET), form) is None


def test_an_interval_target_still_asks_for_the_forest_layer(tmp_path):
    """An interval target still asks for the forest layer."""
    from gui.scripts.allocation_runner import validate_form

    err = validate_form(
        _alloc_project(tmp_path, "forest_loss_2010_2019"), _form(tmp_path)
    )
    assert err and "forest layer" in err
