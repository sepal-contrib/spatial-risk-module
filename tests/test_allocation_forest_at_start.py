"""The allocation form's "forest at period start" layer.

A computed rate table needs the forest raster at the start of the prediction's
calibration period. The library resolver only knows the Hansen naming
convention, so the form seeds an explicit choice from what the prediction
already recorded and lets the user override it from the processed rasters.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import gui.scripts.allocation_runner as runner
from gui.scripts.allocation_runner import (
    AllocationForm,
    BordersSelection,
    resolve_defrate_table,
    suggested_forest_file,
    validate_form,
)


def _var(path):
    return SimpleNamespace(path=path)


def _feature(name, path):
    return SimpleNamespace(name=name, path=path)


def _pred(**kw):
    base = dict(
        path=Path("/data/p/far_rf/run/dataset_1.tif"),
        model_key="rf_v1",
        dataset_name="dataset_1",
        window=None,
        defrate_path=None,
        run_params={},
        model_snapshot={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _project(pred, features=(), processed=None):
    dataset = SimpleNamespace(
        name="dataset_1",
        target=_feature("loss_2010_2019", "/data/loss.tif"),
        features=list(features),
    )
    return SimpleNamespace(
        predictions={"rf_run": pred},
        processed_variables=processed or {},
        datasets={"dataset_1": dataset},
        get_dataset=lambda name: dataset if name == "dataset_1" else None,
    )


# --- seeding -----------------------------------------------------------------


def test_seed_prefers_the_predictions_recorded_mask_layer():
    """The Predict dialog's mask (run_params.mask_layer) is the forest at start."""
    pred = _pred(run_params={"mask_layer": "geobosques_2010"})
    project = _project(
        pred,
        features=[_feature("forest_gfc_tc30", "/data/gfc.tif")],
        processed={"geobosques_2010": _var("/data/geobosques_2010.tif")},
    )
    assert suggested_forest_file(project, "rf_run") == "/data/geobosques_2010.tif"


def test_seed_falls_back_to_the_model_snapshots_forest_var():
    """JNR/MW runs name their forest feature on the model snapshot."""
    pred = _pred(model_key="jnr_v1", model_snapshot={"forest_var": "geobosques"})
    project = _project(
        pred,
        features=[
            _feature("altitude", "/data/alt.tif"),
            _feature("geobosques", "/data/geobosques.tif"),
        ],
    )
    assert suggested_forest_file(project, "rf_run") == "/data/geobosques.tif"


def test_seed_falls_back_to_the_hansen_prefix_scan():
    """Nothing recorded: today's library behaviour, the first forest_gfc* feature."""
    pred = _pred()
    project = _project(
        pred,
        features=[
            _feature("altitude", "/data/alt.tif"),
            _feature("forest_gfc_tc30", "/data/gfc.tif"),
        ],
    )
    assert suggested_forest_file(project, "rf_run") == "/data/gfc.tif"


def test_seed_is_none_when_nothing_matches():
    """A non-Hansen dataset with no recorded mask leaves the choice to the user."""
    pred = _pred()
    project = _project(pred, features=[_feature("geobosques", "/data/g.tif")])
    assert suggested_forest_file(project, "rf_run") is None


def test_seed_ignores_a_recorded_mask_that_left_the_project():
    """A stale run_params key must not raise; fall through to the next source."""
    pred = _pred(run_params={"mask_layer": "gone"})
    project = _project(pred, features=[_feature("forest_gfc", "/data/gfc.tif")])
    assert suggested_forest_file(project, "rf_run") == "/data/gfc.tif"


def test_seed_is_none_for_an_unknown_prediction():
    """A stale or cleared risk-map key seeds nothing rather than raising."""
    assert suggested_forest_file(_project(_pred()), "nope") is None


# --- resolution ----------------------------------------------------------------


def test_compute_passes_the_forest_file_to_the_resolver(tmp_path, monkeypatch):
    """The explicit forest file reaches resolve_layers and the rate computation."""
    seen = {}

    def fake_resolve_layers(project, pred, forest_file=None):
        seen["forest_file"] = forest_file
        return {
            "defor_file": "/d.tif",
            "forest_file": forest_file,
            "riskmap_file": str(pred.path),
            "time_interval": 9,
            "period": "dataset_1",
        }

    def fake_defrate_per_cat(**kwargs):
        seen["computed_with"] = kwargs["forest_file"]
        Path(kwargs["tab_file_defrate"]).write_text("cat\n1\n")

    monkeypatch.setattr(runner, "_resolve_layers", fake_resolve_layers)
    monkeypatch.setattr(runner, "_defrate_per_cat", fake_defrate_per_cat)
    project = _project(_pred(path=tmp_path / "prob.tif"))

    src = resolve_defrate_table(project, "rf_run", forest_file="/data/g.tif")

    assert seen == {"forest_file": "/data/g.tif", "computed_with": "/data/g.tif"}
    assert src.provenance == "computed"
    assert src.forest_file == "/data/g.tif"
    assert src.as_dict()["forest_file"] == "/data/g.tif"


def test_persisted_table_ignores_the_forest_file(tmp_path):
    """Only a computed table depends on the forest layer."""
    csv = tmp_path / "defrate_cat_bm.csv"
    csv.write_text("cat\n1\n")
    project = _project(_pred(model_key="jnr_v1", defrate_path=csv))

    src = resolve_defrate_table(project, "rf_run", forest_file="/data/g.tif")

    assert src.provenance == "persisted"
    assert src.forest_file is None
    assert "forest_file" in src.as_dict()


# --- validation ----------------------------------------------------------------


def _form(**kw):
    base = dict(
        name="run",
        prediction_key="rf_run",
        user_defrate_path=None,
        borders=BordersSelection(method="ADMIN0", admin_code="1"),
        mask_file=None,
        defor_juris_ha=100.0,
        years_forecast=4.0,
    )
    base.update(kw)
    return AllocationForm(**base)


def test_validate_requires_a_forest_file_when_the_table_will_be_computed(tmp_path):
    """A to-be-computed table without a forest layer is a form error, not a crash."""
    project = _project(_pred(path=tmp_path / "prob.tif"))
    error = validate_form(project, _form(forest_file=None))
    assert error is not None
    assert "forest" in error.lower()


def test_validate_accepts_an_existing_forest_file(tmp_path):
    """An existing forest raster satisfies the computed-table requirement."""
    forest = tmp_path / "g.tif"
    forest.write_bytes(b"")
    project = _project(_pred(path=tmp_path / "prob.tif"))
    assert validate_form(project, _form(forest_file=str(forest))) is None


def test_validate_rejects_a_missing_forest_file(tmp_path):
    """A forest path that vanished is reported by name, like the other files."""
    project = _project(_pred(path=tmp_path / "prob.tif"))
    error = validate_form(project, _form(forest_file="/nope/g.tif"))
    assert error is not None and "/nope/g.tif" in error


def test_validate_does_not_require_a_forest_file_for_a_persisted_table(tmp_path):
    """Ready-made tables never depend on the forest layer."""
    csv = tmp_path / "defrate_cat_bm.csv"
    csv.write_text("cat\n1\n")
    project = _project(_pred(model_key="jnr_v1", defrate_path=csv))
    assert validate_form(project, _form(forest_file=None)) is None


def test_validate_does_not_require_a_forest_file_for_a_custom_table(tmp_path):
    """A user-chosen table bypasses the forest requirement entirely."""
    csv = tmp_path / "rates.csv"
    csv.write_text("cat\n1\n")
    project = _project(_pred(path=tmp_path / "prob.tif"))
    assert validate_form(project, _form(user_defrate_path=str(csv))) is None


def test_missing_forest_file_at_run_time_is_the_resolver_error(tmp_path, monkeypatch):
    """Belt and braces: no forest file and no Hansen feature is still a clean error."""
    from spatialrisk.evaluation import resolve_layers

    monkeypatch.setattr(runner, "_resolve_layers", resolve_layers)
    project = _project(
        _pred(path=tmp_path / "prob.tif"),
        features=[_feature("geobosques", "/data/g.tif")],
    )
    with pytest.raises(ValueError, match="forest_gfc"):
        resolve_defrate_table(project, "rf_run")
