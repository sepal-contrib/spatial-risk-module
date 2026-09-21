"""Deleting a model or a prediction from a project.

Both remove the registry entry AND clean up the owned on-disk artifacts, with a
within-project safety guard so a stray path can never delete files outside the
project folder.
"""

from pathlib import Path

import spatialrisk.project as proj_mod
from spatialrisk.mlmodels import GLMModel, ICARModel, MWModel
from spatialrisk.predictions.prediction import Prediction
from spatialrisk.project import Project

Project._ensure_model_schemas()


def _setup(tmp_path, monkeypatch):
    """A project rooted at tmp_path/proj with a data/ subfolder for artifacts."""
    monkeypatch.setattr(proj_mod, "downloads_folder", tmp_path)
    p = Project(project_name="proj")
    data_dir = tmp_path / "proj" / "data"
    data_dir.mkdir(parents=True)
    return p, data_dir


# --- predictions -----------------------------------------------------------


def test_delete_prediction_removes_entry_and_file(tmp_path, monkeypatch):
    """The registry entry and the output raster both go."""
    p, data_dir = _setup(tmp_path, monkeypatch)
    raster = data_dir / "pred.tif"
    raster.write_bytes(b"x")
    p.predictions["glm__ds"] = Prediction(
        path=raster, model_key="glm", dataset_name="ds"
    )

    assert p.delete_prediction("glm__ds") is True
    assert "glm__ds" not in p.predictions
    assert not raster.exists()


def test_delete_prediction_missing_key_returns_false(tmp_path, monkeypatch):
    """An unknown key is a no-op, not an error."""
    p, _ = _setup(tmp_path, monkeypatch)
    assert p.delete_prediction("nope") is False


def test_delete_prediction_keeps_file_when_requested(tmp_path, monkeypatch):
    """Unregisters the prediction but keeps the raster when asked to."""
    p, data_dir = _setup(tmp_path, monkeypatch)
    raster = data_dir / "pred.tif"
    raster.write_bytes(b"x")
    p.predictions["k"] = Prediction(path=raster, model_key="glm", dataset_name="ds")

    assert p.delete_prediction("k", delete_file=False) is True
    assert raster.exists()


def test_delete_refuses_to_unlink_outside_project(tmp_path, monkeypatch):
    """A path outside the project folder is never unlinked."""
    p, _ = _setup(tmp_path, monkeypatch)
    outside = tmp_path / "outside.tif"  # sibling of the project folder, not inside it
    outside.write_bytes(b"x")
    p.predictions["k"] = Prediction(path=outside, model_key="glm", dataset_name="ds")

    assert p.delete_prediction("k") is True  # entry is still removed
    assert outside.exists()  # but the out-of-project file is preserved


# --- models ----------------------------------------------------------------


def test_delete_model_removes_entry_and_artifacts(tmp_path, monkeypatch):
    """A deleted model takes its pickle and training sample with it."""
    p, data_dir = _setup(tmp_path, monkeypatch)
    model_path = data_dir / "glm.pkl"
    samples_path = data_dir / "samples.csv"
    model_path.write_bytes(b"x")
    samples_path.write_text("a,b\n")
    model = GLMModel.model_construct(model_path=model_path, samples_path=samples_path)
    p.models["glm"] = model

    assert p.delete_model("glm") is True
    assert "glm" not in p.models
    assert not model_path.exists()
    assert not samples_path.exists()


def test_delete_model_missing_key_returns_false(tmp_path, monkeypatch):
    """An unknown model key is a no-op, not an error."""
    p, _ = _setup(tmp_path, monkeypatch)
    assert p.delete_model("nope") is False


def test_icar_output_files_includes_rho():
    """The iCAR model owns its rho raster, so cleanup must know about it."""
    rho = Path("/proj/data/rho.tif")
    icar = ICARModel.model_construct(model_path=None, samples_path=None, rho_path=rho)
    assert rho in icar.output_files()


def test_mw_output_files_includes_ldefrate_rasters():
    """MW owns one deforestation-rate raster per window."""
    d5 = Path("/proj/data/defrate_5.tif")
    mw = MWModel.model_construct(
        model_path=None, samples_path=None, ldefrate_files={"5": d5}
    )
    assert d5 in mw.output_files()


def test_delete_prediction_removes_the_overview_sidecar(tmp_path, monkeypatch):
    """A left-behind .ovr would be picked up by the next raster of that name."""
    p, data_dir = _setup(tmp_path, monkeypatch)
    raster = data_dir / "pred.tif"
    raster.write_bytes(b"x")
    sidecar = data_dir / "pred.tif.ovr"
    sidecar.write_bytes(b"o")
    p.predictions["glm__ds"] = Prediction(
        path=raster, model_key="glm", dataset_name="ds"
    )

    assert p.delete_prediction("glm__ds") is True
    assert not raster.exists()
    assert not sidecar.exists()
