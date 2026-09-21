"""_register_prediction: building and registering a model's output raster."""

from pathlib import Path

from spatialrisk import Project
from spatialrisk.mlmodels import GLMModel


class _FakeVar:
    def __init__(self, name, year=None):
        self.name = name
        self.year = year


class _FakeDataset:
    def __init__(self, name="ds_2020", year=2020):
        self.name = name
        self.year = year
        self.target = _FakeVar("forest_loss", year)
        self.features = [_FakeVar("slope")]


def test_register_prediction_builds_and_registers(monkeypatch):
    """A run registers under its provenance-derived key with both snapshots."""
    project = Project(project_name="reg_test")
    model = GLMModel(name="m1", model_type="glm", year=2020)
    project.add_model(model, auto_save=False)

    ds = _FakeDataset()
    pred = model._register_prediction(Path("/tmp/glm.tif"), dataset=ds, auto_save=False)

    assert pred is not None
    assert pred.model_key == "glm_m1"  # reverse-looked-up from project.models
    assert pred.dataset_name == "ds_2020"
    assert pred.year == 2020
    assert pred.window is None
    assert pred.model_snapshot["model_type"] == "glm"  # full model config copy
    assert pred.dataset_snapshot["feature_names"] == ["slope"]
    assert project.get_prediction("glm_m1__ds_2020_y2020") is pred


def test_register_prediction_with_window(monkeypatch):
    """A window discriminator keeps multi-output runs distinct."""
    project = Project(project_name="reg_test_w")
    model = GLMModel(name="mw1", model_type="mw", year=2020)
    project.add_model(model, key="mw_custom", auto_save=False)

    pred = model._register_prediction(
        Path("/tmp/mw_5.tif"), dataset=_FakeDataset(), window=5, auto_save=False
    )
    assert pred.model_key == "mw_custom"  # honors custom registry key
    assert pred.window == 5
    assert project.get_prediction("mw_custom__ds_2020_y2020_w5") is pred


def test_register_prediction_noop_without_project():
    """A direct apply() outside a project context registers nothing."""
    model = GLMModel(name="orphan", model_type="glm")
    result = model._register_prediction(
        Path("/tmp/x.tif"), dataset=_FakeDataset(), auto_save=False
    )
    assert result is None


def test_pending_name_overrides_prediction_key_and_label():
    """A pending name keys and labels the output by the user's choice.

    It replaces the provenance-derived key, so distinct runs do not
    collide.
    """
    project = Project(project_name="named_pred")
    model = GLMModel(name="m1", model_type="glm", year=2020)
    project.add_model(model, auto_save=False)
    model._pending_pred_name = "run_a"

    pred = model._register_prediction(
        Path("/tmp/glm.tif"), dataset=_FakeDataset(), auto_save=False
    )

    assert pred.name == "run_a"
    assert pred.model_key == "glm_m1"  # provenance field kept for evaluation
    assert project.get_prediction("run_a") is pred
    assert project.get_prediction("glm_m1__ds_2020_y2020") is None  # NOT the old key


def test_pending_name_keeps_window_suffix_distinct():
    """Multi-output runs (MW windows) stay distinct under one pending name."""
    project = Project(project_name="named_pred_w")
    model = GLMModel(name="mw1", model_type="mw", year=2020)
    project.add_model(model, key="mw_mw1", auto_save=False)
    model._pending_pred_name = "run_b"

    p5 = model._register_prediction(
        Path("/tmp/mw_5.tif"), dataset=_FakeDataset(), window=5, auto_save=False
    )
    p11 = model._register_prediction(
        Path("/tmp/mw_11.tif"), dataset=_FakeDataset(), window=11, auto_save=False
    )

    assert project.get_prediction("run_b_w5") is p5
    assert project.get_prediction("run_b_w11") is p11
    assert p5.name == p11.name == "run_b"
    # filter by name groups both windows of the run together (drives the tile's
    # per-job map toggle).
    assert set(project.filter_predictions(name="run_b")) == {"run_b_w5", "run_b_w11"}


def test_register_prediction_records_pending_run_params():
    """Run-time choices (the ML mask layer) are frozen onto the Prediction.

    The mask is an argument to ``apply()``, not model config, so without this
    it would be absent from the prediction's provenance and the details dialog
    could never say what the run was masked with.
    """
    project = Project(project_name="run_params")
    model = GLMModel(name="m1", model_type="glm", year=2020)
    project.add_model(model, auto_save=False)
    model._pending_run_params = {"mask_layer": "forest_2020"}

    pred = model._register_prediction(
        Path("/tmp/glm.tif"), dataset=_FakeDataset(), auto_save=False
    )

    assert pred.run_params == {"mask_layer": "forest_2020"}


def test_register_prediction_run_params_default_empty():
    """A run that recorded nothing yields an empty dict, never None."""
    project = Project(project_name="run_params_empty")
    model = GLMModel(name="m1", model_type="glm", year=2020)
    project.add_model(model, auto_save=False)

    pred = model._register_prediction(
        Path("/tmp/glm.tif"), dataset=_FakeDataset(), auto_save=False
    )

    assert pred.run_params == {}


def test_run_params_survive_the_project_json_round_trip():
    """Provenance is only useful if it outlives the session that produced it.

    Mirrors exactly what Project.save()/load() do with a prediction:
    ``model_dump(mode="json")`` out, ``Prediction(**data)`` back in.
    """
    from spatialrisk.predictions.prediction import Prediction

    project = Project(project_name="run_params_rt")
    model = GLMModel(name="m1", model_type="glm", year=2020)
    project.add_model(model, auto_save=False)
    model._pending_run_params = {"mask_layer": "forest_2020"}
    pred = model._register_prediction(
        Path("/tmp/glm.tif"), dataset=_FakeDataset(), auto_save=False
    )

    restored = Prediction(**pred.model_dump(mode="json"))

    assert restored.run_params == {"mask_layer": "forest_2020"}


def test_register_prediction_builds_display_overviews(tmp_path, monkeypatch):
    """Every written prediction gets overviews.

    Without them the viewer decimates a whole tiled raster for each zoomed-out
    tile it draws.
    """
    import spatialrisk.overviews as ov

    calls = []
    monkeypatch.setattr(
        ov, "ensure_overviews", lambda p, **kw: calls.append((str(p), kw)) or True
    )

    raster = tmp_path / "glm.tif"
    raster.write_bytes(b"not-a-real-tif")  # ensure_overviews is stubbed out

    project = Project(project_name="ovr_test")
    model = GLMModel(name="m1", model_type="glm", year=2020)
    project.add_model(model, auto_save=False)
    model._register_prediction(raster, dataset=_FakeDataset(), auto_save=False)

    assert [c[0] for c in calls] == [str(raster)]
    # Threshold-gated: small rasters are already fast to decimate.
    assert calls[0][1]["min_pixels"] == ov.OVERVIEW_MIN_PIXELS


def test_register_prediction_overview_failure_is_not_fatal(tmp_path, monkeypatch):
    """A prediction still registers when the overview pass fails.

    On a read-only folder or an unreadable file the display is slower; the
    artifact is not lost.
    """
    import spatialrisk.overviews as ov

    def boom(path, **kwargs):
        raise RuntimeError("no overviews for you")

    monkeypatch.setattr(ov, "ensure_overviews", boom)

    raster = tmp_path / "glm.tif"
    raster.write_bytes(b"x")
    project = Project(project_name="ovr_fail")
    model = GLMModel(name="m1", model_type="glm", year=2020)
    project.add_model(model, auto_save=False)

    pred = model._register_prediction(raster, dataset=_FakeDataset(), auto_save=False)
    assert pred is not None


def test_register_prediction_skips_overviews_for_a_missing_file(monkeypatch):
    """Nothing written, nothing to optimise — and no spurious error."""
    import spatialrisk.overviews as ov

    calls = []
    monkeypatch.setattr(ov, "ensure_overviews", lambda p, **kw: calls.append(p))

    project = Project(project_name="ovr_missing")
    model = GLMModel(name="m1", model_type="glm", year=2020)
    project.add_model(model, auto_save=False)
    model._register_prediction(
        Path("/tmp/does-not-exist.tif"), dataset=_FakeDataset(), auto_save=False
    )
    assert calls == []
