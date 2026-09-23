"""GLM and iCAR predictors charge for the design, not every level."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from _inference_fixture import build_dataset, build_glm, build_icar  # noqa: E402


def test_apply_charges_materialised_columns_not_levels(tmp_path, monkeypatch):
    """A 106-level categorical must not inflate n_design_cols."""
    from patsy import dmatrices
    from sklearn.linear_model import LogisticRegression

    from spatialrisk.mlmodels.linear_predictor import compile_linear_predictor

    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    df = ds.extract_at_points(None)  # the fixture's training frame

    # Re-derive the design with a 106-level categorical on the 'pa' raster
    # (values 0/1 lie inside the domain, so the frame is valid).
    levels = list(range(0, 106))
    formula = f"target + trial ~ scale(alt) + C(pa, levels={levels})"
    _, x = dmatrices(formula, df, NA_action="drop")
    model._x_design_info = x.design_info
    est = LogisticRegression(max_iter=50).fit(np.asarray(x), df["target"].to_numpy())
    model._ml_model = est
    assert len(x.design_info.column_names) == 107

    captured = {}

    def fake_predict_windowed(target, features, block, out, **kw):
        captured.update(kw)
        return out

    monkeypatch.setattr(
        "spatialrisk.mlmodels.windowed_predict.predict_windowed", fake_predict_windowed
    )
    model.apply(tmp_path / "out" / "g.tif", ds, ds.mask_path, 0)

    expected = compile_linear_predictor(x.design_info, est.coef_[0]).working_set_columns
    assert captured["n_design_cols"] == expected
    # Far below the full one-hot-expanded width (107 columns).
    assert captured["n_design_cols"] < len(x.design_info.column_names)

    # Independent of the level count: a 2-level domain on the same factor
    # must charge the identical working set.
    formula_two_level = "target + trial ~ scale(alt) + C(pa, levels=[0, 1])"
    _, x_two_level = dmatrices(formula_two_level, df, NA_action="drop")
    est_two_level = LogisticRegression(max_iter=50).fit(
        np.asarray(x_two_level), df["target"].to_numpy()
    )
    two_level = compile_linear_predictor(
        x_two_level.design_info, est_two_level.coef_[0]
    ).working_set_columns
    assert captured["n_design_cols"] == two_level


def test_icar_apply_charges_materialised_columns_not_levels(tmp_path, monkeypatch):
    """The iCAR closure shares the GLM shape and must share the memory win."""
    from patsy import dmatrices

    from spatialrisk.mlmodels.linear_predictor import compile_linear_predictor

    ds = build_dataset(tmp_path)
    model = build_icar(tmp_path, ds)
    df = ds.extract_at_points(None)  # the fixture's training frame

    # Re-derive the design with a 106-level categorical on the 'pa' raster
    # (values 0/1 lie inside the domain, so the frame is valid).
    levels = list(range(0, 106))
    formula = f"target + trial ~ scale(alt) + C(pa, levels={levels})"
    _, x = dmatrices(formula, df, NA_action="drop")
    model._x_design_info = x.design_info
    # One entry beyond the design width: compile_linear_predictor tolerates
    # it (spec §4.1), though iCAR's fitted betas match the width in practice.
    betas = np.zeros(len(x.design_info.column_names) + 1)
    model._ml_model["betas"] = betas
    assert len(x.design_info.column_names) == 107

    captured = {}

    def fake_predict_windowed(target, features, block, out, **kw):
        captured.update(kw)
        return out

    monkeypatch.setattr(
        "spatialrisk.mlmodels.windowed_predict.predict_windowed", fake_predict_windowed
    )
    model.apply(tmp_path / "out" / "i.tif", ds, ds.mask_path, 0)

    expected = compile_linear_predictor(x.design_info, betas).working_set_columns
    assert captured["n_design_cols"] == expected
    # Far below the full one-hot-expanded width (107 columns).
    assert captured["n_design_cols"] < len(x.design_info.column_names)

    # Independent of the level count: a 2-level domain on the same factor
    # must charge the identical working set.
    formula_two_level = "target + trial ~ scale(alt) + C(pa, levels=[0, 1])"
    _, x_two_level = dmatrices(formula_two_level, df, NA_action="drop")
    betas_two_level = np.zeros(len(x_two_level.design_info.column_names) + 1)
    two_level = compile_linear_predictor(
        x_two_level.design_info, betas_two_level
    ).working_set_columns
    assert captured["n_design_cols"] == two_level


def test_glm_apply_rejects_coefficients_off_the_design_width(tmp_path):
    """A GLM whose coefficients do not match its design raises before predicting.

    ``predict_proba``'s feature-count check used to catch a model fitted on
    another design; the compiled predictor ignores trailing coefficients, so
    ``apply`` checks the width itself rather than shift them onto the wrong
    columns.
    """
    ds = build_dataset(tmp_path)
    model = build_glm(tmp_path, ds)
    estimator = model._ml_model
    estimator.coef_ = np.hstack([estimator.coef_, [[0.5]]])
    out = tmp_path / "out" / "g.tif"
    with pytest.raises(ValueError, match="4 coefficients but its design has 3"):
        model.apply(out, ds, ds.mask_path, 0)
    assert not out.exists()
