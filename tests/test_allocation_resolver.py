"""Rate-table resolution per model family."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gui.scripts.allocation_runner import (
    AllocationResolveError,
    preview_defrate_source,
    resolve_defrate_table,
)


def _pred(**kw):
    """Stand-in for a registered Prediction, with the fields the resolver reads."""
    base = dict(
        path=Path("/data/p/inference/forecast/prob.tif"),
        model_key="icar",
        dataset_name="forecast",
        window=None,
        defrate_path=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _project(predictions):
    """Minimal project stand-in holding a predictions registry."""
    return SimpleNamespace(predictions=predictions, models={}, datasets={})


def test_persisted_defrate_path_wins(tmp_path):
    """A table recorded on the Prediction is used as-is."""
    csv = tmp_path / "defrate_cat_bm_forecast.csv"
    csv.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    project = _project({"jnr_run": _pred(model_key="jnr", defrate_path=csv)})

    src = resolve_defrate_table(project, "jnr_run")

    assert src.path == csv
    assert src.provenance == "persisted"


def test_jnr_without_model_table_carries_a_caveat(tmp_path):
    """JNR tables hold observed-period rates, so the user is warned."""
    csv = tmp_path / "defrate_cat_bm_forecast.csv"
    csv.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    project = _project({"jnr_run": _pred(model_key="jnr", defrate_path=csv)})

    src = resolve_defrate_table(project, "jnr_run")

    assert src.caveat is not None
    assert "observed" in src.caveat.lower()


def test_mw_falls_back_to_sibling_path(tmp_path):
    """Moving-window runs find their table beside the raster."""
    prob = tmp_path / "prob_mw_11_forecast.tif"
    prob.write_bytes(b"")
    sibling = tmp_path / "defrate_cat_mw_11_forecast.csv"
    sibling.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    project = _project({"mw_run_w11": _pred(model_key="mw", window=11, path=prob)})

    src = resolve_defrate_table(project, "mw_run_w11")

    assert src.path == sibling
    assert src.provenance == "mw-sibling"


def test_mw_without_sibling_reports_missing(tmp_path):
    """A moving-window run with no table fails with an actionable message."""
    prob = tmp_path / "prob_mw_11_forecast.tif"
    prob.write_bytes(b"")
    project = _project({"mw_run_w11": _pred(model_key="mw", window=11, path=prob)})

    with pytest.raises(AllocationResolveError, match="rate table"):
        resolve_defrate_table(project, "mw_run_w11", compute=False)


def test_far_prediction_computes_via_resolve_layers(tmp_path, monkeypatch):
    """FAR-family runs compute the table from the prediction's own dataset."""
    import gui.scripts.allocation_runner as runner

    out = tmp_path / "defrate_cat_icar_forecast.csv"
    calls = {}

    def fake_resolve_layers(project, pred):
        return {
            "defor_file": "/d.tif",
            "forest_file": "/f.tif",
            "riskmap_file": str(pred.path),
            "time_interval": 5,
            "period": "forecast",
        }

    def fake_defrate_per_cat(**kwargs):
        calls.update(kwargs)
        Path(kwargs["tab_file_defrate"]).write_text(
            "cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n"
        )

    monkeypatch.setattr(runner, "_resolve_layers", fake_resolve_layers)
    monkeypatch.setattr(runner, "_defrate_per_cat", fake_defrate_per_cat)
    pred = _pred(path=tmp_path / "prob.tif")
    project = _project({"icar_run": pred})

    src = resolve_defrate_table(project, "icar_run")

    assert src.provenance == "computed"
    assert calls["time_interval"] == 5
    assert calls["defor_file"] == "/d.tif"
    assert Path(src.path).exists()
    assert out.exists()


def test_missing_time_interval_is_an_explicit_error(tmp_path, monkeypatch):
    """Without a period length the rate cannot be computed — say so."""
    import gui.scripts.allocation_runner as runner

    monkeypatch.setattr(
        runner,
        "_resolve_layers",
        lambda project, pred: {
            "defor_file": "/d.tif",
            "forest_file": "/f.tif",
            "riskmap_file": "/r.tif",
            "time_interval": None,
            "period": "forecast",
        },
    )
    project = _project({"icar_run": _pred(path=tmp_path / "prob.tif")})

    with pytest.raises(AllocationResolveError, match="period length"):
        resolve_defrate_table(project, "icar_run")


def test_user_path_short_circuits_everything(tmp_path):
    """An explicit override skips every other resolution route."""
    csv = tmp_path / "mine.csv"
    csv.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    project = _project({"icar_run": _pred()})

    src = resolve_defrate_table(project, "icar_run", user_path=csv)

    assert src.path == csv
    assert src.provenance == "user"


def test_unknown_prediction_key_raises():
    """Resolving against a missing registry key fails loudly."""
    with pytest.raises(AllocationResolveError, match="not found"):
        resolve_defrate_table(_project({}), "nope")


def test_named_jnr_prediction_is_recognized_by_family(tmp_path):
    """Family detection is by key prefix, not equality.

    A *named* benchmark model produces predictions keyed 'jnr_<name>' and
    must still get the JNR caveat.
    """
    csv = tmp_path / "defrate_cat_bm_forecast.csv"
    csv.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    project = _project({"jnr_run": _pred(model_key="jnr_v1", defrate_path=csv)})

    src = resolve_defrate_table(project, "jnr_run")

    assert src.caveat is not None


def test_named_mw_prediction_finds_its_sibling_table(tmp_path):
    """Same for MW: 'mw_<name>' predictions resolve like plain 'mw' ones."""
    prob = tmp_path / "prob_mw_11_forecast.tif"
    prob.write_bytes(b"")
    sibling = tmp_path / "defrate_cat_mw_11_forecast.csv"
    sibling.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    project = _project({"mw_run_w11": _pred(model_key="mw_v2", window=11, path=prob)})

    src = resolve_defrate_table(project, "mw_run_w11")

    assert src.path == sibling
    assert src.provenance == "mw-sibling"


def _import_pred(tmp_path, key="my-map"):
    """Stand-in for a registered imported Prediction (dataset_name='imported')."""
    return _pred(
        path=tmp_path / "imported_predictions" / f"{key}.tif",
        model_key=key,
        dataset_name="imported",
    )


def _eval_record(pred_key, created_at, run_id, truth_tag="truth", defrate_csv=None):
    """Stand-in for a saved EvaluationRunRecord scoring *pred_key*."""
    art = SimpleNamespace(prediction_key=pred_key, defrate_csv=defrate_csv)
    return SimpleNamespace(
        prediction_keys=[pred_key],
        created_at=created_at,
        run_id=run_id,
        truth_tag=truth_tag,
        artifacts=[art],
    )


def _project_with_evals(tmp_path, pred, evaluations):
    """Minimal project stand-in with a predictions registry and evaluations."""
    return SimpleNamespace(
        predictions={"my-map__imported": pred},
        models={},
        datasets={},
        evaluations=evaluations,
        folders=SimpleNamespace(project_folder=tmp_path),
    )


def test_import_resolves_to_evaluation_defrate(tmp_path):
    """An imported prediction picks up the rate table from its evaluation."""
    csv = tmp_path / "defrate.csv"
    csv.write_text("cat,nfor,rate_mod,pixel_area\n1,10,0.0,0.09\n")
    pred = _import_pred(tmp_path)
    project = _project_with_evals(
        tmp_path,
        pred,
        {
            "r1": _eval_record(
                "my-map__imported",
                "2026-09-01T10:00:00",
                "aaaa1111",
                defrate_csv=str(csv),
            )
        },
    )

    src = resolve_defrate_table(project, "my-map__imported")

    assert src.path == csv
    assert src.provenance == "evaluation"
    assert "truth" in src.caveat


def test_import_newest_evaluation_wins(tmp_path):
    """When several evaluations scored the import, the newest one's table is used."""
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    old.write_text("x\n")
    new.write_text("x\n")
    pred = _import_pred(tmp_path)
    project = _project_with_evals(
        tmp_path,
        pred,
        {
            "r1": _eval_record(
                "my-map__imported", "2026-09-01T10:00:00", "a", defrate_csv=str(old)
            ),
            "r2": _eval_record(
                "my-map__imported", "2026-09-02T10:00:00", "b", defrate_csv=str(new)
            ),
        },
    )
    assert resolve_defrate_table(project, "my-map__imported").path == new


def test_import_falls_back_to_older_evaluation_when_newest_file_is_gone(tmp_path):
    """A newest record whose CSV vanished is skipped in favour of an older one."""
    old = tmp_path / "old.csv"
    old.write_text("x\n")
    pred = _import_pred(tmp_path)
    project = _project_with_evals(
        tmp_path,
        pred,
        {
            "r1": _eval_record(
                "my-map__imported", "2026-09-01T10:00:00", "a", defrate_csv=str(old)
            ),
            "r2": _eval_record(
                "my-map__imported",
                "2026-09-02T10:00:00",
                "b",
                defrate_csv=str(tmp_path / "missing.csv"),
            ),
        },
    )
    assert resolve_defrate_table(project, "my-map__imported").path == old


def test_import_legacy_record_derives_run_scoped_path(tmp_path):
    """Records saved before defrate_csv existed: derive evaluation/<truth>/<run>/.

    ``artifact_label_for`` combines ``label_for`` (from ``model_key``) with the
    run label (``name`` or, absent one, ``model_key`` again) — for a plain
    "my-map" stand-in with no ``name`` set, that doubles to "my-map_my-map",
    which is what the real evaluation write path (Task 4) also produces for
    such a key. The filename below matches that, not a single "my-map".
    """
    pred = _import_pred(tmp_path)
    derived = (
        tmp_path
        / "evaluation"
        / "truth"
        / "abcd1234"
        / "defrate_cat_my-map_my-map_imported.csv"
    )
    derived.parent.mkdir(parents=True)
    derived.write_text("x\n")
    project = _project_with_evals(
        tmp_path,
        pred,
        {"r1": _eval_record("my-map__imported", "2026-09-01T10:00:00", "abcd1234")},
    )
    assert resolve_defrate_table(project, "my-map__imported").path == derived


def test_import_without_evaluation_raises_actionable_error(tmp_path):
    """No evaluation at all means an explicit, actionable resolver error."""
    pred = _import_pred(tmp_path)
    project = _project_with_evals(tmp_path, pred, {})
    with pytest.raises(
        AllocationResolveError, match="Evaluate this imported map first"
    ):
        resolve_defrate_table(project, "my-map__imported")


def test_import_named_like_mw_never_enters_mw_branch(tmp_path):
    """An import named 'mw_export' must not be treated as a moving-window run."""
    pred = _import_pred(tmp_path, key="mw_export")
    project = SimpleNamespace(
        predictions={"mw_export__imported": pred},
        models={},
        datasets={},
        evaluations={},
        folders=SimpleNamespace(project_folder=tmp_path),
    )
    with pytest.raises(
        AllocationResolveError, match="Evaluate this imported map first"
    ):
        resolve_defrate_table(project, "mw_export__imported")


def test_preview_import_mirrors_resolver(tmp_path):
    """The form preview matches the resolver for imports, with and without evals."""
    pred = _import_pred(tmp_path)
    csv = tmp_path / "defrate.csv"
    csv.write_text("x\n")
    with_eval = _project_with_evals(
        tmp_path,
        pred,
        {
            "r1": _eval_record(
                "my-map__imported", "2026-09-01T10:00:00", "a", defrate_csv=str(csv)
            )
        },
    )
    src = preview_defrate_source(with_eval, "my-map__imported")
    assert src.provenance == "evaluation" and src.path == csv

    without = _project_with_evals(tmp_path, pred, {})
    src = preview_defrate_source(without, "my-map__imported")
    assert src.provenance == "unavailable"
    assert "Evaluate this imported map first" in src.caveat
