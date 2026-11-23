"""Row builders: registry products + session-job overlay with suppression."""

import types

from gui.scripts.product_rows import (
    evaluation_tab_rows,
    inference_rows,
    job_row_key,
    prediction_groups,
    prediction_row_key,
    train_rows,
)


def _proj(models=None, predictions=None, evaluations=None):
    return types.SimpleNamespace(
        models=models or {}, predictions=predictions or {}, evaluations=evaluations or {}
    )


def _model(**kw):
    base = dict(model_type="glm", dataset_name="ds1", sample_name="s1")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _pred(name="run_a", model_key="glm_v1", dataset_name="ds1"):
    return types.SimpleNamespace(name=name, model_key=model_key, dataset_name=dataset_name)


def _rec(run_id="j1", truth_tag="gfc", created_at="2026-07-09T10:00:00", keys=("p1",)):
    return types.SimpleNamespace(
        run_id=run_id, truth_tag=truth_tag, created_at=created_at,
        prediction_keys=list(keys),
    )


# --- train ---

def test_train_rows_products_only():
    p = _proj(models={"glm_v1": _model()})
    rows = train_rows(p, [], {"glm": "GLM"})
    assert [r["kind"] for r in rows] == ["model"]
    assert rows[0]["key"] == "glm_v1"
    assert rows[0]["name"] == "glm_v1"
    assert rows[0]["model_label"] == "GLM"
    assert rows[0]["status"] == "ready"
    assert rows[0]["sample_name"] == "s1"


def test_train_rows_sample_name_defaults_to_dash():
    # Non-sampling families (mw, benchmark) have no sample; jobs may omit it.
    p = _proj(models={"mw_v1": _model(model_type="mw", sample_name=None)})
    jobs = [{"id": "a1", "status": "running", "model_name": "v2",
             "model_label": "GLM", "dataset_name": "ds1", "error": None}]
    rows = train_rows(p, jobs)
    assert rows[0]["sample_name"] == "—"   # job without sample_name
    assert rows[1]["sample_name"] == "—"   # model with sample_name=None


def test_train_rows_running_job_listed_after_products():
    p = _proj(models={"glm_v1": _model()})
    jobs = [{"id": "a1", "status": "running", "model_name": "v2",
             "model_label": "GLM", "dataset_name": "ds1", "sample_name": "s1",
             "error": None}]
    rows = train_rows(p, jobs)
    assert [r["kind"] for r in rows] == ["model", "job"]
    assert rows[1]["sample_name"] == "s1"
    assert rows[1]["status"] == "running"
    assert rows[1]["job_id"] == "a1"


def test_train_rows_completed_job_suppressed_when_registered():
    p = _proj(models={"glm_v2": _model()})
    jobs = [{"id": "a1", "status": "completed", "model_storage_key": "glm_v2",
             "model_name": "v2", "model_label": "GLM", "dataset_name": "ds1",
             "error": None, "deviance": 1.0, "n_samples": 10}]
    rows = train_rows(p, jobs)
    assert [r["kind"] for r in rows] == ["model"]


def test_train_rows_completed_job_kept_when_registration_missing():
    jobs = [{"id": "a1", "status": "completed", "model_storage_key": "glm_v2",
             "model_name": "v2", "model_label": "GLM", "dataset_name": "ds1",
             "error": None, "deviance": 1.0, "n_samples": 10}]
    rows = train_rows(_proj(), jobs)
    assert [r["kind"] for r in rows] == ["job"]
    assert rows[0]["status"] == "completed"


def test_train_rows_none_project():
    assert train_rows(None, []) == []


# --- inference ---

def test_prediction_row_key_prefers_name_falls_back_to_provenance():
    assert prediction_row_key(_pred(name="run_a")) == "run_a"
    assert prediction_row_key(_pred(name=None)) == "glm_v1__ds1"


def test_prediction_groups_multiwindow_share_one_row():
    p = _proj(predictions={
        "run_b_w5": _pred(name="run_b"), "run_b_w11": _pred(name="run_b"),
    })
    groups = prediction_groups(p)
    assert set(groups) == {"run_b"}
    assert sorted(groups["run_b"]["storage_keys"]) == ["run_b_w11", "run_b_w5"]


def test_inference_rows_named_runs_stay_distinct_for_same_provenance():
    # The exact case build_inference_jobs collapsed: two named runs, one combo.
    p = _proj(predictions={
        "run_a": _pred(name="run_a"), "run_b": _pred(name="run_b"),
    })
    rows = inference_rows(p, [])
    assert sorted(r["key"] for r in rows) == ["run_a", "run_b"]


def test_inference_rows_completed_job_suppressed_by_row_key():
    p = _proj(predictions={"run_a": _pred(name="run_a")})
    jobs = [{"id": "a1", "status": "completed", "pred_name": "run_a",
             "model_key": "glm_v1", "dataset_name": "ds1", "error": None}]
    assert [r["kind"] for r in inference_rows(p, jobs)] == ["prediction"]


def test_inference_rows_failed_job_kept_with_error():
    jobs = [{"id": "a1", "status": "failed", "pred_name": "run_x",
             "model_key": "glm_v1", "dataset_name": "ds1", "error": "boom"}]
    rows = inference_rows(_proj(), jobs)
    assert rows[0]["kind"] == "job" and rows[0]["error"] == "boom"


def test_job_row_key_fallback():
    assert job_row_key({"pred_name": "run_a"}) == "run_a"
    assert job_row_key({"model_key": "m", "dataset_name": "d"}) == "m__d"


# --- evaluation ---

def test_evaluation_rows_sorted_oldest_first_and_suppressed_by_run_id():
    p = _proj(evaluations={
        "old": _rec(run_id="j0", created_at="2026-07-01T10:00:00"),
        "new": _rec(run_id="j1", created_at="2026-07-09T10:00:00"),
    })
    jobs = [
        {"id": "j1", "status": "completed", "truth_tag": "gfc",
         "n_maps": 1, "created_at": "2026-07-09T10:00:00", "error": None},
        {"id": "j2", "status": "running", "truth_tag": "gfc",
         "n_maps": 2, "created_at": "2026-07-09T11:00:00", "error": None},
    ]
    rows = evaluation_tab_rows(p, jobs)
    assert [r["kind"] for r in rows] == ["evaluation", "evaluation", "job"]
    assert rows[2]["job_id"] == "j2"
    assert [r["key"] for r in rows[:2]] == ["old", "new"]


def test_evaluation_rows_are_oldest_first():
    p = _proj(evaluations={
        "b": _rec(run_id="r2", created_at="2026-07-09T10:00:00"),
        "a": _rec(run_id="r1", created_at="2026-07-01T10:00:00"),
    })
    assert [r["key"] for r in evaluation_tab_rows(p, [])] == ["a", "b"]


def test_completed_job_keeps_its_position():
    """When a job completes its product takes the slot, not the top of the list."""
    models = {"m1": _model(), "m2": _model()}

    def job(job_id, **kw):
        base = {"id": job_id, "status": "running", "model_name": job_id,
                "model_label": "GLM", "dataset_name": "ds1", "error": None}
        base.update(kw)
        return base

    jobs = [job("j3"), job("j4")]
    rows = train_rows(_proj(models=models), jobs)
    assert [r["key"] for r in rows] == ["m1", "m2", "job_j3", "job_j4"]

    models["m3"] = _model()
    jobs[0] = job("j3", status="completed", model_storage_key="m3")
    rows = train_rows(_proj(models=models), jobs)
    assert [r["key"] for r in rows] == ["m1", "m2", "m3", "job_j4"]
