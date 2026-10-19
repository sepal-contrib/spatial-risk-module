"""Every list shares one row anatomy: name first, <=2 meta columns, status last."""

from datetime import datetime
from types import SimpleNamespace

import pytest
import reacton
import solara

from gui.i18n import t

# Warm the translator before the first render (see test_model_form_dialog_render).
t("common.cancel")

import gui.widget.allocation_list as allocation_list  # noqa: E402
import gui.widget.dataset_list as dataset_list  # noqa: E402
import gui.widget.evaluation_results as evaluation_results  # noqa: E402
import gui.widget.inference_output_list as inference_output_list  # noqa: E402
import gui.widget.sample_set_list as sample_set_list  # noqa: E402
import gui.widget.train_model_list as train_model_list  # noqa: E402
import gui.widget.variable_list as variable_list  # noqa: E402
from gui.scripts.product_rows import relative_date  # noqa: E402
from gui.widget.product_table import W_NAME, W_STATUS  # noqa: E402


class GEEVar:
    """Name-matched stand-in: the lists test ``type(var).__name__ == 'GEEVar'``."""

    def __init__(self, name, year=None):  # noqa: D107
        self.name = name
        self.year = year
        self.data_type = "raster"


class _Local:
    def __init__(self, name, year=None):
        self.name = name
        self.year = year
        self.data_type = "raster"


def _project(**kw):
    base = dict(
        raw_variables={},
        processed_variables={},
        datasets={},
        samples={},
        models={},
        predictions={},
        evaluations={},
        base_raster=None,
    )
    base.update(kw)
    return solara.reactive(SimpleNamespace(**base))


def _noop(*_):
    return None


def _source():
    p = _project(
        raw_variables={"a": _Local("a", 2010), "b": GEEVar("b")},
    )
    return (
        variable_list,
        variable_list.SourceVariableList,
        dict(project=p, on_remove=_noop, on_download=_noop),
    )


def _harmonization():
    from spatialrisk.harmonization import is_harmonizable  # noqa: F401

    p = _project(raw_variables={})
    return (
        variable_list,
        variable_list.HarmonizationVariableList,
        dict(project=p, status=None, on_harmonize=_noop),
    )


def _derived():
    var = SimpleNamespace(name="loss_a", year=2011, data_type="raster")
    p = _project(
        processed_variables={"loss_a_2011": var},
        raw_variables={"a": _Local("loss_a")},
    )
    jobs = solara.reactive(
        [{"id": "j1", "name": "dist_a", "output_key": "x", "status": "running"}]
    )
    return (
        variable_list,
        variable_list.DerivedVariableList,
        dict(project=p, jobs=jobs, on_remove=_noop, on_dismiss=_noop),
    )


def _datasets():
    ds = SimpleNamespace(
        target=SimpleNamespace(name="far"), features=["a", "b"], year=2020
    )
    return (
        dataset_list,
        dataset_list.DatasetList,
        dict(project=_project(datasets={"ds": ds}), on_remove=_noop),
    )


def _samples():
    s = SimpleNamespace(strategy="random", allocation=None, n_total=10, class_counts={})
    jobs = solara.reactive(
        [{"id": "j1", "name": "sj", "strategy": "random", "status": "running"}]
    )
    return (
        sample_set_list,
        sample_set_list.SampleSetList,
        dict(project=_project(samples={"s": s}), sampling_jobs=jobs, on_remove=_noop),
    )


def _models():
    m = SimpleNamespace(model_type="rf", dataset_name="ds", sample_name="s")
    jobs = solara.reactive([{"id": "j1", "model_name": "mj", "status": "running"}])
    return (
        train_model_list,
        train_model_list.TrainModelList,
        dict(
            project=_project(models={"m": m}),
            train_jobs=jobs,
            model_labels={"rf": "RF"},
            on_dismiss=_noop,
            on_delete=_noop,
            on_open=_noop,
        ),
    )


def _predictions():
    pr = SimpleNamespace(name="pr", model_key="m", dataset_name="ds")
    jobs = solara.reactive([{"id": "j1", "pred_name": "pj", "status": "running"}])
    return (
        inference_output_list,
        inference_output_list.InferenceOutputList,
        dict(project=_project(predictions={"pr": pr}), inference_jobs=jobs),
    )


def _evaluations():
    rec = SimpleNamespace(
        truth_tag="t", prediction_keys=["a", "b"], created_at="2020-10-01T10:00:00"
    )
    jobs = solara.reactive([{"id": "j1", "truth_tag": "tj", "status": "running"}])
    return (
        evaluation_results,
        evaluation_results.EvaluationResults,
        dict(
            eval_jobs=jobs,
            project=_project(evaluations={"e": rec}),
            on_open=_noop,
            on_delete=_noop,
        ),
    )


def _allocations():
    rows = [
        {"kind": "job", "key": "j1", "job_id": "j1", "name": "aj", "status": "running"},
        {
            "kind": "record",
            "key": "r",
            "name": "r",
            "source": "x",
            "created_at": "2026-10-01",
            "annual_ha": 10.0,
            "total_ha": 50.0,
            "years_forecast": 5,
            "density_map_path": None,
            "warnings": [],
        },
    ]
    return (
        allocation_list,
        allocation_list.AllocationList,
        dict(rows=rows, on_delete=_noop),
    )


ALL_LISTS = [
    _source,
    _datasets,
    _derived,
    _samples,
    _models,
    _predictions,
    _evaluations,
    _allocations,
]


def _capture(factory):
    mod, component, kwargs = factory()
    seen = {}
    original = mod.ProductTable

    def _cap(**kw):
        seen.update(kw)
        return original(**kw)

    mod.ProductTable = _cap
    try:
        box, rc = reacton.render(component(**kwargs), handle_error=False)
        rc.close()
    finally:
        mod.ProductTable = original
    return seen


@pytest.mark.parametrize("factory", ALL_LISTS)
def test_every_list_has_name_first_status_last_and_at_most_two_meta(factory):
    """Every list has name first status last and at most two meta."""
    widths = [c["width"] for c in _capture(factory)["columns"]]
    assert widths[0] == W_NAME
    assert widths[-1] == W_STATUS
    assert len(widths) <= 4
    assert all(w.endswith("px") for w in widths[1:])


@pytest.mark.parametrize("factory", ALL_LISTS)
def test_every_row_has_every_column(factory):
    """Every row has every column."""
    seen = _capture(factory)
    assert seen["rows"], "fixture must produce rows"
    n = len(seen["columns"])
    for r in seen["rows"]:
        assert len(r["cells"]) == n, r["key"]
    assert seen["rows"][-1]["cells"][-1]["type"] == "status"


def test_harmonization_columns_are_name_chip_year_status():
    """Harmonization columns are name chip year status."""
    from gui.widget.product_table import W_CHIP, W_YEAR

    mod, component, kwargs = _harmonization()
    seen = {}
    original = mod.ProductTable

    def _cap(**kw):
        seen.update(kw)
        return original(**kw)

    mod.ProductTable = _cap
    try:
        _, rc = reacton.render(component(**kwargs), handle_error=False)
        rc.close()
    finally:
        mod.ProductTable = original
    assert [c["width"] for c in seen["columns"]] == [W_NAME, W_CHIP, W_YEAR, W_STATUS]


def test_source_status_is_pending_for_cloud_ready_for_local():
    """Source status is pending for cloud ready for local."""
    rows = _capture(_source)["rows"]
    by_key = {r["key"]: r["cells"][-1]["status"] for r in rows}
    assert by_key == {"a": "ready", "b": "pending"}


def test_source_running_status_while_downloading():
    """Source running status while downloading."""
    mod, component, kwargs = _source()
    kwargs["downloading_keys"] = frozenset({"b"})
    seen = {}
    original = mod.ProductTable
    mod.ProductTable = lambda **kw: (seen.update(kw), original(**kw))[1]
    try:
        _, rc = reacton.render(component(**kwargs), handle_error=False)
        rc.close()
    finally:
        mod.ProductTable = original
    assert {r["key"]: r["cells"][-1]["status"] for r in seen["rows"]}["b"] == "running"


def test_source_subtitle_carries_type_and_origin():
    """Source subtitle carries type and origin."""
    rows = _capture(_source)["rows"]
    subs = {r["key"]: r["cells"][0]["subtitle"] for r in rows}
    assert subs["a"] == f"raster · {t('widgets.variable_list.chip_local')}"
    assert subs["b"] == f"raster · {t('widgets.variable_list.chip_cloud')}"


def test_dataset_subtitle_and_status_ready():
    """Dataset subtitle and status ready."""
    row = _capture(_datasets)["rows"][0]
    assert row["cells"][0]["subtitle"].startswith("far · ")
    assert row["cells"][-1]["status"] == "ready"


def test_allocation_total_moves_to_the_subtitle_and_status_is_last():
    """Allocation total moves to the subtitle and status is last."""
    rows = _capture(_allocations)["rows"]
    record = next(r for r in rows if r["key"] == "r")
    assert record["cells"][-1]["status"] == "ready"
    job = next(r for r in rows if r["key"] == "j1")
    assert job["cells"][1]["value"] == "—"
    assert job["cells"][-1]["status"] == "running"


def test_evaluation_subtitle_is_a_relative_date():
    """Evaluation subtitle is a relative date."""
    row = next(r for r in _capture(_evaluations)["rows"] if r["key"] == "e")
    assert row["cells"][0]["subtitle"] == "2020-10-01"  # > 30 days old


def test_relative_date_buckets():
    """Relative date buckets."""
    now = datetime(2026, 10, 19, 12, 0)
    assert relative_date("2026-10-19T11:59:40", now) == "just now"
    assert relative_date("2026-10-19T09:00:00", now) == "3 hours ago"
    assert relative_date("2026-10-19T11:30:00", now) == "30 minutes ago"
    assert relative_date("2026-10-16T12:00:00", now) == "3 days ago"
    assert relative_date("2026-08-01T12:00:00", now) == "2026-08-01"
    assert relative_date("—", now) == "—"


def test_empty_derived_list_still_shows_title_and_empty_text():
    """Empty derived list still shows title and empty text."""
    seen = _capture_empty_derived()
    assert seen["rows"] == []
    assert seen["empty_text"] == t("widgets.variable_list.derived_empty")
    assert seen["empty_text"] != "widgets.variable_list.derived_empty"
    assert seen["title"]


def _capture_empty_derived():
    mod = variable_list
    seen = {}
    original = mod.ProductTable
    mod.ProductTable = lambda **kw: (seen.update(kw), original(**kw))[1]
    try:
        _, rc = reacton.render(
            mod.DerivedVariableList(project=_project(), jobs=solara.reactive([])),
            handle_error=False,
        )
        rc.close()
    finally:
        mod.ProductTable = original
    return seen
