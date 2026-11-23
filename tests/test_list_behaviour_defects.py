"""Stage 1 Task 6: behaviour defects found while standardizing the lists."""

import inspect
from types import SimpleNamespace

import reacton
import solara
from _notification_host import render_under_notifications


def _capture(monkeypatch, module, name):
    """Replace ``module.name`` with a stub that records the props it got."""
    seen = []

    @solara.component
    def _Stub(**kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(module, name, _Stub)
    return seen


def _named(type_name, **attrs):
    return type(type_name, (), attrs)()


def test_running_model_row_has_no_cancel(monkeypatch):
    """A running training row offers no (fake) cancel action."""
    import gui.widget.train_model_list as mod

    assert "on_cancel" not in inspect.signature(mod.TrainModelList.f).parameters
    seen = _capture(monkeypatch, mod, "ProductTable")
    monkeypatch.setattr(
        mod,
        "train_rows",
        lambda *a, **k: [
            {
                "kind": "job",
                "key": "j1",
                "job_id": "j1",
                "name": "m",
                "model_label": "GLM",
                "dataset_name": "d",
                "sample_name": "s",
                "status": "running",
            }
        ],
    )
    box, rc = reacton.render(
        mod.TrainModelList(
            project=solara.reactive(SimpleNamespace()),
            train_jobs=solara.reactive([]),
            model_labels={},
            on_dismiss=lambda *_: None,
            on_delete=lambda *_: None,
            on_open=lambda *_: None,
        ),
        handle_error=False,
    )
    try:
        assert seen[-1]["rows"][0]["actions"] == []
    finally:
        rc.close()


def test_evaluation_delete_asks_first(monkeypatch):
    """Evaluation delete opens a ConfirmDialog; nothing goes before confirm."""
    import gui.tile.evaluation_tile as et

    results = _capture(monkeypatch, et, "EvaluationResults")
    confirms = _capture(monkeypatch, et, "ConfirmDialog")
    _capture(monkeypatch, et, "EvaluationTableDialog")
    _capture(monkeypatch, et, "EvaluationFormDialog")
    monkeypatch.setattr(et, "variable_items", lambda p: [1])
    monkeypatch.setattr(et, "map_items", lambda p: [1])
    deleted = []
    monkeypatch.setattr(
        et, "delete_evaluation_run", lambda p, k: (deleted.append(k), (True, None))[1]
    )

    class _P:
        def model_copy(self):
            return self

    project = solara.reactive(_P())
    box, rc = render_under_notifications(lambda: et.EvaluationTile(project=project))
    try:
        assert confirms[-1]["open"] is False
        results[-1]["on_delete"]("run1")
        assert deleted == []  # nothing deleted on click
        assert confirms[-1]["open"] is True
        confirms[-1]["on_confirm"]()
        assert deleted == ["run1"]
        assert confirms[-1]["open"] is False
    finally:
        rc.close()


def test_error_alerts_use_the_type_prop():
    """Both alerts use ``type=`` (``type_=`` renders them without red)."""
    import gui.tile.inference_tile as it
    import gui.widget.variable_modal as vm

    for mod in (it, vm):
        src = inspect.getsource(mod)
        assert 'rv.Alert(type="error"' in src
        assert 'type_="error"' not in src


def test_toolbox_new_button_sits_above_the_list():
    """Toolbox New sits above the list like every other tab."""
    from gui.tile import toolbox_tile

    src = inspect.getsource(toolbox_tile)
    assert src.index("toolbox.allocation.new") < src.index("AllocationList(")
    assert 'color="primary"' in src[src.index("toolbox.allocation.new") :][:300]


def _row_actions(monkeypatch, mod, element, ptable="ProductTable"):
    seen = _capture(monkeypatch, mod, ptable)
    box, rc = reacton.render(element, handle_error=False)
    rc.close()
    return {r["key"]: {a["kind"]: a for a in r["actions"]} for r in seen[-1]["rows"]}


def test_variable_edit_and_delete_freeze_while_downloading(monkeypatch):
    """Edit and delete freeze on a downloading row only."""
    import gui.widget.variable_list as vl

    p = SimpleNamespace(
        raw_variables={
            "a": _named("GEEVar", name="a", data_type="raster", year=None),
            "b": _named("GEEVar", name="b", data_type="raster", year=None),
        },
        base_raster=None,
    )
    acts = _row_actions(
        monkeypatch,
        vl,
        vl.SourceVariableList(
            project=solara.reactive(p),
            on_remove=lambda k: None,
            on_edit=lambda k: None,
            on_download=lambda k: None,
            downloading_keys=frozenset({"a"}),
        ),
    )
    assert acts["a"]["edit"].get("disabled") is True
    assert acts["a"]["delete"].get("disabled") is True
    assert not acts["b"]["edit"].get("disabled")
    assert not acts["b"]["delete"].get("disabled")


def test_harmonization_delete_freezes_while_its_row_runs(monkeypatch):
    """Harmonization delete freezes on the running row only."""
    import gui.widget.variable_list as vl
    from spatialrisk.harmonization import HarmonizationStatus, output_key

    def _r(name):
        return _named(
            "LocalRasterVar",
            name=name,
            data_type="raster",
            year=None,
            path=f"/x/{name}.tif",
        )

    raw = {"a": _r("a"), "b": _r("b")}
    p = SimpleNamespace(
        raw_variables=raw,
        processed_variables={output_key(v): _r(v.name) for v in raw.values()},
        base_raster=None,
    )
    status = HarmonizationStatus(pending=[], current=list(raw))
    acts = _row_actions(
        monkeypatch,
        vl,
        vl.HarmonizationVariableList(
            project=solara.reactive(p),
            status=status,
            on_harmonize=lambda k: None,
            running_keys={"a"},
            on_remove=lambda k: None,
        ),
    )
    assert acts["a"]["delete"].get("disabled") is True
    assert not acts["b"]["delete"].get("disabled")


def test_dataset_row_details_opens_the_dialog(monkeypatch):
    """The dataset open action hands the key to on_open."""
    import gui.tile.dataset_tile as dt
    import gui.widget.dataset_list as dl

    ds = SimpleNamespace(
        target=_named("V", name="loss"), features=[], year=2020, name="d"
    )
    p = SimpleNamespace(datasets={"d": ds})
    opened = []
    acts = _row_actions(
        monkeypatch,
        dl,
        dl.DatasetList(
            project=solara.reactive(p), on_open=opened.append, on_edit=lambda k: None
        ),
    )
    assert "open" in acts["d"]
    acts["d"]["open"]["on_click"]()
    assert opened == ["d"]
    # no on_open -> no details action
    acts = _row_actions(monkeypatch, dl, dl.DatasetList(project=solara.reactive(p)))
    assert "open" not in acts["d"]
    # the tile wires the dialog
    src = inspect.getsource(dt)
    assert "DatasetDetailsDialog" in src and "on_open=" in src


def test_dataset_details_dialog_lists_fields_read_only(monkeypatch):
    """The details dialog shows target, features, year, categoricals."""
    import gui.widget.dataset_details_dialog as dd

    seen = []
    monkeypatch.setattr(
        dd, "ro_field", lambda label, value: seen.append((label, value))
    )
    feat = _named("V", name="slope", raster_type="continuous")
    cat = _named("V", name="soil", raster_type="categorical")
    ds = SimpleNamespace(
        target=_named("V", name="loss"), features=[feat, cat], year=2020, name="d"
    )
    closed = []
    box, rc = reacton.render(
        dd.DatasetDetailsDialog(
            dataset=ds, open=True, on_close=lambda: closed.append(1)
        ),
        handle_error=False,
    )
    rc.close()
    values = [v for _, v in seen]
    assert "loss" in values and 2020 in values
    assert any("slope" in str(v) and "soil" in str(v) for v in values)
    assert ["soil"] in values  # categorical features
