"""DerivedVariableList: a submitted layer is a row before it is a product.

Derived layers used to appear in the list only once processing had finished —
submit the form and nothing happened for minutes. The list now renders the
session job overlay (the Train/Sampling/Inference pattern) so the row shows up
on submit with a running status, and turns into the real registry row when the
raster lands.
"""

import reacton
import solara

from gui.i18n import t

# Warm the translator before the first render (see test_model_form_dialog_render).
t("common.cancel")

from gui.widget.variable_list import DerivedVariableList  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.variables.local_raster_var import LocalRasterVar  # noqa: E402


def _project_with_one_derived():
    p = Project(project_name="p")
    p.processed_variables["loss_forest_2010_2015"] = LocalRasterVar.model_construct(
        name="loss_forest_2010_2015",
        year=None,
        data_type="raster",
        raster_type="categorical",
        path=None,
        project=p,
    )
    return solara.reactive(p)


def _empty_project():
    return solara.reactive(Project(project_name="p"))


def _job(job_id="j1", name="forest_dist", status="running", error=None):
    return {
        "id": job_id,
        "name": name,
        "output_key": name,
        "status": status,
        "error": error,
    }


def _captured_rows(project, jobs=None, **overrides):
    """The row specs DerivedVariableList hands to ProductTable."""
    import gui.widget.variable_list as mod

    props = {
        "project": project,
        "jobs": solara.reactive(jobs or []),
        "on_toggle_map": lambda k: None,
        "on_remove": lambda k: None,
        "on_dismiss": lambda i: None,
    }
    props.update(overrides)

    seen = {"rows": None}
    original = mod.ProductTable

    def _capture(**kw):
        seen["rows"] = kw["rows"]
        return original(**kw)

    mod.ProductTable = _capture
    try:
        box, rc = reacton.render(DerivedVariableList(**props), handle_error=False)
        rc.close()
    finally:
        mod.ProductTable = original
    return seen["rows"]


def test_a_running_job_is_listed_before_any_product_exists():
    """The whole point: submit -> a row, with no registry entry yet."""
    rows = _captured_rows(_empty_project(), jobs=[_job()])
    assert [r["key"] for r in rows] == ["job_j1"]
    assert rows[0]["cells"][0]["value"] == "forest_dist"


def test_a_running_job_row_shows_the_running_status():
    """The status column is what tells the user the work is under way."""
    rows = _captured_rows(_empty_project(), jobs=[_job()])
    status_cells = [c for c in rows[0]["cells"] if c.get("type") == "status"]
    assert [c["status"] for c in status_cells] == ["running"]


def test_a_running_job_row_offers_no_actions():
    """Nothing to map or delete yet, and the GDAL pass is not cancellable."""
    rows = _captured_rows(_empty_project(), jobs=[_job()])
    assert rows[0]["actions"] == []


def test_a_failed_job_row_can_be_dismissed():
    """Dismiss is the only action a job row ever offers."""
    rows = _captured_rows(_empty_project(), jobs=[_job(status="failed", error="boom")])
    assert [a["kind"] for a in rows[0]["actions"]] == ["dismiss"]
    assert rows[0]["error"] == "boom"


def test_a_failed_job_row_without_a_message_still_explains_itself():
    """A bare failure falls back to translated copy, never to a blank line."""
    rows = _captured_rows(_empty_project(), jobs=[_job(status="failed")])
    assert rows[0]["error"]


def test_dismiss_calls_back_with_that_rows_own_job_id():
    """Two failed jobs catch a bare closure over the loop variable."""
    seen = []
    jobs = [_job("j7", status="failed"), _job("j8", status="failed")]
    rows = _captured_rows(_empty_project(), jobs=jobs, on_dismiss=seen.append)
    for row in rows:
        row["actions"][0]["on_click"]()
    assert sorted(seen) == ["j7", "j8"]


def test_product_rows_keep_their_actions_and_ready_status():
    """The overlay must not disturb the rows that were already there."""
    rows = _captured_rows(_project_with_one_derived())
    assert [r["key"] for r in rows] == ["loss_forest_2010_2015"]
    assert [a["kind"] for a in rows[0]["actions"]] == ["map_toggle", "delete"]
    status_cells = [c for c in rows[0]["cells"] if c.get("type") == "status"]
    assert [c["status"] for c in status_cells] == ["ready"]


def test_a_job_row_sits_above_the_products():
    """What is happening now leads what is already done."""
    rows = _captured_rows(_project_with_one_derived(), jobs=[_job()])
    assert [r["key"] for r in rows] == ["job_j1", "loss_forest_2010_2015"]


def test_a_completed_job_is_replaced_by_its_product_row():
    """One row throughout: the job becomes the layer, it does not duplicate it."""
    project = _project_with_one_derived()
    job = _job(name="loss_forest_2010_2015", status="completed")
    rows = _captured_rows(project, jobs=[job])
    assert [r["key"] for r in rows] == ["loss_forest_2010_2015"]


def test_the_list_renders_nothing_when_there_are_no_rows_at_all():
    """Unchanged from before: an empty derived list stays invisible."""
    rows = _captured_rows(_empty_project())
    assert rows is None or rows == []
