"""Toolbox — tools that sit outside the 9-step workflow.

Today it hosts one tool, deforestation allocation. New tools are added as their
own panel component plus an entry here; the workflow steps are untouched.
"""

import logging
import uuid

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts.allocation_runner import (
    allocation_rows,
    delete_allocation_run,
    run_allocation,
)
from gui.scripts.density_map import add_density_on_map, density_layer_key
from gui.scripts.notify_bridge import tracked_job
from gui.scripts.solara_threads import spawn_in_context, update_job
from gui.store.project_writers import writing
from gui.widget.allocation_form import AllocationDetailsDialog, AllocationFormDialog
from gui.widget.allocation_list import AllocationList
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.help import InfoButton
from gui.widget.text_style import MUTED

logger = logging.getLogger("spatial_risk")


def _density_legend(run_key: str, name: str, vmin: float, vmax: float):
    """The legend one allocation run's density raster publishes."""
    from gui.scripts.legend_data import Label, density_spec
    from gui.scripts.legend_registry import LayerLegend

    return LayerLegend(
        layer_id=density_layer_key(run_key),
        label=Label(literal=name),
        spec=density_spec(vmin, vmax),
    )


def _drop_density_layer(map_, key: str, legend_port) -> None:
    """Remove a density layer and its legend. Used by toggle-off AND delete."""
    if map_ is not None:
        map_.remove_layer(key, none_ok=True)
    if legend_port is not None:
        legend_port.unregister(key)


#: The toolbox's tool list (the dialog's icon rail). One entry per tool; the
#: panel itself is selected on ``key`` in the component body.
#: ``icon`` must exist in the MDI font jupyter-vuetify bundles (~4.9):
#: mdi-earth-remove is a 5.x glyph and rendered as an empty button.
_TOOLS = [
    {
        "key": "allocation",
        "label_key": "toolbox.tool_allocation",
        "description_key": "toolbox.allocation.description",
        "info_key": "toolbox.allocation.info_details",
        "icon": "mdi-earth-off",
    }
]

# Module-level so in-flight jobs and on-map layers survive re-renders.
# Both are cleared on project switch by gui/solara_app.py (reset_jobs_on_load /
# render_map_on_switch) — add any new reactive here to those effects too.
allocation_jobs = solara.reactive([])
density_on_map = solara.reactive(set())


@solara.component
def _RailTooltip(label, children=[]):
    """solara.Tooltip, pinned to the activator's right.

    The rail hugs the dialog's left edge, so solara.Tooltip's hardcoded
    ``bottom`` placement clips against it; same v_on wiring, ``right=True``.
    """

    def set_v_on():
        for child in children:
            widget = solara.get_widget(child)
            widget.v_on = "tooltip.on"

    solara.use_effect(set_v_on, children)

    return rv.Tooltip(
        right=True,
        v_slots=[{"name": "activator", "variable": "tooltip", "children": children}],
        children=[label],
    )


def _run_allocation_job(job_id, form, project, project_reactive=None, notifier=None):
    """Worker body: entered on the pool thread, so tracked_job is entered here."""
    try:
        with tracked_job(
            notifier, t("notifications.task_allocation", name=form.name)
        ), writing(project.project_name):
            record = run_allocation(
                project,
                form,
                project_reactive=project_reactive,
                notifier=notifier,
                jobs_reactive=allocation_jobs,
                job_id=job_id,
            )
        update_job(
            allocation_jobs,
            job_id,
            status="completed",
            annual_ha=record.annual_ha,
            total_ha=record.total_ha,
            # Lets allocation_rows drop this job row once the record it points
            # at is in the registry, so the run stops rendering twice. Mirrors
            # train_tile's model_storage_key stamp.
            record_key=record.storage_key(),
        )
        logger.info(
            "Allocation completed: '%s' — %.1f ha/yr", form.name, record.annual_ha
        )
    except Exception as exc:  # surfaced on the job row
        logger.exception("Allocation failed for '%s'", form.name)
        update_job(allocation_jobs, job_id, status="failed", error=str(exc))


@solara.component
def ToolboxTile(project, map_=None, sepal_client=None, legend_port=None):
    """Tool list + the selected tool's panel.

    Args:
        project: solara.Reactive[Project] — the live project.
        map_: SepalMap for the density-raster toggle; None disables it.
        sepal_client: passed to the form's file pickers.
        legend_port: LegendPort for publishing/withdrawing density legends;
            None disables legend publication (e.g. in tests without one).
    """
    notifications = use_notifications()
    p = project.value

    form_open = solara.use_reactive(False)
    # Failed-job editing: the pencil reopens the form seeded with the job's
    # submission; submitting launches a fresh run and drops the old failed row
    # so the rerun replaces it instead of piling up next to it.
    prefill = solara.use_reactive(None)
    editing_job_id = solara.use_ref(None)
    pending_delete, set_pending_delete = solara.use_state(None)
    selected_tool, set_selected_tool = solara.use_state(_TOOLS[0]["key"])
    selected_run_key, set_selected_run_key = solara.use_state(None)

    def launch(form):
        job_id = str(uuid.uuid4())[:8]
        allocation_jobs.set(
            list(allocation_jobs.value)
            + [
                {
                    "id": job_id,
                    "name": form.name,
                    "status": "running",
                    "error": None,
                    # Submission snapshot, kept so a failed row can be re-edited.
                    "entry": form,
                }
            ]
        )
        form_open.set(False)
        spawn_in_context(_run_allocation_job, (job_id, form, p, project, notifications))
        logger.info("Allocation started: '%s' (job=%s)", form.name, job_id)
        # The rerun replaces the failed row it was launched from. Runs after
        # the new job is in the list, so the filter sees both.
        if editing_job_id.current is not None:
            on_dismiss(editing_job_id.current)
            editing_job_id.current = None
            prefill.set(None)

    def on_dismiss(job_id):
        # Job rows only — never touches the allocation registry.
        allocation_jobs.set([j for j in allocation_jobs.value if j["id"] != job_id])

    def on_edit(row):
        editing_job_id.current = row["job_id"]
        prefill.set(row["entry"])
        form_open.set(True)

    def open_new_form():
        # A New immediately after an Edit must not inherit the seed.
        editing_job_id.current = None
        prefill.set(None)
        form_open.set(True)

    def confirm_delete():
        key = pending_delete
        set_pending_delete(None)
        if key and p is not None:
            # Deleting a run used to leave its density layer on the map (and now
            # its legend too) — drop both before the record goes away.
            layer_key = density_layer_key(key)
            if layer_key in density_on_map.value:
                _drop_density_layer(map_, layer_key, legend_port)
                density_on_map.set(density_on_map.value - {layer_key})
            delete_allocation_run(p, key)
            project.set(p.model_copy())

    def toggle_density(row):
        key = density_layer_key(row["key"])
        if key in density_on_map.value:
            _drop_density_layer(map_, key, legend_port)
            density_on_map.set(density_on_map.value - {key})
        else:
            _layer, (vmin, vmax) = add_density_on_map(
                map_, row["density_map_path"], key=key, layer_name=row["name"]
            )
            density_on_map.set(density_on_map.value | {key})
            if legend_port is not None:
                legend_port.register(
                    _density_legend(row["key"], row["name"], vmin, vmax)
                )

    rows = allocation_rows(p, allocation_jobs.value) if p is not None else []

    with solara.Column(style="gap:12px;"):
        if p is None:
            solara.Info(t("toolbox.allocation.empty"))
            return

        # Icon rail + the selected tool's panel. The rail mirrors the app
        # drawer: icon-only buttons, tooltip = tool name, selected in primary.
        # The panel header carries the title, the one-line description under
        # it, and an info popup with the method details and references.
        tool = next(entry for entry in _TOOLS if entry["key"] == selected_tool)
        with solara.Row(style="gap:16px;align-items:stretch;"):
            with solara.Column(
                style="flex:0 0 48px;gap:4px;align-items:center;"
                "padding:4px 0;border-right:1px solid rgba(128,128,128,0.25);"
            ):
                for entry in _TOOLS:
                    with _RailTooltip(t(entry["label_key"])):
                        solara.Button(
                            "",
                            icon_name=entry["icon"],
                            icon=True,
                            color=(
                                "primary" if selected_tool == entry["key"] else None
                            ),
                            on_click=lambda key=entry["key"]: set_selected_tool(key),
                        )

            with solara.Column(style="flex:1;min-width:0;gap:12px;"):
                with solara.Column(
                    style="gap:2px;"
                    "border-bottom:1px solid rgba(128,128,128,0.25);"
                    "padding-bottom:8px;"
                ):
                    with solara.Row(style="gap:6px;align-items:center;"):
                        solara.Text(
                            t(tool["label_key"]),
                            style="font-weight:600;font-size:0.95rem;",
                        )
                        InfoButton(
                            title=t(tool["label_key"]),
                            markdown=t(tool["info_key"]),
                        )
                    solara.Text(
                        t(tool["description_key"]),
                        style=MUTED + "font-size:0.8rem;",
                    )

                AllocationList(
                    rows=rows,
                    on_delete=set_pending_delete,
                    on_open=set_selected_run_key,
                    on_toggle_density=toggle_density if map_ is not None else None,
                    density_on_map=density_on_map.value,
                    on_edit=on_edit,
                    on_dismiss=on_dismiss,
                )
                solara.Button(
                    t("toolbox.allocation.new"),
                    icon_name="mdi-plus",
                    block=True,
                    on_click=open_new_form,
                )

        ConfirmDialog(
            open=pending_delete is not None,
            title=t("toolbox.allocation.delete_title"),
            message=t("toolbox.allocation.delete_message", name=pending_delete or ""),
            on_confirm=confirm_delete,
            on_cancel=lambda: set_pending_delete(None),
        )
        AllocationFormDialog(
            open_=form_open,
            project=project,
            on_launch=launch,
            on_close=lambda: form_open.set(False),
            sepal_client=sepal_client,
            # Every job in the list — running or failed — still shows its name
            # in the list above, so none is excluded by status.
            running_names=frozenset(
                job["name"] for job in allocation_jobs.value if job.get("name")
            ),
            # Rejected outright, so a second click cannot start a second run.
            active_names=frozenset(
                job["name"]
                for job in allocation_jobs.value
                if job.get("name") and job.get("status") == "running"
            ),
            prefill=prefill,
        )
        AllocationDetailsDialog(
            project=project,
            run_key=selected_run_key,
            on_close=lambda: set_selected_run_key(None),
        )
