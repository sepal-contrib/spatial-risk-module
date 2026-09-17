"""Step 3 — Harmonization tile (mirrors notebooks/2.process_factory.ipynb)."""

import asyncio
import logging

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts import process_actions
from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT, tracked_job
from gui.scripts.solara_threads import publish_if_current, to_thread_in_context
from gui.scripts.variable_identity import base_raster_key, is_base_raster
from gui.store.project_writers import writing
from gui.tile.derived_map import derived_on_map, use_derived_map_toggle
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.creation_dialog import CreationDialog
from gui.widget.help import InfoButton
from gui.widget.text_style import MUTED
from gui.widget.variable_list import HarmonizationVariableList
from spatialrisk.harmonization import harmonization_status

logger = logging.getLogger("spatial_risk")


def _raw_raster_keys(p):
    """Keys of raw raster variables (candidates for the base)."""
    from spatialrisk.variables.models import DataType

    if p is None:
        return []
    return [
        k
        for k, v in p.raw_variables.items()
        if getattr(v, "data_type", None) != DataType.vector
    ]


# A button, not a clickable div: focus, keyboard activation and the disabled
# state come for free. Vuetify's fixed height and uppercasing are the only
# things a two-line label needs undone.
STRIP_STYLE = (
    "height:auto;min-height:46px;padding:6px 12px;"
    "text-transform:none;letter-spacing:normal;"
)
STRIP_CAPTION = (
    "font-size:0.62rem;font-weight:700;text-transform:uppercase;"
    "letter-spacing:0.08em;opacity:0.7;line-height:1.35;"
)
# Opacity, not a `text--*` class: theme-scoped classes go black inside dark
# dialogs under voila (see gui/widget/text_style).
# `display:block` is what makes the ellipsis apply at all — solara.Text renders
# an inline span, and a long raster name otherwise runs to the panel edge.
STRIP_SUMMARY = (
    "font-size:0.8rem;line-height:1.35;display:block;max-width:100%;"
    "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
)
# Vuetify's button content is `flex:1 0 auto` and centred, so it refuses to
# shrink: without this a long raster name ignores the ellipsis and spills out
# BOTH sides of the button. Same defect and same fix as ProductTable's
# `.v-chip__content` rule.
STRIP_CSS = """
.sr-reference-strip .v-btn__content {
  width: 100%; min-width: 0; max-width: 100%; flex: 1 1 auto;
}
"""


@solara.component
def ReferenceStrip(project, on_open):
    """Clickable statement of the current reference grid; opens the form.

    It carries the whole choice — raster, CRS and pixel size — because it
    replaces a form that used to state all three permanently above the list.
    Unset, it prompts instead, in warning colours: that is also what tells the
    user why Run is disabled, so no separate "set a reference first" line is
    needed.
    """
    p = project.value
    base = p.base_raster if p is not None else None
    if base is not None:
        summary = t(
            "tiles.process.reference_summary",
            name=base.name,
            crs=base.default_crs,
            resolution=round(base.default_resolution or 0),
        )
    else:
        summary = t("tiles.process.reference_unset")

    # solara.Style needs a container to render into — at a component's top
    # level it is silently dropped, and the strip then overflows its button.
    with solara.Column(style="width:100%;gap:0;"):
        solara.Style(STRIP_CSS)
        # The label is built as `children=`, not a nested `with`: reacton
        # reparents elements passed this way, and it keeps the two text lines
        # inside the button's own content box where the CSS above can reach.
        solara.Button(
            classes=["sr-reference-strip"],
            block=True,
            outlined=True,
            color="primary" if base is not None else "warning",
            style=STRIP_STYLE,
            on_click=on_open,
            children=[
                solara.Row(
                    style="width:100%;align-items:center;gap:8px;flex-wrap:nowrap;",
                    children=[
                        rv.Icon(children=["mdi-crosshairs-gps"], small=True),
                        solara.Column(
                            style=(
                                "gap:0;align-items:flex-start;"
                                "min-width:0;flex:1 1 auto;"
                            ),
                            children=[
                                solara.Text(
                                    t("tiles.process.reference_label"),
                                    style=STRIP_CAPTION,
                                ),
                                solara.Text(summary, style=STRIP_SUMMARY),
                            ],
                        ),
                        rv.Icon(children=["mdi-chevron-right"], small=True),
                    ],
                )
            ],
        )


@solara.component
def BaseProjectionForm(
    project,
    base_key,
    set_base_key,
    epsg,
    set_epsg,
    resolution,
    set_resolution,
    on_auto_utm,
    autofill_pending,
):
    """Reference & projection form (Select, EPSG ⌖ + resolution).

    Rendered as the body of ``CreationDialog``, which owns the submit and
    cancel actions — the form itself is fields only.

    A separate component because ``rv.use_event`` is a hook and must run
    unconditionally every render — ProcessTile early-returns before the form
    when there are no variables, so the hook cannot live in its body (and no
    early return may precede the hook here either; ``_raw_raster_keys``
    already handles a ``None`` project). Receives the ``project`` reactive
    (not ``.value``): value-equal ``model_copy`` snapshots would suppress
    child re-renders.
    """
    with solara.Column(style="gap:14px;"):
        rv.Select(
            label=t("tiles.process.base_raster_label"),
            items=_raw_raster_keys(project.value),
            v_model=base_key,
            on_v_model=set_base_key,
            dense=True,
            outlined=True,
            hint=t("tiles.process.base_raster_hint"),
        )
        with solara.Row(style="gap:8px;align-items:flex-start;flex-wrap:nowrap;"):
            epsg_field = rv.TextField(
                label=t("tiles.process.epsg_label"),
                v_model=epsg,
                on_v_model=set_epsg,
                dense=True,
                outlined=True,
                placeholder=t("tiles.process.epsg_placeholder"),
                style_="flex:1 1 55%;min-width:0;",
                hint=t("tiles.process.epsg_hint"),
                append_icon="mdi-crosshairs-gps",
            )
            rv.TextField(
                label=t("tiles.process.resolution_label"),
                v_model=resolution,
                on_v_model=set_resolution,
                dense=True,
                outlined=True,
                type="number",
                style_="flex:1 1 45%;min-width:0;",
                hint=t("tiles.process.resolution_hint"),
            )
        rv.use_event(epsg_field, "click:append", lambda *_: on_auto_utm())
        if autofill_pending:
            solara.Text(
                t("tiles.process.detecting_projection"),
                style=MUTED + "font-size:0.8rem;font-style:italic;",
            )


@solara.component
def ProcessTile(project, processing, map_=None, legend_port=None):
    """Base/projection → run harmonization (downloading lives in Step 2 — Variables).

    Args:
        project: Reactive holding the current Project (or None).
        processing: Reactive processing-settings state.
        map_: SepalMap instance used by the "show on map" toggle.
        legend_port: LegendPort for publishing/withdrawing harmonized-output
            legends; None disables legend publication (e.g. in tests without
            one).
    """
    base_key, set_base_key = solara.use_state("")
    epsg, set_epsg = solara.use_state("")
    resolution, set_resolution = solara.use_state("30")
    notifications = use_notifications()
    on_toggle_map = use_derived_map_toggle(
        project, map_, notifications, legend_port=legend_port
    )
    pending_remove, set_pending_remove = solara.use_state(None)
    reference_open = solara.use_reactive(False)
    # Raw key the next run is restricted to (per-row harmonize button), or
    # None for a full run — the same shape as pending_download in Step 2.
    pending_harmonize = solara.use_reactive(None)
    # Last resolved status, tagged with the inputs it was computed from: the
    # hint task yields None while a run is in flight (it would be reading files
    # the run is rewriting), and the list must not blank out its rows meanwhile
    # — but a status computed for other inputs (another project, a different
    # base) must never be shown against the current rows.
    last_status = solara.use_ref((None, None))

    def _do_remove(key: str):
        """Unregister a harmonized output (the raster stays on disk)."""
        p = project.value
        if process_actions.remove_processed_variable(p, key, map_, legend_port):
            project.set(p.model_copy())

    p = project.value
    has_vars = p is not None and bool(p.raw_variables)
    has_base = p is not None and p.base_raster is not None

    # Restore the form from a loaded project. The base raster is stored in the
    # model, but base_key / epsg / resolution are transient use_state that
    # default empty, so after a load the "Base raster" Select looked unset. Keyed
    # on the stored base's key, so it fires when a project is loaded (or the base
    # changes) but not on an in-progress dropdown selection. We restore the
    # stored CRS / resolution too — recomputing them (see autofill_base) could
    # diverge for a non-UTM base CRS.
    restored_key = base_raster_key(p)

    def _restore_base_form():
        if not restored_key:
            return
        set_base_key(restored_key)
        if p.base_raster.default_crs:
            set_epsg(str(p.base_raster.default_crs))
        if p.base_raster.default_resolution:
            set_resolution(str(round(p.base_raster.default_resolution)))

    solara.use_effect(_restore_base_form, [restored_key])

    @solara.lab.use_task(
        dependencies=[base_key], raise_error=False, prefer_threaded=True
    )
    async def autofill_base():
        """On base-raster selection, pre-fill EPSG (UTM) + resolution; stay editable."""
        if p is None or not base_key:
            return
        var = p.raw_variables.get(base_key)
        if var is None:
            return
        # The selection already backs the current base raster (e.g. restored
        # after a project load): keep its stored CRS / resolution rather than
        # recomputing them from the source file, which could differ (e.g. a
        # non-UTM base CRS).
        if is_base_raster(p, var):
            return
        res = await asyncio.to_thread(process_actions.base_raster_resolution, var)
        if res:
            set_resolution(str(round(res)))
        path = getattr(var, "path", None)
        if path is None:
            return  # not downloaded yet — auto-UTM needs the GeoTIFF on disk
        set_epsg(await asyncio.to_thread(process_actions.auto_utm_epsg, path))

    def on_auto_utm():
        # The ⌖ icon has no disabled state — gate here (was the old
        # button's ``disabled=not base_key or autofill_base.pending``).
        if p is None or not base_key or autofill_base.pending:
            return
        try:
            base = p.raw_variables[base_key]
            path = getattr(base, "path", None)
            if path is None:
                notifications.error(
                    t("tiles.process.error_download_first"),
                    timeout=ERROR_TOAST_TIMEOUT,
                )
                return
            set_epsg(process_actions.auto_utm_epsg(path))
        except Exception as exc:
            notifications.error(
                t("tiles.process.error_auto_utm", exc=exc), timeout=ERROR_TOAST_TIMEOUT
            )

    def validate_reference():
        """Name the missing field — the form's submit used to just sit disabled."""
        if not base_key:
            return t("tiles.process.error_pick_reference")
        if not epsg.strip():
            return t("tiles.process.error_need_epsg")
        return None

    def on_set_base():
        if p is None:
            return
        try:
            res = float(resolution) if str(resolution).strip() else 30.0
            process_actions.set_base_raster(p, base_key, epsg.strip(), res)
            project.set(p.model_copy())
        except Exception as exc:
            notifications.error(
                t("tiles.process.error_set_base", exc=exc), timeout=ERROR_TOAST_TIMEOUT
            )

    # Rebuilt from scalars on every render: the project reactive is republished
    # via model_copy(), which compares equal to its predecessor, so a use_task
    # keyed on it would never retrigger. `processing.value` is what refreshes
    # the hint after a run whose key sets did not change (a re-harmonization
    # after the reference raster moved).
    #
    # Raw variables contribute their edit-sensitive scalars, not just their
    # keys: editing a variable in place keeps name+year, so the key SET does not
    # move and a key-only dependency would never refire — leaving the hint
    # reading "already harmonized" about a layer the edit just made stale, which
    # is precisely the advice not to press Run. These are attribute reads, no
    # disk I/O, so they are safe in a render body.
    #
    # Known gap, accepted: re-setting the SAME reference raster at the SAME CRS
    # and resolution after its source extent changed yields a new geobox this
    # key cannot see. Harmless — F3 recomputes status from disk the instant Run
    # is pressed, so only the hint goes stale, never the run. The honest fix
    # (stat() on the base file here) is blocking disk I/O in the render body,
    # i.e. the websocket loop, which a cosmetic staleness does not justify.
    hint_key = (
        base_raster_key(p),
        getattr(p.base_raster, "default_crs", None) if p and p.base_raster else None,
        (
            getattr(p.base_raster, "default_resolution", None)
            if p and p.base_raster
            else None
        ),
        (
            tuple(
                (
                    k,
                    str(getattr(v, "path", None)),
                    getattr(v, "raster_type", None),
                    getattr(v, "rasterization_method", None),
                )
                for k, v in sorted(p.raw_variables.items())
            )
            if p
            else ()
        ),
        tuple(sorted(p.processed_variables)) if p else (),
        processing.value,
    )

    @solara.lab.use_task(
        dependencies=[hint_key], raise_error=False, prefer_threaded=True
    )
    async def harmonization_hint():
        """How many layers Run would actually process, checked off-thread.

        Reading each output's header and mtime is disk I/O; done in the render
        body it would block the session's websocket loop. Returns None when
        there is nothing to say — no project, no reference raster, or a run in
        flight rewriting the very files we would be inspecting.
        """
        if p is None or p.base_raster is None or processing.value:
            return None
        try:
            return await asyncio.to_thread(harmonization_status, p)
        except Exception:
            # The hint is advisory, so failing it must never surface as an
            # error — but ``raise_error=False`` swallows the exception with no
            # trace at all. The race is real: this walks ``p.raw_variables`` on
            # a worker thread while a Variables-tab download can be adding keys
            # to it (a multi-image GEEVar), which raises "dictionary changed
            # size during iteration". Logging turns an invisible blank hint
            # into a diagnosable one.
            logger.debug("Harmonization hint failed", exc_info=True)
            return None

    @solara.lab.use_task(dependencies=None, raise_error=False, prefer_threaded=True)
    async def process_task():
        if p is None:
            return
        processing.set(True)
        only = pending_harmonize.value
        keys = [only] if only is not None else None
        title = (
            t(
                "notifications.task_processing_one",
                name=getattr(p.raw_variables.get(only), "name", only),
            )
            if only is not None
            else t("notifications.task_processing")
        )

        def _tracked_run():
            # tracked_job is entered on the pool thread itself so the library's
            # per-stage log lines (download/reproject/rasterize) land on THIS
            # job's tracker; to_thread_in_context gives that thread the kernel
            # context its bus updates need to reach the browser.
            with tracked_job(
                notifications,
                title,
                error_format=lambda exc: t("tiles.process.error_processing", exc=exc),
            ):
                process_actions.run_processing(p, keys=keys)

        with writing(p.project_name):
            try:
                await to_thread_in_context(_tracked_run)
            except Exception:
                logger.exception("processing failed")  # toast raised by tracked_job
            finally:
                processing.set(False)
            publish_if_current(project, p)

    def run_processing():
        """Run button: drop a re-click while a run is already in flight.

        ``TaskAsyncio.__call__`` sets ``pending`` synchronously on this thread, so
        this is a hard guard; the button's ``disabled`` only reaches the browser a
        round-trip later, so a real double-click does land twice. Re-invoking would
        cancel the in-flight task — which does NOT stop its asyncio.to_thread body —
        unwinding the `with writing(...)` block (dropping the writer mark) while the
        orphaned executor thread keeps writing rasters and calling project.save().
        Same pattern as ProjectPanel.confirm_delete.
        """
        if process_task.pending:
            return
        pending_harmonize.set(None)
        process_task()

    def harmonize_one(key: str):
        """Row button: harmonize just this raw variable (same guard as Run)."""
        if process_task.pending:
            return
        pending_harmonize.set(key)
        process_task()

    with solara.Column(style="gap:16px;"):
        with solara.Row(style="gap:4px;align-items:center;"):
            solara.Text(t("tiles.process.description"))
            InfoButton(t("tiles.process.info_header"), t("tiles.process.info_md"))
        if not has_vars:
            solara.Info(t("tiles.process.error_no_variables"))
            return

        # The reference grid is chosen once and then read; the form lives in a
        # dialog and this states the choice. Everything below is the list you
        # actually work in — the shape of Step 2, whose variable form is
        # likewise a dialog.
        ReferenceStrip(project=project, on_open=lambda: reference_open.set(True))

        # Everything in hint_key except processing.value: a run in flight must
        # keep the status it started from, not invalidate it.
        status_key = hint_key[:-1]
        if harmonization_hint.value is not None:
            last_status.current = (status_key, harmonization_hint.value)
        status = (
            last_status.current[1] if last_status.current[0] == status_key else None
        )
        run_in_flight = processing.value or process_task.pending
        # Only once the status has actually landed: while it is still being
        # checked (status None) we do not know there is nothing to do, and
        # disabling on a maybe would block a run the user is entitled to.
        nothing_pending = status is not None and not status.pending
        if run_in_flight:
            only = pending_harmonize.value
            running_keys = (
                {only} if only is not None else set(status.pending if status else ())
            )
        else:
            running_keys = set()

        HarmonizationVariableList(
            project=project,
            status=status,
            on_harmonize=harmonize_one,
            running_keys=running_keys,
            harmonize_disabled=run_in_flight or not has_base,
            on_toggle_map=on_toggle_map,
            derived_on_map=derived_on_map,
            on_remove=set_pending_remove,
        )

        # Only the still-resolving case needs a line of its own: until the
        # status lands no row can state one. Once it does, every row says its
        # own, so a summary above them would only repeat the column beside.
        if status is None and harmonization_hint.pending:
            solara.Text(
                t("tiles.process.checking_status"),
                style=MUTED + "font-size:0.8rem;font-style:italic;",
            )

        # Same icon as the per-row action, same shape as Step 2's Download-all
        # under its source list.
        solara.Button(
            t("tiles.process.harmonize_all_button"),
            icon_name="mdi-hammer",
            color="primary",
            small=True,
            block=True,
            on_click=run_processing,
            # `disabled=` is a render-time prop: like `processing` (only set INSIDE
            # the coroutine), it reaches the browser a round-trip after the task
            # starts, so neither actually stops a fast double-click. This is
            # cosmetic only — it makes the button also *look* disabled during that
            # window. The real guard is run_processing()'s synchronous
            # process_task.pending check above, same gate as ProjectPanel's
            # confirm_delete. variables_tile/postprocess_tile still wire
            # on_click straight to their task (same gap, filed as a follow-up).
            # ``nothing_pending`` is not a double-click guard but a no-op guard:
            # with every layer already on the reference grid a run rewrites
            # nothing, and the sentence that used to say so is gone, so the
            # button carries it — as Download-all does with no cloud layers left.
            disabled=run_in_flight or not has_base or nothing_pending,
        )
        if processing.value:
            solara.ProgressLinear(True)

    # `will_replace` is the creation flow's overwrite guard; setting a reference
    # is idempotent, so there is nothing to confirm.
    CreationDialog(
        open_=reference_open,
        title=t("tiles.process.reference_dialog_title"),
        create_label=t("tiles.process.set_base_button"),
        create_icon="mdi-target",
        validate=validate_reference,
        will_replace=lambda: None,
        launch=on_set_base,
        max_width="520px",
        children=[
            BaseProjectionForm(
                project=project,
                base_key=base_key,
                set_base_key=set_base_key,
                epsg=epsg,
                set_epsg=set_epsg,
                resolution=resolution,
                set_resolution=set_resolution,
                on_auto_utm=on_auto_utm,
                autofill_pending=autofill_base.pending,
            )
        ],
    )

    ConfirmDialog(
        open=pending_remove is not None,
        on_cancel=lambda: set_pending_remove(None),
        on_confirm=lambda: (_do_remove(pending_remove), set_pending_remove(None)),
        title=t("tiles.process.confirm_remove_title"),
        message=t("tiles.process.confirm_remove_message", name=pending_remove or ""),
        confirm_label=t("common.remove"),
    )
