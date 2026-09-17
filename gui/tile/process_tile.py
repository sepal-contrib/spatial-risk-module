"""Step 3 — Harmonization tile (mirrors notebooks/2.process_factory.ipynb)."""

import asyncio
import logging

import reacton.ipyvuetify as rv
import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts import process_actions
from gui.scripts.inflight import InflightKeys
from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT, tracked_job
from gui.scripts.solara_threads import (
    publish_if_current,
    spawn_in_context,
    to_thread_in_context,
)
from gui.scripts.variable_identity import base_raster_key, is_base_raster
from gui.store.project_writers import writing
from gui.tile.derived_map import derived_on_map, use_derived_map_toggle
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.creation_dialog import CreationDialog
from gui.widget.help import InfoButton
from gui.widget.text_style import MUTED
from gui.widget.variable_list import HarmonizationVariableList
from spatialrisk.harmonization import (
    HarmonizationStatus,
    harmonization_status,
    harmonization_status_from_disk,
)

logger = logging.getLogger("spatial_risk")

# Only one reference warp at a time (see InflightKeys).
reference_inflight = InflightKeys(key="reference_inflight")


def base_sig_of(p):
    """The open project's base grid signature, or None."""
    base = getattr(p, "base_raster", None) if p is not None else None
    return getattr(base, "grid_signature", None) if base is not None else None


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
def ReferenceStrip(project, on_open, pending=False):
    """Clickable statement of the current reference grid; opens the form.

    It carries the whole choice — raster, CRS and pixel size — because it
    replaces a form that used to state all three permanently above the list.
    Unset, it prompts instead, in warning colours: that is also what tells the
    user why Run is disabled, so no separate "set a reference first" line is
    needed.

    ``pending`` is True while the reference warp runs on its worker thread: the
    strip then says so, carries a progress bar and stops opening the form,
    because the old reference it still holds is about to be replaced and the UI
    is otherwise unchanged (the warp no longer freezes it, so nothing else
    signals that work is happening). Disabling it is what keeps the user from
    typing a correction that ``on_set_base`` would only have to refuse — and the
    pending line it carries is the reason, so the disabled state is not mute.
    """
    p = project.value
    base = p.base_raster if p is not None else None
    if pending:
        summary = t("tiles.process.reference_pending")
    elif base is not None:
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
            disabled=pending,
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
        if pending:
            solara.ProgressLinear(True)


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
        # Non-blocking: CreationDialog's two channels both stop the user
        # (validate() -> error Alert, will_replace() -> confirm), and neither
        # says "go ahead, but know this". Rendering it here also lets it
        # update live as the field is typed.
        _, warning = process_actions.validate_projection(epsg, resolution)
        if warning == "geographic_crs":
            rv.Alert(
                type="warning",
                dense=True,
                children=[t("tiles.process.warn_geographic_crs", epsg=epsg.strip())],
            )
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
    # Last resolved *disk* verdict, tagged with the inputs it was computed
    # from: the hint task yields None while a run is in flight (it would be
    # reading files the run is rewriting), and the unstamped rows must not
    # blank out meanwhile — but a verdict computed for other inputs (another
    # project, a different base) must never be shown against the current rows.
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

    _VALIDATION_MESSAGES = {
        "need_epsg": "tiles.process.error_need_epsg",
        "bad_epsg": "tiles.process.error_bad_epsg",
        "bad_resolution": "tiles.process.error_bad_resolution",
    }

    def validate_reference():
        """Name the missing or bad field — the submit used to just sit disabled."""
        if not base_key:
            return t("tiles.process.error_pick_reference")
        error, _ = process_actions.validate_projection(epsg, resolution)
        if error is None:
            return None
        return t(_VALIDATION_MESSAGES[error], epsg=epsg.strip())

    def on_set_base():
        """Warp the chosen raster onto the new grid, off the websocket loop.

        `set_base_raster` runs a full GDAL warp that writes a raster. Solara
        widget callbacks run inside process_kernel_messages under the session's
        context lock, so doing that here froze the whole UI for the duration.
        All continuation work — the republish and the error toast — lives in the
        worker, per gui/scripts/solara_threads.
        """
        if p is None:
            return
        if not reference_inflight.claim("reference"):
            # The strip is disabled while a warp runs, but `disabled=` is a
            # render-time prop and reaches the browser a round-trip late, so a
            # click can still get the form open. CreationDialog closes on any
            # launch that returns, so refusing in silence would look exactly
            # like a successful submit — and the corrected EPSG the user just
            # typed would never be applied. Say so instead.
            notifications.warning(
                t("tiles.process.reference_already_running"),
                timeout=ERROR_TOAST_TIMEOUT,
            )
            return
        # Safe on this thread: validate_reference() has already parsed it.
        res = float(resolution)
        key, code = base_key, epsg.strip()

        def _worker():
            try:
                # auto_save=False is load-bearing: use_as_base_raster() saves by
                # default, and publish_if_current can only stop the *reactive*
                # publish. A worker that saved first would already have written a
                # project the user switched away from or deleted mid-warp — and
                # Project.save() recreates the folder it writes into. So check
                # liveness first, then save.
                process_actions.set_base_raster(p, key, code, res, auto_save=False)
                if publish_if_current(project, p):
                    p.save()
            except Exception as exc:
                logger.exception("setting the reference raster failed")
                notifications.error(
                    t("tiles.process.error_set_base", exc=exc),
                    timeout=ERROR_TOAST_TIMEOUT,
                )
            finally:
                reference_inflight.release("reference")

        try:
            spawn_in_context(_worker)
        except Exception as exc:
            # The worker's finally is what releases the claim, so a thread that
            # never starts would hold it for the rest of the session.
            reference_inflight.release("reference")
            logger.exception("could not start the reference worker")
            notifications.error(
                t("tiles.process.error_set_base", exc=exc), timeout=ERROR_TOAST_TIMEOUT
            )

    # Pure and in-memory (see harmonization.harmonization_status): no file
    # opens, so unlike the old disk scan this belongs in the render body. Every
    # output stamped by this version carries the grid it was written onto, and
    # comparing those strings against the base's is what the rows are drawn
    # from — a re-render costs attribute reads, not `1 + N` file opens on the
    # session's websocket loop.
    pure = harmonization_status(p) if p is not None else None
    unknown_keys = tuple(sorted(pure.unknown)) if pure is not None else ()
    # Only the entries the signatures cannot answer reach disk. In a project
    # harmonized by this version the tuple is empty, the task returns
    # immediately, and the status path performs no I/O at all.
    #
    # The project name and base signature are in the key for staleness, not for
    # retriggering: `unknown_keys` alone aliases badly — two different projects,
    # or the same project against a different base, can produce an identical
    # tuple, and `last_status` would then serve one state's disk verdict against
    # another's rows. That is exactly the invariant the existing `status_key`
    # tagging exists to hold (see the `last_status` comment above).
    hint_key = (
        getattr(p, "project_name", None),
        base_sig_of(p),
        unknown_keys,
        processing.value,
    )

    @solara.lab.use_task(
        dependencies=[hint_key], raise_error=False, prefer_threaded=True
    )
    async def harmonization_hint():
        """Disk verdicts for the layers with no recorded grid signature.

        Kept threaded: this is the only remaining disk I/O on the status path,
        and in the render body it would block the session's websocket loop.
        Returns None when there is nothing to check or a run is rewriting the
        very files we would inspect.
        """
        if p is None or p.base_raster is None or processing.value or not unknown_keys:
            return None
        try:
            return await asyncio.to_thread(
                harmonization_status_from_disk, p, list(unknown_keys)
            )
        except Exception:
            # Advisory only, and raise_error=False swallows it silently
            # otherwise. The race is real: this walks raw_variables on a worker
            # thread while a Variables-tab download can be adding keys.
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
        ReferenceStrip(
            project=project,
            on_open=lambda: reference_open.set(True),
            pending="reference" in reference_inflight,
        )

        # Everything in hint_key except processing.value: a run in flight must
        # keep the status it started from, not invalidate it.
        status_key = hint_key[:-1]
        if harmonization_hint.value is not None:
            last_status.current = (status_key, harmonization_hint.value)
        disk = last_status.current[1] if last_status.current[0] == status_key else None
        # The pure verdict stands; the disk one overrides for unknown keys only.
        if pure is None:
            status = None
        elif not pure.unknown:
            status = pure  # steady state: no disk involved
        elif disk is None:
            # `pure` still carries its `unknown` list, which is what makes the
            # rows render "checking" rather than "harmonized".
            status = pure
        else:
            status = HarmonizationStatus(
                pending=pure.pending + disk.pending,
                current=pure.current + disk.current,
            )
        run_in_flight = processing.value or process_task.pending
        # Only once the status has actually landed: while it is still being
        # checked we do not know there is nothing to do, and disabling on a
        # maybe would block a run the user is entitled to. An `unknown` entry
        # is such a maybe — it sits in neither list, so keying off an empty
        # `pending` alone would kill the button on exactly the legacy projects
        # that most need a run.
        nothing_pending = (
            status is not None
            and not status.pending
            and not getattr(status, "unknown", ())
        )
        if not run_in_flight:
            running_keys = set()
        elif pending_harmonize.value is not None:
            running_keys = {pending_harmonize.value}
        elif status is None:
            running_keys = set()
        else:
            # A bulk run recomputes its own status from disk over the whole
            # project, so an unstamped layer is genuinely under consideration
            # and must not read as idle while the run works on it. Normally
            # moot — the merge above resolves `unknown` away before a run
            # starts — but not on a legacy project where Run is pressed before
            # the disk verdict lands, which is the case this exists for.
            running_keys = set(status.pending) | set(
                getattr(status, "unknown", None) or ()
            )

        HarmonizationVariableList(
            project=project,
            status=status,
            on_harmonize=harmonize_one,
            running_keys=running_keys,
            # The reference warp gates the row hammers too, not just the strip
            # and Harmonize-all: a per-row run started mid-warp hands GDAL a
            # ``p.base_raster`` the reference worker is about to replace, and
            # races its ``Project.save()`` against the worker's. Before the warp
            # moved off the websocket loop the frozen UI made that click
            # impossible; now it has to be refused explicitly.
            harmonize_disabled=run_in_flight
            or not has_base
            or "reference" in reference_inflight,
            on_toggle_map=on_toggle_map,
            derived_on_map=derived_on_map,
            on_remove=set_pending_remove,
        )

        # Only the still-resolving case needs a line of its own: until the
        # status lands no row can state one. Once it does, every row says its
        # own, so a summary above them would only repeat the column beside.
        # "Still resolving" now means unresolved *unknowns*: the pure verdict
        # lands on the first render, so a missing status is no longer the test.
        if (
            pure is not None
            and pure.unknown
            and disk is None
            and harmonization_hint.pending
        ):
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
            disabled=run_in_flight
            or not has_base
            or nothing_pending
            or "reference" in reference_inflight,
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
