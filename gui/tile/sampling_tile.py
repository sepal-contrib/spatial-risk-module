"""Step 5 — Sampling tile.

Generates persistent, selectable samples from a processed raster
variable (defines the grid and, for stratified sampling, the strata) and an
optional mask variable (restricts valid pixels). Step 6 (Train) selects a
sample and a dataset, then extracts features at the sample points.
"""

import logging
import threading
import uuid

import solara
from pysepal.solara.notifications import use_notifications

from gui.i18n import t
from gui.scripts.inflight import InflightKeys
from gui.scripts.notify_bridge import tracked_job
from gui.scripts.solara_threads import publish_if_current, spawn_in_context, update_job
from gui.store.project_writers import writing
from gui.widget.confirm_dialog import ConfirmDialog
from gui.widget.help import InfoButton
from gui.widget.sample_form_dialog import SampleDetailsDialog, SampleFormDialog
from gui.widget.sample_set_list import SampleSetList
from spatialrisk.gdal_env import sampling_gdal_env

logger = logging.getLogger("spatial_risk")


# Module-level reactives shared across re-renders.
sampling_jobs = solara.reactive([])
samples_on_map = solara.reactive(set())
# Guards every read-modify-write of samples_on_map: toggle workers run
# concurrently (one thread per sample) and the remove handler runs on the
# kernel thread, so an unlocked `set(value | {key})` can drop a key.
samples_on_map_lock = threading.Lock()
samples_pending = InflightKeys(key="samples_pending")

_sampling_slot = threading.Semaphore(1)
"""Single-slot queue: at most one sampling job reads its raster at a time.

Sampling is memory-heavy -- a single job can peak in the tens of GiB on a
country-scale raster (see Track E baseline). `spawn_in_context` starts one
real thread per sample name, so distinct sample names used to run their reads
concurrently in this process, multiplying peak memory. Acquiring this
semaphore around the read (`sample.generate()`) forces jobs to queue: a job
still gets its own thread and its status card still shows "running" the whole
time (this app has no "queued" status to show instead), but only one job is
actually inside the raster read at once. This does not reduce how much memory
sampling needs in total -- it only stops several peaks from stacking up in one
process; the fix for the size of a single peak lives in `spatialrisk/sampling`.
"""


def _sample_layer_key(name: str) -> str:
    return f"sample_{name}"


def _remove_sample_layers(map_, base_key):
    """Clear both rendering paths for a sample.

    Clears PMTiles (base key) and the GeoJSON fallback
    (base_key__event / base_key__forest).
    """
    from gui.scripts.map_helpers import remove_sample_points_from_map
    from gui.scripts.pmtiles_map import remove_sample_pmtiles_from_map

    remove_sample_pmtiles_from_map(map_, base_key)
    remove_sample_points_from_map(map_, base_key)


def _toggle_sample_on_map(key, project_reactive, map_, turn_on):
    """Background worker: add/remove a sample's map layer off the kernel thread.

    Prefers PMTiles vector tiles; falls back to GeoJSON when the sample has no
    .pmtiles (old project / tippecanoe missing) or PMTiles add fails.
    """
    base_key = _sample_layer_key(key)
    try:
        if turn_on:
            cur = project_reactive.value
            ss = cur.samples.get(key) if cur else None
            if ss is None:
                return
            # Backfill: samples generated without tippecanoe (old project, or a
            # deploy env that lacked the binary) carry pmtiles_path=None in the
            # manifest forever — convert on first toggle and persist, so they
            # heal once the binary is available.
            prev_pm = getattr(ss, "pmtiles_path", None)
            ensure = getattr(ss, "ensure_pmtiles", None)
            if callable(ensure):
                try:
                    ensure()
                except Exception:
                    logger.exception("PMTiles backfill failed for %s", key)
                if getattr(ss, "pmtiles_path", None) != prev_pm:
                    try:
                        cur.save()
                    except Exception:
                        logger.exception(
                            "Could not persist backfilled pmtiles for %s", key
                        )
            drew = False
            if getattr(ss, "pmtiles_path", None):
                try:
                    from gui.scripts.pmtiles_map import add_sample_pmtiles_on_map

                    add_sample_pmtiles_on_map(map_, ss.pmtiles_path, key, base_key)
                    drew = True
                except Exception:
                    logger.exception("PMTiles add failed for %s; GeoJSON fallback", key)
            if not drew:
                if ss.points_path is None:
                    return
                from gui.scripts.map_helpers import add_sample_points_on_map

                add_sample_points_on_map(map_, ss.points_path, key, base_key)
            with samples_on_map_lock:
                samples_on_map.set(samples_on_map.value | {key})
        else:
            _remove_sample_layers(map_, base_key)
            with samples_on_map_lock:
                samples_on_map.set(samples_on_map.value - {key})
    except Exception:
        logger.exception("sample map toggle failed for %s", key)
    finally:
        samples_pending.release(key)


def _update_job(job_id, *, skip_if_cancelled=True, **changes):
    update_job(sampling_jobs, job_id, skip_if_cancelled=skip_if_cancelled, **changes)


def _run_sampling(
    job_id,
    name,
    raster_var,
    mask_var,
    strategy,
    allocation,
    adapt,
    n_samples,
    spacing_m,
    seed,
    project_reactive,
    notifier=None,
    task_title=None,
):
    """Generate a Sample in the background and register it on the project."""
    try:
        from spatialrisk.sample import Sample

        p = project_reactive.value
        if p is None:
            return  # project was closed/deleted while the job was queued
        with tracked_job(
            notifier, task_title or f"Generating sample '{name}'"
        ), writing(p.project_name):
            folder = p.folders.samples_folder
            sample = Sample(
                project=p,
                name=name,
                raster_var_name=raster_var,
                mask_var_name=mask_var if mask_var else None,
                strategy=strategy,
                allocation=allocation if strategy == "stratified" else None,
                adapt=adapt,
                n_samples=n_samples,
                spacing_m=spacing_m,
                seed=seed,
                points_path=folder / f"{name}.gpkg",
            )
            # Serialize the raster read across concurrent sampling jobs (see
            # `_sampling_slot`) and budget GDAL's own block cache for it --
            # bounding the numpy arrays in `spatialrisk/sampling` does not
            # bound process RSS, since GDAL's native cache is separate from
            # them (see `spatialrisk.gdal_env.sampling_gdal_env`).
            with _sampling_slot, sampling_gdal_env():
                sample.generate()
            p.add_sample(sample, auto_save=True)
            publish_if_current(project_reactive, p)
            _update_job(
                job_id,
                status="completed",
                n_total=sample.n_total,
                class_counts=sample.class_counts,
            )
            logger.info("Sample generated: %s (%d points)", name, sample.n_total)
    except Exception as exc:
        logger.exception("Sampling failed for %s", name)
        _update_job(job_id, status="failed", error=str(exc))


@solara.component
def SamplingTile(project, map_=None):
    """Sampling tab: generate persistent samples and add them to the map."""
    p = project.value
    raster_keys = (
        sorted(
            k
            for k, v in p.processed_variables.items()
            if getattr(v, "data_type", None) == "raster"
        )
        if p
        else []
    )

    # All hooks are called unconditionally before any early return so the hook
    # count is stable across renders (this tile is gated, so it renders before
    # raster variables exist and then again once they do).
    pending_remove, set_pending_remove = solara.use_state(None)
    details_key, set_details_key = solara.use_state(None)
    dialog_open = solara.use_reactive(False)
    notifications = use_notifications()

    if p is None:
        return
    if not raster_keys:
        solara.Info(t("tiles.sampling.error_no_raster"))
        return

    # Names already taken: persisted samples plus still-running jobs (a sample
    # registers asynchronously inside the worker, so p.samples lags a click).
    existing_names = frozenset(p.samples)
    running_names = frozenset(
        j["name"] for j in sampling_jobs.value if j["status"] == "running"
    )

    def _do_remove(key):
        if map_ is not None and key in samples_on_map.value:
            _remove_sample_layers(map_, _sample_layer_key(key))
            with samples_on_map_lock:
                samples_on_map.set(samples_on_map.value - {key})
        cur = project.value
        if cur is not None and key in cur.samples:
            cur.delete_sample(key, auto_save=True)
            project.set(cur.model_copy())

    def on_submit(entry):
        nm = entry["name"]
        if entry["replace"]:
            # Confirmed in the dialog: drop the superseded sample (and its
            # map layer) before regenerating under the same name.
            _do_remove(nm)
        job_id = str(uuid.uuid4())[:8]
        sampling_jobs.set(
            list(sampling_jobs.value)
            + [
                {
                    "id": job_id,
                    "name": nm,
                    "strategy": entry["strategy"],
                    "raster_var_name": entry["raster_var"],
                    "mask_var_name": entry["mask_var"],
                    "status": "running",
                    "error": None,
                    "n_total": None,
                    "class_counts": None,
                }
            ]
        )
        spawn_in_context(
            _run_sampling,
            (
                job_id,
                nm,
                entry["raster_var"],
                entry["mask_var"],
                entry["strategy"],
                entry["allocation"],
                entry["adapt"],
                entry["n_samples"],
                entry["spacing_m"],
                entry["seed"],
                project,
                notifications,
                t("notifications.task_sampling", name=nm),
            ),
        )
        logger.info(
            "Sampling started: %s (raster=%s, job=%s)", nm, entry["raster_var"], job_id
        )

    def on_toggle_map(key):
        if map_ is None:
            return
        cur = project.value
        if cur is None:
            return
        ss = cur.samples.get(key)
        if ss is None:
            return
        turn_on = key not in samples_on_map.value
        if (
            turn_on
            and ss.points_path is None
            and getattr(ss, "pmtiles_path", None) is None
        ):
            return
        if not samples_pending.claim(key):  # idempotent: ignore re-clicks
            return
        try:
            spawn_in_context(_toggle_sample_on_map, (key, project, map_, turn_on))
        except Exception:
            samples_pending.release(key)
            logger.exception("could not start the map-toggle worker")

    def on_dismiss(job_id):
        # Failed job rows only — never touches the sample registry.
        sampling_jobs.set([j for j in sampling_jobs.value if j["id"] != job_id])

    with solara.Column(style="gap: 16px;"):
        with solara.Row(style="gap:4px;align-items:center;"):
            solara.Text(t("tiles.sampling.description"))
            InfoButton(t("tiles.sampling.info_header"), t("tiles.sampling.info_md"))

        solara.Button(
            t("tiles.sampling.new_button"),
            icon_name="mdi-plus",
            color="primary",
            small=True,
            block=True,
            on_click=lambda: dialog_open.set(True),
        )

        SampleSetList(
            project=project,
            sampling_jobs=sampling_jobs,
            on_map=frozenset(samples_on_map.value),
            on_toggle_map=on_toggle_map,
            on_remove=set_pending_remove,
            on_dismiss=on_dismiss,
            pending=frozenset(samples_pending.value),
            on_open=set_details_key,
        )

        ConfirmDialog(
            open=pending_remove is not None,
            on_cancel=lambda: set_pending_remove(None),
            on_confirm=lambda: (_do_remove(pending_remove), set_pending_remove(None)),
            title=t("tiles.sampling.confirm_remove_title"),
            message=t(
                "tiles.sampling.confirm_remove_message", name=pending_remove or ""
            ),
            confirm_label=t("tiles.sampling.confirm_remove_label"),
        )

    SampleFormDialog(
        project=project,
        open_=dialog_open,
        existing_names=existing_names,
        running_names=running_names,
        on_submit=on_submit,
    )

    SampleDetailsDialog(
        project=project,
        sample_key=details_key,
        on_close=lambda: set_details_key(None),
    )
