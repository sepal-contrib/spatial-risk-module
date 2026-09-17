"""Bridge between the allocation tool's UI and the numeric core.

Solara-free by contract (this module is imported by tests without a render
harness); heavy geo dependencies are imported lazily inside functions.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from gui.scripts.prediction_import import IMPORT_DATASET_NAME

logger = logging.getLogger("spatial_risk")

#: Families whose apply() writes a per-category rate table next to the raster.
_MW_FAMILY = "mw"
_JNR_FAMILY = "jnr"


def _family(model_key) -> str:
    """First token of a model key — same derivation as run_inference's.

    Imported predictions must be handled before this is consulted: an import
    named ``mw_x`` is not a moving-window run.
    """
    return str(model_key or "").split("_")[0]


_JNR_CAVEAT = (
    "This table's rates are the deforestation observed in the prediction's own "
    "period (the app applies the JNR benchmark without a model-period rate table). "
    "Override it if you have a calibration/historical table."
)


class AllocationResolveError(ValueError):
    """Raised when the rate table for a prediction cannot be resolved."""


@dataclass
class DefrateSource:
    """Where an allocation run's rate table came from."""

    path: Optional[Path]
    provenance: str  # "persisted" | "mw-sibling" | "computed" | "user" | "evaluation"
    caveat: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Serializable form, stored on the AllocationRun record."""
        return {
            "path": str(self.path) if self.path else None,
            "provenance": self.provenance,
            "caveat": self.caveat,
        }


def _resolve_layers(project, pred):
    """Indirection so tests can stub the evaluation resolver."""
    from spatialrisk.evaluation import resolve_layers

    return resolve_layers(project, pred)


def _defrate_per_cat(**kwargs):
    """Indirection so tests can stub the (slow, GDAL-bound) rate computation."""
    from spatialrisk.rmj.deforrate import defrate_per_cat

    return defrate_per_cat(**kwargs)


def resolve_defrate_table(
    project,
    pred_key: str,
    *,
    user_path: Optional[Path] = None,
    compute: bool = True,
) -> DefrateSource:
    """Find (or compute) the per-category rate table for a registered prediction.

    Order: explicit user override → for an imported map, the newest evaluation
    run that included it → the table persisted on the Prediction → the MW
    sibling-path convention → computed from the prediction's dataset. An MW or
    JNR prediction whose table is missing raises rather than recomputing it.

    The import branch comes before every family check on purpose: an import
    named ``mw_...`` or ``jnr_...`` must never be routed by ``_family`` into a
    resolver that expects an app-produced run layout.
    """
    if user_path:
        return DefrateSource(path=Path(user_path), provenance="user")

    pred = (project.predictions or {}).get(pred_key)
    if pred is None:
        raise AllocationResolveError(
            f"Prediction '{pred_key}' not found in this project."
        )

    if pred.dataset_name == IMPORT_DATASET_NAME:
        return _defrate_from_evaluation(project, pred_key, pred)

    persisted = getattr(pred, "defrate_path", None)
    if persisted and Path(persisted).exists():
        caveat = _JNR_CAVEAT if _family(pred.model_key) == _JNR_FAMILY else None
        return DefrateSource(
            path=Path(persisted), provenance="persisted", caveat=caveat
        )

    if _family(pred.model_key) == _MW_FAMILY:
        sibling = Path(pred.path).parent / (
            f"defrate_cat_mw_{pred.window}_{pred.dataset_name}.csv"
        )
        if sibling.exists():
            return DefrateSource(path=sibling, provenance="mw-sibling")
        raise AllocationResolveError(
            f"No rate table for moving-window prediction '{pred_key}': expected "
            f"{sibling.name} beside the raster. Re-run inference or select a table "
            "manually."
        )

    if _family(pred.model_key) == _JNR_FAMILY:
        raise AllocationResolveError(
            f"No rate table recorded for JNR prediction '{pred_key}'. Re-run "
            "inference or select a table manually."
        )

    out = (
        Path(pred.path).parent / f"defrate_cat_{pred.model_key}_{pred.dataset_name}.csv"
    )
    if out.exists():
        return DefrateSource(path=out, provenance="computed")
    if not compute:
        raise AllocationResolveError(
            f"No rate table available for '{pred_key}' and computing is disabled."
        )

    layers = _resolve_layers(project, pred)
    if not layers.get("time_interval"):
        raise AllocationResolveError(
            "Cannot determine the period length from the dataset's target name "
            "(expected two 4-digit years, e.g. 'forest_loss_2015_2020'). Select a "
            "rate table manually."
        )
    logger.info("Computing rate table for '%s' → %s", pred_key, out.name)
    _defrate_per_cat(
        defor_file=layers["defor_file"],
        forest_file=layers["forest_file"],
        riskmap_file=layers["riskmap_file"],
        time_interval=layers["time_interval"],
        tab_file_defrate=out,
        verbose=False,
    )
    return DefrateSource(path=out, provenance="computed")


def _defrate_from_evaluation(project, pred_key: str, pred) -> DefrateSource:
    """Rate table of the newest evaluation run that scored *pred_key*.

    Imports have no dataset to compute a table from, so the evaluation step is
    the only place their per-category rates exist. Newest record first; a
    record whose file vanished is skipped in favour of an older one. Records
    saved before ``EvaluationPlotArtifact.defrate_csv`` existed fall back to
    the run-scoped path the evaluation would have written.
    """
    from spatialrisk.evaluation import artifact_label_for, run_output_dir

    records = [
        r
        for r in (getattr(project, "evaluations", None) or {}).values()
        if pred_key in (getattr(r, "prediction_keys", None) or [])
    ]
    records.sort(key=lambda r: getattr(r, "created_at", "") or "", reverse=True)
    for record in records:
        path = None
        for art in getattr(record, "artifacts", None) or []:
            if getattr(art, "prediction_key", None) == pred_key and getattr(
                art, "defrate_csv", None
            ):
                path = Path(art.defrate_csv)
                break
        if path is None:
            try:
                path = run_output_dir(project, record.truth_tag, record.run_id) / (
                    f"defrate_cat_{artifact_label_for(pred)}_{pred.dataset_name}.csv"
                )
            except ValueError:
                continue
        if path.exists():
            return DefrateSource(
                path=path,
                provenance="evaluation",
                caveat=(
                    f"Rates from the evaluation against '{record.truth_tag}' "
                    f"({record.created_at})."
                ),
            )
    raise AllocationResolveError(
        "Evaluate this imported map first (it has no dataset to compute a rate "
        "table from), or select a rate table manually."
    )


def preview_defrate_source(
    project, pred_key: Optional[str], *, user_path: Optional[Path] = None
) -> DefrateSource:
    """Form-time preview of ``resolve_defrate_table``: never computes, never raises.

    Same resolution order as the run itself, so what the form shows is what the
    run will use. Provenance ``"unavailable"`` (path None, reason in ``caveat``)
    stands in for the resolver's errors; provenance ``"computed"`` with a
    not-yet-existing path means "will be computed from the dataset".
    """
    if user_path:
        return DefrateSource(path=Path(user_path), provenance="user")
    pred = (getattr(project, "predictions", None) or {}).get(pred_key)
    if pred is None:
        return DefrateSource(
            path=None,
            provenance="unavailable",
            caveat=f"Prediction '{pred_key}' not found in this project.",
        )
    try:
        return resolve_defrate_table(project, pred_key, compute=False)
    except AllocationResolveError as exc:
        if pred.dataset_name == IMPORT_DATASET_NAME or _family(pred.model_key) in (
            _MW_FAMILY,
            _JNR_FAMILY,
        ):
            return DefrateSource(path=None, provenance="unavailable", caveat=str(exc))
        out = (
            Path(pred.path).parent
            / f"defrate_cat_{pred.model_key}_{pred.dataset_name}.csv"
        )
        return DefrateSource(path=out, provenance="computed")


#: Raster suffixes a processed variable must have to be offered as a mask.
_MASK_SUFFIXES = {".tif", ".tiff", ".vrt"}


def mask_items(project) -> List[dict]:
    """Mask choices for the form: the project's processed raster variables.

    ``[{"text": <variable name>, "value": <raster path>}]``, name-sorted.
    Vector variables are skipped — the allocation core wants a raster aligned
    with the risk map (it warps it to the cropped grid itself).
    """
    items = []
    for key, var in (getattr(project, "processed_variables", None) or {}).items():
        path = getattr(var, "path", None)
        if path and Path(path).suffix.lower() in _MASK_SUFFIXES:
            items.append({"text": key, "value": str(path)})
    return sorted(items, key=lambda d: d["text"])


#: Canonical name of the materialized borders file inside a run's out_dir.
BORDERS_FILENAME = "project_borders.gpkg"

#: Every project-borders method the picker can produce. Anything else is
#: rejected by both validate_form and the resolver rather than silently
#: resolving to a default.
_BORDER_METHODS = ("FILE", "ADMIN0", "ADMIN1", "ADMIN2", "ASSET")

_ADMIN_METHODS = ("ADMIN0", "ADMIN1", "ADMIN2")


@dataclass
class BordersSelection:
    """How the user chose the project borders.

    Only FILE names something that already exists; ADMIN and ASSET are intents
    that ``resolve_borders_file`` materializes at run time. No display label
    lives here — pysepal's ``process_admin`` derives the readable name from
    pygaul when the geometry is fetched.
    """

    method: str
    file_path: Optional[str] = None
    admin_code: Optional[str] = None
    asset: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Serializable form, stored on the AllocationRun record."""
        return {
            "method": self.method,
            "file_path": self.file_path,
            "admin_code": self.admin_code,
            "asset": self.asset,
        }


@dataclass
class AllocationForm:
    """User inputs of one allocation run (already parsed from the form widgets).

    The risk map is always one of the project's predictions: external risk
    maps are imported on the inference tab (where they become predictions),
    so the allocation tool never takes a raster file directly.
    """

    name: str
    prediction_key: Optional[str]
    user_defrate_path: Optional[str]
    borders: Optional[BordersSelection]
    mask_file: Optional[str]
    defor_juris_ha: float
    years_forecast: float
    #: None (no raster) | "project" (cropped extent) | "aoi" (whole risk map).
    density_extent: Optional[str] = None


def _allocate(**kwargs):
    """Indirection so tests can stub the (slow, GDAL-bound) core."""
    from spatialrisk.allocation import allocate_deforestation

    return allocate_deforestation(**kwargs)


def _validate_borders(borders: Optional[BordersSelection]) -> Optional[str]:
    """Per-method check of the borders selection, or None when it is runnable.

    ADMIN and ASSET cannot be existence-checked here — they are fetched at run
    time — so this only catches an incomplete selection.
    """
    if borders is None or borders.method not in _BORDER_METHODS:
        return "Choose the project borders."
    if borders.method == "FILE":
        if not borders.file_path:
            return "Choose the project borders vector file."
        if not Path(borders.file_path).exists():
            return (
                "The selected project borders file does not exist: "
                f"{borders.file_path}"
            )
        return None
    if borders.method in _ADMIN_METHODS:
        if not borders.admin_code:
            return "Choose the administrative area for the project borders."
        return None
    asset = borders.asset or {}
    if not asset.get("asset_id"):
        return "Choose the Earth Engine asset for the project borders."
    column = asset.get("column")
    if column not in (None, "ALL") and asset.get("value") is None:
        return f"Choose a value for the '{column}' filter on the borders asset."
    return None


#: Geometry types a cutline can be built from.
_POLYGONAL = {"Polygon", "MultiPolygon"}


def _build_asset_fc(asset):
    """Indirection so tests can stub the (EE-bound) strict asset builder."""
    from gui.scripts.aoi_io import build_asset_feature_collection

    return build_asset_feature_collection(asset)


def _ee_export_vector(fc, filename, selectors=None, verbose=True, **kwargs):
    """Indirection so tests can stub the (network-bound) EE export."""
    from spatialrisk.gee.vector_export import ee_export_vector

    return ee_export_vector(
        fc, filename, selectors=selectors, verbose=verbose, **kwargs
    )


def _admin_gdf(method: str, admin_code: str):
    """GAUL boundary for an admin selection, via pysepal's non-GEE WFS path.

    ``process_admin`` builds the AoiResult — including the readable name it
    derives from pygaul's local parquet — and ``get_gdf_async`` fetches the
    geometry from the FAO GAUL WFS. ``gee=False`` keeps Earth Engine out of the
    most common case entirely; the GEE path would hand back a lazy
    FeatureCollection that still needed exporting.

    Worker-thread only: ``asyncio.run`` fails inside a running loop.
    """
    import asyncio

    from pysepal.solara.components.aoi import process_admin

    async def fetch():
        result = await process_admin(method, admin_code, gee=False)
        return await result.get_gdf_async()

    gdf = asyncio.run(fetch())
    if gdf is None:
        raise AllocationResolveError(
            f"Could not fetch the boundary for administrative area {admin_code}."
        )
    return gdf


def _asset_gdf(asset: Optional[Dict[str, Any]], out_dir: Path):
    """Local geometry for a GEE asset selection.

    The EE table download cannot write GPKG (csv/geojson/json/kml/kmz/shp
    only), so the export lands as GeoJSON and geopandas rewrites it to the
    canonical file. ``selectors=[]`` keeps the payload to geometry — a cutline
    needs no attributes — and skips a ``propertyNames().getInfo()`` round-trip.
    """
    import geopandas as gpd
    from pysepal.scripts import utils as su

    su.init_ee()
    fc = _build_asset_fc(asset or {})

    tmp = Path(out_dir) / "project_borders_export.geojson"
    try:
        _ee_export_vector(fc, str(tmp), selectors=[], verbose=False)
        if not tmp.exists():
            raise AllocationResolveError(
                "The Earth Engine borders export produced no file."
            )
        return gpd.read_file(tmp)
    finally:
        # A partial download must not survive a failed or timed-out export:
        # the run directory has no owner until the run finishes registering,
        # so nothing else in the app can ever clean this up.
        tmp.unlink(missing_ok=True)


def _check_borders_gdf(gdf) -> None:
    """Reject geometry the core would only fail on deep inside GDAL.

    A null or empty geometry passes ``gdf.empty`` (the GeoDataFrame still has
    rows) and, when EVERY geometry is null, ``set(gdf.geom_type.dropna())``
    collapses to the empty set — which is a subset of ``_POLYGONAL``, so an
    attribute-only layer used to sail through this check, get written to
    project_borders.gpkg, and only fail later as a confusing gdal.Warp
    cutline error. A layer that mixes real polygons with null/empty rows is
    rejected outright rather than silently thinned: dropping the null rows
    would crop against an incomplete boundary with no indication anything
    was wrong.
    """
    if gdf is None or gdf.empty:
        raise AllocationResolveError(
            "The selected project borders contain no features."
        )
    if gdf.crs is None:
        raise AllocationResolveError(
            "The selected project borders have no CRS, so they cannot be "
            "reprojected onto the risk map."
        )
    usable = gdf.geometry.notna() & ~gdf.geometry.is_empty
    if not usable.any():
        raise AllocationResolveError(
            "The selected project borders have no usable geometry: every "
            "feature is null or empty."
        )
    if not usable.all():
        raise AllocationResolveError(
            "The selected project borders mix real geometry with "
            f"{int((~usable).sum())} null or empty feature(s). Fix the "
            "source data before running."
        )
    kinds = set(gdf.geom_type.unique())
    if not kinds <= _POLYGONAL:
        raise AllocationResolveError(
            "The project borders must be polygons; got "
            f"{', '.join(sorted(kinds))}. A points or lines layer cannot crop "
            "a risk map."
        )


def resolve_borders_file(selection: BordersSelection, out_dir: Path) -> Path:
    """Materialize *selection* as ``<out_dir>/project_borders.gpkg``.

    Every method converges on one canonical file. The allocation core needs a
    vector file on disk (it reprojects it and hands the result to gdal.Warp as
    a cutline), and a run that owns a copy of its geometry stays reproducible
    when the user later edits, moves or deletes what they picked. Rewriting a
    FILE selection also collapses a shapefile's sidecars into one file.

    Worker-thread only: the remote branches drive async pysepal code with
    ``asyncio.run``, which fails inside a thread that already runs a loop.
    """
    import geopandas as gpd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if selection.method == "FILE":
        gdf = gpd.read_file(selection.file_path)
    elif selection.method in _ADMIN_METHODS:
        gdf = _admin_gdf(selection.method, selection.admin_code)
    elif selection.method == "ASSET":
        gdf = _asset_gdf(selection.asset, out_dir)
    else:
        raise AllocationResolveError(
            f"Unknown project-borders method {selection.method!r}."
        )

    # Before the write, so a rejected selection leaves nothing on disk.
    _check_borders_gdf(gdf)

    target = out_dir / BORDERS_FILENAME
    gdf.to_file(target, driver="GPKG")
    return target


def validate_form(project, form: AllocationForm) -> Optional[str]:
    """Return a user-facing error message, or None when the form is runnable."""
    if not (form.name or "").strip():
        return "Give the allocation run a name."
    if not form.prediction_key:
        return "Choose a risk map from the project's predictions."
    if form.years_forecast is None or form.years_forecast <= 0:
        return "The forecast period must be at least one year."
    if form.defor_juris_ha is None or form.defor_juris_ha < 0:
        return "Expected jurisdictional deforestation cannot be negative (hectares)."
    borders_error = _validate_borders(form.borders)
    if borders_error:
        return borders_error
    for label, value in (
        ("rate table", form.user_defrate_path),
        ("mask", form.mask_file),
    ):
        if value and not Path(value).exists():
            return f"The selected {label} file does not exist: {value}"
    return None


def run_allocation(
    project,
    form: AllocationForm,
    project_reactive=None,
    notifier=None,
    jobs_reactive=None,
    job_id: Optional[str] = None,
):
    """Execute one allocation and register it on the project.

    Runs on a worker thread (the GDAL calls release the GIL). Wrapped by the
    caller in ``tracked_job`` + ``writing`` when a notifier is available.
    """
    from spatialrisk.allocations import AllocationRun

    run_id = uuid.uuid4().hex[:8]
    record_key = AllocationRun(
        name=form.name,
        run_id=run_id,
        # Only storage_key() is wanted here, and that uses name + run_id alone.
        # The real borders path is not known until out_dir exists (below).
        borders_file="",
        defor_juris_ha=form.defor_juris_ha,
        years_forecast=form.years_forecast,
        annual_ha=0.0,
        total_ha=0.0,
        out_dir="",
        csv_path="",
    ).storage_key()
    out_dir = Path(project.folders.project_folder) / "allocation" / record_key

    pred = (project.predictions or {}).get(form.prediction_key)
    if pred is None:
        raise AllocationResolveError(
            f"Prediction '{form.prediction_key}' is no longer in this project."
        )
    riskmap_file = pred.path
    source = resolve_defrate_table(
        project, form.prediction_key, user_path=form.user_defrate_path
    )

    # Resolution runs before the core creates out_dir. A failure here — the
    # export can raise after partially writing into out_dir — must not leave
    # an unregistered run folder behind forever: no AllocationRun record
    # exists yet for delete_allocation_run to ever act on. Only remove the
    # directory THIS call created (never one that already existed).
    created_out_dir = not out_dir.exists()
    try:
        borders_path = resolve_borders_file(form.borders, out_dir)
    except Exception:
        if created_out_dir:
            # Best-effort: a cleanup failure (e.g. a stray open file handle)
            # must never replace the real resolution error the user needs to
            # see with a confusing "directory not empty" of its own.
            try:
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                logger.warning(
                    "Could not clean up run directory %s after a failed resolution.",
                    out_dir,
                )
        raise

    logger.info("Allocating deforestation for '%s'…", form.name)
    result = _allocate(
        riskmap_file=riskmap_file,
        defrate_table=source.path,
        defor_juris_ha=form.defor_juris_ha,
        years_forecast=form.years_forecast,
        project_borders=str(borders_path),
        out_dir=out_dir,
        forest_mask_file=form.mask_file,
        defor_density_map=form.density_extent,
    )

    record = AllocationRun(
        name=form.name,
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        prediction_key=form.prediction_key,
        prediction_snapshot={
            "path": str(pred.path),
            "model_key": pred.model_key,
            "dataset_name": pred.dataset_name,
            "year": getattr(pred, "year", None),
            "window": getattr(pred, "window", None),
            "name": getattr(pred, "name", None),
        },
        defrate_source=source.as_dict(),
        borders_file=str(borders_path),
        borders_source=form.borders.as_dict(),
        mask_file=form.mask_file,
        defor_juris_ha=form.defor_juris_ha,
        years_forecast=form.years_forecast,
        annual_ha=result.annual_ha,
        total_ha=result.total_ha,
        out_dir=str(result.out_dir),
        csv_path=str(result.csv_path),
        density_extent=form.density_extent,
        density_map_path=(
            str(result.density_map_path) if result.density_map_path else None
        ),
        warnings=list(result.warnings),
    )
    project.add_allocation(record, auto_save=True)
    if project_reactive is not None:
        from gui.scripts.solara_threads import publish_if_current

        publish_if_current(project_reactive, project)
    return record


def _run_source(record) -> Optional[str]:
    """'<MODEL · run> — <dataset>' the run came from, None if external."""
    snapshot = getattr(record, "prediction_snapshot", None) or {}
    if not snapshot.get("model_key"):
        return None
    from types import SimpleNamespace

    from spatialrisk.evaluation import run_label_for

    pred = SimpleNamespace(
        model_key=snapshot["model_key"],
        window=snapshot.get("window"),
        name=snapshot.get("name"),
    )
    return f"{run_label_for(pred)} — {snapshot.get('dataset_name', '')}"


def allocation_rows(project, jobs=None) -> List[dict]:
    """Rows for the allocation list: in-flight jobs first, then saved runs."""
    saved = (getattr(project, "allocations", None) or {}) if project is not None else {}

    rows: List[dict] = []
    for job in jobs or []:
        # A finished job is superseded by its own record row — otherwise the run
        # renders twice, once as a stale "completed" job with no figures. The
        # key is only checked once it is actually in the registry: the worker
        # flips the status before the project reactive republishes, and a job
        # dropped during that window would blink out of the list entirely.
        if job.get("status") == "completed":
            record_key = job.get("record_key")
            if record_key and record_key in saved:
                continue
        rows.append(
            {
                "kind": "job",
                "key": job.get("id"),
                "job_id": job.get("id"),
                "name": job.get("name"),
                "status": job.get("status"),
                "error": job.get("error"),
                # The AllocationForm the job was launched with, so a failed row
                # can reopen the dialog prefilled. None for jobs launched
                # before edit-on-failure existed.
                "entry": job.get("entry"),
            }
        )
    for key, record in saved.items():
        rows.append(
            {
                "kind": "record",
                "key": key,
                "name": record.name,
                "source": _run_source(record),
                "created_at": record.created_at,
                "annual_ha": record.annual_ha,
                "total_ha": record.total_ha,
                "years_forecast": record.years_forecast,
                "density_map_path": record.density_map_path,
                "warnings": record.warnings,
                "provenance": (record.defrate_source or {}).get("provenance"),
            }
        )
    return rows


def delete_allocation_run(project, key: str) -> bool:
    """Delete a saved allocation: registry entry, commit, THEN artifacts.

    Deletion is a transaction, mirroring evaluation_helpers.delete_evaluation_run:
    a failed manifest save must never leave a record pointing at files that are
    already gone, so the record is restored and the files kept.
    """
    record = (getattr(project, "allocations", None) or {}).get(key)
    if record is None:
        return False
    project.delete_allocation(key)
    try:
        project.save()
    except Exception:
        project.allocations[key] = record
        raise

    run_dir = Path(record.out_dir).resolve()
    allocation_root = (Path(project.folders.project_folder) / "allocation").resolve()
    if run_dir.is_dir() and allocation_root in run_dir.parents and run_dir.name == key:
        shutil.rmtree(run_dir, ignore_errors=True)
    else:
        logger.warning(
            "Refusing to delete '%s': not a run directory under %s",
            run_dir,
            allocation_root,
        )
    return True
