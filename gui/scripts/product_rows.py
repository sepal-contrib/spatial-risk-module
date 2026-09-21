"""Render-time row builders: registry products + session-job overlay.

Solara-free (architecture contract #7): pure transforms over the Project
document and the module-level job lists. Each builder returns plain dicts the
tab widgets turn into ProductTable rows (callbacks are attached in the tiles).

Suppression contract: workers never remove their own job rows (the job lists
are unsynchronized read-copy-set reactives). Instead a *completed* job row is
dropped here, at render time, ONLY when its registered product is present in
the project snapshot being rendered. A completed job whose registration didn't
land keeps rendering, so a run can never silently vanish.

These builders replace gui/scripts/job_restore.py (deleted): products now
derive from the registries at render time instead of being mirrored into fake
"completed" job dicts at load time.
"""

from typing import Any, Dict, List, Optional


def _active_jobs_first(jobs: Optional[List[dict]]) -> List[dict]:
    """Session jobs, newest first (they sit above the product rows)."""
    return list(reversed(jobs or []))


# --- Train ------------------------------------------------------------------


def train_rows(
    project: Any,
    jobs: Optional[List[dict]],
    model_labels: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Job rows (newest first) then one row per registered model."""
    labels = model_labels or {}
    models = (getattr(project, "models", None) or {}) if project is not None else {}

    rows: List[dict] = []
    for job in _active_jobs_first(jobs):
        if job.get("status") == "completed":
            key = job.get("model_storage_key")
            if key and key in models:
                continue  # superseded by its product row below
        rows.append(
            {
                "kind": "job",
                "key": f"job_{job['id']}",
                "job_id": job["id"],
                "name": job.get("model_name") or job.get("model_label", "—"),
                "model_label": job.get("model_label", job.get("model_type", "—")),
                "dataset_name": job.get("dataset_name", "—"),
                "sample_name": job.get("sample_name") or "—",
                "status": job.get("status", "running"),
                "error": job.get("error"),
            }
        )
    for key, model in models.items():
        model_type = getattr(model, "model_type", "") or ""
        rows.append(
            {
                "kind": "model",
                "key": key,
                "name": key,
                "model_label": labels.get(model_type, model_type),
                "dataset_name": getattr(model, "dataset_name", None) or "—",
                "sample_name": getattr(model, "sample_name", None) or "—",
                "status": "ready",
                "error": None,
            }
        )
    return rows


# --- Inference ---------------------------------------------------------------


def prediction_row_key(pred: Any) -> str:
    """Row identity: the user-chosen name, else the legacy provenance token."""
    name = getattr(pred, "name", None)
    if name:
        return name
    model_key = getattr(pred, "model_key", None) or "—"
    dataset_name = getattr(pred, "dataset_name", None) or "—"
    return f"{model_key}__{dataset_name}"


def job_row_key(job: dict) -> str:
    """Row key a session inference job maps to (mirrors prediction_row_key)."""
    if job.get("pred_name"):
        return job["pred_name"]
    return f"{job.get('model_key', '—')}__{job.get('dataset_name', '—')}"


def prediction_groups(project: Any) -> Dict[str, dict]:
    """Registered predictions grouped by row key (multi-raster runs share one)."""
    predictions = (
        (getattr(project, "predictions", None) or {}) if project is not None else {}
    )
    groups: Dict[str, dict] = {}
    for storage_key, pred in predictions.items():
        row_key = prediction_row_key(pred)
        group = groups.setdefault(
            row_key,
            {
                "row_key": row_key,
                "name": getattr(pred, "name", None),
                "model_key": getattr(pred, "model_key", None) or "—",
                "dataset_name": getattr(pred, "dataset_name", None) or "—",
                "storage_keys": [],
            },
        )
        group["storage_keys"].append(storage_key)
    return groups


def inference_rows(project: Any, jobs: Optional[List[dict]]) -> List[dict]:
    """Job rows (newest first) then one row per registered prediction group."""
    groups = prediction_groups(project)

    rows: List[dict] = []
    for job in _active_jobs_first(jobs):
        if job.get("status") == "completed" and job_row_key(job) in groups:
            continue  # superseded by its product row below
        rows.append(
            {
                "kind": "job",
                "key": f"job_{job['id']}",
                "job_id": job["id"],
                "name": job.get("pred_name") or job.get("model_key", "—"),
                "model_key": job.get("model_key", "—"),
                "dataset_name": job.get("dataset_name", "—"),
                "storage_keys": [],
                "status": job.get("status", "running"),
                "error": job.get("error"),
                # The submission entry the job was launched from, so a failed
                # row can reopen the Predict dialog prefilled for a rerun.
                "entry": job.get("entry"),
            }
        )
    for row_key, group in groups.items():
        rows.append(
            {
                "kind": "prediction",
                "key": row_key,
                "name": group["name"] or row_key,
                "model_key": group["model_key"],
                "dataset_name": group["dataset_name"],
                "storage_keys": list(group["storage_keys"]),
                "status": "ready",
                "error": None,
            }
        )
    return rows


# --- Sampling -----------------------------------------------------------------

#: Class counts shown before the label collapses into a "+N more" suffix. A
#: continuous variable used as strata (altitude, distance…) yields one class per
#: distinct pixel value, which would otherwise render an unbounded string.
MAX_DISPLAYED_STRATA = 10


def _class_sort_key(item):
    """Numeric order for numeric class keys, string order for anything else."""
    key = item[0]
    try:
        return (0, float(key), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(key))


def format_sample_points(
    n_total: Optional[int],
    class_counts: Optional[Dict[str, int]],
    strategy: Optional[str],
    more_fmt: str = "+{n} more",
) -> str:
    """Render the Points cell for a sample set.

    Only stratified sets break the total down by class — for random/systematic
    the pixel value a point happens to land on carries no meaning. ``more_fmt``
    is a caller-supplied (translated) template taking ``{n}``.
    """
    if n_total is None:
        return "—"
    if str(strategy or "").lower() != "stratified" or not class_counts:
        return str(n_total)

    items = sorted(class_counts.items(), key=_class_sort_key)
    parts = [f"{k}:{v}" for k, v in items[:MAX_DISPLAYED_STRATA]]
    hidden = len(items) - len(parts)
    if hidden > 0:
        parts.append(more_fmt.format(n=hidden))
    return f"{n_total} ({', '.join(parts)})"


ACTIVE_SAMPLING_STATUSES = frozenset({"running", "tiling"})
"""Sampling job statuses that are still in flight.

``running`` while the points are drawn, ``tiling`` once they are on disk and
the map archive is being built (which on SEPAL used to take far longer than
the draw itself). Both keep the name reserved and hide the dismiss action.
"""


def sample_rows(project: Any, jobs: Optional[List[dict]]) -> List[dict]:
    """Job rows (newest first) then one row per registered sample set."""
    samples = (getattr(project, "samples", None) or {}) if project is not None else {}

    rows: List[dict] = []
    for job in _active_jobs_first(jobs):
        if job.get("status") == "completed" and job.get("name") in samples:
            continue  # superseded by its product row below
        rows.append(
            {
                "kind": "job",
                "key": f"job_{job['id']}",
                "job_id": job["id"],
                "name": job.get("name", "—"),
                "strategy": job.get("strategy", "—"),
                "allocation": None,
                "n_total": job.get("n_total"),
                "class_counts": job.get("class_counts"),
                "status": job.get("status", "running"),
                "error": job.get("error"),
            }
        )
    for key, s in samples.items():
        rows.append(
            {
                "kind": "sample",
                "key": key,
                "name": key,
                "strategy": getattr(s, "strategy", "—"),
                "allocation": getattr(s, "allocation", None),
                "n_total": getattr(s, "n_total", None),
                "class_counts": getattr(s, "class_counts", None) or {},
                "status": "ready",
                "error": None,
            }
        )
    return rows


# --- Evaluation ---------------------------------------------------------------


def evaluation_tab_rows(project: Any, jobs: Optional[List[dict]]) -> List[dict]:
    """Job rows (newest first) then saved records, newest first."""
    evaluations = (
        (getattr(project, "evaluations", None) or {}) if project is not None else {}
    )

    rows: List[dict] = []
    for job in _active_jobs_first(jobs):
        if job.get("status") == "completed" and any(
            getattr(rec, "run_id", None) == job["id"] for rec in evaluations.values()
        ):
            continue  # superseded by its saved record below
        rows.append(
            {
                "kind": "job",
                "key": f"job_{job['id']}",
                "job_id": job["id"],
                "truth_tag": job.get("truth_tag", "—"),
                "n_maps": job.get("n_maps", 0),
                "created_at": job.get("created_at", "—"),
                "status": job.get("status", "running"),
                "error": job.get("error"),
            }
        )
    records = sorted(
        evaluations.items(),
        key=lambda kv: getattr(kv[1], "created_at", "") or "",
        reverse=True,
    )
    for key, rec in records:
        rows.append(
            {
                "kind": "evaluation",
                "key": key,
                "truth_tag": getattr(rec, "truth_tag", key),
                "n_maps": len(getattr(rec, "prediction_keys", None) or []),
                "created_at": getattr(rec, "created_at", "") or "—",
                "status": "ready",
                "error": None,
            }
        )
    return rows


# --- Derived layers -----------------------------------------------------------


def derived_rows(
    project: Any,
    jobs: Optional[List[dict]],
    keys: Optional[List[str]] = None,
) -> List[dict]:
    """Job rows (newest first) then one row per registered derived variable.

    ``keys`` restricts the *product* rows to those registry keys (None = all);
    the Post-process tab passes ``postprocess_output_keys`` so the harmonization
    outputs stay out of the derived list. Job rows are never filtered — their
    output is not in the registry yet, so it cannot appear in ``keys``.

    A completed job is suppressed by its ``output_key``, never by its display
    name: ``add_as_processed`` stores a year-bearing variable under
    ``{name}_{year}``, and an edge/dist output inherits its source's year, so
    ``forest`` (2010) -> dist registers ``forest_dist_2010`` under the name
    ``forest_dist``.
    """
    processed = (
        (getattr(project, "processed_variables", None) or {})
        if project is not None
        else {}
    )

    rows: List[dict] = []
    for job in _active_jobs_first(jobs):
        if job.get("status") == "completed" and job.get("output_key") in processed:
            continue  # superseded by its product row below
        rows.append(
            {
                "kind": "job",
                "key": f"job_{job['id']}",
                "job_id": job["id"],
                "name": job.get("name", "—"),
                "status": job.get("status", "running"),
                "error": job.get("error"),
            }
        )
    for key, var in processed.items():
        if keys is not None and key not in keys:
            continue
        rows.append(
            {
                "kind": "variable",
                "key": key,
                "name": getattr(var, "name", key),
                "status": "ready",
                "error": None,
            }
        )
    return rows
