# gui/tile/evaluation_helpers.py
"""Pure (solara-free) helpers for the Step 7 evaluation tile.

Build dropdown items, locate the default forest variable, parse the interval,
and resolve user selections into kwargs for
``spatialrisk.evaluation.evaluate_against_truth``. Kept import-light so the
logic is unit-testable without a running GUI.
"""

import json
import re
import shutil
from pathlib import Path

from spatialrisk.evaluation import (
    interval_from_target,
    run_label_for,
    run_output_dir,
)

# Accuracy indices produced by validate_two_layer; value = column key in the
# stored ``indices`` table, text = display label. All four are always computed.
ALL_METRICS = ["MedAE", "R2", "RMSE", "wRMSE"]
_METRIC_LABELS = {"R2": "R²"}

# "df.attrs has no run_id at all", as distinct from "df.attrs says the run had
# no run id" — see build_evaluation_record.
_NO_RUN_ID_ATTR = object()


def metric_items():
    """[{text, value}] for the metric selector — one per accuracy index."""
    return [{"text": _METRIC_LABELS.get(m, m), "value": m} for m in ALL_METRICS]


def parse_csizes(text):
    """Parse a cell-size field into a list of positive ints.

    Accepts comma- and/or space-separated values (e.g. "100, 300 1000").
    Duplicates are dropped, preserving first-seen order. Returns
    (sizes, error): on success error is None; on failure sizes is None and
    error is a user-facing message.
    """
    tokens = [t for t in re.split(r"[,\s]+", (text or "").strip()) if t]
    if not tokens:
        return None, "Enter at least one cell size (pixels)."
    sizes = []
    for tok in tokens:
        try:
            val = int(tok)
        except (TypeError, ValueError):
            return None, "Cell size(s) must be whole numbers."
        if val <= 0:
            return None, "Cell size(s) must be positive."
        if val not in sizes:
            sizes.append(val)
    return sizes, None


def rows_for_record(record):
    """Display rows for an EvaluationRecord, tolerant of legacy/stale instances.

    Reads ``indices``/``metrics`` defensively so records created before the
    ``metrics`` field existed (or stale in-memory instances after a hot-reload)
    fall back to showing all columns instead of raising AttributeError.
    """
    indices = getattr(record, "indices", None) or []
    return displayed_indices(indices, getattr(record, "metrics", None))


def displayed_indices(indices, metrics):
    """Drop unselected metric columns from stored index rows for display.

    ``metrics`` is the list of metric keys to keep; an empty/falsy list keeps
    every column (legacy runs predate metric selection). Non-metric context
    columns (model, period, ncell, cell-size, …) are always kept.
    """
    if not metrics:
        return indices
    drop = {m for m in ALL_METRICS if m not in metrics}
    return [{k: v for k, v in row.items() if k not in drop} for row in indices]


def variable_items(project):
    """[{text, value}] for each processed-variable instance (value = storage key)."""
    if project is None or not getattr(project, "processed_variables", None):
        return []
    items = []
    for key, var in project.processed_variables.items():
        year = getattr(var, "year", None)
        text = f"{var.name} ({year})" if year is not None else var.name
        items.append({"text": text, "value": key})
    return sorted(items, key=lambda d: d["text"])


def map_items(project):
    """[{text: '<MODEL · run> — <dataset>', value: key}], one per prediction."""
    if project is None or not getattr(project, "predictions", None):
        return []
    items = [
        {"text": f"{run_label_for(pred)} — {pred.dataset_name}", "value": key}
        for key, pred in project.predictions.items()
    ]
    return sorted(items, key=lambda d: d["text"])


def default_forest_key(project):
    """Storage key of the first Hansen forest instance, or None.

    Matches any variable whose name resolves to the ``forest_gfc`` catalogue
    key — the bare legacy ``forest_gfc`` and the parameterised
    ``forest_gfc_tc30`` alike, since the tree-cover threshold is baked into the
    variable name. Returns the first match in insertion order.
    """
    # Imported inside the function to keep this module import-light (the
    # catalogue pulls in earthengine-api at import time).
    from gui.scripts.predefined_variables import resolve_predefined

    if project is None or not getattr(project, "processed_variables", None):
        return None
    for key, var in project.processed_variables.items():
        if resolve_predefined(var.name)[0] == "forest_gfc":
            return key
    return None


def expected_forest_for_truth(truth_name):
    """``(variable name, year)`` of the forest-at-start layer a target pairs with.

    Interval targets declare ``forest_at_start`` in the catalogue:
    ``deforestation_gfc_tc30_2015_2020`` -> ``("forest_gfc_tc30", 2015)``,
    ``degradation_tmf_2015_2020`` -> ``("forest_tmf", 2015)``. None for any
    other name (Process-step loss layers, custom truths).
    """
    from gui.scripts.predefined import build_predefined_name, entry_for_name

    _key, params, entry = entry_for_name(truth_name or "")
    pair = (entry or {}).get("forest_at_start")
    if pair is None or params.get("start_year") is None:
        return None
    forest_key, forest_params = pair(params)
    return build_predefined_name(forest_key, forest_params), int(params["start_year"])


def _paired_forest(project, truth_key):
    """(expected (name, year) or None, matching storage key or None)."""
    pv = getattr(project, "processed_variables", None) or {}
    truth = pv.get(truth_key) if truth_key else None
    expected = expected_forest_for_truth(truth.name) if truth is not None else None
    if expected is None:
        return None, None
    name, year = expected
    for key, var in pv.items():
        if var.name == name and getattr(var, "year", None) == year:
            return expected, key
    return expected, None


def forest_key_for_truth(project, truth_key):
    """Storage key of the processed layer an interval target pairs with, or None."""
    return _paired_forest(project, truth_key)[1]


def missing_forest_for_truth(project, truth_key):
    """``(name, year)`` the user must add and process, or None when present/N/A."""
    expected, key = _paired_forest(project, truth_key)
    return expected if key is None else None


def parse_interval(project, truth_key):
    """Interval (int) parsed from the truth variable's name, or None."""
    if project is None or not truth_key:
        return None
    pv = getattr(project, "processed_variables", None)
    if not pv:
        return None
    var = pv.get(truth_key)
    if var is None:
        return None
    return interval_from_target(var.name)


def build_truth_spec(project, truth_key, forest_key, interval):
    """Resolve selections into evaluate_against_truth kwargs.

    Returns (spec, error). On success spec is a dict with keys
    defor_file, forest_file, time_interval, truth_tag and error is None.
    On failure spec is None and error is a user-facing message.
    """
    if not truth_key:
        return None, "Select a truth (deforestation) variable."
    if not forest_key:
        return None, "Select a forest-at-start variable."
    if forest_key == truth_key:
        return None, (
            "The forest-at-start layer must be a forest map, not the truth layer "
            "itself."
        )
    if interval is None or not str(interval).strip():
        return None, (
            "The interval (years) could not be read from the truth variable's "
            "name: enter the number of years it covers."
        )
    try:
        ti = int(interval)
    except (TypeError, ValueError):
        return None, "Interval (years) must be a whole number."
    if ti <= 0:
        return None, "Interval (years) must be a positive number."
    truth_var = project.processed_variables.get(truth_key)
    forest_var = project.processed_variables.get(forest_key)
    if truth_var is None or forest_var is None:
        return None, "Selected variable no longer exists in the project."
    year = getattr(truth_var, "year", None)
    tag = f"{truth_var.name}_{year}" if year is not None else truth_var.name
    return {
        "defor_file": truth_var.path,
        "forest_file": forest_var.path,
        "time_interval": ti,
        "truth_tag": tag,
    }, None


def build_evaluation_record(
    project, df, spec, resolved_keys, run_id, created_at, csizes=(300,), metrics=None
):
    """Build an EvaluationRecord from a result DataFrame and the run's truth spec.

    ``indices`` is materialized via ``df.to_json`` so values are JSON-native
    (numpy scalars/NaN become float/None), keeping ``Project.save()``'s
    ``json.dumps`` happy. ``metrics`` records which accuracy-index columns the
    user chose to show (empty = all).

    The record's run id is taken from ``df.attrs["run_id"]`` whenever the key is
    PRESENT (i.e. the frame came back from ``evaluate_against_truth``), falling
    back to the ``run_id`` parameter only when it is absent (plain/synthetic
    DataFrames, e.g. in tests, whose ``attrs`` is empty). Preferring the
    DataFrame's own attr is deliberate: it is the run id that was ACTUALLY
    used to choose the directory the artifacts were written into, so
    ``csv_path`` (derived from it here) and ``artifacts`` (also read off
    ``df.attrs``) can never disagree with each other — even if a caller passes
    a ``run_id`` that doesn't match the one the DataFrame was produced with.

    The presence test needs a sentinel rather than ``is None``, because
    ``evaluate_against_truth`` writes ``attrs["run_id"] = run_id``
    unconditionally, ``None`` included. Treating that ``None`` as "the frame
    carries nothing" would let the parameter win for a run that was NOT
    run-scoped — writing its artifacts into the shared truth folder while
    ``csv_path`` pointed at a run directory that was never created. Not
    reachable from the GUI (which passes the same id to both calls), but it is
    the exact disagreement the paragraph above promises cannot happen.

    A frame that declares an UNSCOPED run (``attrs["run_id"] is None``) has no
    record to build: its artifacts live in the shared truth folder, and
    ``EvaluationRecord.run_id`` is a required string, so the layout the frame
    describes is not representable. That combination raises ``ValueError`` here
    rather than silently choosing one of the two answers — the caller passed a
    run id to one function and not the other, and only the caller knows which
    was meant.

    ``csv_path`` points at THIS run's aggregate CSV inside
    ``evaluation/<truth_tag>/<run_id>/``, resolved through ``run_output_dir``
    (the one place that layout is defined) so the existing
    ``Path(csv_path).parent`` figure-directory derivation keeps resolving to
    the run's own PNGs. ``artifacts`` comes from ``df.attrs`` (populated by
    ``evaluate_against_truth`` when it is given a run id) and stays empty when
    the run was not run-scoped.
    """
    from spatialrisk.evaluations import EvaluationRecord

    truth_tag = spec["truth_tag"]
    indices = json.loads(df.to_json(orient="records"))
    attrs = getattr(df, "attrs", None) or {}
    actual_run_id = attrs.get("run_id", _NO_RUN_ID_ATTR)
    if actual_run_id is _NO_RUN_ID_ATTR:
        actual_run_id = run_id
    elif actual_run_id is None:
        raise ValueError(
            "this DataFrame came from an evaluation run WITHOUT a run id, so "
            "its artifacts are in the shared evaluation/<truth_tag>/ folder; "
            f"a record scoped to run_id={run_id!r} would point at a directory "
            "that was never written. Pass the same run_id to "
            "evaluate_against_truth and to build_evaluation_record."
        )
    csv_path = str(
        run_output_dir(project, truth_tag, actual_run_id) / "indices_all.csv"
    )
    artifacts = list(attrs.get("artifacts") or [])
    return EvaluationRecord(
        name=truth_tag,
        truth_tag=truth_tag,
        truth_defor=str(spec["defor_file"]),
        truth_forest=str(spec["forest_file"]),
        time_interval=int(spec["time_interval"]),
        prediction_keys=list(resolved_keys),
        csizes=list(csizes),
        metrics=list(metrics) if metrics else [],
        created_at=created_at,
        indices=indices,
        csv_path=csv_path,
        run_id=actual_run_id,
        artifacts=artifacts,
    )


def run_artifact_dir(project, record):
    """This run's own artifact directory, or None if it has none on disk.

    Returns ``evaluation/<truth_tag>/<run_id>`` when that directory exists.
    Legacy records (saved before run-scoping) wrote straight into the shared
    ``evaluation/<truth_tag>/`` folder and therefore have no run directory —
    they get None, which is what keeps their files out of any cleanup. A
    record with a path-like stored ``truth_tag``/``run_id`` makes
    ``run_output_dir`` raise ``ValueError`` (containment check); on this
    read path that also becomes None rather than propagating.
    """
    if project is None or record is None:
        return None
    truth_tag = getattr(record, "truth_tag", None)
    run_id = getattr(record, "run_id", None)
    if not truth_tag or not run_id:
        return None
    try:
        candidate = run_output_dir(project, str(truth_tag), str(run_id))
    except ValueError:
        return None
    return candidate if candidate.is_dir() else None


def delete_run_artifacts(project, record):
    """Remove ONE run's artifact directory. Returns True if it was removed.

    Guarded twice over: the resolved target must sit strictly inside the
    project's ``evaluation/`` folder, and its final path component must be the
    record's run id. That makes it impossible to remove the shared
    ``evaluation/<truth_tag>/`` folder (whose legacy files must stay
    recoverable), a sibling run, or anything outside the project.

    Call ONLY after the project manifest commit has succeeded — see
    ``delete_evaluation_run``.
    """
    target_dir = run_artifact_dir(project, record)
    if target_dir is None:
        return False
    try:
        target = target_dir.resolve()
        eval_root = (Path(project.folders.project_folder) / "evaluation").resolve()
    except (OSError, RuntimeError):
        return False
    if eval_root not in target.parents or target.name != str(record.run_id):
        return False
    shutil.rmtree(target, ignore_errors=True)
    return not target.exists()


def delete_evaluation_run(project, key):
    """Delete a saved evaluation: registry entry, then commit, THEN artifacts.

    Deletion is a transaction. The registry entry is removed without
    autosaving, the manifest is committed, and only after a successful commit
    is the run's artifact directory unlinked — a failed save can never leave a
    record pointing at files that are already gone. When the save DOES fail,
    the complete pre-deletion registry snapshot is restored on the live
    project (the whole dict, not a re-insert, so ordering is preserved):
    nothing stays mutated in memory, and a LATER successful save cannot
    silently persist the failed deletion.

    Returns ``(deleted, error)``: ``(True, None)`` on success; ``(False,
    None)`` when *key* is not registered; ``(False, message)`` when the
    manifest save failed (registry restored, artifacts kept on disk).
    """
    record = project.get_evaluation(key)
    if record is None:
        return False, None
    snapshot = dict(project.evaluations)
    project.delete_evaluation(key, auto_save=False)
    try:
        project.save()
    except Exception as exc:  # broad: roll back; artifacts must survive
        project.evaluations = snapshot
        return False, str(exc)
    delete_run_artifacts(project, record)
    return True, None
