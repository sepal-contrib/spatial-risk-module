"""Assess prediction (``model.apply``) on the machine that matters (SEPAL).

``apply`` now walks the target grid on the shared stripe engine
(:mod:`spatialrisk.mlmodels.windowed_predict`) with a worker count chosen by
:func:`spatialrisk.gdal_env.plan_inference`. On a 16-core dev box the pool
took a synthetic 64 Mpx GLM from 14.96 s to 3.31 s, while the same pool made
a 9 Mpx random forest 21 % *slower* than ``workers=1`` (2026-09-22, task 7).
Those numbers come from a synthetic stack on one machine; this test produces
them on SEPAL, where the cores, the memory reading and the storage all
differ, and on a real project's feature stack, so the per-model default can
be chosen from evidence instead of from a laptop.

What it reports, each configuration in a **fresh subprocess** so peak RSS is
per run and the import baseline is shown separately:

* the resources the planner can see -- affinity cores, the cgroup memory
  limit/usage, psutil's host view and which of them
  :func:`spatialrisk.parallel.free_memory_bytes` ended up using;
* the plan :func:`spatialrisk.gdal_env.plan_inference` makes for this stack
  (workers, rows per stripe, the per-stripe working set and whether cores or
  memory bound it), and the engine's own plan log line from every stage;
* one row per (model, workers): ``workers=1`` is the serial path on the
  calling thread -- the baseline the pool is judged against, and the one
  where a random forest keeps its own ``n_jobs=-1`` -- and every other count
  is the stripe pool. ``auto`` means the policy's own choice.

Every row is hashed: all rows of one model must produce the identical raster,
whatever the worker count.

Run on SEPAL from the module root::

    pytest tests/test_inference_bench_sepal.py -s -k sepal

By default it reads the manifest of ``test_peru_amazonia`` under
``~/module_results/spatial_risk_module`` (only the JSON, no app code), takes
that project's dataset -- its target and feature variables -- and fits a GLM
and a 100-tree random forest on a 5 000-point stratified sample of the
target drawn with :func:`spatialrisk.sampling.service.generate_points` and
extracted with ``Dataset.extract_at_points``, exactly as the app does.

Two deliberate simplifications keep the measurement about the engine:

* every feature enters the formula as a plain numeric term, never
  ``C(<name>)`` or ``scale(<name>)``, so the design width is
  ``n_features + 1`` on any project (what
  :func:`~spatialrisk.gdal_env.plan_inference` assumes by default), no
  categorical level scan reads a whole raster before the clock starts, and a
  layer that is constant over its own domain -- a forest mask is all 1 inside
  the forest -- cannot make ``scale()`` divide by zero and drop every row;
* ``_register_prediction`` is stubbed out, so the overview pyramid it builds
  -- a fixed cost that does not scale with workers -- stays out of the timed
  region. Both match ``benchmarks/inference_bench.py``.

A country target is far too big to predict several times over (a 2.2 Gpx
stack is tens of minutes per row), so the whole stack -- target *and*
features, which the engine reads by the target's own window indices and
which therefore must stay pixel-aligned -- is copied once into a shorter,
co-registered stack of whole 256-row stripes, keeping each source's dtype,
nodata and block layout. The window does not change the plan: the stripe
working set is a function of the raster *width*, so the same worker count is
planned as for the full raster, the run is simply shorter.

Environment knobs (all optional):

``SPATIAL_RISK_BENCH_PROJECT``
    project name, found under ``SPATIAL_RISK_DATA_DIR`` or
    ``~/module_results/spatial_risk_module`` as the app finds it.
``SPATIAL_RISK_BENCH_DATASET``
    dataset key in the manifest (else the only/first one).
``SPATIAL_RISK_BENCH_TARGET_VAR`` / ``SPATIAL_RISK_BENCH_FEATURE_VARS``
    processed-variable names, overriding the dataset (the second is a comma
    list). Needed for a project whose manifest holds no dataset.
``SPATIAL_RISK_BENCH_MODELS``
    comma list of ``glm`` / ``rf`` (default both).
``SPATIAL_RISK_BENCH_WORKERS``
    comma list, ``half`` / ``all`` of the cores and ``auto`` (the policy)
    allowed; default ``1,2,half,all,auto``.
``SPATIAL_RISK_BENCH_MPX`` / ``SPATIAL_RISK_BENCH_ROWS`` /
``SPATIAL_RISK_BENCH_ROW_OFFSET``
    the prediction window: a megapixel budget (default 64, ``0`` = predict
    the whole raster), or an explicit height, and where it starts (default:
    centred, where an AOI-clipped raster actually has data).
``SPATIAL_RISK_BENCH_N_SAMPLES`` / ``SPATIAL_RISK_BENCH_TREES`` /
``SPATIAL_RISK_BENCH_SEED``
    training points (default 5 000), forest size (default 100), seed (42).
``SPATIAL_RISK_BENCH_OUT``
    write the records as JSON (what ``scripts/sepal/compare_bench.py`` reads).

On a machine without the project the SEPAL test is skipped; the smoke test
below runs the same machinery on a small synthetic stack so the file keeps
working between SEPAL sessions.
"""

import hashlib
import json
import logging
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

MODULE_DIR_ENV = "SPATIAL_RISK_MODULE_DIR"


_SKIP_DIRS = {
    "module_results",
    "micromamba",
    "miniconda3",
    "anaconda3",
    "snap",
    "node_modules",
    "site-packages",
    "__pycache__",
}


def _search_home_for_module(max_depth: int):
    """Directories under ``$HOME`` that contain the ``spatialrisk`` package."""
    home = Path.home()
    found = []
    for dirpath, dirnames, _files in os.walk(home):
        depth = len(Path(dirpath).relative_to(home).parts)
        if "spatialrisk" in dirnames and (
            Path(dirpath, "spatialrisk", "mlmodels", "windowed_predict.py").exists()
        ):
            found.append(Path(dirpath))
            dirnames[:] = []
            continue
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in _SKIP_DIRS and depth < max_depth
        ]
    return sorted(found, key=lambda p: len(p.parts))


def _find_module_root() -> Path:
    """Locate the spatial-risk-module checkout so ``spatialrisk`` imports.

    This file is meant to be copied on its own to SEPAL, so it cannot rely on
    living inside the repo. Order: ``spatialrisk`` already importable (an
    installed package, or pytest started from the repo), ``SPATIAL_RISK_MODULE_DIR``,
    the folder this file sits in and its parent, then a search under the home
    directory for a ``spatialrisk`` package (six levels deep, skipping hidden,
    environment and results folders so it stays quick on a full SEPAL home).
    """
    import importlib.util

    if importlib.util.find_spec("spatialrisk") is not None:
        import spatialrisk

        return Path(spatialrisk.__file__).resolve().parents[1]
    here = Path(__file__).resolve().parent
    candidates = []
    if os.environ.get(MODULE_DIR_ENV):
        candidates.append(Path(os.environ[MODULE_DIR_ENV]).expanduser())
    candidates += [here, here.parent]
    candidates += _search_home_for_module(max_depth=6)
    for root in candidates:
        if (root / "spatialrisk" / "mlmodels" / "windowed_predict.py").exists():
            sys.path.insert(0, str(root))
            return root.resolve()
    raise ImportError(
        "cannot find the spatial-risk-module checkout (spatialrisk package); "
        f"set {MODULE_DIR_ENV} to its root directory"
    )


ROOT = _find_module_root()

PROJECT_ENV = "SPATIAL_RISK_BENCH_PROJECT"
#: The SEPAL project this test benchmarks unless the env vars say otherwise:
#: <data dir>/test_peru_amazonia/test_peru_amazonia_project.json, its dataset's
#: target and features.
DEFAULT_PROJECT = "test_peru_amazonia"
DATASET_ENV = "SPATIAL_RISK_BENCH_DATASET"
TARGET_VAR_ENV = "SPATIAL_RISK_BENCH_TARGET_VAR"
FEATURE_VARS_ENV = "SPATIAL_RISK_BENCH_FEATURE_VARS"
MODELS_ENV = "SPATIAL_RISK_BENCH_MODELS"
WORKERS_ENV = "SPATIAL_RISK_BENCH_WORKERS"
N_SAMPLES_ENV = "SPATIAL_RISK_BENCH_N_SAMPLES"
TREES_ENV = "SPATIAL_RISK_BENCH_TREES"
SEED_ENV = "SPATIAL_RISK_BENCH_SEED"
MPX_ENV = "SPATIAL_RISK_BENCH_MPX"
ROWS_ENV = "SPATIAL_RISK_BENCH_ROWS"
ROW_OFFSET_ENV = "SPATIAL_RISK_BENCH_ROW_OFFSET"
OUT_ENV = "SPATIAL_RISK_BENCH_OUT"

MODELS = ("glm", "rf")
DEFAULT_WORKERS = "1,2,half,all,auto"
DEFAULT_MPX = 64.0
DEFAULT_N_SAMPLES = 5000
DEFAULT_TREES = 100
DEFAULT_SEED = 42

#: Stripe height the engine plans in, and therefore the unit the prediction
#: window is rounded to (:data:`spatialrisk.gdal_env.INFERENCE_TARGET_STRIPE_ROWS`).
STRIPE_ROWS = 256
#: Rows copied per read/write while the shorter stack is built.
COPY_CHUNK_ROWS = 1024
#: How long one stage may take before it is killed. Generous: a country-scale
#: window at one worker is minutes, and the point is only to fail a wedged
#: configuration rather than to police a slow one.
STAGE_TIMEOUT_S = 3600

#: The engine's plan line, e.g. "pred_glm_1.tif: 8 worker(s), 256 rows/stripe,
#: budget 24594 MiB (psutil.available), reserved 18072 MiB". Parsed only for
#: the table; the line itself is kept verbatim in every record.
PLAN_LINE_RE = re.compile(
    r"(?P<workers>\d+) worker\(s\), (?P<rows>\d+) rows/stripe, "
    r"budget (?P<budget>[\d.]+) MiB \((?P<source>[^)]*)\)"
)


# --------------------------------------------------------------------------- #
# resource detection
# --------------------------------------------------------------------------- #
def resource_readings() -> Dict:
    """Everything the inference policy reads before it plans, as it reads it.

    These are the production helpers, not a copy of them: the point of the
    assessment is to see what ``plan_inference`` actually saw. ``free`` /
    ``free_source`` are :func:`spatialrisk.parallel.free_memory_bytes`, which
    takes the tighter of psutil's host reading and the cgroup limit minus
    usage -- so a SEPAL sandbox that sets ``memory.max = max`` is planned
    against the *host's* available memory, shared with everything else on it.
    """
    from spatialrisk.parallel import (
        available_cores,
        cgroup_memory,
        free_memory_bytes,
        host_memory,
    )

    free, source = free_memory_bytes()
    return {
        "cores": available_cores(),
        "cpu_count": os.cpu_count(),
        "cgroup": cgroup_memory(),
        "host": host_memory(),
        "free": free,
        "free_source": source,
    }


def policy_plan(target: Path, features: List[Tuple[str, str, Optional[str]]]):
    """The plan :func:`~spatialrisk.gdal_env.plan_inference` makes for this stack.

    The same arguments ``windowed_predict._plan_and_reserve`` assembles for a
    run with no mask and no extra layer, and with the design width this
    benchmark's formula produces (one ``scale()`` column per feature plus the
    intercept). Nothing is reserved on the ledger, which is what a fresh
    stage subprocess sees too.
    """
    import rasterio

    from spatialrisk.gdal_env import plan_inference

    with rasterio.open(target) as src:
        width = src.width
        height = src.height
    itemsizes = []
    for _name, path, _rtype in features:
        with rasterio.open(path) as src:
            itemsizes.append(np.dtype(src.dtypes[0]).itemsize)
    plan = plan_inference(
        width=width,
        tile_rows=STRIPE_ROWS,
        n_features=len(features),
        feature_itemsizes=itemsizes,
        n_design_cols=len(features) + 1,
        with_mask=False,
        with_extra=False,
    )
    return plan, (height, width)


# --------------------------------------------------------------------------- #
# worker stages (each runs in its own subprocess)
# --------------------------------------------------------------------------- #
def _peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


class _PlanCapture(logging.Handler):
    """Keep the engine's plan line so the record shows what it decided."""

    def __init__(self):
        """Start with no line captured."""
        super().__init__(level=logging.INFO)
        self.line = None

    def emit(self, record):
        """Remember the first message that looks like the engine's plan line."""
        message = record.getMessage()
        if self.line is None and "rows/stripe" in message:
            self.line = message


class _Var:
    """The two attributes the dataset and the engine ask a variable for."""

    def __init__(self, name, path, raster_type=None):
        """Name the variable and point it at its raster."""
        self.name = name
        self.path = Path(path)
        self.raster_type = raster_type


class _Sample:
    """A stand-in for ``spatialrisk.sampling.Sample``: points and a label."""

    strategy = "stratified"
    allocation = "equal"

    def __init__(self, name, points):
        """Hold the drawn points the model will be fitted on."""
        self.name = name
        self._points = points

    def load_points(self):
        """The GeoDataFrame ``Dataset.extract_at_points`` is called with."""
        return self._points


def _bench_dataset(cfg: Dict):
    """A real :class:`spatialrisk.dataset.Dataset` over the benchmark stack.

    No ``Project`` is built: ``Dataset`` keeps its project reference as a
    plain attribute and ``extract_at_points`` only ever asks a variable for
    its name and its path.
    """
    from spatialrisk.dataset import Dataset

    target = _Var(*cfg["target"])
    features = [_Var(*f) for f in cfg["features"]]
    return Dataset(None, name="bench", target=target, features=features)


def _draw_points(cfg: Dict, target: Path):
    """The training sample: a stratified draw over the benchmark target.

    ``equal`` allocation, so both classes of a binary deforestation target
    reach the fit whatever their prevalence. The draw is seeded and does not
    depend on the sampling worker count, so every stage fits the same model.
    """
    from spatialrisk.sampling.service import generate_points

    return generate_points(
        target,
        None,
        strategy="stratified",
        n_samples=cfg["n_samples"],
        allocation="equal",
        seed=cfg["seed"],
    )


def _fit(cfg: Dict, dataset, folder: Path):
    """Fit the stage's model on a fresh sample. Runs before the clock starts."""
    stage = cfg["stage"]
    if stage == "rf":
        from spatialrisk.mlmodels.rf_model import RFModel

        model = RFModel(name="bench", n_trees=cfg["trees"], random_seed=cfg["seed"])
    else:
        from spatialrisk.mlmodels.glm_model import GLMModel

        model = GLMModel(name="bench", random_seed=cfg["seed"])

    model.dataset = dataset
    model.sample = _Sample("bench", _draw_points(cfg, dataset.target.path))
    terms = " + ".join(v.name for v in dataset.features)
    model.formula = f"I({dataset.target.name}) + trial ~ {terms}"
    model.fit(folder=folder)
    if model._x_design_info is None:
        # The GLM rebuilds this lazily inside apply(); do it up front so the
        # clock covers the prediction and not patsy.
        import pandas as pd
        from patsy import dmatrices

        _y, x_ref = dmatrices(
            model.formula, pd.read_csv(model.samples_path).dropna(), NA_action="drop"
        )
        model._x_design_info = x_ref.design_info
    # Keep the overview pyramid _register_prediction builds out of the timed
    # region: it is a fixed cost that does not scale with the worker count.
    model._register_prediction = lambda *a, **k: None
    return model


def _raster_digest(path: Path) -> Tuple[str, int]:
    """``(sha256 of the band, predicted pixels)``, read in row chunks.

    Hashing the band row by row keeps a country-sized output off the heap and
    is deterministic, so two worker counts that wrote the same pixels give
    the same digest. 0 is the output's nodata, so the non-zero count is how
    many pixels the run actually predicted -- a window with no valid data
    would otherwise look like a very fast run.
    """
    import rasterio
    from rasterio.windows import Window

    digest = hashlib.sha256()
    predicted = 0
    with rasterio.open(path) as src:
        for r0 in range(0, src.height, COPY_CHUNK_ROWS):
            rows = min(COPY_CHUNK_ROWS, src.height - r0)
            block = src.read(1, window=Window(0, r0, src.width, rows))
            digest.update(np.ascontiguousarray(block).tobytes())
            predicted += int(np.count_nonzero(block))
    return digest.hexdigest()[:16], predicted


def run_stage(cfg: Dict) -> Dict:
    """Execute one stage in this process and return its record."""
    from spatialrisk.mlmodels import windowed_predict  # noqa: F401  (import cost)
    from spatialrisk.sampling import blocked  # noqa: F401  (import cost)

    stage = cfg["stage"]
    if stage == "baseline":
        peak = _peak_mib()
        return {
            "seconds": 0.0,
            "cpu_s": 0.0,
            "peak_mib": peak,
            "setup_mib": peak,
            "detail": "imports only",
        }
    if stage not in MODELS:
        raise ValueError(f"unknown stage {stage!r}")

    logger = logging.getLogger("spatial_risk")
    logger.setLevel(logging.INFO)
    capture = _PlanCapture()
    with tempfile.TemporaryDirectory(prefix="inference_bench_") as tmp:
        dataset = _bench_dataset(cfg)
        model = _fit(cfg, dataset, Path(tmp))
        output = Path(tmp) / f"pred_{stage}.tif"

        setup_mib, base_cpu = _peak_mib(), _cpu_s()
        # The handler goes on for the prediction only: sample generation has a
        # stripe plan of its own and logs a line that looks just like this one.
        logger.addHandler(capture)
        try:
            t0 = time.perf_counter()
            model.apply(output_file=output, workers=cfg.get("workers"))
            seconds = time.perf_counter() - t0
        finally:
            logger.removeHandler(capture)
        cpu = _cpu_s() - base_cpu
        peak = _peak_mib()
        digest, predicted = _raster_digest(output)

    planned = PLAN_LINE_RE.search(capture.line or "")
    detail = f"{predicted / 1e6:.1f} Mpx predicted"
    if planned:
        detail = (
            f"{planned['workers']}w x {planned['rows']} rows, "
            f"budget {float(planned['budget']):.0f} MiB, {detail}"
        )
    return {
        "seconds": seconds,
        "cpu_s": cpu,
        "peak_mib": peak,
        "setup_mib": setup_mib,
        "detail": detail,
        "digest": digest,
        "predicted_px": predicted,
        "plan_log": capture.line,
        "plan_workers": int(planned["workers"]) if planned else None,
        "plan_rows": int(planned["rows"]) if planned else None,
    }


def spawn_stage(cfg: Dict) -> Dict:
    """Run ``run_stage(cfg)`` in a fresh interpreter and parse its JSON line.

    A stage that stops making progress is killed after
    :data:`STAGE_TIMEOUT_S` and reported with its stderr, so one wedged
    configuration fails the run instead of hanging it with nothing to read
    (the stage's output is captured, so a hung child prints nothing).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(cfg)]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
            timeout=STAGE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"stage {cfg['stage']!r} (workers={cfg.get('workers')}) did not finish "
            f"in {STAGE_TIMEOUT_S} s and was killed; its stderr tail:\n"
            + (exc.stderr or "")[-2000:]
        ) from exc
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# which stack to assess: a saved project's own dataset
# --------------------------------------------------------------------------- #
def data_dir() -> Path:
    """Where the app keeps projects: ``SPATIAL_RISK_DATA_DIR`` or the SEPAL default."""
    env = os.environ.get("SPATIAL_RISK_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "module_results" / "spatial_risk_module").resolve()


def read_manifest(project_name: str) -> Tuple[Dict, Path]:
    """The project's JSON manifest (as the app saves it) and its folder."""
    folder = data_dir() / project_name
    manifest = folder / f"{project_name}_project.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no project manifest at {manifest}; set SPATIAL_RISK_DATA_DIR if the "
            "projects live elsewhere"
        )
    return json.loads(manifest.read_text(encoding="utf-8")), folder


def variable_key(variables: Dict, name: str, year=None) -> str:
    """The ``processed_variables`` key for a dataset's variable name.

    A dataset stores bare variable names (``forest_gfc``) while the manifest
    keys temporal variables by year (``forest_gfc_2020``); the app resolves
    that through ``Project.get_all_instances``, which needs the whole object
    graph. Here the manifest's own ``name``/``year`` fields are enough:
    prefer the instance of the dataset's year, else a static one, else the
    first by name so the choice is at least deterministic.
    """
    candidates = [k for k, v in variables.items() if (v.get("name") or k) == name]
    if not candidates and name in variables:
        candidates = [name]
    if not candidates:
        raise ValueError(
            f"variable {name!r} not in the project; have {sorted(variables)}"
        )
    if year is not None:
        dated = [k for k in candidates if variables[k].get("year") == year]
        if dated:
            return dated[0]
    static = [k for k in candidates if variables[k].get("year") is None]
    return static[0] if static else sorted(candidates)[0]


def inputs_from_manifest(
    manifest: Dict,
    folder: Path,
    *,
    dataset_name=None,
    target_var=None,
    feature_vars=None,
) -> Dict:
    """The target and features a project's dataset uses, read off its manifest.

    No ``Project`` object is built: the manifest already stores every
    variable's ``path`` and every dataset's target and feature names, which
    is all the benchmark needs. Precedence: explicit variable names, else the
    named dataset, else the only/first one. A project with no dataset at all
    (variables processed but never assembled) needs the explicit names --
    guessing a target out of a variable list would benchmark a model nobody
    would fit.
    """
    variables = manifest.get("processed_variables") or {}
    datasets = manifest.get("datasets") or {}
    year = None
    if target_var is not None or feature_vars:
        if not (target_var and feature_vars):
            raise ValueError(f"{TARGET_VAR_ENV} and {FEATURE_VARS_ENV} go together")
        source = "explicit variables"
        target_key, feature_keys = target_var, list(feature_vars)
    else:
        if dataset_name:
            chosen = datasets.get(dataset_name)
            if chosen is None:
                raise ValueError(
                    f"dataset {dataset_name!r} not in project; have {sorted(datasets)}"
                )
        elif datasets:
            chosen = datasets[sorted(datasets)[0]]
        else:
            raise ValueError(
                "project has no dataset; name the variables with "
                f"{TARGET_VAR_ENV} and {FEATURE_VARS_ENV}"
            )
        year = chosen.get("year")
        target_key = variable_key(
            variables, chosen["target_name"], chosen.get("target_year") or year
        )
        feature_keys = [
            variable_key(variables, name, year) for name in chosen["feature_names"]
        ]
        source = f"dataset {chosen.get('name') or sorted(datasets)[0]!r}"

    def entry(key):
        var = variables.get(key)
        if var is None or not var.get("path"):
            raise ValueError(
                f"variable {key!r} not found in project or has no path; "
                f"rasters: {sorted(k for k, v in variables.items() if v.get('path'))}"
            )
        path = Path(var["path"])
        if not path.is_absolute():
            path = folder / path
        return (key, str(path), var.get("raster_type"))

    return {
        "target": entry(target_key),
        "features": [entry(k) for k in feature_keys],
        "source": source,
    }


def inputs_from_env() -> Dict:
    """Resolve the assessment stack from the environment's project manifest."""
    name = os.environ.get(PROJECT_ENV) or DEFAULT_PROJECT
    manifest, folder = read_manifest(name)
    features = os.environ.get(FEATURE_VARS_ENV)
    inputs = inputs_from_manifest(
        manifest,
        folder,
        dataset_name=os.environ.get(DATASET_ENV),
        target_var=os.environ.get(TARGET_VAR_ENV),
        feature_vars=[f.strip() for f in features.split(",") if f.strip()]
        if features
        else None,
    )
    inputs["source"] = f"project {name!r} ({folder}), {inputs['source']}"
    return inputs


# --------------------------------------------------------------------------- #
# the prediction window: a shorter, co-registered copy of the stack
# --------------------------------------------------------------------------- #
def window_rows(height: int, width: int, *, rows=None, mpx=None) -> Optional[int]:
    """How many rows to predict over: whole stripes, or None for the whole raster.

    An explicit ``rows`` wins; otherwise the megapixel budget is divided by
    the raster's width. Either way the answer is rounded down to whole
    256-row stripes (at least one), and a window that would cover the raster
    means "no window at all".
    """
    if rows is None:
        if not mpx:
            return None
        rows = int(mpx * 1e6 / max(1, width))
    rows = max(STRIPE_ROWS, (int(rows) // STRIPE_ROWS) * STRIPE_ROWS)
    return None if rows >= height else rows


def crop_stack(
    inputs: Dict, rows: int, offset: int, out_dir: Path
) -> Tuple[Dict, Dict]:
    """Copy target and features into ``out_dir``, keeping only ``rows`` rows.

    The engine reads every feature with the *target's* window, so the layers
    must stay pixel-aligned: they are all cut at the same offset, and each
    keeps its source's dtype, nodata, compression and block layout so the
    decode cost per stripe is the one the real raster has. A layer that is
    not on the target's grid is refused here rather than cut: cutting it at
    the target's offset would shift it, and nothing downstream could tell --
    every stage shares the one cropped stack, so the digests would agree on
    the same wrong answer. Returns the new inputs and the target's class
    histogram, which is counted for free while its rows go past.
    """
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    counts: Dict[int, int] = {}
    with rasterio.open(inputs["target"][1]) as src:
        grid = (src.height, src.width, src.transform)

    def copy(entry, tally=False):
        name, path, rtype = entry
        dst_path = out_dir / f"{name}.tif"
        with rasterio.open(path) as src:
            if (src.height, src.width) != grid[:2] or not src.transform.almost_equals(
                grid[2]
            ):
                raise ValueError(
                    f"{name!r} is not on the target's grid: "
                    f"{src.width}x{src.height} at {tuple(src.transform)[:6]} vs "
                    f"{grid[1]}x{grid[0]} at {tuple(grid[2])[:6]}. The engine reads "
                    "every feature with the target's own window indices, so an "
                    "unharmonised stack cannot be windowed (or predicted)."
                )
            profile = src.profile.copy()
            # The window's transform, built with affine rather than
            # rasterio.windows.transform: that helper goes through the
            # deprecated ``Affine * (x, y)`` form and warns once per layer.
            profile.update(
                height=rows, transform=src.transform * Affine.translation(0, offset)
            )
            with rasterio.open(dst_path, "w", **profile) as dst:
                for r0 in range(0, rows, COPY_CHUNK_ROWS):
                    take = min(COPY_CHUNK_ROWS, rows - r0)
                    block = src.read(1, window=Window(0, offset + r0, src.width, take))
                    dst.write(block, 1, window=Window(0, r0, src.width, take))
                    if tally and np.issubdtype(block.dtype, np.integer):
                        values, seen = np.unique(block, return_counts=True)
                        for value, n in zip(values.tolist(), seen.tolist()):
                            counts[value] = counts.get(value, 0) + n
        return (name, str(dst_path), rtype)

    cropped = {
        "target": copy(inputs["target"], tally=True),
        "features": [copy(f) for f in inputs["features"]],
        "source": inputs["source"] + f", rows {offset}-{offset + rows}",
    }
    return cropped, counts


def _check_classes(target_path: Path, counts: Dict[int, int]) -> str:
    """Refuse a window the models cannot be fitted on, and describe the one kept."""
    import rasterio

    with rasterio.open(target_path) as src:
        nodata = src.nodata
    valid = {
        value: n for value, n in counts.items() if nodata is None or value != nodata
    }
    if len(valid) < 2:
        raise ValueError(
            f"the prediction window holds {len(valid)} target class(es) "
            f"({valid or 'none'}): nothing to fit. Move or widen it with "
            f"{ROW_OFFSET_ENV} / {ROWS_ENV} / {MPX_ENV}."
        )
    return ", ".join(f"{value}: {n:,}" for value, n in sorted(valid.items()))


# --------------------------------------------------------------------------- #
# the assessment itself
# --------------------------------------------------------------------------- #
def _resolve_workers(spec: str, cores: int):
    """Worker counts from a comma list: ``half``/``all`` of cores, ``auto`` = policy."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok == "half":
            value = max(1, cores // 2)
        elif tok == "all":
            value = cores
        elif tok == "auto":
            value = None
        elif tok:
            value = max(1, int(tok))
        else:
            continue
        if value not in out:
            out.append(value)
    return sorted((v for v in out if v is not None)) + ([None] if None in out else [])


def _resolve_models(spec: str) -> List[str]:
    """Model stages from a comma list, in the order :data:`MODELS` declares them."""
    asked = {tok.strip().lower() for tok in spec.split(",") if tok.strip()}
    unknown = asked - set(MODELS)
    if unknown:
        raise ValueError(f"unknown model(s) {sorted(unknown)}; have {list(MODELS)}")
    return [m for m in MODELS if m in asked]


def _digests_agree(records) -> bool:
    """Every worker count of one model wrote the identical raster."""
    families: Dict[str, list] = {}
    for r in records:
        if r.get("digest") is not None:
            families.setdefault(r["stage"], []).append(r["digest"])
    return all(all(d == ds[0] for d in ds[1:]) for ds in families.values())


def assess(
    inputs: Dict,
    *,
    models_spec: str = ",".join(MODELS),
    workers_spec: str = DEFAULT_WORKERS,
    n_samples: int = DEFAULT_N_SAMPLES,
    trees: int = DEFAULT_TREES,
    seed: int = DEFAULT_SEED,
    rows=None,
    mpx=DEFAULT_MPX,
    offset=None,
    out=None,
) -> Dict:
    """Print the assessment table for one feature stack and return every record."""
    import rasterio

    mib = 1024 * 1024
    readings = resource_readings()
    cores = readings["cores"]

    def fmt(b):
        return "n/a" if b is None else f"{b / mib:.0f} MiB"

    with rasterio.open(inputs["target"][1]) as src:
        height, width = src.height, src.width
    rows = window_rows(height, width, rows=rows, mpx=mpx)

    print()
    print(f"inputs: {inputs['source']}")
    print(f"target: {inputs['target'][1]}")
    print(
        f"features ({len(inputs['features'])}): "
        + ", ".join(f[0] for f in inputs["features"])
    )
    cg, host = readings["cgroup"], readings["host"]
    print(
        f"cores: affinity={cores} cpu_count={readings['cpu_count']} | "
        f"cgroup v{cg['version']} limit={fmt(cg['limit'])} usage={fmt(cg['usage'])} | "
        f"host total={fmt(host['total'])} available={fmt(host['available'])}"
    )
    print(f"free for a job: {fmt(readings['free'])} ({readings['free_source']})")

    workspace = tempfile.TemporaryDirectory(prefix="inference_bench_stack_")
    try:
        if rows is None:
            print(f"raster: {width} x {height}, predicted whole")
            window = {"rows": height, "offset": 0, "cropped": False}
        else:
            offset = (height - rows) // 2 if offset is None else int(offset)
            offset = max(0, min(offset, height - rows))
            print(
                f"raster: {width} x {height}; predicting rows {offset}-{offset + rows}"
                f" ({rows * width / 1e6:.0f} Mpx) — copying the stack",
                flush=True,
            )
            t0 = time.perf_counter()
            inputs, counts = crop_stack(inputs, rows, offset, Path(workspace.name))
            classes = _check_classes(Path(inputs["target"][1]), counts)
            print(
                f"copied in {time.perf_counter() - t0:.1f} s; "
                f"target classes — {classes}"
            )
            window = {"rows": rows, "offset": offset, "cropped": True}

        plan, (pheight, pwidth) = policy_plan(
            Path(inputs["target"][1]), inputs["features"]
        )
        n_stripes = -(-pheight // plan.rows_per_stripe)
        print(
            f"stripes: {n_stripes} x {plan.rows_per_stripe} rows of {pwidth} px = "
            f"{fmt(plan.stripe_bytes)} working set each"
        )
        print(
            f"policy: {plan.workers} workers (by cores {plan.by_cores}, by memory "
            f"{plan.by_memory}), budget {fmt(plan.memory_budget_bytes)} "
            f"({plan.memory_source}), {plan.gdal_threads} GDAL thread(s)"
        )

        base_cfg = {
            "target": list(inputs["target"]),
            "features": [list(f) for f in inputs["features"]],
            "n_samples": n_samples,
            "trees": trees,
            "seed": seed,
        }
        base = spawn_stage({**base_cfg, "stage": "baseline"})
        print(f"import baseline: {base['peak_mib']:.0f} MiB\n")

        header = (
            f"{'stage':<8}{'workers':>8}{'seconds':>9}{'cpu s':>9}"
            f"{'peak MiB':>10}{'net MiB':>9}  detail"
        )
        print(header)
        print("-" * len(header), flush=True)
        records = []
        for model in _resolve_models(models_spec):
            for workers in _resolve_workers(workers_spec, cores):
                cfg = {**base_cfg, "stage": model}
                if workers is not None:
                    cfg["workers"] = workers
                rec = spawn_stage(cfg)
                rec.update(stage=model, workers=workers)
                records.append(rec)
                print(
                    f"{model:<8}{'auto' if workers is None else workers:>8}"
                    f"{rec['seconds']:>9.1f}{rec['cpu_s']:>9.1f}"
                    f"{rec['peak_mib']:>10.0f}"
                    f"{rec['peak_mib'] - rec['setup_mib']:>9.0f}  {rec['detail']}",
                    flush=True,
                )
    finally:
        workspace.cleanup()

    result = {
        "cores": cores,
        "cgroup": readings["cgroup"],
        "host": readings["host"],
        "free": readings["free"],
        "free_source": readings["free_source"],
        "window": window,
        "stripes": {
            "shape": [pheight, pwidth],
            "rows_per_stripe": plan.rows_per_stripe,
            "n_stripes": n_stripes,
            "stripe_working_set": plan.stripe_bytes,
        },
        "policy": {
            "workers": plan.workers,
            "by_cores": plan.by_cores,
            "by_memory": plan.by_memory,
            "memory_budget_bytes": plan.memory_budget_bytes,
            "memory_source": plan.memory_source,
            "plan": repr(plan),
        },
        "baseline_mib": base["peak_mib"],
        "records": records,
    }
    if out:
        Path(out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nrecords written to {out}")
    return result


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_resolve_workers_expands_half_all_and_auto():
    """``half``/``all`` scale with the cores, ``auto`` sorts last as None."""
    assert _resolve_workers("1,2,half,all,auto", 8) == [1, 2, 4, 8, None]
    # duplicates collapse (half == all == 1 on a single core) and stay ordered
    assert _resolve_workers("all,half,1", 1) == [1]
    assert _resolve_workers("4, 2 ,", 16) == [2, 4]


def test_resolve_models_keeps_the_declared_order_and_rejects_typos():
    """A comma list picks stages; anything else is a clear error, not a skip."""
    assert _resolve_models("rf,glm") == ["glm", "rf"]
    assert _resolve_models("GLM") == ["glm"]
    with pytest.raises(ValueError, match="unknown model"):
        _resolve_models("glm,icar")


def test_window_rows_rounds_to_whole_stripes_and_can_be_switched_off():
    """A megapixel budget becomes whole 256-row stripes; 0 means the whole raster."""
    assert window_rows(50_000, 40_000, mpx=64) == 1536  # 1600 -> 6 stripes
    assert window_rows(50_000, 40_000, rows=700) == 512
    assert window_rows(50_000, 40_000, mpx=0) is None
    # never below one stripe, and a window covering the raster is no window
    assert window_rows(50_000, 40_000, mpx=0.001) == STRIPE_ROWS
    assert window_rows(1_000, 40_000, mpx=1_000_000) is None


def _manifest():
    """A manifest shaped like the app's ``<project>_project.json``."""
    return {
        "processed_variables": {
            "loss_2020_2024": {"name": "loss_2020_2024", "path": "/p/loss.tif"},
            "altitude": {"name": "altitude", "path": "/p/alt.tif"},
            "forest_2020": {"name": "forest", "year": 2020, "path": "data/f20.tif"},
            "forest_2024": {"name": "forest", "year": 2024, "path": "/p/f24.tif"},
            "nopath": {"name": "nopath", "path": None},
        },
        "datasets": {
            "calibration": {
                "name": "calibration",
                "target_name": "loss_2020_2024",
                "target_year": None,
                "year": 2020,
                "feature_names": ["altitude", "forest"],
            }
        },
    }


def test_inputs_from_manifest_resolves_the_dataset_year_of_each_feature():
    """A temporal feature takes the dataset's year; a relative path joins the folder."""
    got = inputs_from_manifest(_manifest(), Path("/proj"))
    assert got["target"] == ("loss_2020_2024", "/p/loss.tif", None)
    assert [f[0] for f in got["features"]] == ["altitude", "forest_2020"]
    assert got["features"][1][1] == str(Path("/proj/data/f20.tif"))
    assert got["source"] == "dataset 'calibration'"


def test_inputs_from_manifest_rejects_what_it_cannot_resolve():
    """Explicit names win; a missing dataset, variable or path is a clear error."""
    explicit = inputs_from_manifest(
        _manifest(), Path("/proj"), target_var="forest_2024", feature_vars=["altitude"]
    )
    assert explicit["target"][0] == "forest_2024"
    assert explicit["source"] == "explicit variables"
    with pytest.raises(ValueError, match="not in project"):
        inputs_from_manifest(_manifest(), Path("/proj"), dataset_name="nope")
    with pytest.raises(ValueError, match="has no path"):
        inputs_from_manifest(
            _manifest(), Path("/proj"), target_var="nopath", feature_vars=["altitude"]
        )
    with pytest.raises(ValueError, match="no dataset"):
        inputs_from_manifest({"processed_variables": {}}, Path("/proj"))
    # half an override would otherwise be ignored and the dataset used instead
    with pytest.raises(ValueError, match="go together"):
        inputs_from_manifest(_manifest(), Path("/proj"), feature_vars=["altitude"])


def test_digests_agree_compares_within_one_model():
    """Two models writing different rasters is normal; one model must not."""
    same = [
        {"stage": "glm", "digest": "a"},
        {"stage": "glm", "digest": "a"},
        {"stage": "rf", "digest": "b"},
    ]
    assert _digests_agree(same)
    assert not _digests_agree(same + [{"stage": "glm", "digest": "c"}])


def test_plan_line_regex_reads_the_engines_own_log_line():
    """The record's worker/stripe columns come from the line the engine logged."""
    line = (
        "pred_glm.tif: 8 worker(s), 256 rows/stripe, budget 24594 MiB "
        "(psutil.available), reserved 18072 MiB"
    )
    match = PLAN_LINE_RE.search(line)
    assert match["workers"] == "8" and match["rows"] == "256"
    assert match["source"] == "psutil.available"


def _write_raster(path, array, *, nodata, dtype=None, north=None):
    """Write a tiled single-band GeoTIFF with 30 m pixels.

    ``north`` moves the top edge, which is how a test builds a layer that is
    the right shape but not on the target's grid.
    """
    import rasterio
    from rasterio.transform import from_origin

    array = array if dtype is None else array.astype(dtype)
    north = array.shape[0] * 30 if north is None else north
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype.name,
        crs="EPSG:3857",
        transform=from_origin(0, north, 30, 30),
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    ) as dst:
        dst.write(array, 1)


def _synthetic_stack(folder: Path, height=600, width=300, layers=3, seed=1) -> Dict:
    """A target with a real signal in ``f0`` plus ``layers`` feature rasters."""
    rng = np.random.default_rng(seed)
    features = []
    signal = None
    for i in range(layers):
        arr = (rng.random((height, width)) * 100).astype("float32")
        arr[rng.random(arr.shape) < 0.02] = -9999.0
        path = folder / f"f{i}.tif"
        _write_raster(path, arr, nodata=-9999.0)
        features.append((f"f{i}", str(path), "continuous"))
        if signal is None:
            signal = arr
    target = ((signal / 100 + rng.random(signal.shape) * 0.5) > 0.6).astype("uint8")
    target[signal == -9999.0] = 255
    path = folder / "loss.tif"
    _write_raster(path, target, nodata=255)
    return {
        "target": ("loss", str(path), "categorical"),
        "features": features,
        "source": "synthetic stack",
    }


@pytest.mark.raster
def test_crop_stack_puts_every_layer_on_the_targets_cropped_grid(tmp_path):
    """Every layer is cut at the same offset, onto the target's cropped grid.

    The digest assertion structurally cannot catch a bad crop: every stage
    shares the one cropped stack, so a feature cut at the wrong offset gives
    consistently wrong predictions with agreeing digests. This compares each
    cropped layer against its own source's window instead.
    """
    import rasterio
    from affine import Affine
    from rasterio.windows import Window

    inputs = _synthetic_stack(tmp_path)
    rows, offset = 256, 172
    out = tmp_path / "cropped"
    out.mkdir()
    cropped, counts = crop_stack(inputs, rows, offset, out)

    with rasterio.open(inputs["target"][1]) as src:
        width = src.width
        want_transform = src.transform * Affine.translation(0, offset)
        window = Window(0, offset, width, rows)
        target_block = src.read(1, window=window)

    sources = [inputs["target"]] + inputs["features"]
    for entry, source in zip([cropped["target"]] + cropped["features"], sources):
        name = entry[0]
        assert name == source[0]
        with rasterio.open(entry[1]) as dst, rasterio.open(source[1]) as src:
            assert (dst.height, dst.width) == (rows, width), name
            assert dst.transform == want_transform, name
            assert (dst.dtypes, dst.nodata) == (src.dtypes, src.nodata), name
            np.testing.assert_array_equal(
                dst.read(1), src.read(1, window=window), err_msg=name
            )

    # the histogram counted during the copy is the window's own, and two
    # classes are enough to fit on
    values, seen = np.unique(target_block, return_counts=True)
    assert counts == dict(zip(values.tolist(), seen.tolist()))
    assert _check_classes(Path(cropped["target"][1]), counts)


@pytest.mark.raster
def test_crop_stack_refuses_a_layer_off_the_targets_grid(tmp_path):
    """A layer on another grid is an error, not a silently shifted read."""
    out = tmp_path / "cropped"
    out.mkdir()
    ones = np.ones((600, 300), dtype="float32")

    taller = _synthetic_stack(tmp_path)
    _write_raster(tmp_path / "tall.tif", np.ones((640, 300), "float32"), nodata=-9999.0)
    taller["features"].append(("tall", str(tmp_path / "tall.tif"), "continuous"))
    with pytest.raises(ValueError, match="not on the target's grid"):
        crop_stack(taller, 256, 172, out)

    # same shape, shifted origin: only the transform gives it away
    shifted = _synthetic_stack(tmp_path)
    _write_raster(tmp_path / "shift.tif", ones, nodata=-9999.0, north=600 * 30 + 30)
    shifted["features"].append(("shift", str(tmp_path / "shift.tif"), "continuous"))
    with pytest.raises(ValueError, match="not on the target's grid"):
        crop_stack(shifted, 256, 172, out)


@pytest.mark.raster
def test_check_classes_refuses_a_window_it_cannot_fit(tmp_path):
    """One class (or only nodata) in the window names the knobs that move it."""
    path = tmp_path / "t.tif"
    _write_raster(path, np.ones((256, 300), dtype="uint8"), nodata=255)
    assert _check_classes(path, {0: 10, 1: 5, 255: 3}) == "0: 10, 1: 5"
    with pytest.raises(ValueError, match="nothing to fit"):
        _check_classes(path, {1: 10, 255: 7})
    with pytest.raises(ValueError, match=ROW_OFFSET_ENV):
        _check_classes(path, {255: 7})


@pytest.mark.raster
@pytest.mark.slow
def test_inference_bench_smoke_crops_the_stack(tmp_path):
    """The megapixel budget shortens the stack and the run still predicts it.

    0.1 Mpx over a 300 px wide raster is 333 rows, which rounds down to one
    whole 256-row stripe, centred in the 600-row stack.
    """
    inputs = _synthetic_stack(tmp_path)
    result = assess(inputs, models_spec="glm", workers_spec="1", n_samples=300, mpx=0.1)
    assert result["window"] == {"rows": 256, "offset": 172, "cropped": True}
    assert result["stripes"]["shape"] == [256, 300]
    assert result["stripes"]["n_stripes"] == 1
    assert result["records"][0]["predicted_px"] > 0


@pytest.mark.raster
@pytest.mark.slow
def test_inference_bench_smoke(tmp_path):
    """The whole subprocess pipeline works and the pool agrees with the serial path."""
    inputs = _synthetic_stack(tmp_path)
    result = assess(
        inputs,
        models_spec="glm",
        workers_spec="1,2",
        n_samples=400,
        mpx=0,
        out=tmp_path / "records.json",
    )
    stages = [(r["stage"], r["workers"]) for r in result["records"]]
    assert stages == [("glm", 1), ("glm", 2)]
    assert _digests_agree(result["records"]), "the pool changed the raster"
    assert all(r["predicted_px"] > 0 for r in result["records"])
    assert result["records"][1]["plan_workers"] == 2
    assert result["policy"]["workers"] >= 1
    assert (tmp_path / "records.json").exists()


@pytest.mark.raster
@pytest.mark.slow
def test_sepal_inference_assessment():
    """Time serial and pooled ``apply`` for each model on a real project's stack."""
    try:
        inputs = inputs_from_env()
    except (FileNotFoundError, ValueError) as exc:
        # ValueError: the project is here but has no dataset and no explicit
        # variables -- still "not a machine this benchmark can run on".
        pytest.skip(f"not on a machine with the project: {exc}")
    for name, path, _rtype in [inputs["target"]] + inputs["features"]:
        assert Path(path).exists(), f"{name}: {path}"

    rows = os.environ.get(ROWS_ENV)
    offset = os.environ.get(ROW_OFFSET_ENV)
    result = assess(
        inputs,
        models_spec=os.environ.get(MODELS_ENV) or ",".join(MODELS),
        workers_spec=os.environ.get(WORKERS_ENV, DEFAULT_WORKERS),
        n_samples=int(os.environ.get(N_SAMPLES_ENV) or DEFAULT_N_SAMPLES),
        trees=int(os.environ.get(TREES_ENV) or DEFAULT_TREES),
        seed=int(os.environ.get(SEED_ENV) or DEFAULT_SEED),
        rows=int(rows) if rows else None,
        mpx=float(os.environ.get(MPX_ENV) or DEFAULT_MPX),
        offset=int(offset) if offset else None,
        out=os.environ.get(OUT_ENV),
    )
    assert _digests_agree(result["records"]), "a worker count changed the raster"


if __name__ == "__main__":
    # Worker entry: ``python tests/test_inference_bench_sepal.py --worker '<json>'``
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(run_stage(json.loads(sys.argv[2]))))
    else:
        sys.exit("run this file through pytest; --worker is internal")
