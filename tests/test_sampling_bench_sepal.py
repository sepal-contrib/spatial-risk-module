"""Assess sample generation on the machine that matters (SEPAL).

The row-stripe scan in ``spatialrisk/sampling`` bounded memory but walks the
stripes serially on one Python thread, and the numpy work per stripe -- not
tile decoding -- is where the time goes: on a 2.2 Gpx raster plus mask,
``GDAL_NUM_THREADS`` 1 -> 8 moved a full stratified run only 36.7 -> 31.6 s,
while a pool of worker threads that each own a dataset handle took pass 1
from 13.9 s to 3.1 s on 8 workers (2026-09-18, 16-core dev box). This test
produces the same numbers on SEPAL, where the core count, the cgroup memory
limit and the storage all differ, so the worker policy can be chosen from
evidence instead of from a laptop.

What it reports, each configuration in a **fresh subprocess** so peak RSS is
per run and the import baseline is shown separately:

* the resources the process can see: affinity cores vs ``os.cpu_count``, the
  cgroup memory limit/usage (v2 and v1 files), and psutil's host view;
* the stripe plan for the raster and what the *reference policy* -- half the
  cores, 50 % of free memory -- would choose for it;
* ``serial_pass1`` / ``serial_full``: the pre-pool production path (one
  worker, multi-threaded GDAL decode), the baseline the pool is judged against;
* ``pool_pass1`` / ``pool_full`` for several worker counts: the library's
  stripe pool (``RasterScan(workers=w)`` / ``generate_points(workers=w)``),
  pass 1 checked to produce identical class counts and the full run identical
  points; ``auto`` in the workers list means the resource policy's choice.

Run on SEPAL from the module root::

    pytest tests/test_sampling_bench_sepal.py -s -k sepal

By default it reads the manifest of ``test_peru_amazonia`` under
``~/module_results/spatial_risk_module`` (only the JSON, no app code) and
benchmarks the raster, mask and design of its ``stratified_1`` sample.
``SPATIAL_RISK_BENCH_PROJECT`` names another project (found under
``SPATIAL_RISK_DATA_DIR`` or ``~/module_results/spatial_risk_module``, as the
app finds them); ``SPATIAL_RISK_BENCH_SAMPLE`` picks another sample by name
(else the most recent one); ``SPATIAL_RISK_BENCH_RASTER_VAR`` /
``SPATIAL_RISK_BENCH_MASK_VAR`` name variables directly (a project without
samples falls back to its dataset target with no mask). Explicit files also
work: ``SPATIAL_RISK_BENCH_RASTER`` / ``SPATIAL_RISK_BENCH_MASK`` (co-registered,
as ``generate_points`` requires).

Optional: ``SPATIAL_RISK_BENCH_STRATEGY`` (stratified|random|systematic),
``SPATIAL_RISK_BENCH_N_SAMPLES``, ``SPATIAL_RISK_BENCH_ALLOCATION`` (override
the sample's design), ``SPATIAL_RISK_BENCH_WORKERS`` (comma list, ``half`` /
``all`` allowed; default ``1,2,half,all``), ``SPATIAL_RISK_BENCH_OUT`` (write
the records as JSON).

On a machine without the project the SEPAL test is skipped; the smoke test
below runs the same machinery on a small synthetic raster so the file keeps
working between SEPAL sessions.
"""

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
            Path(dirpath, "spatialrisk", "sampling", "blocked.py").exists()
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
        if (root / "spatialrisk" / "sampling" / "blocked.py").exists():
            sys.path.insert(0, str(root))
            return root.resolve()
    raise ImportError(
        "cannot find the spatial-risk-module checkout (spatialrisk package); "
        f"set {MODULE_DIR_ENV} to its root directory"
    )


ROOT = _find_module_root()

PROJECT_ENV = "SPATIAL_RISK_BENCH_PROJECT"
#: The SEPAL project this test benchmarks unless the env vars say otherwise:
#: <data dir>/test_peru_amazonia/test_peru_amazonia_project.json, sample
#: ``stratified_1`` (loss_geobosques_2010_forest_gfc_tc30_2020 raster,
#: geobosques_2010 mask, 10 000 points, proportional allocation).
DEFAULT_PROJECT = "test_peru_amazonia"
DEFAULT_SAMPLE = "stratified_1"
SAMPLE_ENV = "SPATIAL_RISK_BENCH_SAMPLE"
RASTER_VAR_ENV = "SPATIAL_RISK_BENCH_RASTER_VAR"
MASK_VAR_ENV = "SPATIAL_RISK_BENCH_MASK_VAR"
RASTER_ENV = "SPATIAL_RISK_BENCH_RASTER"
MASK_ENV = "SPATIAL_RISK_BENCH_MASK"
STRATEGY_ENV = "SPATIAL_RISK_BENCH_STRATEGY"
N_SAMPLES_ENV = "SPATIAL_RISK_BENCH_N_SAMPLES"
ALLOCATION_ENV = "SPATIAL_RISK_BENCH_ALLOCATION"
WORKERS_ENV = "SPATIAL_RISK_BENCH_WORKERS"
OUT_ENV = "SPATIAL_RISK_BENCH_OUT"

#: Reference policy: the share of cores and of free memory one sampling job
#: may take. Half the cores mirrors ``spatialrisk.parallel.worker_threads``;
#: half the free memory leaves the app server and a concurrent job room.
CORE_FRACTION = 0.5
MEMORY_FRACTION = 0.5

#: Working set of one in-flight stripe per worker, in bytes per stripe pixel:
#: the raster stripe, the mask stripe, the validity bool, the compacted values
#: (``arr[valid]``) and the intp upcast ``np.bincount`` makes of them (chunked
#: at 4 Mpx by ``blocked._COUNT_CHUNK``, so counted as a flat 32 MiB below).
#: Deliberately generous; the measured "net MiB" column is what refines it.
BYTES_PER_STRIPE_PIXEL = 4
BINCOUNT_CHUNK_BYTES = (1 << 22) * 8


# --------------------------------------------------------------------------- #
# resource detection
# --------------------------------------------------------------------------- #
def parse_cgroup_bytes(text: str) -> Optional[int]:
    """Parse a cgroup memory file: ``max`` (v2) or the v1 no-limit sentinel -> None."""
    text = text.strip()
    if not text or text == "max":
        return None
    value = int(text)
    # cgroup v1 reports "unlimited" as a huge page-aligned number (2**63 - 4096).
    if value >= (1 << 62):
        return None
    return value


def _read_bytes(path: str) -> Optional[int]:
    try:
        return parse_cgroup_bytes(Path(path).read_text())
    except (OSError, ValueError):
        return None


def cgroup_memory() -> Dict[str, Optional[int]]:
    """The container's memory limit and current usage, whichever cgroup version."""
    limit = _read_bytes("/sys/fs/cgroup/memory.max")
    if limit is not None or Path("/sys/fs/cgroup/memory.current").exists():
        return {
            "version": 2,
            "limit": limit,
            "usage": _read_bytes("/sys/fs/cgroup/memory.current"),
        }
    limit = _read_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None:
        return {
            "version": 1,
            "limit": limit,
            "usage": _read_bytes("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        }
    return {"version": None, "limit": None, "usage": None}


def host_memory() -> Dict[str, int]:
    """Total and available bytes as psutil sees the host."""
    import psutil

    vm = psutil.virtual_memory()
    return {"total": int(vm.total), "available": int(vm.available)}


def free_memory_bytes(cg: Dict, host: Dict) -> Tuple[int, str]:
    """Bytes a job may still allocate, and which reading bounded it.

    In a container the cgroup limit is the wall the OOM killer enforces, and
    it can be far below what psutil reports for the host; outside one only the
    host reading exists. The tighter of the two wins.
    """
    free = host["available"]
    source = "psutil.available"
    if cg.get("limit") is not None:
        cg_free = cg["limit"] - (cg.get("usage") or 0)
        if cg_free < free:
            free, source = cg_free, f"cgroup v{cg['version']} limit - usage"
    return max(0, free), source


def affinity_cores() -> int:
    """Cores this process may run on (the cgroup quota on SEPAL, not the host)."""
    from spatialrisk.parallel import available_cores

    return available_cores()


def stripe_working_set_bytes(stripe_pixels: int, with_mask: bool) -> int:
    """Estimated bytes one worker holds while it processes one stripe."""
    per_px = BYTES_PER_STRIPE_PIXEL + (1 if with_mask else 0)
    return stripe_pixels * per_px + BINCOUNT_CHUNK_BYTES


def reference_policy(
    cores: int, free_bytes: int, stripe_bytes: int, *, gdal_cache_bytes: int
) -> Dict[str, int]:
    """Workers the reference policy would run: half the cores, 50 % of free RAM.

    The memory budget is what is left of the 50 % share after GDAL's block
    cache, divided by one stripe's working set; the answer is never below one
    worker (the serial path must always be allowed to run).
    """
    by_cores = max(1, int(cores * CORE_FRACTION))
    budget = max(0, int(free_bytes * MEMORY_FRACTION) - gdal_cache_bytes)
    by_memory = max(1, budget // max(1, stripe_bytes))
    return {
        "by_cores": by_cores,
        "by_memory": by_memory,
        "workers": min(by_cores, by_memory),
    }


# --------------------------------------------------------------------------- #
# worker stages (each runs in its own subprocess)
# --------------------------------------------------------------------------- #
def _peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_stage(cfg: Dict) -> Dict:
    """Execute one stage in this process and return its record."""
    from spatialrisk.gdal_env import (
        DEFAULT_SAMPLING_CACHEMAX_BYTES,
        sampling_gdal_env,
        sampling_num_threads,
    )
    from spatialrisk.sampling import blocked, generate_points

    stage = cfg["stage"]
    if stage == "baseline":
        return {"seconds": 0.0, "peak_mib": _peak_mib(), "detail": "imports only"}

    raster, mask = cfg["raster"], cfg.get("mask")
    strategy = cfg["strategy"]
    # The production defaults: the GDAL cache budget and half-the-cores decode
    # threads that gui/tile/sampling_tile.py applies around Sample.generate().
    # serial_* pins one worker; pool_* asks for cfg["workers"] (None = policy).
    # generate_points budgets GDAL's decode threads and cache from its own plan
    # inside this outer env, exactly as it does under the tile's env.
    workers = 1 if stage.startswith("serial") else cfg.get("workers")
    with sampling_gdal_env(
        cachemax_bytes=DEFAULT_SAMPLING_CACHEMAX_BYTES,
        num_threads=sampling_num_threads(),
    ):
        t0 = time.perf_counter()
        if stage in ("serial_full", "pool_full"):
            gdf = generate_points(
                raster,
                mask,
                strategy=strategy,
                n_samples=cfg["n_samples"],
                allocation=cfg.get("allocation", "equal"),
                seed=cfg.get("seed", 42),
                workers=workers,
            )
            detail = f"{len(gdf)} points"
            digest = {
                "points": int(len(gdf)),
                "rows": int(gdf["row"].sum()),
                "cols": int(gdf["col"].sum()),
            }
        elif stage in ("serial_pass1", "pool_pass1"):
            with blocked.RasterScan(raster, mask, workers=workers) as scan:
                if strategy == "stratified":
                    counts, n_valid = blocked.count_values(scan)
                    digest = {str(int(k)): int(v) for k, v in counts.items()}
                    detail = f"{n_valid} valid, {len(counts)} classes"
                else:
                    n_valid = blocked.count_valid(scan)
                    digest = {"valid": int(n_valid)}
                    detail = f"{n_valid} valid"
                if scan.plan is not None:
                    detail += f" (policy: {scan.plan.workers} workers)"
        else:
            raise ValueError(f"unknown stage {stage!r}")
        seconds = time.perf_counter() - t0
    return {
        "seconds": seconds,
        "peak_mib": _peak_mib(),
        "detail": detail,
        "digest": digest,
    }


def spawn_stage(cfg: Dict) -> Dict:
    """Run ``run_stage(cfg)`` in a fresh interpreter and parse its JSON line."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(cfg)]
    proc = subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=env, cwd=str(ROOT)
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# which raster to assess: a saved project's own sampling inputs
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


def inputs_from_manifest(
    manifest: Dict, folder: Path, *, sample_name=None, raster_var=None, mask_var=None
) -> Dict:
    """The sampling inputs a project actually uses, read straight off its manifest.

    No ``Project`` object is built: the manifest already stores every raster's
    ``path`` and every sample's variable names and design, which is all the
    benchmark needs. Precedence: explicit variable names, else the named
    sample, else the most recently generated sample (the design the user ran in
    the app), else the first dataset's target with no mask.
    """
    variables = manifest.get("processed_variables") or {}
    design = {"strategy": "stratified", "n_samples": 10000, "allocation": "equal"}
    if raster_var is not None:
        source = "explicit variables"
    else:
        samples = manifest.get("samples") or {}
        chosen = None
        if sample_name:
            chosen = samples.get(sample_name)
            if chosen is None:
                raise ValueError(
                    f"sample {sample_name!r} not in project; have {sorted(samples)}"
                )
        elif samples:
            chosen = max(samples.values(), key=lambda s: s.get("created_at") or "")
        if chosen is not None:
            raster_var = chosen["raster_var_name"]
            mask_var = chosen.get("mask_var_name")
            design = {
                "strategy": chosen.get("strategy") or "stratified",
                "n_samples": chosen.get("n_samples"),
                "allocation": chosen.get("allocation") or "equal",
            }
            source = f"sample {chosen.get('name')!r}"
        else:
            datasets = list((manifest.get("datasets") or {}).values())
            target = datasets[0].get("target_name") if datasets else None
            if not target:
                raise ValueError(
                    "project has no samples and no dataset target; set "
                    f"{RASTER_VAR_ENV} to a raster variable name"
                )
            raster_var, source = target, "dataset target"

    def path_of(name):
        if not name:
            return None
        var = variables.get(name)
        if var is None or not var.get("path"):
            raise ValueError(
                f"variable {name!r} not found in project or has no path; "
                f"rasters: {sorted(k for k, v in variables.items() if v.get('path'))}"
            )
        path = Path(var["path"])
        if not path.is_absolute():
            path = folder / path
        return path

    return {
        "raster": path_of(raster_var),
        "mask": path_of(mask_var),
        "raster_var": raster_var,
        "mask_var": mask_var,
        "source": source,
        **design,
    }


def inputs_from_env() -> Dict:
    """Resolve the assessment inputs: explicit files, else a project manifest."""
    inputs = None
    if os.environ.get(RASTER_ENV):
        mask = os.environ.get(MASK_ENV)
        inputs = {
            "raster": Path(os.environ[RASTER_ENV]).expanduser(),
            "mask": Path(mask).expanduser() if mask else None,
            "source": "explicit files",
            "strategy": "stratified",
            "n_samples": 10000,
            "allocation": "equal",
        }
    else:
        name = os.environ.get(PROJECT_ENV) or DEFAULT_PROJECT
        manifest, folder = read_manifest(name)
        sample = os.environ.get(SAMPLE_ENV)
        if sample is None and DEFAULT_SAMPLE in (manifest.get("samples") or {}):
            sample = DEFAULT_SAMPLE
        inputs = inputs_from_manifest(
            manifest,
            folder,
            sample_name=sample,
            raster_var=os.environ.get(RASTER_VAR_ENV),
            mask_var=os.environ.get(MASK_VAR_ENV),
        )
        inputs["source"] = f"project {name!r} ({folder}), {inputs['source']}"
    if os.environ.get(STRATEGY_ENV):
        inputs["strategy"] = os.environ[STRATEGY_ENV]
    if os.environ.get(N_SAMPLES_ENV):
        inputs["n_samples"] = int(os.environ[N_SAMPLES_ENV])
    if inputs.get("n_samples") is None:
        inputs["n_samples"] = 10000  # a spacing-only systematic design has none
    if os.environ.get(ALLOCATION_ENV):
        inputs["allocation"] = os.environ[ALLOCATION_ENV]
    return inputs


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


def _stripe_plan(raster, mask) -> Dict:
    from spatialrisk.sampling import blocked

    with blocked.RasterScan(raster, mask) as scan:
        with_mask = scan._msrc is not None
        px = scan.rows_per_stripe * scan.width
        return {
            "shape": scan.shape,
            "rows_per_stripe": scan.rows_per_stripe,
            "n_stripes": len(scan.windows),
            "stripe_pixels": px,
            "stripe_working_set": stripe_working_set_bytes(px, with_mask),
            "with_mask": with_mask,
        }


def assess(
    raster,
    mask,
    *,
    strategy,
    n_samples,
    workers_spec,
    allocation="equal",
    source=None,
    out=None,
) -> Dict:
    """Print the assessment table for one raster and return every record."""
    from spatialrisk.gdal_env import DEFAULT_SAMPLING_CACHEMAX_BYTES

    mib = 1024 * 1024
    cores = affinity_cores()
    cg = cgroup_memory()
    host = host_memory()
    free, free_source = free_memory_bytes(cg, host)
    plan = _stripe_plan(raster, mask)
    policy = reference_policy(
        cores,
        free,
        plan["stripe_working_set"],
        gdal_cache_bytes=DEFAULT_SAMPLING_CACHEMAX_BYTES,
    )

    def fmt(b):
        return "n/a" if b is None else f"{b / mib:.0f} MiB"

    print()
    if source:
        print(f"inputs: {source}")
    print(f"raster: {raster}")
    print(f"mask:   {mask or '-'}")
    print(f"strategy={strategy} n_samples={n_samples} allocation={allocation}")
    print(
        f"cores: affinity={cores} cpu_count={os.cpu_count()} | "
        f"cgroup v{cg['version']} limit={fmt(cg['limit'])} usage={fmt(cg['usage'])} | "
        f"host total={fmt(host['total'])} available={fmt(host['available'])}"
    )
    print(f"free for a job: {fmt(free)} ({free_source})")
    print(
        f"stripes: {plan['n_stripes']} x {plan['rows_per_stripe']} rows of "
        f"{plan['shape'][1]} px = {fmt(plan['stripe_working_set'])} working set each"
    )
    print(
        f"reference policy (half cores, 50% free RAM): {policy['workers']} workers "
        f"(by cores {policy['by_cores']}, by memory {policy['by_memory']})"
    )

    base_cfg = {
        "raster": str(raster),
        "mask": str(mask) if mask else None,
        "strategy": strategy,
        "n_samples": n_samples,
        "allocation": allocation,
    }
    base = spawn_stage({**base_cfg, "stage": "baseline"})
    print(f"import baseline: {base['peak_mib']:.0f} MiB\n")

    header = (
        f"{'stage':<13}{'workers':>8}{'seconds':>9}{'peak MiB':>10}{'net MiB':>9}"
        "  detail"
    )
    print(header)
    print("-" * len(header))
    records = []
    plan_rows = [("serial_pass1", None), ("serial_full", None)]
    for w in _resolve_workers(workers_spec, cores):
        plan_rows += [("pool_pass1", w), ("pool_full", w)]
    for stage, workers in plan_rows:
        cfg = {**base_cfg, "stage": stage}
        if workers is not None:
            cfg["workers"] = workers
        rec = spawn_stage(cfg)
        rec.update(stage=stage, workers=workers)
        records.append(rec)
        print(
            f"{stage:<13}{_workers_label(stage, workers):>8}"
            f"{rec['seconds']:>9.1f}{rec['peak_mib']:>10.0f}"
            f"{rec['peak_mib'] - base['peak_mib']:>9.0f}  {rec['detail']}"
        )

    result = {
        "cores": cores,
        "cgroup": cg,
        "host": host,
        "free": free,
        "free_source": free_source,
        "stripes": plan,
        "policy": policy,
        "baseline_mib": base["peak_mib"],
        "records": records,
    }
    if out:
        Path(out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nrecords written to {out}")
    return result


def _workers_label(stage: str, workers) -> str:
    """``-`` for the serial rows, ``auto`` for a pool row left to the policy."""
    if workers is not None:
        return str(workers)
    return "auto" if stage.startswith("pool") else "-"


def _digests_agree(records) -> bool:
    """Every pass-1 record reports the same counts, every full run the same points.

    Serial and pooled records are compared within their stage family (``*_pass1``
    against each other, ``*_full`` against each other), whatever the worker count.
    """
    families: Dict[str, list] = {}
    for r in records:
        if r.get("digest") is not None:
            families.setdefault(r["stage"].split("_", 1)[1], []).append(r["digest"])
    return all(all(d == ds[0] for d in ds[1:]) for ds in families.values())


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_parse_cgroup_bytes_treats_max_and_v1_sentinel_as_unlimited():
    """``max`` (v2), the v1 2**63 sentinel and blanks all mean "no limit"."""
    assert parse_cgroup_bytes("max\n") is None
    assert parse_cgroup_bytes("") is None
    assert parse_cgroup_bytes("9223372036854771712") is None
    assert parse_cgroup_bytes("4294967296\n") == 4 * 1024**3


def test_free_memory_takes_the_tighter_of_cgroup_and_host():
    """A container limit below the host's free memory is the binding one."""
    host = {"total": 64 << 30, "available": 48 << 30}
    unlimited = {"version": None, "limit": None, "usage": None}
    assert free_memory_bytes(unlimited, host) == (48 << 30, "psutil.available")
    tight = {"version": 2, "limit": 8 << 30, "usage": 3 << 30}
    free, source = free_memory_bytes(tight, host)
    assert free == 5 << 30 and source.startswith("cgroup v2")
    loose = {"version": 1, "limit": 200 << 30, "usage": 1 << 30}
    assert free_memory_bytes(loose, host)[1] == "psutil.available"


def test_reference_policy_is_half_cores_capped_by_half_free_memory():
    """Half the cores, unless half the free RAM minus the cache holds fewer stripes."""
    gib = 1 << 30
    roomy = reference_policy(16, 48 * gib, 120 << 20, gdal_cache_bytes=512 << 20)
    assert roomy == {"by_cores": 8, "by_memory": roomy["by_memory"], "workers": 8}
    assert roomy["by_memory"] > 8
    # 4 GiB free: 2 GiB budget - 512 MiB cache = 1.5 GiB / 120 MiB -> 12, so
    # cores still bind on 16 cores but memory binds on a 2-core, 1 GiB box.
    small = reference_policy(2, 1 * gib, 120 << 20, gdal_cache_bytes=512 << 20)
    assert small["workers"] == 1
    assert reference_policy(1, 0, 120 << 20, gdal_cache_bytes=0)["workers"] == 1


def _manifest(samples=(), target=None):
    """A manifest shaped like the app's ``<project>_project.json``."""
    variables = {
        "loss": {"path": "/p/loss.tif"},
        "forest": {"path": "/p/forest.tif"},
        "rel": {"path": "data/rel.tif"},
        "nopath": {"path": None},
    }
    datasets = {"ds": {"target_name": target}} if target else {}
    return {
        "processed_variables": variables,
        "samples": {s["name"]: s for s in samples},
        "datasets": datasets,
    }


def _sample(name, created_at, *, mask="forest", strategy="stratified"):
    return {
        "name": name,
        "created_at": created_at,
        "raster_var_name": "loss",
        "mask_var_name": mask,
        "strategy": strategy,
        "n_samples": 2000,
        "allocation": "proportional",
    }


def test_inputs_from_manifest_prefer_the_latest_sample_design():
    """The most recent sample's raster, mask and design are what gets timed."""
    old = _sample("a", "2026-01-01T00:00:00", strategy="random", mask=None)
    new = _sample("b", "2026-09-01T00:00:00")
    got = inputs_from_manifest(_manifest([old, new]), Path("/proj"))
    assert got["raster"] == Path("/p/loss.tif")
    assert got["mask"] == Path("/p/forest.tif")
    assert (got["strategy"], got["n_samples"], got["allocation"]) == (
        "stratified",
        2000,
        "proportional",
    )
    assert got["source"] == "sample 'b'"
    # a named sample wins over recency; explicit variables win over both
    by_name = inputs_from_manifest(
        _manifest([old, new]), Path("/proj"), sample_name="a"
    )
    assert by_name["strategy"] == "random" and by_name["mask"] is None
    explicit = inputs_from_manifest(
        _manifest([old, new]), Path("/proj"), raster_var="rel"
    )
    assert explicit["raster"] == Path("/proj/data/rel.tif")  # relative -> project
    assert explicit["mask"] is None


def test_inputs_from_manifest_falls_back_to_the_dataset_target():
    """No samples: the dataset target with no mask; nothing at all: a clear error."""
    got = inputs_from_manifest(_manifest(target="loss"), Path("/proj"))
    assert got["raster"] == Path("/p/loss.tif") and got["mask"] is None
    assert got["source"] == "dataset target"
    with pytest.raises(ValueError, match="no samples and no dataset target"):
        inputs_from_manifest(_manifest(), Path("/proj"))
    with pytest.raises(ValueError, match="not in project"):
        inputs_from_manifest(
            _manifest([_sample("a", "x")]), Path("/proj"), sample_name="z"
        )
    with pytest.raises(ValueError, match="has no path"):
        inputs_from_manifest(_manifest(), Path("/proj"), raster_var="nopath")


def _write_raster(path, array, *, nodata=255):
    """Write a tiled single-band GeoTIFF with 1 x 1 m pixels."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype.name,
        crs="EPSG:3857",
        transform=from_origin(0, array.shape[0], 1, 1),
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(array, 1)


@pytest.mark.raster
@pytest.mark.slow
def test_assessment_machinery_runs_on_a_synthetic_raster(tmp_path):
    """The whole subprocess pipeline works and the pool agrees with the serial pass."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 3, size=(1100, 900)).astype("uint8")
    arr[rng.random(arr.shape) < 0.05] = 255
    mask = np.ones(arr.shape, dtype="uint8")
    mask[:, 700:] = 0
    _write_raster(tmp_path / "strata.tif", arr)
    _write_raster(tmp_path / "mask.tif", mask, nodata=0)

    result = assess(
        tmp_path / "strata.tif",
        tmp_path / "mask.tif",
        strategy="stratified",
        n_samples=500,
        workers_spec="1,2",
        out=tmp_path / "records.json",
    )
    stages = [(r["stage"], r["workers"]) for r in result["records"]]
    assert stages == [
        ("serial_pass1", None),
        ("serial_full", None),
        ("pool_pass1", 1),
        ("pool_full", 1),
        ("pool_pass1", 2),
        ("pool_full", 2),
    ]
    assert _digests_agree(result["records"])
    assert result["policy"]["workers"] >= 1
    assert (tmp_path / "records.json").exists()


@pytest.mark.raster
@pytest.mark.slow
def test_sepal_sampling_assessment():
    """Time the production sampling path and the stripe-pool probe on a real raster."""
    try:
        inputs = inputs_from_env()
    except FileNotFoundError as exc:
        pytest.skip(f"not on a machine with the project: {exc}")
    raster, mask = inputs["raster"], inputs["mask"]
    assert raster.exists(), raster
    assert mask is None or mask.exists(), mask

    result = assess(
        raster,
        mask,
        strategy=inputs["strategy"],
        n_samples=inputs["n_samples"],
        allocation=inputs["allocation"],
        source=inputs["source"],
        workers_spec=os.environ.get(WORKERS_ENV, "1,2,half,all"),
        out=os.environ.get(OUT_ENV),
    )
    assert _digests_agree(result["records"]), "pool counts differ from the serial pass"


if __name__ == "__main__":
    # Worker entry: ``python tests/test_sampling_bench_sepal.py --worker '<json>'``
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(run_stage(json.loads(sys.argv[2]))))
    else:
        sys.exit("run this file through pytest; --worker is internal")
