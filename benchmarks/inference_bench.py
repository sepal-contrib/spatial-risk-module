"""Benchmark the ML predictors' ``apply``: 128-row bands vs tile-aligned bands.

Runs the pre-optimisation block loop (``original``, inlined below verbatim:
forestatrisk's default 128-row bands, BLAS threads left to OpenBLAS) and the
current ``apply`` (``current``: 256-row bands aligned to the output tiles,
BLAS pinned to one thread) on the same feature stack and the same fitted
model, each in a fresh subprocess, and reports wall clock, CPU time
(user + system) and peak RSS. Fitting happens before the clock starts. Both
rasters are hashed and compared so a speed-up that changes results is
reported as a failure, not a win.

A synthetic stack is written to a temporary directory: ``--layers`` tiled
float32 GeoTIFFs of ``--size`` pixels with nodata holes, plus a uint8 target.
The model is a GLM by default; ``--model rf`` fits a 100-tree random forest::

    python benchmarks/inference_bench.py
    python benchmarks/inference_bench.py --size 8000 8000 --layers 4
    python benchmarks/inference_bench.py --model rf --size 3000 3000

Output is a plain table on stdout.
"""
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

IMPLS = ("original", "current")

# Measure the checkout this file lives in, not whichever one the editable
# install points at: a script puts its own directory on sys.path, so without
# this a worktree's benchmark would silently import the main checkout.
_REPO = Path(__file__).resolve().parents[1]


def _spawn_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


# --------------------------------------------------------------------------
# Reference: the pre-optimisation block loop shared by glm/rf ``apply``,
# copied verbatim (128-row bands, no BLAS pinning).
# --------------------------------------------------------------------------
def _original_apply(model, dataset, output_file):
    import forestatrisk as far
    import numpy as np
    import pandas as pd
    import rasterio
    from patsy.highlevel import build_design_matrices

    from spatialrisk.raster_profile import rasterio_profile

    feature_paths = {var.name: var.path for var in dataset.features}
    with rasterio.open(dataset.target.path) as ref:
        profile = ref.profile.copy()
    profile.update(dtype="uint16", count=1, nodata=0)
    profile.update(rasterio_profile("uint16"))

    with rasterio.open(output_file, "w", **profile) as dst:
        blockinfo = far.misc.makeblock(str(dataset.target.path))
        nblock, nblock_x = blockinfo[0], blockinfo[1]
        x_off, y_off, nx, ny = blockinfo[3], blockinfo[4], blockinfo[5], blockinfo[6]

        for b in range(nblock):
            px = b % nblock_x
            py = b // nblock_x
            col_start, row_start = x_off[px], y_off[py]
            n_cols, n_rows = nx[px], ny[py]
            window = rasterio.windows.Window(col_start, row_start, n_cols, n_rows)

            mask_invalid = np.zeros(n_rows * n_cols, dtype=bool)

            block_dict = {}
            for name, path in feature_paths.items():
                with rasterio.open(path) as src:
                    arr = src.read(1, window=window).astype(float)
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                block_dict[name] = arr.ravel()

            block_df_full = pd.DataFrame(block_dict)
            valid_mask = ~block_df_full.isnull().any(axis=1).to_numpy() & ~mask_invalid
            block_df = block_df_full[valid_mask]

            out_arr = np.zeros(n_rows * n_cols, dtype=np.uint16)

            if not block_df.empty:
                (x_block,) = build_design_matrices(
                    [model._x_design_info], block_df, NA_action="drop"
                )
                proba = model._ml_model.predict_proba(np.asarray(x_block))[:, 1]
                out_arr[valid_mask] = far.misc.rescale(proba).astype(np.uint16)

            dst.write(out_arr.reshape(n_rows, n_cols), 1, window=window)
    return output_file


class _Var:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.raster_type = "continuous"


class _Dataset:
    name, year = "bench", 2020

    def __init__(self, rasters, seed):
        self.target = _Var("target", Path(rasters[0]))
        self.features = [_Var(f"f{i}", Path(p)) for i, p in enumerate(rasters[1:])]
        self._seed = seed

    def extract_at_points(self, points, *, drop_nodata=True):
        """A deterministic training sample with a real signal in f0."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(self._seed)
        n = 5000
        d = {v.name: rng.random(n) * 100 for v in self.features}
        d["target"] = (d["f0"] / 100 + rng.random(n) * 0.5 > 0.6).astype(int)
        d["trial"] = 1
        d["cell_id"] = np.arange(n)
        return pd.DataFrame(d)


class _Sample:
    name = "bench"

    def load_points(self):
        return object()


def _fit(args, folder):
    ds = _Dataset(args.rasters, args.seed)
    if args.model == "rf":
        from spatialrisk.mlmodels.rf_model import RFModel

        model = RFModel(name="bench", n_trees=100, random_seed=args.seed)
    else:
        from spatialrisk.mlmodels.glm_model import GLMModel

        model = GLMModel(name="bench", random_seed=args.seed)
    model.dataset = ds
    model.sample = _Sample()
    model.formula = "target + trial ~ " + " + ".join(v.name for v in ds.features)
    model.fit(folder=folder)
    if model._x_design_info is None:
        # GLM rebuilds this lazily inside apply(); do it up front so the
        # reference loop sees the same design and the clock excludes it.
        import pandas as pd
        from patsy import dmatrices

        _, x_ref = dmatrices(
            model.formula, pd.read_csv(model.samples_path).dropna(), NA_action="drop"
        )
        model._x_design_info = x_ref.design_info
    model._register_prediction = lambda *a, **k: None
    return model, ds


def _worker(args) -> dict:
    """Fit, then run one implementation in this process and return its metrics."""
    import numpy as np
    import rasterio

    with tempfile.TemporaryDirectory(prefix="inference_bench_out_") as out:
        model, ds = _fit(args, Path(out))
        pred = Path(out) / f"pred_{args.impl}.tif"

        base_mib, base_cpu = _peak_mib(), _cpu_s()
        t0 = time.perf_counter()
        if args.impl == "original":
            _original_apply(model, ds, pred)
        else:
            model.apply(output_file=pred)
        wall = time.perf_counter() - t0
        cpu = _cpu_s() - base_cpu
        peak = _peak_mib()
        with rasterio.open(pred) as src:
            arr = src.read(1)
            blocks = src.block_shapes[0]
        digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    import spatialrisk

    return {
        "impl": args.impl,
        "spatialrisk": str(Path(spatialrisk.__file__).resolve().parent),
        "wall_s": wall,
        "cpu_s": cpu,
        "peak_mib": peak,
        "setup_peak_mib": base_mib,
        "blocks": blocks,
        "digest": digest,
    }


def _spawn(argv):
    out = subprocess.run(
        [sys.executable, __file__, *argv],
        check=True,
        capture_output=True,
        text=True,
        env=_spawn_env(),
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write_synthetic(tmp, size, layers, seed):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    h, w = size
    rng = np.random.default_rng(seed)
    common = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        crs="EPSG:3857",
        transform=from_origin(0, h * 30, 30, 30),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    )
    paths = []
    p = tmp / "target.tif"
    with rasterio.open(p, "w", dtype="uint8", nodata=255, **common) as dst:
        for _, win in dst.block_windows(1):
            blk = (rng.random((win.height, win.width)) < 0.3).astype("uint8")
            dst.write(blk, 1, window=win)
    paths.append(str(p))
    for i in range(layers):
        p = tmp / f"f{i}.tif"
        with rasterio.open(p, "w", dtype="float32", nodata=-9999.0, **common) as dst:
            for _, win in dst.block_windows(1):
                blk = rng.random((win.height, win.width), dtype=np.float32) * 100
                blk[rng.random(blk.shape) < 0.05] = -9999.0
                dst.write(blk, 1, window=win)
        paths.append(str(p))
    return paths


def main():
    """Parse arguments, build the fixture, run both implementations."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=("glm", "rf"), default="glm")
    ap.add_argument(
        "--size", nargs=2, type=int, default=(4000, 4000), metavar=("H", "W")
    )
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--impls", default=",".join(IMPLS))
    # Internal subprocess entry point.
    ap.add_argument("--worker", choices=IMPLS, dest="impl")
    ap.add_argument("--rasters", nargs="+", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.impl:
        print(json.dumps(_worker(args)))
        return

    tmp = tempfile.TemporaryDirectory(prefix="inference_bench_")
    print(
        f"writing target + {args.layers} x {args.size[0]}x{args.size[1]} "
        "float32 layers ...",
        flush=True,
    )
    args.rasters = _write_synthetic(Path(tmp.name), args.size, args.layers, args.seed)
    common = [
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--rasters",
        *args.rasters,
    ]
    rows = []
    for impl in args.impls.split(","):
        for _ in range(args.repeat):
            rows.append(_spawn(["--worker", impl, *common]))

    mpx = args.size[0] * args.size[1] / 1e6
    print(f"\nspatialrisk from {rows[0]['spatialrisk']}")
    print(
        f"{args.model} on {args.layers} layers, {mpx:.0f} Mpx, "
        f"{os.cpu_count()} cpus (setup peak, i.e. imports + fit, in its own column)"
    )
    hdr = (
        f"{'impl':<10} {'wall s':>8} {'cpu s':>8} {'peak MiB':>10} "
        f"{'setup MiB':>10} {'blocks':>10}  digest"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['impl']:<10} {r['wall_s']:>8.2f} {r['cpu_s']:>8.2f} "
            f"{r['peak_mib']:>10.0f} {r['setup_peak_mib']:>10.0f} "
            f"{str(tuple(r['blocks'])):>10}  {r['digest']}"
        )
    digests = {r["digest"] for r in rows}
    print(
        "\noutputs identical: "
        + ("YES" if len(digests) == 1 else "NO  <-- results differ!")
    )
    tmp.cleanup()
    if len(digests) != 1:
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
