"""Benchmark ``Dataset.extract_at_points``: full-band reads vs windowed reads.

Runs the pre-optimisation algorithm (``original``, inlined below verbatim) and
the current ``Dataset.extract_at_points`` (``windowed``) on the same stack and
point set, each in a fresh subprocess, and reports wall clock, CPU time
(user + system) and peak RSS for each. The import baseline (numpy, rasterio,
geopandas, spatialrisk) is measured once and subtracted so the extraction's
own footprint is visible. Both outputs are hashed and compared so a speed-up
that changes results is reported as a failure, not a win.

By default a synthetic stack is written to a temporary directory: ``--layers``
tiled float32 GeoTIFFs of ``--size`` pixels with nodata holes. Pass real
rasters instead (first one is the target)::

    python benchmarks/extract_bench.py
    python benchmarks/extract_bench.py --size 20000 20000 --layers 4 --points 20000
    python benchmarks/extract_bench.py --rasters target.tif dist_road.tif elev.tif

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

IMPLS = ("original", "windowed")


def _peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


# --------------------------------------------------------------------------
# Reference: the pre-optimisation algorithm, copied verbatim from dataset.py.
# --------------------------------------------------------------------------
def _original_extract(ds, points, *, drop_nodata=True):
    import numpy as np
    import pandas as pd
    import rasterio

    all_vars = [ds.target] + ds.features
    n_pts = len(points)
    df_data = {}
    valid = np.ones(n_pts, dtype=bool)
    target_cell = None
    for i, var in enumerate(all_vars):
        with rasterio.open(var.path) as src:
            arr = src.read(1)
            nodata = src.nodata
            vcrs = src.crs
            vtransform = src.transform
            vheight, vwidth = src.height, src.width
        vpts = points.to_crs(vcrs) if points.crs != vcrs else points
        r, c = rasterio.transform.rowcol(
            vtransform, vpts.geometry.x.to_numpy(), vpts.geometry.y.to_numpy()
        )
        r = np.asarray(r, dtype=int)
        c = np.asarray(c, dtype=int)
        in_bounds = (r >= 0) & (r < vheight) & (c >= 0) & (c < vwidth)
        rc = np.clip(r, 0, vheight - 1)
        cc = np.clip(c, 0, vwidth - 1)
        vals = arr[rc, cc]
        layer_valid = in_bounds.copy()
        if np.issubdtype(vals.dtype, np.floating):
            layer_valid &= ~np.isnan(vals)
        if nodata is not None:
            layer_valid &= vals != nodata
        valid &= layer_valid
        df_data[var.name] = vals
        if i == 0:
            target_cell = r * vwidth + c
    df_data["cell_id"] = target_cell
    df_data["trial"] = 1
    df = pd.DataFrame(df_data)
    if drop_nodata:
        df = df[valid].reset_index(drop=True)
    return df


class _Var:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.year = None


def _open_bounds(path):
    import rasterio

    with rasterio.open(path) as src:
        return src.bounds, src.crs


def _make_points(rasters, n, seed):
    """Random points over the target footprint, with a 2% margin outside it."""
    import geopandas as gpd
    import numpy as np

    b, crs = _open_bounds(rasters[0])
    rng = np.random.default_rng(seed)
    mx, my = 0.02 * (b.right - b.left), 0.02 * (b.top - b.bottom)
    xs = rng.uniform(b.left - mx, b.right + mx, n)
    ys = rng.uniform(b.bottom - my, b.top + my, n)
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs)


def _worker(args) -> dict:
    """Run one implementation in this process and return its metrics."""
    import pandas as pd

    from spatialrisk.dataset import Dataset

    ds = Dataset(project=None, name="bench")
    ds.target = _Var("target", Path(args.rasters[0]))
    ds.features = [_Var(f"f{i}", Path(p)) for i, p in enumerate(args.rasters[1:], 1)]
    pts = _make_points(args.rasters, args.points, args.seed)

    base_mib, base_cpu = _peak_mib(), _cpu_s()
    t0 = time.perf_counter()
    if args.impl == "original":
        df = _original_extract(ds, pts)
    else:
        df = ds.extract_at_points(pts)
    wall = time.perf_counter() - t0
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    ).hexdigest()[:16]
    return {
        "impl": args.impl,
        "wall_s": wall,
        "cpu_s": _cpu_s() - base_cpu,
        "peak_mib": _peak_mib(),
        "setup_peak_mib": base_mib,
        "rows": int(len(df)),
        "digest": digest,
    }


def _baseline() -> dict:
    return {"peak_mib": _peak_mib()}


def _spawn(argv):
    out = subprocess.run(
        [sys.executable, __file__, *argv], check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write_synthetic(tmp, size, layers, seed):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    h, w = size
    rng = np.random.default_rng(seed)
    paths = []
    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        nodata=-9999.0,
        crs="EPSG:3857",
        transform=from_origin(0, h * 30, 30, 30),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    )
    for i in range(layers):
        p = tmp / f"layer{i}.tif"
        with rasterio.open(p, "w", **profile) as dst:
            for _, win in dst.block_windows(1):
                blk = rng.random((win.height, win.width), dtype=np.float32) * 100
                blk[rng.random(blk.shape) < 0.05] = -9999.0
                dst.write(blk, 1, window=win)
        paths.append(str(p))
    return paths


def main():
    """Parse arguments, build the fixture if needed, run both implementations."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rasters", nargs="+", help="target first, then features")
    ap.add_argument(
        "--size", nargs=2, type=int, default=(8000, 8000), metavar=("H", "W")
    )
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--points", type=int, default=10_000)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--impls", default=",".join(IMPLS))
    # Internal subprocess entry points.
    ap.add_argument("--worker", choices=IMPLS, dest="impl")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    if args.baseline:
        import geopandas  # noqa: F401

        import spatialrisk.dataset  # noqa: F401

        print(json.dumps(_baseline()))
        return
    if args.impl:
        print(json.dumps(_worker(args)))
        return

    tmp = None
    if not args.rasters:
        tmp = tempfile.TemporaryDirectory(prefix="extract_bench_")
        print(
            f"writing {args.layers} x {args.size[0]}x{args.size[1]} float32 layers ...",
            flush=True,
        )
        args.rasters = _write_synthetic(
            Path(tmp.name), args.size, args.layers, args.seed
        )
    import rasterio

    total_px = 0
    for p in args.rasters:
        with rasterio.open(p) as src:
            total_px += src.width * src.height
    base = _spawn(["--baseline"])
    common = [
        "--rasters",
        *args.rasters,
        "--points",
        str(args.points),
        "--seed",
        str(args.seed),
    ]
    rows = []
    for impl in args.impls.split(","):
        for _ in range(args.repeat):
            rows.append(_spawn(["--worker", impl, *common]))

    print(
        f"\n{len(args.rasters)} layers, {total_px / 1e6:.0f} Mpx total, "
        f"{args.points} points, import baseline {base['peak_mib']:.0f} MiB "
        "(subtracted from 'extract' column)"
    )
    hdr = (
        f"{'impl':<10} {'wall s':>8} {'cpu s':>8} {'peak MiB':>10} "
        f"{'extract MiB':>12} {'rows':>7}  digest"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['impl']:<10} {r['wall_s']:>8.2f} {r['cpu_s']:>8.2f} "
            f"{r['peak_mib']:>10.0f} {r['peak_mib'] - base['peak_mib']:>12.0f} "
            f"{r['rows']:>7}  {r['digest']}"
        )
    digests = {r["digest"] for r in rows}
    print(
        "\noutputs identical: "
        + ("YES" if len(digests) == 1 else "NO  <-- results differ!")
    )
    if tmp:
        tmp.cleanup()
    if len(digests) != 1:
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
