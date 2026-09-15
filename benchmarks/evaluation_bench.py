"""Benchmark the evaluation tallies: per-square Categorical vs band-wise threads.

Runs the pre-optimisation loops (``original``, inlined below verbatim from
``compute_validation`` / ``defrate_per_cat``) and the current library code
(``threaded``) on the same defor / forest / risk stack, each stage in a fresh
subprocess, and reports wall clock, CPU time (user + system) and peak RSS.
The import baseline is measured once and subtracted. Both tallies are hashed
and compared, so a speed-up that changes results is reported as a failure.

By default a synthetic stack (uint8 defor + forest, uint16 risk with nodata
holes) of ``--size`` pixels is written to a temporary directory. Pass real
rasters instead::

    python benchmarks/evaluation_bench.py
    python benchmarks/evaluation_bench.py --size 20000 20000 --csizes 300,1000
    python benchmarks/evaluation_bench.py --rasters defor.tif forest.tif risk.tif

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

IMPLS = ("original", "threaded")


def _peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


# --------------------------------------------------------------------------
# Reference: the pre-optimisation loops, copied verbatim.
# --------------------------------------------------------------------------
def _original_validation(defor_file, forest_file, riskmap_file, tab, csize):
    import numpy as np
    import pandas as pd
    from osgeo import gdal

    from spatialrisk.evaluation import make_square

    defor_ds = gdal.Open(str(defor_file))
    defor_band = defor_ds.GetRasterBand(1)
    forest_ds = gdal.Open(str(forest_file))
    forest_band = forest_ds.GetRasterBand(1)
    risk_ds = gdal.Open(str(riskmap_file))
    risk_band = risk_ds.GetRasterBand(1)
    dens = pd.read_csv(tab)
    cat = dens["cat"].values
    defor_dens_period = dens["defor_dens"].values * 5.0
    nsquare, nsquare_x, _, x, y, nx, ny = make_square(defor_file, csize)
    df = pd.DataFrame(
        {
            "cell": list(range(nsquare)),
            "nfor_obs": 0,
            "ndefor_obs": 0,
            "ndefor_pred_ha": 0.0,
        }
    )
    for s in range(nsquare):
        px, py = s % nsquare_x, s // nsquare_x
        defor_data = defor_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        forest_data = forest_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        defor_mask = defor_data == 1
        forest_start = (forest_data == 1) | defor_mask
        df.loc[s, "nfor_obs"] = int(forest_start.sum())
        df.loc[s, "ndefor_obs"] = int(defor_mask.sum())
        risk_data = risk_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        risk_cat = pd.Categorical(risk_data.flatten(), categories=cat)
        risk_count = risk_cat.value_counts().values
        df.loc[s, "ndefor_pred_ha"] = np.nansum(risk_count * defor_dens_period)
    del defor_ds, forest_ds, risk_ds
    return df[df["nfor_obs"] > 0]


def _original_defrate(defor_file, forest_file, riskmap_file, blk_rows=128):
    import pandas as pd
    from osgeo import gdal
    from riskmapjnr.misc import makeblock

    defor_ds = gdal.Open(str(defor_file))
    forest_ds = gdal.Open(str(forest_file))
    cat_ds = gdal.Open(str(riskmap_file))
    defor_band, forest_band, cat_band = (
        defor_ds.GetRasterBand(1),
        forest_ds.GetRasterBand(1),
        cat_ds.GetRasterBand(1),
    )
    nblock, nblock_x, _, x, y, nx, ny = makeblock(str(defor_file), blk_rows=blk_rows)[
        :7
    ]
    cat = [c + 1 for c in range(65535)]
    df = pd.DataFrame({"cat": cat, "nfor": 0, "ndefor": 0})
    for b in range(nblock):
        px, py = b % nblock_x, b // nblock_x
        defor_arr = defor_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        forest_arr = forest_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        cat_arr = cat_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        defor_mask = defor_arr == 1
        forest_start = (forest_arr == 1) | defor_mask
        cat_for = pd.Categorical(cat_arr[forest_start].flatten(), categories=cat)
        df["nfor"] += cat_for.value_counts().values
        cat_defor = pd.Categorical(cat_arr[defor_mask].flatten(), categories=cat)
        df["ndefor"] += cat_defor.value_counts().values
    return df


def _digest(df, cols):
    import pandas as pd

    h = pd.util.hash_pandas_object(df[cols].reset_index(drop=True), index=False)
    return hashlib.sha256(h.to_numpy().tobytes()).hexdigest()[:16]


def _worker(args) -> dict:
    """Run one (impl, stage) in this process and return its metrics."""
    from spatialrisk.evaluation import compute_validation
    from spatialrisk.rmj.deforrate import defrate_per_cat

    defor, forest, risk = args.rasters
    tab = Path(args.tab)
    base_cpu = _cpu_s()
    t0 = time.perf_counter()
    if args.stage == "defrate":
        if args.impl == "original":
            df = _original_defrate(defor, forest, risk)
        else:
            df = defrate_per_cat(defor, forest, risk, 5.0, tab_file_defrate=tab)
        digest = _digest(df, ["nfor", "ndefor"])
        detail = f"{int(df['ndefor'].sum())} defor px"
    else:
        csize = int(args.stage)
        if args.impl == "original":
            df = _original_validation(defor, forest, risk, tab, csize)
        else:
            df = compute_validation(
                defor, forest, risk, tab, 5.0, csize
            ).plot_data.points
        digest = _digest(df, ["cell", "nfor_obs", "ndefor_obs", "ndefor_pred_ha"])
        detail = f"{len(df)} cells"
    return {
        "impl": args.impl,
        "stage": args.stage,
        "wall_s": time.perf_counter() - t0,
        "cpu_s": _cpu_s() - base_cpu,
        "peak_mib": _peak_mib(),
        "detail": detail,
        "digest": digest,
    }


def _spawn(argv):
    out = subprocess.run(
        [sys.executable, __file__, *argv], check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write_synthetic(tmp, size, seed):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    h, w = size
    rng = np.random.default_rng(seed)
    base = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        nodata=0,
        crs="EPSG:3857",
        transform=from_origin(0, h * 30, 30, 30),
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    )
    specs = {"defor": "uint8", "forest": "uint8", "risk": "uint16"}
    paths = []
    for name, dtype in specs.items():
        p = tmp / f"{name}.tif"
        with rasterio.open(p, "w", dtype=dtype, **base) as dst:
            for _, win in dst.block_windows(1):
                shape = (win.height, win.width)
                if name == "forest":
                    a = (rng.random(shape) < 0.6).astype(dtype)
                elif name == "defor":
                    a = (rng.random(shape) < 0.02).astype(dtype)
                else:
                    a = rng.integers(1, 1001, size=shape).astype(dtype)
                    a[rng.random(shape) < 0.4] = 0
                dst.write(a, 1, window=win)
        paths.append(str(p))
    return paths


def main():
    """Parse arguments, build the fixture if needed, run both implementations."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rasters", nargs=3, metavar=("DEFOR", "FOREST", "RISK"))
    ap.add_argument(
        "--size", nargs=2, type=int, default=(8000, 8000), metavar=("H", "W")
    )
    ap.add_argument("--csizes", default="300")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--impls", default=",".join(IMPLS))
    # Internal subprocess entry points.
    ap.add_argument("--worker", choices=IMPLS, dest="impl")
    ap.add_argument("--stage")
    ap.add_argument("--tab")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    if args.baseline:
        import spatialrisk.evaluation
        import spatialrisk.rmj.deforrate  # noqa: F401

        print(json.dumps({"peak_mib": _peak_mib()}))
        return
    if args.impl:
        print(json.dumps(_worker(args)))
        return

    tmp = tempfile.TemporaryDirectory(prefix="evaluation_bench_")
    if not args.rasters:
        print(f"writing 3 x {args.size[0]}x{args.size[1]} layers ...", flush=True)
        args.rasters = _write_synthetic(Path(tmp.name), args.size, args.seed)
    tab = str(Path(tmp.name) / "defrate.csv")
    base = _spawn(["--baseline"])
    common = ["--rasters", *args.rasters, "--tab", tab]
    stages = ["defrate"] + args.csizes.split(",")
    rows = []
    for stage in stages:
        # The threaded defrate writes the CSV the validation stages consume.
        for impl in reversed(args.impls.split(",")):
            for _ in range(args.repeat):
                rows.append(_spawn(["--worker", impl, "--stage", stage, *common]))

    import rasterio

    with rasterio.open(args.rasters[0]) as src:
        mpx = src.width * src.height / 1e6
    print(
        f"\n{mpx:.0f} Mpx per layer, import baseline {base['peak_mib']:.0f} MiB "
        "(subtracted from 'run MiB')"
    )
    hdr = (
        f"{'stage':<9} {'impl':<9} {'wall s':>8} {'cpu s':>8} {'peak MiB':>9} "
        f"{'run MiB':>8}  detail            digest"
    )
    print(hdr)
    print("-" * len(hdr))
    mismatch = False
    for stage in stages:
        got = [r for r in rows if r["stage"] == stage]
        for r in got:
            print(
                f"{r['stage']:<9} {r['impl']:<9} {r['wall_s']:>8.2f} "
                f"{r['cpu_s']:>8.2f} {r['peak_mib']:>9.0f} "
                f"{r['peak_mib'] - base['peak_mib']:>8.0f}  "
                f"{r['detail']:<17} {r['digest']}"
            )
        if len({r["digest"] for r in got}) != 1:
            mismatch = True
    print("\noutputs identical: " + ("NO  <-- results differ!" if mismatch else "YES"))
    tmp.cleanup()
    if mismatch:
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
