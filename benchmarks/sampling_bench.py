"""Benchmark the sample-generation read path on any raster.

Times the stages of ``spatialrisk.sampling.generate_points`` -- stripe decoding,
the pass-1 count and the full run -- across a list of ``GDAL_NUM_THREADS``
values, and reports wall clock plus peak RSS for each. Every configuration
runs in a fresh subprocess so the peak is per run, and the import baseline
(numpy, rasterio, spatialrisk) is measured once and shown separately so the
scan's own footprint is visible.

Examples::

    python benchmarks/sampling_bench.py loss.tif --mask forest.tif
    python benchmarks/sampling_bench.py dist.tif --strategy random --threads 1,4,8
    python benchmarks/sampling_bench.py dist.tif --stages decode,full --repeat 2

Any file rasterio can open works; the raster and mask must be co-registered,
exactly as ``generate_points`` requires. Output is a plain table on stdout.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

STAGES = ("decode", "pass1", "full")


def _peak_mib() -> float:
    """Peak resident set size of this process, in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _run_stage(args) -> dict:
    """Execute one stage in this process and return its timing record."""
    from spatialrisk.gdal_env import sampling_gdal_env
    from spatialrisk.sampling import blocked, generate_points

    cachemax = args.cachemax_mb * 1024 * 1024
    with sampling_gdal_env(cachemax_bytes=cachemax, num_threads=args.threads):
        t0 = time.perf_counter()
        if args.stage == "full":
            gdf = generate_points(
                args.raster,
                mask_path=args.mask,
                strategy=args.strategy,
                n_samples=args.n_samples,
                seed=args.seed,
                allocation=args.allocation,
                spacing_m=args.spacing_m,
            )
            detail = f"{len(gdf)} points"
        else:
            with blocked.RasterScan(args.raster, args.mask) as scan:
                if args.stage == "decode":
                    # Pure tile decoding of the same stripes the scan walks,
                    # without the validity predicate, so this isolates GDAL.
                    n = _decode_windows(args, scan.windows)
                    detail = f"{n / 1e6:.0f} Mpx, {len(scan.windows)} stripes"
                elif args.strategy == "stratified":
                    counts, n_valid = blocked.count_values(scan)
                    detail = f"{n_valid} valid, {len(counts)} classes"
                else:
                    detail = f"{blocked.count_valid(scan)} valid"
        seconds = time.perf_counter() - t0
    return {"seconds": seconds, "peak_mib": _peak_mib(), "detail": detail}


def _decode_windows(args, windows) -> int:
    """Read every window of the raster (and mask); return pixels decoded."""
    import rasterio

    n = 0
    paths = [args.raster] + ([args.mask] if args.mask else [])
    for path in paths:
        with rasterio.open(path) as src:
            for window in windows:
                n += src.read(1, window=window).size
    return n


def _baseline() -> dict:
    """Import cost alone, so scan footprints can be read net of it."""
    import spatialrisk.sampling  # noqa: F401  (import for its memory cost)

    return {"seconds": 0.0, "peak_mib": _peak_mib(), "detail": "imports only"}


def _spawn(argv, extra) -> dict:
    """Run this script as a worker subprocess and parse its JSON result."""
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", *argv, *extra]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def _parse(argv=None):
    """Build the CLI parser; ``--worker`` flags are internal."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("raster", help="raster to sample (any rasterio-readable file)")
    p.add_argument("--mask", help="co-registered mask raster (0/nodata = excluded)")
    p.add_argument(
        "--strategy",
        default="stratified",
        choices=("stratified", "random", "systematic"),
    )
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allocation", default="equal")
    p.add_argument("--spacing-m", type=float, default=None)
    p.add_argument("--cachemax-mb", type=int, default=512, help="GDAL_CACHEMAX")
    p.add_argument(
        "--threads",
        default="1,half",
        help="comma list of GDAL_NUM_THREADS; 'half' = the sampling default",
    )
    p.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma list from {', '.join(STAGES)}",
    )
    p.add_argument("--repeat", type=int, default=1, help="runs per configuration")
    # internal, set when this file re-executes itself as a worker
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--stage", choices=(*STAGES, "baseline"), help=argparse.SUPPRESS)
    p.add_argument("--thread-count", dest="threads", type=int, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _resolve_threads(spec: str):
    """Turn the ``--threads`` list into ints, expanding ``half``."""
    from spatialrisk.gdal_env import sampling_num_threads

    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        out.append(sampling_num_threads() if tok == "half" else int(tok))
    return out


def main(argv=None) -> int:
    """Entry point: orchestrate worker subprocesses and print the table."""
    args = _parse(argv)
    if args.worker:
        rec = _baseline() if args.stage == "baseline" else _run_stage(args)
        print(json.dumps(rec))
        return 0

    stages = [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s): {', '.join(unknown)}; choose from {STAGES}")
    threads = _resolve_threads(args.threads)

    passthrough = [args.raster]
    if args.mask:
        passthrough += ["--mask", args.mask]
    passthrough += [
        "--strategy",
        args.strategy,
        "--n-samples",
        str(args.n_samples),
        "--seed",
        str(args.seed),
        "--allocation",
        args.allocation,
        "--cachemax-mb",
        str(args.cachemax_mb),
    ]
    if args.spacing_m is not None:
        passthrough += ["--spacing-m", str(args.spacing_m)]

    base = _spawn(passthrough, ["--stage", "baseline", "--thread-count", "1"])
    print(f"raster: {args.raster}")
    if args.mask:
        print(f"mask:   {args.mask}")
    print(f"strategy={args.strategy} n_samples={args.n_samples} seed={args.seed}")
    print(f"import baseline: {base['peak_mib']:.0f} MiB\n")
    header = (
        f"{'stage':<8}{'threads':>8}{'run':>5}{'seconds':>10}"
        f"{'peak MiB':>10}{'net MiB':>9}  detail"
    )
    print(header)
    print("-" * len(header))
    for stage in stages:
        for n in threads:
            for run in range(1, args.repeat + 1):
                rec = _spawn(passthrough, ["--stage", stage, "--thread-count", str(n)])
                net = rec["peak_mib"] - base["peak_mib"]
                print(
                    f"{stage:<8}{n:>8}{run:>5}{rec['seconds']:>10.1f}"
                    f"{rec['peak_mib']:>10.0f}{net:>9.0f}  {rec['detail']}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
