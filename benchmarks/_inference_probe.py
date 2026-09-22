"""Peak RSS of one GLM prediction in a fresh process.

Usage: ``python _inference_probe.py <dir> <workers>``. Prints one JSON line
``{"peak_kib": ..., "baseline_kib": ...}``. Used by the memory probe in
``tests/test_inference_plan.py`` to pin ``plan_inference``'s working-set
model against reality.
"""
import argparse
import json
import sys
from pathlib import Path

# Measure the checkout this file lives in, not whichever one the editable
# install points at: a probe launched from a worktree would otherwise import
# the main checkout's spatialrisk (see benchmarks/inference_bench.py::_spawn_env).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO))


def _vmhwm_kib():
    # NOT resource.getrusage(RUSAGE_SELF).ru_maxrss: tests/test_sampling_blocked.py's
    # _PROBE found that stat is inherited verbatim across fork+exec, so a probe
    # spawned from a process that already did other work reports the PARENT's
    # peak and the before/after delta collapses to 0. /proc/self/status VmHWM
    # resets with the new mm on exec, so it is per-process and order-safe.
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("VmHWM not available")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("workers", type=int)
    ap.add_argument("--size", type=int, nargs=2, default=(3000, 3000))
    a = ap.parse_args()
    import inference_bench as ib

    out = Path(a.out_dir)
    rasters = ib._write_synthetic(out, tuple(a.size), 4, 42)
    args = argparse.Namespace(rasters=rasters, seed=42, model="glm")
    model, ds = ib._fit(args, out)
    baseline = _vmhwm_kib()
    model.apply(out / f"pred_{a.workers}.tif", ds, workers=a.workers)
    print(json.dumps({"peak_kib": _vmhwm_kib(), "baseline_kib": baseline}))


if __name__ == "__main__":
    main()
