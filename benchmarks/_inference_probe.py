"""Peak RSS of one GLM prediction in a fresh process.

Usage: ``python _inference_probe.py <dir> <workers> [--size H W] [--features N]``.
Prints one JSON line ``{"peak_kib": ..., "baseline_kib": ..., "n_design_cols":
...}``, where ``n_design_cols`` is the working width the run actually charged,
read off the engine's own plan line. Used by the memory probe in
``tests/test_inference_plan.py`` to pin ``plan_inference``'s working-set
model against reality: the test plans with that printed width, so it always
checks the charge the model really makes.
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Measure the checkout this file lives in, not whichever one the editable
# install points at: a probe launched from a worktree would otherwise import
# the main checkout's spatialrisk (see benchmarks/inference_bench.py::_spawn_env).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO))

#: The charged width in the engine's plan line, e.g. "pred_1.tif: 1 worker(s),
#: 256 rows/stripe (99 MiB each, 1 working cols), budget ...".
_WORKING_COLS_RE = re.compile(r"rows/stripe \([\d.]+ MiB each, (\d+) working cols\)")


class _ChargeCapture(logging.Handler):
    """Keep the working width the engine's plan line reports."""

    def __init__(self):
        """Start with nothing captured."""
        super().__init__(level=logging.INFO)
        self.n_design_cols = None

    def emit(self, record):
        """Remember the first plan line's working width."""
        if self.n_design_cols is None:
            match = _WORKING_COLS_RE.search(record.getMessage())
            if match:
                self.n_design_cols = int(match[1])


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
    """Fit a GLM on a synthetic float32 stack, predict once, print the peaks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("workers", type=int)
    ap.add_argument("--size", type=int, nargs=2, default=(3000, 3000))
    ap.add_argument("--features", type=int, default=4)
    a = ap.parse_args()
    import inference_bench as ib

    out = Path(a.out_dir)
    rasters = ib._write_synthetic(out, tuple(a.size), a.features, 42)
    args = argparse.Namespace(rasters=rasters, seed=42, model="glm")
    model, ds = ib._fit(args, out)
    engine_log = logging.getLogger("spatial_risk")
    engine_log.setLevel(logging.INFO)
    capture = _ChargeCapture()
    engine_log.addHandler(capture)
    baseline = _vmhwm_kib()
    model.apply(out / f"pred_{a.workers}.tif", ds, workers=a.workers)
    peak = _vmhwm_kib()
    if capture.n_design_cols is None:
        raise RuntimeError("the engine logged no plan line with a working width")
    print(
        json.dumps(
            {
                "peak_kib": peak,
                "baseline_kib": baseline,
                "n_design_cols": capture.n_design_cols,
            }
        )
    )


if __name__ == "__main__":
    main()
