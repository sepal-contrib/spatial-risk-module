"""Golden rasters from the pre-engine serial loops, and apply() must match them.

``test_generate_goldens`` writes the .npy files once (from whatever apply()
implementation is checked out, so run it ONLY on the commit before the engine
lands, then commit the files). Afterwards it is skipped and the three
``test_*_matches_golden`` tests are the acceptance gate for the refactor.
"""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from _inference_fixture import (  # noqa: E402
    GOLDEN_DIR,
    build_dataset,
    build_glm,
    build_icar,
    build_rf,
    read_raster,
)

BUILDERS = {"glm": build_glm, "rf": build_rf, "icar": build_icar}


def _predict(kind, tmp_path):
    ds = build_dataset(tmp_path)
    model = BUILDERS[kind](tmp_path, ds)
    out = tmp_path / "out" / f"{kind}.tif"
    model.apply(out, ds, ds.mask_path, 0)
    return read_raster(out)


@pytest.mark.skipif(
    (GOLDEN_DIR / "meta.json").exists(), reason="goldens already generated"
)
def test_generate_goldens(tmp_path):
    """Write the golden .npy/.meta.json files from the current apply() loops."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    for kind in BUILDERS:
        arr, m = _predict(kind, tmp_path / kind)
        np.save(GOLDEN_DIR / f"{kind}.npy", arr)
        meta[kind] = m
    (GOLDEN_DIR / "meta.json").write_text(json.dumps(meta, indent=1))


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_apply_matches_golden(kind, tmp_path):
    """apply() must reproduce the committed golden raster pixel-for-pixel."""
    if not (GOLDEN_DIR / "meta.json").exists():
        pytest.skip("run test_generate_goldens first")
    arr, m = _predict(kind, tmp_path)
    expected = np.load(GOLDEN_DIR / f"{kind}.npy")
    meta = json.loads((GOLDEN_DIR / "meta.json").read_text())[kind]
    np.testing.assert_array_equal(arr, expected)
    assert m == meta
    # sanity on the fixture itself: something was predicted, something masked.
    # Note: target nodata (rows 0:5) does NOT propagate to the output — the
    # target raster is only read for its profile/transform, never as a
    # feature, so apply()'s valid_mask is untouched by it.
    assert (arr[10:20] == 0).all()  # alt nodata band
    assert (arr[100:130] == 0).all()  # mask-suppressed rows
    assert (arr[400:420, 100:200] == 0).all()  # mask nodata
    assert (arr[600:650] == 0).all()  # unused-feature nodata still invalidates
    assert (arr[200:250] > 0).all()
