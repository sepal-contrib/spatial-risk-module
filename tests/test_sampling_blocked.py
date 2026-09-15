"""Byte-identity regression tests for the blocked (row-stripe) sampling path.

Track E tasks E1/E2/E3/E5 rewrite ``generate_points`` so that peak memory
scales with one raster stripe plus the number of requested points, instead of
with the raster itself (the old path held the full band, a full validity bool,
the mask, and two int64 arrays from ``np.where(valid)`` — 21.6 bytes of working
RAM per raster pixel, ~45 GiB on a 2.22 Gpx country raster).

The optimisation is only legitimate if it is *invisible*: stored samples must
stay reproducible from their seed. So this module inlines the pre-E2 algorithm
verbatim as ``_reference_generate_points`` and asserts the new implementation
returns the same rows, cols, strata and coordinates across a matrix of
allocations, stripe heights, sample counts, dtypes and nodata conventions.

A failure here means the port drifted from the algorithm, not that a test is
too strict — every case below was validated against the prototype first.
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write(path, array, *, nodata=255, crs="EPSG:3857", tiled=False):
    """Write a single-band GeoTIFF with 1 x 1 m pixels at a known origin."""
    from rasterio.transform import from_origin

    kwargs = {}
    if tiled:
        kwargs = dict(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype.name,
        nodata=nodata,
        crs=crs,
        transform=from_origin(0, array.shape[0], 1, 1),
        **kwargs,
    ) as dst:
        dst.write(array, 1)


def _class_array(h, w, n_classes, *, seed=0, nodata_frac=0.1, nodata=255):
    """Pseudo-random class raster with a deterministic sprinkling of nodata."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, n_classes, size=(h, w)).astype("uint8")
    holes = rng.random((h, w)) < nodata_frac
    arr[holes] = nodata
    return arr


# --------------------------------------------------------------------------- #
# the pre-E2 algorithm, inlined verbatim as the reference
# --------------------------------------------------------------------------- #
def _ref_random(valid_indices, *, n_samples, seed=None):
    """Copy of the pre-E2 ``random.py::RandomSampling.select``."""
    rows, cols = valid_indices
    n_valid = len(rows)
    if n_samples is None or n_samples >= n_valid:
        return valid_indices
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_valid, size=n_samples, replace=False)
    return rows[idx], cols[idx]


def _ref_stratified(
    valid_indices,
    *,
    n_samples,
    seed=None,
    strata_values=None,
    allocation="equal",
    adapt=False,
    pixel_area_ha=None,
):
    """Copy of the pre-E2 ``stratified.py::StratifiedSampling.select``.

    Including the ``int(c)`` class-key collision (E6 territory) — the port has
    to reproduce the bug, not fix it, or output stops being identical.
    """
    from spatialrisk.sampling import allocation as alloc

    allocators = {
        "equal": lambda counts, n, adapt, pa: alloc.allocate_equal(counts, n),
        "proportional": lambda counts, n, adapt, pa: alloc.allocate_proportional(
            counts, n
        ),
        "deforisk": lambda counts, n, adapt, pa: alloc.allocate_deforisk(
            counts, n, adapt=adapt, pixel_area_ha=pa
        ),
    }
    if strata_values is None:
        raise ValueError("Stratified sampling requires strata_values.")
    rows, cols = valid_indices
    classes = np.unique(strata_values)
    class_counts = {int(c): int((strata_values == c).sum()) for c in classes}

    if n_samples is None:
        per_class = dict(class_counts)
    else:
        allocator = allocators.get(allocation or "equal")
        if allocator is None:
            raise ValueError(f"Unknown allocation method: {allocation}")
        per_class = allocator(class_counts, n_samples, adapt, pixel_area_ha)

    rng = np.random.default_rng(seed)
    out_rows, out_cols = [], []
    for c in sorted(class_counts, reverse=True):
        n_c = per_class.get(c, 0)
        if n_c <= 0:
            continue
        members = np.where(strata_values == c)[0]
        pick = rng.choice(members, size=min(n_c, len(members)), replace=False)
        out_rows.append(rows[pick])
        out_cols.append(cols[pick])
    if not out_rows:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(out_rows), np.concatenate(out_cols)


def _ref_systematic(
    valid_indices, *, n_samples=None, shape=None, spacing_m=None, res_m=None
):
    """Copy of the pre-E3 ``systematic.py::SystematicSampling.select``."""
    rows, cols = valid_indices
    n_valid = len(rows)
    if spacing_m is not None:
        if spacing_m <= 0:
            raise ValueError("spacing_m must be > 0.")
        if res_m is None:
            raise ValueError(
                "Distance-based systematic sampling requires res_m "
                "(pixel size in metres)."
            )
        if shape is None:
            raise ValueError("Systematic sampling requires the raster shape.")
        res_y, res_x = res_m
        step_row = max(1, int(round(spacing_m / res_y)))
        step_col = max(1, int(round(spacing_m / res_x)))
    else:
        if n_samples is None or n_samples >= n_valid:
            return valid_indices
        if shape is None:
            raise ValueError("Systematic sampling requires the raster shape.")
        step = max(1, int(round(np.sqrt(n_valid / n_samples))))
        step_row = step_col = step

    valid_mask = np.zeros(shape, dtype=bool)
    valid_mask[rows, cols] = True
    gr, gc = np.meshgrid(
        np.arange(0, shape[0], step_row),
        np.arange(0, shape[1], step_col),
        indexing="ij",
    )
    gr, gc = gr.ravel(), gc.ravel()
    keep = valid_mask[gr, gc]
    return gr[keep], gc[keep]


def _reference_generate_points(
    raster_path,
    mask_path=None,
    *,
    strategy,
    n_samples,
    allocation=None,
    seed=None,
    adapt=False,
    spacing_m=None,
):
    """Copy of the pre-E2 ``service.py::generate_points``."""
    from shapely.geometry import Point

    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        shape = raster.shape

    valid = ~np.isnan(raster)
    if nodata is not None:
        valid &= raster != nodata

    if mask_path is not None:
        with rasterio.open(mask_path) as msrc:
            mask = msrc.read(1)
            if mask.shape != shape:
                raise ValueError(
                    f"Mask shape {mask.shape} != raster shape {shape}; "
                    "raster and mask must be co-registered."
                )
            m_valid = mask != 0
            if msrc.nodata is not None:
                m_valid &= mask != msrc.nodata
        valid &= m_valid

    valid_indices = np.where(valid)
    strata_values = raster[valid_indices]
    pixel_area_ha = abs(transform.a * transform.e) / 10_000.0
    res_m = (abs(transform.e), abs(transform.a))

    if strategy == "random":
        rows, cols = _ref_random(valid_indices, n_samples=n_samples, seed=seed)
    elif strategy == "stratified":
        rows, cols = _ref_stratified(
            valid_indices,
            n_samples=n_samples,
            seed=seed,
            strata_values=strata_values,
            allocation=allocation,
            adapt=adapt,
            pixel_area_ha=pixel_area_ha,
        )
    elif strategy == "systematic":
        rows, cols = _ref_systematic(
            valid_indices,
            n_samples=n_samples,
            shape=shape,
            spacing_m=spacing_m,
            res_m=res_m,
        )
    else:
        raise ValueError(strategy)

    xs, ys = rasterio.transform.xy(transform, list(rows), list(cols), offset="center")
    return gpd.GeoDataFrame(
        {
            "strata": raster[rows, cols].astype(int),
            "row": np.asarray(rows, dtype=int),
            "col": np.asarray(cols, dtype=int),
        },
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )


def _assert_same(new, ref):
    """Assert two point GeoDataFrames are identical in value, order and dtype."""
    assert list(new.columns) == list(ref.columns)
    assert len(new) == len(ref)
    for col in ("strata", "row", "col"):
        assert new[col].dtype == ref[col].dtype, col
        np.testing.assert_array_equal(
            new[col].to_numpy(), ref[col].to_numpy(), err_msg=col
        )
    np.testing.assert_array_equal(new.geometry.x.to_numpy(), ref.geometry.x.to_numpy())
    np.testing.assert_array_equal(new.geometry.y.to_numpy(), ref.geometry.y.to_numpy())
    assert str(new.crs) == str(ref.crs)


# --------------------------------------------------------------------------- #
# E2 — the identity matrix
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def matrix_raster(tmp_path_factory):
    """A 600 x 400 five-class raster with nodata holes plus a partial mask."""
    d = tmp_path_factory.mktemp("matrix")
    arr = _class_array(600, 400, 5, seed=3)
    mask = np.ones((600, 400), dtype="uint8")
    mask[:37, :] = 0  # a masked band that straddles small stripes
    mask[:, 300:] = 0  # and a masked column block
    rpath, mpath = d / "strata.tif", d / "mask.tif"
    _write(rpath, arr)
    _write(mpath, mask, nodata=0)
    return rpath, mpath


@pytest.mark.parametrize("allocation", ["equal", "proportional", "deforisk"])
@pytest.mark.parametrize("stripe", [64, 97, 128, 256, 512])
@pytest.mark.parametrize("n", [150, 5000, 100_000])
def test_stratified_blocked_is_identical(matrix_raster, allocation, stripe, n):
    """Blocked stratified output matches the inlined pre-E2 algorithm exactly."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = matrix_raster
    new = generate_points(
        rpath,
        mpath,
        strategy="stratified",
        n_samples=n,
        allocation=allocation,
        seed=7,
        rows_per_stripe=stripe,
    )
    ref = _reference_generate_points(
        rpath,
        mpath,
        strategy="stratified",
        n_samples=n,
        allocation=allocation,
        seed=7,
    )
    _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [64, 97, 128, 256, 512])
@pytest.mark.parametrize("n", [150, 5000, 100_000])
def test_random_blocked_is_identical(matrix_raster, stripe, n):
    """Blocked simple-random output matches the inlined pre-E2 algorithm."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = matrix_raster
    new = generate_points(
        rpath,
        mpath,
        strategy="random",
        n_samples=n,
        seed=7,
        rows_per_stripe=stripe,
    )
    ref = _reference_generate_points(
        rpath, mpath, strategy="random", n_samples=n, seed=7
    )
    _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [64, 97, 128, 256, 512])
@pytest.mark.parametrize("n", [150, 5000, 100_000])
def test_systematic_count_mode_blocked_is_identical(matrix_raster, stripe, n):
    """Blocked systematic count mode keeps the same step and the same nodes."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = matrix_raster
    new = generate_points(
        rpath, mpath, strategy="systematic", n_samples=n, rows_per_stripe=stripe
    )
    ref = _reference_generate_points(rpath, mpath, strategy="systematic", n_samples=n)
    _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [64, 97, 128, 256, 512])
@pytest.mark.parametrize("spacing", [3.0, 17.0, 10_000.0])
def test_systematic_distance_mode_blocked_is_identical(matrix_raster, stripe, spacing):
    """Blocked systematic distance mode keeps the same grid nodes."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = matrix_raster
    new = generate_points(
        rpath,
        mpath,
        strategy="systematic",
        n_samples=None,
        spacing_m=spacing,
        rows_per_stripe=stripe,
    )
    ref = _reference_generate_points(
        rpath, mpath, strategy="systematic", n_samples=None, spacing_m=spacing
    )
    _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [1, 4096])
def test_degenerate_stripe_heights_are_identical(matrix_raster, stripe):
    """A one-row stripe and a stripe taller than the raster both behave."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = matrix_raster
    for strategy, kwargs in (
        ("random", {"n_samples": 200}),
        ("stratified", {"n_samples": 200, "allocation": "equal"}),
        ("systematic", {"n_samples": 200}),
    ):
        new = generate_points(
            rpath,
            mpath,
            strategy=strategy,
            seed=5,
            rows_per_stripe=stripe,
            **kwargs,
        )
        ref = _reference_generate_points(
            rpath, mpath, strategy=strategy, seed=5, **kwargs
        )
        _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [64, 256])
def test_seven_classes_are_identical(tmp_path, stripe):
    """Seven strata exercise the per-class inner loop of pass 2."""
    from spatialrisk.sampling.service import generate_points

    arr = _class_array(300, 220, 7, seed=11)
    rpath = tmp_path / "seven.tif"
    _write(rpath, arr)
    for allocation in ("equal", "proportional", "deforisk"):
        new = generate_points(
            rpath,
            strategy="stratified",
            n_samples=400,
            allocation=allocation,
            seed=2,
            rows_per_stripe=stripe,
        )
        ref = _reference_generate_points(
            rpath,
            strategy="stratified",
            n_samples=400,
            allocation=allocation,
            seed=2,
        )
        _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [32, 64, 256])
def test_class_absent_from_most_stripes_is_identical(tmp_path, stripe):
    """A class confined to a few rows must not shift any other class's ranks."""
    from spatialrisk.sampling.service import generate_points

    arr = np.zeros((400, 150), dtype="uint8")
    arr[:, 75:] = 1
    arr[201:204, 10:40] = 3  # rare class, only inside one stripe
    arr[399, 0:5] = 4  # rarer still, only in the last row
    rpath = tmp_path / "rare.tif"
    _write(rpath, arr)
    for allocation in ("equal", "proportional", "deforisk"):
        new = generate_points(
            rpath,
            strategy="stratified",
            n_samples=1000,
            allocation=allocation,
            seed=4,
            rows_per_stripe=stripe,
        )
        ref = _reference_generate_points(
            rpath,
            strategy="stratified",
            n_samples=1000,
            allocation=allocation,
            seed=4,
        )
        _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [64, 256])
def test_heavy_mask_is_identical(tmp_path, stripe):
    """A mask that removes most of the raster (and whole stripes) is identical."""
    from spatialrisk.sampling.service import generate_points

    arr = _class_array(500, 300, 4, seed=13)
    mask = np.zeros((500, 300), dtype="uint8")
    mask[120:140, 50:90] = 1  # one small island of unmasked pixels
    mask[300:305, :] = 1
    rpath, mpath = tmp_path / "s.tif", tmp_path / "m.tif"
    _write(rpath, arr)
    _write(mpath, mask, nodata=None)
    for strategy, kwargs in (
        ("random", {"n_samples": 100}),
        ("stratified", {"n_samples": 100, "allocation": "deforisk"}),
        ("systematic", {"n_samples": 100}),
    ):
        new = generate_points(
            rpath,
            mpath,
            strategy=strategy,
            seed=6,
            rows_per_stripe=stripe,
            **kwargs,
        )
        ref = _reference_generate_points(
            rpath, mpath, strategy=strategy, seed=6, **kwargs
        )
        _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [64, 256])
def test_float32_nan_nodata_is_identical(tmp_path, stripe):
    """float32 + ``nodata=nan``: validity comes from ``~isnan``, not equality."""
    from spatialrisk.sampling.service import generate_points

    arr = _class_array(300, 200, 4, seed=17, nodata_frac=0.0).astype("float32")
    rng = np.random.default_rng(19)
    arr[rng.random(arr.shape) < 0.15] = np.nan
    rpath = tmp_path / "f32.tif"
    _write(rpath, arr, nodata=float("nan"))
    for strategy, kwargs in (
        ("random", {"n_samples": 300}),
        ("stratified", {"n_samples": 300, "allocation": "proportional"}),
        ("systematic", {"n_samples": 300}),
    ):
        new = generate_points(
            rpath, strategy=strategy, seed=8, rows_per_stripe=stripe, **kwargs
        )
        ref = _reference_generate_points(rpath, strategy=strategy, seed=8, **kwargs)
        _assert_same(new, ref)


@pytest.mark.parametrize("stripe", [64, 256])
def test_nan_pixels_with_sentinel_nodata_is_identical(tmp_path, stripe):
    """NaN pixels *and* a sentinel nodata must both be excluded, in both passes."""
    from spatialrisk.sampling.service import generate_points

    arr = _class_array(300, 200, 3, seed=23, nodata_frac=0.0).astype("float32")
    rng = np.random.default_rng(29)
    arr[rng.random(arr.shape) < 0.1] = np.nan
    arr[rng.random(arr.shape) < 0.1] = -9999.0
    rpath = tmp_path / "mixed.tif"
    _write(rpath, arr, nodata=-9999.0)
    new = generate_points(
        rpath,
        strategy="stratified",
        n_samples=250,
        allocation="equal",
        seed=9,
        rows_per_stripe=stripe,
    )
    ref = _reference_generate_points(
        rpath, strategy="stratified", n_samples=250, allocation="equal", seed=9
    )
    _assert_same(new, ref)
    assert not np.isnan(new["strata"].to_numpy().astype(float)).any()


def test_float_class_collision_bug_is_preserved(tmp_path):
    """The ``int(c)`` class-key collision at stratified.py:35 must survive E2.

    Strata ``[1.0, 1.9]`` collapse to the single key ``1``, whose count comes
    from the *last* class seen (1.9) while the draw happens over the members of
    ``sv == 1`` (the 1.0 pixels). That is wrong, and deliberately out of scope
    (task E6 is gated on a product decision) — the blocked port must reproduce
    it or output stops being identical for existing float projects.
    """
    from spatialrisk.sampling.service import generate_points

    arr = np.full((200, 100), 1.0, dtype="float32")
    arr[:, 60:] = 1.9
    arr[:, 90:] = 2.4
    rpath = tmp_path / "collide.tif"
    _write(rpath, arr, nodata=-1.0)
    for stripe in (32, 256):
        new = generate_points(
            rpath,
            strategy="stratified",
            n_samples=50,
            allocation="equal",
            seed=3,
            rows_per_stripe=stripe,
        )
        ref = _reference_generate_points(
            rpath, strategy="stratified", n_samples=50, allocation="equal", seed=3
        )
        _assert_same(new, ref)


def test_tiled_raster_is_identical(tmp_path):
    """A 256 x 256-tiled raster still enumerates pixels in global row-major order.

    Tiled block order is *not* row-major, which is exactly why the scan uses
    full-width stripes rather than ``src.block_windows()``.
    """
    from spatialrisk.sampling.service import generate_points

    arr = _class_array(700, 700, 3, seed=31)
    rpath = tmp_path / "tiled.tif"
    _write(rpath, arr, tiled=True)
    new = generate_points(
        rpath, strategy="stratified", n_samples=900, allocation="deforisk", seed=1
    )
    ref = _reference_generate_points(
        rpath, strategy="stratified", n_samples=900, allocation="deforisk", seed=1
    )
    _assert_same(new, ref)


def test_n_far_above_availability_is_identical(matrix_raster):
    """Asking for more points than exist caps per class with no redistribution."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = matrix_raster
    for strategy, kwargs in (
        ("random", {"n_samples": 10_000_000}),
        ("stratified", {"n_samples": 10_000_000, "allocation": "equal"}),
        ("systematic", {"n_samples": 10_000_000}),
    ):
        new = generate_points(
            rpath, mpath, strategy=strategy, seed=1, rows_per_stripe=97, **kwargs
        )
        ref = _reference_generate_points(
            rpath, mpath, strategy=strategy, seed=1, **kwargs
        )
        _assert_same(new, ref)


def test_all_nodata_raster_returns_an_empty_frame(tmp_path):
    """A raster with no valid pixel at all must not crash the empty-array path."""
    from spatialrisk.sampling.service import generate_points

    rpath = tmp_path / "void.tif"
    _write(rpath, np.full((80, 60), 255, dtype="uint8"))
    for strategy, kwargs in (
        ("random", {"n_samples": 10}),
        ("stratified", {"n_samples": 10, "allocation": "equal"}),
        ("systematic", {"n_samples": 10}),
    ):
        new = generate_points(rpath, strategy=strategy, seed=1, **kwargs)
        ref = _reference_generate_points(rpath, strategy=strategy, seed=1, **kwargs)
        assert len(new) == 0
        _assert_same(new, ref)


def test_zero_samples_is_identical(matrix_raster):
    """n_samples=0 is bounded, so it is allowed — and must stay empty."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = matrix_raster
    for strategy in ("random", "stratified"):
        new = generate_points(
            rpath,
            mpath,
            strategy=strategy,
            n_samples=0,
            allocation="equal",
            seed=1,
            rows_per_stripe=97,
        )
        ref = _reference_generate_points(
            rpath, mpath, strategy=strategy, n_samples=0, allocation="equal", seed=1
        )
        assert len(new) == 0
        _assert_same(new, ref)


def test_default_stripe_height_needs_no_argument(matrix_raster):
    """The stripe height is an optimisation knob, not part of the contract."""
    from spatialrisk.sampling.service import generate_points

    rpath, mpath = matrix_raster
    new = generate_points(
        rpath,
        mpath,
        strategy="stratified",
        n_samples=800,
        allocation="equal",
        seed=12,
    )
    ref = _reference_generate_points(
        rpath,
        mpath,
        strategy="stratified",
        n_samples=800,
        allocation="equal",
        seed=12,
    )
    _assert_same(new, ref)


def test_mask_shape_mismatch_still_raises(tmp_path):
    """The co-registration check must survive the rewrite, message included."""
    from spatialrisk.sampling.service import generate_points

    _write(tmp_path / "s.tif", _class_array(40, 40, 2, seed=1))
    _write(tmp_path / "m.tif", np.ones((40, 39), dtype="uint8"))
    with pytest.raises(ValueError, match="must be co-registered"):
        generate_points(
            tmp_path / "s.tif",
            tmp_path / "m.tif",
            strategy="random",
            n_samples=10,
        )


# --------------------------------------------------------------------------- #
# E1 — the service-boundary guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["random", "stratified", "systematic"])
def test_none_n_samples_is_rejected_before_any_read(strategy):
    """``n_samples=None`` outside systematic-spacing means "every valid pixel".

    The path does not exist, so a raster read would raise something else — this
    asserts the guard fires *first*, which is the point: ``Sample`` can be
    constructed directly and bypass the dialog's own check.
    """
    from spatialrisk.sampling.service import generate_points

    with pytest.raises(ValueError, match="n_samples"):
        generate_points(
            Path("/nonexistent/definitely-not-a-raster.tif"),
            strategy=strategy,
            n_samples=None,
        )


def test_systematic_spacing_still_allows_none(tmp_path):
    """Systematic + ``spacing_m`` is the one legitimate ``n_samples=None`` mode."""
    from spatialrisk.sampling.service import generate_points

    _write(tmp_path / "s.tif", np.zeros((40, 40), dtype="uint8"))
    gdf = generate_points(
        tmp_path / "s.tif", strategy="systematic", n_samples=None, spacing_m=10.0
    )
    assert sorted(gdf["row"].unique().tolist()) == [0, 10, 20, 30]
    assert len(gdf) == 16


# --------------------------------------------------------------------------- #
# E2/E3 — structural guarantees
# --------------------------------------------------------------------------- #
def test_no_unwindowed_band_read_remains():
    """A read without ``window=`` would reinstate the whole-raster allocation."""
    import spatialrisk.sampling.blocked as blocked
    import spatialrisk.sampling.service as service

    for mod in (service, blocked):
        src = Path(mod.__file__).read_text()
        for call in re.findall(r"\.read\([^)]*\)", src):
            assert "window=" in call, f"{mod.__name__}: unwindowed read {call!r}"


def test_systematic_select_allocates_nothing_shape_sized(monkeypatch):
    """The full-shape bool and the two int64 meshgrids are gone from ``select``.

    ``np.zeros(shape, bool)`` cost 2.1 GiB on a 2.22 Gpx raster and the
    ``meshgrid`` pair cost 33.1 GiB at one-pixel spacing — the single largest
    allocation anywhere in the pipeline.
    """
    from spatialrisk.sampling import systematic as sysmod

    def _boom(*_a, **_k):
        raise AssertionError("meshgrid must not be used any more")

    monkeypatch.setattr(sysmod.np, "meshgrid", _boom)
    H = W = 64
    rr, cc = np.mgrid[0:H, 0:W]
    r = sysmod.SystematicSampling().select(
        (rr.ravel(), cc.ravel()), shape=(H, W), spacing_m=8.0, res_m=(1.0, 1.0)
    )
    assert sorted(set(r[0].tolist())) == list(range(0, H, 8))


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "float32"])
def test_value_counts_matches_unique_on_every_dtype(dtype):
    """The bincount fast path in pass 1 must agree with np.unique exactly.

    Pass 1 histograms small unsigned dtypes instead of sorting them (~4x
    faster per stripe); a disagreement there would change every class count
    and therefore every allocation.
    """
    from spatialrisk.sampling.blocked import _value_counts

    rng = np.random.default_rng(5)
    low = 0 if np.dtype(dtype).kind == "u" else -20  # negatives break bincount
    vals = rng.integers(low, 40, size=5000).astype(dtype)
    values, counts = _value_counts(vals)
    ref_values, ref_counts = np.unique(vals, return_counts=True)
    np.testing.assert_array_equal(np.asarray(values, dtype=float), ref_values)
    np.testing.assert_array_equal(counts, ref_counts)


def test_blocked_stripes_are_full_width_and_tile_aligned(tmp_path):
    """Stripes span the raster width and align to the tile height (256/512).

    Full width is what makes the walk globally row-major; tile alignment stops
    a stripe boundary from splitting a tile row and re-decoding it twice.
    """
    from spatialrisk.sampling.blocked import RasterScan

    arr = _class_array(1200, 900, 2, seed=1)
    rpath = tmp_path / "aligned.tif"
    _write(rpath, arr, tiled=True)
    with RasterScan(rpath) as scan:
        assert all(int(w.width) == 900 for w in scan.windows)
        assert scan.rows_per_stripe % 256 == 0
        assert sum(int(w.height) for w in scan.windows) == 1200
        offsets = [int(w.row_off) for w in scan.windows]
        assert offsets == sorted(offsets)


# --------------------------------------------------------------------------- #
# E2/E5 — memory and wall clock
# --------------------------------------------------------------------------- #
# Both branches import *everything* both paths need before the baseline is
# taken, so the reported figure is the algorithm's working set and not a
# difference in how much of the package each branch happened to import.
_PROBE = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {tests!r})
    import numpy as np
    import shapely.geometry
    import test_sampling_blocked as tb
    from spatialrisk.sampling.service import generate_points

    def _hwm_kib():
        # NOT resource.getrusage(RUSAGE_SELF).ru_maxrss: that high-water mark is
        # INHERITED verbatim across fork+exec, so a probe spawned from a pytest
        # process that has already run the suite reports the PARENT's peak and
        # the delta collapses to 0 (measured: parent 1980488 KiB -> child
        # ru_maxrss 1980488, while VmHWM stayed 13232). /proc/self/status VmHWM
        # is reset with the new mm on exec, so it is per-process and order-safe.
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
        raise RuntimeError("VmHWM not available")

    mode, rpath, mpath = sys.argv[1], sys.argv[2], sys.argv[3]
    base = _hwm_kib()
    fn = tb._reference_generate_points if mode == "ref" else generate_points
    t0 = time.perf_counter()
    gdf = fn(rpath, mpath, strategy="stratified", n_samples=20000,
             allocation="deforisk", seed=1)
    dt = time.perf_counter() - t0
    peak = _hwm_kib()
    print(base, peak, dt, len(gdf))
    """
)


def _probe(mode, rpath, mpath):
    """Run one generate_points in a fresh process; return (peak_KiB, seconds)."""
    code = _PROBE.format(tests=str(Path(__file__).parent))
    out = subprocess.run(
        [sys.executable, "-c", code, mode, str(rpath), str(mpath)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    base, peak, dt = int(out[0]), int(out[1]), float(out[2])
    return peak - base, dt


@pytest.fixture(scope="module")
def big_raster(tmp_path_factory):
    """A 3000 x 3000 (9 Mpx) fixture — big enough for the old cost to show."""
    d = tmp_path_factory.mktemp("big")
    rpath, mpath = d / "s.tif", d / "m.tif"
    _write(rpath, _class_array(3000, 3000, 2, seed=41, nodata_frac=0.05))
    _write(mpath, np.ones((3000, 3000), dtype="uint8"), nodata=0)
    return rpath, mpath


def test_peak_memory_is_bounded_by_a_stripe_not_the_raster(big_raster):
    """Working set must stop scaling with the raster (21.6 B/px before E2)."""
    rpath, mpath = big_raster
    ref_kib, _ = _probe("ref", rpath, mpath)
    new_kib, _ = _probe("new", rpath, mpath)
    assert new_kib < ref_kib / 3, f"ref={ref_kib} KiB new={new_kib} KiB"


def test_wall_clock_does_not_regress(big_raster):
    """The second stripe pass must cost no more than the allocations it removes."""
    rpath, mpath = big_raster
    _, ref_s = _probe("ref", rpath, mpath)
    _, new_s = _probe("new", rpath, mpath)
    assert new_s < ref_s * 1.5, f"ref={ref_s:.3f}s new={new_s:.3f}s"
