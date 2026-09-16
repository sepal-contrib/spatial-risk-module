"""Inspection and scale checks for user-imported prediction rasters."""

from pathlib import Path

import numpy as np
import odc.geo.xr  # noqa: F401  # registers the .odc accessor
import pytest
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_origin

from spatialrisk.predictions.import_raster import (
    RISK_MAX,
    ImportRasterError,
    RasterInfo,
    _warp_cast_dtype,
    adapt_raster,
    check_scale,
    inspect_raster,
    raster_range,
)


def _write(path, data, *, count=1, crs="EPSG:4326", nodata=None, res=0.01):
    """Tiny GeoTIFF; ``data`` is 2-D and repeated for every band."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=from_origin(-55.0, -24.0, res, res),
        nodata=nodata,
    ) as dst:
        for b in range(1, count + 1):
            dst.write(data, b)
    return path


# --- inspect_raster -----------------------------------------------------------


def test_inspect_reports_structure(tmp_path):
    """inspect_raster reports band count, dtype, CRS, nodata and geometry."""
    src = _write(tmp_path / "a.tif", np.ones((4, 6), dtype=np.uint16), nodata=0)
    info = inspect_raster(src)
    assert isinstance(info, RasterInfo)
    assert info.band_count == 1
    assert info.dtype == "uint16"
    assert info.crs == "EPSG:4326"
    assert info.nodata == 0
    assert info.width == 6 and info.height == 4
    assert info.resolution == pytest.approx((0.01, 0.01))


def test_inspect_rejects_two_bands(tmp_path):
    """A multi-band raster is not a valid single-band prediction."""
    src = _write(tmp_path / "two.tif", np.ones((4, 4), dtype=np.uint8), count=2)
    with pytest.raises(ImportRasterError, match="2 band"):
        inspect_raster(src)


def test_inspect_rejects_missing_crs(tmp_path):
    """A raster with no CRS cannot be placed on the project grid."""
    src = _write(tmp_path / "nocrs.tif", np.ones((4, 4), dtype=np.uint8), crs=None)
    with pytest.raises(ImportRasterError, match="coordinate reference system"):
        inspect_raster(src)


def test_inspect_rejects_missing_file(tmp_path):
    """An unreadable path raises ImportRasterError instead of propagating."""
    with pytest.raises(ImportRasterError, match="Cannot open"):
        inspect_raster(tmp_path / "nope.tif")


def test_inspect_reads_no_pixel_data(tmp_path, monkeypatch):
    """Safe for a Solara handler: metadata only."""
    src = _write(tmp_path / "a.tif", np.ones((4, 4), dtype=np.uint16))
    real_open = rasterio.open

    class _Guard:
        def __init__(self, ds):
            self._ds = ds

        def read(self, *a, **k):
            raise AssertionError("inspect_raster must not read pixels")

        def __getattr__(self, name):
            return getattr(self._ds, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._ds.__exit__(*exc)

    monkeypatch.setattr(
        "spatialrisk.predictions.import_raster.rasterio.open",
        lambda *a, **k: _Guard(real_open(*a, **k)),
    )
    inspect_raster(src)


# --- raster_range -------------------------------------------------------------


def test_range_excludes_nodata_and_nan(tmp_path):
    """raster_range ignores nodata and NaN pixels when computing min/max."""
    data = np.array([[0.2, 0.8], [np.nan, -9999.0]], dtype=np.float32)
    src = _write(tmp_path / "f.tif", data, nodata=-9999.0)
    vmin, vmax = raster_range(src)
    assert vmin == pytest.approx(0.2, abs=1e-6)
    assert vmax == pytest.approx(0.8, abs=1e-6)


def test_range_rejects_unreadable_file(tmp_path):
    """A corrupt file raises ImportRasterError instead of a silent bogus range.

    Without scoping gdal.ExceptionMgr, gdal.Open on a corrupt file just
    returns None (or logs and returns None) depending on process-wide GDAL
    exception state, rather than reliably raising; this pins the
    deterministic, always-raises behaviour.
    """
    src = tmp_path / "garbage.tif"
    src.write_bytes(b"not a real tiff file" * 20)
    with pytest.raises(ImportRasterError, match="Cannot open"):
        raster_range(src)


# --- check_scale --------------------------------------------------------------


def test_probability_scale_accepts_unit_interval():
    """0..1 values are valid for the probability scale."""
    check_scale(0.0, 1.0, "probability")


def test_probability_scale_rejects_values_above_one():
    """A range above 1 contradicts a declared probability scale."""
    with pytest.raises(ImportRasterError, match=r"0\.0 to 100\.0"):
        check_scale(0.0, 100.0, "probability")


def test_risk_scale_rejects_a_range_at_or_below_one():
    """A 0..1 file declared as risk is a probability raster in disguise."""
    with pytest.raises(ImportRasterError, match="probabilit"):
        check_scale(0.0, 1.0, "risk")


def test_risk_scale_lower_bound_names_the_observed_range():
    """The sub-1 risk error states the range it saw, not just the rule."""
    with pytest.raises(ImportRasterError, match=r"0\.25 to 0\.5"):
        check_scale(0.25, 0.5, "risk")


def test_risk_scale_accepts_the_app_value_scale():
    """A genuine 1..65535 risk raster passes."""
    check_scale(1.0, 65535.0, "risk")


def test_risk_scale_rejects_values_above_65535():
    """A range above the UInt16 max contradicts a declared risk scale."""
    with pytest.raises(ImportRasterError, match="65535"):
        check_scale(1.0, 70000.0, "risk")


def test_any_scale_rejects_negative_values():
    """Negative values are invalid under either declared scale."""
    with pytest.raises(ImportRasterError, match="negative"):
        check_scale(-1.0, 0.5, "probability")
    with pytest.raises(ImportRasterError, match="negative"):
        check_scale(-1.0, 10.0, "risk")


def test_negative_value_message_states_observed_range():
    """The negative-values error names both the min and the max, not just min."""
    with pytest.raises(ImportRasterError, match=r"-1(\.0)? to 0\.5"):
        check_scale(-1.0, 0.5, "probability")


def test_constant_raster_is_rejected():
    """A raster with no value spread carries no risk information."""
    with pytest.raises(ImportRasterError, match="constant"):
        check_scale(0.4, 0.4, "probability")


def test_unknown_scale_is_rejected():
    """An unrecognized scale token is rejected outright."""
    with pytest.raises(ImportRasterError, match="value scale"):
        check_scale(0.0, 1.0, "percent")


# --- adapt_raster -------------------------------------------------------------

# A 20x20 UTM 21S grid at 1 km covering roughly the source footprint of _write.
_UTM = "EPSG:32721"


def _target_geobox():
    """The project grid the imports are warped onto: 20x20 px at 1 km, UTM 21S."""
    return GeoBox.from_bbox(
        (700000.0, 7330000.0, 720000.0, 7350000.0), crs=_UTM, resolution=1000.0
    )


def _far_rescale(p):
    """The uint16 value ``far.misc.rescale`` maps probability ``p`` to."""
    import forestatrisk as far

    return int(far.misc.rescale(np.array([p], dtype=np.float64))[0])


def test_adapt_probability_lands_on_geobox_as_uint16(tmp_path):
    """A float probability raster lands on the geobox as tiled uint16, nodata 0."""
    # Left half 0.2, right half 0.8, one nodata pixel; EPSG:4326 at ~0.002 deg.
    data = np.full((60, 60), 0.2, dtype=np.float32)
    data[:, 30:] = 0.8
    data[0, 0] = -9999.0
    src = _write(tmp_path / "prob.tif", data, nodata=-9999.0, res=0.002)
    dst = tmp_path / "out" / "prob_adapted.tif"
    dst.parent.mkdir()

    out = adapt_raster(src, dst, _target_geobox(), "probability")

    assert out == dst
    with rasterio.open(dst) as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "uint16"
        assert ds.nodata == 0
        assert ds.crs.to_epsg() == 32721
        assert (ds.width, ds.height) == (20, 20)
        assert ds.transform.a == 1000.0 and ds.transform.e == -1000.0
        assert ds.is_tiled
        values = set(np.unique(ds.read(1)).tolist())
    # Pinned literals, so this is a real oracle: comparing far.misc.rescale
    # against itself would pass even if the formula changed under us.
    assert (_far_rescale(0.2), _far_rescale(0.8)) == (13107, 52428)
    # Only rescaled source values (and nodata 0) may appear: nearest resampling
    # invents nothing.
    assert values <= {0, 13107, 52428}
    assert values & {13107, 52428}
    # Positive, not merely a superset: the source covers only part of the
    # geobox, so the rest of the project grid MUST come out as nodata. Without
    # this line the assertion above passes even when 0 never appears at all.
    assert 0 in values


def test_adapt_risk_preserves_integer_values(tmp_path):
    """A uint16 risk raster keeps its exact values through the warp."""
    data = np.full((60, 60), 30000, dtype=np.uint16)
    data[:, 30:] = 65535
    src = _write(tmp_path / "risk.tif", data, nodata=0, res=0.002)
    dst = tmp_path / "risk_adapted.tif"

    adapt_raster(src, dst, _target_geobox(), "risk")

    with rasterio.open(dst) as ds:
        values = set(np.unique(ds.read(1)).tolist())
    assert values <= {0, 30000, 65535}
    assert values & {30000, 65535}
    assert 0 in values  # declared nodata outside the footprint, not a value


def test_adapt_risk_accepts_float_integer_values(tmp_path):
    """Whole numbers stored as floats are accepted on the risk scale."""
    data = np.full((60, 60), 1234.0, dtype=np.float32)
    src = _write(tmp_path / "riskf.tif", data, nodata=None, res=0.002)
    dst = tmp_path / "riskf_adapted.tif"

    adapt_raster(src, dst, _target_geobox(), "risk")

    with rasterio.open(dst) as ds:
        values = set(np.unique(ds.read(1)).tolist())
    assert values <= {0, 1234}
    assert 1234 in values
    assert 0 in values  # NaN fill outside the footprint, not a value


def test_adapt_removes_temp_file_on_success(tmp_path):
    """The intermediate warp file does not survive a successful adaptation."""
    data = np.full((60, 60), 0.5, dtype=np.float32)
    src = _write(tmp_path / "p.tif", data, res=0.002)
    dst = tmp_path / "p_adapted.tif"
    adapt_raster(src, dst, _target_geobox(), "probability")
    assert not list(tmp_path.glob(".*warp*")), "temp warp file must be deleted"


def test_adapt_removes_temp_file_on_failure(tmp_path, monkeypatch):
    """A failed second pass leaves neither the temp file nor a partial output."""
    data = np.full((60, 60), 0.5, dtype=np.float32)
    src = _write(tmp_path / "p.tif", data, res=0.002)
    dst = tmp_path / "p_adapted.tif"

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("spatialrisk.predictions.import_raster._write_uint16", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        adapt_raster(src, dst, _target_geobox(), "probability")
    assert not list(tmp_path.glob(".*warp*"))
    assert not dst.exists()


def test_adapt_rejects_unknown_scale(tmp_path):
    """An unrecognized scale token is refused before any file is touched."""
    src = _write(tmp_path / "p.tif", np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ImportRasterError, match="value scale"):
        adapt_raster(src, tmp_path / "o.tif", _target_geobox(), "percent")


def test_adapt_keeps_a_pre_existing_destination_when_the_warp_fails(
    tmp_path, monkeypatch
):
    """A pass-1 failure must not delete a destination this call did not create."""
    data = np.full((60, 60), 0.5, dtype=np.float32)
    src = _write(tmp_path / "p.tif", data, res=0.002)
    dst = tmp_path / "p_adapted.tif"
    dst.write_bytes(b"the previous adaptation")

    def boom(*a, **k):
        raise RuntimeError("warp exploded")

    monkeypatch.setattr("spatialrisk.geo_utils.xr_reproject", boom)
    with pytest.raises(RuntimeError, match="warp exploded"):
        adapt_raster(src, dst, _target_geobox(), "probability")
    assert dst.read_bytes() == b"the previous adaptation"
    assert not list(tmp_path.glob(".*warp*"))


def test_adapt_risk_maps_a_value_rounding_to_zero_to_nodata(tmp_path):
    """On the risk scale 0 IS nodata, so a pixel rounding to 0 becomes a hole.

    The contract (and the dialog's spec block) says an undeclared 0 in a risk
    source is no data. Flooring such a pixel at 1 would hide the hole as valid
    zero-risk data — the failure mode this feature exists to remove. A file
    whose whole range sits at or below 1 never gets this far: ``check_scale``
    rejects it as a probability raster in disguise.
    """
    data = np.full((60, 60), 0.4, dtype=np.float32)
    data[:, 30:] = 5000.0
    src = _write(tmp_path / "small.tif", data, nodata=None, res=0.002)
    dst = tmp_path / "small_adapted.tif"

    adapt_raster(src, dst, _target_geobox(), "risk")

    with rasterio.open(dst) as ds:
        values = set(np.unique(ds.read(1)).tolist())
    assert values <= {0, 5000}
    assert 0 in values and 5000 in values
    assert 1 not in values  # a 1-floor would have turned the hole into data


def test_adapt_risk_treats_an_undeclared_integer_fill_as_nodata(tmp_path):
    """An integer source with no nodata: the warp's 0 fill is a hole, not zero risk.

    ``xr_reproject`` fills outside the source footprint with 0 for an integer
    source that declares no nodata (odc's ``resolve_fill_value``). Nothing in
    the warped file distinguishes that fill from real data, so it must not
    survive as a valid pixel: here the source covers roughly half the geobox
    and the rest of the project grid must come out as nodata.
    """
    data = np.full((60, 60), 30000, dtype=np.uint16)
    data[:, 30:] = 65535
    src = _write(tmp_path / "nofill.tif", data, nodata=None, res=0.002)
    dst = tmp_path / "nofill_adapted.tif"

    adapt_raster(src, dst, _target_geobox(), "risk")

    with rasterio.open(dst) as ds:
        values = set(np.unique(ds.read(1)).tolist())
    assert 0 in values, "the fill outside the source footprint must be nodata"
    assert values <= {0, 30000, 65535}


def test_adapt_probability_keeps_an_integer_zero_and_still_drops_the_fill(tmp_path):
    """An integer probability source: 0 is data, the warp fill is not.

    The two are the same number on disk, and an integer warp would fill outside
    the footprint with that same 0. ``_warp_cast_dtype`` therefore sends a
    probability source with no declared nodata through the warp as float32, so
    the fill arrives as NaN: the source's own zeros stay data and rescale to 1,
    while the pixels the source never covered come out as nodata. Getting this
    wrong in either direction is silent -- either a hole reads as lowest risk,
    or the lowest-risk pixels vanish from the map.
    """
    data = np.zeros((60, 60), dtype=np.uint8)
    data[:, 30:] = 1
    src = _write(tmp_path / "p8.tif", data, nodata=None, res=0.002)
    dst = tmp_path / "p8_adapted.tif"

    adapt_raster(src, dst, _target_geobox(), "probability")

    with rasterio.open(dst) as ds:
        values = set(np.unique(ds.read(1)).tolist())
    assert values == {0, 1, RISK_MAX}, (
        "expected nodata (0) outside the footprint, the source's own zeros as "
        f"1 and its ones as {RISK_MAX}; got {sorted(values)}"
    )


def test_adapt_probability_keeps_an_integer_zero_when_nodata_is_declared(tmp_path):
    """The cast must not disturb a source that declares its own nodata.

    Here 255 is the declared fill, so the zeros are unambiguous without any
    help from the dtype. The pixels outside the footprint are warped to 255 and
    dropped by the declared-nodata rule, not by the float one.
    """
    data = np.zeros((60, 60), dtype=np.uint8)
    data[:, 30:] = 1
    src = _write(tmp_path / "p8nd.tif", data, nodata=255, res=0.002)
    dst = tmp_path / "p8nd_adapted.tif"

    adapt_raster(src, dst, _target_geobox(), "probability")

    with rasterio.open(dst) as ds:
        values = set(np.unique(ds.read(1)).tolist())
    assert values == {0, 1, RISK_MAX}, sorted(values)


def test_adapt_probability_keeps_a_genuine_float_zero_as_one(tmp_path):
    """A float 0.0 probability is real data: it rescales to 1, never to nodata.

    The fill rule is keyed on an INTEGER warped dtype, because a float warp
    fills with NaN. A float source's genuine zeros therefore stay valid, and
    ``far.misc.rescale`` clamps them to 1e-6 -> 1, exactly as the spec says.
    """
    data = np.zeros((60, 60), dtype=np.float32)
    data[:, 30:] = 0.8
    src = _write(tmp_path / "pz.tif", data, nodata=None, res=0.002)
    dst = tmp_path / "pz_adapted.tif"

    adapt_raster(src, dst, _target_geobox(), "probability")

    with rasterio.open(dst) as ds:
        values = set(np.unique(ds.read(1)).tolist())
    assert 1 in values, "a genuine 0.0 probability must survive as 1"
    assert values <= {0, 1, _far_rescale(0.8)}


def test_adapt_requests_the_display_overview_pyramid(tmp_path, monkeypatch):
    """The adapted raster is handed to ensure_overviews with the display threshold."""
    import spatialrisk.overviews as overviews

    calls = []

    def spy(path, *a, **kwargs):
        calls.append((Path(path), kwargs))
        return False

    monkeypatch.setattr(overviews, "ensure_overviews", spy)
    data = np.full((60, 60), 0.5, dtype=np.float32)
    src = _write(tmp_path / "p.tif", data, res=0.002)
    dst = tmp_path / "p_adapted.tif"

    adapt_raster(src, dst, _target_geobox(), "probability")

    assert len(calls) == 1
    path, kwargs = calls[0]
    assert path == dst
    assert kwargs["min_pixels"] == overviews.OVERVIEW_MIN_PIXELS


def test_adapt_survives_a_failing_overview_build(tmp_path, monkeypatch):
    """Overviews are an optimisation: a failure must not fail the import."""
    import spatialrisk.overviews as overviews

    def boom(*a, **k):
        raise RuntimeError("no room for a pyramid")

    monkeypatch.setattr(overviews, "ensure_overviews", boom)
    data = np.full((60, 60), 0.5, dtype=np.float32)
    src = _write(tmp_path / "p.tif", data, res=0.002)
    dst = tmp_path / "p_adapted.tif"

    out = adapt_raster(src, dst, _target_geobox(), "probability")

    assert out == dst
    assert dst.exists()
    with rasterio.open(dst) as ds:
        assert ds.dtypes[0] == "uint16"


@pytest.mark.parametrize(
    ("dtype", "nodata", "scale", "expected"),
    [
        ("uint8", None, "probability", "float32"),
        ("uint16", None, "probability", "float32"),
        ("uint8", 255, "probability", None),
        ("float32", None, "probability", None),
        ("uint8", None, "risk", None),
        ("uint16", 0, "risk", None),
    ],
)
def test_warp_cast_dtype_is_narrow(tmp_path, dtype, nodata, scale, expected):
    """Only an integer probability source with no declared nodata is cast.

    Widening this would round a declared nodata value away from the one on
    disk and turn every nodata pixel back into data; narrowing it brings back
    the fill that eats a genuine 0. Both failures are silent, so the gate is
    pinned here rather than inferred from the adapted pixels.
    """
    src = _write(
        tmp_path / f"{dtype}-{nodata}.tif",
        np.ones((4, 4), dtype=np.dtype(dtype)),
        nodata=nodata,
    )
    assert _warp_cast_dtype(src, scale) == expected
