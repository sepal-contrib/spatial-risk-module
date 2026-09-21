"""Which raw variables still need harmonizing onto the current base grid?

Step 3 used to re-derive every layer on every press. These tests pin the three
conditions that let a layer be skipped: registered output, output on the
current geobox, output not older than its source.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_origin

from spatialrisk.harmonization import (
    HarmonizationStatus,
    harmonization_status_from_disk,
    is_current,
    output_key,
)
from spatialrisk.variables.models import DataType, RasterizationMethod, RasterType

# shape.yx == (20, 40); transform == from_origin(0, 200, 10, 10)
GEOBOX = GeoBox.from_bbox((0, 0, 400, 200), "EPSG:32618", resolution=10)


def _write(path: Path, transform=None, crs="EPSG:32618", shape=(20, 40)):
    """Write a 1-band float32 raster; defaults land exactly on GEOBOX."""
    transform = transform if transform is not None else from_origin(0, 200, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(np.zeros(shape, "float32"), 1)
    return path


def _touch(path: Path, when: float):
    os.utime(path, (when, when))


def _var(
    name,
    path,
    year=None,
    data_type=DataType.raster,
    active=True,
    raster_type=RasterType.continuous,
    rasterization_method=None,
):
    """A duck-typed raw variable.

    ``raster_type`` must be pinned rather than left as a MagicMock attribute:
    ``is_current`` compares it against the output's, and two auto-created
    MagicMocks never compare equal — every layer would read as pending.
    """
    v = MagicMock()
    v.name = name
    v.year = year
    v.path = path
    v.data_type = data_type
    v.active = active
    v.raster_type = raster_type
    v.rasterization_method = rasterization_method
    return v


def _project(raw, processed, base=True):
    p = MagicMock()
    p.raw_variables = raw
    p.processed_variables = processed
    if base:
        p.base_raster.get_base_geobox.return_value = GEOBOX
    else:
        p.base_raster = None
    return p


def test_output_key_mirrors_add_as_processed():
    """name_year for a temporal layer, bare name for a static one."""
    assert output_key(_var("forest", None, year=2020)) == "forest_2020"
    assert output_key(_var("altitude", None)) == "altitude"


def test_new_variable_is_pending(tmp_path):
    """No registered output at all — nothing to skip."""
    src = _write(tmp_path / "src.tif")
    p = _project({"altitude": _var("altitude", src)}, {})
    assert harmonization_status_from_disk(p) == HarmonizationStatus(
        pending=["altitude"], current=[]
    )


def test_aligned_and_fresh_output_is_current(tmp_path):
    """Registered, on the base geobox, newer than its source — skippable."""
    src = _write(tmp_path / "src.tif")
    out = _write(tmp_path / "out.tif")
    _touch(src, 1000)
    _touch(out, 2000)
    p = _project(
        {"altitude": _var("altitude", src)},
        {"altitude": _var("altitude", out)},
    )
    status = harmonization_status_from_disk(p)
    assert status.pending == []
    assert status.current == ["altitude"]
    assert status.total == 1


def test_grid_mismatch_is_pending(tmp_path):
    """An output from a previous reference raster is on the wrong grid."""
    src = _write(tmp_path / "src.tif")
    out = _write(tmp_path / "out.tif", transform=from_origin(0, 200, 25, 25))
    _touch(src, 1000)
    _touch(out, 2000)
    p = _project(
        {"altitude": _var("altitude", src)},
        {"altitude": _var("altitude", out)},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_shape_mismatch_is_pending(tmp_path):
    """Same CRS and transform, fewer rows/columns — a cropped base extent.

    The origin and pixel size are untouched, so only ``src.shape`` differs; an
    output covering a smaller area than the current reference grid cannot be
    reused, and the model would train on layers that do not overlap.
    """
    src = _write(tmp_path / "src.tif")
    out = _write(tmp_path / "out.tif", shape=(10, 40))
    _touch(src, 1000)
    _touch(out, 2000)
    p = _project(
        {"altitude": _var("altitude", src)},
        {"altitude": _var("altitude", out)},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_crs_mismatch_is_pending(tmp_path):
    """Same transform and shape, different CRS — still the wrong grid."""
    src = _write(tmp_path / "src.tif")
    out = _write(tmp_path / "out.tif", crs="EPSG:32619")
    _touch(src, 1000)
    _touch(out, 2000)
    p = _project(
        {"altitude": _var("altitude", src)},
        {"altitude": _var("altitude", out)},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_output_older_than_source_is_pending(tmp_path):
    """The raw variable was re-pointed at a newer file under the same key."""
    src = _write(tmp_path / "src.tif")
    out = _write(tmp_path / "out.tif")
    _touch(out, 1000)
    _touch(src, 2000)
    p = _project(
        {"altitude": _var("altitude", src)},
        {"altitude": _var("altitude", out)},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_changed_raster_type_is_pending(tmp_path):
    """Editing categorical->continuous changes the resampling, not the file."""
    src = _write(tmp_path / "src.tif")
    out = _write(tmp_path / "out.tif")
    _touch(src, 1000)
    _touch(out, 2000)
    p = _project(
        {"altitude": _var("altitude", src, raster_type=RasterType.categorical)},
        {"altitude": _var("altitude", out, raster_type=RasterType.continuous)},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_changed_rasterization_method_is_pending(tmp_path):
    """Binary -> unique changes the rasterize mode, not the shapefile."""
    src = tmp_path / "roads.geojson"
    src.write_text("{}")
    out = _write(tmp_path / "roads.tif")
    _touch(src, 1000)
    _touch(out, 2000)
    p = _project(
        {
            "roads": _var(
                "roads",
                src,
                data_type=DataType.vector,
                rasterization_method=RasterizationMethod.unique,
            )
        },
        # binary rasterization produced a continuous output; unique wants
        # categorical, so the registered output is the wrong product.
        {"roads": _var("roads", out, raster_type=RasterType.continuous)},
    )
    assert harmonization_status_from_disk(p).pending == ["roads"]


def test_vectors_sharing_one_output_file_are_both_pending(tmp_path):
    """Rasterize writes {name}.tif with no year, so siblings collide.

    Until Task F5 adds the year to that path, one of the two registry entries
    describes bytes that belong to the other year — neither may be claimed.
    """
    src_a = tmp_path / "roads_2000.geojson"
    src_a.write_text("{}")
    src_b = tmp_path / "roads_2020.geojson"
    src_b.write_text("{}")
    out = _write(tmp_path / "roads.tif")
    _touch(src_a, 1000)
    _touch(src_b, 1000)
    _touch(out, 2000)
    shared = _var("roads", out, raster_type=RasterType.continuous)
    p = _project(
        {
            "roads_2000": _var("roads", src_a, year=2000, data_type=DataType.vector),
            "roads_2020": _var("roads", src_b, year=2020, data_type=DataType.vector),
        },
        {
            "roads_2000": shared,
            "roads_2020": _var(
                "roads", out, year=2020, raster_type=RasterType.continuous
            ),
        },
    )
    status = harmonization_status_from_disk(p)
    assert status.current == []
    assert sorted(status.pending) == ["roads_2000", "roads_2020"]


def test_missing_output_file_is_pending(tmp_path):
    """Registered but deleted from disk."""
    src = _write(tmp_path / "src.tif")
    p = _project(
        {"altitude": _var("altitude", src)},
        {"altitude": _var("altitude", tmp_path / "gone.tif")},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_unreadable_output_is_pending(tmp_path):
    """A truncated/corrupt output is re-derived, never trusted."""
    src = _write(tmp_path / "src.tif")
    out = tmp_path / "out.tif"
    out.write_bytes(b"not a geotiff")
    _touch(src, 1000)
    _touch(out, 2000)
    p = _project(
        {"altitude": _var("altitude", src)},
        {"altitude": _var("altitude", out)},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_missing_source_file_is_pending(tmp_path):
    """Deliberate: the run then fails loudly, exactly as it does today."""
    out = _write(tmp_path / "out.tif")
    p = _project(
        {"altitude": _var("altitude", tmp_path / "gone.tif")},
        {"altitude": _var("altitude", out)},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_gee_var_without_local_path_is_pending(tmp_path):
    """A cloud-backed layer has no file to compare — it must be downloaded."""
    out = _write(tmp_path / "out.tif")
    p = _project(
        {"altitude": _var("altitude", None)},
        {"altitude": _var("altitude", out)},
    )
    assert harmonization_status_from_disk(p).pending == ["altitude"]


def test_temporal_layers_are_tracked_per_year(tmp_path):
    """forest_2000 can be current while forest_2020 is pending."""
    src_a = _write(tmp_path / "a.tif")
    out_a = _write(tmp_path / "a_out.tif")
    src_b = _write(tmp_path / "b.tif")
    _touch(src_a, 1000)
    _touch(out_a, 2000)
    p = _project(
        {
            "forest_2000": _var("forest", src_a, year=2000),
            "forest_2020": _var("forest", src_b, year=2020),
        },
        {"forest_2000": _var("forest", out_a, year=2000)},
    )
    status = harmonization_status_from_disk(p)
    assert status.current == ["forest_2000"]
    assert status.pending == ["forest_2020"]


def test_inactive_vector_is_neither_pending_nor_current(tmp_path):
    """rasterize_all skips inactive vectors, so status must not claim them."""
    src = tmp_path / "roads.geojson"
    src.write_text("{}")
    p = _project(
        {"roads": _var("roads", src, data_type=DataType.vector, active=False)},
        {},
    )
    status = harmonization_status_from_disk(p)
    assert status.pending == []
    assert status.current == []
    assert status.total == 0


def test_active_vector_is_a_candidate(tmp_path):
    """An active vector is harmonized (rasterized), so it belongs in the split."""
    src = tmp_path / "roads.geojson"
    src.write_text("{}")
    p = _project(
        {"roads": _var("roads", src, data_type=DataType.vector, active=True)},
        {},
    )
    assert harmonization_status_from_disk(p).pending == ["roads"]


def test_no_base_raster_reports_everything_pending(tmp_path):
    """No grid to compare against; run_processing raises before it matters."""
    src = _write(tmp_path / "src.tif")
    p = _project({"altitude": _var("altitude", src)}, {}, base=False)
    status = harmonization_status_from_disk(p)
    assert status.pending == ["altitude"]
    assert status.current == []


def test_is_current_returns_false_without_a_registered_output(tmp_path):
    """No registered output means nothing to trust — and no file is opened."""
    src = _write(tmp_path / "src.tif")
    assert is_current(_var("altitude", src), None, GEOBOX) is False


def test_keys_restricts_the_scan_to_the_listed_raw_keys(tmp_path):
    """The tile re-checks only its unstamped layers; the rest are not even read."""
    src = _write(tmp_path / "src.tif")
    p = _project(
        {"altitude": _var("altitude", src), "slope": _var("slope", src)},
        {},
    )
    status = harmonization_status_from_disk(p, keys=["slope"])
    assert status.pending == ["slope"]
    assert status.total == 1


def test_empty_keys_scans_nothing(tmp_path):
    """An empty restriction must mean 'no layers', not 'every layer'."""
    src = _write(tmp_path / "src.tif")
    p = _project({"altitude": _var("altitude", src)}, {})
    assert harmonization_status_from_disk(p, keys=[]) == HarmonizationStatus(
        pending=[], current=[]
    )
