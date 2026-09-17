"""The stored grid signature: formatting, stability, and persistence."""

from spatialrisk.harmonization import geobox_signature


class _Shape:
    def __init__(self, rows, cols):
        self.yx = (rows, cols)


class _Geobox:
    """Duck-typed stand-in — harmonization.py never imports odc types."""

    def __init__(self, crs="EPSG:32618", transform=None, rows=100, cols=200):
        self.crs = crs
        self.transform = transform or (30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
        self.shape = _Shape(rows, cols)


def test_signature_is_stable_across_calls():
    """Same grid, same string — the comparison depends on it."""
    assert geobox_signature(_Geobox()) == geobox_signature(_Geobox())


def test_signature_distinguishes_crs():
    """A reprojection to a different CRS must invalidate every output."""
    assert geobox_signature(_Geobox(crs="EPSG:32618")) != geobox_signature(
        _Geobox(crs="EPSG:4326")
    )


def test_signature_distinguishes_resolution():
    """30 m and 100 m are different grids even in the same CRS."""
    a = _Geobox(transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0))
    b = _Geobox(transform=(100.0, 0.0, 500000.0, 0.0, -100.0, 4000000.0))
    assert geobox_signature(a) != geobox_signature(b)


def test_signature_distinguishes_shape():
    """Same origin and pixel size, different extent."""
    assert geobox_signature(_Geobox(rows=100)) != geobox_signature(_Geobox(rows=101))


def test_near_identical_coefficients_do_not_collapse():
    """%.10g rounded -29.999999999 to -30; repr must keep them apart."""
    a = _Geobox(transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0))
    b = _Geobox(transform=(30.0, 0.0, 500000.0, 0.0, -29.999999999, 4000000.0))
    assert geobox_signature(a) != geobox_signature(b)


def test_signature_survives_a_json_round_trip():
    """It is persisted as JSON; float formatting must not drift."""
    import json

    sig = geobox_signature(_Geobox())
    assert json.loads(json.dumps({"s": sig}))["s"] == sig


def test_field_round_trips_through_model_dump(tmp_path):
    """Save writes model_dump and load does LocalRasterVar(**data) — pin that."""
    from spatialrisk.variables.local_raster_var import LocalRasterVar
    from spatialrisk.variables.models import RasterType

    path = tmp_path / "layer.tif"
    path.write_bytes(b"")
    var = LocalRasterVar(
        name="layer",
        path=path,
        raster_type=RasterType.continuous,
        grid_signature="EPSG:32618|30|0|5e5|0|-30|4e6|100x200",
    )
    dumped = var.model_dump(mode="json")
    assert dumped["grid_signature"] == var.grid_signature
    assert (
        LocalRasterVar(**{**dumped, "path": path}).grid_signature == var.grid_signature
    )
