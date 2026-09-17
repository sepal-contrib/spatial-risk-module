"""The display status is decided in memory and never touches the filesystem."""

import pytest

from spatialrisk.harmonization import harmonization_status
from spatialrisk.variables.models import DataType, RasterType


class _Var:
    def __init__(self, name, sig=None, data_type=DataType.raster, year=None):
        self.name = name
        self.year = year
        self.data_type = data_type
        self.raster_type = RasterType.continuous
        self.active = True
        self.path = f"/nonexistent/{name}.tif"
        self.grid_signature = sig


class _Project:
    def __init__(self, raw, processed, base_sig):
        self.raw_variables = raw
        self.processed_variables = processed
        self.base_raster = _Var("base", sig=base_sig) if base_sig else None


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    """Any file access from the pure path is a failure, not a slow path."""
    import rasterio

    def _boom(*a, **k):
        raise AssertionError("harmonization_status must not open files")

    monkeypatch.setattr(rasterio, "open", _boom)


def test_matching_signature_is_current():
    """Output written onto the live base grid needs no work."""
    p = _Project({"a": _Var("a")}, {"a": _Var("a", sig="SIG")}, base_sig="SIG")
    assert harmonization_status(p).current == ["a"]


def test_stale_signature_is_pending():
    """The reference moved, so the output is on the wrong grid."""
    p = _Project({"a": _Var("a")}, {"a": _Var("a", sig="OLD")}, base_sig="NEW")
    assert harmonization_status(p).pending == ["a"]


def test_missing_output_is_pending():
    """Never harmonized at all."""
    p = _Project({"a": _Var("a")}, {}, base_sig="SIG")
    assert harmonization_status(p).pending == ["a"]


def test_changed_raster_type_is_pending_despite_a_matching_grid():
    """Right grid, wrong resampling semantics — is_current catches this on disk."""
    raw = _Var("a")
    raw.raster_type = RasterType.categorical
    out = _Var("a", sig="SIG")
    out.raster_type = RasterType.continuous
    p = _Project({"a": raw}, {"a": out}, base_sig="SIG")
    assert harmonization_status(p).pending == ["a"]


def test_unstamped_output_is_unknown():
    """A project written before the field existed defers to the disk check."""
    p = _Project({"a": _Var("a")}, {"a": _Var("a", sig=None)}, base_sig="SIG")
    status = harmonization_status(p)
    assert status.unknown == ["a"]
    assert status.pending == [] and status.current == []


def test_no_base_raster_makes_everything_pending():
    """No grid to compare against."""
    p = _Project({"a": _Var("a")}, {}, base_sig=None)
    assert harmonization_status(p).pending == ["a"]


def test_unstamped_base_makes_everything_unknown():
    """The base predates the field, so nothing can be compared."""
    p = _Project({"a": _Var("a")}, {"a": _Var("a", sig="SIG")}, base_sig=None)
    p.base_raster = _Var("base", sig=None)
    assert harmonization_status(p).unknown == ["a"]


def test_inactive_vector_is_excluded():
    """rasterize_all filters on active, so counting it would make the hint lie."""
    v = _Var("v", data_type=DataType.vector)
    v.active = False
    p = _Project({"v": v}, {}, base_sig="SIG")
    assert harmonization_status(p).total == 0


def test_a_never_harmonized_layer_is_pending_even_against_an_unstamped_base():
    """A missing output is a complete answer without any base signature.

    Deferring it to ``unknown`` makes every row of a legacy project the cached
    disk verdict — and no in-place variable edit moves that cache, because the
    edit keeps the ``{name}_{year}`` key and only drops the processed entry. The
    row then goes on reading harmonized over a raster the edit invalidated.
    """
    p = _Project(
        {"a": _Var("a"), "b": _Var("b")},
        {"b": _Var("b", sig="SIG")},
        base_sig=None,
    )
    p.base_raster = _Var("base", sig=None)
    status = harmonization_status(p)
    assert status.pending == ["a"]
    assert status.unknown == ["b"]
    assert status.current == []
