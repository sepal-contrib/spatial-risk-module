"""Tests for generate_change_var / change_layer_candidates (post-process tab)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gui.scripts import process_actions
from spatialrisk.variables.models import DataType


class _Folders:
    def __init__(self, root: Path):
        self.processed_data_folder = root


class _Proj:
    def __init__(self, root: Path):
        self.processed_variables = {}
        self._root = root
        self.saved = False

    @property
    def folders(self):
        return _Folders(self._root)

    def save(self):
        self.saved = True


def _layer(name, year, path, data_type=DataType.raster):
    v = MagicMock()
    v.name = name
    v.year = year
    v.path = Path(path)
    v.data_type = data_type
    return v


def _proj(tmp_path):
    p = _Proj(tmp_path)
    p.processed_variables = {
        "forest_gfc_2015": _layer("forest_gfc", 2015, tmp_path / "a.tif"),
        "forest_gfc_2020": _layer("forest_gfc", 2020, tmp_path / "b.tif"),
        "forest_tmf_2020": _layer("forest_tmf", 2020, tmp_path / "c.tif"),
        "altitude": _layer("altitude", None, tmp_path / "d.tif"),
        "roads": _layer("roads", None, tmp_path / "e.geojson", DataType.vector),
    }
    return p


def _generate(p, op, start, end):
    """Run generate_change_var with the heavy pieces patched out."""
    with patch.object(process_actions, "_check_same_grid"), patch(
        "spatialrisk.processing.process_change_xarray"
    ) as m_proc, patch("spatialrisk.variables.LocalRasterVar") as m_lrv:
        m_lrv.return_value = MagicMock(name="change_var")
        out = process_actions.generate_change_var(p, op, start, end)
    return out, m_proc, m_lrv


def test_candidates_temporal_rasters_only(tmp_path):
    """Only rasters with a year can be a start or end layer."""
    cands = process_actions.change_layer_candidates(_proj(tmp_path))
    assert cands == ["forest_gfc_2015", "forest_gfc_2020", "forest_tmf_2020"]


def test_same_source_naming(tmp_path):
    """One layer, two years: named, computed, registered and saved."""
    p = _proj(tmp_path)
    out, m_proc, m_lrv = _generate(p, "loss", "forest_gfc_2015", "forest_gfc_2020")
    kwargs = m_lrv.call_args.kwargs
    assert kwargs["name"] == "loss_forest_gfc_2015_2020"
    assert kwargs["tags"] == ["loss", "change", "2015_2020"]
    assert kwargs["path"] == tmp_path / "loss_forest_gfc_2015_2020.tif"
    m_proc.assert_called_once_with(
        str(tmp_path / "a.tif"),
        str(tmp_path / "b.tif"),
        str(tmp_path / "loss_forest_gfc_2015_2020.tif"),
        op="loss",
    )
    out.add_as_processed.assert_called_once_with(auto_save=False)
    assert p.saved is True
    # Downstream (dataset static-target path) relies on these: a categorical,
    # non-temporal var bound to this project.
    from spatialrisk.variables.models import RasterType

    assert kwargs["raster_type"] == RasterType.categorical
    assert kwargs["project"] is p
    assert "year" not in kwargs  # static — year defaults to None


def test_cross_source_naming(tmp_path):
    """Two different layers: both names go into the output name."""
    p = _proj(tmp_path)
    _, _, m_lrv = _generate(p, "loss", "forest_gfc_2015", "forest_tmf_2020")
    assert m_lrv.call_args.kwargs["name"] == "loss_forest_gfc_2015_forest_tmf_2020"


def test_gain_naming(tmp_path):
    """A gain layer takes the gain prefix."""
    p = _proj(tmp_path)
    _, _, m_lrv = _generate(p, "gain", "forest_gfc_2015", "forest_gfc_2020")
    assert m_lrv.call_args.kwargs["name"] == "gain_forest_gfc_2015_2020"


def test_existing_layer_is_recomputed_not_reused(tmp_path):
    """A registered layer may come from an older grid or older inputs."""
    p = _proj(tmp_path)
    sentinel = MagicMock(name="already_there")
    p.processed_variables["loss_forest_gfc_2015_2020"] = sentinel
    out, m_proc, _ = _generate(p, "loss", "forest_gfc_2015", "forest_gfc_2020")
    assert out is not sentinel
    m_proc.assert_called_once()
    out.add_as_processed.assert_called_once_with(auto_save=False)
    assert p.saved is True


def test_existing_tif_is_overwritten(tmp_path):
    """A file left by a run that died mid-write must not be registered as is."""
    p = _proj(tmp_path)
    (tmp_path / "loss_forest_gfc_2015_2020.tif").touch()
    out, m_proc, _ = _generate(p, "loss", "forest_gfc_2015", "forest_gfc_2020")
    m_proc.assert_called_once()
    out.add_as_processed.assert_called_once_with(auto_save=False)
    assert p.saved is True


def test_dialog_asks_before_recomputing_an_existing_change_layer(tmp_path):
    """Only a loss or gain that is already registered asks to be replaced."""
    from gui.widget.derived_layer_dialog import replaced_change_key

    p = _proj(tmp_path)
    loss = {
        "op": "loss",
        "start_key": "forest_gfc_2015",
        "end_key": "forest_gfc_2020",
        "pp_key": "",
    }
    assert replaced_change_key(p, loss) is None
    p.processed_variables["loss_forest_gfc_2015_2020"] = MagicMock()
    assert replaced_change_key(p, loss) == "loss_forest_gfc_2015_2020"
    assert replaced_change_key(p, {**loss, "op": "gain"}) is None
    dist = {"op": "dist", "start_key": "", "end_key": "", "pp_key": "altitude"}
    assert replaced_change_key(p, dist) is None
    assert replaced_change_key(None, loss) is None


def test_rejects_bad_op(tmp_path):
    """Only loss and gain are change operations."""
    with pytest.raises(ValueError, match="op"):
        process_actions.generate_change_var(
            _proj(tmp_path), "delta", "forest_gfc_2015", "forest_gfc_2020"
        )


def test_rejects_same_layer_twice(tmp_path):
    """Start and end must be different layers."""
    with pytest.raises(ValueError, match="different"):
        process_actions.generate_change_var(
            _proj(tmp_path), "loss", "forest_gfc_2015", "forest_gfc_2015"
        )


def test_rejects_missing_key(tmp_path):
    """An unknown key is named in the error."""
    with pytest.raises(ValueError, match="nope_2020"):
        process_actions.generate_change_var(
            _proj(tmp_path), "loss", "forest_gfc_2015", "nope_2020"
        )


def test_rejects_static_layer(tmp_path):
    """A layer without a year cannot be differenced."""
    with pytest.raises(ValueError, match="year"):
        process_actions.generate_change_var(
            _proj(tmp_path), "loss", "forest_gfc_2015", "altitude"
        )


def test_rejects_year_order(tmp_path):
    """The start year must come before the end year."""
    with pytest.raises(ValueError, match="earlier"):
        process_actions.generate_change_var(
            _proj(tmp_path), "loss", "forest_gfc_2020", "forest_gfc_2015"
        )


def test_check_same_grid_raises_on_mismatch(tmp_path):
    """Two rasters on different grids are refused."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    def _write(path, origin_x):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(origin_x, 2, 1, 1),
            nodata=255,
        ) as dst:
            dst.write(np.zeros((2, 2), dtype="uint8"), 1)

    a, b, c = tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "c.tif"
    _write(a, 0)
    _write(b, 0)
    _write(c, 5)  # shifted grid

    va, vb, vc = MagicMock(), MagicMock(), MagicMock()
    va.path, vb.path, vc.path = a, b, c
    va.name, vb.name, vc.name = "va", "vb", "vc"

    process_actions._check_same_grid(va, vb)  # same grid: no raise
    with pytest.raises(ValueError, match="grid"):
        process_actions._check_same_grid(va, vc)
