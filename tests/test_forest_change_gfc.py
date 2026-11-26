"""deforestation_gfc == the Process step's loss_forest_gfc_tc<T>_<start>_<end>.

The interval layer is built from the very ``_get_forest_gfc`` expressions the
Process step downloads at ``start`` and ``end``; these tests pin the truth
table of the plan's proof and run the real ``process_change_xarray`` on the
two forest exports to show the arrays are equal pixel for pixel.
"""

import inspect
import itertools

import numpy as np
import pytest
import rasterio
from np_ee import FakeAoi, NpImage, export, fake_ee
from rasterio.transform import from_origin

import gui.scripts.predefined.forest as forest
import gui.scripts.predefined.forest_change as fc
from gui.scripts.predefined import GROUP_ORDER, PREDEFINED_CATALOGUE

T = 30
#: (tree cover, lossyear (0 = no loss, masked in the asset), inside the AOI)
ROWS = list(itertools.product((T - 1, T), range(0, 26), (True, False)))
PAIRS = [(2001, 2002), (2010, 2011), (2015, 2020), (2025, 2026), (2001, 2026)]


def _expected(tc, lossyear, inside, start, end):
    if not inside or tc < T:
        return 255
    if lossyear == 0:
        return 0
    if lossyear < start - 2000:
        return 255
    if lossyear <= end - 2001:
        return 1
    return 0


@pytest.fixture
def aoi(monkeypatch):
    """One pixel per truth-table row; ``forest.ee`` serves the fake GFC asset."""
    tc = np.array([[row[0] for row in ROWS]])
    lossyear = np.array([[row[1] for row in ROWS]])
    inside = np.array([[row[2] for row in ROWS]])
    gfc = NpImage(
        {"treecover2000": tc, "lossyear": lossyear}, {"lossyear": lossyear > 0}
    )
    monkeypatch.setattr(forest, "ee", fake_ee(tc.shape, images={"gfc": gfc}))
    return FakeAoi(inside)


def _defo(aoi, start, end):
    image = fc._get_deforestation_gfc(
        aoi, tree_cover_threshold=T, start_year=start, end_year=end
    )
    return export(image, aoi)


@pytest.mark.parametrize("start,end", PAIRS)
def test_builder_matches_the_documented_truth_table(aoi, start, end):
    """Each row of the proof table, for edge and mid-range periods."""
    want = np.array([[_expected(*row, start, end) for row in ROWS]], dtype=np.uint8)
    np.testing.assert_array_equal(_defo(aoi, start, end), want)


def _write(path, array):
    # Two identical rows: process_change_xarray squeezes, and a one-row
    # raster would lose its y dimension.
    array = np.vstack([array, array])
    profile = {
        "driver": "GTiff",
        "width": array.shape[1],
        "height": array.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(-63.0, -10.0, 0.00025, 0.00025),
        "nodata": 255,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
    return str(path)


@pytest.mark.raster
@pytest.mark.parametrize("start,end", PAIRS)
def test_builder_equals_the_process_step_loss_layer(aoi, tmp_path, start, end):
    """Bit-identical to process_change_xarray(forest_gfc@start, forest_gfc@end)."""
    from spatialrisk.processing import process_change_xarray

    forest_start = export(forest._get_forest_gfc(aoi, start, T), aoi)
    forest_end = export(forest._get_forest_gfc(aoi, end, T), aoi)
    out = tmp_path / "loss.tif"
    process_change_xarray(
        _write(tmp_path / "start.tif", forest_start),
        _write(tmp_path / "end.tif", forest_end),
        str(out),
        op="loss",
    )
    with rasterio.open(out) as src:
        loss = src.read(1)
    np.testing.assert_array_equal(_defo(aoi, start, end), loss[:1])


@pytest.mark.parametrize(
    "start,end",
    [
        (None, 2020),
        (2015, None),
        (2020, 2015),
        (2015, 2015),
        (2000, 2005),
        (2015, 2027),
    ],
)
def test_builder_rejects_a_bad_interval(aoi, start, end):
    """Missing, reversed, equal or out-of-range years fail before any EE call."""
    with pytest.raises(ValueError):
        fc._get_deforestation_gfc(aoi, start_year=start, end_year=end)


def test_entry_is_a_non_temporal_interval_layer():
    """Storage key = name; the period lives in interval_years, not in years."""
    entry = PREDEFINED_CATALOGUE["deforestation_gfc"]
    assert entry["temporal"] is False
    assert "years" not in entry
    assert entry["interval_years"] == forest.ENTRIES["forest_gfc"]["years"]
    assert entry["raster_type"] == "categorical"
    assert entry["get_image"] is fc._get_deforestation_gfc
    assert entry["group"] == "forest_change"
    assert entry["forest_at_start"]({"tree_cover_threshold": 45}) == (
        "forest_gfc",
        {"tree_cover_threshold": 45},
    )


def test_tree_cover_param_is_forest_gfc_s():
    """Same knob, default and name segment as forest_gfc (tc30)."""
    (spec,) = PREDEFINED_CATALOGUE["deforestation_gfc"]["params"]
    assert spec == forest.ENTRIES["forest_gfc"]["params"][0]
    assert spec is not forest.ENTRIES["forest_gfc"]["params"][0]  # a copy


def test_builder_kwarg_defaults_match_the_catalogue():
    """The tc default is 30 like forest_gfc; the years have no default."""
    sig = inspect.signature(fc._get_deforestation_gfc)
    assert sig.parameters["tree_cover_threshold"].default == 30
    assert sig.parameters["start_year"].default is None
    assert sig.parameters["end_year"].default is None


def test_forest_change_group_follows_forest():
    """The new dropdown header sits right after the forest group."""
    assert GROUP_ORDER.index("forest_change") == GROUP_ORDER.index("forest") + 1
