"""``keys=`` restricts the bulk harmonization methods to a subset of layers.

Step 3 re-derived every variable on every press; the incremental run
(``process_actions.run_processing``) passes only the pending keys. ``None``
must keep the process-everything behaviour the notebooks rely on.
"""

from unittest.mock import MagicMock

from spatialrisk import Project
from spatialrisk.variables.models import DataType


def _raster(name):
    v = MagicMock()
    v.name = name
    v.year = None
    v.active = True
    v.data_type = DataType.raster
    v.reproject_and_match.return_value = _product(name)
    return v


def _vector(name):
    v = MagicMock()
    v.name = name
    v.year = None
    v.active = True
    v.data_type = DataType.vector
    v.rasterize.return_value = _product(name)
    return v


def _product(name):
    out = MagicMock()
    out.name = name
    out.year = None
    return out


def _project():
    p = Project(project_name="keys-filter")
    p.base_raster = MagicMock()
    p.raw_variables = {"a": _raster("a"), "b": _raster("b"), "roads": _vector("roads")}
    return p


def test_reproject_keys_none_processes_every_raster():
    """keys=None keeps today's process-everything behaviour for rasters."""
    p = _project()
    p.reproject_and_match_all(keys=None, add_to_processed=False, auto_save=False)
    assert p.raw_variables["a"].reproject_and_match.called
    assert p.raw_variables["b"].reproject_and_match.called


def test_reproject_keys_restricts_to_the_listed_keys():
    """Only the listed key is reprojected; unlisted rasters are left alone."""
    p = _project()
    p.reproject_and_match_all(keys=["b"], add_to_processed=False, auto_save=False)
    assert not p.raw_variables["a"].reproject_and_match.called
    assert p.raw_variables["b"].reproject_and_match.called


def test_reproject_empty_keys_processes_nothing():
    """An empty pending list must mean 'no work', not 'everything'."""
    p = _project()
    out = p.reproject_and_match_all(keys=[], add_to_processed=False, auto_save=False)
    assert out == {}
    assert not p.raw_variables["a"].reproject_and_match.called


def test_rasterize_keys_restricts_to_the_listed_keys():
    """Only the listed key is rasterized; unlisted vectors are left alone."""
    p = _project()
    p.raw_variables["rivers"] = _vector("rivers")
    p.rasterize_all(keys=["rivers"], add_to_processed=False, auto_save=False)
    assert not p.raw_variables["roads"].rasterize.called
    assert p.raw_variables["rivers"].rasterize.called


def test_rasterize_keys_none_processes_every_active_vector():
    """keys=None keeps today's process-everything behaviour for active vectors."""
    p = _project()
    p.rasterize_all(keys=None, add_to_processed=False, auto_save=False)
    assert p.raw_variables["roads"].rasterize.called


def test_rasterize_empty_keys_processes_nothing():
    """An empty pending list must mean 'no work', not 'everything'."""
    p = _project()
    out = p.rasterize_all(keys=[], add_to_processed=False, auto_save=False)
    assert out == {}
    assert not p.raw_variables["roads"].rasterize.called
