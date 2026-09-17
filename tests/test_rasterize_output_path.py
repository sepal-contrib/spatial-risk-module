"""Each year of a temporal vector needs its own rasterized output file.

`rasterize` wrote `{name}.tif` for every year while `add_as_processed`
registered under `{name}_{year}`, so two registry entries pointed at one file
and the later rasterization silently overwrote the earlier one.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spatialrisk.project import Project
from spatialrisk.variables.local_vector_var import LocalVectorVar
from spatialrisk.variables.models import DataType, RasterizationMethod

# Real (non-mocked) LocalVectorVar construction below needs the models fully
# resolved. Their module-level auto-rebuild-on-import never actually
# completes (spatialrisk.variables.__init__ doesn't export ``Variable``, so the
# bottom-of-file rebuild block's import fails and is swallowed) -- this is the
# established workaround used by test_process_tile_hint.py,
# test_base_raster_year_identity.py and others.
Project._ensure_model_schemas()


@pytest.fixture
def base():
    """A stubbed base raster with a mocked geobox."""
    b = MagicMock()
    b.data_type = DataType.raster
    b.get_base_geobox.return_value = MagicMock()
    return b


def _vector(tmp_path, name, year):
    src = tmp_path / f"{name}.geojson"
    src.write_text("{}")
    project = MagicMock()
    project.folders.data_raw_folder = tmp_path
    # model_construct(): a bare mock fails strict validation against the
    # ``Optional["Project"]`` field (see test_reproject_all_nodata_guard.py's
    # ``local_var`` fixture and test_get_variable_single_year.py for the same
    # pattern) -- construction here doesn't need field validation anyway.
    return LocalVectorVar.model_construct(
        name=name,
        path=src,
        year=year,
        project=project,
        rasterization_method=RasterizationMethod.binary,
    )


def _rasterize(var, base):
    """Run rasterize with the GDAL work patched out; return the output path."""
    with patch("spatialrisk.variables.local_vector_var.xr_rasterize") as xr_rast:
        out = var.rasterize(base=base)
    return Path(xr_rast.call_args.kwargs["output_path"]), Path(out.path)


def test_static_vector_keeps_the_unsuffixed_path(tmp_path, base):
    """year=None must not change on disk — existing files stay valid."""
    var = _vector(tmp_path, "roads", None)
    written, registered = _rasterize(var, base)
    assert written.name == "roads.tif"
    assert registered.name == "roads.tif"


def test_temporal_vectors_get_distinct_paths(tmp_path, base):
    """The whole point: two years, two files."""
    a = _vector(tmp_path, "roads", 2000)
    b = _vector(tmp_path, "roads", 2020)
    written_a, registered_a = _rasterize(a, base)
    written_b, registered_b = _rasterize(b, base)
    assert written_a.name == "roads_2000.tif"
    assert written_b.name == "roads_2020.tif"
    assert written_a != written_b
    # The registered path must be the one actually written: harmonization_status
    # stats the registered path, so a divergence here would have it inspecting a
    # file the rasterization never produced.
    assert registered_a == written_a
    assert registered_b == written_b
