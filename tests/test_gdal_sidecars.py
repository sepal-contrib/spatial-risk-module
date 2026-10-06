"""Overview sidecars stay visible to GDAL after the map's tile server starts.

localtileserver's ``create_app`` sets ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR``
in ``os.environ`` -- process-wide, so from the first map layer on GDAL stops
looking for ``.ovr`` sidecars anywhere in the app. The tile server then renders
every tile of a large raster from full resolution (it computes whole-raster
statistics per tile), and ``ensure_overviews`` rebuilds a pyramid that is
already on disk. Measured on a 2.2 Gpx BOL raster: ~50 s per tile hidden,
~1 s visible.
"""

import os

import numpy as np
import pytest
import rasterio
from localtileserver.web.fastapi_app import create_app
from rasterio.transform import from_origin

from spatialrisk.gdal_env import keep_gdal_sidecars_visible
from spatialrisk.overviews import ensure_overviews

READDIR = "GDAL_DISABLE_READDIR_ON_OPEN"
#: Everything localtileserver's ``create_app`` writes to ``os.environ``.
TILE_SERVER_ENV = (READDIR, "GDAL_NUM_THREADS")


@pytest.fixture(autouse=True)
def clean_gdal_env():
    """Start each test with the tile-server vars unset; put the originals back.

    Not ``monkeypatch.delenv``: on a var that is not set it records nothing,
    so whatever ``create_app`` then sets would leak into later test files.
    """
    saved = {k: os.environ.get(k) for k in TILE_SERVER_ENV}
    for k in TILE_SERVER_ENV:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _write_raster_with_pyramid(path):
    """A small UInt16 raster plus its external ``.ovr`` sidecar."""
    data = (np.arange(256 * 256, dtype=np.uint16).reshape(256, 256) % 65535) + 1
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "height": 256,
        "width": 256,
        "nodata": 0,
        "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 1 / 256, 1 / 256),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    assert ensure_overviews(str(path)) is True


def test_tile_server_setup_hides_sidecars_process_wide():
    """Why the guard exists: the tile server's setup writes EMPTY_DIR to os.environ."""
    create_app()

    assert os.environ[READDIR] == "EMPTY_DIR"


def test_existing_pyramid_is_reused_after_tile_server_setup(tmp_path):
    """A pyramid already on disk is not rebuilt once the tile server has started."""
    tif = tmp_path / "pred.tif"
    _write_raster_with_pyramid(tif)
    create_app()

    keep_gdal_sidecars_visible()

    assert ensure_overviews(str(tif)) is False


def test_tile_reads_see_the_pyramid_after_tile_server_setup(tmp_path):
    """The tile server's read path (rasterio) sees the sidecar levels."""
    tif = tmp_path / "pred.tif"
    _write_raster_with_pyramid(tif)
    create_app()

    keep_gdal_sidecars_visible()

    with rasterio.open(tif) as src:
        assert len(src.overviews(1)) > 0


def test_guard_holds_when_it_runs_before_the_tile_server(tmp_path):
    """Startup order does not matter: the server's setdefault cannot undo it."""
    tif = tmp_path / "pred.tif"
    _write_raster_with_pyramid(tif)

    keep_gdal_sidecars_visible()
    create_app()

    with rasterio.open(tif) as src:
        assert len(src.overviews(1)) > 0


def test_app_startup_keeps_sidecars_visible():
    """Importing the app (as voila and ``solara run`` do) installs the guard."""
    import subprocess
    import sys
    from pathlib import Path

    env = {**os.environ, READDIR: "EMPTY_DIR"}
    out = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            f"import os, gui.solara_app; print(os.environ[{READDIR!r}])",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )

    assert out.stdout.strip().splitlines()[-1] == "FALSE"
