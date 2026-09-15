"""Thread-count policy shared by the raster scans."""

import pytest

rasterio = pytest.importorskip("rasterio")


def test_worker_threads_env_override_and_floor(monkeypatch):
    """The env override wins and the thread count never drops below one."""
    from spatialrisk.parallel import NUM_THREADS_ENV, worker_threads

    monkeypatch.delenv(NUM_THREADS_ENV, raising=False)
    assert worker_threads() >= 1
    monkeypatch.setenv(NUM_THREADS_ENV, "3")
    assert worker_threads() == 3
    monkeypatch.setenv(NUM_THREADS_ENV, "0")
    assert worker_threads() == 1


def test_worker_threads_specific_override_beats_generic_and_cores(monkeypatch):
    """A job-specific variable wins over the generic one, which wins over cores."""
    from spatialrisk.parallel import NUM_THREADS_ENV, worker_threads

    monkeypatch.delenv(NUM_THREADS_ENV, raising=False)
    monkeypatch.delenv("SPATIAL_RISK_TEST_THREADS", raising=False)
    assert worker_threads("SPATIAL_RISK_TEST_THREADS", cores=16) == 8
    assert worker_threads("SPATIAL_RISK_TEST_THREADS", cores=1) == 1
    monkeypatch.setenv(NUM_THREADS_ENV, "5")
    assert worker_threads("SPATIAL_RISK_TEST_THREADS", cores=16) == 5
    monkeypatch.setenv("SPATIAL_RISK_TEST_THREADS", "2")
    assert worker_threads("SPATIAL_RISK_TEST_THREADS", cores=16) == 2


def test_scan_env_caps_the_block_cache():
    """Inside the env rasterio's GDAL carries the capped budget.

    Checked through rasterio rather than ``osgeo.gdal.GetCacheMax`` on
    purpose: in some environments rasterio ships its own libgdal, so the two
    caches are different objects and only rasterio reads honour this cap.
    """
    from spatialrisk.parallel import SCAN_CACHEMAX_BYTES, scan_env

    with scan_env():
        assert rasterio.env.getenv()["GDAL_CACHEMAX"] == SCAN_CACHEMAX_BYTES
