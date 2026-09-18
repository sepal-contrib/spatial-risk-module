"""Concurrent ``download_ee_image`` calls must not trip geedim's single loop.

geedim drives every download through ``geedim.utils.AsyncRunner``, a
process-wide singleton wrapping ONE ``asyncio.Runner``. ``AsyncRunner.run``
only checks whether the *calling* thread has a running loop; two plain worker
threads (one per download action) both call ``run_until_complete`` on the
shared loop and the second one dies with "This event loop is already running".
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import ee

from spatialrisk.gee.ee_raster_export import download_ee_image


def _fake_image():
    """An ``ee.Image`` stand-in whose geedim export blocks on the shared runner."""
    from geedim.utils import AsyncRunner

    image = MagicMock(spec=ee.Image)
    prepared = MagicMock()

    def to_geotiff(**_kw):
        AsyncRunner().run(asyncio.sleep(0.3))

    prepared.gd.toGeoTIFF.side_effect = to_geotiff
    image.gd.prepareForExport.return_value = prepared
    return image


def test_two_threads_downloading_at_once_both_succeed(tmp_path):
    """Two downloads on two threads take turns instead of the second one crashing."""
    from geedim.utils import AsyncRunner

    # The app's first download creates the singleton; only a *second* download
    # started while another runs hits the shared loop. (Two threads creating
    # it simultaneously get two runners — geedim's ``singleton`` has no lock —
    # which would mask the bug.)
    AsyncRunner().run(asyncio.sleep(0))
    errors = []

    def worker(i):
        try:
            download_ee_image(_fake_image(), str(tmp_path / f"{i}.tif"))
        except Exception as exc:  # the assertion below reads the list
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []


def test_wait_notice_fires_only_for_the_download_that_queues(tmp_path):
    """The thread stuck behind the lock reports it; the running one stays silent."""
    from spatialrisk.gee import ee_raster_export
    from spatialrisk.gee.progress import geedim_download_wait

    notices = []
    first_in_geedim = threading.Event()

    def make_image(name):
        image = MagicMock(spec=ee.Image)
        prepared = MagicMock()

        def to_geotiff(**_kw):
            first_in_geedim.set()
            time.sleep(0.3)

        prepared.gd.toGeoTIFF.side_effect = to_geotiff
        image.gd.prepareForExport.return_value = prepared
        return image

    def worker(name):
        with geedim_download_wait(lambda: notices.append(name)):
            ee_raster_export.download_ee_image(
                make_image(name), str(tmp_path / f"{name}.tif")
            )

    first = threading.Thread(target=worker, args=("first",))
    first.start()
    assert first_in_geedim.wait(5)
    second = threading.Thread(target=worker, args=("second",))
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert notices == ["second"]
