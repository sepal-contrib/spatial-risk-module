"""The app gives GDAL half the cores, whatever the map's tile server sets.

localtileserver's ``create_app`` does
``os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")``, so before the first
map layer GDAL ran single-threaded and after it on every core: the same
harmonization write took 28.5 s or 9.5 s depending on whether a layer had been
drawn. Pyramid builds and processing writes gain from threads (BOL slope
pyramid 82 s at 1 thread, 39 s at 4); tile drawing does not (12 tiles 1.1-1.2 s
at 1, 2 and 8). Half the cores is the app's policy for every raster scan
(:func:`spatialrisk.parallel.worker_threads`), leaving the rest to the tile
server and the Solara UI.
"""

import os

import pytest
from localtileserver.web.fastapi_app import create_app

import spatialrisk.gdal_env as ge
from spatialrisk.parallel import NUM_THREADS_ENV

THREADS = "GDAL_NUM_THREADS"
#: Everything that decides the pinned value, restored after each test.
ENV = (THREADS, NUM_THREADS_ENV, "GDAL_DISABLE_READDIR_ON_OPEN")


@pytest.fixture(autouse=True)
def clean_thread_env():
    """Start each test with the thread vars unset; put the originals back.

    Not ``monkeypatch.delenv``: on a var that is not set it records nothing,
    so whatever ``create_app`` then sets would leak into later test files.
    """
    saved = {k: os.environ.get(k) for k in ENV}
    for k in ENV:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def eight_cores(monkeypatch):
    """Pretend the instance has eight cores (a SEPAL c8)."""
    monkeypatch.setattr(ge, "_available_cores", lambda: 8)


def test_tile_server_setup_takes_every_core():
    """Why the pin exists: the tile server's setup writes ALL_CPUS to os.environ."""
    create_app()

    assert os.environ[THREADS] == "ALL_CPUS"


def test_pin_gives_gdal_half_the_cores(eight_cores):
    """Eight cores: four GDAL threads."""
    ge.pin_gdal_num_threads()

    assert os.environ[THREADS] == "4"


def test_pin_replaces_the_tile_server_value(eight_cores):
    """Run after the tile server started, the pin still wins."""
    create_app()

    ge.pin_gdal_num_threads()

    assert os.environ[THREADS] == "4"


def test_pin_holds_when_it_runs_before_the_tile_server(eight_cores):
    """Startup order does not matter: the server's setdefault cannot undo it."""
    ge.pin_gdal_num_threads()
    create_app()

    assert os.environ[THREADS] == "4"


def test_one_core_still_gets_one_thread(monkeypatch):
    """Never zero threads, which GDAL would not accept."""
    monkeypatch.setattr(ge, "_available_cores", lambda: 1)

    ge.pin_gdal_num_threads()

    assert os.environ[THREADS] == "1"


def test_app_thread_override_is_honoured(eight_cores):
    """SPATIAL_RISK_NUM_THREADS sets GDAL's count too, like every other scan."""
    os.environ[NUM_THREADS_ENV] = "3"

    ge.pin_gdal_num_threads()

    assert os.environ[THREADS] == "3"


def test_app_startup_pins_gdal_threads():
    """Importing the app (as voila and ``solara run`` do) installs the pin."""
    import subprocess
    import sys
    from pathlib import Path

    env = {**os.environ, THREADS: "ALL_CPUS", NUM_THREADS_ENV: "3"}
    out = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            f"import os, gui.solara_app; print(os.environ[{THREADS!r}])",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )

    assert out.stdout.strip().splitlines()[-1] == "3"
