"""PMTiles built on local disk, and the job status flow around it.

On SEPAL the sample archive used to be written straight onto the NFS home
(``~/module_results``): tippecanoe's seek-heavy writer took 57-70 s for a
10 000-point sample there and 1.0 s on the container's local disk (measured
2026-09-21 on a c8). The archive is therefore built in a local scratch dir and
moved into place. The sampling job reports "tiling" between the points being
written and the archive being ready, and never leaks a "running" row.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")


# --------------------------------------------------------------------------- #
# local scratch
# --------------------------------------------------------------------------- #
def test_local_scratch_prefers_env_then_var_tmp(tmp_path, monkeypatch):
    """SPATIAL_RISK_LOCAL_SCRATCH wins; else /var/tmp; else the app scratch."""
    from spatialrisk import gdal_env

    monkeypatch.setenv(gdal_env.LOCAL_SCRATCH_ENV, str(tmp_path / "mine"))
    assert gdal_env.local_scratch_dir() == tmp_path / "mine"
    assert (tmp_path / "mine").is_dir()

    monkeypatch.delenv(gdal_env.LOCAL_SCRATCH_ENV)
    monkeypatch.setattr(gdal_env, "_LOCAL_SCRATCH_ROOTS", (tmp_path / "vartmp",))
    got = gdal_env.local_scratch_dir()
    assert got.parent == tmp_path / "vartmp"
    assert got.is_dir()

    # nothing local usable -> the ordinary scratch dir
    monkeypatch.setattr(gdal_env, "_LOCAL_SCRATCH_ROOTS", (tmp_path / "file",))
    (tmp_path / "file").write_text("not a dir")
    monkeypatch.setattr(gdal_env, "scratch_dir", lambda: tmp_path / "fallback")
    assert gdal_env.local_scratch_dir() == tmp_path / "fallback"


def _tiny_gpkg(path):
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"strata": [1, 0, 1]},
        geometry=[Point(i / 10, i / 10) for i in range(3)],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GPKG")
    return path


def test_archive_is_built_in_local_scratch_then_moved(tmp_path, monkeypatch):
    """Both -o and -t point under the local scratch; the result is moved into place."""
    import subprocess

    from spatialrisk import gdal_env, pmtiles_convert

    local = tmp_path / "local"
    local.mkdir()
    monkeypatch.setattr(gdal_env, "local_scratch_dir", lambda: local)
    monkeypatch.setattr(pmtiles_convert, "local_scratch_dir", lambda: local)
    monkeypatch.setattr(
        pmtiles_convert.shutil, "which", lambda _: "/usr/bin/tippecanoe"
    )
    src = _tiny_gpkg(tmp_path / "s.gpkg")
    final = tmp_path / "nfs" / "s.pmtiles"
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        out = cmd[cmd.index("-o") + 1]
        captured["out"] = out
        captured["tmp"] = cmd[cmd.index("-t") + 1]
        with open(out, "wb") as fh:
            fh.write(b"PMTILES")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pmtiles_convert.subprocess, "run", fake_run)
    result = pmtiles_convert.gpkg_to_pmtiles(src, final)

    assert result == final
    assert final.read_bytes() == b"PMTILES"
    assert str(captured["out"]).startswith(str(local))
    assert str(captured["tmp"]).startswith(str(local))
    assert not list(final.parent.glob("*.part")), "partial file left behind"
    assert not list(local.rglob("*.pmtiles")), "scratch archive left behind"


def test_failed_conversion_leaves_no_partial_archive(tmp_path, monkeypatch):
    """A tippecanoe failure raises and the destination stays untouched."""
    import subprocess

    from spatialrisk import pmtiles_convert

    (tmp_path / "l").mkdir()
    monkeypatch.setattr(pmtiles_convert, "local_scratch_dir", lambda: tmp_path / "l")
    monkeypatch.setattr(
        pmtiles_convert.shutil, "which", lambda _: "/usr/bin/tippecanoe"
    )
    src = _tiny_gpkg(tmp_path / "s.gpkg")
    final = tmp_path / "nfs" / "s.pmtiles"

    def fake_run(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr(pmtiles_convert.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        pmtiles_convert.gpkg_to_pmtiles(src, final)
    assert not final.exists()
    assert not list(final.parent.glob("*")) if final.parent.exists() else True


# --------------------------------------------------------------------------- #
# Sample: points first, tiles second
# --------------------------------------------------------------------------- #
def _write_raster(path, array, nodata=255, crs="EPSG:3857"):
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="uint8",
        nodata=nodata,
        crs=crs,
        transform=from_origin(0, array.shape[0], 1, 1),
    ) as dst:
        dst.write(array, 1)


class _Var:
    def __init__(self, path):
        self.path = path


class _StubProject:
    def __init__(self, variables):
        self._vars = variables

    def get_variable(self, name, year=None):
        return self._vars[name]


def _sample(tmp_path, **kw):
    from spatialrisk.sample import Sample

    strata = np.zeros((20, 20), dtype="uint8")
    strata[:, 10:] = 1
    rpath, mpath = tmp_path / "r.tif", tmp_path / "m.tif"
    _write_raster(rpath, strata)
    _write_raster(mpath, np.ones((20, 20), dtype="uint8"))
    project = _StubProject({"target": _Var(rpath), "forest_mask": _Var(mpath)})
    return Sample(
        project=project,
        name="s",
        raster_var_name="target",
        mask_var_name="forest_mask",
        strategy="random",
        n_samples=50,
        seed=1,
        points_path=tmp_path / "s.gpkg",
        **kw,
    )


def test_draw_points_writes_points_and_counts_without_tiles(tmp_path, monkeypatch):
    """draw_points leaves the archive to build_tiles, which then fills pmtiles_path."""
    from spatialrisk import pmtiles_convert

    calls = []
    monkeypatch.setattr(pmtiles_convert, "tippecanoe_available", lambda: True)
    monkeypatch.setattr(
        pmtiles_convert, "gpkg_to_pmtiles", lambda s, o: calls.append(o) or o
    )
    sample = _sample(tmp_path)
    sample.draw_points()
    assert (tmp_path / "s.gpkg").exists()
    assert sample.n_total == 50
    assert sum(sample.class_counts.values()) == 50
    assert sample.created_at
    assert sample.pmtiles_path is None
    assert calls == []

    sample.build_tiles()
    assert calls == [tmp_path / "s.pmtiles"]
    assert sample.pmtiles_path == tmp_path / "s.pmtiles"


def test_generate_is_draw_then_build(tmp_path, monkeypatch):
    """generate() is the two phases in order."""
    from spatialrisk import pmtiles_convert

    order = []
    monkeypatch.setattr(pmtiles_convert, "tippecanoe_available", lambda: True)
    monkeypatch.setattr(
        pmtiles_convert, "gpkg_to_pmtiles", lambda s, o: order.append("tiles") or o
    )
    from spatialrisk.sample import Sample

    sample = _sample(tmp_path)
    orig = Sample.draw_points

    def spy(self):
        order.append("points")
        return orig(self)

    monkeypatch.setattr(Sample, "draw_points", spy)
    sample.generate()
    assert order == ["points", "tiles"]


def test_build_tiles_failure_is_nonfatal(tmp_path, monkeypatch):
    """A tippecanoe crash leaves the points intact and pmtiles_path unset."""
    from spatialrisk import pmtiles_convert

    monkeypatch.setattr(pmtiles_convert, "tippecanoe_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("tippecanoe blew up")

    monkeypatch.setattr(pmtiles_convert, "gpkg_to_pmtiles", boom)
    sample = _sample(tmp_path)
    sample.draw_points()
    sample.build_tiles()  # must not raise
    assert sample.pmtiles_path is None
    assert sample.n_total == 50


# --------------------------------------------------------------------------- #
# job status flow in the tile
# --------------------------------------------------------------------------- #
def _fresh_jobs(st, job):
    st.sampling_jobs.set([job])
    return st.sampling_jobs


def _stub_sample(monkeypatch, st, seen):
    """Replace Sample with a stub recording the order of the two phases."""
    from spatialrisk import sample as sample_mod

    class FakeSample:
        def __init__(self, **kw):
            self.name = kw["name"]
            self.n_total = 0
            self.class_counts = {}
            self.pmtiles_path = None

        def draw_points(self):
            self.n_total = 7
            self.class_counts = {"0": 3, "1": 4}
            seen.append(("draw", [j["status"] for j in st.sampling_jobs.value]))
            return self

        def build_tiles(self):
            seen.append(("tiles", [j["status"] for j in st.sampling_jobs.value]))
            self.pmtiles_path = "x.pmtiles"
            return self

    monkeypatch.setattr(sample_mod, "Sample", FakeSample)


def _project_reactive(registered, folder):
    import solara

    class P:
        project_name = "p"
        folders = SimpleNamespace(samples_folder=folder)

        def add_sample(self, sample, auto_save=True):
            registered.append(sample.name)

        def model_copy(self):
            return self

    return solara.reactive(P())


def test_job_goes_running_tiling_completed(monkeypatch, tmp_path):
    """The job row passes through tiling, with counts, before completing."""
    from gui.tile import sampling_tile as st

    seen, registered = [], []
    _stub_sample(monkeypatch, st, seen)
    monkeypatch.setattr(st, "publish_if_current", lambda *a, **k: None)
    jobs = _fresh_jobs(st, {"id": "j1", "name": "s", "status": "running"})
    st._run_sampling(
        "j1",
        "s",
        "r",
        None,
        "random",
        None,
        False,
        7,
        None,
        1,
        _project_reactive(registered, tmp_path),
    )
    assert seen == [("draw", ["running"]), ("tiles", ["tiling"])]
    job = jobs.value[0]
    assert job["status"] == "completed"
    assert job["n_total"] == 7 and job["class_counts"] == {"0": 3, "1": 4}
    assert registered == ["s"]
    st.sampling_jobs.set([])


def test_tiling_status_carries_the_counts_before_tiles_exist(monkeypatch, tmp_path):
    """The row can show the point counts while tippecanoe is still running."""
    from gui.tile import sampling_tile as st

    snapshots, registered = [], []
    _stub_sample(monkeypatch, st, snapshots)
    monkeypatch.setattr(st, "publish_if_current", lambda *a, **k: None)
    counts_during_tiling = {}

    from spatialrisk import sample as sample_mod

    orig_build = sample_mod.Sample.build_tiles

    def build(self):
        counts_during_tiling.update(st.sampling_jobs.value[0])
        return orig_build(self)

    monkeypatch.setattr(sample_mod.Sample, "build_tiles", build)
    _fresh_jobs(st, {"id": "j1", "name": "s", "status": "running"})
    st._run_sampling(
        "j1",
        "s",
        "r",
        None,
        "random",
        None,
        False,
        7,
        None,
        1,
        _project_reactive(registered, tmp_path),
    )
    assert counts_during_tiling["status"] == "tiling"
    assert counts_during_tiling["n_total"] == 7
    st.sampling_jobs.set([])


def test_closed_project_marks_the_job_cancelled_not_running(monkeypatch):
    """A queued job whose project vanished is cancelled, never left running."""
    import solara

    from gui.tile import sampling_tile as st

    jobs = _fresh_jobs(st, {"id": "j1", "name": "s", "status": "running"})
    st._run_sampling(
        "j1", "s", "r", None, "random", None, False, 7, None, 1, solara.reactive(None)
    )
    assert jobs.value[0]["status"] == "cancelled"
    st.sampling_jobs.set([])


def test_update_job_is_atomic_under_concurrent_finishes():
    """Two workers finishing at once must both land their status update."""
    import solara

    from gui.scripts.solara_threads import update_job

    jobs = solara.reactive(
        [{"id": "a", "status": "running"}, {"id": "b", "status": "running"}]
    )
    # Slow down the read-modify-write window so an unlocked update would race.
    real_set = jobs.set

    def slow_set(value):
        time.sleep(0.01)
        real_set(value)

    jobs.set = slow_set
    start = threading.Barrier(2)

    def finish(job_id):
        start.wait()
        update_job(jobs, job_id, status="completed")

    ts = [threading.Thread(target=finish, args=(i,)) for i in ("a", "b")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert [j["status"] for j in jobs.value] == ["completed", "completed"]


# --------------------------------------------------------------------------- #
# rows and labels
# --------------------------------------------------------------------------- #
def test_tiling_job_row_is_active_and_shows_counts():
    """A tiling job is still active and its row carries the point counts."""
    from gui.scripts.product_rows import ACTIVE_SAMPLING_STATUSES, sample_rows

    assert (
        "tiling" in ACTIVE_SAMPLING_STATUSES and "running" in ACTIVE_SAMPLING_STATUSES
    )
    jobs = [
        {
            "id": "j1",
            "name": "s",
            "strategy": "random",
            "status": "tiling",
            "error": None,
            "n_total": 7,
            "class_counts": {"0": 3, "1": 4},
        }
    ]
    rows = sample_rows(SimpleNamespace(samples={}), jobs)
    assert rows[0]["status"] == "tiling" and rows[0]["n_total"] == 7


def test_tiling_status_has_colour_icon_and_label_in_every_locale():
    """The new status renders like running and has a label in every locale."""
    import json
    from pathlib import Path

    from gui.widget.product_table import STATUS_COLORS, STATUS_ICONS

    assert STATUS_COLORS["tiling"] == "info"
    assert "mdi-spin" in STATUS_ICONS["tiling"]
    for locale_dir in (Path("gui") / "messages").iterdir():
        data = json.loads((locale_dir / "widgets.json").read_text(encoding="utf-8"))
        assert data["widgets"]["product_table"]["status_tiling"], locale_dir.name
