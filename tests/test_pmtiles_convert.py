"""Tests for the GPKG -> PMTiles conversion helper."""
import subprocess
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")


def _tiny_gpkg(path):
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"strata": [1, 0, 1]},
        geometry=[Point(i / 10, i / 10) for i in range(3)],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GPKG")
    return path


def test_available_false_raises(tmp_path, monkeypatch):
    """Without the binary the conversion refuses to run."""
    from spatialrisk import pmtiles_convert

    src = _tiny_gpkg(tmp_path / "a.gpkg")
    monkeypatch.setattr(pmtiles_convert.shutil, "which", lambda _: None)
    # no sibling binary next to the interpreter either
    (tmp_path / "bin").mkdir()
    monkeypatch.setattr(
        pmtiles_convert.sys, "executable", str(tmp_path / "bin" / "python")
    )
    with pytest.raises(RuntimeError):
        pmtiles_convert.gpkg_to_pmtiles(src, tmp_path / "a.pmtiles")


def test_resolver_falls_back_to_interpreter_sibling(tmp_path, monkeypatch):
    """A kernel PATH without the env bin still finds tippecanoe next to python.

    SEPAL's jupyter kernels run the env interpreter without putting its bin/
    on PATH, so shutil.which misses a perfectly installed binary.
    """
    from spatialrisk import pmtiles_convert

    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "tippecanoe"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(pmtiles_convert.shutil, "which", lambda _: None)
    monkeypatch.setattr(pmtiles_convert.sys, "executable", str(bindir / "python"))

    assert pmtiles_convert.tippecanoe_available() is True
    assert pmtiles_convert.resolve_tippecanoe() == str(fake)


def test_conversion_runs_the_resolved_binary(tmp_path, monkeypatch):
    """The subprocess must run the resolved path, not rely on PATH again."""
    import subprocess as sp

    from spatialrisk import pmtiles_convert

    src = _tiny_gpkg(tmp_path / "s.gpkg")
    out = tmp_path / "s.pmtiles"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "tippecanoe"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(pmtiles_convert.shutil, "which", lambda _: None)
    monkeypatch.setattr(pmtiles_convert.sys, "executable", str(bindir / "python"))
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        # tippecanoe writes the archive where -o points (local scratch)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"PMTILES")
        return sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pmtiles_convert.subprocess, "run", fake_run)
    pmtiles_convert.gpkg_to_pmtiles(src, out)
    assert captured["cmd"][0] == str(fake)


def test_builds_expected_command(tmp_path, monkeypatch):
    """The tippecanoe invocation keeps its point-retention flags."""
    from spatialrisk import pmtiles_convert

    src = _tiny_gpkg(tmp_path / "s.gpkg")
    out = tmp_path / "s.pmtiles"
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        # pretend tippecanoe wrote the archive where -o points (local scratch)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"PMTILES")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        pmtiles_convert.shutil, "which", lambda _: "/usr/bin/tippecanoe"
    )
    monkeypatch.setattr(pmtiles_convert.subprocess, "run", fake_run)
    result = pmtiles_convert.gpkg_to_pmtiles(src, out, layer="points", max_zoom=12)

    assert result == out
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/tippecanoe"
    built = Path(cmd[cmd.index("-o") + 1])
    assert built.name == out.name and built != out  # assembled in scratch, then moved
    assert "-t" in cmd
    assert "-l" in cmd and "points" in cmd
    assert "-z" in cmd and "12" in cmd
    assert "-Z" in cmd and "0" in cmd
    assert "-r1" in cmd
    assert "--drop-densest-as-needed" in cmd


@pytest.mark.skipif(
    __import__("shutil").which("tippecanoe") is None, reason="tippecanoe not installed"
)
def test_real_conversion_writes_pmtiles(tmp_path):
    """End-to-end conversion produces a non-empty archive."""
    from spatialrisk.pmtiles_convert import gpkg_to_pmtiles

    src = _tiny_gpkg(tmp_path / "s.gpkg")
    out = gpkg_to_pmtiles(src, tmp_path / "s.pmtiles")
    assert out.exists() and out.stat().st_size > 0
