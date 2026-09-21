"""Convert a points GPKG to PMTiles via tippecanoe (Solara-free domain helper).

tippecanoe reads GeoJSON, not GPKG, so we reproject to WGS84 and export a
temporary GeoJSON first. Kept thin and mockable so the eager-conversion step in
``Sample.generate`` can be tested without the binary.

The archive is assembled on **local disk** and moved into place afterwards.
tippecanoe's PMTiles writer seeks and rewrites constantly, and on SEPAL the
sample folder (``~/module_results``) and ``/tmp`` are both NFS: a 10 000-point
sample took 57-70 s to tile there and 1.0 s on the container's local disk
(c8 sandbox, 2026-09-21). The final copy is one sequential write.
"""
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from spatialrisk.gdal_env import local_scratch_dir

logger = logging.getLogger("spatial_risk")


def resolve_tippecanoe() -> Optional[str]:
    """Absolute path to the ``tippecanoe`` binary, or None when absent.

    PATH first, then the file next to ``sys.executable`` — SEPAL's jupyter
    kernels run the env's interpreter without putting its ``bin/`` on PATH,
    so a perfectly installed conda binary is invisible to ``shutil.which``
    (same fallback vectortileserver uses).
    """
    found = shutil.which("tippecanoe")
    if found:
        return found
    sibling = Path(sys.executable).parent / "tippecanoe"
    return str(sibling) if sibling.exists() else None


def tippecanoe_available() -> bool:
    """True if the ``tippecanoe`` binary can be resolved."""
    return resolve_tippecanoe() is not None


def gpkg_to_pmtiles(
    gpkg_path, out_path, *, layer="points", min_zoom=0, max_zoom=14
) -> Path:
    """Convert ``gpkg_path`` to a ``.pmtiles`` archive at ``out_path``.

    ``layer`` is the tippecanoe layer name (the MapLibre ``source-layer``).
    Raises ``RuntimeError`` if tippecanoe is missing, ``CalledProcessError`` if
    it fails. ``-r1`` disables tippecanoe's default rate-based point dropping
    (2.5x per zoom below maxzoom) so every point renders at every zoom;
    ``--drop-densest-as-needed`` stays as a guard that only thins tiles
    exceeding the 500KB limit, so million-point samples still tile.
    """
    import geopandas as gpd

    tippecanoe = resolve_tippecanoe()
    if tippecanoe is None:
        raise RuntimeError("tippecanoe not found on PATH or next to the interpreter")

    gpkg_path, out_path = Path(gpkg_path), Path(out_path)
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Everything tippecanoe touches -- its input, its temp tiles (-t) and the
    # archive it assembles (-o) -- stays on local disk; see the module docstring.
    # Pinned dir: tempfile's candidate list ends with the CWD, which on SEPAL is
    # the read-only module mount.
    with tempfile.TemporaryDirectory(dir=local_scratch_dir()) as td:
        geojson = Path(td) / "points.geojson"
        gdf.to_file(geojson, driver="GeoJSON")
        built = Path(td) / out_path.name
        cmd = [
            tippecanoe,
            "-o",
            str(built),
            "-t",
            td,
            "-l",
            layer,
            "-Z",
            str(min_zoom),
            "-z",
            str(max_zoom),
            "-r1",
            "--drop-densest-as-needed",
            "--force",
            str(geojson),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Copy next to the destination, then rename: readers never see a
        # half-written archive, and a failure leaves no ``.part`` behind.
        partial = out_path.with_name(out_path.name + ".part")
        try:
            shutil.copyfile(built, partial)
            partial.replace(out_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    logger.info("PMTiles written: %s", out_path)
    return out_path
