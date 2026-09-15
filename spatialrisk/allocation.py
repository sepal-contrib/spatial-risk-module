"""Allocate expected jurisdictional deforestation to a project area.

Port of the deforisk plugin's allocation step (``forestatrisk.allocate_deforestation``),
with the upstream positional-index bug fixed: the per-class deforestation density is
looked up by the ``cat`` VALUE, not by row position, so sparse tables (JNR drops
``nfor == 0`` rows) map correctly.  Mirrors
``notebooks_legacy/7.deforestation_allocation.ipynb``.

Solara-free and gui-free by contract: only numeric/geo dependencies, lazily imported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from spatialrisk.raster_profile import gdal_creation_options

logger = logging.getLogger("spatial_risk")

PathLike = Union[str, Path]

#: Nodata written into the deforestation-density raster (Float64).
DENSITY_NODATA = -9999.0

#: Required columns of a deforestation-rate-per-category table.
REQUIRED_COLUMNS = ("cat", "nfor", "rate_mod", "pixel_area")

#: Extents the optional density raster can be written for.
DENSITY_EXTENTS = ("project", "aoi")


class AllocationInputError(ValueError):
    """Raised when an allocation input is missing, malformed or inconsistent."""


@dataclass
class AllocationResult:
    """Outcome of one allocation run."""

    annual_ha: float
    total_ha: float
    out_dir: Path
    csv_path: Path
    defrate_path: Path
    cropped_riskmap_path: Path
    density_map_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)


def validate_defrate_table(df: pd.DataFrame) -> None:
    """Check a rate table against the allocation contract, or raise."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AllocationInputError(
            f"Rate table is missing required column(s): {', '.join(missing)}. "
            f"Found: {list(df.columns)}"
        )
    cat = df["cat"]
    if not np.issubdtype(cat.dtype, np.number) or (cat % 1 != 0).any():
        raise AllocationInputError("Rate table 'cat' must contain whole numbers.")
    if (cat <= 0).any():
        raise AllocationInputError("Rate table 'cat' values must be strictly positive.")
    if cat.duplicated().any():
        dupes = sorted(cat[cat.duplicated()].unique().tolist())
        raise AllocationInputError(f"Rate table has duplicate 'cat' values: {dupes}")
    for col in ("nfor", "rate_mod", "pixel_area"):
        values = df[col]
        if not np.isfinite(values).all():
            raise AllocationInputError(
                f"Rate table column '{col}' has non-finite values."
            )
        if (values < 0).any():
            raise AllocationInputError(
                f"Rate table column '{col}' has negative values."
            )
    pixel_areas = df["pixel_area"].unique()
    if len(pixel_areas) != 1:
        raise AllocationInputError(
            f"Rate table 'pixel_area' must be constant; found {sorted(pixel_areas)}."
        )
    if pixel_areas[0] <= 0:
        raise AllocationInputError("Rate table 'pixel_area' must be strictly positive.")
    if (df["nfor"] * df["rate_mod"]).sum() <= 0:
        raise AllocationInputError(
            "Rate table denominator sum(nfor * rate_mod) is zero: the table carries "
            "no deforestation risk, so no deforestation can be allocated."
        )


def _normalize_density_extent(value):
    """Map the ``defor_density_map`` argument to None, 'project' or 'aoi'.

    Bools stay accepted: ``True`` meant "whole jurisdiction" before extents
    existed, and callers written against that signature must keep working.
    """
    if value is None or value is False:
        return None
    if value is True:
        return "aoi"
    if value in DENSITY_EXTENTS:
        return value
    raise AllocationInputError(
        "defor_density_map must be None, True, False, 'project' or 'aoi'; "
        f"got {value!r}."
    )


def reproject_vector(src: PathLike, dst: PathLike, dst_crs) -> Path:
    """Reproject a vector file to *dst_crs* (anything geopandas' ``to_crs`` accepts)."""
    import geopandas as gpd

    gdf = gpd.read_file(str(src))
    if gdf.empty:
        raise AllocationInputError(f"Vector file '{src}' has no features.")
    if gdf.crs is None:
        raise AllocationInputError(
            f"Vector file '{src}' has no CRS; cannot reproject it onto the risk map."
        )
    gdf.to_crs(dst_crs).to_file(str(dst))
    return Path(dst)


def _load_table(defrate_table: Union[pd.DataFrame, PathLike]) -> pd.DataFrame:
    if isinstance(defrate_table, pd.DataFrame):
        return defrate_table.copy()
    path = Path(defrate_table)
    if not path.exists():
        raise AllocationInputError(f"Rate table not found: {path}")
    return pd.read_csv(path)


def _raster_crs_and_pixel_area(riskmap_file: PathLike):
    """(osr.SpatialReference, pixel area in ha); raises if not metre-based."""
    from osgeo import gdal, osr

    ds = gdal.Open(str(riskmap_file))
    if ds is None:
        raise AllocationInputError(f"Cannot open risk map: {riskmap_file}")
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    ds = None
    srs = osr.SpatialReference()
    if wkt:
        srs.ImportFromWkt(wkt)
    if not wkt or not srs.IsProjected():
        raise AllocationInputError(
            "The risk map must use a projected, metre-based CRS: allocation converts "
            "pixel counts to hectares. Reproject it (e.g. to the local UTM zone) first."
        )
    unit = srs.GetLinearUnits()
    pixel_area_ha = abs(gt[1] * gt[5]) * (unit**2) / 10000.0
    return srs, pixel_area_ha


def allocate_deforestation(
    riskmap_file: PathLike,
    defrate_table: Union[pd.DataFrame, PathLike],
    defor_juris_ha: float,
    years_forecast: float,
    project_borders: PathLike,
    out_dir: PathLike,
    forest_mask_file: Optional[PathLike] = None,
    defor_density_map: Union[bool, str, None] = None,
    blk_rows: int = 128,
) -> AllocationResult:
    """Allocate *defor_juris_ha* of jurisdictional deforestation to a project area.

    Parameters
    ----------
    riskmap_file : path
        Jurisdictional risk map with integer risk categories (nodata 0).
    defrate_table : DataFrame or path
        Per-category rate table: ``cat, nfor, rate_mod, pixel_area``.
    defor_juris_ha : float
        Expected deforestation over the whole jurisdiction for the forecast
        period, in hectares.
    years_forecast : float
        Length of the forecast period, in years.
    project_borders : path
        Vector file with the project boundary (any CRS; reprojected internally).
    out_dir : path
        Directory for the run's outputs (created if needed).
    forest_mask_file : path, optional
        Binary raster (1 = eligible) aligned with the risk map. When given, only
        eligible pixels receive deforestation.
    defor_density_map : {None, 'project', 'aoi'} or bool
        Also write the deforestation-density raster (ha/pixel/yr, Float64,
        nodata ``-9999``): ``'project'`` covers the cropped project area only,
        ``'aoi'`` the whole jurisdictional risk map (large at high
        resolution). Booleans are the legacy form: ``True`` == ``'aoi'``,
        ``False`` == ``None``.
    blk_rows : int
        Rows per block for the density-map write.
    """
    from osgeo import gdal

    if years_forecast is None or years_forecast <= 0:
        raise AllocationInputError("Forecast length must be greater than zero years.")
    if defor_juris_ha is None or defor_juris_ha < 0:
        raise AllocationInputError(
            "Expected jurisdictional deforestation cannot be negative."
        )

    density_extent = _normalize_density_extent(defor_density_map)

    df_rate = _load_table(defrate_table)
    validate_defrate_table(df_rate)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    copts = ["COMPRESS=DEFLATE", "BIGTIFF=YES"]

    srs, raster_pixel_area = _raster_crs_and_pixel_area(riskmap_file)
    table_pixel_area = float(df_rate["pixel_area"].iloc[0])
    if abs(raster_pixel_area - table_pixel_area) > 0.01 * table_pixel_area:
        raise AllocationInputError(
            f"Risk-map pixel area ({raster_pixel_area:.4f} ha) does not match the rate "
            f"table's pixel_area ({table_pixel_area:.4f} ha): the table belongs to a "
            "different raster."
        )

    # --- crop the risk map to the project borders -------------------------
    borders_repro = out_dir / "project_borders_repro.gpkg"
    reproject_vector(project_borders, borders_repro, srs.ExportToWkt())
    cropped = out_dir / "project_riskmap.tif"
    warped = gdal.Warp(
        str(cropped),
        str(riskmap_file),
        cropToCutline=True,
        warpOptions=["CUTLINE_ALL_TOUCHED=TRUE"],
        cutlineDSName=str(borders_repro),
        creationOptions=copts,
    )
    if warped is None:
        raise AllocationInputError(
            "Could not crop the risk map to the project borders: they do not "
            "intersect the risk map."
        )
    warped = None

    # --- count project pixels per category --------------------------------
    counts = _count_categories(cropped, forest_mask_file, out_dir, copts)
    if counts.empty or counts["counts"].sum() == 0:
        raise AllocationInputError(
            "No eligible risk-map pixels inside the project borders: the borders do "
            "not intersect the risk map (or the mask removes every pixel)."
        )

    # --- densities per category -------------------------------------------
    sum_pi = (df_rate["nfor"] * df_rate["rate_mod"]).sum()
    correction_factor = defor_juris_ha / (table_pixel_area * sum_pi)
    df_rate["rate_abs"] = df_rate["rate_mod"] * correction_factor
    df_rate["defor_dens"] = df_rate["rate_abs"] * table_pixel_area / years_forecast
    defrate_path = out_dir / "defrate_cat_forecast.csv"
    df_rate.to_csv(defrate_path, index=False)

    dens_by_cat = df_rate.set_index("cat")["defor_dens"]
    mapped = counts["cat"].map(dens_by_cat)
    uncovered = counts.loc[mapped.isna(), "counts"].sum()
    total_px = counts["counts"].sum()
    if uncovered:
        share = uncovered / total_px
        if share > 0.999:
            raise AllocationInputError(
                "None of the project's risk classes appear in the rate table: the "
                "table does not belong to this risk map."
            )
        missing = sorted(counts.loc[mapped.isna(), "cat"].tolist())
        msg = (
            f"{int(uncovered)} project pixel(s) ({share:.1%}) carry risk classes "
            f"absent from the rate table ({missing[:10]}...); they were allocated "
            "zero deforestation."
        )
        logger.warning(msg)
        warnings.append(msg)
    mapped = mapped.fillna(0.0)

    annual_ha = float((counts["counts"] * mapped).sum())
    total_ha = float(annual_ha * years_forecast)

    csv_path = out_dir / "defor_project.csv"
    pd.DataFrame(
        {
            "period": ["annual", "entire"],
            "length (yr)": [1, years_forecast],
            "deforestation (ha)": [round(annual_ha, 1), round(total_ha, 1)],
        }
    ).to_csv(csv_path, header=True, index=False)

    density_map_path = None
    if density_extent == "project":
        # project_mask.tif was warped onto the cropped grid by
        # _count_categories, so it is the only mask aligned with `cropped`.
        project_mask = (
            out_dir / "project_mask.tif" if forest_mask_file is not None else None
        )
        density_map_path = _write_density_map(
            cropped, dens_by_cat, out_dir, project_mask, blk_rows
        )
    elif density_extent == "aoi":
        density_map_path = _write_density_map(
            riskmap_file, dens_by_cat, out_dir, forest_mask_file, blk_rows
        )

    logger.info(
        "Allocation complete: %.1f ha/yr, %.1f ha over %s yr",
        annual_ha,
        total_ha,
        years_forecast,
    )
    return AllocationResult(
        annual_ha=annual_ha,
        total_ha=total_ha,
        out_dir=out_dir,
        csv_path=csv_path,
        defrate_path=defrate_path,
        cropped_riskmap_path=cropped,
        density_map_path=density_map_path,
        warnings=warnings,
    )


def _count_categories(cropped, forest_mask_file, out_dir, copts) -> pd.DataFrame:
    """DataFrame(cat, counts) of eligible nonzero pixels in the cropped risk map."""
    from osgeo import gdal

    ds = gdal.Open(str(cropped))
    risk = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    eligible = risk != 0
    if forest_mask_file is not None:
        mask_cropped = out_dir / "project_mask.tif"
        ref = gdal.Open(str(cropped))
        gt, ncol, nrow = ref.GetGeoTransform(), ref.RasterXSize, ref.RasterYSize
        bounds = (gt[0], gt[3] + nrow * gt[5], gt[0] + ncol * gt[1], gt[3])
        ref = None
        gdal.Warp(
            str(mask_cropped),
            str(forest_mask_file),
            outputBounds=bounds,
            width=ncol,
            height=nrow,
            creationOptions=copts,
        )
        mds = gdal.Open(str(mask_cropped))
        mask = mds.GetRasterBand(1).ReadAsArray()
        mds = None
        eligible &= mask == 1
    values, counts = np.unique(risk[eligible], return_counts=True)
    return pd.DataFrame(
        {"cat": values.astype(np.int64), "counts": counts.astype(np.int64)}
    )


def _write_density_map(source_raster, dens_by_cat, out_dir, mask_file, blk_rows):
    """Write the ha/pixel/yr density raster on *source_raster*'s grid, block by block.

    *mask_file* (when given) MUST be grid-aligned with *source_raster*: blocks
    are read from both at the same offsets with no warping.
    """
    from osgeo import gdal

    src = gdal.Open(str(source_raster))
    band = src.GetRasterBand(1)
    ncol, nrow = src.RasterXSize, src.RasterYSize
    out_path = out_dir / "deforestation_density_map.tif"
    if out_path.exists():
        out_path.unlink()
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(
        str(out_path),
        ncol,
        nrow,
        1,
        gdal.GDT_Float64,
        gdal_creation_options("float64"),
    )
    dst.SetGeoTransform(src.GetGeoTransform())
    dst.SetProjection(src.GetProjection())
    dst_band = dst.GetRasterBand(1)
    dst_band.SetNoDataValue(DENSITY_NODATA)

    mask_ds = gdal.Open(str(mask_file)) if mask_file is not None else None
    step = max(1, int(blk_rows))
    for y in range(0, nrow, step):
        rows = min(step, nrow - y)
        risk = band.ReadAsArray(0, y, ncol, rows)
        dens = np.full(risk.shape, DENSITY_NODATA, dtype=np.float64)
        eligible = risk != 0
        if mask_ds is not None:
            eligible &= mask_ds.GetRasterBand(1).ReadAsArray(0, y, ncol, rows) == 1
        if eligible.any():
            values = pd.Series(risk[eligible].ravel()).map(dens_by_cat).fillna(0.0)
            dens[eligible] = values.to_numpy()
        dst_band.WriteArray(dens, 0, y)

    dst_band.FlushCache()
    dst_band.ComputeStatistics(False)
    src = dst = mask_ds = None
    return out_path
