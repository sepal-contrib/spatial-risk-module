"""Native deforestation-rate computations from two explicit layers.

These functions replace the ``riskmapjnr`` (rmj) routines that require a packed
multi-temporal ``fcc123`` raster plus period/value-specific branching. Instead
they take TWO explicit, period-self-contained binary rasters:

    defor_file  : deforestation during the period.
                  ``1`` = deforested, anything else (incl. nodata) = not.
    forest_file : forest extent at the START of the period (the population at
                  risk).  ``1`` = forest, anything else (incl. nodata) = not.

Period- and value-agnostic semantics (no ``defor_values``, no ``period`` string,
no ``fcc123`` convention, no ``check_fcc``):

    numerator   (deforested)        = (defor  == 1)
    denominator (forest at start)   = (forest == 1) | (defor == 1)

The union makes the denominator robust to whether ``forest_file`` encodes forest
at the *start* or the *end* of the period, since
``forest_at_start = forest_remaining UNION deforested``.

Numerics (``rescale``, block iteration, the moving-window ``uniform_filter``) are
inherited verbatim from ``riskmapjnr.misc`` / SciPy, so results are identical to
the rmj routines when the inputs are equivalent (see ``tests`` / the equivalence
check in the single-layer migration plan).
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.ndimage
from osgeo import gdal

# Reuse rmj's low-level helpers so the numerics match bit-for-bit. These are
# period/value-agnostic (pure raster utilities), unlike the high-level routines.
from riskmapjnr.misc import makeblock, progress_bar, rescale

from spatialrisk.raster_profile import gdal_creation_options

PathLike = Union[str, "os.PathLike[str]"]


def _open_band(path: PathLike):
    """Open a single-band raster, returning (dataset, band)."""
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open raster: {path}")
    return ds, ds.GetRasterBand(1)


def validate_binary_defor(
    defor_file: PathLike, *, layer_name: Optional[str] = None
) -> None:
    """Check that a forest-loss raster really is binary before it is consumed.

    The functions in this module treat ``defor == 1`` as the event and *anything
    else* — including nodata — as "not deforested". A categorical or continuous
    layer therefore never raises: it trains silently on a wrong numerator. This
    guard makes that case loud.

    Exact min/max is enough to catch every realistic wrong-layer case (a
    multi-period categorical stack, a tree-cover percentage, a rescaled
    0-65535 layer) in a single nodata-aware C-level pass.

    Parameters
    ----------
    defor_file : path
        The binary deforestation raster to check.
    layer_name : str, optional
        Name to quote in error messages. Defaults to the file name.

    Raises:
    ------
    FileNotFoundError
        If the raster cannot be opened.
    ValueError
        If any value falls outside ``{0, 1}``, or the layer has no ``1`` pixels.
    """
    name = layer_name or os.path.basename(str(defor_file))
    try:
        defor_ds, defor_band = _open_band(defor_file)
    except RuntimeError as exc:
        # GDAL exceptions are enabled process-wide, so a missing or unreadable
        # file surfaces as RuntimeError rather than the None _open_band checks.
        raise FileNotFoundError(f"Cannot open raster: {defor_file}") from exc
    try:
        vmin, vmax = defor_band.ComputeRasterMinMax(False)
    finally:
        del defor_band, defor_ds

    if vmin < 0 or vmax > 1:
        raise ValueError(
            f"The forest-loss layer '{name}' is not binary: its values range "
            f"from {vmin:g} to {vmax:g}, but only 0 and 1 are allowed "
            f"(1 = deforested, 0 = not).\n"
            f"  Values other than 1 are silently counted as 'not deforested', "
            f"so training on this layer would give a wrong result rather than "
            f"an error.\n"
            f"  Common causes: a multi-period categorical layer (e.g. 1/2/3 "
            f"per period), a percentage or continuous layer, or a fill value "
            f"such as 255 that is not declared as the raster's nodata value."
        )

    if vmax == 0:
        raise ValueError(
            f"The forest-loss layer '{name}' has no deforested pixels: no "
            f"pixel holds the value 1, so there is nothing to train on.\n"
            f"  Check that the right layer and period were selected."
        )


def dist_edge_threshold(
    defor_file: PathLike,
    dist_file: PathLike,
    dist_bins,
    defor_threshold: float = 99.5,
    tab_file_dist: Optional[PathLike] = None,
    fig_file_dist: Optional[PathLike] = None,
    figsize=(6.4, 4.8),
    dpi: int = 100,
    blk_rows: int = 128,
    verbose: bool = False,
) -> dict:
    """Distance-to-edge threshold capturing ``defor_threshold`` % of deforestation.

    Equivalent to ``riskmapjnr.dist_edge_threshold`` but driven by a single
    binary deforestation layer (``defor == 1``) instead of an fcc raster +
    ``defor_values``. The forest layer is not needed here (only deforested
    pixels and their distance-to-edge enter the calculation).

    Parameters
    ----------
    defor_file : path
        Binary deforestation raster (``1`` = deforested).
    dist_file : path
        Distance-to-forest-edge raster (metres) for the period's initial year.
    dist_bins : array-like
        Right-closed distance bin edges (e.g. ``np.arange(0, 5000, 30)``).
    defor_threshold : float
        Cumulative percentage of deforestation defining the threshold (default 99.5).
    tab_file_dist : path or None
        CSV output with the cumulative distribution. Defaults to ``None``, i.e.
        no CSV is written. A bare filename default would resolve against the
        process CWD, which is read-only when the app runs on SEPAL, so callers
        that want the table must pass an explicit path.
    fig_file_dist : path or None
        PNG plot output. Skipped if ``None`` (the default).

    Returns:
    -------
    dict
        ``{"tot_def", "dist_thresh", "perc_thresh"}`` (matches rmj).
    """
    defor_ds, defor_band = _open_band(defor_file)
    dist_ds, dist_band = _open_band(dist_file)

    res_df = pd.DataFrame(
        {"distance": dist_bins[1:], "npix": 0, "area": 0.0, "cum": 0.0, "perc": 0.0}
    )
    npix_def = 0

    nblock, nblock_x, _, x, y, nx, ny = makeblock(str(dist_file), blk_rows=blk_rows)[:7]
    for b in range(nblock):
        if verbose:
            progress_bar(nblock, b + 1)
        px, py = b % nblock_x, b // nblock_x
        dist_data = dist_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        defor_data = defor_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        defor_mask = defor_data == 1
        # Number of deforested pixels
        npix_def += int(defor_mask.sum())
        # Consider only deforested pixels for distances
        dist_def = dist_data * defor_mask
        dist_def = dist_def[dist_def > 0]
        # Categorize distance and count per bin
        dist_cat = pd.cut(dist_def.flatten(), dist_bins, right=True)
        counts = pd.DataFrame({"dist": dist_cat}).groupby("dist", observed=False).size()
        res_df.loc[:, "npix"] += counts.values

    # Areas (ha) and cumulative percentage of total deforestation
    gt = dist_ds.GetGeoTransform()
    pix_area = gt[1] * (-gt[5])
    res_df.loc[:, "area"] = res_df["npix"].values * pix_area / 10000
    tot_area_def = npix_def * pix_area / 10000
    res_df.loc[:, "cum"] = res_df["area"].cumsum().values
    res_df.loc[:, "perc"] = 100 * res_df["cum"].values / tot_area_def

    if tab_file_dist is not None:
        res_df.to_csv(str(tab_file_dist), sep=",", header=True, index=False)

    try:
        index_thresh = np.nonzero(res_df["perc"].values > defor_threshold)[0][0]
    except IndexError as exc:
        raise ValueError(
            "Increase maximal distance defined in argument 'dist_bins'."
        ) from exc
    dist_thresh = res_df.loc[index_thresh, "distance"]
    perc_thresh = np.around(res_df.loc[index_thresh, "perc"], 2)

    if fig_file_dist is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.subplot(111)
        plt.plot(res_df["distance"], res_df["perc"], "b-")
        plt.vlines(
            dist_thresh,
            ymin=np.min(res_df["perc"]),
            ymax=perc_thresh,
            colors="k",
            linestyles="dashed",
        )
        plt.hlines(
            perc_thresh,
            xmin=0,
            xmax=dist_thresh,
            colors="k",
            linestyles="dashed",
        )
        plt.xlabel("Distance to forest edge (m)")
        plt.ylabel("Percentage of total deforestation (%)")
        fig.savefig(str(fig_file_dist))
        plt.close(fig)

    del defor_ds, dist_ds
    return {
        "tot_def": tot_area_def,
        "dist_thresh": dist_thresh,
        "perc_thresh": perc_thresh,
    }


def local_defor_rate(
    defor_file: PathLike,
    forest_file: PathLike,
    ldefrate_file: PathLike,
    win_size: int,
    time_interval: float,
    rescale_min_val: int = 2,
    rescale_max_val: int = 65535,
    blk_rows: int = 128,
    verbose: bool = False,
) -> None:
    """Local deforestation rate in a moving window, from two explicit layers.

    Equivalent to ``riskmapjnr.local_defor_rate`` with
    ``numerator = (defor == 1)`` and
    ``denominator = (forest == 1) | (defor == 1)`` (forest at period start).
    The annual rate ``theta = 1 - (1 - n/d) ** (1/time_interval)`` is rescaled to
    ``[rescale_min_val, rescale_max_val]`` (UInt16, nodata 0).

    Parameters
    ----------
    defor_file, forest_file : path
        Binary deforestation / forest-at-start rasters (``== 1``).
    ldefrate_file : path
        Output UInt16 raster (nodata 0).
    win_size : int
        Odd moving-window size in pixels, ``<= blk_rows``.
    time_interval : float
        Period length in years.
    """
    win_size = int(win_size)
    if (win_size % 2) == 0:
        raise ValueError("'win_size' must be an odd number.")
    if win_size > blk_rows:
        raise ValueError("'win_size' must be lower or equal to 'blk_rows'.")

    defor_ds, defor_band = _open_band(defor_file)
    forest_ds, forest_band = _open_band(forest_file)
    xsize = defor_band.XSize
    ysize = defor_band.YSize
    if forest_band.XSize != xsize or forest_band.YSize != ysize:
        raise ValueError("'defor_file' and 'forest_file' must share the same grid.")

    driver = gdal.GetDriverByName("GTiff")
    if os.path.isfile(str(ldefrate_file)):
        os.remove(str(ldefrate_file))
    out_ds = driver.Create(
        str(ldefrate_file),
        xsize,
        ysize,
        1,
        gdal.GDT_UInt16,
        gdal_creation_options("uint16"),
    )
    out_ds.SetProjection(defor_ds.GetProjection())
    out_ds.SetGeoTransform(defor_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)

    iter_block = 0
    for i in range(0, ysize, blk_rows):
        nblock = (ysize // blk_rows) + 1
        iter_block += 1
        if verbose:
            progress_bar(nblock, iter_block)

        extra_lines = win_size // 2
        if (i + blk_rows + 2 * extra_lines - 1) < ysize:
            rows = blk_rows + 2 * extra_lines
        else:
            rows = ysize - i + extra_lines
        yoff = max(0, i - extra_lines)

        defor_arr = defor_band.ReadAsArray(0, yoff, xsize, rows)
        forest_arr = forest_band.ReadAsArray(0, yoff, xsize, rows)

        defor_mask = defor_arr == 1
        # Forest at start of the period (population at risk). Union with the
        # deforested pixels reconstructs it whether forest_file is start or end.
        forest_start = (forest_arr == 1) | defor_mask

        # Windowed count of deforested pixels
        defor_data = defor_mask.astype(int)
        win_defor = scipy.ndimage.uniform_filter(
            defor_data, size=win_size, mode="constant", cval=0, output=float
        ) * (win_size**2)
        win_defor = np.rint(win_defor).astype(int)

        # Windowed count of forest-at-start pixels
        for_data = forest_start.astype(int)
        w = np.where(for_data > 0)
        win_for = scipy.ndimage.uniform_filter(
            for_data, size=win_size, mode="constant", cval=0, output=float
        ) * (win_size**2)
        win_for = np.rint(win_for).astype(int)

        # Annual deforestation rate, rescaled
        out_data = np.zeros(defor_arr.shape, int)
        theta = 1 - (1 - win_defor[w] / win_for[w]) ** (1 / time_interval)
        out_data[w] = rescale(theta, rescale_min_val, rescale_max_val)

        if yoff == 0:
            out_band.WriteArray(out_data)
        else:
            out_band.WriteArray(out_data[extra_lines:], 0, yoff + extra_lines)

    out_band.FlushCache()
    cb = gdal.TermProgress_nocb if verbose else 0
    out_band.ComputeStatistics(False, cb)
    del out_ds, defor_ds, forest_ds


def defrate_per_cat(
    defor_file: PathLike,
    forest_file: PathLike,
    riskmap_file: PathLike,
    time_interval: float,
    tab_file_defrate: Optional[PathLike] = None,
    blk_rows: int = 128,
    verbose: bool = False,
) -> pd.DataFrame:
    """Observed deforestation rate per risk category, from two explicit layers.

    Equivalent to ``riskmapjnr.defrate_per_cat`` (and ``benchmark.defrate_per_class``)
    but with the hard-coded ``period`` branch replaced by explicit layers:
    ``nfor`` counts ``forest-at-start`` pixels per category and ``ndefor`` counts
    ``deforested`` pixels per category. Works for any period and for both the MW
    category map and the JNR vulnerability-class map.

    Parameters
    ----------
    defor_file, forest_file : path
        Binary deforestation / forest-at-start rasters (``== 1``).
    riskmap_file : path
        Categorical risk/vulnerability raster (UInt16 categories 1..65535).
    time_interval : float
        Period length in years.
    tab_file_defrate : path or None
        CSV output. Defaults to ``None``, i.e. no CSV is written and the table
        is only returned. A bare filename default would resolve against the
        process CWD, which is read-only when the app runs on SEPAL, so callers
        that want the file must pass an explicit path.

    Returns:
    -------
    pandas.DataFrame
        Per-category table (cat, nfor, ndefor, rate_obs, rate_mod, rate_abs, ...),
        also written to ``tab_file_defrate`` if provided.
    """
    defor_ds, defor_band = _open_band(defor_file)
    forest_ds, forest_band = _open_band(forest_file)
    cat_ds, cat_band = _open_band(riskmap_file)

    gt = defor_ds.GetGeoTransform()
    xres = gt[1]
    yres = -gt[5]

    nblock, nblock_x, _, x, y, nx, ny = makeblock(str(defor_file), blk_rows=blk_rows)[
        :7
    ]

    n_cat = 65535
    cat = [c + 1 for c in range(n_cat)]
    df = pd.DataFrame({"cat": cat, "nfor": 0, "ndefor": 0})

    for b in range(nblock):
        if verbose:
            progress_bar(nblock, b + 1)
        px, py = b % nblock_x, b // nblock_x
        defor_arr = defor_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        forest_arr = forest_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        cat_arr = cat_band.ReadAsArray(x[px], y[py], nx[px], ny[py])

        defor_mask = defor_arr == 1
        forest_start = (forest_arr == 1) | defor_mask

        data_for = cat_arr[forest_start]
        data_defor = cat_arr[defor_mask]

        cat_for = pd.Categorical(data_for.flatten(), categories=cat)
        df["nfor"] += cat_for.value_counts().values
        cat_defor = pd.Categorical(data_defor.flatten(), categories=cat)
        df["ndefor"] += cat_defor.value_counts().values

    # Observed annual deforestation rate per category
    df["time_interval"] = time_interval
    df["rate_obs"] = 1 - (1 - df["ndefor"] / df["nfor"]) ** (1 / time_interval)

    # Relative spatial deforestation probability from model (per category)
    df["rate_mod"] = ((df["cat"] - 2) * 999999 / 65533 + 1) * 1e-6
    df.loc[df["cat"] == 1, "rate_mod"] = 0

    # Correction factor → absolute deforestation probability
    sum_ndefor = df["ndefor"].sum()
    sum_pi = (df["nfor"] * df["rate_mod"]).sum()
    correction_factor = sum_ndefor / sum_pi
    df["rate_abs"] = df["rate_mod"] * correction_factor

    # Pixel area and deforestation density
    pixel_area = xres * yres / 10000
    df["pixel_area"] = pixel_area
    df["defor_dens"] = df["rate_abs"] * pixel_area / time_interval

    if tab_file_defrate is not None:
        df.to_csv(str(tab_file_defrate), sep=",", header=True, index=False)

    del defor_ds, forest_ds, cat_ds
    return df


def defrate_per_class(
    defor_file: PathLike,
    forest_file: PathLike,
    vulnerability_file: PathLike,
    time_interval: float,
    tab_file_defrate: Optional[PathLike] = None,
    deforate_model: Optional[PathLike] = None,
    n_cat_max: int = 30999,
    blk_rows: int = 128,
    verbose: bool = False,
) -> pd.DataFrame:
    """Observed deforestation rate per JNR vulnerability class, two explicit layers.

    Native, period/value-agnostic replacement for the legacy
    ``rmj.legacy.defrate_per_class`` (itself a re-implementation of
    ``riskmapjnr.benchmark.defrate_per_class``). Counts ``forest-at-start``
    pixels and ``deforested`` pixels per vulnerability class:

        numerator   (deforested)      = (defor  == 1)
        denominator (forest at start) = (forest == 1) | (defor == 1)

    Unlike :func:`defrate_per_cat`, the relative model rate is **not** the
    moving-window ``((cat - 2) ...)`` formula but either the observed
    ``ndefor / nfor`` (when ``deforate_model is None``) or the per-class rates
    read from a model-period CSV (``deforate_model``), exactly mirroring the
    JNR benchmark's quantity-adjustment workflow.

    Parameters
    ----------
    defor_file, forest_file : path
        Binary deforestation / forest-at-start rasters (``== 1``).
    vulnerability_file : path
        Vulnerability map (UInt16, ``vulnerability_class * 1000 + subj_id``)
        produced by ``vulnerability_map``.
    time_interval : float
        Period length in years.
    tab_file_defrate : path or None
        CSV output. Defaults to ``None``, i.e. no CSV is written and the table
        is only returned. A bare filename default would resolve against the
        process CWD, which is read-only when the app runs on SEPAL, so callers
        that want the file must pass an explicit path.
    deforate_model : path or None
        CSV with rates from the model period (calibration / historical). When
        given, observed rates are used only for the quantity-adjustment
        correction; ``rate_mod`` comes from the model. When ``None``,
        ``rate_mod = ndefor / nfor``.
    n_cat_max : int
        Highest vulnerability-class code (default 30999 = 30 classes x ~1000
        subjurisdiction ids).

    Returns:
    -------
    pandas.DataFrame
        Per-class table (cat, nfor, ndefor, rate_obs, rate_mod, rate_abs, ...),
        also written to ``tab_file_defrate`` if provided.
    """
    defor_ds, defor_band = _open_band(defor_file)
    forest_ds, forest_band = _open_band(forest_file)
    vuln_ds, vuln_band = _open_band(vulnerability_file)

    gt = defor_ds.GetGeoTransform()
    xres = gt[1]
    yres = -gt[5]

    nblock, nblock_x, _, x, y, nx, ny = makeblock(str(defor_file), blk_rows=blk_rows)[
        :7
    ]

    cat = [c + 1 for c in range(n_cat_max)]
    df = pd.DataFrame({"cat": cat, "nfor": 0, "ndefor": 0})

    for b in range(nblock):
        if verbose:
            progress_bar(nblock, b + 1)
        px, py = b % nblock_x, b // nblock_x
        defor_arr = defor_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        forest_arr = forest_band.ReadAsArray(x[px], y[py], nx[px], ny[py])
        vuln_arr = vuln_band.ReadAsArray(x[px], y[py], nx[px], ny[py])

        defor_mask = defor_arr == 1
        forest_start = (forest_arr == 1) | defor_mask

        data_for = vuln_arr[forest_start]
        data_defor = vuln_arr[defor_mask]

        cat_for = pd.Categorical(data_for.flatten(), categories=cat)
        df["nfor"] += cat_for.value_counts().values
        cat_defor = pd.Categorical(data_defor.flatten(), categories=cat)
        df["ndefor"] += cat_defor.value_counts().values

    # Drop classes with no forest pixels
    df = df[df["nfor"] != 0].copy()

    # Observed annual deforestation rate per class (always computed)
    df["rate_obs"] = 1 - (1 - df["ndefor"] / df["nfor"]) ** (1 / time_interval)

    # Relative model rate: from a model-period CSV, or fall back to observed
    if deforate_model is not None:
        df_mod = pd.read_csv(str(deforate_model))
        df = df.merge(right=df_mod, on="cat", how="left", suffixes=(None, "_mod"))
    else:
        df["rate_mod"] = df["ndefor"] / df["nfor"]

    # Quantity-adjustment correction → absolute deforestation probability
    sum_ndefor = df["ndefor"].sum()
    sum_pi = (df["nfor"] * df["rate_mod"]).sum()
    correction_factor = sum_ndefor / sum_pi
    df["rate_abs"] = df["rate_mod"] * correction_factor

    # Pixel area and deforestation density
    df["time_interval"] = time_interval
    pixel_area = xres * yres / 10000
    df["pixel_area"] = pixel_area
    df["defor_dens"] = df["rate_abs"] * pixel_area / time_interval

    if tab_file_defrate is not None:
        df.to_csv(str(tab_file_defrate), sep=",", header=True, index=False)

    del defor_ds, forest_ds, vuln_ds
    return df
