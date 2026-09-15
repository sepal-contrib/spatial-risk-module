"""Set deforestation category to zero beyond the forest-edge distance threshold."""

from pathlib import Path
from typing import Union

from spatialrisk.raster_profile import reencode_in_place


def set_defor_cat_zero(
    ldefrate_file: Union[str, Path],
    forest_edge_file: Union[str, Path],
    dist_thresh: float,
    output_file: Union[str, Path],
    blk_rows: int = 256,
    verbose: bool = False,
) -> None:
    """Set local deforestation rate to zero beyond the distance threshold.

    Thin wrapper around ``riskmapjnr.set_defor_cat_zero`` with cleaner
    parameter names.

    Pixels whose distance to forest edge exceeds ``dist_thresh`` are set
    to zero in the output raster, reflecting negligible deforestation risk
    far from the forest edge.

    Parameters
    ----------
    ldefrate_file : str or Path
        Local deforestation rate raster produced by ``local_defor_rate``
        (uint16).
    forest_edge_file : str or Path
        Distance-to-forest-edge raster (metres) for this period's initial year.
    dist_thresh : float
        Distance threshold in metres (from ``dist_edge_threshold``).
        Pixels beyond this distance are zeroed out.
    output_file : str or Path
        Output probability / risk-category GeoTIFF path.
    blk_rows : int
        Number of rows per processing block (default: ``256``).
    verbose : bool
        Print progress messages (default: ``False``).
    """
    import riskmapjnr as rmj

    rmj.set_defor_cat_zero(
        ldefrate_file=str(ldefrate_file),
        dist_file=str(forest_edge_file),
        dist_thresh=dist_thresh,
        ldefrate_with_zero_file=str(output_file),
        blk_rows=blk_rows,
        verbose=verbose,
    )
    # riskmapjnr writes LZW row strips; the app reads this map as random
    # tiles (viewer) and full-width bands (evaluation), so re-encode it in
    # the canonical tiled layout. Guarded so a stubbed upstream is harmless.
    if Path(output_file).exists():
        reencode_in_place(output_file)
