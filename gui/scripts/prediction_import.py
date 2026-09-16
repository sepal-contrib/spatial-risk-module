"""Import a user-supplied raster as a first-class :class:`Prediction`.

Lets the analyst bring a prediction map produced outside the app (e.g. exported
from QGIS or another pipeline) into Step 7 — Inference, so it renders on the map
and can be scored in Step 8 — Evaluation alongside computed predictions.

Solara-free (architecture contract #7): a pure adapter over the Project document,
called from the Inference tile. The raster is inspected, range-checked against
the declared value scale, then warped onto the project's base-raster grid and
written as a 1..65535 UInt16 file by ``spatialrisk.predictions.import_raster``.
"""

import re
from pathlib import Path
from typing import Any

IMPORT_DIR_NAME = "imported_predictions"
IMPORT_DATASET_NAME = "imported"


def sanitize_import_name(name: str) -> str:
    """Filesystem- and label-safe token from a free-text name.

    Spaces collapse to hyphens; characters outside ``[A-Za-z0-9._-]`` are dropped.
    Avoids ``_`` runs that ``evaluation.label_for`` would split on. Falls back to
    ``"imported"`` if nothing survives.
    """
    token = re.sub(r"\s+", "-", name.strip())
    token = re.sub(r"[^A-Za-z0-9._-]", "", token)
    return token or IMPORT_DATASET_NAME


def resolve_import_key(project: Any, name: str, src_suffix: str = "") -> str:
    """The model_key ``import_prediction`` would assign to *name* right now.

    Pure read — mirrors the import's duplicate disambiguation (suffix ``-2``,
    ``-3``, … while the registry key or the destination file is taken) so the
    GUI can preview the key before the copy happens.
    """
    dest_dir = Path(project.folders.project_folder) / IMPORT_DIR_NAME
    base = sanitize_import_name(name)
    model_key = base
    suffix = 2
    while (
        f"{model_key}__{IMPORT_DATASET_NAME}" in getattr(project, "predictions", {})
        or (dest_dir / f"{model_key}{src_suffix}").exists()
    ):
        model_key = f"{base}-{suffix}"
        suffix += 1
    return model_key


def import_prediction(
    project: Any,
    src_path: str,
    name: str,
    value_scale: str,
    auto_save: bool = True,
):
    """Adapt *src_path* onto the project grid and register it as a Prediction.

    Parameters
    ----------
    project : Project
        Active project. Must have a ``base_raster``: its geobox is the target
        grid. The adapted raster lands under
        ``project.folders.project_folder / "imported_predictions"``.
    src_path : str
        Path to the local GeoTIFF to import.
    name : str
        User-typed display name. Used (sanitized) as the prediction's
        ``model_key`` so it labels the outputs list and the Evaluation table.
    value_scale : str
        ``"probability"`` (floats 0..1, rescaled to 1..65535) or ``"risk"``
        (whole numbers 1..65535, used as is).
    auto_save : bool
        Persist the project JSON after registering (default True).

    Raises:
    ------
    FileNotFoundError
        *src_path* does not exist.
    ImportRasterError
        No base raster, or the file fails inspection or the scale check.
    """
    from spatialrisk.predictions.import_raster import (
        ImportRasterError,
        adapt_raster,
        check_scale,
        inspect_raster,
        raster_range,
    )
    from spatialrisk.predictions.prediction import Prediction

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Raster to import not found: {src}")
    base = getattr(project, "base_raster", None)
    if base is None:
        raise ImportRasterError(
            "The project has no base raster yet: process the project's variables "
            "first, then import the prediction."
        )

    # The dialog already inspected the file, but the adapter must not trust it.
    info = inspect_raster(src)
    vmin, vmax = raster_range(src)
    check_scale(vmin, vmax, value_scale)

    dest_dir = Path(project.folders.project_folder) / IMPORT_DIR_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_key = resolve_import_key(project, name, ".tif")
    dest = dest_dir / f"{model_key}.tif"

    adapt_raster(src, dest, base.get_base_geobox(), value_scale)

    pred = Prediction(
        name=name,
        path=dest,
        model_key=model_key,
        dataset_name=IMPORT_DATASET_NAME,
        display_palette="far",
        run_params={
            "source_path": str(src),
            "value_scale": value_scale,
            "source_crs": info.crs,
            "source_resolution": list(info.resolution),
            "source_dtype": info.dtype,
            "source_range": [vmin, vmax],
        },
    )
    pred.add_to_project(project, auto_save=auto_save)
    return pred
