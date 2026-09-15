"""Random Forest risk model using sklearn with Patsy formulas."""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from spatialrisk.mlmodels.base import BaseRiskModel
from spatialrisk.mlmodels.stats import RFStats


class RFModel(BaseRiskModel):
    """Random Forest risk model with Patsy formula support.

    Attributes:
    ----------
    n_trees : int
        Number of decision trees (default: 100).
    max_depth : int
        Maximum tree depth (default: 15).
    min_samples_leaf : int
        Minimum samples per leaf node (default: 2).
    random_seed : int, optional
        Random seed for reproducibility.
    """

    model_type: str = "rf"
    n_trees: int = 100
    max_depth: int = 15
    min_samples_leaf: int = 2
    random_seed: Optional[int] = None
    stats: Optional[RFStats] = None

    def _collect_stats_from_design(self, y, x) -> None:
        """Build self.stats from the fitted forest + patsy design (A §2.3)."""
        from spatialrisk.mlmodels.stats import collect_rf_stats, sample_design_label

        y_arr = np.asarray(y)[:, 0]
        try:
            self.stats = collect_rf_stats(
                self._ml_model,
                x.design_info,
                n_rows=int(x.shape[0]),
                n_events=int(y_arr.sum()),
                sample_design=sample_design_label(self.sample),
            )
        except Exception as exc:  # stats must never fail a training run
            print(f"  ⚠ model statistics skipped: {exc}")
            self.stats = None

    def fit(
        self,
        formula: Optional[str] = None,
        folder: Optional[Union[str, Path]] = None,
    ) -> "RFModel":
        """Train a Random Forest classifier.

        Parameters
        ----------
        formula : str, optional
            Patsy formula. If omitted, falls back to self.formula or
            auto-generates via generate_patsy_formula(self.dataset).
        folder : str or Path, optional
            Folder for saving the model pickle. Defaults to the project model
            folder; raises when the model has no project either.

        Returns:
        -------
        self
        """
        from patsy import dmatrices
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import log_loss

        # Auto-save full training CSV if samples_path not already set
        if self.samples_path is None:
            _folder = self._resolve_output_folder(folder)
            _folder.mkdir(parents=True, exist_ok=True)
            _csv = _folder / f"samples_{self.model_type}_{self.name or 'model'}.csv"
        else:
            _csv = None

        df, formula = self._prepare_samples(formula, output_csv=_csv)

        print(
            f"\n🔧 Training Random Forest "
            f"(n_trees={self.n_trees}, max_depth={self.max_depth})..."
        )

        df = df.dropna()
        y, x = dmatrices(self.formula, df, NA_action="drop")
        self._x_design_info = x.design_info

        clf = RandomForestClassifier(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=self.random_seed,
            oob_score=True,
        )
        y_arr = np.asarray(y)[:, 0]
        x_arr = np.asarray(x)
        clf.fit(x_arr, y_arr)
        self._ml_model = clf

        # Training metrics
        self.n_samples = len(df)
        y_pred = clf.predict_proba(x_arr)[:, 1]
        self.deviance = 2.0 * log_loss(y_arr, y_pred, normalize=False)

        self._collect_stats_from_design(y, x)

        self._stamp_now()
        self.trained = True
        print(
            f"✓ RF trained — {self.n_samples:,} samples, "
            f"deviance={self.deviance:.2f}, trained_at={self.trained_at}"
        )

        self.save(folder=folder)
        return self

    def apply(
        self,
        output_file: Union[str, Path],
        dataset: Optional[Any] = None,
        mask: Optional[Union[str, Path]] = None,
        mask_value: Union[int, float, list] = 0,
    ) -> Path:
        """Generate a deforestation probability GeoTIFF.

        Processes the feature rasters block-by-block. Outputs a UInt16
        raster scaled to [1, 65535] with 0 as nodata, using ``far.misc.rescale``.

        Parameters
        ----------
        output_file : str or Path
            Path for the output GeoTIFF.
        dataset : Dataset, optional
            Dataset with target and features configured. If omitted, uses
            self.dataset. Must contain all features in self.feature_names.
        mask : str or Path, optional
            Path to a mask raster. Pixels matching ``mask_value`` (or the
            raster's nodata) are set to nodata (0) in the output.
            If omitted, prediction runs over the full raster stack.
        mask_value : int, float, or list of int/float, optional
            Value(s) in the mask raster that identify pixels to suppress.
            Defaults to 0. Ignored when ``mask`` is None.
        """
        import forestatrisk as far
        import rasterio
        from patsy.highlevel import build_design_matrices

        from spatialrisk.parallel import PREDICT_BAND_ROWS, single_thread_math
        from spatialrisk.raster_profile import rasterio_profile

        if self._ml_model is None:
            self.load_model()

        active_dataset = self._resolve_dataset(dataset)

        if self._x_design_info is None:
            if self.samples_path is not None and Path(self.samples_path).exists():
                from patsy import dmatrices as _dmatrices

                _df = pd.read_csv(self.samples_path).dropna()
                _, x_ref = _dmatrices(self.formula, _df, NA_action="drop")
                self._x_design_info = x_ref.design_info
            else:
                raise RuntimeError(
                    "Cannot reconstruct design info: samples_path not set or "
                    "file missing. Re-run fit() to regenerate samples."
                )

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        feature_paths = {var.name: var.path for var in active_dataset.features}

        print(f"\n🗺  Predicting RF raster → {output_file}")

        with rasterio.open(active_dataset.target.path) as ref:
            profile = ref.profile.copy()

        # Tiled + ZSTD (deflate fallback) rather than the target's own
        # layout: see spatialrisk.raster_profile for the measurements.
        profile.update(dtype="uint16", count=1, nodata=0)
        profile.update(rasterio_profile("uint16"))

        _mask_values = (
            (mask_value if isinstance(mask_value, (list, tuple)) else [mask_value])
            if mask is not None
            else None
        )

        # Tile-aligned bands, BLAS/OpenMP on one thread: see spatialrisk.parallel.
        with single_thread_math():
            with rasterio.open(output_file, "w", **profile) as dst:
                blockinfo = far.misc.makeblock(
                    str(active_dataset.target.path), blk_rows=PREDICT_BAND_ROWS
                )
                nblock, nblock_x = blockinfo[0], blockinfo[1]
                x_off, y_off, nx, ny = (
                    blockinfo[3],
                    blockinfo[4],
                    blockinfo[5],
                    blockinfo[6],
                )

                for b in range(nblock):
                    px = b % nblock_x
                    py = b // nblock_x
                    col_start, row_start = x_off[px], y_off[py]
                    n_cols, n_rows = nx[px], ny[py]
                    window = rasterio.windows.Window(
                        col_start, row_start, n_cols, n_rows
                    )

                    # Apply mask before prediction
                    mask_invalid = np.zeros(n_rows * n_cols, dtype=bool)
                    if mask is not None:
                        with rasterio.open(mask) as mask_src:
                            mask_block = mask_src.read(1, window=window)
                            mask_nodata = mask_src.nodata
                        mask_invalid = np.isin(mask_block.ravel(), _mask_values)
                        if mask_nodata is not None:
                            mask_invalid |= mask_block.ravel() == mask_nodata

                    # Read feature data for this block, replacing nodata with NaN
                    block_dict = {}
                    for name, path in feature_paths.items():
                        with rasterio.open(path) as src:
                            arr = src.read(1, window=window).astype(float)
                            if src.nodata is not None:
                                arr[arr == src.nodata] = np.nan
                        block_dict[name] = arr.ravel()

                    block_df_full = pd.DataFrame(block_dict)
                    valid_mask = (
                        ~block_df_full.isnull().any(axis=1).to_numpy() & ~mask_invalid
                    )
                    block_df = block_df_full[valid_mask]

                    out_arr = np.zeros(n_rows * n_cols, dtype=np.uint16)

                    if not block_df.empty:
                        (x_block,) = build_design_matrices(
                            [self._x_design_info], block_df, NA_action="drop"
                        )
                        proba = self._ml_model.predict_proba(np.asarray(x_block))[:, 1]
                        out_arr[valid_mask] = far.misc.rescale(proba).astype(np.uint16)

                    dst.write(
                        out_arr.reshape(n_rows, n_cols),
                        1,
                        window=window,
                    )

        print(f"✓ RF raster written: {output_file}")
        self._register_prediction(output_file, dataset=active_dataset)
        return output_file
