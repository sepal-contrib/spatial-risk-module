"""Random Forest risk model using sklearn with Patsy formulas."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from spatialrisk.mlmodels.base import BaseRiskModel, _mask_values
from spatialrisk.mlmodels.stats import RFStats

logger = logging.getLogger("spatial_risk")


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
        *,
        workers: Optional[int] = None,
    ) -> Path:
        """Generate a deforestation probability GeoTIFF.

        Streams the feature rasters in full-width stripes through
        :func:`spatialrisk.mlmodels.windowed_predict.predict_windowed`, and
        each stripe's design through
        :class:`spatialrisk.mlmodels.design_matrix.DesignBuilder` in float32
        row chunks, so a many-level categorical costs neither patsy's per-value
        level loop nor a whole-stripe one-hot matrix. Outputs a UInt16 raster
        scaled to [1, 65535] with 0 as nodata.

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
        mask_value : int, float, or list of int/float, optional
            Value(s) in the mask raster that identify pixels to suppress.
            Defaults to 0. Ignored when ``mask`` is None.
        workers : int, optional
            Stripe worker threads. None -- the default -- runs serially on
            the calling thread, keeping the estimator's own ``n_jobs``, which
            is where a forest's parallelism already lives; unlike the other
            predictors the resource policy is not consulted (the numbers
            behind that are in the body). ``1`` is the same path. Any other
            worker count pins the estimator to ``n_jobs=1`` for the run: with
            a stripe pool the outer workers own the cores, so joblib's
            per-call tree fan-out would multiply them (workers x n_jobs)
            instead of adding parallelism.
        """
        from spatialrisk.mlmodels.design_matrix import compile_design_builder
        from spatialrisk.mlmodels.windowed_predict import predict_windowed

        if self._ml_model is None:
            self.load_model()
        active_dataset = self._resolve_dataset(dataset)
        self._ensure_design_info()

        output_file = Path(output_file)
        feature_paths = {var.name: var.path for var in active_dataset.features}
        logger.info("Predicting RF raster -> %s", output_file)

        estimator = self._ml_model
        builder = compile_design_builder(self._x_design_info)
        logger.info("RF design: %s", builder.describe())

        def predict_chunk(x):
            # Looked up per call, not bound once: the n_jobs pin below and the
            # tests' spies act on the estimator the closure reads.
            return estimator.predict_proba(x)[:, 1]

        def predict_block(block_df, extras):
            # Each float32 chunk is exactly what sklearn made of patsy's whole
            # float64 matrix, and every tree reads each row on its own, so the
            # chunking changes no probability (tests/test_rf_direct_design.py).
            return builder.evaluate(block_df, predict_chunk)

        # A forest is the one predictor whose default is NOT the resource
        # policy: joblib already spreads its trees over every core, while a
        # pooled run gives each stripe worker one joblib thread and is
        # therefore capped at the worker count -- which the memory budget
        # pins low on the machine that matters. Measured 2026-09-22 on a
        # SEPAL c8 (8 cores, 13.9 GiB free, 62 Mpx of a 40412 px wide,
        # 8-feature stack, where plan_inference chose 2 workers, memory-bound):
        # serial 17.4 s, 2 workers 30.8 s (+77 %), 4 workers 21.2 s (+22 %) --
        # every pooled count lost. Pooling only paid on a 16-core dev box,
        # where the policy could afford 8 workers (15.6 s -> 12.2 s), and even
        # there 2 workers cost 32.9 s. An explicit ``workers`` is still
        # honoured and is the way to pool a forest deliberately; pinning it
        # here does mean SPATIAL_RISK_INFERENCE_WORKERS, which plan_inference
        # reads, no longer reaches one. Caveat on those c8 numbers: they were
        # taken while an explicit worker count did not reach the plan's
        # gdal_threads/cachemax, so the serial arm read with
        # GDAL_NUM_THREADS=1 and a two-worker cache; serial RF is now, if
        # anything, a little faster than recorded here. Those runs also
        # charged a forest its full design width per pixel; now that the
        # design builder charges one column, the budget no longer pins a
        # pooled forest low. The serial default stands until a c8 re-measure.
        workers = 1 if workers is None else workers
        # With a stripe pool the outer workers own the cores: joblib's per-call
        # tree fan-out would multiply them (workers x n_jobs). Serial keeps the
        # pickled n_jobs (-1), which is where a single run's parallelism lives.
        pooled = int(workers) > 1
        saved_n_jobs = estimator.n_jobs
        if pooled:
            estimator.n_jobs = 1
        try:
            predict_windowed(
                active_dataset.target.path,
                feature_paths,
                predict_block,
                output_file,
                mask=mask,
                mask_values=_mask_values(mask_value),
                workers=workers,
                n_design_cols=builder.working_set_columns,
                log=logger,
            )
        finally:
            estimator.n_jobs = saved_n_jobs
        logger.info("RF raster written: %s", output_file)
        self._register_prediction(output_file, dataset=active_dataset)
        return output_file
