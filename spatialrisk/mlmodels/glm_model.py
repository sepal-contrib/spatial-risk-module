"""GLM risk model using sklearn LogisticRegression with Patsy formulas."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from spatialrisk.mlmodels.base import BaseRiskModel, _mask_values
from spatialrisk.mlmodels.stats import GLMStats

logger = logging.getLogger("spatial_risk")


class GLMModel(BaseRiskModel):
    """Logistic Regression risk model with Patsy formula support.

    Attributes:
    ----------
    solver : str
        sklearn LogisticRegression solver (default: "lbfgs").
    max_iter : int
        Maximum number of solver iterations (default: 1000).
    random_seed : int, optional
        Random seed for reproducibility.
    """

    model_type: str = "glm"
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_seed: Optional[int] = None
    stats: Optional[GLMStats] = None

    def _collect_stats_from_design(self, y, x) -> None:
        """Build self.stats from the fitted estimator + patsy design (A §2.1)."""
        from spatialrisk.mlmodels.stats import collect_glm_stats, sample_design_label

        y_arr = np.asarray(y)[:, 0]
        try:
            self.stats = collect_glm_stats(
                self._ml_model,
                x.design_info,
                n_rows=int(x.shape[0]),
                n_events=int(y_arr.sum()),
                sample_design=sample_design_label(self.sample),
                max_iter=self.max_iter,
            )
        except Exception as exc:  # stats must never fail a training run
            print(f"  ⚠ model statistics skipped: {exc}")
            self.stats = None

    def fit(
        self,
        formula: Optional[str] = None,
        folder: Optional[Union[str, Path]] = None,
    ) -> "GLMModel":
        """Train a logistic regression model.

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
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss

        # Auto-save full training CSV if samples_path not already set
        if self.samples_path is None:
            _folder = self._resolve_output_folder(folder)
            _folder.mkdir(parents=True, exist_ok=True)
            _csv = _folder / f"samples_{self.model_type}_{self.name or 'model'}.csv"
        else:
            _csv = None

        df, formula = self._prepare_samples(formula, output_csv=_csv)

        print(f"\n🔧 Training GLM ({self.solver}, max_iter={self.max_iter})...")

        y, x = dmatrices(self.formula, df, NA_action="drop")
        # self._x_design_info = x.design_info

        clf = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_seed,
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
            f"✓ GLM trained — {self.n_samples:,} samples, "
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
        """Generate a risk probability GeoTIFF.

        Streams the feature rasters in full-width stripes through
        :func:`spatialrisk.mlmodels.windowed_predict.predict_windowed`. Outputs
        a UInt16 raster scaled to [1, 65535] with 0 as nodata.

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
            Stripe worker threads. None lets the resource policy choose;
            1 runs serially on the calling thread.
        """
        from scipy.special import expit

        from spatialrisk.mlmodels.linear_predictor import compile_linear_predictor
        from spatialrisk.mlmodels.windowed_predict import predict_windowed

        if self._ml_model is None:
            self.load_model()
        active_dataset = self._resolve_dataset(dataset)
        self._ensure_design_info()

        output_file = Path(output_file)
        feature_paths = {var.name: var.path for var in active_dataset.features}
        logger.info("Predicting GLM raster -> %s", output_file)

        design_info = self._x_design_info
        estimator = self._ml_model
        if list(estimator.classes_) != [0, 1]:
            raise ValueError(
                "GLM classes must be [0, 1] for the risk column, "
                f"got {list(estimator.classes_)}"
            )
        predictor = compile_linear_predictor(design_info, estimator.coef_[0])
        intercept = float(estimator.intercept_[0])
        logger.info("GLM design: %s", predictor.describe())

        def predict_block(block_df, extras):
            # expit(decision) is exactly what LogisticRegression.predict_proba
            # applies for a binary model, so this equals predict_proba(...)[:, 1].
            return expit(predictor.eta(block_df) + intercept)

        predict_windowed(
            active_dataset.target.path,
            feature_paths,
            predict_block,
            output_file,
            mask=mask,
            mask_values=_mask_values(mask_value),
            workers=workers,
            n_design_cols=predictor.working_set_columns,
            log=logger,
        )
        logger.info("GLM raster written: %s", output_file)
        self._register_prediction(output_file, dataset=active_dataset)
        return output_file
