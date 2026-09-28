"""iCAR Bayesian risk model using forestatrisk with Patsy formulas.

The intrinsic Conditional Auto-Regressive (iCAR) model accounts for
spatial autocorrelation through a latent spatial random effect (rho).
Training uses MCMC via forestatrisk.model_binomial_iCAR.
"""

import concurrent.futures
import logging
import multiprocessing
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from spatialrisk.mlmodels.base import BaseRiskModel, _mask_values
from spatialrisk.mlmodels.stats import ICARStats

logger = logging.getLogger("spatial_risk")


def compute_cell_indices(
    cell_id: "np.ndarray",
    raster_path: str,
    csize_km: float,
) -> "np.ndarray":
    """Convert pixel-based cell_id values to spatial cell indices.

    ``dataset.extract_at_points()`` stores ``cell_id = row * ncols + col`` (a flat
    pixel index).  forestatrisk's ``model_binomial_iCAR`` expects the ``cell``
    column to contain the index into the spatial-cell grid produced by
    ``cellneigh(raster, csize, rank=1)``, i.e. values in ``[0, ncell)``.

    This function performs that conversion using the same algorithm as the
    sampling notebook::

        bigJ = floor((pts_x - Xmin) / csize_m)
        bigI = floor((Ymax  - pts_y) / csize_m)
        cell = bigI * ncol_cells + bigJ

    Parameters
    ----------
    cell_id : array-like of int
        Flat pixel indices (row * ncols + col) from the samples DataFrame.
    raster_path : str
        Path to the reference raster (same one passed to ``cellneigh``).
    csize_km : float
        Spatial cell size in kilometres (must match the value used in
        ``cellneigh``).

    Returns:
    -------
    np.ndarray of int
        Spatial cell indices aligned with the ``cellneigh`` output.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        gt = src.transform
        ncols_r = src.width
        Xmin = gt.c
        Xmax = gt.c + gt.a * src.width
        Ymax = gt.f

    csize_m = csize_km * 1000
    ncol_cells = int(np.ceil((Xmax - Xmin) / csize_m))

    pixel_ids = np.asarray(cell_id, dtype=int)
    pixel_row = pixel_ids // ncols_r
    pixel_col = pixel_ids % ncols_r
    pts_x = (pixel_col + 0.5) * gt.a + gt.c
    pts_y = (pixel_row + 0.5) * gt.e + gt.f
    bigJ = ((pts_x - Xmin) / csize_m).astype(int)
    bigI = ((Ymax - pts_y) / csize_m).astype(int)
    return bigI * ncol_cells + bigJ


def _mcmc_worker(payload: dict) -> dict:
    """Run forestatrisk's MCMC sampler. Executed in a spawned child process.

    Must stay a module-level function so multiprocessing can pickle it.
    """
    import os

    import forestatrisk as far

    mod = far.model_binomial_iCAR(
        suitability_formula=payload["formula"],
        data=payload["data"],
        n_neighbors=payload["n_neighbors"],
        neighbors=payload["neighbors"],
        burnin=payload["burnin"],
        mcmc=payload["mcmc"],
        thin=payload["thin"],
        priorVrho=payload["prior_vrho"],
        seed=payload["seed"],
        verbose=payload["verbose"],
    )
    # The full MCMC trace lives and dies here: summarise it in-process and send
    # only the plain dicts/floats back (spec §2.2 non-goal — no trace crosses).
    summary = None
    try:
        from spatialrisk.mlmodels.stats import summarize_icar_mcmc

        summary = summarize_icar_mcmc(mod.mcmc, mod._x_design_info.column_names)
    except Exception as exc:  # summary must never fail the training run
        print(f"  ⚠ posterior summary skipped: {exc}")

    # Only picklable posterior summaries cross the process boundary (the full
    # model object holds patsy design objects that cannot be pickled).
    return {
        "betas": np.array(mod.betas),
        "rho": np.array(mod.rho),
        "Vrho": float(mod.Vrho) if hasattr(mod, "Vrho") else None,
        "deviance": float(mod.deviance),
        "posterior_summary": summary,
        "worker_pid": os.getpid(),
    }


def run_icar_mcmc(
    formula: str,
    data: "pd.DataFrame",
    n_neighbors: "np.ndarray",
    neighbors: "np.ndarray",
    *,
    burnin: int,
    mcmc: int,
    thin: int,
    prior_vrho: float,
    seed: int,
    verbose: int = 1,
) -> dict:
    """Run the iCAR MCMC in a spawned child process and return its posteriors.

    forestatrisk's ``hbm`` C extension holds the GIL for the entire sampler
    run, so executing it in-process stalls every other Python thread — in the
    GUI that freezes the whole Solara server until training finishes. A
    separate process has its own GIL, keeping the app responsive. "spawn"
    (not "fork") because forking a multithreaded server process with GDAL/EE
    state loaded is unsafe.
    """
    payload = {
        "formula": formula,
        "data": data,
        "n_neighbors": n_neighbors,
        "neighbors": neighbors,
        "burnin": burnin,
        "mcmc": mcmc,
        "thin": thin,
        "prior_vrho": prior_vrho,
        "seed": seed,
        "verbose": verbose,
    }
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        result = pool.submit(_mcmc_worker, payload).result()
    print(f"  MCMC ran in subprocess (pid {result['worker_pid']})")
    return result


def build_icar_stats(
    posteriors, design_column_names, *, n_rows, n_events, sample_design
) -> ICARStats:
    """ICARStats from the worker's posteriors dict (Spec A §2.2).

    Preferred source is posteriors["posterior_summary"] (mean/SD/CI computed
    from the trace inside the worker). When it is absent — the worker's
    summary failed, or an old pickle path — fall back to point estimates
    zipped against design_column_names[:-1], under the same 'cell' guard the
    summary path enforces: mislabelling raises, it never guesses.
    """
    from spatialrisk.mlmodels.stats import Coefficient, binomial_null_deviance

    deviance_summary = None
    summary = posteriors.get("posterior_summary")
    if summary:
        coefficients = [Coefficient(**b) for b in _summary_kwargs(summary["betas"])]
        # .get, not ["vrho"]: a summary missing that key would raise KeyError,
        # which fit() catches by dropping the WHOLE stats record — throwing away
        # the coefficients that were present for the sake of one absent scalar.
        vrho_summary = summary.get("vrho")
        vrho = (
            Coefficient(name="Vrho", **_summary_kwargs_one(vrho_summary))
            if vrho_summary
            else None
        )
        dev_summary = summary.get("deviance")
        deviance_summary = (
            Coefficient(name="Deviance", **_summary_kwargs_one(dev_summary))
            if dev_summary
            else None
        )
    else:
        names = list(design_column_names)
        if not names or names[-1] != "cell":
            raise ValueError(
                "expected the design's last column to be 'cell'; got "
                f"{names[-1] if names else None!r}"
            )
        betas = np.asarray(posteriors["betas"], dtype=float)
        beta_names = names[:-1]
        if len(beta_names) != len(betas):
            raise ValueError(
                f"{len(beta_names)} beta names vs {len(betas)} point estimates"
            )
        coefficients = [
            Coefficient(name=n, estimate=float(b)) for n, b in zip(beta_names, betas)
        ]
        v = posteriors.get("Vrho")
        vrho = Coefficient(name="Vrho", estimate=float(v)) if v is not None else None

    rho = np.asarray(posteriors.get("rho", []), dtype=float)
    rho = rho[np.isfinite(rho)]
    rho_kw = (
        {
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "rho_mean": float(rho.mean()),
            "rho_std": float(rho.std(ddof=1)) if rho.size > 1 else None,
        }
        if rho.size
        else {}
    )
    return ICARStats(
        n_rows=n_rows,
        n_events=n_events,
        sample_design=sample_design,
        coefficients=coefficients,
        vrho=vrho,
        deviance_summary=deviance_summary,
        # Counts-only, so the point-estimate fallback (= recovery of an old
        # model) provides the "% explained" reference just as well.
        null_deviance=binomial_null_deviance(n_events, n_rows),
        **rho_kw,
    )


def _summary_kwargs(betas):
    """Map summary beta dicts to Coefficient kwargs (mean -> estimate)."""
    return [{"name": b["name"], **_summary_kwargs_one(b)} for b in betas]


def _summary_kwargs_one(d):
    """One summary dict -> Coefficient kwargs, minus the name.

    rhat/ess via .get: a summary produced before the diagnostics existed
    (an old worker pickle) simply leaves them None.
    """
    return {
        "estimate": d["mean"],
        "std": d["std"],
        "ci_low": d["ci_low"],
        "ci_high": d["ci_high"],
        "rhat": d.get("rhat"),
        "ess": d.get("ess"),
    }


class ICARModel(BaseRiskModel):
    """Bayesian iCAR spatial risk model.

    Requires the ``cell_id`` column present in DataFrames produced by
    ``dataset.extract_at_points()``, which encodes the raster cell index and
    enables construction of the spatial neighbourhood graph.

    Attributes:
    ----------
    csize : float
        Cell size (km) for building the spatial neighbourhood (default: 10).
    mcmc : int
        Total MCMC iterations (default: 6000).
    burnin : int
        Number of burn-in iterations (default: 4000).
    thin : int
        Thinning factor (default: 1).
    prior_vrho : float
        Prior variance for rho. -1 uses a uniform prior (default: -1).
    beta_start : float
        Starting value for betas. -99 triggers automatic initialisation
        (default: -99).
    random_seed : int, optional
        Random seed for reproducibility.
    rho_path : Path, optional
        Path to the interpolated rho GeoTIFF saved after training.
    """

    model_type: str = "icar"
    csize: float = 10.0
    mcmc: int = 4000
    burnin: int = 4000
    thin: int = 1
    prior_vrho: float = -1.0
    beta_start: float = -99.0
    random_seed: Optional[int] = None
    stats: Optional[ICARStats] = None
    rho_path: Optional[Path] = None
    csize_interpolate: float = 0.1

    def output_files(self) -> list:
        """iCAR also owns its spatial random-effect (rho) raster."""
        files = super().output_files()
        if self.rho_path:
            files.append(Path(self.rho_path))
        return files

    def fit(
        self,
        formula: Optional[str] = None,
        folder: Optional[Union[str, Path]] = None,
    ) -> "ICARModel":
        """Train the iCAR model via MCMC.

        Parameters
        ----------
        formula : str, optional
            Patsy formula. If omitted, falls back to self.formula or
            auto-generates via generate_patsy_formula(self.dataset).
            The ``cell`` term required by forestatrisk is appended
            automatically if absent.
        folder : str or Path, optional
            Folder for saving the model pickle and rho raster. Defaults to
            the project icar_model folder; raises when the model has no
            project either.

        Returns:
        -------
        self
        """
        import forestatrisk as far
        from patsy import dmatrices

        # Auto-save full training CSV if samples_path not already set
        if self.samples_path is None:
            _folder = self._resolve_output_folder(folder)
            _folder.mkdir(parents=True, exist_ok=True)
            _csv = _folder / f"samples_{self.model_type}_{self.name or 'model'}.csv"
        else:
            _csv = None

        df, formula = self._prepare_samples(formula, output_csv=_csv)

        if "cell_id" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'cell_id' column. "
                "Use dataset.extract_at_points() to generate samples."
            )

        # Target raster path — available directly from self.dataset. Absolute:
        # it is handed straight to forestatrisk (cellneigh, interpolate_rho),
        # which reopens it by name, so its meaning must not depend on the CWD.
        raster_path = str(Path(self.dataset.target.path).resolve())

        # forestatrisk expects the column to be named "cell" and values must be
        # spatial cell indices matching cellneigh(raster, csize, rank=1).
        # cell_id stores raw pixel indices (row * ncols + col), so we convert.
        df = df.copy()
        df["cell"] = compute_cell_indices(df["cell_id"].values, raster_path, self.csize)

        # Append cell term to formula if not present
        icar_formula = self.formula
        if "+ cell" not in icar_formula and "~cell" not in icar_formula:
            icar_formula = icar_formula + " + cell"

        print(
            f"\n🔧 Training iCAR model "
            f"(mcmc={self.mcmc}, burnin={self.burnin}, csize={self.csize} km)..."
        )

        df = df.dropna()
        y, x = dmatrices(icar_formula, df, NA_action="drop")

        n_obs = len(df)

        print("  Building spatial neighbourhood...")
        n_neighbors, adj = far.cellneigh(raster_path, self.csize, rank=1)

        # MCMC — isolated in a subprocess so the GIL-holding sampler cannot
        # stall the calling process (see run_icar_mcmc).
        posteriors = run_icar_mcmc(
            icar_formula,
            df,
            n_neighbors,
            adj,
            burnin=self.burnin,
            mcmc=self.mcmc,
            thin=self.thin,
            prior_vrho=self.prior_vrho,
            seed=self.random_seed if self.random_seed is not None else 1234,
            verbose=1,
        )

        self._ml_model = {
            "betas": posteriors["betas"],
            "rho": posteriors["rho"],
            "Vrho": posteriors["Vrho"],
            "deviance": posteriors["deviance"],
            "formula": icar_formula,
        }
        self.n_samples = n_obs
        self.deviance = self._ml_model["deviance"]

        from spatialrisk.mlmodels.stats import sample_design_label

        try:
            self.stats = build_icar_stats(
                posteriors,
                x.design_info.column_names,
                # ModelStatsBase.n_rows is the post-NA-drop DESIGN row count, and
                # GLM/RF both take it from x.shape[0]. df is already dropna()'d
                # above so this equals n_obs today; reading the design keeps the
                # three families reporting the same quantity by construction.
                n_rows=int(x.shape[0]),
                n_events=int(np.asarray(y)[:, 0].sum()),
                sample_design=sample_design_label(self.sample),
            )
        except Exception as exc:  # stats must never fail a training run
            print(f"  ⚠ model statistics skipped: {exc}")
            self.stats = None

        self._stamp_now()
        self.trained = True
        print(
            f"✓ iCAR trained — {self.n_samples:,} samples, "
            f"deviance={self.deviance:.2f}, trained_at={self.trained_at}"
        )

        # Resolve output folder
        out_dir = self._resolve_output_folder(folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save pickle
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.name or "model"
        pickle_path = out_dir / f"icar_{base}_{ts}.pickle"
        payload = {
            "ml_model": self._ml_model,
            "formula": self.formula,
            "samples_path": self.samples_path,
        }
        with open(pickle_path, "wb") as fh:
            pickle.dump(payload, fh)
        self.model_path = pickle_path
        print(f"  iCAR model saved to: {pickle_path}")

        # Interpolate rho to full raster grid.
        # The path is made absolute before it crosses into forestatrisk because
        # interpolate_rho writes a second, unrequested file next to the one we
        # ask for: rho_orig.tif, placed at
        # os.path.join(os.path.dirname(output_file), "rho_orig.tif"). A bare
        # filename makes os.path.dirname() return "", so that sibling is created
        # in the process CWD instead — the read-only shared mount on SEPAL.
        rho_path = (out_dir / f"rho_{base}_{ts}.tif").resolve()
        far.interpolate_rho(
            rho=self._ml_model["rho"],
            input_raster=raster_path,
            output_file=str(rho_path),
            csize_orig=self.csize,
            csize_new=self.csize_interpolate,
        )
        self.rho_path = rho_path
        print(f"  Rho raster saved to: {rho_path}")

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
        """Generate a spatial deforestation probability GeoTIFF.

        Uses the stored betas and the interpolated rho raster as the
        spatial random effect. Streams the feature rasters in full-width
        stripes through
        :func:`spatialrisk.mlmodels.windowed_predict.predict_windowed`.
        Outputs a UInt16 raster scaled to [1, 65535] with 0 as nodata.

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
        from spatialrisk.mlmodels.linear_predictor import compile_linear_predictor
        from spatialrisk.mlmodels.windowed_predict import ExtraLayer, predict_windowed

        if self._ml_model is None:
            self.load_model()
        active_dataset = self._resolve_dataset(dataset)
        self._ensure_design_info()
        if self.rho_path is None or not Path(self.rho_path).exists():
            raise RuntimeError(
                "rho_path is not set or file not found. "
                "Ensure the model was trained with fit() before predicting."
            )

        output_file = Path(output_file)
        feature_paths = {var.name: var.path for var in active_dataset.features}
        logger.info("Predicting iCAR raster -> %s", output_file)

        design_info = self._x_design_info
        betas = np.array(self._ml_model["betas"])
        predictor = compile_linear_predictor(design_info, betas)
        logger.info("iCAR design: %s", predictor.describe())

        def predict_block(block_df, extras):
            # iCAR prediction: logit(p) = X @ betas + rho
            linear_pred = predictor.eta(block_df) + extras["rho"]
            return 1.0 / (1.0 + np.exp(-linear_pred))

        predict_windowed(
            # Absolute for the same reason as in fit(): the path must not be
            # read relative to the process CWD.
            Path(active_dataset.target.path).resolve(),
            feature_paths,
            predict_block,
            output_file,
            mask=mask,
            mask_values=_mask_values(mask_value),
            mask_by_bounds=True,
            extra_layers={"rho": ExtraLayer(Path(self.rho_path), "bilinear")},
            workers=workers,
            n_design_cols=predictor.working_set_columns,
            log=logger,
        )
        logger.info("iCAR raster written: %s", output_file)
        self._register_prediction(output_file, dataset=active_dataset)
        return output_file
