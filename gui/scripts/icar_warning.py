"""Inputs and wording for the New model dialog's slow-iCAR warning.

Solara-free, so the dialog and tests can both import it. The estimate lives in
:mod:`spatialrisk.mlmodels.icar_cost`; this module reads what the form holds
(sample, the burnin/mcmc text fields, the formula) and words the result.
"""

from typing import Optional

from gui.i18n import t
from gui.scripts.model_registry import MODEL_REGISTRY
from spatialrisk.mlmodels.icar_cost import ICARCostEstimate, estimate_icar_training

_ICAR_DEFAULTS = {p["key"]: p["default"] for p in MODEL_REGISTRY["icar"]["params"]}


def categorical_level_counts(dataset) -> dict:
    """``{name: number of categories}`` for the dataset's categorical features.

    Read from the categories stored when each layer was harmonized, so it is
    cheap enough for a render body. A layer harmonized before they were stored
    is left out; the Harmonization step asks for it to be harmonized again.
    """
    counts = {}
    for var in getattr(dataset, "features", None) or []:
        if getattr(var, "raster_type", None) != "categorical":
            continue
        levels = getattr(var, "categorical_levels", None)
        if levels:
            counts[var.name] = len(levels)
    return counts


def sample_size(sample) -> int:
    """Points actually drawn (class counts), else the requested count."""
    if sample is None:
        return 0
    drawn = sum(int(v) for v in (getattr(sample, "class_counts", None) or {}).values())
    return drawn or int(getattr(sample, "n_samples", None) or 0)


def _int_param(params: dict, key: str) -> int:
    try:
        return int(float(params.get(key, _ICAR_DEFAULTS[key])))
    except (TypeError, ValueError):
        return int(_ICAR_DEFAULTS[key])


def icar_iterations(params: dict) -> int:
    """Total of burnin and mcmc as typed in the form, defaults where unreadable."""
    return _int_param(params, "burnin") + _int_param(params, "mcmc")


def icar_slow_estimate(
    formula: str, levels: dict, sample, params: dict
) -> Optional[ICARCostEstimate]:
    """The estimate when it is worth a warning, else ``None``.

    Worth a warning means slow *and* driven by a categorical the user can drop.
    """
    n_obs = sample_size(sample)
    if not formula or not n_obs:
        return None
    est = estimate_icar_training(
        formula, levels, n_obs=n_obs, iterations=icar_iterations(params)
    )
    if est is None or not est.is_slow or not est.culprits:
        return None
    return est


def _duration(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 120:
        return t("tiles.train.duration_minutes", n=max(1, int(minutes + 0.5)))
    return t("tiles.train.duration_hours", n=int(seconds / 3600.0 + 0.5))


def icar_slow_message(est: ICARCostEstimate) -> str:
    """Warning text naming the main culprit and the two ways to drop it."""
    top = est.culprits[0]
    return t(
        "tiles.train.icar_slow_warning",
        duration=_duration(est.seconds),
        name=top.name,
        levels=top.levels,
        speedup=int(top.speedup + 0.5),
    )
