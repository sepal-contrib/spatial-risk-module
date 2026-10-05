"""Rough iCAR training-time estimate, for warning before a long run.

forestatrisk's Gibbs sampler (``hbm.binomial_iCAR``) updates one coefficient at
a time and recomputes the full linear predictor over every observation for each
proposal, so one MCMC iteration costs about
``n_obs * (A * n_coef**2 + B * n_coef)``. Every category of a ``C()`` term adds
a coefficient, which is why a many-level categorical (a sub-jurisdiction map,
106 levels over the Peruvian Amazon) can turn minutes into hours.

A and B were fitted on 2026-10-01 to two single-core runs on a developer laptop
(18.3 ms per iteration at 20k samples x 15 coefficients, 41.3 ms at 10k x 40).
Other machines differ, so callers should present the result as "about".
"""

from dataclasses import dataclass, field
from typing import Optional

from spatialrisk.far_helpers import _factor_names

SECONDS_PER_OBS_COEF2 = 1.7e-9
SECONDS_PER_OBS_COEF = 3.6e-8

# Warn above this estimate.
SLOW_TRAINING_SECONDS = 600.0
# A categorical is worth naming when dropping it at least halves the run.
CULPRIT_MIN_SPEEDUP = 2.0

# forestatrisk consumes the formula's last column as the spatial-cell index.
_CELL_TERM = "cell"


@dataclass(frozen=True)
class Culprit:
    """A categorical term and how much faster training is without it."""

    name: str
    levels: int
    speedup: float


@dataclass(frozen=True)
class ICARCostEstimate:
    """Estimated MCMC wall time and the categoricals that drive it."""

    seconds: float
    n_coefficients: int
    culprits: list = field(default_factory=list)  # Culprit, largest speedup first

    @property
    def is_slow(self) -> bool:
        """True when the estimate is long enough to warn about."""
        return self.seconds > SLOW_TRAINING_SECONDS


def estimate_mcmc_seconds(n_obs: int, n_coef: int, iterations: int) -> float:
    """Seconds for ``iterations`` (burnin + mcmc) sampler iterations."""
    per_iteration = n_obs * (
        SECONDS_PER_OBS_COEF2 * n_coef**2 + SECONDS_PER_OBS_COEF * n_coef
    )
    return iterations * per_iteration


def _term_columns(formula: str, categorical_levels: dict) -> list:
    """``(categorical_name_or_None, n_columns)`` per RHS term, patsy-style.

    Treatment coding: a ``C(x)`` factor contributes ``levels - 1`` columns,
    except the first categorical when the model has no intercept, which keeps
    all its levels. Any other factor contributes one. Raises on a formula patsy
    cannot parse.
    """
    from patsy import ModelDesc

    terms = ModelDesc.from_formula(formula).rhs_termlist
    has_intercept = any(not term.factors for term in terms)
    full_rank_spent = has_intercept
    out = []
    for term in terms:
        if not term.factors:
            out.append((None, 1))
            continue
        columns = 1
        categorical = None
        for factor in term.factors:
            names = _factor_names(factor.code)
            if names == {_CELL_TERM}:
                columns = 0
                break
            is_c = factor.code.replace(" ", "").startswith("C(")
            known = sorted(names & set(categorical_levels))
            if is_c and known:
                levels = int(categorical_levels[known[0]])
                columns *= levels if not full_rank_spent else levels - 1
                full_rank_spent = True
                categorical = known[0] if len(term.factors) == 1 else None
        if columns:
            out.append((categorical, columns))
    return out


def count_coefficients(formula: str, categorical_levels: dict) -> int:
    """Number of design columns forestatrisk samples a coefficient for."""
    return sum(columns for _, columns in _term_columns(formula, categorical_levels))


def estimate_icar_training(
    formula: str, categorical_levels: dict, *, n_obs: int, iterations: int
) -> Optional[ICARCostEstimate]:
    """Estimate for ``formula``; ``None`` when the formula does not parse.

    ``categorical_levels`` maps each categorical layer's name to its number of
    levels (its raster's full domain, which is what fit() declares).
    """
    try:
        terms = _term_columns(formula, categorical_levels)
    except Exception:
        return None
    n_coef = sum(columns for _, columns in terms)
    seconds = estimate_mcmc_seconds(n_obs, n_coef, iterations)
    culprits = []
    for name, columns in terms:
        if name is None:
            continue
        without = estimate_mcmc_seconds(n_obs, n_coef - columns, iterations)
        speedup = seconds / without if without > 0 else float("inf")
        if speedup >= CULPRIT_MIN_SPEEDUP:
            culprits.append(Culprit(name, int(categorical_levels[name]), speedup))
    culprits.sort(key=lambda c: c.speedup, reverse=True)
    return ICARCostEstimate(seconds=seconds, n_coefficients=n_coef, culprits=culprits)
