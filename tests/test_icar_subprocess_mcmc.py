# tests/test_icar_subprocess_mcmc.py
"""ICAR MCMC must run in a child process.

forestatrisk's ``hbm`` C extension never releases the GIL (it links no
PyEval_SaveThread/PyGILState symbols), so running the sampler in-process
freezes every other Python thread — including the whole Solara server — for
the duration of training. ``run_icar_mcmc`` isolates the sampler in a spawned
subprocess and returns only the picklable posterior summaries fit() needs.
"""
import os

import numpy as np
import pandas as pd


def _tiny_mcmc_inputs():
    """Minimal valid sampler inputs: 2x2 grid of spatial cells, rook adjacency."""
    n_neighbors = np.array([2, 2, 2, 2])
    # neighbors of cell 0, cell 1, cell 2, cell 3 (concatenated)
    neighbors = np.array([1, 2, 0, 3, 0, 3, 1, 2])

    rng = np.random.RandomState(42)
    n = 200
    x = rng.normal(size=n)
    cell = rng.randint(0, 4, size=n)
    logit = -0.5 + 1.0 * x
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"y": y, "trial": np.ones(n, dtype=int), "x": x, "cell": cell})
    return df, n_neighbors, neighbors


def test_run_icar_mcmc_executes_in_child_process():
    """Sampler runs off-process and hands back the posteriors fit() stores."""
    from spatialrisk.mlmodels.icar_model import run_icar_mcmc

    df, n_neighbors, neighbors = _tiny_mcmc_inputs()
    result = run_icar_mcmc(
        formula="y + trial ~ x + cell",
        data=df,
        n_neighbors=n_neighbors,
        neighbors=neighbors,
        burnin=100,
        mcmc=100,
        thin=1,
        prior_vrho=-1.0,
        seed=1234,
        verbose=0,
    )

    # The whole point: the GIL-holding sampler ran in another process.
    assert result["worker_pid"] != os.getpid()

    # Posterior summaries have the shapes fit() stores in _ml_model.
    assert result["betas"].shape == (2,)  # Intercept + x (cell column dropped)
    assert result["rho"].shape == (4,)  # one random effect per spatial cell
    assert np.isfinite(result["deviance"])
    assert np.isfinite(result["Vrho"])

    # The posterior summary reads two forestatrisk internals (mod.mcmc and
    # mod._x_design_info) inside a bare `except Exception` that prints and moves
    # on, so an upstream rename would take the credible intervals out silently:
    # summary=None -> fit() falls back to point estimates -> the panel looks
    # exactly like a legacy model, with nothing raising and no test failing.
    # These lines are what makes that upgrade break loudly instead.
    summary = result["posterior_summary"]
    assert summary is not None
    # Column names come from the design, whose LAST column forestatrisk consumes
    # as the 'cell' term — an off-by-one would mislabel every coefficient.
    assert summary["betas"][0]["name"] == "Intercept"
    assert summary["vrho"]["ci_low"] < summary["vrho"]["ci_high"]


def test_run_icar_mcmc_starts_the_chain_at_beta_start():
    """beta_start crosses the process boundary into forestatrisk.

    With no burn-in and a start 1000 units from the mode, a 100-iteration chain
    of unit-scale random-walk steps cannot get anywhere near the data's
    estimates (about -0.7 and 0.6 here): the posterior means stay at the start.
    Dropped on the way, forestatrisk would start from its own default of 0.
    """
    from spatialrisk.mlmodels.icar_model import run_icar_mcmc

    df, n_neighbors, neighbors = _tiny_mcmc_inputs()
    result = run_icar_mcmc(
        formula="y + trial ~ x + cell",
        data=df,
        n_neighbors=n_neighbors,
        neighbors=neighbors,
        burnin=0,
        mcmc=100,
        thin=1,
        prior_vrho=-1.0,
        beta_start=1000.0,
        seed=1234,
        verbose=0,
    )

    assert result["betas"].min() > 500
