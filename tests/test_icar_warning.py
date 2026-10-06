"""The New model dialog's slow-iCAR warning: inputs read from the form.

The estimate itself is pinned in test_icar_cost.py; this covers what the
dialog feeds it (sample size, iterations typed as text) and when it speaks.
"""

from gui.i18n import t

# See test_manage_projects_render: warm the translator before first use.
t("common.cancel")

from gui.scripts.icar_warning import (  # noqa: E402
    icar_iterations,
    icar_slow_estimate,
    icar_slow_message,
    sample_size,
)
from spatialrisk.sample import Sample  # noqa: E402

SLOW_FORMULA = "I(defor) + trial ~ " + " + ".join(
    [f"scale(x{i})" for i in range(6)] + ["C(subj)"]
)
LEVELS = {"subj": 107}


def _sample(n_samples=20_000, class_counts=None):
    return Sample(
        name="s1",
        raster_var_name="forest",
        strategy="random",
        n_samples=n_samples,
        class_counts=class_counts or {},
    )


def test_sample_size_counts_the_points_actually_drawn():
    """Class counts (what sampling drew) win over the requested count."""
    assert sample_size(_sample(10_000, {"0": 9_000, "1": 11_000})) == 20_000


def test_sample_size_falls_back_to_the_requested_count():
    """Without class counts the requested count stands in."""
    assert sample_size(_sample(10_000)) == 10_000


def test_iterations_read_text_fields_as_numbers():
    """Typed strings and stored ints both count."""
    # The form's text fields store what the user typed, as a string.
    assert icar_iterations({"burnin": "1000", "mcmc": 500}) == 1500


def test_unreadable_iterations_fall_back_to_the_defaults():
    """Garbage in the fields falls back to the 4000 + 4000 defaults."""
    assert icar_iterations({"burnin": "abc", "mcmc": ""}) == 8000


def test_slow_many_level_categorical_is_reported():
    """A slow run driven by subj is worth a warning."""
    est = icar_slow_estimate(SLOW_FORMULA, LEVELS, _sample(), {})

    assert est is not None
    assert est.culprits[0].name == "subj"


def test_removing_the_categorical_from_the_formula_silences_it():
    """Deleting C(subj) from the formula clears the warning."""
    formula = SLOW_FORMULA.replace(" + C(subj)", "")

    assert icar_slow_estimate(formula, LEVELS, _sample(), {}) is None


def test_fewer_iterations_can_bring_it_under_the_threshold():
    """Iterations scale the estimate linearly."""
    # 113 coefficients at 20k samples: about 0.52 s per iteration -> 500 is ~4 min
    assert (
        icar_slow_estimate(
            SLOW_FORMULA, LEVELS, _sample(), {"burnin": 250, "mcmc": 250}
        )
        is None
    )


def test_no_sample_selected_gives_no_warning():
    """Without a sample there is nothing to estimate."""
    assert icar_slow_estimate(SLOW_FORMULA, LEVELS, None, {}) is None


def test_message_names_the_variable_and_the_numbers():
    """The text carries the duration, the culprit, its levels and the speedup."""
    est = icar_slow_estimate(SLOW_FORMULA, LEVELS, _sample(), {})

    msg = icar_slow_message(est)

    # 4124 s -> 69 min; dropping subj makes it about 77x faster.
    assert "69 min" in msg
    assert "subj" in msg and "107" in msg and "77" in msg
    assert "C(subj)" in msg


def test_message_switches_to_hours_for_long_runs():
    """Runs past two hours read in hours."""
    est = icar_slow_estimate(SLOW_FORMULA, LEVELS, _sample(60_000), {})

    # 3 x 4124 s = 12372 s -> 3 h
    assert "3 h" in icar_slow_message(est)
