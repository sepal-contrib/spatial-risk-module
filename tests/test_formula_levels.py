"""Categorical ``levels=[...]`` are a background safety measure, not UI copy.

The dialog shows/edits bare ``C(x)`` terms; ``inject_categorical_levels``
re-arms them at fit time (predict re-parses the stored formula string
against the samples CSV, so levels must be present there), and
``strip_categorical_levels`` shortens stored formulas for display.
"""

import types

import pytest

import spatialrisk.far_helpers as fh
from spatialrisk.far_helpers import (
    generate_patsy_formula,
    inject_categorical_levels,
    strip_categorical_levels,
)


def _var(name, raster_type):
    return types.SimpleNamespace(name=name, raster_type=raster_type)


def _dataset():
    return types.SimpleNamespace(
        target=types.SimpleNamespace(name="fcc"),
        features=[
            _var("altitude", "continuous"),
            _var("pa", "categorical"),
        ],
    )


def test_generate_without_levels_emits_bare_c(monkeypatch):
    """Display formula uses bare C(x) and reads no raster."""
    called = []
    monkeypatch.setattr(
        fh, "get_categorical_levels", lambda v: called.append(v) or [0, 1]
    )
    formula = generate_patsy_formula(_dataset(), include_levels=False)
    assert formula == "I(fcc) + trial ~ scale(altitude) + C(pa)"
    assert not called  # no raster read for the display formula


def test_generate_with_levels_still_default(monkeypatch):
    """Default generation still embeds the level domain."""
    monkeypatch.setattr(fh, "get_categorical_levels", lambda v: [0, 1])
    formula = generate_patsy_formula(_dataset())
    assert "C(pa, levels=[0, 1])" in formula


def test_inject_adds_levels_to_bare_c(monkeypatch):
    """Bare C(x) terms gain levels from the raster at fit time."""
    monkeypatch.setattr(fh, "get_categorical_levels", lambda v: [0, 1])
    out = inject_categorical_levels(
        "I(fcc) + trial ~ scale(altitude) + C(pa)", _dataset()
    )
    assert out == "I(fcc) + trial ~ scale(altitude) + C(pa, levels=[0, 1])"


def test_inject_leaves_explicit_levels_alone(monkeypatch):
    """User-supplied explicit levels are preserved verbatim."""
    monkeypatch.setattr(fh, "get_categorical_levels", lambda v: [0, 1, 2])
    formula = "I(fcc) + trial ~ C(pa, levels=[0, 1])"
    assert inject_categorical_levels(formula, _dataset()) == formula


def test_inject_refuses_a_layer_without_stored_categories():
    """A bare C(x) would train, then break at prediction: refuse up front."""
    formula = "I(fcc) + trial ~ C(pa)"
    with pytest.raises(fh.MissingCategoricalLevels, match="'pa'"):
        inject_categorical_levels(formula, _dataset())


def test_inject_ignores_lhs_and_non_categorical(monkeypatch):
    """Injection touches only categorical RHS terms."""
    monkeypatch.setattr(fh, "get_categorical_levels", lambda v: [0, 1])
    # 'pa' on the LHS must not be touched; altitude is continuous.
    out = inject_categorical_levels("C(pa) ~ scale(altitude)", _dataset())
    assert out == "C(pa) ~ scale(altitude)"


def test_strip_removes_levels_lists():
    """Stored long form renders short for display."""
    long = "I(fcc) + trial ~ scale(altitude) + C(pa, levels=[0, 1, 255])"
    assert strip_categorical_levels(long) == (
        "I(fcc) + trial ~ scale(altitude) + C(pa)"
    )


def test_strip_is_noop_without_levels():
    """Already-short formulas pass through unchanged."""
    short = "I(fcc) + trial ~ scale(altitude) + C(pa)"
    assert strip_categorical_levels(short) == short
