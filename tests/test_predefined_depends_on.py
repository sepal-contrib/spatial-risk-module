"""Conditional params: ``depends_on`` gates a spec on its controller.

A spec with ``depends_on`` only counts when its controller holds one of the
allowed values. Inactive specs are hidden in the modal, coerced to their
default, and so left out of the name.

Exercised on a synthetic entry injected with ``monkeypatch.setitem`` so the
tests do not depend on which real layer uses the feature.
"""

import pytest

from gui.scripts.predefined import PREDEFINED_CATALOGUE
from gui.scripts.predefined.params import (
    active_param_specs,
    build_predefined_name,
    coerce_param_values,
    inactive_to_defaults,
    resolve_predefined,
)

KEY = "demo_dry"

DEMO = {
    "label_key": "vars.predefined.precipitation",
    "description_key": "vars.predefined_info.precipitation",
    "var_type": "GEEVar",
    "raster_type": "continuous",
    "temporal": False,
    "group": "climate",
    "get_image": lambda aoi, year=None, metric="dsr", threshold=1, window=365: None,
    "params": [
        {
            "key": "metric",
            "label_key": "x",
            "hint_key": "x",
            "type": "choice",
            "default": "dsr",
            "options": ["dsr", "drydays", "raintotal"],
            "option_label_key_prefix": "x.",
            "suffix_prefix": "",
            "omit_default": True,
        },
        {
            "key": "threshold",
            "label_key": "x",
            "hint_key": "x",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 50,
            "suffix_prefix": "t",
            "omit_default": True,
            "depends_on": {"metric": ["dsr", "drydays"]},
        },
        {
            "key": "window",
            "label_key": "x",
            "hint_key": "x",
            "type": "int",
            "default": 365,
            "min": 7,
            "max": 999,
            "suffix_prefix": "w",
            "omit_default": True,
        },
    ],
}


@pytest.fixture(autouse=True)
def demo(monkeypatch):
    """Demo."""
    monkeypatch.setitem(PREDEFINED_CATALOGUE, KEY, DEMO)


def _keys(specs):
    return [s["key"] for s in specs]


def test_active_specs_follow_the_controller():
    """Active specs follow the controller."""
    assert _keys(active_param_specs(KEY, {"metric": "dsr"})) == [
        "metric",
        "threshold",
        "window",
    ]
    assert _keys(active_param_specs(KEY, {"metric": "raintotal"})) == [
        "metric",
        "window",
    ]


def test_missing_controller_counts_as_its_default():
    """Missing controller counts as its default."""
    assert "threshold" in _keys(active_param_specs(KEY, {}))


def test_raw_form_values_work():
    """Raw form values work."""
    raw = {"metric": "drydays", "threshold": "abc", "window": ""}
    assert "threshold" in _keys(active_param_specs(KEY, raw))


def test_inactive_to_defaults_resets_only_inactive_params():
    """Inactive to defaults resets only inactive params."""
    values = {"metric": "raintotal", "threshold": 5, "window": 30}
    assert inactive_to_defaults(KEY, values) == {
        "metric": "raintotal",
        "threshold": 1,
        "window": 30,
    }
    assert values["threshold"] == 5  # the input is not mutated
    active = {"metric": "dsr", "threshold": 5, "window": 30}
    assert inactive_to_defaults(KEY, active) == active


def test_build_leaves_an_inactive_param_out_of_the_name():
    """Build leaves an inactive param out of the name."""
    assert (
        build_predefined_name(
            KEY, {"metric": "raintotal", "threshold": 5, "window": 365}
        )
        == "demo_dry_raintotal"
    )
    assert (
        build_predefined_name(KEY, {"metric": "dsr", "threshold": 5, "window": 365})
        == "demo_dry_t5"
    )


def test_coerce_ignores_an_invalid_hidden_value():
    """A stale or blank hidden field must not block submit."""
    values, bad = coerce_param_values(
        KEY, {"metric": "raintotal", "threshold": "", "window": "30"}
    )
    assert bad is None
    assert values == {"metric": "raintotal", "threshold": 1, "window": 30}


def test_coerce_still_checks_an_active_dependent():
    """Coerce still checks an active dependent."""
    values, bad = coerce_param_values(
        KEY, {"metric": "dsr", "threshold": "", "window": "30"}
    )
    assert values == {}
    assert bad["key"] == "threshold"


def test_resolve_rejects_a_segment_of_an_inactive_param():
    """Resolve rejects a segment of an inactive param."""
    assert resolve_predefined("demo_dry_raintotal_t5") == (None, {})


def test_resolve_fills_the_default_of_an_inactive_param():
    """Resolve fills the default of an inactive param."""
    assert resolve_predefined("demo_dry_raintotal") == (
        KEY,
        {"metric": "raintotal", "threshold": 1, "window": 365},
    )
    assert resolve_predefined("demo_dry_drydays_t5") == (
        KEY,
        {"metric": "drydays", "threshold": 5, "window": 365},
    )


def test_param_rows_list_only_active_params():
    """Param rows list only active params."""
    from gui.scripts.predefined.params import param_rows

    hidden = {"metric": "raintotal", "threshold": 1, "window": 365}
    shown = {"metric": "dsr", "threshold": 5, "window": 365}
    assert [text for _, text in param_rows(KEY, hidden)][1:] == ["365"]
    assert [text for _, text in param_rows(KEY, shown)][1:] == ["5", "365"]
