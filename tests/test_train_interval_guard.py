"""MW / JNR derive the period length from the target name.

A single-period target (burned_area_sep) must be refused in the Train dialog,
not deep in fit.
"""

import inspect

import pytest

from gui.i18n import t
from gui.scripts.model_registry import interval_target_error


@pytest.mark.parametrize("model_key", ["jnr", "mw", "jnr_v2", "mw_custom"])
def test_interval_families_need_a_two_year_target(model_key):
    """Interval families need a two year target."""
    assert interval_target_error(model_key, "burned_area_sep") == (
        "tiles.train.error_needs_interval_target",
        {"name": "burned_area_sep"},
    )
    assert interval_target_error(model_key, "forest_loss_2015_2020") is None


@pytest.mark.parametrize("model_key", ["glm", "rf", "icar"])
def test_ml_families_take_any_target(model_key):
    """Ml families take any target."""
    assert interval_target_error(model_key, "burned_area_sep") is None


def test_message_names_the_target():
    """Message names the target."""
    key, kwargs = interval_target_error("mw", "active_fire_days")
    assert "active_fire_days" in t(key, **kwargs)


def test_train_dialog_validate_checks_the_target():
    """Train dialog validate checks the target."""
    from gui.widget import model_form_dialog

    src = " ".join(inspect.getsource(model_form_dialog.ModelFormDialog).split())
    assert "interval_target_error(selected_key" in src
