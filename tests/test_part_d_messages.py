"""Part D strings exist, non-empty, in every locale (parity is test_i18n's job)."""

import json
from pathlib import Path

import pytest

LOCALES = ("en", "es-ES", "fr", "pt-BR")
MESSAGES = Path(__file__).resolve().parents[1] / "gui" / "messages"
MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)

LAYERS = ("burned_area", "active_fire", "fire_history", "dry_spell")
MODAL = (
    ["param_period", "param_period_hint", "period_year"]
    + [f"period_{m}" for m in MONTHS]
    + [
        "param_burn_sensor",
        "param_burn_sensor_hint",
        "burn_sensor_modis",
        "burn_sensor_viirs",
    ]
    + ["param_fire_metric", "param_fire_metric_hint"]
    + [f"fire_metric_{v}" for v in ("any", "days", "frp")]
    + ["param_fire_confidence", "param_fire_confidence_hint"]
    + [f"fire_confidence_{v}" for v in ("low", "nominal", "high")]
    + ["param_fire_source", "param_fire_source_hint"]
    + [f"fire_source_{v}" for v in ("modis", "viirs", "firms")]
    + ["param_history_metric", "param_history_metric_hint"]
    + ["history_metric_count", "history_metric_since"]
    + ["param_history_window", "param_history_window_hint"]
    + ["param_history_source", "param_history_source_hint"]
    + ["history_source_burned", "history_source_active"]
    + ["param_dry_metric", "param_dry_metric_hint"]
    + [f"dry_metric_{v}" for v in ("dsr", "drydays", "raintotal")]
    + ["param_rain_threshold", "param_rain_threshold_hint"]
    + ["param_dry_window", "param_dry_window_hint"]
    + [
        "param_rain_source",
        "param_rain_source_hint",
        "rain_source_era",
        "rain_source_chirps",
    ]
    + ["error_period_unavailable", "error_frp_firms"]
)
KEYS = (
    [("vars", f"vars.predefined.{k}") for k in LAYERS]
    + [("vars", f"vars.predefined_info.{k}") for k in LAYERS]
    + [("vars", "vars.predefined_group.fire")]
    + [("vars", f"vars.modal.{k}") for k in MODAL]
    + [
        ("common", f"legend.class.{k}")
        for k in ("burned", "unburned", "fire", "no_fire")
    ]
    + [
        ("common", f"legend.unit.{k}")
        for k in ("days_value", "mw_value", "years_value", "mm_value")
    ]
    + [("tiles", "tiles.train.error_needs_interval_target")]
)


def _lookup(locale, filename, dotted):
    node = json.loads((MESSAGES / locale / f"{filename}.json").read_text())
    for part in dotted.split("."):
        node = node[part]
    return node


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("filename, dotted", KEYS)
def test_part_d_key_exists(locale, filename, dotted):
    """Part d key exists."""
    assert str(_lookup(locale, filename, dotted)).strip()


@pytest.mark.parametrize("locale", LOCALES)
def test_unavailable_message_names_every_placeholder(locale):
    """Unavailable message names every placeholder."""
    text = _lookup(locale, "vars", "vars.modal.error_period_unavailable")
    for field in ("{year}", "{start}", "{end}", "{first}", "{last}"):
        assert field in text, (locale, field)


@pytest.mark.parametrize("locale", LOCALES)
def test_climate_texts_mention_the_month(locale):
    """The rewritten temperature/precipitation texts cover monthly periods."""
    hint = _lookup(locale, "vars", "vars.modal.param_aggregation_hint")
    info = _lookup(locale, "vars", "vars.predefined_info.precipitation")
    assert "CHIRPS" in info
    assert not hint.startswith("How the 12 monthly values are combined")
