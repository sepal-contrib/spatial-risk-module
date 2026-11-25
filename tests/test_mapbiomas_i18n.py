"""Every MapBiomas message key exists in all four locales with en's placeholders.

The catalogue-wide tests in test_i18n only resolve keys in the active locale,
so this file reads the JSON catalogs of every locale directly.
"""

import json
import re

import pytest

from gui import i18n

LOCALES = ("en", "es-ES", "fr", "pt-BR")
REGIONS = (
    "brazil",
    "amazon",
    "bolivia",
    "colombia",
    "ecuador",
    "paraguay",
    "peru",
    "venezuela",
)
GROUPS = ("forest", "natural", "farming", "bare", "water")
PLACEHOLDER = re.compile(r"\{(\w+)\}")

ROUND1_KEYS = (
    [
        f"vars.{section}.{key}"
        for section in ("predefined", "predefined_info")
        for key in ("mapbiomas_lulc", "deforestation_mapbiomas", "mapbiomas_fire")
    ]
    + [
        f"vars.modal.{key}"
        for key in (
            "param_mapbiomas_region",
            "param_mapbiomas_region_hint",
            "param_lulc_mode",
            "param_lulc_mode_hint",
            "lulc_mode_mask",
            "lulc_mode_groups",
            "param_lulc_groups",
            "param_lulc_groups_hint",
            "param_fire_product",
            "param_fire_product_hint",
            "fire_product_burned",
            "fire_product_since",
            "fire_product_frequency",
            "param_fire_window",
            "param_fire_window_hint",
            "param_mapbiomas_fire_source",
            "param_mapbiomas_fire_source_hint",
            "mapbiomas_fire_source_collection",
            "mapbiomas_fire_source_monitor",
            "param_defo_vegetation",
            "param_defo_vegetation_hint",
            "defo_vegetation_primary",
            "defo_vegetation_all",
        )
    ]
    + [f"vars.mapbiomas.region_{r}" for r in REGIONS]
    + [f"vars.mapbiomas.group_{g}" for g in GROUPS]
    + [
        f"vars.mapbiomas.{key}"
        for key in (
            "error_outside_region",
            "error_year",
            "error_period",
            "error_not_available",
            "error_interval_range",
        )
    ]
    + ["legend.unit.years_value"]
)

ROUND2_KEYS = (
    [
        f"vars.{section}.{key}"
        for section in ("predefined", "predefined_info")
        for key in ("logging_mapbiomas", "mapbiomas_water")
    ]
    + [
        f"vars.modal.{key}"
        for key in (
            "param_logging_timing",
            "param_logging_timing_hint",
            "logging_timing_yearly",
            "logging_timing_period",
        )
    ]
    + ["legend.unit.days_value", "legend.class.logged"]
)

# The four-indicator MapBiomas degradation layer was dropped for selective logging.
REMOVED_KEYS = [
    "vars.predefined.mapbiomas_degradation",
    "vars.predefined_info.mapbiomas_degradation",
    "vars.modal.param_degradation_variable",
    "vars.modal.param_degradation_variable_hint",
] + [
    f"vars.modal.degradation_variable_{v}"
    for v in ("edge", "secondary", "logging", "canopy")
]

EXPECTED_FIELDS = {
    "vars.mapbiomas.error_outside_region": {"region", "pct", "min"},
    "vars.mapbiomas.error_year": {"region", "first", "last"},
    "vars.mapbiomas.error_period": {"region", "first", "last"},
    "vars.mapbiomas.error_not_available": {"region"},
    "vars.mapbiomas.error_interval_range": {"region", "first", "last"},
    "legend.unit.years_value": {"value"},
    "legend.unit.days_value": {"value"},
}


def _lookup(locale, dotted):
    """Value of ``dotted`` in any catalog file of ``locale`` (None when absent)."""
    for path in sorted((i18n.MESSAGES_DIR / locale).glob("*.json")):
        node = json.loads(path.read_text(encoding="utf-8"))
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, str):
            return node
    return None


@pytest.mark.parametrize("key", ROUND1_KEYS + ROUND2_KEYS)
def test_key_in_every_locale_with_en_placeholders(key):
    """Key in every locale with en placeholders."""
    en = _lookup("en", key)
    assert en, f"{key} missing in en"
    for locale in LOCALES[1:]:
        value = _lookup(locale, key)
        assert value, f"{key} missing in {locale}"
        assert set(PLACEHOLDER.findall(value)) == set(PLACEHOLDER.findall(en)), (
            key,
            locale,
        )


@pytest.mark.parametrize("key", REMOVED_KEYS)
def test_removed_keys_are_gone_from_every_locale(key):
    """Removed keys are gone from every locale."""
    assert all(_lookup(locale, key) is None for locale in LOCALES), key


@pytest.mark.parametrize("key", sorted(EXPECTED_FIELDS))
def test_messages_carry_the_fields_the_code_passes(key):
    """Messages carry the fields the code passes."""
    assert set(PLACEHOLDER.findall(_lookup("en", key))) == EXPECTED_FIELDS[key]


def test_outside_region_message_renders():
    """Outside region message renders."""
    msg = i18n.t("vars.mapbiomas.error_outside_region", region="Peru", pct=42, min=95)
    assert "42" in msg and "Peru" in msg and "95" in msg
