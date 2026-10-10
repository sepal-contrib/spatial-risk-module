"""Predefined Earth Engine layer catalogue, one module per theme.

Public API. Theme modules (``terrain``, ``infrastructure``, ``water``,
``forest``, ``climate``, ``admin``) own the builders and their ``ENTRIES``;
``catalogue`` merges them; ``params`` maps names <-> parameter values.
"""

from gui.scripts.predefined.aoi import get_aoi_ee_feature, resolve_aoi_ee
from gui.scripts.predefined.catalogue import (
    GROUP_ORDER,
    PREDEFINED_CATALOGUE,
    PREDEFINED_NAMES,
)
from gui.scripts.predefined.params import (
    advanced_param_specs,
    basic_param_specs,
    build_predefined_name,
    changed_advanced_count,
    coerce_param_values,
    default_param_values,
    describe_params,
    normalize_value,
    omits_default,
    param_rows,
    param_specs,
    parse_segment,
    resolve_predefined,
    spec_segments,
)

__all__ = [
    "GROUP_ORDER",
    "PREDEFINED_CATALOGUE",
    "PREDEFINED_NAMES",
    "advanced_param_specs",
    "basic_param_specs",
    "build_predefined_name",
    "changed_advanced_count",
    "coerce_param_values",
    "default_param_values",
    "describe_params",
    "get_aoi_ee_feature",
    "normalize_value",
    "omits_default",
    "param_rows",
    "param_specs",
    "parse_segment",
    "resolve_aoi_ee",
    "resolve_predefined",
    "spec_segments",
]
