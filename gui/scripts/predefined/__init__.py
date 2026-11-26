"""Predefined Earth Engine layer catalogue, one module per theme.

Public API. Theme modules (``terrain``, ``infrastructure``, ``water``,
``forest``, ``climate``, ``admin``) own the builders and their ``ENTRIES``;
``catalogue`` merges them; ``params`` maps names <-> parameter values.
"""

from gui.scripts.predefined.aoi import get_aoi_ee_feature, resolve_aoi_ee
from gui.scripts.predefined.catalogue import (
    CUSTOM_TAG,
    DYNAMIC_ATTRS,
    GROUP_ORDER,
    PREDEFINED_CATALOGUE,
    PREDEFINED_NAMES,
    entry_for_name,
    entry_for_variable,
    resolved_entry,
    validate_entry,
)
from gui.scripts.predefined.display import default_vis, is_editable, vis_override
from gui.scripts.predefined.params import (
    active_param_specs,
    advanced_param_specs,
    aoi_param_defaults,
    basic_param_specs,
    build_predefined_name,
    changed_advanced_count,
    coerce_param_values,
    default_param_values,
    describe_params,
    inactive_to_defaults,
    normalize_value,
    omits_default,
    param_rows,
    param_specs,
    parse_segment,
    resolve_predefined,
    spec_segments,
)
from gui.scripts.predefined.periods import (
    MONTHS,
    month_index,
    period_param,
    period_range,
    period_start,
)

__all__ = [
    "CUSTOM_TAG",
    "DYNAMIC_ATTRS",
    "GROUP_ORDER",
    "PREDEFINED_CATALOGUE",
    "PREDEFINED_NAMES",
    "MONTHS",
    "active_param_specs",
    "advanced_param_specs",
    "aoi_param_defaults",
    "basic_param_specs",
    "build_predefined_name",
    "changed_advanced_count",
    "coerce_param_values",
    "default_vis",
    "default_param_values",
    "describe_params",
    "entry_for_name",
    "entry_for_variable",
    "get_aoi_ee_feature",
    "inactive_to_defaults",
    "is_editable",
    "month_index",
    "normalize_value",
    "omits_default",
    "param_rows",
    "param_specs",
    "parse_segment",
    "period_param",
    "period_range",
    "period_start",
    "resolve_aoi_ee",
    "resolve_predefined",
    "resolved_entry",
    "spec_segments",
    "validate_entry",
    "vis_override",
]
