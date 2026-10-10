"""Predefined Earth Engine layer catalogue, one module per theme.

Public API. Theme modules (``terrain``, ``infrastructure``, ``water``,
``forest``, ``climate``, ``admin``) own the builders and their ``ENTRIES``;
``catalogue`` merges them; ``params`` maps names <-> parameter values.
"""

from gui.scripts.predefined.aoi import get_aoi_ee_feature, resolve_aoi_ee
from gui.scripts.predefined.catalogue import PREDEFINED_CATALOGUE, PREDEFINED_NAMES
from gui.scripts.predefined.params import (
    build_predefined_name,
    coerce_param_values,
    default_param_values,
    param_specs,
    resolve_predefined,
)

__all__ = [
    "PREDEFINED_CATALOGUE",
    "PREDEFINED_NAMES",
    "build_predefined_name",
    "coerce_param_values",
    "default_param_values",
    "get_aoi_ee_feature",
    "param_specs",
    "resolve_aoi_ee",
    "resolve_predefined",
]
