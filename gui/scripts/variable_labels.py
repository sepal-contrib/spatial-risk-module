"""Display labels for processed layers: ``defor (2015)``, the bare name if static.

Selects keep storage keys (``defor_2015``) as their values and show these
labels as text, so every place that lists layers names a year the same way.
"""

from typing import Iterable, List, Optional


def year_label(name: str, year: Optional[int]) -> str:
    """``name (year)`` for one year of a temporal variable, else ``name``."""
    return name if year is None else f"{name} ({year})"


def layer_label(project, key: Optional[str]) -> Optional[str]:
    """Label of the processed variable stored under ``key``.

    Falls back to ``key`` itself when the project no longer has it (a layer
    deleted after a sample or prediction recorded it).
    """
    variables = getattr(project, "processed_variables", None) or {}
    var = variables.get(key) if key else None
    if var is None:
        return key
    return year_label(getattr(var, "name", None) or key, getattr(var, "year", None))


def layer_items(project, keys: Iterable[str]) -> List[dict]:
    """Select items for storage ``keys``: labelled text, the key as value."""
    return [{"text": layer_label(project, k), "value": k} for k in keys]
