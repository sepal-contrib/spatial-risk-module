"""The predefined layer catalogue, assembled from the theme modules.

Map visualization keys consumed by ``_styled_layer`` (gui/tile/variables_tile.py):
  * ``vis_params`` — a GEE visualization dict. A palette with no ``min``/``max``
    is stretched dynamically to the image's min/max over the AOI. Binary masks
    render filled solid: ``0`` -> first colour (white), ``1`` -> feature colour.
  * ``random_visualizer`` — render via ``image.randomVisualizer()`` (one random
    RGB colour per distinct value); used for multi-class categorical rasters
    whose value range is arbitrary. Takes precedence over ``vis_params``.
  * ``legend_class_keys`` — optional (label key for value 0, label key for 1);
    categorical entries default to legend.class.absent / legend.class.present.
  * ``class_labels`` — ``[{"value", "label_key", "color"}]``: one colour and
    legend chip per class value (render kind ``class_palette``); values not
    listed are transparent. Takes precedence over ``vis_params``.
  * ``unit_key`` — optional i18n key formatting a continuous value with its
    unit, e.g. "legend.unit.m_value" -> "{value} m".
"""

from gui.scripts.predefined import (
    admin,
    climate,
    fire,
    forest,
    forest_change,
    infrastructure,
    mapbiomas,
    terrain,
    water,
)

_THEMES = (
    terrain,
    infrastructure,
    water,
    forest,
    forest_change,
    fire,
    climate,
    admin,
    mapbiomas,
)


def _merge(themes):
    """One dict from the themes' ENTRIES; a key declared twice is a bug."""
    merged = {}
    for module in themes:
        overlap = set(merged) & set(module.ENTRIES)
        if overlap:
            raise ValueError(f"Duplicate predefined keys: {sorted(overlap)}")
        merged.update(module.ENTRIES)
    return merged


PREDEFINED_CATALOGUE = _merge(_THEMES)

PREDEFINED_NAMES = list(PREDEFINED_CATALOGUE.keys())

# Order of the group headers in the Add Variable dropdown. Each later part that
# adds a theme module appends its group here; the catalogue lint requires every
# group to have at least one entry and an i18n label.
GROUP_ORDER = (
    "terrain",
    "infrastructure",
    "water",
    "forest",
    "forest_change",
    "fire",
    "climate",
    "admin",
)


#: Entry fields that may be callables of the full param values (Part D), e.g.
#: ``active_fire``'s raster type depends on its ``metric``. Read them through
#: ``resolved_entry`` / ``entry_for_name``, never from the raw entry.
DYNAMIC_ATTRS = (
    "temporal",
    "years",
    "raster_type",
    "default_scale",
    "vis_params",
    "unit_key",
    "legend_class_keys",
    "class_labels",
)


def _full_params(catalogue_key, params):
    """Defaults overlaid with ``params``, inactive conditional params reset."""
    # params.py imports this module at load time, so import it lazily here.
    from gui.scripts.predefined.params import (
        default_param_values,
        inactive_to_defaults,
    )

    full = {**default_param_values(catalogue_key), **(params or {})}
    return inactive_to_defaults(catalogue_key, full)


def resolved_entry(catalogue_key, params):
    """Shallow copy of an entry with every dynamic field evaluated.

    Each ``DYNAMIC_ATTRS`` field holding a callable is replaced by its value for
    the full params (defaults overlaid with ``params``). Static fields are
    copied as they are, except ``interval_years``, which is [] while
    ``interval_depends_on`` is unmet. Raises KeyError for an unknown key, like
    ``PREDEFINED_CATALOGUE[key]``.
    """
    from gui.scripts.predefined.params import _depends_active

    entry = dict(PREDEFINED_CATALOGUE[catalogue_key])
    full = _full_params(catalogue_key, params)
    for attr in DYNAMIC_ATTRS:
        if callable(entry.get(attr)):
            entry[attr] = entry[attr](full)
    # An interval pair switched off by a param (logging_mapbiomas By year):
    # not an interval target, for every consumer of ``interval_years``.
    if not _depends_active({"depends_on": entry.get("interval_depends_on")}, full):
        entry["interval_years"] = []
    return entry


def entry_for_name(name):
    """``(catalogue key, full params, resolved entry)`` behind a variable name.

    ``(None, {}, None)`` for anything that is not a catalogue variable (custom
    and post-processed layers). The params are always complete: a bare legacy
    name such as ``precipitation`` gets its defaults filled in.
    """
    from gui.scripts.predefined.params import resolve_predefined

    key, params = resolve_predefined(name or "")
    if key is None:
        return None, {}, None
    full = _full_params(key, params)
    return key, full, resolved_entry(key, full)


#: Tag on a layer the user added from a file or asset (``_build_variable``).
CUSTOM_TAG = "custom"


def entry_for_variable(var):
    """``entry_for_name`` for a layer; none for a user's layer named like one."""
    if CUSTOM_TAG in (getattr(var, "tags", None) or ()):
        return None, {}, None
    return entry_for_name(getattr(var, "name", "") or "")


def validate_entry(catalogue_key, params, year):
    """The entry's ``validate(full params, year)`` result, or None.

    Returns ``(i18n key, format kwargs)`` describing why the selection cannot be
    built (e.g. the period is past the source's last month), or None when it
    can, or when the entry declares no hook. Static checks only: it runs in
    Solara renders and must never call Earth Engine.
    """
    entry = PREDEFINED_CATALOGUE.get(catalogue_key or "")
    hook = entry.get("validate") if entry else None
    if not callable(hook):
        return None
    return hook(_full_params(catalogue_key, params), year)
