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
  * ``unit_key`` — optional i18n key formatting a continuous value with its
    unit, e.g. "legend.unit.m_value" -> "{value} m".
"""

from gui.scripts.predefined import (
    admin,
    climate,
    forest,
    infrastructure,
    terrain,
    water,
)

_THEMES = (terrain, infrastructure, water, forest, climate, admin)


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
