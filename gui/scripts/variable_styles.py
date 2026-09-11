"""Display styling for local (downloaded) variable rasters.

A predefined GEE variable carries its look through the ``PREDEFINED_CATALOGUE``
(``gui/scripts/predefined_variables.py``), keyed by variable name. That palette
is applied while the layer is still GEE-backed (see ``_styled_layer`` in
``gui/tile/variables_tile.py``). Once the Process step downloads the layer to a
local GeoTIFF (``GEEVar.to_local_raster`` -> ``LocalRasterVar``) the GEE image is
gone, but the variable's ``name`` is preserved — so the same catalogue entry is
still reachable.
Names carrying a parameter suffix (``forest_gfc_tc30``) are mapped back to their
catalogue key by ``resolve_predefined``.

This module rebuilds that catalogue look as a matplotlib ``Colormap`` + value
range so the local raster renders with the *same* colours it had as a GEE layer,
instead of the previous hardcoded grayscale. It mirrors ``prediction_styles`` and
is consumed by ``variable_map.add_raster_var_on_map`` via
``localtileserver.get_leaflet_tile_layer`` (which — unlike ``SepalMap.add_raster``
— honours a Colormap object and pinned ``vmin``/``vmax``).

Post-process outputs (edge / dist / loss / gain) are the exception to the
name-keyed rule: the post-process step *renames* them, so they never hit the
catalogue. Their look comes from ``postprocess_styles`` instead, which is
consulted first here.

Kept free of Solara/ipyvuetify/localtileserver so the mapping stays unit-testable.
"""

from typing import Optional

import matplotlib
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgb


def _colormap_from_palette(hexes, name: str = "variable") -> Colormap:
    """Build a 256-level colormap interpolating a GEE-style hex palette.

    GEE palettes are hex strings without a leading ``#`` (e.g. ``"006633"``);
    ``to_rgb`` requires the ``#``, so it is added when missing. A single-colour
    palette is duplicated because ``from_list`` needs at least two stops. The
    colours are spaced evenly across [0, 1] — the same interpolation GEE applies
    across ``[min, max]`` — so pinning ``vmin``/``vmax`` reproduces the GEE look.
    """
    colors = [to_rgb(h if str(h).startswith("#") else f"#{h}") for h in hexes]
    if len(colors) < 2:
        colors = colors * 2
    return LinearSegmentedColormap.from_list(name or "variable", colors, N=256)


def resolve_variable_style(var) -> dict:
    """Resolve tile-layer styling for a downloaded variable raster.

    Returns ``{"colormap": Colormap, "vmin": float|None, "vmax": float|None,
    "render_kind": str}``. ``vmin``/``vmax`` are ``None`` when the value range
    should auto-stretch to the file's actual min/max (localtileserver does this
    when they are not pinned). ``render_kind`` names which branch was taken, one
    of ``"postprocess_distance"``, ``"postprocess_change"``, ``"custom_palette"``,
    ``"catalogue_palette"``, ``"random_visualizer"``, ``"categorical_fallback"``,
    ``"continuous_fallback"`` —
    the legend builder uses it instead of re-deriving this dispatch logic.

    Selection mirrors ``_styled_layer`` (the GEE-side authority):

    * post-process output (edge/dist/loss/gain, per ``postprocess_styles``) -> its
      QGIS-derived ramp, pinned;
    * the variable's own ``vis_params`` (picked in the Variables modal) -> that
      palette, pinned to its ``min``/``max`` when given (else auto-stretched);
    * predefined catalogue entry with ``vis_params.palette`` -> that palette,
      pinned to its ``min``/``max`` when given (else auto-stretched);
    * predefined ``random_visualizer`` (multi-class categorical, e.g. subj) ->
      a qualitative colormap auto-stretched across the class values;
    * otherwise by ``raster_type``: categorical -> 0=black, 1=white pinned to
      [0, 1]; continuous -> grayscale auto-stretched.
    """
    from gui.scripts.postprocess_styles import resolve_postprocess_style
    from gui.scripts.predefined_variables import (
        PREDEFINED_CATALOGUE,
        resolve_predefined,
    )

    # Post-process outputs (edge/dist/loss/gain) first: they are renamed because they
    # measure a *new* quantity, so a parent's catalogue palette would be meaningless
    # for them — and their names miss the catalogue anyway. None => not one of ours.
    postprocess = resolve_postprocess_style(var)
    if postprocess is not None:
        from gui.scripts.postprocess_styles import resolve_postprocess_legend

        legend = resolve_postprocess_legend(var) or {}
        render_kind = legend.get("render_kind", "postprocess_change")
        return {**postprocess, "render_kind": render_kind}

    name = getattr(var, "name", "") or ""

    # A palette the user picked in the Variables modal (custom layers have no
    # catalogue entry to take one from) wins over everything below. Same shape
    # the GEE side applies verbatim; a missing min/max means auto-stretch.
    own = getattr(var, "vis_params", None) or {}
    if own.get("palette"):
        return {
            "colormap": _colormap_from_palette(own["palette"], name),
            "vmin": own.get("min"),
            "vmax": own.get("max"),
            "render_kind": "custom_palette",
        }

    # Parameterised layers are named <key>_<suffix> (forest_gfc_tc30) — resolve
    # back to the catalogue key so they keep the palette they had as GEE layers.
    cat_key, _params = resolve_predefined(name)
    cat = PREDEFINED_CATALOGUE.get(cat_key) if cat_key else None
    if cat:
        if cat.get("random_visualizer"):
            # Many arbitrary class values -> qualitative ramp (approximation of
            # GEE's per-class randomVisualizer, which has no local equivalent).
            return {
                "colormap": matplotlib.colormaps["tab20"],
                "vmin": None,
                "vmax": None,
                "render_kind": "random_visualizer",
            }
        vis = cat.get("vis_params")
        if vis and vis.get("palette"):
            return {
                "colormap": _colormap_from_palette(vis["palette"], name),
                "vmin": vis.get("min"),
                "vmax": vis.get("max"),
                "render_kind": "catalogue_palette",
            }

    rt = _raster_type_str(var)
    if rt == "categorical":
        return {
            "colormap": _colormap_from_palette(["000000", "ffffff"], name),
            "vmin": 0,
            "vmax": 1,
            "render_kind": "categorical_fallback",
        }
    return {
        "colormap": matplotlib.colormaps["gray"],
        "vmin": None,
        "vmax": None,
        "render_kind": "continuous_fallback",
    }


def _raster_type_str(var) -> Optional[str]:
    """Normalise a variable's ``raster_type`` (enum or str) to a plain string."""
    rt = getattr(var, "raster_type", None)
    if rt is None:
        return ""
    return rt.value if hasattr(rt, "value") else str(rt)
