"""Language-neutral legend specifications and their pysepal rendering.

The registry (``gui/scripts/legend_registry.py``) stores ``LegendSpec`` objects,
not translated strings: ``Page()`` converts the selected spec to pysepal's
``LegendData`` on every render, so legends re-translate live when the locale
changes. Builders in this module therefore emit i18n *keys* (``Label.key``) and
never import ``gui.i18n``.

Kept free of Solara / ipyvuetify / ipyleaflet / localtileserver so it stays
unit-testable, like its ``*_styles.py`` siblings.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Tuple

#: How many stops a gradient is sampled into. pysepal's Legend.vue spaces the
#: supplied colours evenly ((i / (len - 1)) * 100), so a coarse sample would
#: smear the QGIS ramps, whose nodes sit far from even positions (the MW palette
#: puts three stops inside the first 3% of 1..65535). 256 mirrors the same LUT
#: size localtileserver renders the tiles from, so legend and tiles agree.
GRADIENT_STOPS = 256


@dataclass(frozen=True)
class Label:
    """A piece of legend text: either an i18n key or a ready-made literal.

    ``args`` is a tuple of ``(name, value)`` pairs rather than a dict so the
    dataclass stays hashable/frozen; it is expanded into ``t(key, **args)``.
    """

    key: str = ""
    literal: str = ""
    args: Tuple[Tuple[str, object], ...] = ()


@dataclass(frozen=True)
class LegendSpec:
    """What to draw for one layer, before translation.

    ``kind``:
        * ``"gradient"`` — a colour bar; ``colors`` are its stops and ``labels``
          its two endpoints.
        * ``"chips"``    — one colour chip per class; ``colors`` and ``labels``
          are zipped pairwise.
        * ``"note"``     — no colours at all, just ``labels`` as text rows (used
          for GEE ``randomVisualizer`` layers, whose colours are assigned
          server-side at random and cannot be shown).
    """

    kind: str
    title: Label = field(default_factory=Label)
    colors: Tuple[str, ...] = ()
    labels: Tuple[Label, ...] = ()


def resolve_label(label: Label, t: Callable[..., str]) -> str:
    """Translate one ``Label``; a key wins over a literal."""
    if label.key:
        return t(label.key, **dict(label.args))
    return label.literal


def gradient_colors(cmap, n: int = GRADIENT_STOPS) -> Tuple[str, ...]:
    """Sample a matplotlib ``Colormap`` into ``n`` evenly spaced hex stops."""
    from matplotlib.colors import to_hex

    if n < 2:
        raise ValueError("a gradient needs at least two stops")
    return tuple(to_hex(cmap(i / (n - 1))) for i in range(n))


def to_legend_data(spec: LegendSpec, t: Callable[..., str]):
    """Build a FRESH pysepal ``LegendData`` from ``spec``.

    Always constructs new objects: ``LegendData`` holds mutable lists, and a
    shared/mutated instance would trip reacton's prop-equality bailout.
    """
    from pysepal.solara.components.legend import (
        DiscreteEntry,
        GradientEntry,
        LegendData,
    )

    title = resolve_label(spec.title, t)

    if spec.kind == "gradient":
        return LegendData(
            gradients=[
                GradientEntry(
                    colors=list(spec.colors),
                    labels=[resolve_label(label, t) for label in spec.labels],
                    title=title,
                )
            ]
        )

    # chips / note — a chip-less first row carries the title, which DiscreteEntry
    # renders as a plain text row (pysepal's own "totals row" idiom).
    items = [DiscreteEntry(title, "")] if title else []
    for index, label in enumerate(spec.labels):
        color = spec.colors[index] if index < len(spec.colors) else ""
        items.append(DiscreteEntry(resolve_label(label, t), color))
    return LegendData(items=items)


def format_number(value: float) -> str:
    """Format a legend endpoint: 3 significant digits, no trailing zeros.

    ``30`` stays ``"30"`` (not ``"30.0"``); ``0.012456`` becomes ``"0.0125"``.

    Uses fixed-point rather than ``:g`` formatting: Python's ``:g`` switches to
    scientific notation once the exponent reaches the requested precision (e.g.
    ``f"{1234.0:.3g}"`` -> ``"1.23e+03"``), which would misrender the larger
    endpoints legends actually see.
    """
    value = float(value)
    if not math.isfinite(value):
        return ""
    if value == 0:
        return "0"
    digits = 3
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(digits - magnitude - 1, 0)
    rounded = round(value, decimals)
    if decimals <= 0:
        return f"{rounded:.0f}"
    text = f"{rounded:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def prediction_spec(model_key: str, display_palette: str = None) -> LegendSpec:
    """Legend for a prediction raster: its QGIS ramp with semantic endpoints.

    The pixel values are rescaled integers (far.misc.rescale -> 1..65535, jnr
    1001..30999), which mean nothing to a reader, so the endpoints are labelled
    as risk instead. ``display_palette`` mirrors ``resolve_display_style`` and is
    what imported predictions carry.
    """
    from gui.scripts.prediction_styles import resolve_display_style

    style = resolve_display_style(model_key, display_palette)
    return LegendSpec(
        kind="gradient",
        title=Label(key="legend.prediction.title"),
        colors=gradient_colors(style["colormap"]),
        labels=(Label(key="legend.risk.low"), Label(key="legend.risk.high")),
    )


def density_spec(vmin: float, vmax: float) -> LegendSpec:
    """Legend for a deforestation-density raster, labelled with its real range.

    ``add_density_on_map`` stretches the ramp to the raster's own min/max and
    hands that range here, so no second band read is needed. A degenerate range
    (constant or all-nodata raster) falls back to Low/High rather than printing
    the same number twice.
    """
    from gui.scripts.density_map import density_colormap

    if vmax > vmin:
        labels = (
            Label(literal=format_number(vmin)),
            Label(literal=format_number(vmax)),
        )
    else:
        labels = (Label(key="legend.range.low"), Label(key="legend.range.high"))

    return LegendSpec(
        kind="gradient",
        title=Label(key="legend.density.title"),
        colors=gradient_colors(density_colormap()),
        labels=labels,
    )


#: Default chip labels for a 0/1 presence mask, overridable per catalogue entry
#: via ``legend_class_keys``.
DEFAULT_CLASS_KEYS = ("legend.class.absent", "legend.class.present")


def _catalogue_entry(var):
    """The PREDEFINED_CATALOGUE entry behind ``var``'s name, or None.

    Parameterised names (``forest_gfc_tc30``) resolve back to their catalogue
    key, the same way the style resolvers do.
    """
    from gui.scripts.predefined_variables import (
        PREDEFINED_CATALOGUE,
        resolve_predefined,
    )

    name = getattr(var, "name", "") or ""
    cat_key, _params = resolve_predefined(name)
    return PREDEFINED_CATALOGUE.get(cat_key) if cat_key else None


def _is_categorical(render_kind: str, cat, var=None) -> bool:
    """Does this layer draw discrete classes rather than a continuous ramp?

    A catalogue palette can be either (``slope`` is continuous, ``protected_area``
    is categorical), so the entry's own ``raster_type`` decides there; a
    user-picked palette (``custom_palette``) likewise follows the variable's
    ``raster_type``.
    """
    if render_kind == "postprocess_change":
        return True
    if render_kind == "categorical_fallback":
        return True
    if render_kind == "catalogue_palette" and cat is not None:
        return cat.get("raster_type") == "categorical"
    if render_kind == "custom_palette":
        rt = getattr(var, "raster_type", None)
        rt = rt.value if hasattr(rt, "value") else (str(rt) if rt is not None else "")
        return rt == "categorical"
    return False


def _value_labels(vmin, vmax, unit_key: str):
    """Endpoint labels for a gradient: real numbers, or Low/High when unpinned."""
    if (
        vmin is None
        or vmax is None
        or not math.isfinite(float(vmin))
        or not math.isfinite(float(vmax))
        or float(vmax) <= float(vmin)
    ):
        return (Label(key="legend.range.low"), Label(key="legend.range.high"))
    key = unit_key or "legend.unit.plain_value"
    return (
        Label(key=key, args=(("value", format_number(vmin)),)),
        Label(key=key, args=(("value", format_number(vmax)),)),
    )


def variable_spec(
    *, colors, vmin, vmax, render_kind: str, var, title: Label
) -> LegendSpec:
    """Legend for a variable layer, dispatched on the style resolver's verdict.

    Args:
        colors: Chip colours (categorical) or gradient stops (continuous),
            already hex-prefixed.
        vmin: The pinned lower value, or None when the layer is auto-stretched
            by localtileserver.
        vmax: The pinned upper value, or None when the layer is auto-stretched
            by localtileserver.
        render_kind: One of the six values reported by
            ``variable_styles.resolve_variable_style`` / ``_styled_layer``.
        var: The variable object (used for its catalogue lookup and, for
            post-process rasters, its classification).
        title: Dropdown/heading text for this layer.
    """
    if render_kind == "random_visualizer":
        # GEE assigns these colours server-side at random; there is no palette
        # to show, so the legend says so instead of inventing one.
        return LegendSpec(
            kind="note",
            title=title,
            labels=(Label(key="legend.note.random_classes"),),
        )

    cat = _catalogue_entry(var)

    if render_kind == "postprocess_change":
        from gui.scripts.postprocess_styles import resolve_postprocess_legend

        legend = resolve_postprocess_legend(var) or {}
        class_colors = tuple(legend.get("class_colors") or colors)
        class_keys = tuple(legend.get("class_keys") or DEFAULT_CLASS_KEYS)
        return LegendSpec(
            kind="chips",
            title=title,
            colors=class_colors,
            labels=tuple(Label(key=key) for key in class_keys),
        )

    if _is_categorical(render_kind, cat, var):
        class_keys = tuple((cat or {}).get("legend_class_keys") or DEFAULT_CLASS_KEYS)
        return LegendSpec(
            kind="chips",
            title=title,
            colors=tuple(colors[: len(class_keys)]),
            labels=tuple(Label(key=key) for key in class_keys),
        )

    unit_key = (cat or {}).get("unit_key", "")
    if render_kind == "postprocess_distance":
        from gui.scripts.postprocess_styles import resolve_postprocess_legend

        unit_key = (resolve_postprocess_legend(var) or {}).get(
            "unit_key", ""
        ) or unit_key

    return LegendSpec(
        kind="gradient",
        title=title,
        colors=tuple(colors),
        labels=_value_labels(vmin, vmax, unit_key),
    )


def variable_spec_from_style(style: dict, var, title: Label) -> LegendSpec:
    """Adapter for local rasters: build from ``resolve_variable_style``'s dict."""
    render_kind = style.get("render_kind", "continuous_fallback")
    if render_kind == "random_visualizer":
        colors = ()
    elif render_kind == "postprocess_change":
        # variable_spec discards `colors` for this render_kind in favour of
        # resolve_postprocess_legend's own class_colors, so there is nothing
        # useful to sample from the colormap here.
        colors = ()
    elif _is_categorical(render_kind, _catalogue_entry(var), var):
        # Chips need the palette's own colours, not a sampled ramp: read the
        # colormap at its ends (0/1 masks) so the chips match the tiles.
        colors = (
            _hex_at(style["colormap"], 0.0),
            _hex_at(style["colormap"], 1.0),
        )
    else:
        colors = gradient_colors(style["colormap"])

    return variable_spec(
        colors=colors,
        vmin=style.get("vmin"),
        vmax=style.get("vmax"),
        render_kind=render_kind,
        var=var,
        title=title,
    )


def variable_spec_from_vis(
    vis: dict, render_kind: str, var, title: Label
) -> LegendSpec:
    """Adapter for GEE layers: build from the ``vis`` dict ``_styled_layer`` used."""
    palette = [_with_hash(c) for c in (vis or {}).get("palette", [])]
    if render_kind == "random_visualizer" or not palette:
        colors = tuple(palette)
    elif _is_categorical(render_kind, _catalogue_entry(var), var):
        colors = tuple(palette)
    else:
        from gui.scripts.variable_styles import _colormap_from_palette

        colors = gradient_colors(
            _colormap_from_palette(palette, getattr(var, "name", "") or "variable")
        )

    return variable_spec(
        colors=colors,
        vmin=(vis or {}).get("min"),
        vmax=(vis or {}).get("max"),
        render_kind=render_kind,
        var=var,
        title=title,
    )


def _with_hash(color: str) -> str:
    """GEE palettes omit the leading '#'; legend colours need it."""
    color = str(color)
    return color if color.startswith("#") else f"#{color}"


def _hex_at(cmap, position: float) -> str:
    """One hex colour sampled from a matplotlib Colormap."""
    from matplotlib.colors import to_hex

    return to_hex(cmap(position))
