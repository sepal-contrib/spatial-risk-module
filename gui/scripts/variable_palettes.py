"""Palette presets a user can pick for a custom variable, and their storage form.

A custom (non-catalogue) variable has no palette of its own, so the Variables
modal offers a choice and stores it on the variable as ``vis_params`` — the
GEE visualization shape (``{"palette": [hex-without-#, ...], "min", "max"}``)
that ``_styled_layer`` (GEE side) and ``resolve_variable_style`` (downloaded
raster side) both understand.

* Categorical: a class list, ``{"classes": [...]}`` (``variable_classes``),
  painted by hand or from one of the qualitative palettes below.
* Continuous: one of the named ramps below, optionally inverted, pinned to a
  ``min``/``max`` or, without one, stretched to the layer's own values.

Kept free of Solara so the modal's round-trips stay unit-testable.
"""

from typing import Dict, List, Optional, Tuple

# Low -> high ramps. "grayscale" is today's continuous fallback; blues, greens,
# cool_warm and terrain are ramps the catalogue already uses (precipitation,
# forest fraction, temperature, altitude).
RAMPS: Dict[str, List[str]] = {
    "grayscale": ["000000", "ffffff"],
    "blues": ["ffffff", "c6dbef", "4292c6", "08306b"],
    "greens": ["ffffff", "a1d99b", "31a354", "00441b"],
    "oranges": ["ffffff", "fdae6b", "e6550d", "7f2704"],
    "cool_warm": ["313695", "74add1", "fee090", "f46d43", "a50026"],
    "viridis": ["440154", "3b528b", "21918c", "5ec962", "fde725"],
    # The catalogue's altitude ramp, so a custom DEM matches the built-in one.
    "terrain": ["006633", "e5ffcc", "662a00", "d8d8d8", "f5f5f5"],
    # Yellow -> dark red: pressure / density / risk-like quantities.
    "heat": ["ffffb2", "fd8d3c", "e31a1c", "800026"],
    # Perceptually uniform, colour-blind safe, dark low end (unlike the whites).
    "magma": ["000004", "3b0f70", "8c2981", "de4968", "fe9f6d", "fcfdbf"],
}

# A new mask's presence colour: the catalogue's own forest green.
DEFAULT_PRESENCE = "2e7d32"
DEFAULT_RAMP = "grayscale"


def _bare(value: str) -> str:
    """Normalise to the GEE palette form: lowercase, no ``#``."""
    return str(value).strip().lstrip("#").lower()


def continuous_vis(ramp: str, invert: bool = False, span=None) -> dict:
    """``vis_params`` for a continuous layer: a named ramp, optionally reversed.

    ``span`` is ``(min, max)`` to pin the ramp to; None leaves the renderer to
    stretch it to the layer's own values.
    """
    palette = list(RAMPS[ramp])
    if invert:
        palette.reverse()
    vis = {"palette": palette}
    if span is not None:
        vis["min"], vis["max"] = span
    return vis


def ramp_choice(vis: Optional[dict]) -> Tuple[Optional[str], bool]:
    """``(ramp_key, inverted)`` for a continuous ``vis_params``, or ``(None, False)``.

    Inversion is stored as the reversed palette, so it is recovered by matching
    the palette against each ramp both ways.
    """
    palette = [_bare(c) for c in (vis or {}).get("palette") or []]
    if not palette:
        return None, False
    for key, ramp in RAMPS.items():
        if palette == ramp:
            return key, False
        if palette == list(reversed(ramp)):
            return key, True
    return None, False


def ramp_css(ramp: str, invert: bool = False) -> str:
    """A CSS ``linear-gradient`` previewing a ramp, for the modal's ramp buttons."""
    palette = continuous_vis(ramp, invert)["palette"]
    return "linear-gradient(to right, " + ", ".join(f"#{c}" for c in palette) + ")"


def ramp_menu_css(menu_class: str, invert: bool) -> str:
    """CSS drawing each ramp's bar before the n-th item of the ramp select's menu.

    A widget slot cannot read the row it is drawn for, so the bars are CSS.
    """
    return "\n".join(
        f".{menu_class} .v-list-item:nth-child({i}) .v-list-item__title::before {{"
        f"content:'';display:inline-block;width:72px;height:12px;border-radius:2px;"
        f"margin-right:10px;vertical-align:middle;background:{ramp_css(key, invert)};}}"
        for i, key in enumerate(RAMPS, start=1)
    )


# Qualitative palettes a categorical layer's classes can be painted from.
QUALITATIVE = ("set1", "set2", "tab10", "paired")
DEFAULT_QUALITATIVE = "set1"
_QUALITATIVE_MPL = {
    "set1": "Set1",
    "set2": "Set2",
    "tab10": "tab10",
    "paired": "Paired",
}


def qualitative_colors(name: str, n: int) -> List[str]:
    """``n`` bare hex colours of a qualitative palette, cycled when ``n`` exceeds it."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    base = [_bare(to_hex(c)) for c in colormaps[_QUALITATIVE_MPL[name]].colors]
    return [base[i % len(base)] for i in range(n)]


def qualitative_css(name: str) -> str:
    """A hard-stop CSS ``linear-gradient`` of a palette's first six colours."""
    colors = qualitative_colors(name, 6)
    step = 100 / len(colors)
    stops = []
    for i, c in enumerate(colors):
        stops += [f"#{c} {i * step:.2f}%", f"#{c} {(i + 1) * step:.2f}%"]
    return "linear-gradient(to right, " + ", ".join(stops) + ")"
