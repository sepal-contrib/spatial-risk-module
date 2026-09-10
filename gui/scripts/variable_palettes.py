"""Palette presets a user can pick for a custom variable, and their storage form.

A custom (non-catalogue) variable has no palette of its own, so the Variables
modal offers a choice and stores it on the variable as ``vis_params`` — the
GEE visualization shape (``{"palette": [hex-without-#, ...], "min", "max"}``)
that ``_styled_layer`` (GEE side) and ``resolve_variable_style`` (downloaded
raster side) both understand.

* Categorical (0/1 mask): one *presence* colour; absence stays white, matching
  every catalogue mask (``["ffffff", colour]`` pinned to 0..1).
* Continuous: one of the named ramps below, optionally inverted, stretched to
  the layer's own min/max (palette only, no pin) — as the grayscale fallback is.

Kept free of Solara so the modal's round-trips stay unit-testable.
"""

import re
from typing import Dict, List, Optional, Tuple

# Presence swatches, in dialog order. Green and blue are the catalogue's own
# forest / water values so a custom mask sits naturally next to those layers.
PRESENCE_COLORS: List[str] = [
    "2e7d32",  # green (forest)
    "2196f3",  # blue (water)
    "00897b",  # teal
    "c9a400",  # yellow
    "ff9800",  # orange
    "d73027",  # red
    "e91e63",  # pink
    "7b1fa2",  # purple
    "795548",  # brown
    "616161",  # grey
]

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

DEFAULT_PRESENCE = PRESENCE_COLORS[0]
DEFAULT_RAMP = "grayscale"

_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")


def is_hex(value: Optional[str]) -> bool:
    """True for a six-digit hex colour, with or without the leading ``#``."""
    return bool(value) and bool(_HEX.match(str(value).strip()))


def _bare(value: str) -> str:
    """Normalise to the GEE palette form: lowercase, no ``#``."""
    return str(value).strip().lstrip("#").lower()


def categorical_vis(presence_hex: str) -> dict:
    """``vis_params`` for a 0/1 mask: white -> presence colour, pinned to 0..1."""
    return {"palette": ["ffffff", _bare(presence_hex)], "min": 0, "max": 1}


def continuous_vis(ramp: str, invert: bool = False) -> dict:
    """``vis_params`` for a continuous layer: a named ramp, optionally reversed."""
    palette = list(RAMPS[ramp])
    if invert:
        palette.reverse()
    return {"palette": palette}


def presence_color(vis: Optional[dict]) -> Optional[str]:
    """The ``#hex`` presence colour a categorical ``vis_params`` encodes, or None."""
    palette = (vis or {}).get("palette") or []
    if len(palette) != 2:
        return None
    return f"#{_bare(palette[1])}"


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
