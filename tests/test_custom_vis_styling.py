"""A user-chosen ``vis_params`` drives the layer's look on both sides.

The Variables modal offers presence swatches (categorical) and colour ramps
(continuous); what it stores must win over the catalogue and raster-type
fallbacks in the GEE styler, the local-raster styler and the legend, and an
asset-id GEE variable must be mappable before it is downloaded.
"""

from matplotlib.colors import Colormap

from gui.scripts import variable_palettes as vp
from gui.scripts.legend_data import (
    Label,
    variable_spec_from_style,
    variable_spec_from_vis,
)
from gui.scripts.map_helpers import is_mappable
from gui.scripts.variable_styles import resolve_variable_style


def _var(
    name="custom", raster_type="categorical", vis_params=None, cls="LocalRasterVar"
):
    """A stand-in variable with the fields the stylers read."""
    rt = type("RT", (), {"value": raster_type})()
    return type(cls, (), {"name": name, "raster_type": rt, "vis_params": vis_params})()


def _rgb(cmap, x):
    """A colormap sample as an 8-bit RGB tuple."""
    return tuple(round(c * 255) for c in cmap(x)[:3])


# ---- presets -----------------------------------------------------------------


def test_categorical_vis_round_trips_the_presence_colour():
    """Stored as white -> colour pinned to 0..1 (bare hex); editing reads it back."""
    vis = vp.categorical_vis("#2E7D32")
    assert vis == {"palette": ["ffffff", "2e7d32"], "min": 0, "max": 1}
    assert vp.presence_color(vis) == "#2e7d32"
    assert vp.presence_color(None) is None


def test_continuous_vis_round_trips_the_ramp_and_inversion():
    """Stored palette-only (auto-stretch), reversed when inverted.

    Editing reads both back; an unknown palette reads as no ramp.
    """
    assert vp.continuous_vis("blues") == {"palette": vp.RAMPS["blues"]}
    inverted = vp.continuous_vis("blues", invert=True)
    assert inverted == {"palette": list(reversed(vp.RAMPS["blues"]))}
    assert vp.ramp_choice(vp.continuous_vis("blues")) == ("blues", False)
    assert vp.ramp_choice(inverted) == ("blues", True)
    assert vp.ramp_choice({"palette": ["123456", "abcdef"]}) == (None, False)
    assert vp.ramp_choice(None) == (None, False)


def test_valid_hex_accepts_six_digit_forms_only():
    """Only six-digit hex (with or without #) counts as a colour."""
    assert vp.is_hex("#2e7d32") and vp.is_hex("2E7D32")
    assert not vp.is_hex("#2e7d3") and not vp.is_hex("green") and not vp.is_hex("")


# ---- local raster styler -------------------------------------------------------


def test_local_style_uses_the_variable_palette_before_the_fallback():
    """A downloaded custom mask renders the user's colour, not black/white."""
    vis = vp.categorical_vis("#7b1fa2")
    style = resolve_variable_style(_var(vis_params=vis))

    assert style["render_kind"] == "custom_palette"
    assert isinstance(style["colormap"], Colormap)
    assert (style["vmin"], style["vmax"]) == (0, 1)
    assert _rgb(style["colormap"], 0.0) == (255, 255, 255)
    assert _rgb(style["colormap"], 1.0) == (123, 31, 162)


def test_local_style_continuous_palette_auto_stretches():
    """A custom ramp is unpinned (stretched to the file) and honours inversion."""
    style = resolve_variable_style(
        _var(raster_type="continuous", vis_params=vp.continuous_vis("blues", True))
    )
    assert style["render_kind"] == "custom_palette"
    assert style["vmin"] is None and style["vmax"] is None
    assert _rgb(style["colormap"], 0.0) == (8, 48, 107)  # inverted: dark end first


def test_local_style_variable_palette_beats_the_catalogue():
    """A catalogue name with its own vis_params renders the user's choice."""
    style = resolve_variable_style(
        _var(
            name="slope",
            raster_type="continuous",
            vis_params=vp.continuous_vis("greens"),
        )
    )
    assert style["render_kind"] == "custom_palette"
    assert _rgb(style["colormap"], 1.0) == (0, 68, 27)


# ---- GEE styler ---------------------------------------------------------------


def test_gee_style_uses_the_variable_palette_without_touching_gee():
    """The GEE styler applies vis_params verbatim, no min/max round-trip."""
    from gui.tile.variables_tile import _styled_layer

    class _Iface:
        def get_info(self, *_):
            raise AssertionError("no stretch needed for a pinned palette")

    image = object()
    vis = vp.categorical_vis("#2196f3")
    out_image, out_vis, kind = _styled_layer(image, _var(vis_params=vis), _Iface())
    assert out_image is image
    assert out_vis == vis
    assert kind == "custom_palette"


# ---- legend -------------------------------------------------------------------


def test_legend_follows_the_raster_type_on_both_sides():
    """A custom mask is two chips on both sides; a ramp is a gradient.

    White absent, colour present, whether the legend is built from the GEE
    vis or the local style.
    """
    vis = vp.categorical_vis("#2196f3")
    mask = _var(vis_params=vis)
    from_vis = variable_spec_from_vis(vis, "custom_palette", mask, Label(literal="x"))
    from_style = variable_spec_from_style(
        resolve_variable_style(mask), mask, Label(literal="x")
    )
    assert (from_vis.kind, from_vis.colors) == ("chips", ("#ffffff", "#2196f3"))
    assert (from_style.kind, from_style.colors) == ("chips", ("#ffffff", "#2196f3"))

    ramp = vp.continuous_vis("blues")
    cont = _var(raster_type="continuous", vis_params=ramp)
    assert (
        variable_spec_from_vis(ramp, "custom_palette", cont, Label(literal="x")).kind
        == "gradient"
    )


# ---- mappability ----------------------------------------------------------------


def test_asset_id_gee_variable_is_mappable_before_download():
    """An asset-id GEEVar gets a map button before it is downloaded.

    One with neither image nor asset id does not.
    """
    assert is_mappable(
        type("GEEVar", (), {"gee_images": None, "path": "projects/p/assets/x"})()
    )
    assert not is_mappable(type("GEEVar", (), {"gee_images": None, "path": None})())
