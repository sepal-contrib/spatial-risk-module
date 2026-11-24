"""A catalogue layer's own palette is offered, and recognised, as the "default" ramp."""

from gui.scripts import variable_palettes as vp

# A catalogue palette as written in the catalogue: uppercase, with ``#``.
DEFAULT = ["#D9D9D9", "E31A1C"]


def test_ramp_choice_recognises_the_default_palette():
    """The default palette reads as "default" both ways; a preset keeps its key."""
    assert vp.ramp_choice({"palette": ["d9d9d9", "e31a1c"]}, DEFAULT) == (
        "default",
        False,
    )
    assert vp.ramp_choice({"palette": ["#E31A1C", "d9d9d9"]}, DEFAULT) == (
        "default",
        True,
    )
    blues = {"palette": vp.RAMPS["blues"]}
    assert vp.ramp_choice(blues, vp.RAMPS["blues"]) == ("blues", False)
    assert vp.ramp_choice({"palette": ["d9d9d9", "e31a1c"]}) == (None, False)


def test_continuous_vis_default_ramp():
    """The "default" ramp stores the bare layer palette, reversed when inverted."""
    assert vp.continuous_vis("default", default_palette=DEFAULT) == {
        "palette": ["d9d9d9", "e31a1c"]
    }
    inverted = vp.continuous_vis(
        "default", invert=True, span=(0, 1), default_palette=DEFAULT
    )
    assert inverted == {"palette": ["e31a1c", "d9d9d9"], "min": 0, "max": 1}
    assert "#d9d9d9" in vp.ramp_css("default", default_palette=DEFAULT)
