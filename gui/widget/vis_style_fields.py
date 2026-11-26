"""The Variables modal's visualization fields for a custom raster layer.

A continuous layer's ``RampSelect``; a categorical layer's ``ClassList`` and
``PaletteMenu``; and the ``LegendPreview`` that ends either.

Ported from the deforestation alerts app (gui/widget/vis_style_fields.py),
without its band and RGB fields: a variable is one band.
"""

from typing import Optional

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.scripts import variable_classes as vc
from gui.scripts import variable_palettes as palettes
from gui.scripts.legend_data import LegendSpec, resolve_label
from gui.widget.enter_key import ENTER_OWN
from gui.widget.text_style import MUTED

_RAMP_ROW = "display:flex;align-items:center;gap:16px;"
RAMP_MENU = "sr-ramp-menu"  # the ramp select's menu, whose rows ramp_menu_css draws
_SELECTION_BAR = (
    "display:inline-block;width:72px;height:12px;border-radius:2px;margin-right:10px;"
    "vertical-align:middle;background:{gradient};"
)


@solara.component
def RampSelect(
    value, on_value, invert, on_invert, disabled=False, default_palette=None
):
    """The colour ramp select and, on its row, the Invert switch.

    The chosen ramp shows its bar and name; the menu rows' bars are CSS
    (``ramp_menu_css``), as a widget slot cannot read the row it is drawn for.
    A catalogue layer's ``default_palette`` is offered first, as "default".
    """
    items = [
        {"text": t(f"vars.modal.ramp_{key}"), "value": key}
        for key in palettes.ramp_keys(default_palette)
    ]
    slots = []
    if value is not None:
        gradient = palettes.ramp_css(value, invert, default_palette)
        chosen = rv.Html(
            tag="span",
            children=[
                rv.Html(tag="span", style_=_SELECTION_BAR.format(gradient=gradient)),
                t(f"vars.modal.ramp_{value}"),
            ],
        )
        slots = [{"name": "selection", "children": chosen}]
    with rv.Html(tag="div", style_=_RAMP_ROW):
        solara.Style(palettes.ramp_menu_css(RAMP_MENU, invert, default_palette))
        rv.Select(
            label=t("vars.modal.ramp_label"),
            items=items,
            v_model=value,
            on_v_model=on_value,
            dense=True,
            outlined=True,
            hide_details=True,
            disabled=disabled,
            menu_props={"contentClass": RAMP_MENU},
            v_slots=slots,
            class_="sr-ramp-select",
            style_="flex:1;min-width:0;",
        )
        rv.Switch(
            label=t("vars.modal.ramp_invert_label"),
            v_model=invert,
            on_v_model=on_invert,
            dense=True,
            hide_details=True,
            disabled=disabled,
            class_="mt-0 pt-0",
        )


_PREVIEW_BAR = (
    "height:12px;border-radius:3px;border:1px solid rgba(0,0,0,0.1);"
    "background:{gradient};"
)
_TICKS = "display:flex;justify-content:space-between;font-size:0.75rem;opacity:0.75;"
_CHIPS = "display:flex;flex-wrap:wrap;gap:4px 12px;font-size:0.8125rem;"
_CHIP = "display:inline-flex;align-items:center;gap:4px;"
_SWATCH = (
    "display:inline-block;width:12px;height:12px;border-radius:2px;"
    "border:1px solid rgba(0,0,0,0.2);background:{color};"
)


@solara.component
def LegendPreview(spec: LegendSpec, mid: Optional[str] = None):
    """The legend the layer will get.

    A gradient bar with its min, ``mid`` and max ticks, or one chip per class.
    """
    labels = [resolve_label(label, t) for label in spec.labels]
    with rv.Html(tag="div", class_="sr-legend-preview", style_="margin-top:8px;"):
        if spec.kind == "gradient":
            gradient = "linear-gradient(to right, " + ", ".join(spec.colors) + ")"
            rv.Html(tag="div", key="bar", style_=_PREVIEW_BAR.format(gradient=gradient))
            ticks = [labels[0], mid, labels[-1]] if mid is not None else labels
            with rv.Html(tag="div", key="ticks", style_=_TICKS):
                for i, text in enumerate(ticks):
                    rv.Html(tag="span", key=f"tick{i}", children=[text])
        else:
            with rv.Html(tag="div", key="chips", style_=_CHIPS):
                for i, (color, text) in enumerate(zip(spec.colors, labels)):
                    with rv.Html(tag="span", key=f"chip{i}", style_=_CHIP):
                        rv.Html(tag="span", style_=_SWATCH.format(color=color))
                        rv.Html(tag="span", children=[text])


# The class rows: a swatch (its colour picker in a menu), the value, the label
# and the hide toggle; a hidden class is dimmed, as it is drawn transparent.
_CLASS_ROW = "display:flex;align-items:center;gap:8px;"
_CLASS_SWATCH = (
    "min-width:22px;width:22px;height:22px;padding:0;border-radius:4px;"
    "border:1px solid rgba(0,0,0,0.25);background:{color} !important;"
)
_CLASS_VALUE = "min-width:3em;text-align:right;font-variant-numeric:tabular-nums;"
_CLASS_HEADER_SKIP = "min-width:calc(22px + 3em + 8px);"  # swatch, gap and value
_PALETTE_BAR = (
    "display:inline-block;width:72px;height:12px;border-radius:2px;margin-right:8px;"
    "background:{gradient};"
)


def _parse_value(text):
    """The integer typed in "Add a value", or None when it is not a whole number."""
    try:
        number = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


@solara.component
def ClassList(classes, on_classes, disabled=False):
    """One row per class (swatch, value, label, hide, remove), then "Add a value".

    A new class takes the next colour of the palette the classes are painted
    from (the default palette's when they follow none). Enter in the field
    adds the value, never presses the form's Add.
    """
    added, set_added = solara.use_state("")
    add_error, set_add_error = solara.use_state(None)
    palette = vc.palette_of(classes) or palettes.DEFAULT_QUALITATIVE
    full = len(classes) >= vc.MAX_CLASSES

    def _update(i, **changes):
        on_classes([{**c, **changes} if j == i else c for j, c in enumerate(classes)])

    def _relabel(i, text):
        # Typing replaces a catalogue label: it stops following the language.
        on_classes(
            [
                {k: v for k, v in c.items() if k != "label_key"} | {"label": text}
                if j == i
                else c
                for j, c in enumerate(classes)
            ]
        )

    def _add(text):
        value = _parse_value(text)
        if value is None or value in {c["value"] for c in classes}:
            set_add_error(t("vars.modal.err_class_value"))
            return
        color = palettes.qualitative_colors(palette, len(classes) + 1)[-1]
        on_classes([*classes, vc.make(value, color, str(value))])
        set_added("")
        set_add_error(None)

    # The rows sit in their own box: a row added before "Add a value" must not
    # take its field's widget (and with it the Enter handler, use_event below).
    with rv.Html(tag="div", class_="sr-class-list", style_="margin-top:8px;"):
        with rv.Html(tag="div", key="rows"):
            # One "Label" over the column: a floating label per field needs the
            # top padding the compact rows drop, and rises into the row above.
            if classes:
                with rv.Html(tag="div", key="header", style_=_CLASS_ROW + MUTED):
                    rv.Html(tag="span", style_=_CLASS_HEADER_SKIP)
                    rv.Html(
                        tag="span",
                        class_="text-caption",
                        children=[t("vars.modal.class_label")],
                    )
            for i, c in enumerate(classes):
                color = f"#{c['color']}"
                with rv.Html(
                    tag="div",
                    key=f"class{c['value']}",
                    class_="sr-class-row",
                    style_=_CLASS_ROW + ("" if c["visible"] else "opacity:0.45;"),
                ):
                    swatch = solara.Button(
                        children=[],
                        style=_CLASS_SWATCH.format(color=color),
                        disabled=disabled,
                        classes=["sr-class-swatch"],
                    )
                    with solara.lab.Menu(
                        activator=swatch,
                        close_on_content_click=False,
                        use_activator_width=False,
                    ):
                        rv.ColorPicker(
                            v_model=color,
                            on_v_model=lambda value, i=i: _update(
                                i, color=palettes._bare(value)[:6]
                            ),
                            hide_inputs=True,
                            hide_mode_switch=True,
                            mode="hexa",
                        )
                    rv.Html(tag="span", style_=_CLASS_VALUE, children=[str(c["value"])])
                    rv.TextField(
                        placeholder=t("vars.modal.class_label"),
                        v_model=t(c["label_key"]) if c.get("label_key") else c["label"],
                        on_v_model=lambda value, i=i: _relabel(i, value or ""),
                        dense=True,
                        hide_details=True,
                        disabled=disabled,
                        class_="sr-class-label mt-0 pt-0",
                    )
                    solara.Button(
                        icon_name="mdi-eye" if c["visible"] else "mdi-eye-off",
                        icon=True,
                        small=True,
                        on_click=lambda *_, i=i, shown=c["visible"]: _update(
                            i, visible=not shown
                        ),
                        disabled=disabled,
                        classes=["sr-class-hide"],
                    )
                    solara.Button(
                        icon_name="mdi-close",
                        icon=True,
                        small=True,
                        on_click=lambda *_, i=i: on_classes(
                            classes[:i] + classes[i + 1 :]
                        ),
                        disabled=disabled,
                        classes=["sr-class-remove"],
                    )
        with rv.Html(tag="div", key="add", style_=_CLASS_ROW):
            field = rv.TextField(
                label=t("vars.modal.class_add"),
                v_model=added,
                on_v_model=set_added,
                type="number",
                dense=True,
                class_=ENTER_OWN,
                disabled=disabled or full,
                error_messages=[add_error] if add_error else [],
                style_="max-width:12em;",
            )
            # The widget's own value: Enter may be handled before the typed
            # text re-renders.
            rv.use_event(
                field, "keydown.enter", lambda widget, *_: _add(widget.v_model)
            )
            solara.Button(
                icon_name="mdi-plus",
                icon=True,
                small=True,
                on_click=lambda *_: _add(added),
                disabled=disabled or full,
                classes=["sr-class-add"],
            )


@solara.component
def PaletteMenu(value, on_value, disabled=False):
    """The qualitative palettes, each as its colour bar and name.

    A pick recolours every class. ``value`` is None when the classes follow no
    palette (hand-picked colours): then none is selected.
    """
    bar = (
        _PALETTE_BAR.format(gradient=palettes.qualitative_css(value))
        if value is not None
        else "display:none;"
    )
    text = t("vars.modal.palette_menu")
    if value is not None:
        text = f"{text}: {t(f'vars.modal.palette_{value}')}"
    activator = solara.Button(
        children=[rv.Html(tag="span", style_=bar), text],
        icon_name="mdi-palette",
        outlined=True,
        color="primary",
        small=True,
        disabled=disabled,
        classes=["sr-palette-menu"],
    )
    with solara.lab.Menu(activator=activator, use_activator_width=False):
        with rv.List(dense=True):
            for key in palettes.QUALITATIVE:
                solara.Button(
                    children=[
                        rv.Html(
                            tag="span",
                            style_=_PALETTE_BAR.format(
                                gradient=palettes.qualitative_css(key)
                            ),
                        ),
                        t(f"vars.modal.palette_{key}"),
                    ],
                    text=True,
                    on_click=lambda *_, k=key: on_value(k),
                    style="display:flex;justify-content:flex-start;width:100%;"
                    "text-transform:none;",
                    classes=["sr-palette"]
                    + (["sr-palette--selected"] if key == value else []),
                )
