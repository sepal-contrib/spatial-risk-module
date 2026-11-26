"""The default display style of a catalogue layer, and what a user changed.

A style is ``{"classes": [...]}`` (categorical) or ``{"palette": [...],
"min"?, "max"?}`` (continuous): the shape the display editor edits and a
variable's ``vis_params`` stores. Free of Solara and i18n: class labels carry
their i18n key in ``label_key``.
"""

from typing import Optional

from gui.scripts import variable_classes as classes_mod
from gui.scripts.predefined.catalogue import resolved_entry
from gui.scripts.variable_palettes import _bare

_DEFAULT_CLASS_KEYS = ("legend.class.absent", "legend.class.present")


def default_vis(catalogue_key: str, params: dict) -> Optional[dict]:
    """The resolved entry's style in editable shape, None when not editable."""
    entry = resolved_entry(catalogue_key, params)
    if entry.get("random_visualizer"):
        return None
    if entry.get("class_labels"):
        found = classes_mod.from_class_labels(entry["class_labels"])
        return {"classes": [{**c, "label_key": c["label"]} for c in found]}
    vis = entry.get("vis_params") or {}
    palette = vis.get("palette") or []
    if entry.get("raster_type") == "categorical":
        if len(palette) != 2 or (vis.get("min"), vis.get("max")) != (0, 1):
            return None
        keys = entry.get("legend_class_keys") or _DEFAULT_CLASS_KEYS
        return {
            "classes": [
                classes_mod.make(v, palette[v], keys[v], label_key=keys[v])
                for v in (0, 1)
            ]
        }
    if not palette:
        return None
    return {
        "palette": list(palette),
        **{k: vis[k] for k in ("min", "max") if k in vis},
    }


def is_editable(catalogue_key: str, params: dict) -> bool:
    """Whether the layer's style can be edited."""
    return default_vis(catalogue_key, params) is not None


def _normal(style: Optional[dict]):
    """A style with colours bare and labels dropped where the key gives them."""
    if style is None:
        return None
    out = dict(style)
    if "palette" in out:
        out["palette"] = [_bare(c) for c in out["palette"]]
    if "classes" in out:
        out["classes"] = [
            {
                **{
                    k: v
                    for k, v in c.items()
                    if not (k == "label" and "label_key" in c)
                },
                "color": _bare(c["color"]),
            }
            for c in out["classes"]
        ]
    return out


def vis_override(default: Optional[dict], current: Optional[dict]) -> Optional[dict]:
    """``current`` when it differs from ``default``, else None (nothing to store)."""
    return None if _normal(default) == _normal(current) else current
