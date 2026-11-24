"""Class lists for a categorical custom variable, and how they are drawn.

A class is a plain dict, ``{"value": int, "color": bare hex, "label": str,
"visible": bool}``, stored on the variable as ``vis_params["classes"]``. A
value that is not listed, or whose class is hidden, is drawn transparent.

Ported from the deforestation alerts app (gui/scripts/vis_classes.py), on
dicts instead of a model because ``vis_params`` is a free dict here. Kept free
of Solara and i18n; matplotlib and rasterio are imported where needed.
"""

import hashlib
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple
from xml.sax.saxutils import escape

from gui.scripts import variable_palettes as palettes
from gui.scripts.variable_palettes import _bare

MAX_CLASSES = 50

ERR_NO_CLASSES = "vars.modal.err_no_classes"
ERR_DUPLICATE = "vars.modal.err_duplicate_class"
ERR_LIMIT = "vars.modal.err_class_limit"
ERR_NO_VISIBLE = "vars.modal.err_no_visible_class"


def make(value, color, label: str = "", visible: bool = True) -> dict:
    """One class, in its stored form."""
    return {
        "value": int(value),
        "color": _bare(color)[:6],
        "label": str(label),
        "visible": bool(visible),
    }


def default_classes(absent: str = "0", present: str = "1") -> List[dict]:
    """A new layer's classes: the 0/1 mask most custom layers are.

    0 is white and 1 the default presence colour, the look a custom mask had
    before classes existed.
    """
    return [
        make(0, "ffffff", absent),
        make(1, palettes.DEFAULT_PRESENCE, present),
    ]


def from_vis(vis: Optional[dict], absent: str = "0", present: str = "1") -> List[dict]:
    """The classes a saved ``vis_params`` holds, or None when it holds none.

    A mask saved before classes existed (``["ffffff", colour]`` pinned to 0..1)
    reads as its two classes, so editing it keeps its look.
    """
    vis = vis or {}
    if vis.get("classes"):
        return [make(**c) for c in vis["classes"]]
    palette = vis.get("palette") or []
    if len(palette) == 2 and vis.get("min") == 0 and vis.get("max") == 1:
        return [make(0, palette[0], absent), make(1, palette[1], present)]
    return None


def validate(classes) -> Optional[str]:
    """The i18n key of what stops the classes being saved, or None."""
    values = [c["value"] for c in classes]
    if not values:
        return ERR_NO_CLASSES
    if len(set(values)) != len(values):
        return ERR_DUPLICATE
    if len(values) > MAX_CLASSES:
        return ERR_LIMIT
    if not any(c["visible"] for c in classes):
        return ERR_NO_VISIBLE
    return None


def is_zero_one(values) -> bool:
    """Whether the distinct values are exactly ``{0, 1}``."""
    return sorted({int(v) for v in values}) == [0, 1]


def classes_from_values(
    values, *, palette: str = palettes.DEFAULT_QUALITATIVE, known: Optional[dict] = None
) -> List[dict]:
    """One class per distinct value, sorted.

    ``known`` (an Earth Engine band's ``*_class_*`` properties) gives the
    colours and names it lists. A 0/1 find without ``known`` hides 0.
    """
    ordered = sorted({int(v) for v in values})
    colors = palettes.qualitative_colors(palette, len(ordered))
    meta = {}
    if known:
        for v, color, name in zip(
            known.get("values") or [],
            known.get("palette") or [],
            known.get("names") or [],
        ):
            meta[int(v)] = (_bare(color), str(name))
    zero_one = ordered == [0, 1] and not known
    out = []
    for i, v in enumerate(ordered):
        color, label = meta.get(v, (colors[i], str(v)))
        if zero_one and v == 1:
            color = colors[0]
        out.append(make(v, color, label, visible=not (zero_one and v == 0)))
    return out


def recolor(classes, palette: str) -> List[dict]:
    """The same classes painted from a palette, in list order."""
    colors = palettes.qualitative_colors(palette, len(classes))
    return [{**c, "color": colors[i]} for i, c in enumerate(classes)]


def merge_found(classes, found) -> List[dict]:
    """A new find merged into the listed classes, sorted by value.

    A listed class keeps its colour, label and visibility; a found value not
    listed comes from ``found``; a listed value not found (added by hand) stays.
    """
    listed = {c["value"]: c for c in classes}
    merged = {c["value"]: listed.get(c["value"], c) for c in found}
    merged.update({v: c for v, c in listed.items() if v not in merged})
    return [merged[v] for v in sorted(merged)]


def palette_of(classes) -> Optional[str]:
    """The qualitative palette the classes are painted from in list order, or None."""
    if not classes:
        return None
    colors = [_bare(c["color"]) for c in classes]
    return next(
        (
            key
            for key in palettes.QUALITATIVE
            if palettes.qualitative_colors(key, len(classes)) == colors
        ),
        None,
    )


def from_class_labels(class_labels) -> List[dict]:
    """A catalogue entry's ``class_labels`` as classes, sorted by value.

    Each label holds its i18n key (this module is i18n-free): the legend
    translates it.
    """
    return sorted(
        (make(c["value"], c["color"], c["label_key"]) for c in class_labels),
        key=lambda c: c["value"],
    )


def visible_classes(classes) -> List[dict]:
    """The classes that are drawn, sorted by value."""
    return sorted(
        (c for c in classes or [] if c.get("visible", True)), key=lambda c: c["value"]
    )


def outline_style(classes) -> Optional[dict]:
    """The map style of a vector layer's outline, or None for the default.

    A vector is one overlay, so it takes one colour: its first visible class
    that is a feature (0 is the background).
    """
    color = next(
        (c["color"] for c in visible_classes(classes) if c["value"] != 0), None
    )
    if color is None:
        return None
    return {"color": "#" + _bare(color), "weight": 2, "fillOpacity": 0.0}


def ee_remap(classes) -> Tuple[List[int], List[int], dict]:
    """``(from, to, vis)`` for ``Image.remap`` over the visible classes.

    ``remap`` masks every value it is not given, so the rest is transparent.
    """
    shown = visible_classes(classes)
    m = len(shown)
    vis = {
        "min": 0,
        "max": max(m - 1, 1),
        "palette": [_bare(c["color"]) for c in shown],
    }
    return [c["value"] for c in shown], list(range(m)), vis


def class_colormap(classes):
    """A ``ListedColormap`` of the visible classes, in value order."""
    from matplotlib.colors import ListedColormap

    colors = ["#" + _bare(c["color"]) for c in visible_classes(classes)]
    if len(colors) == 1:
        # localtileserver turns it into a LinearSegmentedColormap, which a
        # single colour cannot make.
        colors = colors * 2
    return ListedColormap(colors, name="classes")


def legend_colors_labels(classes) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``(colours with #, labels)`` of the visible classes, for the legend chips."""
    shown = visible_classes(classes)
    return (
        tuple("#" + _bare(c["color"]) for c in shown),
        tuple(c.get("label") or str(c["value"]) for c in shown),
    )


def _num(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def lut_text(classes) -> str:
    """A VRT ``LUT`` mapping the k-th visible class to k and the rest to 0.

    Each class is a narrow spike over its value, so the values between two
    classes, and beyond the first and last, map to 0.
    """
    parts = []
    for k, c in enumerate(visible_classes(classes), start=1):
        v = c["value"]
        parts += [f"{_num(v - 0.25)}:0", f"{_num(v)}:{k}", f"{_num(v + 0.25)}:0"]
    return ",".join(parts)


def categorical_vrt(path, classes) -> Path:
    """A one-band Byte VRT of ``path`` with the visible classes as 1..m.

    Every other value is 0, its nodata. localtileserver drops a colormap's
    alpha and stretches it linearly, so sparse class values (10, 20, 95) and
    transparent ones cannot be drawn from the file itself. Written to the
    app's scratch dir, named by the file, its mtime and the LUT, and always
    rewritten (a moved file would leave a stale SourceFilename). The file's
    earlier styles are removed.
    """
    import rasterio

    from spatialrisk import gdal_env

    path = Path(path).resolve()
    lut = lut_text(classes)
    # The style is in the name too: the tile URL changes with it, so the
    # browser never shows tiles cached for the last style.
    of_file = hashlib.sha1(str(path).encode()).hexdigest()[:12]
    stamp = f"{path.stat().st_mtime_ns}|{lut}"
    digest = hashlib.sha1(stamp.encode()).hexdigest()[:12]
    out = gdal_env.scratch_dir() / f"var_classes_{of_file}_{digest}.vrt"
    with rasterio.open(path) as src:
        w, h, crs, gt = src.width, src.height, src.crs, src.transform.to_gdal()
        # The file's nodata stays nodata even when a visible class has its value.
        nodata = "" if src.nodata is None else f"<NODATA>{src.nodata!r}</NODATA>"
    srs = f"<SRS>{escape(crs.to_wkt())}</SRS>" if crs is not None else ""
    xml = f"""<VRTDataset rasterXSize="{w}" rasterYSize="{h}">
  {srs}
  <GeoTransform>{", ".join(repr(float(c)) for c in gt)}</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <NoDataValue>0</NoDataValue>
    <ComplexSource>
      <SourceFilename relativeToVRT="0">{escape(str(path))}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" />
      <DstRect xOff="0" yOff="0" xSize="{w}" ySize="{h}" />
      {nodata}
      <LUT>{lut}</LUT>
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>
"""
    part = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.part")
    part.write_text(xml)
    os.replace(part, out)
    for old in out.parent.glob(f"var_classes_{of_file}_*.vrt"):
        if old != out:
            old.unlink(missing_ok=True)  # the last style's file
    return out
