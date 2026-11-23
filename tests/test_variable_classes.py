"""A categorical custom variable's class list: found, edited, drawn, explained.

Covers the pure helpers (``variable_classes``, ``variable_probes``) and the
three places a class list is drawn from: the GEE styler, the local raster
styler (through a VRT) and the legend.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import gui.scripts.variable_map as vmap
from gui.scripts import variable_classes as vc
from gui.scripts import variable_probes as probes
from gui.scripts.legend_data import (
    Label,
    variable_spec_from_style,
    variable_spec_from_vis,
)
from gui.scripts.variable_styles import resolve_variable_style

CLASSES = [
    vc.make(10, "#006400", "Trees"),
    vc.make(50, "fa0000", "Built-up"),
    vc.make(80, "0064c8", "Water", visible=False),
]


def _var(vis_params, raster_type="categorical"):
    """A stand-in variable with the fields the stylers read."""
    rt = type("RT", (), {"value": raster_type})()
    return type(
        "LocalRasterVar",
        (),
        {"name": "landcover", "raster_type": rt, "vis_params": vis_params},
    )()


def _tif(path, data, nodata=None):
    """Write ``data`` (2-D) as a single-band GeoTIFF."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, data.shape[0], 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return str(path)


# ---- the class list -------------------------------------------------------------


def test_a_mask_saved_before_classes_reads_as_its_two_classes():
    """The old presence colour is class 1; anything else holds no classes."""
    legacy = {"palette": ["ffffff", "7b1fa2"], "min": 0, "max": 1}
    assert vc.from_vis(legacy, "Absent", "Present") == [
        vc.make(0, "ffffff", "Absent"),
        vc.make(1, "7b1fa2", "Present"),
    ]
    assert vc.from_vis({"classes": CLASSES}) == CLASSES
    assert vc.from_vis({"palette": ["000000", "ffffff"]}) is None  # a ramp
    assert vc.from_vis(None) is None


def test_validate_names_what_stops_a_save():
    """Empty, duplicated, oversized and all-hidden lists are refused."""
    assert vc.validate(CLASSES) is None
    assert vc.validate([]) == vc.ERR_NO_CLASSES
    assert vc.validate([CLASSES[0], CLASSES[0]]) == vc.ERR_DUPLICATE
    assert vc.validate([CLASSES[2]]) == vc.ERR_NO_VISIBLE
    many = [vc.make(i, "000000") for i in range(vc.MAX_CLASSES + 1)]
    assert vc.validate(many) == vc.ERR_LIMIT


def test_found_values_take_the_palette_or_the_asset_own_classes():
    """Values are sorted and painted in order; an asset's class table wins."""
    found = vc.classes_from_values([30, 10, 10])
    assert [(c["value"], c["label"]) for c in found] == [(10, "10"), (30, "30")]
    assert vc.palette_of(found) == "set1"

    known = {
        "values": [10, 30],
        "palette": ["#006400", "ffbb22"],
        "names": ["Trees", "Shrub"],
    }
    named = vc.classes_from_values([10, 30], known=known)
    assert [(c["color"], c["label"]) for c in named] == [
        ("006400", "Trees"),
        ("ffbb22", "Shrub"),
    ]

    # Only 0 and 1: 0 starts hidden, so the mask is drawn over the basemap.
    mask = vc.classes_from_values([0, 1])
    assert [c["visible"] for c in mask] == [False, True]


def test_a_new_find_keeps_what_the_user_set():
    """Listed classes keep their colour, label and visibility; hand-added stay."""
    listed = [
        vc.make(10, "123456", "Mine", visible=False),
        vc.make(99, "abcdef", "By hand"),
    ]
    merged = vc.merge_found(listed, vc.classes_from_values([10, 20]))
    assert [c["value"] for c in merged] == [10, 20, 99]
    assert merged[0] == listed[0] and merged[2] == listed[1]


def test_recolor_repaints_every_class_and_keeps_the_rest():
    """A palette pick changes colours only."""
    painted = vc.recolor(CLASSES, "tab10")
    assert vc.palette_of(painted) == "tab10"
    assert [(c["value"], c["label"], c["visible"]) for c in painted] == [
        (c["value"], c["label"], c["visible"]) for c in CLASSES
    ]
    assert vc.palette_of(CLASSES) is None  # hand-picked colours follow none


# ---- drawn on the map ------------------------------------------------------------


def test_gee_style_remaps_the_visible_classes():
    """The GEE styler remaps visible values to 0..m-1; the rest is masked."""
    from gui.tile.variables_tile import _styled_layer

    calls = []

    class _Image:
        def remap(self, src, dst):
            calls.append((src, dst))
            return "REMAPPED"

    image, vis, kind = _styled_layer(_Image(), _var({"classes": CLASSES}), None)
    assert (image, kind) == ("REMAPPED", "custom_classes")
    assert calls == [([10, 50], [0, 1])]
    assert vis == {"min": 0, "max": 1, "palette": ["006400", "fa0000"]}


def test_the_files_nodata_is_never_drawn_as_a_class(tmp_path):
    """A mask tagged nodata=0 keeps its empty area clear with class 0 visible."""
    path = _tif(tmp_path / "mask.tif", np.array([[0, 1]], dtype="uint8"), nodata=0)
    vrt = vc.categorical_vrt(path, [vc.make(0, "ffffff"), vc.make(1, "2e7d32")])
    with rasterio.open(vrt) as src:
        assert src.read(1).tolist() == [[0, 2]]

    untagged = _tif(tmp_path / "plain.tif", np.array([[0, 1]], dtype="uint8"), None)
    vrt = vc.categorical_vrt(untagged, [vc.make(0, "ffffff"), vc.make(1, "2e7d32")])
    with rasterio.open(vrt) as src:
        assert src.read(1).tolist() == [[1, 2]]


def test_local_style_draws_the_classes_through_a_vrt(tmp_path, monkeypatch):
    """A local raster is drawn from a VRT: visible classes 1..m, the rest 0."""
    data = np.array([[10, 50, 80], [20, 255, 10]], dtype="uint8")
    path = _tif(tmp_path / "lc.tif", data, nodata=255)
    var = _var({"classes": CLASSES})

    style = resolve_variable_style(var)
    assert style["render_kind"] == "custom_classes"
    assert (style["vmin"], style["vmax"], style["nodata"]) == (1, 2, 0)
    assert style["colormap"].colors == ["#006400", "#fa0000"]

    vrt = vc.categorical_vrt(path, style["classes"])
    with rasterio.open(vrt) as src:
        assert src.nodata == 0
        # 80 is hidden, 20 is not listed, 255 is the file's nodata: all 0.
        assert src.read(1).tolist() == [[1, 2, 0], [0, 0, 1]]

    # And that VRT is what reaches the tile server.
    import localtileserver

    seen = {}

    class _Client:
        def __init__(self, source):
            seen["path"] = source

    monkeypatch.setattr(localtileserver, "TileClient", _Client, raising=False)
    monkeypatch.setattr(
        localtileserver,
        "get_leaflet_tile_layer",
        lambda client, **kw: seen.update(kw) or "LAYER",
        raising=False,
    )
    monkeypatch.setattr(
        "spatialrisk.overviews.ensure_overviews", lambda *a, **k: None, raising=False
    )

    class _Map:
        def remove_layer(self, *a, **k):
            pass

        def add_layer(self, *a, **k):
            pass

    vmap.add_raster_var_on_map(_Map(), path, var=var, layer_name="lc", key="k")
    assert seen["path"] == str(vrt)
    assert (seen["vmin"], seen["vmax"], seen["nodata"]) == (1, 2, 0)


def test_no_visible_class_falls_back_to_the_raster_type_look():
    """A list with nothing to draw is not a style (the modal refuses to save it)."""
    style = resolve_variable_style(_var({"classes": [CLASSES[2]]}))
    assert style["render_kind"] == "categorical_fallback"


def test_legend_is_one_chip_per_visible_class_on_both_sides():
    """The legend names the visible classes in the user's words."""
    var = _var({"classes": CLASSES})
    title = Label(literal="x")
    from_style = variable_spec_from_style(resolve_variable_style(var), var, title)
    from_vis = variable_spec_from_vis(
        vc.ee_remap(CLASSES)[2], "custom_classes", var, title
    )
    for spec in (from_style, from_vis):
        assert spec.kind == "chips"
        assert spec.colors == ("#006400", "#fa0000")
        assert [label.literal for label in spec.labels] == ["Trees", "Built-up"]


# ---- the probes -----------------------------------------------------------------


def test_file_probes_read_the_classes_and_the_stretch(tmp_path):
    """A local raster gives its values (nodata left out) and its 2-98 % span."""
    lc = _tif(tmp_path / "lc.tif", np.array([[1, 2], [2, 255]], dtype="uint8"), 255)
    assert probes.file_classes(lc) == {"values": [1, 2], "known": None}

    ramp = np.arange(10000, dtype="float32").reshape(100, 100)
    span = probes.file_stretch(_tif(tmp_path / "dem.tif", ramp))["span"]
    assert span == pytest.approx((199.98, 9799.02), rel=1e-3)

    flat = _tif(tmp_path / "flat.tif", np.ones((4, 4), dtype="float32"))
    assert probes.file_stretch(flat) == {"error": probes.ERR_NO_STRETCH}
    assert probes.file_classes(_tif(tmp_path / "f.tif", ramp / 7)) == {
        "error": probes.ERR_NOT_INTEGER
    }


def test_gee_class_reply_is_checked_and_reads_the_asset_class_table():
    """Too many values, or none, are refused; a class table comes back as known."""
    info = {
        "band": "Map",
        "n": 2,
        "hist": {"10": 5, "80": 1},
        "meta": {
            "Map_class_values": [10, 80],
            "Map_class_palette": ["006400", "0064c8"],
            "Map_class_names": ["Trees", "Water"],
        },
    }
    out = probes.classes_from_info(info)
    assert out["values"] == [10, 80]
    assert out["known"]["names"] == ["Trees", "Water"]

    assert probes.classes_from_info({"band": "b", "n": 0, "hist": {}}) == {
        "error": probes.ERR_NO_VALUES
    }
    # The server trims the histogram; its true size still refuses the band.
    big = {"band": "b", "n": 9000, "hist": {"1": 1, "2": 1}}
    assert probes.classes_from_info(big) == {"error": probes.ERR_TOO_MANY}


def test_min_and_max_are_both_empty_or_an_ordered_pair():
    """Empty means auto-stretch; a half-filled or reversed range is invalid."""
    assert probes.parse_span("", "") == (True, None)
    assert probes.parse_span("0", "2.5") == (True, (0.0, 2.5))
    assert probes.parse_span("3", "1") == (False, None)
    assert probes.parse_span("3", "") == (False, None)
    assert probes.parse_span("a", "1") == (False, None)


def test_a_harmonized_layer_keeps_the_style_picked_for_it(tmp_path):
    """Aligning a custom layer to the base grid carries its ``vis_params`` (#55)."""
    from types import SimpleNamespace

    from odc.geo.geobox import GeoBox

    from spatialrisk.variables.local_raster_var import LocalRasterVar
    from spatialrisk.variables.models import DataType, RasterType

    data = np.array([[10, 50], [80, 10]], dtype="uint8")
    project = SimpleNamespace(folders=SimpleNamespace(processed_data_folder=tmp_path))
    raw = LocalRasterVar.model_construct(
        name="landcover",
        raster_type=RasterType.categorical,
        path=tmp_path / "lc.tif",
        project=project,
        data_type=DataType.raster,
        vis_params={"classes": CLASSES},
    )
    _tif(raw.path, data, nodata=255)
    geobox = GeoBox.from_bbox((0, 0, 2, 2), crs="EPSG:4326", resolution=0.5)

    processed = raw.reproject_and_match(geobox)

    assert processed.vis_params == {"classes": CLASSES}
    assert resolve_variable_style(processed)["render_kind"] == "custom_classes"


# ---- vector layers -----------------------------------------------------------------


def _vector_file(path, names):
    """Write one square per name to a GeoJSON file."""
    import geopandas as gpd
    from shapely.geometry import box

    shapes = [box(i, 0, i + 1, 1) for i in range(len(names))]
    gpd.GeoDataFrame({"name": names}, geometry=shapes, crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )
    return str(path)


def test_a_vector_file_lists_one_named_class_per_feature(tmp_path):
    """Ids run from 1 in file order, as the "unique" rasterization burns them."""
    out = probes.vector_classes(
        _vector_file(tmp_path / "a.geojson", ["North", "South"])
    )
    found = vc.classes_from_values(out["values"], known=out["known"])
    assert [(c["value"], c["label"]) for c in found] == [(1, "North"), (2, "South")]

    # A feature with no name is labelled by its id.
    gaps = probes.vector_classes(_vector_file(tmp_path / "c.geojson", ["North", None]))
    assert gaps["known"]["names"] == ["North", "2"]

    many = _vector_file(tmp_path / "b.geojson", [str(i) for i in range(51)])
    assert probes.vector_classes(many) == {"error": probes.ERR_TOO_MANY_FEATURES}


def test_a_vector_outline_takes_its_first_feature_class_colour():
    """The background class (0) and hidden classes never colour the outline."""
    classes = [vc.make(0, "ffffff", "Absent"), vc.make(1, "#7B1FA2", "Present")]
    assert vc.outline_style(classes)["color"] == "#7b1fa2"
    assert vc.outline_style([classes[0]]) is None
    assert vc.outline_style(None) is None


def test_a_rasterized_vector_keeps_its_classes(tmp_path):
    """Rasterizing carries the classes; unstyled per-feature ids get one colour each."""
    from types import SimpleNamespace

    from spatialrisk.variables.local_raster_var import LocalRasterVar
    from spatialrisk.variables.local_vector_var import LocalVectorVar
    from spatialrisk.variables.models import DataType, RasterType

    project = SimpleNamespace(folders=SimpleNamespace(data_raw_folder=tmp_path))
    base = LocalRasterVar.model_construct(
        name="base", path=tmp_path / "base.tif", data_type=DataType.raster
    )
    _tif(base.path, np.ones((2, 2), dtype="uint8"))
    classes = [vc.make(1, "7b1fa2", "North"), vc.make(2, "0064c8", "South")]
    vector = LocalVectorVar.model_construct(
        name="zones",
        path=_vector_file(tmp_path / "z.geojson", ["North", "South"]),
        project=project,
        rasterization_method="unique",
        vis_params={"classes": classes},
        year=None,
        tags=[],
    )

    raster = vector.rasterize(base=base)
    assert raster.vis_params == {"classes": classes}
    assert resolve_variable_style(raster)["render_kind"] == "custom_classes"

    raster.vis_params = None
    style = resolve_variable_style(raster)
    assert (style["render_kind"], style["nodata"]) == ("random_visualizer", 0)
    # Pinned so that each id takes its own colour of 20, cycling.
    ids = [style["colormap"](i / 255) for i in (1, 2, 3, 21)]
    assert len(set(ids)) == 3 and ids[0] == ids[3]
    assert raster.raster_type == RasterType.categorical


def test_file_probes_read_the_picked_band(tmp_path):
    """A multi-band file's probes read the band picked in the modal."""
    path = tmp_path / "multi.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    ) as dst:
        dst.write(np.array([[1, 2], [1, 2]], dtype="uint8"), 1)
        dst.write(np.array([[5, 6], [7, 8]], dtype="uint8"), 2)

    assert probes.file_classes(str(path))["values"] == [1, 2]
    assert probes.file_classes(str(path), band=2)["values"] == [5, 6, 7, 8]
