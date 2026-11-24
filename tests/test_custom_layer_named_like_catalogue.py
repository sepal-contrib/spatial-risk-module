"""A user's own file or asset named like a catalogue layer stays the user's."""

from pathlib import Path

from gui.scripts.predefined import CUSTOM_TAG
from gui.scripts.variable_styles import resolve_variable_style
from gui.tile.variables_tile import _variable_to_entry, same_layer
from gui.widget.variable_details_dialog import harmonized_details
from spatialrisk.project import Project
from spatialrisk.variables.gee_var import GEEVar, download_path
from spatialrisk.variables.local_raster_var import LocalRasterVar
from spatialrisk.variables.models import DataType, RasterType

Project._ensure_model_schemas()


def _download(tmp_path, monkeypatch, name, **fields):
    monkeypatch.setenv("SPATIAL_RISK_DATA_DIR", str(tmp_path / "data"))
    p = Project(project_name="named-like")
    var = LocalRasterVar(
        name=name,
        path=download_path(p.folders.data_raw_folder, name, None),
        raster_type=RasterType.continuous,
        project=p,
        **fields,
    )
    p.raw_variables[name] = var
    return p, var


def test_a_downloaded_custom_asset_reopens_as_custom(tmp_path, monkeypatch):
    """Downloaded from the user's asset 'roads': editing must not swap in OSM."""
    p, var = _download(tmp_path, monkeypatch, "roads", tags=[CUSTOM_TAG])
    assert _variable_to_entry("roads", var, p)["source"] == "custom"


def test_a_download_saved_on_main_reopens_as_custom_like_on_main(tmp_path, monkeypatch):
    """No params, no tag: it could be the user's asset, so the safe side."""
    p, var = _download(tmp_path, monkeypatch, "roads")
    assert _variable_to_entry("roads", var, p)["source"] == "custom"


def test_a_downloaded_catalogue_layer_reopens_as_catalogue(tmp_path, monkeypatch):
    """A catalogue layer always carries params, {} when it has none."""
    p, var = _download(tmp_path, monkeypatch, "altitude", params={})
    entry = _variable_to_entry("altitude", var, p)
    assert (entry["source"], entry["predefined_key"]) == ("predefined", "altitude")


def test_a_colour_edit_of_a_custom_asset_is_the_same_layer():
    """The reopened asset_id/scale match the modal's path/default_scale."""
    var = GEEVar(
        name="my_asset",
        path="projects/me/assets/a",
        band="b1",
        default_scale=30.0,
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
    )
    old = _variable_to_entry("my_asset", var, None)
    new = {
        "source": "custom",
        "type": "GEEVar",
        "name": "my_asset",
        "year": None,
        "path": "projects/me/assets/a",
        "default_scale": 30.0,
        "band": "b1",
        "raster_type": RasterType.continuous,
        "data_type": DataType.raster,
        "vis_params": {"palette": ["000000", "ff0000"]},
    }
    assert same_layer(old, new)
    assert not same_layer(old, {**new, "path": "projects/me/assets/b"})
    assert not same_layer(old, {**new, "default_scale": 100.0})


def test_a_custom_file_named_like_a_catalogue_variant_draws_and_reads_as_custom(
    tmp_path, monkeypatch
):
    """slope_pct is a catalogue name; the user's file keeps its own look and details."""
    p, var = _download(tmp_path, monkeypatch, "slope_pct", tags=[CUSTOM_TAG])
    var.path = Path(tmp_path / "my_slope.tif")
    assert resolve_variable_style(var)["render_kind"] == "continuous_fallback"
    labels = [label for label, _value in harmonized_details(p, "slope_pct").rows]

    var.tags = []  # the same layer from the catalogue
    assert resolve_variable_style(var)["render_kind"] != "continuous_fallback"
    catalogue_labels = [label for label, _ in harmonized_details(p, "slope_pct").rows]
    assert len(catalogue_labels) > len(labels)
