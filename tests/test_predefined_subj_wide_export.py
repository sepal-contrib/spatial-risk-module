"""subj above 254 units: widened to uint16 on the download worker, user warned.

Unit values run 1..n and the categorical export uses 255 as nodata, while
toByte() saturates (ee.Image(300).toByte() == 255, verified 2026-10-09): every
unit from 255 on would silently become nodata. An Amazon-size AOI has ~500
GAUL districts.
"""

import inspect
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from _ee_expr import fake_aoi, patch_ee

import gui.scripts.predefined.admin as admin
import gui.scripts.predefined.download_hooks as download_hooks
import spatialrisk.gee.ee_fao_gaul as ee_fao_gaul
import spatialrisk.gee.ee_rasterize_unique_values as ee_rasterize
from gui.scripts import process_actions
from gui.scripts.predefined.download_hooks import prepare_predefined_download
from gui.tile.variables_tile import _run_download
from spatialrisk.project import Project  # noqa: F401 — GEEVar forward reference
from spatialrisk.variables.gee_var import GEEVar
from spatialrisk.variables.models import DataType, RasterType

GEEVar.model_rebuild()


@pytest.fixture
def fake_ee(monkeypatch):
    """One fake shared by the builder and the two spatialrisk helpers."""
    return patch_ee(monkeypatch, admin, ee_fao_gaul, ee_rasterize)


def _subj_var(fake_ee, tmp_path, name="subj"):
    return SimpleNamespace(
        name=name,
        aoi=fake_aoi(fake_ee),
        gee_images=["byte-image"],
        export_nodata=None,
        expected_local_path=tmp_path / f"{name}.tif",
    )


def _units(monkeypatch, n_units):
    calls = []

    def count(units, attribute):
        calls.append(attribute)
        return n_units

    monkeypatch.setattr(admin, "_count_units", count)
    return calls


@pytest.mark.parametrize("n_units", [1, 254])
def test_up_to_254_units_keep_the_byte_export(monkeypatch, fake_ee, tmp_path, n_units):
    """Review focus 1: at or below 254 nothing changes (identical to today)."""
    _units(monkeypatch, n_units)
    var = _subj_var(fake_ee, tmp_path)
    assert prepare_predefined_download(var) is None
    assert var.gee_images == ["byte-image"]
    assert var.export_nodata is None


@pytest.mark.parametrize(
    "n_units,key",
    [
        (255, "vars.subj_notice.wide"),
        (999, "vars.subj_notice.wide"),
        (1000, "vars.subj_notice.wide_jnr"),
    ],
)
def test_255_units_and_more_switch_to_uint16(
    monkeypatch, fake_ee, tmp_path, n_units, key
):
    """Review focus 1: uint16 + nodata 65535; JNR's 999-unit cap is flagged."""
    _units(monkeypatch, n_units)
    var = _subj_var(fake_ee, tmp_path)
    assert prepare_predefined_download(var) == (key, {"name": "subj", "n": n_units})
    assert var.export_nodata == 65535
    (image,) = var.gee_images
    assert repr(image).endswith(".clip(ee.FeatureCollection('AOI')).toUint16()")
    assert "GAUL_2024_L2" in repr(image)


def test_province_level_counts_level_one_units(monkeypatch, fake_ee, tmp_path):
    """The hook reads the level from the name (subj_province)."""
    calls = _units(monkeypatch, 300)
    var = _subj_var(fake_ee, tmp_path, name="subj_province")
    key, kwargs = prepare_predefined_download(var)
    assert calls == ["gaul1_name"]
    assert "GAUL_2024_L1" in repr(var.gee_images[0])
    assert kwargs == {"name": "subj_province", "n": 300}


def test_existing_file_skips_the_count_unless_overwriting(
    monkeypatch, fake_ee, tmp_path
):
    """A reused download costs no getInfo; a re-download re-checks."""
    calls = _units(monkeypatch, 300)
    var = _subj_var(fake_ee, tmp_path)
    var.expected_local_path.write_bytes(b"tif")
    assert prepare_predefined_download(var) is None
    assert calls == []
    assert prepare_predefined_download(var, overwrite=True)[0] == (
        "vars.subj_notice.wide"
    )


@pytest.mark.parametrize(
    "name", ["altitude", "my_custom_layer", "loss_forest_gfc_tc30_2015_2020"]
)
def test_layers_without_a_hook_are_untouched(name):
    """No hook -> None before anything touches the project or the path."""
    var = SimpleNamespace(name=name, gee_images=["img"])  # no expected_local_path
    assert prepare_predefined_download(var) is None


def test_asset_id_geevar_is_skipped():
    """A GEEVar with no built image has nothing to widen."""
    var = SimpleNamespace(name="subj", gee_images=None)
    assert prepare_predefined_download(var) is None


def test_a_custom_asset_named_subj_keeps_its_own_image(monkeypatch, fake_ee, tmp_path):
    """Shown on the map, an asset GEEVar holds an image; the hook leaves it alone."""
    calls = _units(monkeypatch, 999)
    var = _subj_var(fake_ee, tmp_path)
    var.path = "projects/me/assets/subj"
    assert prepare_predefined_download(var) is None
    assert var.gee_images == ["byte-image"] and calls == []


class _Proj:
    def __init__(self):
        self.raw_variables = {}


def _project_with_subj(monkeypatch):
    project = _Proj()
    project.raw_variables["subj"] = GEEVar(
        name="subj",
        data_type=DataType.raster,
        raster_type=RasterType.categorical,
        gee_images=["img"],
    )
    local = MagicMock(name="LocalRasterVar")
    monkeypatch.setattr(GEEVar, "to_local_raster", lambda *a, **k: local)
    return project


def test_materialize_forwards_the_notice(monkeypatch):
    """The hook runs before the export and its notice reaches on_notice."""
    seen = []
    monkeypatch.setattr(
        download_hooks,
        "prepare_predefined_download",
        lambda var, overwrite=False: seen.append((var.name, overwrite))
        or ("vars.subj_notice.wide", {"name": "subj", "n": 300}),
    )
    project = _project_with_subj(monkeypatch)
    notices = []
    process_actions.materialize_raw_layers(
        project, overwrite=True, on_notice=lambda key, kw: notices.append((key, kw))
    )
    assert seen == [("subj", True)]
    assert notices == [("vars.subj_notice.wide", {"name": "subj", "n": 300})]


def test_materialize_logs_the_notice_without_a_callback(monkeypatch, caplog):
    """The harmonize path has no toast channel: the task log shows it."""
    monkeypatch.setattr(
        download_hooks,
        "prepare_predefined_download",
        lambda var, overwrite=False: (
            "vars.subj_notice.wide",
            {"name": "subj", "n": 300},
        ),
    )
    project = _project_with_subj(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="spatial_risk"):
        process_actions.materialize_raw_layers(project)
    assert "300" in caplog.text
    assert "subj" in caplog.text


def test_run_download_turns_notices_into_warning_toasts():
    """The Variables tile wires on_notice to notifier.warning (source-level)."""
    src = inspect.getsource(_run_download)
    assert "on_notice=" in src
    assert "notifier.warning(" in src


def test_notice_keys_exist_with_their_placeholders():
    """Both notices resolve in every locale with {name} and {n}."""
    from _catalogue_checks import LOCALES, locale_messages

    for locale in LOCALES:
        messages = locale_messages(locale)
        for key in ("vars.subj_notice.wide", "vars.subj_notice.wide_jnr"):
            assert "{name}" in messages[key] and "{n}" in messages[key], (locale, key)
