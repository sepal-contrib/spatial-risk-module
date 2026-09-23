"""E2-1: legacy ADMIN/ASSET manifests are healed with aoi_spec on load.

``do_load`` (``gui/solara_app.py``) restores the AOI inside
``app_state.restoring_project()``, so the attach-on-select effect is refused
during that window, by design. For ADMIN/ASSET methods the autoselect re-run
publishes an ``AoiResult`` that compares EQUAL to the one ``do_load`` just
restored (frozen-dataclass equality), so solara's ``equals_extra`` skip means
``aoi_result`` never changes and the effect never re-fires on its own — the
legacy manifest keeps ``aoi_spec`` null forever, until some later manual Save.
DRAW heals only because its fresh result differs (a normalized name, a fresh
GeoDataFrame).

``ProjectPanel`` is directly coupled to the process-global ``app_state``
singleton (unlike a tile, which receives it as an argument) and needs a real
``DATA_DIR`` to scan, so it cannot be rendered standalone without one; this is
also why ``test_delete_confirm.py`` and ``test_project_panel_notifications.py``
assert its wiring at the source level instead. This module renders it for
real, driven through the actual Load button, because a source-string
assertion cannot tell whether the manifest ends up healed on disk.
"""

import json

import ipyvuetify as vw
import pytest

from gui.i18n import t

t("common.load")  # warm the translator before the first render

from _notification_host import render_under_notifications  # noqa: E402

import gui.solara_app as app  # noqa: E402
import spatialrisk.project as proj  # noqa: E402
from gui.store.state_manager import app_state  # noqa: E402

proj.Project._ensure_model_schemas()


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point both the app's and spatialrisk's project roots at ``tmp_path``."""
    monkeypatch.setattr(proj, "downloads_folder", tmp_path)
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _restore_app_state():
    """The Load button drives the real process-global ``app_state``.

    ``ProjectPanel`` imports it directly (it is not passed in as an argument,
    unlike a tile), so there is no way to inject a fresh instance for this
    render. Snapshot every reactive it touches and restore them after, so
    this test cannot leak a loaded project into whichever test runs next.
    """
    snapshot = {
        "project": app_state.project.value,
        "project_dirty": app_state.project_dirty.value,
        "last_saved": app_state.last_saved.value,
        "aoi_result": app_state.aoi_result.value,
        "aoi_spec": app_state.aoi_spec.value,
        "project_loaded_signal": app_state.project_loaded_signal.value,
    }
    try:
        yield
    finally:
        app_state.project.value = snapshot["project"]
        app_state.project_dirty.value = snapshot["project_dirty"]
        app_state.last_saved.value = snapshot["last_saved"]
        app_state.aoi_result.value = snapshot["aoi_result"]
        app_state.aoi_spec.value = snapshot["aoi_spec"]
        app_state.project_loaded_signal.value = snapshot["project_loaded_signal"]


def _write_legacy_admin1_manifest(data_dir, name, admin="21758"):
    """A pre-AoiSpec ADMIN1 manifest: no ``aoi_spec``, only the legacy keys."""
    p = proj.Project(project_name=name)
    p.aoi = {"method": "ADMIN1", "name": "COL_x", "gee": True, "admin": admin}
    p.save()
    return p


def _manifest_path(data_dir, name):
    return data_dir / name / f"{name}_project.json"


def _select_and_load(box):
    """Drive the real Load flow: pick the (only) row, then click Load."""
    group = _find(box, vw.ListItemGroup)
    assert len(group) == 1, "expected exactly one project list"
    rows = _find(group[0], vw.ListItem)
    assert len(rows) == 1, "expected exactly one project row"
    group[0].v_model = rows[0].value  # fires on_select, like a real row click

    load_btn = [b for b in _find(box, vw.Btn) if b.children == [t("common.load")]]
    assert len(load_btn) == 1, "expected exactly one Load button"
    assert load_btn[0].disabled is False, "Load must be enabled once selected"
    load_btn[0].fire_event("click", {})


def test_load_heals_a_legacy_admin1_manifest(data_dir, monkeypatch):
    """Loading a legacy ADMIN1 project writes aoi_spec and keeps legacy keys."""
    import pygaul

    monkeypatch.setattr(pygaul, "Items", lambda admin: object())

    def fake_names(admin="", complete=False, **_):
        import pandas as pd

        assert complete
        return pd.DataFrame([{"gaul0_code": "col0", "gaul1_code": admin}])

    monkeypatch.setattr(pygaul, "Names", fake_names)

    _write_legacy_admin1_manifest(data_dir, "legacy_admin1")

    box, rc = render_under_notifications(lambda: app.ProjectPanel(), handle_error=False)
    try:
        _select_and_load(box)
    finally:
        rc.close()

    manifest = json.loads(_manifest_path(data_dir, "legacy_admin1").read_text())
    assert manifest["aoi"]["aoi_spec"]["method"] == "ADMIN1"
    # The legacy fields the pre-v4 app relied on must still be there.
    assert manifest["aoi"]["method"] == "ADMIN1"
    assert manifest["aoi"]["admin"] == "21758"
    assert manifest["aoi"]["gee"] is True


def test_load_heals_a_legacy_asset_manifest(data_dir, monkeypatch):
    """Loading a legacy ASSET project writes aoi_spec and keeps legacy keys."""
    import ee

    sentinel_fc = ee.FeatureCollection.__new__(ee.FeatureCollection)
    monkeypatch.setattr(
        "gui.scripts.aoi_io.build_asset_feature_collection", lambda asset: sentinel_fc
    )

    name = "legacy_asset"
    p = proj.Project(project_name=name)
    p.aoi = {
        "method": "ASSET",
        "name": "aoi",
        "gee": True,
        "asset": {
            "asset_id": "users/me/aoi",
            "type": "TABLE",
            "column": "ALL",
            "value": None,
        },
    }
    p.save()

    box, rc = render_under_notifications(lambda: app.ProjectPanel(), handle_error=False)
    try:
        _select_and_load(box)
    finally:
        rc.close()

    manifest = json.loads(_manifest_path(data_dir, name).read_text())
    assert manifest["aoi"]["aoi_spec"]["method"] == "ASSET"
    assert manifest["aoi"]["asset"]["asset_id"] == "users/me/aoi"
    assert manifest["aoi"]["method"] == "ASSET"


def test_reloading_an_up_to_date_manifest_does_not_bump_its_mtime(
    data_dir, monkeypatch
):
    """A manifest that already carries a matching aoi_spec is left untouched.

    The heal call (``attach_current_aoi``) is idempotent, so a normal load of
    a project that was already healed (or saved under v4 from the start) must
    not rewrite the manifest — otherwise every load would bump its mtime.
    """
    import pygaul

    monkeypatch.setattr(pygaul, "Items", lambda admin: object())

    def fake_names(admin="", complete=False, **_):
        import pandas as pd

        assert complete
        return pd.DataFrame([{"gaul0_code": "col0", "gaul1_code": admin}])

    monkeypatch.setattr(pygaul, "Names", fake_names)

    name = "already_healed_admin1"
    _write_legacy_admin1_manifest(data_dir, name)
    manifest = _manifest_path(data_dir, name)

    # First load heals it.
    box, rc = render_under_notifications(lambda: app.ProjectPanel(), handle_error=False)
    try:
        _select_and_load(box)
    finally:
        rc.close()
    assert json.loads(manifest.read_text())["aoi"]["aoi_spec"] is not None
    healed_mtime = manifest.stat().st_mtime_ns

    # Second load of the now-healed manifest must be a no-op on disk.
    box, rc = render_under_notifications(lambda: app.ProjectPanel(), handle_error=False)
    try:
        _select_and_load(box)
    finally:
        rc.close()

    assert manifest.stat().st_mtime_ns == healed_mtime


def test_a_failed_heal_does_not_turn_a_successful_load_into_a_load_error(
    data_dir, monkeypatch
):
    """project.save() raising during the heal must not surface as a load error.

    The load itself already succeeded (the project and AOI are installed); a
    heal failure is logged and swallowed, not raised through do_load's own
    except-and-set-load-error handler.
    """
    import pygaul

    monkeypatch.setattr(pygaul, "Items", lambda admin: object())

    def fake_names(admin="", complete=False, **_):
        import pandas as pd

        assert complete
        return pd.DataFrame([{"gaul0_code": "col0", "gaul1_code": admin}])

    monkeypatch.setattr(pygaul, "Names", fake_names)

    name = "heal_fails"
    _write_legacy_admin1_manifest(data_dir, name)

    def boom_save(self, *a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(proj.Project, "save", boom_save)

    box, rc = render_under_notifications(lambda: app.ProjectPanel(), handle_error=False)
    try:
        _select_and_load(box)
        # The project still loaded: the Manage dialog closed, no load error banner.
        assert app_state.project.value is not None
        assert app_state.project.value.project_name == name
        assert not _find(box, vw.Alert)
    finally:
        rc.close()
