"""ProjectPanel: the Continue card, and a rename driven from the Manage list."""

import functools
import json

import ipyvuetify as vw

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from _notification_host import render_under_notifications  # noqa: E402

import gui.solara_app as app  # noqa: E402
from gui.scripts.project_io import list_projects, rename_project  # noqa: E402
from gui.store.state_manager import app_state  # noqa: E402


def _button(rc, label):
    return next(b for b in rc.find(vw.Btn).widgets if label in b.children)


def _icon_button(rc, icon):
    return next(
        b
        for b in rc.find(vw.Btn).widgets
        if any(isinstance(c, vw.Icon) and c.children == [icon] for c in b.children)
    )


def _panel(tmp_path, monkeypatch):
    for name in ("alpha", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}_project.json").write_text(
            json.dumps({"project_name": name}), encoding="utf-8"
        )
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        app, "rename_project", functools.partial(rename_project, data_dir=tmp_path)
    )
    if app_state.project.value is not None:
        app_state.close_project_state()
    return render_under_notifications(lambda: app.ProjectPanel(), handle_error=False)


def test_continue_card_replaces_the_empty_state(tmp_path, monkeypatch):
    """With saved projects the start window offers Continue, not the empty state."""
    box, rc = _panel(tmp_path, monkeypatch)
    try:
        assert _button(rc, t("project.button_continue")) is not None
        assert not [
            w
            for w in rc.find(vw.Html).widgets
            if w.children == [t("project.empty_heading")]
        ]
    finally:
        rc.close()


def test_rename_from_the_manage_list(tmp_path, monkeypatch):
    """The row pencil opens a prefilled form; Rename moves the folder and the list."""
    box, rc = _panel(tmp_path, monkeypatch)
    try:
        _button(rc, t("project.manage_count", count=2)).fire_event("click", None)
        _icon_button(rc, "mdi-pencil-outline").fire_event("click", None)  # row "alpha"
        field = next(
            f
            for f in rc.find(vw.TextField).widgets
            if f.label == t("project.dialog_new_name_label")
        )
        assert field.v_model == "alpha"  # prefilled with the current name
        rename = _button(rc, t("project.button_rename"))
        assert rename.disabled  # unchanged name

        field.v_model = "other"
        assert _button(rc, t("project.button_rename")).disabled  # taken

        field.v_model = "beta"
        _button(rc, t("project.button_rename")).fire_event("click", None)
        assert list_projects(tmp_path) == ["beta", "other"]
        titles = [w.children[0] for w in rc.find(vw.ListItemTitle).widgets]
        assert titles == ["beta", "other"]  # the open Manage list followed
    finally:
        rc.close()


def test_new_refuses_a_saved_projects_name(tmp_path, monkeypatch):
    """A taken name cannot be created; the form offers to load that project."""
    box, rc = _panel(tmp_path, monkeypatch)
    try:
        _button(rc, t("project.button_new_project")).fire_event("click", None)
        field = next(
            f
            for f in rc.find(vw.TextField).widgets
            if f.label == t("project.dialog_new_name_label")
        )
        field.v_model = "alpha"
        assert _button(rc, t("common.create")).disabled
        assert _button(rc, t("project.dialog_new_load_instead")) is not None
        field.v_model = "gamma"
        assert not _button(rc, t("common.create")).disabled
    finally:
        rc.close()
