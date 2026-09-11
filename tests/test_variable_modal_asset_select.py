"""A custom GEE layer picks its asset through pysepal's AssetSelectComponent.

The modal used to take the asset id in a bare text field, so a typo only
surfaced at download time. The selector lists the user's own assets, validates
whatever is typed against Earth Engine, and (for the modal) is restricted to
IMAGE assets — a TABLE is not a raster layer. The modal keeps storing a plain
asset-id string: the selector's ``{asset_id, type, column, value}`` dict is
unpacked at the boundary.
"""

import inspect

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

import gui.widget.variable_modal as mod  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402


def _find(widget, cls, out=None):
    """Collect widgets of ``cls`` in the rendered tree."""
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _custom_gee(**extra):
    return {
        "source": "custom",
        "type": "GEEVar",
        "name": "x",
        "asset_id": "projects/p/assets/x",
        **extra,
    }


def _stub_selector(monkeypatch):
    """Stand in for the real selector, which drives async GEE calls.

    The signature assertion guards the stub: the test would otherwise keep
    passing if pysepal dropped ``initial`` or ``types``.
    """
    params = inspect.signature(mod.AssetSelectComponent.f).parameters
    assert {"types", "initial", "value", "on_value"} <= set(params)

    seen = {}

    @solara.component
    def FakeSelector(types=None, value=None, on_value=None, initial=None, **_):
        seen["types"] = types
        seen["initial"] = initial
        seen["on_value"] = on_value
        solara.Text("stub")

    monkeypatch.setattr(mod, "AssetSelectComponent", FakeSelector)
    return seen


def _render(initial_entry, on_add=lambda entry: None):
    box, _rc = reacton.render(
        VariableModal(
            open_=solara.reactive(True),
            on_add=on_add,
            initial_entry=initial_entry,
        )
    )
    return box


def test_gee_layer_uses_the_asset_selector_restricted_to_images(monkeypatch):
    """No bare text field; the selector lists IMAGE assets only."""
    seen = _stub_selector(monkeypatch)
    box = _render(_custom_gee())
    assert seen["types"] == ["IMAGE"]
    labels = [f.label for f in _find(box, vw.TextField)]
    assert "GEE asset ID" not in labels


def test_editing_seeds_the_selector_with_the_stored_asset_id(monkeypatch):
    """The seed goes via ``initial``: pysepal treats ``value`` as output-only."""
    seen = _stub_selector(monkeypatch)
    _render(_custom_gee(asset_id="projects/p/assets/saved"))
    assert seen["initial"] == {"asset_id": "projects/p/assets/saved"}


def test_a_fresh_layer_passes_no_seed(monkeypatch):
    """An empty asset id must not seed a validation round-trip."""
    seen = _stub_selector(monkeypatch)
    _render(_custom_gee(asset_id=""))
    assert seen["initial"] is None


def test_selection_dict_is_unpacked_into_the_submitted_path(monkeypatch):
    """The selector publishes a dict; the entry keeps a plain asset-id path."""
    seen = _stub_selector(monkeypatch)
    added = []
    box = _render(_custom_gee(), on_add=added.append)
    seen["on_value"](
        {
            "asset_id": "projects/p/assets/y",
            "type": "IMAGE",
            "column": "ALL",
            "value": None,
        }
    )
    submit = next(
        b for b in _find(box, vw.Btn) if t("vars.modal.submit_add") in str(b.children)
    )
    submit.click()
    assert added and added[0]["path"] == "projects/p/assets/y"


def test_clearing_the_selector_clears_the_asset_id(monkeypatch):
    """A cleared / invalid selection publishes None and must not crash."""
    seen = _stub_selector(monkeypatch)
    added = []
    box = _render(_custom_gee(), on_add=added.append)
    seen["on_value"](None)
    submit = next(
        b for b in _find(box, vw.Btn) if t("vars.modal.submit_add") in str(b.children)
    )
    submit.click()
    assert added and not added[0]["path"].startswith("projects/p/")
