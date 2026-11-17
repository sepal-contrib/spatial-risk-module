"""A custom GEE layer picks its asset through pysepal's AssetSelectComponent.

The modal used to take the asset id in a bare text field, so a typo only
surfaced at download time. The selector lists the user's own assets, validates
whatever is typed against Earth Engine, and (for the modal) is restricted to
IMAGE assets — a TABLE is not a raster layer. The modal keeps storing a plain
asset-id string: the selector's ``{asset_id, type, column, value}`` dict is
unpacked at the boundary.

pysepal 4's ``value`` is two-way: an edited layer's stored id seeds the
selector through it, once. Echoing the modal's id back on later renders would
replace the selector's draft with a bare ``{"asset_id": ...}``.
"""

import asyncio
import inspect

import ipyvuetify as vw
import pysepal.solara.components.inputs.asset_select as asset_select_mod
import reacton
import solara

from gui.i18n import t

# See test_manage_projects_render: warm the translator before the first render.
t("common.cancel")

import gui.widget.variable_modal as mod  # noqa: E402
from gui.widget.variable_modal import VariableModal  # noqa: E402

SAVED = "projects/p/assets/saved"


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

    The stub takes exactly pysepal 4's keywords, so a keyword the real
    component rejects (the fork's ``initial``) fails here as it would at
    render. The signature assertion guards the stub itself.
    """
    params = set(inspect.signature(mod.AssetSelectComponent.f).parameters)
    assert {"types", "value", "on_value"} <= params
    assert "initial" not in params

    seen = {"values": []}

    @solara.component
    def FakeSelector(
        types=None,
        folder="",
        value=None,
        on_value=None,
        loading=False,
        on_loading=None,
        gee_interface=None,
    ):
        seen["types"] = types
        seen["values"].append(value)
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


def _submit(box):
    next(
        b for b in _find(box, vw.Btn) if t("vars.modal.submit_add") in str(b.children)
    ).click()


def test_gee_layer_uses_the_asset_selector_restricted_to_images(monkeypatch):
    """No bare text field; the selector lists IMAGE assets only."""
    seen = _stub_selector(monkeypatch)
    box = _render(_custom_gee())
    assert seen["types"] == ["IMAGE"]
    labels = [f.label for f in _find(box, vw.TextField)]
    assert "GEE asset ID" not in labels


def test_editing_seeds_the_selector_through_value(monkeypatch):
    """The restore seed goes in ``value``: pysepal 4 has no ``initial``."""
    seen = _stub_selector(monkeypatch)
    _render(_custom_gee(asset_id=SAVED))
    assert seen["values"][-1] == {"asset_id": SAVED}


def test_a_fresh_layer_passes_no_seed(monkeypatch):
    """An empty asset id must not seed a validation round-trip."""
    seen = _stub_selector(monkeypatch)
    _render(_custom_gee(asset_id=""))
    assert seen["values"][-1] is None


def test_a_pick_is_not_echoed_back_into_the_selector(monkeypatch):
    """The modal's own copy of the id never flows back in as a new ``value``.

    A pick publishes the full dict; the modal keeps only the id. Handing
    ``{"asset_id": id}`` back would read as an outside change and replace the
    selector's draft, so the selector only ever sees the seed it mounted with.
    """
    seen = _stub_selector(monkeypatch)
    added = []
    box = _render(_custom_gee(asset_id=SAVED), on_add=added.append)
    seen["on_value"](None)  # a new pick first clears the published value
    seen["on_value"](
        {
            "asset_id": "projects/p/assets/y",
            "type": "IMAGE",
            "column": "ALL",
            "value": None,
        }
    )
    assert all(v == {"asset_id": SAVED} for v in seen["values"]), seen["values"]
    _submit(box)
    assert added and added[0]["path"] == "projects/p/assets/y"


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
    _submit(box)
    assert added and added[0]["path"] == "projects/p/assets/y"


def test_clearing_the_selector_clears_the_asset_id(monkeypatch):
    """A cleared / invalid selection publishes None and must not crash."""
    seen = _stub_selector(monkeypatch)
    added = []
    box = _render(_custom_gee(), on_add=added.append)
    seen["on_value"](None)
    _submit(box)
    assert added and not added[0]["path"].startswith("projects/p/")


class _FakeGee:
    """Just enough of pysepal's GEE interface for an IMAGE pick, offline."""

    async def get_folder_async(self):
        return "projects/p/assets"

    async def get_assets_async(self, folder):
        return [{"id": SAVED, "type": "IMAGE"}]

    async def get_asset_async(self, asset_id):
        return {"type": "IMAGE"}


def test_the_real_selector_shows_the_stored_asset_and_keeps_it(monkeypatch):
    """Seeded through ``value=``, the selector shows the id and never clears it.

    Runs the genuine pysepal selector on a live loop (its lookups are async
    tasks) against a fake GEE interface. Once the asset has been validated,
    a re-render of the modal (typing a name) must leave the selector on the
    stored id, and the submitted entry must still carry it: nothing published
    None back into the modal.
    """
    monkeypatch.setattr(
        asset_select_mod, "get_current_gee_interface", lambda: _FakeGee()
    )
    added = []

    async def run():
        box, rc = reacton.render(
            VariableModal(
                open_=solara.reactive(True),
                on_add=added.append,
                initial_entry=_custom_gee(asset_id=SAVED),
            ),
            handle_error=False,
        )
        try:

            def settled():
                combos = _find(box, vw.Combobox)
                return bool(combos) and not combos[0].loading

            deadline = asyncio.get_running_loop().time() + 3
            while not settled() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert settled(), "the asset lookup never finished"
            assert _find(box, vw.Combobox)[0].v_model == SAVED

            name = next(
                f
                for f in _find(box, vw.TextField)
                if f.label == t("vars.modal.custom_name_label")
            )
            name.v_model = "renamed"
            await asyncio.sleep(0.05)
            assert _find(box, vw.Combobox)[0].v_model == SAVED
            _submit(box)
        finally:
            rc.close()

    asyncio.run(run())
    assert added and added[0]["path"] == SAVED
    assert added[0]["name"] == "renamed"
