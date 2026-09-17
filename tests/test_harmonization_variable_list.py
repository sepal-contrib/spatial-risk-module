"""Step 3 lists every harmonizable source variable with its status.

Before this list the only view of harmonization state was the Harmonized
variables table, which is empty until the first run — nothing told the user
which layers still needed work. The list mirrors Step 2's source list: every
variable, a Status column, a per-row action that is live only where there is
work to do, and the bulk button underneath.
"""

from types import SimpleNamespace

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t
from gui.widget.variable_list import (
    HarmonizationVariableList,
    harmonization_row_status,
)
from spatialrisk.harmonization import HarmonizationStatus, is_harmonizable


def _named(type_name, **attrs):
    return type(type_name, (), attrs)()


def _raster(name, year=None, **extra):
    return _named("LocalRasterVar", name=name, data_type="raster", year=year, **extra)


def _project():
    p = SimpleNamespace(
        raw_variables={
            "fc": _raster("fc", path="/x/fc.tif", year=2020),
            "roads": _named(
                "LocalVectorVar",
                name="roads",
                data_type="vector",
                year=None,
                active=True,
            ),
            "rivers": _named("GEEVar", name="rivers", data_type="raster", year=None),
            "inactive": _named(
                "LocalVectorVar",
                name="inactive",
                data_type="vector",
                year=None,
                active=False,
            ),
        },
        # Output keys carry the year (``output_key``): fc_2020, not fc.
        processed_variables={"fc_2020": _raster("fc", path="/x/out/fc.tif", year=2020)},
        base_raster=None,
    )
    return p


def test_is_harmonizable_matches_what_run_touches():
    """Rasters and active vectors only — exactly what the two *_all calls take."""
    assert is_harmonizable(_raster("a"))
    assert is_harmonizable(_named("LocalVectorVar", data_type="vector", active=True))
    assert not is_harmonizable(
        _named("LocalVectorVar", data_type="vector", active=False)
    )
    assert not is_harmonizable(_named("Other", data_type="table"))


def test_row_status_precedence():
    """Precedence: running > not_downloaded > checking > pending / harmonized."""
    status = HarmonizationStatus(pending=["roads"], current=["fc"])
    assert harmonization_row_status("fc", None, status, set(), False) == "harmonized"
    assert harmonization_row_status("roads", None, status, set(), False) == "pending"
    assert (
        harmonization_row_status("rivers", None, status, set(), True)
        == "not_downloaded"
    )
    # A run in flight on that key beats everything; a still-computing status
    # reads as "checking" for local layers only.
    assert harmonization_row_status("fc", None, status, {"fc"}, False) == "running"
    assert harmonization_row_status("fc", None, None, set(), False) == "checking"
    assert harmonization_row_status("rivers", None, None, set(), True) == (
        "not_downloaded"
    )


def _render(project, status, **kw):
    t("common.close")  # prime the catalog (first t() inside a first render)
    calls = []
    box, rc = reacton.render(
        HarmonizationVariableList(
            project=project,
            status=status,
            on_harmonize=calls.append,
            on_remove=lambda k: calls.append(("remove", k)),
            **kw,
        ),
        handle_error=False,
    )
    return rc, calls


def _buttons(rc):
    return [
        b
        for b in rc.find(vw.Btn).widgets
        if b.children and getattr(b.children[0], "children", None)
    ]


def _texts(rc):
    """Every string leaf in render order.

    ``rc.find`` does not descend into the nested ``rv.Html`` cells that
    ProductTable builds, so walk the widget tree by hand.
    """
    out = []

    def walk(w):
        for c in getattr(w, "children", None) or []:
            if isinstance(c, str):
                out.append(c)
            else:
                walk(c)

    for root in rc.find(vw.Html).widgets:
        walk(root)
    return out


def _status_labels(rc):
    labels = {
        t(f"widgets.product_table.status_{s}")
        for s in ("harmonized", "pending", "not_downloaded", "checking", "running")
    }
    return [s for s in _texts(rc) if s in labels]


def test_lists_every_harmonizable_variable_with_its_status():
    """One row per harmonizable raw variable, status resolved per row."""
    project = solara.reactive(_project())
    rc, _ = _render(project, HarmonizationStatus(pending=["roads"], current=["fc"]))
    try:
        names = _texts(rc)
        assert "fc" in names and "roads" in names and "rivers" in names
        assert "inactive" not in names  # Run never touches an inactive vector
        # A temporal layer repeats its name per year — the year column is
        # what tells the rows apart (same as the source list in Step 2).
        assert names.count("2020") == 1
        assert _status_labels(rc) == [
            t("widgets.product_table.status_harmonized"),
            t("widgets.product_table.status_pending"),
            t("widgets.product_table.status_not_downloaded"),
        ]
    finally:
        rc.close()


def test_harmonize_button_is_live_only_where_there_is_work():
    """The harmonized row's button is dimmed; pending / cloud rows can be run."""
    project = solara.reactive(_project())
    rc, calls = _render(project, HarmonizationStatus(pending=["roads"], current=["fc"]))
    try:
        sync = [b for b in _buttons(rc) if b.children[0].children == ["mdi-sync"]]
        assert [b.disabled for b in sync] == [True, False, False]
        sync[1].click()
        assert calls == ["roads"]
        # Only the harmonized row has an output to remove.
        trash = [
            b for b in _buttons(rc) if b.children[0].children == ["mdi-delete-outline"]
        ]
        assert len(trash) == 1
        trash[0].click()
        assert calls[-1] == ("remove", "fc_2020")  # the OUTPUT key
    finally:
        rc.close()


def test_running_row_spins_and_every_button_is_disabled():
    """A run in flight shows on its row and freezes every harmonize button."""
    project = solara.reactive(_project())
    rc, _ = _render(
        project,
        HarmonizationStatus(pending=["roads"], current=["fc"]),
        running_keys={"roads"},
        harmonize_disabled=True,
    )
    try:
        sync = [b for b in _buttons(rc) if b.children[0].children == ["mdi-sync"]]
        assert all(b.disabled for b in sync)
        assert [b.loading for b in sync] == [False, True, False]
        assert t("widgets.product_table.status_running") in _status_labels(rc)
    finally:
        rc.close()


def test_status_none_reads_as_checking():
    """Before the off-thread status resolves, local rows say so and stay inert."""
    project = solara.reactive(_project())
    rc, _ = _render(project, None)
    try:
        assert t("widgets.product_table.status_checking") in _status_labels(rc)
        sync = [b for b in _buttons(rc) if b.children[0].children == ["mdi-sync"]]
        # Nothing is known yet for local layers, so nothing local is runnable.
        assert [b.disabled for b in sync] == [True, True, False]
    finally:
        rc.close()
