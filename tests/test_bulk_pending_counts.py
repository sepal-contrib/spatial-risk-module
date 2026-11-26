"""The two bulk buttons say what they act on: the layers still pending (#39).

"Download all layers (N)" counted only the layers still cloud-backed, so "all"
promised more than the button did, and "Harmonize all" gave no count at all
although it, too, skips every layer already on the reference grid. Both now
read "<verb> N pending", and N is the number the button would actually act on.

The Harmonize count is not re-derived here: it is the tile's own status (the
in-memory verdict, with the threaded disk verdict for layers that predate the
grid stamp), the same one the rows and the no-op disable are drawn from.
"""

import json
import threading
import time

import ipyvuetify as vw
import pytest
import reacton
import solara
from _notification_host import render_under_notifications

from gui import i18n
from gui.i18n import t
from gui.tile import process_tile
from spatialrisk.harmonization import HarmonizationStatus
from spatialrisk.project import Project
from spatialrisk.variables.local_raster_var import LocalRasterVar

Project._ensure_model_schemas()

TIMEOUT = 5.0

DOWNLOAD_KEY = "tiles.variables.download_button"
HARMONIZE_KEY = "tiles.process.harmonize_all_button"


# --- Strings -----------------------------------------------------------------


def _catalog(lang):
    """The raw ``tiles.json`` catalog of one shipped locale."""
    return json.loads((i18n.MESSAGES_DIR / lang / "tiles.json").read_text())["tiles"]


def test_english_labels_say_pending_and_carry_the_count():
    """The list footer's "<verb> N pending" wording, including zero."""
    assert t(DOWNLOAD_KEY, count=3) == "Download 3 pending"
    assert t(DOWNLOAD_KEY, count=0) == "Download 0 pending"
    assert t(HARMONIZE_KEY, count=2) == "Harmonize 2 pending"
    assert t(HARMONIZE_KEY, count=0) == "Harmonize 0 pending"


@pytest.mark.parametrize("lang", ["en", "es-ES", "fr", "pt-BR"])
def test_every_locale_counts_both_buttons_and_dropped_the_old_wording(lang):
    """A real translation of the new wording, not a leftover "all" label."""
    tiles = _catalog(lang)
    download = tiles["variables"]["download_button"]
    harmonize = tiles["process"]["harmonize_all_button"]
    assert "{count}" in download, download
    assert "{count}" in harmonize, harmonize
    old = {
        "en": ("Download all layers", "Harmonize all"),
        "es-ES": ("Descargar todas las capas", "Armonizar todo"),
        "fr": ("Télécharger toutes les couches", "Tout harmoniser"),
        "pt-BR": ("Baixar todas as camadas", "Harmonizar tudo"),
    }[lang]
    assert old[0] not in download
    assert old[1] not in harmonize


@pytest.mark.parametrize("lang", ["en", "es-ES", "fr", "pt-BR"])
def test_variables_help_names_the_download_button_as_it_reads(lang):
    """The info text tells the user which button to press — by its real name."""
    tiles = _catalog(lang)
    name = tiles["variables"]["download_button"].replace("{count}", "N")
    assert f"**{name}**" in tiles["variables"]["info_md"]


# --- Helpers -----------------------------------------------------------------


def _leaves(w):
    """Every string under ``w``: a Btn's icon name and its label."""
    for c in getattr(w, "children", None) or []:
        if isinstance(c, str):
            yield c
        else:
            yield from _leaves(c)


def _bulk_buttons(rc, icon):
    """The full-width buttons carrying ``icon`` (row actions are icon-only)."""
    return [b for b in rc.find(vw.Btn).widgets if b.block and icon in list(_leaves(b))]


def _bulk_button(rc, icon):
    """The one full-width button carrying ``icon`` (row actions are icon-only)."""
    hits = _bulk_buttons(rc, icon)
    assert len(hits) == 1, f"expected one block {icon} button, got {len(hits)}"
    return hits[0]


def _label(button, icon):
    """The button's text, without its icon name."""
    return "".join(s for s in _leaves(button) if s != icon).strip()


def _wait_until(predicate, timeout: float = TIMEOUT) -> bool:
    """Poll ``predicate`` until it is truthy or ``timeout`` elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _raster(p, name):
    """A bare, never-downloaded-anywhere local raster variable."""
    return LocalRasterVar.model_construct(
        name=name, data_type="raster", raster_type="continuous", path=None, project=p
    )


# --- Variables: Download N pending ------------------------------------------


class GEEVar:
    """Stands in for GEEVar: the tile picks cloud layers by class name."""

    def __init__(self, name):
        """Record the name; the tile only reads ``name`` and ``data_type``."""
        self.name = name
        self.data_type = "raster"


DOWNLOAD_ICON = "mdi-cloud-download-outline"


def _render_variables(monkeypatch, project):
    """Mount the Variables tile with the list and the modal stubbed out."""
    from gui.tile import variables_tile

    @solara.component
    def _StubList(project, on_remove, **_):
        solara.Text("")

    @solara.component
    def _StubModal(*args, **kwargs):
        solara.Text("")

    class _Notifier:
        """Stands in for pysepal's Notifier; nothing here raises a toast."""

        def error(self, message, *, timeout=None):
            """Drop it: no download runs in these tests."""

    monkeypatch.setattr(variables_tile, "SourceVariableList", _StubList)
    monkeypatch.setattr(variables_tile, "VariableModal", _StubModal)
    monkeypatch.setattr(variables_tile, "use_notifications", lambda: _Notifier())
    _box, rc = reacton.render(
        variables_tile.VariablesTile(project=project, map_=None), handle_error=False
    )
    return rc


def test_download_counts_only_the_layers_still_in_the_cloud(monkeypatch):
    """Two cloud layers and one local one: the button offers the two."""
    p = Project(project_name="missing-downloads")
    p.raw_variables["roads"] = GEEVar("roads")
    p.raw_variables["rivers"] = GEEVar("rivers")
    p.raw_variables["dem"] = _raster(p, "dem")
    rc = _render_variables(monkeypatch, solara.reactive(p))
    try:
        button = _bulk_button(rc, DOWNLOAD_ICON)
        assert _label(button, DOWNLOAD_ICON) == "Download 2 pending"
        assert button.disabled is False
    finally:
        rc.close()


def test_download_is_gone_with_nothing_pending(monkeypatch):
    """Everything local: no "Download 0 pending" button at all."""
    p = Project(project_name="nothing-to-download")
    p.raw_variables["dem"] = _raster(p, "dem")
    rc = _render_variables(monkeypatch, solara.reactive(p))
    try:
        assert _bulk_buttons(rc, DOWNLOAD_ICON) == []
    finally:
        rc.close()


# --- Harmonization: Harmonize N pending -------------------------------------


HAMMER = "mdi-hammer"


def _project(n_vars, *, legacy_outputs=False):
    """A project with a base raster and ``n_vars`` raw rasters.

    ``legacy_outputs`` registers an unstamped harmonized output per layer, the
    shape of a project saved before grid signatures existed — every layer is
    then ``unknown`` to the in-memory check and goes to the disk task.
    """
    p = Project(project_name="missing-harmonize")
    for i in range(n_vars):
        name = f"layer{i}"
        p.raw_variables[name] = _raster(p, name)
        if legacy_outputs:
            p.processed_variables[name] = _raster(p, name)
    p.base_raster = _raster(p, "layer0")
    return p


def _render_process(project):
    """Mount the real ProcessTile; caller closes the context."""
    processing = solara.reactive(False)
    _box, rc = render_under_notifications(
        lambda: process_tile.ProcessTile(project=project, processing=processing),
        handle_error=False,
    )
    return rc


def _harmonize_label(rc):
    """The bulk Harmonize button's text."""
    return _label(_bulk_button(rc, HAMMER), HAMMER)


def test_harmonize_counts_the_pending_layers(monkeypatch):
    """Two of three layers off the reference grid: the button says two."""
    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=["layer1", "layer2"], current=["layer0"]),
    )
    rc = _render_process(solara.reactive(_project(3), equals=lambda a, b: a is b))
    try:
        assert _harmonize_label(rc) == "Harmonize 2 pending"
        assert _bulk_button(rc, HAMMER).disabled is False
    finally:
        rc.close()


def test_harmonize_is_gone_when_all_are_current(monkeypatch):
    """Nothing left to harmonize: no button, as with Download."""
    monkeypatch.setattr(
        process_tile,
        "harmonization_status",
        lambda p: HarmonizationStatus(pending=[], current=list(p.raw_variables)),
    )
    rc = _render_process(solara.reactive(_project(2), equals=lambda a, b: a is b))
    try:
        assert _bulk_buttons(rc, HAMMER) == []
    finally:
        rc.close()


def test_harmonize_count_settles_once_the_disk_verdict_lands(monkeypatch):
    """Unstamped layers count as missing until the threaded check answers.

    A bulk run re-checks them on disk, so while their verdict is out they are
    work the button may still do — the count must not read "(0)" over a live
    button. Once the verdict lands, the count is the real one.
    """
    release = threading.Event()

    def _disk(project, keys=None):
        release.wait(TIMEOUT)
        return HarmonizationStatus(pending=["layer0"], current=["layer1"])

    monkeypatch.setattr(process_tile, "harmonization_status_from_disk", _disk)
    project = solara.reactive(
        _project(2, legacy_outputs=True), equals=lambda a, b: a is b
    )
    rc = _render_process(project)
    try:
        assert _harmonize_label(rc) == "Harmonize 2 pending"
        assert _bulk_button(rc, HAMMER).disabled is False
        release.set()
        assert _wait_until(
            lambda: _harmonize_label(rc) == "Harmonize 1 pending"
        ), f"the count never settled: {_harmonize_label(rc)!r}"
    finally:
        release.set()
        rc.close()
