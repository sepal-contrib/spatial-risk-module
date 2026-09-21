"""Dialog-side guard against unbounded sample requests (Track E, Task E1).

`n_samples=None` means "return every valid pixel" downstream
(`spatialrisk/sampling/random.py`, `stratified.py`) — on a country-scale
raster that is hundreds of GiB of point construction. Clearing the count
field used to silently produce exactly that `None`, in every strategy that
renders the field (random, stratified — not only systematic-spacing, which
intentionally means "no fixed count, use the grid spacing instead").

This file owns the dialog-side guard only: it never opens a raster and never
imports anything from `spatialrisk/sampling/`. The service-boundary guard in
`generate_points` itself is covered elsewhere.
"""

from types import SimpleNamespace

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

# See test_model_form_dialog_render.py: warm the translator before the first
# render — the first t() *during* a render breaks reacton's widget map.
t("common.cancel")

from gui.widget.sample_form_dialog import SampleFormDialog  # noqa: E402
from spatialrisk.project import Project  # noqa: E402


def _find(widget, cls, out=None):
    out = [] if out is None else out
    if isinstance(widget, cls):
        out.append(widget)
    for child in getattr(widget, "children", []) or []:
        if hasattr(child, "children") or isinstance(child, cls):
            _find(child, cls, out)
    return out


def _project_with_raster(name="fcc"):
    p = Project(project_name="p")
    p.processed_variables[name] = SimpleNamespace(data_type="raster", raster_type=None)
    return solara.reactive(p)


def _render(project=None, on_submit=None):
    project = project if project is not None else _project_with_raster()
    box, rc = reacton.render(
        SampleFormDialog(
            project=project,
            open_=solara.reactive(True),
            existing_names=frozenset(),
            running_names=frozenset(),
            on_submit=on_submit or (lambda entry: None),
        )
    )
    return box, rc


def _text_field(box, label):
    fields = _find(box, vw.TextField)
    return next(f for f in fields if f.label == label)


def _select(box, label):
    sels = _find(box, vw.Select)
    return next(s for s in sels if s.label == label)


def _create_button(box):
    label = t("tiles.sampling.generate_button")
    # The button carries a leading icon (icon_name="mdi-plus"), so its
    # children are [Icon, label] — match on membership, not exact equality.
    return next(b for b in _find(box, vw.Btn) if label in (b.children or []))


def _alert_text(box):
    alerts = _find(box, vw.Alert)
    return alerts[0].children[0] if alerts else None


def test_blank_count_blocks_submit_for_random():
    """Clearing the count field in random mode must not reach on_submit as None."""
    submitted = []
    box, _rc = _render(on_submit=lambda entry: submitted.append(entry))
    n_field = _text_field(box, t("tiles.sampling.n_samples_label"))
    n_field.v_model = ""

    _create_button(box).fire_event("click", None)

    assert not submitted
    assert _alert_text(box) == t("tiles.sampling.error_invalid_n_samples")


def test_zero_count_blocks_submit():
    """A count of 0 (or negative) is not a valid request either."""
    submitted = []
    box, _rc = _render(on_submit=lambda entry: submitted.append(entry))
    _text_field(box, t("tiles.sampling.n_samples_label")).v_model = "0"

    _create_button(box).fire_event("click", None)

    assert not submitted
    assert _alert_text(box) == t("tiles.sampling.error_invalid_n_samples")


def test_blank_count_blocks_submit_for_stratified():
    """The count field also renders (and must be guarded) for stratified."""
    submitted = []
    box, _rc = _render(on_submit=lambda entry: submitted.append(entry))
    _select(box, t("tiles.sampling.strategy_label")).v_model = "stratified"
    _text_field(box, t("tiles.sampling.n_samples_label")).v_model = ""

    _create_button(box).fire_event("click", None)

    assert not submitted
    assert _alert_text(box) == t("tiles.sampling.error_invalid_n_samples")


def test_valid_count_still_submits():
    """A well-formed positive count must not be blocked by the new guard."""
    submitted = []
    box, _rc = _render(on_submit=lambda entry: submitted.append(entry))
    _text_field(box, t("tiles.sampling.n_samples_label")).v_model = "5000"

    _create_button(box).fire_event("click", None)

    assert len(submitted) == 1
    assert submitted[0]["n_samples"] == 5000


def test_systematic_spacing_mode_keeps_intentional_none():
    """Systematic + explicit spacing must still submit n_samples=None."""
    submitted = []
    box, _rc = _render(on_submit=lambda entry: submitted.append(entry))
    _select(box, t("tiles.sampling.strategy_label")).v_model = "systematic"
    _select(box, t("tiles.sampling.define_grid_label")).v_model = "spacing"
    _text_field(box, t("tiles.sampling.spacing_label")).v_model = "1000"

    _create_button(box).fire_event("click", None)

    assert len(submitted) == 1
    assert submitted[0]["n_samples"] is None
    assert submitted[0]["spacing_m"] == 1000


def test_systematic_count_mode_is_still_guarded():
    """Systematic's OWN count sub-mode is a count mode too — blank must block."""
    submitted = []
    box, _rc = _render(on_submit=lambda entry: submitted.append(entry))
    _select(box, t("tiles.sampling.strategy_label")).v_model = "systematic"
    # sys_mode defaults to "n_samples" already, so the count field is showing.
    _text_field(box, t("tiles.sampling.n_samples_label")).v_model = ""

    _create_button(box).fire_event("click", None)

    assert not submitted
    assert _alert_text(box) == t("tiles.sampling.error_invalid_n_samples")


def test_deforisk_hint_warns_about_per_class_multiplication():
    """The deforisk hint must flag that the count multiplies by class count.

    deforisk allocation draws n_samples PER CLASS, not as a split total, so a
    legal-looking count (e.g. 10 000) can still produce a huge result once
    multiplied across many strata. This is intentionally a hint, not a hard
    cap: the class count is not known without reading the raster, which is
    exactly the expensive read this task avoids doing in the dialog.
    """
    box, _rc = _render()
    _select(box, t("tiles.sampling.strategy_label")).v_model = "stratified"
    _select(box, t("tiles.sampling.allocation_label")).v_model = "deforisk"

    field = _text_field(box, t("tiles.sampling.n_samples_label"))
    assert field.hint == t("tiles.sampling.n_samples_hint_deforisk")
    # The hint text itself must call out the multiplication risk explicitly
    # (not just "not a split total") -- guards against a translation-only
    # tweak that loses the warning.
    hint_lower = t("tiles.sampling.n_samples_hint_deforisk").lower()
    assert "class" in hint_lower


def test_error_key_resolves_and_is_distinct_from_spacing_error():
    """New key exists in the catalog and isn't accidentally aliased."""
    assert t("tiles.sampling.error_invalid_n_samples") != (
        "tiles.sampling.error_invalid_n_samples"
    )
    assert t("tiles.sampling.error_invalid_n_samples") != t(
        "tiles.sampling.error_invalid_spacing"
    )
