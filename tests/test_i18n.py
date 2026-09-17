"""Tests for gui.i18n: key lookup, interpolation, pluralization and parity."""

import json
import re

from gui import i18n


def test_t_resolves_simple_key():
    """A plain dotted key resolves to its catalog value."""
    assert i18n.t("common.save") == "Save"


def test_t_nested_dotted_key():
    """A dotted key resolves through nested catalog objects."""
    # tabs live in shell.json (added in Task 2) — here use common only
    assert i18n.t("common.cancel") == "Cancel"


def test_t_missing_key_returns_key():
    """An unknown key falls back to the raw dotted key."""
    assert i18n.t("does.not.exist") == "does.not.exist"


def test_t_interpolates_named_placeholders():
    """Named kwargs interpolate into ``{name}``-style placeholders."""
    # common.json carries a test-only template key
    assert i18n.t("common._test_fmt", name="X") == "hello X"


def test_plural_selects_variant():
    """plural() picks the one/other variant by count and injects n."""
    # common._test_one / common._test_other added in Step 3. Do NOT pass n= as a
    # kwarg: plural() takes n positionally and injects it into fmt itself (passing
    # both raises "multiple values for argument 'n'").
    assert i18n.plural(1, "common._test_one", "common._test_other") == "1 item"
    assert i18n.plural(3, "common._test_one", "common._test_other") == "3 items"


def test_app_available_locales():
    """The four shipped locales are exposed to the selector."""
    # pysepal's LocaleSelect only lists IETF codes present in its locale table:
    # there is no bare "es" (only es-ES/es-AR/...) and no bare "pt" (only
    # pt-PT/pt-BR), so Spanish ships as "es-ES" and Portuguese as "pt-BR".
    # French does have a bare "fr" entry, so it ships as "fr".
    assert set(i18n.app_available_locales()) == {"en", "es-ES", "fr", "pt-BR"}


def _flat_keys(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat_keys(v, key))
        else:
            out[key] = v
    return out


def _merged_for_lang(lang):
    merged = {}
    dupes = []
    for f in sorted((i18n.MESSAGES_DIR / lang).glob("*.json")):
        data = json.loads(f.read_text())
        for k in _flat_keys(data):
            if k in merged:
                dupes.append((lang, k, f.name))
        merged.update(_flat_keys(data))
    return merged, dupes


def test_no_duplicate_keys_within_language():
    """No key is declared twice across a language's message files."""
    for lang in i18n.app_available_locales():
        _, dupes = _merged_for_lang(lang)
        assert not dupes, f"Duplicate keys in {lang}: {dupes}"


def _translated_locales():
    return [ln for ln in i18n.app_available_locales() if ln != "en"]


def test_every_locale_has_full_key_parity_with_en():
    """Every translated locale declares exactly the en key set.

    A missing key silently falls back to English, so a gap is invisible in the
    UI — only a parity check surfaces it.
    """
    en_keys = set(_merged_for_lang("en")[0])
    for lang in _translated_locales():
        keys = set(_merged_for_lang(lang)[0])
        assert not (en_keys - keys), f"{lang} missing: {sorted(en_keys - keys)}"
        assert not (keys - en_keys), f"{lang} has orphans: {sorted(keys - en_keys)}"


def test_no_locale_has_empty_values():
    """No catalog value is empty.

    ``Translator.delete_empty`` drops empty strings and falls back to English,
    so ``"key": ""`` is an invisible gap rather than a visible placeholder.
    """
    for lang in _translated_locales():
        merged, _ = _merged_for_lang(lang)
        empty = sorted(k for k, v in merged.items() if not str(v).strip())
        assert not empty, f"Empty values in {lang}: {empty}"


def test_placeholder_sets_match_en_in_every_locale():
    """Each translated value carries exactly en's ``{placeholder}`` names.

    This is the one mechanical failure a reader cannot catch: ``t()`` swallows
    every exception and returns the raw dotted key, so a renamed placeholder
    ({nome} for {name}) yields no error and no log line — just a dotted key
    rendered into the UI.
    """
    placeholder = re.compile(r"{(\w+)}")
    en_merged, _ = _merged_for_lang("en")
    for lang in _translated_locales():
        merged, _ = _merged_for_lang(lang)
        for key, en_value in en_merged.items():
            expected = set(placeholder.findall(en_value))
            actual = set(placeholder.findall(merged[key]))
            assert expected == actual, (
                f"{lang} placeholder mismatch in '{key}': "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )


def test_relative_time_plural_keys_exist():
    """Relative-time and chip-count plural keys resolve for both languages."""
    for k in (
        "time.days_ago_one",
        "time.days_ago_other",
        "chips.models_one",
        "chips.models_other",
    ):
        assert i18n.t(k, n=1) != k  # resolves, not the raw key


def test_every_model_label_and_description_key_resolves():
    """Every MODEL_REGISTRY entry (and its params/variables) resolves."""
    from gui import i18n
    from gui.tile.train_tile import MODEL_REGISTRY

    for key, spec in MODEL_REGISTRY.items():
        assert i18n.t(spec["label_key"]) != spec["label_key"], key
        # description may be "" in es and fall back to en, but must resolve in en
        assert i18n.t(spec["description_key"]) != spec["description_key"], key
        # Train tile renders a structured summary above the prose description.
        summary = f"models.{key}.summary_md"
        assert i18n.t(summary) != summary, key
        for p in spec.get("params", []) + spec.get("variables", []):
            assert i18n.t(p["label_key"]) != p["label_key"], (key, p.get("key"))
            # Train tile renders a per-parameter hint resolved by convention
            # (models.<model>.params.<key>.hint) — every param must carry one.
            hint = f"models.{key}.params.{p['key']}.hint"
            assert i18n.t(hint) != hint, (key, p.get("key"))


def test_every_stat_card_key_resolves_in_every_locale():
    """Every ``card_*`` key ``stat_cards`` can emit has a label in all locales.

    The Statistics tab resolves card labels by convention —
    ``t(f"tiles.train.stats.{card['key']}")`` — so ``_code_keys`` below, which
    only sees literal ``t("...")`` calls, is blind to them. A typo or a card
    added without a catalog entry renders the raw dotted path in the UI with
    nothing failing, so the emitter's source is scraped instead.
    """
    from gui.scripts import model_stats_view

    src = (i18n.MESSAGES_DIR.parent / "scripts" / "model_stats_view.py").read_text()
    assert model_stats_view.stat_cards  # the file scraped is the one imported
    keys = sorted(set(re.findall(r'\badd(?:_num)?\(\s*["\'](card_\w+)["\']', src)))
    # guard: a refactor that renames the helpers must not silently scrape zero
    assert len(keys) >= 9, keys
    for lang in i18n.app_available_locales():
        merged, _ = _merged_for_lang(lang)
        for key in keys:
            dotted = f"tiles.train.stats.{key}"
            assert dotted in merged, (lang, dotted)
            assert str(merged[dotted]).strip(), (lang, dotted)


def test_every_predefined_variable_description_key_resolves():
    """Every PREDEFINED_CATALOGUE entry declares a description that resolves."""
    from gui import i18n
    from gui.scripts.predefined_variables import PREDEFINED_CATALOGUE

    for key, meta in PREDEFINED_CATALOGUE.items():
        dk = meta.get("description_key")
        assert dk, key
        assert i18n.t(dk) != dk, key


def _code_keys():
    """All literal keys passed to t("...") / plural(_, "one", "other") in gui/."""
    gui_dir = i18n.MESSAGES_DIR.parent
    t_pat = re.compile(r'\bt\(\s*["\']([\w.]+)["\']')
    plural_pat = re.compile(
        r'\bplural\([^,]+,\s*["\']([\w.]+)["\']\s*,\s*["\']([\w.]+)["\']'
    )
    keys = set()
    for f in gui_dir.rglob("*.py"):
        if "messages" in f.parts:
            continue
        src = f.read_text()
        keys.update(t_pat.findall(src))
        for one, other in plural_pat.findall(src):
            keys.update((one, other))
    return keys


def _en_keys():
    merged, _ = _merged_for_lang("en")  # helper from earlier in this file
    return set(merged)


def test_every_referenced_key_exists_in_en():
    """Every key referenced via t()/plural() in gui/ has an en catalog entry."""
    referenced = _code_keys()
    missing = sorted(
        k
        for k in referenced
        if k not in _en_keys() and not k.startswith("common._test")
    )
    assert not missing, f"t()/plural() keys with no en catalog entry: {missing}"


def test_t_accepts_key_named_format_placeholder():
    """A ``{key}`` placeholder doesn't collide with t()'s reserved param name."""
    # Regression: t()'s first param was named `key`, which collided with a
    # {key} format placeholder when callers passed key=... — the render crashed
    # with "t() got multiple values for argument 'key'". The lookup key must be
    # positional-only so a same-named placeholder interpolates instead.
    assert i18n.t("tiles.train.model_name_saved_as", key="m1") == "Saved as 'm1'."
    assert (
        i18n.t("tiles.dataset.success_registered", key="ds1")
        == "Dataset 'ds1' registered."
    )


def test_every_key_placeholder_call_site_renders():
    """Every ``{key}``-placeholder catalog value renders without leaking it."""
    # Every catalog value with a {key} placeholder must be callable with key=
    # without colliding with t()'s reserved parameter name (full blast radius
    # of the param-collision bug across dataset_tile/variables_tile/train_tile).
    for dotted in (
        "tiles.dataset.success_registered",
        "tiles.dataset.form_header_edit",
        "tiles.train.model_name_exists_warning",
        "tiles.train.model_name_saved_as",
        "tiles.train.confirm_delete_model_message",
        "tiles.train.confirm_overwrite_message",
    ):
        out = i18n.t(dotted, key="X")
        assert out != dotted and "{key}" not in out, dotted
    # error_toggle_map interpolates both key and exc
    out = i18n.t("tiles.variables.error_toggle_map", key="rivers", exc="boom")
    assert (
        "rivers" in out and "boom" in out and out != "tiles.variables.error_toggle_map"
    )


def test_no_hardcoded_step_numbers_in_messages():
    """Message catalogs must not hardcode step numbers.

    Step numbers derive from the STEPS registry; they drifted for months
    before the pipeline header.
    """
    import re

    # one alternative per shipped locale: en / es-ES / fr / pt-BR
    step_word = r"(?:Step|Paso|Étape|Etapa|Passo)"
    for lang in i18n.app_available_locales():
        for f in sorted((i18n.MESSAGES_DIR / lang).glob("*.json")):
            text = f.read_text()
            hits = re.findall(rf"{step_word} \d", text)
            assert not hits, (lang, f.name, hits)


def test_harmonization_rename_values():
    """Process/Post-process tabs are renamed Harmonization/Derived layers."""
    # Process -> Harmonization, Post-process -> Derived layers (2026-07-16)
    assert i18n.t("workflow.tab_process") == "Harmonization"
    assert i18n.t("workflow.tab_postprocess") == "Derived layers"
    assert i18n.t("tiles.process.header") == "### Harmonization"
    assert i18n.t("tiles.process.harmonize_all_button") == "Harmonize all"
    assert i18n.t("tiles.postprocess.header") == "### Derived layers"
    assert i18n.t("widgets.variable_list.processed_title") == "Harmonized variables"
    # keys orphaned by the compact tile were dropped (missing-key behavior)
    assert i18n.t("tiles.process.auto_utm_button") == "tiles.process.auto_utm_button"
    assert (
        i18n.t("tiles.process.run_processing_subtitle")
        == "tiles.process.run_processing_subtitle"
    )


def test_translator_ignores_machine_config(monkeypatch, tmp_path):
    """A machine-config locale must not leak in when no pysepal session exists.

    The root fix for the 21-test failure: a machine config holding a
    non-English locale must not leak into translated strings under pytest.
    """
    config = tmp_path / ".sepal-ui-config"
    config.write_text("[sepal-ui]\nlocale = es-ES\n")
    monkeypatch.setattr("pysepal.conf.config_file", config)
    monkeypatch.setattr("pysepal.translator.translator.config_file", config)

    i18n.reset_translator()
    try:
        translator = i18n.get_translator()
        assert translator._target == "en"
    finally:
        i18n.reset_translator()


def test_set_app_locale_swaps_translator_live(monkeypatch):
    """set_app_locale() swaps the active translator without a restart."""
    i18n.reset_translator()
    try:
        english = i18n.t("app.title")
        i18n.set_app_locale("es-ES")
        spanish = i18n.t("app.title")
        assert i18n.get_translator()._target == "es-ES"
        assert english != spanish  # app.title is translated in es-ES
        i18n.set_app_locale("en")
        assert i18n.t("app.title") == english
    finally:
        i18n.reset_translator()


def test_every_predefined_param_label_and_hint_resolves():
    """A catalogue param renders its label and hint through t().

    A missing key would surface the raw dotted key in the modal.
    """
    from gui import i18n
    from gui.scripts.predefined_variables import PREDEFINED_CATALOGUE

    for key, meta in PREDEFINED_CATALOGUE.items():
        for spec in meta.get("params", []):
            assert i18n.t(spec["label_key"]) != spec["label_key"], (key, spec["key"])
            assert i18n.t(spec["hint_key"]) != spec["hint_key"], (key, spec["key"])


def test_param_range_error_interpolates():
    """The validation message names the field and its bounds."""
    from gui import i18n

    msg = i18n.t("vars.modal.error_param_range", label="Tree cover", min=1, max=100)
    assert msg != "vars.modal.error_param_range"
    assert "Tree cover" in msg and "1" in msg and "100" in msg
