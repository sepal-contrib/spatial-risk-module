"""Tests for the project panel's pure helpers."""

from datetime import datetime

from gui.scripts.project_io import ProjectInfo
from gui.scripts.project_ui_helpers import (
    NameValidation,
    aoi_project_name,
    compute_app_title,
    delete_confirm_valid,
    filter_project_infos,
    format_last_saved,
    format_relative,
    format_size,
    manage_projects_label,
    overwrite_needed,
    project_count_chips,
    validate_project_name,
)


class _P:
    def __init__(self, name):
        self.project_name = name


def _info(**kw):
    base = dict(
        name="p",
        raw_count=0,
        processed_count=0,
        model_count=0,
        modified=None,
        readable=True,
        trained_model_count=0,
        prediction_count=0,
    )
    base.update(kw)
    return ProjectInfo(**base)


def test_chips_order_and_basic_labels():
    """Chips order and basic labels."""
    chips = project_count_chips(_info(raw_count=9, processed_count=13))
    labels = [c.label for c in chips]
    assert labels[0] == "9 raw"
    assert labels[1] == "13 harmonized"
    assert labels[2].endswith("models")  # "0 models"
    assert labels[3] == "0 predictions"
    # raw / processed are never accented
    assert chips[0].accent is False and chips[1].accent is False


def test_models_chip_no_trained_suffix_when_zero_models():
    """Models chip no trained suffix when zero models."""
    models_chip = project_count_chips(_info(model_count=0))[2]
    assert models_chip.label == "0 models"
    assert models_chip.accent is False


def test_models_chip_shows_trained_count_and_accents():
    """Models chip shows trained count and accents."""
    models_chip = project_count_chips(_info(model_count=2, trained_model_count=1))[2]
    assert models_chip.label == "2 models (1 trained)"
    assert models_chip.accent is True


def test_models_present_but_none_trained_is_not_accented():
    """Models present but none trained is not accented."""
    models_chip = project_count_chips(_info(model_count=2, trained_model_count=0))[2]
    assert models_chip.label == "2 models (0 trained)"
    assert models_chip.accent is False


def test_predictions_chip_accent_toggles_on_count():
    """Predictions chip accent toggles on count."""
    none = project_count_chips(_info(prediction_count=0))[3]
    some = project_count_chips(_info(prediction_count=2))[3]
    assert none.label == "0 predictions" and none.accent is False
    assert some.label == "2 predictions" and some.accent is True


def test_app_title_no_project():
    """App title no project."""
    assert compute_app_title(None, False) == "Spatial Risk"


def test_app_title_clean():
    """App title clean."""
    assert compute_app_title(_P("mtq"), False) == "Spatial Risk"


def test_app_title_dirty():
    """App title dirty."""
    assert compute_app_title(_P("mtq"), True) == "Spatial Risk ●"


def test_format_relative_buckets():
    """Format relative buckets."""
    now = datetime(2026, 6, 17, 12, 0, 0)
    assert format_relative(datetime(2026, 6, 17, 11, 59, 30), now) == "just now"
    assert format_relative(datetime(2026, 6, 17, 11, 55, 0), now) == "5 min ago"
    assert format_relative(datetime(2026, 6, 17, 9, 0, 0), now) == "3 hr ago"
    assert format_relative(datetime(2026, 6, 15, 12, 0, 0), now) == "2 days ago"


def test_format_last_saved_never():
    """Format last saved never."""
    assert format_last_saved(None, datetime(2026, 6, 17, 12, 0, 0)) == "never saved"


def test_format_last_saved_relative():
    """Format last saved relative."""
    now = datetime(2026, 6, 17, 12, 5, 0)
    when = datetime(2026, 6, 17, 12, 0, 0)
    assert format_last_saved(when, now) == "saved 5 min ago"


def test_validate_empty():
    """Validate empty."""
    v = validate_project_name("   ", [])
    assert isinstance(v, NameValidation)
    assert v.valid is False
    assert "empty" in v.error.lower()


def test_validate_bad_chars():
    """Validate bad chars."""
    v = validate_project_name("a/b", [])
    assert v.valid is False
    assert v.error


def test_validate_ok_unique():
    """Validate ok unique."""
    v = validate_project_name("  my-proj ", ["other"])
    assert v.valid is True
    assert v.cleaned == "my-proj"
    assert v.exists is False
    assert v.error is None


def test_validate_ok_but_exists():
    """Validate ok but exists."""
    v = validate_project_name("mtq", ["mtq", "other"])
    assert v.valid is True
    assert v.exists is True


def test_overwrite_needed():
    """Overwrite needed."""
    assert overwrite_needed("mtq", None, ["mtq"]) is True
    assert overwrite_needed("mtq", datetime(2026, 1, 1), ["mtq"]) is False
    assert overwrite_needed("fresh", None, ["mtq"]) is False


def test_manage_projects_label_with_projects():
    """Manage projects label with projects."""
    assert manage_projects_label(3) == "Manage projects… (3)"
    assert manage_projects_label(1) == "Manage projects… (1)"


def test_manage_projects_label_zero_or_unknown():
    """Manage projects label zero or unknown."""
    # 0 saved → invite to create; None (scan failed) → same neutral copy.
    assert manage_projects_label(0) == "No saved projects yet"
    assert manage_projects_label(None) == "No saved projects yet"


def test_filter_project_infos_is_case_insensitive_substring():
    """Filter project infos is case insensitive substring."""
    infos = [_info(name="GUY"), _info(name="mtq_ui"), _info(name="testjuly")]
    assert [i.name for i in filter_project_infos(infos, "mtq")] == ["mtq_ui"]
    assert [i.name for i in filter_project_infos(infos, "guy")] == ["GUY"]
    assert [i.name for i in filter_project_infos(infos, "U")] == [
        "GUY",
        "mtq_ui",
        "testjuly",
    ]


def test_filter_project_infos_blank_query_returns_all_in_order():
    """Filter project infos blank query returns all in order."""
    infos = [_info(name="b"), _info(name="a")]
    assert [i.name for i in filter_project_infos(infos, "")] == ["b", "a"]
    assert [i.name for i in filter_project_infos(infos, "   ")] == ["b", "a"]


def test_filter_project_infos_no_match_is_empty():
    """Filter project infos no match is empty."""
    assert filter_project_infos([_info(name="GUY")], "zzz") == []


def test_format_size_units():
    """Format size units."""
    assert format_size(0) == "0 B"
    assert format_size(512) == "512 B"
    assert format_size(1024) == "1.0 KB"
    assert format_size(3 * 1024 * 1024) == "3.0 MB"
    assert format_size(int(3.2 * 1024**3)) == "3.2 GB"


def test_delete_confirm_valid_requires_exact_name():
    """Delete confirm valid requires exact name."""
    assert delete_confirm_valid("GUY", "GUY") is True
    assert delete_confirm_valid("  GUY  ", "GUY") is True  # surrounding space tolerated
    assert delete_confirm_valid("guy", "GUY") is False  # case-sensitive
    assert delete_confirm_valid("GU", "GUY") is False
    assert delete_confirm_valid("", "GUY") is False


def test_aoi_project_name_appends_yyyymmdd():
    """Aoi project name appends yyyymmdd."""
    when = datetime(2026, 6, 23, 14, 30, 0)
    assert aoi_project_name("San Marino", when) == "San Marino_20260623"


def test_aoi_project_name_pads_month_and_day():
    """Aoi project name pads month and day."""
    when = datetime(2026, 1, 5, 0, 0, 0)
    assert aoi_project_name("amazon", when) == "amazon_20260105"
