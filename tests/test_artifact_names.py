"""Unit tests for the shared artifact-naming helpers."""
from pathlib import Path

from gui.scripts.artifact_names import (
    default_pred_name,
    mask_name_token,
    name_field_messages,
    prediction_name_exists,
    sanitize_key,
    suggest_copy_name,
    suggest_name,
    suggest_version,
)
from spatialrisk import Project
from spatialrisk.predictions.prediction import Prediction


def test_suggest_name_first_free_slot():
    """The suggestion takes the lowest free slot, per prefix."""
    assert suggest_name("dataset", set()) == "dataset_1"
    assert suggest_name("dataset", {"dataset_1"}) == "dataset_2"
    assert suggest_name("dataset", {"dataset_1", "dataset_3"}) == "dataset_2"
    assert suggest_name("random", {"stratified_1"}) == "random_1"


def test_suggest_version_scopes_to_model_key():
    """Version numbering is per model key, not global."""
    assert suggest_version("glm", set()) == "v1"
    assert suggest_version("glm", {"glm_v1", "glm_v2"}) == "v3"
    # another model's versions don't block
    assert suggest_version("rf", {"glm_v1"}) == "v1"


def test_sanitize_key_matches_existing_tile_behaviour():
    """A user-typed name normalises to the same token the tiles produced."""
    # Same regex as train_tile._sanitize_name / inference_tile._sanitize_pred_name.
    assert sanitize_key("glm_v1__calibration") == "glm_v1__calibration"
    assert sanitize_key("  my run 1 ") == "my_run_1"
    assert sanitize_key("a/b:c") == "a_b_c"
    assert sanitize_key("***") == ""
    assert sanitize_key(None) == ""


def test_default_pred_name():
    """The suggestion pairs the model with the dataset it is applied to."""
    assert default_pred_name("glm_v1", "calibration") == "glm_v1__calibration"
    assert default_pred_name("", "calibration") == ""
    assert default_pred_name("glm_v1", "") == ""


def test_default_pred_name_appends_a_mask_token():
    """A mask changes the output, so it belongs in the suggested name."""
    assert (
        default_pred_name("glm_v1", "calibration", "forest_gfc_tc30")
        == "glm_v1__calibration__forest_gfc_tc30"
    )


def test_default_pred_name_without_a_mask_is_unchanged():
    """Families that take no mask keep the two-part name."""
    assert default_pred_name("mw_calib", "calibration", None) == "mw_calib__calibration"


def test_mask_name_token_uses_the_layer_key():
    """The layer key identifies the mask, sanitized for use in a path."""
    assert mask_name_token("forest gfc/tc30") == "forest_gfc_tc30"


def test_mask_name_token_marks_the_explicit_no_mask_choice():
    """Predicting everywhere is a choice, so it gets a token of its own.

    Without one, a masked and an unmasked run of the same model over the same
    dataset would both suggest '<model>__<dataset>' and collide.
    """
    assert mask_name_token(None) == "nomask"
    assert mask_name_token("") == "nomask"


def test_prediction_name_exists_detects_key_and_name():
    """A prediction name collides on either its registry key or its name."""
    project = Project(project_name="exists_test")
    # A name-keyed prediction (the new path).
    project.add_prediction(
        Prediction(
            name="run_a",
            path=Path("/tmp/a.tif"),
            model_key="glm_v1",
            dataset_name="calibration",
        ),
        key="run_a",
        auto_save=False,
    )

    assert prediction_name_exists(project, "run_a") is True  # matches key + name
    assert prediction_name_exists(project, "run_b") is False  # absent
    assert prediction_name_exists(project, "") is False  # empty never collides
    assert prediction_name_exists(None, "run_a") is False  # no project


def test_name_field_messages_states():
    """Each helper-line state maps to its message key and error flag."""
    # empty name: required; only an error once a submit was attempted
    assert name_field_messages("", False, False) == (
        "widgets.artifact_name.required",
        False,
    )
    assert name_field_messages("", False, True) == (
        "widgets.artifact_name.required",
        True,
    )
    # taken key: warning message, never a hard error
    assert name_field_messages("x", True, True) == (
        "widgets.artifact_name.exists_warning",
        False,
    )
    # free key: saved-as preview
    assert name_field_messages("x", False, False) == (
        "widgets.artifact_name.saved_as",
        False,
    )


def test_suggest_copy_name_first_free_copy():
    """A duplicate is '<name>_copy', then '_copy_2', '_copy_3'..."""
    assert suggest_copy_name("calib", {"calib"}) == "calib_copy"
    assert suggest_copy_name("calib", {"calib", "calib_copy"}) == "calib_copy_2"
    taken = {"calib", "calib_copy", "calib_copy_2"}
    assert suggest_copy_name("calib", taken) == "calib_copy_3"
