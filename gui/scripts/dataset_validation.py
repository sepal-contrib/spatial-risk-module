"""Choices and up-front checks for the New/Edit dataset dialog.

Data/logic only (returns i18n *keys*); label resolution happens in the widget.
Solara-free so the dialog and tests can both import it.

A temporal variable (one layer per year) is offered once per year, so picking
a variable also picks its year: every temporal variable in a dataset carries
its own year and nothing is left to align afterwards (#40).
"""

from typing import List, Optional, Sequence

from gui.scripts.variable_labels import year_label


def choice_value(name: str, year: Optional[int]) -> str:
    """The select value for ``name`` at ``year`` (the bare name when static)."""
    return name if year is None else f"{name}@{year}"


def variable_choices(project) -> List[dict]:
    """One choice per static variable and one per year of a temporal one.

    Each choice is ``{"value", "text", "name", "year"}``; ``year`` is None for
    a static variable. Temporal choices read ``defor (2015)``.
    """
    choices = []
    for name in project.list_unique_variable_names(source="processed"):
        if project.is_temporal(name, source="processed"):
            for year in project.get_variable_years(name, source="processed"):
                choices.append(
                    {
                        "value": choice_value(name, year),
                        "text": year_label(name, year),
                        "name": name,
                        "year": year,
                    }
                )
        else:
            choices.append({"value": name, "text": name, "name": name, "year": None})
    return choices


def dataset_form_error(
    choices: Sequence[dict], target: str, features: Sequence[str]
) -> Optional[str]:
    """None when the selection can be registered, else an i18n key.

    Args:
        choices: ``variable_choices(project)``.
        target: the selected target choice value ("" when unset).
        features: the selected feature choice values.
    """
    # A target removed from the project since an edited dataset was saved
    # has no item in the select either, so it reads as "nothing selected".
    if not target or target not in {c["value"] for c in choices}:
        return "tiles.dataset.error_target_required"
    if not features:
        return "tiles.dataset.error_features_required"
    return None
