"""Read-only details of a registered dataset (target, features, year)."""

from typing import Callable, Optional

import reacton.ipyvuetify as rv
import solara

from gui.i18n import t
from gui.widget.details_fields import ro_field


def _label(var) -> str:
    """Variable name with its year when it has one (``forest (2020)``)."""
    name = getattr(var, "name", None) or str(var)
    year = getattr(var, "year", None)
    return f"{name} ({year})" if year else name


@solara.component
def DatasetDetailsDialog(dataset, open: bool, on_close: Callable[[], None]):
    """Key/value view of a dataset: target, features, year, categorical features.

    Args:
        dataset: the registered ``Dataset``, or None while the dialog is closed.
        open: whether the dialog is shown.
        on_close: () -> None; clears the tile's selected key.
    """
    ds: Optional[object] = dataset
    with rv.Dialog(
        v_model=bool(open and ds is not None),
        on_v_model=lambda v: None if v else on_close(),
        max_width="560px",
    ):
        with rv.Card():
            with rv.CardTitle():
                solara.Text(
                    t(
                        "tiles.dataset.details_title",
                        key=(getattr(ds, "name", None) or "") if ds else "",
                    )
                )
            with rv.CardText():
                if ds is not None:
                    features = list(getattr(ds, "features", None) or [])
                    categorical = [
                        _label(f)
                        for f in features
                        if str(
                            getattr(getattr(f, "raster_type", None), "value", None)
                            or getattr(f, "raster_type", None)
                        )
                        == "categorical"
                    ]
                    with solara.Column(style="gap:4px;"):
                        target = getattr(ds, "target", None)
                        ro_field(
                            t("tiles.dataset.target_variable_label"),
                            _label(target) if target is not None else None,
                        )
                        ro_field(
                            t("tiles.dataset.feature_variables_label"),
                            [_label(f) for f in features],
                        )
                        ro_field(
                            t("tiles.dataset.details_year_label"),
                            getattr(ds, "year", None),
                        )
                        ro_field(
                            t("tiles.dataset.details_categorical_label"),
                            categorical or None,
                        )
            with rv.CardActions(style_="justify-content: flex-end;"):
                solara.Button(
                    t("common.close"),
                    on_click=lambda: on_close(),
                    text=True,
                    small=True,
                )
