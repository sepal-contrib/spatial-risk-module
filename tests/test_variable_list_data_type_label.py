"""The type chip reads "raster", not "DataType.raster".

``DataType`` is declared ``class DataType(str, Enum)``, so an ``isinstance(...,
str)`` test is true for its members as well as for a plain string — and the
label expression that leaned on that test returned the member itself, which the
chip renders through ``str()`` as "DataType.raster".

Both list widgets carried the same expression, and every existing test of them
builds variables with ``data_type="raster"`` — a plain string, the one input the
expression got right. These use the real enum, which is what the app has.
"""

import ipyvuetify as vw
import reacton
import solara

from gui.i18n import t

t("common.cancel")  # warm the translator before the first render

from gui.widget.variable_list import (  # noqa: E402
    HarmonizationVariableList,
    SourceVariableList,
)
from spatialrisk.harmonization import HarmonizationStatus  # noqa: E402
from spatialrisk.project import Project  # noqa: E402
from spatialrisk.variables.local_raster_var import LocalRasterVar  # noqa: E402
from spatialrisk.variables.models import DataType, RasterType  # noqa: E402

Project._ensure_model_schemas()


def _texts(rc):
    """Every string leaf in render order.

    ``rc.find`` does not descend into the nested ``rv.Html`` cells ProductTable
    builds, so walk from each of them by hand (see test_harmonization_variable_list).
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


def _project():
    p = Project(project_name="chips")
    p.raw_variables["altitude"] = LocalRasterVar.model_construct(
        name="altitude",
        path="/x/altitude.tif",
        project=p,
        # The enum member, exactly as validation and a project load produce it.
        data_type=DataType.raster,
        raster_type=RasterType.continuous,
        active=True,
    )
    return solara.reactive(p, equals=lambda a, b: a is b)


def test_source_list_chip_shows_the_bare_data_type():
    """Step 2's type chip."""
    _box, rc = reacton.render(
        SourceVariableList(project=_project(), on_remove=lambda key: None),
        handle_error=False,
    )
    texts = _texts(rc)

    assert "raster" in texts
    assert "DataType.raster" not in texts


def test_harmonization_list_chip_shows_the_bare_data_type():
    """Step 3's type chip — the same expression, copied."""
    _box, rc = reacton.render(
        HarmonizationVariableList(
            project=_project(),
            status=HarmonizationStatus(pending=["altitude"], current=[]),
            on_harmonize=lambda key: None,
        ),
        handle_error=False,
    )
    texts = _texts(rc)

    assert "raster" in texts
    assert "DataType.raster" not in texts


def test_a_plain_string_data_type_still_renders():
    """Variables built with a bare string (older manifests, test doubles).

    Built through ``model_construct``: a plain assignment would be validated
    straight back into the enum, which is the case above.
    """
    project = _project()
    project.value.raw_variables["altitude"] = LocalRasterVar.model_construct(
        name="altitude",
        path="/x/altitude.tif",
        project=project.value,
        data_type="raster",
        raster_type=RasterType.continuous,
        active=True,
    )

    _box, rc = reacton.render(
        SourceVariableList(project=project, on_remove=lambda key: None),
        handle_error=False,
    )

    assert "raster" in _texts(rc)
