"""Old import path of the predefined layer catalogue, kept as a re-export.

The catalogue moved to the ``gui.scripts.predefined`` package (one module per
theme). New code imports from there. The private builder names stay
importable here because tests and notebooks reach for them through this path.
Do not run ``ruff --fix`` on this file: every import is a re-export.
"""

from gui.scripts.predefined import *  # noqa: F403
from gui.scripts.predefined.admin import _get_subj  # noqa: F401
from gui.scripts.predefined.aoi import _aoi_bounds  # noqa: F401
from gui.scripts.predefined.climate import (  # noqa: F401
    _get_precipitation,
    _get_temperature,
)
from gui.scripts.predefined.forest import _get_forest_gfc, _get_forest_tmf  # noqa: F401
from gui.scripts.predefined.infrastructure import (  # noqa: F401
    _get_protected_area,
    _get_roads,
    _get_towns,
)
from gui.scripts.predefined.terrain import _get_altitude, _get_slope  # noqa: F401
from gui.scripts.predefined.water import _get_rivers  # noqa: F401
