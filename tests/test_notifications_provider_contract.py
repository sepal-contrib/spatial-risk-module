"""pysepal 4's NotificationProvider contract.

Every tile that calls ``use_notifications()`` must be rendered under the
app's ``NotificationProvider`` — pysepal 4 raises ``NotificationProviderError``
synchronously at render when nothing is mounted, where the fork silently
handed back a no-op notifier. Pinned here against ``VariablesTile``; the 49
other tests fixed alongside this one (see the migration notes) cover every
other call site.
"""

import pytest
import reacton
import solara
from _notification_host import NotificationHost
from pysepal.solara import NotificationProviderError
from pysepal.solara.notifications.bus import get_current_bus

from gui.tile import variables_tile
from spatialrisk.project import Project


def _tile():
    return variables_tile.VariablesTile(
        project=solara.reactive(Project(project_name="p"))
    )


def test_tile_raises_without_provider():
    """No provider mounted anywhere in the process.

    The real raise, not one swallowed by a leaked bus from an earlier,
    unclosed render.
    """
    assert get_current_bus() is None
    with pytest.raises(NotificationProviderError):
        reacton.render(_tile(), handle_error=False)


def test_tile_renders_under_provider():
    """Mounting the provider first lets the tile resolve a real notifier.

    Sibling order matters: rendering the tile before the provider would
    still find no bus mounted and raise.
    """
    box, rc = reacton.render(NotificationHost(build=_tile), handle_error=False)
    try:
        assert box is not None
    finally:
        rc.close()
