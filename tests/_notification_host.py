"""Shared render helper: mount NotificationProvider before the element under test.

Not a test module (leading underscore) so pytest does not collect it.

pysepal 4's ``use_notifications(required=True)`` — the default every tile in
this app calls with — raises ``NotificationProviderError`` when nothing is
mounted above it (see tests/test_notifications_provider_contract.py). Every
render of a tile under test therefore needs a NotificationProvider mounted
first; this is the one place that wiring lives instead of a copy per module.
"""

import reacton
import solara
from pysepal.solara import NotificationProvider


@solara.component
def NotificationHost(build):
    """Mount ``NotificationProvider()``, then render ``build()``'s element.

    ``build`` is a zero-argument callable that constructs the element to
    test, not a pre-built one: the component call must happen inside this
    component's own render pass to attach as ``NotificationProvider``'s
    sibling. Sibling order matters here — the provider's own render creates
    the bus before ``build()``'s element resolves its notifier, so it must
    come first.
    """
    NotificationProvider()
    build()


def render_under_notifications(build, **render_kwargs):
    """``reacton.render`` an element under a mounted NotificationProvider.

    Returns whatever ``reacton.render`` returns (``box, rc``). Callers still
    own ``rc.close()`` (directly, or via an existing close-every-context
    fixture) — a render left open leaks the provider's bus at PROCESS scope
    and lets a later, provider-less render silently succeed instead of
    raising ``NotificationProviderError``.
    """
    return reacton.render(NotificationHost(build=build), **render_kwargs)
