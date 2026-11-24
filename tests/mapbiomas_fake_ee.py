"""Recording stand-in for the ``ee`` module, for MapBiomas builder tests.

Every constructor and method call appends ``(name, args, kwargs)`` to
``fake.calls`` and returns a new chainable ``Node``. ``getInfo()`` returns
``fake.info``. Patch it onto the module that owns the builder::

    fake = FakeEE()
    monkeypatch.setattr(mapbiomas, "ee", fake)
"""

import types


class Node:
    """A lazily built ee object: any method call is recorded and chained."""

    def __init__(self, fake, origin):
        """Remember the owning fake and the call that made this node."""
        self._fake = fake
        self.origin = origin

    def __getattr__(self, method):
        """Record any method call and return a new chainable node."""
        if method.startswith("__"):
            raise AttributeError(method)

        def _call(*args, **kwargs):
            self._fake.calls.append((method, args, kwargs))
            if method == "getInfo":
                return self._fake.info
            return Node(self._fake, method)

        return _call


class FakeEE:
    """Namespace with the parts of ``ee`` that the MapBiomas builders use."""

    def __init__(self, info=None):
        """Start with no calls; ``getInfo()`` returns ``info``."""
        self.calls = []
        self.info = info
        fake = self

        class Geometry(Node):
            """``ee.Geometry`` usable with ``isinstance``."""

            def __init__(self, *args, **kwargs):
                Node.__init__(self, fake, "Geometry")

        self.Geometry = Geometry
        self.Image = self._ctor("Image")
        self.ImageCollection = self._ctor("ImageCollection")
        self.Filter = types.SimpleNamespace(eq=self._ctor("Filter.eq"))
        self.Reducer = types.SimpleNamespace(mean=self._ctor("Reducer.mean"))

    def _ctor(self, name):
        def _make(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return Node(self, name)

        return _make

    def names(self):
        """Recorded call names, in order."""
        return [name for name, _args, _kwargs in self.calls]

    def args_of(self, name):
        """Positional args of every recorded ``name`` call."""
        return [args for call, args, _kwargs in self.calls if call == name]

    def kwargs_of(self, name):
        """Keyword args of every recorded ``name`` call."""
        return [kwargs for call, _args, kwargs in self.calls if call == name]
