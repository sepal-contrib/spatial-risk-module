"""Recording stand-in for the ``ee`` module (no Earth Engine session).

``install(monkeypatch, module)`` replaces ``module.ee`` with a ``FakeEE`` and
returns it. Every call lands in ``fake.calls`` as ``(name, args)``:

* a constructor (``ee.ImageCollection(id)``) logs its class name;
* a static method (``ee.Image.constant(0)``) logs ``"Image.constant"``;
* a chained method (``.filterDate(a, b)``) logs its bare name.

Chained calls return the same recorder, so a whole builder chain records in
order. ``map`` logs ``("map", ())``, runs its function once on a fresh
recorder (so the mapped expression is recorded too) and logs
``("end_map", ())``. Arguments that are themselves recorders compare by
identity, so tests assert exact equality only on chains of plain values.
"""


class Rec:
    """Chainable recorder standing in for any ee object."""

    def __init__(self, calls, label="obj"):
        self._calls = calls
        self._label = label

    def __getattr__(self, method):
        if method.startswith("__"):
            raise AttributeError(method)

        def _call(*args, **kwargs):
            if method == "map":
                self._calls.append(("map", ()))
                args[0](Rec(self._calls, "mapped"))
                self._calls.append(("end_map", ()))
                return self
            self._calls.append((method, args))
            return self

        return _call

    def __repr__(self):
        return f"<Rec {self._label}>"


class _Factory:
    """``ee.<Name>``: a callable constructor that also has static methods."""

    def __init__(self, calls, name):
        self._calls = calls
        self._name = name

    def __call__(self, *args, **kwargs):
        self._calls.append((self._name, args))
        return Rec(self._calls, self._name)

    def __getattr__(self, method):
        if method.startswith("__"):
            raise AttributeError(method)

        def _static(*args, **kwargs):
            self._calls.append((f"{self._name}.{method}", args))
            return Rec(self._calls, f"{self._name}.{method}")

        return _static


class FakeEE:
    """Drop-in for a theme module's ``ee``."""

    def __init__(self):
        self.calls = []
        for name in (
            "Date",
            "FeatureCollection",
            "Filter",
            "Image",
            "ImageCollection",
            "List",
            "Number",
            "Reducer",
        ):
            setattr(self, name, _Factory(self.calls, name))

    def methods(self):
        """Just the call names, in order."""
        return [name for name, _args in self.calls]


def install(monkeypatch, module):
    """Patch ``module.ee`` with a fresh ``FakeEE`` and return it."""
    fake = FakeEE()
    monkeypatch.setattr(module, "ee", fake)
    return fake
