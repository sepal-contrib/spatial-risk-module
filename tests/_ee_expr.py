"""Expression-recording fake of the ``ee`` module for builder tests.

Every constructor and method call returns an ``Expr`` whose text is the call
chain that produced it, e.g.
``ee.Image('USGS/SRTMGL1_003').select('elevation').clip(ee.FeatureCollection('AOI'))``.
Two builders produce the same Earth Engine expression exactly when their texts
match, so a pre-change builder copied verbatim into a test pins default parity.
Callables passed to EE (``collection.map(fn)``) render as ``<fn>``; floats are
rounded to 12 decimals. Nothing reaches Earth Engine: ``getInfo`` raises.
"""

import types

EE_TYPES = (
    "Algorithms",
    "Date",
    "Dictionary",
    "Feature",
    "FeatureCollection",
    "Filter",
    "Geometry",
    "Image",
    "ImageCollection",
    "Join",
    "List",
    "Number",
    "Projection",
    "Reducer",
    "String",
    "Terrain",
)


def _render(value):
    if isinstance(value, Expr):
        return value._text
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items())
        return "{" + ", ".join(f"{k!r}: {_render(v)}" for k, v in items) + "}"
    if isinstance(value, float):
        return repr(round(value, 12))
    if callable(value):
        return "<fn>"
    return repr(value)


def _call_text(args, kwargs):
    parts = [_render(a) for a in args]
    parts += [f"{k}={_render(v)}" for k, v in sorted(kwargs.items())]
    return ", ".join(parts)


class _ExprMeta(type):
    """Metaclass so class-level calls (``ee.Filter.inList``) record too."""

    def __getattr__(cls, name):
        """Record a static call such as ``ee.Terrain.slope(...)``."""
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: Expr(f"ee.{cls.__name__}.{name}({_call_text(a, k)})")


class Expr(metaclass=_ExprMeta):
    """A recorded Earth Engine expression; any method call extends it."""

    def __init__(self, text):
        """Wrap the expression text."""
        self._text = text

    def __getattr__(self, name):
        """Record a method call such as ``.select('elevation')``."""
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: Expr(f"{self._text}.{name}({_call_text(a, k)})")

    def getInfo(self):
        """Fail loudly: unit tests must never contact Earth Engine."""
        raise AssertionError("unit tests must never call getInfo")

    def __repr__(self):
        """The recorded expression text."""
        return self._text

    def __eq__(self, other):
        """Equal when the recorded texts are equal."""
        return isinstance(other, Expr) and other._text == self._text

    def __hash__(self):
        """Hash of the recorded text."""
        return hash(self._text)


def _ee_type(name):
    def __init__(self, *args, **kwargs):
        Expr.__init__(self, f"ee.{name}({_call_text(args, kwargs)})")

    return _ExprMeta(name, (Expr,), {"__init__": __init__, "__doc__": f"ee.{name}"})


def make_fake_ee():
    """A namespace standing in for the ``ee`` module."""
    return types.SimpleNamespace(**{name: _ee_type(name) for name in EE_TYPES})


def fake_aoi(fake_ee):
    """The AOI every builder test clips to: ``ee.FeatureCollection('AOI')``."""
    return fake_ee.FeatureCollection("AOI")


def patch_ee(monkeypatch, *modules):
    """Replace ``ee`` in every given module by one shared fake and return it."""
    fake = make_fake_ee()
    for module in modules:
        monkeypatch.setattr(module, "ee", fake, raising=False)
    return fake
