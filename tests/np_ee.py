"""Numpy stand-in for the slice of the Earth Engine API the forest builders use.

Lets builder tests evaluate real builder code pixel by pixel, with no EE
session. Semantics mirror Earth Engine where the builders depend on them
(verified live 2026-10-09 on a Rondonia box):

* ``where(test, value)`` replaces only where ``test`` is valid and non-zero and
  ``value`` is valid; a masked test or value pixel keeps the input value; the
  output keeps the input's mask;
* comparisons and ``And`` / ``Or`` / ``Not`` carry the intersection of masks;
* ``updateMask(m)`` keeps a pixel where it was valid and ``m`` is valid and
  non-zero;
* ``clip(aoi)`` masks every band outside ``aoi.footprint``;
* ``remap(from, to)`` masks values not listed in ``from``.

Masks are boolean. EE's fractional AOI-edge masks export as valid pixels; the
download-level smoke script (scripts/smoke/defo_layers_smoke.py) covers them.
"""

# Methods mirror the Earth Engine names they stand in for; no docstrings.
# ruff: noqa: D102, D107

import types

import numpy as np


class NpImage:
    """A multi-band image: ``bands`` and ``masks`` map band name -> 2-D array."""

    def __init__(self, bands, masks=None):
        self.bands = {name: np.asarray(v, dtype=np.int64) for name, v in bands.items()}
        masks = masks or {}
        self.masks = {
            name: (
                np.asarray(masks[name], dtype=bool)
                if name in masks
                else np.ones(v.shape, dtype=bool)
            )
            for name, v in self.bands.items()
        }

    @classmethod
    def single(cls, value, mask):
        """A one-band image."""
        return cls({"B1": value}, {"B1": mask})

    @property
    def value(self):
        """The pixel values of a single-band image (select a band first)."""
        (name,) = self.bands
        return self.bands[name]

    @property
    def mask(self):
        """The validity mask of a single-band image."""
        (name,) = self.masks
        return self.masks[name]

    def _operand(self, other):
        if isinstance(other, NpImage):
            return other.value, other.mask
        shape = self.value.shape
        return np.full(shape, other, dtype=np.int64), np.ones(shape, dtype=bool)

    def _binary(self, other, op):
        value, mask = self._operand(other)
        return NpImage.single(op(self.value, value).astype(np.int64), self.mask & mask)

    def eq(self, other):
        return self._binary(other, np.equal)

    def neq(self, other):
        return self._binary(other, np.not_equal)

    def gte(self, other):
        return self._binary(other, np.greater_equal)

    def lt(self, other):
        return self._binary(other, np.less)

    def lte(self, other):
        return self._binary(other, np.less_equal)

    def And(self, other):
        return self._binary(other, lambda a, b: (a != 0) & (b != 0))

    def Or(self, other):
        return self._binary(other, lambda a, b: (a != 0) | (b != 0))

    def Not(self):
        return NpImage.single((self.value == 0).astype(np.int64), self.mask.copy())

    def where(self, test, value):
        test_value, test_mask = self._operand(test)
        new_value, new_mask = self._operand(value)
        hit = test_mask & (test_value != 0) & new_mask
        return NpImage.single(np.where(hit, new_value, self.value), self.mask.copy())

    def updateMask(self, mask_image):
        value, mask = self._operand(mask_image)
        return NpImage.single(self.value.copy(), self.mask & mask & (value != 0))

    def remap(self, from_values, to_values):
        out = np.zeros_like(self.value)
        listed = np.zeros(self.value.shape, dtype=bool)
        for src, dst in zip(from_values, to_values):
            hit = self.value == src
            out[hit] = dst
            listed |= hit
        return NpImage.single(out, self.mask & listed)

    def select(self, name):
        return NpImage({name: self.bands[name]}, {name: self.masks[name]})

    def clip(self, aoi):
        return NpImage(
            self.bands, {name: m & aoi.footprint for name, m in self.masks.items()}
        )

    def rename(self, _name):
        return NpImage.single(self.value, self.mask)

    def toByte(self):
        return self


class NpCollection:
    """``ee.ImageCollection(id)``: logs ``filterBounds`` args, mosaics to one image."""

    def __init__(self, image, log):
        self.image = image
        self.log = log

    def filterBounds(self, region):
        self.log.append(region)
        return self

    def mosaic(self):
        return self.image


class FakeAoi:
    """An AOI whose footprint is a boolean array; its bounds are ``"BOUNDS"``."""

    def __init__(self, footprint):
        self.footprint = np.asarray(footprint, dtype=bool)

    def geometry(self):
        return types.SimpleNamespace(bounds=lambda: "BOUNDS")


def fake_ee(shape, images=None, collections=None):
    """An ``ee`` namespace serving ``images`` / ``collections`` by asset id.

    A table with a single entry serves it for any id, so a test does not pin
    the asset version a builder reads. ``filter_bounds`` lists every
    ``filterBounds`` argument.
    """
    images = dict(images or {})
    collections = dict(collections or {})
    log = []

    def _pick(table, asset_id, kind):
        if asset_id in table:
            return table[asset_id]
        if len(table) == 1:
            return next(iter(table.values()))
        raise KeyError(f"fake ee has no {kind} {asset_id!r}")

    def Image(arg=None):
        if isinstance(arg, str):
            return _pick(images, arg, "image")
        return NpImage({"constant": np.full(shape, arg)})

    def ImageCollection(asset_id):
        return NpCollection(_pick(collections, asset_id, "collection"), log)

    return types.SimpleNamespace(
        Image=Image,
        ImageCollection=ImageCollection,
        Geometry=type("Geometry", (), {}),
        filter_bounds=log,
    )


def export(image, aoi, nodata=255):
    """What GEEVar._download writes for a categorical layer: clip, 255 fill, uint8."""
    clipped = image.clip(aoi)
    return np.where(clipped.mask, clipped.value, nodata).astype(np.uint8)
