"""Row-stripe (blocked) raster scanning for sample generation.

WHY THIS MODULE EXISTS
----------------------
The original ``generate_points`` read the whole band, built a whole-raster
validity bool, then called ``np.where(valid)`` — two int64 arrays over every
valid pixel. Measured on a 196 Mpx fixture (fresh process, import baseline
subtracted) that is **21.6 bytes of working RAM per raster pixel**, i.e. about
45 GiB for a 2.22 Gpx country raster, which is the reported OOM. Nothing about
the *result* justifies that: a 20 000-point sample is 20 000 coordinates.

So the scan is split in two passes:

1. **Pass 1** counts valid pixels per raster value, one stripe at a time.
2. The strategy draws *ranks* — positions in the row-major enumeration of the
   valid pixels of a class — using exactly the RNG call the in-memory code
   made, so the drawn sample is unchanged for a given seed.
3. **Pass 2** re-walks the stripes and turns those ranks back into (row, col).

WHY FULL-WIDTH STRIPES AND NOT ``src.block_windows()``
------------------------------------------------------
The ranks are positions in the *global row-major* enumeration that
``np.where`` produced. A tiled raster's block windows enumerate tile by tile,
so pixel k of a block walk is not pixel k of a row-major walk and the
recovered points differ. This was verified: block windows are **not**
bit-identical, full-width stripes are.

WHY THE STRIPE HEIGHT IS ALIGNED TO THE TILE HEIGHT
---------------------------------------------------
A stripe boundary in the middle of a row of tiles makes GDAL decode those
tiles twice, once for each stripe that touches them. With a small block cache
that doubles decoding work for nothing, so the default stripe height is a whole
number of tile rows (512 for the usual 256 px tiles).

WHY PASS 2 IS STRIPE-OUTERMOST, CLASSES-INNERMOST
-------------------------------------------------
Classes share the stripe that has just been decoded, so the whole scan costs
two physical reads of the raster rather than ``1 + n_classes``.
"""
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

#: Stripes are sized toward this many rows, then snapped down to a whole
#: number of tile rows. 512 rows of a country-scale raster is ~25 MB.
_TARGET_STRIPE_ROWS = 512

#: How many values `_value_counts` histograms at a time. np.bincount upcasts its
#: input to intp internally, so chunking keeps that hidden temporary at a few
#: tens of MB instead of 8x a whole stripe.
_COUNT_CHUNK = 1 << 22


def valid_block(arr, nodata, mask_arr=None, mask_nodata=None):
    """Return the per-pixel validity of one stripe.

    This is the single definition of "valid" used by *both* passes — if the two
    passes disagreed even on one pixel, every rank after it would shift and the
    recovered sample would silently differ from the drawn one.

    It reproduces the original whole-raster predicate exactly:

    * ``~np.isnan(arr)`` then ``arr != nodata`` (the ``isnan`` term is a no-op
      on integer rasters, so it is only paid on floating ones);
    * ``mask != 0`` then ``mask != mask_nodata``.

    Note that a NaN *mask* pixel is still admitted as valid, because
    ``nan != 0`` and ``nan != nan`` are both True. That is a pre-existing bug
    (task E6, gated on a product decision); it is reproduced here on purpose so
    this change stays behaviour-preserving.
    """
    if np.issubdtype(arr.dtype, np.floating):
        v = ~np.isnan(arr)
        if nodata is not None:
            v &= arr != nodata
    elif nodata is not None:
        v = arr != nodata
    else:
        v = np.ones(arr.shape, dtype=bool)
    if mask_arr is not None:
        v &= mask_arr != 0
        if mask_nodata is not None:
            v &= mask_arr != mask_nodata
    return v


class RasterScan:
    """A raster (plus optional mask) walked as full-width horizontal stripes.

    Holds both datasets open for the lifetime of the scan so the two passes do
    not pay the open/overview cost twice, and exposes the metadata the
    strategies need without anyone having to read a band.

    Use as a context manager; ``iter_stripes()`` may be called once per pass.
    """

    def __init__(self, raster_path, mask_path=None, *, rows_per_stripe=None):
        """Open the raster (and mask) and plan the stripe windows.

        Raises ValueError if the mask is not co-registered with the raster, and
        closes whatever was opened before re-raising, so a failed scan never
        leaks a dataset handle.
        """
        import rasterio
        from rasterio.windows import Window

        self._src = rasterio.open(raster_path)
        self._msrc = None
        try:
            if mask_path is not None:
                self._msrc = rasterio.open(mask_path)
                if self._msrc.shape != self._src.shape:
                    # Co-registration check, kept verbatim from the in-memory
                    # version: ranks are shared between the two grids, so a
                    # shape mismatch would silently sample the wrong pixels.
                    raise ValueError(
                        f"Mask shape {self._msrc.shape} != raster shape "
                        f"{self._src.shape}; raster and mask must be "
                        "co-registered."
                    )
            self.rows_per_stripe = self._stripe_height(rows_per_stripe)
            self.windows: List[Window] = [
                Window(0, r, self.width, min(self.rows_per_stripe, self.height - r))
                for r in range(0, self.height, self.rows_per_stripe)
            ]
        except BaseException:
            self.close()
            raise

    def _stripe_height(self, requested: Optional[int]) -> int:
        """Pick a stripe height, aligned to the raster's own tile height.

        An explicit request is honoured as given (tests pin awkward heights on
        purpose); otherwise aim for ``_TARGET_STRIPE_ROWS`` rows but snap to a
        whole number of tile rows so no tile is decoded twice.
        """
        if requested is not None:
            if int(requested) < 1:
                raise ValueError("rows_per_stripe must be >= 1.")
            return int(requested)
        block_rows = int(self._src.block_shapes[0][0]) or 1
        if block_rows >= _TARGET_STRIPE_ROWS:
            return block_rows
        return (_TARGET_STRIPE_ROWS // block_rows) * block_rows

    # -- metadata ---------------------------------------------------------- #
    @property
    def height(self) -> int:
        """Raster height in pixels."""
        return int(self._src.height)

    @property
    def width(self) -> int:
        """Raster width in pixels."""
        return int(self._src.width)

    @property
    def shape(self) -> Tuple[int, int]:
        """Raster shape as ``(height, width)``."""
        return (self.height, self.width)

    @property
    def dtype(self) -> np.dtype:
        """dtype of band 1 — needed to build correctly-typed empty results."""
        return np.dtype(self._src.dtypes[0])

    @property
    def nodata(self):
        """Band-1 nodata value, or None."""
        return self._src.nodata

    @property
    def transform(self):
        """The raster's affine transform."""
        return self._src.transform

    @property
    def crs(self):
        """The raster's CRS."""
        return self._src.crs

    # -- iteration --------------------------------------------------------- #
    def iter_stripes(self) -> Iterator[Tuple[object, np.ndarray, np.ndarray]]:
        """Yield ``(window, band_values, validity)`` for each stripe in order.

        Stripes are yielded top to bottom and span the full width, which is
        what makes a stripe walk enumerate pixels in the same global row-major
        order as ``np.where`` over the whole band.
        """
        mask_nodata = self._msrc.nodata if self._msrc is not None else None
        for window in self.windows:
            arr = self._src.read(1, window=window)
            mask_arr = (
                self._msrc.read(1, window=window) if self._msrc is not None else None
            )
            yield window, arr, valid_block(arr, self.nodata, mask_arr, mask_nodata)

    def close(self):
        """Close the raster and the mask, ignoring datasets never opened."""
        for attr in ("_msrc", "_src"):
            src = getattr(self, attr, None)
            if src is not None:
                src.close()
                setattr(self, attr, None)

    def __enter__(self):
        """Return self; the datasets are already open."""
        return self

    def __exit__(self, *_exc):
        """Close both datasets."""
        self.close()
        return False


# --------------------------------------------------------------------------- #
# pass 1 — counting
# --------------------------------------------------------------------------- #
def _value_counts(vals: np.ndarray):
    """Per-value counts for one stripe's valid pixels, as (values, counts).

    ``np.unique`` sorts, which on an 8-bit class raster is ~4x slower than a
    histogram (91 ms vs 26 ms per 5 M-pixel stripe, measured) — and pass 1 pays
    it on every stripe. Small unsigned dtypes therefore take the ``bincount``
    route; everything else (floats, wide or signed integers) falls back to
    ``np.unique``, which has no domain restriction.
    """
    if vals.dtype.kind == "u" and vals.dtype.itemsize <= 2:
        width = 1 << (8 * vals.dtype.itemsize)
        total = np.zeros(width, dtype=np.int64)
        for start in range(0, vals.size, _COUNT_CHUNK):
            total += np.bincount(vals[start : start + _COUNT_CHUNK], minlength=width)
        present = np.nonzero(total)[0]
        return present, total[present]
    return np.unique(vals, return_counts=True)


def count_valid(scan: RasterScan) -> int:
    """Pass 1 for the strategies that only need the valid-pixel total.

    Random and systematic sampling rank pixels globally, not per class, so they
    can skip the compaction and the ``np.unique`` that ``count_values`` pays per
    stripe (~0.3 s per 25 Mpx stripe). The result is a Python ``int`` because a
    2.22 Gpx raster overflows signed int32.
    """
    n_valid = 0
    for _window, _arr, valid in scan.iter_stripes():
        n_valid += int(valid.sum())
    return n_valid


def count_values(scan: RasterScan) -> Tuple[Dict, int]:
    """Pass 1: count valid pixels per distinct raster value.

    Returns ``(raw_counts, n_valid)``. The keys are the *raw* values (Python
    scalars), not the ``int()``-cast class keys stratified sampling uses —
    ``class_tables`` needs both, and conflating them is exactly the
    pre-existing bug this change has to preserve.

    Peak memory is one stripe: the band values, the validity bool, and the
    compacted valid values. ``n_valid`` is a Python ``int`` because a
    2.22 Gpx raster overflows signed int32.
    """
    raw_counts: Dict = {}
    n_valid = 0
    for _window, arr, valid in scan.iter_stripes():
        vals = arr[valid]
        n_valid += int(vals.size)
        if vals.size:
            uniq, counts = _value_counts(vals)
            for value, count in zip(uniq.tolist(), counts.tolist()):
                raw_counts[value] = raw_counts.get(value, 0) + int(count)
    return raw_counts, n_valid


def class_tables(raw_counts: Dict) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Reproduce the historical stratified class bookkeeping, bug included.

    The in-memory code built its table as::

        {int(c): int((strata_values == c).sum()) for c in np.unique(sv)}

    The *key* is a lossy ``int()`` cast while the *count* is measured on the
    raw value, so float strata sharing an integer part collide and the largest
    raw value wins: strata ``[1.0, 1.0, 1.9]`` yield ``{1: 1}`` even though
    ``sv == 1`` has two members. That is a genuine bug, deliberately left to
    task E6 (it cannot be fixed without changing existing samples), so it is
    reproduced here rather than quietly repaired.

    Returns two tables, because the old code really did use two populations:

    * ``class_counts`` — what the allocator sees (the buggy one).
    * ``match_counts`` — how many valid pixels actually satisfy ``arr == key``,
      which is what ``np.where(sv == c)`` enumerated and therefore the
      population the rank draw must be made over. They differ only in the
      collision case above; keeping them apart is what makes the blocked draw
      bit-identical there too.
    """
    class_counts: Dict[int, int] = {}
    for value in sorted(raw_counts):  # ascending, i.e. np.unique order
        class_counts[int(value)] = raw_counts[value]

    match_counts: Dict[int, int] = {key: 0 for key in class_counts}
    for value, count in raw_counts.items():
        key = int(value)
        # `sv == c` compares the raw value against the int key, so only values
        # that are exactly equal to their own cast contribute.
        if value == key and key in match_counts:
            match_counts[key] += count
    return class_counts, match_counts


# --------------------------------------------------------------------------- #
# pass 2 — rank -> coordinate
# --------------------------------------------------------------------------- #
def empty_result(scan: RasterScan):
    """An empty ``(rows, cols, values)`` triple with the right dtypes."""
    return (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=scan.dtype),
    )


def _join(parts, scan: RasterScan):
    """Concatenate collected per-stripe pieces, or return empty arrays."""
    if not parts[0]:
        return empty_result(scan)
    return tuple(np.concatenate(p) for p in parts)


def collect_ranks(scan: RasterScan, wanted: Dict) -> Dict:
    """Pass 2: turn chosen global ranks back into pixel coordinates.

    ``wanted`` maps a class key to an **ascending** array of global ranks over
    that class's valid pixels; the key ``None`` means "every valid pixel",
    which is what simple random sampling enumerates. Returns
    ``{key: (rows, cols, values)}`` in the order the ranks were given, so the
    caller only has to undo its own sort.

    Two details here are load-bearing:

    * the rank window is **half-open with ``side="left"`` on both bounds**. A
      right-inclusive upper bound hands a rank that lands exactly on a stripe
      boundary to the previous stripe, which is off by one pixel;
    * ``seen`` is a Python ``int``, since a country-scale raster's pixel count
      does not fit in signed int32.

    Only the row span that actually contains wanted ranks is expanded with
    ``np.nonzero``: over a 512-row stripe of a 49 000 px-wide raster the full
    expansion would be ~300 MB of int64 for, typically, a few hundred points.
    """
    if not wanted:
        return {}
    seen = {key: 0 for key in wanted}
    done = {key: ranks.size == 0 for key, ranks in wanted.items()}
    rows: Dict = {key: [] for key in wanted}
    cols: Dict = {key: [] for key in wanted}
    vals: Dict = {key: [] for key in wanted}

    for window, arr, valid in scan.iter_stripes():
        row_off, col_off = int(window.row_off), int(window.col_off)
        for key, ranks in wanted.items():
            if done[key]:
                continue
            sel = valid if key is None else (valid & (arr == key))
            n_here = int(sel.sum())
            if n_here == 0:
                continue
            start = seen[key]
            lo = int(np.searchsorted(ranks, start, side="left"))
            hi = int(np.searchsorted(ranks, start + n_here, side="left"))
            seen[key] = start + n_here
            if hi >= ranks.size:
                done[key] = True
            if hi <= lo:
                continue
            local = ranks[lo:hi] - start
            # Narrow to the contiguous band of stripe rows holding those
            # ranks before expanding, so nonzero() stays proportional to the
            # points wanted instead of to the stripe.
            per_row = sel.sum(axis=1)
            cumulative = np.cumsum(per_row)
            r_lo = int(np.searchsorted(cumulative, local[0], side="right"))
            r_hi = int(np.searchsorted(cumulative, local[-1], side="right")) + 1
            base = int(cumulative[r_lo - 1]) if r_lo else 0
            band = sel[r_lo:r_hi]
            band_rows, band_cols = np.nonzero(band)
            take = local - base
            picked_rows = band_rows[take] + r_lo
            picked_cols = band_cols[take]
            rows[key].append(picked_rows + row_off)
            cols[key].append(picked_cols + col_off)
            vals[key].append(arr[picked_rows, picked_cols])
        if all(done.values()):
            # Every rank is accounted for; stop before the generator reads the
            # next stripe (this is what keeps a small sample from scanning the
            # whole raster twice).
            break

    return {key: _join((rows[key], cols[key], vals[key]), scan) for key in wanted}


def restore_order(order: np.ndarray, collected):
    """Scatter rank-sorted results back into the order they were drawn in.

    ``order = argsort(picks)`` means ``picks[order]`` is ascending, so result
    ``k`` of the sorted walk belongs at output position ``order[k]``. One sort
    and one scatter — not the double ``argsort`` the first draft used.
    """
    out = []
    for values in collected:
        scattered = np.empty_like(values)
        scattered[order] = values
        out.append(scattered)
    return tuple(out)


def collect_all(scan: RasterScan):
    """Every valid pixel, in global row-major order.

    This is the historical "take all" answer (``n_samples`` None, or larger
    than the number of valid pixels). It is inherently output-sized — there is
    no way to return a billion points cheaply — which is why task E1 rejects
    an unbounded request at the service boundary instead.
    """
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    for window, arr, valid in scan.iter_stripes():
        rr, cc = np.nonzero(valid)
        if rr.size == 0:
            continue
        rows.append(rr + int(window.row_off))
        cols.append(cc + int(window.col_off))
        vals.append(arr[rr, cc])
    return _join((rows, cols, vals), scan)


def collect_grid(scan: RasterScan, step_row: int, step_col: int):
    """Valid pixels sitting on a regular ``(step_row, step_col)`` grid.

    Only the grid *nodes* are ever materialised. The in-memory version built
    two int64 ``meshgrid`` arrays over the whole raster plus a full-shape bool
    — 33.1 GiB and 2.1 GiB respectively on a 2.22 Gpx raster, the single
    largest allocation anywhere in the pipeline. Here the column nodes are
    computed once and the row nodes per stripe, so nothing is proportional to
    the raster.

    Node order is global row-major (stripes ascend, grid rows ascend within a
    stripe, ``np.nonzero`` is row-major), matching the old
    ``meshgrid(..., indexing="ij").ravel()`` order exactly.
    """
    grid_cols = np.arange(0, scan.width, step_col)
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    for window, arr, valid in scan.iter_stripes():
        row_off, stripe_rows = int(window.row_off), int(window.height)
        # first grid row at or after the top of this stripe (ceil division)
        first = -(-row_off // step_row) * step_row
        if first >= row_off + stripe_rows:
            continue
        grid_rows = np.arange(first, row_off + stripe_rows, step_row)
        local_rows = grid_rows - row_off
        on_grid = valid[np.ix_(local_rows, grid_cols)]
        i, j = np.nonzero(on_grid)
        if i.size == 0:
            continue
        rows.append(grid_rows[i])
        cols.append(grid_cols[j])
        vals.append(arr[local_rows[i], grid_cols[j]])
    return _join((rows, cols, vals), scan)
