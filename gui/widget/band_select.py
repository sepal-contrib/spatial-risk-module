"""Band list of a picked raster (an Earth Engine image or a local file).

Ported from the deforestation alerts app (gui/widget/raster_input.py): the
app's layers are single-band, so a multi-band source needs one band picked.
"""

import solara

from gui.i18n import t
from gui.widget.text_style import MUTED


def _file_bands(path: str) -> tuple:
    """``(text, 1-based index)`` per band: its description, else "Band N"."""
    import rasterio

    with rasterio.open(path) as src:
        return tuple(
            (d or f"Band {i}", i) for i, d in enumerate(src.descriptions, start=1)
        )


def use_band_list(gee_interface, origin: str, ref: str):
    """Task whose value is ``[(text, value)]``, one per band of ``ref``.

    An Earth Engine band's value is its name; a file band's is its 1-based
    index. Without ``gee_interface`` an asset lists no bands (exported whole).
    Holds a hook: call it unconditionally.
    """

    def _load():
        if not ref:
            return ()
        if origin == "gee":
            if gee_interface is None:
                return ()
            import ee

            names = gee_interface.get_info(ee.Image(ref).bandNames())
            return tuple((n, n) for n in names)
        return _file_bands(ref)

    # Threaded, not asyncio: works under a bare reacton.render (no event loop),
    # and the blocking GEE/rasterio calls stay off the websocket loop.
    return solara.lab.use_task(
        _load, dependencies=[origin, ref], prefer_threaded=True, raise_error=False
    )


def band_choices(task, ref) -> list:
    """The task's bands once read for ``ref``; [] while loading or on error.

    The task restarts in an effect, so in the render where ``ref`` changes it
    still holds the previous raster's list: an empty ``ref`` means none.
    """
    if not ref or not task.finished or task.error or not task.value:
        return []
    return list(task.value)


def bands_block_save(task, ref) -> bool:
    """Save waits while the picked raster's bands are read, or once they failed."""
    return bool(ref) and (bool(task.pending) or bool(task.error))


def band_status(task, ref) -> None:
    """Show "Reading the bands…" while loading, the error once failed, else nothing."""
    if not ref:
        return
    if task.pending:
        solara.Text(t("vars.modal.bands_loading"), style=MUTED)
    elif task.error:
        solara.Text(
            t("vars.modal.bands_failed", error=str(task.exception)),
            classes=["error--text"],
        )
