"""materialize_raw_layers: progress log lines and on_progress events."""

import logging

from gui.scripts import process_actions


class _Local:
    def add_as_raw(self, auto_save=False):
        pass


class GEEVar:  # name matters: _is_geevar checks type(var).__name__ == "GEEVar"
    """Stand-in downloadable variable (dispatched on the class name)."""

    data_type = "raster"  # not DataType.vector -> takes the to_local_raster path

    def to_local_raster(self, overwrite=False):
        """Pretend-download returning a registered-able local var."""
        return _Local()


class _Project:
    def __init__(self, raw_variables):
        self.raw_variables = raw_variables


def test_download_emits_count_lines(caplog):
    """The run logs per-layer position lines and a final count."""
    project = _Project({"forest_gfc": GEEVar(), "dem": GEEVar()})
    with caplog.at_level(logging.INFO, logger="spatial_risk"):
        out = process_actions.materialize_raw_layers(project)
    assert out == ["forest_gfc", "dem"]
    messages = [r.getMessage() for r in caplog.records]
    assert "Downloading layer 1/2: forest_gfc" in messages
    assert "Downloading layer 2/2: dem" in messages
    assert "Downloaded 2 layer(s)." in messages


def test_on_progress_reports_layer_starts():
    """Each layer announces itself with zero tile counts before downloading."""
    project = _Project({"forest_gfc": GEEVar(), "dem": GEEVar()})
    calls = []
    process_actions.materialize_raw_layers(
        project, on_progress=lambda *a: calls.append(a)
    )
    assert ("forest_gfc", 0, 2, 0, 0) in calls
    assert ("dem", 1, 2, 0, 0) in calls
    assert calls.index(("forest_gfc", 0, 2, 0, 0)) < calls.index(("dem", 1, 2, 0, 0))


def test_on_progress_receives_geedim_tile_ticks():
    """Tile ticks from geedim's bar flow through with the layer's identity."""
    import io

    class _TiledGEEVar(GEEVar):
        def to_local_raster(self, overwrite=False):
            # mimic geedim's map_tiles driving its bar during the download
            import geedim.utils as gd_utils

            kwargs = gd_utils.get_tqdm_kwargs(desc="img", unit="tiles")
            bar = gd_utils.auto_leave_tqdm(
                range(3), total=3, file=io.StringIO(), **kwargs
            )
            for _ in bar:
                pass
            return _Local()

    class _TiledProjectVar(_TiledGEEVar):
        pass

    _TiledProjectVar.__name__ = "GEEVar"  # _is_geevar dispatches on the class name

    project = _Project({"rivers": _TiledProjectVar()})
    calls = []
    process_actions.materialize_raw_layers(
        project, on_progress=lambda *a: calls.append(a)
    )
    assert calls == [
        ("rivers", 0, 1, 0, 0),
        ("rivers", 0, 1, 1, 3),
        ("rivers", 0, 1, 2, 3),
        ("rivers", 0, 1, 3, 3),
    ]
