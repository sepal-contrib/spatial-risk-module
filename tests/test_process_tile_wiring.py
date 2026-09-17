"""Wiring: compact Harmonization tile — auto-UTM inside the EPSG field."""
import inspect


def test_base_projection_form_owns_the_hook():
    """The auto-UTM hook lives in the form, not in the tile."""
    # rv.use_event is a hook; ProcessTile early-returns before the form, so
    # the hook lives in a child component that is mounted conditionally.
    from gui.tile.process_tile import BaseProjectionForm

    src = inspect.getsource(BaseProjectionForm)
    assert 'append_icon="mdi-crosshairs-gps"' in src
    assert '"click:append"' in src
    assert "rv.use_event" in src
    # full-width section action
    assert "block=True" in src


def test_process_tile_is_compact():
    """The tile keeps the compact shape: no standalone UTM button, no subtitles."""
    from gui.tile.process_tile import ProcessTile

    src = inspect.getsource(ProcessTile)
    assert "BaseProjectionForm" in src
    # the standalone Auto (UTM) button and long subtitle are gone
    assert "auto_utm_button" not in src
    assert "run_processing_subtitle" not in src
    # run button is a full-width block action
    assert "block=True" in src
    # persistent hints dropped for compactness
    assert "persistent_hint" not in src
    # double-click guard on the run task is preserved
    assert "process_task.pending" in src
