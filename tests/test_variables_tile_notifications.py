"""The Variables tile reports failures as pysepal toasts, not app_state reactives.

Rendering the whole tile needs a project, a map and the GEE stack, so the wiring
is asserted at the source level — the same approach as test_process_tile_wiring
and test_delete_confirm.
"""

import inspect

from gui.tile.variables_tile import VariablesTile, _run_download


def test_variables_tile_has_no_process_error_parameter():
    """VariablesTile signature no longer accepts process_error parameter."""
    sig = inspect.signature(VariablesTile.f)  # .f — the undecorated component
    assert "process_error" not in sig.parameters


def test_variables_tile_toasts_every_failure_path():
    """All failure paths in VariablesTile toast, not process_error."""
    src = inspect.getsource(VariablesTile.f)
    assert "process_error" not in src, "no reactive writes may survive"
    # Download is a tracked job: its message goes through tracked_job. The job
    # itself now lives in the module-level per-click worker (one thread per
    # click) rather than a component-scoped use_task, so look for it there.
    assert "error_format=" in inspect.getsource(_run_download)
    # Every direct toast (error or warning) passes the shared dwell constant.
    direct_toasts = src.count("notifications.error(") + src.count(
        "notifications.warning("
    )
    assert direct_toasts == src.count("ERROR_TOAST_TIMEOUT")
    for key in (
        "tiles.variables.error_download",
        "tiles.variables.error_toggle_map",
        "tiles.variables.error_no_project",
        "tiles.variables.error_base_raster_reset",
        "tiles.variables.error_base_raster_removed",
        "tiles.variables.error_add_variable",
        "tiles.variables.error_save_variable",
    ):
        assert key in src, f"{key} lost its call site"


def test_base_raster_notices_use_the_warning_channel():
    """Base-raster reset/removal report a side effect, not a failure.

    pysepal's NotificationBus drops every existing ERROR toast when a new one
    arrives, so these notices must not use notifications.error — that would
    silently wipe a real error toast (and vice versa). They use the warning
    channel instead, which dedups rather than clobbering.
    """
    src = inspect.getsource(VariablesTile.f)
    for key in (
        "tiles.variables.error_base_raster_reset",
        "tiles.variables.error_base_raster_removed",
    ):
        # Find every call site of this key and check it hangs off
        # notifications.warning(.
        idx = 0
        found = 0
        while True:
            idx = src.find(key, idx)
            if idx == -1:
                break
            found += 1
            call_start = src.rfind("notifications.", 0, idx)
            assert src[call_start : call_start + len("notifications.warning(")] == (
                "notifications.warning("
            ), f"{key} call site is not on the warning channel"
            idx += len(key)
        assert found > 0, f"{key} lost its call site"
    # Three sites total: two for reset (_do_add, on_save), one for removed (_do_remove).
    assert src.count("tiles.variables.error_base_raster_reset") == 2
    assert src.count("tiles.variables.error_base_raster_removed") == 1
