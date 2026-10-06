"""Step 1 — AOI selection tile."""

import solara
from pysepal.solara.components.aoi import AoiView

from gui.i18n import t
from gui.scripts.map_helpers import draw_aoi_on_map


@solara.component
def AoiTile(map_, gee_interface, aoi_result, aoi_spec, restore_signal, loading):
    """AOI selection step using pysepal's AoiView, restored through ``spec=``.

    Args:
        map_: SepalMap instance (shared with left panel).
        gee_interface: Current GEEInterface from session.
        aoi_result: Reactive holding the current AoiResult (or None).
        aoi_spec: Reactive holding the picker's AoiSpec (or None): AoiView's
            two-way spec channel. A project load writes the restored spec
            here, and AoiView re-runs the picker on it (no remount).
        restore_signal: project_loaded_signal value. Each change is a project
            switch (load, New or Close), on which the picker is reset when
            AoiView would not restore it by itself.
        loading: Reactive bool for loading state.
    """
    clear_ref = solara.use_ref(None)
    last_signal = solara.use_ref(restore_signal)

    def reset_picker_on_switch():
        # A switch only, never the mount: a mounting picker restores from
        # aoi_spec itself, and a reset here would wipe that restore.
        if restore_signal == last_signal.current:
            return
        last_signal.current = restore_signal

        # A switch installs its AoiResult before bumping the signal but writes
        # the spec only after it (do_load), so aoi_spec still holds the
        # outgoing selection here: the incoming spec is read off the result.
        restored = aoi_result.peek()
        incoming = getattr(restored, "spec", None)
        if incoming is not None and (
            incoming.method == "DRAW" or incoming != aoi_spec.peek()
        ):
            # A new spec, AoiView restores by itself. A drawing is never reset:
            # the reset's dc.clear() is a browser round trip that lands after
            # the restore's re-seed and empties it; the same drawing is still
            # in the control anyway.
            return

        clear = clear_ref.current
        if clear is None:
            return
        # v4 treats spec=None from outside as a no-op, so without this a switch
        # to a project with no spec (New, Close, no AOI, a legacy SHAPE/POINTS
        # manifest) kept the previous selection, and Select would re-run it.
        # An equal spec is not re-applied either, and the switch's overlay
        # wipe would leave that AOI off the map; after the reset, do_load's
        # spec write is a change again.
        clear()
        dc = getattr(map_, "dc", None)
        if dc is not None:
            # dc.clear() only messages the browser, so the kernel's data stays
            # equal and re-seeding the same drawing later would be a no-op.
            dc.data = []
        if restored is not None:
            # clear() nulls the value too; the loaded AOI stays the app's.
            aoi_result.set(restored)
            # clear() also drops every "aoi" map layer, including the loaded
            # AOI the shell's switch effect has already drawn (it runs before
            # this one), so draw it again. A no-op for geometry-less AOIs.
            draw_aoi_on_map(map_, restored)

    solara.use_effect(reset_picker_on_switch, [restore_signal])

    with solara.Column(style="gap: 16px;"):
        solara.Text(t("tiles.aoi.description"))

        AoiView(
            value=aoi_result,
            spec=aoi_spec,
            clear_ref=clear_ref,
            # Each restore re-runs the selection (EE lookup for ADMIN/ASSET, the
            # file for SHAPE/POINTS) and publishes a fresh AoiResult later; the
            # load keeps load_aoi's result meanwhile, so a failed re-run leaves
            # the restored AOI in place (v4 does not null value on an error).
            autoselect=True,
            loading=loading,
            methods="ALL",
            map_=map_,
            gee=True,
        )
