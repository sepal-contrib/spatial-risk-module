"""Reactive application state for the Spatial Risk GUI."""

from datetime import datetime
from typing import Optional

import solara


class AppState:
    """Reactive application state for Spatial Risk."""

    def __init__(self):
        """Build the reactives that back the app's project/UI state."""
        # Project (spatialrisk.project.Project | None)
        # Use identity equality: Project is a mutable pydantic model — field-value
        # comparison (pydantic __eq__) would suppress re-renders after in-place
        # mutations. Any new reference must fire, so we compare by identity only.
        self.project = solara.reactive(None, equals=lambda a, b: a is b)

        # Unsaved-changes tracking (best-effort: "changed since last save/load").
        self.project_dirty = solara.reactive(False)
        self.last_saved = solara.reactive(None)  # datetime | None

        # AOI (pysepal AoiResult | None). The picker inputs that produced it
        # ride on ``AoiResult.spec``; see gui/scripts/aoi_io.py.
        self.aoi_result = solara.reactive(None)

        # The AOI picker's selection (pysepal AoiSpec | None), the two-way
        # channel of ``AoiView(spec=)``: the picker publishes each selection
        # here, and a project load writes the restored spec here to put the
        # picker back on it. Writing None does NOT clear the picker (a v4
        # no-op); AoiTile resets it on a project switch instead.
        self.aoi_spec = solara.reactive(None)

        # Nesting depth for the project-swap guard — see restoring_project().
        # A solara.reactive, not a plain attribute: reactive VALUES are stored
        # per-kernel, so each browser session gets its own guard (a plain
        # attribute on this process-global singleton would let one session's
        # in-flight restore suppress every other session's legitimate AOI
        # attach on a shared SEPAL server). It is a counter, not a bool, so
        # nested/overlapping restore windows cannot clear the guard while an
        # outer window is still active. No component reads this reactive
        # during render, so setting it never triggers a re-render — reading
        # ``.value`` here (from an effect body or a callback) does not create
        # a render subscription.
        self._restoring_depth = solara.reactive(0)

        # Bumped each time a project is loaded from disk. The map view subscribes
        # to this so it can zoom to the AOI on load (a plain project-reference
        # watch would also fire on every in-place mutation).
        self.project_loaded_signal = solara.reactive(0)

        # Legends published by the layers currently on the map, newest last.
        # Entries are language-neutral (see gui/scripts/legend_data.py); Page()
        # translates the selected one at render time so legends follow a locale
        # switch. Keyed by map-layer key.
        self.layer_legends = solara.reactive(())
        self.selected_legend = solara.reactive("")

        # Explicit handle tiles receive as an argument (they must not import
        # this singleton — see the tile contract). Built once so its identity
        # is stable: a port rebuilt per render would churn reacton props.
        self.legend_port = self._build_legend_port()

        # Variable processing
        self.processing = solara.reactive(False)

        # Global UI state
        self.loading = solara.reactive(False)

        # Mark the project dirty whenever its reference changes (tiles call
        # project.set(p.model_copy()) on every mutation). Suppressed during load
        # so reloading from disk does not mark the project dirty.
        self._suppress_dirty = False
        self.project.subscribe(self._on_project_changed)

    def _on_project_changed(self, new_value) -> None:
        if self._suppress_dirty:
            return
        self.project_dirty.set(new_value is not None)

    def mark_saved(self, when: datetime) -> None:
        """Record a successful save: clear dirty, stamp last-saved time."""
        self.project_dirty.set(False)
        self.last_saved.set(when)

    def restoring_project(self):
        """Mark a project-restore window: aoi/project reactives are mid-swap.

        ``do_load``, ``new_project_state`` and ``close_project_state`` each
        write ``aoi_result`` and ``project`` as separate reactive sets, and
        Solara can run a render (and its effects) BETWEEN them. The
        attach-on-select effect would then pair one project with the other's
        AOI and persist it into the wrong folder — silently overwriting that
        project's stored AOI and unlinking its geometry sidecar. Every such
        swap must wrap itself in this context manager;
        :meth:`attach_current_aoi` refuses to write inside it.

        Nesting-safe: the guard is a depth counter, not a boolean, so an
        inner window closing (e.g. a helper called from within a larger
        swap) does not prematurely clear the guard for the outer window
        still in flight. Uses an exception-safe ``finally`` so a raise
        anywhere inside always unwinds the counter.
        """
        import contextlib

        @contextlib.contextmanager
        def _window():
            self._restoring_depth.value += 1
            try:
                yield
            finally:
                self._restoring_depth.value -= 1

        return _window()

    @property
    def restoring(self) -> bool:
        """Whether a project-restore window is currently open (any depth)."""
        return self._restoring_depth.value > 0

    def attach_current_aoi(self, data_dir) -> bool:
        """Persist the session AOI into the open project — unless mid-restore.

        The effect body reads both reactives fresh at run time, so a run that
        executes after the restore window closes always sees the settled
        (aoi, project) pair; the guard only rejects runs inside the window,
        where the pair can be cross-project.
        """
        if self.restoring:
            return False
        from gui.scripts.aoi_io import attach_aoi

        return attach_aoi(self.project.value, self.aoi_result.value, data_dir=data_dir)

    def load_project_state(self, project, when: Optional[datetime]) -> None:
        """Install a loaded project without marking it dirty."""
        self._suppress_dirty = True
        try:
            self.project.set(project)
        finally:
            self._suppress_dirty = False
        self.project_dirty.set(False)
        self.last_saved.set(when)
        self.project_loaded_signal.set(self.project_loaded_signal.value + 1)

    def new_project_state(self, project) -> None:
        """Install a freshly created project (dirty, never saved).

        Resets the workflow context so the user starts clean at the AOI step.

        aoi_result is cleared BEFORE the new project is installed, and the
        whole sequence runs inside restoring_project(): a render between
        these sets could otherwise let the attach-on-select effect pair the
        NEW project with the OUTGOING session AOI and persist it into the
        new project's folder (see restoring_project()). Clearing first means
        even an unguarded reader never observes (new project, old AOI); the
        guard is defence in depth on top of that ordering.
        """
        with self.restoring_project():
            self.aoi_result.set(None)
            self.aoi_spec.set(None)
            self.project.set(project)  # subscription marks dirty=True
            self.last_saved.set(None)
            # Bump the same signal a load does so the shell's on-switch effects
            # run (clear the previous project's map overlays + tracking,
            # rebuild the empty Train/Inference job lists). The signal means
            # "a project was installed" — loaded from disk OR newly created.
            self.project_loaded_signal.set(self.project_loaded_signal.value + 1)

    def close_project_state(self) -> None:
        """Return to the no-project state — the open project was deleted.

        The only path back to ``project=None`` after start-up. ``project=None`` is
        already a supported render state (it is what the app boots into), and the
        signal bump re-runs the shell's on-switch effects, so the map overlays,
        job lists and log console are torn down by the existing code.

        Message state is not reset here: load/save/delete outcomes are pysepal
        toasts, which expire on their own.

        aoi_result is cleared before project (and the whole sequence is
        wrapped in restoring_project()) for the same cross-project-attach
        reason as new_project_state — see there.
        """
        with self.restoring_project():
            self.aoi_result.set(None)
            self.aoi_spec.set(None)
            self.project.set(None)  # subscription sets dirty=False
            self.last_saved.set(None)
            self.project_loaded_signal.set(self.project_loaded_signal.value + 1)

    def register_legends(self, *legends) -> None:
        """Publish (or replace) legends and select the newest one."""
        from gui.scripts.legend_registry import upsert

        if not legends:
            return
        self.layer_legends.set(upsert(self.layer_legends.value, *legends))
        self.selected_legend.set(legends[-1].layer_id)

    def unregister_legends(self, *layer_ids: str) -> None:
        """Withdraw legends for layers that left the map."""
        from gui.scripts.legend_registry import next_selection, remove

        if not layer_ids:
            return
        remaining = remove(self.layer_legends.value, *layer_ids)
        self.layer_legends.set(remaining)
        self.selected_legend.set(next_selection(remaining, self.selected_legend.value))

    def clear_legends(self) -> None:
        """Drop every legend (project switch / close)."""
        self.layer_legends.set(())
        self.selected_legend.set("")

    def _build_legend_port(self):
        """Bundle the legend operations a tile is allowed to reach."""
        from gui.scripts.legend_registry import LegendPort

        return LegendPort(
            register=self.register_legends,
            unregister=self.unregister_legends,
            generation=lambda: self.project_loaded_signal.value,
        )

    @property
    def aoi_complete(self) -> bool:
        """Whether an AOI has been captured."""
        return self.aoi_result.value is not None

    @property
    def variables_complete(self) -> bool:
        """Whether the project has raw variables and a base raster."""
        p = self.project.value
        if p is None:
            return False
        has_vars = bool(p.raw_variables)
        has_base = p.base_raster is not None
        return has_vars and has_base


app_state = AppState()
