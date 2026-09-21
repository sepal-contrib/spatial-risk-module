"""Shared "show on map" toggle for derived (processed) variables.

Processing and post-processing both write into ``project.processed_variables``,
and both tiles list a slice of that registry. The on-map state is therefore kept
here — module-level, like ``vars_on_map`` — so a layer toggled on in Process
still reads as "on" in Post-process (they can show the same variable) and both
tiles share one layer-key namespace.

Rendering reuses the source-variable helpers (``add_raster_var_on_map`` /
``add_vector_on_map``): processed rasters keep the variable's ``name``, so a
downloaded catalogue layer keeps its palette after alignment. The *displayed*
name is prefixed with an origin marker (``[H]`` / ``[D]``, see
``gui.scripts.layer_labels``) because a raw variable and its harmonized
counterpart share a registry key and would otherwise be indistinguishable in
the layer control.
"""

import logging
import threading

import solara

from gui.i18n import t
from gui.scripts.inflight import InflightKeys
from gui.scripts.layer_labels import processed_layer_label
from gui.scripts.map_helpers import add_vector_on_map, is_mappable
from gui.scripts.notify_bridge import ERROR_TOAST_TIMEOUT
from gui.scripts.solara_threads import spawn_in_context
from gui.scripts.variable_map import add_raster_var_on_map

logger = logging.getLogger("spatial_risk")

# Keys of processed variables currently displayed on the map (drives the toggle
# state in every DerivedVariableList).
derived_on_map = solara.reactive(set())

# Guards the two read-modify-write updates to ``derived_on_map`` — the
# worker's add (below) and ``drop_derived_from_map``'s discard — which now
# run concurrently on independent worker threads (one per toggle) and, for
# the discard side, also from the kernel thread via ``process_actions``'
# delete path. A plain read of ``.value`` or a wholesale
# ``derived_on_map.set(set())`` reset needs no lock.
derived_on_map_lock = threading.Lock()

# Processed-variable keys whose map toggle is running (see InflightKeys).
derived_toggle_inflight = InflightKeys(key="derived_toggle_inflight")


def derived_layer_key(key: str) -> str:
    """Unique map-layer key for a processed variable."""
    return f"derived_{key}"


def _derived_legend(key: str, var):
    """The legend a processed-variable raster publishes while it is on the map.

    Processed rasters are always local files, so the style resolver is the only
    source needed — post-process outputs (edge/dist/loss/gain) get their
    QGIS ramp and class labels from it.
    """
    from gui.scripts.legend_data import Label, variable_spec_from_style
    from gui.scripts.legend_registry import LayerLegend
    from gui.scripts.variable_styles import resolve_variable_style

    label = Label(literal=getattr(var, "name", "") or key)
    return LayerLegend(
        layer_id=derived_layer_key(key),
        label=label,
        spec=variable_spec_from_style(resolve_variable_style(var), var, label),
    )


def drop_derived_from_map(key: str, map_, legend_port=None) -> None:
    """Remove a processed variable's layer, legend, and on-map state.

    The single removal chokepoint for processed variables: the toggle's
    off-branch and ``process_actions``' delete path both route through it.
    ``legend_port`` may be None — that is a no-op, not a crash.
    """
    if map_ is not None:
        map_.remove_layer(derived_layer_key(key), none_ok=True)
    if legend_port is not None:
        legend_port.unregister(derived_layer_key(key))
    with derived_on_map_lock:
        if key in derived_on_map.value:
            remaining = set(derived_on_map.value)
            remaining.discard(key)
            derived_on_map.set(remaining)


def use_derived_map_toggle(project, map_, notifier, legend_port=None):
    """Hook: an ``on_toggle_map(key)`` callback for processed variables.

    Returns None when there is no map (the caller then renders no toggle). No
    reacton hooks are used here, so the caller may call it conditionally.

    Every layer-add is offloaded to a worker thread — the ``TileClient`` /
    geopandas reads block, exactly like the source-variable toggle. ``notifier``
    is passed in rather than resolved with ``use_notifications()`` so the hook
    stays usable from tests with no NotificationProvider mounted. ``legend_port``
    is a ``LegendPort`` (see ``gui/scripts/legend_registry.py``); None disables
    legend publication.
    """

    def _toggle_on_map(key, p):
        """Worker: add or remove one processed variable's layer, then record it.

        Runs on its own ``spawn_in_context`` thread — the blocking adds and the
        on-map/legend bookkeeping together, so toggling another layer while
        this one loads cannot skip the bookkeeping (a shared ``use_task`` used
        to be cancelled at its ``await`` by the second toggle).
        """
        try:
            var = p.processed_variables.get(key) if p is not None else None
            if var is None or not is_mappable(var):
                return
            if key in derived_on_map.value:
                drop_derived_from_map(key, map_, legend_port)
                return

            layer_key = derived_layer_key(key)
            label = processed_layer_label(p, key)
            generation = legend_port.generation() if legend_port is not None else None
            legend = None
            if type(var).__name__ == "LocalVectorVar":
                add_vector_on_map(map_, str(var.path), label, layer_key)
            else:  # LocalRasterVar — same palette resolution as source rasters
                add_raster_var_on_map(
                    map_,
                    str(var.path),
                    var=var,
                    layer_name=label,
                    key=layer_key,
                    fit_bounds=False,
                )
                legend = _derived_legend(key, var)

            # A project switch during the add means this layer is stale — take
            # it back off rather than publish a legend for it.
            if legend_port is not None and legend_port.generation() != generation:
                map_.remove_layer(layer_key, none_ok=True)
                return

            with derived_on_map_lock:
                derived_on_map.set(set(derived_on_map.value) | {key})
            if legend is not None and legend_port is not None:
                legend_port.register(legend)
        except Exception as exc:
            logger.exception("map toggle failed for processed var %s", key)
            notifier.error(
                t("tiles.variables.error_toggle_map", key=key, exc=exc),
                timeout=ERROR_TOAST_TIMEOUT,
            )
        finally:
            derived_toggle_inflight.release(key)

    def on_toggle_map(key: str):
        cur = project.value
        if cur is None or not derived_toggle_inflight.claim(key):
            return
        try:
            spawn_in_context(_toggle_on_map, (key, cur))
        except Exception as exc:
            # The worker's finally is what releases the claim, so a thread
            # that never starts would hold this key for the rest of the
            # session.
            derived_toggle_inflight.release(key)
            logger.exception("could not start the map-toggle worker")
            notifier.error(
                t("tiles.variables.error_toggle_map", key=key, exc=exc),
                timeout=ERROR_TOAST_TIMEOUT,
            )

    return on_toggle_map if map_ is not None else None
