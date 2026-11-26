"""Download-time adjustments for predefined layers.

Some layers can only settle their export once the AOI is known server-side:
``subj`` needs its unit count to know whether a byte raster holds it. An entry
may declare ``"prepare_download": callable(var, params) -> (i18n_key, kwargs) |
None``. ``process_actions.materialize_raw_layers`` calls
``prepare_predefined_download`` on the download worker right before the
export; the hook may replace ``var.gee_images`` and set ``var.export_nodata``,
and a returned notice is shown to the user as a warning.

Never call this from a Solara handler or render: hooks may block on Earth
Engine.
"""

from gui.scripts.predefined import default_param_values, entry_for_name


def prepare_predefined_download(var, overwrite=False):
    """Run the entry's ``prepare_download`` hook for ``var``.

    Returns the hook's ``(i18n_key, kwargs)`` notice, or None when the layer
    has no hook, has no built image (asset-id GEEVar), or will reuse the file
    already on disk (``GEEVar._download`` skips the export, so the hook's
    Earth Engine cost would buy nothing).
    """
    if getattr(var, "path", None):
        return None  # a user's own asset, even one named like a catalogue layer
    key, params, entry = entry_for_name(getattr(var, "name", "") or "")
    hook = entry.get("prepare_download") if entry else None
    if hook is None or not getattr(var, "gee_images", None):
        return None
    if not overwrite and var.expected_local_path.exists():
        return None
    return hook(var, {**default_param_values(key), **params})
