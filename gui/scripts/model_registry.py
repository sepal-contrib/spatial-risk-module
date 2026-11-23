"""Model registry: every trainable model family and its form spec.

Data only (classes + param specs with i18n *keys*); label resolution happens
in the widgets. Solara-free so tiles, dialogs and tests can all import it.
"""

from spatialrisk.mlmodels import (
    GLMModel,
    ICARModel,
    JNRBenchmarkModel,
    MWModel,
    RFModel,
)

MODEL_REGISTRY = {
    "jnr": {
        "label_key": "models.jnr.label",
        "class": JNRBenchmarkModel,
        "description_key": "models.jnr.description",
        "params": [
            {
                "key": "blk_rows",
                "label_key": "models.jnr.params.blk_rows.label",
                "type": "int",
                "default": 128,
            },
            {
                "key": "defor_threshold",
                "label_key": "models.jnr.params.defor_threshold.label",
                "type": "float",
                "default": 99.5,
            },
            {
                "key": "defor_var",
                "label_key": "models.jnr.params.defor_var.label",
                "type": "select",
                "default": "",
                "group": "variables",
            },
            {
                "key": "forest_edge_var",
                "label_key": "models.jnr.params.forest_edge_var.label",
                "type": "select",
                "default": "",
                "group": "variables",
            },
            {
                "key": "forest_var",
                "label_key": "models.jnr.params.forest_var.label",
                "type": "select",
                "default": "",
                "group": "variables",
            },
            {
                "key": "subj_var",
                "label_key": "models.jnr.params.subj_var.label",
                "type": "select",
                "default": "",
                "group": "variables",
            },
        ],
        "has_sampling": False,
    },
    "mw": {
        "label_key": "models.mw.label",
        "class": MWModel,
        "description_key": "models.mw.description",
        "params": [
            {
                "key": "win_size_list",
                "label_key": "models.mw.params.win_size_list.label",
                "type": "text",
                "default": "5, 11, 21",
            },
            {
                "key": "blk_rows",
                "label_key": "models.mw.params.blk_rows.label",
                "type": "int",
                "default": 256,
            },
            {
                "key": "defor_threshold",
                "label_key": "models.mw.params.defor_threshold.label",
                "type": "float",
                "default": 99.5,
            },
            {
                "key": "defor_var",
                "label_key": "models.mw.params.defor_var.label",
                "type": "select",
                "default": "",
                "group": "variables",
            },
            {
                "key": "forest_edge_var",
                "label_key": "models.mw.params.forest_edge_var.label",
                "type": "select",
                "default": "",
                "group": "variables",
            },
            {
                "key": "forest_var",
                "label_key": "models.mw.params.forest_var.label",
                "type": "select",
                "default": "",
                "group": "variables",
            },
        ],
        "has_sampling": False,
    },
    "glm": {
        "label_key": "models.glm.label",
        "class": GLMModel,
        "description_key": "models.glm.description",
        "params": [
            {
                "key": "solver",
                "label_key": "models.glm.params.solver.label",
                "type": "select",
                "default": "lbfgs",
                "items": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            },
            {
                "key": "max_iter",
                "label_key": "models.glm.params.max_iter.label",
                "type": "int",
                "default": 1000,
            },
            {
                "key": "random_seed",
                "label_key": "models.glm.params.random_seed.label",
                "type": "int",
                "default": 1234,
            },
        ],
        "has_sampling": True,
        "has_formula": True,
    },
    "rf": {
        "label_key": "models.rf.label",
        "class": RFModel,
        "description_key": "models.rf.description",
        "params": [
            {
                "key": "n_trees",
                "label_key": "models.rf.params.n_trees.label",
                "type": "int",
                "default": 100,
            },
            {
                "key": "max_depth",
                "label_key": "models.rf.params.max_depth.label",
                "type": "int",
                "default": 15,
            },
            {
                "key": "min_samples_leaf",
                "label_key": "models.rf.params.min_samples_leaf.label",
                "type": "int",
                "default": 2,
            },
            {
                "key": "random_seed",
                "label_key": "models.rf.params.random_seed.label",
                "type": "int",
                "default": 1234,
            },
        ],
        "has_sampling": True,
        "has_formula": True,
    },
    "icar": {
        "label_key": "models.icar.label",
        "class": ICARModel,
        "description_key": "models.icar.description",
        "params": [
            {
                "key": "csize",
                "label_key": "models.icar.params.csize.label",
                "type": "float",
                "default": 10.0,
            },
            {
                "key": "mcmc",
                "label_key": "models.icar.params.mcmc.label",
                "type": "int",
                "default": 4000,
            },
            {
                "key": "burnin",
                "label_key": "models.icar.params.burnin.label",
                "type": "int",
                "default": 4000,
            },
            {
                "key": "thin",
                "label_key": "models.icar.params.thin.label",
                "type": "int",
                "default": 1,
            },
            {
                "key": "prior_vrho",
                "label_key": "models.icar.params.prior_vrho.label",
                "type": "float",
                "default": -1.0,
            },
            {
                "key": "beta_start",
                "label_key": "models.icar.params.beta_start.label",
                "type": "float",
                "default": -99.0,
            },
            {
                "key": "random_seed",
                "label_key": "models.icar.params.random_seed.label",
                "type": "int",
                "default": 1234,
            },
            {
                "key": "csize_interpolate",
                "label_key": "models.icar.params.csize_interpolate.label",
                "type": "float",
                "default": 0.1,
            },
        ],
        "has_sampling": True,
        "has_formula": True,
    },
}

MODEL_KEYS = list(MODEL_REGISTRY.keys())


#: Families whose fit/apply derive the period length from the target's name
#: (``forest_loss_2015_2020`` -> 5 years).
INTERVAL_FAMILIES = ("jnr", "mw")


def interval_target_error(model_key, target_name):
    """``(i18n key, kwargs)`` when an MW/JNR model cannot use this target.

    Their fit and apply need the period length, read from two 4-digit years
    in the target's name. A single-period target (``burned_area_sep``,
    ``active_fire``) has none: refuse it in the dialog instead of failing
    deep in the fit. Other families take any target (None).
    """
    if str(model_key or "").split("_")[0] not in INTERVAL_FAMILIES:
        return None
    # Lazy: spatialrisk.evaluation pulls in matplotlib.
    from spatialrisk.evaluation import interval_from_target

    if interval_from_target(target_name) is not None:
        return None
    return ("tiles.train.error_needs_interval_target", {"name": target_name or ""})
