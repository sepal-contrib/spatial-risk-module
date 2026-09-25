"""Patsy-formula helpers: generation, variable extraction, design info."""

import ast
import re
import warnings
from typing import TYPE_CHECKING

import pandas as pd
from patsy import dmatrices

if TYPE_CHECKING:
    from spatialrisk.dataset import Dataset


def extract_variables(formula: str, mode: str = "predictors") -> set:
    """Extract variable names from a Patsy formula (handles I/scale/C etc.).

    Parameters
    ----------
    formula : str
        A Patsy formula string, e.g. 'I(1 - fcc) + trial ~ scale(altitude) + C(pa)'.
    mode : {'predictors', 'target', 'I', 'all'}
        - 'predictors': extract right-hand side variables (default)
        - 'target': extract left-hand side variables
        - 'I': extract variables only inside I() expressions on the LHS
        - 'all': extract all variables from both sides

    Returns:
    --------
    set
        A set of raw variable names (unique, untransformed).

    Example:
    --------
    >>> formula = "I(1-fcc) + trial ~ scale(altitude) + scale(dist_edge) + C(pa)"
    >>> extract_variables(formula)
    {'altitude', 'dist_edge', 'pa'}
    >>> extract_variables(formula, mode='target')
    {'fcc', 'trial'}
    >>> extract_variables(formula, mode='I')
    {'fcc'}
    >>> extract_variables(formula, mode='all')
    {'altitude', 'dist_edge', 'pa', 'fcc', 'trial'}
    """
    # --- Split formula ---
    parts = formula.split("~", 1)
    lhs = parts[0].strip()
    rhs = parts[1].strip() if len(parts) > 1 else ""

    # --- Determine which text to parse based on mode ---
    if mode == "I":
        target_expr = lhs
    elif mode == "target":
        target_expr = lhs
    elif mode == "predictors":
        target_expr = rhs
    elif mode == "all":
        target_expr = formula
    else:
        raise ValueError("mode must be one of: 'predictors', 'target', 'I', 'all'")

    raw_vars = set()

    # --- Match function-like patterns: scale(...), C(...), I(...), etc. ---
    func_pattern = r"[a-zA-Z_][a-zA-Z0-9_]*\(([^)]*)\)"
    matches = re.findall(func_pattern, target_expr)

    for expr in matches:
        tokens = re.split(r"[+\-*/\(\)\s]", expr)
        tokens = [t.strip() for t in tokens if t.strip()]
        for token in tokens:
            if re.match(r"^\d+(\.\d+)?$", token):  # skip numbers
                continue
            if token.lower() in {"i", "scale", "c", "poly", "bs", "cr"}:
                continue
            raw_vars.add(token)

    # --- Extract standalone variables ---
    standalone = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", target_expr)
    for var in standalone:
        if var.lower() in {"i", "scale", "c", "poly", "bs", "cr"}:
            continue
        raw_vars.add(var)

    # --- Special handling for mode="I" ---
    if mode == "I":
        I_expressions = re.findall(r"I\((.*?)\)", lhs)
        I_vars = set()
        for expr in I_expressions:
            I_vars.update(re.findall(r"[A-Za-z_]\w*", expr))
        raw_vars = {v for v in raw_vars if v in I_vars}

    # --- Validate identifiers ---
    raw_vars = {v for v in raw_vars if re.match(r"^[A-Za-z_]\w*$", v)}

    return raw_vars


# Transform callables that may appear in formula factor code; they are
# function names, not data variables.
_FORMULA_TRANSFORMS = {
    "I",
    "C",
    "scale",
    "center",
    "standardize",
    "Q",
    "np",
    "log",
    "poly",
    "bs",
    "cr",
}


def _factor_names(code: str) -> set:
    """Variable names referenced by one patsy factor's Python code."""
    tree = ast.parse(code, mode="eval")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return names - _FORMULA_TRANSFORMS


def formula_variables(formula: str) -> tuple:
    """(lhs_vars, rhs_vars) for a patsy formula, via patsy + ast.

    Unlike :func:`extract_variables`, keyword-argument names such as the
    ``levels=`` in ``C(x, levels=[...])`` are ``ast.keyword`` nodes — not
    ``ast.Name`` — so they are never mistaken for variables. Raises
    ``patsy.PatsyError`` or ``SyntaxError`` on unparsable input.
    """
    from patsy import ModelDesc

    desc = ModelDesc.from_formula(formula)
    lhs, rhs = set(), set()
    for termlist, out in ((desc.lhs_termlist, lhs), (desc.rhs_termlist, rhs)):
        for term in termlist:
            for factor in term.factors:
                out |= _factor_names(factor.code)
    return lhs, rhs


def get_design_info(patsy_formula, samples_file):
    """Get design info from patsy."""
    dataset = pd.read_csv(samples_file)
    dataset = dataset.dropna(axis=0)
    dataset["trial"] = 1
    y, x = dmatrices(patsy_formula, dataset, 0, "drop")
    y_design_info = y.design_info
    x_design_info = x.design_info
    return (y_design_info, x_design_info)


def get_categorical_levels(var) -> "list | None":
    """Return the full set of unique values in a categorical raster.

    Reads the raster band block-by-block (memory-safe for large rasters such
    as a sub-jurisdiction map) and accumulates the distinct pixel values,
    dropping nodata and NaN. Integral values are returned as Python ``int``.

    These levels are intended to be injected into a Patsy ``C(var, levels=...)``
    term so that the design matrix declares its complete categorical domain up
    front — preventing an unseen-level error when prediction encounters a
    value that never appeared in the training sample: GLM, iCAR and RF
    prediction raise ``ValueError`` (message ``"<factor code>: values not
    among the training levels: [...]"``) from the compiled linear predictor
    or the RF design builder. A categorical patsy still builds itself -- an
    expression such as ``C(x + 0)``, or string levels -- raises ``PatsyError``.

    Parameters
    ----------
    var : LocalRasterVar
        Categorical variable exposing a ``.path`` attribute.

    Returns:
    --------
    list or None
        Sorted list of unique levels, or ``None`` if the raster cannot be read
        (so the caller can fall back to a bare ``C(var)`` term).
    """
    import numpy as np
    import rasterio

    try:
        values: set = set()
        with rasterio.open(var.path) as src:
            nodata = src.nodata
            for _, window in src.block_windows(1):
                block = src.read(1, window=window)
                block = (
                    block[~np.isnan(block)]
                    if np.issubdtype(block.dtype, np.floating)
                    else block.ravel()
                )
                uniques = np.unique(block)
                if nodata is not None:
                    uniques = uniques[uniques != nodata]
                values.update(uniques.tolist())

        def _coerce(v):
            return int(v) if float(v).is_integer() else v

        return sorted(_coerce(v) for v in values)
    except Exception as exc:  # fall back to data-discovered levels
        warnings.warn(
            f"Could not read categorical levels for '{getattr(var, 'name', var)}' "
            f"from {getattr(var, 'path', '?')}: {exc}. "
            "Falling back to levels discovered from the training sample.",
            UserWarning,
            stacklevel=2,
        )
        return None


def generate_patsy_formula(dataset: "Dataset", include_levels: bool = True) -> str:
    """Generate a Patsy formula from a Dataset's target and features.

    Continuous variables are wrapped in ``scale(...)`` and categorical ones in
    ``C(...)``, classified by each variable's ``raster_type`` attribute.

    Parameters
    ----------
    dataset : Dataset
        Dataset instance with configured target and features
    include_levels : bool
        When False, categorical terms are emitted as bare ``C(x)`` — no
        raster is read. Used for the display/edit formula in the GUI;
        :func:`inject_categorical_levels` re-arms the levels at fit time.

    Returns:
    --------
    str
        Patsy formula string

    Example:
    --------
    >>> dataset.set_target('fcc', year=2020)
    >>> dataset.set_features(['altitude', 'pa', 'dist_edge'])
    >>> generate_patsy_formula(dataset)
    "I(fcc) + trial ~ scale(altitude) + scale(dist_edge) + C(pa, levels=[0, 1])"

    Notes:
    ------
    Categorical terms declare their full level domain via ``levels=...``, read
    from each categorical raster with :func:`get_categorical_levels`. This
    prevents an unseen-level error at prediction time when a pixel carries a
    value that never appeared in the training sample: GLM, iCAR and RF raise
    ``ValueError`` (message ``"<factor code>: values not among the training
    levels: [...]"``) from the compiled linear predictor or the RF design
    builder. A categorical patsy still builds itself -- an expression such as
    ``C(x + 0)``, or string levels -- raises ``PatsyError``.
    """
    # Validate dataset configuration
    if not dataset.target:
        raise ValueError("Dataset target not set. Use dataset.set_target() first.")
    if not dataset.features:
        raise ValueError("Dataset features not set. Use dataset.set_features() first.")

    dependent_variable = dataset.target.name

    # Print dataset configuration
    print("\n📊 Generating Patsy formula:")
    print(f"  Target: {dependent_variable}")
    print(f"  Features: {', '.join([f.name for f in dataset.features])}")

    continuous = []
    categorical = []  # holds the LocalRasterVar so its raster path is reachable

    for var in dataset.features:
        # Check if variable has raster_type attribute (LocalRasterVar)
        if hasattr(var, "raster_type") and var.raster_type:
            if var.raster_type == "continuous":
                continuous.append(var.name)
            elif var.raster_type == "categorical":
                categorical.append(var)
            else:
                # Default to continuous if raster_type is not set
                continuous.append(var.name)
        else:
            # Default to continuous if no raster_type attribute
            continuous.append(var.name)

    parts = []
    if continuous:
        parts += [f"scale({x})" for x in continuous]
    for var in categorical:
        # Declare the full categorical domain so prediction never hits an
        # "unexpected level". Fall back to a bare C() if the raster is unreadable.
        levels = get_categorical_levels(var) if include_levels else None
        if levels is not None:
            parts.append(f"C({var.name}, levels={levels})")
        else:
            parts.append(f"C({var.name})")

    # Print classification results
    if continuous:
        print(f"  Continuous: {', '.join(continuous)}")
    if categorical:
        print(f"  Categorical: {', '.join(v.name for v in categorical)}")

    rhs = " + ".join(parts) if parts else "1"  # intercept-only model if empty
    formula = f"I({dependent_variable}) + trial ~ {rhs}"

    print(f"\n✓ Formula: {formula}\n")

    return formula


def inject_categorical_levels(formula: str, dataset: "Dataset") -> str:
    """Arm bare ``C(x)`` terms with the raster's full level domain.

    The displayed/edited formula keeps categorical terms short (``C(pa)``),
    but prediction re-parses the *stored* formula string against the training
    CSV, so the level domain must be explicit there — otherwise a pixel value
    absent from the sample raises a patsy "unexpected level" error. Applied to
    the RHS only; terms that already carry ``levels=`` (or anything beyond the
    bare name) and unreadable rasters are left untouched.
    """
    parts = formula.split("~", 1)
    if len(parts) != 2:
        return formula
    lhs, rhs = parts

    categorical = {
        var.name: var
        for var in dataset.features or []
        if getattr(var, "raster_type", None) == "categorical"
    }
    for name, var in categorical.items():
        pattern = rf"\bC\(\s*{re.escape(name)}\s*\)"
        if not re.search(pattern, rhs):
            continue
        levels = get_categorical_levels(var)
        if levels is None:
            continue
        rhs = re.sub(pattern, f"C({name}, levels={levels})", rhs)

    return f"{lhs}~{rhs}"


def strip_categorical_levels(formula: str) -> str:
    """Display form of a stored formula: ``C(x, levels=[...])`` -> ``C(x)``."""
    return re.sub(
        r"\bC\(\s*([A-Za-z_]\w*)\s*,\s*levels=\[[^\]]*\]\s*\)",
        r"C(\1)",
        formula,
    )
