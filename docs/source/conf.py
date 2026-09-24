"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup ----------------------------------------------------------------

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))

package_path = os.path.abspath("../..")
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))

DOC_DIR = Path(__file__).parent


# -- Project information -------------------------------------------------------

project = "Deforisk Analysis Framework"
copyright = f"2020-{datetime.now().year}, Deforisk Team"
author = "Deforisk Team"
release = "2.0"
version = "2.0"


# -- General configuration -----------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add mappings for intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
}

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
    "tasklist",
]

# nbsphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Continue build even if notebooks have errors
nbsphinx_timeout = 600  # Timeout for notebook execution (in seconds)


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = "pydata_sphinx_theme"
html_last_updated_fmt = "%b %d, %Y"

# Theme options for PyData theme
html_theme_options = {
    "logo": {
        "text": "Risk Analysis Framework",
    },
    "use_edit_page_button": True,
    "show_prev_next": True,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sepal-contrib/spatial-risk-module",
            "icon": "fa-brands fa-github",
        },
    ],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "show_version_warning_banner": True,
    "navigation_with_keys": True,
}

html_context = {
    "github_user": "sepal-contrib",
    "github_repo": "spatial-risk-module",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css"
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ["custom.css"]

# Output file base name for HTML help builder
htmlhelp_basename = "DeforiskAnalysisFrameworkdoc"


# -- Napoleon settings ---------------------------------------------------------

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# -- Options for autosummary/autodoc output ------------------------------------

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Suppress certain warnings
suppress_warnings = ["autodoc"]

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = []


# -- Options for TODO ----------------------------------------------------------

todo_include_todos = True
