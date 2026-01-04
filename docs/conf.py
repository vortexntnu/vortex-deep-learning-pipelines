# pylint: skip-file
import os
import sys

# Add project directory to sys.path for autodoc to find your modules
sys.path.insert(0, os.path.abspath("../YOLO-detect_buoys"))

# Project information
project = "vortex image build"
copyright = "2024, Yauhen Yavorski"
author = "Yauhen Yavorski"

# General configuration
extensions = [
    "sphinx.ext.autodoc",  # Automatically document docstrings
    "sphinx.ext.napoleon",  # Support for Google-style and NumPy-style docstrings
    "sphinx.ext.viewcode",  # Link to source code in the docs
    "sphinx.ext.autosummary",  # Generate summaries of functions/classes
    "myst_parser",  # Support for Markdown files (optional)
]

# Automatically generate autosummary pages
autosummary_generate = True

# Napoleon settings (optional)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Templates and exclude patterns
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for HTML output (theme and static paths)
html_theme = "sphinx_rtd_theme"  # Change to ReadTheDocs theme
html_static_path = ["_static"]
