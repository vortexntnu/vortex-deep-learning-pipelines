import os
import sys

sys.path.insert(0, os.path.abspath("../YOLO-detect_buoys"))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "vortex image build"
copyright = "2024, Yauhen Yavorski"
author = "Yauhen Yavorski"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

extensions = [
    "sphinx.ext.autodoc",  # To automatically document docstrings
    "sphinx.ext.napoleon",  # To support Google-style and NumPy-style docstrings
    "sphinx.ext.viewcode",  # To link to the source code
    "sphinx.ext.autosummary",  # To create summaries of your code
]
