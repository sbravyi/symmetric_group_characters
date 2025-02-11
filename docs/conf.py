# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ChaMPS"
copyright = "2025, Sergey Bravyi, David Gosset, Vojtech Havlicek, Louis Schatzki"
author = "Sergey Bravyi, David Gosset, Vojtech Havlicek, Louis Schatzki"
release = "0.42"

import sys
import os

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

html_context = {
    'css_files': [
        '_static/pygments.css',
        '_static/basic.css',
        '_static/alabaster.css'
    ]
}

html_static_path_version = False

