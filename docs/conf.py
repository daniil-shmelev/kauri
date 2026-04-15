# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))   # for kauri_sphinx extension

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'kauri'
copyright = '2025\u20132026, Daniil Shmelev'
author = 'Daniil Shmelev'
release = '2.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    "sphinxcontrib.bibtex",
    'kauri_sphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
bibtex_bibfiles = ["refs.bib"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = "_static/logo_light.png"

add_module_names = False

mathjax3_config = {
    'tex': {
        'macros': {
            'shuffle': r'\unicode{x29E2}',
        }
    }
}

html_title = "Kauri"
html_show_sourcelink = False
html_show_sphinx = False
