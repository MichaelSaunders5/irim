# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

project = 'Irim'
project_copyright = '2023, Michael Saunders'
author = 'Michael Saunders'
release = '0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.imgmath',
    'autoapi.extension',
    "sphinx.ext.intersphinx",
    'sphinx.ext.viewcode'
]

autoapi_type = 'python'
autoapi_dirs = ['../../src']
autoapi_options = ['members', 'show-inheritance', 'show-module-summary']
autoapi_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []

imgmath_font_size = 10

intersphinx_mapping = {'numpy': ('https://numpy.org/doc/stable/', None)}
# 'python': ('https://docs.python.org/3/', None),
# 'scipy': ('https://docs.scipy.org/doc/scipy/', None),
# 'matplotlib': ('https://matplotlib.org/stable/', None)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_use_smartquotes = True
