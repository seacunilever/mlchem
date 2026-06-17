# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))  # parent of the package dir

project = 'mlchem'
copyright = '2025, Leonardo Contreas'
author = 'Leonardo Contreas'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
]

# Allow Sphinx to consume both reStructuredText and Markdown sources so the
# README can be reused as the welcome page (single source of truth).
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Enable the MyST `{include}` directive used by docs/source/welcome.md.
myst_enable_extensions = [
    'colon_fence',
]

templates_path = ['_templates']
exclude_patterns = []
add_module_names = False
autodoc_class_signature = 'separated'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Render the numpydoc "Methods" section as a styled description list instead
# of letting Napoleon emit `.. method::` directives that collide with autodoc.
napoleon_custom_sections = [('Methods', 'params_style')]

toc_object_entries_show_parents = 'hide'
