# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust if needed

project = 'mlchem'
copyright = '2025, Leonardo Contreas'
author = 'Leonardo Contreas'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]
autosummary_generate = True


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
    'private-members': True,
    'show-inheritance': True
}


