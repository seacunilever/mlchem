# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import ast
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../../..'))  # parent of the package dir

project = 'mlchem'
copyright = '2025, Leonardo Contreas'
author = 'Leonardo Contreas'

# The package, wheel metadata, and docs all read the same version source.
_version_path = Path(__file__).resolve().parents[2] / '_version.py'
_version_tree = ast.parse(_version_path.read_text(encoding='utf-8'))
_version_assignment = next(
    (
        node for node in _version_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == '__version__'
            for target in node.targets
        )
    ),
    None,
)
if _version_assignment is None:
    raise RuntimeError(f"Could not find __version__ in {_version_path}.")
release = ast.literal_eval(_version_assignment.value)
if not isinstance(release, str):
    raise RuntimeError(f"Expected __version__ in {_version_path} to be a string.")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
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
myst_heading_anchors = 3

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
