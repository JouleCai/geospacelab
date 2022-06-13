# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import warnings
import geospacelab
# sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = 'GeospaceLAB'
copyright = '2021, Lei Cai'
author = 'Lei Cai'

# The full version, including alpha/beta/rc tags
release = 'beta-v' + geospacelab.__version__

# Unless we catch the warning explicitly somewhere, a warning should cause the
# docs build to fail. This is especially useful for getting rid of deprecated
# usage in the gallery.
warnings.filterwarnings('error', append=True)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'numpydoc',  # Needs to be loaded *after* autodoc.
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.mathmpl',
    'matplotlib.sphinxext.plot_directive',
    'sphinxcontrib.inkscapeconverter',
    'sphinxext.custom_roles',
    'sphinxext.github',
    'sphinxext.math_symbol_table',
    'sphinxext.missing_references',
    'sphinxext.mock_gui_toolkits',
    'sphinxext.skip_deprecated',
    'sphinxext.redirect_from',
    'sphinx_copybutton',
    'sphinx_panels',
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Settings for autodoc
# Mock importing packages
autodoc_mock_imports = ['cartopy', 'apexpy']
autodoc_member_order = 'bysource'
add_module_names = False

# Settings for napoleon
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Settings for copybutton
copybutton_prompt_text = ">>> "

# Settings for intersphinx
intersphinx_mapping = {
    'python': ('http://docs.python.org/2', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('http://matplotlib.sourceforge.net/', None)
}