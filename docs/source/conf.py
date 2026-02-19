# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------

project = 'GR00T-WholeBodyControl'
copyright = '2026, NVIDIA'
author = 'NVIDIA GEAR Team'
release = '1.0.0'
version = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'autodocsumm',
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinxemoji.sphinxemoji',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.video',
]

# mathjax hacks
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

# emoji style
sphinxemoji_style = "twemoji"

# supported file extensions for source files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# BibTeX configuration
bibtex_bibfiles = []

# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = False

# generate links to the documentation of objects in external projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store']

# List of zero or more Sphinx-specific warning categories to be squelched
suppress_warnings = [
    "ref.python",
]

# -- MyST Parser configuration -----------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

# -- Options for HTML output -------------------------------------------------

import sphinx_book_theme

html_title = "GR00T-WholeBodyControl Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_favicon = "_static/favicon.ico"
html_show_copyright = True
html_show_sphinx = False  # This removes "Built with Sphinx" footer
html_last_updated_fmt = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "path_to_docs": "docs/",
    "collapse_navigation": True,
    "repository_url": "https://github.com/NVlabs/GR00T-WholeBodyControl",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "use_sidenotes": True,
    "logo": {
        "text": "GR00T-WholeBodyControl Documentation",
        "image_light": "_static/NVIDIA-logo-white.png",
        "image_dark": "_static/NVIDIA-logo-black.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVlabs/GR00T-WholeBodyControl",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "GEAR-SONIC Website",
            "url": "https://nvlabs.github.io/GEAR-SONIC/",
            "icon": "fa-solid fa-globe",
            "type": "fontawesome",
        },
        {
            "name": "Paper",
            "url": "https://arxiv.org/abs/2511.07820",
            "icon": "fa-solid fa-file-pdf",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",
}

templates_path = [
    "_templates",
]

# -- Internationalization ----------------------------------------------------

language = "en"
