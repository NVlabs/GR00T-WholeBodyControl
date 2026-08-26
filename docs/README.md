# Documentation

This directory contains the source code for the GR00T-WholeBodyControl documentation website.

## Building Locally

### Prerequisites

Install the required Python packages:

```bash
pip install sphinx sphinx-book-theme sphinx-design sphinxemoji \
            autodocsumm sphinxcontrib-bibtex myst-parser \
            sphinx-copybutton
```

### Build the Documentation

```bash
cd docs
make html
```

The built documentation will be in `build/html/`. Open `build/html/index.html` in your browser.

### Live Preview

Start a local web server to preview:

```bash
cd build/html
python -m http.server 8000
```

Then open http://localhost:8000

### Clean Build

To remove all built files and rebuild from scratch:

```bash
make clean
make html
```

## Deployment

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch via the GitHub Actions workflow at `.github/workflows/docs.yml`.

The live documentation will be available at:
**https://nvlabs.github.io/GR00T-WholeBodyControl/**

## Documentation Structure

- `source/` - All documentation source files
  - `conf.py` - Sphinx configuration
  - `index.rst` - Main landing page
  - `_static/` - Static assets (CSS, images, logos)
  - `tutorials/` - Tutorial pages
  - `getting_started/` - Getting started guides
  - `user_guide/` - User guide
  - `api/` - API reference
  - `resources/` - Additional resources

## Writing Documentation

- Use Markdown (`.md`) or reStructuredText (`.rst`) files
- Markdown is recommended for simplicity
- Place new files in the appropriate subdirectory
- Update `index.rst` to add new sections to the navigation

## Theme

The documentation uses the `sphinx_book_theme` with NVIDIA branding, matching the Isaac Lab documentation style.
