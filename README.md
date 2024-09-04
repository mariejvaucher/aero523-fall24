# AEROSP 523 Computational Fluid Dynamics: Code Examples

[![Site](https://github.com/A-CGray/AE523-Fall23/actions/workflows/deploy.yml/badge.svg)](https://A-CGray.github.io/AE523-Fall23/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-CGray/AE523-Fall23/HEAD)

Code examples for the Fall 2023 term of the University of Michigan's AEROSP 523 CFD course

Find the rendered site [here](https://mariejvaucher.github.io/aero523-fall24/Content/PythonSetup.html)

## How to edit to the site

### Install prerequisites

All packages required to build and run the website pages are listed in `requirements.txt`.
It's probably a good idea to create a new python environment for working on the site.
Once you've created and activated your environment, run:

```bash
pip install -r requirements.txt
```

### Create new chapters and pages

First create either a markdown file or jupyter notebook in whichever directory feels most appropriate (Examples, Content, some new directory etc).

If the page is part of a new chapter, also create a markdown file for the chapter in the same directory, e.g in `NewChapter.md`

``````markdown
# Chapter Title

```{tableofcontents}
```
``````

Add the new page to the table of contents file, `_toc.yml`, for example, to add the new chapter and new file within the chapter, you can add:

```yml
chapters:
.
.
.
- file: Path/To/NewChapter
  sections:
  - file: Path/To/NewPage.ipynb
```

### Build the site

To build the site locally, run:

```bash
make
```

Which should rebuild only the new content you added.
To rebuild the whole site from scratch, run:

```bash
make clean&&make
```

To check the built website, open up `_build/html/index.html` in your web browser.

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).
