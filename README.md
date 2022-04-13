# seq2rel: Datasets

[![ci](https://github.com/JohnGiorgi/seq2rel-ds/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JohnGiorgi/seq2rel-ds/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JohnGiorgi/seq2rel-ds/branch/main/graph/badge.svg?token=69PIN7H6UW)](https://codecov.io/gh/JohnGiorgi/seq2rel-ds)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub](https://img.shields.io/github/license/JohnGiorgi/seq2rel?color=blue)

This is a companion repository to [`seq2rel`](https://github.com/JohnGiorgi/seq2rel), which makes it easy to preprocess training data.

## Installation

This repository requires Python 3.8 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. If you need pointers on setting up a virtual environment, please see the [AllenNLP install instructions](https://github.com/allenai/allennlp#installing-via-pip).

### Installing the library and dependencies

If you _do not_ plan on modifying the source code, install from `git` using `pip`

```bash
pip install git+https://github.com/JohnGiorgi/seq2rel-ds.git
```

Otherwise, clone the repository and install from source using [Poetry](https://python-poetry.org/):

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/JohnGiorgi/seq2rel-ds
cd seq2rel-ds

# Install the package with poetry
poetry install
```

## Usage

Installing this package gives you access to a simple command-line tool, `seq2rel-ds`. To see the list of available commands, run:

```bash
seq2rel-ds --help
```

> Note, you can also call the underlying python files directly, e.g. `python path/to/seq2rel_ds/main.py --help`.

To preprocess a dataset (and in most cases, download it), call one of the commands, e.g.

```bash
seq2rel-ds cdr main "path/to/cdr"
```

> Note, you have to include `main` because [`typer`](https://typer.tiangolo.com/) does not support default commands.

This will create the preprocessed `tsv` files under the specified output directory, e.g.

```
cdr
 ┣ train.tsv
 ┣ valid.tsv
 ┗ test.tsv
```

which can then be used to train a [`seq2rel`](https://github.com/JohnGiorgi/seq2rel) model.