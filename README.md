# seq2rel Datasets

[![ci](https://github.com/JohnGiorgi/seq2rel-ds/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JohnGiorgi/seq2rel-ds/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JohnGiorgi/seq2rel-ds/branch/main/graph/badge.svg?token=69PIN7H6UW)](https://codecov.io/gh/JohnGiorgi/seq2rel-ds)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub](https://img.shields.io/github/license/JohnGiorgi/seq2rel?color=blue)

This is a sister repository to [`seq2rel`](https://github.com/JohnGiorgi/seq2rel) which aims to make it easy to generate training data.

## Installation

This repository requires Python 3.8 or later. The preferred way to install is via pip:

```bash
pip install git+https://github.com/JohnGiorgi/seq2rel-ds.git
```

If you need pointers on setting up an appropriate Python environment, please see the [AllenNLP install instructions](https://github.com/allenai/allennlp#installing-via-pip).

### Installing from source

You can also install from source using [Poetry](https://python-poetry.org/):

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

There are two submodules:

1. `preprocess`: used to preprocess existing datasets in a format that can be used with [`seq2rel`](https://github.com/JohnGiorgi/seq2rel).
2. `align`: used to generate data for distantly supervised learning with [`seq2rel`](https://github.com/JohnGiorgi/seq2rel).

### `preprocess`

To see the list of datasets available for preprocessing, call

```
seq2rel-ds preprocess --help
```

To preprocess the data (and in some cases, download it), call one of the commands, e.g.

```
seq2rel-ds preprocess ade "path/to/output/directory"
```

### `align`

To see the list of strategies available for creating training data via weak/distant supervision, call

```
seq2rel-ds align --help
```

#### BioGRID

To use [BioGRID](https://thebiogrid.org/) and distant supervision to create training data, you first need to download a BioGRID release in the `.tab3` format from [here](https://downloads.thebiogrid.org/BioGRID). Then, to preprocess this data, call

```
seq2rel-ds align biogrid preprocess "path/to/output/directory" "path/to/biogrid/release.tab3"
```

with the data preprocessed, call the following to create the training data

```bash
seq2rel-ds align biogrid "path/to/output/directory" "aligned_examples.tsv"
```

`"aligned_examples.tsv"` can then be used to train a [`seq2rel`](https://github.com/JohnGiorgi/seq2rel) model.

