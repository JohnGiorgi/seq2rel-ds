
seq2rel Datasets
================

This is a sister repository to `\ ``seq2rel`` <https://github.com/JohnGiorgi/seq2rel>`_ which aims to make it easy to generate training data.

Installation
------------

This repository requires Python 3.8 or later. The preferred way to install is via pip:

.. code-block::

   pip install seq2rel-ds

If you need pointers on setting up an appropriate Python environment, please see the `AllenNLP install instructions <https://github.com/allenai/allennlp#installing-via-pip>`_.

Installing from source
^^^^^^^^^^^^^^^^^^^^^^

You can also install from source. 

Using ``pip``\ :

.. code-block::

   pip install git+https://github.com/JohnGiorgi/seq2rel-ds

Using `Poetry <https://python-poetry.org/>`_\ :

.. code-block:: bash

   # Install poetry for your system: https://python-poetry.org/docs/#installation
   curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

   # Clone and move into the repo
   git clone https://github.com/JohnGiorgi/seq2rel-ds
   cd seq2rel-ds

   # Install the package with poetry
   poetry install

Usage
-----

There are two submodules:


#. `preprocess`: used to preprocess existing datasets in a format that can be used with [`seq2rel`](https://github.com/JohnGiorgi/seq2rel).
#. ``align``\ : used to generate training data using weak/distant supervision.

preprocess
^^^^^^^^^^

To see the list of datasets available for preprocessing, call

.. code-block::

   seq2rel-ds preprocess --help

To preprocess the data (and in some cases, download it), call one of the commands, e.g.

.. code-block::

   seq2rel-ds preprocess ade "path/to/output/directory"

align
^^^^^

To see the list of strategies available for creating training data via weak/distant supervision, call

.. code-block::

   seq2rel-ds align --help

BioGRID
~~~~~~~

To use BioGRID and distant supervision to create training data, you first need to download a BioGRID release

.. code-block::

   seq2rel-ds align biogrid preprocess "path/to/output/directory" "path/to/biogrid/release"

..

   Note, it is strongly recommend to provide a `ScispaCy <https://allenai.github.io/scispacy/>`_ model name with the ``--scispacy-model`` flag. We currently recommend the ``"en_ner_jnlpba_md"`` model.

