Inferlylens documentation
==========================================

Inferlylens makes it easy to manipulate, plot and get insights from Gaussian process models. It is build on top of GPflow and tensorflow.

Install with poetry
###################

To install the library run ``poetry install`` in a terminal at the root of the repository. This will install the library and all its dependencies in a virtual environment.

Development setup
######################

The project uses *ruff* for formating and linting and *pytest* for testing. We also use taskify to make it easy run tests, or build the documentation:::

   poetry run task check
   poetry run task test

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   API reference <api>

.. toctree::
   :maxdepth: 1
   :hidden:

   quick_start.ipynb
