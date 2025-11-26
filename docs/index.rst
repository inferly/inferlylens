Inferlylens documentation
==========================================

Inferlylens is library that wraps GPflow models to make them easier to manipulate and plot.

Install with uv
###################

To install the library and all its dependencies in a virtual environment, you simply need to run

   uv sync
   
in a terminal at the root of the repository.

Development setup
######################

The project uses *ruff* for formating and linting and *pytest* for testing. We also use *taskipy* to make it easy run tests, or build the documentation:::

   uv run task check
   uv run task test

.. toctree::
   :maxdepth: 1
   :hidden:

   quick_start

.. toctree::
   :maxdepth: 4
   :titlesonly:
   :hidden:

   API reference <api>
