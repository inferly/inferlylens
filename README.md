[![Quality checks and Tests](https://github.com/NicolasDurrande/inferlylens/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/NicolasDurrande/inferlylens/actions/workflows/quality-checks.yaml)

# Inferlylens

Inferlylens makes it easy to plot and get insights from Gaussian process models. It is build on top of GPflow and tensorflow.


## Install

### Using poetry

To install the library run
```
poetry install
```
in a terminal at the root of the repo

## Development
The project uses *ruff* for formating and linting and *pytest* for testing. We also use taskify to make it easy run tests, or build the documentation:
```
poetry run task check
poetry run task test
```
In order to build the documentation, install [pandoc](https://pandoc.org/installing.html) and run `poetry run task doc`.