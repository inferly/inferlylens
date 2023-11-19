[![Quality checks and Tests](https://github.com/NicolasDurrande/inferlyclient/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/NicolasDurrande/inferlyclient/actions/workflows/quality-checks.yaml)

# inferlyclient: Visualising and interacting with Gaussian process models

This library facilitates the manipulation of GP models and makes it easy to get insights from them.

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
In order to build the documentation, install `pandoc` and run `poetry run task docs`.
Code coverage can be obtained with `poetry run task cov`.