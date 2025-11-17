[![Quality checks and Tests](https://github.com/NicolasDurrande/inferlylens/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/NicolasDurrande/inferlylens/actions/workflows/quality-checks.yaml)

# Inferlylens

Inferlylens makes it easy to plot and get insights from Gaussian process models. It is build on top of GPflow and tensorflow.


## Install

### Using poetry

To install the library, install `uv` and run
```
uv sync
```
in a terminal at the root of the repo

## Development
The project uses *ruff* for formating and linting and *pytest* for testing. We also use taskify to make it easy run tests, or build the documentation:
```
uv run task check
uv run task test
```
In order to build the documentation, install [pandoc](https://pandoc.org/installing.html) and run `uv run task doc`.