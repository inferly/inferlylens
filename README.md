[![Quality checks and Tests](https://github.com/inferly/inferlylens/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/inferly/inferlylens/actions/workflows/quality-checks.yaml)

# Inferlylens

Inferlylens makes it easy to plot and get insights from Gaussian process models. It is built on top of [GPflow](https://www.gpflow.org/) and [TensorFlow](https://www.tensorflow.org/).


## Install

### Using poetry

To install the library, install `uv` and run
```
uv sync
```
in a terminal at the root of the repo

## Development
The project uses *ruff* for formatting and linting and *pytest* for testing. We also use taskipy to make it easy run tests:
```
uv run task check
uv run task test
```
In order to build the documentation, install [pandoc](https://pandoc.org/installing.html) and run `uv run --group doc task doc`.