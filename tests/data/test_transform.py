import numpy as np
import pandas as pd

from inferlylens.data import standardise_rescaling, unit_cube_rescaling

from ..utils import DATA_DIR

df = pd.read_parquet(DATA_DIR / "banana.parquet")


def test_unit_cube_rescaling():
    """Ensure that min (resp. max) value of data rescaled with unit_cube_rescaling is 0 (resp. 1)."""
    bijector = unit_cube_rescaling(df)

    np.testing.assert_almost_equal(bijector.forward(df).numpy().min(axis=0), 0, decimal=5)
    np.testing.assert_almost_equal(bijector.forward(df).numpy().max(axis=0), 1, decimal=5)


def test_standardise_rescaling():
    """Ensure that data rescaled with standardise_rescaling has mean 0 and std 1."""
    bijector = standardise_rescaling(df)

    np.testing.assert_almost_equal(bijector.forward(df).numpy().mean(axis=0), 0, decimal=5)
    np.testing.assert_almost_equal(bijector.forward(df).numpy().std(axis=0), 1, decimal=5)
