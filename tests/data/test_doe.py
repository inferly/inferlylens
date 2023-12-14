import numpy as np
import pandas as pd

from inferlylens.data import cartesian_product, doe_kmeans


def test_doe_kmeans():
    """Smoke test of creating a doe and making sure shape is right."""
    X = doe_kmeans(n=15, dim=2)
    assert X.shape == (15, 2)

    X = doe_kmeans(10, 3, ["x", "y", "z"])
    assert X.shape == (10, 3)


def test_cartesian_product():
    """Smoke test for the cartesian product of two DoEs."""
    X1 = doe_kmeans(20, 3, ["x", "y", "z"])

    X2 = pd.DataFrame(np.linspace(0, 1, 10), columns=["t"])

    X = cartesian_product(X1, X2)

    assert X.shape == (200, 4)
    assert X.columns.to_list() == ["x", "y", "z", "t"]
