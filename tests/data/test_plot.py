import pandas as pd

from inferlylens.data.plots import gridplot, pairsplot

from ..utils import DATA_DIR

df = pd.read_parquet(DATA_DIR / "banana.parquet")


def test_pairsplot():
    """Smoke test for pairsplot."""
    fig = pairsplot(df, ["x1", "x2", "y"], thinning=0.9)
    assert fig is not None


def test_gridplot():
    """Smoke test for gridplot."""
    fig = gridplot(df, ["x1", "x2"], ["y"], thinning=0.9)
    assert fig is not None

    fig = gridplot(df, ["x1", "x2"], ["y"], color="y", thinning=0.9)
    assert fig is not None
