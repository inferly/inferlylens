import pandas as pd

import inferlylens.data as inflydata
from inferlylens.data.plots import gridplot, pairsplot

from ..utils import DATA_DIR

df = pd.read_parquet(DATA_DIR / "banana.parquet")


def test_pairsplot():
    """Smoke test for pairsplot."""
    fig = pairsplot(df, ["x1", "x2", "y"], thinning=0.9)
    assert fig is not None

    dataset = inflydata.Dataset(df, input_names=["x1", "x2"], output_names=["y"])
    dataset.split([90, 30, 10], ["train", "test", "val"])
    fig = pairsplot(dataset.df, ["x1", "x2", "y"], color="split", thinning=0.9)
    assert fig is not None


def test_gridplot():
    """Smoke test for gridplot."""
    fig = gridplot(df, ["x1", "x2"], ["y"], thinning=0.9)
    assert fig is not None

    fig = gridplot(df, ["x1", "x2"], ["y"], color="y", thinning=0.9)
    assert fig is not None

    dataset = inflydata.Dataset(df, input_names=["x1", "x2"], output_names=["y"])
    dataset.split([90, 30, 10], ["train", "test", "val"])
    fig = gridplot(dataset.df, ["x1"], ["x2"], color="split", thinning=1.0)
    assert fig is not None
