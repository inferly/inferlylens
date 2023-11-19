import pandas as pd
import pytest

from inferlyclient.data import Dataset

from ..utils import DATA_DIR


def test_dataset():
    """Test the Dataset class."""
    df = pd.read_parquet(DATA_DIR / "banana.parquet")

    dataset = Dataset(df, ["x1", "x2"], ["y"])

    with pytest.raises(ValueError):
        dataset = Dataset(df, ["x1", "g2"], ["y"])

    train_set, test_set = dataset.split([90, 10], ["train", "test"])
    assert train_set.df.shape[0] == 90
    assert test_set.df.shape[0] == 10
