from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Main class for handling datasets.

    Args:
        df (pandas dataframe): Dataframe containing the data.
        input_names (list of strings): Name of the columns associated with input variables.
        output_names (list of strings): Name of the columns associated with output variables.

    Attributes:
        input_names (list of strings): value passed at construction.
        output_names (list of strings): value passed at construction.
        data (pandas DataFrame): n x input_dim dataframe
    """

    df: pd.DataFrame
    input_names: list[str]
    output_names: list[str]

    def __post_init__(self):
        """Check consisency of the inputs provided."""
        for name in self.input_names + self.output_names:
            if name not in self.df.columns:
                raise ValueError(f"variable '{name}' not found in dataset")

    def split(self, sizes: list[int], labels: list[Any], seed: int | None = None) -> list["Dataset"]:
        """Split a dataset, typically into a training and test set.

        The method also adds a column to the dataset's dataframe to indicate whether
        a point is in the training or test set.

        Args:
            sizes (list of integers): number of points to keep for each split.
            labels (list): labels associated with each split.
            seed (int): Seed for the random number generator.

        Returns:
            train_set (Dataset): Training set with n_train entries.
            test_set (Dataset): Test set with remaining entries.
        """
        n = self.df.shape[0]
        assert np.sum(sizes) <= n, "The sum of `sizes` must be less or equal to the number of data points."
        full_labels = [[label] * size for size, label in zip(sizes, labels)] + [[None] * (n - np.sum(sizes))]
        full_labels = [label for sublist in full_labels for label in sublist]
        rng = np.random.default_rng(seed)
        self.df["label"] = rng.permutation(full_labels)

        datasets = []
        for lab in labels:
            filtered_df = pd.DataFrame(self.df[self.df["label"] == lab])  # makes pyright happy
            datasets.append(Dataset(filtered_df, self.input_names, self.output_names))

        return datasets
