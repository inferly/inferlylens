import gpflow
import numpy as np
import pandas as pd

import inferlyclient as infly
from inferlyclient.models.gpmodel import GPmodel
from inferlyclient.models.metrics import q2


def test_metrics():
    """Test q2 extreme values.

    Ensure that the q2 values are
        - 1 when the model predicts perfectly
        - 0 when the model predicts as well as the mean value of the test set
    """
    # load a dataset
    df = pd.read_parquet("datasets/banana.parquet").astype("float64").iloc[:50]
    dataset = infly.data.Dataset(df, ["x1", "x2"], ["y"])
    trainset, testset = dataset.split(sizes=[40, 10], labels=["train", "test"], seed=1234)
    X = trainset.df[trainset.input_names]
    Y = trainset.df[trainset.output_names]
    Ytest = testset.df[testset.output_names]

    # create a GPmodel with transform
    kernel = gpflow.kernels.Matern32(lengthscales=0.001, variance=100.0)
    model = gpflow.models.GPR((X, Y), kernel, mean_function=gpflow.mean_functions.Constant(np.mean(Ytest)))
    model.likelihood.variance.assign(1.001e-6)
    gpmodel = GPmodel(model)

    np.testing.assert_allclose(1.0, np.asarray(q2(gpmodel, trainset)))

    # hack the test set to be the mean of predictions
    np.testing.assert_allclose(0.0, np.asarray(q2(gpmodel, testset)))
