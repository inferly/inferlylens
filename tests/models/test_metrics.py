import gpflow
import numpy as np
import pandas as pd

import inferlyclient as infly
from inferlyclient.models.gpmodel import GPmodel
from inferlyclient.models.metrics import nlpd, q2, rmse

from ..utils import DATA_DIR

df = pd.read_parquet(DATA_DIR / "banana.parquet").astype("float64").iloc[:50]


def test_nlpd_shape():
    """Make sure rmse works with multioutputs."""
    dataset = infly.data.Dataset(df, ["x1"], ["x2", "y"])
    trainset, testset = dataset.split(sizes=[40, 10], labels=["train", "test"], seed=1234)
    X = trainset.df[trainset.input_names]
    Y = trainset.df[trainset.output_names]

    kernel = gpflow.kernels.Matern32(lengthscales=0.001, variance=100.0)
    model = GPmodel(gpflow.models.GPR((X, Y), kernel))
    np.testing.assert_allclose(2, np.asarray(nlpd(model, testset)).shape)


def test_rmse_shape():
    """Make sure rmse works with multioutputs."""
    dataset = infly.data.Dataset(df, ["x1"], ["x2", "y"])
    trainset, testset = dataset.split(sizes=[40, 10], labels=["train", "test"], seed=1234)
    X = trainset.df[trainset.input_names]
    Y = trainset.df[trainset.output_names]

    kernel = gpflow.kernels.Matern32(lengthscales=0.001, variance=100.0)
    model = GPmodel(gpflow.models.GPR((X, Y), kernel))
    np.testing.assert_allclose(2, np.asarray(rmse(model, testset)).shape)


def test_q2_shape():
    """Make sure rmse works with multioutputs."""
    dataset = infly.data.Dataset(df, ["x1"], ["x2", "y"])
    trainset, testset = dataset.split(sizes=[40, 10], labels=["train", "test"], seed=1234)
    X = trainset.df[trainset.input_names]
    Y = trainset.df[trainset.output_names]

    kernel = gpflow.kernels.Matern32(lengthscales=0.001, variance=100.0)
    model = GPmodel(gpflow.models.GPR((X, Y), kernel))
    np.testing.assert_allclose(2, np.asarray(q2(model, testset)).shape)


def test_q2():
    """Test q2 extreme values.

    Ensure that the q2 values are
        - 1 when the model predicts perfectly
        - 0 when the model predicts as well as the mean value of the test set
    """
    # load a dataset
    dataset = infly.data.Dataset(df, ["x1", "x2"], ["y"])
    trainset, testset = dataset.split(sizes=[40, 10], labels=["train", "test"], seed=1234)
    X = trainset.df[trainset.input_names]
    Y = trainset.df[trainset.output_names]
    Ytest = testset.df[testset.output_names]

    # create a GPmodel
    kernel = gpflow.kernels.Matern32(lengthscales=0.001, variance=100.0)
    model = gpflow.models.GPR((X, Y), kernel, mean_function=gpflow.mean_functions.Constant(np.mean(Ytest)))
    model.likelihood.variance.assign(1.001e-6)
    gpmodel = GPmodel(model)

    # check interpolation
    np.testing.assert_allclose(1.0, np.asarray(q2(gpmodel, trainset)))

    # check when model reverts to the mean
    np.testing.assert_allclose(0.0, np.asarray(q2(gpmodel, testset)))
