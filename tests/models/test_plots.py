import gpflow
import numpy as np
import pandas as pd

from inferlyclient.data import Dataset
from inferlyclient.data.transform import standardise_rescaling, unit_cube_rescaling
from inferlyclient.models.gpmodel import GPmodel
from inferlyclient.models.plots import plot_actual_vs_predicted, plot_lengthscales, plot_slices

from ..utils import DATA_DIR


def test_model_plot_slices():
    """Smoke test for the plot_slices function."""
    # load a dataset
    df = pd.read_parquet(DATA_DIR / "banana.parquet").astype("float64")
    X, Y = df.iloc[:50, :2], df.iloc[:50, 2:]

    # create transforms
    input_transform = unit_cube_rescaling(X)
    output_transform = standardise_rescaling(Y)

    # create a GPmodel with transform
    Xt, Yt = input_transform.forward(X), output_transform.forward(Y)
    kernel = gpflow.kernels.Matern32(lengthscales=0.2, variance=1.0)
    model = gpflow.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(0.01)
    gpmodel = GPmodel(model, input_transform, output_transform)

    fig = plot_slices(
        gpmodel, ["x1", "x2"], ["y"], reference_point=np.zeros((1, 2)), xlim=np.array([[-1, 0], [1, 2]])
    )
    # fig.show()
    assert fig is not None


def test_model_plot_lengthscales():
    """Smoke test for the plot_lengthscales function."""
    # load a dataset
    df = pd.read_parquet(DATA_DIR / "banana.parquet").astype("float64")
    X, Y = df.iloc[:50, :2], df.iloc[:50, 2:]

    # create transforms
    input_transform = unit_cube_rescaling(X)
    output_transform = standardise_rescaling(Y)

    # create a GPmodel with transform
    Xt, Yt = input_transform.forward(X), output_transform.forward(Y)
    kernel = gpflow.kernels.Matern32(lengthscales=0.2, variance=1.0)
    model = gpflow.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(0.01)
    gpmodel = GPmodel(model, input_transform, output_transform)

    fig = plot_lengthscales(gpmodel.gpflow.kernel, ["x1", "x2"])
    # fig.show()
    assert fig is not None


def test_model_plot_actual_vs_predicted():
    """Smoke test for the plot_actual_vs_predicted function."""
    # load a dataset
    inputs = ["x1", "x2"]
    outputs = ["y"]
    df = pd.read_parquet(DATA_DIR / "banana.parquet").astype("float64")
    train_set, test_set = Dataset(df, inputs, outputs).split([50, 100], ["train", "test"])

    # create transforms
    input_transform = unit_cube_rescaling(train_set.df[inputs])
    output_transform = standardise_rescaling(train_set.df[outputs])

    # create a GPmodel with transform
    data = (input_transform.forward(train_set.df[inputs]), output_transform.forward(train_set.df[outputs]))
    kernel = gpflow.kernels.Matern32(lengthscales=0.2, variance=1.0)
    model = gpflow.models.GPR(data, kernel)
    model.likelihood.variance.assign(0.01)
    gpmodel = GPmodel(model, input_transform, output_transform)

    fig = plot_actual_vs_predicted(gpmodel, test_set)
    # fig.show()
    assert fig is not None
