import sys
from pathlib import Path

import gpflow
import numpy as np
import pandas as pd

from inferlyclient.data.transform import standardise_rescaling, unit_cube_rescaling
from inferlyclient.models.gpmodel import GPmodel
from inferlyclient.models.plots import plot_slices


def test_model_plot_slices():
    """Smoke test for the plot_slices function."""
    # load a dataset
    root = Path(sys.modules['inferlyclient'].__file__).parent.parent
    df = pd.read_parquet(root / "datasets/banana.parquet").astype("float64")
    X, Y = df.iloc[:50, :2], df.iloc[:50, 2:]
    Y = Y - Y.mean()

    # create transforms
    input_transform = unit_cube_rescaling(X)
    output_transform = standardise_rescaling(Y)

    # create a GPmodel with transform
    Xt, Yt = input_transform.forward(X), output_transform.forward(Y)
    kernel = gpflow.kernels.Matern32(lengthscales=0.2, variance=1.0)
    model = gpflow.models.GPR((Xt, Yt), kernel)
    model.likelihood.variance.assign(0.01)
    gpmodel = GPmodel(model, input_transform, output_transform)

    plot_slices(
        gpmodel, ["x1", "x2"], ["y"], reference_point=np.zeros((1, 2)), xlim=np.array([[-1, 0], [1, 2]])
    )
    # fig.show()
