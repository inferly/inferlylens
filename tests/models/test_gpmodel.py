import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from inferlylens.data import Dataset
from inferlylens.data.transform import standardise_rescaling, unit_cube_rescaling
from inferlylens.models.gpmodel import GPmodel

from ..utils import DATA_DIR


def test_gpmodel():
    """Test for GPmodel class and the three predict functions."""
    # load a dataset
    df = pd.read_parquet(DATA_DIR / "banana.parquet").astype("float64")
    inputs, outputs = ["x1", "x2"], ["y"]
    X, Y = df.iloc[:50, :2], df.iloc[:50, 2:]
    Y = Y - Y.mean()

    # create a baseline GPflow model without transform
    kernel = gpflow.kernels.Matern32(lengthscales=np.ones(2), variance=1.0)
    model = gpflow.models.GPR(data=(X.values, Y.values), kernel=kernel)
    model.likelihood.variance.assign(0.01)

    # create transforms
    input_transform = unit_cube_rescaling(X)
    input_scale = input_transform.bijectors[0]
    output_transform = standardise_rescaling(Y)
    output_scale = output_transform.bijectors[0]

    # create a GPmodel with transform
    Xt, Yt = input_transform.forward(X), output_transform.forward(Y)
    lengthscale_t = input_scale.forward(kernel.lengthscales)
    var_t = output_scale.forward(output_scale.forward(kernel.variance))[0]
    kernel_t = gpflow.kernels.Matern32(lengthscales=lengthscale_t, variance=var_t)
    model_t = gpflow.models.GPR((Xt, Yt), kernel_t)
    noise_t = output_scale.forward(output_scale.forward(model.likelihood.variance))[0]
    model_t.likelihood.variance.assign(noise_t)
    gpmodel = GPmodel(model_t, input_transform, output_transform)

    # test that the mean and variance predictions are the same
    Xnew, Ynew = df.iloc[100:103, :2].values, df.iloc[100:103, 2:].values
    mean, var = model.predict_y(Xnew)
    mean_t, var_t = gpmodel.predict_y(Xnew)
    np.testing.assert_allclose(mean, mean_t)
    np.testing.assert_allclose(var, var_t)

    # test that the log density predictions are the same
    # Xnew, Ynew = df.iloc[100:103, :2].values, df.iloc[100:103, 2:].values
    testset = Dataset(df.iloc[100:103, :], inputs, outputs)
    Xnew, Ynew = (np.asarray(testset.df[var]) for var in [inputs, outputs])
    density = model.predict_log_density((Xnew, Ynew))
    density_t = gpmodel.predict_log_density(testset)

    np.testing.assert_allclose(density, density_t)

    # test quantile predictions
    m, v = model.predict_y(Xnew)
    quantile = tfp.distributions.Normal(loc=m, scale=np.sqrt(v.numpy())).quantile([0.5, 0.9])
    quantile_t = gpmodel.predict_quantiles(Xnew, np.array([0.5, 0.9]))[:, :, 0]

    np.testing.assert_allclose(tf.transpose(quantile), quantile_t)
