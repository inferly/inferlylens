import numpy as np
import pandas as pd
import tensorflow as tf

from ..data import Dataset
from .gpmodel import GPmodel


def rmse(model: GPmodel, test_data: Dataset) -> pd.Series:
    """Root mean squared error.

    Args:
        model (GPmodel): A GPmodel object.
        test_data (tuple[tf.Tensor, tf.Tensor]): A tuple of tensors (Xtest, Ytest) with the test data.

    Returns:
        pd.Series: The root mean squared error for each output.
    """
    Xtest, Ytest = (test_data.df[io].to_numpy() for io in [test_data.input_names, test_data.output_names])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(Ytest - model.predict_y(Xtest)[0]), axis=0))
    return pd.Series(rmse.numpy(), index=test_data.output_names)


def nlpd(model: GPmodel, test_data: Dataset) -> pd.Series:
    """Negative log predictive density (i.e. minus the average of the log likelihood on test set).

    Note that this computes the average of the *pointwise* log likelihood, which means that
    the covariance between inputs is ignored.

    Args:
        model (GPmodel): A GPmodel object.
        test_data (inferlylens Dataset): test data.

    Returns:
        pd.Series: The negative log predictive density for each output.
    """
    min_log_likelihood = -model.predict_log_density(test_data).numpy().mean(axis=0)
    return pd.Series(min_log_likelihood, index=test_data.output_names)


def q2(model: GPmodel, test_data: Dataset) -> pd.Series:
    """Percentage of explained variance.

    Q2 is equal to 1 if the model predicts perfectly, and 0 if the model predicts as well as the
    mean value of the test set.

    Args:
        model (GPmodel): A GPmodel object.
        test_data (inferlylens Dataset): test data.

    Returns:
        pd.Series: The q2 metric for each output.
    """
    Xtest, Ytest = test_data.df[test_data.input_names], test_data.df[test_data.output_names]
    model_var = np.sum((np.asarray(Ytest) - model.predict_y(Xtest)[0]) ** 2, axis=0)
    var = np.sum((Ytest - Ytest.mean()) ** 2, axis=0)
    return 1 - model_var / var
