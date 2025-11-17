import pandas as pd
import tensorflow_probability as tfp


def unit_cube_rescaling(data: pd.DataFrame) -> tfp.bijectors.Bijector:
    """Computes the linear bijector that maps the data to the unit cube.

    Args:
        data (pandas DataFrame): dataframe used to infer the bounds.

    Returns:
        tfp.bijectors.Bijector: bijector that maps the data to the unit cube.
    """
    Shift = tfp.bijectors.Shift(-data.min().to_numpy())
    Scale = tfp.bijectors.Scale(1 / (data.max().to_numpy() - data.min().to_numpy()))

    return tfp.bijectors.Chain([Scale, Shift])


def standardise_rescaling(data: pd.DataFrame) -> tfp.bijectors.Bijector:
    """Computes the linear bijector that maps the data to zero mean unit variance.

    Args:
        data (pandas DataFrame): dataframe used to infer the first two moments.

    Returns:
        tfp.bijectors.Bijector: bijector that maps the data to the standard normal.
    """
    Shift = tfp.bijectors.Shift(-data.mean().to_numpy())
    Scale = tfp.bijectors.Scale(1 / data.std(ddof=0).to_numpy())

    return tfp.bijectors.Chain([Scale, Shift])
