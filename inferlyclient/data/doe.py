import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2


def doe_kmeans(n: int, dim: int, input_names: list[str] | None = None) -> pd.DataFrame:
    """Generates a Design of Experiment (DoE) where points are uniformely distributed in the unit cube.

    The DoE is obtained by running a k-means algorithm with $n$ classes on a large number of
    uniformly sampled points. This procedure ensures that the resulting DoE has a lower discrepency than
    $n$ uniformly sampled points.

    Args:
        n (int): number of points in the DoE.
        dim (int): dimension of the space.
        input_names (Optional, list of strings): variable name for each dimension.

    Returns:
        pandas DataFrame of shape n x dim.
    """
    if input_names is not None:
        if dim != len(input_names):
            raise ValueError("`input_names` must be a list with of length `dim`.")
    else:
        input_names = [f"x{i+1}" for i in range(dim)]

    U = np.random.uniform(size=(10 * n * dim, dim))
    X = kmeans2(U, n, minit="points", iter=100)[0]
    return pd.DataFrame(X, columns=input_names)


def cartesian_product(X1: pd.DataFrame, X2: pd.DataFrame) -> pd.DataFrame:
    r"""Generates the cartesian product of two DoEs.

    **Example**
    The following code generates a cartesian product of two dataframes with 11 and 5 points respectively.
    The resulting output is a dataframe with 55 points.

    .. code-block:: python

        import numpy as np
        import pandas as pd
        import inferlyclient as infly

        X1 = pd.DataFrame(np.linspace(0, 1, 11), columns=['x1'])
        X2 = pd.DataFrame(np.linspace(0, 1, 5), columns=['x2'])
        X = infly.doe.cartesian_product(X1, X2)

        print(X)

    Args:
        X1 (pd.DataFrame): first Doe, say with $n_1$ rows and $dim_1$ columns.
        X2 (pd.DataFrame): second Doe, say with $n_2$ rows and $dim_2$ columns.

    Returns:
        dataframe with $n_1 \times n_2$ rows and $d_1 + d_2$ columns.
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    X1rep = pd.DataFrame(np.repeat(X1.values, n2, axis=0), columns=X1.columns)
    X2rep = pd.DataFrame(np.tile(X2.values, (n1, 1)), columns=X2.columns)

    return pd.concat([X1rep, X2rep], axis=1)
