import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models.model import GPModel as gpflowGPModel

from ..data import Dataset
from ..types import TensorType


class GPmodel:
    """Wrapper around GPflow models to include input and output transforms.

    Args:
        gpflow: a gpflow model, where the data has already been transformed
        input_transform: a tensorflow probability bijector
        output_transform: a tensorflow probability bijector

    Methods:
        predict_y
        predict_quantiles
        predict_log_density
    """

    def __init__(
        self,
        gpflow: gpflowGPModel,
        input_transform: tfp.bijectors.Bijector = tfp.bijectors.Identity(),
        output_transform: tfp.bijectors.Bijector = tfp.bijectors.Identity(),
    ) -> None:

        self.gpflow = gpflow
        self.input_transform = input_transform
        self.output_transform = output_transform

    def predict_y(self, Xnew: TensorType) -> tuple[tf.Tensor, tf.Tensor]:
        """Wrapper around gpflow.models.predict_y that applies the input and output transforms.

        Args:
            Xnew (tf.Tensor): n x input_dim tensor of inputs

        Returns:
            mean (tf.Tensor): n x output_dim tensor of means
            var (tf.Tensor): n x output_dim tensor of variances
        """
        mean, var = self.gpflow.predict_y(self.input_transform.forward(Xnew))
        dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var))
        inv_output_transform = tfp.bijectors.Invert(self.output_transform)
        dist_new = tfp.distributions.TransformedDistribution(dist, inv_output_transform)
        return dist_new.mean(), dist_new.variance()

    def predict_quantiles(self, Xnew: TensorType, levels: TensorType) -> tf.Tensor:
        """Wrapper around gpflow.models.predict_y that applies the input and output transforms.

        Args:
            Xnew (tf.Tensor, shape n * input_dim): input_dim tensor of inputs.
            levels (tf.Tensor, shape m): tensor of quantile levels, expressed in [0, 1].

        Returns:
            quantiles (tf.Tensor): m x n x output_dim tensor of quantiles.
        """
        mean, var = self.gpflow.predict_y(self.input_transform.forward(Xnew), full_cov=False)
        dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var))
        quantiles = dist.quantile(levels[:, None, None])
        return self.output_transform.inverse(quantiles)

    def predict_log_density(self, data: Dataset) -> tf.Tensor:
        """Wrapper around gpflow.models.predict_density that applies the input and output transforms."""
        Xnew, Ynew = data.df[data.input_names], data.df[data.output_names]
        x = self.input_transform.forward(Xnew)
        y = self.output_transform.forward(Ynew)

        if y.shape[-1] == 1:
            density = self.gpflow.predict_log_density((x, y))
        elif isinstance(self.gpflow.likelihood, gpflow.likelihoods.Gaussian):
            mu, var = self.gpflow.predict_y(x)
            density = gpflow.logdensities.gaussian(y, mu, var)
        else:  # pragma: no cover
            return NotImplementedError(
                "predict_log_density for multioutput models is only implemented for the Gaussian likelihood."
            )

        return density + self.output_transform.forward_log_det_jacobian(y, event_ndims=0)
