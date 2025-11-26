from typing import Any

import numpy as np
import tensorflow as tf

TensorType = np.ndarray[Any, Any] | tf.Tensor | tf.Variable
