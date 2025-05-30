"""datasets.py.

Utilities for loading the MNIST dataset by tensorflow.

Ref: https://www.tensorflow.org/datasets/catalog/mnist?hl=es

Example:
-------
>>> from mi_cnn.data.datasets import get_mnist_data
>>> (x_train, y_train), (x_test, y_test) = get_mnist_data(
...     flatten=True,       # 28×28 -> 784
... )

"""

from __future__ import annotations

import numpy as np
from tensorflow.keras.datasets import mnist

__all__ = ["get_mnist_data"]


def get_mnist_data(
    flatten: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return *train* and *test* splits of MNSIT dataset.

    Parameters
    ----------
    flatten
        Whether flat each image 28x28 into a 784 array (default: ``False``).

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        Tuples with NumPy arrays.  ``x_train`` y ``x_test`` are normalized
        in range ``[0, 1]`` (``float32``).  ``y_train``/``y_test`` are integers
        ``uint8`` in range *[0, 9]*.
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # convertion to float32 and normalisation
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    return (x_train, y_train), (x_test, y_test)
