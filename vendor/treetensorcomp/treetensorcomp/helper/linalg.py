from collections.abc import Sequence
from functools import reduce

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr


def generate_random_isometry(n: int, m: int) -> NDArray[np.float64]:
    """
    Generate a random isometry from an m-dimensional space to an n-dimensional space.

    Uses the method from [arXiv:math-ph/0609050](https://arxiv.org/abs/math-ph/0609050),
    generating a random unitary then extracting the isometry.

    Parameters
    ----------
    n : int
        Number of rows in the isometry matrix.
    m : int
        Number of columns in the isometry matrix.

    Returns
    -------
    NDArray[np.float64]
        The n x m random isometry matrix.

    """
    if not n >= m:
        msg = "n must be greather than or equal to m"
        raise ValueError(msg)

    # Generate a random complex matrix
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)

    # Perform QR decomposition
    q, r = qr(z)

    # Normalize the diagonal of R to have unit magnitude
    d = np.diagonal(r)
    ph = d / np.abs(d)

    # Adjust the phases of Q
    q = q @ np.diag(ph) @ q

    # Return the first m columns of Q as the isometry
    return q[:, :m]


def sp_tensor_prod(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate the shape-preserving tensor product of two arrays.

    Parameters
    ----------
    a, b : NDArray[np.float64]
        The left and right arrays to evaluate the tensor product of.

    Returns
    -------
    NDArray[np.float64]
        The resulting array from the tensor product.

    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    # Pad with singleton dimensions to ensure broadcastable (final multiplication step)
    shape_a = a.shape + (1,) * b.ndim
    shape_b = (1,) * a.ndim + b.shape

    # Elementwise product yields simple tensor product with these shapes
    return a.reshape(shape_a) * b.reshape(shape_b)


def sp_multi_tensor_prod(arrays: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    """
    Evaluate the repeated tensor product of a sequence of arrays.

    Parameters
    ----------
    arrays : Sequence[NDArray[np.float64]]
        Sequence of arrays to evaluate the tensor product (from left to right).

    Returns
    -------
    NDArray[np.float64]
        The resulting from the repeated tensor products.

    """
    if not isinstance(arrays, (list, tuple)) or not all(
        isinstance(arr, np.ndarray) for arr in arrays
    ):
        msg = "arrays must be a list or tuple or numpy arrays."
        raise TypeError(msg)

    # Reduce with tensor product operation
    return reduce(sp_tensor_prod, [np.atleast_1d(arr) for arr in arrays])
