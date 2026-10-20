from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

from treetensorcomp.quantics.functions import QuanticsFunction


def multinormal_distribution(
    variable_names: Sequence[str],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    name: str = "multi_normal",
) -> QuanticsFunction:
    """
    Create a QuanticsFunction representing a multivariate normal distribution.

    Parameters
    ----------
    variable_names : Sequence[str]
        List of n names for the n variables.
    mu : NDArray[np.float64]
        n-dimensional array of means for the n variables.
    sigma : NDArray[np.float64]
        n x n, positive semidefinite covariance matrix for the n variables.
    name : str, optional
        Name of the function.

    Returns
    -------
    QuanticsFunction
        Resulting multivate normal probability density function.

    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    n = len(variable_names)

    if mu.shape != (n,):
        msg = "mu must be a length-n vector."
        raise ValueError(msg)

    if sigma.shape != (n, n):
        msg = "sigma must be n x n matrix"
        raise ValueError(msg)

    # Check the sigma matrix is a valid covariance matrix
    if not np.allclose(sigma, sigma.T):
        msg = "Covariance matrix sigma must be symmetric"
        raise ValueError(msg)

    if np.any(np.linalg.eigvalsh(sigma) < -1e-12):
        msg = "Covariance matrix sigma must be positive semidefinite"
        raise ValueError(msg)

    # Set variable domains to be +-5 standard deviations from the mean
    variable_domains = [
        (mu[i] - 5 * np.sqrt(sigma[i, i]), mu[i] + 5 * np.sqrt(sigma[i, i]))
        for i in range(n)
    ]

    def pdf_func(
        *,
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        **kwargs,
    ) -> NDArray[np.float64]:
        vals = [np.asarray(kwargs[var]) for var in variable_names]
        vals_broadcasted = np.broadcast_arrays(*vals)
        x = np.stack(vals_broadcasted, axis=-1)
        return multivariate_normal.pdf(x, mean=mu, cov=sigma)

    latex = (
        r"\mathcal{N}_{(\mu, \Sigma)}(\vec{x}="
        + ",".join(variable_names)
        + r") = \frac{1}{\sqrt{(2\pi)^n |\Sigma|}} \exp\left(-\frac{1}{2}"
        + r" (\vec{x} - \mu)^T \Sigma^{-1} (\vec{x} - \mu)\right)"
    )

    parameter_assignments = {"mu": mu, "sigma": sigma}

    return QuanticsFunction(
        function=pdf_func,
        variable_names=variable_names,
        parameter_assignments=parameter_assignments,
        variable_domains=variable_domains,
        name=name,
        latex=latex,
        assert_safe_signature=False,
    )
