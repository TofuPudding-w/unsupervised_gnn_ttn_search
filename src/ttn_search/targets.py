"""Discrete Gaussian probability amplitude targets; q0 is the least significant bit."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass
class Target:
    state: np.ndarray
    metadata: dict

    @property
    def hash(self):
        digest = hashlib.sha256(np.asarray(self.state, dtype='<c16').tobytes())
        digest.update(json.dumps(self.metadata, sort_keys=True).encode())
        return digest.hexdigest()


def gaussian_target(n=8, family='hierarchical', seed=0) -> Target:
    """
    Entrance point for generating a discrete Gaussian target state with a given number of qubits, family type, and random seed.
    The target state is represented as a normalized complex numpy array, and metadata includes parameters used for
    """
    if n < 4 or n % 2:
        raise ValueError('Gaussian pilot uses an even n >= 4 with two bits per variable')
    rng = np.random.default_rng(seed)
    d = n // 2
    # Define the factor loading matrix based on the specified family type.
    # Row j corresponds to variable j, 
    # and column 0 is the global factor, the remaining d+1 columns are local factors.
    loadings = np.zeros((d, d + 2))
    # Let the column 0 be the global factor, which is shared by all variables.
    loadings[:, 0] = 0.25
    if family == 'hierarchical':
        # hierarchical：变量对共享局部因子，同时有较弱的全局因子
        for j in range(d):
            # The j only depends on the 1 + j // 2-th local factor,
            # which is shared with the other variable in the same pair (j//2).
            loadings[j, 1 + j // 2] = rng.uniform(0.8, 1.2)
    elif family == 'overlapping':
        # overlapping：变量对共享局部因子，形成交叠关系
        for j in range(d):
            # The j-th variable depends on the 1 + j-th local factor,
            # which is shared with the next variable (j+1)%d, creating an overlapping
            loadings[j, 1 + j] = rng.uniform(0.7, 1.1)
            loadings[(j + 1) % d, 1 + j] = rng.uniform(0.5, 0.9)
    else:
        raise ValueError(f'Unknown family: {family}')
    # Compute the covariance matrix
    covariance = loadings @ loadings.T + 0.25 * np.eye(d)
    # Define the mean vector, 
    # ensuring it is within the range [-0.3, 0.3] for each dimension
    mean = rng.uniform(-0.3, 0.3, d)
    labels = np.arange(2**n)
    # Generated data x is a 2D array 
    # where each row corresponds to a label and each column corresponds to a variable.
    x = np.stack([-2 + 4 * ((labels >> (2*j)) & 3) / 3 for j in range(d)], axis=1)
    delta = x - mean
    logp = -0.5 * np.einsum('bi,ij,bj->b', delta, np.linalg.inv(covariance), delta)
    # A 2D array of shape (2**n, 1)
    state = np.exp(0.5 * (logp - logp.max())).astype(complex)
    state /= np.linalg.norm(state)
    return Target(state, dict(n=n, family=family, seed=seed, encoding='sqrt_probability',
        bits_per_variable=2, domain=[-2, 2], grid='inclusive_four_points',
        bit_order='q0_least_significant; variable j uses q(2j),q(2j+1)',
        covariance=covariance.tolist(), mean=mean.tolist()))
