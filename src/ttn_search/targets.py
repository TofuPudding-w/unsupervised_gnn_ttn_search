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
    if n < 4 or n % 2:
        raise ValueError('Gaussian pilot uses an even n >= 4 with two bits per variable')
    rng = np.random.default_rng(seed)
    d = n // 2
    loadings = np.zeros((d, d + 2))
    loadings[:, 0] = 0.25
    if family == 'hierarchical':
        for j in range(d):
            loadings[j, 1 + j // 2] = rng.uniform(0.8, 1.2)
    elif family == 'overlapping':
        for j in range(d):
            loadings[j, 1 + j] = rng.uniform(0.7, 1.1)
            loadings[(j + 1) % d, 1 + j] = rng.uniform(0.5, 0.9)
    else:
        raise ValueError(f'Unknown family: {family}')
    covariance = loadings @ loadings.T + 0.25 * np.eye(d)
    mean = rng.uniform(-0.3, 0.3, d)
    labels = np.arange(2**n)
    x = np.stack([-2 + 4 * ((labels >> (2*j)) & 3) / 3 for j in range(d)], axis=1)
    delta = x - mean
    logp = -0.5 * np.einsum('bi,ij,bj->b', delta, np.linalg.inv(covariance), delta)
    state = np.exp(0.5 * (logp - logp.max())).astype(complex)
    state /= np.linalg.norm(state)
    return Target(state, dict(n=n, family=family, seed=seed, encoding='sqrt_probability',
        bits_per_variable=2, domain=[-2, 2], grid='inclusive_four_points',
        bit_order='q0_least_significant; variable j uses q(2j),q(2j+1)',
        covariance=covariance.tolist(), mean=mean.tolist()))
