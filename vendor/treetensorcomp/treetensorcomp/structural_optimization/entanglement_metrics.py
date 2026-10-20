from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import ishermitian, sqrtm

from treetensorcomp.quantics.index_settings import IndexID
from treetensorcomp.tensor_networks.tensors import Tensor

# Entanglement metrics: mutual information / entanglement entropy / variation of information.
# Generate these from various inputs, such as labelled Tensor and matrix inputs.


def _matricize_rdm(rho: Tensor, indices: list[str]) -> NDArray[np.float64]:
    """
    Matricize a density matrix Tensor into a numpy NDArray.

    Parameters
    ----------
    rho : Tensor
        The density matrix in Tensor representation. Must be Hermitian.
    indices : list[str]
        The identifiers representing the (ket) indices of the density matrix.

    Returns
    -------
    NDArray[np.float64]
        Matricized density matrix.

    """
    assert isinstance(rho, Tensor), "rho must be a Tensor instance"
    assert isinstance(indices, list) and all(isinstance(i, str) for i in indices), (
        "indices must be a list of strings"
    )
    conjugate_indices = [i + "*" for i in indices]
    assert set(indices + conjugate_indices) == set(rho.indices), (
        "rho must be a density matrix with indices and conjugate indices matching the input indices"
    )
    density_matrix, shape_dict = rho.matricize(conjugate_indices, indices)

    # Check it's Hermitian
    assert ishermitian(density_matrix, rtol=1e-8), (
        "The density matrix must be Hermitian"
    )

    return density_matrix


def _get_subsystem_entropies(
    rho: Tensor,
    indices: Sequence[IndexID],
) -> tuple[float, float, float]:
    """
    Calculate the joint and individual entropies of a two qubit density matrix.

    Parameters
    ----------
    rho : Tensor
        The density matrix in Tensor representation. Must be normalized and Hermitian.
    indices : Sequence[IndexID]
        The IndexIDs representing the (ket) indices of the density matrix.

    Returns
    -------
    s_ab: float
        The joint entropy of the system.

    s_a, s_b: float
        The individual entropies of each of the qubits in the system.

    """
    assert isinstance(indices, (list, tuple)) and all(
        isinstance(i, IndexID) for i in indices
    ), "indices must be a list or tuple of IndexIDs"
    assert len(indices) == 2, "mutual_information requires exactly two indices"
    assert all(dim == 2 for dim in rho.tensor_ndarray.shape), (
        "All dimensions of rho.tensor_ndarray must be 2 (qubit indices)"
    )

    # Qubit subspace reduced density matrices
    rho_a = rho.partial_trace((str(indices[1]),))
    rho_b = rho.partial_trace((str(indices[0]),))

    # Calculate the von Neumann entropies
    s_ab = von_neumann_entropy(rho, indices)
    s_a = von_neumann_entropy(rho_a, [indices[0]])
    s_b = von_neumann_entropy(rho_b, [indices[1]])

    return s_ab, s_a, s_b


def von_neumann_entropy(rho: Tensor, indices: Sequence[IndexID]) -> float:
    """
    Calculate the von Neumann entropy of a density matrix.

    Parameters
    ----------
    rho : Tensor
        The density matrix in Tensor representation. Must be normalized and Hermitian.
    indices : Sequence[IndexID]
        The IndexIDs representing the (ket) indices of the density matrix.

    Returns
    -------
    float
        von Neumann entropy of the system.

    """
    density_matrix = _matricize_rdm(rho, [str(i) for i in indices])
    assert np.isclose(np.trace(density_matrix), 1), (
        "Entanglement metrics require a normalized density matrix (trace ≈ 1)"
    )

    # Calculate eigenvalues and use them to compute the von Neumann entropy
    # It's better to do this than with matrix log for numerical stability
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Keep only positive eigenvalues

    return -np.sum(eigenvalues * np.log(eigenvalues))


def mutual_information(rho: Tensor, indices: Sequence[IndexID]) -> float:
    """
    Calculate the mutual information of a two qubit density matrix.

    Parameters
    ----------
    rho : Tensor
        The density matrix in Tensor representation. Must be normalized and Hermitian.
    indices : Sequence[IndexID]
        The IndexIDs representing the (ket) indices of the density matrix.

    Returns
    -------
    float
        Quantum mutual information of the system.

    """
    s_ab, s_a, s_b = _get_subsystem_entropies(rho, indices)
    return s_a + s_b - s_ab


def concurrence(rho: Tensor, indices: Sequence[IndexID]) -> float:
    """
    Calculate the concurrence of a two qubit density matrix.

    Parameters
    ----------
    rho : Tensor
        The density matrix in Tensor representation. Must be normalized and Hermitian.
    indices : Sequence[IndexID]
        The IndexIDs representing the (ket) indices of the density matrix.

    Returns
    -------
    float
        Concurrence of the system.

    """
    assert isinstance(indices, (list, tuple)) and all(
        isinstance(i, IndexID) for i in indices
    ), "indices must be a list or tuple of IndexIDs"
    assert len(indices) == 2, "concurrence requires exactly two indices"
    assert all(dim == 2 for dim in rho.tensor_ndarray.shape), (
        "All dimensions of rho.tensor_ndarray must be 2 (qubit indices)"
    )
    density_matrix = _matricize_rdm(rho, [str(i) for i in indices])
    assert np.isclose(np.trace(density_matrix), 1), (
        "Entanglement metrics require a normalized density matrix (trace ≈ 1)"
    )

    # Spin-flip operation matrix for two qubits
    spin_flip_matrix = np.array(
        [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
        dtype=float,
    )

    # Compute the R matrix
    rho_tilde = spin_flip_matrix @ np.conjugate(density_matrix) @ spin_flip_matrix
    sqrt_rho = sqrtm(density_matrix)
    r_matrix = sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)

    # Eigenvalues of R (in descending order)
    lambdas = np.linalg.eigvalsh(r_matrix)[::-1]

    # Calculate concurrence
    return max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])


def entanglement_formation(rho: Tensor, indices: Sequence[IndexID]) -> float:
    """
    Calculate the entanglement of formation of a two qubit density matrix.

    Parameters
    ----------
    rho : Tensor
        The density matrix in Tensor representation. Must be normalized and Hermitian.
    indices : Sequence[IndexID]
        The IndexIDs representing the (ket) indices of the density matrix.

    Returns
    -------
    float
        Entanglement of formation of the system.

    """
    c = concurrence(rho, indices)

    if c == 0:
        return 0.0

    if c == 1:
        return 1.0

    def h(x: float) -> float:
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    return h((1 + np.sqrt(1 - c**2.0)) / 2)


def projector_entropy(rho: Tensor, indices: Sequence[IndexID]) -> float:
    """
    Calculate the entropy of the dominant eigenvector projector of a two qubit density matrix.

    Parameters
    ----------
    rho : Tensor
        The density matrix in Tensor representation. Must be normalized and Hermitian.
    indices : Sequence[IndexID]
        The IndexIDs representing the (ket) indices of the density matrix.

    Returns
    -------
    float
        Entropy of the dominant eigenvector of the system.

    """
    assert isinstance(indices, (list, tuple)) and all(
        isinstance(i, IndexID) for i in indices
    ), "indices must be a list or tuple of IndexIDs"
    assert len(indices) == 2, "projector_entropy requires exactly two indices"
    assert all(dim == 2 for dim in rho.tensor_ndarray.shape), (
        "All dimensions of rho.tensor_ndarray must be 2 (qubit indices)"
    )

    density_matrix = _matricize_rdm(rho, [str(i) for i in indices])

    assert np.isclose(np.trace(density_matrix), 1), (
        "Entanglement metrics require a normalized density matrix (trace ≈ 1)"
    )

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)

    # Get the dominant eigenvector (corresponding to the largest eigenvalue)
    dominant_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    # Form the projector onto this eigenvector
    projector = np.outer(dominant_eigenvector, dominant_eigenvector.conj())

    # Now we'll turn this into a Tensor to plug into the usual formula for QMI/entanglement entropy
    projector_tensor = Tensor(
        projector.reshape(2, 2, 2, 2),
        [str(i) + "*" for i in indices] + [str(i) for i in indices],
    )

    return mutual_information(projector_tensor, indices)
