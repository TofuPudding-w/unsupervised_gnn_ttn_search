"""Exact small-state MI affinity + pinned upstream SCENT recursive partitioning.

This is an explicitly identified SCENT affinity variant, not the paper's
projector-entropy Monte Carlo default. No TCI or compiler is invoked here.
"""
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from .trees import canonical, validate

REVISION = 'f05c28fbf959ecc023fd4ed89d1e00691d5ca564'


def reduced_density(state, labels):
    n = int(np.log2(len(state)))
    others = [q for q in range(n) if q not in labels]
    matrix = state.reshape([2]*n, order='F').transpose(labels + others).reshape(2**len(labels), -1)
    return matrix @ matrix.conj().T


def entropy(rho):
    values = np.clip(np.linalg.eigvalsh(rho), 0, 1)
    values = values[values > 1e-15]
    return float(-np.sum(values * np.log2(values)))


def affinity_matrix(state):
    """
    Compute the pairwise affinity matrix for a quantum state based on mutual information.
    The affinity between two qubits is defined as the maximum of zero and the sum of their
    I(i:j) = S(i) + S(j) - S(ij), 
    where S is the von Neumann entropy of the reduced density matrix.
    """
    state = state / np.linalg.norm(state)
    n = int(np.log2(len(state)))
    singles = [entropy(reduced_density(state, [i])) for i in range(n)]
    affinity = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            affinity[i, j] = affinity[j, i] = max(0., singles[i] + singles[j] - entropy(reduced_density(state, [i, j])))
    return affinity


def scent_tree(state, seed=0):
    """
    Generate a tree structure using the SCENT algorithm based on the provided quantum state.
    The tree is constructed by recursively partitioning the variables based on their mutual information,
    and the resulting tree structure is returned along with metadata about the process.
    """

    # 1. Environment preparation: 
    # ensure the pinned SCENT vendor source is available and import necessary modules.
    
    start = time.perf_counter()
    path = Path(__file__).resolve().parents[2] / 'vendor' / 'treetensorcomp'
    if not path.exists():
        raise RuntimeError('Pinned SCENT vendor source is missing')
    sys.path.insert(0, str(path)) if str(path) not in sys.path else None
    import networkx as nx
    from treetensorcomp.quantics.index_settings import IndexID
    from treetensorcomp.structural_optimization.clustering import _subdivide_cluster
    
    # 2. Calculate the affinity matrix

    n = int(np.log2(len(state)))    # The length of the state is 2^n, so log2(len(state)) is the number of qubits.
    affinity = affinity_matrix(state)
    # Give a exclusive id to each qubit for SCENT's internal processing.
    ids = [IndexID('q', i + 1) for i in range(n)]
    # Establish the mapping from SCENT's internal ids to the original qubit indices for later reconstruction.
    mapping = {str(index): i for i, index in enumerate(ids)}
    graph = nx.Graph()

    # 3. Implement the binary hierarchical clustering using SCENT's recursive partitioning.

    # Upstream uses sklearn random_state=None. Isolate and restore its global RNG.
    rng_state = np.random.get_state()   # Store the current state of the global random number generator to restore it later.
    try:
        np.random.seed(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _subdivide_cluster(graph, ids, affinity, 2)
    finally:
        np.random.set_state(rng_state)  # Restore the global random number generator state to its original state to avoid affecting other parts of the program.
    
    # 4. Convert the generated graph into a canonical tree structure
    
    # Preserve the recursive root BEFORE upstream polish_graph chooses a gauge center.
    root = next(iter(graph.nodes))
    def convert(node, parent=None):
        index = graph.nodes[node].get('external_index')
        if index is not None:
            return mapping[str(index)]
        children = [x for x in graph.neighbors(node) if x != parent]
        if len(children) != 2:
            raise ValueError('SCENT did not return a binary hierarchy')
        return canonical(tuple(convert(x, node) for x in children))
    tree = convert(root)
    validate(tree, n)
    return tree, dict(variant='SCENT_exact_state_mutual_information', revision=REVISION,
        seed=seed, time_s=time.perf_counter()-start, affinity=affinity.tolist(),
        warnings=[str(w.message) for w in caught])
