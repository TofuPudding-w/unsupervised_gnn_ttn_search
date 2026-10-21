"""Nested dense SVD and local isometry synthesis, with explicit endian mapping."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import Statevector
from .trees import Tree, leaves, canonical, validate


@dataclass
class Node:
    # subtree structure
    tree: Tree
    labels: list[int]
    basis: np.ndarray
    local: np.ndarray
    children: tuple['Node', 'Node'] | None

    @property
    def rank(self):
        return self.basis.shape[1]


@dataclass
class Representation:
    root: Node
    state: np.ndarray
    ranks: list[int]
    parameter_count: int
    isometry_error: float


def fidelity(a, b):
    return float(np.clip(abs(np.vdot(a, b))**2 / (np.vdot(a, a).real * np.vdot(b, b).real), 0, 1))


def build(state, tree: Tree, local_discarded_weight: float = 0.0) -> Representation:
    """
    Entrance point for building a TTN representation of a quantum state given a tree structure.
    The representation includes the root node, the flattened state, the ranks of each node, the
    """

    # 1. Initialize and validate inputs

    state = np.asarray(state, dtype=complex)
    n = int(np.log2(len(state)))
    if state.ndim != 1 or len(state) != 2**n or not np.isfinite(state).all() or np.linalg.norm(state) == 0:
        raise ValueError('Expected finite, nonzero power-of-two statevector')
    if local_discarded_weight < 0:
        raise ValueError('Negative truncation budget')
    tree = canonical(tree)
    validate(tree, n)
    state = state / np.linalg.norm(state)
    # Axis q in this tensor is physical qubit q, unlike NumPy's default C-order flattening.
    tensor = state.reshape([2]*n, order='F')

    # 2. Recursively build the TTN representation from the tree structure

    nodes = []
    def visit(t, root=False):
        # This is a back edge traversal of the tree, where we compute the local isometries and bases for each node.
        if isinstance(t, int):  # Leaf node
            node = Node(t, [t], np.eye(2, dtype=complex), np.eye(2, dtype=complex), None)
        else:   # Internal node with two children
            a, b = visit(t[0]), visit(t[1])
            labels = a.labels + b.labels
            other = [q for q in range(n) if q not in labels]
            unfolding = tensor.transpose(labels + other).reshape(2**len(labels), -1)
            product = np.kron(a.basis, b.basis)
            reduced = product.conj().T @ unfolding
            if root:
                coeff = reduced / np.linalg.norm(reduced)
            else:
                # SVD decomposition
                u, s, _ = np.linalg.svd(reduced, full_matrices=False)
                # Always drop numerical zero modes, then the requested absolute discarded weight.
                numerical_rank = max(1, int(np.count_nonzero(s > max(s[0]*1e-12, 1e-15))))
                rank = numerical_rank
                for k in range(1, numerical_rank + 1):
                    if np.sum(s[k:]**2) <= local_discarded_weight:
                        rank = k
                        break
                coeff = u[:, :rank]
            node = Node(t, labels, product @ coeff, coeff, (a, b))
        nodes.append(node)
        return node
    root = visit(tree, True)
    # Reconstruct using local tensors independently of the stored dense bases.
    def reconstruct(node):
        if node.children is None:
            return node.local
        return np.kron(*(reconstruct(c) for c in node.children)) @ node.local
    dense = reconstruct(root).reshape([2]*n)
    ordered = dense.transpose(np.argsort(root.labels)).flatten(order='F')
    error = max(float(np.linalg.norm(x.local.conj().T @ x.local - np.eye(x.local.shape[1]))) for x in nodes)
    return Representation(root, ordered, [x.rank for x in nodes if x is not root],
                          sum(x.local.size for x in nodes), error)


def nbits(rank):
    return (rank - 1).bit_length()


def padded_isometry(node: Node):
    if node.children is None:
        raise ValueError('Padding requires an internal node with two children')
    a, b = node.children
    da, db = 2**nbits(a.rank), 2**nbits(b.rank)
    rows = [i*db+j for i in range(a.rank) for j in range(b.rank)]
    mat = np.zeros((da*db, node.rank), dtype=complex)
    mat[rows] = node.local
    extra = 2**nbits(node.rank) - node.rank
    if extra:
        mat = np.column_stack((mat, null_space(mat.conj().T)[:, :extra]))
    return mat


def compile_reference(rep: Representation, seed=0, optimization_level=1):
    """
    Compile a TTN representation into a Qiskit quantum circuit, preserving the physical qubit ordering and using only the 'u' and 'cx' basis gates.
    The compilation process involves mapping the TTN structure to a sequence of isometries and ensuring
    """

    # 1. Initialize the circuit and prepare for isometry synthesis

    n = len(rep.root.labels)
    circuit = QuantumCircuit(n)     # Initialize a quantum circuit with n qubits, all starting in the |0> state.
    free = list(range(n))           # A free list of qubit indices available for isometry synthesis, initially containing all qubits.
    physical_to_logical = {}        # Record the mapping from physical qubit indices to logical qubit indices in the TTN representation.

    # 2. Core isometry synthesis: 
    # recursively emit isometries for each node in the TTN representation
    
    def emit(node, incoming):
        if node.children is None:       # Leaf node
            if len(incoming) != 1:
                raise ValueError('Physical leaf must own one wire')
            physical_to_logical[incoming[0]] = node.tree
            return
        # Internal node with two children
        a, b = node.children
        total = nbits(a.rank) + nbits(b.rank)
        need = total - len(incoming)
        if need < 0 or need > len(free):
            raise ValueError('Invalid isometry register allocation')
        wires = incoming + free[:need]
        del free[:need]
        # Add isometry to the circuit, using the padded isometry matrix for the current node.
        mat = padded_isometry(node)
        if total:
            circuit.append(Isometry(mat, 0, 0), wires)
        else:
            circuit.global_phase += float(np.angle(mat[0, 0]))
        # Recursively emit isometries for the child nodes, splitting the wires according to their ranks.
        right_bits = nbits(b.rank)
        emit(a, wires[right_bits:])
        emit(b, wires[:right_bits])

    # 3. Start the recursive emission from the root node, using all available qubits
    
    emit(rep.root, [])
    if free or set(physical_to_logical.values()) != set(range(n)):
        raise ValueError('Incomplete physical output mapping')

    # 4. Transpile the circuit to the desired basis gates and optimization level, preserving the physical qubit ordering.
    
    # All input wires are |0>; consistently relabel the whole circuit, without adding artificial SWAPs.
    logical = QuantumCircuit(n)
    logical.global_phase = circuit.global_phase
    for instruction in circuit.data:
        logical.append(instruction.operation,
            [physical_to_logical[circuit.find_bit(q).index] for q in instruction.qubits])
    return transpile(logical, basis_gates=['u', 'cx'], optimization_level=optimization_level,
                     seed_transpiler=seed)


def circuit_state(circuit):
    return np.asarray(Statevector.from_instruction(circuit).data)
