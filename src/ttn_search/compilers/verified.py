"""Verified diagonal corrections for Qiskit 2.6.0 isometry synthesis.

The upstream multiplexer can return incorrect relative signs in its diagonal.
We independently evaluate its 2x2 control branches and check D @ U = target.
Non-diagonal residuals use a verified local Euler fallback. The common path
avoids constructing the larger full unitary matrix. Private
Qiskit hooks are deliberately version-gated; no global monkeypatch is used.
"""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Isometry, UCGate, UCRYGate, UCRZGate, DiagonalGate
from qiskit.exceptions import QiskitError
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.quantum_info import Operator
from scipy.linalg import block_diag
from ..reference import nbits, padded_isometry

COMPILER_VERSION = 'verified-uc-diagonal-v1-qiskit-2.6.0'
CHECK_TOLERANCE = 1e-8


def euler_multiplexer(gates):
    """Independent local fallback using uniformly controlled Euler rotations."""
    count = len(gates)
    circuit = QuantumCircuit(count.bit_length())
    angles = np.array([OneQubitEulerDecomposer('ZYZ').angles_and_phase(g) for g in gates])
    if count == 1:
        circuit.unitary(gates[0], [0])
    else:
        circuit.append(UCRZGate(angles[:, 2].tolist()), range(circuit.num_qubits))
        circuit.append(UCRYGate(angles[:, 0].tolist()), range(circuit.num_qubits))
        circuit.append(UCRZGate(angles[:, 1].tolist()), range(circuit.num_qubits))
        circuit.append(DiagonalGate(np.exp(1j*angles[:, 3]).tolist()), range(1, circuit.num_qubits))
    # Rare fallback is independently checked as a full operator, including phase.
    error = float(np.max(np.abs(np.asarray(Operator(circuit).data)-np.asarray(block_diag(*gates)))))
    if error > CHECK_TOLERANCE:
        raise QiskitError(f'Euler multiplexer verification failed: {error:.3g}')
    return circuit, np.ones(2*count, dtype=complex), error


def checked_multiplexer(gates, stats=None):
    """Return frozen up-to-diagonal circuit and independently verified diagonal.

Controls are q[1:] in little-endian order; target is q[0]. Simulation costs
O(number_of_branches * emitted_gates), storing only one 2x2 matrix per branch.
No global-phase equivalence is allowed: the returned diagonal must be exact.
"""
    if qiskit.__version__ != '2.6.0':
        raise QiskitError('verified-uc-diagonal-v1 requires revalidation for this Qiskit version')
    expected = np.asarray(gates, dtype=complex)
    gate = UCGate(list(expected), up_to_diagonal=True)
    circuit, _ = gate._dec_ucg()
    # _get_diagonal expands simplified controls to the original register.
    original = np.asarray(gate._get_diagonal(), dtype=complex).reshape(-1, 2)
    count = len(expected)
    actual = np.broadcast_to(np.eye(2, dtype=complex), (count, 2, 2)).copy()
    branches = np.arange(count)
    for instruction in circuit.data:
        wires = [circuit.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == 'unitary' and wires == [0]:
            actual = np.asarray(instruction.operation.params[0]) @ actual
        elif instruction.operation.name == 'cx' and len(wires) == 2 and wires[1] == 0:
            selected = ((branches >> (wires[0]-1)) & 1).astype(bool)
            actual[selected] = actual[selected][:, ::-1, :]
        else:
            raise QiskitError('Unexpected multiplexer structure; backend adapter needs revalidation')
    actual *= np.exp(1j * float(circuit.global_phase))
    original_error = float(np.max(np.abs(original[:, :, None] * actual - expected)))
    diagonal = original
    corrected = original_error > CHECK_TOLERANCE
    if corrected:
        # Per-row least-squares phase, constrained to unit modulus.
        overlaps = np.sum(expected * actual.conj(), axis=2)
        diagonal = overlaps / np.maximum(np.abs(overlaps), 1e-30)
        # Preserve exact sign corrections when the observed discrepancy is ±1.
        ratio = diagonal / original
        signs = np.where(ratio.real >= 0, 1., -1.)
        if np.max(np.abs(ratio-signs)) < CHECK_TOLERANCE:
            diagonal = original * signs
    error = float(np.max(np.abs(diagonal[:, :, None] * actual - expected)))
    fallback = error > CHECK_TOLERANCE
    if fallback:
        circuit, flat_diagonal, error = euler_multiplexer(list(expected))
        diagonal = flat_diagonal.reshape(-1, 2)
    if stats is not None:
        stats['multiplexers_checked'] = stats.get('multiplexers_checked', 0) + 1
        stats['diagonals_corrected'] = stats.get('diagonals_corrected', 0) + int(corrected and not fallback)
        stats['euler_fallbacks'] = stats.get('euler_fallbacks', 0) + int(fallback)
        stats['max_verified_residual'] = max(stats.get('max_verified_residual', 0.), error)
        stats['max_original_residual'] = max(stats.get('max_original_residual', 0.), original_error)
    return circuit, diagonal.flatten()


class VerifiedIsometry(Isometry):
    """Same elimination algorithm with checked UCG and MCG append operations."""
    def __init__(self, matrix, stats=None):
        super().__init__(matrix, 0, 0)
        self.verification_stats = stats

    def _append_ucg_up_to_diagonal(self, circ, q, single_qubit_gates, control_labels, target_label):
        # Qiskit labels are MSB-first, while q is little-endian.
        n = nbits(self.iso_data.shape[0])
        controls = [q[n-1-label] for label in reversed(control_labels)]
        target = q[n-1-target_label]
        circuit, diagonal = checked_multiplexer(single_qubit_gates, self.verification_stats)
        circ.compose(circuit, qubits=[target]+controls, inplace=True)
        return diagonal

    def _append_mcg_up_to_diagonal(self, circ, q, gate, control_labels, target_label):
        # The frozen backend also represents MCG as a multiplexer. Idle dirty
        # ancillas remain untouched; no clean/dirty ancillas are requested here.
        gates = [np.eye(2, dtype=complex) for _ in range(2**len(control_labels))]
        gates[-1] = gate
        return self._append_ucg_up_to_diagonal(circ, q, gates, control_labels, target_label)


def compile_verified(rep, seed=0, optimization_level=1, *, stats=None):
    """Reference allocation with eagerly checked synthesis; explicit opt-in v1."""
    if qiskit.__version__ != '2.6.0':
        raise QiskitError('Unsupported Qiskit version for verified compiler v1')
    n = len(rep.root.labels)
    circuit = QuantumCircuit(n)
    free = list(range(n))
    mapping = {}

    def emit(node, incoming):
        if node.children is None:
            if len(incoming) != 1:
                raise ValueError('Physical leaf must own one wire')
            mapping[incoming[0]] = node.tree
            return
        a, b = node.children
        total = nbits(a.rank)+nbits(b.rank)
        need = total-len(incoming)
        if need < 0 or need > len(free):
            raise ValueError('Invalid isometry register allocation')
        wires = incoming+free[:need]
        del free[:need]
        matrix = padded_isometry(node)
        if total:
            # Eager definition prevents high-level synthesis from reconstructing
            # an ordinary upstream Isometry and bypassing the adapter.
            definition = VerifiedIsometry(matrix, stats).definition
            if definition is None:
                raise QiskitError('Missing verified isometry definition')
            circuit.compose(definition, qubits=wires, inplace=True)
        else:
            circuit.global_phase += float(np.angle(matrix[0, 0]))
        right = nbits(b.rank)
        emit(a, wires[right:])
        emit(b, wires[:right])

    emit(rep.root, [])
    if free or set(mapping.values()) != set(range(n)):
        raise ValueError('Incomplete physical output mapping')
    logical = QuantumCircuit(n)
    logical.global_phase = circuit.global_phase
    for instruction in circuit.data:
        logical.append(instruction.operation, [mapping[circuit.find_bit(q).index] for q in instruction.qubits])
    return transpile(logical, basis_gates=['u', 'cx'], optimization_level=optimization_level, seed_transpiler=seed)
