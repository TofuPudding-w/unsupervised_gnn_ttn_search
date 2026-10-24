"""Standalone reproducer for the installed backend; no historical compiler edits."""
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import block_diag
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCGate
from qiskit.quantum_info import Operator


def fixture_matrix():
    with np.load(Path(__file__).parent / 'fixtures' / 'qiskit_uc_regression.npz') as data:
        return data['gates']


def residual(actual, expected):
    phase = np.vdot(expected, actual)
    return np.linalg.norm(actual - expected * phase / abs(phase))


def test_dense_unitary_oracle_for_failed_multiplexer():
    gates = fixture_matrix()
    expected = block_diag(*gates)
    np.testing.assert_allclose(expected.conj().T @ expected, np.eye(32), atol=1e-12)
    circuit = QuantumCircuit(5)
    circuit.unitary(expected, range(5))
    compiled = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=1, seed_transpiler=17)
    assert residual(Operator(compiled).data, expected) < 1e-8


@pytest.mark.parametrize('simplify', [False, True])
@pytest.mark.xfail(reason='Known installed Qiskit UCGate decomposition defect; XPASS signals backend change', strict=True)
def test_upstream_multiplexer_regression(simplify):
    gates = fixture_matrix()
    gate = UCGate(list(gates), up_to_diagonal=True, mux_simp=simplify)
    actual = np.asarray(gate._get_diagonal())[:, None] * Operator(gate).data
    assert residual(actual, block_diag(*gates)) < 1e-8
