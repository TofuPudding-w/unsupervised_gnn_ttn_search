from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import block_diag
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from ttn_search.compilers.verified import checked_multiplexer, compile_verified
from ttn_search.reference import build, circuit_state, fidelity
from ttn_search.targets import gaussian_target
from ttn_search.trees import random_tree


def assert_mux(gates):
    stats = {}
    circuit, diagonal = checked_multiplexer(gates, stats)
    # Independent full-operator verification, including relative/global phase.
    actual = diagonal[:, None] * Operator(circuit).data
    np.testing.assert_allclose(actual, np.asarray(block_diag(*gates)), atol=1e-8, rtol=0)
    return stats


def test_saved_failure_corrected_without_more_gates():
    with np.load(Path(__file__).parent/'fixtures'/'qiskit_uc_regression.npz') as data:
        gates = data['gates']
    stats = assert_mux(gates)
    circuit, _ = checked_multiplexer(gates)
    assert stats['diagonals_corrected'] == 1
    assert circuit.count_ops()['cx'] == 15


@pytest.mark.parametrize('controls', range(5))
def test_random_complex_and_repeated_control_branches(controls):
    rng = np.random.default_rng(90+controls)
    gates = []
    for _ in range(2**controls):
        u, _ = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        gates.append(u*np.exp(1j*rng.uniform(-np.pi,np.pi)))
    assert_mux(gates)
    assert_mux([gates[0]]*len(gates))
    if controls > 1:
        assert_mux(gates[:len(gates)//2]*2)


def test_nondiagonal_error_uses_verified_local_fallback(monkeypatch):
    from ttn_search.compilers import verified
    class Broken:
        def __init__(self, *args, **kwargs):
            pass
        def _dec_ucg(self):
            c = QuantumCircuit(1)
            c.unitary(np.array([[0,1],[1,0]]), [0])
            return c, np.ones(2)
        def _get_diagonal(self):
            return np.ones(2)
    monkeypatch.setattr(verified, 'UCGate', Broken)
    stats = assert_mux([np.eye(2)])
    assert stats['euler_fallbacks'] == 1


def test_euler_fallback_complex_phases():
    from ttn_search.compilers.verified import euler_multiplexer
    rng = np.random.default_rng(761)
    gates = [np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0] for _ in range(16)]
    circuit, diagonal, error = euler_multiplexer(gates)
    np.testing.assert_allclose(diagonal[:,None]*np.asarray(Operator(circuit).data), np.asarray(block_diag(*gates)), atol=1e-8, rtol=0)
    assert error < 1e-8


def test_historical_ttn_failure_now_passes():
    target = gaussian_target(10, 'hierarchical', 13)
    rep = build(target.state, random_tree(10, np.random.default_rng(2026)), .0003/9)
    stats = {}
    circuit = compile_verified(rep, 17, 1, stats=stats)
    assert stats['diagonals_corrected'] >= 1
    assert fidelity(rep.state, circuit_state(circuit)) > 1-1e-8
    assert fidelity(target.state, circuit_state(circuit)) >= .999
    assert set(circuit.count_ops()) <= {'u', 'cx'}


@pytest.mark.parametrize('n', [3, 4, 5])
def test_complex_state_register_order(n):
    rng = np.random.default_rng(n)
    state = rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    rep = build(state, random_tree(n, rng))
    assert fidelity(rep.state, circuit_state(compile_verified(rep))) > 1-1e-9
