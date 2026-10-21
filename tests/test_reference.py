import numpy as np
import pytest
from ttn_search.trees import canonical, tree_hash, features, random_tree, validate, neighbors
from ttn_search.reference import build, compile_reference, circuit_state, fidelity, padded_isometry


def test_tree_identity_and_labels():
    a = canonical(((0, 1), (2, 3)))
    assert tree_hash(a) == tree_hash(((3, 2), (1, 0)))
    assert tree_hash(a) != tree_hash(((0, 2), (1, 3)))
    assert not np.allclose(features(a), features(canonical(((0, 2), (1, 3)))))
    with pytest.raises(ValueError):
        validate(((0, 1), (1, 3)), 4)
    for t in neighbors(a):
        validate(t, 4)


@pytest.mark.parametrize('n', [3, 4, 5])
def test_complex_exact_reconstruction_and_circuit(n):
    rng = np.random.default_rng(n)
    state = rng.normal(size=2**n) + 1j*rng.normal(size=2**n)
    state /= np.linalg.norm(state)
    for _ in range(2):
        tree = random_tree(n, rng)
        rep = build(state, tree)
        assert rep.isometry_error < 1e-10
        assert fidelity(state, rep.state) > 1 - 1e-10
        circuit = compile_reference(rep)
        assert set(circuit.count_ops()) <= {'u', 'cx'}
        assert fidelity(state, circuit_state(circuit)) > 1 - 1e-9


def test_rank_three_padding_and_physical_order():
    state = np.zeros(16, complex)
    state[[0, 5, 10]] = np.sqrt([0.5, 0.3, 0.2])
    rep = build(state, canonical(((0, 1), (2, 3))))
    assert 3 in rep.ranks
    assert rep.root.children is not None
    for node in rep.root.children:
        mat = padded_isometry(node)
        np.testing.assert_allclose(mat.conj().T @ mat, np.eye(mat.shape[1]), atol=1e-12)
    assert fidelity(state, circuit_state(compile_reference(rep))) > 1 - 1e-10


def test_product_and_truncated_states():
    rng = np.random.default_rng(22)
    product = np.zeros(32, complex)
    product[19] = 1j
    rep = build(product, random_tree(5, rng))
    assert fidelity(product, circuit_state(compile_reference(rep))) > 1 - 1e-10
    state = np.zeros(16, complex)
    state[[0, 5, 10]] = np.sqrt([0.6, 0.39999, 0.00001])
    rep = build(state, canonical(((0, 1), (2, 3))), 2e-5)
    assert 0.9999 < fidelity(state, rep.state) < 1
    assert fidelity(rep.state, circuit_state(compile_reference(rep))) > 1 - 1e-10
