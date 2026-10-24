import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from ttn_search.suitability.design import generated_target,apply_zz,candidate_sets
from ttn_search.suitability.diagnostics import profile_build,describe,FEATURE_SETS
from ttn_search.reference import build,fidelity
from ttn_search.trees import balanced,leaves
from ttn_search.scent import affinity_matrix


@pytest.mark.parametrize('seed',[101,102])
@pytest.mark.parametrize('alpha',[.3,1.2])
def test_generator_reproducibility_cut_ranks_and_zz(seed,alpha):
    target,tree=generated_target(seed,alpha,0.)
    assert abs(np.linalg.norm(target.state)-1)<1e-12
    assert np.array_equal(target.state,generated_target(seed,alpha,0.)[0].state)
    def visit(t):
        if isinstance(t,int):return
        ids=leaves(t)
        if len(ids)<10:
            other=[i for i in range(10) if i not in ids]
            mat=target.state.reshape([2]*10,order='F').transpose(ids+other).reshape(2**len(ids),-1)
            assert np.linalg.matrix_rank(mat,tol=1e-10)<=min(4,2**min(len(ids),10-len(ids)))
        visit(t[0]);visit(t[1])
    visit(tree)
    coupled,_=generated_target(seed,alpha,.15)
    circuit=QuantumCircuit(10)
    for i,j in target.metadata['zz_pairs']:circuit.rzz(.15,i,j)
    actual=Statevector(target.state).evolve(circuit).data
    np.testing.assert_allclose(actual,coupled.state,atol=1e-12)
    np.testing.assert_allclose(apply_zz(coupled.state,target.metadata['zz_pairs'],-.15),target.state,atol=1e-12)


def test_candidate_quotas_and_shared_random_prefix():
    scent=balanced(list(range(10)));p=candidate_sets(scent,12)
    assert len(p['R'])==len(set(p['R']))==64
    assert len(p['M'])==len(set(p['M']))==64
    assert p['M'][:32]==p['R'][:32]
    assert len(p['perturbation_distance'])==32
    assert all(k not in p['R'][:32] for k in p['perturbation_distance'])
    assert set(p['perturbation_distance'].values())<={1,2}
    assert p==candidate_sets(scent,12)


def test_profile_preserves_representation_and_frozen_features():
    target,tree=generated_target(101,.3,.15)
    actual,diagnostic=profile_build(target.state,tree,.0003/9)
    expected=build(target.state,tree,.0003/9)
    np.testing.assert_allclose(actual.state,expected.state,atol=1e-12)
    assert actual.ranks==expected.ranks
    desc=describe(target.state,actual,affinity_matrix(target.state),diagnostic)
    assert all(k in desc['features'] for keys in FEATURE_SETS for k in keys)
    assert diagnostic['timing']['total']>=diagnostic['timing']['svd']
    assert fidelity(target.state,actual.state)>=.9999
