import numpy as np
import pytest
from ttn_search.reference import build,fidelity
from ttn_search.screening.shared_svd import SharedSVD
from ttn_search.trees import canonical,random_tree,neighbors


@pytest.mark.parametrize('tolerance',[0.,1e-3])
def test_shared_subtrees_match_independent_complex_svd(tolerance):
    rng=np.random.default_rng(129)
    state=rng.normal(size=64)+1j*rng.normal(size=64)
    tree=canonical((((0,1),2),((3,4),5)))
    trees=[tree,*neighbors(tree),random_tree(6,rng)]
    cache=SharedSVD(state,tolerance)
    for t in trees:
        expected=build(state,t,tolerance);actual=cache.build(t)
        assert actual.ranks==expected.ranks
        assert actual.parameter_count==expected.parameter_count
        assert fidelity(actual.state,expected.state)>1-1e-12
        np.testing.assert_allclose(actual.state,expected.state,atol=1e-12)
    assert cache.svd_calls<len(trees)*4
    calls=cache.svd_calls
    cache.build(tree)
    assert cache.svd_calls==calls
    assert cache.retained_array_bytes>state.nbytes


def test_context_owns_state_and_separates_truncation():
    state=np.zeros(16,complex);state[[0,5,10]]=np.sqrt([.6,.39999,.00001])
    saved=state.copy();tree=canonical(((0,1),(2,3)))
    cache=SharedSVD(state,2e-5);state[:]=0
    actual=cache.build(tree);expected=build(saved,tree,2e-5)
    np.testing.assert_allclose(actual.state,expected.state,atol=1e-12)
    assert actual.ranks!=SharedSVD(saved,0).build(tree).ranks
    with pytest.raises(ValueError):actual.root.local[0,0]=0
    with pytest.raises(ValueError):cache.build(((0,1),(1,3)))
