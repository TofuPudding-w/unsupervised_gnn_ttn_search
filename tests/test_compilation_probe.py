import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from ttn_search.reference import build,circuit_state,fidelity
from ttn_search.targets import gaussian_target
from ttn_search.trees import balanced,comb

sys.path.insert(0,str(Path('scripts').resolve()))
spec=importlib.util.spec_from_file_location('compilation_probe',Path('scripts/compilation_probe.py'))
assert spec is not None and spec.loader is not None
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

@pytest.mark.parametrize('backend',['reference','unitary_qsd'])
@pytest.mark.parametrize('tree',[balanced([0,1,2,3]),comb([2,0,3,1])])
def test_alternative_and_block_mapping_preserve_complex_state(backend,tree):
    rng=np.random.default_rng(19)
    state=rng.normal(size=16)+1j*rng.normal(size=16)
    rep=build(state,tree,1e-6)
    circuit,blocks=probe.compile_blocks(rep,backend)
    assert fidelity(rep.state,circuit_state(circuit))>1-1e-10
    assert sum(b['critical_chain_cx'] for b in blocks)==probe.two_qubit_depth(circuit)
