import importlib.util
from pathlib import Path
from ttn_search.reference import padded_isometry
from ttn_search.targets import gaussian_target
from ttn_search.trees import balanced,comb

spec=importlib.util.spec_from_file_location('decision_common',Path('scripts/decision_common.py'))
assert spec is not None and spec.loader is not None
common=importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def test_dimensions_match_actual_padded_maps():
    target=gaussian_target(8,'overlapping',4)
    pipeline={'fidelity':.999,'repr_fraction':.1}
    for tree in [balanced(list(range(8))),comb(list(range(8)))]:
        rep,info=common.construct(target,tree,pipeline)
        shapes=[padded_isometry(node).shape for node in common.nodes(rep)]
        assert info['features'][6]==sum(r*c for r,c in shapes)
        assert info['features'][7]==max(r*c for r,c in shapes)
        assert info['features'][2]==rep.parameter_count
        assert len(info['features'])==8
        assert info['construction_s']>=0


def test_tolerance_scaling_preserves_global_configuration():
    target=gaussian_target(8,'overlapping',4)
    pipeline={'fidelity':.999,'repr_fraction':.1}
    tree=balanced(list(range(8)))
    _,a=common.construct(target,tree,pipeline)
    _,b=common.construct(target,tree,pipeline,.8)
    assert b['local_tolerance']==.8*a['local_tolerance']
    assert pipeline=={'fidelity':.999,'repr_fraction':.1}
