from dataclasses import asdict
import numpy as np
import pytest
from qiskit.transpiler.exceptions import TranspilerError
from ttn_search.evaluate import Pipeline,evaluate
from ttn_search.screening import evaluator as fresh
from ttn_search.reference import padded_isometry
from ttn_search.targets import gaussian_target
from ttn_search.trees import balanced,comb


@pytest.mark.parametrize('family,seed,shape',[('hierarchical',0,'balanced'),('overlapping',4,'balanced'),('overlapping',0,'comb')])
def test_prepared_evaluation_matches_reference(tmp_path,monkeypatch,family,seed,shape):
    target=gaussian_target(8,family,seed)
    tree=(balanced if shape=='balanced' else comb)(list(range(8)))
    policy=Pipeline()
    original=evaluate(target,tree,policy,tmp_path/'old')
    prepared=fresh.prepare(target,tree,policy)
    builds=[];real=fresh.build
    def spy(*args,**kwargs):
        builds.append(args[2]);return real(*args,**kwargs)
    monkeypatch.setattr(fresh,'build',spy)
    result=fresh.evaluate_fresh(target,tree,policy,tmp_path/'fresh',prepared)
    assert result['status']==original['status']=='success'
    assert result['cx_count']==original['cx_count']
    assert result['fidelity_final']==pytest.approx(original['fidelity_final'],abs=1e-12)
    assert [a['tolerance_factor'] for a in result['attempts']]==[a['tolerance_factor'] for a in original['attempts']]
    assert len(builds)==len(result['attempts'])-1
    assert result['attempts'][0]['reused_initial'] and not result['cache_hit']


def test_retry_rebuilds_and_counts_actual_compiler_calls(tmp_path,monkeypatch):
    target=gaussian_target(8,'hierarchical',0);tree=balanced(list(range(8)));policy=Pipeline()
    prepared=fresh.prepare(target,tree,policy)
    calls=[]
    def fail(*args,**kwargs):
        calls.append(1);raise TranspilerError('forced synthesis failure')
    monkeypatch.setattr(fresh,'compile_reference',fail)
    result=fresh.evaluate_fresh(target,tree,policy,tmp_path,prepared)
    assert len(calls)==result['compiler_calls']==3
    assert [a['tolerance_factor'] for a in result['attempts']]==[1.,.1,0.]
    assert [a['reused_initial'] for a in result['attempts']]==[True,False,False]
    assert result['status']=='compiler_error' and 'fidelity_final' not in result
    assert result['incremental_build_s']==sum(a['time_build_s'] for a in result['attempts'][1:])


def test_prepared_identity_and_dimension_score(tmp_path):
    target=gaussian_target(8,'overlapping',4);tree=balanced(list(range(8)));policy=Pipeline()
    prepared=fresh.prepare(target,tree,policy)
    rep=prepared.representation;assert rep is not None
    total=0
    def visit(node):
        nonlocal total
        if node.children is None:return
        total+=int(np.prod(padded_isometry(node).shape))
        for child in node.children:visit(child)
    visit(rep.root)
    assert fresh.scores(prepared)==(rep.parameter_count,total)
    with pytest.raises(ValueError,match='does not match'):
        fresh.evaluate_fresh(target,comb(list(range(8))),policy,tmp_path,prepared)
