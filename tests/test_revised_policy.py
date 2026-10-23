import importlib.util
from pathlib import Path
from ttn_search.targets import gaussian_target
from ttn_search.trees import balanced
from ttn_search.screening import evaluator

spec=importlib.util.spec_from_file_location('revised_policy',Path('scripts/revised_policy.py'))
assert spec is not None and spec.loader is not None
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

def test_three_first_and_strict_budgets(tmp_path):
    target=gaussian_target(4,'overlapping',0)
    r=policy.evaluate(target,balanced(list(range(4))),tmp_path)
    assert r['attempts'][0]['tolerance_factor']==3
    assert r['fidelity_final']>=.999
    assert 1-r['fidelity_repr']<=.0001+1e-12
    assert len((tmp_path/'synthesis_starts.jsonl').read_text().splitlines())==r['compiler_calls']

def test_representation_failure_falls_back_and_charges_build(monkeypatch,tmp_path):
    original=evaluator.prepare
    def prepare(target,tree,pipeline,factor):
        item=original(target,tree,pipeline,factor)
        if factor==3:item.attempt['status']='representation_infeasible'
        return item
    monkeypatch.setattr(evaluator,'prepare',prepare)
    r=policy.evaluate(gaussian_target(4,'overlapping',0),balanced(list(range(4))),tmp_path)
    assert [a['tolerance_factor'] for a in r['attempts']]==[3,1]
    assert not r['attempts'][0]['compile_attempted']
    assert r['compiler_calls']==1
    assert r['incremental_build_s']==sum(a['time_build_s'] for a in r['attempts'])
