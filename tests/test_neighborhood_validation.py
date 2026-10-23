"""Search semantics tested with inexpensive synthetic labels."""
import importlib.util
from pathlib import Path
from ttn_search.trees import balanced, neighbors, tree_hash, canonical

spec = importlib.util.spec_from_file_location('neighborhood_runner',Path('scripts/run_neighborhood_validation.py'))
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_shared_start_unique_budget_and_initial_neighborhood():
    tree = balanced(list(range(4)))
    adjacent = {tree_hash(t) for t in neighbors(tree)}
    def oracle(t):
        return dict(status='success',fidelity_final=1.,cx_count=100,
                    evaluation_key=tree_hash(t),evaluation_time_s=1.,cache_hit=False)
    for method in ['random','initial_neighborhood','iterative_neighborhood']:
        base,trace = runner.search(tree,oracle,method,0,8,n=4)
        assert len(trace)==8 and len({r['tree_hash'] for r in trace})==8
        assert tree_hash(tree) not in {r['tree_hash'] for r in trace}
        assert all(r['best_cx']==100 and not r['improved'] for r in trace)
        if method != 'random':
            assert {r['tree_hash'] for r in trace[:len(adjacent)]}==adjacent


def test_iterative_moves_and_failed_evaluation_counts():
    tree = balanced(list(range(4)))
    calls = []
    def oracle(t):
        calls.append(t)
        failed = len(calls)==3
        return dict(status='infeasible' if failed else 'success',fidelity_final=.9 if failed else 1.,
                    cx_count=100-len(calls),evaluation_key=tree_hash(t),evaluation_time_s=1.,cache_hit=False)
    _,trace=runner.search(tree,oracle,'iterative_neighborhood',0,8,n=4)
    assert len(calls)==9
    assert trace[0]['improved']
    assert canonical(trace[1]['anchor'])==canonical(trace[0]['tree'])
    assert not trace[1]['improved'] and trace[1]['best_cx']==trace[0]['best_cx']
    assert canonical(trace[2]['anchor'])==canonical(trace[0]['tree'])
