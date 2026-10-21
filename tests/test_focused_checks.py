"""Budget and information-boundary checks for the fixed neighborhood baseline."""
import importlib.util
from pathlib import Path
from ttn_search.trees import candidate_pool, canonical, neighbors, tree_hash


def test_neighborhood_priority_shared_initialization_and_failure_budget():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_focused_checks.py'
    spec = importlib.util.spec_from_file_location('focused_checks', script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pool = candidate_pool(4, 12, 4, canonical(((0, 1), (2, 3))))
    near = {tree_hash(t) for t in neighbors(pool[0])}
    neighbor_ids = {i for i,t in enumerate(pool) if tree_hash(t) in near}
    calls = []
    def oracle(i):
        calls.append(i)
        return dict(evaluation_key=str(i), evaluation_time_s=1., status='compiler_error')
    init = [0, 11]
    trace = module.neighborhood_search(pool, oracle, 12, init, 0)
    assert calls[:2] == init
    assert len(set(calls)) == len(calls) == 12
    assert trace[-1]['best_cx'] is None
    assert trace[-1]['charged_evaluation_time_s'] == 12
    for step in range(2, 12):
        remaining = neighbor_ids-set(calls[:step])
        if remaining:
            assert calls[step] in remaining
            assert trace[step]['acquisition'] == 'SCENT_neighbor'
