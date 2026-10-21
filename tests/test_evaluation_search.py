import numpy as np
from ttn_search.evaluate import Pipeline, evaluate
from ttn_search.targets import gaussian_target
from ttn_search.trees import candidate_pool, canonical, tree_hash
from ttn_search.search import initial_ids, run_search
from ttn_search.scent import scent_tree, affinity_matrix


def test_cache_separates_target_and_pipeline(tmp_path):
    tree = canonical(((0, 1), (2, 3)))
    target = gaussian_target(4, seed=0)
    a = evaluate(target, tree, Pipeline(), tmp_path)
    b = evaluate(target, tree, Pipeline(), tmp_path)
    c = evaluate(gaussian_target(4, seed=1), tree, Pipeline(), tmp_path)
    d = evaluate(target, tree, Pipeline(compiler_seed=99), tmp_path)
    assert a['status'] == 'success'
    assert not a['cache_hit'] and b['cache_hit']
    assert len({a['evaluation_key'], c['evaluation_key'], d['evaluation_key']}) == 3
    assert a['fidelity_comp'] > 1-1e-9
    assert a['fidelity_final'] >= 0.999


def test_search_common_initial_budget_failures_and_no_duplicates():
    pool = candidate_pool(4, 12, 0, canonical(((0, 1), (2, 3))))
    init = initial_ids(12, 3, 4)
    assert init[0] == 0
    for method in ('random', 'raw', 'pca'):
        calls = []
        def evaluator(i):
            calls.append(i)
            if i % 3 == 0:
                return {'status': 'numerical_error', 'evaluation_time_s': 1.}
            return {'status': 'success', 'fidelity_final': 0.99999, 'cx_count': 20-i, 'evaluation_time_s': 2.}
        trace = run_search(pool, evaluator, method, 7, init, 0.999, 4)
        assert calls[:3] == init
        assert len(calls) == len(set(calls)) == 7
        assert trace[-1]['charged_evaluation_time_s'] == sum(1 if i%3==0 else 2 for i in calls)
        assert trace[-1]['best_cx'] == min(20-i for i in calls if i%3)
    trace = run_search(pool, lambda i: {'status':'timeout','evaluation_time_s':1.}, 'raw', 5, init, 0.999)
    assert trace[-1]['best_cx'] is None


def test_upstream_scent_reproducible_and_rng_isolation():
    target = gaussian_target(4)
    before = np.random.get_state()
    a, info = scent_tree(target.state, 6)
    after = np.random.get_state()
    b, _ = scent_tree(target.state, 6)
    assert tree_hash(a) == tree_hash(b)
    assert np.array_equal(before[1], after[1])
    assert info['variant'] == 'SCENT_exact_state_mutual_information'
    affinity = affinity_matrix(target.state)
    np.testing.assert_allclose(affinity, affinity.T)
    assert np.min(affinity) >= 0


def test_qiskit_compiler_failure_retries_and_records_cost(monkeypatch, tmp_path):
    import importlib
    from qiskit.transpiler.exceptions import TranspilerError
    module = importlib.import_module('ttn_search.evaluate')
    original = module.compile_reference
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TranspilerError('HighLevelSynthesis is unable to synthesize "isometry"')
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'compile_reference', fail_once)
    record = evaluate(gaussian_target(4), canonical(((0, 1), (2, 3))), Pipeline(), tmp_path)
    assert record['status'] == 'success'
    assert len(calls) == 2
    failure = record['attempts'][0]
    assert failure['status'] == 'compiler_error'
    assert failure['error_type'] == 'TranspilerError'
    assert failure['failed_phase'] == 'compile'
    assert failure['time_compile_s'] > 0
    assert 'HighLevelSynthesis' in failure['traceback']
    assert 'fidelity_final' not in failure


def test_exhausted_compiler_failures_do_not_abort_search(monkeypatch, tmp_path):
    import importlib
    from qiskit.transpiler.exceptions import TranspilerError
    module = importlib.import_module('ttn_search.evaluate')

    def always_fail(*args, **kwargs):
        raise TranspilerError('synthesis failed')

    monkeypatch.setattr(module, 'compile_reference', always_fail)
    target = gaussian_target(4)
    pool = candidate_pool(4, 3, 0, canonical(((0, 1), (2, 3))))
    def oracle(i):
        return evaluate(target, pool[i], Pipeline(), tmp_path)
    trace = run_search(pool, oracle, 'raw', 3, [0], 0.999)
    assert len(trace) == 3
    assert all(step['status'] == 'compiler_error' for step in trace)
    assert trace[-1]['best_cx'] is None
    record = oracle(0)
    assert record['cache_hit']
    assert len(record['attempts']) == 3
    assert 'fidelity_final' not in record and 'cx_count' not in record


def test_pilot_hierarchical_seed1_candidate79(tmp_path):
    from threadpoolctl import threadpool_limits
    # Captured from the interrupted pilot; does not depend on generated artifacts.
    tree = canonical(((5, (4, (1, 7))), (6, (2, (0, 3)))))
    with threadpool_limits(limits=1):
        record = evaluate(gaussian_target(8, 'hierarchical', 1), tree, Pipeline(), tmp_path)
    assert record['status'] == 'success'
    assert record['fidelity_final'] >= 0.999
    assert record['fidelity_comp'] >= 1 - 0.0004
    assert all(a.get('fidelity_final', 0) < 0.999 or a['status'] != 'success'
               for a in record['attempts'][:-1])
