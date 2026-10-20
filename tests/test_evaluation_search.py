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
