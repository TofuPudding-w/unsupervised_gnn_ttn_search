"""Finite-pool constrained BO with shared initialization and explicit failed attempts."""
import time
import warnings
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from .trees import features, random_tree


def prepare_features(pool, kind, seed=12345, training_count=512):
    x = np.array([features(t) for t in pool])
    n = len(features(pool[0]))  # Only used to initialize; infer physical leaves below.
    from .trees import leaves
    n = len(leaves(pool[0]))
    rng = np.random.default_rng(seed)
    training = np.array([features(random_tree(n, rng)) for _ in range(training_count)])
    scaler = StandardScaler().fit(training)
    x = scaler.transform(x)
    if kind == 'pca':
        pca = PCA(n_components=min(8, x.shape[1])).fit(scaler.transform(training))
        x = pca.transform(x)
    return x


def initial_ids(pool_size, count, seed):
    if not 1 <= count <= pool_size:
        raise ValueError('Invalid initial count')
    rng = np.random.default_rng(seed)
    return [0] + rng.choice(np.arange(1, pool_size), count-1, replace=False).tolist()


def gp_predict(x, indices, y, unseen):
    # One scalar length scale controls complexity in this small-data pilot.
    gp = GaussianProcessRegressor(kernel=ConstantKernel(1., (0.01, 100.))*Matern(length_scale=3.,
        length_scale_bounds=(0.05, 100.), nu=2.5), alpha=1e-6, normalize_y=True,
        n_restarts_optimizer=0, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gp.fit(x[indices], y)
    mean, std = gp.predict(x[unseen], return_std=True)
    return mean, np.maximum(std, 1e-9), [str(w.message) for w in caught]


def propose(x, observed, unseen, threshold, rng):
    measured = [(i, r) for i, r in observed.items() if 'fidelity_final' in r]
    costs = [(i, r) for i, r in measured if r['status'] == 'success']
    log = {'acquisition': 'random_fallback', 'warnings': []}
    if len(measured) < 2:
        return int(rng.choice(unseen)), log
    try:
        fi = [i for i, _ in measured]
        fy = np.log10(np.maximum([1-r['fidelity_final'] for _, r in measured], 1e-14))
        fm, fs, notes = gp_predict(x, fi, fy, unseen)
        probability = norm.cdf((np.log10(1-threshold)-fm)/fs)
        log['warnings'].extend(notes)
        if len(costs) >= 2:
            ci = [i for i, _ in costs]
            cy = np.log1p([r['cx_count'] for _, r in costs])
            cm, cs, notes = gp_predict(x, ci, cy, unseen)
            log['warnings'].extend(notes)
            delta = min(cy)-cm
            z = delta/cs
            score = (delta*norm.cdf(z)+cs*norm.pdf(z))*probability
            log['acquisition'] = 'constrained_ei'
        else:
            score = probability + 0.05*fs/(fs.max()+1e-12)
            log['acquisition'] = 'feasibility_uncertainty'
        choice = int(np.argmax(score))
        log.update(score=float(score[choice]), probability_feasible=float(probability[choice]))
        return int(unseen[choice]), log
    except (ValueError, np.linalg.LinAlgError) as exc:
        log['warnings'].append(str(exc))
        return int(rng.choice(unseen)), log


def run_search(pool, evaluator, method, budget, init, threshold, seed=0, on_step=None):
    if not 1 <= budget <= len(pool) or len(init) > budget or len(init) != len(set(init)):
        raise ValueError('Invalid budget or common initial observations')
    if any(i < 0 or i >= len(pool) for i in init):
        raise ValueError('Initial candidate outside pool')
    if method not in ('random', 'raw', 'pca'):
        raise ValueError('Unknown search method')
    start = time.perf_counter()
    x = None if method == 'random' else prepare_features(pool, method)
    feature_time = time.perf_counter()-start
    rng = np.random.default_rng(seed)
    observed, trace = {}, []
    evaluation_total, proposal_total, incumbent = 0., 0., None
    while len(observed) < budget:
        step = len(observed)
        before = time.perf_counter()
        unseen = [i for i in range(len(pool)) if i not in observed]
        if step < len(init):
            index, acquisition = init[step], {'acquisition': 'common_initial'}
        elif method == 'random':
            index, acquisition = int(rng.choice(unseen)), {'acquisition': 'random'}
        else:
            index, acquisition = propose(x, observed, unseen, threshold, rng)
        proposal_total += time.perf_counter()-before
        record = evaluator(index)
        observed[index] = record
        evaluation_total += record['evaluation_time_s']
        if record['status'] == 'success' and record['fidelity_final'] >= threshold:
            if incumbent is None or record['cx_count'] < observed[incumbent]['cx_count']:
                incumbent = index
        row = dict(step=step+1, candidate_id=index, status=record['status'],
            evaluation_key=record.get('evaluation_key'), cache_hit=record.get('cache_hit', False),
            best_candidate_id=incumbent, best_cx=None if incumbent is None else observed[incumbent]['cx_count'],
            charged_evaluation_time_s=evaluation_total, proposal_time_s=proposal_total,
            feature_time_s=feature_time, replay_estimated_online_time_s=evaluation_total+proposal_total+feature_time,
            elapsed_actual_s=time.perf_counter()-start, **acquisition)
        trace.append(row)
        if on_step:
            on_step(trace)
    return trace
