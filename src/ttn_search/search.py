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
    # Scale the features to have zero mean and unit variance based on the training set,
    # and optionally apply PCA for dimensionality reduction if requested.
    scaler = StandardScaler().fit(training)
    x = np.asarray(scaler.transform(x))
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
    """
    Use the Gaussian Process model 
    to predict the mean and standard deviation of the target variable for unseen candidates.
    """

    # 1. Build the model

    # One scalar length scale controls complexity in this small-data pilot.
    gp = GaussianProcessRegressor(kernel=ConstantKernel(1., (0.01, 100.))*Matern(length_scale=3.,
        length_scale_bounds=(0.05, 100.), nu=2.5), alpha=1e-6, normalize_y=True,
        n_restarts_optimizer=0, random_state=0)

    # 2. Fit the model to the observed data and predict for unseen candidates.

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gp.fit(x[indices], y)

    # 3. Predict the mean and standard deviation for the unseen candidates using the fitted model.
    
    prediction = gp.predict(x[unseen], return_std=True, return_cov=False)
    # Library stubs can expose all predict() return variants. Validate the
    # mean/std pair explicitly so both the runtime and type checker agree.
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise ValueError('GP prediction with return_std=True must return (mean, std)')
    mean, std = prediction
    return mean, np.maximum(std, 1e-9), [str(w.message) for w in caught]


def propose(x, observed, unseen, threshold, rng):
    """
    The BO proposal function uses the Gaussian Process model
    ===========================================================
    Propose the next candidate to evaluate 
    based on the Gaussian Process model and acquisition function.
    The acquisition function balances exploration and exploitation, 
    considering both the predicted mean and uncertainty.
    """

    # 1. Extract the measured and successful candidates from the observed evaluations.

    # All the TTN candidates that have been evaluated and have a fidelity_final value.
    # Used to train model 1
    measured = [(i, r) for i, r in observed.items() if 'fidelity_final' in r]
    # All the TTN candidates that have been evaluated and were successful (status == 'success').
    # Used to train model 2
    costs = [(i, r) for i, r in measured if r['status'] == 'success']
    log = {'acquisition': 'random_fallback', 'warnings': []}

    # 2. If there are fewer than 2 measured candidates, 
    # fall back to random selection from the unseen candidates.

    if len(measured) < 2:
        return int(rng.choice(unseen)), log

    # 3. Use the Gaussian Process model to predict the mean and standard deviation
    
    try:
        # (1) Model 1: 
        # Predict the fidelity of unseen candidates based on the measured candidates.

        fi = [i for i, _ in measured]
        # Use the error to fit the GP model, as the acquisition function is based on the error.
        fy = np.log10(np.maximum([1-r['fidelity_final'] for _, r in measured], 1e-14))
        fm, fs, notes = gp_predict(x, fi, fy, unseen)
        # Calculate the probability of feasibility for each unseen candidate based on the predicted mean and standard deviation.
        probability = norm.cdf((np.log10(1-threshold)-fm)/fs)
        log['warnings'].extend(notes)

        # (2) Model 2:
        # Predict the resource cost (cx_count) of unseen candidates based on the successful candidates.

        if len(costs) >= 2:
            # If there are at least 2 successful candidates, 
            # use the GP model to predict the resource cost of unseen candidates.
            ci = [i for i, _ in costs]
            cy = np.log1p([r['cx_count'] for _, r in costs])
            cm, cs, notes = gp_predict(x, ci, cy, unseen)
            log['warnings'].extend(notes)
            delta = min(cy)-cm
            z = delta/cs
            score = (delta*norm.cdf(z)+cs*norm.pdf(z))*probability
            log['acquisition'] = 'constrained_ei'
        else:
            # If there are fewer than 2 successful candidates,
            # use the feasibility + uncertainty acquisition function to propose the next candidate.

            # The score is, based on the fidelity that reach the threshold, 
            # the cost of the candidate, and the uncertainty of the prediction.
            score = probability + 0.05*fs/(fs.max()+1e-12)
            log['acquisition'] = 'feasibility_uncertainty'

        # 4. Select the candidate with the highest acquisition score to evaluate next.
        
        choice = int(np.argmax(score))
        log.update(score=float(score[choice]), probability_feasible=float(probability[choice]))
        return int(unseen[choice]), log
    except (ValueError, np.linalg.LinAlgError) as exc:
        log['warnings'].append(str(exc))
        return int(rng.choice(unseen)), log


def run_search(pool, evaluator, method, budget, init, threshold, seed=0, on_step=None):
    """
    Run a constrained Bayesian optimization search over a finite pool of candidates.
    The search uses a Gaussian Process model to predict the performance of unseen candidates
    and proposes the next candidate to evaluate based on an acquisition function 
    that balances exploration and exploitation.
    The search continues until the budget of evaluations is exhausted or all candidates have been evaluated.
    ----
    Parameters:
    - pool: List of candidate trees to evaluate.
    - evaluator: Function that takes a candidate index and returns its evaluation record.
    - method: Search method to use ('random', 'raw', 'pca').
    - budget: Maximum number of evaluations to perform.
    - init: List of initial candidate indices to evaluate before using the model.
    - threshold: Fidelity threshold for considering a candidate successful.
    - seed: Random seed for reproducibility.
    - on_step: Optional callback function to call after each evaluation step with the current trace.
    ----
    Returns:
    - trace: List of dictionaries containing the evaluation results and metadata for each step.
    """

    # 1. Validate the input parameters

    if not 1 <= budget <= len(pool) or len(init) > budget or len(init) != len(set(init)):
        raise ValueError('Invalid budget or common initial observations')
    if any(i < 0 or i >= len(pool) for i in init):
        raise ValueError('Initial candidate outside pool')
    if method not in ('random', 'raw', 'pca'):
        raise ValueError('Unknown search method')

    # 2. Prepare the feature matrix for the pool if using a model-based method
    
    start = time.perf_counter()
    x = None if method == 'random' else prepare_features(pool, method)
    feature_time = time.perf_counter()-start
    rng = np.random.default_rng(seed)

    # 3. Initialize the search state: 

    observed, trace = {}, []
    evaluation_total, proposal_total, incumbent = 0., 0., None
    while len(observed) < budget:
        # Evaluate one candidate at a time, 
        # either from the common initial set or proposed by the model.
        step = len(observed)
        before = time.perf_counter()
        unseen = [i for i in range(len(pool)) if i not in observed]
        if step < len(init):
            # First evaluate the common initial candidates before using the model to propose new candidates.
            index, acquisition = init[step], {'acquisition': 'common_initial'}
        elif method == 'random':
            # The random method selects candidates uniformly at random from the unseen pool.
            index, acquisition = int(rng.choice(unseen)), {'acquisition': 'random'}
        else:
            # Use the Gaussian Process model 
            # to propose the next candidate based on the acquisition function.
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
 