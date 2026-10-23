"""Small, preregistered Stage B development experiment. Run from project root."""
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ttn_search.evaluate import Pipeline, digest, evaluate, software, source_hash, write_json
from ttn_search.learning.encoder import Encoder, batch, corpus, embed, loss, train
from ttn_search.search import gp_predict, initial_ids, propose
from ttn_search.targets import gaussian_target
from ttn_search.trees import canonical, features, neighbors, tree_hash

OUT = Path('artifacts/stage_b')
OLD = Path('artifacts/stage_a_pilot_v2')


def replay(pool, records, x, method, seed, budget=64):
    """
    Only evaluated records enter the surrogate, 
    even though labels exist offline.
    ----
    Parameters:
    - pool: list of candidate trees
    - records: list of evaluation records corresponding to the pool
    - x: feature representation of the pool (can be None for random methods)
    - method: acquisition method ('random', 'neighborhood', or surrogate-based)
    - seed: random seed for reproducibility
    - budget: number of candidates to evaluate (default is 64)
    ----
    Returns:
    - trace: list of dictionaries containing the evaluation trace for each step
    """

    # 1. Initialize 8 random trees for reproducibility

    init = initial_ids(len(pool), 8, seed)
    adjacent = {tree_hash(t) for t in neighbors(pool[0])}
    rng = np.random.default_rng(seed)
    observed = {}
    trace = []
    best = None
    proposal_time = charged = 0.

    # 2. Iterate through the budget
    # For each step, propose a not-selected candidate tree based on the chosen method

    for step in range(budget):
        start = time.perf_counter()
        unseen = [i for i in range(len(pool)) if i not in observed]
        if step < len(init):
            i, log = init[step], {'acquisition':'common_initial'}
        elif method in ('random', 'neighborhood'):
            # Neighborhood: select a candidate tree from the unseen set that is adjacent to the SCENT tree
            preferred = [i for i in unseen if tree_hash(pool[i]) in adjacent] if method == 'neighborhood' else []
            # Random: randomly select a candidate tree from the unseen set
            i, log = int(rng.choice(preferred or unseen)), {'acquisition':method}
        else:
            # Surrogate: use the Gaussian Process surrogate model
            i, log = propose(x, observed, unseen, .999, rng)
        proposal_time += time.perf_counter()-start

        # 3. Record the evaluation results for the proposed candidate tree

        r = records[i]
        observed[i] = r
        charged += r['evaluation_time_s']
        if r['status'] == 'success' and r['fidelity_final'] >= .999:
            # Record the best CNOT count
            best = r['cx_count'] if best is None else min(best, r['cx_count'])
        trace.append(dict(step=step+1, candidate_id=i, best_cx=best, status=r['status'],
                          evaluation_key=r['evaluation_key'], proposal_time_s=proposal_time,
                          charged_evaluation_time_s=charged, **log))
    return trace


def main():

    # 1. Environment setup: 
    # limit the number of threads for reproducibility and performance

    torch.set_num_threads(1)
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    # Read the historical manifest to ensure that the evaluator has not changed since the last run
    historical = json.loads((OLD/'manifest.json').read_text())
    assert historical['source_hash'] == source_hash() and historical['software'] == software(), 'Evaluator changed'
    pools = {}
    excluded = set()

    # 2. Build the new pools for each target family and seed combination
    for family in ('hierarchical','overlapping'):
        for seed in range(3):
            label = f'{family}_{seed}'
            # Read all the trees from the old pool 
            # and exclude them from the new pool to ensure uniqueness
            full = [canonical(json.loads(t)) for t in json.loads((OLD/label/'pool.json').read_text())['trees']]
            excluded.update(tree_hash(t) for t in full)
            # Each new pool consists of one SCENT tree + 127 randomly sampled trees from the original pool
            ids = [0]+np.random.default_rng(20261021).choice(np.arange(1,len(full)),127,replace=False).tolist()
            pools[label] = [full[i] for i in ids]
            # Write the new pool to the output directory
            write_json(OUT/label/'pool.json', dict(original_ids=ids, trees=pools[label]))
    files = [Path(__file__), *sorted(Path('src/ttn_search/learning').glob('*.py'))]
    protocol = dict(evaluator_source_hash=source_hash(), software=software(), torch=torch.__version__,
        learning_source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest(),
        corpus_seed=20261021, training_count=2000, validation_count=250, test_count=250,
        training_seeds=[0,1,2], epochs=100, batch_size=128, learning_rate=.001,
        pool_size=128, pool_selection='SCENT plus 127 uniformly sampled original pool indices; no cost-based selection',
        search_seeds=[0,1,2], initial_count=8, budgets=[16,32,64],
        prediction_train_sizes=[8,16], prediction_split_seeds=list(range(5)),
        feature_scaling='fit only on the same 2000 training trees for raw, PCA, and each encoder',
        caveats=['Existing six development targets, not independent confirmation',
                 'All labels collected offline; search reveals only chosen observations',
                 'New 128-candidate pools: do not directly compare endpoints with Stage A',
                 'Reconstruction checkpoint selected only by validation loss',
                 'Random tree generator is pair-merging, not uniform over tree topologies'])
    write_json(OUT/'protocol.json', protocol)

    # 3. Model training

    # Generate a corpus of 2500 random trees, 
    # excluding those already in the training or validation sets
    trees = corpus(2500, excluded)
    training, validation, testing = trees[:2000], trees[2000:2250], trees[2250:]
    write_json(OUT/'corpus.json', dict(train=training, validation=validation, test=testing))
    # Generate the raw feature representation of the training set.
    raw = np.array([features(t) for t in training])
    raw_scaler = StandardScaler().fit(raw)
    # Apply PCA to reduce the dimensionality of the raw features to 8 components.
    pca = PCA(n_components=8).fit(raw_scaler.transform(raw))

    models = {}; 
    scalers = {}; 
    training_results = []
    torch.manual_seed(0)
    # Save an untrained model for contrast
    models['untrained'] = Encoder()

    # Use 3 seeds to train 3 separate encoder models
    for seed in range(3):
        # Train the encoder model on the training set, 
        # validate on the validation set, and save the model state.
        model, result = train(training, validation, seed)
        models[f'gnn_{seed}'] = model
        torch.save(model.state_dict(), OUT/f'gnn_{seed}.pt')

        # When evaluating the testing set, we needn't the parameters' update
        with torch.no_grad():
            # Test the training result
            truth = torch.tensor(np.array([features(t) for t in testing]), dtype=torch.float32)
            result['test_loss'] = float(loss(model.decoder(model(*batch(testing))), truth))
            mean = torch.tensor(raw.mean(axis=0), dtype=torch.float32).expand_as(truth)
            result['mean_predictor_test_loss'] = float(loss(mean, truth))
        result['latent_std'] = embed(model, testing).std(axis=0).tolist()
        training_results.append(result)
        write_json(OUT/'training.json',training_results)
        print('training',seed,'test loss',result['test_loss'],'mean baseline',result['mean_predictor_test_loss'],flush=True)

    # 4. Build the representations

    for name, model in models.items():
        scalers[name] = StandardScaler().fit(embed(model, training))
    cache = {}

    # 5. Reuse the data

    for directory in [OLD, Path('artifacts/stage_a_focused'), OUT]:
        for path in (directory/'cache').glob('*/record.json'):
            r = json.loads(path.read_text()); cache[r['evaluation_key']] = r
    prediction = []; trials = []; new_count = 0

    for label, pool in pools.items():
        family, target_seed = label.rsplit('_',1)
        target = gaussian_target(8,family,int(target_seed))
        records = []
        for tree in pool:
            key = digest(dict(target_hash=target.hash,tree_hash=tree_hash(tree),pipeline=asdict(Pipeline()),
                              software_versions=software(),source_hash=source_hash()))
            if key not in cache:
                cache[key] = evaluate(target,tree,Pipeline(),OUT/'cache')
                new_count += 1
            records.append(cache[key])
        write_json(OUT/label/'records.json',records)

        # Conditional cost-ranking diagnostic must not silently discard failed candidates.
        assert all(r['status']=='success' for r in records), 'Handle infeasible labels before cost ranking'
        xraw = raw_scaler.transform(np.array([features(t) for t in pool]))
        representations = {'raw':xraw, 'pca':pca.transform(xraw)}
        representations.update({name:scalers[name].transform(embed(model,pool)) for name,model in models.items()})
        costs = np.array([r['cx_count'] for r in records])
        for size in [8,16]:
            for seed in range(5):
                known = initial_ids(len(pool),size,seed)
                held = [i for i in range(len(pool)) if i not in known]
                rng = np.random.default_rng(1000*size+seed)
                random_regret = float(np.mean([min(costs[rng.choice(held,5,replace=False)])-min(costs[held]) for _ in range(256)]))
                for method, x in representations.items():
                    mean, _, warnings = gp_predict(x,known,np.log1p(costs[known]),held)
                    chosen = np.array(held)[np.argsort(mean,kind='stable')[:5]]
                    rho = float(spearmanr(mean,costs[held])[0])
                    prediction.append(dict(target=label,family=family,method=method,train_size=size,seed=seed,
                        spearman=rho if np.isfinite(rho) else None,top5_regret=int(min(costs[chosen])-min(costs[held])),
                        random_top5_regret=random_regret,known=known,held=held,recommended=chosen.tolist(),warnings=warnings))
        for seed in range(3):
            for method in ['random','neighborhood',*representations]:
                trace = replay(pool, records, representations.get(method),method,seed)
                assert len({r['candidate_id'] for r in trace}) == 64
                assert [r['candidate_id'] for r in trace[:8]] == initial_ids(128,8,seed)
                write_json(OUT/label/f'{method}_{seed}.json',trace)
                trials.append(dict(target=label,family=family,seed=seed,method=method,
                    checkpoints={str(b):trace[b-1]['best_cx'] for b in [8,16,32,64]},
                    proposal_time_s=trace[-1]['proposal_time_s']))
        write_json(OUT/'prediction.json',prediction)
        write_json(OUT/'search.json',trials)
        print(label,'complete; new evaluations',new_count,flush=True)
    write_json(OUT/'completion.json',dict(elapsed_s=time.perf_counter()-started,new_evaluations=new_count,
        training_time_s=sum(r['training_time_s'] for r in training_results),trajectories=len(trials),prediction_fits=len(prediction)))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
