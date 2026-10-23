"""New-target neighborhood validation with a frozen protocol and shared evaluator."""
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ttn_search.evaluate import Pipeline, evaluate, source_hash, software, write_json
from ttn_search.scent import scent_tree
from ttn_search.targets import gaussian_target
from ttn_search.trees import neighbors, random_tree, tree_hash

OUT = Path('artifacts/neighborhood_validation')
CONFIG = Path('configs/neighborhood_validation.json')


def search(start_tree, oracle, method, seed, budget, n=8):
    """SCENT plus `budget` unique queries; strict first-improvement local moves."""
    rng = np.random.default_rng(seed)
    seen = {tree_hash(start_tree)}
    base = oracle(start_tree)
    best_tree, best_record = start_tree, base
    current = start_tree
    trace = []
    # Candidate ordering uses structure and RNG only, never unseen costs.
    for step in range(budget):
        before = time.perf_counter()
        anchor = start_tree if method == 'initial_neighborhood' else current
        adjacent = [t for t in neighbors(anchor) if tree_hash(t) not in seen] if method != 'random' else []
        if adjacent:
            tree = adjacent[int(rng.integers(len(adjacent)))]
            action = 'neighbor'
        else:
            for _ in range(10000):
                tree = random_tree(n, rng)
                if tree_hash(tree) not in seen:
                    break
            else:
                raise RuntimeError('Unable to draw an unseen random tree')
            action = 'random_fallback' if method != 'random' else 'random'
        proposal_time = time.perf_counter()-before
        parent = best_tree
        prior = best_record
        record = oracle(tree)
        seen.add(tree_hash(tree))
        improved = (record['status'] == 'success' and record['fidelity_final'] >= .999 and
                    (best_record['status'] != 'success' or record['cx_count'] < best_record['cx_count']))
        if improved:
            best_tree, best_record = tree, record
            if method == 'iterative_neighborhood':
                current = tree
        trace.append(dict(extra_evaluations=step+1, tree=tree, tree_hash=tree_hash(tree),
            anchor=anchor if adjacent else None, prior_best_tree=parent,
            prior_best_key=prior['evaluation_key'], evaluation_key=record['evaluation_key'],
            status=record['status'], cx=record.get('cx_count'), fidelity=record.get('fidelity_final'),
            best_cx=best_record.get('cx_count') if best_record['status']=='success' else None,
            best_tree=best_tree, improved=improved, action=action, proposal_time_s=proposal_time,
            evaluation_time_s=record['evaluation_time_s'], cache_hit=record['cache_hit'],
            lookup_wall_s=record.get('lookup_wall_s')))
    return base, trace


def main():
    config = json.loads(CONFIG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(config=config, pipeline=asdict(Pipeline()), evaluator_source_hash=source_hash(),
        software=software(), runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initialization='SCENT only; budgets count additional unique evaluations; seeds shared',
        methods=dict(random='global random pair-merging trees, no fixed candidate pool',
            initial_neighborhood='uniform unseen one-hop neighbors of original SCENT, then global random',
            iterative_neighborhood='uniform unseen neighbors of current best; accept strict improvements immediately; random fallback when exhausted'),
        caveats=['new instances of existing target families, not new families or system sizes',
            'shared caching: actual execution time recorded; charged evaluation times are not independent online timings',
            'SCENT means exact-state mutual-information adapter, not full paper pipeline',
            'random fallback improvements also update iterative incumbent; this is first-improvement with restarts',
            'fidelity policy unchanged; no new resource caps'])
    protocol = json.loads(json.dumps(protocol))
    path = OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Protocol changed: use a separate output directory')
    write_json(path,protocol)
    started = time.perf_counter()
    trials = []
    actual_evaluations = 0
    for family in config['families']:
        for target_seed in config['target_seeds']:
            label = f'{family}_{target_seed}'
            directory = OUT/label
            directory.mkdir(exist_ok=True)
            target = gaussian_target(config['n'],family,target_seed)
            start_tree, metadata = scent_tree(target.state,config['scent_seed'])
            write_json(directory/'target.json',dict(target_hash=target.hash,metadata=target.metadata,
                scent_tree=start_tree,scent_metadata=metadata))
            memory = {}
            def oracle(tree):
                nonlocal actual_evaluations
                start = time.perf_counter()
                key = tree_hash(tree)
                if key in memory:
                    return {**memory[key], 'cache_hit':True, 'lookup_wall_s':time.perf_counter()-start}
                r = evaluate(target,tree,Pipeline(),OUT/'cache')
                if not r['cache_hit']: actual_evaluations += 1
                memory[key] = r
                return {**r,'lookup_wall_s':time.perf_counter()-start}
            for seed in config['search_seeds']:
                for method in config['methods']:
                    before = time.perf_counter()
                    base, trace = search(start_tree,oracle,method,seed,max(config['additional_budgets']),config['n'])
                    elapsed = time.perf_counter()-before
                    assert len({r['tree_hash'] for r in trace}) == len(trace)
                    write_json(directory/f'{method}_{seed}.json',trace)
                    trials.append(dict(target=label,family=family,seed=seed,method=method,
                        baseline_cx=base.get('cx_count'),actual_wall_s=elapsed,
                        charged_evaluation_s=base['evaluation_time_s']+sum(r['evaluation_time_s'] for r in trace),
                        checkpoints={str(b):trace[b-1]['best_cx'] for b in config['additional_budgets']},
                        improvements=sum(r['improved'] for r in trace),
                        neighbor_improvements=sum(r['improved'] and r['action']=='neighbor' for r in trace),
                        evaluated_neighbors=sum(r['action']=='neighbor' for r in trace)))
            write_json(OUT/'search.json',trials)
            print(label,'complete; new evaluations',actual_evaluations,flush=True)
    write_json(OUT/'completion.json',dict(elapsed_s=time.perf_counter()-started,
        new_evaluations=actual_evaluations,trajectories=len(trials)))


if __name__ == '__main__':
    with threadpool_limits(limits=1): main()
