"""Frozen fresh-timing benchmark: generate, screen, compile, and verify."""
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ttn_search.evaluate import Pipeline,write_json,software,source_hash
from ttn_search.learning.encoder import Encoder,embed
from ttn_search.screening.evaluator import prepare,scores,evaluate_fresh
from ttn_search.scent import scent_tree
from ttn_search.search import propose
from ttn_search.targets import gaussian_target
from ttn_search.trees import canonical,random_tree,tree_hash

OUT=Path('artifacts/screening_benchmark')
CONFIG=Path('configs/screening_benchmark.json')
MODELS=Path('artifacts/decision_experiment/gnn')


def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(target,config,only_scent=False):
    tree,metadata=scent_tree(target.state,config['scent_seed'])
    pool=[tree];seen={tree_hash(tree)}
    rng=np.random.default_rng(config['pool_seed']+target.metadata['seed'])
    while not only_scent and len(pool)<config['pool_size']:
        t=random_tree(config['n'],rng);key=tree_hash(t)
        if key not in seen:pool.append(t);seen.add(key)
    return pool,metadata


def frozen_assets(config):
    """Prepare scaler files from the original training corpus only; no new labels."""
    before=time.perf_counter()
    data=json.loads(Path('artifacts/stage_b/corpus.json').read_text())
    training=[canonical(t) for t in data['train']]
    for seed in config['gnn_model_for_repeat']:
        model=Encoder();model.load_state_dict(torch.load(MODELS/f'gnn_{seed}.pt',weights_only=True))
        scaler=StandardScaler().fit(embed(model,training))
        np.savez(OUT/f'scaler_{seed}.npz',mean=scaler.mean_,scale=scaler.scale_)
    return time.perf_counter()-before


def run_one(target,config,method,repeat,directory,expected_pool):
    start=time.perf_counter();directory.mkdir(parents=True,exist_ok=True)
    before=time.perf_counter();pool,scent_meta=generate(target,config,method=='scent')
    generation_s=time.perf_counter()-before
    assert [tree_hash(t) for t in pool]==expected_pool[:len(pool)]
    rng=np.random.default_rng(repeat)
    prepared={};prescreen_s=0.;prescreen_build_s=0.;prescreen_verify_s=0.;ranking_s=0.;model_s=0.
    candidates=np.arange(len(pool));x=None;score_rows=[]
    if method in ['parameters','padded']:
        before=time.perf_counter()
        for i,t in enumerate(pool):
            prepared[i]=prepare(target,t,Pipeline())
            prescreen_build_s+=prepared[i].attempt['time_build_s']
            prescreen_verify_s+=prepared[i].attempt['time_repr_verify_s']
        prescreen_s=time.perf_counter()-before
        before=time.perf_counter()
        for i in range(len(pool)):
            a,b=scores(prepared[i])
            score_rows.append(dict(candidate_id=i,parameters=a if np.isfinite(a) else None,
                padded=b if np.isfinite(b) else None,initial_status=prepared[i].attempt['status']))
        values=np.array([scores(prepared[i])[0 if method=='parameters' else 1] for i in range(len(pool))])
        tie_order=rng.permutation(candidates)
        order=tie_order[np.argsort(values[tie_order],kind='stable')].tolist()
        ranking_s=time.perf_counter()-before
        write_json(directory/'scores.json',score_rows)
    elif method=='frozen_gnn_bo':
        before=time.perf_counter();seed=config['gnn_model_for_repeat'][repeat]
        model=Encoder();model.load_state_dict(torch.load(MODELS/f'gnn_{seed}.pt',weights_only=True))
        with np.load(OUT/f'scaler_{seed}.npz') as scaler:x=(embed(model,pool)-scaler['mean'])/scaler['scale']
        model_s=time.perf_counter()-before;order=[0]
    elif method=='random_scent':
        order=[0]+rng.permutation(np.arange(1,len(pool))).tolist()
    elif method=='random':order=rng.permutation(candidates).tolist()
    else:order=[0]
    limit=1 if method=='scent' else max(config['candidate_budgets'])
    observed={};trace=[];best=None;best_id=None;compiler_calls=0
    compile_s=verify_s=incremental_build_s=incremental_repr_verify_s=retry_s=io_s=0.
    for step in range(limit):
        before=time.perf_counter()
        if method=='frozen_gnn_bo' and step:
            i,acquisition=propose(x,observed,[i for i in candidates.tolist() if i not in observed],.999,rng)
        else:i=order[step];acquisition={'acquisition':'fixed_order'}
        ranking_s+=time.perf_counter()-before
        record=evaluate_fresh(target,pool[i],Pipeline(),directory/f'candidate_{i}',prepared.get(i))
        observed[i]=record;compiler_calls+=record['compiler_calls']
        compile_s+=record['compile_s'];verify_s+=record['verify_s'];retry_s+=record['retry_s'];io_s+=record['io_s']
        incremental_build_s+=record['incremental_build_s'];incremental_repr_verify_s+=record['incremental_repr_verify_s']
        if record['status']=='success' and record['fidelity_final']>=.999:
            if best is None or record['cx_count']<best:best=record['cx_count'];best_id=i
        row=dict(k=step+1,candidate_id=i,tree_hash=tree_hash(pool[i]),status=record['status'],
            best_cx=best,best_candidate_id=best_id,compiler_calls=compiler_calls,elapsed_s=time.perf_counter()-start,
            compile_s=compile_s,verify_s=verify_s,retry_s=retry_s,io_s=io_s,ranking_s=ranking_s,
            construction_s=prescreen_build_s+incremental_build_s,
            representation_verification_s=prescreen_verify_s+incremental_repr_verify_s,
            initial_reused=record['initial_reused'],record_path=str((directory/f'candidate_{i}'/'record.json').resolve()),**acquisition)
        trace.append(row);write_json(directory/'trace.json',trace)
    result=dict(method=method,repeat=repeat,target=f"{target.metadata['family']}_{target.metadata['seed']}",
        family=target.metadata['family'],pool_size=len(pool),generation_s=generation_s,scent_s=scent_meta['time_s'],
        prescreen_s=prescreen_s,prescreen_build_s=prescreen_build_s,prescreen_repr_verify_s=prescreen_verify_s,
        model_setup_s=model_s,ranking_s=ranking_s,construction_s=prescreen_build_s+incremental_build_s,
        representation_verification_s=prescreen_verify_s+incremental_repr_verify_s,compile_s=compile_s,
        circuit_verification_s=verify_s,retry_s=retry_s,io_s=io_s,compiler_calls=compiler_calls,
        candidate_evaluations=limit,actual_wall_s=time.perf_counter()-start,trace=trace)
    write_json(directory/'summary.json',result)
    return result


def main():
    torch.set_num_threads(1);config=json.loads(CONFIG.read_text());OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),*sorted(Path('src/ttn_search/screening').glob('*.py')),
        Path('src/ttn_search/search.py'),Path('src/ttn_search/learning/encoder.py')]
    protocol=dict(config=config,pipeline=asdict(Pipeline()),software=software(),torch=torch.__version__,
        evaluator_source_hash=source_hash(),files={str(p):fingerprint(p) for p in paths},
        model_hashes={str(MODELS/f'gnn_{s}.pt'):fingerprint(MODELS/f'gnn_{s}.pt') for s in config['gnn_model_for_repeat']},
        corpus_hash=fingerprint('artifacts/stage_b/corpus.json'),
        rules=['Untouched target seeds 8–12; freeze before evaluating their circuits',
            'All search methods use SCENT plus 127 unique random pair-merging trees; no heuristic prefix or fixed neighbor enrichment',
            'SCENT alone constructs only its own tree; all other methods generate the identical full candidate pool',
            'Screening needs no known circuit labels and compiles its ranked top K without forced SCENT initialization',
            'Random is uniform without replacement; random_scent is a stronger control starting with SCENT',
            'GNN checkpoints and scalers frozen from original training only; repeat 0/1/2 uses seed 0/1/2; first SCENT then existing constrained BO, no forced eight-point initialization',
            'Each method/repeat independently constructs and compiles; no cross-method result or TTN cache',
            'Screening reuses its own first-attempt TTNs; retries rebuild only when required by unchanged tolerance schedule',
            'Actual elapsed time starts before SCENT/pool generation and includes model loading, screening, ranking, compilation, retries, verification and artifact writing',
            'Target state creation, imports, a shared old-target warmup, and offline scaler preparation reported separately',
            'K counts candidate evaluation opportunities including failures; actual synthesis-call budgets are reported separately',
            'Time curves use only completed results, capped at 16 candidate evaluations; finished methods do not spend remaining time on extra queries',
            'Method order randomized per target/repeat before any labels; all compute sequential with one BLAS/PyTorch thread',
            'New instances of existing families and same reference compiler; Stage C expansion remains on hold'])
    protocol=json.loads(json.dumps(protocol))
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Frozen protocol changed; use a new output directory')
    write_json(path,protocol)
    started=time.perf_counter();offline_s=frozen_assets(config)
    # Import and library warmup uses an old development target, not a new benchmark label.
    before=time.perf_counter();warm=gaussian_target(8,'hierarchical',0)
    warm_pool,_=generate(warm,config,True)
    evaluate_fresh(warm,warm_pool[0],Pipeline(),OUT/'warmup')
    warmup_s=time.perf_counter()-before
    summaries=[];target_creation_s=0.
    for family in config['families']:
        for seed in config['target_seeds']:
            before=time.perf_counter();target=gaussian_target(config['n'],family,seed);target_creation_s+=time.perf_counter()-before
            label=f'{family}_{seed}';reference_pool,meta=generate(target,config)
            hashes=[tree_hash(t) for t in reference_pool]
            write_json(OUT/label/'pool.json',dict(trees=reference_pool,hashes=hashes,target_hash=target.hash,metadata=target.metadata))
            for repeat in config['repeats']:
                execution_order=np.random.default_rng(20261024+seed*101+repeat).permutation(config['methods']).tolist()
                write_json(OUT/label/f'order_{repeat}.json',execution_order)
                for method in execution_order:
                    directory=OUT/label/f'{method}_{repeat}'
                    if (directory/'summary.json').exists():result=json.loads((directory/'summary.json').read_text())
                    else:result=run_one(target,config,method,repeat,directory,hashes)
                    summaries.append(result)
                write_json(OUT/'runs.json',summaries)
            print(label,'complete',len(summaries),'independently timed method runs',flush=True)
    write_json(OUT/'completion.json',dict(invocation_wall_s=time.perf_counter()-started,runs=len(summaries),
        summed_method_wall_s=sum(r['actual_wall_s'] for r in summaries),offline_scaler_preparation_s=offline_s,
        old_target_warmup_s=warmup_s,target_state_creation_s=target_creation_s,
        candidate_evaluations=sum(r['candidate_evaluations'] for r in summaries),
        actual_compiler_calls=sum(r['compiler_calls'] for r in summaries)))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
