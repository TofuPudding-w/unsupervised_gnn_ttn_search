"""Fresh repaired-compiler benchmark followed by exhaustive offline pool audit."""
import fcntl
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from qiskit import qpy
from threadpoolctl import threadpool_limits
from ttn_search.learning.encoder import Encoder,embed
from ttn_search.evaluate import Pipeline,write_json,software
from ttn_search.screening.verified_evaluator import evaluate_verified,prepare,scores,COMPILER_VERSION,COMPILER_HASH,EVALUATOR_HASH
from ttn_search.search import propose
from ttn_search.targets import gaussian_target
from ttn_search.trees import tree_hash,canonical
from run_revised_gnn import generate,CONFIG as ORIGINAL_CONFIG
from analyze_c1_depth import two_qubit_depth

OUT=Path('artifacts/verified_evaluator_v1')
MODELS=Path('artifacts/revised_gnn/models')
POLICY=Pipeline(tolerance_factors=(3.,1.,.1,0.))
CONFIG=dict(ORIGINAL_CONFIG)

def evaluate(target,tree,directory,prepared=None):
    return evaluate_verified(target,tree,POLICY,directory,prepared)

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
            prepared[i]=prepare(target,t,POLICY,3.)
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
    elif method.startswith('gnn_'):
        before=time.perf_counter();seed=repeat;size=int(method.split('_')[1])
        model=Encoder(n=10);model.load_state_dict(torch.load(MODELS/f'gnn_{size}_{seed}.pt',weights_only=True))
        with np.load(MODELS/f'scaler_{size}_{seed}.npz') as scaler:x=(embed(model,pool)-scaler['mean'])/scaler['scale']
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
        if method.startswith('gnn_') and step:
            i,acquisition=propose(x,observed,[i for i in candidates.tolist() if i not in observed],.999,rng)
        else:i=order[step];acquisition={'acquisition':'fixed_order'}
        ranking_s+=time.perf_counter()-before
        record=evaluate(target,pool[i],directory/f'candidate_{i}',prepared.get(i))
        if record['status']=='success':
            with Path(record['circuit_path']).open('rb') as stream:verified_circuit=qpy.load(stream)[0]
            record['d2']=two_qubit_depth(verified_circuit)
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
            best_d2=observed[best_id].get('d2') if best_id is not None else None,initial_reused=record['initial_reused'],record_path=str((directory/f'candidate_{i}'/'record.json').resolve()),**acquisition)
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


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def next_directory(parent):
    """Preserve interrupted attempts; never silently reuse their timing or labels."""
    parent.mkdir(parents=True,exist_ok=True)
    index=0
    while (parent/f'execution_{index:03d}').exists():index+=1
    result=parent/f'execution_{index:03d}'
    result.mkdir()
    write_json(result/'started.json',dict(utc=time.time(),previous_executions=index))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()


def run():
    torch.set_num_threads(1)
    assert read('artifacts/compiler_repair/completion.json')['repaired_compilation_failures']==0
    paths=[str(p) for p in Path('src/ttn_search').rglob('*.py')]+[__file__,'scripts/run_revised_gnn.py','scripts/analyze_c1_depth.py','scripts/report_verified_evaluator.py']
    model_paths=sorted(str(p) for p in MODELS.glob('*'))
    original_model_hashes=read('artifacts/revised_gnn/frozen_models.json')
    assert all(fingerprint(p)==original_model_hashes[p] for p in model_paths)
    pool_paths={13:'artifacts/stage_c1/overlapping/pool.json',**{s:f'artifacts/revised_gnn/overlapping_{s}/pool.json' for s in range(14,19)}}
    protocol=dict(compiler_version=COMPILER_VERSION,compiler_hash=COMPILER_HASH,evaluator_hash=EVALUATOR_HASH,
        config=CONFIG,policy=asdict(POLICY),software=software(),files={p:fingerprint(p) for p in paths},
        models={p:fingerprint(p) for p in model_paths},pool_files={p:fingerprint(p) for p in pool_paths.values()},
        rules=['Fresh independent online evaluations: seven frozen methods, five development targets, three repeats; no retraining',
            'Same 1/2/4/8/16 checkpoints and cyclic execution order as historical benchmark',
            'Strict original global representation/compilation/final-fidelity requirements and 3/1/0.1/0 retries',
            'Online methods never see offline pool results; exhaustive audit runs after all online comparisons',
            'Offline pool audit reuses only verified outcomes from this compiler version, then fills missing candidates',
            'Interrupted executions are preserved and reported separately; no hidden reuse or fabricated elapsed times',
            'This is a fixed-evaluation benchmark, not an uncapped fixed-time experiment; timing includes all current-method work'])
    protocol=read_json_roundtrip(protocol)
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==protocol,'Frozen protocol/source/model mismatch'
    else:write_json(path,protocol)
    targets={};pools={}
    for seed,path in pool_paths.items():
        target=gaussian_target(10,'overlapping',seed)
        source=read(path);pool=[canonical(t) for t in source['trees']]
        targets[seed]=target;pools[seed]=pool
        if 'target_hash' in source:assert source['target_hash']==target.hash
        write_json(OUT/f'overlapping_{seed}'/'pool.json',dict(trees=pool,hashes=[tree_hash(t) for t in pool],target_hash=target.hash))
    runs=[];base=np.random.default_rng(20261023).permutation(CONFIG['methods']).tolist()
    for j,seed in enumerate(CONFIG['target_seeds']):
        for repeat in CONFIG['repeats']:
            offset=(j*3+repeat)%len(base);order=base[offset:]+base[:offset]
            write_json(OUT/f'overlapping_{seed}'/f'order_{repeat}.json',order)
            for method in order:
                parent=OUT/f'overlapping_{seed}'/f'{method}_{repeat}'
                path=parent/'summary.json'
                if path.exists():result=read(path)
                else:
                    directory=next_directory(parent)
                    result=run_one(targets[seed],CONFIG,method,repeat,directory,[tree_hash(t) for t in pools[seed]])
                    result.update(compiler_version=COMPILER_VERSION,compiler_hash=COMPILER_HASH,evaluator_hash=EVALUATOR_HASH,
                        execution_directory=str(directory.resolve()),previous_executions=read(directory/'started.json')['previous_executions'])
                    write_json(path,result)
                assert result['compiler_version']==COMPILER_VERSION and result['compiler_hash']==COMPILER_HASH
                runs.append(result);write_json(OUT/'runs.json',runs)
                write_json(OUT/'status.json',dict(phase='online',completed_runs=len(runs),total_runs=105,utc=time.time()))
                print('online',len(runs),'/105',seed,repeat,method,'best',result['trace'][-1]['best_cx'],flush=True)
    write_json(OUT/'online_completion.json',dict(runs=len(runs),evaluations=sum(r['candidate_evaluations'] for r in runs)))
    allrows=[]
    for seed in range(13,19):
        existing={}
        for run in runs:
            if run['target']!=f'overlapping_{seed}':continue
            for row in run['trace']:
                record=read(row['record_path'])
                signature=(record['status'],record.get('cx_count'),record['tolerance_factor'])
                if row['candidate_id'] in existing:
                    prior=read(existing[row['candidate_id']]);assert signature==(prior['status'],prior.get('cx_count'),prior['tolerance_factor'])
                existing[row['candidate_id']]=row['record_path']
        rows=[]
        for i,tree in enumerate(pools[seed]):
            parent=OUT/'offline'/f'overlapping_{seed}'/str(i);path=parent/'result.json'
            if path.exists():row=read(path)
            else:
                start=time.perf_counter();item=prepare(targets[seed],tree,POLICY,3.);a,b=scores(item)
                reused=i in existing
                if reused:record_path=existing[i];record=read(record_path)
                else:
                    directory=next_directory(parent);record=evaluate(targets[seed],tree,directory,item)
                    record_path=str((directory/'record.json').resolve())
                assert record['compiler_version']==COMPILER_VERSION and record['compiler_hash']==COMPILER_HASH
                depth=None
                if record['status']=='success':
                    with Path(record['circuit_path']).open('rb') as stream:circuit=qpy.load(stream)[0]
                    depth=two_qubit_depth(circuit)
                row=dict(target=f'overlapping_{seed}',candidate_id=i,tree_hash=tree_hash(tree),parameters=a if np.isfinite(a) else None,
                    padded=b if np.isfinite(b) else None,status=record['status'],cx=record.get('cx_count'),d2=depth,
                    fidelity=record.get('fidelity_final'),accepted_factor=record['tolerance_factor'],record_path=record_path,
                    reused_online_result=reused,offline_wall_s=time.perf_counter()-start)
                write_json(path,row)
            rows.append(row);allrows.append(row)
            write_json(OUT/'offline'/f'overlapping_{seed}'/'pool_results.json',rows)
            if i%16==15:
                write_json(OUT/'status.json',dict(phase='offline',completed_candidates=len(allrows),total_candidates=768,utc=time.time()))
                print('offline',len(allrows),'/768',flush=True)
    write_json(OUT/'pool_results.json',allrows)
    assert all(fingerprint(p)==h for p,h in protocol['files'].items()),'Source changed during execution'
    write_json(OUT/'completion.json',dict(compiler_version=COMPILER_VERSION,runs=len(runs),
        online_candidate_evaluations=sum(r['candidate_evaluations'] for r in runs),
        online_synthesis_calls=sum(r['compiler_calls'] for r in runs),offline_candidates=len(allrows),
        offline_new_evaluations=sum(not r['reused_online_result'] for r in allrows)))
    from report_verified_evaluator import main as report
    report()
    write_json(OUT/'status.json',dict(phase='complete',utc=time.time()))


def read_json_roundtrip(value):
    return json.loads(json.dumps(value))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
