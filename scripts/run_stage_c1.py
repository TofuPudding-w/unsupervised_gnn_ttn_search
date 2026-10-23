"""C1 pilot: independent evaluation-budget and hard-deadline experiments."""
import hashlib
import json
import multiprocessing as mp
import resource
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ttn_search.evaluate import Pipeline,write_json,source_hash,software
from ttn_search.screening.evaluator import prepare,scores,evaluate_fresh
from ttn_search.scent import scent_tree
from ttn_search.targets import gaussian_target
from ttn_search.trees import random_tree,tree_hash

OUT=Path('artifacts/stage_c1')
CONFIG=Path('configs/stage_c1.json')

def generate(target,config,only_scent=False):
    tree,metadata=scent_tree(target.state,config['scent_seed'])
    pool=[tree];seen={tree_hash(tree)}
    rng=np.random.default_rng(config['pool_seed']+target.metadata['seed'])
    while not only_scent and len(pool)<config['pool_size']:
        t=random_tree(config['n'],rng);key=tree_hash(t)
        if key not in seen:pool.append(t);seen.add(key)
    return pool,metadata

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
    elif method=='random_scent':
        order=[0]+rng.permutation(np.arange(1,len(pool))).tolist()
    elif method=='random':order=rng.permutation(candidates).tolist()
    else:order=[0]
    limit=1 if method=='scent' else (len(pool) if config['mode']=='time' else max(config['candidate_budgets']))
    observed={};trace=[];best=None;best_id=None;compiler_calls=0
    compile_s=verify_s=incremental_build_s=incremental_repr_verify_s=retry_s=io_s=0.
    for step in range(limit):
        before=time.perf_counter()
        i=order[step];acquisition={'acquisition':'fixed_order'}
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


def worker(target,config,method,directory,hashes,connection):
    with threadpool_limits(limits=1):
        connection.send(time.monotonic())
        result=run_one(target,config,method,0,directory,hashes)
        result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(directory/'summary.json',result)

def main():
    config=json.loads(CONFIG.read_text());OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),*Path('src/ttn_search/screening').glob('*.py')]
    protocol=dict(config=config,core_hash=source_hash(),software=software(),
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        rules=['Two fresh targets, one repetition: resource/scale pilot, not final statistical benchmark',
            '128 candidates; identical scoring, compiler and fidelity 0.999',
            'Separate fixed-evaluation and hard-deadline runs; time mode has no 16-candidate cap',
            'Time includes generation and whole-pool construction; target creation and process setup excluded',
            'Parent terminates worker at deadline, retains only completed trace rows before cutoff',
            'Fixed-evaluation safety ceiling 300 seconds per run; incomplete checkpoints reported missing',
            '12 qubits deferred until resource evidence is reviewed; no GNN pretraining'])
    pp=OUT/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise RuntimeError('Protocol changed')
    write_json(pp,protocol)
    runs=[]
    for family in config['families']:
        target=gaussian_target(config['n'],family,config['target_seed'])
        pool,_=generate(target,config);hashes=[tree_hash(t) for t in pool]
        write_json(OUT/family/'pool.json',dict(trees=pool,hashes=hashes,target_hash=target.hash))
        for mode in ['evaluation','time']:
            order=np.random.default_rng(2718).permutation(config['methods']).tolist()
            for method in order:
                directory=OUT/family/mode/method;directory.mkdir(parents=True,exist_ok=True)
                final=directory/'controlled.json'
                if final.exists():runs.append(json.loads(final.read_text()));continue
                settings=dict(config,mode=mode)
                parent,child=mp.Pipe()
                process=mp.Process(target=worker,args=(target,settings,method,directory,hashes,child))
                process.start();started=parent.recv()
                limit=config['deadline_s'] if mode=='time' else config['evaluation_safety_s']
                peak=0
                while process.is_alive() and time.monotonic()-started<limit:
                    try:
                        for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                            if line.startswith('VmHWM:'):peak=max(peak,int(line.split()[1]))
                    except FileNotFoundError:pass
                    process.join(timeout=min(.05,max(0,limit-(time.monotonic()-started))))
                stopped=process.is_alive()
                if stopped:process.terminate()
                process.join()
                trace_path=directory/'trace.json'
                try:trace=json.loads(trace_path.read_text())
                except (FileNotFoundError,json.JSONDecodeError):trace=[]
                eligible=[r for r in trace if r['elapsed_s']<=limit]
                summary=json.loads((directory/'summary.json').read_text()) if (directory/'summary.json').exists() else None
                result=dict(family=family,n=config['n'],method=method,mode=mode,
                    deadline_s=limit,stopped_at_deadline=stopped,exitcode=process.exitcode,
                    peak_rss_kib=peak,trace=eligible,summary=summary,
                    first_result_s=next((r['elapsed_s'] for r in eligible if r['best_cx'] is not None),None),
                    note='An interrupted in-flight compilation is not a verified result; partial artifacts retained.')
                write_json(final,result);runs.append(result);write_json(OUT/'runs.json',runs)
                print(family,mode,method,len(eligible),'completed; stopped',stopped,flush=True)
    write_json(OUT/'completion.json',dict(runs=len(runs)))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
