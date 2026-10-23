"""Matched-update, size-specific GNN study under the frozen revised evaluator."""
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from qiskit import qpy
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ttn_search.learning.encoder import Encoder,batch,loss,embed,corpus
from ttn_search.evaluate import write_json,source_hash,software
from ttn_search.screening.evaluator import prepare,scores
from ttn_search.search import propose
from ttn_search.targets import gaussian_target
from ttn_search.trees import random_tree,tree_hash,canonical,features
from ttn_search.scent import scent_tree
from revised_policy import POLICY,evaluate
from analyze_c1_depth import two_qubit_depth

OUT=Path('artifacts/revised_gnn')
MODELS=OUT/'models'
CONFIG=dict(n=10,families=['overlapping'],target_seeds=[14,15,16,17,18],pool_size=128,pool_seed=20261024,
    repeats=[0,1,2],candidate_budgets=[1,2,4,8,16],scent_seed=0,
    methods=['scent','random','random_scent','parameters','padded','gnn_2000','gnn_10000'],
    training_sizes=[2000,10000],training_updates=1600,batch_size=128,validation_every=100,training_seeds=[0,1,2])

def generate(target,config,only_scent=False):
    tree,metadata=scent_tree(target.state,config['scent_seed']);pool=[tree];seen={tree_hash(tree)}
    rng=np.random.default_rng(config['pool_seed']+target.metadata['seed'])
    while not only_scent and len(pool)<config['pool_size']:
        t=random_tree(config['n'],rng);key=tree_hash(t)
        if key not in seen:pool.append(t);seen.add(key)
    return pool,metadata


def train_model(training,validation,testing,size,seed):
    torch.manual_seed(seed);model=Encoder(n=10);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    x,a=batch(training[:size]);vx,va=batch(validation);tx,ta=batch(testing)
    y=torch.tensor(np.array([features(t) for t in training[:size]]),dtype=torch.float32)
    vy=torch.tensor(np.array([features(t) for t in validation]),dtype=torch.float32)
    ty=torch.tensor(np.array([features(t) for t in testing]),dtype=torch.float32)
    best=float('inf');weights=copy.deepcopy(model.state_dict());history=[];start=time.perf_counter()
    for step in range(1,CONFIG['training_updates']+1):
        model.train();ids=torch.randint(size,(CONFIG['batch_size'],))
        optimizer.zero_grad();value=loss(model.decoder(model(x[ids],a[ids])),y[ids],n=10);value.backward();optimizer.step()
        if step%CONFIG['validation_every']==0:
            model.eval()
            with torch.no_grad():val=float(loss(model.decoder(model(vx,va)),vy,n=10))
            history.append(dict(step=step,validation=val,train_batch_loss=float(value.detach())))
            if val<best:best=val;weights=copy.deepcopy(model.state_dict())
    model.load_state_dict(weights);model.eval()
    with torch.no_grad():test=float(loss(model.decoder(model(tx,ta)),ty,n=10))
    torch.save(model.state_dict(),MODELS/f'gnn_{size}_{seed}.pt')
    scaler=StandardScaler().fit(embed(model,training[:2000]))
    np.savez(MODELS/f'scaler_{size}_{seed}.npz',mean=scaler.mean_,scale=scaler.scale_)
    result=dict(size=size,seed=seed,updates=CONFIG['training_updates'],history=history,best_validation=best,test_loss=test,training_s=time.perf_counter()-start)
    write_json(MODELS/f'training_{size}_{seed}.json',result)
    return result

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


def main():
    torch.set_num_threads(1)
    assert Path('artifacts/revised_policy/completion.json').exists(),'Complete development policy validation and pool first'
    policy=Path('artifacts/revised_policy/policy_frozen.json')
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(exist_ok=True)
    protocol=dict(config=CONFIG,policy=json.loads(policy.read_text()),policy_hash=hashlib.sha256(policy.read_bytes()).hexdigest(),core_hash=source_hash(),software=software(),
        files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,'scripts/revised_policy.py','src/ttn_search/learning/encoder.py','src/ttn_search/screening/evaluator.py']},
        rules=['New ten-qubit models; not transfer of eight-qubit checkpoints',
            'Nested 2000/10000 random corpora; common held-out 500 validation and 500 test trees; pool trees excluded',
            'Same 1600 updates, batch size 128, Adam and three paired seeds; validation-only checkpoint selection',
            'Equal update budget is not a convergence guarantee; no mixed pretraining',
            'Frozen five untouched overlapping targets; all outcomes retained; depth reported for selected CNOT incumbent, not depth-constrained BO',
            'Fresh independent method runs, no result cache; repeated cyclic execution orders balance positions across fifteen target/repetition groups',
            'Fixed-evaluation comparison only: report total elapsed cost and checkpoint time, not an uncapped fixed-time claim',
            'Online clocks include full pool generation, construction, proposals/model loading, all retries and verification; training separately reported'])
    protocol=json.loads(json.dumps(protocol));p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    write_json(p,protocol)
    targets={};pools={};excluded=set()
    for seed in CONFIG['target_seeds']:
        target=gaussian_target(10,'overlapping',seed);pool,_=generate(target,CONFIG)
        targets[seed]=target;pools[seed]=pool;excluded.update(tree_hash(t) for t in pool)
        write_json(OUT/f'overlapping_{seed}'/'pool.json',dict(trees=pool,hashes=[tree_hash(t) for t in pool],target_hash=target.hash))
    path=OUT/'corpus.json'
    if path.exists():data=json.loads(path.read_text())
    else:
        trees=corpus(11000,excluded,seed=20261023,n=10)
        data=dict(train=trees[:10000],validation=trees[10000:10500],test=trees[10500:]);write_json(path,data)
    training,validation,testing=[[canonical(t) for t in data[k]] for k in ['train','validation','test']]
    stats=[]
    for size in CONFIG['training_sizes']:
        for seed in CONFIG['training_seeds']:
            path=MODELS/f'training_{size}_{seed}.json'
            stats.append(json.loads(path.read_text()) if path.exists() else train_model(training,validation,testing,size,seed))
            print('training complete',size,seed,flush=True)
    write_json(OUT/'training.json',stats)
    models={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.glob('*')}
    write_json(OUT/'frozen_models.json',models)
    runs=[];base=np.random.default_rng(20261023).permutation(CONFIG['methods']).tolist()
    for j,seed in enumerate(CONFIG['target_seeds']):
        for repeat in CONFIG['repeats']:
            offset=(j*3+repeat)%len(base);order=base[offset:]+base[:offset]
            write_json(OUT/f'overlapping_{seed}'/f'order_{repeat}.json',order)
            for method in order:
                directory=OUT/f'overlapping_{seed}'/f'{method}_{repeat}';path=directory/'summary.json'
                result=json.loads(path.read_text()) if path.exists() else run_one(targets[seed],CONFIG,method,repeat,directory,[tree_hash(t) for t in pools[seed]])
                runs.append(result);write_json(OUT/'runs.json',runs)
            print('target/repeat complete',seed,repeat,len(runs),flush=True)
    write_json(OUT/'completion.json',dict(runs=len(runs),candidate_evaluations=sum(r['candidate_evaluations'] for r in runs),compiler_calls=sum(r['compiler_calls'] for r in runs),training_s=sum(r['training_s'] for r in stats)))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
