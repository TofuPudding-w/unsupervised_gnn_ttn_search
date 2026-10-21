"""Focused development checks; historical cache reuse requires exact source/config matching."""
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast
import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from ttn_search.evaluate import Pipeline, evaluate, write_json, source_hash, software, digest
from ttn_search.search import initial_ids, prepare_features, gp_predict, run_search
from ttn_search.targets import gaussian_target
from ttn_search.trees import canonical, tree_hash, neighbors

OLD = Path('artifacts/stage_a_pilot_v2')
OUT = Path('artifacts/stage_a_focused')


def neighborhood_search(pool, oracle, budget, init, seed):
    """Same initial observations; random order of one-hop SCENT neighbors, then random unseen."""
    neighbor_hashes = {tree_hash(t) for t in neighbors(pool[0])}
    ids = {i for i,t in enumerate(pool) if tree_hash(t) in neighbor_hashes}
    rng = np.random.default_rng(seed)
    seen=set(); trace=[]; best=None; charged=0.; proposal=0.
    for step in range(budget):
        start=time.perf_counter()
        unseen=[i for i in range(len(pool)) if i not in seen]
        adjacent=[i for i in unseen if i in ids]
        i=init[step] if step<len(init) else int(rng.choice(adjacent or unseen))
        proposal+=time.perf_counter()-start
        r=oracle(i); seen.add(i); charged+=r['evaluation_time_s']
        if r['status']=='success':best=r['cx_count'] if best is None else min(best,r['cx_count'])
        trace.append(dict(step=step+1,candidate_id=i,evaluation_key=r['evaluation_key'],status=r['status'],best_cx=best,
            acquisition='common_initial' if step<len(init) else 'SCENT_neighbor' if adjacent else 'random',
            charged_evaluation_time_s=charged,proposal_time_s=proposal,cache_hit=r.get('cache_hit',False)))
    return trace


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    start=time.perf_counter(); manifest=json.loads((OLD/'manifest.json').read_text())
    if manifest['source_hash']!=source_hash() or manifest['software']!=software():
        raise RuntimeError('Historical evaluator source/environment changed; do not silently mix records')
    protocol=dict(source_hash=source_hash(),software=software(),old_manifest=str(OLD/'manifest.json'),
        neighborhood='all six targets; 256 existing candidates; seeds 0/1/2; budgets 8/16/32/64; common eight initial points; random one-hop neighbors then random unseen',
        prediction='all six targets; complete first 32 pilot candidates; training 8/16 including SCENT; five shared splits; raw/PCA; predict held-out cost; top-five regret and low-cost recall; ties included in low-cost group',
        sensitivity='hierarchical seed 1 (anomaly target) and overlapping seed 1 (headroom target); same 256 pools; seeds 0/1/2; budget 32; four methods; global error fractions 0.2/0.2 versus original 0.1/0.4',
        random_recommendation='256 seeded random five-item recommendations per held-out split, for expected simple regret',
        caveats=['development only; targets chosen after pilot inspection','cache replay is not online speed','all first-32 labels feasible; no feasibility classifier assessment','first-32 pool prefix is heuristic-heavy','compiler unchanged for matched comparisons'])
    write_json(OUT/'protocol.json',protocol)
    records={}
    for path in (OLD/'cache').glob('*/record.json'):
        r=json.loads(path.read_text());records[r['evaluation_key']]=r
    trials=[];prediction=[];paired=[];new_counts={'primary':0,'alternative':0}
    for family in ['hierarchical','overlapping']:
        for t in range(3):
            label=f'{family}_{t}';directory=OUT/label;directory.mkdir(exist_ok=True)
            target=gaussian_target(8,family,t)
            pool=[canonical(json.loads(x)) for x in json.loads((OLD/label/'pool.json').read_text())['trees']]
            def oracle(i,pipeline=Pipeline()):
                identity=dict(target_hash=target.hash,tree_hash=tree_hash(pool[i]),pipeline=asdict(pipeline),software_versions=software(),source_hash=source_hash())
                key=digest(identity)
                if key in records:return {**records[key],'cache_hit':True,'lookup_time_s':0.}
                r=evaluate(target,pool[i],pipeline,OUT/'cache');records[key]=r
                if not r['cache_hit']:new_counts['primary' if pipeline.repr_fraction==.1 else 'alternative']+=1
                return r
            for seed in range(3):
                init=initial_ids(len(pool),8,seed)
                trace=neighborhood_search(pool,oracle,64,init,seed)
                write_json(directory/f'neighborhood_{seed}.json',trace)
                for method in ['random','raw','pca','neighborhood']:
                    tr=trace if method=='neighborhood' else json.loads((OLD/label/f'{method}_{seed}.json').read_text())
                    trials.append(dict(target=label,family=family,seed=seed,method=method,
                        checkpoints={str(b):tr[b-1]['best_cx'] for b in [8,16,32,64]}))
            print(label,'neighborhood complete',flush=True)
            small=pool[:32]; rr=[oracle(i) for i in range(32)]
            if not all(r['status']=='success' for r in rr):raise RuntimeError('Predictive cost diagnostic assumes feasible diagnostic labels')
            costs=np.array([r['cx_count'] for r in rr]); feats={m:prepare_features(small,m) for m in ['raw','pca']}
            for size in [8,16]:
                for seed in range(5):
                    train=initial_ids(32,size,seed); held=[i for i in range(32) if i not in train]
                    rng=np.random.default_rng(seed+1000*size)
                    random_regret=float(np.mean([min(costs[rng.choice(held,5,replace=False)])-min(costs[held]) for _ in range(256)]))
                    for method in ['raw','pca']:
                        mean,std,warnings=gp_predict(feats[method],train,np.log1p(costs[train]),held)
                        chosen=np.asarray(held)[np.argsort(mean,kind='stable')[:5]]
                        low=set(i for i in held if costs[i]<=np.quantile(costs[held],.2))
                        rho=float(cast(float, spearmanr(mean,costs[held])[0]))
                        prediction.append(dict(target=label,family=family,train_size=size,split_seed=seed,method=method,
                            train_ids=train,heldout_ids=held,recommended_ids=chosen.tolist(),spearman=rho if np.isfinite(rho) else None,
                            top5_simple_regret=int(min(costs[chosen])-min(costs[held])),
                            random_top5_expected_regret=random_regret,low_cost_recall=len(set(chosen)&low)/len(low),
                            low_cost_precision=len(set(chosen)&low)/5,warnings=warnings))
            print(label,'prediction complete',flush=True)
            if t==1:
                alternate=Pipeline(repr_fraction=.2,comp_fraction=.2)
                # Fixed common candidate subset for a paired rank comparison.
                for i in range(32):
                    old=oracle(i);new=oracle(i,alternate)
                    paired.append(dict(target=label,candidate_id=i,primary_cx=old.get('cx_count'),alternate_cx=new.get('cx_count'),
                        primary_status=old['status'],alternate_status=new['status'],primary_fidelity=old.get('fidelity_final'),alternate_fidelity=new.get('fidelity_final')))
                for seed in range(3):
                    init=initial_ids(len(pool),8,seed)
                    for method in ['random','raw','pca','neighborhood']:
                        evaluator=lambda i:oracle(i,alternate)
                        trace=neighborhood_search(pool,evaluator,32,init,seed) if method=='neighborhood' else run_search(pool,evaluator,method,32,init,.999,seed)
                        write_json(directory/f'alternative_{method}_{seed}.json',trace)
                        trials.append(dict(target=label,family=family,seed=seed,method=method,allocation='0.2/0.2',
                            checkpoints={str(b):trace[b-1]['best_cx'] for b in [8,16,32]}))
                print(label,'allocation sensitivity complete',new_counts,flush=True)
            write_json(OUT/'search_comparisons.json',trials)
            write_json(OUT/'prediction.json',prediction)
            write_json(OUT/'allocation_pairs.json',paired)
    write_json(OUT/'completion.json',dict(elapsed_s=time.perf_counter()-start,new_evaluations=new_counts))
    print('Completed',new_counts,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
