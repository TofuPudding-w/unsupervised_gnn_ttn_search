"""Audit a rank transition and test construction-only descriptors on saved outcomes."""
import json
import time
import numpy as np
from qiskit import QuantumCircuit,transpile,qpy
from qiskit.circuit.library import Isometry
from qiskit.exceptions import QiskitError
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from decision_common import freeze,STAGE_B,NEIGHBOR,construct,nodes
from ttn_search.evaluate import write_json
from ttn_search.reference import padded_isometry,nbits,compile_reference,circuit_state,fidelity
from ttn_search.search import initial_ids,gp_predict
from ttn_search.targets import gaussian_target
from ttn_search.trees import canonical,tree_hash,features


def local_blocks(rep,target,pipeline,tolerance):
    rows=[]
    tensor=target.state.reshape([2]*target.metadata['n'],order='F')
    for node in nodes(rep):
        assert node.children is not None
        a,b=node.children
        input_bits=nbits(node.rank);output_bits=nbits(a.rank)+nbits(b.rank)
        row=dict(leaves=sorted(node.labels),children=[sorted(a.labels),sorted(b.labels)],
            rank=node.rank,child_ranks=[a.rank,b.rank],input_bits=input_bits,output_bits=output_bits,
            padded_rows=2**output_bits,padded_columns=2**input_bits,local_tolerance=tolerance)
        try:
            c=QuantumCircuit(output_bits)
            if output_bits: c.append(Isometry(padded_isometry(node),0,0),range(output_bits))
            c=transpile(c,basis_gates=['u','cx'],optimization_level=pipeline['optimization_level'],seed_transpiler=pipeline['compiler_seed'])
            row['cx']=int(c.count_ops().get('cx',0))
        except (QiskitError,ValueError,np.linalg.LinAlgError) as exc: row['error']=str(exc)
        if node is not rep.root:
            other=[q for q in range(target.metadata['n']) if q not in node.labels]
            unfolding=tensor.transpose(node.labels+other).reshape(2**len(node.labels),-1)
            reduced=np.kron(a.basis,b.basis).conj().T@unfolding
            singular=np.linalg.svd(reduced,compute_uv=False)
            row['singular_values']=singular.tolist()
            row['discarded_weight_at_8']=float(np.sum(singular[8:]**2))
            row['discarded_weight_at_9']=float(np.sum(singular[9:]**2))
        rows.append(row)
    return rows


def transition(config,out):
    selected=json.loads((NEIGHBOR/'exploratory_best_target_path.json').read_text())
    step=next(s for s in selected['steps'] if [s['before']['cx'],s['after']['cx']]==config['transition_cnot_pair'])
    family,seed=config['transition_target'].rsplit('_',1)
    target=gaussian_target(8,family,int(seed))
    pipeline=json.loads((NEIGHBOR/'protocol.json').read_text())['pipeline']
    rows=[]
    for multiplier in config['tolerance_multipliers']:
        for side in ['before','after']:
            rep,info=construct(target,step[side]['tree'],pipeline,multiplier)
            row=dict(side=side,multiplier=multiplier,tree=step[side]['tree'],**info,
                blocks=local_blocks(rep,target,pipeline,info['local_tolerance']))
            try:
                before=time.perf_counter()
                c=compile_reference(rep,pipeline['compiler_seed'],pipeline['optimization_level'])
                row['compile_s']=time.perf_counter()-before
                state=circuit_state(c)
                row.update(cx=int(c.count_ops().get('cx',0)),final_fidelity=fidelity(target.state,state),
                    compiler_fidelity=fidelity(rep.state,state))
                row['feasible']=(row['final_fidelity']>=pipeline['fidelity'] and
                    1-row['representation_fidelity']<=(1-pipeline['fidelity'])*pipeline['repr_fraction']+1e-12 and
                    1-row['compiler_fidelity']<=(1-pipeline['fidelity'])*pipeline['comp_fraction']+1e-12)
                directory=out/'transition_circuits';directory.mkdir(exist_ok=True)
                with (directory/f'{side}_{multiplier}.qpy').open('wb') as f:qpy.dump(c,f)
                np.savez_compressed(directory/f'{side}_{multiplier}.npz',target=target.state,representation=rep.state,output=state)
                if multiplier==1.:assert row['cx']==step[side]['cx'], 'Nominal historical cost did not reproduce'
            except (QiskitError,ValueError,np.linalg.LinAlgError) as exc:
                row.update(feasible=False,error=str(exc))
            rows.append(row)
    write_json(out/'transition.json',dict(target=config['transition_target'],
        policy='single construction and compilation per multiplier, no retries; original global error budgets unchanged',rows=rows))
    print('Transition and tolerance sweep complete',flush=True)


def main():
    config,out=freeze('cost');started=time.perf_counter()
    transition(config,out)
    record_map={};cohorts={}
    for family in ['hierarchical','overlapping']:
        for seed in range(3):
            label=f'{family}_{seed}'
            records=json.loads((STAGE_B/label/'records.json').read_text())
            for r in records:record_map[r['evaluation_key']]=r
            cohorts['stage_b/'+label]=[r['evaluation_key'] for r in records]
    new_records=[json.loads(p.read_text()) for p in (NEIGHBOR/'cache').glob('*/record.json')]
    for r in new_records:record_map[r['evaluation_key']]=r
    for family in ['hierarchical','overlapping']:
        for seed in range(3,8):
            label=f'{family}_{seed}'
            meta=json.loads((NEIGHBOR/label/'target.json').read_text())
            base=next(r for r in new_records if r['target_hash']==meta['target_hash'] and r['tree_hash']==tree_hash(canonical(meta['scent_tree'])))
            keys=set()
            for search_seed in range(3):
                keys.update(r['evaluation_key'] for r in json.loads((NEIGHBOR/label/f'random_{search_seed}.json').read_text()))
            keys.discard(base['evaluation_key'])
            cohorts['new_random/'+label]=[base['evaluation_key']]+sorted(keys,key=lambda k:record_map[k]['tree_hash'])
    targets={};extracted={}
    for key,r in record_map.items():
        family=r['target_metadata']['family'];seed=r['target_metadata']['seed'];label=f'{family}_{seed}'
        if label not in targets:targets[label]=gaussian_target(8,family,seed)
        target=targets[label];assert target.hash==r['target_hash']
        tree=canonical(json.loads(r['tree_serialization']))
        rep,info=construct(target,tree,r['pipeline'])
        # Verify numerical compatibility with historical labels, without using accepted-retry features.
        verification_rep=rep if r['tolerance_factor']==1. else construct(target,tree,r['pipeline'],r['tolerance_factor'])[0]
        assert verification_rep.parameter_count==r['parameter_count']
        assert max(verification_rep.ranks)==r['max_bond_dimension']
        assert abs(fidelity(target.state,verification_rep.state)-r['fidelity_repr'])<1e-10
        extracted[key]=dict(target=label,tree=tree,status=r['status'],cx=r.get('cx_count'),
            historical_source_hash=r['source_hash'],selected_tolerance_factor=r['tolerance_factor'],
            original_evaluation_s=r['evaluation_time_s'],
            original_compile_s=sum(a.get('time_compile_s',0.) for a in r['attempts']),**info)
        if len(extracted)%500==0:print('Extracted',len(extracted),'construction-only descriptors',flush=True)
    write_json(out/'descriptors.json',extracted);write_json(out/'cohorts.json',cohorts)
    prediction=[];screening=[];counterexamples=[]
    for name,keys in cohorts.items():
        cohort,label=name.split('/');family=label.rsplit('_',1)[0]
        rows=[extracted[k] for k in keys]
        assert all(r['status']=='success' for r in rows)
        costs=np.array([r['cx'] for r in rows]);d=np.array([r['features'] for r in rows],dtype=float)
        raw=np.array([features(canonical(r['tree'])) for r in rows])
        for size in config['known_counts']:
            for seed in config['split_seeds']:
                known=initial_ids(len(keys),size,seed);held=np.array([i for i in range(len(keys)) if i not in known])
                rng=np.random.default_rng(1000*size+seed)
                random_regret=float(np.mean([min(costs[rng.choice(held,5,replace=False)])-min(costs[held]) for _ in range(256)]))
                tie_order=np.random.default_rng(20261023+size*100+seed).permutation(len(held))
                for method in ['max_rank','parameters','sum_padded_entries','rank_width_gp','raw_gp']:
                    if method.endswith('_gp'):
                        values=np.log1p(d) if method=='rank_width_gp' else raw
                        x=StandardScaler().fit(values[known]).transform(values)
                        mean,_,_=gp_predict(x,known,np.log1p(costs[known]),held)
                    else:
                        mean=d[held,config['descriptor_names'].index(method)]
                    selected=held[tie_order[np.argsort(mean[tie_order],kind='stable')[:5]]]
                    rho=float(spearmanr(mean,costs[held])[0])
                    regret=int(min(costs[selected])-min(costs[held]))
                    low=set(held[costs[held]<=np.quantile(costs[held],.2)].tolist())
                    prediction.append(dict(cohort=cohort,target=label,family=family,method=method,known_count=size,seed=seed,
                        known=known,held=held.tolist(),recommended=selected.tolist(),spearman=rho if np.isfinite(rho) else None,
                        top5_regret=regret,random_top5_regret=random_regret,low_cost_precision=len(set(selected)&low)/5))
                    if seed==0 and size==8 and method=='sum_padded_entries':
                        truebest=int(held[np.argmin(costs[held])])
                        chosenbest=int(selected[np.argmin(costs[selected])])
                        counterexamples.append(dict(cohort=cohort,target=label,regret=regret,
                            recommended_best_key=keys[chosenbest],heldout_best_key=keys[truebest],
                            recommended_best_cx=int(costs[chosenbest]),heldout_best_cx=int(costs[truebest]),
                            recommended_descriptor=float(d[chosenbest,6]),heldout_best_descriptor=float(d[truebest,6])))
        # Label-free screening with SCENT initialization; all candidate construction is charged.
        for seed in config['search_seeds']:
            for method in ['random','parameters','sum_padded_entries']:
                order=np.random.default_rng(seed).permutation(np.arange(1,len(keys)))
                if method!='random':order=order[np.argsort(d[order,config['descriptor_names'].index(method)],kind='stable')]
                order=np.concatenate(([0],order))
                screening.append(dict(cohort=cohort,target=label,family=family,seed=seed,method=method,
                    checkpoints={str(b):int(min(costs[order[:b]])) for b in config['search_budgets']},
                    construction_s=sum(r['construction_s'] for r in rows) if method!='random' else 0.,
                    candidate_count=len(keys)))
        print(name,'prediction and screening complete',flush=True)
    write_json(out/'prediction.json',prediction);write_json(out/'screening.json',screening)
    write_json(out/'counterexamples.json',counterexamples)
    write_json(out/'completion.json',dict(elapsed_s=time.perf_counter()-started,descriptors=len(extracted),
        cohorts=len(cohorts),prediction_fits=len(prediction),screening_runs=len(screening),
        construction_s=sum(r['construction_s'] for r in extracted.values()),
        historical_compilation_s=sum(r['original_compile_s'] for r in extracted.values())))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
