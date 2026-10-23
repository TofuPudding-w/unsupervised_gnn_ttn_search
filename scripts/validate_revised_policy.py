"""Development cohort followed by exhaustive revised-policy pool evaluation."""
import json
import hashlib
import time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits
from qiskit import qpy
from ttn_search.evaluate import write_json,source_hash,software
from ttn_search.targets import gaussian_target
from ttn_search.trees import balanced,random_tree,canonical
from ttn_search.scent import scent_tree
from ttn_search.reference import build,compile_reference,circuit_state,fidelity
from ttn_search.screening.evaluator import prepare,scores
from revised_policy import POLICY,evaluate
from analyze_c1_depth import two_qubit_depth

OUT=Path('artifacts/revised_policy')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cohort=[(8,f,s) for f in ['hierarchical','overlapping'] for s in [0,3]]+[(10,f,13) for f in ['hierarchical','overlapping']]
    protocol=dict(cohort=cohort,trees=['scent','balanced','random_seed_2026'],factors=[1,3],policy=asdict(POLICY),
        source_hash=source_hash(),software=software(),files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,'scripts/revised_policy.py','src/ttn_search/screening/evaluator.py']},
        rules=['All original sub-budgets remain enforced: representation infidelity <=0.0001, compilation <=0.0004, final F>=0.999',
            'Fixed attempt order 3,1,0.1,0; stop at first feasible; retain failed build/verify/compile costs',
            'Representation-infeasible attempts do not compile; numerical/compiler failures trigger next fixed tolerance',
            'After exhausting attempts, report measured infeasibility or operational error; never fabricate a feasible label',
            'No new per-candidate hard time or memory limit; same sequential CPU setting as reference',
            'Cohort directly compiles both factors for diagnosis even if representation sub-budget fails; pool evaluator enforces early rejection',
            'Pool descriptors use only first 3x construction, not eventual accepted retry ranks'])
    protocol=json.loads(json.dumps(protocol));p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    write_json(p,protocol);cohortrows=[]
    for n,family,seed in cohort:
        target=gaussian_target(n,family,seed);sc,_=scent_tree(target.state,0)
        for name,tree in zip(['scent','balanced','random'],[sc,balanced(list(range(n))),random_tree(n,np.random.default_rng(2026))]):
            for factor in [1,3]:
                directory=OUT/'cohort'/f'{n}_{family}_{seed}_{name}_{factor}';directory.mkdir(parents=True,exist_ok=True);path=directory/'result.json'
                if path.exists():row=json.loads(path.read_text())
                else:
                    row=dict(n=n,family=family,seed=seed,tree=name,factor=factor)
                    try:
                        start=time.perf_counter();rep=build(target.state,tree,.0001*factor/(n-1));row['build_s']=time.perf_counter()-start
                        start=time.perf_counter();circuit=compile_reference(rep,17,1);row['compile_s']=time.perf_counter()-start
                        start=time.perf_counter();output=circuit_state(circuit)
                        fr=fidelity(target.state,rep.state);fc=fidelity(rep.state,output);ff=fidelity(target.state,output)
                        row.update(status='measured',repr_error=1-fr,comp_error=1-fc,final_fidelity=ff,
                            feasible=ff>=.999 and 1-fr<=.0001+1e-12 and 1-fc<=.0004+1e-12,
                            cx=int(circuit.count_ops().get('cx',0)),d2=two_qubit_depth(circuit),verify_s=time.perf_counter()-start)
                        with (directory/'circuit.qpy').open('wb') as stream:qpy.dump(circuit,stream)
                    except Exception as exc:row.update(status='error',error=repr(exc),feasible=False)
                    write_json(path,row)
                cohortrows.append(row)
        print('cohort',n,family,seed,'complete',flush=True)
    write_json(OUT/'cohort.json',cohortrows)
    assert all(any(r['feasible'] for r in cohortrows if (r['n'],r['family'],r['seed'],r['tree'])==(n,f,s,t)) for n,f,s in cohort for t in ['scent','balanced','random']), 'Development safety gate failed'
    write_json(OUT/'policy_frozen.json',dict(policy=asdict(POLICY),cohort_gate='passed',rules=protocol['rules']))
    pool=json.loads(Path('artifacts/stage_c1/overlapping/pool.json').read_text());target=gaussian_target(10,'overlapping',13);rows=[]
    for i,tree in enumerate(pool['trees']):
        directory=OUT/'pool'/str(i);directory.mkdir(parents=True,exist_ok=True);path=directory/'result.json'
        if path.exists():row=json.loads(path.read_text())
        else:
            start=time.perf_counter();item=prepare(target,canonical(tree),POLICY,3.);a,b=scores(item)
            record=evaluate(target,canonical(tree),directory,item)
            d2=None
            if record['status']=='success':
                with Path(record['circuit_path']).open('rb') as stream:circuit=qpy.load(stream)[0]
                d2=two_qubit_depth(circuit)
            row=dict(candidate_id=i,parameters=a if np.isfinite(a) else None,padded=b if np.isfinite(b) else None,
                status=record['status'],cx=record.get('cx_count'),d2=d2,fidelity=record.get('fidelity_final'),
                accepted_factor=record['tolerance_factor'],attempts=len(record['attempts']),compiler_calls=record['compiler_calls'],
                wall_s=time.perf_counter()-start,record_path=str(directory/'record.json'))
            write_json(path,row)
        rows.append(row);write_json(OUT/'pool.json',rows)
        if i%16==0:print('pool',i,'complete',flush=True)
    write_json(OUT/'completion.json',dict(cohort_evaluations=len(cohortrows),pool_evaluations=len(rows)))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
