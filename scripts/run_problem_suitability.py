"""Frozen problem-suitability experiment; no GNN training or ternary structures."""
import argparse
import fcntl
import hashlib
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from qiskit import qpy
from threadpoolctl import threadpool_limits
from ttn_search.suitability.design import cohort,candidate_sets
from ttn_search.suitability.diagnostics import profile_build,describe,FEATURE_SETS
from ttn_search.scent import scent_tree,affinity_matrix
from ttn_search.trees import canonical,tree_hash,leaves
from ttn_search.reference import fidelity
from ttn_search.evaluate import Pipeline,write_json,software
from ttn_search.screening.verified_evaluator import evaluate_verified,prepare,scores,COMPILER_VERSION,COMPILER_HASH,EVALUATOR_HASH
from analyze_c1_depth import two_qubit_depth

OUT=Path('artifacts/problem_suitability_v1')
POLICY=Pipeline(tolerance_factors=(3.,1.,.1,0.))


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fresh(parent):
    i=0
    while (parent/f'execution_{i:03d}').exists():i+=1
    path=parent/f'execution_{i:03d}';path.mkdir(parents=True)
    write_json(path/'started.json',dict(utc=time.time(),previous_executions=i))
    return path


def freeze(targets):
    manifest_path=OUT/'manifest.json'
    if not manifest_path.exists():
        manifest=[]
        for index,(name,group,target,generating_tree) in enumerate(targets):
            start=time.perf_counter();scent,meta=scent_tree(target.state,0)
            before=time.perf_counter();independent_affinity=affinity_matrix(target.state);affinity_s=time.perf_counter()-before
            np.testing.assert_allclose(independent_affinity,meta['affinity'],atol=1e-12)
            pools=candidate_sets(scent,20261110+index)
            reference=tree_hash(generating_tree) if generating_tree is not None else None
            if generating_tree is not None:pools['trees'].setdefault(reference,generating_tree)
            spectrum=[]
            if generating_tree is not None:
                def visit(t):
                    if isinstance(t,int):return
                    ids=leaves(t)
                    if len(ids)<10:
                        other=[q for q in range(10) if q not in ids]
                        mat=target.state.reshape([2]*10,order='F').transpose(ids+other).reshape(2**len(ids),-1)
                        s=np.linalg.svd(mat,compute_uv=False)
                        spectrum.append(dict(labels=ids,values=s.tolist()))
                        if target.metadata['gamma']==0:
                            assert int(np.sum(s>1e-10))<=min(4,2**min(len(ids),10-len(ids)))
                    visit(t[0]);visit(t[1])
                visit(generating_tree)
            directory=OUT/name;directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/'target.npz',state=target.state)
            row=dict(name=name,group=group,target_hash=target.hash,target_metadata=target.metadata,
                target_file_hash=sha(directory/'target.npz'),pool_seed=20261110+index,pools=pools,
                generating_reference=reference,generating_cut_spectra=spectrum,scent_metadata=meta,affinity_diagnostic_s=affinity_s,
                generation_s=time.perf_counter()-start)
            write_json(directory/'design.json',row);manifest.append(row)
            print('freeze',index+1,'/12',name,len(pools['trees']),flush=True)
        assert len(manifest)==12 and sum(len(r['pools']['trees']) for r in manifest)<=1160
        write_json(manifest_path,manifest)
    manifest=read(manifest_path)
    for row,(_,_,target,_) in zip(manifest,targets):
        assert target.hash==row['target_hash'] and sha(OUT/row['name']/'target.npz')==row['target_file_hash']
    paths=sorted(str(p) for p in Path('src/ttn_search').rglob('*.py'))+[
        __file__,'scripts/report_problem_suitability.py','scripts/analyze_c1_depth.py','PROBLEM_SUITABILITY_PROTOCOL.md']
    protocol=dict(compiler_version=COMPILER_VERSION,compiler_hash=COMPILER_HASH,evaluator_hash=EVALUATOR_HASH,
        policy=asdict(POLICY),software=software(),manifest_hash=sha(manifest_path),files={p:sha(p) for p in paths},
        feature_sets=FEATURE_SETS,ridge_alpha=1.,folds='leave_one_base_group_out',training_weight='equal_total_weight_per_group',
        screening_budgets=[1,2,4,8],tie_seeds=[0,1,2],signal_threshold='max(20,0.05*pool_minimum)',
        signal_gate='At least two independent base groups have SCENT gap above threshold and padded K1 regret above threshold in all three tie orders, in at least one pool.',
        memory_method='Linux process lifetime ru_maxrss KiB sampled per target; includes evaluator and retained profile graphs; not phase-specific peak memory',
        profiling='Retain profile and reference Prepared representations for first-eight pool-prefix union; diagnostic profiling separate from formal labels; all features use first 3x.')
    protocol=json.loads(json.dumps(protocol));path=OUT/'protocol.json'
    if path.exists():assert read(path)==protocol,'Frozen inputs/source mismatch'
    else:write_json(path,protocol)
    return manifest,protocol


def run():
    targets=cohort();manifest,protocol=freeze(targets);results=[]
    total=sum(len(r['pools']['trees']) for r in manifest)
    for design,(_,_,target,_) in zip(manifest,targets):
        name=design['name'];affinity=np.array(design['scent_metadata']['affinity'])
        subset=set(design['pools']['R'][:8]+design['pools']['M'][:8]);retained=[];profile_rows=[]
        for key,tree in design['pools']['trees'].items():
            parent=OUT/name/'candidates'/key;path=parent/'result.json'
            if path.exists():row=read(path)
            else:
                tree=canonical(tree);start=time.perf_counter()
                item=prepare(target,tree,POLICY,3.);parameters,padded=scores(item)
                prepared_s=time.perf_counter()-start
                before=time.perf_counter();rep,diagnostic=profile_build(target.state,tree,(1-POLICY.fidelity)*POLICY.repr_fraction*3/9)
                assert item.representation is not None
                assert rep.ranks==item.representation.ranks
                assert fidelity(rep.state,item.representation.state)>1-1e-10
                np.testing.assert_allclose(rep.state,item.representation.state,atol=1e-10,rtol=0)
                descriptors=describe(target.state,rep,affinity,diagnostic)
                assert descriptors['parameters']==parameters and descriptors['padded']==padded
                feature_work_s=time.perf_counter()-before
                if key in subset:retained.append((rep,item))
                directory=fresh(parent)
                record=evaluate_verified(target,tree,POLICY,directory,item)
                depth=None
                if record['status']=='success':
                    with Path(record['circuit_path']).open('rb') as stream:circuit=qpy.load(stream)[0]
                    depth=two_qubit_depth(circuit)
                row=dict(target=name,group=design['group'],tree_hash=key,parameters=parameters,padded=padded,
                    initial_status=item.attempt['status'],status=record['status'],cx=record.get('cx_count'),d2=depth,
                    final_fidelity=record.get('fidelity_final'),accepted_factor=record['tolerance_factor'],
                    compiler_version=record['compiler_version'],compiler_calls=record['compiler_calls'],
                    record_path=str((directory/'record.json').resolve()),prepared_s=prepared_s,feature_work_s=feature_work_s,
                    diagnostic=diagnostic,descriptors=descriptors,in_R=key in design['pools']['R'],in_M=key in design['pools']['M'],
                    is_generating_reference=key==design['generating_reference'],
                    perturbation_distance=design['pools']['perturbation_distance'].get(key))
                write_json(path,row)
            results.append(row)
            if key in subset:profile_rows.append(row)
            write_json(OUT/'results.json',results)
            write_json(OUT/'status.json',dict(phase='evaluating',completed=len(results),total=total,target=name,utc=time.time()))
            print('evaluation',len(results),'/',total,name,row['status'],row['cx'],flush=True)
        before=time.perf_counter()
        for pool in ['R','M']:
            bykey={r['tree_hash']:r for r in results if r['target']==name}
            sorted(design['pools'][pool],key=lambda k:bykey[k]['padded'])
        write_json(OUT/name/'profile_summary.json',dict(subset=sorted(subset),profile_rows=[r['tree_hash'] for r in profile_rows],
            ranking_s=time.perf_counter()-before,retained_representations_this_execution=len(retained),
            process_lifetime_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            note='On resume, completed profile rows are retained as measurements, but their object graphs are not reconstructed; retained count exposes partial-session profiling. No phase-specific RSS claim.'))
    assert all(sha(p)==h for p,h in protocol['files'].items())
    write_json(OUT/'evaluations_complete.json',dict(evaluations=len(results),compiler_calls=sum(r['compiler_calls'] for r in results)))
    from report_problem_suitability import main as report
    report()
    write_json(OUT/'completion.json',dict(targets=12,base_groups=6,evaluations=len(results)))
    write_json(OUT/'status.json',dict(phase='complete',utc=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with threadpool_limits(limits=1):
            if args.freeze_only:
                manifest,_=freeze(cohort());print('Frozen evaluations:',sum(len(d['pools']['trees']) for d in manifest),flush=True)
            else:run()


if __name__=='__main__':main()
