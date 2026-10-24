"""Complete-pool audit and prespecified group-held-out diagnostic."""
import hashlib
import json
from pathlib import Path
import numpy as np
from qiskit import qpy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from ttn_search.evaluate import write_json
from analyze_c1_depth import two_qubit_depth

OUT=Path('artifacts/problem_suitability_v1')
def read(p):return json.loads(Path(p).read_text())


def retrieval(order,bykey,best,k):
    chosen=[bykey[key] for key in order[:k] if bykey[key]['status']=='success']
    low={key for key,r in bykey.items() if r['status']=='success' and r['cx']<=best*1.05}
    return dict(k=k,best_cx=min((r['cx'] for r in chosen),default=None),
        regret=min((r['cx'] for r in chosen),default=best)-best if chosen else None,
        low_cost_recall=len(set(order[:k])&low)/len(low) if low else None,
        feasible_selected=len(chosen))


def main():
    protocol=read(OUT/'protocol.json');manifest=read(OUT/'manifest.json');rows=read(OUT/'results.json')
    assert len(rows)==sum(len(d['pools']['trees']) for d in manifest)
    assert hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()==protocol['manifest_hash']
    for p,h in protocol['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    audit=dict(evaluations=len(rows),successful=0,synthesis_calls=0,retries=0,diagonals_corrected=0,euler_fallbacks=0,qpy_checks=0)
    records={}
    for row in rows:
        r=read(row['record_path']);records[(row['target'],row['tree_hash'])]=r
        design=next(d for d in manifest if d['name']==row['target'])
        assert r['target_hash']==design['target_hash'] and r['tree_hash']==row['tree_hash']
        assert r['compiler_version']==protocol['compiler_version'] and r['compiler_hash']==protocol['compiler_hash']
        assert r['evaluator_hash']==protocol['evaluator_hash'] and r['pipeline']==protocol['policy'] and not r['cache_hit']
        log=Path(row['record_path']).parent/'synthesis_starts.jsonl'
        starts=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        assert len(starts)==r['compiler_calls']
        assert [a['tolerance_factor'] for a in r['attempts']]==protocol['policy']['tolerance_factors'][:len(r['attempts'])]
        audit['synthesis_calls']+=r['compiler_calls'];audit['retries']+=len(r['attempts'])-1
        for a in r['attempts']:
            audit['diagonals_corrected']+=a.get('compiler_stats',{}).get('diagonals_corrected',0)
            audit['euler_fallbacks']+=a.get('compiler_stats',{}).get('euler_fallbacks',0)
        if r['status']=='success':
            assert r['fidelity_final']>=.999 and r['fidelity_repr']>=.9999-1e-12 and r['fidelity_comp']>=.9996-1e-12
            with Path(r['circuit_path']).open('rb') as stream:c=qpy.load(stream)[0]
            assert int(c.count_ops().get('cx',0))==row['cx'] and two_qubit_depth(c)==row['d2']
            audit['successful']+=1;audit['qpy_checks']+=1
    audit['all_logged_synthesis_starts']=sum(len(p.read_text().splitlines()) for p in OUT.rglob('synthesis_starts.jsonl'))
    audit['extra_starts_outside_selected_records']=audit['all_logged_synthesis_starts']-audit['synthesis_calls']
    write_json(OUT/'audit.json',audit)
    tables=[];retrievals=[];signal_groups=set();counterexamples=[]
    for d in manifest:
        allrows={r['tree_hash']:r for r in rows if r['target']==d['name']}
        for pool in ['R','M']:
            keys=d['pools'][pool];bykey={k:allrows[k] for k in keys};good=[r for r in bykey.values() if r['status']=='success']
            scent=bykey[keys[0]];best=min((r['cx'] for r in good),default=None)
            frontier=[r['tree_hash'] for r in good if not any(s['cx']<=r['cx'] and s['d2']<=r['d2'] and (s['cx']<r['cx'] or s['d2']<r['d2']) for s in good)]
            gap=scent['cx']-best if scent['status']=='success' and best is not None else None
            table=dict(target=d['name'],group=d['group'],pool=pool,feasible=len(good),scent_cx=scent['cx'],best_cx=best,
                minimum_d2=min((r['d2'] for r in good),default=None),scent_gap=gap,scent_relative_gap=gap/best if gap is not None and best else None,
                trees_beating_scent=sum(r['cx']<scent['cx'] for r in good) if scent['status']=='success' else None,frontier=frontier)
            tables.append(table)
            if best is None:continue
            padded_regrets=[]
            for score in ['padded','parameters']:
                for seed in protocol['tie_seeds']:
                    tie=np.random.default_rng(seed).permutation(len(keys)).tolist()
                    order=sorted([keys[i] for i in tie],key=lambda key:bykey[key][score])
                    result=dict(target=d['name'],group=d['group'],pool=pool,score=score,tie_seed=seed,
                        checkpoints=[retrieval(order,bykey,best,k) for k in protocol['screening_budgets']])
                    retrievals.append(result)
                    if score=='padded':padded_regrets.append(result['checkpoints'][0]['regret'])
                    if seed==0:
                        for a,b in zip(order,order[1:]):
                            if bykey[a]['status']==bykey[b]['status']=='success' and bykey[a]['cx']>bykey[b]['cx']:
                                counterexamples.append(dict(target=d['name'],pool=pool,score=score,earlier=a,later=b,
                                    delta=bykey[a]['cx']-bykey[b]['cx'],tied_score=bykey[a][score]==bykey[b][score],
                                    retry_involved=any(len(records[(d['name'],k)]['attempts'])>1 for k in [a,b])))
            threshold=max(20,.05*best)
            if gap is not None and gap>=threshold and all(v is not None and v>=threshold for v in padded_regrets):signal_groups.add(d['group'])
    write_json(OUT/'pool_metrics.json',tables);write_json(OUT/'retrieval.json',retrievals)
    write_json(OUT/'counterexamples.json',counterexamples)
    model_results=[]
    if len(signal_groups)>=2:
        eligible=[r for r in rows if (r['in_R'] or r['in_M']) and r['status']=='success']
        groups=sorted({r['group'] for r in eligible})
        for held in groups:
            train=[r for r in eligible if r['group']!=held];test=[r for r in rows if r['group']==held and (r['in_R'] or r['in_M'])]
            if not train or not test:continue
            train_groups=sorted({r['group'] for r in train})
            weights=np.array([len(train)/(len(train_groups)*sum(s['group']==r['group'] for s in train)) for r in train])
            for features in protocol['feature_sets']:
                x=np.array([[r['descriptors']['features'][k] for k in features] for r in train]);y=np.log1p([r['cx'] for r in train])
                scaler=StandardScaler().fit(x,sample_weight=weights)
                model=Ridge(alpha=1.).fit(scaler.transform(x),y,sample_weight=weights)
                pred=model.predict(scaler.transform([[r['descriptors']['features'][k] for k in features] for r in test]))
                predictions={(r['target'],r['tree_hash']):float(p) for r,p in zip(test,pred)}
                for d in manifest:
                    if d['group']!=held:continue
                    for pool in ['R','M']:
                        keys=d['pools'][pool];lookup={r['tree_hash']:r for r in test if r['target']==d['name']}
                        good=[r for r in lookup.values() if r['tree_hash'] in keys and r['status']=='success']
                        if not good:continue
                        best=min(r['cx'] for r in good)
                        for seed in protocol['tie_seeds']:
                            tie=np.random.default_rng(seed).permutation(len(keys)).tolist()
                            order=sorted([keys[i] for i in tie],key=lambda k:predictions[(d['name'],k)])
                            model_results.append(dict(held_group=held,training_groups=train_groups,target=d['name'],pool=pool,
                                features=features,tie_seed=seed,checkpoints=[retrieval(order,{k:lookup[k] for k in keys},best,k) for k in protocol['screening_budgets']]))
    write_json(OUT/'ridge_diagnostic.json',dict(gate_groups=sorted(signal_groups),gate_passed=len(signal_groups)>=2,results=model_results))
    lines=['# Problem-suitability development experiment','',
        'Twelve target settings in six base-generation groups; paired TTN spectrum/coupling variants are not independent targets. Ten qubits, fixed repaired compiler, unchanged strict 3/1/0.1/0 evaluation policy. No GNN training, routing, noise or ternary structures.', '',
        'All targets and both 64-tree pools were instantiated and fingerprinted before compilation. R = SCENT + 63 random trees. M = SCENT + the same first 31 random trees + 32 fixed one-/two-step SCENT perturbations. Shared labels are reused only within this offline exhaustive experiment, not credited as online speedups.', '',
        '## Pool quality','', '| Target | Group | Pool | Feasible / 64 | SCENT CX | Pool minimum CX | Minimum D2 | SCENT gap | Better trees |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for t in tables:lines.append(f"| {t['target']} | {t['group']} | {t['pool']} | {t['feasible']} | {t['scent_cx']} | {t['best_cx']} | {t['minimum_d2']} | {t['scent_gap']} | {t['trees_beating_scent']} |")
    lines+=['','Full Pareto sets and relative gaps are in pool_metrics.json. These are fixed-pool minima, not global state-preparation bounds. Generating-tree reference results are excluded unless that tree naturally appears in the pool.', '',
        '## Fixed-score retrieval','', '| Target | Pool | Score | Tie seed | Regret K=1 | K=2 | K=4 | K=8 |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in retrievals:lines.append('| '+f"{r['target']} | {r['pool']} | {r['score']} | {r['tie_seed']} | "+' | '.join(str(c['regret']) for c in r['checkpoints'])+' |')
    lines+=['','Recall of candidates within 5% of pool optimum and feasible retrieval counts are in retrieval.json. Infeasible selections consume budget; missing feasible results have null regret, never an invented cost.', '',
        '## Mechanism data and generating-tree references','',
        'Every candidate retains its local block sizes, first-attempt reduced construction spectra/discarded weights, actual physical-cut Schmidt spectra, affinity-cut summaries, source memberships and perturbation distance. These data include successes, failures, score ties and counterexamples. counterexamples.json lists adjacent cost inversions and retry flags; it is descriptive, not causal evidence.', '',
        '| Target | Reference in R | Reference in M | Status | CX | D2 |','|---|---|---|---|---:|---:|']
    for r in rows:
        if r['is_generating_reference']:lines.append(f"| {r['target']} | {r['in_R']} | {r['in_M']} | {r['status']} | {r['cx']} | {r['d2']} |")
    lines+=['','## Prespecified learning gate','',f"Independent groups meeting both the SCENT-headroom and stable padded-K1-error threshold: {sorted(signal_groups)}.", '',
        ('The gate passed; ridge_diagnostic.json contains the fixed three nested feature sets, leave-one-base-group-out retrieval results, and explicit training groups. No feature tuning or GNN training was performed.' if len(signal_groups)>=2 else 'The gate did not pass. No Ridge fitting or GNN training was performed.'), '',
        'The threshold is max(20 CNOT, 5% of pool minimum), used as an exploratory practical criterion, not a significance test. Ridge uses equal total sample weight per training group (weights normalized to sample count), training-only weighted standardization, alpha=1, and log(1+CNOT). Feasible costs alone train the regression, while held-out rankings include all pool candidates; no feasibility predictor is claimed.', '',
        '## Construction profiling','',
        '| Target | Setup | Projection incl. Kronecker | SVD | Assembly | Reconstruction | Isometry check | Verification | Unclassified | Descriptor extraction |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in manifest:
        keys=set(d['pools']['R'][:8]+d['pools']['M'][:8]);subset=[r for r in rows if r['target']==d['name'] and r['tree_hash'] in keys]
        parts=[sum(r['diagnostic']['timing'][k] for r in subset) for k in ['setup','projection','svd','assembly','reconstruction','isometry_check','verification','unclassified']]
        parts.append(sum(r['descriptors']['descriptor_s'] for r in subset))
        lines.append('| '+d['name']+' | '+' | '.join(f'{v:.6f}' for v in parts)+' |')
    lines+=['','Times above sum the prespecified first-eight-prefix union (the shared prefix makes this eight candidates here). Instrumented and original states/ranks are checked per candidate. Prepared and profile representations are retained across this subset in an uninterrupted target session. profile_summary.json records ranking time, retained counts and Linux process-lifetime peak RSS; RSS includes other work and is not a per-phase or isolated-screening peak. Projection includes Kronecker-product construction. Spectral features require additional SVDs timed as descriptor extraction; they are not free topology features.', '',
        'These are diagnostic component timings, not online speedup measurements. Formal evaluation record times remain separate. Small cache savings do not bound the potential benefit of a learned preselector; any later speed claim must include target summaries, inference, shortlisted builds and wrong-selection costs.', '',
        '## Audit','',f"{audit['evaluations']} completed candidate evaluations; {audit['successful']} feasible; {audit['synthesis_calls']} synthesis starts; {audit['retries']} retries; {audit['diagonals_corrected']} corrected diagonals; {audit['euler_fallbacks']} Euler fallbacks. Independent resource checks: {audit['qpy_checks']} QPY circuits. Extra starts outside selected records: {audit['extra_starts_outside_selected_records']}.", '',
        'All target settings remain in the report. Any failure is retained in results.json with its full evaluator record. The complete manifest includes all generating metadata for audit, but predictive features do not include generating-tree identity, seed, alpha or gamma. Main experiment and later ternary preflight remain separate.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('Audited report written',OUT/'REPORT.md',flush=True)


if __name__=='__main__':main()
