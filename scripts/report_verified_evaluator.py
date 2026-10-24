"""Audit and summarize the separately versioned repaired-compiler experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np
from qiskit import qpy
from ttn_search.evaluate import write_json
from analyze_c1_depth import two_qubit_depth

OUT=Path('artifacts/verified_evaluator_v1')


def read(path):
    return json.loads(Path(path).read_text())


def mean_quality(values):
    return 'infeasible' if any(v is None for v in values) else f'{np.mean(values):.2f}'


def main():
    completion=read(OUT/'completion.json');protocol=read(OUT/'protocol.json')
    runs=read(OUT/'runs.json');pools=read(OUT/'pool_results.json')
    assert len(runs)==105 and len(pools)==768
    expected={(f'overlapping_{s}',m,r) for s in range(14,19) for m in protocol['config']['methods'] for r in range(3)}
    assert {(r['target'],r['method'],r['repeat']) for r in runs}==expected
    assert {(r['target'],r['candidate_id']) for r in pools}=={(f'overlapping_{s}',i) for s in range(13,19) for i in range(128)}
    for group in ['files','models','pool_files']:
        for p,h in protocol[group].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    paths={r['record_path'] for r in pools}
    paths.update(t['record_path'] for r in runs for t in r['trace'])
    starts=0;retries=0;corrections=0;fallbacks=0;min_fidelity=1.;failed=0
    records={}
    for path in sorted(paths):
        record=read(path);records[path]=record
        assert record['compiler_version']==protocol['compiler_version']
        assert record['compiler_hash']==protocol['compiler_hash'] and record['evaluator_hash']==protocol['evaluator_hash']
        assert record['pipeline']==protocol['policy'] and not record['cache_hit']
        log=Path(path).parent/'synthesis_starts.jsonl'
        logs=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert len(logs)==record['compiler_calls']==sum(a['compile_attempted'] for a in record['attempts'])
        assert [r['tolerance_factor'] for r in logs]==[a['tolerance_factor'] for a in record['attempts'] if a['compile_attempted']]
        assert [a['tolerance_factor'] for a in record['attempts']]==protocol['policy']['tolerance_factors'][:len(record['attempts'])]
        starts+=len(logs);retries+=len(record['attempts'])-1
        for attempt in record['attempts']:
            stats=attempt.get('compiler_stats',{})
            corrections+=stats.get('diagonals_corrected',0);fallbacks+=stats.get('euler_fallbacks',0)
        if record['status']=='success':
            assert record['fidelity_final']>=.999 and record['fidelity_repr']>=.9999-1e-12 and record['fidelity_comp']>=.9996-1e-12
            min_fidelity=min(min_fidelity,record['fidelity_final'])
        else:failed+=1
    for run in runs:
        ids=[];best=None;calls=0;last=0
        for t in run['trace']:
            record=records[t['record_path']];ids.append(t['candidate_id']);calls+=record['compiler_calls']
            assert record['tree_hash']==t['tree_hash']
            assert record['target_hash']==read(OUT/run['target']/'pool.json')['target_hash']
            if record['status']=='success':best=min(best,record['cx_count']) if best is not None else record['cx_count']
            assert t['best_cx']==best and t['compiler_calls']==calls
            assert t['elapsed_s']>=last;last=t['elapsed_s']
        assert len(ids)==len(set(ids))==run['candidate_evaluations']
        # Independently recompute the final incumbent's resources.
        final=run['trace'][-1]
        if final['best_candidate_id'] is not None:
            record=records[next(t['record_path'] for t in run['trace'] if t['candidate_id']==final['best_candidate_id'])]
            with Path(record['circuit_path']).open('rb') as stream:circuit=qpy.load(stream)[0]
            assert int(circuit.count_ops().get('cx',0))==final['best_cx']
            assert two_qubit_depth(circuit)==final['best_d2']
    for row in pools:
        record=records[row['record_path']]
        assert record['tree_hash']==row['tree_hash']
        assert record['target_hash']==read(OUT/row['target']/'pool.json')['target_hash']
        if row['status']=='success':
            with Path(record['circuit_path']).open('rb') as stream:circuit=qpy.load(stream)[0]
            assert int(circuit.count_ops().get('cx',0))==row['cx']
            assert two_qubit_depth(circuit)==row['d2']
    all_starts=sum(len(p.read_text().splitlines()) for p in OUT.rglob('synthesis_starts.jsonl'))
    abandoned_starts=all_starts-starts
    assert abandoned_starts>=0
    interrupted=[str(p.parent) for p in OUT.rglob('started.json') if not (p.parent/'summary.json').exists() and not (p.parent/'record.json').exists()]
    audit=dict(record_count=len(paths),synthesis_starts=starts,retries=retries,diagonals_corrected=corrections,
        euler_fallbacks=fallbacks,failed_evaluations=failed,minimum_final_fidelity=min_fidelity,
        all_synthesis_starts=all_starts,additional_starts_outside_selected_records=abandoned_starts,
        incomplete_executions=interrupted,incumbent_qpy_checks=105,pool_qpy_checks=sum(r['status']=='success' for r in pools))
    write_json(OUT/'audit.json',audit)
    lines=['# Separately versioned repaired-compiler evaluation','',f"Compiler: `{protocol['compiler_version']}`.", '',
        'The prior targets/pools, frozen GNN checkpoints, seeds and strict 3→1→0.1→0 accuracy policy are retained. No training or tuning on these outcomes occurs. These are development targets, not a new held-out cohort.', '',
        'Online comparisons are fresh and independent (no outcome cache), with all builds, screening, proposals, synthesis, verification and retries charged. The offline exhaustive audit runs afterward and reuses only outcomes from this compiler version. Offline work is not charged to online selection or used to guide it. These are fixed-evaluation comparisons with measured elapsed times, not uncapped fixed-time experiments.', '',
        '## Online method comparison','',
        '| Method | K=1 | K=2 | K=4 | K=8 | K=16 | Final incumbent D2 | Mean online seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in protocol['config']['methods']:
        group=[r for r in runs if r['method']==method]
        values=[mean_quality([r['trace'][min(k,len(r['trace']))-1]['best_cx'] for r in group]) for k in [1,2,4,8,16]]
        lines.append('| '+method+' | '+' | '.join(values)+f" | {mean_quality([r['trace'][-1]['best_d2'] for r in group])} | {np.mean([r['actual_wall_s'] for r in group]):.3f} |")
    lines+=['','Means cover five independent targets and three repetitions. Depth belongs to the minimum-CNOT incumbent; this is not a depth-constrained search. Frozen training cost is not re-incurred here and is not included in these online times.', '',
        '## Exhaustive pool outcomes','',
        '| Target | Feasible / 128 | SCENT CX | Best CX | Best D2 | Trees beating SCENT CX | Padded K=1 regret | Parameter K=1 regret | Pareto candidates |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    frontiers={}
    for seed in range(13,19):
        rows=[r for r in pools if r['target']==f'overlapping_{seed}'];good=[r for r in rows if r['status']=='success']
        scent=next(r for r in rows if r['candidate_id']==0)
        best=min(r['cx'] for r in good);depth=min(r['d2'] for r in good)
        frontier=[r['candidate_id'] for r in good if not any(s['cx']<=r['cx'] and s['d2']<=r['d2'] and (s['cx']<r['cx'] or s['d2']<r['d2']) for s in good)]
        frontiers[str(seed)]=frontier
        tie=np.random.default_rng(0).permutation(128).tolist();byid={r['candidate_id']:r for r in rows}
        selected=[min(tie,key=lambda i:byid[i][score] if byid[i][score] is not None else float('inf')) for score in ['padded','parameters']]
        regret=[byid[i]['cx']-best if byid[i]['status']=='success' else 'infeasible' for i in selected]
        lines.append(f"| overlapping_{seed} | {len(good)} | {scent['cx']} | {best} | {depth} | {sum(r['cx']<scent['cx'] for r in good)} | {regret[0]} | {regret[1]} | {frontier} |")
    write_json(OUT/'frontiers.json',frontiers)
    lines+=['','Pool minima are within this fixed candidate set and evaluator, not state-preparation lower bounds. Screening tie order above is frozen seed 0. Every candidate, including failed candidates, remains in the audit.', '',
        '## Per-target online endpoints','',
        '| Target | Method | Mean final CX | Mean final D2 | Mean online seconds |','|---|---|---:|---:|---:|']
    for seed in range(14,19):
        for method in protocol['config']['methods']:
            group=[r for r in runs if r['target']==f'overlapping_{seed}' and r['method']==method]
            lines.append(f"| overlapping_{seed} | {method} | {mean_quality([r['trace'][-1]['best_cx'] for r in group])} | {mean_quality([r['trace'][-1]['best_d2'] for r in group])} | {np.mean([r['actual_wall_s'] for r in group]):.3f} |")
    old_runs=read('artifacts/revised_gnn/runs.json')
    old_by_key={(r['target'],r['method'],r['repeat']):r for r in old_runs}
    delta=[]
    for run in runs:
        old=old_by_key[(run['target'],run['method'],run['repeat'])]
        delta.append(dict(target=run['target'],method=run['method'],repeat=run['repeat'],old_cx=old['trace'][-1]['best_cx'],new_cx=run['trace'][-1]['best_cx'],
            old_d2=old['trace'][-1]['best_d2'],new_d2=run['trace'][-1]['best_d2']))
    write_json(OUT/'historical_comparison.json',delta)
    lines+=['','## Audit and limitations','',
        f"Audited {len(paths)} distinct fresh evaluation records, {starts} synthesis starts, {retries} retries, {corrections} corrected diagonals and {fallbacks} local Euler fallbacks. Failed completed evaluations: {failed}. Minimum successful final fidelity: {min_fidelity:.12f}. Independent QPY resource checks cover 105 online incumbents and every successful pool candidate.", '',
        f"Additional synthesis starts outside selected completed records: {abandoned_starts}. Incomplete execution directories: {len(interrupted)}. They are preserved rather than counted as completed benchmark runs; any such work is additional resource usage and prevents interpreting selected-run timings as full from-zero execution cost.", '',
        'Historical endpoint changes are recorded in historical_comparison.json. Old and new labels are never combined. Differences can arise from repaired first-attempt circuits, avoided fallback, or changed adaptive BO observations. These targets are already development data; no broad unseen-target or alternative-compiler generalization is established.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('Report and audit written:',OUT/'REPORT.md',flush=True)


if __name__=='__main__':main()
