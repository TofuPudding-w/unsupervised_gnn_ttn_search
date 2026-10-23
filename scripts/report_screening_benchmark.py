"""Summarize actual completed-result prefixes at equal query, call, and time budgets."""
import json
from pathlib import Path
import numpy as np

OUT=Path('artifacts/screening_benchmark')


def at_budget(run,budget,kind):
    key={'candidate':'k','compiler':'compiler_calls','time':'elapsed_s'}[kind]
    eligible=[r for r in run['trace'] if r[key]<=budget]
    return eligible[-1]['best_cx'] if eligible else None


def cell(values):
    solved=[v for v in values if v is not None]
    mean=f'{np.mean(solved):.2f}' if solved else '—'
    return f'{mean} ({len(solved)}/{len(values)})'


def main():
    config=json.loads((OUT/'protocol.json').read_text())['config']
    runs=json.loads((OUT/'runs.json').read_text())
    done=json.loads((OUT/'completion.json').read_text())
    lines=['# Fresh end-to-end screening benchmark','',
        'Ten previously untouched eight-qubit instances: hierarchical and overlapping seeds 8–12. Each search method receives the same SCENT-plus-127-random-tree pool. Random trees use unique pair merging; no balanced/comb/neighbor prefix is inserted. SCENT alone constructs and evaluates only its tree. Three independently executed repetitions per method and target; method order is randomized before labels are observed.','',
        'Every method repetition independently generates candidates, constructs its required TTNs, synthesizes circuits, and verifies fidelity. There is no cross-method outcome or TTN cache. Parameter/padded screening constructs all 128 trees once, ranks without any circuit labels, and passes its own initial representations to compilation. Only required retries rebuild them.','',
        'The random_scent control starts with SCENT, then visits a random permutation. Plain random has no forced SCENT start. Frozen GNN BO uses extended model seeds 0/1/2 for repetitions 0/1/2, starts with SCENT, and uses the existing constrained BO after observations become available; it has no forced eight-circuit initialization. The GNN models and scaler training data were frozen before new target evaluation.','',
        '## Equal candidate-evaluation budget','',
        'K counts candidate opportunities, including failures; a candidate may require more than one compiler call. Cells show mean best feasible CNOT and completed-feasible runs / total runs. Means are conditional on a result being available, so compare availability before conditional CNOT at tight budgets.','',
        '| Family | Method | K=1 | K=2 | K=4 | K=8 | K=16 |','|---|---|---|---|---|---|---|']
    for family in config['families']:
        for method in config['methods']:
            rows=[r for r in runs if r['family']==family and r['method']==method]
            lines.append('| '+family+' | '+method+' | '+' | '.join(cell([at_budget(r,b,'candidate') for r in rows]) for b in config['candidate_budgets'])+' |')
    lines+=['','## Equal actual compiler-call budget','',
        'Each attempted synthesis, including failed synthesis and precision retries, counts. A verified result is credited only when the candidate has completed within the call budget; a candidate spanning the boundary cannot contribute its later result.','',
        '| Family | Method | Calls=1 | Calls=2 | Calls=4 | Calls=8 | Calls=16 |','|---|---|---|---|---|---|---|']
    for family in config['families']:
        for method in config['methods']:
            rows=[r for r in runs if r['family']==family and r['method']==method]
            lines.append('| '+family+' | '+method+' | '+' | '.join(cell([at_budget(r,b,'compiler') for r in rows]) for b in config['compiler_call_budgets'])+' |')
    lines+=['','## Equal elapsed time','',
        'Wall time starts before SCENT affinity/clustering and pool generation. Screening must pay for whole-pool construction and ranking before its first result. Model loading/embedding, actual compilation, retries, verification, and artifact writing performed before the result are included. No future result or interpolated result is credited. All methods are capped at 16 candidate evaluations; finished runs retain their incumbent without spending remaining time on further candidates. SCENT alone ends after one evaluation.','',
        '| Family | Method | 0.5 s | 1 s | 2 s | 4 s | 8 s | 16 s |','|---|---|---|---|---|---|---|---|']
    for family in config['families']:
        for method in config['methods']:
            rows=[r for r in runs if r['family']==family and r['method']==method]
            lines.append('| '+family+' | '+method+' | '+' | '.join(cell([at_budget(r,b,'time') for r in rows]) for b in config['time_budgets_s'])+' |')
    lines+=['','## Per-target final endpoints and actual run time','',
        'Means over the three repetitions. SCENT uses one evaluation; search methods use sixteen. Repetitions are not independent target samples.','',
        '| Target | Method | Mean CNOT | Mean wall seconds | Mean compiler calls |','|---|---|---|---|---|']
    for target in dict.fromkeys(r['target'] for r in runs):
        for method in config['methods']:
            rows=[r for r in runs if r['target']==target and r['method']==method]
            lines.append('| '+target+' | '+method+' | '+cell([r['trace'][-1]['best_cx'] for r in rows])+f" | {np.mean([r['actual_wall_s'] for r in rows]):.3f} | {np.mean([r['compiler_calls'] for r in rows]):.2f} |")
    lines+=['','## Timing components','',
        'Seconds per complete run, averaged by family/method. Construction includes screened-but-uncompiled trees. Verification separates representation checks and circuit-state checks. Retry time overlaps construction/compilation/verification and must not be added again. File serialization and miscellaneous allocation/logging overhead are included in wall time; phase timers need not sum exactly to it.','',
        '| Family | Method | Generation | Construction | Ranking/proposals | Model setup | Compile | Representation checks | Circuit checks | Retry (overlap) | Wall |','|---|---|---|---|---|---|---|---|---|---|---|']
    fields=['generation_s','construction_s','ranking_s','model_setup_s','compile_s','representation_verification_s','circuit_verification_s','retry_s','actual_wall_s']
    for family in config['families']:
        for method in config['methods']:
            rows=[r for r in runs if r['family']==family and r['method']==method]
            lines.append('| '+family+' | '+method+' | '+' | '.join(f'{np.mean([r[k] for r in rows]):.4f}' for k in fields)+' |')
    lines+=['','## Scope and accounting','',
        f"Completed {done['runs']} method runs, {done['candidate_evaluations']} candidate evaluations, and {done['actual_compiler_calls']} actual synthesis calls. Summed method wall time: {done['summed_method_wall_s']:.1f} s. This invocation wall time: {done['invocation_wall_s']:.1f} s.", '',
        f"Outside per-method clocks: scaler export from old training trees {done['offline_scaler_preparation_s']:.3f} s; shared warmup on an old development target {done['old_target_warmup_s']:.3f} s; target-state creation {done['target_state_creation_s']:.3f} s. Imports and one uncompiled reference-pool consistency generation per target are also outside per-method clocks; each actual run generates its own pool inside its clock. Prior GNN training is an offline cost, not charged again as online time.", '',
        'The optimum over all 128 trees is unknown: only selected candidates were compiled. Measurements apply to this machine, eight-qubit dense construction, existing target families, and the reference isometry compiler. They do not establish transfer to larger systems or the SCENT approximate compiler. Shared libraries/OS caches may remain warm despite independent circuit evaluation; randomized method order and repeated fresh runs mitigate ordering effects but do not remove all timing noise. Stage C larger/mixed pretraining remains on hold.','']
    lines+=['## Interpretation','',
        'Padded-size screening improves early selection on the untouched overlapping targets: mean CNOT is 352.00 at K=1 and 345.93 at K=4, versus 456.20 and 372.93 for parameter count. At K=1 it wins on three targets and ties on two; at K=4 it wins on two and ties on three. Both screeners reach the same per-target endpoints by K=16 (mean 339.80). The evidence supports earlier useful selection, not a better final shortlist endpoint than parameter count.','',
        'At two seconds every overlapping run has a result. Padded screening averages 347.33 CNOT, parameter count 428.00, random 412.07, SCENT-initialized random 480.53, and frozen GNN BO 495.67. Whole-pool construction is included. Padded screening beats parameter count on three targets and ties on two at this deadline. At one second completion fractions differ, so conditional CNOT alone is not a fair comparison.','',
        'On hierarchical targets SCENT already averages 75.8 CNOT; screeners reach 75.2 by K=2. SCENT finishes within 0.5 seconds in all hierarchical runs, whereas neither screener has a result then. Screening is not uniformly preferable when the starting tree is already inexpensive.','',
        'Keep larger/mixed pretraining on hold. The next scientific question is transfer to a modestly larger system or materially different compiler; the current results establish neither. The five targets per family, not the fifteen repetitions, determine independent target coverage.','']
    audit_path=OUT/'audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        lines+=['## Independent artifact audit','',
            f"Status: {audit['status']}. Checked {audit['runs']} runs and {audit['candidate_evaluations']} candidate evaluations; {audit['retry_attempts']} retry attempts; minimum verified final fidelity {audit['minimum_fidelity']:.12f}.",'']
        if audit.get('source_review'):
            lines+=['A concurrent documentation edit changed the on-disk reference.py hash after execution began. The exact original core bytes were recovered and matched the frozen combined hash; an AST comparison confirmed that current executable code is unchanged after removing docstrings. The running process retained its imported code. See source_snapshot/, source_review.json and audit.json. A protocol resume still intentionally rejects changed source bytes.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    # Standalone scientific plot of verified anytime results, no interpolated early success.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in config['methods']:
        rows=[r for r in runs if r['family']=='overlapping' and r['method']==method]
        values=[np.mean([at_budget(r,b,'candidate') for r in rows if at_budget(r,b,'candidate') is not None]) for b in config['candidate_budgets']]
        axes[0].plot(config['candidate_budgets'],values,marker='o',label=method)
        times=config['time_budgets_s']
        vals=[]
        for t in times:
            available=[at_budget(r,t,'time') for r in rows]
            vals.append(np.mean(available) if all(v is not None for v in available) else np.nan)
        axes[1].plot(times,vals,marker='o',label=method)
    axes[0].set(xlabel='Candidate evaluations',ylabel='Mean best verified CNOT',title='Overlapping targets: equal candidate budget')
    axes[1].set(xlabel='Actual elapsed seconds',ylabel='Mean best verified CNOT',title='Time: shown only when every run has a result')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
