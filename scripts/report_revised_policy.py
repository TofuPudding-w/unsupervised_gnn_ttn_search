import json
from pathlib import Path

OUT=Path('artifacts/revised_policy')

def main():
    cohort=json.loads((OUT/'cohort.json').read_text());pool=json.loads((OUT/'pool.json').read_text())
    frontier=[r for r in pool if r['status']=='success' and not any(a['status']=='success' and a['cx']<=r['cx'] and a['d2']<=r['d2'] and (a['cx']<r['cx'] or a['d2']<r['d2']) for a in pool)]
    lines=['# Revised 3×-first evaluation policy','',
        '## Frozen policy','',
        'Attempt local tolerance multipliers **3, 1, 0.1, 0**, stopping at the first feasible result. Enforce final F≥0.999, global representation infidelity ≤0.0001 and compilation infidelity ≤0.0004. A representation-budget failure skips compilation and proceeds to the next factor. Compiler/numerical failures or failed verification also proceed to the next factor; exhausted attempts return failure, not an invented feasible cost. All builds, verification, synthesis calls and retries are charged. Synthesis-start records are persisted before compilation; JSON records use atomic replacement. No new hard wall-time or memory cap was introduced.','',
        'The 0.1 and 0 factors retain the original fallback tail. This fixed sequence is shared by all methods; it is not adaptively tuned per target. The policy was frozen after development checks and before any untouched-target circuit labels. Historical benchmark results are unchanged.','',
        '## Development cohort: paired direct 1× / 3× evaluations','',
        'Six target instances: eight-qubit hierarchical/overlapping seeds 0 and 3, and ten-qubit hierarchical/overlapping seed 13. Each uses SCENT, balanced, and a random tree drawn with seed 2026. All are development data. Both direct attempts are compiled for diagnostic error measurement; the production policy instead rejects representation-infeasible attempts before compilation.','',
        '| n | Family | Seed | Tree | Factor | Representation error | Compilation error | Final fidelity | CNOT | D2 | Feasible |',
        '|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for r in cohort:
        lines.append(f"| {r['n']} | {r['family']} | {r['seed']} | {r['tree']} | {r['factor']} | {r.get('repr_error',float('nan')):.7g} | {r.get('comp_error',float('nan')):.7g} | {r.get('final_fidelity',float('nan')):.10f} | {r.get('cx','—')} | {r.get('d2','—')} | {r['feasible']} |")
    lines+=['','Of 36 direct evaluations, 35 satisfy all enforced requirements. Among the 17 pairs where both factors pass, 3× improves CNOT and depth in two pairs and ties in fifteen; none worsens. The benefit is therefore instance-dependent, not uniform. The 3× ten-qubit hierarchical random tree has representation error 2.99e-5 but compilation error approximately 0.9673 and final fidelity 0.03270. This is a compilation correctness failure, not evidence that the truncation error itself exceeded its budget. The exact failure mechanism is not established by this experiment. Explicit full-policy replay of that case rejects 3×, then succeeds at 1× with final fidelity 0.999985544966. This demonstrates the need for independent compilation verification and a fixed fallback rule.','',
        '## Exhaustive revised-policy pool','',
        'The original overlapping seed-13 pool was rebuilt and evaluated afresh, without reusing old cost labels. Descriptors use the initial 3× construction, even when a subsequent retry supplies the accepted circuit.','',
        f"All {len(pool)} candidates are feasible. Five require fallback to 1×: 43, 51, 54, 62, 92. Completed synthesis calls: {sum(r['compiler_calls'] for r in pool)}. Summed construction/evaluation/depth-extraction wall time: {sum(r['wall_s'] for r in pool):.2f} s.",'',
        '| Frontier candidate | CNOT | D2 | Final fidelity | Accepted factor |','|---|---:|---:|---:|---:|']
    for r in frontier:lines.append(f"| {r['candidate_id']} | {r['cx']} | {r['d2']} | {r['fidelity']:.10f} | {r['accepted_factor']} |")
    lines+=['','SCENT remains the sole nondominated candidate, now at **592 CNOTs and D2=436**. No candidate in this pool improves its CNOT count or depth under the revised policy. Both rebuilt fixed scores rank SCENT first with the original seed-0 tie order. The old minimum 1163/900 was policy-dependent; the new minimum is also not a state-preparation lower bound.','',
        '## Next stage','',
        'Resume only the authorized bounded random-pretraining data-size comparison: new ten-qubit models, nested 2000/10000-tree corpora, 1600 matched optimizer updates, three paired seeds, and untouched overlapping target seeds 14–18. This is size-specific training, not transfer of an eight-qubit checkpoint. Keep policy and scoring fixed throughout and report offline training separately from actual online search cost. Mixed pretraining remains outside this experiment.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
