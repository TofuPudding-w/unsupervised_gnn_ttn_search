"""Report the frozen accuracy/compiler probe and audit its saved results."""
import json
from pathlib import Path

OUT=Path('artifacts/compilation_probe')

def main():
    rows=json.loads((OUT/'results.json').read_text())
    assert len(rows)==json.loads((OUT/'completion.json').read_text())['evaluations']==21
    good=[r for r in rows if r['status']=='success']
    for r in good:
        assert r['feasible']==(r['fidelity_final']>=.999)
        assert sum(b['critical_chain_cx'] for b in r['blocks'])==r['assembly_d2']
        assert r['assembly_fidelity']>.999999
    lines=['# Ten-qubit accuracy-budget and compilation diagnostic','',
        '## Scope','',
        'Development target: overlapping seed 13. Frozen pool candidates: 0 (SCENT), 74 (original padded-score first choice), and 1 (fixed random-pool example). This is a small mechanism experiment, not held-out validation. The existing benchmark and its precision policy are unchanged.','',
        'Each representation uses local discarded-weight tolerance (0.0001/9) × multiplier. Reference multipliers are 1, 3, 10, 30, 100; the alternative is tested at 1 and 10. No precision retries occur. Final fidelity ≥0.999 determines diagnostic acceptance; compliance with the original representation/compilation sub-budgets is reported separately. A 120-second ceiling applies to each tree/compiler/tolerance combination.','',
        '## Main result','',
        '**The same SCENT tree becomes substantially shallower by changing truncation tolerance.** At multiplier 3, it falls from 1163 to 592 CNOTs and from D2=900 to D2=436, with final fidelity 0.99996344. This setting also meets the original global representation/compilation sub-budgets. The nominal local tolerance was conservative for this target; no tree change is needed for this reduction.','',
        'The earlier exhaustive optimum of 1163 CNOTs and minimum depth 900 were specific to the original evaluation policy. They are not state-preparation lower bounds. The present sweep does not establish the optimum of all 128 trees under a revised tolerance policy.','',
        '## Reference tolerance sweep','',
        '| Candidate | Multiplier | Final fidelity | CNOT | D2 | Max rank | Padded entries | Final feasible | Original budgets feasible |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        if r['backend']!='reference':continue
        if r['status']!='success':lines.append(f"| {r['candidate']} | {r['factor']} | {r['status']} | — | — | — | — | — | — |");continue
        lines.append(f"| {r['candidate']} | {r['factor']} | {r['fidelity_final']:.10f} | {r['cx']} | {r['d2']} | {max(r['ranks'])} | {r['padded_entries']} | {r['feasible']} | {r['original_budget_feasible']} |")
    lines+=['','Factor 100 fails final fidelity for all three trees, so its gate savings are not admissible. Factors 10 and 30 can pass the final requirement while violating the original conservative representation allocation. They are a proposed change in evaluation policy, not improvements under an identical policy. Gate counts need not vary monotonically with tolerance because tensor entries and synthesis simplification also change.','',
        '## Where the SCENT depth originates','',
        'Internal blocks are independently compiled and assembled using the same register allocation and physical-output mapping. A longest CX-dependency chain is traced through that assembly. Contributions below belong to one such chain; ties can yield other chains. The assembly and globally compiled reference both have D2=900 at factor 1 and 436 at factor 3 in this case. Block attribution is not generally an exact partition of the globally optimized circuit.','',
        '| Factor | Block leaves | Rank | Child ranks | Input/output bits | Local CNOT | Local D2 | CX on assembly critical chain |',
        '|---|---|---:|---|---|---:|---:|---:|']
    for factor in [1,3]:
        r=next(r for r in good if r['candidate']==0 and r['backend']=='reference' and r['factor']==factor)
        for b in sorted(r['blocks'],key=lambda x:x['critical_chain_cx'],reverse=True):
            lines.append(f"| {factor} | {b['leaves']} | {b['rank']} | {b['child_ranks']} | {b['input_bits']}/{b['output_bits']} | {b['cx']} | {b['d2']} | {b['critical_chain_cx']} |")
    lines+=['','The six-leaf block contributes 535→257 chain layers; the root contributes 240→54. These reductions sum to 464, accounting for the observed 900→436 change in assembly depth. The root child ranks cross 9→8, reducing its output register from eight to six qubits. The six-leaf block input register drops from four to three qubits. Large local synthesis blocks on a shared-wire dependency chain dominate this example.','',
        'At nominal tolerance the sum of independent block depths is 1144, but the assembled circuit depth is 900; at factor 3 the corresponding values are 582 and 436. Thus the depth calculation already permits parallelism between disjoint operations. Merely scheduling the same fixed gate-dependency graph cannot reduce its ASAP critical-path depth. Commutation-aware transformations or a different gate decomposition could change that graph, but were not tested here.','',
        '## Alternative compiler: full-unitary completion plus QSD','',
        'Each padded isometry is completed with a null-space basis to a square unitary; Qiskit Quantum Shannon decomposition synthesizes that unitary. The same wire allocation, all-to-all connectivity, u/cx gate set, optimization level 1 and seed 17 are used. This changes the synthesis method, rather than only changing a transpiler setting. It is exact synthesis of a completed unitary, not an approximate SCENT compiler.','',
        '| Candidate | Multiplier | Reference CNOT / D2 | QSD CNOT / D2 | QSD final fidelity |',
        '|---|---:|---|---|---:|']
    for r in rows:
        if r['backend']!='unitary_qsd':continue
        ref=next(a for a in rows if a['backend']=='reference' and a['candidate']==r['candidate'] and a['factor']==r['factor'])
        value=f"{r['cx']} / {r['d2']}" if r['status']=='success' else r['status']
        ff=f"{r['fidelity_final']:.10f}" if r['status']=='success' else '—'
        lines.append(f"| {r['candidate']} | {r['factor']} | {ref.get('cx')} / {ref.get('d2')} | {value} | {ff} |")
    lines+=['','Full-unitary completion is substantially worse in these comparisons. It specifies action on input subspaces that the state-preparation task never uses; its overhead is therefore unsurprising. This rules out this particular completion/synthesis route as an improvement here, not other isometry-specific or approximate compilers.','',
        '## Diagnostic timing and validation','',
        '| Candidate | Backend | Multiplier | Build (s) | Compilation plus block analysis (s) | Verification (s) |',
        '|---|---|---:|---:|---:|---:|']
    for r in good:lines.append(f"| {r['candidate']} | {r['backend']} | {r['factor']} | {r['build_s']:.4f} | {r['compile_and_analysis_s']:.4f} | {r['verification_s']:.4f} |")
    lines+=['','The compilation timer includes independently synthesizing local blocks for analysis and then compiling the final circuit. For the reference it therefore includes duplicated synthesis work. These are diagnostic overhead measurements, not fair production compiler-latency comparisons. Timings also include process/machine variability; no speedup claim is based on them.','',
        f"Completed {len(rows)} configurations; {len(good)} returned circuits. Statevectors verify the final and representation fidelities. Block assemblies reproduce their representation with fidelity above 0.999999. Four unit tests cover both compilers on complex states and two tree shapes, physical-label mapping, and critical-chain depth accounting.",'',
        '## Decision','',
        'Prioritize validation of a revised tolerance policy on development targets, then freeze it before untouched-target evaluation. Factor 3 is a promising development setting, not a universal prescription; all candidate trees must receive the same predeclared policy and final verification. Rebuild screening descriptors under that policy and recheck ranking/headroom. Do not silently replace historical results or compare a relaxed-policy selector against a stricter SCENT baseline.','',
        'A depth-constrained benchmark is now more meaningful because this target has an observed feasible circuit at D2=436. No application depth cap has been chosen; 436 is not a lower bound or a guarantee on other targets. Retain fidelity–CNOT–depth curves before choosing externally motivated or development-frozen caps. Expanded GNN training remains on hold.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('Report written; result and critical-chain checks passed.')

if __name__=='__main__':main()
