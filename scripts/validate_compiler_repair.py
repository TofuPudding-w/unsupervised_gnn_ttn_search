"""Paired validation of an opt-in compiler repair; historical artifacts untouched."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from qiskit import qpy
from threadpoolctl import threadpool_limits
from ttn_search.compilers.verified import compile_verified, COMPILER_VERSION
from ttn_search.reference import build, compile_reference, fidelity, circuit_state
from ttn_search.targets import gaussian_target
from ttn_search.trees import random_tree, canonical
from ttn_search.evaluate import write_json, software
from analyze_c1_depth import two_qubit_depth

OUT = Path('artifacts/compiler_repair')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path('artifacts/screening_headroom/audit.json').read_text())
    cases = [dict(name='hierarchical_13_random', family='hierarchical', seed=13,
        tree=random_tree(10, np.random.default_rng(2026)), role='original_failure')]
    for target in audit['targets']:
        seed = int(target['target'].split('_')[-1])
        selected = [r for r in target['rows'] if r['retry'] or r['padded_rank']==1 or r['candidate_id']==0]
        # One prespecified successful expensive control in each pool as well.
        controls = [r for r in target['rows'] if not r['retry'] and r['candidate_id']!=0 and r['padded_rank']!=1]
        selected += [min(controls, key=lambda r:r['candidate_id'])]
        for row in selected:
            record = json.loads(Path(row['record_path']).read_text())
            cases.append(dict(name=f"overlapping_{seed}_{row['candidate_id']}", family='overlapping', seed=seed,
                tree=canonical(json.loads(record['tree_serialization'])),
                role='historical_retry' if row['retry'] else 'successful_control'))
    protocol = dict(version=COMPILER_VERSION, software=software(), cases=cases,
        files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['src/ttn_search/compilers/verified.py', 'src/ttn_search/reference.py', __file__]},
        policy='Paired direct 3x attempts; alternate backend order; same TTN and compiler seed17/opt1; no label replacement or policy fallback; independent final verification')
    write_json(OUT/'protocol.json', protocol)
    results = []
    for i, case in enumerate(cases):
        directory = OUT/case['name']; directory.mkdir(exist_ok=True)
        state = gaussian_target(10, case['family'], case['seed']).state
        start = time.perf_counter(); rep = build(state, case['tree'], .0001*3/9)
        build_s = time.perf_counter()-start
        row = dict(name=case['name'], role=case['role'], build_s=build_s,
            representation_fidelity=fidelity(state, rep.state), backends={})
        for backend in (['frozen', 'repaired'] if i%2==0 else ['repaired', 'frozen']):
            stats = {}; start = time.perf_counter()
            c = compile_reference(rep,17,1) if backend=='frozen' else compile_verified(rep,17,1,stats=stats)
            compile_s = time.perf_counter()-start
            start = time.perf_counter(); output = circuit_state(c)
            fc = fidelity(rep.state, output); ff = fidelity(state, output)
            verify_s = time.perf_counter()-start
            row['backends'][backend] = dict(compilation_fidelity=fc, final_fidelity=ff,
                cx=int(c.count_ops().get('cx',0)), d2=two_qubit_depth(c), compile_s=compile_s,
                verify_s=verify_s, stats=stats,
                feasible=ff>=.999 and 1-fc<=.0004 and 1-row['representation_fidelity']<=.0001)
            with (directory/f'{backend}.qpy').open('wb') as stream:qpy.dump(c, stream)
        assert row['backends']['repaired']['compilation_fidelity']>1-1e-8, row['name']
        write_json(directory/'result.json', row); results.append(row)
        write_json(OUT/'results.json', results)
        print(case['name'], row['backends']['frozen']['compilation_fidelity'],
            row['backends']['repaired']['compilation_fidelity'], row['backends']['repaired']['stats']['diagonals_corrected'], flush=True)
    old_fail = sum(r['backends']['frozen']['compilation_fidelity']<1-.0004 for r in results)
    controls = [r for r in results if r['role']=='successful_control']
    lines = ['# Versioned compiler repair validation', '',
        f'Compiler: `{COMPILER_VERSION}`. Installed Qiskit remains unchanged. The frozen `compile_reference` remains available and unchanged.', '',
        '## Mechanism and repair', '',
        'The saved UCGate circuit implements the intended gate up to a diagonal, but the returned diagonal has incorrect relative signs. Independently reconstructing the control-branch 2×2 matrices reveals the mismatch. The adapter checks D U = target, recomputes row phases only when needed, and uses a local uniformly controlled Euler-rotation fallback if the residual exceeds 1e-8 per matrix entry. The fallback includes control-dependent phases and is independently verified as a full operator; it is rejected if verification fails. Exact sign corrections are retained when consistent with that threshold. The diagonal correction adds no gates; the Euler fallback can increase the local gate count.', '',
        'The adapter freezes each checked circuit and uses its verified diagonal in the existing Isometry elimination algorithm, including multi-controlled gate steps. It does not alter input tensors, truncation tolerances, or fidelity requirements. Both ordinary and repaired circuits undergo independent statevector verification. The precise internal Rust numerical cause of the bad signs is not claimed to be repaired upstream.', '',
        'Verification stores one 2×2 matrix per control assignment, rather than a full multiplexer operator. It costs O(branches × emitted gates), so the extra work is real and may limit scaling. Private Qiskit interfaces are version-gated to 2.6.0. The rare Euler fallback uses a full local operator for verification and therefore has higher memory cost. Unexpected circuit structure, unsupported versions and failed fallback verification fail explicitly. No global monkeypatches or vendor edits are used.', '',
        '## Paired direct 3× checks', '',
        f'{len(results)} cases: original hierarchical failure, every unique retry candidate in the six-pool audit, SCENT/padded-first controls and one additional successful candidate per pool. All are development cases, not new held-out tests. {old_fail} direct frozen compilations fail the compilation budget; all repaired compilations have fidelity greater than 1−1e-8 to the reconstructed TTN.', '',
        '| Case | Role | Frozen Fcomp | Repaired Fcomp | CX old/new | D2 old/new | Corrected diagonals / Euler fallbacks | Compile seconds old/new |',
        '|---|---|---:|---:|---|---|---:|---|']
    for r in results:
        a=r['backends']['frozen'];b=r['backends']['repaired']
        lines.append(f"| {r['name']} | {r['role']} | {a['compilation_fidelity']:.10f} | {b['compilation_fidelity']:.10f} | {a['cx']}/{b['cx']} | {a['d2']}/{b['d2']} | {b['stats']['diagonals_corrected']}/{b['stats']['euler_fallbacks']} | {a['compile_s']:.3f}/{b['compile_s']:.3f} |")
    repr_failures = [r['name'] for r in results if 1-r['representation_fidelity'] > .0001]
    lines += ['', f'Representation-budget failures at direct 3×: {repr_failures}. These still require the frozen tolerance fallback despite correct compilation. Direct synthesis here is diagnostic; the production evaluator must continue to reject representation-infeasible attempts before synthesis.', '']
    changed = sum((r['backends']['frozen']['cx'],r['backends']['frozen']['d2']) != (r['backends']['repaired']['cx'],r['backends']['repaired']['d2']) for r in controls)
    lines += ['', f'Of {len(controls)} successful controls, {changed} change CNOT or two-qubit depth. Mean compilation time across all cases: frozen {np.mean([r["backends"]["frozen"]["compile_s"] for r in results]):.3f} s, repaired {np.mean([r["backends"]["repaired"]["compile_s"] for r in results]):.3f} s. These are single paired measurements with alternating order, not a repeated timing benchmark.', '',
        '## Status and migration', '',
        'The opt-in repaired compiler is implemented and validated on this bounded cohort. This does not certify arbitrary matrices or larger systems. Keep final-fidelity and sub-budget checks and the common fallback policy when integrating it into evaluations.', '',
        'Use `from ttn_search.compilers.verified import compile_verified, COMPILER_VERSION`. The existing reference compiler is not redirected. Historical labels, reports and fallback outcomes remain untouched. This report compares direct 3× attempts; corrected 3× costs are not interchangeable with historical accepted 1× fallback costs. A full pool or method comparison requires a separately versioned evaluator run, rebuilding outcomes for every compared method. That refresh is not performed here.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write_json(OUT/'completion.json', dict(cases=len(results), frozen_compilation_failures=old_fail,
        repaired_compilation_failures=0, version=COMPILER_VERSION))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
