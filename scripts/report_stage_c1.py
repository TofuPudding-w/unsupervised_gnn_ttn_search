"""Audit and summarize saved C1 prefixes without rerunning experiments."""
import hashlib
import json
from pathlib import Path

OUT=Path('artifacts/stage_c1')

def read(p):return json.loads(Path(p).read_text())
def point(run,budget,key):
    rows=[x for x in run['trace'] if x[key]<=budget]
    return rows[-1]['best_cx'] if rows else None

def main():
    runs=read(OUT/'runs.json');protocol=read(OUT/'protocol.json');config=protocol['config']
    assert len(runs)==read(OUT/'completion.json')['runs']==20
    assert {(r['family'],r['mode'],r['method']) for r in runs}=={(f,m,a) for f in config['families'] for m in ['evaluation','time'] for a in config['methods']}
    h=hashlib.sha256()
    for p in sorted(Path('src/ttn_search').glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    source_matches=h.hexdigest()==protocol['core_hash']
    runner_matches=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in protocol['files'].items())
    fidelities=[];records=0;calls=0;failures=0;retry_attempts=0
    for run in runs:
        pool=read(OUT/run['family']/'pool.json');best=None;seen=set();cumulative=0;previous=0
        for row in run['trace']:
            assert row['elapsed_s']>=previous and row['elapsed_s']<=run['deadline_s'];previous=row['elapsed_s']
            i=row['candidate_id'];assert i not in seen;seen.add(i)
            rec=read(row['record_path']);assert rec['tree_hash']==pool['hashes'][i]==row['tree_hash']
            assert rec['target_hash']==pool['target_hash'] and not rec['cache_hit']
            assert rec['pipeline']['fidelity']==.999
            cumulative+=rec['compiler_calls'];assert cumulative==row['compiler_calls']
            assert rec['compiler_calls']==sum(a['compile_attempted'] for a in rec['attempts'])
            if rec['status']=='success':
                assert rec['fidelity_final']>=.999
                assert 1-rec['fidelity_repr']<=.0001+1e-12 and 1-rec['fidelity_comp']<=.0004+1e-12
                fidelities.append(rec['fidelity_final']);best=rec['cx_count'] if best is None else min(best,rec['cx_count'])
            else:failures+=1
            assert best==row['best_cx']
            records+=1;calls+=rec['compiler_calls'];retry_attempts+=len(rec['attempts'])-1
    lines=['# Revised Stage C1: 10-qubit scaling pilot','',
        '## Scope and conclusion','',
        '**C1 is complete for this bounded pilot; C2 and the 12-qubit experiment have not been run.** Two targets (hierarchical and overlapping, seed 13), one repetition per method, 128 candidates, the unchanged padded-size score, dense SVD construction, and the reference isometry compiler were used. The pool contains SCENT plus 127 unique random pair-merge trees. SCENT alone evaluates only its own tree. The required final fidelity is **0.999**, not 0.9999.','',
        '**Engineering feasibility passed at 10 qubits, but a screening advantage over SCENT was not demonstrated on these two targets.** Parameter screening selects SCENT first on both targets. Padded screening selects SCENT first on hierarchical (81 CNOTs); on overlapping it first returns 1171 CNOTs, then reaches the SCENT cost of 1163 at its second candidate. SCENT-initialized random search also retains those costs. Plain random finishes worse. Screening therefore spends extra preparation time without improving the available SCENT result in this pilot.','',
        'This differs from the fresh eight-qubit overlapping benchmark. It does not establish that screening stops working at ten qubits: there is only one target per family, and target identity changes along with size. The follow-up exhaustively evaluated the overlapping pool and established SCENT as its unique policy-defined CNOT optimum (1163); the hierarchical pool optimum remains unestablished. No conclusion about compiler independence follows from C1.','',
        '## Fixed candidate-evaluation budgets','',
        'These are separate runs from the fixed-time experiment. A 300-second safety ceiling applies, but all ten fixed-evaluation runs finished normally. SCENT uses one evaluation; other methods complete sixteen. Cells are best verified CNOT, with SCENT carried forward as a baseline.','',
        '| Family | Method | K=1 | K=2 | K=4 | K=8 | K=16 | First result (s) | Run wall (s) | Compiler calls |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        if r['mode']!='evaluation':continue
        s=r['summary'];assert s and not r['stopped_at_deadline'] and r['exitcode']==0
        vals=[str(point(r,b,'k')) for b in config['candidate_budgets']]
        lines.append(f"| {r['family']} | {r['method']} | "+' | '.join(vals)+f" | {r['first_result_s']:.3f} | {s['actual_wall_s']:.3f} | {s['compiler_calls']} |")
    lines+=['','## Independent fixed-time runs','',
        'Each method may continue until 30 seconds or pool exhaustion, without a sixteen-candidate cap. The parent terminates unfinished work at the deadline. Only completed, saved, verified prefixes within each deadline receive credit. A dash means no verified result yet. SCENT ends after one evaluation and retains that result.','',
        '| Family | Method | 1 s | 2 s | 4 s | 8 s | 16 s | 30 s | Completed candidates | Calls in completed candidates | First result (s) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        if r['mode']!='time':continue
        vals=[str(v) if (v:=point(r,b,'elapsed_s')) is not None else '—' for b in [1,2,4,8,16,30]]
        lines.append(f"| {r['family']} | {r['method']} | "+' | '.join(vals)+f" | {len(r['trace'])} | {r['trace'][-1]['compiler_calls']} | {r['first_result_s']:.3f} |")
    lines+=['','Time runs completed 43 padded-screened hierarchical candidates and 24 padded-screened overlapping candidates, demonstrating that the old sixteen-evaluation cap is removed. Parameter screening completed 33 and 25 respectively. None improved the SCENT incumbent. The number completed is a throughput observation, not evidence of better selected circuits.','',
        '**Accounting limitation:** compiler-call totals above include retries in completed candidates. An interrupted in-flight candidate may have incurred additional synthesis calls that were not flushed to a final record. These are lower bounds on total attempted calls, not complete physical-call accounting for deadline-terminated runs. Partial artifacts are retained. The runner needs attempt-start logging before this is used as a formal fixed-time benchmark.','',
        '## Construction and memory','',
        'Whole-pool build time excludes representation checks and ranking; first-result and wall times include those costs. For terminated runs, the initial pool-build cost is recovered from the first prefix minus any incremental build cost of its selected candidate. Peak RSS is sampled from the worker process and expressed in MiB; it is not whole-machine memory or an exact allocation profile.','',
        '| Family | Mode | Method | Whole-pool build (s) | First result (s) | Sampled peak RSS (MiB) |',
        '|---|---|---|---:|---:|---:|']
    for r in runs:
        build=0.
        if r['method'] in ['padded','parameters']:
            first=r['trace'][0];rec=read(first['record_path'])
            build=first['construction_s']-rec['incremental_build_s']
        lines.append(f"| {r['family']} | {r['mode']} | {r['method']} | {build:.3f} | {r['first_result_s']:.3f} | {r['peak_rss_kib']/1024:.1f} |")
    lines+=['','Whole-pool construction is affordable in this pilot, but timings vary substantially between independent runs. In particular, hierarchical padded and parameter runs take different build times despite using the same construction work. With one repetition and the same randomized method order reused across groups, these differences should not be attributed to the score itself. Replication and stronger execution-order balancing are needed for reliable speed comparisons.','',
        '## Fidelity, retries and record checks','',
        f'Audited {len(runs)} runs and {records} completed candidate evaluations. All prefix incumbents, tree/target identities, cumulative completed-call counts and fidelity constraints are consistent. Completed candidates contain {calls} synthesis calls and {retry_attempts} retry attempts. Failed completed evaluations: {failures}. Minimum final fidelity among successful evaluations: **{min(fidelities):.12f}**. These totals exclude unfinished work killed at a deadline.','',
        f'Current core source matches frozen protocol: {source_matches}. Runner/evaluator fingerprints match: {runner_matches}. Protocol and artifacts remain the authoritative record of execution; this report does not rerun circuits.','',
        'The one completed infeasible evaluation is hierarchical/time/parameters candidate 49. It did not replace the feasible incumbent. Deadline interruptions are counted separately and are not labeled fidelity failures.','',
        '## Next decision','',
        'Keep larger/mixed GNN pretraining on hold. The resource evidence makes a bounded 12-qubit feasibility probe reasonable, but does not justify a large sweep. Before claiming screening generalization, add multiple untouched 10-qubit instances with repeated timing measurements and paired comparisons to SCENT. Improve interrupted-attempt accounting and preserve progress atomically. C2 should then test a materially different synthesis method on a small subset; seed or optimization-level changes alone do not establish compiler independence.','',
        'No new 12-qubit training, compilation, or alternative-compiler experiment was launched while writing this report.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(f'Wrote {OUT}/REPORT.md; checked {records} completed candidates, minimum fidelity {min(fidelities):.12f}')

if __name__=='__main__':main()
