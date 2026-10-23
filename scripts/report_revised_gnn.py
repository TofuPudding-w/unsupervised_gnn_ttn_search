"""Audit and summarize fixed-policy, matched-update ten-qubit GNN comparisons."""
import hashlib
import json
from pathlib import Path
import numpy as np
from qiskit import qpy
from analyze_c1_depth import two_qubit_depth
from ttn_search.evaluate import source_hash,write_json
from ttn_search.trees import tree_hash,canonical

OUT=Path('artifacts/revised_gnn')
def read(p):return json.loads(Path(p).read_text())
def checkpoint(run,b):
    return run['trace'][min(b,len(run['trace']))-1]

def main():
    protocol=read(OUT/'protocol.json');config=protocol['config'];done=read(OUT/'completion.json')
    runs=read(OUT/'runs.json');training=read(OUT/'training.json')
    assert len(runs)==done['runs']==105
    expected={(f'overlapping_{seed}',method,repeat) for seed in config['target_seeds'] for method in config['methods'] for repeat in config['repeats']}
    assert {(r['target'],r['method'],r['repeat']) for r in runs}==expected
    assert source_hash()==protocol['core_hash']
    for p,h in protocol['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    for p,h in read(OUT/'frozen_models.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    corpus=read(OUT/'corpus.json');sets={k:{tree_hash(canonical(t)) for t in v} for k,v in corpus.items()}
    assert len(sets['train'])==10000 and len(sets['validation'])==len(sets['test'])==500
    assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['test']&sets['validation']
    for seed in config['target_seeds']:assert not set(read(OUT/f'overlapping_{seed}'/'pool.json')['hashes'])&set.union(*sets.values())
    minf=1.;failures=0;count=0;calls=0;retries=0
    for run in runs:
        pool=read(OUT/run['target']/'pool.json');best=None;seen=set();cumulative=0;last=0.
        for row in run['trace']:
            rec=read(row['record_path']);i=row['candidate_id'];assert i not in seen;seen.add(i)
            assert rec['tree_hash']==pool['hashes'][i] and rec['target_hash']==pool['target_hash']
            assert rec['pipeline']==protocol['policy']['policy'] and not rec['cache_hit']
            assert rec['initial_reused']==(run['method'] in ['padded','parameters'])
            assert len((Path(row['record_path']).parent/'synthesis_starts.jsonl').read_text().splitlines())==rec['compiler_calls']
            cumulative+=rec['compiler_calls'];assert cumulative==row['compiler_calls']
            assert row['elapsed_s']>last;last=row['elapsed_s']
            if rec['status']=='success':
                assert rec['fidelity_final']>=.999 and 1-rec['fidelity_repr']<=.0001+1e-12 and 1-rec['fidelity_comp']<=.0004+1e-12
                minf=min(minf,rec['fidelity_final']);best=rec['cx_count'] if best is None else min(best,rec['cx_count'])
            else:failures+=1
            assert best==row['best_cx'];count+=1;calls+=rec['compiler_calls'];retries+=len(rec['attempts'])-1
        assert run['candidate_evaluations']==len(run['trace'])==(1 if run['method']=='scent' else 16)
    for run in runs:
        end=run['trace'][-1]
        chosen=next(t for t in run['trace'] if t['candidate_id']==end['best_candidate_id'])
        record=read(chosen['record_path'])
        with Path(record['circuit_path']).open('rb') as stream:circuit=qpy.load(stream)[0]
        assert int(circuit.count_ops().get('cx',0))==end['best_cx']
        assert two_qubit_depth(circuit)==end['best_d2']
    assert count==done['candidate_evaluations'] and calls==done['compiler_calls']
    write_json(OUT/'audit.json',dict(status='passed',incumbent_qpy_depth_checks=len(runs),runs=len(runs),evaluations=count,compiler_calls=calls,retries=retries,failed_evaluations=failures,minimum_final_fidelity=minf))
    lines=['# Ten-qubit GNN data-size comparison under the revised policy','',
        '## Design','',
        'Five previously untouched overlapping targets, seeds 14–18; 128 shared candidates per target, comprising SCENT and unique random pair-merge trees; three paired repetitions. All methods use the frozen strict-budget evaluator (3→1→0.1→0 local tolerance attempts). No policy or score is tuned on these target outcomes. SCENT alone uses one evaluation; other methods use up to sixteen.','',
        'These are newly trained ten-qubit GNNs, not transferred eight-qubit checkpoints. Nested random training corpora contain 2000 or 10000 trees, with common disjoint 500-tree validation/test sets. Benchmark pool topologies are excluded from all three corpus splits. Architecture, Adam .001, batch size 128, and 1600 updates are fixed; validation is checked every 100 updates. Three training seeds are retained, each paired with its corresponding search repetition. Scalers use the same 2000-tree subset for both corpus sizes, with separate frozen embeddings. No mixed SCENT pretraining is used.','',
        'Equal updates isolate additional data coverage, not convergence: sampling with replacement yields approximately 102 training-set passes for 2000 trees and 20 for 10000. Each checkpoint is selected by structural validation loss only. Training-seed and search-seed effects are paired, not independently factorial.','',
        'All online method runs generate and compile independently without outcome caches. Screeners pay for all 128 first-attempt TTNs and reuse them for selected candidates; GNNs pay for model loading, embedding and proposals. SCENT and GNN BO start with SCENT; plain random and fixed screeners do not receive a free SCENT circuit. Cyclic orders balance execution positions across fifteen target/repetition groups to within one occurrence.','',
        '## Reconstruction and training cost','',
        '| Training trees | Seed | Best validation loss | Test loss | Selected update | Training/setup seconds |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in training:
        best=min(r['history'],key=lambda h:h['validation'])
        lines.append(f"| {r['size']} | {r['seed']} | {r['best_validation']:.6f} | {r['test_loss']:.6f} | {best['step']} | {r['training_s']:.2f} |")
    lines+=['','Training/setup time includes training, validation/test evaluation, model saving and scaler embedding/export. Initial corpus generation, tensor preparation and common input/reference-pool setup are not fully timed. Thus reported training plus online time is not a complete from-zero amortization measurement.','',
        '## Mean best verified CNOT','',
        'Means over five targets and three repetitions; independent target count is five, not fifteen. Failed evaluations consume budget.','',
        '| Method | K=1 | K=2 | K=4 | K=8 | K=16 | Mean final incumbent D2 | Mean complete online time (s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in config['methods']:
        group=[r for r in runs if r['method']==method]
        vals=[np.mean([checkpoint(r,b)['best_cx'] for r in group]) for b in config['candidate_budgets']]
        lines.append('| '+method+' | '+' | '.join(f'{v:.2f}' for v in vals)+f" | {np.mean([r['trace'][-1]['best_d2'] for r in group]):.2f} | {np.mean([r['actual_wall_s'] for r in group]):.3f} |")
    lines+=['','Depth is that of the minimum-CNOT incumbent selected by the search, not the minimum-depth candidate or a depth-constrained BO result. Cost ties retain the first incumbent. These are fixed-evaluation comparisons with actual elapsed costs; no uncapped fixed-time claim is made.','',
        '## Per-target final quality and improvement over SCENT','',
        'Each entry is averaged over three repetitions. Positive gain means fewer CNOTs than SCENT. All targets are retained, including those with no observed improvement. Pool optima are unknown because these new pools are not exhaustively evaluated.','',
        '| Target | Method | Mean best CNOT | Gain over SCENT | Mean incumbent D2 | Mean online time (s) |',
        '|---|---|---:|---:|---:|---:|']
    for seed in config['target_seeds']:
        target=f'overlapping_{seed}';group=[r for r in runs if r['target']==target]
        base=np.mean([r['trace'][-1]['best_cx'] for r in group if r['method']=='scent'])
        for method in config['methods']:
            g=[r for r in group if r['method']==method];cost=np.mean([r['trace'][-1]['best_cx'] for r in g])
            lines.append(f"| {target} | {method} | {cost:.2f} | {base-cost:.2f} | {np.mean([r['trace'][-1]['best_d2'] for r in g]):.2f} | {np.mean([r['actual_wall_s'] for r in g]):.3f} |")
    lines+=['','## Observed improvement opportunities','',
        'Best across the candidates actually evaluated by any method, not an exhaustive pool optimum. An observed improvement demonstrates headroom; absence of one does not certify its absence from the pool. This table is descriptive and is not used to remove targets from the aggregate.','',
        '| Target | SCENT CNOT | Best observed CNOT | Observed gain | Unique evaluated candidates |','|---|---:|---:|---:|---:|']
    for seed in config['target_seeds']:
        group=[r for r in runs if r['target']==f'overlapping_{seed}']
        baseline=next(r['trace'][-1]['best_cx'] for r in group if r['method']=='scent')
        best=min(r['trace'][-1]['best_cx'] for r in group)
        seen={t['candidate_id'] for r in group for t in r['trace']}
        lines.append(f'| overlapping_{seed} | {baseline} | {best} | {baseline-best} | {len(seen)} |')
    lines+=['','## Timing at evaluation checkpoints','',
        'Mean actual elapsed seconds until each candidate checkpoint completes, including generation, construction, retries and verification. SCENT carries its single result forward.','',
        '| Method | K=1 | K=2 | K=4 | K=8 | K=16 |','|---|---:|---:|---:|---:|---:|']
    for method in config['methods']:
        group=[r for r in runs if r['method']==method]
        lines.append('| '+method+' | '+' | '.join(f"{np.mean([checkpoint(r,b)['elapsed_s'] for r in group]):.3f}" for b in config['candidate_budgets'])+' |')
    lines+=['','## Paired data-size comparison','',
        '| Budget | Expanded wins | Ties | Expanded losses | Mean CNOT change (10000 − 2000) |','|---:|---:|---:|---:|---:|']
    for b in config['candidate_budgets']:
        differences=[]
        for seed in config['target_seeds']:
            for repeat in config['repeats']:
                pair={r['method']:checkpoint(r,b)['best_cx'] for r in runs if r['target']==f'overlapping_{seed}' and r['repeat']==repeat and r['method'].startswith('gnn_')}
                differences.append(pair['gnn_10000']-pair['gnn_2000'])
        lines.append(f"| {b} | {sum(d<0 for d in differences)} | {sum(d==0 for d in differences)} | {sum(d>0 for d in differences)} | {np.mean(differences):.2f} |")
    lines+=['','## Interpretation','',
        'Increasing the random corpus from 2000 to 10000 trees produced no improvement in best verified CNOT at any checkpoint in any of the fifteen paired runs. Both GNN settings retain the SCENT incumbent on all five targets. Their complete online runs cost about 32 seconds on average versus about 0.85 seconds for SCENT alone, before charging offline training. This bounded experiment establishes no added selection value for the larger corpus. It does not prove that GNN selection is impossible or that longer training cannot help.','',
        'Reconstruction improves for seed 0, is nearly unchanged for seed 1, and worsens for seed 2. Every selected checkpoint is at update 1600, so convergence is not established. The matched-update experiment tests coverage at a fixed optimization budget; it does not compare fully converged models.','',
        'Target 15 demonstrates useful selection headroom: SCENT gives 955 CNOTs and depth 795; both screeners consistently find 805 CNOTs and depth 742. Padded screening does so at its first candidate; parameter screening reaches it by K=4. SCENT-initialized random succeeds in one of three repetitions; neither GNN does. On four other targets, no tested method improves SCENT. These pools are not exhaustive, so no-improvement cases are not certified zero-headroom cases.','',
        'The current evidence supports the shared verified tolerance policy and simple screening as strong baselines. It does not justify automatically increasing structural pretraining further or adding mixed pretraining. Any next learning experiment should change an explicitly diagnosed bottleneck—such as target-dependent information or the training objective—and retain the fixed-score baselines. No such follow-up is launched by this report.','']
    lines+=['','## Audit and limitations','',
        f"Audited {len(runs)} runs, {count} candidate evaluations, {calls} synthesis starts, and {retries} retry attempts. Failed completed evaluations: {failures}. Minimum successful final fidelity: {minf:.12f}. Frozen source/model fingerprints, corpus separation, identical target/pool identity, policy compliance, no-cache flags, unique selections, incumbent prefixes and attempt counts pass checks. Final CNOT counts and two-qubit depths were independently recomputed from all 105 incumbent QPY circuits.",'',
        'This is a bounded matched-update test of random structural pretraining, not evidence about every GNN architecture or target-dependent learning objective. Five targets, one size, one compiler, and a sixteen-query horizon limit generalization. No larger system or mixed pretraining was added.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in config['methods']:
        group=[r for r in runs if r['method']==method]
        axes[0].plot(config['candidate_budgets'],[np.mean([checkpoint(r,b)['best_cx'] for r in group]) for b in config['candidate_budgets']],marker='o',label=method)
        axes[1].scatter(np.mean([r['actual_wall_s'] for r in group]),np.mean([r['trace'][-1]['best_cx'] for r in group]),label=method)
    axes[0].set(xlabel='Candidate evaluations',ylabel='Mean best verified CNOT',title='Five untouched overlapping targets')
    axes[1].set(xlabel='Mean complete online run time (s)',ylabel='Mean final CNOT',title='Fixed-evaluation quality and cost')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    print('Report and audit complete.')

if __name__=='__main__':main()
