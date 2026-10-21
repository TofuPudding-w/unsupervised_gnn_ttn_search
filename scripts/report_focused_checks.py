"""Audit and summarize saved focused checks without fitting or evaluating candidates."""
import hashlib
import json
import statistics as stats
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from ttn_search.evaluate import write_json
from ttn_search.trees import canonical,tree_hash,neighbors

P=Path('artifacts/stage_a_focused')
OLD=Path('artifacts/stage_a_pilot_v2')


def avg(rows,key):return stats.mean(r[key] for r in rows)


def main():
    runs=json.loads((P/'search_comparisons.json').read_text())
    predictions=json.loads((P/'prediction.json').read_text())
    pairs=json.loads((P/'allocation_pairs.json').read_text())
    anomaly=json.loads((P/'compiler_anomaly.json').read_text())
    completion=json.loads((P/'completion.json').read_text())
    audited=0
    records={r['evaluation_key']:r for base in [OLD,P] for f in (base/'cache').glob('*/record.json') for r in [json.loads(f.read_text())]}
    for td in P.glob('*_[012]'):
        pool=[canonical(json.loads(t)) for t in json.loads((OLD/td.name/'pool.json').read_text())['trees']]
        hashes=[tree_hash(t) for t in pool];near={tree_hash(t) for t in neighbors(pool[0])}
        for tracefile in td.glob('*_[012].json'):
            trace=json.loads(tracefile.read_text());seen=[];best=None
            for row in trace:
                i=row['candidate_id'];assert i not in seen
                if row['acquisition']=='SCENT_neighbor':assert hashes[i] in near
                rec=records[row['evaluation_key']];assert rec['tree_hash']==hashes[i]
                expected_fraction = .2 if tracefile.name.startswith('alternative_') else .1
                assert rec['pipeline']['repr_fraction'] == expected_fraction
                if rec['status']=='success':
                    assert rec['fidelity_final'] >= rec['pipeline']['fidelity']
                    best=rec['cx_count'] if best is None else min(best,rec['cx_count'])
                assert row['best_cx']==best
                seen.append(i);audited+=1
            seed=int(tracefile.stem[-1]);reference=json.loads((OLD/td.name/f'random_{seed}.json').read_text())
            assert seen[:8]==[r['candidate_id'] for r in reference[:8]]
    for r in predictions:
        assert not set(r['train_ids'])&set(r['heldout_ids'])
        assert set(r['recommended_ids'])<=set(r['heldout_ids'])
    lines=['# Stage A focused checks', '',
        'Development diagnostics only. Historical evaluations are reused only with matching evaluator source, software, target, tree and pipeline identities. Timing is cache replay, not independent online speed.', '',
        f'Completed {completion["new_evaluations"]}; elapsed {completion["elapsed_s"]:.1f} seconds. Audited {audited} new trajectory steps and {len(predictions)} held-out prediction results.', '',
        '## Neighborhood-first comparison', '',
        'After the shared eight initial evaluations: choose unseen one-hop SCENT neighbors in seeded random order, then sample remaining candidates randomly. This is not adaptive local descent. All methods use identical 256-tree pools.', '',
        '| Family | Method | CX @8 | CX @16 | CX @32 | CX @64 |', '|---|---|---:|---:|---:|---:|']
    for family in ['hierarchical','overlapping']:
        for method in ['random','raw','pca','neighborhood']:
            rs=[r for r in runs if r['family']==family and r['method']==method and 'allocation' not in r]
            means=[stats.mean(r['checkpoints'][str(b)] for r in rs) for b in [8,16,32,64]]
            lines.append('| '+family+' | '+method+' | '+' | '.join(f'{m:.2f}' for m in means)+' |')
    lines+=['','Per-target endpoints average the three search seeds; targets, not seeds, are the independent units.','','| Target | Random @64 | Raw @64 | PCA @64 | Neighborhood @64 |','|---|---:|---:|---:|---:|']
    for target in sorted({r['target'] for r in runs}):
        means=[stats.mean(r['checkpoints']['64'] for r in runs if r['target']==target and r['method']==m and 'allocation' not in r) for m in ['random','raw','pca','neighborhood']]
        lines.append('| '+target+' | '+' | '.join(f'{m:.2f}' for m in means)+' |')
    lines+=['','## Held-out surrogate diagnosis','','Complete first-32 candidate subsets; train sizes 8/16 include SCENT. Five shared splits per target. Recommend five unseen candidates from predictive mean; simple regret is their best CX minus the held-out minimum. Random is the mean of 256 matched random recommendations. Lower regret is better. Spearman compares predicted and measured costs (higher is better). All diagnostic labels are feasible; feasibility discrimination is untested.','','| Family | Training count | Features | Mean Spearman | Top-5 regret | Random top-5 regret | Low-cost recall |','|---|---:|---|---:|---:|---:|---:|']
    for fam in ['hierarchical','overlapping']:
        for size in [8,16]:
            for method in ['raw','pca']:
                rs=[r for r in predictions if r['family']==fam and r['train_size']==size and r['method']==method]
                rho=stats.mean(r['spearman'] for r in rs if r['spearman'] is not None)
                lines.append(f'| {fam} | {size} | {method} | {rho:.3f} | {avg(rs,"top5_simple_regret"):.2f} | {avg(rs,"random_top5_expected_regret"):.2f} | {avg(rs,"low_cost_recall"):.3f} |')
    lines+=['','Low-cost set is the bottom 20% quantile including ties. First-32 subsets are heuristic-heavy, so prediction results do not certify performance over the entire 256-tree pool. Repeated splits are not additional independent targets.','','## Error allocation sensitivity','','Fixed final threshold 0.999, same compiler. Original global error fractions 0.1/0.4 versus 0.2/0.2. The two development targets were deliberately selected for the compilation anomaly and observed search headroom.','','| Target | Method | Original CX @32 | Alternative CX @32 |','|---|---|---:|---:|']
    pair_notes=[]
    for target in ['hierarchical_1','overlapping_1']:
        for method in ['random','raw','pca','neighborhood']:
            rs=[r for r in runs if r['target']==target and r['method']==method]
            a=stats.mean(r['checkpoints']['32'] for r in rs if 'allocation' not in r)
            b=stats.mean(r['checkpoints']['32'] for r in rs if 'allocation' in r)
            lines.append(f'| {target} | {method} | {a:.2f} | {b:.2f} |')
        pp=[r for r in pairs if r['target']==target]
        feasible=[r for r in pp if r['primary_status']==r['alternate_status']=='success']
        rho=float(np.asarray(spearmanr([r['primary_cx'] for r in feasible],[r['alternate_cx'] for r in feasible])[0]))
        changed=sum(r['primary_cx']!=r['alternate_cx'] for r in feasible)
        pair_notes.append(f'\n{target}: {len(feasible)}/{len(pp)} paired candidates feasible in both allocations; {changed} changed CX; rank correlation {rho:.4f}.\n')
    lines.extend(pair_notes)
    lines+=['','## Compilation anomaly','','Direct application of completed ideal maps reproduces the TTN with fidelity '+str(anomaly['ideal_map_fidelity'])+'. The root 64×1 state preparation is responsible for the discrepancy; other local maps have errors at approximately numerical precision. Changing transpiler optimization level does not remove it.','','| Synthesis epsilon | Optimization level | CX | Compilation fidelity |','|---:|---:|---:|---:|']
    for r in anomaly['trials']:
        lines.append(f'| {r["epsilon"]} | {r["optimization_level"]} | {r.get("cx_count")} | {r.get("fidelity_comp")} |')
    lines+=['','This isolates sensitivity to synthesis epsilon, not a proof of a universally safe epsilon. No production compiler settings or fidelity thresholds were changed. Any future compiler policy change requires matched revalidation across methods.']
    lines+=['','## Interpretation and next decision','',
        '- Neighborhood-first is a necessary baseline: it has the lowest mean overlapping-target cost at 32/64 evaluations, driven by overlapping target 1. Random still wins at 16 evaluations; there is no universal neighborhood advantage.',
        '- Raw/PCA features predict costs well in the hierarchical diagnostic subsets, but SCENT already provides the best observed circuit there. On overlapping targets, mean rank correlation is near zero and top-five regret is similar to or worse than random recommendations. This points to weak useful prediction on the relevant targets, not just an acquisition-rule problem.',
        '- The alternate error allocation leaves these selected search endpoints unchanged. The check covers two development targets and one alternative allocation, not general robustness.',
        '- Keep the simple neighborhood baseline. Before large pretraining, a small GNN experiment should first pass the same held-out prediction checks on more representative candidate subsets. Structure reconstruction quality alone is insufficient.',
        '- Keep global circuit verification mandatory. Validate any epsilon/strict-exactness policy across representative candidates before adopting it for new comparisons.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(P/'audit.json',dict(trajectory_steps=audited,prediction_splits=len(predictions),
        script_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('scripts').glob('*focused*.py')},
        anomaly_script_hash=hashlib.sha256(Path('scripts/check_compiler_anomaly.py').read_bytes()).hexdigest()))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
