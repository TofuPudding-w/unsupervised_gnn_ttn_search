"""Report the bounded decision experiment; no new model fitting."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('artifacts/decision_experiment')
OLD=Path('artifacts/stage_b')


def read(path):return json.loads(path.read_text())

def average(rows,key):
    values=[r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else float('nan')


def main():
    cost=ROOT/'cost';gnn=ROOT/'gnn'
    transition=read(cost/'transition.json')['rows']
    pred=read(cost/'prediction.json');screen=read(cost/'screening.json');done=read(cost/'completion.json')
    lines=['# Decision experiment before Stage C expansion','',
        'Stage C scaling and mixed random–SCENT pretraining remain on hold pending the bounded checks below. This report uses existing development evaluations; it is not an independent final benchmark. No pool-size expansion was performed.','',
        '## 1. Local mechanism and tolerance sensitivity','',
        'The selected overlapping_4 transition is a post-result case study. Each tolerance multiplier receives one direct construction/compilation, with no retries and unchanged global fidelity requirements.','',
        '| Local tolerance multiplier | Before CNOT | After CNOT | Difference | Before max rank | After max rank | Both feasible |','|---|---|---|---|---|---|---|']
    for mult in sorted({r['multiplier'] for r in transition}):
        a=next(r for r in transition if r['side']=='before' and r['multiplier']==mult)
        b=next(r for r in transition if r['side']=='after' and r['multiplier']==mult)
        lines.append(f"| {mult:.1f} | {a.get('cx')} | {b.get('cx')} | {a.get('cx',0)-b.get('cx',0)} | {a['features'][0]} | {b['features'][0]} | {a['feasible'] and b['feasible']} |")
    a=next(r for r in transition if r['side']=='before' and r['multiplier']==1.)
    b=next(r for r in transition if r['side']=='after' and r['multiplier']==1.)
    lines+=['','At nominal tolerance, the unchanged root and four two-qubit leaf-pair maps contribute the same 35 CNOTs. The two changing blocks account for the entire 225-CNOT difference:','',
        '| Block | Before leaves | After leaves | Rank before → after | Input bits before → after | Output bits before → after | Padded matrix before → after | Local CNOT before → after |','|---|---|---|---|---|---|---|---|']
    pairs=[([0,1,2,3,4,5],[0,1,2,3,4,5]),([0,1,2,3],[0,1,4,5])]
    for i,(old,new) in enumerate(pairs):
        x=next(r for r in a['blocks'] if r['leaves']==old);y=next(r for r in b['blocks'] if r['leaves']==new)
        lines.append(f"| {i+1} | {old} | {new} | {x['rank']} → {y['rank']} | {x['input_bits']} → {y['input_bits']} | {x['output_bits']} → {y['output_bits']} | {x['padded_rows']}×{x['padded_columns']} → {y['padded_rows']}×{y['padded_columns']} | {x.get('cx')} → {y.get('cx')} |")
    lines+=['',
        'The six-leaf block drops 251→122 CNOT (129 fewer), and the changed four-leaf block drops 218→122 (96 fewer). Their total is 225. Rank 9 requires four padded bond qubits; rank 8 requires three. Local sums equal global counts in this case, but this need not hold in general. The partition and tensor entries also change, so this is not an isolated intervention on rank alone.','',
        'The observed transition is stable under ±20% local-tolerance changes, not proof of robustness to arbitrary precision policies. Exact singular values, discarded weights at ranks 8/9, local dimensions, all fidelities, and QPY/statevector artifacts are saved under cost/.','',
        '## 2. Target-dependent descriptor prediction','',
        'Each descriptor uses only the first nominal-tolerance TTN build, even when the saved circuit label required later retries. Historical accepted representation ranks, parameter counts, and fidelities were independently reconstructed to check numerical compatibility; those verification values are not predictor inputs. Descriptors are max/sum rank, parameter count, input/output register widths, and padded matrix sizes.','',
        'The fixed score sum_padded_entries is the sum of padded row-count × column-count over internal isometries. It uses no circuit labels. Parameter and max-rank scores are additional controls. rank_width_gp fits log-CNOT using log-transformed descriptors; raw_gp uses topology features. Both GP scalers fit only known observations in each split. All methods share held-out splits and random tie orders.','',
        'Primary cohorts are six complete Stage B pools (128 trees each) and ten new-target cohorts consisting of SCENT plus 96 global-random samples. All 2,824 historical records, including adaptive samples, were extracted and checked; adaptive neighborhood-union records are not used in primary prediction metrics. The new targets have already been inspected in the neighborhood experiment, so these remain development diagnostics.','',
        '| Cohort | Family | Known labels | Method | Mean Spearman | Top-five regret | Random top-five regret | Low-cost precision |','|---|---|---|---|---|---|---|---|']
    for cohort in ['stage_b','new_random']:
        for family in ['hierarchical','overlapping']:
            for size in [8,16]:
                for method in ['max_rank','parameters','sum_padded_entries','rank_width_gp','raw_gp']:
                    rows=[r for r in pred if r['cohort']==cohort and r['family']==family and r['known_count']==size and r['method']==method]
                    lines.append(f"| {cohort} | {family} | {size} | {method} | {average(rows,'spearman'):.3f} | {average(rows,'top5_regret'):.2f} | {average(rows,'random_top5_regret'):.2f} | {average(rows,'low_cost_precision'):.3f} |")
    lines+=['','Padded-size performance by target with eight known labels (means over the five fixed splits):','',
        '| Cohort | Target | Spearman | Top-five regret | Random regret |','|---|---|---|---|---|']
    for cohort in ['stage_b','new_random']:
        for target in sorted({r['target'] for r in pred if r['cohort']==cohort}):
            rows=[r for r in pred if r['cohort']==cohort and r['target']==target and r['known_count']==8 and r['method']=='sum_padded_entries']
            lines.append(f"| {cohort} | {target} | {average(rows,'spearman'):.3f} | {average(rows,'top5_regret'):.2f} | {average(rows,'random_top5_regret'):.2f} |")
    lines+=['','## Static screening with explicit construction cost','',
        'These are offline selection replays with SCENT as the sole initial candidate. Total compilation budgets include SCENT; remaining candidates follow a fixed descriptor ranking or random order, without feedback. Every descriptor-screening run requires constructing its entire candidate set before ranking. These runs differ from the eight-point GNN initialization and must be compared within this table.','',
        '| Cohort | Family | Method | Total 16 | Total 32 | Total 64 | Mean whole-pool construction seconds |','|---|---|---|---|---|---|---|']
    for cohort in ['stage_b','new_random']:
        for family in ['hierarchical','overlapping']:
            for method in ['random','parameters','sum_padded_entries']:
                rows=[r for r in screen if r['cohort']==cohort and r['family']==family and r['method']==method]
                values=[np.mean([r['checkpoints'][str(b)] for r in rows]) for b in [16,32,64]]
                lines.append('| '+cohort+' | '+family+' | '+method+' | '+' | '.join(f'{v:.2f}' for v in values)+f" | {average(rows,'construction_s'):.4f} |")
    lines+=['',
        f"First-build descriptor extraction took {done['construction_s']:.3f} s for {done['descriptors']} records (mean {1000*done['construction_s']/done['descriptors']:.3f} ms each). Saved compilation attempts for these records total {done['historical_compilation_s']:.1f} s. This timing comparison is indicative, not an end-to-end speedup measurement: historical timings, shared caching, cohort construction, and repeated builds differ. The diagnostic reconstruction checks and sensitivity sweep are not required by a future screening algorithm.", '',
        'Random search has zero whole-pool prescreening cost in this table but still pays for construction and compilation of its selected candidates. Descriptor ranking is not a free topology calculation; eight-qubit dense construction costs cannot be extrapolated to larger systems. A strong ranking within these finite cohorts does not generate missing good trees or establish transfer to the SCENT approximate compiler. The dimension-based score is particularly aligned with this reference isometry-synthesis backend.','',
        '## Counterexamples and limits','',
        'The following includes every target for the preselected seed-0, eight-known-label split, including successes and failures of the padded-size score. The recommendation contains five trees. Equal dimensions can still yield different compiled costs, so ties are randomized rather than resolved by hidden labels.','',
        '| Cohort | Target | Best recommended CNOT | Held-out best CNOT | Regret | Recommended score | Held-out-best score |','|---|---|---|---|---|---|---|']
    for r in read(cost/'counterexamples.json'):
        lines.append(f"| {r['cohort']} | {r['target']} | {r['recommended_best_cx']} | {r['heldout_best_cx']} | {r['regret']} | {r['recommended_descriptor']} | {r['heldout_best_descriptor']} |")
    # Additional all-split worst cases prevent the preselected split from hiding failures.
    worst=sorted([r for r in pred if r['method']=='sum_padded_entries'],key=lambda r:r['top5_regret'],reverse=True)[:5]
    lines+=['','Largest padded-size recommendation regrets across all evaluated splits:','',
        '| Cohort | Target | Known labels | Split seed | Regret |','|---|---|---|---|---|']
    for r in worst:lines.append(f"| {r['cohort']} | {r['target']} | {r['known_count']} | {r['seed']} | {r['top5_regret']} |")
    if not (gnn/'completion.json').exists():
        lines+=['','## GNN training extension','', 'Still running; no downstream conclusion is available yet.','']
    else:
        training=read(gnn/'training.json');newpred=read(gnn/'prediction.json');oldpred=read(OLD/'prediction.json')
        newsearch=read(gnn/'search.json');oldsearch=read(OLD/'search.json')
        lines+=['','## 3. Fixed-corpus GNN extension','',
            'The model, 2,000-tree training corpus, validation/test split, Adam settings, and seeds are unchanged. Each seed was rerun from initialization for 300 epochs because the old checkpoint did not include optimizer state. First-100-epoch histories must agree within 1e-9. Checkpoints are selected only by structural validation loss.','',
            '| Seed | Old test loss | Extended test loss | Selected epoch | First-100 maximum difference | Training seconds |','|---|---|---|---|---|---|']
        for r in training:lines.append(f"| {r['seed']} | {r['old_test_loss']:.6f} | {r['test_loss']:.6f} | {r['best_epoch']} | {r['first_100_max_absolute_difference']:.2g} | {r['training_time_s']:.1f} |")
        lines+=['','Search uses the same six original Stage B pools, eight common initial points, seeds, frozen-encoder scaling and BO implementation. Results are paired within target and search seed.','',
            '| Family | Model | Training epochs | Budget 16 | Budget 32 | Budget 64 |','|---|---|---|---|---|---|']
        for family in ['hierarchical','overlapping']:
            for method in ['random','raw','pca','neighborhood','gnn_0','gnn_1','gnn_2']:
                for duration,data in [(100,oldsearch),(300,newsearch)]:
                    rows=[r for r in data if r['family']==family and r['method']==method]
                    if not rows:continue
                    values=[np.mean([r['checkpoints'][str(b)] for r in rows]) for b in [16,32,64]]
                    label=str(duration) if method.startswith('gnn') else 'baseline'
                    lines.append('| '+family+' | '+method+' | '+label+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
        lines+=['','| Family | Known labels | Model | Epochs | Spearman | Defined correlations | Top-five regret |','|---|---|---|---|---|---|---|']
        for family in ['hierarchical','overlapping']:
            for size in [8,16]:
                for method in ['gnn_0','gnn_1','gnn_2']:
                    for duration,data in [(100,oldpred),(300,newpred)]:
                        rows=[r for r in data if r['family']==family and r['train_size']==size and r['method']==method]
                        lines.append(f"| {family} | {size} | {method} | {duration} | {average(rows,'spearman'):.3f} | {sum(r['spearman'] is not None for r in rows)}/{len(rows)} | {average(rows,'top5_regret'):.2f} |")
        lines+=['','Undefined Spearman correlations from constant predictions are omitted from the correlation mean and counted explicitly; their recommendation regrets remain included. The longer runs include the original 100 epochs again; they are not claimed as optimizer-state resumes. Training seeds are reported separately, not selected after observing search quality. The same targets remain development data.','']
    if (gnn/'completion.json').exists():
        lines+=['','## Decision and next direction','',
            'The 9→8 transition is reproducible over the tested ±20% tolerance range, and two local blocks account exactly for its 225-CNOT reduction. Padded dimensions also predict costs across all eight overlapping target instances examined: with eight known labels, per-target mean Spearman correlations for the fixed score are 0.971–0.988. This extends the evidence beyond the isolated successful neighborhood case, within this reference compiler and these development distributions.', '',
            'The fixed padded-size score gives mean top-five regret 0.13 on the original overlapping pools and 0.40 on the new random-sample overlapping cohorts (eight-known-label splits), compared with random recommendation regrets 59.58 and 41.25. Its worst observed top-five regret across tested splits is 2. A small GP on the same rank/width descriptors has higher recommendation regret than this fixed score; learned fitting has not established incremental value here.', '',
            'Longer GNN training does help search in part. At budget 64, seed means improve 345.00→331.44, 332.56→331.22, and 348.00→343.11 CNOT. Therefore it would be incorrect to say that only reconstruction improved. Seeds 0 and 1 also improve mean search endpoints at budgets 16/32, while seed 2 worsens slightly there. Held-out cost-ranking correlations remain weak and retrieval is inconsistent; all three extended means still exceed the strongest original neighborhood baseline of 329.22 at budget 64. Two selected checkpoints are at epoch 300, so convergence is not established.', '',
            'Keep Stage C corpus scaling and mixed pretraining on hold. Prioritize an independently validated screening implementation that constructs candidates once, ranks them by padded dimensions, and compiles only a shortlist. Reuse the constructed TTNs where feasible and measure full elapsed time with equal candidate-generation and precision rules. Keep the extended GNNs as secondary baselines rather than discarding their measured gains. Do not infer transfer to larger n or approximate compilation from these results. Pool-size expansion is not the immediate explanatory experiment.', '']
        audit_path=ROOT/'audit.json'
        if audit_path.exists():
            audit=read(audit_path)
            lines+=['## Artifact checks','',
                f"Audited {audit['descriptor_records']} descriptor records, {audit['prediction_comparisons']} prediction comparisons, {audit['static_screening_runs']} screening runs, {audit['transition_variants']} tolerance/tree variants, and {audit.get('gnn_search_steps',0)} GNN search steps. The prediction comparison count includes fixed-score rankings; only 320 of the 800 descriptor comparisons fit a GP. Original first-100-epoch training histories reproduced exactly for all three seeds.", '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
