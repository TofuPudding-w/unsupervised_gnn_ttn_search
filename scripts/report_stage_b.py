"""Summarize every training seed; never select a winner using test performance."""
import json
from pathlib import Path
import numpy as np

OUT = Path('artifacts/stage_b')


def followup_diagnostics(search):
    """Recompute pool density and small-budget summaries from saved pilot labels."""
    targets = list(dict.fromkeys(r['target'] for r in search))
    records = {t: json.loads((OUT/t/'records.json').read_text()) for t in targets}
    lines = ['## Why all evaluated trees met the fidelity requirement', '',
        'The recorded reference policy has no imposed maximum bond dimension or CNOT cap. SVD retains enough singular values for its discarded-weight tolerance. The final fidelity threshold is 0.999 (final infidelity budget 0.001); the separate representation infidelity budget is 0.0001. Failed attempts tighten the initial local truncation tolerance by factors 1, 0.1, and 0; numerical-zero modes can still be removed at zero tolerance. Each candidate stops at its first feasible attempt. No hard time or memory limit was enforced.', '',
        'Thus the experiment compares how much circuit resource each tree needs to meet the same precision requirement. All-feasible results are compatible with substantial cost differences; they do not validate feasibility prediction. These conclusions concern this reference compiler, not the paper\'s approximate compiler.', '',
        '| Target | Candidates | Minimum CNOT | Maximum CNOT | Within 5% of pool minimum | Within 10% |',
        '|---|---|---|---|---|---|']
    for target, rr in records.items():
        assert all(r['status'] == 'success' for r in rr)
        costs = np.array([r['cx_count'] for r in rr])
        best = costs.min()
        lines.append(f'| {target} | {len(rr)} | {best} | {costs.max()} | {np.sum(costs <= 1.05*best)} | {np.sum(costs <= 1.10*best)} |')
    lines += ['',
        'The hierarchical search incumbents all equal 75 because SCENT is included in initialization, not because every hierarchical tree costs 75. Pool minima are observed finite-pool optima, not global optima.', '',
        '## Random search, pool coverage, and smaller budgets', '',
        'Random search samples unseen candidates uniformly after the shared eight-point initialization and retains its best feasible result. It therefore includes SCENT plus additional evaluations. Beating the single SCENT result does not establish lower total computation than SCENT.', '',
        'At total budget 64, random search draws 56 of the 120 remaining candidates. Conditional on a unique optimum not already appearing in initialization, its probability of finding that optimum is 56/120 = 46.7%. Several near-optimal trees make finding a good result easier. The density table shows that overlapping_0 and overlapping_2 have many near-optimal candidates, whereas overlapping_1 has only one within 10% of its optimum.', '',
        'The following values are mean best feasible CNOT over the three overlapping targets and three search seeds. Extra budgets 4/8/16 correspond to total budgets 12/16/24. GNN rows keep each training seed separate.', '',
        '| Method | Extra 4 (total 12) | Extra 8 (total 16) | Extra 16 (total 24) |',
        '|---|---|---|---|']
    for method in ['random', 'neighborhood', 'raw', 'pca', 'gnn_0', 'gnn_1', 'gnn_2']:
        traces = [json.loads((OUT/r['target']/f"{method}_{r['seed']}.json").read_text())
                  for r in search if r['family'] == 'overlapping' and r['method'] == method]
        values = [np.mean([tr[b-1]['best_cx'] for tr in traces]) for b in [12, 16, 24]]
        lines.append('| '+method+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['',
        'All hierarchical incumbents remain 75 at these budgets. On overlapping targets, the trained GNNs still trail random search at these smaller budgets. High pool coverage can explain part of random search\'s strength, but does not by itself explain the lack of GNN improvement. Small system size remains an untested explanation, not an established cause.', '',
        '## Priority for the next development experiment', '',
        'Prioritize independent neighborhood validation and mechanism analysis before expanding the GNN. The current neighborhood-first method already samples one-hop SCENT neighbors in random order, restricted to those present in the pool, then falls back to random unseen candidates. It has no learned ranking and does not update the neighborhood around new incumbents. A separate "random-order one-hop" baseline is only different if its candidate domain, initialization, or fallback differs; those differences must be explicit.', '',
        'Neighbors are rooted local binary-tree rotations that preserve physical leaf labels, not arbitrary leaf-label swaps. Validate on a predeclared set of new targets with a shared SCENT starting point and equal additional budgets, generating neighborhoods without cost-based filtering. Compare global random sampling, the full initial neighborhood followed by random fallback, and feedback-driven iterative neighborhood search. Record actual computation as well as evaluation counts and target-wise improvements; the existing pool construction may favor the supplied neighborhood.', '',
        'For successful and unsuccessful moves, inspect changed leaf partitions, bond ranks, fidelity, and local synthesis costs. Separately synthesized local CNOT counts need not sum to the final globally transpiled circuit count, so preserve both measurements. Treat rank reduction and changes in synthesis difficulty as competing explanations until the evidence separates them.', '',
        'Give the original GNN one bounded diagnostic: keep corpus, architecture, candidate pools, and all three seeds fixed, predeclare a longer training budget, and select checkpoints only by validation reconstruction loss. If reconstruction improves without consistent retrieval and small-budget search improvements, pause expansion of this objective. Do not simultaneously enlarge the corpus, change the architecture, and add mixed pretraining.', '',
        'These are follow-up proposals, not completed experiments. The present neighborhood signal is promising but does not establish a general method or novelty claim.', '']
    return lines


def main():
    training = json.loads((OUT/'training.json').read_text())
    prediction = json.loads((OUT/'prediction.json').read_text())
    search = json.loads((OUT/'search.json').read_text())
    completion = json.loads((OUT/'completion.json').read_text())
    lines = ['# Stage B development results', '',
        'Six existing eight-qubit development targets. Each pool contains SCENT plus 127 uniformly sampled trees from its original 256-candidate pool. All methods share eight initial evaluations, including SCENT. Budgets include initialization. This is offline replay, not a measured online speedup.', '',
        'Training uses 2,000 unique random pair-merging trees, plus 250 validation and 250 test trees. All original candidate-pool structures are excluded. Four direction-aware GIN layers, hidden size 64, latent size eight; equal-weight LCA/leaf-depth reconstruction losses. Each seed trains for 100 epochs; checkpoint selection uses validation reconstruction loss only. Scalers and PCA are fitted only to the shared training corpus.', '',
        '## Reconstruction', '', '| Training seed | Selected epoch | Test loss | Constant-mean test loss | Training seconds |', '|---|---|---|---|---|']
    for r in training:
        lines.append(f"| {r['seed']} | {r['best_epoch']} | {r['test_loss']:.6f} | {r['mean_predictor_test_loss']:.6f} | {r['training_time_s']:.1f} |")
    for family in ['hierarchical','overlapping']:
        lines += ['',f'## {family}: search', '', 'Mean best feasible CNOT count over three targets and three shared search seeds. Training seeds are separate methods, not additional independent target samples.', '', '| Method | 8 | 16 | 32 | 64 | Mean proposal seconds |','|---|---|---|---|---|---|']
        for method in dict.fromkeys(r['method'] for r in search):
            rows = [r for r in search if r['family']==family and r['method']==method]
            values = [np.mean([r['checkpoints'][str(b)] for r in rows]) for b in [8,16,32,64]]
            lines.append('| '+method+' | '+' | '.join(f'{v:.2f}' for v in values)+f" | {np.mean([r['proposal_time_s'] for r in rows]):.2f} |")
        lines += ['',f'## {family}: held-out cost prediction', '', 'Mean over three targets and five common splits. Top-five regret is the cheapest recommended candidate minus the cheapest held-out candidate. Lower is better; random recommendation regret averages 256 draws per split.', '', '| Known labels | Representation | Spearman rho | Top-five regret | Random regret |','|---|---|---|---|---|']
        for size in [8,16]:
            for method in dict.fromkeys(r['method'] for r in prediction):
                rows = [r for r in prediction if r['family']==family and r['train_size']==size and r['method']==method]
                rho = [r['spearman'] for r in rows if r['spearman'] is not None]
                lines.append(f"| {size} | {method} | {np.mean(rho):.3f} | {np.mean([r['top5_regret'] for r in rows]):.2f} | {np.mean([r['random_top5_regret'] for r in rows]):.2f} |")
    lines += ['', '## Low-cost retrieval', '',
        'The low-cost set is the bottom quintile of held-out costs, including ties. Each recommendation contains five candidates. Reported across three targets and five splits.', '',
        '| Family | Known labels | Representation | Top-five precision | Recall |', '|---|---|---|---|---|']
    for family in ['hierarchical','overlapping']:
        for size in [8,16]:
            for method in dict.fromkeys(r['method'] for r in prediction):
                rows = [r for r in prediction if r['family']==family and r['train_size']==size and r['method']==method]
                precision=[]; recall=[]
                for r in rows:
                    records=json.loads((OUT/r['target']/'records.json').read_text())
                    costs=np.array([x['cx_count'] for x in records])
                    low={i for i in r['held'] if costs[i]<=np.quantile(costs[r['held']],.2)}
                    hits=len(set(r['recommended']) & low)
                    precision.append(hits/5); recall.append(hits/len(low))
                lines.append(f'| {family} | {size} | {method} | {np.mean(precision):.3f} | {np.mean(recall):.3f} |')
    lines += ['', '## Per-target search endpoints', '',
        'Mean best CNOT at budget 64 over the three search seeds. The pool minimum uses all 128 labels and is an offline oracle.', '',
        '| Target | SCENT | Pool minimum | Random | Neighborhood | Raw | PCA | Untrained | GNN 0 | GNN 1 | GNN 2 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for target in dict.fromkeys(r['target'] for r in search):
        records = json.loads((OUT/target/'records.json').read_text())
        values = [records[0]['cx_count'],min(r['cx_count'] for r in records)]
        values += [np.mean([r['checkpoints']['64'] for r in search if r['target']==target and r['method']==m])
                   for m in ['random','neighborhood','raw','pca','untrained','gnn_0','gnn_1','gnn_2']]
        lines.append('| '+target+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['', *followup_diagnostics(search)]
    lines += ['', '## Scope and accounting', '',
        f"Completed {completion['trajectories']} trajectories and {completion['prediction_fits']} held-out fits. Newly compiled {completion['new_evaluations']} target–tree pairs; reused only exact evaluator/source/environment/configuration matches. Total run time {completion['elapsed_s']:.1f} seconds, including {completion['training_time_s']:.1f} seconds of training.", '',
        'Training and feature extraction are additional overheads; proposal time alone is not total online time. Random and neighborhood search do not incur encoder training. All evaluated labels were feasible, so these tests do not establish feasibility discrimination. Candidate subsampling and the common training corpus for PCA differ from Stage A; compare methods within this run.', '',
        'No independent generalization claim is justified. Do not scale the corpus or introduce mixed pretraining solely because reconstruction succeeds. The decision must also consider held-out retrieval, search results across training seeds, and overhead.', '',
        'See protocol.json, corpus.json, training.json, prediction.json, search.json, and per-target records/traces for reproducibility. The circuit evaluator and its numerical-synthesis policy are unchanged.', '']
    lines += ['## Interpretation and next decision', '',
        'The Stage B implementation and pilot are complete. The gate to larger or mixed pretraining is not met: reconstruction improves over the constant-mean baseline, but trained embeddings do not consistently improve held-out cost retrieval or BO across seeds. One GNN seed beats random search at budget 64 on overlapping targets; none beats neighborhood-first or raw-feature BO in the aggregate at that budget. At budgets 16 and 32, all three trained GNNs trail random search in aggregate.', '',
        'Hierarchical targets have no observed improvement over the supplied SCENT tree, although other trees can be more expensive. Their identical search incumbents do not mean all tree structures have equal costs. On overlapping targets, SCENT is suboptimal in these pools, but the learning advantage remains unproven.', '',
        'Prioritize the independent neighborhood and mechanism checks described above; retain a fixed-data, longer-training GNN diagnostic as a bounded secondary task. All checkpoints selected epoch 99 of 100, so this fixed-budget pilot does not establish convergence. Do not select the best training seed after observing downstream results or expand to 10,000 trees based only on reconstruction.', '']
    audit_path = OUT/'audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += ['## Validation', '',
            f"Artifact audit passed for {audit['trace_steps_checked']} search steps. Minimum final fidelity across the 768 pool labels: {audit['min_final_fidelity']:.12f}. Corpus splits and all original candidate pools are hash-disjoint. Trained models pass node-renumbering and physical-label checks, with finite, varying latent coordinates. These numerical checks do not establish useful cost geometry.", '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
