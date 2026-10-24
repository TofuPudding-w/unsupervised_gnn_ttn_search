"""Descriptive audit of existing labels; no training or new circuit evaluations."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from qiskit import qpy
from ttn_search.evaluate import write_json
from analyze_c1_depth import two_qubit_depth

OUT = Path('artifacts/screening_headroom')


def load(path):
    return json.loads(Path(path).read_text())


def summarize(target, scores, records, exhaustive):
    """Deduplicated candidates, with full-pool score ranks and measured costs only."""
    rows = []
    inconsistent = []
    for i, observations in sorted(records.items()):
        signatures = {(r['status'], r.get('cx_count'), r['tolerance_factor']) for r in observations}
        if len(signatures) != 1:
            inconsistent.append(dict(candidate=i, signatures=[list(s) for s in signatures]))
        r = observations[0]
        if r['status'] != 'success':
            continue
        with Path(r['circuit_path']).open('rb') as stream:
            depth = two_qubit_depth(qpy.load(stream)[0])
        rows.append(dict(candidate_id=i, **scores[i], cx=r['cx_count'], d2=depth,
            accepted_factor=r['tolerance_factor'], retry=len(r['attempts']) > 1,
            observations=len(observations), record_path=r['_source']))
    assert not inconsistent, f'Conflicting outcomes: {inconsistent}'
    best = min(r['cx'] for r in rows)
    ties = []
    for score in ['padded', 'parameters']:
        groups = defaultdict(list)
        for row in rows:
            groups[row[score]].append(row)
        for value, group in groups.items():
            if len(group) < 2:
                continue
            ties.append(dict(score=score, value=value, count=len(group),
                min_cx=min(r['cx'] for r in group), max_cx=max(r['cx'] for r in group),
                min_d2=min(r['d2'] for r in group), max_d2=max(r['d2'] for r in group),
                retries=sum(r['retry'] for r in group), ids=[r['candidate_id'] for r in group]))
    orders = {}; curves = {}; ranks = {}
    measured = {r['candidate_id']: r for r in rows}
    tie_order = np.random.default_rng(0).permutation(len(scores)).tolist()
    for score in ['padded', 'parameters']:
        order = sorted(tie_order, key=lambda i: scores[i][score])
        orders[score] = order
        ranks[score] = {i: j+1 for j, i in enumerate(order)}
        curve = []
        for k in [1, 2, 4, 8, 16]:
            known = [measured[i] for i in order[:k] if i in measured]
            complete = len(known) == k
            value = min((r['cx'] for r in known), default=None)
            curve.append(dict(k=k, fully_measured=complete, measured=len(known), best_cx=value,
                regret_to_observed_best=value-best if value is not None else None))
        curves[score] = curve
    for row in rows:
        row['padded_rank'] = ranks['padded'][row['candidate_id']]
        row['parameter_rank'] = ranks['parameters'][row['candidate_id']]
    # Counterexamples: adjacent measured candidates in fixed padded order with lower later cost.
    ordered = sorted(rows, key=lambda r: r['padded_rank'])
    inversions = []
    for a, b in zip(ordered, ordered[1:]):
        if b['cx'] < a['cx']:
            inversions.append(dict(earlier=a['candidate_id'], later=b['candidate_id'],
                earlier_cx=a['cx'], later_cx=b['cx'], same_score=a['padded']==b['padded'],
                retry_in_pair=a['retry'] or b['retry']))
    correlations = {}
    for name, subset in [('all', rows), ('no_retry', [r for r in rows if not r['retry']])]:
        for score in ['padded', 'parameters']:
            value = float(spearmanr([r[score] for r in subset], [r['cx'] for r in subset]).statistic)
            correlations[f'{name}_{score}'] = value if np.isfinite(value) else None
    return dict(target=target, exhaustive=exhaustive, pool_size=len(scores), measured=len(rows),
        best_observed=best, scent=measured[0]['cx'], retry_candidates=sum(r['retry'] for r in rows),
        correlations=correlations, curves=curves, rows=rows, ties=ties, inversions=inversions)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = load('artifacts/revised_policy/pool.json')
    scores = {r['candidate_id']: {k: r[k] for k in ['parameters', 'padded']} for r in old}
    records = defaultdict(list)
    for row in old:
        path = row['record_path']; record = load(path); record['_source'] = path
        records[row['candidate_id']].append(record)
    results = [summarize('overlapping_13', scores, records, True)]
    runs = load('artifacts/revised_gnn/runs.json')
    for seed in range(14, 19):
        target = f'overlapping_{seed}'
        score_rows = load(f'artifacts/revised_gnn/{target}/padded_0/scores.json')
        scores = {r['candidate_id']: {k: r[k] for k in ['parameters', 'padded']} for r in score_rows}
        records = defaultdict(list)
        for run in runs:
            if run['target'] != target:
                continue
            for row in run['trace']:
                path = row['record_path']; record = load(path); record['_source'] = path
                records[row['candidate_id']].append(record)
        results.append(summarize(target, scores, records, False))
    timing = []
    for method in ['scent', 'random', 'random_scent', 'parameters', 'padded', 'gnn_2000', 'gnn_10000']:
        group = [r for r in runs if r['method'] == method]
        timing.append(dict(method=method, **{key:float(np.mean([r[key] for r in group])) for key in
            ['prescreen_s', 'prescreen_build_s', 'ranking_s', 'compile_s', 'circuit_verification_s', 'retry_s', 'actual_wall_s']}))
    write_json(OUT / 'audit.json', dict(targets=results, timing=timing,
        caveats=['Seeds 14–18 are adaptively selected observed unions, not exhaustive or IID samples.',
            'Repeated evaluations are deduplicated; target count is six, all ten-qubit overlapping.',
            'Fixed order is seed-0 tie shuffle followed by stable ascending score.',
            'Rank/width features are from first 3x construction; labels include all frozen retries.',
            'No new circuits, model fitting, or held-out generalization claims.']))
    lines = ['# Screening headroom audit', '',
        'This audit uses the frozen revised evaluator. It does not train a model or compile new candidates. Candidate repeats are deduplicated and checked for identical status, CNOT and accepted factor. Depth is independently read from one accepted QPY per candidate.', '',
        '**Only seed 13 is exhaustive.** Seeds 14–18 contain the union selected by previous search methods; correlations, tie statistics and the observed best are selection-biased descriptive quantities. They are not estimates for an unseen full pool. All six targets are ten-qubit overlapping instances; there is no cross-family conclusion.', '',
        '| Target | Evaluated / pool | SCENT | Best observed | Retry candidates | Padded Spearman | No-retry padded Spearman |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['target']} | {r['measured']}/{r['pool_size']} | {r['scent']} | {r['best_observed']} | {r['retry_candidates']} | {r['correlations']['all_padded']:.3f} | {r['correlations']['no_retry_padded']:.3f} |")
    lines += ['', '## Top-K retrieval', '',
        'Numbers are CNOT regret to the best **observed** candidate. For seed 13 this is the pool optimum. All listed prefixes must be fully measured; no unknown label is imputed. Ties use the original seed-0 randomized order, not hindsight cost sorting.', '',
        '| Target | Score | K=1 | K=2 | K=4 | K=8 | K=16 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        for score in ['padded', 'parameters']:
            curve = r['curves'][score]
            assert all(c['fully_measured'] for c in curve)
            lines.append('| '+r['target']+' | '+score+' | '+' | '.join(str(c['regret_to_observed_best']) for c in curve)+' |')
    lines += ['', '## Equal-score ambiguity', '',
        'Equal padded size does not uniquely determine circuit cost or depth. The following selects the largest CNOT spread per target, including failures/retries rather than hiding them. These are mechanism counterexamples, not evidence that improving these often-expensive candidates changes the useful shortlist.', '',
        '| Target | Padded score | Measured tied trees | CNOT range | D2 range | Retrying trees |', '|---|---:|---:|---|---|---:|']
    for r in results:
        tie = max((t for t in r['ties'] if t['score']=='padded'), key=lambda t:t['max_cx']-t['min_cx'])
        lines.append(f"| {r['target']} | {tie['value']} | {tie['count']} | {tie['min_cx']}–{tie['max_cx']} | {tie['min_d2']}–{tie['max_d2']} | {tie['retries']} |")
    lines += ['', 'The JSON preserves every tie group, each candidate’s two score ranks, accepted tolerance and depth, and adjacent ranking inversions with retry flags. Thus failed-attempt behavior can be separated from ordinary synthesis variation.', '',
        '## Depth cross-check and ranking counterexamples', '',
        '| Target | First padded D2 | Lowest observed D2 | Adjacent cost inversions | Inversions touching retry |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        first = next(row for row in r['rows'] if row['padded_rank'] == 1)
        lines.append(f"| {r['target']} | {first['d2']} | {min(row['d2'] for row in r['rows'])} | {len(r['inversions'])} | {sum(v['retry_in_pair'] for v in r['inversions'])} |")
    lines += ['', 'An inversion is a lower-cost candidate immediately after a higher-cost candidate in the measured subset of the fixed padded order. This includes score ties. These counts are descriptive, not pairwise error rates for the full pool.', '',
        '## Actual existing-run timing', '',
        'Means over the five targets and three repeats. Prescreen includes 128 initial builds and representation verification. Retry time overlaps build/compile/verification totals; do not sum it again. These are fixed-evaluation runs, not uncapped fixed-time experiments.', '',
        '| Method | Whole-pool prescreen s | Build component s | Compile s | Final verification s | Retry s | Total online s |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in timing:
        lines.append('| '+r['method']+' | '+' | '.join(f"{r[k]:.3f}" for k in ['prescreen_s','prescreen_build_s','compile_s','circuit_verification_s','retry_s','actual_wall_s'])+' |')
    lines += ['', '## Decision', '',
        'Padded screening has zero regret to the best observed CNOT at K=1 on all six audited targets. Therefore this dataset demonstrates no remaining top-K CNOT benefit for a learned reranker over the frozen padded order. Nonzero global ranking error and equal-score cost variation alone do not justify training: they can concern candidates that should never enter the shortlist.', '',
        'This does not prove padded screening is optimal in the incompletely measured pools, nor does it rule out depth-oriented selection. First evaluate any proposed depth objective explicitly, rather than interpreting CNOT-incumbent depth as minimum achievable depth.', '',
        'The prescreen time is an upper bound on the online time a perfect zero-cost descriptor shortcut could save while reproducing the same selected candidates. A real model must pay for target features, inference and selected TTN builds, and can lose quality. At this size, measure these costs before constructing a supervised dataset for speedup claims.', '',
        'Recommended next action: stabilize/version the compiler first. Then decide whether to complete the small remaining pool subsets for stronger headroom evidence or run a narrowly timed descriptor-shortcut feasibility test. Do not start a target-conditioned GNN ranking study solely because global correlations are imperfect. Historical verified labels remain valid for their frozen failure/fallback policy; a changed compiler needs separately versioned labels.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps([dict(target=r['target'], measured=r['measured'], retries=r['retry_candidates']) for r in results], indent=2), flush=True)


if __name__ == '__main__':
    main()
