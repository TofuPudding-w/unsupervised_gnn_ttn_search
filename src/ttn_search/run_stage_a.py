"""Run development diagnostics/search. Artifacts distinguish cached replay from actual work."""
import argparse
from dataclasses import asdict
from pathlib import Path
import platform
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .evaluate import Pipeline, evaluate, write_json, source_hash, software
from .scent import scent_tree
from .targets import gaussian_target
from .trees import candidate_pool, serialize
from .search import initial_ids, run_search


def run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    pipeline = Pipeline(**config.get('pipeline', {}))
    manifest = dict(config=config, pipeline=asdict(pipeline), software=software(), source_hash=source_hash(),
                    platform=platform.platform(), threads=1,
                    limitations=['reference exact compiler only', 'SCENT exact-MI variant',
                    'shared cache timing is charged replay, not measured independent online speed',
                    'small development run; no statistical learning claim',
                    'no hard per-candidate timeout or memory cap yet'])
    write_json(output/'manifest.json', manifest)
    summaries = []
    for family in config['families']:
        for target_seed in config['target_seeds']:
            target = gaussian_target(config['n'], family, target_seed)
            task = output/f'{family}_{target_seed}'
            task.mkdir(exist_ok=True)
            scent, scent_info = scent_tree(target.state, config['pool_seed'])
            pool_start = time.perf_counter()
            pool = candidate_pool(config['n'], config['pool_size'], config['pool_seed'], scent)
            pool_time = time.perf_counter()-pool_start
            write_json(task/'target.json', target.metadata)
            write_json(task/'pool.json', dict(trees=[serialize(t) for t in pool], scent=scent_info,
                                             time_pool_s=pool_time))
            results = {}
            def oracle(i):
                rec = evaluate(target, pool[i], pipeline, output/'cache')
                results[i] = rec
                print(f'{family}/{target_seed} tree={i} {rec["status"]} '
                      f'cx={rec.get("cx_count")} F={rec.get("fidelity_final")} '
                      f'cache={rec["cache_hit"]}', flush=True)
                return rec
            diagnostic = min(config.get('diagnostic_count', 0), len(pool))
            for i in range(diagnostic):
                oracle(i)
            if 0 not in results:
                oracle(0)
            trajectories = {}
            for seed in config['search_seeds']:
                init = initial_ids(len(pool), config['initial_count'], seed)
                for method in ('random', 'raw', 'pca'):
                    name = f'{method}_{seed}'
                    trace = run_search(pool, oracle, method, config['budget'], init, pipeline.fidelity, seed,
                        on_step=lambda trace, name=name: write_json(task/f'{name}.json', trace))
                    trajectories[name] = trace[-1]
            feasible = [r for r in results.values() if r['status']=='success']
            summary = dict(family=family, seed=target_seed, evaluated_unique=len(results),
                feasible_count=len(feasible), scent_cx=results[0].get('cx_count'),
                scent_status=results[0]['status'], best_seen_cx=min((r['cx_count'] for r in feasible), default=None),
                worst_seen_cx=max((r['cx_count'] for r in feasible), default=None),
                minimum_final_fidelity=min((r['fidelity_final'] for r in feasible), default=None),
                time_scent_s=scent_info['time_s'], time_pool_s=pool_time,
                unique_evaluation_time_s=sum(r['evaluation_time_s'] for r in results.values()),
                trajectories=trajectories)
            summaries.append(summary)
            write_json(task/'summary.json', summary)
    write_json(output/'summary.json', dict(targets=summaries, elapsed_actual_s=time.perf_counter()-start))
    lines = ['# Stage A development run', '',
        'Backend: dense hierarchical SVD and exact Qiskit isometry synthesis. SCENT uses exact mutual information and upstream recursive spectral partitioning.', '',
        '**Development observations only.** Cached replay is not actual online speed; no GNN is trained. No independent-test or statistical superiority claim is made.', '',
        '| Target | Unique evaluations | Feasible | SCENT CX | Best seen CX | Worst seen CX |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'| {s["family"]}/{s["seed"]} | {s["evaluated_unique"]} | {s["feasible_count"]} | {s["scent_cx"]} | {s["best_seen_cx"]} | {s["worst_seen_cx"]} |')
    lines += ['', '## Search endpoints', '', '| Target | Method/seed | Budget | Best CX |', '|---|---|---:|---:|']
    for s in summaries:
        for name, trace in s['trajectories'].items():
            lines.append(f'| {s["family"]}/{s["seed"]} | {name} | {trace["step"]} | {trace["best_cx"]} |')
    lines += ['', 'Times in trajectory records distinguish actual elapsed time and charged evaluation replay; add target SCENT/pool overhead for full online accounting.',
              'Source fingerprint, environment, target parameters, candidate pools, attempts, QPY circuits and exact-state evidence are retained alongside this report.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/stage_a_smoke.json')
    parser.add_argument('--output', default='artifacts/stage_a_smoke')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    with threadpool_limits(limits=1):
        run(config, args.output)


if __name__ == '__main__':
    main()
