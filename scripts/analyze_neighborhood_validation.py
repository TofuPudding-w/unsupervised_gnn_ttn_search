"""Audit new-target searches and diagnose selected local moves without changing labels."""
import json
import hashlib
import time
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Isometry
from qiskit.exceptions import QiskitError
from threadpoolctl import threadpool_limits
from ttn_search.evaluate import write_json, source_hash, software
from ttn_search.reference import build, padded_isometry, nbits, fidelity
from ttn_search.targets import gaussian_target
from ttn_search.trees import canonical, tree_hash, neighbors

OUT = Path('artifacts/neighborhood_validation')


def main():
    started = time.perf_counter()
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert protocol['evaluator_source_hash']==source_hash() and protocol['software']==software()
    config = protocol['config']
    search = json.loads((OUT/'search.json').read_text())
    done = json.loads((OUT/'completion.json').read_text())
    records = {p.parent.name:json.loads(p.read_text()) for p in (OUT/'cache').glob('*/record.json')}
    for record in records.values():
        assert record['source_hash'] == protocol['evaluator_source_hash']
        assert record['software_versions'] == protocol['software']
        assert record['pipeline'] == protocol['pipeline']
    analyzed = {}
    def describe(label, key):
        if key in analyzed: return analyzed[key]
        rec = records[key]
        family, seed = label.rsplit('_',1)
        target = gaussian_target(config['n'],family,int(seed))
        tree = canonical(json.loads(rec['tree_serialization']))
        tol = (1-protocol['pipeline']['fidelity'])*protocol['pipeline']['repr_fraction']*rec['tolerance_factor']/(config['n']-1)
        rep = build(target.state,tree,tol)
        assert abs(fidelity(target.state,rep.state)-rec['fidelity_repr'])<1e-10
        rows = []
        def visit(node):
            if node.children is None: return
            a,b = node.children
            row = dict(leaves=sorted(node.labels),children=[sorted(a.labels),sorted(b.labels)],
                rank=node.rank,child_ranks=[a.rank,b.rank],local_parameters=int(node.local.size))
            count = nbits(a.rank)+nbits(b.rank)
            try:
                if count:
                    circuit = QuantumCircuit(count)
                    circuit.append(Isometry(padded_isometry(node),0,0),range(count))
                    compiled = transpile(circuit,basis_gates=['u','cx'],
                        optimization_level=protocol['pipeline']['optimization_level'],
                        seed_transpiler=protocol['pipeline']['compiler_seed'])
                    row['standalone_local_cx']=int(compiled.count_ops().get('cx',0))
                else: row['standalone_local_cx']=0
            except (QiskitError,ValueError,np.linalg.LinAlgError) as exc:
                row['local_synthesis_error']=str(exc)
            rows.append(row)
            visit(a); visit(b)
        visit(rep.root)
        result = dict(tree=tree,evaluation_key=key,cx=rec['cx_count'],fidelity=rec['fidelity_final'],
            tolerance_factor=rec['tolerance_factor'],parameter_count=rep.parameter_count,
            max_bond_dimension=max(rep.ranks),nodes=rows,
            standalone_local_cx_sum=sum(r['standalone_local_cx'] for r in rows) if all('standalone_local_cx' in r for r in rows) else None)
        analyzed[key] = result
        return result
    steps = 0; cases = []; target_rows = []; accepted_moves = []
    for family in config['families']:
        for seed in config['target_seeds']:
            label=f'{family}_{seed}'
            meta=json.loads((OUT/label/'target.json').read_text())
            start=canonical(meta['scent_tree']); initial_hash=tree_hash(start)
            initial_record=next(r for r in records.values() if r['tree_hash']==initial_hash and r['target_hash']==meta['target_hash'])
            assert initial_record['status']=='success'
            adjacent={tree_hash(t) for t in neighbors(start)}
            # All initial one-hop neighbors should be covered at the maximum budget.
            neighbor_records={r['tree_hash']:r for r in records.values() if r['target_hash']==meta['target_hash'] and r['tree_hash'] in adjacent}
            assert len(neighbor_records)==len(adjacent)
            feasible=[r for r in neighbor_records.values() if r['status']=='success']
            random_keys=set()
            for search_seed in config['search_seeds']:
                random_trace=json.loads((OUT/label/f'random_{search_seed}.json').read_text())
                random_keys.update(r['evaluation_key'] for r in random_trace)
            random_records=[records[k] for k in random_keys]
            random_feasible=[r for r in random_records if r['status']=='success']
            target_rows.append(dict(target=label,baseline=initial_record['cx_count'],neighbors=len(adjacent),
                feasible_neighbors=len(feasible),improving_neighbors=sum(r['cx_count']<initial_record['cx_count'] for r in feasible),
                neighbor_min=min(r['cx_count'] for r in feasible) if feasible else None,
                random_unique=len(random_records),random_feasible=len(random_feasible),
                random_improving=sum(r['cx_count']<initial_record['cx_count'] for r in random_feasible)))
            for search_seed in config['search_seeds']:
                for method in config['methods']:
                    trace=json.loads((OUT/label/f'{method}_{search_seed}.json').read_text())
                    assert len(trace)==max(config['additional_budgets'])
                    seen={initial_hash}; best=initial_record['cx_count']; current=start
                    for row in trace:
                        assert row['tree_hash'] not in seen
                        if row['action']=='neighbor':
                            anchor=start if method=='initial_neighborhood' else current
                            assert canonical(row['anchor'])==anchor
                            assert row['tree_hash'] in {tree_hash(t) for t in neighbors(anchor)}
                        elif method != 'random':
                            anchor=start if method=='initial_neighborhood' else current
                            assert not ({tree_hash(t) for t in neighbors(anchor)}-seen)
                        rec=records[row['evaluation_key']]
                        assert rec['tree_hash']==row['tree_hash'] and rec['target_hash']==meta['target_hash']
                        improved=rec['status']=='success' and rec['fidelity_final']>=.999 and rec['cx_count']<best
                        assert row['improved']==improved
                        if improved:
                            prior=records[row['prior_best_key']]
                            accepted_moves.append(dict(target=label,method=method,seed=search_seed,
                                extra_evaluations=row['extra_evaluations'],action=row['action'],
                                before_tree=row['prior_best_tree'],after_tree=row['tree'],
                                before_key=prior['evaluation_key'],after_key=rec['evaluation_key'],
                                before_cx=prior['cx_count'],after_cx=rec['cx_count'],
                                before_max_rank=prior['max_bond_dimension'],after_max_rank=rec['max_bond_dimension'],
                                before_parameters=prior['parameter_count'],after_parameters=rec['parameter_count'],
                                before_tolerance=prior['tolerance_factor'],after_tolerance=rec['tolerance_factor'],
                                after_fidelity=rec['fidelity_final']))
                            best=rec['cx_count']
                            if method=='iterative_neighborhood':current=canonical(row['tree'])
                        assert row['best_cx']==best
                        seen.add(row['tree_hash']);steps+=1
            trace=json.loads((OUT/label/'iterative_neighborhood_0.json').read_text())
            for improving in [True,False]:
                chosen=next((r for r in trace if r['action']=='neighbor' and r['status']=='success' and r['improved']==improving),None)
                if chosen is None: continue
                before=describe(label,chosen['prior_best_key']);after=describe(label,chosen['evaluation_key'])
                old={tuple(r['leaves']) for r in before['nodes']};new={tuple(r['leaves']) for r in after['nodes']}
                cases.append(dict(target=label,selection='first improving neighbor' if improving else 'first feasible non-improving neighbor',
                    search_seed=0,extra_evaluations=chosen['extra_evaluations'],before=before,after=after,
                    removed_partitions=[list(t) for t in sorted(old-new)],added_partitions=[list(t) for t in sorted(new-old)]))
            print(label,'audited and diagnosed',flush=True)
    # Explicitly exploratory: inspect the strongest iterative-vs-random target after results.
    target_differences={}
    for row in target_rows:
        target=row['target']
        means={m:np.mean([r['checkpoints']['32'] for r in search if r['target']==target and r['method']==m])
               for m in ['random','iterative_neighborhood']}
        target_differences[target]=float(means['iterative_neighborhood']-means['random'])
    strongest=min(target_differences,key=lambda t:target_differences[t])
    exploratory=[]
    if target_differences[strongest]<0:
        for move in accepted_moves:
            if move['target']==strongest and move['method']=='iterative_neighborhood' and move['seed']==0:
                exploratory.append(dict(move=move,before=describe(strongest,move['before_key']),after=describe(strongest,move['after_key'])))
    write_json(OUT/'exploratory_best_target_path.json',dict(
        selection='post-result largest target-mean iterative advantage versus random, search seed 0',
        target=strongest,steps=exploratory))
    write_json(OUT/'mechanisms.json',cases)
    write_json(OUT/'accepted_moves.json',accepted_moves)
    write_json(OUT/'audit.json',dict(search_steps=steps,targets=target_rows,selected_mechanism_cases=len(cases),
        analysis_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),analysis_elapsed_s=time.perf_counter()-started,
        minimum_accepted_fidelity=min(r['fidelity_final'] for r in records.values() if r['status']=='success')))
    lines=['# New-target neighborhood validation','',
        'Ten new eight-qubit target instances (five per existing family, seeds 3–7). SCENT exact-MI adapter is the common starting point; all budgets below are additional evaluations. Three search seeds per target. Candidates are generated online without a prebuilt pool or cost-based selection. This is new-instance validation within the existing generators, not an independent new-family or larger-system test.','',
        'Initial-neighborhood search randomly visits unseen one-hop SCENT neighbors, then samples global random trees. Iterative search accepts strict improvements immediately and explores the new incumbent’s neighborhood; it also falls back to random when its unseen neighborhood is exhausted. Improving random fallbacks become new incumbents. Global random trees use pair merging, not uniform sampling over all topologies.','',
        '## Results','',
        '| Family | Method | Extra 8 | Extra 16 | Extra 32 | Mean accepted improvements |','|---|---|---|---|---|---|']
    for family in config['families']:
        for method in config['methods']:
            rows=[r for r in search if r['family']==family and r['method']==method]
            values=[np.mean([r['checkpoints'][str(b)] for r in rows]) for b in config['additional_budgets']]
            lines.append('| '+family+' | '+method+' | '+' | '.join(f'{v:.2f}' for v in values)+f" | {np.mean([r['improvements'] for r in rows]):.2f} |")
    lines+=['','Means average seeds within each target and then targets; the balanced design makes this equal to the displayed grand mean. Seeds are not independent target samples. Lower CNOT is better.','',
        '## Per-target endpoints and initial-neighborhood density','',
        '| Target | SCENT | Improving / feasible / total neighbors | Best neighbor | Random (32) | Initial neighborhood (32) | Iterative (32) |','|---|---|---|---|---|---|---|']
    for row in target_rows:
        values=[np.mean([r['checkpoints']['32'] for r in search if r['target']==row['target'] and r['method']==m]) for m in config['methods']]
        lines.append(f"| {row['target']} | {row['baseline']} | {row['improving_neighbors']} / {row['feasible_neighbors']} / {row['neighbors']} | {row['neighbor_min']} | "+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','## Descriptive improvement density','',
        'Strictly improving feasible candidates divided by all unique evaluated candidates in each group. The initial neighborhood is fully enumerated; random candidates are the deduplicated union of the three fixed-budget global-random runs. These are different-size samples of different distributions, and improvement over SCENT does not mean near-optimality.', '',
        '| Target | Initial neighbors improving / total | Global random improving / total |', '|---|---|---|']
    for row in target_rows:
        lines.append(f"| {row['target']} | {row['improving_neighbors']} / {row['neighbors']} | {row['random_improving']} / {row['random_unique']} |")
    lines+=['','## Paired target comparisons','',
        'Counts compare mean CNOT over search seeds within each target at extra budget 32. A win means a lower mean cost; five targets per family is a small sample.', '',
        '| Family | Comparison | Wins | Ties | Losses | Mean CNOT difference |', '|---|---|---|---|---|---|']
    for family in config['families']:
        for a,b in [('initial_neighborhood','random'),('iterative_neighborhood','random'),('iterative_neighborhood','initial_neighborhood')]:
            differences=[]
            for seed in config['target_seeds']:
                target=f'{family}_{seed}'
                means={m:np.mean([r['checkpoints']['32'] for r in search if r['target']==target and r['method']==m]) for m in [a,b]}
                differences.append(means[a]-means[b])
            d=np.array(differences)
            lines.append(f'| {family} | {a} minus {b} | {np.sum(d < -1e-9)} | {np.sum(np.abs(d) <= 1e-9)} | {np.sum(d > 1e-9)} | {d.mean():.2f} |')
    lines+=['','## Selected move mechanisms','',
        'Selection was fixed to the first successful improving neighbor and first feasible non-improving neighbor in iterative seed 0, if present. These examples are not representative prevalence estimates. Full partitions, node ranks and local costs for these selected cases are in mechanisms.json. Every accepted move, including later improvements and random fallbacks, is also saved in accepted_moves.json with before/after costs, ranks, parameter counts and tolerance factors.','',
        '| Target | Case | CNOT before → after | Parameters before → after | Max rank before → after | Tolerance factor before → after |','|---|---|---|---|---|---|']
    for c in cases:
        a,b=c['before'],c['after']
        lines.append(f"| {c['target']} | {c['selection']} | {a['cx']} → {b['cx']} | {a['parameter_count']} → {b['parameter_count']} | {a['max_bond_dimension']} → {b['max_bond_dimension']} | {a['tolerance_factor']} → {b['tolerance_factor']} |")
    lines+=['','## Exploratory follow-up: strongest iterative improvement','',
        'This section selects the target with the largest mean iterative advantage after observing results. It is a mechanism case study, not another independent validation result. The complete seed-0 path, node partitions and standalone local synthesis counts are in exploratory_best_target_path.json.', '',
        '| Target | Extra evaluation | Action | CNOT before → after | Max rank before → after | Max bond register bits before → after | Local CNOT sum before → after |',
        '|---|---|---|---|---|---|---|']
    for item in exploratory:
        a,b,m=item['before'],item['after'],item['move']
        lines.append(f"| {m['target']} | {m['extra_evaluations']} | {m['action']} | {a['cx']} → {b['cx']} | {a['max_bond_dimension']} → {b['max_bond_dimension']} | {nbits(a['max_bond_dimension'])} → {nbits(b['max_bond_dimension'])} | {a['standalone_local_cx_sum']} → {b['standalone_local_cx_sum']} |")
    lines+=['',
        'Padded isometry registers use ceil(log2(rank)) qubits. Crossing a power-of-two rank boundary can therefore change circuit dimensions abruptly; this is a concrete mechanism to investigate, not proof that all observed CNOT reduction is caused by that boundary. Initial improvements can also occur with unchanged maximum rank, because the distribution of ranks and local maps changes.', '',
        '## Interpretation', '',
        'This run does not support a general advantage for searching the initial SCENT neighborhood. All five new overlapping targets have only four improving one-hop neighbors out of twelve, and their best one-hop costs remain 502–504 while random search reaches much lower costs. The earlier small-budget advantage did not transfer to this SCENT-only initialization and pool-free protocol.', '',
        'Iterative search has a lower average endpoint at extra budget 32, but wins on only one of five overlapping targets and loses on four. The winning target is overlapping_4: all three iterative seeds reach 273 through accepted neighbor moves, with no improving random fallback on those paths. This supports investigating a reproducible multi-step improvement on that instance, not claiming reliable population-wide superiority. At budgets 8 and 16, random search remains better on average. Hierarchical targets show no improvement over SCENT.', '',
        'Next, inspect the rank/padded-register transition and the other targets failing to reach a comparable transition under equal budgets. Keep random search as a strong baseline. Do not scale or rename the method based on the mean alone; a general recommendation to prioritize local search over other approaches is not yet justified.', '']
    lines+=['','Standalone local synthesis uses the same basis, compiler seed, and optimization level but separate circuits. Its counts are diagnostic and need not sum to the globally optimized circuit count. Local synthesis errors are recorded, not treated as zero cost. Parameter or rank changes are associations, not a causal attribution; differences in tolerance attempts must also be considered.','',
        '## Accounting and validation','',
        f"Completed {done['trajectories']} trajectories and audited {steps} additional search steps. Executed {done['new_evaluations']} uncached target–tree evaluations; search run elapsed time {done['elapsed_s']:.1f} s. Mechanism analysis time is separate from this run time.",'',
        'Per-run actual wall times and charged historical evaluation times are in search.json; per-query cache hits and lookup wall times are in traces. Shared caching favors later methods in actual wall time, so no independent online speedup is claimed. SCENT construction time is stored per target and is a common prerequisite. No hard evaluation time or memory caps were introduced. The original GNN was not retrained in this experiment.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
