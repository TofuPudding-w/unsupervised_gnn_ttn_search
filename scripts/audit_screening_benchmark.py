"""Independent consistency audit of frozen pools, rankings, and measured prefixes."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from ttn_search.evaluate import source_hash, write_json
from ttn_search.trees import canonical, tree_hash

OUT = Path('artifacts/screening_benchmark')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    protocol = read(OUT/'protocol.json')
    config = protocol['config']
    assert config == read('configs/screening_benchmark.json')
    source_review = None
    if source_hash() != protocol['evaluator_source_hash']:
        # Documentation edits can occur while the process retains imported code.
        # Recover the exact frozen bytes, then allow only AST-identical doc edits.
        class RemoveDocs(ast.NodeTransformer):
            def generic_visit(self, node):
                super().generic_visit(node)
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                        node.body = node.body[1:]
                return node
        snapshot = OUT/'source_snapshot'
        files = sorted(snapshot.glob('*.py'))
        assert [p.name for p in files] == [p.name for p in sorted(Path('src/ttn_search').glob('*.py'))]
        h = hashlib.sha256()
        for path in files:
            h.update(path.name.encode()); h.update(path.read_bytes())
            before = RemoveDocs().visit(ast.parse(path.read_bytes()))
            after = RemoveDocs().visit(ast.parse((Path('src/ttn_search')/path.name).read_bytes()))
            assert ast.dump(before) == ast.dump(after), path
        assert h.hexdigest() == protocol['evaluator_source_hash']
        source_review = read(OUT/'source_review.json')
    for path, digest in {**protocol['files'], **protocol['model_hashes']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256(Path('artifacts/stage_b/corpus.json').read_bytes()).hexdigest() == protocol['corpus_hash']
    runs = read(OUT/'runs.json')
    expected = {(f'{f}_{s}', m, r) for f in config['families'] for s in config['target_seeds']
                for m in config['methods'] for r in config['repeats']}
    assert len(runs) == len(expected)
    assert {(r['target'], r['method'], r['repeat']) for r in runs} == expected
    records = calls = retries = 0
    fidelities = []
    for run in runs:
        directory = OUT/run['target']/f"{run['method']}_{run['repeat']}"
        assert read(directory/'summary.json') == run
        assert read(directory/'trace.json') == run['trace']
        pool = read(OUT/run['target']/'pool.json')
        hashes = [tree_hash(canonical(t)) for t in pool['trees']]
        assert hashes == pool['hashes'] and len(set(hashes)) == config['pool_size']
        screen = run['method'] in ['parameters', 'padded']
        rng = np.random.default_rng(run['repeat'])
        if screen:
            score_rows = read(directory/'scores.json')
            assert len(score_rows) == len(hashes)
            assert [s['candidate_id'] for s in score_rows] == list(range(len(hashes)))
            key = run['method']
            values = np.array([s[key] if s[key] is not None else np.inf for s in score_rows])
            tie = rng.permutation(len(hashes))
            order = tie[np.argsort(values[tie], kind='stable')].tolist()
            assert run['prescreen_s'] >= run['prescreen_build_s'] + run['prescreen_repr_verify_s']
        elif run['method'] == 'random':
            order = rng.permutation(len(hashes)).tolist()
        elif run['method'] == 'random_scent':
            order = [0] + rng.permutation(np.arange(1, len(hashes))).tolist()
        elif run['method'] == 'scent':
            order = [0]
        else:
            order = None
            assert run['trace'][0]['candidate_id'] == 0
        seen = set(); incumbent = None; cumulative = 0; elapsed = 0.
        build_s = run['prescreen_build_s']; repr_s = run['prescreen_repr_verify_s']
        compile_s = verify_s = retry_s = 0.
        for k, row in enumerate(run['trace'], 1):
            i = row['candidate_id']
            assert i not in seen
            seen.add(i)
            if order is not None:
                assert i == order[k-1]
            rec = read(row['record_path'])
            assert rec['tree_hash'] == hashes[i] == row['tree_hash']
            assert rec['target_hash'] == pool['target_hash']
            assert rec['pipeline'] == protocol['pipeline'] and not rec['cache_hit']
            assert rec['initial_reused'] == screen == row['initial_reused']
            assert rec['attempts'][0]['reused_initial'] == screen
            assert all(not a['reused_initial'] for a in rec['attempts'][1:])
            factors = [a['tolerance_factor'] for a in rec['attempts']]
            assert factors == protocol['pipeline']['tolerance_factors'][:len(factors)]
            assert rec['compiler_calls'] == sum(a['compile_attempted'] for a in rec['attempts'])
            assert np.isclose(rec['incremental_build_s'], sum(a['time_build_s'] for a in rec['attempts'] if not a['reused_initial']))
            if rec['status'] == 'success':
                assert rec['fidelity_final'] >= protocol['pipeline']['fidelity']
                assert 1-rec['fidelity_repr'] <= .0001+1e-12
                assert 1-rec['fidelity_comp'] <= .0004+1e-12
                assert Path(rec['circuit_path']).is_file()
                fidelities.append(rec['fidelity_final'])
                incumbent = rec['cx_count'] if incumbent is None else min(incumbent, rec['cx_count'])
            cumulative += rec['compiler_calls']
            build_s += rec['incremental_build_s']; repr_s += rec['incremental_repr_verify_s']
            compile_s += rec['compile_s']; verify_s += rec['verify_s']; retry_s += rec['retry_s']
            assert row['k'] == k and row['best_cx'] == incumbent and row['compiler_calls'] == cumulative
            assert elapsed < row['elapsed_s'] <= run['actual_wall_s']
            elapsed = row['elapsed_s']
            for key, value in [('construction_s',build_s), ('representation_verification_s',repr_s),
                               ('compile_s',compile_s), ('verify_s',verify_s), ('retry_s',retry_s)]:
                assert np.isclose(row[key],value), key
            records += 1; calls += rec['compiler_calls']; retries += len(rec['attempts'])-1
        assert run['compiler_calls'] == cumulative
        assert run['candidate_evaluations'] == len(run['trace']) == (1 if run['method']=='scent' else 16)
    completion = read(OUT/'completion.json')
    assert completion['candidate_evaluations'] == records
    assert completion['actual_compiler_calls'] == calls
    assert np.isclose(completion['summed_method_wall_s'], sum(r['actual_wall_s'] for r in runs))
    result = dict(status='passed', source_review=source_review, runs=len(runs), candidate_evaluations=records,
                  compiler_calls=calls, retry_attempts=retries, feasible_evaluations=len(fidelities),
                  minimum_fidelity=min(fidelities),
                  checks=['Frozen code, config, corpus and models', 'Identical unique candidate pools',
                          'Label-free score ordering and seeded random orders', 'No cross-method cache',
                          'Prepared TTN reuse and retry schedule', 'Fidelity constraints and saved circuits',
                          'Prefix incumbents, monotonic actual clocks and synthesis counts',
                          'Construction and verification charged exactly once'])
    write_json(OUT/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
