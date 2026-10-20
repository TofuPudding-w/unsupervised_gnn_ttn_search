"""One evaluator, shared precision rules, atomic cache and concrete circuit artifacts."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import time
import traceback
import numpy as np
from qiskit import qpy
from .reference import build, compile_reference, circuit_state, fidelity
from .trees import tree_hash, serialize


@dataclass(frozen=True)
class Pipeline:
    fidelity: float = 0.999
    repr_fraction: float = 0.1
    comp_fraction: float = 0.4
    tolerance_factors: tuple = (1., 0.1, 0.)
    optimization_level: int = 1
    compiler_seed: int = 17

    def __post_init__(self):
        if not 0 < self.fidelity < 1 or min(self.repr_fraction, self.comp_fraction) < 0:
            raise ValueError('Invalid fidelity or error allocation')
        if (self.repr_fraction**0.5 + self.comp_fraction**0.5)**2 > 1 + 1e-12:
            raise ValueError('Stage error budgets exceed final error bound')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def software():
    return {p: version(p) for p in ('numpy', 'scipy', 'qiskit', 'scikit-learn', 'networkx')}


def source_hash():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def evaluate(target, tree, pipeline, cache_dir):
    started = time.perf_counter()
    identity = dict(target_hash=target.hash, tree_hash=tree_hash(tree), pipeline=asdict(pipeline),
                    software_versions=software(), source_hash=source_hash())
    key = digest(identity)
    directory = Path(cache_dir) / key
    record_path = directory / 'record.json'
    if record_path.exists():
        result = json.loads(record_path.read_text())
        if not result.get('circuit_path') or Path(result['circuit_path']).exists():
            return {**result, 'cache_hit': True, 'lookup_time_s': time.perf_counter()-started}
    directory.mkdir(parents=True, exist_ok=True)
    attempts, best = [], None
    error = 1 - pipeline.fidelity
    for factor in pipeline.tolerance_factors:
        attempt = dict(tolerance_factor=factor, time_build_s=0., time_compile_s=0., time_verify_s=0.)
        try:
            start = time.perf_counter()
            rep = build(target.state, tree, error*pipeline.repr_fraction*factor / max(1, len(target.state).bit_length()-2))
            attempt['time_build_s'] = time.perf_counter()-start
            f_repr = fidelity(target.state, rep.state)
            attempt.update(fidelity_repr=f_repr, max_bond_dimension=max(rep.ranks),
                           parameter_count=rep.parameter_count, isometry_error=rep.isometry_error)
            if 1-f_repr > error*pipeline.repr_fraction + 1e-12:
                attempt['status'] = 'representation_infeasible'
                attempts.append(attempt)
                continue
            start = time.perf_counter()
            circuit = compile_reference(rep, pipeline.compiler_seed, pipeline.optimization_level)
            attempt['time_compile_s'] = time.perf_counter()-start
            start = time.perf_counter()
            output = circuit_state(circuit)
            fc, ff = fidelity(rep.state, output), fidelity(target.state, output)
            attempt.update(fidelity_comp=fc, fidelity_final=ff,
                cx_count=int(circuit.count_ops().get('cx', 0)), depth_total=circuit.depth(),
                depth_two_qubit=circuit.depth(filter_function=lambda inst: len(inst.qubits) == 2))
            attempt['time_verify_s'] = time.perf_counter()-start
            feasible = ff >= pipeline.fidelity and 1-fc <= error*pipeline.comp_fraction + 1e-12
            attempt['status'] = 'success' if feasible else 'infeasible'
            circuit_path = (directory / f'attempt_{len(attempts)}.qpy').resolve()
            with circuit_path.open('wb') as stream:
                qpy.dump(circuit, stream)
            attempt['circuit_path'] = str(circuit_path)
            np.savez_compressed(directory / f'attempt_{len(attempts)}.npz', target=target.state,
                                representation=rep.state, circuit_state=output)
            # Fixed reference policy stops at first feasible tolerance; one deterministic synthesis per attempt.
            if feasible:
                best = dict(attempt)
                attempts.append(attempt)
                break
        except (ValueError, np.linalg.LinAlgError) as exc:
            attempt.update(status='numerical_error', error=str(exc), traceback=traceback.format_exc())
        attempts.append(attempt)
    if best is None:
        measured = [a for a in attempts if 'fidelity_final' in a]
        if measured:
            best = max(measured, key=lambda a: a['fidelity_final']).copy()
        else:
            best = dict(status='numerical_error' if any(a['status']=='numerical_error' for a in attempts) else 'representation_infeasible')
    result = dict(**identity, **best, evaluation_key=key, tree_serialization=serialize(tree),
                  target_metadata=target.metadata, fidelity_method='exact_statevector',
                  attempts=attempts, evaluation_time_s=time.perf_counter()-started,
                  compiler_policy='one_exact_synthesis_per_tolerance; stop_first_feasible',
                  resource_limits='No hard timeout or memory enforcement in initial small-system pilot')
    write_json(record_path, result)
    return {**result, 'cache_hit': False, 'lookup_time_s': 0.}
