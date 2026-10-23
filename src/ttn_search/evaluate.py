"""One evaluator, shared precision rules, atomic cache and concrete circuit artifacts."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import time
import traceback
import numpy as np
from qiskit import qpy
from qiskit.exceptions import QiskitError
from .reference import build, compile_reference, circuit_state, fidelity
from .trees import tree_hash, serialize


@dataclass(frozen=True)
class Pipeline:
    fidelity: float = 0.999
    repr_fraction: float = 0.1
    comp_fraction: float = 0.4
    # The tolerance factors are used to scale the error budget 
    # for representation and compilation.
    tolerance_factors: tuple = (1., 0.1, 0.)
    optimization_level: int = 1
    compiler_seed: int = 17

    def __post_init__(self):
        # Check if the fidelity and error allocation parameters are valid.
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
    """
    Evaluate a single tree against a target state with a fixed pipeline,
    caching results and artifacts.
    Returns a record dictionary with fidelity, circuit, and cache information.
    ----
    Parameters:
    - target: A Target object containing the quantum state and metadata.
    - tree: A tree structure defining the TTN representation.
    - pipeline: A Pipeline object specifying fidelity and compilation parameters.
    - cache_dir: A directory path for caching evaluation results and artifacts.
    ----
    Returns:
    - A dictionary containing evaluation results, including fidelity metrics,
      circuit information, cache hit status, and timing details.
    """

    # 1. Generate a unique key for the evaluation based on the target, tree, and pipeline.

    started = time.perf_counter()
    identity = dict(target_hash=target.hash, tree_hash=tree_hash(tree), pipeline=asdict(pipeline),
                    software_versions=software(), source_hash=source_hash())
    key = digest(identity)

    # 2. Check if the evaluation result is already cached; if so, read it.

    # joint the cache directory with the unique key to form the full path
    directory = Path(cache_dir) / key     
    record_path = directory / 'record.json'
    if record_path.exists():
        result = json.loads(record_path.read_text())
        if not result.get('circuit_path') or Path(result['circuit_path']).exists():
            return {**result, 'cache_hit': True, 'lookup_time_s': time.perf_counter()-started}

    # 3. If not cached, perform the evaluation: 
    # build, compile, and verify the TTN representation.
        
    directory.mkdir(parents=True, exist_ok=True)
    # Records combine numeric metrics with status strings, paths and diagnostics.
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    error = 1 - pipeline.fidelity
    for factor in pipeline.tolerance_factors:
        attempt: dict[str, Any] = dict(
            tolerance_factor=factor, time_build_s=0., time_compile_s=0., time_verify_s=0.
        )
        phase = 'build'
        start = time.perf_counter()
        try:
            start = time.perf_counter()
            # Build the TTN representation of the target state based on the provided tree structure.
            rep = build(target.state, tree, error*pipeline.repr_fraction*factor / max(1, len(target.state).bit_length()-2))
            attempt['time_build_s'] = time.perf_counter()-start

            # 4. Verify the representation's fidelity and isometry error against the target state.

            f_repr = fidelity(target.state, rep.state)
            attempt.update(fidelity_repr=f_repr, max_bond_dimension=max(rep.ranks),
                           parameter_count=rep.parameter_count, isometry_error=rep.isometry_error)

            # 5. Check if the representation is feasible within the allocated error budget; 
            # if not, mark as infeasible.
            
            if 1-f_repr > error*pipeline.repr_fraction + 1e-12:
                attempt['status'] = 'representation_infeasible'
                attempts.append(attempt)
                continue

            # 6. Compile the TTN representation into a quantum circui.

            phase = 'compile'
            start = time.perf_counter()
            circuit = compile_reference(rep, pipeline.compiler_seed, pipeline.optimization_level)
            attempt['time_compile_s'] = time.perf_counter()-start

            # 7. Verify the compiled circuit against the target state, 
            # checking fidelity and resource counts.

            phase = 'verify'
            start = time.perf_counter()
            # Compute the output state of the compiled circuit
            output = circuit_state(circuit)
            # Calculate the fidelity of the representation and the final output state against the target state.
            fc, ff = fidelity(rep.state, output), fidelity(target.state, output)
            attempt.update(fidelity_comp=fc, fidelity_final=ff,
                cx_count=int(circuit.count_ops().get('cx', 0)), depth_total=circuit.depth(),
                depth_two_qubit=circuit.depth(filter_function=lambda inst: len(inst.qubits) == 2))
            attempt['time_verify_s'] = time.perf_counter()-start
            # Determine if the final fidelity meets the required threshold
            feasible = ff >= pipeline.fidelity and 1-fc <= error*pipeline.comp_fraction + 1e-12
            attempt['status'] = 'success' if feasible else 'infeasible'

            # 8. Save the compiled circuit and state artifacts to the cache directory.

            circuit_path = (directory / f'attempt_{len(attempts)}.qpy').resolve()
            with circuit_path.open('wb') as stream:
                qpy.dump(circuit, stream)
            attempt['circuit_path'] = str(circuit_path)
            np.savez_compressed(directory / f'attempt_{len(attempts)}.npz', target=target.state,
                                representation=rep.state, circuit_state=output)
            # Fixed reference policy stops at first feasible tolerance; 
            # one deterministic synthesis per attempt.
            if feasible:
                best = dict(attempt)
                attempts.append(attempt)
                break
        except (QiskitError, ValueError, np.linalg.LinAlgError) as exc:
            attempt[f'time_{phase}_s'] = time.perf_counter()-start
            status = 'compiler_error' if isinstance(exc, QiskitError) and phase == 'compile' else 'numerical_error'
            attempt.update(status=status, error=str(exc), error_type=type(exc).__name__,
                           failed_phase=phase, traceback=traceback.format_exc())
        attempts.append(attempt)

    # 9. If no feasible attempt was found, select the best attempt based on fidelity metrics.
    
    if best is None:
        measured = [a for a in attempts if 'fidelity_final' in a]
        if measured:
            best = max(measured, key=lambda a: a['fidelity_final']).copy()
        else:
            # Keep the actual operational failure; never invent a fidelity or
            # label a compiler failure as measured representation infeasibility.
            failures = [a for a in attempts if a['status'] in ('compiler_error', 'numerical_error')]
            best = dict(failures[-1] if failures else attempts[-1])
    result = dict(**identity, **best, evaluation_key=key, tree_serialization=serialize(tree),
                  target_metadata=target.metadata, fidelity_method='exact_statevector',
                  attempts=attempts, evaluation_time_s=time.perf_counter()-started,
                  compiler_policy='one_exact_synthesis_per_tolerance; stop_first_feasible',
                  resource_limits='No hard timeout or memory enforcement in initial small-system pilot')
    write_json(record_path, result)
    return {**result, 'cache_hit': False, 'lookup_time_s': 0.}
