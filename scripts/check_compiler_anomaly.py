"""Isolate the pilot's synthesis anomaly without changing the reference compiler."""
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, Isometry
from qiskit.quantum_info import Operator, Statevector
import ttn_search.reference as ref
from ttn_search.targets import gaussian_target
from ttn_search.trees import canonical
from ttn_search.evaluate import write_json, source_hash


def investigate(output):
    records = [json.loads(f.read_text()) for f in Path('artifacts/stage_a_pilot_v2/cache').glob('*/record.json')]
    saved = min(records, key=lambda r:r['fidelity_comp'])
    target = gaussian_target(8, saved['target_metadata']['family'], saved['target_metadata']['seed'])
    tree = canonical(json.loads(saved['tree_serialization']))
    rep = ref.build(target.state, tree, (1-.999)*.1*saved['tolerance_factor']/7)
    original_transpile = ref.transpile
    try:
        ref.transpile = lambda circuit, **kwargs: circuit
        logical = ref.compile_reference(rep)
    finally:
        ref.transpile = original_transpile
    ideal = QuantumCircuit(logical.num_qubits)
    ideal.global_phase = logical.global_phase
    local = []
    for index, inst in enumerate(logical.data):
        matrix = np.asarray(inst.operation.params[0])
        missing = matrix.shape[0]-matrix.shape[1]
        unitary = np.column_stack((matrix, null_space(matrix.conj().T)[:, :missing])) if missing else matrix
        wires = [logical.find_bit(q).index for q in inst.qubits]
        ideal.append(UnitaryGate(unitary), wires)
        actual = np.asarray(Operator(inst.operation).data)[:, :matrix.shape[1]]
        overlap = np.vdot(matrix, actual)
        phase = overlap/abs(overlap) if abs(overlap) else 1
        local.append(dict(index=index, wires=wires, shape=list(matrix.shape),
            input_isometry_error=float(np.linalg.norm(matrix.conj().T@matrix-np.eye(matrix.shape[1]))),
            synthesized_map_error=float(np.linalg.norm(actual-phase*matrix)),
            worst_column_fidelity=min(ref.fidelity(matrix[:, j],actual[:, j]) for j in range(matrix.shape[1]))))
    trials=[]
    for epsilon in (1e-8,1e-10,1e-12):
        for level in (0,1):
            trial: dict[str, Any] = dict(epsilon=epsilon,optimization_level=level)
            try:
                c=QuantumCircuit(logical.num_qubits);c.global_phase=logical.global_phase
                for inst in logical.data:
                    c.append(Isometry(inst.operation.params[0],0,0,epsilon=epsilon),[logical.find_bit(q).index for q in inst.qubits])
                compiled=transpile(c,basis_gates=['u','cx'],optimization_level=level,seed_transpiler=17)
                trial.update(status='success',cx_count=int(compiled.count_ops().get('cx',0)),fidelity_comp=ref.fidelity(rep.state,ref.circuit_state(compiled)))
            except Exception as exc:
                trial.update(status='error',error_type=type(exc).__name__,error=str(exc))
            trials.append(trial)
    result=dict(source_hash=source_hash(),saved_evaluation_key=saved['evaluation_key'],tree=saved['tree_serialization'],
        saved_fidelity_comp=saved['fidelity_comp'],representation_isometry_error=rep.isometry_error,
        ideal_map_fidelity=ref.fidelity(rep.state,Statevector.from_instruction(ideal).data),local_maps=local,trials=trials)
    write_json(output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        investigate('artifacts/stage_a_focused/compiler_anomaly.json')
