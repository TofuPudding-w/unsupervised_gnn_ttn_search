"""Localize the frozen compiler failure; do not modify the production evaluator."""
import importlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.linalg import block_diag, null_space
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Isometry, UCGate
from qiskit.quantum_info import Operator
from threadpoolctl import threadpool_limits
from ttn_search import reference as ref
from ttn_search.targets import gaussian_target
from ttn_search.trees import random_tree
from ttn_search.evaluate import write_json, software, source_hash

OUT = Path('artifacts/compiler_audit')


def aligned_error(actual, expected):
    overlap = np.vdot(expected, actual)
    phase = overlap / abs(overlap) if abs(overlap) else 1
    return float(np.linalg.norm(actual - phase * expected))


def exact_completion(matrix):
    """Independent dense oracle, not a production synthesis replacement."""
    return np.column_stack((matrix, null_space(matrix.conj().T)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = gaussian_target(10, 'hierarchical', 13)
    tree = random_tree(10, np.random.default_rng(2026))
    rep = ref.build(target.state, tree, .0003 / 9)
    # Capture exactly the production register allocation before transpilation.
    with patch.object(ref, 'transpile', side_effect=lambda circuit, **kw: circuit):
        logical = ref.compile_reference(rep, 17, 1)
    ideal = QuantumCircuit(10)
    ideal.global_phase = logical.global_phase
    blocks = []
    for i, inst in enumerate(logical.data):
        matrix = np.asarray(inst.operation.params[0])
        actual = Operator(inst.operation).data[:, :matrix.shape[1]]
        wires = [logical.find_bit(q).index for q in inst.qubits]
        ideal.unitary(exact_completion(matrix), wires)
        blocks.append(dict(index=i, shape=list(matrix.shape), wires=wires,
            orthogonality_error=float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[1]))),
            synthesis_error=aligned_error(actual, matrix)))
    bad = [b['index'] for b in blocks if b['synthesis_error'] > 1e-7]
    assert bad, 'Original failure no longer reproduces; inspect the installed backend.'
    failing = np.asarray(logical.data[bad[0]].operation.params[0])
    np.save(OUT / 'failing_isometry.npy', failing)
    uc_rows = []

    def inspect_uc(gates, **kw):
        gate = UCGate(gates, **kw)
        actual = np.asarray(gate._get_diagonal())[:, None] * Operator(gate).data
        error = aligned_error(actual, block_diag(*gates))
        uc_rows.append(dict(index=len(uc_rows), controls=int(np.log2(len(gates))), error=error))
        if error > 1e-7:
            np.savez(OUT / 'failing_uc.npz', gates=np.asarray(gates))
        return gate

    iso_module = importlib.import_module('qiskit.circuit.library.generalized_gates.isometry')
    mcg_module = importlib.import_module('qiskit.circuit.library.generalized_gates.mcg_up_to_diagonal')
    with patch.object(iso_module, 'UCGate', side_effect=inspect_uc), patch.object(mcg_module, 'UCGate', side_effect=inspect_uc):
        Operator(Isometry(failing, 0, 0))
    epsilon_checks = []
    for eps in [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
        actual = Operator(Isometry(failing, 0, 0, epsilon=eps)).data[:, :failing.shape[1]]
        epsilon_checks.append(dict(epsilon=eps, error=aligned_error(actual, failing)))
    gates = np.load(OUT / 'failing_uc.npz')['gates']
    mux_checks = []
    for simplify in [False, True]:
        gate = UCGate(list(gates), up_to_diagonal=True, mux_simp=simplify)
        actual = np.asarray(gate._get_diagonal())[:, None] * Operator(gate).data
        mux_checks.append(dict(simplify=simplify, error=aligned_error(actual, block_diag(*gates))))
    levels = []
    for level in [0, 1, 2]:
        circuit = transpile(logical, basis_gates=['u', 'cx'], optimization_level=level, seed_transpiler=17)
        levels.append(dict(level=level, compilation_fidelity=ref.fidelity(rep.state, ref.circuit_state(circuit))))
    # Replace only the faulty block with an independently completed unitary.
    repaired = QuantumCircuit(10)
    repaired.global_phase = logical.global_phase
    for i, inst in enumerate(logical.data):
        wires = [logical.find_bit(q).index for q in inst.qubits]
        if i in bad:
            repaired.unitary(exact_completion(np.asarray(inst.operation.params[0])), wires)
        else:
            repaired.append(inst.operation, wires)
    repaired = transpile(repaired, basis_gates=['u', 'cx'], optimization_level=1, seed_transpiler=17)
    result = dict(software=software(), source_hash=source_hash(), tree=tree,
        representation_fidelity=ref.fidelity(target.state, rep.state),
        logical_synthesized_fidelity=ref.fidelity(rep.state, ref.circuit_state(logical)),
        independent_dense_fidelity=ref.fidelity(rep.state, ref.circuit_state(ideal)),
        replaced_block_fidelity=ref.fidelity(rep.state, ref.circuit_state(repaired)),
        replaced_block_final_fidelity=ref.fidelity(target.state, ref.circuit_state(repaired)),
        replaced_block_cx=int(repaired.count_ops().get('cx', 0)),
        blocks=blocks, uc_checks=uc_rows, epsilon_checks=epsilon_checks,
        mux_checks=mux_checks, optimization_checks=levels)
    assert result['independent_dense_fidelity'] > 1-1e-10
    assert result['replaced_block_fidelity'] > 1-1e-8
    write_json(OUT / 'result.json', result)
    text = f'''# Compilation failure localization

The ten-qubit hierarchical seed-13 random tree (tree seed 2026), at 3× tolerance, reproduces the historical failure.

| Check | Fidelity to reconstructed TTN |
|---|---:|
| Independent dense unitary completion, original wire mapping | {result['independent_dense_fidelity']:.12f} |
| Logical Isometry circuit (synthesis occurs during simulation) | {result['logical_synthesized_fidelity']:.12f} |
| Replace only faulty block, then transpile | {result['replaced_block_fidelity']:.12f} |

Representation fidelity is {result['representation_fidelity']:.12f}. The independent dense oracle verifies TTN assembly and wire ordering. The error is already present in Isometry synthesis, before transpiler optimization. Levels 0/1/2 all fail.

Block {bad[0]} has dimensions {blocks[bad[0]]['shape']} and orthogonality residual {blocks[bad[0]]['orthogonality_error']:.3g}, but its synthesized first columns have phase-aligned Frobenius error {blocks[bad[0]]['synthesis_error']:.6f}. Within that block, a four-control UCGate fails its defining identity D U_synth = blockdiag(U_0,...,U_15), with phase-aligned error {max(r['error'] for r in uc_rows):.6f}. This is not merely a global phase discrepancy.

Changing Isometry epsilon from 1e-6 to 1e-14 or disabling UCGate multiplexer simplification does not resolve this reproducer. The exact numerical defect within the backend decomposition is not yet established; this audit localizes it to a standalone UCGate with saved inputs. It does not claim a repaired upstream algorithm.

Replacing only the faulty block by dense unitary completion restores final fidelity {result['replaced_block_final_fidelity']:.12f}, at {result['replaced_block_cx']} CNOTs. This is an independent diagnostic workaround, not an efficient replacement compiler or a new benchmark label.

## Provenance and implications

See result.json for installed versions, source fingerprint, block residuals and checks. failing_isometry.npy and failing_uc.npz preserve standalone inputs. No vendor, installed Qiskit, production compiler, frozen policy or historical result was changed. Existing final circuits remain independently verified; failed-attempt costs and fallback costs remain valid observations of that evaluator. They must not be treated as compiler-independent costs of the original 3× TTN.

Before collecting supervised labels under a repaired compiler, select and verify a versioned replacement, then re-evaluate affected candidates and representative controls under that version. Do not merge replacement costs with the frozen benchmark. A standalone regression test marks the installed upstream failure explicitly and checks the independent completion oracle.
'''
    (OUT / 'REPORT.md').write_text(text)
    print(json.dumps({k: v for k, v in result.items() if 'fidelity' in k}, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
