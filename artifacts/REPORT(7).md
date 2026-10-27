# Ten-qubit accuracy-budget and compilation diagnostic

## Scope

Development target: overlapping seed 13. Frozen pool candidates: 0 (SCENT), 74 (original padded-score first choice), and 1 (fixed random-pool example). This is a small mechanism experiment, not held-out validation. The existing benchmark and its precision policy are unchanged.

Each representation uses local discarded-weight tolerance (0.0001/9) × multiplier. Reference multipliers are 1, 3, 10, 30, 100; the alternative is tested at 1 and 10. No precision retries occur. Final fidelity ≥0.999 determines diagnostic acceptance; compliance with the original representation/compilation sub-budgets is reported separately. A 120-second ceiling applies to each tree/compiler/tolerance combination.

## Main result

**The same SCENT tree becomes substantially shallower by changing truncation tolerance.** At multiplier 3, it falls from 1163 to 592 CNOTs and from D2=900 to D2=436, with final fidelity 0.99996344. This setting also meets the original global representation/compilation sub-budgets. The nominal local tolerance was conservative for this target; no tree change is needed for this reduction.

The earlier exhaustive optimum of 1163 CNOTs and minimum depth 900 were specific to the original evaluation policy. They are not state-preparation lower bounds. The present sweep does not establish the optimum of all 128 trees under a revised tolerance policy.

## Reference tolerance sweep

| Candidate | Multiplier | Final fidelity | CNOT | D2 | Max rank | Padded entries | Final feasible | Original budgets feasible |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 1 | 0.9999908411 | 1163 | 900 | 9 | 1232 | True | True |
| 0 | 3 | 0.9999634445 | 592 | 436 | 8 | 656 | True | True |
| 0 | 10 | 0.9997197831 | 590 | 436 | 6 | 656 | True | False |
| 0 | 30 | 0.9991206356 | 592 | 436 | 5 | 656 | True | False |
| 0 | 100 | 0.9979023441 | 388 | 235 | 5 | 464 | False | False |
| 74 | 1 | 0.9999891635 | 1171 | 1094 | 10 | 1232 | True | True |
| 74 | 3 | 0.9999432166 | 1171 | 1094 | 9 | 1232 | True | True |
| 74 | 10 | 0.9998597025 | 629 | 562 | 8 | 720 | True | False |
| 74 | 30 | 0.9992098027 | 629 | 562 | 6 | 720 | True | False |
| 74 | 100 | 0.9983330761 | 473 | 407 | 5 | 560 | False | False |
| 1 | 1 | 0.9999909816 | 2198 | 2152 | 18 | 2244 | True | True |
| 1 | 3 | 0.9999781581 | 1196 | 1158 | 16 | 1220 | True | True |
| 1 | 10 | 0.9999099896 | 1196 | 1158 | 13 | 1220 | True | True |
| 1 | 30 | 0.9996984065 | 1196 | 1158 | 11 | 1220 | True | False |
| 1 | 100 | 0.9985183071 | 1196 | 1158 | 9 | 1220 | False | False |

Factor 100 fails final fidelity for all three trees, so its gate savings are not admissible. Factors 10 and 30 can pass the final requirement while violating the original conservative representation allocation. They are a proposed change in evaluation policy, not improvements under an identical policy. Gate counts need not vary monotonically with tolerance because tensor entries and synthesis simplification also change.

## Where the SCENT depth originates

Internal blocks are independently compiled and assembled using the same register allocation and physical-output mapping. A longest CX-dependency chain is traced through that assembly. Contributions below belong to one such chain; ties can yield other chains. The assembly and globally compiled reference both have D2=900 at factor 1 and 436 at factor 3 in this case. Block attribution is not generally an exact partition of the globally optimized circuit.

| Factor | Block leaves | Rank | Child ranks | Input/output bits | Local CNOT | Local D2 | CX on assembly critical chain |
|---|---|---:|---|---|---:|---:|---:|
| 1 | [0, 1, 2, 3, 8, 9] | 9 | [4, 8] | 4/5 | 546 | 535 | 535 |
| 1 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 1 | [9, 9] | 0/8 | 247 | 247 | 240 |
| 1 | [0, 1, 8, 9] | 8 | [4, 4] | 3/4 | 122 | 119 | 119 |
| 1 | [8, 9] | 4 | [2, 2] | 2/2 | 6 | 6 | 6 |
| 1 | [2, 3] | 4 | [2, 2] | 2/2 | 6 | 6 | 0 |
| 1 | [0, 1] | 4 | [2, 2] | 2/2 | 6 | 6 | 0 |
| 1 | [4, 5, 6, 7] | 9 | [4, 4] | 4/4 | 218 | 213 | 0 |
| 1 | [4, 5] | 4 | [2, 2] | 2/2 | 6 | 6 | 0 |
| 1 | [6, 7] | 4 | [2, 2] | 2/2 | 6 | 6 | 0 |
| 3 | [0, 1, 2, 3, 8, 9] | 8 | [4, 7] | 3/5 | 261 | 257 | 257 |
| 3 | [0, 1, 8, 9] | 7 | [4, 4] | 3/4 | 122 | 119 | 119 |
| 3 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 1 | [8, 8] | 0/6 | 57 | 57 | 54 |
| 3 | [8, 9] | 4 | [2, 2] | 2/2 | 6 | 6 | 6 |
| 3 | [2, 3] | 4 | [2, 2] | 2/2 | 6 | 6 | 0 |
| 3 | [0, 1] | 4 | [2, 2] | 2/2 | 6 | 6 | 0 |
| 3 | [4, 5, 6, 7] | 8 | [4, 4] | 3/4 | 122 | 119 | 0 |
| 3 | [4, 5] | 4 | [2, 2] | 2/2 | 6 | 6 | 0 |
| 3 | [6, 7] | 4 | [2, 2] | 2/2 | 6 | 6 | 0 |

The six-leaf block contributes 535→257 chain layers; the root contributes 240→54. These reductions sum to 464, accounting for the observed 900→436 change in assembly depth. The root child ranks cross 9→8, reducing its output register from eight to six qubits. The six-leaf block input register drops from four to three qubits. Large local synthesis blocks on a shared-wire dependency chain dominate this example.

At nominal tolerance the sum of independent block depths is 1144, but the assembled circuit depth is 900; at factor 3 the corresponding values are 582 and 436. Thus the depth calculation already permits parallelism between disjoint operations. Merely scheduling the same fixed gate-dependency graph cannot reduce its ASAP critical-path depth. Commutation-aware transformations or a different gate decomposition could change that graph, but were not tested here.

## Alternative compiler: full-unitary completion plus QSD

Each padded isometry is completed with a null-space basis to a square unitary; Qiskit Quantum Shannon decomposition synthesizes that unitary. The same wire allocation, all-to-all connectivity, u/cx gate set, optimization level 1 and seed 17 are used. This changes the synthesis method, rather than only changing a transpiler setting. It is exact synthesis of a completed unitary, not an approximate SCENT compiler.

| Candidate | Multiplier | Reference CNOT / D2 | QSD CNOT / D2 | QSD final fidelity |
|---|---:|---|---|---:|
| 0 | 1 | 1163 / 900 | 30283 / 29735 | 0.9999908411 |
| 0 | 10 | 590 / 436 | 2411 / 2258 | 0.9997197831 |
| 74 | 1 | 1171 / 1094 | 7886 / 7768 | 0.9999891635 |
| 74 | 10 | 629 / 562 | 2350 / 2296 | 0.9998597025 |
| 1 | 1 | 2198 / 2152 | 8198 / 8091 | 0.9999909816 |
| 1 | 10 | 1196 / 1158 | 2662 / 2619 | 0.9999099896 |

Full-unitary completion is substantially worse in these comparisons. It specifies action on input subspaces that the state-preparation task never uses; its overhead is therefore unsurprising. This rules out this particular completion/synthesis route as an improvement here, not other isometry-specific or approximate compilers.

## Diagnostic timing and validation

| Candidate | Backend | Multiplier | Build (s) | Compilation plus block analysis (s) | Verification (s) |
|---|---|---:|---:|---:|---:|
| 0 | reference | 1 | 0.0046 | 0.5723 | 0.0917 |
| 0 | reference | 3 | 0.0038 | 0.4232 | 0.0488 |
| 0 | reference | 10 | 0.0034 | 0.3458 | 0.0473 |
| 0 | reference | 30 | 0.0028 | 0.3555 | 0.0522 |
| 0 | reference | 100 | 0.0032 | 0.2584 | 0.0346 |
| 0 | unitary_qsd | 1 | 0.0051 | 1.3004 | 12.4162 |
| 0 | unitary_qsd | 10 | 0.0036 | 0.1274 | 0.2418 |
| 74 | reference | 1 | 0.0053 | 0.5502 | 0.0875 |
| 74 | reference | 3 | 0.0027 | 0.5414 | 0.0938 |
| 74 | reference | 10 | 0.0028 | 0.3335 | 0.0489 |
| 74 | reference | 30 | 0.0026 | 0.3516 | 0.0966 |
| 74 | reference | 100 | 0.0083 | 0.4235 | 0.0384 |
| 74 | unitary_qsd | 1 | 0.0026 | 0.3958 | 0.8334 |
| 74 | unitary_qsd | 10 | 0.0027 | 0.1257 | 0.2557 |
| 1 | reference | 1 | 0.0034 | 1.0782 | 0.1787 |
| 1 | reference | 3 | 0.0027 | 0.6006 | 0.0948 |
| 1 | reference | 10 | 0.0031 | 0.6252 | 0.1064 |
| 1 | reference | 30 | 0.0028 | 2.3719 | 0.3096 |
| 1 | reference | 100 | 0.0065 | 2.1898 | 0.4355 |
| 1 | unitary_qsd | 1 | 0.0088 | 1.0962 | 3.2230 |
| 1 | unitary_qsd | 10 | 0.0073 | 0.4039 | 1.0371 |

The compilation timer includes independently synthesizing local blocks for analysis and then compiling the final circuit. For the reference it therefore includes duplicated synthesis work. These are diagnostic overhead measurements, not fair production compiler-latency comparisons. Timings also include process/machine variability; no speedup claim is based on them.

Completed 21 configurations; 21 returned circuits. Statevectors verify the final and representation fidelities. Block assemblies reproduce their representation with fidelity above 0.999999. Four unit tests cover both compilers on complex states and two tree shapes, physical-label mapping, and critical-chain depth accounting.

## Decision

Prioritize validation of a revised tolerance policy on development targets, then freeze it before untouched-target evaluation. Factor 3 is a promising development setting, not a universal prescription; all candidate trees must receive the same predeclared policy and final verification. Rebuild screening descriptors under that policy and recheck ranking/headroom. Do not silently replace historical results or compare a relaxed-policy selector against a stricter SCENT baseline.

A depth-constrained benchmark is now more meaningful because this target has an observed feasible circuit at D2=436. No application depth cap has been chosen; 436 is not a lower bound or a guarantee on other targets. Retain fidelity–CNOT–depth curves before choosing externally motivated or development-frozen caps. Expanded GNN training remains on hold.
