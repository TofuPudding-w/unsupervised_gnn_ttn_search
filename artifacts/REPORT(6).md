# C1 logical two-qubit depth analysis

## Definition and scope

Read the accepted saved QPY circuits; no recompilation, routing, optimization or new search. Primary analysis covers the exhaustively evaluated 128-candidate overlapping pool at ten qubits, seed 13. All accepted circuits satisfy the original fidelity requirement 0.999.

D2 is the ASAP depth of the ordered two-qubit-gate subsequence: each CX is assigned one plus the maximum layer already used on its two wires. Disjoint CX gates can share a layer. One-qubit gates have zero duration for this metric. The result was independently checked against Qiskit filtered depth for every loaded circuit. Total logical depth including u gates is also recorded. This all-to-all structural metric is not hardware-scheduled execution time; no coupling-map routing or gate durations are included.

## Exhaustive pool results

- SCENT: **1163 CNOTs, D2=900, total logical depth=1789**.
- Pool D2 range: **900–2677**.
- Candidates with smaller D2 than SCENT: **0**.
- Candidates with larger CNOT count but smaller D2: **0**.
- Nondominated candidates in (CNOT, D2): **1**.

| Frontier candidate | CNOT | D2 | Total logical depth | Fidelity |
|---|---:|---:|---:|---:|
| 0 | 1163 | 900 | 1789 | 0.9999908411 |

## Exploratory depth caps

These caps are derived from SCENT depth (75%, 90%, one layer below, equal, and 110%) to describe the existing pool. They are development analyses, not prespecified independent tests. Feasibility here requires both the original fidelity constraint and D2 ≤ cap.

| D2 cap | Feasible candidates | Lowest feasible CNOT | SCENT eligible |
|---|---:|---:|---|
| 675 | 0 | none | False |
| 810 | 0 | none | False |
| 899 | 0 | none | False |
| 900 | 1 | 1163 | True |
| 990 | 1 | 1163 | True |

## Retrospective method comparison

Original overlapping search selections are held fixed; only their incumbent feasibility is recomputed. This is not a depth-aware search rerun. Fixed-evaluation and 30-second runs are separate experiments. Deadline-interrupted work remains excluded. No ten-qubit GNN was evaluated.

| Mode | Method | Cap | Best at K=1 | K=4 | K=16 | Best at run endpoint |
|---|---|---:|---:|---:|---:|
| evaluation | padded | 899 | none | none | none | none |
| evaluation | padded | 900 | none | 1163 | 1163 | 1163 |
| evaluation | random_scent | 899 | none | none | none | none |
| evaluation | random_scent | 900 | 1163 | 1163 | 1163 | 1163 |
| evaluation | scent | 899 | none | none | none | none |
| evaluation | scent | 900 | 1163 | 1163 | 1163 | 1163 |
| evaluation | random | 899 | none | none | none | none |
| evaluation | random | 900 | none | none | none | none |
| evaluation | parameters | 899 | none | none | none | none |
| evaluation | parameters | 900 | 1163 | 1163 | 1163 | 1163 |
| time | padded | 899 | none | none | none | none |
| time | padded | 900 | none | 1163 | 1163 | 1163 |
| time | random_scent | 899 | none | none | none | none |
| time | random_scent | 900 | 1163 | 1163 | 1163 | 1163 |
| time | scent | 899 | none | none | none | none |
| time | scent | 900 | 1163 | 1163 | 1163 | 1163 |
| time | random | 899 | none | none | none | none |
| time | random | 900 | none | none | none | none |
| time | parameters | 899 | none | none | none | none |
| time | parameters | 900 | 1163 | 1163 | 1163 | 1163 |

## Interpretation and next experiment

SCENT minimizes both CNOT count and two-qubit depth in this fixed pool. A depth cap below its D2 leaves no feasible candidate; a cap at or above it retains SCENT as the lowest-CNOT choice. Therefore adding a depth constraint does not create improvement headroom here. This conclusion does not extend to unseen pools, target instances, other compilers, or routed hardware circuits.

Yes, compare methods under a depth constraint in a broader benchmark, alongside the original unconstrained resource comparison. Keep CNOT count and D2 separate, report infeasibility explicitly, and choose caps from development data or hardware requirements before evaluating untouched targets. A depth-aware BO must model or otherwise incorporate the depth constraint; filtering its previous trace does not establish its prospective constrained performance.
