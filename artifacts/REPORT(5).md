# Revised Stage C1: 10-qubit scaling pilot

## Scope and conclusion

**C1 is complete for this bounded pilot; C2 and the 12-qubit experiment have not been run.** Two targets (hierarchical and overlapping, seed 13), one repetition per method, 128 candidates, the unchanged padded-size score, dense SVD construction, and the reference isometry compiler were used. The pool contains SCENT plus 127 unique random pair-merge trees. SCENT alone evaluates only its own tree. The required final fidelity is **0.999**, not 0.9999.

**Engineering feasibility passed at 10 qubits, but a screening advantage over SCENT was not demonstrated on these two targets.** Parameter screening selects SCENT first on both targets. Padded screening selects SCENT first on hierarchical (81 CNOTs); on overlapping it first returns 1171 CNOTs, then reaches the SCENT cost of 1163 at its second candidate. SCENT-initialized random search also retains those costs. Plain random finishes worse. Screening therefore spends extra preparation time without improving the available SCENT result in this pilot.

This differs from the fresh eight-qubit overlapping benchmark. It does not establish that screening stops working at ten qubits: there is only one target per family, and target identity changes along with size. The optimum over all 128 candidates is unknown. No conclusion about compiler independence follows from C1.

## Fixed candidate-evaluation budgets

These are separate runs from the fixed-time experiment. A 300-second safety ceiling applies, but all ten fixed-evaluation runs finished normally. SCENT uses one evaluation; other methods complete sixteen. Cells are best verified CNOT, with SCENT carried forward as a baseline.

| Family | Method | K=1 | K=2 | K=4 | K=8 | K=16 | First result (s) | Run wall (s) | Compiler calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hierarchical | padded | 81 | 81 | 81 | 81 | 81 | 1.645 | 6.808 | 16 |
| hierarchical | random_scent | 81 | 81 | 81 | 81 | 81 | 0.391 | 20.163 | 20 |
| hierarchical | scent | 81 | 81 | 81 | 81 | 81 | 0.178 | 0.179 | 1 |
| hierarchical | random | 1084 | 662 | 662 | 172 | 172 | 1.713 | 7.854 | 17 |
| hierarchical | parameters | 81 | 81 | 81 | 81 | 81 | 0.355 | 3.439 | 16 |
| overlapping | padded | 1171 | 1163 | 1163 | 1163 | 1163 | 2.627 | 22.091 | 16 |
| overlapping | random_scent | 1163 | 1163 | 1163 | 1163 | 1163 | 1.193 | 28.862 | 16 |
| overlapping | scent | 1163 | 1163 | 1163 | 1163 | 1163 | 0.364 | 0.364 | 1 |
| overlapping | random | 2697 | 1743 | 1743 | 1171 | 1171 | 0.940 | 25.900 | 17 |
| overlapping | parameters | 1163 | 1163 | 1163 | 1163 | 1163 | 2.309 | 23.237 | 16 |

## Independent fixed-time runs

Each method may continue until 30 seconds or pool exhaustion, without a sixteen-candidate cap. The parent terminates unfinished work at the deadline. Only completed, saved, verified prefixes within each deadline receive credit. A dash means no verified result yet. SCENT ends after one evaluation and retains that result.

| Family | Method | 1 s | 2 s | 4 s | 8 s | 16 s | 30 s | Completed candidates | Calls in completed candidates | First result (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hierarchical | padded | — | 81 | 81 | 81 | 81 | 81 | 43 | 50 | 1.282 |
| hierarchical | random_scent | 81 | 81 | 81 | 81 | 81 | 81 | 24 | 28 | 0.366 |
| hierarchical | scent | 81 | 81 | 81 | 81 | 81 | 81 | 1 | 1 | 0.244 |
| hierarchical | random | — | — | 662 | 662 | 172 | 172 | 23 | 26 | 2.556 |
| hierarchical | parameters | — | 81 | 81 | 81 | 81 | 81 | 33 | 36 | 1.180 |
| overlapping | padded | — | — | 1163 | 1163 | 1163 | 1163 | 24 | 24 | 2.328 |
| overlapping | random_scent | — | 1163 | 1163 | 1163 | 1163 | 1163 | 19 | 19 | 1.101 |
| overlapping | scent | — | 1163 | 1163 | 1163 | 1163 | 1163 | 1 | 1 | 1.123 |
| overlapping | random | — | — | 2697 | 1743 | 1533 | 1171 | 16 | 17 | 2.723 |
| overlapping | parameters | — | — | 1163 | 1163 | 1163 | 1163 | 25 | 25 | 2.374 |

Time runs completed 43 padded-screened hierarchical candidates and 24 padded-screened overlapping candidates, demonstrating that the old sixteen-evaluation cap is removed. Parameter screening completed 33 and 25 respectively. None improved the SCENT incumbent. The number completed is a throughput observation, not evidence of better selected circuits.

**Accounting limitation:** compiler-call totals above include retries in completed candidates. An interrupted in-flight candidate may have incurred additional synthesis calls that were not flushed to a final record. These are lower bounds on total attempted calls, not complete physical-call accounting for deadline-terminated runs. Partial artifacts are retained. The runner needs attempt-start logging before this is used as a formal fixed-time benchmark.

## Construction and memory

Whole-pool build time excludes representation checks and ranking; first-result and wall times include those costs. For terminated runs, the initial pool-build cost is recovered from the first prefix minus any incremental build cost of its selected candidate. Peak RSS is sampled from the worker process and expressed in MiB; it is not whole-machine memory or an exact allocation profile.

| Family | Mode | Method | Whole-pool build (s) | First result (s) | Sampled peak RSS (MiB) |
|---|---|---|---:|---:|---:|
| hierarchical | evaluation | padded | 1.294 | 1.645 | 366.2 |
| hierarchical | evaluation | random_scent | 0.000 | 0.391 | 407.0 |
| hierarchical | evaluation | scent | 0.000 | 0.178 | 299.9 |
| hierarchical | evaluation | random | 0.000 | 1.713 | 411.3 |
| hierarchical | evaluation | parameters | 0.237 | 0.355 | 368.1 |
| hierarchical | time | padded | 0.814 | 1.282 | 445.5 |
| hierarchical | time | random_scent | 0.000 | 0.366 | 435.1 |
| hierarchical | time | scent | 0.000 | 0.244 | 290.2 |
| hierarchical | time | random | 0.000 | 2.556 | 438.4 |
| hierarchical | time | parameters | 0.759 | 1.180 | 449.7 |
| overlapping | evaluation | padded | 0.886 | 2.627 | 413.8 |
| overlapping | evaluation | random_scent | 0.000 | 1.193 | 406.6 |
| overlapping | evaluation | scent | 0.000 | 0.364 | 316.4 |
| overlapping | evaluation | random | 0.000 | 0.940 | 435.5 |
| overlapping | evaluation | parameters | 0.772 | 2.309 | 412.7 |
| overlapping | time | padded | 0.783 | 2.328 | 441.0 |
| overlapping | time | random_scent | 0.000 | 1.101 | 434.0 |
| overlapping | time | scent | 0.000 | 1.123 | 314.6 |
| overlapping | time | random | 0.000 | 2.723 | 432.9 |
| overlapping | time | parameters | 0.763 | 2.374 | 445.5 |

Whole-pool construction is affordable in this pilot, but timings vary substantially between independent runs. In particular, hierarchical padded and parameter runs take different build times despite using the same construction work. With one repetition and the same randomized method order reused across groups, these differences should not be attributed to the score itself. Replication and stronger execution-order balancing are needed for reliable speed comparisons.

## Fidelity, retries and record checks

Audited 20 runs and 339 completed candidate evaluations. All prefix incumbents, tree/target identities, cumulative completed-call counts and fidelity constraints are consistent. Completed candidates contain 363 synthesis calls and 24 retry attempts. Failed completed evaluations: 1. Minimum final fidelity among successful evaluations: **0.999971733240**. These totals exclude unfinished work killed at a deadline.

Current core source matches frozen protocol: True. Runner/evaluator fingerprints match: True. Protocol and artifacts remain the authoritative record of execution; this report does not rerun circuits.

The one completed infeasible evaluation is hierarchical/time/parameters candidate 49. It did not replace the feasible incumbent. Deadline interruptions are counted separately and are not labeled fidelity failures.

## Next decision

Keep larger/mixed GNN pretraining on hold. The resource evidence makes a bounded 12-qubit feasibility probe reasonable, but does not justify a large sweep. Before claiming screening generalization, add multiple untouched 10-qubit instances with repeated timing measurements and paired comparisons to SCENT. Improve interrupted-attempt accounting and preserve progress atomically. C2 should then test a materially different synthesis method on a small subset; seed or optimization-level changes alone do not establish compiler independence.

No new 12-qubit training, compilation, or alternative-compiler experiment was launched while writing this report.

## Follow-up: exhaustive overlapping-pool diagnosis

The subsequent offline diagnostic completed all 128 candidates: 58 existing records
reused and 70 new evaluations. All succeeded under the fixed fidelity/compiler
policy. SCENT is the unique best candidate at 1163 CNOTs; no other tree improves
or ties it. Parameter count ranks SCENT first; padded size ranks it second,
following a 1171-CNOT tree. Thus this pool has zero improvement headroom over
SCENT: its lack of online improvement is not evidence that the selectors missed
a better member of the pool. Better trees outside the pool remain an open question.
See [the full diagnostic](../stage_c1_headroom/REPORT.md).
