# Revised 3×-first evaluation policy

## Frozen policy

Attempt local tolerance multipliers **3, 1, 0.1, 0**, stopping at the first feasible result. Enforce final F≥0.999, global representation infidelity ≤0.0001 and compilation infidelity ≤0.0004. A representation-budget failure skips compilation and proceeds to the next factor. Compiler/numerical failures or failed verification also proceed to the next factor; exhausted attempts return failure, not an invented feasible cost. All builds, verification, synthesis calls and retries are charged. Synthesis-start records are persisted before compilation; JSON records use atomic replacement. No new hard wall-time or memory cap was introduced.

The 0.1 and 0 factors retain the original fallback tail. This fixed sequence is shared by all methods; it is not adaptively tuned per target. The policy was frozen after development checks and before any untouched-target circuit labels. Historical benchmark results are unchanged.

## Development cohort: paired direct 1× / 3× evaluations

Six target instances: eight-qubit hierarchical/overlapping seeds 0 and 3, and ten-qubit hierarchical/overlapping seed 13. Each uses SCENT, balanced, and a random tree drawn with seed 2026. All are development data. Both direct attempts are compiled for diagnostic error measurement; the production policy instead rejects representation-infeasible attempts before compilation.

| n | Family | Seed | Tree | Factor | Representation error | Compilation error | Final fidelity | CNOT | D2 | Feasible |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 8 | hierarchical | 0 | scent | 1 | 5.161146e-07 | 0 | 0.9999994839 | 75 | 31 | True |
| 8 | hierarchical | 0 | scent | 3 | 5.161146e-07 | 0 | 0.9999994839 | 75 | 31 | True |
| 8 | hierarchical | 0 | balanced | 1 | 5.161146e-07 | 0 | 0.9999994839 | 75 | 31 | True |
| 8 | hierarchical | 0 | balanced | 3 | 5.161146e-07 | 0 | 0.9999994839 | 75 | 31 | True |
| 8 | hierarchical | 0 | random | 1 | 3.330669e-16 | 0 | 1.0000000000 | 388 | 371 | True |
| 8 | hierarchical | 0 | random | 3 | 3.330669e-16 | 0 | 1.0000000000 | 388 | 371 | True |
| 8 | hierarchical | 3 | scent | 1 | 2.222541e-07 | 0 | 0.9999997777 | 75 | 31 | True |
| 8 | hierarchical | 3 | scent | 3 | 2.222541e-07 | 0 | 0.9999997777 | 75 | 31 | True |
| 8 | hierarchical | 3 | balanced | 1 | 2.222541e-07 | 1.110223e-16 | 0.9999997777 | 75 | 31 | True |
| 8 | hierarchical | 3 | balanced | 3 | 2.222541e-07 | 1.110223e-16 | 0.9999997777 | 75 | 31 | True |
| 8 | hierarchical | 3 | random | 1 | 2.220446e-16 | 1.110223e-16 | 1.0000000000 | 386 | 369 | True |
| 8 | hierarchical | 3 | random | 3 | 2.220446e-16 | 1.110223e-16 | 1.0000000000 | 386 | 369 | True |
| 8 | overlapping | 0 | scent | 1 | 1.341469e-05 | 4.440892e-16 | 0.9999865853 | 504 | 477 | True |
| 8 | overlapping | 0 | scent | 3 | 1.341469e-05 | 4.440892e-16 | 0.9999865853 | 504 | 477 | True |
| 8 | overlapping | 0 | balanced | 1 | 1.341469e-05 | 2.220446e-16 | 0.9999865853 | 707 | 466 | True |
| 8 | overlapping | 0 | balanced | 3 | 1.341469e-05 | 2.220446e-16 | 0.9999865853 | 707 | 466 | True |
| 8 | overlapping | 0 | random | 1 | 0 | 0 | 1.0000000000 | 386 | 369 | True |
| 8 | overlapping | 0 | random | 3 | 0 | 0 | 1.0000000000 | 386 | 369 | True |
| 8 | overlapping | 3 | scent | 1 | 1.236697e-05 | 0 | 0.9999876330 | 707 | 466 | True |
| 8 | overlapping | 3 | scent | 3 | 1.236697e-05 | 0 | 0.9999876330 | 707 | 466 | True |
| 8 | overlapping | 3 | balanced | 1 | 7.812176e-06 | 0 | 0.9999921878 | 707 | 466 | True |
| 8 | overlapping | 3 | balanced | 3 | 3.875433e-05 | 0 | 0.9999612457 | 707 | 466 | True |
| 8 | overlapping | 3 | random | 1 | 0 | 4.440892e-16 | 1.0000000000 | 388 | 371 | True |
| 8 | overlapping | 3 | random | 3 | 0 | 4.440892e-16 | 1.0000000000 | 388 | 371 | True |
| 10 | hierarchical | 13 | scent | 1 | 1.5537e-06 | 0 | 0.9999984463 | 81 | 34 | True |
| 10 | hierarchical | 13 | scent | 3 | 1.5537e-06 | 0 | 0.9999984463 | 81 | 34 | True |
| 10 | hierarchical | 13 | balanced | 1 | 1.178311e-06 | 0 | 0.9999988217 | 256 | 177 | True |
| 10 | hierarchical | 13 | balanced | 3 | 1.178311e-06 | 0 | 0.9999988217 | 256 | 177 | True |
| 10 | hierarchical | 13 | random | 1 | 1.445503e-05 | 0 | 0.9999855450 | 1417 | 1374 | True |
| 10 | hierarchical | 13 | random | 3 | 2.99119e-05 | 0.9672979 | 0.0327010870 | 1417 | 1374 | False |
| 10 | overlapping | 13 | scent | 1 | 9.158852e-06 | 2.220446e-16 | 0.9999908411 | 1163 | 900 | True |
| 10 | overlapping | 13 | scent | 3 | 3.65555e-05 | 7.771561e-16 | 0.9999634445 | 592 | 436 | True |
| 10 | overlapping | 13 | balanced | 1 | 1.109311e-05 | 0 | 0.9999889069 | 1445 | 829 | True |
| 10 | overlapping | 13 | balanced | 3 | 3.741203e-05 | 3.330669e-16 | 0.9999625880 | 1445 | 829 | True |
| 10 | overlapping | 13 | random | 1 | 7.112009e-06 | 0 | 0.9999928880 | 2459 | 2407 | True |
| 10 | overlapping | 13 | random | 3 | 2.324509e-05 | 4.440892e-16 | 0.9999767549 | 1419 | 1374 | True |

Of 36 direct evaluations, 35 satisfy all enforced requirements. Among the 17 pairs where both factors pass, 3× improves CNOT and depth in two pairs and ties in fifteen; none worsens. The benefit is therefore instance-dependent, not uniform. The 3× ten-qubit hierarchical random tree has representation error 2.99e-5 but compilation error approximately 0.9673 and final fidelity 0.03270. This is a compilation correctness failure, not evidence that the truncation error itself exceeded its budget. The exact failure mechanism is not established by this experiment. Explicit full-policy replay of that case rejects 3×, then succeeds at 1× with final fidelity 0.999985544966. This demonstrates the need for independent compilation verification and a fixed fallback rule.

## Exhaustive revised-policy pool

The original overlapping seed-13 pool was rebuilt and evaluated afresh, without reusing old cost labels. Descriptors use the initial 3× construction, even when a subsequent retry supplies the accepted circuit.

All 128 candidates are feasible. Five require fallback to 1×: 43, 51, 54, 62, 92. Completed synthesis calls: 132. Summed construction/evaluation/depth-extraction wall time: 236.51 s.

| Frontier candidate | CNOT | D2 | Final fidelity | Accepted factor |
|---|---:|---:|---:|---:|
| 0 | 592 | 436 | 0.9999634445 | 3.0 |

SCENT remains the sole nondominated candidate, now at **592 CNOTs and D2=436**. No candidate in this pool improves its CNOT count or depth under the revised policy. Both rebuilt fixed scores rank SCENT first with the original seed-0 tie order. The old minimum 1163/900 was policy-dependent; the new minimum is also not a state-preparation lower bound.

## Next stage

Resume only the authorized bounded random-pretraining data-size comparison: new ten-qubit models, nested 2000/10000-tree corpora, 1600 matched optimizer updates, three paired seeds, and untouched overlapping target seeds 14–18. This is size-specific training, not transfer of an eight-qubit checkpoint. Keep policy and scoring fixed throughout and report offline training separately from actual online search cost. Mixed pretraining remains outside this experiment.
