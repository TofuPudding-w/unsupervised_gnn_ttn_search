# Fresh end-to-end screening benchmark

Ten previously untouched eight-qubit instances: hierarchical and overlapping seeds 8–12. Each search method receives the same SCENT-plus-127-random-tree pool. Random trees use unique pair merging; no balanced/comb/neighbor prefix is inserted. SCENT alone constructs and evaluates only its tree. Three independently executed repetitions per method and target; method order is randomized before labels are observed.

Every method repetition independently generates candidates, constructs its required TTNs, synthesizes circuits, and verifies fidelity. There is no cross-method outcome or TTN cache. Parameter/padded screening constructs all 128 trees once, ranks without any circuit labels, and passes its own initial representations to compilation. Only required retries rebuild them.

The random_scent control starts with SCENT, then visits a random permutation. Plain random has no forced SCENT start. Frozen GNN BO uses extended model seeds 0/1/2 for repetitions 0/1/2, starts with SCENT, and uses the existing constrained BO after observations become available; it has no forced eight-circuit initialization. The GNN models and scaler training data were frozen before new target evaluation.

## Equal candidate-evaluation budget

K counts candidate opportunities, including failures; a candidate may require more than one compiler call. Cells show mean best feasible CNOT and completed-feasible runs / total runs. Means are conditional on a result being available, so compare availability before conditional CNOT at tight budgets.

| Family | Method | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|---|
| hierarchical | scent | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) |
| hierarchical | random | 375.93 (15/15) | 356.93 (15/15) | 264.47 (15/15) | 203.93 (15/15) | 143.47 (15/15) |
| hierarchical | random_scent | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.60 (15/15) | 75.60 (15/15) |
| hierarchical | parameters | 76.80 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) |
| hierarchical | padded | 76.80 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) |
| hierarchical | frozen_gnn_bo | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) |
| overlapping | scent | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) |
| overlapping | random | 549.60 (15/15) | 439.87 (15/15) | 383.73 (15/15) | 373.60 (15/15) | 364.60 (15/15) |
| overlapping | random_scent | 700.60 (15/15) | 483.00 (15/15) | 390.67 (15/15) | 370.60 (15/15) | 359.27 (15/15) |
| overlapping | parameters | 456.20 (15/15) | 435.27 (15/15) | 372.93 (15/15) | 343.00 (15/15) | 339.80 (15/15) |
| overlapping | padded | 352.00 (15/15) | 348.87 (15/15) | 345.93 (15/15) | 339.93 (15/15) | 339.80 (15/15) |
| overlapping | frozen_gnn_bo | 700.60 (15/15) | 493.20 (15/15) | 424.53 (15/15) | 377.93 (15/15) | 362.27 (15/15) |

## Equal actual compiler-call budget

Each attempted synthesis, including failed synthesis and precision retries, counts. A verified result is credited only when the candidate has completed within the call budget; a candidate spanning the boundary cannot contribute its later result.

| Family | Method | Calls=1 | Calls=2 | Calls=4 | Calls=8 | Calls=16 |
|---|---|---|---|---|---|---|
| hierarchical | scent | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) |
| hierarchical | random | 375.93 (15/15) | 356.93 (15/15) | 264.47 (15/15) | 203.93 (15/15) | 147.40 (15/15) |
| hierarchical | random_scent | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.60 (15/15) | 75.60 (15/15) |
| hierarchical | parameters | 76.80 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) |
| hierarchical | padded | 76.80 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) |
| hierarchical | frozen_gnn_bo | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) |
| overlapping | scent | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) |
| overlapping | random | 549.60 (15/15) | 439.87 (15/15) | 383.73 (15/15) | 373.60 (15/15) | 364.60 (15/15) |
| overlapping | random_scent | 700.60 (15/15) | 483.00 (15/15) | 390.67 (15/15) | 370.60 (15/15) | 359.27 (15/15) |
| overlapping | parameters | 456.20 (15/15) | 435.27 (15/15) | 372.93 (15/15) | 343.00 (15/15) | 339.80 (15/15) |
| overlapping | padded | 352.00 (15/15) | 348.87 (15/15) | 345.93 (15/15) | 339.93 (15/15) | 339.80 (15/15) |
| overlapping | frozen_gnn_bo | 700.60 (15/15) | 493.20 (15/15) | 424.53 (15/15) | 377.93 (15/15) | 362.27 (15/15) |

## Equal elapsed time

Wall time starts before SCENT affinity/clustering and pool generation. Screening must pay for whole-pool construction and ranking before its first result. Model loading/embedding, actual compilation, retries, verification, and artifact writing performed before the result are included. No future result or interpolated result is credited. All methods are capped at 16 candidate evaluations; finished runs retain their incumbent without spending remaining time on further candidates. SCENT alone ends after one evaluation.

| Family | Method | 0.5 s | 1 s | 2 s | 4 s | 8 s | 16 s |
|---|---|---|---|---|---|---|---|
| hierarchical | scent | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) |
| hierarchical | random | 312.43 (7/15) | 372.20 (15/15) | 267.33 (15/15) | 218.47 (15/15) | 143.47 (15/15) | 143.47 (15/15) |
| hierarchical | random_scent | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.60 (15/15) | 75.60 (15/15) | 75.60 (15/15) |
| hierarchical | parameters | — (0/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) |
| hierarchical | padded | — (0/15) | 75.21 (14/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) | 75.20 (15/15) |
| hierarchical | frozen_gnn_bo | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) | 75.80 (15/15) |
| overlapping | scent | — (0/15) | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) | 700.60 (15/15) |
| overlapping | random | — (0/15) | 548.60 (15/15) | 412.07 (15/15) | 374.00 (15/15) | 364.60 (15/15) | 364.60 (15/15) |
| overlapping | random_scent | — (0/15) | 699.62 (13/15) | 480.53 (15/15) | 375.13 (15/15) | 364.07 (15/15) | 359.27 (15/15) |
| overlapping | parameters | 539.00 (1/15) | 357.12 (8/15) | 428.00 (15/15) | 353.87 (15/15) | 339.80 (15/15) | 339.80 (15/15) |
| overlapping | padded | — (0/15) | 355.46 (13/15) | 347.33 (15/15) | 339.93 (15/15) | 339.80 (15/15) | 339.80 (15/15) |
| overlapping | frozen_gnn_bo | 449.67 (3/15) | 649.13 (15/15) | 495.67 (15/15) | 398.13 (15/15) | 369.60 (15/15) | 362.27 (15/15) |

## Per-target final endpoints and actual run time

Means over the three repetitions. SCENT uses one evaluation; search methods use sixteen. Repetitions are not independent target samples.

| Target | Method | Mean CNOT | Mean wall seconds | Mean compiler calls |
|---|---|---|---|---|
| hierarchical_8 | scent | 76.00 (3/3) | 0.261 | 1.00 |
| hierarchical_8 | random | 125.00 (3/3) | 7.665 | 17.33 |
| hierarchical_8 | random_scent | 76.00 (3/3) | 7.452 | 17.33 |
| hierarchical_8 | parameters | 76.00 (3/3) | 4.544 | 16.00 |
| hierarchical_8 | padded | 76.00 (3/3) | 4.139 | 16.00 |
| hierarchical_8 | frozen_gnn_bo | 76.00 (3/3) | 9.011 | 18.00 |
| hierarchical_9 | scent | 78.00 (3/3) | 0.190 | 1.00 |
| hierarchical_9 | random | 174.67 (3/3) | 8.164 | 16.33 |
| hierarchical_9 | random_scent | 77.00 (3/3) | 7.450 | 16.67 |
| hierarchical_9 | parameters | 75.00 (3/3) | 5.095 | 18.00 |
| hierarchical_9 | padded | 75.00 (3/3) | 5.192 | 18.00 |
| hierarchical_9 | frozen_gnn_bo | 78.00 (3/3) | 9.545 | 18.00 |
| hierarchical_10 | scent | 75.00 (3/3) | 0.147 | 1.00 |
| hierarchical_10 | random | 196.00 (3/3) | 8.375 | 16.00 |
| hierarchical_10 | random_scent | 75.00 (3/3) | 7.951 | 16.33 |
| hierarchical_10 | parameters | 75.00 (3/3) | 4.867 | 18.00 |
| hierarchical_10 | padded | 75.00 (3/3) | 4.867 | 18.00 |
| hierarchical_10 | frozen_gnn_bo | 75.00 (3/3) | 8.275 | 17.33 |
| hierarchical_11 | scent | 75.00 (3/3) | 0.152 | 1.00 |
| hierarchical_11 | random | 82.33 (3/3) | 7.129 | 16.00 |
| hierarchical_11 | random_scent | 75.00 (3/3) | 8.096 | 17.33 |
| hierarchical_11 | parameters | 75.00 (3/3) | 4.113 | 16.00 |
| hierarchical_11 | padded | 75.00 (3/3) | 4.075 | 16.00 |
| hierarchical_11 | frozen_gnn_bo | 75.00 (3/3) | 7.245 | 16.00 |
| hierarchical_12 | scent | 75.00 (3/3) | 0.158 | 1.00 |
| hierarchical_12 | random | 139.33 (3/3) | 7.059 | 16.00 |
| hierarchical_12 | random_scent | 75.00 (3/3) | 7.832 | 16.67 |
| hierarchical_12 | parameters | 75.00 (3/3) | 4.931 | 17.00 |
| hierarchical_12 | padded | 75.00 (3/3) | 5.221 | 17.00 |
| hierarchical_12 | frozen_gnn_bo | 75.00 (3/3) | 8.385 | 16.00 |
| overlapping_8 | scent | 707.00 (3/3) | 0.747 | 1.00 |
| overlapping_8 | random | 365.00 (3/3) | 9.344 | 16.00 |
| overlapping_8 | random_scent | 354.67 (3/3) | 10.011 | 16.00 |
| overlapping_8 | parameters | 354.00 (3/3) | 6.696 | 16.00 |
| overlapping_8 | padded | 354.00 (3/3) | 7.915 | 16.00 |
| overlapping_8 | frozen_gnn_bo | 356.00 (3/3) | 9.557 | 16.00 |
| overlapping_9 | scent | 707.00 (3/3) | 0.771 | 1.00 |
| overlapping_9 | random | 378.00 (3/3) | 11.078 | 16.00 |
| overlapping_9 | random_scent | 358.00 (3/3) | 9.609 | 16.00 |
| overlapping_9 | parameters | 356.00 (3/3) | 7.293 | 16.00 |
| overlapping_9 | padded | 356.00 (3/3) | 6.871 | 16.00 |
| overlapping_9 | frozen_gnn_bo | 378.67 (3/3) | 8.961 | 16.00 |
| overlapping_10 | scent | 675.00 (3/3) | 0.802 | 1.00 |
| overlapping_10 | random | 361.33 (3/3) | 10.579 | 16.00 |
| overlapping_10 | random_scent | 364.33 (3/3) | 10.700 | 16.33 |
| overlapping_10 | parameters | 279.00 (3/3) | 8.424 | 16.00 |
| overlapping_10 | padded | 279.00 (3/3) | 7.807 | 16.00 |
| overlapping_10 | frozen_gnn_bo | 339.33 (3/3) | 11.108 | 16.00 |
| overlapping_11 | scent | 707.00 (3/3) | 0.716 | 1.00 |
| overlapping_11 | random | 360.00 (3/3) | 10.482 | 16.00 |
| overlapping_11 | random_scent | 357.33 (3/3) | 10.400 | 16.00 |
| overlapping_11 | parameters | 356.00 (3/3) | 8.257 | 16.00 |
| overlapping_11 | padded | 356.00 (3/3) | 7.579 | 16.00 |
| overlapping_11 | frozen_gnn_bo | 362.00 (3/3) | 10.657 | 16.00 |
| overlapping_12 | scent | 707.00 (3/3) | 0.726 | 1.00 |
| overlapping_12 | random | 358.67 (3/3) | 8.959 | 16.00 |
| overlapping_12 | random_scent | 362.00 (3/3) | 8.294 | 16.00 |
| overlapping_12 | parameters | 354.00 (3/3) | 8.567 | 16.00 |
| overlapping_12 | padded | 354.00 (3/3) | 7.693 | 16.00 |
| overlapping_12 | frozen_gnn_bo | 375.33 (3/3) | 6.574 | 16.00 |

## Timing components

Seconds per complete run, averaged by family/method. Construction includes screened-but-uncompiled trees. Verification separates representation checks and circuit-state checks. Retry time overlaps construction/compilation/verification and must not be added again. File serialization and miscellaneous allocation/logging overhead are included in wall time; phase timers need not sum exactly to it.

| Family | Method | Generation | Construction | Ranking/proposals | Model setup | Compile | Representation checks | Circuit checks | Retry (overlap) | Wall |
|---|---|---|---|---|---|---|---|---|---|---|
| hierarchical | scent | 0.0336 | 0.0025 | 0.0000 | 0.0000 | 0.1100 | 0.0001 | 0.0135 | 0.0000 | 0.1816 |
| hierarchical | random | 0.0860 | 0.0516 | 0.0000 | 0.0000 | 6.4080 | 0.0010 | 0.9736 | 0.1119 | 7.6783 |
| hierarchical | random_scent | 0.0902 | 0.0531 | 0.0001 | 0.0000 | 6.4724 | 0.0010 | 0.9758 | 0.3330 | 7.7562 |
| hierarchical | parameters | 0.0889 | 0.3446 | 0.0065 | 0.0000 | 3.6098 | 0.0055 | 0.4884 | 0.3083 | 4.7100 |
| hierarchical | padded | 0.0882 | 0.3257 | 0.0060 | 0.0000 | 3.6162 | 0.0054 | 0.4892 | 0.3138 | 4.6989 |
| hierarchical | frozen_gnn_bo | 0.0845 | 0.0515 | 0.4778 | 0.0423 | 6.6226 | 0.0010 | 1.0181 | 0.4342 | 8.4923 |
| overlapping | scent | 0.0322 | 0.0032 | 0.0000 | 0.0000 | 0.6050 | 0.0001 | 0.1017 | 0.0000 | 0.7526 |
| overlapping | random | 0.0887 | 0.0540 | 0.0001 | 0.0000 | 8.3601 | 0.0010 | 1.3248 | 0.0000 | 10.0883 |
| overlapping | random_scent | 0.0926 | 0.0528 | 0.0001 | 0.0000 | 8.1456 | 0.0010 | 1.2661 | 0.0480 | 9.8027 |
| overlapping | parameters | 0.0847 | 0.3350 | 0.0067 | 0.0000 | 6.2524 | 0.0055 | 0.9658 | 0.0000 | 7.8474 |
| overlapping | padded | 0.0825 | 0.4176 | 0.0071 | 0.0000 | 5.8958 | 0.0060 | 0.8994 | 0.0000 | 7.5729 |
| overlapping | frozen_gnn_bo | 0.0755 | 0.0460 | 0.4329 | 0.0386 | 7.4246 | 0.0009 | 1.1885 | 0.0000 | 9.3713 |

## Scope and accounting

Completed 180 method runs, 2430 candidate evaluations, and 2495 actual synthesis calls. Summed method wall time: 1184.3 s. This invocation wall time: 1191.7 s.

Outside per-method clocks: scaler export from old training trees 2.022 s; shared warmup on an old development target 2.165 s; target-state creation 0.009 s. Imports and one uncompiled reference-pool consistency generation per target are also outside per-method clocks; each actual run generates its own pool inside its clock. Prior GNN training is an offline cost, not charged again as online time.

The optimum over all 128 trees is unknown: only selected candidates were compiled. Measurements apply to this machine, eight-qubit dense construction, existing target families, and the reference isometry compiler. They do not establish transfer to larger systems or the SCENT approximate compiler. Shared libraries/OS caches may remain warm despite independent circuit evaluation; randomized method order and repeated fresh runs mitigate ordering effects but do not remove all timing noise. Stage C larger/mixed pretraining remains on hold.

## Interpretation

Padded-size screening improves early selection on the untouched overlapping targets: mean CNOT is 352.00 at K=1 and 345.93 at K=4, versus 456.20 and 372.93 for parameter count. At K=1 it wins on three targets and ties on two; at K=4 it wins on two and ties on three. Both screeners reach the same per-target endpoints by K=16 (mean 339.80). The evidence supports earlier useful selection, not a better final shortlist endpoint than parameter count.

At two seconds every overlapping run has a result. Padded screening averages 347.33 CNOT, parameter count 428.00, random 412.07, SCENT-initialized random 480.53, and frozen GNN BO 495.67. Whole-pool construction is included. Padded screening beats parameter count on three targets and ties on two at this deadline. At one second completion fractions differ, so conditional CNOT alone is not a fair comparison.

On hierarchical targets SCENT already averages 75.8 CNOT; screeners reach 75.2 by K=2. SCENT finishes within 0.5 seconds in all hierarchical runs, whereas neither screener has a result then. Screening is not uniformly preferable when the starting tree is already inexpensive.

Keep larger/mixed pretraining on hold. The next scientific question is transfer to a modestly larger system or materially different compiler; the current results establish neither. The five targets per family, not the fifteen repetitions, determine independent target coverage.

## Independent artifact audit

Status: passed. Checked 180 runs and 2430 candidate evaluations; 65 retry attempts; minimum verified final fidelity 0.999892416027.

A concurrent documentation edit changed the on-disk reference.py hash after execution began. The exact original core bytes were recovered and matched the frozen combined hash; an AST comparison confirmed that current executable code is unchanged after removing docstrings. The running process retained its imported code. See source_snapshot/, source_review.json and audit.json. A protocol resume still intentionally rejects changed source bytes.
