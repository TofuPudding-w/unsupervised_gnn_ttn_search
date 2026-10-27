# Ten-qubit GNN data-size comparison under the revised policy

## Design

Five previously untouched overlapping targets, seeds 14–18; 128 shared candidates per target, comprising SCENT and unique random pair-merge trees; three paired repetitions. All methods use the frozen strict-budget evaluator (3→1→0.1→0 local tolerance attempts). No policy or score is tuned on these target outcomes. SCENT alone uses one evaluation; other methods use up to sixteen.

These are newly trained ten-qubit GNNs, not transferred eight-qubit checkpoints. Nested random training corpora contain 2000 or 10000 trees, with common disjoint 500-tree validation/test sets. Benchmark pool topologies are excluded from all three corpus splits. Architecture, Adam .001, batch size 128, and 1600 updates are fixed; validation is checked every 100 updates. Three training seeds are retained, each paired with its corresponding search repetition. Scalers use the same 2000-tree subset for both corpus sizes, with separate frozen embeddings. No mixed SCENT pretraining is used.

Equal updates isolate additional data coverage, not convergence: sampling with replacement yields approximately 102 training-set passes for 2000 trees and 20 for 10000. Each checkpoint is selected by structural validation loss only. Training-seed and search-seed effects are paired, not independently factorial.

All online method runs generate and compile independently without outcome caches. Screeners pay for all 128 first-attempt TTNs and reuse them for selected candidates; GNNs pay for model loading, embedding and proposals. SCENT and GNN BO start with SCENT; plain random and fixed screeners do not receive a free SCENT circuit. Cyclic orders balance execution positions across fifteen target/repetition groups to within one occurrence.

## Reconstruction and training cost

| Training trees | Seed | Best validation loss | Test loss | Selected update | Training/setup seconds |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0 | 0.010262 | 0.010207 | 1600 | 78.59 |
| 2000 | 1 | 0.010529 | 0.010502 | 1600 | 74.49 |
| 2000 | 2 | 0.008979 | 0.008758 | 1600 | 90.03 |
| 10000 | 0 | 0.009266 | 0.009105 | 1600 | 87.06 |
| 10000 | 1 | 0.010794 | 0.010535 | 1600 | 80.97 |
| 10000 | 2 | 0.009130 | 0.009069 | 1600 | 79.64 |

Training/setup time includes training, validation/test evaluation, model saving and scaler embedding/export. Initial corpus generation, tensor preparation and common input/reference-pool setup are not fully timed. Thus reported training plus online time is not a complete from-zero amortization measurement.

## Mean best verified CNOT

Means over five targets and three repetitions; independent target count is five, not fifteen. Failed evaluations consume budget.

| Method | K=1 | K=2 | K=4 | K=8 | K=16 | Mean final incumbent D2 | Mean complete online time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| scent | 663.80 | 663.80 | 663.80 | 663.80 | 663.80 | 508.20 | 0.849 |
| random | 1894.53 | 1589.80 | 1417.40 | 1291.67 | 1243.27 | 1125.13 | 30.116 |
| random_scent | 663.80 | 663.80 | 663.80 | 663.80 | 653.80 | 504.67 | 30.436 |
| parameters | 663.80 | 663.80 | 633.80 | 633.80 | 633.80 | 497.60 | 22.469 |
| padded | 633.80 | 633.80 | 633.80 | 633.80 | 633.80 | 497.60 | 20.563 |
| gnn_2000 | 663.80 | 663.80 | 663.80 | 663.80 | 663.80 | 508.20 | 32.256 |
| gnn_10000 | 663.80 | 663.80 | 663.80 | 663.80 | 663.80 | 508.20 | 32.016 |

Depth is that of the minimum-CNOT incumbent selected by the search, not the minimum-depth candidate or a depth-constrained BO result. Cost ties retain the first incumbent. These are fixed-evaluation comparisons with actual elapsed costs; no uncapped fixed-time claim is made.

## Per-target final quality and improvement over SCENT

Each entry is averaged over three repetitions. Positive gain means fewer CNOTs than SCENT. All targets are retained, including those with no observed improvement. Pool optima are unknown because these new pools are not exhaustively evaluated.

| Target | Method | Mean best CNOT | Gain over SCENT | Mean incumbent D2 | Mean online time (s) |
|---|---|---:|---:|---:|---:|
| overlapping_14 | scent | 592.00 | 0.00 | 436.00 | 0.715 |
| overlapping_14 | random | 1336.67 | -744.67 | 1188.00 | 31.405 |
| overlapping_14 | random_scent | 592.00 | 0.00 | 436.00 | 30.168 |
| overlapping_14 | parameters | 592.00 | 0.00 | 436.00 | 19.877 |
| overlapping_14 | padded | 592.00 | 0.00 | 436.00 | 21.913 |
| overlapping_14 | gnn_2000 | 592.00 | 0.00 | 436.00 | 31.317 |
| overlapping_14 | gnn_10000 | 592.00 | 0.00 | 436.00 | 31.387 |
| overlapping_15 | scent | 955.00 | 0.00 | 795.00 | 1.093 |
| overlapping_15 | random | 1180.33 | -225.33 | 1118.67 | 32.758 |
| overlapping_15 | random_scent | 905.00 | 50.00 | 777.33 | 30.185 |
| overlapping_15 | parameters | 805.00 | 150.00 | 742.00 | 25.192 |
| overlapping_15 | padded | 805.00 | 150.00 | 742.00 | 18.443 |
| overlapping_15 | gnn_2000 | 955.00 | 0.00 | 795.00 | 37.450 |
| overlapping_15 | gnn_10000 | 955.00 | 0.00 | 795.00 | 36.406 |
| overlapping_16 | scent | 588.00 | 0.00 | 435.00 | 0.672 |
| overlapping_16 | random | 1195.00 | -607.00 | 1083.00 | 32.843 |
| overlapping_16 | random_scent | 588.00 | 0.00 | 435.00 | 32.430 |
| overlapping_16 | parameters | 588.00 | 0.00 | 435.00 | 23.288 |
| overlapping_16 | padded | 588.00 | 0.00 | 435.00 | 22.437 |
| overlapping_16 | gnn_2000 | 588.00 | 0.00 | 435.00 | 35.901 |
| overlapping_16 | gnn_10000 | 588.00 | 0.00 | 435.00 | 38.881 |
| overlapping_17 | scent | 592.00 | 0.00 | 436.00 | 1.049 |
| overlapping_17 | random | 1315.00 | -723.00 | 1179.67 | 26.972 |
| overlapping_17 | random_scent | 592.00 | 0.00 | 436.00 | 31.752 |
| overlapping_17 | parameters | 592.00 | 0.00 | 436.00 | 23.506 |
| overlapping_17 | padded | 592.00 | 0.00 | 436.00 | 20.019 |
| overlapping_17 | gnn_2000 | 592.00 | 0.00 | 436.00 | 27.358 |
| overlapping_17 | gnn_10000 | 592.00 | 0.00 | 436.00 | 26.593 |
| overlapping_18 | scent | 592.00 | 0.00 | 439.00 | 0.714 |
| overlapping_18 | random | 1189.33 | -597.33 | 1056.33 | 26.604 |
| overlapping_18 | random_scent | 592.00 | 0.00 | 439.00 | 27.642 |
| overlapping_18 | parameters | 592.00 | 0.00 | 439.00 | 20.484 |
| overlapping_18 | padded | 592.00 | 0.00 | 439.00 | 20.005 |
| overlapping_18 | gnn_2000 | 592.00 | 0.00 | 439.00 | 29.256 |
| overlapping_18 | gnn_10000 | 592.00 | 0.00 | 439.00 | 26.811 |

## Observed improvement opportunities

Best across the candidates actually evaluated by any method, not an exhaustive pool optimum. An observed improvement demonstrates headroom; absence of one does not certify its absence from the pool. This table is descriptive and is not used to remove targets from the aggregate.

| Target | SCENT CNOT | Best observed CNOT | Observed gain | Unique evaluated candidates |
|---|---:|---:|---:|---:|
| overlapping_14 | 592 | 592 | 0 | 99 |
| overlapping_15 | 955 | 805 | 150 | 95 |
| overlapping_16 | 588 | 588 | 0 | 99 |
| overlapping_17 | 592 | 592 | 0 | 92 |
| overlapping_18 | 592 | 592 | 0 | 104 |

## Timing at evaluation checkpoints

Mean actual elapsed seconds until each candidate checkpoint completes, including generation, construction, retries and verification. SCENT carries its single result forward.

| Method | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---:|---:|---:|---:|---:|
| scent | 0.848 | 0.848 | 0.848 | 0.848 | 0.848 |
| random | 2.094 | 3.950 | 7.610 | 15.124 | 30.114 |
| random_scent | 0.873 | 2.894 | 6.595 | 14.261 | 30.433 |
| parameters | 1.856 | 3.009 | 5.369 | 10.910 | 22.467 |
| padded | 1.743 | 2.872 | 5.051 | 9.902 | 20.562 |
| gnn_2000 | 0.903 | 3.027 | 7.172 | 15.483 | 32.254 |
| gnn_10000 | 0.936 | 2.906 | 6.879 | 14.775 | 32.013 |

## Paired data-size comparison

| Budget | Expanded wins | Ties | Expanded losses | Mean CNOT change (10000 − 2000) |
|---:|---:|---:|---:|---:|
| 1 | 0 | 15 | 0 | 0.00 |
| 2 | 0 | 15 | 0 | 0.00 |
| 4 | 0 | 15 | 0 | 0.00 |
| 8 | 0 | 15 | 0 | 0.00 |
| 16 | 0 | 15 | 0 | 0.00 |

## Interpretation

Increasing the random corpus from 2000 to 10000 trees produced no improvement in best verified CNOT at any checkpoint in any of the fifteen paired runs. Both GNN settings retain the SCENT incumbent on all five targets. Their complete online runs cost about 32 seconds on average versus about 0.85 seconds for SCENT alone, before charging offline training. This bounded experiment establishes no added selection value for the larger corpus. It does not prove that GNN selection is impossible or that longer training cannot help.

Reconstruction improves for seed 0, is nearly unchanged for seed 1, and worsens for seed 2. Every selected checkpoint is at update 1600, so convergence is not established. The matched-update experiment tests coverage at a fixed optimization budget; it does not compare fully converged models.

Target 15 demonstrates useful selection headroom: SCENT gives 955 CNOTs and depth 795; both screeners consistently find 805 CNOTs and depth 742. Padded screening does so at its first candidate; parameter screening reaches it by K=4. SCENT-initialized random succeeds in one of three repetitions; neither GNN does. On four other targets, no tested method improves SCENT. These pools are not exhaustive, so no-improvement cases are not certified zero-headroom cases.

The current evidence supports the shared verified tolerance policy and simple screening as strong baselines. It does not justify automatically increasing structural pretraining further or adding mixed pretraining. Any next learning experiment should change an explicitly diagnosed bottleneck—such as target-dependent information or the training objective—and retain the fixed-score baselines. No such follow-up is launched by this report.


## Audit and limitations

Audited 105 runs, 1455 candidate evaluations, 1485 synthesis starts, and 32 retry attempts. Failed completed evaluations: 0. Minimum successful final fidelity: 0.999904003767. Frozen source/model fingerprints, corpus separation, identical target/pool identity, policy compliance, no-cache flags, unique selections, incumbent prefixes and attempt counts pass checks. Final CNOT counts and two-qubit depths were independently recomputed from all 105 incumbent QPY circuits.

This is a bounded matched-update test of random structural pretraining, not evidence about every GNN architecture or target-dependent learning objective. Five targets, one size, one compiler, and a sixteen-query horizon limit generalization. No larger system or mixed pretraining was added.
