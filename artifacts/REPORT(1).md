# Decision experiment before Stage C expansion

Stage C scaling and mixed random–SCENT pretraining remain on hold pending the bounded checks below. This report uses existing development evaluations; it is not an independent final benchmark. No pool-size expansion was performed.

## 1. Local mechanism and tolerance sensitivity

The selected overlapping_4 transition is a post-result case study. Each tolerance multiplier receives one direct construction/compilation, with no retries and unchanged global fidelity requirements.

| Local tolerance multiplier | Before CNOT | After CNOT | Difference | Before max rank | After max rank | Both feasible |
|---|---|---|---|---|---|---|
| 0.8 | 504 | 279 | 225 | 9 | 8 | True |
| 0.9 | 504 | 279 | 225 | 9 | 8 | True |
| 1.0 | 504 | 279 | 225 | 9 | 8 | True |
| 1.1 | 504 | 279 | 225 | 9 | 8 | True |
| 1.2 | 504 | 279 | 225 | 9 | 8 | True |

At nominal tolerance, the unchanged root and four two-qubit leaf-pair maps contribute the same 35 CNOTs. The two changing blocks account for the entire 225-CNOT difference:

| Block | Before leaves | After leaves | Rank before → after | Input bits before → after | Output bits before → after | Padded matrix before → after | Local CNOT before → after |
|---|---|---|---|---|---|---|---|
| 1 | [0, 1, 2, 3, 4, 5] | [0, 1, 2, 3, 4, 5] | 4 → 4 | 2 → 2 | 6 → 5 | 64×4 → 32×4 | 251 → 122 |
| 2 | [0, 1, 2, 3] | [0, 1, 4, 5] | 9 → 8 | 4 → 3 | 4 → 4 | 16×16 → 16×8 | 218 → 122 |

The six-leaf block drops 251→122 CNOT (129 fewer), and the changed four-leaf block drops 218→122 (96 fewer). Their total is 225. Rank 9 requires four padded bond qubits; rank 8 requires three. Local sums equal global counts in this case, but this need not hold in general. The partition and tensor entries also change, so this is not an isolated intervention on rank alone.

The observed transition is stable under ±20% local-tolerance changes, not proof of robustness to arbitrary precision policies. Exact singular values, discarded weights at ranks 8/9, local dimensions, all fidelities, and QPY/statevector artifacts are saved under cost/.

## 2. Target-dependent descriptor prediction

Each descriptor uses only the first nominal-tolerance TTN build, even when the saved circuit label required later retries. Historical accepted representation ranks, parameter counts, and fidelities were independently reconstructed to check numerical compatibility; those verification values are not predictor inputs. Descriptors are max/sum rank, parameter count, input/output register widths, and padded matrix sizes.

The fixed score sum_padded_entries is the sum of padded row-count × column-count over internal isometries. It uses no circuit labels. Parameter and max-rank scores are additional controls. rank_width_gp fits log-CNOT using log-transformed descriptors; raw_gp uses topology features. Both GP scalers fit only known observations in each split. All methods share held-out splits and random tie orders.

Primary cohorts are six complete Stage B pools (128 trees each) and ten new-target cohorts consisting of SCENT plus 96 global-random samples. All 2,824 historical records, including adaptive samples, were extracted and checked; adaptive neighborhood-union records are not used in primary prediction metrics. The new targets have already been inspected in the neighborhood experiment, so these remain development diagnostics.

| Cohort | Family | Known labels | Method | Mean Spearman | Top-five regret | Random top-five regret | Low-cost precision |
|---|---|---|---|---|---|---|---|
| stage_b | hierarchical | 8 | max_rank | 0.873 | 0.13 | 139.48 | 1.000 |
| stage_b | hierarchical | 8 | parameters | 0.981 | 0.00 | 139.48 | 1.000 |
| stage_b | hierarchical | 8 | sum_padded_entries | 0.990 | 0.00 | 139.48 | 1.000 |
| stage_b | hierarchical | 8 | rank_width_gp | 0.961 | 0.00 | 139.48 | 1.000 |
| stage_b | hierarchical | 8 | raw_gp | 0.343 | 0.13 | 139.48 | 1.000 |
| stage_b | hierarchical | 16 | max_rank | 0.872 | 0.40 | 131.33 | 1.000 |
| stage_b | hierarchical | 16 | parameters | 0.982 | 0.00 | 131.33 | 1.000 |
| stage_b | hierarchical | 16 | sum_padded_entries | 0.991 | 0.00 | 131.33 | 1.000 |
| stage_b | hierarchical | 16 | rank_width_gp | 0.983 | 0.00 | 131.33 | 1.000 |
| stage_b | hierarchical | 16 | raw_gp | 0.454 | 0.00 | 131.33 | 1.000 |
| stage_b | overlapping | 8 | max_rank | 0.760 | 2.67 | 59.58 | 0.600 |
| stage_b | overlapping | 8 | parameters | 0.834 | 63.53 | 59.58 | 0.427 |
| stage_b | overlapping | 8 | sum_padded_entries | 0.986 | 0.13 | 59.58 | 1.000 |
| stage_b | overlapping | 8 | rank_width_gp | 0.911 | 27.27 | 59.58 | 0.853 |
| stage_b | overlapping | 8 | raw_gp | 0.076 | 41.00 | 59.58 | 0.173 |
| stage_b | overlapping | 16 | max_rank | 0.754 | 10.53 | 54.94 | 0.560 |
| stage_b | overlapping | 16 | parameters | 0.827 | 65.87 | 54.94 | 0.440 |
| stage_b | overlapping | 16 | sum_padded_entries | 0.985 | 0.13 | 54.94 | 1.000 |
| stage_b | overlapping | 16 | rank_width_gp | 0.989 | 0.00 | 54.94 | 1.000 |
| stage_b | overlapping | 16 | raw_gp | 0.050 | 46.93 | 54.94 | 0.187 |
| new_random | hierarchical | 8 | max_rank | 0.870 | 0.00 | 163.71 | 1.000 |
| new_random | hierarchical | 8 | parameters | 0.967 | 0.00 | 163.71 | 1.000 |
| new_random | hierarchical | 8 | sum_padded_entries | 0.986 | 0.00 | 163.71 | 1.000 |
| new_random | hierarchical | 8 | rank_width_gp | 0.855 | 0.00 | 163.71 | 1.000 |
| new_random | hierarchical | 8 | raw_gp | 0.089 | 151.24 | 163.71 | 0.328 |
| new_random | hierarchical | 16 | max_rank | 0.878 | 0.00 | 158.08 | 1.000 |
| new_random | hierarchical | 16 | parameters | 0.968 | 0.00 | 158.08 | 1.000 |
| new_random | hierarchical | 16 | sum_padded_entries | 0.986 | 0.00 | 158.08 | 1.000 |
| new_random | hierarchical | 16 | rank_width_gp | 0.993 | 0.00 | 158.08 | 1.000 |
| new_random | hierarchical | 16 | raw_gp | 0.295 | 109.08 | 158.08 | 0.664 |
| new_random | overlapping | 8 | max_rank | 0.771 | 13.00 | 41.25 | 0.584 |
| new_random | overlapping | 8 | parameters | 0.883 | 4.84 | 41.25 | 0.664 |
| new_random | overlapping | 8 | sum_padded_entries | 0.972 | 0.40 | 41.25 | 0.992 |
| new_random | overlapping | 8 | rank_width_gp | 0.907 | 18.64 | 41.25 | 0.752 |
| new_random | overlapping | 8 | raw_gp | 0.038 | 29.64 | 41.25 | 0.224 |
| new_random | overlapping | 16 | max_rank | 0.743 | 10.92 | 43.67 | 0.536 |
| new_random | overlapping | 16 | parameters | 0.876 | 3.32 | 43.67 | 0.672 |
| new_random | overlapping | 16 | sum_padded_entries | 0.973 | 0.32 | 43.67 | 1.000 |
| new_random | overlapping | 16 | rank_width_gp | 0.993 | 0.00 | 43.67 | 1.000 |
| new_random | overlapping | 16 | raw_gp | 0.030 | 53.48 | 43.67 | 0.192 |

Padded-size performance by target with eight known labels (means over the five fixed splits):

| Cohort | Target | Spearman | Top-five regret | Random regret |
|---|---|---|---|---|
| stage_b | hierarchical_0 | 0.993 | 0.00 | 142.56 |
| stage_b | hierarchical_1 | 0.983 | 0.00 | 132.66 |
| stage_b | hierarchical_2 | 0.993 | 0.00 | 143.22 |
| stage_b | overlapping_0 | 0.984 | 0.40 | 38.88 |
| stage_b | overlapping_1 | 0.985 | 0.00 | 101.89 |
| stage_b | overlapping_2 | 0.988 | 0.00 | 37.95 |
| new_random | hierarchical_3 | 0.986 | 0.00 | 163.38 |
| new_random | hierarchical_4 | 0.986 | 0.00 | 163.31 |
| new_random | hierarchical_5 | 0.986 | 0.00 | 163.52 |
| new_random | hierarchical_6 | 0.985 | 0.00 | 165.25 |
| new_random | hierarchical_7 | 0.986 | 0.00 | 163.10 |
| new_random | overlapping_3 | 0.971 | 0.00 | 34.41 |
| new_random | overlapping_4 | 0.974 | 0.00 | 64.86 |
| new_random | overlapping_5 | 0.973 | 1.20 | 36.23 |
| new_random | overlapping_6 | 0.973 | 0.80 | 36.33 |
| new_random | overlapping_7 | 0.971 | 0.00 | 34.44 |

## Static screening with explicit construction cost

These are offline selection replays with SCENT as the sole initial candidate. Total compilation budgets include SCENT; remaining candidates follow a fixed descriptor ranking or random order, without feedback. Every descriptor-screening run requires constructing its entire candidate set before ranking. These runs differ from the eight-point GNN initialization and must be compared within this table.

| Cohort | Family | Method | Total 16 | Total 32 | Total 64 | Mean whole-pool construction seconds |
|---|---|---|---|---|---|---|
| stage_b | hierarchical | random | 75.00 | 75.00 | 75.00 | 0.0000 |
| stage_b | hierarchical | parameters | 75.00 | 75.00 | 75.00 | 0.3915 |
| stage_b | hierarchical | sum_padded_entries | 75.00 | 75.00 | 75.00 | 0.3915 |
| stage_b | overlapping | random | 368.22 | 360.11 | 342.67 | 0.0000 |
| stage_b | overlapping | parameters | 329.22 | 329.00 | 329.00 | 0.4090 |
| stage_b | overlapping | sum_padded_entries | 329.00 | 329.00 | 329.00 | 0.4090 |
| new_random | hierarchical | random | 75.00 | 75.00 | 75.00 | 0.0000 |
| new_random | hierarchical | parameters | 75.00 | 75.00 | 75.00 | 0.2738 |
| new_random | hierarchical | sum_padded_entries | 75.00 | 75.00 | 75.00 | 0.2738 |
| new_random | overlapping | random | 355.73 | 355.73 | 350.00 | 0.0000 |
| new_random | overlapping | parameters | 346.80 | 346.80 | 346.80 | 0.2825 |
| new_random | overlapping | sum_padded_entries | 346.80 | 346.80 | 346.80 | 0.2825 |

First-build descriptor extraction took 8.292 s for 2824 records (mean 2.936 ms each). Saved compilation attempts for these records total 1151.7 s. This timing comparison is indicative, not an end-to-end speedup measurement: historical timings, shared caching, cohort construction, and repeated builds differ. The diagnostic reconstruction checks and sensitivity sweep are not required by a future screening algorithm.

Random search has zero whole-pool prescreening cost in this table but still pays for construction and compilation of its selected candidates. Descriptor ranking is not a free topology calculation; eight-qubit dense construction costs cannot be extrapolated to larger systems. A strong ranking within these finite cohorts does not generate missing good trees or establish transfer to the SCENT approximate compiler. The dimension-based score is particularly aligned with this reference isometry-synthesis backend.

## Counterexamples and limits

The following includes every target for the preselected seed-0, eight-known-label split, including successes and failures of the padded-size score. The recommendation contains five trees. Equal dimensions can still yield different compiled costs, so ties are randomized rather than resolved by hidden labels.

| Cohort | Target | Best recommended CNOT | Held-out best CNOT | Regret | Recommended score | Held-out-best score |
|---|---|---|---|---|---|---|
| stage_b | hierarchical_0 | 76 | 76 | 0 | 132.0 | 132.0 |
| stage_b | hierarchical_1 | 76 | 76 | 0 | 132.0 | 132.0 |
| stage_b | hierarchical_2 | 76 | 76 | 0 | 132.0 | 132.0 |
| stage_b | overlapping_0 | 354 | 354 | 0 | 436.0 | 436.0 |
| stage_b | overlapping_1 | 277 | 277 | 0 | 336.0 | 336.0 |
| stage_b | overlapping_2 | 356 | 356 | 0 | 436.0 | 436.0 |
| new_random | hierarchical_3 | 81 | 81 | 0 | 132.0 | 132.0 |
| new_random | hierarchical_4 | 81 | 81 | 0 | 132.0 | 132.0 |
| new_random | hierarchical_5 | 81 | 81 | 0 | 132.0 | 132.0 |
| new_random | hierarchical_6 | 79 | 79 | 0 | 132.0 | 132.0 |
| new_random | hierarchical_7 | 81 | 81 | 0 | 132.0 | 132.0 |
| new_random | overlapping_3 | 356 | 356 | 0 | 436.0 | 436.0 |
| new_random | overlapping_4 | 314 | 314 | 0 | 384.0 | 384.0 |
| new_random | overlapping_5 | 354 | 354 | 0 | 436.0 | 436.0 |
| new_random | overlapping_6 | 356 | 354 | 2 | 436.0 | 436.0 |
| new_random | overlapping_7 | 356 | 356 | 0 | 436.0 | 436.0 |

Largest padded-size recommendation regrets across all evaluated splits:

| Cohort | Target | Known labels | Split seed | Regret |
|---|---|---|---|---|
| stage_b | overlapping_0 | 8 | 4 | 2 |
| stage_b | overlapping_0 | 16 | 1 | 2 |
| new_random | overlapping_5 | 8 | 1 | 2 |
| new_random | overlapping_5 | 8 | 3 | 2 |
| new_random | overlapping_5 | 8 | 4 | 2 |

## 3. Fixed-corpus GNN extension

The model, 2,000-tree training corpus, validation/test split, Adam settings, and seeds are unchanged. Each seed was rerun from initialization for 300 epochs because the old checkpoint did not include optimizer state. First-100-epoch histories must agree within 1e-9. Checkpoints are selected only by structural validation loss.

| Seed | Old test loss | Extended test loss | Selected epoch | First-100 maximum difference | Training seconds |
|---|---|---|---|---|---|
| 0 | 0.009351 | 0.003758 | 300 | 0 | 223.4 |
| 1 | 0.008552 | 0.003964 | 296 | 0 | 205.3 |
| 2 | 0.010276 | 0.004615 | 300 | 0 | 211.7 |

Search uses the same six original Stage B pools, eight common initial points, seeds, frozen-encoder scaling and BO implementation. Results are paired within target and search seed.

| Family | Model | Training epochs | Budget 16 | Budget 32 | Budget 64 |
|---|---|---|---|---|---|
| hierarchical | random | baseline | 75.00 | 75.00 | 75.00 |
| hierarchical | raw | baseline | 75.00 | 75.00 | 75.00 |
| hierarchical | pca | baseline | 75.00 | 75.00 | 75.00 |
| hierarchical | neighborhood | baseline | 75.00 | 75.00 | 75.00 |
| hierarchical | gnn_0 | 100 | 75.00 | 75.00 | 75.00 |
| hierarchical | gnn_0 | 300 | 75.00 | 75.00 | 75.00 |
| hierarchical | gnn_1 | 100 | 75.00 | 75.00 | 75.00 |
| hierarchical | gnn_1 | 300 | 75.00 | 75.00 | 75.00 |
| hierarchical | gnn_2 | 100 | 75.00 | 75.00 | 75.00 |
| hierarchical | gnn_2 | 300 | 75.00 | 75.00 | 75.00 |
| overlapping | random | baseline | 349.33 | 348.00 | 337.56 |
| overlapping | raw | baseline | 352.78 | 342.67 | 330.11 |
| overlapping | pca | baseline | 356.11 | 345.44 | 333.11 |
| overlapping | neighborhood | baseline | 340.33 | 336.78 | 329.22 |
| overlapping | gnn_0 | 100 | 355.89 | 346.11 | 345.00 |
| overlapping | gnn_0 | 300 | 346.22 | 342.44 | 331.44 |
| overlapping | gnn_1 | 100 | 356.11 | 350.44 | 332.56 |
| overlapping | gnn_1 | 300 | 353.78 | 346.33 | 331.22 |
| overlapping | gnn_2 | 100 | 355.44 | 349.33 | 348.00 |
| overlapping | gnn_2 | 300 | 357.78 | 349.56 | 343.11 |

| Family | Known labels | Model | Epochs | Spearman | Defined correlations | Top-five regret |
|---|---|---|---|---|---|---|
| hierarchical | 8 | gnn_0 | 100 | -0.107 | 15/15 | 201.20 |
| hierarchical | 8 | gnn_0 | 300 | -0.001 | 15/15 | 107.33 |
| hierarchical | 8 | gnn_1 | 100 | -0.095 | 15/15 | 130.47 |
| hierarchical | 8 | gnn_1 | 300 | -0.044 | 15/15 | 123.07 |
| hierarchical | 8 | gnn_2 | 100 | 0.037 | 15/15 | 23.07 |
| hierarchical | 8 | gnn_2 | 300 | -0.022 | 15/15 | 40.47 |
| hierarchical | 16 | gnn_0 | 100 | -0.057 | 15/15 | 113.47 |
| hierarchical | 16 | gnn_0 | 300 | 0.074 | 15/15 | 69.53 |
| hierarchical | 16 | gnn_1 | 100 | -0.093 | 15/15 | 112.80 |
| hierarchical | 16 | gnn_1 | 300 | -0.058 | 15/15 | 166.93 |
| hierarchical | 16 | gnn_2 | 100 | 0.064 | 15/15 | 0.67 |
| hierarchical | 16 | gnn_2 | 300 | -0.041 | 15/15 | 1.33 |
| overlapping | 8 | gnn_0 | 100 | 0.034 | 15/15 | 48.67 |
| overlapping | 8 | gnn_0 | 300 | 0.073 | 15/15 | 49.60 |
| overlapping | 8 | gnn_1 | 100 | -0.016 | 15/15 | 60.00 |
| overlapping | 8 | gnn_1 | 300 | 0.086 | 15/15 | 51.87 |
| overlapping | 8 | gnn_2 | 100 | 0.075 | 15/15 | 54.40 |
| overlapping | 8 | gnn_2 | 300 | 0.037 | 14/15 | 57.20 |
| overlapping | 16 | gnn_0 | 100 | -0.043 | 15/15 | 50.27 |
| overlapping | 16 | gnn_0 | 300 | -0.002 | 15/15 | 54.80 |
| overlapping | 16 | gnn_1 | 100 | -0.035 | 15/15 | 60.07 |
| overlapping | 16 | gnn_1 | 300 | 0.014 | 15/15 | 65.73 |
| overlapping | 16 | gnn_2 | 100 | 0.032 | 15/15 | 65.60 |
| overlapping | 16 | gnn_2 | 300 | -0.012 | 15/15 | 70.53 |

Undefined Spearman correlations from constant predictions are omitted from the correlation mean and counted explicitly; their recommendation regrets remain included. The longer runs include the original 100 epochs again; they are not claimed as optimizer-state resumes. Training seeds are reported separately, not selected after observing search quality. The same targets remain development data.


## Decision and next direction

The 9→8 transition is reproducible over the tested ±20% tolerance range, and two local blocks account exactly for its 225-CNOT reduction. Padded dimensions also predict costs across all eight overlapping target instances examined: with eight known labels, per-target mean Spearman correlations for the fixed score are 0.971–0.988. This extends the evidence beyond the isolated successful neighborhood case, within this reference compiler and these development distributions.

The fixed padded-size score gives mean top-five regret 0.13 on the original overlapping pools and 0.40 on the new random-sample overlapping cohorts (eight-known-label splits), compared with random recommendation regrets 59.58 and 41.25. Its worst observed top-five regret across tested splits is 2. A small GP on the same rank/width descriptors has higher recommendation regret than this fixed score; learned fitting has not established incremental value here.

Longer GNN training does help search in part. At budget 64, seed means improve 345.00→331.44, 332.56→331.22, and 348.00→343.11 CNOT. Therefore it would be incorrect to say that only reconstruction improved. Seeds 0 and 1 also improve mean search endpoints at budgets 16/32, while seed 2 worsens slightly there. Held-out cost-ranking correlations remain weak and retrieval is inconsistent; all three extended means still exceed the strongest original neighborhood baseline of 329.22 at budget 64. Two selected checkpoints are at epoch 300, so convergence is not established.

Keep Stage C corpus scaling and mixed pretraining on hold. Prioritize an independently validated screening implementation that constructs candidates once, ranks them by padded dimensions, and compiles only a shortlist. Reuse the constructed TTNs where feasible and measure full elapsed time with equal candidate-generation and precision rules. Keep the extended GNNs as secondary baselines rather than discarding their measured gains. Do not infer transfer to larger n or approximate compilation from these results. Pool-size expansion is not the immediate explanatory experiment.

## Artifact checks

Audited 2824 descriptor records, 800 prediction comparisons, 144 screening runs, 10 tolerance/tree variants, and 3456 GNN search steps. The prediction comparison count includes fixed-score rankings; only 320 of the 800 descriptor comparisons fit a GP. Original first-100-epoch training histories reproduced exactly for all three seeds.
